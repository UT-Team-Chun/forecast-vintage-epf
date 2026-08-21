"""Real JEPX spot-market CSV downloader and parser.

JEPX (Japan Electric Power Exchange) publishes a yearly spot-market CSV at:

    https://www.jepx.jp/market/excel/spot_{YYYY}.csv

where ``YYYY`` is the **Japanese fiscal year** (April through March of the
following calendar year). The file is encoded in **Shift-JIS (cp932)** and
its column layout grew over time (intraday market columns appeared in
fiscal 2021), so this loader parses by header name, never by position.

The loader respects JEPX's terms of use by:

- Downloading at most once per (year, force=False) call.
- Caching to ``EpfConfig.raw_data_dir / "jepx" / "spot_{year}.csv"`` so
  repeat experiments reuse the same bytes.
- Updating a ``DatasetRegistry`` entry with SHA-256 / source URL / fetch
  timestamp, enabling per-run traceability.
- Never committing the raw CSV to git (``data/`` is gitignored).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alg.core.epf.data.registry import (
    DatasetRegistry,
    hash_file,
    make_entry,
)
from alg.models.epf.area import AreaCode

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

JEPX_SPOT_URL_TEMPLATE = "https://www.jepx.jp/market/excel/spot_{year}.csv"
JEPX_ENCODING = "cp932"
USER_AGENT = "forecast-vintage-epf/0.1 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"

JST = timezone(timedelta(hours=9), name="JST")
SLOTS_PER_DAY = 48
SLOT_MINUTES = 30

_AREA_PRICE_RE = re.compile(r"^エリアプライス(.+?)(?:\([^)]*\))?$")
_SYSTEM_PRICE_HEADERS = (
    "システムプライス(円/kWh)",
    "システムプライス(¥/kWh)",
)
_VOLUME_HEADERS = ("約定総量(kWh)",)
_DATE_COLUMN = "年月日"
_SLOT_COLUMN = "時刻コード"

_JP_NAME_TO_AREA: dict[str, AreaCode] = {
    area.japanese_name: area for area in AreaCode.jepx_traded_areas()
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TransportError)),
    reraise=True,
)
def _download(url: str, client: httpx.Client) -> bytes:
    response = client.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content


def fetch_jepx_spot_year(
    year: int,
    *,
    cache_dir: Path,
    registry: DatasetRegistry | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> Path:
    """Download (with caching) one fiscal-year JEPX spot CSV.

    Args:
        year: JEPX fiscal year (e.g. ``2024`` covers 2024-04 to 2025-03).
        cache_dir: Root for the local raw-data tree.
        registry: If supplied, the registry is updated with the fetched
            file's SHA-256 + source URL + timestamp.
        force: Re-download even if a cached file already exists.
        client: Optional ``httpx.Client`` (handy for tests/mocks).

    Returns:
        Path to the local cached CSV file.

    """
    target_dir = (cache_dir / "jepx").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"spot_{year}.csv"
    source_url = JEPX_SPOT_URL_TEMPLATE.format(year=year)

    if target_path.exists() and not force:
        logger.info("JEPX cache hit for year=%s: %s", year, target_path)
    else:
        logger.info("Fetching JEPX year=%s from %s", year, source_url)
        owned_client = client is None
        http_client = client or httpx.Client(timeout=60.0, follow_redirects=True)
        try:
            payload = _download(source_url, http_client)
        finally:
            if owned_client:  # pragma: no cover - default-client cleanup, exercised only in prod
                http_client.close()
        target_path.write_bytes(payload)

    if registry is not None:
        entry = make_entry(
            name=f"jepx_spot_{year}",
            raw_data_dir=cache_dir,
            file_path=target_path,
            source_url=source_url,
        )
        # If we have a cached file and the SHA matches, this is effectively a no-op.
        existing = registry.find(entry.name)
        if existing is None or existing.sha256 != entry.sha256:
            registry.upsert(entry)
    return target_path


def _read_sjis_csv(path: Path) -> pl.DataFrame:
    text = path.read_bytes().decode(JEPX_ENCODING)
    return pl.read_csv(text.encode("utf-8"), infer_schema_length=10_000)


def _extract_area_columns(df: pl.DataFrame, path: Path) -> dict[str, AreaCode]:
    """Return a ``{csv_column: AreaCode}`` map for area-price columns."""
    area_columns: dict[str, AreaCode] = {}
    for col in df.columns:
        match = _AREA_PRICE_RE.match(col)
        if match is None:
            continue
        area = _JP_NAME_TO_AREA.get(match.group(1))
        if area is not None:
            area_columns[col] = area
    if not area_columns:
        msg = f"JEPX CSV has no recognisable area-price columns: {path}"
        raise ValueError(msg)
    return area_columns


def _build_slot_timestamps(dates: list[str], slots: list[int], path: Path) -> list[datetime]:
    """Convert ``(date, slot)`` pairs into tz-aware JST datetimes at slot start."""
    out: list[datetime] = []
    for date_str, slot in zip(dates, slots, strict=True):
        if not 1 <= slot <= SLOTS_PER_DAY:
            msg = f"Slot code out of range 1..48 in {path}: {slot}"
            raise ValueError(msg)
        base = datetime.strptime(date_str, "%Y/%m/%d").replace(tzinfo=JST)
        out.append(base + timedelta(minutes=(slot - 1) * SLOT_MINUTES))
    return out


def _to_long_format(
    df: pl.DataFrame,
    *,
    area_columns: dict[str, AreaCode],
    system_col: str | None,
    volume_col: str | None,
) -> pl.DataFrame:
    """Unpivot area-price columns to long format and normalise auxiliary columns."""
    index_cols = ["datetime_jst"]
    if system_col is not None:
        index_cols.append(system_col)
    if volume_col is not None:
        index_cols.append(volume_col)

    long_df = df.unpivot(
        on=list(area_columns.keys()),
        index=index_cols,
        variable_name="_area_col",
        value_name="price_yen_kwh",
    )
    long_df = long_df.with_columns(
        area=pl.col("_area_col").replace_strict(
            {col: code.value for col, code in area_columns.items()},
            default=None,
        ),
    ).drop("_area_col")

    if system_col is not None:
        long_df = long_df.rename({system_col: "system_price_yen_kwh"})
    else:
        long_df = long_df.with_columns(system_price_yen_kwh=pl.lit(None, dtype=pl.Float64))

    if volume_col is not None:
        long_df = long_df.with_columns(volume_mwh=pl.col(volume_col) / 1000.0).drop(volume_col)
    else:
        long_df = long_df.with_columns(volume_mwh=pl.lit(None, dtype=pl.Float64))

    return long_df.select(
        "datetime_jst",
        "area",
        "price_yen_kwh",
        "system_price_yen_kwh",
        "volume_mwh",
    ).sort(["datetime_jst", "area"])


def parse_jepx_spot_csv(path: Path) -> pl.DataFrame:
    """Parse a single JEPX yearly CSV into a long-format polars DataFrame.

    Returns columns:

    - ``datetime_jst`` : tz-aware Datetime (Asia/Tokyo, start of slot)
    - ``area``         : AreaCode value (Utf8)
    - ``price_yen_kwh``: Float64
    - ``system_price_yen_kwh``: Float64 (nullable)
    - ``volume_mwh``   : Float64 (nullable)
    """
    df = _read_sjis_csv(path)
    for required in (_DATE_COLUMN, _SLOT_COLUMN):
        if required not in df.columns:
            msg = f"JEPX CSV is missing required column '{required}': {path}"
            raise ValueError(msg)

    area_columns = _extract_area_columns(df, path)
    system_col = next((c for c in _SYSTEM_PRICE_HEADERS if c in df.columns), None)
    volume_col = next((c for c in _VOLUME_HEADERS if c in df.columns), None)

    timestamps = _build_slot_timestamps(
        df[_DATE_COLUMN].cast(pl.Utf8).to_list(),
        df[_SLOT_COLUMN].cast(pl.Int32).to_list(),
        path,
    )

    keep_cols: dict[str, pl.Expr] = {col: pl.col(col).cast(pl.Float64) for col in area_columns}
    if system_col is not None:
        keep_cols[system_col] = pl.col(system_col).cast(pl.Float64)
    if volume_col is not None:
        keep_cols[volume_col] = pl.col(volume_col).cast(pl.Float64)
    df = df.with_columns(**keep_cols).with_columns(
        datetime_jst=pl.Series(
            timestamps,
            dtype=pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
        ),
    )
    return _to_long_format(
        df,
        area_columns=area_columns,
        system_col=system_col,
        volume_col=volume_col,
    )


def load_jepx_spot(  # noqa: PLR0913 - intentional rich keyword-only API
    years: Iterable[int],
    *,
    cache_dir: Path,
    areas: Iterable[AreaCode] | None = None,
    registry: DatasetRegistry | None = None,
    download: bool = True,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """High-level loader: fetches (or reuses cached) yearly CSVs and concatenates.

    Args:
        years: Fiscal years to load (each becomes one CSV).
        cache_dir: Local raw-data root.
        areas: Optional filter; defaults to all JEPX-traded areas.
        registry: Optional ``DatasetRegistry`` for traceability.
        download: If False, only cached files are used; missing files raise.
        client: Optional ``httpx.Client`` (for tests).

    Returns:
        Long-format polars DataFrame sorted by (datetime_jst, area).

    """
    requested_areas = tuple(areas) if areas is not None else AreaCode.jepx_traded_areas()
    for area in requested_areas:
        if not area.is_jepx_traded():
            msg = f"Area {area.value!r} is not traded on JEPX (no spot price data)"
            raise ValueError(msg)
    area_values = {a.value for a in requested_areas}

    frames: list[pl.DataFrame] = []
    for year in years:
        target = cache_dir / "jepx" / f"spot_{year}.csv"
        if not target.exists():
            if not download:
                msg = f"JEPX year {year} not in cache and download=False: {target}"
                raise FileNotFoundError(msg)
            fetch_jepx_spot_year(
                year,
                cache_dir=cache_dir,
                registry=registry,
                client=client,
            )
        elif registry is not None and registry.find(f"jepx_spot_{year}") is None:
            # Cached file exists but registry is missing an entry — backfill.
            entry = make_entry(
                name=f"jepx_spot_{year}",
                raw_data_dir=cache_dir,
                file_path=target,
                source_url=JEPX_SPOT_URL_TEMPLATE.format(year=year),
            )
            registry.upsert(entry)
        frames.append(parse_jepx_spot_csv(target))

    df = pl.concat(frames) if frames else _empty_long_df()
    return df.filter(pl.col("area").is_in(list(area_values))).sort(["datetime_jst", "area"])


def _empty_long_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
            "area": pl.Utf8,
            "price_yen_kwh": pl.Float64,
            "system_price_yen_kwh": pl.Float64,
            "volume_mwh": pl.Float64,
        },
    )


def verify_cache(path: Path, expected_sha256: str) -> bool:
    """Return True when the local file matches ``expected_sha256``."""
    return path.exists() and hash_file(path) == expected_sha256
