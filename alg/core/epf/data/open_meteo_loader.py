"""Open-Meteo archive (ERA5-based) weather loader.

Open-Meteo's free archive API returns hourly ERA5 reanalysis values
without authentication. We use it to fetch four upstream physical
signals that the JEPX literature points at as drivers of FLOOR / SPIKE:

- ``cloud_cover``: total cloud cover (%) — strongly anti-correlated
  with photovoltaic generation, hence with FLOOR events.
- ``shortwave_radiation``: GHI (W/m²) — the direct driver of solar PV.
- ``temperature_2m``: 2-metre temperature (°C) — drives heating /
  cooling demand and therefore evening peaks.
- ``wind_speed_10m``: 10-metre wind (km/h) — drives wind generation
  output (less material for Kyushu than other areas).

For Kyushu we centre the query on Fukuoka (33.59 N, 130.40 E) by
default; the latitude / longitude are configurable for future areas.

Endpoint pattern (verified 2026-05-12):

    https://archive-api.open-meteo.com/v1/archive?
        latitude=33.59&longitude=130.40
        &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
        &hourly=temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m
        &timezone=Asia/Tokyo

Responses are pretty JSON. We cache one file per (location, year)
under ``data/raw/open_meteo/`` and register every cached payload with
``DatasetRegistry`` so the per-run hashes log to MLflow tags.
"""

from __future__ import annotations

import json
import logging
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

from alg.core.epf.data.registry import DatasetRegistry, make_entry

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_USER_AGENT = "forecast-vintage-epf/0.1 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"
JST = timezone(timedelta(hours=9), name="JST")

# Variables we request. The order of columns in the returned DataFrame
# matches this tuple so callers can rely on stable schema.
WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
)

# Default coordinates for Kyushu (Fukuoka). Future area-specific loaders
# can override.
DEFAULT_KYUSHU_LAT = 33.59
DEFAULT_KYUSHU_LON = 130.40

# Stable column names in the canonical output DataFrame.
WEATHER_OUTPUT_COLUMNS: tuple[str, ...] = (
    "datetime_jst",
    "temperature_c",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "wind_speed_kmh",
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TransportError)),
    reraise=True,
)
def _download(url: str, params: dict[str, str], client: httpx.Client) -> bytes:
    headers = {"User-Agent": OPEN_METEO_USER_AGENT}
    response = client.get(url, params=params, headers=headers, timeout=60.0)
    response.raise_for_status()
    return response.content


def _cache_filename(year: int, latitude: float, longitude: float) -> str:
    lat_token = f"{latitude:.4f}".replace(".", "p").replace("-", "m")
    lon_token = f"{longitude:.4f}".replace(".", "p").replace("-", "m")
    return f"open_meteo_{lat_token}_{lon_token}_{year}.json"


def fetch_open_meteo_year(  # noqa: PLR0913 - intentional rich keyword-only API
    year: int,
    *,
    cache_dir: Path,
    latitude: float = DEFAULT_KYUSHU_LAT,
    longitude: float = DEFAULT_KYUSHU_LON,
    registry: DatasetRegistry | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
    end_date_override: str | None = None,
) -> Path:
    """Download (with caching) one calendar-year Open-Meteo archive payload.

    ``end_date_override`` (ISO ``YYYY-MM-DD``) is useful for the current
    calendar year, where ERA5 reanalysis has a ~5-day publication lag
    and the default ``{year}-12-31`` request would 400. The cache
    filename remains keyed by ``year`` (one file per calendar year)
    so subsequent calls can extend the range incrementally by setting
    ``force=True``.
    """
    target_dir = (cache_dir / "open_meteo").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _cache_filename(year, latitude, longitude)
    end_date = end_date_override or f"{year}-12-31"
    params = {
        "latitude": f"{latitude}",
        "longitude": f"{longitude}",
        "start_date": f"{year}-01-01",
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "Asia/Tokyo",
    }
    source_url = f"{OPEN_METEO_URL}?{httpx.QueryParams(params)}"

    if target_path.exists() and not force:
        logger.info(
            "Open-Meteo cache hit year=%s coords=(%s,%s): %s",
            year,
            latitude,
            longitude,
            target_path,
        )
    else:
        logger.info("Fetching Open-Meteo year=%s coords=(%s,%s)", year, latitude, longitude)
        owned_client = client is None
        http_client = client or httpx.Client(timeout=60.0, follow_redirects=True)
        try:
            payload = _download(OPEN_METEO_URL, params, http_client)
        finally:
            if owned_client:  # pragma: no cover - default-client cleanup
                http_client.close()
        target_path.write_bytes(payload)

    if registry is not None:
        entry = make_entry(
            name=f"open_meteo_{year}_{latitude}_{longitude}",
            raw_data_dir=cache_dir,
            file_path=target_path,
            source_url=source_url,
        )
        existing = registry.find(entry.name)
        if existing is None or existing.sha256 != entry.sha256:
            registry.upsert(entry)
    return target_path


def parse_open_meteo_json(path: Path) -> pl.DataFrame:
    """Parse one cached Open-Meteo JSON file into a polars DataFrame.

    Returns columns ``datetime_jst``, ``temperature_c``,
    ``cloud_cover_pct``, ``shortwave_radiation_wm2``, ``wind_speed_kmh``.
    ``datetime_jst`` is tz-aware (Asia/Tokyo).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "hourly" not in payload:
        msg = f"Open-Meteo payload is missing 'hourly' section: {path}"
        raise ValueError(msg)
    hourly = payload["hourly"]
    for col in ("time", *WEATHER_VARIABLES):
        if col not in hourly:
            msg = f"Open-Meteo payload is missing column '{col}': {path}"
            raise ValueError(msg)

    # API returns naive ISO strings interpreted in the requested timezone.
    times_naive = [datetime.fromisoformat(t) for t in hourly["time"]]
    times_jst = [t.replace(tzinfo=JST) for t in times_naive]

    return pl.DataFrame(
        {
            "datetime_jst": times_jst,
            "temperature_c": hourly["temperature_2m"],
            "cloud_cover_pct": hourly["cloud_cover"],
            "shortwave_radiation_wm2": hourly["shortwave_radiation"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
        },
        schema={
            "datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
            "temperature_c": pl.Float64,
            "cloud_cover_pct": pl.Float64,
            "shortwave_radiation_wm2": pl.Float64,
            "wind_speed_kmh": pl.Float64,
        },
    ).sort("datetime_jst")


def load_open_meteo_weather(  # noqa: PLR0913 - intentional rich keyword-only API
    years: Iterable[int],
    *,
    cache_dir: Path,
    latitude: float = DEFAULT_KYUSHU_LAT,
    longitude: float = DEFAULT_KYUSHU_LON,
    registry: DatasetRegistry | None = None,
    download: bool = True,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Fetch (or reuse cached) yearly Open-Meteo archive files and concatenate."""
    frames: list[pl.DataFrame] = []
    for year in years:
        target = cache_dir / "open_meteo" / _cache_filename(year, latitude, longitude)
        if not target.exists():
            if not download:
                msg = f"Open-Meteo year {year} not in cache and download=False: {target}"
                raise FileNotFoundError(msg)
            fetch_open_meteo_year(
                year,
                cache_dir=cache_dir,
                latitude=latitude,
                longitude=longitude,
                registry=registry,
                client=client,
            )
        elif (
            registry is not None
            and registry.find(
                f"open_meteo_{year}_{latitude}_{longitude}",
            )
            is None
        ):
            entry = make_entry(
                name=f"open_meteo_{year}_{latitude}_{longitude}",
                raw_data_dir=cache_dir,
                file_path=target,
                source_url=OPEN_METEO_URL,
            )
            registry.upsert(entry)
        frames.append(parse_open_meteo_json(target))

    if not frames:
        return _empty_weather_df()
    return pl.concat(frames).sort("datetime_jst")


def _empty_weather_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
            "temperature_c": pl.Float64,
            "cloud_cover_pct": pl.Float64,
            "shortwave_radiation_wm2": pl.Float64,
            "wind_speed_kmh": pl.Float64,
        },
    )
