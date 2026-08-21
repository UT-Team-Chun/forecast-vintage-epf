"""Open-Meteo Historical-Forecast API loader (full-period operational PROXY).

This is the third weather tier in the project, sitting between
``open_meteo_loader.py`` (ERA5 reanalysis = ex-post ground truth, used in
the paper as a *lagged-reanalysis proxy* / upper-bound information set) and
``open_meteo_forecast_loader.py`` (Single-Runs API = true exact-issuance
JMA-MSM operational forecast, but only archived from ~2025-08-08).

**What this loader provides.** The Open-Meteo *Historical Forecast API*
(``https://historical-forecast-api.open-meteo.com``) is a continuous
reconstruction archive of the best available past model forecast for each
timestamp. Unlike the Single-Runs API it covers the full paper period
(2024-09 onward) in a single date-range request, but it does **not** expose
the exact model-run initialisation time. It is therefore an
*operational-grade PROXY* for an ex-ante forecast, not a true
exact-issuance series.

**Why it exists in Round-27.** It is the cheap *de-risk PROBE* weather tier:
re-running the cross-area gradient and the slot-level interaction with this
proxy answers "does the mechanism survive operational-grade forecasts at
all?" before committing to the heavy exact-issuance JMA-GPV GRIB2 build.

**Issuance-ambiguity caveat (load-bearing — do not paper over).** Because
the reconstruction blends model runs to give the *best* past forecast for a
timestamp, some of its skill could come from a run issued *after* the JEPX
bid-close (D-1 10:00 JST = D-1 01:00 UTC) — i.e., subtle look-ahead
relative to a strict operational setting. Two honest guards:

1. We never call this "ex-ante"; ``model = "jma_msm_histfc"`` and
   ``forecast_kind = "historical_reconstruction"`` flag the reconstruction
   provenance everywhere downstream.
2. We attach a *bid-close proxy* timestamp (``bid_close_proxy_utc`` =
   D-1 01:00 UTC for the JST delivery date) and a ``lead_hours_proxy``, so
   the lead-time convention is comparable to the Single-Runs loader, and so
   an overlap calibration (this proxy vs the true Single-Runs run on the
   2025-08-08+ window) can bound the look-ahead.

Endpoint pattern (one request per coordinate covers the whole date range)::

    https://historical-forecast-api.open-meteo.com/v1/forecast?
        latitude={lat}&longitude={lon}
        &start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}
        &models=jma_msm
        &hourly=temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m
        &timezone=Asia/Tokyo
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta, timezone
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

OPEN_METEO_HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_USER_AGENT = "forecast-vintage-epf/0.1 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"
JST = timezone(timedelta(hours=9), name="JST")

# Same raw API variable set + canonical output column names as the other
# two Open-Meteo loaders, so this is a drop-in for ``attach_hourly_weather``.
WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
)
WEATHER_OUTPUT_COLUMNS: tuple[str, ...] = (
    "datetime_jst",
    "temperature_c",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "wind_speed_kmh",
)

# JMA Mesoscale Model reconstruction. The "_histfc" suffix marks the
# Historical-Forecast reconstruction provenance (vs "jma_msm" exact-issuance
# in the Single-Runs loader and ERA5 in open_meteo_loader).
DEFAULT_MODEL = "jma_msm"
MODEL_PROVENANCE_TOKEN = "jma_msm_histfc"  # noqa: S105 - provenance token, not a secret
FORECAST_KIND = "historical_reconstruction"

# JEPX bid-close is D-1 10:00 JST = D-1 01:00 UTC. The bid-close proxy run
# hour mirrors the Single-Runs admissible run (D-1 00:00 UTC).
BID_CLOSE_PROXY_RUN_HOUR_UTC = 0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TransportError)),
    reraise=True,
)
def _download(url: str, params: dict[str, str], client: httpx.Client) -> bytes:
    headers = {"User-Agent": OPEN_METEO_USER_AGENT}
    response = client.get(url, params=params, headers=headers, timeout=120.0)
    response.raise_for_status()
    return response.content


def _cache_filename(
    start: date,
    end: date,
    latitude: float,
    longitude: float,
    model: str,
) -> str:
    lat_token = f"{latitude:.4f}".replace(".", "p").replace("-", "m")
    lon_token = f"{longitude:.4f}".replace(".", "p").replace("-", "m")
    return f"open_meteo_histfc_{model}_{lat_token}_{lon_token}_{start:%Y%m%d}_{end:%Y%m%d}.json"


def fetch_open_meteo_historical_forecast(  # noqa: PLR0913 - rich keyword-only API
    start_date: date,
    end_date: date,
    *,
    cache_dir: Path,
    latitude: float,
    longitude: float,
    model: str = DEFAULT_MODEL,
    registry: DatasetRegistry | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> Path:
    """Download (with caching) one date-range reconstruction for a coordinate.

    A single request returns the continuous hourly reconstruction over
    ``[start_date, end_date]`` (inclusive), so the whole paper period for one
    area is one cached file. The cached payload lives under
    ``cache_dir / "open_meteo_historical_forecast"``.
    """
    target_dir = (cache_dir / "open_meteo_historical_forecast").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _cache_filename(
        start_date,
        end_date,
        latitude,
        longitude,
        model,
    )

    params = {
        "latitude": f"{latitude}",
        "longitude": f"{longitude}",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "models": model,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "Asia/Tokyo",
    }
    source_url = f"{OPEN_METEO_HISTFC_URL}?{httpx.QueryParams(params)}"

    if target_path.exists() and not force:
        logger.debug(
            "Open-Meteo Historical-Forecast cache hit %s..%s coords=(%s,%s) model=%s: %s",
            start_date,
            end_date,
            latitude,
            longitude,
            model,
            target_path,
        )
    else:
        logger.info(
            "Fetching Open-Meteo Historical-Forecast %s..%s coords=(%s,%s) model=%s",
            start_date,
            end_date,
            latitude,
            longitude,
            model,
        )
        owned_client = client is None
        http_client = client or httpx.Client(timeout=120.0, follow_redirects=True)
        try:
            payload = _download(OPEN_METEO_HISTFC_URL, params, http_client)
        finally:
            if owned_client:
                http_client.close()
        target_path.write_bytes(payload)

    if registry is not None:
        entry = make_entry(
            name=(
                f"open_meteo_histfc_{model}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
                f"_{latitude}_{longitude}"
            ),
            raw_data_dir=cache_dir,
            file_path=target_path,
            source_url=source_url,
        )
        existing = registry.find(entry.name)
        if existing is None or existing.sha256 != entry.sha256:
            registry.upsert(entry)
    return target_path


def _bid_close_proxy_utc(delivery_time_jst: datetime) -> datetime:
    """Return the bid-close proxy run time (D-1 00:00 UTC) for a delivery slot.

    The JST delivery date is D; the most recent bid-close-admissible model
    run is initialised at D-1 ``BID_CLOSE_PROXY_RUN_HOUR_UTC`` UTC.
    """
    d = delivery_time_jst.astimezone(JST).date()
    return datetime(
        d.year,
        d.month,
        d.day,
        BID_CLOSE_PROXY_RUN_HOUR_UTC,
        0,
        tzinfo=UTC,
    ) - timedelta(days=1)


def parse_open_meteo_historical_forecast_json(
    path: Path,
    *,
    drop_nulls: bool = True,
) -> pl.DataFrame:
    """Parse one cached reconstruction payload into the canonical schema.

    Returns ``(datetime_jst, temperature_c, cloud_cover_pct,
    shortwave_radiation_wm2, wind_speed_kmh)`` plus reconstruction-honest
    metadata ``model`` (= ``"jma_msm_histfc"``), ``forecast_kind``,
    ``bid_close_proxy_utc`` and ``lead_hours_proxy``. The proxy columns are
    *not* a true issuance stamp — the Historical-Forecast API does not expose
    one — they encode the bid-close convention so the lead-time semantics
    match the Single-Runs loader for overlap calibration.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "hourly" not in payload:
        msg = f"Open-Meteo Historical-Forecast payload is missing 'hourly': {path}"
        raise ValueError(msg)
    hourly = payload["hourly"]
    for col in ("time", *WEATHER_VARIABLES):
        if col not in hourly:
            msg = f"Open-Meteo Historical-Forecast payload is missing '{col}': {path}"
            raise ValueError(msg)

    times_naive = [datetime.fromisoformat(t) for t in hourly["time"]]
    times_jst = [t.replace(tzinfo=JST) for t in times_naive]
    bid_close = [_bid_close_proxy_utc(t) for t in times_jst]
    lead_hours_proxy = [
        int((t.astimezone(UTC) - b).total_seconds() // 3600)
        for t, b in zip(times_jst, bid_close, strict=True)
    ]

    df = pl.DataFrame(
        {
            "datetime_jst": times_jst,
            "temperature_c": hourly["temperature_2m"],
            "cloud_cover_pct": hourly["cloud_cover"],
            "shortwave_radiation_wm2": hourly["shortwave_radiation"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
            "bid_close_proxy_utc": bid_close,
            "lead_hours_proxy": lead_hours_proxy,
            "model": [MODEL_PROVENANCE_TOKEN] * len(times_jst),
            "forecast_kind": [FORECAST_KIND] * len(times_jst),
        },
        schema={
            "datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
            "temperature_c": pl.Float64,
            "cloud_cover_pct": pl.Float64,
            "shortwave_radiation_wm2": pl.Float64,
            "wind_speed_kmh": pl.Float64,
            "bid_close_proxy_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
            "lead_hours_proxy": pl.Int64,
            "model": pl.Utf8,
            "forecast_kind": pl.Utf8,
        },
    ).sort("datetime_jst")

    if drop_nulls:
        df = df.filter(pl.col("temperature_c").is_not_null())
    return df


def load_open_meteo_historical_forecast_archive(  # noqa: PLR0913 - rich keyword-only API
    delivery_days: Iterable[date],
    *,
    cache_dir: Path,
    latitude: float,
    longitude: float,
    model: str = DEFAULT_MODEL,
    registry: DatasetRegistry | None = None,
    download: bool = True,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Fetch the reconstruction over the span of ``delivery_days`` and return rows.

    Unlike the Single-Runs loader (one request per run), this issues a single
    date-range request covering ``[min(delivery_days), max(delivery_days)]``
    and returns the canonical-schema rows whose JST calendar date is in
    ``delivery_days``. Output is a drop-in for
    ``open_meteo_loader.load_open_meteo_weather`` /
    ``attach_hourly_weather``.
    """
    wanted = sorted(set(delivery_days))
    if not wanted:
        return _empty_histfc_df()
    start_date, end_date = wanted[0], wanted[-1]

    target = (
        cache_dir
        / "open_meteo_historical_forecast"
        / _cache_filename(
            start_date,
            end_date,
            latitude,
            longitude,
            model,
        )
    )
    if not target.exists():
        if not download:
            msg = (
                f"Open-Meteo Historical-Forecast payload not in cache and download=False: {target}"
            )
            raise FileNotFoundError(msg)
        target = fetch_open_meteo_historical_forecast(
            start_date,
            end_date,
            cache_dir=cache_dir,
            latitude=latitude,
            longitude=longitude,
            model=model,
            registry=registry,
            client=client,
        )
    elif (
        registry is not None
        and registry.find(
            f"open_meteo_histfc_{model}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
            f"_{latitude}_{longitude}",
        )
        is None
    ):
        entry = make_entry(
            name=(
                f"open_meteo_histfc_{model}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
                f"_{latitude}_{longitude}"
            ),
            raw_data_dir=cache_dir,
            file_path=target,
            source_url=OPEN_METEO_HISTFC_URL,
        )
        registry.upsert(entry)

    df = parse_open_meteo_historical_forecast_json(target)
    wanted_set = set(wanted)
    return df.filter(
        pl.col("datetime_jst").dt.date().is_in(list(wanted_set)),
    ).sort("datetime_jst")


def _empty_histfc_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
            "temperature_c": pl.Float64,
            "cloud_cover_pct": pl.Float64,
            "shortwave_radiation_wm2": pl.Float64,
            "wind_speed_kmh": pl.Float64,
            "bid_close_proxy_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
            "lead_hours_proxy": pl.Int64,
            "model": pl.Utf8,
            "forecast_kind": pl.Utf8,
        },
    )
