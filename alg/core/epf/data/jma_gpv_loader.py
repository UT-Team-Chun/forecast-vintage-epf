"""JMA-MSM GPV GRIB2 loader (true full-period EXACT-issuance forecast tier).

This is the fourth and most operationally honest weather tier, added in
Round-28 to answer the reviewer's headline concern that the full-period
operational head-to-head used the Open-Meteo *Historical-Forecast
reconstruction* (``open_meteo_historical_forecast_loader.py``), which does
not expose the exact model-run initialisation. This loader fetches the raw
JMA Mesoscale-Model (MSM) GPV GRIB2 files from the Kyoto University RISH
archive and pins an EXACT model run, so the issuance time of every value is
known. Whether a pinned run is bid-close *admissible* depends on its
dissemination time, not its initialisation time -- see the next paragraph.

**Run selection vs. the bid gate (corrected in Phase 53).** The JEPX
day-ahead gate closure for delivery day D is D-1 10:00 JST = D-1 01:00 UTC.
JMA *initialises* MSM every 3 h (00/03/06/... UTC), but initialisation is not
availability: per JMBSC the GPV files are disseminated ~2 h 30 m after the
initialisation time. The **D-1 00:00 UTC run** used by
``load_jma_gpv_forecast_multi`` is initialised D-1 09:00 JST yet only becomes
available ~D-1 11:30 JST — i.e. ~1.5 h AFTER gate closure. It is a true
*exact-issuance* series (single pinned model run, known lead:
``model = "jma_msm_gpv"``, ``forecast_kind = "exact_issuance"``,
``issuance_time_utc`` = the actual run, ``lead_hours`` = the actual lead),
but it is NOT pre-gate admissible; treat it as a post-gate anchor. An earlier
version of this docstring wrongly called it bid-close-admissible by conflating
initialisation with dissemination.

**Pre-gate exact-admissible run set** (availability = init + 2.5 h <=
D-1 10:00 JST): the **D-2 21 UTC run** (disseminated ~D-1 08:30 JST, 1.5 h
before the gate; FH18-39 cover delivery-day hours 00-21 JST) topped up with
the **D-2 12 UTC run** (disseminated ~D-2 23:30 JST; FH49-50 cover hours
22-23 JST beyond the 21 UTC file horizon). Use ``fetch_gpv_run_points`` with
``SURFACE_FH_FILES_PREGATE_21Z`` / ``SURFACE_FH_FILES_PREGATE_12Z`` for that
path (Phase 53 sandbox script builds the per-day merged series).

**Variables -> canonical schema** (matches ``DEFAULT_WEATHER_COLS``):
  * ``temperature_c``         <- ``t`` (2 m Temperature, K) - 273.15
  * ``cloud_cover_pct``       <- random-overlap total of low/mid/high cloud
  * ``shortwave_radiation_wm2`` <- ``avg_sdswrf`` (Time-mean surface downward
    short-wave radiation flux, W/m^2; stepType=avg -> already a per-hour mean,
    so NO de-accumulation is needed and it is directly comparable to ERA5 /
    Historical-Forecast ``shortwave_radiation``)
  * ``wind_speed_kmh``        <- sqrt(u10^2 + v10^2) * 3.6

RISH archive URL pattern (no login; covers the full study period 2024-09+)::

    http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/
        {YYYY}/{MM}/{DD}/Z__C_RJTD_{YYYYMMDD}000000_MSM_GPV_Rjp_Lsurf_{FHRANGE}_grib2.bin

Raw GRIB2 files (~50-80 MB each) are downloaded to a temp dir, the area grid
points are extracted in one pass, the tiny per-area hourly series is cached as
parquet, and the raw is discarded -- so disk stays small and the multi-area
fetch downloads each run once.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import eccodes as ec  # optional geo dependency (declared in the alg[geo] extra)
import httpx
import numpy as np
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from alg.core.epf.data.registry import DatasetRegistry, make_entry

log = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9), name="JST")
RISH_BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
# Surface FH-range files of one run that, for the D-1 00 UTC run, cover
# delivery day D (FH15-38 JST 00-23): FH00-15 (gives FH15=00 JST), FH16-33
# (01-18 JST, includes the 09-16 JST identification window), FH34-39 (19-24 JST).
SURFACE_FH_FILES_FULL = ("FH00-15", "FH16-33", "FH34-39")
# Daytime/identification-only subset (FH16-33 = 01-18 JST incl. full midday).
SURFACE_FH_FILES_DAYTIME = ("FH16-33",)
# Pre-gate exact-admissible set (Phase 53). For delivery day D:
#   D-2 21 UTC run: FH16-33 + FH34-39 files contain FH18-39 = D 00:00-21:00 JST.
SURFACE_FH_FILES_PREGATE_21Z = ("FH16-33", "FH34-39")
#   D-2 12 UTC run: FH40-51 file contains FH49-50 = D 22:00-23:00 JST (horizon
#   tail beyond the 21 UTC run's last surface file).
SURFACE_FH_FILES_PREGATE_12Z = ("FH40-51",)
# Single-run whole-delivery-day sets (Phase 58, exact-run ladder). Unlike the
# Phase 53 composite above, each of these covers all 24 delivery hours of day D
# from ONE pinned run, so the resulting series is a real forecast a bidder could
# have held rather than a fixed-lead mosaic.
#   D-2 12 UTC run: FH27-50 = D 00:00-23:00 JST. The 78 h horizon exists only for
#   the 00/12 UTC cycles, so this is the ONLY pre-gate run that spans a whole
#   delivery day (dissemination D-2 23:30 JST, +10 h 30 m before the gate).
SURFACE_FH_FILES_D2_12Z_FULLDAY = ("FH16-33", "FH34-39", "FH40-51")
#   D-1 00 UTC run: FH15-38 = D 00:00-23:00 JST. First POST-gate run
#   (dissemination ~D-1 11:30 JST, 1 h 30 m AFTER the gate) - same file set as
#   ``SURFACE_FH_FILES_FULL``, named separately so callers state their intent.
SURFACE_FH_FILES_D1_00Z_FULLDAY = SURFACE_FH_FILES_FULL
# Beyond FH51 the archive ships ONE file, FH52-78 (~118 MB) -- not the FH52-63 +
# FH64-75 pair an earlier plan guessed at (both 404). Verified by listing the
# 2024-09-01 directory and HEAD-probing five dates across 2024-2026. From a
# D-3 12 UTC run it covers delivery hours 01:00-23:00 JST, so it alone spans the
# 08:00-16:00 JST common evaluation window.
SURFACE_FH_FILES_DEEP = ("FH52-78",)
WEATHER_COLS = ("temperature_c", "cloud_cover_pct", "shortwave_radiation_wm2", "wind_speed_kmh")


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TransportError)),
)
def _download(url: str, dest: Path, client: httpx.Client) -> str | None:
    """Stream one file to disk and return its Last-Modified header, if any.

    The header is returned rather than discarded because the manuscript's central
    claim is about DISSEMINATION time, and for months the paper carried a
    "measured +2 h 16 m" lag whose measurement no cache had actually persisted.
    The archive's own timestamp is the only observable evidence; from now on every
    download keeps it.
    """
    with client.stream("GET", url) as r:
        r.raise_for_status()
        last_modified = r.headers.get("Last-Modified")
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    return last_modified


def _run_url(run_utc: datetime, fh_range: str) -> str:
    # Init hour comes from ``run_utc`` itself (00/03/.../21 UTC). Existing
    # callers pass midnight datetimes, for which this renders the same
    # ``...000000...`` filename as before.
    return (
        f"{RISH_BASE}/{run_utc:%Y}/{run_utc:%m}/{run_utc:%d}/"
        f"Z__C_RJTD_{run_utc:%Y%m%d%H%M%S}_MSM_GPV_Rjp_Lsurf_{fh_range}_grib2.bin"
    )


_GRIB_SHORTNAMES = ("t", "10u", "10v", "lcc", "mcc", "hcc", "avg_sdswrf")


def _extract_points(
    grib_path: Path,
    coords: dict[str, tuple[float, float]],
) -> dict[str, pl.DataFrame]:
    """Decode one surface GRIB2 file and extract per-area hourly canonical rows.

    Streams message-by-message via the low-level eccodes API with multi-field
    support ON (the JMA-MSM files pack multiple fields per GRIB2 message, so
    ``codes_grib_new_from_file`` collapses them to one handle without it). Peak
    RSS is ~150 MB vs ``cfgrib.open_datasets``'s ~7.5 GiB eager full-grid load,
    which OOM-killed multi-day sweeps on both the cluster and the local sandbox.
    Values are identical to the cfgrib nearest-grid-point extraction.
    """
    ec.codes_grib_multi_support_on()
    needed = set(_GRIB_SHORTNAMES)
    # per area: {valid_utc -> {shortName -> nearest-grid-point value}}
    per_area: dict[str, dict[datetime, dict[str, float]]] = {a: {} for a in coords}
    with grib_path.open("rb") as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                sn = ec.codes_get(gid, "shortName")
                if sn not in needed:
                    continue
                vd = int(ec.codes_get(gid, "validityDate"))
                vt = int(ec.codes_get(gid, "validityTime"))
                valid = datetime(
                    vd // 10000, (vd // 100) % 100, vd % 100, vt // 100, vt % 100, tzinfo=UTC
                )
                for area, (lat, lon) in coords.items():
                    near = ec.codes_grib_find_nearest(gid, lat, lon)[0]
                    per_area[area].setdefault(valid, {})[sn] = float(near.value)
            finally:
                ec.codes_release(gid)

    out: dict[str, pl.DataFrame] = {}
    for area in coords:
        valids = sorted(per_area[area])
        if not valids:
            out[area] = pl.DataFrame()
            continue

        def col(sn: str, _v: list = valids, _a: str = area) -> np.ndarray:
            return np.array([per_area[_a][t].get(sn, np.nan) for t in _v], dtype=float)

        u, w = col("10u"), col("10v")
        lcc, mcc, hcc = col("lcc") / 100.0, col("mcc") / 100.0, col("hcc") / 100.0
        dt_jst = pl.Series("datetime_jst", valids).dt.convert_time_zone("Asia/Tokyo")
        out[area] = pl.DataFrame(
            {
                "datetime_jst": dt_jst,
                "temperature_c": col("t") - 273.15,
                "cloud_cover_pct": (1.0 - (1.0 - lcc) * (1.0 - mcc) * (1.0 - hcc)) * 100.0,
                "shortwave_radiation_wm2": col("avg_sdswrf"),
                "wind_speed_kmh": np.sqrt(u**2 + w**2) * 3.6,
            }
        )
    return out


def fetch_gpv_run_points(
    run_utc: datetime,
    fh_files: tuple[str, ...],
    area_coords: dict[str, tuple[float, float]],
    *,
    client: httpx.Client | None = None,
) -> dict[str, pl.DataFrame]:
    """Download the given surface FH files of ONE MSM run and extract points.

    Init-hour-agnostic building block for arbitrary runs (00/03/.../21 UTC),
    added in Phase 53 for the pre-gate exact-admissible tier. Unlike
    ``load_jma_gpv_forecast_multi`` it does not cache and does not swallow
    failures: any HTTP error (after tenacity retries) or GRIB decode error
    raises. Files are fetched sequentially and each raw GRIB is deleted right
    after extraction, so disk stays bounded to one file.

    Returns ``{area: DataFrame[datetime_jst, *WEATHER_COLS, issuance_time_utc,
    lead_hours]}`` with ALL valid hours contained in ``fh_files``; the caller
    applies delivery-day / slot filtering.
    """
    if run_utc.tzinfo is None:
        run_utc = run_utc.replace(tzinfo=UTC)
    owns_client = client is None
    http = client or httpx.Client(timeout=180.0, follow_redirects=True)
    per_file: list[dict[str, pl.DataFrame]] = []
    last_modified: dict[str, str | None] = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            for fh in fh_files:
                dest = Path(td) / f"{fh}.grib2"
                last_modified[fh] = _download(_run_url(run_utc, fh), dest, http)
                per_file.append(_extract_points(dest, area_coords))
                dest.unlink()
    finally:
        if owns_client:
            http.close()
    fetch_gpv_run_points.last_modified = last_modified  # type: ignore[attr-defined]

    out: dict[str, pl.DataFrame] = {}
    for area in area_coords:
        frames = [pf[area] for pf in per_file if pf[area].height > 0]
        if not frames:
            out[area] = pl.DataFrame()
            continue
        df = pl.concat(frames).unique(subset=["datetime_jst"]).sort("datetime_jst")
        out[area] = df.with_columns(
            pl.lit(run_utc).alias("issuance_time_utc"),
            (
                (
                    pl.col("datetime_jst").dt.convert_time_zone("UTC") - pl.lit(run_utc)
                ).dt.total_hours()
            ).alias("lead_hours"),
        )
    return out


def _cache_path(cache_dir: Path, area: str, lat: float, lon: float) -> Path:
    d = cache_dir / "jma_gpv"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"jma_gpv_{area}_{lat:.4f}_{lon:.4f}.parquet"


def load_jma_gpv_forecast_multi(  # noqa: PLR0913, PLR0912, PLR0915, C901 - download/cache/merge loop
    delivery_days: list[date],
    area_coords: dict[str, tuple[float, float]],
    *,
    cache_dir: Path,
    registry: DatasetRegistry | None = None,
    daytime_only: bool = True,
    client: httpx.Client | None = None,
) -> dict[str, pl.DataFrame]:
    """Exact-issuance JMA-MSM GPV forecast per area over ``delivery_days``.

    For each delivery day D, the D-1 00:00 UTC run's surface files are
    downloaded once (shared across all areas), all area grid points are
    extracted, and the per-area hourly canonical series is cached. Returns
    ``{area: DataFrame[datetime_jst, *WEATHER_COLS, issuance_time_utc,
    lead_hours]}`` covering each delivery day's hours.
    """
    fh_files = SURFACE_FH_FILES_DAYTIME if daytime_only else SURFACE_FH_FILES_FULL
    # Resume from cache: keep only delivery days not already cached for area[0].
    cache = {a: _cache_path(cache_dir, a, lat, lon) for a, (lat, lon) in area_coords.items()}
    cached: dict[str, pl.DataFrame] = {}
    for a, p in cache.items():
        if p.exists():
            cached[a] = pl.read_parquet(p)
    have_days = (
        set(next(iter(cached.values()))["datetime_jst"].dt.date().to_list()) if cached else set()
    )
    todo = [d for d in delivery_days if d not in have_days]
    log.info(
        "JMA-GPV: %d delivery days requested, %d cached, %d to fetch",
        len(delivery_days),
        len(have_days),
        len(todo),
    )

    owns_client = client is None
    http = client or httpx.Client(timeout=180.0, follow_redirects=True)
    new_rows: dict[str, list[pl.DataFrame]] = {a: [] for a in area_coords}
    try:
        for d in todo:
            run_utc = datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(days=1)
            with tempfile.TemporaryDirectory() as td:
                per_file: list[dict[str, pl.DataFrame]] = []
                ok = True
                for fh in fh_files:
                    url = _run_url(run_utc, fh)
                    dest = Path(td) / f"{fh}.grib2"
                    try:
                        _download(url, dest, http)
                        per_file.append(_extract_points(dest, area_coords))
                    except (httpx.HTTPError, ValueError) as exc:
                        log.warning(
                            "JMA-GPV %s run %s %s skipped: %s",
                            d.isoformat(),
                            run_utc.isoformat(),
                            fh,
                            exc,
                        )
                        ok = False
                        break
                if not ok:
                    continue
                for area in area_coords:
                    frames = [pf[area] for pf in per_file]
                    day_df = (
                        pl.concat(frames)
                        .unique(subset=["datetime_jst"])
                        .sort("datetime_jst")
                        .filter(pl.col("datetime_jst").dt.date() == d)
                        .with_columns(
                            pl.lit(run_utc).alias("issuance_time_utc"),
                        )
                    )
                    new_rows[area].append(day_df)
    finally:
        if owns_client:
            http.close()

    result: dict[str, pl.DataFrame] = {}
    for area, (lat, lon) in area_coords.items():
        parts = []
        if area in cached:
            drop_c = [c for c in ("issuance_time_utc",) if c in cached[area].columns]
            parts.append(cached[area].drop(drop_c))
        if new_rows[area]:
            fresh = pl.concat(new_rows[area]).drop(
                [c for c in ("issuance_time_utc",) if c in new_rows[area][0].columns],
            )
            parts.append(fresh)
        if not parts:
            result[area] = pl.DataFrame()
            continue
        merged = pl.concat(parts).unique(subset=["datetime_jst"]).sort("datetime_jst")
        merged.write_parquet(cache[area])
        if registry is not None:
            entry = make_entry(
                name=f"jma_gpv_extract_{area}_{lat:.4f}_{lon:.4f}",
                raw_data_dir=cache_dir,
                file_path=cache[area],
                source_url=f"{RISH_BASE} (D-1 00UTC MSM run, exact-issuance; "
                "disseminated ~D-1 11:30 JST = post-gate)",
            )
            registry.upsert(entry)
        result[area] = merged
    return result
