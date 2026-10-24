r"""Phase 112 data: run-pinned JMA-MSM tiers for the 2022-24 scarcity backcast.

WHY THIS EXISTS
---------------
Two Applied Energy referees (APEN-D-26-19971, R1.6 and R5.3) objected that the
evaluation window holds almost no scarcity pricing, so the paper cannot say
whether forecast freshness matters more when prices are set by tight supply
rather than by photovoltaic surplus. A third (R1.9) asked for a longer window.
The Kyoto RISH mirror holds the same MSM GRIB2 files back to 2021 with their
original write times, so the identical design can be re-run on the two years
that precede the primary window -- a span that contains the summer-2022 supply
crunch (about 2,500 area-slots at or above 50 JPY/kWh, capped at 200).
Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md.

WHAT IS FETCHED
---------------
For every delivery day D in [--start, --end], two pinned runs, exactly as
phase58 defines them:
  A  msm_d2_12z  D-2 12 UTC, files FH16-33/FH34-39/FH40-51, leads 27..50
  C  msm_d1_00z  D-1 00 UTC, files FH00-15/FH16-33/FH34-39, leads 15..38
Each downloaded file is decoded ONCE for 55 points: the 46 capacity-table
prefecture points of the primary measurement (2024-09 vintage, the same file
the primary ladder reads) and the 9 single representative points of the
sensitivity arm. The weights are not applied here; the ladder applies them.

FREEZE POLICY
-------------
Writes ONLY under data/raw/jma_gpv_backcast/. The frozen 2024-26 caches under
jma_gpv_exactrun{,_mp}/ are never touched, so nothing the primary ladder reads
can change.

ARCHIVE TIMESTAMPS
------------------
Every file's Last-Modified header is appended to _last_modified.jsonl, which
extends the dissemination audit back to 2022 at no extra request cost.

FAILURE POLICY
--------------
A missing file (HTTP 404 after retries) marks that plan-day missing and the sweep
continues: a genuine archive gap is a data fact to report, not a reason to stop a
ten-hour download. Any other error -- decode failure, wrong hours, wrong leads,
NaN -- aborts, because those mean the extraction itself is wrong.

CLI (from the repository root) -- daemonise it, the phase53 lesson:
  nohup uv run python \
      scripts/08_extract_backcast_runs.py \
      --start 2022-01-01 --end 2024-07-31 > /tmp/phase112_extract.log 2>&1 & disown
  uv run python ... --validate
  # the 2020-21 extension runs as a second process into its own root:
  nohup uv run python ... --start 2020-08-01 --end 2021-12-31 \
      --root-name jma_gpv_backcast_early > .../_sweep.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import eccodes as ec
import httpx
import numpy as np
import polars as pl
from tenacity import RetryError

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points
from alg.core.epf.data.jma_gpv_loader import (
    _GRIB_SHORTNAMES,
    SURFACE_FH_FILES_D1_00Z_FULLDAY,
    SURFACE_FH_FILES_D2_12Z_FULLDAY,
    WEATHER_COLS,
    _download,
    _run_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase112x")

JST = timezone(timedelta(hours=9), name="JST")
RAW = EpfConfig().resolved("raw_data_dir")
ROOT = RAW / "jma_gpv_backcast"
PROGRESS = ROOT / "_progress.json"
LASTMOD = ROOT / "_last_modified.jsonl"


def _set_root(name: str) -> None:
    """Point every output at RAW/<name>. Two sweeps over disjoint date ranges run
    as separate processes, and they must not share a progress file or a parquet:
    each flush rewrites the file, so a shared root would race."""
    global ROOT, PROGRESS, LASTMOD  # noqa: PLW0603 - module-level paths, set once from the CLI
    ROOT = RAW / name
    PROGRESS = ROOT / "_progress.json"
    LASTMOD = ROOT / "_last_modified.jsonl"
WEIGHT_VINTAGE = "2024-09"
POLITE_SLEEP_S = 1.0
FLUSH_EVERY_DAYS = 5

# Identical to phase58's AREA_COORDS (the single-point sensitivity arm).
AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}

# key -> (days_before, init_hour_utc, files); the phase58 plans A and C verbatim.
PLANS: dict[str, tuple[int, int, tuple[str, ...]]] = {
    "msm_d2_12z": (2, 12, SURFACE_FH_FILES_D2_12Z_FULLDAY),
    "msm_d1_00z": (1, 0, SURFACE_FH_FILES_D1_00Z_FULLDAY),
}


def _coords() -> dict[str, tuple[float, float]]:
    """55 point keys: 'cw__<area>__pNN' (46) and 'sp__<area>' (9)."""
    out: dict[str, tuple[float, float]] = {}
    for area, pts in area_points(WEIGHT_VINTAGE).items():
        for i, p in enumerate(pts):
            out[f"cw__{area}__p{i:02d}"] = (p.lat, p.lon)
    for area, ll in AREA_COORDS.items():
        out[f"sp__{area}"] = ll
    return out


def _run_utc(plan: str, d: date) -> datetime:
    days_before, hour, _ = PLANS[plan]
    base = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return (base - timedelta(days=days_before)).replace(hour=hour)


def _expected_leads(plan: str, d: date) -> list[int]:
    first = datetime(d.year, d.month, d.day, 0, tzinfo=JST).astimezone(UTC)
    lo = int((first - _run_utc(plan, d)).total_seconds() // 3600)
    return list(range(lo, lo + 24))


def _load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"complete": {p: [] for p in PLANS}, "missing": {p: [] for p in PLANS}}


def _flush(buffers: dict[str, dict[str, list[pl.DataFrame]]]) -> None:
    for plan, per_key in buffers.items():
        pdir = ROOT / plan
        pdir.mkdir(parents=True, exist_ok=True)
        for key, frames in per_key.items():
            if not frames:
                continue
            path = pdir / f"{key}.parquet"
            new = pl.concat(frames)
            if path.exists():
                new = pl.concat([pl.read_parquet(path), new])
            (new.unique("datetime_jst", keep="last").sort("datetime_jst")
                .write_parquet(path))
            frames.clear()


def _extract_points_indexed(grib_path: Path,
                            coords: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """The loader's ``_extract_points`` with the nearest-point search hoisted.

    The loader calls ``codes_grib_find_nearest`` for every point of every message,
    which is 8 s per file at 55 points and would have made this sweep a 32-hour
    job. The grid is fixed within a file (asserted), so the nearest index is found
    once on the first message and reused; values are then read from one decoded
    array per message. Verified bit-identical to the loader (max |diff| 0.0 over
    48 points x every field of one FH40-51 file) before this sweep ran.
    """
    ec.codes_grib_multi_support_on()
    needed = set(_GRIB_SHORTNAMES)
    idx: dict[str, int] | None = None
    sig0: tuple | None = None
    per: dict[str, dict] = {k: {} for k in coords}
    with grib_path.open("rb") as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                sn = ec.codes_get(gid, "shortName")
                if sn not in needed:
                    continue
                sig = tuple(ec.codes_get(gid, k) for k in (
                    "gridType", "Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees"))
                if idx is None:
                    sig0 = sig
                    idx = {k: int(ec.codes_grib_find_nearest(gid, la, lo)[0].index)
                           for k, (la, lo) in coords.items()}
                elif sig != sig0:
                    msg = f"{grib_path.name}: grid changed within file {sig} vs {sig0}"
                    raise ValueError(msg)
                vd = int(ec.codes_get(gid, "validityDate"))
                vt = int(ec.codes_get(gid, "validityTime"))
                valid = datetime(vd // 10000, (vd // 100) % 100, vd % 100,
                                 vt // 100, vt % 100, tzinfo=UTC)
                vals = ec.codes_get_values(gid)
                for k, i in idx.items():
                    per[k].setdefault(valid, {})[sn] = float(vals[i])
            finally:
                ec.codes_release(gid)
    return per


def _canonical(per: dict[datetime, dict], run: datetime) -> pl.DataFrame:
    """Loader formulas verbatim (jma_gpv_loader._extract_points)."""
    valids = sorted(per)

    def col(sn: str) -> np.ndarray:
        return np.array([per[t].get(sn, np.nan) for t in valids], dtype=float)

    u, w = col("10u"), col("10v")
    lcc, mcc, hcc = col("lcc") / 100.0, col("mcc") / 100.0, col("hcc") / 100.0
    dt_jst = pl.Series("datetime_jst", valids).dt.convert_time_zone("Asia/Tokyo")
    return pl.DataFrame({
        "datetime_jst": dt_jst,
        "temperature_c": col("t") - 273.15,
        "cloud_cover_pct": (1.0 - (1.0 - lcc) * (1.0 - mcc) * (1.0 - hcc)) * 100.0,
        "shortwave_radiation_wm2": col("avg_sdswrf"),
        "wind_speed_kmh": np.sqrt(u**2 + w**2) * 3.6,
    }).with_columns(
        pl.lit(run).alias("issuance_time_utc"),
        ((pl.col("datetime_jst").dt.convert_time_zone("UTC") - pl.lit(run))
         .dt.total_hours()).alias("lead_hours"),
    )


def _fetch_run(run: datetime, files: tuple[str, ...], coords: dict,
               client: httpx.Client) -> tuple[dict[str, pl.DataFrame], dict]:
    merged: dict[str, dict] = {k: {} for k in coords}
    lastmod: dict[str, str | None] = {}
    with tempfile.TemporaryDirectory() as td:
        for fh in files:
            dest = Path(td) / f"{fh}.grib2"
            lastmod[fh] = _download(_run_url(run, fh), dest, client)
            for k, rows in _extract_points_indexed(dest, coords).items():
                merged[k].update(rows)
            dest.unlink()
    return {k: _canonical(v, run) for k, v in merged.items()}, lastmod


def _fetch(plan: str, d: date, coords: dict, client: httpx.Client) -> dict[str, pl.DataFrame]:
    _, _, files = PLANS[plan]
    run = _run_utc(plan, d)
    per_key, lastmod = _fetch_run(run, files, coords, client)
    with LASTMOD.open("a", encoding="utf-8") as f:
        for fh, lm in lastmod.items():
            f.write(json.dumps({"delivery_day": d.isoformat(), "plan": plan,
                                "run_utc": run.isoformat(), "fh": fh,
                                "last_modified": lm}) + "\n")
    leads = _expected_leads(plan, d)
    out: dict[str, pl.DataFrame] = {}
    for key in coords:
        df = (per_key[key].filter(pl.col("datetime_jst").dt.date() == d)
              .sort("datetime_jst"))
        hours = df["datetime_jst"].dt.hour().to_list()
        if hours != list(range(24)):
            msg = f"{plan} {d} {key}: expected hours 0..23, got {hours}"
            raise ValueError(msg)
        if df["lead_hours"].to_list() != leads:
            msg = f"{plan} {d} {key}: leads {df['lead_hours'].to_list()} != {leads}"
            raise ValueError(msg)
        bad = df.select(pl.col(list(WEATHER_COLS)).is_nan().any()).row(0)
        if any(bad):
            msg = f"{plan} {d} {key}: NaN in channels {bad}"
            raise ValueError(msg)
        out[key] = df
    return out


def sweep(start: date, end: date) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    coords = _coords()
    prog = _load_progress()
    done = {p: set(prog["complete"][p]) for p in PLANS}
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if not all(d.isoformat() in done[p] for p in PLANS)]
    log.info("phase112 extract %s..%s: %d days, %d to fetch, %d points",
             start, end, len(days), len(todo), len(coords))
    buffers: dict[str, dict[str, list[pl.DataFrame]]] = {
        p: {k: [] for k in coords} for p in PLANS}
    t0, n = time.time(), 0
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        for d in todo:
            for plan in PLANS:
                if d.isoformat() in done[plan]:
                    continue
                try:
                    rows = _fetch(plan, d, coords, client)
                except (httpx.HTTPStatusError, RetryError) as e:
                    # the loader's tenacity wrapper re-raises as RetryError, so the
                    # status has to be read off the last attempt's exception
                    inner = (e.last_attempt.exception() if isinstance(e, RetryError)
                             else e)
                    if not (isinstance(inner, httpx.HTTPStatusError)
                            and inner.response.status_code == 404):
                        raise
                    log.warning("%s %s: missing in archive (%s)", plan, d,
                                inner.request.url)
                    prog["missing"][plan].append(d.isoformat())
                    continue
                for key, df in rows.items():
                    buffers[plan][key].append(df)
                prog["complete"][plan].append(d.isoformat())
                done[plan].add(d.isoformat())
                time.sleep(POLITE_SLEEP_S)
            n += 1
            if n % FLUSH_EVERY_DAYS == 0 or d == todo[-1]:
                _flush(buffers)
                PROGRESS.write_text(json.dumps(prog, indent=0))
                rate = (time.time() - t0) / n
                log.info("%s done | %d/%d | %.1f s/day | ETA %.1f h", d, n, len(todo),
                         rate, rate * (len(todo) - n) / 3600)
    _flush(buffers)
    PROGRESS.write_text(json.dumps(prog, indent=0))
    log.info("phase112 extract complete: missing A=%d C=%d",
             len(prog["missing"]["msm_d2_12z"]), len(prog["missing"]["msm_d1_00z"]))


def validate() -> dict:
    prog = _load_progress()
    rep: dict = {"complete": {p: len(v) for p, v in prog["complete"].items()},
                 "missing": prog["missing"], "points": {}}
    for plan in PLANS:
        files = sorted((ROOT / plan).glob("*.parquet"))
        rows = {f.stem: pl.read_parquet(f).height for f in files}
        rep["points"][plan] = {"n_files": len(files),
                               "rows_min": min(rows.values()) if rows else 0,
                               "rows_max": max(rows.values()) if rows else 0}
    print(json.dumps(rep, indent=1))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-07-31")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--root-name", default="jma_gpv_backcast",
                    help="cache directory under data/raw (one per concurrent sweep)")
    a = ap.parse_args()
    _set_root(a.root_name)
    if a.validate:
        validate()
        return 0
    sweep(date.fromisoformat(a.start), date.fromisoformat(a.end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
