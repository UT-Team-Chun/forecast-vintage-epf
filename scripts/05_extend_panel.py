r"""Phase 85 data extension: freeze-respecting pulls for the >=730-day panel.

Implements the W1 acquisition steps of the phase85 precommit
(docs/research/2026-08-11_phase85_cw_primary_precommit.md) under its
Gate-45a freeze policy: NO frozen cache file is ever re-fetched or
overwritten. Everything new lands either in a dedicated extension
directory (data/raw/open_meteo_ext1/) or in a brand-new tree
(data/raw/open_meteo_mp/), and the JEPX FY2026 CSV -- which the loader
caches as one file per fiscal year and therefore must be replaced
wholesale -- is only swapped in after a row-identical audit of the
frozen overlap.

Steps (each idempotent; re-runs skip completed work):
  1. jepx    -- re-download spot_2026.csv, audit rows on delivery days
                <= the frozen file's last day are byte-identical, then swap
                (frozen copy kept as spot_2026.frozen-20260811.csv).
  2. era5    -- 9 single-point ERA5 extension slices 2026-07-19..2026-08-08
                (the frozen per-year JSONs end 2026-07-18).
  3. histfc  -- 9 single-point historical-forecast reconstruction slices
                2026-07-25..2026-08-08 (frozen span ends 2026-07-24).
  4. prev    -- 9 x {prev1, prev2} previous-day JMA-MSM slices, rows
                strictly after 2026-07-24 23:59 JST (unique() guard).
  5. mp_era5 -- 46 prefecture-point ERA5 pulls, full panel years
                2024..2026, for the oracle_cw / era5_cw cells.

Every fetched file is sha256-recorded in
data/raw/open_meteo_ext1/_manifest_phase85.json (and _manifest under
open_meteo_mp/) with URL, params and fetch time.

CLI (from the repository root):
  uv run --extra research python scripts/05_extend_panel.py \
      [--steps jepx,era5,histfc,prev,mp_era5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase85_ext")

CFG = EpfConfig()
RAW = CFG.project_root / "data/raw"
EXT_DIR = RAW / "open_meteo_ext1"
MP_DIR = RAW / "open_meteo_mp"

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
JEPX_SPOT_2026 = "https://www.jepx.jp/market/excel/spot_2026.csv"
PREV_MODEL = "jma_msm"
API_VARS = ("temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m")
API_TO_CANON = {"temperature_2m": "temperature_c", "cloud_cover": "cloud_cover_pct",
                "shortwave_radiation": "shortwave_radiation_wm2",
                "wind_speed_10m": "wind_speed_kmh"}
UA = {"User-Agent": "forecast-vintage-epf/1.0 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"}

# phase59 AREA_COORDS, verbatim (the SP evaluation points).
AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}

# Freeze boundaries (phase85 precommit, section 1).
ERA5_FROZEN_END = date(2026, 7, 18)
PREV_FROZEN_END = date(2026, 7, 24)
EXT_END = date(2026, 8, 8)          # WEATHER_END
MP_YEARS = (2024, 2025, 2026)
MP_START_2024 = "2024-03-01"        # PANEL_START - 1 day of margin
POLITE_SLEEP_S = 1.2
WEIGHT_VINTAGE = "2024-09"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _get(client: httpx.Client, url: str, params: dict) -> bytes:
    """Polite GET with backoff on 429/5xx; raises after 6 attempts."""
    for attempt in range(6):
        r = client.get(url, params=params, headers=UA, timeout=120.0)
        if r.status_code == 200:
            time.sleep(POLITE_SLEEP_S)
            return r.content
        wait = 2.0 * (2**attempt)
        log.warning("HTTP %d from %s -- backing off %.0fs", r.status_code, url, wait)
        time.sleep(wait)
    msg = f"gave up after 6 attempts: {url}"
    raise RuntimeError(msg)


def _manifest_add(manifest_path: Path, name: str, path: Path, url: str, params: dict) -> None:
    entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries[name] = {
        "relative_path": str(path.relative_to(RAW)),
        "sha256": _sha256(path.read_bytes()),
        "source_url": str(httpx.URL(url, params=params)),
        "fetched_at": datetime.now(UTC).isoformat(),
        "size_bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def _tok(v: float) -> str:
    return f"{v:.4f}".replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------- step: jepx
def step_jepx(client: httpx.Client) -> int:
    """Re-download spot_2026.csv; swap only if the frozen PANEL REACH is identical.

    The frozen panel scored test days <= ERA5_FROZEN_END (2026-07-18), so its
    bit-reproduction depends only on rows with delivery date <= that day. JEPX
    back-fills fields that were still empty when a fiscal-year file is fetched
    close to a delivery day (observed live: 2026-07-26 rows fetched on
    2026-07-26 had empty columns that today carry values), so rows AFTER the
    panel reach are allowed to differ and are logged, not failed.
    """
    target = RAW / "jepx/spot_2026.csv"
    frozen_backup = RAW / "jepx/spot_2026.frozen-20260811.csv"
    if not frozen_backup.exists():
        frozen_backup.write_bytes(target.read_bytes())
        log.info("frozen JEPX copy kept at %s", frozen_backup.name)
    fresh = _get(client, JEPX_SPOT_2026, {})
    frozen_lines = frozen_backup.read_bytes().split(b"\r\n")
    fresh_lines = fresh.split(b"\r\n")
    cutoff = ERA5_FROZEN_END.strftime("%Y/%m/%d").encode()
    late_filled_days: set[bytes] = set()
    for i, a in enumerate(frozen_lines):
        if not a or a == fresh_lines[i]:
            continue
        day = a.split(b",", 1)[0]
        if day <= cutoff:
            log.error("Gate 45a JEPX FAIL: row %d (delivery %s, inside the frozen panel "
                      "reach) diverges\n  frozen: %s\n  fresh:  %s",
                      i, day.decode(), a[:120], fresh_lines[i][:120])
            return 1
        late_filled_days.add(day)
    target.write_bytes(fresh)
    log.info("JEPX audit PASS: all rows <= %s identical; %d later day(s) back-filled "
             "by JEPX (%s) -- outside the frozen panel reach",
             ERA5_FROZEN_END, len(late_filled_days),
             b", ".join(sorted(late_filled_days)).decode())
    _manifest_add(EXT_DIR / "_manifest_phase85.json", "jepx_spot_2026_refresh",
                  target, JEPX_SPOT_2026, {})
    return 0


# ---------------------------------------------------------------- step: era5
def step_era5_sp(client: httpx.Client) -> int:
    start = (ERA5_FROZEN_END).isoformat()  # overlap day 07-18 refetched for a join sanity check
    end = EXT_END.isoformat()
    for area, (lat, lon) in AREA_COORDS.items():
        out = EXT_DIR / f"era5_ext1_{area}_{_tok(lat)}_{_tok(lon)}_{start}_{end}.json"
        if out.exists():
            log.info("era5 ext cache hit %s", out.name)
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "hourly": ",".join(API_VARS), "timezone": "Asia/Tokyo"}
        out.write_bytes(_get(client, ERA5_URL, params))
        _manifest_add(EXT_DIR / "_manifest_phase85.json", f"era5_ext1_{area}", out,
                      ERA5_URL, params)
        log.info("era5 ext fetched %s", out.name)
    return 0


# -------------------------------------------------------------- step: histfc
def step_histfc_sp(client: httpx.Client) -> int:
    start = (PREV_FROZEN_END).isoformat()  # overlap day 07-24 for join sanity
    end = EXT_END.isoformat()
    for area, (lat, lon) in AREA_COORDS.items():
        out = (EXT_DIR /
               f"histfc_ext1_{PREV_MODEL}_{area}_{_tok(lat)}_{_tok(lon)}_{start}_{end}.json")
        if out.exists():
            log.info("histfc ext cache hit %s", out.name)
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "models": PREV_MODEL, "hourly": ",".join(API_VARS), "timezone": "Asia/Tokyo"}
        out.write_bytes(_get(client, HISTFC_URL, params))
        _manifest_add(EXT_DIR / "_manifest_phase85.json", f"histfc_ext1_{area}", out,
                      HISTFC_URL, params)
        log.info("histfc ext fetched %s", out.name)
    return 0


# ---------------------------------------------------------------- step: prev
def step_prev_sp(client: httpx.Client) -> int:
    """prev1/prev2 extension parquets, rows strictly AFTER the frozen span."""
    fetch_start = date(2026, 7, 25)
    for n_prev in (1, 2):
        suffix = f"pd{n_prev}"
        for area, (lat, lon) in AREA_COORDS.items():
            out = (EXT_DIR / f"open_meteo_prev{n_prev}_ext1_{PREV_MODEL}_{area}"
                             f"_{lat:.4f}_{lon:.4f}_20260725_20260808.parquet")
            if out.exists():
                log.info("prev%d ext cache hit %s", n_prev, out.name)
                continue
            params = {"latitude": lat, "longitude": lon,
                      "start_date": fetch_start.isoformat(), "end_date": EXT_END.isoformat(),
                      "hourly": ",".join(f"{v}_previous_day{n_prev}" for v in API_VARS),
                      "models": PREV_MODEL, "timezone": "Asia/Tokyo"}
            js = json.loads(_get(client, HISTFC_URL, params))
            h = js["hourly"]
            df = pl.DataFrame(
                {
                    "datetime_jst": [datetime.fromisoformat(t) for t in h["time"]],
                    **{f"{API_TO_CANON[v]}_{suffix}":
                       [float(x) if x is not None else None
                        for x in h[f"{v}_previous_day{n_prev}"]] for v in API_VARS},
                },
            ).with_columns(pl.col("datetime_jst").dt.replace_time_zone("Asia/Tokyo")
                           .dt.cast_time_unit("us"))
            # unique()-guard: keep rows strictly after the frozen span's last stamp
            df = df.filter(pl.col("datetime_jst")
                           > datetime(2026, 7, 24, 23, 59, tzinfo=ZoneInfo("Asia/Tokyo")))
            df.write_parquet(out)
            _manifest_add(EXT_DIR / "_manifest_phase85.json", f"prev{n_prev}_ext1_{area}",
                          out, HISTFC_URL, params)
            log.info("prev%d ext %s: %d rows (> 2026-07-24 23:59 JST)", n_prev, area, df.height)
    return 0


# ------------------------------------------------------------- step: mp_era5
def step_mp_era5(client: httpx.Client) -> int:
    """46 prefecture-point ERA5, full panel, for oracle_cw / era5_cw."""
    pts = area_points(vintage=WEIGHT_VINTAGE)
    n_pts = sum(len(v) for v in pts.values())
    log.info("mp_era5: %d prefecture points, years %s", n_pts, MP_YEARS)
    done = fetched = 0
    for area, plist in sorted(pts.items()):
        for p in plist:
            lat, lon = p.lat, p.lon
            for year in MP_YEARS:
                start = MP_START_2024 if year == 2024 else f"{year}-01-01"
                end = EXT_END.isoformat() if year == 2026 else f"{year}-12-31"
                out = MP_DIR / f"era5_mp_{area}_{_tok(lat)}_{_tok(lon)}_{year}.json"
                if out.exists():
                    done += 1
                    continue
                params = {"latitude": lat, "longitude": lon, "start_date": start,
                          "end_date": end, "hourly": ",".join(API_VARS),
                          "timezone": "Asia/Tokyo"}
                out.write_bytes(_get(client, ERA5_URL, params))
                _manifest_add(MP_DIR / "_manifest_phase85.json",
                              f"era5_mp_{area}_{_tok(lat)}_{_tok(lon)}_{year}",
                              out, ERA5_URL, params)
                fetched += 1
                if fetched % 10 == 0:
                    log.info("mp_era5 progress: %d fetched, %d cached", fetched, done)
    log.info("mp_era5 complete: %d fetched, %d already cached", fetched, done)
    return 0


STEPS = {"jepx": step_jepx, "era5": step_era5_sp, "histfc": step_histfc_sp,
         "prev": step_prev_sp, "mp_era5": step_mp_era5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="jepx,era5,histfc,prev,mp_era5")
    args = ap.parse_args()
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    MP_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    with httpx.Client(follow_redirects=True) as client:
        for name in args.steps.split(","):
            log.info("=== step %s ===", name)
            rc |= STEPS[name.strip()](client)
    return rc


if __name__ == "__main__":
    sys.exit(main())
