r"""Phase 86 (extraction): run-pinned NCEP GFS tiers from the AWS open-data archive.

Gate G-GFS (docs/research/2026-08-11_phase86_gfs_provider_precommit.md) PASSED on
2026-08-11: s3://noaa-gfs-bdp-pds serves every probed cycle 2024-03..2026-08 with
a contemporaneous ``LastModified`` (~ init + 3.6-3.9 h -- a measured dissemination
proxy, same construction as phase72's RISH audit) and per-file ``.idx`` indexes
that allow byte-range extraction of single fields.

Plans (delivery day = JST calendar day, hourly valid times):
    gfs_d2_12z   init D-2 12 UTC, FH27-50  -- same init clock as the JMA headline
                 tier; disseminated ~ D-1 00:54 JST, ~9 h pre-gate.
    gfs_d2_18z   init D-2 18 UTC, FH21-44  -- the freshest GFS run whose upload
                 (~ D-1 06:42 JST) precedes the 10:00 gate.

Fields (canonical mapping mirrors DEFAULT_WEATHER_COLS):
    DSWRF surface        -> shortwave_radiation_wm2
    TMP 2 m              -> temperature_c        (K - 273.15)
    UGRD/VGRD 10 m       -> wind_speed_kmh       (sqrt(u^2+v^2) * 3.6)
    TCDC entire atmosphere -> cloud_cover_pct

Output mirrors the exactrun cache schema so the scoring step can reuse the
phase85 attach machinery verbatim: one parquet per (plan, area) with
``datetime_jst`` + the four canonical columns + ``issuance_time_utc`` +
``lead_hours``, under data/raw/gfs_exactrun/<plan>/. A per-plan
``_progress.json`` makes re-runs incremental; ``_dissemination.parquet``
records each cycle's S3 LastModified (the dissemination-time audit).

CLI (from the repository root):
  uv run python scripts/06_extract_gfs.py \
      --plan gfs_d2_12z [--start 2024-03-01] [--end 2026-08-08] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import eccodes as ec  # optional geo dependency (declared in the alg[geo] extra)
import httpx
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase86_gfs")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
OUT_ROOT = RAW / "gfs_exactrun"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = {"User-Agent": "forecast-vintage-epf/1.0 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"}

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
# idx substrings -> canonical output handling
IDX_FIELDS = (
    ":DSWRF:surface:",
    ":TMP:2 m above ground:",
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":TCDC:entire atmosphere:",
)
PLANS: dict[str, dict] = {
    "gfs_d2_12z": {"days_before": 2, "init_h": 12, "fh_first": 27, "fh_last": 50},
    "gfs_d2_18z": {"days_before": 2, "init_h": 18, "fh_first": 21, "fh_last": 44},
}


def _key(init: datetime, fh: int) -> str:
    return (f"gfs.{init:%Y%m%d}/{init:%H}/atmos/"
            f"gfs.t{init:%H}z.pgrb2.0p25.f{fh:03d}")


def _idx_ranges(idx_text: str) -> list[tuple[int, int | None]]:
    """Byte ranges of the five wanted fields from a .idx listing."""
    lines = idx_text.strip().splitlines()
    offsets = [int(ln.split(":")[1]) for ln in lines]
    out: list[tuple[int, int | None]] = []
    for i, ln in enumerate(lines):
        if any(tag in ln for tag in IDX_FIELDS):
            end = offsets[i + 1] - 1 if i + 1 < len(offsets) else None
            out.append((offsets[i], end))
    return out


def _decode(msg_bytes: bytes, points: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """One GRIB2 message -> {shortName: {area: value}} at nearest grid points."""
    gid = ec.codes_new_from_message(msg_bytes)
    try:
        sn = ec.codes_get(gid, "shortName")
        vals = {}
        for area, (lat, lon) in points.items():
            near = ec.codes_grib_find_nearest(gid, lat, lon)[0]
            vals[area] = float(near.value)
        return {sn: vals}
    finally:
        ec.codes_release(gid)


def _fetch_fh(client: httpx.Client, init: datetime, fh: int) -> dict[str, dict]:
    """All five fields of one forecast-hour file, via ranged GETs."""
    key = _key(init, fh)
    idx = client.get(f"{BUCKET}/{key}.idx", headers=UA)
    idx.raise_for_status()
    merged: dict[str, dict] = {}
    for start, end in _idx_ranges(idx.text):
        rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
        r = client.get(f"{BUCKET}/{key}", headers={**UA, "Range": rng})
        r.raise_for_status()
        merged.update(_decode(r.content, AREA_COORDS))
    return merged


def _head_last_modified(client: httpx.Client, init: datetime, fh: int) -> str | None:
    r = client.head(f"{BUCKET}/{_key(init, fh)}", headers=UA)
    return r.headers.get("Last-Modified") if r.status_code == 200 else None


def _day_rows(client: httpx.Client, plan: str, day: date) -> tuple[list[dict], dict]:
    """All hourly rows of one delivery day, plus the dissemination record."""
    spec = PLANS[plan]
    init = (datetime(day.year, day.month, day.day, tzinfo=UTC)
            - timedelta(days=spec["days_before"]) + timedelta(hours=spec["init_h"]))
    rows = []
    for fh in range(spec["fh_first"], spec["fh_last"] + 1):
        fields = _fetch_fh(client, init, fh)
        valid_utc = init + timedelta(hours=fh)
        dt_jst = valid_utc.astimezone(JST)
        u10, v10 = fields.get("10u", {}), fields.get("10v", {})
        sw = fields.get("sdswrf", fields.get("dswrf", {}))
        for area in AREA_COORDS:
            wind_ms = ((u10.get(area, float("nan")) ** 2
                        + v10.get(area, float("nan")) ** 2) ** 0.5)
            rows.append({
                "area": area, "datetime_jst": dt_jst,
                "shortwave_radiation_wm2": sw.get(area),
                "temperature_c": (fields.get("2t", {}).get(area, float("nan"))
                                  - 273.15),
                "wind_speed_kmh": wind_ms * 3.6,
                "cloud_cover_pct": fields.get("tcc", {}).get(area),
                "issuance_time_utc": init, "lead_hours": fh,
            })
    lm = _head_last_modified(client, init, spec["fh_last"])
    dissem = {"delivery_day": day.isoformat(), "plan": plan,
              "init_utc": init.isoformat(), "last_modified": lm}
    return rows, dissem


def run_plan(plan: str, start: date, end: date, workers: int) -> int:
    out_dir = OUT_ROOT / plan
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "_progress.json"
    prog: dict = json.loads(prog_path.read_text()) if prog_path.exists() else {
        "plan": plan, "days_done": [], "dissemination": []}
    done = set(prog["days_done"])
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.isoformat() not in done]
    log.info("%s: %d days requested, %d already complete, %d to fetch",
             plan, len(days), len(done), len(todo))
    if not todo:
        _finalise(plan, out_dir, prog)
        return 0

    t0 = time.time()
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)

    def work(day: date) -> tuple[date, list[dict] | None, dict | None, str | None]:
        try:
            with httpx.Client(timeout=60.0) as client:
                rows, dissem = _day_rows(client, plan, day)
            return day, rows, dissem, None
        except Exception as exc:
            return day, None, None, f"{type(exc).__name__}: {exc}"[:200]

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for day, rows, dissem, err in pool.map(work, todo):
            if err:
                n_fail += 1
                log.warning("%s %s FAILED: %s", plan, day, err)
                continue
            pl.DataFrame(rows).write_parquet(parts_dir / f"{day.isoformat()}.parquet")
            prog["days_done"].append(day.isoformat())
            prog["dissemination"].append(dissem)
            n_ok += 1
            if n_ok % 25 == 0:
                prog_path.write_text(json.dumps(prog))
                rate = (time.time() - t0) / n_ok
                log.info("%s: %d/%d days (%.1f s/day, ~%.0f min left)", plan,
                         n_ok, len(todo), rate, rate * (len(todo) - n_ok) / 60)
    prog_path.write_text(json.dumps(prog))
    log.info("%s sweep: %d ok, %d failed", plan, n_ok, n_fail)
    _finalise(plan, out_dir, prog)
    return 0 if n_fail == 0 else 1


def _finalise(plan: str, out_dir: Path, prog: dict) -> None:
    """Assemble per-area parquets + the dissemination audit from day parts."""
    parts = sorted((out_dir / "_parts").glob("*.parquet"))
    if not parts:
        return
    allrows = pl.concat([pl.read_parquet(p) for p in parts])
    for area in AREA_COORDS:
        lat, lon = AREA_COORDS[area]
        (allrows.filter(pl.col("area") == area)
         .drop("area")
         .unique("datetime_jst").sort("datetime_jst")
         .write_parquet(out_dir / f"gfs_{plan}_{area}_{lat:.4f}_{lon:.4f}.parquet"))
    dis = pl.DataFrame(prog["dissemination"]).unique("delivery_day").sort("delivery_day")
    dis.write_parquet(out_dir / "_dissemination.parquet")
    log.info("%s finalised: %d hourly rows x 9 areas, %d dissemination records",
             plan, allrows.height // 9, dis.height)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2026-08-08")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    return run_plan(args.plan, date.fromisoformat(args.start),
                    date.fromisoformat(args.end), args.workers)


if __name__ == "__main__":
    sys.exit(main())
