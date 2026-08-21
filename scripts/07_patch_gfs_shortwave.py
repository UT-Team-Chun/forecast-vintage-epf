r"""Phase 86 repair: re-fetch ONLY the DSWRF field and fill the SW column.

The first extraction looked up eccodes shortName 'dswrf'; NCEP's GRIB2 tables
expose the field as 'sdswrf', so shortwave arrived as null while the other
channels landed. This one-shot repair re-fetches just the DSWRF byte ranges
(one ranged GET per forecast hour), decodes once per message, extracts all
nine area points, and rewrites the per-area parquets in place.

CLI: uv run python scripts/07_patch_gfs_shortwave.py --plan gfs_d2_12z
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta

import eccodes as ec
import httpx
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase86_swpatch")

CFG = EpfConfig()
OUT_ROOT = CFG.resolved("raw_data_dir") / "gfs_exactrun"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = {"User-Agent": "forecast-vintage-epf/1.0 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"}
AREA_COORDS = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
PLANS = {"gfs_d2_12z": (2, 12, 27, 50), "gfs_d2_18z": (2, 18, 21, 44)}
START, END = date(2024, 3, 1), date(2026, 8, 8)


def fetch_day(plan: str, day: date) -> list[dict]:
    db, ih, f0, f1 = PLANS[plan]
    init = (datetime(day.year, day.month, day.day, tzinfo=UTC)
            - timedelta(days=db) + timedelta(hours=ih))
    rows = []
    with httpx.Client(timeout=60.0) as client:
        for fh in range(f0, f1 + 1):
            key = (f"gfs.{init:%Y%m%d}/{init:%H}/atmos/"
                   f"gfs.t{init:%H}z.pgrb2.0p25.f{fh:03d}")
            try:
                idx = client.get(f"{BUCKET}/{key}.idx", headers=UA)
                idx.raise_for_status()
                lines = idx.text.strip().splitlines()
                offs = [int(x.split(":")[1]) for x in lines]
                rng = None
                for i, ln in enumerate(lines):
                    if ":DSWRF:surface:" in ln:
                        end = offs[i + 1] - 1 if i + 1 < len(offs) else None
                        rng = f"bytes={offs[i]}-" if end is None else f"bytes={offs[i]}-{end}"
                        break
                if rng is None:
                    continue
                r = client.get(f"{BUCKET}/{key}", headers={**UA, "Range": rng})
                r.raise_for_status()
                gid = ec.codes_new_from_message(r.content)
                try:
                    vals = {a: float(ec.codes_grib_find_nearest(gid, la, lo)[0].value)
                            for a, (la, lo) in AREA_COORDS.items()}
                finally:
                    ec.codes_release(gid)
                dt = (init + timedelta(hours=fh)).astimezone()
                rows.extend({"area": a, "datetime_utc": init + timedelta(hours=fh),
                             "sw": v} for a, v in vals.items())
            except httpx.HTTPError as exc:
                log.warning("%s %s f%03d: %s", plan, day, fh, str(exc)[:80])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    t0 = time.time()
    allrows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rows in enumerate(pool.map(lambda d: fetch_day(args.plan, d), days)):
            allrows.extend(rows)
            if (i + 1) % 50 == 0:
                log.info("%d/%d days (%.1f s/day)", i + 1, len(days),
                         (time.time() - t0) / (i + 1))
    sw = (pl.DataFrame(allrows)
          .with_columns(pl.col("datetime_utc").dt.replace_time_zone("UTC")
                        .dt.convert_time_zone("Asia/Tokyo").dt.cast_time_unit("us")
                        .alias("datetime_jst"))
          .select("area", "datetime_jst", "sw")
          .unique(["area", "datetime_jst"]))
    out_dir = OUT_ROOT / args.plan
    for area, (la, lo) in AREA_COORDS.items():
        path = out_dir / f"gfs_{args.plan}_{area}_{la:.4f}_{lo:.4f}.parquet"
        df = pl.read_parquet(path).drop("shortwave_radiation_wm2")
        df = (df.join(sw.filter(pl.col("area") == area).drop("area"),
                      on="datetime_jst", how="left")
                .rename({"sw": "shortwave_radiation_wm2"}))
        n_null = df["shortwave_radiation_wm2"].null_count()
        df.write_parquet(path)
        log.info("%s: %d rows, %d SW nulls remain", area, df.height, n_null)
    return 0


if __name__ == "__main__":
    sys.exit(main())
