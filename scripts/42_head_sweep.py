r"""Phase 84 (part 1): measured dissemination timestamps for plans E and F.

phase72 measured the RISH mirror's ``Last-Modified`` for the three runs the
paper's tiers use (D-2 12Z, D-2 21Z, D-1 00Z). The cycle curve also plots the
D-3 12Z and D-2 18Z rungs, whose dissemination times were until now taken from
the JMBSC specification (init + 2 h 30 m) rather than measured. This sweep
closes that gap so phase84's buffer analysis can place EVERY rung of the cycle
curve at a measured dissemination time.

Mechanics are phase72's verbatim: one HTTP HEAD per (delivery day, run,
FH file), 0.5 s politeness sleep, resumable JSON checkpoint, no bodies
downloaded. Output is a separate parquet so phase72's frozen artefact is
untouched; phase84's analysis concatenates the two.

CLI (from the repository root):
  uv run python scripts/42_head_sweep.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase84_head")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase84_head_checkpoint.json"
OUT_PARQUET = RUN / "phase84_head_ef_per_file.parquet"

BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
START, END = date(2024, 3, 1), date(2026, 8, 8)
SLEEP_S = 0.5
UA = {"User-Agent": "forecast-vintage-epf/1.0 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"}

# (run label, days before delivery, init hour UTC, FH files)
RUNS: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    ("msm_d3_12z", 3, 12, ("FH52-78",)),
    ("msm_d2_18z", 2, 18, ("FH16-33", "FH34-39")),
)


def main() -> int:
    ckpt: dict = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    n_days = (END - START).days + 1
    todo = []
    for i in range(n_days):
        day = START + timedelta(days=i)
        for label, db, ih, fh_files in RUNS:
            init = datetime(day.year, day.month, day.day, tzinfo=UTC) \
                - timedelta(days=db) + timedelta(hours=ih)
            for fh in fh_files:
                key = f"{day.isoformat()}|{label}|{fh}"
                if key in ckpt:
                    continue
                url = (f"{BASE}/{init:%Y/%m/%d}/Z__C_RJTD_{init:%Y%m%d%H}0000_MSM_GPV"
                       f"_Rjp_Lsurf_{fh}_grib2.bin")
                todo.append((key, day, label, fh, init, url))
    log.info("sweep E/F %s..%s: %d records total, %d to fetch",
             START, END, n_days * 3, len(todo))
    with httpx.Client(timeout=30.0) as client:
        for i, (key, day, label, fh, init, url) in enumerate(todo):
            try:
                r = client.head(url, headers=UA)
                lm = r.headers.get("Last-Modified")
                ckpt[key] = {"delivery_day": day.isoformat(), "run": label, "fh": fh,
                             "status": r.status_code,
                             "init_utc": init.isoformat(),
                             "last_modified": lm}
            except httpx.HTTPError as exc:
                ckpt[key] = {"delivery_day": day.isoformat(), "run": label, "fh": fh,
                             "status": -1, "init_utc": init.isoformat(),
                             "last_modified": None, "error": str(exc)[:120]}
            time.sleep(SLEEP_S)
            if (i + 1) % 200 == 0:
                CKPT.write_text(json.dumps(ckpt))
                log.info("progress %d/%d", i + 1, len(todo))
    CKPT.write_text(json.dumps(ckpt))

    rows = []
    for rec in ckpt.values():
        lm = rec.get("last_modified")
        lag_min = None
        if lm:
            lm_dt = datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=UTC)
            lag_min = (lm_dt - datetime.fromisoformat(rec["init_utc"])).total_seconds() / 60.0
        rows.append({**{k: rec.get(k) for k in
                        ("delivery_day", "run", "fh", "status", "init_utc")},
                     "last_modified_utc": lm, "lag_minutes": lag_min})
    df = pl.DataFrame(rows).sort("delivery_day", "run", "fh")
    df.write_parquet(OUT_PARQUET)
    ok = df.filter(pl.col("status") == 200)
    log.info("wrote %s: %d records, %d ok", OUT_PARQUET.name, df.height, ok.height)
    for label, _, _, _ in RUNS:
        sub = ok.filter(pl.col("run") == label)["lag_minutes"]
        if sub.len():
            log.info("  %s lag min: median=%.2f p95=%.2f max=%.2f",
                     label, sub.median(), sub.quantile(0.95), sub.max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
