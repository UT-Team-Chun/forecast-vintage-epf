r"""Phase 93: mean and dispersion of the headline Kyushu cell, both fold designs.

WHY
---
\SOTA (1.816 +- 0.92) and \SOTAweekly (1.573 +- 0.75) are printed in the
supplement's weekly-versus-daily table. Both were EXEMPT from the drift guard on
the stated ground that they are "benchmark MAE reported by the cited paper
[Lago et al.], not computed here".

That is false. They are this project's own Kyushu M1/XGB+weather results -- the
macros.tex comments say so -- and the table that prints them labels them
"Headline cell (Kyushu M1/XGB+weather)". So the manuscript is right and the
exemption was wrong, which left two printed numbers unchecked behind a reason
that would not survive being read.

The weekly pair is already in phase11_summary.json. The daily pair is not in any
JSON: only the per-fold parquet holds it. This records both so the guard can bind
all four numbers.

Nothing here is a new hypothesis or a gate: it is mean and standard deviation of
a committed per-fold table.

CLI (from the repository root):
  uv run --extra research python scripts/44_headline_cell_dispersion.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase93")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
DAILY_FOLDS = RUN / "phase23_daily_per_fold_mae.parquet"
WEEKLY_JSON = RUN / "phase11_summary.json"
OUT_JSON = RUN / "phase93_headline_cell_dispersion.json"

DAILY_CELL = "kyushu/flat/xgboost/single_weather"
WEEKLY_CELL = "flat/xgboost/single_weather"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    if not DAILY_FOLDS.exists():
        log.error("missing %s", DAILY_FOLDS)
        return 2
    folds = pl.read_parquet(DAILY_FOLDS)
    daily = folds.filter(pl.col("cell") == DAILY_CELL)["mae"]
    if daily.len() == 0:
        log.error("cell %s absent from %s", DAILY_CELL, DAILY_FOLDS.name)
        return 3

    weekly_rows = json.loads(WEEKLY_JSON.read_text())
    weekly = next((r for r in weekly_rows if r.get("cell") == WEEKLY_CELL), None)
    if weekly is None:
        log.error("cell %s absent from %s", WEEKLY_CELL, WEEKLY_JSON.name)
        return 4

    out = {
        "phase": "93",
        "title": "Headline Kyushu cell: mean and dispersion under both fold designs",
        "why": ("\\SOTA and \\SOTAweekly were exempt from the guard as literature "
                "values from Lago et al. They are this project's own Kyushu "
                "M1/XGB+weather results, so the exemption rested on a false "
                "premise and left two printed numbers unchecked"),
        "cell": "Kyushu, flat (M1) / XGBoost / single-point weather",
        "daily_rolling": {
            "source": DAILY_FOLDS.name,
            "cell": DAILY_CELL,
            "n_folds": int(daily.len()),
            "mae_mean": round(float(daily.mean()), 6),
            # sample standard deviation across folds, which is what the +- prints
            "mae_sd": round(float(daily.std()), 6),
        },
        "weekly_anchor": {
            "source": WEEKLY_JSON.name,
            "cell": WEEKLY_CELL,
            "n_folds": int(weekly["n_folds"]),
            "mae_mean": round(float(weekly["mae_mean"]), 6),
            "mae_sd": round(float(weekly["mae_std"]), 6),
        },
        "reading": ("the daily-rolling design gives a higher mean MAE than the "
                    "weekly anchor on the same cell; the supplement's table "
                    "reports both, and neither is a literature benchmark"),
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("daily  n=%d mean=%.4f sd=%.4f", out["daily_rolling"]["n_folds"],
             out["daily_rolling"]["mae_mean"], out["daily_rolling"]["mae_sd"])
    log.info("weekly n=%d mean=%.4f sd=%.4f", out["weekly_anchor"]["n_folds"],
             out["weekly_anchor"]["mae_mean"], out["weekly_anchor"]["mae_sd"])
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
