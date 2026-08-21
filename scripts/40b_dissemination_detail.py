r"""Phase 98: the dissemination audit in full, and a test of the poll-grid objection.

WHY
---
A referee-simulation pass called the dissemination measurement this paper's
constitutive contribution and its least documented one: the audit existed as prose
in Section 2.2 and two figure captions, the supplementary section it is cited to
carried no subsection on it, only a median and a worst case were reported, and the
raw capture was not said to be deposited. The same referee raised a specific
alternative explanation -- the archive is a third-party academic mirror, so its
file-write times "are equally consistent with a fixed polling cadence" rather than
with dissemination.

That objection is testable, and this phase tests it. If the mirror wrote on a poll
schedule, the lags would snap to multiples of the poll interval: few distinct
values, evenly spaced by the interval. If they track a real event, they vary at the
resolution the filesystem records.

Everything here is derived from the committed per-file capture. Nothing touches the
network, so the numbers are reproducible from the replication package alone.

CLI (from the repository root):
  uv run python scripts/40b_dissemination_detail.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase98")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PER_FILE = RUN / "phase72_dissemination_audit_per_file.parquet"
OUT_JSON = RUN / "phase98_dissemination_detail.json"

QUANTILES = (0.0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    if not PER_FILE.exists():
        log.error("missing %s", PER_FILE)
        return 2
    d = pl.read_parquet(PER_FILE)
    lag_all = d["lag_minutes"].to_numpy().astype(float)

    per_cycle = {}
    for run in sorted(d["run"].unique().to_list()):
        sub = d.filter(pl.col("run") == run)
        lag = sub["lag_minutes"].to_numpy().astype(float)
        per_cycle[run] = {
            "n_files": int(sub.height),
            "n_delivery_days": int(sub["delivery_day"].n_unique()),
            # "fh" is a label such as "FH00-15", not an integer
            "forecast_hour_labels": sorted(sub["fh"].unique().to_list())[:6],
            "lag_minutes": {f"q{int(q * 100)}": round(float(np.quantile(lag, q)), 3)
                            for q in QUANTILES},
            "n_distinct_lags": int(len(np.unique(np.round(lag, 4)))),
        }

    # ---- the poll-grid test ----
    uniq = np.sort(np.unique(np.round(lag_all, 4)))
    gaps = np.diff(uniq)
    frac = lag_all - np.floor(lag_all)
    med = float(np.median(lag_all))
    quant = {
        "hypothesis": ("if the mirror wrote on a fixed polling cadence, lags would "
                       "snap to multiples of the poll interval: few distinct "
                       "values, evenly spaced by that interval"),
        "n_files": int(len(lag_all)),
        "n_distinct_lag_values": int(len(uniq)),
        "distinct_fractional_minute_values": int(len(np.unique(np.round(frac, 3)))),
        "median_gap_between_distinct_lags_seconds": round(float(np.median(gaps) * 60), 1),
        "share_within_one_minute_of_median_pct": round(
            100.0 * float(np.mean(np.abs(lag_all - med) < 1.0)), 2),
        "verdict": None,   # set below
    }
    # A poll grid coarse enough to matter (a minute or more) would leave the
    # fractional parts nearly constant and the gaps at the interval.
    grid_like = (quant["distinct_fractional_minute_values"] <= 3
                 or quant["median_gap_between_distinct_lags_seconds"] >= 30.0)
    quant["verdict"] = (
        "consistent with a poll grid" if grid_like else
        "NOT consistent with a poll grid: the lags vary at second resolution "
        "within a band about a minute wide, which is what a real completion time "
        "recorded by a filesystem looks like and not what a poll schedule "
        "produces")

    out = {
        "phase": "98",
        "title": "Dissemination audit in full, and the poll-grid test",
        "why": ("the audit was the paper's least documented contribution, reported "
                "as a median and a worst case, and a referee offered a poll "
                "schedule as an equally consistent explanation"),
        "source": {"file": PER_FILE.name,
                   "note": "committed per-file capture: one row per archived file, "
                           "with its run, forecast hour, initialisation time, "
                           "mirror write time and the derived lag"},
        "totals": {"n_files": int(d.height),
                   "n_delivery_days": int(d["delivery_day"].n_unique()),
                   "n_cycles": int(d["run"].n_unique()),
                   "status_values": sorted(d["status"].unique().to_list())
                   if "status" in d.columns else None},
        "per_cycle": per_cycle,
        "poll_grid_test": quant,
        "what_the_timestamp_is": (
            "the mirror's file-write time, an UPPER BOUND on true dissemination "
            "completion. Every pre-gate classification is therefore conservative: "
            "a mirror write before the gate implies the feed delivered before the "
            "gate. The post-gate classification does not rest on this measurement "
            "-- the published specification is its source, and these timestamps "
            "corroborate it"),
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    for run, v in per_cycle.items():
        q = v["lag_minutes"]
        log.info("%-14s %4d files over %3d days, lag q0/q50/q99/q100 "
                 "= %.2f / %.2f / %.2f / %.2f", run, v["n_files"],
                 v["n_delivery_days"], q["q0"], q["q50"], q["q99"], q["q100"])
    log.info("poll-grid test: %d distinct lags, %d distinct fractional minutes, "
             "median gap %.1f s -> %s", quant["n_distinct_lag_values"],
             quant["distinct_fractional_minute_values"],
             quant["median_gap_between_distinct_lags_seconds"],
             quant["verdict"][:60])
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
