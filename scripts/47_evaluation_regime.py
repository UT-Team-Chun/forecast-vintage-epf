r"""Phase 94: what price regime the evaluation window actually is.

WHY
---
A referee-simulation pass observed that every headline number in this paper is
measured on a window with essentially no scarcity pricing, and that Section 2.1
lets a sentence about the FY2022 LNG crisis imply otherwise -- FY2022 is training
data only. The test window is 2024-08-01 to 2026-08-05.

If weather drives the LOWER tail (photovoltaic surplus into the floor) and never
the upper tail (scarcity), then the equivalence result, the MAE improvements and
the skill ratios are all conditioned on that regime, and the paper has to say so
rather than leave a reader to infer coverage it does not have.

This measures the regime so the scoping sentence is provenance-backed rather than
an impression: percentiles, the share of slots at or below the near-floor
threshold, and the share at or above a scarcity level.

CLI (from the repository root):
  uv run --extra research python scripts/47_evaluation_regime.py
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
log = logging.getLogger("phase94")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PANEL = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase94_evaluation_regime.json"

NEAR_FLOOR = 0.5      # the paper's near-floor band
SCARCITY = 50.0       # the level Section 2.1 uses when describing spike-rich eras


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    if not PANEL.exists():
        log.error("missing %s", PANEL)
        return 2
    d = pl.read_parquet(PANEL, columns=["area", "test_day", "y_actual"])
    y = d["y_actual"]
    n = d.height
    near = int((y <= NEAR_FLOOR).sum())
    scarce = int((y >= SCARCITY).sum())
    out = {
        "phase": "94",
        "title": "Price regime of the evaluation window",
        "why": ("every headline number is conditioned on this window; Section 2.1 "
                "cited the FY2022 LNG crisis, which is training data only, and so "
                "implied spike coverage the evaluation era does not have"),
        "window": {"first_test_day": str(d["test_day"].min()),
                   "last_test_day": str(d["test_day"].max()),
                   "n_slots": n, "n_areas": d["area"].n_unique()},
        "percentiles_jpykwh": {f"p{int(q * 100)}": round(float(y.quantile(q)), 2)
                               for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "max_jpykwh": round(float(y.max()), 2),
        "near_floor": {"threshold": NEAR_FLOOR, "n_slots": near,
                       "share_pct": round(100.0 * near / n, 2)},
        "scarcity": {"threshold": SCARCITY, "n_slots": scarce,
                     "share_pct": round(100.0 * scarce / n, 4)},
        "floor_to_scarcity_ratio": round(near / scarce, 1) if scarce else None,
        "reading": (
            f"near-floor slots outnumber slots at or above {SCARCITY:g} JPY/kWh by "
            f"{near / scarce:.0f} to 1. Weather enters this window through "
            f"photovoltaic surplus at the lower tail, essentially never through "
            f"scarcity at the upper tail, so every result here is a "
            f"floor-regime result"
            if scarce else "no scarcity slots at all in the window"),
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("window %s .. %s, %d slots", out["window"]["first_test_day"],
             out["window"]["last_test_day"], n)
    log.info("p95 %.2f  p99 %.2f  max %.2f",
             out["percentiles_jpykwh"]["p95"], out["percentiles_jpykwh"]["p99"],
             out["max_jpykwh"])
    log.info("near-floor %.2f%% vs scarcity %.4f%% -> ratio %s : 1",
             out["near_floor"]["share_pct"], out["scarcity"]["share_pct"],
             out["floor_to_scarcity_ratio"])
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
