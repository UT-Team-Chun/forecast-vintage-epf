r"""Phase 89b: the PRICE target on the physical common panel.

WHY
---
Figure 5 was redesigned so that its two panels sit on one sample -- that was the
answer to the reviewer's objection that the price and physical targets were
scored on different panels. The caption says both panels are "scored on the same
696 delivery days and eight areas". Panel (b) is: it reads phase89's
common-panel column. Panel (a) was NOT: it read phase85's pooled ladder, which
is 732 days and nine areas. The figure asserted a restriction it had not
applied, and so did its caption.

This computes the price-target improvement on exactly the panel phase89 defines
-- the intersection of the physical and price day sets, restricted to the eight
areas whose operators publish photovoltaic actuals -- so panel (a) can be what
the caption says it is.

Nothing here is a new hypothesis or a gate: it is an existing metric recomputed
on a stated sample, with phase85's estimator, block length and seed unchanged so
the two versions are comparable. The full-panel figures stay in the ladder table
and the text; this file exists so the FIGURE stops overstating its own sample.

CLI (from the repository root):
  uv run --extra research python scripts/52_price_on_common_panel.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase89b")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PRICE = RUN / "phase85_exact_ladder_per_slot.parquet"
PHYS = RUN / "phase89_physical_per_slot.parquet"
P89_JSON = RUN / "phase89_physical_target.json"
OUT_JSON = RUN / "phase89b_price_on_common_panel.json"

# phase85's inference settings, unchanged so the two panels are comparable.
SEED, N_BOOT, BOOT_BLOCK = 2026, 10_000, 7
TIERS = ("era5_cw", "msm_d2_12z_cw", "msm_gate_composite_cw", "msm_d1_00z_cw",
         "oracle_cw")
BASELINE = "price_only"


def _j(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _block_boot_ci(d: np.ndarray) -> tuple[float, float]:
    """phase85's circular block bootstrap, same seed and block length."""
    n = len(d)
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    idx = idx.reshape(N_BOOT, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    for p in (PRICE, PHYS):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    phys = pl.read_parquet(PHYS, columns=["area", "day"])
    phys_days = set(phys["day"].cast(pl.Utf8).unique().to_list())
    phys_areas = sorted(phys["area"].unique().to_list())

    cols = ["area", "test_day", "y_actual", f"y_pred_{BASELINE}"] + [
        f"y_pred_{t}" for t in TIERS]
    price = pl.read_parquet(PRICE, columns=cols)
    full_days = price["test_day"].n_unique()
    full_areas = price["area"].n_unique()

    common = price.filter(pl.col("test_day").is_in(list(phys_days))
                          & pl.col("area").is_in(phys_areas))
    n_days = common["test_day"].n_unique()
    log.info("full price panel: %d days x %d areas | common panel: %d days x "
             "%d areas (%d slots)", full_days, full_areas, n_days,
             common["area"].n_unique(), common.height)

    # Cross-check against phase89's own count, so a silent divergence in how the
    # intersection is taken cannot pass: the figure's two panels must agree on
    # what "the common panel" means.
    ref = json.loads(P89_JSON.read_text())["gates"][
        "gate51_physical_gain_crossfit_common"]["n_days"]
    if n_days != ref:
        log.error("common-panel day count %d does not match phase89's %d",
                  n_days, ref)
        return 3

    out: dict = {
        "phase": "89b",
        "title": "Price-target improvement on the physical common panel",
        "why": ("Figure 5's panel (a) read the FULL ladder (732 days, 9 areas) "
                "while its panel (b) read the common panel (696 days, 8 areas), "
                "and the caption claimed both were the common panel"),
        "panel": {"days": n_days, "areas": common["area"].n_unique(),
                  "slots": common.height,
                  "full_panel_days": full_days, "full_panel_areas": full_areas,
                  "areas_list": phys_areas},
        "inference": {"seed": SEED, "n_boot": N_BOOT,
                      "block_days": BOOT_BLOCK,
                      "note": "phase85's estimator and settings, unchanged"},
        "tiers": {},
    }

    per_day = (common.with_columns(
        (pl.col(f"y_pred_{BASELINE}") - pl.col("y_actual")).abs().alias("ae_base"),
        *[(pl.col(f"y_pred_{t}") - pl.col("y_actual")).abs().alias(f"ae_{t}")
          for t in TIERS])
        .group_by("area", "test_day")
        .agg(pl.col("ae_base").mean().alias("mae_base"),
             *[pl.col(f"ae_{t}").mean().alias(f"mae_{t}") for t in TIERS]))

    for t in TIERS:
        d = np.asarray(
            (per_day.with_columns(
                (pl.col("mae_base") - pl.col(f"mae_{t}")).alias("imp"))
             .group_by("test_day").agg(pl.col("imp").mean())
             .sort("test_day")["imp"]), dtype=float)
        lo, hi = _block_boot_ci(d)
        out["tiers"][t] = {"improvement": _j(float(d.mean())),
                           "ci95": [_j(lo), _j(hi)], "n_days": len(d)}
        log.info("%-24s %+.4f [%.4f, %.4f]", t, d.mean(), lo, hi)

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha()}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
