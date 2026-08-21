r"""Phase 95: how many INDEPENDENT areas the paper's "nine of nine" rests on.

WHY
---
"Positive in all nine areas" is this paper's principal robustness currency: it
appears in the abstract, in the vintage-ladder table, and behind the floor result.
A referee-simulation pass objected that nine JEPX areas are not nine replications,
because interconnection capacity is allocated through the spot auction and area
prices are identical whenever no interface binds. If the areas are near-copies,
"nine of nine" is closer to one observation repeated than to nine.

The objection is right to demand a number, and there are two different numbers:

1. INSTITUTIONAL coupling -- how often area prices literally coincide. This is the
   mechanism the referee names.
2. STATISTICAL coupling of the quantity the claims are actually made about -- the
   per-area daily improvement series. "Nine of nine areas improved" is a statement
   about nine such series, so the effective sample size is theirs, not the prices'.

Both are reported. The effective number of independent series uses the
participation ratio of the correlation matrix's eigenvalues,
n_eff = (sum lambda)^2 / sum lambda^2, which is 9 for uncorrelated areas and 1 for
perfectly correlated ones.

CLI (from the repository root):
  uv run --extra research python scripts/48_area_independence.py
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
log = logging.getLogger("phase95")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PRICES = RUN / "phase22c" / "_jepx_inputs.parquet"
LADDER = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase95_area_independence.json"

HEADLINE = "msm_d2_12z_cw"
BASELINE = "price_only"
NEAR_FLOOR = 0.5


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _n_eff(corr: np.ndarray) -> float:
    """Participation ratio of the eigenvalue spectrum: 9 if independent, 1 if identical."""
    lam = np.linalg.eigvalsh(corr)
    lam = np.clip(lam, 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def main() -> int:
    for p in (PRICES, LADDER):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    # ---- 1. institutional coupling: how often do area prices coincide? ----
    pr = pl.read_parquet(PRICES)
    wide = pr.pivot(on="area", index="datetime_jst", values="price_yen_kwh").drop_nulls()
    areas = [c for c in wide.columns if c != "datetime_jst"]
    mat = wide.select(areas).to_numpy()
    n_slots = mat.shape[0]
    # a slot is "unsplit" when every area clears at the same price
    unsplit = int((np.ptp(mat, axis=1) < 1e-9).sum())
    med = np.median(mat, axis=1, keepdims=True)
    in_main_block = {a: float(100.0 * np.mean(np.abs(mat[:, i] - med[:, 0]) < 1e-9))
                     for i, a in enumerate(areas)}
    # near-floor slots are the regime the floor claims live in
    floor_rows = (mat <= NEAR_FLOOR).any(axis=1)
    unsplit_floor = (int((np.ptp(mat[floor_rows], axis=1) < 1e-9).sum())
                     if floor_rows.any() else 0)

    # ---- 2. statistical coupling of the quantity the claims are about ----
    lad = pl.read_parquet(LADDER, columns=["area", "test_day", "y_actual",
                                           f"y_pred_{BASELINE}", f"y_pred_{HEADLINE}"])
    per_day = (lad.with_columns(
        ((pl.col(f"y_pred_{BASELINE}") - pl.col("y_actual")).abs()
         - (pl.col(f"y_pred_{HEADLINE}") - pl.col("y_actual")).abs()).alias("imp"))
        .group_by("area", "test_day").agg(pl.col("imp").mean()))
    imp_wide = (per_day.pivot(on="area", index="test_day", values="imp")
                .drop_nulls().sort("test_day"))
    imp_areas = [c for c in imp_wide.columns if c != "test_day"]
    X = imp_wide.select(imp_areas).to_numpy()
    corr_imp = np.corrcoef(X, rowvar=False)
    # and of the raw prices, for comparison
    corr_price = np.corrcoef(mat, rowvar=False)

    iu = np.triu_indices(len(imp_areas), 1)
    out = {
        "phase": "95",
        "title": "Effective independence of the nine JEPX areas",
        "why": ("\"positive in all nine areas\" is the paper's principal robustness "
                "currency; interconnection capacity is allocated through the spot "
                "auction, so area prices coincide whenever no interface binds and "
                "nine areas may not be nine replications"),
        "institutional_coupling": {
            "n_slots": n_slots,
            "n_areas": len(areas),
            "slots_all_areas_same_price_pct": round(100.0 * unsplit / n_slots, 2),
            "share_in_main_block_pct": {a: round(v, 1)
                                        for a, v in in_main_block.items()},
            "near_floor_slots": int(floor_rows.sum()),
            "near_floor_slots_unsplit_pct": (
                round(100.0 * unsplit_floor / int(floor_rows.sum()), 2)
                if floor_rows.any() else None),
            "reading": ("the share of slots on which every area clears at one "
                        "price. High values mean the areas are one market most of "
                        "the time and the per-area counts are not independent"),
        },
        "statistical_coupling": {
            "quantity": ("per-area daily improvement of the headline pre-gate tier "
                         "over the price-only baseline -- the series the "
                         "'nine of nine' counts are made about"),
            "n_days": int(X.shape[0]),
            "mean_pairwise_corr": round(float(corr_imp[iu].mean()), 3),
            "min_pairwise_corr": round(float(corr_imp[iu].min()), 3),
            "max_pairwise_corr": round(float(corr_imp[iu].max()), 3),
            "effective_n_areas": round(_n_eff(corr_imp), 2),
            "effective_n_areas_prices": round(_n_eff(corr_price), 2),
            "reading": ("the effective number of independent improvement series, "
                        "by the participation ratio of the correlation matrix's "
                        "eigenvalues. Nine means independent, one means identical"),
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    ic, sc = out["institutional_coupling"], out["statistical_coupling"]
    log.info("prices identical across all areas in %.2f%% of %d slots "
             "(near-floor slots: %s%%)", ic["slots_all_areas_same_price_pct"],
             n_slots, ic["near_floor_slots_unsplit_pct"])
    log.info("improvement series: mean pairwise r = %.3f, effective n = %.2f of %d",
             sc["mean_pairwise_corr"], sc["effective_n_areas"], len(imp_areas))
    log.info("prices themselves: effective n = %.2f", sc["effective_n_areas_prices"])
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
