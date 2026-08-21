r"""Phase 82: economically weighted metrics, relative displays, and slices.

Precommit: docs/research/2026-08-11_phase82_weighted_metrics_precommit.md.
Equal-weighting nine areas gives Tokyo and Hokkaido the same voice. This phase
re-pools the CW-primary headline contrasts under demand weights and PV-capacity
weights, adds relative displays (percent of baseline MAE, rMAE), and slices by
daylight/midday/floor/season. Descriptive robustness -- no gate; the fixed
reporting rule is that any sign flip under any weighting is reported
prominently and the abstract's claims are scoped to what survives all of them.
JEPX per-area traded volume is not carried in our spot cache, so the
volume-weighted variant is reported as unavailable rather than proxied.

CLI (from the repository root):
  uv run python scripts/61_weighted_metrics.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import pv_gw_by_area

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase82")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PANEL = RUN / "phase85_exact_ladder_per_slot.parquet"
OCCTO = RUN / "phase25.3_occto_generation.parquet"
OUT_JSON = RUN / "phase82_weighted_metrics.json"
PRECOMMIT = "docs/research/2026-08-11_phase82_weighted_metrics_precommit.md"

SEED, N_BOOT, BOOT_BLOCK = 2026, 10_000, 7
TIERS = ("msm_d2_12z_cw", "msm_gate_composite_cw", "msm_d1_00z_cw",
         "oracle_cw", "era5_cw")
HEADLINE, POST = "msm_d2_12z_cw", "msm_d1_00z_cw"
NEAR_FLOOR = 0.5
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SLICES = {"all": None, "daylight_6_18": (6, 18), "midday_9_15": (9, 15)}


def _j(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    means = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _weights(areas: list[str]) -> dict[str, dict[str, float]]:
    pv = pv_gw_by_area()
    occ = pl.read_parquet(OCCTO)
    dem = {r["area"]: r["demand_mw"]
           for r in occ.group_by("area").agg(pl.col("demand_mw").mean()).to_dicts()}
    out = {"unweighted": dict.fromkeys(areas, 1.0),
           "demand": {a: dem.get(a, 0.0) for a in areas},
           "pv_capacity": {a: pv.get(a, 0.0) for a in areas}}
    for k, w in out.items():
        tot = sum(w.values())
        out[k] = {a: v / tot for a, v in w.items()}
        missing = [a for a in areas if out[k][a] == 0.0]
        if missing:
            log.warning("weighting %s has zero weight for %s", k, missing)
    return out


def _pooled(per_day: pl.DataFrame, cell: str, w: dict[str, float]) -> np.ndarray:
    df = (per_day.with_columns(
        ((pl.col("mae_price_only") - pl.col(f"mae_{cell}"))
         * pl.col("area").replace_strict(w, default=0.0)).alias("wimp"))
        .group_by("test_day").agg(pl.col("wimp").sum()).sort("test_day"))
    return np.asarray(df["wimp"], dtype=float)


def _per_day(panel: pl.DataFrame, cells: list[str]) -> pl.DataFrame:
    return (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .sort("area", "test_day"))


def main() -> int:
    panel = pl.read_parquet(PANEL)
    cw_flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    panel = (panel.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
             .with_columns(pl.col("datetime_jst").dt.hour().alias("hour"),
                           pl.col("datetime_jst").dt.month().alias("month"),
                           (pl.col("y_actual") <= NEAR_FLOOR).alias("floor")))
    cells = [*TIERS, "price_only"]
    areas = sorted(panel["area"].unique().to_list())
    weights = _weights(areas)

    out: dict = {"phase": 82, "title": "Weighted metrics, relative displays, slices",
                 "precommit_doc": PRECOMMIT,
                 "traded_volume_weighting": "unavailable: per-area traded volume "
                                            "is not in the spot cache; not proxied",
                 "weights": {k: {a: _j(v, 4) for a, v in w.items()}
                             for k, w in weights.items()},
                 "by_weighting": {}, "slices": {}, "relative": {}}

    per_day_all = _per_day(panel, cells)
    for wname, w in weights.items():
        rec = {}
        for tier in (HEADLINE, POST):
            d = _pooled(per_day_all, tier, w)
            lo, hi = _boot_ci(d)
            rec[tier] = {"improvement": _j(float(d.mean())),
                         "ci95": [_j(lo), _j(hi)], "n_days": len(d)}
        dgt = _pooled(per_day_all, POST, w) - _pooled(per_day_all, HEADLINE, w)
        lo, hi = _boot_ci(dgt)
        rec["gate_timing_delta"] = {"delta": _j(float(dgt.mean())),
                                    "ci95": [_j(lo), _j(hi)]}
        out["by_weighting"][wname] = rec
        log.info("[%s] headline %.4f, gate delta %.4f [%s, %s]", wname,
                 rec[HEADLINE]["improvement"], rec["gate_timing_delta"]["delta"],
                 lo, hi)

    # slices (unweighted, headline tier)
    for sname, hours in SLICES.items():
        sub = panel if hours is None else panel.filter(
            pl.col("hour").is_between(*hours))
        pd_s = _per_day(sub, cells)
        d = _pooled(pd_s, HEADLINE, weights["unweighted"])
        lo, hi = _boot_ci(d)
        out["slices"][sname] = {"improvement": _j(float(d.mean())),
                                "ci95": [_j(lo), _j(hi)], "n_days": len(d)}
    for label, f in (("floor", pl.col("floor")), ("non_floor", ~pl.col("floor"))):
        pd_s = _per_day(panel.filter(f), cells)
        d = _pooled(pd_s, HEADLINE, weights["unweighted"])
        lo, hi = _boot_ci(d)
        out["slices"][label] = {"improvement": _j(float(d.mean())),
                                "ci95": [_j(lo), _j(hi)], "n_days": len(d)}
    for season, months in SEASONS.items():
        pd_s = _per_day(panel.filter(pl.col("month").is_in(list(months))), cells)
        d = _pooled(pd_s, HEADLINE, weights["unweighted"])
        lo, hi = _boot_ci(d)
        out["slices"][season] = {"improvement": _j(float(d.mean())),
                                 "ci95": [_j(lo), _j(hi)], "n_days": len(d)}

    # relative displays
    base_mae = float(per_day_all["mae_price_only"].mean())
    for tier in TIERS:
        tier_mae = float(per_day_all[f"mae_{tier}"].mean())
        out["relative"][tier] = {
            "baseline_mae": _j(base_mae), "tier_mae": _j(tier_mae),
            "improvement_pct_of_baseline": _j(100.0 * (base_mae - tier_mae) / base_mae, 1),
            "rmae_vs_price_only": _j(tier_mae / base_mae)}

    flips = [w for w, r in out["by_weighting"].items()
             if r[HEADLINE]["ci95"][0] is not None and r[HEADLINE]["ci95"][0] <= 0]
    out["sign_stability"] = {"headline_ci_excludes_zero_under_all_weights": not flips,
                             "weightings_with_ci_touching_zero": flips}

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "n_boot": N_BOOT, "boot_block_days": BOOT_BLOCK}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
