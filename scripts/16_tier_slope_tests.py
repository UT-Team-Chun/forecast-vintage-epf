r"""Phase 55: tier slope-difference tests for the cross-area concentration claim.

MOTIVATION (reviewer major issue 4). The manuscript's heterogeneity discussion
says "proxy concentration significant, operational n.s." -- but a significant
correlation on one tier next to a non-significant one on another does NOT
establish that the two correlations DIFFER (Gelman & Stern: the difference
between 'significant' and 'not significant' is not itself significant).
This phase runs the direct test of the tier CONTRAST: for each tier pair
(era5-proxy vs prev2 pre-gate operational; era5-proxy vs prev1 first
post-gate), test H0 "cross-area concentration equal across tiers" on both
the full phase51 window and the committed-protocol bridge subset.

DESIGN (pre-committed before results; see INTERPRETATION_RULE below):
  Data     : phase51 per-slot parquet (714 test days x 9 areas x 6 cells,
             identical folds). Per area a, day t, tier k: daily improvement
             d_k(a,t) = daily MAE(price_only) - daily MAE(k)  [yen/kWh,
             positive = tier better]. Area-level improvement
             I_ak = mean_t d_k(a,t) (days equally weighted -- matches the
             day-block bootstrap unit; differs from phase51's slot-mean by
             <1e-3 because slot counts are near-constant).
  x-axis   : PV/peak capacity ratio from data/reference/area_capacity.csv
             via area_capacity_loader (single source of truth; asserted equal
             to the parquet's pv_peak_ratio column).
  Stats    : per tier k: OLS slope beta_k of I_ak on PV/peak (n=9) and
             Spearman rho_k. Tier contrast Delta = stat(era5) - stat(op_tier)
             (positive = proxy MORE concentrated than the operational tier).
  Test 1   : per-day paired circular block bootstrap (block length 7 days,
             10,000 draws, seed 2026, day indices SHARED across areas and
             tiers) -> percentile 95% CI of Delta_slope and Delta_rho, plus
             two-sided percentile p = 2*min(P(D*<=0), P(D*>=0)).
  Test 2   : permutation variant -- tier labels permuted WITHIN each
             (area, day) pair (independent fair coin per pair, 10,000 draws,
             seed 2027) after removing each tier's grand mean over the window
             (slope and Spearman are invariant to that additive centering, so
             the observed Delta is unchanged; centering stops the permutation
             null from being rejected by the huge tier LEVEL difference,
             which is not the hypothesis under test). Two-sided
             p = (1 + #{|Delta_perm| >= |Delta_obs|}) / (1 + n_perm).
  Test 2b  : exact area-level variant -- tier labels swapped for entire
             areas (all 2^9 = 512 patterns enumerated, identity included);
             exact two-sided p = #{|Delta_perm| >= |Delta_obs|} / 512.
             Robust to within-area serial dependence by construction.
  Windows  : (a) full phase51 window (all test days 2024-08-01..2026-07-18);
             (b) committed-protocol bridge subset test_day in
             [2024-09-15, 2025-11-06] -- 2024-09-15 = first test day
             reachable under the committed phase35/41 protocol, 2025-11-06 =
             last committed-record test day (phase40/41 extension), i.e. the
             closest reconstruction of the OLD committed window that the
             phase51 parquet allows (protocol still differs: phase51 initial
             train end 2024-07-18, STEP=1 d, 9-area all-hours panel).

CLI (fast, ~1-2 min):
  cd <repository root>
  uv run python \
      scripts/16_tier_slope_tests.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import peak_gw_by_area, pv_gw_by_area

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase55")

SEED = 2026
N_BOOT = 10_000
N_PERM = 10_000
PERM_CHUNK = 1_000
BLOCK_LEN = 7
CELLS = ("price_only", "era5", "prev2", "prev1", "histfc", "oracle")
TIERS = ("era5", "prev2", "prev1", "histfc", "oracle")
PAIRS = (("era5", "prev2"), ("era5", "prev1"))
BRIDGE_START, BRIDGE_END = "2024-09-15", "2025-11-06"

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
IN_PARQUET = RUN / "phase51_vintage_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase55_tier_slope_tests.json"

INTERPRETATION_RULE = (
    "Pre-committed BEFORE results: whatever the outcome, phase55 lands as "
    "'suggestive heterogeneity' evidence in the Discussion/Supplement. A "
    "significant tier difference does NOT get promoted to a headline claim "
    "(the n = 8-9 cross-area power limits in Section 7 stand as written); a "
    "non-significant tier difference closes the old 'significant vs n.s.' "
    "contrast honestly, i.e. the manuscript must not imply the correlations "
    "were shown to DIFFER."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _per_day_improvements(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (area, test_day): daily MAE per cell and improvement vs price_only."""
    per_day = (
        panel.with_columns(
            *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
              for c in CELLS])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in CELLS],
             pl.len().alias("n_slots"))
        .sort("area", "test_day"))
    return per_day.with_columns(
        *[(pl.col("mae_price_only") - pl.col(f"mae_{k}")).alias(f"imp_{k}")
          for k in TIERS])


def _matrices(per_day: pl.DataFrame, areas: list[str],
              days: list[str]) -> dict[str, np.ndarray]:
    """imp[tier] as (n_areas, n_days) arrays, area/day order fixed."""
    out: dict[str, np.ndarray] = {}
    for k in TIERS:
        piv = (per_day.pivot(on="test_day", index="area", values=f"imp_{k}")
               .sort("area"))
        assert piv["area"].to_list() == areas
        mat = piv.select(days).to_numpy().astype(float)
        assert not np.isnan(mat).any(), f"missing (area, day) cells for {k}"
        out[k] = mat
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slope of y on x across the last axis (areas). y: (..., n_areas)."""
    xc = x - x.mean()
    return (y @ xc) / (xc @ xc)


def _spearman_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of y (..., n_areas) against fixed x."""
    rx = stats.rankdata(x)
    rxc = rx - rx.mean()
    ry = stats.rankdata(y, axis=-1)
    ryc = ry - ry.mean(axis=-1, keepdims=True)
    denom = np.sqrt((ryc ** 2).sum(axis=-1) * (rxc ** 2).sum())
    return (ryc @ rxc) / denom


def _bootstrap_indices(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap day indices, shared across all series."""
    n_blocks = int(np.ceil(n_days / BLOCK_LEN))
    starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]) % n_days
    return idx.reshape(N_BOOT, -1)[:, :n_days]


def _day_weights(idx: np.ndarray, n_days: int) -> np.ndarray:
    """(N_BOOT, n_days) resample multiplicity matrix from bootstrap indices."""
    w = np.zeros((idx.shape[0], n_days))
    for b in range(idx.shape[0]):
        w[b] = np.bincount(idx[b], minlength=n_days)
    return w


def _ci_p(draws: np.ndarray, point: float) -> dict:
    p = 2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
    return {"point": round(point, 4),
            "boot_lo95": round(float(np.percentile(draws, 2.5)), 4),
            "boot_hi95": round(float(np.percentile(draws, 97.5)), 4),
            "boot_p_two_sided": round(min(p, 1.0), 4)}


def _perm_within_area_day(dc_a: np.ndarray, dc_b: np.ndarray, x: np.ndarray,
                          obs_dslope: float, obs_drho: float,
                          rng: np.random.Generator) -> dict:
    """Tier labels permuted independently within each (area, day) pair."""
    n_ge_slope = n_ge_rho = 0
    n_areas, n_days = dc_a.shape
    done = 0
    while done < N_PERM:
        m = min(PERM_CHUNK, N_PERM - done)
        s = rng.random((m, n_areas, n_days)) < 0.5
        pa = np.where(s, dc_b, dc_a)   # permuted 'era5' series
        pb = np.where(s, dc_a, dc_b)   # permuted operational series
        ia, ib = pa.mean(axis=2), pb.mean(axis=2)          # (m, n_areas)
        dslope = _slope(x, ia) - _slope(x, ib)
        drho = _spearman_vec(x, ia) - _spearman_vec(x, ib)
        n_ge_slope += int(np.sum(np.abs(dslope) >= abs(obs_dslope) - 1e-12))
        n_ge_rho += int(np.sum(np.abs(drho) >= abs(obs_drho) - 1e-12))
        done += m
    return {"n_perm": N_PERM,
            "p_two_sided_delta_slope": round((1 + n_ge_slope) / (1 + N_PERM), 4),
            "p_two_sided_delta_rho": round((1 + n_ge_rho) / (1 + N_PERM), 4)}


def _perm_area_level_exact(dc_a: np.ndarray, dc_b: np.ndarray, x: np.ndarray,
                           obs_dslope: float, obs_drho: float) -> dict:
    """Tier labels swapped for whole areas; all 2^9 patterns enumerated."""
    n_areas = dc_a.shape[0]
    ia0, ib0 = dc_a.mean(axis=1), dc_b.mean(axis=1)        # (n_areas,)
    patterns = np.array([[(m >> a) & 1 for a in range(n_areas)]
                         for m in range(1 << n_areas)], dtype=bool)
    ia = np.where(patterns, ib0[None, :], ia0[None, :])    # (512, n_areas)
    ib = np.where(patterns, ia0[None, :], ib0[None, :])
    dslope = _slope(x, ia) - _slope(x, ib)
    drho = _spearman_vec(x, ia) - _spearman_vec(x, ib)
    n = patterns.shape[0]
    return {"n_patterns": n,
            "p_two_sided_delta_slope": round(
                float(np.sum(np.abs(dslope) >= abs(obs_dslope) - 1e-12)) / n, 4),
            "p_two_sided_delta_rho": round(
                float(np.sum(np.abs(drho) >= abs(obs_drho) - 1e-12)) / n, 4)}


def _analyse_window(imp: dict[str, np.ndarray], x: np.ndarray, areas: list[str],
                    days: list[str], rng_boot: np.random.Generator,
                    rng_perm: np.random.Generator) -> dict:
    n_days = len(days)
    idx = _bootstrap_indices(n_days, rng_boot)
    w = _day_weights(idx, n_days)

    # per-tier point stats + per-tier bootstrap draws of (slope, rho)
    boot_i = {k: (imp[k] @ w.T).T / n_days for k in TIERS}       # (N_BOOT, 9)
    per_tier: dict = {}
    draws: dict[str, dict[str, np.ndarray]] = {}
    for k in TIERS:
        i_k = imp[k].mean(axis=1)
        lin = stats.linregress(x, i_k)
        sp = stats.spearmanr(x, i_k)
        slope_draws = _slope(x, boot_i[k])
        rho_draws = _spearman_vec(x, boot_i[k])
        draws[k] = {"slope": slope_draws, "rho": rho_draws}
        per_tier[k] = {
            "improvement_by_area": {a: round(float(v), 4)
                                    for a, v in zip(areas, i_k, strict=True)},
            "ols_slope_yenkwh_per_pvpeak": round(float(lin.slope), 4),
            "ols_slope_p_classical": round(float(lin.pvalue), 4),
            "ols_slope_boot_ci95": [round(float(np.percentile(slope_draws, 2.5)), 4),
                                    round(float(np.percentile(slope_draws, 97.5)), 4)],
            "spearman_rho": round(float(sp.statistic), 3),
            "spearman_p_classical": round(float(sp.pvalue), 3),
            "spearman_rho_boot_ci95": [round(float(np.percentile(rho_draws, 2.5)), 3),
                                       round(float(np.percentile(rho_draws, 97.5)), 3)],
        }

    # tier-pair contrasts: Delta = stat(era5) - stat(operational tier)
    pair_tests: dict = {}
    for a, b in PAIRS:
        i_a, i_b = imp[a].mean(axis=1), imp[b].mean(axis=1)
        obs_dslope = float(_slope(x, i_a) - _slope(x, i_b))
        obs_drho = float(_spearman_vec(x, i_a[None, :])[0]
                         - _spearman_vec(x, i_b[None, :])[0])
        dslope_draws = draws[a]["slope"] - draws[b]["slope"]
        drho_draws = draws[a]["rho"] - draws[b]["rho"]
        dc_a = imp[a] - imp[a].mean()
        dc_b = imp[b] - imp[b].mean()
        # centering leaves slope/rho contrasts unchanged (invariance check)
        chk = float(_slope(x, dc_a.mean(axis=1)[None, :])[0]
                    - _slope(x, dc_b.mean(axis=1)[None, :])[0])
        assert abs(chk - obs_dslope) < 1e-9
        pair_tests[f"{a}_vs_{b}"] = {
            "sign_convention": f"Delta = stat({a}) - stat({b}); positive = "
                               f"proxy more concentrated than {b}",
            "delta_slope": _ci_p(dslope_draws, obs_dslope),
            "delta_rho": _ci_p(drho_draws, obs_drho),
            "perm_within_area_day": _perm_within_area_day(
                dc_a, dc_b, x, obs_dslope, obs_drho, rng_perm),
            "perm_area_level_exact": _perm_area_level_exact(
                dc_a, dc_b, x, obs_dslope, obs_drho),
        }

    return {"n_test_days": n_days, "first_test_day": days[0],
            "last_test_day": days[-1], "per_tier": per_tier,
            "pair_tests": pair_tests}


def _realised_saturation(areas: list[str]) -> np.ndarray:
    """Vintage-free saturation index: max realised PV potential / peak demand.

    ``data/reference/area_capacity.csv`` is an FY2023-24 "approx" vintage whose
    nameplate figures are exceeded by realised OCCTO output in seven area-channels
    (phase62). The table index is therefore reported alongside this observable one,
    so the cross-area conclusion can be shown not to depend on the stale figures.
    Areas absent from the OCCTO panel (Chubu) fall back to the table index.
    """
    occ = pl.read_parquet(RUN / "phase25.3_occto_generation.parquet")
    per = (occ.with_columns((pl.col("pv_mw")
                            + pl.col("pv_curtailment_mw").fill_null(0.0))
                           .alias("pv_pot"))
           .group_by("area")
           .agg(pl.col("pv_pot").max().alias("pv_max"),
                pl.col("demand_mw").max().alias("peak")))
    idx = {r["area"]: r["pv_max"] / r["peak"] for r in per.to_dicts() if r["peak"]}
    pv, peak = pv_gw_by_area(), peak_gw_by_area()
    return np.array([idx.get(a, pv[a] / peak[a]) for a in areas], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saturation-index", choices=("table", "realised"),
                    default="table",
                    help="'table' = PV/peak from area_capacity.csv (headline); "
                         "'realised' = max realised PV potential / peak demand "
                         "(vintage-free robustness, writes a _realised JSON)")
    args = ap.parse_args()
    t0 = time.time()
    panel = pl.read_parquet(IN_PARQUET)
    per_day = _per_day_improvements(panel)
    areas = sorted(per_day["area"].unique().to_list())
    all_days = sorted(per_day["test_day"].unique().to_list())
    counts = per_day.group_by("test_day").agg(pl.len().alias("n_areas"))
    assert counts.filter(pl.col("n_areas") != len(areas)).height == 0, \
        "test days not aligned across areas"
    log.info("panel: %d areas x %d test days (%s..%s)", len(areas),
             len(all_days), all_days[0], all_days[-1])

    pv, peak = pv_gw_by_area(), peak_gw_by_area()
    x_table = np.array([pv[a] / peak[a] for a in areas], dtype=float)
    pq_ratio = (panel.group_by("area").agg(pl.col("pv_peak_ratio").first())
                .sort("area")["pv_peak_ratio"].to_numpy())
    assert np.allclose(x_table, pq_ratio, atol=1e-9), \
        "area_capacity.csv PV/peak != parquet pv_peak_ratio"
    x = (x_table if args.saturation_index == "table"
         else _realised_saturation(areas))
    log.info("saturation index = %s: %s", args.saturation_index,
             {a: round(float(v), 4) for a, v in zip(areas, x, strict=True)})

    imp_full = _matrices(per_day, areas, all_days)
    bridge_days = [d for d in all_days if BRIDGE_START <= d <= BRIDGE_END]
    bsl = slice(all_days.index(bridge_days[0]), all_days.index(bridge_days[-1]) + 1)
    assert all_days[bsl] == bridge_days
    imp_bridge = {k: v[:, bsl] for k, v in imp_full.items()}

    rng_boot = np.random.default_rng(SEED)
    rng_perm = np.random.default_rng(SEED + 1)
    log.info("analysing full window (%d days)...", len(all_days))
    res_full = _analyse_window(imp_full, x, areas, all_days, rng_boot, rng_perm)
    log.info("analysing committed bridge window (%d days)...", len(bridge_days))
    res_bridge = _analyse_window(imp_bridge, x, areas, bridge_days,
                                 rng_boot, rng_perm)
    res_full["definition"] = ("all phase51 test days (densified full-window "
                              "protocol, STEP=1 d, 9 areas, all hours)")
    res_bridge["definition"] = (
        f"test_day in [{BRIDGE_START}, {BRIDGE_END}]: 2024-09-15 = first test "
        "day reachable under the committed phase35/41 protocol; 2025-11-06 = "
        "last committed-record test day (phase40/41 extension). Closest "
        "reconstruction of the OLD committed window the phase51 parquet "
        "allows; protocol itself still phase51's (initial train end "
        "2024-07-18, STEP=1 d, 9-area all-hours panel), NOT the committed "
        "8-area phase35 sample on which rho=+0.833 was measured")

    summary = {
        "phase": "55",
        "title": "tier slope-difference tests for the cross-area concentration claim",
        "motivation": ("reviewer major issue 4: 'proxy correlation significant, "
                       "operational n.s.' does not establish that the "
                       "correlations DIFFER; test the tier contrast directly"),
        "interpretation_rule_precommitted": INTERPRETATION_RULE,
        "inputs": {
            "per_slot_parquet": str(IN_PARQUET.relative_to(RUN.parent.parent)),
            "per_slot_parquet_sha256": _sha256(IN_PARQUET),
            "n_slots": int(panel.height),
            "pv_peak_source": "data/reference/area_capacity.csv via "
                              "area_capacity_loader (asserted == parquet column)",
            "pv_peak_by_area": {a: round(float(v), 4)
                                for a, v in zip(areas, x, strict=True)},
        },
        "design": {
            "improvement_unit": ("I_ak = mean over test days of daily "
                                 "[MAE(price_only) - MAE(tier)] per area, "
                                 "yen/kWh; days equally weighted"),
            "slope": "OLS of I_ak on PV/peak across n=9 areas",
            "rank_assoc": "Spearman rho of I_ak vs PV/peak (n=9)",
            "bootstrap": {"kind": "per-day paired circular block bootstrap, "
                                  "day indices shared across areas and tiers",
                          "block_len_days": BLOCK_LEN, "n_boot": N_BOOT,
                          "seed": SEED},
            "permutation_within_area_day": {
                "kind": "tier labels swapped independently within each "
                        "(area, day) pair on grand-mean-centered daily "
                        "improvements (centering removes the tier LEVEL "
                        "difference, which slope/rho contrasts are invariant "
                        "to; without it the test would reject on levels alone)",
                "n_perm": N_PERM, "seed": SEED + 1},
            "permutation_area_level_exact": {
                "kind": "tier labels swapped for entire areas, all 2^9 = 512 "
                        "patterns enumerated (exact; robust to within-area "
                        "serial dependence)"},
            "caveat": ("bootstrap resamples DAYS with the 9-area panel fixed: "
                       "it quantifies time-sampling noise of the tier "
                       "contrast, not uncertainty from the areas themselves "
                       "(the 9 areas are the full population of JEPX zones)"),
        },
        "old_claim_context": {
            "phase35_committed_window": "proxy rho=+0.833 (p=0.01, n=8, "
                                        "all-hours committed sample)",
            "phase40_extension": "proxy rho=+0.72 (p=0.03)",
            "phase51_extended_window": "all tiers n.s. (era5 rho=+0.03, "
                                       "prev2 +0.38, prev1 +0.42; n=9)",
        },
        "windows": {"full_phase51": res_full, "committed_bridge": res_bridge},
        "metadata": {
            "script": Path(__file__).name,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"],
                                      capture_output=True, text=True,
                                      check=False).stdout.strip(),
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0, 1),
        },
    }
    out_path = (OUT_JSON if args.saturation_index == "table"
                else OUT_JSON.with_name(f"{OUT_JSON.stem}_realised.json"))
    summary["saturation_index"] = args.saturation_index
    summary["inputs"]["saturation_index_by_area"] = {
        a: round(float(v), 4) for a, v in zip(areas, x, strict=True)}
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote %s (%.1f s)", out_path, time.time() - t0)

    for wname, res in summary["windows"].items():
        for pname, pt in res["pair_tests"].items():
            log.info("[%s] %s: Dslope %s | Drho %s | perm(day) %s | perm(area) %s",
                     wname, pname, pt["delta_slope"], pt["delta_rho"],
                     pt["perm_within_area_day"], pt["perm_area_level_exact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
