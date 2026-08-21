r"""Phase 103: decompose the headline weather gain into a floor channel and the rest.

QUESTION
--------
The manuscript's third claim (ae_05_results.tex) is that the weather value
concentrates in floor transitions: "on near-floor slots ... a gain of about
\FloorAbsGain (1.84) JPY/kWh, roughly seven times the pooled \PreGateImp (0.2529)
the ladder reports". The paper never DECOMPOSES the pooled gain, so a referee
cannot see how much of the TOTAL improvement runs through the floor channel and
how much through the continuous-price channel. A 7x intensity on a slice that is
3.33 % of slots (\EvalNearFloorPct, phase94) is arithmetically compatible with the
floor channel carrying only a minority of the aggregate gain. This phase measures
that share two ways, both exact.

DEFINITIONS (the paper's own, not invented here)
-----------------------------------------------
tau = 0.5 JPY/kWh  -- \thrFloorDefault in docs/paper/shared/macros.tex; the
   "near_floor_threshold" of phase82b (data/runs/phase82b_absolute_metrics.json),
   the same band phase94 counts at 3.33 % of slots and the band the seven-times
   sentence is computed on. F_i = 1{y_i <= tau} on the REALISED price.
tau_admin = 0.01 JPY/kWh -- the administrative floor; carried as a secondary
   threshold in Part I only (Part II's fitted components are estimated at tau
   only, for compute; this is declared here, not chosen after seeing results).
Panel / folds / cells: phase85's CW-primary ladder verbatim
   (scripts/10_primary_ladder_cw.py), 732 daily folds x
   9 areas = 315,414 slots, baseline cell `price_only`, headline cell
   `msm_d2_12z_cw`. Part II reuses that harness by importing the module and
   rebinding `_fit` and `_design`, asserting the rebinding took (phase96's
   pattern; phase91's lesson that a silent module-global swap turns a contrast
   into a copy).
Aggregation: the PAPER'S estimator, i.e. phase85's `_pooled_daily` --- mean over
   test days of the cross-area mean of per-(area, day) MAE differences. Every
   share below is therefore a share OF THE PUBLISHED 0.2529, not of some other
   weighting. Slot-pooled figures are reported alongside for phase82b comparison.

PART I -- exact state partition of the published gain (no refits)
----------------------------------------------------------------
For each (area a, day d) with n_{ad} slots, write the cell's MAE difference as a
sum over its own slots and split that sum by the realised state:

    mae_diff(a,d) = (1/n_ad) * sum_i [ |y_i - b_i| - |y_i - h_i| ]
                  = c_F(a,d) + c_NF(a,d),
    c_F(a,d)  = (1/n_ad) * sum_{i: F_i=1} [ |y_i - b_i| - |y_i - h_i| ],
    c_NF(a,d) = (1/n_ad) * sum_{i: F_i=0} [ ... ].

Because c_F and c_NF are SUMS divided by the cell count (not conditional means),
they are defined even for cells with no floor slot, and the identity holds
slot-by-slot. Averaging over areas then days (phase85's nested estimator, a
linear operator) gives

    Delta_paper = C_F + C_NF     exactly, where C_F = mean_d mean_a c_F(a,d).

Equivalently C_F = p_F * Delta_{|F} under uniform cell sizes: the floor SHARE of
the aggregate gain is prevalence x intensity, which is exactly the arithmetic the
manuscript's seven-times sentence leaves unstated. Reported shares:
s_F = C_F / Delta_paper, s_NF = 1 - s_F, pooled and per area, with the block
bootstrap over test days (phase85's `_block_boot_ci` / `_ratio_entry`, block 7,
10,000 draws, seed 2026) reused verbatim.

PART II -- hurdle (two-part) decomposition with fitted components
-----------------------------------------------------------------
Part I says WHERE the gain lands; it does not say WHICH sub-model the information
enters. For that, build the hurdle system the referee asked for. With
pi = P(y <= tau | X), mu = E[y | y > tau, X] and f the floor-level component,

    yhat(pi, mu) = pi * f + (1 - pi) * mu.

Per fold and area, on the SAME train rows phase85 uses, estimate
    pi^b, pi^h : XGBClassifier(300 x depth 4, lr 0.05, hist, logloss, seed 2026)
                 on the baseline / headline column set, label 1{y <= tau};
    mu^b, mu^h : phase85's OWN XGBRegressor parameters, fitted on the train rows
                 with y > tau, target y;
    f          : mean of y over train rows with y <= tau. COMMON TO BOTH ARMS and
                 never re-estimated per arm, so it cannot absorb any part of the
                 contrast (fallback tau_admin=0.01 if a fold's train has no floor
                 row; counted and reported).

Four systems, MAE under the paper's nested estimator:
    A = MAE(pi^b, mu^b)   B = MAE(pi^h, mu^b)
    C = MAE(pi^b, mu^h)   D = MAE(pi^h, mu^h)
Total hurdle gain G = A - D. Two exact readings of the same four numbers:

  (i) main effects + interaction
      g_pi  = A - B                      (floor-state channel, level held at base)
      g_mu  = A - C                      (conditional-level channel, pi held)
      g_int = G - g_pi - g_mu = B + C - A - D
      identity: g_pi + g_mu + g_int = G.
  (ii) symmetric Shapley split (no residual term by construction)
      S_pi = 0.5 * [ (A - B) + (C - D) ]
      S_mu = 0.5 * [ (A - C) + (B - D) ]
      identity: S_pi + S_mu = G.

HONESTY NOTE ON "EXACT". Both identities are algebraic in A, B, C, D, so
asserting them to floating tolerance verifies the arithmetic, not the science; it
is a tautology and is reported as one. The substantive check is EXTERNAL: does
the hurdle system reproduce the ladder? G is compared with the ladder's
Delta_paper and A, D with the ladder's published MAE levels (2.1506 / 1.8989).
If G is far from Delta_paper, the shares are properties of the hurdle system and
are labelled as such rather than presented as shares of the paper's number.

PRE-REGISTERED DECISION BRANCHES (fixed before any number was computed)
----------------------------------------------------------------------
B1 s_F >= 2/3      -> the floor channel carries the aggregate gain; the third
                      claim holds in aggregate as well as in intensity.
B2 1/3 < s_F < 2/3 -> MIXED. Report both halves; the claim survives only as
                      "the floor regime is where the gain is most intense", and
                      the aggregate is shared with the continuous channel.
B3 s_F <= 1/3      -> CONTRADICTS a strong reading of the third claim: most of
                      the pooled improvement runs through non-floor slots. Report
                      plainly, with prevalence, and say the seven-times sentence
                      is an intensity statement that cannot be read as an
                      aggregate one. THIS BRANCH IS EXPECTED IF prevalence is
                      3.33 % and intensity is 7x (3.33% x 7 ~ 23 %), and it is
                      reported as the headline if it fires.
B4 S_pi / G >= 1/2 -> the improvement runs mainly through better floor-state
                      classification.
B5 S_pi / G < 1/2  -> it runs mainly through the conditional level; the floor
                      channel is real but is not the main conduit. Report.
B6 |G - Delta_paper| / Delta_paper > 0.5 -> the hurdle reconstruction is not a
                      faithful stand-in for the ladder. Part II is then reported
                      as a decomposition of the hurdle system only, flagged in
                      the artefact and in the return, and Part I carries the
                      answer.
B7 build incomplete (any area missing) -> report Part I on the full panel and
                      Part II only on the areas actually built, with the area
                      list, rather than silently pooling a smaller panel.
B8 any null in a required column of the CW-complete panel -> abort and report
                      the null counts. Nulls are counted and reported in every
                      branch, including the success branch.

POSITIVE CONTROLS
-----------------
P1 (Part I, panel identity) Re-derive the ladder's published headline from the
   phase85 parquet with the same estimator and require |Delta - 0.2529| < 5e-4,
   and the near-floor share to match phase94's 3.33 % and phase82b's floor-slice
   MAEs (5.223 -> 3.383). If these fail, the panel or the floor definition is
   wrong and nothing else is reportable.
P2 (Part II, parameter fidelity) On the first fold of each area, fit MY regressor
   parameters on ALL train rows and require the prediction to equal phase85's own
   `_fit` output to 1e-9. This proves the estimator differs from the ladder's
   only in its training subset, not in its hyper-parameters or its data.
P3 (Part II, instrument sensitivity) Substitute an ORACLE floor state pi* = F_i
   into the baseline arm. The classification channel measured against it must be
   large and positive: perfect floor-state knowledge must buy a big MAE gain. If
   it does not, the decomposition cannot detect a classification channel at all
   and any null from B5 is uninterpretable -- that would be reported as
   UNDETERMINED, not as a null.

AMENDMENTS AFTER FIRST COMPUTATION (disclosed, not backdated)
------------------------------------------------------------
A1 The two floor-slice legs of P1 were anchored on the macros
   \AbsFloorBaselineMae (5.223) and \AbsFloorPreGateMae (3.383) and FAILED
   (observed 5.242 / 3.392). Diagnosis: those macros come from phase82b's
   RELATIVE block, computed on the sub-panel where the naive-168h benchmark
   exists (10,379 floor slots), while the ladder panel has 10,504. The observed
   values match phase82b's own ABSOLUTE block bit-for-bit. The failing check is
   left in the artefact and a like-for-like check against phase82b's JSON is
   added beside it; no threshold and no definition was changed to make a number
   pass.
A2 A squared-error partition at tau = 0.5 was ADDED after the absolute-loss
   result was known, to answer in advance "you picked the metric kindest to the
   non-floor bulk". The absolute loss remains primary and its numbers are
   untouched by the addition.
A4 A realised-price BAND partition (<=0.5 / 0.5-3 / 3-8 / 8-15 / >15 JPY/kWh)
   was ADDED for the same reason as A2: the surplus channel may be wider than the
   administrative band, and a referee should be able to see where in the price
   distribution the gain sits. Same identity, disjoint and exhaustive bands.
A3 A four-state TRANSITION partition (floor entry / floor persistence / floor
   exit / non-floor persistence) was ADDED, also after the first result, because
   the manuscript's wording is "floor transitions" and the state partition tests
   only "near-floor slots". It is the same exact identity over four disjoint
   states instead of two and changes no earlier number.

CLI (from the repository root):
  S=scripts/32b_floor_channel_decomposition.py
  uv run python $S --step partition
  uv run python ... --step build --areas kyushu
  uv run python ... --step analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase103")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P85_PATH = Path(__file__).parent / "10_primary_ladder_cw.py"
P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
P82B_JSON = RUN / "phase82b_absolute_metrics.json"
P94_JSON = RUN / "phase94_evaluation_regime.json"
CKPT = RUN / "phase103_hurdle_checkpoints"      # phase85's harness output (ignored)
ACC = RUN / "phase103_hurdle_components"        # my per-area component parquets
OUT_JSON = RUN / "phase103_floor_channel_decomposition.json"

BASE_CELL = "price_only"
HEAD_CELL = "msm_d2_12z_cw"
TAU = 0.5
TAU_ADMIN = 0.01
SEED = 2026
# phase85's `_fit` hyper-parameters, verbatim; P2 asserts the copy is faithful.
XGB_REG = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
           "random_state": SEED, "n_jobs": 4, "tree_method": "hist"}
XGB_CLF = {**XGB_REG, "eval_metric": "logloss"}
# published anchors for P1
LADDER_IMP = 0.2529
LADDER_MAE_BASE = 2.1506
LADDER_MAE_HEAD = 1.8989
FLOOR_MAE_BASE = 5.223
FLOOR_MAE_HEAD = 3.383
NEAR_FLOOR_PCT = 3.33
P1_TOL = 5e-4
P2_TOL = 1e-9
IDENTITY_TOL = 1e-9
ORACLE_MIN_GAIN = 0.05     # P3: perfect floor state must buy at least this much
FIDELITY_MAX_REL = 0.5     # B6


# ----------------------------------------------------------------- utilities
def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,  # noqa: S607
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_phase85() -> object:
    spec = importlib.util.spec_from_file_location("p85", P85_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p85"] = mod
    spec.loader.exec_module(mod)
    return mod


def _j(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _daily(per_cell: pl.DataFrame, col: str) -> np.ndarray:
    """phase85's nested estimator: mean over areas within a day, ordered by day."""
    return np.asarray(
        (per_cell.group_by("test_day").agg(pl.col(col).mean())
         .sort("test_day")[col]), dtype=float)


def _entry(mod: object, d: np.ndarray) -> dict:
    lo, hi = mod._block_boot_ci(d)
    return {"mean": _j(float(d.mean())), "ci95": [_j(lo), _j(hi)],
            "dm_l3": mod._hln_dm(d, lag=3), "n_days": len(d)}


# ------------------------------------------------------- Part I: partition
def _loss(col: str, loss: str) -> pl.Expr:
    e = pl.col(col) - pl.col("y_actual")
    return e.abs() if loss == "abs" else e.pow(2)


def _state_partition(mod: object, panel: pl.DataFrame, tau: float, *,  # noqa: PLR0913 - one explicit knob per pre-registered variant
                     base: str, head: str, tag: str, loss: str = "abs") -> dict:
    """Exact split of the paper's own estimator into floor and non-floor parts.

    `loss` is "abs" (the paper's MAE, primary) or "sq" (squared error, a
    pre-registered metric robustness: a referee will ask whether the floor share
    is an artefact of choosing the metric that is kindest to the non-floor bulk).
    """
    p = panel.with_columns(
        (pl.col("y_actual") <= tau).alias("is_floor"),
        (_loss(f"y_pred_{base}", loss)
         - _loss(f"y_pred_{head}", loss)).alias("dae"))
    per_cell = (p.group_by("area", "test_day").agg(
        pl.len().alias("n_slots"),
        pl.col("dae").mean().alias("mae_diff"),
        (pl.when(pl.col("is_floor")).then(pl.col("dae"))
         .otherwise(0.0)).sum().alias("sum_f"),
        (pl.when(~pl.col("is_floor")).then(pl.col("dae"))
         .otherwise(0.0)).sum().alias("sum_nf"),
        pl.col("is_floor").sum().alias("n_floor"))
        .with_columns((pl.col("sum_f") / pl.col("n_slots")).alias("c_f"),
                      (pl.col("sum_nf") / pl.col("n_slots")).alias("c_nf"),
                      (pl.col("n_floor") / pl.col("n_slots")).alias("p_f"))
        .sort("area", "test_day"))
    tot = _daily(per_cell, "mae_diff")
    c_f = _daily(per_cell, "c_f")
    c_nf = _daily(per_cell, "c_nf")
    resid = float(np.max(np.abs(tot - (c_f + c_nf))))
    if resid > IDENTITY_TOL:
        msg = f"state partition does not sum: max |resid| = {resid:.3e}"
        raise RuntimeError(msg)

    n_floor = int(p["is_floor"].sum())
    floor_only = p.filter(pl.col("is_floor"))
    nonfloor = p.filter(~pl.col("is_floor"))
    def _pool(frame: pl.DataFrame, col: str) -> float:
        return float(frame.select(_loss(col, loss)).to_series().mean())

    out = {
        "threshold_jpykwh": tau,
        "loss": "absolute error (the paper's MAE)" if loss == "abs"
                else "squared error (MSE units, JPY^2/kWh^2)",
        "identity": {"rule": "Delta_paper = C_F + C_NF, day/area nested estimator",
                     "max_abs_residual": resid, "tolerance": IDENTITY_TOL,
                     "exact": True},
        "prevalence": {"n_floor_slots": n_floor, "n_slots": int(p.height),
                       "share_pct": _j(100.0 * n_floor / p.height, 3)},
        "total_gain": _entry(mod, tot),
        "floor_contribution": _entry(mod, c_f),
        "nonfloor_contribution": _entry(mod, c_nf),
        "floor_share": mod._ratio_entry(c_f, tot),
        "nonfloor_share": mod._ratio_entry(c_nf, tot),
        "slot_pooled": {
            "mae_base_all": _j(_pool(panel, f"y_pred_{base}")),
            "mae_head_all": _j(_pool(panel, f"y_pred_{head}")),
            "mae_base_floor": _j(_pool(floor_only, f"y_pred_{base}"), 3),
            "mae_head_floor": _j(_pool(floor_only, f"y_pred_{head}"), 3),
            "mae_base_nonfloor": _j(_pool(nonfloor, f"y_pred_{base}"), 3),
            "mae_head_nonfloor": _j(_pool(nonfloor, f"y_pred_{head}"), 3),
        },
    }
    out["slot_pooled"]["intensity_floor"] = _j(
        out["slot_pooled"]["mae_base_floor"] - out["slot_pooled"]["mae_head_floor"], 3)
    out["slot_pooled"]["intensity_nonfloor"] = _j(
        out["slot_pooled"]["mae_base_nonfloor"]
        - out["slot_pooled"]["mae_head_nonfloor"], 4)
    sp = out["slot_pooled"]
    pooled_gain = sp["mae_base_all"] - sp["mae_head_all"]
    sp["intensity_pooled"] = _j(pooled_gain)
    sp["intensity_multiple_floor_over_pooled"] = _j(
        sp["intensity_floor"] / pooled_gain, 2)
    # the arithmetic the manuscript's seven-times sentence leaves unstated:
    # share of aggregate gain ~= prevalence x intensity multiple
    sp["prevalence_times_multiple"] = _j(
        (n_floor / p.height) * sp["intensity_multiple_floor_over_pooled"], 3)
    # the same share under slot-pooled instead of nested day/area weighting, so a
    # referee can see the choice of estimator does not drive it
    sp["floor_share_slot_pooled"] = _j(
        (n_floor / p.height) * sp["intensity_floor"] / pooled_gain, 3)
    per_area = {}
    for a in sorted(per_cell["area"].unique().to_list()):
        sub = per_cell.filter(pl.col("area") == a).sort("test_day")
        t = np.asarray(sub["mae_diff"], dtype=float)
        f = np.asarray(sub["c_f"], dtype=float)
        pa = p.filter(pl.col("area") == a)
        per_area[a] = {
            "total_gain": _j(float(t.mean())),
            "floor_contribution": _j(float(f.mean())),
            "nonfloor_contribution": _j(float(t.mean() - f.mean())),
            "floor_share": mod._ratio_entry(f, t),
            "floor_prevalence_pct": _j(100.0 * float(pa["is_floor"].mean()), 3),
            "n_days": len(t),
        }
    out["per_area"] = per_area
    log.info("[%s tau=%s] total %.4f = floor %.4f + nonfloor %.4f | floor share %s",
             tag, tau, tot.mean(), c_f.mean(), c_nf.mean(),
             out["floor_share"]["ratio"])
    return out


def _transition_partition(mod: object, panel: pl.DataFrame, tau: float, *,
                          base: str, head: str) -> dict:
    """Split the same exact gain into four floor-transition states.

    The manuscript says the value concentrates in floor TRANSITIONS, which is a
    stronger statement than "on near-floor slots". Within each area the test rows
    are consecutive delivery days (step = 1 day), so the previous row is the
    previous half-hour; the share of rows for which that is literally true is
    reported, because the claim is about transitions and a broken lag would
    invent them. Four disjoint states -> four contributions that still sum to the
    published gain exactly.
    """
    p = (panel.sort("area", "datetime_jst").with_columns(
        (pl.col("y_actual") <= tau).alias("F"),
        ((pl.col(f"y_pred_{base}") - pl.col("y_actual")).abs()
         - (pl.col(f"y_pred_{head}") - pl.col("y_actual")).abs()).alias("dae"))
        .with_columns(
            pl.col("F").shift(1).over("area").alias("F_prev"),
            (pl.col("datetime_jst").diff().over("area")
             .dt.total_minutes()).alias("gap_min")))
    contiguous = float((p["gap_min"] == 30).sum()) / (p.height - p["area"].n_unique())
    p = p.with_columns(pl.col("F_prev").fill_null(pl.col("F")))
    p = p.with_columns(
        pl.when(pl.col("F") & ~pl.col("F_prev")).then(pl.lit("floor_entry"))
        .when(pl.col("F") & pl.col("F_prev")).then(pl.lit("floor_persist"))
        .when(~pl.col("F") & pl.col("F_prev")).then(pl.lit("floor_exit"))
        .otherwise(pl.lit("nonfloor_persist")).alias("state"))
    states = ["floor_entry", "floor_persist", "floor_exit", "nonfloor_persist"]
    per_cell = (p.group_by("area", "test_day").agg(
        pl.len().alias("n_slots"),
        pl.col("dae").mean().alias("mae_diff"),
        *[(pl.when(pl.col("state") == s).then(pl.col("dae")).otherwise(0.0)
           ).sum().alias(f"sum_{s}") for s in states])
        .with_columns(*[(pl.col(f"sum_{s}") / pl.col("n_slots")).alias(f"c_{s}")
                        for s in states])
        .sort("area", "test_day"))
    tot = _daily(per_cell, "mae_diff")
    parts = {s: _daily(per_cell, f"c_{s}") for s in states}
    resid = float(np.max(np.abs(tot - sum(parts.values()))))
    if resid > IDENTITY_TOL:
        msg = f"transition partition does not sum: {resid:.3e}"
        raise RuntimeError(msg)
    counts = {s: int((p["state"] == s).sum()) for s in states}
    return {
        "threshold_jpykwh": tau,
        "identity": {"rule": "Delta_paper = sum over four transition states",
                     "max_abs_residual": resid, "tolerance": IDENTITY_TOL,
                     "exact": True},
        "lag_integrity": {"share_of_rows_whose_predecessor_is_30_min_earlier":
                          _j(contiguous, 5),
                          "note": "if this were far below 1, the transition "
                                  "labels would be artefacts of panel gaps"},
        "prevalence_pct": {s: _j(100.0 * counts[s] / p.height, 3) for s in states},
        "counts": counts,
        "total_gain": _entry(mod, tot),
        "contributions": {s: _entry(mod, parts[s]) for s in states},
        "shares": {s: mod._ratio_entry(parts[s], tot) for s in states},
    }


PRICE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("b0_floor_le_0.5", -1e9, 0.5), ("b1_0.5_to_3", 0.5, 3.0),
    ("b2_3_to_8", 3.0, 8.0), ("b3_8_to_15", 8.0, 15.0),
    ("b4_gt_15", 15.0, 1e9))


def _band_partition(mod: object, panel: pl.DataFrame, *,
                    base: str, head: str) -> dict:
    """Split the same exact gain across realised-price bands.

    A referee can reasonably say the PV-surplus channel is wider than the
    administrative floor band: prices of 1-3 JPY/kWh are also surplus prices.
    The bands are disjoint and exhaustive, so the contributions still sum to the
    published gain; nothing is re-weighted or re-fitted.
    """
    p = panel.with_columns(
        ((pl.col(f"y_pred_{base}") - pl.col("y_actual")).abs()
         - (pl.col(f"y_pred_{head}") - pl.col("y_actual")).abs()).alias("dae"))
    expr = pl.when(pl.col("y_actual") <= PRICE_BANDS[0][2]).then(
        pl.lit(PRICE_BANDS[0][0]))
    for name, lo, hi in PRICE_BANDS[1:]:
        expr = expr.when((pl.col("y_actual") > lo)
                         & (pl.col("y_actual") <= hi)).then(pl.lit(name))
    p = p.with_columns(expr.otherwise(pl.lit("unassigned")).alias("band"))
    if int((p["band"] == "unassigned").sum()):
        msg = "price bands are not exhaustive"
        raise RuntimeError(msg)
    names = [b[0] for b in PRICE_BANDS]
    per_cell = (p.group_by("area", "test_day").agg(
        pl.len().alias("n_slots"), pl.col("dae").mean().alias("mae_diff"),
        *[(pl.when(pl.col("band") == n).then(pl.col("dae")).otherwise(0.0)
           ).sum().alias(f"sum_{n}") for n in names])
        .with_columns(*[(pl.col(f"sum_{n}") / pl.col("n_slots")).alias(f"c_{n}")
                        for n in names])
        .sort("area", "test_day"))
    tot = _daily(per_cell, "mae_diff")
    parts = {n: _daily(per_cell, f"c_{n}") for n in names}
    resid = float(np.max(np.abs(tot - sum(parts.values()))))
    if resid > IDENTITY_TOL:
        msg = f"band partition does not sum: {resid:.3e}"
        raise RuntimeError(msg)
    counts = {n: int((p["band"] == n).sum()) for n in names}
    return {
        "bands_jpykwh": {b[0]: [b[1], b[2]] for b in PRICE_BANDS},
        "identity": {"rule": "Delta_paper = sum over disjoint realised-price bands",
                     "max_abs_residual": resid, "tolerance": IDENTITY_TOL,
                     "exact": True},
        "counts": counts,
        "prevalence_pct": {n: _j(100.0 * counts[n] / p.height, 3) for n in names},
        "total_gain": _entry(mod, tot),
        "contributions": {n: _entry(mod, parts[n]) for n in names},
        "shares": {n: mod._ratio_entry(parts[n], tot) for n in names},
    }


def _null_audit(panel: pl.DataFrame, cols: list[str]) -> dict:
    audit = {c: int(panel[c].null_count()
                    + (panel[c].is_nan().sum() if panel[c].dtype.is_float() else 0))
             for c in cols}
    return {"per_column_nulls_or_nans": audit,
            "total": int(sum(audit.values()))}


def _read_ladder_panel() -> pl.DataFrame:
    panel = pl.read_parquet(P85_PARQUET)
    flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    return panel.filter(pl.all_horizontal(pl.col(f) for f in flags))


def partition() -> dict:
    mod = _load_phase85()
    panel = _read_ladder_panel()
    need = ["y_actual", f"y_pred_{BASE_CELL}", f"y_pred_{HEAD_CELL}"]
    audit = _null_audit(panel, need)
    if audit["total"]:
        msg = f"B8: nulls/NaNs in required columns: {audit}"
        raise RuntimeError(msg)
    out: dict = {
        "phase": "103",
        "title": "Floor-channel vs continuous-channel decomposition of the "
                 "headline weather gain",
        "status": "part1_only",
        "panel": {"n_slots": int(panel.height),
                  "n_areas": int(panel["area"].n_unique()),
                  "n_test_days": int(panel["test_day"].n_unique()),
                  "first_test_day": panel["test_day"].min(),
                  "last_test_day": panel["test_day"].max(),
                  "source": str(P85_PARQUET),
                  "cw_complete_filter": "all cw_ok_* true (100 % of the phase85 "
                                        "panel, per phase85's cw_coverage)"},
        "null_audit": audit,
        "contrast": {"baseline_cell": BASE_CELL, "headline_cell": HEAD_CELL},
        "part1_state_partition": {
            "tau_0.5": _state_partition(mod, panel, TAU, base=BASE_CELL,
                                        head=HEAD_CELL, tag="primary"),
            "tau_0.01": _state_partition(mod, panel, TAU_ADMIN, base=BASE_CELL,
                                         head=HEAD_CELL, tag="admin"),
            "tau_0.5_squared_loss": _state_partition(
                mod, panel, TAU, base=BASE_CELL, head=HEAD_CELL,
                tag="sq-loss", loss="sq"),
        },
        "part1b_transition_partition": _transition_partition(
            mod, panel, TAU, base=BASE_CELL, head=HEAD_CELL),
        "part1c_price_band_partition": _band_partition(
            mod, panel, base=BASE_CELL, head=HEAD_CELL),
    }
    prim = out["part1_state_partition"]["tau_0.5"]
    # ---- three independently coded partitions must agree on the floor part
    tr = out["part1b_transition_partition"]["contributions"]
    band0 = out["part1c_price_band_partition"]["contributions"]["b0_floor_le_0.5"]
    trans_floor = (tr["floor_entry"]["mean"] + tr["floor_persist"]["mean"])
    out["cross_checks"] = {
        "why": "the two-state, four-state and band partitions are separate code "
               "paths; their floor parts must be the same number",
        "state_floor": prim["floor_contribution"]["mean"],
        "transition_entry_plus_persist": _j(trans_floor),
        "band_le_0.5": band0["mean"],
        "max_abs_gap": _j(max(abs(prim["floor_contribution"]["mean"] - trans_floor),
                              abs(prim["floor_contribution"]["mean"]
                                  - band0["mean"])), 6),
        "pass": bool(max(abs(prim["floor_contribution"]["mean"] - trans_floor),
                         abs(prim["floor_contribution"]["mean"]
                             - band0["mean"])) < 1e-4),
    }
    # ---- P1: the published anchors
    checks = {
        "headline_improvement": {
            "observed": prim["total_gain"]["mean"], "published": LADDER_IMP,
            "tolerance": P1_TOL,
            "pass": abs(prim["total_gain"]["mean"] - LADDER_IMP) < P1_TOL},
        "slot_pooled_mae_base": {
            "observed": prim["slot_pooled"]["mae_base_all"],
            "published": LADDER_MAE_BASE, "tolerance": 1e-3,
            "pass": abs(prim["slot_pooled"]["mae_base_all"]
                        - LADDER_MAE_BASE) < 1e-3},
        "slot_pooled_mae_head": {
            "observed": prim["slot_pooled"]["mae_head_all"],
            "published": LADDER_MAE_HEAD, "tolerance": 1e-3,
            "pass": abs(prim["slot_pooled"]["mae_head_all"]
                        - LADDER_MAE_HEAD) < 1e-3},
        "floor_slice_mae_base": {
            "observed": prim["slot_pooled"]["mae_base_floor"],
            "published": FLOOR_MAE_BASE, "tolerance": 2e-3,
            "pass": abs(prim["slot_pooled"]["mae_base_floor"]
                        - FLOOR_MAE_BASE) < 2e-3},
        "floor_slice_mae_head": {
            "observed": prim["slot_pooled"]["mae_head_floor"],
            "published": FLOOR_MAE_HEAD, "tolerance": 2e-3,
            "pass": abs(prim["slot_pooled"]["mae_head_floor"]
                        - FLOOR_MAE_HEAD) < 2e-3},
        "near_floor_share_pct": {
            "observed": prim["prevalence"]["share_pct"],
            "published": NEAR_FLOOR_PCT, "tolerance": 5e-3,
            "pass": abs(prim["prevalence"]["share_pct"] - NEAR_FLOOR_PCT) < 5e-3},
    }
    # The two floor-slice checks above are anchored on the MACRO values
    # (\AbsFloorBaselineMae, \AbsFloorPreGateMae). Those macros come from
    # phase82b's RELATIVE block, which is computed on the sub-panel where the
    # naive-168h benchmark exists (10,379 floor slots), not on the full ladder
    # panel (10,504). The failure is a row-set provenance difference, not a
    # disagreement about the floor definition. Both readings are kept: the
    # macro-anchored check stays visible with its failure, and the like-for-like
    # check below is read straight out of phase82b's own artefact.
    p82b = json.loads(P82B_JSON.read_text())
    abs_floor = p82b["absolute"]["floor"]
    rel_floor = p82b["relative"]["floor"]["tiers"]
    like = {
        "floor_slice_mae_base_fullpanel": {
            "observed": prim["slot_pooled"]["mae_base_floor"],
            "phase82b_absolute_block": abs_floor[BASE_CELL]["mae"],
            "n_slots_phase82b": abs_floor[BASE_CELL]["n_slots"],
            "n_slots_here": prim["prevalence"]["n_floor_slots"],
            "pass": abs(prim["slot_pooled"]["mae_base_floor"]
                        - abs_floor[BASE_CELL]["mae"]) < 2e-3},
        "floor_slice_mae_head_fullpanel": {
            "observed": prim["slot_pooled"]["mae_head_floor"],
            "phase82b_absolute_block": abs_floor[HEAD_CELL]["mae"],
            "pass": abs(prim["slot_pooled"]["mae_head_floor"]
                        - abs_floor[HEAD_CELL]["mae"]) < 2e-3},
    }
    out["positive_control_p1"] = {
        "why": "if the panel or the floor definition is not the paper's, no share "
               "below is a share of the paper's number",
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
        "macro_anchor_diagnosis": {
            "finding": "the two floor-slice checks fail against the MACRO values "
                       "and pass exactly against phase82b's absolute block",
            "cause": "the macros 5.223 / 3.383 are the naive-available sub-panel "
                     "(phase82b relative block, 10,379 floor slots); the full "
                     "ladder panel has 10,504 floor slots and 5.2422 / 3.392",
            "macro_subpanel_values": {BASE_CELL: rel_floor[BASE_CELL]["mae"],
                                      HEAD_CELL: rel_floor[HEAD_CELL]["mae"],
                                      "n_slots": p82b["relative"]["floor"][
                                          "naive168"]["n_slots"]},
            "like_for_like_checks": like,
            "all_pass": all(v["pass"] for v in like.values()),
            "consequence": "Part I is computed on the full ladder panel, so the "
                           "shares are shares of the published 0.2529, which is "
                           "also a full-panel number",
        },
        "sources": {"phase85": str(P85_JSON), "phase82b": str(P82B_JSON),
                    "phase94": str(P94_JSON)},
    }
    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "inputs": [str(P85_PARQUET), str(P85_JSON), str(P82B_JSON),
                                  str(P94_JSON)],
                       "seed": SEED}
    return out


# ------------------------------------------------- Part II: hurdle components
_ARMS: dict[tuple[str, ...], str] = {}
_ACC: dict[str, dict] = {}
_DIAG: dict = {}


def _hurdle_hook(train: pl.DataFrame, test: pl.DataFrame,
                 cols: list[str]) -> np.ndarray:
    arm = _ARMS.get(tuple(cols))
    if arm is None:
        return np.full(test.height, np.nan)
    dts = test["datetime_jst"].to_list()
    td = dts[0].date().isoformat()
    ytr = train["price_yen_kwh"].to_numpy().astype(float)
    lab = (ytr <= TAU).astype(int)
    xtr = train.select(cols).to_numpy().astype(float)
    xte = test.select(cols).to_numpy().astype(float)
    rec = _ACC.setdefault(td, {"dt": dts,
                               "y": test["price_yen_kwh"].to_numpy().astype(float)})
    if rec["dt"] != dts:
        msg = f"arms disagree on the test rows of {td}"
        raise RuntimeError(msg)
    f_hat = float(ytr[lab == 1].mean()) if lab.sum() else TAU_ADMIN
    if "f_hat" in rec and abs(rec["f_hat"] - f_hat) > 1e-12:
        msg = "the floor-level component differs between arms of the same fold"
        raise RuntimeError(msg)
    rec["f_hat"] = f_hat
    if not lab.sum():
        _DIAG["folds_without_floor_train_rows"] = (
            _DIAG.get("folds_without_floor_train_rows", 0) + 1)

    # -- classification component: P(y <= tau | X)
    if 2 <= int(lab.sum()) < len(lab):
        clf = xgb.XGBClassifier(**XGB_CLF)
        clf.fit(xtr, lab)
        pi = clf.predict_proba(xte)[:, 1].astype(float)
    else:
        pi = np.full(test.height, float(lab.mean()))
        _DIAG["degenerate_classifier_folds"] = (
            _DIAG.get("degenerate_classifier_folds", 0) + 1)
    # -- continuous component: E[y | y > tau, X]
    nf = lab == 0
    if int(nf.sum()) < 2:
        msg = f"fold {td} has fewer than two non-floor train rows"
        raise RuntimeError(msg)
    reg = xgb.XGBRegressor(**XGB_REG)
    reg.fit(xtr[nf], ytr[nf])
    mu = reg.predict(xte).astype(float)

    # -- P2 (once per area, baseline arm): my parameters on ALL train rows must
    #    reproduce phase85's own regressor bit-for-bit.
    if arm == "base" and "p2_max_abs_diff" not in _DIAG:
        mine = xgb.XGBRegressor(**XGB_REG)
        mine.fit(xtr, ytr)
        ref = _DIAG["_orig_fit"](train, test, cols)
        diff = float(np.max(np.abs(mine.predict(xte).astype(float) - ref)))
        _DIAG["p2_max_abs_diff"] = diff
        _DIAG["p2_pass"] = bool(diff <= P2_TOL)
        if diff > P2_TOL:
            msg = (f"P2 FAILED: my regressor differs from phase85's `_fit` by "
                   f"{diff:.3e} on the first fold")
            raise RuntimeError(msg)

    rec[f"pi_{arm}"] = pi
    rec[f"mu_{arm}"] = mu
    return mu


def build(areas: list[str], *, max_folds: int = 0, acc_dir: Path | None = None,  # noqa: C901 - one fold hook plus the rebinding assertions
          ckpt_dir: Path | None = None) -> int:
    acc_dir = acc_dir or ACC
    ckpt_dir = ckpt_dir or CKPT
    mod = _load_phase85()
    if max_folds:
        mod.MAX_FOLDS = max_folds
    orig_design, orig_fit = mod._design, mod._fit
    _DIAG["_orig_fit"] = orig_fit

    def design_capturing(df: pl.DataFrame) -> tuple:
        out, cells = orig_design(df)
        for name, arm in ((BASE_CELL, "base"), (HEAD_CELL, "head")):
            if name not in cells:
                msg = f"phase85 has no cell named {name!r}"
                raise RuntimeError(msg)
            _ARMS[tuple(cells[name])] = arm
        if len(_ARMS) != 2:
            msg = "the two arms share an identical column list"
            raise RuntimeError(msg)
        return out, cells

    def fit_dispatch(train: pl.DataFrame, test: pl.DataFrame,
                     cols: list[str]) -> np.ndarray:
        return _hurdle_hook(train, test, cols)

    mod._design, mod._fit = design_capturing, fit_dispatch
    if mod._design is orig_design or mod._fit is orig_fit:
        msg = "a rebinding did not take effect"
        raise RuntimeError(msg)
    if (mod._design.__name__ != "design_capturing"
            or mod._fit.__name__ != "fit_dispatch"):
        msg = "a rebinding bound the wrong function"
        raise RuntimeError(msg)
    mod.CKPT = ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    acc_dir.mkdir(parents=True, exist_ok=True)
    for area in areas:
        target = acc_dir / f"{area}.parquet"
        if target.exists():
            log.info("component checkpoint hit %s -- skipping", target.name)
            continue
        _ACC.clear()
        for k in [k for k in _DIAG if not k.startswith("_")]:
            del _DIAG[k]
        log.info("=== %s (hurdle components, tau=%s) ===", area, TAU)
        mod._run_area(area)
        if not _ACC:
            msg = (f"{area}: no folds accumulated -- phase85's `_run_area` may "
                   "have short-circuited on a stale checkpoint")
            raise RuntimeError(msg)
        rows = []
        for td, rec in _ACC.items():
            missing = [k for k in ("pi_base", "mu_base", "pi_head", "mu_head")
                       if k not in rec]
            if missing:
                msg = f"{area} {td} is missing {missing}"
                raise RuntimeError(msg)
            rows.extend({"area": area, "datetime_jst": rec["dt"][s],
                         "test_day": td, "y_actual": float(rec["y"][s]),
                         "f_hat": rec["f_hat"],
                         "pi_base": float(rec["pi_base"][s]),
                         "pi_head": float(rec["pi_head"][s]),
                         "mu_base": float(rec["mu_base"][s]),
                         "mu_head": float(rec["mu_head"][s])}
                        for s in range(len(rec["dt"])))
        pl.DataFrame(rows).write_parquet(target)
        (acc_dir / f"{area}_diag.json").write_text(json.dumps(
            {k: v for k, v in _DIAG.items() if not k.startswith("_")}, indent=1))
        log.info("%s: wrote %d component rows", area, len(rows))
    return 0


def _hurdle_mae_daily(df: pl.DataFrame, pi: str, mu: str) -> pl.DataFrame:
    return (df.with_columns(
        ((pl.col(pi) * pl.col("f_hat") + (1.0 - pl.col(pi)) * pl.col(mu))
         - pl.col("y_actual")).abs().alias("ae"))
        .group_by("area", "test_day").agg(pl.col("ae").mean().alias("mae"))
        .sort("area", "test_day"))


def _system_daily(df: pl.DataFrame, pi: str, mu: str) -> np.ndarray:
    return _daily(_hurdle_mae_daily(df, pi, mu).rename({"mae": "v"}), "v")


def analyse() -> int:
    mod = _load_phase85()
    out = partition()
    files = sorted(ACC.glob("*.parquet"))
    if not files:
        out["status"] = "part1_only_no_components"
        out["part2_note"] = "B7: no component checkpoints; Part II not computed"
        OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        log.warning("no component checkpoints under %s", ACC)
        return 0
    comp = pl.concat([pl.read_parquet(f) for f in files]).sort("area",
                                                              "datetime_jst")
    ladder = _read_ladder_panel().select(
        "area", "datetime_jst",
        pl.col("y_actual").alias("y_ladder"),
        pl.col(f"y_pred_{BASE_CELL}").alias("ladder_base"),
        pl.col(f"y_pred_{HEAD_CELL}").alias("ladder_head"))
    j = comp.join(ladder, on=["area", "datetime_jst"], how="inner")
    if j.height != comp.height:
        msg = (f"row-set mismatch: {comp.height} component rows joined to "
               f"{j.height} ladder rows -- the panels are not the same")
        raise RuntimeError(msg)
    y_gap = float((j["y_actual"] - j["y_ladder"]).abs().max())
    if y_gap > 0.0:
        msg = f"realised prices differ from the ladder panel by up to {y_gap:.3e}"
        raise RuntimeError(msg)
    n_areas_built = int(comp["area"].n_unique())
    complete = n_areas_built == 9
    audit = _null_audit(j, ["pi_base", "pi_head", "mu_base", "mu_head", "f_hat",
                            "y_actual", "y_ladder", "ladder_base", "ladder_head"])
    if audit["total"]:
        msg = f"B8: nulls/NaNs in the hurdle panel: {audit}"
        raise RuntimeError(msg)

    # the four systems, on the paper's nested estimator over the JOINED rows
    a = _system_daily(j, "pi_base", "mu_base")
    b = _system_daily(j, "pi_head", "mu_base")
    c = _system_daily(j, "pi_base", "mu_head")
    d = _system_daily(j, "pi_head", "mu_head")
    g = a - d
    g_pi, g_mu = a - b, a - c
    g_int = g - g_pi - g_mu
    s_pi = 0.5 * ((a - b) + (c - d))
    s_mu = 0.5 * ((a - c) + (b - d))
    r_main = float(np.max(np.abs(g_pi + g_mu + g_int - g)))
    r_shap = float(np.max(np.abs(s_pi + s_mu - g)))
    if max(r_main, r_shap) > IDENTITY_TOL:
        msg = (f"decomposition does not sum: main {r_main:.3e}, "
               f"shapley {r_shap:.3e}")
        raise RuntimeError(msg)

    # the ladder's own gain on the SAME joined rows, for the fidelity check
    lad = (j.with_columns(
        ((pl.col("ladder_base") - pl.col("y_actual")).abs()
         - (pl.col("ladder_head") - pl.col("y_actual")).abs()).alias("dae"))
        .group_by("area", "test_day").agg(pl.col("dae").mean().alias("v"))
        .sort("area", "test_day"))
    lad_daily = _daily(lad, "v")
    rel = abs(float(g.mean()) - float(lad_daily.mean())) / abs(
        float(lad_daily.mean()))

    # P3: oracle floor state substituted into the baseline arm
    j_or = j.with_columns((pl.col("y_actual") <= TAU).cast(pl.Float64)
                          .alias("pi_oracle"))
    oracle = _system_daily(j_or, "pi_oracle", "mu_base")
    oracle_gain = float((a - oracle).mean())
    diag_files = {p.stem.removesuffix("_diag"): json.loads(p.read_text())
                  for p in sorted(ACC.glob("*_diag.json"))}

    part2 = {
        "status": "complete" if complete else "partial",
        "areas_built": sorted(comp["area"].unique().to_list()),
        "n_areas_built": n_areas_built,
        "area_selection": {
            "rule": "B7. Part I runs on all nine areas. Part II needs four "
                    "walk-forward fits per fold per area and the host machine was "
                    "carrying external load, so the arm was cut to areas spanning "
                    "the Part I floor-share range rather than run at nine areas "
                    "on a schedule that would not finish.",
            "disclosure": "the four-area choice was made AFTER Part I was known "
                          "(highest floor share, second, mid, lowest); per-area "
                          "numbers are reported for every area built, so no "
                          "statistic here is selected on its own value",
            "pooled_base": "every pooled Part II figure, including the ladder "
                           "gain it is compared with, is computed on exactly "
                           "these areas and these rows",
        },
        "panel": {"n_slots": int(j.height),
                  "n_test_days": int(j["test_day"].n_unique()),
                  "joined_to_ladder_rows": int(j.height),
                  "component_rows": int(comp.height)},
        "null_audit": audit,
        "components": {
            "pi": "XGBClassifier(300 x 4, lr 0.05, hist, logloss, seed 2026), "
                  "label 1{y <= 0.5}",
            "mu": "phase85's XGBRegressor parameters, trained on train rows with "
                  "y > 0.5",
            "f": "train-window mean price over rows with y <= 0.5, COMMON to both "
                 "arms",
            "yhat": "pi * f + (1 - pi) * mu",
        },
        "system_mae": {"A_base_base": _j(float(a.mean())),
                       "B_headpi_basemu": _j(float(b.mean())),
                       "C_basepi_headmu": _j(float(c.mean())),
                       "D_head_head": _j(float(d.mean()))},
        "total_hurdle_gain": _entry(mod, g),
        "main_effects": {
            "floor_state_channel_g_pi": _entry(mod, g_pi),
            "conditional_level_channel_g_mu": _entry(mod, g_mu),
            "interaction_g_int": _entry(mod, g_int),
            "identity": {"rule": "g_pi + g_mu + g_int = G",
                         "max_abs_residual": r_main, "tolerance": IDENTITY_TOL,
                         "exact": True,
                         "honesty_note": "algebraic in A,B,C,D -- verifies the "
                                         "arithmetic, not the science"},
        },
        "shapley": {
            "floor_state_channel_S_pi": _entry(mod, s_pi),
            "conditional_level_channel_S_mu": _entry(mod, s_mu),
            "identity": {"rule": "S_pi + S_mu = G",
                         "max_abs_residual": r_shap, "tolerance": IDENTITY_TOL,
                         "exact": True},
            "share_floor_state": mod._ratio_entry(s_pi, g),
            "share_conditional_level": mod._ratio_entry(s_mu, g),
        },
        "external_fidelity_b6": {
            "ladder_gain_same_rows": _j(float(lad_daily.mean())),
            "hurdle_gain": _j(float(g.mean())),
            "relative_gap": _j(rel, 3),
            "threshold": FIDELITY_MAX_REL,
            "faithful": bool(rel <= FIDELITY_MAX_REL),
            "ladder_levels": {"published_base": LADDER_MAE_BASE,
                              "published_head": LADDER_MAE_HEAD},
        },
        "positive_control_p2": {
            "why": "proves my regressor differs from the ladder's only in its "
                   "training subset, not in hyper-parameters or data",
            "max_abs_diff_vs_phase85_fit": max(
                (v.get("p2_max_abs_diff", float("nan"))
                 for v in diag_files.values()), default=None),
            "all_pass": all(v.get("p2_pass") for v in diag_files.values())
                        if diag_files else None,
            "per_area": diag_files, "tolerance": P2_TOL},
        "row_set_check": {
            "component_rows": int(comp.height), "joined_rows": int(j.height),
            "max_abs_price_gap_vs_ladder": y_gap,
            "note": "the hurdle components and the ladder forecasts are on the "
                    "identical slot set with identical realised prices"},
        "positive_control_p3": {
            "why": "perfect floor-state knowledge must buy a large MAE gain, or "
                   "the instrument cannot detect a classification channel at all",
            "oracle_mae": _j(float(oracle.mean())),
            "gain_over_A": _j(oracle_gain),
            "min_required": ORACLE_MIN_GAIN,
            "pass": bool(oracle_gain >= ORACLE_MIN_GAIN),
        },
    }
    # component diagnostics: is the weather arm's floor classifier actually
    # better? A large Brier gain with a small S_pi is itself a finding.
    fl = j.with_columns((pl.col("y_actual") <= TAU).cast(pl.Float64).alias("F"))
    on_f = fl.filter(pl.col("F") == 1.0)
    off_f = fl.filter(pl.col("F") == 0.0)

    def _m(frame: pl.DataFrame, expr: pl.Expr, nd: int = 4) -> float | None:
        """Mean of an expression, or None on an empty slice -- never a silent 0."""
        if frame.height == 0:
            return None
        return _j(float(frame.select(expr).to_series().mean()), nd)

    part2["component_diagnostics"] = {
        "n_floor_slots": int(on_f.height), "n_nonfloor_slots": int(off_f.height),
        "brier_pi_base": _m(fl, (pl.col("pi_base") - pl.col("F")) ** 2, 5),
        "brier_pi_head": _m(fl, (pl.col("pi_head") - pl.col("F")) ** 2, 5),
        "mean_pi_base_on_floor": _m(on_f, pl.col("pi_base")),
        "mean_pi_head_on_floor": _m(on_f, pl.col("pi_head")),
        "mean_pi_base_off_floor": _m(off_f, pl.col("pi_base")),
        "mean_pi_head_off_floor": _m(off_f, pl.col("pi_head")),
        "mae_mu_base_off_floor": _m(
            off_f, (pl.col("mu_base") - pl.col("y_actual")).abs()),
        "mae_mu_head_off_floor": _m(
            off_f, (pl.col("mu_head") - pl.col("y_actual")).abs()),
        "mean_f_hat": _m(fl, pl.col("f_hat")),
    }
    per_area = {}
    for area in part2["areas_built"]:
        sub = j.filter(pl.col("area") == area)
        aa = _system_daily(sub, "pi_base", "mu_base")
        bb = _system_daily(sub, "pi_head", "mu_base")
        cc = _system_daily(sub, "pi_base", "mu_head")
        dd = _system_daily(sub, "pi_head", "mu_head")
        gg = aa - dd
        sp = 0.5 * ((aa - bb) + (cc - dd))
        sm = 0.5 * ((aa - cc) + (bb - dd))
        lad_a = _daily(
            sub.with_columns(
                ((pl.col("ladder_base") - pl.col("y_actual")).abs()
                 - (pl.col("ladder_head") - pl.col("y_actual")).abs()).alias("dae"))
            .group_by("area", "test_day").agg(pl.col("dae").mean().alias("v"))
            .sort("test_day"), "v")
        per_area[area] = {
            "total_hurdle_gain": _j(float(gg.mean())),
            "ladder_gain_same_rows": _j(float(lad_a.mean())),
            "S_pi": _j(float(sp.mean())), "S_mu": _j(float(sm.mean())),
            "share_floor_state": mod._ratio_entry(sp, gg),
            "system_mae": {"A": _j(float(aa.mean())), "B": _j(float(bb.mean())),
                           "C": _j(float(cc.mean())), "D": _j(float(dd.mean()))},
            "n_days": len(gg),
        }
    part2["per_area"] = per_area
    out["part2_hurdle_decomposition"] = part2
    out["status"] = "complete" if complete else "part2_partial"

    # ---- verdict per the pre-registered branches
    s_f = out["part1_state_partition"]["tau_0.5"]["floor_share"]["ratio"]
    branch = "B1" if s_f >= 2 / 3 else ("B3" if s_f <= 1 / 3 else "B2")
    pi_share = part2["shapley"]["share_floor_state"]["ratio"]
    branch2 = "B4" if (pi_share is not None and pi_share >= 0.5) else "B5"
    out["branches_fired"] = {
        "part1": branch, "part1_floor_share": s_f,
        "part2": branch2, "part2_floor_state_share": pi_share,
        "b6_fidelity": "faithful" if part2["external_fidelity_b6"]["faithful"]
                       else "B6 fired -- Part II describes the hurdle system only",
        "b7_completeness": part2["status"],
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("Part I: floor share %s of %.4f; Part II: S_pi %.4f / S_mu %.4f "
             "(G %.4f, ladder %.4f)", s_f,
             out["part1_state_partition"]["tau_0.5"]["total_gain"]["mean"],
             float(s_pi.mean()), float(s_mu.mean()), float(g.mean()),
             float(lad_daily.mean()))
    log.info("branches: %s", out["branches_fired"])
    log.info("wrote %s", OUT_JSON)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("partition", "build", "analyse"),
                    required=True)
    ap.add_argument("--areas", default="")
    ap.add_argument("--max-folds", type=int, default=0,
                    help="smoke only: cap phase85's fold loop")
    args = ap.parse_args()
    if args.step == "partition":
        out = partition()
        OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        log.info("P1 all_pass=%s", out["positive_control_p1"]["all_pass"])
        log.info("wrote %s", OUT_JSON)
        return 0
    if args.step == "build":
        mod = _load_phase85()
        areas = [a for a in args.areas.split(",") if a] or list(mod.AREA_COORDS)
        if args.max_folds:
            # A capped run is a timing probe, never a reportable artefact, so it
            # writes to separate directories that `analyse` never reads.
            log.warning("SMOKE: capping folds at %d -- not a reportable run",
                        args.max_folds)
            return build(areas, max_folds=args.max_folds,
                         acc_dir=RUN / "phase103_smoke_components",
                         ckpt_dir=RUN / "phase103_smoke_checkpoints")
        return build(areas)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
