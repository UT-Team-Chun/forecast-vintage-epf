r"""Phase 103b: the FITTED hurdle decomposition of the headline weather gain.

QUESTION
--------
The manuscript's third claim (docs/paper/paper_a_main/sections/ae_05_results.tex,
subsec:accuracy) is that the weather value concentrates in the floor regime: "on
near-floor slots ... a gain of about \FloorAbsGain (1.84) JPY/kWh, roughly seven
times the pooled \PreGateImp (0.253) the ladder reports". That is an INTENSITY
statement on a slice. The referee asked for the IDENTIFICATION: how much of the
TOTAL improvement runs through the floor channel (predicting whether the slot is
at/near the floor at all) versus the continuous-price channel (predicting the
level, given that it is not at the floor)?

WHAT IS ALREADY KNOWN, AND WHAT IS NOT (disclosed, because it bounds what this
pre-registration can honestly claim)
-----------------------------------------------------------------------------
phase103 (scripts/32b_floor_channel_decomposition.py,
artefact data/runs/phase103_floor_channel_decomposition.json, status
"part1_only") already answered the STATE question with an exact, refit-free
partition of the published 0.2529: floor slots (y <= 0.5 JPY/kWh, 3.33 % of
slots) carry 0.0615 of it, a share of 0.243 [0.192, 0.302]. Its Part II -- the
fitted hurdle system, which is the referee's actual ask -- was written but NEVER
RUN: data/runs/phase103_hurdle_components/ is empty. This phase runs it, on all
nine areas, and reports the result.

So the 0.243 state share is known to me before this script executes and is NOT
what is being pre-registered here. What is genuinely unknown, and what the
branches below commit to in advance, is the FITTED channel split: a floor state
can carry a quarter of the gain in realised-state accounting while the
information that produces the gain enters entirely through the conditional level,
or vice versa. Nothing in Part I implies either.

DEFINITIONS -- the paper's own, not invented here
------------------------------------------------
tau = 0.5 JPY/kWh. This is \thrFloorDefault in docs/paper/shared/macros.tex, the
   "near_floor_threshold" of phase82b (data/runs/phase82b_absolute_metrics.json)
   and the band phase94 counts at \EvalNearFloorPct = 3.33 % of slots. It is the
   band the seven-times sentence itself is computed on. F_i = 1{y_i <= tau} on
   the REALISED price.
tau_admin = 0.01 JPY/kWh, the administrative floor, used only as the fallback
   floor level when a fold's training window contains no floor row (counted).
Panel / folds / cells / learner: phase85's CW-primary ladder VERBATIM
   (10_primary_ladder_cw.py): 732 daily walk-forward folds x 9
   areas = 315,414 half-hour slots, baseline cell `price_only`, headline cell
   `msm_d2_12z_cw`, XGBRegressor(300 x depth 4, lr 0.05, hist, n_jobs 4, seed
   2026). Reused by importing phase103, which rebinds phase85's `_design` and
   `_fit` and asserts the rebinding took (phase96's pattern). This script asserts
   in turn that ITS rebinding of phase103's output paths took effect, so that a
   silent no-op cannot make it look like phase103's own artefact was refreshed.
Aggregation: the PAPER'S estimator, phase85's `_pooled_daily` -- the mean over
   test days of the cross-area mean of per-(area, day) MAE differences. Every
   share below is therefore a share OF THE PUBLISHED 0.2529.

THE ALGEBRA (stated explicitly; an ad hoc split is not a decomposition)
----------------------------------------------------------------------
Hurdle (two-part) system. With pi = P(y <= tau | X) the floor-state component,
mu = E[y | y > tau, X] the conditional-level component and f the floor level,

    yhat(pi, mu) = pi * f + (1 - pi) * mu.

Per fold and area, on the SAME train rows phase85 uses, both arms fit
    pi^b, pi^h : XGBClassifier(phase85's regressor parameters + logloss), label
                 1{y <= tau}, on the baseline / headline column set;
    mu^b, mu^h : phase85's OWN XGBRegressor parameters, on the train rows with
                 y > tau, target y;
    f          : mean of y over train rows with y <= tau. COMMON TO BOTH ARMS and
                 never re-estimated per arm, so f cannot absorb any part of the
                 contrast (fallback tau_admin if a fold has no floor train row).

Four systems, whose per-slot absolute errors are ae_A .. ae_D:
    A = (pi^b, mu^b)   B = (pi^h, mu^b)   C = (pi^b, mu^h)   D = (pi^h, mu^h).
Per SLOT, the symmetric (Shapley) split of the hurdle improvement is
    s_pi(i) = 0.5 * [ (ae_A(i) - ae_B(i)) + (ae_C(i) - ae_D(i)) ]
    s_mu(i) = 0.5 * [ (ae_A(i) - ae_C(i)) + (ae_B(i) - ae_D(i)) ]
    s_pi(i) + s_mu(i) = ae_A(i) - ae_D(i)                       (identically),
and the main-effects reading of the same four numbers is
    g_pi(i) = ae_A(i) - ae_B(i),  g_mu(i) = ae_A(i) - ae_C(i),
    g_int(i) = (ae_A - ae_D) - g_pi - g_mu = ae_B + ae_C - ae_A - ae_D,
    g_pi + g_mu + g_int = ae_A - ae_D                           (identically).

Bridging to the PAPER'S number. The published gain is a ladder gain, not a hurdle
gain, so a decomposition of the hurdle system alone would not be a decomposition
of 0.2529. Write the per-slot ladder gain d(i) = |y_i - b_i| - |y_i - h_i| and
define the reconstruction term
    r(i) = d(i) - (ae_A(i) - ae_D(i)),
which is what the two-part rebuild fails to reproduce of the ladder's own gain.
Then, per slot and hence per (area, day) cell after dividing sums by the cell's
own slot count n_ad, and hence under the paper's linear nested estimator,

    Delta_paper = S_pi + S_mu + R                                    (EXACT)
                = G_pi + G_mu + G_int + R                            (EXACT)

and, crossing each channel with the realised state F_i, the full 2x2 + residual

    Delta_paper = S_pi^F + S_pi^NF + S_mu^F + S_mu^NF + R            (EXACT).

Each part is a SUM over its own slots divided by the cell's slot count, never a
conditional mean, so every part is defined even for cells with no floor slot and
the identity holds slot by slot. All identities are asserted to 1e-9 and the
maximum absolute residual of each is reported. R is reported as a term of the
decomposition, not hidden: if R is large, the honest reading is that the fitted
two-part system does not reproduce the ladder and the channel shares describe the
hurdle system rather than the published number.

HONESTY NOTE ON "EXACT". The identities are algebraic in ae_A..ae_D and d, so
asserting them verifies arithmetic, not science. The substantive checks are the
positive controls below and the size of R.

PRE-REGISTERED DECISION BRANCHES (fixed before any number was computed)
----------------------------------------------------------------------
Primary quantity: the floor-state channel share of the PUBLISHED gain,
sh_pi = S_pi / Delta_paper (Shapley reading, pooled, block-bootstrap CI).
C1 sh_pi >= 1/2       -> the improvement runs mainly through better floor-state
                         prediction; the third claim holds as an identification
                         statement and not only as a slice intensity.
C2 1/4 <= sh_pi < 1/2 -> MIXED. Both channels carry the gain; report both halves
                         and refuse the strong reading.
C3 sh_pi < 1/4        -> the floor channel is a MINORITY conduit. This
                         CONTRADICTS a strong reading of the third claim and is
                         reported as the headline if it fires, together with the
                         floor prevalence, with the explicit statement that a 7x
                         intensity on a 3.33 % slice cannot carry the aggregate.
C4 sh_pi < 0 or sh_mu < 0 -> a channel is harmful on net; report the sign and do
                         not present a negative part as a "share" without it.
C5 |R| / Delta_paper > 0.5 -> the two-part rebuild is not a faithful stand-in for
                         the ladder. The channel shares are then reported as
                         shares of the hurdle gain G only, labelled as such in
                         the artefact and the return, and the verdict on the
                         paper's number falls back to phase103's Part I.
C6 P3 (oracle) fails  -> the instrument cannot detect a classification channel at
                         all; any null in C3 is UNINTERPRETABLE and the verdict
                         is UNDETERMINED, not NULL.
C7 fewer than nine areas built -> pooled figures are computed on exactly the
                         areas built, the area list is reported, and per-area
                         figures are given for every area built, so that no
                         statistic is selected on its own value.
C8 any null/NaN in a required column -> abort and report the null counts. Nulls
                         are counted and reported in EVERY branch, including
                         success.
C9 floor prevalence in the built panel departs from phase94's 3.33 % by more than
                         0.01 pp -> the floor definition or the panel is not the
                         paper's; report and stop.

POSITIVE CONTROLS
-----------------
P1 (panel identity) Re-derive the ladder's published headline from phase85's
   per-slot parquet under the paper's estimator and require
   |Delta - 0.2529| < 5e-4, plus the pooled MAE levels 2.1506 / 1.8989 and the
   near-floor share 3.33 %. Executed by reusing phase103's `partition()` whole,
   which also carries phase103's disclosed A1 amendment (the two floor-slice
   macros 5.223 / 3.383 are the naive-available sub-panel, 10,379 floor slots,
   while the ladder panel has 10,504; the like-for-like check against phase82b's
   absolute block passes). If P1 fails, no share here is a share of the paper's
   number.
P2 (parameter fidelity) On the first fold of each area, fit MY regressor
   parameters on ALL train rows and require the prediction to equal phase85's own
   `_fit` output to 1e-9. This proves the conditional-level component differs
   from the ladder's regressor only in its training subset -- not in
   hyper-parameters, thread count, seed or data. (Implemented inside phase103's
   hook; verified here from the per-area diagnostics.)
P3 (instrument sensitivity) Substitute the ORACLE floor state pi* = F_i into the
   baseline arm. Perfect floor-state knowledge must buy a large positive gain
   (>= 0.05 JPY/kWh under the paper's estimator). If it does not, the
   decomposition cannot see a classification channel and C6 fires.
P4 (cross-path agreement) The floor part of the state accounting computed here on
   the built rows must equal phase103's Part I floor contribution to 1e-4 when
   all nine areas are built -- two independently written code paths over the same
   definition.

AMENDMENTS AFTER FIRST COMPUTATION (disclosed, not backdated)
------------------------------------------------------------
B1 A smoke run (2 areas x 25 folds, separate directories, throwaway JSON) was
   executed after this pre-registration and before the full build, purely to
   verify the pipeline. Its P3 and P4 both FAILED because the first 25 August
   folds contain almost no floor slots -- which is exactly what a control that
   can fail looks like. No branch, threshold or definition was changed as a
   result.
B2 A mid-flight partial analysis on the first three areas that finished
   (kansai, kyushu, tohoku) was inspected before the remaining six had built.
   After seeing it, ONE thing was added: `totals.in_jepx_ticks`, a unit
   conversion of quantities already pre-registered above (phase90's lesson that
   a share is not "small" until converted into the paper's units). No statistic,
   threshold, branch or definition was added, removed or changed, and the
   partial numbers are reported in the return alongside the final ones so that
   the effect of adding the other six areas is visible rather than hidden.

B3 After the pooled result was known, ONE exploratory statistic was added
   and labelled as such in the artefact: the cross-area Spearman rank
   correlation between an area's floor prevalence and its floor-channel
   share. It decides no branch and is not a headline; it is reported
   because the per-area table ordered itself that way and a referee will
   ask. n = 9 correlated areas.

CLI (from the repository root):
  S=scripts/32_floor_channel_hurdle.py
  uv run python $S --step build --areas kyushu,tokyo
  uv run python $S --step analyse
Build is per-area checkpointed and safe to run in several processes over
disjoint --areas lists.
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
from scipy import stats

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase103b")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
HERE = Path(__file__).parent
P103_PATH = HERE / "32b_floor_channel_decomposition.py"
P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
P103_JSON = RUN / "phase103_floor_channel_decomposition.json"
ACC = RUN / "phase103b_hurdle_components"       # my per-area component parquets
CKPT = RUN / "phase103b_ladder_checkpoints"     # phase85 harness output (ignored)
OUT_JSON = RUN / "phase103b_floor_channel_hurdle.json"

BASE_CELL = "price_only"
HEAD_CELL = "msm_d2_12z_cw"
TAU = 0.5
IDENTITY_TOL = 1e-9
P1_TOL = 5e-4
ORACLE_MIN_GAIN = 0.05      # P3
FIDELITY_MAX_REL = 0.5      # C5
NEAR_FLOOR_PCT = 3.33       # phase94 / \EvalNearFloorPct
PREVALENCE_TOL_PP = 0.01    # C9
LADDER_IMP = 0.2529
CROSS_PATH_TOL = 1e-4       # P4


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,  # noqa: S607
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_p103() -> object:
    """Import phase103 and REDIRECT its outputs, asserting the rebinding took.

    phase103's own artefact is another agent's committed output; a run of this
    script must not write to it. The three module globals below are the only
    paths phase103 writes, and the assertion is here because a silent no-op
    rebinding is exactly the failure mode phase91 recorded.
    """
    spec = importlib.util.spec_from_file_location("p103", P103_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p103"] = mod
    spec.loader.exec_module(mod)
    orig = (mod.ACC, mod.CKPT, mod.OUT_JSON)
    mod.ACC, mod.CKPT, mod.OUT_JSON = ACC, CKPT, OUT_JSON
    if orig == (mod.ACC, mod.CKPT, mod.OUT_JSON):
        msg = "phase103 output-path rebinding did not take effect"
        raise RuntimeError(msg)
    for got, want in ((mod.ACC, ACC), (mod.CKPT, CKPT), (mod.OUT_JSON, OUT_JSON)):
        if got != want:
            msg = f"rebinding bound the wrong path: {got} != {want}"
            raise RuntimeError(msg)
    if mod.TAU != TAU or mod.BASE_CELL != BASE_CELL or mod.HEAD_CELL != HEAD_CELL:
        msg = "phase103 constants differ from this script's; the panels would not match"
        raise RuntimeError(msg)
    return mod


def _j(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# --------------------------------------------------------------------- build
def build(areas: list[str]) -> int:
    """Run phase103's component build verbatim into MY checkpoint directories."""
    mod = _load_p103()
    ACC.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    return int(mod.build(areas, acc_dir=ACC, ckpt_dir=CKPT))


# ------------------------------------------------------------------ analysis
_CHANNELS = ("s_pi", "s_mu")
_STATES = ("F", "NF")


def _slot_frame(j: pl.DataFrame) -> pl.DataFrame:
    """Per-slot absolute errors of the four hurdle systems and of the ladder."""
    def yhat(pi: str, mu: str) -> pl.Expr:
        return pl.col(pi) * pl.col("f_hat") + (1.0 - pl.col(pi)) * pl.col(mu)

    out = j.with_columns(
        (yhat("pi_base", "mu_base") - pl.col("y_actual")).abs().alias("ae_A"),
        (yhat("pi_head", "mu_base") - pl.col("y_actual")).abs().alias("ae_B"),
        (yhat("pi_base", "mu_head") - pl.col("y_actual")).abs().alias("ae_C"),
        (yhat("pi_head", "mu_head") - pl.col("y_actual")).abs().alias("ae_D"),
        ((pl.col("ladder_base") - pl.col("y_actual")).abs()
         - (pl.col("ladder_head") - pl.col("y_actual")).abs()).alias("d_ladder"),
        (pl.col("y_actual") <= TAU).alias("is_floor"),
    )
    return out.with_columns(
        (0.5 * ((pl.col("ae_A") - pl.col("ae_B"))
                + (pl.col("ae_C") - pl.col("ae_D")))).alias("s_pi"),
        (0.5 * ((pl.col("ae_A") - pl.col("ae_C"))
                + (pl.col("ae_B") - pl.col("ae_D")))).alias("s_mu"),
        (pl.col("ae_A") - pl.col("ae_B")).alias("g_pi"),
        (pl.col("ae_A") - pl.col("ae_C")).alias("g_mu"),
        (pl.col("ae_B") + pl.col("ae_C") - pl.col("ae_A")
         - pl.col("ae_D")).alias("g_int"),
        (pl.col("d_ladder") - (pl.col("ae_A") - pl.col("ae_D"))).alias("r_recon"),
    )


_PARTS = ("s_pi", "s_mu", "g_pi", "g_mu", "g_int", "r_recon",
          "s_pi_F", "s_pi_NF", "s_mu_F", "s_mu_NF")


def _per_cell(sl: pl.DataFrame) -> pl.DataFrame:
    """Per-(area, day) contributions: SUM over the cell's own slots / n_slots.

    Not conditional means -- that is what makes the parts additive and defined
    for cells with no floor slot.
    """
    crossed = [
        (pl.when(pl.col("is_floor")).then(pl.col(c)).otherwise(0.0)
         ).sum().alias(f"sum_{c}_F") for c in _CHANNELS
    ] + [
        (pl.when(~pl.col("is_floor")).then(pl.col(c)).otherwise(0.0)
         ).sum().alias(f"sum_{c}_NF") for c in _CHANNELS
    ]
    agg = (sl.group_by("area", "test_day").agg(
        pl.len().alias("n_slots"),
        pl.col("d_ladder").mean().alias("delta"),
        pl.col("ae_A").mean().alias("mae_A"), pl.col("ae_B").mean().alias("mae_B"),
        pl.col("ae_C").mean().alias("mae_C"), pl.col("ae_D").mean().alias("mae_D"),
        pl.col("is_floor").sum().alias("n_floor"),
        *[pl.col(c).sum().alias(f"sum_{c}")
          for c in ("s_pi", "s_mu", "g_pi", "g_mu", "g_int", "r_recon")],
        *crossed)
        .sort("area", "test_day"))
    return agg.with_columns(
        *[(pl.col(f"sum_{p}") / pl.col("n_slots")).alias(p) for p in _PARTS])


def _daily(per_cell: pl.DataFrame, col: str) -> np.ndarray:
    """phase85's nested estimator: cross-area mean within a day, ordered by day."""
    return np.asarray(
        (per_cell.group_by("test_day").agg(pl.col(col).mean())
         .sort("test_day")[col]), dtype=float)


def _check_identity(name: str, total: np.ndarray, parts: list[np.ndarray],
                    identities: dict) -> None:
    resid = float(np.max(np.abs(total - sum(parts))))
    identities[name] = {"max_abs_residual": resid, "tolerance": IDENTITY_TOL,
                        "exact": bool(resid <= IDENTITY_TOL)}
    if resid > IDENTITY_TOL:
        msg = f"{name} does not sum: max |resid| = {resid:.3e}"
        raise RuntimeError(msg)


def _entry(mod85: object, d: np.ndarray) -> dict:
    lo, hi = mod85._block_boot_ci(d)
    return {"mean": _j(float(d.mean())), "ci95": [_j(lo), _j(hi)],
            "dm_l3": mod85._hln_dm(d, lag=3), "n_days": len(d)}


def _null_audit(frame: pl.DataFrame, cols: list[str]) -> dict:
    audit = {c: int(frame[c].null_count()
                    + (frame[c].is_nan().sum() if frame[c].dtype.is_float() else 0))
             for c in cols}
    return {"per_column_nulls_or_nans": audit, "total": int(sum(audit.values()))}


def analyse() -> dict:  # noqa: C901, PLR0912 - one block per pre-registered branch
    mod103 = _load_p103()
    mod85 = mod103._load_phase85()

    files = sorted(ACC.glob("*.parquet"))
    if not files:
        msg = f"no component checkpoints under {ACC}; run --step build first"
        raise RuntimeError(msg)
    comp = pl.concat([pl.read_parquet(f) for f in files]).sort("area", "datetime_jst")
    ladder = mod103._read_ladder_panel().select(
        "area", "datetime_jst", pl.col("y_actual").alias("y_ladder"),
        pl.col(f"y_pred_{BASE_CELL}").alias("ladder_base"),
        pl.col(f"y_pred_{HEAD_CELL}").alias("ladder_head"))
    j = comp.join(ladder, on=["area", "datetime_jst"], how="inner")
    if j.height != comp.height:
        msg = (f"row-set mismatch: {comp.height} component rows joined to {j.height} "
               "ladder rows -- the panels are not the same")
        raise RuntimeError(msg)
    y_gap = float((j["y_actual"] - j["y_ladder"]).abs().max())
    if y_gap > 0.0:
        msg = f"realised prices differ from the ladder panel by {y_gap:.3e}"
        raise RuntimeError(msg)

    # ---- C8: nulls never become a result silently
    audit = _null_audit(j, ["pi_base", "pi_head", "mu_base", "mu_head", "f_hat",
                            "y_actual", "y_ladder", "ladder_base", "ladder_head"])
    if audit["total"]:
        msg = f"C8: nulls/NaNs in the hurdle panel: {audit}"
        raise RuntimeError(msg)

    areas_built = sorted(comp["area"].unique().to_list())
    complete = len(areas_built) == 9

    sl = _slot_frame(j)
    cell = _per_cell(sl)
    delta = _daily(cell, "delta")
    parts = {p: _daily(cell, p) for p in _PARTS}
    identities: dict = {}
    _check_identity("shapley_plus_reconstruction",
                    delta, [parts["s_pi"], parts["s_mu"], parts["r_recon"]],
                    identities)
    _check_identity("main_effects_plus_interaction_plus_reconstruction", delta,
                    [parts["g_pi"], parts["g_mu"], parts["g_int"],
                     parts["r_recon"]], identities)
    _check_identity("channel_x_state_plus_reconstruction", delta,
                    [parts["s_pi_F"], parts["s_pi_NF"], parts["s_mu_F"],
                     parts["s_mu_NF"], parts["r_recon"]], identities)
    identities["note"] = ("algebraic in ae_A..ae_D and the ladder errors: this "
                          "verifies the arithmetic, not the science. The "
                          "substantive checks are P1-P4 and the size of R.")

    # ---- the four systems and the ladder, all on the same rows
    mae = {k: float(_daily(cell, f"mae_{k}").mean()) for k in ("A", "B", "C", "D")}
    g_total = _daily(cell, "s_pi") + _daily(cell, "s_mu")   # == mae_A - mae_D daily
    rel_r = abs(float(parts["r_recon"].mean())) / abs(float(delta.mean()))
    faithful = rel_r <= FIDELITY_MAX_REL

    # ---- prevalence (C9): the seven-times sentence is an intensity on this slice
    n_floor = int(sl["is_floor"].sum())
    prev_pct = 100.0 * n_floor / sl.height
    prevalence = {"n_floor_slots": n_floor, "n_slots": int(sl.height),
                  "share_pct": _j(prev_pct, 3), "phase94_published_pct": NEAR_FLOOR_PCT,
                  "within_tol_pp": bool(abs(prev_pct - NEAR_FLOOR_PCT)
                                        <= PREVALENCE_TOL_PP)}

    # ---- shares, of the PUBLISHED gain and of the hurdle gain
    def shares(den: np.ndarray, keys: tuple[str, ...]) -> dict:
        return {k: mod85._ratio_entry(parts[k], den) for k in keys}

    share_keys = ("s_pi", "s_mu", "r_recon", "s_pi_F", "s_pi_NF", "s_mu_F", "s_mu_NF")
    sh_of_delta = shares(delta, share_keys)
    sh_of_g = {k: mod85._ratio_entry(parts[k], g_total) for k in ("s_pi", "s_mu")}

    # ---- P3: oracle floor state into the baseline arm
    oracle = (j.with_columns((pl.col("y_actual") <= TAU).cast(pl.Float64)
                             .alias("pi_oracle"))
              .with_columns(((pl.col("pi_oracle") * pl.col("f_hat")
                              + (1.0 - pl.col("pi_oracle")) * pl.col("mu_base"))
                             - pl.col("y_actual")).abs().alias("ae"))
              .group_by("area", "test_day").agg(pl.col("ae").mean().alias("mae"))
              .sort("area", "test_day"))
    oracle_daily = _daily(oracle, "mae")
    oracle_gain = float((_daily(cell, "mae_A") - oracle_daily).mean())

    # ---- P2, from the per-area build diagnostics
    diag = {p.stem.removesuffix("_diag"): json.loads(p.read_text())
            for p in sorted(ACC.glob("*_diag.json"))}
    p2 = {"why": "the conditional-level component differs from the ladder's "
                 "regressor only in its training subset",
          "tolerance": 1e-9,
          "max_abs_diff_vs_phase85_fit": max(
              (v.get("p2_max_abs_diff", float("nan")) for v in diag.values()),
              default=None),
          "all_pass": bool(diag) and all(v.get("p2_pass") for v in diag.values()),
          "per_area": diag}

    # ---- P1 + phase103's Part I, reused whole (this is the paper's denominator)
    part1 = mod103.partition()
    p1 = part1["positive_control_p1"]
    prim = part1["part1_state_partition"]["tau_0.5"]

    # ---- P1 restated here, so this artefact carries its own anchor check
    p1_anchor = {"observed": prim["total_gain"]["mean"], "published": LADDER_IMP,
                 "tolerance": P1_TOL,
                 "pass": bool(abs(prim["total_gain"]["mean"] - LADDER_IMP) < P1_TOL)}

    # ---- P4: my state accounting vs phase103's, two independent code paths
    ladder_floor_here = _daily(
        _per_cell(sl.with_columns(pl.col("d_ladder").alias("s_pi"),
                                  pl.lit(0.0).alias("s_mu"))), "s_pi_F")
    p4 = {"why": "the floor part of the ladder gain, computed here by a different "
                 "code path, must equal phase103's Part I floor contribution",
          "here": _j(float(ladder_floor_here.mean())),
          "phase103_part1": prim["floor_contribution"]["mean"],
          "tolerance": CROSS_PATH_TOL,
          "applicable": complete,
          "pass": bool(complete and abs(float(ladder_floor_here.mean())
                                        - prim["floor_contribution"]["mean"])
                       <= CROSS_PATH_TOL)}

    # ---- component diagnostics: is the headline arm's floor classifier better?
    fl = sl.with_columns(pl.col("is_floor").cast(pl.Float64).alias("F"))
    on_f, off_f = fl.filter(pl.col("F") == 1.0), fl.filter(pl.col("F") == 0.0)

    def _m(frame: pl.DataFrame, expr: pl.Expr, nd: int = 4) -> float | None:
        if frame.height == 0:
            return None
        return _j(float(frame.select(expr).to_series().mean()), nd)

    def _f1(col: str, thr: float = 0.5) -> float | None:
        pred = fl.select((pl.col(col) >= thr).alias("p"), pl.col("F"))
        tp = float(pred.filter((pl.col("p")) & (pl.col("F") == 1.0)).height)
        fp = float(pred.filter((pl.col("p")) & (pl.col("F") == 0.0)).height)
        fn = float(pred.filter((~pl.col("p")) & (pl.col("F") == 1.0)).height)
        return _j(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else None

    comp_diag = {
        "n_floor_slots": int(on_f.height), "n_nonfloor_slots": int(off_f.height),
        "brier_pi_base": _m(fl, (pl.col("pi_base") - pl.col("F")) ** 2, 5),
        "brier_pi_head": _m(fl, (pl.col("pi_head") - pl.col("F")) ** 2, 5),
        "floor_f1_at_0.5_pi_base": _f1("pi_base"),
        "floor_f1_at_0.5_pi_head": _f1("pi_head"),
        "mean_pi_base_on_floor": _m(on_f, pl.col("pi_base")),
        "mean_pi_head_on_floor": _m(on_f, pl.col("pi_head")),
        "mean_pi_base_off_floor": _m(off_f, pl.col("pi_base")),
        "mean_pi_head_off_floor": _m(off_f, pl.col("pi_head")),
        "mae_mu_base_off_floor": _m(off_f, (pl.col("mu_base")
                                            - pl.col("y_actual")).abs()),
        "mae_mu_head_off_floor": _m(off_f, (pl.col("mu_head")
                                            - pl.col("y_actual")).abs()),
        "mean_f_hat": _m(fl, pl.col("f_hat")),
        "note": "a large Brier gain with a small S_pi is itself a finding: the "
                "floor-state signal can be real and still carry little of the "
                "aggregate MAE gain, because floor slots are rare",
    }

    # ---- per area
    per_area = {}
    for area in areas_built:
        sub = cell.filter(pl.col("area") == area).sort("test_day")
        d_a = np.asarray(sub["delta"], dtype=float)
        pa = {p: np.asarray(sub[p], dtype=float) for p in _PARTS}
        g_a = pa["s_pi"] + pa["s_mu"]
        per_area[area] = {
            "total_gain_ladder": _j(float(d_a.mean())),
            "total_gain_hurdle": _j(float(g_a.mean())),
            "S_pi": _j(float(pa["s_pi"].mean())),
            "S_mu": _j(float(pa["s_mu"].mean())),
            "R_recon": _j(float(pa["r_recon"].mean())),
            "share_floor_state_of_ladder": mod85._ratio_entry(pa["s_pi"], d_a),
            "share_cond_level_of_ladder": mod85._ratio_entry(pa["s_mu"], d_a),
            "share_reconstruction_of_ladder": mod85._ratio_entry(pa["r_recon"], d_a),
            "share_floor_state_of_hurdle": mod85._ratio_entry(pa["s_pi"], g_a),
            "S_pi_on_floor_slots": _j(float(pa["s_pi_F"].mean())),
            "S_mu_on_floor_slots": _j(float(pa["s_mu_F"].mean())),
            "floor_prevalence_pct": _j(100.0 * float(
                sl.filter(pl.col("area") == area)["is_floor"].mean()), 3),
            "system_mae": {k: _j(float(np.asarray(sub[f"mae_{k}"],
                                                  dtype=float).mean()))
                           for k in ("A", "B", "C", "D")},
            "n_days": len(d_a),
        }

    # ---- EXPLORATORY, POST HOC (amendment B3). Computed AFTER the pre-registered
    # pooled result was known, on seeing that the per-area floor-channel share
    # ordered itself by floor prevalence. It decides no branch and licenses no
    # headline; n = 9 areas whose daily improvement series are correlated (the
    # manuscript puts their effective independence near 5), so the permutation p
    # below is optimistic about how many independent draws it has.
    explor = None
    if len(areas_built) >= 3:
        prev_v = np.array([per_area[a]["floor_prevalence_pct"] for a in areas_built],
                          dtype=float)
        sh_v = np.array([per_area[a]["share_floor_state_of_ladder"]["ratio"]
                         for a in areas_built], dtype=float)
        rho = float(stats.spearmanr(prev_v, sh_v).statistic)
        # Spearman on tie-free vectors is Pearson on the ranks, so the
        # permutation null is a vectorised rank shuffle rather than 100k scipy
        # calls. Ties would break that equality, so they are checked, not assumed.
        n_a = len(areas_built)
        ties = bool(len(set(prev_v.tolist())) < n_a or len(set(sh_v.tolist())) < n_a)
        r_x = stats.rankdata(prev_v)
        r_y = stats.rankdata(sh_v)
        rng = np.random.default_rng(2026)
        n_perm = 100_000
        perm = np.argsort(rng.random((n_perm, n_a)), axis=1)
        ry_p = r_y[perm]
        xc = r_x - r_x.mean()
        yc = ry_p - ry_p.mean(axis=1, keepdims=True)
        null = (yc @ xc) / np.sqrt((yc ** 2).sum(axis=1) * (xc ** 2).sum())
        p_perm = float((np.abs(null) >= abs(rho) - 1e-12).mean())
        explor = {
            "status": "EXPLORATORY, post hoc -- decides no pre-registered branch",
            "question": "does the floor-state channel's share of the gain track "
                        "the area's floor prevalence?",
            "spearman_rho_prevalence_vs_floor_channel_share": _j(rho, 3),
            "permutation_p_two_sided": _j(p_perm, 5),
            "n_permutations": n_perm, "n_areas": n_a,
            "ties_present": ties,
            "note": "areas are not independent replications: the manuscript puts "
                    "the effective independence of the nine per-area improvement "
                    "series near five, so this p is optimistic",
        }

    # ---- branches
    sh_pi = sh_of_delta["s_pi"]["ratio"]
    sh_mu = sh_of_delta["s_mu"]["ratio"]
    fired = []
    if sh_pi is None or sh_mu is None:
        msg = "a channel share is not finite -- the denominator collapsed"
        raise RuntimeError(msg)
    if not faithful:
        fired.append("C5")
    if oracle_gain < ORACLE_MIN_GAIN:
        fired.append("C6")
    if not complete:
        fired.append("C7")
    if not prevalence["within_tol_pp"]:
        fired.append("C9")
    if sh_pi is not None and (sh_pi < 0 or (sh_mu is not None and sh_mu < 0)):
        fired.append("C4")
    primary = ("C1" if sh_pi >= 0.5 else ("C2" if sh_pi >= 0.25 else "C3"))
    fired.append(primary)

    if "C6" in fired:
        verdict = ("UNDETERMINED -- the oracle control did not clear, so a small "
                   "floor-state channel cannot be distinguished from an "
                   "instrument that cannot see one.")
    elif "C5" in fired:
        verdict = ("MIXED/UNDETERMINED on the paper's number -- the two-part "
                   "rebuild does not reproduce the ladder gain; the channel "
                   "shares describe the hurdle system only.")
    elif primary == "C1":
        verdict = ("SUPPORTED -- most of the pooled improvement runs through "
                   "better floor-state prediction.")
    elif primary == "C2":
        verdict = ("MIXED -- both channels carry material parts of the pooled "
                   "improvement; the strong reading of the third claim fails.")
    else:
        verdict = ("NULL on the strong reading -- the floor-state channel is a "
                   "minority conduit of the pooled improvement.")

    out = {
        "phase": "103b",
        "title": "Fitted hurdle (two-part) decomposition of the headline weather "
                 "gain: floor-state channel vs continuous-level channel",
        "status": "complete" if complete else "partial_areas",
        "verdict": verdict,
        "branches_fired": fired,
        "primary_branch": primary,
        "relation_to_phase103": {
            "part1_state_partition": "reused whole from phase103 (refit-free, "
                                     "exact); its floor share is a realised-state "
                                     "accounting, not a channel identification",
            "part1_floor_share_of_published_gain": prim["floor_share"],
            "part2": "phase103's Part II was written but never run (its component "
                     "directory was empty); this phase runs it on all areas built",
        },
        "panel": {"n_slots": int(sl.height), "n_areas": len(areas_built),
                  "areas_built": areas_built,
                  "n_test_days": int(sl["test_day"].n_unique()),
                  "first_test_day": sl["test_day"].min(),
                  "last_test_day": sl["test_day"].max(),
                  "component_rows": int(comp.height), "joined_rows": int(j.height),
                  "max_abs_price_gap_vs_ladder": y_gap,
                  "floor_threshold_jpykwh": TAU,
                  "floor_threshold_source": r"\thrFloorDefault (macros.tex l.275)"},
        "null_audit": audit,
        "prevalence": prevalence,
        "components": {
            "pi": "XGBClassifier(300 x depth 4, lr 0.05, hist, logloss, n_jobs 4, "
                  "seed 2026), label 1{y <= 0.5}, per arm's own column set",
            "mu": "phase85's XGBRegressor parameters verbatim, trained on train "
                  "rows with y > 0.5",
            "f": "train-window mean price over rows with y <= 0.5, COMMON to both "
                 "arms (fallback 0.01 if a fold has no floor train row)",
            "yhat": "pi * f + (1 - pi) * mu",
        },
        "system_mae": {"A_basepi_basemu": _j(mae["A"]), "B_headpi_basemu": _j(mae["B"]),
                       "C_basepi_headmu": _j(mae["C"]),
                       "D_headpi_headmu": _j(mae["D"]),
                       "ladder_base_published": 2.1506,
                       "ladder_head_published": 1.8989},
        "totals": {
            "published_gain_same_rows": _entry(mod85, delta),
            "hurdle_gain_G": _entry(mod85, g_total),
            "reconstruction_R": _entry(mod85, parts["r_recon"]),
            "relative_reconstruction_gap": _j(rel_r, 3),
            # phase90's lesson: a share is not "small" until it is converted into
            # the units the paper reports. One JEPX tick is 0.01 JPY/kWh.
            "in_jepx_ticks": {
                "published_gain": _j(float(delta.mean()) / 0.01, 2),
                "floor_state_channel_S_pi": _j(float(parts["s_pi"].mean()) / 0.01, 2),
                "cond_level_channel_S_mu": _j(float(parts["s_mu"].mean()) / 0.01, 2),
                "reconstruction_R": _j(float(parts["r_recon"].mean()) / 0.01, 2),
                "tick_jpykwh": 0.01},
            "c5_threshold": FIDELITY_MAX_REL, "faithful": bool(faithful),
        },
        "decomposition_shapley": {
            "algebra": "Delta_paper = S_pi + S_mu + R (exact, per slot)",
            "floor_state_channel_S_pi": _entry(mod85, parts["s_pi"]),
            "cond_level_channel_S_mu": _entry(mod85, parts["s_mu"]),
            "reconstruction_R": _entry(mod85, parts["r_recon"]),
            "share_of_published_gain": {
                "floor_state": sh_of_delta["s_pi"],
                "cond_level": sh_of_delta["s_mu"],
                "reconstruction": sh_of_delta["r_recon"]},
            "share_of_hurdle_gain": {"floor_state": sh_of_g["s_pi"],
                                     "cond_level": sh_of_g["s_mu"]},
        },
        "decomposition_main_effects": {
            "algebra": "Delta_paper = g_pi + g_mu + g_int + R (exact, per slot)",
            "floor_state_channel_g_pi": _entry(mod85, parts["g_pi"]),
            "cond_level_channel_g_mu": _entry(mod85, parts["g_mu"]),
            "interaction_g_int": _entry(mod85, parts["g_int"]),
            "reconstruction_R": _entry(mod85, parts["r_recon"]),
        },
        "decomposition_channel_x_state": {
            "algebra": "Delta_paper = S_pi^F + S_pi^NF + S_mu^F + S_mu^NF + R",
            "contributions": {k: _entry(mod85, parts[k])
                              for k in ("s_pi_F", "s_pi_NF", "s_mu_F", "s_mu_NF")},
            "shares_of_published_gain": {k: sh_of_delta[k]
                                         for k in ("s_pi_F", "s_pi_NF", "s_mu_F",
                                                   "s_mu_NF")},
        },
        "identities": identities,
        "component_diagnostics": comp_diag,
        "per_area": per_area,
        "exploratory_post_hoc": explor,
        "positive_control_p1_panel": p1,
        "positive_control_p1_headline_anchor": p1_anchor,
        "positive_control_p2_parameter_fidelity": p2,
        "positive_control_p3_oracle": {
            "why": "perfect floor-state knowledge must buy a large MAE gain, or "
                   "the decomposition cannot detect a classification channel",
            "oracle_mae": _j(float(oracle_daily.mean())),
            "gain_over_system_A": _j(oracle_gain),
            "min_required": ORACLE_MIN_GAIN,
            "pass": bool(oracle_gain >= ORACLE_MIN_GAIN)},
        "positive_control_p4_cross_path": p4,
        "part1_reused_from_phase103": part1["part1_state_partition"]["tau_0.5"],
        "metadata": {
            "script": Path(__file__).name,
            "git_sha": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "inputs": [str(P85_PARQUET), str(P85_JSON), str(P103_JSON),
                       str(P103_PATH), str(ACC)],
            "reused_modules": {
                "phase85": "panel, folds, cells, learner, block bootstrap, HLN-DM",
                "phase103": "component build (rebinds phase85 _design/_fit), "
                            "Part I state partition and its positive controls"},
            "seed": 2026, "n_boot": 10_000, "boot_block": 7,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("Delta %.4f = S_pi %.4f + S_mu %.4f + R %.4f | sh_pi %s sh_mu %s",
             delta.mean(), parts["s_pi"].mean(), parts["s_mu"].mean(),
             parts["r_recon"].mean(), sh_pi, sh_mu)
    log.info("branches %s | %s", fired, verdict)
    log.info("wrote %s", OUT_JSON)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="")
    args = ap.parse_args()
    if args.step == "build":
        mod = _load_p103()
        mod85 = mod._load_phase85()
        areas = [a for a in args.areas.split(",") if a] or list(mod85.AREA_COORDS)
        unknown = [a for a in areas if a not in mod85.AREA_COORDS]
        if unknown:
            msg = f"unknown areas: {unknown}"
            raise RuntimeError(msg)
        return build(areas)
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
