r"""Phase 102: is the post-gate price null a property of the SIGNAL or of the LEARNER?

THE QUESTION
------------
Section 5.3's two-target result is that the run the 10:00 gate excludes keeps
improving a forecast of the PHYSICAL delivery day (phase89 Gate 51: 59.88 MW
[49.64, 70.55] over the pre-gate headline on the 696-day common panel) while
adding nothing detectable to the PRICE (phase85 CW gate timing: +0.0026
[-0.0047, 0.0099] JPY/kWh, equivalent at the one-tick margin).

A referee's first objection is not that the number is wrong but that the
inference is: maybe the information IS in the later run and the LEARNER cannot
extract it. phase96 answered the learner-class version of that objection for
RQ1 (LEAR reproduces the weather increment's direction at roughly half the
size). This is the matching test for RQ2, and it is not the same test, because
the ladder never gave any learner the one quantity that matters here: the
DIFFERENCE between the two runs.

The ladder's post-gate cell replaces the pre-gate field with the post-gate
field. A gradient-boosted tree makes axis-aligned splits on single features; it
cannot form the difference of two of them. So "the post-gate cell does not beat
the pre-gate cell" is consistent with two very different worlds, and the ladder
as built cannot tell them apart:

  (i) the update the later run carries is irrelevant to price formation, or
  (ii) the update is relevant but no tree in this design can see it, because
       seeing it requires subtracting one column from another.

phase90 already showed the update is NOT orthogonal to the price residual
(OLS slope -0.388 JPY/kWh per 531 MW innovation SD, cluster-robust p < 1e-4,
r2_increment 1.19%), and bounded what a LINEAR use of it could take off the
pooled MAE at 0.0036 JPY/kWh -- but that bound is IN SAMPLE by construction
(the slope minimising absolute error is chosen on the rows it is scored on) and
its own JSON records that refitting the slope on the full panel would give
-0.0016, i.e. a LOSS. Nobody has yet handed the innovation to the actual
learner, inside the actual walk-forward, and measured out of sample.

THE TEST
--------
Feed the innovation DIRECTLY as an extra feature on top of the pre-gate feature
set, and score it with phase85's panel, folds, design matrix, estimator and
statistics VERBATIM (module import + function rebinding, rebindings asserted --
phase91 taught this project that a silent module-global swap turns a contrast
into a copy and reports a bug as a null).

Two innovation objects, because there are two defensible readings of "the new
information the later run carries", and the honest thing is to report both:

  WX  the weather-field innovation: post-gate minus pre-gate for each of the
      four capacity-weighted meteorological channels, same delivery slot. This
      is the literal object, and it exists on every slot of the ladder's panel
      (732 days x 9 areas), so it is the PRIMARY.
  PV  phase90's innovation, Delta-PV-hat = proxy_scaled(post) -
      proxy_scaled(pre): the update to the cross-fitted physical PREDICTION.
      This is the exact quantity Section 5.3 regresses the price residual on,
      so it is the one a referee reading that paragraph will ask about. It is a
      photovoltaic quantity and exists only on daylight slots in the eight
      areas whose operators publish actuals; it is zero at night by
      construction and MISSING (NaN, never silently zero) elsewhere, and both
      counts are reported.

PRICE CELLS (fitted; every other phase85 cell is still CONSTRUCTED so the
panel's row set cannot move, and written as NaN so it cannot be mistaken for a
prediction):
  msm_d2_12z_cw       phase85's pre-gate headline, REFIT, and bit-checked
                      against phase85's own parquet (Gate 102A). If this does
                      not reproduce to 1e-9 nothing else here is comparable.
  pre_plus_wx_innov   pre-gate field + the four WX innovation columns
  pre_plus_pv_innov   pre-gate field + Delta-PV-hat
  pre_plus_noise      pre-gate field + four Gaussian columns matched to the WX
                      innovation columns' per-area standard deviations
                      (NEGATIVE CONTROL: the cost of four more columns)
  pre_plus_post_raw   pre-gate field + post-gate field as raw columns. Spans
                      the same information as pre_plus_wx_innov in a different
                      basis, so its gap from pre_plus_wx_innov is the
                      REPRESENTATION effect and nothing else.

BUILD BUDGET, DECLARED
This machine is shared and each fitted cell costs ~730 folds x 9 areas of
XGBoost. The build is therefore split: pass A carries the question
(pre-gate refit, the two innovation cells) on the FULL panel, and pass B
carries the two controls. Whether pass B ran on the full panel or on a THINNED
subsample of the same folds (phase85's schedule with STEP widened, so every
retained fold is the same fold of the full run with the same training set) is
recorded in the artefact under `build_passes`, and a thinned control is scored
in its own block beside the primary contrast recomputed on exactly those days.
A pass that covered only some areas is reported ABSENT, never partial: a
partial pass would change the panel, which is the one thing that must not
differ.

POSITIVE CONTROL (physical target) -- REQUIRED, and it gates the price reading
The innovation is by construction the update to the weather field, so it MUST
help a forecast of the physical delivery day. If it does not, the construction
is wrong and this script reports a broken construction, NOT a price null.
Run in phase89's own idiom (its row set, target, daylight window, stamp
convention, burn-in and expanding-window discipline, reused by importing the
module), extending the origin-OLS scale to more than one regressor:

  A  target ~ s * proxy(pre)                  == phase89's pre-gate cell; must
                                                 reproduce phase89's per-slot
                                                 absolute errors to 1e-6
                                                 (Gate 102B)
  B  target ~ s * proxy(post)                 == phase89's post-gate cell
  C  target ~ a * proxy(pre) + b * PVinnov    the PV innovation as a feature
  D  target ~ a * proxy(pre) + sum b_k * WXinnov_k
                                              the SAME four WX columns the
                                              price arm gets -- the matched
                                              control
  E  target ~ a * proxy(pre) + sum b_k * noise_k
                                              matched-variance Gaussian
                                              placebo (the cost of four more
                                              regressors)

PRE-REGISTERED BRANCHES (fixed before any number is computed)
Materiality margin: 0.01 JPY/kWh, one JEPX price tick -- the paper's OWN
pre-registered equivalence margin for this contrast (phase85 `_margins_for`),
not a threshold invented here. Reference point: the ladder's measured post-gate
delta, +0.0026 JPY/kWh. Primary statistic: delta_wx = pooled daily improvement
of pre_plus_wx_innov minus that of msm_d2_12z_cw, phase85's 7-day circular
block bootstrap (95%) and HLN-DM at L=3 and L=9.

  GATE 102C -- innovation is non-degenerate. Report mean |innovation| per
    channel and its ratio to the pre-gate field's SD. If the WX innovation is
    numerically ~0 (all four channels below 1% of the field SD) then BOTH arms
    are uninformative and this phase reports UNDETERMINED. Direction of failure
    matters: a degenerate innovation would mean the two GPV caches are the same
    file, which is a data defect, not a finding.

  GATE 102B -- physical positive control. PASS requires cell C or D to beat
    cell A with a bootstrap CI excluding zero AND to recover >= 50% of
    phase89's 59.88 MW pre->post gain. If it FAILS, the verdict is
    UNDETERMINED-CONSTRUCTION-BROKEN and no price branch may be read.

  GATE 102D -- price, conditional on 102B PASS and 102C PASS:
    branch (a) LEARNER LIMITATION -- delta_wx CI lower bound > +0.01 (one tick)
       and DM p < 0.05 at both lags. The information IS extractable; the
       manuscript's Section 5.3 / Section 7 reading that the excluded run's
       price-relevant content is worth "about a third of a tick" and that the
       design already realises most of it must be WEAKENED, and the
       equivalence claim at the tick margin is no longer defensible as an
       information statement. THIS WOULD BE BAD NEWS FOR THE PAPER AND WILL BE
       REPORTED PLAINLY IF FOUND.
    branch (b) SIGNAL NULL (the strong version) -- delta_wx CI covers zero, or
       covers it after the noise placebo's own delta is taken as the yardstick.
       With 102B passing, the post-gate null is then a property of the signal's
       RELEVANCE TO PRICE and not of the learner or of the representation:
       the same innovation, handed to a linear model, moves a physical forecast
       by tens of megawatts and, handed to the ladder's own learner in the
       easiest possible form, moves the price forecast by nothing. That is a
       strictly stronger version of the paper's current claim.
    branch (c) SMALL BUT REAL -- CI excludes zero on the positive side but its
       LOWER bound is <= +0.01, so significance is established and materiality
       at the tick is not. Real, sub-tick at the bound; the economic conclusion
       survives and the sentence about the ladder design's limitation gets a
       number instead of an adjective.
       [AMENDMENT, recorded before any price contrast was computed (pass A was
       still building; only the physical positive control and the innovation's
       magnitude had been seen): this branch first said "upper bound <= +0.01",
       which left a gap between (a) and (c) whenever the interval straddled the
       tick. The wording now states the condition the code has always tested,
       lower bound > 0 with lower bound <= tick. No threshold changed.]
    branch (d) NEGATIVE -- delta_wx significantly < 0. The extra columns hurt;
       report with the noise placebo as the yardstick for how much of that is
       dimensionality rather than the innovation being anti-informative.

  GATE 102E -- representation. delta of pre_plus_wx_innov minus delta of
    pre_plus_post_raw. If the differenced basis wins and the raw-union basis
    does not, the ladder's post-gate null was partly a representation artefact
    even if the total is sub-tick, and that is worth one sentence in Section 7.
    Reported whatever the sign.

NOT PRE-REGISTERED, so labelled descriptive wherever it appears: per-area
breakdowns, and the PV-innovation cell's own sub-panel figures.

CLI (from the repository root):
  uv run python scripts/54b_innovation_construction.py --step smoke
  uv run python scripts/54b_innovation_construction.py --step build-price
  uv run python scripts/54b_innovation_construction.py --step build-phys
  uv run python scripts/54b_innovation_construction.py --step analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase102")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
HERE = Path(__file__).parent
P85_PATH = HERE / "10_primary_ladder_cw.py"
P89_PATH = HERE / "50_physical_target.py"

P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
P89_SLOTS = RUN / "phase89_physical_per_slot.parquet"
P89_JSON = RUN / "phase89_physical_target.json"
P90_JSON = RUN / "phase90_mechanism.json"

CKPT_A = RUN / "phase102_price_checkpoints_a"
CKPT_SMOKE = RUN / "phase102_smoke_checkpoints"


def _ckpt(pass_: str) -> Path:
    return RUN / f"phase102_price_checkpoints_{pass_}"
OUT_JSON = RUN / "phase102_innovation_learner.json"
PHYS_PARQUET = RUN / "phase102_physical_per_slot.parquet"

PRE_GATE = "msm_d2_12z_cw"
POST_GATE = "msm_d1_00z_cw"
BASELINE = "price_only"
C_WX = "pre_plus_wx_innov"
C_PV = "pre_plus_pv_innov"
C_NOISE = "pre_plus_noise"
C_RAW = "pre_plus_post_raw"
NEW_CELLS = (C_WX, C_PV, C_NOISE, C_RAW)
# The build is split into two passes because this machine is shared and every
# fitted cell costs ~730 folds x 9 areas of XGBoost. Pass A carries the
# question and its bit-reproduction anchor; pass B carries the two controls.
# Each pass writes its own checkpoint directory and `analyse` joins them on
# (area, datetime_jst), asserting the row sets agree, so a pass that does not
# finish is ABSENT rather than partial. Which cells a checkpoint holds is
# recorded in the artefact.
PASS_CELLS = {"a": (PRE_GATE, C_WX, C_PV), "b": (C_NOISE, C_RAW)}
FIT_CELLS: tuple[str, ...] = ()

PV_INNOV_COL = "dpv_hat_innov"
NOISE_STEM = "noise_innov"

# pre-registered constants
TICK = 0.01                 # JPY/kWh, the paper's own equivalence margin
BIT_TOL = 1e-9              # Gate 102A, phase85's own GATE_TOL
PHYS_REPRO_TOL = 1e-6       # Gate 102B reproduction of phase89's absolute errors
RECOVER_FRAC = 0.50         # Gate 102B: fraction of phase89's pre->post gain
DEGENERATE_FRAC = 0.01      # Gate 102C: |innov| mean / field SD floor
NOISE_SEED = 102_2026
MIN_PHYS_TRAIN_ROWS = 200   # phase89's MIN_TRAIN_ROWS, kept so row sets match
# phase90's NEAR_FLOOR, reused verbatim so the regime slice here means what the
# manuscript's mechanism paragraph means by it
NEAR_FLOOR = 0.5

_wanted_colsets: dict[tuple[str, ...], str] = {}
_design_calls: list[str] = []
_SIDE_DIR: Path = RUN / "phase102_innovation_side"


def _value_counts(s: pl.Series) -> dict[str, int]:
    """Counts as a dict, from ONE call to value_counts.

    Calling value_counts twice and zipping the key column of one call against
    the count column of another silently permutes the labels -- polars does not
    promise an order and the two calls need not agree. The pre-flight caught
    exactly that here: the provenance labels came out cyclically shifted while
    every total was right, which is the hardest kind of wrong number to notice.
    """
    return {r[s.name]: int(r["count"]) for r in s.value_counts().to_dicts()}


def _side_path(area: str) -> Path:
    _SIDE_DIR.mkdir(parents=True, exist_ok=True)
    return _SIDE_DIR / f"{area}.parquet"


# --------------------------------------------------------------------- helpers
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,  # noqa: S607
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _j(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ------------------------------------------------------------ PV innovation
def _pv_innovation() -> pl.DataFrame:
    """phase90's innovation, per slot: proxy_scaled(post) - proxy_scaled(pre).

    Read from phase89's committed per-slot output rather than recomputed, so it
    is the same object Section 5.3 regresses on. Rows exist only where phase89
    scored BOTH tiers (daylight, eight areas, post burn-in).
    """
    d = pl.read_parquet(P89_SLOTS)
    pre = d.filter(pl.col("tier") == PRE_GATE).select(
        "area", "datetime_jst", pl.col("proxy_scaled_cf").alias("pre_scaled"))
    post = d.filter(pl.col("tier") == POST_GATE).select(
        "area", "datetime_jst", pl.col("proxy_scaled_cf").alias("post_scaled"))
    j = pre.join(post, on=["area", "datetime_jst"], how="inner")
    if j.height != pre.height or j.height != post.height:
        msg = (f"phase89 tiers do not share a row set: pre={pre.height} "
               f"post={post.height} inner={j.height}; the innovation would be "
               "defined on a different sample than either tier")
        raise RuntimeError(msg)
    return j.with_columns(
        (pl.col("post_scaled") - pl.col("pre_scaled")).alias(PV_INNOV_COL)
    ).select("area", "datetime_jst", PV_INNOV_COL)


# ------------------------------------------------------------ price arm build
def build_price(mod, areas: list[str], ckpt: Path) -> dict:
    """phase85's panel/folds/design/estimator verbatim; four cells added."""
    global _SIDE_DIR  # noqa: PLW0603 - one side dir per build target
    _SIDE_DIR = ckpt.parent / f"{ckpt.name}_side"
    wx_cols = list(mod.DEFAULT_WEATHER_COLS)
    innov_names = [f"{c}__innov" for c in wx_cols]
    noise_names = [f"{NOISE_STEM}_{i}" for i in range(len(wx_cols))]
    pv = _pv_innovation()
    pv_days = set(pv["datetime_jst"].dt.date().cast(pl.Utf8).unique().to_list())
    pv_areas = set(pv["area"].unique().to_list())
    diag: dict = {"areas": {}}

    orig_design, orig_fit = mod._design, mod._fit

    def design_capturing(df: pl.DataFrame):
        out, cells = orig_design(df)
        area = out["area"].unique().to_list()[0]
        _design_calls.append(area)

        # ---- the WX innovation: post-gate field minus pre-gate field, same slot
        out = out.with_columns(
            *[(pl.col(f"{c}_{POST_GATE}") - pl.col(f"{c}_{PRE_GATE}")).alias(n)
              for c, n in zip(wx_cols, innov_names, strict=True)])

        # ---- Delta-PV-hat: zero at night (it is a PV quantity), NaN where the
        #      operators publish nothing or phase89 scored nothing. Never
        #      silently zero for a missing row.
        out = out.join(pv, on=["area", "datetime_jst"], how="left")
        day_str = pl.col("datetime_jst").dt.date().cast(pl.Utf8)
        hour = pl.col("datetime_jst").dt.hour()
        lo, hi = mod_p89.DAYTIME_HOURS
        in_window = hour.is_between(lo, hi)
        covered = day_str.is_in(list(pv_days)) & pl.lit(area in pv_areas)
        out = out.with_columns(
            pl.when(pl.col(PV_INNOV_COL).is_not_null())
            .then(pl.lit("phase89"))
            .when(covered & ~in_window)
            .then(pl.lit("night_zero"))
            .otherwise(pl.lit("missing"))
            .alias("pv_innov_provenance"))
        out = out.with_columns(
            pl.when(pl.col("pv_innov_provenance") == "phase89")
            .then(pl.col(PV_INNOV_COL))
            .when(pl.col("pv_innov_provenance") == "night_zero")
            .then(pl.lit(0.0))
            .otherwise(pl.lit(float("nan")))
            .alias(PV_INNOV_COL))

        # ---- matched-variance Gaussian placebo, seeded per area
        sds = [float(out[n].std() or 0.0) for n in innov_names]
        rng = np.random.default_rng(
            NOISE_SEED + int.from_bytes(area.encode(), "little") % 10_000)
        out = out.with_columns(
            *[pl.Series(nm, rng.normal(0.0, sd if sd > 0 else 1.0, out.height))
              for nm, sd in zip(noise_names, sds, strict=True)])

        base_pre = cells[PRE_GATE]
        cells[C_WX] = [*base_pre, *innov_names]
        cells[C_PV] = [*base_pre, PV_INNOV_COL]
        cells[C_NOISE] = [*base_pre, *noise_names]
        cells[C_RAW] = [*base_pre, *(f"{c}_{POST_GATE}" for c in wx_cols)]

        _wanted_colsets.clear()
        for name in FIT_CELLS:
            if name not in cells:
                msg = f"cell {name!r} is not in the design"
                raise RuntimeError(msg)
            key = tuple(cells[name])
            if key in _wanted_colsets:
                msg = (f"cells {name!r} and {_wanted_colsets[key]!r} have an "
                       "identical column list, so the dispatch cannot tell them "
                       "apart")
                raise RuntimeError(msg)
            _wanted_colsets[key] = name

        diag["areas"][area] = {
            "n_rows": out.height,
            "innov_wx": {n: {"mean_abs": _j(float(out[n].abs().mean()), 5),
                             "sd": _j(float(out[n].std()), 5),
                             "field_sd": _j(float(out[f"{c}_{PRE_GATE}"].std()), 5),
                             "n_null": int(out[n].null_count()),
                             "n_nan": int(out[n].is_nan().sum())}
                         for c, n in zip(wx_cols, innov_names, strict=True)},
            "pv_innov": {
                "provenance_counts": _value_counts(out["pv_innov_provenance"]),
                "n_nan": int(out[PV_INNOV_COL].is_nan().sum()),
                "mean_abs_where_from_phase89": _j(float(
                    out.filter(pl.col("pv_innov_provenance") == "phase89")
                    [PV_INNOV_COL].abs().mean() or 0.0), 3),
            },
            "noise_sd_used": [_j(s, 5) for s in sds],
        }
        # Side artefact so the innovation can be audited on EXACTLY the rows the
        # ladder scores (this frame is pre-`drop_nulls`; the scored row set is
        # phase85's and is joined back in `analyse`). Without it a null on a
        # scored slot could only be inferred, and rule 8 of this phase's brief
        # requires it counted.
        out.select("area", "datetime_jst", *innov_names, PV_INNOV_COL,
                   "pv_innov_provenance",
                   *[f"{c}_{PRE_GATE}" for c in wx_cols]).write_parquet(
            _side_path(area))
        return out, cells

    def fit_dispatch(train, test, cols):
        name = _wanted_colsets.get(tuple(cols))
        if name is None:
            return np.full(test.height, np.nan)
        return orig_fit(train, test, cols)

    mod._design, mod._fit = design_capturing, fit_dispatch
    if mod._design is orig_design or mod._fit is orig_fit:
        msg = "a rebinding did not take effect"
        raise RuntimeError(msg)
    if (mod._design.__name__ != "design_capturing"
            or mod._fit.__name__ != "fit_dispatch"):
        msg = "a rebinding bound the wrong function"
        raise RuntimeError(msg)
    mod.CKPT = ckpt
    ckpt.mkdir(parents=True, exist_ok=True)
    for a in areas:
        t0 = time.time()
        log.info("=== price arm: %s ===", a)
        mod._run_area(a)
        log.info("=== %s done in %.0f s ===", a, time.time() - t0)
    if not _design_calls:
        msg = ("the rebound _design was never called, so the innovation columns "
               "were never built and any result would be phase85's own")
        raise RuntimeError(msg)
    # one file per invocation: areas are built in separate processes and a
    # shared filename would leave only the last writer's areas on disk
    (ckpt / f"_diagnostics_{'-'.join(areas)}.json").write_text(
        json.dumps(diag, indent=1))
    return diag


# ------------------------------------------------------------------ importance
IMP_STEP_DAYS = 73          # 10 folds spread across the 732-day panel
IMP_N_FOLDS = 10
OUT_IMP = RUN / "phase102_innovation_importance.json"


def build_importance(mod, areas: list[str]) -> dict:
    """Does the learner PLACE SPLITS on the innovation columns? (descriptive)

    A referee can read a null two ways: the tree saw the innovation and found
    nothing, or the tree never looked. This measures which, by recording each
    fitted booster's total-gain split of the pre_plus_wx_innov cell between the
    four innovation columns and everything else.

    The folds are phase85's own: rebinding STEP to 73 days and MAX_FOLDS to 10
    selects ten of the panel's real test days spread across it, and each one's
    training set is `date < test_day` exactly as in the full run, so these are
    the same fits -- a thinned sample of them, not different ones. The
    importance-recording fit is asserted to give bit-identical predictions to
    phase85's own `_fit` before its numbers are used.
    """
    global _SIDE_DIR  # noqa: PLW0603
    tmp = RUN / "phase102_importance_tmp"
    _SIDE_DIR = tmp / "side"
    wx_cols = list(mod.DEFAULT_WEATHER_COLS)
    innov_names = [f"{c}__innov" for c in wx_cols]
    orig_design, orig_fit = mod._design, mod._fit
    acc: dict = {}

    state: dict = {"wanted": None, "designed": 0}

    def design_capturing(df: pl.DataFrame):
        out, cells = orig_design(df)
        out = out.with_columns(
            *[(pl.col(f"{c}_{POST_GATE}") - pl.col(f"{c}_{PRE_GATE}")).alias(n)
              for c, n in zip(wx_cols, innov_names, strict=True)])
        cells[C_WX] = [*cells[PRE_GATE], *innov_names]
        state["wanted"] = list(cells[C_WX])
        state["designed"] += 1
        return out, cells

    def fit_recording(train, test, cols):
        if list(cols) != state["wanted"]:
            return np.full(test.height, np.nan)
        xtr = train.select(cols).to_numpy().astype(float)
        ytr = train["price_yen_kwh"].to_numpy().astype(float)
        xte = test.select(cols).to_numpy().astype(float)
        m = mod.xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=mod.SEED, n_jobs=4,
                                 tree_method="hist")
        m.fit(xtr, ytr)
        pred = m.predict(xte).astype(float)
        ref = orig_fit(train, test, cols)
        if float(np.abs(pred - ref).max()) > BIT_TOL:
            msg = ("the importance-recording fit does not reproduce phase85's "
                   "`_fit`, so its importances describe a different model")
            raise RuntimeError(msg)
        gain = m.get_booster().get_score(importance_type="total_gain")
        tot = sum(gain.values())
        idx = {f"f{i}": c for i, c in enumerate(cols)}
        per_col = {idx[k]: v for k, v in gain.items() if k in idx}
        innov_gain = sum(v for c, v in per_col.items() if c in innov_names)
        acc.setdefault("folds", []).append({
            "n_features": len(cols),
            "innovation_gain_share": _j(innov_gain / tot if tot else None, 5),
            "n_innovation_cols_used": sum(1 for c in per_col if c in innov_names),
            "per_innovation_col_share": {
                c: _j(per_col.get(c, 0.0) / tot if tot else None, 5)
                for c in innov_names},
        })
        return pred

    mod._design, mod._fit = design_capturing, fit_recording
    if mod._design is orig_design or mod._fit is orig_fit:
        msg = "a rebinding did not take effect"
        raise RuntimeError(msg)
    mod.CKPT = tmp
    mod.STEP = timedelta(days=IMP_STEP_DAYS)
    mod.MAX_FOLDS = IMP_N_FOLDS
    out: dict = {"areas": {}}
    for a in areas:
        acc.clear()
        state["wanted"], state["designed"] = None, 0
        (tmp / f"{a}.parquet").unlink(missing_ok=True)
        mod._run_area(a)
        if state["designed"] != 1:
            msg = f"the rebound design ran {state['designed']} times for {a}"
            raise RuntimeError(msg)
        folds = acc.get("folds", [])
        if not folds:
            msg = f"no importance folds recorded for {a}"
            raise RuntimeError(msg)
        shares = [f["innovation_gain_share"] for f in folds
                  if f["innovation_gain_share"] is not None]
        out["areas"][a] = {
            "n_folds": len(folds),
            "mean_innovation_gain_share": _j(float(np.mean(shares)), 5),
            "min": _j(float(np.min(shares)), 5), "max": _j(float(np.max(shares)), 5),
            "mean_n_innovation_cols_used": _j(
                float(np.mean([f["n_innovation_cols_used"] for f in folds])), 2),
            "n_innovation_cols_available": len(innov_names),
            "folds": folds,
        }
        log.info("%s: innovation columns take %.2f%% of total gain (%d folds)",
                 a, 100.0 * out["areas"][a]["mean_innovation_gain_share"],
                 len(folds))
    out["design"] = {
        "cell": C_WX, "fold_step_days": IMP_STEP_DAYS, "n_folds_per_area": IMP_N_FOLDS,
        "importance_type": "total_gain",
        "note": ("DESCRIPTIVE, not pre-registered. Ten of the panel's real test "
                 "days per area; each fit is bit-identical to phase85's `_fit` on "
                 "the same fold (asserted)."),
    }
    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha()}
    OUT_IMP.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_IMP)
    return out


# --------------------------------------------------- physical arm (positive control)
def _expanding_ols(day: np.ndarray, x: np.ndarray, y: np.ndarray,
                   min_rows: int) -> np.ndarray:
    """Expanding-window OLS through the origin, k regressors, one pass.

    Generalises phase89's `_cross_fitted`: the k=1 closed form is a ratio of two
    running sums, and for k>1 it is a k x k solve of running sums. Day t uses
    only rows from days strictly before t, which is phase89's discipline; rows
    before `min_rows` of history are returned as NaN (burn-in) and counted by
    the caller rather than scored with a fit that saw them.
    """
    n, k = x.shape
    order = np.argsort(day, kind="stable")
    x, y, day = x[order], y[order], day[order]
    # running sums of the outer products and of X'y
    xx = np.cumsum(x[:, :, None] * x[:, None, :], axis=0)
    xy = np.cumsum(x * y[:, None], axis=0)
    _, first = np.unique(day, return_index=True)
    prev_end = np.empty(n, dtype=int)
    for i, start in enumerate(first):
        stop = first[i + 1] if i + 1 < len(first) else n
        prev_end[start:stop] = start - 1
    pred = np.full(n, np.nan)
    for i, start in enumerate(first):
        stop = first[i + 1] if i + 1 < len(first) else n
        pe = start - 1
        if pe < 0 or (pe + 1) < min_rows:
            continue
        a, b = xx[pe], xy[pe]
        try:
            beta = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(beta)):
            continue
        pred[start:stop] = x[start:stop] @ beta
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    return pred[inv]


def _occto_potential(mod89) -> pl.DataFrame:
    """phase89's PRIMARY target frame: potential output, per-area stamp shift.

    phase89 builds this inline in `main()`, so it cannot be imported. It is
    reproduced here step for step -- frozen panel plus the recovered Kyushu
    month, the per-area interval-label shift, curtailed energy added back --
    and the reproduction is then VERIFIED, not assumed: Gate 102B requires the
    one-regressor cell built on this frame to reproduce phase89's own committed
    per-slot cross-fitted prediction to 1e-6 MW. If the reproduction fails, the
    reconstruction is wrong and no physical number here is usable.
    """
    occ = pl.read_parquet(mod89.OCCTO_PARQUET)
    if mod89.OCCTO_EXT.exists():
        ext = pl.read_parquet(mod89.OCCTO_EXT).select(occ.columns)
        overlap = ext.join(occ.select("area", "datetime_jst"),
                           on=["area", "datetime_jst"], how="inner").height
        if overlap:
            msg = f"OCCTO extension overlaps the frozen panel on {overlap} rows"
            raise ValueError(msg)
        occ = pl.concat([occ, ext])
    parts = []
    for a in sorted(x for x in occ["area"].unique().to_list()
                    if x in mod89.STAMP_SHIFT_MIN):
        curt = pl.col("pv_curtailment_mw").fill_null(0.0)
        parts.append(occ.filter(pl.col("area") == a)
                     .with_columns(pl.col("datetime_jst").dt.offset_by(
                         f"{mod89.STAMP_SHIFT_MIN[a]}m"))
                     .with_columns((pl.col("pv_mw") + curt).alias("target_mw"))
                     .select("area", "datetime_jst", "target_mw",
                             curt.alias("curtail_mw")))
    return pl.concat(parts)


def build_phys(mod85, mod89) -> dict:
    """phase89's row set, target, daylight window and burn-in; extra regressors."""
    wx_cols = list(mod85.DEFAULT_WEATHER_COLS)
    occ = _occto_potential(mod89)
    ref = pl.read_parquet(P89_SLOTS)
    areas = sorted(ref["area"].unique().to_list())
    frames: list[pl.DataFrame] = []
    checks: dict = {"areas": {}}
    for area in areas:
        # one proxy frame carrying BOTH tiers plus both tiers' weather fields, so
        # phase89's `_slots` (its asof join, daylight filter and finiteness
        # filter) can be reused verbatim on a single call
        prox = mod89._tier_frame(PRE_GATE, area)
        post = mod89._tier_frame(POST_GATE, area).rename(
            {"pv_proxy": "post_proxy"})
        prox = prox.join(post, on="datetime_jst", how="inner")
        for tier in (PRE_GATE, POST_GATE):
            prox = prox.join(mod85._load_cw_jma(tier, area), on="datetime_jst",
                             how="inner")
        rows = mod89._slots(prox, occ.filter(pl.col("area") == area), "target_mw")
        n_after_p89 = rows.height
        extra = ["post_proxy", *[f"{c}_{t}" for c in wx_cols
                                for t in (PRE_GATE, POST_GATE)]]
        rows = rows.drop_nulls(extra).filter(
            pl.all_horizontal(pl.col(c).is_finite() for c in extra))
        rows = rows.rename({"pv_proxy": "pre_proxy"}).with_columns(
            *[(pl.col(f"{c}_{POST_GATE}") - pl.col(f"{c}_{PRE_GATE}"))
              .alias(f"{c}__innov") for c in wx_cols],
            (pl.col("post_proxy") - pl.col("pre_proxy")).alias("proxy_innov"))
        rng = np.random.default_rng(
            NOISE_SEED + 7 + int.from_bytes(area.encode(), "little") % 10_000)
        rows = rows.with_columns(
            *[pl.Series(f"{NOISE_STEM}_{i}",
                        rng.normal(0.0,
                                   float(rows[f"{c}__innov"].std() or 1.0),
                                   rows.height))
              for i, c in enumerate(wx_cols)])
        checks["areas"][area] = {
            "n_rows": rows.height,
            "n_rows_dropped_beyond_phase89": n_after_p89 - rows.height,
            "n_rows_phase89_scored": int(ref.filter(
                (pl.col("area") == area) & (pl.col("tier") == PRE_GATE)).height),
        }
        frames.append(rows.sort("datetime_jst"))
    panel = pl.concat(frames)
    panel.write_parquet(PHYS_PARQUET)
    log.info("physical panel: %d rows, %d areas, %d days", panel.height,
             panel["area"].n_unique(), panel["day"].n_unique())
    return {"panel": panel, "checks": checks}


def _phys_cells(panel: pl.DataFrame, wx_cols: list[str]) -> dict:
    innov = [f"{c}__innov" for c in wx_cols]
    noise = [f"{NOISE_STEM}_{i}" for i in range(len(wx_cols))]
    return {
        "A_pre_only": ["pre_proxy"],
        "B_post_only": ["post_proxy"],
        "C_pre_plus_pv_innov": ["pre_proxy", "proxy_innov"],
        "D_pre_plus_wx_innov": ["pre_proxy", *innov],
        "E_pre_plus_noise": ["pre_proxy", *noise],
    }


def analyse_phys(mod89, panel: pl.DataFrame, wx_cols: list[str],
                 price_days: set[str]) -> dict:
    cells = _phys_cells(panel, wx_cols)
    ref = (pl.read_parquet(P89_SLOTS)
           .filter(pl.col("tier") == PRE_GATE)
           .select("area", "datetime_jst",
                   pl.col("proxy_scaled_cf").alias("p89_pred")))
    per_area_daily: dict[str, dict[str, pl.DataFrame]] = {c: {} for c in cells}
    repro: dict = {}
    burn: dict = {}
    for area in sorted(panel["area"].unique().to_list()):
        sub = panel.filter(pl.col("area") == area).sort("datetime_jst")
        day = sub["day"].to_numpy()
        y = np.asarray(sub["target_mw"], dtype=float)
        preds = {}
        for name, cols in cells.items():
            preds[name] = _expanding_ols(day, sub.select(cols).to_numpy()
                                         .astype(float), y, MIN_PHYS_TRAIN_ROWS)
            burn.setdefault(name, {})[area] = int(
                (~np.isfinite(preds[name])).sum())
        # ONE row set for every cell, so a level difference cannot be a panel
        # difference: every cell must have produced a finite prediction.
        mask = np.logical_and.reduce([np.isfinite(p) for p in preds.values()])
        # Gate 102B: cell A against phase89's own committed per-slot prediction
        chk = (sub.select("area", "datetime_jst")
               .with_columns(pl.Series("mine", preds["A_pre_only"]))
               .join(ref, on=["area", "datetime_jst"], how="inner")
               .filter(pl.col("mine").is_finite()))
        diff = np.abs(np.asarray(chk["mine"] - chk["p89_pred"], dtype=float))
        repro[area] = {
            "n_phase89_rows": int(ref.filter(pl.col("area") == area).height),
            "n_compared": int(chk.height),
            "max_abs_diff_mw": _j(float(diff.max()) if diff.size else None, 9),
            "n_scored_all_cells": int(mask.sum()),
        }
        for name, p in preds.items():
            per_area_daily[name][area] = (
                sub.with_columns(pl.Series("_ae", np.abs(p - y)),
                                 pl.Series("_ok", mask))
                .filter(pl.col("_ok"))
                .group_by("day").agg(pl.col("_ae").mean().alias("mae"))
                .sort("day"))
    pools = {n: mod89._pool(d, "mae") for n, d in per_area_daily.items()}

    def _delta(a: str, b: str, days: set[str] | None) -> dict:
        j = (pools[a].select("day", pl.col("pooled").alias("x"))
             .join(pools[b].select("day", pl.col("pooled").alias("z")),
                   on="day", how="inner").sort("day"))
        if days is not None:
            j = j.filter(pl.col("day").cast(pl.Utf8).is_in(list(days)))
        d = np.asarray(j["x"] - j["z"], dtype=float)
        lo, hi = mod89._boot_ci(d)
        return {"delta_mw": _j(float(d.mean()), 2), "ci95": [_j(lo, 2), _j(hi, 2)],
                "n_days": len(d), "excludes_zero": bool(lo > 0 or hi < 0)}

    per_area_delta: dict = {}
    for name in cells:
        if name == "A_pre_only":
            continue
        rec = {}
        for area, d in per_area_daily[name].items():
            j = (per_area_daily["A_pre_only"][area]
                 .select("day", pl.col("mae").alias("a"))
                 .join(d.select("day", pl.col("mae").alias("x")), on="day",
                       how="inner")
                 .filter(pl.col("day").cast(pl.Utf8).is_in(list(price_days)))
                 .sort("day"))
            v = np.asarray(j["a"] - j["x"], dtype=float)
            lo, hi = mod89._boot_ci(v)
            rec[area] = {"delta_mw": _j(float(v.mean()), 2),
                         "ci95": [_j(lo, 2), _j(hi, 2)], "n_days": len(v)}
        per_area_delta[name] = rec

    levels_common, levels_wide = {}, {}
    for n, p in pools.items():
        levels_wide[n] = {"mae_mw": _j(float(p["pooled"].mean()), 1),
                          "n_days": p.height}
        c = p.filter(pl.col("day").cast(pl.Utf8).is_in(list(price_days)))
        levels_common[n] = {"mae_mw": _j(float(c["pooled"].mean()), 1),
                            "n_days": c.height}
    out = {
        "estimator": ("expanding-window OLS through the origin on phase89's row "
                      "set, target, daylight window, per-area stamp convention "
                      "and burn-in (min_train_rows=200); day t fitted on days "
                      "strictly before t. k=1 is phase89's own model."),
        "cells": dict(cells),
        "levels_common_panel": levels_common,
        "levels_wide_panel": levels_wide,
        "gains_vs_A_common": {n: _delta("A_pre_only", n, price_days)
                              for n in cells if n != "A_pre_only"},
        "gains_vs_A_wide": {n: _delta("A_pre_only", n, None)
                            for n in cells if n != "A_pre_only"},
        "per_area_gains_vs_A_common": per_area_delta,
        "gate102b_reproduces_phase89": repro,
        "burn_in_rows": burn,
    }
    return out


# --------------------------------------------------------------------- analyse
def _cluster_ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> dict:
    """OLS with an intercept and a variance clustered on `cluster` (phase90's).

    Reported for one purpose only: to say whether the innovation is linearly
    related to the pre-gate price residual IN THIS REPRESENTATION on the full
    nine-area panel, and what an IN-SAMPLE linear rule could take off the MAE.
    Both are in-sample by construction and are labelled so.
    """
    xd = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    resid = y - xd @ beta
    xtx_inv = np.linalg.pinv(xd.T @ xd)
    meat = np.zeros((xd.shape[1], xd.shape[1]))
    order = np.argsort(cluster, kind="stable")
    xs, rs, cs = xd[order], resid[order], cluster[order]
    _, first = np.unique(cs, return_index=True)
    bounds = [*first.tolist(), len(cs)]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        s = xs[a:b].T @ rs[a:b]
        meat += np.outer(s, s)
    vcv = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(vcv))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    mae_raw = float(np.abs(y).mean())
    mae_after = float(np.abs(resid).mean())
    return {
        "coefficients": [_j(v, 6) for v in beta[1:].tolist()],
        "se_cluster": [_j(v, 6) for v in se[1:].tolist()],
        "t": [_j(v, 3) for v in (beta[1:] / np.where(se[1:] > 0, se[1:], np.nan))],
        "n": int(len(y)), "n_clusters": int(len(first)),
        "r2_increment": _j(1.0 - ss_res / ss_tot if ss_tot else None, 6),
        "mae_pre_gate_residual": _j(mae_raw, 5),
        "mae_after_removing_projection": _j(mae_after, 5),
        "in_sample_mae_ceiling": _j(mae_raw - mae_after, 5),
        "note": ("IN SAMPLE (the coefficients are fitted on the rows they are "
                 "scored on); an upper bound on a linear use of this "
                 "representation, not an achievable gain. Comparable with "
                 "phase90's implied_mae_reduction."),
    }


def _read_pass(ckpt: Path, cells: tuple[str, ...], n_areas: int) -> pl.DataFrame | None:
    """One pass's checkpoints, or None if the pass did not finish every area.

    A pass that covers only some areas would change the panel, so it is dropped
    whole rather than reported on a different sample from the others.
    """
    files = sorted(ckpt.glob("*.parquet"))
    if len(files) < n_areas:
        log.warning("pass at %s has %d/%d areas -- treating the pass as ABSENT",
                    ckpt.name, len(files), n_areas)
        return None
    d = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    keep = ["area", "datetime_jst", *(["y_actual"] if "y_actual" in d.columns else []),
            *[f"y_pred_{c}" for c in cells]]
    out = d.select(keep)
    for c in cells:
        bad = int(out[f"y_pred_{c}"].is_null().sum()
                  + out[f"y_pred_{c}"].is_nan().sum())
        if bad:
            msg = (f"cell {c!r} in {ckpt.name} has {bad} null/NaN predictions; a "
                   "NaN must never reach a metric")
            raise RuntimeError(msg)
    return out


def analyse(mod85, mod89) -> int:  # noqa: C901, PLR0912, PLR0915 - one block per pre-registered gate
    n_areas = len(mod85.AREA_COORDS)
    pass_a = _read_pass(CKPT_A, PASS_CELLS["a"], n_areas)
    if pass_a is None:
        log.error("pass A is incomplete; the primary contrast cannot be computed")
        return 2
    present = list(PASS_CELLS["a"])
    pass_state = {"a": "full"}
    mine = pass_a
    thinned: dict[str, pl.DataFrame] = {}
    for tag in ("b",):
        extra = _read_pass(_ckpt(tag), PASS_CELLS[tag], n_areas)
        if extra is None:
            extra = _read_pass(_ckpt(f"{tag}_thin"), PASS_CELLS[tag], n_areas)
            if extra is None:
                pass_state[tag] = "absent"
                continue
            # A THINNED control pass: phase85's own fold schedule with STEP
            # widened, so every fold it keeps is bit-identical to the same fold
            # of the full run -- a subsample of the panel, not a different one.
            # It is never merged into the full-panel contrasts; it gets its own
            # block, where the PRIMARY contrast is recomputed on exactly the
            # same days so the two are comparable.
            pass_state[tag] = "thinned"
            thinned[tag] = extra.drop("y_actual", strict=False)
            continue
        pass_state[tag] = "full"
        extra = extra.drop("y_actual", strict=False)
        mine = mine.join(extra, on=["area", "datetime_jst"], how="inner")
        if mine.height != pass_a.height:
            msg = f"pass {tag} does not share pass A's row set"
            raise RuntimeError(msg)
        present.extend(PASS_CELLS[tag])
    new_present = [c for c in NEW_CELLS if c in present]
    ref_cols = ["area", "datetime_jst", "test_day", "y_actual",
                f"y_pred_{BASELINE}", f"y_pred_{PRE_GATE}", f"y_pred_{POST_GATE}",
                f"cw_ok_{PRE_GATE}", f"cw_ok_{POST_GATE}"]
    ref = pl.read_parquet(P85_PARQUET, columns=ref_cols)
    cw_false = int((~(ref[f"cw_ok_{PRE_GATE}"] & ref[f"cw_ok_{POST_GATE}"])).sum())
    ref = ref.drop(f"cw_ok_{PRE_GATE}", f"cw_ok_{POST_GATE}")

    # ---- Gate 102A: the refit pre-gate cell must reproduce phase85 bit-for-bit
    key = ["area", "datetime_jst"]
    j = (mine.select(*key, pl.col(f"y_pred_{PRE_GATE}").alias("new_pre"),
                     pl.col("y_actual").alias("new_y"))
         .join(ref, on=key, how="inner"))
    gate_a = {
        "n_slots_mine": mine.height, "n_slots_phase85": ref.height,
        "n_slots_matched": j.height,
        "row_set_identical": bool(mine.height == ref.height == j.height),
        "max_abs_diff_pre_gate": _j(float((j["new_pre"]
                                           - j[f"y_pred_{PRE_GATE}"]).abs().max()
                                          or 0.0), 12),
        "max_abs_diff_y_actual": _j(float((j["new_y"] - j["y_actual"]).abs().max()
                                          or 0.0), 12),
        "tolerance": BIT_TOL,
    }
    gate_a["verdict"] = ("PASS" if gate_a["row_set_identical"]
                         and (gate_a["max_abs_diff_pre_gate"] or 0.0) <= BIT_TOL
                         and (gate_a["max_abs_diff_y_actual"] or 0.0) <= BIT_TOL
                         else "FAIL")
    gate_a["consequence_if_fail"] = (
        "Abort. The panel, folds or estimator moved, so no contrast here is "
        "comparable with the manuscript's ladder.")
    log.info("GATE 102A (bit-repro of phase85 pre-gate cell): %s", gate_a["verdict"])

    panel = (ref.join(mine.select(*key, *[f"y_pred_{c}" for c in new_present]),
                      on=key, how="inner"))
    nulls = {c: int(panel[f"y_pred_{c}"].is_null().sum()
                    + panel[f"y_pred_{c}"].is_nan().sum())
             for c in (BASELINE, PRE_GATE, POST_GATE, *new_present)}

    # ---- the innovation audited on EXACTLY the rows the ladder scores
    wx_cols = list(mod85.DEFAULT_WEATHER_COLS)
    innov_names = [f"{c}__innov" for c in wx_cols]
    side_dir = CKPT_A.parent / f"{CKPT_A.name}_side"
    side = pl.concat([pl.read_parquet(f) for f in sorted(side_dir.glob("*.parquet"))])
    scored = panel.join(side, on=key, how="left")
    if scored.height != panel.height:
        msg = "joining the innovation side artefact changed the row count"
        raise RuntimeError(msg)
    innov_on_scored = {
        "n_scored_slots": scored.height,
        "cw_ok_false_slots_in_phase85_panel": cw_false,
        "wx": {n: {"n_null": int(scored[n].null_count()),
                   "n_nan": int(scored[n].is_nan().sum()),
                   "mean_abs": _j(float(scored[n].abs().mean()), 5),
                   "sd": _j(float(scored[n].std()), 5),
                   "p95_abs": _j(float(scored[n].abs().quantile(0.95)), 5),
                   "field_sd": _j(float(scored[f"{c}_{PRE_GATE}"].std()), 5),
                   "mean_abs_over_field_sd": _j(
                       float(scored[n].abs().mean() / scored[f"{c}_{PRE_GATE}"].std()),
                       4)}
               for c, n in zip(wx_cols, innov_names, strict=True)},
        "pv_provenance_counts": _value_counts(scored["pv_innov_provenance"]),
        "pv_n_nan": int(scored[PV_INNOV_COL].is_nan().sum()
                        + scored[PV_INNOV_COL].null_count()),
        "pv_n_present_cross_check_vs_phase90": {
            "phase102_n_slots_with_innovation": int(
                (scored["pv_innov_provenance"] == "phase89").sum()),
            "phase90_n_slots": json.loads(P90_JSON.read_text())
                ["implied_price_value_full_panel"]["n_slots_with_innovation"],
        },
    }
    xc = innov_on_scored["pv_n_present_cross_check_vs_phase90"]
    xc["agree"] = bool(xc["phase102_n_slots_with_innovation"]
                       == xc["phase90_n_slots"])
    n_bad = sum(v["n_null"] + v["n_nan"] for v in innov_on_scored["wx"].values())
    innov_on_scored["wx_missing_on_any_scored_slot"] = int(n_bad)

    # ---- in-sample linear reference in THIS representation, full nine areas
    # every array below is taken from `scored`, never from `panel`: a polars join
    # does not promise row order, and mixing the two frames would silently
    # misalign the residual with the innovation.
    resid = np.asarray(scored["y_actual"] - scored[f"y_pred_{PRE_GATE}"], dtype=float)
    xw = scored.select(innov_names).to_numpy().astype(float)
    clus = np.asarray([f"{a}|{d}" for a, d in
                       zip(scored["area"].to_list(),
                           scored["test_day"].to_list(), strict=True)])
    ok = np.isfinite(xw).all(axis=1) & np.isfinite(resid)
    lin_wx = _cluster_ols(resid[ok], xw[ok], clus[ok])
    xp = np.asarray(scored[PV_INNOV_COL], dtype=float).reshape(-1, 1)
    okp = np.isfinite(xp).ravel() & np.isfinite(resid)
    lin_pv = _cluster_ols(resid[okp], xp[okp], clus[okp])
    linear_reference = {
        "why": ("phase90 bounded a LINEAR use of the PV innovation in sample at "
                "0.0036 JPY/kWh on the full panel and recorded that refitting the "
                "slope on that panel would give -0.0016. The same two quantities "
                "are recomputed here for the WX innovation, so branch (b) can be "
                "read as 'not even an in-sample linear rule finds much' rather "
                "than 'the tree failed'."),
        "wx_innovation_4col": {**lin_wx, "n_dropped_non_finite": int((~ok).sum())},
        "pv_innovation_1col": {**lin_pv, "n_dropped_non_finite": int((~okp).sum())},
    }
    areas = sorted(panel["area"].unique().to_list())
    per_day = mod85._per_day_mae(panel, [BASELINE, PRE_GATE, POST_GATE, *new_present])
    daily = {c: mod85._pooled_daily(per_day, c)
             for c in (PRE_GATE, POST_GATE, *new_present)}

    tiers = {c: mod85._entry(d) for c, d in daily.items()}
    contrasts = {
        "post_gate_vs_pre_gate_REPRODUCTION":
            mod85._diff_entry(daily[POST_GATE], daily[PRE_GATE]),
    }
    for cell, label in ((C_WX, "wx_innov"), (C_PV, "pv_innov"),
                        (C_NOISE, "noise"), (C_RAW, "post_raw")):
        if cell in daily:
            contrasts[f"{label}_vs_pre_gate"] = mod85._diff_entry(daily[cell],
                                                                  daily[PRE_GATE])
    for cell, label in ((C_NOISE, "wx_innov_vs_noise"),
                        (C_RAW, "wx_innov_vs_post_raw")):
        if cell in daily:
            contrasts[label] = mod85._diff_entry(daily[C_WX], daily[cell])
    per_area = {}
    for c in new_present:
        rec = {}
        for a in areas:
            sub = per_day.filter(pl.col("area") == a)
            d = np.asarray(sub[f"mae_{PRE_GATE}"] - sub[f"mae_{c}"], dtype=float)
            rec[a] = {"delta": _j(float(d.mean()), 4),
                      "p_l3": mod85._hln_dm(d, lag=3)["p_hln"]}
        ps = [rec[a]["p_l3"] for a in areas if rec[a]["p_l3"] is not None]
        names = [a for a in areas if rec[a]["p_l3"] is not None]
        for nm, padj in zip(names, mod85._holm(ps), strict=True):
            rec[nm]["p_l3_holm"] = _j(padj, 6)
        per_area[c] = rec

    # ---- the PV cell's own sub-panel (descriptive: the cell's feature only
    #      exists on daylight slots in eight areas)
    p89_areas = sorted(pl.read_parquet(P89_SLOTS, columns=["area"])["area"]
                       .unique().to_list())
    lo_h, hi_h = mod89.DAYTIME_HOURS
    sub = panel.filter(pl.col("area").is_in(p89_areas)
                       & pl.col("datetime_jst").dt.hour().is_between(lo_h, hi_h))
    pd_sub = mod85._per_day_mae(sub, [BASELINE, PRE_GATE, C_PV, C_WX])
    daily_sub = {c: mod85._pooled_daily(pd_sub, c) for c in (PRE_GATE, C_PV, C_WX)}
    subpanel = {
        "why": ("Delta-PV-hat exists only on daylight slots in the eight areas "
                "with published actuals; on the full panel the cell sees NaN "
                "elsewhere. This is the same contrast restricted to where the "
                "feature is defined. DESCRIPTIVE, not pre-registered."),
        "n_slots": sub.height, "n_days": int(sub["test_day"].n_unique()),
        "n_areas": len(p89_areas),
        "pv_innov_vs_pre_gate": mod85._diff_entry(daily_sub[C_PV],
                                                  daily_sub[PRE_GATE]),
        "wx_innov_vs_pre_gate": mod85._diff_entry(daily_sub[C_WX],
                                                 daily_sub[PRE_GATE]),
    }

    # ---- regime slices (DESCRIPTIVE, not pre-registered). The manuscript's
    #      decision-relevant regime is the near-floor slice, where it reports the
    #      weather gain as roughly seven times the pooled one, so a reader will
    #      ask whether the innovation hides there. The slice is defined by the
    #      REALISED price, which is the same selection caveat Section 5.1
    #      attaches to its own floor-slice numbers.
    regimes: dict = {
        "near_floor_threshold_jpykwh": NEAR_FLOOR,
        "caveat": ("slices are defined on the realised price, so a "
                   "downward-biased forecast would also score well; and the "
                   "daily sample shrinks to days that contain slice slots"),
        "not_a_new_finding": (
            "the near-floor gate-timing contrast is ALREADY committed: "
            "phase66v2 (phase66v2_floor_regime_v3_cw_full.json, pairing "
            "P4cw_gate_timing, allhours, threshold 0.5) reports "
            "d_mae_floor_hours slot-wise pooled = -0.0624 "
            "[-0.1594, -0.0316], HLN-DM p = 0.0062 (L=3) / 0.0025 (L=9) over "
            "369 days -- the post-gate run's floor-hours error is lower -- and "
            "its Gate 19 reading already states that the pooled null is "
            "MAE-specific. The figure recomputed here differs only in pooling "
            "(per-area daily MAE then averaged across areas, phase85's "
            "convention, rather than slot-wise) and is reported so the "
            "innovation cells can be read on the same slice, NOT as news."),
        "slices": {}}
    for label, expr in (("near_floor", pl.col("y_actual") <= NEAR_FLOOR),
                        ("non_floor", pl.col("y_actual") > NEAR_FLOOR)):
        pslice = panel.filter(expr)
        pdl = mod85._per_day_mae(pslice, [BASELINE, PRE_GATE, POST_GATE,
                                          *new_present])
        dsl = {c: mod85._pooled_daily(pdl, c)
               for c in (PRE_GATE, POST_GATE, *new_present)}
        rec = {"n_slots": pslice.height,
               "share_of_panel": _j(pslice.height / panel.height, 4),
               "n_days": int(len(dsl[PRE_GATE])),
               "post_gate_vs_pre_gate": mod85._diff_entry(dsl[POST_GATE],
                                                          dsl[PRE_GATE])}
        for cell, lab in ((C_WX, "wx_innov"), (C_PV, "pv_innov"),
                          (C_NOISE, "noise"), (C_RAW, "post_raw")):
            if cell in dsl:
                rec[f"{lab}_vs_pre_gate"] = mod85._diff_entry(dsl[cell],
                                                              dsl[PRE_GATE])
        regimes["slices"][label] = rec

    # ---- thinned control panel (only if a control pass ran thinned)
    thin_block: dict = {"verdict": "NOT_EXECUTED"} if not thinned else {
        "why": ("the two control cells are cheap only on a thinned fold "
                "schedule on this shared machine. Every fold kept is the same "
                "fold of the full run (same training set, same estimator, same "
                "seed), so this is a subsample of the panel. The PRIMARY "
                "contrast is recomputed on exactly these days beside the "
                "controls, so the comparison is like for like; the full-panel "
                "primary number above remains the headline."),
    }
    if thinned:
        tcells: list[str] = []
        tpanel = panel
        for tag, frame in thinned.items():
            tpanel = tpanel.join(frame, on=key, how="inner")
            tcells.extend(PASS_CELLS[tag])
        tpd = mod85._per_day_mae(tpanel, [BASELINE, PRE_GATE, POST_GATE, C_WX,
                                          C_PV, *tcells])
        tdaily = {c: mod85._pooled_daily(tpd, c)
                  for c in (PRE_GATE, POST_GATE, C_WX, C_PV, *tcells)}
        thin_block.update({
            "n_days": int(len(tdaily[PRE_GATE])),
            "n_slots": tpanel.height,
            "n_days_full_panel": int(panel["test_day"].n_unique()),
            "cells": tcells,
            "wx_innov_vs_pre_gate": mod85._diff_entry(tdaily[C_WX],
                                                     tdaily[PRE_GATE]),
            "pv_innov_vs_pre_gate": mod85._diff_entry(tdaily[C_PV],
                                                      tdaily[PRE_GATE]),
            "post_gate_vs_pre_gate": mod85._diff_entry(tdaily[POST_GATE],
                                                       tdaily[PRE_GATE]),
        })
        for cell, lab in ((C_NOISE, "noise"), (C_RAW, "post_raw")):
            if cell in tdaily:
                thin_block[f"{lab}_vs_pre_gate"] = mod85._diff_entry(
                    tdaily[cell], tdaily[PRE_GATE])
                thin_block[f"wx_innov_vs_{lab}"] = mod85._diff_entry(
                    tdaily[C_WX], tdaily[cell])

    # ---- physical positive control
    phys_build = build_phys(mod85, mod89)
    price_days = set(panel["test_day"].unique().to_list())
    phys = analyse_phys(mod89, phys_build["panel"], wx_cols, price_days)
    phys["construction_checks"] = phys_build["checks"]

    p89 = json.loads(P89_JSON.read_text())
    p89_gain = p89["gates"]["gate51_physical_gain_crossfit_common"]["delta_mw"]

    # ---- Gate 102C: innovation non-degenerate, measured on the scored slots
    ratios = {n: v["mean_abs_over_field_sd"]
              for n, v in innov_on_scored["wx"].items()}
    gate_c = {
        "rule": (f"the WX innovation is degenerate if every channel's mean "
                 f"|innovation| is below {DEGENERATE_FRAC:.0%} of that "
                 "channel's pre-gate field SD, on the scored slots"),
        "mean_abs_over_field_sd": ratios,
        "wx_missing_on_any_scored_slot":
            innov_on_scored["wx_missing_on_any_scored_slot"],
    }
    gate_c["verdict"] = ("PASS" if any((v or 0.0) >= DEGENERATE_FRAC
                                       for v in ratios.values())
                         and innov_on_scored["wx_missing_on_any_scored_slot"] == 0
                         else "FAIL")
    gate_c["consequence_if_fail"] = (
        "The two GPV caches carry the same field; that is a data defect and no "
        "branch below may be read.")

    # ---- Gate 102B: physical positive control
    best = max(("C_pre_plus_pv_innov", "D_pre_plus_wx_innov"),
               key=lambda k: phys["gains_vs_A_common"][k]["delta_mw"] or -1e9)
    bg = phys["gains_vs_A_common"][best]
    repro_max = max((v["max_abs_diff_mw"] or 0.0)
                    for v in phys["gate102b_reproduces_phase89"].values())
    gate_b = {
        "rule": (f"PASS requires the better of cells C/D to beat cell A with a "
                 f"bootstrap CI excluding zero AND to recover >= "
                 f"{RECOVER_FRAC:.0%} of phase89's {p89_gain} MW pre->post gain; "
                 f"and cell A must reproduce phase89's own cross-fitted "
                 f"prediction to {PHYS_REPRO_TOL} MW"),
        "phase89_pre_to_post_gain_mw": p89_gain,
        "best_cell": best, **bg,
        "recovered_fraction_of_phase89_gain": _j(
            (bg["delta_mw"] or 0.0) / p89_gain, 3) if p89_gain else None,
        "cell_A_reproduction_max_abs_diff_mw": _j(repro_max, 9),
        "cell_A_reproduces": bool(repro_max <= PHYS_REPRO_TOL),
    }
    gate_b["verdict"] = ("PASS" if bg["excludes_zero"] and (bg["delta_mw"] or 0) > 0
                         and (gate_b["recovered_fraction_of_phase89_gain"] or 0)
                         >= RECOVER_FRAC and gate_b["cell_A_reproduces"]
                         else "FAIL")
    gate_b["consequence_if_fail"] = (
        "The innovation construction is wrong. Report a broken construction, "
        "NOT a price null.")
    log.info("GATE 102B (physical positive control): %s -- %s +%.2f MW %s "
             "(%.0f%% of phase89's %.2f MW)", gate_b["verdict"], best,
             bg["delta_mw"] or 0.0, bg["ci95"],
             100.0 * (gate_b["recovered_fraction_of_phase89_gain"] or 0.0),
             p89_gain)

    # ---- Gate 102D: the price branch
    wx = contrasts["wx_innov_vs_pre_gate"]
    lo, hi = wx["ci95"]
    p3 = wx["dm_l3"]["p_hln"]
    p9 = wx["dm_l9"]["p_hln"]
    sig_pos = (lo is not None and lo > 0 and p3 is not None and p3 < 0.05
               and p9 is not None and p9 < 0.05)
    sig_neg = (hi is not None and hi < 0 and p3 is not None and p3 < 0.05)
    if gate_b["verdict"] != "PASS" or gate_c["verdict"] != "PASS":
        branch, reading = "undetermined", (
            "a prerequisite gate failed; the price contrast is not interpretable")
    elif sig_pos and lo > TICK:
        branch, reading = "a", (
            "LEARNER LIMITATION: handed the innovation directly, the learner "
            "converts it into more than one price tick of pooled accuracy. The "
            "manuscript's reading of the post-gate price null must be weakened "
            "and the tick-margin equivalence claim can no longer be presented "
            "as an information statement.")
    elif sig_pos:
        branch, reading = "c", (
            "SMALL BUT REAL: the innovation does add out-of-sample price "
            "accuracy, but bounded below one tick. The economic conclusion "
            "survives; the ladder-design limitation now has a number.")
    elif sig_neg:
        branch, reading = "d", (
            "NEGATIVE: the extra columns hurt. Read against the matched-variance "
            "noise placebo before calling the innovation anti-informative.")
    else:
        branch, reading = "b", (
            "SIGNAL NULL, the strong version: the same innovation moves a "
            "physical forecast by tens of MW and, handed to the ladder's own "
            "learner in the easiest possible form, moves the price forecast by "
            "nothing. The post-gate null is a property of the signal's "
            "relevance to price, not of the learner or of the representation.")
    gate_d = {
        "materiality_margin_jpykwh": TICK,
        "margin_provenance": ("one JEPX price tick; phase85's own pre-registered "
                              "equivalence margin for this contrast"),
        "ladder_measured_post_gate_delta": _j(
            contrasts["post_gate_vs_pre_gate_REPRODUCTION"]["delta"], 4),
        "delta_wx": wx["delta"], "ci95": wx["ci95"],
        "dm_p_l3": p3, "dm_p_l9": p9,
        "noise_placebo": (
            {"panel": "full", **{k: contrasts["noise_vs_pre_gate"][k]
                                 for k in ("delta", "ci95", "n_days")}}
            if C_NOISE in daily else
            {"panel": "thinned subsample of the same folds",
             **{k: thin_block["noise_vs_pre_gate"][k]
                for k in ("delta", "ci95", "n_days")},
             "matched_primary_on_the_same_days":
                 thin_block["wx_innov_vs_pre_gate"]["delta"]}
            if "noise_vs_pre_gate" in thin_block else
            {"panel": None, "note": ("no price-side placebo was built; the "
                                     "physical arm's cell E is the only "
                                     "placebo yardstick available")}),
        "branch": branch, "reading": reading,
    }
    log.info("GATE 102D: branch %s -- delta_wx %+.4f %s (DM p L3=%s L9=%s)",
             branch, wx["delta"] or 0.0, wx["ci95"], p3, p9)

    # ---- Gate 102E: representation
    gate_e = ({
        "rule": ("the differenced basis minus the raw-union basis; both span the "
                 "same information, so a gap is a representation effect"),
        **contrasts["wx_innov_vs_post_raw"],
        "wx_innov_delta": wx["delta"],
        "post_raw_delta": contrasts["post_raw_vs_pre_gate"]["delta"],
    } if C_RAW in daily else {
        "panel": "thinned subsample of the same folds",
        "rule": ("the differenced basis minus the raw-union basis; both span the "
                 "same information, so a gap is a representation effect"),
        **contrasts.get("wx_innov_vs_post_raw", {}),
        **{k: v for k, v in thin_block.items()
           if k in ("wx_innov_vs_post_raw", "post_raw_vs_pre_gate",
                    "wx_innov_vs_pre_gate", "n_days")},
    } if "wx_innov_vs_post_raw" in thin_block else {
        "verdict": "NOT_EXECUTED",
        "reason": ("the raw-union and noise-placebo cells are build pass B, which "
                   "did not complete for all nine areas on this machine; a "
                   "partial pass would change the panel, so it is reported "
                   "absent rather than on a different sample"),
    })

    p90 = json.loads(P90_JSON.read_text())
    out = {
        "phase": "102",
        "title": ("Is the post-gate price null a property of the signal or of "
                  "the learner? The innovation fed directly to the price model"),
        "question": ("Section 5.3 reports that the excluded run adds a lot to the "
                     "physical target and nothing detectable to the price. A "
                     "referee will ask whether the information is there and the "
                     "learner cannot extract it. The ladder's post-gate cell "
                     "REPLACES the pre-gate field; a tree cannot form the "
                     "difference of two columns. This hands the difference over "
                     "explicitly, out of sample."),
        "preregistration": ("branches, margin and the positive-control "
                            "requirement are in this script's module docstring, "
                            "written before any number was computed"),
        "panel": {"n_slots": panel.height, "n_areas": len(areas),
                  "n_test_days": int(panel["test_day"].n_unique()),
                  "first_test_day": panel["test_day"].min(),
                  "last_test_day": panel["test_day"].max()},
        "build_passes": {
            "cells_per_pass": {k: list(v) for k, v in PASS_CELLS.items()},
            "pass_present": pass_state,
            "note": ("the build is split into passes because the machine is "
                     "shared; a pass that did not cover all nine areas is "
                     "reported ABSENT, never partial"),
        },
        "null_counts_in_predictions": nulls,
        "innovation_on_scored_slots": innov_on_scored,
        "in_sample_linear_reference": linear_reference,
        "gate102a_bit_reproduction": gate_a,
        "gate102c_innovation_non_degenerate": gate_c,
        "gate102b_physical_positive_control": gate_b,
        "gate102d_price_branch": gate_d,
        "gate102e_representation": gate_e,
        "price_tiers_improvement_vs_price_only": tiers,
        "price_contrasts": contrasts,
        "price_per_area_delta_vs_pre_gate": per_area,
        "price_pv_subpanel": subpanel,
        "price_regime_slices_descriptive": regimes,
        "thinned_control_panel": thin_block,
        "learner_uses_the_innovation_descriptive": (
            {a: {k: v for k, v in rec.items() if k != "folds"}
             for a, rec in json.loads(OUT_IMP.read_text())["areas"].items()}
            | {"_design": json.loads(OUT_IMP.read_text())["design"],
               "_why": ("a null can mean the tree looked and found nothing or "
                        "that it never looked; this is which")}
            if OUT_IMP.exists() else {"verdict": "NOT_EXECUTED"}),
        "physical_positive_control": phys,
        "reference_numbers": {
            "phase85_gate_timing_delta": (
                json.loads(P85_JSON.read_text())["arm_cw_primary"]
                ["gate46_47_48_gate_timing"]["delta"]),
            "phase89_gate51_delta_mw": p89_gain,
            "phase90_price_residual_slope": p90["regressions"]
                ["price_residual_on_innovation"]["slope"],
            "phase90_price_r2_increment": p90["regressions"]
                ["price_residual_on_innovation"]["r2_increment"],
            "phase90_in_sample_ceiling_full_panel": p90
                ["implied_price_value_full_panel"]["implied_mae_reduction"],
            "phase90_refit_on_full_panel_would_give": p90
                ["implied_price_value_full_panel"]["refit_on_full_panel_would_give"],
            "yardstick_oracle_minus_pre_gate": {
                "value": _j(
                    json.loads(P85_JSON.read_text())["arm_cw_primary"]["tiers"]
                    ["oracle_cw"]["improvement"]
                    - json.loads(P85_JSON.read_text())["arm_cw_primary"]["tiers"]
                    [PRE_GATE]["improvement"], 4),
                "why": ("phase85's own scale for what SWAPPING the four weather "
                        "columns for four different ones is worth on this panel: "
                        "realised weather (oracle) minus the pre-gate forecast. It "
                        "is the closest available yardstick for how large a "
                        "four-column change can be here, and it is NEGATIVE -- "
                        "realised weather scores worse than the forecast."),
            },
        },
        "inputs": {
            "phase85_per_slot": str(P85_PARQUET),
            "phase85_json": str(P85_JSON),
            "phase89_per_slot": str(P89_SLOTS),
            "phase89_json": str(P89_JSON),
            "phase90_json": str(P90_JSON),
            "phase102_physical_per_slot": str(PHYS_PARQUET),
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "seed": mod85.SEED, "n_boot": mod85.N_BOOT,
                     "boot_block_days": mod85.BOOT_BLOCK,
                     "noise_seed": NOISE_SEED,
                     "generated_utc": datetime.now(tz=JST).isoformat()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    for c, e in tiers.items():
        log.info("  %-20s imp=%+.4f CI=[%+.4f,%+.4f]", c, e["improvement"],
                 e["ci95"][0], e["ci95"][1])
    for k, v in contrasts.items():
        log.info("  %-38s %+.4f [%+.4f,%+.4f] p3=%s", k, v["delta"],
                 v["ci95"][0], v["ci95"][1], v["dm_l3"]["p_hln"])
    return 0 if gate_a["verdict"] == "PASS" else 1


# ------------------------------------------------------------------------ main
mod_p89 = None


def main() -> int:
    global mod_p89  # noqa: PLW0603 - the rebound _design needs phase89's constants
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=("smoke", "build-price", "build-phys", "importance",
                             "analyse"))
    ap.add_argument("--areas", default="")
    ap.add_argument("--pass", dest="pass_", default="a", choices=("a", "b"))
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--thin", type=int, default=None,
                    help="widen phase85 STEP to N days: a matched subsample of "
                         "the panel's own folds, for the cheap control cells")
    args = ap.parse_args()
    mod85 = _load_module(P85_PATH, "p85")
    mod_p89 = _load_module(P89_PATH, "p89")

    global FIT_CELLS  # noqa: PLW0603 - the pass selects which cells are fitted
    FIT_CELLS = PASS_CELLS[args.pass_]
    if args.step == "smoke":
        mod85.MAX_FOLDS = args.max_folds or 3
        if args.max_folds is None:
            mod85.INITIAL_TRAIN_END = datetime(2026, 6, 20, tzinfo=JST)
        areas = [a for a in args.areas.split(",") if a] or ["kyushu"]
        t0 = time.time()
        d = build_price(mod85, areas, CKPT_SMOKE)
        log.info("smoke: %d folds x %d areas in %.1f s -> %.3f s/fold/area",
                 mod85.MAX_FOLDS, len(areas), time.time() - t0,
                 (time.time() - t0) / (mod85.MAX_FOLDS * len(areas)))
        log.info("smoke diagnostics: %s", json.dumps(d, indent=1)[:1500])
        return 0
    if args.step == "build-price":
        areas = [a for a in args.areas.split(",") if a] or list(mod85.AREA_COORDS)
        tag = args.pass_
        if args.thin and args.thin > 1:
            # phase85's own schedule with STEP widened: every retained test day
            # is a real test day of the full run and its training set is
            # unchanged, so the fits are the same fits
            mod85.STEP = timedelta(days=args.thin)
            tag = f"{tag}_thin"
        build_price(mod85, areas, _ckpt(tag))
        return 0
    if args.step == "importance":
        areas = [a for a in args.areas.split(",") if a] or list(mod85.AREA_COORDS)
        build_importance(mod85, areas)
        return 0
    if args.step == "build-phys":
        b = build_phys(mod85, mod_p89)
        log.info("construction checks: %s", json.dumps(b["checks"], indent=1))
        price_days = set(pl.read_parquet(P85_PARQUET, columns=["test_day"])
                         ["test_day"].unique().to_list())
        r = analyse_phys(mod_p89, b["panel"],
                         list(mod85.DEFAULT_WEATHER_COLS), price_days)
        log.info("physical arm (dry run):\n%s", json.dumps(
            {k: r[k] for k in ("levels_common_panel", "gains_vs_A_common",
                               "gate102b_reproduces_phase89")}, indent=1))
        return 0
    return analyse(mod85, mod_p89)


if __name__ == "__main__":
    sys.exit(main())
