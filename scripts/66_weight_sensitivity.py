r"""Phase 91: does the static capacity weighting carry the result?

WHY
---
The capacity-weighted field weights each prefecture point by its share of the
area's registered photovoltaic capacity, taken from one register snapshot and
held fixed over two years of delivery days. The seventh review asked whether
the headline survives a different weighting. Three answers are possible and
only two are computable:

  2024-09    the weights the paper uses
  2025-12    a later register snapshot, already in the reference CSV
  equal      1/n per point, which throws the register away entirely

Monthly or quarterly time-varying weights are NOT computable here and this is a
data-model fact, not a choice: ``AreaPoint`` has no time axis and the reference
CSV holds exactly two workbook snapshots. That limitation is recorded in the
manuscript rather than worked around.

WHAT IS RE-ESTIMATED, AND WHAT IS NOT
-------------------------------------
Three cells per area: the price-only baseline, the pre-gate headline and the
first excluded run. Not the other eleven, and nothing downstream -- the floor
models, the physical target and the second-provider check keep the 2024-09
weights, because the question is whether the headline and the gate contrast
move, not whether every result can be rebuilt three times.

The 2024-09 arm is re-estimated too, even though phase 85 already has it. If
this harness cannot reproduce phase 85's headline to the fourth decimal on the
same weights, its other two arms mean nothing; Gate 60 checks exactly that.

Gate 45c, which pins the capacity-weighted cells to phase 74 bit for bit, does
NOT apply to the re-weighted arms and cannot: different weights are different
inputs. The JSON says so explicitly rather than leaving a reader to assume the
reproduction guarantee extends here.

The loaders, the design matrix, the fold schedule, the learner and the seed all
come from the phase 85 module by import, with the weight vintage rebound. A
copied fold loop would be a second implementation to keep in step, and the
comparison only means something if everything except the weights is identical.

Branches are fixed in
docs/research/2026-08-13_phase91_weight_sensitivity_precommit.md.

CLI (from the repository root):
  uv run python scripts/66_weight_sensitivity.py \
      --step build --weights 2025-12,equal --areas tokyo,kyushu
  uv run python scripts/66_weight_sensitivity.py \
      --step analyse
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase91")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase91_checkpoints"
OUT_JSON = RUN / "phase91_weight_sensitivity.json"
P85_JSON = RUN / "phase85_exact_ladder.json"
PRECOMMIT = "docs/research/2026-08-13_phase91_weight_sensitivity_precommit.md"

PHASE85_MODULE = "10_primary_ladder_cw"
WEIGHTS = ("2024-09", "2025-12", "equal")
CELLS = ("price_only", "msm_d2_12z_cw", "msm_d1_00z_cw")
HEADLINE, POST_GATE = "msm_d2_12z_cw", "msm_d1_00z_cw"
REPRO_TOL = 5e-4  # Gate 60: the 2024-09 arm must match phase85 to this
EQUIV_MARGIN = 0.01   # one JEPX price tick, the pre-registered margin


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


def _equal_weight_points(real_area_points):
    """Wrap ``area_points`` so every point in an area carries 1/n."""

    def _wrapped(vintage: str = "2024-09", csv_path=None):
        base = real_area_points(vintage, csv_path) if csv_path is not None \
            else real_area_points(vintage)
        return {a: [dataclasses.replace(p, weight=1.0 / len(pts)) for p in pts]
                for a, pts in base.items()}

    return _wrapped


def _build_area(mod, area: str, weight: str, max_folds: int | None = None) -> None:
    """phase85's panel and fold loop, three cells, one weight vintage.

    ``max_folds`` exists for smoke-testing the wiring without paying for a full
    arm. A truncated run writes to a DIFFERENT checkpoint name, so it can never
    be picked up by the analysis as if it were complete.
    """
    suffix = "" if max_folds is None else f"__SMOKE{max_folds}"
    ckpt = CKPT / f"{weight}__{area}{suffix}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    diagnostics: dict = {}
    lat, lon = mod.AREA_COORDS[area]
    price = (mod.load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=mod.RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "area", "price_yen_kwh"))
    era5 = mod._load_era5_sp(area, lat, lon, diagnostics)  # noqa: SLF001
    histfc = mod._load_histfc_sp(area, lat, lon, diagnostics)  # noqa: SLF001
    df = mod.attach_hourly_weather(price.sort("datetime_jst"), era5)
    df = mod.attach_hourly_features(
        df, histfc, feature_cols=[f"{c}_fc" for c in mod.DEFAULT_WEATHER_COLS])
    for n_prev, sfx in ((1, "_pd1"), (2, "_pd2")):
        df = mod.attach_hourly_features(
            df, mod._load_prev(area, n_prev),  # noqa: SLF001
            feature_cols=[f"{c}{sfx}" for c in mod.DEFAULT_WEATHER_COLS])
    for tier in mod.SP_PINNED:
        df = mod.attach_hourly_features(
            df, mod._load_pinned_single(tier, area),  # noqa: SLF001
            feature_cols=[f"{c}_{tier}" for c in mod.DEFAULT_WEATHER_COLS])
    for tier in mod.CW_JMA_TIERS:
        df = mod.attach_hourly_features(
            df, mod._load_cw_jma(tier, area),  # noqa: SLF001
            feature_cols=[f"{c}_{tier}" for c in mod.DEFAULT_WEATHER_COLS])
    df = mod.attach_hourly_features(
        df, mod._load_cw_era5(area),  # noqa: SLF001
        feature_cols=[f"{c}_{mod.CW_ERA5_REAL}" for c in mod.DEFAULT_WEATHER_COLS])

    df, cells = mod._design(df)  # noqa: SLF001
    # The row set must be phase85's exactly, or the comparison is confounded by
    # a different panel. Weights touch only the CW feature VALUES, and the
    # drop_nulls below is over the SP cells alone, so the surviving rows are
    # identical across weight vintages by construction.
    sp_cols = [c for name in mod.SP_CELLS for c in cells[name]]
    need = sorted({*sp_cols, "price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date()
                  .is_between(mod.PANEL_START, mod.ERA5_END))
          .sort("datetime_jst"))
    log.info("[%s] %s panel: %d slots", weight, area, df.height)

    wanted = {c: cells[c] for c in CELLS}
    flag_col = {HEADLINE: f"{mod.DEFAULT_WEATHER_COLS[0]}_{HEADLINE}",
                POST_GATE: f"{mod.DEFAULT_WEATHER_COLS[0]}_{POST_GATE}"}
    panel_last = df["datetime_jst"].max().date()
    rows: list[dict] = []
    t0, n_folds = time.time(), 0
    for k in range(max_folds if max_folds is not None else mod.MAX_FOLDS):
        test_day = (mod.INITIAL_TRAIN_END + mod.STEP * k + mod.VAL_WINDOW).date()
        if test_day > panel_last:
            break
        train = df.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = (df.filter(pl.col("datetime_jst").dt.date() == test_day)
                  .sort("datetime_jst"))
        if train.height < mod.MIN_TRAIN_ROWS or test.height == 0:
            continue
        preds = {n: mod._fit(train, test, cols)  # noqa: SLF001
                 for n, cols in wanted.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        ok = {t: (~test[c].is_nan() & test[c].is_not_null()).to_list()
              for t, c in flag_col.items()}
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": y[s],
                     **{f"cw_ok_{t}": bool(ok[t][s]) for t in flag_col},
                     **{f"y_pred_{n}": float(preds[n][s]) for n in wanted}}
                    for s in range(test.height))
        n_folds += 1
        if n_folds % 100 == 0:
            log.info("[%s] %s fold %d | %.2f s/fold", weight, area, n_folds,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
    log.info("[%s] %s done: %d folds, %.0f s", weight, area, n_folds,
             time.time() - t0)


def build(weights: list[str], areas: list[str],
          max_folds: int | None = None) -> int:
    mod = importlib.import_module(PHASE85_MODULE)
    if mod.WEIGHT_VINTAGE != "2024-09":
        log.error("phase85 WEIGHT_VINTAGE is %r, expected '2024-09'; the "
                  "comparison would not be anchored where the paper is",
                  mod.WEIGHT_VINTAGE)
        return 2
    real_area_points = mod.area_points
    try:
        for w in weights:
            if w == "equal":
                mod.area_points = _equal_weight_points(real_area_points)
                mod.WEIGHT_VINTAGE = "2024-09"  # geometry from the same snapshot
                log.info("weights: EQUAL (1/n per point, register discarded)")
            else:
                mod.area_points = real_area_points
                mod.WEIGHT_VINTAGE = w
                log.info("weights: register vintage %s", w)
            if max_folds is not None:
                log.warning("SMOKE TEST: %d folds only -- these checkpoints are "
                            "named __SMOKE and are not analysable", max_folds)
            for a in areas:
                _build_area(mod, a, w, max_folds)
    finally:
        mod.area_points = real_area_points
        mod.WEIGHT_VINTAGE = "2024-09"
        log.info("restored phase85 globals")
    return 0


# ------------------------------------------------------------------ analysis
def _pooled_daily(panel: pl.DataFrame, cell: str) -> np.ndarray:
    """phase85's pooled daily improvement, on the CW-complete subset."""
    sub = panel.filter(pl.col(f"cw_ok_{cell}"))
    per_day = (sub.with_columns(
        (pl.col("y_pred_price_only") - pl.col("y_actual")).abs().alias("ae_base"),
        (pl.col(f"y_pred_{cell}") - pl.col("y_actual")).abs().alias("ae_cell"))
        .group_by("area", "test_day")
        .agg(pl.col("ae_base").mean().alias("mae_base"),
             pl.col("ae_cell").mean().alias("mae_cell"))
        .with_columns((pl.col("mae_base") - pl.col("mae_cell")).alias("imp"))
        .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day"))
    return np.asarray(per_day["imp"], dtype=float)


def _paired_daily(panel: pl.DataFrame, a: str, b: str) -> np.ndarray:
    sub = panel.filter(pl.col(f"cw_ok_{a}") & pl.col(f"cw_ok_{b}"))
    per_day = (sub.with_columns(
        (pl.col(f"y_pred_{a}") - pl.col("y_actual")).abs().alias("ae_a"),
        (pl.col(f"y_pred_{b}") - pl.col("y_actual")).abs().alias("ae_b"))
        .group_by("area", "test_day")
        .agg(pl.col("ae_a").mean().alias("mae_a"),
             pl.col("ae_b").mean().alias("mae_b"))
        .with_columns((pl.col("mae_a") - pl.col("mae_b")).alias("d"))
        .group_by("test_day").agg(pl.col("d").mean()).sort("test_day"))
    return np.asarray(per_day["d"], dtype=float)


def analyse() -> dict:
    mod = importlib.import_module(PHASE85_MODULE)
    out: dict = {
        "phase": 91, "title": "Capacity-weight sensitivity of the headline and "
                              "the gate contrast",
        "precommit_doc": PRECOMMIT,
        "scope": {
            "cells_re_estimated": list(CELLS),
            "not_propagated": ("floor models, the physical target and the "
                               "second-provider check keep the 2024-09 weights"),
            "gate45c_applicability": (
                "Gate 45c pins the capacity-weighted cells to phase74 bit for "
                "bit and does NOT extend to the re-weighted arms: different "
                "weights are different inputs. The 2024-09 arm here is checked "
                "against phase85 instead, by Gate 60"),
            "time_varying_weights": (
                "not computable: AreaPoint carries no time axis and the "
                "reference CSV holds two workbook snapshots, so a monthly or "
                "quarterly weight series cannot be built. Recorded as a "
                "limitation rather than approximated"),
        },
        "arms": {}, "gates": {},
    }
    for w in WEIGHTS:
        files = [f for f in sorted(CKPT.glob(f"{w}__*.parquet"))
                 if "__SMOKE" not in f.name]
        if not files:
            log.warning("no checkpoints for weights %s -- arm skipped", w)
            continue
        panel = pl.concat([pl.read_parquet(f) for f in files])
        arm: dict = {"n_areas": len(files), "n_slots": panel.height,
                     "n_days": panel["test_day"].n_unique(), "tiers": {}}
        for cell in (HEADLINE, POST_GATE):
            d = _pooled_daily(panel, cell)
            lo, hi = mod._block_boot_ci(d)  # noqa: SLF001
            arm["tiers"][cell] = {"improvement": _j(float(d.mean())),
                                  "ci95": [_j(lo), _j(hi)], "n_days": len(d)}
        # Sign convention MUST match phase85, whose gate_timing_delta is
        # (excluded-run improvement) - (headline improvement), i.e. positive when
        # the excluded run is better. _paired_daily returns
        # MAE(headline) - MAE(excluded), which is the same quantity with the same
        # sign: the improvements share one baseline, so it cancels. An earlier
        # version negated this and printed -0.0026 against phase85's +0.0026 --
        # the same number with the opposite meaning, in a file whose whole job is
        # to be compared with phase85.
        dg = _paired_daily(panel, HEADLINE, POST_GATE)
        lo, hi = mod._block_boot_ci(dg)  # noqa: SLF001
        # The paper's headline is an EQUIVALENCE claim at one price tick, so the
        # sensitivity has to test that claim and not only the point estimate's
        # sign. Same bootstrap TOST as the inference-sensitivity phase: draws of
        # the mean paired difference, and the test rejects a difference as large
        # as the margin when neither tail reaches it.
        rng = np.random.default_rng(mod.SEED)
        n = len(dg)
        n_blocks = int(np.ceil(n / mod.BOOT_BLOCK))
        starts = rng.integers(0, n, size=(mod.N_BOOT, n_blocks))
        idx = (starts[:, :, None] + np.arange(mod.BOOT_BLOCK)[None, None, :]) % n
        draws = dg[idx.reshape(mod.N_BOOT, -1)[:, :n]].mean(axis=1)
        p_tost = max(float(np.mean(draws <= -EQUIV_MARGIN)),
                     float(np.mean(draws >= EQUIV_MARGIN)))
        arm["equivalence_at_tick"] = {
            "margin": EQUIV_MARGIN,
            "p_tost": _j(max(p_tost, 1.0 / mod.N_BOOT), 5),
            "equivalent": bool(p_tost < 0.05),
            "ci90": [_j(float(np.quantile(draws, 0.05))),
                     _j(float(np.quantile(draws, 0.95)))],
        }
        arm["gate_timing_delta"] = {
            "delta": _j(float(dg.mean())), "ci95": [_j(lo), _j(hi)],
            "n_days": len(dg),
            "definition": ("MAE(headline) - MAE(excluded run), pooled daily; "
                           "positive means the EXCLUDED run is better, matching "
                           "phase85's gate_timing_delta"),
        }
        out["arms"][w] = arm
        log.info("[%s] headline %.4f %s | gate delta %+.4f %s", w,
                 arm["tiers"][HEADLINE]["improvement"],
                 arm["tiers"][HEADLINE]["ci95"],
                 arm["gate_timing_delta"]["delta"],
                 arm["gate_timing_delta"]["ci95"])

    # ---- Gate 60: does this harness reproduce phase85 on the paper's weights?
    if "2024-09" in out["arms"] and P85_JSON.exists():
        p85 = json.loads(P85_JSON.read_text())["arm_cw_primary"]
        ref = p85["tiers"][HEADLINE]["improvement"]
        got = out["arms"]["2024-09"]["tiers"][HEADLINE]["improvement"]
        ref_post = p85["tiers"][POST_GATE]["improvement"]
        got_post = out["arms"]["2024-09"]["tiers"][POST_GATE]["improvement"]
        ref_gate = p85["gate46_47_48_gate_timing"]["delta"]
        got_gate = out["arms"]["2024-09"]["gate_timing_delta"]["delta"]
        deltas = {"headline": abs(got - ref), "excluded_run": abs(got_post - ref_post),
                  "gate_timing": abs(got_gate - ref_gate)}
        out["gates"]["gate60_harness_reproduces_phase85"] = {
            "phase85_headline": ref, "phase91_headline": got,
            "phase85_excluded_run": ref_post, "phase91_excluded_run": got_post,
            "phase85_gate_timing": ref_gate, "phase91_gate_timing": got_gate,
            "abs_differences": {k: _j(v, 6) for k, v in deltas.items()},
            "abs_difference": _j(abs(got - ref), 6), "tol": REPRO_TOL,
            "ok": max(deltas.values()) <= REPRO_TOL,
            "reading": ("the re-implementation is anchored on all three "
                        "quantities INCLUDING THE SIGN of the gate-timing "
                        "delta: whatever the other arms show is a weight "
                        "effect, not a harness effect"
                        if max(deltas.values()) <= REPRO_TOL else
                        "THE HARNESS DOES NOT REPRODUCE phase85 -- every other "
                        "number in this file is uninterpretable until that is "
                        "explained"),
        }

    # ---- Gate 59: do the three weightings agree? --------------------------
    arms = out["arms"]
    if len(arms) >= 2:
        his = [(w, a["tiers"][HEADLINE]) for w, a in arms.items()]
        overlap = all(
            x[1]["ci95"][0] <= y[1]["ci95"][1] and y[1]["ci95"][0] <= x[1]["ci95"][1]
            for i, x in enumerate(his) for y in his[i + 1:])
        signs = {w: (a["gate_timing_delta"]["delta"] or 0.0) >= 0
                 for w, a in arms.items()}
        straddle = {w: (a["gate_timing_delta"]["ci95"][0] <= 0
                        <= a["gate_timing_delta"]["ci95"][1])
                    for w, a in arms.items()}
        spread = max(a["tiers"][HEADLINE]["improvement"] for a in arms.values()) - \
            min(a["tiers"][HEADLINE]["improvement"] for a in arms.values())
        equiv = {w: bool(a["equivalence_at_tick"]["equivalent"])
                 for w, a in arms.items()}
        agree = (overlap and len(set(signs.values())) == 1
                 and all(straddle.values()) and all(equiv.values()))
        out["gates"]["gate59_weight_sensitivity"] = {
            "arms": list(arms),
            # The manuscript names this arm in prose ("under a later 2025-12
            # snapshot"). Binding that label to the arm LIST let any member
            # satisfy the check, so swapping it for the paper's own vintage
            # passed and read as a real sensitivity result. Named alone here.
            "later_register_arm": next(
                (w for w in arms if w not in ("2024-09", "equal")), None),
            "headline_ci_overlap_all_pairs": overlap,
            "headline_spread": _j(spread),
            "gate_delta_sign_by_arm": signs,
            "gate_delta_straddles_zero_by_arm": straddle,
            # The headline is an equivalence claim, so the gate tests THAT under
            # each weighting rather than only the point estimate's sign. An
            # earlier version checked overlap and sign only, and would have
            # passed while the pre-registered equivalence failed under one arm.
            "equivalence_at_tick_by_arm": equiv,
            "branch": "a" if agree else "b",
            "reading": ("branch a: the static weighting is not carrying the "
                        "result -- the headline intervals overlap under all "
                        "three weightings, the gate-timing delta straddles zero "
                        "in each, and the one-tick equivalence holds in each"
                        if agree else
                        "branch b: the weighting matters. Promote static "
                        "weights to a primary limitation and show which arm "
                        "changes what"),
        }

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "seed": mod.SEED, "n_boot": mod.N_BOOT,
                       "boot_block_days": mod.BOOT_BLOCK,
                       "weights_csv": "data/reference/area_points_capacity.csv"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--weights", default=",".join(WEIGHTS))
    ap.add_argument("--areas", default="")
    ap.add_argument("--max-folds", type=int, default=None,
                    help="smoke test only: truncate the fold loop and write to "
                         "__SMOKE checkpoints the analysis ignores")
    args = ap.parse_args()
    if args.step == "build":
        mod = importlib.import_module(PHASE85_MODULE)
        weights = [w.strip() for w in args.weights.split(",") if w.strip()]
        unknown = [w for w in weights if w not in WEIGHTS]
        if unknown:
            log.error("unknown weightings %s (known: %s)", unknown, list(WEIGHTS))
            return 2
        areas = ([a.strip() for a in args.areas.split(",") if a.strip()]
                 or list(mod.AREA_COORDS))
        return build(weights, areas, args.max_folds)
    if not CKPT.exists():
        log.error("missing %s", CKPT)
        return 2
    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    for g, v in out["gates"].items():
        log.info("%s: %s", g.upper(), v["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
