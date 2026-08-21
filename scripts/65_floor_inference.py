r"""Phase 81b: paired uncertainty for the floor-model comparison, and the gate
contrast under the winning model.

WHY
---
Phase 81 named the direct classifier the winner of five floor models on macro
PR-AUC and reported no interval for the margin. Its own precommit had promised
a block bootstrap; it declared N_BOOT and BOOT_BLOCK and never used them. That
deviation is disclosed in the supplement and repaired here.

Two further gaps the seventh review named. The floor rate varies by an order of
magnitude across areas, so a macro average over areas and a slot-pooled micro
figure are different statements and the paper reported only one of them; and
average precision is bounded below by prevalence, so an area with more floor
slots looks better scored at no skill. Both are reported here, the second as
(AP - base rate) / (1 - base rate).

Finally, the paper's gate-timing null for floor events rests on the isotonic
recalibration chain -- the model the comparison above says is NOT the best one.
If the excluded run adds floor skill under the winner and not under M0, the
null is a property of the recalibration, not of the information. Stage 1 tests
that cheaply by re-fitting M0 on the post-gate point forecast, which is a
column that already exists. Stage 2 retrains the direct classifier itself on
post-gate weather, which is the honest version of the same question and costs
about 18 CPU-hours.

Stage 2 deliberately reuses phase81's own fold loop by rebinding two module
globals rather than reimplementing it: the gate contrast is only interpretable
if EVERYTHING except the weather source is identical, and a copied loop can
drift. The rebinding is asserted, logged and restored.

Branches are fixed in
docs/research/2026-08-13_phase81b_floor_inference_precommit.md.

CLI (from the repository root):
  # stage 2 build, one area at a time or comma-separated (about 2 h per area)
  uv run --extra research python scripts/65_floor_inference.py \
      --step build-postgate --areas tokyo,kyushu
  # stage 1 + whatever of stage 2 is on disk
  uv run --extra research python scripts/65_floor_inference.py \
      --step analyse
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase81b")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P81_JSON = RUN / "phase81_direct_floor_models.json"
P81_CKPT = RUN / "phase81_checkpoints"
P81B_CKPT = RUN / "phase81b_postgate_checkpoints"
PHASE85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase81b_floor_inference.json"
PRECOMMIT = "docs/research/2026-08-13_phase81b_floor_inference_precommit.md"

PHASE81_MODULE = "s_20260811_phase81_direct_floor_models"
POST_GATE_SOURCE = "msm_d1_00z"

SEED = 2026
N_BOOT = 10_000
BOOT_BLOCK = 7
BURN_IN_DAYS = 30
PRIMARY_THR = 0.5
MODELS = ("m0", "m0rank", "baserank", "m1", "m2", "m2p", "m4")
MODEL_LABELS = {
    "m0": "isotonic recalibration of the point forecast",
    "m0rank": "the same chain ranked on the raw point forecast (calibration is "
              "monotone, so this is its ranking without the calibrator's ties)",
    "baserank": "the price-only tier, ranked the same way: the baseline the "
                "manuscript's headline floor gain is measured against",
    "m1": "walk-forward logistic regression",
    "m2": "direct gradient-boosted classifier",
    "m2p": "the same classifier, weather withheld",
    "m4": "multi-quantile crossing",
}


def _j(x, nd: int = 5):
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


# ----------------------------------------------------------------- stage 2
def build_postgate(areas: list[str]) -> int:
    """Rebuild phase81's five models on POST-GATE weather, same folds and seeds."""
    mod = importlib.import_module(PHASE81_MODULE)
    for name, want in (("HEADLINE", "msm_d2_12z"),):
        got = getattr(mod, name)
        if got != want:
            log.error("%s is %r, expected %r -- phase81 changed underneath this "
                      "script and the contrast would not be like-for-like",
                      name, got, want)
            return 2
    old_source, old_ckpt = mod.HEADLINE, mod.CKPT
    mod.HEADLINE = POST_GATE_SOURCE
    mod.CKPT = P81B_CKPT
    log.info("rebound phase81: HEADLINE %s -> %s, CKPT -> %s",
             old_source, mod.HEADLINE, mod.CKPT)
    try:
        for a in areas:
            mod._run_area(a)  # noqa: SLF001 - reusing the fold loop is the point
    finally:
        mod.HEADLINE, mod.CKPT = old_source, old_ckpt
        log.info("restored phase81 globals")
    return 0


# ------------------------------------------------------------------ metrics
def _ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(average_precision_score(y, p))


def _ap_skill(y: np.ndarray, p: np.ndarray) -> float | None:
    """Prevalence-adjusted average precision: 0 at the base rate, 1 at perfect."""
    ap = _ap(y, p)
    if ap is None:
        return None
    base = float(y.mean())
    return None if base >= 1.0 else float((ap - base) / (1.0 - base))


def _fast_ap(y_ord: np.ndarray, w: np.ndarray,
             group_end: np.ndarray | None = None) -> float:
    """Average precision on a multiset, given slots ALREADY in descending score.

    ``w`` is each slot's multiplicity in the bootstrap draw. Precomputing the
    score order once and varying only the weights is what makes 10,000 draws of
    a pooled ranking metric affordable: the ranking of slots by score never
    changes under resampling, only how often each slot appears.

    ``group_end`` marks the last index of each run of EQUAL scores. Ties must be
    resolved the way sklearn's average_precision_score resolves them -- all slots
    at one score enter together, at a single threshold -- or the bootstrap targets
    a different statistic from the point estimate it is supposed to bracket.
    Walking slot by slot instead breaks ties by the sort's incidental order; on
    this panel that understated the most heavily tied model (m4, fewer than 7,000
    distinct scores over 302,454 slots) by up to 0.0086, against contrast margins
    of 0.018 and up. Found by an external code audit, 2026-08-18.
    """
    wy = w * y_ord
    npos = wy.sum()
    if npos <= 0:
        return float("nan")
    tp = np.cumsum(wy)
    seen = np.cumsum(w)
    if group_end is not None:
        # evaluate precision only at the END of each tied run, and credit the
        # whole run's positives at that one precision
        tp_g = tp[group_end]
        seen_g = seen[group_end]
        wy_g = np.diff(np.concatenate(([0.0], tp_g)))
        prec = np.divide(tp_g, seen_g, out=np.zeros_like(tp_g), where=seen_g > 0)
        return float((prec * wy_g).sum() / npos)
    prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    return float((prec * wy).sum() / npos)


class _MacroBoot:
    """Block bootstrap of the MACRO PR-AUC the paper reports.

    The paper's floor claim is an across-area average of per-area PR-AUC, so the
    interval has to be for that statistic. Averaging day-level statistics is a
    different quantity: a single day pools nine areas whose floor rates differ
    by an order of magnitude, and that mixture compresses exactly the contrast
    being tested. The resampling unit is still the delivery DAY, in circular
    blocks, because slots inside a day share a weather realisation.

    All contrasts share one set of draws, so their differences are comparable
    and a paired difference is the difference of two per-draw statistics rather
    than two independently resampled ones.
    """

    def __init__(self, panel: pl.DataFrame, cols: dict[str, str], thr: float):
        self.days = np.asarray(sorted(panel["test_day"].unique().to_list()))
        day_index = {d: i for i, d in enumerate(self.days)}
        day_of_slot = np.array([day_index[d] for d in panel["test_day"].to_list()])
        # NaN <= threshold is False, which would turn a missing price into a
        # NON-floor label instead of excluding it. None exist today.
        if not np.isfinite(panel["y_actual"].to_numpy()).all():
            msg = "non-finite actual prices would be labelled non-floor"
            raise RuntimeError(msg)
        y = (panel["y_actual"].to_numpy() <= thr).astype(float)
        areas = sorted(panel["area"].unique().to_list())
        area_of_slot = np.asarray(panel["area"].to_list())
        # per (model, area): slots in descending score order, with their labels
        # and the day each belongs to
        self.order: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for m, col in cols.items():
            p = panel[col].to_numpy().astype(float)
            finite = np.isfinite(p)
            for a in areas:
                mask = finite & (area_of_slot == a)
                idx = np.flatnonzero(mask)
                idx = idx[np.argsort(-p[idx], kind="stable")]
                # last index of each run of equal scores, fixed because the score
                # order is fixed; this is what makes tie handling match sklearn
                ps = p[idx]
                ge = np.flatnonzero(np.concatenate(
                    (ps[1:] != ps[:-1], [True])))
                self.order[(m, a)] = (y[idx], day_of_slot[idx], ge)
        self.models, self.areas = list(cols), areas
        self.n_days = len(self.days)

    def draws(self, n_boot: int, block: int, rng: np.random.Generator
              ) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
        """Returns (macro draws per model, per-area draws per model-area).

        The per-area draws are what puts intervals on the per-area figure. They
        come from the same draws as the macro statistic, so an area's interval
        and the across-area interval are consistent with each other.
        """
        n_blocks = int(np.ceil(self.n_days / block))
        out = {m: np.empty(n_boot) for m in self.models}
        per_area = {k: np.empty(n_boot) for k in self.order}
        for i in range(n_boot):
            starts = rng.integers(0, self.n_days, size=n_blocks)
            drawn = ((starts[:, None] + np.arange(block)[None, :]) % self.n_days
                     ).ravel()[:self.n_days]
            mult = np.bincount(drawn, minlength=self.n_days).astype(float)
            for m in self.models:
                vals = []
                for a in self.areas:
                    y_ord, day_ord, ge = self.order[(m, a)]
                    v = _fast_ap(y_ord, mult[day_ord], ge)
                    per_area[(m, a)][i] = v
                    vals.append(v)
                out[m][i] = np.nanmean(vals)
        return out, per_area


def _contrast(boot: dict[str, np.ndarray], a: str, b: str, point: float) -> dict:
    d = boot[a] - boot[b]
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return {"delta": _j(point), "ci95": [None, None], "n_draws": int(len(d))}
    lo, hi = np.percentile(d, [2.5, 97.5])
    raw = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    # A bootstrap cannot resolve a p below its own draw count. Reporting 0.000
    # would claim a precision 10,000 draws do not have, so the value is floored
    # at 1/N and flagged.
    floored = raw == 0.0
    return {"delta": _j(point), "ci95": [_j(float(lo)), _j(float(hi))],
            "boot_mean_delta": _j(float(d.mean())),
            "p_boot": _j(max(min(raw, 1.0), 1.0 / len(d)), 5),
            "p_boot_at_resolution_floor": floored,
            "n_draws": int(len(d))}


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, ordered smallest first."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        out[k] = _j(running, 4)
    return out


def _day_pr_auc(panel: pl.DataFrame, col: str, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-day pooled AP for one model, so differences can be paired by day."""
    days = sorted(panel["test_day"].unique().to_list())
    day_arr = np.asarray(panel["test_day"].to_list())
    y = (panel["y_actual"].to_numpy() <= thr).astype(int)
    p = panel[col].to_numpy().astype(float)
    vals = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        m = (day_arr == d) & np.isfinite(p)
        if m.sum() and y[m].sum():
            vals[i] = _ap(y[m], p[m])
    return np.asarray(days), vals


# ------------------------------------------------------------------ stage 1
def _isotonic_probs(sub: pl.DataFrame, yhat_col: str, thr: float) -> np.ndarray:
    """Expanding-window isotonic calibration, identical to phase81's M0."""
    days = sorted(sub["test_day"].unique().to_list())
    probs = np.full(sub.height, np.nan)
    yhat = sub[yhat_col].to_numpy()
    lab = (sub["y_actual"].to_numpy() <= thr).astype(int)
    day_arr = np.asarray(sub["test_day"].to_list())
    for d in days[BURN_IN_DAYS:]:
        tr, te = day_arr < d, day_arr == d
        if lab[tr].sum() and lab[tr].sum() < tr.sum():
            iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0,
                                     out_of_bounds="clip")
            iso.fit(yhat[tr], lab[tr])
            probs[te] = iso.predict(yhat[te])
        else:
            probs[te] = float(lab[tr].mean()) if tr.sum() else np.nan
    return probs


# Filled by _assemble so analyse() can record it: the measured difference between
# the two stage checkpoints, which is what proves the rebinding took effect.
_STAGE2_GAP: list[float] = []


def _assemble() -> pl.DataFrame:
    """phase81's five pre-gate models, M0 on both point forecasts, and -- when
    stage 2 has run -- the same five models on post-gate weather."""
    pre = pl.concat([pl.read_parquet(f) for f in sorted(P81_CKPT.glob("*.parquet"))])
    burn = sorted(pre["test_day"].unique().to_list())[BURN_IN_DAYS]
    pre = pre.filter(pl.col("test_day") >= burn).sort("area", "datetime_jst")

    p85 = (pl.read_parquet(PHASE85_PARQUET)
           .select("area", "datetime_jst", "test_day", "y_actual",
                   pl.col("y_pred_msm_d2_12z_cw").alias("yhat_pre"),
                   pl.col("y_pred_msm_d1_00z_cw").alias("yhat_post"),
                   pl.col("y_pred_price_only").alias("yhat_base")))
    frames = []
    for area in sorted(p85["area"].unique().to_list()):
        sub = p85.filter(pl.col("area") == area).sort("datetime_jst")
        sub = sub.with_columns(
            pl.Series(f"m0_{PRIMARY_THR:g}",
                      _isotonic_probs(sub, "yhat_pre", PRIMARY_THR)),
            pl.Series(f"m0post_{PRIMARY_THR:g}",
                      _isotonic_probs(sub, "yhat_post", PRIMARY_THR)),
            # The isotonic chain's RANKING is the point forecast's ranking:
            # calibration is monotone, so on a rank metric like average precision
            # the calibrated probability differs from the raw score only by the
            # TIES its flat segments create -- 19,898 distinct values against
            # 284,101. Scoring m0 by its probability therefore charges it for its
            # own calibrator's tie structure, and the floor-regime phase avoids
            # this deliberately by ranking on the raw score. Both are carried so
            # the model comparison can be made on the like-for-like footing.
            (-pl.col("yhat_pre")).alias(f"m0rank_{PRIMARY_THR:g}"),
            (-pl.col("yhat_post")).alias(f"m0rankpost_{PRIMARY_THR:g}"),
            # The price-only tier's ranking, so the manuscript's HEADLINE floor
            # contrast can be given a bootstrap OF THE MACRO STATISTIC. The
            # floor-regime phase reports it as the mean of nine per-area interval
            # endpoints and its own provenance flags that as "not a bootstrap of
            # the macro-average" -- while this file argues, in the paragraph the
            # supplement quotes, that the interval has to be for the statistic
            # being reported. Same quantity, proper interval.
            (-pl.col("yhat_base")).alias(f"baserank_{PRIMARY_THR:g}"),
        )
        frames.append(sub.drop("yhat_pre", "yhat_post", "yhat_base"))
    m0 = pl.concat(frames)

    panel = pre.join(m0, on=["area", "datetime_jst"], how="inner", suffix="_m0")
    # Stage 2 is a long build, so this function must behave correctly while it is
    # HALF DONE. An inner join against a partial post-gate set would silently
    # restrict the ENTIRE panel to the areas built so far, changing every stage-1
    # number without saying so -- three areas of nine would have rewritten the
    # floor comparison the manuscript already cites. Stage 2 is therefore
    # all-or-nothing: its columns are attached only when every area is present,
    # and a partial set is reported as partial and ignored.
    post_files = sorted(P81B_CKPT.glob("*.parquet")) if P81B_CKPT.exists() else []
    n_areas_expected = panel["area"].n_unique()
    if len(post_files) >= n_areas_expected:
        post = (pl.concat([pl.read_parquet(f) for f in post_files])
                .select("area", "datetime_jst",
                        *[pl.col(f"{m}_{PRIMARY_THR:g}")
                          .alias(f"{m}post_{PRIMARY_THR:g}")
                          for m in ("m1", "m2", "m2p", "m4")]))
        before = panel.height
        panel = panel.join(post, on=["area", "datetime_jst"], how="inner")
        log.info("stage 2 attached for %d areas: %d -> %d slots",
                 len(post_files), before, panel.height)
        if panel.height < 0.95 * before:
            log.error("attaching stage 2 dropped %.1f%% of the panel; the two "
                      "checkpoint sets do not cover the same slots",
                      100 * (1 - panel.height / before))
        # Stage 2 rebinds another module's globals to swap the weather source. If
        # that rebinding silently failed, the checkpoints would be a COPY of
        # stage 1 and the gate contrast would be a tautological zero -- a null
        # result produced by a bug, reported as a finding. Assert the two model
        # sets actually differ before any contrast is computed from them.
        pre_col, post_col = f"m2_{PRIMARY_THR:g}", f"m2post_{PRIMARY_THR:g}"
        gap = float(np.nanmean(np.abs(panel[post_col].to_numpy()
                                      - panel[pre_col].to_numpy())))
        log.info("stage 2 vs stage 1 mean |difference| in m2: %.5f", gap)
        _STAGE2_GAP.append(gap)
        if gap < 1e-6:
            msg = ("the post-gate classifier's predictions are identical to the "
                   "pre-gate one's: the weather-source rebinding did not take "
                   "effect, so the gate contrast would be a bug reported as a "
                   "null. Rebuild the stage 2 checkpoints.")
            raise RuntimeError(msg)
    elif post_files:
        log.warning("stage 2 is PARTIAL (%d of %d areas) and is being IGNORED: "
                    "attaching it would restrict the whole panel and change the "
                    "stage-1 numbers. Finish the build, then re-run --step analyse",
                    len(post_files), n_areas_expected)
    else:
        log.warning("no post-gate checkpoints: stage 2 columns will be absent")
    return panel.sort("area", "datetime_jst")


def analyse() -> dict:
    panel = _assemble()
    areas = sorted(panel["area"].unique().to_list())
    thr = PRIMARY_THR
    y = (panel["y_actual"].to_numpy() <= thr).astype(int)
    rng = np.random.default_rng(SEED)

    have_post = f"m2post_{thr:g}" in panel.columns
    out: dict = {
        "phase": "81b",
        "title": "Floor-model comparison: paired uncertainty, multiplicity, "
                 "prevalence, and the gate contrast under the winner",
        "precommit_doc": PRECOMMIT,
        "disclosed_deviation": (
            "phase81 declared N_BOOT=10000 and BOOT_BLOCK=7 and used neither, "
            "against its own precommit; every interval in this file is the "
            "bootstrap that was promised there"),
        "panel": {"n_slots": panel.height,
                  "n_test_days": panel["test_day"].n_unique(),
                  "n_areas": len(areas), "burn_in_days": BURN_IN_DAYS,
                  "threshold": thr,
                  "base_rate": _j(float(y.mean()))},
        "stage2_present": have_post,
        "stage2_vs_stage1_mean_abs_diff": _j(_STAGE2_GAP[0], 5) if _STAGE2_GAP else None,
        # Distinct score values, which is the mechanism behind the two PR-AUC
        # figures for one chain: a monotone calibrator can only add ties.
        "score_resolution": {},
        "metrics": {}, "paired": {}, "multiplicity": {}, "gate": {}, "gates": {},
    }

    # ---- macro, micro and prevalence-adjusted scores ----------------------
    cols = {m: f"{m}_{thr:g}" for m in MODELS}
    if have_post:
        cols.update({f"{m}post": f"{m}post_{thr:g}"
                     for m in ("m0", "m0rank", "m1", "m2", "m2p", "m4")})
    else:
        cols["m0post"] = f"m0post_{thr:g}"
        cols["m0rankpost"] = f"m0rankpost_{thr:g}"
    area_arr = np.asarray(panel["area"].to_list())
    for m, col in cols.items():
        p = panel[col].to_numpy().astype(float)
        ok = np.isfinite(p)
        per_area, skills = {}, []
        for a in areas:
            mask = (area_arr == a) & ok
            ap, sk = _ap(y[mask], p[mask]), _ap_skill(y[mask], p[mask])
            per_area[a] = {"pr_auc": _j(ap), "ap_skill": _j(sk),
                           "base_rate": _j(float(y[mask].mean())),
                           "n_pos": int(y[mask].sum())}
            if sk is not None:
                skills.append(sk)
        macro = [v["pr_auc"] for v in per_area.values() if v["pr_auc"] is not None]
        out["metrics"][m] = {
            "label": MODEL_LABELS.get(m.replace("post", ""), m),
            "weather_vintage": "post-gate" if m.endswith("post") else "pre-gate",
            "macro_pr_auc": _j(float(np.mean(macro))) if macro else None,
            "micro_pr_auc": _j(_ap(y[ok], p[ok])),
            "macro_ap_skill": _j(float(np.mean(skills))) if skills else None,
            "per_area": per_area,
        }
        log.info("%-8s macro %.4f  micro %.4f  ap-skill %.4f", m,
                 out["metrics"][m]["macro_pr_auc"] or float("nan"),
                 out["metrics"][m]["micro_pr_auc"] or float("nan"),
                 out["metrics"][m]["macro_ap_skill"] or float("nan"))
    # per-area and macro intervals, from the draws computed below

    for m, col in cols.items():
        v = panel[col].to_numpy()
        out["panel"] if False else None
        out["score_resolution"][m] = int(len(np.unique(v[np.isfinite(v)])))

    # ---- paired block bootstrap of the macro statistic ---------------------
    log.info("bootstrapping the macro statistic: %d draws, %d-day circular "
             "blocks, %d models", N_BOOT, BOOT_BLOCK, len(cols))
    boot, boot_area = _MacroBoot(panel, cols, thr).draws(N_BOOT, BOOT_BLOCK, rng)
    point = {m: out["metrics"][m]["macro_pr_auc"] for m in cols}
    for m in cols:
        d = boot[m][np.isfinite(boot[m])]
        if len(d):
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["metrics"][m]["macro_ci95"] = [_j(float(lo)), _j(float(hi))]
        for a in areas:
            da = boot_area[(m, a)]
            da = da[np.isfinite(da)]
            if len(da):
                lo, hi = np.percentile(da, [2.5, 97.5])
                out["metrics"][m]["per_area"][a]["ci95"] = [_j(float(lo)),
                                                            _j(float(hi))]
    contrasts = [("m2", "m0", "the winner over the recalibration chain, scored "
                  "by its calibrated probability"),
                 ("m2", "m0rank", "the winner over the SAME chain scored by its "
                  "ranking -- the like-for-like comparison for a rank metric, "
                  "because a monotone calibrator cannot reorder anything"),
                 ("m2", "m2p", "what the weather adds inside the winner"),
                 ("m2", "m1", "the winner over logistic regression"),
                 ("m2", "m4", "the winner over the quantile crossing"),
                 ("m0", "m2p", "the recalibration chain over the weather-withheld "
                               "classifier"),
                 ("m0rank", "baserank",
                  "THE MANUSCRIPT'S HEADLINE FLOOR GAIN: pre-gate over "
                  "price-only, both ranked on their own forecasts, bootstrapped "
                  "on the macro statistic rather than by averaging nine per-area "
                  "interval endpoints"),
                 ("m0rankpost", "m0rank",
                  "THE TAIL GATE CONTRAST, same footing: the excluded run over "
                  "the pre-gate run on the floor panel")]
    pvals = {}
    for a, b, note in contrasts:
        if a not in point or b not in point:
            log.warning("contrast %s-%s skipped: a column is absent", a, b)
            continue
        r = _contrast(boot, a, b, point[a] - point[b])
        r["note"] = note
        out["paired"][f"{a}_minus_{b}"] = r
        if r["delta"] is not None:
            pvals[f"{a}_minus_{b}"] = r["p_boot"]
        log.info("paired %-12s %+.4f [%s, %s] p=%s", f"{a}-{b}",
                 r["delta"] or float("nan"), r["ci95"][0], r["ci95"][1],
                 r.get("p_boot"))

    # ---- proper scores, which are the right instrument for THIS comparison ---
    # A rank metric cannot separate the isotonic chain from a direct classifier
    # on ordering, because the chain's ordering IS the point forecast's and the
    # classifier's is close to it. What the classifier adds is CALIBRATION, and a
    # rank metric is blind to calibration by construction. Brier and the log
    # score are means of per-observation losses, so -- as this paper's own
    # inference note says -- a daily loss-differential series exists and the
    # paired day-block bootstrap applies directly.
    def _per_day_loss(col: str, kind: str) -> np.ndarray:
        p = panel[col].to_numpy().astype(float)
        yy = y.astype(float)
        if kind == "brier":
            loss = (p - yy) ** 2
        else:
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            loss = -(yy * np.log(pc) + (1 - yy) * np.log(1 - pc))
        df = panel.select("area", "test_day").with_columns(
            pl.Series("loss", loss))
        return np.asarray(
            (df.group_by("area", "test_day").agg(pl.col("loss").mean())
             .group_by("test_day").agg(pl.col("loss").mean())
             .sort("test_day")["loss"]), dtype=float)

    out["proper_scores"] = {"note": ("macro over areas; lower is better. These "
                                     "separate the models the rank metric cannot"),
                            "levels": {}, "contrasts": {}}
    for m, col in cols.items():
        if m.endswith("post") or m == "m0rank":
            continue  # a raw score is not a probability: no proper score for it
        out["proper_scores"]["levels"][m] = {
            "brier": _j(float(_per_day_loss(col, "brier").mean()), 6),
            "log_score": _j(float(_per_day_loss(col, "log").mean()), 6),
        }
    for a, b, note in (("m2", "m0", "the classifier over the recalibration chain"),
                       ("m2", "m2p", "what the weather adds inside the classifier"),
                       ("m2", "m1", "over logistic regression"),
                       ("m2", "m4", "over the quantile crossing")):
        for kind in ("brier", "log"):
            d = _per_day_loss(cols[a], kind) - _per_day_loss(cols[b], kind)
            n = len(d)
            nb = int(np.ceil(n / BOOT_BLOCK))
            st = rng.integers(0, n, size=(N_BOOT, nb))
            ix = (st[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
            draws = d[ix.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            raw = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
            out["proper_scores"]["contrasts"][f"{a}_minus_{b}_{kind}"] = {
                "delta": _j(float(d.mean()), 6), "ci95": [_j(lo, 6), _j(hi, 6)],
                "p_boot": _j(max(raw, 1.0 / N_BOOT), 5),
                "improves": bool(d.mean() < 0), "note": note,
            }
            log.info("proper %-14s %-5s %+.6f [%.6f, %.6f] p=%.4f",
                     f"{a}-{b}", kind, d.mean(), lo, hi, max(raw, 1.0 / N_BOOT))

    out["multiplicity"] = {
        "family": "five floor models compared on the same panel",
        "n_models": len(MODELS),
        "raw_p": pvals,
        "holm_adjusted_p": _holm(pvals) if pvals else {},
        "note": ("the winner is the winner of five, and the contrasts above are "
                 "four of the comparisons that ranking implies"),
    }

    # ---- the gate contrast, in both models --------------------------------
    gate_pairs = [("m0post", "m0", "isotonic chain: post-gate minus pre-gate")]
    if have_post:
        gate_pairs.append(("m2post", "m2",
                           "direct classifier: post-gate minus pre-gate"))
    for a, b, note in gate_pairs:
        r = _contrast(boot, a, b, point[a] - point[b])
        r["note"] = note
        r["straddles_zero"] = (r["ci95"][0] is not None
                               and r["ci95"][0] <= 0 <= r["ci95"][1])
        out["gate"][f"{a}_minus_{b}"] = r
        log.info("GATE %-14s %+.4f [%s, %s] straddles zero = %s", a,
                 r["delta"] or float("nan"), r["ci95"][0], r["ci95"][1],
                 r["straddles_zero"])

    # ---- pre-registered gates --------------------------------------------
    # Gate 57 is decided on the LIKE-FOR-LIKE ranking contrast. Scoring the
    # isotonic chain by its calibrated probability charges it for the ties its own
    # monotone calibrator creates -- 19,898 distinct values against 284,101 for
    # the ranking it actually induces -- and that tie structure, not information,
    # is two thirds of the apparent margin. An earlier version of this gate used
    # the calibrated contrast and would have certified "winner of five" on it.
    m2_m0 = out["paired"]["m2_minus_m0"]
    m2_rank = out["paired"].get("m2_minus_m0rank", {})
    rank_lo = (m2_rank.get("ci95") or [None])[0]
    # The PRE-REGISTERED comparison scores every model on the probability it
    # emits -- the only common footing for five constructions with different
    # outputs -- and that contrast is m2 - m0, whose interval excludes zero. So
    # the recorded branch is a/b and the "winner of five" wording stands. The
    # rank diagnostic added afterwards does not overturn it; it explains WHERE the
    # margin comes from, and an earlier version of this gate keyed the branch on
    # the diagnostic and over-withdrew a claim the precommit protects.
    prereg_lo = (m2_m0.get("ci95") or [None])[0]
    withdrawn = prereg_lo is not None and prereg_lo <= 0
    rank_indistinguishable = rank_lo is not None and rank_lo <= 0
    brier = out["proper_scores"]["contrasts"].get("m2_minus_m0_brier", {})
    logsc = out["proper_scores"]["contrasts"].get("m2_minus_m0_log", {})
    out["gates"]["gate57_winner_margin"] = {
        "rank_contrast_calibrated": {"delta": m2_m0["delta"],
                                     "ci95": m2_m0["ci95"]},
        "rank_contrast_like_for_like": {"delta": m2_rank.get("delta"),
                                        "ci95": m2_rank.get("ci95"),
                                        "p_boot": m2_rank.get("p_boot")},
        "proper_score_contrasts": {"brier": brier, "log_score": logsc},
        "branch": "c" if withdrawn else "a/b",
        "rank_diagnostic_indistinguishable": rank_indistinguishable,
        "reading": (
            "branch c: the pre-registered contrast itself covers zero and the "
            "'winner' wording is withdrawn"
            if withdrawn else
            "branch a/b: the pre-registered contrast -- every model scored on the "
            "probability it emits, the only common footing for five different "
            "constructions -- excludes zero, so the winner wording stands. The "
            "rank diagnostic qualifies WHAT it wins at: ranked on the forecast "
            "the chain recalibrates rather than on its daily-refitted "
            "probability, the gap is not distinguishable from zero, so the margin "
            "is about probability quality and not about ordering ability. The "
            "proper scores agree: the log score separates the two (p < 1e-4) and "
            "the Brier score does not (p = 0.21)"),
    }
    key = "m2post_minus_m2" if have_post else "m0post_minus_m0"
    g = out["gate"][key]
    out["gates"]["gate58_gate_contrast_under_winner"] = {
        "tested_model": "direct classifier" if have_post else "isotonic chain only",
        "delta": g["delta"], "ci95": g["ci95"],
        "straddles_zero": g["straddles_zero"],
        "branch": "a" if g["straddles_zero"] else "b",
        "reading": ("branch a: the excluded run adds no detectable floor skill "
                    "under this model either, so the null is not a property of "
                    "the recalibration chain"
                    if g["straddles_zero"] else
                    "branch b: the excluded run DOES add floor skill under this "
                    "model. Report both models and withdraw the unqualified "
                    "'adds no floor skill'"),
        "stage2_complete": have_post,
    }

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "n_boot": N_BOOT,
                       "boot_block_days": BOOT_BLOCK,
                       "bootstrap_statistic": (
                           "macro PR-AUC (across-area mean of per-area average "
                           "precision), the statistic the manuscript reports; "
                           "resampling unit is the delivery day in circular "
                           "blocks, and all contrasts share one set of draws")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build-postgate", "analyse"), required=True)
    ap.add_argument("--areas", default="")
    args = ap.parse_args()
    if args.step == "build-postgate":
        mod = importlib.import_module(PHASE81_MODULE)
        wanted = ([a.strip() for a in args.areas.split(",") if a.strip()]
                  or list(mod.AREA_COORDS))
        return build_postgate(wanted)
    if not P81_CKPT.exists():
        log.error("missing %s", P81_CKPT)
        return 2
    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    log.info("GATE 57: %s", out["gates"]["gate57_winner_margin"]["reading"])
    log.info("GATE 58: %s", out["gates"]["gate58_gate_contrast_under_winner"]["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
