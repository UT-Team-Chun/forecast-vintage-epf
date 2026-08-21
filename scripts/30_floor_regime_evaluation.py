r"""Phase 66 v2: floor-regime evaluation on the CW-primary 732-day panel.

MECHANICAL FORK of s_20260730_phase66_floor_regime_evaluation.py (2026-08-11):
the only edits are the source parquet (phase59 -> phase85), fresh checkpoint /
output names, and nothing else, per the phase85 precommit's freeze policy.
Invoke with an explicit --tiers list (price_only + the three CW tiers + the
three SP JMA tiers + prev2) to hold runtime near the known 3.8 h.

ORIGINAL DOCSTRING:
Phase 66: floor-event probabilistic evaluation, per forecast-vintage tier.

WHY THIS EXISTS
---------------
The paper's conceptual framework says weather information "predicts a regime, not
only a level", but every headline comparison in it is a pooled MAE. A regime claim
evaluated only by a mean absolute error is not evaluated at all: the floor is a
3.4% event, and a metric dominated by the other 96.6% can move independently of it.

The fourth external review made the point in both directions, and the second
direction is the one that can cost us. Gate timing is a null in MAE
(+0.003 JPY/kWh, 95% CI [-0.004, 0.010]) -- but that is a statement about MAE, not
about the tail. If the first post-gate run detects floor events better, the
market-design conclusion has to be qualified. Gates 18-21 were fixed before this
script ran, including that branch, in
``docs/research/2026-07-29_phase66_floor_regime_precommit.md``.

WHAT IT DOES
------------
Reads the run-pinned ladder's per-slot panel and nothing else -- no refit of any
price model, no new data. For each tier it fits the SAME expanding-window isotonic
calibrator the decision model uses (``phase52::_floor_prob_and_c`` stage 1), so the
probabilities are leakage-free by construction, then scores:

  proper losses      Brier, log loss (differences only; the level is eps-dependent)
  ranking            PR-AUC, ROC-AUC, recall at fixed FPR
  calibration        10 equal-count bins with Wilson intervals, ECE, MCE, slope
  regime-conditional MAE on floor hours vs normal hours, asymmetric loss

Inference is split by what each metric admits, and the split is recorded in the
JSON rather than left implicit: Brier, log loss and the MAEs are means of
per-observation losses, so a daily loss-differential series exists and HLN-corrected
Diebold-Mariano applies. PR-AUC, ROC-AUC and recall-at-FPR are rank/threshold
functionals with no such series; they get the paired stationary bootstrap only, and
the bootstrap RECOMPUTES the functional on each resampled set of days rather than
averaging daily values.

CLI (from the repository root):
  uv run python s_20260730_phase66_floor_regime_evaluation.py
  ... --n-boot 2000 --areas kyushu,tokyo        # smoke test
  ... --step analyse                            # re-aggregate from checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase66")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
SRC = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase66v2_floor_regime.json"
CKPT = RUN / "phase66v2_checkpoints"
PRECOMMIT = "docs/research/2026-07-29_phase66_floor_regime_precommit.md"

# ----------------------------------------------------------------- pre-committed
THRESHOLDS: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0)
PRIMARY_THRESHOLD = 0.5
MIDDAY_HOURS = frozenset(range(9, 16))  # 09:00-15:59 JST, as phase42
BURN_IN_TEST_DAYS = 30
FPR_TARGETS: tuple[float, ...] = (0.005, 0.01, 0.05, 0.10)
ASYM_RATIOS: tuple[float, ...] = (0.25, 1.0, 2.0, 4.0, 8.0)
LOGLOSS_EPS = 1e-6
LOGLOSS_EPS_GRID: tuple[float, ...] = (1e-3, 1e-6, 1e-12)
CALIB_BINS = 10
MIN_POSITIVES = 100
N_BOOT_DEFAULT = 10_000
N_BOOT_FUNCTIONAL = 2_000  # rank functionals; see _pair_entry
BOOT_MEAN_BLOCK = 7.0
SEED = 2026
DM_LAGS: tuple[int, ...] = (3, 9)

# Metric -> which inference it admits. Recorded in the JSON: applying HLN-DM to a
# rank functional is a category error, so the reason is stated, not assumed.
DM_APPLICABLE = ("brier", "log_loss", "mae_floor_hours", "mae_normal_hours", "mae_all")
BOOTSTRAP_ONLY = (
    "pr_auc",
    "roc_auc",
    "recall_at_fpr",
    "precision",
    "recall",
    "f1",
    "calibration_slope",
    "ece",
    "mce",
)

HEADLINE = "msm_d2_12z"
BASELINE = "price_only"
POST_GATE = "msm_d1_00z"

# The seven pre-registered contrasts.
PAIRINGS: dict[str, tuple[str, str]] = {
    "P1_stale_vs_price": ("era5", BASELINE),
    "P2_pregate_vs_price": (HEADLINE, BASELINE),
    "P3_pregate_vs_stale": (HEADLINE, "era5"),
    "P4_gate_timing": (POST_GATE, HEADLINE),
    "P5_composite_vs_pinned": ("prev2", HEADLINE),
    "P6_recon_vs_pinned": ("histfc", HEADLINE),
    "P7_oracle_vs_pregate": ("oracle", HEADLINE),
    # ADDED 2026-08-13 (phase66v3, precommit
    # docs/research/2026-08-13_phase66v3_floor_cw_pairings_precommit.md).
    # HEADLINE above is the SINGLE-POINT tier, so P1-P7 are all single-point --
    # while the `levels` block scores the _cw tiers too. The manuscript ended up
    # printing capacity-weighted LEVELS beside a single-point GAIN, whose
    # difference does not reconcile (0.635 - 0.339 = 0.296, not the 0.276 the
    # pairing reports). Gate 35 promoted the capacity-weighted field to primary
    # and that promotion never reached these contrasts. These two recompute the
    # same contrasts on the primary measurement; nothing else changes.
    "P2cw_pregate_vs_price": (f"{HEADLINE}_cw", BASELINE),
    "P4cw_gate_timing": (f"{POST_GATE}_cw", f"{HEADLINE}_cw"),
}

AREA_ORDER = (
    "hokkaido",
    "tohoku",
    "tokyo",
    "chubu",
    "hokuriku",
    "kansai",
    "chugoku",
    "shikoku",
    "kyushu",
)



def _resolve_out(out_json: Path, n_boot: int, canonical: int, *, force: bool) -> Path:
    """Refuse to overwrite canonical provenance from a reduced-draw run.

    A `--n-boot 500` smoke run silently overwrote this phase's committed JSON twice
    during development, and the drift guard caught it only because a macro happened
    to be bound to an affected field. Below-canonical runs now write a *_draft.json
    unless --force-canonical is passed, so a debugging pass cannot become the
    published number.
    """
    if n_boot >= canonical or force:
        return out_json
    draft = out_json.with_name(out_json.stem + "_draft.json")
    log.warning("n_boot=%d is below the canonical %d: writing %s instead of %s "
                "(pass --force-canonical to override)",
                n_boot, canonical, draft.name, out_json.name)
    return draft

def _j(x: float | None, nd: int = 6) -> float | None:
    """JSON-safe round that preserves None rather than coercing it to 0."""
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CFG.project_root,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ------------------------------------------------------------------- calibration
def _expanding_pfloor(y_pred: np.ndarray, label: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    """P(floor) per slot from an expanding-window isotonic map, day by day.

    Identical in form to ``phase52::_floor_prob_and_c`` stage 1: decreasing
    isotonic from the tier's price forecast onto the binary floor label, fitted on
    strictly earlier test days only, with a base-rate fallback when the past has
    fewer than two distinct predictions or no label variation. Reusing that exact
    construction is deliberate -- the decision model's curtailment call is built on
    it, so the probabilities scored here are the ones the paper already relies on.
    """
    out = np.full(len(y_pred), np.nan)
    days = np.unique(day_index)
    for d in days:
        past = day_index < d
        cur = day_index == d
        yp_past, lab_past = y_pred[past], label[past]
        if len(np.unique(yp_past)) >= 2 and 0 < lab_past.sum() < len(lab_past):
            iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(yp_past, lab_past.astype(float))
            out[cur] = iso.predict(y_pred[cur])
        else:
            out[cur] = float(lab_past.mean()) if len(lab_past) else 0.0
    return out


# ----------------------------------------------------------------------- metrics
def _recall_at_fpr(
    label: np.ndarray, score: np.ndarray, targets: tuple[float, ...]
) -> dict[str, float | None]:
    """Recall at the largest threshold whose FPR does not exceed each target.

    The operationally readable metric: a trader who will tolerate one false alarm
    in a hundred non-floor slots wants to know what fraction of floor slots that
    buys. Reported as None when the negative class is empty.
    """
    neg = label == 0
    pos = label == 1
    if not neg.any() or not pos.any():
        return {f"{t:g}": None for t in targets}
    order = np.argsort(-score, kind="stable")
    lab_sorted = label[order]
    tp = np.cumsum(lab_sorted == 1)
    fp = np.cumsum(lab_sorted == 0)
    fpr = fp / neg.sum()
    tpr = tp / pos.sum()
    out: dict[str, float | None] = {}
    for t in targets:
        ok = np.nonzero(fpr <= t)[0]
        out[f"{t:g}"] = float(tpr[ok[-1]]) if len(ok) else 0.0
    return out


def _calibration(label: np.ndarray, p: np.ndarray, n_bins: int) -> dict:
    """Equal-COUNT reliability bins with Wilson intervals.

    Equal-width bins are useless here: at a 3.4% base rate almost every slot falls
    in [0, 0.1) and a fixed grid produces one populated bin.
    """
    n = len(p)
    order = np.argsort(p, kind="stable")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    bins, ece, mce = [], 0.0, 0.0
    for i in range(n_bins):
        sl = order[edges[i] : edges[i + 1]]
        if not len(sl):
            continue
        mp, of, m = float(p[sl].mean()), float(label[sl].mean()), len(sl)
        # Wilson score interval: exact-ish at tiny observed frequencies, where a
        # normal-approximation interval would extend below zero.
        z = 1.959964
        denom = 1.0 + z * z / m
        centre = (of + z * z / (2 * m)) / denom
        half = z * np.sqrt(of * (1 - of) / m + z * z / (4 * m * m)) / denom
        bins.append(
            {
                "mean_pred": _j(mp, 5),
                "obs_freq": _j(of, 5),
                "n": m,
                "wilson95": [_j(max(0.0, centre - half), 5), _j(min(1.0, centre + half), 5)],
            }
        )
        ece += (m / n) * abs(mp - of)
        mce = max(mce, abs(mp - of))
    # Calibration slope: regression of the label on the predicted probability.
    slope, intercept = (None, None)
    if float(np.var(p)) > 0:
        b, a = np.polyfit(p, label.astype(float), 1)
        slope, intercept = float(b), float(a)
    return {
        "n_bins": n_bins,
        "binning": "equal_count",
        "bins": bins,
        "ece": _j(ece, 5),
        "mce": _j(mce, 5),
        "slope": _j(slope, 4),
        "intercept": _j(intercept, 5),
    }


def _asym_loss(
    y: np.ndarray, yhat: np.ndarray, ratios: tuple[float, ...]
) -> dict[str, float | None]:
    """Mean asymmetric absolute loss: over-prediction penalised `r` times under."""
    err = yhat - y
    out = {}
    for r in ratios:
        w = np.where(err > 0, r, 1.0)
        out[f"{r:g}"] = _j(float(np.mean(w * np.abs(err))), 5)
    return out


def _metrics(
    label: np.ndarray, p: np.ndarray, y: np.ndarray, yhat: np.ndarray, threshold: float
) -> dict:
    """All per-cell metrics for one (scope, threshold, area, tier)."""
    n, n_pos = len(label), int(label.sum())
    base = n_pos / n if n else 0.0
    d: dict = {
        "n": n,
        "n_positives": n_pos,
        "base_rate": _j(base, 6),
        "underpowered": n_pos < MIN_POSITIVES,
    }
    if n_pos == 0 or n_pos == n:
        return d
    # Rank metrics use the RAW score (-y_pred), not the calibrated probability:
    # isotonic calibration is monotone, so it cannot change a ranking, and using
    # the raw score keeps PR-AUC independent of the calibrator's tie structure.
    score = -yhat
    d["pr_auc"] = _j(float(average_precision_score(label, score)), 5)
    d["roc_auc"] = _j(float(roc_auc_score(label, score)), 5)
    d["pr_auc_lift"] = _j(d["pr_auc"] / base, 3) if base > 0 else None
    d["recall_at_fpr"] = _recall_at_fpr(label, score, FPR_TARGETS)
    d["brier"] = _j(float(np.mean((p - label) ** 2)), 7)
    pc = np.clip(p, LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    d["log_loss"] = _j(float(-np.mean(label * np.log(pc) + (1 - label) * np.log(1 - pc))), 6)
    d["log_loss_eps_sensitivity"] = {
        f"{e:g}": _j(
            float(
                -np.mean(
                    label * np.log(np.clip(p, e, 1 - e))
                    + (1 - label) * np.log(1 - np.clip(p, e, 1 - e))
                )
            ),
            5,
        )
        for e in LOGLOSS_EPS_GRID
    }
    # Point predictions at the threshold itself (the naive decision rule).
    pred_floor = yhat <= threshold
    tp = int((pred_floor & (label == 1)).sum())
    fp = int((pred_floor & (label == 0)).sum())
    fn = int((~pred_floor & (label == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    d["precision"], d["recall"] = _j(prec, 5), _j(rec, 5)
    d["f1"] = _j(2 * prec * rec / (prec + rec), 5) if (prec + rec) else 0.0
    fl = label == 1
    d["mae_floor_hours"] = _j(float(np.mean(np.abs(y[fl] - yhat[fl]))), 5)
    d["mae_normal_hours"] = _j(float(np.mean(np.abs(y[~fl] - yhat[~fl]))), 5)
    d["mae_all"] = _j(float(np.mean(np.abs(y - yhat))), 5)
    d["asym_loss"] = _asym_loss(y, yhat, ASYM_RATIOS)
    d["calibration"] = _calibration(label, p, CALIB_BINS)
    return d


# --------------------------------------------------------------------- inference
def _stationary_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (phase61's construction)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / BOOT_MEAN_BLOCK
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.arange(n)
    restart_pos = np.where(restart, pos[None, :], -1)
    last = np.maximum.accumulate(restart_pos, axis=1)
    row = np.arange(n_boot)[:, None]
    return (starts[row, last] + (pos[None, :] - last)) % n


def _boot_ci_mean(d: np.ndarray, n_boot: int, seed: int = SEED) -> list[float | None]:
    if not len(d):
        return [None, None]
    idx = _stationary_idx(len(d), n_boot, seed)
    m = d[idx].mean(axis=1)
    return [_j(float(np.quantile(m, 0.025)), 6), _j(float(np.quantile(m, 0.975)), 6)]


def _boot_ci_functional(
    label: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    day_codes: np.ndarray,
    fn,
    n_boot: int,
    seed: int = SEED,
) -> tuple[list[float | None], float | None]:
    """Bootstrap CI for the PAIRED difference of a non-additive functional.

    Resamples whole DAYS and recomputes the functional on each resampled slot set.
    Averaging per-day PR-AUCs would be a different estimand and is badly behaved
    when a day contains no floor slot at all -- which most days do.
    """
    days = np.unique(day_codes)
    slots_by_day = [np.nonzero(day_codes == d)[0] for d in days]
    idx = _stationary_idx(len(days), n_boot, seed)
    diffs = np.empty(n_boot)
    diffs.fill(np.nan)
    for b in range(n_boot):
        sel = np.concatenate([slots_by_day[i] for i in idx[b]])
        lab = label[sel]
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        diffs[b] = fn(lab, score_a[sel]) - fn(lab, score_b[sel])
    ok = diffs[np.isfinite(diffs)]
    if not len(ok):
        return [None, None], None
    lo, hi = float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))
    # One-sided bootstrap p for "a is no better than b" under the paired draws.
    p_one = float(np.mean(ok <= 0.0))
    return [_j(lo, 6), _j(hi, 6)], _j(p_one, 6)


def _hln_dm(d: np.ndarray, lag: int) -> dict:
    """HLN-corrected Newey-West Diebold-Mariano on a daily loss differential."""
    n = len(d)
    if n < 3:
        return {"t_hln": None, "p_hln": None, "n_days": n, "nw_lag": lag}
    d_bar = float(d.mean())
    var_hac = float(np.var(d, ddof=0))
    for ell in range(1, lag + 1):
        if ell >= n:
            break
        gamma = float(np.mean((d[: n - ell] - d_bar) * (d[ell:] - d_bar)))
        var_hac += 2.0 * (1.0 - ell / (lag + 1)) * gamma
    se = float((max(var_hac, 1e-12) * ((n - 1) / n) / n) ** 0.5)
    t = d_bar / se if se > 0 else float("nan")
    p = 2.0 * (1.0 - float(stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else None
    return {
        "mean_diff": _j(d_bar, 7),
        "t_hln": _j(t, 4),
        "p_hln": _j(p, 8),
        "n_days": n,
        "nw_lag": lag,
    }


def _scope_masks(panel: pl.DataFrame, areas: list[str]) -> dict[str, np.ndarray]:
    """Pre-registered subsets. None of them may depend on a tier's own forecast.

    Conditioning a subset on a tier's predicted irradiance would make the subset
    tier-dependent and the comparison meaningless. The high/low-PV split uses
    TERTILES of pv_peak_ratio rather than a median cut, because hokkaido (0.499)
    and chubu (0.519) straddle 0.5 and a knife-edge split is a robustness
    liability.
    """
    hour = panel["datetime_jst"].dt.hour().to_numpy()
    area = panel["area"].to_numpy()
    ratio = {a: float(panel.filter(pl.col("area") == a)["pv_peak_ratio"][0]) for a in areas}
    ranked = sorted(ratio, key=lambda a: -ratio[a])
    k = max(1, len(ranked) // 3)
    high, low = set(ranked[:k]), set(ranked[-k:])
    return {
        "allhours": np.ones(panel.height, dtype=bool),
        "midday": np.isin(hour, list(MIDDAY_HOURS)),
        "high_pv_areas": np.isin(area, list(high)),
        "low_pv_areas": np.isin(area, list(low)),
    }


def calibrate(
    panel: pl.DataFrame, tiers: list[str], areas: list[str], thresholds: tuple[float, ...]
) -> pl.DataFrame:
    """Add pfloor_<tier>_<threshold> columns, fitted expanding-window per area."""
    out = panel
    for thr in thresholds:
        tag = f"{thr:g}".replace(".", "p")
        for tier in tiers:
            vals = np.full(panel.height, np.nan)
            for a in areas:
                m = (panel["area"] == a).to_numpy()
                sub = panel.filter(pl.col("area") == a)
                day_codes = sub["test_day"].rank("dense").cast(pl.Int32).to_numpy()
                y_raw = sub["y_actual"].to_numpy()
                # NaN <= thr is False, so a missing price would silently become a
                # NON-floor label and depress prevalence rather than be excluded.
                # There are none today; this refuses to let that change quietly.
                if not np.isfinite(y_raw).all():
                    msg = (f"{a}: {int((~np.isfinite(y_raw)).sum())} "
                           f"non-finite actual prices would be labelled "
                           f"non-floor; drop them explicitly instead")
                    raise RuntimeError(msg)
                label = (y_raw <= thr).astype(int)
                vals[m] = _expanding_pfloor(sub[f"y_pred_{tier}"].to_numpy(), label, day_codes)
            out = out.with_columns(pl.Series(f"pfloor_{tier}_{tag}", vals))
        log.info("calibrated threshold %g for %d tiers", thr, len(tiers))
    return out


def _daily_loss_diff(day_codes: np.ndarray, loss_a: np.ndarray, loss_b: np.ndarray) -> np.ndarray:
    """Per-day mean loss differential, for the DM test on additive losses."""
    days = np.unique(day_codes)
    out = np.empty(len(days))
    for i, d in enumerate(days):
        m = day_codes == d
        out[i] = float(loss_a[m].mean() - loss_b[m].mean())
    return out


def _pair_entry(
    panel_np: dict,
    sel: np.ndarray,
    thr: float,
    tier_a: str,
    tier_b: str,
    tag: str,
    n_boot: int,
    n_boot_fn: int | None = None,
) -> dict:
    """Paired comparison of two tiers on one (scope, threshold, area-set) cell.

    Two bootstrap sizes, for a computational reason that is disclosed rather than
    hidden. The additive-loss bootstrap is a vectorised mean over a resampled index
    matrix and costs nothing at 10,000 draws. The rank-functional bootstrap has to
    recompute PR-AUC on each resampled slot set -- 3 ms per draw on a 34,000-slot
    area -- so the full grid at 10,000 draws would take about six hours of the same
    arithmetic. It runs at ``n_boot_fn`` instead, and the two gate-critical
    contrasts are re-run at the full size afterwards to confirm the verdict does
    not depend on the choice (see ``gate_robustness`` in the output).
    """
    if n_boot_fn is None:
        n_boot_fn = n_boot
    label = (panel_np["y"][sel] <= thr).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return {"skipped": "no label variation in cell"}
    y = panel_np["y"][sel]
    ya, yb = panel_np[f"yp_{tier_a}"][sel], panel_np[f"yp_{tier_b}"][sel]
    pa = panel_np[f"pf_{tier_a}_{tag}"][sel]
    pb = panel_np[f"pf_{tier_b}_{tag}"][sel]
    days = panel_np["day"][sel]
    e: dict = {
        "tier_a": tier_a,
        "tier_b": tier_b,
        "n": len(label),
        "n_positives": int(label.sum()),
    }

    # ---- rank functionals: bootstrap only, recomputed on resampled day sets
    for name, fn in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)):
        va, vb = fn(label, -ya), fn(label, -yb)
        ci, p_one = _boot_ci_functional(label, -ya, -yb, days, fn, n_boot_fn)
        e[f"d_{name}"] = {
            "delta": _j(float(va - vb), 6),
            "ci95": ci,
            "p_boot_one_sided": p_one,
            "level_a": _j(float(va), 6),
            "level_b": _j(float(vb), 6),
        }
    ra = _recall_at_fpr(label, -ya, FPR_TARGETS)
    rb = _recall_at_fpr(label, -yb, FPR_TARGETS)
    e["d_recall_at_fpr"] = {
        k: {
            "delta": _j((ra[k] - rb[k]) if (ra[k] is not None and rb[k] is not None) else None, 5),
            "level_a": ra[k],
            "level_b": rb[k],
        }
        for k in ra
    }

    # ---- additive losses: bootstrap AND Diebold-Mariano
    def _add(name: str, la: np.ndarray, lb: np.ndarray) -> None:
        dd = _daily_loss_diff(days, la, lb)
        e[f"d_{name}"] = {
            "delta": _j(float(la.mean() - lb.mean()), 7),
            "ci95": _boot_ci_mean(dd, n_boot),
            **{f"dm_l{lag}": _hln_dm(dd, lag) for lag in DM_LAGS},
        }

    _add("brier", (pa - label) ** 2, (pb - label) ** 2)
    ca = np.clip(pa, LOGLOSS_EPS, 1 - LOGLOSS_EPS)
    cb = np.clip(pb, LOGLOSS_EPS, 1 - LOGLOSS_EPS)

    def _ll(c: np.ndarray) -> np.ndarray:
        return -(label * np.log(c) + (1 - label) * np.log(1 - c))

    _add("log_loss", _ll(ca), _ll(cb))
    fl = label == 1
    if fl.any():
        # Floor-hours MAE is additive but exists only on floor slots, so the daily
        # differential is over days that HAVE a floor slot -- state the count.
        dsel = days[fl]
        dd = _daily_loss_diff(dsel, np.abs(y[fl] - ya[fl]), np.abs(y[fl] - yb[fl]))
        e["d_mae_floor_hours"] = {
            "delta": _j(
                float(np.mean(np.abs(y[fl] - ya[fl])) - np.mean(np.abs(y[fl] - yb[fl]))), 6
            ),
            "ci95": _boot_ci_mean(dd, n_boot),
            "n_days_with_floor": len(np.unique(dsel)),
            **{f"dm_l{lag}": _hln_dm(dd, lag) for lag in DM_LAGS},
        }
    _add("mae_all", np.abs(y - ya), np.abs(y - yb))
    return e


def _macro_average(per_area: dict[str, dict], key_path: tuple[str, ...]) -> dict:
    """Macro-average a nested statistic across areas, skipping underpowered cells.

    Macro rather than slot-pooled: the slot-pooled PR curve is a base-rate mixture
    dominated by Kyushu and Shikoku (pooled AP 0.642 against 0.467 in Tokyo and
    0.827 in Kyushu), so pooling silently answers a different question. Both are
    reported; this one is primary.
    """
    vals, used, skipped = [], [], []
    for a, cell in per_area.items():
        if cell.get("skipped") is not None:
            skipped.append(a)
            continue
        node: object = cell
        for k in key_path:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, (int, float)) and node is not None:
            vals.append(float(node))
            used.append(a)
        else:
            skipped.append(a)
    if not vals:
        return {"value": None, "n_areas": 0, "areas_used": [], "areas_skipped": skipped}
    return {
        "value": _j(float(np.mean(vals)), 6),
        "n_areas": len(vals),
        "areas_used": used,
        "areas_skipped": skipped,
    }


# Each gate names its PRIMARY input first and its sensitivity arm second. The
# paper's primary floor measurement is the capacity-weighted one, so the gates
# must be judged on that; the single-point pairings are the demoted arm and are
# kept as a fallback so a single-point-only run still evaluates something, with
# the arm it used recorded in the JSON.
GATE_INPUTS: dict[str, tuple[str, str]] = {
    "gate18_regime_claim": ("P2cw_pregate_vs_price", "P2_pregate_vs_price"),
    "gate19_gate_timing_in_tail": ("P4cw_gate_timing", "P4_gate_timing"),
    "gate20_threshold_stability": ("P2cw_pregate_vs_price", "P2_pregate_vs_price"),
    "gate21_midday": ("P2cw_pregate_vs_price", "P2_pregate_vs_price"),
}

# The verdict for a gate whose input this run did not compute. It is NOT "fails":
# every gate here carries an if_fails or a branch reading that prescribes surgery
# on the manuscript -- gate 18's says to withdraw the regime framing from
# Section 3 -- and a run that simply did not compute a pairing must never emit
# that instruction. Before 2026-08-14 it did: the capacity-weighted full-grid run
# left all four gates reading passes=false with delta=null, because this function
# looked only at the single-point pairing names.
NOT_EVALUATED = "not evaluated in this run: its input pairing was not computed"


def _evaluate_gates(res: dict) -> dict:
    """Gates 18-21, judged mechanically against the pre-committed rules.

    A gate is evaluated on the capacity-weighted arm when this run computed it,
    on the single-point arm otherwise, and NOT AT ALL when neither is present --
    which is recorded as evaluated=false rather than as a failure.
    """
    tag = f"{PRIMARY_THRESHOLD:g}".replace(".", "p")

    def pooled_on(pairing: str, scope: str, metric: str) -> dict | None:
        node = (
            res.get("pairings", {})
            .get(pairing, {})
            .get(scope, {})
            .get(tag, {})
            .get("pooled_macro_avg", {})
        )
        return node.get(metric)

    def resolve(gate: str, scope: str, metric: str) -> tuple[dict | None, str | None]:
        """Return the gate's input and the arm it came from, primary first."""
        for arm_name, pairing in zip(("capacity_weighted", "single_point"),
                                     GATE_INPUTS[gate], strict=True):
            node = pooled_on(pairing, scope, metric)
            if node is not None and node.get("delta") is not None:
                return node, arm_name
        return None, None

    g: dict = {}
    p2, arm2 = resolve("gate18_regime_claim", "allhours", "d_pr_auc")
    lo = p2.get("ci95_lo") if p2 else None
    g["gate18_regime_claim"] = {
        "rule": "pre-gate vs price-only d_PR-AUC, macro-average, threshold 0.5, "
        "all-hours: bootstrap 95% CI lower bound > 0",
        "arm": arm2,
        "evaluated": p2 is not None,
        "delta": p2.get("delta") if p2 else None,
        "ci95": [lo, p2.get("ci95_hi") if p2 else None],
        "passes": (bool(lo is not None and lo > 0) if p2 is not None else None),
        "if_fails": ("withdraw the 'predicts a regime, not only a level' framing "
                     "from Section 3 and report that floor-event detection does "
                     "not improve on this sample")
        if p2 is not None else NOT_EVALUATED,
    }
    p4, arm4 = resolve("gate19_gate_timing_in_tail", "allhours", "d_pr_auc")
    lo4 = p4.get("ci95_lo") if p4 else None
    hi4 = p4.get("ci95_hi") if p4 else None
    includes_zero = bool(lo4 is not None and hi4 is not None and lo4 <= 0 <= hi4)
    # The branch reading is chosen ONLY when the gate was evaluated. Before this
    # was fixed, an unevaluated gate printed 19b -- the branch that tells the
    # author to qualify the gate-closure conclusion in the abstract -- because
    # includes_zero is False for None inputs. The alarming branch was the default.
    g["gate19_gate_timing_in_tail"] = {
        "rule": "post-gate vs pre-gate d_PR-AUC, macro-average, threshold 0.5: "
        "bootstrap 95% CI includes 0",
        "arm": arm4,
        "evaluated": p4 is not None,
        "delta": p4.get("delta") if p4 else None,
        "ci95": [lo4, hi4],
        "passes": includes_zero if p4 is not None else None,
        "reading": (
            (
                "19a: the gate-timing null extends beyond MAE to regime detection"
                if includes_zero
                else "19b: the null is MAE-SPECIFIC -- the excluded run detects "
                "floor events better. Qualify the gate-closure conclusion in "
                "the abstract and Discussion; macro the effect and report it "
                "rather than confining it to the supplement"
            )
            if p4 is not None
            else NOT_EVALUATED
        ),
    }
    signs, arm20 = {}, None
    for pairing, arm_name in zip(GATE_INPUTS["gate20_threshold_stability"],
                                 ("capacity_weighted", "single_point"), strict=True):
        cand = {}
        for thr in THRESHOLDS:
            t = f"{thr:g}".replace(".", "p")
            node = (
                res.get("pairings", {})
                .get(pairing, {})
                .get("allhours", {})
                .get(t, {})
                .get("pooled_macro_avg", {})
                .get("d_pr_auc")
            )
            cand[f"{thr:g}"] = (
                None
                if not node or node.get("delta") is None
                else (1 if node["delta"] > 0 else (-1 if node["delta"] < 0 else 0))
            )
        if any(v is not None for v in cand.values()):
            signs, arm20 = cand, arm_name
            break
    present = [v for v in signs.values() if v is not None]
    missing = [k for k, v in signs.items() if v is None] or [
        f"{t:g}" for t in THRESHOLDS] if len(present) < len(THRESHOLDS) else []
    # "All four thresholds agree" is only ANSWERABLE when all four exist. A run
    # that computed one threshold used to pass this stability gate on a single
    # sign; making that a failure instead would repeat the very defect fixed
    # above, so incomplete coverage is UNEVALUATED. The all-hours-only cw run
    # carries one threshold and was reporting a stability verdict from it.
    complete = len(present) == len(THRESHOLDS)
    g["gate20_threshold_stability"] = {
        "rule": "sign of the Gate-18 contrast identical across all four thresholds",
        "arm": arm20 if complete else None,
        "evaluated": complete,
        "signs": signs,
        "passes": (len(set(present)) == 1) if complete else None,
        "note": None if complete else (
            f"{NOT_EVALUATED}; thresholds computed: "
            f"{sorted(k for k, v in signs.items() if v is not None) or 'none'}"),
    }
    pm, arm21 = resolve("gate21_midday", "midday", "d_pr_auc")
    lom = pm.get("ci95_lo") if pm else None
    g["gate21_midday"] = {
        "rule": "Gate-18 contrast on the midday subset: CI lower bound > 0",
        "arm": arm21,
        "evaluated": pm is not None,
        "delta": pm.get("delta") if pm else None,
        "ci95": [lom, pm.get("ci95_hi") if pm else None],
        "passes": (bool(lom is not None and lom > 0)) if pm is not None else None,
        "note": None if pm is not None else NOT_EVALUATED,
    }
    unevaluated = [k for k, v in g.items() if not v.get("evaluated")]
    if unevaluated:
        log.warning(
            "%d of %d gates were NOT evaluated (input pairing absent from this "
            "run): %s. They are recorded as evaluated=false with passes=null, "
            "NOT as failures -- do not read a branch prescription from them.",
            len(unevaluated), len(g), ", ".join(unevaluated))
    return g


def _flatten_ci(entry: dict) -> dict:
    """Copy an entry with ci95 split into ci95_lo/ci95_hi for easy gate access."""
    out = dict(entry)
    ci = entry.get("ci95")
    if isinstance(ci, list) and len(ci) == 2:
        out["ci95_lo"], out["ci95_hi"] = ci[0], ci[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-canonical", action="store_true",
                    help="write the canonical output even below the canonical "
                         "draw count")
    ap.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT_DEFAULT,
        help="draws for the vectorised additive-loss bootstrap",
    )
    ap.add_argument(
        "--n-boot-functional",
        type=int,
        default=N_BOOT_FUNCTIONAL,
        help="draws for rank functionals, which must be recomputed "
        "per draw; the gate contrasts are re-checked at --n-boot",
    )
    ap.add_argument("--areas", default=",".join(AREA_ORDER))
    ap.add_argument("--tiers", default="")
    ap.add_argument("--thresholds", default=",".join(f"{t:g}" for t in THRESHOLDS))
    ap.add_argument("--step",
                    choices=["all", "calibrate", "analyse", "regates"],
                    default="all",
                    help="regates: re-run ONLY the gate verdicts against the "
                         "pairings already in the output JSON. The gates are a "
                         "pure function of those pairings, so fixing the gate "
                         "logic does not require recomputing the bootstraps.")
    # A top-up run scores tiers the first pass left out (it was restricted to
    # keep the bootstrap inside a night). Two things have to be separable for
    # that to be safe: WHICH contrasts are recomputed -- P2 and P4 carry the
    # full scope x threshold grid and cost hours, so a run that only needs a
    # level must be able to skip them -- and WHERE the run writes, because the
    # calibrated checkpoint is specific to the tier set that produced it and
    # silently reusing another run's would score the wrong columns.
    ap.add_argument("--pairings", default="",
                    help="comma-separated subset of PAIRINGS to compute "
                         "(default: all whose tiers are present)")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint dir and the output JSON, "
                         "so a top-up run cannot overwrite the primary one")
    args = ap.parse_args()

    if args.step == "regates":
        # The gates are a pure function of the pairings already written. This
        # exists because the gate logic was WRONG in a way that only a
        # capacity-weighted run exposed -- it read the single-point pairing
        # names, so the full-grid run left all four gates reading passes=false
        # with delta=null, and gate 18's if_fails text says to withdraw the
        # regime framing from Section 3. Recomputing 2.5 h of bootstraps to fix
        # a verdict function would be the wrong trade; re-deriving the verdicts
        # from the committed pairings is exact.
        target = (OUT_JSON if not args.tag
                  else OUT_JSON.with_name(f"{OUT_JSON.stem}_{args.tag}.json"))
        if not target.exists():
            log.error("no output to re-gate at %s", target)
            return 2
        payload = json.loads(target.read_text())
        before = {k: v.get("passes") for k, v in payload.get("gates", {}).items()}
        payload["gates"] = _evaluate_gates(payload)
        payload.setdefault("metadata", {})["gates_reevaluated"] = {
            "why": "the gate evaluator read only the single-point pairing names, "
                   "so a capacity-weighted run left every gate unevaluated and "
                   "printed passes=false; unevaluated is now recorded as such",
            "verdicts_before": before,
            "verdicts_after": {k: v.get("passes")
                               for k, v in payload["gates"].items()},
        }
        target.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                          encoding="utf-8")
        for k, v in payload["gates"].items():
            log.info("  %-32s arm=%-17s evaluated=%s passes=%s delta=%s",
                     k, v.get("arm"), v.get("evaluated"), v.get("passes"),
                     v.get("delta"))
        log.info("re-gated %s", target)
        return 0
    t0 = time.time()

    if not SRC.exists():
        log.error("missing %s -- run phase59 first", SRC)
        return 2
    areas = [a for a in args.areas.split(",") if a]
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x)
    panel = pl.read_parquet(SRC).filter(pl.col("area").is_in(areas)).sort(["area", "datetime_jst"])
    tiers = (
        args.tiers.split(",")
        if args.tiers
        else [c.removeprefix("y_pred_") for c in panel.columns if c.startswith("y_pred_")]
    )
    # Burn-in: the calibrator has no past on day 1, so its first days are a
    # base-rate constant. Excluding them is the same convention phase52 uses.
    keep_days = sorted(panel["test_day"].unique())[BURN_IN_TEST_DAYS:]
    log.info(
        "panel %d slots, %d tiers, %d areas; scoring %d of %d test days",
        panel.height,
        len(tiers),
        len(areas),
        len(keep_days),
        panel["test_day"].n_unique(),
    )

    ckpt_dir = CKPT if not args.tag else CKPT.with_name(f"{CKPT.name}_{args.tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "calibrated.parquet"
    if args.step == "analyse" and ckpt.exists():
        cal = pl.read_parquet(ckpt)
        log.info("loaded calibrated panel from checkpoint (%d rows)", cal.height)
    else:
        cal = calibrate(panel, tiers, areas, thresholds)
        cal.write_parquet(ckpt)
        log.info("wrote %s", ckpt)
        if args.step == "calibrate":
            return 0

    cal = cal.filter(pl.col("test_day").is_in(keep_days))
    masks = _scope_masks(cal, areas)
    np_panel: dict = {
        "y": cal["y_actual"].to_numpy(),
        "day": cal["test_day"].rank("dense").cast(pl.Int32).to_numpy(),
        "area": cal["area"].to_numpy(),
    }
    for t in tiers:
        np_panel[f"yp_{t}"] = cal[f"y_pred_{t}"].to_numpy()
        for thr in thresholds:
            tg = f"{thr:g}".replace(".", "p")
            np_panel[f"pf_{t}_{tg}"] = cal[f"pfloor_{t}_{tg}"].to_numpy()

    res: dict = {
        "phase": 66,
        "title": "Floor-event probabilistic evaluation per weather-vintage tier",
        "precommit_doc": PRECOMMIT,
        "source_parquet": SRC.name,
        "protocol": {
            "floor_thresholds": list(thresholds),
            "primary_threshold": PRIMARY_THRESHOLD,
            "scopes": list(masks),
            "midday_hours_jst": sorted(MIDDAY_HOURS),
            "calibrator": "expanding-window isotonic (increasing=False, y_min=0, "
            "y_max=1, clip); phase52::_floor_prob_and_c stage 1",
            "burn_in_test_days": BURN_IN_TEST_DAYS,
            "logloss_eps": LOGLOSS_EPS,
            "fpr_targets": list(FPR_TARGETS),
            "n_boot": args.n_boot,
            "n_boot_functional": args.n_boot_functional,
            "boot_mean_block": BOOT_MEAN_BLOCK,
            "seed": SEED,
            "dm_nw_lags": list(DM_LAGS),
            "min_positives": MIN_POSITIVES,
            "rank_metrics_use_raw_score": True,
        },
        "inference_applicability": {
            "dm_applicable": list(DM_APPLICABLE),
            "bootstrap_only": list(BOOTSTRAP_ONLY),
            "reason": "Brier, log loss and the MAEs are means of per-observation "
            "losses, so a daily loss-differential series exists and "
            "HLN-DM applies. PR-AUC, ROC-AUC and recall-at-FPR are "
            "rank/threshold functionals with no such series; their "
            "bootstrap recomputes the functional on each resampled set "
            "of days rather than averaging daily values.",
            "log_loss_caveat": "the LEVEL is eps-dependent because the isotonic map "
            "assigns exactly 0 to some slots; only DIFFERENCES "
            "are reported as findings, with an eps grid shown.",
        },
        "coverage": {
            "n_slots": int(cal.height),
            "n_test_days": len(keep_days),
            "n_areas": len(areas),
        },
        "levels": {},
        "pairings": {},
    }

    # ---------------------------------------------------------------- levels
    for scope, smask in masks.items():
        res["levels"][scope] = {}
        for thr in thresholds:
            tg = f"{thr:g}".replace(".", "p")
            res["levels"][scope][f"{thr:g}"] = {}
            for a in [*areas, "__pooled__"]:
                amask = smask if a == "__pooled__" else (smask & (np_panel["area"] == a))
                if not amask.any():
                    continue
                cells = {}
                for t in tiers:
                    label = (np_panel["y"][amask] <= thr).astype(int)
                    cells[t] = _metrics(
                        label,
                        np_panel[f"pf_{t}_{tg}"][amask],
                        np_panel["y"][amask],
                        np_panel[f"yp_{t}"][amask],
                        thr,
                    )
                key = "pooled_slotwise" if a == "__pooled__" else a
                res["levels"][scope][f"{thr:g}"][key] = cells
            # macro-average of the per-area levels
            per_area = {
                a: res["levels"][scope][f"{thr:g}"][a]
                for a in areas
                if a in res["levels"][scope][f"{thr:g}"]
            }
            macro = {}
            for t in tiers:
                sub = {a: c[t] for a, c in per_area.items() if not c[t].get("underpowered", True)}
                macro[t] = {
                    m: _macro_average(sub, (m,))
                    for m in (
                        "pr_auc",
                        "roc_auc",
                        "brier",
                        "log_loss",
                        "pr_auc_lift",
                        "mae_floor_hours",
                    )
                }
            res["levels"][scope][f"{thr:g}"]["pooled_macro_avg"] = macro
        log.info("levels done for scope %s", scope)

    # -------------------------------------------------------------- pairings
    # Grid restriction, stated so nobody has to infer it. The four gates need the
    # full scope x threshold grid only for P2 (the regime claim and its threshold
    # and midday sensitivities) and P4 (gate timing). Every other contrast is a
    # headline-table row, so it is computed at the primary threshold, all hours.
    # Computing 112 cell-groups where 39 are used would add hours of the same
    # bootstrap for numbers nothing reads.
    # The CW variants carry the paper's primary claims now, so they need the
    # full scope x threshold grid too: the manuscript cites the midday subset
    # and the threshold sweep, and those must come from the same measurement
    # as the headline rather than from the single-point arm.
    FULL_GRID = ("P2_pregate_vs_price", "P4_gate_timing",
                 "P2cw_pregate_vs_price", "P4cw_gate_timing")
    wanted = {p for p in args.pairings.split(",") if p} or set(PAIRINGS)
    unknown = wanted - set(PAIRINGS)
    if unknown:
        msg = f"unknown pairing(s): {sorted(unknown)}"
        raise SystemExit(msg)
    for name, (ta, tb) in PAIRINGS.items():
        if name not in wanted:
            continue
        if ta not in tiers or tb not in tiers:
            continue
        res["pairings"][name] = {}
        full = name in FULL_GRID
        scopes_here = masks if full else {"allhours": masks["allhours"]}
        thrs_here = thresholds if full else (PRIMARY_THRESHOLD,)
        for scope, smask in scopes_here.items():
            res["pairings"][name][scope] = {}
            for thr in thrs_here:
                if thr not in thresholds:
                    continue
                tg = f"{thr:g}".replace(".", "p")
                cell: dict = {}
                for a in areas:
                    amask = smask & (np_panel["area"] == a)
                    if amask.sum() < 2:
                        continue
                    cell[a] = _pair_entry(
                        np_panel, amask, thr, ta, tb, tg, args.n_boot, args.n_boot_functional
                    )
                if smask.any():
                    cell["pooled_slotwise"] = _pair_entry(
                        np_panel, smask, thr, ta, tb, tg, args.n_boot, args.n_boot_functional
                    )
                # macro-average the deltas AND their CI endpoints across areas
                ok = {
                    a: c
                    for a, c in cell.items()
                    if a in areas
                    and c.get("skipped") is None
                    and c.get("n_positives", 0) >= MIN_POSITIVES
                }
                macro = {}
                for metric in ("d_pr_auc", "d_roc_auc", "d_brier", "d_log_loss", "d_mae_all"):
                    sub = {a: c[metric] for a, c in ok.items() if metric in c}
                    if not sub:
                        continue
                    # Averaging the per-area interval ENDPOINTS is a conservative
                    # summary, not a bootstrap of the macro-average itself: it
                    # ignores cross-area dependence in both directions. It is
                    # reported alongside areas_positive so a reader can see how
                    # many areas carry the sign, which is the robust statement.
                    los = [
                        c["ci95"][0]
                        for c in sub.values()
                        if isinstance(c.get("ci95"), list) and c["ci95"][0] is not None
                    ]
                    his = [
                        c["ci95"][1]
                        for c in sub.values()
                        if isinstance(c.get("ci95"), list) and c["ci95"][1] is not None
                    ]
                    macro[metric] = _flatten_ci(
                        {
                            "delta": _macro_average(sub, ("delta",))["value"],
                            "ci95": [
                                _j(float(np.mean(los)), 6) if los else None,
                                _j(float(np.mean(his)), 6) if his else None,
                            ],
                            "ci95_note": "mean of the per-area interval endpoints; "
                            "not a bootstrap of the macro-average",
                            "n_areas": len(sub),
                            "areas_positive": sum(
                                1
                                for c in sub.values()
                                if c.get("delta") is not None and c["delta"] > 0
                            ),
                        }
                    )
                cell["pooled_macro_avg"] = macro
                res["pairings"][name][scope][tg] = cell
        log.info("pairing %s done", name)

    res["gates"] = _evaluate_gates(res)

    # --------------------------------------------------- gate robustness re-check
    # The two gate-critical contrasts, recomputed with the rank-functional
    # bootstrap at the FULL draw count. If a gate verdict flipped between 2,000
    # and 10,000 draws, the verdict was Monte-Carlo noise and must not be
    # reported; this block proves it either way instead of asserting it.
    if args.n_boot_functional < args.n_boot:
        tg = f"{PRIMARY_THRESHOLD:g}".replace(".", "p")
        recheck: dict = {
            "n_boot_functional_recheck": args.n_boot,
            "why": "confirm the Gate 18/19 verdicts do not depend on "
            "the reduced rank-functional bootstrap",
        }
        # Resolve through GATE_INPUTS, exactly as _evaluate_gates does. This
        # block used to hard-code the single-point pairings, so the 10,000-draw
        # recheck confirmed the verdicts of the DEMOTED arm while the gates
        # themselves were judged on the capacity-weighted one -- a recheck of a
        # different quantity than the thing being rechecked.
        for gate in ("gate18_regime_claim", "gate19_gate_timing_in_tail"):
            pairing = next(
                (cand for cand in GATE_INPUTS[gate]
                 if cand in PAIRINGS
                 and all(t in tiers for t in PAIRINGS[cand])),
                None)
            if pairing is None:
                recheck[gate] = {"skipped": "neither arm's tiers are present"}
                continue
            ta, tb = PAIRINGS[pairing]
            recheck.setdefault("arm_used", {})[gate] = pairing
            los, his, deltas = [], [], []
            for a in areas:
                amask = masks["allhours"] & (np_panel["area"] == a)
                lab = (np_panel["y"][amask] <= PRIMARY_THRESHOLD).astype(int)
                if lab.sum() < MIN_POSITIVES:
                    continue
                ci, _ = _boot_ci_functional(
                    lab,
                    -np_panel[f"yp_{ta}"][amask],
                    -np_panel[f"yp_{tb}"][amask],
                    np_panel["day"][amask],
                    average_precision_score,
                    args.n_boot,
                )
                if ci[0] is not None:
                    los.append(ci[0])
                    his.append(ci[1])
                deltas.append(
                    float(
                        average_precision_score(lab, -np_panel[f"yp_{ta}"][amask])
                        - average_precision_score(lab, -np_panel[f"yp_{tb}"][amask])
                    )
                )
            lo = _j(float(np.mean(los)), 6) if los else None
            hi = _j(float(np.mean(his)), 6) if his else None
            orig = res["gates"][gate]
            if gate == "gate18_regime_claim":
                passes = bool(lo is not None and lo > 0)
            else:
                passes = bool(lo is not None and hi is not None and lo <= 0 <= hi)
            recheck[gate] = {
                "delta": _j(float(np.mean(deltas)), 6) if deltas else None,
                "ci95": [lo, hi],
                "passes": passes,
                "verdict_unchanged": passes == orig.get("passes"),
                "n_areas": len(los),
            }
            log.info(
                "  recheck %s at %d draws: passes=%s (unchanged=%s)",
                gate,
                args.n_boot,
                passes,
                passes == orig.get("passes"),
            )
        res["gate_robustness"] = recheck

    res["metadata"] = {
        "script": Path(__file__).name,
        "git_sha": _git_sha(),
        "runtime_s": round(time.time() - t0, 1),
    }
    out_json = (OUT_JSON if not args.tag
                else OUT_JSON.with_name(f"{OUT_JSON.stem}_{args.tag}.json"))
    res["run_scope"] = {"tiers": list(tiers), "pairings": sorted(wanted),
                        "tag": args.tag or None}
    _out = _resolve_out(out_json, args.n_boot, N_BOOT_DEFAULT,
                        force=args.force_canonical)
    _out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log.info("wrote %s (runtime %.1f s)", _out, time.time() - t0)
    for k, v in res["gates"].items():
        log.info("  %s: passes=%s", k, v.get("passes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
