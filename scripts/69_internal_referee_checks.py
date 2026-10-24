r"""Phase 118: three checks an internal referee read asked for, on stored predictions.

WHY THIS EXISTS
---------------
An independent referee read of the Energy manuscript (2026-10-23) raised three
points that need no new model, only different cuts of frozen predictions:
  W1  the price bound is pooled over 48 slots, so night slots dilute it; what is
      the post-gate increment on the daylight slots where PV acts?
  W2  the physical-target contrast set the post-gate run against the D-2 12 UTC
      run, crediting it with gains the freshest PRE-gate run already delivers, and
      compared a daylight physical figure with an all-slot price figure. Split
      freshness before the gate from freshness across it, on one window.
  W5  the floor result is never put in decision terms in the main text, and the
      supplement's cost-loss curve does not score the post-gate run.
Not part of the phase112 pre-registration. Precommit (rules fixed before this
script first ran): docs/research/2026-10-23_phase118_internal_referee_precommit.md

Every statistic is imported from the phase that produced the published number
(phase85's estimator, phase89's physical construction, phase81b's isotonic
probabilities), so a number here differs from its published counterpart only by
the cut. Gate 118-0 checks that the imports reproduce the published levels first.

CLI (from the repository root):
  uv run python scripts/69_internal_referee_checks.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase118")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
HERE = Path(__file__).parent
P85_SCRIPT = HERE / "10_primary_ladder_cw.py"
P89_SCRIPT = HERE / "50_physical_target.py"
P81B_SCRIPT = HERE / "65_floor_inference.py"
P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
P89_JSON = RUN / "phase89_physical_target.json"
P102_JSON = RUN / "phase102_signal_vs_learner.json"
OUT_JSON = RUN / "phase118_internal_referee_checks.json"
TABLE_COST_LOSS = CFG.project_root / "docs/paper/paper_a_main/tables/phase118_cost_loss.tex"
TABLE_ALPHAS = ("0.02", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80",
                "0.86", "0.90", "0.98")
PRECOMMIT = "docs/research/2026-10-23_phase118_internal_referee_precommit.md"

BASE, PRE, COMP, POST, ORACLE = ("price_only", "msm_d2_12z_cw", "msm_gate_composite_cw",
                                 "msm_d1_00z_cw", "oracle_cw")
DAYLIGHT = (9, 15)                     # 09:00-15:30 JST, phase89's physical window
FLOOR_THR = 0.5
ALPHAS = np.round(np.arange(0.02, 0.99, 0.02), 2)   # phase97's grid
SEED, N_BOOT, BOOT_BLOCK = 2026, 10_000, 7
REPRO_TOL_MW, REPRO_TOL_PRICE = 0.15, 0.0001


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _daylight(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("datetime_jst").dt.hour().is_between(*DAYLIGHT))


def _cw_panel() -> pl.DataFrame:
    panel = pl.read_parquet(P85_PARQUET)
    flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    return panel.filter(pl.all_horizontal(pl.col(f) for f in flags))


def _pooled_mae(per_day: pl.DataFrame, cells: list[str]) -> pl.DataFrame:
    """Daily MAE per cell, averaged over the areas present that day (phase85's pooling)."""
    return (per_day.group_by("test_day")
            .agg(*[pl.col(f"mae_{c}").mean() for c in cells]).sort("test_day"))


# ------------------------------------------------------------------ Q118a
def daylight_row(p85, cw: pl.DataFrame) -> dict:
    """The gate contrast on the daylight slots of all nine areas."""
    day = _daylight(cw)
    per_day = p85._per_day_mae(day, [BASE, PRE, POST])
    pre = p85._pooled_daily(per_day, PRE)
    post = p85._pooled_daily(per_day, POST)
    base_mae = float(per_day[f"mae_{BASE}"].mean())
    pre_entry = p85._entry(pre)
    gate = p85._diff_entry(post, pre)
    ucb = gate["ci95"][1]
    # phase102 scored the post-gate revision as extra features on this same
    # nine-area daylight panel; express it against the same pre-gate value.
    innov = json.loads(P102_JSON.read_text())[
        "pv_cell_daylight_sub_panel_descriptive"]["decomposition_of_the_restriction"][
        "hours_only_all_nine_areas"]
    innov_delta = innov["contrasts_vs_pre_gate"]["pre_plus_wx_innov_minus_pre_gate"]["delta"]
    return {"window_jst": "09:00-15:30", "n_areas": day["area"].n_unique(),
            "n_days": len(pre), "n_slots": day.height,
            "price_only_mae": _r(base_mae),
            "pre_gate_improvement": pre_entry,
            "pre_gate_relative_pct": _r(100 * pre_entry["improvement"] / base_mae, 3),
            "gate_contrast": gate,
            "gate_ucb95": ucb,
            "gate_ucb_share_of_pre_gate_pct": _r(100 * ucb / pre_entry["improvement"], 3),
            "gate_ucb_relative_pct": _r(100 * ucb / base_mae, 3),
            "ci_excludes_zero": bool(gate["ci95"][0] > 0 or gate["ci95"][1] < 0),
            "innovation_features": {
                "source": "phase102 hours_only_all_nine_areas (same slots)",
                "n_slots_phase102": innov["n_slots"], "delta": innov_delta,
                "share_of_pre_gate_pct": _r(100 * innov_delta / pre_entry["improvement"], 2)}}


# ------------------------------------------------------------------ Q118b
def _occto_potential(p89) -> tuple[pl.DataFrame, list[str]]:
    """phase89's potential target with its per-area stamp shift, verbatim."""
    occ_raw = pl.read_parquet(p89.OCCTO_PARQUET)
    if p89.OCCTO_EXT.exists():
        ext = pl.read_parquet(p89.OCCTO_EXT).select(occ_raw.columns)
        occ_raw = pl.concat([occ_raw, ext])
    areas = sorted(a for a in occ_raw["area"].unique().to_list()
                   if a in {*p89.STAMP_SHIFT_MIN})
    parts = []
    for a in areas:
        q = (occ_raw.filter(pl.col("area") == a)
             .with_columns(pl.col("datetime_jst")
                           .dt.offset_by(f"{p89.STAMP_SHIFT_MIN[a]}m")))
        curt = pl.col("pv_curtailment_mw").fill_null(0.0)
        q = q.with_columns((pl.col("pv_mw") + curt).alias("target_mw"))
        parts.append(q.select("area", "datetime_jst", "target_mw",
                              curt.alias("curtail_mw")))
    return pl.concat(parts), areas


def _physical_pools(p89) -> tuple[dict[str, pl.DataFrame], list[str]]:
    occ, areas = _occto_potential(p89)
    pools: dict[str, pl.DataFrame] = {}
    for tier in (PRE, COMP, POST, ORACLE):
        per_area: dict[str, pl.DataFrame] = {}
        for area in areas:
            df = p89._slots(p89._tier_frame(tier, area),
                            occ.filter(pl.col("area") == area), "target_mw")
            if df.height < p89.MIN_TRAIN_ROWS:
                msg = f"{tier}/{area}: {df.height} rows, below the training minimum"
                raise RuntimeError(msg)
            scored = p89._cross_fitted(df, "target_mw").filter(
                pl.col("abs_err_cf").is_finite())
            per_area[area] = (scored.group_by("day")
                              .agg(pl.col("abs_err_cf").mean().alias("mae_cf"))
                              .sort("day"))
        pools[tier] = p89._pool(per_area, "mae_cf").select(
            pl.col("day").cast(pl.Utf8).alias("test_day"),
            pl.col("pooled").alias(f"phys_{tier}"))
        log.info("physical %-24s pooled %.1f MW over %d days", tier,
                 pools[tier][f"phys_{tier}"].mean(), pools[tier].height)
    return pools, areas


def _contrast(p85, a: np.ndarray, b: np.ndarray, scale: float, label: str) -> dict:
    """mean(a - b) with phase85's interval and test; positive = b has lower error."""
    e = p85._diff_entry(a, b)
    # from the unrounded mean: _diff_entry rounds delta to 4 places, and a ratio
    # of a rounded numerator lands on rounding boundaries the prose then prints
    e["relative_pct"] = _r(100 * float(np.mean(a - b)) / scale, 3)
    e["relative_ci95_pct"] = [_r(100 * x / scale, 3) for x in e["ci95"]]
    e["scale"] = _r(scale)
    e["reads"] = label
    return e


def same_window(p85, p89, cw: pl.DataFrame) -> dict:
    pools, areas = _physical_pools(p89)
    phys = pools[PRE]
    for t in (COMP, POST, ORACLE):
        phys = phys.join(pools[t], on="test_day", how="inner")

    price_slots = _daylight(cw.filter(pl.col("area").is_in(areas)))
    cells = [BASE, PRE, COMP, POST, ORACLE]
    price = _pooled_mae(p85._per_day_mae(price_slots, cells), cells)

    # phase89's "common panel" is its pooled days that the price panel also has
    p89_common_days = set(pl.read_parquet(P85_PARQUET, columns=["test_day"])
                          ["test_day"].unique().to_list())
    j = (phys.filter(pl.col("test_day").is_in(list(p89_common_days)))
         .join(price, on="test_day", how="inner").sort("test_day"))
    log.info("same-window panel: %d days, %d areas", j.height, len(areas))

    ph = {t: np.asarray(j[f"phys_{t}"], dtype=float) for t in (PRE, COMP, POST, ORACLE)}
    pr = {t: np.asarray(j[f"mae_{t}"], dtype=float) for t in cells}
    ph_pre, pr_base, pr_pre = ph[PRE].mean(), pr[BASE].mean(), pr[PRE].mean()

    def both(a: str, b: str, what: str) -> dict:
        return {
            "physical_mw": _contrast(p85, ph[a], ph[b], ph_pre,
                                     f"{what}; relative to the pre-gate tier's MW error"),
            "price_vs_price_only": _contrast(p85, pr[a], pr[b], pr_base,
                                             f"{what}; relative to price-only MAE"),
            "price_vs_pre_gate": _contrast(p85, pr[a], pr[b], pr_pre,
                                           f"{what}; relative to the pre-gate tier's MAE"),
        }

    # The precommit defines the ex-post distortion relative to the freshest
    # gate-available run's own error, not the D-2 12 UTC tier's, so it gets a
    # ratio of means with its own interval.
    def ex_post_ratio(x: dict[str, np.ndarray]) -> dict:
        e = p85._ratio_entry(x[COMP] - x[ORACLE], x[COMP])
        return {"pct": _r(100 * e["ratio"], 2),
                "ci95_pct": [_r(100 * v, 2) for v in e["ci95"]],
                "reads": "(freshest pre-gate - realised) / freshest pre-gate; positive = "
                         "realised weather overstates skill, negative = understates"}

    return {
        "window_jst": "09:00-15:30", "areas": areas, "n_areas": len(areas),
        "n_days": j.height,
        "levels": {
            "physical_mw": {t: _r(ph[t].mean(), 1) for t in ph},
            "price_jpykwh": {t: _r(pr[t].mean()) for t in pr},
        },
        "freshness_before_gate": both(PRE, COMP, "D-2 12 UTC minus freshest pre-gate run"),
        "freshness_across_gate": both(COMP, POST, "freshest pre-gate minus post-gate run"),
        "freshness_total": both(PRE, POST, "D-2 12 UTC minus post-gate run"),
        "ex_post_vs_freshest_pre_gate": both(
            COMP, ORACLE, "freshest pre-gate minus realised weather (positive = "
                          "evaluating on realised weather overstates skill)"),
        # for the figure: every tier against the same D-2 12 UTC reference
        "ex_post_vs_pre_gate": both(PRE, ORACLE, "D-2 12 UTC minus realised weather"),
        "ex_post_distortion": {"physical": ex_post_ratio(ph), "price": ex_post_ratio(pr)},
    }


# ------------------------------------------------------------------ Q118c
def _holm_count(p: np.ndarray, level: float = 0.05) -> int:
    """How many of the p-values survive a Holm step-down at `level`."""
    order = np.argsort(p)
    m = len(p)
    n = 0
    for rank, i in enumerate(order):
        if p[i] * (m - rank) <= level:
            n += 1
        else:
            break
    return n


def _daily_aggregates(lab: np.ndarray, p: np.ndarray, day_idx: np.ndarray,
                      n_days: int) -> np.ndarray:
    """Per-day sum of expense at every alpha: protect at cost alpha when p > alpha,
    otherwise pay 1 on an event. Shape (n_days, n_alpha)."""
    out = np.zeros((n_days, len(ALPHAS)))
    for k, a in enumerate(ALPHAS):
        out[:, k] = np.bincount(day_idx, weights=np.where(p > a, a, lab),
                                minlength=n_days)
    return out


def _value(w: np.ndarray, n: np.ndarray, ev: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Murphy's value score for day weights w (rows = draws), one area.

    Returns (n_draws, n_alpha); NaN where climatology and perfection coincide.
    """
    nn = w @ n
    s = (w @ ev) / nn
    ef = (w @ exp) / nn[:, None]
    ec = np.minimum(ALPHAS[None, :], s[:, None])
    ep = ALPHAS[None, :] * s[:, None]
    den = ec - ep
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ec - ef) / den, np.nan)


def cost_loss(p81b) -> dict:
    p85 = (pl.read_parquet(P85_PARQUET)
           .select("area", "datetime_jst", "test_day", "y_actual",
                   pl.col(f"y_pred_{BASE}").alias("yhat_base"),
                   pl.col(f"y_pred_{PRE}").alias("yhat_pre"),
                   pl.col(f"y_pred_{POST}").alias("yhat_post")))
    models = {"price_only": "yhat_base", "pre_gate": "yhat_pre", "post_gate": "yhat_post"}
    frames = []
    for area in sorted(p85["area"].unique().to_list()):
        sub = p85.filter(pl.col("area") == area).sort("datetime_jst")
        sub = sub.with_columns(*[pl.Series(f"p_{m}",
                                           p81b._isotonic_probs(sub, col, FLOOR_THR))
                                 for m, col in models.items()])
        frames.append(sub)
        log.info("isotonic probabilities: %s", area)
    allp = pl.concat(frames).filter(
        pl.all_horizontal(pl.col(f"p_{m}").is_finite() for m in models))
    days = sorted(allp["test_day"].unique().to_list())
    pos = {d: i for i, d in enumerate(days)}
    areas = sorted(allp["area"].unique().to_list())

    agg: dict[str, dict] = {}
    for a in areas:
        s = allp.filter(pl.col("area") == a)
        lab = (s["y_actual"].to_numpy() <= FLOOR_THR).astype(float)
        di = np.array([pos[d] for d in s["test_day"].to_list()])
        agg[a] = {"n": np.bincount(di, minlength=len(days)).astype(float),
                  "ev": np.bincount(di, weights=lab, minlength=len(days)),
                  **{m: _daily_aggregates(lab, s[f"p_{m}"].to_numpy(), di, len(days))
                     for m in models}}

    def macro(w: np.ndarray) -> dict[str, np.ndarray]:
        return {m: np.nanmean(np.stack([_value(w, agg[a]["n"], agg[a]["ev"], agg[a][m])
                                        for a in areas]), axis=0)
                for m in models}

    point = macro(np.ones((1, len(days))))
    rng = np.random.default_rng(SEED)
    n = len(days)
    nb = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, nb))
    idx = ((starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
           ).reshape(N_BOOT, -1)[:, :n]
    w = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(float)
    boot = macro(w)

    def diff(a: str, b: str) -> dict:
        pt = (point[a] - point[b])[0]
        bd = boot[a] - boot[b]
        lo, hi = np.nanquantile(bd, 0.025, axis=0), np.nanquantile(bd, 0.975, axis=0)
        k_max = int(np.nanargmax(pt))
        return {"by_alpha": {f"{al:.2f}": {"delta": _r(pt[i]), "ci95": [_r(lo[i]), _r(hi[i])]}
                             for i, al in enumerate(ALPHAS)},
                "min_delta": _r(np.nanmin(pt)), "max_delta": _r(pt[k_max]),
                "alpha_at_max": float(ALPHAS[k_max]),
                "ci95_at_max": [_r(lo[k_max]), _r(hi[k_max])],
                "max_ucb": _r(np.nanmax(hi)),
                "alpha_at_max_ucb": float(ALPHAS[int(np.nanargmax(hi))]),
                "n_alpha": len(ALPHAS),
                "n_alpha_positive": int(np.sum(pt > 0)),
                "n_alpha_ci_above_zero": int(np.sum(lo > 0)),
                "n_alpha_ci_below_zero": int(np.sum(hi < 0)),
                "multiplicity": "descriptive across 49 ratios; no adjustment"}

    pre_d, post_d = diff("pre_gate", "price_only"), diff("post_gate", "pre_gate")

    # POST HOC (second referee read, 2026-10-23): an actor can always fall back on
    # climatology, whose value score is zero, so a forecast scoring below zero is
    # never used. phase108 applied exactly this correction to the supplement's
    # classifier curve; here the same rule, max(V, 0) on the area-averaged curve,
    # is applied to the main article's tiers before differencing.
    def diff_usable(a: str, b: str) -> dict:
        pt = (np.maximum(point[a], 0) - np.maximum(point[b], 0))[0]
        bd = np.maximum(boot[a], 0) - np.maximum(boot[b], 0)
        lo, hi = np.nanquantile(bd, 0.025, axis=0), np.nanquantile(bd, 0.975, axis=0)
        k_max = int(np.nanargmax(pt))
        keys_ = [f"{al:.2f}" for al in ALPHAS]
        above_ = [bool(x > 0) for x in lo]
        top_ = len(keys_)
        while top_ > 0 and above_[top_ - 1]:
            top_ -= 1
        return {"by_alpha": {k: {"delta": _r(pt[i]), "ci95": [_r(lo[i]), _r(hi[i])]}
                             for i, k in enumerate(keys_)},
                "max_delta": _r(pt[k_max]), "alpha_at_max": float(ALPHAS[k_max]),
                "ci95_at_max": [_r(lo[k_max]), _r(hi[k_max])],
                "max_ucb": _r(np.nanmax(hi)),
                "n_alpha_positive": int(np.sum(pt > 1e-12)),
                "n_alpha_ci_above_zero": int(np.sum(lo > 0)),
                "alphas_ci_above_zero": [k for k, a in zip(keys_, above_, strict=True) if a],
                # one-sided bootstrap p (share of draws at or below zero) and its
                # Holm adjustment across the grid: 49 pointwise intervals invite
                # chance exclusions, and the adjusted count is the one to quote
                "n_alpha_holm_below_005": _holm_count(np.mean(bd <= 0, axis=0)),
                "top_block_from_alpha": float(keys_[top_]) if top_ < len(keys_) else None,
                "top_block_n": len(keys_) - top_,
                "rule": "differences of max(V, 0): forecasts worse than climatology are "
                        "replaced by climatology (V = 0), as in phase108"}

    usable = {"pre_gate_minus_price_only": diff_usable("pre_gate", "price_only"),
              "post_gate_minus_pre_gate": diff_usable("post_gate", "pre_gate")}

    # POST-HOC description of where the post-gate increment is detectable, added
    # after the by-alpha table was seen (the precommit fixed only the counts and
    # the maximum UCB). Defined mechanically so it cannot be tuned: the maximal
    # contiguous block of ratios at the TOP of the grid on which the post-gate
    # interval excludes zero, and the pre-gate forecast's own value there.
    keys = [f"{al:.2f}" for al in ALPHAS]
    above = [post_d["by_alpha"][k]["ci95"][0] > 0 for k in keys]
    top = len(keys)
    while top > 0 and above[top - 1]:
        top -= 1
    block = keys[top:]
    rest = keys[:top]
    peak = f"{pre_d['alpha_at_max']:.2f}"
    where = {
        "definition": "post-hoc: maximal contiguous block of ratios at the top of the "
                      "grid where the post-gate increment's 95% interval excludes zero",
        "alphas_ci_above_zero": [k for k, a in zip(keys, above, strict=True) if a],
        "top_block_from_alpha": float(block[0]) if block else None,
        "top_block_n": len(block),
        "pre_gate_value_max_in_block": (_r(max(point["pre_gate"][0][keys.index(k)]
                                               for k in block)) if block else None),
        "post_gate_value_max_in_block": (_r(max(point["post_gate"][0][keys.index(k)]
                                                for k in block)) if block else None),
        "n_ci_above_zero_below_block": int(sum(above[:top])),
        "max_abs_delta_below_block": _r(max(abs(post_d["by_alpha"][k]["delta"])
                                            for k in rest)),
        "max_ucb_below_block": _r(max(post_d["by_alpha"][k]["ci95"][1] for k in rest)),
        "post_gate_at_pre_gate_peak": {"alpha": float(peak), **post_d["by_alpha"][peak]},
    }

    base_rate = float(np.mean([agg[a]["ev"].sum() / agg[a]["n"].sum() for a in areas]))
    return {"event": f"price <= {FLOOR_THR} JPY/kWh",
            "probabilities": "expanding-window isotonic calibration of each tier's "
                             "price forecast (phase81b, 30-day burn-in)",
            "n_days": n, "n_areas": len(areas), "n_slots": allp.height,
            "base_rate_macro_pct": _r(100 * base_rate, 2),
            "value_curves": {m: {f"{al:.2f}": _r(point[m][0][i])
                                 for i, al in enumerate(ALPHAS)} for m in models},
            "max_value_pre_gate": _r(np.nanmax(point["pre_gate"][0])),
            "pre_gate_minus_price_only": pre_d,
            "post_gate_minus_pre_gate": post_d,
            "post_gate_where_detectable": where,
            "usable_value": usable}


# ------------------------------------------------------------------ bound summary
def bound_summary(daylight: dict) -> dict:
    """Every post-gate upper bound as a share of the matching pre-gate gain.

    POST HOC (second referee read): the text quoted one bound, while two further
    ways of using the post-gate run (its revision as extra features) and the
    learners give others. Collected here so the largest is quoted from one place.
    """
    p102 = json.loads(P102_JSON.read_text())
    pre_full = p102["improvements_vs_price_only"]["msm_d2_12z_cw"]["improvement"]
    c_full = p102["contrasts_vs_pre_gate_full_panel"]
    hours = p102["pv_cell_daylight_sub_panel_descriptive"]["decomposition_of_the_restriction"][
        "hours_only_all_nine_areas"]
    pre_day = hours["levels_mae"]["price_only"] - hours["levels_mae"]["msm_d2_12z_cw"]
    p113 = json.loads((RUN / "phase113_learner_strength.json").read_text())["learners"]
    entries = {
        "replacement_pooled": 100 * c_full["post_gate_minus_pre_gate_REPRODUCED"]["ci95"][1] / pre_full,
        "wx_revision_pooled": 100 * c_full["pre_plus_wx_innov_minus_pre_gate"]["ci95"][1] / pre_full,
        "pv_revision_pooled": 100 * c_full["pre_plus_pv_innov_minus_pre_gate"]["ci95"][1] / pre_full,
        "replacement_daylight": daylight["gate_ucb_share_of_pre_gate_pct"],
        "wx_revision_daylight": 100 * hours["contrasts_vs_pre_gate"][
            "pre_plus_wx_innov_minus_pre_gate"]["ci95"][1] / pre_day,
        "pv_revision_daylight": 100 * hours["contrasts_vs_pre_gate"][
            "pre_plus_pv_innov_minus_pre_gate"]["ci95"][1] / pre_day,
        **{f"learner_{k}": 100 * v["gate_ucb_share_of_pre_gate"]
           for k, v in p113.items() if k != "xgb_fixed"},
    }
    entries = {k: _r(v, 2) for k, v in entries.items()}
    k_max = max(entries, key=entries.get)
    return {"entries_pct": entries, "max": {"key": k_max, "pct": entries[k_max]},
            "pv_revision_pooled": c_full["pre_plus_pv_innov_minus_pre_gate"],
            "reads": "95% upper end of each post-gate increment as % of the pre-gate gain "
                     "on the same slots and learner"}


# ------------------------------------------------------------------ gate 118-0
def reproduction(p85, cw: pl.DataFrame, sw: dict) -> dict:
    """The imports must reproduce the published levels before any cut is read."""
    per_day = p85._per_day_mae(cw, [BASE, PRE])
    pre = float(p85._pooled_daily(per_day, PRE).mean())
    pub = json.loads(P85_JSON.read_text())["arm_cw_primary"]["tiers"][PRE]["improvement"]
    p89 = json.loads(P89_JSON.read_text())["targets"]["potential"]["tiers"]
    phys = {t: {"here": sw["levels"]["physical_mw"][t],
                "published": p89[t]["pooled_mae_cf_common_mw"]}
            for t in (PRE, COMP, POST, ORACLE)}
    ok_phys = all(abs(v["here"] - v["published"]) <= REPRO_TOL_MW for v in phys.values())
    ok_price = abs(pre - pub) <= REPRO_TOL_PRICE
    return {"price_pre_gate_full_panel": {"here": _r(pre), "published": pub,
                                          "ok": ok_price},
            "physical_levels_common_panel": phys, "physical_ok": ok_phys,
            "note": "physical levels are compared on phase89's own common panel; the "
                    "same-window panel adds the CW-flag restriction of the price panel, "
                    "so a match here means the construction, not the day set, is shared",
            "passed": bool(ok_phys and ok_price)}


def _fx(x: float, nd: int = 3) -> str:
    v = f"{x:+.{nd}f}"
    return v.replace("-", "$-$")


def write_cost_loss_table(cl: dict) -> None:
    """The value curve on a readable grid of ratios, for supplement S2."""
    vc = cl["value_curves"]
    pre = cl["usable_value"]["pre_gate_minus_price_only"]["by_alpha"]
    post = cl["usable_value"]["post_gate_minus_pre_gate"]["by_alpha"]
    rows = []
    for a in TABLE_ALPHAS:
        rows.append(
            f"{a} & {_fx(vc['price_only'][a])} & {_fx(vc['pre_gate'][a])} & "
            f"{_fx(vc['post_gate'][a])} & {_fx(pre[a]['delta'])} "
            f"[{_fx(pre[a]['ci95'][0])}, {_fx(pre[a]['ci95'][1])}] & "
            f"{_fx(post[a]['delta'])} [{_fx(post[a]['ci95'][0])}, {_fx(post[a]['ci95'][1])}] \\\\")
    tex = "\n".join([
        "% GENERATED by 69_internal_referee_checks.py -- do not edit by hand.",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Murphy's value score $V$ of the main article's floor probabilities "
        "(isotonic calibration of each tier's price forecast) at selected cost--loss "
        "ratios $C/L$, averaged over the nine areas. The gain and increment columns "
        "difference the usable value $\\max(V, 0)$, because an actor whose forecast "
        "scores below climatology falls back on climatology ($V = 0$): the pre-gate "
        "run's gain over the price-only model and the post-gate run's increment over the "
        "pre-gate run. Intervals are 95\\% block-bootstrap intervals over days; the full "
        f"curve has {cl['pre_gate_minus_price_only']['n_alpha']} ratios.}}",
        "\\label{tab:cost_loss_tiers}", "\\small",
        "\\begin{tabular}{rrrrcc}", "\\toprule",
        "$C/L$ & $V$, price only & $V$, pre-gate & $V$, post-gate & Pre-gate gain & Post-gate increment \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    TABLE_COST_LOSS.write_text(tex, encoding="utf-8")
    log.info("wrote %s", TABLE_COST_LOSS)


def main() -> int:
    p85 = _load("p85", P85_SCRIPT)
    p89 = _load("p89", P89_SCRIPT)
    p81b = _load("p81b", P81B_SCRIPT)
    cw = _cw_panel()

    sw = same_window(p85, p89, cw)
    # Gate 118-0 compares physical levels on phase89's own common panel, which is
    # the same-window panel without the price CW restriction. Recompute them there.
    pools, _ = _physical_pools(p89)
    days85 = set(pl.read_parquet(P85_PARQUET, columns=["test_day"])["test_day"]
                 .unique().to_list())
    sw_p89 = {"levels": {"physical_mw": {
        t: _r(pools[t].filter(pl.col("test_day").is_in(list(days85)))
              [f"phys_{t}"].mean(), 1) for t in pools}}}
    gate0 = reproduction(p85, cw, sw_p89)
    log.info("gate 118-0: %s", "PASS" if gate0["passed"] else "FAIL")
    if not gate0["passed"]:
        log.error("reproduction failed: %s", json.dumps(gate0, indent=1))
        return 1

    out = {"phase": "118", "precommit": PRECOMMIT, "git_sha": _git_sha(),
           "sources": [P85_PARQUET.name, "phase89 construction (imported)",
                       "phase81b isotonic probabilities (imported)"],
           "gate118_0_reproduction": gate0,
           "q118a_daylight_row": daylight_row(p85, cw),
           "q118b_same_window": sw,
           "q118c_cost_loss": cost_loss(p81b),
           "bound_summary": None,
           "metadata": {"seed": SEED, "n_boot": N_BOOT, "block_days": BOOT_BLOCK}}
    out["bound_summary"] = bound_summary(out["q118a_daylight_row"])
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    log.info("wrote %s", OUT_JSON)
    if TABLE_COST_LOSS.parent.exists():  # the table lives with the manuscript
        write_cost_loss_table(out["q118c_cost_loss"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
