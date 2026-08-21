r"""Phase 90: does the post-gate innovation reach the physical day but not the price?

WHY
---
The paper's third result is currently two mean-absolute-error columns: the
post-gate run tracks the physical day better, and adds nothing detectable to the
price. That is a divergence between two targets, not evidence about price
formation, and the seventh review listed six alternative explanations for it.
This phase runs the sharper test the review asked for.

THE TEST
--------
Define the post-gate innovation per slot as the change in the scaled physical
proxy when the later run replaces the earlier one:

    dPV_t = proxy_scaled(post-gate)_t - proxy_scaled(pre-gate)_t

and the two pre-gate residuals

    e_phys_t = target_t          - proxy_scaled(pre-gate)_t
    e_price_t = y_actual_t       - y_pred(pre-gate cell)_t

then ask whether dPV explains each. If the innovation carries information the
physical day realises but the price does not, the first regression should have
a slope and the second should not.

Slopes are OLS with a cluster-robust variance on area x date, because slots in a
day and areas in a day are not independent. A placebo regression on a Gaussian
draw with the same variance is run alongside, and if the placebo is significant
the whole phase is void -- that is an implementation check, not a result.

Branches are fixed in
docs/research/2026-08-13_phase90_mechanism_precommit.md. No branch licenses the
word "identifies": participants' expectations are not observed here.

CLI (from the repository root):
  uv run python scripts/51_mechanism_test.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase90")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PHYS_SLOTS = RUN / "phase89_physical_per_slot.parquet"
PRICE_SLOTS = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase90_mechanism.json"
PRECOMMIT = "docs/research/2026-08-13_phase90_mechanism_precommit.md"

SEED = 2026
PRE_GATE, POST_GATE = "msm_d2_12z_cw", "msm_d1_00z_cw"
NEAR_FLOOR = 0.5


def _j(x, nd: int = 6):
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


def _cluster_ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> dict:
    """Slope of y on x with an intercept, cluster-robust SE on `cluster`.

    Two-column design, so the sandwich is written out rather than pulled from a
    library: X'X is 2x2 and the meat is a sum of per-cluster outer products.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((2, 2))
    order = np.argsort(cluster, kind="stable")
    cl = cluster[order]
    Xo, ro = X[order], resid[order]
    bounds = np.flatnonzero(np.r_[True, cl[1:] != cl[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        s = Xo[a:b].T @ ro[a:b]
        meat += np.outer(s, s)
    g = len(bounds) - 1
    scale = g / (g - 1) * (n - 1) / (n - 2) if g > 1 else 1.0
    vcv = xtx_inv @ (scale * meat) @ xtx_inv
    se = float(np.sqrt(vcv[1, 1]))
    slope = float(beta[1])
    t = slope / se if se > 0 else float("nan")
    # R^2 increment of x over an intercept-only fit
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    return {"slope": _j(slope), "se_cluster": _j(se), "t": _j(t, 3),
            # erfc(|t|/sqrt(2)) IS the two-sided normal tail, P(|Z| > |t|).
            # The previous expression was `2 * erfc(...) / 2 * 2`, which reduces
            # to 2 * erfc(...) -- every p-value in this file was exactly DOUBLED.
            # The direction mattered: it made the main slopes look conservative
            # but made the PLACEBO look more null than it is, and the placebo is
            # a pre-registered void condition for the whole table.
            "p_two_sided": _j(math.erfc(abs(t) / math.sqrt(2))
                              if math.isfinite(t) else None, 6),
            "n": int(n), "n_clusters": int(g),
            "r2_increment": _j(1.0 - ss_res / ss_tot if ss_tot > 0 else None, 6)}


def _report(name: str, y: np.ndarray, x: np.ndarray, cl: np.ndarray) -> dict:
    r = _cluster_ols(y, x, cl)
    log.info("  %-34s slope %+.5f  (cluster SE %.5f, t %+.2f, p %.4f, n %d/%d cl)",
             name, r["slope"], r["se_cluster"], r["t"], r["p_two_sided"],
             r["n"], r["n_clusters"])
    return r


def main() -> int:
    for p in (PHYS_SLOTS, PRICE_SLOTS):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    phys = pl.read_parquet(PHYS_SLOTS)
    pre = (phys.filter(pl.col("tier") == PRE_GATE)
           .select("area", "datetime_jst", "day", "target_mw", "curtail_mw",
                   pl.col("proxy_scaled_cf").alias("proxy_pre")))
    post = (phys.filter(pl.col("tier") == POST_GATE)
            .select("area", "datetime_jst",
                    pl.col("proxy_scaled_cf").alias("proxy_post")))
    j = pre.join(post, on=["area", "datetime_jst"], how="inner")
    log.info("physical slots joined across both tiers: %d", j.height)

    price = (pl.read_parquet(PRICE_SLOTS,
                             columns=["area", "datetime_jst", "test_day",
                                      "y_actual", f"y_pred_{PRE_GATE}"])
             .rename({f"y_pred_{PRE_GATE}": "y_pred_pre"}))
    m = j.join(price, on=["area", "datetime_jst"], how="inner")
    log.info("slots with both a physical and a price residual: %d", m.height)

    m = m.with_columns(
        (pl.col("proxy_post") - pl.col("proxy_pre")).alias("dpv"),
        (pl.col("target_mw") - pl.col("proxy_pre")).alias("e_phys"),
        (pl.col("y_actual") - pl.col("y_pred_pre")).alias("e_price"),
        (pl.col("curtail_mw") > 0).alias("curtailed"),
        (pl.col("y_actual") <= NEAR_FLOOR).alias("floor"),
        (pl.col("area") + "|" + pl.col("test_day")).alias("cluster"),
    ).drop_nulls(["dpv", "e_phys", "e_price"])

    # The physical residual is in MW and the price residual in JPY/kWh, so the
    # innovation is standardised: slopes are then "per one standard deviation of
    # the innovation" and the two regressions are on a comparable footing.
    sd = float(np.std(m["dpv"].to_numpy()))
    m = m.with_columns((pl.col("dpv") / sd).alias("dpv_z"))
    log.info("innovation sd = %.2f MW; slots = %d, clusters = %d",
             sd, m.height, m["cluster"].n_unique())

    rng = np.random.default_rng(SEED)
    m = m.with_columns(pl.Series("placebo", rng.standard_normal(m.height)))

    out: dict = {
        "phase": 90, "title": "Mechanism: does the post-gate innovation reach the price?",
        "precommit_doc": PRECOMMIT,
        "design": {
            "innovation": ("proxy_scaled(post-gate) - proxy_scaled(pre-gate) per "
                           "slot, standardised; both proxies are the cross-fitted "
                           "physical predictions of phase89"),
            "physical_residual": "target_mw - proxy_scaled(pre-gate)",
            "price_residual": "y_actual - y_pred(pre-gate cell)",
            "inference": "OLS with cluster-robust SE on area x delivery day",
            "innovation_sd_mw": _j(sd, 2),
            "n_slots": m.height, "n_clusters": m["cluster"].n_unique(),
        },
        "regressions": {}, "splits": {}, "gates": {},
    }

    x = m["dpv_z"].to_numpy()
    cl = m["cluster"].to_numpy()
    log.info("full sample:")
    out["regressions"]["physical_residual_on_innovation"] = _report(
        "e_phys ~ innovation", m["e_phys"].to_numpy(), x, cl)
    out["regressions"]["price_residual_on_innovation"] = _report(
        "e_price ~ innovation", m["e_price"].to_numpy(), x, cl)
    out["regressions"]["placebo_physical"] = _report(
        "e_phys ~ placebo (must be null)", m["e_phys"].to_numpy(),
        m["placebo"].to_numpy(), cl)
    out["regressions"]["placebo_price"] = _report(
        "e_price ~ placebo (must be null)", m["e_price"].to_numpy(),
        m["placebo"].to_numpy(), cl)

    for name, mask in (("floor", m["floor"]), ("non_floor", ~m["floor"]),
                       ("curtailed", m["curtailed"]),
                       ("not_curtailed", ~m["curtailed"])):
        sub = m.filter(mask)
        if sub.height < 500:
            continue
        log.info("split %s (n=%d):", name, sub.height)
        out["splits"][name] = {
            "physical": _report(f"  e_phys ~ innovation [{name}]",
                                sub["e_phys"].to_numpy(),
                                sub["dpv_z"].to_numpy(), sub["cluster"].to_numpy()),
            "price": _report(f"  e_price ~ innovation [{name}]",
                             sub["e_price"].to_numpy(),
                             sub["dpv_z"].to_numpy(), sub["cluster"].to_numpy()),
        }

    # ---- what the innovation could be worth, in the paper's own units ------
    # ADDED after Gate 56 fired, and not a gate: the manuscript needed to say
    # what "1.2% of residual variance" means against a one-tick equivalence
    # margin, and asserting "too little to matter" turned out to be WRONG when
    # the arithmetic was done. Removing the innovation's linear projection from
    # the pre-gate price residual cuts its MAE by the amount below -- fitted
    # in-sample on the same slots, so it is a CEILING on what a linear use of
    # this information could buy, not an achievable gain.
    e_price = m["e_price"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    beta_p = np.linalg.lstsq(X, e_price, rcond=None)[0]
    resid_p = e_price - X @ beta_p
    mae_before = float(np.abs(e_price).mean())
    mae_after = float(np.abs(resid_p).mean())
    out["implied_price_value"] = {
        "panel": ("the slots where both a physical and a price residual exist: "
                  "daylight hours in the eight areas with published PV actuals, "
                  "which is a HARDER sub-panel than the full ladder -- see "
                  "implied_price_value_full_panel for the like-for-like figure"),
        "n_slots": int(len(e_price)),
        "mae_pre_gate_residual": _j(mae_before, 4),
        "mae_after_removing_projection": _j(mae_after, 4),
        "implied_mae_reduction": _j(mae_before - mae_after, 4),
        "in_ticks": _j((mae_before - mae_after) / 0.01, 2),
        "reading": ("an in-sample ceiling on what a linear use of the excluded "
                    "run's innovation could take off the pooled MAE. Compare it "
                    "with the equivalence margin (0.01) and with the gate-timing "
                    "delta the ladder measures: the ceiling is about one tick, "
                    "and the learner realises a fraction of it"),
    }
    log.info("implied ceiling: MAE %.4f -> %.4f, reduction %.4f (%.2f ticks)",
             mae_before, mae_after, mae_before - mae_after,
             (mae_before - mae_after) / 0.01)

    # ---- the same ceiling on the FULL ladder panel --------------------------
    # The figure above is computed where the innovation EXISTS: daylight hours in
    # eight areas, whose pre-gate residual MAE is 2.45. The equivalence margin and
    # the gate-timing delta the manuscript compares it with are FULL-panel
    # quantities, where that MAE is 1.89, so comparing them directly mixes panels.
    #
    # The construction has to be chosen carefully. Refitting a single projection
    # on the full panel with the innovation zero-filled makes the ceiling come out
    # NEGATIVE (-0.0016): three quarters of the slots carry z = 0, the intercept
    # moves to suit them, and a least-squares projection is not guaranteed to
    # reduce a mean ABSOLUTE error. What answers the question is to apply the
    # sub-panel's own slope on the slots where the innovation exists, leave every
    # other slot untouched, and measure the change in the full-panel MAE. That is
    # "what this information could buy, expressed on the panel the ladder scores".
    price_full = (pl.read_parquet(PRICE_SLOTS,
                                  columns=["area", "datetime_jst", "y_actual",
                                           f"y_pred_{PRE_GATE}"])
                  .rename({f"y_pred_{PRE_GATE}": "y_pred_pre"}))
    innov = m.select("area", "datetime_jst", pl.col("dpv_z").alias("z"))
    full = (price_full.join(innov, on=["area", "datetime_jst"], how="left")
            .with_columns(pl.col("z").fill_null(0.0))
            .with_columns((pl.col("y_actual") - pl.col("y_pred_pre")).alias("e")))
    zf, ef = full["z"].to_numpy(), full["e"].to_numpy()
    slope_sub = float(beta_p[1])              # fitted on the sub-panel above
    ef_adj = ef - slope_sub * zf              # zero where the innovation is absent
    mae_bf, mae_af = float(np.abs(ef).mean()), float(np.abs(ef_adj).mean())
    covered = int((zf != 0).sum())
    # An OLS slope minimises SQUARED error, so removing its projection is ONE
    # linear use of the innovation, not the best one in MAE. Calling its result a
    # "ceiling" overstated nothing but understated the bound: the supremum over
    # linear rules is the least-absolute-deviations slope, and mean|e - b z| is
    # convex piecewise-linear in b with its minimum at the |z|-weighted median of
    # e/z over the slots where z is non-zero. Flagged by an external code audit,
    # 2026-08-18; the bound moves from 0.0034 to 0.0036 and the realised share
    # from 75% to 72%, so the paper now claims a genuine maximum.
    nz = zf != 0
    ratio, wt = ef[nz] / zf[nz], np.abs(zf[nz])
    o = np.argsort(ratio)
    ratio, wt = ratio[o], wt[o]
    cw = np.cumsum(wt)
    slope_lad = float(ratio[np.searchsorted(cw, cw[-1] / 2.0)])
    mae_lad = float(np.abs(ef - slope_lad * zf).mean())
    if mae_lad > mae_af:  # the weighted median must not lose to the OLS slope
        msg = (f"the MAE-optimal slope ({slope_lad:.5f}) gives a WORSE mean "
               f"absolute error ({mae_lad:.6f}) than the OLS slope "
               f"({mae_af:.6f}); the weighted-median solution is wrong")
        raise RuntimeError(msg)
    out["implied_price_value_full_panel"] = {
        "panel": ("every slot the ladder scores, nine areas. The sub-panel's own "
                  "slope is applied where the innovation exists and nowhere else"),
        "n_slots": int(len(ef)),
        "n_slots_with_innovation": covered,
        "share_of_slots_covered_pct": _j(100.0 * covered / len(ef), 1),
        "mae_pre_gate_residual": _j(mae_bf, 4),
        # the OLS projection: one linear use, kept for comparability with the
        # earlier draft that reported it as the ceiling
        "ols_slope": _j(slope_sub, 5),
        "mae_after_removing_ols_projection": _j(mae_af, 4),
        "ols_mae_reduction": _j(mae_bf - mae_af, 4),
        # the SUPREMUM over linear uses, which is what "ceiling" has to mean
        "mae_optimal_slope": _j(slope_lad, 5),
        "mae_after_removing_optimal": _j(mae_lad, 4),
        "implied_mae_reduction": _j(mae_bf - mae_lad, 4),
        "in_ticks": _j((mae_bf - mae_lad) / 0.01, 2),
        "refit_on_full_panel_would_give": _j(
            float(np.abs(ef).mean() - np.abs(
                ef - np.column_stack([np.ones(len(zf)), zf]) @ np.linalg.lstsq(
                    np.column_stack([np.ones(len(zf)), zf]), ef, rcond=None)[0]
            ).mean()), 4),
        "reading": ("the like-for-like ceiling: compare THIS with the equivalence "
                    "margin and with the gate-timing delta, both full-panel. The "
                    "sub-panel figure is larger because the sub-panel is harder "
                    "and because the innovation is zero on three quarters of the "
                    "ladder's slots by construction"),
    }
    log.info("full-panel ceiling: MAE %.4f -> %.4f (MAE-optimal slope %+.5f), "
             "reduction %.4f (%.2f ticks) over %.1f%% of slots; the OLS "
             "projection gives only %.4f", mae_bf, mae_lad, slope_lad,
             mae_bf - mae_lad, (mae_bf - mae_lad) / 0.01,
             100.0 * covered / len(ef), mae_bf - mae_af)

    # ---- how much of that ceiling the ladder's forecasts actually realise ----
    # Done HERE, in one place, because the comparison is what the manuscript
    # makes and it spans two provenance files: a reader (or a drift guard)
    # otherwise has to divide two numbers from different JSONs by hand. The first
    # version of this comparison mixed panels and reported the ceiling as "about
    # one tick" against a full-panel margin; it is a third of a tick.
    ladder = RUN / "phase85_exact_ladder.json"
    if ladder.exists():
        gt = json.loads(ladder.read_text())["arm_cw_primary"][
            "gate46_47_48_gate_timing"]
        ceiling = mae_bf - mae_lad   # the supremum, not the OLS projection
        realised = float(gt["delta"])
        out["realised_vs_ceiling"] = {
            "ceiling_full_panel": _j(ceiling, 4),
            "measured_gate_timing_delta": _j(realised, 4),
            "measured_ci95": gt.get("ci95"),
            "realised_share_pct": _j(100.0 * realised / ceiling, 1)
            if ceiling > 0 else None,
            "extraction_gap": _j(ceiling - realised, 4),
            "equivalence_margin": 0.01,
            "reading": ("both the ceiling and what the design realises sit inside "
                        "the equivalence margin, which is why the pooled test "
                        "reports equivalence. The gap between them bounds how "
                        "much a better encoding of this information could add"),
        }
        log.info("realised %.4f of a %.4f ceiling (%.0f%%); gap %.4f, margin 0.01",
                 realised, ceiling, 100.0 * realised / ceiling, ceiling - realised)

    phys_p = out["regressions"]["physical_residual_on_innovation"]["p_two_sided"]
    price_p = out["regressions"]["price_residual_on_innovation"]["p_two_sided"]
    placebo_ok = all(out["regressions"][k]["p_two_sided"] > 0.05
                     for k in ("placebo_physical", "placebo_price"))
    branch = ("a" if phys_p < 0.05 <= price_p else
              "b" if phys_p < 0.05 and price_p < 0.05 else
              "c")
    out["gates"]["gate56_mechanism"] = {
        "placebo_null": placebo_ok,
        "physical_p": phys_p, "price_p": price_p, "branch": branch,
        "reading": {
            "a": ("the innovation explains the physical residual and not the "
                  "price residual: CONSISTENT WITH prices reflecting pre-gate "
                  "expectations, but this does not identify that mechanism and "
                  "the alternatives are not excluded"),
            "b": ("the innovation explains both residuals: the divergence is "
                  "not about price formation and the expectations reading is "
                  "withdrawn"),
            "c": ("the innovation explains neither residual: the physical gain "
                  "is not usable information and the channel claim is reduced "
                  "to the MAE comparison"),
        }[branch],
        "wording_rule": ("no branch licenses 'identifies' or 'locates the "
                         "expectations channel'; the permitted phrasing is "
                         "'target divergence, consistent with'"),
    }
    if not placebo_ok:
        out["gates"]["gate56_mechanism"]["VOID"] = (
            "a placebo regression was significant; treat every slope here as an "
            "implementation artefact until that is explained")

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "seed": SEED}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    log.info("GATE 56: branch %s | placebo null = %s", branch, placebo_ok)
    log.info("  %s", out["gates"]["gate56_mechanism"]["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
