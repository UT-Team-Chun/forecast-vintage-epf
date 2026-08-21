r"""Phase 64: emit the S4 supplementary tables from provenance JSON.

WHY THIS EXISTS
---------------
The wedge, nomination-band and physical-outcome tables in Supplement S4 were first
written with literal numbers. When the decision model was re-run on the run-pinned
tier every one of those literals went stale at once, and nothing would have caught
it: ``verify_macros`` guards macros, and these were not macros.

The repo already has the right pattern for this --- S3 does
``\input{tables/phase23_daily_robustness.tex}`` --- so these tables are generated
the same way. A number that appears only in a supplementary table is now as
drift-guarded as one in the main text, because regenerating is the only way to
change it.

Outputs (all under docs/paper/paper_a_main/tables/):
  phase60_wedge_regressions.tex     area-day regressions on wedge exposure
  phase60_split_sensitivity.tex     two-window estimates as the split date moves
  phase61_nomination_band.tex       R1 information-responsive and R3 tolerance band
  phase61_physical_by_policy.tex    physical outcomes per commitment policy
  phase89_physical_tiers.tex        physical-target error per vintage, and what
                                    the two superseded conventions were worth
  phase90_mechanism.tex             each residual regressed on the excluded
                                    run's innovation, with the placebos
  phase91_weight_sensitivity.tex    three point weightings, three cells
  phase54v2_enriched_nwp.tex        the four-cell enriched-baseline
                                    decomposition, per area
  phase83_margin_curve.tex          the equivalence margin curve, every
                                    dependence scheme (MAIN TEXT)
  phase85_vintage_ladder.tex        the main-article ladder table (MAIN TEXT)
  phase82b_absolute.tex             absolute MAE/RMSE and skill against the
                                    naive 168-hour benchmark (MAIN TEXT)

The three main-text tables are also emitted into the Japanese manuscript's
tables/ directory with Japanese wording and identical numbers.

Each file is a bare ``tabular`` body: the caption, label and float stay in the
.tex prose so the wording remains editable by hand.

CLI (from the repository root):
  uv run python scripts/90_emit_tables.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import pv_peak_by_area

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase64")

CFG = EpfConfig()
# Layout-aware for the same reason verify_macros.py is: this file is shared
# with the public companion, which keeps provenance at data/provenance/ and
# ships no manuscript tree at all. Resolving by CONTENT rather than by
# existence -- both directories may exist, only one holds the JSONs.
_RUN_CANDIDATES = (CFG.resolved("run_dir"), CFG.project_root / "data/provenance")
RUN = max(_RUN_CANDIDATES,
          key=lambda c: len(list(c.rglob("*.json"))) if c.exists() else -1)
OUT = CFG.project_root / "docs/paper/paper_a_main/tables"
# The Japanese manuscript carries the same three main-text tables. Its numbers
# were literals too -- six of them -- so it gets generated bodies with Japanese
# headers rather than a hand-kept copy that can drift from the English one.
OUT_JA = CFG.project_root / "docs/paper/paper_a_main_ja/tables"
P60 = RUN / "phase60_wedge_continuous.json"
P61 = RUN / "phase61_system_metrics_band.json"
P76 = RUN / "phase76_nomination_rules.json"
P73 = RUN / "phase73_cycle_curve.json"
P74 = RUN / "phase74_cw_ladder.json"
P74G = RUN / "phase74_gate35_proxy.json"
P59 = RUN / "phase59_exact_ladder.json"
P259 = RUN / "phase25.9_floor_probit_summary.json"
P89 = RUN / "phase89_physical_target.json"
P90 = RUN / "phase90_mechanism.json"
P83 = RUN / "phase83_inference_sensitivity.json"
P82 = RUN / "phase82_weighted_metrics.json"
P82B = RUN / "phase82b_absolute_metrics.json"
P85 = RUN / "phase85_exact_ladder.json"
P91 = RUN / "phase91_weight_sensitivity.json"
P54V2 = RUN / "phase54v2_current.json"
P88 = RUN / "phase88_lit_audit.json"

AREA_LABELS = {
    "hokkaido": "Hokkaido",
    "tohoku": "Tohoku",
    "tokyo": "Tokyo",
    "chubu": "Chubu",
    "hokuriku": "Hokuriku",
    "kansai": "Kansai",
    "chugoku": "Chugoku",
    "shikoku": "Shikoku",
    "kyushu": "Kyushu",
}

POLICY_LABELS = [
    ("i_perfect", "Perfect information"),
    ("i_op", "Pre-gate vintage (run-pinned)"),
    ("i_base", "Price-only baseline"),
    ("i_proxy", "Stale-observation proxy"),
    ("always_hedge", "Never sell forward"),
]


def _g(d: dict, *path):
    for k in path:
        d = d[k]
    return d


def _f(x: float | None, nd: int = 2, *, signed: bool = False) -> str:
    """LaTeX-ready number, or an em dash when the provenance has none."""
    if x is None:
        return "---"
    s = f"{x:+.{nd}f}" if signed else f"{x:,.{nd}f}"
    return f"${s}$"


def _p(x: float | None) -> str:
    if x is None:
        return "---"
    return "$< 10^{-4}$" if x < 1e-4 else f"${x:.3f}$"


def _hu(x: float, nd: int = 1) -> str:
    """Round half UP for display, the way the hand-written macros do.

    ``f"{70.55:.1f}"`` is ``70.5``: 70.55 has no exact binary representation and
    the nearest double is below the midpoint. macros.tex says 70.6, so the
    supplement's table and the main text's prose printed the same confidence
    interval two different ways -- and PhysDeltaHi's tolerance was wide enough to
    certify both, so the guard could not adjudicate. Display rounding is a
    presentation decision and it has to be made in ONE place.
    """
    return str(Decimal(str(x)).quantize(Decimal(f"1e-{nd}"), rounding=ROUND_HALF_UP))


def _pct_small(x: float | None) -> str:
    """Percentages that can be genuinely tiny: 0.00 would read as exactly zero."""
    if x is None:
        return "---"
    return "$< 0.01$" if 0 < x < 0.01 else f"${x:.2f}$"


def _coef_rows(coefs: dict, names: list[tuple[str, str]]) -> list[str]:
    rows = []
    for key, label in names:
        c = coefs.get(key)
        if c is None:
            continue
        rows.append(
            f"\\quad {label} & {_f(c['beta'], 2, signed=True)} & "
            f"{_f(c['se'], 2)} & {_p(c['p'])} & \\\\"
        )
    return rows


def wedge_regressions(p60: dict) -> str:
    prim = _g(p60, "c2_continuous", "primary")
    tw = _g(p60, "c2_continuous", "cluster_day_area", "coefficients", "exposure_floor")
    fl = _g(p60, "c4_components", "floor_component")
    nf = _g(p60, "c4_components", "nonfloor_component")
    n = prim["n_obs"]
    L = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dependent variable / regressor & $\beta$ & s.e. & $p$ & $R^2$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Profit difference, all slots (month FE)}} \\",
    ]
    L += _coef_rows(
        prim["coefficients"],
        [("exposure_floor", "floor-slot exposure"), ("exposure_nonfloor", "non-floor exposure")],
    )
    L[-1] = L[-1][:-3] + f"{_f(prim['r2'], 3)} \\\\"  # R^2 on the last row
    L.append(
        r"\quad \emph{same, two-way day $\times$ area clustering} & "
        f"{_f(tw['beta'], 2, signed=True)} & {_f(tw['se'], 2)} & "
        f"{_p(tw['p'])} & \\\\"
    )
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\emph{Floor-slot profit component (area FE)}} \\"]
    L += _coef_rows(
        fl["coefficients"],
        [
            ("exposure_floor", "floor-slot exposure"),
            ("exposure_nonfloor", "non-floor exposure"),
            ("post", "post-2025-10-31 (level)"),
            ("post_x_exposure_floor", r"post $\times$ floor exposure"),
        ],
    )
    L[-1] = L[-1][:-3] + f"{_f(fl['r2'], 3)} \\\\"
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\emph{Non-floor profit component (area FE)}} \\"]
    L += _coef_rows(
        nf["coefficients"],
        [
            ("exposure_floor", "floor-slot exposure"),
            ("exposure_nonfloor", "non-floor exposure"),
            ("post", "post-2025-10-31 (level)"),
            ("post_x_exposure_nonfloor", r"post $\times$ non-floor exposure"),
        ],
    )
    L[-1] = L[-1][:-3] + f"{_f(nf['r2'], 3)} \\\\"
    L += [
        r"\midrule",
        f"\\multicolumn{{5}}{{l}}{{$n = {n:,}$ area-days; "
        r"standard errors clustered by day.} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(L) + "\n"


def split_sensitivity(p60: dict) -> str:
    rows = _g(p60, "c7_split_sensitivity", "rows")
    legacy = _g(p60, "c7_split_sensitivity", "legacy_split")
    L = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Split date & Shift & Earlier window & Later window & Days (earlier/later) \\",
        r"\midrule",
    ]
    for r in rows:
        sh = r["shift_months"]
        shift = "$0$" if sh == 0 else f"${sh:+d}$ mo"
        pre = f"{r['pre_yen_per_mw_yr'] / 1000:+,.0f}"
        post = f"{r['post_yen_per_mw_yr'] / 1000:+,.0f}"
        days = f"{r['n_days_pre']} / {r['n_days_post']}"
        if r["split_day"] == legacy:
            L.append(
                f"\\textbf{{{r['split_day']}}} & \\textbf{{{shift}}} & "
                f"$\\mathbf{{{pre}}}$ & $\\mathbf{{{post}}}$ & {days} \\\\"
            )
        else:
            L.append(f"{r['split_day']} & {shift} & ${pre}$ & ${post}$ & {days} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def nomination_band(p61: dict) -> str:
    r1 = _g(p61, "nomination_band", "families", "R1_information_responsive", "rows")
    r3 = {
        r["eps"]: r
        for r in _g(p61, "nomination_band", "families", "R3_tolerance_band_settlement", "rows")
    }
    sig = _g(p61, "findings", "max_eps_with_significantly_negative_cnpe")
    L = [
        r"\begin{tabular}{lrlrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{R1 information-responsive band} & R3 tolerance band \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
        r"$\varepsilon$ & CNPE (k JPY) & 95\% CI & $\Delta|q-G|$ (MWh) "
        r"& CNPE (k JPY) \\",
        r"\midrule",
    ]
    for r in r1:
        eps = r["eps"]
        c = r["cnpe_pre"]
        lo, hi = c.get("ci95") or [None, None]
        cell = f"{c['annualised'] / 1000:+,.0f}"
        ci = f"$[{lo / 1000:+,.0f}, {hi / 1000:+,.0f}]$" if lo is not None else "---"
        dabs = f"{r['d_abs_imbalance_pre_mwh']:+,.0f}"
        t3 = r3.get(eps)
        c3 = f"{t3['cnpe_pre']['annualised'] / 1000:+,.0f}" if t3 else "---"
        bold = abs(eps - sig) < 1e-9 if sig is not None else False
        fmt = (lambda v: f"$\\mathbf{{{v}}}$") if bold else (lambda v: f"${v}$")
        L.append(f"${eps:g}$ & {fmt(cell)} & {ci} & ${dabs}$ & ${c3}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


CW_PAIRS = (
    ("msm_d2_12z", "msm_d2_12z_cw", "$D{-}2$ 12\\,UTC run (headline)"),
    ("msm_gate_composite", "msm_gate_composite_cw", "Freshest admissible composite"),
    ("msm_d1_00z", "msm_d1_00z_cw", "$D{-}1$ 00\\,UTC run (post-gate)"),
)


def cw_ladder(p74: dict, p74g: dict) -> str:
    rows = []
    for sp, cw, label in CW_PAIRS:
        a, b = p74["tiers"][sp], p74["tiers"][cw]
        rows.append(
            f"{label} & {a['improvement']:.4f} & {b['improvement']:.4f} & "
            f"$[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]$ & "
            f"$+{b['improvement'] - a['improvement']:.4f}$ \\\\"
        )
    g36 = p74["gate36_gate_timing_cw"]
    mi = p74["measurement_interaction"]
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Tier & Single-point & Capacity-weighted & 95\\% CI (CW) & Uplift \\\\\n"
        "\\midrule\n" + body + "\n\\midrule\n"
        "\\multicolumn{5}{l}{\\emph{Gate timing under capacity-weighted measurement}} \\\\\n"
        f"Post-gate minus headline pre-gate & & {g36['delta']:.4f} & "
        f"$[{g36['ci95'][0]:.4f}, {g36['ci95'][1]:.4f}]$ & "
        f"$p = {g36['dm_l3']['p_hln']:.2f}$ \\\\\n"
        f"One-tick equivalence (TOST, $L{{=}}3$) & & & & "
        f"$p = {g36['tost_one_tick']['l3']:.3f}$ \\\\\n"
        f"Measurement $\\times$ lead interaction & & {mi['gate_step_cw_minus_single']:.4f} & "
        f"$[{mi['ci95'][0]:.4f}, {mi['ci95'][1]:.4f}]$ & "
        f"$p = {mi['dm_l3']['p_hln']:.2f}$ \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
        + f"% Gate 5 max|pred diff| = {p74['gate5']['max_abs_pred_diff']} over "
        + f"{p74['gate5']['n_rows_joined']} slots; Gate 35 CW wins "
        + f"{p74g['cw_wins']}/{p74g['n_areas']}\n"
    )


CYCLE_LABELS = {
    "msm_d3_12z": "$D{-}3$ 12\\,UTC",
    "msm_d2_12z": "$D{-}2$ 12\\,UTC (headline)",
    "msm_d2_18z": "$D{-}2$ 18\\,UTC",
    "msm_d2_21z": "$D{-}2$ 21\\,UTC (freshest admissible)",
    "msm_d1_00z": "$D{-}1$ 00\\,UTC (post-gate)",
}


def cycle_curve(p73: dict) -> str:
    rows = []
    for key in ("msm_d3_12z", "msm_d2_12z", "msm_d2_18z", "msm_d2_21z", "msm_d1_00z"):
        r = p73["rungs"][key]
        ci = r["ci95"]
        margin = r["gate_margin_hours_measured"]
        rows.append(
            f"{CYCLE_LABELS[key]} & ${margin:+.1f}$ & {r['improvement']:.4f} & "
            f"$[{ci[0]:.4f}, {ci[1]:.4f}]$ & {_p(r['dm_l3']['p_hln'])} \\\\"
        )
    n = p73["panel"]["n_days"]
    g41 = p73["gate41_slope"]
    return (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Cycle & Gate margin (h) & Improvement & 95\\% CI & DM $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        + f"% {n} common-window days; Gate 41 slope {g41['slope_jpykwh_per_gate_margin_hour']} "
        + f"{g41['ci95']} landed on branch {g41['branch']}\n"
    )


RULE_LABELS = {
    "r_mean": ("Conditional mean (the paper's rule)", "$\\hat g(1-\\hat c)$"),
    "r_median": ("Conditional median", "$+\\,\\hat Q_{0.5}$ of expanding residuals"),
    "r_q30": ("Conservative quantile", "$+\\,\\hat Q_{0.3}$"),
    "r_q70": ("Aggressive quantile", "$+\\,\\hat Q_{0.7}$"),
    "r_margin5": ("5\\% operating margin", "$0.95\\,\\hat g(1-\\hat c)$"),
}


def nomination_rules(p76: dict) -> str:
    rows = []
    for key in ("r_mean", "r_median", "r_q30", "r_q70", "r_margin5"):
        r = p76["rules"][key]
        label, form = RULE_LABELS[key]
        ci = r["cnpe_ci95"]
        rows.append(
            f"{label} & {form} & ${r['cnpe_yen_per_mw_yr'] / 1000.0:,.0f}$ & "
            f"$[{ci[0] / 1000.0:,.0f}, {ci[1] / 1000.0:,.0f}]$ & "
            f"{_p(r['cnpe_dm_l3']['p_hln'])} & "
            f"{r['gross_deviation_reduction_mwh_per_mw_yr']:.1f} \\\\"
        )
    body = "\n".join(rows)
    n = p76["panel"]["n_days"]
    return (
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Rule & Nomination & CNPE & 95\\% CI & DM $p$ & Gross red. \\\\\n"
        " &  & \\multicolumn{2}{c}{(thousand JPY per MW-yr)} & ($L{=}3$) & "
        "(MWh per MW-yr) \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
        + f"% identical {n}-day reduced panel for every rule\n"
    )


# Row and column wording for the three main-text tables, in both manuscript
# languages. Only the WORDS live here; every number comes from provenance, so
# the two languages cannot print different values for the same cell.
L10N = {
    "en": {
        "ladder_head": ("Weather information set", "Available at the gate?",
                        "Improv.\\ (\\jpykwh)", "95\\% CI", "Pooled DM $p$",
                        "Areas"),
        "ladder_blocks": ("Run-pinned: one identified initialisation per value",
                          "Lead-indexed composites: assembled across "
                          "initialisations", "References"),
        "ladder_rows": {
            "msm_d2_12z_cw": ("$D{-}2$ 12\\,UTC run (leads 27--50\\,h)",
                              "\\textbf{Yes}, $+10$\\,h\\,30\\,m"),
            "msm_gate_composite_cw": ("Freshest admissible run per slot",
                                      "\\textbf{Yes}, $+1$\\,h\\,30\\,m"),
            "msm_d1_00z_cw": ("$D{-}1$ 00\\,UTC run (leads 15--38\\,h)",
                              "\\textbf{No}, $1$\\,h\\,30\\,m late"),
            "prev2": ("Previous-day-2 series ($\\approx48$\\,h lead, SP)",
                      "Yes by dissemination"),
            "prev1": ("Previous-day-1 series ($\\approx24$\\,h lead, SP)",
                      "No, post-gate class"),
            "histfc": ("Operational reconstruction (SP)",
                       "No, issuance ambiguous"),
            "era5_cw": ("Stale observations (reanalysis, lag-48 slots)",
                        "\\textbf{No}, publishes late"),
            "oracle_cw": ("Ex-post realised weather --- \\emph{not a ceiling}",
                          "\\textbf{No}, never ex ante"),
        },
        "abs_head": ("Weather information set", "MAE", "RMSE", "rMAE", "rRMSE",
                     "MAE, midday", "MAE, floor", "vs naive"),
        "abs_rows": {
            "naive168": "Naive 168-hour seasonal persistence",
            "price_only": "Price-only baseline (no weather)",
            "era5_cw": "Stale observations",
            "msm_d2_12z_cw": "Pre-gate run",
            "msm_gate_composite_cw": "Freshest admissible per slot",
            "msm_d1_00z_cw": "First excluded run",
            "oracle_cw": "Realised weather (not available ex ante)",
        },
        "margin_head": ("Equivalence margin", "\\% of",
                        "$p_{\\mathrm{TOST}}$ by dependence scheme",
                        "(\\jpykwh)", "baseline"),
        "margin_schemes": ("7~d", "14~d", "28~d", "PW", "month"),
        "margin_rows": {
            "abs_0.005": "$\\pm 0.005$",
            "abs_0.01": "$\\pm 0.01$ (one JEPX tick, primary)",
            "abs_0.02": "$\\pm 0.02$",
            "abs_0.05": "$\\pm 0.05$",
            "rel_10pct": "$\\pm 10\\%$ of the pre-gate improvement",
            "rel_20pct": "$\\pm 20\\%$ of the pre-gate improvement",
        },
        "margin_foot": ("Point estimate {pt}~\\jpykwh{{}} over {n}~days; "
                        "baseline MAE {base}~\\jpykwh{{}} on all {n} days; "
                        "sign-flip permutation $p = {perm}$"),
    },
    "ja": {
        "ladder_head": ("気象情報集合", "ゲートクローズ時点で利用可能か",
                        "改善 (\\jpykwh)", "95\\% CI", "プール DM $p$",
                        "エリア数"),
        "ladder_blocks": ("run 固定: 値ごとに 1 つの同定された初期時刻",
                          "リードタイム索引付きコンポジット: 複数の初期時刻に"
                          "またがって組み立てたもの", "参照"),
        "ladder_rows": {
            "msm_d2_12z_cw": ("$D{-}2$ 12\\,UTC run (リード 27--50\\,h)",
                              "\\textbf{Yes}、$+10$\\,h\\,30\\,m"),
            "msm_gate_composite_cw": ("スロットごとに最も新しい適格 run",
                                      "\\textbf{Yes}、$+1$\\,h\\,30\\,m"),
            "msm_d1_00z_cw": ("$D{-}1$ 00\\,UTC run (リード 15--38\\,h)",
                              "\\textbf{No}、$1$\\,h\\,30\\,m 遅い"),
            "prev2": ("前日-2 系列 (リード $\\approx48$\\,h、SP)",
                      "配信の観点では Yes"),
            "prev1": ("前日-1 系列 (リード $\\approx24$\\,h、SP)",
                      "No、締切後クラス"),
            "histfc": ("実運用予報の再構成 (SP)", "No、発行時刻が曖昧"),
            "era5_cw": ("陳腐化観測 (再解析、lag-48 スロット)",
                        "\\textbf{No}、公開が遅い"),
            "oracle_cw": ("事後の実現気象 —— \\emph{天井ではない}",
                          "\\textbf{No}、事前には決して不可"),
        },
        "abs_head": ("気象情報集合", "MAE", "RMSE", "rMAE", "rRMSE",
                     "MAE (日中)", "MAE (床)", "naive 比"),
        "abs_rows": {
            "naive168": "naive 168 時間季節持続",
            "price_only": "price-only ベースライン (気象なし)",
            "era5_cw": "陳腐化観測",
            "msm_d2_12z_cw": "締切前 run",
            "msm_gate_composite_cw": "スロットごとに最も新しい適格 run",
            "msm_d1_00z_cw": "最初の排除された run",
            "oracle_cw": "実現気象 (事前には利用不可)",
        },
        "margin_head": ("等価性マージン", "ベースライン",
                        "依存構造の扱いごとの $p_{\\mathrm{TOST}}$",
                        "(\\jpykwh)", "に対する \\%"),
        "margin_schemes": ("7~d", "14~d", "28~d", "PW", "月"),
        "margin_rows": {
            "abs_0.005": "$\\pm 0.005$",
            "abs_0.01": "$\\pm 0.01$ (JEPX の 1 ティック、主要)",
            "abs_0.02": "$\\pm 0.02$",
            "abs_0.05": "$\\pm 0.05$",
            "rel_10pct": "締切前改善の $\\pm 10\\%$",
            "rel_20pct": "締切前改善の $\\pm 20\\%$",
        },
        "margin_foot": ("点推定 {pt}~\\jpykwh{{}} ({n} 日); "
                        "ベースライン MAE {base}~\\jpykwh{{}} (全 {n} 日); "
                        "符号反転置換検定 $p = {perm}$"),
    },
}


# The main-article ladder table. Labels and availability strings are prose and
# live here; every NUMBER comes from provenance. It was written with literal
# numbers until 2026-08-13, and five 95% intervals in it had gone unguarded --
# the macro ratchet only sees \newcommand, so a bare "[0.210, 0.303]" in a
# table body was invisible to it.
LADDER_LAYOUT = [
    (0, "arm_cw_primary",
     [("msm_d2_12z_cw", True), ("msm_gate_composite_cw", False),
      ("msm_d1_00z_cw", False)]),
    (1, "arm_sp_robustness",
     [("prev2", False), ("prev1", False), ("histfc", False)]),
    (2, "arm_cw_primary", [("era5_cw", False), ("oracle_cw", False)]),
]


def _dm_p(p3: float, p9: float) -> str:
    """Render the WEAKER of the two bandwidths, so the cell cannot flatter."""
    worst = max(p3, p9)
    for thr, txt in ((1e-6, "$< 10^{-6}$"), (1e-5, "$< 10^{-5}$"),
                     (1e-4, "$< 10^{-4}$"), (1e-3, "$< 10^{-3}$")):
        if worst < thr:
            return txt
    return f"${worst:.3f}$"


def vintage_ladder(p85: dict, lang: str = "en") -> str:
    """The main-article ladder table.

    Written with literal numbers until 2026-08-13, and five 95% intervals in the
    English version and six in the Japanese one had gone unguarded: the macro
    ratchet only sees \\newcommand, so a bare "[0.210, 0.303]" in a table body
    was invisible to it. Both languages now read the same provenance.
    """
    t = L10N[lang]
    rows = ["\\toprule", " & ".join(t["ladder_head"]) + " \\\\", "\\midrule"]
    for bi, arm, tiers in LADDER_LAYOUT:
        if bi:
            rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{t['ladder_blocks'][bi]}}}}} "
                    "\\\\")
        for key, bold in tiers:
            r = _g(p85, arm, "tiers", key)
            label, avail = t["ladder_rows"][key]
            imp = f"{r['improvement']:.3f}"
            imp = f"$\\mathbf{{{imp}}}$" if bold else f"${imp}$"
            rows.append(
                f"{label} & {avail} & {imp} & "
                f"$[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]$ & "
                f"{_dm_p(r['dm_l3']['p_hln'], r['dm_l9']['p_hln'])} & "
                f"{r['areas_sig_l3']}/9 \\\\")
    rows.append("\\bottomrule")
    return ("\\begin{tabular}{llrllc}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


ABS_ORDER = [("naive168", False), ("price_only", False), ("era5_cw", False),
             ("msm_d2_12z_cw", True), ("msm_gate_composite_cw", False),
             ("msm_d1_00z_cw", False), ("oracle_cw", False)]


def absolute_performance(p82b: dict, lang: str = "en") -> str:
    """Absolute MAE and RMSE, and skill against the EPF-standard denominator.

    Every cell is read from the ``relative`` block, which is computed on the
    sub-panel where the naive benchmark exists. Reading the levels from the full
    panel and the ratios from the sub-panel would put a numerator and a
    denominator from different samples in the same row.
    """
    t = L10N[lang]
    h = t["abs_head"]

    def cell(slice_: str, key: str, field: str):
        blk = _g(p82b, "relative", slice_)
        src = blk["naive168"] if key == "naive168" else blk["tiers"][key]
        return src.get(field)

    rows = ["\\toprule",
            f"{h[0]} & {h[1]} & {h[2]} & {h[3]} & {h[4]} & {h[5]} & {h[6]} \\\\",
            f"& \\multicolumn{{2}}{{c}}{{(\\jpykwh)}} & "
            f"\\multicolumn{{2}}{{c}}{{{h[7]}}} & "
            f"\\multicolumn{{2}}{{c}}{{(\\jpykwh)}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"]
    for key, bold in ABS_ORDER:
        def _b(x, nd=3, *, _bold=bold):
            if x is None:
                return "---"
            txt = f"{x:.{nd}f}"
            return f"$\\mathbf{{{txt}}}$" if _bold else f"${txt}$"
        lab = t["abs_rows"][key]
        lab = f"\\textbf{{{lab}}}" if bold else lab
        rmae = 1.0 if key == "naive168" else cell("all", key, "rmae_vs_naive168")
        rrmse = 1.0 if key == "naive168" else cell("all", key, "rrmse_vs_naive168")
        rows.append(
            f"{lab} & {_b(cell('all', key, 'mae'))} & "
            f"{_b(cell('all', key, 'rmse'))} & {_b(rmae)} & {_b(rrmse)} & "
            f"{_b(cell('midday', key, 'mae'))} & "
            f"{_b(cell('floor', key, 'mae'))} \\\\")
    rows.append("\\bottomrule")
    return ("\\begin{tabular}{@{}lrrrrrr@{}}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


MARGIN_ORDER = [("abs_0.005", False), ("abs_0.01", True), ("abs_0.02", False),
                ("abs_0.05", False), ("rel_10pct", False), ("rel_20pct", False)]
MARGIN_SCHEME_KEYS = ("circular_b7", "circular_b14", "circular_b28",
                      "stationary_pw", "month_cluster")


def equivalence_margins(p83: dict, p82: dict, lang: str = "en") -> str:
    """The margin curve: at what margin does the gate-timing null become an
    equivalence claim, and does the answer depend on the dependence scheme?

    Reported in the main text because the one-tick pass is the closest call in
    the paper: at half a tick it does not pass, and a reader is entitled to see
    where the boundary is rather than only the side of it we landed on.
    """
    t = L10N[lang]
    h = t["margin_head"]
    base = _g(p82, "relative", "msm_d2_12z_cw", "baseline_mae")
    rows = [
        "\\toprule",
        f"{h[0]} & {h[1]} & \\multicolumn{{5}}{{c}}{{{h[2]}}} \\\\",
        "\\cmidrule(lr){3-7}",
        f"{h[3]} & {h[4]} & " + " & ".join(t["margin_schemes"]) + " \\\\",
        "\\midrule",
    ]
    for key, primary in MARGIN_ORDER:
        cells = []
        for skey in MARGIN_SCHEME_KEYS:
            m = _g(p83, "schemes", skey, "margins", key)
            txt = "$< 10^{-3}$" if m["p_tost"] < 1e-3 else f"${m['p_tost']:.3f}$"
            cells.append(f"\\textbf{{{txt}}}" if primary else txt)
        delta = _g(p83, "schemes", "circular_b7", "margins", key)["delta_margin"]
        lab = t["margin_rows"][key]
        lab = f"\\textbf{{{lab}}}" if primary else lab
        rows.append(f"{lab} & ${100 * delta / base:.2f}$ & "
                    + " & ".join(cells) + " \\\\")
    foot = t["margin_foot"].format(
        pt=f"{_g(p83, 'point'):+.5f}", n=_g(p83, "n_days"), base=f"{base:.3f}",
        perm=f"{_g(p83, 'permutation_signflip', 'p'):.3f}")
    rows += ["\\midrule",
             f"\\multicolumn{{7}}{{@{{}}l@{{}}}}{{\\emph{{{foot}}}}} \\\\",
             "\\bottomrule"]
    return ("\\begin{tabular}{@{}lrrrrrr@{}}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


def enriched_nwp(p54v2: dict) -> str:
    """The four-cell enriched-baseline decomposition, per area.

    Written with literal numbers until 2026-08-13, and every cell in it came from
    the SUPERSEDED phase54 run (610 days, 234,080 slots) while the surrounding
    macros were bound to phase54v2 (698 days, 265,872 slots). The pooled row was
    the visible symptom: 1.896 - 1.723 = 0.173, but the same row printed
    \\PreGateOverEnrichedAdm = 0.200, so a reader subtracting the printed levels
    got a different answer from the printed delta. The macro ratchet cannot see
    table bodies, which is exactly how it survived two revisions.
    """
    r = _g(p54v2, "results")
    pa, dl = r["per_area"], r["deltas_daily_mean_with_ci95"]
    pooled = r["pooled_mae_daily_mean"]
    rows = [
        "\\toprule",
        " & \\multicolumn{4}{c}{Test MAE (\\jpykwh)} & "
        "\\multicolumn{3}{c}{Nested differences (\\jpykwh)} & \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
        "Area & (1) price-only & (2) enriched & (3) $+$proxies & (4) $+$raw & "
        "$\\Delta_{3-2}$ & $\\Delta_{4-3}$ & $\\Delta_{4-2}$ & DM $p$ \\\\",
        "\\midrule",
    ]
    for area in AREA_LABELS:
        a = pa.get(area)
        if a is None:  # Chubu is not in the OCCTO-covered eight
            continue
        rows.append(
            f"{AREA_LABELS[area]} & {_f(a['mae_price_only'], 3)} & "
            f"{_f(a['mae_enriched_adm'], 3)} & {_f(a['mae_enriched_proxies'], 3)} & "
            f"{_f(a['mae_enriched_proxies_raw'], 3)} & {_f(a['d32'], 3)} & "
            f"{_f(a['d43'], 3, signed=True)} & {_f(a['d42'], 3)} & "
            f"{_p(a['dm_p_l3_d42'])} \\\\")
    rows += [
        "\\midrule",
        f"Pooled & {_f(pooled['price_only'], 3)} & "
        f"{_f(pooled['enriched_adm'], 3)} & {_f(pooled['enriched_proxies'], 3)} & "
        f"{_f(pooled['enriched_proxies_raw'], 3)} & "
        f"{_f(dl['d32_proxies_over_enriched']['point_daily_mean'], 3)} & "
        f"{_f(dl['d43_raw_over_proxies']['point_daily_mean'], 3, signed=True)} & "
        f"{_f(dl['d42_total_prev2_over_enriched']['point_daily_mean'], 3)} & "
        "--- \\\\",
        "\\bottomrule",
    ]
    return ("\\begin{tabular}{lrrrrrrrr}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


AUDIT_ROWS = [
    ("gate_time_stated", "States the market's gate-closure time", "LitGateTimePct"),
    ("model_run_id_stated", "Identifies the model run or cycle used", "LitRunIdPct"),
    ("init_time_stated", "States the initialisation time of that run", "LitInitPct"),
    ("publication_time_stated",
     "States when the input was published or disseminated", "LitPubTimePct"),
    ("gate_admissibility_checked",
     "Verifies the input was available before the gate", "LitAdmissiblePct"),
    ("revision_handling", "Handles input revisions or vintages", "LitRevisionPct"),
]
VERDICTS = ("Yes", "Partial", "No", "Not stated")


def lit_audit_counts(p88: dict) -> str:
    """The audit's per-column verdict counts.

    Generated from 2026-08-13. All 24 counts were hand-typed, and two rows were
    wrong: model_run_id_stated printed No 28 / Not stated 9 against a provenance
    27 / 10, and init_time_stated printed 17 / 20 against 16 / 21. Each wrong row
    still summed to 43, so no arithmetic check caught it, and the Yes counts --
    every rate the paper cites -- were unaffected. But the paper invites a reader
    to re-check any cell against the released dataset, and a reader who did would
    have found the table disagreeing with it.
    """
    pcv = _g(p88, "per_column_verdicts")
    rows = ["\\toprule",
            "Question asked of the published text & " + " & ".join(VERDICTS)
            + " & Yes \\% \\\\",
            "\\midrule"]
    for key, label, pct in AUDIT_ROWS:
        counts = pcv[key]
        # An absent verdict means zero: gate_admissibility_checked has no
        # "Not stated" and revision_handling has no "Yes".
        cells = " & ".join(f"{counts.get(v, 0):2d}" for v in VERDICTS)
        rows.append(f"{label:<50s} & {cells} & \\{pct}\\% \\\\")
    rows.append("\\bottomrule")
    return ("\\begin{tabular}{lrrrrr}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


WEIGHT_ARMS = [
    ("2024-09", "Register snapshot 2024-09 (the paper's weights)"),
    ("2025-12", "Register snapshot 2025-12"),
    ("equal", "Equal weight per point (register discarded)"),
]


def weight_sensitivity(p91: dict) -> str:
    """Three weightings, three cells, one fold schedule."""
    arms = p91["arms"]
    g60 = _g(p91, "gates", "gate60_harness_reproduces_phase85")
    rows = ["\\toprule",
            "Point weighting & Headline & 95\\% CI & Excluded & "
            "Gate diff. & 95\\% CI & $p_{\\mathrm{TOST}}$ \\\\",
            "& \\multicolumn{2}{c}{(\\jpykwh)} & (\\jpykwh) & "
            "\\multicolumn{2}{c}{(\\jpykwh)} & at one tick \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}"
            "\\cmidrule(lr){7-7}"]
    for key, label in WEIGHT_ARMS:
        a = arms.get(key)
        if a is None:
            continue
        h = a["tiers"]["msm_d2_12z_cw"]
        pg = a["tiers"]["msm_d1_00z_cw"]
        gt = a["gate_timing_delta"]
        eq = a.get("equivalence_at_tick", {})
        ptxt = ("---" if not eq else
                (f"\\textbf{{{eq['p_tost']:.3f}}}" if not eq["equivalent"]
                 else f"{eq['p_tost']:.3f}"))
        rows.append(
            f"{label} & {_f(h['improvement'], 4)} & "
            f"$[{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}]$ & "
            f"{_f(pg['improvement'], 4)} & {_f(gt['delta'], 4, signed=True)} & "
            f"$[{gt['ci95'][0]:.4f}, {gt['ci95'][1]:.4f}]$ & {ptxt} \\\\")
    rows += [
        "\\midrule",
        f"\\multicolumn{{7}}{{@{{}}l@{{}}}}{{\\emph{{Reproduction check: the "
        f"{WEIGHT_ARMS[0][0]} arm matches the published ladder to "
        f"{max(g60['abs_differences'].values()):.4f} on the headline, the "
        f"excluded run and the gate difference}}}} \\\\",
        "\\bottomrule",
    ]
    return ("\\begin{tabular}{@{}lrlrrlr@{}}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


PHYS_TIER_LABELS = [
    ("era5_cw_stale", "Stale observations (day-lagged reanalysis)"),
    ("msm_d2_12z_cw", "Pre-gate run (D$-$2 12~UTC)"),
    ("msm_gate_composite_cw", "Freshest admissible per slot"),
    ("msm_d1_00z_cw", "First excluded run (D$-$1 00~UTC)"),
    ("oracle_cw", "Realised weather (not available ex ante)"),
]

PHYS_TARGET_LABELS = [
    ("potential", "Generation with curtailed energy added back (primary)"),
    ("curtail_excluded", "Curtailed slots excluded"),
    ("recorded", "Raw recorded generation"),
]


def physical_tiers(p89: dict) -> str:
    """Per-tier physical-target error, and what each design choice was worth.

    Columns 2-3 are the two panels; columns 4-5 are the two superseded
    conventions, kept so a reader can price the corrections rather than take
    them on trust.
    """
    t = _g(p89, "targets", "potential", "tiers")
    g51 = _g(p89, "gates", "gate51_physical_gain_crossfit_common")
    g51w = _g(p89, "gates", "gate51_sensitivity_wide_panel")
    g52 = _g(p89, "gates", "gate52_curtailment_sign_stability")
    n_common, n_wide = g51["n_days"], g51w["n_days"]
    rows = [
        "\\toprule",
        "Weather vintage & \\multicolumn{2}{c}{Cross-fitted scale} & "
        "\\multicolumn{2}{c}{Superseded conventions} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "& common & physical-only & full-sample & uniform \\\\",
        f"& ({n_common}~d) & ({n_wide}~d) & scale & $-$30~min stamp \\\\",
        "\\midrule",
    ]
    for key, label in PHYS_TIER_LABELS:
        r, leg = t[key], t.get(f"{key}__legacyshift", {})
        rows.append(
            f"{label} & {_f(r['pooled_mae_cf_common_mw'], 1)} & "
            f"{_f(r['pooled_mae_cf_mw'], 1)} & {_f(r['pooled_mae_is_mw'], 1)} & "
            f"{_f(leg.get('pooled_mae_cf_mw'), 1)} \\\\")
    rows += [
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Excluded run minus pre-gate run on the common "
        "panel, as a difference of pooled MAE}} \\\\",
    ]
    for key, label in PHYS_TARGET_LABELS:
        d = g52["delta_by_target"].get(key)
        rows.append(f"{label} & \\multicolumn{{4}}{{l}}{{{_f(d, 1)}~MW}} \\\\")
    # The headline gain is the MEAN OF DAILY differences with a block bootstrap
    # over days, which is not the same estimator as the difference of pooled
    # MAEs above; both are reported so neither has to stand in for the other.
    rows += [
        "\\addlinespace[2pt]",
        f"\\quad primary target, mean of daily differences & "
        f"\\multicolumn{{4}}{{l}}{{{_f(g51['delta_mw'], 1)}~MW "
        f"$[{_hu(g51['ci95'][0])}, {_hu(g51['ci95'][1])}]$}} \\\\",
    ]
    rows.append("\\bottomrule")
    return ("\\begin{tabular}{@{}lrrrr@{}}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


MECH_ROW_LABELS = [
    ("physical_residual_on_innovation", "Physical residual", False),
    ("price_residual_on_innovation", "Price residual", True),
    ("placebo_physical", "\\quad placebo, physical (must be null)", False),
    ("placebo_price", "\\quad placebo, price (must be null)", True),
]

MECH_SPLIT_LABELS = [
    ("floor", "Floor slots"),
    ("non_floor", "Non-floor slots"),
    ("curtailed", "Slots with recorded curtailment"),
    ("not_curtailed", "Slots without"),
]


P97 = RUN / "phase97_cost_loss.json"
P80 = RUN / "phase80_jma_version.json"
P78 = RUN / "phase78_lineage_audit.json"
P86 = RUN / "phase86_gfs_tier.json"
COST_LOSS_ROWS = ("0.05", "0.1", "0.2", "0.3", "0.42", "0.5", "0.64", "0.8", "0.9")


def cost_loss(p97: dict) -> str:
    r"""The floor forecasts' Murphy value score, with and without pre-gate weather."""
    m2 = p97["models"]["m2"]["value_curve"]
    m2p = p97["models"]["m2p"]["value_curve"]
    diff = p97["weather_increment_in_value"]["difference"]
    ci = p97["weather_increment_in_value"]["ci95"]
    cut = p97["models"]["m2p"]["positive_value_alpha_range"][1]
    rows = []
    for a in COST_LOSS_ROWS:
        if a not in m2 or m2[a] is None:
            continue
        lo, hi = ci.get(a, [None, None])
        star = "" if lo is None or lo > 0 else r"\phantom{*}"
        rows.append(
            f"${float(a):.2f}$ & ${m2[a]:+.3f}$ & ${m2p[a]:+.3f}$ & "
            f"${diff[a]:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]${star} \\\\")
    body = "\n".join(rows)
    return (
        "% GENERATED by " + Path(__file__).name + " from phase97_cost_loss.json.\n"
        "% Do not edit by hand; regenerate.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Decision value of the floor-event forecasts, in Murphy's "
        "cost--loss frame \\citep{murphy1977costloss}. An actor protects when the "
        "forecast probability exceeds $\\alpha = C/L$, so $\\alpha$ indexes the "
        "actor: cheap protection is a small $\\alpha$. The value score is $1$ for a "
        "perfect forecast and $0$ for one no better than always or never "
        "protecting. Both columns are the same classifier on the same folds, with "
        "and without the pre-gate weather channels; the difference is the decision "
        "value of pre-gate weather, with 95\\% day-block bootstrap intervals. "
        "Without weather the forecast stops beating climatology above "
        f"$\\alpha = {cut:.2f}$; with it every ratio in the grid gains.}}\n"
        "\\label{tab:cost_loss}\n\\small\n"
        "\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$C/L$ & with weather & without & difference & 95\\% CI \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )

VERSION_TIER_ROWS = [
    ("msm_d2_12z_cw", "Headline pre-gate tier"),
    ("msm_gate_composite_cw", "Freshest admissible per slot"),
    ("msm_d1_00z_cw", "First post-gate run"),
    ("era5_cw", "Stale-observation proxy"),
    ("oracle_cw", "Ex-post realised weather"),
]
BUFFER_RUN_ROWS = [
    ("msm_d2_12z", "$D{-}2$ 12\\,UTC (headline pre-gate)"),
    ("msm_d2_21z", "$D{-}2$ 21\\,UTC (freshest pre-gate)"),
    ("msm_d2_18z", "$D{-}2$ 18\\,UTC"),
    ("msm_d3_12z", "$D{-}3$ 12\\,UTC"),
    ("msm_d1_00z", "$D{-}1$ 00\\,UTC (excluded by the gate)"),
]
GFS_TIER_ROWS = [
    ("gfs_d2_12z", "GFS $D{-}2$ 12\\,UTC"),
    ("gfs_d2_18z", "GFS $D{-}2$ 18\\,UTC (freshest pre-gate)"),
    ("msm_d2_12z", "JMA-MSM $D{-}2$ 12\\,UTC"),
    ("msm_gate_composite", "JMA-MSM freshest admissible per slot"),
]

def version_split(p80: dict) -> str:
    r"""The 2025-05-20 MSM physics upgrade, tier by tier and for the gate-timing null.

    Written because the body cited \suppref{S4} for this and S4 contained nothing
    about the upgrade -- a pointer that a referee following it would find empty.
    """
    rows = []
    for slug, label in VERSION_TIER_ROWS:
        t = p80["tiers"].get(slug)
        if t is None:
            continue
        v1, v2, d = t["v1_pre_upgrade"], t["v2_post_upgrade"], t["difference"]
        rows.append(
            f"{label} & ${v1['mean']:+.3f}$ & ${v2['mean']:+.3f}$ & "
            f"${d['delta_v2_minus_v1']:+.4f}$ & "
            f"$[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]$ & "
            f"{_p(d['p_boot'])} \\\\")
    gt = p80["gate_timing_by_version"]
    for key, label in (("exact", "Gate-timing difference, all days"),
                       ("excluding_pm14d", r"\quad excluding $\pm 14$\,d of the upgrade")):
        g = gt[key]
        i = g["interaction"]
        rows.append(
            f"\\emph{{{label}}} & ${g['v1']['mean']:+.4f}$ & ${g['v2']['mean']:+.4f}$ & "
            f"${i['delta_v2_minus_v1']:+.4f}$ & "
            f"$[{i['ci95'][0]:+.4f}, {i['ci95'][1]:+.4f}]$ & {_p(i['p_boot'])} \\\\")
    lag = p80["dissemination_lag_by_version"]
    n1 = p80["tiers"]["msm_d2_12z_cw"]["v1_pre_upgrade"]["n_days"]
    n2 = p80["tiers"]["msm_d2_12z_cw"]["v2_post_upgrade"]["n_days"]
    return (
        "% GENERATED by " + Path(__file__).name + " from phase80_jma_version.json.\n"
        "% Do not edit by hand; regenerate.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{The 2025-05-20 MSM physics upgrade, split at the upgrade date. "
        f"Pre-upgrade covers {n1} test days and post-upgrade {n2}. The first rows "
        "give each tier's improvement over the price-only baseline on either side "
        "of the boundary; the italic rows give the gate-timing difference, which is "
        "the quantity the paper's second claim rests on, with and without a "
        "fourteen-day transition window around the upgrade. Every interaction "
        "interval covers zero, so the pooled contrasts are not artefacts of "
        "averaging across two model versions. Measured dissemination lag is "
        f"unchanged across the boundary (median {lag['v1']['median']:.1f} against "
        f"{lag['v2']['median']:.1f} minutes over {lag['v1']['n_files']:,} and "
        f"{lag['v2']['n_files']:,} files), so the split does not confound accuracy "
        "with timing. Intervals are 95\\% day-block bootstrap.}\n"
        "\\label{tab:supp_version_split}\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Quantity & pre-upgrade & post-upgrade & difference & 95\\% CI & $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    ).replace("{:,}", "")


def buffer_audit(p78: dict) -> str:
    r"""Admissibility of every run under processing buffers added to measured arrival.

    Also written because the body pointed at \suppref{S1} for it and S1 had none.
    """
    buffers = p78["buffers_min"]
    rows = []
    for slug, label in BUFFER_RUN_ROWS:
        r = p78["run_dissemination"].get(slug)
        if r is None:
            continue
        cells = []
        for b in buffers:
            frac = r["admissible_under_buffer_min"].get(str(b))
            cells.append("--" if frac is None
                         else (r"\checkmark" if frac >= 1.0
                               else (r"\times" if frac <= 0.0 else f"{frac*100:.0f}\\%")))
        cells = [c if c.startswith("\\") is False else f"${c}$" for c in cells]
        rows.append(
            f"{label} & ${r['gate_minus_init_min']}$ & "
            f"${r['worst_lag_min']:.2f}$ & ${r['worst_margin_min']:+.2f}$ & "
            + " & ".join(cells) + " \\\\")
    hdr = " & ".join(f"$+{b}$" for b in buffers)
    return (
        "% GENERATED by " + Path(__file__).name + " from phase78_lineage_audit.json.\n"
        "% Do not edit by hand; regenerate.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Admissibility under a processing buffer. Dissemination "
        "completion is not the same instant as a bid entering a trading system, so "
        "every measured arrival is delayed by the buffer in the column head "
        "(minutes) and the gate test re-run on all delivery days. Column three is "
        "the worst observed completion lag after initialisation and column four the "
        "worst margin against the gate, negative where the run arrives too late. "
        "$\\checkmark$ means admissible on every day, $\\times$ on none. The "
        "pre-gate classification survives to a 90-minute buffer; at 120 minutes the "
        "freshest cycle's slots fall back to the $D{-}2$ 12\\,UTC run, which is the "
        "already-estimated headline cell, so the cost of that fallback is measured "
        "rather than modelled. The excluded run is inadmissible at every buffer, "
        "including none. Time admissibility is not sufficiency: the $D{-}2$ 18\\,UTC "
        "and 21\\,UTC runs clear the gate but the mirror truncates them at "
        "forecast hour 39, so neither spans a whole delivery day, which is why "
        "the headline rung is the older 12\\,UTC run.}\n"
        "\\label{tab:supp_buffer_audit}\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrr" + "r" * len(buffers) + "}\n\\toprule\n"
        " & gate $-$ init & worst lag & worst margin & "
        "\\multicolumn{" + str(len(buffers)) + "}{c}{buffer (min)} \\\\\n"
        "Run & (min) & (min) & (min) & " + hdr + " \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def second_provider(p86: dict) -> str:
    r"""The NCEP GFS cells, on the single-point measurement, against the JMA ladder."""
    rows = []
    for slug, label in GFS_TIER_ROWS:
        t = p86["tiers"].get(slug)
        if t is None:
            continue
        rows.append(
            f"{label} & ${t['improvement']:+.4f}$ & "
            f"$[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]$ & {_p(t['p_l3'])} \\\\")
    q = p86["questions"]
    q2, q3 = q["q2_jma_minus_gfs_same_init"], q["q3_within_gfs_vintage_step"]
    extra = [
        r"\midrule",
        f"\\emph{{JMA $-$ GFS at the same initialisation}} & ${q2['delta']:+.4f}$ & "
        f"$[{q2['ci95'][0]:+.4f}, {q2['ci95'][1]:+.4f}]$ & {_p(q2['p_l3'])} \\\\",
        f"\\emph{{Within-GFS vintage step to its freshest pre-gate cycle}} & "
        f"${q3['delta']:+.4f}$ & $[{q3['ci95'][0]:+.4f}, {q3['ci95'][1]:+.4f}]$ & "
        f"{_p(q3['p_l3'])} \\\\",
    ]
    n = p86["panel"]
    return (
        "% GENERATED by " + Path(__file__).name + " from phase86_gfs_tier.json.\n"
        "% Do not edit by hand; regenerate.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{A second, independent forecast provider. The NCEP Global "
        "Forecast System was retrieved from its public archive by byte range, with "
        "dissemination times read from that store's own timestamps, and scored on "
        f"the same panel ({n['n_slots']:,} slot observations over "
        f"{n['n_test_days']} test days), the same folds, the same learner and the "
        "same baseline as the JMA cells. The comparison sits inside the "
        "single-point arm, where all four cells share one measurement; the "
        "capacity-weighted field was deliberately not rebuilt for GFS, because the "
        "question here is external generality rather than a measurement claim. "
        "Three questions were fixed before computation. GFS beats the price-only "
        "baseline, so the result is not specific to one forecast centre. At the "
        "same initialisation the national meso-scale model is better than the "
        "global model at $0.25^\\circ$, which is a provider difference and not a "
        "vintage difference. And the within-GFS step towards the gate is not "
        "distinguishable from zero, so the flatness of the ladder near the gate is "
        "reproduced by a provider whose model, resolution and dissemination "
        "schedule all differ. Intervals are 95\\% day-block bootstrap; $p$ is "
        "Harvey--Leybourne--Newbold Diebold--Mariano at $L=3$.}\n"
        "\\label{tab:supp_second_provider}\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Cell & improvement (JPY/kWh) & 95\\% CI & $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows + extra) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def mechanism(p90: dict) -> str:
    """Each pre-gate residual regressed on the excluded run's innovation."""
    reg, spl = p90["regressions"], p90["splits"]
    d = p90["design"]

    def _row(label: str, r: dict, price: bool) -> str:
        nd = 3 if price else 1
        return (f"{label} & {_f(r['slope'], nd, signed=True)} & "
                f"{_f(r['se_cluster'], nd)} & {_f(r['t'], 1, signed=True)} & "
                f"{_p(r['p_two_sided'])} & "
                f"{_pct_small(100 * r['r2_increment'])} \\\\")

    rows = [
        "\\toprule",
        "Dependent variable & Slope & Cluster SE & $t$ & $p$ & "
        "$\\Delta R^2$ (\\%) \\\\",
        "\\midrule",
        f"\\multicolumn{{6}}{{l}}{{\\emph{{Full sample: {d['n_slots']:,} slots, "
        f"{d['n_clusters']:,} area-day clusters}}}} \\\\",
    ]
    for key, label, price in MECH_ROW_LABELS:
        rows.append(_row(label, reg[key], price))
    rows += ["\\midrule",
             "\\multicolumn{6}{l}{\\emph{Price residual, by subset}} \\\\"]
    for key, label in MECH_SPLIT_LABELS:
        if key in spl:
            rows.append(_row(label, spl[key]["price"], True))
    rows.append("\\bottomrule")
    return ("\\begin{tabular}{@{}lrrrrr@{}}\n" + "\n".join(rows)
            + "\n\\end{tabular}\n")


def physical_by_policy(p61: dict) -> str:
    per = _g(p61, "system_metrics", "windows", "pre", "per_policy")
    L = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy (information set) & $|q-G|$ (MWh) & shortfall & surplus "
        r"& Balancing cost (k JPY) \\",
        r"\midrule",
    ]
    for key, label in POLICY_LABELS:
        v = per.get(key)
        if v is None:
            continue
        L.append(
            f"{label} & {v['abs_imbalance_mwh']:,.1f} & "
            f"{v['shortfall_mwh']:,.1f} & {v['surplus_mwh']:,.1f} & "
            f"{v['balancing_cost_k_yen'] / 1000:,.0f} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# Control words that are pure formatting: deleting them changes no meaning.
# Everything else must either expand or appear in the symbol table inside
# _expand_macros, because the fallback is deletion.
_EXPAND_DROPPABLE = frozenset({
    "small", "footnotesize", "normalsize", "centering", "noindent", "par",
    "hfill", "quad", "qquad", "medskip", "smallskip", "bigskip", "linebreak",
    "newline", "protect", "relax", "phantom", "hspace", "vspace", "label",
    "ref", "cite", "citep", "citet", "footnote", "left", "right", "bigl",
    "bigr", "mathbf", "mathit", "boldsymbol", "displaystyle", "limits",
    "nolimits", "textsuperscript", "textsubscript", "ensuremath",
})


def _expand_macros(body: str, defs: dict[str, str]) -> str:
    """Expand \newcommand bodies and strip LaTeX down to plain text.

    Shared by the abstract and the highlights emitters: two copies of this would
    drift, and the highlights were hand-typed precisely because there was no
    expander to hand.
    """
    for _ in range(6):  # macros may expand into macros
        body = re.sub(r"\\([A-Za-z]+)\{?\}?",
                      lambda mm: defs.get(mm.group(1), mm.group(0)), body)
    # Order matters. Unwrap single-argument commands FIRST, keeping the argument:
    # stripping braces before control words turns \unit{JPY/kWh} into
    # \unitJPY/kWh, and the greedy \\[A-Za-z]+ then eats "JPY" and prints
    # "0.229 /kWh".
    body = body.replace("{,}", ",").replace(r"\%", "%")
    # \suppref is a one-argument \newcommand, so the zero-argument definition
    # sweep above cannot expand it; spell out what it renders as.
    body = re.sub(r"\\suppref\{([^{}]*)\}", r"Supplementary Material, Section \1", body)
    for _ in range(4):  # \unit{\jpykwh}-style nesting
        body = re.sub(r"\\(?:unit|text|mathrm|emph|textbf|texttt|url|SI)\{([^{}]*)\}",
                      r"\1", body)
    # Symbols that carry meaning and must survive as characters. \pm was being
    # deleted by the sweep below, so the Editorial-Manager abstract said the
    # excluded run is "equivalent within 0.01 JPY/kWh" where the manuscript says
    # "within +/-0.01" -- a one-sided reading of the paper's only equivalence
    # bound, in the text an editor reads first.
    for cw, ch in (("pm", "\u00b1"), ("approx", "\u2248"), ("times", "\u00d7"),
                   ("leq", "\u2264"), ("geq", "\u2265"), ("sim", "~"),
                   ("degree", "\u00b0"), ("rho", "\u03c1"), ("Delta", "\u0394"),
                   ("alpha", "\u03b1"), ("kappa", "\u03ba"), ("mu", "\u03bc")):
        body = re.sub(rf"\\{cw}(?![A-Za-z])\s*", ch, body)
    # Anything still unresolved is about to be DELETED. That is how a symbol goes
    # missing without a trace, so stop instead: either it is a macro that should
    # have expanded, or it belongs in the table above.
    leftover = sorted(set(re.findall(r"\\([A-Za-z]+)", body)) - _EXPAND_DROPPABLE)
    if leftover:
        msg = ("_expand_macros would silently delete these control words: "
               f"{leftover}. Add each to the symbol table if it carries meaning, "
               "or to _EXPAND_DROPPABLE if it is pure formatting.")
        raise SystemExit(msg)
    body = re.sub(r"\\[A-Za-z]+\s*", "", body)  # remaining control words
    return (
        body.replace("~", " ")
        .replace("---", "\u2014")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
    )


EM_ABSTRACT_LIMIT = 250  # Applied Energy guide for authors
EM_HIGHLIGHT_CHARS = 85  # Elsevier: max 85 characters per highlight, incl. spaces
EM_HIGHLIGHT_MAX = 5     # Elsevier: 3 to 5 highlights

# The highlights, with every quantity as a macro. They were hand-typed until
# 2026-08-13 -- the same defect the abstract had, in the paragraph an editor reads
# second. Bullet 4 states the BOUNDED form of the gate result rather than the
# equivalence form: the one-tick equivalence is the single claim in this paper
# that a later capacity register does not support (Section 7), and a highlight has
# no room for that caveat, whereas "no detectable pooled accuracy" holds on all
# three weightings tested.
EM_HIGHLIGHTS: tuple[str, ...] = (
    r"Weather inputs are judged by dissemination time, not by initialisation time",
    r"Archive timestamps show the run initialised before the gate arrives after it",
    r"A run disseminated before the gate cuts day-ahead price MAE by \PreGateImp JPY/kWh",
    r"The excluded run adds no detectable 24-hour pooled accuracy at that gate",
    r"Pre-gate weather raises floor-event PR-AUC in all \FloorPrAucAreasPos price areas",
)


EM_HIGHLIGHT_NOTES = r"""## What these bullets deliberately do not say

- Bullet 4 says 24-HOUR pooled, because that qualifier is load-bearing: the
  same innovation handed to the learner directly is worth about one and a half
  price ticks on the daylight window the physical claim is measured on
  (Section 5.3). The pooled null is partly a dilution over night slots, and an
  unqualified "pooled" would overstate it in a bullet with no room to explain.
- Bullet 4 states the BOUNDED gate result, not the one-tick equivalence.
  Equivalence is the single claim in this paper that a later capacity-register
  snapshot does not support (Section 7, Supplementary S4); "no detectable
  pooled accuracy" holds on all three weightings tested, so it is the form
  that belongs in a sentence with no room for a caveat.
- No bullet reports the \PreGateMultiple-fold multiple of the reanalysis
  proxy. Its denominator (\StaleObsImp JPY/kWh) is small enough that the ratio
  overstates the effect; the absolute improvement and the absolute difference
  are reported instead.
- No bullet asserts a price-formation mechanism. The innovation explains an
  order of magnitude more of the physical residual variance than of the price
  residual variance, which the paper reports as target divergence and not as
  identification.
- The floor bullet may assert "all nine areas" because the capacity-weighted
  provenance the headline now uses shows the contrast positive in nine of
  nine. That condition was recorded here while the re-estimation was pending;
  it is met.
"""


def emit_em_highlights() -> tuple[Path, str]:
    r"""Regenerate the Editorial-Manager highlights from macros.

    Hand-kept until now, which meant the five sentences an editor reads carried
    unguarded copies of 0.253, 0.01, 14% and 1%. Elsevier's per-bullet character
    limit is enforced here rather than tracked in a trailing comment that says
    "re-verify if edited".
    """
    macros_tex = (CFG.project_root / "docs/paper/shared/macros.tex").read_text(
        encoding="utf-8")
    defs = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                   macros_tex)
    )
    # Validate the template BEFORE expanding. _expand_macros strips control words
    # it cannot resolve, so a typo -- \PreGateImpX for \PreGateImp -- does not
    # leave a backslash to detect: it silently deletes the number and the bullet
    # ships reading "cuts day-ahead price MAE by JPY/kWh". Checking afterwards
    # cannot see that; checking the template can.
    unknown = sorted({
        name for tmpl in (*EM_HIGHLIGHTS, EM_HIGHLIGHT_NOTES)
        for name in re.findall(r"\\([A-Za-z]+)", tmpl) if name not in defs
    })
    if unknown:
        msg = (f"highlights reference macros that do not exist in macros.tex: "
               f"{unknown}. Expansion would delete them silently.")
        raise SystemExit(msg)
    if len(EM_HIGHLIGHTS) > EM_HIGHLIGHT_MAX:
        msg = (f"{len(EM_HIGHLIGHTS)} highlights; Elsevier allows "
               f"{EM_HIGHLIGHT_MAX}")
        raise SystemExit(msg)
    lines, over = [], []
    for raw in EM_HIGHLIGHTS:
        text = re.sub(r"\s+", " ", _expand_macros(raw, defs)).strip()
        if re.search(r"\\[A-Za-z]", text):
            msg = f"unexpanded macro in highlight: {text}"
            raise SystemExit(msg)
        n = len(text)
        if n > EM_HIGHLIGHT_CHARS:
            over.append(f"{n} chars: {text}")
        lines.append(f"- {text}  <!-- {n} chars -->")
    if over:
        # A bullet that overruns is silently truncated by the submission form,
        # which is how a highlight loses its unit and its meaning.
        msg = ("highlights over Elsevier's "
               f"{EM_HIGHLIGHT_CHARS}-character limit:\n  " + "\n  ".join(over))
        raise SystemExit(msg)
    out = CFG.project_root / "docs/paper/paper_a_main/submission/_highlights_for_em.md"
    doc = (
        f"<!-- GENERATED by {Path(__file__).name} from EM_HIGHLIGHTS and "
        f"shared/macros.tex. Do not edit by hand; edit the template there and "
        f"regenerate. -->\n\n"
        f"# Highlights (Editorial Manager)\n\n"
        f"{len(lines)} bullets, each <= {EM_HIGHLIGHT_CHARS} characters "
        f"including spaces (Elsevier limit). Counts are computed, and an "
        f"overrun fails this script rather than printing a warning.\n\n"
        + "\n".join(lines) + "\n\n"
        + EM_HIGHLIGHT_NOTES
    )
    # The prose block carries quantities too (the multiple, its denominator), so
    # it goes through the same expander -- and then NOTHING in the file may still
    # look like a macro. \StaleObsImp shipped unexpanded in the first version of
    # this emitter, in a sentence explaining why a ratio was dropped.
    doc = _expand_macros(doc, defs)
    leftover = re.findall(r"\\[A-Za-z]+", doc)
    if leftover:
        msg = f"unexpanded macros in the highlights file: {sorted(set(leftover))}"
        raise SystemExit(msg)
    return out, doc


def emit_em_highlights_upload() -> tuple[Path, str]:
    r"""Write the bullets alone, as the file that is actually uploaded.

    ``_highlights_for_em.md`` carries the bullets AND the rationale for what they
    deliberately omit -- which bound was chosen over which, why a ratio with a
    small denominator was left out. That commentary is why the bullets are
    defensible and it belongs in the repository, but it is editorial reasoning
    about how to phrase a highlight, and an editor opening the Highlights
    attachment must not find it. The upload manifest pointed at that annotated
    file, so the only Highlights artefact in the tree was one that could not be
    sent. This writes the other half: five bullets, no counts, no notes, nothing
    to strip by hand on submission day.
    """
    macros_tex = (CFG.project_root / "docs/paper/shared/macros.tex").read_text(encoding="utf-8")
    defs = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", macros_tex)
    )
    lines = []
    for raw in EM_HIGHLIGHTS:
        text = re.sub(r"\s+", " ", _expand_macros(raw, defs)).strip()
        if len(text) > EM_HIGHLIGHT_CHARS:
            msg = f"highlight over {EM_HIGHLIGHT_CHARS} chars: {text}"
            raise SystemExit(msg)
        lines.append(f"- {text}")
    doc = "# Highlights\n\n" + "\n".join(lines) + "\n"
    if re.search(r"\\[A-Za-z]", doc):
        msg = f"unexpanded macro in the upload highlights: {doc!r}"
        raise SystemExit(msg)
    out = CFG.project_root / "docs/paper/paper_a_main/submission/Highlights.md"
    return out, doc



DECLARATION_SECTIONS = (
    "CRediT authorship contribution statement",
    "Declaration of competing interest",
    "Declaration of generative AI and AI-assisted technologies in the writing process",
    "Data availability",
    "Funding",
)


def emit_declarations() -> tuple[Path, str]:
    r"""Regenerate the Editorial-Manager declarations from the manuscript itself.

    ``submission/declarations.md`` is the file whose own header tells the submitter
    to paste its contents into the Editorial Manager form, and it was a hand-kept
    transcription of five end-matter sections carrying two ticked boxes that
    certified byte-identity with ``main.tex``. Both certifications were false by
    2026-08-21: the AI declaration had grown a third role in the manuscript (the
    literature audit's primary coding) and this file still described two, and the
    data list named the imbalance-settlement disclosure, which backs no quantity
    reported in this paper -- so a submitter working from it would have filed a
    source declaration for data the article does not use.

    Transcription is the defect. Expanding the manuscript's own sections removes
    the copy, exactly as it was removed for the abstract and the highlights.
    """
    main_tex = (CFG.project_root / "docs/paper/paper_a_main/main.tex").read_text(encoding="utf-8")
    macros_tex = (CFG.project_root / "docs/paper/shared/macros.tex").read_text(encoding="utf-8")
    defs = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", macros_tex)
    )
    # Section numbers come from the compiled .aux. If the manuscript has not been
    # built, a \ref falls back to its label name, which is visible and ugly
    # rather than silently wrong -- but say so, because it means a stale build.
    aux = CFG.project_root / "docs/paper/paper_a_main/main.aux"
    labels: dict[str, str] = {}
    if aux.exists():
        labels = dict(re.findall(
            r"\\newlabel\{([^{}]*)\}\{\{([^{}]*)\}", aux.read_text(encoding="utf-8")))
    parts = []
    for title in DECLARATION_SECTIONS:
        pat = (r"\\section\*\{" + re.escape(title)
               + r"\}(.*?)(?=\\section\*\{|\\bibliography|\\end\{document\})")
        m = re.search(pat, main_tex, re.DOTALL)
        if m is None:
            msg = (f"main.tex has no \\section*{{{title}}}; the declarations file "
                   "cannot be generated from a manuscript that does not carry it")
            raise SystemExit(msg)
        raw = m.group(1)
        # A LaTeX comment is invisible in the PDF and must stay invisible here:
        # "% ---- references ----" was reaching the file an editor reads, with
        # its dashes helpfully converted to em dashes.
        raw = re.sub(r"(?m)^\s*%.*$", "", raw)
        # Resolve cross-references against the compiled .aux, so "Section~\ref
        # {sec:market_related}" becomes "Section 2.3" rather than shipping the
        # label name -- which is what the AI declaration was doing.
        raw = re.sub(r"\\ref\{([^{}]*)\}",
                     lambda mm: labels.get(mm.group(1), mm.group(1)), raw)
        body = _expand_macros(raw, defs)
        # itemize survives macro expansion as bare \item text; make it markdown.
        body = re.sub(r"\\item\s*", "\n- ", body)
        body = body.replace("\\&", "&").replace("\\,", " ")
        body = re.sub(r"(?<!-)--(?!-)", "\u2013", body)  # en dash
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        parts.append(f"## {title}\n\n{body}\n")
    doc = (
        f"<!-- GENERATED by {Path(__file__).name} from the end-matter sections of "
        f"main.tex. Do not edit by hand; edit main.tex and regenerate. Every section "
        f"below is the manuscript's own text, so the two cannot diverge. -->\n\n"
        "# Declarations (Editorial Manager)\n\n"
        "Paste each section into the corresponding Editorial Manager field. These are\n"
        "the manuscript's end-matter sections, expanded to plain text; the manuscript\n"
        "is the single source, so there is nothing here to keep in step by hand.\n\n"
        + "\n".join(parts)
        + "\n### Confirm before upload\n\n"
        "- [ ] Competing-interest statement: final author confirmation, and the\n"
        "      signed .docx produced with Elsevier's declaration-of-interest tool\n"
        "      attached at the Attach Files step.\n"
        "- [ ] CRediT roles as intended; corresponding author as intended.\n"
        "- [ ] Funding: none. If that changes, change main.tex, not this file.\n"
    )
    leftover = re.findall(r"\\[A-Za-z]+", doc)
    if leftover:
        msg = f"unexpanded macros in the declarations file: {sorted(set(leftover))}"
        raise SystemExit(msg)
    out = CFG.project_root / "docs/paper/paper_a_main/submission/declarations.md"
    return out, doc


def emit_em_abstract() -> tuple[Path, str]:
    r"""Regenerate the Editorial-Manager abstract by expanding the real one.

    ``submission/_abstract_for_em.md`` is what gets pasted into the submission
    form, and it was a hand-kept transcription of the abstract in ``main.tex``.
    It drifted, of course: it still carried the retracted system-cost sentence
    after the manuscript had dropped it, and its stated word count was off by
    one. Expanding the macros mechanically removes the copy.
    """
    main_tex = (CFG.project_root / "docs/paper/paper_a_main/main.tex").read_text(encoding="utf-8")
    macros_tex = (CFG.project_root / "docs/paper/shared/macros.tex").read_text(encoding="utf-8")
    defs = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", macros_tex)
    )
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.DOTALL)
    if m is None:
        msg = "no abstract environment found in main.tex"
        raise SystemExit(msg)
    text = re.sub(r"\s+", " ", _expand_macros(m.group(1), defs)).strip()
    # A bare em dash is punctuation, not a word: str.split() counted each of the
    # abstract's five as one, so the reported count ran five over the prose and
    # would have sent someone trimming an abstract that already complied.
    n_words = len([t for t in text.split() if re.search(r"[0-9A-Za-z]", t)])
    if n_words > EM_ABSTRACT_LIMIT:
        # The limit is a submission requirement, so exceeding it is a build
        # failure, not a line of output. Printing it and continuing is how a
        # 260-word abstract reaches the form: the number was right there and
        # nothing stopped on it.
        msg = (f"abstract is {n_words} words; Applied Energy allows "
               f"{EM_ABSTRACT_LIMIT}. Trim main.tex, do not raise the limit.")
        raise SystemExit(msg)
    out = CFG.project_root / "docs/paper/paper_a_main/submission/_abstract_for_em.md"
    doc = (
        f"<!-- GENERATED by {Path(__file__).name} from the abstract in "
        f"main.tex. Do not edit by hand; regenerate. -->\n\n"
        f"# Abstract (Editorial Manager)\n\n"
        f"Word count: {n_words} of {EM_ABSTRACT_LIMIT} allowed by Applied "
        f"Energy (em dashes excluded: they are punctuation)\n\n{text}\n"
    )
    return out, doc


LADDER_PER_AREA_TIERS = [
    ("era5", "Stale obs."),
    ("prev2", "Composite pre-gate"),
    ("msm_d2_12z", "Pre-gate run"),
    ("msm_d1_00z", "Post-gate run"),
    ("histfc", "Recon."),
    ("oracle", "Ex-post"),
]
# The primary measurement's per-area detail. The single-point table above is
# the robustness arm's; printing only it under a caption that quotes the
# extended panel's day count would put phase59's 714-day numbers under a
# 732-day heading, which is what the supplement did until 2026-08-12.
LADDER_PER_AREA_TIERS_CW = [
    ("era5_cw", "Stale obs."),
    ("msm_d2_12z_cw", "Pre-gate run"),
    ("msm_gate_composite_cw", "Freshest admissible"),
    ("msm_d1_00z_cw", "Post-gate run"),
    ("oracle_cw", "Ex-post"),
]


def ladder_per_area_generic(pa: dict, tiers: list, headline: str,
                            area_labels: dict[str, str],
                            pv_peak: dict[str, float]) -> str:
    """One per-area ladder table, for whichever measurement arm is passed."""
    order = sorted(area_labels, key=lambda a: -pv_peak[a])
    cols = "l r " + " ".join(["r"] * len(tiers)) + " r"
    L = [
        f"\\begin{{tabular}}{{{cols.replace(' ', '')}}}",
        r"\toprule",
        r" & & \multicolumn{%d}{c}{Improvement over price-only (\jpykwh)} & \\"
        % len(tiers),
        r"\cmidrule(lr){3-%d}" % (2 + len(tiers)),
        "Area & PV/peak & " + " & ".join(lbl for _, lbl in tiers)
        + r" & DM $p$ (headline) \\",
        r"\midrule",
    ]
    for a in order:
        cells = []
        for tier, _ in tiers:
            v = pa.get(tier, {}).get(a, {}).get("improvement")
            cell = "---" if v is None else f"{v:.3f}"
            if tier == headline and v is not None:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        p = pa.get(headline, {}).get(a, {}).get("p_l3")
        pcell = "---" if p is None else (r"$<10^{-6}$" if p < 1e-6 else f"${p:.2g}$")
        L.append(
            f"{area_labels[a]} & {pv_peak[a]:.2f} & " + " & ".join(cells)
            + f" & {pcell} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def ladder_per_area(p59: dict, area_labels: dict[str, str], pv_peak: dict[str, float]) -> str:
    r"""Per-area vintage ladder, generated from the run-pinned ladder provenance.

    The hand-typed version of this table showed the COMPOSITE tiers from phase51
    (Hokkaido pre-gate 0.081) while the main text's headline had moved to the
    run-pinned phase59 ladder (Hokkaido 0.090), and its caption still pointed at
    phase51 as its provenance. The supplement therefore disagreed with the
    article about which tier the paper was about.

    Both constructions are now shown side by side, which is the paper's actual
    argument rather than an accident: the composite sits BELOW the single
    run-pinned run in most areas, so assembling a series across initialisations
    understates the value of the forecast.
    """
    pa = p59["per_area"]
    order = sorted(area_labels, key=lambda a: -pv_peak[a])
    cols = "l r " + " ".join(["r"] * len(LADDER_PER_AREA_TIERS)) + " r"
    L = [
        f"\\begin{{tabular}}{{{cols.replace(' ', '')}}}",
        r"\toprule",
        r" & & \multicolumn{%d}{c}{Improvement over price-only (\jpykwh)} & \\"
        % len(LADDER_PER_AREA_TIERS),
        r"\cmidrule(lr){3-%d}" % (2 + len(LADDER_PER_AREA_TIERS)),
        "Area & PV/peak & "
        + " & ".join(lbl for _, lbl in LADDER_PER_AREA_TIERS)
        + r" & DM $p$ (headline) \\",
        r"\midrule",
    ]
    for a in order:
        cells = []
        for tier, _ in LADDER_PER_AREA_TIERS:
            v = pa.get(tier, {}).get(a, {}).get("improvement")
            cell = "---" if v is None else f"{v:.3f}"
            if tier == "msm_d2_12z" and v is not None:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        p = pa.get("msm_d2_12z", {}).get(a, {}).get("p_l3")
        pcell = "---" if p is None else (r"$<10^{-6}$" if p < 1e-6 else f"${p:.2g}$")
        L.append(
            f"{area_labels[a]} & {pv_peak[a]:.2f} & " + " & ".join(cells) + f" & {pcell} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


AREA_JA_ORDER = ("kyushu", "shikoku", "chugoku", "kansai", "hokuriku",
                 "tohoku", "hokkaido", "tokyo")


def curtail_crosstab(p259: dict, area_labels: dict[str, str]) -> str:
    r"""Per-area floor-vs-curtailment crosstab, moved out of the main text.

    The main text keeps the linkage and the range of risk ratios; the per-area
    detail belongs here. It was 40 hand-typed cells before, which is 40 numbers
    that no check could see.
    """
    rows = {r["area"]: r for r in _g(p259, "floor_x_curtail_crosstab")}
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Area & Floor rate (\%) & Curtail rate (\%) & "
         r"$P(\text{floor}\mid\text{curtail})$ & "
         r"$P(\text{floor}\mid\neg\text{curtail})$ & Risk ratio \\",
         r"\midrule"]
    for a in AREA_JA_ORDER:
        r = rows.get(a)
        if r is None:
            continue
        n = r["n_slots"]
        floor_rate = 100.0 * (r["n_floor_and_curtail"] + r["n_floor_no_curtail"]) / n
        curt_rate = 100.0 * (r["n_floor_and_curtail"] + r["n_no_floor_curtail"]) / n
        L.append(f"{area_labels[a]} & {floor_rate:.2f} & {curt_rate:.2f} & "
                 f"{r['P(floor|curtail)']:.3f} & {r['P(floor|no_curtail)']:.3f} & "
                 f"${r['risk_ratio_curtail_vs_no']:.1f}\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def decision_table(p61: dict, by_tier: dict[str, dict]) -> str:
    r"""Main-text Table 5: physical and private outcomes, all rows from provenance.

    This table was written with literal numbers, and by the time anyone checked,
    THREE of its rows had gone stale against the run-pinned re-run -- the later
    window (+60 against a provenance +48, which an external reviewer caught
    because it contradicted the body text on the same page), the full window
    (-139 against -169) and the matched row (-313 [-599,-33] against
    -372 [-671,-80]). Three further rows cited alternate-vintage runs that had
    left no artefact on disk at all, so nothing could be checked against
    anything.

    Two changes, both structural. Every cell is now read from a provenance JSON.
    And the alternate-vintage rows are scored on the SAME 396/151/547-day
    windows as the headline rather than on 87- and 50-day matched subsets, which
    is both a larger sample and an actual like-for-like comparison; the matched
    row is kept as one line because it is the only place the committed window's
    reconstruction tier is directly comparable.
    """
    win = _g(p61, "system_metrics", "windows")
    WINDOWS = [
        ("Earlier", "pre_collapse", "pre"),
        ("Later", "post_collapse", "post"),
        ("Full", "all", "pooled"),
    ]
    TIERS = [
        ("msm_d2_12z", r"Pre-gate run ($D{-}2$ 12\,UTC)", True),
        ("msm_d1_00z", r"First post-gate run ($D{-}1$ 00\,UTC)", False),
        ("histfc", "Reconstruction (blended composite)", False),
    ]
    L = [
        r"\begin{tabular}{llrrlrl}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{Physical: $\Delta|q-G|$ (MWh)} "
        r"& \multicolumn{2}{c}{Private: CNPE (thousand JPY)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Vintage class & Window & Days & $\Delta$ & 95\% CI & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for ti, (tier, label, physical) in enumerate(TIERS):
        if ti:
            L.append(r"\midrule")
        for wlabel, regime, pwin in WINDOWS:
            v = _g(
                by_tier[tier], "windows", "prev2", "regimes", regime, "voi", "op_vs_base", "pooled"
            )
            pt = v["voi_yen_per_mw_yr"] / 1000.0
            lo, hi = (x / 1000.0 for x in v["bootstrap_ci95_yen_per_mw_yr"])
            # Bold the headline cell only: the pre-gate run's earlier window.
            strong = physical and regime == "pre_collapse"
            wrap = (lambda s: f"$\\mathbf{{{s}}}$") if strong else (lambda s: f"${s}$")
            if physical:
                b = win[pwin]
                dabs = wrap(f"{b['op_vs_base']['d_abs_imbalance_mwh']:+,.0f}")
                inf = b.get("d_abs_imbalance_daily_inference") or {}
                ci = inf.get("ci95")
                dci = f"$[{ci[0]:+,.0f}, {ci[1]:+,.0f}]$" if ci else "---"
            else:
                dabs, dci = "---", "---"
            L.append(
                f"{label if wlabel == 'Earlier' else ''} & {wlabel} & "
                f"{v['n_days']} & {dabs} & {dci} & "
                f"{wrap(f'{pt:+,.0f}')} & $[{lo:+,.0f}, {hi:+,.0f}]$ \\\\"
            )
    m = _g(
        by_tier["msm_d2_12z"],
        "windows",
        "prev2",
        "regimes",
        "matched_committed_pre",
        "voi",
        "op_vs_base",
        "pooled",
    )
    mpt = m["voi_yen_per_mw_yr"] / 1000.0
    mlo, mhi = (x / 1000.0 for x in m["bootstrap_ci95_yen_per_mw_yr"])
    L += [
        r"\midrule",
        f"Pre-gate, matched & Earlier & {m['n_days']} & --- & --- & "
        f"${mpt:+,.0f}$ & $[{mlo:+,.0f}, {mhi:+,.0f}]$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    # Every output of this emitter lands inside the manuscript tree, so a
    # repository that ships no manuscript -- the public companion -- has
    # nothing for it to do. Say that and succeed. This has to come FIRST: the
    # provenance-existence loop below, and the json.loads calls after it, are
    # preconditions for emitting, and the companion deliberately ships no
    # paper-B provenance -- so anywhere later in main() this exits 2 on a file
    # whose absence is by design. `make verify-paper` is the first command the
    # companion's README gives a reviewer, and it was failing on exactly that.
    if not OUT.parent.exists():
        log.info("no manuscript tree here (%s absent); the generated tables and "
                 "the EM abstract live with the paper, so there is nothing to "
                 "emit or check in this repository", OUT.parent)
        return 0
    by_tier = {
        t: RUN / f"phase52_fip_commitment{s}.json"
        for t, s in (
            ("msm_d2_12z", ""),
            ("msm_d1_00z", "_op_msm_d1_00z"),
            ("histfc", "_op_histfc"),
        )
    }
    for path in (P60, P61, P59, P259, P76, P73, P74, P74G, P89, P90, P83,
                 P82, P82B, P85, P91, P54V2, P88, *by_tier.values()):
        if not path.exists():
            log.error("missing %s", path)
            if path in by_tier.values():
                log.error(
                    "  regenerate with: s_20260726_phase52_fip_commitment.py "
                    "--windows prev2 --op-tier-prev2 <tier> "
                    "--ladder-parquet phase59_exact_ladder_per_slot.parquet "
                    "--out-suffix _op_<tier>"
                )
            return 2
    p60 = json.loads(P60.read_text(encoding="utf-8"))
    p61 = json.loads(P61.read_text(encoding="utf-8"))
    p59 = json.loads(P59.read_text(encoding="utf-8"))
    p259 = json.loads(P259.read_text(encoding="utf-8"))
    p76 = json.loads(P76.read_text(encoding="utf-8"))
    p73 = json.loads(P73.read_text(encoding="utf-8"))
    p74 = json.loads(P74.read_text(encoding="utf-8"))
    p74g = json.loads(P74G.read_text(encoding="utf-8"))
    p89 = json.loads(P89.read_text(encoding="utf-8"))
    p90 = json.loads(P90.read_text(encoding="utf-8"))
    p97 = json.loads(P97.read_text(encoding="utf-8"))
    p80 = json.loads(P80.read_text(encoding="utf-8"))
    p78 = json.loads(P78.read_text(encoding="utf-8"))
    p86 = json.loads(P86.read_text(encoding="utf-8"))
    p83 = json.loads(P83.read_text(encoding="utf-8"))
    p82 = json.loads(P82.read_text(encoding="utf-8"))
    p82b = json.loads(P82B.read_text(encoding="utf-8"))
    p85 = json.loads(P85.read_text(encoding="utf-8"))
    p91 = json.loads(P91.read_text(encoding="utf-8"))
    p54v2 = json.loads(P54V2.read_text(encoding="utf-8"))
    p88 = json.loads(P88.read_text(encoding="utf-8"))
    tiers = {t: json.loads(p.read_text(encoding="utf-8")) for t, p in by_tier.items()}
    for t, j in tiers.items():  # a mislabelled artefact would silently swap a row
        on_disk = _g(j, "provenance", "prev2")["op_tier"]
        if on_disk != t:
            log.error("%s holds op_tier '%s', expected '%s'", by_tier[t].name, on_disk, t)
            return 2
    OUT.mkdir(parents=True, exist_ok=True)
    header = (
        "% GENERATED by 90_emit_tables.py -- do not\n"
        "% edit by hand. Regenerate after any re-run of the phase that\n"
        "% produces its provenance JSON.\n"
    )
    artefacts: list[tuple[Path, str]] = [
        (OUT / name, header + body)
        for name, body in (
            ("phase52_decision.tex", decision_table(p61, tiers)),
            ("phase59_ladder_per_area.tex", ladder_per_area(p59, AREA_LABELS, pv_peak_by_area())),
            ("phase85_ladder_per_area_cw.tex",
             ladder_per_area_generic(
                 json.loads((RUN / "phase85_exact_ladder.json").read_text(
                     encoding="utf-8"))["arm_cw_primary"]["per_area"],
                 LADDER_PER_AREA_TIERS_CW, "msm_d2_12z_cw",
                 AREA_LABELS, pv_peak_by_area())),
            ("phase25_curtail_crosstab.tex", curtail_crosstab(p259, AREA_LABELS)),
            ("phase60_wedge_regressions.tex", wedge_regressions(p60)),
            ("phase60_split_sensitivity.tex", split_sensitivity(p60)),
            ("phase61_nomination_band.tex", nomination_band(p61)),
            ("phase61_physical_by_policy.tex", physical_by_policy(p61)),
            ("phase76_nomination_rules.tex", nomination_rules(p76)),
            ("phase73_cycle_curve.tex", cycle_curve(p73)),
            ("phase80_version_split.tex", version_split(p80)),
            ("phase78_buffer_audit.tex", buffer_audit(p78)),
            ("phase86_second_provider.tex", second_provider(p86)),
            ("phase74_cw_ladder.tex", cw_ladder(p74, p74g)),
            ("phase89_physical_tiers.tex", physical_tiers(p89)),
            ("phase90_mechanism.tex", mechanism(p90)),
            ("phase97_cost_loss.tex", cost_loss(p97)),
            ("phase91_weight_sensitivity.tex", weight_sensitivity(p91)),
            ("phase54v2_enriched_nwp.tex", enriched_nwp(p54v2)),
            ("phase88_lit_audit_counts.tex", lit_audit_counts(p88)),
            ("phase83_margin_curve.tex", equivalence_margins(p83, p82)),
            ("phase85_vintage_ladder.tex", vintage_ladder(p85)),
            ("phase82b_absolute.tex", absolute_performance(p82b)),
        )
    ]
    artefacts.append(emit_em_abstract())
    artefacts.append(emit_em_highlights())
    artefacts.append(emit_em_highlights_upload())
    artefacts.append(emit_declarations())
    # The Japanese manuscript's three main-text tables. Same provenance, same
    # numbers, Japanese wording; emitted only when that tree is present.
    if OUT_JA.parent.exists():
        OUT_JA.mkdir(parents=True, exist_ok=True)
        artefacts += [
            (OUT_JA / name, header + body)
            for name, body in (
                ("phase85_vintage_ladder.tex", vintage_ladder(p85, "ja")),
                ("phase82b_absolute.tex", absolute_performance(p82b, "ja")),
                ("phase83_margin_curve.tex",
                 equivalence_margins(p83, p82, "ja")),
            )
        ]

    # --check compares what is on disk with what this script would write now.
    # Regenerating is only a guarantee if somebody actually regenerates: a table
    # generated from provenance goes stale the moment its phase is re-run and the
    # emitter is not, which is exactly how the literals it replaced went stale.
    # This is content-based on purpose -- mtimes are meaningless after a git
    # checkout, so a timestamp check would pass on a tree that is wrong.
    if "--check" in sys.argv:
        drifted = []
        for dest, want in artefacts:
            have = dest.read_text(encoding="utf-8") if dest.exists() else None
            if have != want:
                drifted.append(f"{dest.name}: {'missing' if have is None else 'stale'}")
        if drifted:
            log.error("generated artefacts out of date with provenance:")
            for d in drifted:
                log.error("  %s", d)
            log.error("run this script without --check to regenerate")
            return 1
        log.info("all %d generated artefacts match provenance", len(artefacts))
        return 0

    for dest, body in artefacts:
        dest.write_text(body, encoding="utf-8")
        log.info("wrote %s (%d lines)", dest.name, body.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
