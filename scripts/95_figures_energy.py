r"""Phase 116 figures: the Energy manuscript's four figures, redrawn for legibility.

WHY THIS EXISTS
---------------
Applied Energy referee R1.12: "Figure 1 is unclear, font size too small,
overlapping text, blurry. Check all figures and improve them." Each figure is
drawn at the width it prints -- the elsarticle review layout's text block, 345 pt
-- and included at width=\textwidth, so the 7.5-8 pt set here is the size on the
page (Elsevier asks for at least 7 pt). An earlier version set 9 pt on 6-7 inch
canvases that LaTeX then scaled by about 0.68, which printed at about 6 pt.
One Okabe-Ito hue per entity across all four figures (pre-gate runs blue, the
post-gate run vermillion, references grey); text stays in ink, thin marks, values
only where the reader needs them. Panel (b) of Figure 1 is the identification
argument itself: documented lower bounds and measured arrivals around the gate.

Every number is read from provenance; nothing is typed in.

Outputs (docs/paper/paper_a_main/figures/, or ./figures where no manuscript is present):
  information_timeline_a.pdf  Fig. 1  runs, coverage, and the hour around the gate
  ladder_forest.pdf           Fig. 2  the vintage ladder
  floor_skill.pdf             Fig. 3  floor-event detection
  two_targets.pdf             Fig. 4  price vs physical target, same slots (phase118)

CLI (from the repository root):
  uv run python scripts/95_figures_energy.py
"""

from __future__ import annotations

import json
import logging
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase116_figs")
logging.getLogger("fontTools").setLevel(logging.WARNING)  # font-subsetting chatter

CFG = EpfConfig()
_PROV_DIRS = [d for d in (CFG.resolved("run_dir"), CFG.project_root / "data/provenance")
              if d.is_dir()]
# Beside the manuscript when it is here; the public companion ships no manuscript
# and keeps the figures it reproduces in ./figures, as the other figure scripts do.
_MS = CFG.project_root / "docs/paper/paper_a_main"
FIGDIR = (_MS / "figures") if (_MS / "main.tex").exists() else (CFG.project_root / "figures")


def _prov(name: str) -> dict:
    for d in _PROV_DIRS:
        if (d / name).exists():
            return json.loads((d / name).read_text())
    msg = f"provenance {name} not found in {_PROV_DIRS}"
    raise FileNotFoundError(msg)


def _f(x: float, nd: int = 3, sign: bool = False) -> str:
    """Round half up from the stored decimal, as the macro guard does.

    Python's float formatting rounds 0.2405 to 0.240 (the binary value sits just
    below the half), while the guard and the prose print 0.241: the figure and
    the text disagreed in the third decimal.
    """
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    out = f"{q:.{nd}f}"
    return ("+" + out) if sign and q >= 0 else out


AREAS = ("hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku",
         "shikoku", "kyushu")
# Okabe-Ito, one hue per entity in every figure: pre-gate runs blue, the post-gate
# run vermillion, references grey, JMA's documented times bluish green. Colour
# marks identity only; every piece of text is set in ink.
PRE, POST, REF, DOC = "#0072B2", "#D55E00", "#7F7F7F", "#009E73"
INK, MUT, RULE = "#1A1A1A", "#595959", "#BDBDBD"
# Figure 4's two series are targets, not runs, so they take hues no run uses; the
# orange sits below 3:1 against white, so it always travels with a distinct marker
# and a direct label.
PHYS_C, PRICE_C = "#E69F00", "#000000"
# The elsarticle review layout's \textwidth, in inches. Every figure is drawn at
# this width and included at width=\textwidth, so a point size here is the point
# size on the page.
TEXT_W = 345 / 72
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 5,
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2,
    "pdf.fonttype": 42, "savefig.facecolor": "white"})


def _minus(text: str) -> str:
    """A typographic minus: matplotlib applies one to tick labels, not to text."""
    return text.replace("-", "−")


def _hhmm(hours_rel_d: float) -> str:
    """Clock time of an offset in hours from 00:00 JST on delivery day D."""
    m = round((hours_rel_d % 24) * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------------ Figure 1
def fig_timeline() -> None:
    """(a) runs and delivery-day coverage; (b) the hour around the gate.

    Hours are relative to 00:00 JST on delivery day D. The gate is at -14
    (10:00 JST on D-1). Arrival = initialisation + the measured median lag of
    that run (phase78, 881 days); the 00 UTC run's documented markers come from
    phase115, which located each value in JMA's own documents.
    """
    p78 = _prov("phase78_lineage_audit.json")["run_dissemination"]
    p115 = _prov("phase115_dissemination_bounds.json")
    sched = p115["msm00_run_schedule_by_year"]["2020"]
    job_start = -15 + (int(sched["start_utc"][:2]) * 60 + int(sched["start_utc"][2:])) / 60
    job_end = -15 + (int(sched["end_utc"][:2]) * 60 + int(sched["end_utc"][2:])) / 60
    cutoff = -15 + 50 / 60
    gate = -14.0
    runs = [  # key, label, init (h rel. D 00 JST), horizon h, colour
        ("msm_d2_12z", "$D{-}2$ 12 UTC", -27.0, 78.0, PRE),
        ("msm_d2_21z", "$D{-}2$ 21 UTC", -18.0, 39.0, PRE),
        ("msm_d1_00z", "$D{-}1$ 00 UTC", -15.0, 78.0, POST),
    ]
    fig = plt.figure(figsize=(TEXT_W, 3.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.75,
                          left=0.165, right=0.78, top=0.9, bottom=0.2)

    ax = fig.add_subplot(gs[0])
    rows = [2, 1, 0]
    ax.add_patch(Rectangle((0, -0.55), 24, 3.6, facecolor="#EFEFEF", edgecolor="none",
                           zorder=0))
    ax.text(12, 2.45, "Delivery day $D$", ha="center", va="bottom", color=MUT, fontsize=7.5)
    ax.axvline(gate, color=INK, lw=0.9, zorder=5)
    ax.text(gate + 0.5, 2.45, "Gate", ha="left", va="bottom", fontsize=7.5)
    for y, (key, _, init, horizon, c) in zip(rows, runs, strict=True):
        arr = init + p78[key]["median_lag_min"] / 60
        ax.plot([init, arr], [y, y], color=c, lw=1.0, zorder=3)
        ax.plot(init, y, "o", ms=4.6, mfc="white", mec=c, mew=1.1, zorder=4)
        ax.plot(arr, y, "o", ms=4.6, mfc=c, mec=c, zorder=4)
        end = init + horizon
        ax.add_patch(Rectangle((0, y - 0.19), min(end, 24), 0.38, facecolor=c, alpha=0.32,
                               edgecolor="none", zorder=2))
        if end < 24:
            ax.add_patch(Rectangle((end, y - 0.19), 24 - end, 0.38, facecolor="none",
                                   edgecolor=c, hatch="//////", lw=0.5, zorder=2))
        margin = gate - arr
        hh, mm = divmod(round(abs(margin) * 60), 60)
        ax.text(1.04, y, f"{hh} h {mm:02d} min {'before' if margin > 0 else 'after'}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5)
    ax.text(1.04, 2.45, "Archive arrival\nrelative to the gate", ha="left", va="bottom",
            transform=ax.get_yaxis_transform(), color=MUT, fontsize=7, linespacing=1.05)
    ax.set_xticks([-24, -12, 0, 12, 24], ["$D{-}1$ 00:00", "12:00", "$D$ 00:00", "12:00",
                                          "$D{+}1$ 00:00"])
    ax.set_xlim(-28.5, 24.5)
    ax.set_ylim(-0.55, 3.05)
    ax.set_yticks(rows, [r[1] for r in runs])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("(a) Runs, archive arrival and coverage of $D$ (JST)")

    bx = fig.add_subplot(gs[1])
    bx.axvline(gate, color=INK, lw=0.9, zorder=5)
    bx.text(gate + 0.04, 1.4, "Gate", ha="left", va="bottom", fontsize=7.5)
    arr21 = -18 + p78["msm_d2_21z"]["median_lag_min"] / 60
    bx.plot(arr21, 1, "o", ms=4.6, mfc=PRE, mec=PRE, zorder=4)
    bx.text(arr21, 0.72, _hhmm(arr21), ha="center", va="top", fontsize=7)
    bx.add_patch(Rectangle((job_start, -0.15), job_end - job_start, 0.3, facecolor=DOC,
                           alpha=0.22, edgecolor="none", zorder=1))
    bx.plot(-15, 0, "o", ms=4.6, mfc="white", mec=POST, mew=1.1, zorder=4)
    bx.text(-15, -0.28, "09:00", ha="center", va="top", fontsize=7)
    for t in (cutoff, job_start):
        bx.plot(t, 0, "D", ms=4.0, mfc=DOC, mec=DOC, zorder=4)
    bx.text(gate - 0.05, 0.25, f"{_hhmm(cutoff)}, {_hhmm(job_start)}", ha="right",
            va="bottom", fontsize=7)
    arr00 = -15 + p78["msm_d1_00z"]["median_lag_min"] / 60
    bx.plot(arr00, 0, "o", ms=4.6, mfc=POST, mec=POST, zorder=4)
    bx.text(arr00, -0.28, _hhmm(arr00), ha="center", va="top", fontsize=7)
    bx.set_xticks([-16, -15, -14, -13, -12], ["08:00", "09:00", "10:00", "11:00", "12:00"])
    bx.set_xlim(-16.15, -11.85)
    bx.set_ylim(-0.6, 1.85)
    bx.set_yticks([1, 0], ["$D{-}2$ 21 UTC", "$D{-}1$ 00 UTC"])
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("$D{-}1$ (JST)")
    bx.set_title("(b) The hour around the gate")

    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.0, ms=4.6,
                      label="Initialisation"),
               Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.6,
                      label="Archive arrival (median)"),
               Line2D([], [], marker="D", ls="", mfc=DOC, mec=DOC, ms=4.0,
                      label="Cut-off and job start (JMA)"),
               Rectangle((0, 0), 1, 1, facecolor=DOC, alpha=0.22, edgecolor="none",
                         label="Scheduled job (JMA, to 2020)"),
               Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, hatch="//////",
                         lw=0.5, label="Beyond the run's horizon")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.02, 0.0),
               handlelength=1.4)
    out = FIGDIR / "information_timeline_a.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ Figure 2
def fig_ladder() -> None:
    cw = _prov("phase85_exact_ladder.json")["arm_cw_primary"]
    tiers = [("era5_cw", "Stale observation\n(ERA5, 1-day lag)", REF),
             ("msm_d2_12z_cw", "Pre-gate run\n($D{-}2$ 12 UTC)", PRE),
             ("msm_gate_composite_cw", "Freshest pre-gate\nrun per slot", PRE),
             ("msm_d1_00z_cw", "Post-gate run\n($D{-}1$ 00 UTC)", POST),
             ("oracle_cw", "Realised weather\n(ex post)", REF)]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.45))
    fig.subplots_adjust(left=0.235, right=0.745, top=0.9, bottom=0.17)
    n = len(tiers)
    for i, (k, _, c) in enumerate(tiers):
        y = n - 1 - i
        t = cw["tiers"][k]
        xs = [cw["per_area"][k][a]["improvement"] for a in AREAS]
        ax.scatter(xs, [y] * len(xs), s=9, color=c, alpha=0.4, zorder=2, linewidths=0)
        lo, hi = t["ci95"]
        ax.plot([lo, hi], [y, y], color=c, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot(t["improvement"], y, "o", ms=5, color=c, mec="white", mew=0.6, zorder=4)
        ax.text(1.04, y, f'{_f(t["improvement"])} [{_f(lo)}, {_f(hi)}]', ha="left",
                va="center", transform=ax.get_yaxis_transform(), fontsize=7.5)
    ax.text(1.04, n - 0.45, "Estimate [95% CI]", ha="left", va="bottom",
            transform=ax.get_yaxis_transform(), color=MUT, fontsize=7)
    ax.set_yticks(range(n), [lab for _, lab, _ in tiers][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=RULE, lw=0.6, zorder=1)
    ax.set_xlim(-0.02, 0.43)
    ax.set_ylim(-0.6, n - 0.45)
    ax.set_xlabel("Reduction in MAE over the price-only model (JPY/kWh)")
    out = FIGDIR / "ladder_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ Figure 3
def fig_floor() -> None:
    d = _prov("phase81b_floor_inference.json")
    met, paired = d["metrics"], d["paired"]
    rows = [("baserank", "Price only", REF, "s"), ("m0rank", "Pre-gate run", PRE, "o"),
            ("m0rankpost", "Post-gate run", POST, "D")]
    fig = plt.figure(figsize=(TEXT_W, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.1, left=0.1,
                          right=0.985, top=0.86, bottom=0.25)
    ax = fig.add_subplot(gs[0])
    off = {"baserank": -0.22, "m0rank": 0.0, "m0rankpost": 0.22}
    for key, lab, c, mk in rows:
        pa = met[key]["per_area"]
        xs = [i + off[key] for i in range(len(AREAS))]
        ys = [pa[a]["pr_auc"] for a in AREAS]
        for x, a in zip(xs, AREAS, strict=True):
            lo, hi = pa[a]["ci95"]
            ax.plot([x, x], [lo, hi], color=c, lw=0.8, zorder=2)
        ax.plot(xs, ys, mk, ls="", ms=3.8 if mk != "D" else 3.4, color=c, mec="white",
                mew=0.4, zorder=3, label=lab)
    ax.set_xticks(range(len(AREAS)), [a.capitalize() for a in AREAS], rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(AREAS) - 0.45)
    ax.set_ylabel("Floor-event PR-AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.07, 1.0), ncol=3, borderaxespad=0.15,
              handlelength=1.0)
    ax.set_title("(a)")
    bx = fig.add_subplot(gs[1])
    contr = [(paired["m0rank_minus_baserank"], "Pre-gate run vs\nprice only", PRE),
             (paired["m0rankpost_minus_m0rank"], "Post-gate run vs\npre-gate run", POST)]
    for i, (r, lab, c) in enumerate(contr):
        y = 1 - i
        lo, hi = r["ci95"]
        bx.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="butt")
        bx.plot(r["delta"], y, "o", ms=5, color=c, mec="white", mew=0.6)
        bx.text(-0.05, y + 0.16, lab, ha="left", va="bottom", fontsize=7, linespacing=1.0)
        bx.text(hi + 0.015, y, _f(r["delta"], sign=True), ha="left", va="center",
                fontsize=7.5)
    bx.axvline(0, color=RULE, lw=0.6)
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_ylim(-0.45, 1.85)
    bx.set_xlim(-0.06, 0.5)
    bx.set_xlabel("Change in mean PR-AUC")
    bx.set_title("(b)")
    out = FIGDIR / "floor_skill.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ Figure 4
def fig_two_targets() -> None:
    """Error reduction against the D-2 12 UTC run, both targets on one window.

    phase118 scores the physical PV target and the price on the same days, areas
    and daylight slots, so the two series are like for like. The earlier version
    set a daylight physical MAE beside an all-slot price improvement and credited
    the post-gate run with gains the freshest pre-gate run already delivers.
    """
    sw = _prov("phase118_internal_referee_checks.json")["q118b_same_window"]
    cats = [("freshness_before_gate", "Freshest\npre-gate run"),
            ("freshness_total", "Post-gate\nrun"),
            ("ex_post_vs_pre_gate", "Realised\nweather")]
    series = [("physical_mw", "PV generation (physical target)", PHYS_C, "o", -0.12),
              ("price_vs_pre_gate", "Day-ahead price", PRICE_C, "s", 0.12)]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.5))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.87, bottom=0.17)
    for key, lab, c, mk, dx in series:
        for i, (ck, _) in enumerate(cats):
            e = sw[ck][key]
            lo, hi = e["relative_ci95_pct"]
            x = i + dx
            ax.plot([x, x], [lo, hi], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot(x, e["relative_pct"], mk, ms=5 if mk == "o" else 4.4, color=c,
                    mec="white", mew=0.6, zorder=3, label=lab if i == 0 else None)
            ax.text(x + 0.055, e["relative_pct"], _minus(f'{_f(e["relative_pct"], 1)}%'),
                    ha="left", va="center", fontsize=7)
    ax.axhline(0, color=RULE, lw=0.6, zorder=1)
    ax.set_xticks(range(len(cats)), [c for _, c in cats])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(cats) - 0.4)
    ax.set_ylabel("Error reduction relative to\nthe $D{-}2$ 12 UTC run (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.15)
    out = FIGDIR / "two_targets.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_timeline()
    fig_ladder()
    fig_floor()
    fig_two_targets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
