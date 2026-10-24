r"""Phase 116 figures: the Energy manuscript's four figures, redrawn for legibility.

WHY THIS EXISTS
---------------
Applied Energy referee R1.12: "Figure 1 is unclear, font size too small,
overlapping text, blurry. Check all figures and improve them." The old timeline
was drawn at 5.6-7.6 pt on a 6.3-inch canvas with a legend that ran into the
axis labels. These versions set every label at 9 pt or more on a canvas no wider
than the printed text block, use a colour-blind-safe palette, keep text out of
the data region, and draw panel (b) of Figure 1 as the identification argument
itself: documented lower bounds and measured arrivals around the gate.

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
# Okabe-Ito: distinguishable under the common colour-vision deficiencies.
PRE, POST, REF, DOC, INK, MUT = "#0072B2", "#D55E00", "#7f7f7f", "#009E73", "#111111", "#555555"
GATE = "#CC0000"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


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
        ("msm_d2_12z", "$D{-}2$ 12 UTC run", -27.0, 78.0, PRE),
        ("msm_d2_21z", "$D{-}2$ 21 UTC run", -18.0, 39.0, PRE),
        ("msm_d1_00z", "$D{-}1$ 00 UTC run", -15.0, 78.0, POST),
    ]
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.55,
                          left=0.02, right=0.99, top=0.94, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    ax.add_patch(Rectangle((0, 0.45), 24, 3.15, facecolor="#f2f2f2", edgecolor="none",
                           zorder=0))
    ax.text(12, 3.7, "delivery day $D$", ha="center", va="bottom", color=MUT)
    ax.axvline(gate, color=GATE, lw=1.6, zorder=5)
    ax.text(gate + 0.4, 3.7, "gate 10:00", ha="left", va="bottom", color=GATE)
    for i, (key, label, init, horizon, c) in enumerate(runs):
        y = 3.0 - i
        lag_h = p78[key]["median_lag_min"] / 60
        arr = init + lag_h
        ax.plot([init, arr], [y, y], color=c, lw=1.2, zorder=3)
        ax.plot(init, y, "o", ms=6.5, mfc="white", mec=c, mew=1.5, zorder=4)
        ax.plot(arr, y, "o", ms=6.5, mfc=c, mec=c, zorder=4)
        end = init + horizon
        ax.add_patch(Rectangle((0, y - 0.17), min(end, 24), 0.34, facecolor=c, alpha=0.85,
                               edgecolor="none", zorder=2))
        if end < 24:
            ax.add_patch(Rectangle((end, y - 0.17), 24 - end, 0.34, facecolor="none",
                                   edgecolor=c, hatch="////", lw=0.6, zorder=2))
        ax.text(init - 0.6, y, label, ha="right", va="center", color=INK)
        margin = gate - arr
        hh, mm = divmod(round(abs(margin) * 60), 60)
        tag = (f"{hh} h {mm:02d} min\n{'before' if margin > 0 else 'after'} the gate\n"
               "archive median")
        ax.text(24.8, y, tag, ha="left", va="center", color=c if margin > 0 else GATE,
                linespacing=1.1)
    ticks = [-24, -18, -12, -6, 0, 6, 12, 18, 24]
    labels = ["$D{-}1$\n00:00", "06:00", "12:00", "18:00", "$D$\n00:00", "06:00",
              "12:00", "18:00", "$D{+}1$\n00:00"]
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-41, 36)
    ax.set_ylim(0.45, 4.15)
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_title("(a) The three runs that matter for delivery day $D$ (JST)", loc="left")

    bx = fig.add_subplot(gs[1])
    bx.axvline(gate, color=GATE, lw=1.6, zorder=5)
    bx.text(gate + 0.03, 2.75, "gate", ha="left", va="center", color=GATE)
    arr21 = -18 + p78["msm_d2_21z"]["median_lag_min"] / 60
    bx.plot(arr21, 2.15, "o", ms=7, mfc=PRE, mec=PRE, zorder=4)
    bx.text(arr21 + 0.07, 2.15, "$D{-}2$ 21 UTC run in archive", ha="left", va="center",
            color=PRE)
    y = 1.0
    bx.add_patch(Rectangle((job_start, y - 0.13), job_end - job_start, 0.26, facecolor=DOC,
                           alpha=0.18, edgecolor="none", zorder=1))
    bx.text(10.65 - 24, y + 0.22, "analysis and forecast job (scheduled)", ha="center",
            va="bottom", color=DOC)
    bx.plot(-15, y, "o", ms=7, mfc="white", mec=POST, mew=1.5, zorder=4)
    for t in (cutoff, job_start):
        bx.plot(t, y, "D", ms=6.5, mfc=DOC, mec=DOC, zorder=4)
    bx.annotate("observation cut-off 09:50,\njob start 09:55", xy=(cutoff, y + 0.05),
                xytext=(-14.62, 1.6), ha="right", va="center", color=DOC,
                arrowprops={"arrowstyle": "-", "color": DOC, "lw": 0.8})
    lag00 = p78["msm_d1_00z"]["median_lag_min"] / 60
    bx.plot(-15 + lag00, y, "o", ms=7, mfc=POST, mec=POST, zorder=4)
    bx.text(-15 + lag00 + 0.07, y, "$D{-}1$ 00 UTC run\nin archive", ha="left", va="center",
            color=POST, linespacing=1.1)
    bx.text(-15, y - 0.22, "initialised 09:00", ha="center", va="top", color=POST)
    ticks_b = [-16, -15, -14, -13, -12]
    bx.set_xticks(ticks_b, ["08:00", "09:00", "10:00", "11:00", "12:00"])
    bx.set_xlim(-16.15, -11.45)
    bx.set_ylim(0.35, 3.05)
    bx.set_yticks([])
    bx.spines[["left"]].set_visible(False)
    bx.set_xlabel("$D{-}1$ (JST)")
    bx.set_title("(b) The hour around the gate", loc="left")
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.3, ms=6.5,
                      label="initialisation"),
               Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=6.5,
                      label="archive arrival (measured median)"),
               Line2D([], [], marker="D", ls="", mfc=DOC, mec=DOC, ms=6,
                      label="documented by JMA"),
               Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, hatch="////", lw=0.6,
                         label="beyond the run's horizon")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
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
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    fig.subplots_adjust(left=0.25, right=0.70, top=0.97, bottom=0.15)
    for i, (k, lab, c) in enumerate(tiers):
        y = len(tiers) - 1 - i
        t = cw["tiers"][k]
        xs = [cw["per_area"][k][a]["improvement"] for a in AREAS]
        ax.scatter(xs, [y] * len(xs), s=14, color=c, alpha=0.35, zorder=2, linewidths=0)
        lo, hi = t["ci95"]
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, zorder=3, solid_capstyle="butt")
        ax.plot(t["improvement"], y, "o", ms=7, color=c, mec="white", mew=0.8, zorder=4)
        ax.text(1.02, y, f'{_f(t["improvement"])} [{_f(lo)}, {_f(hi)}]', ha="left",
                va="center", transform=ax.get_yaxis_transform(), color=INK)
    ax.set_yticks(range(len(tiers)), [lab for _, lab, _ in tiers][::-1])
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xlim(-0.02, 0.43)
    ax.set_xlabel("Reduction in MAE over the price-only model (JPY/kWh)")
    out = FIGDIR / "ladder_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ Figure 3
def fig_floor() -> None:
    d = _prov("phase81b_floor_inference.json")
    met, paired = d["metrics"], d["paired"]
    rows = [("baserank", "price only", REF, "s"), ("m0rank", "pre-gate run", PRE, "o"),
            ("m0rankpost", "post-gate run", POST, "D")]
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.32, left=0.085,
                          right=0.985, top=0.9, bottom=0.25)
    ax = fig.add_subplot(gs[0])
    off = {"baserank": -0.2, "m0rank": 0.0, "m0rankpost": 0.2}
    for key, lab, c, mk in rows:
        pa = met[key]["per_area"]
        xs = [i + off[key] for i in range(len(AREAS))]
        ys = [pa[a]["pr_auc"] for a in AREAS]
        for x, a in zip(xs, AREAS, strict=True):
            lo, hi = pa[a]["ci95"]
            ax.plot([x, x], [lo, hi], color=c, lw=1.0, alpha=0.7, zorder=2)
        ax.plot(xs, ys, mk, ls="", ms=5.5, color=c, mec="white", mew=0.5, zorder=3,
                label=lab)
    ax.set_xticks(range(len(AREAS)), [a.capitalize() for a in AREAS], rotation=35,
                  ha="right")
    ax.set_ylabel("Floor-event PR-AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.set_title("(a) By area, with 95% intervals", loc="left")
    bx = fig.add_subplot(gs[1])
    contr = [(paired["m0rank_minus_baserank"], "pre-gate run\nover price only", PRE),
             (paired["m0rankpost_minus_m0rank"], "post-gate run\nover pre-gate run", POST)]
    for i, (r, lab, c) in enumerate(contr):
        y = 1 - i
        lo, hi = r["ci95"]
        bx.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="butt")
        bx.plot(r["delta"], y, "o", ms=7, color=c, mec="white", mew=0.8)
        bx.text(hi + 0.012, y, _f(r["delta"], sign=True), ha="left", va="center",
                color=INK)
    bx.axvline(0, color="#bbbbbb", lw=0.8)
    bx.set_yticks([1, 0], [lab for _, lab, _ in contr])
    bx.set_ylim(-0.6, 1.6)
    bx.set_xlim(-0.05, 0.52)
    bx.set_xlabel("Change in mean PR-AUC")
    bx.set_title("(b) Contrasts", loc="left")
    out = FIGDIR / "floor_skill.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ Figure 4
PHYS_C, PRICE_C = "#E69F00", "#0072B2"


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
    series = [("physical_mw", "PV generation (physical target)", PHYS_C, "o", -0.11),
              ("price_vs_pre_gate", "Day-ahead price", PRICE_C, "s", 0.11)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.2)
    for key, lab, c, mk, dx in series:
        for i, (ck, _) in enumerate(cats):
            e = sw[ck][key]
            lo, hi = e["relative_ci95_pct"]
            x = i + dx
            ax.plot([x, x], [lo, hi], color=c, lw=2.0, solid_capstyle="butt", zorder=2)
            ax.plot(x, e["relative_pct"], mk, ms=7, color=c, mec="white", mew=0.8,
                    zorder=3, label=lab if i == 0 else None)
            ax.text(x + 0.06, e["relative_pct"], f'{_f(e["relative_pct"], 1)}%',
                    ha="left", va="center", color=INK, fontsize=8.5)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xticks(range(len(cats)), [c for _, c in cats])
    ax.set_xlim(-0.5, len(cats) - 0.4)
    ax.set_ylabel("Error reduction vs the\n$D{-}2$ 12 UTC run (%)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.17), ncol=2)
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
