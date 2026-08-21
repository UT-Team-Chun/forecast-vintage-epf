r"""Results figures R1-R3 for Paper A, generated from committed provenance.

R1  ladder forest plot: per-area improvement with 95% CIs, CW-primary tiers
    (phase85, arm_cw_primary) -- markers encode measurement, one panel.
R2  floor-event skill: per-area PR-AUC of the direct classifier (m2), the
    isotonic recalibration (m0) and the weather-withheld control (m2p) at the
    near-floor threshold, with block-bootstrap intervals and the three
    contrasts that decide the section (phase81b).
R3  two targets: the vintages in dissemination order, scored against the price
    (phase85) and against the operators' photovoltaic actuals on the SAME
    delivery days (phase89), with the measured share of each target's residual
    variance that the excluded run's innovation explains (phase90).

Layout-aware output (same convention as the shared figure scripts): writes into
the manuscript tree when main.tex is present, else ./figures.

CLI (from the repository root):
  uv run python scripts/93_figures_results.py
"""

from __future__ import annotations

import json
import logging
import sys

from pathlib import Path

from decimal import ROUND_HALF_UP, Decimal

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase87_figs")

CFG = EpfConfig()
# Every artefact this script reads is committed provenance. In this checkout it
# lives in data/runs; in the public replication package, which ships no
# data/runs at all, it lives in data/provenance. Resolving only the first meant
# the article's three results figures could not be regenerated where the
# package promises they are -- the same fallback 91_figure_timeline.py uses.
_PROV_DIRS = [d for d in (CFG.resolved("run_dir"),
                          CFG.project_root / "data/provenance") if d.is_dir()]


class _Prov:
    """``RUN / "x.json"`` resolved against whichever provenance dir holds it."""

    def __truediv__(self, name: str) -> Path:
        for d in _PROV_DIRS:
            if (d / name).exists():
                return d / name
        tried = ", ".join(str(d / name) for d in _PROV_DIRS) or "(no provenance dir)"
        msg = f"provenance {name} not found; tried {tried}"
        raise FileNotFoundError(msg)


RUN = _Prov()
_MS = CFG.project_root / "docs/paper/paper_a_main"
FIGDIR = (_MS / "figures") if (_MS / "main.tex").exists() else (CFG.project_root / "figures")

AREAS = ("hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
         "kansai", "chugoku", "shikoku", "kyushu")
INK, SEC, ACC, WARN = "#1a1a2e", "#5c5c70", "#0f6f8f", "#b3401f"


def r1_ladder_forest() -> None:
    d = json.loads((RUN / "phase85_exact_ladder.json").read_text())
    cw = d["arm_cw_primary"]
    tiers = [("era5_cw", "Stale observations (lag-48)", SEC),
             ("msm_d2_12z_cw", "Pre-gate run (D$-$2 12Z)", ACC),
             ("msm_gate_composite_cw", "Freshest admissible per slot", ACC),
             ("msm_d1_00z_cw", "First post-gate run", WARN),
             ("oracle_cw", "Ex-post realised (not a ceiling)", SEC)]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y = 0
    yticks, ylabels = [], []
    for tier, label, color in tiers:
        pooled = cw["tiers"][tier]
        per = cw["per_area"][tier]
        xs = [per[a]["improvement"] for a in AREAS]
        ys = [y + 0.30 * (i / (len(AREAS) - 1) - 0.5) for i in range(len(AREAS))]
        ax.scatter(xs, ys, s=10, color=color, alpha=0.45, zorder=2)
        lo, hi = pooled["ci95"]
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, zorder=3)
        ax.scatter([pooled["improvement"]], [y], s=46, color=color, zorder=4,
                   edgecolor="white", linewidth=0.8)
        ax.text(0.455, y, f'{pooled["improvement"]:.3f} [{lo:.3f}, {hi:.3f}]',
                fontsize=8, color=INK, va="center", family="monospace")
        yticks.append(y)
        ylabels.append(label)
        y -= 1
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_xlim(-0.02, 0.62)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_yticks(yticks, ylabels, fontsize=9)
    ax.set_xlabel("Pooled MAE improvement over the price-only baseline (JPY/kWh)",
                  fontsize=9)
    ax.set_title("The forecast-vintage ladder on the capacity-weighted field\n"
                 "(pooled estimate with 95% CI; faint points are the nine areas)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "ladder_forest.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def r2_floor_skill() -> None:
    """Per-area floor skill WITH intervals, plus the contrasts that decide it.

    Redesigned 2026-08-13. The earlier version drew three per-area lines and no
    uncertainty at all, which is how a 0.055 macro gap and a 0.015 gate contrast
    could look equally solid. Intervals come from phase81b's block bootstrap of
    the macro statistic -- the same draws for every model, so an area's interval
    and the across-area contrast are consistent with each other.
    """
    d = json.loads((RUN / "phase81b_floor_inference.json").read_text())
    met, paired, gate = d["metrics"], d["paired"], d["gate"]

    fig = plt.figure(figsize=(8.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.30,
                          left=0.075, right=0.985, top=0.86, bottom=0.24)
    ax, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    models = [("m2p", "price-only features", SEC, "o", -0.22),
              ("m0", "isotonic on the point forecast", INK, "s", 0.0),
              ("m2", "direct classifier", ACC, "D", 0.22)]
    x = list(range(len(AREAS)))
    for key, label, color, marker, dx in models:
        pa = met[key]["per_area"]
        xs = [i + dx for i in x]
        ys = [pa[a]["pr_auc"] for a in AREAS]
        for xi, a in zip(xs, AREAS, strict=True):
            ci = pa[a].get("ci95")
            if ci and ci[0] is not None:
                ax.plot([xi, xi], ci, color=color, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(xs, ys, marker=marker, ms=4.5, lw=0.9, color=color, label=label,
                zorder=3)
    ax.set_xticks(x, [a.capitalize() for a in AREAS], rotation=40, fontsize=8,
                  ha="right")
    ax.set_ylabel("Floor-event PR-AUC (near-floor, all hours)", fontsize=9)
    ax.set_title("(a) per area, with 95% block-bootstrap intervals", fontsize=9,
                 loc="left", pad=6)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # ---- (b) the contrasts, on the statistic the paper reports -------------
    rows = [(paired["m2_minus_m2p"], "weather, inside\nthe classifier", ACC),
            (paired["m2_minus_m0"], "direct classifier\nover isotonic", ACC),
            (gate["m0post_minus_m0"], "the run the gate\nexcludes", WARN)]
    yy = list(range(len(rows)))[::-1]
    for y, (r, lab, c) in zip(yy, rows, strict=True):
        lo, hi = r["ci95"]
        axc.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=3)
        axc.scatter([r["delta"]], [y], s=42, color=c, zorder=4,
                    edgecolor="white", linewidth=0.8)
        axc.text(hi + 0.012, y, f'{r["delta"]:+.3f}', fontsize=8, color=INK,
                 va="center", family="monospace")
    axc.axvline(0, color="#999", lw=0.8, zorder=1)
    axc.set_yticks(yy, [r[1] for r in rows], fontsize=8)
    axc.set_xlabel("$\\Delta$ macro PR-AUC", fontsize=8.5)
    axc.set_title("(b) contrasts", fontsize=9, loc="left", pad=6)
    axc.set_xlim(-0.06, 0.40)

    for a in (ax, axc):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Floor-event detection: what the weather adds, what a direct "
                 "model adds on top of\nrecalibrating the point forecast, and "
                 "what the excluded run does not add", fontsize=10.2)
    fig.text(0.005, 0.005,
             f'{d["panel"]["n_test_days"]} test days, '
             f'{d["panel"]["n_areas"]} areas, base rate '
             f'{100 * d["panel"]["base_rate"]:.1f}%; intervals are '
             f'{d["metadata"]["n_boot"]:,} circular-block draws over delivery '
             f'days', fontsize=7, color=SEC)
    out = FIGDIR / "floor_skill.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def r3_two_targets() -> None:
    """Paired panels on the common panel, plus the measured target asymmetry.

    Redesigned 2026-08-13. The superseded version was a scatter of physical
    error against price improvement, whose two axes came from different panels
    (791 physical days over eight areas against 732 price days over nine) and
    whose title asserted a price-formation mechanism that phase90 refutes. The
    figure now shares one horizontal axis -- the vintages in dissemination
    order -- across two stacked panels scored on the SAME delivery days, and
    reports the mechanism test as what it measured: an asymmetry in degree.
    """
    p89 = json.loads((RUN / "phase89_physical_target.json").read_text())
    p90 = json.loads((RUN / "phase90_mechanism.json").read_text())
    # Panel (a) reads the price target ON THE COMMON PANEL, not the full ladder.
    # It used to read phase85's pooled tiers -- 732 days and nine areas -- while
    # panel (b) read the 696-day, eight-area common panel, and the caption
    # claimed both were the common panel. The figure asserted a restriction it
    # had not applied. The full-panel values stay in the ladder table.
    cwt = json.loads(
        (RUN / "phase89b_price_on_common_panel.json").read_text())["tiers"]
    phys = p89["targets"]["potential"]["tiers"]

    # (physical key, price key, x label, colour role)
    tiers = [("era5_cw_stale", "era5_cw", "Stale\nobservations", "edge"),
             ("msm_d2_12z_cw", "msm_d2_12z_cw", "Pre-gate\nrun", "adm"),
             ("msm_gate_composite_cw", "msm_gate_composite_cw",
              "Freshest\nadmissible", "adm"),
             ("msm_d1_00z_cw", "msm_d1_00z_cw", "Post-gate\nrun", "excl"),
             ("oracle_cw", "oracle_cw", "Realised\nweather", "edge")]
    col = {"edge": SEC, "adm": ACC, "excl": WARN}
    x = list(range(len(tiers)))

    fig = plt.figure(figsize=(8.6, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], hspace=0.16,
                          wspace=0.34, left=0.10, right=0.985, top=0.865,
                          bottom=0.135)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_phys = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_mech = fig.add_subplot(gs[:, 1])

    # ---- (a) the price target -------------------------------------------
    for i, (_, pk, _, role) in enumerate(tiers):
        c = col[role]
        lo, hi = cwt[pk]["ci95"]
        ax_price.plot([i, i], [lo, hi], color=c, lw=1.6, zorder=2)
        ax_price.scatter([i], [cwt[pk]["improvement"]], s=52, color=c, zorder=4,
                         edgecolor="white", linewidth=0.8)
    ax_price.axhline(0, color="#c8c8d0", lw=0.7, zorder=1)
    ax_price.set_ylabel("Price-target MAE\nimprovement (JPY/kWh)", fontsize=8.5)
    ax_price.set_title("(a) the price target: improvement over the price-only "
                       "baseline, 95% CI", fontsize=9, loc="left", pad=6)
    ax_price.tick_params(labelbottom=False)

    # ---- (b) the physical target, same delivery days ---------------------
    ys = [phys[fk]["pooled_mae_cf_common_mw"] for fk, _, _, _ in tiers]
    ax_phys.plot(x, ys, color="#b8b8c4", lw=1.0, zorder=1)
    for i, (fk, _, _, role) in enumerate(tiers):
        ax_phys.scatter([i], [phys[fk]["pooled_mae_cf_common_mw"]], s=52,
                        color=col[role], zorder=4, edgecolor="white",
                        linewidth=0.8)
    g51 = p89["gates"]["gate51_physical_gain_crossfit_common"]
    ax_phys.plot([1, 3.06], [ys[1], ys[1]], color=WARN, lw=0.8, ls=(0, (3, 2)),
                 zorder=2)
    ax_phys.annotate("", xy=(3.06, ys[3]), xytext=(3.06, ys[1]),
                     arrowprops={"arrowstyle": "->", "color": WARN, "lw": 1.1})
    # Half-up display rounding, matching macros.tex: "%.1f" on 70.55 gives 70.5
    # because the nearest double is below the midpoint, and the figure then
    # disagreed with the text and the table about the same interval.
    _hu = lambda v: str(Decimal(str(v)).quantize(Decimal("0.1"),
                                                 rounding=ROUND_HALF_UP))
    ax_phys.annotate(f'$-${g51["delta_mw"]:.1f} MW\n'
                     f'[{_hu(g51["ci95"][0])}, {_hu(g51["ci95"][1])}]',
                     xy=(3.06, 0.5 * (ys[1] + ys[3])), xytext=(7, -7),
                     textcoords="offset points", fontsize=8, color=WARN)
    ax_phys.set_ylabel("Physical-target error\n(MW, per-area mean; lower better)",
                       fontsize=8.5)
    ax_phys.set_title("(b) the physical target, on the same delivery days",
                      fontsize=9, loc="left", pad=6)
    ax_phys.set_xticks(x, [t[2] for t in tiers], fontsize=8)
    ax_phys.set_xlim(-0.5, len(tiers) - 0.5)
    ax_phys.set_xlabel("Weather vintages in dissemination order "
                       "$\\rightarrow$ fresher", fontsize=8.5)

    # ---- (c) how much of the innovation each target's error carries -------
    reg = p90["regressions"]
    shares = [(100 * reg["physical_residual_on_innovation"]["r2_increment"],
               "physical\nresidual", ACC),
              (100 * reg["price_residual_on_innovation"]["r2_increment"],
               "price\nresidual", WARN),
              (100 * reg["placebo_physical"]["r2_increment"],
               "placebo\n(physical)", "#c8c8d0"),
              (100 * reg["placebo_price"]["r2_increment"],
               "placebo\n(price)", "#c8c8d0")]
    yy = list(range(len(shares)))[::-1]
    for y, (v, lab, c) in zip(yy, shares, strict=True):
        ax_mech.barh([y], [v], height=0.55, color=c, zorder=3)
        txt = f"{v:.1f}%" if v >= 1 else ("< 0.01%" if v < 0.01 else f"{v:.2f}%")
        ax_mech.text(v + 0.35, y, txt, va="center", fontsize=8, color=INK)
    ax_mech.set_yticks(yy, [s[1] for s in shares], fontsize=8)
    ax_mech.set_xlim(0, 1.32 * shares[0][0])
    ax_mech.set_xlabel("Share of residual variance explained\n"
                       "by the excluded run's innovation (%)", fontsize=8.5)
    ax_mech.set_title("(c) the asymmetry, measured", fontsize=9, loc="left",
                      pad=6)
    for ax in (ax_price, ax_phys, ax_mech):
        ax.spines[["top", "right"]].set_visible(False)

    n_days = g51["n_days"]
    fig.suptitle("Two targets, one panel: fresher information keeps improving the "
                 "physical forecast\nafter its price improvement has stopped, and "
                 "the price carries the same information weakly",
                 fontsize=10.2)
    fig.text(0.005, 0.005,
             f'panels (a) and (b): the same {n_days} delivery days and 8 areas; '
             f'(c): {p90["design"]["n_slots"]:,} slots, '
             f'{p90["design"]["n_clusters"]:,} area-day clusters',
             fontsize=7, color=SEC)
    # Renamed 2026-08-13 from expectations_channel.pdf. The figure is submitted
    # as a separate file, so its NAME appears in the Editorial Manager
    # manifest -- and the concept it was named for was retracted. The old
    # name is kept in the research log, which is the record of phase79.
    out = FIGDIR / "two_targets.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    r1_ladder_forest()
    try:
        r2_floor_skill()
    except FileNotFoundError:
        log.warning("phase81b JSON missing -- floor_skill.pdf skipped")
    r3_two_targets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
