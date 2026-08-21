r"""Phase 87: the information timeline, split into Figures 1a and 1b.

WHY A THIRD VERSION
-------------------
The sixth external review asked for the combined phase63 figure to be split:
the clock (which run is in hand at the gate) and the measured cycle curve are
two different claims, and the caption of the combined figure had drifted into
"plateau" language the phase73 retraction no longer licenses. This split also
rebinds the curve panel to the extended-panel provenance
(``phase73v2_cycle_curve.json``, 735 days) and the dissemination lag to the
five-cycle 881-day audit (``phase78_lineage_audit.json``) instead of the
three-cycle phase72 sample.

Outputs (manuscript tree when present, else ./figures):
  information_timeline_a.pdf  -- the clock + 48-slot composite strip
  information_timeline_b.pdf  -- the measured cycle curve + buffer band

Every number is read from committed provenance and the script fails loudly if
a value is missing, so the figures cannot drift away from the analyses they
illustrate.

CLI (from the repository root):
  uv run --extra research python scripts/91_figure_timeline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase87_timeline")

ROOT = EpfConfig().project_root
_PROV_DIRS = [d for d in (ROOT / "data/runs", ROOT / "data/provenance")
              if d.is_dir()]


def _prov(name: str) -> Path:
    for d in _PROV_DIRS:
        if (d / name).exists():
            return d / name
    tried = ", ".join(str(d / name) for d in _PROV_DIRS) or "(no provenance dir)"
    msg = f"provenance {name} not found; tried {tried}"
    raise FileNotFoundError(msg)


_MANUSCRIPT = ROOT / "docs/paper/paper_a_main"
_HAS_MS = (_MANUSCRIPT / "main.tex").exists()
FIGDIR = (_MANUSCRIPT / "figures") if _HAS_MS else (ROOT / "figures")

PRE, POST, PROXY = "#008300", "#ec835a", "#898781"
GATE, INK, SEC, MUT = "#d03b3b", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND, STRIP = "#e1e0d9", "#c3c2b7", "#efeee8", "#f4f3ef"
WBOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.5}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.4,
                     "text.color": INK, "axes.unicode_minus": True,
                     "figure.dpi": 150})

# time axis: hours since D-2 18:00 JST
T_D1, T_GATE, T_D, T_END = 6.0, 16.0, 30.0, 54.0
NOMINAL_LAG = 2.5  # JMBSC specification, init + 2 h 30 m


def _measured_lag_hours(p78: dict) -> tuple[float, int]:
    """Median per-run dissemination lag across the five audited cycles.

    phase78 measures every archived cycle over the full panel; the medians
    must agree to within a minute or the single-lag drawing is dishonest.
    """
    runs = p78["run_dissemination"]
    medians = sorted(v["median_lag_min"] for v in runs.values())
    if medians[-1] - medians[0] > 1.0:
        msg = f"phase78 per-run median lags disagree: {medians}"
        raise ValueError(msg)
    n_days = min(v["n_days"] for v in runs.values())
    return round(medians[len(medians) // 2] / 60.0, 3), n_days


RUNS = {
    "d2_12z": {"init": 3.0, "horizon": 78.0, "label": "$D{-}2$ 12 UTC",
               "colour": PRE},
    "d2_21z": {"init": 12.0, "horizon": 39.0, "label": "$D{-}2$ 21 UTC",
               "colour": PRE},
    "d1_00z": {"init": 15.0, "horizon": 78.0, "label": "$D{-}1$ 00 UTC",
               "colour": POST},
}
Y_ROWS = {"d2_12z": 6.6, "d2_21z": 5.5, "d1_00z": 4.4, "era5": 3.3}
Y_STRIP = 1.9
COMPOSITE_SWITCH_HOUR = 22
HOURS_PER_DAY = 24
BAR_HALF = 0.13


def _bar(ax, y, x0, x1, colour, *, solid=True) -> None:
    if solid:
        ax.add_patch(FancyBboxPatch(
            (x0, y - BAR_HALF), x1 - x0, 2 * BAR_HALF,
            boxstyle="round,pad=0,rounding_size=0.30", mutation_aspect=0.30,
            facecolor=colour, edgecolor="none", zorder=3.5))
    else:
        ax.add_patch(Rectangle(
            (x0, y - BAR_HALF), x1 - x0, 2 * BAR_HALF, facecolor="none",
            edgecolor=colour, hatch="////", linewidth=0.5, alpha=0.75, zorder=3.4))


def draw_clock(measured_lag: float, n_days: int, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04)

    ax.add_patch(Rectangle((T_D, 0.62), T_END - T_D, 6.48, facecolor=BAND,
                           edgecolor="none", zorder=0))
    ax.text((T_D + T_END) / 2, 7.24, "delivery day $D$", ha="center",
            va="bottom", fontsize=6.4, color=SEC)
    for t, lab in ((T_D1, "$D{-}1$ 00:00"), (T_D, "$D$ 00:00"),
                   (T_END, "$D{+}1$ 00:00")):
        ax.axvline(t, color=AXIS, lw=0.5, zorder=0.5)
        ax.text(t, 0.30, lab, ha="center", va="top", fontsize=5.9, color=MUT)
    ax.axvline(T_GATE, color=GATE, lw=1.25, zorder=4)
    ax.text(T_GATE - 0.35, 7.24, "gate closes\n$D{-}1$ 10:00 JST", ha="right",
            va="bottom", fontsize=6.2, color=GATE, linespacing=1.15)

    for r in RUNS.values():
        y = Y_ROWS[[k for k, v in RUNS.items() if v is r][0]]
        nominal = r["init"] + NOMINAL_LAG
        measured = r["init"] + measured_lag
        ax.plot([r["init"]], [y], "o", ms=3.6, mfc="white", mec=r["colour"],
                mew=0.9, zorder=5)
        ax.plot([measured], [y], "o", ms=2.4, mfc=r["colour"], mec="none",
                alpha=0.45, zorder=4.6)
        ax.plot([nominal], [y], "o", ms=3.8, mfc=r["colour"], mec="white",
                mew=0.5, zorder=5)
        ax.plot([r["init"], nominal], [y, y], "-", color=r["colour"], lw=0.7,
                alpha=0.55, zorder=3)
        cov0 = max(T_D, nominal)
        horizon_end = r["init"] + r["horizon"]
        if horizon_end > cov0:
            _bar(ax, y, cov0, min(horizon_end, T_END), r["colour"], solid=True)
        if horizon_end < T_END:
            _bar(ax, y, max(horizon_end, cov0), T_END, r["colour"], solid=False)
        margin_h = T_GATE - nominal
        hh, mm = divmod(int(round(abs(margin_h) * 60)), 60)
        span = f"{hh} h" + (f" {mm} m" if mm else "")
        tag = (f"+{span} before gate" if margin_h > 0 else f"{span} AFTER gate")
        ax.text(r["init"] - 0.5, y, r["label"], ha="right", va="center",
                fontsize=6.3, color=INK)
        ax.text(T_END + 0.5, y, tag, ha="left", va="center", fontsize=5.9,
                color=r["colour"] if margin_h > 0 else GATE)

    y = Y_ROWS["era5"]
    _bar(ax, y, T_D, T_END, PROXY, solid=False)
    ax.text(T_D - 0.5, y, "ERA5 lag-48 slots", ha="right", va="center",
            fontsize=6.3, color=SEC)
    ax.text(T_END + 0.5, y, "publishes days later", ha="left", va="center",
            fontsize=5.9, color=PROXY)

    ax.text(T_D - 0.5, Y_STRIP, "composite tier,\nper slot", ha="right",
            va="center", fontsize=6.2, color=INK, linespacing=1.15)
    for h in range(HOURS_PER_DAY):
        x0 = T_D + h
        governing = "d2_21z" if h < COMPOSITE_SWITCH_HOUR else "d2_12z"
        shade = 0.30 if governing == "d2_21z" else 0.62
        ax.add_patch(Rectangle((x0 + 0.06, Y_STRIP - 0.22), 0.88, 0.44,
                               facecolor=PRE, alpha=shade, edgecolor="white",
                               linewidth=0.35, zorder=3.5))
    ax.axvline(T_D + COMPOSITE_SWITCH_HOUR, ymin=0.055, ymax=0.20, color=INK,
               lw=0.9, zorder=6)
    ax.text(T_D + COMPOSITE_SWITCH_HOUR, Y_STRIP - 0.45,
            "21 UTC run's archived files end at FH39;\n"
            "22:00-23:00 fall back to the 12 UTC run",
            ha="center", va="top", fontsize=5.7, color=SEC, linespacing=1.2,
            bbox=WBOX)
    ax.text(T_D + 10, Y_STRIP + 0.34, "$D{-}2$ 21 UTC", ha="center", va="bottom",
            fontsize=5.7, color=INK)
    ax.text(T_D + 23, Y_STRIP + 0.34, "12 UTC", ha="center", va="bottom",
            fontsize=5.7, color=INK)

    lag_h, lag_m = divmod(int(round(measured_lag * 60)), 60)
    ax.text(T_END + 15.5, -0.42,
            f"open = initialisation; filled = dissemination (spec 2 h 30 m; "
            f"measured median {lag_h} h {lag_m:02d} m, {n_days} days $\\times$ "
            "5 cycles)",
            ha="right", va="top", fontsize=5.6, color=SEC)

    ax.set_xlim(-7.5, T_END + 16.0)
    ax.set_ylim(-0.75, 7.75)
    ax.axis("off")
    ax.set_title("Which run is in hand when the gate closes", fontsize=7.6,
                 color=INK, loc="left", pad=7)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    log.info("wrote %s", out_pdf)


def draw_curve(p73: dict, p78: dict, out_pdf: Path) -> None:
    rungs = p73["rungs"]
    order = ["msm_d3_12z", "msm_d2_12z", "msm_d2_18z", "msm_d2_21z", "msm_d1_00z"]
    xs = [-rungs[k]["gate_margin_hours_measured"] for k in order]
    ys = [rungs[k]["improvement"] for k in order]
    los = [rungs[k]["ci95"][0] for k in order]
    his = [rungs[k]["ci95"][1] for k in order]

    fig, bx = plt.subplots(figsize=(4.6, 3.3))
    fig.subplots_adjust(left=0.135, right=0.97, top=0.90, bottom=0.14)
    bx.set_xlim(min(xs) - 4.0, max(xs) + 6.5)
    pady = 0.16 * (max(his) - min(los))
    bx.set_ylim(min(los) - pady * 2.8, max(his) + pady * 2.4)

    # worst-case processing margin of the latest pre-gate run (21 UTC composite
    # leg): the shaded band is the region a submission pipeline must clear.
    worst_margin_h = p78["run_dissemination"]["msm_d2_21z"]["worst_margin_min"] / 60.0
    bx.axvspan(-worst_margin_h, 0.0, color=GATE, alpha=0.06, zorder=1)
    bx.axvline(0.0, color=GATE, lw=1.1, zorder=4)
    bx.text(0.0, bx.get_ylim()[1], "gate", ha="center", va="top", fontsize=6.0,
            color=GATE, bbox=WBOX, zorder=6)

    bx.plot(xs, ys, "-", color=AXIS, lw=0.7, zorder=2)
    curve_offsets = {
        "msm_d3_12z": (0, 9, "center", "bottom"),
        "msm_d2_12z": (0, -14, "center", "top"),
        "msm_d2_18z": (0, 9, "center", "bottom"),
        "msm_d2_21z": (2, -14, "center", "top"),
        "msm_d1_00z": (7, -1, "left", "center"),
    }
    for k, x, y, lo, hi in zip(order, xs, ys, los, his, strict=True):
        col = POST if k == "msm_d1_00z" else PRE
        bx.plot([x, x], [lo, hi], "-", color=SEC, lw=0.7, zorder=3)
        for cap in (lo, hi):
            bx.plot([x - 0.55, x + 0.55], [cap, cap], "-", color=SEC, lw=0.7,
                    zorder=3)
        bx.plot([x], [y], "o", ms=4.6, mfc=col, mec="white", mew=0.6, zorder=5)
        dx, dy, ha, va = curve_offsets[k]
        # spell the rung the way the caption and the text do: "D-3 12Z", not
        # the provenance key "msm_d3_12z", which appears nowhere in the paper
        day, cyc = k.replace("msm_", "").split("_")
        label = f"$D{{-}}{day[1]}$ {cyc[:2]}Z"
        bx.annotate(label, (x, y),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                    fontsize=5.6, color=col, zorder=6, bbox=WBOX)

    step = p73["pairwise_adjacent"]["msm_d2_12z_minus_msm_d3_12z"]
    widest = p73["pairwise_summary"]["widest_ci_ticks_after_first"]
    bx.annotate(
        f"one significant step (+{step['diff']:.3f} "
        f"[{step['ci95'][0]:.3f}, {step['ci95'][1]:.3f}]);\n"
        "after it no adjacent step is detectable.\n"
        "The gate step passes a one-tick equivalence test;\n"
        f"the two intermediate steps reach {widest:.1f} ticks.",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=5.6, color=SEC, linespacing=1.25, bbox=WBOX, zorder=6)
    worst_min = p78["run_dissemination"]["msm_d2_21z"]["worst_margin_min"]
    n78 = p78["run_dissemination"]["msm_d2_21z"]["n_days"]
    bx.annotate(
        f"worst measured margin of the\nlatest pre-gate run: "
        f"{worst_min:.0f} min ({n78} days)",
        xy=(-worst_margin_h - 0.4, bx.get_ylim()[0] + pady * 0.4),
        ha="right", va="bottom", fontsize=5.6, color=GATE, alpha=0.85,
        linespacing=1.25, bbox=WBOX, zorder=6)

    n_days = p73["panel"]["n_days"]
    bx.set_xlabel("measured dissemination time relative to the gate (h)",
                  fontsize=6.4, color=SEC)
    bx.set_ylabel(f"pooled MAE improvement, {n_days} days (JPY/kWh)",
                  fontsize=6.4, color=SEC)
    bx.tick_params(labelsize=6.0, colors=SEC)
    bx.grid(True, color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        bx.spines[side].set_color(AXIS)
    # 16:30, not 16:00: the provenance records an inclusive hour filter [8, 16]
    # and the slots are half-hourly labelled by start, so the panel holds 18
    # slots per area-day and the last one starts at 16:30
    # (phase73v2_cycle_curve.json panel.n_slots = 735 x 9 x 18 = 119,070).
    # Section 5, Section 2 and Table S29 all print 16:30; this title did not,
    # so one window printed two ways inside a single float.
    bx.set_title("The measured cycle curve (common 08:00-16:30 window)",
                 fontsize=7.4, color=INK, loc="left", pad=6)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    log.info("wrote %s", out_pdf)


def main() -> int:
    p73 = json.loads(_prov("phase73v2_cycle_curve.json").read_text(encoding="utf-8"))
    p78 = json.loads(_prov("phase78_lineage_audit.json").read_text(encoding="utf-8"))
    lag, n_days = _measured_lag_hours(p78)
    draw_clock(lag, n_days, FIGDIR / "information_timeline_a.pdf")
    draw_curve(p73, p78, FIGDIR / "information_timeline_b.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
