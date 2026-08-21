r"""Phase 50 (rev. 2026-07-28): Applied-Energy submission figures.

Re-render only, no new compute.

REVISION 2026-07-28 (gate-closure correction).  An external review established
that JMA-MSM is disseminated at initialization + 2 h 30 m (JMBSC spec), so the
D-1 00 UTC run reaches users at 11:30 JST -- AFTER the 10:00 JST JEPX
day-ahead gate.  The former "JMA-GPV, exact issuance, available at bid close"
tier is therefore the FIRST POST-GATE vintage, and the old
"proxy -> MSM -> GPV ~= 7x" headline is retracted.  Figures 1 and 2 below are
rebuilt around the dissemination-ordered vintage ladder; every "exact
issuance" / "available at bid close" assertion is removed.  Figure 3 is
unchanged legacy (no longer included by the manuscript).

REVISION 2026-07-28b (number audit).  Two further corrections, both in the
tier panels of figures 1 and 2:
  (a) the PRE-GATE row/lane carried the D-2 21 UTC run's dissemination
      ("08:30 JST, +90 min").  That is not the pre-gate TIER: the
      previous-day-2 series is in hand by 02:00 JST on D-1, at least eight
      hours before the gate (phase51 .dissemination_accounting.tiers
      .prev2_pre_gate; methods Section 4).  Both timelines now run from
      D-2 23:00 so the tier's own dissemination instant is on the axis.
  (b) the roadmap's block-D headline printed the cross-area rank correlation
      rho = +0.83, which is digit-identical to the concentration claim the
      manuscript RETIRES in Section 5.5.  It is replaced by the slot-level
      near-floor risk ratio that Section 5.4 leads with.
Every changed value is asserted in _check_provenance().

  1. docs/paper/paper_a_main/figures/analysis_roadmap.pdf  (fig:roadmap)
     Top-to-bottom analysis overview: data sources -> weather tiers ordered by
     DISSEMINATION time with a tick/cross against the 10:00 JST gate -> the
     common daily walk-forward -> four analysis blocks -> headline outputs.
     Drawn at 127.5 mm = the manuscript \linewidth (elsarticle review, letter),
     so it is included at scale 1.0 and every label prints at its stated size
     (>= 6 pt).  Saved WITHOUT bbox_inches="tight" to preserve that scale.
  2. docs/paper/paper_a_main/submission/graphical_abstract.png (+ .pdf)
     Elsevier graphical abstract, exact 13 x 5 cm @ 300 dpi (1535 x 590 px,
     spec >= 531 x 1328 px h x w).  Three panels: (A) condensed dissemination
     timeline against the gate, (B) the vintage ladder bars, (C) the
     settlement-regime sign flip of the commitment decision value.
  3. docs/paper/paper_a_main/figures/skill_scarcity.pdf
     Legacy simplified redraw (original preserved as
     figures/skill_scarcity_ee_version.pdf).  Not referenced by main.tex or
     supplementary.tex; retained so the script stays reproducible.

Quantitative inputs.  Figures 1 and 2 use the corrected headline numbers,
hardcoded verbatim with a source comment at each site (see CORRECTED HEADLINE
NUMBERS below); their provenance files are

  data/runs/phase51_vintage_ladder.json    (vintage ladder, 714 daily folds)
  data/runs/phase52_fip_commitment.json    (FIP commitment VoI by regime)
  data/runs/phase54_nwp_proxy_baseline.json (NWP-proxy-enriched baseline)
  data/runs/phase25.9_floor_probit_summary.json (curtailment/floor crosstab)

and every value equals the corresponding macro in docs/paper/shared/macros.tex
section "Gate-closure revision".  Figure 3 still reads
  data/runs/phase36_skill_degradation.json
  data/runs/phase35_allhours_headtohead.json

CLI:
  cd <repository root>
  uv run python s_20260724_phase50_ae_figures.py
"""

from __future__ import annotations

import csv
import decimal
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless backend before pyplot import

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase50_ae_figures")

# Layout-aware, for the same reason verify_macros.py is: this file ships in
# the research monorepo (manuscript tree present; provenance in data/runs/)
# and in the public replication package (no manuscript; frozen provenance in
# data/provenance/; figures belong in the repository's own figures/). The old
# parents[5] was the monorepo depth hardcoded, which on a normal clone of the
# package resolved into the user's home directory -- and the mkdir below then
# CREATED directories there.
ROOT = EpfConfig().project_root
_MANUSCRIPT = ROOT / "docs/paper/paper_a_main"
_HAS_MS = (_MANUSCRIPT / "main.tex").exists()
FIG_DIR = (_MANUSCRIPT / "figures") if _HAS_MS else (ROOT / "figures")
SUB_DIR = (_MANUSCRIPT / "submission") if _HAS_MS else (ROOT / "figures")
_PROV_DIRS = [d for d in (ROOT / "data/runs", ROOT / "data/provenance")
              if d.is_dir()]
CAPACITY_CSV = ROOT / "data/reference/area_capacity.csv"


def _runs(name: str) -> Path:
    """First provenance dir holding ``name``; loud about where it looked."""
    for d in _PROV_DIRS:
        if (d / name).exists():
            return d / name
    tried = ", ".join(str(d / name) for d in _PROV_DIRS) or "(no provenance dir)"
    msg = f"provenance {name} not found; tried {tried}"
    raise FileNotFoundError(msg)

# ---------------------------------------------------------------------------
# Palette.
#
# Figures 1-2 (rebuilt) use the vintage palette of
# figures/information_timeline.pdf (phase56,
# s_20260727_phase56_timeline_figure.py) unchanged, so the three
# gate-revision figures read as one system:
#   node validate_palette.js "#008300,#ec835a" --mode light -> ALL CHECKS PASS
#   (protan dE 10.1; the #ec835a sub-3:1 contrast WARN is relieved by the
#   direct value label carried on every post-gate mark).
# Grey #898781 is the deliberate NON-series neutral for the tiers that carry
# no admissible-forecast role (stale observation, reconstruction, ex post);
# it is below the chroma floor by design, not a categorical slot.
# The settlement panel encodes POLARITY, not identity, so it takes the
# documented diverging pair blue <-> red:
#   node validate_palette.js "#2a78d6,#d03b3b" --mode light --pairs all
#   -> ALL CHECKS PASS (CVD dE 23.8, normal 31.6, both >= 3:1).
# Red is never a data mark in the same panel as green: it is the gate rule
# (status critical, always labelled) in the tier panels and the negative arm
# in the settlement panel, which carries no green.
# ---------------------------------------------------------------------------
C_PRE = "#008300"  # pre-gate vintage (previous-day-2 run) accent
C_POST = "#ec835a"  # first post-gate vintage (previous-day-1 run) accent
C_NEUTRAL = "#898781"  # stale observation / reconstruction / ex-post benchmark
C_GATE = "#d03b3b"  # the D-1 10:00 JST gate, and inadmissibility glyphs
C_NEG = "#d03b3b"  # diverging arm: value destroyed at settlement
C_POS = "#2a78d6"  # diverging arm: value created at settlement

# Legacy accents, figure 3 only (unreferenced EE-era skill figure).
C_PROXY = "#eb6834"  # tier 1: ERA5 lag-48 day-lagged reanalysis proxy (+ variants)
C_MSM = "#1baf7a"  # tier 2: JMA-MSM operational reconstruction (histfc)
C_GPV = "#4a3aa7"  # tier 3: previous-day-1 class (+ noise-degraded versions)
C_BASE = "#898781"  # price-only baseline / neutral references (muted ink)
INK = "#0b0b0b"  # primary text
SEC = "#52514e"  # secondary text
MUT = "#898781"  # muted text / axis labels
GRID = "#e1e0d9"  # hairline gridlines
AXIS = "#c3c2b7"  # baselines / box hairlines
GOOD = "#0ca30c"  # status: available at deadline / saving (always icon+label)
GOOD_TXT = "#006300"  # success text on light surface
BAD = "#d03b3b"  # status: not available at deadline (always icon+label)
FILL_LIGHT = "#f4f3ef"  # recessive box fill
WBOX = dict(facecolor="white", edgecolor="none", pad=0.6)  # label relief bbox
# Sequential blue ramp (magnitude: PV share of peak), steps 100..700 from the
# dataviz reference palette; blue is deliberately NOT a tier accent.
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

# ---------------------------------------------------------------------------
# CORRECTED HEADLINE NUMBERS (gate-closure revision).
# Hardcoded verbatim; each site names the provenance JSON and its field.
# Every value equals its macro in docs/paper/shared/macros.tex.
# ---------------------------------------------------------------------------
# phase51_vintage_ladder.json
#   .headtohead.pooled_improvement_yen_kwh_daily_mean_with_ci95.<tier>
#   pooled improvement over the price-only model, JPY/kWh, per-day block
#   bootstrap (10,000 draws, 7-day blocks, seed 2026).
# SUPERSEDED SOURCE: these came from phase51's composite-tier ladder. The ladder
# was rebuilt on run-pinned tiers (phase59) and every value below now reads from
# phase59_exact_ladder.json instead. Gate 7 landed on the pre-registered 7b branch,
# so the gate-timing figure is an UPPER BOUND, not an effect.
#
# PAPER A REBUILD (phase87): every ladder value below now reads from the
# CW-primary extended-panel ladder, phase85_exact_ladder.json
# .arm_cw_primary, and the settlement block is gone with Paper B. The two-target
# contrast takes its place -- rebuilt on phase89/phase90 (2026-08-13), which
# supersede phase79.
IMP_STALE = (0.047, 0.030, 0.063)  # era5_cw               \StaleObsImp
IMP_PRE = (0.253, 0.208, 0.300)  # msm_d2_12z_cw         \PreGateImp
IMP_COMPOSITE = (0.254, 0.209, 0.301)  # msm_gate_composite_cw \CompositeImp
IMP_POST = (0.256, 0.210, 0.303)  # msm_d1_00z_cw         \PostGateImp
IMP_EXPOST = 0.240  # oracle_cw             \ExPostImp
PRE_MULTIPLE = (5.39, 4.23, 7.71)  # bootstrapped ratio    \PreGateMultiple
# The figures annotate the DIFFERENCE, not the ratio. The ratio's denominator
# (0.047) is small enough that 5.39x reads as a larger effect than the 0.206
# JPY/kWh it summarises; the seventh review asked for it out of the headline
# positions, and it stays in the ladder table and Section 5.2 text.
PRE_MINUS_STALE = (0.2061, 0.169, 0.2456)  # dec.headline_vs_stale.difference
GATE_VALUE = (0.003, -0.005, 0.010)  # equivalent at 1 tick  \GateTimingValue
GATE_TOST_P = 0.024  # TOST p, HLN-DM L=3, tick margin  \GateTimingTostP
# The bootstrap value, kept because the manuscript reports both and this file
# used to render THIS one under a comment naming the primary macro -- the more
# favourable of two p-values, unlabelled, in the paper's only image.
GATE_TOST_P_BOOT = 0.023  # \GateTimingTostPBoot
PRE_RETENTION = 99  # 0.2529 / 0.2555       \PreGateRetention
GATE_SHARE = 1  # 100 - retention
# phase85 .panel
N_DAYS, N_AREAS, N_SLOTS = 732, 9, "315,414"  # 2024-08-01..2026-08-05
STALE_DM_AREAS = "7 of 9"  # Holm-adjusted DM-significant areas, stale tier
# phase54v2_current.json .results.deltas_daily_mean_with_ci95
PRE_OVER_ENRICHED = 0.200  # d42 0.200 [0.166, 0.235]  \PreGateOverEnrichedAdm
PROXY_CARRIED_SHARE = 96  # d32 / d42                 \ProxyCarriedShare
# phase89_physical_target.json .targets.potential -- the physical target,
# scored against the operators' recorded photovoltaic generation with curtailed
# energy added back (MW, per-area mean of the scaled daily MAE), on the delivery
# days the price panel and the published actuals have in common, with the
# per-area scale fitted only on days preceding the day being scored.
# SUPERSEDES phase79, whose scale was fitted on the full sample (look-ahead in a
# paper about look-ahead) and whose panel was the wider physical-only one.
PHYS_PRE = 878.4  # msm_d2_12z_cw pooled_mae_cf_common_mw  \PhysPreGateMae
PHYS_POST = 818.6  # msm_d1_00z_cw pooled_mae_cf_common_mw  \PhysPostGateMae
PHYS_DELTA = (59.88, 49.64, 70.55)  # gate51 delta_mw + ci95  \PhysDelta
PHYS_DAYS = 696  # gate51 n_days                          \PhysDeltaDays
# phase90_mechanism.json .regressions -- shares of each pre-gate residual's
# VARIANCE explained by the innovation the excluded run carries. Gate 56 landed
# on branch b: the price residual is explained too, so the claim is an
# order-of-magnitude asymmetry, NOT "the price never saw it".
MECH_PHYS_R, MECH_PRICE_R = 14.0, 1.2  # \MechPhysR, \MechPriceR
# The commensurable pair that now carries panel C. Kept next to the variance
# shares so the difference between the two framings stays visible in one place.
PHYS_REL_PCT, PRICE_REL_PCT = 6.8, 0.14   # \PhysRelPooled, \PriceRelReduction
ASYM_RATIO = 50                            # \AsymmetryRatio
# phase25.9_floor_probit_summary.json .floor_x_curtail_crosstab / .n_obs_total
#   near-floor risk ratio P(floor|curtail)/P(floor|no curtail), FY2024-FY2025.
#   REVISION 2026-07-28: the roadmap's block-D headline used to print the
#   cross-area rank correlation rho = +0.83 (macros.tex \CurtailFloorSpearman).
#   That number collides digit-for-digit with the RETIRED cross-area
#   concentration claim the manuscript closes in Section 5.5 ("+0.83 recorded
#   on the earlier committed sample"), so a reader meets a retired headline in
#   the roadmap. The block now leads with the slot-level risk ratio, which is
#   the sentence Section 5.4 itself leads with and is not a cross-area statistic.
CURTAIL_RR_LO, CURTAIL_RR_HI = 19, 96  # shikoku 19.27 .. tohoku 96.00
CURTAIL_AREAS, CURTAIL_SLOTS = 8, "279,590"
# phase51 .dissemination_accounting (JMBSC spec; gate = D-1 10:00 JST).
# Hours are measured from D-1 00:00 JST, so -0.5 = 23:30 JST on D-2.
# phase72: the MEASURED median completion lag, not the JMBSC spec (150 min).
# The spec is what this figure used to state as fact; the measurement is the
# paper's contribution and what the abstract reports.
DISSEM_LAG_MIN = 136  # \DissemMedianLagMin
DISSEM_LAG_H, DISSEM_LAG_M = DISSEM_LAG_MIN // 60, DISSEM_LAG_MIN % 60
DISSEM_DAYS = "881"  # \DissemAuditDays
SPEC_LAG_MIN = 150  # phase72 .interpretation.spec_lag_minutes (JMBSC)
SPEC_LAG_H, SPEC_LAG_M = SPEC_LAG_MIN // 60, SPEC_LAG_MIN % 60
GATE_HOUR = 10.0  # .gate "D-1 10:00 JST"
# .tiers.prev2_pre_gate -- the pre-gate TIER, not the D-2 21 UTC run: the
# governing run for the latest delivery slot is initialised no later than
# D-2 23:30 JST and disseminated by D-1 02:00 JST, i.e. at least +8 h before
# the gate (>= +31 h30 for the earliest slot). The former "08:30 JST, +90 min"
# annotation was the D-2 21 UTC run's dissemination, a different object.
PRE_INIT, PRE_DISSEM, PRE_MARGIN_H = -0.5, 2.0, 8
# .tiers.gpv_d1_00utc_paper_tier -- the first post-gate tier's defining
# product: init D-1 09:00 JST -> disseminated 11:30 JST, 90 min after the gate.
POST_INIT, POST_DISSEM = 9.0, 11.5  # phase51 accounting: the JMBSC SCHEDULE.
# The lanes are the published schedule and phase51 verifies them as such. The
# subtitle now says so, and gives the measured lag beside it: the figure used
# to state "+2 h 30 m" bare, next to an abstract reporting 2 h 16 min measured
# from 881 days of timestamps -- two numbers for one quantity, neither labelled.
# The freshest-admissible composite's governing run for most delivery slots is
# the D-2 21 UTC cycle: init 06:00 JST on D-1, in hand about 08:30 JST, which
# is the latest pre-gate arrival in the ladder and the one the buffer analysis
# stresses (phase78 worst measured margin 103.6 min).
COMP_INIT, COMP_DISSEM = 6.0, 8.5
# The pre-gate ROW of the roadmap is the D-2 12 UTC RUN, not the retired
# previous-day-2 composite tier, and the two have different clocks: the run
# initialises at 21:00 JST on D-2 (12 UTC) and is in hand by 23:30 JST, which
# is what the row's own verdict text says. PRE_INIT/PRE_DISSEM describe the
# composite tier and are retained only for the superseded-tier figures; the
# graphical abstract's pre-gate lane now uses the pinned run's clock below.
D2_12Z_INIT, D2_12Z_DISSEM = -3.0, -0.5

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SEC,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "text.color": INK,
        "axes.unicode_minus": True,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _load_json(name: str) -> dict:
    with _runs(name).open() as fh:
        return json.load(fh)


def _load_pv_peak() -> dict[str, float]:
    """PV/peak per area from the single-source-of-truth capacity table."""
    out: dict[str, float] = {}
    with CAPACITY_CSV.open() as fh:
        rows = (r for r in fh if not r.lstrip().startswith("#"))
        for rec in csv.DictReader(rows):
            out[rec["area"]] = float(rec["pv_gw"]) / float(rec["peak_gw"])
    return out


def _ramp(v: float) -> str:
    """Interpolate the blue sequential ramp at v in [0, 1] -> hex."""
    v = min(max(v, 0.0), 1.0)
    pos = v * (len(BLUE_RAMP) - 1)
    i = int(pos)
    if i >= len(BLUE_RAMP) - 1:
        return BLUE_RAMP[-1]
    frac = pos - i
    a = tuple(int(BLUE_RAMP[i][j : j + 2], 16) for j in (1, 3, 5))
    b = tuple(int(BLUE_RAMP[i + 1][j : j + 2], 16) for j in (1, 3, 5))
    return "#" + "".join(f"{round(a[k] + frac * (b[k] - a[k])):02x}" for k in range(3))


def _luma(hexcol: str) -> float:
    r, g, b = (int(hexcol[j : j + 2], 16) / 255 for j in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Plain least-squares (display-only fit line for the mini-scatters)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    return slope, my - slope * mx


def _box(ax, x, y, w, h, *, fc=FILL_LIGHT, ec=AXIS, lw=0.7, r=0.010, aspect=2.16) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            mutation_aspect=aspect,  # = canvas width/height -> equal corner radii
        )
    )


# ---------------------------------------------------------------------------
# layout audit: catches clipped labels, out-of-canvas text and sub-6-pt type
# before the figure is ever opened (the validator checks color, not layout)
# ---------------------------------------------------------------------------
_TRACKED: list[tuple[object, float | None, str]] = []


def _t(ax, x, y, s, *, size, limit=None, tag="", **kw):
    """ax.text() that registers the artist for the layout audit.

    limit -- optional right edge (axes fraction) the text must not cross.
    """
    art = ax.text(x, y, s, fontsize=size, **kw)
    _TRACKED.append((art, limit, tag or s[:28]))
    return art


def _audit(fig, ax, *, min_pt: float = 6.0, scale: float = 1.0) -> None:
    """Log every label that overflows its box, leaves the canvas, or is small.

    scale -- factor the figure is scaled by on the page (1.0 when the PDF is
    drawn at the final printed width), so `min_pt` is checked at print size.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = 0
    for art, limit, tag in _TRACKED:
        pt = art.get_fontsize() * scale
        if pt < min_pt - 1e-6:
            log.warning("audit: %.2f pt (< %.1f) on %r", pt, min_pt, tag)
            bad += 1
        bb = art.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if limit is not None and x1 > limit + 1e-4:
            log.warning(
                "audit: overflows by %.1f mm on %r", (x1 - limit) * fig.get_figwidth() * 25.4, tag
            )
            bad += 1
        if x0 < -0.002 or x1 > 1.002 or y0 < -0.002 or y1 > 1.002:
            log.warning("audit: off-canvas [%.3f,%.3f]x[%.3f,%.3f] on %r", x0, x1, y0, y1, tag)
            bad += 1
    log.info("audit: %d labels, %d problems", len(_TRACKED), bad)
    _TRACKED.clear()


def _arrow(ax, x0, y0, x1, y1, *, color=MUT, lw=1.1, rad=0.0, style="-|>", ms=7) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
            color=color,
            linewidth=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _tint(hexcol: str, f: float) -> str:
    """Mix hexcol toward white by fraction f in [0, 1] (dissemination lag bars)."""
    rgb = tuple(int(hexcol[j : j + 2], 16) for j in (1, 3, 5))
    return "#" + "".join(f"{round(c + f * (255 - c)):02x}" for c in rgb)


# ---------------------------------------------------------------------------
# Figure 1: analysis roadmap (fig:roadmap)  -- rebuilt for the gate correction
# ---------------------------------------------------------------------------
# Drawn at the manuscript \linewidth (elsarticle "review", letter paper:
# 361.4 pt = 127.5 mm), and saved without bbox_inches="tight", so
# \includegraphics[width=\linewidth] scales it by exactly 1.0 and the point
# sizes below are the printed point sizes.
RM_W_MM, RM_H_MM = 127.5, 57.6  # was 96.0 before stage 2 was removed


def fig_roadmap(out: Path) -> None:
    fig = plt.figure(figsize=(RM_W_MM / 25.4, RM_H_MM / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    asp = RM_W_MM / RM_H_MM

    def Y(mm: float) -> float:
        return mm / RM_H_MM

    def chev(x: float, y_top: float, y_bot: float) -> None:
        _arrow(ax, x, Y(y_top), x, Y(y_bot), color=AXIS, lw=1.0)

    # ---- stage 1: data sources --------------------------------------------
    _t(
        ax,
        0.010,
        Y(56.8),
        "Data sources",
        size=7.6,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        limit=0.30,
    )
    chips = [
        ("JEPX spot prices", "day-ahead, 9 areas,\n30-min slots"),
        ("JMA-MSM runs", "archived cycles pinned\nby dissemination time"),
        ("ERA5 reanalysis", "day-lagged fields and\nrealised weather"),
        ("OCCTO actuals", "recorded PV generation\nand curtailment"),
    ]
    cw, cgap = 0.230, 0.02333
    for i, (t1, t2) in enumerate(chips):
        cx = 0.010 + i * (cw + cgap)
        _box(ax, cx, Y(42.2), cw, Y(10.8) - Y(0.0), aspect=asp)
        _t(
            ax,
            cx + 0.008,
            Y(51.5),
            t1,
            size=6.4,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
            limit=cx + cw - 0.006,
        )
        _t(
            ax,
            cx + 0.008,
            Y(48.2),
            t2,
            size=6.0,
            color=SEC,
            ha="left",
            va="top",
            linespacing=1.35,
            limit=cx + cw - 0.006,
        )
    chev(0.5, 41.4, 38.6)

    # ---- stage 2 REMOVED (2026-08-13) -------------------------------------
    # This figure used to carry a five-tier dissemination clock between the
    # data sources and the evaluation engine. It reproduced Figure 1a panel for
    # panel, and the duplication was the main reason the seventh review found
    # the roadmap too dense. The tiers are introduced once, in Figure 1a; here
    # the sources feed the engine directly.

    # ---- stage 3: the common evaluation engine -----------------------------
    _box(ax, 0.005, Y(31.6), 0.990, Y(6.6) - Y(0.0), aspect=asp)
    _t(
        ax,
        0.5,
        Y(36.4),
        f"One expanding-window daily walk-forward  ·  {N_DAYS} daily walk-forward "
        f"test days × {N_AREAS} areas  ·  {N_SLOTS} half-hour slots",
        size=6.2,
        color=INK,
        ha="center",
        va="center",
        limit=0.995,
    )
    _t(
        ax,
        0.5,
        Y(33.6),
        "XGBoost carries the ladder; LEAR and naive references  ·  "
        "identical folds and one price-only baseline for every tier",
        size=6.0,
        color=SEC,
        ha="center",
        va="center",
        limit=0.995,
    )
    chev(0.5, 31.2, 29.4)

    # ---- stages 4 and 5: analysis blocks and their headline results --------
    cards = [
        (
            "A",
            "Per-area point\naccuracy",
            "ΔMAE vs price-only",
            C_NEUTRAL,
            f"+{IMP_STALE[0]:.3f} JPY/kWh",
            "from the stale tier;\n"
            f"pooled over {N_DAYS} days,\n"
            f"DM-significant in\n{STALE_DM_AREAS} areas\n(Holm-adjusted)",
        ),
        (
            "B",
            "Vintage ladder\nand gate timing",
            "five tiers, one sample",
            C_PRE,
            f"+{PRE_MINUS_STALE[0]:.3f} pre-gate",
            f"{IMP_STALE[0]:.3f}→{IMP_PRE[0]:.3f} JPY/kWh;\n"
            f"excluded run equivalent\n"
            f"(pooled MAE, 1 tick,\np = {GATE_TOST_P:.3f});\n"
            f"+{PRE_OVER_ENRICHED:.3f} over the\n"
            "enriched baseline",
        ),
        (
            "C",
            "Two targets,\nphysical vs price",
            f"{PHYS_DAYS} common days",
            C_POST,
            f"+{_hu(PHYS_DELTA[0])} MW post-gate",
            # five lines, not seven: naming the statistic in full ("residual
            # variance", correct) had pushed this card's last line below the
            # canvas, which the layout audit flagged as off-canvas.
            f"scaled MAE vs PV\n"
            f"potential: {PHYS_PRE:.0f}→{PHYS_POST:.0f} MW\n"
            f"[{_hu(PHYS_DELTA[1])}, {_hu(PHYS_DELTA[2])}];\n"
            # "residual variance" is the statistic (an R-squared increment), not
            # "residuals"; naming it as a heading keeps the precision and the fit.
            f"residual variance:\n"
            f"{MECH_PHYS_R:.1f}% physical,\n{MECH_PRICE_R:.1f}% price",
        ),
        (
            "D",
            "Curtailment\nchannel",
            "floor slots vs OCCTO",
            C_NEUTRAL,
            f"{CURTAIL_RR_LO}–{CURTAIL_RR_HI}× more likely",
            "near-floor prices when\nPV curtailment is active;\n"
            f"{CURTAIL_AREAS} areas, {CURTAIL_SLOTS} slots;\n"
            "co-occurrence, not\ncausation",
        ),
    ]
    kw, kgap = 0.2375, 0.01333
    for i, (tag, title, sub, col, head, body) in enumerate(cards):
        kx = 0.005 + i * (kw + kgap)
        lim = kx + kw - 0.008
        _box(ax, kx, Y(0.8), kw, Y(28.0) - Y(0.0), fc="white", aspect=asp)
        ax.add_patch(
            Rectangle(
                (kx + 0.0035, Y(20.6)),
                kw - 0.007,
                Y(8.2),
                facecolor=FILL_LIGHT,
                edgecolor="none",
                zorder=0.5,
            )
        )
        ax.plot([kx, kx + kw], [Y(20.6), Y(20.6)], color=AXIS, lw=0.6)
        ax.plot(
            [kx + 0.0035, kx + 0.0035],
            [Y(1.6), Y(19.8)],
            color=col,
            lw=1.8,
            solid_capstyle="round",
        )
        _t(
            ax,
            kx + 0.010,
            Y(27.9),
            f"{tag}.  {title}",
            size=6.3,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
            linespacing=1.30,
            limit=lim,
        )
        _t(ax, kx + 0.010, Y(22.9), sub, size=6.0, color=SEC, ha="left", va="top", limit=lim)
        _t(
            ax,
            kx + 0.014,
            Y(19.4),
            head,
            size=6.4,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
            limit=lim,
        )
        _t(
            ax,
            kx + 0.014,
            Y(16.4),
            body,
            size=6.0,
            color=SEC,
            ha="left",
            va="top",
            linespacing=1.26,
            limit=lim,
        )

    _audit(fig, ax, min_pt=6.0, scale=1.0)
    fig.savefig(out)  # no bbox_inches: keep the exact \linewidth scale
    plt.close(fig)
    log.info("wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 2: graphical abstract (Elsevier, exactly 13 x 5 cm @ 300 dpi)
# ---------------------------------------------------------------------------
GA_W_MM, GA_H_MM = 130.0, 50.0


def _bar(ax, xc: float, y0: float, y1: float, color: str, *, width: float = 0.045) -> None:
    """Bar from baseline y0 to y1: ~1 mm rounded data-end, square at the base.

    The radius is clamped to 40% of the bar height so short bars do not turn
    into lozenges.
    """
    h_mm = abs(y1 - y0) * GA_H_MM
    r = min(1.0, 0.4 * h_mm) / GA_W_MM
    ax.add_patch(
        FancyBboxPatch(
            (xc - width / 2, min(y0, y1)),
            width,
            abs(y1 - y0),
            boxstyle=f"round,pad=0,rounding_size={r}",
            mutation_aspect=GA_W_MM / GA_H_MM,
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
    )
    # square off the baseline end
    r_y = r * GA_W_MM / GA_H_MM
    ax.add_patch(
        Rectangle(
            (xc - width / 2, y0 - (r_y if y1 < y0 else 0.0)),
            width,
            r_y,
            facecolor=color,
            edgecolor="none",
            zorder=2.1,
        )
    )


def _hu(x: float, nd: int = 1) -> str:
    """Half-up display rounding, matching the manuscript's table emitter.

    Python rounds 70.55 to "70.5" (binary representation, then banker's rules)
    while the paper's emitter prints 70.6, so the graphical abstract and Table 4
    disagreed on the last digit of the same interval.
    """
    q = decimal.Decimal(str(x)).quantize(
        decimal.Decimal(1).scaleb(-nd), rounding=decimal.ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def fig_graphical_abstract(out_png: Path, out_pdf: Path) -> None:
    fig = plt.figure(figsize=(GA_W_MM / 25.4, GA_H_MM / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def Y(mm: float) -> float:
        return mm / GA_H_MM

    for xdiv in (0.336, 0.668):
        ax.plot([xdiv, xdiv], [Y(4.6), Y(48.4)], color=GRID, lw=0.8)

    def head(x: float, letter: str, title: str, sub: str, lim: float) -> None:
        _t(
            ax,
            x,
            Y(49.2),
            f"{letter}  {title}",
            size=7.0,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
            limit=lim,
        )
        _t(
            ax,
            x,
            Y(46.2),
            sub,
            size=6.0,
            color=SEC,
            ha="left",
            va="top",
            linespacing=1.35,
            limit=lim,
        )

    # ================= panel A: two vintages, one gate ======================
    head(
        0.010,
        "A",
        "Two vintages, one gate",
        f"MSM dissemination\n"
        f"+{DISSEM_LAG_H} h {DISSEM_LAG_M} m measured\n"
        f"({DISSEM_DAYS} d; spec {SPEC_LAG_H} h {SPEC_LAG_M} m)",
        0.330,
    )
    # D−2 23:00 .. D−1 12:30 JST, in hours from D−1 00:00 (same convention as
    # the roadmap): the pre-gate tier is in hand by 02:00 JST, not 08:30
    ta0, ta1 = 0.030, 0.310
    ha0, ha1 = -1.0, 12.5

    def T(hour: float) -> float:
        return ta0 + (hour - ha0) / (ha1 - ha0) * (ta1 - ta0)

    xg = T(GATE_HOUR)
    # stops above the 11:30 stamp, which now sits close to the gate line
    ax.plot([xg, xg], [Y(20.8), Y(37.6)], color=C_GATE, lw=1.6, zorder=3)
    _t(
        ax,
        xg,
        Y(38.0),
        "10:00 gate",
        size=6.2,
        fontweight="bold",
        color=C_GATE,
        ha="center",
        va="bottom",
        limit=0.332,
    )
    lanes = [
        # Panel B attaches the HEADLINE improvement to this lane, and the headline
        # is the pinned D-2 12 UTC run, so the lane must carry that run's clock.
        # It previously carried the retired composite tier's (<= 02:00, +8 h) on
        # the grounds that the label named the tier -- but "pre-gate vintage" beside
        # 0.253 reads as the headline, and a referee read it that way.
        ("pre-gate vintage", C_PRE, D2_12Z_INIT, D2_12Z_DISSEM, "23:30", "✓", GOOD_TXT, 34.0),
        ("first post-gate run", C_POST, POST_INIT, POST_DISSEM, "11:30", "✗", C_GATE, 23.0),
    ]
    for name, col, t_i, t_d, stamp, glyph, vcol, ymm in lanes:
        y = Y(ymm)
        _t(
            ax,
            ta0,
            Y(ymm + 2.4),
            name,
            size=6.2,
            fontweight="bold",
            color=INK,
            ha="left",
            va="bottom",
            limit=0.332,
        )
        ax.plot([ta0, ta1], [y, y], color=GRID, lw=0.8, zorder=1)
        ax.add_patch(
            Rectangle(
                (T(t_i), y - 0.012),
                T(t_d) - T(t_i),
                0.024,
                facecolor=_tint(col, 0.55),
                edgecolor="none",
                zorder=2,
            )
        )
        ax.plot([T(t_i)], [y], marker="o", ms=4.0, mfc="white", mec=col, mew=1.0, zorder=4)
        ax.plot([T(t_d)], [y], marker="o", ms=5.0, color=col, mec="white", mew=0.9, zorder=4)
        _t(
            ax,
            T(t_d),
            Y(ymm - 2.6),
            stamp,
            size=6.0,
            fontweight="bold",
            color=INK,
            ha="center",
            va="top",
        )
        _t(ax, 0.322, y, glyph, size=8.0, fontweight="bold", color=vcol, ha="center", va="center")
    _t(
        ax,
        ta0,
        Y(15.4),
        "only the pre-gate vintage is\nin the market at gate closure",
        size=6.2,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        linespacing=1.35,
        limit=0.332,
    )
    _t(
        ax,
        ta0,
        Y(8.0),
        "○ initialization    ● disseminated",
        size=6.0,
        color=MUT,
        ha="left",
        va="center",
        limit=0.332,
    )

    # ================= panel B: the vintage ladder ==========================
    head(
        0.345, "B", "The value is pre-gate", "improvement over a price-only\nmodel, JPY/kWh", 0.664
    )
    base_mm, span_mm, vmax = 16.4, 21.0, 0.30

    def V(v: float) -> float:
        return Y(base_mm + v / vmax * span_mm)

    ax.plot([0.360, 0.650], [Y(base_mm), Y(base_mm)], color=AXIS, lw=0.8)
    bars = [
        (0.395, IMP_STALE, C_NEUTRAL, "stale\nobservation"),
        (0.500, IMP_PRE, C_PRE, "pre-gate\nvintage"),
        (0.605, IMP_POST, C_POST, "post-gate\nrun"),
    ]
    for bx, (val, lo, hi), col, lab in bars:
        _bar(ax, bx, Y(base_mm), V(val), col)
        ax.plot([bx, bx], [V(lo), V(hi)], color=SEC, lw=0.8, zorder=4)
        for yv in (lo, hi):
            ax.plot([bx - 0.008, bx + 0.008], [V(yv), V(yv)], color=SEC, lw=0.8, zorder=4)
        _t(
            ax,
            bx,
            V(hi) + Y(0.9),
            f"{val:.3f}",
            size=6.4,
            fontweight="bold",
            color=INK,
            ha="center",
            va="bottom",
        )
        _t(
            ax,
            bx,
            Y(base_mm - 1.1),
            lab,
            size=6.0,
            color=SEC,
            ha="center",
            va="top",
            linespacing=1.30,
        )
    # 5.31x: the measured span between the stale and pre-gate tier levels
    xm = 0.4475
    for bx, v in ((0.395, IMP_STALE[0]), (0.500, IMP_PRE[0])):
        ax.plot(
            [min(bx + 0.0225, xm), max(bx - 0.0225, xm)],
            [V(v), V(v)],
            color=AXIS,
            lw=0.6,
            zorder=1,
        )
    _arrow(ax, xm, V(IMP_STALE[0]), xm, V(IMP_PRE[0]), color=INK, lw=0.9, style="<|-|>", ms=5)
    _t(
        ax,
        xm,
        (V(IMP_STALE[0]) + V(IMP_PRE[0])) / 2,
        f"+{PRE_MINUS_STALE[0]:.3f}",
        size=6.4,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
        bbox=WBOX,
        zorder=5,
    )
    _t(
        ax,
        0.505,
        Y(8.2),
        "excluded run: equivalent (24-h pooled)",
        size=6.2,
        color=SEC,
        ha="center",
        va="center",
        limit=0.664,
    )

    # ================= panel C: two targets, measured =======================
    # The post-gate run keeps improving the PHYSICAL forecast while leaving the
    # price equivalent within a tick. Two different units, so two labelled
    # mini-scales rather than one axis, and a third row for the mechanism test
    # that says how far apart the two targets actually are.
    head(
        0.678,
        "C",
        "Two targets, measured",
        "the excluded run,\nphysical vs price",
        0.997,
    )
    row_y = (34.5, 19.5)
    # --- row 1: the physical target (MW of area photovoltaic output) --------
    _t(
        ax,
        0.690,
        Y(row_y[0] + 5.2),
        "physical: PV potential (MW)",
        size=6.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
        limit=0.997,
    )
    px0, px1 = 0.700, 0.930
    # The row is a mini-scale anchored on the two vintages it compares: the
    # pre-gate run sits at the left end and the axis runs a little past the
    # post-gate run, so the shaded gain occupies most of the row. The former
    # version hardcoded 900 MW as the right end, which inverted the scale once
    # phase89's common-panel levels fell below it.
    _phys_span = PHYS_PRE - PHYS_POST
    _phys_right = PHYS_POST - 0.30 * _phys_span

    def PX(mw: float) -> float:
        return px0 + (PHYS_PRE - mw) / (PHYS_PRE - _phys_right) * (px1 - px0)

    ax.plot([px0, px1], [Y(row_y[0]), Y(row_y[0])], color=GRID, lw=0.8, zorder=1)
    ax.add_patch(
        Rectangle(
            (PX(PHYS_PRE), Y(row_y[0]) - 0.014),
            PX(PHYS_POST) - PX(PHYS_PRE),
            0.028,
            facecolor=_tint(C_POST, 0.55),
            edgecolor="none",
            zorder=2,
        )
    )
    for mw, col in ((PHYS_PRE, C_PRE), (PHYS_POST, C_POST)):
        ax.plot([PX(mw)], [Y(row_y[0])], marker="o", ms=4.6, color=col,
                mec="white", mew=0.8, zorder=4)
    _t(
        ax,
        0.700,
        Y(row_y[0] - 4.4),
        f"improves {_hu(PHYS_DELTA[0])} MW "
        f"[{_hu(PHYS_DELTA[1])}, {_hu(PHYS_DELTA[2])}]",
        size=6.0,
        fontweight="bold",
        color=GOOD_TXT,
        ha="left",
        va="center",
        limit=0.997,
    )
    # --- row 2: the price target (JPY/kWh) ---------------------------------
    _t(
        ax,
        0.690,
        Y(row_y[1] + 5.2),
        "price target: the ladder panel",
        size=6.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
        limit=0.997,
    )
    zx0, zx1 = 0.700, 0.930
    zc = (zx0 + zx1) / 2
    tick_half = (zx1 - zx0) / 2  # one JEPX tick, the equivalence margin
    ax.add_patch(
        Rectangle(
            (zx0, Y(row_y[1]) - 0.020),
            zx1 - zx0,
            0.040,
            facecolor=FILL_LIGHT,
            edgecolor="none",
            zorder=1,
        )
    )
    ax.plot([zc, zc], [Y(row_y[1]) - 0.024, Y(row_y[1]) + 0.024], color=AXIS,
            lw=0.8, zorder=2)
    gv, glo, ghi = GATE_VALUE
    ax.plot(
        [zc + glo / 0.01 * tick_half, zc + ghi / 0.01 * tick_half],
        [Y(row_y[1]), Y(row_y[1])],
        color=SEC, lw=1.1, zorder=3,
    )
    ax.plot([zc + gv / 0.01 * tick_half], [Y(row_y[1])], marker="o", ms=4.6,
            color=C_POST, mec="white", mew=0.8, zorder=4)
    _t(
        ax,
        0.700,
        Y(row_y[1] - 5.0),
        f"24-h pooled: equivalent within\nthe ±1 tick band (p = {GATE_TOST_P:.3f})",
        size=6.2,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
        limit=0.997,
    )
    _t(
        ax,
        0.690,
        Y(7.2),
        # Was the variance-share pair (14.0% vs 1.2%). Three referees objected that
        # shares of two different dependent variables cannot be divided, and the
        # commensurable pair -- each target's own error reduction -- is both correct
        # and four times the contrast. \PhysRelPooled / \PriceRelReduction.
        f"error falls {PHYS_REL_PCT:.1f}% physical\n"
        f"vs {PRICE_REL_PCT:.2f}% price\n"
        f"({ASYM_RATIO:.0f}x, each in its own units)",
        size=6.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
        limit=0.997,
    )

    # ---- footer: the sample every panel shares -----------------------------
    _t(
        ax,
        0.5,
        Y(2.0),
        f"{N_DAYS} daily walk-forward folds  ·  {N_AREAS} JEPX areas  ·  "
        f"panel C: {PHYS_DAYS} of those folds, the days the published PV actuals cover",
        size=6.0,
        color=MUT,
        ha="center",
        va="center",
        limit=0.999,
    )

    _audit(fig, ax, min_pt=6.0, scale=1.0)
    # Elsevier GA spec: >= 531 x 1328 px (h x w); 13 x 5 cm at 300 dpi
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    log.info("wrote %s and %s", out_png, out_pdf)


# ---------------------------------------------------------------------------
# Figure 3: simplified skill-scarcity redraw (fig:skill_scarcity)
# ---------------------------------------------------------------------------
def fig_skill_scarcity(p36: dict, p35: dict, out: Path) -> None:
    curve = p36["curve"]  # committed noise-degradation sweep (no recomputation)
    dc_x = [r["skill_corr"] for r in curve]
    dc_val = [r["mean_improvement_yen_kwh"] for r in curve]
    dc_rho = [r["concentration_rho_vs_pvpeak"] for r in curve]

    # Actual information sets -- macros.tex: \SkillProxy 0.57, \SkillHistfc 0.89
    # (midday forecast-vs-realised correlation); \ImpProxy 0.034, \ImpOps 0.239,
    # \ImpOracle 0.268 (phase27.1). Realised weather sits at correlation 1 by
    # construction.
    info_x = [0.57, 0.89, 1.00]
    info_val = [0.034, 0.239, 0.268]
    info_col = [C_PROXY, C_MSM, C_BASE]
    # Concentration points -- proxy: macros.tex \NoiselessSpearman -0.733;
    # operational: phase35 histfc rho (= \OpsCrossArea -0.29 convention).
    rho_hist = p35["cross_area_spearman_vs_pvpeak"]["histfc_reconstruction"]["rho"]
    conc_x = [0.57, 0.89]
    conc_y = [-0.733, rho_hist]
    conc_col = [C_PROXY, C_MSM]
    # Noise-degraded proxy -- macros.tex \StressSpearman -0.267 /
    # \RealisticSpearman -0.450 / \NoiselessSpearman -0.733 (phase24.11.8);
    # x placement is the ordinal effective-accuracy position of the committed
    # EE figure (94_figures_theory.py), retained unchanged.
    deg_x = [0.30, 0.43, 0.57]
    deg_y = [-0.267, -0.450, -0.733]
    # Two-day-lagged proxy (D-2) -- macros.tex \BcadmSpearman -0.300 at
    # accuracy 0.546 (phase27.0 probe_s_skill_axis.skill_lag96_corr).
    lag96 = (0.546, -0.300)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.48, 3.0))

    # ---- (a) mean value vs forecast accuracy ------------------------------
    axa.plot(
        dc_x,
        dc_val,
        "-o",
        color=C_GPV,
        lw=2.2,
        ms=3.5,
        alpha=0.35,
        zorder=2,
        label="operational forecast + added noise\n(controlled degradation)",
    )
    axa.plot(info_x, info_val, "-", color=AXIS, lw=1.0, zorder=1)
    # identity of the three one-point series is carried by the legend (the
    # top-right corner is too crowded for non-colliding direct labels)
    info_lab = ["day-lagged proxy", "operational forecast", "realised weather (reference)"]
    for x, y, col, lab in zip(info_x, info_val, info_col, info_lab, strict=True):
        axa.plot([x], [y], "o", color=col, ms=7, mec="white", mew=0.9, zorder=3, label=lab)
    axa.set_xlim(0.10, 1.10)
    axa.set_ylim(0.0, 0.30)
    axa.set_xlabel("forecast accuracy (midday correlation with realised weather)", fontsize=7)
    axa.set_ylabel("mean weather value:\nMAE improvement (JPY/kWh)", fontsize=7.5)
    axa.set_title("(a) More accurate forecasts are worth more", fontsize=8.5, color=INK)
    axa.legend(fontsize=6.2, loc="upper left", frameon=False)

    # ---- (b) cross-area concentration vs forecast accuracy ----------------
    axb.axhline(0.0, color=AXIS, lw=0.8)
    axb.plot(
        dc_x,
        dc_rho,
        "o",
        color=C_GPV,
        ms=4.5,
        mfc="none",
        mew=1.1,
        alpha=0.6,
        label="operational forecast + added noise",
    )
    axb.plot(
        deg_x,
        deg_y,
        "^",
        color=C_PROXY,
        ms=5.5,
        mfc="none",
        mew=1.1,
        label="day-lagged proxy + extra noise",
    )
    axb.plot(
        [lag96[0]],
        [lag96[1]],
        "D",
        color=C_PROXY,
        ms=5.5,
        mfc="none",
        mew=1.1,
        label="two-day-lagged proxy (D−2)",
    )
    for x, y, col in zip(conc_x, conc_y, conc_col, strict=True):
        axb.plot([x], [y], "s", color=col, ms=7.5, mec="white", mew=0.9, zorder=3)
    axb.annotate(
        "day-lagged proxy",
        (conc_x[0], conc_y[0]),
        textcoords="offset points",
        xytext=(-10, -3),
        fontsize=7,
        color=SEC,
        ha="right",
        va="center",
        bbox=WBOX,
    )
    axb.annotate(
        "operational forecast",
        (conc_x[1], conc_y[1]),
        textcoords="offset points",
        xytext=(6, -11),
        fontsize=7,
        color=SEC,
        ha="right",
    )
    axb.text(
        0.56,
        0.062,
        "ρ near zero: value spread evenly across areas",
        fontsize=6.5,
        color=MUT,
        ha="center",
        va="center",
        style="italic",
    )
    axb.text(
        0.115,
        -0.845,
        "more negative: value concentrated in floor-prone areas",
        fontsize=6.3,
        color=MUT,
        ha="left",
        va="center",
        style="italic",
    )
    axb.set_xlim(0.10, 1.02)
    axb.set_ylim(-0.87, 0.12)
    axb.set_xlabel("forecast accuracy (midday correlation with realised weather)", fontsize=7)
    axb.set_ylabel("cross-area concentration of\nweather value (Spearman ρ)", fontsize=7.5)
    axb.set_title("(b) Concentration fades as accuracy rises", fontsize=8.5, color=INK)
    axb.legend(fontsize=6.2, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.16))

    for ax in (axa, axb):
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    log.info("wrote %s", out)


def _check_provenance() -> None:
    """Assert the hardcoded headline numbers still match their JSON fields."""
    # The ladder is now phase59's run-pinned one; phase51 remains only as the
    # dissemination-clock source and the Gate 5 reproducibility reference.
    p51_all = _load_json("phase51_vintage_ladder.json")
    p85 = _load_json("phase85_exact_ladder.json")
    cw = p85["arm_cw_primary"]
    tiers = cw["tiers"]
    dec = cw["gate46_47_48_decomposition"]
    p89 = _load_json("phase89_physical_target.json")
    p89t = p89["targets"]["potential"]["tiers"]
    p89g = p89["gates"]["gate51_physical_gain_crossfit_common"]
    p90r = _load_json("phase90_mechanism.json")["regressions"]
    p54 = _load_json("phase54v2_current.json")["results"]
    checks = [
        ("stale", IMP_STALE[0], tiers["era5_cw"]["improvement"]),
        ("pre-gate", IMP_PRE[0], tiers["msm_d2_12z_cw"]["improvement"]),
        ("composite", IMP_COMPOSITE[0], tiers["msm_gate_composite_cw"]["improvement"]),
        ("post-gate", IMP_POST[0], tiers["msm_d1_00z_cw"]["improvement"]),
        ("ex-post", IMP_EXPOST, tiers["oracle_cw"]["improvement"]),
        ("multiple", PRE_MULTIPLE[0], dec["headline_vs_stale"]["ratio"]["ratio"]),
        ("pre minus stale", PRE_MINUS_STALE[0],
         dec["headline_vs_stale"]["difference"]["delta"]),
        ("gate value", GATE_VALUE[0], cw["gate46_47_48_gate_timing"]["delta"]),
        (
            # The PRIMARY test (HLN-DM at L=3), which is what the manuscript
            # leads with. This check used to read the "boot" branch, so the
            # figure rendering the bootstrap value under a label naming the
            # primary one passed: the check agreed with the mistake.
            "gate TOST p (primary, dm_l3)",
            GATE_TOST_P,
            cw["gate46_47_48_gate_timing"]["equivalence"]["margins"]["tick"][
                "dm_l3"]["p_tost"],
        ),
        (
            "gate TOST p (bootstrap)",
            GATE_TOST_P_BOOT,
            cw["gate46_47_48_gate_timing"]["equivalence"]["margins"]["tick"][
                "boot"]["p_tost"],
        ),
        ("physical pre", PHYS_PRE,
         p89t["msm_d2_12z_cw"]["pooled_mae_cf_common_mw"]),
        ("physical post", PHYS_POST,
         p89t["msm_d1_00z_cw"]["pooled_mae_cf_common_mw"]),
        ("physical delta", PHYS_DELTA[0], p89g["delta_mw"]),
        # The ENDPOINTS too. Only the point estimate was checked, so the
        # interval this figure prints could drift from the table's freely.
        ("physical delta lo", PHYS_DELTA[1], p89g["ci95"][0]),
        ("physical delta hi", PHYS_DELTA[2], p89g["ci95"][1]),
        # The dissemination lag the panel A subtitle states as measured.
        ("spec lag (min)", SPEC_LAG_MIN,
         _load_json("phase72_dissemination_audit.json")["interpretation"][
             "spec_lag_minutes"]),
        ("dissemination lag (min)", DISSEM_LAG_MIN,
         _load_json("phase72_dissemination_audit.json")["per_run"][
             "msm_d2_12z"]["per_run_completion_lag_minutes"]["median"]),
        ("physical days", PHYS_DAYS, p89g["n_days"]),
        ("mechanism physical share", MECH_PHYS_R,
         100 * p90r["physical_residual_on_innovation"]["r2_increment"]),
        ("mechanism price share", MECH_PRICE_R,
         100 * p90r["price_residual_on_innovation"]["r2_increment"]),
        (
            "enriched",
            PRE_OVER_ENRICHED,
            p54["deltas_daily_mean_with_ci95"]["d42_total_prev2_over_enriched"][
                "point_daily_mean"
            ],
        ),
    ]
    for name, shown, raw in checks:
        tol = max(abs(raw) * 0.006, 0.0006)  # display rounding only
        if abs(shown - raw) > tol:
            raise AssertionError(f"{name}: figure shows {shown}, JSON has {raw}")

    # ---- exact matches: sample sizes, counts and rounded integers ----------
    p259 = _load_json("phase25.9_floor_probit_summary.json")
    rr = [r["risk_ratio_curtail_vs_no"] for r in p259["floor_x_curtail_crosstab"]]
    # the HOLM-adjusted count, which is what Methods promises and what Section
    # 5.1 now reports; the raw count is 8 and this figure used to print it
    stale_sig = tiers["era5_cw"]["areas_sig_l3_holm"]
    exact = [
        ("days", N_DAYS, p85["panel"]["n_test_days"]),
        ("areas", N_AREAS, p85["panel"]["n_areas"]),
        ("slots", int(N_SLOTS.replace(",", "")), p85["panel"]["n_slots"]),
        # card A: DM-significant areas, stale tier, at both NW bandwidths
        ("stale DM areas", STALE_DM_AREAS, f"{stale_sig} of {p85['panel']['n_areas']}"),
        # card B: the share of the excluded run's level the gate forgoes. Gate 47
        # landed on 47b, so this is an equivalence statement, not an effect.
        ("gate share", GATE_SHARE,
         round(100 - cw["headline_retention_of_post_gate_pct"])),
        ("retention", PRE_RETENTION,
         round(cw["headline_retention_of_post_gate_pct"])),
        # card D: near-floor risk-ratio range, printed floor-to-ceiling
        ("curtail RR low", CURTAIL_RR_LO, int(min(rr))),
        ("curtail RR high", CURTAIL_RR_HI, round(max(rr))),
        ("curtail areas", CURTAIL_AREAS, len(p259["areas"])),
        ("curtail slots", int(CURTAIL_SLOTS.replace(",", "")), p259["n_obs_total"]),
    ]
    for name, shown, raw in exact:
        if shown != raw:
            raise AssertionError(f"{name}: figure shows {shown!r}, JSON has {raw!r}")

    # ---- the dissemination clock the two tier panels draw ------------------
    # Times are strings in the provenance file, so assert that the instants and
    # margins the figures print are literally the ones recorded there.
    diss = p51_all["dissemination_accounting"]
    tiers = diss["tiers"]

    def _hhmm(h: float) -> str:
        """Hours from D−1 00:00 JST -> wall clock (−0.5 -> '23:30' on D−2)."""
        m = round(h * 60) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"

    texts = [
        ("gate", f"D-1 {_hhmm(GATE_HOUR)} JST", diss["gate"]),
        # pre-gate TIER: disseminated <= D-1 02:00 JST, >= +8 h before the gate
        (
            "pre-gate dissemination",
            f"D-1 {_hhmm(PRE_DISSEM)} JST",
            tiers["prev2_pre_gate"]["disseminated"],
        ),
        ("pre-gate init", f"D-2 {_hhmm(PRE_INIT)} JST", tiers["prev2_pre_gate"]["init"]),
        ("pre-gate margin", f"+{PRE_MARGIN_H}h00m", tiers["prev2_pre_gate"]["margin_vs_gate"]),
        # first post-gate tier: the D-1 00 UTC run, 11:30 JST, 90 min late
        (
            "post-gate dissemination",
            f"D-1 {_hhmm(POST_DISSEM)} JST",
            tiers["gpv_d1_00utc_paper_tier"]["disseminated"],
        ),
        (
            "post-gate margin",
            f"-{int(POST_DISSEM - GATE_HOUR)}h{round((POST_DISSEM - GATE_HOUR) % 1 * 60):02d}m",
            tiers["gpv_d1_00utc_paper_tier"]["margin_vs_gate"],
        ),
    ]
    for name, needle, haystack in texts:
        if needle not in haystack:
            raise AssertionError(f"{name}: figure implies {needle!r}, JSON records {haystack!r}")
    log.info(
        "provenance: %d values + %d exact + %d timing strings match their JSON fields",
        len(checks),
        len(exact),
        len(texts),
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    _check_provenance()

    fig_roadmap(FIG_DIR / "analysis_roadmap.pdf")
    fig_graphical_abstract(SUB_DIR / "graphical_abstract.png", SUB_DIR / "graphical_abstract.pdf")

    log.info("done: 2 figures written (the legacy skill-scarcity redraw is not "
             "part of Paper A and is left to the phase50 script)")


if __name__ == "__main__":
    main()
