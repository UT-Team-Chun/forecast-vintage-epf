r"""Round-28 manuscript figures: VoI ridge (schematic) + skill-scarcity (data).

(1) fig:voi_ridge -- a SCHEMATIC illustration of the threshold-crossing
    value-of-information surface VoI(z, kappa^pv) from Section 3, at low vs
    high forecast skill. Purely illustrative of the model's SIGNS (single-peak
    at the floor-entry boundary z=0; ridge height rising in kappa^pv; the
    high-skill ridge taller but broader/flatter = washed-out concentration).
    No data fit.

(2) fig:skill_scarcity -- the HONEST dual-message skill-scarcity figure from
    the committed empirics (phase27.0/27.1 + noise-degradation macros): total
    weather value rises monotonically with forecast skill, while the cross-area
    concentration collapses from the lagged proxy to the operational forecast.
    The points are the real information sets; we do NOT fit a concentration(tau)
    curve (the controlled synthetic sweep, phase27.2, reproduced the sign but
    not a clean monotone curve -- see the research log).

Writes PDFs to the paper figures dir. CLI:
  cd <repository root>
  uv run python scripts/94_figures_theory.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless backend before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from alg.config import EpfConfig

log = logging.getLogger("phase27.4")
# Both outputs are supplement figures the article carries, so this script has to
# resolve the same two layouts its siblings do: the manuscript tree when there
# is one, else ./figures. Counting parent directories fixed it to this
# checkout's depth, which is why the public replication package shipped both
# PDFs with no producer -- the guard then certified two figures nothing here
# could regenerate.
_ROOT = EpfConfig().project_root
_MS = _ROOT / "docs/paper/paper_a_main"
FIG_DIR = (_MS / "figures") if (_MS / "main.tex").exists() else (_ROOT / "figures")


def _voi(z: np.ndarray, kappa: float, tau: float) -> np.ndarray:
    """Schematic VoI = V_reg(tau) Psi(z) + V_slope(tau) (kappa)^2 Phi(z) + b(tau).

    Psi(z)=Phi(z)(1-Phi(z)) is the Bernoulli crossing (regime-resolution) kernel
    (single-peaked at z=0); the within-branch slope piece is monotone Phi(z);
    b(tau) is the kappa-free demand/wind channel that switches on at high skill
    (so the high-tau ridge is taller but less kappa-differentiated).
    """
    psi = norm.cdf(z) * (1.0 - norm.cdf(z))
    v_reg = tau
    v_slope = 0.8 * tau
    b = 0.9 * max(tau - 0.6, 0.0) ** 1.5  # kappa-free channel, late switch-on
    return v_reg * psi + v_slope * (kappa**2) * norm.cdf(z) + b


def voi_ridge_figure(out: Path) -> None:
    z = np.linspace(-3.2, 3.2, 200)
    kappas = np.linspace(0.25, 0.92, 120)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (corr, lab) in zip(
        axes, [(0.57, r"low skill, corr$(S,W)\approx0.57$ (lagged-reanalysis proxy)"),
               (0.89, r"high skill, corr$(S,W)\approx0.89$ (operational forecast)")], strict=True,
    ):
        tau = corr ** 2  # model skill tau = corr^2; the panels are captioned by corr
        grid = np.array([[_voi(np.array([zz]), kk, tau)[0] for zz in z] for kk in kappas])
        im = ax.contourf(z, kappas, grid, levels=18, cmap="viridis")
        ax.axvline(0.0, color="white", ls="--", lw=1.1)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel(r"standardized distance to floor-entry threshold $z=(R_0-R^*)/\sigma_R$",
                      fontsize=8)
        ax.annotate("floor branch\n" + r"$\partial p/\partial W=0$, VoI$\to0$",
                    xy=(-2.6, 0.30), fontsize=6.5, color="white", ha="left", va="center")
        ax.annotate("deep-normal\nVoI$\\to0$", xy=(1.7, 0.30), fontsize=6.5,
                    color="white", ha="left", va="center")
        ax.annotate(r"$R^*$", xy=(0.05, 0.88), fontsize=8, color="white")
        fig.colorbar(im, ax=ax, shrink=0.85, label="VoI (schematic)")
    axes[0].set_ylabel(r"PV exposure $\kappa^{\mathrm{pv}}_a$ (PV/peak)", fontsize=8)
    fig.suptitle(
        "Value of weather information peaks at the floor-ENTRY boundary; "
        "skill raises the ridge but broadens it (concentration washes out)",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def skill_scarcity_figure(out: Path) -> None:
    # Real information-set points (committed provenance / macros).
    # Total value: mean MAE improvement (yen/kWh) -- phase27.1.
    tv_tau = [0.57, 0.89, 1.00]
    tv_val = [0.034, 0.239, 0.268]
    tv_lab = ["lagged proxy", "operational", "oracle"]
    # Concentration: cross-area FLOOR/PV-peak Spearman vs skill -- proxy (strong)
    # to operational (washout); noise-degradation low-skill side (same-signal).
    cc_tau = [0.57, 0.89]
    cc_rho = [-0.733, -0.286]
    cc_lab = ["lagged proxy", "operational"]
    # Noise-degradation low-skill side (degraded lag-48 proxy; ordinal skill).
    deg_rho = [-0.267, -0.450, -0.733]
    deg_x = [0.30, 0.43, 0.57]  # ordinal effective-skill placement (illustrative)
    # lag-96 (D-2) bid-close-admissible empirical point -- phase27.0
    # probe_s_skill_axis.skill_lag96_corr / concentration_slopes_for_figure.
    # It sits at nearly the same skill as the lag-48 proxy (0.546 vs 0.569) yet
    # its concentration attenuates to rho=-0.30 (= \BcadmSpearman): a *signal-
    # removal* effect (staler weather, autocorrelation timescale), a distinct
    # route from the skill axis along which the operational washout runs.
    lag96_x = 0.546
    lag96_rho = -0.300
    # Phase-36 controlled skill-degradation sweep: a SINGLE-source comparative-static
    # curve (GPV exact-issuance forecast + increasing Gaussian noise) -> skill_corr maps
    # to mean value and to cross-area concentration. Turns the discrete info-set points
    # into a continuous H1' curve. Skips gracefully if the provenance JSON is absent.
    import json as _json
    _p36 = next((d / "phase36_skill_degradation.json"
                 for d in (_ROOT / "data/runs", _ROOT / "data/provenance")
                 if (d / "phase36_skill_degradation.json").exists()), None)
    _curve = _json.loads(_p36.read_text())["curve"] if _p36 else []
    dc_skill = [r["skill_corr"] for r in _curve]
    dc_val = [r["mean_improvement_yen_kwh"] for r in _curve]
    dc_rho = [r["concentration_rho_vs_pvpeak"] for r in _curve]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    if dc_skill:
        axl.plot(dc_skill, dc_val, "-", color="#cccccc", lw=3, alpha=0.8, zorder=1,
                 label="GPV degraded by noise (single-source sweep)")
    axl.plot(tv_tau, tv_val, "o-", color="#1b7837", lw=1.6, ms=7, zorder=2)
    # Per-point label placement: the rightmost ("oracle", x=1.0) is left-aligned so it
    # does not overflow the right spine; extend the limits to give every label margin.
    _a_off = {"lagged proxy": (6, -9, "left"), "operational": (6, 4, "left"), "oracle": (-8, 3, "right")}
    for x, y, txt in zip(tv_tau, tv_val, tv_lab, strict=True):
        dx, dy, ha = _a_off.get(txt, (6, -2, "left"))
        axl.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, ha=ha)
    axl.set_xlim(0.12, 1.12)
    axl.set_ylim(0.0, 0.30)
    if dc_skill:
        axl.legend(fontsize=6.5, loc="lower right", frameon=False)
    axl.set_xlabel(r"forecast skill: midday corr(forecast, realised) $=\sqrt{\tau}$", fontsize=8)
    axl.set_ylabel(r"total weather value: mean $\Delta$MAE (JPY/kWh)", fontsize=8)
    axl.set_title("(a) Total value rises monotonically with skill", fontsize=9)
    axl.grid(alpha=0.3)

    # Markers only (no connecting line): the points are the real information
    # sets; we do NOT assert an interpolated concentration(tau) curve.
    axr.plot(cc_tau, cc_rho, "s", color="#762a83", ms=9, label="real information sets")
    axr.plot(deg_x, deg_rho, "^", color="#b35806", ms=6, alpha=0.8, mfc="none",
             label="degraded proxy (low-skill side)")
    axr.plot([lag96_x], [lag96_rho], "D", color="#1b9e77", ms=8, mfc="none", mew=1.6,
             label="lag-96 (D-2, signal-removed)")
    # Per-point labels: "operational" is the rightmost marker, so left-align it
    # to keep it off the right spine; "lagged proxy" goes up-right into open space.
    _b_off = {"lagged proxy": (-6, 6, "right"), "operational": (-8, 5, "right")}
    for x, y, txt in zip(cc_tau, cc_rho, cc_lab, strict=True):
        dx, dy, ha = _b_off.get(txt, (6, 4, "left"))
        axr.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, ha=ha)
    axr.annotate("lag-96 (D-2):\nsame skill, signal removed", (lag96_x, lag96_rho),
                 textcoords="offset points", xytext=(4, -22), fontsize=6.5, color="#1b9e77")
    axr.axhline(0.0, color="grey", lw=0.7)
    axr.set_xlim(0.22, 1.02)
    axr.set_ylim(-0.80, 0.06)
    axr.set_xlabel(r"forecast skill: corr(forecast, realised) (proxy $\approx0.57$, operational $\approx0.89$)",
                   fontsize=8)
    axr.set_ylabel(r"cross-area concentration (Spearman $\rho$)", fontsize=8)
    axr.set_title("(b) Concentration washes out as skill rises", fontsize=9)
    # Lower-right corner is the only region clear of all five markers and labels.
    axr.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    axr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _ = EpfConfig()  # ensure project import works
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # LaTeX-free text rendering (avoid requiring a TeX install for \textyen etc.)
    plt.rcParams.update({"text.usetex": False, "axes.unicode_minus": True})
    voi_ridge_figure(FIG_DIR / "voi_ridge.pdf")
    skill_scarcity_figure(FIG_DIR / "skill_scarcity.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
