r"""Phase 83: bootstrap-dependence and equivalence-margin sensitivity.

Precommit: docs/research/2026-08-11_phase83_inference_sensitivity_precommit.md.
The one-tick equivalence p sits near 0.02, so the pre-registered rule is: the
manuscript states equivalence unconditionally ONLY if the TOST at 0.01 JPY/kWh
passes under circular blocks of 7, 14 and 28 days AND under a calendar-month
cluster bootstrap; otherwise the wording weakens to the CI display. This phase
runs that rule for the CW-primary gate-timing differential, adds a stationary
bootstrap at the Politis--White automatic block length, a permutation
(sign-flip) test, and traces the equivalence margin curve over
{0.005, 0.01, 0.02, 0.05} and relative margins {10%, 20% of the headline}.

CLI (from the repository root):
  uv run --extra research python scripts/62_inference_sensitivity.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase83")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PANEL = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase83_inference_sensitivity.json"
PRECOMMIT = "docs/research/2026-08-11_phase83_inference_sensitivity_precommit.md"

SEED, N_BOOT, ALPHA = 2026, 10_000, 0.05
HEADLINE, POST = "msm_d2_12z_cw", "msm_d1_00z_cw"
BLOCKS = (7, 14, 28)
MARGINS_ABS = (0.005, 0.01, 0.02, 0.05)
MARGINS_REL = (0.10, 0.20)


def _j(x: float | None, nd: int = 5) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _circular_draws(d: np.ndarray, block: int, seed: int = SEED) -> np.ndarray:
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)


def _stationary_draws(d: np.ndarray, mean_block: float, seed: int = SEED) -> np.ndarray:
    n = len(d)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((N_BOOT, n), dtype=np.int64)
    pos = rng.integers(0, n, size=N_BOOT)
    for t in range(n):
        idx[:, t] = pos
        restart = rng.random(N_BOOT) < p
        pos = np.where(restart, rng.integers(0, n, size=N_BOOT), (pos + 1) % n)
    return d[idx].mean(axis=1)


def _month_cluster_draws(d: np.ndarray, months: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(months)
    groups = [d[months == m] for m in uniq]
    k = len(groups)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, k, size=k)
        out[b] = np.concatenate([groups[i] for i in pick]).mean()
    return out


def _politis_white_block(d: np.ndarray) -> float:
    """Politis-White (2004) automatic block length (flat-top lag window)."""
    n = len(d)
    x = d - d.mean()
    kn = max(5, int(np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    rho = np.array([1.0] + [float(x[:-k] @ x[k:] / (x @ x)) for k in range(1, m_max + 1)])
    band = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(1, m_max - kn + 1):
        if np.all(np.abs(rho[m:m + kn]) < band):
            m_hat = m
            break
    m_hat = m_hat or m_max
    mm = min(2 * m_hat, m_max)
    lam = np.array([1.0 if abs(k) / mm <= 0.5 else 2.0 * (1 - abs(k) / mm)
                    for k in range(-mm, mm + 1)])
    ks = np.arange(-mm, mm + 1)
    r = np.array([rho[abs(k)] * float(x @ x) / n for k in ks])
    g_hat = float(np.sum(lam * np.abs(ks) * r))
    d_sb = 2.0 * float(np.sum(lam * r)) ** 2
    if d_sb <= 0:
        return 7.0
    b = ((2.0 * g_hat**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, n / 3.0))


def _tost(draws: np.ndarray, margin: float) -> dict:
    p = max(float(np.mean(draws <= -margin)), float(np.mean(draws >= margin)))
    return {"p_tost": _j(max(p, 1.0 / N_BOOT)), "equivalent": bool(p < ALPHA)}


def main() -> int:
    panel = pl.read_parquet(PANEL)
    cw_flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    panel = panel.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
    cells = (HEADLINE, POST, "price_only")
    per_day = (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .group_by("test_day")
        .agg(*[pl.col(f"mae_{c}").mean() for c in cells])
        .sort("test_day"))
    d = np.asarray(per_day[f"mae_{HEADLINE}"] - per_day[f"mae_{POST}"], dtype=float)
    headline_imp = float((per_day["mae_price_only"] - per_day[f"mae_{HEADLINE}"]).mean())
    months = np.asarray(per_day["test_day"].str.slice(0, 7).to_list())

    out: dict = {"phase": 83, "title": "Inference sensitivity: blocks, clusters, margins",
                 "precommit_doc": PRECOMMIT,
                 "differential": "daily pooled MAE(headline_cw) - MAE(post_cw); "
                                 "positive means the post-gate run is better",
                 "point": _j(float(d.mean())), "n_days": len(d),
                 "headline_improvement": _j(headline_imp), "schemes": {}}

    pw = _politis_white_block(d)
    out["politis_white_block_days"] = _j(pw, 2)
    schemes = {f"circular_b{b}": _circular_draws(d, b) for b in BLOCKS}
    schemes["stationary_pw"] = _stationary_draws(d, pw)
    schemes["month_cluster"] = _month_cluster_draws(d, months)

    margins = [("abs_" + str(m), m) for m in MARGINS_ABS]
    margins += [(f"rel_{int(r * 100)}pct", r * headline_imp) for r in MARGINS_REL]
    for name, draws in schemes.items():
        rec = {"ci95": [_j(float(np.quantile(draws, 0.025))),
                        _j(float(np.quantile(draws, 0.975)))],
               "margins": {mn: {"delta_margin": _j(mv), **_tost(draws, mv)}
                           for mn, mv in margins}}
        out["schemes"][name] = rec
        log.info("%-14s CI [%s, %s]  tick-TOST p=%s", name, rec["ci95"][0],
                 rec["ci95"][1], rec["margins"]["abs_0.01"]["p_tost"])

    # permutation (sign-flip) test of the mean differential
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, len(d)))
    perm = (signs * d[None, :]).mean(axis=1)
    p_perm = float(np.mean(np.abs(perm) >= abs(d.mean())))
    out["permutation_signflip"] = {"p": _j(max(p_perm, 1.0 / N_BOOT))}

    # the pre-registered wording rule
    required = ["circular_b7", "circular_b14", "circular_b28", "month_cluster"]
    holds = all(out["schemes"][s]["margins"]["abs_0.01"]["equivalent"]
                for s in required)
    out["wording_rule"] = {
        "unconditional_equivalence_permitted": holds,
        "rule": "tick TOST must pass under circular blocks 7/14/28 AND the "
                "month cluster; otherwise the abstract drops the equivalence "
                "word and shows the CI",
        "verdict_inputs": {s: out["schemes"][s]["margins"]["abs_0.01"]["equivalent"]
                           for s in required},
    }
    log.info("WORDING RULE: unconditional equivalence %s",
             "PERMITTED" if holds else "NOT permitted")

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "n_boot": N_BOOT}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
