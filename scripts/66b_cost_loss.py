r"""Phase 97: the floor forecasts' decision value, in Murphy's cost-loss frame.

WHY
---
Section 3 says that scoring floor events "follows the cost-loss tradition of
valuing event forecasts", citing Murphy (1977), and the supplement's theory refers
to a deployed asymmetric floor-decision loss. The main article never runs one: an
earlier exposure-based measure was withdrawn to a companion manuscript, and what
remained were rank and proper scores. A referee-simulation pass made this its
central objection for this journal -- there is no decision anywhere in the paper,
so nothing answers "who acts differently, and what is it worth?"

This closes that with the standard construction, on data already in hand.

An actor protects against a floor episode at cost C, and suffers loss L if an
episode occurs unprotected. With a probability forecast the optimal rule is to
protect when p exceeds alpha = C/L, so alpha indexes the actor: a cheap hedge is a
small alpha, an expensive one a large alpha. Murphy's value score compares the
forecast's expected expense against climatology and against a perfect forecast,

    V(alpha) = (E_clim - E_forecast) / (E_clim - E_perfect),

which is 1 for a perfect forecast, 0 for one no better than always/never
protecting, and negative for one that misleads. Reporting V over alpha says
exactly which actors gain and by how much, without our having to invent a
particular bidder's balance sheet.

The contrast that matters is m2 against m2p: the same classifier, the same folds,
with and without the pre-gate weather channels. Its difference in V is the decision
value of pre-gate weather for floor protection -- the paper's largest floor
contrast (+0.276 PR-AUC) expressed in a form an operator can act on.

CLI (from the repository root):
  uv run --extra research python scripts/66b_cost_loss.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase97")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase81_checkpoints"
OUT_JSON = RUN / "phase97_cost_loss.json"

THR = 0.5                      # the paper's near-floor band, JPY/kWh
MODELS = {"m2": "classifier with pre-gate weather",
          "m2p": "the same classifier without weather (price features only)",
          "m1": "logistic regression with pre-gate weather",
          "m4": "quantile regressor's implied floor probability"}
ALPHAS = np.round(np.arange(0.02, 0.99, 0.02), 2)
BOOT_BLOCK, N_BOOT, SEED = 7, 2_000, 2026


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _value_curve(lab: np.ndarray, p: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Murphy's value score at each cost-loss ratio, in units of L.

    Expenses are per slot: protecting costs alpha, an unprotected event costs 1.
    Climatology protects always when alpha < base rate and never otherwise; a
    perfect forecast protects exactly on the events.
    """
    s = float(lab.mean())
    out = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        act = p > a
        e_f = float(np.mean(np.where(act, a, lab)))
        e_c = min(a, s)
        e_p = a * s
        out[i] = (e_c - e_f) / (e_c - e_p) if e_c > e_p else np.nan
    return out


def main() -> int:
    files = sorted(CKPT.glob("*.parquet"))
    if not files:
        log.error("no checkpoints in %s", CKPT)
        return 2
    tag = f"{THR:g}"
    per_area_curves: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODELS}
    base_rates: dict[str, float] = {}
    day_of: dict[str, np.ndarray] = {}
    labs: dict[str, np.ndarray] = {}
    probs: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODELS}

    for f in files:
        area = f.stem
        cols = ["y_actual", "test_day"] + [f"{m}_{tag}" for m in MODELS
                                           if f"{m}_{tag}" in
                                           pl.read_parquet_schema(f)]
        d = pl.read_parquet(f, columns=cols)
        y = d["y_actual"].to_numpy()
        if not np.isfinite(y).all():
            msg = f"{area}: non-finite actual prices would be labelled non-floor"
            raise RuntimeError(msg)
        lab = (y <= THR).astype(float)
        if lab.sum() == 0:
            log.warning("%s has no floor slots; skipped", area)
            continue
        base_rates[area] = float(lab.mean())
        labs[area] = lab
        day_of[area] = d["test_day"].to_numpy()
        for m in MODELS:
            c = f"{m}_{tag}"
            if c not in d.columns:
                continue
            p = d[c].to_numpy().astype(float)
            probs[m][area] = p
            per_area_curves[m][area] = _value_curve(lab, p, ALPHAS)

    areas = sorted(labs)
    macro = {m: np.nanmean(np.vstack([per_area_curves[m][a] for a in areas
                                      if a in per_area_curves[m]]), axis=0)
             for m in MODELS if per_area_curves[m]}

    # block bootstrap of the m2 - m2p difference in V, resampling DAYS
    diff_ci: dict[str, list[float]] = {}
    if "m2" in macro and "m2p" in macro:
        rng = np.random.default_rng(SEED)
        all_days = sorted({d for a in areas for d in np.unique(day_of[a])})
        idx_of = {d: i for i, d in enumerate(all_days)}
        n_days = len(all_days)
        nb = int(np.ceil(n_days / BOOT_BLOCK))
        draws = np.empty((N_BOOT, len(ALPHAS)))
        day_idx = {a: np.array([idx_of[d] for d in day_of[a]]) for a in areas}
        for b in range(N_BOOT):
            starts = rng.integers(0, n_days, size=nb)
            drawn = ((starts[:, None] + np.arange(BOOT_BLOCK)[None, :])
                     % n_days).ravel()[:n_days]
            mult = np.bincount(drawn, minlength=n_days).astype(float)
            per = []
            for a in areas:
                w = mult[day_idx[a]]
                keep = w > 0
                if not keep.any():
                    continue
                lb = np.repeat(labs[a][keep], w[keep].astype(int))
                p2 = np.repeat(probs["m2"][a][keep], w[keep].astype(int))
                p2p = np.repeat(probs["m2p"][a][keep], w[keep].astype(int))
                per.append(_value_curve(lb, p2, ALPHAS)
                           - _value_curve(lb, p2p, ALPHAS))
            draws[b] = np.nanmean(np.vstack(per), axis=0)
        lo = np.nanquantile(draws, 0.025, axis=0)
        hi = np.nanquantile(draws, 0.975, axis=0)
        diff_ci = {f"{a:g}": [round(float(l), 4), round(float(h), 4)]
                   for a, l, h in zip(ALPHAS, lo, hi, strict=True)}

    def _pos_range(v: np.ndarray) -> list[float] | None:
        ok = np.where(np.nan_to_num(v, nan=-1) > 0)[0]
        return [float(ALPHAS[ok[0]]), float(ALPHAS[ok[-1]])] if len(ok) else None

    out = {
        "phase": "97",
        "title": "Decision value of the floor forecasts (Murphy cost-loss)",
        "why": ("Section 3 invokes the cost-loss tradition and the article never "
                "runs one, so nothing said which actors gain from the floor result "
                "or by how much"),
        "construction": {
            "rule": "protect when the forecast probability exceeds alpha = C/L",
            "value_score": "V = (E_clim - E_forecast) / (E_clim - E_perfect)",
            "units": "expenses in units of L; alpha indexes the actor",
            "threshold_jpykwh": THR,
            "alphas": [float(a) for a in ALPHAS],
            "macro": "value computed per area, then averaged across areas",
            "bootstrap": {"draws": N_BOOT, "block_days": BOOT_BLOCK, "seed": SEED},
        },
        "base_rate": {"per_area": {a: round(v, 4) for a, v in base_rates.items()},
                      "macro": round(float(np.mean(list(base_rates.values()))), 4)},
        "models": {m: {"description": MODELS[m],
                       "value_curve": {f"{a:g}": (round(float(v), 4)
                                                 if np.isfinite(v) else None)
                                       for a, v in zip(ALPHAS, macro[m], strict=True)},
                       "max_value": round(float(np.nanmax(macro[m])), 4),
                       "alpha_at_max": float(ALPHAS[int(np.nanargmax(macro[m]))]),
                       "positive_value_alpha_range": _pos_range(macro[m])}
                   for m in macro},
        "weather_increment_in_value": {
            "contrast": "m2 minus m2p: the same classifier with and without the "
                        "pre-gate weather channels",
            "difference": {f"{a:g}": round(float(x), 4)
                           for a, x in zip(ALPHAS,
                                           macro["m2"] - macro["m2p"], strict=True)}
            if ("m2" in macro and "m2p" in macro) else {},
            "ci95": diff_ci,
            "peak": None,   # filled below: where the weather channels add most
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    wd = out["weather_increment_in_value"]["difference"]
    if wd:
        best = max(wd, key=lambda k: wd[k])
        out["weather_increment_in_value"]["peak"] = {
            "alpha": float(best), "gain": wd[best],
            "ci95": out["weather_increment_in_value"]["ci95"].get(best),
            "reading": ("the cost-loss ratio at which pre-gate weather adds the "
                        "most decision value"),
        }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    for m in macro:
        e = out["models"][m]
        log.info("%-4s max V %.3f at alpha %.2f; V>0 over alpha in %s",
                 m, e["max_value"], e["alpha_at_max"],
                 e["positive_value_alpha_range"])
    if diff_ci:
        d = out["weather_increment_in_value"]["difference"]
        best = max(d, key=lambda k: d[k])
        log.info("weather adds most value at alpha=%s: %+0.3f %s",
                 best, d[best], diff_ci[best])
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
