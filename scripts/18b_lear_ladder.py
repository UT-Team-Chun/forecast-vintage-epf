r"""Phase 96: the ladder's headline contrast on LEAR instead of XGBoost.

WHY
---
A referee-simulation pass made baseline adequacy its central objection, and it is
correct on the facts: Section 4.4 names LEAR and a rolling ARIMA as scale anchors,
and Figure 3 says "XGBoost carries the ladder; LEAR and naive references" -- but
LEAR appears nowhere on the 732-fold panel. It exists only on a legacy Kyushu
n = 60 weekly-anchor pilot. So the paper's headline number rests on one learner at
one frozen configuration, and a reader cannot tell whether 0.253 JPY/kWh is a
property of the information set or of XGBoost.

This answers that question directly. It reuses phase85's panel, its design matrix,
its fold schedule and its cells VERBATIM by importing the module and rebinding a
single function -- the learner -- so that nothing but the estimator differs. LEAR
is the canonical EPF benchmark (LASSO-ARX with standardised features; Lago et al.),
and it is a strong one: in published benchmarks it beats deep networks on at least
one European market.

Two cells only, because the question is narrow: price-only, and price plus the
headline pre-gate weather tier. Their difference is the weather increment on a
different learner. If it is close to XGBoost's, the increment is a property of the
information set. If it is not, the paper has to say so.

The rebinding is asserted, not assumed: phase91 taught this project that a
module-global swap that silently fails turns a contrast into a copy and reports a
bug as a null.

CLI (from the repository root):
  uv run python scripts/18b_lear_ladder.py --step build
  uv run python scripts/18b_lear_ladder.py --step analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase96")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P85 = (Path(__file__).parent / "10_primary_ladder_cw.py")
CKPT = RUN / "phase96_lear_checkpoints"
OUT_JSON = RUN / "phase96_lear_ladder.json"

BASELINE = "price_only"
HEADLINE = "msm_d2_12z_cw"
# LassoCV over a fixed alpha path: the grid is declared here rather than left to
# sklearn's default so the fit is reproducible and the cost is bounded.
ALPHAS = np.logspace(-4, 1, 24)
CV_FOLDS = 5
SEED = 2026
# a slot-of-day needs at least this many past observations to get its own model
MIN_SLOT_ROWS = 60
BOOT_BLOCK, N_BOOT = 7, 10_000


def _load_phase85():
    spec = importlib.util.spec_from_file_location("p85", P85)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p85"] = mod
    spec.loader.exec_module(mod)
    return mod


def _lasso_arx(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    m = LassoCV(alphas=ALPHAS, cv=CV_FOLDS, max_iter=5000, random_state=SEED,
                n_jobs=1)
    m.fit(scaler.fit_transform(xtr), ytr)
    return m.predict(scaler.transform(xte)).astype(float)


def _lear_fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Canonical recalibrated LEAR: one LASSO-ARX per delivery half-hour.

    Estimating a SEPARATE model per slot-of-day is what makes this LEAR rather
    than a pooled LASSO, and it is how the benchmark is specified in the EPF
    literature -- the price process at 03:00 and at 13:00 are different processes,
    and a linear model cannot absorb that with shared coefficients the way a tree
    ensemble can with interactions. A pooled fit would understate the linear
    benchmark and make the tree ensemble look better than it is, which is the
    opposite of what a benchmark is for.

    Falls back to a pooled fit for any slot with too little history to estimate.
    """
    slot_tr = train["datetime_jst"].dt.hour() * 2 + (
        train["datetime_jst"].dt.minute() // 30)
    slot_te = test["datetime_jst"].dt.hour() * 2 + (
        test["datetime_jst"].dt.minute() // 30)
    str_np = slot_tr.to_numpy()
    ste_np = slot_te.to_numpy()
    xtr_all = train.select(cols).to_numpy().astype(float)
    ytr_all = train["price_yen_kwh"].to_numpy().astype(float)
    xte_all = test.select(cols).to_numpy().astype(float)
    out = np.empty(len(ste_np))
    pooled: np.ndarray | None = None
    for s in np.unique(ste_np):
        te_mask = ste_np == s
        tr_mask = str_np == s
        if tr_mask.sum() < MIN_SLOT_ROWS:
            if pooled is None:
                pooled = _lasso_arx(xtr_all, ytr_all, xte_all)
            out[te_mask] = pooled[te_mask]
            continue
        out[te_mask] = _lasso_arx(xtr_all[tr_mask], ytr_all[tr_mask],
                                  xte_all[te_mask])
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _block_boot_ci(d: np.ndarray) -> tuple[float, float]:
    """phase85's circular block bootstrap, same block length and seed."""
    n = len(d)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    means = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _hln_dm(d: np.ndarray, lag: int) -> dict:
    """Harvey-Leybourne-Newbold corrected Diebold-Mariano on the daily series."""
    n = len(d)
    dbar = float(d.mean())
    dc = d - dbar
    gamma0 = float((dc @ dc) / n)
    s = gamma0
    for k in range(1, lag + 1):
        g = float((dc[k:] @ dc[:-k]) / n)
        s += 2.0 * (1.0 - k / (lag + 1.0)) * g
    if s <= 0:
        return {"t": None, "p": None}
    t = dbar / math.sqrt(s / n)
    t *= math.sqrt((n + 1 - lag + lag * (1 - 1 / lag)) / n) if lag > 1 else 1.0
    p = math.erfc(abs(t) / math.sqrt(2))
    return {"t": round(t, 3), "p": round(p, 6)}


# The two cells this phase needs. Every other cell is still CONSTRUCTED, because
# phase85 derives the panel's row set from the full SP cell list and dropping cells
# would change the sample -- the one thing that must not differ. Their fits are
# skipped and written as NaN, which cannot be mistaken for a prediction.
_WANTED = (BASELINE, HEADLINE)
_wanted_colsets: set[tuple[str, ...]] = set()


def build(areas: list[str]) -> int:
    mod = _load_phase85()

    # ---- capture each area's cell -> column mapping, without altering it ----
    orig_design = mod._design

    def design_capturing(df):
        out, cells = orig_design(df)
        _wanted_colsets.clear()
        for name in _WANTED:
            if name not in cells:
                msg = f"phase85 has no cell named {name!r}"
                raise RuntimeError(msg)
            _wanted_colsets.add(tuple(cells[name]))
        if len(_wanted_colsets) != len(_WANTED):
            msg = ("two wanted cells share an identical column list, so they "
                   "cannot be told apart by their columns")
            raise RuntimeError(msg)
        return out, cells

    def fit_dispatch(train, test, cols):
        if tuple(cols) in _wanted_colsets:
            return _lear_fit(train, test, cols)
        return np.full(test.height, np.nan)

    original_fit, original_design = mod._fit, mod._design
    mod._fit, mod._design = fit_dispatch, design_capturing
    # PROVE both rebindings took. phase91 taught this project that a silent
    # module-global swap turns a contrast into a copy and reports a bug as a null.
    if mod._fit is original_fit or mod._design is original_design:
        msg = "a rebinding did not take effect"
        raise RuntimeError(msg)
    if mod._fit.__name__ != "fit_dispatch" or mod._design.__name__ != "design_capturing":
        msg = "a rebinding bound the wrong function"
        raise RuntimeError(msg)
    mod.CKPT = CKPT
    CKPT.mkdir(parents=True, exist_ok=True)
    for a in areas:
        log.info("=== %s (LEAR) ===", a)
        mod._run_area(a)
    return 0


def analyse() -> int:
    files = sorted(CKPT.glob("*.parquet"))
    if not files:
        log.error("no checkpoints in %s", CKPT)
        return 2
    frames = [pl.read_parquet(f) for f in files]
    d = pl.concat(frames, how="diagonal")
    need = [f"y_pred_{BASELINE}", f"y_pred_{HEADLINE}"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        log.error("checkpoints lack %s", missing)
        return 3
    per_day = (d.with_columns(
        ((pl.col(f"y_pred_{BASELINE}") - pl.col("y_actual")).abs()).alias("ae_base"),
        ((pl.col(f"y_pred_{HEADLINE}") - pl.col("y_actual")).abs()).alias("ae_head"))
        .group_by("area", "test_day")
        .agg(pl.col("ae_base").mean().alias("mae_base"),
             pl.col("ae_head").mean().alias("mae_head")))
    daily = (per_day.with_columns(
        (pl.col("mae_base") - pl.col("mae_head")).alias("imp"))
        .group_by("test_day").agg(pl.col("imp").mean(),
                                  pl.col("mae_base").mean().alias("base"),
                                  pl.col("mae_head").mean().alias("head"))
        .sort("test_day"))
    imp = np.asarray(daily["imp"], dtype=float)
    lo, hi = _block_boot_ci(imp)
    per_area = {}
    for a in sorted(d["area"].unique().to_list()):
        sub = (per_day.filter(pl.col("area") == a)
               .with_columns((pl.col("mae_base") - pl.col("mae_head")).alias("imp"))
               .sort("test_day"))
        v = np.asarray(sub["imp"], dtype=float)
        per_area[a] = {"improvement": round(float(v.mean()), 4),
                       "p_l3": _hln_dm(v, 3)["p"]}
    out = {
        "phase": "96",
        "title": "The headline weather increment on LEAR instead of XGBoost",
        "why": ("Section 4.4 names LEAR as a scale anchor and the ladder never ran "
                "it on the 732-fold panel, so a reader could not tell whether the "
                "headline increment is a property of the information set or of one "
                "learner"),
        "estimator": {"model": "LEAR (LassoCV over a fixed alpha path, standardised "
                               "features, one model per delivery half-hour)",
                      "alphas": [float(ALPHAS.min()), float(ALPHAS.max())],
                      "n_alphas": int(len(ALPHAS)), "cv_folds": CV_FOLDS,
                      "seed": SEED,
                      "note": "phase85's panel, design matrix, folds and cells "
                              "verbatim; only the learner differs"},
        "panel": {"n_test_days": int(daily.height),
                  "n_areas": int(d["area"].n_unique()),
                  "n_slots": int(d.height)},
        "levels_mae": {"price_only": round(float(daily["base"].mean()), 4),
                       "pre_gate": round(float(daily["head"].mean()), 4)},
        "weather_increment": {
            "improvement": round(float(imp.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "dm_l3": _hln_dm(imp, 3), "dm_l9": _hln_dm(imp, 9),
            "areas_improved": sum(1 for v in per_area.values()
                                  if v["improvement"] > 0),
            "n_areas": len(per_area),
        },
        "per_area": per_area,
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    w = out["weather_increment"]
    log.info("LEAR levels: price-only %.4f -> pre-gate %.4f",
             out["levels_mae"]["price_only"], out["levels_mae"]["pre_gate"])
    log.info("LEAR weather increment %+.4f [%.4f, %.4f], DM p(L=3)=%s, %d/%d areas",
             w["improvement"], w["ci95"][0], w["ci95"][1], w["dm_l3"]["p"],
             w["areas_improved"], w["n_areas"])
    log.info("wrote %s", OUT_JSON)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="")
    args = ap.parse_args()
    if args.step == "build":
        mod = _load_phase85()
        areas = ([a for a in args.areas.split(",") if a]
                 or list(mod.AREA_COORDS))
        return build(areas)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
