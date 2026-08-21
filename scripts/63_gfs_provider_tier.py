r"""Phase 86 (scoring): the GFS second-provider tiers on the phase85 folds.

Precommit: docs/research/2026-08-11_phase86_gfs_provider_precommit.md (Gate
G-GFS PASSED 2026-08-11: S3 LastModified is a contemporaneous dissemination
proxy, uploads ~3.6-3.9 h after initialisation).

Trains two single-point cells -- gfs_d2_12z (same init clock as the JMA
headline) and gfs_d2_18z (the freshest pre-gate GFS run) -- on exactly the
phase85 protocol (expanding folds, same constants, same learner), then joins
the phase85 parquet's price_only and SP JMA predictions for the pre-registered
questions: (i) does the GFS pre-gate tier beat price_only, (ii) which way is
the JMA-GFS difference, (iii) does the vintage ordering (12Z < 18Z) hold
within GFS. External-generality check; it does not replace the headline.

CLI (from the repository root):
  uv run --extra research python scripts/63_gfs_provider_tier.py \
      --step build [--areas ...]
  uv run --extra research python ... --step analyse
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import DEFAULT_WEATHER_COLS, attach_hourly_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase86")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
GFS_ROOT = RAW / "gfs_exactrun"
PHASE85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
CKPT = RUN / "phase86_checkpoints"
OUT_JSON = RUN / "phase86_gfs_tier.json"
PRECOMMIT = "docs/research/2026-08-11_phase86_gfs_provider_precommit.md"

INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 900
SEED = 2026
PANEL_START = date(2024, 3, 2)
ERA5_END = date(2026, 8, 5)
N_BOOT, BOOT_BLOCK = 10_000, 7
MIN_TRAIN_ROWS = 100
GFS_TIERS = ("gfs_d2_12z", "gfs_d2_18z")

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}


def _j(x: float | None, nd: int = 4) -> float | None:
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


def _run_area(area: str) -> None:
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    lat, lon = AREA_COORDS[area]
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "price_yen_kwh"))
    df = price.sort("datetime_jst")
    for tier in GFS_TIERS:
        p = GFS_ROOT / tier / f"gfs_{tier}_{area}_{lat:.4f}_{lon:.4f}.parquet"
        w = (pl.read_parquet(p).select("datetime_jst", *DEFAULT_WEATHER_COLS)
             .rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS}))
        df = attach_hourly_features(
            df, w, feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])
    df = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        df = df.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    df = make_calendar_features(df, datetime_col="datetime_jst")
    base = [*(f"lag_{lag}" for lag in (48, 96, 336, 672)),
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
            "is_holiday"]
    cells = {t: [*base, *(f"{c}_{t}" for c in DEFAULT_WEATHER_COLS)]
             for t in GFS_TIERS}
    need = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, ERA5_END))
          .sort("datetime_jst"))
    log.info("%s panel: %d slots", area, df.height)
    panel_last = df["datetime_jst"].max().date()
    rows: list[dict] = []
    t0, n_folds = time.time(), 0
    for k in range(MAX_FOLDS):
        test_day = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        if test_day > panel_last:
            break
        train = df.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = (df.filter(pl.col("datetime_jst").dt.date() == test_day)
                  .sort("datetime_jst"))
        if train.height < MIN_TRAIN_ROWS or test.height == 0:
            continue
        preds = {}
        for name, cols in cells.items():
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=SEED, n_jobs=4, tree_method="hist")
            m.fit(train.select(cols).to_numpy().astype(float),
                  train["price_yen_kwh"].to_numpy().astype(float))
            preds[name] = m.predict(test.select(cols).to_numpy().astype(float))
        dts = test["datetime_jst"].to_list()
        y = test["price_yen_kwh"].to_numpy().astype(float)
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": float(y[s]),
                     **{f"y_pred_{n}": float(v[s]) for n, v in preds.items()}}
                    for s in range(test.height))
        n_folds += 1
        if n_folds % 100 == 0:
            log.info("%s fold %d | %.2f s/fold", area, n_folds,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
    log.info("%s done: %d folds, %.0f s", area, n_folds, time.time() - t0)


def _boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    means = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _hln_p(d: np.ndarray, lag: int = 3) -> float:
    n = len(d)
    d_bar = float(d.mean())
    var_hac = float(np.var(d, ddof=0))
    for ell in range(1, lag + 1):
        gamma = float(np.mean((d[:n - ell] - d_bar) * (d[ell:] - d_bar)))
        var_hac += 2.0 * (1.0 - ell / (lag + 1)) * gamma
    se = float((max(var_hac, 1e-12) * (n - 1) / n / n) ** 0.5)
    t = d_bar / se if se > 0 else float("nan")
    return 2.0 * (1.0 - float(stats.t.cdf(abs(t), df=n - 1)))


def analyse() -> dict:
    gfs = pl.concat([pl.read_parquet(f) for f in sorted(CKPT.glob("*.parquet"))])
    p85 = pl.read_parquet(PHASE85_PARQUET).select(
        "area", "datetime_jst", "test_day",
        "y_pred_price_only", "y_pred_msm_d2_12z", "y_pred_msm_gate_composite")
    panel = gfs.join(p85, on=["area", "datetime_jst"], how="inner")
    cells = [*GFS_TIERS, "price_only", "msm_d2_12z", "msm_gate_composite"]
    per_day = (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("area", pl.col("test_day"))
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .group_by("test_day")
        .agg(*[pl.col(f"mae_{c}").mean() for c in cells]).sort("test_day"))

    def imp(cell: str) -> np.ndarray:
        return np.asarray(per_day["mae_price_only"] - per_day[f"mae_{cell}"],
                          dtype=float)

    out: dict = {"phase": 86, "title": "GFS second-provider tiers",
                 "precommit_doc": PRECOMMIT,
                 "note": "single-point cells, joined to the phase85 folds; "
                         "external-generality check, not a headline replacement",
                 "panel": {"n_slots": panel.height,
                           "n_test_days": per_day.height}, "tiers": {}}
    for t in (*GFS_TIERS, "msm_d2_12z", "msm_gate_composite"):
        d = imp(t)
        lo, hi = _boot_ci(d)
        out["tiers"][t] = {"improvement": _j(float(d.mean())),
                           "ci95": [_j(lo), _j(hi)], "p_l3": _j(_hln_p(d), 8)}
        log.info("%-20s imp=%+.4f [%+.4f, %+.4f]", t, d.mean(), lo, hi)
    q = {}
    d = imp("gfs_d2_12z")
    q["q1_gfs_beats_price_only"] = bool(_boot_ci(d)[0] > 0 and _hln_p(d) < 0.05)
    dd = imp("msm_d2_12z") - imp("gfs_d2_12z")
    lo, hi = _boot_ci(dd)
    q["q2_jma_minus_gfs_same_init"] = {"delta": _j(float(dd.mean())),
                                       "ci95": [_j(lo), _j(hi)],
                                       "p_l3": _j(_hln_p(dd), 8)}
    dv = imp("gfs_d2_18z") - imp("gfs_d2_12z")
    lo, hi = _boot_ci(dv)
    q["q3_within_gfs_vintage_step"] = {"delta": _j(float(dv.mean())),
                                       "ci95": [_j(lo), _j(hi)],
                                       "p_l3": _j(_hln_p(dv), 8)}
    out["questions"] = q
    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "n_boot": N_BOOT}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="")
    args = ap.parse_args()
    if args.step == "build":
        for a in ([x.strip() for x in args.areas.split(",") if x.strip()]
                  or list(AREA_COORDS)):
            _run_area(a)
        return 0
    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
