r"""Phase 81: direct probabilistic floor models vs the isotonic recalibration.

Precommit: docs/research/2026-08-11_phase81_direct_floor_models_precommit.md.
The paper's floor probabilities have so far been a monotone recalibration of the
point price forecast. This phase trains DIRECT models on the same walk-forward,
the same folds and the same information set (the CW-primary headline tier), and
reports every model at both thresholds -- the exact administrative floor
(0.01 JPY/kWh) and the near-floor band (0.5) -- under the frozen metric suite.

Models (fixed; no post-hoc additions):
  M0  isotonic-on-point-forecast (phase66's construction, from the phase85
      parquet's y_pred_msm_d2_12z_cw) -- the baseline.
  M1  expanding walk-forward logistic regression, direct P(floor).
  M2  XGBoost binary classifier (logloss; 300x4 trees, lr 0.05, seed 2026).
  M3  hurdle: in floor-probability space identical to M2 by construction
      (its price leg reuses the existing point forecast); reported as a note.
  M4  quantile-crossing: one multi-quantile XGBoost (pinball, tau in
      {0.02,0.05,0.1,0.2,0.35,0.5}), P(y <= thr) by monotone interpolation of
      the predicted quantile curve.
  M2p price-only-features control (same classifier, weather withheld):
      separates the weather contribution to floor skill.

Reporting rule (frozen): ALL models x ALL metrics in the table; the manuscript
headline is the best model under the primary metric (PR-AUC, macro average,
threshold 0.5, all-hours) with an explicit multiplicity note. If M0 wins, that
is the result ("the point forecast already carries the floor information").

CLI (from the repository root):
  uv run --extra research python scripts/31_direct_floor_models.py \
      --step build [--areas kyushu,...]
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import DEFAULT_WEATHER_COLS, attach_hourly_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase81")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
EXACTRUN_MP_DIR = RAW / "jma_gpv_exactrun_mp"
PHASE85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
CKPT = RUN / "phase81_checkpoints"
OUT_JSON = RUN / "phase81_direct_floor_models.json"
PRECOMMIT = "docs/research/2026-08-11_phase81_direct_floor_models_precommit.md"

INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 900
SEED = 2026
PANEL_START = date(2024, 3, 2)
ERA5_END = date(2026, 8, 5)
N_BOOT = 10_000
BOOT_BLOCK = 7
MIN_TRAIN_ROWS = 100
BURN_IN_DAYS = 30
THRESHOLDS = (0.01, 0.5)
PRIMARY_THR = 0.5
TAUS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
FPR_TARGETS = (0.005, 0.01, 0.05, 0.1)
WEIGHT_VINTAGE = "2024-09"
HEADLINE = "msm_d2_12z"   # CW source plan of the headline tier

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}


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


# -------------------------------------------------------------------- panel
def _load_cw_headline(area: str) -> pl.DataFrame:
    pts = area_points(WEIGHT_VINTAGE)[area]
    acc: pl.DataFrame | None = None
    for i, p in enumerate(pts):
        key = f"{area}_p{i:02d}"
        path = (EXACTRUN_MP_DIR / HEADLINE
                / f"jma_gpv_{HEADLINE}_{key}_{p.lat:.4f}_{p.lon:.4f}.parquet")
        df = (pl.read_parquet(path)
              .select("datetime_jst", *DEFAULT_WEATHER_COLS)
              .with_columns(*[(pl.col(c) * p.weight).alias(c)
                              for c in DEFAULT_WEATHER_COLS]))
        acc = df if acc is None else (
            acc.join(df, on="datetime_jst", how="inner", suffix="_r")
               .with_columns(*[(pl.col(c) + pl.col(f"{c}_r")).alias(c)
                               for c in DEFAULT_WEATHER_COLS])
               .select("datetime_jst", *DEFAULT_WEATHER_COLS))
    return acc.sort("datetime_jst").rename(
        {c: f"{c}_hw" for c in DEFAULT_WEATHER_COLS})


def _panel(area: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "price_yen_kwh"))
    df = attach_hourly_features(
        price.sort("datetime_jst"), _load_cw_headline(area),
        feature_cols=[f"{c}_hw" for c in DEFAULT_WEATHER_COLS])
    df = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        df = df.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    df = make_calendar_features(df, datetime_col="datetime_jst")
    base = [*(f"lag_{lag}" for lag in (48, 96, 336, 672)),
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
            "is_holiday"]
    full = [*base, *(f"{c}_hw" for c in DEFAULT_WEATHER_COLS)]
    df = (df.drop_nulls(subset=[*full, "price_yen_kwh"])
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, ERA5_END))
          .sort("datetime_jst"))
    return df, full, base


def _run_area(area: str) -> None:  # noqa: C901 - one fold loop, five models
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    df, full, base = _panel(area)
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
        xtr_f = train.select(full).to_numpy().astype(float)
        xte_f = test.select(full).to_numpy().astype(float)
        xtr_b = train.select(base).to_numpy().astype(float)
        xte_b = test.select(base).to_numpy().astype(float)
        ytr = train["price_yen_kwh"].to_numpy().astype(float)
        rec: dict[str, np.ndarray] = {}
        for thr in THRESHOLDS:
            lab = (ytr <= thr).astype(int)
            tag = f"{thr:g}"
            if lab.sum() >= 2 and lab.sum() < len(lab):
                m1 = LogisticRegression(max_iter=200, C=1.0)
                m1.fit(np.nan_to_num(xtr_f), lab)
                rec[f"m1_{tag}"] = m1.predict_proba(np.nan_to_num(xte_f))[:, 1]
                m2 = xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, random_state=SEED,
                                       n_jobs=4, tree_method="hist",
                                       eval_metric="logloss")
                m2.fit(xtr_f, lab)
                rec[f"m2_{tag}"] = m2.predict_proba(xte_f)[:, 1]
                m2p = xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, random_state=SEED,
                                        n_jobs=4, tree_method="hist",
                                        eval_metric="logloss")
                m2p.fit(xtr_b, lab)
                rec[f"m2p_{tag}"] = m2p.predict_proba(xte_b)[:, 1]
            else:
                for name in (f"m1_{tag}", f"m2_{tag}", f"m2p_{tag}"):
                    rec[name] = np.full(test.height, float(lab.mean()))
        m4 = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              random_state=SEED, n_jobs=4, tree_method="hist",
                              objective="reg:quantileerror",
                              quantile_alpha=np.array(TAUS))
        m4.fit(xtr_f, ytr)
        q = m4.predict(xte_f)          # (n, len(TAUS))
        q = np.sort(np.atleast_2d(q), axis=1)
        for thr in THRESHOLDS:
            # P(y <= thr): interpolate tau against the (sorted) quantile curve
            probs = np.empty(q.shape[0])
            for i in range(q.shape[0]):
                probs[i] = float(np.interp(thr, q[i], TAUS,
                                           left=0.0, right=max(TAUS)))
            rec[f"m4_{thr:g}"] = probs
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": float(y[s]),
                     **{name: float(v[s]) for name, v in rec.items()}}
                    for s in range(test.height))
        n_folds += 1
        if n_folds % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n_folds, test_day,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
    log.info("%s done: %d folds, %.0f s", area, n_folds, time.time() - t0)


# ------------------------------------------------------------------ metrics
def _pr_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(average_precision_score(y, s))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out: dict = {"n": len(y), "n_pos": int(y.sum()),
                 "pr_auc": _j(_pr_auc(y, p))}
    if y.sum() and y.sum() < len(y):
        out["roc_auc"] = _j(float(roc_auc_score(y, p)))
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out["brier"] = _j(float(brier_score_loss(y, pc)), 6)
        out["log_loss"] = _j(float(-np.mean(y * np.log(pc)
                                            + (1 - y) * np.log(1 - pc))), 6)
        # recall at fixed false-positive rates
        neg = np.sort(p[y == 0])[::-1]
        for f in FPR_TARGETS:
            k = max(int(np.floor(f * len(neg))) - 1, 0)
            thr_f = neg[k] if len(neg) else 1.0
            out[f"recall_at_fpr_{f:g}"] = _j(float((p[y == 1] > thr_f).mean()))
        # reliability: 10 equal-count bins
        order = np.argsort(p)
        bins = np.array_split(order, 10)
        mp = np.array([p[b].mean() for b in bins if len(b)])
        of = np.array([y[b].mean() for b in bins if len(b)])
        w = np.array([len(b) for b in bins if len(b)]) / len(y)
        out["ece"] = _j(float(np.sum(w * np.abs(mp - of))), 6)
        out["mce"] = _j(float(np.max(np.abs(mp - of))), 6)
        if np.std(mp) > 0:
            sl, ic = np.polyfit(mp, of, 1)
            out["calibration_slope"] = _j(float(sl))
            out["calibration_intercept"] = _j(float(ic))
    return out


def analyse() -> dict:
    files = sorted(CKPT.glob("*.parquet"))
    direct = pl.concat([pl.read_parquet(f) for f in files]).sort("area", "datetime_jst")
    burn = sorted(direct["test_day"].unique().to_list())[BURN_IN_DAYS]
    direct = direct.filter(pl.col("test_day") >= burn)

    # M0: isotonic on the phase85 CW headline point forecast, expanding
    p85 = (pl.read_parquet(PHASE85_PARQUET)
           .select("area", "datetime_jst", "test_day", "y_actual",
                   pl.col("y_pred_msm_d2_12z_cw").alias("yhat")))
    m0_frames = []
    for area in sorted(p85["area"].unique().to_list()):
        sub = p85.filter(pl.col("area") == area).sort("datetime_jst")
        days = sorted(sub["test_day"].unique().to_list())
        for thr in THRESHOLDS:
            probs = np.full(sub.height, np.nan)
            yhat = sub["yhat"].to_numpy()
            lab = (sub["y_actual"].to_numpy() <= thr).astype(int)
            day_arr = np.asarray(sub["test_day"].to_list())
            for d in days[BURN_IN_DAYS:]:
                tr = day_arr < d
                te = day_arr == d
                if lab[tr].sum() and lab[tr].sum() < tr.sum():
                    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0,
                                             out_of_bounds="clip")
                    iso.fit(yhat[tr], lab[tr])
                    probs[te] = iso.predict(yhat[te])
                else:
                    probs[te] = float(lab[tr].mean()) if tr.sum() else np.nan
            sub = sub.with_columns(pl.Series(f"m0_{thr:g}", probs))
        m0_frames.append(sub)
    m0 = pl.concat(m0_frames).drop("yhat")
    panel = direct.join(m0, on=["area", "datetime_jst"], how="inner",
                        suffix="_m0").sort("area", "datetime_jst")
    panel = panel.with_columns(pl.col("datetime_jst").dt.hour().alias("hour"))

    models = ["m0", "m1", "m2", "m2p", "m4"]
    out: dict = {"phase": 81, "title": "Direct probabilistic floor models",
                 "precommit_doc": PRECOMMIT,
                 "m3_note": "hurdle's floor-probability equals M2 by construction "
                            "(its price leg reuses the existing point forecast); "
                            "not a separate row",
                 "panel": {"n_slots": panel.height,
                           "n_test_days": panel["test_day"].n_unique(),
                           "burn_in_days": BURN_IN_DAYS},
                 "results": {}}
    areas = sorted(panel["area"].unique().to_list())
    for scope, hours in (("allhours", None), ("midday", (9, 15))):
        sub = panel if hours is None else panel.filter(
            pl.col("hour").is_between(*hours))
        for thr in THRESHOLDS:
            y_all = (sub["y_actual"].to_numpy() <= thr).astype(int)
            blk: dict = {}
            for m in models:
                col = f"m{m[1:]}_{thr:g}" if not m.startswith("m") else f"{m}_{thr:g}"
                p = sub[col].to_numpy().astype(float)
                ok = np.isfinite(p)
                rec = {"pooled": _metrics(y_all[ok], p[ok]), "per_area": {}}
                aucs = []
                for a in areas:
                    mask = (np.asarray(sub["area"].to_list()) == a) & ok
                    pa = _metrics(y_all[mask], p[mask])
                    rec["per_area"][a] = {"pr_auc": pa.get("pr_auc"),
                                          "brier": pa.get("brier")}
                    if pa.get("pr_auc") is not None:
                        aucs.append(pa["pr_auc"])
                rec["macro_pr_auc"] = _j(float(np.mean(aucs))) if aucs else None
                blk[m] = rec
            out["results"][f"{scope}_thr{thr:g}"] = blk
            summary = " ".join(f"{m}:{blk[m]['macro_pr_auc']}" for m in models)
            log.info("[%s thr=%g] %s", scope, thr, summary)

    prim = out["results"][f"allhours_thr{PRIMARY_THR:g}"]
    ranked = sorted((v["macro_pr_auc"], k) for k, v in prim.items()
                    if v["macro_pr_auc"] is not None)
    out["primary_metric"] = {
        "definition": "macro PR-AUC, threshold 0.5, all-hours",
        "ranking_best_last": ranked,
        "winner": ranked[-1][1] if ranked else None,
        "multiplicity_note": "five models compared; the winner is reported with "
                             "this note, per the precommit",
    }
    log.info("PRIMARY winner: %s (%s)", out["primary_metric"]["winner"], ranked)
    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "taus": list(TAUS),
                       "thresholds": list(THRESHOLDS)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="")
    args = ap.parse_args()
    if args.step == "build":
        wanted = ([a.strip() for a in args.areas.split(",") if a.strip()]
                  or list(AREA_COORDS))
        for a in wanted:
            _run_area(a)
        return 0
    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
