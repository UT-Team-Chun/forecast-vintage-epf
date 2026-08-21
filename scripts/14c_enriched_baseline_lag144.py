r"""Phase 54 variant v2_lag144: enriched-baseline lineage branch (phase78 precommit).

MECHANICAL FORK of s_20260726_phase54_nwp_proxy_baseline.py (2026-08-12).
V-lag144: SUPPLY_LAG raised to 144 slots (3 days) as the conservative timing margin.
Only the constants/blocks below differ; frozen phase54 artefacts stay
untouched per the freeze policy.

ORIGINAL DOCSTRING:
Phase 54: NWP-proxy-enriched baseline (reviewer major-issue 7).

The manuscript claims the pre-gate weather tier is worth +0.216 yen/kWh over an
"enriched baseline of all other admissible inputs" (phase51 headline over
price_only; phase32 established the enriched-baseline protocol). A practitioner
objection remains: a real forecaster's PV/WIND/LOAD forecasters would be driven
by the SAME pre-gate NWP vintage, so the raw-weather value could really be "the
value of a baseline lacking renewable/load forecasts". This phase answers it by
adding NWP-DERIVED PROXIES (what those downstream forecasters would produce from
the identical prev2 vintage) to the strictly admissible enriched baseline, then
asking what raw weather still adds.

Four same-sample cells, identical folds, 8 OCCTO areas (Chubu purged):
  1. price_only            price lags 48/96/336/672 + calendar (paper baseline)
  2. enriched_adm          + system/spread/neighbour prices lag-48
                           + OCCTO supply-mix lag-96 (phase32 --supply-lag 96,
                           the GENUINELY bid-close-admissible enrichment)
  3. enriched_proxies      + NWP proxies derived ONLY from prev2 channels:
                             pv_proxy   = sw_prev2 * pv_gw * (1 - 0.004*(T_prev2 - 25))
                             wind_proxy = cubic ramp(v_prev2; cut-in 3, rated 12,
                                          cut-out 25 m/s) * wind_gw
                             hdd = max(18 - T_prev2, 0), cdd = max(T_prev2 - 24, 0)
  4. enriched_proxies_raw  + the 4 raw prev2 channels themselves

Report per area + pooled: MAE per cell; d32 = value of NWP-derived proxies;
d43 = residual raw-weather value on top of engineered proxies; d42 = total
prev2 value over the admissible enriched baseline. Inference: paired HLN-NW DM
(L=3 and L=9) + per-day paired circular block bootstrap (block 7 d, 10,000
draws, seed 2026) 95% CIs -- the phase51 inference stack.

Fold protocol = phase51 densified daily protocol (expanding window, initial
train end 2024-07-18, val-offset 14 d, test 1 d, STEP 1 d), restricted to the
window where the OCCTO supply panel exists (2024-04-01..2026-03-31; lag-96
features admit test days through 2026-04-02).

Honest-framing rule (pre-committed, recorded in the JSON before any sweep
result): if d43 ~ 0 the claim narrows to "the pre-gate NWP vintage carries the
value, whether fed raw or as derived proxies" -- that outcome still supports
the timing thesis; record whichever way it falls.

CLI:
  cd <repository root>
  uv run python s_20260726_phase54_nwp_proxy_baseline.py \
      --step all [--areas kyushu,tokyo] [--fresh] [--workers 4] [--limit-folds N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import (
    peak_gw_by_area,
    pv_gw_by_area,
    reference_sha256,
    wind_gw_by_area,
)
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.data.registry import REGISTRY_FILENAME, DatasetRegistry
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import DEFAULT_WEATHER_COLS, attach_hourly_features
from alg.models.epf.area import AreaCode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase54")
JST = ZoneInfo("Asia/Tokyo")

# 8 OCCTO-covered areas (Chubu purged from the supply-mix panel) -- phase32 set.
AREAS = ("hokkaido", "tohoku", "tokyo", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu")

# phase51 densified daily protocol
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, TEST_WINDOW, STEP = timedelta(days=14), timedelta(days=1), timedelta(days=1)
MAX_FOLDS = 800
SEED = 2026
N_BOOT, BLOCK_LEN = 10_000, 7

# phase32 enriched-baseline protocol constants (supply at lag-96 = admissible)
SUPPLY_LAG = 144
PRODUCTION_LAG_COLS = ("demand_mw", "thermal_lng_mw", "thermal_coal_mw",
                       "interconnector_mw", "pv_curtailment_mw")

# proxy physics constants (pre-specified in the phase54 design)
PV_TEMP_DERATE_PER_C = 0.004       # relative derating per degC above 25 C
PV_TEMP_REF_C = 25.0
WIND_CUT_IN_MS, WIND_RATED_MS, WIND_CUT_OUT_MS = 3.0, 12.0, 25.0
HDD_BASE_C, CDD_BASE_C = 18.0, 24.0

CELLS_ORDER = ("price_only", "enriched_adm", "enriched_proxies", "enriched_proxies_raw")
PROXY_COLS = ("pv_proxy", "wind_proxy", "hdd", "cdd")
PREV2_RAW_COLS = tuple(f"{c}_pd2" for c in DEFAULT_WEATHER_COLS)

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase54v2_lag144_checkpoints"
OUT_JSON = RUN / "phase54v2_lag144.json"
PER_SLOT_PARQUET = RUN / "phase54v2_lag144_per_slot.parquet"
OCCTO_PARQUET = RUN / "phase25.3_occto_generation.parquet"
PREV2_DIR = RAW / "open_meteo_historical_forecast"

HONEST_FRAMING_RULE_VERBATIM = (
    "Pre-committed before any phase54 sweep result exists: if the residual "
    "raw-weather value d43 = MAE(enriched_proxies) - MAE(enriched_proxies_raw) "
    "is ~ 0 (operationalised on the pooled 8-area panel as: the per-day paired "
    "circular block bootstrap (block 7 days, 10,000 draws, seed 2026) 95% CI of "
    "d43 contains 0, OR the pooled HLN-NW DM p >= 0.05 at either L=3 or L=9), "
    "the paper's claim NARROWS to 'the pre-gate NWP vintage carries the value, "
    "whether fed raw or as engineered PV/wind/load proxies' -- that outcome "
    "still supports the timing thesis. Record whichever way it falls. The total "
    "prev2 value over the admissible enriched baseline is d42, and the "
    "proxy-carried part is d32; d42 replaces the manuscript's "
    "over-enriched-baseline reading of the +0.216 headline."
)

AREA_PV_GW = pv_gw_by_area()
AREA_WIND_GW = wind_gw_by_area()
AREA_PEAK_GW = peak_gw_by_area()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_summary() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"phase": "54",
            "title": "NWP-proxy-enriched baseline (reviewer major-issue 7)"}


def _save_summary(summary: dict) -> None:
    summary["metadata"] = summary.get("metadata", {})
    summary["metadata"].update({
        "script": Path(__file__).name,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=False).stdout.strip(),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote %s", OUT_JSON)


def _record_precommit(summary: dict) -> None:
    """Write the honest-framing rule + design BEFORE any sweep result exists."""
    if "precommit" in summary:
        return
    summary["precommit"] = {
        "honest_framing_rule_verbatim": HONEST_FRAMING_RULE_VERBATIM,
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorded_before_any_sweep_result": not any(CKPT.glob("*.parquet")),
    }
    summary["design"] = {
        "cells": {
            "price_only": "price lags 48/96/336/672 + calendar (paper baseline)",
            "enriched_adm": ("+ system lag-48 + (system-own) spread lag-48 + "
                             "neighbour prices lag-48 + OCCTO supply-mix lag-96 "
                             "(phase32 strictly bid-close-admissible enrichment)"),
            "enriched_proxies": ("+ NWP proxies derived ONLY from prev2 channels: "
                                 "pv_proxy = sw_pd2*pv_gw*(1-0.004*(T_pd2-25)); "
                                 "wind_proxy = cubic ramp(v_pd2; 3/12/25 m/s)*wind_gw; "
                                 "hdd = max(18-T_pd2,0); cdd = max(T_pd2-24,0)"),
            "enriched_proxies_raw": "+ the 4 raw prev2 channels themselves",
        },
        "fold_protocol": ("phase51 densified daily protocol: expanding window, "
                          "initial train end 2024-07-18, val-offset 14 d, test 1 d, "
                          "STEP 1 d; XGBRegressor(300, depth 4, lr 0.05, seed 2026, "
                          "hist); common same-sample rows across all 4 cells"),
        "window_constraint": ("OCCTO supply panel 2024-04-01..2026-03-31 (8 areas); "
                              "lag-96 supply features admit test days through "
                              "2026-04-02; prev2 caches cover the full window"),
        "quantities": {
            "d32": "MAE(enriched_adm) - MAE(enriched_proxies) = value of NWP-derived proxies",
            "d43": ("MAE(enriched_proxies) - MAE(enriched_proxies_raw) = residual "
                    "raw-weather value on top of engineered proxies"),
            "d42": ("MAE(enriched_adm) - MAE(enriched_proxies_raw) = total prev2 "
                    "value over the admissible enriched baseline"),
            "enrichment_gain": "MAE(price_only) - MAE(enriched_adm) (baseline strength)",
        },
        "deviation_from_phase32": ("supply forward-fill limited to 1 slot (the "
                                   "interval-END alignment boundary) instead of "
                                   "unlimited, so genuine OCCTO gaps (kyushu ~15 d) "
                                   "drop from the common sample rather than being "
                                   "filled with stale values"),
    }
    _save_summary(summary)


# ------------------------------------------------------------------ features
def _load_prev2(area: str) -> pl.DataFrame:
    files = sorted(PREV2_DIR.glob(f"open_meteo_prev2_jma_msm_{area}_*.parquet"))
    if not files:
        msg = f"no prev2 caches for {area} under {PREV2_DIR}"
        raise FileNotFoundError(msg)
    return (pl.concat([pl.read_parquet(f) for f in files])
            .unique("datetime_jst").sort("datetime_jst"))


def _proxy_exprs(area: str) -> list[pl.Expr]:
    """NWP-derived proxies from prev2 channels only (design-fixed physics)."""
    t = pl.col("temperature_c_pd2")
    sw = pl.col("shortwave_radiation_wm2_pd2")
    v = pl.col("wind_speed_kmh_pd2") / 3.6  # km/h -> m/s
    ramp = (pl.when(v < WIND_CUT_IN_MS).then(0.0)
            .when(v < WIND_RATED_MS)
            .then(((v - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS)) ** 3)
            .when(v < WIND_CUT_OUT_MS).then(1.0)
            .otherwise(0.0))
    return [
        (sw * AREA_PV_GW[area]
         * (1.0 - PV_TEMP_DERATE_PER_C * (t - PV_TEMP_REF_C))).alias("pv_proxy"),
        (ramp * AREA_WIND_GW[area]).alias("wind_proxy"),
        (HDD_BASE_C - t).clip(lower_bound=0.0).alias("hdd"),
        (t - CDD_BASE_C).clip(lower_bound=0.0).alias("cdd"),
    ]


def _build_design(df: pl.DataFrame, neighbour_cols: list[str],
                  ) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase32 cell structure with supply at lag-96 + the two prev2-derived tiers."""
    out = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        out = out.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    base = [f"lag_{lag}" for lag in (48, 96, 336, 672)]
    cal = ["sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
           "is_holiday"]
    out = out.with_columns(
        pl.col("system_price_yen_kwh").shift(48).alias("system_lag_48"),
        (pl.col("system_price_yen_kwh") - pl.col("price_yen_kwh")).shift(48)
        .alias("sys_spread_lag_48"),
    )
    enrich = ["system_lag_48", "sys_spread_lag_48"]
    for nc in neighbour_cols:
        out = out.with_columns(pl.col(nc).shift(48).alias(f"{nc}_lag_48"))
        enrich.append(f"{nc}_lag_48")
    for col in PRODUCTION_LAG_COLS:
        out = out.with_columns(pl.col(col).shift(SUPPLY_LAG).alias(f"{col}_lag_{SUPPLY_LAG}"))
        enrich.append(f"{col}_lag_{SUPPLY_LAG}")
    out = make_calendar_features(out, datetime_col="datetime_jst")
    cells = {
        "price_only": [*base, *cal],
        "enriched_adm": [*base, *cal, *enrich],
        "enriched_proxies": [*base, *cal, *enrich, *PROXY_COLS],
        "enriched_proxies_raw": [*base, *cal, *enrich, *PROXY_COLS, *PREV2_RAW_COLS],
    }
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


def _run_area(area: str, limit_folds: int = 0) -> None:
    ckpt = CKPT / f"{area}.parquet"
    meta_f = CKPT / f"{area}_meta.json"
    if ckpt.exists() and meta_f.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    log.info("=== AREA %s ===", area)
    t0 = time.time()
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW,
                            areas=[AreaCode(area)], registry=reg)
             .select("datetime_jst", "area", "price_yen_kwh", "system_price_yen_kwh")
             .with_columns(pl.col("area").cast(pl.String)))
    # neighbouring-area prices (all other OCCTO areas), wide
    long_all = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW,
                               areas=[AreaCode(a) for a in AREAS], registry=reg)
                .select("datetime_jst", "area", "price_yen_kwh")
                .with_columns(pl.col("area").cast(pl.String)))
    wide_prices = long_all.pivot(values="price_yen_kwh", index="datetime_jst",
                                 on="area").rename(
        {a: f"price_{a}" for a in long_all["area"].unique().to_list()})
    neighbour_cols = [c for c in wide_prices.columns
                      if c not in ("datetime_jst", f"price_{area}")]
    df = price.join(wide_prices, on="datetime_jst", how="left")
    # prev2 channels (hourly caches -> half-hourly grid)
    prev2 = _load_prev2(area)
    df = attach_hourly_features(df, prev2, feature_cols=list(PREV2_RAW_COLS))
    df = df.with_columns(_proxy_exprs(area))
    # OCCTO supply-mix (interval-END -> interval-START align by -30min; forward
    # fill LIMIT 1 fixes only the final boundary slot -- genuine gaps stay null)
    gen_area = (pl.read_parquet(OCCTO_PARQUET)
                .filter(pl.col("area") == area)
                .with_columns(
                    (pl.col("pv_curtailment_mw") > 0).cast(pl.Int8)
                    .alias("pv_curtailment_mw"),
                    (pl.col("datetime_jst") - pl.duration(minutes=30))
                    .alias("datetime_jst")))
    supply_last_day = gen_area["datetime_jst"].max().date()
    df = (df.join(gen_area.select(["datetime_jst", *PRODUCTION_LAG_COLS]),
                  on="datetime_jst", how="left")
          .sort("datetime_jst")
          .with_columns([pl.col(c).forward_fill(limit=1).alias(c)
                         for c in PRODUCTION_LAG_COLS]))

    design, cells = _build_design(df, neighbour_cols)
    union = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    panel_end = supply_last_day + timedelta(days=SUPPLY_LAG // 48)
    design = (design.drop_nulls(subset=union)
              .filter(pl.col("datetime_jst").dt.date() <= panel_end)
              .sort("datetime_jst"))
    log.info("%s common-sample panel: %d slots %s..%s (%d features in cell 4)",
             area, design.height, design["datetime_jst"].min(),
             design["datetime_jst"].max(), len(cells["enriched_proxies_raw"]))

    pvpeak = AREA_PV_GW[area] / AREA_PEAK_GW[area]
    panel_last = design["datetime_jst"].max().date()
    rows: list[dict] = []
    n_folds_run = 0
    for k in range(MAX_FOLDS):
        test_day = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        if test_day > panel_last:
            break
        train = design.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = design.filter(pl.col("datetime_jst").dt.date() == test_day).sort("datetime_jst")
        if train.height < 100 or test.height == 0:
            continue
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        pvp = test["pv_proxy"].to_numpy().astype(float)
        wdp = test["wind_proxy"].to_numpy().astype(float)
        swp = test["shortwave_radiation_wm2_pd2"].to_numpy().astype(float)
        for s in range(test.height):
            rows.append({"area": area, "datetime_jst": dts[s],
                         "test_day": test_day.isoformat(), "y_actual": y[s],
                         **{f"y_pred_{name}": float(preds[name][s]) for name in cells},
                         "pv_proxy": pvp[s], "wind_proxy": wdp[s],
                         "shortwave_radiation_wm2_pd2": swp[s],
                         "pv_peak_ratio": pvpeak})
        n_folds_run += 1
        if n_folds_run % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n_folds_run,
                     test_day, (time.time() - t0) / n_folds_run)
        if limit_folds and n_folds_run >= limit_folds:
            break
    out = pl.DataFrame(rows)
    CKPT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(ckpt)
    meta_f.write_text(json.dumps({
        "area": area, "n_folds": n_folds_run, "n_slots": int(out.height),
        "panel_first_slot": design["datetime_jst"].min().isoformat(),
        "panel_last_slot": design["datetime_jst"].max().isoformat(),
        "panel_slots": int(design.height),
        "first_test_day": out["test_day"].min(), "last_test_day": out["test_day"].max(),
        "n_features_cell4": len(cells["enriched_proxies_raw"]),
        "pv_gw": AREA_PV_GW[area], "wind_gw": AREA_WIND_GW[area],
        "pv_peak_ratio": round(pvpeak, 4),
        "prev2_cache_sha256": {f.name: _sha256(f) for f in sorted(
            PREV2_DIR.glob(f"open_meteo_prev2_jma_msm_{area}_*.parquet"))},
        "runtime_s": round(time.time() - t0, 1)}, indent=2))
    log.info("%s done: %d folds, %d slots, %.0f s", area, n_folds_run,
             out.height, time.time() - t0)


def step_headtohead(areas: list[str] | None, fresh: bool, workers: int,
                    limit_folds: int) -> None:
    if fresh and CKPT.exists():
        for f in CKPT.glob("*"):
            f.unlink()
    wanted = [a for a in (areas or AREAS) if a in AREAS]
    pending = [a for a in wanted
               if not ((CKPT / f"{a}.parquet").exists()
                       and (CKPT / f"{a}_meta.json").exists())]
    log.info("areas pending: %s (workers=%d)", pending, workers)
    if not pending:
        return
    if workers <= 1 or len(pending) == 1:
        for a in pending:
            _run_area(a, limit_folds)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_area, a, limit_folds): a for a in pending}
        for fut, a in futs.items():
            fut.result()  # re-raise worker failures with the area visible
            log.info("worker finished: %s", a)


# ==================================================================== statistics
def _hln_newey_west_dm(d: np.ndarray, *, h: int = 1, lag: int = 3) -> dict:
    """DM on per-day loss diffs, Newey-West (Bartlett) HAC + HLN correction."""
    n = len(d)
    d_bar = float(d.mean())
    gamma_0 = float(np.var(d, ddof=0))
    var_hac = gamma_0
    for ell in range(1, lag + 1):
        gamma = float(np.mean((d[:n - ell] - d_bar) * (d[ell:] - d_bar)))
        var_hac += 2.0 * (1.0 - ell / (lag + 1)) * gamma
    k = (n + 1 - 2 * h + h * (h - 1) / n) / n
    se = float((max(var_hac, 1e-12) * k / n) ** 0.5)
    t_stat = d_bar / se if se > 0 else float("nan")
    p = 2.0 * (1.0 - float(stats.t.cdf(abs(t_stat), df=n - 1)))
    return {"mean_diff": round(d_bar, 5), "t_hln": round(float(t_stat), 3),
            "p_hln": round(float(p), 6), "n": n, "nw_lag": lag}


def _per_day_mae(df: pl.DataFrame) -> pl.DataFrame:
    return (df.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in CELLS_ORDER])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in CELLS_ORDER])
        .sort("area", "test_day"))


def _pooled_daily(per_day: pl.DataFrame) -> pl.DataFrame:
    return (per_day.group_by("test_day")
            .agg(*[pl.col(f"mae_{c}").mean() for c in CELLS_ORDER])
            .sort("test_day"))


def _dm_pair(per_day: pl.DataFrame, a: str, b: str, lag: int) -> dict:
    """DM of cell a vs b (negative mean_diff = a better), per area + pooled."""
    out: dict = {"per_area": {}}
    for area in sorted(per_day["area"].unique().to_list()):
        sub = per_day.filter(pl.col("area") == area).sort("test_day")
        d = (sub[f"mae_{a}"] - sub[f"mae_{b}"]).to_numpy()
        out["per_area"][area] = _hln_newey_west_dm(d, lag=lag)
    pooled = _pooled_daily(per_day)
    d = (pooled[f"mae_{a}"] - pooled[f"mae_{b}"]).to_numpy()
    out["pooled_across_areas"] = _hln_newey_west_dm(d, lag=lag)
    out["n_areas_p_lt_05_a_better"] = sum(
        1 for v in out["per_area"].values() if v["p_hln"] < 0.05 and v["mean_diff"] < 0)
    return out


def _bootstrap_indices(n_days: int) -> np.ndarray:
    """Circular block bootstrap day indices, shared across all series (paired)."""
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n_days / BLOCK_LEN))
    starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]) % n_days
    return idx.reshape(N_BOOT, -1)[:, :n_days]


def _ci(draws: np.ndarray) -> dict:
    return {"lo95": round(float(np.percentile(draws, 2.5)), 4),
            "hi95": round(float(np.percentile(draws, 97.5)), 4)}


def step_aggregate(summary: dict) -> None:
    frames = [pl.read_parquet(f) for f in sorted(CKPT.glob("*.parquet"))]
    if not frames:
        log.error("no head-to-head checkpoints found in %s", CKPT)
        return
    if "precommit" not in summary:
        msg = "precommit block missing from JSON -- must be recorded before results"
        raise RuntimeError(msg)
    panel = pl.concat(frames)
    panel.write_parquet(PER_SLOT_PARQUET)
    metas = {json.loads(f.read_text())["area"]: json.loads(f.read_text())
             for f in sorted(CKPT.glob("*_meta.json"))}
    n_areas = panel["area"].n_unique()
    partial = n_areas < len(AREAS)

    per_day = _per_day_mae(panel)
    pooled_daily = _pooled_daily(per_day)
    test_days = pooled_daily["test_day"].to_list()
    n_days = len(test_days)

    # daily delta series (positive = the richer cell is better)
    deltas = {
        "d21_enrichment_gain": ("price_only", "enriched_adm"),
        "d32_proxies_over_enriched": ("enriched_adm", "enriched_proxies"),
        "d43_raw_over_proxies": ("enriched_proxies", "enriched_proxies_raw"),
        "d42_total_prev2_over_enriched": ("enriched_adm", "enriched_proxies_raw"),
    }
    delta_series = {k: (pooled_daily[f"mae_{a}"] - pooled_daily[f"mae_{b}"]).to_numpy()
                    for k, (a, b) in deltas.items()}
    idx = _bootstrap_indices(n_days)
    boot = {k: {"point_daily_mean": round(float(s.mean()), 4),
                **_ci(s[idx].mean(axis=1))}
            for k, s in delta_series.items()}

    # DM tests (a = richer cell listed first so mean_diff < 0 = richer better)
    dm_pairs = [("enriched_adm", "price_only"),
                ("enriched_proxies", "enriched_adm"),
                ("enriched_proxies_raw", "enriched_proxies"),
                ("enriched_proxies_raw", "enriched_adm")]
    dm = {f"{a}_vs_{b}": {f"nw_l{lg}": _dm_pair(per_day, a, b, lg) for lg in (3, 9)}
          for a, b in dm_pairs}

    # per-area summary
    per_area = {}
    for area in sorted(panel["area"].unique().to_list()):
        sub = per_day.filter(pl.col("area") == area)
        m = {c: float(sub[f"mae_{c}"].mean()) for c in CELLS_ORDER}
        per_area[area] = {
            **{f"mae_{c}": round(m[c], 4) for c in CELLS_ORDER},
            "d21": round(m["price_only"] - m["enriched_adm"], 4),
            "d32": round(m["enriched_adm"] - m["enriched_proxies"], 4),
            "d43": round(m["enriched_proxies"] - m["enriched_proxies_raw"], 4),
            "d42": round(m["enriched_adm"] - m["enriched_proxies_raw"], 4),
            "dm_p_l3_d32": dm["enriched_proxies_vs_enriched_adm"]["nw_l3"]["per_area"][area]["p_hln"],
            "dm_p_l3_d43": dm["enriched_proxies_raw_vs_enriched_proxies"]["nw_l3"]["per_area"][area]["p_hln"],
            "dm_p_l3_d42": dm["enriched_proxies_raw_vs_enriched_adm"]["nw_l3"]["per_area"][area]["p_hln"],
            "pv_peak": round(AREA_PV_GW[area] / AREA_PEAK_GW[area], 4),
            "n_test_days": int(sub.height),
            "n_obs": int(panel.filter(pl.col("area") == area).height),
        }

    # pooled MAE per cell (slot mean and daily mean)
    pooled_mae_slot = {c: round(float(
        (panel[f"y_pred_{c}"] - panel["y_actual"]).abs().mean()), 4)
        for c in CELLS_ORDER}
    pooled_mae_daily = {c: round(float(pooled_daily[f"mae_{c}"].mean()), 4)
                        for c in CELLS_ORDER}

    # honest-framing verdict per the pre-committed operationalisation
    d43_ci = boot["d43_raw_over_proxies"]
    p43_l3 = dm["enriched_proxies_raw_vs_enriched_proxies"]["nw_l3"]["pooled_across_areas"]["p_hln"]
    p43_l9 = dm["enriched_proxies_raw_vs_enriched_proxies"]["nw_l9"]["pooled_across_areas"]["p_hln"]
    d43_dir = dm["enriched_proxies_raw_vs_enriched_proxies"]["nw_l3"]["pooled_across_areas"]["mean_diff"] < 0
    d43_significant = bool(d43_ci["lo95"] > 0 and p43_l3 < 0.05 and p43_l9 < 0.05
                           and d43_dir)
    d42_ci = boot["d42_total_prev2_over_enriched"]
    p42_l3 = dm["enriched_proxies_raw_vs_enriched_adm"]["nw_l3"]["pooled_across_areas"]["p_hln"]
    p42_l9 = dm["enriched_proxies_raw_vs_enriched_adm"]["nw_l9"]["pooled_across_areas"]["p_hln"]
    d32_ci = boot["d32_proxies_over_enriched"]
    p32_l3 = dm["enriched_proxies_vs_enriched_adm"]["nw_l3"]["pooled_across_areas"]["p_hln"]
    p32_l9 = dm["enriched_proxies_vs_enriched_adm"]["nw_l9"]["pooled_across_areas"]["p_hln"]
    verdict = {
        "d43_raw_weather_residual_significant": d43_significant,
        "framing": ("raw weather retains residual value ON TOP of engineered "
                    "NWP proxies -- the over-enriched-baseline claim survives "
                    "with raw channels specifically"
                    if d43_significant else
                    "claim NARROWS per the pre-committed rule: the pre-gate NWP "
                    "vintage carries the value, whether fed raw or as derived "
                    "PV/wind/load proxies; the timing thesis is unaffected"),
        "d43_ci95": [d43_ci["lo95"], d43_ci["hi95"]],
        "d43_dm_p": {"l3": p43_l3, "l9": p43_l9},
        "d42_total_prev2_value_over_enriched_adm": {
            "point": boot["d42_total_prev2_over_enriched"]["point_daily_mean"],
            "ci95": [d42_ci["lo95"], d42_ci["hi95"]],
            "dm_p": {"l3": p42_l3, "l9": p42_l9}},
        "d32_proxy_carried_value": {
            "point": boot["d32_proxies_over_enriched"]["point_daily_mean"],
            "ci95": [d32_ci["lo95"], d32_ci["hi95"]],
            "dm_p": {"l3": p32_l3, "l9": p32_l9}},
    }

    summary["provenance"] = {
        "occto_parquet": {"path": "data/runs/phase25.3_occto_generation.parquet",
                          "sha256": _sha256(OCCTO_PARQUET)},
        "area_capacity_csv_sha256": reference_sha256(),
        "prev2_cache_sha256_per_area": {a: m.get("prev2_cache_sha256", {})
                                        for a, m in metas.items()},
        "partial": partial,
        "areas_present": sorted(panel["area"].unique().to_list()),
    }
    summary["window"] = {
        "first_test_day": test_days[0], "last_test_day": test_days[-1],
        "n_test_days_pooled": n_days,
        "n_folds_per_area": {a: m["n_folds"] for a, m in metas.items()},
        "n_obs": int(panel.height), "n_areas": int(n_areas),
    }
    summary["results"] = {
        "pooled_mae_slot_mean": pooled_mae_slot,
        "pooled_mae_daily_mean": pooled_mae_daily,
        "deltas_daily_mean_with_ci95": boot,
        "dm_hln_nw": dm,
        "per_area": per_area,
        "runtime_s_per_area": {a: m["runtime_s"] for a, m in metas.items()},
    }
    summary["honest_framing_verdict"] = verdict
    _save_summary(summary)
    log.info("d32=%+.4f [%s, %s]  d43=%+.4f [%s, %s]  d42=%+.4f [%s, %s]",
             boot["d32_proxies_over_enriched"]["point_daily_mean"],
             d32_ci["lo95"], d32_ci["hi95"],
             boot["d43_raw_over_proxies"]["point_daily_mean"],
             d43_ci["lo95"], d43_ci["hi95"],
             boot["d42_total_prev2_over_enriched"]["point_daily_mean"],
             d42_ci["lo95"], d42_ci["hi95"])
    log.info("verdict: %s", verdict["framing"])


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["headtohead", "aggregate", "all"])
    ap.add_argument("--areas", default=None,
                    help="comma-separated subset for --step headtohead")
    ap.add_argument("--fresh", action="store_true",
                    help="drop head-to-head checkpoints before running")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-folds", type=int, default=0,
                    help="smoke-test cap on successful folds per area (0 = all)")
    args = ap.parse_args()
    t0 = time.time()
    summary = _load_summary()
    _record_precommit(summary)
    areas = args.areas.split(",") if args.areas else None

    if args.step in ("headtohead", "all"):
        step_headtohead(areas, args.fresh, args.workers, args.limit_folds)
        if args.step == "headtohead":
            return 0
    if args.step in ("aggregate", "all"):
        summary = _load_summary()
        step_aggregate(summary)
    summary = _load_summary()
    summary.setdefault("metadata", {})["total_runtime_s_last_invocation"] = round(
        time.time() - t0, 1)
    _save_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
