r"""Phase 74: the ladder re-estimated on capacity-weighted multi-point weather.

WHY THIS EXISTS
---------------
The fifth review's second must-fix: a single representative point per area may
understate fresher runs specifically -- spatial averaging error can interact
with lead -- so rejecting later-gate value from single-point weather is not
safe. This phase re-runs the run-pinned tiers with each area's weather taken
as the FIT-capacity-weighted average over its prefecture points (46 points,
area_points_capacity.csv, 2024-09 vintage), and re-tests gate timing.

DESIGN
------
The harness is phase59's verbatim: same folds, learner, seed, panel bounds.
Three new cells are ADDED (msm_d2_12z_cw, msm_gate_composite_cw,
msm_d1_00z_cw); every phase59 cell is retrained untouched, and Gate 5 requires
the phase59 cells' per-slot predictions to reproduce phase59's committed
parquet to <= 1e-9 -- if they do not, the harness has unaccounted
non-determinism and NOTHING from this phase may be reported. CW features are
attached where the multi-point caches have contract-complete days and left
NaN elsewhere (XGBoost treats NaN as missing), so the panel rows are identical
to phase59's; CW comparisons are scored on the CW-complete subset with per-day
pairing.

The CW gate composite is plan G (D-2 21Z, hours 00-21) topped up with plan A
(D-2 12Z) for hours 22-23 -- the same freshest-admissible rule as the
single-point composite, rebuilt from the multi-point caches.

Ladder features are the capacity-weighted average of each channel. The Gate 35
solar proxy additionally derates each point's irradiance by its own
temperature BEFORE weighting (the physically-correct fleet proxy); the
weight-then-derate variant is reported as a sensitivity.

GATES (docs/research/2026-08-03_phase74_cw_precommit.md)
--------------------------------------------------------
Gate 5   (blocking) phase59 cells reproduce to <= 1e-9.
Gate 35  CW becomes the manuscript's primary ONLY if the CW solar proxy beats
         the single-point proxy in Pearson r against OCCTO realised PV
         potential in >= 6 of 8 supply-covered areas.
Gate 36  gate-timing on CW tiers: 7a post-gate increment significant (both
         bootstrap CI off zero and DM p < 0.05 at L=3 and L=9) -> that is the
         NEW HEADLINE; 7b null holds (CI covers zero, one-tick equivalence
         established) -> single-point conclusion strengthened; 7c otherwise.

CLI (from the repository root):
  uv run python scripts/13_capacity_weighted_ladder_prior.py --step gate35
  uv run python ... --step build [--areas kyushu,tokyo]
  uv run python ... --step analyse
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.data.open_meteo_historical_forecast_loader import (
    load_open_meteo_historical_forecast_archive,
)
from alg.core.epf.data.open_meteo_loader import load_open_meteo_weather
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import (
    DEFAULT_WEATHER_COLS,
    attach_hourly_features,
    attach_hourly_weather,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase74")
JST = timezone(timedelta(hours=9), name="JST")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
EXACTRUN_DIR = RAW / "jma_gpv_exactrun"
EXACTRUN_MP_DIR = RAW / "jma_gpv_exactrun_mp"
PREGATE_DIR = RAW / "jma_gpv_pregate"
PREV_DIR = RAW / "open_meteo_historical_forecast"
PHASE59_PARQUET = RUN / "phase59_exact_ladder_per_slot.parquet"
OCCTO_PARQUET = RUN / "phase25.3_occto_generation.parquet"
CKPT = RUN / "phase74_checkpoints"
OUT_PARQUET = RUN / "phase74_cw_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase74_cw_ladder.json"
GATE35_JSON = RUN / "phase74_gate35_proxy.json"

# protocol pinned identically to phase51/59 -- any change voids Gate 5
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 800
SEED = 2026
PANEL_START = date(2024, 3, 2)
WEATHER_END = date(2026, 7, 24)
ERA5_END = date(2026, 7, 18)
N_BOOT = 10_000
BOOT_BLOCK = 7
GATE_TOL = 1e-9
MIN_TRAIN_ROWS = 100
WEIGHT_VINTAGE = "2024-09"
OCCTO_SHIFT_MIN = -30       # phase62's alignment of OCCTO stamps
PV_TEMP_DERATE_PER_C, PV_TEMP_REF_C = 0.004, 25.0
EQUIV_ALPHA = 0.05
TICK = 0.01
PRECOMMIT = "docs/research/2026-08-03_phase74_cw_precommit.md"

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
LEGACY_CELLS = ("price_only", "era5", "prev2", "prev1", "histfc", "oracle")
PINNED_TIERS = ("msm_d2_12z", "msm_gate_composite", "msm_d1_00z")
CW_TIERS = ("msm_d2_12z_cw", "msm_gate_composite_cw", "msm_d1_00z_cw")
# cw tier -> (mp plan dir(s)); the composite is G hours 00-21 + A hours 22-23
CW_SOURCE = {
    "msm_d2_12z_cw": ("msm_d2_12z",),
    "msm_d1_00z_cw": ("msm_d1_00z",),
    "msm_gate_composite_cw": ("msm_d2_21z", "msm_d2_12z"),
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


# ==================================================================== loaders
def _load_prev(area: str, n_prev: int) -> pl.DataFrame:
    files = sorted(PREV_DIR.glob(f"open_meteo_prev{n_prev}_jma_msm_{area}_*.parquet"))
    if not files:
        msg = f"no prev{n_prev} cache for {area} under {PREV_DIR}"
        raise FileNotFoundError(msg)
    df = (pl.concat([pl.read_parquet(f) for f in files])
          .unique("datetime_jst").sort("datetime_jst"))
    sfx = f"_pd{n_prev}"
    ren = {c: (c if c.endswith(sfx) else f"{c}{sfx}")
           for c in df.columns if c != "datetime_jst"}
    return df.rename(ren)


def _load_pinned_single(tier: str, area: str) -> pl.DataFrame:
    lat, lon = AREA_COORDS[area]
    if tier == "msm_gate_composite":
        path = PREGATE_DIR / f"jma_gpv_pregate_{area}_{lat:.4f}_{lon:.4f}.parquet"
    else:
        path = (EXACTRUN_DIR / tier
                / f"jma_gpv_{tier}_{area}_{lat:.4f}_{lon:.4f}.parquet")
    df = pl.read_parquet(path).sort("datetime_jst")
    keep = ["datetime_jst", *DEFAULT_WEATHER_COLS]
    return df.select(keep).rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS})


def _mp_point_files(plan: str, area: str) -> list[tuple[Path, float]]:
    """(parquet path, capacity weight) per prefecture point of one area."""
    pts = area_points(WEIGHT_VINTAGE)[area]
    out = []
    for i, p in enumerate(pts):
        key = f"{area}_p{i:02d}"
        path = (EXACTRUN_MP_DIR / plan
                / f"jma_gpv_{plan}_{key}_{p.lat:.4f}_{p.lon:.4f}.parquet")
        if not path.exists():
            msg = f"missing multi-point cache {path}"
            raise FileNotFoundError(msg)
        out.append((path, p.weight))
    return out


def _load_cw_plan(plan: str, area: str) -> pl.DataFrame:
    """Capacity-weighted average of every channel over the area's points."""
    acc: pl.DataFrame | None = None
    for path, w in _mp_point_files(plan, area):
        df = (pl.read_parquet(path)
              .select("datetime_jst", *DEFAULT_WEATHER_COLS)
              .with_columns(*[(pl.col(c) * w).alias(c) for c in DEFAULT_WEATHER_COLS]))
        acc = df if acc is None else (
            acc.join(df, on="datetime_jst", how="inner", suffix="_r")
               .with_columns(*[(pl.col(c) + pl.col(f"{c}_r")).alias(c)
                               for c in DEFAULT_WEATHER_COLS])
               .select("datetime_jst", *DEFAULT_WEATHER_COLS))
    return acc.sort("datetime_jst")


def _load_cw(tier: str, area: str) -> pl.DataFrame:
    if tier == "msm_gate_composite_cw":
        g = _load_cw_plan("msm_d2_21z", area).filter(
            pl.col("datetime_jst").dt.hour() <= 21)
        a = _load_cw_plan("msm_d2_12z", area).filter(
            pl.col("datetime_jst").dt.hour() >= 22)
        df = pl.concat([g, a]).unique("datetime_jst").sort("datetime_jst")
    else:
        df = _load_cw_plan(CW_SOURCE[tier][0], area)
    return df.rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS})


# ==================================================================== design
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase59's design verbatim, plus one cell per CW tier."""
    out = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        out = out.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    for c in DEFAULT_WEATHER_COLS:
        out = out.with_columns(pl.col(c).shift(48).alias(f"{c}_lag_48"))
    out = make_calendar_features(out, datetime_col="datetime_jst")
    base = [*(f"lag_{lag}" for lag in (48, 96, 336, 672)),
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
            "is_holiday"]
    cells = {
        "price_only": list(base),
        "era5": [*base, *(f"{c}_lag_48" for c in DEFAULT_WEATHER_COLS)],
        "prev2": [*base, *(f"{c}_pd2" for c in DEFAULT_WEATHER_COLS)],
        "prev1": [*base, *(f"{c}_pd1" for c in DEFAULT_WEATHER_COLS)],
        "histfc": [*base, *(f"{c}_fc" for c in DEFAULT_WEATHER_COLS)],
        "oracle": [*base, *DEFAULT_WEATHER_COLS],
    }
    for tier in (*PINNED_TIERS, *CW_TIERS):
        cells[tier] = [*base, *(f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS)]
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


# ==================================================================== build
def _run_area(area: str) -> None:
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    lat, lon = AREA_COORDS[area]
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "area", "price_yen_kwh"))
    era5 = load_open_meteo_weather(years=[2024, 2025, 2026], cache_dir=RAW,
                                   latitude=lat, longitude=lon)
    span = [PANEL_START + timedelta(days=i)
            for i in range((WEATHER_END - PANEL_START).days + 1)]
    histfc = (load_open_meteo_historical_forecast_archive(
        span, cache_dir=RAW, latitude=lat, longitude=lon)
        .select("datetime_jst", *DEFAULT_WEATHER_COLS)
        .rename({c: f"{c}_fc" for c in DEFAULT_WEATHER_COLS}))

    df = attach_hourly_weather(price.sort("datetime_jst"), era5)
    df = attach_hourly_features(df, histfc,
                                feature_cols=[f"{c}_fc" for c in DEFAULT_WEATHER_COLS])
    for n_prev, sfx in ((1, "_pd1"), (2, "_pd2")):
        df = attach_hourly_features(
            df, _load_prev(area, n_prev),
            feature_cols=[f"{c}{sfx}" for c in DEFAULT_WEATHER_COLS])
    for tier in PINNED_TIERS:
        df = attach_hourly_features(
            df, _load_pinned_single(tier, area),
            feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])
    for tier in CW_TIERS:
        df = attach_hourly_features(
            df, _load_cw(tier, area),
            feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])

    df, cells = _design(df)
    # Gate 5 requires the panel rows to be phase59's exactly: drop_nulls on the
    # phase59 feature set ONLY. CW features stay NaN where the mp caches lack a
    # day; XGBoost's missing-handling absorbs them in training and the analysis
    # step scores CW comparisons on the CW-complete subset.
    p59_cells = [c for name, c59 in cells.items() if name not in CW_TIERS
                 for c in c59]
    need = sorted({*p59_cells, "price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, ERA5_END))
          .sort("datetime_jst"))
    log.info("%s panel: %d slots %s..%s", area, df.height,
             df["datetime_jst"].min(), df["datetime_jst"].max())

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
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        cw_ok = {t: (~test[f"{DEFAULT_WEATHER_COLS[0]}_{t}"].is_nan()).to_list()
                 for t in CW_TIERS}
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": y[s],
                     **{f"cw_ok_{t}": bool(cw_ok[t][s]) for t in CW_TIERS},
                     **{f"y_pred_{n}": float(preds[n][s]) for n in cells}}
                    for s in range(test.height))
        n_folds += 1
        if n_folds % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n_folds, test_day,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
    log.info("%s done: %d folds, %d slots, %.0f s", area, n_folds, len(rows),
             time.time() - t0)


def build(areas: list[str]) -> None:
    for area in areas:
        _run_area(area)
    if set(areas) == set(AREA_COORDS):
        frames = [pl.read_parquet(CKPT / f"{a}.parquet") for a in AREA_COORDS]
        pl.concat(frames).write_parquet(OUT_PARQUET)
        log.info("wrote %s", OUT_PARQUET)


# ==================================================================== gate 35
def gate35() -> dict:
    """CW vs single-point solar proxy against OCCTO realised PV potential."""
    occ = (pl.read_parquet(OCCTO_PARQUET)
           .with_columns(pl.col("datetime_jst").dt.offset_by(f"{OCCTO_SHIFT_MIN}m"))
           .with_columns((pl.col("pv_mw") + pl.col("pv_curtailment_mw").fill_null(0.0))
                         .alias("pv_potential_mw")))
    plan = "msm_d2_12z"
    per_area: dict = {}
    wins = 0
    areas = sorted(a for a in occ["area"].unique().to_list() if a in AREA_COORDS)
    for area in areas:
        # single point
        sp = _load_pinned_single(plan, area).rename(
            {f"{c}_{plan}": c for c in DEFAULT_WEATHER_COLS})
        sp = sp.with_columns(
            (pl.col("shortwave_radiation_wm2")
             * (1.0 - PV_TEMP_DERATE_PER_C
                * (pl.col("temperature_c") - PV_TEMP_REF_C))).alias("pv_proxy_sp"))
        # multi-point: per-point derate THEN capacity-weighted sum
        acc: pl.DataFrame | None = None
        for path, w in _mp_point_files(plan, area):
            d = pl.read_parquet(path).select(
                "datetime_jst", "shortwave_radiation_wm2", "temperature_c")
            d = d.with_columns(
                (w * pl.col("shortwave_radiation_wm2")
                 * (1.0 - PV_TEMP_DERATE_PER_C
                    * (pl.col("temperature_c") - PV_TEMP_REF_C))).alias("pp"))
            d = d.select("datetime_jst", "pp")
            acc = d if acc is None else (
                acc.join(d, on="datetime_jst", how="inner", suffix="_r")
                   .with_columns((pl.col("pp") + pl.col("pp_r")).alias("pp"))
                   .select("datetime_jst", "pp"))
        cw = acc.rename({"pp": "pv_proxy_cw"}).sort("datetime_jst")
        j = (occ.filter(pl.col("area") == area)
             .join_asof(sp.select("datetime_jst", "pv_proxy_sp").sort("datetime_jst"),
                        on="datetime_jst", strategy="backward", tolerance="1h")
             .join_asof(cw, on="datetime_jst", strategy="backward", tolerance="1h")
             .drop_nulls(["pv_proxy_sp", "pv_proxy_cw", "pv_potential_mw"]))
        r_sp = float(np.corrcoef(j["pv_proxy_sp"].to_numpy(),
                                 j["pv_potential_mw"].to_numpy())[0, 1])
        r_cw = float(np.corrcoef(j["pv_proxy_cw"].to_numpy(),
                                 j["pv_potential_mw"].to_numpy())[0, 1])
        per_area[area] = {"n": int(j.height), "pearson_r_single": _j(r_sp, 4),
                          "pearson_r_cw": _j(r_cw, 4), "cw_wins": bool(r_cw > r_sp)}
        wins += int(r_cw > r_sp)
    out = {
        "phase": 74, "gate": 35,
        "criterion": "CW proxy beats single-point in >= 6 of 8 supply-covered "
                     "areas (Pearson r vs OCCTO realised PV potential); fixed in "
                     "the precommit, no metric substitution",
        "weather_plan": plan, "weight_vintage": WEIGHT_VINTAGE,
        "proxy_construction": "per-point temperature derate, then capacity "
                              "weights (fleet-physical order)",
        "per_area": per_area,
        "cw_wins": wins, "n_areas": len(areas),
        "cw_becomes_primary": bool(wins >= 6),
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "generated_utc": datetime.now(UTC).isoformat()},
    }
    GATE35_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("Gate 35: CW wins %d/%d -> primary=%s", wins, len(areas),
             out["cw_becomes_primary"])
    return out


# ==================================================================== analyse
def _hln_dm(d: np.ndarray, *, lag: int = 3) -> dict:
    n = len(d)
    d_bar = float(d.mean())
    var_hac = float(np.var(d, ddof=0))
    for ell in range(1, lag + 1):
        gamma = float(np.mean((d[:n - ell] - d_bar) * (d[ell:] - d_bar)))
        var_hac += 2.0 * (1.0 - ell / (lag + 1)) * gamma
    k = (n - 1) / n
    se = float((max(var_hac, 1e-12) * k / n) ** 0.5)
    t_stat = d_bar / se if se > 0 else float("nan")
    p = 2.0 * (1.0 - float(stats.t.cdf(abs(t_stat), df=n - 1)))
    return {"mean_diff": _j(d_bar, 5), "t_hln": _j(t_stat, 3), "p_hln": _j(p, 8),
            "n_days": n, "nw_lag": lag}


def _boot_idx(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def gate5(panel: pl.DataFrame) -> dict:
    ref = pl.read_parquet(PHASE59_PARQUET)
    common = [c for c in ref.columns if c.startswith("y_pred_") and c in panel.columns]
    j = ref.select("area", "datetime_jst", *common).join(
        panel.select("area", "datetime_jst", *common),
        on=["area", "datetime_jst"], how="inner", suffix="_74")
    worst = 0.0
    for c in common:
        diff = float((j[c] - j[f"{c}_74"]).abs().max())
        worst = max(worst, diff)
    out = {"n_rows_joined": int(j.height), "cells": common,
           "max_abs_pred_diff": worst, "tol": GATE_TOL,
           "passed": bool(worst <= GATE_TOL and j.height == ref.height)}
    if not out["passed"]:
        log.error("GATE 5 FAILED: %s", out)
    return out


def analyse() -> dict:
    panel = pl.read_parquet(OUT_PARQUET)
    g5 = gate5(panel)
    if not g5["passed"]:
        OUT_JSON.write_text(json.dumps(
            {"phase": 74, "gate5": g5,
             "aborted": "Gate 5 failed; nothing else is reportable"}, indent=2))
        return {"gate5": g5}

    # CW-complete subset: every CW tier present (per-day pairing preserved)
    sub = panel.filter(pl.all_horizontal([pl.col(f"cw_ok_{t}") for t in CW_TIERS]))
    cells = [*LEGACY_CELLS[:1], *PINNED_TIERS, *CW_TIERS]  # price_only + tiers
    per_day = (sub.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .sort("test_day"))
    n_days = per_day.height
    idx = _boot_idx(n_days)
    imp = {t: (per_day["mae_price_only"] - per_day[f"mae_{t}"]).to_numpy()
           for t in [*PINNED_TIERS, *CW_TIERS]}

    tiers_out: dict = {}
    for t, v in imp.items():
        boot = v[idx].mean(axis=1)
        tiers_out[t] = {"improvement": _j(float(v.mean()), 4),
                        "ci95": [_j(float(np.quantile(boot, 0.025)), 4),
                                 _j(float(np.quantile(boot, 0.975)), 4)],
                        "dm_l3": _hln_dm(v, lag=3), "dm_l9": _hln_dm(v, lag=9)}

    # Explicit uplift leaves: prose quotes "the better measurement is worth X",
    # and a difference of two bound macros is not itself bound, so the guard's
    # unchecked-macro ratchet rejects it unless the difference is a leaf here.
    uplift = {
        f"{sp}": _j(tiers_out[f"{sp}_cw"]["improvement"]
                    - tiers_out[sp]["improvement"], 4)
        for sp in PINNED_TIERS
    }
    n_points = sum(len(v) for v in area_points(WEIGHT_VINTAGE).values())

    # headline CW pre-gate tier: the smaller of the two, phase51/59 practice
    pre_cw = min(("msm_d2_12z_cw", "msm_gate_composite_cw"),
                 key=lambda t: tiers_out[t]["improvement"])
    d_gate = imp["msm_d1_00z_cw"] - imp[pre_cw]
    boot = d_gate[idx].mean(axis=1)
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    dm3, dm9 = _hln_dm(d_gate, lag=3), _hln_dm(d_gate, lag=9)
    sig = bool((ci[0] > 0 or ci[1] < 0) and dm3["p_hln"] < 0.05 and dm9["p_hln"] < 0.05)
    # one-tick TOST, both arms, phase67 construction
    d_bar = float(d_gate.mean())
    p_lo = float(np.mean(boot <= -TICK))
    p_hi = float(np.mean(boot >= TICK))
    tost_boot = max(p_lo, p_hi)
    tost_par = {}
    for lag, dm in (("l3", dm3), ("l9", dm9)):
        se = abs(d_bar / dm["t_hln"]) if dm["t_hln"] not in (None, 0) else None
        if se:
            t_lo = (d_bar + TICK) / se
            t_hi = (d_bar - TICK) / se
            tost_par[lag] = max(1.0 - float(stats.t.cdf(t_lo, df=n_days - 1)),
                                float(stats.t.cdf(t_hi, df=n_days - 1)))
    equiv = bool(tost_boot < EQUIV_ALPHA
                 and all(v < EQUIV_ALPHA for v in tost_par.values()))
    branch = "7a" if sig else ("7b" if (ci[0] <= 0 <= ci[1] and equiv) else "7c")

    # measurement-method interaction: does CW move the gate step vs single-point?
    d_gate_sp = imp["msm_d1_00z"] - imp[
        min(("msm_d2_12z", "msm_gate_composite"),
            key=lambda t: tiers_out[t]["improvement"])]
    d_interact = d_gate - d_gate_sp
    boot_i = d_interact[idx].mean(axis=1)

    out = {
        "phase": 74,
        "title": "Run-pinned ladder on capacity-weighted multi-point weather",
        "precommit_doc": PRECOMMIT,
        "gate5": g5,
        "panel": {"n_days_cw_complete": n_days, "n_slots": int(sub.height),
                  "weight_vintage": WEIGHT_VINTAGE},
        "tiers": tiers_out,
        "cw_uplift_by_tier": uplift,
        "n_points": n_points,
        "gate36_gate_timing_cw": {
            "headline_pre_gate_cw": pre_cw,
            "delta": _j(d_bar, 4), "ci95": [_j(v, 4) for v in ci],
            "dm_l3": dm3, "dm_l9": dm9,
            "tost_one_tick": {"boot_p": _j(tost_boot, 6),
                              **{k: _j(v, 6) for k, v in tost_par.items()},
                              "equivalent": equiv},
            "branch": branch,
            "reading": {
                "7a": "the post-gate increment is SIGNIFICANT under CW "
                      "measurement: this is the new headline and the "
                      "single-point conclusion is prominently qualified",
                "7b": "null holds and one-tick equivalence is established: the "
                      "single-point conclusion is not a measurement artefact",
                "7c": "intermediate: described, no equivalence claimed",
            }[branch],
        },
        "measurement_interaction": {
            "gate_step_cw_minus_single": _j(float(d_interact.mean()), 4),
            "ci95": [_j(float(np.quantile(boot_i, 0.025)), 4),
                     _j(float(np.quantile(boot_i, 0.975)), 4)],
            "dm_l3": _hln_dm(d_interact, lag=3),
            "reading": "the reviewer's conjecture is exactly this interaction: "
                       "positive means CW measurement reveals later-gate value "
                       "that the single point hid",
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "generated_utc": datetime.now(UTC).isoformat(),
                     "seed": SEED, "n_boot": N_BOOT},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("wrote %s", OUT_JSON)
    for t in [*PINNED_TIERS, *CW_TIERS]:
        r = tiers_out[t]
        log.info("%-24s imp %.4f %s", t, r["improvement"], r["ci95"])
    log.info("Gate 36 branch %s: delta %.4f %s equiv=%s", branch, d_bar,
             out["gate36_gate_timing_cw"]["ci95"], equiv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["build", "gate35", "analyse"])
    ap.add_argument("--areas", default=",".join(AREA_COORDS))
    args = ap.parse_args()
    if args.step == "build":
        build([a.strip() for a in args.areas.split(",") if a.strip()])
    elif args.step == "gate35":
        gate35()
    else:
        analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
