r"""Phase 85: the ladder re-estimated CW-PRIMARY on >=730 test days.

WHY THIS EXISTS
---------------
Gate 35 (docs/research/2026-08-03_phase74_cw_precommit.md) pre-registered that a
capacity-weighted 46-point measurement winning the proxy validation in >= 6 of 8
areas makes CW the manuscript's primary measurement. It fired 7/8 on 2026-08-04
and was not executed; the sixth external review independently demanded the
promotion. This phase EXECUTES the gate (it does not re-evaluate it) and, in the
same sweep, extends the panel past the Lago >=730-day benchmark the paper cites.
Precommit: docs/research/2026-08-11_phase85_cw_primary_precommit.md.

CELLS (14, one build)
---------------------
CW-primary arm (all reported contrasts stay inside one measurement width):
    msm_d2_12z_cw / msm_gate_composite_cw / msm_d1_00z_cw   (mp GPV caches)
    oracle_cw   capacity-weighted realised ERA5 over the 46 prefecture points
    era5_cw     the same series at lag-48 slots (the stale-proxy rung)
Shared, measurement-free:
    price_only  (price lags + calendar only; trained once)
Single-point robustness arm (phase59's cells, bit-repro anchored):
    era5, prev2, prev1, histfc, oracle, msm_d2_12z, msm_gate_composite, msm_d1_00z

FREEZE POLICY (Gate 45a)
------------------------
Frozen caches are never re-fetched. Extensions live in data/raw/open_meteo_ext1/
(SP tiers) and data/raw/open_meteo_mp/ (46-point ERA5), written by
05_extend_panel.py. For frozen dates the FROZEN values are
used unconditionally; extension rows are admitted strictly after each frozen
span's last stamp. Overlap days are compared and reported as an ERA5T-revision
diagnostic, never silently merged.

GATES 45-50 -- see the precommit. 45b/45c are BLOCKING bit-reproduction of
phase59 (9 SP cells) and phase74 (3 CW cells) on the frozen 714-day window.

CLI (from the repository root):
  uv run --extra research python scripts/10_primary_ladder_cw.py \
      --step build
  uv run --extra research python ... --step analyse
  uv run --extra research python ... --step build --areas kyushu   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
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
from alg.core.epf.data.area_capacity_loader import peak_gw_by_area, pv_gw_by_area
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
log = logging.getLogger("phase85")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
EXACTRUN_DIR = RAW / "jma_gpv_exactrun"
EXACTRUN_MP_DIR = RAW / "jma_gpv_exactrun_mp"
PREGATE_DIR = RAW / "jma_gpv_pregate"
PREV_DIR = RAW / "open_meteo_historical_forecast"
EXT_DIR = RAW / "open_meteo_ext1"
MP_ERA5_DIR = RAW / "open_meteo_mp"
PHASE59_PARQUET = RUN / "phase59_exact_ladder_per_slot.parquet"
PHASE74_PARQUET = RUN / "phase74_cw_ladder_per_slot.parquet"
CKPT = RUN / "phase85_checkpoints"
OUT_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase85_exact_ladder.json"
PRECOMMIT = "docs/research/2026-08-11_phase85_cw_primary_precommit.md"

# protocol constants: identical to phase51/59/74 EXCEPT the two end dates,
# which the precommit extends (frozen-window folds are unchanged because every
# preprocessing step is row-local; Gate 45b verifies this bit-for-bit).
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 900
SEED = 2026
PANEL_START = date(2024, 3, 2)
WEATHER_END = date(2026, 8, 8)    # phase85 precommit
ERA5_END = date(2026, 8, 5)       # phase85 precommit; end-date rule if < 730 days
FROZEN_ERA5_END = date(2026, 7, 18)   # phase51/59/74 freeze boundary
FROZEN_PREV_END = date(2026, 7, 24)
N_BOOT = 10_000
BOOT_BLOCK = 7
GATE_TOL = 1e-9
MIN_TRAIN_ROWS = 100
ALPHA = 0.05
EQUIV_ALPHA = 0.05
MIN_TEST_DAYS = 730
WEIGHT_VINTAGE = "2024-09"

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
API_TO_CANON = {"temperature_2m": "temperature_c", "cloud_cover": "cloud_cover_pct",
                "shortwave_radiation": "shortwave_radiation_wm2",
                "wind_speed_10m": "wind_speed_kmh"}
SP_CELLS = ("price_only", "era5", "prev2", "prev1", "histfc", "oracle",
            "msm_d2_12z", "msm_gate_composite", "msm_d1_00z")
SP_PINNED: dict[str, dict] = {
    "msm_d2_12z": {"kind": "exactrun", "plan": "msm_d2_12z", "mean_lead_h": 38.5,
                   "pre_gate": True},
    "msm_gate_composite": {"kind": "pregate", "plan": None, "mean_lead_h": 30.25,
                           "pre_gate": True},
    "msm_d1_00z": {"kind": "exactrun", "plan": "msm_d1_00z", "mean_lead_h": 26.5,
                   "pre_gate": False},
}
CW_JMA_TIERS = ("msm_d2_12z_cw", "msm_gate_composite_cw", "msm_d1_00z_cw")
CW_SOURCE = {
    "msm_d2_12z_cw": ("msm_d2_12z",),
    "msm_d1_00z_cw": ("msm_d1_00z",),
    "msm_gate_composite_cw": ("msm_d2_21z", "msm_d2_12z"),
}
CW_ERA5_REAL = "cw_era5_realised"   # attached column stem for the CW ERA5 series
CW_CELLS = (*CW_JMA_TIERS, "oracle_cw", "era5_cw")
SW = "shortwave_radiation_wm2"
PHASE74_REF_CELLS = CW_JMA_TIERS   # cells Gate 45c compares against phase74


def _j(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ==================================================================== loaders
def _parse_om_json(path: Path) -> pl.DataFrame:
    """One Open-Meteo hourly payload -> canonical frame (datetime_jst + 4 cols)."""
    h = json.loads(path.read_text())["hourly"]
    return pl.DataFrame({
        "datetime_jst": [datetime.fromisoformat(t) for t in h["time"]],
        **{canon: [float(x) if x is not None else None for x in h[api]]
           for api, canon in API_TO_CANON.items()},
    }).with_columns(pl.col("datetime_jst").dt.replace_time_zone("Asia/Tokyo")
                    .dt.cast_time_unit("us"))


def _frozen_plus_ext(frozen: pl.DataFrame, ext: pl.DataFrame, boundary: date,
                     label: str, diagnostics: dict) -> pl.DataFrame:
    """Frozen rows <= boundary, extension rows strictly after; report overlap drift.

    The frozen values are authoritative inside the frozen window (Gate 45b depends
    on them). The extension payload deliberately overlaps the boundary day so the
    provider-side revision (ERA5T -> ERA5) is measurable; the overlap is compared
    on the shared columns and the max abs drift recorded, then discarded.
    """
    cols = [c for c in frozen.columns if c != "datetime_jst" and c in ext.columns]
    cut = datetime(boundary.year, boundary.month, boundary.day, 23, 59, tzinfo=JST)
    ov = (ext.filter(pl.col("datetime_jst") <= cut)
             .join(frozen, on="datetime_jst", how="inner", suffix="_f"))
    if ov.height and cols:
        drift = max(float((ov[c] - ov[f"{c}_f"]).abs().max() or 0.0) for c in cols)
        diagnostics.setdefault("overlap_drift", {})[label] = {
            "n_overlap_rows": ov.height, "max_abs_drift": _j(drift, 6)}
    out = pl.concat([frozen.filter(pl.col("datetime_jst") <= cut),
                     ext.filter(pl.col("datetime_jst") > cut).select(frozen.columns)])
    return out.unique("datetime_jst").sort("datetime_jst")


def _load_era5_sp(area: str, lat: float, lon: float, diagnostics: dict) -> pl.DataFrame:
    frozen = load_open_meteo_weather(years=[2024, 2025, 2026], cache_dir=RAW,
                                     latitude=lat, longitude=lon)
    ext_files = sorted(EXT_DIR.glob(f"era5_ext1_{area}_*.json"))
    if not ext_files:
        msg = f"missing ERA5 extension for {area} under {EXT_DIR}"
        raise FileNotFoundError(msg)
    ext = _parse_om_json(ext_files[0]).select(frozen.columns)
    return _frozen_plus_ext(frozen, ext, FROZEN_ERA5_END, f"era5_{area}", diagnostics)


def _load_histfc_sp(area: str, lat: float, lon: float, diagnostics: dict) -> pl.DataFrame:
    span = [PANEL_START + timedelta(days=i)
            for i in range((FROZEN_PREV_END - PANEL_START).days + 1)]
    frozen = (load_open_meteo_historical_forecast_archive(
        span, cache_dir=RAW, latitude=lat, longitude=lon)
        .select("datetime_jst", *DEFAULT_WEATHER_COLS))
    ext_files = sorted(EXT_DIR.glob(f"histfc_ext1_jma_msm_{area}_*.json"))
    if not ext_files:
        msg = f"missing histfc extension for {area} under {EXT_DIR}"
        raise FileNotFoundError(msg)
    ext = _parse_om_json(ext_files[0]).select(frozen.columns)
    merged = _frozen_plus_ext(frozen, ext, FROZEN_PREV_END, f"histfc_{area}", diagnostics)
    return merged.rename({c: f"{c}_fc" for c in DEFAULT_WEATHER_COLS})


def _load_prev(area: str, n_prev: int) -> pl.DataFrame:
    """Frozen prev caches + the disjoint extension parquet (rows > frozen end)."""
    frozen_files = sorted(PREV_DIR.glob(f"open_meteo_prev{n_prev}_jma_msm_{area}_*.parquet"))
    if not frozen_files:
        msg = f"no prev{n_prev} cache for {area} under {PREV_DIR}"
        raise FileNotFoundError(msg)
    ext_files = sorted(EXT_DIR.glob(f"open_meteo_prev{n_prev}_ext1_jma_msm_{area}_*.parquet"))
    if not ext_files:
        msg = f"missing prev{n_prev} extension for {area} under {EXT_DIR}"
        raise FileNotFoundError(msg)
    df = (pl.concat([pl.read_parquet(f) for f in [*frozen_files, *ext_files]])
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
        plan = SP_PINNED[tier]["plan"]
        path = (EXACTRUN_DIR / plan
                / f"jma_gpv_{plan}_{area}_{lat:.4f}_{lon:.4f}.parquet")
    if not path.exists():
        msg = f"missing pinned cache for tier {tier}, area {area}: {path}"
        raise FileNotFoundError(msg)
    df = pl.read_parquet(path).sort("datetime_jst")
    return (df.select("datetime_jst", *DEFAULT_WEATHER_COLS)
              .rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS}))


def _mp_point_files(plan: str, area: str) -> list[tuple[Path, float]]:
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


def _load_cw_jma(tier: str, area: str) -> pl.DataFrame:
    if tier == "msm_gate_composite_cw":
        g = _load_cw_plan("msm_d2_21z", area).filter(
            pl.col("datetime_jst").dt.hour() <= 21)
        a = _load_cw_plan("msm_d2_12z", area).filter(
            pl.col("datetime_jst").dt.hour() >= 22)
        df = pl.concat([g, a]).unique("datetime_jst").sort("datetime_jst")
    else:
        df = _load_cw_plan(CW_SOURCE[tier][0], area)
    return df.rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS})


def _load_cw_era5(area: str) -> pl.DataFrame:
    """Capacity-weighted realised ERA5 over the area's 46-point subset."""
    pts = area_points(WEIGHT_VINTAGE)[area]
    acc: pl.DataFrame | None = None
    for p in pts:
        lat_tok = f"{p.lat:.4f}".replace(".", "p").replace("-", "m")
        lon_tok = f"{p.lon:.4f}".replace(".", "p").replace("-", "m")
        files = sorted(MP_ERA5_DIR.glob(f"era5_mp_{area}_{lat_tok}_{lon_tok}_*.json"))
        if len(files) != 3:
            msg = (f"expected 3 mp-ERA5 year files for {area}/"
                   f"{p.prefecture}, got {len(files)}")
            raise FileNotFoundError(msg)
        df = (pl.concat([_parse_om_json(f) for f in files])
              .unique("datetime_jst").sort("datetime_jst")
              .with_columns(*[(pl.col(c) * p.weight).alias(c)
                              for c in DEFAULT_WEATHER_COLS]))
        acc = df if acc is None else (
            acc.join(df, on="datetime_jst", how="inner", suffix="_r")
               .with_columns(*[(pl.col(c) + pl.col(f"{c}_r")).alias(c)
                               for c in DEFAULT_WEATHER_COLS])
               .select("datetime_jst", *DEFAULT_WEATHER_COLS))
    return acc.sort("datetime_jst").rename(
        {c: f"{c}_{CW_ERA5_REAL}" for c in DEFAULT_WEATHER_COLS})


# ==================================================================== design
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase59's design verbatim, plus the five CW cells.

    era5_cw is the lag-48 shift of the attached CW-ERA5 realised series,
    mirroring exactly how the SP era5 cell is built from the SP realised
    series (shift over half-hour slots, applied after the hourly attach).
    """
    out = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        out = out.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    for c in DEFAULT_WEATHER_COLS:
        out = out.with_columns(pl.col(c).shift(48).alias(f"{c}_lag_48"))
        out = out.with_columns(
            pl.col(f"{c}_{CW_ERA5_REAL}").shift(48).alias(f"{c}_{CW_ERA5_REAL}_lag_48"))
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
    for tier in SP_PINNED:
        cells[tier] = [*base, *(f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS)]
    for tier in CW_JMA_TIERS:
        cells[tier] = [*base, *(f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS)]
    cells["oracle_cw"] = [*base, *(f"{c}_{CW_ERA5_REAL}" for c in DEFAULT_WEATHER_COLS)]
    cells["era5_cw"] = [*base, *(f"{c}_{CW_ERA5_REAL}_lag_48"
                                 for c in DEFAULT_WEATHER_COLS)]
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


# ==================================================================== per area
def _run_area(area: str) -> None:  # noqa: C901 - the fold loop mirrors phase59/74 verbatim
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    diagnostics: dict = {}
    lat, lon = AREA_COORDS[area]
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "area", "price_yen_kwh"))
    era5 = _load_era5_sp(area, lat, lon, diagnostics)
    histfc = _load_histfc_sp(area, lat, lon, diagnostics)

    df = attach_hourly_weather(price.sort("datetime_jst"), era5)
    df = attach_hourly_features(df, histfc,
                                feature_cols=[f"{c}_fc" for c in DEFAULT_WEATHER_COLS])
    for n_prev, sfx in ((1, "_pd1"), (2, "_pd2")):
        df = attach_hourly_features(
            df, _load_prev(area, n_prev),
            feature_cols=[f"{c}{sfx}" for c in DEFAULT_WEATHER_COLS])
    for tier in SP_PINNED:
        df = attach_hourly_features(
            df, _load_pinned_single(tier, area),
            feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])
    for tier in CW_JMA_TIERS:
        df = attach_hourly_features(
            df, _load_cw_jma(tier, area),
            feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])
    df = attach_hourly_features(
        df, _load_cw_era5(area),
        feature_cols=[f"{c}_{CW_ERA5_REAL}" for c in DEFAULT_WEATHER_COLS])

    df, cells = _design(df)
    # Gate 45b requires the row set to be phase59's exactly on the frozen window:
    # drop_nulls over the SP feature set ONLY. CW features stay NaN where a
    # cache lacks a day; cw_ok flags mark them and the analysis scores CW
    # contrasts on the CW-complete subset (phase74's construction).
    sp_cols = [c for name in SP_CELLS for c in cells[name]]
    need = sorted({*sp_cols, "price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, ERA5_END))
          .sort("datetime_jst"))
    log.info("%s panel: %d slots %s..%s | overlap drift: %s", area, df.height,
             df["datetime_jst"].min(), df["datetime_jst"].max(),
             diagnostics.get("overlap_drift"))

    pvpeak = pv_gw_by_area()[area] / peak_gw_by_area()[area]
    panel_last = df["datetime_jst"].max().date()
    rows: list[dict] = []
    t0, n_folds = time.time(), 0
    cw_flag_cols = {t: f"{DEFAULT_WEATHER_COLS[0]}_{t}" for t in CW_JMA_TIERS}
    cw_flag_cols["oracle_cw"] = f"{DEFAULT_WEATHER_COLS[0]}_{CW_ERA5_REAL}"
    cw_flag_cols["era5_cw"] = f"{DEFAULT_WEATHER_COLS[0]}_{CW_ERA5_REAL}_lag_48"
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
        cw_ok = {t: (~test[col].is_nan() & test[col].is_not_null()).to_list()
                 for t, col in cw_flag_cols.items()}
        # Carry the shortwave column of every tier (phase59 convention, extended
        # to the CW cells): Paper B's decision model builds its generation
        # forecast from the tier's own irradiance, and phase74's parquet lacking
        # these columns is exactly why its CW arm could not drive anything
        # downstream.
        sw_cols = {f"{SW}_{t}": test[f"{SW}_{t}"].to_numpy().astype(float)
                   for t in SP_PINNED}
        sw_cols.update({f"{SW}_{t}": test[f"{SW}_{t}"].to_numpy().astype(float)
                        for t in CW_JMA_TIERS})
        sw_cols[f"{SW}_oracle_cw"] = (
            test[f"{SW}_{CW_ERA5_REAL}"].to_numpy().astype(float))
        sw_cols[f"{SW}_era5_cw"] = (
            test[f"{SW}_{CW_ERA5_REAL}_lag_48"].to_numpy().astype(float))
        for legacy in (f"{SW}_pd2", f"{SW}_pd1", f"{SW}_fc", f"{SW}_lag_48", SW):
            if legacy in test.columns:
                sw_cols[legacy] = test[legacy].to_numpy().astype(float)
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": y[s],
                     **{f"cw_ok_{t}": bool(cw_ok[t][s]) for t in cw_flag_cols},
                     **{c: float(v[s]) for c, v in sw_cols.items()},
                     "pv_peak_ratio": pvpeak,
                     **{f"y_pred_{n}": float(preds[n][s]) for n in cells}}
                    for s in range(test.height))
        n_folds += 1
        if n_folds % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n_folds, test_day,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    out = pl.DataFrame(rows)
    out.write_parquet(ckpt)
    (CKPT / f"{area}_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1))
    log.info("%s done: %d folds, %d slots, %.0f s", area, n_folds, len(rows),
             time.time() - t0)


# ==================================================================== statistics
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


def _block_boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    idx = idx.reshape(N_BOOT, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _per_day_mae(panel: pl.DataFrame, cells: list[str]) -> pl.DataFrame:
    return (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .sort("area", "test_day"))


def _pooled_daily(per_day: pl.DataFrame, cell: str) -> np.ndarray:
    return np.asarray(
        (per_day.with_columns(
            (pl.col("mae_price_only") - pl.col(f"mae_{cell}")).alias("imp"))
         .group_by("test_day").agg(pl.col("imp").mean())
         .sort("test_day")["imp"]), dtype=float)


def _holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the caller's order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    n, out, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * ps[i]))
        out[i] = running
    return out


def _benjamini_hochberg(ps: list[float]) -> list[float]:
    """BH step-up adjusted p-values, in the caller's order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)
    n, out, running = len(ps), [0.0] * len(ps), 1.0
    for rank, i in enumerate(order):
        running = min(running, min(1.0, ps[i] * n / (n - rank)))
        out[i] = running
    return out


def _entry(d: np.ndarray) -> dict:
    lo, hi = _block_boot_ci(d)
    return {"improvement": _j(float(d.mean()), 4),
            "ci95": [_j(lo, 4), _j(hi, 4)],
            "dm_l3": _hln_dm(d, lag=3), "dm_l9": _hln_dm(d, lag=9),
            "n_days": len(d)}


def _ratio_entry(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    idx = idx.reshape(N_BOOT, -1)[:, :n]
    num, den = a[idx].mean(axis=1), b[idx].mean(axis=1)
    ok = den > 0
    draws = num[ok] / den[ok]
    return {"ratio": _j(float(a.mean() / b.mean()), 3),
            "ci95": [_j(float(np.quantile(draws, 0.025)), 3),
                     _j(float(np.quantile(draws, 0.975)), 3)],
            "n_draws_used": int(ok.sum()), "n_days": n}


def _tost(d: np.ndarray, margins: tuple[tuple[str, float, str], ...],
          *, seed: int = SEED) -> dict:
    """Run phase59/67's TOST construction with the margins passed in."""
    n = len(d)
    d_bar = float(d.mean())
    out: dict = {"alpha": EQUIV_ALPHA, "n_days": n,
                 "precommit_doc": PRECOMMIT, "margins": {}}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    boot = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    ci90 = [_j(float(np.quantile(boot, 0.05)), 5),
            _j(float(np.quantile(boot, 0.95)), 5)]
    out["ci90"] = ci90
    for name, margin, rationale in margins:
        m: dict = {"delta_margin": _j(margin, 5), "rationale": rationale}
        p_lo = float(np.mean(boot <= -margin))
        p_hi = float(np.mean(boot >= margin))
        p_boot = max(p_lo, p_hi)
        m["boot"] = {"p_lower": _j(p_lo, 6), "p_upper": _j(p_hi, 6),
                     "p_tost": _j(p_boot, 6), "equivalent": bool(p_boot < EQUIV_ALPHA)}
        for lag in (3, 9):
            dm = _hln_dm(d, lag=lag)
            se = None
            if dm["t_hln"] is not None and dm["t_hln"] != 0:
                se = abs(d_bar / dm["t_hln"])
            if se is None or se <= 0:
                m[f"dm_l{lag}"] = {"se": None, "p_tost": None, "equivalent": None}
                continue
            t_lo = (d_bar + margin) / se
            t_hi = (d_bar - margin) / se
            p_l = 1.0 - float(stats.t.cdf(t_lo, df=n - 1))
            p_u = float(stats.t.cdf(t_hi, df=n - 1))
            p_par = max(p_l, p_u)
            m[f"dm_l{lag}"] = {"se": _j(se, 6), "t_lower": _j(t_lo, 4),
                               "t_upper": _j(t_hi, 4), "p_lower": _j(p_l, 8),
                               "p_upper": _j(p_u, 8), "p_tost": _j(p_par, 8),
                               "equivalent": bool(p_par < EQUIV_ALPHA)}
        arms = [m["boot"]["equivalent"]] + [m[f"dm_l{k}"].get("equivalent")
                                            for k in (3, 9)]
        m["verdict"] = ("EQUIVALENT" if all(a is True for a in arms)
                        else "NOT_EQUIVALENT")
        m["ci90_within_margin"] = bool(
            ci90[0] is not None and ci90[1] is not None
            and -margin < ci90[0] and ci90[1] < margin)
        if ci90[0] is not None and ci90[1] is not None and margin > 0:
            m["margin_utilisation"] = _j(
                max(abs(ci90[0]), abs(ci90[1])) / margin, 3)
        out["margins"][name] = m
    if ci90[0] is not None and ci90[1] is not None:
        out["smallest_margin_supported"] = _j(
            max(abs(ci90[0]), abs(ci90[1])), 5)
        out["smallest_margin_note"] = (
            "smallest symmetric margin whose interval contains the 90% CI; a "
            "DESCRIPTIVE by-product, NOT a pre-committed margin")
    return out


def _margins_for(headline_imp: float) -> tuple[tuple[str, float, str], ...]:
    """Build the precommit margins: one tick + 20% of the CW headline BY FORMULA."""
    return (
        ("tick", 0.01,
         "one JEPX price tick, the market's own price granularity; a smallness "
         "benchmark for the POOLED MEAN absolute error only -- mean-level "
         "equivalence does not bound individual slot forecast differences "
         "(phase75 tested the decision scale separately and found "
         "non-equivalence at the reviewer's margins)"),
        ("relevance", 0.20 * headline_imp,
         "20% of the CW pre-gate headline improvement, fixed as a FORMULA in the "
         "phase85 precommit before estimation; SECONDARY because it is anchored "
         "on an estimate"),
    )


def _diff_entry(a: np.ndarray, b: np.ndarray,
                margins: tuple[tuple[str, float, str], ...] | None = None) -> dict:
    d = a - b
    lo, hi = _block_boot_ci(d)
    out = {"delta": _j(float(d.mean()), 4), "ci95": [_j(lo, 4), _j(hi, 4)],
           "dm_l3": _hln_dm(d, lag=3), "dm_l9": _hln_dm(d, lag=9),
           "n_days": len(d)}
    if margins is not None:
        out["equivalence"] = _tost(d, margins)
    return out


# ==================================================================== gates
def _gate45(panel: pl.DataFrame) -> dict:
    """Run gates 45b/45c, the BLOCKING bit-reproduction checks.

    SP cells reproduce phase59, CW JMA cells reproduce phase74, on the
    frozen 714-day window.
    """
    out: dict = {"frozen_window_end": FROZEN_ERA5_END.isoformat(),
                 "tolerance": GATE_TOL}
    frozen = panel.filter(pl.col("test_day") <= FROZEN_ERA5_END.isoformat())
    key = ["area", "datetime_jst"]
    for label, ref_path, cells in (
            ("45b_vs_phase59", PHASE59_PARQUET, SP_CELLS),
            ("45c_vs_phase74", PHASE74_PARQUET, PHASE74_REF_CELLS)):
        if not ref_path.exists():
            out[label] = {"verdict": "SKIP", "reason": f"{ref_path.name} not found"}
            continue
        ref = pl.read_parquet(ref_path)
        joined = frozen.select(
            *key, *[f"y_pred_{c}" for c in cells], "y_actual").join(
            ref.select(*key, *[pl.col(f"y_pred_{c}").alias(f"ref_{c}") for c in cells],
                       pl.col("y_actual").alias("ref_y")),
            on=key, how="inner")
        worst = {c: float((joined[f"y_pred_{c}"] - joined[f"ref_{c}"]).abs().max()
                          or 0.0) for c in cells}
        worst["y_actual"] = float((joined["y_actual"] - joined["ref_y"]).abs().max()
                                  or 0.0)
        rec = {"reference": ref_path.name, "reference_sha256": _sha256(ref_path),
               "n_slots_reference": ref.height, "n_slots_frozen_window": frozen.height,
               "n_slots_matched": joined.height,
               "row_set_identical": bool(joined.height == ref.height == frozen.height),
               "max_abs_diff": {k: _j(v, 12) for k, v in worst.items()}}
        rec["verdict"] = ("PASS" if rec["row_set_identical"]
                          and all(v <= GATE_TOL for v in worst.values()) else "FAIL")
        out[label] = rec
    verdicts = [out[k].get("verdict") for k in ("45b_vs_phase59", "45c_vs_phase74")]
    out["verdict"] = "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"
    out["consequence_if_fail"] = (
        "Abort. Either a frozen input moved (45a violation) or the harness has "
        "unaccounted non-determinism; the new numbers cannot be trusted either.")
    return out


# ==================================================================== analysis
def _arm_report(per_day: pl.DataFrame, daily: dict[str, np.ndarray],  # noqa: PLR0913 - one explicit knob per arm design choice
                pre_gate_tiers: list[str], post_gate_tier: str,
                oracle_cell: str | None, stale_cell: str | None,
                areas: list[str], gate_prefix: str) -> dict:
    """Report one measurement arm inside its own measurement width.

    Tier table, pre-gate value gate, gate-timing branches, decomposition.
    """
    arm: dict = {"tiers": {c: _entry(d) for c, d in daily.items()}}
    for c in daily:
        rec = {}
        for a in areas:
            sub = per_day.filter(pl.col("area") == a)
            d = np.asarray(sub["mae_price_only"] - sub[f"mae_{c}"], dtype=float)
            rec[a] = {"improvement": _j(float(d.mean()), 4),
                      "p_l3": _hln_dm(d, lag=3)["p_hln"]}
        arm["tiers"][c]["areas_improved"] = sum(
            1 for v in rec.values()
            if v["improvement"] is not None and v["improvement"] > 0)
        arm["tiers"][c]["areas_sig_l3"] = sum(
            1 for v in rec.values() if v["p_l3"] is not None and v["p_l3"] < ALPHA)
        # ADDED 2026-08-18. Methods states "where one test is run per area we
        # report Holm-adjusted and Benjamini-Hochberg-adjusted p-values", and
        # this is exactly that family -- one HLN-DM test per area. Only the RAW
        # count existed, and Section 5.1 quoted it: for the stale-observation
        # tier that is 8 of 9 raw but 7 of 9 after Holm (Shikoku p = 0.0384 ->
        # 0.0768). Record the adjusted p-values and both counts so the text can
        # report what the Methods promise.
        _names = [a for a in areas if rec[a]["p_l3"] is not None]
        _ps = [rec[a]["p_l3"] for a in _names]
        for _a, _padj in zip(_names, _holm(_ps), strict=True):
            rec[_a]["p_l3_holm"] = _j(_padj, 6)
        for _a, _padj in zip(_names, _benjamini_hochberg(_ps), strict=True):
            rec[_a]["p_l3_bh"] = _j(_padj, 6)
        arm["tiers"][c]["areas_sig_l3_holm"] = sum(
            1 for v in rec.values()
            if v.get("p_l3_holm") is not None and v["p_l3_holm"] < ALPHA)
        arm["tiers"][c]["areas_sig_l3_bh"] = sum(
            1 for v in rec.values()
            if v.get("p_l3_bh") is not None and v["p_l3_bh"] < ALPHA)
        arm.setdefault("per_area", {})[c] = rec

    # pre-gate value gate (46 for CW / part of 50 for SP)
    g: dict = {"candidates": {}}
    for t in pre_gate_tiers:
        e = arm["tiers"][t]
        g["candidates"][t] = {
            "improvement": e["improvement"], "ci95": e["ci95"],
            "dm_p_l3": e["dm_l3"]["p_hln"], "dm_p_l9": e["dm_l9"]["p_hln"],
            "passes": bool(
                e["ci95"][0] is not None and e["ci95"][0] > 0
                and e["dm_l3"]["p_hln"] is not None and e["dm_l3"]["p_hln"] < ALPHA
                and e["dm_l9"]["p_hln"] is not None and e["dm_l9"]["p_hln"] < ALPHA),
        }
    headline = min(g["candidates"], key=lambda t: g["candidates"][t]["improvement"])
    g["headline_tier"] = headline
    g["headline_rule"] = ("the smaller of the two pre-gate tiers, fixed before "
                          "results (conservative, phase51/59 rule)")
    g["verdict"] = ("PASS" if all(v["passes"] for v in g["candidates"].values())
                    else "FAIL")
    arm[f"{gate_prefix}_pregate_value"] = g

    # gate timing with pre-registered branches (47 for CW / disclosure for SP)
    margins = _margins_for(arm["tiers"][headline]["improvement"])
    d = _diff_entry(daily[post_gate_tier], daily[headline], margins)
    lo, hi = d["ci95"]
    sig = (lo is not None and lo > 0
           and d["dm_l3"]["p_hln"] is not None and d["dm_l3"]["p_hln"] < ALPHA
           and d["dm_l9"]["p_hln"] is not None and d["dm_l9"]["p_hln"] < ALPHA)
    tick_equiv = d["equivalence"]["margins"]["tick"]["verdict"] == "EQUIVALENT"
    if sig:
        branch, reading = "a", ("the excluded run adds detectable pooled "
                                "point-forecast value at this measurement -- the "
                                "gate-timing null is overturned and becomes the "
                                "headline")
    elif (lo is not None and lo <= 0 <= (hi if hi is not None else 0)) and tick_equiv:
        branch, reading = "b", ("CI covers zero and the one-tick TOST passes all "
                                "three arms: the null holds, equivalence claimed "
                                "for the POOLED MEAN at the tick margin only")
    else:
        branch, reading = "c", ("neither significant nor equivalent at the tick: "
                                "described, no equivalence claim")
    arm[f"{gate_prefix}_gate_timing"] = {
        "post_gate_tier": post_gate_tier, "pre_gate_tier": headline, **d,
        "branch": branch, "reading": reading,
        "upper_bound_if_b": hi,
    }
    arm["headline_retention_of_post_gate_pct"] = _j(
        100.0 * arm["tiers"][headline]["improvement"]
        / arm["tiers"][post_gate_tier]["improvement"], 1)
    # per-area gate timing with the tick TOST (phase59's per-area construction)
    gt_areas: dict = {}
    for a in areas:
        sub = per_day.filter(pl.col("area") == a)
        dd = np.asarray(sub[f"mae_{headline}"] - sub[f"mae_{post_gate_tier}"],
                        dtype=float)
        e = _tost(dd, margins)
        gt_areas[a] = {"delta": _j(float(dd.mean())),
                       "tick_verdict": e["margins"]["tick"]["verdict"]}
    arm["per_area_gate_timing"] = {
        "areas": gt_areas,
        "n_areas_equivalent_at_tick": sum(
            1 for v in gt_areas.values() if v["tick_verdict"] == "EQUIVALENT"),
    }

    # decomposition (48 for CW / mirror for SP), all within-arm
    dec: dict = {"label_rule": (
        "The stale-proxy multiple is a COMPOSITE contrast of product, vintage and "
        "availability. It is never to be called a vintage effect on its own.")}
    if oracle_cell and stale_cell and oracle_cell in daily and stale_cell in daily:
        dec["within_era5_vintage_effect"] = _diff_entry(daily[oracle_cell],
                                                        daily[stale_cell])
        dec["product_bound"] = _diff_entry(daily[headline], daily[oracle_cell])
        imp_e = arm["tiers"][stale_cell]["improvement"]
        imp_h = arm["tiers"][headline]["improvement"]
        dec["composite_multiple"] = _j(imp_h / imp_e, 3) if imp_e else None
        dec["headline_vs_stale"] = {
            "difference": _diff_entry(daily[headline], daily[stale_cell]),
            "ratio": _ratio_entry(daily[headline], daily[stale_cell]),
        }
    arm[f"{gate_prefix}_decomposition"] = dec
    return arm


def analyse() -> dict:
    files = sorted(f for f in CKPT.glob("*.parquet"))
    if not files:
        msg = f"no checkpoints under {CKPT}; run --step build first"
        raise FileNotFoundError(msg)
    panel = pl.concat([pl.read_parquet(f) for f in files]).sort("area", "datetime_jst")
    panel.write_parquet(OUT_PARQUET)
    areas = sorted(panel["area"].unique().to_list())
    n_days = panel["test_day"].n_unique()
    log.info("panel: %d slots / %d areas / %d test days", panel.height, len(areas),
             n_days)

    out: dict = {
        "phase": 85,
        "title": "CW-primary vintage ladder on >=730 test days (Gate 35 executed)",
        "precommit_doc": PRECOMMIT,
        "gate35_execution_note": (
            "Gate 35 (2026-08-03 precommit) fired 7/8 on 2026-08-04 and is "
            "EXECUTED here, not re-evaluated. The 2026-08-04 deferral and its "
            "reasons are disclosed in the results doc and the manuscript ledger."),
        "panel": {"n_slots": panel.height, "n_areas": len(areas),
                  "n_test_days": n_days,
                  "first_test_day": panel["test_day"].min(),
                  "last_test_day": panel["test_day"].max(),
                  "slots_per_area": int(panel.height / len(areas))},
    }

    # ---- Gate 45 first: nothing else is reportable if it fails
    out["gate45_reproducibility"] = _gate45(panel)
    log.info("GATE 45: %s", out["gate45_reproducibility"]["verdict"])
    if out["gate45_reproducibility"]["verdict"] == "FAIL":
        out["aborted"] = True
        return out

    # ---- Gate 49: panel size
    out["gate49_panel_size"] = {
        "n_test_days": n_days, "required": MIN_TEST_DAYS,
        "verdict": "PASS" if n_days >= MIN_TEST_DAYS else "FAIL",
        "rule_if_fail": "extend ERA5_END day by day per the precommit's "
                        "data-availability rule and rebuild",
    }

    all_cells = [c.removeprefix("y_pred_") for c in panel.columns
                 if c.startswith("y_pred_")]
    per_day_all = _per_day_mae(panel, all_cells)

    # CW arm is scored on the CW-complete subset (cw_ok all true)
    cw_flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    cw_panel = panel.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
    out["cw_coverage"] = {
        "n_slots_cw_complete": cw_panel.height,
        "share_of_panel": _j(cw_panel.height / panel.height, 4),
        "n_test_days_cw": cw_panel["test_day"].n_unique(),
    }
    per_day_cw = _per_day_mae(cw_panel, [*CW_CELLS, "price_only"])

    daily_cw = {c: _pooled_daily(per_day_cw, c) for c in CW_CELLS}
    out["arm_cw_primary"] = _arm_report(
        per_day_cw, daily_cw,
        pre_gate_tiers=["msm_d2_12z_cw", "msm_gate_composite_cw"],
        post_gate_tier="msm_d1_00z_cw",
        oracle_cell="oracle_cw", stale_cell="era5_cw",
        areas=areas, gate_prefix="gate46_47_48")

    sp_daily = {c: _pooled_daily(per_day_all, c) for c in SP_CELLS
                if c != "price_only"}
    out["arm_sp_robustness"] = _arm_report(
        per_day_all, sp_daily,
        pre_gate_tiers=["msm_d2_12z", "msm_gate_composite"],
        post_gate_tier="msm_d1_00z",
        oracle_cell="oracle", stale_cell="era5",
        areas=areas, gate_prefix="gate50_sp")

    # SP-only diagnostics preserved from phase59 (both sides SP)
    diag_margins = _margins_for(
        out["arm_sp_robustness"]["tiers"][
            out["arm_sp_robustness"]["gate50_sp_pregate_value"]["headline_tier"]][
            "improvement"])
    if "prev1" in sp_daily:
        out["gate50_ingestion_sp"] = {
            **_diff_entry(sp_daily["msm_d1_00z"], sp_daily["prev1"], diag_margins),
            "note": "same nominal initialisation, pinned run vs fixed-lead "
                    "composite; both sides single-point. Reported whatever the sign."}
    if "prev2" in sp_daily:
        h_sp = out["arm_sp_robustness"]["gate50_sp_pregate_value"]["headline_tier"]
        out["gate50_composite_vs_pinned_sp"] = {
            **_diff_entry(sp_daily["prev2"], sp_daily[h_sp], diag_margins),
            "note": "fixed-lead composite minus run-pinned pre-gate headline, "
                    "both single-point."}

    # ---- cross-measurement block: the ONLY place CW and SP meet, flagged
    cross: dict = {"flag": (
        "CROSS-MEASUREMENT contrasts: capacity-weighted vs single-point of the "
        "same run. Reported for the uplift and the measurement-x-lead "
        "interaction only; never mixed into a ladder contrast.")}
    per_day_cross = _per_day_mae(cw_panel, [*CW_JMA_TIERS, *SP_PINNED, "price_only"])
    for cw_t, sp_t in (("msm_d2_12z_cw", "msm_d2_12z"),
                       ("msm_gate_composite_cw", "msm_gate_composite"),
                       ("msm_d1_00z_cw", "msm_d1_00z")):
        cross[f"uplift_{sp_t}"] = _diff_entry(_pooled_daily(per_day_cross, cw_t),
                                              _pooled_daily(per_day_cross, sp_t))
    d_cw = (_pooled_daily(per_day_cross, "msm_d1_00z_cw")
            - _pooled_daily(per_day_cross,
                            out["arm_cw_primary"]["gate46_47_48_pregate_value"][
                                "headline_tier"]))
    h_sp = out["arm_sp_robustness"]["gate50_sp_pregate_value"]["headline_tier"]
    d_sp = (_pooled_daily(per_day_cross, "msm_d1_00z")
            - _pooled_daily(per_day_cross, h_sp))
    inter = d_cw - d_sp
    lo, hi = _block_boot_ci(inter)
    cross["measurement_x_lead_interaction"] = {
        "delta": _j(float(inter.mean()), 5), "ci95": [_j(lo, 5), _j(hi, 5)],
        "dm_l3": _hln_dm(inter, lag=3), "dm_l9": _hln_dm(inter, lag=9)}
    out["cross_measurement"] = cross

    # ---- arms-agreement clause (precommit Gate 47)
    b_cw = out["arm_cw_primary"]["gate46_47_48_gate_timing"]["branch"]
    b_sp = out["arm_sp_robustness"]["gate50_sp_gate_timing"]["branch"]
    out["arms_agreement"] = {
        "cw_branch": b_cw, "sp_branch": b_sp, "agree": b_cw == b_sp,
        "rule": ("if the arms disagree, the CW branch governs the manuscript "
                 "(Gate 35 made CW primary) and the interaction is reported "
                 "prominently"),
    }

    out["metadata"] = {
        "script": Path(__file__).name,
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": SEED, "n_boot": N_BOOT, "boot_block_days": BOOT_BLOCK,
        "weight_vintage": WEIGHT_VINTAGE,
        "versions": {"xgboost": xgb.__version__, "polars": pl.__version__,
                     "numpy": np.__version__},
        "uv_lock_sha256": _sha256(CFG.project_root / "backend/packages/alg/uv.lock")
        if (CFG.project_root / "backend/packages/alg/uv.lock").exists() else
        _sha256(CFG.project_root / "backend/uv.lock"),
        "protocol": {"initial_train_end": "2024-07-18", "val_window_days": 14,
                     "step_days": 1, "panel_start": PANEL_START.isoformat(),
                     "weather_end": WEATHER_END.isoformat(),
                     "era5_end": ERA5_END.isoformat(),
                     "frozen_window": f"identical to phase51/59/74 through "
                                      f"{FROZEN_ERA5_END} (Gate 45 verifies)"},
    }
    return out


# ========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="", help="comma-separated subset")
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
    for arm in ("arm_cw_primary", "arm_sp_robustness"):
        for t, e in out.get(arm, {}).get("tiers", {}).items():
            log.info("  [%s] %-24s imp=%+.4f CI=[%+.4f,%+.4f]",
                     arm.split("_")[1], t, e["improvement"],
                     e["ci95"][0], e["ci95"][1])
    return 0 if out.get("gate45_reproducibility", {}).get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
