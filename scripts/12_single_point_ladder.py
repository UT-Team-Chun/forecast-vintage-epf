r"""Phase 59: the vintage ladder rebuilt on RUN-PINNED tiers.

WHY THIS EXISTS
---------------
Phase 51's headline pre-gate tier was Open-Meteo ``previous_day2``: for each valid
time, the value predicted ~48 h earlier. That is a FIXED-LEAD COMPOSITE stitched
across successive initialisations, not the single run Figure 1 draws and the text
names in the singular. Admissibility still held, but the tier did not correspond to
an information set any bidder actually possessed at the gate, so neither did the
"value of allowing the D-1 00 UTC run" that was derived from it.

This phase re-runs the identical protocol with tiers built from single pinned runs
(phase 58 extraction), on the identical panel, and adds the estimands the pinned
construction makes available.

CELLS
-----
Legacy 5, reproduced BYTE-IDENTICALLY for the Gate 5 reproducibility check:
    price_only, era5, prev2, prev1, histfc, oracle
New run-pinned:
    msm_d2_12z        D-2 12 UTC run, whole delivery day from ONE run,
                      disseminated D-2 23:30 JST (+10 h 30 m pre-gate).
                      The only pre-gate run that can span a full day, because the
                      78 h horizon is archived only for the 00/12 UTC cycles.
    msm_gate_composite  per-slot freshest ADMISSIBLE run (phase 53): D-2 21 UTC for
                      delivery hours 00-21, D-2 12 UTC for 22-23. Every constituent
                      run is disseminated before the gate, so this is the actual
                      information frontier a bidder faced.
    msm_d1_00z        D-1 00 UTC run, whole day from one run, disseminated
                      D-1 11:30 JST -- 1 h 30 m AFTER the gate. The run the gate
                      excludes; the run-pinned gate-timing arm.

GATES (pre-committed in docs/research/2026-07-29_phase58_exactrun_precommit.md)
------------------------------------------------------------------------------
  5  BLOCKING: legacy cells must reproduce phase51's per-slot parquet to <=1e-9 on
     an identical panel. If not, the harness has unaccounted non-determinism and
     the new numbers cannot be trusted either -> abort before reporting anything.
  6  run-pinned pre-gate value vs price_only: bootstrap CI lower bound > 0 AND
     pooled DM p < 0.05 at BOTH L=3 and L=9. Headline = the SMALLER of the two
     pre-gate tiers (conservative, per phase51 practice).
  7  gate-timing delta = imp(msm_d1_00z) - imp(headline pre-gate), reported under
     one of three pre-registered readings depending on where its CI falls.
  8  the stale-proxy multiple is reported only as a product-and-vintage composite,
     with the within-ERA5 vintage effect and the product bound broken out.
  9  ingestion diagnostic imp(msm_d1_00z) - imp(prev1): same initialisation, pinned
     run vs fixed-lead composite. Reported whatever its sign.
 10  dose-response of improvement on mean lead hours, run-pinned tiers only, with
     the gate drawn as a band rather than a line.

CLI (from the repository root):
  uv run python scripts/12_single_point_ladder.py --step build
  uv run python ... --step analyse
  uv run python ... --step build --areas kyushu,tokyo     # partial, for smoke tests
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
log = logging.getLogger("phase59")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
EXACTRUN_DIR = RAW / "jma_gpv_exactrun"
PREGATE_DIR = RAW / "jma_gpv_pregate"
PREV_DIR = RAW / "open_meteo_historical_forecast"
PHASE51_PARQUET = RUN / "phase51_vintage_ladder_per_slot.parquet"
CKPT = RUN / "phase59_checkpoints"
OUT_PARQUET = RUN / "phase59_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase59_exact_ladder.json"

# protocol pinned identically to phase51 -- any change here voids Gate 5
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 800
SEED = 2026
PANEL_START = date(2024, 3, 2)
WEATHER_END = date(2026, 7, 24)   # phase51 freeze
ERA5_END = date(2026, 7, 18)      # phase51 freeze
N_BOOT = 10_000
BOOT_BLOCK = 7
GATE_TOL = 1e-9
MIN_TRAIN_ROWS = 100
ALPHA = 0.05

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
LEGACY_CELLS = ("price_only", "era5", "prev2", "prev1", "histfc", "oracle")
# suffix -> (cache kind, plan key or None, mean lead hours over a delivery day)
PINNED_TIERS: dict[str, dict] = {
    "msm_d2_12z": {"kind": "exactrun", "plan": "msm_d2_12z", "mean_lead_h": 38.5,
                   "pre_gate": True,
                   "definition": "D-2 12 UTC run, FH27-50, one run per whole day; "
                                 "disseminated D-2 23:30 JST (+10h30m pre-gate)"},
    "msm_gate_composite": {"kind": "pregate", "plan": None, "mean_lead_h": 30.25,
                           "pre_gate": True,
                           "definition": "per-slot freshest admissible run: D-2 "
                                         "21 UTC for hours 00-21 (+90 min pre-gate), "
                                         "D-2 12 UTC for hours 22-23"},
    "msm_d1_00z": {"kind": "exactrun", "plan": "msm_d1_00z", "mean_lead_h": 26.5,
                   "pre_gate": False,
                   "definition": "D-1 00 UTC run, FH15-38, one run per whole day; "
                                 "disseminated D-1 11:30 JST (1h30m POST-gate)"},
}
# composite tiers indexed by lead rather than by run; excluded from the dose-response
COMPOSITE_TIERS = ("prev2", "prev1", "histfc")
SW = "shortwave_radiation_wm2"  # the channel phase52 drives its commitment from


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
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ==================================================================== loaders
def _load_prev(area: str, n_prev: int) -> pl.DataFrame:
    """Open-Meteo previous-runs series (fixed-lead composite), as phase51 built it."""
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


def _load_pinned(tier: str, area: str) -> pl.DataFrame:
    """One run-pinned tier for one area, renamed to a per-tier suffix."""
    spec = PINNED_TIERS[tier]
    lat, lon = AREA_COORDS[area]
    if spec["kind"] == "pregate":
        path = PREGATE_DIR / f"jma_gpv_pregate_{area}_{lat:.4f}_{lon:.4f}.parquet"
    else:
        path = (EXACTRUN_DIR / spec["plan"]
                / f"jma_gpv_{spec['plan']}_{area}_{lat:.4f}_{lon:.4f}.parquet")
    if not path.exists():
        msg = f"missing pinned cache for tier {tier}, area {area}: {path}"
        raise FileNotFoundError(msg)
    df = pl.read_parquet(path).sort("datetime_jst")
    keep = ["datetime_jst", *DEFAULT_WEATHER_COLS]
    return df.select(keep).rename(
        {c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS})


# ==================================================================== design
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase51's design verbatim, plus one cell per run-pinned tier.

    The legacy column lists are byte-identical to phase51 so that Gate 5 can
    compare predictions. Lag subscripts count HALF-HOUR SLOTS: 48 slots is 24 h.
    """
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
    for tier in PINNED_TIERS:
        cells[tier] = [*base, *(f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS)]
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


# ==================================================================== per area
def _run_area(area: str) -> None:
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    lat, lon = AREA_COORDS[area]
    # FY2023 must be loaded even though no fold tests it: the 672-slot price lag
    # reaches 14 days back, so without it the panel cannot start at PANEL_START
    # (2024-03-02) and the early training window silently shrinks. Dropping FY2023
    # moves the panel start to 2024-04-15, costs ~44 days of training data, and
    # degrades every weather cell by ~0.019 JPY/kWh while barely moving price-only
    # -- which is exactly how Gate 5 caught this.
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
        src = _load_prev(area, n_prev)
        df = attach_hourly_features(
            df, src, feature_cols=[f"{c}{sfx}" for c in DEFAULT_WEATHER_COLS])
    for tier in PINNED_TIERS:
        src = _load_pinned(tier, area)
        df = attach_hourly_features(
            df, src, feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])

    df, cells = _design(df)
    need = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, ERA5_END))
          .sort("datetime_jst"))
    log.info("%s panel: %d slots %s..%s", area, df.height,
             df["datetime_jst"].min(), df["datetime_jst"].max())

    pvpeak = pv_gw_by_area()[area] / peak_gw_by_area()[area]
    panel_last = df["datetime_jst"].max().date()
    rows: list[dict] = []
    t0, n_folds = time.time(), 0
    for k in range(MAX_FOLDS):
        test_day = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        if test_day > panel_last:
            break
        train = df.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = df.filter(pl.col("datetime_jst").dt.date() == test_day).sort("datetime_jst")
        if train.height < MIN_TRAIN_ROWS or test.height == 0:
            continue
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        # Carry the shortwave column of every tier, not just the predictions.
        # phase52's commitment model builds its generation forecast from the
        # tier's own irradiance, so a per-slot panel without these columns
        # cannot drive the downstream decision model and the sweep would have to
        # be repeated. Same schema convention as phase51's parquet.
        sw_cols = {f"{SW}_{t}": test[f"{SW}_{t}"].to_numpy().astype(float)
                   for t in PINNED_TIERS}
        for legacy in (f"{SW}_pd2", f"{SW}_pd1", f"{SW}_fc", f"{SW}_lag_48", SW):
            if legacy in test.columns:
                sw_cols[legacy] = test[legacy].to_numpy().astype(float)
        for s in range(test.height):
            rows.append({"area": area, "datetime_jst": dts[s],
                         "test_day": test_day.isoformat(), "y_actual": y[s],
                         **{c: float(v[s]) for c, v in sw_cols.items()},
                         **{f"y_pred_{n}": float(preds[n][s]) for n in cells},
                         "pv_peak_ratio": pvpeak})
        n_folds += 1
        if n_folds % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n_folds, test_day,
                     (time.time() - t0) / n_folds)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
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
    """Per-day paired circular block bootstrap 95% CI of the mean."""
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
    """Daily improvement over price_only, averaged across areas (phase51 unit)."""
    return np.asarray(
        (per_day.with_columns(
            (pl.col("mae_price_only") - pl.col(f"mae_{cell}")).alias("imp"))
         .group_by("test_day").agg(pl.col("imp").mean())
         .sort("test_day")["imp"]), dtype=float)


def _entry(d: np.ndarray) -> dict:
    lo, hi = _block_boot_ci(d)
    return {"improvement": _j(float(d.mean()), 4),
            "ci95": [_j(lo, 4), _j(hi, 4)],
            "dm_l3": _hln_dm(d, lag=3), "dm_l9": _hln_dm(d, lag=9),
            "n_days": len(d)}


def _ratio_entry(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    """Ratio of two mean improvements, with a paired block-bootstrap interval.

    The manuscript quotes a multiple. Dividing two rounded point estimates gives
    no interval and hides the fact that the denominator is itself uncertain, so
    the ratio is resampled on the same day blocks as everything else.
    """
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


# Pre-committed equivalence margins
# (docs/research/2026-07-29_phase67_equivalence_precommit.md).
#
# delta_1 is one JEPX price tick. Verified on this panel: all 307,638 realised
# prices lie exactly on a 0.01 JPY/kWh grid (max |100y - round(100y)| = 4.5e-13,
# zero off-grid), the minimum is exactly 0.01, and the smallest gap between
# distinct observed prices is exactly 0.0100. An accuracy improvement below one
# tick cannot alter a bid, a clearing price or a settlement line. The grounding is
# a property of the price column and does not depend on any estimate.
#
# delta_2 is 20% of the pre-gate improvement. It is SECONDARY and labelled so,
# because it is anchored on a result: at 4.6x the tick it returns a p so small it
# reads as rhetoric rather than evidence.
EQUIV_MARGINS: tuple[tuple[str, float, str], ...] = (
    ("tick", 0.01,
     "one JEPX price tick, verified as the exact 0.01 JPY/kWh grid of all realised "
     "prices in the panel. A smallness benchmark grounded in the market's own price "
     "granularity, NOT an operational no-impact threshold: the tested quantity is "
     "the pooled mean absolute error, and mean-level equivalence does not bound "
     "individual slot forecast differences or imply that no bid would change. The "
     "decision-scale consequence is tested separately (phase75)"),
    ("relevance", 0.0458,
     "20% of the pre-gate improvement; a practical-relevance margin, reported as "
     "SECONDARY because it is anchored on an estimate rather than on the market's "
     "price granularity"),
)
EQUIV_ALPHA = 0.05


def _tost(d: np.ndarray, *, seed: int = SEED) -> dict:
    """Two one-sided tests of equivalence on a per-day paired differential.

    Failing to reject a null is not evidence that a difference is absent, so a
    paper that wants to say "the deadline forecloses nothing" has to test
    equivalence rather than report a non-significant difference. Both arms are
    run: the bootstrap reuses the same circular day blocks as every interval in
    this file, and the parametric arm reuses the HLN-corrected Newey-West standard
    error at both bandwidths, so the equivalence statement rests on the same
    resampling and the same variance estimator as the rest of the table.

    Applied to EVERY paired difference, not only the convenient one. Running it
    where it passes and omitting it where it fails would be selective, and the
    product-bound arm is expected to fail.
    """
    n = len(d)
    d_bar = float(d.mean())
    out: dict = {"alpha": EQUIV_ALPHA, "n_days": n,
                 "precommit_doc": "docs/research/2026-07-29_phase67_equivalence_precommit.md",
                 "margins": {}}
    # 90% interval: the TOST-equivalent interval at alpha = 0.05 per side.
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    boot = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    ci90 = [_j(float(np.quantile(boot, 0.05)), 5),
            _j(float(np.quantile(boot, 0.95)), 5)]
    out["ci90"] = ci90
    for name, margin, rationale in EQUIV_MARGINS:
        m: dict = {"delta_margin": margin, "rationale": rationale}
        # bootstrap arm: the share of resampled means outside each margin
        p_lo = float(np.mean(boot <= -margin))
        p_hi = float(np.mean(boot >= margin))
        p_boot = max(p_lo, p_hi)
        m["boot"] = {"p_lower": _j(p_lo, 6), "p_upper": _j(p_hi, 6),
                     "p_tost": _j(p_boot, 6), "equivalent": bool(p_boot < EQUIV_ALPHA)}
        # parametric arm at both Newey-West bandwidths
        for lag in (3, 9):
            dm = _hln_dm(d, lag=lag)
            se = None
            if dm["t_hln"] is not None and dm["t_hln"] != 0:
                se = abs(d_bar / dm["t_hln"])
            if se is None or se <= 0:
                m[f"dm_l{lag}"] = {"se": None, "p_tost": None, "equivalent": None}
                continue
            t_lo = (d_bar + margin) / se     # H0: d <= -margin
            t_hi = (d_bar - margin) / se     # H0: d >= +margin
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
    # Descriptive by-product: the smallest symmetric margin the 90% interval fits
    # inside. Gives a reader the continuous information without pretending it was
    # pre-committed -- keep the label.
    if ci90[0] is not None and ci90[1] is not None:
        out["smallest_margin_supported"] = _j(
            max(abs(ci90[0]), abs(ci90[1])), 5)
        out["smallest_margin_note"] = (
            "smallest symmetric margin whose interval contains the 90% CI; a "
            "DESCRIPTIVE by-product, NOT a pre-committed margin")
    return out


def _diff_entry(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference a - b on the shared daily grid."""
    d = a - b
    lo, hi = _block_boot_ci(d)
    return {"delta": _j(float(d.mean()), 4), "ci95": [_j(lo, 4), _j(hi, 4)],
            "dm_l3": _hln_dm(d, lag=3), "dm_l9": _hln_dm(d, lag=9),
            "n_days": len(d), "equivalence": _tost(d)}


# ==================================================================== Gate 5
def gate5(panel: pl.DataFrame) -> dict:
    """Blocking: legacy cells must reproduce phase51 exactly on an identical panel."""
    if not PHASE51_PARQUET.exists():
        return {"verdict": "SKIP", "reason": f"{PHASE51_PARQUET.name} not found"}
    ref = pl.read_parquet(PHASE51_PARQUET)
    out: dict = {"reference": PHASE51_PARQUET.name,
                 "reference_sha256": _sha256(PHASE51_PARQUET),
                 "n_slots_new": panel.height, "n_slots_reference": ref.height}
    key = ["area", "datetime_jst"]
    joined = panel.select(*key, *[f"y_pred_{c}" for c in LEGACY_CELLS], "y_actual").join(
        ref.select(*key, *[pl.col(f"y_pred_{c}").alias(f"ref_{c}")
                           for c in LEGACY_CELLS],
                   pl.col("y_actual").alias("ref_y")),
        on=key, how="inner")
    out["n_slots_matched"] = joined.height
    out["panel_identical"] = (panel.height == ref.height == joined.height)
    worst = {}
    for c in LEGACY_CELLS:
        diff = float((joined[f"y_pred_{c}"] - joined[f"ref_{c}"]).abs().max() or 0.0)
        worst[c] = diff
    worst["y_actual"] = float((joined["y_actual"] - joined["ref_y"]).abs().max() or 0.0)
    out["max_abs_diff"] = {k: _j(v, 12) for k, v in worst.items()}
    out["tolerance"] = GATE_TOL
    out["verdict"] = ("PASS" if out["panel_identical"]
                      and all(v <= GATE_TOL for v in worst.values()) else "FAIL")
    out["consequence_if_fail"] = (
        "Abort. Unaccounted non-determinism in the harness would make the new "
        "run-pinned numbers untrustworthy as well.")
    return out


# ==================================================================== analysis
def analyse() -> dict:
    files = sorted(CKPT.glob("*.parquet"))
    if not files:
        msg = f"no checkpoints under {CKPT}; run --step build first"
        raise FileNotFoundError(msg)
    panel = pl.concat([pl.read_parquet(f) for f in files]).sort("area", "datetime_jst")
    panel.write_parquet(OUT_PARQUET)
    cells = [c.removeprefix("y_pred_") for c in panel.columns
             if c.startswith("y_pred_")]
    areas = sorted(panel["area"].unique().to_list())
    log.info("panel: %d slots / %d areas / %d cells", panel.height, len(areas),
             len(cells))

    out: dict = {
        "phase": 59,
        "title": "Vintage ladder on run-pinned tiers",
        "precommit_doc": "docs/research/2026-07-29_phase58_exactrun_precommit.md",
        "panel": {"n_slots": panel.height, "n_areas": len(areas),
                  "n_test_days": panel["test_day"].n_unique(),
                  "first_test_day": panel["test_day"].min(),
                  "last_test_day": panel["test_day"].max(),
                  "slots_per_area": int(panel.height / len(areas))},
        "tier_definitions": {k: v["definition"] for k, v in PINNED_TIERS.items()},
    }

    # ---- Gate 5 first: nothing else is reportable if it fails
    out["gate5_reproducibility"] = gate5(panel)
    log.info("GATE 5: %s (panel identical=%s, worst legacy diff=%s)",
             out["gate5_reproducibility"]["verdict"],
             out["gate5_reproducibility"].get("panel_identical"),
             max((out["gate5_reproducibility"].get("max_abs_diff") or {}).values(),
                 default=None))
    if out["gate5_reproducibility"]["verdict"] == "FAIL":
        out["aborted"] = True
        return out

    per_day = _per_day_mae(panel, cells)
    daily = {c: _pooled_daily(per_day, c) for c in cells if c != "price_only"}
    out["tiers"] = {c: _entry(d) for c, d in daily.items()}

    # per-area improvement and win counts
    per_area: dict = {}
    for c in daily:
        rec = {}
        for a in areas:
            sub = per_day.filter(pl.col("area") == a)
            d = np.asarray(sub["mae_price_only"] - sub[f"mae_{c}"], dtype=float)
            rec[a] = {"improvement": _j(float(d.mean()), 4),
                      "p_l3": _hln_dm(d, lag=3)["p_hln"]}
        per_area[c] = rec
        out["tiers"][c]["areas_improved"] = sum(
            1 for v in rec.values()
            if v["improvement"] is not None and v["improvement"] > 0)
        out["tiers"][c]["areas_sig_l3"] = sum(
            1 for v in rec.values()
            if v["p_l3"] is not None and v["p_l3"] < ALPHA)
    out["per_area"] = per_area

    # ---- Gate 6: run-pinned pre-gate value, headline = the smaller
    pre_gate = [t for t, s in PINNED_TIERS.items() if s["pre_gate"] and t in daily]
    g6: dict = {"candidates": {}}
    for t in pre_gate:
        e = out["tiers"][t]
        g6["candidates"][t] = {
            "improvement": e["improvement"], "ci95": e["ci95"],
            "dm_p_l3": e["dm_l3"]["p_hln"], "dm_p_l9": e["dm_l9"]["p_hln"],
            # `x or default` cannot be used here: a p-value of exactly 0.0 and a
            # CI bound of exactly 0.0 are both FALSY, so that idiom silently
            # converts the strongest possible evidence into a gate failure. It did
            # exactly that on the first run of this gate.
            "passes": bool(
                e["ci95"][0] is not None and e["ci95"][0] > 0
                and e["dm_l3"]["p_hln"] is not None and e["dm_l3"]["p_hln"] < ALPHA
                and e["dm_l9"]["p_hln"] is not None and e["dm_l9"]["p_hln"] < ALPHA),
        }
    if g6["candidates"]:
        headline = min(g6["candidates"], key=lambda t: g6["candidates"][t]["improvement"])
        g6["headline_tier"] = headline
        g6["headline_rule"] = ("the smaller of the two run-pinned pre-gate tiers, "
                               "fixed before results (conservative)")
        g6["verdict"] = "PASS" if all(v["passes"] for v in g6["candidates"].values()) else "FAIL"
    out["gate6_pregate_value"] = g6

    # ---- Gate 7: what the gate excludes (three pre-registered readings)
    if g6.get("headline_tier") and "msm_d1_00z" in daily:
        h = g6["headline_tier"]
        d = _diff_entry(daily["msm_d1_00z"], daily[h])
        lo, hi = d["ci95"]
        reading = ("7a: the excluded run adds detectable point-forecast value"
                   if lo is not None and lo > 0 else
                   "7c: the two adjacent rungs do not order in this sample; no "
                   "negative freshness value is claimed"
                   if hi is not None and hi < 0 else
                   "7b: the value of the run the gate excludes is not "
                   "distinguishable from zero; reported as an upper bound")
        out["gate7_gate_timing"] = {
            "post_gate_tier": "msm_d1_00z", "pre_gate_tier": h,
            **d, "reading": reading,
            "upper_bound_if_7b": hi,
        }
        log.info("GATE 7: delta=%s CI=%s -> %s", d["delta"], d["ci95"], reading)

        # Per-area gate timing. The supplement's per-area ladder table used to
        # carry a "post vs pre" column taken from a run that left no artefact; it
        # was dropped when the table became provenance-generated, and this is the
        # column that lets it come back. Placed here because the headline tier is
        # only chosen at Gate 6 -- deriving it earlier would hardcode a choice the
        # gate is supposed to make.
        gt: dict = {}
        for a in areas:
            sub = per_day.filter(pl.col("area") == a)
            dd = np.asarray(sub[f"mae_{h}"] - sub["mae_msm_d1_00z"], dtype=float)
            lo_a, hi_a = _block_boot_ci(dd)
            gt[a] = {"delta": _j(float(dd.mean()), 4),
                     "ci95": [_j(lo_a, 4), _j(hi_a, 4)],
                     "p_l3": _hln_dm(dd, lag=3)["p_hln"],
                     "p_l9": _hln_dm(dd, lag=9)["p_hln"],
                     "equivalence": _tost(dd)}
        out["per_area_gate_timing"] = {
            "definition": f"MAE({h}) - MAE(msm_d1_00z) per area; positive means "
                          "the post-gate run is more accurate",
            "areas": gt,
            "n_areas_equivalent_at_tick": sum(
                1 for v in gt.values()
                if v["equivalence"]["margins"]["tick"]["verdict"] == "EQUIVALENT"),
            "n_areas_sig_l3": sum(1 for v in gt.values()
                                  if v["p_l3"] is not None and v["p_l3"] < ALPHA),
        }

    # ---- Gate 8: decompose the product/vintage confound
    g8: dict = {"label_rule": (
        "The stale-proxy multiple is a COMPOSITE contrast of product, vintage and "
        "availability. It is never to be called a vintage effect on its own.")}
    if "oracle" in daily and "era5" in daily:
        g8["within_era5_vintage_effect"] = _diff_entry(daily["oracle"], daily["era5"])
        g8["within_era5_note"] = ("product held fixed (ERA5 both sides): realised "
                                  "vs day-lagged. This is a pure vintage effect.")
    if g6.get("headline_tier") and "oracle" in daily:
        g8["product_bound"] = _diff_entry(daily[g6["headline_tier"]], daily["oracle"])
        g8["product_bound_note"] = ("vintage roughly held fixed at the realised "
                                    "frontier: run-pinned NWP vs realised ERA5. "
                                    "Bounds the product contribution.")
    if "era5" in daily and g6.get("headline_tier"):
        imp_e = out["tiers"]["era5"]["improvement"]
        imp_h = out["tiers"][g6["headline_tier"]]["improvement"]
        g8["composite_multiple"] = _j(imp_h / imp_e, 3) if imp_e else None
    out["gate8_product_confound"] = g8

    # ---- Gate 9: ingestion diagnostic (same init, pinned vs composite)
    if "msm_d1_00z" in daily and "prev1" in daily:
        out["gate9_ingestion"] = {
            **_diff_entry(daily["msm_d1_00z"], daily["prev1"]),
            "note": ("same nominal initialisation, taken from the per-run GRIB2 "
                     "archive versus the fixed-lead previous-runs composite. "
                     "Reported whatever the sign."),
        }

    # ---- Gate 10: dose-response on mean lead hours, run-pinned tiers only
    pts = [(PINNED_TIERS[t]["mean_lead_h"], out["tiers"][t]["improvement"], t)
           for t in PINNED_TIERS if t in daily]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        xc = xs - xs.mean()
        slope = float(ys @ xc / (xc @ xc)) if xc @ xc > 0 else None
        out["gate10_dose_response"] = {
            "points": [{"tier": t, "mean_lead_h": x, "improvement": y}
                       for x, y, t in pts],
            "slope_per_lead_hour": _j(slope, 5),
            "gate_band_lead_hours": [26.5, 30.25],
            "excluded_composites": list(COMPOSITE_TIERS),
            "note": ("Only run-pinned tiers enter the regression, because a "
                     "fixed-lead composite has no single lead. Composites are "
                     "plotted as open markers and annotated as such."),
        }

    # ---- composite vs pinned, the reviewer's central question
    if "prev2" in daily and g6.get("headline_tier"):
        out["composite_vs_pinned"] = {
            **_diff_entry(daily["prev2"], daily[g6["headline_tier"]]),
            "note": ("previous_day2 fixed-lead composite minus the run-pinned "
                     "pre-gate headline. Quantifies what the old construction "
                     "was measuring relative to a real information set."),
        }

    # ---- headline vs the stale proxy: the difference and the ratio, both with
    # bootstrap intervals, so the manuscript's multiple is provenance-backed
    # rather than divided by hand from two rounded numbers.
    if g6.get("headline_tier") and "era5" in daily:
        h = g6["headline_tier"]
        out["headline_vs_stale"] = {
            "difference": _diff_entry(daily[h], daily["era5"]),
            "ratio": _ratio_entry(daily[h], daily["era5"]),
            "note": ("The ratio is a COMPOSITE contrast of product, vintage and "
                     "availability (Gate 8), never a vintage effect alone."),
        }
        out["headline_retention_of_post_gate_pct"] = (
            _j(100.0 * out["tiers"][h]["improvement"]
               / out["tiers"]["msm_d1_00z"]["improvement"], 1)
            if "msm_d1_00z" in out["tiers"] else None)

    out["metadata"] = {
        "script": ""
                  "12_single_point_ladder.py",
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": {"initial_train_end": "2024-07-18", "val_window_days": 14,
                     "step_days": 1, "seed": SEED,
                     "panel_start": PANEL_START.isoformat(),
                     "era5_end": ERA5_END.isoformat(),
                     "identical_to": "phase51 (Gate 5 verifies this)"},
    }
    return out


# ========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=("build", "analyse"), required=True)
    ap.add_argument("--areas", default="", help="comma-separated subset")
    ap.add_argument("--tiers", default="",
                    help="comma-separated run-pinned tiers to include; defaults to "
                         "all whose caches exist. Use to validate Gate 5 before the "
                         "full extraction has landed.")
    args = ap.parse_args()

    if args.tiers:
        keep = {t.strip() for t in args.tiers.split(",") if t.strip()}
        unknown = keep - set(PINNED_TIERS)
        if unknown:
            log.error("unknown tiers: %s", sorted(unknown))
            return 2
        for t in list(PINNED_TIERS):
            if t not in keep:
                del PINNED_TIERS[t]
        log.info("restricted to run-pinned tiers %s", sorted(PINNED_TIERS))

    if args.step == "build":
        wanted = ([a.strip() for a in args.areas.split(",") if a.strip()]
                  or list(AREA_COORDS))
        for a in wanted:
            _run_area(a)
        return 0

    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    for t, e in out.get("tiers", {}).items():
        log.info("  %-20s imp=%+.4f CI=[%+.4f,%+.4f] p(L3)=%.2e areas+=%d sig=%d",
                 t, e["improvement"], e["ci95"][0], e["ci95"][1],
                 e["dm_l3"]["p_hln"] or 0.0, e["areas_improved"], e["areas_sig_l3"])
    return 0 if out.get("gate5_reproducibility", {}).get("verdict") != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
