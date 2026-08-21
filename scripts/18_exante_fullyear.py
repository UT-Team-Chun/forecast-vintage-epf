r"""Phase 49: data-window refresh + full-year previous-day-run (true-forecast) replication.

LAST pre-submission experiment. Dissolves two manuscript caveats:
  (a) "the genuinely ex-ante subset is 5.5 months and seasonal"  (sec 5.2 sentence,
      tab:claims row 2 caveat) -- the run-pinned day-ahead-class coverage was only
      the 2025-08+ Open-Meteo Single-Runs subset (n=40 folds).
  (b) "the full-period operational headline rests on a reconstruction" (ae_07
      limitations caveat 4, % [PHASE49-PENDING]) -- the histfc tier blends model
      runs without a pinned issuance.

KEY PROBE (gate): the Open-Meteo Historical-Forecast API serves ``*_previous_day1``
variables for the JMA-MSM model over Japanese points (same mechanism phase45 used
for Germany/ICON). ``previous_day1`` = for each delivery hour, the value from the
JMA-MSM run initialised ~24-27 h before delivery = a TRUE previous-day model run
(day-ahead class, run-pinned to D-1), covering ~2.4 years (2024-02 -> now), vs the
5.5-month single-runs subset.

Issuance note (honest, load-bearing; same class as phase45 DE): JEPX gate closure
is D-1 10:00 JST. Delivery slots 00:00-10:00 JST use a run initialised strictly
before the gate; slots 10:30-23:30 may use a D-1 run initialised up to ~13.5 h
after the gate (always a previous-day run, never a same-day nowcast). We therefore
call this tier "previous-day-run (true forecast)", NOT "exact-issuance"; a strictly
gate-admissible morning-slot subset is reported separately. The exact-issuance
JMA-GPV tier on the committed FY2024-25 window remains the strict reference.

SCOPE DISCIPLINE: this is a TARGETED extension, not a rebase. The committed
FY2024-25 study-window results (all 142 verified macros) stay untouched; new
numbers are additional corroboration on extended windows, with ONE data-freeze
date recorded in the JSON.

Steps (each gated on the previous):
  probe      : empirical coverage probe of jma_msm/jma_seamless previous_day1
               (earliest + latest full day per variable, 2 points) -> gate.
  refresh    : data-window refresh with freeze record: JEPX spot_2026.csv,
               ERA5 2026 (end_date_override), jepx_imbalance months after 2025-10.
  headtohead : phase35/41 all-hours protocol, 9 areas, 4 same-sample cells
               (price_only / +ERA5-lag48 proxy / +histfc reconstruction /
               +previous_day1 true forecast) over the maximal prev1 window.
  decision   : phase31c real-yen floor-decision protocol on the extended window
               (SEPARATE number; committed 177,427 cell untouched) + old-window
               bridge + phase26.1-panel variant with extended imbalance.
  aggregate  : pooled + per-area + DM (HLN, NW L=3) + cross-area Spearman +
               reconstruction-vs-true gap + admissible-morning subset -> JSON.

CLI:
  cd <repository root>
  uv run --extra research python scripts/18_exante_fullyear.py \
      --step all [--areas kyushu,tokyo] [--fresh] [--skip-probe]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import subprocess
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import peak_gw_by_area, pv_gw_by_area
from alg.core.epf.data.jepx_loader import fetch_jepx_spot_year, load_jepx_spot, parse_jepx_spot_csv
from alg.core.epf.data.open_meteo_historical_forecast_loader import (
    load_open_meteo_historical_forecast_archive,
)
from alg.core.epf.data.open_meteo_loader import fetch_open_meteo_year, load_open_meteo_weather
from alg.core.epf.data.registry import REGISTRY_FILENAME, DatasetRegistry
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import (
    DEFAULT_WEATHER_COLS,
    attach_hourly_features,
    attach_hourly_weather,
)
from alg.models.epf.area import AreaCode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase49")
JST = ZoneInfo("Asia/Tokyo")

# ------------------------------------------------------------------ pinned params
OPEN_METEO_HISTFC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
IMB_BASE = "https://www.imbalanceprices-cs.jp"
PREV1_MODEL = "jma_msm"  # same model family as the histfc reconstruction tier
API_VARS = ("temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m")
API_TO_CANON = {"temperature_2m": "temperature_c", "cloud_cover": "cloud_cover_pct",
                "shortwave_radiation": "shortwave_radiation_wm2",
                "wind_speed_10m": "wind_speed_kmh"}
UA = {"User-Agent": "forecast-vintage-epf/1.0 (+contact: ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"}

AREA_COORDS: dict[AreaCode, tuple[float, float, str]] = {
    AreaCode.HOKKAIDO: (43.07, 141.35, "Sapporo"), AreaCode.TOHOKU: (38.27, 140.87, "Sendai"),
    AreaCode.TOKYO: (35.68, 139.76, "Tokyo"), AreaCode.CHUBU: (35.18, 136.91, "Nagoya"),
    AreaCode.HOKURIKU: (36.56, 136.66, "Kanazawa"), AreaCode.KANSAI: (34.69, 135.50, "Osaka"),
    AreaCode.CHUGOKU: (34.40, 132.46, "Hiroshima"), AreaCode.SHIKOKU: (34.34, 134.05, "Takamatsu"),
    AreaCode.KYUSHU: (33.59, 130.40, "Fukuoka"),
}
DECISION_AREAS_8 = ["hokkaido", "tohoku", "tokyo", "hokuriku", "kansai",
                    "chugoku", "shikoku", "kyushu"]  # committed decision suite (no Chubu)
IMB_AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai",
             "chugoku", "shikoku", "kyushu"]  # T-header order of the disclosure CSV

# walk-forward protocol constants (phase35/41, unchanged)
INITIAL_TRAIN_END = datetime(2024, 9, 1, tzinfo=JST)
VAL_WINDOW, TEST_WINDOW, STEP = timedelta(days=14), timedelta(days=1), timedelta(days=3)
MIN_TEST_DAY = date(2024, 11, 15)
MAX_FOLDS = 400  # safety bound; actual folds limited by panel coverage
SEED = 2026

PANEL_START = date(2024, 4, 1)     # FY2024 JEPX file start (prev1 archive starts earlier)
ERA5_DELAY_DAYS = 8
HISTFC_DELAY_DAYS = 2
FLOOR = 0.5
SLOTS_PER_YEAR = 48 * 365          # phase26.2 / phase31c scaling convention
OLD_WINDOW_LAST_DAY = date(2025, 11, 6)  # last test day of the committed study window
ADMISSIBLE_LAST_SLOT_HOUR = 10     # delivery slots 00:00-10:00 JST strictly pre-gate

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase49_checkpoints"
OUT_JSON = RUN / "phase49_exante_fullyear.json"
PER_SLOT_PARQUET = RUN / "phase49_exante_fullyear_per_slot.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(client: httpx.Client, url: str, params: dict | None = None,
              retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            wait = 2.0 * (attempt + 1)
            log.warning("retry %d for %s (%s); sleeping %.1fs", attempt + 1, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_summary() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"phase": "49",
            "title": ("data-window refresh + full-year previous-day-run (true-forecast) "
                      "replication"),
            "data_freeze_date_utc": None}


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


# ================================================================== STEP 1: probe
def _day_covered(client: httpx.Client, model: str, hourly: str, d: date,
                 lat: float, lon: float) -> bool:
    js = _get_json(client, OPEN_METEO_HISTFC, {
        "latitude": lat, "longitude": lon, "start_date": d.isoformat(),
        "end_date": d.isoformat(), "hourly": hourly, "models": model,
        "timezone": "Asia/Tokyo"})
    h = js.get("hourly", {})
    fracs = [sum(x is not None for x in v) / max(len(v), 1)
             for k, v in h.items() if k != "time"]
    time.sleep(0.35)
    return bool(fracs) and min(fracs) >= 0.99


def _bisect_first(client: httpx.Client, model: str, hourly: str,
                  lo: date, hi: date, lat: float, lon: float) -> str | None:
    if not _day_covered(client, model, hourly, hi, lat, lon):
        return None
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if _day_covered(client, model, hourly, mid, lat, lon):
            hi = mid
        else:
            lo = mid + timedelta(days=1)
    return lo.isoformat()


def step_probe(summary: dict) -> bool:
    """Gate: JMA previous_day1 coverage over Japan. Records outcome regardless."""
    kyushu = AREA_COORDS[AreaCode.KYUSHU][:2]
    tokyo = AREA_COORDS[AreaCode.TOKYO][:2]
    today = datetime.now(UTC).date()
    probe: dict = {
        "endpoint": OPEN_METEO_HISTFC,
        "models_checked": ["jma_msm", "jma_seamless"],
        "model_selected": PREV1_MODEL,
        "variables_api": [f"{v}_previous_day1" for v in API_VARS],
        "points": {"kyushu": list(kyushu), "tokyo": list(tokyo)},
        "per_variable_first_full_day": {},
        "latest_full_day": None,
    }
    with httpx.Client(timeout=30.0, headers=UA) as c:
        for model in ("jma_msm", "jma_seamless"):
            per_var: dict = {}
            for v in API_VARS:
                per_var[v] = _bisect_first(
                    c, model, f"{v}_previous_day1", date(2024, 1, 1), date(2024, 6, 1),
                    *kyushu)
            probe["per_variable_first_full_day"][model] = per_var
        all4 = ",".join(f"{v}_previous_day1" for v in API_VARS)
        probe["all4_first_full_day_kyushu"] = _bisect_first(
            c, PREV1_MODEL, all4, date(2024, 1, 1), date(2024, 6, 1), *kyushu)
        probe["all4_first_full_day_tokyo"] = _bisect_first(
            c, PREV1_MODEL, all4, date(2024, 1, 1), date(2024, 6, 1), *tokyo)
        latest = None
        for back in range(1, 8):
            d = today - timedelta(days=back)
            if _day_covered(c, PREV1_MODEL, all4, d, *kyushu):
                latest = d.isoformat()
                break
        probe["latest_full_day"] = latest
    firsts = [probe["all4_first_full_day_kyushu"], probe["all4_first_full_day_tokyo"]]
    ok = all(firsts) and latest is not None
    cov_days = ((date.fromisoformat(latest) - date.fromisoformat(max(firsts))).days
                if ok else 0)
    probe["operational_tier_definition"] = (
        "previous_day1 variables on the Historical-Forecast API = value for each "
        "delivery hour taken from the JMA-MSM run initialised ~24-27 h before "
        "delivery (a TRUE previous-day model run, day-ahead class, run-pinned to "
        "D-1). JEPX gate closure D-1 10:00 JST: delivery slots 00:00-10:00 JST use "
        "a run initialised strictly before the gate; slots 10:30-23:30 may use a "
        "D-1 run initialised up to ~13.5 h after the gate (reconstruction-ambiguity "
        "class of phase45 DE; never a same-day run). NOT called 'exact-issuance'.")
    probe["gate"] = {
        "criterion": ">= 540 days (1.5 y, full seasonal cycle) of previous_day1 "
                     "coverage over Japanese points, all 4 paper variables",
        "coverage_days": cov_days,
        "decision": "PASS" if (ok and cov_days >= 540) else "STOP -> fallback to "
                    "extending the single-runs ex-ante subset",
    }
    summary["probe"] = probe
    _save_summary(summary)
    log.info("PROBE gate: %s (coverage %d days, first %s, latest %s)",
             probe["gate"]["decision"], cov_days, firsts, latest)
    return ok and cov_days >= 540


# ================================================================ STEP 2: refresh
def _parse_imbalance_month_csv(text: str) -> list[dict]:
    """Phase31 parser: per 30-min slot, shortage block = fields 23..40 (9 areas)."""
    rows: list[dict] = []
    for rec in csv.reader(io.StringIO(text)):
        if not rec or rec[0] != "D":
            continue
        dt = datetime.strptime(f"{rec[1]} {rec[3]}", "%Y%m%d %H:%M").replace(tzinfo=JST)
        shortage = rec[23:41:2]
        for i, area in enumerate(IMB_AREAS):
            v = shortage[i].strip()
            rows.append({"area": area, "datetime_jst": dt,
                         "shortage_imb_yen_kwh": float(v) if v not in ("", "-") else None})
    return rows


def step_refresh(summary: dict) -> None:
    """Refresh JEPX spot 2026, ERA5 2026 (9 points), imbalance months > 2025-10."""
    today = datetime.now(UTC).date()
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    freeze: dict = {"data_freeze_date_utc": today.isoformat(), "files": {}}

    # -- JEPX spot FY2026 (public file grows in place; force re-download) --
    spot_path = fetch_jepx_spot_year(2026, cache_dir=RAW, registry=reg, force=True)
    spot_df = parse_jepx_spot_csv(spot_path)
    freeze["files"]["jepx_spot_2026"] = {
        "path": "data/raw/jepx/spot_2026.csv",
        "source_url": "https://www.jepx.jp/market/excel/spot_2026.csv",
        "sha256": _sha256(spot_path),
        "rows": int(spot_df.height),
        "last_slot_jst": spot_df["datetime_jst"].max().isoformat(),
    }
    log.info("spot_2026 refreshed: last slot %s", spot_df["datetime_jst"].max())

    # -- ERA5 calendar-2026 for the 9 area points (end_date_override, force) --
    era5_end = (today - timedelta(days=ERA5_DELAY_DAYS)).isoformat()
    era5_files = {}
    for area, (lat, lon, _) in AREA_COORDS.items():
        p = fetch_open_meteo_year(2026, cache_dir=RAW, latitude=lat, longitude=lon,
                                  registry=reg, force=True, end_date_override=era5_end)
        era5_files[area.value] = {"path": str(p.relative_to(RAW.parent.parent)),
                                  "sha256": _sha256(p), "end_date": era5_end}
        time.sleep(0.5)
    freeze["files"]["era5_2026_9points"] = {
        "endpoint": "https://archive-api.open-meteo.com/v1/archive",
        "end_date_override": era5_end, "per_area": era5_files}

    # -- imbalance disclosure months after the committed 2024-09..2025-10 window --
    imb_dir = RAW / "jepx_imbalance"
    imb_dir.mkdir(parents=True, exist_ok=True)
    new_files = {}
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=UA) as c:
        listing = _get_json(c, f"{IMB_BASE}/imbalance-price-list/priceList")["imbalance_list"]
        paths = sorted({x["path"] for x in listing
                        if x["path"].split("/")[2][:6] >= "202409"})
        for p in paths:
            cache_f = imb_dir / p.replace("/", "_")
            is_new = not cache_f.exists()
            if is_new:
                cache_f.write_bytes(c.get(f"{IMB_BASE}/public/price/{p}").content)
                time.sleep(0.5)
            if p.split("/")[2][:6] > "202510":
                new_files[p] = {"cached_as": cache_f.name, "sha256": _sha256(cache_f),
                                "newly_downloaded": is_new}
    freeze["files"]["jepx_imbalance_extension"] = {
        "source": f"{IMB_BASE}/public/price/<path> (paths from /imbalance-price-list/priceList)",
        "committed_window": "2024-09..2025-10 (phase31; untouched)",
        "new_months": new_files,
        "n_new_months": len(new_files),
    }
    log.info("imbalance extension: %d months beyond 2025-10", len(new_files))

    summary["data_freeze_date_utc"] = today.isoformat()
    summary["freeze"] = freeze
    _save_summary(summary)


# ============================================================ STEP 3: head-to-head
def _fetch_prev1_area_year(client: httpx.Client, area: str, lat: float, lon: float,
                           year: int, end_day: date) -> pl.DataFrame | None:
    """One calendar year of previous_day1 JMA-MSM forecast for one point; cached."""
    start = max(date(year, 1, 1), PANEL_START - timedelta(days=3))
    end = min(date(year, 12, 31), end_day)
    if start > end:
        return None
    cache = (RAW / "open_meteo_historical_forecast" /
             f"open_meteo_prev1_{PREV1_MODEL}_{area}_{lat:.4f}_{lon:.4f}_{year}.parquet")
    if cache.exists():
        df = pl.read_parquet(cache)
        if year < end.year or df["datetime_jst"].max().date() >= end - timedelta(days=1):
            return df
    js = _get_json(client, OPEN_METEO_HISTFC, {
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(f"{v}_previous_day1" for v in API_VARS),
        "models": PREV1_MODEL, "timezone": "Asia/Tokyo"})
    h = js["hourly"]
    df = pl.DataFrame(
        {
            "datetime_jst": [datetime.fromisoformat(t).replace(tzinfo=JST)
                             for t in h["time"]],
            **{f"{API_TO_CANON[v]}_pd1": [float(x) if x is not None else None
                                          for x in h[f"{v}_previous_day1"]]
               for v in API_VARS},
        },
        schema={"datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
                **{f"{API_TO_CANON[v]}_pd1": pl.Float64 for v in API_VARS}},
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    log.info("prev1 %s %d: %d hourly rows fetched", area, year, df.height)
    time.sleep(1.0)
    return df


def _load_prev1(area: str, lat: float, lon: float, end_day: date) -> pl.DataFrame:
    with httpx.Client(timeout=120.0, headers=UA) as c:
        frames = [_fetch_prev1_area_year(c, area, lat, lon, y, end_day)
                  for y in range(PANEL_START.year, end_day.year + 1)]
    return (pl.concat([f for f in frames if f is not None])
            .unique("datetime_jst").sort("datetime_jst"))


def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase35 design: price lags 48/96/336/672, ERA5 lag-48, calendar; 4 cells."""
    out = df
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
        "histfc": [*base, *(f"{c}_fc" for c in DEFAULT_WEATHER_COLS)],
        "prev1": [*base, *(f"{c}_pd1" for c in DEFAULT_WEATHER_COLS)],
    }
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


def _run_area(area: AreaCode, lat: float, lon: float, label: str) -> None:
    ckpt = CKPT / f"{area.value}.parquet"
    meta_f = CKPT / f"{area.value}_meta.json"
    if ckpt.exists() and meta_f.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    log.info("=== %s (%s) ===", area.value, label)
    today = datetime.now(UTC).date()
    histfc_end = today - timedelta(days=HISTFC_DELAY_DAYS)
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    price = load_jepx_spot(years=[2024, 2025, 2026], cache_dir=RAW, areas=[area],
                           registry=reg).select("datetime_jst", "area", "price_yen_kwh")
    era5 = load_open_meteo_weather(years=[2024, 2025, 2026], cache_dir=RAW,
                                   latitude=lat, longitude=lon, registry=reg)
    span = [PANEL_START + timedelta(days=i)
            for i in range((histfc_end - PANEL_START).days + 1)]
    histfc = load_open_meteo_historical_forecast_archive(
        span, cache_dir=RAW, latitude=lat, longitude=lon, registry=reg,
    ).select("datetime_jst", *DEFAULT_WEATHER_COLS).rename(
        {c: f"{c}_fc" for c in DEFAULT_WEATHER_COLS})
    prev1 = _load_prev1(area.value, lat, lon, histfc_end)
    df = attach_hourly_weather(price, era5)
    df = attach_hourly_features(df, histfc,
                                feature_cols=[f"{c}_fc" for c in DEFAULT_WEATHER_COLS])
    df = attach_hourly_features(df, prev1,
                                feature_cols=[f"{c}_pd1" for c in DEFAULT_WEATHER_COLS])
    df, cells = _design(df)
    need = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    df = df.drop_nulls(subset=need).sort("datetime_jst")
    log.info("%s panel: %d slots %s..%s", area.value, df.height,
             df["datetime_jst"].min(), df["datetime_jst"].max())

    # reconstruction-vs-true feature tracking: midday shortwave corr (histfc vs prev1)
    midday = df.filter(pl.col("datetime_jst").dt.hour().is_between(10, 14))
    corr = float(np.corrcoef(midday["shortwave_radiation_wm2_fc"].to_numpy(),
                             midday["shortwave_radiation_wm2_pd1"].to_numpy())[0, 1])

    pvpeak = pv_gw_by_area()[area.value] / peak_gw_by_area()[area.value]
    rows: list[dict] = []
    n_folds_run = 0
    t0 = time.time()
    for k in range(MAX_FOLDS):
        val_end = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        test_end = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW + TEST_WINDOW).date()
        if val_end < MIN_TEST_DAY:
            continue
        train = df.filter(pl.col("datetime_jst").dt.date() < val_end)
        test = df.filter((pl.col("datetime_jst").dt.date() >= val_end)
                         & (pl.col("datetime_jst").dt.date() < test_end)
                         ).sort("datetime_jst")
        if train.height < 100 or test.height == 0:
            continue
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        for s in range(test.height):
            rows.append({"area": area.value, "datetime_jst": dts[s],
                         "test_day": val_end.isoformat(), "y_actual": y[s],
                         **{f"y_pred_{name}": float(preds[name][s]) for name in cells},
                         "pv_peak_ratio": pvpeak})
        n_folds_run += 1
        if n_folds_run % 25 == 0:
            log.info("%s fold %d (test %s) | %.1f s/fold", area.value, n_folds_run,
                     val_end, (time.time() - t0) / n_folds_run)
    out = pl.DataFrame(rows)
    CKPT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(ckpt)
    meta_f.write_text(json.dumps({
        "area": area.value, "n_folds": n_folds_run, "n_slots": int(out.height),
        "panel_first_slot": df["datetime_jst"].min().isoformat(),
        "panel_last_slot": df["datetime_jst"].max().isoformat(),
        "first_test_day": out["test_day"].min(), "last_test_day": out["test_day"].max(),
        "midday_shortwave_corr_histfc_vs_prev1": round(corr, 4),
        "pv_peak_ratio": round(pvpeak, 4),
        "runtime_s": round(time.time() - t0, 1)}, indent=2))
    log.info("%s done: %d folds, %d slots, %.0f s", area.value, n_folds_run,
             out.height, time.time() - t0)


def step_headtohead(areas: list[str] | None, fresh: bool) -> None:
    if fresh and CKPT.exists():
        for f in CKPT.glob("*"):
            f.unlink()
    wanted = ([AreaCode(a) for a in areas] if areas else list(AREA_COORDS))
    for a in wanted:
        lat, lon, label = AREA_COORDS[a]
        _run_area(a, lat, lon, label)


# ==================================================================== statistics
def _hln_newey_west_dm(d: np.ndarray, *, h: int = 1, lag: int = 3) -> dict:
    """DM on per-fold loss diffs with Newey-West (Bartlett) HAC + HLN correction
    (phase23.5 convention, identical to phase45).
    """
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


def _per_fold_mae(df: pl.DataFrame) -> pl.DataFrame:
    return (df.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in ("price_only", "era5", "histfc", "prev1")])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}")
               for c in ("price_only", "era5", "histfc", "prev1")])
        .sort("area", "test_day"))


def _dm_pair(per_fold: pl.DataFrame, a: str, b: str) -> dict:
    """DM of cell a vs cell b (negative mean_diff = a better), per area + pooled."""
    out: dict = {"per_area": {}}
    for area in sorted(per_fold["area"].unique().to_list()):
        sub = per_fold.filter(pl.col("area") == area).sort("test_day")
        d = (sub[f"mae_{a}"] - sub[f"mae_{b}"]).to_numpy()
        out["per_area"][area] = _hln_newey_west_dm(d)
    pooled = (per_fold.group_by("test_day")
              .agg(pl.col(f"mae_{a}").mean(), pl.col(f"mae_{b}").mean())
              .sort("test_day"))
    d = (pooled[f"mae_{a}"] - pooled[f"mae_{b}"]).to_numpy()
    out["pooled_across_areas"] = _hln_newey_west_dm(d)
    out["n_areas_p_lt_05_a_better"] = sum(
        1 for v in out["per_area"].values() if v["p_hln"] < 0.05 and v["mean_diff"] < 0)
    return out


def step_aggregate(summary: dict) -> None:
    frames = [pl.read_parquet(f) for f in sorted(CKPT.glob("*.parquet"))]
    if not frames:
        log.error("no head-to-head checkpoints found in %s", CKPT)
        return
    panel = pl.concat(frames)
    panel.write_parquet(PER_SLOT_PARQUET)
    metas = {json.loads(f.read_text())["area"]: json.loads(f.read_text())
             for f in sorted(CKPT.glob("*_meta.json"))}
    tiers = ("era5", "histfc", "prev1")
    panel = panel.with_columns(
        *[((pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs()
           - (pl.col("y_pred_price_only") - pl.col("y_actual")).abs()).alias(f"d_{c}")
          for c in tiers])

    def _improvements(sub: pl.DataFrame) -> dict:
        return {c: round(float(-sub[f"d_{c}"].mean()), 4) for c in tiers}

    per_area_rows = (panel.group_by("area").agg(
        pv_peak=pl.col("pv_peak_ratio").first(),
        **{f"imp_{c}": -pl.col(f"d_{c}").mean() for c in tiers},
        n=pl.len()).sort("pv_peak", descending=True))
    per_fold = _per_fold_mae(panel)
    dm = {"prev1_vs_price_only": _dm_pair(per_fold, "prev1", "price_only"),
          "prev1_vs_era5_proxy": _dm_pair(per_fold, "prev1", "era5"),
          "histfc_vs_price_only": _dm_pair(per_fold, "histfc", "price_only"),
          "era5_vs_price_only": _dm_pair(per_fold, "era5", "price_only"),
          "prev1_vs_histfc": _dm_pair(per_fold, "prev1", "histfc")}
    pv = per_area_rows["pv_peak"].to_numpy()
    spearman = {}
    for c in tiers:
        sp = stats.spearmanr(pv, per_area_rows[f"imp_{c}"].to_numpy())
        spearman[c] = {"rho": round(float(sp.statistic), 3),
                       "p": round(float(sp.pvalue), 3)}
    # reconstruction vs true previous-day forecast
    recon = {"per_area_gap_histfc_minus_prev1_yen_kwh": {} }
    gaps = []
    for r in per_area_rows.to_dicts():
        g = float(r["imp_histfc"] - r["imp_prev1"])
        gaps.append(g)
        recon["per_area_gap_histfc_minus_prev1_yen_kwh"][r["area"]] = round(g, 4)
    recon["mean_abs_gap_yen_kwh"] = round(float(np.mean(np.abs(gaps))), 4)
    recon["mean_gap_yen_kwh"] = round(float(np.mean(gaps)), 4)
    recon["midday_shortwave_corr_histfc_vs_prev1"] = {
        a: m["midday_shortwave_corr_histfc_vs_prev1"] for a, m in metas.items()}
    recon["midday_shortwave_corr_mean"] = round(float(np.mean(
        [m["midday_shortwave_corr_histfc_vs_prev1"] for m in metas.values()])), 4)
    recon["dm_prev1_vs_histfc"] = dm["prev1_vs_histfc"]

    # strictly gate-admissible morning subset (delivery slots 00:00-10:00 JST)
    morning = panel.filter(
        (pl.col("datetime_jst").dt.hour() < ADMISSIBLE_LAST_SLOT_HOUR)
        | ((pl.col("datetime_jst").dt.hour() == ADMISSIBLE_LAST_SLOT_HOUR)
           & (pl.col("datetime_jst").dt.minute() == 0)))
    per_fold_m = _per_fold_mae(morning)
    admissible = {
        "definition": ("delivery slots 00:00-10:00 JST only: the previous_day1 run "
                       "(init ~24-27 h before delivery) is initialised strictly "
                       "before the D-1 10:00 JST JEPX gate closure"),
        "n_obs": int(morning.height),
        "pooled_improvement_yen_kwh": _improvements(morning),
        "dm_prev1_vs_price_only_pooled": _dm_pair(per_fold_m, "prev1", "price_only")[
            "pooled_across_areas"],
        "dm_prev1_vs_era5_pooled": _dm_pair(per_fold_m, "prev1", "era5")[
            "pooled_across_areas"],
    }
    test_days = sorted(panel["test_day"].unique().to_list())
    months = sorted({d[:7] for d in test_days})
    by_cal_month = (panel.unique(subset=["area", "test_day"])
                    .with_columns(pl.col("test_day").str.slice(5, 2).alias("mm"))
                    .group_by("mm").agg(pl.n_unique("test_day").alias("n_days"))
                    .sort("mm"))
    old = panel.filter(pl.col("test_day") <= OLD_WINDOW_LAST_DAY.isoformat())
    new = panel.filter(pl.col("test_day") > OLD_WINDOW_LAST_DAY.isoformat())

    summary["headtohead"] = {
        "protocol": ("phase35/41 all-hours protocol unchanged: XGBRegressor(300, depth 4, "
                     "lr 0.05, seed 2026), walk-forward initial_train_end 2024-09-01, "
                     "val 14 d, test 1 d, step 3 d, min test day 2024-11-15, 4 same-sample "
                     "cells on identical folds; panel from 2024-04-01 (FY2024)"),
        "tiers": {"era5": "lag-48 ERA5 lagged-reanalysis proxy (paper convention)",
                  "histfc": "jma_msm Historical-Forecast reconstruction (paper op tier)",
                  "prev1": "jma_msm previous_day1 = TRUE previous-day run (day-ahead "
                           "class, run-pinned to D-1); see probe note"},
        "window": {"first_test_day": test_days[0], "last_test_day": test_days[-1],
                   "n_test_days": len(test_days),
                   "n_folds_per_area": {a: m["n_folds"] for a, m in metas.items()},
                   "n_obs": int(panel.height), "n_areas": int(per_area_rows.height),
                   "calendar_months_covered": months,
                   "n_calendar_months": len(months),
                   "test_days_by_calendar_month": {r["mm"]: r["n_days"]
                                                   for r in by_cal_month.to_dicts()}},
        "pooled_improvement_yen_kwh": _improvements(panel),
        "per_area": {r["area"]: {
            "imp_era5": round(float(r["imp_era5"]), 4),
            "imp_histfc": round(float(r["imp_histfc"]), 4),
            "imp_prev1": round(float(r["imp_prev1"]), 4),
            "pv_peak": round(float(r["pv_peak"]), 3), "n": int(r["n"])}
            for r in per_area_rows.to_dicts()},
        "dm_hln_nw_l3": dm,
        "cross_area_spearman_vs_pvpeak": spearman,
        "reconstruction_vs_true_forecast": recon,
        "strictly_admissible_morning_subset": admissible,
        "window_split": {
            "committed_window_test_days_le_2025_11_06": {
                "n_obs": int(old.height), "improvement": _improvements(old)},
            "extension_only_test_days_gt_2025_11_06": {
                "n_obs": int(new.height), "improvement": _improvements(new)},
        },
        "committed_reference_phase41": {
            "note": "committed FY2024-25 study-window numbers (untouched): "
                    "phase41_allhours_headtohead_9area.json, n=50,544",
            "era5_proxy": 0.0214, "histfc_reconstruction": 0.2342,
            "gpv_exact_issuance": 0.2484},
        "runtime_s_per_area": {a: m["runtime_s"] for a, m in metas.items()},
    }
    # freeze-append: hashes of the weather caches created for this phase
    hf_dir = RAW / "open_meteo_historical_forecast"
    freeze = summary.setdefault("freeze", {"files": {}})
    freeze["files"]["open_meteo_prev1_caches"] = {
        f.name: _sha256(f) for f in sorted(hf_dir.glob("open_meteo_prev1_*.parquet"))}
    freeze["files"]["open_meteo_histfc_extended_span"] = {
        f.name: _sha256(f)
        for f in sorted(hf_dir.glob("open_meteo_histfc_jma_msm_*_20240401_*.json"))}
    _save_summary(summary)


# =================================================================== STEP 4: decision
def _imbalance_long(upto_ym: str | None = None) -> pl.DataFrame:
    rows: list[dict] = []
    for f in sorted((RAW / "jepx_imbalance").glob("*.csv")):
        ym = f.name.split("_")[2][:6] if f.name[0].isdigit() else None
        if upto_ym and ym and ym > upto_ym:
            continue
        rows.extend(_parse_imbalance_month_csv(
            f.read_bytes().decode("shift_jis", errors="replace")))
    return (pl.DataFrame(rows,
                         schema={"area": pl.Utf8,
                                 "datetime_jst": pl.Datetime(time_unit="us",
                                                             time_zone="Asia/Tokyo"),
                                 "shortage_imb_yen_kwh": pl.Float64})
            .unique(subset=["area", "datetime_jst"]))


def _spot_long() -> pl.DataFrame:
    frames = [parse_jepx_spot_csv(RAW / "jepx" / f"spot_{y}.csv")
              for y in (2024, 2025, 2026) if (RAW / "jepx" / f"spot_{y}.csv").exists()]
    return pl.concat(frames).select("datetime_jst", "area", "price_yen_kwh")


def _cf_threshold(y_pred: np.ndarray, floor: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-fitted max-F1 floor threshold (phase31c protocol, seed 2026)."""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(SEED)
    fold_of = {g: int(rng.integers(0, 5)) for g in uniq}
    fold = np.array([fold_of[g] for g in groups])
    pred_floor = np.zeros(len(y_pred), dtype=bool)
    cand = np.quantile(y_pred, np.linspace(0.01, 0.30, 60))
    for k in range(5):
        tr, te = fold != k, fold == k
        best_tau, best_f1 = cand[0], -1.0
        for tau in cand:
            pf = y_pred[tr] <= tau
            tp = np.sum(pf & floor[tr])
            fp = np.sum(pf & ~floor[tr])
            fn = np.sum(~pf & floor[tr])
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        pred_floor[te] = y_pred[te] <= best_tau
    return pred_floor


def _decision_cell(df: pl.DataFrame, pred_col: str) -> dict:
    floor = df["floor"].to_numpy()
    cost = df["real_over_cost"].to_numpy()
    groups = (df["area"] + "|" + df["test_day"]).to_numpy()
    pred_floor = _cf_threshold(df[pred_col].to_numpy().astype(float), floor, groups)
    unhedged = floor & ~pred_floor
    exposure = float(cost[unhedged].sum())
    n_floor = int(floor.sum())
    return {"floor_recall": round(float((floor & pred_floor).sum() / max(n_floor, 1)), 3),
            "unhedged_real_exposure_yen_kwh": round(exposure, 1),
            "real_decision_cost_yen_per_mw_yr": round(
                exposure / df.height * SLOTS_PER_YEAR * 1000.0, 0)}


def _decision_window(df: pl.DataFrame, label: str) -> dict:
    floor = df["floor"].to_numpy()
    cost = df["real_over_cost"].to_numpy()
    res: dict = {"label": label, "n_obs": int(df.height), "n_floor": int(floor.sum()),
                 "first_test_day": df["test_day"].min(),
                 "last_test_day": df["test_day"].max(),
                 "floor_real_over_cost_yen_kwh_median": round(
                     float(np.median(cost[floor])), 3) if floor.sum() else None,
                 "total_real_floor_exposure_yen_kwh": round(float(cost[floor].sum()), 1),
                 "tiers": {}}
    for tier, col in (("price_only", "y_pred_price_only"),
                      ("operational_prev1", "y_pred_prev1"),
                      ("operational_histfc", "y_pred_histfc")):
        res["tiers"][tier] = _decision_cell(df, col)
    po = res["tiers"]["price_only"]["real_decision_cost_yen_per_mw_yr"]
    res["operational_value_yen_per_mw_yr"] = {
        "prev1": round(po - res["tiers"]["operational_prev1"][
            "real_decision_cost_yen_per_mw_yr"], 0),
        "histfc": round(po - res["tiers"]["operational_histfc"][
            "real_decision_cost_yen_per_mw_yr"], 0)}
    return res


def step_decision(summary: dict) -> None:
    if not PER_SLOT_PARQUET.exists():
        frames = [pl.read_parquet(f) for f in sorted(CKPT.glob("*.parquet"))]
        pl.concat(frames).write_parquet(PER_SLOT_PARQUET)
    panel = pl.read_parquet(PER_SLOT_PARQUET)
    imb = _imbalance_long()
    spot = _spot_long()
    df = (panel.join(imb, on=["area", "datetime_jst"], how="inner")
          .join(spot, on=["area", "datetime_jst"], how="left")
          .with_columns(spot=pl.coalesce("price_yen_kwh", "y_actual"))
          .drop_nulls(subset=["shortage_imb_yen_kwh"])
          .with_columns(
              floor=(pl.col("y_actual") <= FLOOR),
              real_over_cost=(pl.col("shortage_imb_yen_kwh") - pl.col("spot"))
              .clip(0.0, None)))
    d8 = df.filter(pl.col("area").is_in(DECISION_AREAS_8))
    old8 = d8.filter(pl.col("test_day") <= OLD_WINDOW_LAST_DAY.isoformat())
    # floor-slot imbalance wedge by quarter (context for old-vs-extended comparison)
    wedge_by_q = (d8.filter(pl.col("floor"))
                  .with_columns(q=(pl.col("test_day").str.slice(0, 4) + "Q"
                                   + ((pl.col("test_day").str.slice(5, 2)
                                       .cast(pl.Int32) - 1) // 3 + 1).cast(pl.Utf8)))
                  .group_by("q").agg(
                      n_floor=pl.len(),
                      wedge_median=pl.col("real_over_cost").median().round(3),
                      wedge_mean=pl.col("real_over_cost").mean().round(3),
                      wedge_p75=pl.col("real_over_cost").quantile(0.75).round(3))
                  .sort("q"))
    decision = {
        "protocol": ("phase31c real-yen floor-decision protocol unchanged: floor = "
                     "y_actual <= 0.5 yen/kWh; real over-forecast cost = "
                     "max(shortage_imbalance - spot, 0); cross-fitted max-F1 threshold "
                     "(5-fold on area|test_day groups, seed 2026); 1 MWh/slot "
                     "continuous-throughput scaling to yen/MW/yr"),
        "committed_reference": {
            "note": "committed phase31c cell (UNTOUCHED): phase26.1 histfc panel x "
                    "imbalance 2024-09..2025-10, 8 areas, window 2024-11-17..2025-11-06",
            "operational_value_yen_per_mw_yr": 177427.0},
        "predictions_source": ("phase49 per-slot walk-forward (this run); tiers "
                               "price_only / prev1 (true previous-day run) / histfc "
                               "(reconstruction)"),
        "extended_window_8area": _decision_window(
            d8, "EXTENDED-WINDOW (phase49 panel, 8 committed decision areas)"),
        "old_window_bridge_8area": _decision_window(
            old8, "bridge: phase49 panel restricted to committed test window "
                  "(<= 2025-11-06), 8 areas"),
        "extended_window_9area": _decision_window(
            df, "EXTENDED-WINDOW (phase49 panel, 9 areas incl. Chubu; secondary)"),
        "floor_wedge_by_quarter_8area": {
            "note": ("median/mean of max(shortage_imbalance - spot, 0) on floored "
                     "slots, by test-day quarter; documents the wedge regime shift "
                     "after the committed window"),
            "rows": wedge_by_q.to_dicts()},
    }
    # phase26.1-panel variant with the extended imbalance cache (committed inputs,
    # imbalance months now through the freeze date -> adds only 2025-11-01..06 slots)
    p26 = RUN / "phase26.1_pv_surplus_per_slot.parquet"
    if p26.exists():
        panel26 = pl.read_parquet(p26).select(
            "area", "datetime_jst", "test_day", "y_actual",
            "y_pred_price_only", "y_pred_histfc")
        df26 = (panel26.join(imb, on=["area", "datetime_jst"], how="inner")
                .join(spot, on=["area", "datetime_jst"], how="left")
                .with_columns(spot=pl.coalesce("price_yen_kwh", "y_actual"))
                .drop_nulls(subset=["shortage_imb_yen_kwh"])
                .with_columns(
                    floor=(pl.col("y_actual") <= FLOOR),
                    real_over_cost=(pl.col("shortage_imb_yen_kwh") - pl.col("spot"))
                    .clip(0.0, None)))
        floor = df26["floor"].to_numpy()
        cost = df26["real_over_cost"].to_numpy()
        cell = {}
        for tier, col in (("price_only", "y_pred_price_only"),
                          ("operational_histfc", "y_pred_histfc")):
            cell[tier] = _decision_cell(df26, col)
        po = cell["price_only"]["real_decision_cost_yen_per_mw_yr"]
        decision["phase26_panel_imbalance_extended"] = {
            "label": "phase31c inputs with imbalance cache extended to freeze date "
                     "(adds matched slots only 2025-11-01..2025-11-06)",
            "n_obs": int(df26.height), "n_floor": int(floor.sum()),
            "total_real_floor_exposure_yen_kwh": round(float(cost[floor].sum()), 1),
            "tiers": cell,
            "operational_value_yen_per_mw_yr": round(
                po - cell["operational_histfc"]["real_decision_cost_yen_per_mw_yr"], 0)}
    summary["decision_extended"] = decision
    _save_summary(summary)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["probe", "refresh", "headtohead", "decision",
                             "aggregate", "all"])
    ap.add_argument("--areas", default=None,
                    help="comma-separated subset for --step headtohead")
    ap.add_argument("--fresh", action="store_true",
                    help="drop head-to-head checkpoints before running")
    ap.add_argument("--skip-probe", action="store_true",
                    help="with --step all: reuse the probe section already in the JSON")
    args = ap.parse_args()
    t0 = time.time()
    summary = _load_summary()
    areas = args.areas.split(",") if args.areas else None

    if args.step in ("probe", "all") and not args.skip_probe:
        gate_ok = step_probe(summary)
        if args.step == "probe":
            return 0
        if not gate_ok:
            log.error("probe gate FAILED -> fallback branch (single-runs subset "
                      "extension) required; stopping.")
            return 1
    if args.step in ("refresh", "all"):
        step_refresh(summary)
        if args.step == "refresh":
            return 0
    if args.step in ("headtohead", "all"):
        step_headtohead(areas, args.fresh)
        if args.step == "headtohead":
            return 0
    if args.step in ("aggregate", "all"):
        summary = _load_summary()
        step_aggregate(summary)
        if args.step == "aggregate":
            return 0
    if args.step in ("decision", "all"):
        summary = _load_summary()
        step_decision(summary)
    summary = _load_summary()
    summary.setdefault("metadata", {})["total_runtime_s_last_invocation"] = round(
        time.time() - t0, 1)
    _save_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
