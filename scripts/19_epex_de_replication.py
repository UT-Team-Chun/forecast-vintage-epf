r"""Phase 45: EPEX-DE multi-year TWO-TIER external replication (proxy vs operational).

The paper's headline (operational weather forecasts carry ~8x the day-ahead
price-forecast value of a lagged-reanalysis proxy, and the value is broad) has no
external replication: phase38's EPEX-DE check tested only a single weather tier on
2024. This script replicates the TIERED design on Germany, multi-year, credential-free:

  price_only  : price lags (24/48/168h) + calendar        [phase38 protocol]
  +proxy      : + ERA5 reanalysis lagged 24 h              (JEPX paper's proxy tier)
  +operational: + Open-Meteo Historical-Forecast API
                 ``*_previous_day1`` variables (icon_seamless) = the DWD-ICON run
                 issued ~1 day before delivery, i.e. the day-ahead-class forecast
                 available around the EPEX DE/LU gate closure (12:00 CET D-1)

Data (all keyless):
  - SMARD.de (Bundesnetzagentur): day-ahead price 4169, PV 4068, load 410,
    wind onshore 4067, wind offshore 1225 (hourly, DE/LU).
  - Open-Meteo archive-api (ERA5) + historical-forecast-api (icon_seamless
    previous_day1). Single DE-centre point (51.2, 10.4), same as phase38.

STEP-1 PROBE (gate; embedded, small requests): icon_seamless base archive starts
2022-11; the ``previous_day1`` lagged-issue variables start ~2024-02 (wind_100m
~2024-03) -> >=2 years of operational-tier coverage overlapping SMARD -> gate PASSES.

Issuance caveat (honest, load-bearing): ``previous_day1`` shifts the model run used
for each delivery hour back by ~24 h (init ~24-27 h before delivery). Morning
delivery hours are strictly admissible w.r.t. the 12:00 CET D-1 gate closure;
afternoon/evening hours may use a D-1 run initialised up to ~11 h after gate
closure. This is the same reconstruction-ambiguity class as the Japan histfc tier
(phase26.0), NOT an exact-issuance build; we never call it "ex-ante".

Secondary checks (pre-registered):
  (i)  channel ablation (add-one family x tier): is the operational tier's gain
       concentrated in wind channels? (Japan phase42: op-over-proxy skill gain is
       largest in wind/cloud channels.)
  (ii) negative-price-exposure temporal analog (per-year + daily Spearman +
       long proxy-only walk-forward 2021->2026) -- EXPLORATORY ONLY (pre-committed
       gate; never promoted regardless of outcome).

CLI:
  cd <repository root>
  uv run --extra research python scripts/19_epex_de_replication.py \
      [--fresh] [--skip-long] [--skip-probe]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig

log = logging.getLogger("phase45")

# ---------------------------------------------------------------- pinned fetch params
SMARD = "https://www.smard.de/app/chart_data"
OPEN_METEO_ERA5 = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HISTFC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SMARD_IDS = {"price": 4169, "pv": 4068, "load": 410,
             "wind_onshore": 4067, "wind_offshore": 1225}
DE_COORD = (51.2, 10.4)  # geographic centre of Germany, same point as phase38
HISTFC_MODEL = "icon_seamless"  # DWD ICON blend; icon_d2 has archive holes (2024-01)
WX_VARS = ("shortwave_radiation", "temperature_2m", "cloud_cover",
           "wind_speed_10m", "wind_speed_100m")
SMARD_START = date(2020, 1, 1)          # proxy-tier history (cheap; exploratory long WF)
HISTFC_START = date(2024, 2, 1)         # previous_day1 archive starts ~2024-02/03
ERA5_DELAY_DAYS = 8                     # ERA5 archive publication delay guard
HISTFC_DELAY_DAYS = 3
NEG_THRESHOLD = 0.0                     # EUR/MWh; DE oversupply boundary (phase38)
BURN_IN_DAYS = 120                      # phase38 expanding-window burn-in
MIN_TEST_HOURS = 20                     # phase38 skip rule
SEED = 2026
FAMILIES: dict[str, tuple[str, ...]] = {
    "solar": ("shortwave_radiation",),
    "cloud": ("cloud_cover",),
    "temp": ("temperature_2m",),
    "wind": ("wind_speed_10m", "wind_speed_100m"),
}
UA = {"User-Agent": "forecast-vintage-epf/1.0"}


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


# ------------------------------------------------------------------------- STEP 1 probe
def _first_covered_day(client: httpx.Client, hourly: str, lo: date, hi: date) -> str:
    """Bisect (day granularity) the first date with fully non-null hourly values."""
    def covered(d: date) -> bool:
        js = _get_json(client, OPEN_METEO_HISTFC, {
            "latitude": DE_COORD[0], "longitude": DE_COORD[1],
            "start_date": d.isoformat(), "end_date": d.isoformat(),
            "hourly": hourly, "models": HISTFC_MODEL, "timezone": "UTC"})
        h = js.get("hourly", {})
        fracs = [sum(x is not None for x in v) / max(len(v), 1)
                 for k, v in h.items() if k != "time"]
        time.sleep(0.4)
        return bool(fracs) and min(fracs) >= 0.99

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if covered(mid):
            hi = mid
        else:
            lo = mid + timedelta(days=1)
    return lo.isoformat()


def _probe_histfc() -> dict:
    """Empirical coverage probe of the Historical-Forecast API for DE (small requests)."""
    with httpx.Client(timeout=30.0, headers=UA) as c:
        base_start = _first_covered_day(
            c, ",".join(WX_VARS), date(2022, 11, 1), date(2022, 12, 15))
        prev1_core = _first_covered_day(
            c, ",".join(f"{v}_previous_day1" for v in WX_VARS if v != "wind_speed_100m"),
            date(2024, 1, 10), date(2024, 3, 20))
        prev1_wind100 = _first_covered_day(
            c, "wind_speed_100m_previous_day1", date(2024, 1, 10), date(2024, 4, 20))
    probe = {
        "endpoint": OPEN_METEO_HISTFC,
        "grid_point": list(DE_COORD),
        "models_checked": ["icon_seamless", "icon_eu", "icon_d2"],
        "model_selected": HISTFC_MODEL,
        "model_notes": ("icon_seamless base archive from ~2022-11; icon_eu from ~2022-12; "
                        "icon_d2 has archive holes (e.g. 2024-01-15 empty) -> excluded."),
        "variables": list(WX_VARS),
        "base_archive_first_full_day": base_start,
        "previous_day1_first_full_day_core4": prev1_core,
        "previous_day1_first_full_day_wind100m": prev1_wind100,
        "operational_tier_definition": (
            "previous_day1 variables on the Historical-Forecast API = value for each "
            "delivery hour taken from the ICON run issued ~1 day earlier (init ~24-27 h "
            "before delivery), i.e. the day-ahead-class forecast around EPEX DE/LU gate "
            "closure (12:00 CET D-1). Morning delivery hours strictly admissible; "
            "afternoon/evening hours may use a D-1 run up to ~11 h after gate closure "
            "(reconstruction ambiguity, same class as the Japan histfc tier)."),
        "gate": None,  # filled in main()
    }
    return probe


# ------------------------------------------------------------------------- STEP 2 data
def _fetch_smard_series(client: httpx.Client, fid: int, lo_ms: int, hi_ms: int) -> pl.DataFrame:
    """One SMARD hourly series across the weekly files covering [lo_ms, hi_ms) (phase38)."""
    idx = _get_json(client, f"{SMARD}/{fid}/DE/index_hour.json")["timestamps"]
    weeks = [t for t in idx if lo_ms - 7 * 86400_000 <= t <= hi_ms]
    rows: list[tuple[int, float]] = []
    for ts in weeks:
        data = _get_json(client, f"{SMARD}/{fid}/DE/{fid}_DE_hour_{ts}.json")["series"]
        rows.extend((t, v) for t, v in data if v is not None and lo_ms <= t < hi_ms)
        time.sleep(0.05)
    df = pl.DataFrame(rows, schema=["ts_ms", "value"], orient="row")
    return df.unique("ts_ms").sort("ts_ms")


def _load_smard_year(raw_root: Path, year: int, end_utc: datetime) -> pl.DataFrame:
    """All 5 SMARD series for one calendar year (cached; current year re-fetched if stale)."""
    cache = raw_root / "smard_de" / f"smard5_{year}.parquet"
    lo = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    hi = int(min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), end_utc).timestamp() * 1000)
    if cache.exists():
        df = pl.read_parquet(cache)
        if year < end_utc.year or df["ts_ms"].max() >= hi - 36 * 3600_000:
            log.info("SMARD cache hit %s (%d rows)", cache.name, df.height)
            return df
    out: pl.DataFrame | None = None
    with httpx.Client(timeout=30.0, headers=UA) as client:
        for name, fid in SMARD_IDS.items():
            s = _fetch_smard_series(client, fid, lo, hi).rename({"value": name})
            log.info("SMARD %d %s (id %d): %d hourly points", year, name, fid, s.height)
            how = "full" if name != "price" else "inner"
            out = s if out is None else out.join(s, on="ts_ms", how=how, coalesce=True)
    assert out is not None
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.sort("ts_ms").write_parquet(cache)
    return out


def _load_smard(raw_root: Path, end_utc: datetime) -> pl.DataFrame:
    years = list(range(SMARD_START.year, end_utc.year + 1))
    df = pl.concat([_load_smard_year(raw_root, y, end_utc) for y in years],
                   how="diagonal").unique("ts_ms").sort("ts_ms")
    return df.with_columns(
        pl.from_epoch(pl.col("ts_ms"), time_unit="ms").dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us").alias("datetime_utc"))


def _fetch_open_meteo_year(raw_root: Path, year: int, *, kind: str,
                           end_day: date) -> pl.DataFrame | None:
    """One calendar year of DE-centre hourly weather (kind = 'era5' | 'prev1'); cached."""
    start = date(year, 1, 1) if kind == "era5" else max(date(year, 1, 1), HISTFC_START)
    end = min(date(year, 12, 31), end_day)
    if start > end:
        return None
    cache = raw_root / "smard_de" / f"open_meteo_de_{kind}_{year}.parquet"
    if cache.exists():
        df = pl.read_parquet(cache)
        if year < end.year or df["datetime_utc"].max().date() >= end - timedelta(days=1):
            log.info("Open-Meteo cache hit %s (%d rows)", cache.name, df.height)
            return df
    if kind == "era5":
        url, params = OPEN_METEO_ERA5, {
            "latitude": DE_COORD[0], "longitude": DE_COORD[1],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": ",".join(WX_VARS), "timezone": "UTC"}
        cols = {v: v for v in WX_VARS}
    else:
        url, params = OPEN_METEO_HISTFC, {
            "latitude": DE_COORD[0], "longitude": DE_COORD[1],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": ",".join(f"{v}_previous_day1" for v in WX_VARS),
            "models": HISTFC_MODEL, "timezone": "UTC"}
        cols = {f"{v}_previous_day1": f"{v}_prev1" for v in WX_VARS}
    with httpx.Client(timeout=120.0, headers=UA) as client:
        h = _get_json(client, url, params)["hourly"]
    time.sleep(1.0)
    df = pl.DataFrame({
        "datetime_utc": [datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                         for t in h["time"]],
        **{out_name: h[api_name] for api_name, out_name in cols.items()},
    }).with_columns(pl.col("datetime_utc").dt.cast_time_unit("us"))
    log.info("Open-Meteo %s %d: %d hourly rows fetched", kind, year, df.height)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def _fetch_open_meteo(raw_root: Path, *, kind: str, start_year: int, end_day: date) -> pl.DataFrame:
    frames = [_fetch_open_meteo_year(raw_root, y, kind=kind, end_day=end_day)
              for y in range(start_year, end_day.year + 1)]
    return pl.concat([f for f in frames if f is not None]).unique("datetime_utc").sort("datetime_utc")


# --------------------------------------------------------------------- panel + features
def _build_panel(smard: pl.DataFrame, era5: pl.DataFrame, prev1: pl.DataFrame) -> pl.DataFrame:
    """Left-join everything onto a complete hourly UTC spine so shifts are true hours."""
    lo, hi = smard["datetime_utc"].min(), smard["datetime_utc"].max()
    spine = pl.DataFrame({"datetime_utc": pl.datetime_range(
        lo, hi, interval="1h", time_zone="UTC", time_unit="us", eager=True)})
    df = (spine.join(smard.drop("ts_ms"), on="datetime_utc", how="left")
          .join(era5, on="datetime_utc", how="left")
          .join(prev1, on="datetime_utc", how="left")
          .sort("datetime_utc"))
    df = df.with_columns(
        pl.col("datetime_utc").dt.hour().alias("hour"),
        pl.col("datetime_utc").dt.weekday().alias("dow"),
        pl.col("datetime_utc").dt.ordinal_day().alias("doy"),
        pl.col("price").shift(24).alias("price_lag24"),
        pl.col("price").shift(48).alias("price_lag48"),
        pl.col("price").shift(168).alias("price_lag168"),
        *[pl.col(v).shift(24).alias(f"{v}_lag24") for v in WX_VARS],
    )
    return df


def _cells() -> dict[str, list[str]]:
    base = ["hour", "dow", "doy", "price_lag24", "price_lag48", "price_lag168"]
    cells = {
        "price_only": base,
        "proxy_full": [*base, *(f"{v}_lag24" for v in WX_VARS)],
        "op_full": [*base, *(f"{v}_prev1" for v in WX_VARS)],
        # phase38 convention (realised ERA5 at delivery hour, 3 channels) -- robustness
        "era5_realized_p38": [*base, "shortwave_radiation", "temperature_2m", "cloud_cover"],
    }
    for fam, vars_ in FAMILIES.items():
        cells[f"proxy_{fam}"] = [*base, *(f"{v}_lag24" for v in vars_)]
        cells[f"op_{fam}"] = [*base, *(f"{v}_prev1" for v in vars_)]
    return cells


def _fit_predict(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=4, verbosity=0, tree_method="hist")
    model.fit(train.select(cols).to_numpy().astype(float),
              train["price"].to_numpy().astype(float))
    return model.predict(test.select(cols).to_numpy().astype(float)).astype(float)


def _walk_forward(df: pl.DataFrame, cells: dict[str, list[str]], *,
                  checkpoint: Path, resume: bool) -> pl.DataFrame:
    """Daily-rolling 1-day-ahead expanding-window XGB (phase38 protocol), all cells
    on IDENTICAL folds. Per-day records with per-cell MAE; checkpointed."""
    need = sorted({c for cols in cells.values() for c in cols} | {"price"})
    df = df.drop_nulls(subset=need).with_columns(
        pl.col("datetime_utc").dt.date().alias("d"))
    days = df["d"].unique().sort().to_list()
    log.info("walk-forward panel: %d rows, %d days (%s..%s), %d cells",
             df.height, len(days), days[0], days[-1], len(cells))
    done: pl.DataFrame | None = None
    done_days: set = set()
    if resume and checkpoint.exists():
        done = pl.read_parquet(checkpoint)
        done_days = set(done["d"].to_list())
        log.info("resume: %d days already computed in %s", len(done_days), checkpoint.name)
    recs: list[dict] = []
    t0 = time.time()
    for i in range(BURN_IN_DAYS, len(days)):
        day = days[i]
        if day in done_days:
            continue
        train = df.filter(pl.col("d") < day)
        test = df.filter(pl.col("d") == day)
        if test.height < MIN_TEST_HOURS:
            continue
        y = test["price"].to_numpy().astype(float)
        rec: dict = {
            "d": day, "n": test.height,
            "neg_share": float(np.mean(y <= NEG_THRESHOLD)),
            "pv_mwh_mean": float(test["pv"].mean()) if test["pv"].null_count() < test.height else None,
            "load_mwh_mean": float(test["load"].mean()) if test["load"].null_count() < test.height else None,
        }
        for name, cols in cells.items():
            pred = _fit_predict(train, test, cols)
            rec[f"mae_{name}"] = float(np.mean(np.abs(pred - y)))
        recs.append(rec)
        if len(recs) % 25 == 0:
            partial = pl.DataFrame(recs)
            (pl.concat([done, partial], how="diagonal") if done is not None else partial
             ).write_parquet(checkpoint)
            el = time.time() - t0
            log.info("day %s (%d/%d) | %.1f s/day | mae price_only=%.2f op=%.2f",
                     day, i - BURN_IN_DAYS + 1, len(days) - BURN_IN_DAYS,
                     el / len(recs), rec["mae_price_only"],
                     rec.get("mae_op_full", float("nan")))
    out = pl.DataFrame(recs) if recs else pl.DataFrame()
    if done is not None:
        out = pl.concat([done, out], how="diagonal") if out.height else done
    out = out.sort("d")
    out.write_parquet(checkpoint)
    return out


# ------------------------------------------------------------------------- statistics
def _hln_newey_west_dm(d: np.ndarray, *, h: int = 1, lag: int = 3) -> dict:
    """Diebold-Mariano on per-fold loss differences d (=loss_a - loss_b) with
    Newey-West HAC (Bartlett) variance + HLN small-sample correction (phase23.5)."""
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
    return {"mean_diff": d_bar, "t_hln": float(t_stat), "p_hln": float(p),
            "n": n, "nw_lag": lag}


def _tier_stats(daily: pl.DataFrame, a: str, b: str) -> dict:
    """Improvement of cell a over cell b (positive = a better), HLN-NW DM at L=3
    (repo convention, phase23.5) + automatic Bartlett lag as robustness."""
    d = (daily[f"mae_{a}"] - daily[f"mae_{b}"]).to_numpy()  # a - b; negative = a better
    lag_auto = max(3, int(np.floor(1.5 * len(d) ** (1 / 3))))
    return {
        "mean_improvement_eur_mwh": float(-d.mean()),
        "dm_hln_l3": _hln_newey_west_dm(d, lag=3),
        "dm_hln_lauto": _hln_newey_west_dm(d, lag=lag_auto),
    }


def _per_year(daily: pl.DataFrame) -> list[dict]:
    out = []
    daily = daily.with_columns(pl.col("d").dt.year().alias("year"))
    for year in sorted(daily["year"].unique().to_list()):
        sub = daily.filter(pl.col("year") == year)
        proxy = _tier_stats(sub, "proxy_full", "price_only")
        op = _tier_stats(sub, "op_full", "price_only")
        head = _tier_stats(sub, "op_full", "proxy_full")
        ratio = (op["mean_improvement_eur_mwh"] / proxy["mean_improvement_eur_mwh"]
                 if proxy["mean_improvement_eur_mwh"] != 0 else float("nan"))
        out.append({
            "year": int(year), "n_days": sub.height,
            "d_first": sub["d"].min().isoformat(), "d_last": sub["d"].max().isoformat(),
            "mae_price_only": round(float(sub["mae_price_only"].mean()), 3),
            "improvement_proxy": round(proxy["mean_improvement_eur_mwh"], 3),
            "improvement_op": round(op["mean_improvement_eur_mwh"], 3),
            "ratio_op_over_proxy": round(ratio, 2),
            "dm_p_proxy_vs_price_l3": round(proxy["dm_hln_l3"]["p_hln"], 5),
            "dm_p_op_vs_price_l3": round(op["dm_hln_l3"]["p_hln"], 5),
            "dm_p_op_vs_proxy_l3": round(head["dm_hln_l3"]["p_hln"], 5),
            "neg_price_share": round(float(sub["neg_share"].mean()), 4),
        })
    return out


def _channel_ablation(daily: pl.DataFrame) -> dict:
    """Add-one-family gains vs price_only per tier + operational-over-proxy delta."""
    out: dict = {"design": ("add-one channel family on top of price_only, per tier, "
                            "identical folds; gain = mean(MAE_price_only - MAE_cell); "
                            "delta_op_minus_proxy = op gain - proxy gain per family")}
    fams = {}
    for fam in FAMILIES:
        g_proxy = _tier_stats(daily, f"proxy_{fam}", "price_only")
        g_op = _tier_stats(daily, f"op_{fam}", "price_only")
        fams[fam] = {
            "gain_proxy_eur_mwh": round(g_proxy["mean_improvement_eur_mwh"], 4),
            "gain_op_eur_mwh": round(g_op["mean_improvement_eur_mwh"], 4),
            "delta_op_minus_proxy": round(g_op["mean_improvement_eur_mwh"]
                                          - g_proxy["mean_improvement_eur_mwh"], 4),
            "dm_p_op_vs_price_l3": round(g_op["dm_hln_l3"]["p_hln"], 5),
            "dm_p_op_vs_proxy_l3": round(
                _tier_stats(daily, f"op_{fam}", f"proxy_{fam}")["dm_hln_l3"]["p_hln"], 5),
        }
    out["families"] = fams
    ranked = sorted(fams, key=lambda f: fams[f]["delta_op_minus_proxy"], reverse=True)
    out["rank_by_delta_op_minus_proxy"] = ranked
    out["wind_rank_of_4"] = ranked.index("wind") + 1
    return out


def _long_proxy_walk_forward(df: pl.DataFrame, *, checkpoint: Path, resume: bool) -> pl.DataFrame:
    """EXPLORATORY: proxy-tier-only walk-forward over the full 2021->now span (price_only
    vs proxy_full), for the negative-price temporal-gradient analog."""
    base = ["hour", "dow", "doy", "price_lag24", "price_lag48", "price_lag168"]
    cells = {"price_only": base, "proxy_full": [*base, *(f"{v}_lag24" for v in WX_VARS)]}
    sub = df.drop(*[f"{v}_prev1" for v in WX_VARS])
    sub = sub.filter(pl.col("datetime_utc") >= datetime(2020, 6, 1, tzinfo=timezone.utc))
    return _walk_forward(sub, cells, checkpoint=checkpoint, resume=resume)


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore walk-forward checkpoints")
    ap.add_argument("--skip-long", action="store_true", help="skip exploratory long proxy WF")
    ap.add_argument("--skip-probe", action="store_true", help="reuse pinned probe constants")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = EpfConfig()
    raw_root = cfg.resolved("raw_data_dir")
    run_dir = cfg.resolved("run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- STEP 1: probe (gate) ----
    if args.skip_probe:
        probe = {"skipped": True, "pinned": {
            "base_archive_first_full_day": "2022-11-12",
            "previous_day1_first_full_day_core4": "2024-02-01",
            "previous_day1_first_full_day_wind100m": "2024-03-01"}}
        op_cov_start = date(2024, 3, 1)
    else:
        probe = _probe_histfc()
        op_cov_start = date.fromisoformat(probe["previous_day1_first_full_day_wind100m"])
    today = datetime.now(timezone.utc).date()
    op_cov_days = (today - timedelta(days=HISTFC_DELAY_DAYS) - op_cov_start).days
    gate_pass = op_cov_days >= 365
    probe["gate"] = {
        "criterion": ">= 1 year of DE operational-forecast (previous_day1) coverage "
                     "overlapping SMARD",
        "operational_coverage_start": op_cov_start.isoformat(),
        "operational_coverage_days": op_cov_days,
        "decision": "PASS" if gate_pass else "STOP",
    }
    log.info("PROBE gate: %s (%d days of previous_day1 coverage from %s)",
             probe["gate"]["decision"], op_cov_days, op_cov_start)
    if not gate_pass:
        out = run_dir / "phase45_epex_de_tiered.json"
        out.write_text(json.dumps({"phase": "45", "probe": probe,
                                   "stopped": "gate failed"}, indent=2))
        log.error("gate failed -- stopping after probe; wrote %s", out)
        return 1

    # ---- STEP 2: data ----
    end_utc = datetime.now(timezone.utc)
    era5_end = today - timedelta(days=ERA5_DELAY_DAYS)
    histfc_end = today - timedelta(days=HISTFC_DELAY_DAYS)
    smard = _load_smard(raw_root, end_utc)
    era5 = _fetch_open_meteo(raw_root, kind="era5", start_year=SMARD_START.year, end_day=era5_end)
    prev1 = _fetch_open_meteo(raw_root, kind="prev1", start_year=HISTFC_START.year,
                              end_day=histfc_end)
    panel = _build_panel(smard, era5, prev1)
    log.info("panel: %d hourly rows %s..%s", panel.height,
             panel["datetime_utc"].min(), panel["datetime_utc"].max())

    # ---- STEP 3: two-tier head-to-head (same-sample, all cells identical folds) ----
    cells = _cells()
    daily = _walk_forward(panel, cells,
                          checkpoint=run_dir / "phase45_epex_de_tiered_per_day.parquet",
                          resume=not args.fresh)
    pooled_proxy = _tier_stats(daily, "proxy_full", "price_only")
    pooled_op = _tier_stats(daily, "op_full", "price_only")
    pooled_head = _tier_stats(daily, "op_full", "proxy_full")
    pooled_p38 = _tier_stats(daily, "era5_realized_p38", "price_only")
    ratio = (pooled_op["mean_improvement_eur_mwh"]
             / pooled_proxy["mean_improvement_eur_mwh"]
             if pooled_proxy["mean_improvement_eur_mwh"] != 0 else float("nan"))
    per_year = _per_year(daily)
    ablation = _channel_ablation(daily)

    # exploratory: negative-price concentration analog (pre-committed EXPLORATORY ONLY)
    imp_op = (daily["mae_price_only"] - daily["mae_op_full"]).to_numpy()
    imp_proxy = (daily["mae_price_only"] - daily["mae_proxy_full"]).to_numpy()
    neg = daily["neg_share"].to_numpy()
    rho_op, p_op = stats.spearmanr(imp_op, neg)
    rho_proxy, p_proxy = stats.spearmanr(imp_proxy, neg)
    exploratory: dict = {
        "label": "EXPLORATORY ONLY (pre-committed gate; do not promote regardless of outcome)",
        "spearman_daily_improvement_vs_negshare": {
            "op": {"rho": round(float(rho_op), 4), "p": round(float(p_op), 4)},
            "proxy": {"rho": round(float(rho_proxy), 4), "p": round(float(p_proxy), 4)},
        },
    }
    if not args.skip_long:
        long_daily = _long_proxy_walk_forward(
            panel, checkpoint=run_dir / "phase45_epex_de_long_proxy_per_day.parquet",
            resume=not args.fresh)
        rows = []
        ld = long_daily.with_columns(pl.col("d").dt.year().alias("year"))
        for year in sorted(ld["year"].unique().to_list()):
            sub = ld.filter(pl.col("year") == year)
            st = _tier_stats(sub, "proxy_full", "price_only")
            pv_share = (float(sub["pv_mwh_mean"].mean()) / float(sub["load_mwh_mean"].mean())
                        if sub["load_mwh_mean"].mean() else None)
            rows.append({"year": int(year), "n_days": sub.height,
                         "improvement_proxy": round(st["mean_improvement_eur_mwh"], 3),
                         "dm_p_l3": round(st["dm_hln_l3"]["p_hln"], 5),
                         "mae_price_only": round(float(sub["mae_price_only"].mean()), 3),
                         "neg_price_share": round(float(sub["neg_share"].mean()), 4),
                         "pv_share_of_load": round(pv_share, 4) if pv_share else None})
        exploratory["long_proxy_walk_forward_per_year"] = rows
        exploratory["long_proxy_note"] = (
            "single-design expanding-window WF (train from 2020-06), proxy tier only; "
            "tests whether proxy weather value grows with negative-price exposure as DE "
            "PV share grows. Overlaps the head-to-head window with a deeper train "
            "history, so 2024+ numbers differ slightly from the main run.")

    # replication gate (pre-registered): promote ONLY if op positive + DM-significant
    # + ratio qualitatively consistent with Japan (op/proxy clearly > 1).
    clean = (pooled_op["mean_improvement_eur_mwh"] > 0
             and pooled_op["dm_hln_l3"]["p_hln"] < 0.05
             and pooled_head["dm_hln_l3"]["p_hln"] < 0.05
             and ratio > 1.5)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    summary = {
        "phase": "45",
        "title": "EPEX-DE multi-year two-tier external replication (proxy vs operational)",
        "market": "EPEX-DE/LU (Germany), day-ahead hourly",
        "probe": probe,
        "fetch_params": {
            "smard": {"endpoint": SMARD, "ids": SMARD_IDS, "region": "DE",
                      "resolution": "hour", "start": SMARD_START.isoformat()},
            "era5": {"endpoint": OPEN_METEO_ERA5, "point": list(DE_COORD),
                     "vars": list(WX_VARS), "start": SMARD_START.isoformat(),
                     "end": era5_end.isoformat()},
            "operational": {"endpoint": OPEN_METEO_HISTFC, "model": HISTFC_MODEL,
                            "vars": [f"{v}_previous_day1" for v in WX_VARS],
                            "point": list(DE_COORD), "start": HISTFC_START.isoformat(),
                            "end": histfc_end.isoformat()},
        },
        "protocol": {
            "learner": "XGBRegressor(n_estimators=300, max_depth=6, lr=0.05, subsample=0.8, "
                       "colsample_bytree=0.8, seed=2026) [phase38]",
            "walk_forward": f"daily-rolling 1-day-ahead, expanding train window, "
                            f"burn-in {BURN_IN_DAYS} days, skip test days < {MIN_TEST_HOURS} h "
                            f"[phase38]",
            "features_base": "hour, dow, doy, price_lag24/48/168 [phase38]",
            "proxy_tier": "ERA5 5 channels lagged 24 h (JEPX paper proxy convention; NOTE "
                          "phase38's script actually fed delivery-hour realised ERA5 -- kept "
                          "here as robustness cell era5_realized_p38)",
            "operational_tier": "icon_seamless previous_day1 5 channels at delivery hour",
            "dm": "HLN small-sample DM with Newey-West (Bartlett) HAC on daily MAE diffs, "
                  "h=1, L=3 (phase23.5 convention) + automatic-lag robustness",
        },
        "coverage": {
            "panel_hourly_rows": int(panel.height),
            "head_to_head_test_days": int(daily.height),
            "head_to_head_first_day": daily["d"].min().isoformat(),
            "head_to_head_last_day": daily["d"].max().isoformat(),
        },
        "pooled": {
            "mae_price_only_eur_mwh": round(float(daily["mae_price_only"].mean()), 3),
            "improvement_proxy_eur_mwh": round(pooled_proxy["mean_improvement_eur_mwh"], 4),
            "improvement_op_eur_mwh": round(pooled_op["mean_improvement_eur_mwh"], 4),
            "ratio_op_over_proxy": round(ratio, 3),
            "dm_proxy_vs_price": pooled_proxy["dm_hln_l3"] | {
                "p_lauto": pooled_proxy["dm_hln_lauto"]["p_hln"],
                "lag_auto": pooled_proxy["dm_hln_lauto"]["nw_lag"]},
            "dm_op_vs_price": pooled_op["dm_hln_l3"] | {
                "p_lauto": pooled_op["dm_hln_lauto"]["p_hln"],
                "lag_auto": pooled_op["dm_hln_lauto"]["nw_lag"]},
            "dm_op_vs_proxy": pooled_head["dm_hln_l3"] | {
                "p_lauto": pooled_head["dm_hln_lauto"]["p_hln"],
                "lag_auto": pooled_head["dm_hln_lauto"]["nw_lag"]},
            "robustness_era5_realized_p38_improvement": round(
                pooled_p38["mean_improvement_eur_mwh"], 4),
        },
        "per_year": per_year,
        "channel_ablation": ablation,
        "exploratory_negative_price_gradient": exploratory,
        "replication_gate": {
            "criteria": "op improvement > 0 AND DM(op vs price) p<0.05 AND "
                        "DM(op vs proxy) p<0.05 AND ratio op/proxy > 1.5",
            "decision": "CLEAN -> promote to External-validity results subsection"
                        if clean else "NOT CLEAN -> keep exploratory/appendix",
        },
        "metadata": {
            "script": Path(__file__).name,
            "git_sha": git_sha,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "per_day_parquet": "data/runs/phase45_epex_de_tiered_per_day.parquet",
            "long_proxy_parquet": None if args.skip_long
            else "data/runs/phase45_epex_de_long_proxy_per_day.parquet",
        },
    }
    out = run_dir / "phase45_epex_de_tiered.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("POOLED: price-only MAE %.2f | +proxy %.3f | +op %.3f EUR/MWh | ratio %.2fx | "
             "DM(op vs price) p=%.2g | DM(op vs proxy) p=%.2g | gate=%s",
             summary["pooled"]["mae_price_only_eur_mwh"],
             pooled_proxy["mean_improvement_eur_mwh"], pooled_op["mean_improvement_eur_mwh"],
             ratio, pooled_op["dm_hln_l3"]["p_hln"], pooled_head["dm_hln_l3"]["p_hln"],
             summary["replication_gate"]["decision"])
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
