r"""Phase 51: forecast-vintage ladder with dissemination-correct admissibility.

MOTIVATION (major-revision load-bearing experiment). An external review proved
from the JMBSC official spec that JMA MSM-GPV files are disseminated at
init + 2h30m. The paper's "exact-issuance" tier (D-1 00 UTC run) is therefore
disseminated ~11:30 JST on D-1 -- 1.5 h AFTER the JEPX D-1 10:00 JST gate
closure. That tier is the FIRST POST-GATE vintage, not a pre-gate one.
Admissibility arithmetic: availability = init + 2.5 h must be <= D-1 10:00 JST
(= D-1 01:00 UTC)  =>  init <= D-2 22:30 UTC.

CORRECTED LADDER (Open-Meteo Historical-Forecast / Previous-Runs API, jma_msm):
``*_previous_day2`` = the value predicted ~48 h before each valid time
=> for EVERY delivery slot of day D the run is disseminated >= 8 h before the
gate (worst case: slot D 23:30 JST, init <= D-2 23:30 JST, disseminated
<= D-1 02:00 JST, gate D-1 10:00 JST). ``previous_day1`` is the first
post-gate class (slots after ~10:00 JST may use a run disseminated after the
gate). The six same-sample cells span the full vintage ladder:

  price_only < era5(lag-48 stale-obs proxy) < prev2 (PRE-GATE, new headline)
      < prev1 (first post-gate) < histfc (reconstruction) < oracle (realised).

Steps (gate-sequenced; each stops the pipeline on failure):
  probe      : previous_day2 coverage probe (9 area points, per-variable
               first full day, latest day, gap scan) + empirical init-mapping
               check + PRE-COMMITTED gate-1 criteria written to the JSON
               BEFORE any sweep result exists.  GATE: >= 600 usable days.
  refresh    : freeze record + caches: JEPX spot_2026, ERA5-2026 refresh
               (9 points), prev2 (4 channels x 9 points, chunked by year),
               prev1 cache extension, histfc reconstruction span from
               2024-03-02.
  headtohead : 6-cell same-sample daily walk-forward, STEP=1 day (no
               thinning), 9 areas, expanding window, val 14 d,
               XGB(300, depth 4, lr 0.05, seed 2026, hist).  First test day
               2024-08-01, last = min(freeze-2d, ERA5 end).  Checkpoint/area.
  aggregate  : per-area + pooled improvements vs price_only; paired HLN-NW DM
               (L=3 AND L=9); per-day paired circular block bootstrap
               (block 7, 10,000 draws) 95% CIs (primary); cross-area Spearman
               vs PV/peak; headline quantities (pre-gate multiple, gate-timing
               value); committed-protocol bridge subset (test >= 2024-09-15);
               gate-1 verdict against the pre-committed criteria; per-slot
               parquet for the decision-model rework.

CLI:
  cd <repository root>
  uv run --extra research python scripts/17_composite_ladder_superseded.py \
      --step probe|refresh|headtohead|aggregate|all [--areas kyushu,tokyo] [--fresh]
"""
from __future__ import annotations

import argparse
import hashlib
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
log = logging.getLogger("phase51")
JST = ZoneInfo("Asia/Tokyo")

# ------------------------------------------------------------------ pinned params
OPEN_METEO_HISTFC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV_MODEL = "jma_msm"
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

# walk-forward protocol (phase49 protocol, densified: STEP=1, earlier start)
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, TEST_WINDOW, STEP = timedelta(days=14), timedelta(days=1), timedelta(days=1)
MAX_FOLDS = 800
SEED = 2026

PANEL_START = date(2024, 3, 2)      # prev2 first full day (2024-02-17) + 672-slot lags
FETCH_START = date(2024, 2, 16)     # phase49-probed prev1 shortwave archive start
ERA5_DELAY_DAYS = 8
WEATHER_DELAY_DAYS = 2
BRIDGE_FIRST_TEST_DAY = date(2024, 9, 15)   # committed phase35/41 protocol first test day
PROBE_GATE_MIN_DAYS = 600
N_BOOT, BLOCK_LEN = 10_000, 7

CELLS_ORDER = ("price_only", "era5", "prev2", "prev1", "histfc", "oracle")
TIERS = ("era5", "prev2", "prev1", "histfc", "oracle")   # vs price_only

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
CKPT = RUN / "phase51_checkpoints"
OUT_JSON = RUN / "phase51_vintage_ladder.json"
PER_SLOT_PARQUET = RUN / "phase51_vintage_ladder_per_slot.parquet"

GATE1_CRITERIA_VERBATIM = (
    "PASS if and only if BOTH hold on the pooled 9-area panel (all test days, "
    "all delivery slots): (a) the pooled improvement difference "
    "imp(prev2) - imp(era5), computed as the mean over test days of "
    "[daily pooled MAE(era5 cell) - daily pooled MAE(prev2 cell)], has a "
    "per-day paired circular block bootstrap (block length 7 days, 10,000 "
    "draws, seed 2026) 95% CI lower bound > 0; AND (b) the pooled "
    "HLN-corrected Newey-West Diebold-Mariano test of the prev2 cell vs the "
    "era5 cell on the per-day pooled MAE differences has p < 0.05 at BOTH "
    "L=3 and L=9. Criteria recorded before any phase51 sweep result exists."
)

DISSEMINATION_ACCOUNTING = {
    "gate": "JEPX day-ahead gate closure D-1 10:00 JST = D-1 01:00 UTC",
    "msm_dissemination_rule": ("JMA MSM-GPV disseminated at init + 2h30m "
                               "(JMBSC official spec; external-review finding)"),
    "admissibility_arithmetic": ("availability = init + 2.5 h must be <= D-1 "
                                 "10:00 JST (= D-1 01:00 UTC) => init <= D-2 "
                                 "22:30 UTC"),
    "tiers": {
        "gpv_d1_00utc_paper_tier": {
            "init": "D-1 00:00 UTC", "disseminated": "D-1 02:30 UTC = D-1 11:30 JST",
            "margin_vs_gate": "-1h30m (AFTER the gate)",
            "verdict": "FIRST POST-GATE vintage; must be relabelled, not 'exact-issuance pre-gate'"},
        "prev2_pre_gate": {
            "init": ("run current 48 h before valid time; worst delivery slot "
                     "D 23:30 JST => init <= D-2 23:30 JST (14:30 UTC)"),
            "disseminated": "<= D-1 02:00 JST (17:00 UTC D-2) under the least favourable reading",
            "margin_vs_gate": ">= +8h00m for EVERY delivery slot (earliest slot: >= +31h30m)",
            "verdict": "SAFELY PRE-GATE for all 48 slots -- corrected conservative tier (new headline)"},
        "prev1_first_post_gate": {
            "init": "run current 24 h before valid time; worst slot D 23:30 JST => init <= D-1 23:30 JST",
            "disseminated": "slots 00:00-10:00 JST pre-gate; slots ~12:30-23:30 JST post-gate (up to ~16 h after)",
            "margin_vs_gate": "mixed sign across slots",
            "verdict": "first post-gate class (phase49's 'previous-day-run' tier)"},
        "histfc_reconstruction": {
            "init": "blend of best available runs per timestamp (lead ~0-6 h); no pinned issuance",
            "disseminated": "mostly after the gate",
            "margin_vs_gate": "undefined (reconstruction)",
            "verdict": "reconstruction reference, not admissible"},
        "era5_lag48_proxy": {
            "init": "not a forecast: realised reanalysis at t - 24 h",
            "disseminated": ("for slots after 10:00 JST the lagged observation itself "
                             "postdates the gate; established paper stale-observation "
                             "baseline convention, not a certified pre-gate feed"),
            "margin_vs_gate": "convention tier",
            "verdict": "stale-observation baseline (paper convention since phase35)"},
        "oracle_realised": {
            "init": "none (ex-post realised ERA5 at delivery time)",
            "disseminated": "never available ex ante",
            "margin_vs_gate": "inadmissible",
            "verdict": "information-ceiling upper bound"},
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(client: httpx.Client, url: str, params: dict | None = None,
              retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            wait = 3.0 * (attempt + 1)
            log.warning("retry %d for %s (%s); sleeping %.1fs", attempt + 1, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_summary() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"phase": "51",
            "title": "forecast-vintage ladder with dissemination-correct admissibility",
            "dissemination_accounting": DISSEMINATION_ACCOUNTING,
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
def _day_covered(client: httpx.Client, hourly: str, d: date,
                 lat: float, lon: float) -> bool:
    js = _get_json(client, OPEN_METEO_HISTFC, {
        "latitude": lat, "longitude": lon, "start_date": d.isoformat(),
        "end_date": d.isoformat(), "hourly": hourly, "models": PREV_MODEL,
        "timezone": "Asia/Tokyo"})
    h = js.get("hourly", {})
    fracs = [sum(x is not None for x in v) / max(len(v), 1)
             for k, v in h.items() if k != "time"]
    time.sleep(0.35)
    return bool(fracs) and min(fracs) >= 0.99


def _bisect_first(client: httpx.Client, hourly: str,
                  lo: date, hi: date, lat: float, lon: float) -> str | None:
    if not _day_covered(client, hourly, hi, lat, lon):
        return None
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if _day_covered(client, hourly, mid, lat, lon):
            hi = mid
        else:
            lo = mid + timedelta(days=1)
    return lo.isoformat()


def _fetch_prev_area_year(client: httpx.Client, area: str, lat: float, lon: float,
                          year: int, end_day: date, n_prev: int) -> pl.DataFrame | None:
    """One calendar year of previous_day{n} JMA-MSM forecast for one point; cached.

    Cache naming follows phase49 (``open_meteo_prev1_...``) with the prev2
    suffix for the new tier. Refetches (and overwrites with a superset) when
    the cached span does not cover [needed_start, needed_end].
    """
    start = max(date(year, 1, 1), FETCH_START)
    end = min(date(year, 12, 31), end_day)
    if start > end:
        return None
    suffix = f"pd{n_prev}"
    cache = (RAW / "open_meteo_historical_forecast" /
             f"open_meteo_prev{n_prev}_{PREV_MODEL}_{area}_{lat:.4f}_{lon:.4f}_{year}.parquet")
    if cache.exists():
        df = pl.read_parquet(cache)
        start_ok = df["datetime_jst"].min().date() <= start
        end_ok = df["datetime_jst"].max().date() >= end - timedelta(days=1)
        cols_ok = all(f"{API_TO_CANON[v]}_{suffix}" in df.columns for v in API_VARS)
        if start_ok and end_ok and cols_ok:
            return df
        log.info("prev%d cache %s insufficient (start_ok=%s end_ok=%s) -- refetching",
                 n_prev, cache.name, start_ok, end_ok)
    js = _get_json(client, OPEN_METEO_HISTFC, {
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(f"{v}_previous_day{n_prev}" for v in API_VARS),
        "models": PREV_MODEL, "timezone": "Asia/Tokyo"})
    h = js["hourly"]
    df = pl.DataFrame(
        {
            "datetime_jst": [datetime.fromisoformat(t).replace(tzinfo=JST)
                             for t in h["time"]],
            **{f"{API_TO_CANON[v]}_{suffix}": [float(x) if x is not None else None
                                               for x in h[f"{v}_previous_day{n_prev}"]]
               for v in API_VARS},
        },
        schema={"datetime_jst": pl.Datetime(time_unit="us", time_zone="Asia/Tokyo"),
                **{f"{API_TO_CANON[v]}_{suffix}": pl.Float64 for v in API_VARS}},
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    log.info("prev%d %s %d: %d hourly rows fetched (%s..%s)", n_prev, area, year,
             df.height, start, end)
    time.sleep(1.0)
    return df


def _load_prev(area: str, lat: float, lon: float, end_day: date,
               n_prev: int) -> pl.DataFrame:
    with httpx.Client(timeout=180.0, headers=UA) as c:
        frames = [_fetch_prev_area_year(c, area, lat, lon, y, end_day, n_prev)
                  for y in range(FETCH_START.year, end_day.year + 1)]
    return (pl.concat([f for f in frames if f is not None])
            .unique("datetime_jst").sort("datetime_jst"))


def _gap_scan_prev2(df: pl.DataFrame, first_day: date, last_day: date) -> dict:
    """Per-day completeness scan of a prev2 frame between first and last day."""
    cols = [f"{API_TO_CANON[v]}_pd2" for v in API_VARS]
    daily = (df.filter(pl.col("datetime_jst").dt.date().is_between(first_day, last_day))
             .group_by(pl.col("datetime_jst").dt.date().alias("d"))
             .agg(*[pl.col(c).null_count().alias(c) for c in cols],
                  pl.len().alias("n"))
             .sort("d"))
    bad = daily.filter((pl.col("n") < 24) | (pl.max_horizontal(cols) > 0))
    expected = (last_day - first_day).days + 1
    return {"days_expected": expected,
            "days_present": int(daily.height),
            "days_missing_entirely": expected - int(daily.height),
            "days_with_partial_nulls": int(bad.height),
            "gap_days": [d.isoformat() for d in bad["d"].to_list()][:50]}


def _init_mapping_check(client: httpx.Client, lat: float, lon: float,
                        sample_days: list[date]) -> dict:
    """Empirically pin the previous_dayN semantics on sample days.

    Fetches current / previous_day1 / previous_day2 for temperature and
    shortwave over [d-1, d] and compares (i) prev2(t) vs prev1(t-24h) --
    same init epoch, different valid time, so exact equality is NOT expected
    under the 'run current N*24h before valid time' semantics; (ii) prev2(t)
    vs prev1(t) same valid time -- MUST differ if the archive stores
    genuinely older runs; (iii) divergence-vs-lead ordering
    |current - prevN| increasing in N.
    """
    check_vars = ("temperature_2m", "shortwave_radiation")
    out: dict = {"sample_days": [d.isoformat() for d in sample_days], "per_day": {}}
    agg: dict[str, list[float]] = {}
    for d in sample_days:
        hourly = ",".join(f"{v}{s}" for v in check_vars
                          for s in ("", "_previous_day1", "_previous_day2"))
        js = _get_json(client, OPEN_METEO_HISTFC, {
            "latitude": lat, "longitude": lon,
            "start_date": (d - timedelta(days=1)).isoformat(),
            "end_date": d.isoformat(), "hourly": hourly,
            "models": PREV_MODEL, "timezone": "Asia/Tokyo"})
        h = js["hourly"]
        rec: dict = {}
        for v in check_vars:
            cur = np.array([np.nan if x is None else x for x in h[v]], dtype=float)
            p1 = np.array([np.nan if x is None else x
                           for x in h[f"{v}_previous_day1"]], dtype=float)
            p2 = np.array([np.nan if x is None else x
                           for x in h[f"{v}_previous_day2"]], dtype=float)
            # day d = indices 24..47; prev1 at t-24h = indices 0..23 of p1
            sl_d = slice(24, 48)
            pairs = {
                "prev2_t_vs_prev1_tminus24": (p2[sl_d], p1[0:24]),
                "prev2_t_vs_prev1_t": (p2[sl_d], p1[sl_d]),
                "prev1_t_vs_current_t": (p1[sl_d], cur[sl_d]),
                "prev2_t_vs_current_t": (p2[sl_d], cur[sl_d]),
            }
            vrec = {}
            for name, (a, b) in pairs.items():
                m = ~(np.isnan(a) | np.isnan(b))
                if m.sum() == 0:
                    vrec[name] = None
                    continue
                diff = a[m] - b[m]
                vrec[name] = {
                    "mae": round(float(np.mean(np.abs(diff))), 4),
                    "exact_match_frac": round(float(np.mean(np.abs(diff) < 1e-6)), 3),
                    "n": int(m.sum())}
                agg.setdefault(f"{v}:{name}:mae", []).append(vrec[name]["mae"])
                agg.setdefault(f"{v}:{name}:exact", []).append(
                    vrec[name]["exact_match_frac"])
            rec[v] = vrec
        out["per_day"][d.isoformat()] = rec
        time.sleep(0.5)
    out["aggregate_mean"] = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    t_keys = ("temperature_2m:prev1_t_vs_current_t:mae",
              "temperature_2m:prev2_t_vs_current_t:mae",
              "temperature_2m:prev2_t_vs_prev1_t:mae")
    mono = all(k in out["aggregate_mean"] for k in t_keys) and (
        out["aggregate_mean"][t_keys[1]] >= out["aggregate_mean"][t_keys[0]]
        and out["aggregate_mean"][t_keys[2]] > 0.0)
    out["interpretation"] = (
        "prev2(t) and prev1(t-24h) share the same init epoch (run current "
        "~48 h before t) but predict DIFFERENT valid times, so exact equality "
        "is impossible by construction; the recorded MAE reflects 24-h weather "
        "evolution within one run. The load-bearing empirical facts are: "
        "(i) prev2(t) != prev1(t) at the same valid time (exact_match_frac "
        "near 0, MAE > 0) => the archive stores genuinely distinct, older "
        "runs; (ii) divergence from the near-realtime 'current' series "
        "increases with vintage age (|current-prev2| >= |current-prev1| on "
        "average) => previous_dayN = run current ~N*24 h before valid time, "
        "consistent with the Open-Meteo Previous-Runs documentation. "
        f"Monotone-divergence check on these samples: {'PASS' if mono else 'FAIL'}.")
    out["monotone_divergence_pass"] = bool(mono)
    return out


def step_probe(summary: dict) -> bool:
    """Gate: jma_msm previous_day2 coverage over the 9 area points."""
    today = datetime.now(UTC).date()
    kyushu = AREA_COORDS[AreaCode.KYUSHU][:2]
    probe: dict = {
        "endpoint": OPEN_METEO_HISTFC,
        "model": PREV_MODEL,
        "variables_api": [f"{v}_previous_day2" for v in API_VARS],
        "points": {a.value: list(AREA_COORDS[a][:2]) for a in AREA_COORDS},
    }
    with httpx.Client(timeout=60.0, headers=UA) as c:
        per_var: dict = {}
        for v in API_VARS:
            per_var[v] = _bisect_first(c, f"{v}_previous_day2",
                                       date(2024, 1, 1), date(2024, 6, 1), *kyushu)
        probe["per_variable_first_full_day_prev2_kyushu"] = per_var
        all4_prev2 = ",".join(f"{v}_previous_day2" for v in API_VARS)
        firsts: dict[str, str | None] = {}
        for a, (lat, lon, _) in AREA_COORDS.items():
            firsts[a.value] = _bisect_first(c, all4_prev2, date(2024, 1, 1),
                                            date(2024, 6, 1), lat, lon)
        probe["all4_first_full_day_prev2_by_area"] = firsts
        probe["all4_first_full_day_histfc_plain_kyushu"] = _bisect_first(
            c, ",".join(API_VARS), date(2024, 1, 1), date(2024, 6, 1), *kyushu)
        latest = None
        for back in range(1, 9):
            d = today - timedelta(days=back)
            if _day_covered(c, all4_prev2, d, *kyushu):
                latest = d.isoformat()
                break
        probe["latest_full_day_prev2_kyushu"] = latest
        probe["init_mapping_check"] = _init_mapping_check(
            c, *kyushu, [date(2024, 6, 15), date(2025, 1, 15), date(2025, 10, 15)])
    ok = all(firsts.values()) and latest is not None
    first_common = max(v for v in firsts.values() if v) if ok else None
    cov_days = ((date.fromisoformat(latest) - date.fromisoformat(first_common)).days + 1
                if ok else 0)
    # gap scan over the full window (Kyushu; caches double as the fetch)
    if ok:
        prev2_ky = _load_prev(AreaCode.KYUSHU.value, *kyushu,
                              date.fromisoformat(latest), 2)
        probe["gap_scan_kyushu_prev2"] = _gap_scan_prev2(
            prev2_ky, date.fromisoformat(first_common), date.fromisoformat(latest))
        cov_days -= probe["gap_scan_kyushu_prev2"]["days_with_partial_nulls"]
        cov_days -= probe["gap_scan_kyushu_prev2"]["days_missing_entirely"]
    probe["tier_definition_prev2"] = (
        "previous_day2 on the Historical-Forecast API = for each delivery "
        "hour, the value from the JMA-MSM run current ~48 h before the valid "
        "time. With MSM-GPV dissemination at init + 2h30m (JMBSC spec), the "
        "run for the WORST delivery slot of day D (23:30 JST) is available "
        "no later than D-1 02:00 JST, i.e. >= 8 h before the D-1 10:00 JST "
        "JEPX gate closure -- SAFELY PRE-GATE for every slot.")
    probe["gate"] = {
        "criterion": f">= {PROBE_GATE_MIN_DAYS} days of usable previous_day2 "
                     "coverage over the 9 area points, all 4 paper variables",
        "coverage_days_usable": cov_days,
        "decision": "PASS" if (ok and cov_days >= PROBE_GATE_MIN_DAYS) else
                    "STOP -- report and do not improvise an alternative design",
    }
    summary["probe"] = probe
    # PRE-COMMITTED gate-1 criteria: recorded here, BEFORE any sweep result.
    summary["gate1_criteria"] = {
        "criteria_verbatim": GATE1_CRITERIA_VERBATIM,
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorded_before_any_sweep_result": True,
    }
    _save_summary(summary)
    log.info("PROBE gate: %s (usable coverage %d days, first common %s, latest %s)",
             probe["gate"]["decision"], cov_days, first_common, latest)
    return ok and cov_days >= PROBE_GATE_MIN_DAYS


# ================================================================ STEP 2: refresh
def step_refresh(summary: dict) -> None:
    today = datetime.now(UTC).date()
    weather_end = today - timedelta(days=WEATHER_DELAY_DAYS)
    latest_probe = summary.get("probe", {}).get("latest_full_day_prev2_kyushu")
    if latest_probe:
        weather_end = min(weather_end, date.fromisoformat(latest_probe))
    era5_end = today - timedelta(days=ERA5_DELAY_DAYS)
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    freeze: dict = {"data_freeze_date_utc": today.isoformat(),
                    "weather_end_date": weather_end.isoformat(),
                    "era5_end_date": era5_end.isoformat(), "files": {}}

    spot_path = fetch_jepx_spot_year(2026, cache_dir=RAW, registry=reg, force=True)
    spot_df = parse_jepx_spot_csv(spot_path)
    freeze["files"]["jepx_spot_2026"] = {
        "path": "data/raw/jepx/spot_2026.csv",
        "sha256": _sha256(spot_path), "rows": int(spot_df.height),
        "last_slot_jst": spot_df["datetime_jst"].max().isoformat()}
    log.info("spot_2026 refreshed: last slot %s", spot_df["datetime_jst"].max())

    era5_files = {}
    for area, (lat, lon, _) in AREA_COORDS.items():
        p = fetch_open_meteo_year(2026, cache_dir=RAW, latitude=lat, longitude=lon,
                                  registry=reg, force=True,
                                  end_date_override=era5_end.isoformat())
        era5_files[area.value] = {"sha256": _sha256(p), "end_date": era5_end.isoformat()}
        time.sleep(0.5)
    freeze["files"]["era5_2026_9points"] = era5_files

    for n_prev in (2, 1):
        for area, (lat, lon, _) in AREA_COORDS.items():
            _load_prev(area.value, lat, lon, weather_end, n_prev)
    hf_dir = RAW / "open_meteo_historical_forecast"
    for n_prev in (2, 1):
        freeze["files"][f"open_meteo_prev{n_prev}_caches"] = {
            f.name: _sha256(f)
            for f in sorted(hf_dir.glob(f"open_meteo_prev{n_prev}_*.parquet"))}

    span = [PANEL_START + timedelta(days=i)
            for i in range((weather_end - PANEL_START).days + 1)]
    histfc_files = {}
    for area, (lat, lon, _) in AREA_COORDS.items():
        load_open_meteo_historical_forecast_archive(
            span, cache_dir=RAW, latitude=lat, longitude=lon, registry=reg)
        time.sleep(1.0)
    for f in sorted(hf_dir.glob(
            f"open_meteo_histfc_jma_msm_*_{PANEL_START:%Y%m%d}_{weather_end:%Y%m%d}.json")):
        histfc_files[f.name] = _sha256(f)
    freeze["files"]["open_meteo_histfc_panel_span"] = histfc_files

    summary["data_freeze_date_utc"] = today.isoformat()
    summary["freeze"] = freeze
    _save_summary(summary)


# ============================================================ STEP 3: head-to-head
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """phase35/49 design: price lags 48/96/336/672, calendar; 6 vintage cells."""
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
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


def _midday_corr(df: pl.DataFrame, a: str, b: str) -> float:
    midday = df.filter(pl.col("datetime_jst").dt.hour().is_between(10, 14))
    return float(np.corrcoef(midday[a].to_numpy(), midday[b].to_numpy())[0, 1])


def _run_area(area: AreaCode, lat: float, lon: float, label: str,
              weather_end: date, era5_end: date) -> None:
    ckpt = CKPT / f"{area.value}.parquet"
    meta_f = CKPT / f"{area.value}_meta.json"
    if ckpt.exists() and meta_f.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    log.info("=== %s (%s) ===", area.value, label)
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    price = load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW, areas=[area],
                           registry=reg).select("datetime_jst", "area", "price_yen_kwh")
    era5 = load_open_meteo_weather(years=[2024, 2025, 2026], cache_dir=RAW,
                                   latitude=lat, longitude=lon, registry=reg)
    span = [PANEL_START + timedelta(days=i)
            for i in range((weather_end - PANEL_START).days + 1)]
    histfc = load_open_meteo_historical_forecast_archive(
        span, cache_dir=RAW, latitude=lat, longitude=lon, registry=reg,
    ).select("datetime_jst", *DEFAULT_WEATHER_COLS).rename(
        {c: f"{c}_fc" for c in DEFAULT_WEATHER_COLS})
    prev1 = _load_prev(area.value, lat, lon, weather_end, 1)
    prev2 = _load_prev(area.value, lat, lon, weather_end, 2)
    df = attach_hourly_weather(price.sort("datetime_jst"), era5)
    df = attach_hourly_features(df, histfc,
                                feature_cols=[f"{c}_fc" for c in DEFAULT_WEATHER_COLS])
    df = attach_hourly_features(df, prev1,
                                feature_cols=[f"{c}_pd1" for c in DEFAULT_WEATHER_COLS])
    df = attach_hourly_features(df, prev2,
                                feature_cols=[f"{c}_pd2" for c in DEFAULT_WEATHER_COLS])
    df, cells = _design(df)
    need = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
          .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, era5_end))
          .sort("datetime_jst"))
    log.info("%s panel: %d slots %s..%s", area.value, df.height,
             df["datetime_jst"].min(), df["datetime_jst"].max())

    sw = "shortwave_radiation_wm2"
    corrs = {
        "midday_sw_corr_prev2_vs_prev1": round(_midday_corr(df, f"{sw}_pd2", f"{sw}_pd1"), 4),
        "midday_sw_corr_prev2_vs_histfc": round(_midday_corr(df, f"{sw}_pd2", f"{sw}_fc"), 4),
        "midday_sw_corr_prev2_vs_realised": round(_midday_corr(df, f"{sw}_pd2", sw), 4),
        "midday_sw_corr_prev1_vs_realised": round(_midday_corr(df, f"{sw}_pd1", sw), 4),
        "midday_sw_corr_histfc_vs_realised": round(_midday_corr(df, f"{sw}_fc", sw), 4),
    }
    pvpeak = pv_gw_by_area()[area.value] / peak_gw_by_area()[area.value]
    panel_last = df["datetime_jst"].max().date()
    rows: list[dict] = []
    n_folds_run = 0
    t0 = time.time()
    for k in range(MAX_FOLDS):
        test_day = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        if test_day > panel_last:
            break
        train = df.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = df.filter(pl.col("datetime_jst").dt.date() == test_day).sort("datetime_jst")
        if train.height < 100 or test.height == 0:
            continue
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        sw_p2 = test[f"{sw}_pd2"].to_numpy().astype(float)
        sw_l48 = test[f"{sw}_lag_48"].to_numpy().astype(float)
        sw_fc = test[f"{sw}_fc"].to_numpy().astype(float)
        for s in range(test.height):
            rows.append({"area": area.value, "datetime_jst": dts[s],
                         "test_day": test_day.isoformat(), "y_actual": y[s],
                         **{f"y_pred_{name}": float(preds[name][s]) for name in cells},
                         f"{sw}_pd2": sw_p2[s], f"{sw}_lag_48": sw_l48[s],
                         f"{sw}_fc": sw_fc[s], "pv_peak_ratio": pvpeak})
        n_folds_run += 1
        if n_folds_run % 50 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area.value, n_folds_run,
                     test_day, (time.time() - t0) / n_folds_run)
    out = pl.DataFrame(rows)
    CKPT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(ckpt)
    meta_f.write_text(json.dumps({
        "area": area.value, "n_folds": n_folds_run, "n_slots": int(out.height),
        "panel_first_slot": df["datetime_jst"].min().isoformat(),
        "panel_last_slot": df["datetime_jst"].max().isoformat(),
        "first_test_day": out["test_day"].min(), "last_test_day": out["test_day"].max(),
        **corrs, "pv_peak_ratio": round(pvpeak, 4),
        "runtime_s": round(time.time() - t0, 1)}, indent=2))
    log.info("%s done: %d folds, %d slots, %.0f s", area.value, n_folds_run,
             out.height, time.time() - t0)


def step_headtohead(summary: dict, areas: list[str] | None, fresh: bool) -> None:
    if fresh and CKPT.exists():
        for f in CKPT.glob("*"):
            f.unlink()
    freeze = summary.get("freeze", {})
    today = datetime.now(UTC).date()
    weather_end = (date.fromisoformat(freeze["weather_end_date"])
                   if "weather_end_date" in freeze
                   else today - timedelta(days=WEATHER_DELAY_DAYS))
    era5_end = (date.fromisoformat(freeze["era5_end_date"])
                if "era5_end_date" in freeze
                else today - timedelta(days=ERA5_DELAY_DAYS))
    wanted = ([AreaCode(a) for a in areas] if areas else list(AREA_COORDS))
    for a in wanted:
        lat, lon, label = AREA_COORDS[a]
        _run_area(a, lat, lon, label, weather_end, era5_end)


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


def _boot_means(series: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return series[idx].mean(axis=1)


def _ci(draws: np.ndarray) -> dict:
    return {"lo95": round(float(np.percentile(draws, 2.5)), 4),
            "hi95": round(float(np.percentile(draws, 97.5)), 4)}


def step_aggregate(summary: dict) -> None:
    frames = [pl.read_parquet(f) for f in sorted(CKPT.glob("*.parquet"))]
    if not frames:
        log.error("no head-to-head checkpoints found in %s", CKPT)
        return
    if "gate1_criteria" not in summary:
        msg = "gate1_criteria missing from JSON -- probe step must run first"
        raise RuntimeError(msg)
    panel = pl.concat(frames)
    panel.write_parquet(PER_SLOT_PARQUET)
    metas = {json.loads(f.read_text())["area"]: json.loads(f.read_text())
             for f in sorted(CKPT.glob("*_meta.json"))}
    panel = panel.with_columns(
        *[((pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs()
           - (pl.col("y_pred_price_only") - pl.col("y_actual")).abs()).alias(f"d_{c}")
          for c in TIERS])

    def _improvements(sub: pl.DataFrame) -> dict:
        return {c: round(float(-sub[f"d_{c}"].mean()), 4) for c in TIERS}

    per_area_rows = (panel.group_by("area").agg(
        pv_peak=pl.col("pv_peak_ratio").first(),
        **{f"imp_{c}": -pl.col(f"d_{c}").mean() for c in TIERS},
        n=pl.len()).sort("pv_peak", descending=True))
    per_day = _per_day_mae(panel)
    pooled_daily = _pooled_daily(per_day)
    test_days = pooled_daily["test_day"].to_list()
    n_days = len(test_days)

    # ---------------- paired circular block bootstrap (primary inference)
    idx = _bootstrap_indices(n_days)
    imp_series = {c: (pooled_daily["mae_price_only"] - pooled_daily[f"mae_{c}"]).to_numpy()
                  for c in TIERS}
    draws = {c: _boot_means(imp_series[c], idx) for c in TIERS}
    boot = {c: {"pooled_improvement_daily_mean": round(float(imp_series[c].mean()), 4),
                **_ci(draws[c])} for c in TIERS}
    d_gate = draws["prev2"] - draws["era5"]                    # gate quantity (a)
    d_timing = draws["prev1"] - draws["prev2"]                 # headline (b)
    ratio_draws = draws["prev2"] / draws["era5"]
    boot_derived = {
        "prev2_minus_era5": {"point": round(float((imp_series["prev2"]
                                                   - imp_series["era5"]).mean()), 4),
                             **_ci(d_gate)},
        "prev1_minus_prev2": {"point": round(float((imp_series["prev1"]
                                                    - imp_series["prev2"]).mean()), 4),
                              **_ci(d_timing)},
        "histfc_minus_prev1": {"point": round(float((imp_series["histfc"]
                                                     - imp_series["prev1"]).mean()), 4),
                               **_ci(draws["histfc"] - draws["prev1"])},
        "pre_gate_value_multiple_prev2_over_era5": {
            "point": round(float(imp_series["prev2"].mean()
                                 / imp_series["era5"].mean()), 3),
            **_ci(ratio_draws),
            "frac_draws_era5_le_0": round(float(np.mean(draws["era5"] <= 0)), 4),
            "note": "ratio CI unstable if era5 improvement draws cross 0; "
                    "the DIFFERENCE CI above is the primary gate quantity"},
        "n_boot": N_BOOT, "block_len_days": BLOCK_LEN, "seed": SEED,
    }

    # ---------------- DM tests, L=3 and L=9
    dm_pairs = [("era5", "price_only"), ("prev2", "price_only"), ("prev1", "price_only"),
                ("histfc", "price_only"), ("oracle", "price_only"),
                ("prev2", "era5"), ("prev1", "prev2"), ("histfc", "prev1"),
                ("prev2", "histfc")]
    dm = {f"{a}_vs_{b}": {f"nw_l{lg}": _dm_pair(per_day, a, b, lg) for lg in (3, 9)}
          for a, b in dm_pairs}

    # ---------------- cross-area concentration Spearman
    pv = per_area_rows["pv_peak"].to_numpy()
    spearman = {}
    for c in TIERS:
        sp = stats.spearmanr(pv, per_area_rows[f"imp_{c}"].to_numpy())
        spearman[c] = {"rho": round(float(sp.statistic), 3),
                       "p": round(float(sp.pvalue), 3)}

    # ---------------- committed-protocol bridge (test days >= 2024-09-15)
    bridge_days = [i for i, d in enumerate(test_days)
                   if d >= BRIDGE_FIRST_TEST_DAY.isoformat()]
    b0 = bridge_days[0]
    idx_b = _bootstrap_indices(len(bridge_days))
    imp_b = {c: imp_series[c][b0:] for c in TIERS}
    draws_b = {c: _boot_means(imp_b[c], idx_b) for c in TIERS}
    bridge_panel = panel.filter(pl.col("test_day") >= BRIDGE_FIRST_TEST_DAY.isoformat())
    per_day_b = _per_day_mae(bridge_panel)
    bridge = {
        "definition": ("subset with test_day >= 2024-09-15 = first test day reachable "
                       "under the committed phase35/41 protocol (initial train end "
                       "2024-09-01 + 14 d val); bridges phase51's densified full-window "
                       "protocol to the committed-window record"),
        "n_test_days": len(bridge_days), "n_obs": int(bridge_panel.height),
        "pooled_improvement_yen_kwh": _improvements(bridge_panel),
        "bootstrap": {c: {"pooled_improvement_daily_mean":
                          round(float(imp_b[c].mean()), 4), **_ci(draws_b[c])}
                      for c in TIERS},
        "prev2_minus_era5": {"point": round(float((imp_b["prev2"]
                                                   - imp_b["era5"]).mean()), 4),
                             **_ci(draws_b["prev2"] - draws_b["era5"])},
        "dm_prev2_vs_era5": {f"nw_l{lg}": _dm_pair(per_day_b, "prev2", "era5", lg)[
            "pooled_across_areas"] for lg in (3, 9)},
    }

    # ---------------- gate-1 verdict against the pre-committed criteria
    gate_ci_lo = boot_derived["prev2_minus_era5"]["lo95"]
    p_l3 = dm["prev2_vs_era5"]["nw_l3"]["pooled_across_areas"]["p_hln"]
    p_l9 = dm["prev2_vs_era5"]["nw_l9"]["pooled_across_areas"]["p_hln"]
    dm_better = dm["prev2_vs_era5"]["nw_l3"]["pooled_across_areas"]["mean_diff"] < 0
    gate_pass = bool(gate_ci_lo > 0 and p_l3 < 0.05 and p_l9 < 0.05 and dm_better)
    summary["gate1_result"] = {
        "decision": "PASS" if gate_pass else "FAIL",
        "criteria_reference": "gate1_criteria (recorded before results)",
        "imp_prev2_minus_imp_era5_point": boot_derived["prev2_minus_era5"]["point"],
        "bootstrap_ci95_lower_bound": gate_ci_lo,
        "bootstrap_ci95_upper_bound": boot_derived["prev2_minus_era5"]["hi95"],
        "dm_pooled_p_hln_l3": p_l3, "dm_pooled_p_hln_l9": p_l9,
        "dm_direction_prev2_better": dm_better,
    }

    months = sorted({d[:7] for d in test_days})
    summary["headtohead"] = {
        "protocol": ("phase49 protocol densified: XGBRegressor(300, depth 4, lr 0.05, "
                     "seed 2026, hist), expanding-window walk-forward, initial train "
                     "end 2024-07-18, val 14 d, test 1 d, STEP=1 d (no thinning), "
                     "6 same-sample cells on identical folds; panel 2024-03-02..era5_end"),
        "cells": {
            "price_only": "price lags 48/96/336/672 + calendar",
            "era5": "+ ERA5 lag-48 stale-observation proxy (paper convention)",
            "prev2": "+ jma_msm previous_day2 = PRE-GATE conservative tier (NEW HEADLINE)",
            "prev1": "+ jma_msm previous_day1 = first post-gate class",
            "histfc": "+ jma_msm Historical-Forecast reconstruction",
            "oracle": "+ realised ERA5 at delivery time (ex-post information ceiling)"},
        "window": {"first_test_day": test_days[0], "last_test_day": test_days[-1],
                   "n_test_days": n_days,
                   "n_folds_per_area": {a: m["n_folds"] for a, m in metas.items()},
                   "n_obs": int(panel.height), "n_areas": int(per_area_rows.height),
                   "calendar_months_covered": months,
                   "n_calendar_months": len(months)},
        "pooled_improvement_yen_kwh_slot_mean": _improvements(panel),
        "pooled_improvement_yen_kwh_daily_mean_with_ci95": boot,
        "derived_quantities": boot_derived,
        "per_area": {r["area"]: {
            **{f"imp_{c}": round(float(r[f"imp_{c}"]), 4) for c in TIERS},
            "pv_peak": round(float(r["pv_peak"]), 3), "n": int(r["n"])}
            for r in per_area_rows.to_dicts()},
        "dm_hln_nw": dm,
        "cross_area_spearman_vs_pvpeak": spearman,
        "feature_space_ladder_midday_sw_corr": {
            a: {k: v for k, v in m.items() if k.startswith("midday_sw_corr")}
            for a, m in metas.items()},
        "committed_protocol_bridge_from_2024_09_15": bridge,
        "headline": {
            "pre_gate_value_multiple":
                boot_derived["pre_gate_value_multiple_prev2_over_era5"],
            "gate_timing_value_prev1_minus_prev2":
                boot_derived["prev1_minus_prev2"],
        },
        "runtime_s_per_area": {a: m["runtime_s"] for a, m in metas.items()},
    }
    _save_summary(summary)
    log.info("GATE 1: %s (prev2-era5 CI lo %.4f, DM p L3 %.2g L9 %.2g)",
             summary["gate1_result"]["decision"], gate_ci_lo, p_l3, p_l9)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["probe", "refresh", "headtohead", "aggregate", "all"])
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
            log.error("probe gate FAILED -- stopping (no alternative design improvised).")
            return 1
    if args.step in ("refresh", "all"):
        summary = _load_summary()
        step_refresh(summary)
        if args.step == "refresh":
            return 0
    if args.step in ("headtohead", "all"):
        summary = _load_summary()
        step_headtohead(summary, areas, args.fresh)
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
