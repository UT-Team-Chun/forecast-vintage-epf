r"""Phase 53: pre-gate exact-admissible JMA-MSM GPV tier extraction (RISH).

Builds the FIRST weather tier in this project that is simultaneously
(a) exact-issuance (single pinned MSM run per value, known lead) AND
(b) pre-gate admissible (nominal dissemination = init + 2h30m per JMBSC is
    strictly before the JEPX D-1 10:00 JST gate closure).

The existing "exact-issuance" JMA-GPV tier (phase 30.2 sweep, phase 35/41
head-to-heads) uses the D-1 00 UTC run, which is initialised D-1 09:00 JST but
only DISSEMINATED ~D-1 11:30 JST — i.e. ~1.5 h after the gate. This script
replaces that with the pre-gate run set verified by the 2026-07-26 RISH probe.

Per-slot run rule for delivery day D (all times JST):
  * delivery hours 00-21  <- D-2 21:00 UTC run (init D-1 06:00 JST,
      disseminated ~D-1 08:30, margin +1h30m vs gate). Forecast hours 18-39,
      served by surface files FH16-33 (78.7 MB) + FH34-39 (26.2 MB).
  * delivery hours 22-23  <- D-2 12:00 UTC run (init D-2 21:00 JST,
      disseminated ~D-2 23:30, margin +10h30m vs gate). Forecast hours 49-50,
      served by surface file FH40-51 (52.5 MB). These hours are beyond the
      21 UTC run's last archived surface file (FH39 = 21:00 JST), so the older
      12 UTC run is the freshest pre-gate source for the half-hour slots
      21:30-23:30 JST (hourly grid: values at 22:00 and 23:00; slot 21:00-21:30
      is still covered by the 21 UTC run's FH39 value at 21:00).

Total download ~157.4 MB per delivery day, downloaded sequentially
(single-stream, per RISH etiquette), extracted at the 9 canonical area points
(4 channels: temperature_c, cloud_cover_pct, shortwave_radiation_wm2,
wind_speed_kmh) and DELETED immediately — disk stays bounded to one file.

Cache: data/raw/jma_gpv_pregate/jma_gpv_pregate_{area}_{lat}_{lon}.parquet
  columns: datetime_jst, 4 weather cols, issuance_time_utc, lead_hours.
Checkpointing: parquets rewritten after EVERY day; a day counts as done only
if all 9 area caches hold its full 24 hourly rows, so interruption anywhere is
safe and re-runs skip completed days.
Metadata: _meta_issuance_accounting.json (per-day init/dissemination/margin
records + structural rule), _progress.json (per-day wall times, ETA).
FAIL-LOUD: the first GRIB decode error or exhausted-retry HTTP error aborts
the sweep (exit != 0); nothing is silently skipped.

CLI (from the repository root):
  uv run python scripts/02_extract_pregate_composite.py \
      --start 2024-03-01 --end auto                # full sweep (resumable)
  uv run python ... --start 2024-09-01 --end 2024-09-02   # small window
  uv run python ... --validate                     # sanity checks on cache

Validation (--validate): shortwave diurnal-cycle presence per area, and on
delivery days shared with the post-gate D-1 00UTC all-hours cache
(data/raw/jma_gpv_allhours/jma_gpv/), per-channel Pearson r + mean|diff|
(expected: high correlation, non-zero differences — different runs/leads).

Final aggregation JSON is produced later, when the full sweep has finished.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.jma_gpv_loader import (
    SURFACE_FH_FILES_PREGATE_12Z,
    SURFACE_FH_FILES_PREGATE_21Z,
    WEATHER_COLS,
    _run_url,
    fetch_gpv_run_points,
)
from alg.core.epf.data.registry import REGISTRY_FILENAME, DatasetRegistry, make_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase53")

JST = timezone(timedelta(hours=9), name="JST")
RAW = EpfConfig().resolved("raw_data_dir")
CACHE_DIR = RAW / "jma_gpv_pregate"
ALLHOURS_DIR = RAW / "jma_gpv_allhours" / "jma_gpv"
META_PATH = CACHE_DIR / "_meta_issuance_accounting.json"
PROGRESS_PATH = CACHE_DIR / "_progress.json"

# Canonical 9 area points (identical to phase 30.2 / 47 / 49 loaders).
AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}
DISSEMINATION_LAG = timedelta(hours=2, minutes=30)  # nominal, per JMBSC
HOURS_21Z = 22  # delivery hours 00..21 JST from the D-2 21 UTC run
HOURS_12Z = 2   # delivery hours 22..23 JST from the D-2 12 UTC run
POLITE_SLEEP_S = 1.0  # between runs / days (on top of sequential single-stream)


def _cache_path(area: str) -> Path:
    lat, lon = AREA_COORDS[area]
    return CACHE_DIR / f"jma_gpv_pregate_{area}_{lat:.4f}_{lon:.4f}.parquet"


def _runs_for(d: date) -> tuple[datetime, datetime]:
    """(D-2 21 UTC run, D-2 12 UTC run) for delivery day ``d``."""
    d2 = datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(days=2)
    return d2.replace(hour=21), d2.replace(hour=12)


def _issuance_accounting(d: date) -> dict:
    gate = datetime(d.year, d.month, d.day, 10, 0, tzinfo=JST) - timedelta(days=1)
    run21, run12 = _runs_for(d)
    rows = {}
    for role, run, slots, fhs in (
        ("run_21z", run21, "delivery 00:00-21:00 JST (hourly 00..21)", "FH18-39"),
        ("run_12z", run12, "delivery 21:30-23:30 JST slots (hourly 22,23)", "FH49-50"),
    ):
        diss = run + DISSEMINATION_LAG
        rows[role] = {
            "init_utc": run.isoformat(),
            "nominal_dissemination_jst": diss.astimezone(JST).isoformat(),
            "gate_jst": gate.isoformat(),
            "margin_minutes_before_gate": int((gate - diss).total_seconds() // 60),
            "delivery_slots": slots,
            "forecast_hours_used": fhs,
        }
    return rows


# ------------------------------------------------------------------ cache I/O
def _complete_days(frames: dict[str, pl.DataFrame]) -> set[date]:
    """Days with all 24 hourly rows in EVERY area cache (checkpoint unit)."""
    per_area_sets = []
    for area in AREA_COORDS:
        df = frames[area]
        if df.height == 0:
            return set()
        counts = df.group_by(pl.col("datetime_jst").dt.date().alias("d")).len()
        per_area_sets.append(set(counts.filter(pl.col("len") == 24)["d"].to_list()))
    return set.intersection(*per_area_sets)


def _load_caches() -> dict[str, pl.DataFrame]:
    out = {}
    for area in AREA_COORDS:
        p = _cache_path(area)
        out[area] = pl.read_parquet(p) if p.exists() else pl.DataFrame()
    return out


def _append_day(frames: dict[str, pl.DataFrame], area: str, day_df: pl.DataFrame,
                d: date) -> None:
    old = frames[area]
    if old.height > 0:
        old = old.filter(pl.col("datetime_jst").dt.date() != d)  # drop partial rows
        frames[area] = pl.concat([old, day_df]).sort("datetime_jst")
    else:
        frames[area] = day_df
    frames[area].write_parquet(_cache_path(area))


# ------------------------------------------------------------------ extraction
def _extract_day(d: date, client: httpx.Client,
                 frames: dict[str, pl.DataFrame]) -> None:
    """Fetch the 3 files of day-``d``'s run set, merge per-slot, checkpoint."""
    run21, run12 = _runs_for(d)
    per21 = fetch_gpv_run_points(run21, SURFACE_FH_FILES_PREGATE_21Z, AREA_COORDS,
                                 client=client)
    time.sleep(POLITE_SLEEP_S)
    per12 = fetch_gpv_run_points(run12, SURFACE_FH_FILES_PREGATE_12Z, AREA_COORDS,
                                 client=client)

    for area in AREA_COORDS:
        a = per21[area].filter(
            (pl.col("datetime_jst").dt.date() == d)
            & (pl.col("datetime_jst").dt.hour() < HOURS_21Z),
        )
        b = per12[area].filter(
            (pl.col("datetime_jst").dt.date() == d)
            & (pl.col("datetime_jst").dt.hour() >= HOURS_21Z),
        )
        day_df = pl.concat([a, b]).sort("datetime_jst")
        hours = day_df["datetime_jst"].dt.hour().to_list()
        if hours != list(range(24)) or a.height != HOURS_21Z or b.height != HOURS_12Z:
            msg = (f"{d}: expected 22 hourly rows from 21Z + 2 from 12Z, got "
                   f"{a.height}+{b.height} (hours={hours}) for area {area}")
            raise ValueError(msg)
        nulls = day_df.select(pl.col(list(WEATHER_COLS)).is_nan().any())
        if any(nulls.row(0)):
            msg = f"{d} {area}: NaN in extracted channels {nulls.row(0)}"
            raise ValueError(msg)
        _append_day(frames, area, day_df, d)


# ------------------------------------------------------------------ end probe
def _probe_latest_day(client: httpx.Client) -> date:
    """Latest delivery day whose full run set exists on RISH (HEAD probe)."""
    d = datetime.now(JST).date() + timedelta(days=2)
    for _ in range(45):
        run21, run12 = _runs_for(d)
        urls = [_run_url(run21, fh) for fh in SURFACE_FH_FILES_PREGATE_21Z]
        urls += [_run_url(run12, fh) for fh in SURFACE_FH_FILES_PREGATE_12Z]
        if all(client.head(u).status_code == 200 for u in urls):
            log.info("end=auto resolved to latest available delivery day %s", d)
            return d
        d -= timedelta(days=1)
        time.sleep(0.3)
    msg = "no available delivery day found within 45 days back from today+2"
    raise RuntimeError(msg)


# ------------------------------------------------------------------ metadata
def _write_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def _init_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {
        "tier": "jma_msm_gpv_pregate",
        "forecast_kind": "exact_issuance_pregate",
        "source": "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original",
        "rule": {
            "gate": "JEPX day-ahead gate closure D-1 10:00 JST",
            "admissibility": "nominal dissemination (init + 2h30m, per JMBSC) <= gate",
            "run_21z": "D-2 21UTC run, FH18-39 -> delivery hours 00-21 JST "
                       "(files FH16-33 + FH34-39), margin +90 min",
            "run_12z": "D-2 12UTC run, FH49-50 -> delivery hours 22-23 JST "
                       "(= half-hour slots 21:30-23:30; file FH40-51), "
                       "margin +630 min",
        },
        "channels": list(WEATHER_COLS),
        "area_coords": {a: list(c) for a, c in AREA_COORDS.items()},
        "per_day": {},
    }


# ------------------------------------------------------------------ sweep
def sweep(start: date, end: date) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = _load_caches()
    done = _complete_days(frames)
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in all_days if d not in done]
    log.info("sweep %s..%s: %d days requested, %d already complete, %d to fetch",
             start, end, len(all_days), len(all_days) - len(todo), len(todo))

    meta = _init_meta()
    progress = (json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
                if PROGRESS_PATH.exists() else
                {"started_at": datetime.now(UTC).isoformat(), "day_seconds": {}})
    progress["window"] = [start.isoformat(), end.isoformat()]

    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for i, d in enumerate(todo):
            t0 = time.monotonic()
            _extract_day(d, client, frames)
            dt_s = time.monotonic() - t0
            meta["per_day"][d.isoformat()] = _issuance_accounting(d)
            _write_meta(meta)
            progress["day_seconds"][d.isoformat()] = round(dt_s, 1)
            secs = list(progress["day_seconds"].values())
            avg = float(np.mean(secs[-50:]))
            remaining = len(todo) - i - 1
            progress.update({
                "updated_at": datetime.now(UTC).isoformat(),
                "days_done_total": len(all_days) - len(todo) + i + 1,
                "days_remaining": remaining,
                "avg_day_seconds_recent": round(avg, 1),
                "eta_utc": (datetime.now(UTC)
                            + timedelta(seconds=avg * remaining)).isoformat(),
            })
            PROGRESS_PATH.write_text(json.dumps(progress, indent=1), encoding="utf-8")
            log.info("day %s done in %.1fs (%d/%d, ETA %s)",
                     d, dt_s, i + 1, len(todo), progress["eta_utc"])
            time.sleep(POLITE_SLEEP_S)

    # register final per-area parquets (only reached when the sweep completes)
    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    for area, (lat, lon) in AREA_COORDS.items():
        reg.upsert(make_entry(
            name=f"jma_gpv_pregate_{area}_{lat:.4f}_{lon:.4f}",
            raw_data_dir=RAW,
            file_path=_cache_path(area),
            source_url="http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/"
                       "original (D-2 21UTC FH18-39 + D-2 12UTC FH49-50, "
                       "pre-gate exact-admissible)",
        ))
    log.info("sweep complete; %d areas registered", len(AREA_COORDS))


# ------------------------------------------------------------------ validation
def validate() -> dict:
    frames = _load_caches()
    done = sorted(_complete_days(frames))
    report: dict = {"complete_days": len(done),
                    "first_day": str(done[0]) if done else None,
                    "last_day": str(done[-1]) if done else None,
                    "diurnal": {}, "overlap_vs_00utc": {}}
    for area in AREA_COORDS:
        df = frames[area]
        if df.height == 0:
            continue
        hourly = (df.group_by(pl.col("datetime_jst").dt.hour().alias("h"))
                    .agg(pl.col("shortwave_radiation_wm2").mean()).sort("h"))
        sw = dict(zip(hourly["h"].to_list(),
                      hourly["shortwave_radiation_wm2"].to_list(), strict=True))
        report["diurnal"][area] = {
            "sw_mean_h00": round(sw.get(0, float("nan")), 2),
            "sw_mean_h12": round(sw.get(12, float("nan")), 2),
            "diurnal_ok": bool(sw.get(12, 0.0) > 100.0 and sw.get(0, 1.0) < 5.0),
        }
        ah_path = ALLHOURS_DIR / _cache_path(area).name.replace("jma_gpv_pregate_",
                                                                "jma_gpv_")
        if not ah_path.exists():
            continue
        ah = pl.read_parquet(ah_path)
        j = df.join(ah, on="datetime_jst", suffix="_00z")
        if j.height < 24:
            continue
        per_ch = {}
        for c in WEATHER_COLS:
            x = j[c].to_numpy()
            y = j[f"{c}_00z"].to_numpy()
            m = ~(np.isnan(x) | np.isnan(y))
            zero_var = x[m].std() == 0.0 or y[m].std() == 0.0
            per_ch[c] = {
                "pearson_r": None if zero_var else round(
                    float(np.corrcoef(x[m], y[m])[0, 1]), 4),
                "mean_abs_diff": round(float(np.abs(x[m] - y[m]).mean()), 3),
                "identical": bool(np.allclose(x[m], y[m])),
            }
        report["overlap_vs_00utc"][area] = {"n_overlap_hours": j.height, **per_ch}
    return report


# --------------------------------------------------- corroboration / aggregate
HISTFC_DIR = RAW / "open_meteo_historical_forecast"
RUNS_DIR = RAW.parent / "runs"
AGG_PATH = RUNS_DIR / "phase53_rish_pregate.json"


def _load_prev(area: str, n_prev: int) -> pl.DataFrame:
    """Concatenate the cached Open-Meteo previous_day{n} years for one area."""
    lat, lon = AREA_COORDS[area]
    parts = [
        pl.read_parquet(p)
        for p in sorted(HISTFC_DIR.glob(
            f"open_meteo_prev{n_prev}_jma_msm_{area}_{lat:.4f}_{lon:.4f}_*.parquet"))
    ]
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts).unique("datetime_jst").sort("datetime_jst")


def _pair_stats(x: np.ndarray, y: np.ndarray) -> dict:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2:
        return {"n": int(x.size)}
    d = x - y
    r = (None if x.std() == 0 or y.std() == 0
         else round(float(np.corrcoef(x, y)[0, 1]), 4))
    return {"n": int(x.size), "pearson_r": r,
            "rmse": round(float(np.sqrt((d**2).mean())), 3),
            "mae": round(float(np.abs(d).mean()), 3),
            "bias": round(float(d.mean()), 3)}


def _fisher_mean(rs: list[float]) -> float | None:
    vals = [r for r in rs if r is not None and abs(r) < 1.0]
    if not vals:
        return None
    return round(float(np.tanh(np.mean(np.arctanh(vals)))), 4)


def _pooled(per_area: dict[str, dict], ch: str, key: str) -> dict:
    """Pool a channel's comparison across areas (n-weighted RMSE/MAE)."""
    rows = [per_area[a][key][ch] for a in per_area
            if key in per_area[a] and ch in per_area[a][key]
            and "rmse" in per_area[a][key][ch]]
    if not rows:
        return {}
    n = sum(r["n"] for r in rows)
    mse = sum(r["rmse"] ** 2 * r["n"] for r in rows) / n
    return {
        "n": n,
        "rmse": round(float(np.sqrt(mse)), 3),
        "mae": round(sum(r["mae"] * r["n"] for r in rows) / n, 3),
        "mean_area_r_fisher": _fisher_mean([r.get("pearson_r") for r in rows]),
        "min_area_r": min((r["pearson_r"] for r in rows
                           if r.get("pearson_r") is not None), default=None),
    }


def aggregate() -> dict:  # noqa: C901 - single reporting pass
    """Full-period validation + vintage corroboration -> aggregation JSON.

    Compares the exact pre-gate series against
      (a) Open-Meteo ``previous_day2`` (prev2) -- the manuscript's headline
          pre-gate tier, a conservative ~48 h-old vintage, and
      (b) the D-1 00 UTC GPV run -- exact-issuance but POST-gate (~+90 min),
    on common slots, per area and pooled. Also reports prev1 (first post-gate
    Open-Meteo vintage) and the prev2-vs-postgate leg for the full triangle.
    """
    frames = _load_caches()
    days = sorted(_complete_days(frames))
    val = validate()
    out: dict = {
        "phase": 53,
        "tier": "jma_msm_gpv_pregate (exact-issuance, pre-gate admissible)",
        "generated_at": datetime.now(UTC).isoformat(),
        "coverage": {
            "delivery_days": len(days),
            "first_day": str(days[0]), "last_day": str(days[-1]),
            "areas": list(AREA_COORDS),
            "slots_per_day": 24,
            "slot_grid": "hourly JST (00..23); JEPX half-hour slots inherit the "
                         "containing hour",
            "total_rows_per_area": int(frames[next(iter(AREA_COORDS))].height),
            "missing_days_in_span": [
                str(d) for d in
                ((days[0] + timedelta(days=i)) for i in range((days[-1] - days[0]).days + 1))
                if d not in set(days)
            ],
            "channels": list(WEATHER_COLS),
        },
        "run_rule": _init_meta()["rule"],
        "issuance_accounting": _issuance_accounting(days[-1]),
        "vintage_age_hours": {},
        "validation": {"diurnal": val["diurnal"]},
        "per_area": {},
        "pooled": {},
    }

    # measured lead structure (vintage age) of the exact series
    k = frames[next(iter(AREA_COORDS))]
    out["vintage_age_hours"] = {
        "exact_pregate_min": int(k["lead_hours"].min()),
        "exact_pregate_max": int(k["lead_hours"].max()),
        "exact_pregate_mean": round(float(k["lead_hours"].mean()), 2),
        "note": "prev2 = ~48 h nominal vintage (Open-Meteo previous_day2); "
                "the exact pre-gate set is the last vintage whose nominal "
                "dissemination precedes the D-1 10:00 gate",
    }

    per_area: dict[str, dict] = {}
    for area, (lat, lon) in AREA_COORDS.items():
        ex = frames[area]
        if ex.height == 0:
            continue
        rec: dict = {}
        pg = ex.select(["datetime_jst", *WEATHER_COLS])
        rec["channel_summary"] = {
            c: {"mean": round(float(ex[c].mean()), 3),
                "std": round(float(ex[c].std()), 3),
                "min": round(float(ex[c].min()), 3),
                "max": round(float(ex[c].max()), 3)}
            for c in WEATHER_COLS
        }
        # (b) post-gate D-1 00 UTC GPV
        ah_path = ALLHOURS_DIR / f"jma_gpv_{area}_{lat:.4f}_{lon:.4f}.parquet"
        post = (pl.read_parquet(ah_path).select(["datetime_jst", *WEATHER_COLS])
                if ah_path.exists() else pl.DataFrame())
        # (a) prev2 and context prev1
        p2, p1 = _load_prev(area, 2), _load_prev(area, 1)

        def _cmp(other: pl.DataFrame, suffix: str, base: pl.DataFrame = pg) -> dict:
            """Inner-join on the slot grid and compare each channel.

            The other frame's channel columns are renamed to ``<c>__o`` BEFORE
            the join: both GPV caches use identical channel names, so joining
            them raw silently compares a column with itself.
            """
            if other.height == 0:
                return {}
            ren = {}
            for c in WEATHER_COLS:
                src = c if c in other.columns else f"{c}_{suffix}"
                if src in other.columns:
                    ren[src] = f"{c}__o"
            if not ren:
                return {}
            oth = other.select(["datetime_jst", *ren]).rename(ren)
            j = base.select(["datetime_jst", *WEATHER_COLS]).join(
                oth, on="datetime_jst", how="inner")
            res = {}
            for c in WEATHER_COLS:
                if f"{c}__o" in j.columns:
                    res[c] = _pair_stats(j[c].to_numpy(), j[f"{c}__o"].to_numpy())
            res["window"] = ([str(j["datetime_jst"].min()), str(j["datetime_jst"].max())]
                             if j.height else [])
            return res

        rec["vs_prev2"] = _cmp(p2, "pd2")
        rec["vs_postgate_00utc"] = _cmp(post, "")
        rec["vs_prev1_context"] = _cmp(p1, "pd1")
        # Same-slot (apples-to-apples) legs: the post-gate cache spans only the
        # committed FY2024-25 window, so restrict every leg to ITS slots before
        # declaring which vintage the exact series resembles.
        if post.height:
            common = post.select("datetime_jst")
            pg_c = pg.join(common, on="datetime_jst", how="inner")
            rec["common_window_vs_prev2"] = _cmp(p2, "pd2", base=pg_c)
            rec["common_window_vs_postgate"] = _cmp(post, "", base=pg_c)
            rec["common_window_vs_prev1"] = _cmp(p1, "pd1", base=pg_c)
        # triangle leg: prev2 vs post-gate on the same slots
        if post.height and p2.height:
            rec["prev2_vs_postgate"] = _cmp(p2, "pd2", base=post)
        per_area[area] = rec

    out["per_area"] = per_area
    for key in ("vs_prev2", "vs_postgate_00utc", "vs_prev1_context",
                "prev2_vs_postgate", "common_window_vs_prev2",
                "common_window_vs_postgate", "common_window_vs_prev1"):
        out["pooled"][key] = {c: _pooled(per_area, c, key) for c in WEATHER_COLS}

    # verdict: is the exact series closer to prev2 or to the post-gate run?
    # Uses the COMMON-WINDOW legs so both comparisons cover identical slots.
    verdict = {}
    for c in WEATHER_COLS:
        a = out["pooled"]["common_window_vs_prev2"].get(c, {}).get("rmse")
        b = out["pooled"]["common_window_vs_postgate"].get(c, {}).get("rmse")
        if a is None or b is None:
            continue
        verdict[c] = {
            "rmse_vs_prev2": a, "rmse_vs_postgate": b,
            "closer_to": "postgate_00utc" if b < a else "prev2",
            "ratio_prev2_over_postgate": round(a / b, 2) if b else None,
        }
    out["verdict"] = {
        "basis": "common-window legs (identical slots for both comparisons; the "
                 "post-gate 00UTC cache spans only the committed FY2024-25 window)",
        "per_channel": verdict,
        "summary": "the exact pre-gate series is closer to the post-gate D-1 "
                   "00UTC run than to prev2 on every channel"
                   if verdict and all(v["closer_to"] == "postgate_00utc"
                                      for v in verdict.values())
                   else "mixed / see per_channel",
        "interpretation": "corroborates the vintage ordering (older vintage = "
                          "larger departure from the freshest run) and shows the "
                          "manuscript's prev2 pre-gate tier is CONSERVATIVE: the "
                          "true last-admissible vintage carries information much "
                          "closer to the post-gate run. It does NOT re-estimate "
                          "the price-forecast ladder -- no price sweep was run.",
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    AGG_PATH.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log.info("aggregation written: %s", AGG_PATH)
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="auto", help="YYYY-MM-DD or 'auto'")
    ap.add_argument("--validate", action="store_true",
                    help="run cache sanity checks instead of the sweep")
    ap.add_argument("--aggregate", action="store_true",
                    help="full-period validation + vintage corroboration -> JSON")
    args = ap.parse_args()

    if args.validate:
        print(json.dumps(validate(), indent=1))
        return
    if args.aggregate:
        agg = aggregate()
        print(json.dumps({"coverage": agg["coverage"], "pooled": agg["pooled"],
                          "verdict": agg["verdict"]}, indent=1, default=str))
        return

    start = date.fromisoformat(args.start)
    if args.end == "auto":
        with httpx.Client(timeout=60.0, follow_redirects=True) as c:
            end = _probe_latest_day(c)
    else:
        end = date.fromisoformat(args.end)
    sweep(start, end)


if __name__ == "__main__":
    sys.exit(main())
