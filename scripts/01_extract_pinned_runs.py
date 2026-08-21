r"""Phase 58: run-pinned (exact-run) JMA-MSM GPV tiers for the vintage ladder.

WHY THIS EXISTS
---------------
The third external review established that the manuscript's headline "pre-gate
vintage" was Open-Meteo ``*_previous_day2``, i.e. "the value predicted ~48 h
before each valid time" -- a FIXED-LEAD MOSAIC stitched from many runs, not the
single run that Figure 1 draws and the text names in the singular. The tier is
still dissemination-admissible, but it does not correspond to an information set
any real bidder held at the gate. This script extracts the run-pinned tiers that
do, one pinned run per whole delivery day.

Two facts from the 2026-07-26/28 RISH probes shape the run plans:
  * The 78 h horizon exists ONLY for the 00/12 UTC cycles; 03/06/09/15/18/21 UTC
    stop at FH39. Therefore **D-2 12 UTC is the only pre-gate run that can cover
    all 48 delivery slots of one day from a single run**, and a constant-48 h-lead
    series is structurally unattainable from any single MSM run. That is a methods
    finding, not an apology.
  * Dissemination = initialisation + 2 h 30 m (JMBSC f-online10200; RISH
    ``Last-Modified`` measures init + 2 h 16 m), so admissibility is judged on
    dissemination, never on initialisation.

RUN PLANS (delivery day D, all times JST unless marked UTC)
-----------------------------------------------------------
  A  msm_d2_12z   D-2 12 UTC run, FH27-50   dissem. D-2 23:30, +10h30m PRE-gate
                  -> the coherent single-run pre-gate headline candidate
  C  msm_d1_00z   D-1 00 UTC run, FH15-38   dissem. D-1 11:30, -1h30m POST-gate
                  -> the first run the gate excludes; run-pinned gate-timing arm
  D  msm_d1_12z   D-1 12 UTC run, FH15-38   dissem. D-1 23:30, POST-gate (deep)
  E  msm_d3_12z   D-3 12 UTC run, FH51-74   dissem. D-3 23:30, deep PRE-gate

Only A and C are needed for the manuscript; D/E are saturation anchors kept
available behind the same CLI.

CACHE (deliberately a NEW directory tree)
-----------------------------------------
  data/raw/jma_gpv_exactrun/{plan_key}/jma_gpv_{plan_key}_{area}_{lat}_{lon}.parquet
  columns: datetime_jst, 4 weather cols, issuance_time_utc, lead_hours

The new tree matters: ``s_20260726_phase54_nwp_proxy_baseline.py`` and
``s_20260726_phase52_fip_commitment.py`` glob their weather caches with
``*_{area}_*.parquet`` and then ``unique("datetime_jst")``. Writing new
per-run files into an existing directory would make those loaders silently keep
an arbitrary run and corrupt published numbers.

INTEGRITY AUDITS (blocking, per the phase58 pre-commit doc)
-----------------------------------------------------------
  --audit A : the D-2 12 UTC values at lead 49/50 must be BIT-IDENTICAL to the
              ones phase53 already stored in data/raw/jma_gpv_pregate/. A
              mismatch means the extraction is not reproducible -> abort.
  --audit C : re-fetch N sample days and require max|diff| == 0 against the
              existing D-1 00 UTC cache (data/raw/jma_gpv_allhours/jma_gpv/),
              which carries no issuance/lead columns. Only after that audit
              passes does --seed copy those 426 days forward, saving ~2.4 h of
              redundant download.

FAIL-LOUD: the first GRIB decode error, exhausted-retry HTTP error, missing
delivery hour or NaN channel aborts the sweep (exit != 0). Nothing is skipped
silently. Per-day checkpointing: a day counts as done only when all 9 area caches
hold its full 24 hourly rows, so interruption anywhere is safe.

CLI (from the repository root):
  uv run python scripts/01_extract_pinned_runs.py \
      --plan C --audit                       # blocking audit, no sweep
  uv run python ... --plan C --seed           # audit, then seed from allhours
  uv run python ... --plan C --start 2024-03-01 --end 2026-07-29
  uv run python ... --plan A --audit
  uv run python ... --plan A --start 2024-03-01 --end 2026-07-29
  uv run python ... --plan A --validate       # cache sanity report

Daemonise (survives session kills; the phase53 lesson):
  nohup uv run python scripts/01_extract_pinned_runs.py \
      --plan C --start 2024-03-01 --end 2026-07-29 \
      > /tmp/phase58_C.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points
from alg.core.epf.data.jma_gpv_loader import (
    SURFACE_FH_FILES_D1_00Z_FULLDAY,
    SURFACE_FH_FILES_D2_12Z_FULLDAY,
    SURFACE_FH_FILES_DEEP,
    WEATHER_COLS,
    _run_url,
    fetch_gpv_run_points,
)
from alg.core.epf.data.registry import REGISTRY_FILENAME, DatasetRegistry, make_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase58")

JST = timezone(timedelta(hours=9), name="JST")
RAW = EpfConfig().resolved("raw_data_dir")
ROOT_DIR = RAW / "jma_gpv_exactrun"
PREGATE_DIR = RAW / "jma_gpv_pregate"          # phase53 composite (audit source for A)
ALLHOURS_DIR = RAW / "jma_gpv_allhours" / "jma_gpv"  # D-1 00 UTC (audit/seed for C)

DISSEMINATION_LAG = timedelta(hours=2, minutes=30)  # nominal, per JMBSC
POLITE_SLEEP_S = 1.0
HOURS_PER_DAY = 24
AUDIT_SAMPLE_DAYS = 10

# Canonical 9 area points (identical to phase 30.2 / 47 / 49 / 51 / 53 loaders).
AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}


@dataclass(frozen=True)
class RunPlan:
    """One pinned MSM run per delivery day.

    ``hours_jst`` is the CONTRACTED delivery-hour coverage. The original plans all
    covered 0..23, but the cycle-curve plans cannot: the 39-hour horizon of the
    18 UTC cycle ends at delivery hour 18, and the deep FH52-78 file only starts
    at delivery hour 01. Making the contract explicit lets the per-day validation
    stay strict (exact expected hours, exact expected leads) without pretending
    every run spans a whole day.
    """

    key: str
    days_before: int      # run date = delivery day D minus this many days
    init_hour_utc: int
    fh_files: tuple[str, ...]
    role: str
    hours_jst: tuple[int, ...] = tuple(range(24))

    def run_utc(self, d: date) -> datetime:
        base = datetime(d.year, d.month, d.day, tzinfo=UTC)
        return (base - timedelta(days=self.days_before)).replace(hour=self.init_hour_utc)

    def lead_range(self, d: date) -> tuple[int, int]:
        """(lead of first covered hour, lead of last covered hour) in whole hours."""
        run = self.run_utc(d)
        first = datetime(d.year, d.month, d.day, self.hours_jst[0],
                         tzinfo=JST).astimezone(UTC)
        lo = int((first - run).total_seconds() // 3600)
        return (lo, lo + (self.hours_jst[-1] - self.hours_jst[0]))

    def dissemination_jst(self, d: date) -> datetime:
        return (self.run_utc(d) + DISSEMINATION_LAG).astimezone(JST)

    def gate_margin_minutes(self, d: date) -> int:
        """Minutes by which nominal dissemination precedes the gate (negative = late)."""
        gate = datetime(d.year, d.month, d.day, 10, 0, tzinfo=JST) - timedelta(days=1)
        return int((gate - self.dissemination_jst(d)).total_seconds() // 60)


RUN_PLANS: dict[str, RunPlan] = {
    "A": RunPlan("msm_d2_12z", 2, 12, SURFACE_FH_FILES_D2_12Z_FULLDAY,
                 "coherent single-run PRE-gate (only run spanning a whole day)"),
    "C": RunPlan("msm_d1_00z", 1, 0, SURFACE_FH_FILES_D1_00Z_FULLDAY,
                 "first POST-gate run (the run the gate excludes)"),
    "D": RunPlan("msm_d1_12z", 1, 12, SURFACE_FH_FILES_D1_00Z_FULLDAY,
                 "deep POST-gate saturation anchor"),
    # Plan E's original file tuple guessed FH52-63 + FH64-75; both 404 on RISH.
    # The archive ships ONE deep file, FH52-78 (~118 MB), verified by directory
    # listing and HEAD probes across 2024-2026. From D-3 12 UTC it covers delivery
    # hours 01-23 JST, which contains the 08:00-16:00 common evaluation window.
    "E": RunPlan("msm_d3_12z", 3, 12, SURFACE_FH_FILES_DEEP,
                 "deep PRE-gate cycle-curve rung (disseminated D-3 23:30 JST)",
                 hours_jst=tuple(range(1, 24))),
    # D-2 18 UTC: a 39-hour cycle, so its horizon ends at delivery hour 18. The
    # intermediate pre-gate rung between D-2 12Z (+10h30m) and D-2 21Z (+90min);
    # disseminated D-1 05:30 JST, 4 h 30 m before the gate.
    "F": RunPlan("msm_d2_18z", 2, 18, ("FH16-33", "FH34-39"),
                 "intermediate PRE-gate cycle-curve rung",
                 hours_jst=tuple(range(19))),
    # D-2 21 UTC covers delivery hours 00-21 (leads 18-39): the freshest
    # admissible cycle. Added for the MULTI-POINT re-sweep so the phase74
    # capacity-weighted gate composite can be rebuilt as plan G hours 00-21 +
    # plan A hours 22-23 without touching the phase53 single-point cache.
    "G": RunPlan("msm_d2_21z", 2, 21, ("FH16-33", "FH34-39"),
                 "freshest admissible cycle (composite component, hours 00-21)",
                 hours_jst=tuple(range(22))),
}

# key -> prefecture, filled only in --multipoint mode; recorded in plan meta.
POINT_PREFS: dict[str, str] = {}


def _enable_multipoint() -> None:
    """Swap the 9 single area points for 46 capacity-table prefecture points.

    Point keys are ``{area}_p{ii}`` in the declaration order of
    area_points_capacity.csv; weights are NOT applied here -- extraction is
    label-agnostic, and phase74 aggregates the cached point parquets with the
    capacity weights. Writes to a NEW cache root so `_complete_days()`
    intersections and the phase52/54 globs of the single-point caches cannot
    collide (pre-committed constraint).
    """
    global ROOT_DIR  # noqa: PLW0603 - deliberate mode switch before any I/O
    pts = area_points()
    coords: dict[str, tuple[float, float]] = {}
    for area in sorted(pts):
        for i, ap_ in enumerate(pts[area]):
            key = f"{area}_p{i:02d}"
            coords[key] = (ap_.lat, ap_.lon)
            POINT_PREFS[key] = ap_.prefecture
    AREA_COORDS.clear()
    AREA_COORDS.update(coords)
    ROOT_DIR = RAW / "jma_gpv_exactrun_mp"
    log.info("multipoint mode: %d points, cache root %s", len(coords), ROOT_DIR)


# ------------------------------------------------------------------ cache paths
def _plan_dir(plan: RunPlan) -> Path:
    return ROOT_DIR / plan.key


def _cache_path(plan: RunPlan, area: str) -> Path:
    lat, lon = AREA_COORDS[area]
    return _plan_dir(plan) / f"jma_gpv_{plan.key}_{area}_{lat:.4f}_{lon:.4f}.parquet"


def _meta_path(plan: RunPlan) -> Path:
    return _plan_dir(plan) / "_meta_issuance_accounting.json"


def _progress_path(plan: RunPlan) -> Path:
    return _plan_dir(plan) / "_progress.json"


# ------------------------------------------------------------------ cache I/O
def _complete_days(frames: dict[str, pl.DataFrame], plan: RunPlan) -> set[date]:
    """Days with every CONTRACTED hourly row in EVERY area cache."""
    want = len(plan.hours_jst)
    per_area: list[set[date]] = []
    for area in AREA_COORDS:
        df = frames[area]
        if df.height == 0:
            return set()
        counts = df.group_by(pl.col("datetime_jst").dt.date().alias("d")).len()
        per_area.append(set(counts.filter(pl.col("len") == want)["d"].to_list()))
    return set.intersection(*per_area)


def _load_caches(plan: RunPlan) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for area in AREA_COORDS:
        p = _cache_path(plan, area)
        out[area] = pl.read_parquet(p) if p.exists() else pl.DataFrame()
    return out


def _append_day(plan: RunPlan, frames: dict[str, pl.DataFrame], area: str,
                day_df: pl.DataFrame, d: date) -> None:
    old = frames[area]
    if old.height > 0:
        old = old.filter(pl.col("datetime_jst").dt.date() != d)  # drop partial rows
        frames[area] = pl.concat([old, day_df]).sort("datetime_jst")
    else:
        frames[area] = day_df
    frames[area].write_parquet(_cache_path(plan, area))


# ------------------------------------------------------------------ extraction
def _fetch_day(plan: RunPlan, d: date, client: httpx.Client) -> dict[str, pl.DataFrame]:
    """Fetch ``plan``'s pinned run and return the 24 delivery-day rows per area."""
    per_area = fetch_gpv_run_points(plan.run_utc(d), plan.fh_files, AREA_COORDS,
                                   client=client)
    lo, hi = plan.lead_range(d)
    out: dict[str, pl.DataFrame] = {}
    for area in AREA_COORDS:
        day_df = (per_area[area]
                  .filter(pl.col("datetime_jst").dt.date() == d)
                  .sort("datetime_jst"))
        hours = day_df["datetime_jst"].dt.hour().to_list()
        if hours != list(plan.hours_jst):
            msg = (f"{plan.key} {d} {area}: expected delivery hours "
                   f"{plan.hours_jst[0]}..{plan.hours_jst[-1]} from run "
                   f"{plan.run_utc(d).isoformat()} (leads {lo}..{hi}), got {hours}")
            raise ValueError(msg)
        leads = day_df["lead_hours"].to_list()
        if leads != list(range(lo, hi + 1)):
            msg = f"{plan.key} {d} {area}: lead mismatch, expected {lo}..{hi}, got {leads}"
            raise ValueError(msg)
        nulls = day_df.select(pl.col(list(WEATHER_COLS)).is_nan().any())
        if any(nulls.row(0)):
            msg = f"{plan.key} {d} {area}: NaN in extracted channels {nulls.row(0)}"
            raise ValueError(msg)
        out[area] = day_df
    return out


# ------------------------------------------------------------------ end probe
def _probe_latest_day(plan: RunPlan, client: httpx.Client) -> date:
    """Latest delivery day whose full file set exists on RISH (HEAD probe)."""
    d = datetime.now(JST).date() + timedelta(days=2)
    for _ in range(45):
        urls = [_run_url(plan.run_utc(d), fh) for fh in plan.fh_files]
        if all(client.head(u).status_code == 200 for u in urls):
            log.info("end=auto resolved to latest available delivery day %s", d)
            return d
        d -= timedelta(days=1)
        time.sleep(0.3)
    msg = f"{plan.key}: no available delivery day within 45 days back from today+2"
    raise RuntimeError(msg)


# ------------------------------------------------------------------ audits
def _audit_plan_a(client: httpx.Client, sample: int) -> dict:
    """D-2 12 UTC leads 49/50 must be bit-identical to the phase53 composite."""
    plan = RUN_PLANS["A"]
    ref = {a: pl.read_parquet(next(PREGATE_DIR.glob(f"jma_gpv_pregate_{a}_*.parquet")))
           for a in AREA_COORDS}
    days = sorted(_complete_days(ref, plan))
    if not days:
        msg = "phase53 pregate cache is empty; cannot audit plan A"
        raise RuntimeError(msg)
    picks = [days[i] for i in np.linspace(0, len(days) - 1, sample).astype(int)]
    worst = 0.0
    for d in picks:
        fetched = _fetch_day(plan, d, client)
        for area in AREA_COORDS:
            got = (fetched[area].filter(pl.col("lead_hours").is_in([49, 50]))
                   .sort("datetime_jst"))
            want = (ref[area]
                    .filter((pl.col("datetime_jst").dt.date() == d)
                            & (pl.col("issuance_time_utc").dt.hour() == 12))
                    .sort("datetime_jst"))
            if got.height != 2 or want.height != 2:
                msg = (f"audit A {d} {area}: expected 2 rows each, got "
                       f"{got.height} fetched / {want.height} reference")
                raise ValueError(msg)
            for c in WEATHER_COLS:
                diff = float(np.max(np.abs(np.asarray(got[c]) - np.asarray(want[c]))))
                worst = max(worst, diff)
                if diff != 0.0:
                    msg = (f"audit A FAILED {d} {area} {c}: max|diff| = {diff!r} "
                           "(expected exactly 0 - extraction is not reproducible)")
                    raise ValueError(msg)
        log.info("audit A %s: bit-identical at leads 49/50 across 9 areas", d)
        time.sleep(POLITE_SLEEP_S)
    return {"plan": "A", "sample_days": [str(x) for x in picks],
            "max_abs_diff": worst, "verdict": "PASS"}


def _audit_plan_c(client: httpx.Client, sample: int) -> dict:
    """Re-fetch sample days and require exact equality with the allhours cache."""
    plan = RUN_PLANS["C"]
    ref = {a: pl.read_parquet(next(ALLHOURS_DIR.glob(f"jma_gpv_{a}_*.parquet")))
           for a in AREA_COORDS}
    days = sorted(_complete_days(ref, plan))
    if not days:
        msg = "allhours cache is empty; cannot audit plan C"
        raise RuntimeError(msg)
    picks = [days[i] for i in np.linspace(0, len(days) - 1, sample).astype(int)]
    worst = 0.0
    for d in picks:
        fetched = _fetch_day(plan, d, client)
        for area in AREA_COORDS:
            got = fetched[area].sort("datetime_jst")
            want = (ref[area].filter(pl.col("datetime_jst").dt.date() == d)
                    .sort("datetime_jst"))
            if want.height != HOURS_PER_DAY:
                msg = f"audit C {d} {area}: reference has {want.height} rows, want 24"
                raise ValueError(msg)
            for c in WEATHER_COLS:
                diff = float(np.max(np.abs(np.asarray(got[c]) - np.asarray(want[c]))))
                worst = max(worst, diff)
                if diff != 0.0:
                    msg = (f"audit C FAILED {d} {area} {c}: max|diff| = {diff!r} "
                           "(expected exactly 0 - the legacy cache is a different run)")
                    raise ValueError(msg)
        log.info("audit C %s: identical to allhours cache across 9 areas", d)
        time.sleep(POLITE_SLEEP_S)
    return {"plan": "C", "sample_days": [str(x) for x in picks],
            "max_abs_diff": worst, "verdict": "PASS"}


def _seed_plan_c() -> int:
    """Copy the audited allhours days into plan C's cache with issuance/lead added.

    Only legitimate AFTER ``--audit C`` passes: the legacy cache carries no
    issuance columns, so they are reconstructed from the plan's own run rule.
    """
    plan = RUN_PLANS["C"]
    _plan_dir(plan).mkdir(parents=True, exist_ok=True)
    frames = _load_caches(plan)
    already = _complete_days(frames, plan)
    seeded = 0
    for area in AREA_COORDS:
        src = pl.read_parquet(next(ALLHOURS_DIR.glob(f"jma_gpv_{area}_*.parquet")))
        src = src.filter(~pl.col("datetime_jst").dt.date().is_in(list(already)))
        if src.height == 0:
            continue
        # Reuse the plan's OWN run rule rather than re-deriving it by truncation:
        # the delivery day spans two UTC dates, so truncating the UTC timestamp
        # would assign the late-evening hours to the following run.
        days = sorted(set(src["datetime_jst"].dt.date().to_list()))
        rule = pl.DataFrame(
            {"_d": days, "issuance_time_utc": [plan.run_utc(d) for d in days]},
            schema={"_d": pl.Date, "issuance_time_utc": pl.Datetime("us", "UTC")},
        )
        add = (src.with_columns(pl.col("datetime_jst").dt.date().alias("_d"))
               .join(rule, on="_d", how="inner")
               .drop("_d")
               .with_columns(((pl.col("datetime_jst").dt.convert_time_zone("UTC")
                               - pl.col("issuance_time_utc")).dt.total_hours())
                             .alias("lead_hours"))
               .sort("datetime_jst"))
        old = frames[area]
        merged = pl.concat([old, add], how="vertical_relaxed") if old.height > 0 else add
        merged.sort("datetime_jst").write_parquet(_cache_path(plan, area))
        seeded = len(days)
    log.info("seeded plan C with %d delivery days from the allhours cache", seeded)
    return seeded


# ------------------------------------------------------------------ metadata
def _init_meta(plan: RunPlan) -> dict:
    p = _meta_path(plan)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "tier": plan.key,
        "forecast_kind": "exact_run_pinned_whole_day",
        "role": plan.role,
        "source": "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original",
        "rule": {
            "gate": "JEPX day-ahead gate closure D-1 10:00 JST",
            "admissibility": "nominal dissemination (init + 2h30m, per JMBSC) <= gate",
            "run": f"delivery day D minus {plan.days_before} d, "
                   f"{plan.init_hour_utc:02d} UTC initialisation",
            "surface_files": list(plan.fh_files),
            "single_run_whole_day": True,
        },
        "channels": list(WEATHER_COLS),
        "area_coords": {a: list(c) for a, c in AREA_COORDS.items()},
        "point_prefectures": dict(POINT_PREFS),
        "per_day": {},
    }


def _day_accounting(plan: RunPlan, d: date) -> dict:
    lo, hi = plan.lead_range(d)
    margin = plan.gate_margin_minutes(d)
    return {
        "init_utc": plan.run_utc(d).isoformat(),
        "nominal_dissemination_jst": plan.dissemination_jst(d).isoformat(),
        "margin_minutes_before_gate": margin,
        "pre_gate": margin >= 0,
        "lead_hours_range": [lo, hi],
    }


# ------------------------------------------------------------------ sweep
def sweep(plan: RunPlan, start: date, end: date) -> None:
    _plan_dir(plan).mkdir(parents=True, exist_ok=True)
    frames = _load_caches(plan)
    done = _complete_days(frames, plan)
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in all_days if d not in done]
    log.info("sweep %s %s..%s: %d requested, %d already complete, %d to fetch",
             plan.key, start, end, len(all_days), len(all_days) - len(todo), len(todo))

    meta = _init_meta(plan)
    pp = _progress_path(plan)
    progress = (json.loads(pp.read_text(encoding="utf-8")) if pp.exists()
                else {"started_at": datetime.now(UTC).isoformat(), "day_seconds": {}})
    progress["window"] = [start.isoformat(), end.isoformat()]
    progress["plan"] = plan.key

    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for i, d in enumerate(todo):
            t0 = time.monotonic()
            day = _fetch_day(plan, d, client)
            for area in AREA_COORDS:
                _append_day(plan, frames, area, day[area], d)
            dt_s = time.monotonic() - t0

            acct = _day_accounting(plan, d)
            # The archive's own file timestamps. Nominal dissemination is derived
            # (init + 2h30m per JMBSC); these are OBSERVED, and they are what
            # substantiates -- or corrects -- the manuscript's "measured" lag.
            acct["archive_last_modified"] = getattr(
                fetch_gpv_run_points, "last_modified", {})
            meta["per_day"][d.isoformat()] = acct
            _meta_path(plan).write_text(json.dumps(meta, indent=1, default=str),
                                        encoding="utf-8")
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
            pp.write_text(json.dumps(progress, indent=1), encoding="utf-8")
            log.info("%s day %s done in %.1fs (%d/%d, ETA %s)",
                     plan.key, d, dt_s, i + 1, len(todo), progress["eta_utc"])
            time.sleep(POLITE_SLEEP_S)

    reg = DatasetRegistry(RAW / REGISTRY_FILENAME)
    for area, (lat, lon) in AREA_COORDS.items():
        reg.upsert(make_entry(
            name=f"jma_gpv_{plan.key}_{area}_{lat:.4f}_{lon:.4f}",
            raw_data_dir=RAW,
            file_path=_cache_path(plan, area),
            source_url=("http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/"
                        f"original (delivery day D minus {plan.days_before} d, "
                        f"{plan.init_hour_utc:02d} UTC run, files "
                        f"{'+'.join(plan.fh_files)}; run-pinned whole delivery day)"),
        ))
    log.info("sweep %s complete; %d areas registered", plan.key, len(AREA_COORDS))


# ------------------------------------------------------------------ validation
def validate(plan: RunPlan) -> dict:
    frames = _load_caches(plan)
    done = sorted(_complete_days(frames, plan))
    report: dict = {
        "plan": plan.key, "role": plan.role,
        "complete_days": len(done),
        "first_day": str(done[0]) if done else None,
        "last_day": str(done[-1]) if done else None,
        "per_area": {},
    }
    if done:
        d0, d1 = done[0], done[-1]
        report["lead_hours_range_first_day"] = list(plan.lead_range(d0))
        report["gate_margin_minutes_first_day"] = plan.gate_margin_minutes(d0)
        report["gate_margin_minutes_last_day"] = plan.gate_margin_minutes(d1)
        report["pre_gate_all_days"] = all(plan.gate_margin_minutes(d) >= 0 for d in done)
    for area in AREA_COORDS:
        df = frames[area]
        if df.height == 0:
            report["per_area"][area] = {"rows": 0}
            continue
        daytime = df.filter(pl.col("datetime_jst").dt.hour().is_between(10, 14))
        night = df.filter(pl.col("datetime_jst").dt.hour().is_in([0, 1, 2, 3]))
        report["per_area"][area] = {
            "rows": df.height,
            "issuance_hours_utc": sorted(set(df["issuance_time_utc"].dt.hour().to_list())),
            "lead_hours_min_max": [int(df["lead_hours"].min()), int(df["lead_hours"].max())],
            "sw_mean_daytime": round(float(daytime["shortwave_radiation_wm2"].mean()), 2),
            "sw_mean_night": round(float(night["shortwave_radiation_wm2"].mean()), 4),
        }
    return report


# ------------------------------------------------------------------ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=sorted(RUN_PLANS),
                    help="run plan key (A=pre-gate single run, C=first post-gate)")
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="auto",
                    help="ISO date or 'auto' (HEAD-probe the latest available day)")
    ap.add_argument("--audit", action="store_true",
                    help="blocking integrity audit against the existing caches")
    ap.add_argument("--audit-sample", type=int, default=AUDIT_SAMPLE_DAYS)
    ap.add_argument("--seed", action="store_true",
                    help="plan C only: audit, then seed from the allhours cache")
    ap.add_argument("--validate", action="store_true", help="cache sanity report only")
    ap.add_argument("--multipoint", action="store_true",
                    help="46 capacity-table prefecture points into a NEW cache "
                         "root (phase74); audits/seed are single-point only")
    args = ap.parse_args()
    plan = RUN_PLANS[args.plan]
    if args.multipoint:
        if args.audit or args.seed:
            log.error("--audit/--seed compare against single-point caches; "
                      "they are undefined in --multipoint mode")
            return 2
        _enable_multipoint()

    if args.validate:
        print(json.dumps(validate(plan), indent=1))
        return 0

    if args.audit or args.seed:
        auditor = {"A": _audit_plan_a, "C": _audit_plan_c}.get(args.plan)
        if auditor is None:
            log.error("no audit defined for plan %s", args.plan)
            return 2
        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            verdict = auditor(client, args.audit_sample)
        log.info("AUDIT %s", json.dumps(verdict))
        _plan_dir(plan).mkdir(parents=True, exist_ok=True)
        (_plan_dir(plan) / "_audit.json").write_text(json.dumps(verdict, indent=1),
                                                     encoding="utf-8")
        if args.seed:
            if args.plan != "C":
                log.error("--seed is only defined for plan C")
                return 2
            _seed_plan_c()
        if args.audit and not args.seed:
            return 0

    start = date.fromisoformat(args.start)
    if args.end == "auto":
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            end = _probe_latest_day(plan, client)
    else:
        end = date.fromisoformat(args.end)
    sweep(plan, start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
