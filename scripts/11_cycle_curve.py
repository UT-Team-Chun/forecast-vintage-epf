r"""Phase 73 v2: the cycle curve on the EXTENDED >=730-day panel.

MECHANICAL FORK of s_20260803_phase73_cycle_curve.py (2026-08-11): the only
edits are the three constants below (panel end 2026-07-18 -> 2026-08-05, a
fresh checkpoint directory, and new output names) so the frozen phase73
artefacts stay untouched per the phase85 precommit's freeze policy. Every
contrast in this curve is within-SP measurement, so the CW promotion does not
touch it (precommit, tier design).

ORIGINAL DOCSTRING:
Phase 73: the improvement-versus-dissemination curve, cycle by cycle.

WHY THIS EXISTS
---------------
The manuscript's saturation claim ("value from this channel saturates well
before the deadline") rested on three run-pinned points and a near-flat
dose-response slope. The fifth review asked for the curve itself: every
archived MSM cycle that can reach a common evaluation window, scored on
identical folds, with the saturation criterion fixed in advance.

DESIGN (docs/research/2026-08-03_phase73_cycle_curve_precommit.md)
------------------------------------------------------------------
Common window: delivery hours 08:00-16:00 JST (slots whose hour is in 8..16),
the PV-dominated daytime band every reachable cycle covers. Rungs, ordered by
measured dissemination (phase72: init + 136 min):

    msm_d3_12z  disseminated D-3 23:16 JST   gate margin +34.7 h
    msm_d2_12z              D-2 23:16 JST                +10.7 h
    msm_d2_18z              D-1 05:16 JST                 +4.7 h
    msm_d2_21z              D-1 08:16 JST                 +1.7 h   (pregate cache,
                                                           pure 21Z at these hours)
    msm_d1_00z              D-1 11:16 JST                 -1.3 h   (post-gate)

D-2 09Z and 15Z are STRUCTURALLY impossible: their 39-hour horizons end before
the common window's final slot. That is an archive property, not missing data.

Folds, learner and hyperparameters are phase59's verbatim. Two departures,
applied identically to every cell: (i) the panel keeps all 24 delivery hours
and leaves NaN where a cycle's horizon does not reach -- XGBoost treats NaN as
missing natively -- so the fold schedule and training rows match phase59;
(ii) the panel is restricted to delivery days present and contract-complete in
EVERY tier cache, so no cell ever scores a fabricated feature. Scoring pools
the common-window slots only.

GATES (pre-committed)
---------------------
Gate 41  slope of pre-gate improvement on measured dissemination-to-gate lead,
         day-block-bootstrap 95% CI: 41a flat (CI covers 0) -> saturation is a
         measured statement; 41b significantly negative in lead (fresher is
         better) -> RETRACT the saturation claim and describe the curve;
         41c positive -> describe.
Gate 42  the common-window gate step (d2_12z -> d1_00z) must be consistent
         with the full-day phase59 estimate; a sign flip with CI off zero is
         reported separately as a daytime-band effect.

CLI (from the repository root):
  uv run python s_20260803_phase73_cycle_curve.py --step build
  uv run python ... --step build --areas kyushu,tokyo   # smoke test
  uv run python ... --step analyse
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import DEFAULT_WEATHER_COLS, attach_hourly_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase73")
JST = timezone(timedelta(hours=9), name="JST")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
EXACTRUN_DIR = RAW / "jma_gpv_exactrun"
PREGATE_DIR = RAW / "jma_gpv_pregate"
CKPT = RUN / "phase73v2_checkpoints"
OUT_PARQUET = RUN / "phase73v2_cycle_curve_per_slot.parquet"
OUT_JSON = RUN / "phase73v2_cycle_curve.json"
OUT_PNG = RUN / "phase73_cycle_curve.png"

# protocol pinned to phase51/59 -- identical folds
INITIAL_TRAIN_END = datetime(2024, 7, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
MAX_FOLDS = 800
SEED = 2026
PANEL_START = date(2024, 3, 2)
ERA5_END = date(2026, 8, 5)   # phase85 precommit extension
N_BOOT = 10_000
BOOT_BLOCK = 7
MIN_TRAIN_ROWS = 100
WINDOW_HOURS = tuple(range(8, 17))     # delivery hours 08..16 JST inclusive
MEASURED_LAG_MIN = 136                 # phase72: init + 2 h 16 m, median over 881 d
TICK = 0.01                            # one JEPX price tick, JPY/kWh
PRECOMMIT = "docs/research/2026-08-03_phase73_cycle_curve_precommit.md"

AREA_COORDS: dict[str, tuple[float, float]] = {
    "hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87), "tokyo": (35.68, 139.76),
    "chubu": (35.18, 136.91), "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
    "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05), "kyushu": (33.59, 130.40),
}

# tier -> (cache kind, plan dir, init offset days, init hour UTC)
TIERS: dict[str, dict] = {
    "msm_d3_12z": {"kind": "exactrun", "plan": "msm_d3_12z", "d": 3, "h": 12},
    "msm_d2_12z": {"kind": "exactrun", "plan": "msm_d2_12z", "d": 2, "h": 12},
    "msm_d2_18z": {"kind": "exactrun", "plan": "msm_d2_18z", "d": 2, "h": 18},
    "msm_d2_21z": {"kind": "pregate", "plan": None, "d": 2, "h": 21},
    "msm_d1_00z": {"kind": "exactrun", "plan": "msm_d1_00z", "d": 1, "h": 0},
}
PRE_GATE_TIERS = ("msm_d3_12z", "msm_d2_12z", "msm_d2_18z", "msm_d2_21z")


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


def gate_margin_hours(tier: str) -> float:
    """Hours from MEASURED dissemination to the 10:00 JST gate (positive = pre)."""
    spec = TIERS[tier]
    init_jst_hour = (spec["h"] + 9) % 24
    init_day_offset = spec["d"] - (1 if spec["h"] + 9 >= 24 else 0)
    # gate is 10:00 JST on D-1
    dissem_min = init_jst_hour * 60 + MEASURED_LAG_MIN
    gate_min = 10 * 60 + (init_day_offset - 1) * 24 * 60
    return round((gate_min - dissem_min) / 60.0, 2)


# ==================================================================== loading
def _load_tier(tier: str, area: str) -> pl.DataFrame:
    spec = TIERS[tier]
    lat, lon = AREA_COORDS[area]
    if spec["kind"] == "pregate":
        path = PREGATE_DIR / f"jma_gpv_pregate_{area}_{lat:.4f}_{lon:.4f}.parquet"
        df = pl.read_parquet(path).sort("datetime_jst")
        # the composite is pure 21Z wherever issuance hour is 21; the common
        # window (08-16 JST) lies inside that range, verified 2026-07-30
        if "issuance_time_utc" in df.columns:
            df = df.filter(pl.col("issuance_time_utc").dt.hour() == spec["h"])
    else:
        path = (EXACTRUN_DIR / spec["plan"]
                / f"jma_gpv_{spec['plan']}_{area}_{lat:.4f}_{lon:.4f}.parquet")
        df = pl.read_parquet(path).sort("datetime_jst")
    keep = ["datetime_jst", *DEFAULT_WEATHER_COLS]
    return df.select(keep).rename({c: f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS})


def _tier_complete_days(tier: str) -> set[date]:
    """Delivery days for which the tier has ALL common-window hours in ALL areas."""
    per_area: list[set[date]] = []
    for area in AREA_COORDS:
        df = _load_tier(tier, area)
        sub = (df.filter(pl.col("datetime_jst").dt.hour().is_in(list(WINDOW_HOURS)))
                 .group_by(pl.col("datetime_jst").dt.date().alias("d")).len())
        per_area.append(set(sub.filter(pl.col("len") == len(WINDOW_HOURS))["d"]
                            .to_list()))
    return set.intersection(*per_area)


# ==================================================================== design
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    out = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        out = out.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    out = make_calendar_features(out, datetime_col="datetime_jst")
    base = [*(f"lag_{lag}" for lag in (48, 96, 336, 672)),
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
            "is_holiday"]
    cells = {"price_only": list(base)}
    for tier in TIERS:
        cells[tier] = [*base, *(f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS)]
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


# ==================================================================== build
def _run_area(area: str, panel_days: set[date]) -> None:
    ckpt = CKPT / f"{area}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s -- skipping", ckpt.name)
        return
    price = (load_jepx_spot(years=[2023, 2024, 2025, 2026], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "area", "price_yen_kwh"))
    df = price.sort("datetime_jst")
    for tier in TIERS:
        df = attach_hourly_features(
            df, _load_tier(tier, area),
            feature_cols=[f"{c}_{tier}" for c in DEFAULT_WEATHER_COLS])
    df, cells = _design(df)
    # NaN tier features are legitimate (horizon gaps) and XGBoost handles them
    # as missing; only the base design must be complete.
    df = (df.drop_nulls(subset=[*cells["price_only"], "price_yen_kwh"])
          .filter(pl.col("datetime_jst").dt.date().is_in(sorted(panel_days)))
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
        hours = test["datetime_jst"].dt.hour().to_list()
        rows.extend({"area": area, "datetime_jst": dts[s],
                     "test_day": test_day.isoformat(), "y_actual": y[s],
                     "in_window": hours[s] in WINDOW_HOURS,
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
    days_per_tier = {t: _tier_complete_days(t) for t in TIERS}
    for t, ds in days_per_tier.items():
        log.info("tier %s: %d common-window-complete days", t, len(ds))
    panel_days = set.intersection(*days_per_tier.values())
    panel_days = {d for d in panel_days if PANEL_START <= d <= ERA5_END}
    log.info("panel day intersection: %d days", len(panel_days))
    for area in areas:
        _run_area(area, panel_days)
    if set(areas) == set(AREA_COORDS):
        frames = [pl.read_parquet(CKPT / f"{a}.parquet") for a in AREA_COORDS]
        pl.concat(frames).write_parquet(OUT_PARQUET)
        log.info("wrote %s", OUT_PARQUET)


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


def analyse() -> dict:
    panel = pl.read_parquet(OUT_PARQUET).filter(pl.col("in_window"))
    cells = ["price_only", *TIERS]
    per_day = (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .sort("test_day"))
    n_days = per_day.height
    imp = {t: (per_day["mae_price_only"] - per_day[f"mae_{t}"]).to_numpy()
           for t in TIERS}
    idx = _boot_idx(n_days)

    rungs: dict = {}
    for t in TIERS:
        boot = imp[t][idx].mean(axis=1)
        rungs[t] = {
            "gate_margin_hours_measured": gate_margin_hours(t),
            "pre_gate": t in PRE_GATE_TIERS,
            "improvement": _j(float(imp[t].mean()), 4),
            "ci95": [_j(float(np.quantile(boot, 0.025)), 4),
                     _j(float(np.quantile(boot, 0.975)), 4)],
            "dm_l3": _hln_dm(imp[t], lag=3),
            "dm_l9": _hln_dm(imp[t], lag=9),
        }

    # pairwise differences, adjacent in dissemination order + the gate step.
    # Each step also gets the one-tick TOST: the prose wants to say the curve is
    # FLAT after the first step, and a non-significant difference is not
    # flatness -- only equivalence at a stated margin is. Reported for every
    # step whether it passes or not.
    order = [*PRE_GATE_TIERS, "msm_d1_00z"]
    pairwise: dict = {}
    for a, b in itertools.pairwise(order):
        d = imp[b] - imp[a]
        boot = d[idx].mean(axis=1)
        d_bar = float(d.mean())
        dm3, dm9 = _hln_dm(d, lag=3), _hln_dm(d, lag=9)
        tost: dict = {"margin": TICK,
                      "boot_p": _j(float(max(np.mean(boot <= -TICK),
                                             np.mean(boot >= TICK))), 6)}
        for lag, dm in (("l3", dm3), ("l9", dm9)):
            se = abs(d_bar / dm["t_hln"]) if dm["t_hln"] not in (None, 0) else None
            tost[lag] = (None if not se else
                         _j(max(1.0 - float(stats.t.cdf((d_bar + TICK) / se,
                                                        df=n_days - 1)),
                                float(stats.t.cdf((d_bar - TICK) / se,
                                                  df=n_days - 1))), 6))
        arms = [tost["boot_p"], tost["l3"], tost["l9"]]
        tost["equivalent"] = bool(all(v is not None and v < 0.05 for v in arms))
        pairwise[f"{b}_minus_{a}"] = {
            "diff": _j(d_bar, 4),
            "ci95": [_j(float(np.quantile(boot, 0.025)), 4),
                     _j(float(np.quantile(boot, 0.975)), 4)],
            "dm_l3": dm3, "dm_l9": dm9,
            "tost_one_tick": tost,
        }
    pairwise_summary = {
        "n_steps_after_first": sum(1 for k in pairwise
                                   if not k.endswith("msm_d3_12z")),
        "n_equivalent_at_one_tick": sum(
            1 for k, v in pairwise.items()
            if not k.endswith("msm_d3_12z") and v["tost_one_tick"]["equivalent"]),
        "n_significant_after_first": sum(
            1 for k, v in pairwise.items()
            if not k.endswith("msm_d3_12z") and v["dm_l3"]["p_hln"] < 0.05),
        "widest_ci_ticks_after_first": _j(max(
            max(abs(v["ci95"][0]), abs(v["ci95"][1])) / TICK
            for k, v in pairwise.items() if not k.endswith("msm_d3_12z")), 2),
        "reading": "A non-significant step is not a flat step. After the first "
                   "step no adjacent difference is distinguishable from zero, "
                   "but the intervals reach beyond one tick, so the honest "
                   "statement is 'no further step is detectable, with the "
                   "resolution this panel affords', not 'flat within a tick'.",
    }

    # ---- Gate 41: slope of pre-gate improvement on measured lead
    x = np.array([gate_margin_hours(t) for t in PRE_GATE_TIERS])
    y_mat = np.stack([imp[t] for t in PRE_GATE_TIERS])   # (4, n_days)
    xc = x - x.mean()
    slope_day = (xc[:, None] * (y_mat - y_mat.mean(axis=0))).sum(axis=0) / (xc**2).sum()
    slope = float(slope_day.mean())
    boot = slope_day[idx].mean(axis=1)
    slope_ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    flat = slope_ci[0] <= 0.0 <= slope_ci[1]
    branch = ("41a" if flat else "41b" if slope < 0 else "41c")
    gate41 = {
        "slope_jpykwh_per_gate_margin_hour": _j(slope, 6),
        "ci95": [_j(v, 6) for v in slope_ci],
        "x_gate_margin_hours": [float(v) for v in x],
        "branch": branch,
        "reading": {
            "41a": "flat: saturation before the deadline is now a measured "
                   "statement across four cycles",
            "41b": "improvement RISES toward the gate: the saturation claim is "
                   "retracted and the curve described",
            "41c": "improvement falls toward the gate (staler is better): "
                   "described, connected to the composite smoothing discussion",
        }[branch],
    }

    # ---- Gate 42: the gate step on the common window
    d_gate = imp["msm_d1_00z"] - imp["msm_d2_12z"]
    boot = d_gate[idx].mean(axis=1)
    gate42 = {
        "common_window_gate_step": _j(float(d_gate.mean()), 4),
        "ci95": [_j(float(np.quantile(boot, 0.025)), 4),
                 _j(float(np.quantile(boot, 0.975)), 4)],
        "dm_l3": _hln_dm(d_gate, lag=3),
        "phase59_full_day_bound": "0.010 JPY/kWh (95% upper bound, phase59)",
        "consistent": None,  # filled below
    }
    gate42["consistent"] = bool(gate42["ci95"][0] <= 0.010)

    out = {
        "phase": 73,
        "title": "Cycle-by-cycle improvement curve on the common 08:00-16:00 window",
        "precommit_doc": PRECOMMIT,
        "panel": {"n_days": n_days, "n_slots": int(panel.height),
                  "window_hours_jst": list(WINDOW_HOURS),
                  "measured_dissemination_lag_min": MEASURED_LAG_MIN,
                  "impossible_cycles": "D-2 09Z and 15Z: 39-hour horizons end "
                                       "before the window's final slot"},
        "rungs": rungs,
        "pairwise_adjacent": pairwise,
        "pairwise_summary": pairwise_summary,
        "gate41_slope": gate41,
        "gate42_gate_step": gate42,
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "generated_utc": datetime.now(UTC).isoformat(),
                     "seed": SEED, "n_boot": N_BOOT},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("wrote %s", OUT_JSON)
    _plot(rungs)
    for t in order:
        r = rungs[t]
        log.info("%-12s margin %+6.1f h  imp %.4f %s", t,
                 r["gate_margin_hours_measured"], r["improvement"], r["ci95"])
    log.info("Gate 41 branch %s: slope %.5f %s", branch, slope, gate41["ci95"])
    log.info("Gate 42 step %.4f %s consistent=%s", gate42["common_window_gate_step"],
             gate42["ci95"], gate42["consistent"])
    return out


def _plot(rungs: dict) -> None:
    """Draw the measured curve: improvement vs dissemination-to-gate margin."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ink, accent = "#333333", "#1a6b8a"
    for t, r in rungs.items():
        x = -r["gate_margin_hours_measured"]          # hours relative to gate, 0 = gate
        y = r["improvement"]
        lo, hi = r["ci95"]
        marker = "o" if r["pre_gate"] else "s"
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, color=accent,
                    ecolor=ink, elinewidth=1.2, capsize=3, markersize=7,
                    markerfacecolor=(accent if r["pre_gate"] else "white"),
                    markeredgecolor=accent, markeredgewidth=1.6, zorder=3)
        ax.annotate(t.replace("msm_", "").replace("_", " ").upper(),
                    (x, hi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=ink)
    ax.axvline(0, color="#b03a2e", linewidth=1.2, zorder=1)
    ax.annotate("10:00 JST gate", (0, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(5, 8), fontsize=8,
                color="#b03a2e")
    ax.set_xlabel("Measured dissemination time relative to the gate (hours)")
    ax.set_ylabel("Pooled MAE improvement over price-only (JPY/kWh)")
    ax.set_title("Weather value by MSM cycle, common 08:00-16:00 window "
                 "(filled = pre-gate, open = post-gate)", fontsize=10)
    ax.grid(visible=True, linewidth=0.4, alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    log.info("wrote %s", OUT_PNG)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["build", "analyse"])
    ap.add_argument("--areas", default=",".join(AREA_COORDS))
    args = ap.parse_args()
    if args.step == "build":
        build([a.strip() for a in args.areas.split(",") if a.strip()])
        return 0
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
