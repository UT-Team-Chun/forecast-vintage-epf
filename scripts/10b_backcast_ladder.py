r"""Phase 112: the ladder's two key contrasts on a 2021-24 window with scarcity pricing.

WHY THIS EXISTS
---------------
Applied Energy referees R1.6 and R5.3: the primary window (2024-08..2026-08) is a
photovoltaic-surplus period with almost no scarcity pricing, so it cannot say
whether a fresher weather run matters more when prices are set by tight supply.
R1.9 asked for a longer window. This re-runs the primary design, unchanged, on
the 1,308 test days that precede it, a span containing the January 2021 crisis,
the tight winter of 2021-22 and the summer-2022 crunch.
Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md
(section "Phase 112"), committed before this script ran.

WHAT IS IDENTICAL TO PHASE 85
-----------------------------
The learner (XGBoost 300 trees, depth 4, learning rate 0.05, hist, seed 2026),
the features (price lags 48/96/336/672 slots, calendar, the four weather
channels of one tier), the weather-to-slot join (hour floor), the fold rule
(expanding window, daily step, first test day 14 days after the initial training
end) and every statistic (phase85's functions are imported, not re-implemented).

WHAT DIFFERS, AND WHY
---------------------
- Dates: panel from 2020-08-01, initial training end 2020-12-18, test days
  2021-01-01..2024-07-31.
- Cells: only the five the precommit names. No ERA5 tiers, so the row set is
  set by these cells alone.
- Weather source: the phase112 backcast caches (jma_gpv_backcast{,_early}),
  decoded with an extractor verified bit-identical to the loader. The 2024-09
  capacity weights are applied here exactly as phase85 applies them.

CLI (from the repository root):
  uv run python scripts/10b_backcast_ladder.py --step build --areas kyushu
  uv run python ... --step analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import xgboost as xgb

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points
from alg.core.epf.data.jepx_loader import load_jepx_spot
from alg.core.epf.features.calendar import make_calendar_features
from alg.core.epf.features.exogenous import DEFAULT_WEATHER_COLS, attach_hourly_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase112")
JST = ZoneInfo("Asia/Tokyo")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
P85 = Path(__file__).parent / "10_primary_ladder_cw.py"
P115_JSON = RUN / "phase115_dissemination_bounds.json"
ROOTS = (RAW / "jma_gpv_backcast_early", RAW / "jma_gpv_backcast")
CKPT = RUN / "phase112_backcast_checkpoints"
OUT_JSON = RUN / "phase112_backcast.json"
PRECOMMIT = "docs/research/2026-10-23_phase112_ae_review_response_precommit.md"

PANEL_START = date(2020, 8, 1)
INITIAL_TRAIN_END = datetime(2020, 12, 18, tzinfo=JST)
VAL_WINDOW, STEP = timedelta(days=14), timedelta(days=1)
LAST_TEST = date(2024, 7, 31)
MAX_FOLDS = 1400
SEED = 2026
MIN_TRAIN_ROWS = 100
WEIGHT_VINTAGE = "2024-09"
PLANS = ("msm_d2_12z", "msm_d1_00z")
BASE, PRE, POST = "price_only", "msm_d2_12z_cw", "msm_d1_00z_cw"
PRE_SP, POST_SP = "msm_d2_12z", "msm_d1_00z"
CELLS = (BASE, PRE, POST, PRE_SP, POST_SP)
EPISODES = {"E1_jan2021": (date(2021, 1, 1), date(2021, 1, 31)),
            "E2_winter2022": (date(2022, 1, 1), date(2022, 3, 31)),
            "E3_summer2022": (date(2022, 6, 1), date(2022, 9, 30))}
SCARCITY = {"S30": 30.0, "S50": 50.0}


def _load_p85():
    spec = importlib.util.spec_from_file_location("p85", P85)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p85"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ------------------------------------------------------------------ weather
def _read_point(plan: str, key: str) -> pl.DataFrame:
    frames = [pl.read_parquet(r / plan / f"{key}.parquet") for r in ROOTS
              if (r / plan / f"{key}.parquet").exists()]
    if not frames:
        msg = f"no backcast cache for {plan}/{key} under {ROOTS}"
        raise FileNotFoundError(msg)
    return (pl.concat(frames).select("datetime_jst", *DEFAULT_WEATHER_COLS)
            .unique("datetime_jst", keep="last").sort("datetime_jst"))


def _tier_frames(area: str) -> dict[str, pl.DataFrame]:
    """SP and CW hourly frames for both plans, columns suffixed by cell name."""
    out: dict[str, pl.DataFrame] = {}
    pts = area_points(WEIGHT_VINTAGE)[area]
    for plan in PLANS:
        sp = _read_point(plan, f"sp__{area}")
        out[plan] = sp.rename({c: f"{c}_{plan}" for c in DEFAULT_WEATHER_COLS})
        acc: pl.DataFrame | None = None
        for i, p in enumerate(pts):
            df = _read_point(plan, f"cw__{area}__p{i:02d}").with_columns(
                *[(pl.col(c) * p.weight).alias(c) for c in DEFAULT_WEATHER_COLS])
            acc = df if acc is None else (
                acc.join(df, on="datetime_jst", how="inner", suffix="_r")
                   .with_columns(*[(pl.col(c) + pl.col(f"{c}_r")).alias(c)
                                   for c in DEFAULT_WEATHER_COLS])
                   .select("datetime_jst", *DEFAULT_WEATHER_COLS))
        cw = f"{plan}_cw"
        out[cw] = acc.sort("datetime_jst").rename(
            {c: f"{c}_{cw}" for c in DEFAULT_WEATHER_COLS})
    return out


# ------------------------------------------------------------------ build
def _design(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    out = df.sort("datetime_jst")
    for lag in (48, 96, 336, 672):
        out = out.with_columns(pl.col("price_yen_kwh").shift(lag).alias(f"lag_{lag}"))
    out = make_calendar_features(out, datetime_col="datetime_jst")
    base = [*(f"lag_{lag}" for lag in (48, 96, 336, 672)),
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
            "is_holiday"]
    cells = {BASE: list(base)}
    for t in (PRE, POST, PRE_SP, POST_SP):
        cells[t] = [*base, *(f"{c}_{t}" for c in DEFAULT_WEATHER_COLS)]
    return out, cells


def _fit(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """phase85 ``_fit`` verbatim."""
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=4, tree_method="hist")
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


def build_area(area: str, first_test: date, last_test: date) -> None:
    """Folds whose test day lies in [first_test, last_test], one checkpoint per segment.

    Segmenting is exact, not an approximation: every fold trains only on slots before
    its test day and every feature is row-local, so a 2021 fold sees the same rows
    whether or not the panel already holds 2022-24 weather. That lets the 2021 folds
    run as soon as the early archive sweep lands.
    """
    ckpt = CKPT / f"{area}__{first_test.isoformat()}_{last_test.isoformat()}.parquet"
    if ckpt.exists():
        log.info("checkpoint hit %s", ckpt.name)
        return
    price = (load_jepx_spot(years=[2020, 2021, 2022, 2023, 2024], cache_dir=RAW)
             .filter(pl.col("area") == area)
             .select("datetime_jst", "area", "price_yen_kwh").sort("datetime_jst"))
    df = price
    for name, fr in _tier_frames(area).items():
        df = attach_hourly_features(df, fr,
                                    feature_cols=[f"{c}_{name}" for c in DEFAULT_WEATHER_COLS])
    df, cells = _design(df)
    need = sorted({c for cols in cells.values() for c in cols} | {"price_yen_kwh"})
    df = (df.drop_nulls(subset=need)
            .filter(pl.col("datetime_jst").dt.date().is_between(PANEL_START, LAST_TEST))
            .sort("datetime_jst"))
    log.info("%s panel: %d slots %s..%s", area, df.height, df["datetime_jst"].min(),
             df["datetime_jst"].max())
    rows: list[dict] = []
    t0, n = time.time(), 0
    for k in range(MAX_FOLDS):
        test_day = (INITIAL_TRAIN_END + STEP * k + VAL_WINDOW).date()
        if test_day > last_test:
            break
        if test_day < first_test:
            continue
        train = df.filter(pl.col("datetime_jst").dt.date() < test_day)
        test = df.filter(pl.col("datetime_jst").dt.date() == test_day).sort("datetime_jst")
        if train.height < MIN_TRAIN_ROWS or test.height == 0:
            continue
        preds = {name: _fit(train, test, cols) for name, cols in cells.items()}
        y = test["price_yen_kwh"].to_numpy().astype(float)
        dts = test["datetime_jst"].to_list()
        rows.extend({"area": area, "datetime_jst": dts[s], "test_day": test_day.isoformat(),
                     "y_actual": y[s], **{f"y_pred_{c}": float(preds[c][s]) for c in cells}}
                    for s in range(test.height))
        n += 1
        if n % 100 == 0:
            log.info("%s fold %d (test %s) | %.2f s/fold", area, n, test_day,
                     (time.time() - t0) / n)
    CKPT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(ckpt)
    log.info("%s done: %d folds, %d slots, %.0f s", area, n, len(rows), time.time() - t0)


# ------------------------------------------------------------------ overlap gate
FROZEN_SP = RAW / "jma_gpv_exactrun"
FROZEN_MP = RAW / "jma_gpv_exactrun_mp"
SP_COORDS = {"hokkaido": (43.07, 141.35), "tohoku": (38.27, 140.87),
             "tokyo": (35.68, 139.76), "chubu": (35.18, 136.91),
             "hokuriku": (36.56, 136.66), "kansai": (34.69, 135.50),
             "chugoku": (34.40, 132.46), "shikoku": (34.34, 134.05),
             "kyushu": (33.59, 130.40)}


def verify_overlap() -> dict:
    """Gate 112-0: on 2024-03-01..07-31 the backcast caches must equal the frozen ones.

    The frozen exactrun caches were extracted by the loader in August 2026 and the
    primary ladder reads them; the backcast caches were extracted by this phase's
    hoisted-index decoder. Equality on the overlap is the strongest available
    evidence that the two eras of the combined record measure the same thing.
    """
    lo, hi = date(2024, 3, 1), LAST_TEST
    worst, n = 0.0, 0
    for plan in PLANS:
        for area, (la, lon) in SP_COORDS.items():
            f = FROZEN_SP / plan / f"jma_gpv_{plan}_{area}_{la:.4f}_{lon:.4f}.parquet"
            ref = pl.read_parquet(f).select("datetime_jst", *DEFAULT_WEATHER_COLS)
            new = _read_point(plan, f"sp__{area}")
            j = ref.join(new, on="datetime_jst", suffix="_n").filter(
                pl.col("datetime_jst").dt.date().is_between(lo, hi))
            n += j.height
            for c in DEFAULT_WEATHER_COLS:
                worst = max(worst, float((j[c] - j[f"{c}_n"]).abs().max() or 0.0))
            for i, p in enumerate(area_points(WEIGHT_VINTAGE)[area]):
                f = (FROZEN_MP / plan /
                     f"jma_gpv_{plan}_{area}_p{i:02d}_{p.lat:.4f}_{p.lon:.4f}.parquet")
                ref = pl.read_parquet(f).select("datetime_jst", *DEFAULT_WEATHER_COLS)
                new = _read_point(plan, f"cw__{area}__p{i:02d}")
                j = ref.join(new, on="datetime_jst", suffix="_n").filter(
                    pl.col("datetime_jst").dt.date().is_between(lo, hi))
                n += j.height
                for c in DEFAULT_WEATHER_COLS:
                    worst = max(worst, float((j[c] - j[f"{c}_n"]).abs().max() or 0.0))
    rec = {"window": [lo.isoformat(), hi.isoformat()], "n_point_hours": n,
           "max_abs_diff": worst, "tolerance": 1e-9,
           "verdict": "PASS" if n > 0 and worst <= 1e-9 else "FAIL"}
    log.info("gate 112-0 overlap: %s", rec)
    return rec


# ------------------------------------------------------------------ analyse
def _block(p85, panel: pl.DataFrame, base: str, pre: str, post: str, label: str) -> dict:
    per_day = p85._per_day_mae(panel, [base, pre, post])

    def pooled(cell: str) -> np.ndarray:
        return np.asarray((per_day.with_columns(
            (pl.col(f"mae_{base}") - pl.col(f"mae_{cell}")).alias("imp"))
            .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day")["imp"]),
            dtype=float)

    d_pre, d_post = pooled(pre), pooled(post)
    base_mae = float(per_day[f"mae_{base}"].mean())
    pre_e = p85._entry(d_pre)
    gate = p85._diff_entry(d_post, d_pre, p85._margins_for(pre_e["improvement"]))
    return {"label": label, "n_days": len(d_pre), "n_slots": panel.height,
            "price_only_mae": round(base_mae, 4), "pre_gate_improvement": pre_e,
            "pre_gate_relative_pct": round(100 * pre_e["improvement"] / base_mae, 3),
            "gate_contrast": gate,
            "gate_relative_pct": round(100 * gate["delta"] / base_mae, 3),
            "gate_ci95_relative_pct": [round(100 * v / base_mae, 3) for v in gate["ci95"]],
            "gate_ucb_share_of_pre_gate": (round(gate["ci95"][1] / pre_e["improvement"], 4)
                                           if pre_e["improvement"] > 0 else None)}


def analyse() -> int:
    p85 = _load_p85()
    files = sorted(CKPT.glob("*.parquet"))
    if not files:
        log.error("no checkpoints in %s", CKPT)
        return 2
    panel = pl.concat([pl.read_parquet(f) for f in files])
    d = panel["datetime_jst"].dt.date()
    in_e = pl.Series([False] * panel.height)
    for lo, hi in EPISODES.values():
        in_e = in_e | d.is_between(lo, hi)
    out: dict = {"phase": "112", "precommit": PRECOMMIT, "git_sha": _git_sha(),
                 "panel": {"areas": sorted(panel["area"].unique().to_list()),
                           "n_test_days": panel["test_day"].n_unique(),
                           "n_slots": panel.height,
                           "first_test_day": panel["test_day"].min(),
                           "last_test_day": panel["test_day"].max(),
                           "scarcity_pct": {k: round(100 * float((panel["y_actual"] >= v).mean()), 3)
                                            for k, v in SCARCITY.items()},
                           "near_floor_pct": round(100 * float((panel["y_actual"] <= 0.5).mean()), 3)},
                 "cw": {}, "sp": {}}
    cuts = {"all": panel,
            "episodes": panel.filter(in_e),
            **{name: panel.filter(d.is_between(lo, hi)) for name, (lo, hi) in EPISODES.items()},
            **{k: panel.filter(pl.col("y_actual") >= v) for k, v in SCARCITY.items()},
            "non_episode": panel.filter(~in_e)}
    for name, sub in cuts.items():
        out["cw"][name] = _block(p85, sub, BASE, PRE, POST, f"CW {name}")
        out["sp"][name] = _block(p85, sub, BASE, PRE_SP, POST_SP, f"SP {name}")
    ep = out["cw"]["episodes"]
    lo_rel, hi_rel = ep["gate_ci95_relative_pct"]
    if hi_rel < 1.0 and ep["gate_contrast"]["ci95"][0] <= 0:
        branch = "i"
    elif ep["gate_contrast"]["ci95"][0] > 0:
        branch = "ii-a" if hi_rel < 1.0 else "ii"
    else:
        branch = "iii"
    allc = out["cw"]["all"]["pre_gate_improvement"]
    out["gate112_0_overlap"] = verify_overlap()
    out["gates"] = {
        "g112a_headline_replicates": {
            "evaluated": True, "passes": bool(allc["ci95"][0] > 0),
            "rule": "backcast pre-gate CW improvement 95% CI lower bound > 0",
            "if_fails": "restrict the headline in the Abstract and Conclusions to the primary window"},
        "g112b_freshness_in_scarcity": {
            "evaluated": True, "branch": branch, "passes": branch == "i",
            "episode_gate_relative_ci95_pct": [lo_rel, hi_rel],
            "rule": "(i) CI of the episode gate contrast, relative to price-only MAE, below "
                    "+1%; (ii) CI lower bound > 0; (iii) otherwise inconclusive",
            "if_fails": "branch ii: state that the near-gate flatness is regime-dependent and "
                        "qualify the Abstract; branch iii: report the interval as inconclusive"}}
    if P115_JSON.exists():
        m = json.loads(P115_JSON.read_text()).get("measured_backcast", {})
        out["gates"]["g112d_dissemination"] = {
            "evaluated": bool(m.get("backcast_files")),
            "passes": m.get("files_before_gate") == 0,
            "files": m.get("backcast_files"), "days": m.get("backcast_days"),
            "files_before_gate": m.get("files_before_gate"),
            "rule": "no file of the D-1 00 UTC run written before the gate, 2020-24",
            "if_fails": "rewrite Section 3 with the count of pre-gate files"}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    log.info("wrote %s", OUT_JSON)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=("build", "analyse", "verify-overlap"), default="build")
    ap.add_argument("--first-test", default="2021-01-01")
    ap.add_argument("--last-test", default=LAST_TEST.isoformat())
    ap.add_argument("--areas", nargs="+",
                    default=["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai",
                             "chugoku", "shikoku", "kyushu"])
    a = ap.parse_args()
    if a.step == "analyse":
        return analyse()
    if a.step == "verify-overlap":
        return 0 if verify_overlap()["verdict"] == "PASS" else 1
    for area in a.areas:
        build_area(area, date.fromisoformat(a.first_test), date.fromisoformat(a.last_test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
