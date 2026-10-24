r"""Phase 114: review checks answerable from the frozen predictions, no refit.

WHY THIS EXISTS
---------------
Several Applied Energy referee points need no new model at all, only a different
cut of predictions that already exist and are frozen:
  R1.6/R5.3  does the gate contrast change in weather or demand regimes the
             evaluation window does contain (hot, cold, calm days; the evening ramp)?
  R1.9       do the early folds, trained on four months, drive the result?
  R1.10      how much of the near-floor band is the exact administrative floor?
  R3.4       what is the floor-detection gain in units an operator counts?
  R1.2       how much could a fresher pre-gate run than the headline add at most?
Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md
(section "Phase 114"), committed before this script ran.

Every contrast uses phase85's own statistics (imported, not re-implemented) on
phase85's capacity-weighted subset (all cw_ok flags true), so a number here and
the matching number in the manuscript differ only by the cut.

CLI (from the repository root):
  uv run python scripts/68_stored_prediction_checks.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase114")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P85 = Path(__file__).parent / "10_primary_ladder_cw.py"
P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
FLOOR_FULL = RUN / "phase66v2_floor_regime_v3_cw_full.json"
OUT_JSON = RUN / "phase114_stored_prediction_checks.json"
PRECOMMIT = "docs/research/2026-10-23_phase112_ae_review_response_precommit.md"

BASE, PRE, POST, COMPOSITE = "price_only", "msm_d2_12z_cw", "msm_d1_00z_cw", \
    "msm_gate_composite_cw"
LONG_TRAIN_FROM = date(2025, 3, 16)      # >= 365 days of training history
EVENING = (16, 20)                       # 16:00-20:30 JST inclusive by hour
FLOOR_EXACT, NEAR_FLOOR = 0.01, 0.5


def _load_p85():
    spec = importlib.util.spec_from_file_location("p85", P85)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p85"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pooled(per_day: pl.DataFrame, cell: str) -> tuple[np.ndarray, list[str]]:
    s = (per_day.with_columns((pl.col(f"mae_{BASE}") - pl.col(f"mae_{cell}")).alias("imp"))
         .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day"))
    return np.asarray(s["imp"], dtype=float), s["test_day"].to_list()


def _block(p85, panel: pl.DataFrame, label: str) -> dict:
    """Pre-gate improvement and gate contrast on a cut of the CW panel."""
    per_day = p85._per_day_mae(panel, [BASE, PRE, POST])
    d_pre, days = _pooled(per_day, PRE)
    d_post, _ = _pooled(per_day, POST)
    base_mae = float(per_day[f"mae_{BASE}"].mean())
    pre = p85._entry(d_pre)
    gate = p85._diff_entry(d_post, d_pre)
    return {"label": label, "n_days": len(days), "n_slots": panel.height,
            "price_only_mae": round(base_mae, 4),
            "pre_gate_improvement": pre,
            "pre_gate_relative_pct": round(100 * pre["improvement"] / base_mae, 3),
            "gate_contrast": gate,
            "gate_relative_pct": round(100 * gate["delta"] / base_mae, 3),
            "gate_ucb_relative_pct": round(100 * gate["ci95"][1] / base_mae, 3)}


def _holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    n, out, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * ps[i]))
        out[i] = running
    return out


def _daily_weather(p85, areas: list[str]) -> pl.DataFrame:
    """Area x day means of the REALISED capacity-weighted ERA5 temperature and wind.

    Realised weather defines the regime because it is exogenous to the price; a
    regime cut on the forecast would select on the information being evaluated.
    """
    t, w = "temperature_c", "wind_speed_kmh"
    frames = []
    for a in areas:
        e = p85._load_cw_era5(a)
        frames.append(e.select(
            pl.lit(a).alias("area"),
            pl.col("datetime_jst").dt.date().cast(pl.Utf8).alias("test_day"),
            pl.col(f"{t}_{p85.CW_ERA5_REAL}").alias("temp"),
            pl.col(f"{w}_{p85.CW_ERA5_REAL}").alias("wind")))
    return (pl.concat(frames).group_by("area", "test_day")
            .agg(pl.col("temp").mean(), pl.col("wind").mean()))


def regimes(p85, cw: pl.DataFrame) -> dict:
    areas = sorted(cw["area"].unique().to_list())
    wx = _daily_weather(p85, areas).join(
        cw.select("area", "test_day").unique(), on=["area", "test_day"], how="inner")
    q = wx.group_by("area").agg(pl.col("temp").quantile(0.95).alias("t95"),
                                pl.col("temp").quantile(0.05).alias("t05"),
                                pl.col("wind").quantile(0.10).alias("w10"))
    wx = wx.join(q, on="area")
    day_sets = {
        "hot_days": wx.filter(pl.col("temp") >= pl.col("t95")),
        "cold_days": wx.filter(pl.col("temp") <= pl.col("t05")),
        "low_wind_days": wx.filter(pl.col("wind") <= pl.col("w10")),
    }
    out: dict = {"definitions": {
        "hot_days": "area-day mean realised CW-ERA5 2-m temperature >= the area's p95 "
                    "over test days",
        "cold_days": "<= the area's p5",
        "low_wind_days": "area-day mean realised CW-ERA5 10-m wind <= the area's p10",
        "evening_ramp": "slots 16:00-20:30 JST",
        "high_price_slots": "realised price >= the area's p95 over test slots "
                            "(selected on the outcome: descriptive only)"}, "cuts": {}}
    for name, ds in day_sets.items():
        sub = cw.join(ds.select("area", "test_day"), on=["area", "test_day"], how="inner")
        out["cuts"][name] = _block(p85, sub, name)
    hr = cw["datetime_jst"].dt.hour()
    out["cuts"]["evening_ramp"] = _block(
        p85, cw.filter((hr >= EVENING[0]) & (hr <= EVENING[1])), "evening_ramp")
    p95 = cw.group_by("area").agg(pl.col("y_actual").quantile(0.95).alias("p95"))
    out["cuts"]["high_price_slots"] = _block(
        p85, cw.join(p95, on="area").filter(pl.col("y_actual") >= pl.col("p95")),
        "high_price_slots")
    names = list(out["cuts"])
    ps = [out["cuts"][n]["gate_contrast"]["dm_l3"]["p_hln"] for n in names]
    holm = _holm(ps)
    for n, p in zip(names, holm, strict=True):
        out["cuts"][n]["gate_p_holm"] = round(p, 6)
    sig = [n for n, p in zip(names, holm, strict=True) if p < 0.05]
    # Added 2026-10-24 (second referee read): the bound as a share of each cut's OWN
    # pre-gate gain, where that gain is distinguishable from zero. On the hottest and
    # coldest days it is not, so a share of it would mean nothing and is left out.
    shares = {n: round(100 * c["gate_contrast"]["ci95"][1]
                       / c["pre_gate_improvement"]["improvement"], 2)
              for n, c in out["cuts"].items()
              if c["pre_gate_improvement"]["ci95"][0] > 0}
    k = max(shares, key=shares.get)
    out["gate_ucb_share_of_own_gain_pct"] = {"by_cut": shares,
                                             "max": {"key": k, "pct": shares[k]},
                                             "raw_p_min": round(min(ps), 4)}
    out["gate_g114a"] = {
        "evaluated": True, "passes": not sig, "significant_after_holm": sig,
        "rule": "no regime's gate contrast significant after Holm across the five cuts",
        "if_fails": "report that freshness carries price-relevant content in that regime"}
    return out


def training_length(p85, cw: pl.DataFrame) -> dict:
    long_ = cw.filter(pl.col("test_day") >= LONG_TRAIN_FROM.isoformat())
    return {"from": LONG_TRAIN_FROM.isoformat(),
            "rule": "test days with at least 365 days of training history",
            **_block(p85, long_, "long_training")}


def threshold_composition(cw_all: pl.DataFrame) -> dict:
    y = cw_all["y_actual"]
    n_band = int((y <= NEAR_FLOOR).sum())
    n_exact = int((y <= FLOOR_EXACT + 1e-9).sum())
    full = json.loads(FLOOR_FULL.read_text())
    pr = {}
    for pk, label in (("P2cw_pregate_vs_price", "pre_gate_vs_price_only"),
                      ("P4cw_gate_timing", "gate_timing")):
        a = full["pairings"][pk]["allhours"]
        pr[label] = {t: a[t]["pooled_macro_avg"]["d_pr_auc"] for t in a}
    return {"panel": "the full ladder panel (all slots), not only the CW-complete cut",
            "n_slots": int(cw_all.height), "n_near_floor_band": n_band,
            "n_exact_floor": n_exact, "share_exact_of_band": round(n_exact / n_band, 4),
            "n_band_above_floor": n_band - n_exact,
            "pr_auc_delta_by_threshold": pr,
            "source": FLOOR_FULL.name}


def operational(cw_all: pl.DataFrame) -> dict:
    """Additional floor slots per year flagged at a fixed false-alarm rate."""
    full = json.loads(FLOOR_FULL.read_text())
    a = full["pairings"]["P2cw_pregate_vs_price"]["allhours"]["0p5"]
    n_days = cw_all["test_day"].n_unique()
    years = n_days / 365.25
    per_area, tot_base, tot_op = {}, 0.0, 0.0
    for area, rec in a.items():
        if area in ("pooled_macro_avg", "pooled_slotwise") or not isinstance(rec, dict):
            continue
        r = rec["d_recall_at_fpr"]["0.01"]
        pos_per_year = rec["n_positives"] / years
        base, op = r["level_b"] * pos_per_year, r["level_a"] * pos_per_year
        per_area[area] = {"floor_slots_per_year": round(pos_per_year, 1),
                          "recall_price_only": round(r["level_b"], 4),
                          "recall_pre_gate": round(r["level_a"], 4),
                          "extra_flagged_per_year": round(op - base, 1)}
        tot_base += base
        tot_op += op
    return {"false_alarm_rate": 0.01, "threshold": NEAR_FLOOR, "years": round(years, 3),
            "per_area": per_area,
            "all_areas_extra_flagged_per_year": round(tot_op - tot_base, 1),
            "all_areas_floor_slots_per_year": round(
                sum(v["floor_slots_per_year"] for v in per_area.values()), 1),
            "unit": "half-hour slots; not a monetary quantity"}


def freshness_bound() -> dict:
    j = json.loads(P85_JSON.read_text())
    arm = j["arm_cw_primary"]
    g = arm["gate46_47_48_gate_timing"]
    tiers = arm["tiers"]
    return {"gate_contrast": g.get("delta"), "gate_ci95": g.get("ci95"),
            "gate_ucb95": (g.get("ci95") or [None, None])[1],
            "headline": tiers[PRE]["improvement"],
            "freshest_admissible_per_slot": tiers[COMPOSITE]["improvement"],
            "composite_minus_headline": round(
                tiers[COMPOSITE]["improvement"] - tiers[PRE]["improvement"], 4),
            "reading": "any pre-gate run fresher than D-2 12 UTC lies between it and the "
                       "D-1 00 UTC run in vintage; if value is monotone in freshness, "
                       "the 95% upper bound on the latter's increment bounds the former's"}


def main() -> int:
    p85 = _load_p85()
    panel = pl.read_parquet(P85_PARQUET)
    flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    cw = panel.filter(pl.all_horizontal(pl.col(f) for f in flags))
    out = {"phase": "114", "precommit": PRECOMMIT, "git_sha": _git_sha(),
           "source": P85_PARQUET.name,
           "full_panel": _block(p85, cw, "full CW panel (reproduces phase85)"),
           "regimes": regimes(p85, cw),
           "training_length": training_length(p85, cw),
           "threshold_composition": threshold_composition(panel),
           "operational": operational(panel),
           "freshness_bound": freshness_bound()}
    out["gates"] = {"g114a_regimes": out["regimes"].pop("gate_g114a")}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
