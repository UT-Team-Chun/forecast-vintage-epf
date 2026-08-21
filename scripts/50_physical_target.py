r"""Phase 89: the physical target, rebuilt without look-ahead.

WHAT PHASE 79 GOT WRONG
-----------------------
phase79 scored each weather vintage against the operators' photovoltaic
potential by fitting an origin-OLS scale factor per (tier, area) -- on the
FULL sample, i.e. on exactly the rows it then scored. That is 104 free
parameters estimated in-sample, and it is a look-ahead step in the headline of
a paper whose entire argument is that evaluation must respect what was knowable
at a deadline. The JSON said so itself ("origin OLS, full sample, per
tier-area") and nobody read it.

Three smaller defects travelled with it, all found by reading the code and the
raw files rather than the summary:

* the OCCTO timestamp shift of -30 min is right for Kyushu only. Seven of the
  eight TSO files label the interval START; only Kyushu labels the end. The
  uniform shift displaced the target by one half-hour slot for seven areas.
* the reported MAE is an unweighted mean ACROSS the eight areas, in MW. Per
  area it ranges from ~182 MW (Hokuriku) to ~2650 MW (Tokyo), so the pooled
  figure is a per-area average, not a fleet total -- and the manuscript quoted
  it bare.
* the panel stops at 791 days because Kyushu's 2026-06 rows are missing from
  the merged parquet even though the CSV is cached and registered; the loader
  swallows per-month failures.

WHAT THIS SCRIPT DOES
---------------------
Gates and branches are fixed in
docs/research/2026-08-13_phase89_physical_target_precommit.md, before numbers.

1. cross-fits the scale: the factor applied on test day t is estimated only on
   days before t (expanding window, the ladder's own discipline), with a
   burn-in that is reported rather than hidden;
2. reports the in-sample value beside it, so the cost of the old look-ahead is
   visible instead of silently corrected (Gate 53);
3. scores three target definitions -- curtailment restored (primary),
   curtailment-excluded, and raw recorded output (Gate 52);
4. shifts OCCTO stamps per area, and reports the old uniform convention beside
   it (Gate 54);
5. restricts the primary comparison to the panel the price arm actually covers
   (Gate 51), and reports the wider physical-only panel as a sensitivity;
6. persists per-slot output so the mechanism test (phase90) does not have to
   recompute any of it.

CLI (from the repository root):
  uv run python scripts/50_physical_target.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.area_points import area_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase89")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
OCCTO_PARQUET = RUN / "phase25.3_occto_generation.parquet"
# Kyushu's final month was missing from the merged panel: its cached CSV carries
# a decimal in a column polars inferred as integer from the first 100 rows, the
# read raised, and the loader's per-month handler swallowed it. The loader is
# fixed, but the frozen panel is NOT rewritten -- landed analyses must stay
# reproducible from their own input -- so the recovered month lives in a
# disjoint extension file and is concatenated here, the same freeze discipline
# the ladder's panel extension uses.
OCCTO_EXT = RUN / "phase25.3_occto_generation_ext.parquet"
PRICE_PANEL = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase89_physical_target.json"
OUT_SLOTS = RUN / "phase89_physical_per_slot.parquet"
PRECOMMIT = "docs/research/2026-08-13_phase89_physical_target_precommit.md"

SEED, N_BOOT, BOOT_BLOCK = 2026, 10_000, 7
DAYTIME_HOURS = (9, 15)
PV_TEMP_DERATE_PER_C, PV_TEMP_REF_C = 0.004, 25.0
WEIGHT_VINTAGE = "2024-09"
MIN_TRAIN_ROWS = 200          # rows needed before a cross-fitted scale is used
EXACTRUN_MP_DIR = RAW / "jma_gpv_exactrun_mp"
OM_MP_DIR = RAW / "open_meteo_mp"

# Interval-label convention, verified against the raw monthly CSVs: seven TSOs
# stamp the interval START, Kyushu stamps the END. phase79 applied -30 min to
# all eight.
STAMP_SHIFT_MIN = {"kyushu": -30, "hokkaido": 0, "tohoku": 0, "tokyo": 0,
                   "hokuriku": 0, "kansai": 0, "chugoku": 0, "shikoku": 0}
LEGACY_SHIFT_MIN = -30

API_TO_CANON = {"temperature_2m": "temperature_c", "cloud_cover": "cloud_cover_pct",
                "shortwave_radiation": "shortwave_radiation_wm2",
                "wind_speed_10m": "wind_speed_kmh"}
SW, TC = "shortwave_radiation_wm2", "temperature_c"

TIERS = ("era5_cw_stale", "msm_d2_12z_cw", "msm_gate_composite_cw",
         "msm_d1_00z_cw", "oracle_cw")
PRE_GATE, POST_GATE = "msm_d2_12z_cw", "msm_d1_00z_cw"

TARGETS = ("potential", "curtail_excluded", "recorded")


def _j(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ------------------------------------------------------------------ loaders
def _parse_om_json(path: Path) -> pl.DataFrame:
    h = json.loads(path.read_text())["hourly"]
    return pl.DataFrame({
        "datetime_jst": [datetime.fromisoformat(t) for t in h["time"]],
        **{canon: [float(x) if x is not None else None for x in h[api]]
           for api, canon in API_TO_CANON.items()},
    }).with_columns(pl.col("datetime_jst").dt.replace_time_zone("Asia/Tokyo")
                    .dt.cast_time_unit("us"))


def _derate(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col(SW) * (1.0 - PV_TEMP_DERATE_PER_C * (pl.col(TC) - PV_TEMP_REF_C)))
        .alias("pv_proxy"))


def _weighted_points(area: str, plan: str) -> pl.DataFrame:
    """Per-point derate, then capacity weight -- the Gate-35 fleet-physical order."""
    acc: pl.DataFrame | None = None
    for i, p in enumerate(area_points(WEIGHT_VINTAGE)[area]):
        key = f"{area}_p{i:02d}"
        if plan == "era5":
            files = sorted(OM_MP_DIR.glob(
                f"era5_mp_{area}_{f'{p.lat:.4f}'.replace('.', 'p')}_"
                f"{f'{p.lon:.4f}'.replace('.', 'p')}_*.json"))
            if not files:
                msg = f"missing era5 multi-point cache for {key}"
                raise FileNotFoundError(msg)
            df = (pl.concat([_parse_om_json(f) for f in files])
                  .unique("datetime_jst").sort("datetime_jst"))
        else:
            path = (EXACTRUN_MP_DIR / plan
                    / f"jma_gpv_{plan}_{key}_{p.lat:.4f}_{p.lon:.4f}.parquet")
            if not path.exists():
                msg = f"missing multi-point cache {path}"
                raise FileNotFoundError(msg)
            df = pl.read_parquet(path)
        df = _derate(df).select("datetime_jst",
                                (pl.col("pv_proxy") * p.weight).alias("pv_proxy"))
        acc = df if acc is None else (
            acc.join(df, on="datetime_jst", how="inner", suffix="_r")
               .select("datetime_jst",
                       (pl.col("pv_proxy") + pl.col("pv_proxy_r")).alias("pv_proxy")))
    return acc.sort("datetime_jst")


def _tier_frame(tier: str, area: str) -> pl.DataFrame:
    if tier == "era5_cw_stale":
        return _weighted_points(area, "era5").with_columns(
            pl.col("pv_proxy").shift(24))
    if tier == "oracle_cw":
        return _weighted_points(area, "era5")
    if tier == "msm_gate_composite_cw":
        g = _weighted_points(area, "msm_d2_21z").filter(
            pl.col("datetime_jst").dt.hour() <= 21)
        a = _weighted_points(area, "msm_d2_12z").filter(
            pl.col("datetime_jst").dt.hour() >= 22)
        return pl.concat([g, a]).unique("datetime_jst").sort("datetime_jst")
    return _weighted_points(area, tier.removesuffix("_cw"))


# ------------------------------------------------------------------ scoring
def _slots(proxy: pl.DataFrame, occ_area: pl.DataFrame,
           target_col: str) -> pl.DataFrame:
    """Daylight slots with the proxy joined and the chosen target selected."""
    return (occ_area.join_asof(proxy.sort("datetime_jst"), on="datetime_jst",
                               strategy="backward", tolerance="1h")
            .filter(pl.col("datetime_jst").dt.hour().is_between(*DAYTIME_HOURS))
            # drop_nulls does NOT drop NaN, and a single NaN poisons the
            # expanding-window OLS sums from that row onward, silently truncating
            # every later day. Require finiteness explicitly.
            .drop_nulls(["pv_proxy", target_col])
            .filter(pl.col("pv_proxy").is_finite() & pl.col(target_col).is_finite())
            .with_columns(pl.col("datetime_jst").dt.date().alias("day"))
            .sort("datetime_jst"))


def _scale_in_sample(df: pl.DataFrame, target_col: str) -> float:
    pred = np.asarray(df["pv_proxy"], dtype=float)
    obs = np.asarray(df[target_col], dtype=float)
    return float(pred @ obs / (pred @ pred))


def _cross_fitted(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Expanding-window origin-OLS scale: day t uses only days before t.

    The closed form of an origin-OLS slope is a ratio of two running sums, so
    the whole path costs one pass: for day t the scale is
    (sum of pred*obs before t) / (sum of pred^2 before t). Days before the
    training minimum are dropped and counted as burn-in rather than scored with
    a scale that saw them.
    """
    pred = np.asarray(df["pv_proxy"], dtype=float)
    obs = np.asarray(df[target_col], dtype=float)
    day = df["day"].to_numpy()
    # cumulative sums of everything strictly before each day boundary
    order = np.argsort(day, kind="stable")
    pred, obs, day = pred[order], obs[order], day[order]
    xy = np.cumsum(pred * obs)
    xx = np.cumsum(pred * pred)
    n = np.arange(1, len(pred) + 1)
    # index of the last row of the previous day, per row
    uniq, first = np.unique(day, return_index=True)
    prev_end = np.empty(len(pred), dtype=int)
    for k, start in enumerate(first):
        stop = first[k + 1] if k + 1 < len(first) else len(pred)
        prev_end[start:stop] = start - 1
    ok = prev_end >= 0
    scale = np.full(len(pred), np.nan)
    enough = ok & (n[prev_end] >= MIN_TRAIN_ROWS)
    scale[enough] = xy[prev_end][enough] / xx[prev_end][enough]
    return (df[order.tolist()]
            .with_columns(pl.Series("scale_cf", scale))
            .with_columns((pl.col("pv_proxy") * pl.col("scale_cf")
                           - pl.col(target_col)).abs().alias("abs_err_cf")))


def _boot_ci(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(d)
    nb = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    means = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _pool(per_area_daily: dict[str, pl.DataFrame], col: str) -> pl.DataFrame:
    frames = [d.select("day", pl.col(col).alias(f"m_{a}"))
              for a, d in per_area_daily.items()]
    pool = frames[0]
    for f in frames[1:]:
        pool = pool.join(f, on="day", how="inner")
    return pool.with_columns(
        pl.mean_horizontal([c for c in pool.columns if c != "day"])
        .alias("pooled")).sort("day")


def main() -> int:
    if not OCCTO_PARQUET.exists():
        log.error("missing %s", OCCTO_PARQUET)
        return 2

    occ_raw = pl.read_parquet(OCCTO_PARQUET)
    if OCCTO_EXT.exists():
        ext = pl.read_parquet(OCCTO_EXT).select(occ_raw.columns)
        overlap = ext.join(occ_raw.select("area", "datetime_jst"),
                           on=["area", "datetime_jst"], how="inner").height
        if overlap:
            msg = (f"extension overlaps the frozen panel on {overlap} rows; the "
                   "frozen rows are authoritative and the extension must be "
                   "strictly after their reach")
            raise ValueError(msg)
        occ_raw = pl.concat([occ_raw, ext])
        log.info("frozen panel + extension: %d rows (+%d)", occ_raw.height, ext.height)
    areas = sorted(a for a in occ_raw["area"].unique().to_list()
                   if a in {*STAMP_SHIFT_MIN})
    log.info("OCCTO areas: %s", areas)

    # the price panel's day set, for the common-panel restriction
    # the price panel stores test_day as a STRING; the physical panel's day is a
    # Date. Compare on the ISO string so the intersection is type-safe.
    price_days = set(pl.read_parquet(PRICE_PANEL, columns=["test_day"])
                     ["test_day"].unique().to_list())
    log.info("price panel days: %d", len(price_days))

    out: dict = {
        "phase": 89, "title": "Physical target, cross-fitted and common-panel",
        "precommit_doc": PRECOMMIT,
        "conventions": {
            "daylight_hours_jst": list(DAYTIME_HOURS),
            # The filter is INCLUSIVE on both bounds and the slots are
            # half-hourly labelled by start, so record the count it implies.
            # Without it the window's prose understated itself by an hour and
            # nothing could catch it: 09:00--15:00 instead of 09:00--15:30.
            "daylight_slots_per_area_day": 2 * (DAYTIME_HOURS[1] - DAYTIME_HOURS[0] + 1),
            "derate": f"shortwave x (1 - {PV_TEMP_DERATE_PER_C} x (T - {PV_TEMP_REF_C}))",
            "weighting": f"per-point derate then capacity weight, vintage {WEIGHT_VINTAGE}",
            "scaling": "origin OLS, EXPANDING WINDOW (day t uses days before t)",
            "scaling_superseded": "phase79 used origin OLS on the full sample",
            "stamp_shift_min": STAMP_SHIFT_MIN,
            "stamp_shift_superseded": f"phase79 applied {LEGACY_SHIFT_MIN} to all areas",
            "unit": ("MW; the pooled figure is the unweighted MEAN ACROSS AREAS of "
                     "each area's daylight-slot MAE, not a fleet total"),
            "min_train_rows": MIN_TRAIN_ROWS,
        },
        "targets": {}, "gates": {},
    }

    slot_rows: list[pl.DataFrame] = []
    for target in TARGETS:
        tgt_out: dict = {"tiers": {}}
        for shift_name, shift_map in (("per_area", STAMP_SHIFT_MIN),
                                      ("legacy_uniform",
                                       dict.fromkeys(areas, LEGACY_SHIFT_MIN))):
            if target != "potential" and shift_name == "legacy_uniform":
                continue  # the stamp sensitivity is run on the primary target only
            occ_parts = []
            for a in areas:
                q = (occ_raw.filter(pl.col("area") == a)
                     .with_columns(pl.col("datetime_jst")
                                   .dt.offset_by(f"{shift_map[a]}m")))
                curt = pl.col("pv_curtailment_mw").fill_null(0.0)
                if target == "potential":
                    q = q.with_columns((pl.col("pv_mw") + curt).alias("target_mw"))
                elif target == "recorded":
                    q = q.with_columns(pl.col("pv_mw").alias("target_mw"))
                else:
                    q = q.filter(curt <= 0).with_columns(
                        pl.col("pv_mw").alias("target_mw"))
                occ_parts.append(q.select("area", "datetime_jst", "target_mw",
                                          curt.alias("curtail_mw")))
            occ = pl.concat(occ_parts)

            missing_cells: list[str] = []
            for tier in TIERS:
                per_area_daily, per_area_meta = {}, {}
                for area in areas:
                    try:
                        proxy = _tier_frame(tier, area)
                    except FileNotFoundError as exc:
                        # A warning and a `continue` here used to let the pooled
                        # figure rest on fewer areas than the paper reports, with
                        # nothing downstream noticing: the count is asserted after
                        # the loop instead of trusted.
                        log.error("%s/%s missing: %s", tier, area, exc)
                        missing_cells.append(f"{tier}/{area}")
                        continue
                    df = _slots(proxy, occ.filter(pl.col("area") == area),
                                "target_mw")
                    if df.height < MIN_TRAIN_ROWS:
                        continue
                    s_is = _scale_in_sample(df, "target_mw")
                    df = _cross_fitted(df, "target_mw")
                    df = df.with_columns(
                        ((pl.col("pv_proxy") * s_is - pl.col("target_mw"))
                         .abs()).alias("abs_err_is"))
                    # NaN is not null in polars: the burn-in rows carry a
                    # NaN scale and must be filtered on finiteness.
                    scored = df.filter(pl.col("abs_err_cf").is_finite())
                    if scored.is_empty():
                        continue
                    daily = (scored.group_by("day")
                             .agg(pl.col("abs_err_cf").mean().alias("mae_cf"),
                                  pl.col("abs_err_is").mean().alias("mae_is"))
                             .sort("day"))
                    per_area_daily[area] = daily
                    per_area_meta[area] = {
                        "mae_cf_mw": _j(float(daily["mae_cf"].mean()), 1),
                        "mae_is_mw": _j(float(daily["mae_is"].mean()), 1),
                        "n_days": daily.height,
                        "burn_in_slots": df.height - scored.height,
                        "scale_in_sample": _j(s_is, 5),
                    }
                    if (target == "potential" and shift_name == "per_area"
                            and tier in (PRE_GATE, POST_GATE)):
                        slot_rows.append(scored.select(
                            pl.lit(area).alias("area"), "datetime_jst", "day",
                            pl.lit(tier).alias("tier"),
                            (pl.col("pv_proxy") * pl.col("scale_cf"))
                            .alias("proxy_scaled_cf"),
                            "target_mw", "curtail_mw"))
                if not per_area_daily:
                    continue
                pool_cf = _pool(per_area_daily, "mae_cf")
                pool_is = _pool(per_area_daily, "mae_is")
                common = pool_cf.filter(pl.col("day").cast(pl.Utf8)
                                    .is_in(list(price_days)))
                key = tier if shift_name == "per_area" else f"{tier}__legacyshift"
                tgt_out["tiers"][key] = {
                    "per_area": per_area_meta,
                    "pooled_mae_cf_mw": _j(float(pool_cf["pooled"].mean()), 1),
                    "pooled_mae_is_mw": _j(float(pool_is["pooled"].mean()), 1),
                    "n_days_pooled": pool_cf.height,
                    "pooled_mae_cf_common_mw": _j(float(common["pooled"].mean()), 1),
                    "n_days_common": common.height,
                }
                tgt_out.setdefault("_pools", {})[key] = pool_cf
                log.info("[%s/%s] %-24s cf %.1f MW | is %.1f MW | %d days (common %d)",
                         target, shift_name, tier, pool_cf["pooled"].mean(),
                         pool_is["pooled"].mean(), pool_cf.height, common.height)
        out["targets"][target] = tgt_out

    # ---------------------------------------------------------------- gates
    def _delta(pools: dict, pre: str, post: str, days: set | None) -> dict:
        j = (pools[pre].select("day", pl.col("pooled").alias("pre"))
             .join(pools[post].select("day", pl.col("pooled").alias("post")),
                   on="day", how="inner").sort("day"))
        if days is not None:
            j = j.filter(pl.col("day").cast(pl.Utf8).is_in(list(days)))
        d = np.asarray(j["pre"] - j["post"], dtype=float)
        lo, hi = _boot_ci(d)
        return {"delta_mw": _j(float(d.mean()), 2), "ci95": [_j(lo, 2), _j(hi, 2)],
                "n_days": len(d),
                "definition": ("daily pooled scaled MAE(pre-gate) - MAE(post-gate); "
                               "positive means the post-gate run tracks the "
                               "physical day better. MW is a per-area mean.")}

    pools_pot = out["targets"]["potential"].pop("_pools")
    for t in TARGETS:
        out["targets"][t].pop("_pools", None)

    g51 = _delta(pools_pot, PRE_GATE, POST_GATE, price_days)
    lo, hi = g51["ci95"]
    g51["excludes_zero"] = bool(lo > 0 or hi < 0)
    g51["branch"] = ("a" if lo > 0 else ("c" if hi < 0 else "b"))
    g51["reading"] = {
        "a": "physical gain stands; report as target divergence, not identification",
        "b": "CI covers zero: the third contribution is a null on the common panel",
        "c": "sign reversed: the pre-gate run tracks the physical day better",
    }[g51["branch"]]
    out["gates"]["gate51_physical_gain_crossfit_common"] = g51

    wide = _delta(pools_pot, PRE_GATE, POST_GATE, None)
    out["gates"]["gate51_sensitivity_wide_panel"] = wide

    signs = {}
    for t in TARGETS:
        tiers = out["targets"][t]["tiers"]
        if PRE_GATE in tiers and POST_GATE in tiers:
            signs[t] = _j(tiers[PRE_GATE]["pooled_mae_cf_common_mw"]
                          - tiers[POST_GATE]["pooled_mae_cf_common_mw"], 2)
    out["gates"]["gate52_curtailment_sign_stability"] = {
        "delta_by_target": signs,
        "signs_agree": len({np.sign(v) for v in signs.values() if v is not None}) == 1,
    }

    pre = out["targets"]["potential"]["tiers"][PRE_GATE]
    out["gates"]["gate53_lookahead_cost"] = {
        "pre_gate_mae_in_sample_mw": pre["pooled_mae_is_mw"],
        "pre_gate_mae_crossfit_mw": pre["pooled_mae_cf_mw"],
        "difference_mw": (_j(pre["pooled_mae_cf_mw"] - pre["pooled_mae_is_mw"], 1)
                          if None not in (pre["pooled_mae_cf_mw"],
                                          pre["pooled_mae_is_mw"]) else None),
        "note": ("the in-sample scale is the superseded phase79 convention; the "
                 "difference is what the look-ahead was worth"),
    }

    legacy = out["targets"]["potential"]["tiers"].get(f"{PRE_GATE}__legacyshift")
    if legacy:
        out["gates"]["gate54_stamp_convention"] = {
            "per_area_shift_mae_mw": pre["pooled_mae_cf_mw"],
            "legacy_uniform_shift_mae_mw": legacy["pooled_mae_cf_mw"],
            "difference_mw": _j(pre["pooled_mae_cf_mw"]
                                - legacy["pooled_mae_cf_mw"], 1),
        }

    if slot_rows:
        pl.concat(slot_rows).write_parquet(OUT_SLOTS)
        log.info("wrote %s", OUT_SLOTS)

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "seed": SEED, "n_boot": N_BOOT,
                       "boot_block_days": BOOT_BLOCK}
    if missing_cells:
        msg = ("tier/area caches missing, so the pooled figures would "
               f"rest on an incomplete grid: {sorted(missing_cells)}. "
               "Rebuild them; a warning here used to be enough to ship a "
               "mean over fewer areas than the paper claims.")
        raise RuntimeError(msg)

    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    log.info("GATE 51: branch %s -- %s (delta %.2f MW %s)",
             g51["branch"], g51["reading"], g51["delta_mw"], g51["ci95"])
    log.info("GATE 52: signs agree = %s (%s)",
             out["gates"]["gate52_curtailment_sign_stability"]["signs_agree"], signs)
    log.info("GATE 53: look-ahead cost %.1f MW",
             out["gates"]["gate53_lookahead_cost"]["difference_mw"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
