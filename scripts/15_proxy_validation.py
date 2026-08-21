r"""Phase 62: how much of the real area does one city point capture?

WHY THIS EXISTS
---------------
The third external review objected that all weather in this study enters through
ONE representative city per area, while a JEPX area spans hundreds of kilometres
and its generation is widely dispersed. Capacity-weighted spatial aggregation is
out of scope for this revision by decision, so the honest response is not to
promise it but to MEASURE the deficit and disclose it. A limitation that is
quantified is worth more than one that is merely conceded.

The proxies in phase54 are "one city's irradiance times the whole area's installed
capacity":

    pv_proxy   = sw_pd2 * pv_gw * (1 - 0.004*(T_pd2 - 25))
    wind_proxy = cubic_ramp(v_pd2) * wind_gw

and OCCTO already publishes what they are proxies FOR, at 30-minute resolution for
eight areas. So the validation costs no new data at all.

WHAT THIS DOES (no new data)
----------------------------
A. **Proxy skill against OCCTO realised output.** Pearson and Spearman
   correlation, MAE, capacity-normalised RMSE and bias, for the PV, wind and
   demand channels, per area and broken down by season, by hour band, and by
   floor vs non-floor slots. Potential output (realised + curtailed) is the PV and
   wind target, since the proxy models the resource, not the dispatch.

B. **Capacity-table integrity.** ``data/reference/area_capacity.csv`` is an
   FY2023-24 "approx" vintage, and it is about to be load-bearing twice over: it
   scales every proxy AND it forms the PV/peak saturation index used in the
   cross-area analysis. Realised OCCTO output exceeds several of its nameplate
   figures, which is impossible and therefore must be disclosed.

C. **A vintage-free saturation index.** Because the capacity table is stale, the
   cross-area index is recomputed as (max realised PV output / peak demand),
   which is observable and carries no capacity vintage. If the cross-area
   conclusion is "undetermined" under both indices, that conclusion does not
   depend on the stale table -- which is the claim the paper actually needs.

CLI (from the repository root):
  uv run python scripts/15_proxy_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import (
    peak_gw_by_area,
    pv_gw_by_area,
    wind_gw_by_area,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase62")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
OCCTO_PARQUET = RUN / "phase25.3_occto_generation.parquet"
LADDER_PARQUET = RUN / "phase51_vintage_ladder_per_slot.parquet"
PREV2_DIR = RAW / "open_meteo_historical_forecast"
OUT_JSON = RUN / "phase62_proxy_validation.json"

SEED = 2026
FLOOR = 0.5
MW_PER_GW = 1000.0
# proxy physics, identical to s_20260726_phase54_nwp_proxy_baseline.py
PV_TEMP_REF_C = 25.0
PV_TEMP_DERATE_PER_C = 0.004
WIND_CUT_IN_MS, WIND_RATED_MS, WIND_CUT_OUT_MS = 3.0, 12.0, 25.0
HDD_BASE_C, CDD_BASE_C = 18.0, 24.0
OCCTO_SHIFT_MIN = -30  # OCCTO stamps interval end; phase54 aligns the same way
DAYTIME_HOURS = (9, 15)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
HOUR_BANDS = {"night": (0, 5), "morning": (6, 10),
              "midday": (11, 14), "evening": (15, 23)}
MIN_N_CORR = 30


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


def _skill(pred: np.ndarray, obs: np.ndarray, denom_mw: float) -> dict:
    """Correlation / error metrics of a resource proxy against realised output."""
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    if len(pred) < MIN_N_CORR or np.std(pred) == 0 or np.std(obs) == 0:
        return {"n": len(pred), "insufficient": True}
    # the proxy is in arbitrary units, so scale it to the target by OLS through
    # the origin before reporting error metrics; correlation is scale-free anyway
    scale = float(pred @ obs / (pred @ pred)) if pred @ pred > 0 else 1.0
    fitted = scale * pred
    return {
        "n": len(pred),
        "pearson_r": _j(float(np.corrcoef(pred, obs)[0, 1])),
        "spearman_r": _j(float(stats.spearmanr(pred, obs).statistic)),
        "mae_mw": _j(float(np.mean(np.abs(fitted - obs))), 1),
        "nrmse_pct_of_capacity": _j(
            100.0 * float(np.sqrt(np.mean((fitted - obs) ** 2))) / denom_mw, 2)
        if denom_mw > 0 else None,
        "bias_mw": _j(float(np.mean(fitted - obs)), 1),
        "implied_scale": _j(scale, 4),
    }


# ============================================================ A. proxy vs OCCTO
def _build_joined() -> pl.DataFrame:
    """Join prev2 weather proxies to OCCTO realised output and floor flags."""
    occ = (pl.read_parquet(OCCTO_PARQUET)
           .with_columns(pl.col("datetime_jst").dt.offset_by(f"{OCCTO_SHIFT_MIN}m")))
    price = (pl.read_parquet(LADDER_PARQUET)
             .select("area", "datetime_jst", "y_actual")
             .with_columns((pl.col("y_actual") <= FLOOR).alias("floor")))
    frames = []
    for area in sorted(occ["area"].unique().to_list()):
        files = sorted(PREV2_DIR.glob(f"open_meteo_prev2_jma_msm_{area}_*.parquet"))
        if not files:
            log.warning("no prev2 cache for %s - skipped", area)
            continue
        w = (pl.concat([pl.read_parquet(f) for f in files])
             .unique("datetime_jst").sort("datetime_jst"))
        cols = [c for c in w.columns if c != "datetime_jst"]
        # the caches already carry the _pd2 suffix in some vintages; normalise
        ren = {c: (c if c.endswith("_pd2") else f"{c}_pd2") for c in cols}
        w = w.rename(ren)
        t = pl.col("temperature_c_pd2")
        sw = pl.col("shortwave_radiation_wm2_pd2")
        v = pl.col("wind_speed_kmh_pd2") / 3.6
        ramp = (pl.when(v < WIND_CUT_IN_MS).then(0.0)
                .when(v < WIND_RATED_MS)
                .then(((v - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS)) ** 3)
                .when(v < WIND_CUT_OUT_MS).then(1.0)
                .otherwise(0.0))
        w = w.with_columns(
            (sw * (1.0 - PV_TEMP_DERATE_PER_C * (t - PV_TEMP_REF_C))).alias("pv_proxy"),
            ramp.alias("wind_proxy"),
            ((HDD_BASE_C - t).clip(lower_bound=0.0)
             + (t - CDD_BASE_C).clip(lower_bound=0.0)).alias("degree_proxy"),
        )
        sub = (occ.filter(pl.col("area") == area)
               .join_asof(w.sort("datetime_jst"), on="datetime_jst",
                          strategy="backward", tolerance="1h")
               .join(price.filter(pl.col("area") == area).drop("area"),
                     on="datetime_jst", how="left"))
        frames.append(sub)
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(
        (pl.col("pv_mw") + pl.col("pv_curtailment_mw").fill_null(0.0))
        .alias("pv_potential_mw"),
        (pl.col("wind_mw") + pl.col("wind_curtailment_mw").fill_null(0.0))
        .alias("wind_potential_mw"),
        pl.col("datetime_jst").dt.month().alias("month"),
        pl.col("datetime_jst").dt.hour().alias("hour"),
    )


def proxy_skill(df: pl.DataFrame) -> dict:
    """Per-area proxy skill overall and by season / hour band / floor status."""
    pv_gw, wind_gw, peak_gw = pv_gw_by_area(), wind_gw_by_area(), peak_gw_by_area()
    channels = {
        "pv": ("pv_proxy", "pv_potential_mw", pv_gw),
        "wind": ("wind_proxy", "wind_potential_mw", wind_gw),
        "demand": ("degree_proxy", "demand_mw", peak_gw),
    }
    out: dict = {"areas": {}, "note": (
        "The proxy is a resource index in arbitrary units, so it is rescaled to "
        "the target by least squares through the origin before error metrics are "
        "reported. Correlations are scale-free. PV and wind targets are POTENTIAL "
        "output (realised + curtailed), because the proxy models the resource "
        "rather than the dispatch decision.")}
    for area in sorted(df["area"].unique().to_list()):
        sub = df.filter(pl.col("area") == area)
        blk: dict = {}
        for ch, (pcol, ocol, cap) in channels.items():
            if pcol not in sub.columns or ocol not in sub.columns:
                continue
            denom = cap.get(area, 0.0) * MW_PER_GW
            # PV is only meaningful in daylight; the others use all slots
            base = (sub.filter(pl.col("hour").is_between(*DAYTIME_HOURS))
                    if ch == "pv" else sub)
            entry = {"overall": _skill(np.asarray(base[pcol], dtype=float),
                                      np.asarray(base[ocol], dtype=float), denom)}
            entry["by_season"] = {
                s: _skill(np.asarray(g[pcol], dtype=float),
                          np.asarray(g[ocol], dtype=float), denom)
                for s, months in SEASONS.items()
                if (g := base.filter(pl.col("month").is_in(list(months)))).height}
            entry["by_hour_band"] = {
                b: _skill(np.asarray(g[pcol], dtype=float),
                          np.asarray(g[ocol], dtype=float), denom)
                for b, (lo, hi) in HOUR_BANDS.items()
                if (g := sub.filter(pl.col("hour").is_between(lo, hi))).height}
            if "floor" in sub.columns:
                entry["by_floor"] = {
                    lab: _skill(np.asarray(g[pcol], dtype=float),
                                np.asarray(g[ocol], dtype=float), denom)
                    for lab, g in (("floor", base.filter(pl.col("floor"))),
                                   ("non_floor", base.filter(~pl.col("floor"))))
                    if g.height}
            blk[ch] = entry
        out["areas"][area] = blk

    # headline summary across areas, which is what the manuscript sentence needs
    for ch in channels:
        rs = [b[ch]["overall"].get("pearson_r") for b in out["areas"].values()
              if ch in b and b[ch]["overall"].get("pearson_r") is not None]
        if rs:
            out.setdefault("summary", {})[ch] = {
                "pearson_r_min": _j(min(rs)), "pearson_r_max": _j(max(rs)),
                "pearson_r_median": _j(float(np.median(rs))),
                "n_areas": len(rs),
                "areas_below_0p3": sorted(
                    a for a, b in out["areas"].items()
                    if ch in b and (b[ch]["overall"].get("pearson_r") or 1.0) < 0.3),
            }
    return out


# ==================================================== B. capacity-table integrity
def capacity_integrity(df: pl.DataFrame) -> dict:
    """Nameplate figures against realised OCCTO maxima. Realised cannot exceed it."""
    pv_gw, wind_gw, peak_gw = pv_gw_by_area(), wind_gw_by_area(), peak_gw_by_area()
    rows = []
    for area in sorted(df["area"].unique().to_list()):
        sub = df.filter(pl.col("area") == area)
        for ch, col, table in (("pv", "pv_potential_mw", pv_gw),
                               ("wind", "wind_potential_mw", wind_gw)):
            if col not in sub.columns:
                continue
            realised = float(sub[col].max() or 0.0)
            nameplate = table.get(area, 0.0) * MW_PER_GW
            rows.append({
                "area": area, "channel": ch,
                "nameplate_mw": _j(nameplate, 1),
                "realised_max_mw": _j(realised, 1),
                "realised_over_nameplate_ratio": _j(
                    realised / nameplate, 3) if nameplate else None,
                "impossible": bool(nameplate and realised > nameplate),
            })
    bad = [r for r in rows if r["impossible"]]
    return {
        "table": "data/reference/area_capacity.csv",
        "table_vintage": "FY2023-24, marked 'approx'",
        "rows": rows,
        "n_impossible": len(bad),
        "worst": max(bad, key=lambda r: r["realised_over_nameplate_ratio"] or 0.0)
        if bad else None,
        "peak_demand_gw": {a: _j(peak_gw.get(a), 3) for a in sorted(peak_gw)},
        "verdict": (
            "The nameplate table is understated: realised potential output exceeds "
            f"it in {len(bad)} of {len(rows)} area-channels. A constant per-area scale factor is "
            "absorbed by the learner, so the proxies used as REGRESSORS are "
            "unaffected. The cross-area PV/peak saturation index is NOT: it "
            "inherits the error directly, so a vintage-free index is computed "
            "below and both are reported."),
    }


# ================================================ C. vintage-free saturation index
def saturation_indices(df: pl.DataFrame) -> dict:
    """PV/peak from the stale table against an observable realised-output index."""
    pv_gw, peak_gw = pv_gw_by_area(), peak_gw_by_area()
    areas = sorted(df["area"].unique().to_list())
    table_idx, realised_idx = {}, {}
    for area in areas:
        sub = df.filter(pl.col("area") == area)
        peak_mw = float(sub["demand_mw"].max() or 0.0)
        pv_max = float(sub["pv_potential_mw"].max() or 0.0)
        if peak_gw.get(area) and pv_gw.get(area):
            table_idx[area] = pv_gw[area] / peak_gw[area]
        if peak_mw > 0:
            realised_idx[area] = pv_max / peak_mw
    common = sorted(set(table_idx) & set(realised_idx))
    x1 = np.array([table_idx[a] for a in common])
    x2 = np.array([realised_idx[a] for a in common])
    sp = stats.spearmanr(x1, x2)
    return {
        "table_pv_over_peak": {a: _j(table_idx[a], 4) for a in common},
        "realised_pv_max_over_peak_demand": {a: _j(realised_idx[a], 4)
                                             for a in common},
        "rank_agreement_spearman": _j(float(sp.statistic)),
        "rank_agreement_p": _j(float(sp.pvalue), 5),
        "rank_changes": [
            {"area": a,
             "table_rank": int(r1), "realised_rank": int(r2)}
            for a, r1, r2 in zip(common, stats.rankdata(-x1), stats.rankdata(-x2),
                                 strict=True) if int(r1) != int(r2)],
        "reading": (
            "The realised index is observable and carries no capacity vintage. "
            "Where the two indices rank areas differently, any cross-area "
            "statistic computed on the table index is sensitive to the stale "
            "capacity figures, and the paper must show its conclusion under "
            "both. Feeding this index into the phase55 slope/rho tests is the "
            "follow-up."),
    }


# ========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    for p in (OCCTO_PARQUET, LADDER_PARQUET):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    df = _build_joined()
    log.info("joined panel: %d rows / %d areas / %s..%s", df.height,
             df["area"].n_unique(), df["datetime_jst"].min(), df["datetime_jst"].max())

    out: dict = {
        "phase": 62,
        "title": "Single-point weather proxy validation against OCCTO actuals",
        "purpose": (
            "Quantify the single-point limitation rather than merely concede it. "
            "Capacity-weighted spatial aggregation is out of scope for this "
            "revision; this phase measures what the one-city proxy does and does "
            "not capture, and audits the capacity table it is scaled by."),
        "inputs": {
            "occto_parquet": OCCTO_PARQUET.name,
            "occto_sha256": _sha256(OCCTO_PARQUET),
            "ladder_parquet": LADDER_PARQUET.name,
        },
        "panel": {"n_rows": df.height, "n_areas": df["area"].n_unique(),
                  "first": str(df["datetime_jst"].min()),
                  "last": str(df["datetime_jst"].max())},
    }

    out["proxy_skill"] = proxy_skill(df)
    for ch, s in out["proxy_skill"].get("summary", {}).items():
        log.info("%s proxy: Pearson r %.2f..%.2f (median %.2f); weak (<0.3) in %s",
                 ch, s["pearson_r_min"], s["pearson_r_max"],
                 s["pearson_r_median"], s["areas_below_0p3"] or "none")

    out["capacity_integrity"] = capacity_integrity(df)
    log.info("capacity table: %d impossible area-channels; worst = %s",
             out["capacity_integrity"]["n_impossible"],
             out["capacity_integrity"]["worst"])

    out["saturation_indices"] = saturation_indices(df)
    log.info("saturation index rank agreement: rho=%s (p=%s); %d rank changes",
             out["saturation_indices"]["rank_agreement_spearman"],
             out["saturation_indices"]["rank_agreement_p"],
             len(out["saturation_indices"]["rank_changes"]))

    out["metadata"] = {
        "script": ""
                  "15_proxy_validation.py",
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
