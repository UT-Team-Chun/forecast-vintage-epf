r"""Phase 82b: absolute performance, and a standard relative denominator.

WHY
---
The ladder table reports improvements only. A reader cannot tell whether
0.253 JPY/kWh comes off a 1.5 or a 3.0 baseline, and the paper's rMAE is
divided by its OWN price-only model rather than by the naive benchmark the
EPF literature uses. The seventh review asked for the absolute level and a
standard denominator in the main text; both are metric computations on the
per-slot forecasts already on disk, so nothing is retrained here.

WHAT IS COMPUTED
----------------
For every tier and every pre-registered slice: absolute MAE and RMSE, the
same relative to a naive 168-hour (one week) seasonal persistence forecast,
and the same relative to the price-only model, so the existing rMAE numbers
stay comparable. Per-area absolute MAE is written for the supplement.

The naive benchmark is y_actual lagged 168 hours in the same area. The lag is
unavailable on the panel's first week and for a week after every internal gap,
and the filter is on SLOTS, not days: four days lose only some of their slots.
Every column of the relative block is computed on the same surviving slots,
because a relative metric whose numerator and denominator span different
samples is not a ratio of anything -- and the block records the sub-panel's
own slot and day counts rather than leaving them to be inferred by
subtraction, which gives the wrong day count (an earlier version reported
732 - 14 = 718 where the sub-panel actually spans 722 days).

Branches are fixed in
docs/research/2026-08-13_phase82b_absolute_metrics_precommit.md.

CLI (from the repository root):
  uv run python scripts/64_absolute_metrics.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase82b")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
SLOTS = RUN / "phase85_exact_ladder_per_slot.parquet"
OUT_JSON = RUN / "phase82b_absolute_metrics.json"
PRECOMMIT = "docs/research/2026-08-13_phase82b_absolute_metrics_precommit.md"

# The CW-primary ladder, in dissemination order, plus the price-only baseline.
TIERS = [
    ("price_only", "Price-only baseline"),
    ("era5_cw", "Stale observations (day-lagged reanalysis)"),
    ("msm_d2_12z_cw", "Pre-gate run (D-2 12 UTC)"),
    ("msm_gate_composite_cw", "Freshest admissible per slot"),
    ("msm_d1_00z_cw", "First excluded run (D-1 00 UTC)"),
    ("oracle_cw", "Realised weather (not available ex ante)"),
]
NAIVE_LAG_H = 168
NEAR_FLOOR = 0.5
# One JEPX price tick: the equivalence margin the ladder was pre-committed to.
EQUIV_MARGIN = 0.01


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


def _metrics(df: pl.DataFrame, col: str) -> dict:
    e = (df[col] - df["y_actual"]).to_numpy()
    n = len(e)
    if n == 0:
        return {"mae": None, "rmse": None, "n_slots": 0}
    return {"mae": _j(float(abs(e).mean())), "rmse": _j(float((e ** 2).mean() ** 0.5)),
            "n_slots": int(n)}


def main() -> int:
    if not SLOTS.exists():
        log.error("missing %s", SLOTS)
        return 2
    cols = ["area", "datetime_jst", "test_day", "y_actual"] + [
        f"y_pred_{t}" for t, _ in TIERS]
    df = pl.read_parquet(SLOTS, columns=cols)
    log.info("panel: %d slots, %d days, %d areas",
             df.height, df["test_day"].n_unique(), df["area"].n_unique())

    # ---- the naive 168-hour benchmark, by self-join on the shifted stamp -----
    naive = (df.select("area", "datetime_jst",
                       pl.col("y_actual").alias("y_pred_naive168"))
             .with_columns(
                 (pl.col("datetime_jst") + timedelta(hours=NAIVE_LAG_H))
                 .alias("datetime_jst")))
    df = df.join(naive, on=["area", "datetime_jst"], how="left")
    n_missing_naive = int(df["y_pred_naive168"].is_null().sum())
    naive_days = sorted(df.filter(pl.col("y_pred_naive168").is_null())
                        ["test_day"].unique().to_list())
    log.info("naive-168h unavailable on %d slots (%d days: %s .. %s)",
             n_missing_naive, len(naive_days),
             naive_days[0] if naive_days else "-",
             naive_days[-1] if naive_days else "-")

    df = df.with_columns(
        pl.col("datetime_jst").dt.hour().alias("hour"),
        (pl.col("y_actual") <= NEAR_FLOOR).alias("floor"),
    )
    slices = {
        "all": pl.lit(True),
        "daylight": pl.col("hour").is_between(6, 17),
        "midday": pl.col("hour").is_between(9, 14),
        "floor": pl.col("floor"),
        "non_floor": ~pl.col("floor"),
    }

    common = df.filter(pl.col("y_pred_naive168").is_not_null())
    out: dict = {
        "phase": "82b", "title": "Absolute performance and standard relative metrics",
        "precommit_doc": PRECOMMIT,
        "design": {
            "panel": {"slots": df.height, "days": df["test_day"].n_unique(),
                      "areas": df["area"].n_unique()},
            "naive_benchmark": (f"y_actual lagged {NAIVE_LAG_H} h in the same area; "
                                "the EPF-standard seasonal persistence forecast"),
            "naive_unavailable_slots": n_missing_naive,
            "naive_unavailable_days": len(naive_days),
            # The sub-panel measured, not inferred: the filter is on slots, so
            # days with SOME surviving slots stay in it.
            "subpanel_slots": int(common.height),
            "subpanel_days": int(common["test_day"].n_unique()),
            "partially_affected_days": sorted(
                set(naive_days) & set(common["test_day"].unique().to_list())),
            "relative_block_rule": ("every relative figure is computed on the "
                                    "sub-panel where the naive benchmark exists, "
                                    "so numerator and denominator span the same "
                                    "slots"),
            "near_floor_threshold": NEAR_FLOOR,
            # Recorded here so the manuscript's "the margin is X% of the
            # baseline" can be checked against a denominator from the SAME file
            # as the baseline it names. It used to divide by a literal frozen in
            # the guard, which was a different panel's baseline.
            "equivalence_margin_jpykwh": EQUIV_MARGIN,
            "slice_definitions": {
                "daylight": "06:00-17:59 JST", "midday": "09:00-14:59 JST",
                "floor": f"y_actual <= {NEAR_FLOOR} JPY/kWh",
            },
        },
        "absolute": {}, "relative": {}, "per_area": {}, "gates": {},
    }

    for sname, expr in slices.items():
        sub, subc = df.filter(expr), common.filter(expr)
        out["absolute"][sname] = {
            t: _metrics(sub, f"y_pred_{t}") for t, _ in TIERS}
        base_naive = _metrics(subc, "y_pred_naive168")
        base_price = _metrics(subc, "y_pred_price_only")
        rel = {"naive168": base_naive, "price_only": base_price, "tiers": {}}
        for t, _ in TIERS:
            m = _metrics(subc, f"y_pred_{t}")
            rel["tiers"][t] = {
                "mae": m["mae"], "rmse": m["rmse"],
                "rmae_vs_naive168": _j(m["mae"] / base_naive["mae"]),
                "rrmse_vs_naive168": _j(m["rmse"] / base_naive["rmse"]),
                "rmae_vs_price_only": _j(m["mae"] / base_price["mae"]),
                "rrmse_vs_price_only": _j(m["rmse"] / base_price["rmse"]),
            }
        out["relative"][sname] = rel
        log.info("slice %-10s n=%6d  naive %.3f  price-only %.3f  pre-gate %.3f",
                 sname, sub.height, base_naive["mae"] or float("nan"),
                 base_price["mae"] or float("nan"),
                 out["absolute"][sname]["msm_d2_12z_cw"]["mae"])

    for area, sub in df.group_by("area"):
        a = area[0] if isinstance(area, tuple) else area
        out["per_area"][a] = {t: _metrics(sub, f"y_pred_{t}")["mae"]
                             for t, _ in TIERS}

    # ---- Gate 55: does the naive denominator change the ORDERING? -----------
    # The paper's rMAE divides by its own price-only model. If the EPF-standard
    # denominator reordered the tiers, the existing relative numbers would be an
    # artefact of that choice; this gate says so either way.
    rel_all = out["relative"]["all"]["tiers"]
    keys = [t for t, _ in TIERS if t != "price_only"]
    by_naive = sorted(keys, key=lambda k: rel_all[k]["rmae_vs_naive168"])
    by_price = sorted(keys, key=lambda k: rel_all[k]["rmae_vs_price_only"])
    out["gates"]["gate55_denominator_ordering"] = {
        "ordering_by_naive168": by_naive,
        "ordering_by_price_only": by_price,
        "orderings_agree": by_naive == by_price,
        "reading": ("the two denominators induce the same tier ordering, so the "
                    "paper's relative figures are not an artefact of dividing by "
                    "its own baseline" if by_naive == by_price else
                    "THE ORDERINGS DIFFER: report both denominators in the main "
                    "text and say which claims depend on the choice"),
    }
    # The naive benchmark should be WORSE than every fitted model; if it beat one,
    # that cell would be the story rather than a denominator.
    naive_mae = out["relative"]["all"]["naive168"]["mae"]
    beaten = [k for k in keys if rel_all[k]["mae"] >= naive_mae]
    out["gates"]["gate55b_naive_is_a_floor"] = {
        "naive_mae": naive_mae, "tiers_not_better_than_naive": beaten,
        "ok": not beaten,
    }

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha()}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    g = out["gates"]["gate55_denominator_ordering"]
    log.info("GATE 55: orderings agree = %s", g["orderings_agree"])
    log.info("  by naive-168h: %s", " < ".join(g["ordering_by_naive168"]))
    log.info("GATE 55b: naive is a floor = %s",
             out["gates"]["gate55b_naive_is_a_floor"]["ok"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
