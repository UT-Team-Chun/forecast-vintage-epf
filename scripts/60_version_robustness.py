r"""Phase 80: robustness of every headline contrast to the 2025-05-20 MSM upgrade.

Precommit: docs/research/2026-08-11_phase80_jma_version_precommit.md.
JMA improved the MSM's physics on 2025-05-20 (near-surface wind and temperature,
all regions and forecast times), inside the test window. This phase splits the
CW-primary ladder's headline contrasts at that date (delivery-day basis),
tests the gate-timing-by-version interaction, reports rolling 90-day skill,
checks dissemination-lag stability across the boundary, and emits the version
timeline. Branches B1 (stable) / B2 (report split, scope pooled claims) are in
the precommit; either is reported.

CLI (from the repository root):
  uv run python scripts/60_version_robustness.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase80")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
PANEL = RUN / "phase85_exact_ladder_per_slot.parquet"
P72 = RUN / "phase72_dissemination_audit_per_file.parquet"
OUT_JSON = RUN / "phase80_jma_version.json"
PRECOMMIT = "docs/research/2026-08-11_phase80_jma_version_precommit.md"

SEED, N_BOOT, BOOT_BLOCK, ALPHA = 2026, 10_000, 7, 0.05
SPLIT = date(2025, 5, 20)
EXCLUDE_DAYS = 14
TIERS_CW = ("msm_d2_12z_cw", "msm_gate_composite_cw", "msm_d1_00z_cw",
            "oracle_cw", "era5_cw")
HEADLINE, POST = "msm_d2_12z_cw", "msm_d1_00z_cw"
ROLL_DAYS = 90
VERSION_TIMELINE = [
    {"date": "2020-03-01", "change": "MSM horizon extended to 78 h for the "
                                     "00/12 UTC cycles (archive property the "
                                     "ladder design rests on)"},
    {"date": "2025-05-20", "change": "physics upgrade: near-surface and "
                                     "lower-tropospheric wind and temperature "
                                     "improved across regions and forecast "
                                     "times (jma2025msmupgrade)"},
]


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


def _boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    means = d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _entry(d: np.ndarray) -> dict:
    lo, hi = _boot_ci(d)
    return {"mean": _j(float(d.mean())), "ci95": [_j(lo), _j(hi)], "n_days": len(d)}


def _two_window_diff(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    """Difference of two INDEPENDENT window means, block bootstrap each side."""
    rng = np.random.default_rng(seed)

    def draws(d: np.ndarray) -> np.ndarray:
        n = len(d)
        n_blocks = int(np.ceil(n / BOOT_BLOCK))
        starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
        idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
        return d[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)

    diff = draws(b) - draws(a)
    point = float(b.mean() - a.mean())
    p = 2.0 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))
    return {"delta_v2_minus_v1": _j(point), "ci95": [_j(float(np.quantile(diff, 0.025))),
                                                     _j(float(np.quantile(diff, 0.975)))],
            "p_boot": _j(max(p, 1.0 / N_BOOT), 5), "n_v1": len(a), "n_v2": len(b)}


def _pooled_daily(per_day: pl.DataFrame, cell: str) -> pl.DataFrame:
    return (per_day.with_columns(
        (pl.col("mae_price_only") - pl.col(f"mae_{cell}")).alias("imp"))
        .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day"))


def main() -> int:
    panel = pl.read_parquet(PANEL)
    cw_flags = [c for c in panel.columns if c.startswith("cw_ok_")]
    panel = panel.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
    cells = [*TIERS_CW, "price_only"]
    per_day = (panel.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in cells])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in cells])
        .sort("area", "test_day"))

    daily = {c: _pooled_daily(per_day, c) for c in TIERS_CW}
    split_iso = SPLIT.isoformat()

    out: dict = {"phase": 80, "title": "MSM 2025-05-20 version robustness",
                 "precommit_doc": PRECOMMIT, "split_date": split_iso,
                 "version_timeline": VERSION_TIMELINE, "tiers": {}}

    for tier, df in daily.items():
        v1 = np.asarray(df.filter(pl.col("test_day") < split_iso)["imp"], dtype=float)
        v2 = np.asarray(df.filter(pl.col("test_day") >= split_iso)["imp"], dtype=float)
        out["tiers"][tier] = {"v1_pre_upgrade": _entry(v1), "v2_post_upgrade": _entry(v2),
                              "difference": _two_window_diff(v1, v2)}
        log.info("%-24s v1 %.4f (n=%d)  v2 %.4f (n=%d)", tier,
                 v1.mean(), len(v1), v2.mean(), len(v2))

    # gate-timing x version interaction (paired daily post-pre difference)
    j = (daily[POST].rename({"imp": "post"})
         .join(daily[HEADLINE].rename({"imp": "pre"}), on="test_day").sort("test_day"))
    gt = j.with_columns((pl.col("post") - pl.col("pre")).alias("d"))
    inter: dict = {}
    for label, excl in (("exact", 0), (f"excluding_pm{EXCLUDE_DAYS}d", EXCLUDE_DAYS)):
        lo_d = (SPLIT.toordinal() - excl, SPLIT.toordinal() + excl)
        keep = gt.with_columns(pl.col("test_day").str.to_date().alias("td"))
        keep = keep.filter(~pl.col("td").map_elements(
            lambda d, r=lo_d: r[0] <= d.toordinal() < r[1], return_dtype=pl.Boolean))
        v1 = np.asarray(keep.filter(pl.col("test_day") < split_iso)["d"], dtype=float)
        v2 = np.asarray(keep.filter(pl.col("test_day") >= split_iso)["d"], dtype=float)
        inter[label] = {"v1": _entry(v1), "v2": _entry(v2),
                        "interaction": _two_window_diff(v1, v2)}
    out["gate_timing_by_version"] = inter

    # rolling 90-day pooled improvement of the headline (for the plot)
    h = daily[HEADLINE].with_columns(
        pl.col("imp").rolling_mean(ROLL_DAYS).alias("roll"))
    out["rolling_headline"] = {
        "window_days": ROLL_DAYS,
        "test_day": h["test_day"].to_list()[ROLL_DAYS - 1:],
        "rolling_improvement": [_j(v) for v in h["roll"].to_list()[ROLL_DAYS - 1:]],
    }

    # dissemination-lag stability across the boundary (phase72 measured lags)
    if P72.exists():
        p72 = pl.read_parquet(P72).filter(pl.col("status") == 200)
        for label, f in (("v1", pl.col("delivery_day") < split_iso),
                         ("v2", pl.col("delivery_day") >= split_iso)):
            sub = p72.filter(f)["lag_minutes"].drop_nulls()
            out.setdefault("dissemination_lag_by_version", {})[label] = {
                "median": _j(float(sub.median()), 2),
                "p99": _j(float(sub.quantile(0.99)), 2),
                "max": _j(float(sub.max()), 2), "n_files": sub.len()}

    ia = out["gate_timing_by_version"]["exact"]["interaction"]
    stable = ia["ci95"][0] is not None and ia["ci95"][0] <= 0 <= ia["ci95"][1]
    out["branch"] = {"value": "B1" if stable else "B2",
                     "reading": ("gate-timing conclusion robust to the version "
                                 "split" if stable else
                                 "interaction detected: report split estimates "
                                 "and scope pooled claims")}
    log.info("BRANCH %s (interaction %.4f [%s, %s])", out["branch"]["value"],
             ia["delta_v2_minus_v1"], ia["ci95"][0], ia["ci95"][1])

    out["metadata"] = {"script": Path(__file__).name, "git_sha": _git_sha(),
                       "generated_utc": datetime.now(UTC).isoformat(),
                       "seed": SEED, "n_boot": N_BOOT, "boot_block_days": BOOT_BLOCK}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
