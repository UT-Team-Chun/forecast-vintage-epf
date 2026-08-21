r"""Phase 92: how many days a year each area actually curtails photovoltaics.

WHY
---
Section 2 stated "Kyushu records roughly 30 curtailment days per year, far more
than any other area". Both halves are wrong on the data this paper uses. The 30
is inherited verbatim from a superseded draft, where it described an FY2018-era
OCCTO level; on the FY2024-FY2026 records the ladder is scored against, Kyushu
curtails on about 150 days a year. And the gap to the next most affected area is
roughly a factor of two, not "far more".

The sentence matters beyond its own accuracy. It is the paragraph that motivates
using curtailment records as a marker of surplus conditions, and it is the number
a reviewer of a Japanese market paper is most likely to know independently.

Nothing here is a new hypothesis or a gate: it counts days in a file the paper
already ships and already reads (phase89 restores curtailed energy from this same
column to build the physical target). It exists so a §2 sentence has provenance
instead of a remembered figure.

CLI (from the repository root):
  uv run --extra research python scripts/43_curtailment_days.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase92")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
MAIN = RUN / "phase25.3_occto_generation.parquet"
# Kyushu's 2026-06 arrives in a separate extension file; the merged loader
# swallowed that month (recorded in the phase89 notes), so both are read here and
# the script asserts which months came from where.
EXT = RUN / "phase25.3_occto_generation_ext.parquet"
OUT_JSON = RUN / "phase92_curtailment_days.json"

COL = "pv_curtailment_mw"
# Japanese fiscal years, which is how the operators and OCCTO report curtailment.
FYS = {
    "FY2024": (dt.date(2024, 4, 1), dt.date(2025, 3, 31)),
    "FY2025": (dt.date(2025, 4, 1), dt.date(2026, 3, 31)),
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    if not MAIN.exists():
        log.error("missing %s", MAIN)
        return 2
    frames = [pl.read_parquet(MAIN)]
    if EXT.exists():
        frames.append(pl.read_parquet(EXT))
    d = pl.concat(frames, how="diagonal").with_columns(
        pl.col("datetime_jst").dt.date().alias("day"))

    if d[COL].null_count():
        log.error("%s has %d nulls; a null is not a zero and the day counts "
                  "would silently exclude those slots", COL, d[COL].null_count())
        return 3

    areas = sorted(d["area"].unique().to_list())
    per_area: dict[str, dict] = {}
    for fy, (lo, hi) in FYS.items():
        window = d.filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
        # Every area must actually cover the fiscal year, or a low count would
        # read as little curtailment rather than as missing months.
        cover = (window.group_by("area")
                 .agg(pl.col("day").n_unique().alias("days_present")))
        expected = (hi - lo).days + 1
        for row in cover.iter_rows(named=True):
            if row["days_present"] < expected - 2:  # leap/DST slack
                log.error("%s covers only %d of %d days of %s; refusing to "
                          "report a rate from an incomplete year",
                          row["area"], row["days_present"], expected, fy)
                return 4
        counted = (window.filter(pl.col(COL) > 0).group_by("area")
                   .agg(pl.col("day").n_unique().alias("days")))
        got = dict(zip(counted["area"].to_list(), counted["days"].to_list(),
                       strict=True))
        for a in areas:
            per_area.setdefault(a, {})[fy] = int(got.get(a, 0))

    two_fy_mean = {a: (v["FY2024"] + v["FY2025"]) / 2 for a, v in per_area.items()}
    top = max(two_fy_mean, key=lambda a: two_fy_mean[a])
    rest = {a: v for a, v in two_fy_mean.items() if a != top}
    runner_up = max(rest, key=lambda a: rest[a])
    ratio = two_fy_mean[top] / two_fy_mean[runner_up]

    out = {
        "phase": "92",
        "title": "Photovoltaic curtailment days per area per fiscal year",
        "why": ("Section 2 said Kyushu curtails on roughly 30 days a year, a "
                "figure inherited from a superseded draft describing an "
                "FY2018-era level. On the records this paper scores against it "
                "is about five times that, and the gap to the next area is a "
                "factor of two rather than 'far more'"),
        "source": {
            "files": [MAIN.name] + ([EXT.name] if EXT.exists() else []),
            "column": COL,
            "note": ("a day counts when any half-hour slot in it records "
                     "positive photovoltaic curtailment; this is the same "
                     "column phase89 adds back to build the physical target"),
        },
        "fiscal_years": {k: [str(v[0]), str(v[1])] for k, v in FYS.items()},
        "per_area_days": per_area,
        "two_fiscal_year_mean_days": {a: round(v, 1)
                                      for a, v in two_fy_mean.items()},
        "most_curtailed": {
            "area": top,
            "days": per_area[top],
            "mean_days": round(two_fy_mean[top], 1),
            "rounded_for_prose": int(round(two_fy_mean[top] / 10.0) * 10),
        },
        "runner_up": {
            "area": runner_up,
            "days": per_area[runner_up],
            "mean_days": round(two_fy_mean[runner_up], 1),
        },
        "ratio_top_to_runner_up": round(ratio, 2),
        "reading": (
            f"{top} curtails on {per_area[top]['FY2024']} days in FY2024 and "
            f"{per_area[top]['FY2025']} in FY2025, about "
            f"{ratio:.1f} times the next most affected area ({runner_up}). "
            "'Far more than any other area' is not supportable at that ratio"),
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    for a in sorted(per_area, key=lambda x: -two_fy_mean[x]):
        log.info("  %-10s FY2024=%3d FY2025=%3d  mean=%5.1f",
                 a, per_area[a]["FY2024"], per_area[a]["FY2025"],
                 two_fy_mean[a])
    log.info("%s is %.2fx %s; wrote %s", top, ratio, runner_up, OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
