r"""Phase 78: the feature-lineage audit -- every family, timed and sourced.

Precommit: docs/research/2026-08-11_phase78_lineage_audit_precommit.md.
Table 1 claimed "operator feeds publish within a day" without a source while
SS4.1 said the same feed arrives in monthly files. This phase replaces
assertion with lineage: for every feature family, the value source, the
first-publication evidence (primary documents gathered 2026-08-11), the
MEASURED dissemination where we have archive timestamps (phase72 + phase84),
the worst-case gate margin, and the admissibility verdict under processing
buffers of 0/30/60/90/120 minutes. Output feeds the manuscript's lineage
table and the no-look-ahead unit test.

CLI (from the repository root):
  uv run --extra research python scripts/41_lineage_audit.py
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase78")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P72 = RUN / "phase72_dissemination_audit_per_file.parquet"
P84 = RUN / "phase84_head_ef_per_file.parquet"
OUT_JSON = RUN / "phase78_lineage_audit.json"
PRECOMMIT = "docs/research/2026-08-11_phase78_lineage_audit_precommit.md"
BUFFERS_MIN = (0, 30, 60, 90, 120)

# gate-minus-init minutes per run (10:00 JST gate; init in UTC)
GATE_MINUS_INIT = {"msm_d2_12z": 780, "msm_d2_21z": 240, "msm_d1_00z": 60,
                   "msm_d2_18z": 420, "msm_d3_12z": 2220}

TSO_EVIDENCE = {
    "occto_framework": {
        "quote": "実需給後１時間程度以内に、公開を行う (システム整備後); "
                 "無条件の下限は１時間値を最低月１回の更新",
        "url": "https://www.occto.or.jp/assets/keitoujouhou/files/"
               "4.souden_kouhyou_koumoku_oldnew2304.pdf"},
    "tepco": {"quote": "毎30分データについて、実需給後1時間以内での公表を予定",
              "url": "https://www.tepco.co.jp/forecast/html/area_data-j.html"},
    "kyuden": {"quote": "実需給後、1時間以内を目処に30分値を更新; 当月分は推計実績を"
                        "含む速報値…実需給の翌月末を目処に、実績値への上書き更新",
               "url": "https://www.kyuden.co.jp/td_area_jukyu/jukyu.html"},
    "tohoku": {"quote": "当日 CSV (realtime_jukyu); 数値は、過去にさかのぼって"
                        "修正されることがあります",
               "url": "https://setsuden.nw.tohoku-epco.co.jp/realtime_jukyu.html"},
    "hepco": {"quote": "当日実績のみグラフで表示; 発電実績および需要実績は速報値",
              "url": "http://denkiyoho.hepco.co.jp/supply_demand_results.html"},
}


def _j(x: float | None, nd: int = 2) -> float | None:
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


def _run_lag_stats() -> dict[str, dict]:
    frames = []
    if P72.exists():
        frames.append(pl.read_parquet(P72).select("delivery_day", "run", "lag_minutes",
                                                  "status"))
    if P84.exists():
        frames.append(pl.read_parquet(P84).select("delivery_day", "run", "lag_minutes",
                                                  "status"))
    df = pl.concat(frames).filter(pl.col("status") == 200)
    out = {}
    for run_key, g in df.group_by("run"):
        run = run_key[0] if isinstance(run_key, tuple) else run_key
        # per-day completion lag = max over the run's files that day
        day = g.group_by("delivery_day").agg(pl.col("lag_minutes").max())
        lag = day["lag_minutes"].drop_nulls()
        gmi = GATE_MINUS_INIT[run]
        worst = float(lag.max())
        rec = {"n_days": day.height, "median_lag_min": _j(float(lag.median())),
               "p99_lag_min": _j(float(lag.quantile(0.99))),
               "worst_lag_min": _j(worst),
               "gate_minus_init_min": gmi,
               "worst_margin_min": _j(gmi - worst),
               "admissible_under_buffer_min": {}}
        for b in BUFFERS_MIN:
            share = float((day["lag_minutes"] + b <= gmi).mean())
            rec["admissible_under_buffer_min"][str(b)] = _j(share, 4)
        out[run] = rec
    return out


def main() -> int:
    runs = _run_lag_stats()
    for r, v in sorted(runs.items()):
        log.info("%-12s median %.1f  worst %.1f  margin %.1f  buffers %s",
                 r, v["median_lag_min"], v["worst_lag_min"], v["worst_margin_min"],
                 v["admissible_under_buffer_min"])

    families = [
        {"family": "Area price lags (48/96/336/672 slots)",
         "source": "JEPX day-ahead auction results",
         "first_publication": "auction results published on D-2, hours after the "
                              "10:00 D-2 gate (jepx2024rules)",
         "measured": None, "gate_margin": ">= 22 h", "revision_risk":
             "OBSERVED: JEPX back-fills recently-empty columns (2026-07-26/27 "
             "rows, caught live by the Gate-45a audit); price columns for "
             "delivery days inside the panel were bit-identical",
         "admissible": True},
        {"family": "System price / spread / neighbour prices (lag-48)",
         "source": "JEPX, one day earlier", "first_publication": "as above",
         "measured": None, "gate_margin": ">= 22 h", "revision_risk": "as above",
         "admissible": True},
        {"family": "Calendar", "source": "deterministic",
         "first_publication": "known in advance", "measured": None,
         "gate_margin": "unbounded", "revision_risk": "none", "admissible": True},
        {"family": "ERA5 reanalysis (lag-48/96) / realised ex-post",
         "source": "ECMWF via Open-Meteo archive",
         "first_publication": "ERA5T ~5 days after valid time; final ERA5 ~2-3 "
                              "months (hersbach2020era5)",
         "measured": "phase85 overlap diagnostic: ZERO drift on the frozen "
                     "boundary day across all areas (the documented ERA5T->ERA5 "
                     "revision risk did not materialise at this boundary; the "
                     "freeze policy protects against it regardless)",
         "gate_margin": "NEGATIVE (publishes after the gate)",
         "revision_risk": "ERA5T -> ERA5 revisions documented (ECMWF); "
                          "not observed at our frozen boundary",
         "admissible": False,
         "role": "research tiers only, labelled non-admissible"},
        {"family": "JMA-MSM pre-gate, run-pinned (D-2 12Z)",
         "source": "RISH archive, per-run GRIB2",
         "first_publication": "spec: init + 2h30m (jmbsc2025msmgpv)",
         "measured": runs.get("msm_d2_12z"), "gate_margin":
             f"worst {runs['msm_d2_12z']['worst_margin_min']:.0f} min",
         "revision_risk": "none: an archived run is immutable",
         "admissible": True},
        {"family": "JMA-MSM pre-gate, freshest per slot (D-2 21Z + 12Z)",
         "source": "RISH archive",
         "first_publication": "spec: init + 2h30m",
         "measured": runs.get("msm_d2_21z"), "gate_margin":
             f"worst {runs['msm_d2_21z']['worst_margin_min']:.0f} min "
             "(admissible at every buffer <= 90 min on all days; the 120-min "
             "buffer removes it and the ladder's fallback is the already-"
             "estimated D-2 12Z cell)",
         "revision_risk": "none", "admissible": True},
        {"family": "JMA-MSM cycle rungs (D-3 12Z, D-2 18Z)",
         "source": "RISH archive",
         "first_publication": "spec: init + 2h30m",
         "measured": {k: runs.get(k) for k in ("msm_d3_12z", "msm_d2_18z")},
         "gate_margin": "hours (see measured)", "revision_risk": "none",
         "admissible": True},
        {"family": "JMA-MSM first post-gate (D-1 00Z)",
         "source": "RISH archive",
         "first_publication": "spec: init + 2h30m -> ~11:30 JST",
         "measured": runs.get("msm_d1_00z"),
         "gate_margin": (
             f"NEGATIVE: completes {-runs['msm_d1_00z']['worst_margin_min']:.0f}"
             " min after the gate at worst"),
         "revision_risk": "none", "admissible": False,
         "role": "gate-timing comparison only"},
        {"family": "OCCTO supply mix (lag-96)",
         "source": "TSO area supply-demand actuals; monthly consolidated CSVs "
                   "used for values",
         "first_publication": "TIMING citable: OCCTO framework requires "
                              "publication ~1 h after real time (system-"
                              "dependent clause), and TEPCO/Kyushu/Tohoku/"
                              "Hokkaido document same-day or within-1-hour "
                              "feeds; unconditional floor is monthly",
         "evidence": TSO_EVIDENCE,
         "measured": None,
         "gate_margin": ">= 23 h for the last D-2 slot under the within-1-hour "
                        "feeds",
         "revision_risk": "DOCUMENTED: gate-time values are provisional "
                          "(Kyushu: overwritten with finals by end of the "
                          "following month); our training values are the "
                          "finalized monthly files, so value provenance -- not "
                          "timing -- is the honest caveat",
         "admissible": "timing yes (cited); value-identity caveated; V-drop and "
                       "V-lag144 sensitivities reported per the precommit"},
        {"family": "Commercial vendor forecasts", "source": "vendor products",
         "first_publication": "publish D-1 afternoon", "measured": None,
         "gate_margin": "NEGATIVE", "revision_risk": "n/a", "admissible": False,
         "role": "excluded from every cell"},
    ]

    out = {"phase": 78, "title": "Feature lineage audit", "precommit_doc": PRECOMMIT,
           "buffers_min": list(BUFFERS_MIN),
           "run_dissemination": runs, "families": families,
           "phase54_variants": "pending: V-current / V-drop / V-lag144 are queued "
                               "behind the phase85 sweep and appended on landing",
           "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                        "generated_utc": datetime.now(UTC).isoformat()}}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s (%d families, %d measured runs)", OUT_JSON,
             len(families), len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
