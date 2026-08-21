"""Phase 105: file-by-file arrival against the 10:00 JST gate.

Pre-registered question (fixed before computing)
------------------------------------------------
The reviewer's first-priority objection: the paper treats the D-1 00 UTC run as
inadmissible on the basis of a *median* dissemination lag, but a median says
nothing about the tail.  Perhaps some forecast-hour files, on some days, did
arrive before 10:00 JST -- in which case a bidder could have used part of the
run and the paper's central admissibility claim is too strong.

This asks the question at the only resolution that can answer it: every
individual file, on every day, against the gate for its own delivery day.

Pre-registered branches
-----------------------
A. At least one file on at least one day arrives before its gate
   -> the admissibility claim must be qualified, and the paper must report the
      share of days and which forecast-hour files were usable.
B. No file on any day arrives before its gate
   -> the claim holds at file resolution, and the margin becomes reportable:
      how late is the EARLIEST file across the whole archive?
C. Forecast-hour files differ systematically, with early ones beating the gate
   -> a partial-run tier is constructible and must be built.

Branch C is the one the reviewer actually proposed, so the per-forecast-hour
breakdown is reported whatever the aggregate says.

Read-only over the phase72 audit; computes no forecast.
"""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta

import pandas as pd

from alg.config import EpfConfig

CFG = EpfConfig()
IN_PQ = CFG.project_root / "data" / "runs" / "phase72_dissemination_audit_per_file.parquet"
OUT_DIR = CFG.project_root / "data" / "runs"

# The day-ahead gate closes 10:00 JST on D-1 for delivery day D. JST = UTC+9,
# so the gate is D-1 01:00 UTC. Expressed in UTC throughout to keep the
# comparison free of local-time arithmetic.
GATE_HOUR_UTC = 1
JST = timedelta(hours=9)


def main() -> None:
    df = pd.read_parquet(IN_PQ)
    df["arrival_utc"] = pd.to_datetime(df["last_modified_utc"], utc=True)
    day = pd.to_datetime(df["delivery_day"]).dt.tz_localize("UTC")
    df["gate_utc"] = day - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_HOUR_UTC)
    # Positive = the file landed AFTER the gate had closed.
    df["min_after_gate"] = (df["arrival_utc"] - df["gate_utc"]).dt.total_seconds() / 60.0
    df["arrival_jst"] = df["arrival_utc"] + JST

    assert (df["status"] == 200).all(), "audit contains non-200 rows; the margin would be wrong"

    rows = []
    for (run, fh), g in df.groupby(["run", "fh"], sort=True):
        rows.append({
            "run": run, "fh": fh, "n_files": int(len(g)),
            "n_days": int(g["delivery_day"].nunique()),
            "earliest_min_after_gate": float(g["min_after_gate"].min()),
            "p01_min_after_gate": float(g["min_after_gate"].quantile(0.01)),
            "median_min_after_gate": float(g["min_after_gate"].median()),
            "latest_min_after_gate": float(g["min_after_gate"].max()),
            "n_files_before_gate": int((g["min_after_gate"] < 0).sum()),
            "earliest_arrival_jst": str(g.loc[g["min_after_gate"].idxmin(), "arrival_jst"]),
        })
    per_cell = pd.DataFrame(rows)

    # Day-level: on how many days did ANY file of a run beat that day's gate?
    day_any = (df.assign(before=df["min_after_gate"] < 0)
                 .groupby(["run", "delivery_day"])["before"].any()
                 .groupby("run").agg(days="count", days_any_pre_gate="sum"))

    summary = {}
    for run, g in df.groupby("run"):
        fh_med = g.groupby("fh")["min_after_gate"].median()
        summary[run] = {
            "n_files": int(len(g)),
            "n_days": int(g["delivery_day"].nunique()),
            "earliest_min_after_gate": float(g["min_after_gate"].min()),
            "earliest_arrival_jst": str(g.loc[g["min_after_gate"].idxmin(), "arrival_jst"]),
            "median_min_after_gate": float(g["min_after_gate"].median()),
            # Bound as leaves rather than computed in prose: the SPREAD is the
            # number that ends the tail argument, so it must be checkable.
            "latest_min_after_gate": float(g["min_after_gate"].max()),
            "arrival_spread_min": float(g["min_after_gate"].max()
                                        - g["min_after_gate"].min()),
            "n_files_before_gate": int((g["min_after_gate"] < 0).sum()),
            "n_days_with_any_pre_gate_file": int(day_any.loc[run, "days_any_pre_gate"]),
            # Branch C: is there an early-arriving SUBSET of forecast-hour files?
            "fh_median_spread_seconds": float((fh_med.max() - fh_med.min()) * 60.0),
            "fh_medians_min_after_gate": {k: float(v) for k, v in fh_med.items()},
        }

    gated = summary.get("msm_d1_00z", {})
    if gated.get("n_days_with_any_pre_gate_file", 0) > 0:
        verdict = "A_SOME_FILES_ADMISSIBLE_claim_must_be_qualified"
    elif gated.get("fh_median_spread_seconds", 0.0) > 600:
        verdict = "C_FH_FILES_DIFFER_check_for_partial_run_tier"
    else:
        verdict = "B_NO_FILE_EVER_BEAT_THE_GATE"

    out = {
        "verdict": verdict,
        "per_run": summary,
        "per_run_fh": per_cell.to_dict(orient="records"),
        "gate_definition": "10:00 JST on D-1 for delivery day D (= D-1 01:00 UTC)",
        "input": str(IN_PQ.relative_to(CFG.project_root)),
        "n_files_total": int(len(df)),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=CFG.project_root).stdout.strip(),
        "script": "scripts/46_file_level_arrival.py",
    }
    js = OUT_DIR / "phase105_file_level_arrival.json"
    js.write_text(json.dumps(out, indent=2))
    per_cell.to_parquet(OUT_DIR / "phase105_file_level_arrival.parquet", index=False)

    print(per_cell.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nVERDICT: {verdict}\n")
    for run, s in summary.items():
        print(f"{run}:")
        print(f"   files {s['n_files']:5d} over {s['n_days']} days | "
              f"files before gate {s['n_files_before_gate']} | "
              f"days with ANY pre-gate file {s['n_days_with_any_pre_gate_file']}")
        print(f"   earliest arrival {s['earliest_arrival_jst']} "
              f"= {s['earliest_min_after_gate']:+.2f} min vs gate | "
              f"median {s['median_min_after_gate']:+.1f} min")
        print(f"   fh median spread {s['fh_median_spread_seconds']:.1f} s")
    print(f"\nwrote {js}")


if __name__ == "__main__":
    main()
