r"""Phase 72: measure dissemination timing from the archive's own timestamps.

WHY THIS EXISTS
---------------
The manuscript's central claim is that admissibility must be judged by
DISSEMINATION time, and Figure 1's caption has carried a "measured +2 h 16 m"
lag for two revisions. A fifth-review audit found that no cache anywhere had
persisted the measurement: the figure was quoting a number whose evidence could
not be reconstructed. That is an unsourced measurement claim in a paper about
measurement timing.

This script makes the measurement real. It issues one HTTP HEAD request per
(delivery day, run, FH file) for the three run plans already cached on disk and
records the archive's ``Last-Modified`` for every file -- 881 days x 8 files =
~7,000 requests, no bodies downloaded. The two new cycle-curve plans (D-3 12Z,
D-2 18Z) record the same header during their own download sweeps, so together
every pinned run in the paper gets an observed timestamp.

WHAT THE TIMESTAMP MEANS, AND ITS DIRECTION OF BIAS
---------------------------------------------------
The RISH archive is a research mirror fed from the JMA dissemination stream, so
its ``Last-Modified`` is an UPPER BOUND on the true dissemination completion:
mirror-write happens after the feed delivers. The bias therefore runs in the
conservative direction for every pre-gate claim -- if the mirror shows a file
before the 10:00 gate, the true dissemination was also before it. For the
post-gate run the mirror timestamp alone cannot prove lateness; there the
JMBSC specification (init + 2 h 30 m = 11:30 JST) is the load-bearing source,
and the mirror timestamps corroborate it.

CLI (from the repository root):
  uv run python scripts/40_dissemination_audit.py
  ... --limit-days 5          # smoke test
  ... --analyse-only          # recompute stats from the checkpoint, no HTTP
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.jma_gpv_loader import _run_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase72")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
OUT_JSON = RUN / "phase72_dissemination_audit.json"
OUT_PARQUET = RUN / "phase72_dissemination_audit_per_file.parquet"
CKPT = RUN / "phase72_head_checkpoint.json"

JST = timezone(timedelta(hours=9), name="JST")
POLITE_SLEEP_S = 0.5
START = date(2024, 3, 1)
END = date(2026, 7, 29)
SPEC_LAG_MIN = 150          # JMBSC: dissemination completes init + 2 h 30 m

# (plan label, days_before, init_hour_utc, fh files, is the run pre-gate by spec)
# These are the runs the PAPER's tiers are built from. The pregate composite's
# 12 UTC top-up files are the same URLs as plan A's FH40-51, so they are not
# requested twice.
RUNS: tuple[tuple[str, int, int, tuple[str, ...], bool], ...] = (
    ("msm_d2_12z", 2, 12, ("FH16-33", "FH34-39", "FH40-51"), True),
    ("msm_d2_21z", 2, 21, ("FH16-33", "FH34-39"), True),
    ("msm_d1_00z", 1, 0, ("FH00-15", "FH16-33", "FH34-39"), False),
)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _j(x: float | None, nd: int = 2) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def sweep(limit_days: int | None) -> None:
    """HEAD every (day, run, file); checkpoint after each day."""
    ckpt: dict = (json.loads(CKPT.read_text()) if CKPT.exists()
                  else {"records": {}})
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    if limit_days:
        days = days[:limit_days]
    todo = [d for d in days if d.isoformat() not in ckpt["records"]]
    log.info("HEAD sweep: %d days requested, %d already done, %d to do "
             "(~%.1f h at %.1fs politeness)",
             len(days), len(days) - len(todo), len(todo),
             len(todo) * 8 * (POLITE_SLEEP_S + 0.2) / 3600, POLITE_SLEEP_S)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for i, d in enumerate(todo):
            recs: list[dict] = []
            for label, days_before, init_h, fh_files, pre_gate in RUNS:
                run_utc = (datetime(d.year, d.month, d.day, tzinfo=UTC)
                           - timedelta(days=days_before)).replace(hour=init_h)
                for fh in fh_files:
                    url = _run_url(run_utc, fh)
                    try:
                        r = client.head(url)
                        lm = r.headers.get("Last-Modified")
                        recs.append({"run": label, "fh": fh, "status": r.status_code,
                                     "init_utc": run_utc.isoformat(),
                                     "last_modified": lm,
                                     "content_length": r.headers.get("Content-Length"),
                                     "pre_gate_by_spec": pre_gate})
                    except httpx.HTTPError as exc:
                        recs.append({"run": label, "fh": fh, "status": -1,
                                     "init_utc": run_utc.isoformat(),
                                     "last_modified": None, "content_length": None,
                                     "pre_gate_by_spec": pre_gate,
                                     "error": str(exc)[:120]})
                    time.sleep(POLITE_SLEEP_S)
            ckpt["records"][d.isoformat()] = recs
            if i % 10 == 0 or i == len(todo) - 1:
                CKPT.write_text(json.dumps(ckpt))
                log.info("day %s done (%d/%d)", d, i + 1, len(todo))
    CKPT.write_text(json.dumps(ckpt))
    log.info("sweep complete: %d days checkpointed", len(ckpt["records"]))


def analyse() -> int:
    ckpt = json.loads(CKPT.read_text())
    rows: list[dict] = []
    for day, recs in ckpt["records"].items():
        for r in recs:
            lag_min, lm_iso = None, None
            if r.get("last_modified"):
                lm = parsedate_to_datetime(r["last_modified"])
                init = datetime.fromisoformat(r["init_utc"])
                lag_min = (lm - init).total_seconds() / 60.0
                lm_iso = lm.isoformat()
            rows.append({"delivery_day": day, "run": r["run"], "fh": r["fh"],
                         "status": r["status"], "init_utc": r["init_utc"],
                         "last_modified_utc": lm_iso, "lag_minutes": lag_min,
                         "pre_gate_by_spec": r["pre_gate_by_spec"]})
    df = pl.DataFrame(rows)
    df.write_parquet(OUT_PARQUET)
    ok = df.filter(pl.col("status") == 200)

    per_run: dict = {}
    for run_label in ok["run"].unique().to_list():
        sub = ok.filter(pl.col("run") == run_label).drop_nulls("lag_minutes")
        lag = sub["lag_minutes"]
        # A run is IN HAND only when its LAST file has arrived; the per-run lag
        # is the max over its files for that day.
        day_max = (sub.group_by("delivery_day")
                      .agg(pl.col("lag_minutes").max().alias("lag_run"))
                      .drop_nulls("lag_run"))
        lr = day_max["lag_run"]
        per_run[run_label] = {
            "n_files": int(sub.height),
            "n_days_complete": int(day_max.height),
            "per_file_lag_minutes": {
                "median": _j(float(lag.median())), "p95": _j(float(lag.quantile(0.95))),
                "p99": _j(float(lag.quantile(0.99))), "max": _j(float(lag.max())),
                "min": _j(float(lag.min()))},
            "per_run_completion_lag_minutes": {
                "median": _j(float(lr.median())), "p95": _j(float(lr.quantile(0.95))),
                "p99": _j(float(lr.quantile(0.99))), "max": _j(float(lr.max()))},
            "share_days_within_spec_150min": _j(float((lr <= SPEC_LAG_MIN).mean()), 4),
        }

    # The admissibility question: was every 21Z file on the mirror before the
    # 10:00 JST gate on D-1? The gate is D-1 01:00 UTC, so gate - init is
    # 4 h for the D-2 21 UTC run and 13 h for the D-2 12 UTC run.
    # CORRECTION 2026-08-03: the first committed version added 9 h instead of
    # subtracting when converting 10:00 JST to UTC, inflating both constants
    # by 18 h (22 h / 31 h). The audit workflow caught it; the before-gate
    # verdicts were unaffected (worst lag 159 min << even the true 240 min)
    # but the printed worst margins were wrong.
    gate_checks: dict = {}
    for run_label, gate_lag_min in (("msm_d2_21z", 4 * 60), ("msm_d2_12z", 13 * 60)):
        sub = (ok.filter(pl.col("run") == run_label).drop_nulls("lag_minutes")
                 .group_by("delivery_day")
                 .agg(pl.col("lag_minutes").max().alias("lag_run")))
        n = int(sub.height)
        n_before = int(sub.filter(pl.col("lag_run") <= gate_lag_min).height)
        worst = float(sub["lag_run"].max()) if n else None
        gate_checks[run_label] = {
            "gate_minus_init_minutes": gate_lag_min,
            "n_days": n, "n_days_all_files_before_gate": n_before,
            "share_before_gate": _j(n_before / n, 6) if n else None,
            "worst_completion_lag_minutes": _j(worst),
            "worst_margin_minutes": _j(gate_lag_min - worst) if worst else None,
        }

    missing = df.filter(pl.col("status") != 200)
    out = {
        "phase": 72,
        "title": "Dissemination timing measured from the archive's own timestamps",
        "method": "one HTTP HEAD per (delivery day, run, FH file); Last-Modified "
                  "recorded; no bodies downloaded",
        "interpretation": {
            "what_the_timestamp_is": "the RISH mirror's file-write time, an UPPER "
                "BOUND on true dissemination completion (mirror-write follows the "
                "feed)",
            "direction_of_bias": "conservative for every pre-gate claim: a mirror "
                "timestamp before the gate implies true dissemination before the "
                "gate. For the post-gate run the mirror alone cannot prove "
                "lateness; the JMBSC specification (init + 2 h 30 m) is the "
                "load-bearing source there, and these measurements corroborate it",
            "spec_lag_minutes": SPEC_LAG_MIN,
        },
        "window": [START.isoformat(), END.isoformat()],
        "totals": {"n_file_records": int(df.height),
                   "n_ok": int(ok.height),
                   "n_days": int(df["delivery_day"].n_unique())},
        "per_run": per_run,
        "gate_checks": gate_checks,
        "missing_or_error_files": {
            "n": int(missing.height),
            "by_run": ({} if missing.height == 0 else
                       {k: int(v) for k, v in missing.group_by("run").len()
                        .iter_rows()}),
            "days": sorted(missing["delivery_day"].unique().to_list())[:50],
        },
        "per_file_parquet": OUT_PARQUET.name,
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "generated_utc": datetime.now(UTC).isoformat()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("wrote %s (%d file records)", OUT_JSON, df.height)
    for run_label, st in per_run.items():
        log.info("  %-12s per-run completion lag: median %.0f min, p99 %.0f, "
                 "max %.0f; within spec %.2f%%", run_label,
                 st["per_run_completion_lag_minutes"]["median"],
                 st["per_run_completion_lag_minutes"]["p99"],
                 st["per_run_completion_lag_minutes"]["max"],
                 100 * st["share_days_within_spec_150min"])
    for run_label, gc in gate_checks.items():
        log.info("  %-12s before gate: %s/%s days (worst margin %.0f min)",
                 run_label, gc["n_days_all_files_before_gate"], gc["n_days"],
                 gc["worst_margin_minutes"] or float("nan"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=None)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    if not args.analyse_only:
        sweep(args.limit_days)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
