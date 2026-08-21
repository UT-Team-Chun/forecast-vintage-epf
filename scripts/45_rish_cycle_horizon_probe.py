"""Phase 100: does the RISH MSM mirror publish long horizons for off-synoptic cycles?

Pre-registered question (fixed before any request is issued)
------------------------------------------------------------
The manuscript states that a D-2 18 UTC run cannot cover the whole of
delivery day D, because the mirror carries forecast hours beyond FH39 only
for the 00 and 12 UTC cycles.  That statement rests on a manual probe and a
code comment; **no artefact in this repository records a 404 for
``18Z x FH40-51``, and no code has ever requested that combination.**

This script settles it by asking the archive directly, for every cycle and
every surface horizon file, over several dates spanning the study window.

Pre-registered branches
-----------------------
A. ``18Z x FH40-51`` returns 200 on a majority of probed dates
   -> the mirror DOES carry a whole-day D-2 18 UTC run.  The manuscript's
      archive-gap caveat is wrong and must be retracted; a coherent rung
      4.7 h nearer the gate becomes constructible.
B. ``18Z x FH40-51`` returns 404 on every probed date, while the positive
   control ``00Z x FH40-51`` returns 200
   -> the caveat is correct and is now backed by a recorded artefact
      instead of a comment.
C. The positive control also fails
   -> the probe itself is broken; report UNDETERMINED and change nothing.

Branch C is what makes this honest: without a control, a 404 is
indistinguishable from a typo in the URL template.

Read-only.  Issues HTTP HEAD requests only; downloads nothing.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from pathlib import Path

from alg.config import EpfConfig
from alg.core.epf.data.jma_gpv_loader import RISH_BASE, _run_url

CFG = EpfConfig()

# Every surface horizon file the mirror is known to name, including the two
# the loader never requests for off-synoptic cycles.
FH_FILES = ("FH00-15", "FH16-33", "FH34-39", "FH40-51", "FH52-78")
CYCLES_UTC = (0, 3, 6, 9, 12, 15, 18, 21)

# Dates spanning the study window; deliberately spread across seasons and
# across the 2025-05-20 model upgrade so a transient outage cannot masquerade
# as a structural gap.
PROBE_DATES = (
    "2024-09-11",
    "2025-01-15",
    "2025-05-14",  # before the 2025-05-20 upgrade
    "2025-06-18",  # after it
    "2025-11-05",
    "2026-03-11",
    "2026-07-15",
)

OUT_DIR = CFG.project_root / "data" / "runs"
DELAY_S = 0.25  # be polite to a university archive


def main() -> None:
    rows: list[dict] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for date_s in PROBE_DATES:
            for hh in CYCLES_UTC:
                run_utc = datetime.fromisoformat(f"{date_s}T{hh:02d}:00:00").replace(
                    tzinfo=timezone.utc
                )
                for fh in FH_FILES:
                    url = _run_url(run_utc, fh)
                    try:
                        r = client.head(url)
                        status, size = r.status_code, r.headers.get("content-length")
                        err = ""
                    except Exception as exc:  # network failure is data, not a crash
                        status, size, err = -1, None, f"{type(exc).__name__}: {exc}"
                    rows.append(
                        {
                            "date": date_s,
                            "cycle_utc": hh,
                            "fh_file": fh,
                            "status": status,
                            "content_length": int(size) if size else None,
                            "error": err,
                            "url": url,
                        }
                    )
                    print(f"{date_s} {hh:02d}Z {fh:9s} -> {status}", flush=True)
                    time.sleep(DELAY_S)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq = OUT_DIR / "phase100_rish_cycle_horizon_matrix.parquet"
    df.to_parquet(pq, index=False)

    ok = df["status"] == 200
    matrix = (
        df.assign(ok=ok)
        .pivot_table(index="cycle_utc", columns="fh_file", values="ok", aggfunc="mean")
        .reindex(columns=list(FH_FILES))
    )

    n_dates = len(PROBE_DATES)
    target = df[(df.cycle_utc == 18) & (df.fh_file == "FH40-51")]
    control = df[(df.cycle_utc == 0) & (df.fh_file == "FH40-51")]
    target_ok = int((target.status == 200).sum())
    control_ok = int((control.status == 200).sum())

    if control_ok == 0:
        verdict = "C_UNDETERMINED_control_failed"
    elif target_ok > n_dates / 2:
        verdict = "A_18Z_CARRIES_WHOLE_DAY"
    elif target_ok == 0:
        verdict = "B_18Z_TRUNCATED_AS_STATED"
    else:
        verdict = "B_partial_18Z_intermittent"

    summary = {
        "verdict": verdict,
        "n_dates": n_dates,
        # Bound as a leaf so the prose count cannot drift from the matrix: how
        # many of the eight cycles carry ANY horizon beyond FH39.
        "n_cycles_with_long_horizon": int(
            sum(1 for _, row in matrix.iterrows()
                if any((row.get(f) or 0) > 0 for f in ("FH40-51", "FH52-78")))
        ),
        "n_cycles_probed": int(len(CYCLES_UTC)),
        "target_18z_fh40_51_ok": target_ok,
        "control_00z_fh40_51_ok": control_ok,
        "availability_by_cycle_and_horizon": {
            str(c): {f: (None if pd.isna(v) else float(v)) for f, v in row.items()}
            for c, row in matrix.iterrows()
        },
        "probe_dates": list(PROBE_DATES),
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "n_requests": int(len(df)),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=CFG.project_root).stdout.strip(),
        "rish_base": RISH_BASE,
        "script": Path(__file__).name,
    }
    js = OUT_DIR / "phase100_rish_cycle_horizon_probe.json"
    js.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== availability (fraction of probed dates returning 200) ===")
    print(matrix.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nVERDICT: {verdict}")
    print(f"  18Z x FH40-51 : {target_ok}/{n_dates} available")
    print(f"  00Z x FH40-51 : {control_ok}/{n_dates} available  (positive control)")
    print(f"\nwrote {pq}\nwrote {js}")


if __name__ == "__main__":
    main()
