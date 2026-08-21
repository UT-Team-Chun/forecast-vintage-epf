r"""Phase 110: export the slot-dispersion descriptives Paper A cites, and nothing else.

THE DEFECT
----------
``\DecShareGtTickPct`` (97.7) and ``\DecMeanAbsDy`` (0.48) are rendered in the
main article -- ae_05_results.tex:83 and ae_06_discussion.tex:68 -- and they are
the sentence that keeps the equivalence claim honest: pooled-MAE equivalence
within one tick does NOT mean the two runs agree slot by slot, and these two
numbers say by how much they disagree.

Both are defined inside the paper-B block of ``macros.tex`` and both come from
``phase75_decision_equivalence.json``, which the public replication package does
not ship, because that artefact also carries the commitment and profit gates that
moved to the companion manuscript. So the main article's Data availability
statement -- "the frozen per-result provenance every number in this paper is
verified against" -- was false for exactly two rendered quantities.

Shipping the whole phase75 artefact would fix the sentence by publishing
settlement content into a package whose provenance is meant to contain none.
Dropping the two numbers would fix it by removing the concession that makes the
equivalence claim defensible. Neither is acceptable.

WHAT THIS DOES
--------------
A field-subset export. It reads the frozen phase75 artefact and writes the
PRICE-FORECAST-DISPERSION descriptives only, plus the panel description and the
original provenance metadata, under a name that says what it is. No value is
recomputed, rescaled or rounded: the numbers are copied. The commitment
descriptives, the floor-split commitment values, and the nomination and profit
gates are not carried over -- they are the companion paper's.

The result is a Paper-A-scoped artefact that the guard can bind both macros
against and that the packaging tool can ship, after which the Data availability
statement is true as written.

Positive control: the export must reproduce the two macro values exactly
(0.97665 -> 97.7 at one decimal, 0.48398 -> 0.48 at two), and must contain no key
whose name matches the settlement vocabulary. Both are asserted below; the script
exits non-zero if either fails, so a future edit to the source artefact cannot
silently change what Paper A cites.

CLI (from the repository root)
  cd <repository root>
  uv run python scripts/56_slot_dispersion.py
  uv run python \
      scripts/56_slot_dispersion.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase110")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
SRC = RUN / "phase75_decision_equivalence.json"
OUT = RUN / "phase110_slot_dispersion.json"

# Field names that would mean the companion paper's content had leaked in.
FORBIDDEN = ("commitment", "profit", "nomination", "cnpe", "settlement",
             "revenue", "imbalance", "dq", "voi")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True, cwd=CFG.project_root,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build() -> dict:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    dispersion = src["descriptives"]["per_slot_abs_price_forecast_diff_jpykwh"]
    out = {
        "phase": 110,
        "title": "Per-slot price-forecast dispersion between the pre-gate and "
                 "post-gate runs (field subset of phase 75)",
        "derivation": (
            "A field subset. Every value is copied verbatim from "
            "phase75_decision_equivalence.json; nothing is recomputed. The "
            "commitment, nomination and profit quantities of that artefact "
            "belong to the companion manuscript and are deliberately absent "
            "here, which is why this file exists separately."
        ),
        "panel": src["panel"],
        "per_slot_abs_price_forecast_diff_jpykwh": dispersion,
        "reading": (
            "Pooled-MAE equivalence within one price tick does not imply the two "
            "runs agree slot by slot. These are the per-slot differences that "
            "equivalence cannot bound, reported unconditionally."
        ),
        "source_artefact": {
            "file": SRC.name,
            "script": src["metadata"]["script"],
            "git_sha": src["metadata"]["git_sha"],
            "generated_utc": src["metadata"]["generated_utc"],
        },
        "metadata": {
            "script": f"{Path(__file__).name}",
            "git_sha": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
        },
    }
    # Positive control 1: the two macro values must survive the copy exactly.
    share = dispersion["share_gt_one_tick"] * 100.0
    mean = dispersion["mean"]
    if round(share, 1) != 97.7:
        msg = f"share_gt_one_tick gives {share:.4f}%, expected 97.7 (\\DecShareGtTickPct)"
        raise SystemExit(msg)
    if round(mean, 2) != 0.48:
        msg = f"mean gives {mean:.5f}, expected 0.48 (\\DecMeanAbsDy)"
        raise SystemExit(msg)
    # Positive control 2: no companion-paper field may have come along.
    blob = json.dumps({k: v for k, v in out.items()
                       if k not in ("derivation", "reading", "source_artefact")}).lower()
    leaked = [w for w in FORBIDDEN if w in blob]
    if leaked:
        msg = f"companion-paper vocabulary in the export: {leaked}"
        raise SystemExit(msg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the artefact on disk, write nothing")
    a = ap.parse_args()
    if not SRC.exists():
        # This is expected in the public replication package. phase75 also carries
        # the commitment and profit gates that belong to the companion manuscript,
        # so shipping it would publish settlement content into a package whose
        # provenance is meant to contain none. The OUTPUT of this script ships,
        # and the script ships with it, so a reader can see exactly which fields
        # were carried over and check the two assertions in build() by eye.
        log.error("missing %s", SRC.name)
        log.error("  This script derives its output from an artefact of the "
                  "companion settlement manuscript, which this repository does "
                  "not ship. The derived artefact itself is present at %s and is "
                  "what the guard checks.", OUT.name)
        return 2
    out = build()
    if a.check:
        if not OUT.exists():
            log.error("missing %s", OUT)
            return 2
        have = json.loads(OUT.read_text(encoding="utf-8"))
        drop = {"metadata"}
        if {k: v for k, v in have.items() if k not in drop} != \
           {k: v for k, v in out.items() if k not in drop}:
            log.error("%s disagrees with a fresh export of %s", OUT.name, SRC.name)
            return 2
        log.info("%s matches a fresh export (both positive controls passed)", OUT.name)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
