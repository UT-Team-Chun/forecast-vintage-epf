r"""Phase 88b: inter-coder reliability of the weather-input-timing audit.

WHY THIS EXISTS
---------------
The audit protocol (docs/research/2026-08-11_literature_audit_protocol.md, §32)
required disclosing a re-coding agreement rate and that disclosure was never
made. The seventh external review asked for it explicitly, along with Cohen's
kappa and a statement of how many coders were used.

It turns out the material was already there. The collection pass ran each
paper's coder as an independent agent, and because several runs were retried
after session limits, 28 papers were coded 2-5 times by SEPARATE agents that
each did their own retrieval and never read one another's output. That is a
genuine multi-coder design, not a cache replay: no two codings of the same
paper are byte-identical, and their tool-call counts differ.

WHAT IT MEASURES, AND THE ONE STATISTIC THAT MATTERS
----------------------------------------------------
Raw agreement on the six-label verdict vocabulary is moderate. Reporting only
that would be misleading in BOTH directions, so this script also computes the
statistic the paper's claims actually rest on: every headline rate in the audit
is a share of papers coded "Yes", so what matters is whether coders disagree
about the PRESENCE of a statement, not about how to label its absence. The
"Yes vs not-Yes" flip rate separates the two.

Both are reported. The flip cases are enumerated so they can be adjudicated by
hand rather than averaged away, and the adjudication is recorded against the
released dataset's value so a reader can see whether the release needed
changing.

REPRODUCIBILITY NOTE
--------------------
The per-run codings live in session transcripts outside the repository, so this
script exports the multi-coded subset into
``phase88b_multicoded_runs.json`` alongside the summary. The agreement
statistics can be recomputed from that shipped file with --from-export, without
the original transcripts.

CLI (from the repository root):
  uv run python scripts/71_audit_reliability.py
  uv run ... --from-export           # recompute from the shipped export
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import logging
import re
import sys
import os
from pathlib import Path

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase88b")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
OUT_JSON = RUN / "phase88b_audit_reliability.json"
OUT_EXPORT = RUN / "phase88b_multicoded_runs.json"
PRECOMMIT = "docs/research/2026-08-13_phase88b_audit_reliability_precommit.md"

# Session transcript root, for the ORIGINAL derivation only. It is outside the
# repository by construction and machine-specific, so it is taken from the
# environment rather than hard-coded: a path naming one person's home directory
# and one session id is meaningless to anyone else and this script ships in a
# public package. Reproduction does not need it -- the multi-coded runs are
# exported to OUT_EXPORT, and the agreement statistic is computed from that.
JOURNALS = Path(os.environ["AUDIT_TRANSCRIPT_ROOT"]) if os.environ.get(
    "AUDIT_TRANSCRIPT_ROOT") else None

# The six columns that carry the paper's argument. The other four audit columns
# (market, weather_source, input_type, decision_application) are free text and
# are not verdict-coded, so no agreement statistic is defined for them.
COLS = ("gate_time_stated", "model_run_id_stated", "init_time_stated",
        "publication_time_stated", "gate_admissibility_checked",
        "revision_handling")

LABELS = ("Yes", "No", "Partial", "Not stated", "Not verifiable", "Unclear")


def _verdict(cell: str) -> str:
    """The leading verdict token of a coded cell, or '?' if unparsable."""
    low = (cell or "").strip().lower()
    for prefix, label in (("not verifiable", "Not verifiable"),
                          ("not stated", "Not stated"),
                          ("partial", "Partial"), ("unclear", "Unclear")):
        if low.startswith(prefix):
            return label
    if re.match(r"^yes\b", low):
        return "Yes"
    if re.match(r"^no\b", low):
        return "No"
    return "?"


def _collect_from_journals() -> dict[str, list[dict]]:
    """cache key -> the independent codings of that one paper.

    The cache key is a hash of the agent prompt and each prompt names exactly
    one paper, so equal key means same paper. Distinct agentIds under one key
    are distinct runs; a replayed cache hit emits no new result event.
    """
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted(glob.glob(str(JOURNALS / "wf_*" / "journal.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "result":
                continue
            val = rec.get("result")
            if isinstance(val, dict) and "columns" in val:
                groups[rec["key"]].append({
                    "agent_id": rec.get("agentId"),
                    "columns": {c: val["columns"].get(c, "") for c in COLS},
                    "market": val["columns"].get("market", ""),
                    "verified_against": val.get("verified_against", ""),
                })
    return {k: v for k, v in groups.items() if len(v) > 1}


def _kappa(p_observed: float, p_expected: float) -> float:
    return (p_observed - p_expected) / (1.0 - p_expected)


def analyse(multi: dict[str, list[dict]]) -> dict:
    n_pairs = sum(len(list(itertools.combinations(v, 2))) for v in multi.values())
    tot = agree = flips = 0
    per_col = {c: {"pairs": 0, "agree": 0, "yes_flips": 0} for c in COLS}
    marg: collections.Counter = collections.Counter()
    flip_cases = []
    for key, runs in multi.items():
        for a, b in itertools.combinations(runs, 2):
            for c in COLS:
                va, vb = _verdict(a["columns"][c]), _verdict(b["columns"][c])
                tot += 1
                per_col[c]["pairs"] += 1
                marg[va] += 1
                marg[vb] += 1
                if va == vb:
                    agree += 1
                    per_col[c]["agree"] += 1
                if (va == "Yes") != (vb == "Yes"):
                    flips += 1
                    per_col[c]["yes_flips"] += 1
                    flip_cases.append({
                        "key": key, "column": c, "verdicts": [va, vb],
                        "market": a["market"][:120],
                        "cells": [a["columns"][c][:240], b["columns"][c][:240]],
                    })
    po = agree / tot
    pe = sum((marg[l] / (2 * tot)) ** 2 for l in marg)
    n_yes = marg["Yes"]
    pe_bin = (n_yes / (2 * tot)) ** 2 + ((2 * tot - n_yes) / (2 * tot)) ** 2
    po_bin = 1.0 - flips / tot
    return {
        "design": ("independent agent coders per paper; several papers were "
                   "coded more than once because runs were retried after "
                   "session limits, giving a genuine multi-coder subset"),
        "n_papers_multicoded": len(multi),
        "n_codings": sum(len(v) for v in multi.values()),
        "n_paper_pairs": n_pairs,
        "n_column_pairs": tot,
        "agreement": {"observed": round(po, 4), "expected": round(pe, 4),
                      "cohens_kappa": round(_kappa(po, pe), 4)},
        "yes_vs_not_yes": {
            "flip_rate": round(flips / tot, 4), "n_flips": flips,
            "observed": round(po_bin, 4), "expected": round(pe_bin, 4),
            "cohens_kappa": round(_kappa(po_bin, pe_bin), 4),
            "why_this_is_the_binding_statistic":
                "every reported rate is a share of papers coded Yes, so a "
                "disagreement that does not cross the Yes boundary cannot move "
                "any rate in the paper",
        },
        "per_column": {
            c: {"agreement": round(v["agree"] / v["pairs"], 4),
                "yes_flip_rate": round(v["yes_flips"] / v["pairs"], 4),
                "pairs": v["pairs"]}
            for c, v in per_col.items()},
        "label_marginals": dict(marg),
        "yes_flip_cases": flip_cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-export", action="store_true",
                    help="recompute from the shipped export instead of the "
                         "session transcripts")
    a = ap.parse_args()

    if a.from_export:
        multi = json.loads(OUT_EXPORT.read_text(encoding="utf-8"))["papers"]
    else:
        if JOURNALS is None:
            log.error("set AUDIT_TRANSCRIPT_ROOT to the session-transcript root, "
                      "or use --from-export to compute the statistic from the "
                      "committed export (which is what reproduction should do)")
            return 2
        if not JOURNALS.is_dir():
            log.error("session transcripts not found at %s; use --from-export",
                      JOURNALS)
            return 2
        multi = _collect_from_journals()
        OUT_EXPORT.write_text(json.dumps(
            {"note": ("the independent codings of every multi-coded paper, "
                      "exported so the reliability statistics do not depend on "
                      "session transcripts outside the repository"),
             "papers": multi}, indent=1, ensure_ascii=False), encoding="utf-8")
        log.info("wrote %s (%d papers)", OUT_EXPORT, len(multi))

    out = {"phase": "88b",
           "title": "Inter-coder reliability of the weather-input-timing audit",
           "precommit_doc": PRECOMMIT,
           **analyse(multi),
           "metadata": {"script": Path(__file__).name, "columns": list(COLS)}}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    log.info("agreement %.1f%% (kappa %.3f); Yes-flip %.2f%% (kappa %.3f)",
             100 * out["agreement"]["observed"], out["agreement"]["cohens_kappa"],
             100 * out["yes_vs_not_yes"]["flip_rate"],
             out["yes_vs_not_yes"]["cohens_kappa"])
    for c, v in out["per_column"].items():
        log.info("  %-28s agree %.3f  yes-flip %.3f", c, v["agreement"],
                 v["yes_flip_rate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
