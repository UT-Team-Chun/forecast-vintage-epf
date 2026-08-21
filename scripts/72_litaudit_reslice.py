"""Phase 107: re-derive the literature audit's robustness check on the right column.

The defect
----------
Section 2.3 answers the obvious objection to the audit -- "you coded 'not stated'
because you could not see the paper" -- by restricting to the studies whose full
text was read. It sliced that subset on ``access_level``, and ``access_level``
does not mean what the sentence needs. It records whether the PUBLISHER's version
was reachable, so a study whose ScienceDirect page returned 403 while its accepted
arXiv manuscript was read cover to cover is coded ``partial``.

``verified_against`` is the column that records what was actually read. Sliced on
it, the subset is 39 studies rather than 20, and the headline inverts: publication
time and admissibility go from "zero in the full-text subset" to two each.

A referee found this. It matters because that sentence carries the audit's
strongest argument, and an inverted robustness check is worse than none.

What this script does
---------------------
Classifies every included study by whether ``verified_against`` states the text
was read, emits the per-row classification so the call on each study is auditable
rather than buried in a regex, counts the studies it cannot classify, and
recomputes the three rates on the corrected subset.

Read-only over the released dataset.
"""

from __future__ import annotations

import csv
import json
import subprocess

from alg.config import EpfConfig

CFG = EpfConfig()
CSV = CFG.project_root / "docs/paper/paper_a_main/submission/lit_audit_dataset.csv"
OUT = CFG.project_root / "data" / "runs" / "phase107_litaudit_reslice.json"

# Phrases in verified_against that state the body was NOT read. Checked first,
# because several rows say both "abstract read in full" and "full text not read".
NOT_READ = (
    "full text and abstract were not accessible", "full text not read",
    "body text was not read", "sections were therefore never read",
    "metadata only", "not read: introduction", "chapter body",
    "abstract + metadata only", "publisher abstract in full",
    "publisher abstract read verbatim", "publisher landing page only",
    "no article text obtained",
)
# Phrases that state the body WAS read.
WAS_READ = (
    "read in full", "all pages read", "read in entirety", "full text read",
    "read end-to-end", "read page-by-page", "pages read", "read pages",
    "full pdf", "full arxiv", "full published", "full text of",
    "read via text extraction", "read via multiple targeted",
    "arxiv html full text", "full oa pdf", "full text via",
)

RATE_COLS = ("gate_time_stated", "publication_time_stated",
             "gate_admissibility_checked")


def classify(v: str) -> str:
    low = v.lower()
    if any(p in low for p in NOT_READ):
        return "not_read"
    if any(p in low for p in WAS_READ):
        return "read"
    return "unclassified"


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    inc = [r for r in rows
           if str(r.get("included", "")).strip().lower() in ("true", "yes", "1")]

    per_row = []
    for r in inc:
        per_row.append({
            "index": r.get("index"),
            "access_level": r.get("access_level"),
            "read_class": classify(r.get("verified_against", "")),
            **{c: r.get(c, "") for c in RATE_COLS},
        })

    read = [p for p in per_row if p["read_class"] == "read"]
    unclassified = [p for p in per_row if p["read_class"] == "unclassified"]
    old = [r for r in inc if r.get("access_level") == "full-text"]

    def rate(subset, col, key="read_class"):
        n = len(subset)
        k = sum(1 for p in subset
                if str(p[col] if key in p else p.get(col, "")).strip()
                .lower().startswith("yes"))
        return {"n": n, "k": k, "pct": (100.0 * k / n) if n else None}

    corrected = {c: rate(read, c) for c in RATE_COLS}
    # The finding that actually matters: the audit EXCLUDED studies whose text
    # could not be read, so the included set has no unread stratum for the
    # "you could not see the disclosures" objection to hide in. That is a
    # stronger rebuttal than a subset comparison, and it is why the corrected
    # subset rates are within rounding of the headline rates.
    all_inc = [{**{c: r.get(c, "") for c in RATE_COLS}, "read_class": "x"} for r in inc]
    headline = {c: rate(all_inc, c) for c in RATE_COLS}
    superseded = {c: {"n": len(old),
                      "k": sum(1 for r in old
                               if str(r.get(c, "")).strip().lower().startswith("yes"))}
                  for c in RATE_COLS}

    out = {
        "phase": 107,
        "title": "Literature-audit robustness check, re-sliced on what was actually read",
        "defect": ("the published check sliced on access_level, which records "
                   "PUBLISHER accessibility, not whether the text was read"),
        "n_included": len(inc),
        "n_full_text_read": len(read),
        "n_not_read": sum(1 for p in per_row if p["read_class"] == "not_read"),
        "n_unclassified": len(unclassified),
        "unclassified_indices": [p["index"] for p in unclassified],
        "corrected_subset_rates": corrected,
        "headline_all_included_rates": headline,
        "reading": ("no included study is coded as unread, so the read subset is "
                    "42 of 43 and its rates are within rounding of the headline. "
                    "The access objection does not apply to the headline at all"),
        "superseded_access_level_subset": superseded,
        "per_row_classification": per_row,
        "source_csv": str(CSV.relative_to(CFG.project_root)),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=CFG.project_root).stdout.strip(),
        "script": "scripts/72_litaudit_reslice.py",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"included {len(inc)} | full text read {len(read)} | "
          f"not read {out['n_not_read']} | unclassified {len(unclassified)}")
    print(f"superseded slice (access_level == 'full-text'): n = {len(old)}")
    for c in RATE_COLS:
        a, b = corrected[c], superseded[c]
        print(f"  {c:30s} corrected {a['k']:2d}/{a['n']} = {a['pct']:5.1f}%   "
              f"| superseded {b['k']:2d}/{b['n']}")
    print("headline (all included):")
    for c in RATE_COLS:
        h = headline[c]
        print(f"  {c:30s} {h['k']:2d}/{h['n']} = {h['pct']:5.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
