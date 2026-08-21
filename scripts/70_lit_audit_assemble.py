r"""Phase 88: assemble the weather-input-timing literature audit.

Reads the per-paper codings produced by the agent-assisted verification pass
(one JSON row per paper, each carrying the ten audit columns, the evidence
quotes, and an explicit record of WHAT TEXT WAS READ), and emits:

  submission/lit_audit_dataset.csv   the machine-readable Supplementary Dataset
  data/runs/phase88_lit_audit.json   provenance for the macros the paper cites

The coding scheme deliberately separates "the paper says X" from "we could not
check": a cell is only counted in a denominator when the coder had access to
enough published text to answer it. Rows whose access was abstract-only are
carried in the dataset with their access level recorded and excluded from the
rate denominators, so the headline rates are rates over VERIFIED text, not over
titles.

CLI (from the repository root):
  uv run --extra research python scripts/70_lit_audit_assemble.py \
      --rows /path/to/lit_coded_all.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase88")

CFG = EpfConfig()
ROOT = CFG.project_root
OUT_CSV = ROOT / "docs/paper/paper_a_main/submission/lit_audit_dataset.csv"
OUT_JSON = CFG.resolved("run_dir") / "phase88_lit_audit.json"

COLUMNS = ("market", "gate_time_stated", "weather_source", "input_type",
           "model_run_id_stated", "init_time_stated", "publication_time_stated",
           "gate_admissibility_checked", "revision_handling",
           "decision_application")

# The coders were told to lead each cell with one of these verdicts. Parsing is
# deliberately conservative: anything that does not start with a recognised
# verdict is classified "unparsed" and reported, never silently bucketed.
VERDICTS = ("Yes", "No", "Partial", "Not stated", "Not verifiable", "Unclear")


def _verdict(cell: str) -> str:
    """Classify a coded cell into one verdict, or 'unparsed'."""
    low = (cell or "").strip().lower()
    prefixes = {"not verifiable": "Not verifiable", "not stated": "Not stated",
                "partial": "Partial", "unclear": "Unclear"}
    for prefix, label in prefixes.items():
        if low.startswith(prefix):
            return label
    if re.match(r"^yes\b", low):
        return "Yes"
    if re.match(r"^no\b", low):
        return "No"
    # Free-text cells (market, weather_source, input_type) carry content, not a
    # verdict; the caller decides whether a column is verdict-coded.
    return "unparsed"


VERDICT_COLUMNS = ("gate_time_stated", "model_run_id_stated", "init_time_stated",
                   "publication_time_stated", "gate_admissibility_checked",
                   "revision_handling")


def _access_level(row: dict) -> str:
    """How much published text the coding rests on, from the coder's own note.

    Negative evidence is checked FIRST and wins. An earlier version matched
    "in full" and "complete" anywhere in the note, which promoted two rows to
    full-text whose notes read "publisher abstract in full ... body text was
    not read" and "complete 22-entry reference list ... chapter body is
    PAYWALLED". A classifier that upgrades a paper on the strength of the word
    "complete" is worse than no classifier, because the access column is what
    tells a reader how much to trust the row.
    """
    va = (row.get("verified_against") or "").lower()
    if not va:
        return "unrecorded"
    blocked = ("not accessible", "no pdf", "full text not read",
               "full text was not read", "body text was not read",
               "body text not read", "paywalled", "no oa full text",
               "closed access", "abstract-only", "abstract only",
               "metadata only", "no article text", "not read directly",
               "full published pdf not accessible")
    if any(k in va for k in blocked):
        return "abstract-only"
    full = ("read in full", "full text read", "read the full text",
            "full pdf", "read in its entirety", "all 10 pages",
            "arxiv v1 full", "complete verbatim text")
    if any(k in va for k in full):
        return "full-text"
    return "partial"


def _input_type(cell: str) -> str:
    """Bucket the free-text input_type cell into the audit's four classes."""
    low = (cell or "").lower()
    has_fc = any(k in low for k in ("operational forecast", "archived forecast",
                                    "forecast", "nwp"))
    has_obs = any(k in low for k in ("realised", "realized", "observation",
                                     "actual", "reanalysis"))
    if low.startswith("mixed") or (has_fc and has_obs):
        return "mixed"
    if has_fc:
        return "forecast"
    if has_obs:
        return "realised/reanalysis"
    return "unclear"


def _dedupe_key(row: dict) -> str:
    """Identity of the WORK, not of the record.

    The collection pass found several papers twice: once as a preprint or
    working paper and once as the version of record, and once as two entries
    for one DOI that differed only in which open-access copy the collector had
    found. Left in, each duplicate inflates the denominator of every reported
    rate -- a 45 that is really 43 makes "4%" out of what is nearly 5%. The key
    is the DOI where one exists, else the arXiv identifier, else a normalised
    title, and the row kept is the one whose coding rests on the most text.
    """
    doi = (row["paper"].get("doi_or_url") or "").strip().lower()
    m = re.search(r"10\.\d{4,9}/\S+", doi)
    if m:
        return "doi:" + m.group(0).rstrip("/.")
    oa = (row["paper"].get("oa_pdf_url") or "") + " " + doi
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", oa)
    if m:
        return "arxiv:" + m.group(1)
    title = re.sub(r"[^a-z0-9]", "", (row["paper"].get("title") or "").lower())
    return "title:" + title[:60]


_ACCESS_RANK = {"full-text": 3, "partial": 2, "abstract-only": 1, "unrecorded": 0}


def _dupe_rank(row: dict) -> tuple:
    """Which record of a duplicated work to keep.

    Ordered, and stated so the choice cannot drift: how much text the coding
    rests on first, then how many timing questions it actually answers, then
    the later publication year as the version of record. Ranking on access
    alone was not enough -- when two records tie, the tie has to break the
    same way every run or the reported counts move without anything being
    re-measured.
    """
    cells = row.get("columns", {})
    answered = sum(1 for c in VERDICT_COLUMNS
                   if _verdict(cells.get(c, "")) not in ("Not verifiable", "unparsed"))
    return (_ACCESS_RANK[_access_level(row)], answered,
            int(row["paper"].get("year") or 0))


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse duplicate works, keeping the best-evidenced record of each."""
    best: dict[str, dict] = {}
    dropped: list[dict] = []
    for row in rows:
        k = _dedupe_key(row)
        cur = best.get(k)
        if cur is None:
            best[k] = row
            continue
        keep, drop = ((row, cur) if _dupe_rank(row) > _dupe_rank(cur)
                      else (cur, row))
        best[k] = keep
        dropped.append({"key": k,
                        "dropped_title": drop["paper"].get("title", ""),
                        "kept_title": keep["paper"].get("title", ""),
                        "reason": "same work, second record"})
    return list(best.values()), dropped


def _status(row: dict) -> str:
    """included / excluded / unresolved.

    A row is UNRESOLVED when the coder judged it in scope but could not read
    enough published text to answer a single audit question. Counting such a
    paper as "included" would inflate the audit's coverage without adding
    evidence, and counting it as "excluded" would hide a paper we simply could
    not check. It is reported as its own category and kept in the dataset.
    """
    if not row.get("include"):
        return "excluded"
    cells = row.get("columns", {})
    verdicts = [_verdict(cells.get(c, "")) for c in VERDICT_COLUMNS]
    if verdicts and all(v == "Not verifiable" for v in verdicts):
        return "unresolved"
    return "included"


def assemble(rows: list[dict]) -> dict:
    included = [r for r in rows if _status(r) == "included"]
    excluded = [r for r in rows if _status(r) == "excluded"]
    unresolved = [r for r in rows if _status(r) == "unresolved"]
    per_column: dict[str, dict[str, int]] = {}
    for col in VERDICT_COLUMNS:
        counts: dict[str, int] = {}
        for r in included:
            v = _verdict(r.get("columns", {}).get(col, ""))
            counts[v] = counts.get(v, 0) + 1
        per_column[col] = counts

    access = {}
    for r in included:
        a = _access_level(r)
        access[a] = access.get(a, 0) + 1

    inputs = {}
    for r in included:
        b = _input_type(r.get("columns", {}).get("input_type", ""))
        inputs[b] = inputs.get(b, 0) + 1

    # Headline rates are computed over papers whose text we could actually
    # check: "Not verifiable" leaves the denominator rather than counting as a
    # failure of the paper.
    def rate(col: str, positive: tuple[str, ...]) -> dict:
        c = per_column[col]
        checkable = sum(n for v, n in c.items() if v != "Not verifiable")
        pos = sum(c.get(v, 0) for v in positive)
        return {"n_checkable": checkable, "n_positive": pos,
                "share": round(pos / checkable, 4) if checkable else None,
                "n_not_verifiable": c.get("Not verifiable", 0)}

    summary = {
        "n_screened": len(rows),
        "n_included": len(included),
        "n_excluded": len(excluded),
        "n_unresolved": len(unresolved),
        "unresolved_papers": [r["paper"].get("title", "") for r in unresolved],
        "excluded_papers": [{"title": r["paper"].get("title", ""),
                             "reason": r.get("exclude_reason", "")}
                            for r in excluded],
        "access_levels": access,
        "input_type_buckets": inputs,
        "per_column_verdicts": per_column,
        "rates": {
            "states_gate_time": rate("gate_time_stated", ("Yes",)),
            "identifies_model_run": rate("model_run_id_stated", ("Yes",)),
            "states_initialisation": rate("init_time_stated", ("Yes",)),
            "states_publication_time": rate("publication_time_stated", ("Yes",)),
            "checks_gate_admissibility": rate("gate_admissibility_checked", ("Yes",)),
            "handles_revisions": rate("revision_handling", ("Yes",)),
        },
    }
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (["index", "title", "authors_short", "year", "venue", "doi_or_url",
               "included", "exclude_reason", "access_level", "verified_against",
               *COLUMNS, "input_type_bucket", "evidence_quotes"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["paper"].get("year", 0),
                                             x["paper"].get("title", ""))):
            p, c = r["paper"], r.get("columns", {})
            w.writerow({
                "index": r.get("index"),
                "title": p.get("title", ""),
                "authors_short": p.get("authors_short", ""),
                "year": p.get("year", ""),
                "venue": p.get("venue", ""),
                "doi_or_url": p.get("doi_or_url", ""),
                "included": {"included": "yes", "excluded": "no",
                             "unresolved": "unresolved"}[_status(r)],
                "exclude_reason": r.get("exclude_reason", ""),
                "access_level": _access_level(r),
                "verified_against": r.get("verified_against", ""),
                **{col: c.get(col, "") for col in COLUMNS},
                "input_type_bucket": _input_type(c.get("input_type", "")),
                "evidence_quotes": " || ".join(r.get("evidence_quotes", []) or []),
            })
    log.info("wrote %s (%d rows)", path, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True,
                    help="JSON array of per-paper coding rows")
    ap.add_argument("--csv", type=Path, default=OUT_CSV)
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    raw = json.loads(a.rows.read_text(encoding="utf-8"))
    rows, dropped = dedupe(raw)
    log.info("loaded %d coded rows; %d duplicate record(s) collapsed -> %d works",
             len(raw), len(dropped), len(rows))
    for d in dropped:
        log.info("  duplicate: %r kept, %r dropped", d["kept_title"][:60],
                 d["dropped_title"][:60])
    write_csv(rows, a.csv)
    summary = assemble(rows)
    summary["deduplication"] = {"n_records_collected": len(raw),
                                "n_distinct_works": len(rows),
                                "dropped": dropped}
    summary["metadata"] = {
        "phase": 88,
        "protocol": "docs/research/2026-08-11_literature_audit_protocol.md",
        "script": "70_lit_audit_assemble.py",
        "dataset": str(a.csv.relative_to(ROOT)),
        "coding_method": ("one agent-assisted coder per paper reading the "
                          "published text, each cell accompanied by verbatim "
                          "evidence quotes and an explicit record of what was "
                          "read; rates are computed over papers whose text "
                          "could be checked"),
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    log.info("wrote %s", a.json)
    print(json.dumps(summary["rates"], indent=1))
    print(json.dumps({"access": summary["access_levels"],
                      "inputs": summary["input_type_buckets"],
                      "n_included": summary["n_included"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
