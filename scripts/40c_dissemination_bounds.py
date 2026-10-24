r"""Phase 115: the two-sided bracket on when the D-1 00 UTC MSM run can reach a bidder.

WHY THIS EXISTS
---------------
All three Applied Energy referees (R1.1, R3.1, R5.2) made the same objection, and
R5 called it a major identification problem: archive write times are UPPER bounds
on dissemination, so they show a pre-gate run was available but cannot show the
D-1 00 UTC run was not. Excluding that run strictly needs a LOWER bound on its
dissemination lag above the 60 minutes between its initialisation and the gate.
The paper had said "neither source supplies one".

JMA's own technical documentation does supply one, of a different kind from a
delivery schedule: the meso-scale analysis cannot start before its observation
cut-off, and the forecast integration cannot start before the analysis. This
script does not reason about those documents; it LOCATES each quoted statement in
the downloaded file, page by page, and records the page, the matched text, the
URL and the file's SHA-256, so every number the manuscript takes from them is
checkable. A quote that is not found fails the run.

It also summarises the MEASURED side over everything the project has fetched:
the phase72 per-file capture (2024-26) and, when present, the phase112 backcast
captures (2020-24).

Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md (D1,
G112d).

CLI (from the repository root):
  uv run python scripts/40c_dissemination_bounds.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase115")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir")
RUN = CFG.resolved("run_dir")
DOCS = RAW / "jma_docs"
OUT_JSON = RUN / "phase115_dissemination_bounds.json"
PHASE72_PER_FILE = RUN / "phase72_dissemination_audit_per_file.parquet"
BACKCAST_LASTMOD = (RAW / "jma_gpv_backcast" / "_last_modified.jsonl",
                    RAW / "jma_gpv_backcast_early" / "_last_modified.jsonl")
RETRIEVED = "2026-10-23"

# file -> (url, bib key, [(label, regex that must match on one page)])
SOURCES: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "outline2022_02.pdf": (
        "https://www.jma.go.jp/jma/jma-eng/jma-center/nwp/outline2022-nwp/pdf/outline2022_02.pdf",
        "jma2022outline",
        [("ma_cutoff_table", r"Data cut-off time\s+(\d+) minutes for analysis at 00, 03, 06, 09, "
                             r"12, 15, 18, and 21 UTC"),
         ("ma_cutoff_text", r"The cut-off time of input observation data for MA is 50 minutes "
                            r"after each analysis time")]),
    "outline2022_01.pdf": (
        "https://www.jma.go.jp/jma/jma-eng/jma-center/nwp/outline2022-nwp/pdf/outline2022_01.pdf",
        "jma2022outline",
        [("suite_schedule_caption", r"Figure 1\.3\.1: Daily schedule of the operational suite "
                                    r"running on the main system")]),
    "nwpkaisetu_R7_1_7_4.pdf": (
        "https://www.jma.go.jp/jma/kishou/books/nwpkaisetu/latest/1_7_4.pdf",
        "jma2025nwphandbook",
        [("ma_cutoff_ja", r"観測打ち切り時刻\s*50分"),
         ("ma_wait_ja", r"データ入電待ち時間が解析時刻後50分")]),
    "nwpkaisetu_R7_1_7_5.pdf": (
        "https://www.jma.go.jp/jma/kishou/books/nwpkaisetu/latest/1_7_5.pdf",
        "jma2025nwphandbook",
        [("msm_range_78", r"(\d+)時間予報\s*\(00,\s*12\s*UTC\)"),
         ("msm_range_39", r"(\d+)時間予報\s*\(03,\s*06,\s*09,\s*15,\s*18,\s*21\s*UTC\)")]),
    **{f"wmo_{y}_japan.pdf": (
        f"https://www.jma.go.jp/jma/jma-eng/jma-center/nwp/report/{y}_Japan.pdf",
        "jmawmo" + str(y),
        [(f"msm00_schedule_{y}", r"Meso-scale\s+00\s+(0055)\s+–\s+(0[12]\d\d)\s+(\d+)")])
       for y in (2016, 2017, 2018, 2019, 2020)},
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pages(pdf: Path) -> list[str]:
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                         capture_output=True, text=True).stdout
    # NFKC folds the "ﬀ" ligature pdftotext emits for "cut-off"
    return [unicodedata.normalize("NFKC", p) for p in txt.split("\f")]



def _hhmm_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[2:])

def documents() -> dict:
    out: dict = {}
    for fname, (url, bib, quotes) in SOURCES.items():
        path = DOCS / fname
        if not path.exists():
            msg = f"missing source document {path}; fetch it from {url}"
            raise FileNotFoundError(msg)
        pages = _pages(path)
        rec = {"url": url, "bib": bib, "retrieved": RETRIEVED, "sha256": _sha(path),
               "n_pages": len(pages), "quotes": {}}
        for label, pat in quotes:
            hit = None
            for i, page in enumerate(pages, start=1):
                m = re.search(pat, page)
                if m:
                    hit = {"pdf_page": i, "matched": re.sub(r"\s+", " ", m.group(0)),
                           "groups": list(m.groups())}
                    break
            if hit is None:
                msg = f"{fname}: quote '{label}' not found -- the document changed or the pattern is wrong"
                raise ValueError(msg)
            rec["quotes"][label] = hit
        out[fname] = rec
    sched = {}
    for y in (2016, 2017, 2018, 2019, 2020):
        g = out[f"wmo_{y}_japan.pdf"]["quotes"][f"msm00_schedule_{y}"]["groups"]
        sched[str(y)] = {"start_utc": g[0], "end_utc": g[1], "forecast_range_h": int(g[2])}
    q = out
    start = sched["2020"]["start_utc"]
    end = sched["2020"]["end_utc"]
    start_min = int(start[:2]) * 60 + int(start[2:])
    end_min = int(end[:2]) * 60 + int(end[2:])
    # Every value below is PARSED from a matched quote, never typed: the guard binds
    # the manuscript's macros to these fields.
    documented = {
        "ma_cutoff_min": int(q["outline2022_02.pdf"]["quotes"]["ma_cutoff_table"]["groups"][0]),
        "job_start_utc_min": start_min, "job_start_jst_min": start_min + 9 * 60,
        "job_end_utc_min": end_min, "job_duration_min": end_min - start_min,
        "job_schedule_year": 2020,
        "range_long_h": int(q["nwpkaisetu_R7_1_7_5.pdf"]["quotes"]["msm_range_78"]["groups"][0]),
        "range_short_h": int(q["nwpkaisetu_R7_1_7_5.pdf"]["quotes"]["msm_range_39"]["groups"][0]),
        "job_start_same_every_year": len({v["start_utc"] for v in sched.values()}) == 1,
        # The 2016-2019 reports schedule the job to 02:05 UTC, the 2020 report to
        # 02:00: quoting only 2020's 65 minutes understated the earlier years.
        "job_duration_min_by_year": {y: _hhmm_min(v["end_utc"]) - _hhmm_min(v["start_utc"])
                                     for y, v in sched.items()},
        "job_duration_min_lo": min(_hhmm_min(v["end_utc"]) - _hhmm_min(v["start_utc"])
                                   for v in sched.values()),
        "job_duration_min_hi": max(_hhmm_min(v["end_utc"]) - _hhmm_min(v["start_utc"])
                                   for v in sched.values())}
    return {"sources": out, "msm00_run_schedule_by_year": sched, "documented": documented}


def _gate_utc(delivery: datetime) -> datetime:
    """Gate for delivery day D: D-1 10:00 JST = D-1 01:00 UTC."""
    return datetime(delivery.year, delivery.month, delivery.day, 1, tzinfo=UTC) - timedelta(days=1)


def _as_utc(x) -> datetime:
    """phase72 stores write times as tz-aware datetimes or ISO strings."""
    if isinstance(x, str):
        x = datetime.fromisoformat(x.replace("Z", "+00:00"))
    return x.astimezone(UTC) if x.tzinfo else x.replace(tzinfo=UTC)


def _pre_gate_12z(rows12: list[dict]) -> dict | None:
    """The D-2 12 UTC run over the backcast captures, and with the 2024-2026 audit.

    Section 3.3 says every file of this run was written before the gate on every
    audited day, at least so many hours ahead, over both audits. The 2024-2026 half
    is phase72; this is the 2020-2024 half and the worst margin across the two.
    """
    if not rows12:
        return None
    b = pl.DataFrame(rows12)
    per_day = b.group_by("delivery_day").agg((pl.col("min_after_gate") < 0).all().alias("ok"))
    out = {"files": b.height, "days": b["delivery_day"].n_unique(),
           "span": [b["delivery_day"].min(), b["delivery_day"].max()],
           "files_at_or_after_gate": int((b["min_after_gate"] >= 0).sum()),
           "days_all_files_before_gate": int(per_day["ok"].sum()),
           "worst_margin_min": round(float(-b["min_after_gate"].max()), 2)}
    if PHASE72_PER_FILE.exists():
        p72 = (pl.read_parquet(PHASE72_PER_FILE)
               .filter((pl.col("run") == "msm_d2_12z") & pl.col("last_modified_utc").is_not_null()))
        m72 = [(_gate_utc(datetime.fromisoformat(str(r["delivery_day"])[:10]))
                - _as_utc(r["last_modified_utc"])).total_seconds() / 60
               for r in p72.iter_rows(named=True)]
        days72 = {str(d)[:10] for d in p72["delivery_day"].to_list()}
        out["combined_2020_2026"] = {
            "days": len(days72 | set(b["delivery_day"].to_list())),
            "worst_margin_min": round(min(min(m72), out["worst_margin_min"]), 2)}
    return out


def measured() -> dict:
    """Write-time statistics for the D-1 00 UTC run over every capture the project holds."""
    rows: list[dict] = []
    rows12: list[dict] = []
    if PHASE72_PER_FILE.exists():
        p72 = pl.read_parquet(PHASE72_PER_FILE)
        log.info("phase72 columns: %s", p72.columns)
    for path in BACKCAST_LASTMOD:
        if not path.exists():
            continue
        seen = set()
        for line in path.read_text().splitlines():
            r = json.loads(line)
            key = (r["plan"], r["run_utc"], r["fh"])
            if (key in seen or r["plan"] not in ("msm_d1_00z", "msm_d2_12z")
                    or not r["last_modified"]):
                continue
            seen.add(key)
            d = datetime.fromisoformat(r["delivery_day"])
            written = parsedate_to_datetime(r["last_modified"]).astimezone(UTC)
            (rows if r["plan"] == "msm_d1_00z" else rows12).append(
                {"source": path.parent.name, "delivery_day": r["delivery_day"],
                 "fh": r["fh"],
                 "min_after_gate": (written - _gate_utc(d)).total_seconds() / 60,
                 "lag_min": (written - datetime.fromisoformat(r["run_utc"])
                             ).total_seconds() / 60})
    if not rows:
        return {"backcast_files": 0}
    df = pl.DataFrame(rows)
    # The manuscript states the 00 UTC run's arrival over 2020-2026, so the 2024-2026
    # audit (phase72, one header per file) and the backcast captures are combined
    # here, file by file, rather than quoted side by side.
    combined = None
    if PHASE72_PER_FILE.exists():
        p72 = (pl.read_parquet(PHASE72_PER_FILE)
               .filter((pl.col("run") == "msm_d1_00z") & pl.col("last_modified_utc").is_not_null()))
        p72_rows = [{"delivery_day": str(r["delivery_day"])[:10], "fh": r["fh"],
                     "min_after_gate": (_as_utc(r["last_modified_utc"])
                                        - _gate_utc(datetime.fromisoformat(str(r["delivery_day"])[:10]))
                                        ).total_seconds() / 60}
                    for r in p72.iter_rows(named=True)]
        allf = pl.concat([df.select("delivery_day", "fh", "min_after_gate"),
                          pl.DataFrame(p72_rows).select("delivery_day", "fh", "min_after_gate")])
        combined = {"files": allf.height,
                    "days": allf["delivery_day"].n_unique(),
                    "span": [allf["delivery_day"].min(), allf["delivery_day"].max()],
                    "files_before_gate": int((allf["min_after_gate"] <= 0).sum()),
                    "days_no_file_before_gate": int(
                        allf.group_by("delivery_day")
                        .agg((pl.col("min_after_gate") > 0).all().alias("ok"))["ok"].sum()),
                    "min_after_gate_min": round(float(allf["min_after_gate"].min()), 2),
                    "max_after_gate_min": round(float(allf["min_after_gate"].max()), 2)}
    return {"combined_2020_2026": combined,
            "pre_gate_12z": _pre_gate_12z(rows12),
            "backcast_files": df.height,
            "backcast_days": df["delivery_day"].n_unique(),
            "backcast_span": [df["delivery_day"].min(), df["delivery_day"].max()],
            "files_before_gate": int((df["min_after_gate"] <= 0).sum()),
            "min_after_gate_min": round(float(df["min_after_gate"].min()), 2),
            "max_after_gate_min": round(float(df["min_after_gate"].max()), 2),
            "median_lag_min": round(float(df["lag_min"].median()), 2),
            # polars yields group keys as tuples, which json cannot key on
            "by_fh": {fh: {"min_after_gate": round(float(g["min_after_gate"].min()), 2),
                           "median_after_gate": round(float(g["min_after_gate"].median()), 2)}
                      for (fh,), g in df.group_by("fh")}}


def main() -> int:
    out = {"phase": "115", "doc": "docs/research/2026-10-23_phase115_dissemination_bounds.md",
           **documents(), "measured_backcast": measured(),
           "bracket": {
               "gate_utc_for_delivery_D": "D-1 01:00 UTC (10:00 JST)",
               "documented_lower_bound": (
                   "observation cut-off at initialisation + 50 min (00:50 UTC, 09:50 JST); "
                   "the analysis-forecast job is scheduled to begin at 00:55 UTC "
                   "(09:55 JST) and the forecast integration follows the analysis"),
               "measured_upper_bound": "mirror write time of every file of the run",
               "kind": "the lower bound is documentary, not measured; the upper bound is "
                       "measured, not documentary"}}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
    log.info("wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
