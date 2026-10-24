r"""Phase 117: does "initialised before the gate, disseminated after it" happen elsewhere?

WHY THIS EXISTS
---------------
Applied Energy referee R5: the paper "does not establish that its main empirical
conclusion ... extends to markets with different gate times, forecast dissemination
schedules, etc." The effect sizes cannot be transferred without re-estimation, but
the TIMING problem the protocol addresses can be checked anywhere from public
documents: for each market's day-ahead gate and each widely used global model, is
there a run initialised before the gate whose forecast of the delivery day is not
yet disseminated at it?
Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md (D2).

SOURCES (all under data/raw/market_gates/, retrieved 2026-10-23; every input below
carries its URL and the verbatim text it rests on)
  * gates: each exchange's or ISO's rulebook;
  * ECMWF: the official dissemination schedule (transfer start per forecast step,
    real-time users); this is the primary column;
  * NOAA GFS: no official dissemination schedule exists, so availability is the
    NOMADS listing time (median over the 10 saved days) of the first needed file and
    the NCO production-status 30-day mean completion of the block holding the last
    needed step. It is reported as a secondary column with that caveat.
DWD ICON is omitted: no published schedule could be found.

CLI (from the repository root):
  uv run python scripts/49_market_gates.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

from alg.config import EpfConfig

CFG = EpfConfig()
SRC = CFG.resolved("raw_data_dir") / "market_gates"
OUT_JSON = CFG.resolved("run_dir") / "phase117_market_gates.json"
TABLE = CFG.project_root / "docs/paper/paper_a_main/tables/gate_nwp_markets.tex"

# name -> (label, gate "HH:MM" local on D-1, UTC offset standard, UTC offset summer,
#          delivery-day start relative to local midnight of D (h), source, quote)
MARKETS: dict[str, tuple] = {
    "sdac": ("EU day-ahead coupling (SDAC)", "12:00", 1, 2, 0,
             "https://nemo-committee.eu/assets/files/sdac-fallback-manual-v12-and-v13.pdf",
             "All times in this document are expressed in CET/CEST; Closure of orderbook 12:00"),
    "gb": ("Great Britain (EPEX hourly)", "09:20", 0, 1, -1,
           "EPEX SPOT Operational Rules (Market Rules zip, 03.11.2026)",
           "Day-Ahead GB 60min Auction ... Closure of the Order Book Daily at 09:20 (GMT/BST)"),
    "pjm": ("PJM", "11:00", -5, -4, 0, "https://www.pjm.com/-/media/DotCom/documents/manuals/m11.pdf",
            "all bids must be received by 11:00 (EPT)"),
    "miso": ("MISO", "10:30", -5, -4, 0, "MISO BPM-002 r26",
             "OD-1 @ 1030 EPT Close Day-Ahead Energy and Operating Reserve Market"),
    "ercot": ("ERCOT", "10:00", -6, -5, 0, "ERCOT Nodal Protocols Section 4 (1 Aug 2026)",
              "A QSE must submit any DAM Energy-Only Offer Curves by 1000"),
    "caiso": ("CAISO", "10:00", -8, -7, 0,
              "https://www.caiso.com/documents/monthly-extended-day-ahead-market-edam-performance-report-for-jun-2026.pdf",
              "All bids submission is due by 10:00 a.m. Pacific Time"),
    "nyiso": ("NYISO", "05:00", -5, -4, 0, "NYISO Day-Ahead Scheduling Manual v10.1",
              "Finalized Day-Ahead bids must be submitted by 05:00 a.m."),
    "isone": ("ISO New England", "10:30", -5, -4, 0, "ISO-NE Market Rule 1 III.1.10.1A",
              "may submit offers and bids in the Day-Ahead Market until 10:30 a.m. on the day before"),
    "iex": ("India (IEX day-ahead)", "11:00", 5.5, 5.5, 0, "IEX Business Rules Ver. 24",
            "Auction timings 10:00 AM to 11:00 AM of previous day (D-1)"),
    "jepx": ("Japan (JEPX)", "10:00", 9, 9, 0, "https://www.jepx.jp/electricpower/outline/pdf/tr_rules.pdf",
             "受け渡しの日の１日前の日の午前10時まで"),
}
ECMWF_SOURCE = ("https://confluence.ecmwf.int/display/DAC/Dissemination+schedule",
                "oper fc 0 to 90 by 1 | 05:45 -> 06:12 | 11:45 -> 12:12 | 17:45 -> 18:12 | "
                "23:45 -> 00:12 ... step 0 starts transferring at 05:45 and step 90 at 06:12, "
                "with the steps in between going out in order at roughly regular intervals")
# NCO production status, 30-day mean END times of the GFS product blocks (UTC)
PRODSTAT = {
    0: {"12": "03:47:53", "24": "03:51:05", "36": "03:54:09", "48": "03:57:28", "72": "04:03:40"},
    6: {"12": "09:41:33", "24": "09:44:44", "36": "09:47:53", "48": "09:51:03", "72": "09:57:10"},
    12: {"12": "15:42:31", "24": "15:45:48", "36": "15:48:52", "48": "15:51:58", "72": "15:58:24"},
    18: {"12": "21:44:52", "24": "21:48:07", "36": "21:51:14", "48": "21:54:29", "72": "22:00:41"},
}
PRODSTAT_SOURCE = "https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat/ (as of 23 Oct 2026 05:02 GMT)"


def _hms(s: str) -> float:
    h, m, x = map(int, s.split(":"))
    return h * 60 + m + x / 60


def _ecmwf(step: int) -> float:
    """Minutes after initialisation at which a step starts transferring (steps <= 90)."""
    return 5 * 60 + 45 + step * 27 / 90


def _gfs_last(step: int, cyc: int) -> float:
    blk = next(b for b in ("12", "24", "36", "48", "72") if step <= int(b))
    t = _hms(PRODSTAT[cyc][blk]) - cyc * 60
    return t + 24 * 60 if t < 0 else t


def _nomads() -> dict[tuple[int, int], float]:
    lags: dict[tuple[int, int], list[float]] = {}
    for f in sorted((SRC / "ncep" / "nomads").glob("*.html")):
        d, c = f.stem.split("_")
        init = dt.datetime.strptime(d + c, "%Y%m%d%H")
        for stp, t in re.findall(r'href="gfs\.t\d\dz\.pgrb2\.0p25\.f(\d{3})".*?(\d{2}-\w{3}-\d{4} \d\d:\d\d)',
                                 f.read_text(errors="replace")):
            lag = (dt.datetime.strptime(t, "%d-%b-%Y %H:%M") - init).total_seconds() / 60
            lags.setdefault((int(c), int(stp)), []).append(lag)
    return {k: st.median(v) for k, v in lags.items()}


def _season(name: str, summer: bool) -> dict:
    _, hhmm, ow, osu, start_rel, _, _ = MARKETS[name]
    off = osu if summer else ow
    h, mi = map(int, hhmm.split(":"))
    gate = (h + mi / 60 - off) * 60                 # minutes after 00:00 UTC of D-1
    d0 = (24 + start_rel - off) * 60                # delivery start, same origin
    return {"gate": gate, "d0": d0, "d1": d0 + 24 * 60}


def _cases(name: str, summer: bool, nomads: dict) -> dict:
    s = _season(name, summer)
    out: dict = {"gate_utc_min": s["gate"], "ecmwf": [], "gfs": []}
    for dayoff in (-1, 0):
        for cyc in (0, 6, 12, 18):
            init = dayoff * 1440 + cyc * 60
            if init >= s["gate"]:
                continue
            first = math.floor((s["d0"] - init) / 60)
            last = math.ceil((s["d1"] - init) / 60)
            run = f"{'D-2' if dayoff < 0 else 'D-1'} {cyc:02d} UTC"
            if last <= 90:
                f_, l_ = init + _ecmwf(first), init + _ecmwf(last)
                out["ecmwf"].append({"run": run, "steps": [first, last],
                                     "late_first_min": round(f_ - s["gate"], 1),
                                     "late_last_min": round(l_ - s["gate"], 1)})
            if last <= 72 and (cyc, first) in nomads:
                f_ = init + nomads[(cyc, first)]
                l_ = init + _gfs_last(last, cyc)
                out["gfs"].append({"run": run, "steps": [first, last],
                                   "late_first_min": round(f_ - s["gate"], 1),
                                   "late_last_min": round(l_ - s["gate"], 1)})
    return out


def _summarise(cases: list[dict]) -> dict | None:
    """The runs initialised before the gate whose first needed step arrives after it."""
    ibda = [c for c in cases if c["late_first_min"] > 0]
    if not ibda:
        return None
    best = min(ibda, key=lambda c: c["late_first_min"])
    return {"run": best["run"], "late_min": best["late_first_min"],
            "late_max": best["late_last_min"]}


def main() -> int:
    nomads = _nomads()
    rows: dict = {}
    for name in MARKETS:
        rec = {"label": MARKETS[name][0], "gate_local": MARKETS[name][1],
               "source": MARKETS[name][5], "quote": MARKETS[name][6], "seasons": {}}
        seasons = (False,) if MARKETS[name][2] == MARKETS[name][3] else (False, True)
        for summer in seasons:
            c = _cases(name, summer, nomads)
            rec["seasons"]["summer" if summer else "standard"] = {
                "gate_utc_min": c["gate_utc_min"], "ecmwf_cases": c["ecmwf"],
                "gfs_cases": c["gfs"], "ecmwf_ibda": _summarise(c["ecmwf"]),
                "gfs_ibda": _summarise(c["gfs"])}
        rows[name] = rec
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    out = {"phase": "117", "retrieved": "2026-10-23", "ecmwf_source": ECMWF_SOURCE,
           "gfs_source": [PRODSTAT_SOURCE, "NOMADS listings, 10 days, data/raw/market_gates/ncep/nomads"],
           "markets": rows,
           "n_markets": len(rows),
           "n_ecmwf_any_season": sum(any(s["ecmwf_ibda"] for s in r["seasons"].values())
                                     for r in rows.values()),
           "n_gfs_any_season": sum(any(s["gfs_ibda"] for s in r["seasons"].values())
                                   for r in rows.values()),
           "source_files_sha256": {str(p.relative_to(SRC)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files}}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if TABLE.parent.exists():  # the tables live with the manuscript
        write_table(rows)
        write_source_table(rows)
    print(f"markets {out['n_markets']}; ECMWF case in {out['n_ecmwf_any_season']}; "
          f"GFS case in {out['n_gfs_any_season']}")
    return 0


def _cell(r: dict, key: str) -> str:
    parts = []
    for season, tag in (("standard", ""), ("summer", "")):
        s = r["seasons"].get(season)
        if s is None:
            continue
        v = s[key]
        parts.append(None if v is None else (v["run"].split(" ", 1)[1], round(v["late_min"])))
    vals = [p for p in parts if p is not None]
    if not vals:
        return "none"
    runs = {p[0] for p in vals}
    if len(parts) == 2 and None in parts:
        run, late = vals[0]
        return f"{run}, summer only ($\\geq${late}~min)"
    return f"{'/'.join(sorted(runs))} ($\\geq${min(p[1] for p in vals)}~min)"


def _tex_escape(x: str) -> str:
    return (x.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
             .replace("_", "\\_").replace("#", "\\#"))


def write_source_table(rows: dict) -> None:
    body = []
    for r in rows.values():
        src = r["source"]
        src = f"\\url{{{src}}}" if src.startswith("http") else _tex_escape(src)
        quote = _tex_escape(r["quote"])
        if r["label"].startswith("Japan"):
            quote = "bids close at 10:00 on the day before delivery (Article 16)"
        body.append(f"{r['label']} & {src} & ``{quote}'' \\\\")
    lines = [
        "% GENERATED by 49_market_gates.py -- do not edit by hand.",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Sources of the day-ahead gate times in Table~3 of the main article, "
        "each read on 2026-10-23, with the text the gate time rests on. ECMWF times: "
        "the dissemination schedule at \\url{" + ECMWF_SOURCE[0] + "} (``step 0 starts "
        "transferring at 05:45 and step 90 at 06:12'' for the 00\\,UTC run). GFS times: "
        "NCO production status (30-day means) and ten days of NOMADS listings.}",
        "\\label{tab:supp_market_sources}", "\\scriptsize",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.2\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.36\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.36\\linewidth}}",
        "\\toprule", "Market & Rule document & Text \\\\", "\\midrule", *body,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TABLE.parent / "phase117_market_sources.tex").write_text("\n".join(lines), encoding="utf-8")


def write_table(rows: dict) -> None:
    body = []
    for r in rows.values():
        utc = []
        for season in ("standard", "summer"):
            s = r["seasons"].get(season)
            if s is not None:
                m = s["gate_utc_min"] % (24 * 60)
                utc.append(f"{int(m // 60):02d}:{int(round(m % 60)):02d}")
        body.append(f"{r['label']} & {r['gate_local']} & {' / '.join(utc)} & "
                    f"{_cell(r, 'ecmwf_ibda')} & {_cell(r, 'gfs_ibda')} \\\\")
    lines = [
        "% GENERATED by 49_market_gates.py -- do not edit by hand.",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Day-ahead gates and, for each global model, the latest run initialised "
        "before the gate whose forecast of the delivery day is scheduled (ECMWF) or was "
        "observed (GFS) to start arriving after it, with the least margin by which its "
        "first needed forecast step misses the gate. Gate times are local on the day before "
        "delivery, then UTC in standard / summer time. ECMWF times follow its published "
        "schedule for the Integrated Forecasting System (IFS); its early-delivery service, "
        "about 20 minutes earlier, would close the smallest margin (India). GFS has no "
        "published schedule, so its times are observed archive arrivals, and margins of a "
        "few minutes (MISO, ISO New England) can change from day to day. Sources: "
        "Supplementary Section~S4.}",
        "\\label{tab:markets}", "\\small",
        "\\begin{tabular}{llllll}".replace("llllll", "lllll"), "\\toprule",
        "Market & Gate & Gate (UTC) & ECMWF (IFS) & NOAA GFS \\\\", "\\midrule", *body,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
