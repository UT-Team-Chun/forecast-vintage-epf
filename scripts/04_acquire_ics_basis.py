r"""Phase 71: acquire the imbalance-price CALCULATION BASIS series (ICS WebAPI).

WHY THIS EXISTS
---------------
The fourth external review's central objection was that the paper valued avoided
portfolio deviation at the settlement imbalance price K and called the result a
system cost. We retracted the claim (see phase65) because K is a settlement
transfer, not a marginal balancing cost, and because we had no measure of the
system's actual imbalance.

Both of those objects turn out to be published, per 30-minute slot, for the whole
panel, by the imbalance-price central calculation system (ICS) that the Kansai,
Chubu and Kyushu transmission operators run jointly:

    調整力の限界的なkWh価格 (円/kWh)   the MARGINAL kWh PRICE of balancing resources
    インバランス量 (kWh)               the imbalance VOLUME being balanced
    系統余剰・不足                     whether the system is long or short
    補正料金算定インデックス (%)        the tightness index (= OCCTO 広域予備率)

WHAT THESE QUANTITIES ARE, PRECISELY -- THIS DECIDES THE ANALYSIS
-----------------------------------------------------------------
They are **wide-area operation block** (広域運用ブロック) quantities, NOT area
quantities. The file has nine area columns per group, but every area sharing a
block carries an identical value. Verified on 2024-08 before writing this script:
7,372 same-block area pairs, **zero** differing. About four blocks span the nine
areas in a typical slot, and an area sits alone in its own block in roughly a
quarter of area-slots.

That is a feature rather than a defect, and it is why this series answers the
review. The block is the region inside which imbalance is netted and balanced
jointly, so a block-level imbalance volume is exactly the NETTED quantity the
reviewer said our per-portfolio gross deviation was not. Any claim built on it
must be stated at block level, with the area-alone subset reported separately as
the case where the block quantity is area-specific.

WHAT IS STILL NOT AVAILABLE (do not paper over these)
-----------------------------------------------------
  * ΔkWh activation prices and activated volumes per product per area. The
    balancing exchange (EPRX) publishes ΔkW PROCUREMENT only, in 円/kW per 30 min;
    a grep for "kWh" over its result files returns nothing. The merit-order list
    that would give the full ΔkWh ladder is published as PNG images.
  * The predecessor ΔkWh dispatch series (EGC) ends at March 2022.
  * Per-balancing-group imbalance volume: not published by anyone.
  * OCCTO 広域予備率 direct download reaches only April of the previous fiscal
    year, so it cannot cross-validate the tightness index before 2025-04.
  * JEPX intraday results are system-wide only -- no per-area price or volume --
    so the intraday weight w_ID stays unidentified at area level.

SOURCE AND TERMS
----------------
WebAPI documented at https://www.imbalanceprices-cs.jp/show/footer/webapi_manual.pdf
(section 3.2). The published usage rule asks that retries be limited to five at
30-second intervals; this script issues ONE request per month, sequentially, with
a polite pause, and caches so a re-run costs nothing.

CLI (from the repository root):
  uv run python scripts/04_acquire_ics_basis.py
  ... --months 202408,202409     # partial
  ... --verify-only              # re-check cached files against the registry
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase71")

CFG = EpfConfig()
RAW = CFG.resolved("raw_data_dir") / "jepx_imbalance_basis"
RUN = CFG.resolved("run_dir")
OUT_PARQUET = RUN / "phase71_ics_basis_per_slot.parquet"
OUT_JSON = RUN / "phase71_ics_basis.json"
REGISTRY = CFG.resolved("raw_data_dir") / "registry.yaml"

API = "https://www.imbalanceprices-cs.jp/api/1.0/imb/basis/{ym}"
UA = "Mozilla/5.0 (compatible; academic research; contact ryoutaokauchi@g.ecc.u-tokyo.ac.jp)"
POLITE_SLEEP_S = 2.0
TIMEOUT_S = 120

# The nine JEPX areas, in the file's column order. Okinawa is absent from this
# file, which happens to match our panel exactly.
ICS_AREAS = (
    "hokkaido",
    "tohoku",
    "tokyo",
    "chubu",
    "hokuriku",
    "kansai",
    "chugoku",
    "shikoku",
    "kyushu",
)
ICS_AREA_JA = ("北海道", "東北", "東京", "中部", "北陸", "関西", "中国", "四国", "九州")

# Group label -> output column. The label is matched as a SUBSTRING of the group
# header cell, because the headers carry parenthetical unit text that has changed
# format before. Each group is nine areas x (value, 変更S revision counter).
GROUPS: tuple[tuple[str, str], ...] = (
    ("インバランス量", "imbalance_kwh"),
    ("系統余剰・不足", "system_surplus_flag"),
    ("広域運用分断エリア区分", "block_id"),
    ("補正料金算定インデックス", "tightness_index_pct"),
    ("調整力の限界的なkWh価格", "marginal_balancing_yen_kwh"),
    ("卸市場価格P", "wholesale_ref_yen_kwh"),
    ("再エネ出力抑制の有無", "vre_curtail_flag"),
    ("再エネ出力抑制時単価の適用有無", "vre_curtail_price_applied"),
)
GROUP_WIDTH = 18  # 9 areas x (value + 変更S)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _months(start: str, end: str) -> list[str]:
    out, y, m = [], int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    while (y, m) <= (ey, em):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_month(ym: str, *, force: bool = False) -> tuple[Path, dict]:
    """Download one month's basis CSV, or reuse the cache. Returns (path, meta)."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"{ym}_imbalance_basis.csv"
    meta_path = RAW / f"{ym}_imbalance_basis.meta.json"
    if dest.exists() and meta_path.exists() and not force:
        return dest, json.loads(meta_path.read_text())
    url = API.format(ym=ym)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT_S)
    r.raise_for_status()
    body = r.content
    if len(body) < 10_000:
        msg = f"{ym}: response only {len(body)} bytes -- not a month of data"
        raise SystemExit(msg)
    dest.write_bytes(body)
    meta = {
        "url": url,
        "sha256": _sha256_bytes(body),
        "bytes": len(body),
        "content_disposition": r.headers.get("Content-Disposition", ""),
        "content_type": r.headers.get("Content-Type", ""),
        "fetched_utc": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("fetched %s (%.2f MB) %s", ym, len(body) / 1e6, meta["content_disposition"])
    time.sleep(POLITE_SLEEP_S)
    return dest, meta


def parse_month(path: Path) -> pl.DataFrame:
    """Parse one basis CSV into a tidy (area, datetime_jst, ...) frame.

    Column positions are resolved from the file's OWN group-header row rather
    than hardcoded: the spec's field list has been revised before, and a
    fixed-offset parser would silently read the neighbouring group.
    """
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="cp932"))))
    hdr = next((r for r in rows if r and r[0] == "H" and len(r) > 100), None)
    if hdr is None:
        msg = f"{path.name}: no wide H group-header row found"
        raise SystemExit(msg)
    starts: dict[str, int] = {}
    for label, out_col in GROUPS:
        idx = next((i for i, cell in enumerate(hdr) if label in cell), None)
        if idx is None:
            log.warning("%s: group %r absent -- column will be null", path.name, label)
            continue
        starts[out_col] = idx
    recs: list[dict] = []
    for r in rows:
        if not r or r[0] != "D" or len(r) < 5:
            continue
        day, slot_to = r[1], r[4]
        # 時間帯_至 runs to "24:00"; the slot's START is what we key on, and
        # slot 01 starts at 00:00, so derive from the slot index instead of
        # parsing a 24:00 that no clock has.
        slot = int(r[2])
        minutes = (slot - 1) * 30
        ts = f"{day[:4]}-{day[4:6]}-{day[6:8]} {minutes // 60:02d}:{minutes % 60:02d}:00"
        for k, area in enumerate(ICS_AREAS):
            rec: dict = {"area": area, "datetime_jst": ts, "slot": slot, "slot_end_label": slot_to}
            for out_col, c0 in starts.items():
                cell = r[c0 + 2 * k] if c0 + 2 * k < len(r) else ""
                rec[out_col] = cell.strip()
                rev = r[c0 + 2 * k + 1] if c0 + 2 * k + 1 < len(r) else ""
                if out_col == "imbalance_kwh":
                    rec["revision_s"] = rev.strip()
            recs.append(rec)
    df = pl.DataFrame(recs)
    num = (
        "imbalance_kwh",
        "tightness_index_pct",
        "marginal_balancing_yen_kwh",
        "wholesale_ref_yen_kwh",
    )
    return df.with_columns(
        pl.col("datetime_jst")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.replace_time_zone("Asia/Tokyo"),
        *[
            pl.col(c)
            .replace({"": None, "-": None, "***": None})
            .cast(pl.Float64, strict=False)
            .alias(c)
            for c in num
            if c in df.columns
        ],
    )


def block_replication_audit(df: pl.DataFrame) -> dict:
    """Quantify what the verifier found: these are BLOCK quantities.

    Recorded in provenance so no downstream analysis can quietly treat the
    imbalance volume or the marginal price as area-specific. An analysis built on
    the wrong reading of this file would answer the reviewer with a number that
    is not what it claims to be.
    """
    sub = df.drop_nulls(["block_id", "imbalance_kwh"])
    g = sub.group_by("datetime_jst", "block_id").agg(
        pl.col("imbalance_kwh").n_unique().alias("n_distinct"),
        pl.len().alias("n_areas"),
        pl.col("marginal_balancing_yen_kwh").n_unique().alias("n_price"),
    )
    multi = g.filter(pl.col("n_areas") > 1)
    alone = g.filter(pl.col("n_areas") == 1)
    return {
        "n_block_slots": int(g.height),
        "n_multi_area_block_slots": int(multi.height),
        "n_single_area_block_slots": int(alone.height),
        "multi_area_blocks_with_differing_volume": int(
            multi.filter(pl.col("n_distinct") > 1).height
        ),
        "multi_area_blocks_with_differing_marginal_price": int(
            multi.filter(pl.col("n_price") > 1).height
        ),
        "mean_areas_per_block": round(float(g["n_areas"].mean()), 3),
        "area_alone_share": round(float(alone.height) / max(1, g.height), 4),
        "reading": (
            "Zero differing values inside any multi-area block confirms "
            "these are wide-area operation BLOCK quantities replicated "
            "into each member area's column. The block is the region "
            "inside which imbalance is netted and balanced, so this is "
            "the netted系统 quantity -- but no claim built on it may be "
            "stated per area except on the single-area-block subset."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="202408")
    ap.add_argument("--end", default="202607")
    ap.add_argument("--months", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    months = args.months.split(",") if args.months else _months(args.start, args.end)
    log.info("acquiring %d months: %s .. %s", len(months), months[0], months[-1])

    metas: dict[str, dict] = {}
    frames: list[pl.DataFrame] = []
    for ym in months:
        path, meta = fetch_month(ym, force=args.force and not args.verify_only)
        if args.verify_only:
            got = _sha256_bytes(path.read_bytes())
            ok = got == meta["sha256"]
            log.info("  %s sha256 %s", ym, "OK" if ok else f"MISMATCH {got}")
            if not ok:
                return 1
        metas[ym] = meta
        frames.append(parse_month(path))
    if args.verify_only:
        log.info("all %d cached files match the registry", len(months))
        return 0

    df = pl.concat(frames).sort("area", "datetime_jst")
    df.write_parquet(OUT_PARQUET)
    audit = block_replication_audit(df)
    cov = {
        "n_rows": int(df.height),
        "n_areas": int(df["area"].n_unique()),
        "first_slot": str(df["datetime_jst"].min()),
        "last_slot": str(df["datetime_jst"].max()),
        "null_share": {
            c: round(float(df[c].null_count()) / df.height, 4)
            for c in (
                "imbalance_kwh",
                "marginal_balancing_yen_kwh",
                "tightness_index_pct",
                "wholesale_ref_yen_kwh",
            )
            if c in df.columns
        },
        "marginal_price_range": [
            round(float(df["marginal_balancing_yen_kwh"].min()), 3),
            round(float(df["marginal_balancing_yen_kwh"].max()), 3),
        ]
        if "marginal_balancing_yen_kwh" in df.columns
        else None,
    }
    out = {
        "phase": 71,
        "title": "Imbalance-price calculation basis (ICS WebAPI): marginal "
        "balancing energy price and block imbalance volume",
        "source": {
            "api": API,
            "manual": "https://www.imbalanceprices-cs.jp/show/footer/webapi_manual.pdf",
            "operator": "インバランス単価中央算定システム (ICS), operated jointly "
            "by Kansai T&D, Chubu PG and Kyushu T&D",
        },
        "months": metas,
        "coverage": cov,
        "block_replication_audit": audit,
        "not_available": [
            "dKWh activation prices and activated volumes per product per area: "
            "EPRX publishes dKW procurement only (yen/kW per 30 min); the "
            "merit-order list is published as PNG images",
            "the predecessor dKWh dispatch series (EGC) ends at March 2022",
            "per-balancing-group imbalance volume: not published",
            "OCCTO wide-area reserve margin before April of the previous fiscal "
            "year, so the tightness index cannot be cross-validated pre-2025-04",
            "JEPX intraday results are system-wide only: no per-area price or "
            "volume, so the intraday weight stays unidentified at area level",
        ],
        "per_slot_parquet": OUT_PARQUET.name,
        "metadata": {"script": Path(__file__).name, "runtime_s": round(time.time() - t0, 1)},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("wrote %s (%d rows) and %s", OUT_PARQUET, df.height, OUT_JSON)
    log.info("coverage: %s .. %s, %d areas", cov["first_slot"], cov["last_slot"], cov["n_areas"])
    log.info(
        "block audit: %d multi-area block-slots, %d with differing volume "
        "(0 expected), area alone in %.1f%% of block-slots",
        audit["n_multi_area_block_slots"],
        audit["multi_area_blocks_with_differing_volume"],
        100 * audit["area_alone_share"],
    )
    log.info("marginal balancing price range: %s JPY/kWh", cov["marginal_price_range"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
