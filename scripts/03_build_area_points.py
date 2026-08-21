r"""Phase 74 infrastructure: build the area -> weighted prefecture-point table.

Parses the METI FIT-portal capacity workbooks (sheet 表A②-1, operating PV
capacity in kW by prefecture) into ``data/reference/area_points_capacity.csv``:
one row per (vintage, area, prefecture) with the prefectural-capital
coordinates and the prefecture's PV-capacity share WITHIN its area. Shares are
within-area, so any level disagreement between the FIT ledger and
area_capacity.csv cancels.

Sources (https://www.fit-portal.go.jp/servlet/servlet.FileDownload?file=<ID>):
  2024-09  015IT00000AoHNV   primary: static weights aligned with panel start
  2025-12  015J2000000Xtry   sensitivity vintage
The downloaded workbooks are stored at data/raw/meti_fit/ and their SHA-256
hashes are written into the CSV header comment.

Split prefectures are assigned whole to their PRIMARY TSO (pre-committed in
docs/research/2026-08-03_phase74_cw_precommit.md): Shizuoka -> Chubu,
Mie -> Chubu, Fukui -> Hokuriku, Niigata -> Tohoku. Okinawa is not a JEPX
bidding zone and is excluded.

CLI (from the repository root):
  uv run --with openpyxl python scripts/03_build_area_points.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import openpyxl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase74_points")

CFG = EpfConfig()
RAW_DIR = CFG.project_root / "data/raw/meti_fit"
OUT_CSV = CFG.project_root / "data/reference/area_points_capacity.csv"

VINTAGES = {
    "2024-09": "meti_fit_a_tables_2024-09_015IT00000AoHNV.xlsx",
    "2025-12": "meti_fit_a_tables_2025-12_015J2000000Xtry.xlsx",
}
SHEET = "表A②－１"  # operating capacity (kW), by prefecture

# prefecture -> (JEPX area, capital lat, capital lon)
PREF: dict[str, tuple[str, float, float]] = {
    "北海道": ("hokkaido", 43.064, 141.347),
    "青森県": ("tohoku", 40.824, 140.740), "岩手県": ("tohoku", 39.704, 141.153),
    "宮城県": ("tohoku", 38.269, 140.872), "秋田県": ("tohoku", 39.719, 140.102),
    "山形県": ("tohoku", 38.240, 140.363), "福島県": ("tohoku", 37.750, 140.468),
    "新潟県": ("tohoku", 37.902, 139.023),
    "茨城県": ("tokyo", 36.342, 140.447), "栃木県": ("tokyo", 36.566, 139.884),
    "群馬県": ("tokyo", 36.391, 139.060), "埼玉県": ("tokyo", 35.857, 139.649),
    "千葉県": ("tokyo", 35.605, 140.123), "東京都": ("tokyo", 35.690, 139.692),
    "神奈川県": ("tokyo", 35.448, 139.643), "山梨県": ("tokyo", 35.664, 138.568),
    "長野県": ("chubu", 36.651, 138.181), "岐阜県": ("chubu", 35.391, 136.722),
    "静岡県": ("chubu", 34.977, 138.383), "愛知県": ("chubu", 35.180, 136.907),
    "三重県": ("chubu", 34.730, 136.509),
    "富山県": ("hokuriku", 36.695, 137.211), "石川県": ("hokuriku", 36.595, 136.626),
    "福井県": ("hokuriku", 36.065, 136.222),
    "滋賀県": ("kansai", 35.005, 135.869), "京都府": ("kansai", 35.021, 135.756),
    "大阪府": ("kansai", 34.686, 135.520), "兵庫県": ("kansai", 34.691, 135.183),
    "奈良県": ("kansai", 34.685, 135.833), "和歌山県": ("kansai", 34.226, 135.168),
    "鳥取県": ("chugoku", 35.504, 134.238), "島根県": ("chugoku", 35.472, 133.051),
    "岡山県": ("chugoku", 34.662, 133.934), "広島県": ("chugoku", 34.397, 132.460),
    "山口県": ("chugoku", 34.186, 131.471),
    "徳島県": ("shikoku", 34.066, 134.559), "香川県": ("shikoku", 34.340, 134.043),
    "愛媛県": ("shikoku", 33.842, 132.766), "高知県": ("shikoku", 33.560, 133.531),
    "福岡県": ("kyushu", 33.607, 130.418), "佐賀県": ("kyushu", 33.249, 130.300),
    "長崎県": ("kyushu", 32.745, 129.874), "熊本県": ("kyushu", 32.790, 130.742),
    "大分県": ("kyushu", 33.238, 131.613), "宮崎県": ("kyushu", 31.911, 131.424),
    "鹿児島県": ("kyushu", 31.560, 130.558),
    # 沖縄県 is not a JEPX bidding zone: excluded on read below.
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pv_kw_by_pref(xlsx: Path) -> dict[str, float]:
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb[SHEET]
    out: dict[str, float] = {}
    for row in ws.iter_rows(values_only=True):
        name = row[0]
        if not isinstance(name, str):
            continue
        name = name.strip()
        if name == "沖縄県":
            continue
        if name in PREF:
            # column B: PV < 10 kW; column D: PV >= 10 kW (both operating, kW)
            lt10, ge10 = row[1], row[3]
            if not isinstance(lt10, (int, float)) or not isinstance(ge10, (int, float)):
                msg = f"{xlsx.name}: non-numeric capacity for {name}: {lt10!r}, {ge10!r}"
                raise ValueError(msg)
            out[name] = float(lt10) + float(ge10)
    wb.close()
    missing = set(PREF) - set(out)
    if missing:
        msg = f"{xlsx.name}: prefectures missing from {SHEET}: {sorted(missing)}"
        raise ValueError(msg)
    return out


def main() -> int:
    lines = [
        # Layout-neutral stamp: this script ships under two filenames (the
        # monorepo's sandbox convention and the replication package's numbered
        # name), and a filename in the header meant regeneration in one
        # repository could never byte-reproduce the file shipped by the other.
        "# area_points_capacity.csv -- generated by the area-points build "
        "script; do not edit by hand.",
        "# Prefecture PV operating capacity (sheet 表A②-1, columns <10kW + >=10kW, kW)",
        "# from the METI FIT portal; weight = share within the JEPX area.",
        "# Split prefectures assigned whole to the primary TSO; Okinawa excluded.",
    ]
    for vintage, fname in VINTAGES.items():
        src = RAW_DIR / fname
        lines.append(f"# source {vintage}: {fname} sha256={_sha256(src)}")
    lines.append("vintage,area,prefecture,lat,lon,pv_kw,weight_in_area")

    for vintage, fname in VINTAGES.items():
        kw = _pv_kw_by_pref(RAW_DIR / fname)
        area_tot: dict[str, float] = {}
        for pref, v in kw.items():
            area_tot[PREF[pref][0]] = area_tot.get(PREF[pref][0], 0.0) + v
        for pref, (area, lat, lon) in PREF.items():  # stable declaration order
            w = kw[pref] / area_tot[area]
            lines.append(f"{vintage},{area},{pref},{lat},{lon},"
                         f"{kw[pref]:.1f},{w:.6f}")
        log.info("%s: total PV %.1f GW over %d prefectures, %d areas",
                 vintage, sum(kw.values()) / 1e6, len(kw), len(area_tot))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows)", OUT_CSV, len(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
