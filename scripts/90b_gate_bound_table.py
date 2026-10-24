r"""Phase 116: one table for the post-gate increment under every specification.

WHY THIS EXISTS
---------------
Applied Energy referees asked, separately, whether the near-gate result survives
a different equivalence margin or spatial weighting (R1.5), scarcity and extreme
weather (R1.6, R5.3), stronger or tuned learners (R1.8, R3.2), the model upgrade
and short early training (R1.9), and the wind measurement (R1.11). Each answer is
the same pair of numbers -- the pre-gate run's improvement and the post-gate
run's increment over it -- under one change. Putting every specification in one
table, computed by one function on one row set, is what lets a reader compare
them; a scatter of numbers across sections would not.

Each row is recomputed from per-slot predictions with phase85's own statistics
(imported). Rows whose inputs do not exist yet are written as pending, so the
table can be rebuilt as phases finish. The main window's rows are scored on
phase85's capacity-weighted row set (all cw_ok flags true); the backcast rows on
their own panel.

CLI (from the repository root):
  uv run python scripts/90b_gate_bound_table.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase116")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P85 = Path(__file__).parent / "10_primary_ladder_cw.py"
N_AREAS = 9
TABLE = CFG.project_root / "docs/paper/paper_a_main/tables/phase116_gate_bound.tex"
OUT_JSON = RUN / "phase116_gate_bound.json"
KEY = ["area", "datetime_jst"]
UPGRADE = date(2025, 5, 20)
LONG_TRAIN_FROM = date(2025, 3, 16)
BASE, PRE, POST = "price_only", "msm_d2_12z_cw", "msm_d1_00z_cw"


def _load_p85():
    spec = importlib.util.spec_from_file_location("p85", P85)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p85"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _ckpt(name: str) -> pl.DataFrame | None:
    """All nine areas of a checkpoint set, or None.

    A partial set must never reach the table: joined on the reference keys it
    yields a row over the finished areas only, with the full day count, which
    reads as a complete result. That happened once (the network row printed a
    two-area figure while the run was still going), so completeness is checked
    here rather than at each call site.
    """
    files = sorted((RUN / name).glob("*.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    if df["area"].n_unique() != N_AREAS:
        log.info("%s: %d of %d areas, row left pending", name, df["area"].n_unique(), N_AREAS)
        return None
    return df


def _block(p85, df: pl.DataFrame, base: str, pre: str, post: str) -> dict:
    cols = [f"y_pred_{c}" for c in (base, pre, post)]
    df = df.drop_nulls(cols).filter(pl.all_horizontal(pl.col(c).is_not_nan() for c in cols))
    per_day = (df.with_columns(*[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs()
                                 .alias(f"ae_{c}") for c in (base, pre, post)])
               .group_by("area", "test_day")
               .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in (base, pre, post)])
               .sort("area", "test_day"))

    def pooled(cell: str) -> np.ndarray:
        return np.asarray((per_day.with_columns(
            (pl.col(f"mae_{base}") - pl.col(f"mae_{cell}")).alias("imp"))
            .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day")["imp"]),
            dtype=float)

    d_pre, d_post = pooled(pre), pooled(post)
    base_mae = float(per_day[f"mae_{base}"].mean())
    pre_e = p85._entry(d_pre)
    gate = p85._diff_entry(d_post, d_pre)
    return {"n_days": len(d_pre), "n_slots": df.height, "price_only_mae": round(base_mae, 4),
            "pre": pre_e["improvement"], "pre_ci95": pre_e["ci95"],
            "gate": gate["delta"], "gate_ci95": gate["ci95"],
            "gate_p_l3": gate["dm_l3"]["p_hln"],
            "ucb_rel_pct": round(100 * gate["ci95"][1] / base_mae, 3),
            "ucb_share_pre_pct": (round(100 * gate["ci95"][1] / pre_e["improvement"], 2)
                                  if pre_e["ci95"][0] > 0 else None)}


def _from_json_block(b: dict) -> dict:
    """Reformat a block written by phase112/114 (same statistics, same function)."""
    return {"n_days": b["n_days"], "n_slots": b["n_slots"],
            "price_only_mae": b["price_only_mae"],
            "pre": b["pre_gate_improvement"]["improvement"],
            "pre_ci95": b["pre_gate_improvement"]["ci95"],
            "gate": b["gate_contrast"]["delta"], "gate_ci95": b["gate_contrast"]["ci95"],
            "gate_p_l3": b["gate_contrast"]["dm_l3"]["p_hln"],
            "ucb_rel_pct": round(100 * b["gate_contrast"]["ci95"][1] / b["price_only_mae"], 3),
            "ucb_share_pre_pct": (round(100 * b["gate_contrast"]["ci95"][1]
                                        / b["pre_gate_improvement"]["improvement"], 2)
                                  if b["pre_gate_improvement"]["ci95"][0] > 0 else None)}


def build_rows(p85) -> list[dict]:  # noqa: C901, PLR0912, PLR0915 - one branch per row family
    ref_all = pl.read_parquet(RUN / "phase85_exact_ladder_per_slot.parquet")
    flags = [c for c in ref_all.columns if c.startswith("cw_ok_")]
    ref = ref_all.filter(pl.all_horizontal(pl.col(f) for f in flags))
    keys = ref.select(KEY)
    rows: list[dict] = []

    def add(group: str, key: str, label: str, block: dict | None, note: str = "") -> None:
        rows.append({"group": group, "key": key, "label": label, "note": note,
                     "pending": block is None, **(block or {})})

    g = "Main window, 2024-08 to 2026-08"
    add(g, "primary", "Primary specification", _block(p85, ref, BASE, PRE, POST))
    # Added after the internal referee read (phase118): the pooled bound includes
    # night slots, on which a PV-driven increment cannot appear.
    add(g, "daylight", "Daylight slots, 09:00--15:30\\textsuperscript{\\dag}",
        _block(p85, ref.filter(pl.col("datetime_jst").dt.hour().is_between(9, 15)),
               BASE, PRE, POST))

    g = "Learner"
    t = _ckpt("phase113_tuned_checkpoints")
    add(g, "tuned", "XGBoost, re-tuned every 91 days",
        None if t is None else _block(p85, keys.join(t, on=KEY), BASE, PRE, POST))
    lp, l96 = _ckpt("phase113_lear_post_checkpoints"), _ckpt("phase96_lear_checkpoints")
    lear = None
    if lp is not None and l96 is not None:
        lj = keys.join(lp.select(*KEY, "test_day", "y_actual", f"y_pred_{POST}"), on=KEY).join(
            l96.select(*KEY, f"y_pred_{BASE}", f"y_pred_{PRE}"), on=KEY)
        lear = _block(p85, lj, BASE, PRE, POST)
    add(g, "lear", "LEAR (linear benchmark)", lear)
    dn = _ckpt("phase113_dnn_checkpoints")
    add(g, "dnn", "Feed-forward neural network",
        None if dn is None else _block(p85, keys.join(dn, on=KEY), BASE, PRE, POST))

    g = "Measurement"
    add(g, "single_point", "Single representative point",
        _block(p85, ref, BASE, "msm_d2_12z", "msm_d1_00z"))
    for arm, label in (("2025-12", "Capacity weights, 2025-12 register"),
                       ("equal", "Equal weights across prefectures")):
        files = sorted((RUN / "phase91_checkpoints").glob(f"{arm}__*.parquet"))
        blk = None
        if len(files) == 9:
            sub = pl.concat([pl.read_parquet(f) for f in files])
            sub = sub.drop([c for c in sub.columns if c.startswith("cw_ok_")])
            blk = _block(p85, keys.join(sub, on=KEY), BASE, PRE, POST)
        add(g, f"weights_{arm}", label, blk)
    rw = _ckpt("phase113_repro_wind_checkpoints")
    nw = None
    if rw is not None and rw["area"].n_unique() == 9:
        nwj = keys.join(rw.select(*KEY, "test_day", "y_actual", f"y_pred_{PRE}_nowind",
                                  f"y_pred_{POST}_nowind"), on=KEY).join(
            ref.select(*KEY, f"y_pred_{BASE}"), on=KEY)
        nw = _block(p85, nwj, BASE, f"{PRE}_nowind", f"{POST}_nowind")
    add(g, "no_wind", "Without the wind channel", nw)

    g = "Sample"
    add(g, "pre_upgrade", "Test days before the 2025-05-20 model upgrade",
        _block(p85, ref.filter(pl.col("test_day") < UPGRADE.isoformat()), BASE, PRE, POST))
    add(g, "post_upgrade", "Test days after the upgrade",
        _block(p85, ref.filter(pl.col("test_day") >= UPGRADE.isoformat()), BASE, PRE, POST))
    era = _ckpt("phase113_era_checkpoints")
    eb = None
    if era is not None and era["area"].n_unique() == 9:
        eb = _block(p85, keys.join(era.filter(pl.col("test_day") >= "2025-07-20"), on=KEY),
                    BASE, PRE, POST)
    add(g, "post_upgrade_trained", "Trained on post-upgrade data only", eb)
    add(g, "long_training", "Folds with at least a year of training",
        _block(p85, ref.filter(pl.col("test_day") >= LONG_TRAIN_FROM.isoformat()),
               BASE, PRE, POST))

    g = "Weather regime, main window"
    p114 = RUN / "phase114_stored_prediction_checks.json"
    cuts = json.loads(p114.read_text())["regimes"]["cuts"] if p114.exists() else {}
    for k, label in (("hot_days", "Hottest days (area top 5\\%)"),
                     ("cold_days", "Coldest days (area bottom 5\\%)"),
                     ("low_wind_days", "Calmest days (area bottom 10\\%)"),
                     ("evening_ramp", "Evening ramp, 16:00--20:30"),
                     ("high_price_slots", "Highest-priced slots (top 5\\%, outcome-selected)")):
        add(g, f"regime_{k}", label, _from_json_block(cuts[k]) if k in cuts else None)

    g = "Backcast, 2021-01 to 2024-07"
    p112 = RUN / "phase112_backcast.json"
    bc = json.loads(p112.read_text())["cw"] if p112.exists() else {}
    for k, label in (("all", "All test days"),
                     ("episodes", "Scarcity episodes"),
                     ("S50", "Slots at or above 50 JPY/kWh (outcome-selected)")):
        add(g, f"backcast_{k}", label, _from_json_block(bc[k]) if k in bc else None)
    return rows


def _fmt(x: float, nd: int = 3) -> str:
    """Round half up from the stored decimal, as the macro guard and the prose do.

    Float formatting turns a stored 0.2385 into 0.238 (the binary value sits just
    below the half) while the text prints 0.239 from the same provenance.
    """
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{q:.{nd}f}"
    return s.replace("-", "$-$") if q < 0 else s


def write_table(rows: list[dict]) -> None:
    lines = [
        "% GENERATED by 90b_gate_bound_table.py -- do not edit by hand.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{The pre-gate run's improvement over the price-only model and the "
        "post-gate run's increment over the pre-gate run, under each specification, with "
        "95\\% block-bootstrap intervals over test days. The upper bound (UCB) is the "
        "upper end of the increment's interval, as a percentage of the price-only MAE and "
        "of the row's own pre-gate gain (a dash where that gain is not distinguishable "
        "from zero). Learner and measurement rows use the primary slots; daylight, sample "
        "and regime rows are subsets of them (one test day has no daylight slot), and the "
        "backcast is a separate window. Regime cuts rank area-days by mean "
        "capacity-weighted temperature or wind. \\textsuperscript{\\dag}Specified after "
        "related estimates had been seen (Section~\\ref{sec:inference}). Units: JPY/kWh.}",
        "\\label{tab:bound}",
        "\\small",
        "\\begin{tabular}{lrccrr}",
        "\\toprule",
        "Specification & Days & Pre-gate improvement & Post-gate increment & \\multicolumn{2}{c}{UCB (\\%)} \\\\",
        " & & & & MAE & gain \\\\",
        "\\midrule"]
    group = None
    for r in rows:
        if r["group"] != group:
            if group is not None:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{r['group']}}}}} \\\\")
            group = r["group"]
        if r["pending"]:
            lines.append(f"\\quad {r['label']} & \\multicolumn{{5}}{{c}}{{[pending]}} \\\\")
            continue
        pre = f"{_fmt(r['pre'])} [{_fmt(r['pre_ci95'][0])}, {_fmt(r['pre_ci95'][1])}]"
        gate = f"{_fmt(r['gate'])} [{_fmt(r['gate_ci95'][0])}, {_fmt(r['gate_ci95'][1])}]"
        days = f"{r['n_days']:,}".replace(",", "{,}")
        share = ("--" if r.get("ucb_share_pre_pct") is None
                 else _fmt(r["ucb_share_pre_pct"], 1))
        lines.append(f"\\quad {r['label']} & {days} & {pre} & {gate} & "
                     f"{_fmt(r['ucb_rel_pct'], 2)} & {share} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", TABLE)


# ------------------------------------------------------------------ supplement tables
SUPP_DIR = CFG.project_root / "docs/paper/paper_a_main/tables"
DOC_LABELS = {
    "ma_cutoff_table": ("JMA NWP Outline 2022, ch.~2", "Table 2.1.2"),
    "ma_cutoff_text": ("JMA NWP Outline 2022, ch.~2", "Section 2.6.2"),
    "suite_schedule_caption": ("JMA NWP Outline 2022, ch.~1", "Figure 1.3.1"),
    "ma_cutoff_ja": ("JMA NWP handbook R7", "Section 1.7.4"),
    "ma_wait_ja": ("JMA NWP handbook R7", "Section 1.7.4"),
    "msm_range_78": ("JMA NWP handbook R7", "Section 1.7.5"),
    "msm_range_39": ("JMA NWP handbook R7", "Section 1.7.5"),
}
DOC_GLOSS = {
    "ma_cutoff_table": "observation cut-off 50 min after each analysis time",
    "ma_cutoff_text": "same, in the description of the meso-scale analysis",
    "suite_schedule_caption": "daily schedule of the operational suite (March 2021)",
    "ma_cutoff_ja": "observation cut-off 50 min (in Japanese)",
    "ma_wait_ja": "data waiting time 50 min after analysis time (in Japanese)",
    "msm_range_78": "78-h forecasts from 00 and 12 UTC (in Japanese)",
    "msm_range_39": "39-h forecasts from the other six cycles (in Japanese)",
}


def _tex_table(label: str, caption: str, colspec: str, header: str, body: list[str]) -> str:
    return "\n".join([
        "% GENERATED by 90b_gate_bound_table.py -- do not edit by hand.",
        "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", "\\small", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        header, "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def write_supp_tables(rows: list[dict]) -> None:
    p115 = RUN / "phase115_dissemination_bounds.json"
    if p115.exists():
        d = json.loads(p115.read_text())
        body = []
        for fname, rec in d["sources"].items():
            for key, q in rec["quotes"].items():
                if key.startswith("msm00_schedule_"):
                    year = key.rsplit("_", 1)[1]
                    src, loc = f"JMA WMO progress report {year}", "Table 4.1-1"
                    gloss = (f"00 UTC meso-scale job {q['groups'][0][:2]}:{q['groups'][0][2:]}"
                             f"--{q['groups'][1][:2]}:{q['groups'][1][2:]} UTC, "
                             f"{q['groups'][2]}-h range")
                else:
                    src, loc = DOC_LABELS[key]
                    gloss = DOC_GLOSS[key]
                body.append(f"{src} & {loc} & {q['pdf_page']} & {gloss} & "
                            f"\\texttt{{{rec['sha256'][:12]}}} \\\\")
        (SUPP_DIR / "phase115_documents.tex").write_text(_tex_table(
            "tab:supp_documents",
            "Where each documented statement used in Section~3.3 of the main article sits. "
            "Every statement was located in the downloaded file by script (page = PDF page "
            "of the downloaded file); the hash identifies the file read.",
            "llrlr", "Document & Location & Page & Statement & SHA-256 (first 12) \\\\",
            body), encoding="utf-8")
    learners = [r for r in rows if r["key"] in ("primary", "tuned", "lear", "dnn")]
    body = []
    for r in learners:
        if r["pending"]:
            body.append(f"{r['label']} & \\multicolumn{{5}}{{c}}{{[pending]}} \\\\")
            continue
        body.append(f"{r['label']} & {r['price_only_mae']:.3f} & {_fmt(r['pre'])} "
                    f"[{_fmt(r['pre_ci95'][0])}, {_fmt(r['pre_ci95'][1])}] & {_fmt(r['gate'])} "
                    f"[{_fmt(r['gate_ci95'][0])}, {_fmt(r['gate_ci95'][1])}] & "
                    f"{r['gate_p_l3']:.2f} & {r['ucb_rel_pct']:.2f} \\\\")
    (SUPP_DIR / "phase116_learners.tex").write_text(_tex_table(
        "tab:supp_learners",
        "The two contrasts under each learner, each with its own price-only model, on the "
        "main window's capacity-weighted slots. MAE and contrasts in JPY/kWh; $p$ is the "
        "HLN--Diebold--Mariano $p$ ($L=3$) of the post-gate increment; UCB is its 95\\% upper "
        "bound as a percentage of that learner's price-only MAE.",
        "lrccrr", "Learner & Price-only MAE & Pre-gate improvement & Post-gate increment & "
        "$p$ & UCB (\\%) \\\\", body), encoding="utf-8")
    p112 = RUN / "phase112_backcast.json"
    if p112.exists():
        bc = json.loads(p112.read_text())
        cuts = (("all", "All test days"), ("episodes", "Scarcity episodes"),
                ("E1_jan2021", "\\quad January 2021"),
                ("E2_winter2022", "\\quad January--March 2022"),
                ("E3_summer2022", "\\quad June--September 2022"),
                ("non_episode", "Outside the episodes"),
                ("S30", "Slots at or above 30 JPY/kWh\\textsuperscript{*}"),
                ("S50", "Slots at or above 50 JPY/kWh\\textsuperscript{*}"))
        body = []
        for arm, title in (("cw", "Capacity-weighted field (primary)"),
                           ("sp", "Single representative point")):
            body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
            for k, label in cuts:
                b = bc[arm][k]
                pre, gate = b["pre_gate_improvement"], b["gate_contrast"]
                days = f"{b['n_days']:,}".replace(",", "{,}")
                body.append(
                    f"{label} & {days} & {_fmt(b['price_only_mae'])} & "
                    f"{_fmt(pre['improvement'])} [{_fmt(pre['ci95'][0])}, {_fmt(pre['ci95'][1])}] & "
                    f"{_fmt(gate['delta'])} [{_fmt(gate['ci95'][0])}, {_fmt(gate['ci95'][1])}] & "
                    f"{_fmt(gate['dm_l3']['p_hln'], 2)} \\\\")
            if arm == "cw":
                body.append("\\addlinespace")
        (SUPP_DIR / "phase112_backcast_cuts.tex").write_text(_tex_table(
            "tab:supp_backcast",
            "The two contrasts over the 2021--2024 backcast, by cut, under both measurements. "
            "Days are test days with at least one slot in the cut; MAE and contrasts in JPY/kWh, "
            "with 95\\% block-bootstrap intervals; $p$ is the HLN--Diebold--Mariano $p$ ($L=3$) "
            "of the post-gate increment. The episodes were defined in advance from the calendar; "
            "\\textsuperscript{*}the price cuts select slots on the realised price and are "
            "descriptive only.",
            "lrrccr", "Cut & Days & Price-only MAE & Pre-gate improvement & Post-gate increment & "
            "$p$ \\\\", body), encoding="utf-8")


def main() -> int:
    # The provenance is computed wherever the per-slot predictions exist; only the
    # tables need the manuscript tree. The public companion ships neither the
    # per-slot predictions (they derive from raw data) nor the manuscript, so there
    # the frozen phase116_gate_bound.json is the provenance -- but a reader who
    # rebuilds the predictions from raw data can regenerate it with this script.
    per_slot = RUN / "phase85_exact_ladder_per_slot.parquet"
    if not per_slot.exists():
        log.info("no per-slot predictions here (%s absent); the shipped %s is the "
                 "provenance", per_slot, OUT_JSON.name)
        return 0
    p85 = _load_p85()
    rows = build_rows(p85)
    done = [r for r in rows if not r["pending"]]
    out = {"phase": "116", "git_sha": _git_sha(), "rows": {r["key"]: r for r in rows},
           "n_rows": len(rows), "n_pending": len(rows) - len(done),
           "max_ucb_rel_pct": max(r["ucb_rel_pct"] for r in done),
           "max_ucb_rel_pct_main_window": max(
               r["ucb_rel_pct"] for r in done
               if not r["key"].startswith(("regime_", "backcast_")))}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    if TABLE.parent.parent.exists():  # the tables live with the manuscript
        write_table(rows)
        write_supp_tables(rows)
    else:
        log.info("no manuscript tree here (%s absent); wrote the provenance only",
                 TABLE.parent.parent)
    log.info("rows %d, pending %d", len(rows), out["n_pending"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
