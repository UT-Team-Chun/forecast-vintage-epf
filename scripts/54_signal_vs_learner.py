r"""Phase 102 (companion run): is the post-gate null a property of the SIGNAL or
of the LEARNER?

THE QUESTION
------------
The manuscript's RQ2 result is a two-target divergence. The forecast run that the
10:00 gate excludes:

  * improves a forecast of the PHYSICAL delivery day a lot
    (phase89 Gate 51: +59.88 MW [49.64, 70.55], n = 696 common days), and
  * adds nothing detectable to the PRICE
    (phase85 CW gate timing: +0.0026 [-0.0047, 0.0099] JPY/kWh, n = 732 days,
    TOST-equivalent at the one-tick margin).

A referee's first objection is not that the numbers are wrong but that the
inference is: maybe the information IS in the later run and the LEARNER cannot
extract it. phase96 answered the learner-CLASS version of that objection for RQ1
(LEAR reproduces the weather increment's direction at roughly half the size).
This is the matching test for RQ2, and it is a DIFFERENT test, because the ladder
never gave any learner the one quantity that matters here: the DIFFERENCE between
the two runs.

phase85's post-gate cell REPLACES the pre-gate field with the post-gate field. A
gradient-boosted tree makes axis-aligned splits on single columns; it cannot form
the difference of two of them. So "the post-gate cell does not beat the pre-gate
cell" is consistent with two very different worlds and the ladder as built cannot
separate them:

  (i)  the update the later run carries is irrelevant to price formation, or
  (ii) the update is relevant but no tree in this design can see it, because
       seeing it requires subtracting one column from another.

phase90 already showed the update is NOT orthogonal to the pre-gate price
residual (OLS slope -0.388 JPY/kWh per innovation SD, cluster-robust p < 1e-4)
and bounded what a LINEAR use of it could take off the pooled MAE at
0.0036 JPY/kWh -- but that bound is IN SAMPLE by construction, and phase90's own
artefact records that refitting its slope on the full panel turns the bound into
a LOSS. Nobody has yet handed the innovation to the actual learner, inside the
actual walk-forward, and measured OUT OF SAMPLE. That is what this does.

THE TEST
--------
Feed the innovation DIRECTLY as extra columns on top of the pre-gate feature set
and score it with phase85's panel, folds, design matrix, estimator and statistics
VERBATIM. Reuse, not reimplementation: this module imports
10_primary_ladder_cw.py (the ladder) and
s_20260819_phase102_innovation_learner.py (the committed innovation construction,
which does the phase85 `_design` / `_fit` rebinding and asserts it took effect --
phase91 taught this project that a silent module-global swap turns a contrast into
a copy and reports a bug as a null). Both rebindings are asserted again here from
the caller's side, and the fit dispatch is proved by GATE A below.

Two innovation objects, because there are two defensible readings of "the new
information the later run carries", and reporting only one would be a choice made
after seeing the data:

  WX  the weather-field innovation: post-gate minus pre-gate for each of the four
      capacity-weighted meteorological channels, same delivery slot. The literal
      object, defined on every CW-complete slot of the ladder's panel. PRIMARY.
  PV  phase90's innovation, Delta-PV-hat = proxy_scaled(post) - proxy_scaled(pre),
      read from phase89's committed per-slot parquet so it is the same object
      Section 5.3 regresses the price residual on. It is a photovoltaic quantity:
      zero at night by construction, MISSING (NaN, never silently zero) outside
      phase89's daylight window / eight publishing areas. Both counts reported.
      SECONDARY.

PRICE CELLS (every other phase85 cell is still CONSTRUCTED so the panel's row set
cannot move, and is written as NaN so it cannot be mistaken for a prediction):
  msm_d2_12z_cw       phase85's pre-gate headline, REFIT here and bit-checked
                      against phase85's own parquet (GATE A). If this does not
                      reproduce to 1e-9 nothing below is comparable to the paper.
  pre_plus_wx_innov   pre-gate field + the four WX innovation columns  [PRIMARY]
  pre_plus_pv_innov   pre-gate field + Delta-PV-hat                    [SECONDARY]
  pre_plus_noise      pre-gate field + four Gaussian columns matched to the WX
                      innovation columns' per-area SDs. NEGATIVE CONTROL: the
                      cost of four more columns, so a null can be read against
                      the dimensionality cost rather than against zero.
  pre_plus_post_raw   pre-gate field + post-gate field as raw columns. Spans the
                      same information as pre_plus_wx_innov in a different basis,
                      so the gap between them is the REPRESENTATION effect.

BUILD BUDGET, DECLARED. This machine is shared and each fitted cell costs ~732
folds x 9 areas of XGBoost, so the build is split into three passes and which pass
supplied which cell is recorded in the artefact under `build_passes`:

  q  THE QUESTION, on the FULL 732-fold panel: pre_plus_wx_innov and
     pre_plus_pv_innov. The full panel here is not a preference. At n = 732 the
     bootstrap CI half-width on this contrast is about 0.0073 JPY/kWh, i.e. inside
     the one-tick margin, so branches (a) and (b) below are distinguishable; on a
     third of the folds the interval straddles the tick and the test decides
     nothing. Power, not taste, fixes this pass at STEP = 1 day.
  r  the GATE A refit of phase85's own pre-gate headline, on phase85's schedule
     with STEP widened to 5 days (147 of the panel's own test days, each with an
     unchanged training set, so the fits are the SAME fits). The full-panel
     contrast then takes its baseline from phase85's stored predictions, which
     GATE A licenses. A declared cost saving, and the thing it buys is checked
     rather than assumed.
  b  the two CONTROL cells, STEP widened to 3 days. Controls calibrate a null and
     do not need tick-level power; the primary and the reference contrasts are
     recomputed on exactly those days beside them so the block is like for like.

A pass that covered only some areas is reported ABSENT, never partial -- a partial
pass would change the panel, which is the one thing that must not differ.

CLI (from the repository root):
  uv run --extra research python scripts/54_signal_vs_learner.py --step build --pass q --areas kyushu
  uv run --extra research python scripts/54_signal_vs_learner.py --step phys
  uv run --extra research python scripts/54_signal_vs_learner.py --step analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase102sl")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
HERE = Path(__file__).parent
P85_PATH = HERE / "10_primary_ladder_cw.py"
P89_PATH = HERE / "50_physical_target.py"
P102_PATH = HERE / "s_20260819_phase102_innovation_learner.py"

P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P85_JSON = RUN / "phase85_exact_ladder.json"
P89_JSON = RUN / "phase89_physical_target.json"
P89_SLOTS = RUN / "phase89_physical_per_slot.parquet"
P90_JSON = RUN / "phase90_mechanism.json"
IMP_JSON = RUN / "phase102_innovation_importance.json"

PHYS_PARQUET = RUN / "phase102sl_physical_per_slot.parquet"
PHYS_JSON = RUN / "phase102sl_phys_arm.json"
OUT_JSON = RUN / "phase102_signal_vs_learner.json"

PRE_GATE = "msm_d2_12z_cw"
POST_GATE = "msm_d1_00z_cw"
BASELINE = "price_only"
C_WX = "pre_plus_wx_innov"
C_PV = "pre_plus_pv_innov"
C_NOISE = "pre_plus_noise"
C_RAW = "pre_plus_post_raw"

# Build passes. Which cells each carries, and on which fold schedule, is
# declared here and recorded in the artefact.
#   q  THE QUESTION, FULL 732-fold panel: the two innovation cells. Full panel is
#      not a preference, it is what the pre-registered branch logic needs -- at
#      n = 732 phase85's CI half-width on this contrast is 0.0073 JPY/kWh, below
#      the one-tick margin, so branch (a) and branch (b) are distinguishable. On a
#      third of the folds the interval straddles the tick and the test decides
#      nothing.
#   r  GATE A bit-reproduction refit of phase85's pre-gate headline, on phase85's
#      own schedule with STEP widened to 5 days: every retained test day is a real
#      test day of the full run with an unchanged training set, so the fits are the
#      SAME fits and a bit-match on them is a bit-match. The full-panel contrast
#      then uses phase85's stored pre-gate predictions as the baseline, which GATE
#      A licenses. Declared cost saving, not a hidden one.
#   b  the two CONTROL cells on the same schedule with STEP widened to 3 days. The
#      controls calibrate a null; they do not need tick-level power. The primary
#      contrast is recomputed on exactly these days beside them so the comparison
#      is like for like.
PASS_CELLS = {"q": (C_WX, C_PV), "r": (PRE_GATE,), "b": (C_NOISE, C_RAW)}
PASS_THIN = {"q": 1, "r": 5, "b": 3}

# pre-registered constants
TICK = 0.01                # JPY/kWh, phase85's own equivalence margin
BIT_TOL = 1e-9             # GATE A, phase85's GATE_TOL
RECOVER_FRAC = 0.50        # GATE B
DEGENERATE_FRAC = 0.01     # GATE C
P89_GAIN_MW = 59.88        # phase89 Gate 51, common panel
P85_GATE_TIMING = 0.0026   # phase85 CW gate timing delta
N_AREAS = 9


def _ckpt(pass_: str) -> Path:
    return RUN / f"phase102sl_ckpt_{pass_}"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,  # noqa: S607
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _j(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ===================================================================== build
def build(mod85, mod102, areas: list[str], pass_: str) -> int:
    """phase102's innovation construction + phase85's harness, both verbatim."""
    mod102.FIT_CELLS = PASS_CELLS[pass_]
    thin = PASS_THIN[pass_]
    if thin > 1:
        # phase85's own schedule with STEP widened: the retained test days are a
        # subset of the full run's test days and each keeps its training set
        mod85.STEP = timedelta(days=thin)
        if mod85.STEP != timedelta(days=thin):
            msg = "STEP was not widened"
            raise RuntimeError(msg)
    ckpt = _ckpt(pass_)
    orig_design, orig_fit = mod85._design, mod85._fit
    t0 = time.time()
    mod102.build_price(mod85, areas, ckpt)
    # assert, from the caller's side, that the rebinding phase102 performs was
    # still in force when the folds were fitted, and that it was CALLED
    if mod85._design is orig_design or mod85._fit is orig_fit:
        msg = "phase85's _design/_fit were not rebound"
        raise RuntimeError(msg)
    if not mod102._design_calls:
        msg = "the rebound _design was never called; the innovation was not built"
        raise RuntimeError(msg)
    if sorted(mod102._design_calls) != sorted(areas):
        msg = f"design called for {mod102._design_calls} but asked for {areas}"
        raise RuntimeError(msg)
    log.info("pass %s: %s in %.0f s", pass_, ",".join(areas), time.time() - t0)
    return 0


# ============================================== physical positive control
def phys(mod85, mod89, mod102) -> int:
    """phase102's physical arm, into THIS phase's own files."""
    mod102.PHYS_PARQUET = PHYS_PARQUET
    b = mod102.build_phys(mod85, mod89)
    price_days = set(pl.read_parquet(P85_PARQUET, columns=["test_day"])
                     ["test_day"].unique().to_list())
    r = mod102.analyse_phys(mod89, b["panel"], list(mod85.DEFAULT_WEATHER_COLS),
                            price_days)
    r["construction_checks"] = b["checks"]
    PHYS_JSON.write_text(json.dumps(r, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    log.info("physical arm -> %s", PHYS_JSON)
    log.info("%s", json.dumps({k: r[k] for k in ("levels_common_panel",
                                                 "gains_vs_A_common")}, indent=1))
    return 0


# =================================================================== analyse
def _read_pass(pass_: str) -> tuple[object, dict]:
    """Read a build pass. ABSENT unless all nine areas are present and finite.

    A pass that covered only some areas is reported ABSENT, never partial: a
    partial pass would change the panel, which is the one thing that must not
    differ between the cells being compared.
    """
    ckpt = _ckpt(pass_)
    cells = PASS_CELLS[pass_]
    info: dict = {"dir": ckpt.name, "cells": list(cells),
                  "fold_step_days": PASS_THIN[pass_]}
    files = sorted(ckpt.glob("*.parquet")) if ckpt.exists() else []
    info["n_area_files"] = len(files)
    if len(files) < N_AREAS:
        info["status"] = "ABSENT"
        info["reason"] = (f"{len(files)}/{N_AREAS} area files present; reported "
                          "absent rather than partial")
        return None, info
    d = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    for c in cells:
        col = f"y_pred_{c}"
        if col not in d.columns:
            info["status"], info["reason"] = "ABSENT", f"{col} missing"
            return None, info
        n_fin = int(d[col].is_finite().sum())
        info[f"n_finite_{c}"] = n_fin
        if n_fin != d.height:
            info["status"] = "ABSENT"
            info["reason"] = (f"{col} has {d.height - n_fin} non-finite rows, so "
                              "the fit dispatch did not reach this cell")
            return None, info
    info["status"] = "PRESENT"
    info["n_slots"] = d.height
    info["n_areas"] = int(d["area"].n_unique())
    info["n_test_days"] = int(d["test_day"].n_unique())
    return d, info


def _side_nulls(pass_: str, scored: pl.DataFrame, wx_cols: list[str]) -> dict:
    """Count innovation nulls ON THE SCORED ROWS. No NaN becomes a result silently."""
    side = _ckpt(pass_).parent / f"{_ckpt(pass_).name}_side"
    files = sorted(side.glob("*.parquet")) if side.exists() else []
    if not files:
        return {"status": "side artefact absent"}
    s = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    innov = [f"{c}__innov" for c in wx_cols]
    key = ["area", "datetime_jst"]
    j = scored.select(key).join(s, on=key, how="left")
    out: dict = {"n_scored_rows": j.height,
                 "n_scored_rows_absent_from_side_artefact": int(
                     j[innov[0]].null_count())}
    for c, n in zip(wx_cols, innov, strict=True):
        fin = j[n].is_finite() & j[n].is_not_null()
        vals = j.filter(fin)[n]
        field = j.filter(fin)[f"{c}_{PRE_GATE}"]
        rec = {"n_null_or_nan_on_scored_rows": int(j.height - int(fin.sum())),
               "mean_abs": _j(float(vals.abs().mean()) if len(vals) else None, 5),
               "sd": _j(float(vals.std()) if len(vals) > 1 else None, 5),
               "pre_gate_field_sd": _j(float(field.std()) if len(field) > 1
                                       else None, 5)}
        rec["mean_abs_over_field_sd"] = (
            _j(rec["mean_abs"] / rec["pre_gate_field_sd"], 5)
            if rec["pre_gate_field_sd"] else None)
        out[n] = rec
    # Where does the innovation have TRAINING support? The scored rows are all
    # CW-complete, but a feature that is null over the whole training set of the
    # early folds cannot be used by those folds, and a null driven by that is a
    # null about coverage, not about price. Measured, not assumed.
    full = s.select("area", "datetime_jst", *innov)
    ch = innov[0]
    obs = full.filter(full[ch].is_not_null() & full[ch].is_finite())
    out["training_support"] = {
        "channel_measured": ch,
        "first_datetime_with_innovation": str(obs["datetime_jst"].min()),
        "last_datetime_with_innovation": str(obs["datetime_jst"].max()),
        "n_rows_design_frame": full.height,
        "n_rows_with_innovation": obs.height,
        "per_area_first": {
            r["area"]: str(r["datetime_jst"]) for r in
            obs.group_by("area").agg(pl.col("datetime_jst").min())
            .sort("area").to_dicts()},
        "note": ("rows outside this range enter XGBoost as missing on the "
                 "innovation columns, exactly as phase85's own CW tier columns "
                 "do on days a cache lacks"),
    }
    pv, prov = j["dpv_hat_innov"], j["pv_innov_provenance"]
    out["pv_innov"] = {
        "provenance_counts": {r["pv_innov_provenance"]: int(r["count"])
                              for r in prov.value_counts().to_dicts()},
        "n_nan_on_scored_rows": int((~(pv.is_finite() & pv.is_not_null())).sum()),
        "mean_abs_where_observed": _j(float(
            j.filter(prov == "phase89")["dpv_hat_innov"].abs().mean()), 3),
        "note": ("night_zero is a structural zero (Delta-PV-hat of a PV quantity "
                 "at night); missing is NaN and is never silently zero"),
    }
    return out


def _build_diagnostics(pass_: str) -> dict:
    """The build's own per-area counts over the WHOLE design frame.

    The scored rows are all CW-complete, so the innovation is finite on every row
    that is SCORED; but the design frame also carries the pre-2024-08 training
    period, where the CW caches do not reach and the innovation columns are NULL.
    Those nulls go into XGBoost as missing values, exactly as phase85's own CW
    tier columns do. Reporting the count here so it is a disclosed property of
    the design and not an invisible one.
    """
    ckpt = _ckpt(pass_)
    files = sorted(ckpt.glob("_diagnostics_*.json")) if ckpt.exists() else []
    if not files:
        return {"status": "absent"}
    out: dict = {"status": "present", "n_files": len(files), "areas": {}}
    for f in files:
        out["areas"].update(json.loads(f.read_text()).get("areas", {}))
    tot = sum(v["n_rows"] for v in out["areas"].values())
    first = next(iter(out["areas"].values()))
    ch = next(iter(first["innov_wx"]))
    nulls = sum(v["innov_wx"][ch]["n_null"] for v in out["areas"].values())
    out["design_frame_rows"] = tot
    out["design_frame_rows_with_null_innovation"] = nulls
    out["design_frame_null_share"] = _j(nulls / tot, 4) if tot else None
    out["note"] = ("nulls are the pre-CW-cache training period; they enter "
                   "XGBoost as missing, as phase85's own CW columns do. Every "
                   "SCORED row is CW-complete, so the innovation is finite on "
                   "every row the contrast is measured on -- see "
                   "innovation_magnitude_and_nulls_on_scored_rows.")
    return out


def _all_missing_pv_areas(pass_: str) -> list[str]:
    """Areas where Delta-PV-hat is missing on EVERY row of the design frame."""
    side = _ckpt(pass_).parent / f"{_ckpt(pass_).name}_side"
    files = sorted(side.glob("*.parquet")) if side.exists() else []
    if not files:
        return []
    s = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    g = s.group_by("area").agg(
        pl.col("dpv_hat_innov").is_finite().sum().alias("n_obs"))
    return sorted(r["area"] for r in g.to_dicts() if r["n_obs"] == 0)


def _per_area(per_day: pl.DataFrame, cells: list[str], mod85) -> dict:
    """Per-area deltas against the pre-gate cell, with multiplicity correction.

    Nine simultaneous per-area DM tests need Holm/BH or a referee will (rightly)
    ask; this project's own lesson from phase23.5-23.7 is that per-area DM results
    which do not survive correction must be reported as not surviving. An area
    whose delta series is identically zero is a DEGENERATE test, not a null one --
    it happens when the added column is missing on every row of that area, so the
    cell IS the baseline -- and it is excluded from the correction and named.
    """
    out: dict = {}
    areas = sorted(per_day["area"].unique().to_list())
    raw: dict[str, dict[str, float]] = {c: {} for c in cells}
    for area in areas:
        sub = per_day.filter(pl.col("area") == area).sort("test_day")
        rec = {}
        for c in cells:
            v = np.asarray(sub[f"mae_{PRE_GATE}"] - sub[f"mae_{c}"], dtype=float)
            lo, hi = mod85._block_boot_ci(v)
            degen = bool(np.all(v == 0.0))
            rec[c] = {"delta": _j(float(v.mean()), 4),
                      "ci95": [_j(lo, 4), _j(hi, 4)],
                      "dm_p_l3": mod85._hln_dm(v, lag=3)["p_hln"],
                      "dm_p_l9": mod85._hln_dm(v, lag=9)["p_hln"],
                      "n_days": len(v), "degenerate_identical_to_baseline": degen}
            if not degen:
                raw[c][area] = rec[c]["dm_p_l3"]
        out[area] = rec
    out["_multiplicity"] = {}
    for c in cells:
        names = sorted(raw[c])
        ps = [raw[c][a] for a in names]
        if not ps:
            out["_multiplicity"][c] = {"status": "no non-degenerate areas"}
            continue
        holm, bh = mod85._holm(ps), mod85._benjamini_hochberg(ps)
        out["_multiplicity"][c] = {
            "n_tests": len(ps),
            "areas_included": names,
            "areas_excluded_degenerate": [a for a in areas if a not in names],
            "p_l3_holm": {a: _j(h, 4) for a, h in zip(names, holm, strict=True)},
            "p_l3_bh": {a: _j(v, 4) for a, v in zip(names, bh, strict=True)},
            "n_holm_below_0.10": sum(1 for h in holm if h < 0.10),
            "n_bh_below_0.10": sum(1 for v in bh if v < 0.10),
            "note": ("per-area DM tests are DESCRIPTIVE here; the pre-registered "
                     "statistic is the pooled one"),
        }
    return out


def analyse(mod85, mod89) -> int:  # noqa: C901, PLR0912, PLR0915 - one block per gate
    wx_cols = list(mod85.DEFAULT_WEATHER_COLS)
    key = ["area", "datetime_jst"]
    q, info_q = _read_pass("q")
    r, info_r = _read_pass("r")
    b, info_b = _read_pass("b")
    log.info("pass q %s | pass r %s | pass b %s", info_q["status"],
             info_r["status"], info_b["status"])
    if q is None:
        msg = (f"pass q absent ({info_q.get('reason')}): the question itself was "
               "not built, so there is nothing to report")
        raise RuntimeError(msg)

    p85 = pl.read_parquet(P85_PARQUET)
    cw_flags = [c for c in p85.columns if c.startswith("cw_ok_")]
    p85_sel = p85.select(*key, "test_day", "y_actual", *cw_flags,
                         *[f"y_pred_{c}" for c in (BASELINE, PRE_GATE, POST_GATE)])
    qsel = q.select(*key, pl.col("y_actual").alias("y_actual_mine"),
                    *[pl.col(f"cw_ok_{t}").alias(f"my_cw_ok_{t}")
                      for t in mod85.CW_JMA_TIERS],
                    *[f"y_pred_{c}" for c in PASS_CELLS["q"]])
    panel = p85_sel.join(qsel, on=key, how="inner")

    # ---------------------------------------------------------------- GATE A
    gate_a: dict = {
        "part1_row_set": {
            "n_slots_phase85": p85.height, "n_slots_pass_q": q.height,
            "n_slots_matched": panel.height,
            "row_set_identical": bool(panel.height == p85.height == q.height),
            "max_abs_diff_y_actual": _j(float(
                (panel["y_actual"] - panel["y_actual_mine"]).abs().max() or 0.0), 12),
            "cw_flags_identical": all(
                bool((panel[f"cw_ok_{t}"] == panel[f"my_cw_ok_{t}"]).all())
                for t in mod85.CW_JMA_TIERS),
            "why": ("adding columns to the design must not move a single row; a "
                    "duplicating join would show up here as a row-count change"),
        },
        "tolerance": BIT_TOL,
    }
    if r is not None:
        rsel = r.select(*key, "test_day",
                        pl.col(f"y_pred_{PRE_GATE}").alias("refit"))
        jr = p85_sel.select(*key, f"y_pred_{PRE_GATE}").join(rsel, on=key,
                                                            how="inner")
        d = (jr["refit"] - jr[f"y_pred_{PRE_GATE}"]).abs()
        gate_a["part2_bit_reproduction"] = {
            "status": "PRESENT",
            "fold_step_days": PASS_THIN["r"],
            "n_slots_refit": r.height, "n_slots_matched": jr.height,
            "refit_rows_are_a_subset_of_phase85": bool(jr.height == r.height),
            "n_test_days_refit": int(r["test_day"].n_unique()),
            "first_test_day": r["test_day"].min(),
            "last_test_day": r["test_day"].max(),
            "max_abs_diff": _j(float(d.max() or 0.0), 12),
            "mean_abs_diff": _j(float(d.mean() or 0.0), 12),
        }
    else:
        gate_a["part2_bit_reproduction"] = {"status": "ABSENT",
                                           "reason": info_r.get("reason")}
    # ---- part 4, which costs nothing: an area where Delta-PV-hat is missing on
    # every row gets an added column XGBoost can never split on, so the
    # pre_plus_pv_innov cell IS phase85's pre-gate cell -- on the FULL 732 folds.
    # Any difference from phase85's stored prediction there is a harness
    # difference with nowhere to hide.
    missing_areas = _all_missing_pv_areas("q")
    part4: dict = {"areas_with_no_observed_pv_innovation": missing_areas}
    worst4 = 0.0
    for area in missing_areas:
        sub = panel.filter(pl.col("area") == area)
        dd = (sub[f"y_pred_{C_PV}"] - sub[f"y_pred_{PRE_GATE}"]).abs()
        mx = float(dd.max()) if sub.height else None
        part4[area] = {"n_slots": sub.height,
                       "n_test_days": int(sub["test_day"].n_unique()),
                       "max_abs_diff": _j(mx, 12),
                       "n_slots_differing": int((dd > 0).sum())}
        worst4 = max(worst4, mx or 0.0)
    part4["status"] = "PRESENT" if missing_areas else "ABSENT"
    part4["max_abs_diff_over_areas"] = _j(worst4, 12) if missing_areas else None
    part4["why"] = ("a full-panel, all-folds bit reproduction of phase85's "
                    "headline cell through this script's harness, obtained as a "
                    "by-product of an all-missing feature column")
    gate_a["part4_full_panel_bit_reproduction"] = part4

    p2 = gate_a["part2_bit_reproduction"]
    p1 = gate_a["part1_row_set"]
    base_ok = (p1["row_set_identical"] and p1["cw_flags_identical"]
               and (p1["max_abs_diff_y_actual"] or 0.0) <= BIT_TOL)
    if p2["status"] == "ABSENT":
        gate_a["verdict"] = "PARTIAL" if base_ok else "FAIL"
    else:
        # `x or 1.0` would read a PERFECT match (0.0, which is falsy) as a
        # failure. An earlier dry run of this script did exactly that and
        # reported FAIL on a max |diff| of 0.0; hence the explicit None test.
        md = p2["max_abs_diff"]
        gate_a["verdict"] = ("PASS" if (base_ok
                                        and p2["refit_rows_are_a_subset_of_phase85"]
                                        and md is not None and md <= BIT_TOL)
                             else "FAIL")
    if (part4["status"] == "PRESENT"
            and (part4["max_abs_diff_over_areas"] or 0.0) > BIT_TOL):
        gate_a["verdict"] = "FAIL"
    gate_a["consequence_if_fail"] = (
        "Abort the price reading: a refit of phase85's own headline cell on "
        "phase85's own folds that does not reproduce means any delta measured "
        "against it mixes the added columns with an unexplained harness "
        "difference.")
    log.info("GATE A: %s (max |refit - phase85| = %s)", gate_a["verdict"],
             p2.get("max_abs_diff"))

    if b is not None:
        bsel = b.select(*key, *[f"y_pred_{c}" for c in PASS_CELLS["b"]])
        panel_b = panel.join(bsel, on=key, how="inner")
        if panel_b.height != b.height:
            msg = ("pass b's rows are not a subset of the scored panel, so the "
                   "controls are not on the panel the question is scored on")
            raise RuntimeError(msg)
    else:
        panel_b = None

    # ---- phase85's CW-complete subset, verbatim
    cw = panel.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
    coverage = {"n_slots_panel": panel.height,
                "n_slots_cw_complete": cw.height,
                "share_of_panel": _j(cw.height / panel.height, 4),
                "n_test_days_cw": int(cw["test_day"].n_unique()),
                "n_areas": int(cw["area"].n_unique()),
                "subset_rule": ("phase85's own: every cw_ok_* flag true, so the "
                                "CW arm is scored where every CW tier exists")}
    log.info("CW-complete: %d slots / %d test days", cw.height,
             coverage["n_test_days_cw"])

    main_cells = [BASELINE, PRE_GATE, POST_GATE, *PASS_CELLS["q"]]
    per_day = mod85._per_day_mae(cw, main_cells)
    imp = {c: mod85._pooled_daily(per_day, c) for c in main_cells if c != BASELINE}
    levels = {c: _j(float(per_day.group_by("test_day")
                          .agg(pl.col(f"mae_{c}").mean())[f"mae_{c}"].mean()), 4)
              for c in main_cells}
    entries = {c: mod85._entry(v) for c, v in imp.items()}
    margins = mod85._margins_for(float(imp[PRE_GATE].mean()))

    contrasts: dict = {
        f"{c}_minus_pre_gate": mod85._diff_entry(imp[c], imp[PRE_GATE], margins)
        for c in PASS_CELLS["q"]}
    contrasts["post_gate_minus_pre_gate_REPRODUCED"] = mod85._diff_entry(
        imp[POST_GATE], imp[PRE_GATE], margins)
    # differenced basis against the ladder's own replace-the-field basis, paired
    contrasts["wx_innov_minus_post_gate_cell"] = mod85._diff_entry(
        imp[C_WX], imp[POST_GATE])

    # ---- GATE A3: the statistics path itself, checked against the published number
    rep = contrasts["post_gate_minus_pre_gate_REPRODUCED"]
    gate_a["part3_statistics_path"] = {
        "reproduced_gate_timing_delta": rep["delta"],
        "published_phase85_value": P85_GATE_TIMING,
        "reproduced_ci95": rep["ci95"],
        "published_ci95": [-0.0047, 0.0099],
        "reproduced_n_days": rep["n_days"], "published_n_days": 732,
        "abs_gap": _j(abs((rep["delta"] or 0.0) - P85_GATE_TIMING), 6),
        "matches": bool(abs((rep["delta"] or 0.0) - P85_GATE_TIMING) < 5e-5
                        and rep["n_days"] == 732),
        "why": ("recomputing phase85's OWN published gate-timing contrast from "
                "its parquet through this script's aggregation, subset and "
                "bootstrap. If this does not land on +0.0026 over 732 days then "
                "the pipeline reporting the innovation contrast is not the "
                "pipeline that produced the paper's number."),
    }
    if not gate_a["part3_statistics_path"]["matches"]:
        gate_a["verdict"] = "FAIL"
    log.info("GATE A3 (statistics path reproduces phase85 gate timing): %s "
             "(%s vs %s)", gate_a["part3_statistics_path"]["matches"],
             rep["delta"], P85_GATE_TIMING)

    # ---- controls, on their own (thinned) day set, with the primary beside them
    controls: dict = {"status": info_b["status"]}
    if panel_b is not None:
        cwb = panel_b.filter(pl.all_horizontal(pl.col(f) for f in cw_flags))
        cells_b = [*main_cells, *PASS_CELLS["b"]]
        pdb = mod85._per_day_mae(cwb, cells_b)
        impb = {c: mod85._pooled_daily(pdb, c) for c in cells_b if c != BASELINE}
        controls = {
            "status": "PRESENT",
            "fold_step_days": PASS_THIN["b"],
            "n_test_days": int(cwb["test_day"].n_unique()),
            "n_slots": cwb.height,
            "note": ("a matched subsample of the SAME folds (phase85's schedule "
                     "with STEP widened to 3 days); the primary and reference "
                     "contrasts are recomputed on exactly these days so every "
                     "number in this block is like for like"),
            "levels_mae": {c: _j(float(pdb.group_by("test_day")
                                       .agg(pl.col(f"mae_{c}").mean())[f"mae_{c}"]
                                       .mean()), 4) for c in cells_b},
            "contrasts_vs_pre_gate": {
                f"{c}_minus_pre_gate": mod85._diff_entry(impb[c], impb[PRE_GATE])
                for c in (*PASS_CELLS["q"], *PASS_CELLS["b"], POST_GATE)},
            "per_area_descriptive": _per_area(pdb, list(PASS_CELLS["b"]), mod85),
        }

    # ---- PV cell on the sub-panel where its column is actually observed
    lo_h, hi_h = mod89.DAYTIME_HOURS
    pv_areas = sorted(pl.read_parquet(P89_SLOTS, columns=["area"])["area"]
                      .unique().to_list())
    day_sub = cw.filter(pl.col("datetime_jst").dt.hour().is_between(lo_h, hi_h)
                        & pl.col("area").is_in(pv_areas))
    pv_block: dict = {"window_jst": [lo_h, hi_h], "areas": pv_areas,
                      "n_slots": day_sub.height,
                      "n_test_days": int(day_sub["test_day"].n_unique()),
                      "note": ("DESCRIPTIVE sub-panel, not pre-registered: the "
                               "rows where Delta-PV-hat is an observed number "
                               "rather than a structural night zero or a NaN")}
    # The window is phase89's own DAYTIME_HOURS constant and the area list is
    # phase89's own coverage; neither was chosen here, which is the only reason a
    # post-hoc subset is worth printing. Both restrictions are then applied
    # SEPARATELY as well, so a reader can see which one carries any difference
    # instead of taking a doubly-restricted number on trust.
    def _sub_block(frame: pl.DataFrame, label: str) -> dict:
        if not frame.height:
            return {"status": "empty"}
        pdx = mod85._per_day_mae(frame, main_cells)
        ix = {c: mod85._pooled_daily(pdx, c) for c in main_cells if c != BASELINE}
        return {
            "restriction": label,
            "n_slots": frame.height,
            "n_test_days": int(frame["test_day"].n_unique()),
            "n_areas": int(frame["area"].n_unique()),
            "levels_mae": {c: _j(float(pdx.group_by("test_day")
                                       .agg(pl.col(f"mae_{c}").mean())[f"mae_{c}"]
                                       .mean()), 4) for c in main_cells},
            "contrasts_vs_pre_gate": {
                f"{c}_minus_pre_gate": mod85._diff_entry(ix[c], ix[PRE_GATE])
                for c in (*PASS_CELLS["q"], POST_GATE)},
            # the representation question stated precisely: differenced basis
            # against the ladder's own replace-the-field basis, PAIRED on days
            "wx_innov_minus_post_gate_cell": mod85._diff_entry(
                ix[C_WX], ix[POST_GATE]),
        }

    if day_sub.height:
        pv_block.update(_sub_block(day_sub, "daylight hours AND phase89 areas"))
        pv_block["decomposition_of_the_restriction"] = {
            "hours_only_all_nine_areas": _sub_block(
                cw.filter(pl.col("datetime_jst").dt.hour().is_between(lo_h, hi_h)),
                "daylight hours only, all nine areas"),
            "areas_only_all_hours": _sub_block(
                cw.filter(pl.col("area").is_in(pv_areas)),
                "phase89 areas only, all hours"),
            "why": ("the headline sub-panel restricts hours AND areas at once; "
                    "printing each restriction alone shows which one carries the "
                    "difference")}

    # ---------------------------------------------------------------- GATE C
    nulls = _side_nulls("q", cw, wx_cols)
    ratios = {f"{c}__innov": nulls.get(f"{c}__innov", {})
              .get("mean_abs_over_field_sd") for c in wx_cols}
    live = [v for v in ratios.values() if v is not None and v >= DEGENERATE_FRAC]
    # The other half of the same worry, and the one this project has been bitten
    # by: a treatment that silently did nothing produces a null that is
    # indistinguishable from the null being reported. So assert, BEFORE reading
    # the contrast, that the treated cell's predictions are not the untreated
    # cell's. If the innovation columns had arrived all-missing, XGBoost would
    # never split on them and this difference would be exactly zero.
    pdiff = (cw[f"y_pred_{C_WX}"] - cw[f"y_pred_{PRE_GATE}"]).abs()
    n_ident = int((pdiff == 0.0).sum())
    differ = {"max_abs_pred_diff": _j(float(pdiff.max()), 6),
              "mean_abs_pred_diff": _j(float(pdiff.mean()), 6),
              "n_rows": cw.height, "n_rows_bit_identical": n_ident,
              "share_bit_identical": _j(n_ident / cw.height, 6),
              "verdict": "PASS" if float(pdiff.max()) > 0 else "FAIL",
              "why": ("a null from a treatment that never reached the model is "
                      "not the null this phase is testing for")}
    gate_c = {"floor": DEGENERATE_FRAC, "mean_abs_over_field_sd": ratios,
              "n_channels_above_floor": len(live),
              "n_channels": len(wx_cols),
              "treated_cell_predictions_differ": differ,
              "verdict": "PASS" if (live and differ["verdict"] == "PASS")
                         else "FAIL",
              "consequence_if_fail": (
                  "UNDETERMINED: a numerically zero innovation means the two GPV "
                  "caches hold the same data, which is a data defect, not a "
                  "finding about price.")}
    log.info("GATE C: %s %s", gate_c["verdict"], ratios)

    # ---------------------------------------------------------------- GATE B
    if not PHYS_JSON.exists():
        msg = f"{PHYS_JSON} missing; run --step phys first"
        raise FileNotFoundError(msg)
    ph = json.loads(PHYS_JSON.read_text())
    gains, repro = ph["gains_vs_A_common"], ph["gate102b_reproduces_phase89"]
    worst = max((x["max_abs_diff_mw"] or 0.0) for x in repro.values())
    cands = {}
    for name in ("C_pre_plus_pv_innov", "D_pre_plus_wx_innov"):
        g = gains[name]
        cands[name] = {"delta_mw": g["delta_mw"], "ci95": g["ci95"],
                       "n_days": g["n_days"],
                       "excludes_zero": g["excludes_zero"],
                       "recovered_fraction_of_phase89_gain": _j(
                           (g["delta_mw"] or 0.0) / P89_GAIN_MW, 3)}
    passes = [k for k, v in cands.items()
              if v["excludes_zero"] and (v["delta_mw"] or 0.0) > 0
              and (v["recovered_fraction_of_phase89_gain"] or 0.0) >= RECOVER_FRAC]
    gate_b = {
        "phase89_reference_gain_mw": P89_GAIN_MW,
        "required_recovery_fraction": RECOVER_FRAC,
        "reproduction_of_phase89_max_abs_mw": _j(worst, 9),
        "reproduction_tolerance_mw": 1e-6,
        "reproduction_ok": bool(worst <= 1e-6),
        "per_area_reproduction": repro,
        "candidates": cands,
        "noise_placebo": gains["E_pre_plus_noise"],
        "post_only_reference": gains["B_post_only"],
        "levels_mae_mw_common_panel": ph["levels_common_panel"],
        "per_area_gains": ph["per_area_gains_vs_A_common"],
        "cells_that_pass": passes,
        "verdict": "PASS" if (passes and worst <= 1e-6) else "FAIL",
        "consequence_if_fail": (
            "UNDETERMINED-CONSTRUCTION-BROKEN: the innovation is by construction "
            "the update to the weather field, so a physical arm that does not "
            "improve means the construction is wrong and NO price branch may be "
            "read."),
    }
    log.info("GATE B: %s %s", gate_b["verdict"],
             {k: (v["delta_mw"], v["recovered_fraction_of_phase89_gain"])
              for k, v in cands.items()})

    # ---------------------------------------------------------------- GATE D
    wx = contrasts[f"{C_WX}_minus_pre_gate"]
    lo, hi = wx["ci95"]
    p3, p9 = wx["dm_l3"]["p_hln"], wx["dm_l9"]["p_hln"]
    if gate_b["verdict"] != "PASS":
        branch, verdict = "gate_b_failed", "UNDETERMINED-CONSTRUCTION-BROKEN"
    elif gate_c["verdict"] != "PASS":
        branch, verdict = "gate_c_failed", "UNDETERMINED-DEGENERATE-INNOVATION"
    elif gate_a["verdict"] == "FAIL":
        branch, verdict = "gate_a_failed", "UNDETERMINED-HARNESS"
    elif lo > TICK and p3 < 0.05 and p9 < 0.05:
        branch, verdict = "a", "LEARNER-LIMITATION"
    elif lo > 0 and lo <= TICK:
        branch, verdict = "c", "SMALL-BUT-REAL"
    elif hi < 0:
        branch, verdict = "d", "NEGATIVE"
    else:
        branch, verdict = "b", "SIGNAL-NULL"
    noise_delta = (controls.get("contrasts_vs_pre_gate", {})
                   .get(f"{C_NOISE}_minus_pre_gate", {}).get("delta"))
    gate_d = {
        "statistic": ("pooled daily improvement of pre_plus_wx_innov minus that "
                      "of msm_d2_12z_cw, on phase85's CW-complete subset, with "
                      "phase85's 7-day circular block bootstrap (10000 draws) "
                      "and HLN-DM at L=3 and L=9 -- i.e. phase85's own "
                      "gate-timing construction with the cell swapped"),
        "delta": wx["delta"], "ci95": wx["ci95"], "n_days": wx["n_days"],
        "dm_p_l3": p3, "dm_p_l9": p9,
        "tick": TICK,
        "equivalence_at_tick": wx["equivalence"]["margins"]["tick"]["verdict"],
        "equivalence_detail": wx["equivalence"],
        "noise_placebo_delta_thinned": noise_delta,
        "ladder_post_gate_delta_published": P85_GATE_TIMING,
        "branch": branch, "verdict": verdict,
    }
    log.info("GATE D: branch %s -> %s | delta %+.4f [%+.4f, %+.4f] p3=%s p9=%s",
             branch, verdict, wx["delta"], lo, hi, p3, p9)

    # ---------------------------------------------------------------- GATE E
    if controls["status"] == "PRESENT":
        cb = controls["contrasts_vs_pre_gate"]
        gate_e = {
            "on_thinned_day_set": controls["n_test_days"],
            "wx_differenced_basis_delta": cb[f"{C_WX}_minus_pre_gate"]["delta"],
            "wx_ci95": cb[f"{C_WX}_minus_pre_gate"]["ci95"],
            "raw_union_basis_delta": cb[f"{C_RAW}_minus_pre_gate"]["delta"],
            "raw_ci95": cb[f"{C_RAW}_minus_pre_gate"]["ci95"],
            "difference": _j((cb[f"{C_WX}_minus_pre_gate"]["delta"] or 0.0)
                             - (cb[f"{C_RAW}_minus_pre_gate"]["delta"] or 0.0), 4),
            "reading": ("positive means the DIFFERENCED basis extracts more than "
                        "the raw-union basis, i.e. part of the ladder's post-gate "
                        "null was a representation artefact"),
        }
    else:
        gate_e = {"status": "ABSENT", "reason": info_b.get("reason")}

    out = {
        "phase": "102 (companion run: signal vs learner)",
        "title": ("Is the post-gate price null a property of the signal or of the "
                  "learner? The run-to-run innovation handed directly to the "
                  "ladder's own learner, out of sample"),
        "question": ("phase85's post-gate cell REPLACES the pre-gate field, and an "
                     "axis-aligned tree cannot form the difference of two "
                     "columns. This feeds the difference itself as extra columns "
                     "and measures what the learner does with it."),
        "verdict": verdict, "branch": branch,
        "gate_a_harness": gate_a,
        "gate_b_physical_positive_control": gate_b,
        "gate_c_innovation_non_degenerate": gate_c,
        "gate_d_price": gate_d,
        "gate_e_representation": gate_e,
        "build_passes": {"q": info_q, "r": info_r, "b": info_b,
                         "cells_read_from_phase85_parquet": [BASELINE, PRE_GATE,
                                                             POST_GATE]},
        "panel": coverage,
        "levels_mae_jpy_kwh": levels,
        "improvements_vs_price_only": entries,
        "contrasts_vs_pre_gate_full_panel": contrasts,
        "per_area_descriptive_full_panel": _per_area(per_day,
                                                    list(PASS_CELLS["q"]), mod85),
        "controls_thinned_panel": controls,
        "pv_cell_daylight_sub_panel_descriptive": pv_block,
        "innovation_magnitude_and_nulls_on_scored_rows": nulls,
        "build_diagnostics_design_frame": _build_diagnostics("q"),
        "feature_importance_descriptive": _importance_summary(),
        "estimator": {
            "model": ("phase85's XGBRegressor verbatim: n_estimators=300, "
                      "max_depth=4, learning_rate=0.05, random_state=2026, "
                      "n_jobs=4, tree_method=hist"),
            "reuse": ("phase85's panel, fold schedule (expanding window, STEP=1d "
                      "for the question, VAL_WINDOW=14d, "
                      "INITIAL_TRAIN_END=2024-07-18), design matrix, CW-complete "
                      "subset, per-day aggregation, 7-day circular block "
                      "bootstrap and HLN-DM are IMPORTED AND CALLED, not "
                      "reimplemented. The innovation construction and the "
                      "_design/_fit rebinding are the committed phase102 "
                      "module's; the rebinding is asserted in force, asserted "
                      "called for exactly the requested areas, and demonstrated "
                      "by GATE A."),
            "physical": ph["estimator"],
        },
        "inputs": {
            "phase85_parquet": str(P85_PARQUET), "phase85_json": str(P85_JSON),
            "phase89_json": str(P89_JSON), "phase89_per_slot": str(P89_SLOTS),
            "phase90_json": str(P90_JSON),
            "phase102_importance_json": str(IMP_JSON),
            "physical_panel_built_here": str(PHYS_PARQUET),
            "physical_arm_json": str(PHYS_JSON),
            "price_checkpoints": {p: str(_ckpt(p)) for p in PASS_CELLS},
            "scripts_reused": [str(P85_PATH), str(P89_PATH), str(P102_PATH)],
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    for c, e in entries.items():
        log.info("  imp vs price-only %-22s %+.4f [%+.4f, %+.4f]", c,
                 e["improvement"], e["ci95"][0], e["ci95"][1])
    for k, v in contrasts.items():
        log.info("  %-46s %+.4f [%+.4f, %+.4f] p3=%s", k, v["delta"],
                 v["ci95"][0], v["ci95"][1], v["dm_l3"]["p_hln"])
    if controls["status"] == "PRESENT":
        for k, v in controls["contrasts_vs_pre_gate"].items():
            log.info("  [thin] %-40s %+.4f [%+.4f, %+.4f]", k, v["delta"],
                     v["ci95"][0], v["ci95"][1])
    return 0 if gate_a["verdict"] in ("PASS", "PARTIAL") else 1


def _importance_summary() -> dict:
    if not IMP_JSON.exists():
        return {"status": "absent"}
    d = json.loads(IMP_JSON.read_text())
    return {
        "source": IMP_JSON.name,
        "note": ("DESCRIPTIVE, from an earlier session of the committed phase102 "
                 "module: total_gain share of the four innovation columns in the "
                 "pre_plus_wx_innov cell, on ten real folds per area. It says the "
                 "learner DOES split on them, which is what makes a null a "
                 "statement about the signal rather than about the feature "
                 "reaching the model."),
        "design": d.get("design"),
        "per_area_mean_innovation_gain_share": {
            a: v.get("mean_innovation_gain_share")
            for a, v in d.get("areas", {}).items()},
        "n_innovation_cols_used": {
            a: v.get("mean_n_innovation_cols_used")
            for a, v in d.get("areas", {}).items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("build", "phys", "analyse"))
    ap.add_argument("--areas", default="")
    ap.add_argument("--pass", dest="pass_", default="q", choices=tuple(PASS_CELLS))
    args = ap.parse_args()

    mod85 = _load_module(P85_PATH, "p85")
    mod89 = _load_module(P89_PATH, "p89")
    mod102 = _load_module(P102_PATH, "p102")
    mod102.mod_p89 = mod89   # phase102's rebound _design reads phase89's constants

    if args.step == "build":
        areas = [a for a in args.areas.split(",") if a] or list(mod85.AREA_COORDS)
        return build(mod85, mod102, areas, args.pass_)
    if args.step == "phys":
        return phys(mod85, mod89, mod102)
    return analyse(mod85, mod89)


if __name__ == "__main__":
    sys.exit(main())
