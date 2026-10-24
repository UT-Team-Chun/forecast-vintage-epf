r"""Phase 113: is the near-gate flatness a property of the information or the learner?

WHY THIS EXISTS
---------------
Applied Energy referee R1.8 (and R3.2): the ladder runs XGBoost at one fixed
configuration, "not tuned by area. A weak exploitation of the later weather
innovation could be mistaken for absence of useful information." R1.9 asked
whether the result survives the MSM model upgrade inside the window, and R1.11
whether 10-m wind (not hub height) distorts it. Each is answerable on the primary
panel by changing ONE thing and leaving the rest verbatim.
Precommit: docs/research/2026-10-23_phase112_ae_review_response_precommit.md
(section "Phase 113"), committed before this script ran.

HOW NOTHING ELSE CHANGES
------------------------
phase96's pattern: import phase85's module, rebind only ``_fit`` (and, where a
variant needs extra cells, ``_design``), and run phase85's own ``_run_area``.
The panel, its row set, the fold schedule and the cw_ok flags are therefore
phase85's. Every rebinding is asserted, because phase91 showed a silent swap
reports a bug as a null.

VARIANTS (one checkpoint tree each; --variant)
----------------------------------------------
  repro_wind  Gate 113-0: the three cells refitted with phase85's own learner on
              the first five test days must match the frozen per-slot output to
              1e-9. In the same pass: the pre/post CW cells with the wind channel
              removed, and (Kyushu) the pre CW cell with wind rescaled to 100 m by a
              fixed-exponent power law -- a monotone transform a tree cannot see.
  era         the three cells trained only on slots from the MSM upgrade
              (2025-05-20) onward; first test day 2025-07-20.
  dnn         the frozen phase46 MLP on the three cells.
  tuned       XGBoost with rolling nested Optuna tuning per area and cell.
  lear_post   LEAR (phase96 estimator) on the post-gate CW cell; phase96 already
              holds LEAR's price-only and pre-gate predictions on this panel.

CLI (from the repository root):
  uv run python scripts/18c_learner_strength.py \
      --variant repro_wind --areas kyushu
  uv run python ... --step analyse
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import subprocess
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import optuna
import polars as pl

# isort: off
# xgboost must load its OpenMP runtime before torch does. In the other order the
# first XGBoost fit segfaults on macOS (exit 139, no traceback): every XGBoost
# variant of this phase died silently right after building its panel until the
# order was swapped. The sorter would put torch first, hence the fence.
import xgboost as xgb
import torch
from torch import nn
# isort: on

from alg.config import EpfConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase113")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
HERE = Path(__file__).parent
P85 = HERE / "10_primary_ladder_cw.py"
P96 = HERE / "18b_lear_ladder.py"
P85_PARQUET = RUN / "phase85_exact_ladder_per_slot.parquet"
P96_CKPT = RUN / "phase96_lear_checkpoints"
OUT_JSON = RUN / "phase113_learner_strength.json"
TUNED_PARAMS_DIR = RUN / "phase113_tuned_params"
PRECOMMIT = "docs/research/2026-10-23_phase112_ae_review_response_precommit.md"

BASE, PRE, POST = "price_only", "msm_d2_12z_cw", "msm_d1_00z_cw"
STD = (BASE, PRE, POST)
WIND = "wind_speed_kmh"
VARIANTS = ("repro_wind", "era", "dnn", "tuned", "lear_post")
SEED = 2026

REPRO_LAST_TEST = date(2024, 8, 5)     # Gate 113-0: the first five test days
UPGRADE = date(2025, 5, 20)            # MSM physics upgrade (jma2025msmupgrade)
ERA_FIRST_TEST = date(2025, 7, 20)
HUB_FACTOR = (100.0 / 10.0) ** (1.0 / 7.0)
HUB_AREAS = ("kyushu",)
FIRST_TEST = date(2024, 8, 1)
TUNE_EVERY = timedelta(days=91)
INNER_VAL = timedelta(days=28)
N_TRIALS = 40
XGB_JOBS = 4

# phase46's frozen network, reproduced field for field (its JSON is the record).
DNN = {"hidden_units": (128, 64), "dropout": 0.2, "learning_rate": 1e-3,
       "weight_decay": 0.0, "batch_size": 256, "max_epochs": 200,
       "val_tail_days": 14, "patience": 10, "min_delta": 1e-6}

_colsets: dict[str, tuple[str, ...]] = {}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _test_day(test: pl.DataFrame) -> date:
    return test["datetime_jst"][0].date()


def _cell_of(cols: list[str]) -> str | None:
    t = tuple(cols)
    for name, cs in _colsets.items():
        if cs == t:
            return name
    return None


# ------------------------------------------------------------------ learners
def _fit_dnn(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray,
             xte: np.ndarray, seed: int) -> np.ndarray:
    """phase46 ``_fit_dnn`` verbatim apart from reading the config dict above."""
    torch.manual_seed(seed)
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    ysd = ysd if ysd > 1e-8 else 1.0
    xf = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    yf = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32)
    xv = torch.tensor((xva - mu) / sd, dtype=torch.float32)
    yv = torch.tensor((yva - ymu) / ysd, dtype=torch.float32)
    xt = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    h1, h2 = DNN["hidden_units"]
    model = nn.Sequential(nn.Linear(xf.shape[1], h1), nn.ReLU(), nn.Dropout(DNN["dropout"]),
                          nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(DNN["dropout"]),
                          nn.Linear(h2, 1))
    opt = torch.optim.Adam(model.parameters(), lr=DNN["learning_rate"],
                           weight_decay=DNN["weight_decay"])
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    n = xf.shape[0]
    for _ in range(DNN["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, DNN["batch_size"]):
            idx = perm[i:i + DNN["batch_size"]]
            opt.zero_grad()
            loss_fn(model(xf[idx]).squeeze(-1), yf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xv).squeeze(-1), yv).item())
        if vl < best_val - DNN["min_delta"]:
            best_val, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= DNN["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(xt).squeeze(-1).numpy().astype(float) * ysd + ymu


def _dnn_cell(train: pl.DataFrame, test: pl.DataFrame, cols: list[str], cell: str) -> np.ndarray:
    td = _test_day(test)
    val_start = td - timedelta(days=DNN["val_tail_days"])
    dts = train["datetime_jst"].dt.date()
    fit, val = train.filter(dts < val_start), train.filter(dts >= val_start)
    area = test["area"][0]
    seed = int(zlib.crc32(f"{area}|{td.isoformat()}|{cell}".encode()) ^ SEED) % (2**31)
    return _fit_dnn(fit.select(cols).to_numpy().astype(float),
                    fit["price_yen_kwh"].to_numpy().astype(float),
                    val.select(cols).to_numpy().astype(float),
                    val["price_yen_kwh"].to_numpy().astype(float),
                    test.select(cols).to_numpy().astype(float), seed)


def _tuning_date(td: date) -> date:
    k = (td - FIRST_TEST).days // TUNE_EVERY.days
    return FIRST_TEST + TUNE_EVERY * k


def _tune(train: pl.DataFrame, cols: list[str], t0: date) -> dict:
    """Optuna TPE over the precommitted space, scored on [t0-28d, t0)."""
    dts = train["datetime_jst"].dt.date()
    inner = train.filter(dts < t0)
    idts = inner["datetime_jst"].dt.date()
    tr, va = inner.filter(idts < t0 - INNER_VAL), inner.filter(idts >= t0 - INNER_VAL)
    xtr, ytr = tr.select(cols).to_numpy().astype(float), tr["price_yen_kwh"].to_numpy()
    xva, yva = va.select(cols).to_numpy().astype(float), va["price_yen_kwh"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        p = {"max_depth": trial.suggest_int("max_depth", 2, 10),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50,
                             eval_metric="mae", tree_method="hist",
                             random_state=SEED, n_jobs=XGB_JOBS, **p)
        m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        pred = m.predict(xva, iteration_range=(0, int(m.best_iteration) + 1))
        return float(np.mean(np.abs(pred - yva)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    return {**best.params, "n_estimators": best.user_attrs["best_iteration"] + 1,
            "inner_val_mae": float(best.value), "tuning_date": t0.isoformat(),
            "n_inner_train": tr.height, "n_inner_val": va.height}


_tuned: dict[tuple[str, str, str], dict] = {}


def _tuned_cell(train: pl.DataFrame, test: pl.DataFrame, cols: list[str], cell: str) -> np.ndarray:
    area, t0 = test["area"][0], _tuning_date(_test_day(test))
    key = (area, cell, t0.isoformat())
    path = TUNED_PARAMS_DIR / f"{area}.json"
    if not _tuned and path.exists():
        for rec in json.loads(path.read_text()):
            _tuned[(rec["area"], rec["cell"], rec["tuning_date"])] = rec
    if key not in _tuned:
        rec = {"area": area, "cell": cell, **_tune(train, cols, t0)}
        _tuned[key] = rec
        TUNED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([v for k, v in _tuned.items() if k[0] == area], indent=1))
        log.info("tuned %s %s @%s: depth %d lr %.3f trees %d inner MAE %.4f", area, cell,
                 t0, rec["max_depth"], rec["learning_rate"], rec["n_estimators"],
                 rec["inner_val_mae"])
    p = {k: v for k, v in _tuned[key].items()
         if k in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                  "min_child_weight", "reg_lambda", "reg_alpha", "n_estimators")}
    m = xgb.XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=XGB_JOBS, **p)
    m.fit(train.select(cols).to_numpy().astype(float),
          train["price_yen_kwh"].to_numpy().astype(float))
    return m.predict(test.select(cols).to_numpy().astype(float)).astype(float)


# ------------------------------------------------------------------ build
def build(variant: str, areas: list[str]) -> int:
    mod = _load(P85, "p85")
    lear = _load(P96, "p96")._lear_fit if variant == "lear_post" else None
    original_fit, original_design = mod._fit, mod._design

    def design_variant(df):
        out, cells = original_design(df)
        _colsets.clear()
        for name in STD:
            _colsets[name] = tuple(cells[name])
        if len(set(_colsets.values())) != len(STD):
            msg = "two standard cells share a column list"
            raise RuntimeError(msg)
        if variant == "repro_wind":
            for name in (PRE, POST):
                nowind = [c for c in cells[name] if not c.startswith(WIND)]
                if len(nowind) != len(cells[name]) - 1:
                    msg = f"{name}: expected exactly one wind column"
                    raise RuntimeError(msg)
                cells[f"{name}_nowind"] = nowind
                _colsets[f"{name}_nowind"] = tuple(nowind)
            area = out["area"][0]
            if area in HUB_AREAS:
                wcol = f"{WIND}_{PRE}"
                out = out.with_columns((pl.col(wcol) * HUB_FACTOR).alias(f"{wcol}_hub"))
                hub = [f"{wcol}_hub" if c == wcol else c for c in cells[PRE]]
                cells[f"{PRE}_hub"] = hub
                _colsets[f"{PRE}_hub"] = tuple(hub)
        return out, cells

    def fit_variant(train, test, cols):
        cell = _cell_of(cols)
        nan = np.full(test.height, np.nan)
        if cell is None:
            return nan
        td = _test_day(test)
        if variant == "repro_wind":
            if cell in STD:
                return original_fit(train, test, cols) if td <= REPRO_LAST_TEST else nan
            return original_fit(train, test, cols)
        if variant == "era":
            if td < ERA_FIRST_TEST:
                return nan
            return original_fit(train.filter(pl.col("datetime_jst").dt.date() >= UPGRADE),
                                test, cols)
        if variant == "dnn":
            return _dnn_cell(train, test, cols, cell)
        if variant == "tuned":
            return _tuned_cell(train, test, cols, cell)
        if variant == "lear_post":
            return lear(train, test, cols) if cell == POST else nan
        msg = f"unknown variant {variant}"
        raise AssertionError(msg)

    mod._fit, mod._design = fit_variant, design_variant
    if mod._fit is original_fit or mod._design is original_design:
        msg = "a rebinding did not take effect"
        raise RuntimeError(msg)
    if mod._fit.__name__ != "fit_variant" or mod._design.__name__ != "design_variant":
        msg = "a rebinding bound the wrong function"
        raise RuntimeError(msg)
    if variant == "dnn":
        # One thread, not phase46's two. XGBoost's OpenMP runtime is loaded first
        # (see the import fence), and torch segfaults the moment it opens a
        # parallel region against it: two threads crashed on the first epoch,
        # one thread trains cleanly. The net is small; areas run in parallel.
        torch.set_num_threads(1)
    mod.CKPT = RUN / f"phase113_{variant}_checkpoints"
    mod.CKPT.mkdir(parents=True, exist_ok=True)
    for a in areas:
        log.info("=== %s (%s) ===", a, variant)
        mod._run_area(a)
    return 0


# ------------------------------------------------------------------ analyse
def _load_ckpt(variant: str) -> pl.DataFrame | None:
    """A variant's predictions, or None until ALL nine areas are in.

    A partial set would still produce a pooled number, and an interim JSON written
    from four areas would carry a learner's "result" the guard could bind to.
    """
    files = sorted((RUN / f"phase113_{variant}_checkpoints").glob("*.parquet"))
    if len(files) < 9:  # noqa: PLR2004 - the nine JEPX areas
        return None
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")


def _cw_ok(df: pl.DataFrame) -> pl.DataFrame:
    flags = [c for c in df.columns if c.startswith("cw_ok_")]
    return df.filter(pl.all_horizontal(pl.col(f) for f in flags))


def _contrast_block(p85, df: pl.DataFrame, base: str, pre: str, post: str,
                    label: str) -> dict:
    """Pre-gate improvement and gate contrast for one learner, phase85's statistics."""
    df = df.drop_nulls([f"y_pred_{base}", f"y_pred_{pre}", f"y_pred_{post}"]).filter(
        pl.col(f"y_pred_{base}").is_not_nan() & pl.col(f"y_pred_{pre}").is_not_nan()
        & pl.col(f"y_pred_{post}").is_not_nan())
    per_day = (df.with_columns(
        *[(pl.col(f"y_pred_{c}") - pl.col("y_actual")).abs().alias(f"ae_{c}")
          for c in (base, pre, post)])
        .group_by("area", "test_day")
        .agg(*[pl.col(f"ae_{c}").mean().alias(f"mae_{c}") for c in (base, pre, post)])
        .sort("area", "test_day"))

    def pooled(cell: str) -> np.ndarray:
        return np.asarray((per_day.with_columns(
            (pl.col(f"mae_{base}") - pl.col(f"mae_{cell}")).alias("imp"))
            .group_by("test_day").agg(pl.col("imp").mean()).sort("test_day")["imp"]),
            dtype=float)

    d_pre, d_post = pooled(pre), pooled(post)
    pre_entry = p85._entry(d_pre)
    gate = p85._diff_entry(d_post, d_pre, p85._margins_for(pre_entry["improvement"]))
    levels = per_day.select(*[pl.col(f"mae_{c}").mean().alias(c) for c in (base, pre, post)]
                            ).row(0, named=True)
    ucb = gate["ci95"][1]
    return {"label": label, "n_test_days": len(d_pre), "n_slots": df.height,
            "n_areas": per_day["area"].n_unique(),
            "mae_levels": {k: round(float(v), 4) for k, v in levels.items()},
            "pre_gate_improvement": pre_entry, "gate_contrast": gate,
            "gate_ucb95": ucb,
            "gate_ucb_share_of_pre_gate": (round(ucb / pre_entry["improvement"], 4)
                                           if pre_entry["improvement"] else None)}


def analyse() -> int:  # noqa: C901, PLR0912, PLR0915 - one block per precommitted variant
    p85 = _load(P85, "p85")
    ref = _cw_ok(pl.read_parquet(P85_PARQUET))
    out: dict = {"phase": "113", "precommit": PRECOMMIT, "git_sha": _git_sha(),
                 "reference": P85_PARQUET.name, "learners": {}, "gates": {}}
    out["learners"]["xgb_fixed"] = _contrast_block(p85, ref, BASE, PRE, POST,
                                                   "XGBoost, fixed configuration (phase85)")
    key = ["area", "datetime_jst"]

    rw = _load_ckpt("repro_wind")
    if rw is not None:
        first = rw.filter(pl.col("test_day") <= REPRO_LAST_TEST.isoformat())
        j = first.select(*key, *[f"y_pred_{c}" for c in STD]).join(
            pl.read_parquet(P85_PARQUET).select(
                *key, *[pl.col(f"y_pred_{c}").alias(f"ref_{c}") for c in STD]),
            on=key, how="inner")
        worst = {c: float((j[f"y_pred_{c}"] - j[f"ref_{c}"]).abs().max() or 0.0)
                 for c in STD}
        out["gate113_0_reproduction"] = {
            "test_days": f"first five (<= {REPRO_LAST_TEST})", "n_slots": j.height,
            "areas": sorted(first["area"].unique().to_list()),
            "max_abs_diff": worst, "tolerance": 1e-9,
            "verdict": "PASS" if j.height and max(worst.values()) <= 1e-9 else "FAIL"}
        nw = rw.select(*key, "test_day", "y_actual",
                       *[c for c in rw.columns if c.startswith("cw_ok_")],
                       f"y_pred_{PRE}_nowind", f"y_pred_{POST}_nowind").join(
            pl.read_parquet(P85_PARQUET).select(*key, f"y_pred_{BASE}"), on=key)
        nw = _cw_ok(nw)
        blk = _contrast_block(p85, nw, BASE, f"{PRE}_nowind", f"{POST}_nowind",
                              "XGBoost fixed, wind channel removed")
        head = out["learners"]["xgb_fixed"]["pre_gate_improvement"]["improvement"]
        blk["headline_change_share"] = round(
            (blk["pre_gate_improvement"]["improvement"] - head) / head, 4)
        out["wind_ablation"] = blk
        hub_col = f"y_pred_{PRE}_hub"
        if hub_col in rw.columns:
            h = (rw.filter(pl.col(hub_col).is_not_null() & pl.col(hub_col).is_not_nan())
                 .select(*key, "y_actual", hub_col)
                 .join(pl.read_parquet(P85_PARQUET).select(*key, f"y_pred_{PRE}",
                                                           f"y_pred_{BASE}"), on=key))
            # Not exactly zero: the rescaled values round differently at the
            # histogram's bin edges, so a handful of slots fall on the other side of
            # a split. Report the size of that, and what it does to the improvement.
            d = (h[hub_col] - h[f"y_pred_{PRE}"]).abs()
            mae = {c: float((h[c] - h["y_actual"]).abs().mean())
                   for c in (hub_col, f"y_pred_{PRE}", f"y_pred_{BASE}")}
            out["hub_height_invariance"] = {
                "areas": list(HUB_AREAS), "factor": HUB_FACTOR, "n_slots": h.height,
                "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                "share_slots_changed": float((d > 1e-9).mean()),
                "improvement_original": mae[f"y_pred_{BASE}"] - mae[f"y_pred_{PRE}"],
                "improvement_hub": mae[f"y_pred_{BASE}"] - mae[hub_col]}
        g = out["wind_ablation"]
        out["gates"]["g113c_wind"] = {
            "evaluated": True,
            "passes": bool(abs(g["headline_change_share"]) < 0.10
                           and g["gate_ucb_share_of_pre_gate"] is not None
                           and g["gate_ucb_share_of_pre_gate"] < 0.10),
            "rule": "headline change < 10% and gate UCB < 10% of the pre-gate improvement",
            "if_fails": "report that the wind channel's measurement moves the result"}

    era = _load_ckpt("era")
    if era is not None:
        e = _cw_ok(era.filter(pl.col("test_day") >= ERA_FIRST_TEST.isoformat()))
        out["single_version"] = {
            "trained_from": UPGRADE.isoformat(), "first_test_day": ERA_FIRST_TEST.isoformat(),
            "restricted": _contrast_block(p85, e, BASE, PRE, POST,
                                          "XGBoost fixed, trained on post-upgrade slots only"),
            "full_history_same_days": _contrast_block(
                p85, ref.filter(pl.col("test_day") >= ERA_FIRST_TEST.isoformat()),
                BASE, PRE, POST, "XGBoost fixed, full history, same test days")}
        r = out["single_version"]["restricted"]
        ci = r["gate_contrast"]["ci95"]
        out["gates"]["g113b_single_version"] = {
            "evaluated": True,
            "passes": bool(ci[0] <= 0 <= ci[1] and r["gate_ucb_share_of_pre_gate"] is not None
                           and r["gate_ucb_share_of_pre_gate"] < 0.10),
            "rule": "gate CI includes 0 and UCB < 10% of the restricted pre-gate improvement",
            "if_fails": "report that the near-gate result does not hold within one model version"}

    for variant, label in (("dnn", "MLP (phase46 frozen configuration)"),
                           ("tuned", "XGBoost, rolling nested tuning")):
        c = _load_ckpt(variant)
        if c is not None:
            out["learners"][variant] = _contrast_block(p85, _cw_ok(c), BASE, PRE, POST, label)
    lp = _load_ckpt("lear_post")
    if lp is not None and P96_CKPT.exists():
        l96 = pl.concat([pl.read_parquet(f) for f in sorted(P96_CKPT.glob("*.parquet"))],
                        how="diagonal")
        lj = _cw_ok(lp.select(*key, "test_day", "y_actual",
                              *[c for c in lp.columns if c.startswith("cw_ok_")],
                              f"y_pred_{POST}").join(
            l96.select(*key, f"y_pred_{BASE}", f"y_pred_{PRE}"), on=key))
        out["learners"]["lear"] = _contrast_block(p85, lj, BASE, PRE, POST,
                                                  "LEAR (phase96 estimator)")
    if TUNED_PARAMS_DIR.exists():
        recs = [r for f in sorted(TUNED_PARAMS_DIR.glob("*.json"))
                for r in json.loads(f.read_text())]
        out["tuned_params"] = {"n_tunings": len(recs), "records": recs}

    learners = out["learners"]
    needed = {"xgb_fixed", "tuned", "dnn", "lear"}
    if needed <= set(learners):
        robust = all(learners[k]["gate_ucb_share_of_pre_gate"] is not None
                     and learners[k]["gate_ucb_share_of_pre_gate"] < 0.10
                     for k in ("tuned", "dnn", "lear"))
        any_sig = [k for k in ("tuned", "dnn", "lear")
                   if learners[k]["gate_contrast"]["ci95"][0] > 0]
        out["gates"]["g113a_learner_robustness"] = {
            "evaluated": True, "passes": bool(robust and not any_sig),
            "significant_learners": any_sig,
            "rule": "for tuned/dnn/lear the gate UCB < 10% of that learner's pre-gate "
                    "improvement, and no learner's gate CI lies above zero",
            "if_fails": "state learner dependence; weaken 'no detectable' to "
                        "'small and learner-dependent'"}
        fixed_pre = learners["xgb_fixed"]["mae_levels"][PRE]
        tuned_pre = learners["tuned"]["mae_levels"][PRE]
        out["gates"]["g113a_prime_level"] = {
            "evaluated": True, "passes": bool((fixed_pre - tuned_pre) / fixed_pre <= 0.05),
            "tuned_minus_fixed_share": round((tuned_pre - fixed_pre) / fixed_pre, 4),
            "rule": "tuned pre-gate MAE no more than 5% below fixed",
            "if_fails": "report the tuned headline beside the fixed one in the text"}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    log.info("wrote %s", OUT_JSON)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=("build", "analyse"), default="build")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--areas", nargs="+",
                    default=["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
                             "kansai", "chugoku", "shikoku", "kyushu"])
    a = ap.parse_args()
    if a.step == "analyse":
        return analyse()
    if a.variant is None:
        ap.error("--variant is required for --step build")
    return build(a.variant, a.areas)


if __name__ == "__main__":
    sys.exit(main())
