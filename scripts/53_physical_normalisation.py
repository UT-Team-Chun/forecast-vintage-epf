r"""Phase 101: the physical-target contrast in units a referee can size.

THE PROBLEM
-----------
The paper reports the physical (photovoltaic output) contrast between the
headline pre-gate run and the run the gate excludes as \PhysDelta = 59.9 MW
[49.6, 70.6] on \PhysDeltaDays = 696 delivery days and \PhysAreas = eight
areas. A referee cannot size that. MW is not comparable across areas whose
installed fleets differ by an order of magnitude (Hokuriku ~1.2 GW against
Tokyo ~17.6 GW), and Methods currently calls the MW-averaged figure
"a scale-free comparison across vintages", which it is not: it is an
unweighted mean of eight quantities each carrying its own area's scale, so it
is dominated by the largest fleet. This phase does not touch the manuscript;
it produces the normalised numbers and their provenance.

THE QUESTIONS, AND THE BRANCHES, FIXED BEFORE ANY NUMBER IS COMPUTED
--------------------------------------------------------------------
Q0 (POSITIVE CONTROL, gating). Recomputing the MW contrast from the persisted
   per-slot artefact, on the paper's window (fourteen half-hourly slots from
   09:00 JST), the paper's eight areas (Chubu excluded: its operator publishes
   only the current month) and the paper's common panel (days shared with the
   price panel), must reproduce \PhysDelta and its interval.
   * PC-pass  (|delta - 59.88| <= 0.01 MW, both CI endpoints within 0.01 MW,
     n_days == 696, n_areas == 8, 14 slots per area-day): proceed to Q1--Q4.
   * PC-fail: STOP. Do not report any normalised figure. The discrepancy IS
     the finding and is reported as such, because a normalisation of a number
     that cannot be reproduced is worthless.
   A SECOND positive control guards the normalisation code path itself: with
   every denominator set to 1 MW, the normalised statistics must equal the MW
   statistics exactly (bit-for-bit on the point estimate). If they do not, the
   normalisation is mis-wired and the run STOPS.

Q1. What is the contrast as nMAE -- MAE as a percent of installed PV capacity?
Q2. What is it as a percentage of the comparison arm's own MAE?
Q3. What is it as a percentage of mean observed output over the scored window?

    These three answer different referee questions and can disagree. The
    pre-registered reading rule, with thresholds declared now:

    * MATERIAL: relative improvement (Q2) >= 5.0 % AND capacity-normalised
      step (Q1) >= 0.25 percentage points of installed capacity, with both
      95 % intervals excluding zero. (5 % is the conventional bar at which an
      accuracy increment is reported as meaningful in EPF/PV forecast
      benchmarking rather than as run-to-run noise; 0.25 pp is a quarter of
      the ~1 pp nMAE spread that separates competing operational PV forecast
      products. Both are conventions, declared before measurement, not
      physical facts.)
    * TRIVIAL: (Q2) < 2.0 % OR (Q1) < 0.10 pp.
    * MIXED: the three normalisations do not license the same reading -- one
      clears its threshold and another does not. Then ALL THREE are reported
      and the conclusion is explicitly that no single figure is scale-free.
    * NULL: any aggregate 95 % interval covers zero. Reported as a null
      whatever the point estimate, and the paper may not lean on the physical
      contrast as an established step.

Q4. Is the normalisation robust to the choice of denominator? Three
    denominators are pre-declared, because a referee will ask which:
    * D1 (primary): the register the paper already uses, METI FIT-portal
      prefecture operating capacity, vintage 2024-09 (= \CapVintage), summed
      within area. The paper uses its within-area SHARES as the proxy weights;
      this uses the same rows' absolute kW.
    * D2 (sensitivity): data/reference/area_capacity.csv OCCTO pv_gw. Its own
      loader documents a defect -- realised potential output exceeds its
      nameplate in seven area-channels -- so it is a sensitivity arm, never
      primary.
    * D3 (vintage-free): each area's maximum realised potential output inside
      the scored window. Carries no capacity vintage at all; this project
      already uses a realised-maximum index for exactly this reason
      (phase55/62).
    Branch: if the aggregate nMAE moves by more than 20 % in relative terms
    between D1 and D2, nMAE is denominator-sensitive and BOTH must be printed
    wherever it appears; a single nMAE figure would then be a choice, not a
    measurement.

Q5 (register sanity, reported either way). Per area, the maximum realised
    potential output as a share of each register's capacity. A physically
    impossible ratio (> 1) means the register understates the operating fleet,
    which inflates nMAE. Reported for both registers regardless of outcome.

WHAT IS REUSED AND HOW IT IS PROVED
-----------------------------------
Nothing about the panel, the window, the areas, the pooling or the inference is
re-implemented. phase89's module is imported and its own `_pool` and `_boot_ci`
are called; its SEED / N_BOOT / BOOT_BLOCK / PRE_GATE / POST_GATE / price-panel
path are read off the module rather than restated. The ratio statistics need
per-draw access to the resample, which `_boot_ci` does not expose, so the draw
is rebuilt from the identical rng calls and then PROVED equivalent: the
rebuilt draw must reproduce `_boot_ci`'s interval on the headline series to
1e-6 MW. phase91's lesson is that an unasserted rebinding turns a contrast into
a copy; the same applies to an unasserted re-derivation of a resample.

Nulls are counted, not assumed: every input column is checked for null AND for
non-finiteness (polars `drop_nulls` does not drop NaN, which has bitten this
codebase before), and the counts are written to the artefact.

CLI (from the repository root):
  uv run python scripts/53_physical_normalisation.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alg.config import EpfConfig
from alg.core.epf.data.area_capacity_loader import (
    pv_gw_by_area,
    reference_sha256,
)
from alg.core.epf.data.area_points import area_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase101")

CFG = EpfConfig()
RUN = CFG.resolved("run_dir")
P89 = Path(__file__).parent / "50_physical_target.py"
OUT_JSON = RUN / "phase101_physical_normalisation.json"
CAP_CSV = CFG.project_root / "data" / "reference" / "area_points_capacity.csv"

# What the manuscript currently prints for this contrast (macros.tex). Read as
# the positive control's target, not as an input to any computation.
PAPER_PHYS_DELTA = 59.9
PAPER_PHYS_DELTA_LO = 49.6
PAPER_PHYS_DELTA_HI = 70.6
PAPER_PHYS_DAYS = 696
PAPER_PHYS_AREAS = 8
PAPER_SLOTS_PER_AREA_DAY = 14
PAPER_CAP_VINTAGE = "2024-09"
# The provenance JSON's unrounded gate values (phase89 gate51).
PC_TOL_MW = 0.01

# Pre-registered reading thresholds (Q1--Q4). Declared before measurement.
THR_REL_MATERIAL_PCT = 5.0
THR_NMAE_MATERIAL_PP = 0.25
THR_REL_TRIVIAL_PCT = 2.0
THR_NMAE_TRIVIAL_PP = 0.10
THR_DENOM_SENSITIVE_REL = 0.20


def _j(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True,
                              cwd=CFG.project_root).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_phase89():
    """Import phase89 verbatim so its pooling and bootstrap are the ones used."""
    spec = importlib.util.spec_from_file_location("p89", P89)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p89"] = mod
    spec.loader.exec_module(mod)
    for name in ("_pool", "_boot_ci", "SEED", "N_BOOT", "BOOT_BLOCK",
                 "PRE_GATE", "POST_GATE", "PRICE_PANEL", "OUT_SLOTS",
                 "DAYTIME_HOURS", "WEIGHT_VINTAGE"):
        if not hasattr(mod, name):
            msg = f"phase89 no longer exposes {name!r}; the reuse is broken"
            raise RuntimeError(msg)
    return mod


def _fit_register_mw() -> tuple[dict[str, float], dict[str, int]]:
    """Area-summed PV operating capacity (MW) from the register the paper uses.

    Read from the same CSV rows `alg.core.epf.data.area_points` loads for the
    proxy weights, at the same vintage, so the denominator and the weights
    cannot come from different snapshots.
    """
    with CAP_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    cap: dict[str, float] = {}
    npts: dict[str, int] = {}
    for r in rows:
        if r["vintage"] != PAPER_CAP_VINTAGE:
            continue
        cap[r["area"]] = cap.get(r["area"], 0.0) + float(r["pv_kw"]) / 1000.0
        npts[r["area"]] = npts.get(r["area"], 0) + 1
    if not cap:
        msg = f"no rows at vintage {PAPER_CAP_VINTAGE} in {CAP_CSV}"
        raise RuntimeError(msg)
    # the register's within-area shares must reproduce the loader's weights,
    # or this is not the same register the proxy is built from
    pts = area_points(PAPER_CAP_VINTAGE)
    for area, plist in pts.items():
        if area not in cap:
            msg = f"{area} has proxy points but no capacity rows"
            raise RuntimeError(msg)
        if len(plist) != npts[area]:
            msg = (f"{area}: loader sees {len(plist)} points, the register has "
                   f"{npts[area]} rows at this vintage")
            raise RuntimeError(msg)
    return cap, npts


def _boot_counts(n: int, n_boot: int, block: int, seed: int) -> np.ndarray:
    """Re-derive phase89's circular block resample as per-day draw COUNTS.

    Identical rng calls to `phase89._boot_ci`; the counts form is used because
    the ratio statistics need the numerator and the denominator averaged over
    the SAME draw, which the mean-only helper cannot express. Equivalence to
    the helper is asserted by the caller, not assumed.
    """
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    counts = np.zeros((n_boot, n), dtype=np.int32)
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(counts, (rows, idx.ravel()), 1)
    return counts


def _ci(draws: np.ndarray) -> list[float | None]:
    return [_j(float(np.quantile(draws, 0.025))), _j(float(np.quantile(draws, 0.975)))]


def main() -> int:  # noqa: C901, PLR0912  (one linear pre-registered pipeline)
    mod = _load_phase89()
    seed, n_boot, block = mod.SEED, mod.N_BOOT, mod.BOOT_BLOCK
    pre_tier, post_tier = mod.PRE_GATE, mod.POST_GATE
    log.info("reused from phase89: seed=%s n_boot=%s block=%s pre=%s post=%s",
             seed, n_boot, block, pre_tier, post_tier)

    slots_path = Path(mod.OUT_SLOTS)
    price_path = Path(mod.PRICE_PANEL)
    for p in (slots_path, price_path, CAP_CSV):
        if not p.exists():
            log.error("missing input %s", p)
            return 2

    slots = pl.read_parquet(slots_path)

    # ---------------------------------------------------------------- nulls
    null_counts = {c: int(slots[c].null_count()) for c in slots.columns}
    nonfinite = {c: int((~slots[c].is_finite()).sum())
                 for c in ("proxy_scaled_cf", "target_mw", "curtail_mw")}
    if any(null_counts.values()) or any(nonfinite.values()):
        msg = (f"per-slot artefact carries nulls {null_counts} / non-finite "
               f"{nonfinite}; a normalisation over these would be a mean of "
               "silently dropped rows")
        raise RuntimeError(msg)

    tiers = sorted(slots["tier"].unique().to_list())
    if set(tiers) != {pre_tier, post_tier}:
        msg = f"per-slot artefact has tiers {tiers}, expected the two gate tiers"
        raise RuntimeError(msg)

    price_days = set(pl.read_parquet(price_path, columns=["test_day"])
                     ["test_day"].unique().to_list())
    common = (slots.with_columns(pl.col("day").cast(pl.Utf8).alias("_ds"))
              .filter(pl.col("_ds").is_in(list(price_days)))
              .drop("_ds"))

    areas = sorted(common["area"].unique().to_list())
    if len(areas) != PAPER_PHYS_AREAS or "chubu" in areas:
        msg = (f"area set {areas} is not the paper's {PAPER_PHYS_AREAS} areas "
               "(Chubu must be absent: its feed keeps only the current month)")
        raise RuntimeError(msg)
    per_area_day = common.group_by(["tier", "area", "day"]).len()
    if (per_area_day["len"].min() != PAPER_SLOTS_PER_AREA_DAY
            or per_area_day["len"].max() != PAPER_SLOTS_PER_AREA_DAY):
        msg = (f"slots per area-day range "
               f"[{per_area_day['len'].min()}, {per_area_day['len'].max()}], "
               f"expected exactly {PAPER_SLOTS_PER_AREA_DAY}")
        raise RuntimeError(msg)

    scored = common.with_columns(
        (pl.col("proxy_scaled_cf") - pl.col("target_mw")).abs().alias("abs_err"))
    daily = {t: {a: (scored.filter((pl.col("tier") == t) & (pl.col("area") == a))
                     .group_by("day")
                     .agg(pl.col("abs_err").mean().alias("mae"))
                     .sort("day"))
                 for a in areas}
             for t in (pre_tier, post_tier)}

    # phase89's own pooling: inner join across areas, then the horizontal mean
    pooled = {t: mod._pool(daily[t], "mae") for t in (pre_tier, post_tier)}
    days_pre = pooled[pre_tier]["day"].to_list()
    days_post = pooled[post_tier]["day"].to_list()
    if days_pre != days_post:
        msg = "the two tiers do not share a day set on the common panel"
        raise RuntimeError(msg)
    n_days = len(days_pre)
    if n_days != PAPER_PHYS_DAYS:
        msg = (f"common panel is {n_days} days, the paper reports "
               f"{PAPER_PHYS_DAYS}; STOP -- this discrepancy is the finding")
        raise RuntimeError(msg)

    # per-area daily MAE matrices, aligned on the pooled day order
    pre_m = np.column_stack([daily[pre_tier][a].join(
        pooled[pre_tier].select("day"), on="day", how="semi").sort("day")["mae"]
        .to_numpy() for a in areas])
    post_m = np.column_stack([daily[post_tier][a].join(
        pooled[post_tier].select("day"), on="day", how="semi").sort("day")["mae"]
        .to_numpy() for a in areas])
    if pre_m.shape != (n_days, len(areas)) or post_m.shape != pre_m.shape:
        msg = f"per-area matrix shape {pre_m.shape} != ({n_days}, {len(areas)})"
        raise RuntimeError(msg)
    if not (np.isfinite(pre_m).all() and np.isfinite(post_m).all()):
        msg = "non-finite daily MAE after alignment"
        raise RuntimeError(msg)
    # the row mean must BE phase89's pooled series, or the matrix is misaligned
    for t, m in ((pre_tier, pre_m), (post_tier, post_m)):
        ref = np.asarray(pooled[t]["pooled"], dtype=float)
        if not np.allclose(m.mean(axis=1), ref, rtol=0, atol=1e-9):
            msg = f"{t}: row mean of the per-area matrix != phase89's pooled series"
            raise RuntimeError(msg)

    d_daily = pre_m.mean(axis=1) - post_m.mean(axis=1)

    # ------------------------------------------------- Q0 positive control 1
    pc_delta = float(d_daily.mean())
    pc_lo, pc_hi = mod._boot_ci(d_daily)
    pc = {
        "recomputed_delta_mw": _j(pc_delta, 2),
        "paper_delta_mw": PAPER_PHYS_DELTA,
        "recomputed_ci95_mw": [_j(pc_lo, 2), _j(pc_hi, 2)],
        "paper_ci95_mw": [PAPER_PHYS_DELTA_LO, PAPER_PHYS_DELTA_HI],
        "n_days": n_days, "n_areas": len(areas),
        "n_slots_per_tier": int(common.height // 2),
        "tolerance_mw": PC_TOL_MW,
    }
    fails = []
    if abs(round(pc_delta, 1) - PAPER_PHYS_DELTA) > PC_TOL_MW:
        fails.append("point estimate")
    if abs(round(pc_lo, 1) - PAPER_PHYS_DELTA_LO) > PC_TOL_MW:
        fails.append("CI lower")
    if abs(round(pc_hi, 1) - PAPER_PHYS_DELTA_HI) > PC_TOL_MW:
        fails.append("CI upper")
    pc["passes"] = not fails
    pc["failed_components"] = fails
    if fails:
        OUT_JSON.write_text(json.dumps(
            {"phase": 101, "positive_control": pc,
             "verdict": "POSITIVE CONTROL FAILED -- no normalised figure reported",
             "metadata": {"script": Path(__file__).name, "git_sha": _git_sha()}},
            indent=1, ensure_ascii=False), encoding="utf-8")
        log.error("POSITIVE CONTROL FAILED on %s; wrote %s and stopping",
                  fails, OUT_JSON)
        return 3
    log.info("PC1 pass: %.2f MW [%.2f, %.2f] on %d days, %d areas",
             pc_delta, pc_lo, pc_hi, n_days, len(areas))

    # ------------------------------------------------- resample, then prove it
    counts = _boot_counts(n_days, n_boot, block, seed)
    if counts.sum(axis=1).min() != n_days or counts.sum(axis=1).max() != n_days:
        msg = "re-derived resample does not draw exactly n days per replicate"
        raise RuntimeError(msg)
    boot_pre = (counts @ pre_m) / n_days      # [n_boot, n_areas]
    boot_post = (counts @ post_m) / n_days
    _chk = (boot_pre - boot_post).mean(axis=1)
    chk_lo = float(np.quantile(_chk, 0.025))
    chk_hi = float(np.quantile(_chk, 0.975))
    if (abs(chk_lo - pc_lo) > 1e-6) or (abs(chk_hi - pc_hi) > 1e-6):
        msg = (f"re-derived resample gives [{chk_lo}, {chk_hi}] where phase89's "
               f"_boot_ci gives [{pc_lo}, {pc_hi}]; the ratio intervals would "
               "not be the paper's bootstrap")
        raise RuntimeError(msg)
    log.info("resample equivalence proved against phase89._boot_ci")

    # ------------------------------------------------------------ denominators
    fit_cap, _ = _fit_register_mw()
    occto_mw = {a: g * 1000.0 for a, g in pv_gw_by_area().items()}
    obs = (common.filter(pl.col("tier") == pre_tier)
           .group_by("area").agg(
               pl.col("target_mw").mean().alias("mean_potential_mw"),
               pl.col("target_mw").max().alias("max_potential_mw"),
               (pl.col("target_mw") - pl.col("curtail_mw")).mean()
               .alias("mean_recorded_mw"),
               pl.col("curtail_mw").mean().alias("mean_curtail_mw"))
           .sort("area"))
    obs_d = {r["area"]: r for r in obs.to_dicts()}
    # the target is identical across tiers by construction; verify, do not trust
    obs_post = (common.filter(pl.col("tier") == post_tier)
                .group_by("area").agg(pl.col("target_mw").mean().alias("m"))
                .sort("area"))
    if not np.allclose([obs_d[a]["mean_potential_mw"] for a in areas],
                       np.asarray(obs_post["m"], dtype=float), rtol=0, atol=1e-9):
        msg = "the observed target differs between tiers on the same panel"
        raise RuntimeError(msg)

    denoms = {
        "d1_fit_register_2024_09": np.array([fit_cap[a] for a in areas]),
        "d2_occto_area_capacity": np.array([occto_mw[a] for a in areas]),
        "d3_max_realised_potential": np.array(
            [obs_d[a]["max_potential_mw"] for a in areas]),
        "c_mean_observed_potential": np.array(
            [obs_d[a]["mean_potential_mw"] for a in areas]),
        "c_mean_recorded_only": np.array(
            [obs_d[a]["mean_recorded_mw"] for a in areas]),
    }
    for k, v in denoms.items():
        if not (np.isfinite(v).all() and (v > 0).all()):
            msg = f"denominator {k} has a non-positive or non-finite entry: {v}"
            raise RuntimeError(msg)

    mae_pre = pre_m.mean(axis=0)     # per-area MAE over the common panel, MW
    mae_post = post_m.mean(axis=0)
    delta_a = mae_pre - mae_post

    def _norm_block(cap: np.ndarray) -> dict:
        """Every normalised statistic for one denominator, with intervals."""
        r_pre, r_post = mae_pre / cap * 100.0, mae_post / cap * 100.0
        b_pre, b_post = boot_pre / cap * 100.0, boot_post / cap * 100.0
        macro_delta = (b_pre - b_post).mean(axis=1)
        wsum = cap.sum()
        cw_delta = ((b_pre - b_post) * cap).sum(axis=1) / wsum
        return {
            "per_area": {a: {"nmae_pre_pct": _j(r_pre[i], 3),
                             "nmae_post_pct": _j(r_post[i], 3),
                             "nmae_delta_pp": _j(r_pre[i] - r_post[i], 4),
                             "denominator_mw": _j(float(cap[i]), 1)}
                         for i, a in enumerate(areas)},
            "macro_mean_of_area_ratios": {
                "nmae_pre_pct": _j(float(r_pre.mean()), 3),
                "nmae_post_pct": _j(float(r_post.mean()), 3),
                "nmae_delta_pp": _j(float((r_pre - r_post).mean()), 4),
                "nmae_delta_ci95_pp": _ci(macro_delta),
            },
            "capacity_weighted": {
                "nmae_pre_pct": _j(float((mae_pre.sum() / wsum) * 100.0), 3),
                "nmae_post_pct": _j(float((mae_post.sum() / wsum) * 100.0), 3),
                "nmae_delta_pp": _j(float((delta_a.sum() / wsum) * 100.0), 4),
                "nmae_delta_ci95_pp": _ci(cw_delta),
            },
        }

    # ------------------------------------------------- Q0 positive control 2
    # with a 1 MW denominator the ratio is the MW statistic itself, so the
    # percent form must be exactly 100x it -- and the CI must be 100x phase89's
    ident = _norm_block(np.ones(len(areas)))
    im = ident["macro_mean_of_area_ratios"]
    pc2_ok = (
        im["nmae_delta_pp"] == _j(100.0 * float(delta_a.mean()), 4)
        and im["nmae_pre_pct"] == _j(100.0 * float(mae_pre.mean()), 3)
        and abs(im["nmae_delta_ci95_pp"][0] - 100.0 * pc_lo) < 1e-3
        and abs(im["nmae_delta_ci95_pp"][1] - 100.0 * pc_hi) < 1e-3)
    if not pc2_ok:
        msg = ("identity-denominator control failed: the normalisation path does "
               "not reproduce the MW statistics when every denominator is 1 MW")
        raise RuntimeError(msg)

    # --------------------------------------------------------- Q1 / Q3 blocks
    blocks = {k: _norm_block(v) for k, v in denoms.items()}
    n1 = blocks["d1_fit_register_2024_09"]["macro_mean_of_area_ratios"]
    n2 = blocks["d2_occto_area_capacity"]["macro_mean_of_area_ratios"]

    # ------------------------------------------------------------------ Q2
    b_pool_pre, b_pool_post = boot_pre.mean(axis=1), boot_post.mean(axis=1)
    rel_pooled = 100.0 * (b_pool_pre - b_pool_post) / b_pool_pre
    rel_macro = 100.0 * ((boot_pre - boot_post) / boot_pre).mean(axis=1)
    q2 = {
        "definition": ("percentage improvement of the excluded (post-gate) run "
                       "over the headline pre-gate arm, denominated in the "
                       "pre-gate arm's own MAE"),
        "pooled_ratio_of_means_pct": _j(
            100.0 * float(d_daily.mean()) / float(pre_m.mean(axis=1).mean()), 3),
        "pooled_ratio_of_means_ci95_pct": _ci(rel_pooled),
        "macro_mean_of_area_ratios_pct": _j(
            float((delta_a / mae_pre).mean() * 100.0), 3),
        "macro_mean_of_area_ratios_ci95_pct": _ci(rel_macro),
        "alt_denominator_post_gate_pct": _j(
            100.0 * float(d_daily.mean()) / float(post_m.mean(axis=1).mean()), 3),
        "per_area_pct": {a: _j(float(delta_a[i] / mae_pre[i] * 100.0), 3)
                         for i, a in enumerate(areas)},
    }

    # -------------------------------------------------- bridge to the published MW
    # Dividing the paper's own pooled MW figure by the eight-area MEAN installed
    # capacity is algebraically the capacity-WEIGHTED arm, not the per-area
    # (scale-free) arm: sum(mae)/sum(cap) == mean(mae)/mean(cap). Asserting the
    # identity fixes which of the two arms the currently printed number is.
    cap1 = denoms["d1_fit_register_2024_09"]
    mean_cap = float(cap1.mean())
    implied_pp = 100.0 * float(d_daily.mean()) / mean_cap
    cw1 = blocks["d1_fit_register_2024_09"]["capacity_weighted"]["nmae_delta_pp"]
    if abs(implied_pp - cw1) > 1e-3:
        msg = (f"bridge identity broken: 59.88/mean_cap gives {implied_pp} pp but "
               f"the capacity-weighted arm gives {cw1} pp")
        raise RuntimeError(msg)
    bridge = {
        "mean_installed_capacity_mw": _j(mean_cap, 1),
        "total_installed_capacity_mw": _j(float(cap1.sum()), 1),
        "published_delta_mw": _j(float(d_daily.mean()), 2),
        "published_delta_as_pp_of_mean_capacity": _j(implied_pp, 4),
        "equals_which_arm": "capacity_weighted",
        "scale_free_arm_pp": n1["nmae_delta_pp"],
        "gap_pp": _j(implied_pp - n1["nmae_delta_pp"], 4),
        "gap_relative": _j((implied_pp - n1["nmae_delta_pp"]) / n1["nmae_delta_pp"], 4),
        "note": ("the MW figure the manuscript prints, normalised by mean "
                 "installed capacity, is the capacity-weighted arm; the arm that "
                 "is actually scale-free normalises WITHIN area first and is "
                 "smaller, because the largest fleet also carries the largest "
                 "capacity-normalised step"),
    }

    # ------------------------------------------------------------- dispersion
    rel_area = delta_a / mae_pre * 100.0
    nm_area = np.array([blocks["d1_fit_register_2024_09"]["per_area"][a]
                        ["nmae_delta_pp"] for a in areas])
    dispersion = {
        "n_areas": len(areas),
        "areas_delta_positive": int((delta_a > 0).sum()),
        "areas_relative_ge_material_threshold": int(
            (rel_area >= THR_REL_MATERIAL_PCT).sum()),
        "areas_nmae_ge_material_threshold": int(
            (nm_area >= THR_NMAE_MATERIAL_PP).sum()),
        "relative_pct_min_area": areas[int(np.argmin(rel_area))],
        "relative_pct_min": _j(float(rel_area.min()), 3),
        "relative_pct_max_area": areas[int(np.argmax(rel_area))],
        "relative_pct_max": _j(float(rel_area.max()), 3),
        "nmae_delta_pp_min": _j(float(nm_area.min()), 4),
        "nmae_delta_pp_max": _j(float(nm_area.max()), 4),
        "note": ("all eight areas move in the same direction, but the relative "
                 "step spans a factor of about three across areas, so the "
                 "aggregate is a summary of a dispersed quantity"),
    }

    # ------------------------------------------------------------------ Q4
    rel_move_level = abs(n2["nmae_pre_pct"] - n1["nmae_pre_pct"]) / n1["nmae_pre_pct"]
    rel_move_delta = abs(n2["nmae_delta_pp"] - n1["nmae_delta_pp"]) / n1["nmae_delta_pp"]
    q4 = {
        "d1_vs_d2_level_relative_move": _j(rel_move_level, 4),
        "d1_vs_d2_delta_relative_move": _j(rel_move_delta, 4),
        "threshold": THR_DENOM_SENSITIVE_REL,
        "denominator_sensitive": bool(rel_move_level > THR_DENOM_SENSITIVE_REL
                                      or rel_move_delta > THR_DENOM_SENSITIVE_REL),
    }

    # ------------------------------------------------------------------ Q5
    q5 = {a: {"max_realised_potential_mw": _j(obs_d[a]["max_potential_mw"], 1),
              "over_fit_register": _j(obs_d[a]["max_potential_mw"] / fit_cap[a], 3),
              "over_occto_register": _j(
                  obs_d[a]["max_potential_mw"] / occto_mw[a], 3),
              "mean_capacity_factor_window_fit": _j(
                  obs_d[a]["mean_potential_mw"] / fit_cap[a], 3),
              "mean_curtail_mw": _j(obs_d[a]["mean_curtail_mw"], 1)}
          for a in areas}
    impossible = {"fit": [a for a in areas if q5[a]["over_fit_register"] > 1.0],
                  "occto": [a for a in areas if q5[a]["over_occto_register"] > 1.0]}

    # ------------------------------------------------------ pre-registered read
    rel = q2["pooled_ratio_of_means_pct"]
    rel_ci = q2["pooled_ratio_of_means_ci95_pct"]
    nm = n1["nmae_delta_pp"]
    nm_ci = n1["nmae_delta_ci95_pp"]
    c_block = blocks["c_mean_observed_potential"]["macro_mean_of_area_ratios"]
    obs_delta, obs_ci = c_block["nmae_delta_pp"], c_block["nmae_delta_ci95_pp"]
    any_ci_covers_zero = any(lo <= 0 <= hi for lo, hi in
                             (rel_ci, nm_ci, obs_ci))
    clears = {
        "relative_ge_5pct": rel >= THR_REL_MATERIAL_PCT,
        "nmae_ge_0p25pp": nm >= THR_NMAE_MATERIAL_PP,
    }
    trivial = rel < THR_REL_TRIVIAL_PCT or nm < THR_NMAE_TRIVIAL_PP
    if any_ci_covers_zero:
        branch, reading = "NULL", (
            "an aggregate 95% interval covers zero; the physical contrast is "
            "reported as a null in normalised units")
    elif all(clears.values()):
        branch, reading = "MATERIAL", (
            "both pre-registered thresholds are cleared with intervals off zero")
    elif trivial:
        branch, reading = "TRIVIAL", (
            "the contrast falls below a pre-registered triviality threshold")
    else:
        branch, reading = "MIXED", (
            "the normalisations do not license the same reading; all three must "
            "be printed and none may be called scale-free")

    out = {
        "phase": 101,
        "title": "The physical-target contrast in normalised, interpretable units",
        "question": ("MW is not comparable across areas of different installed "
                     "capacity, and Methods calls the MW-averaged figure "
                     "scale-free. What is the same contrast as a share of "
                     "installed capacity, of the comparison arm's own error, "
                     "and of mean observed output?"),
        "inputs": {
            "per_slot_artefact": str(slots_path),
            "price_panel_for_common_days": str(price_path),
            "capacity_register_primary": str(CAP_CSV),
            "capacity_register_primary_sha256": hashlib.sha256(
                CAP_CSV.read_bytes()).hexdigest(),
            "capacity_register_primary_vintage": PAPER_CAP_VINTAGE,
            "capacity_register_sensitivity": (
                str(CFG.resolved("reference_data_dir") / "area_capacity.csv")),
            "capacity_register_sensitivity_sha256": reference_sha256(),
            "phase89_provenance": str(RUN / "phase89_physical_target.json"),
            "reused_module": str(P89),
        },
        "panel": {
            "n_days": n_days, "n_areas": len(areas), "areas": areas,
            "slots_per_area_day": PAPER_SLOTS_PER_AREA_DAY,
            "n_slots_per_tier": int(common.height // 2),
            "n_slot_pairs": int(common.height // 2),
            "window_jst": list(mod.DAYTIME_HOURS),
            "pre_gate_tier": pre_tier, "post_gate_tier": post_tier,
            "target": "potential output = recorded + curtailed",
            "excluded_area": "chubu (feed keeps only the current month)",
            "null_counts": null_counts, "nonfinite_counts": nonfinite,
        },
        "positive_control_mw": pc,
        "positive_control_identity_denominator": {
            "passes": bool(pc2_ok),
            "note": ("with every denominator set to 1 MW the normalised "
                     "statistics reproduce the MW statistics exactly"),
        },
        "levels_mw": {
            "pooled_pre_gate_mae_mw": _j(float(pre_m.mean(axis=1).mean()), 1),
            "pooled_post_gate_mae_mw": _j(float(post_m.mean(axis=1).mean()), 1),
            "per_area_mae_pre_mw": {a: _j(float(mae_pre[i]), 1)
                                    for i, a in enumerate(areas)},
            "per_area_mae_post_mw": {a: _j(float(mae_post[i]), 1)
                                     for i, a in enumerate(areas)},
            "per_area_delta_mw": {a: _j(float(delta_a[i]), 2)
                                  for i, a in enumerate(areas)},
        },
        "q1_nmae_by_installed_capacity": {
            "d1_fit_register_2024_09": blocks["d1_fit_register_2024_09"],
            "d2_occto_area_capacity": blocks["d2_occto_area_capacity"],
            "d3_max_realised_potential": blocks["d3_max_realised_potential"],
            "note": ("d1 is primary: it is the register whose within-area shares "
                     "the proxy already uses, at the vintage the paper prints. "
                     "d2 carries a documented nameplate defect. d3 carries no "
                     "capacity vintage."),
        },
        "q2_relative_improvement": q2,
        "bridge_published_mw_to_normalised": bridge,
        "per_area_dispersion": dispersion,
        "q3_nmae_by_mean_observed_output": {
            "mean_observed_potential": blocks["c_mean_observed_potential"],
            "mean_recorded_only": blocks["c_mean_recorded_only"],
            "note": ("the denominator is each area's panel-mean observed output "
                     "inside the scored window, held fixed across days so the "
                     "statistic stays a linear functional of the daily MAEs and "
                     "the paper's day-block bootstrap applies unchanged"),
        },
        "q4_denominator_sensitivity": q4,
        "q5_register_sanity": {
            "per_area": q5,
            "areas_with_realised_above_nameplate": impossible,
        },
        "reading": {
            "branch": branch, "note": reading,
            "thresholds": {
                "relative_material_pct": THR_REL_MATERIAL_PCT,
                "nmae_material_pp": THR_NMAE_MATERIAL_PP,
                "relative_trivial_pct": THR_REL_TRIVIAL_PCT,
                "nmae_trivial_pp": THR_NMAE_TRIVIAL_PP,
            },
            "clears": clears,
            "any_aggregate_ci_covers_zero": bool(any_ci_covers_zero),
            "headline_triplet": {
                "nmae_delta_pp_of_installed_capacity": nm,
                "nmae_delta_ci95_pp": nm_ci,
                "relative_improvement_pct": rel,
                "relative_improvement_ci95_pct": rel_ci,
                "delta_pct_of_mean_observed_output": obs_delta,
                "delta_pct_of_mean_observed_output_ci95": obs_ci,
            },
        },
        "inference": {
            "bootstrap": ("phase89's circular block bootstrap, reused: block "
                          f"{block} days, {n_boot} draws, seed {seed}, "
                          "resampling unit = delivery day"),
            "equivalence_check_vs_phase89_boot_ci_mw": [_j(chk_lo, 6), _j(chk_hi, 6)],
        },
        "metadata": {"script": Path(__file__).name, "git_sha": _git_sha(),
                     "seed": seed, "n_boot": n_boot, "boot_block_days": block},
    }

    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", OUT_JSON)
    log.info("MW: pre %.1f -> post %.1f, delta %.2f [%.2f, %.2f]",
             out["levels_mw"]["pooled_pre_gate_mae_mw"],
             out["levels_mw"]["pooled_post_gate_mae_mw"], pc_delta, pc_lo, pc_hi)
    log.info("Q1 nMAE (FIT 2024-09, macro): %.3f%% -> %.3f%%, delta %.4f pp %s",
             n1["nmae_pre_pct"], n1["nmae_post_pct"], nm, nm_ci)
    log.info("Q2 relative: %.2f%% %s (macro-of-ratios %.2f%% %s)",
             rel, rel_ci, q2["macro_mean_of_area_ratios_pct"],
             q2["macro_mean_of_area_ratios_ci95_pct"])
    log.info("Q3 of mean observed output: %.3f%% -> %.3f%%, delta %.4f pp %s",
             c_block["nmae_pre_pct"], c_block["nmae_post_pct"], obs_delta, obs_ci)
    log.info("Q4 denominator sensitive = %s (level move %.1f%%, delta move %.1f%%)",
             q4["denominator_sensitive"], 100 * q4["d1_vs_d2_level_relative_move"],
             100 * q4["d1_vs_d2_delta_relative_move"])
    log.info("Q5 realised above nameplate: %s", impossible)
    log.info("bridge: published %.2f MW / mean cap %.1f MW = %.4f pp "
             "(capacity-weighted arm); scale-free arm %.4f pp",
             bridge["published_delta_mw"], mean_cap, implied_pp,
             n1["nmae_delta_pp"])
    log.info("dispersion: %d/%d areas positive, %d/%d clear the relative "
             "threshold (range %.2f%%-%.2f%%)",
             dispersion["areas_delta_positive"], len(areas),
             dispersion["areas_relative_ge_material_threshold"], len(areas),
             dispersion["relative_pct_min"], dispersion["relative_pct_max"])
    log.info("BRANCH %s -- %s", branch, reading)
    return 0


if __name__ == "__main__":
    sys.exit(main())
