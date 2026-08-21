r"""Drift guard: verify paper headline macros against regenerated provenance.

The manuscript ships the submission snapshot of every headline
``\newcommand``; each value is produced by an analysis script and persisted as
a provenance JSON. This tool re-reads both and fails (exit 1) if any macro has
drifted from its provenance — the automated check that would have caught the
PV/peak capacity-vintage drift the research log documents.

This file is shared verbatim between two repositories and resolves its own
layout: the research monorepo (macros at ``docs/paper/shared/macros.tex``,
provenance under ``data/runs/``, run as
``uv run python verify_macros.py`` (from its own directory) from
``backend/packages/alg``) and the public replication package (macros at
``docs/macros.tex``, frozen provenance under ``data/provenance/``, run as
``uv run python scripts/verify_macros.py`` or ``make verify-paper`` from the
repository root). Provenance is resolved PER FILE, preferring a freshly
regenerated ``data/runs/<name>`` over the frozen ``data/provenance/<name>``,
so a reader who re-runs part of the chain verifies their own numbers for the
phases they re-ran and the frozen record for the rest.

The macro -> provenance map is a typed list of :class:`MacroCheck` (no scattered
hardcoded paths); to add a macro, append one typed entry.

WHAT --strict MEANS (since 2026-08-13)
--------------------------------------
Without it: every binding, every retired term, every hygiene check. This is the
drafting gate and it must be green at all times.

With it: the above, plus pending bindings must be zero AND no exempt placeholder
may still be a placeholder. The replication DOI and the submission commit are
exempt from provenance binding because no analysis produces them, and their
exemptions promised the placeholder would be "visible in the rendered PDF" --
true for a human, false for a gate that exits 0 while the PDF prints
"zenodo.XXXXXXX". So --strict now fails until the deposit is minted and the tag
exists. That is deliberate: it is the difference between "the numbers are right"
and "this is submittable".
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from alg.config import EpfConfig

_SIGNED_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?")  # decimals and bare integers (e.g. "83\%")


class MacroCheck(BaseModel):
    """One macro <-> provenance assertion."""

    model_config = ConfigDict(frozen=True)

    macro: str  # \newcommand name without backslash, e.g. "Spearman"
    provenance: str  # path relative to EpfConfig.run_dir
    json_path: tuple[str | int, ...]  # nested path into the provenance JSON
    tol: float = 5e-3  # absolute tolerance for the float comparison
    scale: float = 1.0  # provenance * scale before compare (e.g. 100 for a %-macro)
    note: str = ""
    # A macro that prints a RATIO of two provenance leaves -- a share, a relief
    # percentage -- otherwise has to be divided by hand and re-typed into
    # macros.tex, where nothing can see it again. That is exactly how
    # \CnpeBalCostPreM came to print 1.1 against a provenance 1.0597, which in
    # turn made the printed ratio 4.2 unreproducible from the printed numbers.
    # Give the guard the division instead of the quotient.
    denominator_json_path: tuple[str | int, ...] | None = None
    complement: bool = False  # compare scale*(1 - num/den) instead of scale*num/den
    # A macro that prints a DIFFERENCE of two provenance leaves -- "the days
    # that survive a restriction" is a total minus an exclusion count -- has
    # the same problem as a ratio: subtract by hand and the arithmetic is
    # unverifiable. Give the guard the subtraction.
    subtract_json_path: tuple[str | int, ...] | None = None
    # A macro that prints "the LOWEST per-area improvement" cannot be bound by
    # path, because the path names whichever area happens to be lowest today --
    # bind it that way and a re-run that reorders the areas silently keeps
    # passing. Point json_path at the CONTAINER instead and set aggregate to
    # "min"/"max"; leaf_key selects the field inside each entry. This is how
    # \HeadlinePerAreaLo/Hi stopped being permanent TODOs.
    aggregate: str | None = None       # None | "min" | "max" | "count"
    leaf_key: str | None = None        # field to read inside each container entry
    # An aggregate binding protects the VALUE and drops the container key -- but
    # the sentence around the macro usually names the key ("from Hokuriku's 164
    # MW to Tokyo's 2,353 MW"). Those names are hand-typed prose no check reads,
    # so a re-run in which the extremum moves to another area within tolerance
    # keeps the check green on a sentence that credits the wrong operator. Name
    # the key you expect and the guard asserts it.
    aggregate_key: str | None = None
    # A check that belongs to the SETTLEMENT paper, not the forecast-vintage
    # one. The manuscripts split in 2026-08. These macros stay defined and
    # verified HERE, in the research repository where their provenance and the
    # scripts that produced it both live; the forecast-vintage replication
    # package ships neither, because shipping a result the paper no longer
    # makes would advertise an analysis a reader cannot check. Marking them
    # lets one guard serve both checkouts without either lying: where the
    # provenance is present it is verified as always, and where it is absent
    # the check is SKIPPED AND REPORTED. An unmarked check whose provenance is
    # missing remains a hard failure, so this cannot become a way to hide drift.
    paper_b: bool = False


class MacroExempt(BaseModel):
    """A numeric macro that legitimately has no provenance value.

    ``source`` and ``reason`` are mandatory: a macro leaves the guard's reach
    only with a stated, reviewable justification. Exempting by shrugging is the
    failure mode this list exists to prevent.
    """

    model_config = ConfigDict(frozen=True)

    macro: str
    kind: str  # "label" | "literature_value" | "protocol_constant" | "institutional_constant"
    source: str  # bib key, spec name, or "<script>.py:<line>" -- never empty
    reason: str


class MacroPending(BaseModel):
    """A numeric macro that SHOULD be bound to provenance but is not yet.

    Distinct from :class:`MacroExempt` on purpose. Exempt means "no provenance
    value exists"; pending means "one exists and nobody has wired it up". Keeping
    them apart is what stops the exempt list from becoming a dumping ground.

    The guard prints these loudly and counts them, and ``--strict`` (used by the
    final submission gate) fails while any remain. A numeric macro in neither
    CHECKS, WORD_CHECKS, EXEMPT nor here fails immediately, so the set of
    unguarded numbers can only ever shrink.
    """

    model_config = ConfigDict(frozen=True)

    macro: str
    owner_phase: str  # the phase whose provenance holds the value
    reason: str


# Typed map: paper headline macros -> regenerated provenance values.
CHECKS: list[MacroCheck] = [
    MacroCheck(
        macro="Spearman",
        provenance="phase25.7_pv_peak_summary.json",
        json_path=("univariate_floor_rate", "spearman_rho"),
        note="cross-area FLOOR Spearman (headline)",
    ),
    MacroCheck(
        macro="SpearmanP",
        provenance="phase25.7_pv_peak_summary.json",
        json_path=("univariate_floor_rate", "spearman_p"),
        note="cross-area FLOOR Spearman p-value (headline)",
    ),
    MacroCheck(
        macro="Rsqpv",
        provenance="phase25.7_pv_peak_summary.json",
        json_path=("univariate_pv_peak", "r_squared"),
        note="univariate PV/peak R^2 (AIC-best alternative)",
    ),
    MacroCheck(
        macro="PanelBeta",
        provenance="phase23-panel-daily/cluster_bootstrap.json",
        json_path=("spec1_area_main", "coefs", "floor_rate", "observed"),
        note="area-fold panel FLOOR coefficient",
    ),
    MacroCheck(
        macro="PanelBootCI",
        provenance="phase23-panel-daily/cluster_bootstrap.json",
        json_path=("spec1_area_main", "coefs", "floor_rate", "ci90_lo"),
        note="cluster+block bootstrap 90% CI lower",
    ),
    MacroCheck(
        macro="PanelBootCIninetyFive",
        provenance="phase23-panel-daily/cluster_bootstrap.json",
        json_path=("spec1_area_main", "coefs", "floor_rate", "ci95_lo"),
        note="cluster+block bootstrap 95% CI lower",
    ),
    MacroCheck(
        macro="SpearmanPre",
        provenance="phase25_17_floor_pre_period/results.json",
        json_path=("specifications", 1, "cross_area_floor_vs_delta_mae", "spearman_rho"),
        note="pre-evaluation FLOOR Spearman (spec[1] = pre_evaluation_2022_to_2024_09_14)",
    ),
    MacroCheck(
        macro="CurtailFloorSpearman",
        provenance="phase25.9_floor_probit_summary.json",
        json_path=("cross_area_spearman", "curtail_hours_vs_floor_hours", "rho"),
        note="8-area cross-area curtailment-hours vs FLOOR-hours Spearman (incl. Shikoku+Kansai)",
    ),
    MacroCheck(
        macro="CurtailFloorSpearmanP",
        provenance="phase25.9_floor_probit_summary.json",
        json_path=("cross_area_spearman", "curtail_hours_vs_floor_hours", "p_two_sided"),
        note="8-area cross-area curtailment-vs-FLOOR Spearman p-value",
    ),
    # ----- Round-27 / Phase 26: operational-forecast reframe -----
    MacroCheck(
        macro="WithinDayBeta",
        provenance="phase26.0b_treatment_isolation.json",
        json_path=("beta_triple_histfc_forecast_admissible_accurate", "beta_observed"),
        note="within-day triple under admissible operational-forecast moderator",
    ),
    MacroCheck(
        macro="WithinDayP",
        provenance="phase26.0b_treatment_isolation.json",
        json_path=(
            "beta_triple_histfc_forecast_admissible_accurate",
            "p_wild_cluster_bootstrap_two_sided",
        ),
        note="within-day triple wild-cluster p (admissible moderator)",
    ),
    MacroCheck(
        macro="WithinDayBetaContemp",
        provenance="phase26.0b_treatment_isolation.json",
        json_path=("beta_triple_contemporaneous_truth", "beta_observed"),
        note="within-day triple under contemporaneous moderator (reproduces committed -0.33)",
    ),
    # ----- Round-30 / Phase 28.1: within-day FE-saturation collapse (honest caveat) -----
    MacroCheck(
        macro="WithinDayFEbeta",
        provenance="phase28.1_within_day_fe_robustness.json",
        json_path=(
            "histfc_moderator_no_controls",
            "fe_specs_twoway_cluster",
            "spec1_area_date_hour",
            "beta_triple",
        ),
        note="within-day triple attenuates to ~0 under saturated area+date+hour FE",
    ),
    MacroCheck(
        macro="WithinDayFEp",
        provenance="phase28.1_within_day_fe_robustness.json",
        json_path=(
            "histfc_moderator_no_controls",
            "fe_specs_twoway_cluster",
            "spec1_area_date_hour",
            "p_asymptotic_twoway",
        ),
        note="within-day triple two-way-clustered p under saturated FE (n.s.)",
    ),
    # ----- Round-30 / Phase 29: cost-ratio grounding (D-lite) -----
    MacroCheck(
        macro="FloorWedgeShikoku",
        provenance="phase29_cost_ratio_grounding.json",
        json_path=("cited_macros", "FloorWedgeShikoku"),
        note="Shikoku floor-episode system-minus-area wedge (JPY/kWh)",
    ),
    MacroCheck(
        macro="FloorWedgeKyushu",
        provenance="phase29_cost_ratio_grounding.json",
        json_path=("cited_macros", "FloorWedgeKyushu"),
        note="Kyushu floor-episode system-minus-area wedge (JPY/kWh)",
    ),
    MacroCheck(
        macro="OpsCrossArea",
        provenance="phase26.1_pv_surplus_identification.json",
        json_path=("cross_area_gradient", "histfc", "spearman_vs_pv_peak", "rho"),
        note="operational-forecast cross-area gradient (washout)",
    ),
    MacroCheck(
        macro="OpsCrossAreaP",
        provenance="phase26.1_pv_surplus_identification.json",
        json_path=("cross_area_gradient", "histfc", "spearman_vs_pv_peak", "p"),
        note="operational-forecast cross-area gradient p-value (n.s.)",
    ),
    MacroCheck(
        macro="ProxyCrossAreaClean",
        provenance="phase26.1_pv_surplus_identification.json",
        json_path=("cross_area_gradient", "era5", "spearman_vs_pv_peak", "rho"),
        note="ERA5 proxy cross-area gradient on the clean head-to-head sample",
    ),
    MacroCheck(
        macro="OpsFloorFone",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_weather", "floor_f1"),
        note="operational-forecast floor-event F1",
    ),
    MacroCheck(
        macro="PriceOnlyFloorFone",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_price_only", "floor_f1"),
        note="price-only floor-event F1",
    ),
    MacroCheck(
        macro="OpsFloorAUC",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_weather", "roc_auc_floor"),
        note="operational-forecast floor-event ROC-AUC",
    ),
    # ----- Round-28 / Phase 27: deepened decision value + 4-info-set + skill -----
    MacroCheck(
        macro="FloorBaseRate",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_weather", "floor_base_rate"),
        scale=100.0,
        tol=0.05,
        note="FLOOR base rate (%) -- explains F1 vs AUC",
    ),
    MacroCheck(
        macro="OpsFloorPRAUC",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_weather", "pr_auc_floor"),
        note="operational floor PR-AUC",
    ),
    MacroCheck(
        macro="PriceOnlyFloorPRAUC",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "classification_price_only", "pr_auc_floor"),
        note="price-only floor PR-AUC",
    ),
    MacroCheck(
        macro="OracleFloorPRAUC",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "oracle", "classification_weather", "pr_auc_floor"),
        note="realised-weather oracle floor PR-AUC",
    ),
    MacroCheck(
        macro="OracleFloorFone",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "oracle", "classification_weather", "floor_f1"),
        note="realised-weather oracle floor F1 (operational ~= oracle)",
    ),
    MacroCheck(
        macro="OpsFloorBrier",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "calibration_weather", "brier"),
        note="operational floor Brier (cross-fitted isotonic)",
    ),
    MacroCheck(
        macro="PriceOnlyFloorBrier",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "calibration_price_only", "brier"),
        note="price-only floor Brier",
    ),
    MacroCheck(
        macro="OpsFloorFoneCF",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "threshold_opt_weather", "f1_crossfitted_threshold"),
        note="operational floor F1 at cross-fitted decision threshold",
    ),
    MacroCheck(
        macro="PriceOnlyFloorFoneCF",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "threshold_opt_price_only", "f1_crossfitted_threshold"),
        note="price-only floor F1 at cross-fitted threshold",
    ),
    MacroCheck(
        macro="OpsCurtailCoincidence",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "weather_predicted_floor_curtail_coincidence"),
        scale=100.0,
        tol=0.6,
        note="operational-predicted floors coincident with curtailment (%)",
    ),
    MacroCheck(
        macro="PriceOnlyCurtailCoincidence",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "histfc", "price_only_predicted_floor_curtail_coincidence"),
        scale=100.0,
        tol=0.6,
        note="price-only-predicted floors coincident with curtailment (%)",
    ),
    MacroCheck(
        macro="ImpProxy",
        provenance="phase27.1_oracle_tier.json",
        json_path=("mean_improvement_yen_kwh", "era5"),
        note="ERA5 proxy mean weather MAE improvement (yen/kWh)",
    ),
    MacroCheck(
        macro="ImpOps",
        provenance="phase27.1_oracle_tier.json",
        json_path=("mean_improvement_yen_kwh", "histfc"),
        note="operational mean weather MAE improvement (yen/kWh)",
    ),
    MacroCheck(
        macro="ImpOracle",
        provenance="phase27.1_oracle_tier.json",
        json_path=("mean_improvement_yen_kwh", "oracle"),
        note="realised-weather oracle mean MAE improvement (yen/kWh)",
    ),
    MacroCheck(
        macro="SkillProxy",
        provenance="phase27.0_derisk_gate.json",
        json_path=("probe_s_skill_axis", "skill_lag48_corr"),
        note="ERA5 lag-48 proxy forecast skill (midday corr)",
    ),
    MacroCheck(
        macro="SkillHistfc",
        provenance="phase27.0_derisk_gate.json",
        json_path=("probe_s_skill_axis", "skill_histfc_corr"),
        note="operational JMA-MSM forecast skill (midday corr)",
    ),
    MacroCheck(
        macro="CurtailActiveBeta",
        provenance="phase27.0_derisk_gate.json",
        json_path=("probe_b_threshold_gate", "curtail_histfc_intensity", "beta_observed"),
        tol=0.01,
        note="operational within-day intensity, curtail-active (actual-crossing) slots",
    ),
    MacroCheck(
        macro="SpearmanPVpeak",
        provenance="phase25.7_pv_peak_summary.json",
        json_path=("univariate_pv_peak", "spearman_rho"),
        note="cross-area PV/peak vs weather Delta-MAE (AIC-best antecedent)",
    ),
    MacroCheck(
        macro="ProxyFloorFone",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "era5", "classification_weather", "floor_f1"),
        note="ERA5 proxy floor-event F1 (proxy adds ~no decision value)",
    ),
    MacroCheck(
        macro="ProxyFloorPRAUC",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "era5", "classification_weather", "pr_auc_floor"),
        tol=6e-3,
        note="ERA5 proxy floor-event PR-AUC",
    ),
    MacroCheck(
        macro="ProxyFloorBrier",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "era5", "calibration_weather", "brier"),
        note="ERA5 proxy floor-event Brier",
    ),
    MacroCheck(
        macro="ProxyFloorFoneCF",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "era5", "threshold_opt_weather", "f1_crossfitted_threshold"),
        note="ERA5 proxy floor F1 at cross-fitted threshold",
    ),
    MacroCheck(
        macro="OracleFloorBrier",
        provenance="phase26.2_floor_decision_loss.json",
        json_path=("pooled", "oracle", "calibration_weather", "brier"),
        note="realised-weather oracle floor-event Brier",
    ),
    # ----- Phase 27.5: LightGBM model-class robustness of the skill-scarcity pivot -----
    MacroCheck(
        macro="LgbmProxyImp",
        provenance="phase27.5_lgbm_identification.json",
        json_path=("cross_area_gradient", "era5", "mean_improvement"),
        note="LightGBM ERA5 proxy mean MAE improvement (vs XGB 0.034)",
    ),
    MacroCheck(
        macro="LgbmOpsImp",
        provenance="phase27.5_lgbm_identification.json",
        json_path=("cross_area_gradient", "histfc", "mean_improvement"),
        note="LightGBM operational mean MAE improvement (vs XGB 0.239)",
    ),
    MacroCheck(
        macro="LgbmProxyConc",
        provenance="phase27.5_lgbm_identification.json",
        json_path=("cross_area_gradient", "era5", "spearman_vs_pv_peak", "rho"),
        note="LightGBM proxy cross-area gradient vs PV/peak (n.s. at n=8)",
    ),
    MacroCheck(
        macro="LgbmOpsConc",
        provenance="phase27.5_lgbm_identification.json",
        json_path=("cross_area_gradient", "histfc", "spearman_vs_pv_peak", "rho"),
        note="LightGBM operational cross-area gradient (washout, n.s.)",
    ),
    # ----- Phase 27.5: CatBoost (3rd model class) robustness of the skill-scarcity pivot -----
    MacroCheck(
        macro="CatProxyImp",
        provenance="phase27.5_catboost_identification.json",
        json_path=("cross_area_gradient", "era5", "mean_improvement"),
        note="CatBoost ERA5 proxy mean MAE improvement (vs XGB 0.034)",
    ),
    MacroCheck(
        macro="CatOpsImp",
        provenance="phase27.5_catboost_identification.json",
        json_path=("cross_area_gradient", "histfc", "mean_improvement"),
        note="CatBoost operational mean MAE improvement (vs XGB 0.239)",
    ),
    MacroCheck(
        macro="CatProxyConc",
        provenance="phase27.5_catboost_identification.json",
        json_path=("cross_area_gradient", "era5", "spearman_vs_pv_peak", "rho"),
        note="CatBoost proxy cross-area gradient vs PV/peak (p=0.058, n.s. at n=8)",
    ),
    MacroCheck(
        macro="CatOpsConc",
        provenance="phase27.5_catboost_identification.json",
        json_path=("cross_area_gradient", "histfc", "spearman_vs_pv_peak", "rho"),
        note="CatBoost operational cross-area gradient (washout, n.s.)",
    ),
    # ----- Round-31 / Phase 31b: exact-issuance JMA-GPV head-to-head (review #1) -----
    MacroCheck(
        macro="GpvExactImp",
        provenance="phase31b_exact_issuance_headtohead.json",
        json_path=("mean_improvement_yen_kwh", "gpv_exact_issuance"),
        note="exact-issuance GPV mean MAE improvement (daytime same-sample)",
    ),
    MacroCheck(
        macro="HistfcDayImp",
        provenance="phase31b_exact_issuance_headtohead.json",
        json_path=("mean_improvement_yen_kwh", "histfc_reconstruction"),
        note="reconstruction mean improvement on the identical daytime sample (gpv ~= histfc)",
    ),
    MacroCheck(
        macro="GpvExactConc",
        provenance="phase31b_exact_issuance_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "rho"),
        note="exact-issuance GPV cross-area gradient vs PV/peak (washout, n.s.)",
    ),
    MacroCheck(
        macro="GpvExactConcP",
        provenance="phase31b_exact_issuance_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "p"),
        note="exact-issuance GPV cross-area gradient p-value (n.s.)",
    ),
    # ----- Round-33 Phase A: ALL-HOURS exact-issuance head-to-head (headline) -----
    MacroCheck(
        macro="ProxyAllHoursImp",
        provenance="phase35_allhours_headtohead.json",
        json_path=("mean_improvement_yen_kwh", "era5_proxy"),
        note="ERA5 proxy mean MAE improvement (all-hours same-sample)",
    ),
    MacroCheck(
        macro="HistfcAllHoursImp",
        provenance="phase35_allhours_headtohead.json",
        json_path=("mean_improvement_yen_kwh", "histfc_reconstruction"),
        note="operational reconstruction mean improvement (all-hours)",
    ),
    MacroCheck(
        macro="GpvAllHoursImp",
        provenance="phase35_allhours_headtohead.json",
        json_path=("mean_improvement_yen_kwh", "gpv_exact_issuance"),
        note="exact-issuance GPV mean MAE improvement (all-hours headline)",
    ),
    MacroCheck(
        macro="ProxyAllHoursConc",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "era5_proxy", "rho"),
        note="proxy cross-area gradient vs PV/peak, all-hours (significant +0.83)",
    ),
    MacroCheck(
        macro="ProxyAllHoursConcP",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "era5_proxy", "p"),
        note="proxy cross-area gradient p-value, all-hours (significant)",
    ),
    MacroCheck(
        macro="HistfcAllHoursConc",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "histfc_reconstruction", "rho"),
        note="operational reconstruction gradient, all-hours (washout, n.s.)",
    ),
    MacroCheck(
        macro="HistfcAllHoursConcP",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "histfc_reconstruction", "p"),
        note="operational reconstruction gradient p-value, all-hours (n.s.)",
    ),
    MacroCheck(
        macro="GpvAllHoursConc",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "rho"),
        note="exact-issuance GPV gradient vs PV/peak, all-hours (washout, n.s.)",
    ),
    MacroCheck(
        macro="GpvAllHoursConcP",
        provenance="phase35_allhours_headtohead.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "p"),
        note="exact-issuance GPV gradient p-value, all-hours (n.s.)",
    ),
    # ----- Round-37 Phase 2: 9-area point-forecast head-to-head (adds Chubu; review #17) -----
    MacroCheck(
        macro="ProxyAllHoursConcNine",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "era5_proxy", "rho"),
        note="9-area proxy gradient vs PV/peak (still significant +0.72)",
    ),
    MacroCheck(
        macro="ProxyAllHoursConcNineP",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "era5_proxy", "p"),
        note="9-area proxy gradient p-value (significant)",
    ),
    MacroCheck(
        macro="HistfcAllHoursConcNine",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "histfc_reconstruction", "rho"),
        note="9-area reconstruction gradient (washout, n.s.)",
    ),
    MacroCheck(
        macro="HistfcAllHoursConcNineP",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "histfc_reconstruction", "p"),
        note="9-area reconstruction gradient p-value (n.s.)",
    ),
    MacroCheck(
        macro="GpvAllHoursConcNine",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "rho"),
        note="9-area exact-issuance GPV gradient (washout, n.s.)",
    ),
    MacroCheck(
        macro="GpvAllHoursConcNineP",
        provenance="phase41_allhours_headtohead_9area.json",
        json_path=("cross_area_spearman_vs_pvpeak", "gpv_exact_issuance", "p"),
        note="9-area exact-issuance GPV gradient p-value (n.s.)",
    ),
    # ----- Round-37 Phase 3: channel-resolved forecast skill (midday corr; review #17) -----
    # 2-decimal corr display -> tol 0.01.
    MacroCheck(
        macro="ChanProxySW",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "shortwave", "proxy_corr_midday"),
        tol=1e-2,
        note="proxy shortwave skill (midday corr)",
    ),
    MacroCheck(
        macro="ChanGpvSW",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "shortwave", "gpv_corr_midday"),
        tol=1e-2,
        note="GPV shortwave skill (midday corr)",
    ),
    MacroCheck(
        macro="ChanProxyTemp",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "temperature", "proxy_corr_midday"),
        tol=1e-2,
        note="proxy temperature skill (near-saturated)",
    ),
    MacroCheck(
        macro="ChanGpvTemp",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "temperature", "gpv_corr_midday"),
        tol=1e-2,
        note="GPV temperature skill",
    ),
    MacroCheck(
        macro="ChanProxyCloud",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "cloud", "proxy_corr_midday"),
        tol=1e-2,
        note="proxy cloud skill (narrowband: poor)",
    ),
    MacroCheck(
        macro="ChanGpvCloud",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "cloud", "gpv_corr_midday"),
        tol=1e-2,
        note="GPV cloud skill",
    ),
    MacroCheck(
        macro="ChanProxyWind",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "wind", "proxy_corr_midday"),
        tol=1e-2,
        note="proxy wind skill (narrowband: poor)",
    ),
    MacroCheck(
        macro="ChanGpvWind",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "wind", "gpv_corr_midday"),
        tol=1e-2,
        note="GPV wind skill",
    ),
    MacroCheck(
        macro="ChanGainWind",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "wind", "gpv_gain_midday"),
        tol=1e-2,
        note="GPV skill gain over proxy on wind (largest, kappa-free)",
    ),
    MacroCheck(
        macro="ChanGainCloud",
        provenance="phase42_channel_skill.json",
        json_path=("channels", "cloud", "gpv_gain_midday"),
        tol=1e-2,
        note="GPV skill gain over proxy on cloud (kappa-free broad channel)",
    ),
    # ----- Round-34 Phase 3: controlled skill-degradation sweep (curve sorted ascending by skill) -----
    MacroCheck(
        macro="SkillCurveHiSkill",
        provenance="phase36_skill_degradation.json",
        json_path=("curve", 5, "skill_corr"),
        note="forecast skill (corr) at the pristine-forecast level (curve[-1])",
    ),
    MacroCheck(
        macro="SkillCurveHiVal",
        provenance="phase36_skill_degradation.json",
        json_path=("curve", 5, "mean_improvement_yen_kwh"),
        note="mean weather value at the pristine-forecast level (curve[-1])",
    ),
    MacroCheck(
        macro="SkillCurveLoSkill",
        provenance="phase36_skill_degradation.json",
        json_path=("curve", 0, "skill_corr"),
        note="forecast skill (corr) at the most-degraded level (curve[0])",
    ),
    MacroCheck(
        macro="SkillCurveLoVal",
        provenance="phase36_skill_degradation.json",
        json_path=("curve", 0, "mean_improvement_yen_kwh"),
        note="mean weather value at the most-degraded level (curve[0])",
    ),
    # ----- Round-35 Phase B: area-month panel escaping the n=9 cross-section -----
    MacroCheck(
        macro="PanelMonthN",
        provenance="phase37_areamonth_panel.json",
        json_path=("n_cells",),
        note="area-month panel cells (9 areas x 15 months, >=3 folds)",
    ),
    MacroCheck(
        macro="PanelMonthBetaPooled",
        provenance="phase37_areamonth_panel.json",
        json_path=("specs", 0, "beta_floor"),
        note="area-month panel pooled beta on FLOOR (spec0)",
    ),
    MacroCheck(
        macro="PanelMonthBeta",
        provenance="phase37_areamonth_panel.json",
        json_path=("specs", 1, "beta_floor"),
        note="area-month panel area-FE beta on FLOOR (within-area temporal, headline)",
    ),
    MacroCheck(
        macro="PanelMonthWildP",
        provenance="phase37_areamonth_panel.json",
        json_path=("specs", 1, "p_wild_cluster"),
        note="area-month panel area-FE wild-cluster p (G=9)",
    ),
    MacroCheck(
        macro="PanelMonthBetaSatFE",
        provenance="phase37_areamonth_panel.json",
        json_path=("specs", 2, "beta_floor"),
        note="area-month panel area+month-FE beta (attenuated)",
    ),
    MacroCheck(
        macro="PanelMonthSatP",
        provenance="phase37_areamonth_panel.json",
        json_path=("specs", 2, "p_wild_cluster"),
        note="area-month panel area+month-FE wild-cluster p (n.s.)",
    ),
    # ----- Round-35 Phase E: external-validity exploratory on EPEX-DE/LU -----
    MacroCheck(
        macro="DEWeatherValue",
        provenance="phase38_epex_de_exploratory.json",
        json_path=("mean_improvement_eur_mwh",),
        note="DE mean weather improvement EUR/MWh (effect replicates externally)",
    ),
    MacroCheck(
        macro="DENdays",
        provenance="phase38_epex_de_exploratory.json",
        json_path=("n_days",),
        note="DE walk-forward evaluation days",
    ),
    MacroCheck(
        macro="DEConcRho",
        provenance="phase38_epex_de_exploratory.json",
        json_path=("spearman_improvement_vs_negshare",),
        note="DE Spearman(weather improvement, negative-price share) -- weak, n.s.",
    ),
    MacroCheck(
        macro="DEConcP",
        provenance="phase38_epex_de_exploratory.json",
        json_path=("spearman_p",),
        note="DE concentration-analogue p-value (n.s. at the softer boundary)",
    ),
    # ----- Round-33 B3: reviewer-requested literal slot-level intensity panel (transparent null) -----
    MacroCheck(
        macro="IntensityNoFEbeta",
        provenance="phase30.1_distance_to_floor.json",
        json_path=("pv_to_demand", "spec0_no_fe_wild_cluster", "beta"),
        note="literal intensity panel no-FE slope (suggestive, wild-cluster p<0.001)",
    ),
    MacroCheck(
        macro="IntensityNoFEp",
        provenance="phase30.1_distance_to_floor.json",
        json_path=(
            "pv_to_demand",
            "spec0_no_fe_wild_cluster",
            "p_wild_cluster_bootstrap_two_sided",
        ),
        # DECLARED 2026-08-18. The macro prints "<0.001", a CEILING, while the
        # provenance is exactly 0.0, so macro and provenance are not meant to be
        # equal here and the printed-precision default (0.0005) correctly refuses
        # them. The flat 5e-3 default used to hide that distinction. The tolerance
        # is the width of the bound itself: any p at or below the printed ceiling
        # satisfies the sentence.
        tol=0.0011,
        note="literal intensity panel no-FE wild-cluster p (0.0 -> reported <0.001)",
    ),
    MacroCheck(
        macro="IntensityFEbeta",
        provenance="phase30.1_distance_to_floor.json",
        json_path=("verdict", "pv_to_demand_spec1_beta"),
        note="literal intensity panel saturated-FE slope (attenuated)",
    ),
    MacroCheck(
        macro="IntensityFEp",
        provenance="phase30.1_distance_to_floor.json",
        json_path=("verdict", "pv_to_demand_spec1_p"),
        note="literal intensity panel saturated-FE p (n.s.; FE-fragile)",
    ),
    # ----- Round-31 / Phase 31c: real-yen floor-event decision value (review #3) -----
    MacroCheck(
        macro="RealFloorWedge",
        provenance="phase31c_real_yen_decision.json",
        json_path=("floor_real_over_cost_yen_kwh_median",),
        note="median real floor over-cost (shortage imbalance - spot) JPY/kWh",
    ),
    MacroCheck(
        macro="RealDecisionCostPriceOnly",
        provenance="phase31c_real_yen_decision.json",
        json_path=("tiers", "price_only", "real_decision_cost_yen_per_mw_yr"),
        scale=1e-6,
        tol=0.01,
        note="price-only real imbalance exposure (M JPY/MW/yr)",
    ),
    MacroCheck(
        macro="RealDecisionCostOps",
        provenance="phase31c_real_yen_decision.json",
        json_path=("tiers", "operational", "real_decision_cost_yen_per_mw_yr"),
        scale=1e-6,
        tol=0.01,
        note="operational real imbalance exposure (M JPY/MW/yr)",
    ),
    MacroCheck(
        macro="OpsRealValueK",
        provenance="phase31c_real_yen_decision.json",
        json_path=("operational_value_yen_per_mw_yr",),
        scale=1e-3,
        tol=0.6,
        note="operational forecast real decision value (thousand JPY/MW/yr)",
    ),
    # ----- Round-31 / Phase 32: honest enriched-baseline robustness (review #7/#2) -----
    MacroCheck(
        macro="EnrichBaselineGain",
        provenance="phase32_honest_baseline_enriched.json",
        json_path=("enrichment_strength", "mean_enrichment_gain_over_price_yen_kwh"),
        note="enriched baseline beats price-only (mean JPY/kWh) -> baseline is strong",
    ),
    MacroCheck(
        macro="OpsOverEnriched",
        provenance="phase32_honest_baseline_enriched.json",
        json_path=("operational_histfc_over_enriched", "mean_yen_kwh"),
        note="operational forecast mean MAE improvement over the enriched baseline",
    ),
    MacroCheck(
        macro="OpsOverEnrichedNsig",
        provenance="phase32_honest_baseline_enriched.json",
        json_path=("operational_histfc_over_enriched", "n_areas_dm_sig_below_0p05"),
        tol=0.5,
        note="areas where operational fc beats enriched baseline at DM p<0.05 (of 8)",
    ),
    MacroCheck(
        macro="OpsOverEnrichedConc",
        provenance="phase32_honest_baseline_enriched.json",
        json_path=("operational_histfc_over_enriched", "spearman_vs_pv_peak", "rho"),
        note="cross-area Spearman of operational improvement vs PV/peak (broad, n.s.)",
    ),
    MacroCheck(
        macro="ProxyOverEnriched",
        provenance="phase32_honest_baseline_enriched.json",
        json_path=("era5_proxy_over_enriched", "mean_yen_kwh"),
        note="ERA5 proxy mean MAE improvement over the enriched baseline",
    ),
    # ----- Round-37 / Phase 40: GENUINELY bid-close-admissible enriched baseline -----
    # supply mix moved to lag-96 (D-2, observed by D-1 10:00 for every slot); review #17 must-fix.
    MacroCheck(
        macro="EnrichBaselineGainAdm",
        provenance="phase40_admissible_supply_lag96.json",
        json_path=("enrichment_strength", "mean_enrichment_gain_over_price_yen_kwh"),
        note="admissible (lag-96-supply) enriched baseline beats price-only (mean JPY/kWh)",
    ),
    MacroCheck(
        macro="OpsOverEnrichedAdm",
        provenance="phase40_admissible_supply_lag96.json",
        json_path=("operational_histfc_over_enriched", "mean_yen_kwh"),
        note="operational forecast mean MAE improvement over the admissible enriched baseline",
    ),
    MacroCheck(
        macro="OpsOverEnrichedAdmNsig",
        provenance="phase40_admissible_supply_lag96.json",
        json_path=("operational_histfc_over_enriched", "n_areas_dm_sig_below_0p05"),
        tol=0.5,
        note="areas where operational fc beats admissible enriched baseline at DM p<0.05 (of 8)",
    ),
    MacroCheck(
        macro="OpsOverEnrichedAdmConc",
        provenance="phase40_admissible_supply_lag96.json",
        json_path=("operational_histfc_over_enriched", "spearman_vs_pv_peak", "rho"),
        note="cross-area Spearman of operational improvement vs PV/peak over admissible baseline (broad, n.s.)",
    ),
    MacroCheck(
        macro="ProxyOverEnrichedAdm",
        provenance="phase40_admissible_supply_lag96.json",
        json_path=("era5_proxy_over_enriched", "mean_yen_kwh"),
        note="ERA5 proxy mean MAE improvement over the admissible enriched baseline",
    ),
    # ----- Round-31 / Phase 33: FY2022-2025 temporal robustness (review #2) -----
    MacroCheck(
        macro="TemporalFloorRhoCrisis",
        provenance="phase33_temporal_kyushu.json",
        json_path=("cross_area_floor_vs_pvpeak_by_fy", "fy2022", "rho"),
        note="FY2022 (crisis) cross-area Spearman(floor, PV/peak) -- gradient holds in crisis year",
    ),
    MacroCheck(
        macro="TemporalFloorRhoCrisisP",
        provenance="phase33_temporal_kyushu.json",
        json_path=("cross_area_floor_vs_pvpeak_by_fy", "fy2022", "p"),
        note="FY2022 cross-area gradient p-value",
    ),
    MacroCheck(
        macro="TemporalFloorRhoCalm",
        provenance="phase33_temporal_kyushu.json",
        json_path=("cross_area_floor_vs_pvpeak_by_fy", "fy2025", "rho"),
        note="FY2025 (calm) cross-area Spearman(floor, PV/peak)",
    ),
    MacroCheck(
        macro="TemporalFloorRhoCalmP",
        provenance="phase33_temporal_kyushu.json",
        json_path=("cross_area_floor_vs_pvpeak_by_fy", "fy2025", "p"),
        note="FY2025 cross-area gradient p-value",
    ),
    MacroCheck(
        macro="TemporalFloorRhoMin",
        provenance="phase33_temporal_kyushu.json",
        json_path=("cross_area_floor_vs_pvpeak_by_fy", "summary", "rho_min"),
        note="weakest of the four annual cross-area gradients (all four positive)",
    ),
    MacroCheck(
        macro="KyushuWxValueCrisis",
        provenance="phase33_temporal_kyushu.json",
        json_path=("regimes", "crisis_fy2022_2023", "weather_value_mean"),
        note="Kyushu mean weather value, crisis regime FY2022-23",
    ),
    MacroCheck(
        macro="KyushuWxValueCalm",
        provenance="phase33_temporal_kyushu.json",
        json_path=("regimes", "calm_fy2024_2025", "weather_value_mean"),
        note="Kyushu mean weather value, calm regime FY2024-25",
    ),
    # ---- Phase 43: decision-value sensitivity (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="OpsValueRuleMinK",
        provenance="phase43_decision_sensitivity.json",
        json_path=(
            "threshold_rules",
            "max_net_value_crossfit_fa0",
            "operational_value_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="min operational value across threshold rules (kJPY/MW/yr)",
    ),
    MacroCheck(
        macro="OpsValueRuleMaxK",
        provenance="phase43_decision_sensitivity.json",
        json_path=(
            "threshold_rules",
            "max_net_value_crossfit_fa1",
            "operational_value_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="max operational value across threshold rules (kJPY/MW/yr)",
    ),
    MacroCheck(
        macro="OpsFlagCount",
        provenance="phase43_decision_sensitivity.json",
        json_path=(
            "threshold_rules",
            "max_f1_crossfit_reference",
            "tiers",
            "operational",
            "n_flagged",
        ),
        tol=0.5,
        note="operational reference-rule flagged slots",
    ),
    MacroCheck(
        macro="PriceOnlyFlagCount",
        provenance="phase43_decision_sensitivity.json",
        json_path=(
            "threshold_rules",
            "max_f1_crossfit_reference",
            "tiers",
            "price_only",
            "n_flagged",
        ),
        tol=0.5,
        note="price-only reference-rule flagged slots",
    ),
    # ---- Phase 44: point-metric broadening (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="RmaePriceOnly",
        provenance="phase44_metrics_broadening.json",
        json_path=("artifacts", "phase26_1_pv_surplus_8area", "pooled", "price_only", "rmae"),
        note="pooled rMAE, price-only (phase26.1 cache)",
    ),
    MacroCheck(
        macro="RmaeProxy",
        provenance="phase44_metrics_broadening.json",
        json_path=("artifacts", "phase26_1_pv_surplus_8area", "pooled", "era5_proxy", "rmae"),
        note="pooled rMAE, +lag-48 proxy",
    ),
    MacroCheck(
        macro="RmaeOps",
        provenance="phase44_metrics_broadening.json",
        json_path=(
            "artifacts",
            "phase26_1_pv_surplus_8area",
            "pooled",
            "histfc_reconstruction",
            "rmae",
        ),
        note="pooled rMAE, +operational reconstruction",
    ),
    MacroCheck(
        macro="CacheValueMultipleMae",
        provenance="phase44_metrics_broadening.json",
        json_path=(
            "artifacts",
            "phase26_1_pv_surplus_8area",
            "value_multiple_histfc_over_proxy",
            "mae",
        ),
        tol=1e-2,
        note="histfc/proxy improvement multiple, MAE (cache)",
    ),
    MacroCheck(
        macro="CacheValueMultipleRmse",
        provenance="phase44_metrics_broadening.json",
        json_path=(
            "artifacts",
            "phase26_1_pv_surplus_8area",
            "value_multiple_histfc_over_proxy",
            "rmse",
        ),
        tol=1e-2,
        note="histfc/proxy improvement multiple, RMSE (metric-robust)",
    ),
    # ---- Phase 45: EPEX-DE two-tier replication (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="DETestDays",
        provenance="phase45_epex_de_tiered.json",
        json_path=("coverage", "head_to_head_test_days"),
        tol=0.5,
        note="DE head-to-head test days",
    ),
    MacroCheck(
        macro="DEOpsImp",
        provenance="phase45_epex_de_tiered.json",
        json_path=("pooled", "improvement_op_eur_mwh"),
        note="DE operational-tier improvement (EUR/MWh)",
    ),
    MacroCheck(
        macro="DEProxyImp",
        provenance="phase45_epex_de_tiered.json",
        json_path=("pooled", "improvement_proxy_eur_mwh"),
        note="DE lagged-proxy improvement (EUR/MWh, n.s.)",
    ),
    MacroCheck(
        macro="DEOpsOverProxy",
        provenance="phase45_epex_de_tiered.json",
        json_path=("pooled", "ratio_op_over_proxy"),
        tol=0.5,
        note="DE operational/proxy improvement ratio",
    ),
    MacroCheck(
        macro="DEWindGain",
        provenance="phase45_epex_de_tiered.json",
        json_path=("channel_ablation", "families", "wind", "gain_op_eur_mwh"),
        note="DE wind-channel operational gain (EUR/MWh)",
    ),
    MacroCheck(
        macro="DENegGradRho",
        provenance="phase45_epex_de_tiered.json",
        json_path=(
            "exploratory_negative_price_gradient",
            "spearman_daily_improvement_vs_negshare",
            "op",
            "rho",
        ),
        note="DE exploratory neg-price gradient (op tier), NOT promoted",
    ),
    # ---- Phase 46: DNN baseline (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="DnnProxyImp",
        provenance="phase46_dnn_baseline.json",
        json_path=("mean_improvement_yen_kwh", "era5_proxy"),
        note="DNN pooled proxy improvement (JPY/kWh)",
    ),
    MacroCheck(
        macro="DnnOpsImp",
        provenance="phase46_dnn_baseline.json",
        json_path=("mean_improvement_yen_kwh", "histfc_reconstruction"),
        note="DNN pooled operational improvement (JPY/kWh)",
    ),
    MacroCheck(
        macro="DnnGpvImp",
        provenance="phase46_dnn_baseline.json",
        json_path=("mean_improvement_yen_kwh", "gpv_exact_issuance"),
        note="DNN pooled GPV improvement (JPY/kWh)",
    ),
    # ---- Phase 47: probabilistic tier evaluation (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="CrpsProxyImp",
        provenance="phase47_tier_probabilistic.json",
        json_path=("pooled_9area", "improvements_over_price_only", "era5_proxy", "crps"),
        note="9-area pooled CRPS improvement, proxy",
    ),
    MacroCheck(
        macro="CrpsOpsImp",
        provenance="phase47_tier_probabilistic.json",
        json_path=("pooled_9area", "improvements_over_price_only", "histfc", "crps"),
        note="9-area pooled CRPS improvement, operational",
    ),
    MacroCheck(
        macro="CrpsValueMultiple",
        provenance="phase47_tier_probabilistic.json",
        json_path=(
            "pooled_9area",
            "improvements_over_price_only",
            "value_multiple_histfc_over_proxy",
            "crps",
        ),
        tol=1e-2,
        note="operational/proxy CRPS-improvement multiple",
    ),
    # ---- Phase 49: true previous-day-run tier + extended imbalance (AE restructure, 2026-07-24) ----
    MacroCheck(
        macro="PrevRunImp",
        provenance="phase49_exante_fullyear.json",
        json_path=("headtohead", "pooled_improvement_yen_kwh", "prev1"),
        note="true previous-day-run pooled improvement, 21 months, 9 areas",
    ),
    MacroCheck(
        macro="PrevRunProxyImp",
        provenance="phase49_exante_fullyear.json",
        json_path=("headtohead", "pooled_improvement_yen_kwh", "era5"),
        note="same-window proxy improvement (n.s.)",
    ),
    MacroCheck(
        macro="PrevRunTestDays",
        provenance="phase49_exante_fullyear.json",
        json_path=("headtohead", "window", "n_test_days"),
        tol=0.5,
        note="test days per area, extended window",
    ),
    MacroCheck(
        macro="PrevRunReconGap",
        provenance="phase49_exante_fullyear.json",
        json_path=("headtohead", "reconstruction_vs_true_forecast", "mean_abs_gap_yen_kwh"),
        note="mean |gap| reconstruction vs true run",
    ),
    MacroCheck(
        macro="PrevRunAdmImp",
        provenance="phase49_exante_fullyear.json",
        json_path=(
            "headtohead",
            "strictly_admissible_morning_subset",
            "pooled_improvement_yen_kwh",
            "prev1",
        ),
        note="morning strictly-admissible subset, true-run improvement",
    ),
    MacroCheck(
        macro="PrevRunAdmProxyImp",
        provenance="phase49_exante_fullyear.json",
        json_path=(
            "headtohead",
            "strictly_admissible_morning_subset",
            "pooled_improvement_yen_kwh",
            "era5",
        ),
        note="morning subset, proxy improvement",
    ),
    MacroCheck(
        macro="PrevRunDecisionExtK",
        provenance="phase49_exante_fullyear.json",
        json_path=(
            "decision_extended",
            "extended_window_8area",
            "operational_value_yen_per_mw_yr",
            "prev1",
        ),
        scale=1e-3,
        tol=0.6,
        note="extended-window decision value (kJPY/MW/yr, wedge-regime dependent)",
    ),
    # ---- Phase 51: dissemination-correct vintage ladder (gate-closure revision, 2026-07) ----
    # ---- Phase 59: run-pinned ladder. These REPLACE the phase51 composite-tier
    # checks: the headline pre-gate tier is now a single pinned run, so the values
    # must be bound to the ladder that actually produced them. phase51 remains the
    # Gate 5 reproducibility reference inside phase59 itself.
    MacroCheck(
        macro="VintageTestDays",
        provenance="phase85_exact_ladder.json",
        json_path=("panel", "n_test_days"),
        tol=0.5,
        note="consecutive daily test days, 9-area run-pinned ladder",
    ),
    MacroCheck(macro="VintageTestSlots", provenance="phase85_exact_ladder.json",
               json_path=("panel", "n_slots"), tol=0.5,
               note="half-hour slot observations in the primary panel"),
    MacroCheck(
        macro="StaleObsImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "era5_cw", "improvement"),
        note="stale-observation proxy pooled improvement",
    ),
    MacroCheck(macro="StaleObsImpLo", provenance="phase85_exact_ladder.json",
               json_path=("arm_cw_primary", "tiers", "era5_cw", "ci95", 0),
               note="stale-observation improvement, CI lower -- the prose used "
                    "to hardcode the superseded [0.024, 0.058] while the table "
                    "printed the current interval"),
    MacroCheck(macro="StaleObsImpHi", provenance="phase85_exact_ladder.json",
               json_path=("arm_cw_primary", "tiers", "era5_cw", "ci95", 1),
               note="stale-observation improvement, CI upper"),
    MacroCheck(
        macro="PreGateImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_d2_12z_cw", "improvement"),
        note="headline run-pinned pre-gate tier (D-2 12 UTC run)",
    ),
    MacroCheck(
        macro="PreGateImpLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_d2_12z_cw", "ci95", 0),
        note="headline CI lower",
    ),
    MacroCheck(
        macro="PreGateImpHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_d2_12z_cw", "ci95", 1),
        note="headline CI upper",
    ),
    MacroCheck(
        macro="GateCompositeImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_gate_composite_cw", "improvement"),
        note="per-slot freshest admissible run (corroborating pre-gate tier)",
    ),
    MacroCheck(
        macro="GateCompositeImpLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_gate_composite_cw", "ci95", 0),
        note="its CI lower",
    ),
    MacroCheck(
        macro="GateCompositeImpHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_gate_composite_cw", "ci95", 1),
        note="its CI upper",
    ),
    MacroCheck(
        macro="PostGateImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "msm_d1_00z_cw", "improvement"),
        note="the run the gate excludes (D-1 00 UTC), run-pinned",
    ),
    MacroCheck(
        macro="ReconLadderImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_sp_robustness", "tiers", "histfc", "improvement"),
        note="operational reconstruction (post-gate class composite)",
    ),
    MacroCheck(
        macro="ExPostImp",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "tiers", "oracle_cw", "improvement"),
        note="ex-post realised-weather benchmark (not a ceiling)",
    ),
    MacroCheck(
        macro="PreGateMultiple",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "ratio", "ratio"),
        tol=1e-2,
        note="headline over stale proxy: a product+vintage+availability composite",
    ),
    MacroCheck(
        macro="PreGateMultipleLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "ratio", "ci95", 0),
        tol=1e-2,
        note="multiple CI lower (bootstrapped ratio, not a hand division)",
    ),
    MacroCheck(
        macro="PreGateMultipleHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "ratio", "ci95", 1),
        tol=1e-2,
        note="multiple CI upper",
    ),
    MacroCheck(
        macro="PreGateVsStaleDiff",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "difference", "delta"),
        note="headline minus stale proxy",
    ),
    MacroCheck(
        macro="PreGateVsStaleDiffLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "difference", "ci95", 0),
        note="its CI lower",
    ),
    MacroCheck(
        macro="PreGateVsStaleDiffHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "headline_vs_stale", "difference", "ci95", 1),
        note="its CI upper",
    ),
    MacroCheck(
        macro="GateTimingValue",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "delta"),
        note="what the gate excludes; Gate 7b: not distinguishable from zero",
    ),
    MacroCheck(
        macro="GateTimingValueLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "ci95", 0),
        note="gate-timing CI lower",
    ),
    MacroCheck(
        macro="GateTimingValueHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "ci95", 1),
        note="gate-timing CI upper",
    ),
    MacroCheck(
        macro="GateTimingUpperBound",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "upper_bound_if_b"),
        note="95% upper bound on what a later gate would buy",
    ),
    MacroCheck(
        macro="GateTimingP",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "dm_l3", "p_hln"),
        tol=1e-2,
        note="pooled DM p for the gate-timing difference (n.s.)",
    ),
    MacroCheck(
        macro="PreGateRetention",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "headline_retention_of_post_gate_pct"),
        tol=0.6,
        note="headline as a share of the excluded run's improvement (%)",
    ),
    MacroCheck(
        macro="VintageEffectWithinEraFive",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "within_era5_vintage_effect", "delta"),
        note="Gate 8: vintage effect with the product held fixed",
    ),
    MacroCheck(
        macro="VintageEffectWithinEraFiveLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "within_era5_vintage_effect", "ci95", 0),
        note="its CI lower",
    ),
    MacroCheck(
        macro="VintageEffectWithinEraFiveHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "within_era5_vintage_effect", "ci95", 1),
        note="its CI upper",
    ),
    MacroCheck(
        macro="ProductBound",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "product_bound", "delta"),
        note="Gate 8: product contribution with vintage ~fixed (n.s.)",
    ),
    MacroCheck(
        macro="ProductBoundLo",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "product_bound", "ci95", 0),
        note="its CI lower",
    ),
    MacroCheck(
        macro="ProductBoundHi",
        provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_decomposition", "product_bound", "ci95", 1),
        note="its CI upper",
    ),
    MacroCheck(
        macro="CompositeVsPinned",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_composite_vs_pinned_sp", "delta"),
        note="the old fixed-lead composite minus the run-pinned headline",
    ),
    MacroCheck(
        macro="CompositeVsPinnedLo",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_composite_vs_pinned_sp", "ci95", 0),
        note="its CI lower",
    ),
    MacroCheck(
        macro="CompositeVsPinnedHi",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_composite_vs_pinned_sp", "ci95", 1),
        note="its CI upper",
    ),
    MacroCheck(
        macro="CompositeVsPinnedP",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_composite_vs_pinned_sp", "dm_l3", "p_hln"),
        tol=1e-2,
        note="pooled DM p: the composite understated the vintage significantly",
    ),
    MacroCheck(
        macro="IngestionDiff",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_ingestion_sp", "delta"),
        note="Gate 9: same initialisation, pinned run vs fixed-lead composite",
    ),
    MacroCheck(
        macro="IngestionDiffP",
        provenance="phase85_exact_ladder.json",
        json_path=("gate50_ingestion_sp", "dm_l3", "p_hln"),
        tol=1e-2,
        note="its pooled DM p (n.s.: the two ingestions agree)",
    ),
    MacroCheck(
        macro="DoseSlopePerLeadHour",
        provenance="phase59_exact_ladder.json",
        json_path=("gate10_dose_response", "slope_per_lead_hour"),
        tol=1e-4,
        note="Gate 10: improvement per extra hour of lead, run-pinned tiers only",
    ),
    # ---- Phase 52: FIP commitment decision value, prev2-primary ----
    MacroCheck(
        macro="CnpePreK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "voi_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="earlier-window CNPE, kJPY per MW-installed-year",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePreLoK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            0,
        ),
        scale=1e-3,
        tol=0.6,
        note="earlier-window CNPE CI lower",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePreHiK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            1,
        ),
        scale=1e-3,
        tol=0.6,
        note="earlier-window CNPE CI upper",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePostK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "voi_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="later-window CNPE",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePostLoK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            0,
        ),
        scale=1e-3,
        tol=0.6,
        note="later-window CNPE CI lower",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePostHiK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            1,
        ),
        scale=1e-3,
        tol=0.6,
        note="later-window CNPE CI upper",
       paper_b=True,
    ),
    MacroCheck(
        macro="FloorWedgeMedianPre",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "floor_wedge_context",
            "median_k_minus_pda_on_floor_yen_kwh",
        ),
        tol=1e-2,
        note="median settlement wedge on floor slots, pre-collapse",
       paper_b=True,
    ),
    MacroCheck(
        macro="FloorWedgeMedianPost",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "floor_wedge_context",
            "median_k_minus_pda_on_floor_yen_kwh",
        ),
        tol=1e-2,
        note="median settlement wedge on floor slots, post-collapse",
       paper_b=True,
    ),
    MacroCheck(
        macro="AlwaysHedgeBeatPerfectK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "sanity_perfect_dominance",
            "always_hedge_minus_perfect_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="never-sell policy minus perfect information, pre-collapse",
       paper_b=True,
    ),
    MacroCheck(
        macro="AlwaysHedgeBeatPerfectMatchedK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "matched_committed_pre",
            "sanity_perfect_dominance",
            "always_hedge_minus_perfect_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.6,
        note="same, matched committed-window days",
       paper_b=True,
    ),
    # ---- Phase 54: NWP-proxy-enriched baseline ----
    MacroCheck(
        macro="PreGateOverEnrichedAdm",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d42_total_prev2_over_enriched",
            "point_daily_mean",
        ),
        note="pre-gate vintage value over the admissible enriched baseline",
    ),
    MacroCheck(
        macro="PreGateOverEnrichedAdmLo",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d42_total_prev2_over_enriched",
            "lo95",
        ),
        note="CI lower",
    ),
    MacroCheck(
        macro="PreGateOverEnrichedAdmHi",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d42_total_prev2_over_enriched",
            "hi95",
        ),
        note="CI upper",
    ),
    MacroCheck(
        macro="ProxyCarriedValue",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d32_proxies_over_enriched",
            "point_daily_mean",
        ),
        note="value carried by same-vintage engineered proxies",
    ),
    MacroCheck(
        macro="RawResidualOverProxies",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d43_raw_over_proxies",
            "point_daily_mean",
        ),
        note="residual raw-channel value over the proxies",
    ),
    MacroCheck(
        macro="RawResidualLo",
        provenance="phase54v2_current.json",
        json_path=("results", "deltas_daily_mean_with_ci95", "d43_raw_over_proxies", "lo95"),
        note="CI lower",
    ),
    MacroCheck(
        macro="RawResidualHi",
        provenance="phase54v2_current.json",
        json_path=("results", "deltas_daily_mean_with_ci95", "d43_raw_over_proxies", "hi95"),
        note="CI upper",
    ),
    MacroCheck(
        macro="EnrichGainAdmDaily",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d21_enrichment_gain",
            "point_daily_mean",
        ),
        note="price-only to admissible enriched baseline, daily protocol",
    ),
    # ---- Phase 55: tier slope-difference tests ----
    MacroCheck(
        macro="TierDrhoPreGate",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "pair_tests", "era5_vs_prev2", "delta_rho", "point"),
        tol=1e-2,
        note="Spearman-rho difference, stale proxy minus pre-gate vintage",
    ),
    MacroCheck(
        macro="TierDrhoPreGateLo",
        provenance="phase55_tier_slope_tests.json",
        json_path=(
            "windows",
            "full_phase51",
            "pair_tests",
            "era5_vs_prev2",
            "delta_rho",
            "boot_lo95",
        ),
        tol=1e-2,
        note="CI lower",
    ),
    MacroCheck(
        macro="TierDrhoPreGateHi",
        provenance="phase55_tier_slope_tests.json",
        json_path=(
            "windows",
            "full_phase51",
            "pair_tests",
            "era5_vs_prev2",
            "delta_rho",
            "boot_hi95",
        ),
        tol=1e-2,
        note="CI upper",
    ),
    # ---- Phase 53: exact archived pre-gate runs (corroboration) ----
    MacroCheck(
        macro="ExactPreGateDays",
        provenance="phase53_rish_pregate.json",
        json_path=("coverage", "delivery_days"),
        tol=0.5,
        note="delivery days of exact pre-gate extraction",
    ),
    MacroCheck(
        macro="ExactVsPreGateTempRmse",
        provenance="phase53_rish_pregate.json",
        json_path=("pooled", "vs_prev2", "temperature_c", "rmse"),
        tol=1e-2,
        note="exact pre-gate vs previous-day-2 vintage, temperature RMSE",
    ),
    MacroCheck(
        macro="ExactVsPostGateTempRmse",
        provenance="phase53_rish_pregate.json",
        json_path=("pooled", "vs_postgate_00utc", "temperature_c", "rmse"),
        tol=1e-2,
        note="exact pre-gate vs first post-gate run, temperature RMSE",
    ),
    # ---- Phase 62: single-point proxy validation vs OCCTO actuals ----
    MacroCheck(
        macro="ProxyPvRLo",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "pv", "pearson_r_min"),
        tol=1e-2,
        note="min daytime Pearson r, PV proxy vs realised area PV potential",
    ),
    MacroCheck(
        macro="ProxyPvRHi",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "pv", "pearson_r_max"),
        tol=1e-2,
        note="max daytime Pearson r, PV proxy",
    ),
    MacroCheck(
        macro="ProxyPvRMedian",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "pv", "pearson_r_median"),
        tol=1e-2,
        note="median daytime Pearson r, PV proxy",
    ),
    MacroCheck(
        macro="ProxyWindRLo",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "wind", "pearson_r_min"),
        tol=1e-2,
        note="min Pearson r, wind proxy vs realised wind potential",
    ),
    MacroCheck(
        macro="ProxyWindRHi",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "wind", "pearson_r_max"),
        tol=1e-2,
        note="max Pearson r, wind proxy",
    ),
    MacroCheck(
        macro="ProxyWindRMedian",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "wind", "pearson_r_median"),
        tol=1e-2,
        note="median Pearson r, wind proxy",
    ),
    MacroCheck(
        macro="ProxyDemandRLo",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "demand", "pearson_r_min"),
        tol=1e-2,
        note="min Pearson r, degree-hour proxy vs realised demand",
    ),
    MacroCheck(
        macro="ProxyDemandRHi",
        provenance="phase62_proxy_validation.json",
        json_path=("proxy_skill", "summary", "demand", "pearson_r_max"),
        tol=1e-2,
        note="max Pearson r, degree-hour proxy",
    ),
    MacroCheck(
        macro="CapacityWorstRatio",
        provenance="phase62_proxy_validation.json",
        json_path=("capacity_integrity", "worst", "realised_over_nameplate_ratio"),
        tol=5e-2,
        note="worst nameplate breach (Tokyo wind): realised max / nameplate",
    ),
    MacroCheck(
        macro="TierDrhoRealisedIdx",
        provenance="phase55_tier_slope_tests_realised.json",
        json_path=("windows", "full_phase51", "pair_tests", "era5_vs_prev2", "delta_rho", "point"),
        tol=1e-2,
        note="Delta-rho under the vintage-free saturation index",
    ),
    MacroCheck(
        macro="TierDrhoRealisedIdxPermArea",
        provenance="phase55_tier_slope_tests_realised.json",
        json_path=(
            "windows",
            "full_phase51",
            "pair_tests",
            "era5_vs_prev2",
            "perm_area_level_exact",
            "p_two_sided_delta_rho",
        ),
        tol=1e-2,
        note="exact area-level permutation p for Delta-rho, vintage-free index",
    ),
    # ---- Phase 61: physical/system metrics ----
    MacroCheck(
        macro="CnpeAbsImbPre",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "annualised",
        ),
        tol=0.6,
        note="Delta absolute imbalance, MWh per MW-yr, earlier window",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPreLo",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "ci95",
            0,
        ),
        tol=0.6,
        note="same, CI lower",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPreHi",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "ci95",
            1,
        ),
        tol=0.6,
        note="same, CI upper",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPrePct",
        provenance="phase61_system_metrics_band.json",
        json_path=("system_metrics", "windows", "pre", "op_vs_base", "d_abs_imbalance_pct"),
        scale=-1.0,
        tol=0.1,
        note="imbalance reduction, percent (macro carries the sign in prose)",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPost",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "annualised",
        ),
        tol=0.6,
        note="Delta absolute imbalance, later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPostLo",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "ci95",
            0,
        ),
        tol=0.6,
        note="same, CI lower",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPostHi",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "ci95",
            1,
        ),
        tol=0.6,
        note="same, CI upper",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPooled",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pooled",
            "d_abs_imbalance_daily_inference",
            "annualised",
        ),
        tol=0.6,
        note="Delta absolute imbalance, full window",
       paper_b=True,
    ),
    MacroCheck(
        macro="ChatMaePreOp",
        provenance="phase61_system_metrics_band.json",
        json_path=("system_metrics", "windows", "pre", "curtailment_share_forecast_mae", "op"),
        tol=1e-3,
        note="curtailment-share forecast MAE, operational information set",
       paper_b=True,
    ),
    MacroCheck(
        macro="ChatMaePreBase",
        provenance="phase61_system_metrics_band.json",
        json_path=("system_metrics", "windows", "pre", "curtailment_share_forecast_mae", "base"),
        tol=1e-3,
        note="same, price-only information set",
       paper_b=True,
    ),
    MacroCheck(
        macro="AlwaysHedgeAbsImbPre",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "per_policy",
            "always_hedge",
            "abs_imbalance_mwh",
        ),
        tol=1.0,
        note="never-sell policy absolute imbalance, MWh per MW-yr",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeBandSigEps",
        provenance="phase61_system_metrics_band.json",
        json_path=("findings", "max_eps_with_significantly_negative_cnpe"),
        tol=1e-3,
        note="largest nomination band with a significantly negative CNPE",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeBandSigEpsPct",
        provenance="phase61_system_metrics_band.json",
        json_path=("findings", "max_eps_with_significantly_negative_cnpe"),
        scale=100.0,
        tol=0.1,
        note="same boundary as CnpeBandSigEps, expressed as a percentage",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeBandFlipEps",
        provenance="phase61_system_metrics_band.json",
        json_path=("findings", "first_eps_with_positive_point_estimate"),
        tol=1e-3,
        note="band at which the CNPE point estimate turns positive",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbBandTwenty",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "nomination_band",
            "families",
            "R1_information_responsive",
            "rows",
            5,
            "d_abs_imbalance_pre_mwh",
        ),
        tol=0.6,
        note="Delta absolute imbalance at the widest band: shading degrades physics",
       paper_b=True,
    ),
    # ---- Phase 60: wedge as a continuous treatment ----
    MacroCheck(
        macro="FloorSlotSharePct",
        provenance="phase60_wedge_continuous.json",
        json_path=("c1_identity", "floor_slot_share_pct"),
        tol=0.1,
        note="floor slots as a share of all slots",
       paper_b=True,
    ),
    MacroCheck(
        macro="FloorProfitSharePct",
        provenance="phase60_wedge_continuous.json",
        json_path=("c1_identity", "floor_share_of_total_profit_diff_pct"),
        tol=0.1,
        note="share of the profit difference carried by floor slots",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeBetaFloor",
        provenance="phase60_wedge_continuous.json",
        json_path=("c4_components", "floor_component", "coefficients", "exposure_floor", "beta"),
        tol=0.1,
        note="JPY per MW-day per JPY/kWh of floor-slot wedge exposure",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePostP",
        provenance="phase60_wedge_continuous.json",
        json_path=("c4_components", "floor_component", "coefficients", "post", "p"),
        tol=1e-2,
        note="post-split level dummy p, conditional on exposure",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeBreakSpanProfit",
        provenance="phase60_wedge_continuous.json",
        json_path=("c6_changepoint", "daily_profit_effect", "break_date_ci95_span_days"),
        tol=0.6,
        note="bootstrap 95% CI span on the break date, profit series (days)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeBreakSpanWedge",
        provenance="phase60_wedge_continuous.json",
        json_path=("c6_changepoint", "daily_floor_wedge", "break_date_ci95_span_days"),
        tol=0.6,
        note="same, wedge series",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeMeanPre",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "windows", "pre", "mean_yen_kwh"),
        tol=1e-2,
        note="mean floor-slot wedge, earlier window",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeMeanPost",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "windows", "post", "mean_yen_kwh"),
        tol=1e-2,
        note="mean floor-slot wedge, later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeMeanDeclinePct",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "mean_decline_pct"),
        tol=0.6,
        note="decline in the mean floor-slot wedge",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePninetyPost",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "windows", "post", "p90_yen_kwh"),
        tol=1e-2,
        note="persistent right tail of the wedge after the decline",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePostInteraction",
        provenance="phase60_wedge_continuous.json",
        json_path=(
            "c4_components",
            "floor_component",
            "coefficients",
            "post_x_exposure_floor",
            "beta",
        ),
        tol=0.1,
        note="post x floor-exposure interaction: penalty weaker in the later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePostInteractionP",
        provenance="phase60_wedge_continuous.json",
        json_path=(
            "c4_components",
            "floor_component",
            "coefficients",
            "post_x_exposure_floor",
            "p",
        ),
        tol=1e-2,
        note="p-value of that interaction (reported, not buried)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePostVarSharePct",
        provenance="phase60_wedge_continuous.json",
        json_path=("identifying_variation", "post_share_of_exposure_variance_pct"),
        tol=0.1,
        note="share of floor-exposure variance in the later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeSlopePre",
        provenance="phase60_wedge_continuous.json",
        json_path=("identifying_variation", "subsample_slope_pre"),
        tol=0.1,
        note="floor-exposure slope, earlier-window subsample",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeSlopePost",
        provenance="phase60_wedge_continuous.json",
        json_path=("identifying_variation", "subsample_slope_post"),
        tol=0.1,
        note="floor-exposure slope, later-window subsample (thin base)",
       paper_b=True,
    ),
    # ---- magnitude twins: same provenance, sign carried by the prose verb ----
    MacroCheck(
        macro="CnpeAbsImbPreAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "annualised",
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of the earlier-window imbalance cut",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPreLoAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "ci95",
            0,
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of that CI's far endpoint",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPreHiAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "pre",
            "d_abs_imbalance_daily_inference",
            "ci95",
            1,
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of that CI's near endpoint",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPostAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "annualised",
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of the later-window imbalance cut",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPostLoAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "ci95",
            0,
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of that CI's far endpoint",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeAbsImbPostHiAbs",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "system_metrics",
            "windows",
            "post",
            "d_abs_imbalance_daily_inference",
            "ci95",
            1,
        ),
        scale=-1.0,
        tol=0.6,
        note="magnitude of that CI's near endpoint",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePreLoAbsK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            0,
        ),
        scale=-1e-3,
        tol=0.6,
        note="magnitude of the CNPE CI's far endpoint",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpePreHiAbsK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "voi",
            "op_vs_base",
            "pooled",
            "bootstrap_ci95_yen_per_mw_yr",
            1,
        ),
        scale=-1e-3,
        tol=0.6,
        note="magnitude of the CNPE CI's near endpoint",
       paper_b=True,
    ),
    # --- bound 2026-07-29 by the unchecked-macro sweep below. Every one of these
    # was printed in the paper with nothing watching it; \CnpePreAbsK is the
    # headline private figure and \WedgeZeroSharePre/Post are the two numbers the
    # Results section leans on to say the wedge decline is about zero-priced
    # slots rather than a level shift. tol=0.6 wherever the macro prints the
    # integer rounding of a fractional provenance value.
    MacroCheck(
        macro="CnpePreAbsK",
        provenance="phase61_system_metrics_band.json",
        json_path=("system_metrics", "windows", "pre", "op_vs_base", "cnpe_yen"),
        scale=-1e-3,
        tol=0.6,
        note="magnitude of the headline CNPE, thousand JPY",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeZeroSharePre",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "windows", "pre", "share_within_005_of_zero_pct"),
        tol=0.6,
        note="floor slots with a near-zero wedge, earlier window (%)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeZeroSharePost",
        provenance="phase60_wedge_continuous.json",
        json_path=("wedge_distribution", "windows", "post", "share_within_005_of_zero_pct"),
        tol=0.6,
        note="floor slots with a near-zero wedge, later window (%)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeRsqFloorPct",
        provenance="phase60_wedge_continuous.json",
        json_path=("c4_components", "floor_component", "r2"),
        scale=100.0,
        tol=0.6,
        note="R^2 of the floor-slot profit component (%)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgeRsqNonfloorPct",
        provenance="phase60_wedge_continuous.json",
        json_path=("c4_components", "nonfloor_component", "r2"),
        scale=100.0,
        tol=0.6,
        note="R^2 of the non-floor profit component (%)",
       paper_b=True,
    ),
    MacroCheck(
        macro="WedgePlaceboSharePct",
        provenance="phase60_wedge_continuous.json",
        json_path=("c8_placebos", "permuted_split", "share_of_random_splits_at_least_as_large"),
        scale=100.0,
        tol=0.6,
        note="arbitrary split dates reproducing the observed two-window gap (%)",
       paper_b=True,
    ),
    # Ratio macros: the quotient is computed here rather than by hand.
    MacroCheck(macro="CurtailRiskRatioLo",
        provenance="phase25.9_floor_probit_summary.json",
        json_path=("floor_x_curtail_crosstab", 6, "risk_ratio_curtail_vs_no"),
        tol=0.6,
        note="smallest floor-vs-curtailment risk ratio across areas; index 6 is the "
             "argmin of this frozen crosstab and the check fails if that changes"),
    MacroCheck(macro="CurtailRiskRatioHi",
        provenance="phase25.9_floor_probit_summary.json",
        json_path=("floor_x_curtail_crosstab", 1, "risk_ratio_curtail_vs_no"),
        tol=0.6, note="largest such ratio; index 1 is the argmax"),
    # ---- Phase 70: the commitment's two channels, from the ACTUAL baseline.
    MacroCheck(macro="ChannelPriceK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all",
                   "channel_decomposition_from_baseline",
                   "price_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6,
        note="curtailment-call leg, FULL sample: the price forecast's OWN "
             "contribution, positive (fifth review made the full sample primary)", paper_b=True),
    MacroCheck(macro="ChannelGenK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all",
                   "channel_decomposition_from_baseline",
                   "generation_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6,
        note="generation-forecast leg, full sample: where the negative sign comes from", paper_b=True),
    MacroCheck(macro="ChannelPricePreK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "pre_collapse",
                   "channel_decomposition_from_baseline",
                   "price_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6, note="price leg, earlier window (S4 per-window split)", paper_b=True),
    MacroCheck(macro="ChannelGenPreK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "pre_collapse",
                   "channel_decomposition_from_baseline",
                   "generation_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6, note="generation leg, earlier window", paper_b=True),
    MacroCheck(macro="ChannelPricePostK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "post_collapse",
                   "channel_decomposition_from_baseline",
                   "price_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6, note="price leg, later window", paper_b=True),
    MacroCheck(macro="ChannelGenPostK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "post_collapse",
                   "channel_decomposition_from_baseline",
                   "generation_forecast_channel_yen_per_mw_yr"),
        scale=1e-3, tol=0.6, note="generation leg, later window", paper_b=True),
    MacroCheck(macro="ChannelGenAbsK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all",
                   "channel_decomposition_from_baseline",
                   "generation_forecast_channel_yen_per_mw_yr"),
        scale=-1e-3, tol=0.6,
        note="generation leg magnitude, full sample, for signed prose", paper_b=True),
    MacroCheck(macro="ChannelInteractionK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all",
                   "channel_decomposition_2x2_pooled", "interaction"),
        scale=1e-3, tol=0.6,
        note="2x2 interaction, full sample: bounds the order-dependence of the "
             "path decomposition", paper_b=True),
    MacroCheck(macro="CurtailFloorN", provenance="phase25.9_floor_probit_summary.json",
        json_path=("floor_x_curtail_crosstab",), aggregate="count", tol=0.0,
        note="areas entering the curtailment-hours vs floor-hours rank correlation; "
             "the macro prints 'n = 8' and the guard counts the crosstab rows"),
    MacroCheck(macro="HeadlinePerAreaLo", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "per_area", "msm_d2_12z_cw"), aggregate="min",
        leaf_key="improvement", tol=6e-4,
        note="lowest per-area improvement of the headline tier, taken as an "
             "aggregate so a re-run that reorders the areas cannot pass silently"),
    MacroCheck(macro="HeadlinePerAreaHi", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "per_area", "msm_d2_12z_cw"), aggregate="max",
        leaf_key="improvement", tol=6e-4,
        note="highest per-area improvement of the headline tier, same construction"),
    # ---- Phase 74: capacity-weighted re-estimation (Gate 35 fired, Gate 36 = 7b) ----
    MacroCheck(macro="CwProxyWins", provenance="phase74_gate35_proxy.json",
        json_path=("cw_wins",), tol=0.0,
        note="areas where the CW solar proxy beats single-point against OCCTO; "
             "Gate 35's threshold was 6 of 8"),
    MacroCheck(macro="CwProxyAreas", provenance="phase74_gate35_proxy.json",
        json_path=("n_areas",), tol=0.0, note="supply-covered areas compared"),
    MacroCheck(macro="CwProxyKyushuSingle", provenance="phase74_gate35_proxy.json",
        json_path=("per_area", "kyushu", "pearson_r_single"), tol=6e-4,
        note="Kyushu single-point proxy r, the largest CW gain"),
    MacroCheck(macro="CwProxyKyushuCw", provenance="phase74_gate35_proxy.json",
        json_path=("per_area", "kyushu", "pearson_r_cw"), tol=6e-4,
        note="Kyushu capacity-weighted proxy r"),
    MacroCheck(macro="CwInteraction", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "measurement_x_lead_interaction", "delta"), tol=6e-5,
        note="the reviewer's conjectured measurement-by-lead interaction"),
    MacroCheck(macro="CwInteractionLo", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "measurement_x_lead_interaction", "ci95", 0), tol=6e-5,
        note="its CI lower"),
    MacroCheck(macro="CwInteractionHi", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "measurement_x_lead_interaction", "ci95", 1), tol=6e-5,
        note="its CI upper"),
    MacroCheck(macro="CwInteractionP", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "measurement_x_lead_interaction", "dm_l3", "p_hln"), tol=6e-3,
        note="its DM p: the interaction is not detectable"),
    MacroCheck(macro="SpPreGateImp", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "tiers", "msm_d2_12z", "improvement"),
               note="single-point sensitivity arm headline"),
    MacroCheck(macro="SpPreGateImpLo", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "tiers", "msm_d2_12z", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="SpPreGateImpHi", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "tiers", "msm_d2_12z", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="SpGateTiming", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "gate50_sp_gate_timing", "delta"),
               note="single-point gate-timing delta (same null)"),
    MacroCheck(macro="SpGateTimingLo", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "gate50_sp_gate_timing", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="SpGateTimingHi", provenance="phase85_exact_ladder.json",
               json_path=("arm_sp_robustness", "gate50_sp_gate_timing", "ci95", 1),
               note="its CI upper"),
    # RETIRED 2026-08-13: the four \ExpChan* checks bound phase79's
    # physical-target numbers. phase89 supersedes them (cross-fitted scale,
    # common panel, per-operator OCCTO stamps) and the manuscript no longer
    # prints them, so the macros are gone from macros.tex and their checks
    # with them. The \Phys* block below is the replacement.
    MacroCheck(macro="VersionInteraction", provenance="phase80_jma_version.json",
               json_path=("gate_timing_by_version", "exact", "interaction",
                          "delta_v2_minus_v1"),
               note="gate-timing x MSM-version interaction (2025-05-20 split)"),
    MacroCheck(macro="VersionInteractionLo", provenance="phase80_jma_version.json",
               json_path=("gate_timing_by_version", "exact", "interaction", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="VersionInteractionHi", provenance="phase80_jma_version.json",
               json_path=("gate_timing_by_version", "exact", "interaction", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="VersionPreV", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "v1_pre_upgrade", "mean"),
               note="pre-gate improvement before the 2025-05-20 upgrade"),
    MacroCheck(macro="VersionPostV", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "v2_post_upgrade", "mean"),
               note="pre-gate improvement after the upgrade"),
    MacroCheck(macro="VersionPreVDays", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "v1_pre_upgrade", "n_days"),
               tol=0.5, note="pre-upgrade test days"),
    MacroCheck(macro="VersionPostVDays", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "v2_post_upgrade", "n_days"),
               tol=0.5, note="post-upgrade test days"),
    MacroCheck(macro="VersionPreGateDelta", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "difference", "delta_v2_minus_v1"),
               note="post minus pre difference in the pre-gate improvement"),
    MacroCheck(macro="VersionPreGateDeltaLo", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "difference", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="VersionPreGateDeltaHi", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "difference", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="VersionPreGateDeltaP", provenance="phase80_jma_version.json",
               json_path=("tiers", "msm_d2_12z_cw", "difference", "p_boot"),
               tol=0.005, note="its bootstrap p-value"),
    MacroCheck(macro="VersionLagMedianV", provenance="phase80_jma_version.json",
               json_path=("dissemination_lag_by_version", "v1", "median"),
               tol=0.06, note="median dissemination lag, pre-upgrade files"),
    MacroCheck(macro="VersionLagMedianPostV", provenance="phase80_jma_version.json",
               json_path=("dissemination_lag_by_version", "v2", "median"),
               tol=0.06, note="median dissemination lag, post-upgrade files"),
    MacroCheck(macro="HeadlineDemandW", provenance="phase82_weighted_metrics.json",
               json_path=("by_weighting", "demand", "msm_d2_12z_cw", "improvement"),
               note="headline under demand weighting"),
    MacroCheck(macro="HeadlinePvW", provenance="phase82_weighted_metrics.json",
               json_path=("by_weighting", "pv_capacity", "msm_d2_12z_cw", "improvement"),
               note="headline under PV-capacity weighting"),
    MacroCheck(macro="PolitisWhiteBlock", provenance="phase83_inference_sensitivity.json",
               json_path=("politis_white_block_days",), tol=0.06,
               note="automatic block length (Politis-White)"),
    MacroCheck(macro="CompositeWorstMarginMin", provenance="phase78_lineage_audit.json",
               json_path=("run_dissemination", "msm_d2_21z", "worst_margin_min"),
               tol=0.06,
               note="freshest admissible cycle's worst measured gate margin, minutes"),

    # ---- legacy cross-area and theory macros of the supplementary sections --
    # These were the guard's long-standing PENDING list: numbers that a phase
    # had produced but nobody had wired up, still printed by S2 and S3. Each is
    # bound below to the JSON path a search of data/runs/ located. Interval
    # macros get TWO checks, one per bound, because the guard passes a macro
    # when ANY of the floats in its body matches -- so binding only the lower
    # bound would leave the upper one unverified.
    MacroCheck(macro="NoiselessSpearman",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("noiseless", "spearman_rho"), tol=0.0005,
               note="noiseless-proxy cross-area Spearman"),
    MacroCheck(macro="NoiselessPermP",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("noiseless", "permutation_p"), tol=0.0005,
               note="its permutation p"),
    MacroCheck(macro="NoiselessBeta",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("noiseless", "beta_FLOOR"), tol=5e-05,
               note="noiseless-proxy panel slope"),
    MacroCheck(macro="NoiselessBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("noiseless", "ci90_lo"), tol=0.0005,
               note="its 90% CI, lower bound"),
    MacroCheck(macro="NoiselessBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("noiseless", "ci90_hi"), tol=0.0005,
               note="its 90% CI, upper bound"),
    MacroCheck(macro="RealisticSpearman",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("realistic", "spearman_rho"), tol=0.0005,
               note="realistic-proxy cross-area Spearman"),
    MacroCheck(macro="RealisticPermP",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("realistic", "permutation_p"), tol=0.0005,
               note="its permutation p"),
    MacroCheck(macro="RealisticBeta",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("realistic", "beta_FLOOR"), tol=5e-05,
               note="realistic-proxy panel slope"),
    MacroCheck(macro="RealisticBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("realistic", "ci90_lo"), tol=0.0005,
               note="its 90% CI, lower bound"),
    MacroCheck(macro="RealisticBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("realistic", "ci90_hi"), tol=0.0005,
               note="its 90% CI, upper bound"),
    MacroCheck(macro="StressSpearman",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("stress", "spearman_rho"), tol=0.0005,
               note="stress-proxy cross-area Spearman"),
    MacroCheck(macro="StressPermP",
               provenance="phase24_noise_sensitivity/cross_area_summary.json",
               json_path=("stress", "permutation_p"), tol=0.001,
               note="its permutation p"),
    MacroCheck(macro="StressBeta",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("stress", "beta_FLOOR"), tol=5e-05,
               note="stress-proxy panel slope"),
    MacroCheck(macro="StressBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("stress", "ci90_lo"), tol=0.0005,
               note="its 90% CI, lower bound"),
    MacroCheck(macro="StressBetaCI",
               provenance="phase24_noise_sensitivity/panel_beta_summary.json",
               json_path=("stress", "ci90_hi"), tol=0.0005,
               note="its 90% CI, upper bound"),
    MacroCheck(macro="BcadmSpearman",
               provenance="phase24_lag96_robustness/cross_area_lag96.json",
               json_path=("spearman_rho",), tol=0.0005,
               note="admissible-baseline cross-area Spearman"),
    MacroCheck(macro="BcadmPermP",
               provenance="phase24_lag96_robustness/cross_area_lag96.json",
               json_path=("permutation_p",), tol=0.001,
               note="its permutation p"),
    MacroCheck(macro="BcadmBeta",
               provenance="phase24_lag96_robustness/panel_beta_lag96.json",
               json_path=("beta_FLOOR",), tol=0.0001,
               note="admissible-baseline panel slope"),
    MacroCheck(macro="BcadmBetaCI",
               provenance="phase24_lag96_robustness/panel_beta_lag96.json",
               json_path=("ci90_lo",), tol=0.0005,
               note="its 90% CI, lower bound"),
    MacroCheck(macro="BcadmBetaCI",
               provenance="phase24_lag96_robustness/panel_beta_lag96.json",
               json_path=("ci90_hi",), tol=0.0005,
               note="its 90% CI, upper bound"),
    MacroCheck(macro="SpearmanPreP",
               provenance="phase25_17_floor_pre_period/results.json",
               json_path=("specifications", 1, "cross_area_floor_vs_delta_mae",
                          "spearman_p"), tol=0.0005,
               note="pre-period floor-rate Spearman p"),
    MacroCheck(macro="SpearmanTrain",
               provenance="phase25_17_floor_pre_period/results.json",
               json_path=("specifications", 2, "cross_area_floor_vs_delta_mae",
                          "spearman_rho"), tol=0.0005,
               note="train-window cross-area Spearman"),
    MacroCheck(macro="SpearmanTrainP",
               provenance="phase25_17_floor_pre_period/results.json",
               json_path=("specifications", 2, "cross_area_floor_vs_delta_mae",
                          "spearman_p"), tol=0.0005,
               note="its p"),
    MacroCheck(macro="PanelAsymCI",
               provenance="phase23-panel-daily/panel_summary.json",
               json_path=("spec1_area_main", "ci_90", "floor_rate", 0),
               tol=0.0005, note="asymptotic panel CI, lower bound"),
    MacroCheck(macro="PanelAsymCI",
               provenance="phase23-panel-daily/panel_summary.json",
               json_path=("spec1_area_main", "ci_90", "floor_rate", 1),
               tol=0.0005, note="asymptotic panel CI, upper bound"),
    MacroCheck(macro="Rsq",
               provenance="phase23_daily_renewable_regression_summary.json",
               json_path=("multivariate_floor_pv_wind", "r2"), tol=0.005,
               note="cross-area regression R^2 (FLOOR + PV + wind)"),
    MacroCheck(macro="Rsqweekly", provenance="phase22_statistical_rigor.json",
               json_path=("regression_point_fit", "r2"), tol=0.005,
               note="weekly cross-area regression R^2"),
    MacroCheck(macro="Rsqfloor", provenance="phase25.7_pv_peak_summary.json",
               json_path=("univariate_floor_rate", "r_squared"), tol=0.005,
               note="floor-rate regression R^2"),
    MacroCheck(macro="OpsCaptureOracle", provenance="phase27.1_oracle_tier.json",
               json_path=("mean_improvement_yen_kwh", "histfc"),
               denominator_json_path=("mean_improvement_yen_kwh", "oracle"),
               scale=100.0, tol=0.5,
               note="share of the ex-post tier's mean value the operational "
                    "forecast captures (bound as the division, not a re-typed "
                    "quotient)"),
    MacroCheck(macro="ProxyCaptureOracle", provenance="phase27.1_oracle_tier.json",
               json_path=("mean_improvement_yen_kwh", "era5"),
               denominator_json_path=("mean_improvement_yen_kwh", "oracle"),
               scale=100.0, tol=0.5,
               note="same, for the stale proxy"),
    MacroCheck(macro="CurtailActiveP", provenance="phase27.0_derisk_gate.json",
               json_path=("probe_b_threshold_gate", "curtail_histfc_intensity",
                          "p_asymptotic_cluster_t"), tol=5e-06,
               note="curtailment-active intensity p (asymptotic cluster-t); the "
                    "macro used to print the inequality 'p < 0.001', which no "
                    "equality check can guard -- a drift to 0.0015 would have "
                    "left the printed claim false and the check passing"),
    MacroCheck(macro="CurtailActiveBeta", provenance="phase27.0_derisk_gate.json",
               json_path=("probe_b_threshold_gate", "curtail_histfc_intensity",
                          "beta_observed"), tol=0.005,
               note="its slope"),
    # ---- second-provider check (phase86, NCEP GFS; §5.6 and §7) ------------
    MacroCheck(macro="GfsPreGateImp", provenance="phase86_gfs_tier.json",
               json_path=("tiers", "gfs_d2_12z", "improvement"),
               note="GFS pre-gate run's improvement over the price-only "
                    "baseline, single-point arm, same folds as the ladder"),
    MacroCheck(macro="GfsPreGateImpLo", provenance="phase86_gfs_tier.json",
               json_path=("tiers", "gfs_d2_12z", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="GfsPreGateImpHi", provenance="phase86_gfs_tier.json",
               json_path=("tiers", "gfs_d2_12z", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="GfsJmaMinusGfs", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q2_jma_minus_gfs_same_init", "delta"),
               note="JMA minus GFS at the same initialisation: a provider "
                    "difference, never a vintage one"),
    MacroCheck(macro="GfsJmaMinusGfsLo", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q2_jma_minus_gfs_same_init", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="GfsJmaMinusGfsHi", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q2_jma_minus_gfs_same_init", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="GfsJmaMinusGfsP", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q2_jma_minus_gfs_same_init", "p_l3"),
               tol=5e-05, note="its DM p at L=3"),
    MacroCheck(macro="GfsVintageStep", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q3_within_gfs_vintage_step", "delta"),
               note="the within-GFS step from D-2 12 UTC to its freshest "
                    "pre-gate cycle: the JMA ladder's flatness, reproduced"),
    MacroCheck(macro="GfsVintageStepLo", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q3_within_gfs_vintage_step", "ci95", 0),
               note="its CI lower"),
    MacroCheck(macro="GfsVintageStepHi", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q3_within_gfs_vintage_step", "ci95", 1),
               note="its CI upper"),
    MacroCheck(macro="GfsVintageStepP", provenance="phase86_gfs_tier.json",
               json_path=("questions", "q3_within_gfs_vintage_step", "p_l3"),
               tol=0.005, note="its DM p at L=3"),
    # ---- direct floor models (phase81, §4.5 and §5.4) ---------------------
    MacroCheck(macro="FloorDirectBest", provenance="phase81_direct_floor_models.json",
               json_path=("results", "allhours_thr0.5", "m2", "macro_pr_auc"),
               tol=0.0005,
               note="winner of the five pre-registered floor models (direct "
                    "classifier), macro PR-AUC at the near-floor threshold"),
    MacroCheck(macro="FloorDirectIsotonic",
               provenance="phase81_direct_floor_models.json",
               json_path=("results", "allhours_thr0.5", "m0", "macro_pr_auc"),
               tol=0.0005, note="the isotonic recalibration chain, same metric"),
    MacroCheck(macro="FloorDirectPriceOnly",
               provenance="phase81_direct_floor_models.json",
               json_path=("results", "allhours_thr0.5", "m2p", "macro_pr_auc"),
               tol=0.0005,
               note="the same classifier with the weather withheld"),
    MacroCheck(macro="FloorDirectWeatherShare",
               provenance="phase81_direct_floor_models.json",
               json_path=("results", "allhours_thr0.5", "m2p", "macro_pr_auc"),
               denominator_json_path=("results", "allhours_thr0.5", "m2",
                                      "macro_pr_auc"),
               complement=True, scale=100.0, tol=0.5,
               note="share of the direct model's ranking skill that the weather "
                    "channels carry, bound as 1 - price_only/full rather than as "
                    "a re-typed percentage"),
    # ---- physical target, cross-fitted (phase89) --------------------------
    MacroCheck(macro="PhysDelta", provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_physical_gain_crossfit_common",
                          "delta_mw"), tol=0.05,
               note="post-gate physical gain, cross-fitted scale, common panel; "
                    "a PER-AREA mean in MW, not a fleet total"),
    MacroCheck(macro="PhysDeltaLo", provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_physical_gain_crossfit_common",
                          "ci95", 0), tol=0.05, note="its CI lower"),
    MacroCheck(macro="PhysDeltaHi", provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_physical_gain_crossfit_common",
                          "ci95", 1), tol=0.05, note="its CI upper"),
    MacroCheck(macro="PhysDeltaDays", provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_physical_gain_crossfit_common",
                          "n_days"), tol=0.5, note="common-panel days"),
    MacroCheck(macro="PhysPreGateMae", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "msm_d2_12z_cw",
                          "pooled_mae_cf_common_mw"), tol=0.5,
               note="pre-gate physical MAE on the common panel"),
    MacroCheck(macro="PhysPostGateMae", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "msm_d1_00z_cw",
                          "pooled_mae_cf_common_mw"), tol=0.5,
               note="post-gate physical MAE on the common panel"),
    MacroCheck(macro="PhysStaleMae", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "era5_cw_stale",
                          "pooled_mae_cf_common_mw"), tol=0.5,
               note="stale-proxy physical MAE"),
    MacroCheck(macro="PhysOracleMae", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "oracle_cw",
                          "pooled_mae_cf_common_mw"), tol=0.5,
               note="ex-post realised physical MAE"),
    MacroCheck(macro="PhysLookaheadCost",
               provenance="phase89_physical_target.json",
               json_path=("gates", "gate53_lookahead_cost", "difference_mw"),
               tol=0.05,
               note="what the superseded in-sample scale was worth: the "
                    "cross-fitted error is this much higher"),
    MacroCheck(macro="PhysStampCost", provenance="phase89_physical_target.json",
               json_path=("gates", "gate54_stamp_convention", "difference_mw"),
               scale=-1.0, tol=0.5,
               note="what the uniform -30 min OCCTO shift was costing: the "
                    "per-area convention is this much LOWER in error. Sign "
                    "flipped because the JSON records (new - old)"),
    MacroCheck(macro="PhysDeltaExcl", provenance="phase89_physical_target.json",
               json_path=("gates", "gate52_curtailment_sign_stability",
                          "delta_by_target", "curtail_excluded"), tol=0.05,
               note="the same gain with curtailed slots excluded"),
    MacroCheck(macro="PhysDeltaRecorded",
               provenance="phase89_physical_target.json",
               json_path=("gates", "gate52_curtailment_sign_stability",
                          "delta_by_target", "recorded"), tol=0.05,
               note="the same gain against raw recorded output"),
    MacroCheck(macro="PhysLatestMae", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers",
                          "msm_gate_composite_cw", "pooled_mae_cf_common_mw"),
               tol=0.6, note="freshest-admissible physical MAE, common panel"),
    MacroCheck(macro="PhysDeltaWide", provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_sensitivity_wide_panel", "delta_mw"),
               tol=0.05,
               note="the same gain on the wider physical-only panel: reported so "
                    "the common-panel restriction is visible as a choice"),
    MacroCheck(macro="PhysDeltaWideDays",
               provenance="phase89_physical_target.json",
               json_path=("gates", "gate51_sensitivity_wide_panel", "n_days"),
               tol=0.5, note="days on the wider physical-only panel"),
    MacroCheck(macro="PhysMinTrain", provenance="phase89_physical_target.json",
               json_path=("conventions", "min_train_rows"), tol=0.5,
               note="scored slots the expanding window needs before a day is "
                    "scored rather than treated as burn-in"),
    MacroCheck(macro="PhysBootDraws", provenance="phase89_physical_target.json",
               json_path=("metadata", "n_boot"), tol=0.5,
               note="block bootstrap draws behind the physical intervals"),
    # Per-area extremes: bound by aggregate rather than by area name, so a
    # re-run that reorders the areas cannot keep passing on a stale label.
    MacroCheck(macro="PhysPerAreaLo", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "msm_d2_12z_cw",
                          "per_area"), aggregate="min", leaf_key="mae_cf_mw",
               aggregate_key="hokuriku", tol=0.6,
               note="smallest per-area physical MAE at the pre-gate vintage, MW "
                    "-- the low end of the range the per-area mean averages"),
    MacroCheck(macro="PhysPerAreaHi", provenance="phase89_physical_target.json",
               json_path=("targets", "potential", "tiers", "msm_d2_12z_cw",
                          "per_area"), aggregate="max", leaf_key="mae_cf_mw",
               aggregate_key="tokyo", tol=0.6,
               note="and the high end. The area NAMES in the sentence are "
                    "asserted too: a re-run that moves the extremum to another "
                    "operator within tolerance would otherwise keep the check "
                    "green on prose crediting the wrong one"),
    # ---- absolute performance (phase82b) ----------------------------------
    MacroCheck(macro="AbsPanelDays", provenance="phase82b_absolute_metrics.json",
               json_path=("design", "subpanel_days"), tol=0.5,
               note="distinct test days the naive-benchmark sub-panel covers, "
                    "MEASURED. It was bound to panel.days minus "
                    "naive_unavailable_days, which is 718 and describes nothing: "
                    "the benchmark filter is on SLOTS, and four days keep some "
                    "of theirs, so the sub-panel spans 722 days. The subtraction "
                    "was checkable arithmetic about the wrong quantity"),
    # ---- enriched-baseline decomposition (phase54v2) -----------------------
    MacroCheck(macro="EnrichedSlots",
               provenance="phase54v2_current.json",
               json_path=('window', 'n_obs'), tol=0.5,
               note="slot observations in the enriched-baseline window; the prose said 234,080, which is the SUPERSEDED phase54 run's count"),
    MacroCheck(macro="EnrichGainAdmLo",
               provenance="phase54v2_current.json",
               json_path=('results', 'deltas_daily_mean_with_ci95', 'd21_enrichment_gain', 'lo95'), tol=0.0006,
               note="enrichment gain CI lower (was hand-typed from phase54)"),
    MacroCheck(macro="EnrichGainAdmHi",
               provenance="phase54v2_current.json",
               json_path=('results', 'deltas_daily_mean_with_ci95', 'd21_enrichment_gain', 'hi95'), tol=0.0006,
               note="and its upper"),
    MacroCheck(macro="ProxyCarriedLo",
               provenance="phase54v2_current.json",
               json_path=('results', 'deltas_daily_mean_with_ci95', 'd32_proxies_over_enriched', 'lo95'), tol=0.0006,
               note="proxy-carried term CI lower (was hand-typed from phase54: [0.136, 0.197])"),
    MacroCheck(macro="ProxyCarriedHi",
               provenance="phase54v2_current.json",
               json_path=('results', 'deltas_daily_mean_with_ci95', 'd32_proxies_over_enriched', 'hi95'), tol=0.0006,
               note="and its upper"),
    MacroCheck(macro="EnrichPerAreaLo", provenance="phase54v2_current.json",
               json_path=("results", "per_area"), aggregate="min",
               leaf_key="d42", aggregate_key="hokkaido", tol=0.0006,
               note="smallest per-area total value over the enriched baseline; "
                    "the prose said 0.069, the superseded run's value"),
    MacroCheck(macro="EnrichPerAreaHi", provenance="phase54v2_current.json",
               json_path=("results", "per_area"), aggregate="max",
               leaf_key="d42", aggregate_key="kyushu", tol=0.0006,
               note="and the largest; both area names are asserted"),
    MacroCheck(macro="AbsPanelSlots", provenance="phase82b_absolute_metrics.json",
               json_path=("design", "subpanel_slots"), tol=0.5,
               note="slots in that sub-panel; the number every relative figure "
                    "in Table 6 is computed on"),
    MacroCheck(macro="AbsNaiveMae", provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "all", "naive168", "mae"), tol=0.005,
               note="naive 168-hour benchmark MAE, all slots"),
    MacroCheck(macro="AbsPreGateMae", provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "all", "tiers", "msm_d2_12z_cw", "mae"),
               tol=0.005, note="pre-gate absolute MAE on the same sub-panel"),
    MacroCheck(macro="AbsPreGateRmae", provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "all", "tiers", "msm_d2_12z_cw",
                          "rmae_vs_naive168"), tol=0.005,
               note="the same relative to the naive benchmark: the EPF-standard "
                    "denominator, not this paper's own baseline"),
    MacroCheck(macro="AbsFloorNaiveMae",
               provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "floor", "naive168", "mae"), tol=0.005,
               note="naive MAE inside the near-floor band"),
    MacroCheck(macro="AbsFloorBaselineMae",
               provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "floor", "price_only", "mae"), tol=0.005,
               note="price-only MAE inside the near-floor band"),
    MacroCheck(macro="AbsFloorPreGateMae",
               provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "floor", "tiers", "msm_d2_12z_cw", "mae"),
               tol=0.005,
               note="pre-gate MAE inside the near-floor band: the largest "
                    "absolute weather gain in the paper"),
    MacroCheck(macro="FloorAbsGain",
               provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "floor", "price_only", "mae"),
               subtract_json_path=("relative", "floor", "tiers",
                                   "msm_d2_12z_cw", "mae"), tol=0.005,
               note="the pre-gate vintage's absolute MAE gain inside the "
                    "near-floor band: price-only minus pre-gate, subtracted by "
                    "the guard"),
    # ---- floor-model paired inference (phase81b) --------------------------
    MacroCheck(macro="FloorWinnerMargin",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0', 'delta'), tol=0.001,
               note="direct classifier over the isotonic chain, macro PR-AUC, paired block bootstrap"),
    MacroCheck(macro="FloorWinnerMarginLo",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0', 'ci95', 0), tol=0.001,
               note="its CI lower"),
    MacroCheck(macro="FloorWinnerMarginHi",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0', 'ci95', 1), tol=0.001,
               note="its CI upper"),
    MacroCheck(macro="FloorWinnerHolmP",
               provenance="phase81b_floor_inference.json",
               json_path=('multiplicity', 'holm_adjusted_p', 'm2_minus_m0'),
               tol=0.00006,
               note="Holm-adjusted p across the family of five floor models"),
    MacroCheck(macro="FloorWeatherMargin",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m2p', 'delta'), tol=0.001,
               note="what the weather adds inside the winning classifier"),
    MacroCheck(macro="FloorWeatherMarginLo",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m2p', 'ci95', 0), tol=0.001,
               note="its CI lower"),
    MacroCheck(macro="FloorWeatherMarginHi",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m2p', 'ci95', 1), tol=0.001,
               note="its CI upper"),
    # ---- price target on the physical common panel (phase89b) ---------------
    MacroCheck(macro="CommonPriceImpPre",
               provenance="phase89b_price_on_common_panel.json",
               json_path=("tiers", "msm_d2_12z_cw", "improvement"), tol=0.0006,
               note="pre-gate price improvement restricted to the physical common panel; Figure 5's panel (a) shows this, not the full-panel ladder value"),
    MacroCheck(macro="CommonPriceImpPost",
               provenance="phase89b_price_on_common_panel.json",
               json_path=("tiers", "msm_d1_00z_cw", "improvement"), tol=0.0006,
               note="the excluded run, same panel"),
    MacroCheck(macro="CommonPriceImpStale",
               provenance="phase89b_price_on_common_panel.json",
               json_path=("tiers", "era5_cw", "improvement"), tol=0.0006,
               note="the stale proxy, same panel"),
    MacroCheck(macro="FloorIsotonicTiesCal",
               provenance="phase81b_floor_inference.json",
               json_path=('score_resolution', 'm0'), tol=0.5,
               note="distinct values the daily-refitted calibrated probability takes; the mechanism behind two PR-AUC figures for one chain"),
    MacroCheck(macro="FloorIsotonicTiesRaw",
               provenance="phase81b_floor_inference.json",
               json_path=('score_resolution', 'm0rank'), tol=0.5,
               note="and the forecast underneath it, which is 14x finer"),
    MacroCheck(macro="FloorStageTwoGap",
               provenance="phase81b_floor_inference.json",
               json_path=('stage2_vs_stage1_mean_abs_diff',), tol=0.0006,
               note="measured difference between the post-gate and pre-gate classifier outputs: the check that the weather-source rebinding took effect, so the gate contrast is not a bug reported as a null"),
    MacroCheck(macro="FloorRankMargin",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0rank', 'delta'), tol=0.0006,
               note="classifier over the isotonic chain on the LIKE-FOR-LIKE ranking; the calibrated-probability contrast (0.056) charges the chain for its own calibrator's ties"),
    MacroCheck(macro="FloorRankMarginLo",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0rank', 'ci95', 0), tol=0.0006,
               note="its CI lower -- BELOW zero, which is why the winner claim was withdrawn"),
    MacroCheck(macro="FloorRankMarginHi",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0rank', 'ci95', 1), tol=0.0006,
               note="its CI upper"),
    MacroCheck(macro="FloorRankMarginP",
               provenance="phase81b_floor_inference.json",
               json_path=('paired', 'm2_minus_m0rank', 'p_boot'), tol=0.0006,
               note="its bootstrap p"),
    MacroCheck(macro="FloorIsotonicRank",
               provenance="phase81b_floor_inference.json",
               json_path=('metrics', 'm0rank', 'macro_pr_auc'), tol=0.0006,
               note="the isotonic chain scored by the ranking it actually induces"),
    MacroCheck(macro="FloorBrierIsotonic",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'levels', 'm0', 'brier'), tol=6e-05,
               note="isotonic chain, Brier"),
    MacroCheck(macro="FloorBrierDirect",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'levels', 'm2', 'brier'), tol=6e-05,
               note="direct classifier, Brier"),
    MacroCheck(macro="FloorBrierMarginP",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'contrasts', 'm2_minus_m0_brier', 'p_boot'), tol=0.0006,
               note="their Brier contrast's p: also not distinguishable"),
    MacroCheck(macro="FloorLogIsotonic",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'levels', 'm0', 'log_score'), tol=6e-05,
               note="isotonic chain, log score"),
    MacroCheck(macro="FloorLogDirect",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'levels', 'm2', 'log_score'), tol=6e-05,
               note="direct classifier, log score"),
    MacroCheck(macro="FloorLogMargin",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'contrasts', 'm2_minus_m0_log', 'delta'), tol=6e-05,
               note="their log-score contrast: the one instrument that separates the two model classes"),
    MacroCheck(macro="FloorLogMarginLo",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'contrasts', 'm2_minus_m0_log', 'ci95', 0), tol=6e-05,
               note="its CI lower"),
    MacroCheck(macro="FloorLogMarginHi",
               provenance="phase81b_floor_inference.json",
               json_path=('proper_scores', 'contrasts', 'm2_minus_m0_log', 'ci95', 1), tol=6e-05,
               note="its CI upper"),
    MacroCheck(macro="FloorGateContrastDirect",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm2post_minus_m2', 'delta'), tol=0.0006,
               note="the excluded run under the DIRECT CLASSIFIER, retrained on post-gate weather (stage 2)"),
    MacroCheck(macro="FloorGateContrastDirectLo",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm2post_minus_m2', 'ci95', 0), tol=0.0006,
               note="its CI lower"),
    MacroCheck(macro="FloorGateContrastDirectHi",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm2post_minus_m2', 'ci95', 1), tol=0.0006,
               note="its CI upper"),
    MacroCheck(macro="FloorSlotsAreaLo",
               provenance="phase81b_floor_inference.json",
               json_path=("metrics", "m2", "per_area"), aggregate="min",
               leaf_key="n_pos", aggregate_key="tokyo", tol=0.5,
               note="fewest floor positives any area contributes; the caption "
                    "said 'a few hundred' for all nine, understating the "
                    "larger areas by up to sevenfold"),
    MacroCheck(macro="FloorSlotsAreaHi",
               provenance="phase81b_floor_inference.json",
               json_path=("metrics", "m2", "per_area"), aggregate="max",
               leaf_key="n_pos", aggregate_key="shikoku", tol=0.5,
               note="and the most"),
    MacroCheck(macro="FloorGateContrast",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm0post_minus_m0', 'delta'), tol=0.001,
               note="the excluded run under the isotonic chain: the floor gate contrast"),
    MacroCheck(macro="FloorGateContrastLo",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm0post_minus_m0', 'ci95', 0), tol=0.001,
               note="its CI lower (below zero)"),
    MacroCheck(macro="FloorGateContrastHi",
               provenance="phase81b_floor_inference.json",
               json_path=('gate', 'm0post_minus_m0', 'ci95', 1), tol=0.001,
               note="its CI upper"),
    MacroCheck(macro="FloorDirectBestMicro",
               provenance="phase81b_floor_inference.json",
               json_path=('metrics', 'm2', 'micro_pr_auc'), tol=0.001,
               note="the winner, slot-pooled rather than area-averaged"),
    MacroCheck(macro="FloorDirectIsotonicMicro",
               provenance="phase81b_floor_inference.json",
               json_path=('metrics', 'm0', 'micro_pr_auc'), tol=0.001,
               note="the isotonic chain, same"),
    MacroCheck(macro="FloorDirectBestSkill",
               provenance="phase81b_floor_inference.json",
               json_path=('metrics', 'm2', 'macro_ap_skill'), tol=0.001,
               note="the winner, prevalence-adjusted: (AP - base rate) / (1 - base rate)"),
    MacroCheck(macro="FloorDirectIsotonicSkill",
               provenance="phase81b_floor_inference.json",
               json_path=('metrics', 'm0', 'macro_ap_skill'), tol=0.001,
               note="the isotonic chain, same"),
    # Bound against the SAME baseline the manuscript prints. It used to divide by
    # a literal 2.153 frozen in this file -- phase82's full-panel baseline --
    # while \PriceBaselineMae prints 2.142 from phase82b's sub-panel, so the note
    # named a denominator the arithmetic did not use and a reader recomputing the
    # percentage got 0.47 where the macro says 0.46. Both operands now come from
    # provenance, in the same file, so neither can drift unnoticed.
    MacroCheck(macro="GateTimingMarginPctBaseline",
               provenance="phase82b_absolute_metrics.json",
               json_path=("design", "equivalence_margin_jpykwh"),
               denominator_json_path=("relative", "all", "price_only", "mae"),
               scale=100.0, tol=0.006,
               note="the one-tick equivalence margin as a percentage of the "
                    "price-only baseline MAE, both read from the same "
                    "provenance as \\PriceBaselineMae"),
    MacroCheck(macro="PolitisWhiteBlock",
               provenance="phase83_inference_sensitivity.json",
               json_path=("politis_white_block_days",), tol=0.05,
               note="data-driven block length the stationary bootstrap selects"),
    MacroCheck(macro="PriceBaselineMae",
               provenance="phase82b_absolute_metrics.json",
               json_path=("relative", "all", "price_only", "mae"), tol=0.005,
               note="price-only baseline MAE on the absolute-performance "
                    "sub-panel -- the same cell Table 6 prints, so the prose and "
                    "the table cannot drift apart. The full-panel figure (2.153, "
                    "phase82) is what the margin table's percentage divides by, "
                    "and that table prints it from its own provenance"),
    # ---- mechanism test (phase90) -----------------------------------------
    MacroCheck(macro="MechPhysR", provenance="phase90_mechanism.json",
               json_path=("regressions", "physical_residual_on_innovation",
                          "r2_increment"), scale=100.0, tol=0.05,
               note="share of the pre-gate PHYSICAL residual variance the "
                    "post-gate innovation explains, %"),
    MacroCheck(macro="MechPlaceboPhysP",
        provenance="phase90_mechanism.json",
        json_path=("regressions", "placebo_physical", "p_two_sided"),
        note="the physical placebo p. A pre-registered void condition for the "
             "whole mechanism table, and nothing was checking it; it reads "
             "0.051 once the doubled-p bug is fixed"),
    MacroCheck(macro="MechPlaceboPriceP",
        provenance="phase90_mechanism.json",
        json_path=("regressions", "placebo_price", "p_two_sided"),
        note="the price placebo p, comfortably null"),
    MacroCheck(macro="MechPriceR", provenance="phase90_mechanism.json",
               json_path=("regressions", "price_residual_on_innovation",
                          "r2_increment"), scale=100.0, tol=0.05,
               note="the same for the PRICE residual: an order of magnitude "
                    "smaller, but not zero -- which is why the paper reports a "
                    "magnitude asymmetry rather than a mechanism"),
    MacroCheck(macro="MechPriceSlope", provenance="phase90_mechanism.json",
               json_path=("regressions", "price_residual_on_innovation", "slope"),
               tol=0.005, note="JPY/kWh per standard deviation of the innovation"),
    MacroCheck(macro="MechInnovSd", provenance="phase90_mechanism.json",
               json_path=("design", "innovation_sd_mw"), tol=0.5,
               note="one standard deviation of the innovation, MW"),
    # ---- capacity-weight sensitivity (phase91) -----------------------------
    MacroCheck(macro="WeightHeadlineBase",
               provenance="phase91_weight_sensitivity.json",
               json_path=("arms", "2024-09", "tiers", "msm_d2_12z_cw",
                          "improvement"), tol=0.0002,
               note="headline under the paper's own weights; Gate 60 asserts "
                    "this equals the published ladder, so a mismatch here means "
                    "the sensitivity harness has drifted from phase85"),
    MacroCheck(macro="WeightHeadlineLater",
               provenance="phase91_weight_sensitivity.json",
               json_path=("arms", "2025-12", "tiers", "msm_d2_12z_cw",
                          "improvement"), tol=0.0002,
               note="the same under a later register snapshot"),
    MacroCheck(macro="WeightHeadlineEqual",
               provenance="phase91_weight_sensitivity.json",
               json_path=("arms", "equal", "tiers", "msm_d2_12z_cw",
                          "improvement"), tol=0.0002,
               note="the same with the capacity register discarded entirely"),
    MacroCheck(macro="WeightEquivPBase",
               provenance="phase91_weight_sensitivity.json",
               json_path=('arms', '2024-09', 'equivalence_at_tick', 'p_tost'), tol=0.0006,
               note="one-tick TOST p under the paper's own weights"),
    MacroCheck(macro="WeightEquivPLater",
               provenance="phase91_weight_sensitivity.json",
               json_path=('arms', '2025-12', 'equivalence_at_tick', 'p_tost'), tol=0.0006,
               note="and under the later register snapshot, where it EXCEEDS 0.05 -- the claim does not hold there"),
    MacroCheck(macro="WeightEquivPEqual",
               provenance="phase91_weight_sensitivity.json",
               json_path=('arms', 'equal', 'equivalence_at_tick', 'p_tost'), tol=0.0006,
               note="and with the register discarded, where it holds more strongly"),
    MacroCheck(macro="WeightGateDeltaLater",
               provenance="phase91_weight_sensitivity.json",
               json_path=('arms', '2025-12', 'gate_timing_delta', 'delta'), tol=0.0002,
               note="the excluded run's marginal improvement under the later snapshot"),
    MacroCheck(macro="WeightHeadlineSpread",
               provenance="phase91_weight_sensitivity.json",
               json_path=("gates", "gate59_weight_sensitivity",
                          "headline_spread"), tol=0.0002,
               note="max minus min headline across the three weightings"),
    MacroCheck(macro="MechImpliedMaeFull", provenance="phase90_mechanism.json",
               json_path=('implied_price_value_full_panel', 'implied_mae_reduction'), tol=0.0006,
               note="the like-for-like ceiling: the same quantity on the panel the ladder scores, which is what the equivalence margin and the gate delta are measured on"),
    MacroCheck(macro="MechCeilingPrecise",
        provenance="phase90_mechanism.json",
        json_path=("realised_vs_ceiling", "ceiling_full_panel"), tol=6e-5,
        note="the ceiling at four decimals, so Section 5's ordering of three "
             "numbers is visible on the page rather than collapsing to one"),
    MacroCheck(macro="MechCeilingOls",
        provenance="phase90_mechanism.json",
        json_path=("implied_price_value_full_panel", "ols_mae_reduction"),
        note="what a LEAST-SQUARES projection of the innovation reaches. The "
             "paper reported this as the ceiling until 2026-08-18; an L2 fit "
             "is one linear use, not the best one in mean absolute error"),
    MacroCheck(macro="MechImpliedTicksFull", provenance="phase90_mechanism.json",
               json_path=('implied_price_value_full_panel', 'in_ticks'), tol=0.02,
               note="the same in price ticks"),
    MacroCheck(macro="MechInnovSlotShare", provenance="phase90_mechanism.json",
               json_path=('implied_price_value_full_panel', 'share_of_slots_covered_pct'), tol=0.06,
               note="share of the ladder's slots on which the innovation is non-zero; it is a photovoltaic quantity"),
    MacroCheck(macro="MechImpliedMae", provenance="phase90_mechanism.json",
               json_path=("implied_price_value", "implied_mae_reduction"),
               tol=0.0006,
               note="in-sample ceiling on the pooled MAE reduction a linear use "
                    "of the innovation would give -- about one tick, which is "
                    "why the manuscript does NOT say the share is too small to "
                    "move a tick"),
    MacroCheck(macro="MechImpliedTicks", provenance="phase90_mechanism.json",
               json_path=("implied_price_value", "in_ticks"), tol=0.02,
               note="the same, in JEPX price ticks"),
    MacroCheck(macro="MechSlots", provenance="phase90_mechanism.json",
               json_path=("design", "n_slots"), tol=0.5, note="slots in the test"),
    MacroCheck(macro="MechClusters", provenance="phase90_mechanism.json",
               json_path=("design", "n_clusters"), tol=0.5,
               note="area-by-day clusters behind the standard errors"),
    # ---- audit reliability (phase88b, supp. S7) ---------------------------
    MacroCheck(macro="LitCoderPapers", provenance="phase88b_audit_reliability.json",
               json_path=("n_papers_multicoded",), tol=0.5,
               note="papers coded more than once by independent agents"),
    MacroCheck(macro="LitCoderPairs", provenance="phase88b_audit_reliability.json",
               json_path=("n_column_pairs",), tol=0.5,
               note="column-level coding pairs behind the agreement statistics"),
    MacroCheck(macro="LitAgreement", provenance="phase88b_audit_reliability.json",
               json_path=("agreement", "observed"), scale=100.0, tol=0.05,
               note="raw agreement on the six-label verdict vocabulary, %"),
    MacroCheck(macro="LitKappa", provenance="phase88b_audit_reliability.json",
               json_path=("agreement", "cohens_kappa"), tol=0.005,
               note="Cohen's kappa on the same six labels"),
    MacroCheck(macro="LitYesFlipPct", provenance="phase88b_audit_reliability.json",
               json_path=("yes_vs_not_yes", "flip_rate"), scale=100.0, tol=0.05,
               note="share of coding pairs that disagree about whether a "
                    "statement is PRESENT -- the only disagreements that can "
                    "move a reported rate, %"),
    MacroCheck(macro="LitYesFlipN", provenance="phase88b_audit_reliability.json",
               json_path=("yes_vs_not_yes", "n_flips"), tol=0.5,
               note="the count behind it"),
    MacroCheck(macro="LitYesFlipKappa",
               provenance="phase88b_audit_reliability.json",
               json_path=("yes_vs_not_yes", "cohens_kappa"), tol=0.005,
               note="kappa on the binary Yes / not-Yes coding"),
    # ---- literature audit of weather-input timing (phase88, supp. S7) ------
    MacroCheck(macro="LitScreened", provenance="phase88_lit_audit.json",
               json_path=("n_screened",), tol=0.5,
               note="records screened for the timing audit"),
    MacroCheck(macro="LitIncluded", provenance="phase88_lit_audit.json",
               json_path=("n_included",), tol=0.5,
               note="papers coded against published text"),
    MacroCheck(macro="LitExcluded", provenance="phase88_lit_audit.json",
               json_path=("n_excluded",), tol=0.5, note="excluded on scope"),
    MacroCheck(macro="LitUnresolved", provenance="phase88_lit_audit.json",
               json_path=("n_unresolved",), tol=0.5,
               note="in scope but no readable text; excluded from denominators"),
    MacroCheck(macro="LitFullText", provenance="phase88_lit_audit.json",
               json_path=("access_levels", "full-text"), tol=0.5,
               note="coded from the full text"),
    MacroCheck(macro="LitPartial", provenance="phase88_lit_audit.json",
               json_path=("access_levels", "partial"), tol=0.5,
               note="coded from partial text"),
    MacroCheck(macro="LitAbstractOnly", provenance="phase88_lit_audit.json",
               json_path=("access_levels", "abstract-only"), tol=0.5,
               note="coded at abstract level"),
    MacroCheck(macro="LitGateTimePct", provenance="phase88_lit_audit.json",
               json_path=("rates", "states_gate_time", "share"),
               scale=100.0, tol=0.5, note="state the gate-closure time, %"),
    MacroCheck(macro="LitRunIdPct", provenance="phase88_lit_audit.json",
               json_path=("rates", "identifies_model_run", "share"),
               scale=100.0, tol=0.5, note="identify the model run, %"),
    MacroCheck(macro="LitInitPct", provenance="phase88_lit_audit.json",
               json_path=("rates", "states_initialisation", "share"),
               scale=100.0, tol=0.5, note="state the initialisation time, %"),
    MacroCheck(macro="LitPubTimePct", provenance="phase88_lit_audit.json",
               json_path=("rates", "states_publication_time", "share"),
               scale=100.0, tol=0.5, note="state the publication time, %"),
    MacroCheck(macro="LitAdmissiblePct", provenance="phase88_lit_audit.json",
               json_path=("rates", "checks_gate_admissibility", "share"),
               scale=100.0, tol=0.5,
               note="verify pre-gate availability, %"),
    MacroCheck(macro="LitRevisionPct", provenance="phase88_lit_audit.json",
               json_path=("rates", "handles_revisions", "share"),
               scale=100.0, tol=0.5, note="handle input revisions, %"),
    MacroCheck(macro="LitAdmissibleN", provenance="phase88_lit_audit.json",
               json_path=("rates", "checks_gate_admissibility", "n_positive"),
               tol=0.5, note="the two papers that check admissibility"),
    MacroCheck(macro="LitRevisionNotStated", provenance="phase88_lit_audit.json",
               json_path=("per_column_verdicts", "revision_handling", "Not stated"),
               tol=0.5, note="papers that never mention input revisions"),
    MacroCheck(macro="LitRecordsCollected", provenance="phase88_lit_audit.json",
               json_path=("deduplication", "n_records_collected"), tol=0.5,
               note="records collected before duplicate works were collapsed"),
    MacroCheck(macro="LitRevisionPartial", provenance="phase88_lit_audit.json",
               json_path=("per_column_verdicts", "revision_handling", "Partial"),
               tol=0.5, note="papers that touch revisions without resolving them"),
    MacroCheck(macro="LitAdmissiblePartial", provenance="phase88_lit_audit.json",
               json_path=("per_column_verdicts", "gate_admissibility_checked",
                          "Partial"), tol=0.5,
               note="papers arguing availability from a schedule without checking "
                    "it -- the category the strict rate's complement is made of"),
    MacroCheck(macro="LitPubTimePartial", provenance="phase88_lit_audit.json",
               json_path=("per_column_verdicts", "publication_time_stated",
                          "Partial"), tol=0.5,
               note="papers saying something about availability without a time"),
    MacroCheck(macro="EnrichedDays", provenance="phase54v2_current.json",
               json_path=("window", "n_test_days_pooled"), tol=0.5,
               note="enriched-baseline panel days, extended window"),
    MacroCheck(macro="EnrichedDaysKyushu",
        provenance="phase54v2_current.json",
        json_path=("window", "n_folds_per_area", "kyushu"), tol=0.5,
        note="the enriched panel is shorter in Kyushu alone; this was the "
             "last hand-typed count in the data-flow table, which is the "
             "ledger the paper points readers at to reconcile every other "
             "count"),
    MacroCheck(macro="PreGateOverEnrichedDrop", provenance="phase54v2_drop.json",
               json_path=("results", "deltas_daily_mean_with_ci95",
                          "d42_total_prev2_over_enriched", "point_daily_mean"),
               note="lineage branch V-drop: supply-mix block excluded"),
    MacroCheck(macro="PreGateOverEnrichedLagX", provenance="phase54v2_lag144.json",
               json_path=("results", "deltas_daily_mean_with_ci95",
                          "d42_total_prev2_over_enriched", "point_daily_mean"),
               note="lineage branch V-lag144: three-day supply lag"),
    MacroCheck(macro="CwPoints", provenance="phase74_cw_ladder.json",
        json_path=("n_points",), tol=0.0,
        note="prefecture points in the capacity-weighted field"),
    MacroCheck(macro="CwUplift", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "uplift_msm_d2_12z", "delta"), tol=6e-4,
        note="what better area measurement is worth on the headline tier"),
    MacroCheck(macro="CwUpliftPost", provenance="phase85_exact_ladder.json",
        json_path=("cross_measurement", "uplift_msm_d1_00z", "delta"), tol=6e-4,
        note="the same on the post-gate tier: near-identical, which is why the "
             "gaps do not move"),
    MacroCheck(macro="CwPanelDays", provenance="phase74_cw_ladder.json",
        json_path=("panel", "n_days_cw_complete"), tol=0.0,
        note="CW-complete scoring days"),
    # ---- Phase 73: cycle-by-cycle curve (Gate 41 landed 41b: retract & describe) ----
    MacroCheck(macro="CycleDays", provenance="phase73v2_cycle_curve.json",
        json_path=("panel", "n_days"), tol=0.0, note="common-window panel days"),
    MacroCheck(macro="CycleStepUp", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d2_12z_minus_msm_d3_12z", "diff"),
        tol=6e-4, note="the one significant step: D-3 12Z -> D-2 12Z"),
    MacroCheck(macro="CycleStepUpLo", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d2_12z_minus_msm_d3_12z", "ci95", 0),
        tol=6e-4, note="its CI lower"),
    MacroCheck(macro="CycleStepUpHi", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d2_12z_minus_msm_d3_12z", "ci95", 1),
        tol=6e-4, note="its CI upper"),
    MacroCheck(macro="CycleFlatMaxStep", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d2_21z_minus_msm_d2_18z", "diff"),
        tol=6e-4,
        note="largest adjacent step after D-2 12Z; index is today's argmax -- the "
             "check fails if a re-run moves it, which is the point"),
    MacroCheck(macro="CycleSlope", provenance="phase73v2_cycle_curve.json",
        json_path=("gate41_slope", "slope_jpykwh_per_gate_margin_hour"),
        tol=6e-5, note="Gate 41 linear slope; significant only via the D-3 rung"),
    MacroCheck(macro="CycleSlopeLo", provenance="phase73v2_cycle_curve.json",
        json_path=("gate41_slope", "ci95", 0), tol=6e-5, note="slope CI lower"),
    MacroCheck(macro="CycleSlopeHi", provenance="phase73v2_cycle_curve.json",
        json_path=("gate41_slope", "ci95", 1), tol=6e-5, note="slope CI upper"),
    MacroCheck(macro="CycleGateStepCommon", provenance="phase73v2_cycle_curve.json",
        json_path=("gate42_gate_step", "common_window_gate_step"), tol=6e-4,
        note="gate step on the common window; consistent with the full-day bound"),
    MacroCheck(macro="CycleGateStepCommonLo", provenance="phase73v2_cycle_curve.json",
        json_path=("gate42_gate_step", "ci95", 0), tol=6e-4, note="its CI lower"),
    MacroCheck(macro="CycleGateStepCommonHi", provenance="phase73v2_cycle_curve.json",
        json_path=("gate42_gate_step", "ci95", 1), tol=6e-4, note="its CI upper"),
    MacroCheck(macro="CycleStepsAfterFirst", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_summary", "n_steps_after_first"), tol=0.0,
        note="adjacent steps after the one significant rise"),
    MacroCheck(macro="CycleStepsEquiv", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_summary", "n_equivalent_at_one_tick"), tol=0.0,
        note="of those, how many establish one-tick equivalence -- only the gate "
             "step does, which is the one the paper is about"),
    MacroCheck(macro="CycleWidestCiTicks", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_summary", "widest_ci_ticks_after_first"), tol=0.06,
        note="widest post-first-step interval in ticks: why 'flat within a tick' "
             "was the wrong claim"),
    MacroCheck(macro="CycleGateStepTostP", provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d1_00z_minus_msm_d2_21z",
                   "tost_one_tick", "boot_p"), tol=6e-4,
        note="one-tick TOST on the gate step, bootstrap arm"),
    MacroCheck(macro="CycleGateStepTostPDm",
        provenance="phase73v2_cycle_curve.json",
        json_path=("pairwise_adjacent", "msm_d1_00z_minus_msm_d2_21z",
                   "tost_one_tick", "l3"), tol=6e-4,
        note="the HLN-DM L=3 arm of the gate-step equivalence test. The "
             "caption quoted the bootstrap arm alone and unlabelled, which "
             "is the smaller of the two -- the same defect the cover letter "
             "and the graphical abstract had for the pooled test"),
    MacroCheck(macro="FloorPriceRatio",
        provenance="phase68_settlement_pricing.json",
        json_path=("states", "S2_curtailing_and_floor",
                   "marginal_balancing_price", "mean"),
        denominator_json_path=("states", "S2_curtailing_and_floor",
                               "day_ahead_price", "mean"),
        tol=6.0,
        note="the two-prices ratio, printed instead of rounded up to 'three "
             "orders of magnitude' (459x is 2.66 orders)", paper_b=True),
    MacroCheck(macro="CyclePlateauMarginHrs", provenance="phase73v2_cycle_curve.json",
        json_path=("rungs", "msm_d2_12z", "gate_margin_hours_measured"), tol=0.06,
        note="measured dissemination margin of the first plateau cycle"),
    # UNEXEMPTED 2026-08-18. This was exempt as an institutional constant, but
    # phase53 measures it: the 21 UTC run's dissemination sits 90 minutes before
    # the gate. A constant that provenance records is a check, not an exemption.
    MacroCheck(macro="ExactPreGateMargin", provenance="phase53_rish_pregate.json",
        json_path=("issuance_accounting", "run_21z", "margin_minutes_before_gate"),
        tol=0.5,
        note="minutes between the 21 UTC run's dissemination and the 10:00 gate"),
    # ---- Phase 93: the headline Kyushu cell's mean and dispersion ----
    # These two macros were EXEMPT as "benchmark MAE reported by the cited paper
    # [Lago et al.], not computed here". They are this project's own Kyushu
    # M1/XGB+weather results -- macros.tex says so in its own comments and the
    # supplement table labels them as the headline cell -- so the exemption
    # rested on a false premise and left four printed numbers unchecked.
    # Each macro prints "mean +- sd", and _macro_floats returns both, so one
    # check per component pins both halves.
    MacroCheck(macro="SOTA", provenance="phase93_headline_cell_dispersion.json",
        json_path=("daily_rolling", "mae_mean"),
        note="daily-rolling mean MAE of the headline Kyushu cell (n = 140)"),
    MacroCheck(macro="SOTA", provenance="phase93_headline_cell_dispersion.json",
        json_path=("daily_rolling", "mae_sd"),
        note="the dispersion printed after the +-; it lived only in the per-fold "
             "parquet, so nothing could check it"),
    MacroCheck(macro="SOTAweekly",
        provenance="phase93_headline_cell_dispersion.json",
        json_path=("weekly_anchor", "mae_mean"),
        note="weekly-anchor mean MAE of the same cell (n = 60)"),
    MacroCheck(macro="SOTAweekly",
        provenance="phase93_headline_cell_dispersion.json",
        json_path=("weekly_anchor", "mae_sd"),
        note="its dispersion"),
    # ---- The D-2 18 UTC cycle: the fresher rung this archive cannot pin ----
    MacroCheck(macro="CycleEighteenZMarginHrs",
        provenance="phase73v2_cycle_curve.json",
        json_path=("rungs", "msm_d2_18z", "gate_margin_hours_measured"),
        tol=0.06, note="its measured margin before the gate"),
    MacroCheck(macro="CycleEighteenZImp",
        provenance="phase73v2_cycle_curve.json",
        json_path=("rungs", "msm_d2_18z", "improvement"),
        note="what it buys on the common daytime window, which is as far as this "
             "archive lets it be scored"),
    MacroCheck(macro="CycleEighteenZImpLo",
        provenance="phase73v2_cycle_curve.json",
        json_path=("rungs", "msm_d2_18z", "ci95", 0), note="its CI lower"),
    MacroCheck(macro="CycleEighteenZImpHi",
        provenance="phase73v2_cycle_curve.json",
        json_path=("rungs", "msm_d2_18z", "ci95", 1), note="its CI upper"),
    # ---- Phase 95: effective independence of the nine areas ----
    MacroCheck(macro="AreaUnsplitPct",
        provenance="phase95_area_independence.json",
        json_path=("institutional_coupling", "slots_all_areas_same_price_pct"),
        note="share of slots on which every area clears at one price"),
    MacroCheck(macro="AreaImpMeanCorr",
        provenance="phase95_area_independence.json",
        json_path=("statistical_coupling", "mean_pairwise_corr"),
        note="mean pairwise correlation of the per-area daily improvement series"),
    MacroCheck(macro="AreaEffN",
        provenance="phase95_area_independence.json",
        json_path=("statistical_coupling", "effective_n_areas"), tol=0.006,
        note="effective number of independent improvement series. The paper's "
             "'nine of nine' counts rest on this, not on nine"),
    MacroCheck(macro="AreaEffNPrices",
        provenance="phase95_area_independence.json",
        json_path=("statistical_coupling", "effective_n_areas_prices"),
        tol=0.006,
        note="the same measure on the price levels, far lower -- the improvements "
             "are less coupled than the prices they are measured on"),
    # ---- Phase 97: cost-loss decision value ----
    MacroCheck(macro="CostLossMaxV", provenance="phase97_cost_loss.json",
        json_path=("models", "m2", "max_value"),
        note="peak Murphy value score of the floor classifier with pre-gate "
             "weather"),
    MacroCheck(macro="CostLossNoWxCutoff", provenance="phase97_cost_loss.json",
        json_path=("models", "m2p", "positive_value_alpha_range", 1),
        note="the largest cost-loss ratio at which the WEATHERLESS classifier "
             "still beats climatology; above it only the weather-fed one does"),
    MacroCheck(macro="CostLossPeakAlpha", provenance="phase97_cost_loss.json",
        json_path=("weather_increment_in_value", "peak", "alpha"), tol=0.006,
        note="the cost-loss ratio at which pre-gate weather adds the most value"),
    MacroCheck(macro="CostLossPeakGain", provenance="phase97_cost_loss.json",
        json_path=("weather_increment_in_value", "difference", "0.64"),
        note="the largest value the pre-gate weather channels add, at C/L = 0.64"),
    MacroCheck(macro="CostLossPeakGainLo", provenance="phase97_cost_loss.json",
        json_path=("weather_increment_in_value", "ci95", "0.64", 0),
        note="its CI lower"),
    MacroCheck(macro="CostLossPeakGainHi", provenance="phase97_cost_loss.json",
        json_path=("weather_increment_in_value", "ci95", "0.64", 1),
        note="its CI upper"),
    MacroCheck(macro="CostLossBaseRate", provenance="phase97_cost_loss.json",
        json_path=("base_rate", "macro"), scale=100.0, tol=0.06,
        note="the floor base rate the value score is measured against"),
    # ---- Phase 98: the poll-grid test on the dissemination timestamps ----
    MacroCheck(macro="DissemDistinctLags",
        provenance="phase98_dissemination_detail.json",
        json_path=("poll_grid_test", "n_distinct_lag_values"), tol=0.5,
        note="distinct lag values across the capture; a poll grid would give few"),
    MacroCheck(macro="DissemDistinctFracMin",
        provenance="phase98_dissemination_detail.json",
        json_path=("poll_grid_test", "distinct_fractional_minute_values"),
        tol=0.5,
        note="distinct fractional-minute parts: a poll grid would hold these "
             "nearly constant"),
    MacroCheck(macro="DissemGapSeconds",
        provenance="phase98_dissemination_detail.json",
        json_path=("poll_grid_test",
                   "median_gap_between_distinct_lags_seconds"), tol=0.6,
        note="median spacing between distinct lags, in seconds"),
    MacroCheck(macro="DissemWithinMinutePct",
        provenance="phase98_dissemination_detail.json",
        json_path=("poll_grid_test", "share_within_one_minute_of_median_pct"),
        tol=0.006, note="share of files whose lag is within a minute of the median"),
    MacroCheck(macro="GateTimingSe", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence",
                   "margins", "tick", "dm_l3", "se"),
        note="dependence-aware SE of the gate-timing contrast; the equivalence "
             "margin is about one tick and so is the design's resolution"),
    # ---- Phase 96: LEAR on the full panel ----
    MacroCheck(macro="LearBaseMae", provenance="phase96_lear_ladder.json",
        json_path=("levels_mae", "price_only"),
        note="LEAR price-only pooled MAE: a WEAKER baseline than the ladder's "
             "gradient-boosted one (2.151), which is worth reporting"),
    MacroCheck(macro="LearPreGateMae", provenance="phase96_lear_ladder.json",
        json_path=("levels_mae", "pre_gate"), note="LEAR with the pre-gate tier"),
    MacroCheck(macro="LearImp", provenance="phase96_lear_ladder.json",
        json_path=("weather_increment", "improvement"),
        note="the headline weather increment on LEAR: about half the ladder's, "
             "with non-overlapping intervals, so the magnitude is "
             "specification-dependent even though the sign and breadth are not"),
    MacroCheck(macro="LearImpLo", provenance="phase96_lear_ladder.json",
        json_path=("weather_increment", "ci95", 0), note="its CI lower"),
    MacroCheck(macro="LearImpHi", provenance="phase96_lear_ladder.json",
        json_path=("weather_increment", "ci95", 1), note="its CI upper"),
    # ---- Phase 94: the evaluation window is a floor regime, not a spike one ----
    MacroCheck(macro="EvalPNinetyFive", provenance="phase94_evaluation_regime.json",
        json_path=("percentiles_jpykwh", "p95"),
        note="95th percentile of the evaluation window; the era has no scarcity "
             "pricing and every headline number is conditioned on that"),
    MacroCheck(macro="EvalPNinetyNine", provenance="phase94_evaluation_regime.json",
        json_path=("percentiles_jpykwh", "p99"), note="its 99th percentile"),
    MacroCheck(macro="EvalNearFloorPct",
        provenance="phase94_evaluation_regime.json",
        json_path=("near_floor", "share_pct"),
        note="share of slots at or below the near-floor threshold"),
    MacroCheck(macro="EvalScarcityPct",
        provenance="phase94_evaluation_regime.json",
        json_path=("scarcity", "share_pct"), tol=0.0006,
        note="share at or above the spike level Section 2.1 quotes"),
    MacroCheck(macro="EvalFloorScarcityRatio",
        provenance="phase94_evaluation_regime.json",
        json_path=("floor_to_scarcity_ratio",), tol=0.6,
        note="near-floor slots per scarcity slot: about 99 to 1, which is why "
             "this is a floor-regime study"),
    # ---- Phase 92: curtailment days, the figure Section 2 had from a
    # superseded draft (30/yr, an FY2018-era level) rather than from the
    # records this paper scores against (about 150/yr) ----
    MacroCheck(macro="CurtailDaysKyushuFyOne",
        provenance="phase92_curtailment_days.json",
        json_path=("per_area_days", "kyushu", "FY2024"), tol=0.5,
        note="Kyushu photovoltaic curtailment days, FY2024"),
    MacroCheck(macro="CurtailDaysKyushuFyTwo",
        provenance="phase92_curtailment_days.json",
        json_path=("per_area_days", "kyushu", "FY2025"), tol=0.5,
        note="Kyushu photovoltaic curtailment days, FY2025"),
    MacroCheck(macro="CurtailDaysKyushuRound",
        provenance="phase92_curtailment_days.json",
        json_path=("most_curtailed", "rounded_for_prose"), tol=0.5,
        note="the two-year mean rounded to the ten the prose quotes; bound to "
             "a rounding the script performs, not one a sentence performs"),
    MacroCheck(macro="CurtailDaysRunnerUp",
        provenance="phase92_curtailment_days.json",
        json_path=("runner_up", "mean_days"), tol=0.5,
        note="the next most affected area, so the comparative in Section 2 is "
             "a measured ratio rather than the phrase 'far more'"),
    MacroCheck(macro="CurtailRatioTopNext",
        provenance="phase92_curtailment_days.json",
        json_path=("ratio_top_to_runner_up",), tol=0.06,
        note="Kyushu over the next area: 1.9, which is why the sentence now "
             "says roughly twice instead of far more"),
    # ---- Phase 72: dissemination timing measured from archive timestamps ----
    MacroCheck(macro="DissemMedianLagMin", provenance="phase72_dissemination_audit.json",
        json_path=("per_run", "msm_d2_12z", "per_run_completion_lag_minutes", "median"),
        tol=0.6, note="median completion lag = 2 h 16 m, the number Figure 1 quotes"),
    MacroCheck(macro="DissemMaxLagMin", provenance="phase72_dissemination_audit.json",
        json_path=("per_run", "msm_d2_12z", "per_run_completion_lag_minutes", "max"),
        tol=0.6, note="worst single day, 12Z cycle"),
    MacroCheck(macro="DissemAuditDays", provenance="phase72_dissemination_audit.json",
        json_path=("gate_checks", "msm_d2_12z", "n_days_all_files_before_gate"),
        tol=0.0, note="all audited days complete before the gate; equals n_days"),
    MacroCheck(macro="DissemAuditFiles", provenance="phase72_dissemination_audit.json",
        json_path=("totals", "n_file_records"), tol=0.0,
        note="one HEAD per archived file; missing_or_error n = 0"),
    MacroCheck(macro="DissemWorstMarginTwentyOneZHrs",
        provenance="phase72_dissemination_audit.json",
        json_path=("gate_checks", "msm_d2_21z", "worst_margin_minutes"),
        scale=1.0 / 60.0, tol=0.06, note="worst pre-gate margin, hours, 21Z"),
    MacroCheck(macro="DissemWorstMarginTwelveZHrs",
        provenance="phase72_dissemination_audit.json",
        json_path=("gate_checks", "msm_d2_12z", "worst_margin_minutes"),
        scale=1.0 / 60.0, tol=0.06, note="worst pre-gate margin, hours, 12Z"),
    # ---- Phase 75: decision-scale equivalence (fifth review #1 landing) ----
    MacroCheck(macro="DecShareGtTickPct", provenance="phase75_decision_equivalence.json",
        json_path=("descriptives", "per_slot_abs_price_forecast_diff_jpykwh",
                   "share_gt_one_tick"),
        scale=100.0, tol=0.06,
        note="share of slots where the two runs' forecasts differ by > 1 tick", paper_b=True),
    MacroCheck(macro="DecMeanAbsDy", provenance="phase75_decision_equivalence.json",
        json_path=("descriptives", "per_slot_abs_price_forecast_diff_jpykwh", "mean"),
        tol=0.006, note="mean per-slot |yhat_pre - yhat_post|", paper_b=True),
    MacroCheck(macro="DecShareCommitChangePct",
        provenance="phase75_decision_equivalence.json",
        json_path=("descriptives", "per_slot_abs_commitment_diff_mwh", "share_nonzero"),
        scale=100.0, tol=0.06, note="share of slots whose commitment changes", paper_b=True),
    MacroCheck(macro="DecSumAbsDq", provenance="phase75_decision_equivalence.json",
        json_path=("gate37_nomination", "annualised_sum_abs_dq_mwh_per_mw_yr"),
        tol=0.06, note="annualised commitment shift; Gate 37 margin was 1 MWh", paper_b=True),
    MacroCheck(macro="DecSumAbsDqLo", provenance="phase75_decision_equivalence.json",
        json_path=("gate37_nomination", "ci95", 0), tol=0.06,
        note="commitment shift CI lower", paper_b=True),
    MacroCheck(macro="DecSumAbsDqHi", provenance="phase75_decision_equivalence.json",
        json_path=("gate37_nomination", "ci95", 1), tol=0.06,
        note="commitment shift CI upper", paper_b=True),
    MacroCheck(macro="DecProfitDiffAbsK", provenance="phase75_decision_equivalence.json",
        json_path=("gate38_profit", "annualised_mean_jpy_per_mw_yr"),
        scale=-1e-3, tol=0.06,
        note="post-gate commitments settle higher by this much (sign flipped for prose)", paper_b=True),
    MacroCheck(macro="DecProfitDiffLoAbsK",
        provenance="phase75_decision_equivalence.json",
        json_path=("gate38_profit", "ci95_annualised", 1), scale=-1e-3, tol=0.06,
        note="profit differential CI, near-zero end, flipped", paper_b=True),
    MacroCheck(macro="DecProfitDiffHiAbsK",
        provenance="phase75_decision_equivalence.json",
        json_path=("gate38_profit", "ci95_annualised", 0), scale=-1e-3, tol=0.06,
        note="profit differential CI, far end, flipped", paper_b=True),
    MacroCheck(macro="DecProfitDmPLthree",
        provenance="phase75_decision_equivalence.json",
        json_path=("gate38_profit", "dm", "l3", "p_hln"), tol=0.0006,
        note="HLN-NW DM p at L=3 on the paired daily profit differential", paper_b=True),
    MacroCheck(macro="DecProfitDmPLnine",
        provenance="phase75_decision_equivalence.json",
        json_path=("gate38_profit", "dm", "l9", "p_hln"), tol=0.0006,
        note="HLN-NW DM p at L=9", paper_b=True),
    MacroCheck(macro="CnpePostRunK",
        provenance="phase52_fip_commitment_op_msm_d1_00z.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "voi_yen_per_mw_yr"),
        scale=1e-3, tol=0.6,
        note="CNPE of the post-gate run vs price-only; -169k - (-134k) = the "
             "phase75 differential to the yen", paper_b=True),
    # ---- Phase 68: settlement prices vs the marginal balancing cost.
    MacroCheck(macro="KAboveCostShare", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S0_all', 'share_K_above_marginal_cost'),
        scale=100.0,
        tol=0.06, note="share of all slots where K exceeds the marginal balancing cost -- the refuted hypothesis", paper_b=True),
    MacroCheck(macro="DayAheadFloorCurtail", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_curtailing_and_floor', 'day_ahead_price', 'mean'),
        tol=0.0006, note="mean day-ahead price on floor-and-curtailment slots", paper_b=True),
    MacroCheck(macro="MarginalCostFloorCurtail", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_curtailing_and_floor', 'marginal_balancing_price', 'mean'),
        tol=0.006, note="mean marginal balancing price on the same slots", paper_b=True),
    MacroCheck(macro="MarginalCostFloorCurtailMed", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_curtailing_and_floor', 'marginal_balancing_price', 'median'),
        tol=0.006, note="its median, since the distribution is bimodal", paper_b=True),
    MacroCheck(macro="CostExceedsDaShare", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_curtailing_and_floor', 'share_marginal_cost_above_day_ahead'),
        scale=100.0,
        tol=0.06, note="share of those slots where the marginal cost exceeds the day-ahead price", paper_b=True),
    MacroCheck(macro="NineFloorCurtailSlots", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S3_curtailing_and_all_nine_floor', 'n_slot_areas'),
        tol=0.6, note="slot-areas with renewable curtailment AND all nine areas at the floor", paper_b=True),
    MacroCheck(macro="MarginalCostNineFloor", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S3_curtailing_and_all_nine_floor', 'marginal_balancing_price', 'mean'),
        tol=0.006, note="marginal balancing price in that tightest state", paper_b=True),
    MacroCheck(macro="MarginalCostBlockLong", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_block_long', 'marginal_balancing_price', 'mean'),
        tol=0.006, note="same, restricted to slots recorded as a block surplus", paper_b=True),
    MacroCheck(macro="MarginalCostAreaAlone", provenance="phase68_settlement_pricing.json",
        json_path=('states', 'S2_area_alone', 'marginal_balancing_price', 'mean'),
        tol=0.006, note="same, restricted to areas alone in their block (area-specific)", paper_b=True),
    MacroCheck(macro="ZeroRuleAppliedShare", provenance="phase68_settlement_pricing.json",
        json_path=('zero_price_rule', 'share_K_exactly_zero_when_block_long'),
        scale=100.0,
        tol=0.06, note="K is exactly zero this often when the block is recorded long: the rule is applied", paper_b=True),
    MacroCheck(macro="BlockLongShareFloorCurtail", provenance="phase68_settlement_pricing.json",
        json_path=('zero_price_rule', 'block_classification_on_S2', 'share_long'),
        scale=100.0,
        tol=0.06, note="... but floor-and-curtailment slots are classified a block surplus only this often", paper_b=True),
    MacroCheck(macro="BreakSpanConditioned", provenance="phase68_settlement_pricing.json",
        json_path=('changepoint_p_block_long', 'break_date_ci95_span_days'),
        tol=0.6, note="break-date CI span on the CONDITIONED series: still far too wide to assert a date", paper_b=True),
    # ---- Full-sample CNPE: the primary private figure from the fifth review on.
    MacroCheck(macro="CnpeFullSampleDays", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "n_days"),
        tol=0.5, note="scored decision test days in the full sample", paper_b=True),
    MacroCheck(macro="CnpeFullK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "voi_yen_per_mw_yr"),
        scale=1e-3, tol=0.6, note="full-sample (547-day) CNPE, thousand JPY per MW-yr", paper_b=True),
    MacroCheck(macro="CnpeFullLoK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "bootstrap_ci95_yen_per_mw_yr", 0),
        scale=1e-3, tol=0.6, note="its CI lower endpoint", paper_b=True),
    MacroCheck(macro="CnpeFullHiK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "bootstrap_ci95_yen_per_mw_yr", 1),
        scale=1e-3, tol=0.6, note="its CI upper endpoint", paper_b=True),
    MacroCheck(macro="CnpeFullAbsK", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "voi_yen_per_mw_yr"),
        scale=-1e-3, tol=0.6, note="its magnitude", paper_b=True),
    MacroCheck(macro="CnpeFullP", provenance="phase52_fip_commitment.json",
        json_path=("windows", "prev2", "regimes", "all", "voi", "op_vs_base",
                   "pooled", "dm_hln_nw_l3", "p_hln"),
        tol=6e-4, note="full-sample HLN-DM p at L=3", paper_b=True),
    # ---- Phase 69: netting against the wide-area block's own imbalance direction.
    # Gate 39 (phase77) FAILED: the direction-classified construction overstates
    # the net figure by ~27% because 23% of slots flip deviation sign between the
    # arms. Per the pre-registered branch the EXACT marginal-entrant value is
    # primary and these macros are bound to phase77.
    MacroCheck(macro="NetImprovement", provenance="phase77_netting_exact.json",
        json_path=("scopes", "all", "exact", "value"),
        tol=0.06, note="net improvement, exact marginal change in block |imbalance|", paper_b=True),
    MacroCheck(macro="NetImprovementLo", provenance="phase77_netting_exact.json",
        json_path=("scopes", "all", "exact", "ci95", 0),
        tol=0.06, note="its CI lower endpoint (Gate 25's retraction clause turns on this)", paper_b=True),
    MacroCheck(macro="NetImprovementHi", provenance="phase77_netting_exact.json",
        json_path=("scopes", "all", "exact", "ci95", 1),
        tol=0.06, note="its CI upper endpoint", paper_b=True),
    MacroCheck(macro="GrossReduction", provenance="phase69_netting.json",
        json_path=("scopes", "all", "gross_reduction_mwh_per_mw_yr", "value"),
        tol=0.06, note="the reduction before netting -- never a headline", paper_b=True),
    MacroCheck(macro="SurvivingShare", provenance="phase77_netting_exact.json",
        json_path=("scopes", "all", "surviving_share_exact"),
        scale=100.0, tol=0.6, note="net/gross as a percentage, exact", paper_b=True),
    MacroCheck(macro="SurvivingShareFloor", provenance="phase77_netting_exact.json",
        json_path=("scopes", "floor", "surviving_share_exact"),
        scale=100.0, tol=0.6,
        note="the same on floor slots, which is what the paper is about", paper_b=True),
    MacroCheck(macro="NetImprovementFloor", provenance="phase77_netting_exact.json",
        json_path=("scopes", "floor", "exact", "value"),
        tol=0.06, note="net improvement on floor slots, exact", paper_b=True),
    MacroCheck(macro="NetImprovementAlone", provenance="phase77_netting_exact.json",
        json_path=("scopes", "area_alone_in_block", "exact", "value"),
        tol=0.06,
        note="net improvement where the area is alone in its block, so the block "
             "quantity is area-specific", paper_b=True),
    MacroCheck(macro="NetImprovementDirClassified",
        provenance="phase77_netting_exact.json",
        json_path=("scopes", "all", "phase69", "value"), tol=0.06,
        note="the superseded direction-classified figure, quoted in S4's comparison", paper_b=True),
    MacroCheck(macro="NetFlipSlots", provenance="phase77_netting_exact.json",
        json_path=("gate39_first_order_exactness", "n_slots_flipped"), tol=0.0,
        note="slots where adding either entrant actually flips the block's sign", paper_b=True),
    MacroCheck(macro="NetPanelSlots", provenance="phase77_netting_exact.json",
        json_path=("panel", "n_slots"), tol=0.0, note="joined netting panel size", paper_b=True),
    MacroCheck(macro="NetExactLinearGapPct", provenance="phase77_netting_exact.json",
        json_path=("gate39_first_order_exactness", "relative_gap_exact_vs_linear"),
        scale=100.0, tol=0.006, note="exact vs first-order-in-B gap", paper_b=True),
    MacroCheck(macro="NetImprovementPvW", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "pv_capacity", "value"), tol=0.06,
        note="exact net, PV-capacity weights", paper_b=True),
    MacroCheck(macro="NetImprovementPvWLo", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "pv_capacity", "ci95", 0), tol=0.06,
        note="its CI lower", paper_b=True),
    MacroCheck(macro="NetImprovementPvWHi", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "pv_capacity", "ci95", 1), tol=0.06,
        note="its CI upper", paper_b=True),
    MacroCheck(macro="NetImprovementPeakW", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "peak_demand", "value"), tol=0.06,
        note="exact net, peak-demand weights", paper_b=True),
    MacroCheck(macro="NetImprovementPeakWLo", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "peak_demand", "ci95", 0), tol=0.06,
        note="its CI lower", paper_b=True),
    MacroCheck(macro="NetImprovementPeakWHi", provenance="phase77_netting_exact.json",
        json_path=("gate40_weighting", "peak_demand", "ci95", 1), tol=0.06,
        note="its CI upper", paper_b=True),
    MacroCheck(macro="DevSignFlipSharePct", provenance="phase77_netting_exact.json",
        json_path=("gate39_first_order_exactness", "n_slots_dev_sign_flip"),
        denominator_json_path=("gate39_first_order_exactness", "n_slots"),
        scale=100.0, tol=0.06,
        note="share of slots where the operational forecast moves the deviation "
             "across zero: the mechanism behind the phase69 overstatement", paper_b=True),
    MacroCheck(macro="CrossAreaOffsetPct", provenance="phase69_netting.json",
        json_path=("cross_area_netting", "netting_offset_share"),
        scale=100.0, tol=0.6,
        note="the complement: how much of the gross deviation DOES offset across "
             "areas. Derived in the script, not by hand", paper_b=True),
    MacroCheck(macro="CrossAreaNettingRatio", provenance="phase69_netting.json",
        json_path=("cross_area_netting", "netting_ratio_baseline"),
        tol=6e-4,
        note="share of the fleet's gross deviation surviving signed summation "
             "across the nine areas: area errors barely offset, because they are "
             "driven by the same weather", paper_b=True),
    # ---- Phase 66: floor-event probabilistic evaluation. Note the two different
    # threshold key spellings: `levels` uses "0.5" and `pairings` uses "0p5",
    # because a dot in a LaTeX-adjacent key was more trouble than the asymmetry.
    MacroCheck(macro="FloorTestDays", provenance="phase66v2_floor_regime.json",
               json_path=("coverage", "n_test_days"), tol=0.5,
               note="floor-event test days: shorter than the ladder's panel "
                    "because the expanding-window calibrator has no past on "
                    "day 1 and its burn-in days are excluded"),
    MacroCheck(macro="FloorTestSlots", provenance="phase66v2_floor_regime.json",
               json_path=("coverage", "n_slots"), tol=0.5,
               note="slots behind the floor-event metrics"),
    MacroCheck(macro="FloorPrAucBase", provenance="phase66v2_floor_regime.json",
        json_path=("levels", "allhours", "0.5", "pooled_macro_avg", "price_only",
                   "pr_auc", "value"),
        tol=6e-4, note="price-only floor PR-AUC, macro-averaged over nine areas"),
    MacroCheck(macro="FloorPrAucPreGate", provenance="phase66v2_floor_regime.json",
        json_path=("levels", "allhours", "0.5", "pooled_macro_avg", "msm_d2_12z_cw",
                   "pr_auc", "value"),
        tol=6e-4, note="pre-gate run floor PR-AUC"),
    MacroCheck(macro="FloorPrAucOracle", provenance="phase66v2_floor_regime_oracle.json",
        json_path=("levels", "allhours", "0.5", "pooled_macro_avg", "oracle_cw",
                   "pr_auc", "value"),
        tol=6e-4,
        note="realised-weather oracle floor PR-AUC -- BELOW the pre-gate run, "
             "which reproduces the ladder's 'ex-post is not a ceiling' finding "
             "in the regime domain"),
    # REBOUND 2026-08-13 to the CAPACITY-WEIGHTED pairing (phase66v3). phase66v2's
    # HEADLINE constant is the single-point tier, so P1-P7 were all single-point
    # while its levels block scored the _cw tiers -- the manuscript printed CW
    # levels (0.339 -> 0.635) beside an SP gain (0.276) whose difference did not
    # reconcile. Gate 35 promoted the CW field to primary and the promotion never
    # reached these contrasts. The CW pairing's delta IS the difference of the
    # printed levels, to six decimals.
    MacroCheck(macro="FloorPrAucDelta", provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "delta"),
        tol=6e-4, note="Gate 18: pre-gate minus price-only floor PR-AUC"),
    # REBOUND 2026-08-13 to a bootstrap OF THE MACRO STATISTIC. The floor-regime
    # phase reports these endpoints as the MEAN of the nine per-area intervals and
    # its own provenance flags that as "not a bootstrap of the macro-average" --
    # while the supplement argues, in this paper's own words, that an interval has
    # to be for the statistic being reported. phase81b resamples delivery days in
    # blocks and recomputes the macro statistic on each draw. Same quantity (the
    # point estimates agree to four decimals), narrower interval, and now
    # consistent with the methodology the paper states.
    MacroCheck(macro="FloorPrAucDeltaLo",
        provenance="phase81b_floor_inference.json",
        json_path=("paired", "m0rank_minus_baserank", "ci95", 0), tol=6e-4,
        note="its CI lower endpoint, from the macro-statistic bootstrap"),
    MacroCheck(macro="FloorPrAucDeltaHi",
        provenance="phase81b_floor_inference.json",
        json_path=("paired", "m0rank_minus_baserank", "ci95", 1), tol=6e-4,
        note="its CI upper endpoint, same"),
    MacroCheck(macro="FloorPrAucDeltaEndpointLo",
        provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "ci95", 0), tol=6e-4,
        note="the superseded endpoint-averaged lower bound, kept so the "
             "supplement can show what the conservative summary gave"),
    MacroCheck(macro="FloorPrAucDeltaEndpointHi",
        provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "ci95", 1), tol=6e-4,
        note="and its upper bound"),
    MacroCheck(macro="FloorPrAucDeltaSp", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P2_pregate_vs_price', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'delta'), tol=6e-4,
        note="the same contrast on the single-point arm: the value an earlier version printed beside capacity-weighted levels"),
    MacroCheck(macro="FloorPrAucDeltaSpLo", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P2_pregate_vs_price', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'ci95', 0), tol=6e-4,
        note="its CI lower"),
    MacroCheck(macro="FloorPrAucDeltaSpHi", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P2_pregate_vs_price', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'ci95', 1), tol=6e-4,
        note="its CI upper"),
    MacroCheck(macro="GateTimingPrAucSp", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P4_gate_timing', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'delta'), tol=6e-4,
        note="the tail gate contrast on the single-point arm"),
    MacroCheck(macro="GateTimingPrAucSpLo", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P4_gate_timing', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'ci95', 0), tol=6e-4,
        note="its CI lower"),
    MacroCheck(macro="GateTimingPrAucSpHi", provenance="phase66v2_floor_regime.json",
        json_path=('pairings', 'P4_gate_timing', 'allhours', '0p5', 'pooled_macro_avg', 'd_pr_auc', 'ci95', 1), tol=6e-4,
        note="its CI upper"),
    MacroCheck(macro="FloorBrierDelta", provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5",
                   "pooled_macro_avg", "d_brier", "delta"),
        tol=6e-5, note="Brier improvement (a proper loss, so DM-testable)"),
    MacroCheck(macro="FloorLiftTokyo", provenance="phase66v2_floor_regime.json",
        json_path=("levels", "allhours", "0.5", "tokyo", "msm_d2_12z_cw",
                   "pr_auc_lift"),
        tol=0.06,
        note="Tokyo base-rate lift: the LOWEST PR-AUC level but the HIGHEST lift, "
             "which is why the level is never reported without it"),
    MacroCheck(macro="FloorLiftKyushu", provenance="phase66v2_floor_regime.json",
        json_path=("levels", "allhours", "0.5", "kyushu", "msm_d2_12z_cw",
                   "pr_auc_lift"),
        tol=0.06, note="Kyushu base-rate lift, for contrast with Tokyo"),
    MacroCheck(macro="FloorRecallKyushuBase", provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5", "kyushu",
                   "d_recall_at_fpr", "0.01", "level_b"),
        tol=6e-4, note="Kyushu floor recall at FPR<=0.01, price-only"),
    MacroCheck(macro="FloorRecallKyushuOp", provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "allhours", "0p5", "kyushu",
                   "d_recall_at_fpr", "0.01", "level_a"),
        tol=6e-4, note="same, pre-gate run: the operationally readable number"),
    MacroCheck(macro="GateTimingPrAuc", provenance="phase66v2_floor_regime_v3_cw.json",
        json_path=("pairings", "P4cw_gate_timing", "allhours", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "delta"),
        tol=6e-4,
        note="Gate 19a: gate timing is a null in the TAIL as well as in MAE"),
    # Same rebinding, and it matters more here: the manuscript rests an
    # includes-zero reading on this interval, so it must be an interval of the
    # statistic. It still includes zero on the tighter bootstrap.
    MacroCheck(macro="GateTimingPrAucLo",
        provenance="phase81b_floor_inference.json",
        json_path=("paired", "m0rankpost_minus_m0rank", "ci95", 0), tol=6e-4,
        note="its CI lower endpoint (below zero), macro-statistic bootstrap"),
    MacroCheck(macro="GateTimingPrAucHi",
        provenance="phase81b_floor_inference.json",
        json_path=("paired", "m0rankpost_minus_m0rank", "ci95", 1), tol=6e-4,
        note="its CI upper endpoint, same"),
    # REBOUND 2026-08-14 to the capacity-weighted full grid. This was the last
    # floor number in the manuscript still taken from the single-point arm; the
    # midday scope simply had not been computed on the primary arm yet.
    MacroCheck(macro="FloorMiddayDelta",
        provenance="phase66v2_floor_regime_v3_cw_full.json",
        json_path=("pairings", "P2cw_pregate_vs_price", "midday", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "delta"),
        tol=6e-4, note="Gate 21: the same contrast on midday slots, capacity "
                       "weighted -- and the gate now evaluates on this arm"),
    MacroCheck(macro="FloorMiddayDeltaSp", provenance="phase66v2_floor_regime.json",
        json_path=("pairings", "P2_pregate_vs_price", "midday", "0p5",
                   "pooled_macro_avg", "d_pr_auc", "delta"),
        tol=6e-4, note="the single-point arm of the midday contrast, kept so the "
                       "demoted measurement stays visible"),
    # ---- Phase 67: equivalence tests. Bound to the tick-margin block of each
    # contrast's `equivalence` node, so a change of margin or of resampling scheme
    # cannot leave the printed p-values behind.
    MacroCheck(macro="GateTimingEquivMargin", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "margins", "tick",
                   "delta_margin"),
        tol=1e-9, note="equivalence margin: one JEPX price tick, JPY/kWh"),
    MacroCheck(macro="GateTimingTostP", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "margins", "tick",
                   "dm_l3", "p_tost"),
        tol=6e-4, note="gate-timing TOST p at one tick, HLN-DM L=3"),
    MacroCheck(macro="GateTimingTostPBoot", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "margins", "tick",
                   "boot", "p_tost"),
        tol=6e-4, note="same by the paired circular block bootstrap"),
    MacroCheck(macro="GateTimingCiNinetyLo", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "ci90", 0),
        tol=6e-4, note="90% (TOST) interval lower endpoint"),
    MacroCheck(macro="GateTimingCiNinetyHi", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "ci90", 1),
        tol=6e-4, note="90% (TOST) interval upper endpoint"),
    MacroCheck(macro="GateTimingMarginUtil", provenance="phase85_exact_ladder.json",
        json_path=("arm_cw_primary", "gate46_47_48_gate_timing", "equivalence", "margins", "tick",
                   "margin_utilisation"),
        scale=100.0, tol=0.6,
        note="90% bound as a percentage of one tick -- equivalence is TIGHT"),
    MacroCheck(macro="CompositeTostP", provenance="phase85_exact_ladder.json",
        json_path=("gate50_composite_vs_pinned_sp", "equivalence",
                   "margins", "tick", "dm_l3", "p_tost"),
        tol=6e-4,
        note="composite-vs-pinned is NOT equivalent: the gap exceeds one tick"),
    MacroCheck(macro="IngestionTostP", provenance="phase85_exact_ladder.json",
        json_path=("gate50_ingestion_sp", "equivalence",
                   "margins", "tick", "dm_l3", "p_tost"),
        tol=1e-6, note="the two ingestion routes are equivalent at one tick"),
    # Identity decomposition printed in supplement S2. These four replaced literals
    # that had gone stale against the run-pinned re-run (+11.6/-226.3/+34.3/+25.9
    # against +13.3/-265.5/+28.3/+19.6) in a passage that also asserted a dated
    # wedge collapse the changepoint test cannot support.
    MacroCheck(
        macro="CnpeLevelPreK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "mechanism",
            "voi_identity_decomposition",
            "op_vs_base",
            "level_component_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.06,
        note="level component, earlier window, k JPY per MW-yr",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeCovPreK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "pre_collapse",
            "mechanism",
            "voi_identity_decomposition",
            "op_vs_base",
            "timing_component_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.06,
        note="covariance component, earlier window",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeLevelPostK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "mechanism",
            "voi_identity_decomposition",
            "op_vs_base",
            "level_component_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.06,
        note="level component, later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeCovPostK",
        provenance="phase52_fip_commitment.json",
        json_path=(
            "windows",
            "prev2",
            "regimes",
            "post_collapse",
            "mechanism",
            "voi_identity_decomposition",
            "op_vs_base",
            "timing_component_yen_per_mw_yr",
        ),
        scale=1e-3,
        tol=0.06,
        note="covariance component, later window",
       paper_b=True,
    ),
    MacroCheck(
        macro="CnpeBandRthreeTwentyK",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "nomination_band",
            "families",
            "R3_tolerance_band_settlement",
            "rows",
            -1,
            "cnpe_pre",
            "annualised",
        ),
        scale=1e-3,
        tol=0.6,
        note="CNPE under tolerance-band settlement at the widest band, thousand JPY",
       paper_b=True,
    ),
    MacroCheck(
        macro="ToleranceBandReliefPct",
        provenance="phase61_system_metrics_band.json",
        json_path=(
            "nomination_band",
            "families",
            "R3_tolerance_band_settlement",
            "rows",
            -1,
            "cnpe_pre",
            "annualised",
        ),
        denominator_json_path=(
            "nomination_band",
            "families",
            "R3_tolerance_band_settlement",
            "rows",
            0,
            "cnpe_pre",
            "annualised",
        ),
        complement=True,
        scale=100.0,
        tol=0.6,
        note="private penalty removed by tolerance-band settlement at the widest band (%)",
       paper_b=True,
    ),
    MacroCheck(
        macro="ProxyCarriedShare",
        provenance="phase54v2_current.json",
        json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d32_proxies_over_enriched",
            "point_daily_mean",
        ),
        denominator_json_path=(
            "results",
            "deltas_daily_mean_with_ci95",
            "d42_total_prev2_over_enriched",
            "point_daily_mean",
        ),
        scale=100.0,
        tol=0.6,
        note="share of the total increment carried by the engineered proxies (%)",
    ),
    # ---- Phase 105: the excluded run's arrival at file resolution ----
    # The referee asked whether the tail of the arrival distribution ever clears
    # the gate. These three bind the answer: the earliest file in the archive,
    # the latest, and the SPREAD between them. The spread is the load-bearing
    # one -- 0/2,643 can be read as "not this time", under two minutes across
    # 2.5 years cannot.
    MacroCheck(
        macro="GateFastestFileMin",
        provenance="phase105_file_level_arrival.json",
        json_path=('per_run', 'msm_d1_00z') + ("earliest_min_after_gate",),
        note="earliest arrival of any D-1 00 UTC file in the archive, minutes "
             "AFTER its own delivery day's gate",
    ),
    MacroCheck(
        macro="GateSlowestFileMin",
        provenance="phase105_file_level_arrival.json",
        json_path=('per_run', 'msm_d1_00z') + ("latest_min_after_gate",),
        note="latest arrival, minutes after the gate",
    ),
    MacroCheck(
        macro="GateArrivalSpreadMin",
        provenance="phase105_file_level_arrival.json",
        json_path=('per_run', 'msm_d1_00z') + ("arrival_spread_min",),
        note="total spread between the earliest and latest of the audited "
             "files across 2.5 years; bound as its own leaf rather than "
             "subtracted in prose, because it is the figure that closes the "
             "tail objection",
    ),
    # ---- Phase 55: the cross-area gradient (an UNDETERMINED result, bound anyway) ----
    # Bound because they were literals, and because one was mislabelled: the
    # article read +0.53 as a forecast tier when it is the ex-post tier.
    MacroCheck(
        macro="CrossAreaRhoProxy",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "per_tier", "era5", "spearman_rho"),
        note="cross-area rank correlation for the stale-observation proxy",
    ),
    MacroCheck(
        macro="CrossAreaRhoForecastLo",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "per_tier", "histfc", "spearman_rho"),
        note="lowest cross-area rank correlation among the FORECAST tiers",
    ),
    MacroCheck(
        macro="CrossAreaRhoForecastHi",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "per_tier", "prev2", "spearman_rho"),
        note="highest cross-area rank correlation among the FORECAST tiers; the "
             "article previously quoted the ex-post tier's 0.53 here, which "
             "widened a null result it does not belong to",
    ),
    MacroCheck(
        macro="CrossAreaRhoExPost",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "per_tier", "oracle", "spearman_rho"),
        note="the EX-POST realised-weather tier's cross-area rank correlation, "
             "named as such so it cannot be folded back into the forecast range",
    ),
    MacroCheck(
        macro="CrossAreaMinP",
        provenance="phase55_tier_slope_tests.json",
        json_path=("windows", "full_phase51", "per_tier", "oracle",
                   "spearman_p_classical"),
        note="smallest p in the cross-area family, which is what licenses "
             "'none reaches conventional significance at nine areas'",
    ),
    # ---- Phase 101: the physical contrast, normalised ----
    MacroCheck(
        macro="PhysNmaeStepEq",
        provenance="phase101_physical_normalisation.json",
        json_path=('q1_nmae_by_installed_capacity', 'd1_fit_register_2024_09') + ("macro_mean_of_area_ratios", "nmae_delta_pp"),
        note="nMAE step in percentage points of installed capacity with areas "
             "weighted EQUALLY -- the genuinely scale-free reading",
    ),
    MacroCheck(
        macro="PhysNmaeStepEqLo",
        provenance="phase101_physical_normalisation.json",
        json_path=('q1_nmae_by_installed_capacity', 'd1_fit_register_2024_09') + ("macro_mean_of_area_ratios", "nmae_delta_ci95_pp", 0),
        note="lower bound of the area-equal nMAE step",
    ),
    MacroCheck(
        macro="PhysNmaeStepEqHi",
        provenance="phase101_physical_normalisation.json",
        json_path=('q1_nmae_by_installed_capacity', 'd1_fit_register_2024_09') + ("macro_mean_of_area_ratios", "nmae_delta_ci95_pp", 1),
        note="upper bound of the area-equal nMAE step",
    ),
    MacroCheck(
        macro="PhysNmaeStepCw",
        provenance="phase101_physical_normalisation.json",
        json_path=('q1_nmae_by_installed_capacity', 'd1_fit_register_2024_09') + ("capacity_weighted", "nmae_delta_pp"),
        note="the same step CAPACITY-weighted, which is the published MW figure "
             "divided by mean installed capacity. 17% larger than the area-equal "
             "arm because the largest fleet also carries the largest normalised "
             "step, which is why the two must be named rather than conflated",
    ),
    MacroCheck(
        macro="PhysNmaeStepOccto",
        provenance="phase101_physical_normalisation.json",
        json_path=("q1_nmae_by_installed_capacity", "d2_occto_area_capacity",
                   "macro_mean_of_area_ratios", "nmae_delta_pp"),
        note="the area-equal step on the OCCTO register: denominator robustness",
    ),
    MacroCheck(
        macro="PhysNmaeStepMaxreal",
        provenance="phase101_physical_normalisation.json",
        json_path=("q1_nmae_by_installed_capacity", "d3_max_realised_potential",
                   "macro_mean_of_area_ratios", "nmae_delta_pp"),
        note="the area-equal step on a vintage-free denominator (each area's "
             "maximum realised potential output). The STEP is stable across all "
             "three denominators; the LEVEL is not, which is why only the step "
             "is reported as a result",
    ),
    MacroCheck(
        macro="PhysNmaeLevelEq",
        provenance="phase101_physical_normalisation.json",
        json_path=('q1_nmae_by_installed_capacity', 'd1_fit_register_2024_09') + ("macro_mean_of_area_ratios", "nmae_pre_pct"),
        note="the pre-gate LEVEL as a share of installed capacity. Printed only "
             "to be honest that the instrument is coarse -- operational regional "
             "day-ahead photovoltaic forecasts run far below this -- and never as "
             "a claim of photovoltaic forecast skill",
    ),
    MacroCheck(
        macro="PhysRelEq",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "macro_mean_of_area_ratios_pct"),
        note="relative improvement with areas weighted equally (%)",
    ),
    MacroCheck(
        macro="PhysRelEqLo",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "macro_mean_of_area_ratios_ci95_pct", 0),
        note="its lower bound, which sits below five per cent: the reason the "
             "weighting has to be named rather than assumed",
    ),
    MacroCheck(
        macro="PhysRelEqHi",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "macro_mean_of_area_ratios_ci95_pct", 1),
        note="its upper bound",
    ),
    MacroCheck(
        macro="PhysRelPooled",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "pooled_ratio_of_means_pct"),
        note="relative improvement as the pooled ratio of means (%), the paper's "
             "own convention elsewhere; about one point above the area-equal arm",
    ),
    MacroCheck(
        macro="PhysRelAreaMin",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "per_area_pct"),
        aggregate="min",
        aggregate_key="hokkaido",
        note="weakest area's relative improvement. Bound as an extremum so a "
             "re-run that moves it to another area fails rather than passing",
    ),
    MacroCheck(
        macro="PhysRelAreaMax",
        provenance="phase101_physical_normalisation.json",
        json_path=("q2_relative_improvement", "per_area_pct"),
        aggregate="max",
        aggregate_key="kyushu",
        note="strongest area's relative improvement",
    ),
    # ---- Phase 102: signal vs learner, and the window mismatch it exposed ----
    MacroCheck(
        macro="InnovPriceDelta",
        provenance="phase102_signal_vs_learner.json",
        json_path=('gate_d_price', 'delta'),
        note="innovation handed directly as features: pooled 24-hour price gain",
    ),
    MacroCheck(
        macro="InnovPriceLo",
        provenance="phase102_signal_vs_learner.json",
        json_path=('gate_d_price', 'ci95', 0),
        note="its lower bound",
    ),
    MacroCheck(
        macro="InnovPriceHi",
        provenance="phase102_signal_vs_learner.json",
        json_path=('gate_d_price', 'ci95', 1),
        note="its upper bound",
    ),
    MacroCheck(
        macro="InnovPriceP",
        provenance="phase102_signal_vs_learner.json",
        json_path=('gate_d_price', 'dm_p_l3'),
        note="HLN-DM p at L=3 for the pooled innovation contrast",
    ),
    MacroCheck(
        macro="InnovNoiseCost",
        provenance="phase102_signal_vs_learner.json",
        json_path=('gate_d_price', 'noise_placebo_delta_thinned'),
        note="four matched-variance noise columns on the same cell: the dimensionality cost any null must be read against, itself distinguishable from zero",
    ),
    MacroCheck(
        macro="DaylightPriceDelta",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'hours_only_all_nine_areas', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'delta'),
        note="the same innovation contrast restricted to the window the PHYSICAL claim is measured on, where it is significant while the 24-hour pooled figure is not",
    ),
    MacroCheck(
        macro="DaylightPriceLo",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'hours_only_all_nine_areas', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'ci95', 0),
        note="its lower bound, above zero",
    ),
    MacroCheck(
        macro="DaylightPriceHi",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'hours_only_all_nine_areas', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'ci95', 1),
        note="its upper bound",
    ),
    MacroCheck(
        macro="DaylightPriceP",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'hours_only_all_nine_areas', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'dm_l3', 'p_hln'),
        note="HLN-DM p on the daylight window",
    ),
    MacroCheck(
        macro="HoursOnlyDelta",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'hours_only_all_nine_areas', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'delta'),
        note="restriction decomposition, hours only: bound separately because the prose contrasts it against the areas-only arm",
    ),
    MacroCheck(
        macro="AreasOnlyDelta",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'areas_only_all_hours', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'delta'),
        note="areas only, all hours: the restriction that does NOT carry the effect, which identifies the pooled null as a dilution over night slots",
    ),
    MacroCheck(
        macro="AreasOnlyLo",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'areas_only_all_hours', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'ci95', 0),
        tol=5e-5,
        note="its lower bound, below zero",
    ),
    MacroCheck(
        macro="AreasOnlyHi",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'decomposition_of_the_restriction', 'areas_only_all_hours', 'contrasts_vs_pre_gate', 'pre_plus_wx_innov_minus_pre_gate', 'ci95', 1),
        note="its upper bound",
    ),
    # ---- Phase 103b: the floor channel carries a third, not most ----
    MacroCheck(
        macro="HurdleFloorShare",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'share_of_published_gain', 'floor_state', 'ratio'),
        scale=100.0,
        note="floor-STATE channel as a share of the pooled gain (%). The strong reading of the third claim -- that weather value concentrates in floor transitions -- is not supported as an aggregate statement: this is about a third",
    ),
    MacroCheck(
        macro="HurdleFloorShareLo",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'share_of_published_gain', 'floor_state', 'ci95', 0),
        scale=100.0,
        note="its lower bound (%)",
    ),
    MacroCheck(
        macro="HurdleFloorShareHi",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'share_of_published_gain', 'floor_state', 'ci95', 1),
        scale=100.0,
        note="its upper bound (%)",
    ),
    MacroCheck(
        macro="HurdleLevelShare",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'share_of_published_gain', 'cond_level', 'ratio'),
        scale=100.0,
        note="conditional-LEVEL channel share (%): the majority conduit",
    ),
    MacroCheck(
        macro="HurdleFloorChannel",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'floor_state_channel_S_pi', 'mean'),
        note="the floor-state channel in JPY/kWh",
    ),
    MacroCheck(
        macro="HurdleLevelChannel",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('decomposition_shapley', 'cond_level_channel_S_mu', 'mean'),
        note="the conditional-level channel in JPY/kWh",
    ),
    MacroCheck(
        macro="HurdleBrierBase",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('component_diagnostics', 'brier_pi_base'),
        note="Brier score of the floor classifier without weather. Bound because the modest aggregate share is PREVALENCE, not a weak signal, and this is what shows it",
    ),
    MacroCheck(
        macro="HurdleBrierWx",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('component_diagnostics', 'brier_pi_head'),
        note="the same classifier with weather",
    ),
    MacroCheck(
        macro="HurdleKyushuShare",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=('per_area', 'kyushu', 'share_floor_state_of_ladder', 'ratio'),
        scale=100.0,
        note="Kyushu's floor-channel share (%): the only area of nine where the floor channel is the majority conduit",
    ),
    MacroCheck(
        macro="InnovPhysRecoveredPct",
        provenance="phase102_signal_vs_learner.json",
        json_path=("gate_b_physical_positive_control", "candidates",
                   "D_pre_plus_wx_innov", "recovered_fraction_of_phase89_gain"),
        scale=100.0,
        tol=0.05,
        note="share of the paper's published physical gain that the innovation "
             "recovers when handed to the learner directly. This is the positive "
             "control that makes the price null credible: the same object, on the "
             "same folds, recovers essentially all of the physical effect",
    ),
    MacroCheck(
        macro="HurdleResidual",
        provenance="phase103b_floor_channel_hurdle.json",
        json_path=("decomposition_shapley", "share_of_published_gain",
                   "reconstruction", "ratio"),
        scale=100.0,
        tol=0.05,
        note="the reconstruction term, third in the identity. I dropped this macro "
             "once because it was inconvenient to bind, which left the printed "
             "shares summing to 102.4 per cent beneath a sentence asserting they "
             "sum exactly. A referee found it. Bind the awkward term or drop the "
             "claim; do not drop the term and keep the claim",
    ),
    # ---- Phase 106: commensurable two-target asymmetry ----
    MacroCheck(
        macro="PriceRelReduction",
        provenance="phase106_commensurable_asymmetry.json",
        json_path=("price_leg", "relative_reduction_pct"),
        tol=0.005,
        note="the gate-timing gain as a fraction of the price arm's OWN mean "
             "absolute error. Commensurable with the physical leg by construction, "
             "unlike the variance shares the article used to lead with",
    ),
    MacroCheck(
        macro="AsymmetryRatio",
        provenance="phase106_commensurable_asymmetry.json",
        json_path=("ratio_physical_over_price",),
        tol=0.5,
        note="physical over price in commensurable units. Bound to the computed "
             "quotient rather than divided by hand, because a printed ratio whose "
             "arithmetic nobody can check is how this project produced an "
             "unreproducible 4.2 once already",
    ),
    MacroCheck(
        macro="LitReadN",
        provenance="phase107_litaudit_reslice.json",
        json_path=("n_full_text_read",),
        note="included studies whose audit trail records the body was read. "
             "Replaces the four LitFullText* macros, which sliced on publisher "
             "accessibility and inverted the robustness check",
    ),
    MacroCheck(
        macro="DissemSpecLagMin",
        provenance="phase72_dissemination_audit.json",
        json_path=("interpretation", "spec_lag_minutes"),
        note="the published dissemination schedule in minutes after initialisation. "
             "The measured median is shorter, which is why the schedule cannot be "
             "read as a lower bound on the lag",
    ),
    # ---- Phase 108: cost--loss against a usable fallback ----
    MacroCheck(
        macro="CostLossDominatedAlpha",
        provenance="phase108_costloss_dominated_benchmark.json",
        json_path=('alpha_where_without_weather_turns_negative',),
        tol=0.005,
        note="cost--loss ratio above which the no-weather forecast scores below climatology and is therefore never the fallback an actor would choose",
    ),
    MacroCheck(
        macro="CostLossPeakAlphaFixed",
        provenance="phase108_costloss_dominated_benchmark.json",
        json_path=('peak_vs_usable_fallback', 'alpha'),
        tol=0.005,
        note="where the increment peaks once the fallback is max(0, V_without) rather than a dominated forecast. The article headlined 0.64, which is where the DOMINATED comparison is most flattering",
    ),
    MacroCheck(
        macro="CostLossPeakGainFixed",
        provenance="phase108_costloss_dominated_benchmark.json",
        json_path=('peak_vs_usable_fallback', 'value'),
        tol=0.0005,
        note="the increment at that ratio",
    ),
    MacroCheck(
        macro="CostLossVLow",
        provenance="phase108_costloss_dominated_benchmark.json",
        json_path=('v_with_at_lowest_alpha', 'value'),
        tol=0.005,
        note="value score at the cheapest protection on the grid",
    ),
    MacroCheck(
        macro="CostLossVHigh",
        provenance="phase108_costloss_dominated_benchmark.json",
        json_path=('v_with_at_highest_alpha', 'value'),
        tol=0.005,
        note="and at the dearest; the fall between them is what identifies WHICH actors the vintage serves, and the article had it backwards",
    ),
    # ---- gate-timing on the daylight window ----
    MacroCheck(
        macro="GateTimingDaylight",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'contrasts_vs_pre_gate', 'msm_d1_00z_cw_minus_pre_gate', 'delta'),
        tol=0.0005,
        note="the ladder's own gate-timing contrast on the daylight window, primary capacity-weighted arm. Its point estimate exceeds one price tick",
    ),
    MacroCheck(
        macro="GateTimingDaylightLo",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'contrasts_vs_pre_gate', 'msm_d1_00z_cw_minus_pre_gate', 'ci95', 0),
        tol=0.0005,
        note="its lower bound, below zero",
    ),
    MacroCheck(
        macro="GateTimingDaylightHi",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'contrasts_vs_pre_gate', 'msm_d1_00z_cw_minus_pre_gate', 'ci95', 1),
        tol=0.0005,
        note="its upper bound",
    ),
    MacroCheck(
        macro="GateTimingDaylightP",
        provenance="phase102_signal_vs_learner.json",
        json_path=('pv_cell_daylight_sub_panel_descriptive', 'contrasts_vs_pre_gate', 'msm_d1_00z_cw_minus_pre_gate', 'dm_l3', 'p_hln'),
        tol=0.005,
        note="HLN-DM p: not detectable on a third of the slots, which is why the magnitude coinciding with a detectable product contribution is a precision difference rather than a contradiction",
    ),
]

# Numeric macros that legitimately have no provenance value. Kept deliberately
# short; anything that COULD be bound belongs in PENDING_BINDING instead.
EXEMPT: list[MacroExempt] = [
    MacroExempt(
        macro="ReplicationDoi",
        kind="protocol_constant",
        source="Zenodo deposit, minted at submission (PI-gated publication event)",
        reason="an archive identifier, not a measurement: no analysis produces "
               "it and no provenance file can hold it. Carries an explicit "
               "placeholder until the deposit exists, so a submission with an "
               "unminted DOI is visible in the rendered PDF rather than silent",
    ),
    MacroExempt(
        macro="ReplicationConceptDoi",
        kind="protocol_constant",
        source="Zenodo concept DOI (version-independent) of the same deposit",
        reason="the archive's version-independent identifier, cited beside the "
               "version DOI so a reader reaches the latest version; like it, an "
               "identifier rather than a measurement",
    ),
    MacroExempt(
        macro="ReplicationCommit",
        kind="protocol_constant",
        source="git submission tag of forecast-vintage-epf",
        reason="the commit the reported numbers were produced at; a repository "
               "identifier, not a measurement",
    ),
    MacroExempt(
        macro="Mone",
        kind="label",
        source="sections/ae_04_methods.tex",
        reason="model label M_1; the digit is a subscript, not a measurement",
    ),
    MacroExempt(
        macro="Mtwo",
        kind="label",
        source="sections/ae_04_methods.tex",
        reason="model label M_2; the digit is a subscript, not a measurement",
    ),
    MacroExempt(
        macro="PostGateArrival",
        kind="protocol_constant",
        source="phase51 dissemination accounting: D-1 00 UTC + the published "
               "JMBSC lag = 11:30 JST",
        reason="the clock time the excluded run reaches users. Section 5.3 now "
        "argues from the ordering of two events -- the auction clears at 10:00, "
        "this run arrives after -- so the time itself is load-bearing prose "
        "rather than an estimate",
    ),
    MacroExempt(
        macro="PreGateLeadLo",
        kind="protocol_constant",
        source="the run schedule: 12 UTC on D-2 is 21:00 JST, and the first "
               "delivery-day slot is 00:00 JST on D",
        reason="the headline tier's shortest lead at valid time. Deterministic "
        "from the schedule rather than measured. Defined once so the "
        "admissibility table and Section 7 cannot disagree -- Section 7 used to "
        "call the tier 'about 48 hours old' while the table said 27--50",
    ),
    MacroExempt(
        macro="PreGateLeadHi",
        kind="protocol_constant",
        source="the run schedule: the last delivery-day slot is 23:30 JST on D",
        reason="the headline tier's longest lead at valid time; see "
        "PreGateLeadLo",
    ),
    MacroExempt(
        macro="thrFloorDefault",
        kind="protocol_constant",
        source="FLOOR = 0.5 in the commitment decision script "
               "(scripts/20_commitment_decision.py in the replication package)",
        reason="the near-floor threshold FLOOR = 0.5 JPY/kWh; a chosen protocol "
        "constant, bound to its module definition by ConstantCheck in "
        "verify_pdf_numbers.py",
    ),
    MacroExempt(
        macro="thrSpikeDefault",
        kind="protocol_constant",
        source="SPIKE = 50 in the regime-metrics module of the research "
               "monorepo (not part of the replication package's import closure; "
               "the constant is quoted here so the justification is reviewable "
               "in both repositories)",
        reason="the spike threshold 50 JPY/kWh; a chosen protocol constant, bound "
        "to its module definition by ConstantCheck in verify_pdf_numbers.py",
    ),
]

# Numeric macros that a provenance JSON does hold but nothing has wired up yet.
# --strict fails while this list is non-empty. All of the entries below print
# cross-area heterogeneity results that Section 5 reports as UNDETERMINED and
# that the current revision moves to the supplement; binding a number that is
# about to be deleted is waste, so they wait for that edit rather than for
# someone to remember them.
PENDING_BINDING: list[MacroPending] = [
    MacroPending(macro=m, owner_phase=p, reason=r)
    for m, p, r in (
    )
]

# Word-valued macros are verified by their spelled-out form against an integer in
# the provenance JSON, because the manuscript reads better with words than digits.
# Macros whose body is a STRING that provenance nonetheless fixes: a register
# vintage, a scoring window. Exempting them would be a small lie -- the analysis
# does record these, they simply are not floats. Two modes:
#   "substring"  the macro body must appear verbatim inside the provenance string
#   "hourspan"   provenance is [start_hour, end_hour]; the body must be the
#                "HH:00--HH:00" rendering of it
#   "hhmm"       provenance is a count of MINUTES; the body must be the h/min
#                rendering of exactly that count. The abstract quoted "2 h 16
#                min" next to a bound macro reading 136 minutes: two renderings
#                of one measurement, only one of them checked.
STRING_CHECKS: list[tuple[str, str, tuple[str, ...], str, str]] = [
    (
        "CapVintage",
        "phase89_physical_target.json",
        ("conventions", "weighting"),
        "substring",
        "the operating-capacity register vintage the point weights come from; "
        "the physical target and the price models share it",
    ),
    (
        "WeightLaterVintage",
        "phase91_weight_sensitivity.json",
        ("gates", "gate59_weight_sensitivity", "later_register_arm"),
        "substring",
        "the later capacity-register snapshot the sensitivity arm uses. Bound to "
        "the single arm label, not the arm LIST: against the list, every member "
        "satisfied the check, so setting this to the paper's own 2024-09 vintage "
        "passed and read as a real sensitivity result",
    ),
    (
        "DissemMedianLagHM",
        "phase72_dissemination_audit.json",
        ("per_run", "msm_d2_12z", "per_run_completion_lag_minutes", "median"),
        "hhmm",
        "the h/min rendering of the median dissemination lag. The abstract's "
        "most-read number was hand-typed beside the bound minute figure",
    ),
    (
        "DissemMedianLagHMJa",
        "phase72_dissemination_audit.json",
        ("per_run", "msm_d2_12z", "per_run_completion_lag_minutes", "median"),
        "hhmm",
        "the same rendering in Japanese; the JA abstract hand-typed it too",
    ),
    (
        "HeadlineTierName",
        "phase59_exact_ladder.json",
        ("gate6_pregate_value", "headline_tier"),
        "tierslug",
        "the prose name of the headline tier, checked against the slug the gate "
        "actually selected. It was exempt as 'a label', but if the headline tier "
        "were re-selected the prose would silently keep naming the old run",
    ),
    (
        "PhysWindow",
        "phase89_physical_target.json",
        ("conventions", "daylight_hours_jst"),
        "hourspan",
        "the daylight window the physical target is scored over, JST",
    ),
    # ---- Phase 105 ----
    (
        "GateFastestArrivalJst",
        "phase105_file_level_arrival.json",
        ('per_run', 'msm_d1_00z') + ("earliest_arrival_jst",),
        "substring",
        "clock time of the earliest-arriving file in the whole archive; the "
        "prose quotes it beside the minute figure, so both must come from the "
        "same measurement",
    ),
    (
        "PanelStart",
        "phase85_exact_ladder.json",
        ("metadata", "protocol", "panel_start"),
        "substring",
        "first date in the modelling panel. Bound because the body asserted the "
        "FY2022 spike window sits in training data while the panel begins after it",
    ),
]


WORD_CHECKS: list[tuple[str, str, tuple[str, ...], str]] = [
    (
        "PhysAreas",
        "phase89_physical_target.json",
        ("targets", "potential", "tiers", "msm_d2_12z_cw", "per_area"),
        "areas the physical target covers: the operators that publish "
        "half-hourly photovoltaic output, Chubu excluded because its feed "
        "keeps only the current month",
    ),
    (
        "PhysWindowSlots",
        "phase89_physical_target.json",
        ("conventions", "daylight_slots_per_area_day"),
        "half-hourly slots the daylight filter keeps per area-day; the provenance "
        "records the count so the window's prose cannot understate it again",
    ),
    (
        "PhysBootBlock",
        "phase89_physical_target.json",
        ("metadata", "boot_block_days"),
        "block length in days for the physical-target bootstrap",
    ),
    (
        "ProxyWindWeakAreas",
        "phase62_proxy_validation.json",
        ("proxy_skill", "summary", "wind", "areas_below_0p3"),
        "areas where the wind proxy correlation is below 0.3",
    ),
    (
        "CapacityTableImpossible",
        "phase62_proxy_validation.json",
        ("capacity_integrity", "n_impossible"),
        "area-channels where realised output exceeds nameplate",
    ),
    # Spelled out because it is zero, and a zero that matters should read as a
    # sentence rather than a digit: no single area's data can establish gate-timing
    # equivalence at one tick. The pooled equivalence is a POOLED statement, and
    # this macro is how the paper says so.
    (
        "FloorPrAucAreasPos",
        "phase66v2_floor_regime_v3_cw.json",
        ("pairings", "P2cw_pregate_vs_price", "allhours", "0p5",
         "pooled_macro_avg", "d_pr_auc", "areas_positive"),
        "areas where the floor PR-AUC contrast is positive. Bound to the "
        "CAPACITY-WEIGHTED arm because the abstract puts this count in one "
        "sentence with the headline gain, which is capacity-weighted: the two "
        "halves of that sentence were measured on different arms. Both give "
        "nine, so the sentence was true and its provenance was not",
    ),
    (
        "LearAreasImproved",
        "phase96_lear_ladder.json",
        ("weather_increment", "areas_improved"),
        "areas where LEAR also improves: the breadth of the result does not "
        "depend on the learner even though its size does",
    ),
    (
        "StaleObsSigAreasHolm",
        "phase85_exact_ladder.json",
        ("arm_cw_primary", "tiers", "era5_cw", "areas_sig_l3_holm"),
        "areas where the stale proxy is significant after Holm adjustment. The "
        "text quoted the RAW count while Methods promises the adjusted one",
    ),
    (
        "StaleObsSigAreasBh",
        "phase85_exact_ladder.json",
        ("arm_cw_primary", "tiers", "era5_cw", "areas_sig_l3_bh"),
        "the same family under Benjamini-Hochberg, which Methods also promises "
        "and which keeps the count at eight",
    ),
    (
        "GateTimingAreasEquiv",
        "phase59_exact_ladder.json",
        ("per_area_gate_timing", "n_areas_equivalent_at_tick"),
        "areas whose own data establish gate-timing equivalence at one tick",
    ),
    # ---- Phase 105 / Phase 100: counts behind the admissibility claim ----
    (
        "GateAuditFiles",
        "phase105_file_level_arrival.json",
        ('per_run', 'msm_d1_00z') + ("n_files",),
        "D-1 00 UTC files audited at file resolution: 881 delivery days times "
        "three horizon files",
    ),
    (
        "GateFilesPreGate",
        "phase105_file_level_arrival.json",
        ('per_run', 'msm_d1_00z') + ("n_files_before_gate",),
        "audited files that arrived before their own gate. Zero, and bound as a "
        "count so the claim cannot be softened in prose without failing here",
    ),
    (
        "GateDaysAnyPreGate",
        "phase105_file_level_arrival.json",
        ('per_run', 'msm_d1_00z') + ("n_days_with_any_pre_gate_file",),
        "days on which ANY file of the excluded run beat the gate, out of 881. "
        "This is the referee's actual question and the answer is zero",
    ),
    (
        "GateFhSpreadSec",
        "phase105_file_level_arrival.json",
        ('per_run', 'msm_d1_00z') + ("fh_median_spread_seconds",),
        "spread in seconds between the per-horizon-file median arrivals. It "
        "refutes the proposed early-arriving subset: the three files land "
        "within a quarter minute of each other",
    ),
    (
        "ProbeRequests",
        "phase100_rish_cycle_horizon_probe.json",
        ("n_requests",),
        "HEAD requests issued to the archive to settle whether any off-synoptic "
        "cycle carries horizons beyond FH39",
    ),
    (
        "ProbeCycles",
        "phase100_rish_cycle_horizon_probe.json",
        ("n_cycles_probed",),
        "initialisation cycles probed",
    ),
    (
        "ProbeDates",
        "phase100_rish_cycle_horizon_probe.json",
        ("n_dates",),
        "delivery dates probed, spread across seasons and across the "
        "2025-05-20 model upgrade so a transient outage cannot read as a "
        "structural gap",
    ),
    (
        "ProbeLongHorizonCycles",
        "phase100_rish_cycle_horizon_probe.json",
        ("n_cycles_with_long_horizon",),
        "cycles of the eight that carry any horizon beyond FH39: the 00 and 12 "
        "UTC synoptic runs only",
    ),
]
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    # Extended past ten on 2026-08-13. The table stopped at ten, which had two
    # consequences the adversarial audit named: a macro spelled "twelve" or
    # "fifteen" was neither checked nor exempt nor pending -- the ratchet's
    # word test is exact-token membership in this table, so it silently passed --
    # and a WORD_CHECK whose provenance count grew past ten started demanding
    # the digit string, failing a correctly spelled-out macro.
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    24: "twenty-four",
    28: "twenty-eight",
    30: "thirty",
    46: "forty-six",
}


def _dig(obj: Any, path: tuple[str | int, ...]) -> Any:  # noqa: ANN401 — JSON traversal is dynamically typed
    for key in path:
        obj = obj[key]
    return obj


def _macro_floats(macros_tex: str, macro: str) -> list[float]:
    # one level of nesting must be allowed, so that a macro carrying a LaTeX
    # thousands separator -- \newcommand{\X}{1{,}289} -- is captured whole rather
    # than truncated at the inner brace and silently read as 1.
    m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}", macros_tex)
    if m is None:
        msg = f"macro \\{macro} not found in macros.tex"
        raise KeyError(msg)
    body = m.group(1).replace("{,}", "")
    return [float(x) for x in _SIGNED_FLOAT.findall(body)]


def _paper_b_block(macros_tex: str) -> list[str]:
    """The paper-B block in macros.tex must equal the paper_b=True check set.

    Two failure modes this catches, both of which have a history in this
    project: a macro moved into the block without its check being retagged (so
    the forecast-vintage package would ship a number nothing verifies), and a
    check tagged paper_b without its definition moving (so extracting the
    settlement paper would leave the macro behind and break its build). The
    assertion runs in BOTH directions for that reason.
    """
    begin = "% ===== BEGIN paper-B-only macros"
    end = "% ===== END paper-B-only"
    if begin not in macros_tex or end not in macros_tex:
        return ["macros.tex: the paper-B block markers are missing; the "
                "settlement macros are no longer isolated"]
    block = macros_tex.split(begin, 1)[1].split(end, 1)[0]
    in_block = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", block))
    tagged = {c.macro for c in CHECKS if c.paper_b}
    out: list[str] = []
    for m in sorted(tagged - in_block):
        out.append(f"\\{m}: check is tagged paper_b but the definition is not in "
                   "the paper-B block of macros.tex")
    for m in sorted(in_block - tagged):
        out.append(f"\\{m}: defined inside the paper-B block but no check tags it "
                   "paper_b; either retag the check or move the macro out")
    return out


def _macro_hygiene(macros_tex: str) -> list[str]:
    """Catch macros that are text-legal but LaTeX-illegal.

    This guard exists because a value-only check certified two unbuildable macros
    that shipped garbage above the title of both PDFs (2026-07-29):

      * ``\\WedgeP90Post`` -- a TeX control word may contain LETTERS ONLY, so a
        digit splits the name and the definition silently fails.
      * ``ewcommand{...}`` -- a lost leading backslash, produced by zsh's ``echo``
        interpreting the ``\\n`` of ``\\newcommand`` as a newline.

    Both matched the value regex perfectly, so the drift guard passed while the
    document was broken. Comparing values is not enough; the names have to be
    legal too.
    """
    problems: list[str] = []
    for i, line in enumerate(macros_tex.splitlines(), start=1):
        stripped = line.lstrip()
        # A lost leading backslash leaves 'ewcommand' rather than 'newcommand',
        # because it is the "\n" of "\newcommand" that gets eaten (shell echo,
        # a non-raw Python string). Match every realistic mangling, not just the
        # one that keeps the word intact.
        if re.match(r"^(?:re)?n?ewcommand\s*\{", stripped):
            problems.append(
                f"macros.tex:{i}: '{stripped[:48]}' is a command definition with "
                "no leading backslash, so the macro is never defined and the line "
                "is typeset verbatim"
            )
    for name in re.findall(r"\\(?:re)?newcommand\s*\{\\([^}]*)\}", macros_tex):
        if not name.isalpha():
            problems.append(
                f"macros.tex: \\{name} is not a legal control word (letters only); "
                "TeX splits the name at the first non-letter and the definition "
                "fails silently"
            )
    return problems


_MACRO_DEF = re.compile(r"\\(?:re)?newcommand\s*\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
# Every TeX control word in a body of text. Used to ask which macros a document
# actually prints, which is a different question from which ones exist.
_CONTROL_WORD = re.compile(r"\\([A-Za-z]+)")


def _prints_a_number(body: str) -> bool:
    r"""Does this macro body put a QUANTITY on the page, digits or not?

    A digits-only test has a hole: \GateTimingAreasEquiv expands to "zero", which
    is a measured count with no digit in it, so it slipped past the ratchet on the
    very commit that introduced it. Spelled-out numbers are used deliberately in
    this manuscript (a zero that matters reads better as a word), so the ratchet
    has to know about them or it exempts exactly the numbers chosen for emphasis.
    """
    stripped = body.replace("{,}", "")
    if re.search(r"\d", stripped):
        return True
    words = set(NUMBER_WORDS.values())
    return any(w in re.findall(r"[a-z]+", stripped.lower()) for w in words)


def _unchecked_numeric_macros(macros_tex: str) -> tuple[list[str], list[str]]:
    r"""Fail on any numeric macro that no check, exemption or TODO accounts for.

    This is the ratchet. The value checks below can only protect macros somebody
    remembered to list, and on 2026-07-29 fifty numeric macros -- including
    \CnpePreAbsK, the headline private figure -- were listed nowhere, while the
    guard reported "all 261 headline macros match their provenance". One of the
    fifty, \CnpeBalCostPreM, had already drifted to 1.1 against a provenance
    1.0597, which made the printed ratio 4.2 unreproducible from the printed
    numbers and became a reviewer's lead objection.

    Partitioning every numeric macro into checked / exempt / pending turns
    "did we remember?" into a mechanical residual, so the unguarded set can only
    shrink. Returns ``(failures, pending)``.
    """
    accounted = (
        {c.macro for c in CHECKS}
        | {w[0] for w in WORD_CHECKS}
        | {t[0] for t in STRING_CHECKS}
        | {e.macro for e in EXEMPT}
        | {p.macro for p in PENDING_BINDING}
    )
    failures: list[str] = []
    for name, body in _MACRO_DEF.findall(macros_tex):
        if name in accounted or not _prints_a_number(body):
            continue
        failures.append(
            f"\\{name} = '{body[:40]}' prints a number that no MacroCheck, "
            "WORD_CHECKS entry, EXEMPT justification or PENDING_BINDING TODO "
            "accounts for. Bind it to provenance, or justify it in EXEMPT with "
            "a source. Run --emit-unchecked for a stub."
        )
    pending = [
        f"\\{p.macro} (owner {p.owner_phase}): {p.reason}"
        for p in PENDING_BINDING
        if any(n == p.macro for n, _ in _MACRO_DEF.findall(macros_tex))
    ]
    return failures, pending


# Macros that are defined and value-checked but reach no page today. Frozen as
# a baseline so the set can only SHRINK: see _macros_printed for why.
UNPRINTED_BASELINE = frozenset({
    'CommonPriceImpStale', 'CompositeTostP', 'CycleFlatMaxStep',
    'DEConcP', 'DEConcRho', 'DENdays', 'DENegGradRho', 'DEWeatherValue',
    'DEWindGain', 'DoseSlopePerLeadHour', 'ExactVsPostGateTempRmse',
    'ExactVsPreGateTempRmse', 'FloorBaseRate', 'FloorLogDirect',
    'FloorLogIsotonic', 'FloorMiddayDeltaSp', 'FloorWedgeKyushu',
    'FloorWedgeShikoku', 'GpvAllHoursConc', 'GpvAllHoursConcNine',
    'GpvAllHoursConcNineP', 'GpvAllHoursConcP', 'GpvAllHoursImp',
    'HistfcAllHoursConc', 'HistfcAllHoursConcNine',
    'HistfcAllHoursConcNineP', 'HistfcAllHoursConcP', 'HistfcAllHoursImp',
    'IngestionTostP', 'MechImpliedTicks', 'NORMAL', 'OpsCrossAreaP',
    'OpsCurtailCoincidence', 'OpsFlagCount', 'OpsFloorAUC',
    'OpsFloorBrier', 'OpsFloorFoneCF', 'OpsOverEnriched',
    'OpsOverEnrichedAdm', 'OpsOverEnrichedAdmConc',
    'OpsOverEnrichedAdmNsig', 'OpsOverEnrichedConc',
    'OpsOverEnrichedNsig', 'OpsRealValueK', 'OpsValueRuleMaxK',
    'OpsValueRuleMinK', 'OracleFloorBrier', 'PhysDeltaWide',
    'PrevRunDecisionExtK', 'PriceOnlyCurtailCoincidence',
    'PriceOnlyFlagCount', 'PriceOnlyFloorBrier', 'PriceOnlyFloorFone',
    'PriceOnlyFloorFoneCF', 'PriceOnlyFloorPRAUC', 'ProxyAllHoursConc',
    'ProxyAllHoursConcNine', 'ProxyAllHoursConcNineP',
    'ProxyAllHoursConcP', 'ProxyAllHoursImp', 'ProxyDemandRHi',
    'ProxyDemandRLo', 'ProxyFloorBrier', 'ProxyFloorFoneCF',
    'ProxyFloorPRAUC', 'ProxyOverEnriched', 'ProxyOverEnrichedAdm',
    'RealDecisionCostOps', 'RealDecisionCostPriceOnly', 'RealFloorWedge',
    'RmaeOps', 'RmaePriceOnly', 'RmaeProxy', 'SPIKE', 'SkillCurveHiSkill',
    'SkillCurveHiVal', 'SkillCurveLoSkill', 'SkillCurveLoVal',
})

# Trees whose .tex actually gets compiled into a shipped PDF. Blind copies are
# byte-identical to their sources (_blind_trees_current enforces that), so
# scanning them would double the work and change no answer.
_PRINT_SCAN_GLOBS = (
    "paper_a_main/main.tex", "paper_a_main/supplementary.tex",
    "paper_a_main/sections/*.tex", "paper_a_main/tables/*.tex",
    "paper_a_main_ja/main.tex", "paper_a_main_ja/sections/*.tex",
    "paper_a_main_ja/tables/*.tex",
    "paper_b_methods/main.tex", "paper_b_methods/sections/*.tex",
    "paper_b_methods/tables/*.tex",
)


def _macros_printed(cfg: EpfConfig, macros_tex: str) -> list[str]:
    r"""Fail when a checked macro stops appearing in any document.

    Every other check in this file asks "is this number right?". None of them
    asks "is this number still ON THE PAGE?". The ratchet partitions macros
    DEFINED in macros.tex; it never reads the manuscript. So deleting a
    subsection deletes its verified numbers from the paper while the guard
    keeps reporting that all of them match their provenance.

    That is not hypothetical. Measured against the two subsections slated to
    move in the current restructure, 17 macros -- \VersionInteraction{,Lo,Hi},
    \HeadlineDemandW, \HeadlinePvW, \CompositeWorstMarginMin and 11 \Gfs* --
    appear ONLY there and in the Japanese mirror of the same subsections.
    Delete both without re-homing the prose and 17 provenance-bound results
    leave the manuscript with every check still green.

    It is the same family as the tolerance lesson of 2026-08-18: green means
    "nothing I wrote a check for fired", and says nothing about the dimensions
    nobody checked. Deletions are exactly the direction no check ran in.

    A macro used inside another macro's body is printed whenever the outer one
    is, so uses are closed transitively -- otherwise every composed macro would
    read as dark.

    Baseline rather than clean sweep: 79 macros are already unprinted, several
    deliberately (paper-B staging, alternative arms kept for a planned
    revision). Freezing them makes the set monotone -- a new dark macro fails,
    and a baseline entry that becomes printed or disappears must be pruned, so
    the list cannot quietly become a dumping ground.
    """
    paper_b_only: set[str] = set()
    try:
        b0 = macros_tex.index("BEGIN paper-B-only")
        b1 = macros_tex.index("END paper-B-only")
        paper_b_only = {n for n, _ in _MACRO_DEF.findall(macros_tex[b0:b1])}
    except ValueError:
        return ["macros.tex: the paper-B block markers are missing, so the "
                "printed-macro check cannot scope itself"]

    defined = {n for n, _ in _MACRO_DEF.findall(macros_tex)}
    in_scope = defined - paper_b_only

    body_uses = {n: set(_CONTROL_WORD.findall(b))
                 for n, b in _MACRO_DEF.findall(macros_tex)}

    root = cfg.project_root / "docs" / "paper"
    seen: set[str] = set()
    scanned = 0
    for pattern in _PRINT_SCAN_GLOBS:
        for path in sorted(root.glob(pattern)):
            scanned += 1
            seen |= set(_CONTROL_WORD.findall(
                path.read_text(encoding="utf-8", errors="replace")))
    # The replication package ships this guard verbatim and ships docs/macros.tex,
    # but it does not ship the manuscript -- there is nothing there to print a
    # macro INTO. Scanning zero files would make every macro read as dark and turn
    # this check into a wall of false failures, which is what it did when it was
    # first written. Absence of the manuscript is the companion's normal state, so
    # the check declines to run rather than inventing findings.
    if scanned == 0:
        return []

    printed, stack = set(seen), list(seen)
    while stack:
        for used in body_uses.get(stack.pop(), ()):
            if used not in printed:
                printed.add(used)
                stack.append(used)

    dark = in_scope - printed
    problems = [
        f"\\{name} is defined and checked but appears in no document. It was "
        "printed when the baseline was frozen, so an edit removed it from the "
        "manuscript while its check kept passing. Re-home the prose that "
        "carried it, or delete the macro and its check."
        for name in sorted(dark - UNPRINTED_BASELINE)
    ]
    problems += [
        f"\\{name} is in UNPRINTED_BASELINE but is now printed. Remove it from "
        "the baseline so the unprinted set stays monotone."
        for name in sorted((UNPRINTED_BASELINE & in_scope) - dark)
    ]
    problems += [
        f"\\{name} is in UNPRINTED_BASELINE but is no longer defined in "
        "macros.tex. Remove the stale baseline entry."
        for name in sorted(UNPRINTED_BASELINE - defined)
    ]
    return problems


def _double_period_clean(cfg: EpfConfig) -> list[str]:
    r"""Fail on ``\paragraph`` arguments that already end in a period.

    ``elsarticle.cls`` appends its own period to every run-in heading::

        \def\els@aparagraph[#1]#2{\elsparagraph[#1]{#2.}}

    so ``\paragraph{The physical outcome is unambiguous.}`` renders as
    "unambiguous.." in main.pdf. The source contains no ``..`` at all, which is
    why grepping the .tex for months found nothing while eight of them shipped.

    Scoped by DOCUMENT CLASS, not by directory. Only ``elsarticle`` doubles the
    period: this repo's supplement and its other manuscripts use ``article``,
    where the same construct renders correctly. Flagging by directory would have
    demanded 171 edits that fix nothing -- verified against the compiled PDFs,
    which carry 7 of these in main.pdf and zero in supplementary.pdf.
    """
    pat = re.compile(r"\\paragraph\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
    problems: list[str] = []
    for root in sorted((cfg.project_root / "docs/paper").glob("*/*.tex")):
        head = root.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\\documentclass[^\n]*\{elsarticle\}", head):
            continue
        included = [root]
        for rel_in in re.findall(r"\\input\{([^}]*)\}", head):
            cand = (root.parent / rel_in).with_suffix(".tex")
            if cand.exists():
                included.append(cand)
        for tex in included:
            for i, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
                for arg in pat.findall(line):
                    if arg.rstrip().endswith("."):
                        rel = tex.relative_to(cfg.project_root)
                        problems.append(
                            f"{rel}:{i}: \\paragraph{{{arg[:44]}}} ends in a period; "
                            f"{root.name} is elsarticle, which appends its own, so "
                            "this renders as '..'"
                        )
    return problems


class RetiredTerm(BaseModel):
    """A phrase or number the manuscript has retired, with why and where.

    ``allowed_in`` names files where the string may legitimately survive -- a
    sentence that explicitly disclaims the retired reading has to be able to
    quote it.
    """

    model_config = ConfigDict(frozen=True)

    pattern: str  # case-insensitive regex
    retired_because: str
    allowed_in: tuple[str, ...] = ()


RETIRED_TERMS: list[RetiredTerm] = [
    # ---- retired by the seventh review's revision (2026-08-13) ------------
    RetiredTerm(
        pattern=r"expectations channel",
        retired_because="the mechanism test (phase90) landed on the retraction "
        "branch: the innovation explains the PRICE residual too, so the paper "
        "reports a magnitude asymmetry and not a price-formation channel",
    ),
    RetiredTerm(
        pattern=r"prices? (?:are |is )?form(?:ed|s) on pre-gate expectations",
        retired_because="same retraction: the design does not identify how the "
        "price is formed, and the innovation reaches it",
    ),
    RetiredTerm(
        pattern=r"the price never saw it",
        retired_because="refuted by the mechanism test at p < 1e-4",
    ),
    RetiredTerm(
        pattern=r"recorded photovoltaic output",
        retired_because="the physical target is recorded generation PLUS "
        "curtailed energy; 'recorded output' names a dispatch outcome and "
        "misdescribes the computation",
    ),
    RetiredTerm(
        pattern=r"strongest baseline (?:a bidder|anyone) could",
        retired_because="the audit excludes commercial products, so this is the "
        "richest baseline WE could assemble from public data",
    ),
    RetiredTerm(
        pattern=r"second (?:forecast )?provider reproduces",
        retired_because="two cells on the single-point arm corroborate the "
        "ordering; they do not reproduce the capacity-weighted result",
    ),
    RetiredTerm(
        pattern=r"should buy the vintage",
        retired_because="commercial advice the evidence does not support; "
        "replaced by what the engineered proxies capture",
    ),
    RetiredTerm(
        pattern=r"pre-?registrat|pre-?registered",
        retired_because="nothing was deposited in a public registry; the "
        "compliant wording is 'recorded before estimation'",
        # NARROWED 2026-08-13. This used to allow main.pdf and supplementary.pdf
        # -- i.e. the entire article and the entire supplement, the only two
        # documents the body renders into -- so the term was unenforceable
        # everywhere it is written. The two sentences that must quote it to
        # disclaim it are in these two source files and nowhere else; the term is
        # now forbidden in every other .tex and in both PDFs.
        # The two sentences that must quote the term to disclaim it, and the
        # two PDFs they render into. Everywhere else -- every other .tex, every
        # figure, every submission markdown -- it is forbidden.
        allowed_in=("ae_04_methods.tex", "supp_s1_data.tex",
                    "main.pdf", "supplementary.pdf"),
    ),
    RetiredTerm(
        pattern=r"(?:roughly|about|approximately)\s+30\s+(?:curtailment days|\u306e\u51fa\u529b\u5236\u5fa1\u65e5)",
        retired_because="an FY2018-era OCCTO level, inherited verbatim from a "
        "superseded draft. On the FY2024-FY2026 records this paper scores "
        "against, Kyushu curtails on about 150 days a year -- five times the "
        "figure the manuscript printed, in the paragraph that motivates using "
        "curtailment as a surplus marker",
    ),
    RetiredTerm(
        pattern=r"far more than any other area|\u4ed6\u306e\u3069\u306e\u30a8\u30ea\u30a2\u3088\u308a\u306f\u308b\u304b\u306b",
        retired_because="the measured ratio to the next most affected area is "
        "1.9, so the comparative was an overstatement even after the count was "
        "corrected",
    ),
    RetiredTerm(
        pattern=r"system saving",
        retired_because="valued one party's gross deviation at settlement prices "
        "and called it an avoided system cost",
        allowed_in=("main.pdf", "supplementary.pdf"),
    ),  # the disclaimer quotes it
    RetiredTerm(
        pattern=r"(?:volume|imbalance) the system must (?:cover|procure)",
        retired_because="a single portfolio's gross deviation is not system imbalance",
    ),
    RetiredTerm(
        pattern=r"commitment VoI|value of information",
        retired_because="renamed to the compliant-nomination profit effect; a value "
        "of information cannot be negative",
        # The theory supplement still discusses the (non-negative) VoI, and the
        # article and cover letter each explain WHY the empirical quantity is not
        # one -- a disclaimer has to be able to name the thing it disclaims.
        allowed_in=("main.pdf", "supplementary.pdf", "cover_letter.pdf",
                    "cover_letter.md",
                    # the sources those PDFs render, now that .tex is scanned
                    "ae_01_introduction.tex", "supp_s2_theory.tex",
                    "supp_s5_architectures.tex"),
    ),
    RetiredTerm(
        pattern=r"before\s*(?:->|→)\s*after",
        retired_because="asserts a dated settlement-regime break that the "
        "changepoint test cannot identify",
    ),
    RetiredTerm(
        pattern=r"wedge collapse",
        retired_because="the wedge declines continuously; no break date is asserted",
    ),
    RetiredTerm(
        pattern=r"fullest admissible",
        retired_because="replaced by 'enriched public-data, gate-available baseline'",
    ),
    RetiredTerm(
        pattern=r"previous-day-[12] run",
        retired_because="the previous-day tiers are fixed-lead composites, not runs",
    ),
    RetiredTerm(
        pattern=r"1\.1 million|4\.2\s*(?:times|×|x)\b",
        retired_because="the retracted system-saving-over-private-loss ratio",
    ),
    # NOT 0.216: that is the lead-indexed composite's live improvement, which the
    # ladder deliberately reports beside the run-pinned headline. A retired-value
    # list has to distinguish "superseded" from "still on the page for contrast".
    # NOT 0.017 either: the retired gate-timing value is textually identical to a
    # legitimate p-value, and supplement S1 prints one. A retired-VALUE check can
    # only cover numbers whose string form is unambiguous; the rest need the
    # provenance binding that verify_pdf_numbers.py provides.
    RetiredTerm(
        pattern=r"\b5\.31\b|\b177,?000\b|\b215k\b|\+60k\b",
        retired_because="superseded by the run-pinned ladder (phase59) and the "
        "run-pinned decision re-run (phase52 R3)",
    ),
    # ---- Internal bookkeeping must not reach the page (2026-08-07) ----
    RetiredTerm(
        # Two leak forms, and only those: "phase 76" in prose or a table
        # column, and a provenance filename cited as evidence. A build header
        # that names the emitting script is not a leak -- it is how a generated
        # file says it is generated -- so the pattern does not reach for it.
        # Widened 2026-08-13. "phase\s\d" needed literal whitespace, so the
        # hyphenated form the supplement actually used ("the phase-51 panel")
        # was invisible; and the filename branch required "_" right after the
        # digits, so every letter-suffixed provenance name escaped -- including
        # phase81b/82b/88b, the three this revision created. Four leaks were
        # live in the shipped supplement with the guard green.
        pattern=(r"phase[\s\-‐-―]*\d+[a-z]?(?:\.\d+[a-z]?)?"
                 r"(?:_[a-z0-9_]+\.json)?"),
        retired_because="'phase NN' is the authors' research-log index. It is "
        "meaningful inside the monorepo and meaningless to a reader; it leaked "
        "into the supplement as provenance filenames and a pre-commitment "
        "ledger column. Cite results by what they are, not by which sprint "
        "produced them",
    ),
    # ---- Fifth review (2026-08): overclaims scoped back to what was tested ----
    RetiredTerm(
        pattern=r"cannot alter (?:a|any) bid",
        retired_because="one-tick equivalence tests the POOLED MAE; it does not "
        "bound per-slot forecast differences or bid changes (fifth review #1)",
    ),
    RetiredTerm(
        pattern=r"adds nothing in the tail",
        retired_because="the tail result is 'no detectable incremental gain' "
        "(CI crosses zero), not a demonstrated null",
    ),
    RetiredTerm(
        pattern=r"decomposes almost entirely",
        retired_because="hedged to 'consistent with vintage being the dominant "
        "component'; the product arm is a bound, not an equivalence",
    ),
    RetiredTerm(
        pattern=r"15\s*hours fresher",
        retired_because="the freshest-admissible tier is 9 hours fresher by "
        "dissemination and ALREADY SCORED (composite 0.231); the sentence "
        "claiming it was unmeasured was wrong twice",
    ),
    RetiredTerm(
        pattern=r"saturates well before",
        retired_because="Gate 41 landed on branch 41b: the three-point "
        "extrapolation is retracted; the measured curve says the plateau "
        "begins BY the D-2 12Z cycle (phase73)",
    ),
    RetiredTerm(
        pattern=r"cuts? system imbalance|system imbalance by 2[0-9]",
        retired_because="one party's gross deviation is not system imbalance; "
        "only the direction-netted figure may be quoted (own Gate 26)",
    ),
]


def _flatten(text: str) -> str:
    r"""Collapse whitespace so a phrase that wrapped across a line still matches.

    ``pdftotext`` preserves the PDF's line breaks, so "saturates well before"
    rendered as "saturates\nwell before" slipped past every RETIRED_TERMS
    pattern. A deliberate-breakage test found this after the audit had been
    trusted for two revisions. Soft hyphens introduced at a break are dropped
    as well, so "photo-\nvoltaic" matches "photovoltaic".
    """
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", text))


def _control_word_spacing_clean(cfg: EpfConfig) -> list[str]:
    r"""Fail on unit macros followed by a bare space and a letter.

    TeX gobbles the space after a control word, so ``\jpykwh while`` renders
    as "JPY/kWhwhile". The bug is invisible in the source (the space IS there)
    and easy to miss in the PDF. Every unit-style macro that expects to be
    followed by prose is listed here; write ``\jpykwh{} while`` instead.
    """
    pat = re.compile(r"\\(?:jpykwh|maeunit)(?=[ ][a-zA-Z])")
    problems: list[str] = []
    for tex in sorted((cfg.project_root / "docs/paper").rglob("*.tex")):
        for i, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                rel = tex.relative_to(cfg.project_root)
                problems.append(
                    f"{rel}:{i}: unit macro followed by bare space+letter -- TeX "
                    "eats the space; use {} after the macro"
                )
    return problems


def _placeholders_resolved(macros_tex: str, cfg: EpfConfig) -> list[str]:
    """--strict only: no exempt placeholder may still be a placeholder.

    Two macros are EXEMPT from provenance binding because no analysis produces
    them -- the replication DOI and the submission commit. Their exemptions say
    the placeholder is "visible in the rendered PDF rather than silent", which is
    true for a human and false for the gate: --strict is described as the
    submission gate and it passed with "10.5281/zenodo.XXXXXXX" printing in the
    Data availability statement. The placeholder is fine while drafting; it is
    not fine at the moment the gate is asked whether this is submittable.
    """
    problems: list[str] = []
    for macro in ("ReplicationDoi", "ReplicationConceptDoi", "ReplicationCommit"):
        m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}", macros_tex)
        if m and re.search(r"X{4,}|TBD|TODO|\bxxx", m.group(1), re.IGNORECASE):
            problems.append(
                f"\\{macro} is still a placeholder ({m.group(1).strip()!r}). "
                "--strict is the submission gate: mint the deposit and the tag, "
                "or run without --strict while drafting")
    return problems


def _figure_terms_clean(cfg: EpfConfig) -> tuple[list[str], list[str]]:
    """Audit figure and submission PDFs for retired wording.

    Figures are the blind spot the prose checks cannot see: a phrase deleted from
    every .tex file survives in a matplotlib-drawn label until someone
    regenerates the figure, and no grep over sections/ will ever find it. The
    graphical abstract shipped three retired claims this way.

    pdftotext does extract vector text from these PDFs, so this works -- but not
    from raster figures, which are listed as unauditable rather than passed over
    in silence.

    Returns ``(problems, warnings)``. The Japanese variant is deliberately
    regenerated in one pass only after the English text settles, so it is
    knowingly stale for most of a revision; reporting it as a hard failure would
    block every commit until then, and reporting nothing is how it came to print
    three superseded headline numbers. It is therefore a counted warning that
    ``--strict`` -- the submission gate -- turns into a failure.
    """
    problems: list[str] = []
    warnings: list[str] = []
    # Archival submission PDFs are historical records: the Energy Economics
    # manuscript is SUPPOSED to still contain the numbers it was submitted with.
    archival = re.compile(r"^ENEECO-|^_archive")
    # Layout-aware, exactly like the macros/provenance resolution above. The
    # research repo keeps the manuscript under docs/paper/<paper>/; a public
    # companion ships no manuscript at all and keeps the figures it reproduces
    # at the repository root. Looking only for the first layout meant the
    # companion's own copy of this guard failed the check it advertises --
    # silently, because nobody re-ran it there after the split.
    # Detect a manuscript by its main.tex, not by the directory: an emitter run
    # can mkdir docs/paper/<paper>/tables, and an empty directory used to be
    # enough to flip this audit into manuscript mode and then hard-fail for
    # finding no PDFs in it.
    roots: list[tuple[Path, bool]] = []
    # The blind trees are tracked rsync snapshots, not build products, so they
    # drift silently -- and they are exactly what a double-anonymised submission
    # would ship. Auditing only the two named trees meant a retired term in a
    # blind figure or a blind PDF was never seen. The blind copies are warnings
    # rather than hard failures: they are regenerated by
    # docs/paper/arxiv_anonymize.sh, so a stale one is a rebuild away, and
    # failing the primary gate on a derived tree would block a submission that
    # is otherwise clean.
    for rel, hard in (("docs/paper/paper_a_main", True),
                      ("docs/paper/paper_a_main_ja", False),
                      ("docs/paper/paper_a_main_blind", False),
                      ("docs/paper/paper_a_main_ja_blind", False)):
        p = cfg.project_root / rel
        if (p / "main.tex").exists():
            roots.append((p, hard))
    manuscript_present = bool(roots)
    if not roots:
        roots = [(cfg.project_root, True)]
    pdfs: list[tuple[Path, bool]] = []  # (path, is_hard_failure)
    for root, hard in roots:
        for sub in ("figures", "submission", "."):
            pdfs += [(p, hard) for p in sorted((root / sub).glob("*.pdf"))
                     if not archival.match(p.name)]
    if not pdfs:
        # With a manuscript in the tree, no PDFs means nobody built it, and the
        # audit would pass vacuously -- fail. Without one, there is genuinely
        # nothing to audit, and saying so is the honest outcome.
        msg = ("no PDFs found to audit for retired terms "
               "(expected figures/ and submission/)")
        return ([msg], []) if manuscript_present else ([], [msg + " -- no manuscript in this repository"])
    for pdf, hard in pdfs:
        sink = problems if hard else warnings
        try:
            # TWO extractions, because neither alone sees every phrase.
            #
            # "-layout" preserves the visual reading order, which is what prose
            # needs. But in a figure with side-by-side boxes it interleaves the
            # columns: a chip reading "recorded photovoltaic / output and
            # curtailment" comes out as "... recorded photovoltaic" on one line
            # and "... output and curtailment" on the next, with three OTHER
            # chips' words in between. No amount of whitespace flattening can
            # rejoin those two words, and this audit reported zero problems on a
            # figure that plainly contained a retired phrase.
            #
            # Plain extraction emits each text block in order, so the chip's own
            # words stay adjacent -- but it scrambles the reading order of
            # multi-column prose. Scanning the concatenation of both means a
            # phrase has to be absent from BOTH renderings to pass.
            parts = []
            for mode in (["-layout"], []):
                parts.append(subprocess.run(
                    ["pdftotext", *mode, str(pdf), "-"],
                    capture_output=True, text=True, check=True, timeout=60,
                ).stdout)
            # A retired phrase that happens to wrap across a PDF line break was
            # invisible to this audit until a deliberate-breakage test caught it:
            # pdftotext emits "saturates\nwell before", which no single-line
            # pattern matches. Collapse every run of whitespace (including the
            # hyphenation LaTeX inserts at a break) before matching, and search
            # the flattened text.
            text = _flatten("\n".join(parts))
        except FileNotFoundError:
            return ([
                "pdftotext not found; install poppler (brew install poppler) -- "
                "the figure audit cannot run without it"
            ], [])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            sink.append(f"{pdf.name}: pdftotext failed ({exc})")
            continue
        for term in RETIRED_TERMS:
            if pdf.name in term.allowed_in:
                continue
            for m in re.finditer(term.pattern, text, re.IGNORECASE):
                ctx = text[max(0, m.start() - 30) : m.end() + 30].replace("\n", " ")
                sink.append(
                    f"{pdf.relative_to(cfg.project_root)}: retired term "
                    f"'{m.group(0)}' -- {term.retired_because}. Context: ...{ctx}..."
                )
    # The .tex SOURCES. Until 2026-08-13 this audit read only PDFs and
    # submission markdown, so a retired term could not be caught where an author
    # types it -- and main.pdf is gitignored, so in a fresh clone the article
    # body was not audited at all. Scanning the sources fixes both: it works
    # without a build, and it names the file and line an author can edit.
    for root, hard in roots:
        sink = problems if hard else warnings
        for tex in sorted((root / "sections").glob("*.tex")) + [root / "main.tex"]:
            if not tex.exists():
                continue
            src = tex.read_text(encoding="utf-8", errors="replace")
            src = re.sub(r"(?m)^\s*%.*$", " ", src)          # comment lines
            # File paths and cross-reference keys are machine identifiers that
            # never reach a reader: tables/phase83_margin_curve.tex is a
            # filename, not a sentence about phase 83. Strip their arguments, or
            # every generated-table \input trips the research-index term.
            src = re.sub(r"\\(?:input|include|includegraphics"
                         r"|label|ref|eqref|cite[a-z]*|bibliography"
                         r"|graphicspath|texttt)\s*(\[[^]]*\])?\{[^}]*\}",
                         " ", src)
            body = _flatten(src)
            for term in RETIRED_TERMS:
                if tex.name in term.allowed_in:
                    continue
                for m in re.finditer(term.pattern, body, re.IGNORECASE):
                    ctx = body[max(0, m.start() - 30): m.end() + 30]
                    sink.append(
                        f"{tex.relative_to(cfg.project_root)}: retired term "
                        f"'{m.group(0)}' -- {term.retired_because}. "
                        f"Context: ...{ctx}...")

    # Hand-written submission .md files (highlights, EM abstract, cover letter
    # sources) are the other blind spot: they are not compiled from macros.tex,
    # so a retired claim edited out of every .tex survives there untouched. The
    # highlights shipped an unqualified gross-deviation percentage this way.
    for md in sorted((cfg.project_root / "docs/paper/paper_a_main/submission").glob("*.md")):
        # README.md is the process document whose "do not resurrect" list MUST
        # quote every retired claim -- scanning it would forbid its own purpose.
        if archival.match(md.name) or md.name == "README.md":
            continue
        raw_md = md.read_text(encoding="utf-8", errors="replace")
        # HTML comments are machine notes -- the generated-by header, a
        # character count, a "do not resurrect" reminder. A reader of the pasted
        # abstract never sees them, and forbidding a retired term inside one
        # would forbid the note that records the retirement. Strip them first.
        raw_md = re.sub(r"<!--.*?-->", " ", raw_md, flags=re.S)
        text = _flatten(raw_md)
        for term in RETIRED_TERMS:
            if md.name in term.allowed_in:
                continue
            for m in re.finditer(term.pattern, text, re.IGNORECASE):
                ctx = text[max(0, m.start() - 30) : m.end() + 30].replace("\n", " ")
                problems.append(
                    f"{md.relative_to(cfg.project_root)}: retired term "
                    f"'{m.group(0)}' -- {term.retired_because}. Context: ...{ctx}..."
                )
    return problems, warnings


def _build_log_clean(cfg: EpfConfig) -> list[str]:
    """Fail on undefined control sequences in whichever PDFs have been built.

    Checked here rather than left to a manual grep, because the manual grep used
    for months looked only for undefined citations and references and therefore
    could not see either defect above.
    """
    problems: list[str] = []
    # "Author undefined" is natbib's response to \citet under a NUMERIC
    # bibliography style: elsarticle-num emits a bare \bibitem with no
    # [Author(Year)] argument, so the citation renders as the literal string
    # "(author?)". Three of those sat in the Related work section -- one page
    # from the abstract -- through every build and every guard run, because
    # this tuple listed only two needles. A rendered "(author?)" is exactly
    # the class of defect this guard exists to stop, so it is fatal now.
    # "There were undefined references" is LaTeX's end-of-run summary when any
    # \ref or \pageref never resolved. It was absent from this tuple, and the
    # cost was not hypothetical: the supplement shipped for two builds with
    # "Section ??" rendered on pages 3 and 40, because supp_s1 and supp_s4 each
    # \ref'd a label that lives in the BODY's compilation unit and therefore can
    # never appear in supplementary.aux. Separate compilation makes that a
    # permanent hazard rather than a typo, so it is fatal now.
    #
    # Needles are matched with str.count, i.e. as plain substrings -- a regex
    # here would silently match nothing and restore the hole.
    fatal = ("Undefined control sequence", "Missing \\begin{document}",
             "Author undefined", "There were undefined references",
             "There were undefined citations")
    # The Japanese tree was never scanned at all; it is clean today, and adding
    # it costs nothing while closing the same gap on that side.
    for rel in ("docs/paper/paper_a_main/main.log",
                "docs/paper/paper_a_main/supplementary.log",
                "docs/paper/paper_a_main_ja/main.log"):
        log_path = cfg.project_root / rel
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for needle in fatal:
            n = text.count(needle)
            if n:
                problems.append(f"{rel}: {n} x '{needle}'")
    return problems


def _emit_unchecked(macros_tex: str) -> int:
    """Print MacroCheck stubs for every unaccounted numeric macro, then exit 0.

    Fill-in-the-blank beats remembering: the point is that adding a macro to the
    paper without adding it here should cost more effort than doing it properly.
    """
    accounted = (
        {c.macro for c in CHECKS}
        | {w[0] for w in WORD_CHECKS}
        | {t[0] for t in STRING_CHECKS}
        | {e.macro for e in EXEMPT}
        | {p.macro for p in PENDING_BINDING}
    )
    n = 0
    for name, body in _MACRO_DEF.findall(macros_tex):
        if name in accounted or not _prints_a_number(body):
            continue
        n += 1
        print(
            f'    MacroCheck(macro="{name}", provenance="TODO.json",\n'
            f'        json_path=("TODO",),  # prints {body[:40]!r}\n'
            f'        note="TODO"),'
        )
    print(f"\n# {n} unaccounted numeric macro(s).", file=sys.stderr)
    return 0


def _emit_reproducibility() -> int:
    r"""Print the macro -> provenance index as Markdown, then exit 0.

    ``REPRODUCIBILITY.md`` in the public companion is the traceability map the
    supplement points readers at, and it is hand-maintained -- so it drifts out
    of step with CHECKS exactly the way the paper's literals drifted out of step
    with provenance. Generating the table from CHECKS removes the copy.

    Emits only the table; the surrounding narrative in that file is prose worth
    keeping by hand. Redirect into the file's table section.
    """
    print("| Macro | Provenance JSON | JSON path | Note |")
    print("| --- | --- | --- | --- |")
    for chk in sorted(CHECKS, key=lambda c: (c.provenance, c.macro)):
        path = ".".join(str(k) for k in chk.json_path)
        extra = ""
        if chk.subtract_json_path is not None:
            sub = ".".join(str(k) for k in chk.subtract_json_path)
            extra += f" − `{sub}`"
        if chk.denominator_json_path is not None:
            den = ".".join(str(k) for k in chk.denominator_json_path)
            extra += f" ÷ `{den}`" + (" (complement)" if chk.complement else "")
        scale = "" if chk.scale == 1.0 else f" ×{chk.scale:g}"
        print(f"| `\\{chk.macro}` | `{chk.provenance}` | `{path}`{extra}{scale} "
              f"| {chk.note} |")
    for macro, provenance, path, note in WORD_CHECKS:
        print(f"| `\\{macro}` | `{provenance}` | `{'.'.join(path)}` "
              f"| {note} (spelled out) |")
    print(f"\n{len(CHECKS)} value checks + {len(WORD_CHECKS)} word checks. "
          f"{len(EXEMPT)} numeric macros are exempt with a stated source; "
          f"{len(PENDING_BINDING)} await a binding and are listed in "
          f"`PENDING_BINDING` in `verify_macros.py`.", file=sys.stderr)
    return 0



# Numbers that legitimately appear in the submission prose without a macro: they
# are not results. Anything else must be the body of some macro, so a paper number
# that moves cannot leave a stale copy behind in the letter an editor reads first.
# Each exemption states why the number is not a result. A number with no reason
# here and no macro is a stale copy, which is the whole point of the check.
# "~13,950 words" and the like: verified by _submission_counts_current against
# texcount, so exempted by rule rather than by pinning the value.
WORD_COUNT_IN_PROSE = re.compile(r"~?\s*\d{1,3}(?:,\d{3})+\s+words")

PROSE_NUMBER_EXEMPT: dict[str, str] = {
    "0.0001": "a p-value THRESHOLD (\"p < 0.0001\"), not an estimate",
    "0.05": "conventional level, quoted as a level",
    "0.3": "a correlation threshold the proxy audit reports against, not an estimate",
    "10.5281": "the Zenodo DOI prefix",
    # The old exemption here read "page and word counts are checked ... by
    # _submission_counts_current". Pages were; the WORD count never was, and the
    # README drifted to 8,900 against a real 13,646 -- understating the manuscript
    # by 35% in the file an editor reads first. An exemption whose stated reason
    # is false is worse than no exemption, because it stops anyone looking again.
    # No word-count literal lives here any more. The figure quoted as "~N words"
    # is checked against texcount by _submission_counts_current, so pinning its
    # value in this dict would mean editing two places per revision -- and the
    # one time that is forgotten, the number silently becomes unbacked again.
    # _prose_numbers_backed skips it by RULE instead; see WORD_COUNT_IN_PROSE.
    "177,000": "quoted from the RETIRED per-MW-year valuation, in the list of "
               "claims this paper no longer makes",
}


def _submission_counts_current(cfg: EpfConfig) -> list[str]:
    r"""The submission README's page AND word counts must match the built article.

    It said 54 pp and 64 pp while the PDFs were 56 and 65: the revision added
    pages and the checklist an editor reads did not move with them. Page counts
    are not provenance so no macro holds them, which is exactly why nothing
    caught this.

    The word count was a worse case, because PROSE_NUMBER_EXEMPT justified
    leaving it unbound on the grounds that *this function* checked it. It did
    not. The README read "~8,900 words" against a real 13,646.
    """
    out: list[str] = []
    readme = (cfg.project_root
              / "docs/paper/paper_a_main/submission/README.md")
    if not readme.exists():
        return out
    text = readme.read_text(encoding="utf-8")
    for rel, label in (("docs/paper/paper_a_main/main.pdf", "main.pdf"),
                       ("docs/paper/paper_a_main/supplementary.pdf",
                        "supplementary.pdf")):
        pdf = cfg.project_root / rel
        if not pdf.exists():
            continue
        try:
            info = subprocess.run(["pdfinfo", str(pdf)], check=True,
                                  capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        m = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        if not m:
            continue
        actual = m.group(1)
        # the row that names this file, and the pp figure on it
        row = next((ln for ln in text.splitlines() if rel.split("/")[-1] in ln
                    and " pp" in ln), None)
        if row is None:
            continue
        stated = re.search(r"(\d+)\s*pp", row)
        if stated and stated.group(1) != actual:
            out.append(
                f"README.md says {stated.group(1)} pp for {label}; the built "
                f"PDF has {actual}")

    # The body WORD count, which this function was credited with checking for
    # months and did not. The README drifted to "~8,900 words" against a real
    # 13,646 -- a 35% understatement of the manuscript's length, in the file an
    # editor opens first, and length is a screening criterion at this journal.
    #
    # Checked against the same command the README's own checklist names, so the
    # figure and the instruction for reproducing it cannot disagree. A 2%
    # tolerance, because the README quotes a rounded "~" figure on purpose.
    # Page counts also appear parenthetically in the checklist, e.g. "(67 pp)",
    # and the row-scoped check above never saw them: one sat at 61 while the PDF
    # was 67. Any "(N pp)" in the README must match main.pdf.
    main_pdf = cfg.project_root / "docs/paper/paper_a_main/main.pdf"
    if main_pdf.exists():
        try:
            info = subprocess.run(["pdfinfo", str(main_pdf)], check=True,
                                  capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            info = ""
        mp = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        if mp:
            for stale in {g for g in re.findall(r"\((\d+) pp\)", text)
                          if g != mp.group(1)}:
                out.append(
                    f"README.md has a parenthetical '({stale} pp)'; main.pdf has "
                    f"{mp.group(1)} pages")

    stated_w = re.search(r"~([\d,]+) words", text)
    if stated_w:
        main_tex = cfg.project_root / "docs/paper/paper_a_main/main.tex"
        try:
            tc = subprocess.run(
                ["texcount", "-inc", "-total", "-q", str(main_tex.name)],
                cwd=main_tex.parent, check=True, capture_output=True, text=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            # Absent tooling must not read as a passing check. Say so instead.
            out.append(
                "README.md states a body word count but texcount is not "
                "available, so it cannot be verified. Install texcount or "
                "remove the figure -- an unverifiable number in the submission "
                "README is what produced the 8,900-against-13,646 drift")
        else:
            m_w = re.search(r"Words in text:\s+(\d+)", tc)
            if m_w:
                want = int(m_w.group(1))
                got = int(stated_w.group(1).replace(",", ""))
                if abs(got - want) > max(1, round(0.02 * want)):
                    out.append(
                        f"README.md says ~{stated_w.group(1)} words of body "
                        f"text; texcount reports {want:,} (tolerance 2%)")
    return out


# The title, as chosen by the PI on 2026-08-21. Five files carry it and none of
# them derives it from the others, so it is pinned here and compared.
EXPECTED_TITLE = ("Dissemination-aware weather forecast vintages for day-ahead "
                  "electricity price forecasting in Japan")


def _title_consistent(cfg: EpfConfig) -> list[str]:
    r"""The article's title must read identically wherever it is written down.

    It is written in five places that derive it from nothing: main.tex, the
    supplement's \title, the Japanese tree's \title, submission/title_page.md and
    submission/_title_page_clean.md. The README already carried a note that "the
    stale PDF carries an older title", which is what this class of drift looks
    like when nobody is checking -- and when the PI chose among three candidates,
    four of the five still said the old one.

    Line breaks and LaTeX line-continuation markers are normalised away, because
    the title is wrapped differently in each file for typesetting reasons and
    those differences are not drift.
    """
    def norm(t: str) -> str:
        t = t.replace("\\\\", " ").replace("\\textbf", " ").replace("**", " ")
        t = re.sub(r"[{}]", " ", t)
        return re.sub(r"\s+", " ", t).strip().lower()

    want = norm(EXPECTED_TITLE)
    out: list[str] = []
    # As with _macros_printed: the replication package ships this guard but not
    # the manuscript, so "the title is missing from main.tex" is the companion's
    # normal state rather than a finding. Decline where there is no manuscript.
    if not (cfg.project_root / "docs/paper/paper_a_main/main.tex").exists():
        return []
    checks = [
        ("docs/paper/paper_a_main/main.tex", r"\\title\{(.*?)\}\s*\n\s*\n", None),
        ("docs/paper/paper_a_main/supplementary.tex",
         r"Supplementary material for:(.*?)\}\}", None),
        ("docs/paper/paper_a_main/submission/title_page.md",
         r"## Title\s*\n\s*\n(.*?)\n\s*\n", None),
        ("docs/paper/paper_a_main/submission/_title_page_clean.md",
         r"\A\s*\*\*(.*?)\*\*", None),
    ]
    for rel, pat, _ in checks:
        f = cfg.project_root / rel
        if not f.exists():
            out.append(f"{rel}: missing, so the title cannot be checked there")
            continue
        m = re.search(pat, f.read_text(encoding="utf-8"), re.S)
        if not m:
            out.append(f"{rel}: could not locate the title to compare it")
            continue
        got = norm(m.group(1))
        if got != want:
            out.append(f"{rel}: title reads '{got[:78]}' but should read "
                       f"'{want[:78]}'")
    return out


def _provenance_paths_portable(cfg: EpfConfig) -> list[str]:
    r"""No provenance artefact may record an absolute path from this machine.

    The provenance JSONs are shipped verbatim in the public replication package,
    and nine of them recorded absolute paths like
    ``/Users/<name>/main_desk/.../data/raw/jepx/spot_2024.csv``. That leaks the
    author's filesystem layout into a public artefact and tells a reader nothing:
    the path is only meaningful on the machine that wrote it.

    Scripts should stamp repository-relative paths. This check exists because
    several did not, and because the failure is invisible in the research
    checkout -- the paths are correct there, which is exactly why nobody looked.

    Both locations are scanned. In the research checkout the JSONs live in
    ``run_dir``; in the public package they live in ``data/provenance``, and an
    earlier version of this check looked only at ``run_dir``, returned early
    because that directory does not exist there, and so passed vacuously in the
    one repository whose contents it exists to protect. That is the same
    "invisible in the research checkout" failure this docstring warns about,
    pointed the other way.
    """
    home = str(Path.home())
    bad: list[str] = []
    seen: set[Path] = set()
    for d in (cfg.resolved("run_dir"), Path(__file__).resolve().parent.parent / "data/provenance"):
        if not d.exists() or d in seen:
            continue
        seen.add(d)
        for f in sorted(d.rglob("*.json")):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if home in text:
                n = text.count(home)
                bad.append(f"{f.name}: records {n} absolute path(s) under the home "
                           "directory; stamp repository-relative paths instead")
    return bad


def _prose_numbers_backed(macros_tex: str, cfg: EpfConfig) -> list[str]:
    r"""Every result-shaped number in the submission prose must be a macro value.

    The cover letter, the declarations and the package README are hand-written and
    always will be -- they are addressed to people, not compiled. But they quote
    the paper's numbers, and nothing checked those copies: the letter cited the
    TOST p from the block bootstrap (0.023) where the manuscript's primary is
    0.024, which is the more favourable of two and was not labelled as either.

    This does not check that a number is CORRECT -- it checks that some macro
    holds it. A value that no macro holds is either stale or invented, and both
    are reasons to stop. Mislabelling (quoting the right value of the wrong
    quantity) is beyond what a value check can see; that one is on the reader.
    """
    defs = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                   macros_tex)
    )

    def _norm(x: str) -> str:
        return x.replace("{,}", "").replace(",", "").strip()

    known = {_norm(v) for v in defs.values()}
    out: list[str] = []
    sub_dir = cfg.project_root / "docs/paper/paper_a_main/submission"
    for md in sorted(sub_dir.glob("*.md")):
        if md.name.startswith("_"):
            continue  # generated from macros already, and checked as artefacts
        text = md.read_text(encoding="utf-8")
        # Section references are numbered too. "§4.1 Data" is not a result, and
        # exempting the literal "4.1" would exempt it everywhere -- including
        # where it might one day be a real quantity.
        text = re.sub(r"(?:§|Sections?~?\s*|S)\d+(?:\.\d+)*", " ", text)
        # The body word count is quoted as "~13,950 words". It is not a result
        # and no macro holds it, but it IS verified -- _submission_counts_current
        # checks it against the same texcount command the README's checklist
        # names. Skipped by rule so the two checks cannot drift apart.
        text = WORD_COUNT_IN_PROSE.sub(" ", text)
        for raw in dict.fromkeys(
                re.findall(r"(?<![\w.])(\d+\.\d+|\d{1,3}(?:,\d{3})+)(?![\w])",
                           text)):
            # Sign-insensitive: the manuscript's macro bodies carry the sign
            # (-0.082) while the prose writes the magnitude next to a minus that
            # is not part of the token, so a signed result read as unbacked.
            if (raw in PROSE_NUMBER_EXEMPT or _norm(raw) in known
                    or _norm(raw).lstrip("-") in {k.lstrip("-") for k in known}):
                continue
            out.append(
                f"{md.name}: '{raw}' is not the value of any macro -- either it "
                f"is stale, or it is a number this paper does not report")
    return out



# Pre-registered gates whose verdict the manuscript depends on. Each entry says
# which provenance holds the gate, and what this manuscript was written against.
#
# "pass" means the branch that licenses the paper's claim fired.
# "fired:<doc>" means the OTHER branch fired and the manuscript honoured it; the
# doc is the research entry recording what was changed in response. Declaring a
# fired branch is how an honoured null stays honoured: if a later run flips it
# back to passing, this check stops the build and asks whether the concession in
# the manuscript is now unearned.
#
# Added 2026-08-14. Nothing connected gate verdicts to the build before, so the
# capacity-weighted full grid could write passes=false for all four floor gates
# -- and instructions to withdraw a framing from Section 3 and to qualify the
# abstract -- while every check stayed green.
GATE_EXPECTATIONS: list[tuple[str, str, str, str]] = [
    ("phase66v2_floor_regime_v3_cw_full.json", "gate18_regime_claim", "pass",
     "licenses the 'predicts a regime, not only a level' framing in Section 3"),
    ("phase66v2_floor_regime_v3_cw_full.json", "gate19_gate_timing_in_tail",
     "pass", "branch 19a: the gate-timing null extends beyond MAE to regime "
     "detection, which is what Section 5.2 reports"),
    ("phase66v2_floor_regime_v3_cw_full.json", "gate20_threshold_stability",
     "pass", "the sign is identical at all four floor thresholds, which "
     "Section 5.2 states"),
    ("phase66v2_floor_regime_v3_cw_full.json", "gate21_midday", "pass",
     "the contrast holds on midday slots alone"),
]


def _gates_honoured(_resolve_prov) -> list[str]:
    r"""Every pre-registered gate the manuscript leans on still says what it said.

    A gate is not a diagnostic: each carries an if_fails or a branch reading that
    prescribes a change to the paper. So a gate that flips, or that a later run
    leaves unevaluated, has to stop the build rather than sit in a JSON nobody
    opens. Unevaluated is its own failure here -- it is precisely the state the
    full-grid run was in while printing passes=false.
    """
    out: list[str] = []
    for prov, gate, expected, why in GATE_EXPECTATIONS:
        path = _resolve_prov(prov)
        if not path.exists():
            out.append(f"gate {gate}: provenance missing ({path})")
            continue
        node = json.loads(path.read_text()).get("gates", {}).get(gate)
        if node is None:
            out.append(f"gate {gate}: absent from {prov}")
            continue
        # `.get("evaluated", True)` -- absence defaulting to True -- is the very
        # pattern this check exists to catch: a gates block written before the
        # `evaluated` key existed would pass on a missing field. The key was added
        # 2026-08-14 and every gate this registry names carries it, so require it.
        if node.get("evaluated") is not True:
            out.append(
                f"gate {gate} was NOT EVALUATED in {prov} "
                f"({node.get('note') or node.get('if_fails')}). The manuscript "
                f"leans on it: {why}. Re-run the phase with the pairing this "
                f"gate reads, or re-derive the verdicts with --step regates.")
            continue
        passes = node.get("passes")
        actual = "pass" if passes else "fired"
        want = expected.split(":", 1)[0]
        if actual != want:
            prescription = (node.get("if_fails") or node.get("reading") or "")
            out.append(
                f"gate {gate} in {prov} is now '{actual}', declared '{expected}'. "
                f"The manuscript leans on it: {why}. "
                + (f"The branch prescribes: {prescription}" if actual == "fired"
                   else "A concession in the manuscript may now be unearned; "
                        "check the doc named in GATE_EXPECTATIONS."))
    return out



def _blind_trees_current(cfg: EpfConfig) -> list[str]:
    r"""Each blind tree's sections are byte-identical to the tree it came from.

    The anonymiser takes the tree as an argument and processes ONE tree per call,
    so running it bare refreshes only the English blind copy. The Japanese blind
    tree sat on superseded text for an unknown number of build cycles while the
    verification loop reported both blind PDFs as clean -- its self-check looks
    for identifying tokens, and stale text carries none.

    The blind trees are rsync'd from source and patched only in main.tex, so the
    section files must match byte for byte. A blind PDF is what a reviewer reads;
    stale content there is a failure, not a warning.
    """
    out: list[str] = []
    base = cfg.project_root / "docs/paper"
    for src, blind in (("paper_a_main", "paper_a_main_blind"),
                       ("paper_a_main_ja", "paper_a_main_ja_blind")):
        sd, bd = base / src / "sections", base / blind / "sections"
        if not sd.is_dir() or not bd.is_dir():
            continue
        for f in sorted(sd.glob("*.tex")):
            b = bd / f.name
            if not b.exists():
                out.append(f"{blind}: {f.name} is missing; re-run "
                           f"arxiv_anonymize.sh {src}")
            elif f.read_bytes() != b.read_bytes():
                out.append(
                    f"{blind}/sections/{f.name} differs from {src}: the blind "
                    f"copy is stale. Re-run docs/paper/arxiv_anonymize.sh {src}")
    return out


def _printed_precision_tol(macros_tex: str, macro: str) -> float | None:
    r"""Half a unit in the last decimal place the macro actually prints.

    A tolerance defines the set of WRONG values that still pass, so the only
    defensible default is the rounding the printed precision implies and nothing
    more. The dataclass default is a flat 5e-3, which for a quantity printed to
    three decimals admits a value 0.005 away -- enough to flip the sign of every
    small signed difference in this paper (\GateTimingValue prints 0.003 and
    5e-3 accepted -0.002) and enough to carry a p-value across 0.05.

    Returns None when the macro prints no decimal point; the caller then keeps
    the declared tolerance.
    """
    m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{([^}}]*)\}}", macros_tex)
    if not m:
        return None
    body = m.group(1).replace("{,}", "").replace(",", "")
    decs = re.findall(r"-?\d+\.(\d+)", body)
    if not decs:
        return None
    return 0.5 * 10 ** (-min(len(d) for d in decs)) + 1e-9


def _effective_tol(chk: "MacroCheck", macros_tex: str) -> float:
    """chk.tol when set deliberately, else the printed-precision default."""
    if "tol" in chk.model_fields_set:
        return chk.tol
    derived = _printed_precision_tol(macros_tex, chk.macro)
    return chk.tol if derived is None else derived


def _tolerances_respect_boundaries(_resolve_prov, macros_tex: str) -> list[str]:
    r"""No check may tolerate a value on the other side of a decision boundary.

    For most quantities a loose tolerance is merely sloppy. For two kinds it is a
    hole: an interval endpoint or signed delta near ZERO, where a tolerance wider
    than the distance to zero admits the opposite sign; and a p-value near a
    conventional level, where it admits the other side of significance. Both are
    conclusions this manuscript states in words, so a passing wrong value would
    contradict its own text. Found by an independent audit, 2026-08-18.
    """
    out: list[str] = []
    for chk in CHECKS:
        if chk.tol is None or chk.paper_b:
            continue
        path = _resolve_prov(chk.provenance)
        if not path.exists():
            continue
        try:
            raw = _dig(json.loads(path.read_text()), chk.json_path)
        except (KeyError, IndexError, TypeError):
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        tol = _effective_tol(chk, macros_tex)
        val = float(raw) * chk.scale
        name = chk.macro
        # p-values are detected from the LAST path element, not a substring of the
        # whole path: "arm_cw_primary" contains "m_p", which made an improvement
        # in JPY/kWh read as a p-value straddling 0.05.
        leaf = str(chk.json_path[-1]).lower() if chk.json_path else ""
        looks_p = (name.endswith(("P", "Pval", "PValue"))
                   or leaf in {"p", "pval", "p_value", "p_two_sided"}
                   or leaf.startswith(("p_", "pval"))
                   or leaf.endswith(("_p", "_pval", "_p_two_sided")))
        jp = ".".join(str(x) for x in chk.json_path).lower()
        looks_signed = (name.endswith(("Lo", "Hi", "Delta", "Diff", "Margin"))
                        or "ci95" in jp or "delta" in jp)
        if looks_signed and val != 0 and tol >= abs(val):
            out.append(
                f"\\{name}: tol {tol:g} >= |value| {abs(val):g}, so a value of the "
                f"OPPOSITE SIGN would pass. This is an interval endpoint or a "
                f"signed difference; declare a tol below {abs(val):g}")
        if looks_p:
            for level in (0.05, 0.10):
                if val < level <= val + tol or val - tol <= level < val:
                    out.append(
                        f"\\{name}: tol {tol:g} around p = {val:g} spans the "
                        f"{level:g} level, so a value on the other side of "
                        f"significance would pass; declare a tol below "
                        f"{abs(level - val):g}")
                    break
    return out


def main() -> int:
    cfg = EpfConfig()
    # This file is shared verbatim between the research repo and the public
    # companion, whose layouts differ: the research repo keeps macros under
    # docs/paper/shared/ and provenance under data/runs/, the companion keeps
    # them at docs/macros.tex and data/provenance/. Resolve whichever exists so a
    # blind copy in either direction cannot break the guard.
    macros_candidates = (
        cfg.project_root / "docs/paper/shared/macros.tex",
        cfg.project_root / "docs/macros.tex",
    )
    macros_path = next((c for c in macros_candidates if c.exists()), None)
    if macros_path is None:
        msg = f"macros.tex not found at any of {[str(c) for c in macros_candidates]}"
        raise FileNotFoundError(msg)
    macros_tex = macros_path.read_text(encoding="utf-8")
    # Choose by CONTENT, not existence: both layouts may have the directory, but
    # only one holds the provenance JSONs. Picking the merely-existing one gives a
    # wall of "provenance missing" failures that look like real drift.
    # PER-FILE resolution, not a single directory pick. The old rule chose the
    # directory with more JSONs, which meant that in the replication package a
    # reader who ran `make reproduce` still had every check read the frozen
    # data/provenance/ copy -- their freshly regenerated numbers under
    # data/runs/ were never looked at. Prefer the regenerated file when it
    # exists, fall back to the frozen one per file.
    run_candidates = (cfg.resolved("run_dir"), cfg.project_root / "data/provenance")

    def _resolve_prov(rel: str) -> Path:
        for c in run_candidates:
            if (c / rel).exists():
                return c / rel
        return run_candidates[0] / rel  # keep the old path in the error message

    strict = "--strict" in sys.argv
    if "--emit-unchecked" in sys.argv:
        return _emit_unchecked(macros_tex)
    if "--emit-reproducibility" in sys.argv:
        return _emit_reproducibility()

    failures: list[str] = []
    checked = 0
    failures.extend(_macro_hygiene(macros_tex))
    if strict:
        failures.extend(_placeholders_resolved(macros_tex, cfg))
    failures.extend(_paper_b_block(macros_tex))
    failures.extend(_build_log_clean(cfg))
    failures.extend(_prose_numbers_backed(macros_tex, cfg))
    failures.extend(_submission_counts_current(cfg))
    failures.extend(_title_consistent(cfg))
    failures.extend(_provenance_paths_portable(cfg))
    failures.extend(_gates_honoured(_resolve_prov))
    failures.extend(_blind_trees_current(cfg))
    failures.extend(_tolerances_respect_boundaries(_resolve_prov, macros_tex))
    failures.extend(_double_period_clean(cfg))
    failures.extend(_control_word_spacing_clean(cfg))
    fig_problems, fig_warnings = _figure_terms_clean(cfg)
    failures.extend(fig_problems)
    unchecked, pending = _unchecked_numeric_macros(macros_tex)
    failures.extend(unchecked)
    failures.extend(_macros_printed(cfg, macros_tex))
    skipped_paper_b: list[str] = []
    for chk in CHECKS:
        prov_path = _resolve_prov(chk.provenance)
        if not prov_path.exists():
            if chk.paper_b:
                skipped_paper_b.append(chk.macro)
                continue
            failures.append(f"\\{chk.macro}: provenance missing ({prov_path})")
            continue
        prov_json = json.loads(prov_path.read_text())
        if chk.aggregate is not None:
            container = _dig(prov_json, chk.json_path)
            if not isinstance(container, (dict, list)) or not container:
                failures.append(f"\\{chk.macro}: aggregate target is not a non-empty "
                                f"object at {'/'.join(map(str, chk.json_path))}")
                continue
            if chk.aggregate == "count":
                # Count POPULATED entries. len() alone certifies the shape of a
                # container, not that anything was computed into it: a crosstab
                # written as eight nulls by a failed run satisfies "eight cells".
                entries = (container.values() if isinstance(container, dict)
                           else container)
                populated = [e for e in entries if e is not None]
                if len(populated) != len(container):
                    failures.append(
                        f"\\{chk.macro}: {len(container) - len(populated)} of "
                        f"{len(container)} entries at "
                        f"{'/'.join(map(str, chk.json_path))} are null; a count "
                        f"over empty cells is not the count the macro claims")
                    continue
                prov_val = float(len(populated))
            else:
                vals = [float(e[chk.leaf_key]) if chk.leaf_key else float(e)
                        for e in container.values()]
                prov_val = min(vals) if chk.aggregate == "min" else max(vals)
                if chk.aggregate_key is not None and isinstance(container, dict):
                    picked = (min if chk.aggregate == "min" else max)(
                        container, key=lambda k: float(
                            container[k][chk.leaf_key] if chk.leaf_key
                            else container[k]))
                    if picked != chk.aggregate_key:
                        failures.append(
                            f"\\{chk.macro}: the {chk.aggregate} is now "
                            f"'{picked}', not '{chk.aggregate_key}' -- the "
                            f"sentence naming it is wrong even if the number "
                            f"still matches")
        else:
            prov_val = float(_dig(prov_json, chk.json_path))
        if chk.subtract_json_path is not None:
            prov_val -= float(_dig(prov_json, chk.subtract_json_path))
        if chk.denominator_json_path is not None:
            den = float(_dig(prov_json, chk.denominator_json_path))
            if den == 0.0:
                failures.append(f"\\{chk.macro}: ratio denominator is zero")
                continue
            prov_val = prov_val / den
            if chk.complement:
                prov_val = 1.0 - prov_val
        prov_val *= chk.scale
        tex_floats = _macro_floats(macros_tex, chk.macro)
        eff_tol = _effective_tol(chk, macros_tex)
        if not any(abs(prov_val - t) <= eff_tol for t in tex_floats):
            failures.append(
                f"\\{chk.macro}: macros.tex {tex_floats} vs provenance {prov_val:.4f} "
                f"(tol {eff_tol:g}) — {chk.note}",
            )
        else:
            checked += 1
            print(f"  ok  \\{chk.macro} = {prov_val:+.4f}  [{chk.note}]")

    for macro, provenance, path, note in WORD_CHECKS:
        prov_path = _resolve_prov(provenance)
        if not prov_path.exists():
            failures.append(f"\\{macro}: provenance missing ({prov_path})")
            continue
        raw = _dig(json.loads(prov_path.read_text()), path)
        # a container counts its members; a scalar is the count itself. The
        # per-area blocks of the newer provenance are dicts keyed by area,
        # and int(dict) raised rather than counting them.
        count = len(raw) if isinstance(raw, (list, dict)) else int(raw)
        # Whether a count is spelled out or printed as digits is a style choice,
        # not a fact, so BOTH renderings are accepted -- what the check asserts
        # is the VALUE. Before this, a count above ten demanded the digit form
        # and failed a correctly spelled-out macro.
        accepted = {str(count), f"{count:,}".replace(",", "{,}")}
        if count in NUMBER_WORDS:
            accepted.add(NUMBER_WORDS[count])
        want = NUMBER_WORDS.get(count, str(count))
        # brace-aware: a prose macro such as "the $D{-}2$ 12\,UTC run" contains
        # nested braces, and the naive [^}]* stopped at the first one, handing the
        # checker "the $D{-" and no digits at all.
        m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}",
                      macros_tex)
        got = m.group(1).strip() if m else None
        if got not in accepted:
            failures.append(
                f"\\{macro}: macros.tex '{got}' vs provenance '{want}' "
                f"(count {count}; accepted {sorted(accepted)}) — {note}"
            )
        else:
            checked += 1
            print(f"  ok  \\{macro} = {want}  [{note}]")

    for macro, provenance, path, mode, note in STRING_CHECKS:
        prov_path = _resolve_prov(provenance)
        if not prov_path.exists():
            failures.append(f"\\{macro}: provenance missing ({prov_path})")
            continue
        raw = _dig(json.loads(prov_path.read_text()), path)
        # brace-aware, as in the word checks above: "the $D{-}2$ 12\,UTC run"
        # has nested braces and the naive [^}]* returned "the $D{-".
        m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}",
                      macros_tex)
        got = m.group(1).strip() if m else None
        if mode == "substring":
            # A container's repr is not a haystack: comparing against
            # str(['2024-09', '2025-12', 'equal']) let EVERY arm name -- and many
            # nonsense fragments -- satisfy the check, so binding the macro to
            # the wrong arm passed. Lists and dicts now require exact membership;
            # only a genuinely free-text scalar is searched as a substring.
            if isinstance(raw, dict):
                ok = got in raw
                want = f"one of the keys {sorted(raw)}"
            elif isinstance(raw, (list, tuple)):
                ok = got in [str(x) for x in raw]
                want = f"one of {[str(x) for x in raw]}"
            else:
                ok = got is not None and got in str(raw)
                want = f"a substring of '{raw}'"
        elif mode == "hourspan":
            # The provenance records an INCLUSIVE hour filter (polars'
            # is_between defaults to closed="both"), and the panel's slots are
            # half-hourly labelled by start. So hours [9, 15] keep fourteen
            # slots per area-day, 09:00 through 15:30 -- not six hours ending at
            # 15:00. Rendering the upper bound as "15:00" dropped the last hour
            # AND matched a manuscript sentence that was wrong the same way, so
            # the check could not fail however the window moved.
            want = f"{int(raw[0]):02d}:00--{int(raw[1]):02d}:30"
            ok = got == want
        elif mode == "hhmm":
            # Accept either the LaTeX (~) or the Japanese spacing, and require
            # the rendering to be EXACT: "about 2 h" for 136 minutes would be
            # true-ish prose and a silent drift, so a check that tolerated it
            # would not be a check.
            whole = int(round(float(raw)))
            digits = [int(x) for x in re.findall(r"\d+", got or "")]
            ok = len(digits) == 2 and digits[0] * 60 + digits[1] == whole
            want = (f"{whole // 60} h {whole % 60} min "
                    f"(= {whole} minutes, from {raw})")
        elif mode == "tierslug":
            # provenance is a slug like "msm_d2_12z"; the macro is prose such as
            # "the $D{-}2$ 12\,UTC run". Require BOTH numbers the slug encodes to
            # appear, so re-selecting the headline tier cannot leave the sentence
            # naming the previous run.
            sm = re.match(r"msm_d(\d+)_(\d+)z", str(raw))
            if sm is None:
                ok, want = False, f"a slug of the form msm_dN_HHz (got {raw!r})"
            else:
                day, hour = sm.group(1), str(int(sm.group(2)))
                digits = re.findall(r"\d+", got or "")
                ok = day in digits and hour in digits
                want = (f"prose naming D-{day} and {hour} UTC "
                        f"(from slug {raw!r}); got digits {digits}")
        else:  # an unknown mode is a bug in this file, not a paper error
            raise AssertionError(f"unknown STRING_CHECKS mode {mode!r}")
        if not ok:
            failures.append(
                f"\\{macro}: macros.tex '{got}' vs provenance {want} — {note}")
        else:
            checked += 1
            print(f"  ok  \\{macro} = {got}  [{note}]")

    if pending:
        print(
            f"\n{len(pending)} numeric macro(s) awaiting a provenance binding "
            f"(--strict fails on these):"
        )
        for p in pending:
            print(f"  TODO {p}")

    if fig_warnings:
        print(
            f"\n{len(fig_warnings)} retired term(s) in the Japanese variant, which is "
            f"regenerated in one pass after the English text settles "
            f"(--strict fails on these):"
        )
        for w in fig_warnings[:8]:
            print(f"  WARN {w}")
        if len(fig_warnings) > 8:
            print(f"  ... and {len(fig_warnings) - 8} more")

    if failures:
        print("\nMACRO DRIFT DETECTED:", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    if strict and (pending or fig_warnings):
        if pending:
            print(f"\n--strict: {len(pending)} macro(s) still unbound.", file=sys.stderr)
        if fig_warnings:
            print(f"--strict: {len(fig_warnings)} retired term(s) in the Japanese "
                  f"variant; regenerate it before submission.", file=sys.stderr)
        return 1
    if skipped_paper_b:
        shown = ", ".join("\\" + m for m in sorted(skipped_paper_b)[:4])
        more = ", ..." if len(skipped_paper_b) > 4 else ""
        print(f"\n{len(skipped_paper_b)} settlement-paper check(s) skipped: this "
              f"checkout does not ship their provenance ({shown}{more}).")
        # A skip is acceptable while drafting in a checkout that does not carry
        # the settlement provenance. It is NOT acceptable at the submission gate,
        # where "checked" must mean checked: an unverified value would ship under
        # a notice nobody reads.
        # A PARTIAL set is the dangerous case: this checkout carries the
        # settlement provenance but some files are missing, so a real gap hides
        # behind a notice. A WHOLESALE absence is the replication package's known
        # shape -- it ships this paper only, and _paper_b_block proves those
        # macros are not printed here -- so failing there would hand a reader 120
        # failures about a different paper.
        n_paper_b = sum(1 for c in CHECKS if c.paper_b)
        if strict and len(skipped_paper_b) < n_paper_b:
            failures.append(
                f"--strict: {len(skipped_paper_b)} of {n_paper_b} "
                f"settlement-paper check(s) were skipped for missing provenance "
                f"({shown}{more}) while the rest resolved. A partial set means "
                f"this checkout should have them; the submission gate cannot "
                f"pass on unverified macros.")
    print(
        f"\nAll {checked} headline macros match their provenance; "
        f"{len(EXEMPT)} exempt, {len(pending)} pending."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
