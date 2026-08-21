# Scripts

One file per analysis phase. The number prefix is **pipeline order**, not
importance: `0x` acquires and extracts, `1x` is the accuracy ladder, `3x`
floor-event forecasting, `4x` the timing audit, `5x` the expectations
channel, `6x` robustness, `7x` the literature audit, `9x` the outputs.
`verify_macros.py` carries no number because it is a checker you can run at
any time, against no raw data.

**Scope note (2026-08).** This package reproduces the forecast-vintage paper.
The settlement analyses that earlier versions shipped here --- the commitment
decision model, nomination rules, the settlement wedge, the two published
prices, and netting --- belong to a companion manuscript and were removed
rather than left to rot against provenance this package no longer carries.
Their extraction entries are listed as retired in
`tools/extract_from_monorepo.py`, so re-running extraction deletes them if an
older checkout still has them.

## Why the names differ from the provenance files

These scripts are extracted from the authors' research monorepo, where the
sandbox convention is `s_<date>_<phase>_<topic>.py`. The date and the `s_`
prefix carry no meaning outside that repository, so they are dropped here. The
**provenance JSON filenames are unchanged** — they are frozen artefacts, several
are cited by name in the paper's supplement, and rewriting a record of what ran
would defeat the point of having one.

The table below is therefore the bridge: paper → provenance JSON → script here
→ the research-repo filename that the JSON's own `metadata.script` records.
Each script's docstring still opens with its phase number.

| Script | Phase | Provenance JSON | Paper | What it does |
|---|---|---|---|---|
| **Acquire and extract** | | | | |
| [`01_extract_pinned_runs.py`](01_extract_pinned_runs.py) | 58 | *(writes caches, not provenance)* | §4.2 | Pin the MSM runs the ladder needs: download archived surface files, extract the four channels, validate the hour and lead coverage of every day. |
| [`02_extract_pregate_composite.py`](02_extract_pregate_composite.py) | 53 | `phase53_rish_pregate.json` | §4.2 | Build the freshest-admissible per-slot composite (D−2 21 UTC, falling back to 12 UTC where the 21 UTC horizon ends); the JSON records the composite's validation and coverage. |
| [`03_build_area_points.py`](03_build_area_points.py) | 74 | *(writes `data/reference/`, not provenance)* | §5.2 | Parse the FIT register's operating-capacity table into 46 prefecture points with within-area PV weights. |
| [`04_acquire_ics_basis.py`](04_acquire_ics_basis.py) | 71 | `phase71_ics_basis.json` | §4.1 | Fetch the imbalance-price calculation basis. Retained because the lineage audit classifies this feed against the gate; the paper draws no settlement conclusion from it. |
| [`05_extend_panel.py`](05_extend_panel.py) | 85 | *(extends caches; audited in `phase85_exact_ladder.json`)* | §4.2 | Extend every input series past the frozen panel in disjoint-range files, and audit that the frozen rows are untouched before anything is re-estimated. |
| [`06_extract_gfs.py`](06_extract_gfs.py) | 86 | *(writes caches)* | §5.6 | Extract the second provider's pre-gate cycles from the NOAA archive by byte range, with dissemination times read from the store's own timestamps. |
| [`07_patch_gfs_shortwave.py`](07_patch_gfs_shortwave.py) | 86 | *(writes caches)* | §5.6 | Re-fetch the shortwave field alone after the NCEP table's `sdswrf` short name was identified, and join it into the per-area frames. |
| **Accuracy: the vintage ladder** | | | | |
| [`10_primary_ladder_cw.py`](10_primary_ladder_cw.py) | 85 | `phase85_exact_ladder.json` | §5.2 | The primary ladder: 14 cells on the capacity-weighted field over the extended panel, the gate-timing equivalence tests, the measurement-consistent decomposition, and the pre-registered gates (45–50) that block the sweep unless the earlier estimates bit-reproduce. |
| [`11_cycle_curve.py`](11_cycle_curve.py) | 73v2 | `phase73v2_cycle_curve.json` | §2.2 | Improvement by MSM cycle on the common 08:00–16:00 window, extended panel; the pre-registered slope criterion (Gate 41). |
| [`12_single_point_ladder.py`](12_single_point_ladder.py) | 59 | `phase59_exact_ladder.json` | §5.6 | The single-point ladder, now the robustness arm — and the reference the primary sweep must reproduce to 1e-9 before it is allowed to run. |
| [`13_capacity_weighted_ladder_prior.py`](13_capacity_weighted_ladder_prior.py) | 74 | `phase74_cw_ladder.json`<br>`phase74_gate35_proxy.json` | §5.6 | The first capacity-weighted estimate, on the shorter panel. Retained as the second bit-reproduction reference and as the record of where the promotion rule (Gate 35) fired. |
| [`14_enriched_baseline.py`](14_enriched_baseline.py) | 54v2 | `phase54v2_current.json` | §5.2 | What the pre-gate vintage adds over the strongest gate-available baseline a bidder could assemble. |
| [`14b_enriched_baseline_drop_supply.py`](14b_enriched_baseline_drop_supply.py) | 54v2 | `phase54v2_drop.json` | §5.6 | The same contrast with the supply-mix family dropped, the branch the lineage audit pre-registered for the case where first-publication timing could not be established. |
| [`14c_enriched_baseline_lag144.py`](14c_enriched_baseline_lag144.py) | 54v2 | `phase54v2_lag144.json` | §5.6 | The same contrast with the supply mix pushed to lag-144, admissible for every slot under any plausible publication delay. |
| [`15_proxy_validation.py`](15_proxy_validation.py) | 62 | `phase62_proxy_validation.json` | §7 | How much of a real area one representative point captures, against the operators' published output — the validation that selected the capacity-weighted field as primary. |
| [`16_tier_slope_tests.py`](16_tier_slope_tests.py) | 55 | `phase55_tier_slope_tests.json`<br>`phase55_tier_slope_tests_realised.json` | §5.7 | Tier slope-difference tests for the cross-area contrast. |
| [`17_composite_ladder_superseded.py`](17_composite_ladder_superseded.py) | 51 | `phase51_vintage_ladder.json` | §5.2 | The superseded composite-tier ladder. Retained because a figure reads its dissemination accounting; the run-pinned ladders replaced its estimates. |
| [`18_exante_fullyear.py`](18_exante_fullyear.py) | 49 | `phase49_exante_fullyear.json` | S1 | Full-year previous-day-run panel, an earlier data-window refresh. |
| [`19_epex_de_replication.py`](19_epex_de_replication.py) | 45 | `phase45_epex_de_tiered.json` | §7 | EPEX DE/LU two-tier external replication of the vintage contrast, reported as directional corroboration rather than a second audited market. |
| **Floor-event forecasting** | | | | |
| [`30_floor_regime_evaluation.py`](30_floor_regime_evaluation.py) | 66v2 | `phase66v2_floor_regime.json` | §5.4 | Floor-event probabilistic evaluation on the primary measurement: PR-AUC, Brier, reliability, and the operational recall reading. |
| [`31_direct_floor_models.py`](31_direct_floor_models.py) | 81 | `phase81_direct_floor_models.json` | §4.5, §5.4 | The four direct floor models pre-registered beside the isotonic chain, including the price-only control that isolates the weather's contribution. |
| **Timing audit** | | | | |
| [`40_dissemination_audit.py`](40_dissemination_audit.py) | 72 | `phase72_dissemination_audit.json` | §2.2 | Dissemination timing measured from the archive's own Last-Modified headers. |
| [`41_lineage_audit.py`](41_lineage_audit.py) | 78 | `phase78_lineage_audit.json` | §4.3 | Every feature family's lineage: source, first publication, retrieval, revision risk, and measured margin against the gate, plus the processing-buffer re-scoring. |
| [`42_head_sweep.py`](42_head_sweep.py) | 84 | `phase84_head_ef_per_file.parquet` | §4.3 | The archive HEAD sweep that supplies measured dissemination times for the cycles the earlier audit had not covered. |
| **Two targets, and the mechanism test** | | | | |
| [`50_physical_target.py`](50_physical_target.py) | 89 | `phase89_physical_target.json`, `phase89_physical_per_slot.parquet` | §4.5, §5.3 | Score every vintage against the operators' recorded photovoltaic generation with curtailed energy added back, on the days the price panel shares, with the scale fitted only on preceding days. Supersedes phase79, whose scale was fitted on the full sample. |
| [`51_mechanism_test.py`](51_mechanism_test.py) | 90 | `phase90_mechanism.json` | §5.3 | Regress each pre-gate residual on the innovation the excluded run carries, with a variance clustered on area × delivery day, and two placebo regressions. This is the test whose pre-registered retraction branch fired. |
| **Robustness** | | | | |
| [`60_version_robustness.py`](60_version_robustness.py) | 80 | `phase80_jma_version.json` | §5.6 | Split every headline contrast at the 2025-05-20 model upgrade, with and without a transition window. |
| [`61_weighted_metrics.py`](61_weighted_metrics.py) | 82 | `phase82_weighted_metrics.json` | §5.6 | Re-pool the daily improvements under demand and photovoltaic-capacity weights. |
| [`62_inference_sensitivity.py`](62_inference_sensitivity.py) | 83 | `phase83_inference_sensitivity.json` | §5.6 | The equivalence claim against harsher dependence assumptions: circular blocks, month clusters, an automatically selected block length, and a permutation test. |
| [`63_gfs_provider_tier.py`](63_gfs_provider_tier.py) | 86 | `phase86_gfs_tier.json` | §5.6 | Score a second provider's pre-gate cycle on the same folds, so the result is not a property of one forecast centre. |
| [`64_absolute_metrics.py`](64_absolute_metrics.py) | 82b | `phase82b_absolute_metrics.json` | §5.1 | Absolute MAE and RMSE per tier and slice, and skill against a naive 168-hour benchmark, so improvements can be read against a level. |
| [`65_floor_inference.py`](65_floor_inference.py) | 81b | `phase81b_floor_inference.json` | §5.4, S4 | Paired block bootstrap of the macro floor statistic, Holm correction across the five models, micro and prevalence-adjusted readings, and the floor gate contrast under a second model. |
| [`66_weight_sensitivity.py`](66_weight_sensitivity.py) | 91 | `phase91_weight_sensitivity.json` | §7 | Re-estimate the headline and the excluded-run cell under a later capacity-register snapshot and under equal point weights, and run the pre-committed one-tick equivalence test under each. |
| [`52_price_on_common_panel.py`](52_price_on_common_panel.py) | 89b | `phase89b_price_on_common_panel.json` | Fig. 5 | Score the price target on exactly the panel the physical target uses, so the figure's two panels are on one sample rather than only claiming to be. |
| **The literature audit** | | | | |
| [`70_lit_audit_assemble.py`](70_lit_audit_assemble.py) | 88 | `phase88_lit_audit.json` | §2.3, S7 | Assemble the per-paper codings of weather-input timing into the released dataset and the rates the paper cites. |
| [`71_audit_reliability.py`](71_audit_reliability.py) | 88b | `phase88b_audit_reliability.json`, `phase88b_multicoded_runs.json` | §2.3, S7 | Inter-coder agreement on the multi-coded subsample, reported as raw agreement, Cohen's κ, and the Yes-versus-not-Yes flip rate that the reported rates actually depend on. |
| **Outputs** | | | | |
| [`90_emit_tables.py`](90_emit_tables.py) | 64 | *(reads provenance, writes tables)* | — | Emit every provenance-derived LaTeX table and the Editorial Manager abstract. |
| [`91_figure_timeline.py`](91_figure_timeline.py) | 87 | *(reads provenance, writes `figures/`)* | Figs. 1a, 1b | The delivery-day clock and the measured cycle curve, as two figures. |
| [`92_figures_paper.py`](92_figures_paper.py) | 87 | *(reads provenance, writes `figures/`)* | Fig. 2, GA | The analysis roadmap and the graphical abstract. |
| [`93_figures_results.py`](93_figures_results.py) | 87 | *(reads provenance, writes `figures/`)* | Figs. 3–5 | The ladder forest, the floor-skill comparison, and the expectations-channel scatter. |
| [`verify_macros.py`](verify_macros.py) | — | *(reads all of them)* | — | The drift guard: re-derives every headline quantity in the paper from `data/provenance/` and fails on disagreement. Shared verbatim with the research repository. |

## Research-repo filenames

The `metadata.script` field inside each provenance JSON records the filename
that produced it, in the research repository's naming. This mapping resolves
those:

| Provenance records | Here |
|---|---|
| `s_20260729_phase58_exact_run_extract.py` | `01_extract_pinned_runs.py` |
| `s_20260726_phase53_rish_pregate_extract.py` | `02_extract_pregate_composite.py` |
| `s_20260803_phase74_build_area_points.py` | `03_build_area_points.py` |
| `s_20260730_phase71_ics_basis_acquire.py` | `04_acquire_ics_basis.py` |
| `s_20260811_phase85_data_extension.py` | `05_extend_panel.py` |
| `s_20260811_phase86_gfs_extract.py` | `06_extract_gfs.py` |
| `s_20260812_phase86_sw_patch.py` | `07_patch_gfs_shortwave.py` |
| `s_20260811_phase85_cw_primary_ladder.py` | `10_primary_ladder_cw.py` |
| `s_20260811_phase73v2_cycle_curve_extended.py` | `11_cycle_curve.py` |
| `s_20260729_phase59_exact_ladder.py` | `12_single_point_ladder.py` |
| `s_20260803_phase74_cw_ladder.py` | `13_capacity_weighted_ladder_prior.py` |
| `s_20260812_phase54v2_current.py` | `14_enriched_baseline.py` |
| `s_20260812_phase54v2_drop.py` | `14b_enriched_baseline_drop_supply.py` |
| `s_20260812_phase54v2_lag144.py` | `14c_enriched_baseline_lag144.py` |
| `s_20260729_phase62_proxy_validation.py` | `15_proxy_validation.py` |
| `s_20260726_phase55_tier_slope_tests.py` | `16_tier_slope_tests.py` |
| `s_20260726_phase51_vintage_ladder.py` | `17_composite_ladder_superseded.py` |
| `s_20260724_phase49_exante_fullyear.py` | `18_exante_fullyear.py` |
| `s_20260724_phase45_epex_de_tiered.py` | `19_epex_de_replication.py` |
| `s_20260811_phase66v2_floor_regime_cw.py` | `30_floor_regime_evaluation.py` |
| `s_20260811_phase81_direct_floor_models.py` | `31_direct_floor_models.py` |
| `s_20260731_phase72_dissemination_audit.py` | `40_dissemination_audit.py` |
| `s_20260811_phase78_lineage_audit.py` | `41_lineage_audit.py` |
| `s_20260811_phase84_head_sweep_ef.py` | `42_head_sweep.py` |
| `s_20260814_phase92_curtailment_days.py` | `43_curtailment_days.py` |
| `s_20260818_phase93_headline_cell_dispersion.py` | `44_headline_cell_dispersion.py` |
| `s_20260813_phase89_physical_target.py` | `50_physical_target.py` |
| `s_20260813_phase90_mechanism.py` | `51_mechanism_test.py` |
| `s_20260811_phase80_jma_version.py` | `60_version_robustness.py` |
| `s_20260811_phase82_weighted_metrics.py` | `61_weighted_metrics.py` |
| `s_20260811_phase83_inference_sensitivity.py` | `62_inference_sensitivity.py` |
| `s_20260811_phase86_gfs_tier.py` | `63_gfs_provider_tier.py` |
| `s_20260813_phase82b_absolute_metrics.py` | `64_absolute_metrics.py` |
| `s_20260813_phase81b_floor_inference.py` | `65_floor_inference.py` |
| `s_20260813_phase91_weight_sensitivity.py` | `66_weight_sensitivity.py` |
| `s_20260813_phase89b_price_on_common_panel.py` | `52_price_on_common_panel.py` |
| `s_20260812_phase88_lit_audit_assemble.py` | `70_lit_audit_assemble.py` |
| `s_20260813_phase88b_audit_reliability.py` | `71_audit_reliability.py` |
| `s_20260729_phase64_emit_supp_tables.py` | `90_emit_tables.py` |
| `s_20260812_phase87_timeline_split.py` | `91_figure_timeline.py` |
| `s_20260812_phase87_ae_figures.py` | `92_figures_paper.py` |
| `s_20260812_phase87_results_figures.py` | `93_figures_results.py` |
| `s_20260819_phase100_rish_cycle_horizon_probe.py` | `45_rish_cycle_horizon_probe.py` |
| `s_20260819_phase105_file_level_arrival.py` | `46_file_level_arrival.py` |
| `s_20260819_phase101_physical_normalisation.py` | `53_physical_normalisation.py` |
| `s_20260819_phase102_signal_vs_learner.py` | `54_signal_vs_learner.py` |
| `s_20260819_phase103b_floor_channel_hurdle.py` | `32_floor_channel_hurdle.py` |
| `s_20260820_phase106_commensurable_asymmetry.py` | `55_commensurable_asymmetry.py` |
| `s_20260820_phase107_litaudit_reslice.py` | `72_litaudit_reslice.py` |
| `s_20260820_phase108_costloss_dominated.py` | `67_costloss_dominated.py` |
| `s_20260819_phase94_evaluation_regime.py` | `47_evaluation_regime.py` |
| `s_20260819_phase95_area_independence.py` | `48_area_independence.py` |
| `s_20260819_phase96_lear_ladder.py` | `18b_lear_ladder.py` |
| `s_20260819_phase97_cost_loss.py` | `66b_cost_loss.py` |
| `s_20260819_phase98_dissemination_detail.py` | `40b_dissemination_detail.py` |

