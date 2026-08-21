# Makefile for forecast-vintage-epf
#
# Targets are grouped by what they need:
#   - `verify-paper` and `test` need NOTHING but this repository.
#   - `acquire` and `extract` reach the public data sources.
#   - the analysis targets need the extracted caches.
#
# Everything runs from the repository root and assumes `uv` >= 0.5.

PYTHON ?= uv run python
UV_RUN ?= uv run
SCRIPTS = scripts

.PHONY: help setup lock fmt lint test smoke
.PHONY: verify-paper verify-paper-strict tables figures
.PHONY: acquire acquire_ics extract extract_pregate extract_multipoint area_points
.PHONY: reproduce ladder cycle_curve reference_ladders enriched_baseline
.PHONY: proxy_validation cross_area superseded lear epex_de floor_regime
.PHONY: floor_models timing_audit evaluation_scope two_targets robustness
.PHONY: lit_audit

help:
	@echo "No raw data required:"
	@echo "  verify-paper        - re-derive every headline number from data/provenance (~1 s)"
	@echo "  verify-paper-strict - submission-time gate: declared-TODO macros become failures (non-empty today, so it fails by design)"
	@echo "  test / smoke        - unit and import checks"
	@echo ""
	@echo "Data acquisition (public sources; see data/README.md):"
	@echo "  acquire             - imbalance-price basis series"
	@echo "  extract             - pin the MSM runs the ladder needs (long)"
	@echo "  extract_multipoint  - the same runs at 46 prefecture points (for reference_ladders)"
	@echo "  area_points         - build the capacity-weighted prefecture point table"
	@echo ""
	@echo "Analysis (needs the caches above):"
	@echo "  reproduce           - the whole headline chain in dependency order"
	@echo "  ladder cycle_curve reference_ladders enriched_baseline"
	@echo "  proxy_validation cross_area lear epex_de superseded"
	@echo "  floor_regime floor_models timing_audit evaluation_scope"
	@echo "  two_targets robustness lit_audit"
	@echo ""
	@echo "Outputs:"
	@echo "  tables figures      - regenerate provenance-derived tables and the figures"

setup:
	uv sync --all-extras

lock:
	uv lock

# alg/ and scripts/ are verbatim extractions (see pyproject [tool.ruff.format]):
# lint them, never reformat them.
fmt:
	$(UV_RUN) ruff format tests
	$(UV_RUN) ruff check --fix tests

lint:
	$(UV_RUN) ruff format --check tests
	$(UV_RUN) ruff check alg scripts tests

test:
	$(UV_RUN) pytest

smoke:
	$(PYTHON) -c "import alg, alg.config, alg.core.epf.data.jepx_loader, alg.core.epf.data.area_points; print('imports ok')"
	$(UV_RUN) pytest tests/test_smoke.py -v

# ---------------------------------------------------------------- guards
# The drift guard re-derives docs/macros.tex from data/provenance and fails on
# any disagreement, on any numeric macro no check accounts for, and on retired
# wording surviving in a built PDF. It needs no raw data and no network.
verify-paper:
	$(PYTHON) $(SCRIPTS)/verify_macros.py
	$(PYTHON) $(SCRIPTS)/90_emit_tables.py --check

verify-paper-strict:
	$(PYTHON) $(SCRIPTS)/verify_macros.py --strict
	$(PYTHON) $(SCRIPTS)/90_emit_tables.py --check

# ------------------------------------------------------------ acquisition
acquire: acquire_ics

acquire_ics:       ## imbalance-price calculation basis, per slot and block (~1 min)
	$(PYTHON) $(SCRIPTS)/04_acquire_ics_basis.py --start 202408 --end 202607

area_points:       ## prefecture points and their PV-capacity weights, from the FIT register
	$(PYTHON) $(SCRIPTS)/03_build_area_points.py

# Pinning a forecast run means downloading its archived GRIB2 surface files and
# extracting the four channels at each point. This is the long part: the
# archive is a research mirror and the sweeps are rate-limited out of courtesy.
extract: extract_pregate
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan A
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan C
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan E
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan F

extract_pregate:   ## the freshest-admissible per-slot composite
	$(PYTHON) $(SCRIPTS)/02_extract_pregate_composite.py

extract_multipoint: area_points  ## the same runs at 46 prefecture points (capacity-weighted tier)
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan G --multipoint
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan A --multipoint
	$(PYTHON) $(SCRIPTS)/01_extract_pinned_runs.py --plan C --multipoint

# ---------------------------------------------------------------- analysis
# Script numbers follow scripts/README.md. The settlement analysis that used to
# live here (commitment decisions, the settlement wedge, netting, nomination
# rules) belongs to the companion paper and was removed with its scripts; the
# targets that called them were left behind until 2026-08-21 and are gone now.

ladder:            ## the primary run-pinned ladder on the capacity-weighted field
	$(PYTHON) $(SCRIPTS)/10_primary_ladder_cw.py --step build
	$(PYTHON) $(SCRIPTS)/10_primary_ladder_cw.py --step analyse

cycle_curve:       ## improvement by MSM cycle on the common 08:00-16:00 window
	$(PYTHON) $(SCRIPTS)/11_cycle_curve.py --step build
	$(PYTHON) $(SCRIPTS)/11_cycle_curve.py --step analyse

reference_ladders: ## the two ladders the primary sweep must bit-reproduce first
	$(PYTHON) $(SCRIPTS)/12_single_point_ladder.py --step build
	$(PYTHON) $(SCRIPTS)/12_single_point_ladder.py --step analyse
	$(PYTHON) $(SCRIPTS)/13_capacity_weighted_ladder_prior.py --step gate35
	$(PYTHON) $(SCRIPTS)/13_capacity_weighted_ladder_prior.py --step build
	$(PYTHON) $(SCRIPTS)/13_capacity_weighted_ladder_prior.py --step analyse

enriched_baseline: ## what the vintage adds over the strongest gate-available baseline
	$(PYTHON) $(SCRIPTS)/14_enriched_baseline.py
	$(PYTHON) $(SCRIPTS)/14b_enriched_baseline_drop_supply.py
	$(PYTHON) $(SCRIPTS)/14c_enriched_baseline_lag144.py

proxy_validation:  ## single-point weather proxies against the operators' actuals
	$(PYTHON) $(SCRIPTS)/15_proxy_validation.py

cross_area:        ## tier slope-difference tests for the cross-area contrast
	$(PYTHON) $(SCRIPTS)/16_tier_slope_tests.py

superseded:        ## the composite-tier ladder and full-year arm, kept for the record
	$(PYTHON) $(SCRIPTS)/17_composite_ladder_superseded.py
	$(PYTHON) $(SCRIPTS)/18_exante_fullyear.py

lear:              ## the headline increment re-measured with the canonical LEAR benchmark
	$(PYTHON) $(SCRIPTS)/18b_lear_ladder.py --step build
	$(PYTHON) $(SCRIPTS)/18b_lear_ladder.py --step analyse

epex_de:           ## the vintage contrast outside Japan, in one German bidding zone
	$(PYTHON) $(SCRIPTS)/19_epex_de_replication.py

floor_regime:      ## floor-event probabilistic evaluation (long)
	$(PYTHON) $(SCRIPTS)/30_floor_regime_evaluation.py \
	    --n-boot 10000 --n-boot-functional 2000

floor_models:      ## direct classifier vs recalibrated point forecast, and the channel split
	$(PYTHON) $(SCRIPTS)/31_direct_floor_models.py --step build
	$(PYTHON) $(SCRIPTS)/31_direct_floor_models.py --step analyse
	$(PYTHON) $(SCRIPTS)/32b_floor_channel_decomposition.py --step partition
	$(PYTHON) $(SCRIPTS)/32b_floor_channel_decomposition.py --step build
	$(PYTHON) $(SCRIPTS)/32b_floor_channel_decomposition.py --step analyse
	$(PYTHON) $(SCRIPTS)/32_floor_channel_hurdle.py --step build
	$(PYTHON) $(SCRIPTS)/32_floor_channel_hurdle.py --step analyse

timing_audit:      ## dissemination lag, lineage, and arrival against each day's own gate
	$(PYTHON) $(SCRIPTS)/40_dissemination_audit.py
	$(PYTHON) $(SCRIPTS)/40b_dissemination_detail.py
	$(PYTHON) $(SCRIPTS)/41_lineage_audit.py
	$(PYTHON) $(SCRIPTS)/45_rish_cycle_horizon_probe.py
	$(PYTHON) $(SCRIPTS)/46_file_level_arrival.py

evaluation_scope:  ## what the evaluation window contains, and how independent the areas are
	$(PYTHON) $(SCRIPTS)/42_head_sweep.py
	$(PYTHON) $(SCRIPTS)/43_curtailment_days.py
	$(PYTHON) $(SCRIPTS)/44_headline_cell_dispersion.py
	$(PYTHON) $(SCRIPTS)/47_evaluation_regime.py
	$(PYTHON) $(SCRIPTS)/48_area_independence.py

two_targets:       ## the physical target, the mechanism test, and the commensurable contrast
	$(PYTHON) $(SCRIPTS)/50_physical_target.py
	$(PYTHON) $(SCRIPTS)/51_mechanism_test.py
	$(PYTHON) $(SCRIPTS)/52_price_on_common_panel.py
	$(PYTHON) $(SCRIPTS)/53_physical_normalisation.py
	$(PYTHON) $(SCRIPTS)/54b_innovation_construction.py --step build-price
	$(PYTHON) $(SCRIPTS)/54b_innovation_construction.py --step build-phys
	$(PYTHON) $(SCRIPTS)/54b_innovation_construction.py --step analyse
	$(PYTHON) $(SCRIPTS)/54_signal_vs_learner.py --step build --pass q
	$(PYTHON) $(SCRIPTS)/54_signal_vs_learner.py --step build --pass r
	$(PYTHON) $(SCRIPTS)/54_signal_vs_learner.py --step build --pass b
	$(PYTHON) $(SCRIPTS)/54_signal_vs_learner.py --step phys
	$(PYTHON) $(SCRIPTS)/54_signal_vs_learner.py --step analyse
	$(PYTHON) $(SCRIPTS)/55_commensurable_asymmetry.py

robustness:        ## the pre-registered sensitivity families
	$(PYTHON) $(SCRIPTS)/60_version_robustness.py
	$(PYTHON) $(SCRIPTS)/61_weighted_metrics.py
	$(PYTHON) $(SCRIPTS)/62_inference_sensitivity.py
	$(PYTHON) $(SCRIPTS)/63_gfs_provider_tier.py --step build
	$(PYTHON) $(SCRIPTS)/63_gfs_provider_tier.py --step analyse
	$(PYTHON) $(SCRIPTS)/64_absolute_metrics.py
	$(PYTHON) $(SCRIPTS)/65_floor_inference.py --step build-postgate
	$(PYTHON) $(SCRIPTS)/65_floor_inference.py --step analyse
	$(PYTHON) $(SCRIPTS)/66_weight_sensitivity.py --step build
	$(PYTHON) $(SCRIPTS)/66_weight_sensitivity.py --step analyse
	$(PYTHON) $(SCRIPTS)/66b_cost_loss.py
	$(PYTHON) $(SCRIPTS)/67_costloss_dominated.py

# 70_lit_audit_assemble.py is NOT called here: it assembles the audit dataset
# from the screening rows as they were hand-coded (`--rows <coded>.json`), which
# is an input to this repository, not an artefact of it. The assembled dataset
# ships as data/reference/lit_audit_dataset.csv -- the article's machine-readable
# Supplementary Dataset -- and the two checks below read it.
lit_audit:         ## the day-ahead literature audit's reliability and inclusion checks
	$(PYTHON) $(SCRIPTS)/71_audit_reliability.py
	$(PYTHON) $(SCRIPTS)/72_litaudit_reslice.py

# ----------------------------------------------------------------- outputs
tables:            ## every provenance-derived table and the Editorial Manager abstract
	$(PYTHON) $(SCRIPTS)/90_emit_tables.py

figures:           ## every figure the article carries, from the shipped provenance
	$(PYTHON) $(SCRIPTS)/91_figure_timeline.py
	$(PYTHON) $(SCRIPTS)/92_figures_paper.py
	$(PYTHON) $(SCRIPTS)/93_figures_results.py
	$(PYTHON) $(SCRIPTS)/94_figures_theory.py

# The full chain, in dependency order. `ladder` must come first: it writes the
# per-slot panel that reference_ladders bit-reproduces and that the two-target
# and robustness families read.
reproduce: ladder cycle_curve reference_ladders enriched_baseline \
           proxy_validation cross_area lear epex_de floor_regime floor_models \
           timing_audit evaluation_scope two_targets robustness lit_audit \
           tables figures verify-paper
