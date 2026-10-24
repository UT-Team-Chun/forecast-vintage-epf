# forecast-vintage-epf

**Reproduction companion for**:

> Okauchi, R. and Chun, P.-J. (2026). _Dissemination-aware weather forecast
> vintages for day-ahead electricity price forecasting in Japan._
> Manuscript under review.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

---

## What this repository is

Every number, table and figure in the paper is produced by a script in
[`scripts/`](scripts/) and frozen in a provenance JSON under
[`data/provenance/`](data/provenance/). A drift guard re-derives each of the
manuscript's headline quantities from those JSONs and fails if the paper and
the provenance disagree:

```bash
make verify-paper     # no raw data, no network, about a second
```

That single command is the strongest thing this repository offers a reader who
does not want to re-run a multi-hour sweep. It checks **581 headline
quantities** against their provenance, refuses any numeric macro that no check
accounts for, and scans the built PDFs for wording the manuscript has
retracted.

## What it is not

It is not a general-purpose forecasting library, and it does not redistribute
raw market data. JEPX spot files, OCCTO supply-demand actuals, the
transmission operators' imbalance-price disclosures, JMA MSM-GPV grids and
Open-Meteo archives are all publicly retrievable; [`data/README.md`](data/README.md)
gives the exact endpoints and windows, and file hashes where the loaders
record them. The repository ships the
code, the frozen provenance and the acquisition recipe — not the third-party
bytes.

## Relationship to `floor-moderated-epf`

This codebase descends from
[`floor-moderated-epf`](https://github.com/UT-Team-Chun/floor-moderated-epf),
the companion of an **earlier and different** manuscript
(*Forecast-Skill Scarcity, Price Floors, and the Value of Weather Information
in Solar-Saturated Electricity Markets*, archived at
[Zenodo 10.5281/zenodo.20823024](https://doi.org/10.5281/zenodo.20823024)).

The two papers share a research question and a data pipeline, but not a design
or a result. The earlier one evaluated weather inputs by what the archive
contains; this one evaluates them by **when each forecast run was disseminated
to bidders**, which changes which inputs are admissible at all and therefore
changes every headline number. Because they are different papers, they get
different repositories: the earlier companion stays frozen at its `v1.0.0` tag
for the earlier preprint, and nothing here supersedes it.

## The paper in brief

A weather forecast can only inform a bid if it had *reached bidders* before the
gate closed, so the paper evaluates weather inputs by dissemination time, not
initialisation time. On the Japan Electric Power Exchange, whose bids close at
10:00 JST, archive write times show that the runs initialised two days before
delivery arrived before the gate on every audited day, while the forecast
centre's documented observation cut-off and schedule — consistent with every
archive record across 2,189 delivery days from 2020 to 2026 — place the run
initialised at 09:00 after it. Over 732 daily walk-forward folds in nine price
areas, a forecast disseminated more than ten hours before the gate cuts a
price-only model's mean absolute error by 11.7%, still adds value over an
enriched public-data baseline, and improves the detection of floor-price
events. The run disseminated after the gate adds at most about 3.9% of that
value (the upper end of its 95% interval); across four learners and three ways
of using it, the bound stays within about 6% of the matching pre-gate value,
and the increment is undetectable over a 2021–2024 backcast that spans Japan's
scarcity-price episodes. On the same slots, fresher runs keep sharpening a
photovoltaic generation forecast while barely moving the price forecast.

## What's new in v0.2.0

v0.2.0 is the version that corresponds to the manuscript as revised for
*Energy* (v0.1.x corresponds to the earlier version). It adds the analyses the
revision reports, each with its frozen provenance and its pre-registered rules:

- **The 2021–2024 backcast** — `08_extract_backcast_runs.py` pins the two runs
  for August 2020 to July 2024 and `10b_backcast_ladder.py` re-runs the primary
  design, unchanged, on a window that contains Japan's scarcity episodes
  (`make extract_backcast`, `make backcast`).
- **Learner strength** — `18c_learner_strength.py`: both contrasts under
  XGBoost re-tuned every 91 days, LEAR and a feed-forward network, training on
  post-upgrade data only, and the wind-channel checks (`make learner_strength`).
- **The documented lower bound** — `40c_dissemination_bounds.py` locates the
  observation cut-off and the job schedule in JMA's documents and audits the
  archive write times of both runs over 2020–2026.
- **Other markets** — `49_market_gates.py` sets ten day-ahead gates against the
  ECMWF dissemination schedule and observed GFS arrivals.
- **Checks re-cut from stored predictions and the internal referee checks** —
  `68_stored_prediction_checks.py` (weather regimes, training length, the floor
  threshold, operational counts) and `69_internal_referee_checks.py` (the
  daylight row, the two targets on the same slots, the cost–loss usable value).
- **The enriched baseline on the headline input** — `14d_enriched_baseline_pinned.py`.
- **Outputs** — `90b_gate_bound_table.py` (Table 2) and `95_figures_energy.py`
  (the four main figures, replacing the earlier results figures).

Three scripts left with results the revised manuscript no longer reports or
now draws elsewhere (`44_headline_cell_dispersion.py`,
`55_commensurable_asymmetry.py`, `93_figures_results.py`); the extraction tool
lists them as retired.

## Quickstart

```bash
uv sync --all-extras          # or: make setup
make verify-paper             # provenance guard, no raw data needed
make test                     # unit tests
```

To go further you need raw data:

```bash
cp .env.template .env         # point EPF_RAW_DATA_DIR at your download area
make acquire                  # the one feed fetched programmatically (imbalance-price basis)
make extract                  # pin the MSM runs (long; see REPRODUCIBILITY.md)
make reproduce                # the full headline chain
```

## Repository map

| Path | Contents |
|---|---|
| `alg/` | the computation package the scripts import: loaders, feature builders, area metadata |
| `scripts/` | one script per analysis phase, numbered in pipeline order, plus the drift guard and the emitters — [`scripts/README.md`](scripts/README.md) maps each one to its provenance JSON and to the paper |
| `data/provenance/` | frozen JSON for every reported number, one file per phase |
| `data/reference/` | small static tables (area capacity, prefecture points and their capacity weights) |
| `docs/macros.tex` | the manuscript's macro definitions — the guard's left-hand side |
| `docs/architecture.md` | how the pieces fit together |
| `tests/` | import and fixture smoke tests |

## The headline chain

`make reproduce` runs these in dependency order. Runtimes are for a 12-core
laptop; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full table.

| Target | Script | What it produces |
|---|---|---|
| `ladder` | [`10_primary_ladder_cw.py`](scripts/10_primary_ladder_cw.py) | the primary run-pinned ladder on the capacity-weighted field, with the gate-timing equivalence tests |
| `cycle_curve` | [`11_cycle_curve.py`](scripts/11_cycle_curve.py) | improvement by MSM cycle on a common daytime window |
| `reference_ladders` | [`12_single_point_ladder.py`](scripts/12_single_point_ladder.py), [`13_capacity_weighted_ladder_prior.py`](scripts/13_capacity_weighted_ladder_prior.py) | the two earlier ladders the primary sweep must bit-reproduce before it runs |
| `enriched_baseline` | [`14_enriched_baseline.py`](scripts/14_enriched_baseline.py), [`14d_enriched_baseline_pinned.py`](scripts/14d_enriched_baseline_pinned.py) | what the pre-gate vintage adds over the strongest gate-available baseline, on the composite input and on the headline input |
| `proxy_validation` | [`15_proxy_validation.py`](scripts/15_proxy_validation.py) | how much of a real area one representative point captures |
| `cross_area` | [`16_tier_slope_tests.py`](scripts/16_tier_slope_tests.py) | tier slope-difference tests for the cross-area contrast |
| `lear` | [`18b_lear_ladder.py`](scripts/18b_lear_ladder.py) | the headline contrast re-measured with the canonical LEAR benchmark |
| `learner_strength` | [`18c_learner_strength.py`](scripts/18c_learner_strength.py) | both contrasts under re-tuned XGBoost, LEAR and a neural network; single-version training; the wind channel |
| `epex_de` | [`19_epex_de_replication.py`](scripts/19_epex_de_replication.py) | the vintage contrast in one German bidding zone, as directional corroboration |
| `floor_regime` | [`30_floor_regime_evaluation.py`](scripts/30_floor_regime_evaluation.py) | floor-event probabilistic evaluation |
| `floor_models` | [`31_direct_floor_models.py`](scripts/31_direct_floor_models.py), [`32b_floor_channel_decomposition.py`](scripts/32b_floor_channel_decomposition.py), [`32_floor_channel_hurdle.py`](scripts/32_floor_channel_hurdle.py) | direct floor classifiers, and the hurdle split of the gain into floor probability and level |
| `timing_audit` | [`40_dissemination_audit.py`](scripts/40_dissemination_audit.py), [`41_lineage_audit.py`](scripts/41_lineage_audit.py), [`46_file_level_arrival.py`](scripts/46_file_level_arrival.py), [`40c_dissemination_bounds.py`](scripts/40c_dissemination_bounds.py), [`49_market_gates.py`](scripts/49_market_gates.py) | dissemination timing from archive timestamps, feature lineage, every file's arrival against its own gate, the documented lower bound, and the gates of nine other markets |
| `evaluation_scope` | [`47_evaluation_regime.py`](scripts/47_evaluation_regime.py), [`48_area_independence.py`](scripts/48_area_independence.py) | what the evaluation window contains, and how independent the nine areas are |
| `two_targets` | [`50_physical_target.py`](scripts/50_physical_target.py), [`54_signal_vs_learner.py`](scripts/54_signal_vs_learner.py) | the price/physical asymmetry, and whether the post-gate null is the signal or the learner |
| `robustness` | [`60_version_robustness.py`](scripts/60_version_robustness.py) … [`69_internal_referee_checks.py`](scripts/69_internal_referee_checks.py) | the pre-registered sensitivity families, the checks re-cut from stored predictions, and the internal referee checks, including the cost–loss usable value |
| `backcast` | [`10b_backcast_ladder.py`](scripts/10b_backcast_ladder.py) | the primary design on 2021–2024, a window that holds Japan's scarcity episodes, with gates G112a–d |
| `lit_audit` | [`71_audit_reliability.py`](scripts/71_audit_reliability.py), [`72_litaudit_reslice.py`](scripts/72_litaudit_reslice.py) | inter-coder agreement and the inclusion-rule re-slice |
| `tables`, `figures` | [`90_emit_tables.py`](scripts/90_emit_tables.py) … [`95_figures_energy.py`](scripts/95_figures_energy.py) | every generated table, the EM abstract, and all of the paper's figures |

The settlement analyses that earlier versions of this table advertised —
commitment decisions, the settlement wedge, the two published prices, netting,
nomination rules — belong to a companion manuscript and were removed with
their scripts. [`scripts/README.md`](scripts/README.md) records the split.

## Pre-registration

Several results in this paper are governed by criteria fixed *before* the
numbers existed, and the gates were written down in dated documents in the
research repository before each phase ran. Two of them fired against the
authors' expectation and are reported that way:

- **The saturation branch** governed a claim that improvement plateaus across
  forecast cycles. It fired: the three-point extrapolation was retracted and
  replaced by the measured cycle curve over the full panel.
- **The mechanism test's retraction branch** governed whether the pre-gate
  residuals load on the innovation the excluded run carries. It fired too, and
  the mechanism claim it would have supported is not made.

Others held and strengthened the paper: **Gate 35/36** (a capacity-weighted
re-estimation could have overturned the gate-timing null and did not) and
**Gate 5**, which requires that adding new cells reproduce the existing ladder's
per-slot predictions bit for bit — it does, to a maximum absolute difference of
zero across 307,638 slots.

## Reproducibility of the sweeps

The learning sweeps are **CPU- and build-sensitive**: Gate 5 compares
predictions to 1e-9, which floating-point differences between machines can
break. Reproducing the ladder bit-identically therefore means reproducing the
environment (`uv.lock` pins it); reproducing it *statistically* — same
conclusions, differences far below the reported intervals — is what a different
machine should expect. `REPRODUCIBILITY.md` says which is which for each target.

## Citation

See [`CITATION.cff`](CITATION.cff), or cite the paper directly. Every release
of this repository is archived at Zenodo; the concept DOI
[10.5281/zenodo.22041072](https://doi.org/10.5281/zenodo.22041072) always
resolves to the latest version, and each version has its own DOI.

## License

MIT — see [`LICENSE`](LICENSE). Third-party data retain their own terms; see
[`data/README.md`](data/README.md).
