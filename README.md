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
does not want to re-run a multi-hour sweep. It checks **407 headline
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

## The paper in three sentences

A weather forecast can only inform a bid if it had *reached bidders* before the
gate closed, and initialisation time is a weaker test than dissemination time:
Japan's meso-scale model reaches users about 2 h 16 min after initialisation
(the median, measured here from 881 days of archive timestamps), so the previous-day
00 UTC run misses the 10:00 JST gate that a naive reading would admit.
Rebuilding the evaluation as a dissemination-correct ladder of run-pinned
tiers, a single pre-gate run is worth several times a day-lagged reanalysis
proxy, while the run the gate excludes is statistically equivalent in pooled
accuracy at a margin of one exchange price tick — though it still shifts most
committed quantities, so gate timing is cheap in accuracy and not in decisions.
Carried into a compliance-constrained nomination, the same forecast reduces the
fleet's netted day-ahead deviation and *lowers* its owner's profit, because
under a single imbalance price a bidder that commits accurately on floor slots
is charged for its precision.

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
| `ladder` | [`10_vintage_ladder.py`](scripts/10_vintage_ladder.py) | the run-pinned vintage ladder, gate-timing bound and equivalence tests |
| `cycle_curve` | [`11_cycle_curve.py`](scripts/11_cycle_curve.py) | improvement by MSM cycle on a common daytime window |
| `enriched_baseline` | [`13_enriched_baseline.py`](scripts/13_enriched_baseline.py) | what the pre-gate vintage adds over the strongest gate-available baseline |
| `proxy_validation` | [`14_proxy_validation.py`](scripts/14_proxy_validation.py) | how much of a real area one representative point captures |
| `capacity_weighted` | [`12_capacity_weighted_ladder.py`](scripts/12_capacity_weighted_ladder.py) | the ladder re-estimated on a 46-point capacity-weighted field |
| `decision` | [`20_commitment_decision.py`](scripts/20_commitment_decision.py) | the compliance-constrained commitment, three operational tiers |
| `wedge` | [`32_wedge_continuous.py`](scripts/32_wedge_continuous.py) | the settlement wedge as a continuous treatment |
| `system` | [`33_system_metrics_band.py`](scripts/33_system_metrics_band.py) | physical metrics and the nomination-band sensitivity |
| `floor_regime` | [`30_floor_regime_evaluation.py`](scripts/30_floor_regime_evaluation.py) | floor-event probabilistic evaluation |
| `settlement` | [`31_settlement_pricing.py`](scripts/31_settlement_pricing.py) | the two published prices for the same half-hour |
| `netting` | [`34_netting_direction.py`](scripts/34_netting_direction.py), [`35_netting_exact.py`](scripts/35_netting_exact.py) | direction decomposition and its exact marginal recomputation |
| `decision_equivalence` | [`21_decision_equivalence.py`](scripts/21_decision_equivalence.py) | whether the excluded run changes decisions, not just accuracy |
| `nomination_rules` | [`22_nomination_rules.py`](scripts/22_nomination_rules.py) | the profit sign under a family of compliant nomination rules |
| `dissemination_audit` | [`40_dissemination_audit.py`](scripts/40_dissemination_audit.py) | dissemination timing measured from archive timestamps |
| `tables`, `figures` | [`90_emit_tables.py`](scripts/90_emit_tables.py), [`91_figure_timeline.py`](scripts/91_figure_timeline.py), [`92_figures_paper.py`](scripts/92_figures_paper.py) | every generated table, the EM abstract, the paper's figures |

## Pre-registration

Several results in this paper are governed by criteria fixed *before* the
numbers existed, and the gates were written down in dated documents in the
research repository before each phase ran. Two of them fired against the
authors' expectation and are reported that way:

- **Gate 39** compared the published direction-decomposed netting figure with
  an exact marginal recomputation. It failed: the published figure overstated
  the result by roughly a quarter, and the exact value replaced it.
- **Gate 41** governed a saturation claim. It fired the retraction branch, and
  the three-point extrapolation was replaced by the measured cycle curve.

Two others held and strengthened the paper: **Gate 35/36** (a capacity-weighted
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

See [`CITATION.cff`](CITATION.cff), or cite the paper directly. A versioned
archive of this repository is deposited at Zenodo on acceptance and its DOI
added at proof stage.

## License

MIT — see [`LICENSE`](LICENSE). Third-party data retain their own terms; see
[`data/README.md`](data/README.md).
