# Data acquisition

This repository does **not** redistribute raw inputs. Every source below is
public; acquire them under their own terms and point the harness at your
extract via `.env` (copy `.env.template`).

What *is* in git is the part that lets you check the paper without any of this:
`provenance/` holds the frozen JSON behind every reported number, and
`reference/` holds two small static tables the code needs.

## Directory layout

```
data/
├── README.md                            # this file
├── provenance/                          # IN git — frozen JSON, one per analysis phase
│   ├── phase59_exact_ladder.json        #   the run-pinned vintage ladder
│   ├── phase73_cycle_curve.json         #   improvement by MSM cycle
│   ├── phase74_cw_ladder.json           #   capacity-weighted re-estimation
│   ├── phase52_fip_commitment*.json     #   the commitment decision model, per tier
│   ├── phase72_dissemination_audit.json #   dissemination lag, measured
│   └── …                                #   54 files: 52 at top level, 2 in per-phase subdirectories
├── reference/                           # IN git
│   ├── area_capacity.csv                #   per-area PV / wind / peak capacity
│   └── area_points_capacity.csv         #   46 prefecture points and their within-area PV weights
├── raw/                                 # NOT in git — you acquire these
│   ├── jepx/                            #   day-ahead spot CSVs
│   ├── jma_gpv_exactrun/                #   pinned MSM runs, per plan
│   ├── jma_gpv_exactrun_mp/             #   the same at 46 points (capacity-weighted tier)
│   ├── jma_gpv_pregate/                 #   freshest-admissible per-slot composite
│   ├── open_meteo*/                     #   ERA5 and historical-forecast archives
│   ├── occto/                           #   supply-demand actuals incl. solar curtailment
│   ├── ics/                             #   imbalance-price calculation basis
│   └── meti_fit/                        #   FIT register capacity tables
└── runs/                                # NOT in git — where scripts write
```

## 1. JEPX day-ahead spot prices

**Source**: <https://www.jepx.jp/electricpower/market-data/spot/> — JEPX terms
of use, **not redistributed**.

One CSV per fiscal year, nine areas, 30-minute resolution. Fetch FY2023–FY2026
into `data/raw/jepx/`. `alg/core/epf/data/jepx_loader.py` verifies each file's
digest on first read and records URL, hash and fetch time in the dataset
registry it maintains.

## 2. JMA MSM-GPV forecast grids — the source this paper is about

**Source**: Kyoto University RISH archive of the JMA GPV stream,
<http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/> — a
**research mirror**; treat it accordingly.

This is the input the paper's central argument turns on, so a word about what
is being fetched. Every forecast tier here is *run-pinned*: each value comes
from one identified initialisation, not from a series stitched across
initialisations at a fixed nominal lead. `make extract` pins the plans the
ladder needs, one archived surface file at a time:

| Plan | Run | Delivery hours | Role |
|---|---|---|---|
| A | D−2 12 UTC | 00–23 | headline pre-gate tier; the only pre-gate run whose archived horizon spans a whole delivery day |
| C | D−1 00 UTC | 00–23 | the first post-gate run — the one the deadline excludes |
| E | D−3 12 UTC | 01–23 | deep rung of the cycle curve |
| F | D−2 18 UTC | 00–18 | intermediate rung |
| G | D−2 21 UTC | 00–21 | freshest admissible cycle |

Two archive facts constrain what is possible, and both are properties of the
observing system rather than choices: the 78-hour horizon is archived only for
the 00 and 12 UTC cycles (the rest stop at forecast hour 39), and dissemination
completes about 2 h 30 min after initialisation by the published specification —
2 h 16 min by the archive's own timestamps, measured in `phase72`.

**The scripts sleep between requests on purpose.** A full plan is ~880 days of
files. Please do not remove the delay.

## 3. Open-Meteo — reanalysis and historical forecast

**Source**: <https://open-meteo.com/> (ERA5 archive and Historical Forecast
API), CC-BY 4.0 for non-commercial use; check current terms.

Two distinct roles, kept distinct in the paper: the ERA5 archive supplies the
**day-lagged reanalysis proxy** (a research stand-in, not a product anyone could
have bought at the gate), and the Historical Forecast archive supplies the
**blended reconstruction** tier. The loaders cache responses under
`data/raw/open_meteo*/`.

## 4. OCCTO supply-demand actuals

**Source**: the transmission operators' monthly published files, coordinated
under OCCTO.

Supplies realised area demand, solar and wind output, and **solar curtailment**
— the last of which grounds the paper's floor-event analysis in a physical
operator action rather than in the price series alone.

## 5. Imbalance-price calculation basis (ICS)

**Source**: the nine operators' joint disclosure.

`make acquire_ics` fetches this. It carries more than the settlement price:
per slot and per **wide-area operation block** it publishes the signed imbalance
volume being balanced and the marginal kWh price of the balancing resources
called on. The first is what makes the netting result possible — one party's
gross deviation is not system imbalance, and this series is what lets the paper
net against the block's own direction instead of asserting a system saving. The
second is what makes the two-prices comparison a measurement rather than a
model.

## 6. METI FIT register — capacity weights

**Source**: <https://www.fit-portal.go.jp/> published capacity tables
(sheet 表A②-1, operating capacity by prefecture).

`make area_points` parses these into `reference/area_points_capacity.csv`:
46 prefecture points with their PV operating-capacity share **within** each
area. Because the weights are within-area shares, any level disagreement
between the FIT register and `area_capacity.csv` cancels. The generated CSV
records the SHA-256 of each source workbook in its header.

## Provenance files

Each JSON under `provenance/` was written by exactly one script, and the guard
reads the artefact the analysis actually wrote — it does not re-derive numbers
from a description of the analysis. That is what makes `make verify-paper`
meaningful. Two populations, and the distinction is worth being explicit about:

- **24 files are written by the scripts in this repository** — the analysis
  chain the paper's headline numbers come from. Each carries a `metadata`
  block recording the producing script, the git SHA it ran at and the
  generation timestamp (the one acquisition record, `phase71_ics_basis.json`,
  records the script and runtime; its source hashes live in the body), and,
  where the analysis is stochastic, the seed and the bootstrap draw count.
  [`scripts/README.md`](../scripts/README.md) maps each one to its script.
- **30 files are inherited from earlier phases of the same research
  programme** and predate that metadata convention; the scripts that produced
  them are not shipped here. They back numbers the paper cites for context or
  robustness — earlier-window head-to-heads, cross-model identification
  checks, panel-era diagnostics — never a headline result. They are frozen
  artefacts checked by the same guard; regenerating one means going back to
  the research repository they came from.
