# Reproducibility

This document says, for each result in the paper, what it takes to reproduce it
and what "reproduced" should mean on a machine that is not ours.

## Three levels, and which one you want

| Level | Command | Needs | Time | What it establishes |
|---|---|---|---|---|
| **1. Provenance** | `make verify-paper` | this repo only | ~1 s | Every headline number in the manuscript matches the frozen JSON it came from, no macro prints an unaccounted number, and no retracted wording survives in the shipped figure PDFs. |
| **2. Analysis** | `make reproduce` | extracted caches; network for `timing_audit` | ~1 day | Every JSON this repository's scripts own is regenerated from the per-slot panels. |
| **3. Acquisition** | `make acquire extract` | network, ~30 GB transient | ~1–2 days | The panels are rebuilt from the public sources. |

Most readers want level 1. It is the level at which a claim of the form *"the
paper says X and the analysis produced X"* is checkable in about a second, with
no data, no network and no GPU. `make verify-paper-strict` is the stricter
submission-time variant: the same checks, but macros still on the guard's
declared-TODO ledger become failures instead of counted warnings — the ledger
is not yet empty, so today that target fails by design and level 1 is the one
that must pass.

## Level 1: the drift guard

`scripts/verify_macros.py` re-derives 407 headline quantities from
`data/provenance/` and compares them to `docs/macros.tex`, the definitions the
manuscript actually prints. It also enforces four things that ordinary
number-checking misses:

- **An unchecked-macro ratchet.** Any macro that puts a quantity on the page and
  is not accounted for by a check, an exemption with a stated source, or a
  declared TODO is a failure. Numbers cannot enter the paper unguarded, and
  spelled-out quantities ("zero") are caught alongside digits.
- **Aggregate bindings.** A macro that prints "the lowest per-area improvement"
  is bound to the *argmin over the container*, not to whichever area happens to
  be lowest today, so a re-run that reorders the areas cannot pass silently.
- **A retired-term audit over built PDFs and submission files.** Wording the
  manuscript has retracted is scanned for in the compiled artefacts, not just
  the sources, with whitespace flattened first — a phrase that wraps across a
  PDF line break used to slip through, and a deliberate-breakage test caught it.
- **Generated-artefact freshness.** Tables and the Editorial Manager abstract
  are emitted from provenance; `--check` fails if what is on disk is not what
  the emitter would write now.

The guard is meant to be attacked. If you want to convince yourself it works,
change a digit in `docs/macros.tex` and run it.

## Level 2: the analysis chain

`make reproduce` runs the analyses in dependency order.
[`scripts/README.md`](scripts/README.md) maps each script to the provenance
JSON it writes and to the section of the paper it backs. Times are for a
12-core Apple-silicon laptop with 64 GB. One target in this level reaches the
network: `timing_audit` is largely HTTP HEAD requests against the archive.

| Target | Script | Time | Notes |
|---|---|---|---|
| `ladder` | `10_primary_ladder_cw.py` | ~4 h | 9 areas × ~730 folds × 14 cells; per-area checkpoints, so it resumes |
| `cycle_curve` | `11_cycle_curve.py` | ~3 h | 6 cells on the common daytime window; parallelises by area |
| `reference_ladders` | `12_single_point_ladder.py`, `13_capacity_weighted_ladder_prior.py` | ~6.5 h | the two bit-reproduction references; **Gate 5 blocks the primary sweep** if the shared cells do not reproduce their per-slot predictions |
| `enriched_baseline` | `14_enriched_baseline.py`, its two variants, and `14d` | ~45 min | the strongest gate-available baseline, off the extracted caches; `14d` re-measures it on the headline input (8 areas × 701 folds × 4 cells, four workers) |
| `proxy_validation` | `15_proxy_validation.py` | minutes | representative point vs the operators' published output (Gate 35's criterion) |
| `cross_area` | `16_tier_slope_tests.py` | ~1 min | slope-difference tests across areas |
| `lear` | `18b_lear_ladder.py` | ~2 h | the same folds under the LEAR benchmark |
| `learner_strength` | `18c_learner_strength.py` | several hours | five variants (the reproduction gate with the wind checks, post-upgrade training, the network, rolling re-tuning, post-upgrade LEAR); the re-tuned variant's 40-trial searches dominate |
| `epex_de` | `19_epex_de_replication.py` | ~20 min | one German bidding zone, two tiers |
| `floor_regime` | `30_floor_regime_evaluation.py` | ~4 h | functional bootstrap dominates |
| `floor_models` | `31`, `32b`, `32` | ~1.5 h | direct classifiers, then the channel partition and the hurdle fit |
| `timing_audit` | `40`, `40b`, `41`, `45`, `46`, `40c`, `49` | ~2 h | ~7,000 HTTP HEAD requests at one per second, out of courtesy to the mirror; `40c` and `49` read local documents and logs in seconds |
| `evaluation_scope` | `42`, `43`, `47`, `48` | ~10 min | window composition and area independence |
| `two_targets` | `50`, `51`, `52`, `53`, `54b`, `54` | ~5 h | the physical target and the signal-versus-learner refits dominate |
| `robustness` | `60` … `69` | ~6 h | `65` and `66` re-estimate; the rest, `68` and `69` included, post-process stored predictions |
| `backcast` | `10b_backcast_ladder.py`, with `40c` before its analysis | ~2.5 h | 9 areas × 1,308 folds × 5 cells, two areas at a time; gate 112-0 first checks the new extraction against the frozen caches where they overlap |
| `lit_audit` | `71`, `72` | ~1 min | read `data/reference/lit_audit_dataset.csv` |
| `tables`, `figures` | `90`, `90b` … `95` | ~1 min | read provenance only; the table emitters write nothing where no manuscript tree is present |

### Determinism, and the honest caveat

Every stochastic step is seeded (`SEED = 2026`): the stationary and circular
block bootstraps, the permutation tests, and the gradient-boosting fits. On the
same machine and the same locked environment, re-running a phase reproduces its
JSON to the digit.

Across machines it will not, exactly. XGBoost's histogram method depends on
build flags and core count, and floating-point summation order differs. This
matters in one specific place: **Gate 5** requires the cells shared with an
earlier phase to reproduce that phase's per-slot predictions to within 1e-9,
and it is designed to abort the whole sweep if they do not. On our hardware the
observed difference is exactly zero across all 307,638 slots. On different
hardware Gate 5 may fail *for reasons that are not a defect in the analysis*.
If that happens:

- The correct reading is "this machine cannot bit-reproduce the reference
  panel", not "the result is wrong".
- Compare the improvements and their intervals instead. Differences of order
  1e-6 JPY/kWh against reported effects of order 1e-1 with intervals of order
  1e-2 are not a disagreement.

We chose a hard gate over a soft one because its purpose is to catch *silent
harness drift* when we add cells, and for that job a tolerance that absorbs real
non-determinism would be useless.

## Level 3: acquisition

`data/README.md` gives the endpoints, windows and hashes. Two notes:

- **The MSM-GPV archive is a research mirror.** The extraction scripts sleep
  between requests deliberately. Please do not remove that; a full pin of one
  run plan is ~880 days of files. The GRIB2 decoder needs the `geo` extra
  (`uv sync --extra geo`; the wheel bundles the ecCodes binary).
- **`make extract` is resumable.** Each plan writes per-area caches and a
  progress file, and a re-run skips days already complete against the plan's
  declared hour coverage.
- **`make extract_backcast` is the second long download** (~13 h for 2022–2024;
  the 2020–2021 sweep can run alongside it into its own cache). It also records
  every archive file's write time, which the arrival audit in `40c` reads.

## What is not in this repository

- Raw third-party data (JEPX, OCCTO, the operators' imbalance disclosures, JMA
  MSM-GPV, Open-Meteo, the METI FIT register). All are public;
  `data/README.md` says how to get each one.
- Per-slot panels. They are large and fully derivable; the frozen JSON that the
  paper's numbers actually come from *is* included.
- The manuscript sources. This repository backs the paper's numbers; the paper
  itself is with the publisher.

## Environment

`uv.lock` pins the exact dependency graph. `uv sync --all-extras` reproduces it.
Python 3.12 is required (`requires-python = ">=3.12,<3.13"`); the GRIB2 decoder
and XGBoost are the version-sensitive pieces. The dependency list is exactly
the import surface of `alg/` and `scripts/` — if it is installed, something
here imports it.
