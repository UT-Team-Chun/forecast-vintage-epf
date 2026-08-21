# Architecture

How the pieces fit, and why they are arranged this way.

## The shape of the problem

The paper's claim is about *timing*: a weather forecast counts as usable only if
it had reached bidders before the gate closed. That makes the unit of analysis a
**pinned run** — one identified initialisation, whose dissemination instant is
recorded — rather than "the forecast at lead N", which is a series assembled
across many initialisations and was never held by anyone as a whole.

Every design decision below follows from that.

```
   public sources                 alg/                        scripts/                data/provenance/
┌──────────────────┐      ┌────────────────────┐      ┌────────────────────┐      ┌──────────────────┐
│ JEPX spot        │─────▶│ jepx_loader        │      │ 01-04  extract     │      │                  │
│ JMA MSM-GPV      │─────▶│ jma_gpv_loader     │─────▶│ 10     ladder      │─────▶│ *_ladder.json    │
│ Open-Meteo       │─────▶│ open_meteo_*       │      │ 11     cycle curve │─────▶│ *_cycle.json     │
│ OCCTO actuals    │─────▶│ area_capacity      │      │ 12-13  references  │─────▶│ *_ladder.json    │
│ ICS basis        │─────▶│ (script-local)     │      │ 30-32  floor       │─────▶│ *_floor*.json    │
│ METI FIT tables  │─────▶│ area_points        │      │ 4x-7x  downstream  │─────▶│ …                │
└──────────────────┘      └────────────────────┘      └─────────┬──────────┘      └────────┬─────────┘
                                                                 │                          │
                                                                 ▼                          ▼
                                                       figures / tables  ◀────────  verify_macros.py
                                                                                    (docs/macros.tex)
```

## `alg/` — what the scripts import

A deliberately small surface: loaders that fetch-and-cache, feature builders,
and static area metadata. It contains no analysis. Everything that makes a
claim lives in a script, so that a claim always has exactly one producing file
and one provenance artefact.

| Module | Responsibility |
|---|---|
| `config.py` | typed settings (paths, timezone, default area) from env or `.env` |
| `core/epf/data/jepx_loader.py` | day-ahead area prices, with digest verification |
| `core/epf/data/jma_gpv_loader.py` | archived MSM-GPV surface files: fetch, decode, extract points, and return the archive's `Last-Modified` so dissemination can be *measured* |
| `core/epf/data/open_meteo_loader.py` | ERA5 reanalysis (the day-lagged proxy tier) |
| `core/epf/data/open_meteo_historical_forecast_loader.py` | the blended reconstruction tier |
| `core/epf/data/area_capacity_loader.py` | per-area PV / wind / peak capacity |
| `core/epf/data/area_points.py` | the 46 prefecture points and their within-area capacity weights |
| `core/epf/data/registry.py` | dataset registry: URL, hash, fetch time per cached input |
| `core/epf/features/calendar.py` | calendar features |
| `core/epf/features/exogenous.py` | attaching hourly weather to half-hourly slots |
| `models/epf/area.py`, `models/epf/calendar.py` | area and calendar value objects |

The GRIB2 decoder streams message-by-message through the low-level eccodes API
with multi-field support on. That is not an optimisation detail: the eager
whole-grid alternative peaks around 7.5 GiB and was killed by the OS on
multi-day sweeps, so the streaming path is what makes an 880-day pin feasible
at all.

## `scripts/` — one file per claim

Each script owns one analysis phase and writes its provenance JSON (a few
write more than one — the decision model once per operational tier — plus
per-slot parquets where a downstream step needs them). The JSON's `metadata`
block records the script's own filename and the git SHA it ran at, and, where
the analysis is stochastic, the seed and the bootstrap draw count.

Three conventions are load-bearing:

- **Pre-registered gates.** Where a result could have gone either way, the
  decision rule was written down before the numbers existed and is echoed into
  the JSON, branch and all. Gates that fired against the authors' expectation
  are reported as such rather than quietly re-specified.
- **Reduced-draw runs cannot overwrite provenance.** A script invoked with fewer
  bootstrap draws than the canonical count writes `*_draft.json` instead, unless
  explicitly forced. Debug runs used to clobber the real artefact; now they
  cannot.
- **Blocking checks abort the run.** `12_capacity_weighted_ladder.py` will not
  report anything unless the cells it shares with the ladder
  (`10_vintage_ladder.py`) reproduce its per-slot predictions to 1e-9. A
  harness with unaccounted non-determinism invalidates the new numbers too, so
  the correct response is to stop.

## `verify_macros.py` — the guard

The manuscript prints numbers through LaTeX macros. The guard reads
`docs/macros.tex`, re-derives each macro's value from `data/provenance/`, and
fails on disagreement. Around that core it enforces:

- an **unchecked-macro ratchet** — a macro that puts a quantity on the page and
  is accounted for by nothing is itself a failure;
- **aggregate bindings** — "the lowest per-area value" binds to the argmin over
  a container, not to whichever key is lowest today;
- a **retired-term audit** over built PDFs and submission files, with whitespace
  flattened so a phrase that wraps across a line still matches;
- **generated-artefact freshness** — tables emitted from provenance must equal
  what the emitter would write now.

The file is shared verbatim with the authors' research repository and resolves
its own layout (`docs/macros.tex` + `data/provenance/` here; different paths
there). A blind copy in either direction therefore cannot silently disable it.

## Why provenance is JSON in git

Because the alternative is a number in a paper whose origin is a memory. The
JSONs are small, diffable, and enough to re-derive every claim without touching
raw data — which is what makes `make verify-paper` a one-second check rather
than a two-day one.
