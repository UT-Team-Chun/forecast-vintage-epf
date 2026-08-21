# Contributing to forecast-vintage-epf

This repository is the artefact accompanying the paper
*"Dissemination-aware weather forecast vintages for day-ahead electricity
price forecasting in Japan"* (manuscript under review). It is **a
research artefact, not an actively maintained OSS project**. The code, scripts, dataset registry, and frozen
provenance JSONs are pinned at the published-paper version
(`v0.1.0` tags the submission snapshot; `v1.0.0` tags the
camera-ready snapshot on acceptance, matching the version in `pyproject.toml`).

## What contributions are welcome

- **Reproducibility issues**: open a GitHub issue if a script in
  [`scripts/`](scripts/) does not reproduce the reported number on a
  fresh clone. Include the script name, the affected paper macro /
  provenance file under [`data/provenance/`](data/provenance/), and
  the discrepancy you see vs. the value listed in
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Paste your `uv pip list`
  output (or `pip freeze`) and OS / Python version.
- **Bug reports**: open an issue if any module under [`alg/`](alg/)
  raises an unexpected exception on the provided synthetic fixtures
  or on the documented public data sources (JEPX day-ahead CSV,
  Open-Meteo ERA5 JSON).
- **Citation / licensing questions**: open an issue. See
  [`CITATION.cff`](CITATION.cff) for the preferred citation.
- **Pointers to relevant work**: open an issue if you have published
  or are working on something the paper should have cited; we will
  consider it for any future revision.

## What is out of scope

- **Adding new features**: the codebase is frozen to match the
  paper. New methodology belongs in downstream work and separate
  releases.
- **Refactoring for style**: the algorithmic content matters, not
  the Python style; large refactors will not be merged.
- **Adding new corpora / markets**: the harness is market-agnostic
  in principle, but bundling a different market belongs in your own
  fork (please cite this paper and mention `forecast-vintage-epf` in
  your README).

## Pull requests

PRs are accepted but reviewed at the author's pace (this is a side
artefact to ongoing research). Keep them small, scoped to a single
issue, and link the issue in the PR description.

## Contact for matters that do not fit a public issue

Data-sharing inquiries, research collaboration, or anything that does
not belong in a public thread: reach the corresponding author via the
[ORCID profile](https://orcid.org/0009-0001-5391-7133) linked in
[`README.md`](README.md). Please do not use email addresses scraped
from commit metadata.
