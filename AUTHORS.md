# Authors

## Paper and analysis

- **Ryota Okauchi** — Graduate School of Engineering, The University of Tokyo
  ([ORCID 0009-0001-5391-7133](https://orcid.org/0009-0001-5391-7133))
  Conceptualization, methodology, software, formal analysis, data curation,
  investigation, visualization, writing — original draft.

- **Pang-jo Chun** — Graduate School of Engineering, The University of Tokyo
  Conceptualization, supervision, writing — review & editing, project
  administration. Corresponding author for the manuscript.

## Software provenance

The computation package under `alg/` is extracted from the authors' research
monorepo. Only the modules the published analyses import are included here; the
extraction is mechanical (the transitive import closure of the scripts in
`scripts/`), so nothing in this repository depends on code that is not in it.

The drift guard `scripts/verify_macros.py` is shared verbatim with the research
repository and is layout-aware: it reads macros from `docs/macros.tex` and
provenance from `data/provenance/` here, and from their research-repo locations
there. A blind copy in either direction therefore cannot silently disable it.

## Generative AI disclosure

During the preparation of this work the authors used Anthropic's Claude to
assist with drafting and editing the manuscript and with developing and
reviewing the analysis code. After using this tool, the authors reviewed and
edited all text and code, verified all numerical results against the committed
provenance, and take full responsibility for the content of the publication.
