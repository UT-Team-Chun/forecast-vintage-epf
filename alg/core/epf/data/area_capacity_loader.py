"""Single source of truth for per-area renewable capacity & peak demand.

Every analysis that needs PV/peak (or wind/peak) capacity ratios MUST obtain
the figures from this loader rather than hardcoding a per-script dict. This
module exists because hardcoded per-script capacity dicts drifted into two
inconsistent vintages (a documented OCCTO-2024 set used by Phase 15 / 23.4 and
the paper abstract / §4.1, and an undocumented set used by the Phase 25.6-25.12
scripts) that disagree on the headline PV/peak cross-area correlation
(-0.800 vs -0.45). See ``docs/research/2026-06-08_phase25.23_*.md``.

The backing data lives in ``data/reference/area_capacity.csv`` with provenance
columns. To change the canonical capacity vintage, edit that ONE file and
re-run the analyses; no script edits are required.

KNOWN VINTAGE DEFECT (measured in Phase 62, 2026-07-29). The current table is an
FY2023-24 snapshot marked ``approx``, and realised OCCTO potential output exceeds
its nameplate figures in SEVEN area-channels -- which is physically impossible and
therefore a table error, not a data surprise. The worst case is Tokyo wind
(100 MW nameplate against a 477 MW realised maximum, 4.8x); Tokyo PV, Tohoku wind
and Hokkaido wind are also exceeded. Consequences differ by use:

* As a per-area SCALE on engineered proxies (Phase 54/62), a constant factor is
  absorbed by the learner, so accuracy results are unaffected.
* As the cross-area PV/peak SATURATION INDEX (Phase 55), the error passes straight
  through and reorders areas: Shikoku moves from 0.921 to 0.532 under an
  observable index. Any cross-area statistic must therefore be reported under
  both indices. ``s_20260726_phase55_tier_slope_tests.py --saturation-index
  realised`` recomputes it from ``max realised PV potential / peak demand``, which
  carries no capacity vintage; the published conclusion is unchanged under both.

Refreshing the table to a current OCCTO supply plan would remove the defect, but
it would also change every published cross-area figure, so it is deliberately
NOT done inside a revision cycle. See ``docs/research/2026-07-29_phase62_*.md``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from alg.config import EpfConfig
from alg.models.epf.area import AreaCode

REFERENCE_FILENAME = "area_capacity.csv"


class AreaCapacity(BaseModel):
    """One area's installed capacity and peak demand (GW).

    ``area`` is the typed :class:`AreaCode` enum (not a bare string) so the
    single source of truth is validated upstream; an unknown area string in
    the CSV raises at load time rather than silently propagating.
    """

    model_config = ConfigDict(frozen=True)

    area: AreaCode
    pv_gw: float
    wind_gw: float
    peak_gw: float
    source: str
    as_of: str
    note: str

    @property
    def pv_peak(self) -> float:
        """PV capacity / peak demand ratio (the headline engineering antecedent)."""
        return self.pv_gw / self.peak_gw

    @property
    def wind_peak(self) -> float:
        """Wind capacity / peak demand ratio."""
        return self.wind_gw / self.peak_gw


def _reference_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    cfg = EpfConfig()
    return cfg.resolved("reference_data_dir") / REFERENCE_FILENAME


def reference_sha256(path: Path | None = None) -> str:
    """SHA-256 of the backing CSV — record this in run provenance."""
    p = _reference_path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_area_capacity(path: Path | None = None) -> dict[AreaCode, AreaCapacity]:
    """Return ``{AreaCode: AreaCapacity}`` from the single-source reference CSV.

    This is the typed canonical API. Comment lines (``#``) and blank lines are
    ignored. An unknown area string raises via :class:`AreaCode` validation.
    """
    p = _reference_path(path)
    rows: dict[AreaCode, AreaCapacity] = {}
    with p.open(encoding="utf-8") as fh:
        data_lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    for r in reader:
        cap = AreaCapacity(
            area=AreaCode(r["area"].strip()),
            pv_gw=float(r["pv_gw"]),
            wind_gw=float(r["wind_gw"]),
            peak_gw=float(r["peak_gw"]),
            source=r["source"].strip(),
            as_of=r["as_of"].strip(),
            note=r["note"].strip(),
        )
        rows[cap.area] = cap
    return rows


# ----- Thin string-keyed adapters for drop-in compatibility with existing
# ----- usage (dict indexing by area string, polars ``.map`` on a str column,
# ----- and ``for area in DICT`` iteration). Keys are ``AreaCode.value`` —
# ----- derived from the enum, never hardcoded. The typed source of truth is
# ----- ``load_area_capacity`` above. -----


def pv_gw_by_area(path: Path | None = None) -> dict[str, float]:
    """``{area_str: pv_gw}`` — drop-in replacement for the old ``AREA_PV_GW`` dict."""
    return {a.value: c.pv_gw for a, c in load_area_capacity(path).items()}


def peak_gw_by_area(path: Path | None = None) -> dict[str, float]:
    """``{area_str: peak_gw}`` — drop-in replacement for the old ``AREA_PEAK_GW`` dict."""
    return {a.value: c.peak_gw for a, c in load_area_capacity(path).items()}


def wind_gw_by_area(path: Path | None = None) -> dict[str, float]:
    """``{area_str: wind_gw}`` — drop-in replacement for the old ``AREA_WIND_GW`` dict."""
    return {a.value: c.wind_gw for a, c in load_area_capacity(path).items()}


def pv_peak_by_area(path: Path | None = None) -> dict[str, float]:
    """``{area_str: pv_gw / peak_gw}``."""
    return {a.value: c.pv_peak for a, c in load_area_capacity(path).items()}


def area_capacity_dict(path: Path | None = None) -> dict[str, dict[str, float]]:
    """``{area_str: {"pv_gw":..., "wind_gw":..., "peak_gw":...}}``.

    Drop-in replacement for the old nested ``AREA_CAPACITY`` dict.
    """
    return {
        a.value: {"pv_gw": c.pv_gw, "wind_gw": c.wind_gw, "peak_gw": c.peak_gw}
        for a, c in load_area_capacity(path).items()
    }
