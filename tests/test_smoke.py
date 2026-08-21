"""Smoke tests for forecast-vintage-epf.

These run on a fresh clone with no raw data and no network. They check the
three things that can silently rot in a reproduction package:

1. the import surface (every module the scripts need is actually here),
2. the provenance set (every frozen JSON parses, and names a script that
   exists in this repository),
3. **the drift guard itself** — the guard is what makes the paper's numbers
   checkable, so CI runs it rather than trusting that someone did.

They deliberately do not exercise any analysis path; those need data the user
acquires (see ``data/README.md``).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "data" / "provenance"
SCRIPTS = ROOT / "scripts"


# ---------------------------------------------------------------- imports
def test_alg_top_level_imports() -> None:
    import alg  # noqa: F401
    from alg.config import EpfConfig

    cfg = EpfConfig()
    assert cfg.default_area == "kyushu"
    assert cfg.tz == "Asia/Tokyo"


def test_every_alg_module_imports() -> None:
    """Import every module under alg/, so a missing transitive dep fails here."""
    import importlib

    failures: list[str] = []
    for path in sorted((ROOT / "alg").rglob("*.py")):
        mod = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - the message is the assertion
            failures.append(f"{mod}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_area_points_table_loads_and_weights_sum_to_one() -> None:
    from alg.core.epf.data.area_points import area_points

    pts = area_points()
    assert len(pts) == 9, f"expected nine areas, got {sorted(pts)}"
    assert sum(len(v) for v in pts.values()) == 46
    for area, entries in pts.items():
        total = sum(e.weight for e in entries)
        assert abs(total - 1.0) < 1e-4, f"{area}: weights sum to {total}"


# ------------------------------------------------------------- provenance
def test_provenance_files_present_and_parse() -> None:
    files = sorted(PROV.rglob("*.json"))
    assert len(files) >= 50, f"only {len(files)} provenance files found"
    for f in files:
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{f.relative_to(ROOT)} is not valid JSON: {exc}")


def _rename_map() -> dict[str, str]:
    """Research-repo filename -> filename here, parsed from scripts/README.md.

    The scripts are renamed for readability (the monorepo's ``s_<date>_<phase>``
    convention means nothing outside it), but the provenance JSONs still record
    the name that actually ran, because they are frozen artefacts. The manifest
    table is the single source of truth for that mapping, so this test reads it
    rather than duplicating it.
    """
    text = (SCRIPTS / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*`(s_\d{8}_[^`]+\.py)`\s*\|\s*`([^`]+\.py)`\s*\|", text, re.M)
    assert rows, "no research-repo mapping table found in scripts/README.md"
    return dict(rows)


def test_every_provenance_names_a_script_that_exists() -> None:
    """A frozen result whose producing script is missing is not reproducible."""
    mapping = _rename_map()
    missing: list[str] = []
    for f in sorted(PROV.rglob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8")).get("metadata") or {}
        script = meta.get("script")
        if not script:
            continue  # older phases predate the metadata block
        recorded = Path(script).name
        here = mapping.get(recorded, recorded)
        if not (SCRIPTS / here).exists():
            missing.append(f"{f.name} records {recorded} -> expected scripts/{here}")
    assert not missing, "provenance names scripts not in this repo:\n" + "\n".join(missing)


def test_rename_map_is_complete_and_accurate() -> None:
    """Every renamed script is in the manifest, and every manifest entry exists."""
    mapping = _rename_map()
    here = {p.name for p in SCRIPTS.glob("*.py")}
    unlisted = here - set(mapping.values()) - {"verify_macros.py"}
    dangling = set(mapping.values()) - here
    assert not unlisted, f"scripts missing from the manifest table: {sorted(unlisted)}"
    assert not dangling, f"manifest names scripts that do not exist: {sorted(dangling)}"


# ----------------------------------------------------------------- scripts
def test_every_script_parses() -> None:
    """Cheap guard against a truncated or half-copied script."""
    bad: list[str] = []
    for f in sorted(SCRIPTS.glob("*.py")):
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append(f"{f.name}: {exc}")
    assert not bad, "scripts failed to parse:\n" + "\n".join(bad)


# ------------------------------------------------------------- the guard
def test_drift_guard_passes() -> None:
    """Re-derive every headline number from provenance and compare to the macros.

    This is the test that matters. If it fails, the manuscript and the frozen
    provenance disagree, and the failure message names the macro.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPTS / "verify_macros.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"drift guard failed:\n{tail}"
    assert "match their provenance" in proc.stdout
