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


# ------------------------------------------------ the package must be runnable
#
# Everything below was added on 2026-08-21, after an independent audit found
# that `make verify-paper` exited 2, `make reproduce` died at its third target,
# and four shipped scripts raised on a constant that named a file this package
# does not contain. Every one of those was invisible to the checks above: the
# guard re-derives NUMBERS, and `test_every_script_parses` only parses. Nothing
# resolved a path, ran a target, or read a link. These do.


def _makefile_invocations() -> list[tuple[int, str, str]]:
    """(line number, script name, rest of the recipe line) for each script call."""
    out = []
    for i, line in enumerate((ROOT / "Makefile").read_text().splitlines(), 1):
        m = re.search(r"\$\(SCRIPTS\)/(\S+\.py)(.*)", line)
        if m:
            out.append((i, m.group(1), m.group(2)))
    return out


def _required_flags(path: Path) -> list[str]:
    """Flags the script's argparse declares required=True."""
    flags: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        kw = {k.arg: k.value for k in node.keywords}
        req = kw.get("required")
        if isinstance(req, ast.Constant) and req.value is True:
            flags += [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and str(a.value).startswith("--")
            ]
    return flags


def test_makefile_supplies_every_required_flag() -> None:
    """No recipe may call a script without an argument the script demands.

    Ten of them did. `make reproduce` -- advertised as the whole headline chain
    -- aborted on "error: the following arguments are required: --step" at its
    third target, so five of the analysis targets were dead on arrival while
    every other check in this file passed.
    """
    bad = []
    for lineno, name, rest in _makefile_invocations():
        script = SCRIPTS / name
        if not script.exists():
            bad.append(f"Makefile:{lineno} calls {name}, which is not shipped")
            continue
        missing = [f for f in _required_flags(script) if f not in rest]
        if missing:
            bad.append(f"Makefile:{lineno} {name} needs {missing}")
    assert not bad, "unrunnable Makefile recipes:\n" + "\n".join(bad)


def test_makefile_advertises_only_targets_that_exist() -> None:
    """`.PHONY` and `make help` are the package's self-description."""
    text = (ROOT / "Makefile").read_text()
    defined = {m.group(1) for m in re.finditer(r"^([a-z][\w-]*):", text, re.M)}
    phony: set[str] = set()
    for m in re.finditer(r"^\.PHONY:(.*)$", text, re.M):
        phony |= set(m.group(1).split())
    assert not phony - defined, f".PHONY names undefined targets: {sorted(phony - defined)}"


def test_sibling_script_references_resolve() -> None:
    """Every constant naming a sibling script must name one that is here.

    The packaging tool rewrites sibling references, including the ones that are
    executable constants rather than prose. When the rewrite missed a dotted
    module name the result was a plausible-looking sibling that imports nothing
    -- and both steps of two shipped scripts died on it.
    """
    here = {p.name for p in SCRIPTS.glob("*.py")}
    stems = {p.stem for p in SCRIPTS.glob("*.py")}
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        text = f.read_text(encoding="utf-8")
        tree = ast.parse(text)
        # Path constants: HERE / "x.py", Path(__file__).parent / "x.py"
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and node.right.value.endswith(".py")
                and node.right.value not in here
            ):
                bad.append(f"{f.name}:{node.lineno} -> {node.right.value}")
        # importlib targets, whether inline or via a module-level constant
        consts = {
            t.id: n.value.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "import_module" or not node.args:
                continue
            arg = node.args[0]
            target = (
                arg.value
                if isinstance(arg, ast.Constant)
                else consts.get(arg.id)
                if isinstance(arg, ast.Name)
                else None
            )
            if target and target.split(".")[0] not in stems | {"alg"}:
                bad.append(f"{f.name}:{node.lineno} import_module({target!r})")
    assert not bad, "sibling references that do not resolve here:\n" + "\n".join(bad)


def test_no_script_documents_an_undefined_extra() -> None:
    """A copy-pasteable command must not select a dependency group we lack.

    39 of the shipped scripts told the reader to run `uv run --extra research`,
    which fails before the interpreter starts: that extra exists only in the
    monorepo this package was extracted from.
    """
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    defined = set(pyproject["project"].get("optional-dependencies", {}))
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        for m in re.finditer(r"--extra\s+([\w-]+)", f.read_text(encoding="utf-8")):
            if m.group(1) not in defined:
                bad.append(f"{f.name}: --extra {m.group(1)}")
    assert not bad, "docstrings select undefined extras:\n" + "\n".join(bad)


def test_every_relative_markdown_link_resolves() -> None:
    """The landing page had twelve 404s, all of them scripts we deliberately removed."""
    bad = []
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in {".venv", ".git", "node_modules"} for part in md.parts):
            continue
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for _, url in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", line):
                if url.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                if not (md.parent / url.split("#")[0]).exists():
                    bad.append(f"{md.relative_to(ROOT)}:{i} -> {url}")
    assert not bad, "broken relative links:\n" + "\n".join(bad)


def test_rename_table_matches_the_packaging_tool() -> None:
    """``scripts/README.md`` and ``RENAMES`` must agree in BOTH directions.

    ``test_rename_map_is_complete_and_accurate`` checks the manifest against
    the files on disk, which cannot see the monorepo-name column -- and that
    column is what a reader uses to match a frozen provenance stamp to a
    shipped script. The tool's docstring claimed this test existed; it did not.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_extract", ROOT / "tools" / "extract_from_monorepo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tool = dict(mod.RENAMES)
    doc = {new: old for old, new in _rename_map().items()}
    assert tool == doc, "RENAMES and scripts/README.md disagree:\n" + "\n".join(
        f"  {k}: tool={tool.get(k)!r} readme={doc.get(k)!r}"
        for k in sorted(set(tool) | set(doc))
        if tool.get(k) != doc.get(k)
    )


def test_every_tracked_figure_has_a_producer() -> None:
    """`figures/` claims to be regenerable, so nothing may sit there unproduced.

    Four of the seven tracked figures had no producing script in this package
    -- one of them, the supplement's VoI schematic, had none anywhere here --
    while the guard's retired-term audit scanned and passed them, certifying
    files nothing could rebuild.
    """
    tracked = {p.name for p in (ROOT / "figures").iterdir() if p.suffix in {".pdf", ".png"}}
    written: set[str] = set()
    for f in sorted(SCRIPTS.glob("9*.py")):
        text = f.read_text(encoding="utf-8")
        written |= set(re.findall(r'["\'](\w+\.(?:pdf|png))["\']', text))
    orphans = tracked - written
    assert not orphans, f"tracked figures no script writes: {sorted(orphans)}"


def test_data_readme_counts_match_the_tree() -> None:
    """The counts in data/README.md are the reader's map of the evidence base.

    They were stale by 31 files and mis-split by ~19, having frozen at the
    moment the guard began discovering nested provenance on its own. Deriving
    them here is one glob and one json.loads, so they cannot drift again.
    """
    top = sorted(PROV.glob("*.json"))
    nested = sorted(p for p in PROV.rglob("*.json") if p.parent != PROV)
    # "Written by the scripts in this repository" means: the file names its
    # script in a stamp of its own, OR scripts/README.md lists it beside a script
    # that ships here. The stamp alone misfiled records whose producer IS shipped
    # but which carry no metadata block -- the seven v0.2.0 analyses, the
    # composite's validation, the multi-coded export -- as "inherited from
    # earlier phases, scripts not shipped", which data/README.md then asserted.
    listed: set[str] = set()
    for line in (SCRIPTS / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*\[`([^`]+\.py)`\]\([^)]+\)\s*\|[^|]*\|([^|]*)\|", line)
        if m and (SCRIPTS / m.group(1)).exists():
            listed |= set(re.findall(r"`([^`]+\.json)`", m.group(2)))
    written = 0
    for f in top + nested:
        j = json.loads(f.read_text(encoding="utf-8"))
        meta = j.get("metadata") if isinstance(j, dict) else None
        if (
            (isinstance(meta, dict) and "script" in meta)
            or (isinstance(j, dict) and "script" in j)
            or f.name in listed
        ):
            written += 1
    text = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    expected = [
        f"{len(top) + len(nested)} files: {len(top)} at top level, "
        f"{len(nested)} in per-phase subdirectories",
        f"**{written} files are written by the scripts in this repository**",
        f"**{len(top) + len(nested) - written} files are inherited from earlier phases",
    ]
    missing = [e for e in expected if e not in text]
    assert not missing, "data/README.md counts are stale; the tree says:\n" + "\n".join(missing)
