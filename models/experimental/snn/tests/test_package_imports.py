# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every intra-package import must name something that exists, and every module must parse.

Two failure modes this guards, both of which land on the *first* step of a CI run rather than
during review:

* a name imported from a sibling module that was renamed or removed. ``autoflake`` reports the
  unused case, and ``ruff`` reports an undefined name only within a module's own scope, so a
  ``from ...synapses import active_tiles`` that no longer exists is invisible to both -- the import
  raises at collection, once the simulator job is already running.
* a test module that does not parse, or that imports something undefined at module scope, which
  fails collection the same way.

The device test modules are the reason this matters most. They cannot be run on a host without
tt-metal, so nothing else in this package exercises their import path at all -- an import error in
one of them is discovered only by the CI job it was supposed to inform.

Checked with ``ast`` rather than by importing, so this runs in the device-free half.

Scope note: undefined *module-scope* names are ``ruff``'s F821, which resolves a module's own
scope properly. An earlier version of this file tried to cover that here and reported
function parameters as undefined; a check that manufactures false positives is a check that
stops being read, so it was removed rather than fixed.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SOURCES = sorted(PACKAGE.rglob("*.py"))


def _module_key(path: Path) -> str:
    """``.../snn/synapses.py`` -> ``models.experimental.snn.snn.synapses``."""
    relative = path.relative_to(PACKAGE.parents[2])
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _public_names(tree: ast.Module) -> set[str]:
    """Module-level names, plus instance attributes assigned anywhere via ``self.x = ...``.

    Instance attributes matter because the device tests read a layer's private buffers
    (``layer._v_mem``); a check that only saw module-level defs would flag every one of them.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            targets = [node.target] if hasattr(node, "target") else list(node.targets)
            for target in targets:
                stack = [target]
                while stack:
                    item = stack.pop()
                    if isinstance(item, (ast.Tuple, ast.List)):
                        stack.extend(item.elts)
                    elif isinstance(item, ast.Starred):
                        stack.append(item.value)
                    elif isinstance(item, ast.Name):
                        names.add(item.id)
                    elif (
                        isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "self"
                    ):
                        names.add(item.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


_EXPORTED: dict[str, set[str]] = {}


def _exports(module: str) -> set[str]:
    if module not in _EXPORTED:
        path = PACKAGE.parents[2] / (module.replace(".", "/") + ".py")
        _EXPORTED[module] = _public_names(ast.parse(path.read_text())) if path.exists() else set()
    return _EXPORTED[module]


def test_the_package_was_found():
    assert len(SOURCES) >= 20, f"only found {len(SOURCES)} sources under {PACKAGE}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_module_parses(path):
    ast.parse(path.read_text())


def _is_submodule(module: str, name: str) -> bool:
    """True when ``name`` is a ``.py`` inside the package directory ``module`` names.

    ``from package import submodule`` is ordinary Python and needs no entry in the package's
    ``__init__`` -- Python imports the submodule and binds it. A resolver that only consults
    ``__init__`` therefore rejects a valid import and pushes callers into working around it in
    individual files, which is how a wrong check spreads instead of being fixed.
    """
    return (PACKAGE.parents[2] / module.replace(".", "/") / f"{name}.py").exists()


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_intra_package_imports_resolve(path):
    """``from models.experimental.snn... import X`` -- X must exist in the module it names.

    ``X`` is either a name that module exports or one of its submodules; both are importable, and
    neither has to appear in the module's ``__init__``.
    """
    unresolved = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
            continue
        if not node.module.startswith("models.experimental.snn"):
            continue
        available = _exports(node.module)
        for alias in node.names:
            if alias.name == "*" or alias.name in available:
                continue
            if _is_submodule(node.module, alias.name):
                continue
            unresolved.append(f"{node.module} has no {alias.name!r} (line {node.lineno})")
    assert not unresolved, "\n      ".join(["unresolvable intra-package imports:", *unresolved])


def test_the_resolver_distinguishes_a_submodule_from_a_missing_name():
    """A fix that accepts submodules must not have stopped rejecting names that do not exist.

    Without this, "resolve it as a submodule" could degenerate into "accept anything", which would
    turn the import check into a no-op.
    """
    assert _is_submodule("models.experimental.snn.snn", "layout"), "layout is a real submodule"
    assert _is_submodule("models.experimental.snn.reference", "cpu_baseline"), "cpu_baseline is a real submodule"
    for absent in ("no_such_module", "layout_typo", "config_typo"):
        assert not _is_submodule(
            "models.experimental.snn.snn", absent
        ), f"{absent!r} is not a submodule and must not resolve as one"
    assert "no_such_name" not in _exports(
        "models.experimental.snn.snn"
    ), "a name that is neither exported nor a submodule must stay unresolved"
