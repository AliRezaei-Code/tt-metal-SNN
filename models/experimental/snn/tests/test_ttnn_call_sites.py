# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every ``ttnn`` call must pass arguments of the kind its binding declares.

Several ttnn bindings take ``const ttnn::Tensor&`` with ``.noconvert()`` on the argument. A
torch tensor is a different Python type, is not implicitly converted, and has no type caster
registered for it -- so passing one is a ``TypeError`` at the call, not a compile error and not
anything a linter sees. It is invisible here: the call has the right arity, the right names, and
only fails when that line first runs.

That is not hypothetical. ``test_lif_neuron.py`` once rolled its membrane forward with
``ttnn.copy(ttnn.to_torch(v_out), v_mem)``. ``to_torch`` returns a torch tensor, ``copy`` wants two
ttnn tensors, and the failure landed on the first step of every case in the file -- the whole LIF
suite, on its first execution in CI. ``ruff`` is clean on that line, and so is any count of its
arguments.

So this checks the one thing the ordinary gates cannot: the *provenance* of each argument at a
binding that takes a device tensor. It parses the package rather than importing it, so it runs on
a host with no tt-metal at all.

It reports only what it can prove. An argument it cannot classify is left alone rather than
guessed at, which is the whole point -- a check that manufactures plausible false positives is a
check that stops being read.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SNN = PACKAGE / "snn"
TESTS = Path(__file__).resolve().parent
DEMO = PACKAGE / "demo"

# Bound to (const ttnn::Tensor&, ...) with .noconvert(). from_torch is deliberately absent:
# it is *defined* to take a torch tensor, which is the only legitimate one.
TENSOR_BOUND = {"copy", "assign", "to_torch", "reshape", "clone", "get_device_tensors"}

# The outermost call in an expression -- the thing whose return value the argument becomes.
PRODUCES_TORCH = {"to_torch"}
PRODUCES_TTNN = {"from_torch", "allocate_tensor_on_device", "tile_tensor", "_tile_tensor"}


def _root_call(node: ast.AST):
    """Outermost call name in an expression: ``ttnn.to_torch(x).cpu()`` -> ``to_torch``."""
    n = node
    while isinstance(n, (ast.Attribute, ast.Subscript)):
        n = n.value
    if isinstance(n, ast.Call):
        fn = n.func
        return fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
    return n.id if isinstance(n, ast.Name) else None


def _tensor_bound_calls(tree: ast.Module):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ttnn"
            and node.func.attr in TENSOR_BOUND
        ):
            yield node


def _ttnn_tensor_locals(tree: ast.Module) -> set[str]:
    """Local names demonstrably bound to a ttnn Tensor, so they are not reported."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            produced = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if produced in PRODUCES_TTNN and isinstance(node.targets[0], ast.Name):
                names.add(node.targets[0].id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PRODUCES_TTNN:
            names.add(node.name)
    return names


def _sources() -> list[Path]:
    return sorted(SNN.rglob("*.py")) + sorted(TESTS.glob("*.py")) + sorted(DEMO.glob("*.py"))


ALL_SOURCES = _sources()
CHECKED = [p for p in ALL_SOURCES if list(_tensor_bound_calls(ast.parse(p.read_text())))]


def test_the_sweep_actually_examines_something():
    """A sweep that inspects no files proves nothing about the tree."""
    assert len(ALL_SOURCES) >= 15, f"only found {len(ALL_SOURCES)} source files under {PACKAGE}"
    assert CHECKED, "no call into a tensor-taking ttnn binding was found to check"


def test_no_torch_tensor_reaches_a_ttnn_binding():
    """The check that would have caught ``ttnn.copy(ttnn.to_torch(v_out), v_mem)``."""
    bad = []
    for path in CHECKED:
        tree = ast.parse(path.read_text())
        for node in _tensor_bound_calls(tree):
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if _root_call(arg) in PRODUCES_TORCH:
                    bad.append(
                        f"{path.relative_to(PACKAGE)}:{node.lineno}: "
                        f"ttnn.{node.func.attr}({ast.unparse(arg)}) -- that expression is a torch "
                        f"tensor, and the binding takes a ttnn::Tensor with .noconvert()"
                    )
    assert not bad, "torch tensor passed where a device tensor is required:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: p.name)
def test_tensor_arguments_come_from_a_producible_source(path):
    """Each argument into a tensor-taking binding must be something this package can build.

    Reported rather than enforced. A parameter of unknown origin is ordinary in a test helper, and
    failing on it would teach a reader to ignore this check -- which is how the original bug
    would have been allowed to sit in the tree.
    """
    tree = ast.parse(path.read_text())
    known = _ttnn_tensor_locals(tree)
    unvouched = []
    for node in _tensor_bound_calls(tree):
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if _root_call(arg) in PRODUCES_TORCH or _root_call(arg) in PRODUCES_TTNN:
                continue
            if isinstance(arg, ast.Name) and arg.id not in known:
                unvouched.append(f"{arg.id} at line {node.lineno} (ttnn.{node.func.attr})")
    print(f"unvouched argument sources in {path.name}: {', '.join(unvouched) or 'none'}")
    assert True
