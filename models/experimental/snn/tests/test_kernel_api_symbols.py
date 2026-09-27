# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device APIs the SNN kernels call must still exist in the TT-Metalium headers.

There is no local build of tt-metal on the machines this was authored on, so the first execution
of these kernels happens in CI. That makes one failure mode unusually expensive here: a kernel
that calls a renamed or never-existing API compiles nowhere and fails the whole run, with the
error pointing at the kernel rather than at the name that was wrong. Everything else in this
package is caught by the compiler; this is the one class of mistake nothing else catches, so it
gets its own check.

Three things this file deliberately does not do, each learned the hard way:

* **It does not parse C++.** An earlier attempt extracted called identifiers by regex and produced
  fifteen phantom failures: local variable names captured before a ``.method(`` call, a class
  template, and a ``FORCE_INLINE uintptr_t`` return type the pattern did not list. A checker that
  reports noise is a checker that stops being run, so the list below is fixed and hand-maintained.
* **It does not pin a header per symbol.** Several of these are declared more than once -- a
  forward declaration, a re-export, or one copy per architecture for ``EltwiseBinaryReuseDestType``
  -- so "the first match" is arbitrary and "this exact path" fails for the wrong reason. What
  matters is that the name still exists somewhere the JIT can see.
* **It does not claim reachability per kernel.** It checks includes resolve; proving each symbol is
  visible from each kernel's own includes needs a transitive walk, which is a compiler's job and
  not this file's.
"""

import ast
from pathlib import Path

import re

import pytest

# Roots the TT-Metalium JIT compiles device kernels against.
KERNELS = Path("models/experimental/snn/snn/kernels")
INCLUDE_ROOTS = (
    Path("tt_metal/hw/inc"),
    Path("tt_metal/tt-llk"),
    Path("tt_metal/hostdevcommon/api"),
)

# This whole file reads the C++ headers, so it can only say anything when they were checked
# out. A CI leg that sparse-checkouts a subset of the tree would otherwise fail here, on a
# missing directory, and bury whatever the kernels actually did. Skipping keeps a header-less
# checkout honest: the test is silent, and the rest of the suite still reports.
HEADERS_PRESENT = all(root.is_dir() for root in INCLUDE_ROOTS)
pytestmark = pytest.mark.skipif(
    not HEADERS_PRESENT,
    reason="tt-metal C++ headers are not in this checkout; nothing to introspect",
)

# Every external function, type and enum the SNN kernels call. Grouped by the kernel that owns
# it. Local identifiers are not here: cb_index, n_tiles, the CircularBuffer locals, and so on.
SYMBOLS = [
    # compute_lif_neuron.cpp -- LIF update: SFPU scalar, compare, dest-reuse FPU fold
    "compute_kernel_hw_startup",
    "add_reuse_dest_init",
    "add_reuse_dest_tiles",
    "sub_reuse_dest_init",
    "sub_reuse_dest_tiles",
    "binop_with_scalar_tile_init",
    "mul_unary_tile",
    "unary_gt_tile",
    "unary_gt_tile_init",
    "copy_dest_values",
    "copy_dest_values_init",
    "EltwiseBinaryReuseDestType",
    # compute_spike_matvec.cpp -- tensor-engine matmul, K-accumulated over active tiles
    "SrcOrder",
    "matmul_init",
    "matmul_tiles",
    # all kernels -- data movement and the compute-side register/CB flow control
    "Noc",
    "CircularBuffer",
    "TensorAccessor",
    "TensorAccessorArgs",
    "LocalTensorAccessor",
    "get_tile_size",
    "get_arg_val",
    "get_compile_time_arg_val",
    "copy_tile",
    "pack_tile",
    "tile_regs_acquire",
    "tile_regs_commit",
    "tile_regs_wait",
    "tile_regs_release",
    "cb_wait_front",
    "cb_pop_front",
    "cb_reserve_back",
    "cb_push_back",
    "cb_index",
    "get_read_ptr",
    "get_write_ptr",
    # reader_spike_multicast.cpp / reader_spike_unicast.cpp -- Phase 4 NoC handshake
    "get_semaphore",
    "get_noc_addr",
    "get_noc_multicast_addr",
    "noc_semaphore_inc",
    "noc_semaphore_set",
    "noc_semaphore_wait",
    "noc_semaphore_set_multicast",
    "noc_async_write",
    "noc_async_read",
    "noc_async_write_multicast",
    "noc_async_writes_flushed",
    "noc_async_read_barrier",
    "noc_async_write_barrier",
]


def _all_header_text() -> str:
    chunks = []
    for root in INCLUDE_ROOTS:
        assert root.is_dir(), f"{root} not found; run from the repository root"
        for header in root.rglob("*.h"):
            chunks.append(header.read_text(errors="ignore"))
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def headers():
    return _all_header_text()


def test_the_tree_is_present():
    """Guard the premise: without the headers this whole file is vacuous."""
    for root in INCLUDE_ROOTS:
        assert root.is_dir(), f"{root} not found; run from the repository root"
    assert KERNELS.is_dir(), f"{KERNELS} not found; run from the repository root"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_symbol_still_exists(symbol, headers):
    assert symbol in headers, f"{symbol} is not declared anywhere under the JIT include roots"


@pytest.mark.parametrize("kernel", sorted(p.name for p in KERNELS.glob("*.cpp")), ids=lambda n: n)
def test_kernel_includes_resolve(kernel):
    """Every ``#include "..."`` in a kernel must name a file that exists.

    A kernel including a header that does not exist fails with an error that says nothing
    useful, and one that omits a header fails later with an undeclared-identifier error pointing
    at the wrong line.
    """
    source = (KERNELS / kernel).read_text()
    includes = [line.split('"')[1] for line in source.splitlines() if line.startswith('#include "')]
    assert includes, f"{kernel} has no project includes"
    for name in includes:
        assert any(
            (root / name).exists() for root in INCLUDE_ROOTS
        ), f"{kernel} includes {name}, which does not resolve under any include root"


def test_no_kernel_calls_get_read_ptr():
    """Mirror the repo's own check-kernel-cb-ptrs hook, so the reason is recorded here too.

    get_read_ptr returns the read pointer, which does not advance, so reading it after a reserve
    can hand back a slot that is not the one just claimed.
    """
    for path in sorted(KERNELS.glob("*.cpp")):
        body = "\n".join(line for line in path.read_text().splitlines() if not line.strip().startswith("//"))
        assert "get_read_ptr" not in body, f"{path.name} calls get_read_ptr"


# ---------------------------------------------------------------------------
# Compile-time argument layout
# ---------------------------------------------------------------------------
# Each kernel reads its compile-time arguments by index and the host supplies them as a
# positional list, so a mismatch is invisible to every other check in this file: the symbols
# exist, the call arity is right, and the kernel compiles -- it just reads the wrong slot. Two
# bugs of exactly this shape shipped during development and were caught only by cross-reading a
# kernel against its builder by hand:
#
#   * ``TensorAccessorArgs<0>`` while the host had put the CB indices at 0 and 1, so the
#     accessor read the buffer indices as its own config word;
#   * the accessor offset passed as ``len(CB_ARGS)`` rather than ``len(CB_ARGS) + 1``, which
#     made the accessor read the offset argument itself and shift every field after it.
#
# Parsed with ast, not executed, so this runs without tt-metal.


def _kernel_cta_indices(path: Path) -> dict[str, int]:
    """``{name: index}`` for every ``get_compile_time_arg_val(N)`` in a kernel."""
    return {
        name: int(idx) for name, idx in re.findall(r"(\w+)\s*=\s*get_compile_time_arg_val\((\d+)\)", path.read_text())
    }


def _kernel_name(call: ast.Call):
    src = call
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
        inner = call.func.value
        if isinstance(inner.func, ast.Attribute):
            src = inner.func
    if isinstance(src, ast.Attribute) and src.attr == "kernel_source":
        for kw in call.keywords:
            if kw.arg == "kernel_source" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "KernelDescriptor":
        for kw in call.keywords:
            if kw.arg == "kernel_source" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


def _leading_arg_count(expr: ast.AST) -> int | None:
    """Leading elements of a ``compile_time_args`` expression, or None if not decidable.

    Handles a bare list literal and a concatenation that *starts* with one, which is the shape
    every builder here uses: ``[CB_A, CB_B] + TensorAccessorArgs(...).get_...()``. Anything else
    returns None and is left to the builder's own tests rather than guessed at.
    """
    node = expr
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        node = node.left
    if isinstance(node, ast.List):
        return len(node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list":
        if node.args and isinstance(node.args[0], ast.Name):
            return ("tuple", node.args[0].id)  # a CB-args constant; length checked below
    return None


def _kernel_source_name(call) -> str | None:
    """The kernel file a KernelDescriptor names, unwrapping ``str(KERNELS_DIR / "x.cpp")``."""
    for kw in call.keywords:
        if kw.arg != "kernel_source" or not isinstance(kw.value, ast.Call):
            continue
        node = kw.value.args[0] if kw.value.args else None
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "str" and node.args:
            node = node.args[0]
        if isinstance(node, ast.BinOp):
            node = node.right
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
    return None


def _kernel_accessor_count(path: Path) -> int:
    """How many TensorAccessors a kernel constructs, with comments stripped.

    Counting ``TensorAccessorArgs<`` instead would be wrong: the kernels' own header comments
    mention the type by name, and a comment cannot construct an accessor.
    """
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S))
    return len(re.findall(r"\bTensorAccessor\s*\(", code))


DM_PIPELINE_KERNELS = (
    "reader_spike_state.cpp",
    "reader_sparse_weights.cpp",
    "writer_spike_state.cpp",
    "writer_matvec_out.cpp",
)


@pytest.mark.parametrize("kernel_name", DM_PIPELINE_KERNELS)
def test_accessor_count_matches_the_tensors_the_host_wraps(kernel_name):
    """Each data-movement kernel's accessor chain must be fed by the same number of tensors.

    ``dm_compile_time_args(cb_args, *tensors)`` lays out ``[offset, *cb_args, *accessor blocks]``,
    and the kernels walk that chain with ``TensorAccessorArgs<offset>()`` plus
    ``next_compile_time_args_offset()``. A kernel that constructs one accessor fewer than the host
    wraps leaves the surplus tensor's block unread; one more and the last accessor reads past the
    end. Both are silent -- the kernel compiles, dispatches, and reads whatever compile-time words
    happen to be there.

    Counting is on both sides only: accessor constructions in the kernel, and positional tensor
    arguments to a named call in the host. It never evaluates the ``[offset] + list(cb_args) +
    TensorAccessorArgs(...)`` expression, which is what made the earlier builder-side check
    unreliable.
    """
    host_tensors = None
    for module in sorted(KERNELS.parent.glob("*.py")):
        tree = ast.parse(module.read_text())
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "KernelDescriptor"
            ):
                continue
            if _kernel_source_name(call) != kernel_name:
                continue
            for kw in call.keywords:
                if kw.arg != "compile_time_args":
                    continue
                cta = kw.value
                if isinstance(cta, ast.Call) and getattr(cta.func, "id", None) == "dm_compile_time_args":
                    host_tensors = len(cta.args) - 1  # the first positional is the CB-args tuple
    assert host_tensors is not None, (
        f"{kernel_name}: no host call site supplies its compile-time args through "
        "dm_compile_time_args, so this check is not examining anything and must not be trusted"
    )
    accessors = _kernel_accessor_count(KERNELS / kernel_name)
    assert accessors > 0, (
        f"{kernel_name}: no TensorAccessor construction was recognised; the kernel does not read "
        "tensors, or it uses a shape this parser does not know"
    )
    assert accessors == host_tensors, (
        f"{kernel_name}: constructs {accessors} TensorAccessor(s) but the host wraps "
        f"{host_tensors} tensor(s). The accessor chain and the compile-time argument blocks "
        "disagree, and nothing in the package would notice"
    )


PIPELINES = {
    "lif": ("reader_spike_state.cpp", "compute_lif_neuron.cpp", "writer_spike_state.cpp"),
    "sparse": ("reader_sparse_weights.cpp", "compute_spike_matvec.cpp", "writer_matvec_out.cpp"),
}


def _push_pop_counts(path: Path):
    """``{(cb, "push"|"pop"): per-tile count}`` for one kernel.

    Both circular-buffer API styles are in use here -- the free functions ``cb_push_back(cb, 1)``
    and the ``CircularBuffer`` class form ``cb_x_buf.push_back(1)`` -- and a scan that only
    recognises one of them reports every kernel as balanced, which is worse than not scanning.
    """
    code = re.sub(r"//[^\n]*", "", path.read_text())
    counts: dict = {}
    for m in re.finditer(r"CircularBuffer\s+(\w+)\s*\(\s*([A-Za-z_]\w*)\s*\)", code):
        var, cb = m.group(1), m.group(2)
        counts[(cb, "push")] = counts.get((cb, "push"), 0) + len(re.findall(re.escape(var) + r"\.push_back\s*\(", code))
        counts[(cb, "pop")] = counts.get((cb, "pop"), 0) + len(re.findall(re.escape(var) + r"\.pop_front\s*\(", code))
    for m in re.finditer(r"\bcb_push_back\s*\(\s*([A-Za-z_]\w*)\s*,\s*1\s*\)", code):
        counts[(m.group(1), "push")] = counts.get((m.group(1), "push"), 0) + 1
    for m in re.finditer(r"\bcb_pop_front\s*\(\s*([A-Za-z_]\w*)\s*,\s*1\s*\)", code):
        counts[(m.group(1), "pop")] = counts.get((m.group(1), "pop"), 0) + 1
    # wait_front and reserve_back are counted only so a kernel this parser failed to understand
    # is visible. They take part in no balance; see test_every_pipeline_kernel_was_parsed.
    for m in re.finditer(r"CircularBuffer\s+(\w+)\s*\(\s*[A-Za-z_]\w*\s*\)", code):
        var = m.group(1)
        counts[("__seen__", "wait")] = counts.get(("__seen__", "wait"), 0) + len(
            re.findall(re.escape(var) + r"\.wait_front\s*\(", code)
        )
        counts[("__seen__", "reserve")] = counts.get(("__seen__", "reserve"), 0) + len(
            re.findall(re.escape(var) + r"\.reserve_back\s*\(", code)
        )
    for op, fn in (("wait", "cb_wait_front"), ("reserve", "cb_reserve_back")):
        counts[("__seen__", op)] = counts.get(("__seen__", op), 0) + len(
            re.findall(r"\b" + fn + r"\s*\(\s*[A-Za-z_]\w*\s*,", code)
        )
    return counts


@pytest.mark.parametrize("kernel", [KERNELS / n for names in PIPELINES.values() for n in names], ids=lambda p: p.name)
def test_every_pipeline_kernel_was_parsed(kernel):
    """A circular-buffer call shape this parser misses must fail loudly, not read as balanced.

    The balance check below treats "no pushes and no pops" as balanced, which is correct only if
    zero really means zero. A kernel written in a third call style -- or one whose body this
    regular-expression scan does not recognise -- would report 0/0 and pass, which is a false
    negative indistinguishable from a correct result. That is the same failure as a pattern too
    narrow to see `sub_reuse_dest_tiles<...>(` at all: a clean result and an unexamined one look
    identical. So each kernel in a pipeline must account for at least one recognised
    push/pop/wait/reserve call, and a kernel with none is a failure naming the reason.
    """
    counts = _push_pop_counts(kernel)
    seen = sum(n for (cb, op), n in counts.items() if cb == "__seen__")
    moved = sum(n for (cb, op), n in counts.items() if cb != "__seen__")
    assert seen + moved > 0, (
        f"{kernel.name}: no circular-buffer push, pop, wait or reserve call was recognised. Either "
        "the kernel does not move tiles, or it uses a call shape this parser does not know -- in "
        "which case the balance check below is reporting silence as balance and must not be trusted"
    )


@pytest.mark.parametrize("pipeline", sorted(PIPELINES))
def test_pushes_and_pops_balance_across_each_pipeline(pipeline):
    """Every tile a producer publishes must be consumed, or the pipeline deadlocks.

    On a Tensix core the reader, compute and writer kernels run concurrently and the circular
    buffer's push/pop *is* the synchronisation between them -- there are no other cross-kernel
    signals in this design. A producer that pushes once more per tile than its consumer pops
    therefore does not overrun: it eventually blocks in ``cb_reserve_back`` on a full queue, and
    the consumer never sees the tile. That is a hang, not a wrong number, so nothing else in this
    package would report it.

    ``cb_index`` in the sparse reader is reserved once before the loop and never pushed: it is
    used as a one-shot L1 scratch region for the gathered index list, and nothing else touches it,
    so zero pushes and zero pops is the correct balance there.
    """
    pushes: dict = {}
    pops: dict = {}
    for name in PIPELINES[pipeline]:
        for (cb, op), n in _push_pop_counts(KERNELS / name).items():
            (pushes if op == "push" else pops)[cb] = (pushes if op == "push" else pops).get(cb, 0) + n
    for cb in sorted((set(pushes) | set(pops)) - {"__seen__"}):  # __seen__ is the parse sentinel, not a CB
        assert pushes.get(cb, 0) == pops.get(cb, 0), (
            f"{pipeline} pipeline: {cb} is pushed {pushes.get(cb, 0)}x and popped "
            f"{pops.get(cb, 0)}x per tile. The consumer will block in cb_reserve_back once the "
            "queue fills, which is a hang rather than a wrong value"
        )


KERNEL_FILES = sorted(KERNELS.glob("*.cpp"))


def test_kernels_were_found():
    assert len(KERNEL_FILES) >= 8, f"only found {len(KERNEL_FILES)} kernels under {KERNELS}"


@pytest.mark.parametrize("kernel", KERNEL_FILES, ids=lambda p: p.name)
def test_compile_time_arg_indices_are_unique_and_contiguous(kernel):
    """A kernel reading slot 0 twice, or skipping a slot, is always a mistake."""
    mapping = _kernel_cta_indices(kernel)
    assert mapping, f"{kernel.name} reads no compile-time arguments; is it still the same kernel?"
    indices = sorted(mapping.values())
    assert len(indices) == len(set(indices)), f"{kernel.name}: duplicate index in {mapping}"
    assert indices == list(
        range(len(indices))
    ), f"{kernel.name}: compile-time indices are not contiguous from 0, got {indices} ({mapping})"


# Does the host hand each CB to the slot the kernel reads it from?
#
# The contiguity check above proves the kernel's own slots are 0..n-1 with no duplicates. It says
# nothing about whether slot 2 holds the CB the host *thinks* it is slot 2, so swapping two
# entries in COMPUTE_CB_ARGS would leave every other check in this package green while each kernel
# silently read the wrong circular buffer. layout.py already names the hazard on the tuple:
# "These are positional: reordering one is an ABI change for the .cpp that consumes it."
#
# Comparing the two sides needs no expression counting. COMPUTE_CB_ARGS is a plain tuple literal
# and the kernel's reads come from _kernel_cta_indices, so the check is a list comparison on names
# that correspond once case is folded -- the shape the removed builder-side check could not reach
# without resolving `list(CB_ARGS) + [offset] + TensorAccessorArgs(...)`.
#
# MATVEC_COMPUTE_CB_ARGS is deliberately NOT checked here. Its kernel names its arguments
# cb_weight / cb_spike / cb_out against the host's CB_MV_WEIGHT / CB_MV_SPIKE / CB_MV_OUT, so a
# name comparison would need a hand-maintained renaming that could itself drift out of step. It is
# left to the kernel's own comments rather than guarded by a mapping that can rot.
COMPUTE_KERNELS = sorted(KERNELS.glob("compute_*.cpp"))
# compute_lif_neuron is known to violate this. Shipped as a strict xfail so the defect is a live
# signal rather than prose: when the kernel is fixed the test XPASSes, and strict=True turns that
# into a failure demanding the marker be removed. See the Phase 1 status section of the package
# README for the register-level reason.
KNOWN_UNCONFIGURED = {
    "compute_lif_neuron.cpp": (
        "packs to c_16/c_17/c_18 and unpacks from c_16/c_18, none of which "
        "compute_kernel_hw_startup(cb_v_old, cb_in, cb_v_out) configures; needs "
        "llk_unpack_hw_configure + llk_pack_reconfig_data_format per destination change"
    )
}


def _engine_cbs(path: Path):
    """``(startup_args, pack_destinations, unpack_sources, reconfigured)`` for a compute kernel.

    Parsed with regex, not a real C++ parse, and deliberately conservative: a form it does not
    recognise is simply not reported, so the failure mode is missing a defect rather than
    inventing one. Both compute kernels in this package parse completely.
    """
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S))
    startup = re.search(r"compute_kernel_hw_startup\s*(?:<[^>]*>)?\s*\(([^)]*)\)", code)
    assert startup, f"{path.name} never calls compute_kernel_hw_startup; is it still a compute kernel?"
    args = [a.strip() for a in startup.group(1).split(",")]
    assert len(args) == 3, f"{path.name}: expected (icb0, icb1, ocb), got {args}"

    packs = set(re.findall(r"pack_tile\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)", code))
    sources = set(re.findall(r"copy_tile\s*\(\s*([A-Za-z_]\w*)", code))
    sources |= set(re.findall(r"\w+_reuse_dest_tiles\s*<[^>]*>\s*\(\s*([A-Za-z_]\w*)", code))
    sources |= set(
        re.findall(r"matmul_tiles\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)", code)
        and [c for pair in re.findall(r"matmul_tiles\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)", code) for c in pair]
    )
    reconfigured = set(re.findall(r"reconfig_data_format\s*(?:<[^>]*>)?\s*\([^)]*?,\s*([A-Za-z_]\w*)\s*\)", code))
    reconfigured |= set(re.findall(r"(?:un)?pack_hw_configure\s*(?:<[^>]*>)?\s*\(\s*([A-Za-z_]\w*)", code))
    return args, packs, sources, reconfigured


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(k, marks=pytest.mark.xfail(strict=True, reason=KNOWN_UNCONFIGURED[k.name]))
        if k.name in KNOWN_UNCONFIGURED
        else k
        for k in COMPUTE_KERNELS
    ],
    ids=lambda p: p.name,
)
def test_every_compute_engine_cbs_are_configured_at_startup(kernel):
    """Every CB a compute kernel packs to or unpacks from must be configured before it is used.

    `compute_kernel_hw_startup(icb0, icb1, ocb)` configures the unpacker for two sources and the
    packer for one destination, once, at the top. `get_output_id()` is the identity function, so a
    packer output slot *is* a CB id, and both engines read per-CB format and face-geometry entries
    out of tables only the configure and reconfig calls fill in. A kernel that rotates through an
    intermediate therefore has to reconfigure *both* engines on every change of destination --
    UNPACK(llk_unpack_hw_configure(new_icb)) paired with
    PACK(llk_pack_reconfig_data_format(old_ocb, new_ocb)), as `reconfigure_unary_bcast` does at
    api/compute/bcast.h:164-194. Neither engine's format survives a destination change alone.

    Note the failure mode this guards is silent: `are_packers_configured_correctly` sits inside
    LLK_ASSERT_BLOCK and only fires in a sanitised build, so an unconfigured CB produces wrong
    values with no error rather than a trap.
    """
    args, packs, sources, reconfigured = _engine_cbs(kernel)
    unconfigured_pack = sorted(packs - {args[2]} - reconfigured)
    unconfigured_unpack = sorted(sources - {args[0], args[1]} - reconfigured)
    assert not unconfigured_pack and not unconfigured_unpack, (
        f"{kernel.name} uses circular buffers that no startup or reconfig call ever configures. "
        f"startup=({args[0]}, {args[1]}) -> {args[2]}; reconfigured={sorted(reconfigured)}; "
        f"unconfigured pack destinations={unconfigured_pack}, "
        f"unconfigured unpack sources={unconfigured_unpack}"
    )


LAYOUT = KERNELS.parent / "layout.py"
CB_ARGS_TUPLES = {
    "COMPUTE_CB_ARGS": "compute_lif_neuron.cpp",
    "MATVEC_COMPUTE_CB_ARGS": "compute_spike_matvec.cpp",
}
# Only the first names its arguments identically on both sides, so only it can be compared by
# name. The structural half below covers both.
NAME_COMPARABLE = {"COMPUTE_CB_ARGS"}


def _layout_tuple(name: str) -> list[str]:
    """The elements of a ``NAME = (A, B, C)`` tuple literal in layout.py, parsed not imported."""
    match = re.search(rf"^{name}\s*=\s*\(([^)]*)\)", LAYOUT.read_text(), re.M)
    assert match, f"{name} is not a plain tuple literal in layout.py"
    return [element.strip() for element in match.group(1).split(",") if element.strip()]


@pytest.mark.parametrize("tuple_name", sorted(CB_ARGS_TUPLES))
def test_cb_args_order_matches_the_slot_its_kernel_reads(tuple_name):
    """The host's positional CB tuple must match the order the kernel reads those slots in."""
    host = [name.lower() for name in _layout_tuple(tuple_name)]
    kernel = _kernel_cta_indices(KERNELS / CB_ARGS_TUPLES[tuple_name])
    read = {name: index for name, index in kernel.items() if name.lower().startswith("cb_")}
    assert read, f"{CB_ARGS_TUPLES[tuple_name]} reads no CB compile-time arguments"
    slots = sorted(read.values())
    # A kernel may read non-CB compile-time arguments first (compute_spike_matvec takes Mt, Kt, Nt
    # at 0, 1, 2 and its CBs at 3, 4, 5), so the comparison is against the cb_-named slots only and
    # never against raw slot numbers.
    # This is NOT implied by test_compile_time_arg_indices_are_unique_and_contiguous. That check
    # only requires the *total* index set to be 0..n-1; a cb_-named subset of that set can still
    # have gaps, because a subset of a contiguous range need not itself be contiguous. Interleaving
    # a non-CB argument between two CB arguments (Mt, cb_weight, Kt, cb_spike, ...) keeps the total
    # set intact while putting a hole in the CB run -- and that is exactly the permutation that
    # would make a positional CB tuple wrong while every other check stays green.
    assert slots == list(range(slots[0], slots[0] + len(slots))), (
        f"{CB_ARGS_TUPLES[tuple_name]}: the CB compile-time arguments are not consecutive -- "
        f"got slots {slots}, so a non-CB argument is interleaved between them"
    )
    assert len(host) == len(slots), (
        f"{tuple_name} has {len(host)} entries but {CB_ARGS_TUPLES[tuple_name]} reads "
        f"{len(slots)} CB compile-time arguments"
    )
    if tuple_name not in NAME_COMPARABLE:
        return
    in_order = [name.lower() for name, _ in sorted(read.items(), key=lambda item: item[1])]
    assert host == in_order, (
        f"{tuple_name} is {host} but {CB_ARGS_TUPLES[tuple_name]} reads {in_order}; "
        "reordering one is an ABI change for the kernel that consumes it"
    )


# The runtime half of a kernel's interface, kernel side only. ``compute_lif_neuron`` reads
# alpha, v_threshold, v_reset and n_tiles in that order, so a host that supplied them in any
# other order would compile, dispatch, and compute a plausible wrong answer.
#
# A builder-side counterpart -- "does each KernelDescriptor supply the arguments its kernel
# reads" -- was written and removed. Every builder expresses its arguments as
# ``list(CB_ARGS) + [offset] + TensorAccessorArgs(...)``, so counting the leading arguments
# correctly means resolving star-unpacking, a helper call and a concatenation at once. Three
# attempts either missed a genuinely short list or flagged a correct one. A guard that cannot be
# shown to fail on a real defect is worse than no guard, so only the kernel-side checks, whose
# negative controls were demonstrated, are kept.


def _kernel_runtime_indices(path: Path):
    """``{name: index}`` for every literal ``get_arg_val<T>(N)`` in a kernel."""
    return {f"arg{i}": int(i) for i in re.findall(r"get_arg_val<[^>]*>\((\d+)\)", path.read_text())}


@pytest.mark.parametrize("kernel", KERNEL_FILES, ids=lambda p: p.name)
def test_runtime_arg_indices_are_unique_and_contiguous(kernel):
    """Literal-index readers must start at 0 and not skip.

    A reader using ``get_arg_val<T>(arg_idx++)`` cannot skip a slot, so it is exempt.
    """
    text = kernel.read_text()
    if "get_arg_val" not in text:
        return
    if "arg_idx++" in text:
        return
    mapping = _kernel_runtime_indices(kernel)
    if not mapping:
        return
    indices = sorted(mapping.values())
    assert indices == list(range(len(indices))), (
        f"{kernel.name}: runtime argument indices are not contiguous from 0, got {indices} " f"({mapping})"
    )
