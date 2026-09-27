# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The circular-buffer map in `snn/layout.py`, which nothing else checks on a host without ttnn.

`layout.py` is the single owner of buffer geometry: the page size comes from the dtype, the total
from the page and the depth, and `cb_descriptor` derives both from here so a descriptor's
`total_size` and `page_size` cannot disagree. That ownership is why the map needs its own checks.
Three failure modes it leaves open, none of which any device-free test would catch:

* A CB index that some kernel or argument tuple names but the size and dtype tables do not define.
  Nothing notices until `cb_descriptor` raises `KeyError` while building a descriptor -- which
  needs a real ttnn, so on a bare host it is invisible and every other test still passes.
* A dtype drifting away from its page size. bfloat16 at 32x32 is half a float32 tile, so a bf16
  buffer given the fp32 page size reads twice as far as the kernel expects, and the reverse reads
  half as far. Neither shows up until a kernel runs.
* A buffer added on top of a full budget. `multicore._check_l1` and the fabric guard both raise
  when `l1_bytes(...) > L1_BYTES_PER_CORE`, so if the table is wrong the budget check is wrong too,
  and a new buffer that breaks the budget should fail here rather than at dispatch.

This module is ttnn-free; that is the point of it. It imports `layout` as a submodule, which
`test_package_imports.py` now resolves as a submodule rather than demanding an `__init__` entry.
"""

import numpy as np

from models.experimental.snn.snn import layout as L

# Every CB index any kernel reaches. The six argument tuples are how the host names them; the two
# Phase 4 buffers are named directly in fabric.py and multicore.py, so listing only the tuples
# would leave them unchecked.
REFERENCED = {
    "COMPUTE_CB_ARGS": L.COMPUTE_CB_ARGS,
    "READER_CB_ARGS": L.READER_CB_ARGS,
    "WRITER_CB_ARGS": L.WRITER_CB_ARGS,
    "MATVEC_COMPUTE_CB_ARGS": L.MATVEC_COMPUTE_CB_ARGS,
    "MATVEC_READER_CB_ARGS": L.MATVEC_READER_CB_ARGS,
    "MATVEC_WRITER_CB_ARGS": L.MATVEC_WRITER_CB_ARGS,
    "fabric (named directly)": (L.CB_SPIKE_OUT,),
    "noc handoff (named directly)": (L.CB_SPIKE_IN,),
}

# The two programs' buffer sets, stated rather than left incidental, because these are the two
# numbers `_check_l1` and the fabric guard will compare against the 128 KiB a core has.
LIF_PIPELINE = L.READER_CB_ARGS + L.COMPUTE_CB_ARGS + L.WRITER_CB_ARGS
MATVEC_PIPELINE = L.MATVEC_READER_CB_ARGS + L.MATVEC_COMPUTE_CB_ARGS + L.MATVEC_WRITER_CB_ARGS

# Page size is tile elements times the dtype's itemsize, derived rather than written as literals
# so a change to TILE_ELEMENTS or to a dtype is caught here instead of silently resizing every
# descriptor built from this table.
ITEMSIZE = {"float32": np.dtype(np.float32).itemsize, "bfloat16": 2, "uint32": np.dtype(np.uint32).itemsize}


def test_every_referenced_cb_has_a_size_and_a_dtype():
    """A missing table entry is a KeyError that only appears when a descriptor is built."""
    offenders = {
        name: [index for index in indices if index not in L.CB_TOTAL_BYTES or index not in L.CB_DTYPE]
        for name, indices in REFERENCED.items()
    }
    offenders = {name: idx for name, idx in offenders.items() if idx}
    assert not offenders, (
        f"circular buffers referenced but absent from the size/dtype tables: {offenders}. "
        "cb_descriptor would raise KeyError while building a descriptor, which needs a real ttnn "
        "and so is invisible to every device-free test"
    )


def test_page_size_is_the_tile_times_the_dtype_width():
    """Each dtype's page is one whole tile of that dtype -- not another dtype's tile."""
    wrong = {
        index: (dtype, L.CB_PAGE_BYTES[index], L.TILE_ELEMENTS * ITEMSIZE[dtype])
        for index, dtype in L.CB_DTYPE.items()
        if L.CB_PAGE_BYTES[index] != L.TILE_ELEMENTS * ITEMSIZE[dtype]
    }
    assert not wrong, (
        f"page size does not match TILE_ELEMENTS * itemsize for the declared dtype: {wrong}. "
        "A bf16 buffer given the fp32 page size would read twice as far as the kernel expects"
    )


def test_total_is_whole_pages_over_the_declared_depth():
    """`cb_descriptor` sets total_size from CB_TOTAL_BYTES and page_size from CB_PAGE_BYTES.

    If the two disagreed, every descriptor for that buffer would declare a total that is not a
    whole number of pages, and no host-side check would notice.
    """
    for index in L.CB_DTYPE:
        page, total = L.CB_PAGE_BYTES[index], L.CB_TOTAL_BYTES[index]
        assert total == L.CB_TILES * page, (
            f"CB {index} declares {total} total bytes over a {page} B page, which is not " f"{L.CB_TILES} pages"
        )
        assert total % page == 0, f"CB {index}: the page size does not divide the declared total"


def test_l1_bytes_sums_only_what_it_is_given():
    """It is the budget check, so it must be additive and must not include unlisted buffers."""
    assert L.l1_bytes(()) == 0
    assert L.l1_bytes((L.CB_V_OLD,)) == L.CB_TOTAL_BYTES[L.CB_V_OLD]
    assert L.l1_bytes((L.CB_V_OLD, L.CB_IN)) == L.CB_TOTAL_BYTES[L.CB_V_OLD] + L.CB_TOTAL_BYTES[L.CB_IN]
    # Repeating a buffer counts it twice: a caller listing one twice is asking for two of them,
    # and collapsing that would understate the very budget it is checking.
    assert L.l1_bytes((L.CB_V_OLD, L.CB_V_OLD)) == 2 * L.CB_TOTAL_BYTES[L.CB_V_OLD]


def test_each_program_fits_a_core():
    """The two numbers `_check_l1` and the fabric guard will compare against 128 KiB."""
    assert L.L1_BYTES_PER_CORE == 128 * 1024, "the per-core L1 figure is a hardware constant"
    assert L.CB_TILES == 2, "the map double-buffers every queue, which is what the readers assume"
    for name, group in (("LIF", LIF_PIPELINE), ("matvec", MATVEC_PIPELINE)):
        need = L.l1_bytes(group)
        assert (
            need <= L.L1_BYTES_PER_CORE
        ), f"the {name} pipeline needs {need} B of L1, over the {L.L1_BYTES_PER_CORE} B one core has"


def test_every_buffer_in_the_map_fits_a_core_together():
    """The claim a reviewer would most want stated: the whole map, not just one program.

    A single program fits comfortably, but the map is what a future buffer gets added to, and a
    CB that pushes the total past the per-core figure should fail on a bare host rather than at
    dispatch.
    """
    total = sum(L.CB_TOTAL_BYTES.values())
    assert total <= L.L1_BYTES_PER_CORE, (
        f"all {len(L.CB_DTYPE)} buffers together need {total} B, over the " f"{L.L1_BYTES_PER_CORE} B one core has"
    )
    headroom = L.L1_BYTES_PER_CORE - total
    assert headroom > 0, "the map exactly fills a core's L1, leaving no room for another buffer"
