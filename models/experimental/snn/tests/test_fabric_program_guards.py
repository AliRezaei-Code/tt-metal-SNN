# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The input guards on `snn.fabric.fabric_spike_program`, which are its only device-free behaviour.

`fabric.py` builds a host-side descriptor and requires the caller to supply the writer kernel, so
the argument checks at the top of `fabric_spike_program` are the only part of the module that runs
without tt-metal. They matter because of what they prevent: a missing `writer_kernel_source` is
otherwise accepted here and only fails later at JIT time, with a message that says nothing about
which file was expected.

These are the three guards reachable from a caller's arguments. Two more sit below them -- the
`CB_SPIKE_OUT` L1 check and the `sender_args` length check -- and both compare a module constant
against a module constant, so neither branch can be reached from any argument. They are documented
at their definitions and deliberately have no tests here: asserting them would restate a literal
rather than check a property of the function.
"""

from pathlib import Path

import pytest

from models.experimental.snn.snn.fabric import fabric_spike_program

KERNELS_DIR = Path(__file__).resolve().parents[1] / "snn" / "kernels"
# Any kernel that exists satisfies the path check; which one it is does not affect these guards.
REAL_KERNEL = str(KERNELS_DIR / "writer_spike_state.cpp")


def _call(**overrides):
    """Call `fabric_spike_program` with only the argument under test varied.

    `grid`, `src_node` and `dst_node` are passed as `None` deliberately. Every guard under test
    runs before this function dereferences a `ttnn` object, so a regression that let one of them
    fall through would surface as an `AttributeError` on `None` rather than passing silently.
    """
    kwargs = {
        "writer_kernel_source": REAL_KERNEL,
        "grid": None,
        "src_node": None,
        "dst_node": None,
        "link_idx": 0,
        "dst_noc_x": 0,
        "dst_noc_y": 0,
        "dst_l1_addr": 0,
        "dst_sem_bank_addr": 0,
        "n_tiles": 1,
    }
    kwargs.update(overrides)
    return fabric_spike_program(**kwargs)


def test_a_missing_writer_kernel_is_named_in_the_error():
    """The caller must be told which path was expected, and that this module only builds the host half."""
    missing = str(KERNELS_DIR / "no_such_fabric_writer.cpp")
    with pytest.raises(FileNotFoundError) as caught:  # allow-pytest.raises: root conftest only
        _call(writer_kernel_source=missing)
    message = str(caught.value)
    assert "no_such_fabric_writer.cpp" in message, "the error must name the file that was not found"
    assert "host" in message and "descriptor" in message, "the error must say this module builds the host half only"


@pytest.mark.parametrize("n_tiles", [0, -1])
def test_a_non_positive_tile_count_is_rejected(n_tiles):
    """Zero or negative tiles would build a program that writes nothing, so it is a caller error."""
    with pytest.raises(ValueError, match="n_tiles"):  # allow-pytest.raises: root conftest only
        _call(n_tiles=n_tiles)


def test_a_negative_link_index_is_rejected():
    """`link_idx` indexes a fabric link; a negative one cannot name one."""
    with pytest.raises(ValueError, match="link_idx"):  # allow-pytest.raises: root conftest only
        _call(link_idx=-1)
