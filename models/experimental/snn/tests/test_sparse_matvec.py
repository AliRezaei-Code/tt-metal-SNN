# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for the spike-driven synaptic mat-vec (``models/experimental/snn/snn/synapses.py``).

``spike_matvec_program`` computes ``current = W @ spikes`` where the host passes down only the
*active* input tiles and the reader walks that list instead of the whole fan-in. That is the
entire point of the op: a time step in which 3% of the input population fired should cost 3% of
the weight traffic. So the tests here are shaped around the silence, not around the arithmetic:

  * **0% activity** is the case that matters. A reader that walks every page regardless of
    ``n_active`` fetches tile 0 of the weights against the all-zero spike tile and produces a
    non-zero current; the assertion that the current is exactly zero is the only thing in this
    file that can tell the two apart. ``out`` is seeded with a non-zero sentinel so that
    "exactly zero" cannot be confused with "the writer never ran".
  * **A strict subset of tiles** is the case that pins the *indexing*. If the reader ignored
    the index list and walked pages ``0 .. n_active - 1`` instead, a spike pattern living in
    tiles 3 and 5 would be read out of tiles 0 and 1 and the current would come out wrong.
  * **100% activity** is the dense-product check, and the middle rates check that partial
    sparsity is not being taken as all-or-nothing.

Tolerances: current is compared at ``CURRENT_RTOL``/``CURRENT_ATOL``, derived there. Nothing
here is compared exactly except the zero-current case, the active-tile indices, and the tile
counts -- those are integer-valued by construction and have no tolerance to argue about.

The weights come from the ``grid_weights`` fixture, which puts them on a 1/256 grid that is
exactly representable in bfloat16, so the device multiplies bit-for-bit the same values the
reference does and the tolerance only has to cover float32 accumulation. Without that,
comparing against the float32 dense product would need a ~0.4% tolerance to absorb bfloat16
weight rounding, and the test would stop being able to see anything smaller.
"""

import numpy as np
import pytest
import ttnn

from models.experimental.snn.reference.lif import sparse_matvec
from models.experimental.snn.snn.config import SynapseConfig
from models.experimental.snn.snn.synapses import (
    active_fraction,
    active_tiles,
    broadcast_spike_tiles,
    extract_current,
    pack_weight_tiles,
    spike_matvec_program,
)

# 64 outputs (2 tile rows) by 256 inputs (8 input tiles): big enough that a partial active
# set is a real subset rather than everything-or-nothing, small enough to stay cheap under
# ttsim.
FAN_OUT = 64
FAN_IN = 256
TILE = 32
N_INPUT_TILES = FAN_IN // TILE

# rtol/atol for the synaptic current.
#
# The weights are uploaded as bfloat16 but the compute kernel is built with
# `fp32_dest_acc_en=True` (`snn/synapses.py`), and every spike value is exactly 0.0 or 1.0, so
# each product is either a weight or an exact zero. The obvious remaining error would be the
# order of the 256-term float32 accumulation -- a reduction tree of depth 8, worth a handful of
# ulps. It does not arise here at all, and it is worth being precise about why: every term is a
# multiple of 1/256 and the magnitude is bounded by fan_in, so a sum needs at most ~16 mantissa
# bits and is *exact* in float32's 24. Measured against a float64 product, the relative error over
# the sweep is 0.0, not "a handful of ulps".
#
# So 1e-4 is belt and braces rather than a fitted bound: it absorbs the exact-case result, and
# it would still be four orders of magnitude tighter than the ~0.4% bfloat16 weight rounding
# that a non-grid weight matrix would introduce. The grid, not this tolerance, is what buys the
# resolution.
CURRENT_RTOL = 1e-4
CURRENT_ATOL = 1e-4

# The current for a silent population is zero, so the output tensor must not start at zero:
# otherwise "the writer never ran" and "the silence was handled correctly" are the same
# observation.
OUT_SENTINEL = -9999.0

ACTIVITY_LEVELS = [0.0, 0.05, 0.5, 1.0]
# 0% is covered on its own below, because its assertion is exact rather than tolerant.
DISPATCH_LEVELS = [0.05, 0.5, 1.0]
# Input tiles each density is expected to activate, ceil(round(activity * 256) / 32).
ACTIVE_TILES_BY_LEVEL = {0.0: 0, 0.05: 1, 0.5: 4, 1.0: 8}


def _tile_counts(fan_out, fan_in):
    """Tile rows and tile columns for a synapse, re-derived rather than imported.

    The framework has a ``tile_counts`` helper, but calling the same helper the code under
    test calls would make a bug in that helper invisible: a ceiling that rounds the wrong way
    would produce a self-consistent program and a self-consistent expectation. The hardware
    fixes the tile at 32x32, so the arithmetic is two lines and worth stating here.
    """
    return -(-fan_out // TILE), -(-fan_in // TILE)


def _run_matvec(weights, spikes, device, tile_tensor, read_back):
    """One ``spike_matvec_program`` dispatch. Returns the ``(fan_out,)`` current vector."""
    synapse = SynapseConfig(fan_in=FAN_IN, fan_out=weights.shape[0])
    mt, kt = _tile_counts(synapse.fan_out, synapse.fan_in)
    active = active_tiles(spikes)

    weights_t = tile_tensor(pack_weight_tiles(weights), ttnn.bfloat16, device)
    spikes_t = tile_tensor(broadcast_spike_tiles(spikes), ttnn.bfloat16, device)
    # The active list is a flat uint32 scratch the reader claims one tile of and never
    # publishes; see reader_sparse_weights.cpp. Unused slots stay zero.
    index = np.zeros(kt * TILE, dtype=np.uint32)
    index[: len(active)] = active
    active_t = tile_tensor(index, ttnn.uint32, device)

    out_t = tile_tensor(np.full((1, mt, TILE, TILE), OUT_SENTINEL, dtype=np.float32), ttnn.float32, device)

    program = spike_matvec_program(weights_t, active_t, spikes_t, out_t, mt, len(active))
    ttnn.generic_op([weights_t, active_t, spikes_t, out_t], program)
    return extract_current(read_back(out_t), synapse.fan_out)


def test_silence_selects_no_tiles_and_produces_exactly_zero_current(
    device, reset_seeds, tile_tensor, read_back, grid_weights
):
    """Exact, and the load-bearing test of the file.

    Two separate claims, and both are needed. ``active_tiles`` coming back empty is the
    host-side contract that makes the zero-length dispatch legal at all. The current being
    *exactly* zero is the device-side one: a reader that walked all eight input tiles anyway
    would multiply weight tile 0 by the (zero) spike tile and return a non-zero vector, and
    this is the only assertion in the file that can see the difference.
    """
    rng = np.random.default_rng(213919)
    weights = grid_weights(rng, FAN_OUT, FAN_IN)
    spikes = np.zeros(FAN_IN, dtype=np.float32)

    active = active_tiles(spikes)
    assert active.size == 0, "a silent population must select no input tiles"
    assert active_fraction(len(active), N_INPUT_TILES) == 0.0

    current = _run_matvec(weights, spikes, device, tile_tensor, read_back)
    assert np.array_equal(
        current, np.zeros(FAN_OUT, dtype=np.float32)
    ), f"silent input produced current {current[:4]} instead of zeros"


@pytest.mark.parametrize("activity", DISPATCH_LEVELS, ids=["5_percent", "50_percent", "100_percent"])
def test_matvec_matches_the_dense_product(
    device, reset_seeds, tile_tensor, read_back, grid_weights, activity, spike_pattern
):
    """Tolerant, at ``CURRENT_RTOL``.

    The ``100_percent`` row is the dense-product check in the strict sense: every input tile
    is active, the sparse path degenerates to the full fan-in, and the result has to equal
    ``W @ ones`` -- the column sums of the weight matrix. The 5% and 50% rows are the same
    assertion on a strict subset of tiles, which is what shows partial sparsity is not being
    taken as all-or-nothing.
    """
    rng = np.random.default_rng(213919)
    weights = grid_weights(rng, FAN_OUT, FAN_IN)
    spikes = spike_pattern(rng, activity, FAN_IN, TILE)

    current = _run_matvec(weights, spikes, device, tile_tensor, read_back)
    expected = sparse_matvec(weights, spikes)

    assert np.allclose(
        current, expected, rtol=CURRENT_RTOL, atol=CURRENT_ATOL
    ), f"largest current deviation {np.max(np.abs(current - expected)):.3e} at {activity:.0%} activity"
    if activity == 1.0:
        # Say what the number is supposed to be, rather than only that two computations of it
        # agree: a spike vector of all ones makes the mat-vec a column sum.
        assert np.allclose(current, weights.sum(axis=1), rtol=CURRENT_RTOL, atol=CURRENT_ATOL)


def test_only_the_listed_input_tiles_are_fetched(device, reset_seeds, tile_tensor, read_back, grid_weights):
    """Tolerant, at ``CURRENT_RTOL``, and specifically about index addressing.

    Spikes live in input tiles 3 and 5 of 8 and nowhere else, so the host's active list is
    ``[3, 5]`` while ``n_active`` is 2. A reader that walked the first two *pages* -- 0 and
    1, both silent -- would multiply weight tiles 0 and 1 by the zero spike tiles and return
    zeros, and a reader that walked all eight would get the right answer by accident. Only
    reading the listed pages produces this result.
    """
    rng = np.random.default_rng(213919)
    weights = grid_weights(rng, FAN_OUT, FAN_IN)
    spikes = np.zeros(FAN_IN, dtype=np.float32)
    spikes[3 * TILE : 4 * TILE] = 1.0
    spikes[5 * TILE : 6 * TILE] = 1.0

    assert active_tiles(spikes).tolist() == [3, 5]

    current = _run_matvec(weights, spikes, device, tile_tensor, read_back)
    expected = sparse_matvec(weights, spikes)
    # The 64 columns in the two active tiles do not cancel to zero, so the assertion below is
    # not satisfiable by a reader that fetched the wrong pages.
    assert np.max(np.abs(expected)) > 0.1
    assert np.allclose(current, expected, rtol=CURRENT_RTOL, atol=CURRENT_ATOL)
