# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for ``lif_neuron`` (``models/experimental/snn/snn/neuron.py``).

The LIF kernel is the only part of the framework that carries state: the reader pulls the
membrane out of DRAM and the writer puts the next one back, so the contract under test is a
*trace* over many steps, not a single update. A kernel that got step 1 right and then lost a
tile, skipped the reset, or stopped clearing the spike buffer still passes a one-step test.

Every assertion below is either exact or tolerant, and the distinction is deliberate:

  * the spike train is compared **exactly**. A spike is 0.0 or 1.0, bfloat16 represents
    both without loss, and ``unary_gt_tile`` is a comparison rather than an approximation,
    so there is nothing to be tolerant about. The device has to fire on the same steps the
    reference does -- the same steps, not the same steps plus or minus one.
  * the membrane is compared with the tight tolerance derived at ``MEMBRANE_RTOL``.

Tile counts 1, 2 and 4 are all exercised. Every circular buffer in the pipeline is two tiles
deep (``snn/layout.py:24``), so one tile never wraps, two fills each queue exactly, and four
forces the reader and the compute kernel to lap each other twice inside a single dispatch. A
buffer sized or addressed for one tile only corrupts the tail of a longer run and is
invisible at one tile, which is why the shape is a parameter and not a constant.

Two sentinel values guard the output tensors. A spike is 0.0 or 1.0 and a membrane is O(1),
so ``-1.0`` and ``-9999.0`` can only survive to the read-back if the writer never touched
that tile. Without them, "the spike train is all zeros" and "the writer silently did
nothing" are the same observation.
"""

import numpy as np
import pytest
import ttnn

from models.experimental.snn.reference.lif import lif_run
from models.experimental.snn.snn.config import LIFConfig
from models.experimental.snn.snn.neuron import lif_neuron

SPIKING = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)
ALPHA = np.float32(SPIKING.decay_factor)

# rtol/atol for the membrane trace, and why they are this tight.
#
# The compute kernel is built with `fp32_dest_acc_en=True` and `MathFidelity.HiFi4`
# (`snn/neuron.py`), so the membrane path is `mul_unary_tile` followed by
# `add_reuse_dest_tiles` into a float32 destination register. The one step that is not an
# FPU op is `unary_gt_tile`, which is an exact comparison and never writes the membrane
# register. What is left is float32 rounding applied to a *contractive* map -- |alpha| is
# 0.951, so errors shrink rather than grow -- at most 13 times, to values of order 1. That
# is a couple of ulps, i.e. below 1e-6. These bounds sit an order of magnitude above that
# and still four orders of magnitude below the 3.4% margin the periodic stimulus below
# leaves under the threshold, so a drift large enough to flip a spike fails long before
# the tolerance does.
MEMBRANE_RTOL = 1e-5
MEMBRANE_ATOL = 1e-6

# Survives only if the writer skipped this tile. -1.0 is not a spike and -9999.0 is not a
# membrane, so neither can be confused with a correct result.
SPIKE_SENTINEL = -1.0
V_OUT_SENTINEL = -9999.0

# One drive per neuron slot, cycling four regimes inside every tile: permanently silent,
# integrating without ever crossing, firing on a period, and firing every step. The
# per-tile scale below makes the four tiles genuinely different (they spike 120 / 128 / 128
# / 144 times over 8 steps), so a kernel that serviced only the first tile, or that fed
# tile 0's current to the whole population, lands on a different trace.
SLOT_CURRENT = np.array([0.0, 0.4, 0.6, 1.2], dtype=np.float32)

# I = 0.26 against v_threshold = 1.0 and v_reset = 1.0 fires on a clean period of 4. The
# membrane before each step, worked out by hand from alpha = exp(-1/20) = 0.951229:
#
#   0.2600  0.5073  0.7426  0.9664 | 1.1792 spike -> 0.1792
#   0.4305  0.6695  0.8968          | 1.1131 spike -> 0.1131
#   0.3676  0.6097  0.8399          | 1.0590 spike -> 0.0590
#
# so spikes land on 0-based step indices 4, 8 and 12. The tightest margin in that schedule
# is 3.4% under the threshold, which is why the schedule is a safe place to assert exact
# spike positions: a membrane error of 0.03 would be needed to move a spike, and the
# tolerance above is 1e-5.
PERIODIC_CURRENT = 0.26
PERIODIC_STEPS = 13
PERIODIC_SPIKES = [4, 8, 12]

# I = 0.04 integrates towards I / (1 - alpha) = 0.8202 and never crosses, so a neuron
# driven this way is quiet forever. I = 1.2 crosses on every step.
QUIET_CURRENT = 0.04
LOUD_CURRENT = 1.2

# N_tiles chosen to straddle the two-tile circular-buffer depth; see the module docstring.
TILE_COUNTS = [1, 2, 4]
STEPS = 8


def _stimulus(n_tiles):
    """``(v_mem, current)`` for ``n_tiles * 32`` neurons, one distinct regime per slot."""
    index = np.arange(n_tiles * 32)
    scale = 1.0 + 0.125 * (index // 32)
    return (0.05 * (index // 32)).astype(np.float32), (SLOT_CURRENT[index % 4] * scale).astype(np.float32)


def _pack_column_zero(values, n_tiles):
    """Lay a per-neuron vector into column 0 of every output tile.

    Column 0 is where the per-neuron data lives for these kernels: ``SparseLIFLayer``
    writes the input current there and ``synapses.extract_current`` reads it back. A row-
    major fill would put the vector along a row instead, and the kernel is elementwise over
    the whole tile, so the values would be right but the read-back would be transposed.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    assert len(values) == n_tiles * 32, f"{len(values)} neurons does not fill {n_tiles} tiles"
    block = np.zeros((1, n_tiles, 32, 32), dtype=np.float32)
    block[0, np.arange(len(values)) // 32, np.arange(len(values)) % 32, 0] = values
    return block


def _unpack_column_zero(block):
    """Inverse of :func:`_pack_column_zero`: the per-neuron vector back out of a tile stack."""
    return np.asarray(block, dtype=np.float32)[0, :, :, 0].reshape(-1).copy()


def _run(v_init, currents, device, tile_tensor, read_back, config=SPIKING):
    """Drive ``lif_neuron`` once per entry in ``currents``.

    Returns ``(membrane_trace, spike_train)`` with the initial membrane prepended as row 0,
    so row ``k`` is the state *after* ``k`` time steps and ``trace[k - 1]`` is always the
    state the step started from -- which is what the reset arithmetic needs.
    """
    n_tiles = len(v_init) // 32
    n_steps = len(currents)

    v_mem = tile_tensor(_pack_column_zero(v_init, n_tiles), ttnn.float32, device)
    v_out = tile_tensor(np.full((1, n_tiles, 32, 32), V_OUT_SENTINEL, dtype=np.float32), ttnn.float32, device)
    spikes = tile_tensor(np.full((1, n_tiles, 32, 32), SPIKE_SENTINEL, dtype=np.float32), ttnn.bfloat16, device)

    trace = np.empty((n_steps + 1, n_tiles * 32), dtype=np.float32)
    train = np.empty((n_steps, n_tiles * 32), dtype=np.float32)
    trace[0] = v_init
    for step in range(n_steps):
        current = tile_tensor(_pack_column_zero(currents[step], n_tiles), ttnn.float32, device)
        lif_neuron(v_mem, current, spikes, v_out, config)
        trace[step + 1] = _unpack_column_zero(read_back(v_out))
        train[step] = _unpack_column_zero(read_back(spikes))
        # The writer produced the next membrane in v_out; it is the next step's v_mem.
        # Both arguments are device tensors: `ttnn.copy` is bound to
        # `(const ttnn::Tensor&, const ttnn::Tensor&)` with `.noconvert()` on both, so passing
        # `ttnn.to_torch(v_out)` -- which returns a torch tensor, not a ttnn one -- would raise a
        # TypeError on the first step of every test in this file.
        ttnn.copy(v_out, v_mem)
    return trace, train


@pytest.mark.parametrize("n_tiles", TILE_COUNTS, ids=["1_tile", "2_tiles", "4_tiles"])
def test_spike_train_matches_the_reference_exactly(device, reset_seeds, tile_tensor, read_back, n_tiles):
    """Exact. The device must fire on the same steps the reference does."""
    v_init, current = _stimulus(n_tiles)
    _, device_train = _run(v_init, [current] * STEPS, device, tile_tensor, read_back)
    _, reference_train = lif_run(v_init, current, SPIKING, STEPS)

    # Not a guard for its own sake: without spikes this whole module would be satisfied by
    # a kernel that never fires, which is the cheapest possible way to pass.
    assert device_train.sum() > 0
    assert np.array_equal(device_train, reference_train)


@pytest.mark.parametrize("n_tiles", TILE_COUNTS, ids=["1_tile", "2_tiles", "4_tiles"])
def test_membrane_trace_matches_the_reference(device, reset_seeds, tile_tensor, read_back, n_tiles):
    """Tolerant, at ``MEMBRANE_RTOL``. See that constant for the derivation."""
    v_init, current = _stimulus(n_tiles)
    device_trace, _ = _run(v_init, [current] * STEPS, device, tile_tensor, read_back)
    reference_trace, _ = lif_run(v_init, current, SPIKING, STEPS)

    assert np.allclose(
        device_trace[1:], reference_trace, rtol=MEMBRANE_RTOL, atol=MEMBRANE_ATOL
    ), f"largest membrane deviation {np.max(np.abs(device_trace[1:] - reference_trace)):.3e}"
    if n_tiles > 1:
        # The stimulus gives each tile a different drive, so the tiles have to come back
        # different. A pipeline that serviced only tile 0, or that read tile 0's current
        # for the whole population, would make every tile identical to the first.
        for tile in range(1, n_tiles):
            assert not np.array_equal(
                device_trace[-1, tile * 32 : (tile + 1) * 32], device_trace[-1, 0:32]
            ), f"tile {tile} came back identical to tile 0"


def test_reset_subtracts_exactly_v_reset(device, reset_seeds, tile_tensor, read_back):
    """Tolerant, at ``MEMBRANE_RTOL``, against the device's own arithmetic.

    On a spiking step the post-reset value is ``alpha * v_previous + I - v_reset``; on a
    quiet step it is ``alpha * v_previous + I``. A kernel missing the reset subtraction
    lands a full ``v_reset`` (1.0) high, which is five orders of magnitude outside the
    tolerance. A kernel applying the subtraction unconditionally lands ``v_reset`` low on
    every quiet step, which the second half of the loop catches.
    """
    v_init = np.zeros(32, dtype=np.float32)
    current = np.full(32, PERIODIC_CURRENT, dtype=np.float32)
    trace, train = _run(v_init, [current] * PERIODIC_STEPS, device, tile_tensor, read_back)

    spiking = np.flatnonzero(train[:, 0])
    assert list(spiking) == PERIODIC_SPIKES, f"firing schedule moved: {spiking.tolist()}"
    # Every neuron here is driven identically, so they all fire on the same steps. A kernel
    # that leaked a tile's data into a neighbour's would break this but not the per-element
    # reference comparison, which only ever checks values against their own reference.
    assert np.array_equal(train[:, 1:], np.repeat(train[:, [0]], train.shape[1] - 1, axis=1))

    for step in range(PERIODIC_STEPS):
        pre_reset = ALPHA * trace[step] + current
        expected = pre_reset - train[step] * np.float32(SPIKING.v_reset)
        assert trace[step + 1] == pytest.approx(
            expected, rel=MEMBRANE_RTOL, abs=MEMBRANE_ATOL
        ), f"step {step}: reset branch did not subtract v_reset"


def test_firing_schedule_is_periodic_and_lands_on_the_hand_derived_steps(device, reset_seeds, tile_tensor, read_back):
    """Exact, against a schedule derived by hand in the ``PERIODIC_CURRENT`` comment.

    Comparing to ``lif_run`` alone would be circular: the reference and the kernel could
    agree on a schedule that is wrong about the physics. Pinning the positions pins it.
    """
    v_init = np.zeros(32, dtype=np.float32)
    current = np.full(32, PERIODIC_CURRENT, dtype=np.float32)
    _, train = _run(v_init, [current] * PERIODIC_STEPS, device, tile_tensor, read_back)

    assert np.flatnonzero(train[:, 0]).tolist() == PERIODIC_SPIKES
    # Silence between the spikes is part of the schedule, so assert the zeros too: a kernel
    # that fires every step would match the positions of a period-1 train on the wrong
    # stimulus and this catches the difference.
    assert train[:, 0].astype(int).tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_quiet_drive_then_loud_drive_rewrites_the_spike_buffer(device, reset_seeds, tile_tensor, read_back):
    """Exact. Catches a spike output that is never written or never cleared.

    Four steps of sub-threshold drive followed by four steps of super-threshold drive. Both
    failure modes of a spike output show up as the wrong trace here. A writer that skipped
    the spike circular buffer leaves the ``-1.0`` sentinel, which the quiet-then-loud
    transition cannot explain; a writer that pushed but never cleared the buffer leaves
    the quiet phase's zeros in place through the loud phase. Either way the trace is not
    ``[0, 0, 0, 0, 1, 1, 1, 1]``.
    """
    v_init = np.zeros(32, dtype=np.float32)
    drive = np.array([QUIET_CURRENT] * 4 + [LOUD_CURRENT] * 4, dtype=np.float32)
    currents = [np.full(32, value, dtype=np.float32) for value in drive]
    trace, train = _run(v_init, currents, device, tile_tensor, read_back)

    assert train[:, 0].astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # The quiet phase integrates towards QUIET_CURRENT / (1 - alpha) = 0.8202 without ever
    # crossing, so it has to be strictly increasing; a kernel that dropped the leak would
    # instead ramp at full drive and cross on the second quiet step.
    assert np.all(np.diff(trace[1:5, 0]) > 0)
    assert trace[4, 0] < SPIKING.v_threshold


def test_threshold_is_strict_so_an_exact_hit_does_not_fire(device, reset_seeds, tile_tensor, read_back):
    """Exact. ``v_new == v_threshold`` must not spike.

    ``unary_gt_tile`` is a strict greater-than, and ``reference.lif`` matches it. A kernel
    written against ``unary_ge_tile`` would fire here and nowhere else in this file, since
    every other stimulus is driven well clear of the threshold. The arithmetic is exact by
    construction: ``alpha * 0.0`` is 0.0 and ``0.0 + 1.0`` is 1.0 in float32, so ``v_new``
    lands on the threshold bit for bit.
    """
    v_init = np.zeros(32, dtype=np.float32)
    current = np.full(32, SPIKING.v_threshold, dtype=np.float32)
    trace, train = _run(v_init, [current], device, tile_tensor, read_back)

    assert train[0].astype(int).tolist() == [0] * 32
    assert trace[1] == pytest.approx(SPIKING.v_threshold, rel=MEMBRANE_RTOL, abs=MEMBRANE_ATOL)
