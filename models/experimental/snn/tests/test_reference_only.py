# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth checks that need no device, no JIT and no ttnn.

This is the only suite in the package that runs on a machine without the Tenstorrent
toolchain, so it is where the reference semantics are pinned. The device suites assert the
kernels agree with what this file defines; if these expectations are wrong, every device test
is measuring the wrong thing.
"""

import math

import numpy as np
import pytest

from models.experimental.snn.reference.cpu_baseline import cpu_sparsity_report
from models.experimental.snn.reference.lif import lif_run, lif_step, sparse_matvec
from models.experimental.snn.snn.config import LIFConfig, SynapseConfig
from models.experimental.snn.snn.layout import TILE_WIDTH
from models.experimental.snn.snn.synapses import (
    active_fraction,
    active_tiles,
    broadcast_spike_tiles,
    pack_weight_tiles,
    tile_counts,
)

SPIKING = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)
# alpha = exp(-1/20); the exact per-step decay the assertions below are written against.
ALPHA = math.exp(-1.0 / 20.0)


def test_decay_factor_is_the_exact_exponential():
    assert SPIKING.decay_factor == pytest.approx(ALPHA, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0.0},
        {"tau": -1.0},
        {"dt": 0.0},
        {"dt": -1.0},
        {"v_reset": -0.5},
        {"v_threshold": 0.0},
    ],
)
def test_config_rejects_parameters_that_would_silently_corrupt_state(kwargs):
    base = {"dt": 1.0, "tau": 20.0, "v_threshold": 1.0, "v_reset": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free suite, no root conftest
        LIFConfig(**base)


def test_constant_current_fires_from_the_second_step_onward():
    """The concrete firing case, worked out by hand.

    I = 1.0, v_mem = 0, alpha = exp(-1/20). Step 1 gives v = 1.0, which is not strictly
    greater than the threshold, so no spike. From step 2 the reset subtracts exactly the
    drive that produced the spike, leaving v to decay geometrically: v[n] = alpha**(n-1).
    """
    current = np.array([1.0], dtype=np.float32)
    membrane, spikes = lif_run(np.zeros(1, dtype=np.float32), current, SPIKING, 6)

    assert spikes[0, 0] == 0.0, "v == threshold exactly must not spike; the comparison is strict"
    assert np.all(spikes[1:, 0] == 1.0)
    for n in range(2, 7):
        assert membrane[n - 1, 0] == pytest.approx(ALPHA ** (n - 1), rel=1e-5)


def test_reset_subtracts_exactly_v_reset():
    """A spiking step must lose exactly v_reset, not the whole membrane potential."""
    v_out, spikes = lif_step(np.array([0.0], dtype=np.float32), np.array([3.0], dtype=np.float32), SPIKING)
    assert spikes[0] == 1.0
    assert v_out[0] == pytest.approx(3.0 - SPIKING.v_reset, rel=1e-6)


def test_reset_to_zero_still_fires():
    zero_reset = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=0.0)
    v_out, spikes = lif_step(np.array([0.0], dtype=np.float32), np.array([3.0], dtype=np.float32), zero_reset)
    assert spikes[0] == 1.0
    assert v_out[0] == pytest.approx(3.0, rel=1e-6)


def test_no_current_never_spikes_and_decays_geometrically():
    """Zero drive cannot raise the membrane, so the neuron stays quiet forever.

    The potential never reaches zero either — it decays by ``alpha`` each step, so after ``n``
    steps it is ``alpha ** n``. A kernel that clamped to zero instead of decaying would pass
    a "never spikes" check but produce a different trace, so assert the trace, not just the
    absence of spikes.
    """
    n_steps = 40
    membrane, spikes = lif_run(np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32), SPIKING, n_steps)
    assert not spikes.any(), "with zero drive the membrane can never reach the threshold"
    assert membrane[-1, 0] == pytest.approx(ALPHA**n_steps, rel=1e-5)
    assert np.all(np.diff(membrane[:, 0]) < 0), "the membrane must decay monotonically"


def test_lif_run_is_elementwise_across_neurons():
    v_mem = np.array([0.0, 5.0, -1.0], dtype=np.float32)
    current = np.array([0.5, 0.0, 4.0], dtype=np.float32)
    v_out, spikes = lif_step(v_mem, current, SPIKING)

    v_new = ALPHA * v_mem + current
    expected_spikes = (v_new > SPIKING.v_threshold).astype(np.float32)
    assert np.array_equal(spikes, expected_spikes)
    assert np.allclose(v_out, v_new - expected_spikes * SPIKING.v_reset, rtol=1e-6, atol=1e-6)


def test_lif_run_shapes_and_repeatability():
    membrane, spikes = lif_run(np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32), SPIKING, 3)
    assert membrane.shape == (3, 5)
    assert spikes.shape == (3, 5)

    again, _ = lif_run(np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32), SPIKING, 3)
    assert np.array_equal(membrane, again)


@pytest.mark.parametrize("activity", [0.0, 0.05, 0.5, 1.0])
def test_sparse_matvec_matches_the_dense_product(activity):
    rng = np.random.default_rng(213919)
    weights = rng.normal(size=(16, 8)).astype(np.float32)
    spikes = (rng.random(8) < activity).astype(np.float32)
    assert np.allclose(sparse_matvec(weights, spikes), weights @ spikes, rtol=1e-6, atol=1e-6)


def test_sparse_matvec_of_an_all_quiet_layer_is_exactly_zero():
    """The 0% activity case: a layer that fired nowhere contributes no current."""
    weights = np.ones((6, 4), dtype=np.float32)
    assert np.array_equal(sparse_matvec(weights, np.zeros(4, dtype=np.float32)), np.zeros(6, dtype=np.float32))


def test_sparse_matvec_weighting_is_per_output_neuron():
    """One active input must reach exactly the outputs its weight column names."""
    weights = np.zeros((3, 2), dtype=np.float32)
    weights[:, 1] = [1.0, 2.0, 4.0]
    current = sparse_matvec(weights, np.array([0.0, 1.0], dtype=np.float32))
    assert np.array_equal(current, np.array([1.0, 2.0, 4.0], dtype=np.float32))


def test_synapse_config_rejects_degenerate_fan_out():
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free suite, no root conftest
        SynapseConfig(fan_in=8, fan_out=0)


# --- active-tile selection -------------------------------------------------
# These pin the granularity the sparse reader depends on. Grouping by TILE_ELEMENTS
# (1024) instead of TILE_WIDTH (32) raises for every fan-in that is not a multiple of 1024,
# which is every fan-in this framework actually uses.


@pytest.mark.parametrize("fan_in", [64, 256, 784, 1000, 1024])
def test_active_tiles_handles_fan_in_that_are_not_tile_multiples(fan_in):
    spikes = np.zeros(fan_in, dtype=np.float32)
    assert active_tiles(spikes).size == 0


def test_active_tiles_reports_exactly_the_blocks_that_fired():
    spikes = np.zeros(256, dtype=np.float32)
    spikes[3 * TILE_WIDTH + 5] = 1.0  # neuron 101, i.e. tile 3
    assert active_tiles(spikes).tolist() == [3]


def test_active_tiles_aggregates_within_a_block():
    """Several spikes in one 32-neuron block must yield one index, not several."""
    spikes = np.zeros(256, dtype=np.float32)
    spikes[TILE_WIDTH + 1] = 1.0
    spikes[TILE_WIDTH + 30] = 1.0
    assert active_tiles(spikes).tolist() == [1]


def test_active_tiles_returns_ascending_indices():
    spikes = np.zeros(256, dtype=np.float32)
    spikes[200] = 1.0
    spikes[10] = 1.0
    assert active_tiles(spikes).tolist() == [0, 6]


def test_sparsity_report_denominator_spans_every_step_not_just_one():
    """The reported fraction must divide by the tiles over the whole run, not by one step's worth.

    ``cpu_sparsity_report`` is the CPU mirror of ``SparseLIFLayer.sparsity_report`` -- the demo
    prints the two cost columns side by side -- so the arithmetic here is the arithmetic the demo
    quotes. Two things have to hold at once and neither is checked anywhere else:

    * the denominator is ``kt * n_steps``. Dropping the ``* n_steps`` reports a fraction ``n_steps``
      times too large, so a 3-step run reading 1/3 would print as 1.0.
    * the numerator accumulates. Resetting it per step leaves the count holding only the *last*
      step's activity, which here is zero, and the report reads 0.0 instead of 1/3.

    Layer 0's input is re-presented unchanged on every step, so its activity is all-or-nothing and
    cannot distinguish the two. Layer 1 sees ``trains[0][t]``, which does vary, so the check is
    made there.
    """
    fan = 32  # one tile wide, so kt == 1 and the arithmetic is easy to state exactly
    n_steps = 3
    weights = [np.zeros((fan, fan), dtype=np.float32) for _ in range(2)]

    # Layer 0 never fires, so layer 1 is handed an active tile on step 0 only.
    trains = [np.zeros((n_steps, fan), dtype=np.float32) for _ in range(2)]
    trains[0][0, 0] = 1.0
    input_spikes = np.zeros(fan, dtype=np.float32)

    reports = cpu_sparsity_report(input_spikes, weights, trains)

    assert reports[0]["tiles_fetched"] == 0, "layer 0's input is silent on every step"
    assert reports[1]["tiles_fetched"] == 1, "layer 1 saw exactly one active tile across the run"
    assert (
        reports[1]["total_tiles"] == n_steps
    ), f"total_tiles is {reports[1]['total_tiles']}, expected kt*n_steps = 1*{n_steps}"
    assert reports[1]["active_fraction"] == pytest.approx(1 / n_steps), (
        f"active_fraction is {reports[1]['active_fraction']}, expected 1/{n_steps}: the "
        "denominator must span every step and the numerator must accumulate"
    )


def test_active_fraction_matches_the_reported_count():
    spikes = np.zeros(256, dtype=np.float32)
    spikes[3 * TILE_WIDTH] = 1.0
    active = active_tiles(spikes)
    assert active_fraction(len(active), 8) == pytest.approx(1 / 8)


# --- tile packing must agree with active-tile indexing ---------------------
# The reader addresses weight tile m * Kt + t with the t it gets from active_tiles, so the
# packing order and the spike-tile layout have to agree on what tile t means.


def test_tile_counts_agree_with_the_broadcast_layout():
    synapse = SynapseConfig(fan_in=784, fan_out=256)
    _, kt = tile_counts(synapse)
    broadcast = broadcast_spike_tiles(np.zeros(784, dtype=np.float32))
    assert broadcast.shape == (1, kt, 32, 32)


def test_packed_weights_have_one_tile_per_output_and_input_block():
    weights = np.arange(256 * 784, dtype=np.float32).reshape(256, 784)
    packed = pack_weight_tiles(weights)
    mt, kt = tile_counts(SynapseConfig(fan_in=784, fan_out=256))
    assert packed.shape == (1, mt * kt, 32, 32)
    # Tile m * Kt + t must hold W[32m : 32m+32, 32t : 32t+32].
    assert packed[0, 0 * kt + 0, 0, 0] == pytest.approx(weights[0, 0])
    assert packed[0, 0 * kt + 1, 0, 0] == pytest.approx(weights[0, TILE_WIDTH])
    assert packed[0, 1 * kt + 2, 5, 7] == pytest.approx(weights[32 + 5, 2 * TILE_WIDTH + 7])


def test_packed_weights_reject_non_2d_input():
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free suite, no root conftest
        pack_weight_tiles(np.zeros(4, dtype=np.float32))


# --- moved from test_sparse_matvec.py, which is in the device half ----------------
# Both were host-side all along -- their own docstrings said so -- but that module imports ttnn at
# module scope for its other tests, so `collect_ignore` dropped these two with everything else and
# neither had ever run anywhere. The classification is by import, not by intent: a file that mixes
# host-side and device-side work loses the host half by construction. What these add over the
# active-tile tests already in this file is the density sweep, which pins the ACTIVITY_LEVELS the
# device tests are parameterised over -- if those constants drifted from the tile counts they
# claim, every device case would silently test the wrong density and still pass.

TILE = 32
FAN_IN = 256
N_INPUT_TILES = FAN_IN // TILE
# The density sweep, copied so the two halves cannot disagree about what "5% activity" means.
ACTIVITY_LEVELS = [0.0, 0.05, 0.5, 1.0]
# 13 spikes fill one input tile, 128 fill four, 256 fill all eight.
ACTIVE_TILES_BY_LEVEL = {0.0: 0, 0.05: 1, 0.5: 4, 1.0: 8}


def test_active_tiles_and_fraction_are_hand_computable():
    """Exact, and it needs no device: both functions are host-side indexing.

    A spike in exactly one of eight input tiles must select one tile, and the fetched share
    must be 1/8. If ``active_tiles`` grouped by something other than 32-wide input blocks --
    by output neuron, or by the broadcast tile shape -- the count would come out wrong while
    every device test depending on it quietly fetched the wrong pages.
    """
    spikes = np.zeros(FAN_IN, dtype=np.float32)
    spikes[3 * TILE : 4 * TILE] = 1.0

    active = active_tiles(spikes)
    assert active.tolist() == [3]
    assert active_fraction(len(active), N_INPUT_TILES) == 1.0 / 8.0

    silent = active_tiles(np.zeros(FAN_IN, dtype=np.float32))
    assert silent.size == 0
    assert active_fraction(len(silent), N_INPUT_TILES) == 0.0


@pytest.mark.parametrize("activity", ACTIVITY_LEVELS, ids=["0_percent", "5_percent", "50_percent", "100_percent"])
def test_activity_level_activates_the_expected_number_of_tiles(activity, spike_pattern):
    """Exact. Pins the density sweep to a known population.

    13 spikes fill one input tile, 128 fill four, 256 fill all eight. Checking the tile count
    on the host costs nothing and makes the density claim in each device test checkable, so a
    5% case that accidentally selected six tiles fails here rather than passing quietly.
    """
    rng = np.random.default_rng(213919)
    spikes = spike_pattern(rng, activity, FAN_IN, TILE)
    expected_tiles = ACTIVE_TILES_BY_LEVEL[activity]
    assert int(spikes.sum()) == round(activity * FAN_IN)

    active = active_tiles(spikes)
    assert len(active) == expected_tiles, f"{activity:.0%} activity selected {len(active)} tiles"
    assert active_fraction(len(active), N_INPUT_TILES) == pytest.approx(expected_tiles / 8.0)
