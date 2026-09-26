# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for the two-layer forward pass (``models/experimental/snn/snn/layer.py``).

``SparseNet`` is the deliverable shape from the brief: two fully connected LIF populations with
the first layer's spike train as the second layer's input, a sparse synaptic mat-vec in front of
each. Every op it runs is covered elsewhere in this package, so what is under test here is the
*wiring*:

  * the spike train crosses from layer 0 to layer 1 through the host once per step, and a
    layer that consumed the wrong vector still produces plausible-looking output;
  * the membrane of a layer feeds its own next step through DRAM, and a break in that hand-off
    is invisible to a single-step test of either op;
  * ``SparseLIFLayer.step`` skips the mat-vec entirely when the input population is silent and
    drives the LIF with zero current instead -- a different code path from the one
    ``test_sparse_matvec.py`` exercises.

Two shapes, 64 -> 32 -> 4 and 32 -> 16 -> 2. The second one puts the hidden and output
populations below one tile, so it covers the zero-padding path on both axes; the first spans
two input tiles. Six steps each. ttsim runs 10-50x slower than silicon and the whole suite
shares one 20-minute CI budget, so the shape list is short and every run is asserted to be
non-vacuous rather than assumed to be. The non-vacuity is deliberately asymmetric: the *hidden*
layer must fire on every step, while the output layer is only required to fire *somewhere in the
run*. On the 32 -> 16 -> 2 shape the output layer is legitimately silent on step 1 (it sees a
hidden population that has not yet built up), and requiring per-step output firing would fail a
correct run. The assertions below encode exactly that split.

Tolerances: both spike trains are compared **exactly**. The hidden membrane is compared at
``MEMBRANE_RTOL``, but only after the exact spike-train assertion has already agreed, so a
wrong spike position fails on the exact assertion first and a right-but-imprecise membrane is
the only thing that reaches the tolerant one.
"""

import numpy as np
import pytest
import ttnn

from models.experimental.snn.reference.lif import lif_step, sparse_matvec
from models.experimental.snn.snn.config import LIFConfig, SynapseConfig
from models.experimental.snn.snn.layer import SparseNet

SPIKING = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)

# rtol/atol for a membrane trace. The compute kernel is built with `fp32_dest_acc_en=True`
# and `MathFidelity.HiFi4`, so the membrane path is a float32 FPU multiply-accumulate and the
# only non-FPU step is `unary_gt_tile`, an exact comparison that never writes the membrane
# register. What is left is float32 rounding applied at most 6 times, on a contractive map
# (|alpha| = 0.951), to values of order 1 -- a couple of ulps, under 1e-6. 1e-4 is two orders
# of magnitude of headroom over that bound and is only ever reached once the exact spike-train
# assertion has already agreed.
MEMBRANE_RTOL = 1e-4
MEMBRANE_ATOL = 1e-4

STEPS = 6
SEED = 213919
# (fan_in, hidden, fan_out). 32 -> 16 -> 2 keeps both populations inside a single tile.
SHAPES = [(64, 32, 4), (32, 16, 2)]
SHAPE_IDS = ["64_32_4", "32_16_2"]

# A membrane seeded below threshold, so the leak is the only thing acting on it.
PRELOADED_MEMBRANE = 0.5


def _build_network(rng, fan_in, hidden, fan_out, device, grid_weights):
    """A ``SparseNet`` with reproducible weights and a 25%-active input spike vector."""
    weights = [grid_weights(rng, hidden, fan_in), grid_weights(rng, fan_out, hidden)]
    configs = [SynapseConfig(fan_in=fan_in, fan_out=hidden), SynapseConfig(fan_in=hidden, fan_out=fan_out)]
    return SparseNet(configs, SPIKING, weights, device), weights, (rng.random(fan_in) < 0.25).astype(np.float32)


def _reference_run(weights, input_spikes, config, n_steps):
    """The same two-layer forward pass in NumPy, from ``reference/lif.py`` alone.

    Layer 0's current is ``W0 @ input_spikes`` and layer 1's is ``W1 @ layer0_spikes``, and
    the external input is re-injected every step -- ``SparseNet.run`` rebinds the signal to
    ``input_spikes`` before each step rather than feeding layer 0 its own previous output.
    Getting that wrong chains the layers across time and produces a plausible-looking but
    entirely different trace, which is exactly the kind of bug this reference exists to catch.
    """
    signal = np.asarray(input_spikes, dtype=np.float32).reshape(-1)
    hidden_v = np.zeros(weights[0].shape[0], dtype=np.float32)
    out_v = np.zeros(weights[1].shape[0], dtype=np.float32)
    hidden_membrane, out_membrane, hidden_train, out_train = [], [], [], []

    for _ in range(n_steps):
        hidden_v, hidden_spikes = lif_step(hidden_v, sparse_matvec(weights[0], signal), config)
        out_v, out_spikes = lif_step(out_v, sparse_matvec(weights[1], hidden_spikes), config)
        hidden_membrane.append(hidden_v.copy())
        out_membrane.append(out_v.copy())
        hidden_train.append(hidden_spikes.copy())
        out_train.append(out_spikes.copy())

    return {
        "hidden_membrane": np.array(hidden_membrane),
        "out_membrane": np.array(out_membrane),
        "hidden_spikes": np.array(hidden_train),
        "out_spikes": np.array(out_train),
    }


def _hidden_membrane(layer, neurons, read_back):
    """The hidden layer's final membrane, out of the private buffer ``run`` leaves it in.

    ``SparseLIFLayer`` keeps the membrane in ``_v_mem`` with no public accessor, and ``step``
    ends by copying ``_v_out`` into ``_v_mem``, so after a run that buffer holds the state
    after the final step. The per-neuron vector lives in column 0 of each output tile, which
    is where ``SparseLIFLayer`` writes the input current and where ``synapses.extract_current``
    reads it back.
    """
    return read_back(layer._v_mem)[0, :, :, 0].reshape(-1)[:neurons].copy()


def _preload_membrane(layer, value, device, tile_tensor):
    """Seed a layer's membrane in DRAM, which ``SparseLIFLayer`` exposes no setter for."""

    block = np.zeros((1, layer.mt, 32, 32), dtype=np.float32)
    block[0, : layer.synapse.fan_out, 0] = value
    ttnn.copy(tile_tensor(block, ttnn.float32, device), layer._v_mem)


@pytest.mark.parametrize("fan_in,hidden,fan_out", SHAPES, ids=SHAPE_IDS)
def test_two_layer_forward_pass_matches_the_reference(
    device, reset_seeds, read_back, grid_weights, fan_in, hidden, fan_out
):
    """Exact on both spike trains, tolerant on the hidden membrane."""
    rng = np.random.default_rng(SEED)
    net, weights, input_spikes = _build_network(rng, fan_in, hidden, fan_out, device, grid_weights)

    hidden_train, out_train = net.run(input_spikes, STEPS)
    reference = _reference_run(weights, input_spikes, SPIKING, STEPS)

    # Non-vacuity first: a run in which nothing fires satisfies every `==` below for free.
    assert reference["hidden_spikes"].sum() > 0, "the hidden layer never fired"
    assert reference["out_spikes"].sum() > 0, "the output layer never fired"
    # Every step has to take the sparse path. A step whose hidden layer is silent takes the
    # zero-current shortcut in SparseLIFLayer.step instead, which is a different code path
    # and is covered by the two silence tests below.
    assert reference["hidden_spikes"].sum(axis=1).min() > 0, "some step had a silent hidden layer"
    assert (reference["hidden_spikes"].sum(axis=0) > 0).sum() > 1, "only one hidden neuron ever fired"

    assert np.array_equal(hidden_train, reference["hidden_spikes"]), (
        f"hidden spikes/step on device {hidden_train.sum(axis=1).astype(int).tolist()} "
        f"vs reference {reference['hidden_spikes'].sum(axis=1).astype(int).tolist()}"
    )
    assert np.array_equal(out_train, reference["out_spikes"]), (
        f"output spikes/step on device {out_train.sum(axis=1).astype(int).tolist()} "
        f"vs reference {reference['out_spikes'].sum(axis=1).astype(int).tolist()}"
    )

    membrane = _hidden_membrane(net.layers[0], hidden, read_back)
    expected = reference["hidden_membrane"][-1]
    assert np.allclose(
        membrane, expected, rtol=MEMBRANE_RTOL, atol=MEMBRANE_ATOL
    ), f"largest hidden membrane deviation {np.max(np.abs(membrane - expected)):.3e}"


@pytest.mark.parametrize("fan_in,hidden,fan_out", SHAPES, ids=SHAPE_IDS)
def test_silent_input_fires_nothing_anywhere(device, reset_seeds, grid_weights, fan_in, hidden, fan_out):
    """Exact. Also proves the zero-active-tile shortcut runs without tripping over itself.

    A silent input means the mat-vec is skipped and the LIF is driven with zero current, so
    no membrane can rise and nothing can fire. Beyond the spike count, this is the assertion
    that a network runs at all on an all-zero input rather than choking on an empty active
    list -- the one input shape the sparse reader never sees a non-zero page for.
    """
    rng = np.random.default_rng(SEED)
    net, _, _ = _build_network(rng, fan_in, hidden, fan_out, device, grid_weights)

    hidden_train, out_train = net.run(np.zeros(fan_in, dtype=np.float32), STEPS)

    assert np.array_equal(hidden_train, np.zeros((STEPS, hidden), dtype=np.float32))
    assert np.array_equal(out_train, np.zeros((STEPS, fan_out), dtype=np.float32))


def test_silent_input_still_leaks_a_preloaded_membrane(device, reset_seeds, tile_tensor, read_back, grid_weights):
    """Exact on the spikes, tolerant on the membrane, and the only way to see the leak.

    From a zero membrane, a silent input and a frozen buffer are indistinguishable: both leave
    it at 0. Seeding the membrane below threshold separates them. A correct run multiplies it
    by alpha; a layer that skipped the LIF entirely, or that treated zero current as "no
    update", leaves it at the seeded value. With ``PRELOADED_MEMBRANE = 0.5`` and
    ``alpha = exp(-1/20)``, a correct step lands on 0.4756 and a frozen one stays at 0.5 --
    a gap of 0.024, which is 2.4e2 times the 1e-4 tolerance below, so the two are not confusable.
    """
    fan_in, hidden, fan_out = 32, 16, 2
    rng = np.random.default_rng(SEED)
    net, _, _ = _build_network(rng, fan_in, hidden, fan_out, device, grid_weights)

    net.reset()
    _preload_membrane(net.layers[0], PRELOADED_MEMBRANE, device, tile_tensor)

    hidden_spikes = net.layers[0].step(np.zeros(fan_in, dtype=np.float32))
    net.layers[1].step(hidden_spikes)

    decayed = np.float32(PRELOADED_MEMBRANE) * np.float32(SPIKING.decay_factor)
    assert decayed < SPIKING.v_threshold, "the preload has to sit below threshold or this test proves nothing"
    assert np.array_equal(hidden_spikes, np.zeros(hidden, dtype=np.float32))

    membrane = _hidden_membrane(net.layers[0], hidden, read_back)
    assert np.allclose(
        membrane, np.full(hidden, decayed, dtype=np.float32), rtol=MEMBRANE_RTOL, atol=MEMBRANE_ATOL
    ), f"silent input did not leak the preloaded membrane: {membrane[0]} vs {decayed}"
    assert not np.allclose(
        membrane, np.full(hidden, PRELOADED_MEMBRANE, dtype=np.float32)
    ), "membrane was frozen: the zero-current shortcut skipped the LIF update entirely"
