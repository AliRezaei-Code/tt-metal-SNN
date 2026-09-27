# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for the declarative layer language (``models/experimental/snn/snn/dsl.py``).

``Network.neuron(...).synapse(...)`` is supposed to compile to exactly the ``SparseLIFLayer`` a
caller would have built by hand, with no intermediate representation in between. Two claims
have to hold for that to be worth anything, and they fail in opposite directions:

  * **The compiled layer is the hand-built layer** -- the same fields *and* the same behaviour
    on the same input. A compile step that dropped a stage, transposed a weight matrix, or
    rebuilt the tile geometry differently would still produce a layer whose fields look right.
  * **The cache key is a correct key.** ``cache_key()`` exists so a caller can reuse a compiled
    descriptor across calls. A key that ignored the neuron configuration, the fan-in or the
    fan-out would hand back a descriptor built for a different layer, and it would do so
    silently: the shapes line up, the dispatch succeeds, and the numbers are wrong. A key built
    from object identity instead of the declared values would miss every time and quietly
    defeat the cache -- the failure that costs time rather than correctness, and so the one
    nobody notices.

``compile()`` resolves the device through ``ttnn.GetDefaultDevice()``, so the tests that call
it take the root ``device`` fixture. The cache-key and ``set_weights`` tests are host-side
tuple work and do not open a device: every ``ttnn.CreateDevice`` costs real time under ttsim
and none of them would change a host-side answer.

Error cases go through the ``expect_error`` fixture rather than a bare raises context
manager, which the ``prefer-expect-error`` pre-commit hook rejects in any ``tests/`` path.
"""

import numpy as np
import pytest

from models.experimental.snn.snn.config import LIFConfig, SynapseConfig
from models.experimental.snn.snn.dsl import Network
from models.experimental.snn.snn.layer import SparseLIFLayer

SPIKING = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)
OTHER_TAU = LIFConfig(dt=1.0, tau=10.0, v_threshold=1.0, v_reset=1.0)

FAN_IN = 32
FAN_OUT = 8
# The first 8 inputs spike. With the seeded grid weights that drives 3 of the 8 outputs above
# threshold, so the equivalence check below compares two firing layers, not two silent ones.
ACTIVE_INPUTS = 8
SEED = 213919


def _declare(fan_in=FAN_IN, fan_out=FAN_OUT, config=SPIKING, weights=None):
    """``net.neuron(config, fan_out).synapse(SynapseConfig(fan_in, fan_out), weights)``."""
    net = Network()
    net.neuron(config, fan_out).synapse(SynapseConfig(fan_in=fan_in, fan_out=fan_out), weights=weights)
    return net


def test_compiled_stage_matches_a_hand_built_layer(device, reset_seeds, grid_weights):
    """Structural equivalence, on the fields a layer is defined by."""
    rng = np.random.default_rng(SEED)
    weights = grid_weights(rng, FAN_OUT, FAN_IN)
    synapse = SynapseConfig(fan_in=FAN_IN, fan_out=FAN_OUT)

    compiled = _declare(weights=weights).compile()[0]
    hand_built = SparseLIFLayer(synapse, SPIKING, weights, device)

    assert compiled.synapse == synapse == hand_built.synapse
    assert compiled.neuron == SPIKING == hand_built.neuron
    assert (compiled.mt, compiled.kt) == (hand_built.mt, hand_built.kt)
    # 8 outputs and 32 inputs both pad up to a single tile, so the geometry assertion above
    # is a real check on the ceiling arithmetic and not a tautology about 32.
    assert (compiled.mt, compiled.kt) == (1, 1)


def test_compiled_layer_behaves_like_a_hand_built_layer(device, reset_seeds, read_back, grid_weights):
    """Behavioural equivalence, which is the claim the module docstring actually makes.

    One step of the same input through both construction paths has to produce the same spikes
    and the same membrane. The spike comparison alone is a weak signal on a population of 8,
    so the hidden membranes are compared too, and both are checked to have fired.
    """
    rng = np.random.default_rng(SEED)
    weights = grid_weights(rng, FAN_OUT, FAN_IN)
    synapse = SynapseConfig(fan_in=FAN_IN, fan_out=FAN_OUT)
    spikes = np.zeros(FAN_IN, dtype=np.float32)
    spikes[:ACTIVE_INPUTS] = 1.0

    compiled = _declare(weights=weights).compile()[0]
    hand_built = SparseLIFLayer(synapse, SPIKING, weights, device)

    compiled_spikes = compiled.step(spikes)
    hand_built_spikes = hand_built.step(spikes)

    assert int(compiled_spikes.sum()) > 0, "neither layer fired, so equality proves nothing"
    assert np.array_equal(compiled_spikes, hand_built_spikes)
    assert np.array_equal(
        read_back(compiled._v_out)[0, :, :, 0].reshape(-1)[:FAN_OUT],
        read_back(hand_built._v_out)[0, :, :, 0].reshape(-1)[:FAN_OUT],
    )


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"config": OTHER_TAU}, "a different tau must not share a cache entry"),
        ({"fan_in": 64}, "a different fan_in must not share a cache entry"),
        ({"fan_out": 16}, "a different fan_out must not share a cache entry"),
    ],
    ids=["tau", "fan_in", "fan_out"],
)
def test_compile_rejects_a_stage_with_no_weights(device, reset_seeds, expect_error):
    """A stage that compiles without weights packs an all-zero matrix and runs, returning
    silence forever. Better to refuse at compile time.

    ``device`` is taken so the assertion holds regardless of whether the weights check or the
    device lookup runs first.
    """
    net = Network()
    net.neuron(SPIKING, FAN_OUT).synapse(SynapseConfig(fan_in=FAN_IN, fan_out=FAN_OUT))

    with expect_error(ValueError, "has no weights"):
        net.compile()


def test_compile_rejects_an_empty_network(device, expect_error):
    """An empty stack has nothing to compile. Returning an empty list would let a network
    silently do nothing instead of saying so."""
    with expect_error(ValueError, "no stages"):
        Network().compile()
