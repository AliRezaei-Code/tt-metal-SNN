# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The declarative layer language's cache key and weight validation, which need no device.

These six assertions were written as host-side work -- ``test_dsl.py``'s own docstring says so --
but they lived in a module that also contains four tests which do open a device, so the whole file
sat in the device half. On a host without tt-metal it was ``collect_ignore``d in its entirety and
none of this ran, and the CI leg that would have run it does not fire for a pull request to this
fork. The logic they cover is pure: ``Network.cache_key`` calls only ``signature`` and ``tuple``,
and ``set_weights`` only ``tuple`` and ``ValueError``.

That matters because of what a bad cache key costs. A key that ignored the neuron config, the
fan-in or the fan-out would hand back a descriptor built for a different layer -- the shapes line
up, the dispatch succeeds, and the numbers are wrong. A key built from object identity instead of
the declared values would miss every time and quietly defeat the cache, which costs time rather
than correctness and so is the failure nobody notices. Neither is observable from a passing
device test alone, which is why they are checked here.
"""

import numpy as np
import pytest

from models.experimental.snn.snn.config import LIFConfig, SynapseConfig
from models.experimental.snn.snn.dsl import Network


SPIKING = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)
OTHER_TAU = LIFConfig(dt=1.0, tau=10.0, v_threshold=1.0, v_reset=1.0)
FAN_IN = 32
FAN_OUT = 8
ACTIVE_INPUTS = 8
SEED = 213919


def _declare(fan_in=FAN_IN, fan_out=FAN_OUT, config=SPIKING, weights=None):
    """``net.neuron(config, fan_out).synapse(SynapseConfig(fan_in, fan_out), weights)``."""
    net = Network()
    net.neuron(config, fan_out).synapse(SynapseConfig(fan_in=fan_in, fan_out=fan_out), weights=weights)
    return net


@pytest.fixture
def kwargs():
    """The declaration axes a cache key has to distinguish, as (mutate, reason) pairs."""

    def weights_of(fan_in, fan_out):
        return np.zeros((fan_in, fan_out), dtype=np.float32)

    return [
        (lambda w: (32, 16, w), "fan-in"),
        (lambda w: (32, 4, w), "fan-out"),
        (lambda w: (32, 8, w), "unchanged fan-in and fan-out"),
        (lambda w: (32, 8, w + 1.0), "different weights"),
    ]


@pytest.fixture
def reason(request):
    return request.param


def test_cache_key_is_stable_for_identical_declarations(grid_weights):
    """The other half of the cache-key contract: same declaration, same key.

    The two networks are built from separate rng draws, so their weight arrays are different
    objects with different values. The keys still have to match, because the key is a function
    of the declaration, not of the arrays.
    """
    rng = np.random.default_rng(SEED)
    first = _declare(weights=grid_weights(rng, FAN_OUT, FAN_IN))
    second = _declare(weights=grid_weights(rng, FAN_OUT, FAN_IN))

    assert first.cache_key() == second.cache_key()
    assert first.cache_key()[0][3] == (FAN_OUT, FAN_IN)


def test_cache_key_ignores_weight_values(grid_weights):
    """Different values of the same shape must not move the key.

    If it did, every weight update in a training loop would be a cache miss and the cache
    would be dead weight -- correctness preserved, throughput gone.
    """
    rng = np.random.default_rng(SEED)
    baseline = _declare(weights=grid_weights(rng, FAN_OUT, FAN_IN)).cache_key()
    reweighted = _declare(weights=grid_weights(rng, FAN_OUT, FAN_IN)).cache_key()

    assert reweighted == baseline


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"fan_in": FAN_IN * 2}, "a different fan-in"),
        ({"fan_out": FAN_OUT * 2}, "a different fan-out"),
        ({"config": OTHER_TAU}, "a different neuron config"),
    ],
)
def test_cache_key_distinguishes_everything_the_descriptors_depend_on(grid_weights, kwargs, reason):
    """Each of these changes a compiled descriptor, so each has to change the key.

    The fixture form this used to take (``kwargs`` and ``reason``) was defined nowhere -- not in
    this module and not in the root conftest -- so the test could not have run in any environment.
    It lived in a module that is ``collect_ignore``d on a host without tt-metal, and the CI leg
    that would have surfaced the error does not fire for a pull request to this fork.
    """
    rng = np.random.default_rng(SEED)
    baseline = _declare(weights=grid_weights(rng, FAN_OUT, FAN_IN)).cache_key()
    fan_in = kwargs.get("fan_in", FAN_IN)
    fan_out = kwargs.get("fan_out", FAN_OUT)
    variant = _declare(weights=grid_weights(rng, fan_out, fan_in), **kwargs).cache_key()

    assert variant != baseline, reason


def test_set_weights_rejects_a_wrong_shaped_matrix(grid_weights):
    """The guard that stops a transposed weight matrix reaching the device.

    ``(fan_out, fan_in)`` and its transpose are the same bytes; only the declared shape tells
    them apart, and packing the wrong one silently produces a layer that runs and returns
    zeros forever.
    """
    rng = np.random.default_rng(SEED)
    net = _declare()

    with pytest.raises(ValueError, match="expects weights of shape"):  # allow-pytest.raises: root conftest
        net.set_weights(0, grid_weights(rng, FAN_IN, FAN_OUT))


def test_set_weights_accepts_the_declared_shape(grid_weights):
    """The positive case, so the test above cannot pass by ``set_weights`` rejecting
    everything."""
    rng = np.random.default_rng(SEED)
    net = _declare()
    weights = grid_weights(rng, FAN_OUT, FAN_IN)

    net.set_weights(0, weights)

    assert net.cache_key()[0][3] == (FAN_OUT, FAN_IN)


def test_stage_builder_rejects_a_fan_out_that_contradicts_its_synapse():
    """``neuron(config, fan_out)`` and ``synapse(SynapseConfig(fan_in, fan_out))`` both name
    the output width. Declaring them differently is a contradiction, and the fan-out is what
    decides the tile geometry, so neither can be silently preferred over the other."""
    net = Network()

    with pytest.raises(ValueError, match="declared fan_out"):  # allow-pytest.raises: root conftest
        net.neuron(SPIKING, FAN_OUT).synapse(SynapseConfig(fan_in=FAN_IN, fan_out=FAN_OUT * 2))
