# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A small declarative layer language that compiles to TT-Metalium program descriptors.

The shape mirrors TT-Lang: a user declares structure, and the compiler owns circular-buffer
layout, core mapping and descriptor construction. Nothing here introduces an intermediate
representation -- a network compiles straight to the same ``ProgramDescriptor`` objects that
``neuron.py`` and ``synapses.py`` build by hand, and ``test_dsl.py`` asserts the two agree.

    net = Network()
    net.neuron(LIFConfig(tau=20.0), fan=256).synapse(SynapseConfig(784, 256), weights=w1)
    net.neuron(LIFConfig(tau=10.0), fan=10).synapse(SynapseConfig(256, 10), weights=w2)
    program = net.compile()
"""

from models.experimental.snn.snn.config import LIFConfig, SynapseConfig


class _Node:
    """One declared stage of the network."""

    def __init__(self, neuron: LIFConfig, fan_in: int, fan_out: int):
        self.neuron = neuron
        self.synapse = SynapseConfig(fan_in=fan_in, fan_out=fan_out)
        self.weights = None
        self.name = None

    def signature(self) -> tuple:
        """Structural identity of this stage.

        Two networks with the same signature compile to interchangeable descriptors, so this is
        also the cache key. It deliberately excludes the weights themselves: recompiling for new
        weights is the caller's job via ``set_weights``, and keying on the array would make every
        training step a cache miss.
        """
        return (
            self.neuron,
            self.synapse.fan_in,
            self.synapse.fan_out,
            None if self.weights is None else self.weights.shape,
        )


class Network:
    """An ordered stack of LIF stages."""

    def __init__(self):
        self._stages: list = []

    def neuron(self, config: LIFConfig, fan_out: int) -> "_StageBuilder":
        """Declare a neuron population of ``fan_out`` neurons."""
        return _StageBuilder(self, config, fan_out)

    def stage(self, neuron: LIFConfig, fan_in: int, fan_out: int) -> _Node:
        node = _Node(neuron, fan_in, fan_out)
        self._stages.append(node)
        return node

    def set_weights(self, index: int, weights) -> None:
        """Attach or replace the weight matrix of stage ``index``."""
        node = self._stages[index]
        expected = (node.synapse.fan_out, node.synapse.fan_in)
        if tuple(weights.shape) != expected:
            raise ValueError(f"stage {index} expects weights of shape {expected}, got {tuple(weights.shape)}")
        node.weights = weights

    def compile(self) -> list:
        """Compile every stage to a ``SparseLIFLayer``.

        Returns the layers rather than raw descriptors: a layer is the unit that owns its device
        buffers and dispatches, which is what a caller actually needs to run a network.
        """
        from models.experimental.snn.snn.layer import SparseLIFLayer

        if not self._stages:
            raise ValueError("network has no stages; declare one with .neuron(config, fan_out)")
        layers = []
        for index, node in enumerate(self._stages):
            if node.weights is None:
                raise ValueError(f"stage {index} has no weights; call net.set_weights({index}, weights)")
            layers.append(SparseLIFLayer(node.synapse, node.neuron, node.weights, device=_require_device()))
        return layers

    def cache_key(self) -> tuple:
        """Signature of the whole stack, usable as a descriptor-cache key."""
        return tuple(node.signature() for node in self._stages)


class _StageBuilder:
    """The ``.neuron(...).synapse(...)`` half of a declaration, bound to a fan-in."""

    def __init__(self, network: Network, config: LIFConfig, fan_out: int):
        self._network = network
        self._config = config
        self._fan_out = fan_out

    def synapse(self, synapse: SynapseConfig, weights=None) -> _Node:
        """Close the stage with its incoming synapse and optional weight matrix."""
        node = self._network.stage(self._config, synapse.fan_in, self._fan_out)
        if node.synapse != synapse:
            raise ValueError(f"declared fan_out={self._fan_out} but synapse says {synapse}")
        if weights is not None:
            self._network.set_weights(len(self._network._stages) - 1, weights)
        return node


def _require_device():
    """The device a compile targets.

    Left explicit rather than reached for through a global default so that compiling a network
    says out loud which device it is for; a network built for one mesh cannot be silently
    launched on another.
    """
    import ttnn

    device = ttnn.GetDefaultDevice()
    if device is None:
        raise RuntimeError("no default device; open one with ttnn.open_device() before compiling")
    return device
