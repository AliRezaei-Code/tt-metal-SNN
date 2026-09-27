# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse LIF layers and the two-layer fully connected network.

One layer is two device programs in sequence:

  1. ``spike_matvec``  spikes of the previous population -> input current for this population
  2. ``lif_neuron``    (input current, previous membrane) -> (spikes, new membrane)

State lives on the device between time steps: the LIF writer kernel fills the membrane straight
into DRAM, and ``step`` then rotates that buffer into the reader's role for the next step, so
the membrane never round-trips through the host.

The synaptic current does. A step is two dispatches -- ``spike_matvec``, then ``lif_neuron`` --
and between them ``extract_current`` pulls the matvec result back, because that kernel writes
one 32x32 tile per output tile with the answer in column 0 (a GEMV on a GEMM engine broadcasts
the spike vector across all 32 columns). Slicing that column and repacking it into a padded
tile layout is a host pass today; keeping it on device would need a tile-repack kernel between
the two dispatches. The spike vector crosses back once per step as well, because the next
layer's active-tile list is chosen on the host.
"""

import numpy as np
import torch

import ttnn

from models.experimental.snn.snn.config import LIFConfig, SynapseConfig
from models.experimental.snn.snn.neuron import lif_neuron
from models.experimental.snn.snn.synapses import (
    active_fraction,
    active_tiles,
    broadcast_spike_tiles,
    extract_current,
    pack_weight_tiles,
    spike_matvec_program,
    tile_counts,
)

_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.uint32: torch.int32}


def _tile_tensor(host_array, dtype, device):
    """Host array -> tile-layout DRAM tensor of the requested ttnn dtype."""
    return ttnn.from_torch(
        torch.as_tensor(np.ascontiguousarray(host_array)).to(_TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class SparseLIFLayer:
    """One fully connected population: a sparse synapse onto a LIF neuron bank.

    ``weights`` is a dense ``(fan_out, fan_in)`` matrix on the host. It is packed into tiles once,
    at construction, because packing costs far more than a single time step's matmul and it never
    changes between steps.
    """

    def __init__(self, synapse: SynapseConfig, neuron: LIFConfig, weights: np.ndarray, device):
        self.synapse = synapse
        self.neuron = neuron
        self.device = device
        self.mt, self.kt = tile_counts(synapse)

        self._weights = _tile_tensor(pack_weight_tiles(weights), ttnn.bfloat16, device)
        self._zero_spike_block = np.zeros((1, self.kt, 32, 32), dtype=np.float32)
        self._zero_current = np.zeros((1, self.mt, 32, 32), dtype=np.float32)
        self._zero_state = np.zeros((1, self.mt, 32, 32), dtype=np.float32)
        self._index_scratch = np.zeros(self.kt * 32, dtype=np.uint32)

        self._spike_tiles = _tile_tensor(self._zero_spike_block, ttnn.bfloat16, device)
        self._active = _tile_tensor(self._index_scratch, ttnn.uint32, device)
        self._current = _tile_tensor(self._zero_current, ttnn.float32, device)
        self._v_mem = _tile_tensor(self._zero_state, ttnn.float32, device)
        self._spikes = _tile_tensor(self._zero_state, ttnn.bfloat16, device)
        self._v_out = _tile_tensor(self._zero_state, ttnn.float32, device)
        self._out = _tile_tensor(self._zero_state, ttnn.float32, device)

        # Running totals for the demo's cost table.
        self.tiles_fetched = 0
        self.time_steps = 0

    def reset(self):
        """Zero the membrane, the spikes and the fetched-tile counters, reusing existing buffers."""
        ttnn.copy_host_to_device_tensor(_tile_tensor(self._zero_state, ttnn.float32, self.device), self._v_mem)
        ttnn.copy_host_to_device_tensor(_tile_tensor(self._zero_state, ttnn.float32, self.device), self._v_out)
        ttnn.copy_host_to_device_tensor(_tile_tensor(self._zero_state, ttnn.bfloat16, self.device), self._spikes)
        self.tiles_fetched = 0
        self.time_steps = 0

    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """Advance one time step and return this layer's output spike vector."""
        active = active_tiles(input_spikes)
        n_active = int(len(active))

        ttnn.copy_host_to_device_tensor(
            _tile_tensor(broadcast_spike_tiles(input_spikes), ttnn.bfloat16, self.device), self._spike_tiles
        )
        self._index_scratch.fill(0)
        self._index_scratch[:n_active] = active
        ttnn.copy_host_to_device_tensor(_tile_tensor(self._index_scratch, ttnn.uint32, self.device), self._active)

        if n_active == 0:
            # A silent population delivers no current. The LIF kernel still runs, with zero
            # drive, so the membrane decays exactly as reference.lif says it should.
            ttnn.copy_host_to_device_tensor(_tile_tensor(self._zero_current, ttnn.float32, self.device), self._current)
        else:
            program = spike_matvec_program(self._weights, self._active, self._spike_tiles, self._out, self.mt, n_active)
            ttnn.generic_op([self._weights, self._active, self._spike_tiles, self._out], program)
            current = extract_current(ttnn.to_torch(self._out).numpy(), self.synapse.fan_out)
            padded = np.zeros((1, self.mt, 32, 32), dtype=np.float32)
            padded[0, : self.synapse.fan_out, 0] = current
            ttnn.copy_host_to_device_tensor(_tile_tensor(padded, ttnn.float32, self.device), self._current)

        lif_neuron(self._v_mem, self._current, self._spikes, self._v_out, self.neuron)
        # The writer left the new membrane in _v_out, so the two buffers swap roles rather than
        # copying: a per-timestep copy would dominate the step it is meant to measure, and
        # ttnn.copy only accepts two device tensors. _v_out is pure output -- the next LIF call
        # overwrites every element of it -- so rotating the handles is enough. The descriptor is
        # rebuilt each step and reads buffer_address() fresh, so it picks the swap up.
        self._v_mem, self._v_out = self._v_out, self._v_mem

        self.tiles_fetched += n_active
        self.time_steps += 1
        return self.spikes()[: self.synapse.fan_out]

    def run(self, input_spikes: np.ndarray, n_steps: int) -> np.ndarray:
        """Advance ``n_steps`` steps, returning the spike train ``(n_steps, fan_out)``."""
        train = np.zeros((n_steps, self.synapse.fan_out), dtype=np.float32)
        for step in range(n_steps):
            train[step] = self.step(input_spikes)
        return train

    def spikes(self) -> np.ndarray:
        """This layer's current output spikes as a host vector.

        The spike train is stored bfloat16, and torch.Tensor.numpy() rejects that dtype, so the
        cast happens here. It is exact: a spike is 0.0 or 1.0, both of which bfloat16 holds
        without rounding.
        """
        return ttnn.to_torch(self._spikes).to(torch.float32).numpy().reshape(-1)

    def sparsity_report(self) -> dict:
        """How much of the weight matrix the synapse actually touched, per time step."""
        total = self.kt * self.time_steps
        return {
            "steps": self.time_steps,
            "tiles_fetched": self.tiles_fetched,
            "total_tiles": total,
            "active_fraction": active_fraction(self.tiles_fetched, total),
        }


class SparseNet:
    """A stack of :class:`SparseLIFLayer` run for a fixed number of time steps.

    This is the deliverable shape from the brief: two fully connected layers with a spike train
    passed from one population to the next.
    """

    def __init__(self, layer_configs, neuron_config: LIFConfig, weights, device):
        if len(weights) != len(layer_configs):
            raise ValueError(f"got {len(weights)} weight matrices for {len(layer_configs)} layers")
        self.layers = [SparseLIFLayer(cfg, neuron_config, w, device) for cfg, w in zip(layer_configs, weights)]
        self.device = device

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def run(self, input_spikes: np.ndarray, n_steps: int) -> list:
        """Run ``n_steps`` steps of the whole stack, returning each layer's spike train.

        The encoded input is re-presented at the start of every time step. A rate-coded SNN holds
        the same input for the whole window; carrying the last layer's output forward would also
        feed the first layer a signal of the wrong width from step 1 onward.
        """
        self.reset()
        encoded = np.asarray(input_spikes, dtype=np.float32).reshape(-1)
        trains = [np.zeros((n_steps, layer.synapse.fan_out), dtype=np.float32) for layer in self.layers]
        for step in range(n_steps):
            signal = encoded
            for index, layer in enumerate(self.layers):
                signal = layer.step(signal)
                trains[index][step] = signal
        return trains

    def sparsity_report(self) -> list:
        return [layer.sparsity_report() for layer in self.layers]
