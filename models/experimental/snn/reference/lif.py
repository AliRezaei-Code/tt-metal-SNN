# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for the device kernels.

This module is deliberately free of any ``ttnn`` import: it is the ground truth the device
kernels are checked against, and it has to stay runnable on a machine with no Tenstorrent
toolchain, which is the only machine most contributors have.
"""

import numpy as np

from models.experimental.snn.snn.config import LIFConfig


def lif_step(v_mem: np.ndarray, input_current: np.ndarray, config: LIFConfig):
    """One LIF time step. Returns ``(v_out, spikes)``.

    Mirrors ``compute_lif_neuron.cpp`` exactly, including the hard reset by subtraction and
    the strict ``>`` threshold. A kernel that used ``>=`` would spike one step early on an
    exact-threshold input, which is the kind of off-by-one these references exist to catch.
    """
    v_new = config.decay_factor * v_mem + input_current
    spikes = (v_new > config.v_threshold).astype(np.float32)
    v_out = v_new - spikes * np.float32(config.v_reset)
    return v_out.astype(np.float32), spikes


def lif_run(v_mem: np.ndarray, input_current: np.ndarray, config: LIFConfig, n_steps: int):
    """Run ``n_steps`` time steps. Returns ``(membrane_trace, spike_train)``.

    Both outputs have a leading step axis: shape ``(n_steps,) + input_current.shape``.
    """
    v_mem = np.array(v_mem, dtype=np.float32, copy=True)
    current = np.asarray(input_current, dtype=np.float32)
    membrane_trace = np.empty((n_steps,) + v_mem.shape, dtype=np.float32)
    spike_train = np.empty((n_steps,) + v_mem.shape, dtype=np.float32)
    for step in range(n_steps):
        v_mem, spikes = lif_step(v_mem, current, config)
        membrane_trace[step] = v_mem
        spike_train[step] = spikes
    return membrane_trace, spike_train


def sparse_matvec(dense_weights: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    """Dense ground truth for the spike-driven synaptic mat-vec.

    ``dense_weights`` follows the fully-connected convention ``(fan_out, fan_in)``, so the
    layer computes ``W @ spikes`` and the incoming current is one value per output neuron.
    """
    spikes = np.asarray(spikes, dtype=np.float32).reshape(-1)
    weights = np.asarray(dense_weights, dtype=np.float32)
    return (weights @ spikes).astype(np.float32)
