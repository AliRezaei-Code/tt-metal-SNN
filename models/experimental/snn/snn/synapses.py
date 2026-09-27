# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spike-driven synaptic mat-vec: current = W @ spikes, gathered only where spikes occurred.

Layout
------
``weights`` is tile-layout ``[1, Mt * Kt, 32, 32]``. Tile ``m * Kt + t`` holds the 32x32 block
``W[32m : 32m+32, 32t : 32t+32]``, so input tile ``t`` addresses weight tile ``m * Kt + t``.

``spikes`` is tile-layout ``[1, Kt, 32, 32]``. Tile ``t`` holds the 32 input-neuron spike bits
for ``32t : 32t+32``, **broadcast across all 32 columns**. The tensor engine multiplies whole
32x32 blocks and cannot be told a column is redundant, so the one-wide spike vector is widened
to fill the block and the reader skips the block entirely when it is silent. The result is that
every output column of a computed tile holds the same dot product and only column 0 is kept.

Cost model
----------
A time step whose active input tiles are a fraction ``a`` of the fan-in moves ``a`` of the
weight bytes and spends ``a`` of the matmul's K iterations. The widening above costs a fixed
32x in arithmetic regardless of ``a``. ``active_fraction`` reports ``a`` so the demo can quote
the real trade instead of implying the sparsity is free.
"""

from pathlib import Path

import numpy as np

from models.experimental.snn.snn.config import SynapseConfig
from models.experimental.snn.snn.descriptors import (
    cb_descriptor,
    runtime_args,
    dm_compile_time_args,
)
from models.experimental.snn.snn.layout import (
    CB_MV_INDEX,
    CB_MV_OUT,
    CB_MV_SPIKE,
    CB_MV_WEIGHT,
    CB_TOTAL_BYTES,
    L1_BYTES_PER_CORE,
    MATVEC_COMPUTE_CB_ARGS,
    MATVEC_READER_CB_ARGS,
    MATVEC_WRITER_CB_ARGS,
    TILE_HEIGHT,
    TILE_WIDTH,
    single_core_grid,
)

KERNELS_DIR = Path(__file__).with_name("kernels")


def active_tiles(spike_bits: np.ndarray) -> np.ndarray:
    """Tile indices whose input block carried at least one spike, ascending.

    This is the whole point of the op: a silent block is never fetched, so a time step in which
    3% of the input *tiles* carry a spike costs 3% of the weight traffic instead of 100%.

    The unit matters, and the distinction is easy to state wrongly. A tile is ``TILE_WIDTH`` = 32
    neurons wide and is fetched if *any* of them fired, so what maps one-to-one onto bytes moved
    is the fraction of **tiles** holding a spike, not the fraction of **neurons** that fired. At
    low density the neuron rate badly understates the traffic: the demo's layer 0 fires 10.1% of
    its neurons and still fetches 60% of its weight tiles, because a 32-wide block nearly always
    holds at least one spike once one neuron in ten does. The saving is real, but it is a saving
    against the dense 100%, not a proportionality to the firing rate.

    Granularity is TILE_WIDTH neurons, not TILE_ELEMENTS. The reader fetches one *input tile*,
    which is a column of 32 neurons of the fan-in; 32x32 is the shape of a stored block, and
    grouping by it here would index past the end of any fan-in that is not a multiple of 1024.
    The granularity must match ``broadcast_spike_tiles`` and ``pack_weight_tiles`` exactly,
    because the reader uses these indices to address both.
    """
    flat = np.asarray(spike_bits, dtype=np.float32).reshape(-1)
    kt = -(-len(flat) // TILE_WIDTH)
    padded = np.zeros(kt * TILE_WIDTH, dtype=np.float32)
    padded[: len(flat)] = flat
    return np.nonzero(padded.reshape(kt, TILE_WIDTH).any(axis=1))[0].astype(np.uint32)


def tile_counts(synapse: SynapseConfig) -> tuple:
    """Tiles along each logical axis, after padding the neuron counts up to whole tiles."""
    mt = -(-synapse.fan_out // TILE_HEIGHT)
    kt = -(-synapse.fan_in // TILE_WIDTH)
    return mt, kt


def active_fraction(fetched: int, total_tiles: int) -> float:
    """Share of input tiles actually fetched, for the demo's cost table.

    Takes a *count* rather than the tile array, because two of the three callers only ever have a
    running count: ``SparseLIFLayer`` and the CPU baseline accumulate across time steps and never
    hold the list. A caller that does have an array passes ``len(active)``. All three used to
    compute this division themselves, and the two that were untested are now on the one path that
    is.
    """
    if total_tiles == 0:
        return 0.0
    return float(fetched) / float(total_tiles)


def broadcast_spike_tiles(spike_bits: np.ndarray) -> np.ndarray:
    """Pad the spike vector to whole tiles and replicate it across every column of each tile."""
    flat = np.asarray(spike_bits, dtype=np.float32).reshape(-1)
    kt = -(-len(flat) // TILE_WIDTH)
    padded = np.zeros(kt * TILE_WIDTH, dtype=np.float32)
    padded[: len(flat)] = flat
    tiles = padded.reshape(kt, TILE_WIDTH)
    return np.repeat(tiles[:, :, None], TILE_WIDTH, axis=2).reshape(1, kt, TILE_HEIGHT, TILE_WIDTH)


def pack_weight_tiles(weights: np.ndarray) -> np.ndarray:
    """Pack a dense ``(fan_out, fan_in)`` matrix into the ``[1, Mt*Kt, 32, 32]`` tile stack.

    Zero padding is safe: a padded input neuron has a zero spike, so a padded weight column
    contributes nothing to the dot product, and a padded output row is discarded on read.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D (fan_out, fan_in), got shape {w.shape}")
    mt, kt = -(-w.shape[0] // TILE_HEIGHT), -(-w.shape[1] // TILE_WIDTH)
    padded = np.zeros((mt * TILE_HEIGHT, kt * TILE_WIDTH), dtype=np.float32)
    padded[: w.shape[0], : w.shape[1]] = w
    blocks = padded.reshape(mt, TILE_HEIGHT, kt, TILE_WIDTH)
    # Tile order is m-major then t, matching the m * Kt + t addressing the reader computes.
    return np.ascontiguousarray(blocks.transpose(0, 2, 1, 3)).reshape(1, mt * kt, TILE_HEIGHT, TILE_WIDTH)


def extract_current(out_tiles: np.ndarray, fan_out: int) -> np.ndarray:
    """Pull column 0 of each output tile row back out to a ``(fan_out,)`` current vector."""
    mt = out_tiles.shape[1]
    blocks = np.asarray(out_tiles, dtype=np.float32).reshape(mt, TILE_HEIGHT, TILE_WIDTH)
    return blocks[:, :, 0].reshape(-1)[:fan_out].copy()


def spike_matvec_program(weights, active, spikes_tiles, out, mt: int, n_active: int):
    """Build the sparse reader / matmul / writer program for one synaptic time step.

    ``weights``, ``active`` and ``spikes_tiles`` are read-only; ``out`` is written. The caller
    owns the device tensors, because the active-tile list has to be recomputed on the host for
    every time step and the weight packing is expensive enough to be worth caching.
    """
    import ttnn

    grid = single_core_grid()

    l1_bytes = sum(CB_TOTAL_BYTES[i] for i in (*MATVEC_READER_CB_ARGS, *MATVEC_COMPUTE_CB_ARGS))
    if l1_bytes > L1_BYTES_PER_CORE:
        raise ValueError(f"mat-vec circular buffers need {l1_bytes} B of L1, over {L1_BYTES_PER_CORE} B")

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "reader_sparse_weights.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=dm_compile_time_args(MATVEC_READER_CB_ARGS, weights, spikes_tiles, active),
        runtime_args=runtime_args(
            grid,
            lambda _core: [weights.buffer_address(), spikes_tiles.buffer_address(), active.buffer_address(), n_active],
        ),
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "compute_spike_matvec.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        # Mt, Kt, Nt, then the three circular buffers. The values are not the obvious ones:
        #   Mt    output tiles for this layer.
        #   Kt    n_active, NOT the full fan-in -- the reader gathers only the input tiles
        #         that carried a spike, so the gathered operand's K extent is the active
        #         count. A step where 3% of the input fired therefore does a 3%-width matmul,
        #         which is the entire point of the sparse path.
        #   Nt    1, a literal: the kernel writes one 32x32 tile per output tile with the
        #         answer in column 0 -- a GEMV on a GEMM engine broadcasts the spike vector
        #         across all 32 columns -- so the useful data is a strided column.
        # No runtime args at all: every operand the compute kernel touches is a circular
        # buffer, so the empty list below is by design rather than by omission.
        compile_time_args=[mt, n_active, 1, *MATVEC_COMPUTE_CB_ARGS],
        runtime_args=runtime_args(grid, lambda _core: []),
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "writer_matvec_out.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=dm_compile_time_args(MATVEC_WRITER_CB_ARGS, out),
        runtime_args=runtime_args(grid, lambda _core: [out.buffer_address(), mt]),
        config=ttnn.WriterConfigDescriptor(),
    )

    cbs = [
        cb_descriptor(CB_MV_WEIGHT, grid),
        cb_descriptor(CB_MV_SPIKE, grid),
        cb_descriptor(CB_MV_INDEX, grid),
        cb_descriptor(CB_MV_OUT, grid),
    ]

    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)
