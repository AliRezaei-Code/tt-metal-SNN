# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tile geometry and the circular-buffer map shared by every SNN kernel.

The device kernels cannot share a header with this module: TT-Metalium JIT-compiles each
kernel translation unit on its own, with no project include path. So the circular-buffer
indices live here and reach the kernels as compile-time arguments, which keeps a single
owner for the map instead of duplicating literals across every .cpp file.
"""

import numpy as np

# A tile is fixed by the hardware, not by this framework.
TILE_WIDTH = 32
TILE_HEIGHT = 32
TILE_ELEMENTS = TILE_WIDTH * TILE_HEIGHT

F32_TILE_BYTES = TILE_ELEMENTS * np.dtype(np.float32).itemsize
BF16_TILE_BYTES = TILE_ELEMENTS * np.dtype(np.float32).itemsize // 2
U32_TILE_BYTES = TILE_ELEMENTS * np.dtype(np.uint32).itemsize

# Two tiles per buffer so the producer can stage the next tile while the consumer drains the
# current one. Every queue below is double buffered.
CB_TILES = 2

# --- LIF neuron: reader -> compute -> writer ---------------------------------
CB_V_OLD = 0
CB_IN = 1
# Compute-internal hand-offs, and compute -> writer.
CB_V_NEW = 16
CB_SPIKE = 17
CB_RESET = 18
CB_V_OUT = 19

# --- Synaptic mat-vec: reader -> compute -> writer ----------------------------
CB_MV_WEIGHT = 2
CB_MV_SPIKE = 3
CB_MV_OUT = 22
# Staging for the active-tile index list. The reader claims one tile of it and never publishes,
# so the slot is private scratch rather than a queue.
CB_MV_INDEX = 23

# --- Phase 4: inter-core spike routing ----------------------------------------
# NoC-delivered spikes land in CB_SPIKE_IN, written by a peer core rather than this core's own
# reader, and leave through CB_SPIKE_OUT after a multicast.
CB_SPIKE_IN = 20
CB_SPIKE_OUT = 21

# The spike train is bfloat16 everywhere: a spike is exactly 0.0 or 1.0, so the narrower format
# is lossless, and in an SNN the spike train is the dominant traffic.
CB_DTYPE = {
    CB_V_OLD: "float32",
    CB_IN: "float32",
    CB_V_NEW: "float32",
    CB_SPIKE: "bfloat16",
    CB_RESET: "float32",
    CB_V_OUT: "float32",
    CB_MV_WEIGHT: "bfloat16",
    CB_MV_SPIKE: "bfloat16",
    CB_MV_OUT: "float32",
    CB_MV_INDEX: "uint32",
    CB_SPIKE_IN: "bfloat16",
    CB_SPIKE_OUT: "bfloat16",
}

_TILE_BYTES = {"float32": F32_TILE_BYTES, "bfloat16": BF16_TILE_BYTES, "uint32": U32_TILE_BYTES}

CB_PAGE_BYTES = {index: _TILE_BYTES[name] for index, name in CB_DTYPE.items()}
CB_TOTAL_BYTES = {index: CB_TILES * page for index, page in CB_PAGE_BYTES.items()}

# Compile-time argument tuples, in the exact order each kernel reads them. These are positional:
# reordering one is an ABI change for the .cpp that consumes it.
COMPUTE_CB_ARGS = (CB_V_OLD, CB_IN, CB_V_NEW, CB_SPIKE, CB_RESET, CB_V_OUT)
READER_CB_ARGS = (CB_V_OLD, CB_IN)
WRITER_CB_ARGS = (CB_SPIKE, CB_V_OUT)

MATVEC_READER_CB_ARGS = (CB_MV_WEIGHT, CB_MV_SPIKE, CB_MV_INDEX)
MATVEC_COMPUTE_CB_ARGS = (CB_MV_WEIGHT, CB_MV_SPIKE, CB_MV_OUT)
MATVEC_WRITER_CB_ARGS = (CB_MV_OUT,)

# One core's L1 is 128 KiB. Any pipeline whose buffers sum above this cannot launch.
L1_BYTES_PER_CORE = 128 * 1024


def l1_bytes(indices) -> int:
    """Bytes of L1 a set of circular buffers occupies."""
    return sum(CB_TOTAL_BYTES[i] for i in indices)


# The SFPU scalar arguments (binop_with_scalar.h, comp.h) are fp32 bit patterns, not values.
# Decoding them on the device would cost more than shipping them, and they only change when
# the neuron configuration does.
def fp32_bits(value: float) -> int:
    """Reinterpret a float as the uint32 the SFPU scalar arguments expect."""
    return int(np.float32(value).view(np.uint32))


def single_core_grid():
    """The one Tensix core a single-core SNN layer runs on."""
    import ttnn

    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
