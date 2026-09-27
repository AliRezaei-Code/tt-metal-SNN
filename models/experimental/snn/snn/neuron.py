# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Leaky integrate-and-fire neuron: program descriptor and op wrapper.

Three kernels run concurrently on one Tensix core, coordinated only by circular buffers:

  reader  DRAM(v_mem, I)      -> c_0, c_1
  compute c_0, c_1            -> c_16, c_17, c_18, c_19
  writer  c_17, c_19          -> DRAM(spikes, v_out)

There is no semaphore anywhere in this pipeline. The reader/compute/writer contract is that
every role issues ``cb_reserve_back`` / ``cb_push_back`` / ``cb_wait_front`` / ``cb_pop_front``
in the same order, so the circular buffer pointers stay address-identical on both ends; that,
not an explicit barrier, is the synchronisation.
"""

from pathlib import Path

import ttnn

from models.experimental.snn.snn.config import LIFConfig
from models.experimental.snn.snn.descriptors import (
    cb_descriptor,
    runtime_args,
    dm_compile_time_args,
)
from models.experimental.snn.snn.layout import (
    CB_IN,
    CB_RESET,
    CB_SPIKE,
    CB_V_NEW,
    CB_V_OLD,
    CB_V_OUT,
    COMPUTE_CB_ARGS,
    L1_BYTES_PER_CORE,
    READER_CB_ARGS,
    WRITER_CB_ARGS,
    fp32_bits,
    l1_bytes as layout_l1_bytes,
    single_core_grid,
)


KERNELS_DIR = Path(__file__).with_name("kernels")


# Compile-time argument layout of the data-movement kernels, in order:
#   [accessor_offset, *cb_indices, *TensorAccessor blocks...]
# The offset counts itself, which is why it is one more than the number of CB indices. Passing
# it keeps TensorAccessorArgs<N> in the .cpp pointed at the right slot instead of at the CB
# indices, and keeps the two in step when the CB list changes.


def lif_neuron_program(v_mem, input_current, spikes, v_out, config: LIFConfig):
    """Build the reader/compute/writer program for one LIF time step.

    ``v_mem``, ``input_current`` and ``v_out`` are float32 TILE_LAYOUT DRAM tensors;
    ``spikes`` is a bfloat16 TILE_LAYOUT DRAM tensor, because a spike is exactly 0.0 or 1.0
    and the spike train is the dominant traffic in an SNN.
    """
    grid = single_core_grid()
    n_tiles = v_mem.shape[1]

    # The union of the three roles' buffers, so the number tracks this program rather than the
    # whole module: Phase 4 adds circular buffers the LIF program never declares.
    l1_bytes = layout_l1_bytes((*READER_CB_ARGS, *COMPUTE_CB_ARGS, *WRITER_CB_ARGS))
    if l1_bytes > L1_BYTES_PER_CORE:
        raise ValueError(f"LIF circular buffers need {l1_bytes} B of L1, over the {L1_BYTES_PER_CORE} B a core has")

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "reader_spike_state.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=dm_compile_time_args(READER_CB_ARGS, v_mem, input_current),
        runtime_args=runtime_args(
            grid, lambda _core: [v_mem.buffer_address(), input_current.buffer_address(), n_tiles]
        ),
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "compute_lif_neuron.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=list(COMPUTE_CB_ARGS),
        runtime_args=runtime_args(
            grid,
            lambda _core: [
                fp32_bits(config.decay_factor),
                fp32_bits(config.v_threshold),
                fp32_bits(config.v_reset),
                n_tiles,
            ],
        ),
        # fp32_dest_acc_en keeps the membrane potential in float32 end to end. With it off, the
        # multiply-accumulate would round through bfloat16 and the threshold comparison would
        # drift over a long simulation.
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "writer_spike_state.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=dm_compile_time_args(WRITER_CB_ARGS, spikes, v_out),
        runtime_args=runtime_args(grid, lambda _core: [spikes.buffer_address(), v_out.buffer_address(), n_tiles]),
        config=ttnn.WriterConfigDescriptor(),
    )

    cbs = [
        cb_descriptor(CB_V_OLD, grid),
        cb_descriptor(CB_IN, grid),
        cb_descriptor(CB_V_NEW, grid),
        cb_descriptor(CB_SPIKE, grid),
        cb_descriptor(CB_RESET, grid),
        cb_descriptor(CB_V_OUT, grid),
    ]

    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)


def lif_neuron(v_mem, input_current, spikes, v_out, config: LIFConfig):
    """Advance one LIF time step. ``spikes`` and ``v_out`` are updated in place."""
    program = lif_neuron_program(v_mem, input_current, spikes, v_out, config)
    return ttnn.generic_op([v_mem, input_current, spikes, v_out], program)
