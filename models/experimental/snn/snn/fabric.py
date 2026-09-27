# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inter-chip spike route: send a locally produced spike tile to a peer chip over TT-Fabric.

WHAT IS VERIFIED
    Only the host-side descriptor construction below. Every API this module calls was read
    out of this tree and every file:line is cited in the function that uses it. Nothing here
    has been executed.

WHAT IS NOT VERIFIED
    No device is attached to the machine this was written on, there is no second chip, and no
    fabric link has ever been opened. Nothing in this module, or in any kernel it describes,
    has observed: a packet leaving one chip, a packet arriving on another, the atomic
    increment landing on the remote semaphore, or any fabric bandwidth figure. A program
    returned by :func:`fabric_spike_program` running without hanging is the strongest claim
    that could be made even after a real run, and it is not the same as delivery being correct.

WHY THE WRITER KERNEL IS A PARAMETER
    The packet itself cannot be built from Python. ``WorkerToFabricEdmSender``,
    ``NocUnicastAtomicIncFusedCommandHeader`` and ``to_noc_fused_unicast_write_atomic_inc``
    are device-side C++ living under ``tt_metal/fabric/hw/inc/``, which the TT-Metalium JIT
    compiles into a kernel. They are not bound by ``ttnn/cpp/ttnn-nanobind/fabric.cpp``, which
    exposes only device-mesh-level configuration types (``MeshId``, ``FabricNodeId``,
    ``FabricRouterConfig``) plus the host-side connection helpers below. So
    :func:`fabric_spike_program` takes the path to the caller's writer kernel rather than
    inventing one, and returns the exact runtime-argument vector that kernel must consume.

    What that kernel has to do is pinned down here so the host and device sides cannot drift:

    1. Read ``dst_noc_x, dst_noc_y, dst_l1_addr, dst_sem_bank_addr, n_tiles`` (args 0..4).
    2. Build the sender over the *remaining* args, starting at index 5::

           connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<
               ProgrammableCoreType::TENSIX>(arg_idx);

       which is the call at ``models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp:201``.
    3. Take a header from ``PacketHeaderPool`` and arm it::

           header->to_noc_fused_unicast_write_atomic_inc(
               tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                   dst_noc_base, remote_sem_noc, 1, /*flush=*/false},
               payload_bytes);

       the four fields being ``{noc_address, semaphore_noc_address, val, flush}`` from
       ``tt_metal/fabric/fabric_edm_packet_header.hpp:297-306``.

    Carrying the destination L1 address and the destination semaphore address *in the packet*
    is the whole reason the receiver's arrival signal is free: the hardware writes the payload
    and then atomically increments the remote semaphore as one fused operation, so "the data
    arrived" and "the receiver was told" cannot come apart. The receiver on the far chip is
    the same five-step handshake as ``kernels/reader_spike_multicast.cpp``, waiting on that
    semaphore instead of on a NoC multicast of it.

    ``flush=false`` matches the all-reduce reference: the writer issues
    ``connection.send_current_slot_stateful_non_blocking(...)`` and the transport is flushed
    separately, so folding a flush into the header would stall on every packet.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from models.experimental.snn.snn.descriptors import (
    cb_descriptor,
    runtime_args,
)
from models.experimental.snn.snn.layout import (
    CB_SPIKE_OUT,
    L1_BYTES_PER_CORE,
    l1_bytes,
)

# The writer's own runtime-argument layout. These names are the contract with the device
# kernel; the indices are asserted in fabric_spike_program.
#
#   0  dst_noc_x          receiver core x in DEVICE coordinates
#   1  dst_noc_y          receiver core y in DEVICE coordinates
#   2  dst_l1_addr        L1 address on the receiver the payload lands at
#   3  dst_sem_bank_addr  L1 address of the receiver's arrival semaphore
#   4  n_tiles            spike tiles to send
#   5..                   fabric connection args, consumed by build_from_args<>()
FABRIC_SENDER_RT_ARGS = 5


def fabric_spike_program(
    *,
    writer_kernel_source: str,
    grid: ttnn.CoreRangeSet,
    src_node: ttnn.FabricNodeId,
    dst_node: ttnn.FabricNodeId,
    link_idx: int,
    dst_noc_x: int,
    dst_noc_y: int,
    dst_l1_addr: int,
    dst_sem_bank_addr: int,
    n_tiles: int,
    core_type: ttnn.CoreType = ttnn.CoreType.WORKER,
) -> ttnn.ProgramDescriptor:
    """Build the writer program that ships spike tiles to ``dst_node``.

    ``writer_kernel_source`` is a path to a ``.cpp`` that implements the three device-side
    steps in this module's docstring. It is required rather than defaulted because no such
    kernel ships with the framework yet, and a descriptor pointing at a file that does not
    exist would fail at JIT time with a message that says nothing about why.

    The fabric connection is set up *after* the program exists, because
    ``ttnn.setup_fabric_connection`` (bound at ``ttnn/cpp/ttnn-nanobind/fabric.cpp:150``)
    mutates the descriptor: it appends the worker-to-fabric mux ``SemaphoreDescriptor``s and
    returns the flat argument vector that ``build_from_args`` reads. This is the order the
    in-tree Python example uses at
    ``models/demos/deepseek_v3_b1/fused_ops/moe_routed_expert/op.py:2047-2070``.

    The kernel also needs the fabric preprocessor defines. The legacy C++ ``CreateKernel``
    path adds them itself (``tt_metal/impl/host_api/tt_metal.cpp:1529-1532``); the
    ``KernelDescriptor`` path does not, which is why
    ``ttnn.get_fabric_kernel_defines()`` exists and is applied here.

    ``dst_noc_x`` / ``dst_noc_y`` are the receiver core's *device* coordinates and
    ``dst_l1_addr`` / ``dst_sem_bank_addr`` are L1 offsets on the receiver -- the same
    convention ``kernels/reader_spike_multicast.cpp`` uses for its sender coordinates. The
    host is responsible for translating the mesh coordinate to a device coordinate before
    filling these in.
    """
    kernel_path = Path(writer_kernel_source)
    if not kernel_path.is_file():
        raise FileNotFoundError(
            f"fabric writer kernel {writer_kernel_source!r} does not exist; this module builds the host "
            "descriptor only, and the device-side WorkerToFabricEdmSender call has to live in a kernel"
        )
    if n_tiles <= 0:
        raise ValueError(f"n_tiles must be positive, got {n_tiles}")
    if link_idx < 0:
        raise ValueError(f"link_idx must be non-negative, got {link_idx}")

    # The next two checks are maintainer tripwires, not runtime ones: both operands are module
    # constants, so neither branch can be reached by any argument a caller passes. They are left
    # in place because editing CB_TOTAL_BYTES or the sender argument block without updating the
    # paired constant should fail loudly at build time. Neither is unit-tested -- a test would
    # assert that 21 * CB_TILES * page_bytes <= 128 KiB, which is a restatement of the constants
    # rather than a property of this function.
    total_l1 = l1_bytes((CB_SPIKE_OUT,))
    if total_l1 > L1_BYTES_PER_CORE:
        raise ValueError(f"CB_SPIKE_OUT needs {total_l1} B of L1, over the {L1_BYTES_PER_CORE} B one core has")

    # sender_args is a literal five-element list, so this compares a constant to a constant. Like
    # the L1 check above it is a tripwire against editing one without the other, and for the same
    # reason it is not unit-tested.
    sender_args = [dst_noc_x, dst_noc_y, dst_l1_addr, dst_sem_bank_addr, n_tiles]
    if len(sender_args) != FABRIC_SENDER_RT_ARGS:
        raise RuntimeError(f"sender argument block is {len(sender_args)}, expected {FABRIC_SENDER_RT_ARGS}")

    writer = ttnn.KernelDescriptor(
        kernel_source=str(kernel_path),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[CB_SPIKE_OUT],
        defines=ttnn.get_fabric_kernel_defines(),
        runtime_args=runtime_args(grid, lambda _core: sender_args),
        config=ttnn.WriterConfigDescriptor(),
    )

    cbs = [cb_descriptor(CB_SPIKE_OUT, grid)]
    program = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=cbs)

    # Appending through the per-core view mutates the kernel's argument vector in place and
    # adds the mux semaphores to program.semaphores.
    for core in ttnn.corerange_to_cores(grid):
        fabric_args = ttnn.setup_fabric_connection(src_node, dst_node, link_idx, program, core, core_type)
        program.kernels[0].runtime_args[core.x][core.y].extend(fabric_args)

    return program


def fabric_spike(
    spikes,
    *,
    writer_kernel_source: str,
    grid: ttnn.CoreRangeSet,
    src_node: ttnn.FabricNodeId,
    dst_node: ttnn.FabricNodeId,
    link_idx: int,
    dst_noc_x: int,
    dst_noc_y: int,
    dst_l1_addr: int,
    dst_sem_bank_addr: int,
    n_tiles: int,
    core_type: ttnn.CoreType = ttnn.CoreType.WORKER,
):
    """Ship ``n_tiles`` spike tiles from ``spikes`` to ``dst_node``.

    Unverified end to end -- see the module docstring. What this adds over
    :func:`fabric_spike_program` is only the dispatch.
    """
    program = fabric_spike_program(
        writer_kernel_source=writer_kernel_source,
        grid=grid,
        src_node=src_node,
        dst_node=dst_node,
        link_idx=link_idx,
        dst_noc_x=dst_noc_x,
        dst_noc_y=dst_noc_y,
        dst_l1_addr=dst_l1_addr,
        dst_sem_bank_addr=dst_sem_bank_addr,
        n_tiles=n_tiles,
        core_type=core_type,
    )
    return ttnn.generic_op([spikes], program)
