# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Intra-chip scaling: partition the neuron population across Tensix cores and run the
LIF pipeline on a grid instead of on ``(0, 0)``.

Two things make a multi-core SNN layer different from a single-core one, and both live
here rather than in the kernels.

**The partition.** ``partition_neurons`` splits the output neuron population -- expressed
in tiles, because that is the granularity the kernels walk -- across the compute grid with
``ttnn.split_work_to_cores``. That helper carves the *extra* work off the front of the core
order: when the tile count does not divide evenly, the first ``tiles % num_cores`` cores
each take one extra tile and the rest take the floor. Nothing is left over and no core is
idle, which is what the remainder distribution has to guarantee for a fixed-shape
time-stepping loop: every core must be programmed on every step, or the dispatch hangs.

**The per-core argument set.** A kernel created on a ``CoreRangeSet`` runs on *every* core
in that set, and a core with no runtime arguments is undefined behaviour, not a no-op. So
every descriptor here carries an argument entry for every core in the grid, including cores
that drew zero tiles: they get ``n_tiles = 0``, which makes every ``for`` loop in the
kernels a no-op without the dispatch ever seeing a missing argument.

Sharding precondition: the kernels index DRAM pages from 0, so the four tensors must
already be sharded across exactly this grid, one column-of-tiles of the neuron axis per
core. ``multicast_spike_program`` checks that when the binding exposes a shard spec.

The NoC half of a multi-core layer is not in this module. It is the sender/receiver pair
``kernels/reader_spike_multicast.cpp`` and ``kernels/reader_spike_unicast.cpp``, whose
handshake needs the two semaphores below to exist on every participating core with stable
ids; ``spike_handshake_semaphores`` allocates them so that the ids a peer passes to
``get_semaphore()`` resolve to the same L1 address here and there.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import ttnn

from models.experimental.snn.snn.config import LIFConfig
from models.experimental.snn.snn.descriptors import (
    runtime_args,
    cb_descriptor,
)
from models.experimental.snn.snn.layout import (
    CB_IN,
    CB_RESET,
    CB_SPIKE,
    CB_SPIKE_IN,
    CB_V_NEW,
    CB_V_OLD,
    CB_V_OUT,
    COMPUTE_CB_ARGS,
    L1_BYTES_PER_CORE,
    READER_CB_ARGS,
    TILE_WIDTH,
    WRITER_CB_ARGS,
    fp32_bits,
    l1_bytes,
)

KERNELS_DIR = Path(__file__).with_name("kernels")

# Semaphore initial values, matching tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp:13-14
# (constexpr INVALID = 0, VALID = 1). The device kernels read the same two values, so a
# mismatch here would deadlock rather than fail loudly.
SEM_INVALID = 0

# Id of the sender's readiness counter, and of the receiver's arrival flag. Both programs in
# a handoff -- this one and the peer's -- must agree on them, and both allocate on the full
# grid, so the ids are stable across the two descriptors.
SEM_RECEIVERS_READY = 0
SEM_TILE_SENT = 1


class CoreShard(NamedTuple):
    """One core's slice of the output neuron population.

    ``tile_start`` and ``tile_count`` are in tiles along the neuron (column) axis. A
    ``tile_count`` of 0 means the grid was larger than the work and this core is idle.
    """

    core: ttnn.CoreCoord
    tile_start: int
    tile_count: int


def compute_grid(num_cores: int) -> ttnn.CoreRangeSet:
    """A single-row ``CoreRangeSet`` of ``num_cores`` worker cores starting at ``(0, 0)``.

    One row, not a square: the neuron axis is one-dimensional, so a rectangular grid would
    only add a second enumeration order for the partition to disagree with.
    """
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    # Positional: the binding declares no nb::arg names for this overload and no default for
    # row_wise, so all three arguments are required.
    return ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(num_cores, 1), False)


def partition_neurons(fan_out: int, num_cores: int) -> tuple[ttnn.CoreRangeSet, list[CoreShard]]:
    """Split ``fan_out`` output neurons across ``num_cores`` cores.

    Returns the grid the split is valid on and one :class:`CoreShard` per core *in the
    grid*, ordered by the core enumeration ``split_work_to_cores`` used, so
    ``shards[i].tile_start + shards[i].tile_count == shards[i + 1].tile_start`` holds
    throughout and the ranges tile the population exactly.

    Remainder distribution: ``split_work_to_cores`` gives each of the first
    ``tiles % num_cores`` cores in its enumeration order one extra tile and every other core
    the floor. That ordering is why the per-core work is read off group membership rather
    than off position -- the helper's two groups are returned as ``CoreRangeSet``s, and
    testing ``core_group_1.contains(core)`` is exact regardless of how the cores are
    enumerated here.

    Cores beyond the work (``fan_out`` smaller than the grid) are returned with
    ``tile_count == 0``; they still need runtime arguments, because the program is created on
    the whole grid.
    """
    if fan_out <= 0:
        raise ValueError(f"fan_out must be positive, got {fan_out}")

    # The kernels walk whole 32-wide tiles, so the partition is in tiles. A population that
    # is not a multiple of TILE_WIDTH occupies a partially-filled last tile per shard.
    total_tiles = math.ceil(fan_out / TILE_WIDTH)

    grid = compute_grid(num_cores)
    num_used, all_cores, core_group_1, core_group_2, units_g1, units_g2 = ttnn.split_work_to_cores(
        grid, total_tiles, False
    )
    if num_used == 0:
        # split_work_to_cores returns an empty grid only for units_to_divide == 0, which
        # fan_out > 0 rules out. Guard anyway: an empty grid would silently produce a
        # program that computes nothing.
        raise ValueError(f"split_work_to_cores produced no cores for fan_out={fan_out}, num_cores={num_cores}")

    # CoreCoord is hashable and compares by value, so it keys the per-core counts directly.
    # The work is read off group membership rather than off position: the helper returns its
    # two groups as CoreRangeSets, and contains() is exact no matter which order the cores
    # are walked in here.
    per_core_tiles = {}
    for core in ttnn.corerange_to_cores(all_cores):
        per_core_tiles[core] = units_g1 if core_group_1.contains(core) else units_g2

    shards: list[CoreShard] = []
    cursor = 0
    for core in ttnn.corerange_to_cores(grid):
        count = per_core_tiles.get(core, 0)
        shards.append(CoreShard(core, cursor, count))
        cursor += count
    if cursor != total_tiles:
        raise RuntimeError(f"partition covered {cursor} of {total_tiles} tiles; split_work_to_cores disagrees")

    return grid, shards


def spike_handshake_semaphores(grid: ttnn.CoreRangeSet) -> list[ttnn.SemaphoreDescriptor]:
    """The two semaphores a spike handoff needs, allocated on every core of ``grid``.

    ``SemaphoreDescriptor`` is bound in ``ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1086``,
    so the handshake is expressible from Python and this returns real descriptors rather
    than a placeholder.

    Both semaphores are created on the *whole* grid, not just on the cores that touch them.
    That is the whole point: ``CreateSemaphore`` guarantees one id maps to the same L1 offset
    on every core it was created on, which is what lets a receiver build the sender's NoC
    address from its own local semaphore address instead of being handed the sender's. A
    receiver that never reads its own ``receivers_ready`` still has to allocate it.

    Initial values come from the lab 3 example program: the readiness counter starts at 0
    because no receiver is ready yet, and the arrival flag starts at ``INVALID`` (0) because
    no tile has been sent yet.
    """
    return [
        ttnn.SemaphoreDescriptor(SEM_RECEIVERS_READY, ttnn.CoreType.WORKER, grid, 0),
        ttnn.SemaphoreDescriptor(SEM_TILE_SENT, ttnn.CoreType.WORKER, grid, SEM_INVALID),
    ]


def _check_l1(indices) -> None:
    total = l1_bytes(indices)
    if total > L1_BYTES_PER_CORE:
        raise ValueError(f"circular buffers need {total} B of L1, over the {L1_BYTES_PER_CORE} B one core has")


def _check_sharded(tensor, grid: ttnn.CoreRangeSet, name: str) -> None:
    """The kernels index pages from 0, so a shard must already be laid out on this grid.

    The test is containment of *every core in the program grid*, not equality of core counts. A
    tensor sharded across 2x4 and a program running on 1x8 have the same eight cores, so a count
    comparison passes them -- and then core (3,1) of the program reads whichever shard sits at
    that position in the shard's own grid. Nothing raises; the layer simply computes the wrong
    neurons' membrane. A silent wrong answer is the worst shape for a precondition, so the
    containment test is the one worth paying for.
    """
    shard_spec = getattr(tensor, "shard_spec", None)
    if shard_spec is None:
        return
    shard_grid = shard_spec.grid
    missing = [core for core in ttnn.corerange_to_cores(grid) if not shard_grid.contains(core)]
    if missing:
        raise ValueError(
            f"{name} is sharded across a grid missing {len(missing)} of the {grid.num_cores()} cores "
            f"the program runs on, e.g. {missing[0]}; every core in the grid would read a shard "
            f"that is not there"
        )


def multicast_spike_program(
    v_mem,
    input_current,
    spikes,
    v_out,
    config: LIFConfig,
    grid: ttnn.CoreRangeSet,
    per_core_tiles: dict[ttnn.CoreCoord, int],
    semaphores: list[ttnn.SemaphoreDescriptor] | None = None,
):
    """Build the reader/compute/writer LIF program for a whole core grid.

    Identical to ``neuron.lif_neuron_program`` in every respect except the grid and the
    per-core tile counts: the same kernels, the same circular buffers, the same compute
    configuration. What changes is that ``n_tiles`` is no longer one number -- each core gets
    its own, taken from ``per_core_tiles``, defaulting to 0 so that an idle core in an
    oversized grid is still fully programmed.

    ``semaphores`` defaults to the two-semaphore spike handshake on ``grid``. The kernels
    built here do not touch them -- they are the DRAM-side LIF pipeline -- but allocating
    them here is what pins ids 0 and 1 to the same L1 offsets on this grid that a peer core's
    ``reader_spike_multicast.cpp`` will resolve, since both sides call ``get_semaphore()``
    with the same id and the host must have created that id on both grids.
    """
    _check_l1(COMPUTE_CB_ARGS)

    for name, tensor in (("v_mem", v_mem), ("input_current", input_current), ("spikes", spikes), ("v_out", v_out)):
        _check_sharded(tensor, grid, name)

    if semaphores is None:
        semaphores = spike_handshake_semaphores(grid)

    def tiles_for(core: ttnn.CoreCoord) -> int:
        return per_core_tiles.get(core, 0)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "reader_spike_state.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=list(READER_CB_ARGS)
        + ttnn.TensorAccessorArgs(v_mem).get_compile_time_args()
        + ttnn.TensorAccessorArgs(input_current).get_compile_time_args(),
        runtime_args=runtime_args(
            grid, lambda c: [v_mem.buffer_address(), input_current.buffer_address(), tiles_for(c)]
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
            lambda c: [
                fp32_bits(config.decay_factor),
                fp32_bits(config.v_threshold),
                fp32_bits(config.v_reset),
                tiles_for(c),
            ],
        ),
        # fp32_dest_acc_en keeps the membrane potential in float32 end to end, exactly as in
        # the single-core program; over a grid the accumulation order changes, not the
        # precision, because each core still accumulates only its own shard.
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / "writer_spike_state.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=list(WRITER_CB_ARGS)
        + ttnn.TensorAccessorArgs(spikes).get_compile_time_args()
        + ttnn.TensorAccessorArgs(v_out).get_compile_time_args(),
        runtime_args=runtime_args(grid, lambda c: [spikes.buffer_address(), v_out.buffer_address(), tiles_for(c)]),
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

    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=semaphores, cbs=cbs)


def multicast_spike(v_mem, input_current, spikes, v_out, config: LIFConfig, grid, per_core_tiles, semaphores=None):
    """Advance one LIF time step across ``grid``. ``spikes`` and ``v_out`` are updated in place."""
    program = multicast_spike_program(
        v_mem, input_current, spikes, v_out, config, grid, per_core_tiles, semaphores=semaphores
    )
    return ttnn.generic_op([v_mem, input_current, spikes, v_out], program)


def noc_spike_program(grid, n_tiles: int, sender_device_coords, *, multicast: bool = True, semaphores=None):
    """Wire the **receiving** half of the NoC spike handoff over ``grid``.

    .. warning::
       **The sending half is not implemented in this package.** The kernels named here,
       ``kernels/reader_spike_multicast.cpp`` and ``kernels/reader_spike_unicast.cpp``, are both
       receivers: they reserve a destination slot, clear ``tile_sent`` to ``INVALID``, increment
       the *sender's* readiness counter, and then block on
       ``noc_semaphore_wait(tile_sent, VALID)``. No kernel in this package issues the
       ``noc_async_write_multicast`` + ``noc_semaphore_set_multicast`` pair that would set that
       flag, and ``writer_spike_state.cpp`` / ``writer_matvec_out.cpp`` only write to DRAM. A grid
       programmed with this would announce readiness and then **block forever at step 4**. The
       receiver is complete and correct; the peer that feeds it does not exist yet.

    ``multicast_spike_program`` above runs the LIF pipeline on a grid; it does not hand spikes
    between cores. This is the piece that participates in the handoff, and it is what makes
    ``kernels/reader_spike_multicast.cpp`` and ``kernels/reader_spike_unicast.cpp`` reachable at
    all -- without it they are source files no program descriptor ever names.

    ``n_tiles`` is the number of spike tiles every receiver gets. It is a single scalar, not a
    per-core count, and that is a property of the protocol rather than a simplification: a
    multicast sender delivers the *same* tile set to every receiver, so the receivers cannot
    disagree. The kernels take it as a compile-time argument, so a grid whose cores received
    different counts would not fit this handshake and would need a different one.

    ``sender_device_coords`` maps each core to the peer it receives from, as ``(x, y)`` in DEVICE
    coordinates. The kernels address peers over the NoC, so these are the coordinates from
    ``mesh_device.worker_core_from_logical_core``, never logical ones.

    The four runtime arguments are in the order both kernel headers document, and both kernels
    read them with a running ``arg_idx`` rather than literals -- so the two stay interchangeable.
    The semaphores come from :func:`spike_handshake_semaphores`, which allocates both ids on every
    core of the grid: that is what makes one id resolve to the same L1 address everywhere, which
    is what lets a receiver build its sender's NoC address from its own local address.

    ``multicast`` selects the rectangle-multicast sender for a one-to-many grid; ``False`` picks
    the unicast one-to-one variant. The rest of the protocol is identical, which is why the two
    kernels are otherwise the same file.
    """
    if n_tiles < 0:
        raise ValueError(f"n_tiles must be non-negative, got {n_tiles}")

    semaphores = spike_handshake_semaphores(grid) if semaphores is None else semaphores
    kernel_name = "reader_spike_multicast.cpp" if multicast else "reader_spike_unicast.cpp"

    def args_for(core):
        try:
            sender_x, sender_y = sender_device_coords(core)
        except KeyError as exc:
            raise KeyError(f"no sender device coordinates for core {core}") from exc
        return [int(sender_x), int(sender_y), SEM_RECEIVERS_READY, SEM_TILE_SENT]

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS_DIR / kernel_name),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        # CB_SPIKE_IN, then n_tiles: both compile-time, because the receiver's K loop bound has
        # to be known when the JIT builds the kernel.
        compile_time_args=[CB_SPIKE_IN, n_tiles],
        runtime_args=runtime_args(grid, args_for),
        config=ttnn.ReaderConfigDescriptor(),
    )

    cbs = [cb_descriptor(CB_SPIKE_IN, grid)]
    _check_l1((CB_SPIKE_IN,))
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=list(semaphores), cbs=cbs)


def noc_spike(grid, n_tiles: int, sender_device_coords, *, multicast: bool = True, semaphores=None):
    """Build one NoC spike handoff program, for a caller to enqueue alongside its own.

    This wires receivers only; see :func:`noc_spike_program` for why the sending half is absent
    and what that means for a grid that uses it.
    """
    return noc_spike_program(grid, n_tiles, sender_device_coords, multicast=multicast, semaphores=semaphores)
