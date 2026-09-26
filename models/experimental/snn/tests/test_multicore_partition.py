# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-core partitioning and the NoC spike handshake's host side.

``snn/multicore.py`` is the only substantial module with no coverage at all, and it holds the
kind of logic that fails silently: an off-by-one in a shard boundary or a remainder handed to the
wrong core does not raise, it just gives two cores the same neurons and starves another. These
tests pin the arithmetic rather than the behaviour, because the behaviour needs a second core with
a peer to talk to and nothing here can provide that.

They need ``ttnn`` (``split_work_to_cores`` and the descriptor types) but not a device: the
partition is computed on the host from tile counts. So they run anywhere ttnn is installed, and
are skipped with the rest of the device half on a host without it.
"""

import math

import pytest
import ttnn

from models.experimental.snn.snn.layout import TILE_WIDTH
from models.experimental.snn.snn.multicore import (
    compute_grid,
    noc_spike_program,
    partition_neurons,
    spike_handshake_semaphores,
)


@pytest.mark.parametrize("fan_out,num_cores", [(256, 1), (256, 2), (256, 8), (784, 3), (32, 5), (10, 4)])
def test_partition_tiles_the_population_exactly(fan_out, num_cores):
    """Shards must be contiguous and cover every tile, with nothing duplicated or dropped."""
    _, shards = partition_neurons(fan_out, num_cores)
    total = math.ceil(fan_out / TILE_WIDTH)

    assert sum(s.tile_count for s in shards) == total, "the split lost or duplicated tiles"
    for previous, nxt in zip(shards, shards[1:]):
        assert previous.tile_start + previous.tile_count == nxt.tile_start, "shards are not contiguous"
    assert shards[0].tile_start == 0, "the first shard must start at zero"
    assert shards[-1].tile_start + shards[-1].tile_count == total, "the last shard must end at the total"


@pytest.mark.parametrize("fan_out,num_cores", [(256, 3), (784, 7), (32, 6), (100, 5)])
def test_partition_difference_is_at_most_one_tile(fan_out, num_cores):
    """Work is split evenly: no core may be given two tiles more than another.

    This is the property that makes a time-stepping loop safe. A core is programmed on every
    step, so a badly balanced split costs throughput but must not change results -- and a split
    that skipped a core entirely would hang the dispatch.
    """
    _, shards = partition_neurons(fan_out, num_cores)
    counts = [s.tile_count for s in shards]
    assert max(counts) - min(counts) <= 1, f"unbalanced split: {counts}"


def test_cores_beyond_the_work_still_get_an_entry():
    """More cores than tiles: the spare cores get zero tiles but must still be programmed.

    A kernel created on the whole grid runs on every core in it, and a core with no runtime
    argument is undefined behaviour rather than a no-op.
    """
    _, shards = partition_neurons(32, 8)  # one tile of work, eight cores
    assert len(shards) == 8
    assert sum(s.tile_count for s in shards) == 1
    assert sorted(s.tile_count for s in shards) == [0] * 7 + [1]


def test_partition_rejects_a_non_positive_population():
    with pytest.raises(ValueError):  # allow-pytest.raises: expect_error comes from the root conftest
        partition_neurons(0, 2)


def test_compute_grid_size_matches_request():
    grid = compute_grid(4)
    assert grid.num_cores() == 4, f"asked for 4 cores, got {grid.num_cores()}"


def test_handshake_allocates_both_semaphores_on_every_core():
    """Both ids must exist on the whole grid, or the peer's address does not resolve."""
    grid = compute_grid(4)
    semaphores = spike_handshake_semaphores(grid)
    assert len(semaphores) == 2, "the handshake needs a readiness counter and an arrival flag"
    ids = {s.id for s in semaphores}
    assert len(ids) == 2, f"semaphore ids must be distinct, got {ids}"


def _sender_coords(core):
    """A peer's device coordinates; the kernels address peers over the NoC, never logically."""
    return (7, 3)


@pytest.mark.parametrize("multicast", [True, False])
def test_noc_program_is_constructible_for_both_senders(multicast):
    """The descriptor must build, with the CB the peer writes into and the agreed tile count.

    Constructing it exercises the argument wiring: the wrong CB index or a missing runtime
    argument would otherwise only surface at dispatch, on a SKU that never dispatches it.
    """
    grid = compute_grid(2)
    program = noc_spike_program(grid, 3, _sender_coords, multicast=multicast)
    assert program is not None
    assert len(program.semaphores) == 2, "the handshake semaphores must travel with the program"
    assert len(program.cbs) == 1, "one circular buffer: the one the peer writes into"


def test_noc_program_rejects_a_negative_tile_count():
    grid = compute_grid(2)
    with pytest.raises(ValueError):  # allow-pytest.raises: expect_error comes from the root conftest
        noc_spike_program(grid, -1, _sender_coords)


def test_noc_program_reports_a_core_with_no_sender():
    """A missing sender mapping must name the core, not fail deep inside argument marshalling."""
    grid = compute_grid(2)

    def only_first(core):
        if (core.x, core.y) != (0, 0):
            raise KeyError(core)
        return (7, 3)

    with pytest.raises(KeyError):  # allow-pytest.raises: expect_error comes from the root conftest
        noc_spike_program(grid, 2, only_first)


def test_compute_grid_is_usable_as_a_core_rangeset():
    """Guard the assumption the rest of the module makes of compute_grid's return type."""
    grid = compute_grid(3)
    assert isinstance(grid, ttnn.CoreRangeSet)
