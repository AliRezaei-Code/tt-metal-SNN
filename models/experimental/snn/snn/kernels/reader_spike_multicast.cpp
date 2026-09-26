// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Receiver half of an intra-chip spike handoff. A peer core multicasts a bfloat16 spike
// tile straight into this core's L1 at the slot reserved below; nothing is copied on
// arrival, so by the time cb_push_back runs the tile is already where the local compute
// kernel expects to find it. This is the "push" half of the NoC: a receiver never reads,
// it only makes room and then says so.
//
// The five steps per tile are the ones in
// docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst, in order:
//
//   1. cb_reserve_back(1)                              make the destination slot exist
//   2. noc_semaphore_set(tile_sent, INVALID)           clear the previous arrival
//   3. noc_semaphore_inc(sender's receivers_ready, 1)  announce readiness (remote atomic)
//   4. noc_semaphore_wait(tile_sent, VALID)            block until the tile has landed
//   5. cb_push_back(1)                                 publish the tile to the consumer
//
// Step 1 precedes step 3 on purpose. The sender derives the multicast destination from its
// own circular-buffer pointer, which is only valid on this core if this core has already
// advanced its write pointer to the same slot. Announcing readiness before reserving would
// let the sender write into a slot that does not exist yet.
//
// MANDATORY INVARIANT 1 -- the sender and every receiver must issue
// cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front in the SAME ORDER. The
// NoC addresses the destination from a single L1 offset that is assumed identical on all
// participating cores, and the only thing that makes those offsets identical is that the
// pointers have moved through the same sequence. Ordering, not timing, is what is
// synchronised; the semaphore handshake supplies the timing.
//
// MANDATORY INVARIANT 2 -- the data multicast and the semaphore multicast must use the
// SAME NoC instance. On some architectures the two go into separate command-buffer FIFOs
// and are not issued in call order, so program-order issue alone would not imply
// program-order completion. Routing both over one NoC is what makes "the sender issued the
// tile write before the VALID" mean "the tile arrived before VALID". The receiver relies on
// this completely: it never checks the tile, it only trusts that seeing VALID implies the
// data is there. (The matching noc_async_writes_flushed() between the two sender-side
// operations is the sender's half of the same requirement.)
//
// A third constraint is the host's, not the kernel's: BOTH semaphores must be created on
// every participating core, so one semaphore id resolves to the same L1 address
// everywhere. That is why the receiver can build the sender's NoC address out of its own
// local semaphore address instead of being handed the sender's.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"

// Compile-time args, in order:
//   0  cb_spike_in   circular buffer the peer writes into (layout.CB_SPIKE_IN)
//   1  n_tiles       number of spike tiles the peer sends
//
// Runtime args, in order:
//   0  sender_noc_x          sender's x in DEVICE coordinates (lab3.rst: device kernels
//   1  sender_noc_y          address peers over the NoC, never logical coordinates)
//   2  receivers_ready_sem   semaphore id of the sender's readiness counter; this core
//                            allocates it too, purely to learn its L1 address
//   3  tile_sent_sem         semaphore id of the local arrival flag
void kernel_main() {
    constexpr uint32_t cb_spike_in = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);

    uint32_t arg_idx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receivers_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_sent_sem = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t receivers_ready_sem_addr = static_cast<uint32_t>(get_semaphore(receivers_ready_sem));
    const uint32_t tile_sent_sem_addr = static_cast<uint32_t>(get_semaphore(tile_sent_sem));

    volatile tt_l1_ptr uint32_t* const tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_sem_addr);

    // The sender's readiness counter, addressed over the NoC. Reusing the local L1 offset
    // is only sound because the host created this semaphore id on every participating core.
    const uint64_t receivers_ready_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, receivers_ready_sem_addr);

    // The sender waits for the readiness counter to reach the number of receivers. Each
    // increment is atomic but the receivers' increments are unordered relative to one
    // another, so the count -- not the arrival order -- is what the sender relies on.
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        WAYPOINT("SMRW");

        // 1. Reserve first. The sender's multicast destination is this core's write pointer,
        //    so the slot has to exist before readiness is announced.
        cb_reserve_back(cb_spike_in, 1);

        // 2. Clear the arrival flag left over from the previous tile, so the wait below can
        //    only be released by a send that happens after this point.
        noc_semaphore_set(tile_sent_sem_ptr, INVALID);

        // 3. Tell the sender a slot is available. This one really does cross the NoC: the
        //    semaphore lives in the sender's L1, not this core's.
        noc_semaphore_inc(receivers_ready_sem_noc_addr, 1);

        // 4. Block until the sender has both issued the tile write and set this flag to
        //    VALID. Local L1 read, no NoC traffic. Reaching VALID implies the tile has
        //    landed, by MANDATORY INVARIANT 2 above.
        noc_semaphore_wait(tile_sent_sem_ptr, VALID);

        // 5. The tile is already in the reserved slot -- the NoC wrote it there directly.
        //    Publishing it is the only thing left.
        cb_push_back(cb_spike_in, 1);
    }

    // The kernel exits holding n_tiles tiles it never popped. The consumer drains them with
    // cb_wait_front / cb_pop_front in its own order; see MANDATORY INVARIANT 1.
    WAYPOINT("SMRX");
}
