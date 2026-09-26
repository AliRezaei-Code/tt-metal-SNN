// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Receiver half of a one-to-one spike handoff. A peer core writes a single bfloat16 spike
// tile straight into this core's L1 at the slot reserved below; nothing is copied on
// arrival, so by the time cb_push_back runs the tile is already where the local compute
// kernel expects to find it.
//
// This file is deliberately a line-for-line twin of reader_spike_multicast.cpp: same five
// steps, same compile-time args, same runtime-arg order, same circular-buffer handling, same
// two mandatory invariants. A receiver is pairing-agnostic, and keeping it that way is the
// property that lets the same compiled kernel serve either kind of peer.
//
// THE ONE LINE THAT DIFFERS is not in this file. It is in the pairing's SENDER, and it is
// how the sender builds the destination address:
//
//   multicast sender:  get_noc_multicast_addr(x0, y0, x1, y1, l1_addr)   (rectangle)
//   unicast   sender:  get_noc_addr(x, y, l1_addr)                       (single core)
//
// with the matching data move `noc_async_write_multicast(...)` versus `noc_async_write(...)`
// and the matching flag publish `noc_semaphore_set_multicast(...)` versus
// `noc_semaphore_set(...)` on the arrival semaphore. Everything a receiver can observe is
// the same, because the semaphore protocol is defined at the receiver, not at the transport.
// A receiver that had to know which pairing it was in would be a latent bug: the moment a
// multicast sender is retargeted at a single core, or vice versa, the receiver would have to
// be rebuilt.
//
// There is no source-exclusion guarantee to lean on here either. lab3.rst:544-551 notes that
// noc_async_write_multicast and noc_semaphore_set_multicast exclude the core that issues them
// by default, so a multicast receiver can never be the multicast sender. A unicast sender
// names its single destination explicitly and offers no such protection, so for this pairing
// the host must guarantee sender != receiver.
//
// MANDATORY INVARIANT 1 -- the sender and the receiver must issue
// cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front in the SAME ORDER. The NoC
// addresses the destination from a single L1 offset that is assumed identical on both cores,
// and the only thing that makes those offsets identical is that the pointers have moved
// through the same sequence. Ordering, not timing, is what is synchronised; the semaphore
// handshake supplies the timing.
//
// MANDATORY INVARIANT 2 -- the data write and the semaphore write must use the SAME NoC
// instance. On some architectures they go into separate command-buffer FIFOs and are not
// issued in call order, so program-order issue alone would not imply program-order
// completion. Routing both over one NoC is what makes "the sender issued the tile write
// before the VALID" mean "the tile arrived before VALID". The receiver relies on this
// completely: it never checks the tile, it only trusts that seeing VALID implies the data is
// there. (The matching flush between the two sender-side operations is the sender's half of
// the same requirement.)
//
// A third constraint is the host's, not the kernel's: BOTH semaphores must be created on
// every participating core, so one semaphore id resolves to the same L1 address everywhere.
// That is why the receiver can build the sender's NoC address out of its own local semaphore
// address instead of being handed the sender's.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"

// Compile-time args, in order:
//   0  cb_spike_in   circular buffer the peer writes into (layout.CB_SPIKE_IN)
//   1  n_tiles       number of spike tiles the peer sends
//
// Runtime args, in order (identical to reader_spike_multicast.cpp, on purpose):
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
    // Identical to the multicast receiver: a receiver has exactly one sender, so it can only
    // ever unicast to it.
    const uint64_t receivers_ready_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, receivers_ready_sem_addr);

    // The sender waits for the readiness counter to reach the number of receivers, which is
    // one here. The increment is atomic but unordered relative to nothing, so the count -- not
    // arrival order -- is what the sender relies on.
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        WAYPOINT("SURW");

        // 1. Reserve first. The sender's destination is this core's write pointer, so the slot
        //    has to exist before readiness is announced.
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
    WAYPOINT("SURX");
}
