// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sparse reader for the synaptic mat-vec: fetches only the input tiles that carried a spike.
//
// The host passes a DRAM-resident list of the tile indices that are non-silent for this time
// step. The reader walks that list instead of the full fan-in, so a time step where a small
// fraction of the input population fired reads that fraction of the weight matrix. This is the
// only place sparsity is exploited: the hardware reads whole tiles, so tile granularity is the
// finest skip available.
//
// The active list is a flat uint32 array in a TILE_LAYOUT tensor, so its accessor page is one
// whole 4 KiB tile and the leading indices are contiguous inside page 0. The weight and spike
// buffers are also tile-layout. The list is staged into a reserved circular-buffer slot because
// the index has to be resident before it can address the tile reads that follow.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // TensorAccessorArgs<N> reads from compile-time argument N onward; see reader_spike_state.cpp.
    constexpr uint32_t accessor_offset = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_spike = get_compile_time_arg_val(2);
    constexpr uint32_t cb_index = get_compile_time_arg_val(3);

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t spike_addr = get_arg_val<uint32_t>(1);
    const uint32_t active_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_active = get_arg_val<uint32_t>(3);

    const uint32_t tile_size_bytes = get_tile_size(cb_weight);

    constexpr auto weight_args = TensorAccessorArgs<accessor_offset>();
    const auto weights = TensorAccessor(weight_args, weight_addr);
    constexpr auto spike_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto spikes = TensorAccessor(spike_args, spike_addr);
    constexpr auto active_args = TensorAccessorArgs<spike_args.next_compile_time_args_offset()>();
    const auto active = TensorAccessor(active_args, active_addr);

    Noc noc;
    CircularBuffer cb_weight_buf(cb_weight);
    CircularBuffer cb_spike_buf(cb_spike);
    CircularBuffer cb_index_buf(cb_index);

    // Claim the staging slot once, outside the loop. It is deliberately never pushed back: the
    // slot is scratch for this reader alone, and publishing it would advertise a queue nobody
    // reads. Reserving it a second time would deadlock, so this must not move into the loop.
    //
    // LocalTensorAccessor is the documented endpoint type for a NoC transaction against a plain
    // L1 region; it is what replaced the older "pin a circular buffer and use its pointer as a
    // raw L1 address" pattern that this code would otherwise hand-roll.
    //
    // The whole index list moves in one transfer from page 0 rather than striding page by page.
    // That is only sound because the host hands over a TILE_LAYOUT uint32 tensor, whose page
    // size is one whole 4 KiB tile, so the leading n_active * 4 bytes are contiguous inside
    // page 0. A row-major host tensor would have a 4-byte page and this would read the wrong
    // thing, so the two ends have to agree on the layout. `snn/layer.py` builds it with
    // `_tile_tensor`.
    //
    // The argument split follows noc_traits_t: the TensorAccessor source takes a page_id, while
    // the local-L1 destination takes only an offset, since it addresses a flat region.
    cb_index_buf.reserve_back(1);
    LocalTensorAccessor<uint32_t> index_scratch(cb_index_buf.get_write_ptr());
    noc.async_read(active, index_scratch, n_active * sizeof(uint32_t), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    for (uint32_t k = 0; k < n_active; k++) {
        // Reserve before reading, so a slow compute kernel back-pressures the NoC rather than
        // making the reader buffer tiles somewhere with nowhere to put them.
        cb_weight_buf.reserve_back(1);
        cb_spike_buf.reserve_back(1);

        const uint32_t tile_index = index_scratch[k];
        noc.async_read(
            weights, cb_weight_buf, tile_size_bytes, {.page_id = tile_index}, {.offset_bytes = 0});
        noc.async_read(spikes, cb_spike_buf, tile_size_bytes, {.page_id = tile_index}, {.offset_bytes = 0});

        noc.async_read_barrier();
        cb_weight_buf.push_back(1);
        cb_spike_buf.push_back(1);
    }
}
