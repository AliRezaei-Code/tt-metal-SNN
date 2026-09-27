// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the LIF neuron: pulls the previous membrane potential and the input current from
// DRAM into the two circular buffers the compute kernel consumes. The page size of both DRAM
// buffers is the tile size, so page i is exactly tile i and a plain sequential walk is correct.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // TensorAccessorArgs<N> reads its fields from compile-time argument N onward, so it has to
    // skip both the CB indices and the offset argument itself. The host passes the offset
    // rather than having this file hard-code a literal that would silently drift.
    constexpr uint32_t accessor_offset = get_compile_time_arg_val(0);
    constexpr uint32_t cb_v_old = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);

    const uint32_t v_old_addr = get_arg_val<uint32_t>(0);
    const uint32_t in_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    const uint32_t tile_size_bytes = get_tile_size(cb_v_old);

    constexpr auto v_old_args = TensorAccessorArgs<accessor_offset>();
    const auto v_old = TensorAccessor(v_old_args, v_old_addr);
    constexpr auto in_args = TensorAccessorArgs<v_old_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_v_old_buf(cb_v_old);
    CircularBuffer cb_in_buf(cb_in);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_v_old_buf.reserve_back(1);
        cb_in_buf.reserve_back(1);

        noc.async_read(v_old, cb_v_old_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(input, cb_in_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();
        cb_v_old_buf.push_back(1);
        cb_in_buf.push_back(1);
    }
}
