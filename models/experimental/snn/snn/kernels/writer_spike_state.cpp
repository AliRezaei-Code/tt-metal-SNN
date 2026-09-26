// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the LIF neuron: drains the spike train and the post-reset membrane potential from
// the compute kernel's output circular buffers back to DRAM. The two streams are written by the
// same kernel on one NoC so that a spike and the state it produced land within one command
// barrier of each other.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // TensorAccessorArgs<N> reads its fields from compile-time argument N onward, so it has to
    // skip both the CB indices and the offset argument itself. The host passes the offset
    // rather than having this file hard-code a literal that would silently drift.
    constexpr uint32_t accessor_offset = get_compile_time_arg_val(0);
    constexpr uint32_t cb_spike = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v_out = get_compile_time_arg_val(2);

    const uint32_t spike_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    const uint32_t spike_tile_bytes = get_tile_size(cb_spike);
    const uint32_t v_out_tile_bytes = get_tile_size(cb_v_out);

    constexpr auto spike_args = TensorAccessorArgs<accessor_offset>();
    const auto spikes = TensorAccessor(spike_args, spike_addr);
    constexpr auto v_out_args = TensorAccessorArgs<spike_args.next_compile_time_args_offset()>();
    const auto v_out = TensorAccessor(v_out_args, v_out_addr);

    Noc noc;
    CircularBuffer cb_spike_buf(cb_spike);
    CircularBuffer cb_v_out_buf(cb_v_out);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_spike_buf.wait_front(1);
        cb_v_out_buf.wait_front(1);

        noc.async_write(cb_spike_buf, spikes, spike_tile_bytes, {}, {.page_id = i});
        noc.async_write(cb_v_out_buf, v_out, v_out_tile_bytes, {}, {.page_id = i});

        noc.async_write_barrier();
        cb_spike_buf.pop_front(1);
        cb_v_out_buf.pop_front(1);
    }
}
