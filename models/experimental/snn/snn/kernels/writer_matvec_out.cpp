// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the synaptic mat-vec: drains the per-tile-row current back to DRAM. Only column 0
// of each output tile carries a result (see compute_spike_matvec.cpp); the host slices it.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // TensorAccessorArgs<N> reads from compile-time argument N onward; see reader_spike_state.cpp.
    constexpr uint32_t accessor_offset = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<accessor_offset>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out_buf(cb_out);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_out_buf.wait_front(1);
        noc.async_write(cb_out_buf, out, tile_size_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb_out_buf.pop_front(1);
    }
}
