// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Spike-driven synaptic mat-vec: out = W @ spikes, for one tile row of output neurons.
//
// The reduction is the ordinary blocked matmul from matmul_single_core, with the K loop driven
// by the *active* input tiles the reader chose rather than by every tile in the fan-in. A time
// step in which only 3% of input neurons fire therefore costs 3% of the weight traffic and 3%
// of the FPU work; the silence costs nothing at all.
//
// The tensor engine multiplies whole 32x32 blocks, so the one-wide spike vector is broadcast
// across all 32 columns of its tile and every output column holds the same dot product. Column
// 0 is the answer and the other 31 are redundant. That 32x waste is inherent to running a
// matrix-VECTOR product on a matrix-MATMULTIPLY engine, and it is stated rather than hidden:
// the sparsity win above and this arithmetic waste are independent, and the demo reports both.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

void kernel_main() {
    // Tile rows of output neurons, and how many *active* input tiles the reader will deliver.
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    // Output tile columns. A mat-vec has one logical column, but a tile is 32 wide regardless.
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr uint32_t cb_weight = get_compile_time_arg_val(3);
    constexpr uint32_t cb_spike = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);

    // SrcOrder::Reverse because in0 carries the weight block and in1 the spike block; the
    // natural operand order of matmul_tiles maps them the other way round.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_weight, cb_spike, cb_out);
    matmul_init(cb_weight, cb_spike);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Acquiring zeroes the destination register, which is the matmul's accumulator.
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_weight, 1);
                cb_wait_front(cb_spike, 1);
                // Also accumulates into the destination tile, so the K loop sums the blocks.
                matmul_tiles(cb_weight, cb_spike, 0, 0, 0);
                cb_pop_front(cb_weight, 1);
                cb_pop_front(cb_spike, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
