// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Leaky integrate-and-fire neuron update, hard reset by subtraction:
//
//   alpha = exp(-dt / tau)          (supplied by the host as an fp32 bit pattern)
//   v_new = alpha * v_old + I
//   s     = 1.0 if v_new > V_th else 0.0
//   v_out = v_new - s * V_reset
//
// alpha is computed on the host rather than with exp_tile: it is a hyperparameter that is
// constant for the whole simulation, and ttsim does not implement SFPLOADMACRO, so keeping the
// transcendental off the device costs nothing and keeps this kernel runnable in the simulator.
//
// The update runs as two register cycles because the FPU can only fold a circular-buffer tile
// into the destination register (EltwiseBinaryReuseDestType has no "both operands from DST").
// The spike * V_reset term is therefore packed to cb_reset and read back for the subtraction.
//
// Every value a cycle publishes lives in its own destination register. A single register is not
// enough: the spike and the spike * V_reset are derived from the same value, and both are
// needed downstream, so the scaled term goes to a third register via copy_dest_values rather
// than overwriting the spike after packing it.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    // Circular buffer indices come from the host so that the CB map has a single owner.
    constexpr uint32_t cb_v_old = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v_new = get_compile_time_arg_val(2);
    constexpr uint32_t cb_spike = get_compile_time_arg_val(3);
    constexpr uint32_t cb_reset = get_compile_time_arg_val(4);
    constexpr uint32_t cb_v_out = get_compile_time_arg_val(5);

    // SFPU scalars are fp32 bit patterns, not values: see binop_with_scalar.h and comp.h.
    const uint32_t alpha = get_arg_val<uint32_t>(0);
    const uint32_t v_threshold = get_arg_val<uint32_t>(1);
    const uint32_t v_reset = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t reg_v = 0;
    constexpr uint32_t reg_spike = 1;
    constexpr uint32_t reg_reset = 2;

    compute_kernel_hw_startup(cb_v_old, cb_in, cb_v_out);

    // One init per op family, hoisted out of the loop: the op sequence never changes.
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in);
    sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_reset);
    binop_with_scalar_tile_init();
    unary_gt_tile_init();
    copy_dest_values_init();

    for (uint32_t i = 0; i < n_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_v_old, 1);
        cb_wait_front(cb_in, 1);

        copy_tile(cb_v_old, 0, reg_v);
        mul_unary_tile(reg_v, alpha);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in, 0, reg_v);
        // reg_v now holds v_new; the subtraction for the reset is deferred to the next cycle.

        copy_dest_values(reg_v, reg_spike);
        unary_gt_tile(reg_spike, v_threshold);
        copy_dest_values(reg_spike, reg_reset);
        mul_unary_tile(reg_reset, v_reset);

        tile_regs_commit();
        tile_regs_wait();
        // HAZARD -- neither compute engine is reconfigured for the intermediates this kernel uses.
        //
        // compute_kernel_hw_startup(cb_v_old, cb_in, cb_v_out) configures UNPACK for two sources
        // and PACK for one destination, once, at the top. get_output_id() is the identity
        // function, so a packer output slot *is* a CB id, and both llk_pack and the unpack path
        // read per-CB format and face-geometry entries that only the *_hw_configure and
        // *_reconfig_data_format calls fill in. This kernel then uses:
        //
        //   UNPACK  cb_v_old (c_0), cb_in (c_1)   -- configured
        //           cb_v_new (c_16), cb_reset (c_18) -- NOT configured
        //   PACK    cb_v_out (c_19)               -- configured
        //           cb_v_new, cb_spike, cb_reset    -- NOT configured
        //
        // The likely failure is silent rather than a trap: are_packers_configured_correctly()
        // sits inside LLK_ASSERT_BLOCK and only fires in a sanitised build, so a ttsim run would
        // most likely compute with stale format registers and return wrong voltages with no
        // error. The fix is to reconfigure BOTH engines on every change of destination --
        // UNPACK(llk_unpack_hw_configure(new_icb)) paired with
        // PACK(llk_pack_reconfig_data_format(old_ocb, new_ocb)) -- as reconfigure_unary_bcast
        // does at api/compute/bcast.h:164-194. Neither engine's format survives a destination
        // change on its own. See the Phase 1 status section of models/experimental/snn/README.md.
        //
        // The calls themselves, and where they go. The reference implementation is
        // `reconfigure_unary_bcast` at api/compute/bcast.h:164-195, which switches a broadcast
        // operand and uses exactly these two entry points:
        //
        //     UNPACK((llk_unpack_hw_configure<is_fp32_dest_acc_en>(new_icb)));
        //     PACK((llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(old_ocb, new_ocb)));
        //
        // The packer overload is two-argument by design: it takes old and new and compares the two
        // formats on the way through. A one-argument call would not compile, which is a useful
        // property -- it means the switch cannot be written halfway.
        //
        // Placement, in this kernel specifically:
        //   * before the first `pack_tile` that targets a CB other than `cb_v_out`, pair the
        //     PACK reconfigure with the UNPACK one for the same tile, since this kernel's two
        //     engines move together within a cycle;
        //   * the UNPACK reconfigure belongs after `cb_wait_front` for the CB being read and
        //     before the read that uses it -- the same ordering `reconfigure_unary_bcast` relies
        //     on, where the reconfigure precedes the op that consumes the operand;
        //   * neither call goes between `tile_regs_acquire()` and `tile_regs_commit()`; the
        //     reconfigure writes engine configuration registers, and the docstring on
        //     `compute_kernel_hw_startup` is explicit that reprogramming them while an op is in
        //     flight is a data race.
        //
        // The alternative, and the one to prefer if the sequencing is not obvious: this kernel's
        // use of four intermediates exists only because the FPU cannot fold a circular-buffer tile
        // into the destination register. Two of the four -- cb_v_new and cb_reset -- exist to carry
        // a value from one register cycle to the next. Reducing the kernel to the two packer
        // outputs the in-tree eltwise_binary example uses removes every reconfigure from it, at
        // the cost of one extra host round trip per tile.
        //
        // NOT FIXED HERE ON PURPOSE: this kernel has never been compiled, and writing unpack/pack
        // register sequencing that has never executed is how the file reached this state.
        cb_reserve_back(cb_v_new, 1);
        pack_tile(reg_v, cb_v_new);
        cb_reserve_back(cb_spike, 1);
        pack_tile(reg_spike, cb_spike);
        cb_reserve_back(cb_reset, 1);
        pack_tile(reg_reset, cb_reset);

        cb_push_back(cb_v_new, 1);
        cb_push_back(cb_spike, 1);
        cb_push_back(cb_reset, 1);
        cb_pop_front(cb_v_old, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();

        tile_regs_acquire();
        cb_wait_front(cb_v_new, 1);
        cb_wait_front(cb_reset, 1);

        copy_tile(cb_v_new, 0, reg_v);
        sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_reset, 0, reg_v);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_v_out, 1);
        pack_tile(reg_v, cb_v_out);

        cb_push_back(cb_v_out, 1);
        cb_pop_front(cb_v_new, 1);
        cb_pop_front(cb_reset, 1);
        tile_regs_release();
    }
}
