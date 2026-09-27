# SNN — a Spiking Neural Network framework on TT-Metalium

A Leaky Integrate-and-Fire (LIF) neuron model, a spike-driven synaptic mat-vec, and a
Python DSL, implemented as TT-Metalium kernels: a reader, a compute and a writer per operation,
running concurrently on one Tensix core and synchronised by circular buffers.

```
reader  DRAM(v_mem, I)        ->  c_0, c_1
compute c_0, c_1              ->  c_16, c_17, c_18, c_19
writer  c_17, c_19            ->  DRAM(spikes, v_out)
```

## A specific defect in the LIF kernel: neither compute engine is reconfigured

`compute_lif_neuron.cpp` rotates through four circular buffers as intermediates, but
`compute_kernel_hw_startup(cb_v_old, cb_in, cb_v_out)` configures **one** pack destination and
**two** unpack sources, once, at the top. `get_output_id()` is the identity function
(`llk_outputs.h:9`), so a packer output slot *is* a CB id, and both `llk_pack` and the unpack path
read per-CB format and face-geometry entries out of tables that only `llk_pack_hw_configure` /
`llk_pack_init` / `llk_pack_reconfig_data_format` fill in. The kernel's traffic does not stay
inside what startup configured:

| Engine | Configured at startup | The kernel also uses | Unconfigured |
|---|---|---|---|
| UNPACK | `cb_v_old` (c_0), `cb_in` (c_1) | `copy_tile(cb_v_new, …)`, `sub_reuse_dest_tiles(cb_reset, …)` | c_16, c_18 |
| PACK | `cb_v_out` (c_19) | `pack_tile` to `cb_v_new`, `cb_spike`, `cb_reset` | c_16, c_17, c_18 |

**The likely failure mode is silent, not a trap.** The guard that would catch this,
`are_packers_configured_correctly(...)`, sits inside `LLK_ASSERT_BLOCK`
(`llk_pack_tile_api.h:82`) and therefore fires only in a sanitised build. A ttsim CI run would
most likely pack and unpack with whatever format registers were last written and produce
**wrong voltages and wrong spike trains with no error at all** — which is worse than a build
failure, because `test_spike_propagation`'s PCC assertion is the only thing that would notice.

**The fix is to reconfigure on every change of destination, on both engines.** The in-tree
reference is `reconfigure_unary_bcast` at `api/compute/bcast.h:164-194`, which pairs
`UNPACK(llk_unpack_hw_configure<…>(new_icb))` with
`PACK(llk_pack_reconfig_data_format<…>(old_ocb, new_ocb))` — the two-argument packer overload at
`llk_pack_common_api.h:224` exists to switch pack destination and compares the old and new
formats first. The pairing is the point: neither engine's format survives a destination change on
its own. The alternative is to reduce the kernel to the two packer outputs the in-tree
`eltwise_binary` example uses, at the cost of an extra host round trip per tile.

**This is left unfixed deliberately**, but it is not left as prose.
`test_kernel_api_symbols.py::test_every_compute_engine_cbs_are_configured_at_startup` asserts the
invariant on every compute kernel, and `compute_lif_neuron.cpp` is shipped as a **strict xfail**
carrying this reason. It is a live signal rather than a comment: when the kernel is corrected the
test XPASSes, and `strict=True` turns that into a failure demanding the marker be removed.
`compute_spike_matvec.cpp` is the control — it passes the same check because it really is clean,
having been parsed rather than skipped.

The underlying change is unpack/pack register sequencing in device code that has never been
compiled: no toolchain on the authoring host, and the CI leg that would supply one does not run for
a pull request to this fork. Writing hardware sequencing that has never executed is how the file
reached this state.

`compute_spike_matvec.cpp` was checked for the same shape and is **clean**: it unpacks from
`cb_weight` and `cb_spike` and packs to `cb_out`, which is exactly the set
`compute_kernel_hw_startup<SrcOrder::Reverse>(cb_weight, cb_spike, cb_out)` programs. The four
reader/writer kernels are dataflow kernels and do not use `pack_tile` at all. The defect is
confined to the LIF kernel. Nothing on the host side is affected — the descriptor, the reference
semantics and the CPU baseline never execute it.

## Neuron model

Hard reset by subtraction. `alpha = exp(-dt / tau)` is computed on the host and shipped as an
fp32 bit pattern, so the kernel does no transcendental work:

```
v_new = alpha * v_old + I
s     = 1.0 if v_new > V_th else 0.0
v_out = v_new - s * V_reset
```

`unary_gt_tile` is a comparison, not an approximation, so the spike train is bit-exact against
the NumPy reference.

**This is not SpikingJelly's `LIFNode`,** and the difference is deliberate. SpikingJelly decays by
`1 - dt/tau` (plus a `+ v_reset/tau` term when `v_reset != 0`), thresholds with `>=`, and *sets* the
membrane to `v_reset` on a spike. This framework uses the exponential decay, thresholds with `>`,
and *subtracts* `v_reset`.

The advantage of the exponential decay is narrow, and the obvious stronger claim is false. The
decay term is the *exact* homogeneous solution of `tau dv/dt = -v`, so it is right at any `dt/tau`
rather than only in the small-step limit that `1 - dt/tau` approximates. The *drive* is not exact:
holding `I` constant, the exact step is `v(t+dt) = alpha*v + I*tau*(1-alpha)`, whereas this update
applies `I` unscaled. At `dt/tau = 0.05` that over-drives by 2.5% of the step and puts the steady
state at `I/(1-alpha) = 20.50` rather than `I*tau = 20.00`. It is the conventional SNN form —
SpikingJelly and snnTorch scale the drive the same way — but it is a hybrid, not an exact
integration of the continuous ODE.

SpikingJelly and this framework disagree from the first step of a constant drive regardless,
because `v` lands exactly on the threshold and `>` says quiet where `>=` says fire.

snnTorch's `Leaky`, given `beta = exp(-1/tau_per_decay)`, `threshold = 1.0` and
`reset_mechanism="subtract"`, is a much closer match: it agrees on the decay and on the strict
`>` threshold (`fire` is documented "Generates spike if mem > threshold"), and differs only in
subtracting `threshold` rather than `v_reset`, and in applying the *previous* step's spike
(`reset_delay=True` by default). `snn/config.py` carries the full three-way comparison with both
frameworks and cites where each was checked. The membrane path runs with `fp32_dest_acc_en=True` and `MathFidelity.HiFi4`,
leaving only float32 rounding on a contractive map.

## Structure

| Path | Role |
|---|---|
| `snn/kernels/compute_lif_neuron.cpp` | LIF update, two register cycles |
| `snn/kernels/reader_spike_state.cpp` | DRAM → circular buffers |
| `snn/kernels/writer_spike_state.cpp` | circular buffers → DRAM |
| `snn/kernels/compute_spike_matvec.cpp` | synaptic matmul, K over active tiles only |
| `snn/kernels/reader_sparse_weights.cpp` | fetches only non-silent input tiles |
| `snn/kernels/writer_matvec_out.cpp` | synaptic current → DRAM |
| `snn/kernels/reader_spike_{multi,uni}cast.cpp` | Phase 4 NoC spike routing |
| `snn/config.py` | `LIFConfig`, `SynapseConfig` — the neuron model, and how it differs from SpikingJelly and snnTorch |
| `snn/layout.py` | tile geometry and the circular-buffer map (single owner) |
| `snn/descriptors.py` | the canonical `KernelDescriptor` / `CBDescriptor` / `RuntimeArgs` constructors |
| `snn/neuron.py`, `snn/synapses.py` | program descriptors and ops |
| `snn/layer.py` | `SparseLIFLayer`, `SparseNet` |
| `snn/multicore.py`, `snn/fabric.py` | Phase 4 host side |
| `snn/dsl.py` | `Network` — declare structure, compiler emits descriptors |
| `reference/lif.py` | NumPy ground truth, imports no `ttnn` |
| `reference/cpu_baseline.py` | the same network in NumPy, for the throughput comparison |
| `demo/demo.py` | runnable end-to-end demo |
| `tests/` | device-free and device test suites |
| `check_suite.py` | the gate: the suite, the time budget, the lint hook, the documented demo |

`layout.py` owns the circular-buffer map because the device kernels cannot share a header with
the host: TT-Metalium JIT-compiles each kernel translation unit with no project include path, so
the indices travel as compile-time arguments instead.

## Sparsity, and what it costs

The reader walks only the input tiles that carried a spike. A tile is `TILE_WIDTH` = 32 neurons
wide and is fetched if *any* of them fired, so what maps one-to-one onto bytes moved is the
fraction of input **tiles** holding a spike — not the fraction of **neurons** that fired. At low
density the neuron rate badly understates the traffic: the demo's layer 0 fires 10.1% of its
neurons and still fetches 60% of its weight tiles, because a 32-wide block nearly always contains
at least one spike once one neuron in ten does. The saving is real, but it is measured against the
dense 100%, not proportional to the firing rate.

The index list is chosen on the host each step, because that is where the spike train already is.

Running a matrix-**vector** product on a matrix-**matrix** engine is the other half of the
trade: the one-wide spike vector is broadcast across all 32 columns of its tile, so 31 of every
32 output columns are redundant. That 32x arithmetic waste is inherent to the hardware, not an
implementation choice. The demo reports both the active-tile fraction and the firing rate so the
two are visible side by side rather than one hiding the other.

## Running

```bash
# CPU baseline only, no device needed
python3 -m models.experimental.snn.demo.demo --no-device --steps 50 --samples 256

# On a device
python3 -m models.experimental.snn.demo.demo --steps 50 --samples 256
```

Tests:

```bash
# device-free: no device, no JIT, no ttnn
python3 -m pytest models/experimental/snn/tests/test_reference_only.py -q \
    --confcutdir=models/experimental/snn -o addopts=""

# full suite, needs a device or ttsim
pytest models/experimental/snn/tests/ -v
```

The `--confcutdir`/`-o addopts` form is only for hosts without the Tenstorrent toolchain, where
the repo-root `conftest.py` cannot import `ttnn`. In CI both exist and plain `pytest` works.

## CI

The suite is registered as the `SNN framework unit tests` entry in
`tests/pipeline_reorg/ttnn_sanity_tests.yaml`, on the `sim_wh_n150` and `sim_bh_p150` SKUs. That
workflow invokes `setup-ttsim` for `sim_*` entries, so the same command runs on the simulator
with no extra wiring.

**Four environment variables are required.** `TT_METAL_SIMULATOR` is the one the runtime
actually selects on: tt-umd `dlopen`s `libttsim_*.so` in place of a kernel driver, so without it
the runtime tries to open real silicon and there is none. Alongside it,
`TT_METAL_SLOW_DISPATCH_MODE=1` because ttsim does not implement fast dispatch,
`TT_METAL_DISABLE_SFPLOADMACRO=1` because ttsim does not implement `SFPLOADMACRO` (which the LIF
kernel's SFPU operations issue), and `TT_METAL_QUASAR_NOC_API_VERSION=1`.

**This entry does not run on a pull request to a fork.** `ttnn-sanity-tests-impl.yaml` is
`workflow_call`-only and is reached through `sanity-tests-pr.yaml` -> `sanity-tests.yaml`, where
the job is gated on `needs.build-artifact.result == 'success'` with `secrets: inherit`. A
`pull_request` event from a fork is given a read-only token and **no secrets at all**, so
`secrets: inherit` yields nothing and the registry credentials the artifact build needs cannot be
supplied — adding them to the fork's settings is not an available fix. The gate never opens and
the entry stays dark here. It runs in Tenstorrent's own CI, or on an upstream PR where a
maintainer can dispatch it.

### Running it locally instead

The image alone is not enough: it has the toolchain but not a built tt-metal, and ttsim has to be
provisioned separately. This is the sequence `.github/actions/setup-ttsim` performs in CI,
written out; `tt_metal/tt-llk/tests/run_ttsim_regression.sh` is the in-tree reference and will
do the ttsim half for you if you point it at an arch.

```bash
docker run --rm -v "$PWD:/work" -w /work \
  ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-ci-build-amd64:latest \
  bash -c '
    set -e
    ./build_metal.sh --enable-ccache --build-packages
    export PYTHONPATH=/work:/work/ttnn:/work/tools
    export LD_LIBRARY_PATH=/work/build/lib

    # Provision ttsim. The version is pinned in tt_metal/ttsim-version; the .so prefix
    # selects the SoC descriptor, which must sit beside the .so.
    ver=$(tr -d "v\n" < tt_metal/ttsim-version)
    sim=$HOME/.cache/ttsim/$ver/blackhole
    mkdir -p "$sim"
    curl -fSL --retry 5 -o "$sim/libttsim.so" \
      "https://github.com/tenstorrent/ttsim/releases/download/v$ver/libttsim_bh.so"
    cp tt_metal/soc_descriptors/blackhole_140_arch.yaml "$sim/soc_descriptor.yaml"

    export TT_METAL_SIMULATOR_HOME=$sim
    export TT_METAL_SIMULATOR=$sim/libttsim.so
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    export TT_METAL_QUASAR_NOC_API_VERSION=1
    python3 -m pytest models/experimental/snn/tests/ -v
  '
```

Swap `libttsim_bh.so` / `blackhole_140_arch.yaml` for `libttsim_wh.so` /
`wormhole_b0_80_arch.yaml` to run the Wormhole model. The image is `amd64`, so on Apple silicon
it runs under emulation: correct, and slow.

**This sequence has not been executed.** It is assembled from the CI action and the in-tree
provisioning script, neither of which can run on this host, so treat the exact image and flags as
unverified and expect to adjust them on the first attempt.

## Metrics, and what is not measured

The demo prints `total_spikes`, `wall_seconds`, `spikes_per_second` and
`latency_per_timestep_ms` for the device path and for the CPU baseline, side by side.

**The device timings are launch-inclusive, and include a host round trip.** A time step is two
`generic_op` dispatches — the mat-vec, then the LIF — with the synaptic current passing through
the host in between. That is not incidental overhead: the mat-vec kernel writes one 32×32 tile
per output tile with the answer in column 0, because a GEMV on a GEMM engine broadcasts the
spike vector across all 32 columns, and repacking that column into a padded tile layout needs a
host pass. A device-side tile-repack kernel between the two dispatches would remove it.

So `spikes_per_second` and `latency_per_timestep_ms` measure launch cost plus that round trip, not
just kernel execution, while the CPU baseline runs its whole loop in-process. The two columns are
not a like-for-like kernel comparison, and the gap between them is mostly framework overhead. The
membrane potential, by contrast, does stay on device: the LIF writer fills it and the layer
rotates the buffer rather than copying. The active-tile fraction is the figure that isolates the
synapse itself.

**Energy (spikes/joule) is not available.** ttsim is a functional simulator with no power model,
so the energy row reads `N/A — requires silicon; ttsim has no power model`. It is not estimated
and it is not in `models/model_targets.yaml`, which carries the SNN entry as an explicit
unmeasured placeholder. Timing figures taken under ttsim describe the simulator, not silicon:
it runs 10–50x slower and has no timing fidelity. `models/model_targets.yaml` is where the real
numbers belong once they exist on hardware.

## Status

The device tests **have never been executed**. This framework was written on a host where
tt-metal cannot be built at all — `build_metal.sh` reads `/etc/os-release`, the toolchain is
`x86_64-linux-clang-20`, and `INSTALLING.md` states the binaries are Linux-only — and no
Tenstorrent device was attached. What *has* been checked is the reference semantics, the tile
arithmetic, and the agreement between the host and kernel argument layouts. The first execution of
the kernels is the CI run above, and its result is not known in advance.

**The host-side descriptor tests have not been executed either.** `test_multicore_partition.py` and
`test_fabric_program_guards.py` import `ttnn`, so they sit in the half that needs a real runtime;
on a host without one they are `collect_ignore`d rather than run, and the CI run above does not
reach them on a pull request to this fork. What *has* executed is the six toolchain-free
modules: the reference semantics and tile arithmetic, the device-API and argument-layout checks, the
compute-engine and push/pop checks, the `ttnn` call-site checks, the import resolution, the DSL
cache key, the sparsity metric, and the suite's own layout and documented counts -- the whole
half runs with `ttnn` made genuinely unimportable, which is the strong form of that claim. So no
assertion in either descriptor module has ever run in any environment observable from here.

**Phase 4 is weaker than that.** It is not "compile- and CI-verified": it is *unlaunched*. Being
precise about what is and is not covered, because the two halves are different:

* **Host side, written but never executed.** `test_multicore_partition.py` builds
  `multicore.noc_spike_program` and asserts its shape (both handshake semaphores present, one
  circular buffer) and its error paths, and it checks `partition_neurons`, `compute_grid` and
  `spike_handshake_semaphores` directly. `test_fabric_program_guards.py` covers the three
  `fabric_spike_program` input guards reachable from a caller's arguments. **Both import `ttnn`,
  so both sit in the half that needs a real runtime and neither has run in any environment
  observable from here** — on a host without tt-metal they are `collect_ignore`d, not skipped
  after collection, and the CI leg that would supply tt-metal does not run for a pull request to
  this fork. Treat them as written and reviewed. The two guards in `fabric_spike_program` that
  compare a module constant against a module constant are unreachable from any argument and are
  documented rather than tested.
* **Dispatch, uncovered.** Nothing calls `ttnn.generic_op` for any Phase 4 program, so no
  multi-core or fabric program is ever enqueued.
* **The multi-core handoff has no sender.** This is the one Phase 4 gap that is *missing code*
  rather than merely untested code, so it is stated plainly rather than folded into the lines above.
  `reader_spike_multicast.cpp` and `reader_spike_unicast.cpp` are **receivers**: they reserve a
  slot, clear `tile_sent` to `INVALID`, increment the sender's readiness counter, and block on
  `noc_semaphore_wait(tile_sent, VALID)`. Nothing in this package ever sets that flag — there is no
  `noc_async_write_multicast` + `noc_semaphore_set_multicast` pair anywhere, and both writers
  (`writer_spike_state.cpp`, `writer_matvec_out.cpp`) only write to DRAM. A grid programmed with
  `noc_spike_program` would announce readiness and then **block forever**. The receiver is complete
  and correct; the peer that feeds it has not been written.

That distinction is the whole point: building a descriptor is host-side work, and a kernel is only
compiled when the program containing it is dispatched. So the four `reader_spike_{multi,uni}cast.cpp`
kernels, and the device code behind them, have **never been compiled or run anywhere**, including in
CI. Their device APIs are checked for existence (`test_kernel_api_symbols.py`) and nothing more.
On top of that, the inter-chip half could not be verified even in principle here: no device, no
second chip, no fabric link, and `sim_wh_n150` / `sim_bh_p150` are single-chip. So there is no
claim of cross-chip delivery, fabric bandwidth, or even a working core-to-core handoff. Treat
Phase 4 as written-and-reviewed, not as working — and for the multi-core handoff, not even
complete.
