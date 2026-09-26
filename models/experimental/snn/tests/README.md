# SNN framework tests (`tests/`)

Tests for the spiking-neural-network framework in `models/experimental/snn/`. One file per
device op, plus a suite that needs no device at all.

## What is covered

| File | Covers | Device? |
|---|---|---|
| `test_lif_neuron.py` | `lif_neuron` over 1, 2 and 4 tiles: exact spike train against `reference.lif.lif_run`, the reset branch, the strict threshold, and a spike buffer that is rewritten every step | yes |
| `test_sparse_matvec.py` | `spike_matvec_program` at 0% / 5% / 50% / 100% spike activity, a strict subset of input tiles, and the host-side active-tile indexing | yes, except two host-only cases |
| `test_spike_propagation.py` | Two-layer `SparseNet` forward pass at 64→32→4 and 32→16→2 against a NumPy reference, plus the silent-input shortcut | yes |
| `test_dsl.py` | `Network().neuron(...).synapse(...)` compiling to the same `SparseLIFLayer` a caller would build by hand, and `cache_key()` being a correct descriptor-cache key | partly |
| `test_dsl_cache_key.py` | The DSL's cache key and weight-shape guards, which need no device | no |
| `test_cb_map.py` | The CB size/dtype tables, and that every buffer any kernel names is in them | no |
| `test_reference_only.py` | The NumPy references and the config validation, with no ttnn import at all | no |
| `test_multicore_partition.py` | `partition_neurons` shard arithmetic, `compute_grid`, the handshake semaphores, and the shape of `noc_spike_program` | yes |
| `test_fabric_program_guards.py` | The three `fabric_spike_program` input guards a caller can actually reach | yes |
| `test_kernel_api_symbols.py` | Every device API the kernels call still exists, with the right arity and contiguous indices | no |
| `test_ttnn_call_sites.py` | Every `ttnn` binding the host modules call, its argument count, and its kind | no |
| `test_package_imports.py` | Every module parses, every intra-package import resolves, and no unresolvable name is imported | no |
| `test_suite_layout.py` | The test/toolchain split itself, against a frozen record of both halves | no |
| `conftest.py` | The `tile_tensor`, `read_back` and `grid_weights` fixtures every other file uses | — |

What each device test is actually there to catch:

- **`test_lif_neuron.py`** — a circular buffer sized or addressed for one tile only (caught by
  the 1/2/4-tile parameterisation, since every queue is two tiles deep and only the 4-tile
  case laps the reader and the compute kernel); a missing or unconditional reset subtraction;
  a threshold written as `>=` instead of `>`; a spike output that is never written or never
  cleared, which is what the quiet-then-loud schedule isolates.
- **`test_sparse_matvec.py`** — a reader that walks every page instead of the active list
  (the 0% case is the only assertion that can tell, and `out` is seeded with a non-zero
  sentinel so "exactly zero" cannot be confused with "the writer never ran"); a reader that
  ignores the index list and reads pages `0 .. n_active - 1` (the 3-and-5 case); tile
  arithmetic in the active-tile list.
- **`test_spike_propagation.py`** — the spike train crossing from one layer to the next
  through the host; the membrane hand-off through DRAM between steps; a layer that consumed
  the wrong vector; the zero-current shortcut when a population is silent.
- **`test_dsl.py`** — a cache key that ignores the neuron config, the fan-in or the fan-out,
  which would silently hand back a descriptor built for a different layer; a cache key built
  from object identity, which would defeat the cache; a compile step that transposes weights
  or rebuilds the tile geometry differently.
- **`test_multicore_partition.py`** — an off-by-one in a shard boundary, or a remainder handed to
  the wrong core, which would give two cores the same neurons and starve another without raising.
- **`test_fabric_program_guards.py`** — a missing `writer_kernel_source` slipping through, which
  would otherwise fail later at JIT time with a message naming no file.
- **`test_kernel_api_symbols.py`** — a device API renamed or its arity changed, which compiles
  nowhere and fails the whole run with the error pointing at the kernel, not the name.
- **`test_ttnn_call_sites.py`** — a host module calling a `ttnn` binding with the wrong number of
  arguments, or handing it a torch tensor where a `ttnn::Tensor` is declared.
- **`test_package_imports.py`** — a rename that left a dangling intra-package import, or a module
  that no longer parses, either invisible until something imports it.
- **`test_suite_layout.py`** — a device test dropped from `_DEVICE_TESTS`, or a toolchain-free one
  added to it, either of which silently changes which half of the suite runs. This is the one file
  here whose checks have been *demonstrated* to discriminate: they are written against a frozen
  record of both halves rather than against `_DEVICE_TESTS`, so deleting a module and renaming it
  out of the list turns the suite red instead of quietly shrinking it. (The first version of these
  guards lived in `conftest.py`, where pytest collects no test functions at all, so they had never
  run in any environment.)

## Running

**One command, correct in both places:**

```bash
pytest models/experimental/snn/tests/ -v
```

`tests/conftest.py` splits the suite on whether a tt-metal runtime is actually importable, so the
same invocation does the right thing in CI and on a bare host. In CI, where `ttnn` is a real
install, all twelve modules are collected. On a host without the toolchain, the six device modules
are ignored and the toolchain-free half runs:

```
test_cb_map.py                  6 tests   the CB size/dtype tables, every referenced index, and the per-core L1 budgets
test_dsl_cache_key.py            8 tests   the DSL's cache key and weight-shape guards, no device needed
test_reference_only.py          38 tests   the references, tile arithmetic, threshold edges, cost metrics, density sweep
test_kernel_api_symbols.py      92 tests   device APIs, CB ABI order, engine config, push/pop balance, accessor counts
test_package_imports.py         62 tests   every module parses, every import resolves, and submodules are told from typos
test_suite_layout.py            11 tests   the test/toolchain split, and the count block below
test_ttnn_call_sites.py          6 tests   every ttnn call site's argument count and kind
                                  ---
                                  223        total on a host with no toolchain

These are *collected* counts, which is what `test_suite_layout.py` compares against. Pytest
reports 222 passing rather than 223: the difference is one strict xfail,
`test_every_compute_engine_cbs_are_configured_at_startup[compute_lif_neuron.cpp]`, which names
the Phase 1 packer defect in its reason. Both numbers are correct and they measure different
things, so a run showing "183 passed" is not disagreeing with the table above.
```

The split has to be `collect_ignore` rather than a marker: those six modules import `ttnn` at
module scope, so they fail during *collection*, before pytest evaluates any `-m` expression.
Deselecting a module that will not import is not possible. Note what that means when reading a
result: on a bare host those six modules are not skipped after collection, they are never
collected at all, so a green run says nothing about them.

The runtime probe is deliberately not `find_spec`. Run from the repository root, `import ttnn`
*succeeds* as an empty namespace package — `ttnn.__file__` is `None` — because the C++ source
directory has the same name, and a device test would then die on its first attribute use with an
error that looks like a broken test. The conftest checks `__file__` is set and that
`ttnn.from_torch` exists, which is what an installed runtime looks like.

The toolchain-free half needs **numpy and pytest only** — `torch` and `ttnn` are imported inside
the fixtures that use them, not at conftest scope, so neither is required to collect it.

Reaching it still takes one flag, because two conftests sit above this directory and both import
third-party packages at module scope:

```
conftest.py                        loguru, torch, tracy, tt_umd, ttnn
models/conftest.py                 PIL, torchvision, loguru
models/experimental/snn/tests/...  this one — numpy, pytest
```

`--confcutdir` stops pytest collecting anything above this package, so neither of the first two
is ever imported:

```bash
python3 -m pytest models/experimental/snn/tests/ -v \
    --confcutdir=models/experimental/snn -o addopts=""
```

`ttsim` needs two environment variables, both set by the CI entry:

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1     # ttsim has no fast dispatch
export TT_METAL_DISABLE_SFPLOADMACRO=1   # ttsim does not implement SFPLOADMACRO
```

## Conventions

- **Exact where the value is discrete, tolerant where it is a float.** Spike trains,
  active-tile indices, tile counts and firing schedules are compared with `array_equal` or
  against a literal list — a spike is 0.0 or 1.0 and bfloat16 holds both exactly, and
  `unary_gt_tile` is a comparison rather than an approximation, so there is nothing to be
  tolerant about. Membranes and synaptic currents use `np.allclose`; every tolerance constant
  in a file carries the derivation next to it. The synaptic current is additionally modelled
  at the precision the device actually uses: `grid_weights` puts weights on a 1/256 grid,
  which bfloat16 represents exactly, so the reference multiplies the same values the device
  does and the tolerance only has to cover float32 accumulation.
- **Non-vacuity is asserted, not assumed.** Every test that compares two traces first checks
  that something actually fired, and every device output tensor is seeded with a value that
  cannot be a correct result (`-1.0` for a spike, `-9999.0` for a membrane or a current), so
  a writer that never ran is distinguishable from a writer that ran correctly.
- **`reset_seeds` plus an explicit `np.random.default_rng(213919)`.** The fixture follows
  repo convention; the explicit generator is what actually makes the data reproducible, and it
  also makes the fixed stimulus of the LIF tests readable.
- **`expect_error`, not a bare raises context manager**, which the `prefer-expect-error`
  pre-commit hook rejects in any `tests/` path.
- **Tolerances and tolerances only.** Nothing here asserts that a value is "reasonable" or
  "not obviously wrong"; every assertion is either an exact match against the reference or a
  bound derived in a comment from the arithmetic that produces the number.
- **Small shapes.** ttsim runs 10-50x slower than silicon and the whole suite shares one
  20-minute CI budget, so shapes are the smallest in which both layers fire and every case is
  asserted to be non-vacuous rather than assumed to be.

## Status

**The device tests have never been executed.** They were written on a machine that cannot
build or run tt-metal — `build_metal.sh` reads `/etc/os-release` and the toolchain is
`x86_64-linux-clang-20`, against a `darwin/arm64` host with no Tenstorrent device attached.
No claim in this directory about device behaviour has been observed; it is all derived from
the kernel sources and the NumPy reference.

First execution happens in CI, on a `sim_*` SKU, via the `SNN framework unit tests` entry in
`tests/pipeline_reorg/ttnn_sanity_tests.yaml` (`sim_wh_n150` and `sim_bh_p150`). Expect
first-run failures there: shapes and tolerances were reasoned out, not measured, and the
SFPU coverage of ttsim in particular is not guaranteed.

What *has* been run, on the authoring host, is everything that needs no device:

- `test_reference_only.py` — passes.
- The host-only cases in `test_sparse_matvec.py` and `test_dsl.py`, including every
  `expect_error` match string — pass.
- Every hard-coded expectation in the device tests (the `[4, 8, 12]` firing schedule, the
  `[0, 0, 0, 0, 1, 1, 1, 1]` quiet-then-loud train, the per-tile stimulus spike counts, the
  tile counts per activity level, the non-vacuity of both `SparseNet` shapes, the bfloat16
  round-trip of the weight grid) — checked against `reference/lif.py` on the host. The
  reference-side arithmetic is correct; what is unverified is the device agreeing with it.

## The device-free suite stays device-free

`test_reference_only.py` must run on a machine with no Tenstorrent toolchain at all, and that is
a property that decays quietly. It holds because the modules it reaches are ttnn-free by
construction: `reference/lif.py` imports no `ttnn`, and `snn/synapses.py` — which owns
`active_tiles`, the tile-selection rule the device and the CPU baseline must agree on — imports
`ttnn` lazily inside the one function that needs it. `snn/descriptors.py` does the same.

It is verifiable, and worth keeping verifiable: run the suite with `ttnn` made genuinely
unimportable rather than merely absent. A checkout that has a `ttnn/` directory on `sys.path`
will import it as an empty namespace package and fail later, on first attribute use, which reads
like a passing run:

```python
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            raise ImportError(f"BLOCKED: {name}")
        return None
sys.meta_path.insert(0, Block())
import pytest; sys.exit(pytest.main([...]))
```

If that reports 38 passed, the suite genuinely does not need tt-metal.

### Where the ttnn-free boundary actually sits

The import sweep over the package finds **four** modules that need a working `ttnn` at import
time, because each does a module-scope `import ttnn`:

```
snn/layer.py     import ttnn, and _TORCH_DTYPE = {ttnn.float32: ..., ttnn.bfloat16: ...}
snn/neuron.py    import ttnn
snn/multicore.py import ttnn
snn/fabric.py    import ttnn, and ttnn.CoreType.WORKER as a default argument
```

Everything else — `reference/`, `snn/config.py`, `snn/layout.py`, `snn/descriptors.py`,
`snn/synapses.py` and `snn/dsl.py` — imports with `ttnn` genuinely unimportable. `snn/dsl.py` looks
like it should not qualify, since it drives `SparseLIFLayer`, but it imports that inside
`compile()` rather than at module scope. `snn/descriptors.py` likewise imports `ttnn` inside each
function.

This was previously recorded as "exactly one module", naming only `snn/layer.py`, and asserting
that `snn/neuron.py`, `snn/fabric.py` and `snn/multicore.py` import without a working `ttnn`. That
is false: with a `sys.meta_path` blocker on `ttnn`, all three raise `ImportError: BLOCKED: ttnn`.
The conclusion the paragraph was reaching is unaffected, because it turns on which modules the free
half *reaches*, not on how many modules in total need ttnn.

`test_reference_only.py` reaches only `reference/lif.py`, `snn/config.py`, `snn/layout.py` and
`snn/synapses.py`; `test_dsl_cache_key.py` adds `snn/dsl.py`. All five import with `ttnn`
unimportable, so the device-free property holds. The rule for anyone editing: do not add a
module-scope `import ttnn`, or a default argument that reads one, to any module the free-half
tests reach.

A note on the sweep itself, so a later run is not misread: with a stub that returns plain classes
for attributes it does not know, `snn/fabric.py` appears to fail on `ttnn.CoreType.WORKER`. That is
the stub's limitation, not a defect — `CoreType` is exported at `ttnn/ttnn/__init__.py:268`, and
the sweep passes with a mock that resolves arbitrary attribute chains the way the real nanobind
module does.
