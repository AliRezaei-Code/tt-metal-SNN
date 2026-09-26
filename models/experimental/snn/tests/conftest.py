# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the SNN tests, and the split between the two halves of the suite.

The package has two kinds of test that need opposite things:

* ``test_reference_only.py`` and ``test_kernel_api_symbols.py`` need no Tenstorrent toolchain at
  all. They pin the reference semantics, the tile arithmetic and the device API names, and they
  are the only feedback available on a machine that cannot build tt-metal.
* ``test_lif_neuron.py``, ``test_sparse_matvec.py``, ``test_spike_propagation.py``,
  ``test_dsl.py`` and ``test_multicore_partition.py`` need ``ttnn``. The first four drive real
  kernels and so need a device or ttsim; the last only needs ``ttnn``, because the multi-core
  partition is computed on the host. They are grouped here because the split that matters
  locally is "can this be imported at all", and that is the same question for all five.

That split is enforced here rather than documented, because the alternative does not work. Those
four modules import ``ttnn`` during *collection*, so a ``-m "not device"`` marker is consulted
too late: pytest fails collecting them before it ever evaluates the marker. ``collect_ignore``
is the only mechanism early enough. The result is that one command is correct in both places --
locally the device half is skipped and the free half runs; in CI, where ``ttnn`` is a real
install, all of them run.
"""


import numpy as np
import pytest

from models.experimental.snn.snn.layout import TILE_WIDTH

# 2^8: every k/256 with |k| <= 256 is exactly representable in bfloat16's 8-bit significand.
BF16_EXACT_DENOMINATOR = 256

# Modules that cannot be collected without a working tt-metal runtime.
_DEVICE_TESTS = (
    "test_lif_neuron.py",
    "test_sparse_matvec.py",
    "test_spike_propagation.py",
    "test_dsl.py",
    "test_multicore_partition.py",
    "test_fabric_program_guards.py",
)


def _ttnn_runtime_available() -> bool:
    """True when ``ttnn`` is a real runtime, not the source tree shadowing the name.

    Run from the repository root, ``import ttnn`` *succeeds* against the C++ source directory
    -- it is also called ``ttnn`` -- and yields an empty namespace package with no attributes at
    all. A device test would sail through that import and die on its first attribute use, with
    an error that reads like a broken test rather than a missing runtime.

    "Is it a real runtime" has three observable shapes, and this accepts all of them:

    * a wheel or an installed regular package -- ``ttnn.from_torch`` is right there;
    * a namespace directory whose actual package is one level down, which is how
      ``models-unit-tests-impl.yaml`` describes its prebuilt-image path ("/work/ttnn is
      required because no wheel is installed: it is a namespace dir whose package is
      /work/ttnn/ttnn"). The attributes live on ``ttnn.ttnn``;
    * the source shadow, where neither level has any.

    Probing attributes is what separates those; asking whether ``__file__`` is set is not,
    because a legitimate namespace install has none. Getting this wrong is invisible -- the
    device half would simply never run -- so the test below asserts the shape directly.
    """
    try:
        import ttnn
    except Exception:
        return False
    for candidate in (ttnn, getattr(ttnn, "ttnn", None)):
        if candidate is not None and hasattr(candidate, "from_torch"):
            return True
    return False


collect_ignore = [] if _ttnn_runtime_available() else list(_DEVICE_TESTS)


@pytest.fixture
def tile_tensor():
    """Factory: host array -> tile-layout DRAM tensor of the requested ttnn dtype.

    Both ``torch`` and ``ttnn`` are imported inside the fixture, not at module scope, so that
    importing this file needs neither. The device-free half is numpy and pytest only, and it is
    collected even where a torch or tt-metal install is absent.
    """
    import torch
    import ttnn

    torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.uint32: torch.int32}

    def make(host_array, dtype, device):
        return ttnn.from_torch(
            torch.as_tensor(np.ascontiguousarray(host_array)).to(torch_dtype[dtype]),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return make


@pytest.fixture
def read_back():
    """Factory: device tensor -> host numpy, widening bfloat16 on the way."""
    import torch
    import ttnn

    def read(tensor):
        host = ttnn.to_torch(tensor)
        if host.dtype is torch.bfloat16:
            host = host.to(torch.float32)
        return host.numpy()

    return read


@pytest.fixture
def spike_pattern():
    """Factory: reproducible 0/1 spike vector at a requested density, filling whole input tiles.

    This belongs here rather than in ``snn/synapses.py`` because it is test-data generation, and
    that module is imported by the reference implementation, the CPU baseline and the demo, none of
    which need a spike-pattern generator. Putting it there inverts the dependency: production code
    carrying a helper only its own tests use.

    It is shared because both halves of the suite need it -- the free-half density sweep and the
    device matvec cases -- and a copy per half could drift, leaving the free-half test pinning tile
    counts for a pattern the device tests never run.

    Placement starts on a seeded tile boundary and fills that tile before starting the next, so
    the number of active tiles is exactly ``ceil(count / tile)``. Scattering 13 spikes at random
    across eight tiles lands in about six of them, which would turn "5% activity" into a coin flip
    and make the density sweep meaningless.
    """

    def make(rng, activity, fan_in, tile=TILE_WIDTH):
        n_input_tiles = -(-fan_in // tile)
        count = int(round(activity * fan_in))
        start = int(rng.integers(0, n_input_tiles)) * tile
        spikes = np.zeros(fan_in, dtype=np.float32)
        spikes[(start + np.arange(count)) % fan_in] = 1.0
        return spikes

    return make


@pytest.fixture
def grid_weights():
    """Factory: reproducible ``(fan_out, fan_in)`` weights that survive bfloat16 unchanged."""

    def make(rng, fan_out, fan_in):
        grid = rng.integers(-BF16_EXACT_DENOMINATOR, BF16_EXACT_DENOMINATOR + 1, size=(fan_out, fan_in))
        return (grid / BF16_EXACT_DENOMINATOR).astype(np.float32)

    return make
