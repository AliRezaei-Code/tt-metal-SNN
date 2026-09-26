# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free CPU baseline for the two-layer SNN: the same network, the same loop, NumPy only.

Why this module exists
----------------------
The demo's throughput comparison is only meaningful if the two paths execute the same arithmetic in
the same order. So this module does not reimplement anything: the neuron update comes from
``reference.lif.lif_step`` and the synaptic mat-vec from ``reference.lif.sparse_matvec``, which are
the same ground truth the device kernels are tested against. If those and the kernels disagree, the
comparison is worthless, so a second implementation here would defeat the purpose.

The time-stepping loop mirrors ``snn.layer.SparseNet.run``: one network time step re-presents the
encoded input to layer 0, then chains layer 0 -> layer 1 within that same step, so a run of N steps
is N passes of the stack and the membrane of each population carries across steps. Both paths
therefore report the same four numbers over the same denominator (one time step = one pass through
the whole stack).

Tile padding
------------
The device pads the spike vector and the weight matrix up to whole 32-wide tiles
(``snn.synapses.broadcast_spike_tiles`` / ``pack_weight_tiles``) and keeps only column 0 of each
output tile. The padding contributes no terms: a padded input neuron has a zero spike, so its weight
column multiplies out to an exact zero, and a padded output row is discarded on read. Both paths
therefore sum the same products, but not in the same order -- the device accumulates one 32-wide
tile product at a time -- so they agree to float32 rounding rather than bit for bit. For scale,
replaying the tile arithmetic with NumPy on a 784 -> 256 layer gives 6e-6 of difference from the
ordering alone against 3e-3 from rounding the weights to the bfloat16 the device actually stores.
Padding is not where a device-versus-CPU difference comes from.

This path therefore does *not* pad. Zero-padding the weights here would change nothing numerically,
but zero-*truncating* the input would, and that is the mistake this note exists to prevent.
"""

import gzip
import math
import struct
import time
from pathlib import Path

import numpy as np

from models.experimental.snn.reference.lif import lif_step, sparse_matvec
from models.experimental.snn.snn.config import LIFConfig
from models.experimental.snn.snn.layout import TILE_WIDTH
from models.experimental.snn.snn.synapses import active_fraction, active_tiles

# The repo-wide seed: the root conftest.py ``reset_seeds`` fixture seeds 213919, so a run started
# from that fixture and a run started from this default see the same weights and the same samples.
DEFAULT_SEED = 213919

# Width of the uniform weight distribution, expressed in units of "the drive a neuron needs to
# reach threshold". See ``weight_scale`` for the derivation and ``build_weights`` for why the value
# is what it is.
SYNAPTIC_HEADROOM = 6.0

# Background firing probability of the synthetic set, and the elevated rate inside a class's own
# cue block. The gap between the two is what makes the classes separable; see
# ``synthetic_spike_dataset``.
SYNTHETIC_RATE = 0.02
SYNTHETIC_CUE_RATE = 0.4
SYNTHETIC_CUE_PER_CLASS = 32


def two_layer_sparse_net(input_spikes: np.ndarray, weights: list, neuron_config: LIFConfig, n_steps: int) -> tuple:
    """Run the SNN stack for ``n_steps`` on the host. Returns ``(trains, total_spikes)``.

    ``weights`` is the stack's weight matrices in layer order, each dense ``(fan_out, fan_in)``;
    layer ``i`` consumes the spikes of layer ``i - 1`` and layer 0 consumes ``input_spikes``.
    ``trains`` mirrors the device's: one ``(n_steps, fan_out)`` float32 spike train per layer, and
    ``total_spikes`` is the number of spike events the whole stack emitted, summed over layers and
    time steps. That is the numerator both paths divide by their own wall clock.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    matrices = [np.asarray(w, dtype=np.float32) for w in weights]
    if not matrices:
        raise ValueError("weights must contain at least one layer")
    encoded = np.asarray(input_spikes, dtype=np.float32).reshape(-1)
    # Walk the fan-in chain from the encoded input, so a stack that does not line up is rejected
    # before it silently runs a matvec against the wrong number of input neurons.
    for index, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError(f"weights[{index}] must be 2-D (fan_out, fan_in), got shape {matrix.shape}")
        source = "input_spikes" if index == 0 else f"layer {index - 1}"
        expected = encoded.size if index == 0 else matrices[index - 1].shape[0]
        if matrix.shape[1] != expected:
            raise ValueError(
                f"weights[{index}] expects {expected} inputs from {source}, but it is "
                f"(fan_out, fan_in) = {matrix.shape}"
            )

    membranes = [np.zeros(matrix.shape[0], dtype=np.float32) for matrix in matrices]
    trains = [np.zeros((n_steps, matrix.shape[0]), dtype=np.float32) for matrix in matrices]
    total_spikes = 0

    for step in range(n_steps):
        # The encoded input is re-presented every step: this is a feedforward-through-time network,
        # so the sample drives layer 0 on every step and each layer feeds the next within the step.
        signal = encoded
        for index, matrix in enumerate(matrices):
            current = sparse_matvec(matrix, signal)
            membranes[index], spikes = lif_step(membranes[index], current, neuron_config)
            trains[index][step] = spikes
            total_spikes += int(spikes.sum())
            signal = spikes

    return trains, total_spikes


def active_tile_count(spike_bits: np.ndarray) -> int:
    """How many input tiles the device would fetch for this spike vector.

    Delegates to ``snn.synapses.active_tiles`` rather than reimplementing it. That used to be
    impossible -- ``snn.synapses`` imported ``ttnn`` at module scope and this file has to stay
    importable with NumPy alone -- so it carried its own copy, which is exactly the kind of
    duplication that drifts. ``synapses`` now imports ``ttnn`` lazily, so the tile-selection rule
    has a single owner and the CPU side cannot quietly disagree with the device about which tiles
    are fetched.
    """
    return len(active_tiles(spike_bits))


def cpu_sparsity_report(input_spikes: np.ndarray, weights: list, trains: list) -> list:
    """The device's ``sparsity_report()`` counters, computed from CPU spike trains.

    Same keys and same arithmetic as ``snn.layer.SparseLIFLayer.sparsity_report()`` so the demo can
    put the two cost columns next to each other. Derived from the trains rather than from a second
    simulation, using the loop's own invariant: layer 0's input on every step is ``input_spikes``,
    and layer ``i > 0``'s input on step ``t`` is ``trains[i - 1][t]``.
    """
    n_steps = trains[0].shape[0]
    encoded = np.asarray(input_spikes, dtype=np.float32).reshape(-1)
    reports = []
    for index, matrix in enumerate(weights):
        kt = -(-matrix.shape[1] // TILE_WIDTH)
        fetched = 0
        for step in range(n_steps):
            incoming = encoded if index == 0 else trains[index - 1][step]
            fetched += active_tile_count(incoming)
        total = kt * n_steps
        reports.append(
            {
                "steps": n_steps,
                "tiles_fetched": fetched,
                "total_tiles": total,
                "active_fraction": active_fraction(fetched, total),
            }
        )
    return reports


def measure(net_callable, n_steps: int) -> dict:
    """Time one call of ``net_callable`` and derive the four figures the demo reports.

    ``net_callable`` takes no arguments and returns the ``(trains, total_spikes)`` pair of
    :func:`two_layer_sparse_net`, so the device and CPU paths go through this one timing function
    and cannot drift apart in how they are measured. One time step means one pass through the whole
    stack, on both paths.
    """
    start = time.perf_counter()
    _, total_spikes = net_callable()
    wall_seconds = time.perf_counter() - start

    return {
        "total_spikes": int(total_spikes),
        "wall_seconds": wall_seconds,
        "spikes_per_second": (total_spikes / wall_seconds) if wall_seconds > 0.0 else 0.0,
        "latency_per_timestep_ms": (1000.0 * wall_seconds / n_steps) if n_steps else 0.0,
    }


def weight_scale(neuron_config: LIFConfig, active_inputs: float, headroom: float = SYNAPTIC_HEADROOM) -> float:
    """Half-width of the uniform weight distribution feeding one layer.

    Derivation, for a layer whose input population fires ``active_inputs`` spikes per time step:
    the weights are ``U(-a, a)``, so the synaptic drive ``I = sum_i w_i s_i`` is zero mean with
    standard deviation ``a * sqrt(active_inputs / 3)``. A LIF neuron driven by a constant ``I`` only
    ever reaches threshold when ``I > v_threshold * (1 - alpha)``, because its membrane settles at
    ``I / (1 - alpha)``. Solving for ``a`` with the drive spread set to ``headroom`` times that
    per-step threshold drive gives

        a = headroom * v_threshold * (1 - alpha) * sqrt(3) / sqrt(active_inputs)

    so the choice of ``headroom`` is the choice of how far above threshold the typical neuron sits.
    Small values go silent, large values put the population in permanent saturation; at
    ``SYNAPTIC_HEADROOM = 6.0`` the measured hidden firing rate on the synthetic set is around 10%
    of the population per time step, which is spiking without either failure.
    """
    if active_inputs <= 0:
        raise ValueError(f"active_inputs must be positive, got {active_inputs}")
    threshold_drive = neuron_config.v_threshold * (1.0 - neuron_config.decay_factor)
    return headroom * threshold_drive * math.sqrt(3.0) / math.sqrt(active_inputs)


def build_weights(
    neuron_config: LIFConfig,
    n_inputs: int,
    hidden: int,
    n_classes: int,
    active_inputs: list,
    seed: int = DEFAULT_SEED,
) -> list:
    """Random synaptic weights for the ``n_inputs -> hidden -> n_classes`` stack, as ``(fan_out, fan_in)``.

    The matrices live here rather than in the demo so the device and CPU paths are handed the same
    numbers: a throughput comparison over two different weight sets measures the wrong thing.

    ``active_inputs`` holds one expected spike count per layer, i.e. how many of that layer's input
    neurons fire per time step, and each layer's weight scale follows from it (see
    :func:`weight_scale`). Layer 0's count is measured from the dataset. Layer 1's has to be guessed,
    because the hidden layer has not run yet, which is why the demo makes one untimed pass first and
    rebuilds from the firing rate it actually observed.

    Seeded rather than handed a ``Generator``, so that rebuilding only to correct the second layer's
    scale leaves the first matrix bit-identical; the demo's calibration pass relies on that.
    """
    if len(active_inputs) != 2:
        raise ValueError(f"expected one active-input count per layer for a two-layer stack, got {len(active_inputs)}")
    rng = np.random.default_rng(seed)
    scale_in = weight_scale(neuron_config, float(active_inputs[0]))
    scale_hidden = weight_scale(neuron_config, float(active_inputs[1]))
    w1 = rng.uniform(-scale_in, scale_in, (hidden, n_inputs)).astype(np.float32)
    w2 = rng.uniform(-scale_hidden, scale_hidden, (n_classes, hidden)).astype(np.float32)
    return [w1, w2]


def synthetic_spike_dataset(
    n_samples: int,
    n_inputs: int,
    n_classes: int,
    rate: float = SYNTHETIC_RATE,
    seed: int = DEFAULT_SEED,
    cue_per_class: int = SYNTHETIC_CUE_PER_CLASS,
    cue_rate: float = SYNTHETIC_CUE_RATE,
) -> tuple:
    """A download-free spike dataset whose classes are linearly separable. Returns ``(spikes, labels)``.

    The generator's contract:

    * Every sample is one Bernoulli draw per input channel; there is no image and no download, so
      the demo runs on a machine that has never seen MNIST.
    * The ``n_inputs`` channels are partitioned into ``n_classes`` disjoint *cue blocks* of
      ``cue_per_class`` channels each, at random positions. A sample of class ``c`` fires its own
      block at ``cue_rate`` and every other channel at the background ``rate``. The blocks are
      disjoint, so the spike count *within* each block is a per-class feature and comparing the
      counts -- argmax over blocks, or any linear readout of the per-block sums -- recovers the
      label exactly. That is what a two-layer SNN with a linear readout on the hidden spikes can
      express.
    * Note that a *single* global threshold on the own-block count does **not** work, and the
      generator does not claim it does: the background carries
      ``n_inputs * rate - cue_per_class * rate`` spikes spread over many blocks, so a
      low own-block count can still be the largest. Only the comparison between blocks
      separates the classes. Measured on 2000 samples (200 per class), argmax over the per-block
      counts classifies every sample correctly.
    * Nothing else varies with the class. Background channels are drawn from one shared
      distribution, so there is no class-dependent bias to exploit beyond the cue blocks, and cue
      positions are permuted so channel order carries no class information.
    * Labels are assigned round-robin, so any prefix of the returned arrays is class balanced and
      ``--samples 8`` still exercises eight distinct classes.
    * ``rate`` is a probability per channel per sample, not a count: a sample of this set carries on
      the order of ``n_inputs * rate + cue_per_class * (cue_rate - rate)`` spikes, which is the
      number :func:`build_weights` needs to size the synaptic scale.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")
    cue_per_class = min(cue_per_class, n_inputs // n_classes)
    if cue_per_class < 1:
        raise ValueError(f"n_inputs={n_inputs} and n_classes={n_classes} leave no room for a cue block")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_inputs)
    labels = np.arange(n_samples) % n_classes
    probabilities = np.full((n_samples, n_inputs), float(rate), dtype=np.float64)
    for class_index in range(n_classes):
        cue_block = order[class_index * cue_per_class : (class_index + 1) * cue_per_class]
        probabilities[np.ix_(labels == class_index, cue_block)] = float(cue_rate)
    spikes = (rng.random((n_samples, n_inputs)) < probabilities).astype(np.float32)
    return spikes, labels.astype(np.int64)


# Standard cache locations searched when no explicit directory is given. The first entry is
# torchvision's ``root="./data"`` default, which is what every other demo in this repo uses.
MNIST_SEARCH_DIRS = ("data/MNIST/raw", "data/mnist", "~/datasets/mnist", "~/.cache/mnist")
_MNIST_IMAGE_STEMS = ("train-images-idx3-ubyte", "t10k-images-idx3-ubyte")
_MNIST_LABEL_STEMS = ("train-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")


def _read_idx(path: Path) -> np.ndarray:
    """Read one IDX file, gzipped or not. Returns the payload described by the IDX header."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 8:
        raise ValueError(f"{path} is too short to be an IDX file")
    magic = struct.unpack(">I", raw[:4])[0]
    if magic not in (0x00000801, 0x00000803):
        raise ValueError(f"{path} has IDX magic 0x{magic:08x}, expected 0x00000801 or 0x00000803")
    dimensions = magic & 0xFF
    shape = struct.unpack(">" + "I" * dimensions, raw[4 : 4 + 4 * dimensions])
    return np.frombuffer(raw[4 + 4 * dimensions :], dtype=np.uint8).reshape(shape)


def _find_mnist_files(directory: Path) -> tuple:
    """Locate one image file and its matching label file in ``directory``, plain or gzipped."""
    for image_stem in _MNIST_IMAGE_STEMS:
        for label_stem in _MNIST_LABEL_STEMS:
            for suffix in ("", ".gz"):
                image_path = directory / f"{image_stem}{suffix}"
                label_path = directory / f"{label_stem}{suffix}"
                if image_path.is_file() and label_path.is_file():
                    return image_path, label_path
    return None, None


def load_mnist(mnist_dir: str = None, n_samples: int = None, seed: int = DEFAULT_SEED) -> tuple:
    """Load MNIST from disk as spikes. Returns ``(spikes, labels, path)``, or ``None`` if absent.

    This never downloads anything and never imports torchvision: MNIST is used only when the IDX
    files are already on disk, either under an explicit ``mnist_dir`` or under one of
    :data:`MNIST_SEARCH_DIRS`. Anything else -- no directory, a partial download, a corrupt header
    -- returns ``None`` and the caller falls back to :func:`synthetic_spike_dataset`, which is the
    default path anyway. That keeps the demo runnable on a machine with no network and no dataset.

    Each pixel is one Bernoulli draw with probability ``pixel / 255``: a single-frame stochastic
    rate code, matching the one-input-vector-per-sample loop the network runs. With ``n_samples``
    set, a seeded permutation picks that many of the images rather than taking a prefix, so the
    subset is reproducible and not ordered by class.
    """
    candidates = ([Path(mnist_dir).expanduser()] if mnist_dir else []) + [
        Path(entry).expanduser() for entry in MNIST_SEARCH_DIRS
    ]
    for directory in candidates:
        if not directory.is_dir():
            continue
        image_path, label_path = _find_mnist_files(directory)
        if image_path is None:
            continue
        try:
            images = _read_idx(image_path)
            labels = _read_idx(label_path)
        except (OSError, ValueError, struct.error):
            continue
        if images.ndim != 3 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
            continue
        flat = images.reshape(images.shape[0], -1).astype(np.float64)
        rng = np.random.default_rng(seed)
        if n_samples is not None and n_samples < flat.shape[0]:
            keep = rng.permutation(flat.shape[0])[:n_samples]
            flat, labels = flat[keep], labels[keep]
        spikes = (rng.random(flat.shape) < flat / 255.0).astype(np.float32)
        return spikes, labels.astype(np.int64), str(image_path)
    return None
