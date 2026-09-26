# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runnable comparison of the two-layer SNN on a Tenstorrent device and on the CPU.

    python -m models.experimental.snn.demo.demo                 # both paths, if a device is present
    python -m models.experimental.snn.demo.demo --no-device     # CPU baseline only
    python -m models.experimental.snn.demo.demo --steps 50 --hidden 256 --samples 256

What it prints
--------------
One table of the four throughput figures, one column per path; the per-layer share of weight tiles
the sparse synapse actually fetched; and a CPU-only readout accuracy. Every number is labelled with
the path that produced it: a device figure and a NumPy figure are not interchangeable, and the whole
point of the side-by-side is that the difference is visible rather than averaged away.

Two figures are deliberately not numbers:

* ``energy_per_spike`` is unmeasurable on this path. ttsim is a functional simulator with no power
  model, and CI has no silicon, so there is nothing to integrate. It is printed as unavailable
  rather than estimated from a datasheet, because an estimated joule figure is a guess wearing a
  measurement's clothes.
* the readout accuracy is fitted, not learned, and on the CPU only. The network weights are random,
  so what the accuracy demonstrates is that the dataset is separable and the wiring carries class
  information, not that the network was trained.

The demo exits 0 when no device is present, which is how it is expected to run on a machine without
one: the CPU column and the whole classification section still execute.
"""

import argparse
import sys

import numpy as np

from models.experimental.snn.reference.cpu_baseline import (
    DEFAULT_SEED,
    build_weights,
    cpu_sparsity_report,
    load_mnist,
    measure,
    synthetic_spike_dataset,
    two_layer_sparse_net,
)
from models.experimental.snn.snn.config import LIFConfig, SynapseConfig

N_INPUTS = 784
N_CLASSES = 10
UNAVAILABLE = "unavailable"

# The one place the energy answer is written, so the table and any log line cannot disagree.
ENERGY_UNAVAILABLE = "N/A — requires silicon; ttsim has no power model"

# Provisional firing rate used to size the output layer's weights before the hidden layer has run
# once. The demo replaces it with the rate it actually measured; see _calibrated_weights.
PROVISIONAL_HIDDEN_RATE = 0.1

# Below this many held-out samples the accuracy split is noise, so the demo says so rather than
# printing a percentage that looks like a result.
MIN_HELDOUT_TO_REPORT = 10


def parse_args(argv: list = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50, help="time steps per sample (default: 50)")
    parser.add_argument("--hidden", type=int, default=256, help="hidden layer width (default: 256)")
    parser.add_argument("--samples", type=int, default=256, help="dataset samples (default: 256)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for data and weights (default: 213919)")
    parser.add_argument(
        "--mnist-dir",
        type=str,
        default=None,
        help="directory holding MNIST IDX files; used only if already on disk, never downloaded",
    )
    parser.add_argument("--device", type=int, default=0, help="device index to open (default: 0)")
    parser.add_argument("--no-device", action="store_true", help="run the CPU baseline only, without opening a device")
    args = parser.parse_args(argv)
    # argparse's own error path beats a traceback from deep inside the network when a flag is silly.
    for name in ("steps", "hidden", "samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1, got {getattr(args, name)}")
    return args


def _load_dataset(args: argparse.Namespace) -> tuple:
    """MNIST when its IDX files are already on disk, otherwise the synthetic set. Never downloads."""
    reason = "no MNIST on disk"
    mnist = load_mnist(args.mnist_dir, n_samples=args.samples, seed=args.seed)
    if mnist is not None:
        spikes, labels, path = mnist
        if spikes.shape[1] == N_INPUTS:
            return spikes, labels, f"MNIST from {path}"
        # The stack is wired for 784 inputs, so a dataset that is not 28x28 is not one this demo can
        # run. Say which file and why, rather than failing three layers later with a shape error.
        reason = f"MNIST at {path} has {spikes.shape[1]} inputs, not {N_INPUTS}"
    spikes, labels = synthetic_spike_dataset(args.samples, N_INPUTS, N_CLASSES, seed=args.seed)
    return spikes, labels, f"synthetic spike set ({spikes.shape[0]} samples; {reason}, nothing downloaded)"


def _calibrated_weights(config: LIFConfig, sample: np.ndarray, args: argparse.Namespace) -> list:
    """Weights whose scale matches the activity each layer actually sees.

    Layer 0's input spike count is measured from the sample. Layer 1's input is the hidden layer's
    own output, which does not exist until the network has run, so this does one untimed probe pass
    with a provisional scale, reads the hidden firing rate off it, and rebuilds with that count. Only
    the second matrix changes: ``build_weights`` is seeded, so the first is bit-identical between
    the probe and the reported run.
    """
    input_spikes = float(np.asarray(sample, dtype=np.float32).sum())
    probe = build_weights(
        config, N_INPUTS, args.hidden, N_CLASSES, [input_spikes, args.hidden * PROVISIONAL_HIDDEN_RATE], args.seed
    )
    trains, _ = two_layer_sparse_net(sample, probe, config, args.steps)
    hidden_rate = float(trains[0].mean())
    return build_weights(
        config, N_INPUTS, args.hidden, N_CLASSES, [input_spikes, max(args.hidden * hidden_rate, 1.0)], args.seed
    )


def _device_path(config: LIFConfig, weights: list, sample: np.ndarray, n_steps: int, device_id: int) -> tuple:
    """Run the stack on a Tenstorrent device. Returns ``(metrics, trains, reports)``.

    ``ttnn`` and the layer are imported inside the function so this module imports on a machine that
    has no Tenstorrent toolchain at all, which is how it is first run.
    """
    import ttnn

    # Run from the repository root, ``import ttnn`` can resolve to the C++ source tree's ``ttnn/``
    # directory as an empty namespace package, which imports fine and then has no runtime in it.
    # Checking here names that, instead of failing later with an attribute error.
    if not hasattr(ttnn, "open_device"):
        raise RuntimeError(
            f"ttnn imported as {getattr(ttnn, '__path__', None) or getattr(ttnn, '__file__', ttnn)!r}, "
            "which is not the tt-metalium Python runtime; no device is reachable from here"
        )

    from models.experimental.snn.snn.layer import SparseNet

    layer_configs = [SynapseConfig(fan_in=matrix.shape[1], fan_out=matrix.shape[0]) for matrix in weights]
    device = ttnn.open_device(device_id=device_id)
    try:
        network = SparseNet(layer_configs, config, weights, device)
        captured = {}

        def run_once():
            trains = network.run(sample, n_steps)
            captured["trains"] = trains
            return trains, sum(int(train.sum()) for train in trains)

        metrics = measure(run_once, n_steps)
        return metrics, captured["trains"], network.sparsity_report()
    finally:
        ttnn.close_device(device)


def _readout_accuracy(spikes: np.ndarray, labels: np.ndarray, weights: list, config, n_steps: int) -> dict:
    """Accuracy of a ridge-regularised linear readout on the hidden population's spike counts.

    The network weights are random, so nothing in the stack is trained; this is the only thing here
    that fits anything, and it is fitted on the spike counts the hidden layer produced. The features
    are the per-neuron total over the run, the standard readout for an SNN, and the fit is a plain
    least-squares solve against one-hot labels. The held-out figure is the one worth reading: the
    fit figure on a few dozen samples is not evidence of anything.
    """
    features = np.zeros((spikes.shape[0], weights[0].shape[0]), dtype=np.float64)
    for index, sample in enumerate(spikes):
        trains, _ = two_layer_sparse_net(sample, weights, config, n_steps)
        features[index] = trains[0].sum(axis=0)
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    n_classes = int(labels.max()) + 1
    targets = np.eye(n_classes)[labels]
    # Clamped to the data: with fewer than two samples there is no split to make, and the fit
    # silently over-reading its own sample count would be a worse lie than an empty held-out set.
    n_fit = min(features.shape[0], max(2, int(0.6 * features.shape[0])))
    ridge = 1e-2 * np.eye(design.shape[1])
    solution = np.linalg.solve(design[:n_fit].T @ design[:n_fit] + ridge, design[:n_fit].T @ targets[:n_fit])
    correct = np.argmax(design @ solution, axis=1) == labels
    held_out = correct[n_fit:]
    return {
        "n_fit": n_fit,
        "n_held_out": int(held_out.size),
        "fit": float(correct[:n_fit].mean()),
        "held_out": float(held_out.mean()) if held_out.size else float("nan"),
        "chance": 1.0 / n_classes,
    }


def _print_table(headers: list, rows: list) -> None:
    """Print an aligned table. ``rows`` are already-formatted strings, one list per line."""
    widths = [max(len(str(row[column])) for row in [headers, *rows]) for column in range(len(headers))]
    line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def main(argv: list = None) -> int:
    args = parse_args(argv)
    config = LIFConfig(dt=1.0, tau=20.0, v_threshold=1.0, v_reset=1.0)

    spikes, labels, source = _load_dataset(args)
    sample = spikes[0]
    weights = _calibrated_weights(config, sample, args)

    print(
        "two-layer SNN: {} -> {} -> {} | neuron: LIF(dt={}, tau={}, v_threshold={}, v_reset={})".format(
            N_INPUTS, args.hidden, N_CLASSES, config.dt, config.tau, config.v_threshold, config.v_reset
        )
    )
    print(f"data: {source}")
    weight_bound = float(np.abs(weights[0]).max())
    print(f"steps per sample: {args.steps} | seed: {args.seed} | layer 0 weight bound: {weight_bound:.4f}")

    captured = {}

    def cpu_once():
        trains, total_spikes = two_layer_sparse_net(sample, weights, config, args.steps)
        captured["trains"] = trains
        return trains, total_spikes

    cpu_metrics = measure(cpu_once, args.steps)
    cpu_trains = captured["trains"]
    cpu_reports = cpu_sparsity_report(sample, weights, cpu_trains)

    device_metrics = device_reports = None
    device_reason = None
    if args.no_device:
        device_reason = "--no-device given"
    else:
        try:
            device_metrics, _, device_reports = _device_path(config, weights, sample, args.steps, args.device)
        except Exception as error:  # noqa: BLE001 - any failure here means "no usable device", not a demo bug
            device_reason = f"{type(error).__name__}: {error}"

    print(
        "firing rate (cpu, share of neurons per time step): "
        + " | ".join(f"layer {index} {train.mean() * 100:.1f}%" for index, train in enumerate(cpu_trains))
    )

    _print_table(
        ["metric", "cpu (numpy)", f"device {args.device}"],
        [
            [
                "total_spikes",
                f"{cpu_metrics['total_spikes']}",
                f"{device_metrics['total_spikes']}" if device_metrics else UNAVAILABLE,
            ],
            [
                "wall_seconds",
                f"{cpu_metrics['wall_seconds']:.6f}",
                f"{device_metrics['wall_seconds']:.6f}" if device_metrics else UNAVAILABLE,
            ],
            [
                "spikes_per_second",
                f"{cpu_metrics['spikes_per_second']:.1f}",
                f"{device_metrics['spikes_per_second']:.1f}" if device_metrics else UNAVAILABLE,
            ],
            [
                "latency_per_timestep_ms",
                f"{cpu_metrics['latency_per_timestep_ms']:.4f}",
                f"{device_metrics['latency_per_timestep_ms']:.4f}" if device_metrics else UNAVAILABLE,
            ],
            ["energy_per_spike", ENERGY_UNAVAILABLE, "unmeasured"],
        ],
    )
    # Scope of the four figures above, stated because the table does not show it. `total_spikes`,
    # `wall_seconds` and the two derived rates are all for ONE sample over `steps` time steps, not
    # for the whole dataset; and `measure` times a single call with no warm-up and no repetition.
    # A 0.8 ms measurement carries nowhere near the precision the printed digits suggest, so these
    # are an order-of-magnitude figure, not a benchmark. Repeating the call would need a state reset
    # between repeats -- the membrane carries over -- so this is stated rather than papered over.
    print(
        f"the four figures above are one sample over {args.steps} steps, timed once with no "
        f"warm-up and no repetition; treat them as order-of-magnitude, not as a benchmark."
    )
    print("energy is unmeasured on both paths: " + ENERGY_UNAVAILABLE + ".")
    if device_reason:
        print(f"device path not run ({device_reason}); the cpu column is the whole result.")

    print()
    print("share of weight tiles the sparse synapse fetched:")
    for index, matrix in enumerate(weights):
        report = cpu_reports[index]
        device_fraction = f"{device_reports[index]['active_fraction']:.3f}" if device_reports else UNAVAILABLE
        print(
            f"  layer {index} ({matrix.shape[1]} -> {matrix.shape[0]})  cpu {report['active_fraction']:.3f}"
            f" ({report['tiles_fetched']}/{report['total_tiles']} tiles)  device {device_fraction}"
        )

    accuracy = _readout_accuracy(spikes, labels, weights, config, args.steps)
    print()
    print(
        "readout on hidden spike counts (cpu only, random weights, fit on {} of {} samples):".format(
            accuracy["n_fit"], spikes.shape[0]
        )
    )
    # At a handful of samples the held-out split is a coin toss either way, so say so rather than
    # letting a number that small stand as if it were a result.
    if accuracy["n_held_out"]:
        held_out = f"{accuracy['held_out'] * 100:.1f}%"
        caveat = (
            ""
            if accuracy["n_held_out"] >= MIN_HELDOUT_TO_REPORT
            else f" (only {accuracy['n_held_out']} held out: too few to read)"
        )
    else:
        held_out = "n/a, no held-out samples"
        caveat = ""
    print(f"  fit {accuracy['fit'] * 100:.1f}% | held out {held_out} | chance {accuracy['chance'] * 100:.1f}%{caveat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
