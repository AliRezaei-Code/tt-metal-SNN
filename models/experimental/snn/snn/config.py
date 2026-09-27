# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Neuron and layer hyperparameters."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LIFConfig:
    """Leaky integrate-and-fire parameters for one neuron population.

    The discrete update the device kernel implements is, with ``alpha = exp(-dt / tau)``::

        v_new = alpha * v_old + I
        s     = 1.0 if v_new > v_threshold else 0.0
        v_out = v_new - s * v_reset

    which is reset by subtraction: a spiking neuron loses exactly ``v_reset`` of potential.

    **This is not SpikingJelly's ``LIFNode``**, and the difference is deliberate rather than an
    oversight. SpikingJelly's ``LIFNode`` -- see ``neuronal_charge_no_decay_input`` and
    ``jit_eval_single_step_forward_hard_reset_no_decay_input`` in
    ``spikingjelly/activation_based/neuron.py`` -- differs in three ways:

    ==========================  ==========================  ================================
    step                        this framework              SpikingJelly ``LIFNode``
    ==========================  ==========================  ================================
    decay                       ``alpha = exp(-dt / tau)``  ``alpha = 1 - dt/tau``, plus a
                                                            ``+ v_reset / tau`` term when
                                                            ``v_reset != 0``
    threshold                   ``v_new > v_threshold``      ``v_new >= v_threshold``
    hard reset                  ``v -= s * v_reset``         ``v = v_reset`` (set, not
                                                            subtracted)
    ==========================  ==========================  ================================

    The decay term is the *exact* homogeneous solution of ``tau dv/dt = -v``, so it is correct at
    any ``dt/tau`` rather than only in the small-step limit the ``1 - dt/tau`` form approximates.
    The drive term is not exact, and the distinction is worth keeping straight: holding ``I``
    constant across the step, the exact solution is
    ``v(t+dt) = alpha*v(t) + I*tau*(1 - alpha)``, while this update applies ``I`` unscaled. At
    ``dt/tau = 0.05`` the step overshoots the exact one by ``I - I*tau*(1-alpha)``, about 2.5% of the
    step, and the steady state is ``I/(1-alpha)`` rather than ``I*tau``. It is the conventional SNN
    form -- SpikingJelly and snnTorch both scale the drive this way -- but it is a hybrid, not an
    exact integration, and nothing here should claim otherwise.

    Against snnTorch's ``Leaky`` (``snntorch/_neurons/leaky.py`` and the ``fire``/``_base_sub``
    in ``snntorch/_neurons/neurons.py``), configured as
    ``Leaky(beta=exp(-1 / tau_per_decay), threshold=1.0, reset_mechanism="subtract")``, this
    framework agrees on two of the three steps and differs on one:

    * decay -- snnTorch multiplies by ``beta`` directly, clipped to ``[0, 1]``, so it agrees
      exactly when the caller sets ``beta = exp(-1 / tau_per_decay)``;
    * threshold -- snnTorch's ``fire`` is documented "Generates spike if mem > threshold", a
      strict comparison, the same as here;
    * reset -- snnTorch subtracts ``threshold`` rather than a separate reset voltage, and with
      the default ``reset_delay=True`` it applies the *previous* step's spike. This framework
      subtracts ``v_reset`` and applies the current step's spike. With ``v_reset == threshold``
      the two differ only by that one-step delay.

    The practical consequence is that the two frameworks disagree on the very first step of a
    constant drive, because ``v`` lands exactly on the threshold: with ``I = 1.0`` and ``v = 0``,
    ``v_new`` is ``1.0`` under both, and ``>`` says quiet where ``>=`` says fire. Comparing a
    spike train against SpikingJelly therefore has to account for this, rather than reading the
    disagreement as a kernel bug.
    """

    dt: float = 1.0
    tau: float = 20.0
    v_threshold: float = 1.0
    v_reset: float = 1.0

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.v_reset < 0:
            raise ValueError(f"v_reset must be non-negative, got {self.v_reset}")
        if self.v_threshold <= 0:
            raise ValueError(f"v_threshold must be positive, got {self.v_threshold}")

    @property
    def decay_factor(self) -> float:
        """``alpha``: the fraction of membrane potential surviving one time step."""
        return math.exp(-self.dt / self.tau)


@dataclass(frozen=True)
class SynapseConfig:
    """Synaptic layer geometry.

    ``fan_in`` and ``fan_out`` are logical neuron counts, not tile counts. The layer pads them
    to whole tiles on the host; see ``layer.py`` for where that happens.
    """

    fan_in: int
    fan_out: int

    def __post_init__(self):
        if self.fan_in <= 0 or self.fan_out <= 0:
            raise ValueError(f"fan_in and fan_out must be positive, got {self.fan_in}x{self.fan_out}")
