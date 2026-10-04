"""Operational amplifiers: ideal (nullor) and a behavioral macromodel."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from ..element import Element


class IdealOpAmp(Element):
    """Ideal op-amp (nullor): infinite gain, bandwidth and output drive.

    Terminals ``(inp, inn, out)``. The output sources whatever current
    ``i_out`` (from ground into ``out``) enforces ``v_inp = v_inn``; this only
    has a solution with negative feedback.
    """

    terminals = ("inp", "inn", "out")
    n_internal = 1
    internal_names = ("i_out",)

    def __init__(self, nodes: Any):
        self.nodes = self._devices(nodes)

    def currents(self, v, x, t):
        i_out = x[..., 0]
        zero = jnp.zeros_like(i_out)
        I = jnp.stack([zero, zero, -i_out], axis=-1)
        return I, (v[..., 0] - v[..., 1])[..., None]


class OpAmp(Element):
    """Single-pole op-amp macromodel with soft output rails.

    Terminals ``(inp, inn, out)`` (output referenced to ground). An internal
    state ``x`` follows the clipped open-loop target with time constant
    ``tau = a0 / (2 pi gbw)``::

        tau dx/dt + x = clip(a0 (v_inp - v_inn)),
        clip(u) = mid + half * tanh((u - mid) / half)

    with ``mid, half`` the centre and half-span of ``[vmin, vmax]``, and the
    output drives ``out`` through ``rout``. Small-signal: DC gain ``a0``
    (near mid-rail), unity-gain bandwidth ``gbw``.

    Args:
        a0: Open-loop DC gain (V/V).
        gbw: Gain-bandwidth product (Hz).
        rout: Output resistance (ohm).
        vmin, vmax: Output rails (V).
    """

    terminals = ("inp", "inn", "out")
    n_internal = 1
    internal_names = ("x",)
    log_a0: Array
    log_gbw: Array
    log_rout: Array
    vmin: Array
    vmax: Array

    def __init__(self, nodes: Any, a0: Any = 1e5, gbw: Any = 1e6, rout: Any = 1.0,
                 vmin: Any = -15.0, vmax: Any = 15.0):
        self.nodes = self._devices(nodes)
        self.log_a0 = self._log_per_device(a0)
        self.log_gbw = self._log_per_device(gbw)
        self.log_rout = self._log_per_device(rout)
        self.vmin = self._per_device(vmin)
        self.vmax = self._per_device(vmax)

    @property
    def a0(self) -> Array:
        return jnp.exp(self.log_a0)

    @property
    def gbw(self) -> Array:
        return jnp.exp(self.log_gbw)

    @property
    def rout(self) -> Array:
        return jnp.exp(self.log_rout)

    @property
    def tau(self) -> Array:
        return self.a0 / (2 * jnp.pi * self.gbw)

    def _clip(self, u: Array) -> Array:
        mid, half = 0.5 * (self.vmax + self.vmin), 0.5 * (self.vmax - self.vmin)
        return mid + half * jnp.tanh((u - mid) / half)

    def currents(self, v, x, t):
        state = x[..., 0]
        i_out = (v[..., 2] - state) / self.rout
        zero = jnp.zeros_like(i_out)
        I = jnp.stack([zero, zero, i_out], axis=-1)
        F = state - self._clip(self.a0 * (v[..., 0] - v[..., 1]))
        return I, F[..., None]

    def charges(self, v, x):
        return None, self.tau[..., None] * x
