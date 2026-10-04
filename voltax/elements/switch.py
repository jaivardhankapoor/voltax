"""Voltage-controlled switch with a smooth on/off transition."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from ..element import Element


class Switch(Element):
    """Voltage-controlled switch (SPICE S element), smoothed.

    Terminals ``(p, n, cp, cn)``. The conductance interpolates log-linearly
    between ``1/roff`` and ``1/ron`` as the control voltage crosses `vth`,
    over a transition width `vwidth`::

        log g = log(1/roff) + sigmoid((v_c - vth) / vwidth) log(roff/ron)
    """

    terminals = ("p", "n", "cp", "cn")
    log_ron: Array
    log_roff: Array
    vth: Array
    vwidth: Array

    def __init__(self, nodes: Any, ron: Any = 1.0, roff: Any = 1e9, vth: Any = 0.5,
                 vwidth: Any = 0.05):
        self.nodes = self._devices(nodes)
        self.log_ron = self._log_per_device(ron)
        self.log_roff = self._log_per_device(roff)
        self.vth = self._per_device(vth)
        self.vwidth = self._per_device(vwidth)

    def conductance(self, vc: Array) -> Array:
        on = jax.nn.sigmoid((vc - self.vth) / self.vwidth)
        return jnp.exp(-self.log_roff + on * (self.log_roff - self.log_ron))

    def currents(self, v, x, t):
        i = self.conductance(v[..., 2] - v[..., 3]) * (v[..., 0] - v[..., 1])
        zero = jnp.zeros_like(i)
        return jnp.stack([i, -i, zero, zero], axis=-1), None
