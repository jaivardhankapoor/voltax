"""Linear controlled sources (SPICE E, G, F, H elements).

All are four-terminal: ``(p, n)`` is the output port, ``(cp, cn)`` the control
port. Current-controlled sources sense the current through a zero-volt
ammeter between ``cp`` and ``cn`` (flowing ``cp -> cn``), so they are fully
self-contained: insert the sense port in series with the branch to measure.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from ..element import Element


def _port(v: Array, a: int, b: int) -> Array:
    return v[..., a] - v[..., b]


def _zeros(x: Array) -> Array:
    return jnp.zeros_like(x)


class VCVS(Element):
    """Voltage-controlled voltage source (E): ``v_out = gain * v_ctrl``."""

    terminals = ("p", "n", "cp", "cn")
    n_internal = 1
    internal_names = ("i",)
    gain: Array

    def __init__(self, nodes: Any, gain: Any):
        self.nodes = self._devices(nodes)
        self.gain = self._per_device(gain)

    def currents(self, v, x, t):
        i = x[..., 0]
        I = jnp.stack([i, -i, _zeros(i), _zeros(i)], axis=-1)
        F = _port(v, 0, 1) - self.gain * _port(v, 2, 3)
        return I, F[..., None]


class VCCS(Element):
    """Voltage-controlled current source (G): ``i_out = gm * v_ctrl``,
    flowing ``p -> n`` through the source.
    """

    terminals = ("p", "n", "cp", "cn")
    gm: Array

    def __init__(self, nodes: Any, gm: Any):
        self.nodes = self._devices(nodes)
        self.gm = self._per_device(gm)

    def currents(self, v, x, t):
        i = self.gm * _port(v, 2, 3)
        return jnp.stack([i, -i, _zeros(i), _zeros(i)], axis=-1), None


class CCCS(Element):
    """Current-controlled current source (F): ``i_out = gain * i_sense``."""

    terminals = ("p", "n", "cp", "cn")
    n_internal = 1
    internal_names = ("i_sense",)
    gain: Array

    def __init__(self, nodes: Any, gain: Any):
        self.nodes = self._devices(nodes)
        self.gain = self._per_device(gain)

    def currents(self, v, x, t):
        i_s = x[..., 0]
        i = self.gain * i_s
        I = jnp.stack([i, -i, i_s, -i_s], axis=-1)
        return I, _port(v, 2, 3)[..., None]


class CCVS(Element):
    """Current-controlled voltage source (H): ``v_out = r * i_sense``."""

    terminals = ("p", "n", "cp", "cn")
    n_internal = 2
    internal_names = ("i", "i_sense")
    r: Array

    def __init__(self, nodes: Any, r: Any):
        self.nodes = self._devices(nodes)
        self.r = self._per_device(r)

    def currents(self, v, x, t):
        i, i_s = x[..., 0], x[..., 1]
        I = jnp.stack([i, -i, i_s, -i_s], axis=-1)
        F = jnp.stack([_port(v, 0, 1) - self.r * i_s, _port(v, 2, 3)], axis=-1)
        return I, F
