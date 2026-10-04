"""Linear passive elements: resistor, capacitor, inductor, transformer."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from ..element import Element, through, two_terminal


class Resistor(Element):
    """Linear resistor, ``i = (v_p - v_n) / r``.

    Args:
        nodes: ``(p, n)`` for one device, or a list of them.
        r: Resistance in ohms (scalar or one per device).
    """

    terminals = ("p", "n")
    log_r: Array

    def __init__(self, nodes: Any, r: Any):
        self.nodes = self._devices(nodes)
        self.log_r = self._log_per_device(r)

    @property
    def r(self) -> Array:
        return jnp.exp(self.log_r)

    @property
    def g(self) -> Array:
        """Conductance ``1/r``."""
        return jnp.exp(-self.log_r)

    def currents(self, v, x, t):
        return through(two_terminal(v) * jnp.exp(-self.log_r)), None


class Capacitor(Element):
    """Linear capacitor, ``q = c (v_p - v_n)``."""

    terminals = ("p", "n")
    log_c: Array

    def __init__(self, nodes: Any, c: Any):
        self.nodes = self._devices(nodes)
        self.log_c = self._log_per_device(c)

    @property
    def c(self) -> Array:
        return jnp.exp(self.log_c)

    def currents(self, v, x, t):
        return None, None

    def charges(self, v, x):
        return through(self.c * two_terminal(v)), None


class Inductor(Element):
    """Inductor with optional series resistance (ESR).

    Internal unknown: the branch current ``i`` (flowing ``p -> n``).
    Equation: ``l di/dt + rs i - (v_p - v_n) = 0``.
    """

    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("i",)
    log_l: Array
    rs: Array

    def __init__(self, nodes: Any, l: Any, rs: Any = 0.0):
        self.nodes = self._devices(nodes)
        self.log_l = self._log_per_device(l)
        self.rs = self._per_device(rs)

    @property
    def l(self) -> Array:
        return jnp.exp(self.log_l)

    def currents(self, v, x, t):
        i = x[..., 0]
        return through(i), (self.rs * i - two_terminal(v))[..., None]

    def charges(self, v, x):
        return None, self.l[..., None] * x


class Transformer(Element):
    """Two magnetically coupled inductors (mutual inductance ``k sqrt(l1 l2)``).

    Terminals ``(p1, n1, p2, n2)``; internal unknowns are the winding
    currents ``(i1, i2)``, each flowing ``p -> n``. Dot convention: both
    windings' dots are on ``p``.
    """

    terminals = ("p1", "n1", "p2", "n2")
    n_internal = 2
    internal_names = ("i1", "i2")
    log_l1: Array
    log_l2: Array
    k: Array

    def __init__(self, nodes: Any, l1: Any, l2: Any, k: Any = 1.0):
        self.nodes = self._devices(nodes)
        self.log_l1 = self._log_per_device(l1)
        self.log_l2 = self._log_per_device(l2)
        self.k = self._per_device(k)

    @property
    def l1(self) -> Array:
        return jnp.exp(self.log_l1)

    @property
    def l2(self) -> Array:
        return jnp.exp(self.log_l2)

    def currents(self, v, x, t):
        i1, i2 = x[..., 0], x[..., 1]
        I = jnp.stack([i1, -i1, i2, -i2], axis=-1)
        return I, -jnp.stack([v[..., 0] - v[..., 1], v[..., 2] - v[..., 3]], axis=-1)

    def charges(self, v, x):
        i1, i2 = x[..., 0], x[..., 1]
        m = self.k * jnp.sqrt(self.l1 * self.l2)
        return None, jnp.stack([self.l1 * i1 + m * i2, m * i1 + self.l2 * i2], -1)
