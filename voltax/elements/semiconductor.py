"""Junction devices: diode and bipolar transistor."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..element import Element, through, two_terminal

VT_300K = 0.025852
"""Thermal voltage kT/q at 300 K, in volts."""


def explin(u: Array, u_max: float = 40.0) -> Array:
    """``exp(u)`` continued linearly above `u_max`.

    Keeps Newton iterates finite when a junction is transiently forward
    biased by volts; exact (and C1-smooth) wherever ``u <= u_max``.
    """
    e = jnp.exp(jnp.minimum(u, u_max))
    return jnp.where(u > u_max, e * (1.0 + u - u_max), e)


class Diode(Element):
    """Shockley diode with optional junction and diffusion charge.

    ``i = is (exp(v / (n vt)) - 1)``, ``q = tt i + cj v`` with
    ``v = v_anode - v_cathode``.

    Args:
        nodes: ``(anode, cathode)`` or a list of them.
        is_: Saturation current (A).
        n: Emission coefficient.
        cj: Junction capacitance (F), linear.
        tt: Transit time (s), sets diffusion charge.
        vt: Thermal voltage (V).
    """

    terminals = ("a", "k")
    log_is: Array
    n: Array
    cj: Array
    tt: Array
    vt: Array

    def __init__(self, nodes: Any, is_: Any = 1e-14, n: Any = 1.0, cj: Any = 0.0,
                 tt: Any = 0.0, vt: Any = VT_300K):
        self.nodes = self._devices(nodes)
        self.log_is = self._log_per_device(is_)
        self.n = self._per_device(n)
        self.cj = self._per_device(cj)
        self.tt = self._per_device(tt)
        self.vt = self._per_device(vt)

    @property
    def is_(self) -> Array:
        return jnp.exp(self.log_is)

    def current(self, vd: Array) -> Array:
        return self.is_ * (explin(vd / (self.n * self.vt)) - 1.0)

    def currents(self, v, x, t):
        return through(self.current(two_terminal(v))), None

    def charges(self, v, x):
        vd = two_terminal(v)
        return through(self.tt * self.current(vd) + self.cj * vd), None


class BJT(Element):
    """Ebers-Moll (transport form) bipolar transistor with Early effect.

    Terminals ``(c, b, e)``. For NPN::

        i_f = is (exp(v_be/vt) - 1),   i_r = is (exp(v_bc/vt) - 1)
        i_c = (i_f - i_r)(1 + v_ce/vaf) - i_r/br
        i_b = i_f/bf + i_r/br

    PNP mirrors all voltages and currents. Charges: ``q_be = tf i_f + cje v_be``
    and ``q_bc = tr i_r + cjc v_bc``.
    """

    terminals = ("c", "b", "e")
    log_is: Array
    log_bf: Array
    log_br: Array
    inv_vaf: Array
    tf: Array
    tr: Array
    cje: Array
    cjc: Array
    vt: Array
    polarity: Literal["npn", "pnp"] = eqx.field(static=True)

    def __init__(self, nodes: Any, polarity: Literal["npn", "pnp"] = "npn",
                 is_: Any = 1e-16, bf: Any = 100.0, br: Any = 1.0,
                 vaf: Any = jnp.inf, tf: Any = 0.0, tr: Any = 0.0, cje: Any = 0.0,
                 cjc: Any = 0.0, vt: Any = VT_300K):
        if polarity not in ("npn", "pnp"):
            raise ValueError(f"polarity must be 'npn' or 'pnp', got {polarity!r}")
        self.nodes = self._devices(nodes)
        self.polarity = polarity
        self.log_is = self._log_per_device(is_)
        self.log_bf = self._log_per_device(bf)
        self.log_br = self._log_per_device(br)
        self.inv_vaf = self._per_device(1.0 / jnp.asarray(vaf, dtype=float))
        self.tf, self.tr = self._per_device(tf), self._per_device(tr)
        self.cje, self.cjc = self._per_device(cje), self._per_device(cjc)
        self.vt = self._per_device(vt)

    @property
    def sign(self) -> float:
        return 1.0 if self.polarity == "npn" else -1.0

    def _junctions(self, v: Array) -> tuple[Array, Array, Array, Array]:
        vc, vb, ve = (self.sign * v[..., k] for k in range(3))
        vbe, vbc = vb - ve, vb - vc
        i_f = jnp.exp(self.log_is) * (explin(vbe / self.vt) - 1.0)
        i_r = jnp.exp(self.log_is) * (explin(vbc / self.vt) - 1.0)
        return vbe, vbc, i_f, i_r

    def currents(self, v, x, t):
        vbe, vbc, i_f, i_r = self._junctions(v)
        early = 1.0 + (vbe - vbc) * self.inv_vaf
        ic = (i_f - i_r) * early - i_r * jnp.exp(-self.log_br)
        ib = i_f * jnp.exp(-self.log_bf) + i_r * jnp.exp(-self.log_br)
        return self.sign * jnp.stack([ic, ib, -ic - ib], axis=-1), None

    def charges(self, v, x):
        vbe, vbc, i_f, i_r = self._junctions(v)
        q_be = self.tf * i_f + self.cje * vbe
        q_bc = self.tr * i_r + self.cjc * vbc
        return self.sign * jnp.stack([-q_bc, q_be + q_bc, -q_be], axis=-1), None
