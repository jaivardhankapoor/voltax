"""MOSFET models: EKV (all-region, charge-based) and Shichman-Hodges Level 1.

Both share a `MOSProcess` holding technology parameters. The process is a
*shared* field: every transistor built with the same process object reads the
same scalars, so gradients with respect to process parameters aggregate over
all of them (useful for process-variation inference).

Polarity is a static field. PMOS devices are evaluated in the mirrored
"n-frame" (all voltages negated) and their currents and charges negated back,
so a single set of equations covers both.
"""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..element import Element, positive
from .semiconductor import VT_300K


class MOSProcess(eqx.Module):
    """Shared technology parameters (all scalars).

    Attributes:
        log_kp: log transconductance parameter ``mu Cox`` (A/V^2).
        vth: Threshold voltage magnitude (V).
        n: Subthreshold slope factor (EKV only).
        lam: Channel-length modulation (1/V).
        vt: Thermal voltage (V).
        gamma: Body-effect coefficient (sqrt(V)); Level 1 only.
        phi: Surface potential (V); Level 1 only.
        log_cox: log gate-oxide capacitance per area (F/m^2).
        log_cov: log gate overlap capacitance per width (F/m).
    """

    log_kp: Array
    vth: Array
    n: Array
    lam: Array
    vt: Array
    gamma: Array
    phi: Array
    log_cox: Array
    log_cov: Array

    def __init__(self, kp: Any = 400e-6, vth: Any = 0.4, n: Any = 1.3,
                 lam: Any = 0.04, vt: Any = VT_300K, gamma: Any = 0.0,
                 phi: Any = 0.7, cox: Any = 8.6e-3, cov: Any = 0.3e-9):
        f = lambda x: jnp.asarray(x, dtype=float)  # noqa: E731
        self.log_kp = positive(kp)
        self.vth, self.n, self.lam, self.vt = f(vth), f(n), f(lam), f(vt)
        self.gamma, self.phi = f(gamma), f(phi)
        self.log_cox = positive(cox)
        self.log_cov = positive(cov)

    def replace(self, **values: Any) -> "MOSProcess":
        """Copy with some parameters replaced (physical units)."""
        current = {k: getattr(self, k) for k in
                   ("kp", "vth", "n", "lam", "vt", "gamma", "phi", "cox", "cov")}
        return MOSProcess(**{**current, **values})

    @classmethod
    def nmos(cls, **overrides: Any) -> "MOSProcess":
        """Generic ~130 nm NMOS defaults."""
        return cls(**{"kp": 400e-6, **overrides})

    @classmethod
    def pmos(cls, **overrides: Any) -> "MOSProcess":
        """Generic ~130 nm PMOS defaults."""
        return cls(**{"kp": 200e-6, **overrides})

    @property
    def kp(self) -> Array:
        return jnp.exp(self.log_kp)

    @property
    def cox(self) -> Array:
        return jnp.exp(self.log_cox)

    @property
    def cov(self) -> Array:
        return jnp.exp(self.log_cov)


class _MOSFET(Element):
    """Common fields and helpers; terminals ``(d, g, s, b)``."""

    terminals = ("d", "g", "s", "b")
    shared = ("process",)
    log_w: Array
    log_l: Array
    dvth: Array
    process: MOSProcess
    polarity: Literal["n", "p"] = eqx.field(static=True)

    def __init__(self, nodes: Any, w: Any = 1e-6, l: Any = 0.13e-6,
                 process: MOSProcess | None = None,
                 polarity: Literal["n", "p"] = "n", dvth: Any = 0.0):
        if polarity not in ("n", "p"):
            raise ValueError(f"polarity must be 'n' or 'p', got {polarity!r}")
        self.nodes = self._devices(nodes)
        self.polarity = polarity
        self.log_w = self._log_per_device(w)
        self.log_l = self._log_per_device(l)
        self.dvth = self._per_device(dvth)
        if process is None:
            process = MOSProcess.nmos() if polarity == "n" else MOSProcess.pmos()
        self.process = process

    @property
    def w(self) -> Array:
        return jnp.exp(self.log_w)

    @property
    def l(self) -> Array:
        return jnp.exp(self.log_l)

    @property
    def sign(self) -> float:
        return 1.0 if self.polarity == "n" else -1.0

    def _overlap_charges(self, vd, vg, vs) -> Array:
        cov = self.process.cov * self.w
        q_gs, q_gd = cov * (vg - vs), cov * (vg - vd)
        return jnp.stack([-q_gd, q_gs + q_gd, -q_gs, jnp.zeros_like(q_gs)], axis=-1)

    def _drain_current(self, vd, vg, vs, vb) -> Array:
        """Drain current in the n-frame (override in subclasses)."""
        raise NotImplementedError

    def ids(self, vd: Array, vg: Array, vs: Array, vb: Array) -> Array:
        """Drain current (into the drain) for terminal voltages, per device.

        Handy for I-V curves: ``m.ids(vd, vg, 0.0, 0.0)``.
        """
        s = self.sign
        return s * self._drain_current(s * vd, s * vg, s * vs, s * vb)

    def currents(self, v, x, t):
        vd, vg, vs, vb = (self.sign * v[..., k] for k in range(4))
        ids = self._drain_current(vd, vg, vs, vb)
        zero = jnp.zeros_like(ids)
        return self.sign * jnp.stack([ids, zero, -ids, zero], axis=-1), None


def _soft_abs(x: Array, eps: float = 1e-3) -> Array:
    return jnp.sqrt(x * x + eps * eps) - eps


class EKVMOSFET(_MOSFET):
    """EKV 2.6-style MOSFET, continuous from weak to strong inversion.

    Current (n-frame, voltages relative to bulk, ``U = vt``)::

        v_p = (v_g - vth) / n
        i_f = ln^2(1 + exp((v_p - v_s) / 2U)),   i_r = same with v_d
        I_ds = 2 n kp (W/L) U^2 (i_f - i_r) (1 + lam |v_ds|)

    Charges are the EKV charge-sheet expressions in terms of
    ``x_{f,r} = sqrt(1/4 + i_{f,r})`` with Ward-Dutton drain/source partition,
    a depletion term giving ``C_gb = Cox WL (n-1)/n`` below threshold, and
    linear overlap capacitance. The model is charge-conserving: the terminal
    charges sum to zero for every bias.
    """

    def _drain_current(self, vd, vg, vs, vb):
        p = self.process
        i_f, i_r = self._inversion(vd, vg, vs, vb)
        i_spec = 2 * p.n * p.kp * (self.w / self.l) * p.vt**2
        return i_spec * (i_f - i_r) * (1.0 + p.lam * _soft_abs(vd - vs))

    def _inversion(self, vd, vg, vs, vb) -> tuple[Array, Array]:
        p = self.process
        vp = (vg - vb - p.vth - self.dvth) / p.n
        i_f = jax.nn.softplus((vp - (vs - vb)) / (2 * p.vt)) ** 2
        i_r = jax.nn.softplus((vp - (vd - vb)) / (2 * p.vt)) ** 2
        return i_f, i_r

    def charges(self, v, x):
        p = self.process
        vd, vg, vs, vb = (self.sign * v[..., k] for k in range(4))
        i_f, i_r = self._inversion(vd, vg, vs, vb)
        xf, xr = jnp.sqrt(0.25 + i_f), jnp.sqrt(0.25 + i_r)
        c_ox = p.cox * self.w * self.l
        scale = c_ox * p.n * p.vt
        s = xf + xr
        q_d = -scale * (
            (4 / 15) * (3 * xr**3 + 6 * xr**2 * xf + 4 * xr * xf**2 + 2 * xf**3)
            / s**2 - 0.5
        )
        q_s = -scale * (
            (4 / 15) * (3 * xf**3 + 6 * xf**2 * xr + 4 * xf * xr**2 + 2 * xr**3)
            / s**2 - 0.5
        )
        q_i = q_d + q_s
        q_b = -(p.n - 1) / p.n * (c_ox * (vg - vb) + q_i)
        q_g = -(q_i + q_b)
        intrinsic = jnp.stack([q_d, q_g, q_s, q_b], axis=-1)
        total = intrinsic + self._overlap_charges(vd, vg, vs)
        return self.sign * total, None


class Level1MOSFET(_MOSFET):
    """Shichman-Hodges (SPICE Level 1) square-law MOSFET, smoothed.

    ``I_ds = kp (W/L) (v_ov v_ds - v_ds^2/2)`` in triode and
    ``kp (W/L) v_ov^2 / 2`` in saturation, times ``(1 + lam v_ds)``. The
    overdrive uses ``softplus`` with width `smooth` (V) so the model is
    differentiable through cutoff (``smooth=0`` gives the exact SPICE
    model); drain and source are swapped for ``v_ds < 0``. Body effect uses
    ``vth + gamma (sqrt(phi + v_sb) - sqrt(phi))``, linearized for a
    forward-biased bulk (``v_sb < 0``) as in SPICE. Charges are overlap only.
    """

    smooth: float = eqx.field(static=True, default=0.02)

    def __init__(self, nodes: Any, w: Any = 1e-6, l: Any = 0.13e-6,
                 process: MOSProcess | None = None,
                 polarity: Literal["n", "p"] = "n", dvth: Any = 0.0,
                 smooth: float = 0.02):
        super().__init__(nodes, w, l, process, polarity, dvth)
        self.smooth = smooth

    def _threshold(self, vsb: Array) -> Array:
        p = self.process
        sphi = jnp.sqrt(p.phi)
        # SPICE: sqrt(phi + vsb) under reverse bias, its tangent (clamped at
        # 0) under forward bias (C1 at vsb = 0)
        reverse = jnp.sqrt(p.phi + jnp.maximum(vsb, 0.0))
        forward = jnp.maximum(sphi + jnp.minimum(vsb, 0.0) / (2 * sphi), 0.0)
        root = jnp.where(vsb >= 0, reverse, forward)
        return p.vth + self.dvth + p.gamma * (root - sphi)

    def _forward(self, vgs: Array, vds: Array, vth: Array) -> Array:
        p, s = self.process, self.smooth
        k = p.kp * self.w / self.l
        if s > 0:
            vov = s * jax.nn.softplus((vgs - vth) / s)
        else:
            vov = jnp.maximum(vgs - vth, 0.0)
        vde = jnp.minimum(vds, vov)
        return k * (vov - 0.5 * vde) * vde * (1.0 + p.lam * vds)

    def _drain_current(self, vd, vg, vs, vb):
        vds = vd - vs
        fwd = self._forward(vg - vs, jnp.maximum(vds, 0.0), self._threshold(vs - vb))
        rev = self._forward(vg - vd, jnp.maximum(-vds, 0.0), self._threshold(vd - vb))
        return fwd - rev

    def charges(self, v, x):
        vd, vg, vs, _ = (self.sign * v[..., k] for k in range(4))
        return self.sign * self._overlap_charges(vd, vg, vs), None
