"""Differentiable waveform measurements (the equivalent of SPICE ``.meas``).

Every function takes plain arrays (e.g. ``sol.t`` and ``sol.v("out")``) and is
differentiable with respect to the waveform, so a design spec such as a
propagation delay, a ring-oscillator frequency or a phase margin can be used
directly as a loss.

Crossing times use linear interpolation between samples. *Which* sample
interval contains the crossing is a discrete choice, so gradients describe how
the crossing moves within that interval; they are exact as long as the
waveform is resolved by the time grid.

Measurements that may not exist (no crossing, no unity-gain point) return
``nan`` rather than raising, so they stay usable inside `jit` and `vmap`.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .analysis import Solution
from .circuit import Circuit

Direction = Literal["rise", "fall", "both"]

# =============================================================================
# Crossings
# =============================================================================


def _segment_crossings(t: Array, y: Array, level: Array | float,
                       direction: Direction) -> tuple[Array, Array]:
    """Interpolated crossing time in every interval, and which ones cross."""
    s = jnp.asarray(y) - level
    a, b = s[:-1], s[1:]
    rise = (a < 0) & (b >= 0)
    fall = (a > 0) & (b <= 0)
    hit = {"rise": rise, "fall": fall, "both": rise | fall}[direction]
    denom = jnp.where(hit, a - b, 1.0)  # avoid 0/0 where unused
    frac = jnp.where(hit, a / denom, 0.0)
    return t[:-1] + frac * (t[1:] - t[:-1]), hit


def crossings(t: Array, y: Array, level: Array | float,
              direction: Direction = "both") -> Array:
    """All crossing times of `level`, with ``nan`` in non-crossing slots.

    Returns an array of length ``len(t) - 1`` (fixed shape, so it works under
    `jit`); use ``jnp.isfinite`` to select the crossings.
    """
    tc, hit = _segment_crossings(t, y, level, direction)
    return jnp.where(hit, tc, jnp.nan)


def crossing(t: Array, y: Array, level: Array | float,
             direction: Direction = "rise", *, k: int = 0,
             after: Array | float | None = None) -> Array:
    """Time of the `k`-th crossing of `level` (counting from 0), optionally
    only counting crossings later than `after`. ``nan`` if there is none.
    """
    tc, hit = _segment_crossings(t, y, level, direction)
    if after is not None:
        hit = hit & (tc > after)
    nth = hit & (jnp.cumsum(hit) == k + 1)
    return jnp.where(jnp.any(nth), jnp.sum(jnp.where(nth, tc, 0.0)), jnp.nan)


def delay(t: Array, trig: Array, targ: Array, trig_level: float, *,
          targ_level: float | None = None, trig_direction: Direction = "rise",
          targ_direction: Direction = "both", k: int = 0) -> Array:
    """``.meas TRIG ... TARG``: time from the `k`-th crossing of `trig` to the
    next crossing of `targ` after it (e.g. an inverter's propagation delay).
    """
    t0 = crossing(t, trig, trig_level, trig_direction, k=k)
    level = trig_level if targ_level is None else targ_level
    return crossing(t, targ, level, targ_direction, after=t0) - t0


def period(t: Array, y: Array, level: Array | float | None = None,
           direction: Direction = "rise", skip: int = 1) -> Array:
    """Mean period of an oscillation from its `direction` crossings of
    `level` (default: the mid-swing ``(min + max) / 2``), ignoring the first
    `skip` crossings (start-up). ``nan`` with fewer than two crossings.
    """
    level = 0.5 * (jnp.min(y) + jnp.max(y)) if level is None else level
    tc, hit = _segment_crossings(t, y, level, direction)
    hit = hit & (jnp.cumsum(hit) > skip)
    n = jnp.sum(hit)
    first = jnp.min(jnp.where(hit, tc, jnp.inf))
    last = jnp.max(jnp.where(hit, tc, -jnp.inf))
    return jnp.where(n >= 2, (last - first) / jnp.maximum(n - 1, 1), jnp.nan)


def frequency(t: Array, y: Array, **kwargs) -> Array:
    """``1 / period(t, y, **kwargs)``."""
    return 1.0 / period(t, y, **kwargs)


# =============================================================================
# Step-response metrics
# =============================================================================


def _swing(y: Array, initial, final) -> tuple[Array, Array]:
    y0 = y[0] if initial is None else jnp.asarray(initial, float)
    y1 = y[-1] if final is None else jnp.asarray(final, float)
    return y0, y1


def rise_time(t: Array, y: Array, low: float = 0.1, high: float = 0.9,
              initial: float | None = None, final: float | None = None) -> Array:
    """Time to go from `low` to `high` of the swing (default 10%-90%).

    The swing runs from `initial` to `final` (default: first and last
    samples). Works for falling edges too (then it is the fall time).
    """
    y0, y1 = _swing(y, initial, final)
    u = (y - y0) / (y1 - y0)  # every edge now rises from 0 to 1
    t_low = crossing(t, u, low, "rise")
    return crossing(t, u, high, "rise", after=t_low) - t_low


fall_time = rise_time
"""Alias of `rise_time`; the swing direction is taken from the data."""


def overshoot(y: Array, initial: float | None = None,
              final: float | None = None) -> Array:
    """Peak overshoot as a fraction of the swing (0.1 = 10%)."""
    y0, y1 = _swing(y, initial, final)
    u = (y - y0) / (y1 - y0)
    return jnp.max(u) - 1.0


def settling_time(t: Array, y: Array, tol: float = 0.02,
                  initial: float | None = None, final: float | None = None,
                  start: float | None = None) -> Array:
    """Time (from `start`, default ``t[0]``) after which `y` stays within
    ``tol`` (fraction of the swing) of `final`.
    """
    y0, y1 = _swing(y, initial, final)
    err = jnp.abs((y - y1) / (y1 - y0)) - tol
    tc, hit = _segment_crossings(t, err, 0.0, "fall")
    last = jnp.max(jnp.where(hit, tc, -jnp.inf))
    start = t[0] if start is None else start
    settled_from_start = jnp.all(err <= 0)
    return jnp.where(settled_from_start, 0.0,
                     jnp.where(jnp.isfinite(last), last - start, jnp.nan))


# =============================================================================
# Integrals, power and energy
# =============================================================================


def integral(t: Array, y: Array) -> Array:
    """Trapezoidal integral of `y` over `t` (along the last axis)."""
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * jnp.diff(t), axis=-1)


def average(t: Array, y: Array) -> Array:
    """Time average of `y`."""
    return integral(t, y) / (t[-1] - t[0])


def rms(t: Array, y: Array) -> Array:
    """Root-mean-square of `y` over time."""
    return jnp.sqrt(average(t, y**2))


def power(circuit: Circuit, sol: Solution, device: str) -> Array:
    """Instantaneous power *absorbed* by `device`, ``sum_k v_k i_k``.

    Uses the device's own `currents` and `charges` (displacement current
    ``dq/dt`` is differentiated numerically along ``sol.t``), so it works for
    any element, including your own. A source delivering power is negative.
    """
    group, idx = circuit.layout.device(device)
    element = circuit.elements[group]
    nodes = element.node_array()
    internals = circuit.layout.group_internals(group)

    def terminal_flows(z, t):
        # evaluate the whole group (parameters are per-device arrays), then
        # pick this device
        v = jnp.concatenate([z[: circuit.n_nodes], jnp.zeros(1)])[nodes]
        x = z[internals]
        current, _ = element.currents(v, x, t)
        charge, _ = element.charges(v, x)
        zero = jnp.zeros_like(v)
        current = zero if current is None else jnp.broadcast_to(current, v.shape)
        charge = zero if charge is None else jnp.broadcast_to(charge, v.shape)
        return v[idx], current[idx], charge[idx]

    v, current, charge = jax.vmap(terminal_flows)(sol.z, sol.t)
    displacement = jnp.gradient(charge, sol.t, axis=0)
    return jnp.sum(v * (current + displacement), axis=-1)


def energy(circuit: Circuit, sol: Solution, device: str) -> Array:
    """Energy absorbed by `device` over the transient (negative if it is a
    source delivering energy)."""
    return integral(sol.t, power(circuit, sol, device))


# =============================================================================
# Frequency response
# =============================================================================


def _db(h: Array) -> Array:
    return 20 * jnp.log10(jnp.abs(h))


def _phase(h: Array) -> Array:
    return jnp.rad2deg(jnp.unwrap(jnp.angle(h)))


def _log_crossing(freqs: Array, y: Array, level, direction) -> Array:
    """Crossing frequency, interpolated linearly in log-frequency."""
    return 10 ** crossing(jnp.log10(freqs), y, level, direction)


def bandwidth(freqs: Array, h: Array, drop_db: float = 3.0) -> Array:
    """First frequency where ``|h|`` falls `drop_db` below its value at
    ``freqs[0]`` (the -3 dB bandwidth of a low-pass)."""
    db = _db(h)
    return _log_crossing(freqs, db, db[0] - drop_db, "fall")


def unity_gain_frequency(freqs: Array, h: Array) -> Array:
    """Frequency where ``|h|`` falls through 0 dB."""
    return _log_crossing(freqs, _db(h), 0.0, "fall")


def phase_margin(freqs: Array, h: Array) -> Array:
    """``180 + phase(h)`` at the unity-gain frequency, in degrees, for a loop
    gain `h` (phase taken relative to its low-frequency value)."""
    phase = _phase(h) - jnp.round(_phase(h)[0] / 360) * 360
    f_u = jnp.log10(unity_gain_frequency(freqs, h))
    return 180.0 + jnp.interp(f_u, jnp.log10(freqs), phase)


def gain_margin(freqs: Array, h: Array) -> Array:
    """``-|h|`` in dB where the phase crosses -180 degrees."""
    phase = _phase(h) - jnp.round(_phase(h)[0] / 360) * 360
    f_180 = crossing(jnp.log10(freqs), phase, -180.0, "fall")
    return -jnp.interp(f_180, jnp.log10(freqs), _db(h))
