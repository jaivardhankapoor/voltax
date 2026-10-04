"""Differentiable source waveforms.

A `Signal` is an Equinox module mapping time to a value. All parameters are
array leaves, so gradients flow through amplitudes, delays, frequencies, etc.

Signals are vectorized like elements: inside a source element every leaf has a
leading device axis, and ``signal(t)`` returns one value per device. When you
hand a scalar-parameter signal to a single-device source, the source adds that
axis for you.

Subclass `Signal` and implement ``__call__`` to add a new waveform.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _f(x) -> Array:
    return jnp.asarray(x, dtype=float)


def _np(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def _within(times: np.ndarray, t_stop: float) -> np.ndarray:
    times = np.asarray(times, dtype=float).ravel()
    return np.unique(times[(times >= 0) & (times <= t_stop)])


class Signal(eqx.Module):
    """Base class: ``signal(t) -> value``. Signals add: ``Constant(1) + Sine()``."""

    def __call__(self, t: Array) -> Array:
        raise NotImplementedError

    def breakpoints(self, t_stop: float) -> np.ndarray:
        """Times in ``[0, t_stop]`` where the waveform has a corner (edges of
        ramps). Used by `voltax.time_grid`; needs concrete parameters, so call
        it outside `jit`. Default: none (smooth waveform).
        """
        return np.zeros(0)

    def __add__(self, other: "Signal | float") -> "Sum":
        return Sum((self, as_signal(other)))

    def __radd__(self, other: "Signal | float") -> "Sum":
        return Sum((as_signal(other), self))


class Constant(Signal):
    """A DC value."""

    value: Array

    def __init__(self, value: float | Array = 0.0):
        self.value = _f(value)

    def __call__(self, t: Array) -> Array:
        return self.value


class Step(Signal):
    """Linear ramp from `v0` to `v1`, starting at `delay`, lasting `rise`."""

    v0: Array
    v1: Array
    delay: Array
    rise: Array

    def __init__(self, v0=0.0, v1=1.0, delay=0.0, rise=1e-12):
        self.v0, self.v1, self.delay, self.rise = _f(v0), _f(v1), _f(delay), _f(rise)

    def __call__(self, t: Array) -> Array:
        frac = jnp.clip((t - self.delay) / self.rise, 0.0, 1.0)
        return self.v0 + (self.v1 - self.v0) * frac

    def breakpoints(self, t_stop: float) -> np.ndarray:
        d, r = _np(self.delay), _np(self.rise)
        return _within(np.concatenate([d, d + r]), t_stop)


class Pulse(Signal):
    """SPICE ``PULSE(v1 v2 delay rise fall width period)``."""

    v1: Array
    v2: Array
    delay: Array
    rise: Array
    fall: Array
    width: Array
    period: Array

    def __init__(
        self, v1=0.0, v2=1.0, delay=0.0, rise=1e-9, fall=1e-9, width=1e-6, period=2e-6
    ):
        self.v1, self.v2, self.delay = _f(v1), _f(v2), _f(delay)
        self.rise, self.fall = _f(rise), _f(fall)
        self.width, self.period = _f(width), _f(period)

    def __call__(self, t: Array) -> Array:
        tc = (t - self.delay) % self.period
        up = jnp.clip(tc / (self.rise + 1e-30), 0.0, 1.0)
        t_fall_end = self.rise + self.width + self.fall
        down = jnp.clip((t_fall_end - tc) / (self.fall + 1e-30), 0.0, 1.0)
        value = self.v1 + (self.v2 - self.v1) * jnp.minimum(up, down)
        return jnp.where(t < self.delay, self.v1, value)

    def breakpoints(self, t_stop: float) -> np.ndarray:
        d, r, w, f, per = (np.broadcast_to(_np(x), _np(self.delay).shape)
                           for x in (self.delay, self.rise, self.width,
                                     self.fall, self.period))
        corners = []
        for di, ri, wi, fi, pi in zip(d, r, w, f, per):
            starts = di + pi * np.arange(max(int((t_stop - di) // pi) + 1, 0))
            for offset in (0.0, ri, ri + wi, ri + wi + fi):
                corners.append(starts + offset)
        return _within(np.concatenate(corners) if corners else np.zeros(0), t_stop)


class Sine(Signal):
    """SPICE ``SIN(offset amplitude freq delay damping phase_deg)``."""

    offset: Array
    amplitude: Array
    freq: Array
    delay: Array
    damping: Array
    phase: Array

    def __init__(self, offset=0.0, amplitude=1.0, freq=1.0, delay=0.0, damping=0.0,
                 phase=0.0):
        self.offset, self.amplitude, self.freq = _f(offset), _f(amplitude), _f(freq)
        self.delay, self.damping, self.phase = _f(delay), _f(damping), _f(phase)

    def __call__(self, t: Array) -> Array:
        tau = t - self.delay
        wave = jnp.sin(2 * jnp.pi * self.freq * tau + jnp.deg2rad(self.phase))
        active = jnp.where(tau >= 0, jnp.exp(-self.damping * tau) * wave, 0.0)
        return self.offset + self.amplitude * active


class PWL(Signal):
    """Piecewise-linear waveform through ``(times[k], values[k])``.

    Before the first point the first value holds, after the last point the
    last value, unless `repeat` is set. As in SPICE ``PWL(...) td= r=``:

    Args:
        times, values: Corner points (times increasing).
        delay: Shift the whole waveform right by `delay` (``td=``).
        repeat: Once past the last point, repeat the segment
            ``[repeat, times[-1]]`` periodically (``r=``; must be one of the
            times). Negative (default) means no repetition.
    """

    times: Array
    values: Array
    delay: Array
    repeat: Array

    def __init__(self, times: Sequence[float] | Array, values: Sequence[float] | Array,
                 delay: float | Array = 0.0, repeat: float | Array = -1.0):
        self.times, self.values = _f(times), _f(values)
        self.delay, self.repeat = _f(delay), _f(repeat)

    def __call__(self, t: Array) -> Array:
        tau = t - self.delay
        t_last = self.times[..., -1]
        period = t_last - self.repeat
        safe = jnp.where(period > 0, period, 1.0)
        # periods to step back; at a period boundary (and within rounding of
        # it) keep the left limit, the value at times[-1], as SPICE does
        k = jnp.maximum(jnp.ceil((tau - t_last) / safe - 1e-9), 0.0)
        tau = jnp.where((self.repeat >= 0) & (period > 0), tau - k * safe, tau)
        lead = self.times.shape[:-1]
        if not lead:
            return jnp.interp(tau, self.times, self.values)
        interp = jnp.interp
        for _ in lead:
            interp = jax.vmap(interp)
        return interp(jnp.broadcast_to(tau, lead), self.times, self.values)

    def breakpoints(self, t_stop: float) -> np.ndarray:
        times = np.atleast_2d(np.asarray(self.times, dtype=float))
        delay = np.broadcast_to(_np(self.delay), times.shape[:1])
        repeat = np.broadcast_to(_np(self.repeat), times.shape[:1])
        out = []
        for ts, d, r in zip(times, delay, repeat):
            out.append(ts + d)
            period = ts[-1] - r
            if r >= 0 and period > 0:
                seg = ts[ts >= r] - r
                n = int(max(t_stop - d - ts[-1], 0) // period) + 1
                for k in range(n):
                    out.append(d + ts[-1] + k * period + seg)
        return _within(np.concatenate(out), t_stop)


class Exp(Signal):
    """SPICE ``EXP(v1 v2 td1 tau1 td2 tau2)``: from `v1` towards `v2` with
    time constant `tau1` starting at `td1`, back towards `v1` with `tau2`
    starting at `td2`."""

    v1: Array
    v2: Array
    td1: Array
    tau1: Array
    td2: Array
    tau2: Array

    def __init__(self, v1=0.0, v2=1.0, td1=0.0, tau1=1e-9, td2=None, tau2=None):
        td2 = td1 + tau1 if td2 is None else td2
        tau2 = tau1 if tau2 is None else tau2
        self.v1, self.v2, self.td1 = _f(v1), _f(v2), _f(td1)
        self.tau1, self.td2, self.tau2 = _f(tau1), _f(td2), _f(tau2)

    def __call__(self, t: Array) -> Array:
        d1 = jnp.maximum(t - self.td1, 0.0)
        d2 = jnp.maximum(t - self.td2, 0.0)
        rise = (self.v2 - self.v1) * -jnp.expm1(-d1 / self.tau1)
        fall = (self.v1 - self.v2) * -jnp.expm1(-d2 / self.tau2)
        return self.v1 + rise + fall

    def breakpoints(self, t_stop: float) -> np.ndarray:
        return _within(np.concatenate([_np(self.td1), _np(self.td2)]), t_stop)


class SFFM(Signal):
    """SPICE ``SFFM(vo va fc mdi fs [phase_c phase_s])``, single-frequency FM:
    ``vo + va sin(2 pi fc t + phase_c + mdi sin(2 pi fs t + phase_s))``
    (phases in degrees)."""

    vo: Array
    va: Array
    fc: Array
    mdi: Array
    fs: Array
    phase_c: Array
    phase_s: Array

    def __init__(self, vo=0.0, va=1.0, fc=1.0, mdi=0.0, fs=1.0, phase_c=0.0,
                 phase_s=0.0):
        self.vo, self.va, self.fc, self.mdi = _f(vo), _f(va), _f(fc), _f(mdi)
        self.fs, self.phase_c, self.phase_s = _f(fs), _f(phase_c), _f(phase_s)

    def __call__(self, t: Array) -> Array:
        mod = self.mdi * jnp.sin(2 * jnp.pi * self.fs * t + jnp.deg2rad(self.phase_s))
        return self.vo + self.va * jnp.sin(2 * jnp.pi * self.fc * t
                                           + jnp.deg2rad(self.phase_c) + mod)


class AM(Signal):
    """SPICE ``AM(va vo mf fc td)``, amplitude modulation (ngspice):
    ``va (vo + sin(2 pi mf tau)) sin(2 pi fc tau)`` with ``tau = t - td``,
    zero before `td`."""

    va: Array
    vo: Array
    mf: Array
    fc: Array
    td: Array

    def __init__(self, va=1.0, vo=0.0, mf=1.0, fc=1.0, td=0.0):
        self.va, self.vo, self.mf, self.fc, self.td = (_f(va), _f(vo), _f(mf),
                                                        _f(fc), _f(td))

    def __call__(self, t: Array) -> Array:
        tau = t - self.td
        value = (self.va * (self.vo + jnp.sin(2 * jnp.pi * self.mf * tau))
                 * jnp.sin(2 * jnp.pi * self.fc * tau))
        return jnp.where(tau > 0, value, 0.0)

    def breakpoints(self, t_stop: float) -> np.ndarray:
        return _within(_np(self.td), t_stop)


class Sum(Signal):
    """Sum of signals (what ``a + b`` builds)."""

    terms: tuple[Signal, ...]

    def __call__(self, t: Array) -> Array:
        return sum(term(t) for term in self.terms)

    def breakpoints(self, t_stop: float) -> np.ndarray:
        return _within(np.concatenate([s.breakpoints(t_stop) for s in self.terms]),
                       t_stop)


class Function(Signal):
    """Arbitrary waveform ``fn(t, params)``.

    `fn` is static (hashable, e.g. a module-level function); `params` is a
    differentiable pytree. Inside a batched source every `params` leaf has a
    leading device axis, so write `fn` elementwise.
    """

    params: Any
    fn: Callable[[Array, Any], Array] = eqx.field(static=True)

    def __init__(self, fn: Callable[[Array, Any], Array], params: Any = None):
        self.fn = fn
        self.params = params

    def __call__(self, t: Array) -> Array:
        return self.fn(t, self.params)


def as_signal(value: Signal | float | Array) -> Signal:
    """Wrap a number as a `Constant`; pass signals through."""
    return value if isinstance(value, Signal) else Constant(value)
