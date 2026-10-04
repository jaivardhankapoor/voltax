"""voltax.measure against closed forms, and its gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import voltax as vx
from voltax import measure

R, C = 1e3, 1e-9
TAU = R * C


def rc_step(r=R, c=C, n=4001):
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 1.0, name="V1")
    b.resistor("in", "out", r, name="R1")
    b.capacitor("out", "0", c, name="C1")
    circuit = b.build()
    ts = jnp.linspace(0, 8 * r * c, n)
    sol = vx.transient(circuit, ts, ic=circuit.state(v={"in": 1.0}), method="trap")
    return circuit, sol


def test_crossings_of_a_sine():
    t = jnp.linspace(0, 3.5, 3501)
    y = jnp.sin(2 * jnp.pi * t)
    assert jnp.isclose(measure.crossing(t, y, 0.5, "rise"), 1 / 12, atol=1e-6)
    assert jnp.isclose(measure.crossing(t, y, 0.5, "fall", k=1), 1 + 5 / 12, atol=1e-6)
    assert jnp.isclose(measure.crossing(t, y, 0.5, after=1.0), 1 + 1 / 12, atol=1e-6)
    assert jnp.isnan(measure.crossing(t, y, 2.0))
    assert int(jnp.sum(jnp.isfinite(measure.crossings(t, y, 0.0, "rise")))) == 3
    assert jnp.isclose(measure.period(t, y), 1.0, atol=1e-6)
    assert jnp.isclose(measure.frequency(t, y, skip=0), 1.0, atol=1e-6)


def test_step_response_metrics_of_rc():
    _, sol = rc_step()
    v = sol.v("out")
    assert jnp.isclose(measure.rise_time(sol.t, v, final=1.0), TAU * np.log(9),
                       rtol=1e-4)
    assert jnp.isclose(measure.crossing(sol.t, v, 0.5), TAU * np.log(2), rtol=1e-4)
    assert jnp.isclose(measure.settling_time(sol.t, v, tol=0.02, final=1.0),
                       TAU * np.log(50), rtol=1e-4)
    assert measure.overshoot(v, final=1.0) < 1e-3
    falling = 1.0 - v
    assert jnp.isclose(measure.fall_time(sol.t, falling, final=0.0),
                       TAU * np.log(9), rtol=1e-4)


def test_delay_between_two_waveforms():
    t = jnp.linspace(0, 1, 1001)
    a = jnp.clip((t - 0.2) * 10, 0, 1)
    b = 1 - jnp.clip((t - 0.35) * 10, 0, 1)
    assert jnp.isclose(measure.delay(t, a, b, 0.5, targ_direction="fall"), 0.15,
                       atol=1e-9)


def test_energy_conservation_charging_a_capacitor():
    # charging C to V through R: the source delivers C V^2, half ends up in C
    circuit, sol = rc_step()
    e_src = measure.energy(circuit, sol, "V1")
    e_r = measure.energy(circuit, sol, "R1")
    e_c = measure.energy(circuit, sol, "C1")
    v_end = sol.v("out")[-1]
    assert jnp.isclose(-e_src, C * v_end, rtol=1e-3)
    assert jnp.isclose(e_c, 0.5 * C * v_end**2, rtol=1e-3)
    assert jnp.isclose(e_src + e_r + e_c, 0.0, atol=1e-3 * C)


def test_average_and_rms():
    t = jnp.linspace(0, 1, 10001)
    y = jnp.sin(2 * jnp.pi * t)
    assert jnp.isclose(measure.average(t, y), 0.0, atol=1e-9)
    assert jnp.isclose(measure.rms(t, y), 1 / np.sqrt(2), rtol=1e-6)


def test_frequency_response_metrics():
    # integrator-like loop gain with two poles: H = A / ((1 + s/w1)(1 + s/w2))
    f = jnp.logspace(0, 9, 2001)
    s = 2j * jnp.pi * f
    a0, p1, p2 = 1e4, 1e2, 1e6
    h = a0 / ((1 + s / (2 * jnp.pi * p1)) * (1 + s / (2 * jnp.pi * p2)))
    # -3 dB (not -3.0103 dB) point of the dominant pole
    assert jnp.isclose(measure.bandwidth(f, h), p1 * np.sqrt(10**0.3 - 1), rtol=1e-3)
    fu = measure.unity_gain_frequency(f, h)
    assert jnp.isclose(jnp.abs(h[jnp.argmin(jnp.abs(f - fu))]), 1.0, rtol=2e-2)
    pm = 180 - np.degrees(np.arctan(fu / p1) + np.arctan(fu / p2))
    assert jnp.isclose(measure.phase_margin(f, h), pm, atol=0.05)
    three = a0 / (1 + s / (2 * jnp.pi * p1)) ** 3
    assert measure.gain_margin(f, three) < 0  # unstable in unity feedback


def test_measurements_are_differentiable():
    def t50(log_r):
        _, sol = rc_step(r=jnp.exp(log_r), n=801)
        return measure.crossing(sol.t, sol.v("out"), 0.5)

    g = jax.grad(t50)(jnp.log(R))
    assert jnp.isclose(g, TAU * np.log(2), rtol=1e-2)  # d t50 / d log R = t50

    def ac_bw(log_c):
        b = vx.CircuitBuilder()
        b.vsource("in", "0", 0.0, ac=1.0)
        b.resistor("in", "out", R)
        b.capacitor("out", "0", jnp.exp(log_c))
        f = jnp.logspace(3, 8, 801)
        return measure.bandwidth(f, vx.ac(b.build(), f).v("out"))

    bw = ac_bw(jnp.log(C))
    assert jnp.isclose(bw, np.sqrt(10**0.3 - 1) / (2 * np.pi * TAU), rtol=1e-3)
    assert jnp.isclose(jax.grad(ac_bw)(jnp.log(C)), -bw, rtol=1e-2)


@pytest.mark.parametrize("fn", [measure.crossing, measure.period])
def test_measurements_work_under_jit_and_vmap(fn):
    t = jnp.linspace(0, 4.5, 901)
    ys = jnp.sin(2 * jnp.pi * t[None] * jnp.array([[1.0], [2.0]]))
    out = jax.jit(jax.vmap(lambda y: fn(t, y, 0.0)))(ys)
    assert out.shape == (2,) and jnp.all(jnp.isfinite(out))


def test_power_uses_each_devices_own_parameters():
    # R1 and R2 are fused into one group; R2 must use its own resistance
    b = vx.CircuitBuilder()
    b.vsource("a", "0", 1.0, name="V1")
    b.resistor("a", "m", 1.0, name="R1")
    b.resistor("m", "0", 3.0, name="R2")
    c = b.build()
    sol = vx.transient(c, jnp.linspace(0, 1, 11))
    assert jnp.allclose(measure.power(c, sol, "R1"), 0.25**2 * 1.0)
    assert jnp.allclose(measure.power(c, sol, "R2"), 0.75**2 / 3.0)
    assert jnp.allclose(measure.power(c, sol, "V1"), -0.25)
