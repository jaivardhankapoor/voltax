"""Analyses: gradients (vs finite differences), transformations, robustness."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import voltax as vx


def rc_ladder(R1=1e3, R2=2e3, C1=1e-9, C2=0.5e-9, Rload=1e4):
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 1.0, name="Vin")
    b.resistor("in", "a", R1, name="R1")
    b.resistor("a", "out", R2, name="R2")
    b.resistor("out", "0", Rload, name="Rload")
    b.capacitor("a", "0", C1, name="C1")
    b.capacitor("out", "0", C2, name="C2")
    return b.build()


def fd_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        g[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return g


def test_matches_previous_implementation():
    """Same RC ladder as the pre-rewrite solver (values recorded from it)."""
    c = rc_ladder()
    ts = jnp.linspace(0, 1e-5, 101)
    sol = vx.transient(c, ts, ic="zero")
    assert jnp.allclose(sol.z[-1], jnp.array([1.0, 0.921635241, 0.766714055,
                                               -7.83647588e-05]), atol=1e-8)
    # the old code stored log-conductances: d/dlog_g = -d/dlog_r
    def loss(log_r):
        circuit = c
        for name, lr in zip(["R1", "R2", "Rload"], log_r):
            circuit = circuit.set(name, r=jnp.exp(lr))
        return vx.transient(circuit, ts, ic="zero").v("out").sum()

    g = jax.grad(loss)(jnp.log(jnp.array([1e3, 2e3, 1e4])))
    assert jnp.allclose(g, -jnp.array([13.35247162, 13.96802873, -11.62111701]),
                        rtol=1e-6)


@pytest.mark.parametrize("method", ["be", "trap"])
def test_transient_gradient_vs_finite_differences(method):
    ts = jnp.linspace(0, 5e-6, 51)

    def loss(log_p):
        R1, R2, C1, C2 = jnp.exp(log_p)
        sol = vx.transient(rc_ladder(R1, R2, C1, C2), ts, ic="zero", method=method)
        return jnp.mean(sol.v("out") ** 2)

    x = np.log([1e3, 2e3, 1e-9, 0.5e-9])
    assert np.allclose(jax.grad(loss)(x), fd_grad(lambda x: float(loss(x)), x),
                       rtol=1e-5, atol=1e-10)


def test_dc_gradient_through_nonlinear_devices():
    def vout(params):
        w, vth = params
        b = vx.CircuitBuilder(nmos=vx.MOSProcess.nmos(vth=vth))
        b.vsource("vdd", "0", 1.2)
        b.vsource("in", "0", 0.55)
        b.resistor("vdd", "out", 20e3)
        b.nmos("out", "in", "0", "0", w=w)
        b.diode("out", "x")
        b.resistor("x", "0", 1e5)
        return vx.dc(b.build()).v("out")

    x = np.array([1.5e-6, 0.42])
    g = jax.grad(vout)(jnp.asarray(x))
    g_fd = fd_grad(lambda x: float(vout(jnp.asarray(x))), x, eps=1e-9)
    assert np.allclose(g, g_fd, rtol=1e-4)


def test_ac_gradient_of_corner_frequency():
    def gain_db(log_rc):
        R, C = jnp.exp(log_rc)
        b = vx.CircuitBuilder()
        b.vsource("in", "0", 0.0, ac=1.0)
        b.resistor("in", "out", R)
        b.capacitor("out", "0", C)
        return vx.ac(b.build(), jnp.array([1e3])).db("out")[0]

    x = np.log([1e3, 1e-7])
    g = jax.grad(gain_db)(jnp.asarray(x))
    # |H| = 1/sqrt(1 + (wRC)^2): d dB / d log R = d dB / d log C
    w_rc = 2 * np.pi * 1e3 * 1e3 * 1e-7
    expected = -20 / np.log(10) * w_rc**2 / (1 + w_rc**2)
    assert np.allclose(g, [expected, expected], rtol=1e-6)


def test_grad_wrt_whole_circuit_pytree():
    c = rc_ladder()
    ts = jnp.linspace(0, 2e-6, 21)
    grads = eqx.filter_grad(lambda c: vx.transient(c, ts, ic="zero").v("out")[-1])(c)
    assert grads.elements["Resistor"].log_r.shape == (3,)
    assert jnp.all(jnp.isfinite(grads.elements["Capacitor"].log_c))


def test_shared_process_gradient_aggregates():
    def vout(log_kp):
        b = vx.CircuitBuilder(nmos=vx.MOSProcess.nmos(kp=jnp.exp(log_kp)))
        b.vsource("vdd", "0", 1.2)
        b.vsource("g", "0", 0.7)
        b.resistor("vdd", "a", 1e4)
        b.resistor("vdd", "b", 1e4)
        b.nmos("a", "g", "0", "0")
        b.nmos("b", "g", "0", "0")
        op = vx.dc(b.build())
        return op.v("a") + op.v("b")

    x = float(np.log(400e-6))
    assert np.isclose(jax.grad(vout)(x), fd_grad(lambda x: float(vout(x[0])),
                                                 np.array([x]))[0], rtol=1e-5)


def test_vmap_over_parameters_and_jit():
    c = rc_ladder()

    @jax.jit
    def v_out(r):
        return vx.dc(c.set("Rload", r=r)).v("out")

    rs = jnp.array([1e3, 1e4, 1e5])
    expected = rs / (rs + 3e3)
    assert jnp.allclose(jax.vmap(v_out)(rs), expected)


def test_gmin_stepping_rescues_cmos_dc():
    b = vx.CircuitBuilder()
    b.vsource("vdd", "0", 1.2)
    for n in ["a", "c"]:
        b.vsource(n, "0", 0.0)
    vx.library.cmos.full_adder(b, "a", "c", "0", "s", "co", "vdd")
    c = b.build()
    plain = vx.dc(c, options=vx.Options(gmin_steps=0))
    rescued = vx.dc(c)
    assert not plain.converged
    assert rescued.converged
    assert rescued.v("s") < 0.01 and rescued.v("co") < 0.01


def test_transient_dc_initial_condition_is_steady():
    c = rc_ladder()
    sol = vx.transient(c, jnp.linspace(0, 1e-6, 11))  # ic="dc"
    assert jnp.allclose(sol.v("out"), 1e4 / 1.3e4)
    assert sol.converged.all()


def test_solution_accessors():
    sol = vx.dc(rc_ladder())
    assert sol["out"] == sol.v("out")
    assert sol["Vin.i"] == sol.i("Vin")
    assert sol.v("0") == 0.0
    assert sol.v().shape == (3,)
    with pytest.raises(KeyError):
        sol.v("nope")


def test_linearize_matches_stamps():
    c = rc_ladder()
    G, C = vx.linearize(c, vx.dc(c).z)
    a, out = c.node("a"), c.node("out")
    assert jnp.isclose(G[a, a], 1 / 1e3 + 1 / 2e3)
    assert jnp.isclose(G[a, out], -1 / 2e3)
    assert jnp.isclose(C[out, out], 0.5e-9)


def test_solve_root_generic():
    def residual(z, p, args):
        return z**2 - p

    z, ok = vx.solve_root(residual, jnp.array([4.0]), jnp.array([1.0]))
    assert ok and jnp.isclose(z[0], 2.0)
    g = jax.grad(lambda p: vx.solve_root(residual, p, jnp.array([1.0]))[0][0])(
        jnp.array([4.0]))
    assert jnp.isclose(g[0], 0.25)  # d sqrt(p)/dp


class Memristor(vx.Element):
    """HP linear-drift memristor: a user-defined element with internal state."""

    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("w",)
    log_ron: jax.Array
    log_roff: jax.Array
    log_k: jax.Array

    def __init__(self, nodes, ron=100.0, roff=16e3, k=1e4):
        self.nodes = self._devices(nodes)
        self.log_ron = self._log_per_device(ron)
        self.log_roff = self._log_per_device(roff)
        self.log_k = self._log_per_device(k)

    def currents(self, v, x, t):
        w = jnp.clip(x[..., 0], 0.0, 1.0)
        r = w * jnp.exp(self.log_ron) + (1 - w) * jnp.exp(self.log_roff)
        i = vx.two_terminal(v) / r
        return vx.through(i), (-jnp.exp(self.log_k) * i)[..., None]  # dw/dt = k i

    def charges(self, v, x):
        return None, x


def test_custom_element_with_internal_state():
    def final_w(log_k):
        b = vx.CircuitBuilder()
        b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1.0))
        b.add(Memristor(("in", "0"), k=jnp.exp(log_k)), name="M1")
        c = b.build()
        ts = jnp.linspace(0, 0.5, 201)
        return vx.transient(c, ts, ic=c.state()).i("M1", "w")[-1]

    w = final_w(jnp.log(1e4))
    assert 0.0 < w < 1.0
    g = jax.grad(final_w)(jnp.log(1e4))
    assert np.isclose(g, fd_grad(lambda x: float(final_w(x[0])),
                                 np.array([np.log(1e4)]), 1e-5)[0], rtol=1e-4)


def test_check_raises_on_singular_circuit():
    b = vx.CircuitBuilder()
    b.vsource("a", "0", 1.0)
    b.vsource("a", "0", 2.0)  # conflicting sources: no solution
    with pytest.raises(RuntimeError, match="did not converge"):
        vx.dc(b.build()).check()
    assert vx.dc(rc_ladder()).check().converged


def test_ac_phase_is_unwrapped():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 0.0, ac=1.0)
    for k in range(4):  # 4 buffered RC poles: phase goes to -360 deg
        b.resistor("in" if k == 0 else f"b{k}", f"n{k}", 1e3)
        b.capacitor(f"n{k}", "0", 1e-9)
        b.vcvs(f"b{k + 1}", "0", f"n{k}", "0", 1.0)
    sol = vx.ac(b.build(), jnp.logspace(3, 8, 200))
    assert sol.phase("b4")[-1] < -300
    assert jnp.all(jnp.diff(sol.phase("b4")) < 1.0)


def schmitt_trigger():
    """Inverting Schmitt trigger: thresholds at +-vmax * R2 / (R1 + R2)."""
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 0.0, name="Vin")
    b.opamp("p", "in", "out", a0=1e3, vmin=-1.0, vmax=1.0)
    b.resistor("out", "p", 10e3)
    b.resistor("p", "0", 10e3)
    return b.build()


def test_dc_sweep_follows_branches_and_shows_hysteresis():
    c = schmitt_trigger()
    up = vx.dc_sweep(c, jnp.linspace(-1, 1, 81), "Vin")
    down = vx.dc_sweep(c, jnp.linspace(1, -1, 81), "Vin", guess=up.z[-1])
    assert up.converged.all() and down.converged.all()
    assert jnp.allclose(up.t, jnp.linspace(-1, 1, 81))
    at_zero_up = up.v("out")[40]
    at_zero_down = down.v("out")[40]
    assert at_zero_up > 0.9 and at_zero_down < -0.9  # two stable states at 0 V
    rise_thr = vx.measure.crossing(up.t, up.v("out"), 0.0, "fall")
    fall_thr = -vx.measure.crossing(-down.t, down.v("out"), 0.0, "rise")
    assert 0.4 < rise_thr < 0.55 and -0.55 < fall_thr < -0.4


def test_dc_sweep_with_callable_and_gradient():
    c = rc_ladder()

    def out_sum(log_r):
        sweep = vx.dc_sweep(c, jnp.array([1e3, 1e4]),
                            lambda c, r: c.set("Rload", r=r * jnp.exp(log_r)))
        return sweep.v("out").sum()

    expected = sum(r / (r + 3e3) for r in (1e3, 1e4))
    assert jnp.isclose(out_sum(0.0), expected)
    g = jax.grad(out_sum)(0.0)
    assert np.isclose(g, fd_grad(lambda x: float(out_sum(x[0])), np.zeros(1))[0],
                      rtol=1e-5)


def test_time_grid_resolves_edges():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Pulse(0, 1, delay=1e-9, rise=1e-11,
                                          fall=1e-11, width=4e-9, period=10e-9))
    b.resistor("in", "out", 1e3)
    b.capacitor("out", "0", 1e-13)
    c = b.build()
    grid = vx.time_grid(c, 6e-9, dt_max=0.5e-9)
    steps = np.diff(np.asarray(grid))
    assert grid[0] == 0 and np.isclose(grid[-1], 6e-9) and steps.max() <= 0.5e-9
    on_ramp = (np.asarray(grid) >= 1e-9) & (np.asarray(grid) <= 1.01e-9)
    assert on_ramp.sum() >= 11  # >= 10 steps across the 10 ps edge
    sol = vx.transient(c, grid, method="trap")
    delay = vx.measure.delay(grid, sol.v("in"), sol.v("out"), 0.5)
    fine = jnp.linspace(0, 6e-9, 60001)
    ref = vx.transient(c, fine, method="trap")
    ref_delay = vx.measure.delay(fine, ref.v("in"), ref.v("out"), 0.5)
    assert len(grid) < 200 and jnp.isclose(delay, ref_delay, rtol=5e-3)
