"""Each element against closed-form physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import voltax as vx


def build(fn):
    b = vx.CircuitBuilder()
    fn(b)
    return b.build()


# ----------------------------------------------------------------- passive


def test_resistor_divider():
    c = build(lambda b: (b.vsource("in", "0", 3.0, name="V"),
                         b.resistor("in", "mid", 1e3), b.resistor("mid", "0", 2e3)))
    op = vx.dc(c)
    assert op.converged
    assert jnp.isclose(op.v("mid"), 2.0)
    assert jnp.isclose(op.i("V"), -1e-3)  # supply delivers 1 mA (SPICE sign)


def test_rc_step_matches_exponential():
    R, C = 1e3, 1e-9
    c = build(lambda b: (b.vsource("in", "0", 1.0), b.resistor("in", "out", R),
                         b.capacitor("out", "0", C)))
    ts = jnp.linspace(0, 5 * R * C, 2001)
    # consistent start (source already on, capacitor empty): trap is then
    # second-order accurate from the first step
    sol = vx.transient(c, ts, ic=c.state(v={"in": 1.0}), method="trap")
    exact = 1 - jnp.exp(-ts / (R * C))
    assert jnp.max(jnp.abs(sol.v("out")[1:] - exact[1:])) < 1e-5


@pytest.mark.parametrize("method,order", [("be", 1), ("trap", 2)])
def test_integration_order(method, order):
    R, C = 1e3, 1e-9
    c = build(lambda b: (b.vsource("in", "0", 1.0), b.resistor("in", "out", R),
                         b.capacitor("out", "0", C)))
    ic = c.state(v={"in": 1.0})
    errs = []
    for n in (50, 100):
        ts = jnp.linspace(0, R * C, n + 1)
        v = vx.transient(c, ts, ic=ic, method=method).v("out")[-1]
        errs.append(abs(float(v) - (1 - np.exp(-1))))
    assert np.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.15)


def test_rl_current_rise():
    R, L = 10.0, 1e-3
    c = build(lambda b: (b.vsource("in", "0", 1.0), b.resistor("in", "a", R),
                         b.inductor("a", "0", L, name="L1")))
    ts = jnp.linspace(0, 5 * L / R, 2001)
    sol = vx.transient(c, ts, ic=c.state(v={"in": 1.0, "a": 1.0}), method="trap")
    exact = (1 - jnp.exp(-ts * R / L)) / R
    assert jnp.max(jnp.abs(sol.i("L1") - exact)) < 1e-6
    assert jnp.isclose(vx.dc(c).i("L1"), 1 / R)  # inductor is a short at DC


def test_inductor_series_resistance_at_dc():
    c = build(lambda b: (b.vsource("in", "0", 1.0),
                         b.inductor("in", "0", 1e-3, rs=4.0, name="L1")))
    assert jnp.isclose(vx.dc(c).i("L1"), 0.25)


def test_transformer_voltage_ratio():
    # well-coupled transformer, light load: v2/v1 ~= k sqrt(l2/l1)
    c = build(lambda b: (b.vsource("in", "0", 0.0, ac=1.0), b.resistor("in", "p", 1e-3),
                         b.transformer("p", "0", "s", "0", 1e-3, 4e-3, k=0.999),
                         b.resistor("s", "0", 1e6)))
    sol = vx.ac(c, jnp.array([1e4]))
    assert jnp.isclose(jnp.abs(sol.v("s")[0]), 0.999 * 2.0, rtol=1e-3)


# ----------------------------------------------------------------- sources


def test_current_source_sign():
    # SPICE: I1 0 out 1m pushes 1 mA into "out"
    c = build(lambda b: (b.isource("0", "out", 1e-3), b.resistor("out", "0", 1e3)))
    assert jnp.isclose(vx.dc(c).v("out"), 1.0)


def test_signals():
    s = vx.signals
    t = jnp.array([0.0, 0.5, 1.5, 2.5, 3.5])
    pulse = s.Pulse(0, 1, delay=0.0, rise=1.0, fall=1.0, width=1.0, period=10.0)
    assert jnp.allclose(pulse(t), jnp.array([0, 0.5, 1, 0.5, 0]))
    assert jnp.allclose(s.PWL([0, 1], [0, 2])(t), jnp.array([0, 1, 2, 2, 2]))
    assert jnp.isclose(s.Sine(1, 2, freq=1.0)(0.25), 3.0)
    assert jnp.isclose((s.Constant(1.0) + s.Step(0, 1, 0, 1))(0.5), 1.5)
    assert jnp.isclose(s.Function(lambda t, p: p * t, 3.0)(2.0), 6.0)


def test_batched_pwl_sources():
    c = vx.Circuit({"V": vx.VoltageSource([(0, -1), (1, -1)],
                                          vx.signals.PWL([[0, 1], [0, 1]],
                                                         [[0, 1], [0, 2]]))}, 2)
    sol = vx.transient(c, jnp.linspace(0, 1, 5), ic="dc")
    assert jnp.allclose(sol.v()[-1], jnp.array([1.0, 2.0]))


# ------------------------------------------------------ controlled sources


def test_controlled_sources():
    def circuit(b):
        b.vsource("c", "0", 0.5)
        b.vcvs("e", "0", "c", "0", 3.0)
        b.resistor("e", "0", 1e3)
        b.vccs("0", "g", "c", "0", 2e-3)  # pushes 1 mA into g
        b.resistor("g", "0", 1e3)
        b.vsource("x", "0", 1.0)
        b.resistor("x", "xs", 1e3)  # 1 mA through the sense ports below
        b.cccs("0", "f", "xs", "y", 2.0)
        b.resistor("f", "0", 1e3)
        b.ccvs("h", "0", "y", "0", 500.0)
        b.resistor("h", "0", 1e3)

    op = vx.dc(build(circuit))
    assert jnp.isclose(op.v("e"), 1.5)
    assert jnp.isclose(op.v("g"), 1.0)
    assert jnp.isclose(op.v("f"), 2.0)
    assert jnp.isclose(op.v("h"), 0.5)


# ------------------------------------------------------------ op-amps


def test_ideal_opamp_inverting_amplifier():
    c = build(lambda b: (b.vsource("in", "0", 0.3), b.resistor("in", "m", 1e3),
                         b.resistor("m", "out", 4.7e3), b.ideal_opamp("0", "m", "out")))
    assert jnp.isclose(vx.dc(c).v("out"), -0.3 * 4.7)


def test_behavioral_opamp_gain_bandwidth_and_rails():
    a0, gbw = 1e5, 1e6
    c = build(lambda b: (b.vsource("in", "0", 0.0, ac=1.0, name="Vin"),
                         b.opamp("in", "0", "out", a0=a0, gbw=gbw),
                         b.resistor("out", "0", 1e4)))
    sol = vx.ac(c, jnp.array([1e-3, gbw]))  # pole at gbw / a0 = 10 Hz
    assert jnp.isclose(jnp.abs(sol.v("out")[0]), a0, rtol=1e-3)
    assert jnp.isclose(jnp.abs(sol.v("out")[1]), 1.0, rtol=1e-3)
    driven = c.set("Vin", value=1.0)  # hard overdrive saturates at the rail
    assert jnp.isclose(vx.dc(driven).v("out"), 15.0, atol=1e-2)


# ------------------------------------------------------- semiconductors


def test_diode_shockley():
    is_, n = 1e-14, 1.5
    d = vx.Diode((0, 1), is_=is_, n=n)
    v = jnp.array([[0.6, 0.0]])
    I, _ = d.currents(v, None, 0.0)
    assert jnp.isclose(I[0, 0], is_ * jnp.expm1(0.6 / (n * vx.VT_300K)))
    assert jnp.isclose(I.sum(), 0.0)


def test_diode_circuit_solves_kvl():
    c = build(lambda b: (b.vsource("in", "0", 5.0, name="V"),
                         b.resistor("in", "a", 1e3), b.diode("a", "0", is_=1e-14)))
    op = vx.dc(c)
    i = -op.i("V")
    assert jnp.isclose(i, (5.0 - op.v("a")) / 1e3)
    assert jnp.isclose(i, 1e-14 * jnp.expm1(op.v("a") / vx.VT_300K), rtol=1e-6)


@pytest.mark.parametrize("polarity", ["npn", "pnp"])
def test_bjt_forward_active(polarity):
    s = 1.0 if polarity == "npn" else -1.0
    q = vx.BJT((0, 1, 2), polarity, is_=1e-16, bf=100.0)
    v = s * jnp.array([[2.0, 0.7, 0.0]])
    I, _ = q.currents(v, None, 0.0)
    ic = 1e-16 * jnp.expm1(0.7 / vx.VT_300K)
    assert jnp.isclose(s * I[0, 0], ic, rtol=1e-6)
    assert jnp.isclose(s * I[0, 1], ic / 100.0, rtol=1e-3)
    assert jnp.isclose(I.sum(), 0.0, atol=1e-18)


# ------------------------------------------------------------- MOSFETs


def test_ekv_strong_inversion_square_law():
    p = vx.MOSProcess.nmos(kp=200e-6, vth=0.4, n=1.0, lam=0.0)
    m = vx.EKVMOSFET((0, 1, 2, 3), w=2e-6, l=1e-6, process=p)
    ids = m.ids(1.5, 1.4, 0.0, 0.0)  # deep saturation, strong inversion
    sq = 0.5 * 200e-6 * 2 * (1.4 - 0.4) ** 2
    assert jnp.isclose(ids[0], sq, rtol=0.05)


def test_ekv_symmetry_and_pmos_mirror():
    n = vx.EKVMOSFET((0, 1, 2, 3))
    p = vx.EKVMOSFET((0, 1, 2, 3), polarity="p")
    vd, vg, vs = 0.3, 0.9, 0.1
    assert jnp.isclose(n.ids(vd, vg, vs, 0.0), -n.ids(vs, vg, vd, 0.0))
    p_same = vx.EKVMOSFET((0, 1, 2, 3), polarity="p", process=n.process)
    assert jnp.isclose(p_same.ids(-vd, -vg, -vs, 0.0), -n.ids(vd, vg, vs, 0.0))
    assert p.ids(0.0, 0.0, 1.2, 1.2)[0] < 0  # PMOS on: current out of drain


def test_ekv_charge_conservation_and_saturation_cgs():
    m = vx.EKVMOSFET((0, 1, 2, 3), w=10e-6, l=1e-6)
    v = jnp.array([[1.2, 1.2, 0.0, 0.0]])
    Q, _ = m.charges(v, None)
    assert jnp.abs(Q.sum()) < 1e-25
    # strong-inversion saturation (charge-sheet / Tsividis):
    #   C_gg = (2/3 + (n-1)/(3n)) C_ox W L + 2 C_ov W
    qg = lambda vg: m.charges(v.at[0, 1].set(vg), None)[0][0, 1]  # noqa: E731
    cgg = jax.grad(qg)(1.2)
    p = m.process
    expected = (2 / 3 + (p.n - 1) / (3 * p.n)) * p.cox * 10e-12 + 2 * p.cov * 10e-6
    assert jnp.isclose(cgg, expected, rtol=0.01)


def test_level1_exact_square_law_and_body_effect():
    p = vx.MOSProcess.nmos(kp=100e-6, vth=0.5, lam=0.02, gamma=0.4, phi=0.7)
    m = vx.Level1MOSFET((0, 1, 2, 3), w=4e-6, l=1e-6, process=p, smooth=0.0)
    k = 100e-6 * 4
    assert jnp.isclose(m.ids(2.0, 1.5, 0.0, 0.0)[0], 0.5 * k * 1.0 * (1 + 0.04))
    assert jnp.isclose(m.ids(0.2, 1.5, 0.0, 0.0)[0], k * (1.0 - 0.1) * 0.2 * 1.004)
    assert m.ids(1.0, 0.4, 0.0, 0.0)[0] == 0.0
    vth_body = 0.5 + 0.4 * (np.sqrt(0.7 + 0.5) - np.sqrt(0.7))
    assert jnp.isclose(m.ids(3.0, 2.0, 0.5, 0.0)[0],
                       0.5 * k * (1.5 - vth_body) ** 2 * (1 + 0.02 * 2.5))


def test_cmos_inverter_vtc():
    def vout(vin):
        b = vx.CircuitBuilder()
        b.vsource("vdd", "0", 1.2)
        b.vsource("in", "0", vin)
        vx.library.cmos.inverter(b, "in", "out", "vdd")
        return vx.dc(b.build()).v("out")

    vtc = jax.vmap(vout)(jnp.array([0.0, 0.6, 1.2]))
    assert vtc[0] > 1.19 and vtc[2] < 0.01 and 0.2 < vtc[1] < 1.0


# --------------------------------------------- switches and trainables


def test_switch():
    c = build(lambda b: (b.vsource("ctl", "0", 0.0, name="Vc"),
                         b.vsource("in", "0", 1.0), b.resistor("in", "out", 1e3),
                         b.switch("out", "0", "ctl", "0", ron=1.0, roff=1e9,
                                  vth=0.5)))
    assert vx.dc(c).v("out") > 0.99
    assert vx.dc(c.set("Vc", value=1.0)).v("out") < 2e-3


@pytest.mark.parametrize("transform", ["log", "softplus", "sigmoid", "linear"])
def test_conductance_transforms_roundtrip(transform):
    g = jnp.array([1e-3, 0.2])
    el = vx.Conductance([(0, 1), (1, -1)], g, transform, g_min=1e-6, g_max=1.0)
    assert jnp.allclose(el.g, g, rtol=1e-6)
    assert jnp.isclose(el.set(0, g=0.5).g[0], 0.5)


def test_nonlinear_resistor_and_capacitor():
    c = build(lambda b: (
        b.vsource("in", "0", 0.2),
        b.add(vx.NonlinearResistor(("in", "0"), lambda v, p: p["a"] * v**3,
                                   {"a": 2.0}), name="B1"),
    ))
    assert jnp.isclose(vx.dc(c).i("V1"), -2.0 * 0.2**3)
    var = vx.NonlinearCapacitor((0, 1), lambda v, p: p * v**2, 3.0)
    Q, _ = var.charges(jnp.array([[0.5, 0.0]]), None)
    assert jnp.isclose(Q[0, 0], 0.75)
