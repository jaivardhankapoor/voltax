"""Verilog-A frontend (voltax.va): preprocessor, parser, compiler, elements.

Small Verilog-A modules are checked against built-in elements and closed
forms; BSIM4 gets fast smoke tests (direct evaluation of the generated code)
and extended tests (circuit solves, reference values from ngspice 42).
"""

import os
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import voltax as vx
from voltax.va import audit
from voltax.va.parser import parse
from voltax.va.preprocess import VAError, preprocess

# =============================================================================
# Preprocessor and parser
# =============================================================================


def _text(src, **kw):
    return "\n".join(ln.text for ln in preprocess(src, **kw))


def test_macros_with_arguments_and_nesting():
    src = "`define SQ(x) ((x)*(x))\n`define TWO 2\ny = `SQ(`TWO + a);"
    out = _text(src)
    assert "y = ((2 + a)*(2 + a));" in out


def test_multiline_define_keeps_line_numbers():
    lines = preprocess("`define F(a) \\\n  a + 1\nx = `F(2);\nz = 3;")
    assert [ln.lineno for ln in lines] == [1, 2, 3, 4]
    assert lines[2].text.strip() == "x = 2 + 1;"
    assert lines[2].macros == ("F",)


def test_ifdef_else_and_defines():
    src = "`ifdef FAST\na=1;\n`elsif MEDIUM\na=2;\n`else\na=3;\n`endif"
    assert "a=1;" in _text(src, defines={"FAST": ""})
    assert "a=2;" in _text(src, defines={"MEDIUM": ""})
    assert "a=3;" in _text(src)


def test_builtin_constants_and_locked_overrides():
    src = '`include "constants.vams"\nx = `P_Q;'
    assert "1.602176462e-19" in _text(src)  # NIST 1998, the LRM default
    assert "1.6021918e-19" in _text(src, defines={"PHYSICAL_CONSTANTS_OLD": ""})
    # an override wins over the file's own `define
    src2 = "`define K 1\nx = `K;"
    assert "x = 2;" in _text(src2, overrides={"K": "2"})


def test_comments_strings_and_continued_strings():
    src = 'a = 1; // c1\n/* multi\nline */ b = "x // y";\n$strobe("p \\\n q");'
    lines = preprocess(src)
    assert lines[0].text.strip() == "a = 1;"
    assert 'b = "x // y";' in lines[2].text
    assert len(lines) == 5


def test_undefined_macro_raises_with_location():
    with pytest.raises(VAError, match=r"<string>:2: undefined macro `NOPE"):
        preprocess("a=1;\nb=`NOPE;")


def test_parser_module_items():
    src = """
module m(p, n);
  inout p, n;
  electrical p, n, x;
  branch (p, x) br;
  (* type="instance" *) parameter real w = 1u from (0:inf);
  parameter integer mode = 0 from [0:2] exclude 1;
  parameter real k = 2.0;
  aliasparam kk = k;
  real a;
  analog function real twice;
    input u; real u;
    twice = 2 * u;
  endfunction
  analog begin : main
    real loc;
    a = k;
    I(br) <+ V(br) / 1k;
    V(x, n) <+ 0;
  end
endmodule
"""
    (m,) = parse(preprocess(src))
    assert m.ports == ["p", "n"] and m.internal_nodes == ["x"]
    assert m.branches == {"br": ("p", "x")}
    assert m.params["w"].instance and m.params["w"].ranges[0][0] == "("
    assert m.params["mode"].type == "integer" and len(m.params["mode"].excludes) == 1
    assert m.aliases == {"kk": "k"} and "twice" in m.functions


# =============================================================================
# Small models vs built-in elements
# =============================================================================

RESISTOR = """
module res(p, n);
  inout p, n; electrical p, n;
  (* type="instance" *) parameter real r = 1k from (0:inf);
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

DIODE = """
`include "disciplines.vams"
module dio(a, c);
  inout a, c; electrical a, c;
  parameter real is = 1e-14 from (0:inf);
  parameter real n = 1.0;
  parameter real cj = 0 from [0:inf);
  parameter real tt = 0 from [0:inf);
  real vd, id;
  analog begin
    vd = V(a, c);
    id = is * (limexp(vd / (n * 0.025852)) - 1.0);
    I(a, c) <+ id + ddt(tt * id + cj * vd);
  end
endmodule
"""

# series R-L with an internal node and a voltage (branch-current) contribution
RL = """
module rl(p, n);
  inout p, n; electrical p, n, mid;
  parameter real r = 10 from (0:inf);
  parameter real l = 1u from (0:inf);
  analog begin
    I(p, mid) <+ V(p, mid) / r;
    V(mid, n) <+ l * ddt(I(mid, n));
  end
endmodule
"""


def test_resistor_matches_builtin():
    R = vx.va.load(RESISTOR).element(name="rva")
    b = vx.CircuitBuilder()
    b.vsource("a", "0", 2.0, name="V1")
    b.add(R(("a", "b"), r=1e3), name="R1")
    b.add(R(("b", "0"), r=3e3), name="R2")
    c = b.build()
    assert set(c.elements) == {"VoltageSource", "rva"}  # both fused in one group
    sol = vx.dc(c)
    assert jnp.isclose(sol.v("b"), 1.5, rtol=1e-12)
    # instance parameters are per-device, log-space and differentiable
    assert jnp.isclose(c.get("R2", "r"), 3e3)
    g = jax.grad(lambda c: vx.dc(c).v("b"))(c)
    d_log_r2 = g.elements["rva"].inst["log_r"][1]
    assert jnp.isclose(d_log_r2, 2.0 * 1e3 * 3e3 / 4e3**2, rtol=1e-8)


def test_diode_with_ddt_matches_builtin_transient():
    D = vx.va.load(DIODE).element({"is": 1e-15, "n": 1.2, "cj": 1e-12, "tt": 1e-9})
    sig = vx.signals.Pulse(0, 1, 1e-9, 1e-9, 1e-9, 5e-9, 20e-9)
    sols = []
    for build in ("va", "builtin"):
        b = vx.CircuitBuilder()
        b.vsource("in", "0", sig, name="V1")
        b.resistor("in", "a", 1e3)
        if build == "va":
            b.add(D(("a", "0")), name="D1")
        else:
            b.diode("a", "0", is_=1e-15, n=1.2, cj=1e-12, tt=1e-9, name="D1")
        sols.append(vx.transient(b.build(), jnp.linspace(0, 20e-9, 201)))
    assert np.max(np.abs(sols[0].v("a") - sols[1].v("a"))) < 1e-12


def test_internal_node_and_branch_current_match_builtin():
    RLm = vx.va.load(RL).element({"r": 10.0, "l": 1e-6})
    dev = RLm(("a", "0"))
    assert dev.internal_names == ("mid", "i(mid,n)")
    sig = vx.signals.Step(0.0, 1.0, 1e-9)
    sols = []
    for build in ("va", "builtin"):
        b = vx.CircuitBuilder()
        b.vsource("a", "0", sig, name="V1")
        if build == "va":
            b.add(RLm(("a", "0")), name="X1")
        else:
            b.resistor("a", "m", 10.0)
            b.inductor("m", "0", 1e-6, name="L1")
        sols.append(vx.transient(b.build(), jnp.linspace(0, 1e-6, 201)))
    i_va = sols[0].i("X1", "i(mid,n)")
    i_ref = sols[1].i("L1")
    assert np.max(np.abs(i_va - i_ref)) < 1e-12
    assert jnp.isclose(i_va[-1], 0.1 * (1 - np.exp(-(1e-6 - 1e-9) * 10 / 1e-6)),
                       rtol=1e-2)


# =============================================================================
# Compiler semantics
# =============================================================================

GUARDED = """
module g(p, n);
  inout p, n; electrical p, n;
  real v;
  analog begin
    v = V(p, n);
    if (v > 0)
      I(p, n) <+ sqrt(v) * 1m;
    else
      I(p, n) <+ 1m * v;
  end
endmodule
"""


def _f(model, params=None, **kw):
    c = model.compile(params or {}, **kw)
    return c, lambda **V: c.f({k: jnp.asarray(v) for k, v in V.items()},
                              {k: 1.0 for k in c.inputs})


def test_traced_branch_is_nan_guarded():
    c, f = _f(vx.va.load(GUARDED))
    assert "rt.guard" in c.source and "jnp.where" in c.source

    def i(v):
        return f(p=v, n=0.0)[0][0]

    for v in (-0.5, 0.25):
        g = jax.grad(i)(v)
        assert jnp.isfinite(g)
    assert jnp.isclose(jax.grad(i)(-0.5), 1e-3)
    assert jnp.isclose(jax.grad(i)(0.25), 1e-3 * 0.5 / 0.5)
    # forward mode too (Newton uses jacfwd)
    assert jnp.isfinite(jax.jacfwd(i)(-0.5))

    # the naive merge, by contrast, poisons the gradient of the untaken branch
    def naive(v):
        return jnp.where(v > 0, jnp.sqrt(v) * 1e-3, 1e-3 * v)

    assert jnp.isnan(jax.grad(naive)(-0.5))


SELECT = """
`define SCALE(x) (2.0 * (x))
module s(p, n);
  inout p, n; electrical p, n;
  parameter integer mode = 0 from [0:2];
  parameter real g0 = 1m;
  parameter integer reps = 3;
  real g; integer k;
  analog function real cube;
    input x; real x;
    cube = x * x * x;
  endfunction
  analog begin
    case (mode)
      0: g = g0;
      1: g = `SCALE(g0);
      default: g = cube(g0 * 10.0);
    endcase
    k = 0;
    for (k = 0; k < reps; k = k + 1)
      g = g + 0.0;
    if (k != reps) $strobe("unreachable");
    I(p, n) <+ g * V(p, n);
  end
endmodule
"""


@pytest.mark.parametrize("mode,gain", [(0, 1e-3), (1, 2e-3), (2, 1e-6)])
def test_static_parameters_fold_branches(mode, gain):
    m = vx.va.load(SELECT)
    c, f = _f(m, {"mode": mode})
    assert "where" not in c.source  # everything static folded away
    assert jnp.isclose(f(p=2.0, n=0.0)[0][0], 2.0 * gain)


def test_traced_model_parameter_keeps_its_branches():
    src = SELECT.replace("parameter real g0 = 1m;", "parameter real g0 = 1m;\n"
                         "  parameter real gmax = 1.0;").replace(
        "I(p, n) <+ g * V(p, n);", "if (g > gmax) g = gmax;\n    I(p, n) <+ g * "
        "V(p, n);")
    m = vx.va.load(src)
    c_static, _ = _f(m)
    c_traced, f = _f(m, differentiable=("gmax",))
    assert "jnp.where" not in c_static.source and "jnp.where" in c_traced.source
    assert [s.kind for s in c_traced.sites if s.deps] == ["clamp"]


def test_equality_branch_limit_mode():
    src = """
module e(p, n);
  inout p, n; electrical p, n;
  real x;
  analog begin
    x = V(p, n);
    if (V(p, n) == 0.0) x = 0.0;   // a guard that patches one point
    I(p, n) <+ 1m * x;
  end
endmodule
"""
    m = vx.va.load(src)
    for mode, slope in (("value", 0.0), ("limit", 1e-3)):
        _, f = _f(m, equality=mode)
        g = jax.grad(lambda v: f(p=v, n=0.0)[0][0])
        assert jnp.isclose(g(0.0), slope)  # only the point itself differs
        assert jnp.isclose(g(0.1), 1e-3)
        assert f(p=0.0, n=0.0)[0][0] == 0.0


def test_equality_detection_conjunctions_and_not_equal():
    src = """
module e2(p, n);
  inout p, n; electrical p, n;
  real x, y;
  analog begin
    x = V(p, n); y = 2 * V(p, n);
    if ((x == 0.0) && (y == 0.0)) x = 0.0; else x = x;
    if (y != 0.0) y = y; else y = 0.0;
    if ((x == 0.0) || (y > 1.0)) y = y;
    I(p, n) <+ 1m * (x + y);
  end
endmodule
"""
    c = vx.va.load(src).compile(equality="limit")
    kinds = [(s.kind, s.detail) for s in c.sites if s.deps]
    assert kinds == [("eq", "special case in the then branch"),
                     ("eq", "special case in the else branch"), ("if", "")]
    g = jax.grad(lambda v: c.f({"p": v, "n": jnp.asarray(0.0)}, {})[0][0])(0.0)
    assert jnp.isclose(g, 3e-3)  # general-branch derivatives at the point


def test_smooth_mode_replaces_clamps():
    src = """
module c(p, n);
  inout p, n; electrical p, n;
  real x;
  analog begin
    x = V(p, n);
    if (x < 0.0) x = 0.0;
    I(p, n) <+ 1m * x;
  end
endmodule
"""
    m = vx.va.load(src)
    _, exact = _f(m)
    c, smooth = _f(m, smooth=0.01)
    assert "rt.smax" in c.source
    assert exact(p=-0.1, n=0.0)[0][0] == 0.0
    v = jnp.linspace(-0.2, 0.2, 9)
    err = jnp.abs(jax.vmap(lambda u: smooth(p=u, n=0.0)[0][0])(v)
                  - 1e-3 * jnp.maximum(v, 0.0))
    assert jnp.max(err) <= 1e-3 * 0.01 * np.log(2) + 1e-15
    # per-line widths: only the listed line is smoothed
    c_line, _ = _f(m, smooth={7: 0.01})
    c_other, _ = _f(m, smooth={6: 0.01})
    assert "rt.smax" in c_line.source and "rt.smax" not in c_other.source


def test_param_given_simparam_ranges_and_unknown_params():
    src = """
module q(p, n);
  inout p, n; electrical p, n;
  parameter real a = 1.0 from (0:10];
  parameter real b = 0.0;
  real g;
  analog begin
    if ($param_given(b)) g = b; else g = a;
    I(p, n) <+ (g + $simparam("gmin", 1.0)) * V(p, n);
  end
endmodule
"""
    m = vx.va.load(src)
    _, f = _f(m)
    assert jnp.isclose(f(p=1.0, n=0.0)[0][0], 1.0 + 1e-12)
    _, f = _f(m, {"b": 3.0}, simparams={"gmin": 0.0})
    assert jnp.isclose(f(p=1.0, n=0.0)[0][0], 3.0)
    with pytest.raises(ValueError, match="outside its range"):
        m.element({"a": 20.0})
    with pytest.warns(UserWarning, match="unknown parameters"):
        m.element({"zz": 1.0})


def test_while_with_traced_condition_unrolls_on_static_counter():
    src = """
module it(p, n);
  inout p, n; electrical p, n;
  real x, xo; integer k;
  analog begin
    x = 0.0; xo = 1.0; k = 0;
    while ((k < 6) && (abs(x - xo) > 1e-3)) begin
      xo = x; x = 0.5 * (x + V(p, n)); k = k + 1;
    end
    I(p, n) <+ x;
  end
endmodule
"""
    c, f = _f(vx.va.load(src))
    # the first test is static (x=0, xo=1); the next five are traced
    assert [s.kind for s in c.sites].count("loop") == 5
    # v=1 never meets the tolerance: 6 iterations; v=0.01 exits after 4
    assert jnp.isclose(f(p=1.0, n=0.0)[0][0], 1.0 - 2.0**-6)
    assert jnp.isclose(f(p=0.01, n=0.0)[0][0], 0.01 * (1.0 - 2.0**-4))
    assert jnp.isclose(jax.grad(lambda v: f(p=v, n=0.0)[0][0])(1.0), 1 - 2.0**-6)


def test_unsupported_constructs_raise():
    bad = """
module u(p, n);
  inout p, n; electrical p, n;
  analog I(p, n) <+ V(p, n) * ddt(V(p, n));
endmodule
"""
    with pytest.raises(VAError, match="ddt"):
        vx.va.load(bad).compile()


# =============================================================================
# Netlist hook
# =============================================================================

SQUARE_LAW = """
module sqmos(d, g, s, b);
  inout d, g, s, b; electrical d, g, s, b;
  parameter integer type = 1 from [-1:1] exclude 0;
  parameter real kp = 200u;
  parameter real vth = 0.4;
  (* type="instance" *) parameter real w = 1u from (0:inf);
  (* type="instance" *) parameter real l = 100n from (0:inf);
  real vgs, vds, vov, id;
  analog begin
    vgs = type * V(g, s); vds = type * V(d, s);
    vov = vgs - vth;
    if (vov < 0) vov = 0;
    if (vds < vov) id = kp * w / l * (vov - 0.5 * vds) * vds;
    else id = 0.5 * kp * w / l * vov * vov;
    I(d, s) <+ type * id;
  end
endmodule
"""


def test_netlist_level_models_hook():
    m = vx.va.load(SQUARE_LAW)
    net = """
Vdd vdd 0 1.0
Vin in 0 1.0
Mn out in 0 0 nch w=1u l=100n
Rl vdd out 10k
.model nch nmos level=99 vth=0.3 kp=100u
"""
    c = vx.parse_netlist(net, models=vx.va.level_models(m, level=99))
    assert "nch_n" in c.elements
    vout = float(vx.dc(c).v("out"))
    # solve the square law by hand: (1 - v)/10k = 1m*(0.7 - v/2) v (triode)
    k = 100e-6 * 10
    a, b_, cc = 0.5 * k, -(k * 0.7 + 1e-4), 1e-4
    v = (-b_ - np.sqrt(b_**2 - 4 * a * cc)) / (2 * a)
    assert np.isclose(vout, v, rtol=1e-9)


# =============================================================================
# BSIM4
# =============================================================================


def load_bsim4(**kw):
    """The BSIM4 Verilog-A (fetched on first use), or skip without network."""
    try:
        return vx.va.load("bsim4", **kw)
    except (RuntimeError, FileNotFoundError) as e:
        pytest.skip(f"BSIM4 Verilog-A unavailable (not bundled; download "
                    f"failed): {e}")


@pytest.fixture(scope="module")
def bsim4():
    return load_bsim4(overrides=vx.va.NGSPICE_BSIM4_OVERRIDES)


@pytest.fixture(scope="module")
def nmos_card():
    _, _, card = vx.va.bundled_card("freepdk45_nmos.inc")
    card = {**vx.va.NGSPICE_BSIM4_DEFAULTS, **card}
    card.pop("level")
    return card


def test_bsim4_compiles_with_expected_topology(bsim4, nmos_card):
    full = bsim4.compile({**nmos_card, "type": 1})
    assert full.terminals == ("d", "g", "s", "b")
    assert full.internal == ("gm", "bi", "sbulk", "dbulk")  # rgatemod=1, rbodymod=1
    intrinsic = bsim4.compile({**nmos_card, "type": 1, "rbodymod": 0,
                               "rgatemod": 0})
    assert intrinsic.internal == ()
    assert not intrinsic.fatal


def test_bsim4_direct_evaluation_matches_ngspice(bsim4, nmos_card):
    """Reference: ngspice 42 BSIM4, same card with igcmod=igbmod=0 and no
    internal nodes, W=90n L=50n, Vgs=Vds=1 V: Id = 8.167597336835004e-05 A."""
    card = {**nmos_card, "type": 1, "igcmod": 0, "igbmod": 0, "rbodymod": 0,
            "rgatemod": 0}
    c = bsim4.compile(card, instance={"w": 90e-9, "l": 50e-9})
    P = {k: 0.0 for k in c.inputs}
    P.update(w=90e-9, l=50e-9, nrd=1.0, nrs=1.0)
    V = {"d": 1.0, "g": 1.0, "s": 0.0, "b": 0.0}

    def ids(V, P):
        return c.f({k: jnp.asarray(v) for k, v in V.items()}, P)[0][0]

    assert np.isclose(float(ids(V, P)), 8.167597336835004e-05, rtol=1e-12)
    # no NaN at Vds = 0, and finite gradients there (bias and W)
    V0 = {**V, "d": 0.0}
    assert float(ids(V0, P)) == 0.0
    g_bias = jax.grad(lambda vd: ids({**V0, "d": vd}, P))(0.0)
    g_w = jax.grad(lambda w: ids(V, {**P, "w": w}))(90e-9)
    assert jnp.isfinite(g_bias) and jnp.isfinite(g_w) and g_w > 0


@pytest.mark.extended
def test_bsim4_circuit_dc_and_gradient(bsim4, nmos_card):
    NCH = bsim4.element(nmos_card, polarity="n", name="nch",
                        differentiable=("vth0",))
    b = vx.CircuitBuilder()
    b.vsource("d", "0", 1.0, name="Vd")
    b.vsource("g", "0", 1.0, name="Vg")
    b.add(NCH(("d", "g", "0", "0"), w=90e-9, l=50e-9), name="M1")
    c = b.build()
    opts = vx.Options(gmin=0.0)
    sol = vx.dc(c, options=opts)
    assert bool(sol.converged)
    # ngspice 42, full card (gate tunneling, gate resistor, body network)
    assert np.isclose(-float(sol.i("Vd")), 8.167597336835e-05, rtol=1e-9)

    def idrain(c):
        return -vx.dc(c, options=opts).i("Vd")

    g = jax.grad(idrain)(c)
    d_vth0 = g.elements["nch_n"].model.values["vth0"]
    h = 1e-5
    group = c.elements["nch_n"]
    fd = (idrain(c.replace("nch_n", group.with_model(vth0=0.4106 + h)))
          - idrain(c.replace("nch_n", group.with_model(vth0=0.4106 - h)))) / (2 * h)
    assert d_vth0 < 0 and np.isclose(d_vth0, fd, rtol=1e-5)


def _nmos_dc_current(element_cls, **inst):
    b = vx.CircuitBuilder()
    b.vsource("d", "0", 1.0, name="Vd")
    b.vsource("g", "0", 1.0, name="Vg")
    b.add(element_cls(("d", "g", "0", "0"), w=1e-6, l=50e-9, **inst), name="M1")
    sol = vx.dc(b.build(), options=vx.Options(gmin=0.0))
    assert bool(sol.converged)
    return -float(sol.i("Vd"))


@pytest.mark.extended
def test_bsim4_tnoimod1_with_explicit_nrd_zero(bsim4, nmos_card):
    """Regression (sky130 decks): ``tnoimod=1, rdsmod=0`` and instance
    ``nrd=nrs=0``. The Verilog-A then creates drain/source prime nodes with a
    zero series conductance, so the channel floats and carries no current;
    ngspice only creates them for noise analysis. Voltax warns, and
    ``force=NGSPICE_BSIM4_FORCE`` gives ngspice's DC model."""
    card = {**nmos_card, "rgatemod": 0, "rbodymod": 0, "rdsmod": 0,
            "tnoimod": 1}
    inst = {"nrd": 0.0, "nrs": 0.0}
    ref = _nmos_dc_current(bsim4.element({**card, "tnoimod": 0}, polarity="n",
                                         name="ref"), **inst)
    assert ref > 1e-5
    with pytest.warns(UserWarning, match="tnoimod=1 and rdsmod=0"):
        floating = _nmos_dc_current(bsim4.element(card, polarity="n",
                                                  name="floating"), **inst)
    assert abs(floating) < 1e-9 * ref
    forced = bsim4.element(card, polarity="n", name="forced",
                           force=vx.va.NGSPICE_BSIM4_FORCE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert np.isclose(_nmos_dc_current(forced, **inst), ref, rtol=1e-12)


@pytest.mark.extended
def test_bsim4_sparse_solver_matches_dense(bsim4, nmos_card):
    _, _, pcard = vx.va.bundled_card("freepdk45_pmos.inc")
    pcard = {**vx.va.NGSPICE_BSIM4_DEFAULTS, **pcard}
    pcard.pop("level")
    N = bsim4.element(nmos_card, polarity="n", name="nch")
    P = bsim4.element(pcard, polarity="p", name="pch")
    assert N.local and P.local
    b = vx.CircuitBuilder()
    b.vsource("vdd", "0", 1.0, name="Vdd")
    b.vsource("in", "0", 0.45, name="Vin")
    nodes = ["in", "n1", "n2"]
    for k in range(2):
        b.add(P((nodes[k + 1], nodes[k], "vdd", "vdd"), w=180e-9, l=50e-9),
              name=f"MP{k}")
        b.add(N((nodes[k + 1], nodes[k], "0", "0"), w=90e-9, l=50e-9),
              name=f"MN{k}")
    c = b.build()
    dense = vx.dc(c, options=vx.Options(solver="dense"))
    sparse = vx.dc(c, options=vx.Options(solver="sparse"))
    assert bool(sparse.converged)
    assert np.allclose(dense.z, sparse.z, rtol=0, atol=1e-12)
    st = vx.sparse.structure(c)
    data = vx.sparse.jacobian(c, dense.z, 0.0, f=1.0, q=0.0, structure=st)
    J = jax.jacfwd(lambda z: c.f(z, 0.0))(dense.z)
    assert np.allclose(st.to_dense(data), J, rtol=1e-12, atol=1e-18)


# Real open-PDK decks (read-only mirrors, see benchmarks/benchmark_pdk_decks.py)
PDK = Path(os.environ.get("VOLTAX_PDK_ROOT", Path.home() / ".cache/voltax/pdk"))
PDK_INVERTERS = {
    # deck, vdd, ngspice 42 v(out) at vin = vdd/2 on the same deck (reltol
    # 1e-9), tolerance. The cards say "version = 4.5", which ngspice runs with
    # its BSIM4.5.0 code; the Verilog-A is BSIM4.8 (gf180mcu: 1.6e-5 V apart).
    "sky130": (
        "google/skywater-pdk-libs-sky130_fd_pr/main",
        lambda root: f'.lib "{root}/models/sky130.lib.spice" tt\n'
        "XMP out in vdd vdd sky130_fd_pr__pfet_01v8 w=2 l=0.15\n"
        "XMN out in 0 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15\n",
        1.8, 0.498914, 1e-5),
    "gf180mcu": (
        "google/globalfoundries-pdk-libs-gf180mcu_fd_pr/main/models/ngspice",
        lambda root: f'.include "{root}/design.ngspice"\n'
        f'.lib "{root}/sm141064.ngspice" typical\n'
        "MP out in vdd vdd pmos_3p3 w=2u l=0.28u\n"
        "MN out in 0 0 nmos_3p3 w=1u l=0.28u\n",
        3.3, 0.468247, 5e-5),
}


@pytest.mark.extended
@pytest.mark.parametrize("pdk", sorted(PDK_INVERTERS))
def test_pdk_inverter_matches_ngspice(pdk):
    path, deck, vdd, ref, tol = PDK_INVERTERS[pdk]
    root = PDK / path
    if not root.exists():
        pytest.skip(f"{root} not found")
    try:
        models = vx.va.ngspice_bsim4_models()
    except (RuntimeError, FileNotFoundError) as e:
        pytest.skip(f"BSIM4 Verilog-A unavailable: {e}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c, info = vx.parse_netlist(f"Vdd vdd 0 {vdd}\nVin in 0 {vdd / 2}\n"
                                   + deck(root), models=models, return_info=True)
    assert not any("EKV" in d.implementation for d in info.devices.values())
    sol = vx.dc(c, options=vx.Options(gmin=0.0))
    assert bool(sol.converged)
    assert abs(float(sol.v("out")) - ref) < tol


@pytest.mark.extended
def test_bsim4_static_audit_finds_mode_swap(bsim4, nmos_card):
    rep = audit.static_report(bsim4, nmos_card)
    lines = {s.line for s in rep.select("branch-on-bias")}
    src = {s.line: s.source for s in rep.select("branch-on-bias")}
    mode = [n for n in lines if "vds >= 0.0" in src[n]]
    assert mode, "the source/drain mode swap must be a bias branch"
    assert any(s.kind == "eq" for s in rep.select("branch-on-bias"))


def test_fetch_cache_checksum_and_errors(tmp_path, monkeypatch):
    import importlib

    fetch_mod = importlib.import_module("voltax.va.fetch")

    src = tmp_path / "remote.va"
    src.write_text(RESISTOR)
    good = fetch_mod.Remote(src.as_uri(), fetch_mod._sha256(src.read_bytes()),
                            "VOLTAX_TEST_VA", "test licence")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setitem(fetch_mod.REMOTES, "testres", good)
    with pytest.warns(UserWarning, match="test licence"):
        path = vx.va.fetch("testres")
    assert path == tmp_path / "cache" / "voltax" / "va" / "testres.va"
    assert vx.va.load("testres").name == "res"  # from the cache now
    # an explicit environment variable wins
    other = tmp_path / "other.va"
    other.write_text(RESISTOR.replace("module res", "module res2"))
    monkeypatch.setenv("VOLTAX_TEST_VA", str(other))
    assert vx.va.load("testres").name == "res2"
    monkeypatch.delenv("VOLTAX_TEST_VA")
    # a corrupted download is rejected; a failed one explains what to do
    bad = fetch_mod.Remote(src.as_uri(), "0" * 64, "VOLTAX_TEST_VA", "x")
    monkeypatch.setitem(fetch_mod.REMOTES, "testres", bad)
    with pytest.raises(RuntimeError, match="checksum mismatch.*VOLTAX_TEST_VA"):
        vx.va.fetch("testres", force=True)
    gone = fetch_mod.Remote((tmp_path / "nope.va").as_uri(), "0" * 64,
                            "VOLTAX_TEST_VA", "x")
    monkeypatch.setitem(fetch_mod.REMOTES, "testres", gone)
    with pytest.raises(RuntimeError, match="could not download.*vx.va.load"):
        vx.va.fetch("testres", force=True)


def test_bsim4_is_not_bundled():
    assert not list(vx.va.MODELS_DIR.glob("*.va"))
    assert "bsim4" in vx.va.fetch.__globals__["REMOTES"]


def test_element_set_get_and_fusion():
    R = vx.va.load(RESISTOR).element()
    r = R([("a", "0"), ("b", "0")], r=[1e3, 2e3])
    r2 = r.set(1, r=5e3)
    assert jnp.allclose(r2.get("r"), jnp.array([1e3, 5e3]))
    with pytest.raises(AttributeError):
        r.set(0, nope=1.0)
    assert eqx.tree_equal(r.model, R(("x", "0")).model)  # shared default model
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R(("a", "0"), r=10.0)
