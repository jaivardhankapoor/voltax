"""General-purpose netlist features: expressions, parameterized subcircuits,
includes/libraries, binning, behavioral sources, waveforms, robustness."""

import math
import warnings
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import voltax as vx
from voltax.netlist import NetlistError, parse_expression
from voltax.netlist.expressions import Scope
from voltax.netlist.lexer import read_lines, split_fields


def op(text, **kw):
    return vx.dc(vx.parse_netlist(text, **kw))


# =============================================================================
# Expressions
# =============================================================================


@pytest.mark.parametrize("text,value", [
    ("1+2*3", 7), ("(1+2)*3", 9), ("2*3-4/2", 4), ("7 % 4", 3),
    ("-2**2", -4), ("(-2)**2", 4), ("2^3^2", 512), ("2**-1", 0.5), ("-3+-2", -5),
    ("1 < 2", 1), ("2 <= 1", 0), ("3 >= 3", 1), ("1 == 1", 1), ("1 != 1", 0),
    ("1 && 0", 0), ("1 || 0", 1), ("!0", 1), ("!(2 > 1)", 0),
    ("1 ? 2 : 3", 2), ("0 ? 2 : 3", 3), ("0 ? 1 : 0 ? 2 : 3", 3),
    ("1+1 > 1 ? 5 : 6", 5),
    ("sqrt(16)", 4), ("exp(0)", 1), ("log(exp(2))", 2), ("ln(1)", 0),
    ("log10(1000)", 3), ("pow(2, 10)", 1024), ("pwr(-2, 2)", -4), ("abs(-3)", 3),
    ("min(3, 1, 2)", 1), ("max(3, 1, 2)", 3), ("sin(0)", 0), ("cos(0)", 1),
    ("tan(0)", 0), ("atan(1)", math.pi / 4), ("sinh(0)", 0), ("cosh(0)", 1),
    ("tanh(0)", 0), ("floor(2.7)", 2), ("ceil(2.1)", 3), ("int(-2.7)", -2),
    ("nint(2.5)", 3), ("nint(-2.4)", -2), ("sgn(-3)", -1), ("sign(-3)", -1),
    ("sign(3, -1)", -3), ("if(1 > 2, 5, 6)", 6), ("limit(5, 0, 3)", 3),
    ("limit(-1, 0, 3)", 0), ("u(1)", 1), ("uramp(-2)", 0), ("pi", math.pi),
    ("agauss(1.5, 0.1, 3)", 1.5),
    # SPICE suffixes inside expressions
    ("1k*2", 2000), ("2meg/2", 1e6), ("10u + 5u", 15e-6), ("3p*1", 3e-12),
    ("1mil", 25.4e-6), ("2.5e-3*2", 5e-3), ("10pF*2", 20e-12),
    # delimiters
    ("{1+1}", 2), ("'2*3'", 6), ("{ {2} * 3 }", 6),
])
def test_expression_values(text, value):
    assert Scope().eval(text) == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_parameters_reference_each_other_in_any_order():
    c, info = vx.parse_netlist("""
.param c = 'b * 2' b = {a + 1}
.param a = 1k
R1 n 0 {c}
I1 0 n 1m
""", return_info=True)
    assert info.params == {"c": 2002.0, "b": 1001.0, "a": 1000.0}
    assert jnp.isclose(vx.dc(c).v("n"), 2.002)


def test_unbraced_parameter_expressions():
    c, info = vx.parse_netlist(".param a = 2 b = a * 3 + 1 cc = - a\nR1 n 0 1\n",
                               return_info=True)
    assert info.params == {"a": 2.0, "b": 7.0, "cc": -2.0}


def test_parameter_cycles_and_unknowns_raise():
    with pytest.raises(NetlistError, match=r"parameter cycle: a -> b -> a"):
        vx.parse_netlist(".param a={b} b={a}\nR1 n 0 {a}\n")
    with pytest.raises(NetlistError, match=r"<netlist>:1: .*unknown parameter 'zz'"):
        vx.parse_netlist("R1 n 0 {zz}\n")
    with pytest.raises(NetlistError, match="unknown function"):
        vx.parse_netlist("R1 n 0 {foo(1)}\n")
    with pytest.raises(NetlistError, match="takes 2 arguments"):
        vx.parse_netlist("R1 n 0 {pow(1)}\n")
    with pytest.raises(NetlistError, match="division by zero"):
        vx.parse_netlist("R1 n 0 {1/0}\n")


def test_expression_parser_errors_and_ast():
    assert parse_expression("{a + 1}") == ("bin", "+", ("id", "a"), ("num", 1.0))
    assert parse_expression("v(a, b) * i(Vx)")[2] == ("v", "a", "b")
    for bad in ("1 +", "(1", "1 ? 2", "1 2", "a $ b", ""):
        with pytest.raises(NetlistError):
            parse_expression(bad)


def test_user_functions():
    c, info = vx.parse_netlist("""
.func sq(x) = x * x
.param f(a, b) = a + 2 * b
.param r = {sq(3) + f(1, 2)}
R1 n 0 {r}
""", return_info=True)
    assert info.params["r"] == 14.0


def test_expressions_everywhere():
    """Element values, model cards, instance and source parameters."""
    c = vx.parse_netlist("""
.param vdd=1.2 wn=2u kpn='200u * 1.5' vt={0.25 * 2}
.model nch nmos (level=1 kp={kpn} vto={vt} lambda='0')
V1 d 0 {vdd}
V2 g 0 PULSE(0 {vdd} {1n} 1n 1n {10n/2} 20n)
M1 d g 0 0 nch w={wn} l={wn/2}
C1 g 0 {1f*10}
""")
    group, idx = c.layout.device("M1")
    m = c.elements[group]
    assert jnp.isclose(m.process.kp, 300e-6) and jnp.isclose(m.process.vth, 0.5)
    assert jnp.isclose(m.w[idx], 2e-6) and jnp.isclose(m.l[idx], 1e-6)
    assert jnp.isclose(c.get("C1", "c"), 10e-15)
    g, i = c.layout.device("V2")
    assert jnp.isclose(c.elements[g].value.v2[i], 1.2)
    assert jnp.isclose(c.elements[g].value.width[i], 5e-9)


# =============================================================================
# Parameterized subcircuits
# =============================================================================

DIVIDER = """
.param rtop=3k
.subckt div top bot out params: ra=1k rb={2*ra}
R1 top out {ra}
R2 out bot {rb}
.ends div
.subckt pair in out r=1k
X1 in 0 mid div ra={r}
X2 mid 0 out div ra={r} rb={rtop}
.ends
V1 in 0 3
"""


def test_subckt_parameters_defaults_and_overrides():
    c = vx.parse_netlist(DIVIDER + "X1 in 0 a div\nX2 in 0 b div ra=2k\n"
                         "X3 in 0 c div rb=1k ra=1k\n")
    sol = vx.dc(c)
    assert jnp.isclose(sol.v("a"), 2.0)  # 1k / 2k
    assert jnp.isclose(sol.v("b"), 2.0)  # 2k / 4k: rb default re-evaluated
    assert jnp.isclose(sol.v("c"), 1.5)
    assert jnp.isclose(c.get("X2.R2", "r"), 4e3)


def test_nested_subckt_with_parameter_passing():
    c = vx.parse_netlist(DIVIDER + "Xp in out pair r=500\n")
    assert jnp.isclose(c.get("Xp.X1.R1", "r"), 500.0)
    assert jnp.isclose(c.get("Xp.X1.R2", "r"), 1e3)
    assert jnp.isclose(c.get("Xp.X2.R2", "r"), 3e3)  # global parameter
    # in -> mid: 500 / (1k || (500 + 3k))
    r_low = 1 / (1 / 1e3 + 1 / 3.5e3)
    v_mid = 3 * r_low / (500 + r_low)
    assert jnp.isclose(vx.dc(c).v("Xp.mid"), v_mid)


def test_subckt_defined_after_use_nested_definitions_and_global_nodes():
    c = vx.parse_netlist("""
.global vdd
Vdd vdd 0 2
X1 out wrap
.subckt wrap o
.subckt inner o
R1 vdd o 1k
.ends inner
X1 o inner
R2 o 0 1k
.ends wrap
""")
    assert jnp.isclose(vx.dc(c).v("out"), 1.0)
    assert "vdd" in c.layout.node_names and "X1.vdd" not in c.layout.node_names


def test_subckt_multiplier():
    c = vx.parse_netlist("""
.subckt cell a b
R1 a b 1k
C1 a b 1p
.ends
V1 in 0 1
X1 in 0 cell m=4
""")
    assert jnp.isclose(c.get("X1.R1", "r"), 250.0)
    assert jnp.isclose(c.get("X1.C1", "c"), 4e-12)


def test_subckt_errors():
    with pytest.raises(NetlistError, match="has no parameter 'zz'"):
        vx.parse_netlist(DIVIDER + "X1 in 0 a div zz=1\n")
    with pytest.raises(NetlistError, match=":11: X1: 1 pins given"):
        vx.parse_netlist(DIVIDER.strip() + "\nX1 in div\n")
    with pytest.raises(NetlistError, match="has no .ends"):
        vx.parse_netlist(".subckt a x\nR1 x 0 1\n")
    with pytest.raises(NetlistError, match="nesting deeper"):
        vx.parse_netlist(".subckt a x\nX1 x a\n.ends\nX1 n a\n")


def test_case_insensitive_names():
    c = vx.parse_netlist("""
.SUBCKT Res A B
R1 a b 1K
.ENDS
V1 IN 0 1
X1 in Out RES
r2 OUT 0 1k
""")
    sol = vx.dc(c)
    assert c.layout.node_names == ("IN", "Out")  # first spelling wins
    assert jnp.isclose(sol.v("Out"), 0.5)


# =============================================================================
# .include / .lib
# =============================================================================


def test_include_relative_paths_once_and_recursion(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "params.inc").write_text(".param rr=2k\n")
    (tmp_path / "sub" / "cells.inc").write_text(
        "* cells\n.include params.inc\n.subckt load a\nR1 a 0 {rr}\n.ends\n")
    deck = tmp_path / "top.sp"
    deck.write_text("title line\n.include 'sub/cells.inc'\n.inc sub/cells.inc\n"
                    "V1 a 0 1\nX1 a load\n.end\n")
    c, info = vx.parse_netlist_file(deck, return_info=True)
    assert jnp.isclose(c.get("X1.R1", "r"), 2e3)
    assert [p.split("/")[-1] for p in info.files] == ["cells.inc", "params.inc"]
    # the same text with an explicit base_dir
    c2 = vx.parse_netlist(deck.read_text(), title=True, base_dir=tmp_path)
    assert jnp.isclose(c2.get("X1.R1", "r"), 2e3)

    (tmp_path / "loop.inc").write_text(".include loop2.inc\n")
    (tmp_path / "loop2.inc").write_text(".include loop.inc\n")
    with pytest.raises(NetlistError, match="recursive include"):
        vx.parse_netlist(".include loop.inc\n", base_dir=tmp_path)
    with pytest.raises(NetlistError, match=r"<netlist>:1: file not found"):
        vx.parse_netlist(".include nope.inc\n", base_dir=tmp_path)


LIB = """* corner library
.lib tt
.param vto_n=0.5
.lib 'models.lib' common
.endl tt

.lib ff
.param vto_n=0.4
.lib models.lib common
.endl ff

.lib common
.model nch nmos (level=1 kp=200u vto={vto_n})
.endl common
"""


def test_lib_sections_select_corners(tmp_path):
    (tmp_path / "models.lib").write_text(LIB)
    deck = """
.lib models.lib {corner}
V1 d 0 1
M1 d d 0 0 nch w=1u l=1u
"""
    vths = {}
    for corner in ("tt", "ff", "TT"):
        c = vx.parse_netlist(deck.format(corner=corner), base_dir=tmp_path)
        g, _ = c.layout.device("M1")
        vths[corner] = float(c.elements[g].process.vth)
    assert vths == {"tt": 0.5, "ff": 0.4, "TT": 0.5}
    with pytest.raises(NetlistError, match=r"section 'ss' not found .*ff, tt"):
        vx.parse_netlist(deck.format(corner="ss"), base_dir=tmp_path)
    (tmp_path / "rec.lib").write_text(".lib a\n.lib rec.lib a\n.endl\n")
    with pytest.raises(NetlistError, match="recursive .lib"):
        vx.parse_netlist(".lib rec.lib a\n", base_dir=tmp_path)


# =============================================================================
# Models: binning, scale, options, levels, geometry, hook
# =============================================================================

BINNED = """
.model nch.1 nmos (level=1 kp=100u vto=0.4 lmin=0.1u lmax=1u wmin=0.1u wmax=10u)
.model nch.2 nmos (level=1 kp=300u vto=0.4 lmin=1u lmax=10u wmin=0.1u wmax=10u)
V1 d 0 1
M1 d d 0 0 nch w=1u l=0.5u
M2 d d 0 0 nch w=1u l=2u
M3 d d 0 0 nch w=1u l=1u
M4 d d 0 0 nch w=1u l=0.9995u
"""


def test_binning_selects_by_geometry():
    with pytest.warns(UserWarning, match="within 1% of a model-bin edge"):
        c, info = vx.parse_netlist(BINNED, return_info=True)
    chosen = {d: info.devices[d].card for d in ("M1", "M2", "M3", "M4")}
    # ngspice: lmin - 1nm <= L <= lmax + 1nm, last-defined bin first, so a
    # shared edge (and anything within 1 nm of it) goes to the upper bin
    assert chosen == {"M1": "nch.1", "M2": "nch.2", "M3": "nch.2", "M4": "nch.2"}
    assert info.devices["M2"].bounds == (1e-6, 10e-6, 0.1e-6, 10e-6)
    assert info.devices["M1"].margin == pytest.approx(0.5)
    g1, _ = c.layout.device("M1")
    g2, _ = c.layout.device("M2")
    assert g1 != g2  # one shared process per bin
    assert jnp.isclose(c.elements[g2].process.kp, 300e-6)
    with pytest.raises(NetlistError, match="no bin of model 'nch' covers L=2e-05"):
        vx.parse_netlist(BINNED + "M5 d d 0 0 nch w=1u l=20u\n")


def test_option_scale_and_temperature():
    c = vx.parse_netlist(BINNED.replace("w=1u l=0.5u", "w=1 l=0.5")
                         .split("M2")[0] + ".option scale=1u\n.temp 127\n"
                         ".model dd d is=1e-14\nD1 d 0 dd\n")
    g, i = c.layout.device("M1")
    assert jnp.isclose(c.elements[g].l[i], 0.5e-6)
    vt = 1.38064852e-23 / 1.6021766208e-19 * 400.15  # ngspice's k/q at 127 C
    assert jnp.isclose(c.elements[g].process.vt, vt)
    gd, i = c.layout.device("D1")
    assert jnp.isclose(c.elements[gd].vt[i], vt)


def test_mos_geometry_parameters_aggregated_warning():
    deck = (".model nch nmos level=1\nV1 d 0 1\n"
            + "".join(f"M{k} d d 0 0 nch w=1u l=1u ad=1p as=1p pd=4u ps=4u\n"
                      for k in range(3)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vx.parse_netlist(deck)
    geometry = [x for x in w if "instance parameters" in str(x.message)]
    assert len(geometry) == 1
    assert "['ad', 'as', 'pd', 'ps'] on 3 MOSFET(s)" in str(geometry[0].message)


def test_model_levels_and_summary():
    deck = """
.model n1 nmos level=1 kp=100u
.model n54 nmos level=54 version=4.5 vth0=0.4 k1=0.5
.model p54 pmos (level = 54, version = 4.5)
.model dd d (is=1e-15 n=1.1)
.model qq npn bf=50
.model jj njf
V1 d 0 1
M1 d d 0 0 n1
"""
    c, info = vx.parse_netlist(deck, return_info=True)
    status = {k: (s.status, s.level) for k, s in info.models.items()}
    assert status == {"n1": ("native", "1"), "n54": ("fallback", "54"),
                      "p54": ("fallback", "54"), "dd": ("native", ""),
                      "qq": ("native", ""), "jj": ("unsupported", "")}
    assert info.models["n54"].version == "4.5"
    assert "fallback" in info.model_report()


class FakeBSIM(vx.Level1MOSFET):
    """Stands in for a compiled BSIM4: records what the hook received."""

    seen = []

    @classmethod
    def from_netlist(cls, spec):
        cls.seen.append(spec)
        polarity = "p" if spec.model.type == "pmos" else "n"
        process = vx.MOSProcess(vth=spec.model.params["vth0"])
        return cls(spec.nodes, spec.instance["w"], spec.instance["l"], process,
                   polarity)


def test_model_hook_receives_card_and_instance_parameters():
    FakeBSIM.seen.clear()
    deck = """
.option scale=1e-6
.model nch.1 nmos level=54 version=4.5 vth0=0.45 lmin=0.1u lmax=1u wmax=1e-4
.subckt cell d g
M1 d g 0 0 nch w=2 l=0.5 ad=1 nf=2 sa=0.3
.ends
V1 d 0 1
X1 d d cell m=2
"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a hooked card raises no warnings
        c, info = vx.parse_netlist(deck, models={"NMOS:54": FakeBSIM},
                                   return_info=True)
    (spec,) = FakeBSIM.seen
    assert spec.name == "X1.M1" and spec.nodes == ("d", "d", "0", "0")
    assert spec.model.name == "nch.1" and spec.model.params["vth0"] == 0.45
    assert spec.model.version == "4.5" and spec.model.level == 54
    assert spec.instance["w"] == pytest.approx(2e-6)
    assert spec.instance["ad"] == pytest.approx(1e-12)  # scale**2
    assert spec.instance["nf"] == 2 and spec.instance["sa"] == 0.3
    assert spec.instance["m"] == 2
    assert spec.options["scale"] == 1e-6 and spec.options["temp"] == 27.0
    assert isinstance(c.elements[c.layout.device("X1.M1")[0]], FakeBSIM)
    assert info.devices["X1.M1"].implementation == "hook: FakeBSIM"
    # a plain callable works too, and the most specific key wins
    hits = []
    vx.parse_netlist(deck, models={"nmos": None, "nch": lambda s: (
        hits.append(s.name), FakeBSIM.from_netlist(s))[1]})
    assert hits == ["X1.M1"]


def test_osdi_n_devices_need_a_hook():
    deck = """
.model sg13_nmos psp103va toxo=2.2n
.subckt lvn d g s b w=1u l=0.13u
N1 d g s b sg13_nmos w={w} l={l} mult=1
.ends
V1 d 0 1
X1 d d 0 0 lvn w=2u
"""
    # no type=+-1 on the card: not recognizably a MOSFET, so no fallback
    with pytest.raises(NetlistError, match=r":4: X1.N1: model 'sg13_nmos' .*"
                                           r"models=\{'psp103va'"):
        vx.parse_netlist(deck)
    FakeBSIM.seen.clear()
    hook = {"psp103va": lambda s: FakeBSIM(s.nodes, s.instance["w"], s.instance["l"])}
    c = vx.parse_netlist(deck, models=hook)
    assert c.layout.device("X1.N1")[0] in c.elements
    with pytest.raises(NetlistError, match="no .model found"):
        vx.parse_netlist("N1 a b nomodel\n")


# =============================================================================
# Behavioral sources
# =============================================================================


def test_b_sources_dc_closed_form():
    c = vx.parse_netlist("""
.param gm=1m vt0=0.25
.func sq(x) = x * x
V1 in 0 DC 1.5
B1 0 a I={gm * sq(v(in) - vt0)}
Ra a 0 1k
B2 b 0 V={2 * v(in, a) + 0.1}
E1 c 0 value={tanh(v(in))}
G1 0 d cur='1m * v(b)'
Rd d 0 1k
H1 e 0 POLY(1) V1 0 1k
Re e 0 1
F1 0 f V1 2
Rf f 0 1k
Bi 0 g I={-i(V1)*1k}
Rg g 0 1
""")
    sol = vx.dc(c)
    va = 1e-3 * (1.5 - 0.25) ** 2 * 1e3
    assert jnp.isclose(sol.v("a"), va)
    assert jnp.isclose(sol.v("b"), 2 * (1.5 - va) + 0.1)
    assert jnp.isclose(sol.v("c"), math.tanh(1.5))
    assert jnp.isclose(sol.v("d"), 1e-3 * sol.v("b") * 1e3)
    i_v1 = sol.i("V1")  # current p -> n through V1
    assert jnp.isclose(sol.v("e"), 1e3 * i_v1)
    assert jnp.isclose(sol.v("f"), 2 * i_v1 * 1e3)
    assert jnp.isclose(sol.v("g"), -i_v1 * 1e3)


def test_b_source_is_differentiable_in_parameters():
    c = vx.parse_netlist("""
.param gm=2m
V1 in 0 0.5
B1 0 out I={gm * v(in)**3}
R1 out 0 1k
""")
    group, _ = c.layout.device("B1")
    grad = eqx.filter_grad(lambda c: vx.dc(c).v("out"))(c)
    assert jnp.isclose(grad.elements[group].params["gm"][0], 0.5**3 * 1e3)


def test_b_source_transient_closed_form():
    # v(out) = V0 sin(w t) driven through a B source; RC load with B current
    c = vx.parse_netlist("""
.param w=6.283185307179586e3
B1 in 0 V={sin(w * time)}
B2 0 out I={v(in) * 1m}
R1 out 0 1k
C1 out 0 1u
""")
    ts = jnp.linspace(0, 2e-3, 2001)
    sol = vx.transient(c, ts, method="trap")
    tau, w = 1e-3, 6.283185307179586e3
    t = np.asarray(ts)
    # out' = (sin(w t) - out) / tau, out(0) = 0
    exact = (np.sin(w * t) - w * tau * np.cos(w * t) + w * tau * np.exp(-t / tau)) \
        / (1 + (w * tau) ** 2)
    assert np.max(np.abs(np.asarray(sol.v("out")) - exact)) < 1e-5


def test_b_source_table_and_poly():
    c = vx.parse_netlist("""
V1 in 0 0.75
E1 a 0 TABLE {v(in) * 2} = (0, 0) (1, 10) (2, 30)
G1 0 b POLY(2) in 0 a 0 1m 2m 0 0 1u
Rb b 0 1k
Ec c 0 POLY(1) in 0 3
""")
    sol = vx.dc(c)
    assert jnp.isclose(sol.v("a"), 20.0)  # interp(1.5)
    x1, x2 = 0.75, 20.0
    # p0 + p1 x1 + p2 x2 + p3 x1^2 + p4 x1 x2
    assert jnp.isclose(sol.v("b"), (1e-3 + 2e-3 * x1 + 1e-6 * x1 * x2) * 1e3)
    assert jnp.isclose(sol.v("c"), 2.25)  # POLY(1) with one coefficient = gain


def test_b_source_errors():
    with pytest.raises(NetlistError, match="not connected"):
        vx.parse_netlist("B1 a 0 V={v(nowhere)}\nR1 a 0 1\n")
    with pytest.raises(NetlistError, match="controlling source 'R1' is not a voltage"):
        vx.parse_netlist("B1 a 0 I={i(R1)}\nR1 a 0 1\n")
    with pytest.raises(NetlistError, match="only allowed in behavioral"):
        vx.parse_netlist("R1 a 0 {v(a)}\n")
    with pytest.raises(NetlistError, match="expected `B<name>"):
        vx.parse_netlist("B1 a 0 1\n")


def test_behavioral_source_element_directly():
    def fn(vc, isense, t, p):
        return p["k"] * vc[..., 0]

    b = vx.CircuitBuilder()
    b.vsource("in", "0", 2.0)
    b.add(vx.BehavioralSource(("out", "0", "in"), fn, {"k": 3.0}, output="v"),
          name="B1")
    b.resistor("out", "0", 1.0)
    sol = vx.dc(b.build())
    # SPICE sign: the branch current flows p -> n through the source
    assert jnp.isclose(sol.v("out"), 6.0) and jnp.isclose(sol.i("B1"), -6.0)


# =============================================================================
# Sources
# =============================================================================


def wave_at(text, ts):
    """Voltage of node `a` driven by source `text` at times `ts`."""
    c = vx.parse_netlist(f"V1 a 0 {text}\nR1 a 0 1\n")
    g, _ = c.layout.device("V1")
    sig = c.elements[g].value
    return np.asarray(jnp.stack([sig(t)[0] for t in ts]))


TS = np.linspace(0, 1e-3, 37)


def test_exp_source():
    got = wave_at("EXP(0.2 1 0.1m 0.05m 0.4m 0.1m)", TS)
    t = TS
    want = 0.2 + np.where(t > 0.1e-3, 0.8 * (1 - np.exp(-(t - 0.1e-3) / 0.05e-3)), 0)
    want += np.where(t > 0.4e-3, -0.8 * (1 - np.exp(-(t - 0.4e-3) / 0.1e-3)), 0)
    assert np.allclose(got, want, atol=1e-12)


def test_sffm_and_am_sources():
    t = TS
    got = wave_at("SFFM(0.1 1 10k 2 1k 30 60)", TS)
    want = 0.1 + np.sin(2 * np.pi * 1e4 * t + np.deg2rad(30)
                        + 2 * np.sin(2 * np.pi * 1e3 * t + np.deg2rad(60)))
    assert np.allclose(got, want, atol=1e-12)
    got = wave_at("AM(2 0.5 1k 10k 0.1m)", TS)
    tau = t - 0.1e-3
    want = np.where(tau > 0, 2 * (0.5 + np.sin(2 * np.pi * 1e3 * tau))
                    * np.sin(2 * np.pi * 1e4 * tau), 0)
    assert np.allclose(got, want, atol=1e-12)


def test_pwl_repeat_delay_and_file(tmp_path):
    ts = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]) * 1e-3
    got = wave_at("PWL(0 0 0.1m 1 0.2m 0) r=0 td=0.05m", ts)
    assert np.allclose(got, [0, 0, 0.5, 1, 0.5, 0, 0.5, 1])
    got = wave_at("PWL(0 0 0.1m 1 0.2m 0.5 0.3m 0) r=0.1m", ts * 1.0 + 0.1e-3)
    # repeats [0.1m, 0.3m]: 0.4m -> 0.2m, 0.45m -> 0.25m
    assert np.allclose(got[-3:], [0.75, 0.5, 0.25])
    (tmp_path / "wave.txt").write_text("# t v\n0 0\n1e-3, 2\n\n2e-3 0 ; end\n")
    c = vx.parse_netlist("V1 a 0 PWL FILE='wave.txt'\nR1 a 0 1\n", base_dir=tmp_path)
    g, _ = c.layout.device("V1")
    assert jnp.isclose(c.elements[g].value(1.5e-3)[0], 1.0)
    bp = c.elements[g].value.breakpoints(3e-3)
    assert np.allclose(bp, [0, 1e-3, 2e-3])
    with pytest.raises(NetlistError, match="r=5e-05 is not one of the time points"):
        wave_at("PWL(0 0 1m 1) r=0.05m", ts)


def test_source_argument_orders_and_tran_defaults():
    c = vx.parse_netlist("""
V1 a 0 AC 1 90 DC 0.5
V2 b 0 SIN(0 1 1k) AC 2
V3 c 0 DC 1 PULSE (0 1)
V4 d 0 sin 0 2 {1k}
I1 0 e AC=1 DC=2m
R1 e 0 1k
V5 f 0 EXP(0 1)
V6 g 0 SFFM(0 1)
.tran 1u 2m
""")
    def sig(name):
        g, i = c.layout.device(name)
        return c.elements[g], i

    v1, i = sig("V1")
    assert jnp.isclose(v1.ac[i], 1.0) and jnp.isclose(v1.ac_phase[i], 90.0)
    assert jnp.isclose(v1.value.value[i], 0.5)
    v3, i = sig("V3")  # PULSE defaults from .tran: tr=tf=tstep, pw=per=tstop
    assert jnp.isclose(v3.value.rise[i], 1e-6) and jnp.isclose(v3.value.width[i], 2e-3)
    v4, i = sig("V4")
    assert jnp.isclose(v4.value.amplitude[i], 2.0)
    assert jnp.isclose(v4.value.freq[i], 1e3)
    v5, i = sig("V5")
    assert jnp.isclose(v5.value.tau1[i], 1e-6) and jnp.isclose(v5.value.td2[i], 1e-6)
    v6, i = sig("V6")
    assert jnp.isclose(v6.value.fc[i], 500.0)
    assert jnp.isclose(vx.dc(c).v("e"), 2.0)
    with pytest.raises(NetlistError, match="EXP argument 4 is required"):
        vx.parse_netlist("V1 a 0 EXP(0 1 0)\nR1 a 0 1\n")


def test_current_source_multiplier():
    c = vx.parse_netlist(".subckt s a\nI1 0 a 1m m=2\n.ends\nX1 o s m=3\nR1 o 0 1\n")
    assert jnp.isclose(vx.dc(c).v("o"), 6e-3)


# =============================================================================
# Robustness
# =============================================================================


def test_lexing_comments_continuations_and_line_endings():
    text = ("first line is not special unless title=True\r\n"
            "* full comment\r\n"
            "V1\tin 0 1 ; inline\r\n"
            "R1 in out $ dollar comment\r\n"
            "+ 1k // slash comment\r\n"
            "R2 out \\\r\n"
            "0 1k\r\n"
            "  * indented comment\r\n"
            ".end\r\n"
            "R3 garbage after end\r\n")
    lines = read_lines(text, title=True)
    assert [ln.text for ln in lines] == ["V1 in 0 1", "R1 in out 1k",
                                         "R2 out 0 1k"]
    assert [ln.loc.line for ln in lines] == [3, 4, 6]
    c = vx.parse_netlist(text, title=True)
    assert jnp.isclose(vx.dc(c).v("out"), 0.5)


def test_split_fields():
    assert split_fields("M1 d g s b nch w = 1u  l= {2 * x}") == [
        "M1", "d", "g", "s", "b", "nch", "w=1u", "l={2 * x}"]
    assert split_fields("V1 a 0 PULSE (0 1 2n) td = 1n") == [
        "V1", "a", "0", "PULSE(0 1 2n)", "td=1n"]
    assert split_fields(".param a = b + 1 c='x == 2'") == [".param", "a=b+1",
                                                          "c='x == 2'"]


def test_errors_carry_file_and_line(tmp_path):
    (tmp_path / "bad.inc").write_text("* header\nR1 a b {nope}\n")
    with pytest.raises(NetlistError, match=r"bad\.inc:2: R1: cannot parse value"):
        vx.parse_netlist("V1 a 0 1\n.include bad.inc\n", base_dir=tmp_path)
    for text, match in [("J1 d g s jmod", r":1: unsupported element 'J1'"),
                        (".step r 1 2 1\n", r":1: unsupported directive '.step'"),
                        ("V1 a 0 1\nM1 d g s b nomodel", ":2: M1: unknown model"),
                        ("R1 a 0 1\n.ends", ":2: .ends without .subckt")]:
        with pytest.raises(NetlistError, match=match):
            vx.parse_netlist(text)


# =============================================================================
# Open-PDK-style model hierarchy (tests/data/pdk_style)
# =============================================================================

PDK = Path(__file__).parent / "data" / "pdk_style"


def parse_pdk(corner="tt", **kw):
    text = (PDK / "tb_inverter.sp").read_text().replace(" tt\n", f" {corner}\n", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vx.parse_netlist(text, title=True, base_dir=PDK, return_info=True,
                                **kw)


def test_pdk_style_hierarchy_parses_and_reports_models():
    c, info = parse_pdk()
    names = [Path(f).relative_to(PDK.resolve()).as_posix() for f in info.files]
    assert names == ["models.lib", "corners/common.spice", "cells/nfet_01v8.spice",
                     "cells/pfet_01v8.spice", "cells/passives.spice"]
    status = {k: (s.type, s.level, s.status) for k, s in info.models.items()}
    assert status == {
        "pdk_diode_pw2nd": ("d", "3", "fallback"),
        "pdk_npn": ("npn", "1", "native"),
        "pdk_nfet_01v8/pdk_nfet_01v8__model.0": ("nmos", "54", "fallback"),
        "pdk_nfet_01v8/pdk_nfet_01v8__model.1": ("nmos", "54", "fallback"),
        "pdk_nfet_01v8/pdk_nfet_01v8__model.2": ("nmos", "54", "fallback"),
        "pdk_pfet_01v8/pdk_pfet_psp": ("pmos", "103", "fallback"),
    }
    # bins chosen from the scaled geometry (W/L in microns, .option scale=1u)
    assert info.devices["Xn.msky"].card == "pdk_nfet_01v8__model.0"
    assert info.devices["Xlong.msky"].card == "pdk_nfet_01v8__model.2"
    assert info.devices["Xn.msky"].margin == pytest.approx(0.16)  # W to 1u edge
    assert info.devices["Xlong.msky"].margin == pytest.approx(1.0)  # L=1u vs 0.5u
    assert info.options["scale"] == 1e-6
    # the subcircuit-local resistor: rsh l / (w + dw), agauss -> nominal
    assert jnp.isclose(c.get("Xr.R0", "r"), 2000 * 3.0 / 0.33)
    sol = vx.dc(c)
    assert sol.converged and 0 < sol.v("mid") < 1.8


def test_pdk_style_corners_and_hook():
    tt, _ = parse_pdk("tt")
    ff, _ = parse_pdk("ff")
    g_tt, g_ff = tt.layout.device("Xn.msky")[0], ff.layout.device("Xn.msky")[0]
    assert tt.elements[g_tt].process is not ff.elements[g_ff].process
    FakeBSIM.seen.clear()
    c, info = parse_pdk("ss", models={"nmos:54": FakeBSIM})
    vth0 = {s.name: s.model.params["vth0"] for s in FakeBSIM.seen}
    # 0.47 + corner (0.03) + lod_shift(0.27, 0.27); long device: 0.44 + 0.03
    assert vth0 == {"Xn.msky": pytest.approx(0.50 + 1e-3 / 0.54),
                    "Xlong.msky": pytest.approx(0.47)}
    long = next(s for s in FakeBSIM.seen if s.name == "Xlong.msky")
    assert long.instance["m"] == 2 and long.instance["w"] == pytest.approx(5e-6)
    assert info.models["pdk_nfet_01v8/pdk_nfet_01v8__model.2"].status == "hook"
    assert info.devices["Xn.msky"].implementation == "hook: FakeBSIM"


# =============================================================================
# .if blocks, includes inside subcircuits, dot-less include, OSDI fallback
# =============================================================================


def test_if_blocks_at_top_level():
    deck = """
.param mode={mode}
.if (mode == 1)
R1 a 0 1k
.elseif (mode == 2)
.param rv=2k
R1 a 0 {{rv}}
.else
  .if (mode > 5)
R1 a 0 6k
  .else
R1 a 0 3k
  .endif
.endif
V1 a 0 1
"""
    got = {m: float(vx.parse_netlist(deck.format(mode=m)).get("R1", "r"))
           for m in (1, 2, 3, 7)}
    assert got == pytest.approx({1: 1e3, 2: 2e3, 3: 3e3, 7: 6e3})


@pytest.mark.parametrize("text,match", [
    (".if (1)\nR1 a 0 1\n", ":1: .if without .endif"),
    (".if (1)\n.else\n.else\n.endif\n", ":3: .else after .else"),
    (".endif\n", ":1: .endif without .if"),
    (".subckt s a\n.if (1)\n.ends\n", ":3: .ends inside an open .if"),
    (".if (1)\n.model m d\n.endif\n", ":2: .model inside .if is not supported"),
    (".if (nope > 1)\nR1 a 0 1\n.endif\n", "unknown parameter 'nope'"),
])
def test_if_block_errors(text, match):
    with pytest.raises(NetlistError, match=match):
        vx.parse_netlist(text)


OSDI = """
.include cells/osdi_mos.spice
V1 d 0 1
X1 d d 0 0 lv_nmos w=1u ng=1
X2 d d 0 0 lv_nmos w=2u ng=2
X3 d d 0 0 lv_nmos w=1u as=1p
X4 d d 0 0 lv_nmos w=1u rfmode=1
X5 0 d d d lv_pmos w=2u m=3
"""


def test_if_blocks_use_instance_parameters_and_include_per_subckt():
    seen = {}

    def hook(spec):
        seen[spec.name] = spec
        polarity = "p" if spec.model.params["type"] < 0 else "n"
        return vx.Level1MOSFET(spec.nodes, spec.instance["w"], spec.instance["l"],
                               polarity=polarity)

    c = vx.parse_netlist(OSDI, base_dir=PDK, models={"psp103va": hook,
                                                     "pspnqs103va": hook})
    assert sorted(seen) == ["X1.Nn", "X2.Nn", "X3.Nn", "X4.Nn", "X5.Np"]
    z1 = 0.34e-6
    assert seen["X1.Nn"].instance["as"] == pytest.approx(1e-6 * z1)  # odd ng
    assert seen["X2.Nn"].instance["as"] == pytest.approx(2e-6 / 2 * 2 * z1)  # even
    assert seen["X3.Nn"].instance["as"] == pytest.approx(1e-12)  # given
    assert seen["X4.Nn"].model.name == "lv_nmos_psp_rf"  # rfmode branch
    # the model file is included inside both subcircuits
    assert seen["X5.Np"].model.name == "lv_pmos_psp"
    assert seen["X5.Np"].instance["mult"] == 3 and seen["X5.Np"].nodes[0] == "0"
    assert vx.dc(c).converged


def test_osdi_mosfets_fall_back_without_a_hook():
    with pytest.warns(UserWarning, match="psp103va \\(Verilog-A MOSFET\\) has no"):
        c, info = vx.parse_netlist(OSDI, base_dir=PDK, return_info=True)
    assert info.devices["X5.Np"].implementation == "EKVMOSFET (fallback)"
    g, i = c.layout.device("X5.Np")
    assert c.elements[g].polarity == "p" and jnp.isclose(c.elements[g].w[i], 6e-6)
    report = info.model_report()
    assert "needs Verilog-A model 'psp103va'" in report
    assert vx.dc(c).converged
    deck = ".model dev weirdva foo=1\nN1 a 0 dev\nR1 a 0 1\n"
    with pytest.raises(NetlistError, match=r"register it with .*'weirdva'"):
        vx.parse_netlist(deck)


def test_dotless_include_is_honoured_or_skipped_with_a_warning():
    deck = ".include cells/legacy.spice\nR1 a 0 {legacy_total}\nV1 a 0 1\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = vx.parse_netlist(deck, base_dir=PDK)
    messages = [str(w.message) for w in caught]
    assert any("legacy.spice:2: `include` without a dot treated as .include" in m
               for m in messages)
    assert any("legacy.spice:3: skipping `include \"missing_esd_model.pm3\"`" in m
               for m in messages)
    assert jnp.isclose(c.get("R1", "r"), 10e3)


def test_include_once_at_top_level():
    deck = ".include cells/legacy_extra.spice\n.include cells/legacy_extra.spice\n" \
           "R1 a 0 {legacy_r}\n"
    c, info = vx.parse_netlist(deck, base_dir=PDK, return_info=True)
    assert len(info.files) == 1 and jnp.isclose(c.get("R1", "r"), 5e3)


# =============================================================================
# Real open-PDK decks (mirrored under ~/.cache/voltax/pdk; skipped without)
# =============================================================================

PDK_ROOT = Path.home() / ".cache" / "voltax" / "pdk"
SKY130 = PDK_ROOT / "google/skywater-pdk-libs-sky130_fd_pr/main/models"
GF180 = PDK_ROOT / "google/globalfoundries-pdk-libs-gf180mcu_fd_pr/main/models/ngspice"
IHP_SG13 = PDK_ROOT / "IHP-GmbH/IHP-Open-PDK/main/ihp-sg13g2/libs.tech/ngspice/models"


def needs(path):
    return pytest.mark.skipif(not path.exists(), reason=f"{path} not mirrored")


def parse_quiet(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vx.parse_netlist(text, return_info=True)


@needs(SKY130)
def test_real_sky130_deck():
    c, info = parse_quiet(f""".lib "{SKY130}/sky130.lib.spice" tt
Vdd vdd 0 1.8
Vin in 0 0.9
XMP out in vdd vdd sky130_fd_pr__pfet_01v8 w=2 l=0.15
XMN out in 0 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15
X3 out in 0 0 sky130_fd_pr__nfet_01v8 w=5 l=1
""")
    assert len(info.files) > 100
    # bins as chosen by ngspice-42 (benchmarks/benchmark_pdk_decks.py)
    card = {d: m.card for d, m in info.devices.items()}
    assert card["XMN.msky130_fd_pr__nfet_01v8"] == "sky130_fd_pr__nfet_01v8__model.71"
    assert card["XMP.msky130_fd_pr__pfet_01v8"] == "sky130_fd_pr__pfet_01v8__model.44"
    assert card["X3.msky130_fd_pr__nfet_01v8"] == "sky130_fd_pr__nfet_01v8__model.23"
    statuses = {(s.type, s.level, s.status) for s in info.models.values()}
    assert ("nmos", "54", "fallback") in statuses and ("r", "", "native") in statuses
    assert vx.dc(c).converged


@needs(GF180)
@pytest.mark.parametrize("design_first", [True, False])
def test_real_gf180_deck_in_either_include_order(design_first):
    design = f'.include "{GF180}/design.ngspice"\n'
    lib = f'.lib "{GF180}/sm141064.ngspice" typical\n'
    c, info = parse_quiet((design + lib if design_first else lib + design) + """
Vdd vdd 0 3.3
Vin in 0 1.65
MP out in vdd vdd pmos_3p3 w=2u l=0.28u
MN out in 0 0 nmos_3p3 w=1u l=0.28u
""")
    assert info.devices["MN"].card == "nmos_3p3.4"  # as ngspice-42
    assert info.devices["MP"].card == "pmos_3p3.8"
    assert vx.dc(c).converged


@needs(IHP_SG13)
def test_real_ihp_deck():
    c, info = parse_quiet(f""".lib "{IHP_SG13}/cornerMOSlv.lib" mos_tt
Vdd vdd 0 1.2
Vin in 0 0.6
XMP out in vdd vdd sg13_lv_pmos w=2u l=0.13u
XMN out in 0 0 sg13_lv_nmos w=1u l=0.13u
""")
    assert info.devices["XMN.Nsg13_lv_nmos"].card == "sg13g2_lv_nmos_psp"
    assert info.devices["XMP.Nsg13_lv_pmos"].implementation == "EKVMOSFET (fallback)"
    assert {s.status for s in info.models.values()} == {"fallback"}
    assert vx.dc(c).converged
