"""CircuitBuilder grouping/naming and the SPICE netlist parser."""

import jax.numpy as jnp
import pytest

import voltax as vx


def test_builder_fuses_devices_and_names_them():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 1.0)
    for i in range(5):
        b.resistor("in", f"n{i}", 1e3 * (i + 1))
    b.resistor("n0", "0", 1e3, group="load")
    c = b.build()
    assert set(c.elements) == {"Resistor", "load", "VoltageSource"}
    assert c.elements["Resistor"].size == 5
    assert jnp.isclose(c.get("R3", "r"), 3e3)
    assert c.layout.device("R6") == ("load", 0)
    assert c.node("n2") == 3 and c.node("gnd") == -1


def test_builder_scopes_and_errors():
    b = vx.CircuitBuilder()
    with b.scope("x1"):
        inner = b.node("mid")
        b.resistor("a", inner, 1.0, name="R")
    assert inner == "x1.mid"
    assert "x1.R" in dict((d, g) for d, g, _ in b.build().layout.devices)
    with pytest.raises(ValueError, match="duplicate"):
        b.resistor("a", "b", 1.0, name="x1.R")
    with pytest.raises(ValueError, match="positive"):
        b.resistor("a", "b", -1.0)


def test_different_processes_are_separate_groups():
    b = vx.CircuitBuilder()
    b.nmos("d", "g", "0")
    b.nmos("d", "g", "0", process=vx.MOSProcess.nmos(vth=0.3))
    b.pmos("d", "g", "vdd")
    c = b.build()
    assert sorted(c.elements) == ["EKVMOSFET_n", "EKVMOSFET_n_2", "EKVMOSFET_p"]


def test_set_is_functional_and_physical():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", 2.0, name="V1")
    b.resistor("in", "0", 1e3, name="R1")
    c = b.build()
    c2 = c.set("R1", r=4e3).set("V1", value=1.0)
    assert jnp.isclose(vx.dc(c2).i("V1"), -0.25e-3)
    assert jnp.isclose(c.get("R1", "r"), 1e3)  # original untouched


@pytest.mark.parametrize("text,value", [
    ("10f", 10e-15), ("1.2u", 1.2e-6), ("4.7k", 4.7e3), ("2meg", 2e6),
    ("10pF", 10e-12), ("1e-3", 1e-3), ("3", 3.0), ("5mil", 5 * 25.4e-6),
])
def test_parse_value(text, value):
    assert vx.parse_value(text) == pytest.approx(value)


NETLIST = """\
* exercise most of the parser
.param rl=2k
.model dfast d(is=1e-14 rs=5)
.model nch nmos (level=1 kp=200u vto=0.5 lambda=0.01)
.subckt inv in out vdd
M1 out in vdd vdd pmos w=2u l=0.13u
M2 out in 0 0 nmos w=1u
.ends inv
Vdd vdd 0 DC 1.2
Vin a 0 PULSE(0 1.2 1n 0.1n 0.1n 5n 10n)
X1 a b vdd inv
X2 b c vdd inv
Cl c 0 10f
Vs s 0 5 AC 1
R1 s m 1k
D1 m 0 dfast
Vx p 0 1
Rx p 0 {rl}
F1 0 f Vx 2
Rf f 0 1k
H1 h 0 Vx 1k
Rh h 0 1k
E1 e 0 s 0 0.5
Re e 0 1k
G1 0 g s 0 1m
Rg g 0 1k
L1 k1 0 1m
L2 k2 0 1m
K1 L1 L2 0.9
Vk k0 0 SIN(0 1 1k)
Rk0 k0 k1
+ 10
Rk k2 0 1k
M3 z s 0 0 nch w=1u l=1u
Rz s z 1k
.tran 1n 10n
.end
"""


def test_netlist_end_to_end():
    c = vx.parse_netlist(NETLIST)
    op = vx.dc(c)
    assert op.converged
    assert op.v("c") < 1e-3  # two inverters: c follows a (low)
    assert jnp.isclose(op.i("Vx"), -0.5e-3)  # 1 V across {rl} = 2k
    # F pushes 2 * i(Vx) = -1 mA from "0" through itself into f (SPICE sign)
    assert jnp.isclose(op.v("f"), -1.0)
    assert jnp.isclose(op.v("h"), -0.5)  # H: 1k x i(Vx)
    assert jnp.isclose(op.v("e"), 2.5)
    assert jnp.isclose(op.v("g"), 5.0)
    assert 0.6 < op.v("m") < 0.8  # diode drop (+ rs)
    assert "X1.M1" in {d for d, _, _ in c.layout.devices}
    assert "Transformer" in c.elements and "Inductor" not in c.elements
    sol = vx.transient(c, jnp.linspace(0, 4e-9, 41))
    assert sol.converged.all() and sol.v("c")[-1] > 1.1


@pytest.mark.parametrize("text,match", [
    ("R1 a b", "missing resistance"),
    ("Z1 a b 1", "unsupported element"),
    (".foo", "unsupported directive"),
    ("X1 a b nosuch", "unknown subcircuit"),
    ("F1 a 0 Vnone 2", "controlling source"),
])
def test_netlist_errors(text, match):
    with pytest.raises(ValueError, match=f"^<netlist>:2: .*{match}"):
        vx.parse_netlist("R0 x 0 1\n" + text)


INV = """\
Vdd vdd 0 1.2
Vin in 0 0.6
M1 out in vdd vdd pch w=0.8u
M2 out in 0 0 nch w=0.4u
.model nch nmos ({extra} kp=400u vto=0.4 cgso=0.2n cgdo=0.2n)
.model pch pmos ({extra} kp=200u vto=-0.4 cgso=0.2n cgdo=0.2n)
"""


def test_netlist_level1_respects_builder_model_and_overlap_caps():
    import functools

    exact = functools.partial(vx.Level1MOSFET, smooth=0.0)
    c = vx.parse_netlist(INV.format(extra="level=1"), mos_model=exact)
    group, _ = c.layout.device("M2")
    m = c.elements[group]
    assert isinstance(m, vx.Level1MOSFET) and m.smooth == 0.0
    assert jnp.isclose(m.process.cov, 0.2e-9)


def test_netlist_warns_about_unsupported_model_parameters():
    with pytest.warns(UserWarning, match="vth0"):
        vx.parse_netlist(INV.format(extra="level=54 vth0=0.4"))
    with pytest.warns(UserWarning, match="Meyer"):
        vx.parse_netlist(INV.format(extra="level=1 tox=4n"))


def test_circuit_conveniences():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1e3), name="V1")
    b.capacitor("in", "a", 1e-9, name="C1", group="caps")
    b.capacitor("a", "0", 2e-9, name="C2", group="caps")
    b.inductor("a", "0", 1e-3, name="L1")
    c = b.build()
    assert c.devices("caps") == ["C1", "C2"]
    z = c.state(v={"a": 0.5}, x={"L1": 1e-3, "V1.i": 2e-3})
    assert z[c.node("a")] == 0.5
    assert z[c.layout.internal("L1")] == 1e-3 and z[c.layout.internal("V1")] == 2e-3
    fast = c.set("V1", freq=2e3, amplitude=0.5)
    group, _ = fast.layout.device("V1")
    assert fast.elements[group].value.freq[0] == 2e3
    assert fast.elements[group].value.amplitude[0] == 0.5
