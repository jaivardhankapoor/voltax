# Getting started

## Install

```bash
uv add voltax                 # or: uv pip install voltax
uv add "voltax[examples]"     # + optax, blackjax, matplotlib for the examples
```

From a clone of the repository (examples, development):

```bash
git clone https://github.com/jaivardhankapoor/voltax && cd voltax
uv sync --all-extras
uv run python examples/01_rc_fitting.py
```

Voltax needs `jax` and `equinox`. Importing it switches JAX to 64-bit floats,
since circuit equations mix femtofarads and kilohms and need double precision.

## Your first circuit

Circuits are described device by device with a `CircuitBuilder`. Nodes are
strings; `"0"` (or `"gnd"`) is ground.

```python
import jax.numpy as jnp
import voltax as vx

b = vx.CircuitBuilder()
b.vsource("in", "0", vx.signals.Pulse(0, 1, delay=1e-6, rise=1e-9, fall=1e-9,
                                      width=5e-6, period=10e-6), name="Vin")
b.resistor("in", "out", 1e3, name="R1")
b.capacitor("out", "0", 1e-9, name="C1")
circuit = b.build()
print(circuit.summary())
```

```text
Circuit: 2 nodes, state size 3
  Capacitor: 1 x Capacitor
  Resistor: 1 x Resistor
  VoltageSource: 1 x VoltageSource, 1 internal/device
```

The state has three entries: two node voltages plus the current through the
voltage source (see [The circuit equations](concepts/formulation.md)).

## Run analyses

```python
op = vx.dc(circuit)                          # operating point
print(op.v("out"), op.converged)

ts = jnp.linspace(0, 20e-6, 2001)
sol = vx.transient(circuit, ts)              # starts from the DC point
v_out = sol.v("out")                         # (2001,) array
i_in = sol.i("Vin")                          # current through Vin, p -> n

ac = vx.ac(circuit.set("Vin", ac=1.0), jnp.logspace(3, 8, 51))
print(ac.db("out")[0], ac.db("out")[-1])      # 0 dB ... -56 dB
```

Results are addressed by name: `sol.v("node")` for voltages, `sol.i("device")`
for a device's internal unknown (the branch current for sources and
inductors). `sol["out"]` and `sol["Vin.i"]` are shorthands.

## Differentiate

Every analysis is differentiable. The simplest pattern is to build the
circuit inside the function you differentiate:

```python
import jax

def settle_error(log_rc):
    r, c = jnp.exp(log_rc)
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Step(0, 1, delay=0.0, rise=1e-9))
    b.resistor("in", "out", r)
    b.capacitor("out", "0", c)
    sol = vx.transient(b.build(), jnp.linspace(0, 5e-6, 501))
    return jnp.mean((sol.v("out") - 1.0) ** 2)

print(jax.grad(settle_error)(jnp.log(jnp.array([1e3, 1e-9]))))
```

For large circuits, differentiate with respect to the circuit itself
(`eqx.filter_grad`) instead; see
[Gradients and optimization](concepts/gradients.md).

## From a SPICE netlist

```python
inverter = vx.parse_netlist("""
Vdd vdd 0 1.2
Vin in  0 PULSE(0 1.2 1n 50p 50p 2n 4n)
M1 out in vdd vdd pmos w=2u l=0.13u
M2 out in 0   0   nmos w=1u l=0.13u
Cl out 0 5f
""")
sol = vx.transient(inverter, jnp.linspace(0, 8e-9, 801))
print(sol.v("out").min(), sol.v("out").max())
```
