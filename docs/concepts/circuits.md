# Building circuits

## The builder

`CircuitBuilder` collects devices with named nodes and compiles them into a
`Circuit`:

```python
import voltax as vx

b = vx.CircuitBuilder()
b.vsource("vdd", "0", 5.0, name="Vdd")
b.resistor("vdd", "c", 2.2e3, name="Rc")
b.bjt("c", "b", "0", "npn", name="Q1", bf=150.0)
b.resistor("vdd", "b", 470e3, name="Rb")
amp = b.build()
print(vx.dc(amp).v("c"))
```

Every typed helper (`resistor`, `nmos`, `opamp`, ...) is a thin wrapper
around `b.add(element, name=..., group=...)`, which accepts *any* element,
including [your own](custom-elements.md):

```python
b.add(vx.Diode(("c", "0"), is_=1e-15), name="Dclamp")
```

Device names default to a SPICE letter plus a counter (`R1`, `M7`, ...).
Names must be unique; they are how you address devices later.

## Nodes

- Nodes are strings. Numbers are converted with `str`, so SPICE-style
  numbered nodes work.
- `"0"`, `"gnd"`, `"GND"` and `"ground"` are ground.
- `b.node()` creates a fresh, unique internal node, which is what reusable
  subcircuit functions use for their internal nets.
- `with b.scope("stage1"):` prefixes device names and `b.node(name)` nodes
  created inside the block, so a subcircuit can be instantiated many times.

```python
from voltax.library import cmos

b = vx.CircuitBuilder()
b.vsource("vdd", "0", 1.2)
b.vsource("a", "0", 0.0)
with b.scope("u1"):
    cmos.nand(b, ["a", "a"], "y", vdd="vdd")
print(b.build().layout.state_names()[:4])
```

## Groups: why circuits are fast

At `build()` the builder **fuses** devices of the same type and static
configuration into a single vectorized *group*. A 10 000-transistor circuit
evaluates its MOSFET physics as one batched array computation, not 10 000
small ones.

```python
c = amp
print(sorted(c.elements))          # group names
print(c.layout.device("Rc"))       # (group, index within the group)
```

- Devices end up in separate groups when their types or static settings
  differ, e.g. NMOS vs PMOS, or two different `MOSProcess` objects.
- Pass `group="name"` to force devices into a named group. This is the
  idiomatic way to mark *the parameters you will optimize*, or *the sources
  you will drive from data*, as one array.

## Reading and changing parameters

Circuits are immutable [Equinox](https://docs.kidger.site/equinox/) modules.
Changes return a new circuit and work inside `jit`, `grad` and `vmap`:

```python
c2 = amp.set("Rb", r=330e3).set("Vdd", value=3.3)
print(amp.get("Rb", "r"), c2.get("Rb", "r"))
```

Values are physical; positive quantities are stored in log-space
(`Resistor.log_r`, `Capacitor.log_c`, ...), and `set`/`get` convert for you.

For whole-array updates use `circuit.replace(group, element)` or
`eqx.tree_at`:

```python
import equinox as eqx
import jax.numpy as jnp

res = amp.elements["Resistor"]
scaled = amp.replace("Resistor", eqx.tree_at(lambda r: r.log_r, res,
                                             res.log_r + jnp.log(1.1)))
```

## Creating elements directly

Every element class takes `nodes` either for one device, `(p, n)`, or for
many, `[(p1, n1), (p2, n2), ...]`, with parameters given as scalars
(broadcast) or one per device. `vx.Circuit({group: element}, node_names)`
assembles a circuit from integer-indexed elements (`-1` is ground). This is
rarely needed but handy in tests:

```python
c = vx.Circuit(
    {"R": vx.Resistor([(0, 1), (1, -1)], [1e3, 2e3]),
     "V": vx.VoltageSource((0, -1), 3.0)},
    node_names=["in", "mid"],
)
print(vx.dc(c).v("mid"))  # 2.0
```
