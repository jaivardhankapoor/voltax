# Writing your own element

Adding a device to Voltax means writing its physics, and nothing else. You
never touch global node indices, matrices or the solver.

## The contract

Subclass `vx.Element` and declare:

| attribute | meaning |
|---|---|
| `terminals` | terminal names, e.g. `("p", "n")` |
| `n_internal` | internal unknowns per device (default 0) |
| `internal_names` | their names, e.g. `("i",)`, used by `sol.i(dev, name)` |
| `shared` | fields shared by all devices (not stacked when fusing), e.g. `("process",)` |
| `local` | `True` (default): each device depends only on its own terminals and internals |

Then implement, **vectorized over N devices**:

```text
currents(self, v, x, t) -> (I, F)    # required
charges(self, v, x)     -> (Q, P)    # optional, default no charge
ac_stimulus(self)       -> (I, F)    # optional, sources only
```

- `v` is `(N, T)` terminal voltages, `x` is `(N, K)` internal unknowns.
- `I`/`Q` are `(N, T)` currents/charges *into the device* at each terminal.
- `F`/`P` are `(N, K)`: the device's internal equations are
  \(\frac{d}{dt}P + F = 0\).
- Return `None` for any part that is zero.

Parameters are ordinary Equinox fields. Per-device parameters have a leading
`(N,)` axis; write the physics with elementwise `jnp` ops and `[..., k]`
indexing and it works for any `N`. The helpers `two_terminal(v)`
(\(v_p - v_n\)) and `through(i)` (`(+i, -i)`) cover the common two-terminal
case.

## Example 1: a stateless nonlinear device

A tunnel-diode-like N-shaped I-V curve, \(i = a v - b v^2 + c v^3\):

```python
import jax
import jax.numpy as jnp
import voltax as vx

class CubicResistor(vx.Element):
    """i = a v - b v^2 + c v^3 (negative differential resistance)."""

    terminals = ("p", "n")
    a: jax.Array
    b: jax.Array
    c: jax.Array

    def __init__(self, nodes, a=1e-3, b=3e-3, c=2.5e-3):
        self.nodes = self._devices(nodes)   # one device or a list
        self.a = self._per_device(a)        # scalar -> (N,)
        self.b = self._per_device(b)
        self.c = self._per_device(c)

    def currents(self, v, x, t):
        u = vx.two_terminal(v)
        return vx.through(self.a * u - self.b * u**2 + self.c * u**3), None
```

Use it like any built-in element; the builder fuses every `CubicResistor`
into one vectorized group:

```python
b = vx.CircuitBuilder()
b.vsource("in", "0", 1.0, name="V1")
b.resistor("in", "d", 100.0)
b.add(CubicResistor(("d", "0")), name="X1")
c = b.build()
print(vx.dc(c).v("d"))
```

For a one-off nonlinearity you can skip the class entirely with
`vx.NonlinearResistor(nodes, fn, params)` / `vx.NonlinearCapacitor`.

## Example 2: a device with internal state

The HP memristor: resistance interpolates between \(R_{on}\) and
\(R_{off}\) with a state \(w \in [0, 1]\) that drifts with the current,
\(dw/dt = k\, i\). The state becomes an internal unknown whose equation is
\(\frac{d}{dt}w - k i = 0\), i.e. `P = w`, `F = -k i`:

```python
class Memristor(vx.Element):
    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("w",)
    log_ron: jax.Array
    log_roff: jax.Array
    log_k: jax.Array

    def __init__(self, nodes, ron=100.0, roff=16e3, k=1e4):
        self.nodes = self._devices(nodes)
        self.log_ron = self._log_per_device(ron)    # positive -> log-space
        self.log_roff = self._log_per_device(roff)
        self.log_k = self._log_per_device(k)

    def currents(self, v, x, t):
        w = jnp.clip(x[..., 0], 0.0, 1.0)
        r = w * jnp.exp(self.log_ron) + (1 - w) * jnp.exp(self.log_roff)
        i = vx.two_terminal(v) / r
        return vx.through(i), (-jnp.exp(self.log_k) * i)[..., None]

    def charges(self, v, x):
        return None, x                              # d w / dt

b = vx.CircuitBuilder()
b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1.0))
b.add(Memristor(("in", "0")), name="M1")
c = b.build()
sol = vx.transient(c, jnp.linspace(0, 2, 801), ic=c.state())
print(sol.i("M1", "w")[::200])   # the state; current is v / r(w)
```

Gradients with respect to `log_ron`, `log_k`, etc. work immediately via the
same implicit differentiation as every built-in device.

## Checklist

- **Charge conservation**: terminal currents (and charges) of a device should
  sum to zero unless it connects to an implicit ground (op-amp outputs).
- **Smoothness**: Newton and gradients like \(C^1\) models. Prefer
  `jax.nn.softplus`/`sigmoid`/`tanh` over `jnp.maximum`/`where` in the
  operating range, and guard exponentials (see `voltax.elements.semiconductor.explin`).
- **Static configuration** (e.g. a polarity string) goes in
  `eqx.field(static=True)`. Devices with different static values land in
  different groups automatically.
- **Shared parameters** (one object referenced by many devices) are listed
  in `shared`. Devices are fused only if they reference the same object.
- **Independent devices**: device `i`'s outputs may only depend on `v[i]`
  and `x[i]`. The solvers build Jacobians from per-device blocks (one JVP
  per terminal/internal slot, see [Analyses](analyses.md#linear-solvers)),
  so a model that mixes devices, e.g. a mutual-inductance matrix across all
  devices of the group, must set `local = False`; its group is then
  differentiated as one block. Row-wise code (`[..., k]`, elementwise ops)
  is always local.
- **Test it** against a closed form: `el.currents(v, x, t)` can be called
  directly on hand-made arrays.
