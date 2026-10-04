# Gradients and optimization

## How gradients are computed

Every nonlinear solve in Voltax (a DC point, or one transient step) finds
\(z^\*\) with \(F(z^\*, p) = 0\) for parameters \(p\). Differentiating that
identity gives, by the implicit function theorem,

\[
\frac{\partial z^\*}{\partial p} = -\left(\frac{\partial F}{\partial z}\right)^{-1}
\frac{\partial F}{\partial p}.
\]

In reverse mode we never form that matrix. For a loss \(L(z^\*)\) we solve
one *adjoint* system with the transposed Jacobian at the solution and take a
vector-Jacobian product of the residual:

\[
\left(\frac{\partial F}{\partial z}\right)^{\!\top} \lambda =
\frac{\partial L}{\partial z^\*}, \qquad
\frac{\partial L}{\partial p} = -\lambda^\top \frac{\partial F}{\partial p}.
\]

So a gradient costs one extra linear solve per Newton solve, independent of
how many iterations the forward pass needed, and Newton's iterates are never
stored. In a transient, \(p\) includes the previous state \(z_n\), so the
adjoint propagates backwards through time (the discrete adjoint method)
via `jax.lax.scan`. `vx.solve_root` exposes the same machinery for your own
residuals.

!!! warning "Forward mode"
    Gradients are implemented as a custom VJP, so use reverse mode
    (`jax.grad`, `jax.vjp`, `eqx.filter_grad`). `jax.jvp` / `jax.jacfwd`
    through an analysis are not supported.

## Pattern 1: build inside the loss

For small circuits, write a function from physical parameters to a circuit
and differentiate through it. It's readable and makes the parametrization
explicit:

```python
import jax
import jax.numpy as jnp
import optax
import voltax as vx

ts = jnp.linspace(0, 5e-6, 201)

def make(log_p):
    r, c = jnp.exp(log_p)
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Step(0, 1, 0.0, 1e-9))
    b.resistor("in", "out", r)
    b.capacitor("out", "0", c)
    return b.build()

target = vx.transient(make(jnp.log(jnp.array([2e3, 1e-9]))), ts).v("out")

@jax.jit
def loss(log_p):
    return jnp.mean((vx.transient(make(log_p), ts).v("out") - target) ** 2)

log_p = jnp.log(jnp.array([1e3, 1e-9]))
opt = optax.adam(0.05)
state = opt.init(log_p)
for _ in range(100):
    g = jax.grad(loss)(log_p)
    updates, state = opt.update(g, state)
    log_p = optax.apply_updates(log_p, updates)
print(jnp.prod(jnp.exp(log_p)))   # R*C -> 2e-6: only the product is identifiable
```

## Pattern 2: differentiate the circuit

For large circuits, or ones loaded from a netlist, treat the circuit itself
as the parameter pytree. `eqx.filter_grad` returns a circuit-shaped
gradient:

```python
import equinox as eqx

circuit = make(jnp.log(jnp.array([1e3, 1e-9])))

@eqx.filter_jit
def loss_c(circuit):
    return jnp.mean((vx.transient(circuit, ts).v("out") - target) ** 2)

grads = eqx.filter_grad(loss_c)(circuit)
print(grads.elements["Resistor"].log_r, grads.elements["Capacitor"].log_c)
```

To optimize only some parameters, partition the circuit. The idiom is to
put the trainable devices in a named group when building
(`b.resistor(..., group="tune")`) and select that group:

```python
params, static = eqx.partition(circuit, eqx.is_inexact_array)
only_r = jax.tree_util.tree_map(lambda _: False, circuit)
only_r = eqx.tree_at(lambda c: c.elements["Resistor"].log_r, only_r, True)
trainable, frozen = eqx.partition(circuit, only_r)

def loss_part(trainable):
    return loss_c(eqx.combine(trainable, frozen))

print(eqx.filter_grad(loss_part)(trainable).elements["Resistor"].log_r)
```

## Parametrizations

Positive physical quantities are stored as logarithms (`log_r`, `log_c`,
`log_l`, `log_w`, `log_kp`, ...). Gradients are therefore with respect to
log-values, which keeps parameters positive under any optimizer and makes
step sizes scale-free across decades.

When conductances are the *weights of a model* (analog neural networks,
topology search), use `vx.Conductance` with an explicit transform:

| `transform` | \(g(\theta)\) | use for |
|---|---|---|
| `"log"` | \(e^\theta\) | positive weights spanning decades |
| `"softplus"` | \(g_{min} + \mathrm{softplus}(\theta)\) | positive weights near linear scale |
| `"sigmoid"` | \(g_{min} + (g_{max}-g_{min})\,\sigma(\theta)\) | "does this edge exist?" relaxations |
| `"linear"` | \(\theta\) | unconstrained (may go negative) |

## Shared parameters

Some parameters belong to many devices at once. All MOSFETs built from one
`MOSProcess` share its scalars (threshold, mobility, ...), so the gradient
with respect to `process.log_kp` sums the contributions of every
transistor. That is exactly what you want for process inference, e.g.
fitting a die's process corner to a ring-oscillator frequency.

## Batching

Because analyses are pure functions, `jax.vmap` gives you Monte Carlo,
sweeps and data batches:

```python
def out_at_end(r):
    return vx.transient(circuit.set("R1", r=r), ts).v("out")[-1]

print(jax.vmap(out_at_end)(jnp.array([500.0, 1e3, 2e3])))
```

To drive a group of sources from data, e.g. the inputs of an analog
classifier, put them in one group and swap their DC values per sample with
`with_dc`:

```python
b = vx.CircuitBuilder()
for k in range(3):
    b.vsource(f"x{k}", "0", 0.0, group="inputs")
    b.conductance(f"x{k}", "sum", 1e-3, group="weights")
b.resistor("sum", "0", 1e3)
net = b.build()

def forward(x):
    src = net.elements["inputs"].with_dc(x)
    return vx.dc(net.replace("inputs", src)).v("sum")

print(jax.vmap(forward)(jnp.eye(3)))
```
