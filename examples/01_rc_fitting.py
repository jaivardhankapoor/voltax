r"""# Fitting an RC ladder to a measured step response

The simplest useful thing a differentiable simulator does: recover component
values from waveforms. We simulate a two-stage RC low-pass filter, treat its
step response as a "measurement", start from values that are 50% off, and
let a gradient-based optimizer find the true R1, R2, C1, C2.

The loss is the mean squared error between simulated and measured node
voltages,

$$ L(\theta) = \frac{1}{T} \sum_{k,\,n} \big(v_n(t_k; \theta)
   - v_n^\text{meas}(t_k)\big)^2, \qquad \theta = (\log R_1, \log R_2,
   \log C_1, \log C_2), $$

and `jax.grad` differentiates straight through `vx.transient`. Each implicit
time step is differentiated with the adjoint (implicit function theorem), so
the backward pass costs one linear solve per step, independent of how many
Newton iterations the forward pass needed. Optimizing in log-space keeps the
values positive and puts ohms and nanofarads on the same scale.

Identifiability matters: the transfer function to `out` alone has only three
free coefficients (DC gain and two poles), so four parameters cannot be
recovered from `v(out)` by itself. Observing the middle node `n1` as well
pins all four down.

What to look at: the loss falls to round-off level within a few dozen L-BFGS
iterations and the fitted values match the truth; the figure overlays the
initial, fitted and measured waveforms.
"""

# %% Setup
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import optax

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"


# %% The circuit
# Vin --R1-- n1 --R2-- out --Rload-- gnd, with C1 at n1 and C2 at out.
# The circuit is rebuilt from physical values on every call. That is cheap
# (it happens once, at trace time, inside `jax.jit`) and the most readable way
# to make a small circuit a differentiable function of its parameters.
def rc_ladder(r1, r2, c1, c2, r_load=10e3):
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Step(0.0, 1.0, delay=0.0, rise=1e-9))
    b.resistor("in", "n1", r1, name="R1")
    b.resistor("n1", "out", r2, name="R2")
    b.resistor("out", "0", r_load, name="Rload")
    b.capacitor("n1", "0", c1, name="C1")
    b.capacitor("out", "0", c2, name="C2")
    return b.build()


TRUE = {"R1": 1e3, "R2": 2e3, "C1": 1e-9, "C2": 0.5e-9}
ts = jnp.linspace(0.0, 10e-6, 101)
NODES = ("n1", "out")


def simulate(log_params):
    """Waveforms ``(len(NODES), T)`` for log-space parameters."""
    sol = vx.transient(rc_ladder(*jnp.exp(log_params)), ts)
    return jnp.stack([sol.v(n) for n in NODES])


# %% Measurement and initial guess
true_log = jnp.log(jnp.array(list(TRUE.values())))
measured = simulate(true_log)
init_log = true_log + jnp.log(1.5)  # every value 50% too large


def loss(log_params):
    return jnp.mean((simulate(log_params) - measured) ** 2)


# %% Optimize with L-BFGS (optax)
# For a handful of smooth parameters a quasi-Newton method converges far
# faster than Adam. `value_and_grad_from_state` reuses the value and gradient
# computed by the line search.
optimizer = optax.lbfgs()
value_and_grad = optax.value_and_grad_from_state(loss)


@jax.jit
def step(params, state):
    value, grad = value_and_grad(params, state=state)
    updates, state = optimizer.update(
        grad, state, params, value=value, grad=grad, value_fn=loss
    )
    return optax.apply_updates(params, updates), state, value


params, state = init_log, optimizer.init(init_log)
t0 = time.time()
for i in range(25 if FAST else 50):
    params, state, value = step(params, state)
    if i % 5 == 0:
        r1, r2, c1, c2 = jnp.exp(params)
        print(f"iter {i:3d}  loss {value:.2e}  R1 {r1 / 1e3:.4f}k  "
              f"R2 {r2 / 1e3:.4f}k  C1 {c1 * 1e9:.4f}n  C2 {c2 * 1e9:.4f}n")
print(f"optimized in {time.time() - t0:.1f}s, final loss {loss(params):.1e}")

for name, true, fit in zip(TRUE, TRUE.values(), jnp.exp(params)):
    print(f"  {name}: true {true:.4e}  fitted {fit:.4e}  "
          f"error {100 * abs(fit - true) / true:.4f}%")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
t_us = ts * 1e6
for k, node in enumerate(NODES):
    ax.plot(t_us, measured[k], "k", lw=4, alpha=0.25)
    ax.plot(t_us, simulate(init_log)[k], "--", color=f"C{k}")
    ax.plot(t_us, simulate(params)[k], color=f"C{k}", label=f"v({node})")
ax.plot([], [], "k", lw=4, alpha=0.25, label="measured")
ax.plot([], [], "k--", label="initial guess")
ax.set(xlabel="time (us)", ylabel="voltage (V)", title="RC ladder step response")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
