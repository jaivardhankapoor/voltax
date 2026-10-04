r"""# Sizing 760 wires of a power grid with one adjoint solve per step

A chip's power grid is a resistive mesh: supply pads at the edges, current
drawn by logic everywhere in between. The voltage lost along the wires ("IR
drop") slows the logic down, and the fix is wider metal, which costs area.
Deciding *which* of hundreds of wires to widen is a large optimization
problem.

With finite differences, each gradient would need one extra DC solve per
wire (761 solves for a 20x20 grid). Voltax differentiates the DC solution by
the adjoint method instead: for the operating point $F(v^*, g) = 0$,

$$ \frac{dL}{dg} = -\lambda^\top \frac{\partial F}{\partial g},
   \qquad \Big(\frac{\partial F}{\partial v}\Big)^{\!\top} \lambda
   = \frac{\partial L}{\partial v^*}, $$

so every gradient costs one extra (transposed) linear solve, however many
wires there are. The loss trades IR drop against total metal,
$L = \overline{(V_{DD} - v)^2} + \mu\, \overline{g}$, and each wire is a
`vx.Conductance` with a sigmoid parametrization, which bounds its
conductance between a minimum and a maximum manufacturable width.

What to look at: the optimized grid has a much smaller worst-case drop than a
uniform grid using the *same total metal*; the figure shows where the
optimizer put the metal (thick wires carry the load currents to the pads).
"""

# %% Setup
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"

N = 20  # grid is N x N nodes
VDD = 1.0
G_MIN, G_MAX = 0.5, 50.0  # wire conductance bounds (S)


# %% Building the grid
def node(i, j):
    return f"n{i}_{j}"


edges = [(node(i, j), node(i, j + 1)) for i in range(N) for j in range(N - 1)]
edges += [(node(i, j), node(i + 1, j)) for i in range(N - 1) for j in range(N)]

b = vx.CircuitBuilder()
for p, n in edges:
    # All wires in one group: a single (760,) parameter array `theta`.
    b.conductance(p, n, 5.0, transform="sigmoid", g_min=G_MIN, g_max=G_MAX,
                  group="wires")
for i, j in [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]:
    b.vsource(node(i, j), "0", VDD, name=f"pad{i}_{j}")
rng = np.random.default_rng(0)
loads = {}  # 20% of the nodes draw 10-50 mA each
for k in rng.choice(N * N, N * N // 5, replace=False):
    loads[(k // N, k % N)] = rng.uniform(10e-3, 50e-3)
for (i, j), current in loads.items():
    # SPICE convention: current flows from the node, through the source, to gnd
    b.isource(node(i, j), "0", current)
circuit = b.build()
print(circuit.summary())

grid_index = jnp.array([[circuit.node(node(i, j)) for j in range(N)] for i in range(N)])


def with_theta(theta):
    return eqx.tree_at(lambda c: c.elements["wires"].theta, circuit, theta)


def ir_drop(theta):
    """(N, N) map of VDD - v."""
    return VDD - vx.dc(with_theta(theta)).v()[grid_index]


# %% Loss: IR drop vs. metal
MU = 2e-4


def loss(theta):
    g = with_theta(theta).elements["wires"].g
    return jnp.mean(ir_drop(theta) ** 2) + MU * jnp.mean(g)


theta0 = circuit.elements["wires"].theta
value_and_grad = jax.jit(jax.value_and_grad(loss))

# Sanity check: the adjoint gradient of one wire against finite differences.
_, grad0 = value_and_grad(theta0)
eps, k = 1e-5, 0
fd = (loss(theta0.at[k].add(eps)) - loss(theta0.at[k].add(-eps))) / (2 * eps)
print(f"dL/dtheta[{k}]: adjoint {grad0[k]:.6e}, finite difference {fd:.6e}")


# %% Optimize all wires with Adam
steps = 100 if FAST else 2000
optimizer = optax.adam(0.1)
theta, opt_state = theta0, optimizer.init(theta0)
t0 = time.time()
for step in range(steps + 1):
    value, grad = value_and_grad(theta)
    if step % (steps // 5) == 0:
        g = with_theta(theta).elements["wires"].g
        print(f"step {step:5d}  loss {value:.3e}  max drop "
              f"{jnp.max(ir_drop(theta)) * 1e3:6.1f} mV  mean g {jnp.mean(g):5.2f} S")
    updates, opt_state = optimizer.update(grad, opt_state)
    theta = optax.apply_updates(theta, updates)
print(f"{len(theta0)} wires optimized in {time.time() - t0:.1f}s "
      f"({(time.time() - t0) / (steps + 1) * 1e3:.0f} ms per gradient)")


# %% Compare against a uniform grid with the same total metal
g_opt = with_theta(theta).elements["wires"].g
wires = circuit.elements["wires"]
theta_uniform = jnp.full_like(theta, wires.to_theta(jnp.mean(g_opt)))
for label, th in (("initial", theta0), ("uniform, same metal", theta_uniform),
                  ("optimized", theta)):
    drop = ir_drop(th)
    g_mean = jnp.mean(with_theta(th).elements["wires"].g)
    print(f"  {label:20s} mean g {g_mean:5.2f} S   max drop "
          f"{jnp.max(drop) * 1e3:6.1f} mV   mean drop {jnp.mean(drop) * 1e3:5.1f} mV")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
vmax = float(jnp.max(ir_drop(theta_uniform))) * 1e3
for ax, th, title in ((axes[0], theta_uniform, "uniform grid (same metal)"),
                      (axes[1], theta, "optimized grid")):
    im = ax.imshow(ir_drop(th) * 1e3, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"{title}: max drop {jnp.max(ir_drop(th)) * 1e3:.0f} mV")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="IR drop (mV)")
ax = axes[2]
coords = {node(i, j): (j, i) for i in range(N) for j in range(N)}
for (p, n), g in zip(edges, np.asarray(g_opt)):
    (x0, y0), (x1, y1) = coords[p], coords[n]
    ax.plot([x0, x1], [y0, y1], "k", lw=0.2 + 4 * g / G_MAX, solid_capstyle="round")
ys, xs = zip(*loads)
ax.scatter(xs, ys, s=2e3 * np.array(list(loads.values())), c="C3", zorder=3,
           label="current loads")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12))
ax.set(title="optimized wire widths", xlim=(-1, N), ylim=(N, -1), aspect="equal")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
