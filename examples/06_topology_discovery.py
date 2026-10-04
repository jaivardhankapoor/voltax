r"""# Discovering a filter topology from a "circuit soup"

Gradient descent can choose *which components exist*, not just their values.
We start from a "soup": every pair of nodes among {in, out, n2, n3, gnd} is
joined by a resistor and a capacitor (18 candidate components), and ask for a
low-pass response with real poles at 20 kHz and 200 kHz. A sparsity penalty
pushes every component towards "absent" (zero conductance or capacitance)
unless the frequency response needs it, so what survives is a circuit you can
read off and build.

The loss combines an AC-analysis fit in dB with a log-sum sparsity penalty,

$$ L = \overline{\big(20\log_{10}|H(f)| - H_\text{dB}^\star(f)\big)^2}
   + \lambda \sum_k \Big[\log\big(1 + \tfrac{g_k}{g_\text{ref}}\big)
   + \log\big(1 + \tfrac{c_k}{c_\text{ref}}\big)\Big]. $$

Unlike an L1 penalty, whose pull on $\log g$ vanishes as $g \to 0$, the
log-sum penalty pushes every unneeded component down at a constant rate in
log-space until it is negligible. Conductances use `vx.Conductance` with a
sigmoid map between `g_min` (open) and `g_max` (short), the standard
relaxation of a discrete on/off choice.

What to look at: of the 18 candidates only about four survive, forming the
textbook two-section RC ladder; pruning the rest leaves the Bode plot
unchanged.
"""

# %% Setup
import itertools
import os
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

G_MIN, G_MAX = 1e-9, 1e-2  # "open" and "short" conductance (S)
G_REF, C_REF = 1e-7, 1e-13  # below these, a component counts as absent
LAMBDA = 0.1


# %% The soup
nodes = ["in", "out", "n2", "n3", "0"]
edges = [e for e in itertools.combinations(nodes, 2) if e != ("in", "0")]
rng = np.random.default_rng(0)

b = vx.CircuitBuilder()
b.vsource("in", "0", 0.0, ac=1.0, name="Vin")
for p, n in edges:
    b.conductance(p, n, G_MAX * rng.uniform(0.05, 0.2), transform="sigmoid",
                  g_min=G_MIN, g_max=G_MAX, name=f"R_{p}_{n}", group="soup_r")
    b.capacitor(p, n, 1e-10 * rng.uniform(0.5, 2.0), name=f"C_{p}_{n}",
                group="soup_c")
soup = b.build()
print(soup.summary())


# %% Target and loss
freqs = jnp.logspace(3, 8, 30 if FAST else 60)
target_db = -10 * jnp.log10((1 + (freqs / 2e4) ** 2) * (1 + (freqs / 2e5) ** 2))


def make_circuit(params):
    c = eqx.tree_at(lambda c: c.elements["soup_r"].theta, soup, params["theta"])
    return eqx.tree_at(lambda c: c.elements["soup_c"].log_c, c, params["log_c"])


def loss(params):
    c = make_circuit(params)
    fit = jnp.mean((vx.ac(c, freqs).db("out") - target_db) ** 2)
    g, cap = c.elements["soup_r"].g, c.elements["soup_c"].c
    sparsity = jnp.sum(jnp.log1p(g / G_REF)) + jnp.sum(jnp.log1p(cap / C_REF))
    return fit + LAMBDA * sparsity, (fit, sparsity)


# %% Optimize
params = {"theta": soup.elements["soup_r"].theta,
          "log_c": soup.elements["soup_c"].log_c}
optimizer = optax.adam(0.1)
opt_state = optimizer.init(params)


@jax.jit
def step(params, opt_state):
    (value, aux), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    return optax.apply_updates(params, updates), opt_state, aux


steps = 8000
for i in range(steps + 1):
    params, opt_state, (fit, sparsity) = step(params, opt_state)
    if i % (steps // 4) == 0:
        print(f"step {i:5d}  dB-MSE {fit:9.4f}  sparsity {sparsity:7.2f}")


# %% Read off and prune the discovered circuit
found = make_circuit(params)
g, cap = found.elements["soup_r"].g, found.elements["soup_c"].c
pruned = vx.CircuitBuilder()
pruned.vsource("in", "0", 0.0, ac=1.0, name="Vin")
print("surviving components:")
for (p, n), gk, ck in zip(edges, g, cap):
    if gk > 100 * G_MIN:
        print(f"  R {p:>3s} - {n:<3s} {1 / gk / 1e3:9.1f} kOhm")
        pruned.resistor(p, n, 1 / gk)
    if ck > C_REF / 10:
        print(f"  C {p:>3s} - {n:<3s} {ck * 1e12:9.2f} pF")
        pruned.capacitor(p, n, ck)
pruned = pruned.build()

fine = jnp.logspace(3, 8, 200)
ac_soup, ac_pruned = vx.ac(found, fine), vx.ac(pruned, fine)
h_soup, h_pruned = ac_soup.db("out"), ac_pruned.db("out")
diff = jnp.max(jnp.abs(ac_soup.v("out") - ac_pruned.v("out")))
print(f"max |H_soup - H_pruned| over 1 kHz - 100 MHz: {diff:.1e}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(fine, -10 * jnp.log10((1 + (fine / 2e4) ** 2) * (1 + (fine / 2e5) ** 2)),
            "k", lw=4, alpha=0.25, label="target")
ax.semilogx(fine, vx.ac(soup, fine).db("out"), "--", label="initial soup")
ax.semilogx(fine, h_soup, label="optimized soup")
ax.semilogx(fine, h_pruned, ":", lw=2, label="pruned circuit")
ax.set(xlabel="frequency (Hz)", ylabel="|H| (dB)", title="Discovered low-pass filter")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
