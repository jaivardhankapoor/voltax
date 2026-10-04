r"""# Training an analog convolutional network made of resistors and diodes

An analog neural network computes with physics: weights are conductances,
summation is Kirchhoff's current law, and the nonlinearity comes from
devices. Here a small convolutional network for the 8x8 scikit-learn digits
is *a circuit*, and training means finding conductances such that the DC
operating point classifies the input.

Architecture (all conductances, ``g = softplus(theta)`` so they stay
positive):

* 64 input sources (pixel intensity in volts) and one -1 V bias source.
* A 3x3 convolution: each of the 36 hidden nodes is joined to its 3x3 input
  patch by 9 conductances. *Weight sharing* is just indexing: the 324
  conductances read their values from 9 kernel parameters.
* A diode from every hidden node to ground (a soft rectifier).
* A fully connected readout: 36 x 10 conductances into 10 output nodes,
  each loaded to ground.

At DC, node $j$ settles at the conductance-weighted average of its
neighbours, $v_j = \sum_k g_{jk} v_k / \sum_k g_{jk}$ (minus diode current),
so the network is a physically constrained, normalized neural net. The
logits are the output voltages, and the cross-entropy gradient with respect
to all 380 parameters comes from one adjoint solve per image.

What to look at: test accuracy (well above the 10% chance level even in fast
mode), the accuracy after snapping every resistor to the nearest E96
standard value, and the learned kernel / readout maps in the figure.
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"


# %% Data
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) / 16.0  # pixel volts in [0, 1]
X_train, X_test, y_train, y_test = map(
    jnp.asarray, train_test_split(X, digits.target, test_size=0.2, random_state=42))
if FAST:  # a small subset keeps the run to about a minute
    X_train, y_train = X_train[:256], y_train[:256]
    X_test, y_test = X_test[:160], y_test[:160]


# %% The circuit
def pixel(r, c):
    return f"x{r}_{c}"


def hidden(r, c):
    return f"h{r}_{c}"


b = vx.CircuitBuilder()
for r in range(8):
    for c in range(8):
        b.vsource(pixel(r, c), "0", 0.0, group="pixels")
b.vsource("bias", "0", -1.0, name="Vbias")
kernel_index = []  # which of the 9 kernel weights each conv conductance uses
for r in range(6):
    for c in range(6):
        for kr in range(3):
            for kc in range(3):
                b.conductance(pixel(r + kr, c + kc), hidden(r, c), 0.1,
                              transform="softplus", group="conv")
                kernel_index.append(3 * kr + kc)
        b.conductance("bias", hidden(r, c), 0.1, transform="softplus", group="bias")
        b.diode(hidden(r, c), "0", is_=1e-12)
        for k in range(10):
            b.conductance(hidden(r, c), f"out{k}", 0.1, transform="softplus",
                          group="readout")
for k in range(10):
    b.conductance(f"out{k}", "0", 0.1, transform="softplus", group="load")
circuit = b.build()
print(circuit.summary())
kernel_index = jnp.array(kernel_index)
OUTPUTS = [circuit.node(f"out{k}") for k in range(10)]


# %% Parameters -> circuit -> logits
def network(params):
    """Write the trainable parameters into the circuit's conductance groups."""
    thetas = {"conv": params["kernel"][kernel_index],  # weight sharing
              "bias": jnp.broadcast_to(params["bias"], (36,)),
              "readout": params["readout"], "load": params["load"]}
    c = circuit
    for group, theta in thetas.items():
        c = eqx.tree_at(lambda c, g=group: c.elements[g].theta, c, theta)
    return c


# This resistor-diode network always converges directly, so we skip the
# gmin-stepping fallback entirely.
OPTIONS = vx.Options(gmin_steps=0)


def logits(params, images):
    c = network(params)

    def one(x):
        sol = vx.dc(c.replace("pixels", c.elements["pixels"].with_dc(x)),
                    options=OPTIONS)
        return 10.0 * sol.v()[jnp.array(OUTPUTS)]

    return jax.vmap(one)(images)


def loss_fn(params, images, labels):
    z = logits(params, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(z, labels).mean()
    return loss, jnp.mean(jnp.argmax(z, axis=-1) == labels)


# %% Training
k1, k2 = jax.random.split(jax.random.key(0))
params = {"kernel": jax.random.normal(k1, (9,)) - 2.0,
          "bias": jnp.array(-6.0),
          "readout": jax.random.normal(k2, (360,)) - 6.0,
          "load": jnp.full(10, -4.0)}
n_params = sum(p.size for p in jax.tree.leaves(params))
print(f"{n_params} trainable parameters driving "
      f"{sum(circuit.elements[g].size for g in ('conv', 'bias', 'readout', 'load'))}"
      " conductances")

optimizer = optax.adam(0.05)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, images, labels):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss, acc


evaluate = jax.jit(loss_fn)
batch, epochs = 32, (4 if FAST else 15)
for epoch in range(epochs):
    t0 = time.time()
    order = jax.random.permutation(jax.random.key(epoch + 1), len(X_train))
    losses, accs = [], []
    for i in range(0, len(X_train) - batch + 1, batch):
        idx = order[i:i + batch]
        params, opt_state, loss, acc = train_step(params, opt_state, X_train[idx],
                                                  y_train[idx])
        losses.append(loss)
        accs.append(acc)
    print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  train acc "
          f"{np.mean(accs):6.1%}  ({time.time() - t0:.1f}s)")
_, test_acc = evaluate(params, X_test, y_test)
print(f"test accuracy: {test_acc:.1%}")


# %% Hardware check: snap every resistor to the E96 series
E96 = jnp.array([
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33,
    1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82,
    1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49,
    2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40,
    3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64,
    4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34,
    6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66,
    8.87, 9.09, 9.31, 9.53, 9.76])


def snap_to_e96(theta):
    r = 1.0 / jax.nn.softplus(theta)
    decade = 10.0 ** jnp.floor(jnp.log10(r))
    nearest = E96[jnp.argmin(jnp.abs(jnp.log(E96[:, None] * decade / r)), axis=0)]
    g = 1.0 / (nearest * decade)
    return g + jnp.log(-jnp.expm1(-g))  # inverse softplus


quantized = jax.tree.map(lambda t: snap_to_e96(jnp.atleast_1d(t)).reshape(t.shape),
                         params)
_, test_acc_q = evaluate(quantized, X_test, y_test)
print(f"test accuracy with E96 (1%) resistors: {test_acc_q:.1%}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 11, figsize=(20, 2.4))
axes[0].imshow(jax.nn.softplus(params["kernel"]).reshape(3, 3), cmap="viridis")
axes[0].set_title("conv kernel")
readout = jax.nn.softplus(params["readout"]).reshape(36, 10)
for k in range(10):
    axes[k + 1].imshow(readout[:, k].reshape(6, 6), cmap="magma")
    axes[k + 1].set_title(f"readout '{k}'")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"Analog CNN conductances (test accuracy {test_acc:.1%})")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
