r"""# Writing your own device: a memristor with an internal state

Every device in voltax, from the resistor to the EKV MOSFET, is an `Element`
subclass with two short methods. This example writes a new one from scratch:
the HP Labs memristor, a resistor whose resistance depends on how much charge
has flowed through it. Nothing else in the library has to change; the new
device works in every analysis and is differentiable like the built-ins.

Voltax circuits are the charge-oriented DAE $\frac{d}{dt}q(z) + f(z, t) = 0$.
An element contributes, per device, terminal currents $I$ and charges $Q$, and
it may own *internal unknowns* $x$ with their own equations
$\frac{d}{dt}P(x) + F(v, x) = 0$. The linear-drift memristor has one internal
state, the normalized width $w \in [0, 1]$ of its doped region:

$$ i = \frac{v}{R_\text{on} w + R_\text{off} (1 - w)}, \qquad
   \frac{dw}{dt} = k\, i\, \big(1 - (2w - 1)^{2p}\big), $$

with $k = \mu_v R_\text{on} / D^2$ and a Joglekar window that keeps $w$ in
range. So `currents` returns $I = (i, -i)$ and $F = -k\,i\,f(w)$, and
`charges` returns $P = w$.

What to look at: under a sine drive the I-V curve is a hysteresis loop
pinched at the origin (the memristor fingerprint) that shrinks as the
frequency rises; the gradient of the loop area with respect to the device
parameters matches finite differences. For a *stateless* nonlinear device
you don't need a subclass at all: use `vx.NonlinearResistor` or
`vx.NonlinearCapacitor` with any `jnp` function.
"""

# %% Setup
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"


# %% The element
class Memristor(vx.Element):
    """HP linear-drift memristor with a Joglekar window.

    Args:
        nodes: ``(p, n)`` for one device, or a list of them.
        r_on, r_off: Fully doped / undoped resistance (ohm).
        k: Drift coefficient ``mu_v r_on / D^2`` (1 / (A s)).
        p: Window exponent (static: it changes the code path, not a value).
    """

    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("w",)

    # Positive parameters live in log-space, like every built-in element.
    log_r_on: jax.Array
    log_r_off: jax.Array
    log_k: jax.Array
    p: int = eqx.field(static=True)

    def __init__(self, nodes, r_on=100.0, r_off=16e3, k=1e4, p=2):
        self.nodes = self._devices(nodes)
        self.log_r_on = self._log_per_device(r_on)
        self.log_r_off = self._log_per_device(r_off)
        self.log_k = self._log_per_device(k)
        self.p = p

    def resistance(self, w):
        return jnp.exp(self.log_r_on) * w + jnp.exp(self.log_r_off) * (1 - w)

    # v: (N, 2) terminal voltages; x: (N, 1) internal states, here w.
    def currents(self, v, x, t):
        w = x[..., 0]
        i = vx.two_terminal(v) / self.resistance(w)
        window = 1 - (2 * w - 1) ** (2 * self.p)
        F = -jnp.exp(self.log_k) * i * window  # so that dw/dt + F = 0
        return vx.through(i), F[..., None]

    def charges(self, v, x):
        return None, x  # P = w: the internal equation is d(w)/dt + F = 0


# %% A circuit: sine source driving the memristor
b = vx.CircuitBuilder()
b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1.0), name="Vs")
b.add(Memristor(("in", "0")), name="X1")
circuit = b.build()
print(circuit.summary())

# A memristor has no unique DC state (any w is an equilibrium at v = 0),
# so we start the transient from an explicit state with w = 0.1.
z0 = circuit.state(x={"X1.w": 0.1})


# %% I-V loops at several drive frequencies, all at once with vmap
n_steps = 400 if FAST else 2000


def iv_loop(freq, c=circuit):
    """Voltage, current and state over two periods of a `freq` Hz sine."""
    c = c.set("Vs", freq=freq)
    ts = jnp.linspace(0.0, 2.0 / freq, n_steps + 1)
    sol = vx.transient(c, ts, ic=z0, method="trap")
    return sol.v("in"), -sol.i("Vs"), sol.i("X1", "w")


freqs = jnp.array([0.5, 1.0, 2.0, 5.0])
v, i, w = jax.jit(jax.vmap(iv_loop))(freqs)
for f, ik, wk in zip(freqs, i, w):
    print(f"{f:4.1f} Hz: peak current {jnp.max(ik) * 1e3:.3f} mA, "
          f"w swings {jnp.min(wk):.3f} .. {jnp.max(wk):.3f}")


# %% Differentiating through the custom device
# A scalar summary of the hysteresis: the area of the positive lobe, the
# integral of i dv over the positive half of the second period (trapezoid
# rule; the two lobes have opposite orientation, so a full period cancels).
# Gradients flow into the new element's parameters exactly as they do for
# built-in ones.
def loop_area(c):
    v, i, _ = iv_loop(1.0, c)
    lobe = slice(n_steps // 2, 3 * n_steps // 4 + 1)
    v, i = v[lobe], i[lobe]
    return jnp.abs(jnp.sum(0.5 * (i[1:] + i[:-1]) * jnp.diff(v)))


mem_group, _ = circuit.layout.device("X1")
area, grads = jax.value_and_grad(loop_area)(circuit)
print(f"positive-lobe area at 1 Hz: {area * 1e6:.3f} uA*V")
eps = 1e-4
for name in ("log_r_off", "log_k"):
    def shifted(d, name=name):
        return eqx.tree_at(lambda c: getattr(c.elements[mem_group], name), circuit,
                           getattr(circuit.elements[mem_group], name) + d)

    fd = (loop_area(shifted(eps)) - loop_area(shifted(-eps))) / (2 * eps)
    adjoint = getattr(grads.elements[mem_group], name)[0]
    print(f"  d(area)/d({name}): adjoint {adjoint:+.6e}, finite diff {fd:+.6e}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for f, vk, ik in zip(freqs, v, i):
    ax1.plot(vk, ik * 1e3, label=f"{f:g} Hz")
ax1.set(xlabel="v (V)", ylabel="i (mA)", title="Pinched hysteresis loops")
ax1.legend()
for f, wk in zip(freqs, w):
    ax2.plot(jnp.linspace(0, 2, n_steps + 1), wk, label=f"{f:g} Hz")
ax2.set(xlabel="time (periods)", ylabel="state w", title="Internal state")
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
