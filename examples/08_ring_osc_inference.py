r"""# Inferring transistor parameters from a ring oscillator with NUTS

A ring oscillator is the classic process monitor: its frequency and waveform
shape depend on how strong the NMOS and PMOS devices came out of the fab.
Here we turn that around and ask for the *posterior distribution* of the
process transconductances $k_n, k_p$ given a noisy measured waveform.

Model, for the voltage at one stage sampled at times $t_k$:

$$ \log k_{n,p} \sim \mathcal{N}(\log k^\text{nom}_{n,p},\ 0.3^2), \qquad
   v^\text{meas}_k \sim \mathcal{N}\big(v(t_k; k_n, k_p),\ \sigma^2\big). $$

The No-U-Turn Sampler (blackjax) needs $\nabla_\theta \log p$, which voltax
provides through `vx.transient` by implicit differentiation. Both parameters
are *shared* leaves: every transistor references the same `MOSProcess`, so
one scalar `log_kp` moves all five NMOS (or PMOS) devices at once and its
gradient sums their contributions.

A ring has no stable DC operating point to start from, so the transient
starts from an explicit, asymmetric initial state built with `circuit.state`.
We observe only about three periods: over many periods a small frequency
error becomes a full phase slip and the likelihood turns multimodal.

What to look at: the posterior mean lands near the true values with a few
percent uncertainty, and the scatter plot shows the $k_n$-$k_p$ correlation
(a stronger NMOS can partly compensate a weaker PMOS).
"""

# %% Setup
import os
import time
from pathlib import Path

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

import voltax as vx
from voltax.library import cmos

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"
VDD = 1.2


# %% A five-stage ring oscillator
b = vx.CircuitBuilder()
b.vsource("vdd", "0", VDD, name="Vdd")
stages = [f"n{i}" for i in range(5)]
cmos.ring_oscillator(b, stages, vdd="vdd")
for node in stages:
    b.capacitor(node, "0", 5e-15)
ring = b.build()
print(ring.summary())

# All NMOS share one process object, and so do all PMOS; the builder names
# their groups by polarity: "EKVMOSFET_n" and "EKVMOSFET_p".
group = {"n": "EKVMOSFET_n", "p": "EKVMOSFET_p"}
ic = ring.state(v={"vdd": VDD, **{n: VDD * (i % 2) for i, n in enumerate(stages)}})
ts = jnp.linspace(0.0, 0.45e-9, 91)


@jax.jit
def simulate(theta):
    """v(n0) for theta = {"log_kn": ..., "log_kp": ...}."""
    c = eqx.tree_at(
        lambda c: (c.elements[group["n"]].process.log_kp,
                   c.elements[group["p"]].process.log_kp),
        ring, (theta["log_kn"], theta["log_kp"]))
    return vx.transient(c, ts, ic=ic).v("n0")


# %% Synthetic measurement
nominal = {"log_kn": jnp.log(400e-6), "log_kp": jnp.log(200e-6)}
truth = {"log_kn": jnp.log(440e-6), "log_kp": jnp.log(180e-6)}  # a "fast-slow" die
SIGMA = 0.03 * VDD
clean = simulate(truth)
measured = clean + SIGMA * jax.random.normal(jax.random.key(1), clean.shape)


# %% Posterior
PRIOR_STD = 0.3


def log_posterior(theta):
    log_prior = sum(-0.5 * ((theta[k] - nominal[k]) / PRIOR_STD) ** 2 for k in theta)
    residual = measured - simulate(theta)
    return log_prior - 0.5 * jnp.sum(residual**2) / SIGMA**2


# %% NUTS: adapt step size and mass matrix on one chain, then run several
# A dense mass matrix lets NUTS move along the strongly correlated
# (k_n, k_p) ridge; capping the tree depth bounds the cost of each draw.
n_chains = 2 if FAST else 4
n_warmup, n_samples = (100, 60) if FAST else (300, 300)
NUTS = {"max_num_doublings": 5}
key_warm, key_init, key_run = jax.random.split(jax.random.key(0), 3)

t0 = time.time()
warmup = blackjax.window_adaptation(blackjax.nuts, log_posterior,
                                    is_mass_matrix_diagonal=False, **NUTS)
(state, params), _ = warmup.run(key_warm, nominal, num_steps=n_warmup)
jax.block_until_ready(params)
print(f"warmup: {n_warmup} steps in {time.time() - t0:.1f}s, "
      f"step size {params['step_size']:.3f}")
kernel = blackjax.nuts(log_posterior, **params)  # params include NUTS settings


def run_chain(key, position):
    def step(state, key):
        state, _ = kernel.step(key, state)
        return state, state.position

    keys = jax.random.split(key, n_samples)
    return jax.lax.scan(step, kernel.init(position), keys)[1]


# start the chains near the adapted position, slightly jittered
starts = {name: x + 0.02 * jax.random.normal(k, (n_chains,))
          for (name, x), k in zip(state.position.items(), jax.random.split(key_init))}
t0 = time.time()
samples = jax.jit(jax.vmap(run_chain))(jax.random.split(key_run, n_chains), starts)
samples = jax.block_until_ready(jax.tree.map(lambda x: x.reshape(-1), samples))
print(f"sampling: {n_chains} chains x {n_samples} draws in {time.time() - t0:.1f}s")

kn, kp = jnp.exp(samples["log_kn"]) * 1e6, jnp.exp(samples["log_kp"]) * 1e6
print("parameter   true   posterior mean +- std   (uA/V^2)")
for name, k in (("kn", kn), ("kp", kp)):
    true = jnp.exp(truth[f"log_{name}"]) * 1e6
    print(f"  {name}      {true:6.0f}   {k.mean():6.1f} +- {k.std():4.1f}")
print(f"  correlation(log k_n, log k_p) = "
      f"{jnp.corrcoef(samples['log_kn'], samples['log_kp'])[0, 1]:+.2f}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t_ps = ts * 1e12
for k in range(0, len(kn), max(1, len(kn) // 20)):
    draw = {"log_kn": samples["log_kn"][k], "log_kp": samples["log_kp"][k]}
    ax1.plot(t_ps, simulate(draw), color="C0", alpha=0.15)
ax1.plot(t_ps, simulate(nominal), "C2--", label="nominal process")
ax1.plot(t_ps, measured, ".", color="C3", ms=3, label="measurement")
ax1.plot([], [], color="C0", label="posterior draws")
ax1.set(xlabel="time (ps)", ylabel="v(n0) (V)", title="Ring oscillator waveform")
ax1.legend(loc="upper right")
ax2.scatter(kn, kp, s=6, alpha=0.4, label="posterior samples")
ax2.plot(jnp.exp(truth["log_kn"]) * 1e6, jnp.exp(truth["log_kp"]) * 1e6, "X",
         color="C3", ms=12, label="truth")
ax2.plot(400, 200, "P", color="C2", ms=10, label="nominal (prior mean)")
ax2.set(xlabel="k_n (uA/V^2)", ylabel="k_p (uA/V^2)", title="Posterior")
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
