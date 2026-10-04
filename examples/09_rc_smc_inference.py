r"""# Gradient-free SMC vs. gradient-based NUTS on an RC filter

Bayesian inference over circuit parameters can be run with or without
gradients. This example compares the two on the smallest interesting
problem: an RC low-pass whose step response is measured with noise.

$$ \log R, \log C \sim \mathcal{N}(\log R_0, 0.7^2) \times
   \mathcal{N}(\log C_0, 0.7^2), \qquad
   v^\text{meas}_k \sim \mathcal{N}\big(1 - e^{-t_k / RC},\ \sigma^2\big). $$

The step response depends on $R$ and $C$ only through $\tau = RC$, so the
posterior is a long thin ridge along the hyperbola $RC = \tau$, bounded only
by the prior. That makes it a good stress test:

* **Tempered SMC with random-walk Metropolis** moves a cloud of particles
  from the prior to the posterior; it needs only log-density *values*.
* **NUTS** follows Hamiltonian trajectories along the ridge using the
  gradient $\nabla \log p$, which voltax supplies by differentiating
  `vx.transient`.

What to look at: both methods recover $\tau$ tightly while $R$ and $C$
individually stay as uncertain as the prior allows; the timing and the
number of log-density evaluations show the cost of each approach.
"""

# %% Setup
import os
import time
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib
from blackjax.mcmc import random_walk
from blackjax.smc import resampling

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"


# %% Model
R0, C0 = 1e3, 1e-9  # prior medians; also the true values
ts = jnp.linspace(0.0, 5e-6, 51)


def rc_circuit(r, c):
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Step(0.0, 1.0, delay=0.0, rise=1e-9))
    b.resistor("in", "out", r)
    b.capacitor("out", "0", c)
    return b.build()


@jax.jit
def simulate(theta):
    return vx.transient(rc_circuit(jnp.exp(theta["log_R"]), jnp.exp(theta["log_C"])),
                        ts).v("out")


truth = {"log_R": jnp.log(R0), "log_C": jnp.log(C0)}
SIGMA = 0.02
measured = simulate(truth) + SIGMA * jax.random.normal(jax.random.key(1), ts.shape)
PRIOR_STD = 0.7


def log_prior(theta):
    return -0.5 * (((theta["log_R"] - truth["log_R"]) / PRIOR_STD) ** 2
                   + ((theta["log_C"] - truth["log_C"]) / PRIOR_STD) ** 2)


def log_likelihood(theta):
    return -0.5 * jnp.sum((measured - simulate(theta)) ** 2) / SIGMA**2


def log_posterior(theta):
    return log_prior(theta) + log_likelihood(theta)


def summarize(name, samples, seconds):
    tau = jnp.exp(samples["log_R"] + samples["log_C"]) * 1e6
    r = jnp.exp(samples["log_R"]) / 1e3
    print(f"{name:5s} {seconds:6.1f}s  tau = {tau.mean():.4f} +- {tau.std():.4f} us"
          f"   R = {r.mean():.2f} +- {r.std():.2f} kOhm")


# %% Tempered SMC with random-walk Metropolis (no gradients)
n_particles = 500 if FAST else 2000
rw_kernel = random_walk.build_additive_step()


def rw_step(key, state, logdensity_fn, sigma):
    return rw_kernel(key, state, logdensity_fn, random_walk.normal(sigma))


smc = blackjax.adaptive_tempered_smc(
    log_prior, log_likelihood,
    mcmc_step_fn=rw_step, mcmc_init_fn=random_walk.init,
    mcmc_parameters={"sigma": jnp.array([[0.05, 0.05]])},
    resampling_fn=resampling.systematic, target_ess=0.5, num_mcmc_steps=10,
)
key_prior, key_smc, key_nuts = jax.random.split(jax.random.key(0), 3)
particles = {name: truth[name] + PRIOR_STD * jax.random.normal(k, (n_particles,))
             for name, k in zip(truth, jax.random.split(key_prior))}  # prior draws

t0 = time.time()
state, n_steps = smc.init(particles), 0
smc_step = jax.jit(smc.step)
while state.tempering_param < 1.0:
    key_smc, key = jax.random.split(key_smc)
    state, _ = smc_step(key, state)
    n_steps += 1
smc_time = time.time() - t0
smc_samples = state.particles
smc_evals = n_steps * n_particles * 10
summarize("SMC", smc_samples, smc_time)
print(f"      {n_steps} tempering steps, ~{smc_evals} likelihood evaluations")


# %% NUTS (uses gradients)
n_chains, n_draws = (4, 250) if FAST else (4, 1000)
t0 = time.time()
key_warm, key_run = jax.random.split(key_nuts)
warmup = blackjax.window_adaptation(blackjax.nuts, log_posterior)
(warm_state, params), _ = warmup.run(key_warm, truth, num_steps=200 if FAST else 500)
nuts = blackjax.nuts(log_posterior, **params)


def run_chain(key):
    def step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.num_integration_steps)

    keys = jax.random.split(key, n_draws)
    return jax.lax.scan(step, warm_state, keys)[1]


nuts_samples, n_leapfrog = jax.jit(jax.vmap(run_chain))(
    jax.random.split(key_run, n_chains))
nuts_samples = jax.tree.map(lambda x: x.reshape(-1), nuts_samples)
jax.block_until_ready(nuts_samples)
nuts_time = time.time() - t0
summarize("NUTS", nuts_samples, nuts_time)
print(f"      {n_chains} chains x {n_draws} draws, "
      f"{int(n_leapfrog.sum())} gradient evaluations after warmup")
print(f"true tau = {R0 * C0 * 1e6:.4f} us")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
r_line = jnp.linspace(0.2, 5.0, 200)
for ax, (name, s, color) in zip(axes[:2], (("SMC (random walk)", smc_samples, "C0"),
                                           ("NUTS (gradients)", nuts_samples, "C1"))):
    ax.plot(r_line, R0 * C0 / (r_line * 1e3) * 1e9, "k--", lw=1, label="RC = tau")
    ax.scatter(jnp.exp(s["log_R"]) / 1e3, jnp.exp(s["log_C"]) * 1e9, s=4, alpha=0.3,
               color=color, label=name)
    ax.set(xscale="log", yscale="log", xlim=(0.2, 5), ylim=(0.2, 5),
           xlabel="R (kOhm)", ylabel="C (nF)", title=name)
    ax.minorticks_off()
    ax.set_xticks([0.3, 1, 3], ["0.3", "1", "3"])
    ax.set_yticks([0.3, 1, 3], ["0.3", "1", "3"])
    ax.legend(loc="upper right")
for name, s, color in (("SMC", smc_samples, "C0"), ("NUTS", nuts_samples, "C1")):
    axes[2].hist(jnp.exp(s["log_R"] + s["log_C"]) * 1e6, bins=40, density=True,
                 alpha=0.5, color=color, label=name)
axes[2].axvline(R0 * C0 * 1e6, color="k", ls="--", label="truth")
axes[2].set(xlabel="tau = RC (us)", ylabel="density", title="Posterior of tau")
axes[2].legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
