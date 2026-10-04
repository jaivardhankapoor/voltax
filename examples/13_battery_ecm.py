r"""# Battery equivalent-circuit model: identifiability and energy prediction

A battery management system (BMS) cannot see inside a cell. It sees the
terminal voltage while a known current flows, and from that it must infer a
model good enough to predict what matters: available power, voltage sag, and
how much energy turns into heat. The workhorse model is the two-RC Thevenin
equivalent circuit (ECM):

    ocv --[R0]-- n1 --[R1 || C1]-- n2 --[R2 || C2]-- term --> load I(t)
     |                                                 |
    Voc                                               gnd

* $V_\text{oc}$: open-circuit voltage (held at 3.7 V here; a state-of-charge
  dependent $V_\text{oc}(\text{SOC})$ is a `Function` signal or a nonlinear
  element away),
* $R_0$: ohmic resistance (electrolyte, contacts): the instantaneous IR jump,
* $R_1 \| C_1$: charge transfer and double layer, $\tau_1 = R_1 C_1 \approx$
  seconds,
* $R_2 \| C_2$: solid-state diffusion, $\tau_2 = R_2 C_2 \approx$ tens of
  seconds.

The load is an ordinary `CurrentSource` from `term` to ground carrying a
pulse profile like a hybrid pulse power characterization (HPPC) test:
discharge and charge pulses separated by rests. With the SPICE convention
(`b.isource(p, n, I)` pushes $I$ from $p$ through the source into $n$), a
positive $I$ draws current out of the + terminal, i.e. a discharge. For a
current $I(t)$ the terminal voltage is

$$ v_\text{term}(t) = V_\text{oc} - R_0 I(t) - u_1(t) - u_2(t), \qquad
   \dot u_k = \frac{I(t)}{C_k} - \frac{u_k}{R_k C_k}. $$

We simulate this circuit with true parameters, add 2 mV sensor noise,
sample at 2 Hz like a BMS would, and fit $\theta = \log(R_0, R_1, R_2, C_1,
C_2)$ by minimizing

$$ L(\theta) = \frac{1}{N\sigma^2} \sum_k \big(v_\text{term}(t_k; \theta)
   - v^\text{meas}_k\big)^2 $$

with L-BFGS through `vx.transient`, from many random initializations at
once (`jax.vmap` over the whole optimizer).

**Identifiability.** The model is symmetric under swapping the two RC
branches: $(R_1, C_1) \leftrightarrow (R_2, C_2)$ gives exactly the same
terminal voltage. So the data determine the parameters only up to this
relabeling, and different initializations land in either mode. Sorting the
branches by time constant ("fast" and "slow") removes the ambiguity, after
which all five parameters are recovered to about a percent: the pulse test
excites both time scales, so nothing else is degenerate.

**Physical prediction.** The heat dissipated in each resistor is

$$ E_{R} = \int_0^T \frac{v_R(t)^2}{R}\, dt, $$

which `vx.measure.energy(circuit, sol, "R0")` computes from the solved
waveforms. A label-switched fit puts the fast-branch heat into "R2", so the
per-resistor bars disagree with the truth, yet the *total* heat, the quantity
a thermal model needs, is predicted to a fraction of a percent by every fit.
We also check `measure.energy` against $R_0 \int I^2 dt$ (the full load
current flows through $R_0$) and the energy balance

$$ E_\text{delivered by }V_\text{oc} = \sum_R E_R + \Delta E_{C_1}
   + \Delta E_{C_2} + E_\text{load}. $$

What to look at: (a) every fit (blue: branches in the true order, red:
swapped) tracks the noisy data to the noise level, $\chi^2/N \approx 1$;
(b) the raw errors of $R_1, R_2, C_1, C_2$ are tens to hundreds of percent
for swapped fits, while $R_0$, $R_0 + R_1 + R_2$ and the parameters relabeled
fast/slow sit within a percent for every fit; (c) the per-resistor heat bars
of swapped fits trade places, but the total heat agrees with the truth for
every fit.
"""

# %% Setup
import os
import time
from pathlib import Path

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


# %% Load profile: HPPC-like pulses (amps; + = discharge)
# (time in s, current in A) corners; 0.1 s ramps between levels.
def pulse_profile():
    segments = [  # (duration, current)
        (10, 0.0), (10, 20.0), (40, 0.0), (10, -15.0), (40, 0.0),
        (30, 10.0), (60, 0.0), (5, 40.0), (45, 0.0),
    ]
    times, values, t = [0.0], [0.0], 0.0
    ramp = 0.1
    for duration, current in segments:
        times += [t + ramp, t + duration]
        values += [current, current]
        t += duration
    return np.array(times), np.array(values)


T_PROFILE, I_PROFILE = pulse_profile()
T_STOP = float(T_PROFILE[-1])
LOAD = vx.signals.PWL(T_PROFILE, I_PROFILE)


# %% The circuit
# Rebuilt from physical values on every call (once per trace, inside `jit`).
def battery_ecm(r0, r1, r2, c1, c2, v_oc=3.7):
    b = vx.CircuitBuilder()
    b.vsource("ocv", "0", v_oc, name="Voc")
    b.resistor("ocv", "n1", r0, name="R0")
    b.resistor("n1", "n2", r1, name="R1")
    b.capacitor("n1", "n2", c1, name="C1")
    b.resistor("n2", "term", r2, name="R2")
    b.capacitor("n2", "term", c2, name="C2")
    b.isource("term", "0", LOAD, name="Iload")  # discharge: out of term
    return b.build()


NAMES = ("R0", "R1", "R2", "C1", "C2")
TRUE = {"R0": 15e-3, "R1": 10e-3, "R2": 20e-3, "C1": 300.0, "C2": 2500.0}
true_log = jnp.log(jnp.array(list(TRUE.values())))
print(f"true time constants: tau1 = {TRUE['R1'] * TRUE['C1']:.1f} s, "
      f"tau2 = {TRUE['R2'] * TRUE['C2']:.1f} s")

# Grid refined at every current edge, up to 0.5 s between samples.
ts = vx.time_grid(battery_ecm(*TRUE.values()), T_STOP, dt_max=0.5 if FAST else 0.25)
t_meas = jnp.arange(0.0, T_STOP, 0.5)  # BMS samples at 2 Hz
print(f"{ts.size} simulation time points, {t_meas.size} measured samples")


def solve(log_params):
    circuit = battery_ecm(*jnp.exp(log_params))
    return circuit, vx.transient(circuit, ts)


def v_term(log_params):
    _, sol = solve(log_params)
    return jnp.interp(t_meas, ts, sol.v("term"))


# %% Synthetic measurement
SIGMA = 2e-3  # 2 mV sensor noise
v_clean = jax.jit(v_term)(true_log)
measured = v_clean + SIGMA * jax.random.normal(jax.random.key(0), t_meas.shape)


def loss(log_params):
    return jnp.mean((v_term(log_params) - measured) ** 2) / SIGMA**2


# %% Fit from many random initializations at once
# Each init is drawn log-uniformly within a factor of 10 of a generic "a few
# milliohm, hundreds of farads" guess, without knowing which branch is fast.
n_inits = 6 if FAST else 16
n_iters = 40 if FAST else 60
nominal = jnp.log(jnp.array([20e-3, 20e-3, 20e-3, 1000.0, 1000.0]))
init_log = nominal + jax.random.uniform(
    jax.random.key(1), (n_inits, 5), minval=-jnp.log(10.0), maxval=jnp.log(10.0))

optimizer = optax.lbfgs()
value_and_grad = optax.value_and_grad_from_state(loss)


@jax.jit
@jax.vmap
def step(params, state):
    value, grad = value_and_grad(params, state=state)
    updates, state = optimizer.update(
        grad, state, params, value=value, grad=grad, value_fn=loss
    )
    return optax.apply_updates(params, updates), state, value


params, state = init_log, jax.vmap(optimizer.init)(init_log)
t0 = time.time()
for i in range(n_iters):
    params, state, values = step(params, state)
    if i % 10 == 0 or i == n_iters - 1:
        print(f"iter {i:3d}  loss (chi2/N) best {values.min():.3f}  "
              f"median {jnp.median(values):.3f}  worst {values.max():.3g}")
fit_time = time.time() - t0
final_loss = jax.jit(jax.vmap(loss))(params)
print(f"fitted {n_inits} inits x {n_iters} L-BFGS iterations in {fit_time:.1f}s; "
      f"noise floor chi2/N = {loss(true_log):.3f}")


# %% Identifiability
fits = np.exp(np.asarray(params))  # (n_inits, 5): R0 R1 R2 C1 C2
truth = np.array(list(TRUE.values()))
rel_err = fits / truth - 1.0
converged = np.asarray(final_loss) < 1.5 * float(loss(true_log))
tau1, tau2 = fits[:, 1] * fits[:, 3], fits[:, 2] * fits[:, 4]
swapped = tau1 > tau2


def canonical(p):
    """Relabel so branch 1 is the fast one (removes the swap symmetry)."""
    p = p.copy()
    s = p[:, 1] * p[:, 3] > p[:, 2] * p[:, 4]
    p[s, 1], p[s, 2] = p[s, 2], p[s, 1].copy()
    p[s, 3], p[s, 4] = p[s, 4], p[s, 3].copy()
    return p


sorted_err = canonical(fits) / truth - 1.0
r_total = fits[:, :3].sum(1)
print(f"\n{converged.sum()}/{n_inits} fits reach the noise floor; "
      f"{swapped[converged].sum()} of them with the RC branches swapped")
print("init  chi2/N  swap " + "".join(f"{n:>9s}" for n in NAMES)
      + "   (relative error, raw labels)")
for k in range(n_inits):
    print(f"{k:4d}  {final_loss[k]:6.3f}  {'yes' if swapped[k] else ' no'}  "
          + "".join(f"{100 * e:+8.1f}%" for e in rel_err[k]))
ok = converged
print("\nacross converged fits   " + "".join(f"{n:>9s}" for n in NAMES)
      + "  R0+R1+R2")
print("max |error|, raw labels " + "".join(
    f"{100 * np.abs(rel_err[ok, j]).max():8.1f}%" for j in range(5))
    + f"{100 * np.abs(r_total[ok] / truth[:3].sum() - 1).max():8.2f}%")
print("max |error|, fast/slow  " + "".join(
    f"{100 * np.abs(sorted_err[ok, j]).max():8.1f}%" for j in range(5)))


# %% Energy: heat in every resistor, true model vs. every fit
RES = ("R0", "R1", "R2")


def energies(log_params):
    circuit, sol = solve(log_params)
    return jnp.stack([vx.measure.energy(circuit, sol, r) for r in RES])


e_true = np.asarray(jax.jit(energies)(true_log))
e_fit = np.asarray(jax.jit(jax.vmap(energies))(params))
tot_true, tot_fit = e_true.sum(), e_fit.sum(1)
tot_err = tot_fit / tot_true - 1.0
max_tot_err = 100 * np.abs(tot_err[ok]).max()
print(f"\nheat dissipated, true model: total {tot_true:.2f} J = "
      + " + ".join(f"{e:.2f} ({n})" for n, e in zip(RES, e_true)))
print(f"total heat error over converged fits: max {max_tot_err:.3f}%, "
      f"mean {100 * np.abs(tot_err[ok]).mean():.3f}%")
per_err = np.abs(e_fit[ok] / e_true - 1).max(0)
print("max per-resistor heat error (raw labels): "
      + ", ".join(f"{n} {100 * e:.0f}%" for n, e in zip(RES, per_err)))

# Cross-checks on the true model. The whole load current flows through R0, so
# its heat is R0 * int I^2 dt with I the programmed waveform; for R2 we
# integrate v^2 / R from the solved node voltages.
circuit, sol = jax.jit(solve)(true_log)
e_r0_analytic = TRUE["R0"] * vx.measure.integral(ts, LOAD(ts) ** 2)
u1 = sol.v("n1") - sol.v("n2")
u2 = sol.v("n2") - sol.v("term")
e_r2_manual = vx.measure.integral(ts, u2**2 / TRUE["R2"])
for name, analytic in (("R0", e_r0_analytic), ("R2", e_r2_manual)):
    e = vx.measure.energy(circuit, sol, name)
    print(f"{name} heat: measure.energy {e:.5f} J vs direct integral "
          f"{analytic:.5f} J (rel. diff {abs(e / analytic - 1):.1e})")

e_source = -vx.measure.energy(circuit, sol, "Voc")  # delivered by the cell
e_load = vx.measure.energy(circuit, sol, "Iload")
de_caps = 0.5 * (TRUE["C1"] * (u1[-1] ** 2 - u1[0] ** 2)
                 + TRUE["C2"] * (u2[-1] ** 2 - u2[0] ** 2))
balance = tot_true + de_caps + e_load
print(f"energy balance: Voc delivers {e_source:.3f} J = heat {tot_true:.3f} "
      f"+ capacitor change {de_caps:.3f} + load {e_load:.3f} = {balance:.3f} J "
      f"(rel. mismatch {abs(balance / e_source - 1):.1e})")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig = plt.figure(figsize=(16, 4.6))
grid = fig.add_gridspec(4, 3, width_ratios=[1.5, 1, 1])
ax_v = fig.add_subplot(grid[:3, 0])
ax_i = fig.add_subplot(grid[3, 0], sharex=ax_v)
ax_p = fig.add_subplot(grid[:, 1])
ax_e = fig.add_subplot(grid[:, 2])
groups = ((ok & ~swapped, "C0", "o", "fit, branches in true order"),
          (ok & swapped, "C3", "o", "fit, branches swapped"),
          (~ok, "0.5", "x", "fit, not converged"))

v_fits = jax.jit(jax.vmap(v_term))(params)
ax_v.plot(t_meas, measured, ".", color="0.6", ms=2, label="measured (2 mV noise)")
for k in np.flatnonzero(ok):
    ax_v.plot(t_meas, v_fits[k], color="C3" if swapped[k] else "C0", lw=0.8,
              alpha=0.6)
ax_v.plot([], [], color="C0", label=f"fits ({ok.sum()} inits)")
ax_v.plot(t_meas, v_clean, "k--", lw=1, label="true model")
ax_v.set(ylabel="terminal voltage (V)", title="(a) Pulse test: measured vs. fitted")
ax_v.tick_params(labelbottom=False)
ax_v.legend(loc="lower right", fontsize=8)
ax_i.fill_between(T_PROFILE, I_PROFILE, color="C1", alpha=0.6, lw=0)
ax_i.axhline(0, color="k", lw=0.5)
ax_i.set(xlabel="time (s)", ylabel="I (A)")

labels = list(NAMES) + ["R0+R1+R2", "R1 fast", "C1 fast", "R2 slow", "C2 slow"]
cols = [rel_err[:, j] for j in range(5)] + [r_total / truth[:3].sum() - 1] + [
    sorted_err[:, j] for j in (1, 3, 2, 4)]
rng = np.random.default_rng(0)
for j, col in enumerate(cols):
    x = j + 0.08 * rng.standard_normal(n_inits)
    for mask, color, marker, _ in groups:
        ax_p.scatter(x[mask], 100 * col[mask], s=14, color=color, marker=marker)
for mask, color, marker, label in groups:
    if mask.any():
        ax_p.scatter([], [], color=color, marker=marker, label=label)
ax_p.axhline(0, color="k", lw=0.8)
ax_p.axvline(5.5, color="0.7", lw=0.8, ls=":")
ax_p.set_yscale("symlog", linthresh=1)
ax_p.set_ylim(-1000, 3000)
ax_p.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax_p.set(ylabel="relative error (%)", title="(b) Parameter recovery across inits")
ax_p.text(5.6, 1500, "relabeled by\ntime constant", fontsize=8, va="top")
ax_p.legend(fontsize=8, loc="lower right")

x = np.arange(4)
fit_bars = np.concatenate([e_fit, tot_fit[:, None]], axis=1)
ax_e.bar(x - 0.2, np.append(e_true, tot_true), 0.4, color="0.3", label="true model")
ax_e.bar(x + 0.2, fit_bars[ok].mean(0), 0.4, color="C0", alpha=0.35,
         label="fits (mean)")
for mask, color, marker, _ in groups[:2]:
    for j in x:
        xs = j + 0.2 + 0.05 * rng.standard_normal(mask.sum())
        ax_e.scatter(xs, fit_bars[mask, j], s=14, color=color, marker=marker,
                     zorder=3)
ax_e.set_xticks(x, ["R0", "R1", "R2", "total"])
ax_e.set(ylabel="heat dissipated (J)",
         title=f"(c) Heat: total within {max_tot_err:.2g}% for every fit")
ax_e.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
