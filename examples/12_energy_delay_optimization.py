r"""# Energy per operation vs. supply voltage: power-aware sizing of CMOS logic

Lowering the supply voltage is the most effective way to save energy in
digital logic, and it is paid for in speed. Designers trade the two by
sweeping $V_{DD}$ and resizing transistors; with a differentiable simulator
both knobs can be tuned at once by gradient descent on the *measured* energy
and delay of a transient simulation.

The circuit is a chain of four CMOS inverters driving a 50 fF load. The first
stage is unit-sized (it stands for the gate that drives the chain); the other
three may be resized. The input toggles once up and once down per clock
period $T$, and the energy of one such operation is what the supply delivers,

$$ E_{op} = \int_0^T V_{DD}\, i_{DD}(t)\, dt
   \;\approx\; C_{eff}\, V_{DD}^2 , $$

because every node that switches is charged once from the supply (half of
$C V^2$ is burnt in the PMOS on the way up, the other half in the NMOS on the
way down). Short-circuit current while both devices conduct and the
voltage-dependence of the transistor capacitances make the measured $C_{eff}$
drift slightly with $V_{DD}$. The gate delay follows the alpha-power law

$$ t_{pd} \propto \frac{C\, V_{DD}}{(V_{DD} - V_{th})^{\alpha}} , $$

so energy falls quadratically as $V_{DD}$ drops while delay blows up near
$V_{th}$. The energy-delay product $\mathrm{EDP} = E_{op}\, t_{pd}$ has a
minimum in between ($V_{DD} = 3V_{th}/(3-\alpha)$ for the alpha-power law).

`vx.measure.energy` integrates the power absorbed by the source named "Vdd"
(negative: it delivers energy), `vx.measure.delay` measures the 50% input to
output delay of both edges, and both are differentiable. We use the
trapezoidal rule on a time grid refined at the input edges by `vx.time_grid`:
backward Euler's first-order error overestimates the energy by several
percent on such a grid.

Finally we minimise $E_{op}$ over $(\log V_{DD}, \log s_2, \log s_3,
\log s_4)$ (the per-stage width multipliers) subject to $t_{pd} \le
t_{spec}$, with an augmented Lagrangian and L-BFGS. The gradients are checked
against finite differences first.

What to look at: (a) energy follows $C_{eff} V_{DD}^2$ closely while delay
grows steeply at low $V_{DD}$; (b) the EDP minimum; (c) the optimizer walks
from a fast, power-hungry start onto the delay constraint and then slides
along it to lower energy, ending below the best plain $V_{DD}$ scaling of the
unit-sized chain: tapering the stages buys back the speed that a lower
supply costs.
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
from voltax.library import cmos

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"

N_STAGES = 4
C_LOAD = 50e-15  # output load (F)
PERIOD = 10e-9  # one operation: input rises, then falls (s)
EDGE = 50e-12  # input rise/fall time (s)
VDD_NOM = 1.2
VTH = 0.4  # threshold of the default EKV process (V)
T_SPEC = 100e-12  # delay constraint for the optimization (s)


# %% The circuit
# Supply "Vdd", a PULSE input "Vin", four inverters from the CMOS library
# and a load capacitor. `cmos.inverter` adds a PMOS then an NMOS, so the
# NMOS group "EKVMOSFET_n" (and the PMOS group) holds one device per stage,
# in stage order.
b = vx.CircuitBuilder()
b.vsource("vdd", "0", VDD_NOM, name="Vdd")
b.vsource("in", "0", vx.signals.Pulse(
    0.0, VDD_NOM, delay=0.5e-9, rise=EDGE, fall=EDGE,
    width=PERIOD / 2 - EDGE, period=PERIOD), name="Vin")
nodes = ["in"] + [f"n{i}" for i in range(1, N_STAGES)] + ["out"]
for a, y in zip(nodes[:-1], nodes[1:]):
    cmos.inverter(b, a, y, "vdd")
b.capacitor("out", "0", C_LOAD, name="CL")
circuit = b.build()
print(circuit.summary())

# One period, starting and ending with the input low and every node settled,
# so the energy stored in the capacitors is the same at both ends and the
# supply energy is exactly the energy dissipated by one up-down operation.
# `time_grid` puts small steps at the input edges and grows them slowly
# (growth=1.1) so the output edges are resolved too.
ts = vx.time_grid(circuit, PERIOD, dt_max=PERIOD / 200, growth=1.1)
print(f"time grid: {ts.size} points, smallest step {jnp.diff(ts).min() * 1e12:.1f} ps")


# %% Energy and delay as differentiable functions of the design
# The design is a pytree: log VDD and the log width multipliers of stages
# 2..4 (stage 1 is fixed). NMOS and PMOS of a stage are scaled together,
# keeping the library's 2:1 PMOS/NMOS ratio.
def build(design):
    vdd = jnp.exp(design["log_vdd"])
    shift = jnp.concatenate([jnp.zeros(1), design["log_s"]])
    c = circuit.set("Vdd", value=vdd).set("Vin", v2=vdd)
    for group in ("EKVMOSFET_n", "EKVMOSFET_p"):
        c = eqx.tree_at(lambda c_, g=group: c_.elements[g].log_w, c,
                        circuit.elements[group].log_w + shift)
    return c, vdd


def energy_delay(c, vdd):
    """Supply energy per operation (J) and mean in->out delay (s)."""
    sol = vx.transient(c, ts, method="trap")
    e_op = -vx.measure.energy(c, sol, "Vdd")
    vin, vout = sol.v("in"), sol.v("out")
    # four inversions: the output follows the input
    t_lh = vx.measure.delay(ts, vin, vout, vdd / 2, trig_direction="rise",
                            targ_direction="rise")
    t_hl = vx.measure.delay(ts, vin, vout, vdd / 2, trig_direction="fall",
                            targ_direction="fall")
    return e_op, 0.5 * (t_lh + t_hl)


@jax.jit
def metrics(design):
    return energy_delay(*build(design))


def design(vdd, log_s=(0.0, 0.0, 0.0)):
    return {"log_vdd": jnp.log(jnp.asarray(vdd, float)),
            "log_s": jnp.asarray(log_s, float)}


unit = design(VDD_NOM)
e_nom, d_nom = metrics(unit)
sol_nom = vx.transient(build(unit)[0], ts, method="trap").check()
print(f"VDD = {VDD_NOM} V, unit sizing: E_op = {e_nom * 1e15:.1f} fJ, "
      f"delay = {d_nom * 1e12:.1f} ps")
print(f"  E_op / VDD^2 = {e_nom / VDD_NOM**2 * 1e15:.1f} fF "
      f"(load alone: {C_LOAD * 1e15:.0f} fF)")


# %% Sweep VDD with jax.vmap
# Every point is a full transient; `vmap` runs them as one batched program.
vdds = jnp.linspace(0.5, 1.5, 11 if FAST else 41)
energies, delays = jax.vmap(lambda v: metrics(design(v)))(vdds)

# Textbook check: least-squares fit E = C_eff VDD^2.
c_eff = jnp.sum(energies * vdds**2) / jnp.sum(vdds**4)
dev = energies / (c_eff * vdds**2) - 1
print(f"fitted C_eff = {c_eff * 1e15:.1f} fF; E_op deviates from C_eff VDD^2 "
      f"by {dev.min() * 100:+.1f}% .. {dev.max() * 100:+.1f}%")
print(f"  E_op/VDD^2 rises from {energies[0] / vdds[0]**2 * 1e15:.1f} fF at "
      f"{vdds[0]:.1f} V to {energies[-1] / vdds[-1]**2 * 1e15:.1f} fF at "
      f"{vdds[-1]:.1f} V (short-circuit current flows only above VDD = 2 Vth)")

# Alpha-power law: log t_pd = log k + log VDD - alpha log(VDD - Vth).
A = jnp.stack([jnp.ones_like(vdds), -jnp.log(vdds - VTH)], axis=1)
coef, *_ = jnp.linalg.lstsq(A, jnp.log(delays / vdds))
alpha = coef[1]
print(f"alpha-power-law fit (Vth = {VTH} V): alpha = {alpha:.2f}")

edp = energies * delays
k = int(jnp.argmin(edp))
print(f"minimum EDP on the sweep at VDD = {vdds[k]:.3f} V "
      f"(alpha-power estimate 3 Vth/(3 - alpha) = {3 * VTH / (3 - alpha):.2f} V)")


# %% Gradients of energy and delay with respect to the design
# `jax.jacrev` of `metrics` gives dE/d(design) and d(delay)/d(design) in two
# adjoint passes (each one transposed linear solve per time step).
jacobian = jax.jit(jax.jacrev(metrics))


def dlog_edp(log_vdd):
    """d log(EDP) / d log(VDD) for the unit-sized chain."""
    dsn = {"log_vdd": log_vdd, "log_s": jnp.zeros(N_STAGES - 1)}
    (e, d), (de, dd) = metrics(dsn), jacobian(dsn)
    return de["log_vdd"] / e + dd["log_vdd"] / d


# The EDP is minimal where d log(EDP) / d log(VDD) = 0; a few secant steps
# on the exact derivative locate it between sweep points.
lo, hi = jnp.log(vdds[max(k - 1, 0)]), jnp.log(vdds[min(k + 1, vdds.size - 1)])
g_lo, g_hi = dlog_edp(lo), dlog_edp(hi)
for _ in range(5):
    mid = hi - g_hi * (hi - lo) / (g_hi - g_lo)
    lo, g_lo, hi, g_hi = hi, g_hi, mid, dlog_edp(mid)
vdd_edp = float(jnp.exp(hi))
e_edp, d_edp = metrics(design(vdd_edp))
print(f"minimum-EDP VDD = {vdd_edp:.3f} V: E_op = {e_edp * 1e15:.1f} fJ, "
      f"delay = {d_edp * 1e12:.1f} ps, EDP = {e_edp * d_edp * 1e24:.2f} fJ*ns")

# Check against central finite differences at an arbitrary design:
# dE/d log VDD, and d delay / d log W of the last stage (both transistors).
probe = design(0.9, (0.3, 0.6, 0.9))
de, dd = jacobian(probe)
h = 1e-4
up_v = {**probe, "log_vdd": probe["log_vdd"] + h}
dn_v = {**probe, "log_vdd": probe["log_vdd"] - h}
up_w = {**probe, "log_s": probe["log_s"].at[2].add(h)}
dn_w = {**probe, "log_s": probe["log_s"].at[2].add(-h)}
fd_e = (metrics(up_v)[0] - metrics(dn_v)[0]) / (2 * h)
fd_d = (metrics(up_w)[1] - metrics(dn_w)[1]) / (2 * h)
ad_e, ad_d = de["log_vdd"], dd["log_s"][2]
print("gradient check (adjoint vs central difference):")
print(f"  dE/dlog VDD       = {ad_e * 1e15:+.4f} fJ  vs {fd_e * 1e15:+.4f} fJ  "
      f"(rel. err {abs(ad_e / fd_e - 1):.1e})")
print(f"  d delay/dlog W_4  = {ad_d * 1e12:+.4f} ps  vs {fd_d * 1e12:+.4f} ps  "
      f"(rel. err {abs(ad_d / fd_d - 1):.1e})")


# %% Minimise energy subject to delay <= T_SPEC
# Augmented Lagrangian for the inequality g = t_pd / t_spec - 1 <= 0:
#   L = E / E_0 + (rho/2) [max(0, g + lam/rho)^2 - (lam/rho)^2],
# minimised over the design by a few L-BFGS steps (optax, with line search),
# then the multiplier is updated, lam <- max(0, lam + rho g).
RHO = 20.0


def lagrangian(dsn, lam):
    e, d = metrics(dsn)
    g = d / T_SPEC - 1
    penalty = 0.5 * RHO * (jnp.maximum(0.0, g + lam / RHO) ** 2 - (lam / RHO) ** 2)
    return e / e_nom + penalty


optimizer = optax.lbfgs()
value_and_grad = optax.value_and_grad_from_state(lagrangian)


@jax.jit
def step(dsn, state, lam):
    value, grad = value_and_grad(dsn, lam, state=state)
    updates, state = optimizer.update(grad, state, dsn, value=value, grad=grad,
                                      value_fn=lagrangian, lam=lam)
    return optax.apply_updates(dsn, updates), state


outer, inner = (4, 6) if FAST else (6, 10)
dsn, lam = unit, jnp.array(0.0)
trajectory = [tuple(map(float, metrics(dsn)))]
t0 = time.perf_counter()
for it in range(outer):
    state = optimizer.init(dsn)  # fresh state: the cached value depends on lam
    for _ in range(inner):
        dsn, state = step(dsn, state, lam)
        trajectory.append(tuple(map(float, metrics(dsn))))
    e, d = trajectory[-1]
    lam = jnp.maximum(0.0, lam + RHO * (d / T_SPEC - 1))
    print(f"outer {it}: E_op {e * 1e15:6.2f} fJ  delay {d * 1e12:6.1f} ps  "
          f"VDD {jnp.exp(dsn['log_vdd']):.3f} V  lambda {lam:.3f}")
print(f"optimization: {outer * inner} L-BFGS steps in "
      f"{time.perf_counter() - t0:.1f} s")
e_opt, d_opt = metrics(dsn)
trajectory = np.array(trajectory)


# %% Compare with plain VDD scaling of the unit-sized chain
# Energy grows with VDD, so the best unit-sized design that meets the spec is
# the lowest VDD whose delay is within it: bracket it on the sweep, then
# solve delay(VDD) = T_SPEC by secant steps in log-log coordinates.
j = int(jnp.argmax(delays <= T_SPEC))  # first (lowest) feasible sweep point
x0, x1 = jnp.log(vdds[j - 1]), jnp.log(vdds[j])
y0, y1 = jnp.log(delays[j - 1] / T_SPEC), jnp.log(delays[j] / T_SPEC)
for _ in range(4):
    x0, y0, x1 = x1, y1, x1 - y1 * (x1 - x0) / (y1 - y0)
    y1 = jnp.log(metrics(design(jnp.exp(x1)))[1] / T_SPEC)
vdd_spec = float(jnp.exp(x1))
e_spec, d_spec = metrics(design(vdd_spec))
scale = jnp.exp(dsn["log_s"])
print(f"delay spec {T_SPEC * 1e12:.0f} ps:")
print(f"  VDD scaling, unit sizing: VDD = {vdd_spec:.3f} V  E_op = "
      f"{e_spec * 1e15:.2f} fJ  delay = {d_spec * 1e12:.1f} ps")
print(f"  optimized               : VDD = {jnp.exp(dsn['log_vdd']):.3f} V  E_op = "
      f"{e_opt * 1e15:.2f} fJ  delay = {d_opt * 1e12:.1f} ps")
print("  optimized stage widths (x unit): 1.00, "
      + ", ".join(f"{s:.2f}" for s in scale))
print(f"  energy saved vs. VDD scaling alone: {(1 - e_opt / e_spec) * 100:.1f}%")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.plot(vdds, energies * 1e15, "o-", ms=3, color="C0", label="$E_{op}$ (simulated)")
ax.plot(vdds, c_eff * vdds**2 * 1e15, "--", color="C0", alpha=0.6,
        label=f"$C_{{eff}} V_{{DD}}^2$, $C_{{eff}}$ = {c_eff * 1e15:.1f} fF")
ax.set(xlabel="$V_{DD}$ (V)", ylabel="energy per operation (fJ)",
       title="(a) energy and delay vs. supply")
ax.legend(loc="upper center")
ax_d = ax.twinx()
ax_d.semilogy(vdds, delays * 1e12, "s-", ms=3, color="C3")
ax_d.set_ylabel("delay (ps)", color="C3")
ax_d.tick_params(axis="y", colors="C3")

ax = axes[1]
ax.plot(vdds, edp * 1e24, "o-", ms=3, color="C2")
ax.axvline(vdd_edp, color="gray", ls=":")
ax.plot([vdd_edp], [e_edp * d_edp * 1e24], "k*", ms=12,
        label=f"min EDP at {vdd_edp:.2f} V")
ax.set(xlabel="$V_{DD}$ (V)", ylabel="EDP (fJ $\\cdot$ ns)",
       title="(b) energy-delay product")
ax.legend()

ax = axes[2]
ax.plot(energies * 1e15, delays * 1e12, "-", color="gray", lw=1,
        label="VDD sweep, unit sizing")
ax.plot(trajectory[:, 0] * 1e15, trajectory[:, 1] * 1e12, ".-", ms=3, color="C1",
        label="optimizer trajectory")
ax.plot(*(trajectory[0] * [1e15, 1e12]), "o", color="C1")
ax.plot(e_opt * 1e15, d_opt * 1e12, "*", color="C1", ms=14, label="optimized")
ax.plot(e_spec * 1e15, d_spec * 1e12, "kD", ms=6,
        label=f"VDD scaling only ({vdd_spec:.2f} V)")
ax.axhline(T_SPEC * 1e12, color="C3", ls="--", label="delay constraint")
ax.set(xlabel="energy per operation (fJ)", ylabel="delay (ps)",
       title="(c) constrained energy minimization",
       xlim=(0.9 * e_opt * 1e15, 1.05 * e_nom * 1e15),
       ylim=(0.75 * T_SPEC * 1e12, 1.4 * T_SPEC * 1e12))
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
