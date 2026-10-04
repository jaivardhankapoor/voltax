r"""# Designing an active filter by gradient descent on its Bode plot

Filter design is usually done with tables: pick a prototype (Butterworth,
Chebyshev, ...), look up the stage Q factors, and solve for component values.
With a differentiable AC analysis we can instead *fit* the component values
to a desired magnitude response directly, including non-idealities such as
the op-amp's finite gain-bandwidth product that the tables ignore.

The circuit is a 4th-order low-pass made of two unity-gain Sallen-Key stages
built around a behavioral `vx.OpAmp`. The target is a Butterworth response
with corner $f_c = 10$ kHz,

$$ |H(f)|^2 = \frac{1}{1 + (f/f_c)^{8}}, \qquad
   L = \frac{1}{F} \sum_f \big(20\log_{10}|H_\text{sim}(f)| - 20\log_{10}
   |H(f)|\big)^2 . $$

`vx.ac` linearizes the circuit around its DC operating point and solves
$(G + j 2\pi f C)\,z = -b$ for all frequencies at once; it is differentiable,
so `jax.grad` of the dB error gives the gradient with respect to all four
capacitors. The resistors are fixed at 10 kOhm: scaling every R up and every
C down by the same factor leaves the response unchanged, so fixing the
impedance level makes the solution unique.

What to look at: the capacitors converge close to the textbook equal-resistor
Sallen-Key design (C1 = 2Q/(wR), C2 = 1/(2QwR) with Q = 0.541 and 1.307),
but not exactly onto it: the optimized design also corrects for the op-amp's
finite bandwidth and so tracks the target better than the textbook values.
The figure shows the Bode plot before and after.
"""

# %% Setup
import os
from pathlib import Path

import equinox as eqx
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

F_C = 10e3  # target corner frequency (Hz)
R = 10e3  # all resistors (ohm)


# %% The circuit
# One unity-gain Sallen-Key low-pass stage:
#
#   inp --R-- a --R-- b ---(+)
#             |       |        opamp --+-- out
#             C1      C2    +-(-)      |
#             |       |     |          |
#            out     gnd    +----------+
#
# All four capacitors go into one named group, "caps", so they can be read
# and replaced as a single (4,) array of log-capacitances.
def sallen_key(b, inp, out, name, c1, c2):
    a, x = f"{name}.a", f"{name}.b"
    b.resistor(inp, a, R, name=f"{name}.R1")
    b.resistor(a, x, R, name=f"{name}.R2")
    b.capacitor(a, out, c1, name=f"{name}.C1", group="caps")
    b.capacitor(x, "0", c2, name=f"{name}.C2", group="caps")
    b.opamp(x, out, out, name=f"{name}.U", a0=1e5, gbw=2e6)


b = vx.CircuitBuilder()
b.vsource("in", "0", 0.0, ac=1.0, name="Vin")
# A naive starting point: stage Qs of 0.5 and 1.0, both at f0 = 15.9 kHz.
# (Starting both stages identical would be a symmetric saddle point.)
sallen_key(b, "in", "mid", "s1", c1=1e-9, c2=1e-9)
sallen_key(b, "mid", "out", "s2", c1=2e-9, c2=0.5e-9)
circuit = b.build()
print(circuit.summary())


# %% Target response and loss
freqs = jnp.logspace(2, 5, 40 if FAST else 120)
target_db = -10 * jnp.log10(1 + (freqs / F_C) ** 8)


def with_caps(log_c):
    return eqx.tree_at(lambda c: c.elements["caps"].log_c, circuit, log_c)


def response_db(log_c):
    return vx.ac(with_caps(log_c), freqs).db("out")


def loss(log_c):
    return jnp.mean((response_db(log_c) - target_db) ** 2)


# %% Optimize the four capacitors with Adam
log_c0 = circuit.elements["caps"].log_c
optimizer = optax.adam(0.05)
opt_state = optimizer.init(log_c0)


@jax.jit
def step(log_c, opt_state):
    value, grad = jax.value_and_grad(loss)(log_c)
    updates, opt_state = optimizer.update(grad, opt_state)
    return optax.apply_updates(log_c, updates), opt_state, value


steps = 1500 if FAST else 4000
log_c = log_c0
for i in range(steps + 1):
    log_c, opt_state, value = step(log_c, opt_state)
    if i % (steps // 5) == 0:
        caps = ", ".join(f"{c * 1e9:.3f}" for c in jnp.exp(log_c))
        print(f"step {i:4d}  dB-MSE {value:.2e}  C = [{caps}] nF")


# %% Compare with the textbook design
# Equal-R unity-gain Sallen-Key: w0 = 1/(R sqrt(C1 C2)), Q = sqrt(C1/C2)/2.
w = 2 * jnp.pi * F_C
textbook = []
for q in (0.5412, 1.3066):  # 4th-order Butterworth stage Qs
    textbook += [2 * q / (w * R), 1 / (2 * q * w * R)]
names = circuit.devices("caps")
fitted_circuit = with_caps(log_c)
print("capacitor   fitted     textbook")
for name, tb in zip(names, textbook):
    fitted = fitted_circuit.get(name, "c")
    print(f"  {name:6s} {fitted * 1e9:7.3f} nF  {tb * 1e9:7.3f} nF")
# The small differences are real: the textbook formulas assume an ideal
# op-amp, while the optimizer compensated for the 2 MHz gain-bandwidth.
for label, lc in (("textbook", jnp.log(jnp.array(textbook))), ("optimized", log_c)):
    err = jnp.max(jnp.abs(response_db(lc) - target_db))
    print(f"max deviation from Butterworth, {label:9s} design: {err:.4f} dB")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fine = jnp.logspace(2, 5, 300)
before, after = vx.ac(circuit, fine), vx.ac(fitted_circuit, fine)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax1.semilogx(fine, -10 * jnp.log10(1 + (fine / F_C) ** 8), "k", lw=4, alpha=0.25,
             label="Butterworth target")
ax1.semilogx(fine, before.db("out"), "--", label="initial")
ax1.semilogx(fine, after.db("out"), label="optimized")
ax1.set(ylabel="|H| (dB)", ylim=(-90, 5), title="4th-order Sallen-Key low-pass")
ax1.legend()
for sol, style in ((before, "--"), (after, "-")):
    ax2.semilogx(fine, sol.phase("out"), style)
ax2.set(xlabel="frequency (Hz)", ylabel="phase (deg)")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
