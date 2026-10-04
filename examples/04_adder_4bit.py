r"""# A 200-transistor 4-bit ripple-carry adder from library gates

Real digital blocks are built hierarchically: transistors make gates, gates
make full adders, full adders make an adder. `voltax.library.cmos` provides
the standard static-CMOS gates as functions that add devices to a
`CircuitBuilder`, and `CircuitBuilder.scope` keeps the internal nets of each
instance apart. At build time all 200 MOSFETs are fused into two vectorized
groups (NMOS and PMOS), so the device physics is a handful of array ops.

We check the logic exhaustively with DC operating points, mapping over input
words, and then run a transient on the worst-case input, where a carry must
ripple through all four stages:

$$ 1111_2 + 0000_2 + c_\text{in}: \quad c_\text{in}: 0 \to 1
   \;\Rightarrow\; s = 0000_2,\ c_\text{out} = 1 . $$

Inputs are driven by one named group of voltage sources (`group="inputs"`),
so `with_dc` can set all nine input voltages from a bit vector inside a
JAX transformation.

What to look at: every checked input combination produces the right sum;
the transient shows the sum bits flipping one after another as the carry
ripples, and the printed carry propagation delay.
"""

# %% Setup
import os
import time
from pathlib import Path

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
A = [f"a{i}" for i in range(4)]
B = [f"b{i}" for i in range(4)]
S = [f"s{i}" for i in range(4)]
INPUTS = A + B + ["cin"]
OUTPUTS = S + ["cout"]


# %% Building the adder
def build_adder(drive, group=None):
    """4-bit adder whose input nodes are driven by `drive[node]` (V or Signal)."""
    b = vx.CircuitBuilder()
    b.vsource("vdd", "0", VDD, name="Vdd")
    for node in INPUTS:
        b.vsource(node, "0", drive[node], name=f"V{node}", group=group)
    cmos.ripple_adder(b, A, B, "cin", S, "cout", vdd="vdd")
    for node in OUTPUTS:
        b.capacitor(node, "0", 2e-15)  # fan-out load
    return b.build()


circuit = build_adder({node: 0.0 for node in INPUTS}, group="inputs")
print(circuit.summary())


# %% Truth table from DC operating points
def adder_dc(bits):
    """Output voltages for one input bit vector ordered like INPUTS."""
    sources = circuit.elements["inputs"].with_dc(VDD * bits)
    sol = vx.dc(circuit.replace("inputs", sources))
    return jnp.stack([sol.v(n) for n in OUTPUTS]), sol.converged


# All 512 (a, b, cin) words as bit vectors; check a random subset in fast mode.
words = jnp.arange(512)
if FAST:
    words = jax.random.choice(jax.random.key(0), words, (48,), replace=False)
bits = ((words[:, None] >> jnp.arange(9)) & 1).astype(float)

t0 = time.time()
# `lax.map` runs the solves one after another inside a single compiled loop.
# (`jax.vmap` works too, but it batches the Newton loop, so every word pays
# for the slowest one's iterations.)
v_out, converged = jax.jit(lambda x: jax.lax.map(adder_dc, x))(bits)
v_out.block_until_ready()  # JAX runs asynchronously; wait before timing
print(f"{len(words)} DC solves in {time.time() - t0:.1f}s, "
      f"all converged: {bool(converged.all())}")

a_val = words & 0xF
b_val = (words >> 4) & 0xF
cin = (words >> 8) & 1
out_bits = (v_out > VDD / 2).astype(int)
result = jnp.sum(out_bits * (2 ** jnp.arange(5)), axis=1)
n_ok = int(jnp.sum(result == a_val + b_val + cin))
print(f"correct sums: {n_ok}/{len(words)}")
for k in range(3):
    print(f"  {int(a_val[k]):2d} + {int(b_val[k]):2d} + {int(cin[k])} = "
          f"{int(result[k])}")
worst = jnp.max(jnp.minimum(v_out, VDD - v_out))
print(f"worst static output level: {worst * 1e6:.2g} uV from a rail")


# %% Transient: the carry ripple
# a = 1111, b = 0000; cin pulses high, forcing the carry through all stages.
ts = jnp.linspace(0.0, 1.5e-9, 301 if FAST else 1501)
cin_pulse = vx.signals.Pulse(0.0, VDD, delay=0.1e-9, rise=20e-12, fall=20e-12,
                             width=0.7e-9, period=10e-9)
drive = {**{n: VDD for n in A}, **{n: 0.0 for n in B}, "cin": cin_pulse}
ripple = build_adder(drive)
t0 = time.time()
sol = jax.block_until_ready(vx.transient(ripple, ts))
print(f"transient: {len(ts)} steps in {time.time() - t0:.1f}s, "
      f"converged: {bool(sol.converged.all())}")


print("50% delay after cin rises (vx.measure.delay):")
for node in OUTPUTS:
    direction = "rise" if node == "cout" else "fall"
    d = vx.measure.delay(ts, sol.v("cin"), sol.v(node), VDD / 2,
                         targ_direction=direction)
    print(f"  {node:4s} {d * 1e12:6.1f} ps")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(ts * 1e9, sol.v("cin"), "k", label="cin")
ax1.plot(ts * 1e9, sol.v("cout"), "C3", lw=2, label="cout")
ax1.set(ylabel="voltage (V)", title="1111 + 0000 + cin: carry ripple")
ax1.legend(loc="right")
for i, node in enumerate(S):
    ax2.plot(ts * 1e9, sol.v(node), label=node, color=f"C{i}")
ax2.axhline(VDD / 2, color="gray", ls=":")
ax2.set(xlabel="time (ns)", ylabel="voltage (V)")
ax2.legend(loc="right")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
