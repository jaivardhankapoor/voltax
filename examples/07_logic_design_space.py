r"""# Searching every way to wire two transistors, then sizing the winner

What logic can you build from one PMOS and one NMOS? We put both transistors
on a "breadboard": each of their six pins (drain, gate, source) can be wired
to one of the nets {vdd, a, b, y, gnd}. That is $5^6 = 15\,625$ circuits,
each to be simulated for all four input combinations: 62 500 DC operating
points. Because a circuit is a pytree and the wiring is just an array of
switch conductances, `jax.vmap` evaluates them all as one batched program.

A wiring realizes a Boolean function if, for every input, the output is a
clean logic level ($y < 0.2\,V_{DD}$ or $y > 0.8\,V_{DD}$). The output is
loaded by 100 kOhm to mid-supply, so a floating output reads as an invalid
$V_{DD}/2$ rather than a lucky leakage level.

The search rediscovers the two classic two-transistor circuits: the CMOS
inverter and the transmission gate. We then turn from discrete search to
gradients and size the inverter. Its switching threshold $V_M$ (where
$v_\text{out} = v_\text{in}$) is the DC solution of an inverter whose output
is tied back to its input, so $\partial V_M / \partial \log W_p$ comes from
implicit differentiation of a single DC solve, and a few Newton steps on
$V_M(W_p) = V_{DD}/2$ give the width for symmetric noise margins.

What to look at: the table of realizable functions (no NAND/NOR/XOR, as
expected with two transistors), the four equivalent inverter wirings
(drain/source swaps), and the PMOS width that centres $V_M$.
In fast mode only input `a` is on the breadboard ($4^6 = 4096$ wirings).
"""

# %% Setup
import itertools
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import voltax as vx
from voltax.library import cmos

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"
VDD = 1.2
G_ON, G_OFF = 1e-2, 1e-12  # a closed / open breadboard switch (S)

INPUT_NETS = ["a"] if FAST else ["a", "b"]
NETS = ["vdd", *INPUT_NETS, "y", "0"]
PINS = ["p.d", "p.g", "p.s", "n.d", "n.g", "n.s"]


# %% The breadboard
# One switch (a linear conductance) from every pin to every net. A wiring is
# a choice of net per pin; it becomes a (6 * len(NETS),) conductance vector.
b = vx.CircuitBuilder()
for pin in PINS:
    for net in NETS:
        b.conductance(pin, net, G_OFF, transform="linear", group="switches")
b.pmos("p.d", "p.g", "p.s", "vdd", w=2e-6)  # bulk fixed to vdd
b.nmos("n.d", "n.g", "n.s", "0", w=1e-6)  # bulk fixed to gnd
b.vsource("mid", "0", VDD / 2, name="Vmid")
b.resistor("y", "mid", 100e3, name="Rload")
for net in ["vdd", *INPUT_NETS]:
    b.vsource(net, "0", 0.0, name=f"V{net}", group="drivers")
board = b.build()
print(board.summary())

input_rows = jnp.array(list(itertools.product([0.0, 1.0], repeat=len(INPUT_NETS))))


def output_levels(wiring):
    """v(y) for every input row, for one wiring (an int per pin)."""
    closed = jax.nn.one_hot(wiring, len(NETS)).reshape(-1)
    g = G_OFF + (G_ON - G_OFF) * closed
    c = board.replace("switches", eqx.tree_at(lambda e: e.theta,
                                              board.elements["switches"], g))

    def one_row(x):
        drive = jnp.concatenate([jnp.array([VDD]), VDD * x])
        return vx.dc(c.replace("drivers", c.elements["drivers"].with_dc(drive))).v("y")

    return jax.vmap(one_row)(input_rows)


# %% Simulate every wiring
wirings = jnp.array(list(itertools.product(range(len(NETS)), repeat=len(PINS))))
t0 = time.time()
# vmap batches of 1024 wirings; lax.map loops over the batches to bound memory
y = jax.jit(lambda w: jax.lax.map(output_levels, w, batch_size=1024))(wirings)
y = np.asarray(y)
print(f"{len(wirings)} wirings x {len(input_rows)} inputs = {y.size} DC solves "
      f"in {time.time() - t0:.1f}s")


# %% Which functions did we get?
valid = ((y < 0.2 * VDD) | (y > 0.8 * VDD)).all(axis=1)
truth = y > VDD / 2
a, *rest = np.asarray(input_rows, bool).T
NAMED = {"0": a & ~a, "1": a | ~a, "a": a, "NOT a": ~a}
if rest:
    (bb,) = rest
    NAMED |= {"b": bb, "NOT b": ~bb, "a AND b": a & bb, "a OR b": a | bb,
              "NAND": ~(a & bb), "NOR": ~(a | bb), "XOR": a ^ bb, "XNOR": ~(a ^ bb)}


def name_of(table):
    return next((n for n, t in NAMED.items() if (t == table).all()), "other")


names = np.array([name_of(t) if ok else "invalid" for t, ok in zip(truth, valid)])
counts = {n: int((names == n).sum()) for n in [*NAMED, "other", "invalid"]}
print("function       wirings")
for n, k in counts.items():
    if k:
        print(f"  {n:12s} {k:6d}")
print("not realizable:", ", ".join(n for n, k in counts.items() if not k))


def describe(wiring):
    return ", ".join(f"{pin}->{NETS[k]}" for pin, k in zip(PINS, wiring))


for target in ("NOT a", "a"):
    print(f"wirings computing y = {target}:")
    for k in np.flatnonzero(names == target):
        print(f"  {describe(np.asarray(wirings[k]))}")


# %% From search to gradients: centre the inverter's switching threshold
# With input and output on the same node, the DC solution *is* V_M.
def switching_threshold(log_wp, wn=1e-6):
    b = vx.CircuitBuilder()
    b.vsource("vdd", "0", VDD)
    cmos.inverter(b, "x", "x", vdd="vdd", wn=wn, wp=jnp.exp(log_wp))
    return vx.dc(b.build()).v("x")


log_wp = jnp.log(1e-6)  # start from equal widths
vm_and_slope = jax.jit(jax.value_and_grad(switching_threshold))
for i in range(4):  # Newton's method on V_M(log W_p) = VDD / 2
    vm, slope = vm_and_slope(log_wp)
    print(f"newton {i}: Wp = {jnp.exp(log_wp) * 1e6:.3f} um  V_M = {vm:.4f} V  "
          f"dV_M/dlogWp = {slope:.4f} V")
    log_wp = log_wp - (vm - VDD / 2) / slope
# With equal thresholds the answer is the mobility ratio kp_n / kp_p = 2.
print(f"symmetric inverter: Wp/Wn = {jnp.exp(log_wp) / 1e-6:.3f}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
shown = {n: k for n, k in counts.items() if k and n != "invalid"}
ax1.barh(list(shown), list(shown.values()), color="C0")
ax1.set_xscale("log")
ax1.set(xlabel="number of wirings", title=f"Functions realized by {len(wirings)} "
        f"wirings ({counts['invalid']} invalid)")
ax1.invert_yaxis()
ratios = jnp.logspace(-1, 1.3, 60)
vms = jax.vmap(switching_threshold)(jnp.log(ratios * 1e-6))
ax2.semilogx(ratios, vms)
ax2.axhline(VDD / 2, color="gray", ls=":")
ax2.plot(jnp.exp(log_wp) / 1e-6, VDD / 2, "o", color="C3", label="Newton solution")
ax2.set(xlabel="Wp / Wn", ylabel="V_M (V)", title="Inverter switching threshold")
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
