r"""# A CMOS inverter chain from a SPICE netlist

Voltax reads ordinary SPICE netlists, so existing designs can be simulated and
differentiated without rewriting them in Python. Here a three-stage CMOS
inverter chain is described as text, parsed into a `Circuit`, and analysed
three ways: a DC transfer curve, a transient pulse response, and the gradient
of the propagation delay with respect to every transistor width.

The transistors use the EKV model, which is smooth from weak to strong
inversion. The propagation delay is measured where the output crosses
$V_{DD}/2$; we locate the crossing sample with a (non-differentiable) search
and then interpolate linearly between the two neighbouring samples,

$$ t_{50} = t_k + (t_{k+1} - t_k)\,
   \frac{V_{DD}/2 - v_k}{v_{k+1} - v_k}, $$

which *is* differentiable in the waveform values, so `jax.grad` gives
$\partial t_{pd} / \partial \log W$ for all six devices in one backward pass.

What to look at: the transfer curve switches near mid-rail; the delay
gradients are negative (wider = faster) and largest for the devices that
drive the critical edges.
"""

# %% Setup
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

import voltax as vx

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAST = os.environ.get("VOLTAX_FAST") == "1"
FIGURE = Path(__file__).parent / "figures" / f"{Path(__file__).stem}.png"
VDD = 1.2


# %% The netlist
# Device and node names are kept exactly as written, so results are addressed
# with them: `sol.v("out")`, `sol.i("Vdd")`, `circuit.get("M1n", "w")`.
NETLIST = """
* Three-stage CMOS inverter chain
.model nch nmos (kp=400u vto=0.4 lambda=0.04)
.model pch pmos (kp=200u vto=0.4 lambda=0.04)

Vdd vdd 0 1.2
Vin in  0 PULSE(0 1.2 0.2n 50p 50p 1.5n 3n)

M1p mid1 in   vdd vdd pch w=2u l=0.13u
M1n mid1 in   0   0   nch w=1u l=0.13u
C1  mid1 0 5f
M2p mid2 mid1 vdd vdd pch w=2u l=0.13u
M2n mid2 mid1 0   0   nch w=1u l=0.13u
C2  mid2 0 5f
M3p out  mid2 vdd vdd pch w=2u l=0.13u
M3n out  mid2 0   0   nch w=1u l=0.13u
C3  out  0 20f
.end
"""

circuit = vx.parse_netlist(NETLIST)
print(circuit.summary())
# Devices of the same kind are fused into vectorized groups; look up which
# group (and index) a named device landed in:
print("M1n lives in", circuit.layout.device("M1n"), "with W =", circuit.get("M1n", "w"))


# %% DC transfer curve of the chain
# `with_dc` swaps the input's PULSE for a DC value; `jax.vmap` sweeps it.
vin_group, _ = circuit.layout.device("Vin")


def dc_out(vin):
    src = circuit.elements[vin_group].with_dc(vin)
    sol = vx.dc(circuit.replace(vin_group, src))
    return sol.v("mid1"), sol.v("out")


vin_sweep = jnp.linspace(0.0, VDD, 41 if FAST else 121)
vtc_mid1, vtc_out = jax.vmap(dc_out)(vin_sweep)
v_switch = vin_sweep[jnp.argmin(jnp.abs(vtc_mid1 - VDD / 2))]
print(f"first-stage switching threshold ~ {v_switch:.3f} V")


# %% Transient response and propagation delay
ts = jnp.linspace(0.0, 3e-9, 301 if FAST else 1201)


def chain_delay(c):
    """Average of rising- and falling-input 50% delays from `in` to `out`.

    `vx.measure.delay` is SPICE's ``.meas TRIG ... TARG``, differentiable:
    crossing times are interpolated between samples.
    """
    sol = vx.transient(c, ts)
    vin, vout = sol.v("in"), sol.v("out")
    # three inversions: a rising input gives a falling output
    t_phl = vx.measure.delay(ts, vin, vout, VDD / 2, trig_direction="rise",
                             targ_direction="fall")
    t_plh = vx.measure.delay(ts, vin, vout, VDD / 2, trig_direction="fall",
                             targ_direction="rise")
    return 0.5 * (t_phl + t_plh)


sol = vx.transient(circuit, ts)
print(f"transient converged at every step: {bool(sol.converged.all())}")
print(f"chain delay: {chain_delay(circuit) * 1e12:.1f} ps")


# %% Gradient of the delay with respect to every transistor width
# `jax.grad` of a function of the whole circuit returns a circuit-shaped
# pytree of gradients; MOSFET widths live in `log_w`, so we get
# d(delay)/d(log W) = W * d(delay)/dW per device.
grads = jax.grad(chain_delay)(circuit)
print("d(delay)/d(log W) in ps (negative = upsizing speeds the chain up):")
for name in ("M1p", "M1n", "M2p", "M2n", "M3p", "M3n"):
    group, idx = circuit.layout.device(name)
    print(f"  {name}: {grads.elements[group].log_w[idx] * 1e12:+8.2f}")


# %% Plot
FIGURE.parent.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(vin_sweep, vtc_mid1, label="v(mid1)")
ax1.plot(vin_sweep, vtc_out, label="v(out)")
ax1.set(xlabel="v(in) (V)", ylabel="voltage (V)", title="DC transfer curves")
ax1.legend()
for node in ("in", "mid1", "mid2", "out"):
    ax2.plot(ts * 1e9, sol.v(node), label=node)
ax2.axhline(VDD / 2, color="gray", ls=":")
ax2.set(xlabel="time (ns)", ylabel="voltage (V)", title="Pulse response")
ax2.legend(loc="right")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print(f"saved {FIGURE}")
