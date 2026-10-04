# Measurements

Design specs are rarely raw waveforms. They are a delay, a bandwidth, a
phase margin, an energy per operation. `voltax.measure` turns waveforms into
those numbers. It plays the role of SPICE's `.meas`, with one difference: every
measurement is differentiable, so a spec can be the loss you optimize.

```python
import jax
import jax.numpy as jnp
import voltax as vx
from voltax import measure

b = vx.CircuitBuilder()
b.vsource("in", "0", vx.signals.Pulse(0, 1, delay=1e-9, rise=1e-11, fall=1e-11,
                                      width=10e-9, period=20e-9), name="Vin")
b.resistor("in", "out", 1e3, name="R1")
b.capacitor("out", "0", 1e-12, name="C1")
c = b.build()
ts = jnp.linspace(0, 20e-9, 2001)
sol = vx.transient(c, ts)

print(measure.delay(ts, sol.v("in"), sol.v("out"), 0.5))      # ~0.69 ns = RC ln 2
print(measure.rise_time(ts[:1000], sol.v("out")[:1000], final=1.0))  # ~2.2 ns = RC ln 9
```

## Time domain

| function | measures |
|---|---|
| `crossing(t, y, level, direction, *, k, after)` | time of the k-th crossing of `level` |
| `crossings(t, y, level, direction)` | all crossings (fixed-size, `nan`-padded) |
| `delay(t, trig, targ, level, ...)` | `.meas TRIG ... TARG`: from a trigger edge to the next target edge |
| `period(t, y)`, `frequency(t, y)` | mean oscillation period from mid-swing crossings, skipping start-up |
| `rise_time`, `fall_time` | 10–90 % edge time (levels configurable) |
| `overshoot`, `settling_time` | step-response quality |
| `integral`, `average`, `rms` | trapezoidal integrals over time |
| `power(circuit, sol, device)`, `energy(...)` | power/energy absorbed by any device, from its own model |

Crossing times are interpolated linearly between samples. Gradients are
exact as long as the time grid resolves the edge; choosing *which* interval
contains a crossing is discrete, so refine the grid near the edges you
optimize.

Measurements that don't exist (no crossing, too few periods) return `nan`
instead of raising, so they keep working inside `jit` and `vmap`.

## Energy

`power` evaluates the device's own `currents` and `charges` at every time
point, so it works for every element, including custom ones. Sources that
deliver power come out negative. Charging a capacitor through a resistor
shows the classic result: the source delivers \(C V^2\), and half is lost in
the resistor whatever its value:

```python
b = vx.CircuitBuilder()
b.vsource("in", "0", 1.0, name="V1")
b.resistor("in", "out", 1e3, name="R1")
b.capacitor("out", "0", 1e-9, name="C1")
rc = b.build()
s = vx.transient(rc, jnp.linspace(0, 10e-6, 4001), ic=rc.state(v={"in": 1.0}),
                 method="trap")
for dev in ["V1", "R1", "C1"]:
    print(dev, measure.energy(rc, s, dev))   # -1e-9, 0.5e-9, 0.5e-9 J
```

!!! note "Energy you are not counting"
    `energy(circuit, sol, "Vdd")` counts only what the supply delivers. Charge
    drawn through an ideal *input* source (charging the first gate, say) is
    not included. When optimizing energy, keep the gates that sources drive
    at a fixed size, or the optimizer will make them large for free.

## Frequency domain

| function | measures |
|---|---|
| `bandwidth(freqs, h, drop_db=3)` | first frequency `drop_db` below the low-frequency gain |
| `unity_gain_frequency(freqs, h)` | 0 dB crossing |
| `phase_margin(freqs, h)` | \(180° + \angle h\) at the unity-gain frequency |
| `gain_margin(freqs, h)` | \(-\lvert h\rvert_{dB}\) where the phase reaches −180° |

Pass complex responses straight from `vx.ac`: `measure.bandwidth(freqs, res.v("out"))`.
Crossings are interpolated in log-frequency.

## Optimizing a spec

```python
def delay_of(log_r):
    circuit = c.set("R1", r=jnp.exp(log_r))
    s = vx.transient(circuit, ts)
    return measure.delay(ts, s.v("in"), s.v("out"), 0.5)

print(jax.grad(delay_of)(jnp.log(1e3)))   # d delay / d log R = delay
```
