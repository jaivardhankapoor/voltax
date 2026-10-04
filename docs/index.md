# Voltax

<p align="center"><img src="assets/voltax.svg" width="360" alt="Voltax logo"></p>

**Voltax is a SPICE-class circuit simulator written in JAX.** Every analysis
(DC, transient, AC) is a differentiable, `jit`-able, `vmap`-able function of
the circuit, so you can fit device parameters to measurements, optimize a
design against a spec, run Bayesian inference over process parameters, or
train an analog neural network, all with ordinary JAX tools.

```python
import equinox as eqx
import jax.numpy as jnp
import voltax as vx

b = vx.CircuitBuilder()
b.vsource("in", "0", 0.0, ac=1.0)
b.resistor("in", "out", 1e3, name="R1")
b.capacitor("out", "0", 100e-9, name="C1")
circuit = b.build()

# magnitude at 1 kHz, and its gradient with respect to every parameter
def gain_db(circuit):
    return vx.ac(circuit, jnp.array([1e3])).db("out")[0]

print(gain_db(circuit))                                  # -1.45 dB
grads = eqx.filter_grad(gain_db)(circuit)
print(grads.elements["Resistor"].log_r)                  # d dB / d log R
```

## Why Voltax

- **Exact gradients, cheaply.** Newton solves are differentiated by the
  implicit function theorem: one extra linear solve per time step, however
  many Newton iterations the forward pass took.
- **One idea for every device.** A device model is a Python class with two
  methods (its currents and its charges) written for a single device and
  vectorized automatically over thousands. No stamping, no global indices.
  See [Writing your own element](concepts/custom-elements.md).
- **Batteries included.** R, C, L, coupled inductors, independent and
  controlled sources, diodes, BJTs, EKV and Level-1 MOSFETs, ideal and
  behavioral op-amps, switches, trainable conductances, nonlinear R/C and
  behavioral (`B`) sources, a SPICE netlist parser for real-world decks
  (parameterized subcircuits, `.param` expressions, `.include`/`.lib`
  corners, binned models), and a CMOS gate library.
- **Industry compact models.** Verilog-A models compile to elements;
  BSIM4 (fetched on first use; its licence is CC BY-NC 4.0) matches
  ngspice's BSIM4 to round-off, including on the sky130/gf180mcu decks. See
  [Verilog-A compact models](concepts/verilog-a.md).
- **JAX all the way down.** Circuits are [Equinox](https://docs.kidger.site/equinox/)
  modules: `jax.vmap` over 64 design variants, `eqx.filter_grad` over every
  parameter, `jax.jit` the whole optimization loop.

## Where to go next

| If you want to... | Read |
|---|---|
| simulate your first circuit | [Getting started](getting-started.md) |
| understand the equations being solved | [The circuit equations](concepts/formulation.md) |
| optimize or infer parameters | [Gradients and optimization](concepts/gradients.md) |
| turn waveforms into specs (delay, bandwidth, energy) | [Measurements](concepts/measurements.md) |
| add a new device | [Writing your own element](concepts/custom-elements.md) |
| load an existing SPICE deck | [SPICE netlists](concepts/netlists.md) |
| see complete applications | [Examples](examples/index.md) |
