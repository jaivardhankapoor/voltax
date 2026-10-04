# Voltax examples

Numbered, self-contained scripts that build from a first gradient through a
transient simulation to Bayesian inference and analog machine learning. Each
file starts with a literate docstring (motivation, the math, what to look
for) and is split into `# %%` cells, so it can be read as a notebook (VS Code,
Jupyter via jupytext) or rendered into the docs.

| # | File | What it shows | Key API |
|---|------|---------------|---------|
| 01 | [`01_rc_fitting.py`](01_rc_fitting.py) | Recover R1, R2, C1, C2 of an RC ladder from step responses; why two observed nodes are needed | `CircuitBuilder`, `vx.transient`, `jax.grad`, `optax.lbfgs` |
| 02 | [`02_netlist_cmos_inverter.py`](02_netlist_cmos_inverter.py) | A SPICE netlist: DC transfer curve, pulse response, gradient of propagation delay w.r.t. every transistor width | `vx.parse_netlist`, `with_dc` + `jax.vmap`, `jax.grad` of a whole `Circuit` |
| 03 | [`03_opamp_filter_design.py`](03_opamp_filter_design.py) | Fit the capacitors of a 4th-order Sallen-Key low-pass to a Butterworth response, beating the textbook values under finite op-amp bandwidth | `b.opamp`, `vx.ac`, `group=` + `eqx.tree_at`, `optax.adam` |
| 04 | [`04_adder_4bit.py`](04_adder_4bit.py) | 200-transistor ripple-carry adder from library gates: truth table from DC solves, carry-ripple transient | `voltax.library.cmos`, `b.scope`, `with_dc`, `jax.lax.map` |
| 05 | [`05_power_grid_opt.py`](05_power_grid_opt.py) | Size 760 power-grid wires against IR drop; adjoint gradient checked against finite differences | `b.conductance(transform="sigmoid")`, `b.isource`, `vx.dc` |
| 06 | [`06_topology_discovery.py`](06_topology_discovery.py) | Start from a fully connected R+C "soup" and let a sparsity penalty discover an RC ladder; prune and verify | `vx.ac`, sigmoid `Conductance`, log-sum sparsity |
| 07 | [`07_logic_design_space.py`](07_logic_design_space.py) | Simulate all 15 625 ways to wire one PMOS + one NMOS, rediscover the inverter and the transmission gate, then size the inverter with Newton on gradients | `jax.vmap` over circuits, `jax.lax.map(batch_size=)`, `jax.value_and_grad` of a DC solve |
| 08 | [`08_ring_osc_inference.py`](08_ring_osc_inference.py) | Posterior over shared NMOS/PMOS process parameters from a noisy ring-oscillator waveform | `cmos.ring_oscillator`, `circuit.state`, shared `MOSProcess`, blackjax NUTS |
| 09 | [`09_rc_smc_inference.py`](09_rc_smc_inference.py) | Gradient-free tempered SMC vs. gradient-based NUTS on a ridge-shaped RC posterior | blackjax `adaptive_tempered_smc`, `window_adaptation` |
| 10 | [`10_custom_element.py`](10_custom_element.py) | Write a new device: an HP memristor with an internal state; pinched hysteresis loops; gradients w.r.t. its parameters | `vx.Element` (`currents`, `charges`, `n_internal`), `b.add`, `circuit.state(x=...)`, `circuit.set(freq=...)` |
| 11 | [`11_mnist_analog_cnn.py`](11_mnist_analog_cnn.py) | Train a resistor-diode convolutional network on 8x8 digits; check accuracy with E96 resistor values | softplus `Conductance`, weight sharing by indexing, `vx.Options(gmin_steps=0)` |
| 12 | [`12_energy_delay_optimization.py`](12_energy_delay_optimization.py) | Energy per operation and delay of a CMOS inverter chain vs. VDD (C_eff V^2 check, minimum-EDP supply), then minimize energy under a delay constraint over VDD and stage widths; adjoint gradients checked against finite differences | `vx.measure.energy`, `vx.measure.delay`, `vx.time_grid`, `method="trap"`, `jax.jacrev`, `optax.lbfgs` augmented Lagrangian |
| 13 | [`13_battery_ecm.py`](13_battery_ecm.py) | Fit a two-RC battery model to an HPPC-like pulse test from 16 random inits: the RC-branch swap is the only degeneracy, and every fit predicts the total heat to <0.01% | `b.isource` + `signals.PWL` load, `vx.time_grid`, `jax.vmap` over `optax.lbfgs`, `vx.measure.energy` |

## Running

Install the extras once from the repository root, then run any script:

```bash
uv sync --extra examples         # optax, blackjax, scikit-learn, matplotlib
uv run python examples/01_rc_fitting.py
```

Set `VOLTAX_FAST=1` for a quick run: every script then shrinks its problem
size or iteration counts so it finishes in about a minute or less on a laptop
CPU (this is what the test suite's smoke runs use):

```bash
VOLTAX_FAST=1 uv run python examples/08_ring_osc_inference.py
```

Figures are written to `examples/figures/<script name>.png` (git-ignored).
Without a display, matplotlib's non-interactive Agg backend is used.
