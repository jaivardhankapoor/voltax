# Examples

Numbered, self-contained scripts in the repository's `examples/` folder,
from a first gradient to Bayesian inference and analog machine learning.
Run any of them with `uv run python examples/<file>`; set `VOLTAX_FAST=1` for a
quick version (this is what CI runs).

| # | Example | What it shows | Key API |
|---|---|---|---|
| 01 | [Fitting an RC ladder to a measured step response](01_rc_fitting.md) | Recover R1, R2, C1, C2 of an RC ladder from step responses; why two observed nodes are needed | `CircuitBuilder`, `vx.transient`, `jax.grad`, `optax.lbfgs` |
| 02 | [A CMOS inverter chain from a SPICE netlist](02_netlist_cmos_inverter.md) | A SPICE netlist: DC transfer curve, pulse response, gradient of propagation delay w.r.t. every transistor width | `vx.parse_netlist`, `with_dc` + `jax.vmap`, `jax.grad` of a whole `Circuit` |
| 03 | [Designing an active filter by gradient descent on its Bode plot](03_opamp_filter_design.md) | Fit the capacitors of a 4th-order Sallen-Key low-pass to a Butterworth response, beating the textbook values under finite op-amp bandwidth | `b.opamp`, `vx.ac`, `group=` + `eqx.tree_at`, `optax.adam` |
| 04 | [A 200-transistor 4-bit ripple-carry adder from library gates](04_adder_4bit.md) | 200-transistor ripple-carry adder from library gates: truth table from DC solves, carry-ripple transient | `voltax.library.cmos`, `b.scope`, `with_dc`, `jax.lax.map` |
| 05 | [Sizing 760 wires of a power grid with one adjoint solve per step](05_power_grid_opt.md) | Size 760 power-grid wires against IR drop; adjoint gradient checked against finite differences | `b.conductance(transform="sigmoid")`, `b.isource`, `vx.dc` |
| 06 | [Discovering a filter topology from a "circuit soup"](06_topology_discovery.md) | Start from a fully connected R+C "soup" and let a sparsity penalty discover an RC ladder; prune and verify | `vx.ac`, sigmoid `Conductance`, log-sum sparsity |
| 07 | [Searching every way to wire two transistors, then sizing the winner](07_logic_design_space.md) | Simulate all 15 625 ways to wire one PMOS + one NMOS, rediscover the inverter and the transmission gate, then size the inverter with Newton on gradients | `jax.vmap` over circuits, `jax.lax.map(batch_size=)`, `jax.value_and_grad` of a DC solve |
| 08 | [Inferring transistor parameters from a ring oscillator with NUTS](08_ring_osc_inference.md) | Posterior over shared NMOS/PMOS process parameters from a noisy ring-oscillator waveform | `cmos.ring_oscillator`, `circuit.state`, shared `MOSProcess`, blackjax NUTS |
| 09 | [Gradient-free SMC vs. gradient-based NUTS on an RC filter](09_rc_smc_inference.md) | Gradient-free tempered SMC vs. gradient-based NUTS on a ridge-shaped RC posterior | blackjax `adaptive_tempered_smc`, `window_adaptation` |
| 10 | [Writing your own device: a memristor with an internal state](10_custom_element.md) | Write a new device: an HP memristor with an internal state; pinched hysteresis loops; gradients w.r.t. its parameters | `vx.Element` (`currents`, `charges`, `n_internal`), `b.add`, `circuit.state(x=...)`, `circuit.set(freq=...)` |
| 11 | [Training an analog convolutional network made of resistors and diodes](11_mnist_analog_cnn.md) | Train a resistor-diode convolutional network on 8x8 digits; check accuracy with E96 resistor values | softplus `Conductance`, weight sharing by indexing, `vx.Options(gmin_steps=0)` |
| 12 | [Energy per operation vs. supply voltage: power-aware sizing of CMOS logic](12_energy_delay_optimization.md) | Energy per operation and delay of a CMOS inverter chain vs. VDD (C_eff V^2 check, minimum-EDP supply), then minimize energy under a delay constraint over VDD and stage widths; adjoint gradients checked against finite differences | `vx.measure.energy`, `vx.measure.delay`, `vx.time_grid`, `method="trap"`, `jax.jacrev`, `optax.lbfgs` augmented Lagrangian |
| 13 | [Battery equivalent-circuit model: identifiability and energy prediction](13_battery_ecm.md) | Fit a two-RC battery model to an HPPC-like pulse test from 16 random inits: the RC-branch swap is the only degeneracy, and every fit predicts the total heat to <0.01% | `b.isource` + `signals.PWL` load, `vx.time_grid`, `jax.vmap` over `optax.lbfgs`, `vx.measure.energy` |
