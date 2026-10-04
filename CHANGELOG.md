# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-04

First public release.

### Added

- **Core.** Charge-oriented DAE formulation `d/dt q(z) + f(z, t) = 0`
  assembled from vectorized device groups (`Element`, `Circuit`,
  `CircuitBuilder`).
- **Analyses.** DC operating point (damped Newton with gmin stepping),
  `dc_sweep` with continuation, transient (backward Euler and trapezoidal),
  `time_grid` with breakpoint refinement, small-signal AC, `linearize`.
- **Gradients.** Implicit differentiation of every Newton solve (adjoint
  method); time points are differentiable.
- **Measurements.** `voltax.measure`: crossings, delay, period, rise and
  settling time, energy and power per device, bandwidth, unity-gain
  frequency, phase and gain margin.
- **Devices.** R, C, L, transformer, V/I sources with differentiable
  waveforms (pulse, sine, PWL, exp, SFFM, AM, function), E/G/F/H, diode, BJT,
  EKV and Level-1 MOSFETs, ideal and behavioral op-amps, switch, trainable
  conductance, nonlinear R/C, behavioral (B) sources.
- **Netlists.** SPICE parser with `.param`/`.func` expressions,
  parameterized and conditional subcircuits, `.include`/`.lib` corners,
  binned model cards and a `models=` hook; reads the sky130, gf180mcu and IHP
  model decks.
- **Sparse solver.** Stamped Jacobians and a pure-JAX supernodal LU with
  nested-dissection ordering, selected automatically above 200 unknowns.
- **Verilog-A.** `voltax.va` compiles Verilog-A compact models to elements;
  BSIM4 4.8 (fetched on demand, not distributed) matches ngspice to ~1e-14;
  differentiability audit (`voltax.va.audit`).
- **Library.** CMOS gates, adders and ring oscillators.
- 13 examples, ngspice benchmarks and MkDocs documentation.

[0.1.0]: https://github.com/jaivardhankapoor/voltax/releases/tag/v0.1.0
