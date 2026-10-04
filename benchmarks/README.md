# voltax vs ngspice benchmarks

Each script builds the same circuit in voltax and in ngspice (the ngspice
netlist is generated from the same Python parameters, or is literally the same
text that voltax parses), runs ngspice in batch mode, reads its binary
rawfile, and reports accuracy and runtime. Plots go to `benchmarks/figures/`,
which git ignores.

| Script | Circuit | What it checks |
|---|---|---|
| `benchmark_rc.py` | RC low-pass step, 100 random (R, C) | Linear transient on a matched backward-Euler grid; error against the exact BE recurrence and against `1 - exp(-t/tau)`; `jax.vmap` over every configuration in one call |
| `benchmark_power_grid.py` | N x N resistive power grid (up to 40 x 40, 1600 nodes, 3120 R) | Linear DC operating point at every node; adjoint gradient `d v / d log R` against an ngspice central finite difference |
| `benchmark_inverter_ngspice.py` | 3-stage CMOS inverter chain | The netlist API: one netlist text goes to both simulators. Exact Level 1, smoothed Level 1 (`level=1` parser default), and voltax EKV against ngspice BSIM4 |
| `benchmark_ota.py` | 5-transistor OTA, Level 1 with body effect | DC operating point, AC gain and phase, transient |
| `benchmark_netlists.py` | Realistic decks: parameterized subcircuit hierarchy, `.lib` corners (Level 1), B-source nonlinear circuit, `EXP`/`SFFM`/`AM`/`PWL r= td=` sources, binned Level-1 models | The netlist parser: the *same text* in both simulators, transient on a matched BE grid; bin selection against ngspice BSIM3 binning on edge cases |
| `benchmark_pdk_decks.py` | sky130, gf180mcu and IHP SG13G2 model libraries (mirrored under `~/.cache/voltax/pdk`) | The whole PDK hierarchy parses and solves (EKV fallbacks for BSIM4/PSP); bin selection against ngspice on 112 sky130 and 98 gf180mcu geometries |
| `benchmark_ring_oscillator.py` | 5- to 51-stage ring oscillator (`library.cmos`) | Exact Level 1 over many oscillation periods, plus the period |
| `benchmark_sparse.py` | power grid up to 100 x 100, ring oscillator up to 501 stages | Dense vs sparse linear solver (`Options(solver=...)`): scaling, crossover, compile time, Jacobian cost ([section below](#sparse-vs-dense-linear-algebra)) |

## Running

```bash
uv run python benchmarks/benchmark_rc.py              # full size
VOLTAX_FAST=1 uv run python benchmarks/benchmark_rc.py   # smaller problems, each < ~30 s
```

You need ngspice on `PATH` (tested with ngspice-42). Shared code, including
the ngspice runner, rawfile parser, timing and table helpers, lives in
`_common.py`. Import `_common` before `jax`, because it pins JAX to a single
CPU thread so the comparison with single-threaded ngspice is fair. It also
pins OpenBLAS (`OPENBLAS_NUM_THREADS=1`, plus `threadpoolctl`), which XLA's
CPU LU and triangular solves call. On a busy machine, OpenBLAS's spinning
worker threads made a 400 x 400 dense LU 100 to 1000 times slower in our
runs. Dense timings taken without pinning are unreliable above roughly 150
unknowns.

## How the comparison works

**Matched discretization.** Adaptive time stepping makes two simulators'
waveforms differ by their truncation errors. That difference tells you
nothing about whether the device equations are right. So the transient
benchmarks run ngspice with backward Euler (`.options method=gear maxord=1`),
a maximum step `dt`, and tight Newton tolerances
(`reltol=1e-6 vntol=1e-9 abstol=1e-15`). voltax then integrates with
`method="be"` on exactly ngspice's accepted time points. Both simulators solve
the same discrete equations, so any remaining difference comes from the
models or the Newton tolerances. With `UIC`, ngspice does not write the
`t = 0` point, so the initial condition is prepended before comparing.

**Exact Level 1.** `vx.Level1MOSFET(smooth=0.0)` is the non-smooth SPICE
square law. The scripts pass it to `CircuitBuilder` (or to `parse_netlist`)
as `mos_model=functools.partial(vx.Level1MOSFET, smooth=0.0)`. They give
ngspice `cgso = cgdo` equal to voltax's `MOSProcess.cov`, so the overlap
capacitances match too. They do not set `tox`: see the caveats below.

**Timing.**
- `voltax 1st call`: trace, compile and run.
- `voltax run`: the best of several warm calls.
- `voltax compile`: the difference between the two.
- `ngspice wall`: the best wall time of the whole `ngspice -b` process,
  including start-up and parsing.
- `ngspice analysis`: ngspice's own "Total analysis time", at 1 ms resolution.

Both simulators run on the same time grid, so they compute the same number of
points. The machine was a shared 4-core VM, so timings vary by ±30 %.

## Latest results

ngspice-42, JAX 0.10.2 on CPU (1 thread), float64.

### Accuracy (full mode; FAST mode gives the same errors)

| Benchmark | Case | Max abs error | Relative error |
|---|---|---|---|
| RC (100 configs) | voltax vs ngspice, same BE grid | 8.6e-9 V | 8.6e-9 |
| | ngspice vs exact BE recurrence | 2.8e-14 V | |
| | voltax (`gmin=0`) vs exact BE recurrence | ~5e-15 V | |
| | voltax `vmap` vs exact BE recurrence | 8.6e-9 V | 8.6e-9 |
| Power grid DC | 10x10 / 20x20 / 30x30 / 40x40 | 2.7e-12 / 1.5e-11 / 3.9e-11 / 7.4e-11 V | ≤ 7.8e-11 |
| | adjoint `dV/dlogR` vs ngspice central FD (40x40) | | 2.4e-8 (FD truncation) |
| Inverter chain | Level 1 exact (`smooth=0`) | 6.3e-9 V | 5.2e-9 |
| | Level 1 `smooth=0.02` (parser default) | 3.7e-5 V | 3.0e-5 |
| | voltax EKV vs ngspice BSIM4 (different models) | 1.25 V, edges up to 205 ps apart | 1.04 |
| OTA (Level 1 + body effect) | DC node voltages / I(Vdd) | 4.8e-9 V / | 3.9e-9 / 3.0e-7 |
| | AC gain (49.5 dB), 141 frequencies | 8.8e-5 dB, 2.9e-4 deg | 1.0e-5 |
| | transient | 2.5e-8 V | 2.1e-8 |
| Ring oscillator | 5 / 21 / 51 stages, 5 ns (2.6 to 26 periods) | 2.2e-7 V | 1.8e-7; period 8e-10 |

The ~1e-8 to 1e-7 relative floor in the voltax results is voltax's default
node-to-ground `gmin = 1e-12` S. ngspice only puts gmin across junctions.
Examples: 1e-12 S x 10 kΩ in the RC, and the ~5 MΩ OTA output for the AC and
I(Vdd) numbers. With `vx.Options(gmin=0)` the RC agrees to 1e-14.

### Netlist parser (`benchmark_netlists.py`, full mode)

| Case | Points | Max abs error | Relative error |
|---|---|---|---|
| Parameterized hierarchy (3 levels, `m=`, `.global`, `.func`, ternary) | 5613 | 1.6e-9 V | 5.0e-9 |
| `.lib` corners tt / ff / ss (Level 1 inverter chain, `.include`d params) | ~3800 | 5.4e-8 / 4.0e-8 / 8.5e-8 V | ≤ 5.4e-8 |
| B / `E vol=` / `G cur=` / `TABLE` / `POLY(2)` / `i(Vsense)` nonlinear circuit | 32970 | 1.2e-7 V | 1.9e-7 |
| `TABLE` + `POLY(2)` at DC (away from ngspice's rounded TABLE corners) | 6 | 2.3e-9 V | 1.0e-9 |
| `EXP`, `SFFM`, `AM`, `PWL r= td=`, `PULSE`, damped `SIN` current | 12902 | 1.4e-9 V | 9.5e-10 |
| Binned Level-1 inverters (ngspice gets the de-binned deck) | 4660 | 1.9e-8 V | 1.2e-8 |
| Bin selection vs ngspice BSIM3 binning, L/W on and within 1 nm of edges | 35 | 35/35 agree | |

ngspice does not bin Level-1 models, hence the de-binned comparison plus
a separate bin-selection check. Two deliberate differences from ngspice are
kept out of the decks (see the docs' "Differences from ngspice"): `m=` on a
subcircuit instance scales nested instances too, and `TABLE` interpolates
without ngspice's corner rounding.

### Runtime

**FAST mode** (`VOLTAX_FAST=1`); the whole script took 4 to 23 s.

| Case | Points | voltax 1st call | voltax compile | voltax run | ngspice wall | ngspice analysis |
|---|---|---|---|---|---|---|
| RC, per config (10 configs) | 512 | 616 ms | ~615 ms | 1.9 ms | 7.0 ms | 1.0 ms |
| RC, `vmap` over 10 configs (per config) | 501 | 725 ms total | | 0.55 ms | | |
| Power grid 10x10 DC | | 837 ms | 837 ms | 0.53 ms | 8.5 ms | <1 ms |
| Power grid 20x20 DC | | 714 ms | 709 ms | 4.6 ms | 19.6 ms | 6 ms |
| Power grid 20x20 gradient (760 params) | | 988 ms | | 6.3 ms | | |
| Inverter chain, Level 1 exact | 5227 | 3.59 s | 3.27 s | 313 ms | 43 ms | 35 ms |
| Inverter chain, EKV vs BSIM4 | 2026 | 3.32 s | 3.21 s | 115 ms | 38 ms | 30 ms |
| OTA DC / AC / tran | – / 141 / 1059 | 2.3 / 2.6 / 3.0 s | | 0.36 / 1.6 / 36 ms | 6.5 / 9.5 / 12 ms | <1 / 1 / 5 ms |
| Ring oscillator, 5 stages | 22934 | 3.38 s | 1.50 s | 1.88 s | 227 ms | 215 ms |
| Ring oscillator, 11 stages | 22932 | 3.43 s | 1.67 s | 1.76 s | 434 ms | 415 ms |

**Full mode**

| Case | Points | voltax 1st call | voltax compile | voltax run | ngspice wall | ngspice analysis |
|---|---|---|---|---|---|---|
| RC, per config (100 configs) | 512 | 755 ms | ~750 ms | 2.0 ms | 7.0 ms | 1.0 ms |
| RC, `vmap` over 100 configs (per config) | 501 | 843 ms total | | 0.33 ms | | |
| Power grid 10x10 DC | | 868 ms | 868 ms | 0.49 ms | 8.6 ms | <1 ms |
| Power grid 20x20 DC | | 849 ms | 845 ms | 4.3 ms | 19.9 ms | 7 ms |
| Power grid 30x30 DC | | 1.50 s | 1.45 s | 54 ms | 52 ms | 31 ms |
| Power grid 40x40 DC | | 1.13 s | 893 ms | 232 ms | 130 ms | 98 ms |
| Power grid 40x40 gradient (3120 params) | | 1.98 s | | 184 ms | | |
| Inverter chain, Level 1 exact | 15031 | 3.42 s | 2.96 s | 453 ms | 105 ms | 93 ms |
| Inverter chain, EKV vs BSIM4 | 6456 | 3.88 s | 3.52 s | 354 ms | 104 ms | 94 ms |
| OTA DC / AC / tran | – / 141 / 2113 | 2.0 / 2.4 / 2.9 s | | 0.8 / 1.3 / 130 ms | 6.8 / 7.0 / 17 ms | <1 / <1 / 9 ms |
| Ring oscillator, 5 stages | 57346 | 4.86 s | 2.16 s | 2.69 s | 557 ms | 539 ms |
| Ring oscillator, 21 stages | 57337 | 9.72 s | 1.84 s | 7.88 s | 1.81 s | 1.74 s |
| Ring oscillator, 51 stages | 57337 | 24.4 s | 4.46 s | 19.9 s | 4.30 s | 4.17 s |

### Takeaways

- **Accuracy.** With the same model and the same discretization, voltax
  reproduces ngspice to 1e-8 to 1e-7 relative error. That is the gmin floor
  described above. This holds for linear DC, transient, AC, the exact
  Level 1 model with body effect, and an oscillator run over 26 periods. The
  adjoint gradients match ngspice finite differences.
- **Where voltax is fast.** A warm voltax call beats the ngspice process for:
  - small DC and AC solves;
  - short transients (RC);
  - batched runs: `vmap` brings the RC step down to ~0.3 ms per
    configuration;
  - gradients: all 3120 sensitivities cost about one extra solve.
- **Where voltax is slow.** Long nonlinear transients run 3 to 7 times slower
  than ngspice per time point. voltax forms a dense Jacobian (`jacfwd`) and
  does a dense LU at every Newton iteration, where ngspice uses sparse KLU.
  Dense LU scaling also shows on the 40x40 grid. The ~1 to 4 s compile is
  paid once per circuit structure and grid length. (These numbers predate
  the stamped Jacobians and the sparse solver; see
  [Sparse vs dense](#sparse-vs-dense-linear-algebra) for the current paths.
  Dense timings above ~150 unknowns may also be inflated by the OpenBLAS
  threading issue described under Running.)

## Sparse vs dense linear algebra

`benchmark_sparse.py` compares three paths:

- **dense (jacfwd)**: the solver before the sparse work. The Jacobian is
  `jax.jacfwd`, one JVP per unknown, followed by a dense LU.
- **dense**: the Jacobian is stamped from per-device blocks, followed by a
  dense LU.
- **sparse**: the stamped Jacobian with the supernodal sparse LU of
  `voltax.sparse`.

ngspice-42 and JAX 0.10.2 on CPU, one thread, float64, with BLAS pinned. The
machine is a shared 4-core VM. Times are the best of 2 to 5 warm calls;
"compile" is the first call minus the warm time. Full mode.

**[1] Power-grid DC operating point** (N x N mesh, 4 pads, 20 % of nodes loaded)

| grid | unknowns | jacfwd + dense | dense | sparse | sparse compile | dense compile | ngspice analysis | sparse max err vs ngspice |
|---|---|---|---|---|---|---|---|---|
| 10x10 | 104 | 453 us | 353 us | 422 us | 3.9 s | 1.3 s | 1.0 ms | 2.7e-12 V |
| 20x20 | 404 | 5.1 ms | 3.5 ms | 1.2 ms | 7.2 s | 1.6 s | 7.0 ms | 1.5e-11 V |
| 30x30 | 904 | 50 ms | 20 ms | 1.6 ms | 11.5 s | 2.1 s | 64 ms | 3.9e-11 V |
| 50x50 | 2 504 | 1.08 s | 413 ms | 4.6 ms | 13.0 s | 1.9 s | 363 ms | 1.2e-10 V |
| 70x70 | 4 904 | – | 2.66 s | 9.5 ms | 16.3 s | 1.9 s | 1.52 s | 2.4e-10 V |
| 100x100 | 10 004 | – | – | 18.2 ms | 19.7 s | – | 6.87 s | 4.1e-10 V |

At 100x100, a dense Jacobian takes 800 MB, and its LU takes about 0.7 TFLOP
(estimated at more than 20 s per Newton iteration), so dense was not run. The
errors are the usual gmin floor (see Accuracy above).

**[2] Adjoint gradient** of the worst node voltage with respect to every resistor

| grid | parameters | jacfwd + dense | dense | sparse | sparse vs dense (rel) |
|---|---|---|---|---|---|
| 10x10 | 180 | 599 us | 465 us | 939 us | 1.6e-15 |
| 20x20 | 760 | 10.8 ms | 5.8 ms | 1.8 ms | 1.6e-15 |
| 30x30 | 1 740 | 111 ms | 38 ms | 3.2 ms | 3.9e-15 |
| 50x50 | 4 900 | 1.59 s | 905 ms | 7.5 ms | 1.4e-14 |
| 70x70 | 9 660 | – | 4.53 s | 15.9 ms | 4.5e-14 |
| 100x100 | 19 800 | – | – | 45.6 ms | – |

**[3] Ring-oscillator transient** (exact Level 1, backward Euler on ngspice's
3445 accepted time points)

| stages | unknowns | jacfwd + dense | dense | sparse | sparse compile | ngspice | sparse max err vs ngspice |
|---|---|---|---|---|---|---|---|
| 51 | 53 | 421 us/step | 392 us | 346 us | 3.6 s | 69 us | 1.3e-8 V |
| 101 | 103 | 1.4 ms | 555 us | 499 us | 4.5 s | 135 us | 1.3e-8 V |
| 201 | 203 | 5.8 ms | 2.0 ms | 833 us | 5.5 s | 270 us | 1.3e-8 V |
| 501 | 503 | 103 ms | 13.5 ms | 1.9 ms | 2.7 s | 763 us | 1.3e-8 V |

The jacfwd path also took 73 s to compile at 501 stages.

**[4] One Newton iteration taken apart** (backward-Euler Jacobian with
h = 1 ps; "colors" counts greedy distance-2 column colors)

| circuit | unknowns | colors | residual | jacfwd | colored JVPs | stamped | dense LU | sparse factor + solve |
|---|---|---|---|---|---|---|---|---|
| ring 51 | 53 | 53 | 13 us | 122 us | 128 us | 39 us | 31 us | 44 us |
| ring 101 | 103 | 103 | 17 us | 445 us | 287 us | 60 us | 115 us | 86 us |
| ring 201 | 203 | 203 | 22 us | 1.8 ms | 1.4 ms | 93 us | 572 us | 160 us |
| ring 501 | 503 | 503 | 60 us | 13.0 ms | 12.6 ms | 172 us | 3.9 ms | 336 us |
| grid 30x30 | 904 | 7 | 12 us | 12.4 ms | 34 us | 12 us | 16.5 ms | 1.0 ms |
| grid 70x70 | 4 904 | 7 | 58 us | 2.00 s | 210 us | 69 us | 1.40 s | 6.1 ms |
| grid 100x100 | 10 004 | 7 | 104 us | – | 527 us | 146 us | – | 13.2 ms |

Fitted slopes, time ~ unknowns^p, over the larger half of the sizes:

| | jacfwd + dense | dense | sparse | ngspice |
|---|---|---|---|---|
| grid DC | 2.94 | 2.70 | **1.01** | 2.09 |
| grid gradient | 2.73 | 2.73 | **1.09** | |
| ring step | 2.75 | 2.02 | **0.85** | |

![sparse scaling](figures/sparse_scaling.png)

### What the numbers say

- **Crossover.** Steady-state, the sparse solver matches dense at about 100
  unknowns and wins beyond: 3x at 400, 12x at 900, 90x at 2 500 and 280x at
  4 900 unknowns on the grid. Ring transients cross a bit earlier, because
  the dense ring Jacobian has a dense supply row, but the margin there is
  thinner (1.1x at 100 stages, 7x at 500).
- **The default.** `Options(solver="auto")` switches to sparse at 200
  unknowns (`sparse_threshold`), not at the ~100 break-even point, because
  the sparse path compiles 2 to 10 times longer.
- **Scaling.** The sparse grid solve is nearly linear in this range (slope
  1.0). Nested dissection's asymptotic flops grow like S^1.5, but at these
  sizes per-level overhead dominates. Dense LU is ~S^2.7.
- **Against ngspice.** Sparse voltax is 6x to 380x faster than ngspice's
  DC analysis on grids of 20x20 and up. ngspice's own time grows ~S^2.1,
  probably from its sparse ordering on large meshes. On the nonlinear ring,
  voltax is 2.5x (at 501 stages) to 5x (at 51) slower per step than
  ngspice; it was 18x slower with the dense path at 501. What is left is
  per-step overhead: Newton evaluates the full MOSFET model and its
  Jacobian, then factors. ngspice also bypasses converged devices.
- **Gradients.** All 19 800 sensitivities of a 100x100 grid cost 46 ms, one
  extra transposed sparse solve. They match the dense adjoint to 1e-14.
- **Jacobians.** Stamping costs a few JVPs per element group (13 to 170 us
  here), whatever the circuit size. `jax.jacfwd` costs one JVP per unknown.
  Greedy graph coloring needs only 7 colors on a grid, but on the ring it
  needs one color per unknown: the `vdd` row touches every transistor, so
  all those columns conflict. Coloring is therefore not used.
- **Where sparse loses.** The sparse path loses below ~100 unknowns, where
  the per-level kernels and padding cost more than a small dense LU. It also
  loses on compile time everywhere: about 4 to 20 s, growing with the number
  of supernode levels, against 1 to 2 s for dense. That is paid once per
  circuit topology and process; the persistent compilation cache removes it
  across runs.

## Caveats found while validating

- `parse_netlist` maps a `level=1` model card to `Level1MOSFET` with the
  default `smooth=0.02`, and offers no way to ask for `smooth=0`. To get the
  exact square law, leave `level` off the card (ngspice defaults to Level 1)
  and pass `mos_model=functools.partial(vx.Level1MOSFET, smooth=0.0)`.
- `Level1MOSFET` has overlap capacitance only, from `MOSProcess.cov`. It has
  no Meyer gate capacitance. The parser accepts `tox`, but Level 1 ignores it,
  and ignores `cgso`/`cgdo` as well. ngspice adds Meyer capacitances whenever
  `tox` is given. With `tox=4n` on the inverter card, the waveforms differ by
  66 mV.
- `parse_netlist` silently ignores parameters it does not know. A BSIM4
  (`level=54`) card becomes a default EKV device with no warning.
