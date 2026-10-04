# Analyses

| function | solves | returns |
|---|---|---|
| `vx.dc(circuit, t=0)` | \(f(z, t) = 0\) | `Solution` with `z` of shape `(S,)` |
| `vx.transient(circuit, ts)` | the DAE on the grid `ts` | `Solution` with `z` of shape `(T, S)` |
| `vx.ac(circuit, freqs)` | \((G + j\omega C)\tilde z = -b\) | `ACSolution` with complex `z` of shape `(F, S)` |
| `vx.dc_sweep(circuit, values, apply)` | a DC point per value, with continuation | `Solution` with `z` of shape `(N, S)` |

All three are `jit`-compiled internally (compilation is cached per circuit
structure), differentiable, and can be `vmap`ped.

## Solutions

```python
import jax.numpy as jnp
import voltax as vx

b = vx.CircuitBuilder()
b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1e3), ac=1.0, name="Vin")
b.resistor("in", "out", 1e3, name="R1")
b.capacitor("out", "0", 1e-7, name="C1")
c = b.build()

sol = vx.transient(c, jnp.linspace(0, 2e-3, 401))
sol.v("out")        # node voltage, shape (401,)
sol.v()             # all node voltages, shape (401, 2)
sol.i("Vin")        # internal unknown (branch current)
sol.converged       # per time point
sol.t               # the time grid
```

## DC operating point

`vx.dc` runs damped Newton from `guess` (default: zeros). If that does not
converge, which is common for digital CMOS where many nodes start floating, it
falls back to **gmin stepping**: it shunts every node to ground with a
conductance that decreases from `gmin_start` (10 mS) to `gmin` (1 pS), warm
starting each solve from the previous one.

To trace a characteristic, sweep with `vmap` (independent points, in
parallel):

```python
import jax

def v_out(v_in):
    return vx.dc(c.set("Vin", value=v_in)).v("out")

print(jax.vmap(v_out)(jnp.linspace(0, 1, 5)))
```

For circuits with more than one stable state, use `vx.dc_sweep`, the
equivalent of SPICE's `.dc`. It starts each point from the previous solution,
so it follows one branch, and sweeping up and then down reveals hysteresis.
Its `t` field holds the sweep values. Here it traces an inverting Schmitt
trigger, which switches at about +0.5 V going up and about -0.5 V coming down:

```python
s = vx.CircuitBuilder()
s.vsource("in", "0", 0.0, name="Vin")
s.opamp("p", "in", "out", a0=1e3, vmin=-1.0, vmax=1.0)
s.resistor("out", "p", 10e3)
s.resistor("p", "0", 10e3)
schmitt = s.build()

up = vx.dc_sweep(schmitt, jnp.linspace(-1, 1, 81), "Vin")
down = vx.dc_sweep(schmitt, jnp.linspace(1, -1, 81), "Vin", guess=up.z[-1])
print(up.v("out")[40], down.v("out")[40])   # +1 and -1 at the same 0 V input
```

`apply` is either a source name (sets its DC value) or a function
`(circuit, value) -> circuit`, so you can sweep any parameter:
`vx.dc_sweep(c, rs, lambda c, r: c.set("R1", r=r))`.

## Transient

`vx.transient(circuit, ts, ic="dc", method="be")`

- `ts` is any increasing time grid (non-uniform is fine: refine around edges).
- `ic` is the initial state: `"dc"` (the SPICE default, the operating point
  at `ts[0]`), `"zero"`, a state vector (`circuit.state(v={"n1": 1.2})` sets
  chosen node voltages), or a previous `Solution`.
- `method` is `"be"` (backward Euler, robust) or `"trap"` (trapezoidal,
  2nd order).

The step size is fixed by `ts`; there is no adaptive step control. Instead,
`vx.time_grid` builds a grid from the sources' breakpoints (pulse and step
edges, PWL corners). It puts at least `ramp_points` steps across every ramp,
then grows the step geometrically after each edge up to `dt_max`:

```python
p = vx.CircuitBuilder()
p.vsource("in", "0", vx.signals.Pulse(0, 1, delay=1e-9, rise=1e-11, fall=1e-11,
                                      width=4e-9, period=10e-9))
p.resistor("in", "out", 1e3)
p.capacitor("out", "0", 1e-13)
rc = p.build()

grid = vx.time_grid(rc, 6e-9, dt_max=0.5e-9)       # ~120 points, 1 ps at edges
sol = vx.transient(rc, grid, method="trap")
print(len(grid), vx.measure.delay(grid, sol.v("in"), sol.v("out"), 0.5))
```

That 121-point grid gives the 50 % delay (69.5 ps) within 0.25 % of a uniform
60 000-point reference. Use `method="trap"` for timing measurements on graded
grids: backward Euler is first-order and about 3 % off here. Lower `growth`
(default 1.2) or `dt_max` for more accuracy.
The grid needs concrete source parameters, so build it outside `jit`. Always
check `sol.converged.all()`.

## AC

`vx.ac(circuit, freqs, op=None)` linearizes around `op` (computed with
`vx.dc` if omitted). Sources with a nonzero `ac` magnitude drive the
circuit:

```python
res = vx.ac(c, jnp.logspace(1, 6, 6))
res.db("out")       # 20 log10 |v(out)|
res.phase("out")    # degrees
```

`vx.linearize(circuit, z)` returns the small-signal \(G\) and \(C\)
matrices themselves, e.g. for pole/zero or stability analysis with
`jnp.linalg.eig`.

## Solver options

`vx.Options` (passed as `options=`) controls Newton for every analysis:

| option | default | meaning |
|---|---|---|
| `max_steps` | 100 | Newton iterations per solve |
| `rtol`, `atol` | 1e-6, 1e-9 | converged when \(\lvert\Delta z\rvert \le atol + rtol \lvert z\rvert\) |
| `max_dv` | 1.0 V | damping: largest node-voltage change per iteration |
| `gmin` | 1e-12 S | node-to-ground conductance; set 0 for bit-exact comparisons with SPICE (which applies gmin across junctions only) |
| `gmin_steps`, `gmin_start` | 10, 1e-2 S | DC gmin-stepping fallback (0 disables it) |
| `solver` | `"auto"` | linear solver: `"dense"`, `"sparse"`, or `"auto"` (see below) |
| `sparse_threshold` | 200 | state size from which `"auto"` picks the sparse solver |

## Linear solvers

Each Newton iteration solves \(J\,\Delta z = -r\), and each gradient one
transposed system \(J^T \lambda = g\) per solve. Voltax builds \(J\) by
*stamping*, like SPICE: every element group's per-device Jacobian blocks come
from \(T + K\) forward-mode JVPs (terminals plus internal unknowns), which
are scattered into the circuit's fixed sparsity pattern. That costs a few
JVPs per element group, however large the circuit. The system is then solved
by:

| `Options(solver=...)` | method | use it for |
|---|---|---|
| `"dense"` | dense LU of the stamped Jacobian | small circuits (up to a few hundred unknowns) |
| `"sparse"` | supernodal sparse LU (`voltax.sparse`) | large circuits: meshes, long chains, big digital blocks |
| `"auto"` (default) | sparse when `circuit.size >= sparse_threshold` | |

```python
b = vx.CircuitBuilder()
n = 30
for i in range(n):
    for j in range(n):
        if j + 1 < n:
            b.resistor(f"n{i}_{j}", f"n{i}_{j + 1}", 0.1)
        if i + 1 < n:
            b.resistor(f"n{i}_{j}", f"n{i + 1}_{j}", 0.1)
b.vsource("n0_0", "0", 1.0)
b.isource(f"n{n // 2}_{n // 2}", "0", 1.0)
grid = b.build()
print(grid.size)  # 901: 900 node voltages and the source current

op_sparse = vx.dc(grid, options=vx.Options(solver="sparse"))
op_dense = vx.dc(grid, options=vx.Options(solver="dense"))
print(jnp.abs(op_sparse.z - op_dense.z).max())  # rounding-level difference
```

The sparse solver is pure JAX: `jit`, `vmap`, `grad` and GPUs work as with
the dense one, and nothing calls back to the host. Its symbolic analysis
(ordering, supernodes, the factorization schedule) runs once per circuit
topology when the analysis is traced, so you pay it in compile time, which is
longer than for the dense path. The numeric factorization is then
\(O(S^{1.5})\) for a 2-D mesh with \(S\) unknowns instead of \(O(S^3)\),
and memory is \(O(S \log S)\) instead of \(O(S^2)\).
`benchmarks/README.md` has the measured crossover against the dense path and
ngspice.

How it works, in brief (details in the `voltax.sparse` docstring):

- **Ordering**: nested dissection on the circuit graph. Very high-degree
  nodes, such as a supply rail that touches every transistor, are
  eliminated last.
- **Supernodes**: the leaves and separators of the dissection become dense
  blocks, factored level by level as batched dense operations.
- **Pivoting**: partial pivoting *inside* each supernode, static across
  supernodes. Unknowns whose diagonal can vanish, such as a voltage source's
  current, are always placed in a block together with one of their device's
  nodes, so MNA's zero diagonals are handled.

Inspect the plan with `vx.sparse.structure(circuit).plan.stats`. Two more
details:

- `vx.sparse.jacobian(circuit, z, t, f=..., q=...)` returns sparse values of
  \(f\,\partial f/\partial z + q\,\partial q/\partial z\) on the
  circuit's pattern.
- On CPU, dense LU goes through OpenBLAS's own thread pool. On a busy
  machine its spinning threads can make `jnp.linalg.solve` 100x or more
  slower above roughly 150 unknowns. When timing anything, set
  `OPENBLAS_NUM_THREADS=1` before starting Python (the benchmarks do).
- Compilation is cached per circuit topology within a process. For large
  circuits, JAX's persistent cache
  (`jax.config.update("jax_compilation_cache_dir", ...)`) also skips it
  across runs.
- The environment variable `VOLTAX_SOLVER=sparse` changes the default of
  `Options.solver`, for example to run a whole test suite on the sparse path.

## Performance on shared machines

Dense solves go through LAPACK. With OpenBLAS's default multithreading, a
busy or oversubscribed machine (several jobs on a few cores) can make solves
above about 200 unknowns 100–600× slower: we measured a 226×226 solve at
1.5 s against 2.4 ms single-threaded. If simulations are unexpectedly slow,
start Python with `OPENBLAS_NUM_THREADS=1` (or call
`threadpoolctl.threadpool_limits(1)`), and use `jax.vmap` or several
processes for parallelism instead.

## When Newton fails

`converged=False` (or NaNs) usually means one of:

- **A structurally singular circuit**: a node with no DC path to ground, a
  loop of voltage sources and inductors, or a current source in series with a
  capacitor. SPICE rejects these too; add a large resistor or a small series
  resistance.
- **A time step too large** for a fast edge: refine `ts` around it.
- **A hard DC problem**: try more `gmin_steps`, a better `guess`, or start
  a transient from `ic="zero"` with ramped supplies (a "pseudo-transient").
