"""Sparse vs dense linear algebra: where `Options(solver="sparse")` pays off.

Four studies, each against the dense path and (where it applies) ngspice:

1. Power-grid DC operating point, N x N grids up to 100 x 100 (10 004
   unknowns). Dense LU is O(S^3) time and O(S^2) memory and stops being
   practical around 50 x 50.
2. The adjoint gradient of the worst IR drop with respect to every resistor
   (up to 19 800 parameters): one transposed solve, so it scales like (1).
3. A ring oscillator (exact Level 1), 51 to 501 stages, transient on
   ngspice's own time grid: time per step, plus the waveform error against
   ngspice.
4. One Newton iteration taken apart: the Jacobian (``jax.jacfwd``, graph
   coloring, per-device stamping) and the linear solve (dense LU vs sparse
   factor + solve).

"dense (jacfwd)" is the solver voltax used before the sparse work: a
``jax.jacfwd`` Jacobian (one JVP per unknown) and dense LU. "dense" is the
current dense path (stamped Jacobian + dense LU); "sparse" adds the
supernodal LU. Timing: best of several warm calls; compile time is the first
call minus that. Results and plots: benchmarks/README.md,
benchmarks/figures/sparse_scaling.png.

Run:  uv run python benchmarks/benchmark_sparse.py   (VOLTAX_FAST=1: small sizes)
"""

from __future__ import annotations

import contextlib
import time

import _common as cm  # first: configures JAX for single-threaded CPU
import benchmark_power_grid as pg
import benchmark_ring_oscillator as ro
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import voltax as vx
from voltax import analysis, sparse

GRIDS = cm.pick([10, 20, 30], [10, 20, 30, 50, 70, 100])
GRID_DENSE_MAX = cm.pick(30, 70)  # 4904 unknowns: ~80 GFLOP per dense LU
GRID_JACFWD_MAX = cm.pick(30, 50)
RINGS = cm.pick([11, 51, 101], [51, 101, 201, 501])
RING_DENSE_MAX = 501
RING_T_END = cm.pick(0.1e-9, 0.3e-9)
MODES = ("dense (jacfwd)", "dense", "sparse")


@contextlib.contextmanager
def mode(name: str):
    """Options for `name`; "dense (jacfwd)" temporarily restores the old
    generic Jacobian (the one `vx.solve_root` uses for arbitrary residuals)."""
    if name != "dense (jacfwd)":
        yield vx.Options(solver=name)
        return
    saved = analysis._linear
    analysis._linear = lambda circuit, residual, opts: analysis._Linear()
    try:  # a distinct static option, so jit caches never mix the two paths
        yield vx.Options(solver="dense", sparse_threshold=-1)
    finally:
        analysis._linear = saved


def loop_time(fn, *args, n: int = 50) -> float:
    """Seconds per call of ``fn(*args)``, timed as `n` calls inside one jitted
    loop (whose inputs change each iteration, so nothing is hoisted)."""

    def nudge(a, acc):
        return a + acc * 1e-300 if eqx.is_inexact_array(a) else a

    def body(i, acc):
        out = fn(*jax.tree.map(lambda a: nudge(a, acc), args))
        return acc + 1e-300 * sum(jnp.sum(jnp.real(x)) for x in jax.tree.leaves(out))

    run = jax.jit(lambda *a: jax.lax.fori_loop(0, n, body, jnp.zeros(())))
    jax.block_until_ready(run(*args))
    best = min(_timed(run, *args) for _ in range(3))
    return best / n


def _timed(fn, *args) -> float:
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def slope(xs, ys) -> float:
    """Least-squares slope of log y vs log x (over the given points)."""
    xs, ys = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(xs, ys, 1)[0])


# =============================================================================
# 1-2. Power grid: DC and adjoint gradient
# =============================================================================


def grid_case(n: int) -> dict:
    spec = pg.grid_spec(n)
    circuit = pg.build_voltax(spec)
    nodes = list(circuit.layout.node_names)
    ng = cm.run_ngspice(pg.ngspice_netlist(spec), [], repeats=cm.pick(3, 5))
    v_ng = np.array([ng[f"v({name})"] for name in nodes]).ravel()
    worst = nodes[int(np.argmin(v_ng))]
    row = dict(n=n, S=circuit.size, params=len(spec[0]), ngspice=ng.analysis,
               ngspice_wall=ng.wall)
    for name in MODES:
        limit = GRID_JACFWD_MAX if name == "dense (jacfwd)" else GRID_DENSE_MAX
        if name != "sparse" and n > limit:
            continue
        with mode(name) as opts:
            reps = 5 if circuit.size < 1000 else 2
            sol, first, steady = cm.time_call(lambda: vx.dc(circuit, options=opts),
                                              repeats=reps)
            grad_fn = eqx.filter_jit(eqx.filter_grad(
                lambda c: vx.dc(c, options=opts).v(worst)))
            g, gfirst, gsteady = cm.time_call(lambda: grad_fn(circuit), repeats=reps)
        row[name] = dict(first=first, steady=steady, gfirst=gfirst, gsteady=gsteady,
                         err=float(np.max(np.abs(np.asarray(sol.v()) - v_ng))),
                         grad=np.asarray(g.elements["Resistor"].log_r))
        print(f"  grid {n}x{n} {name}: dc {cm.fmt_s(steady)} "
              f"(compile {cm.fmt_s(first - steady)}), grad {cm.fmt_s(gsteady)}",
              flush=True)
    if "dense" in row:
        ref = row["dense"]["grad"]
        row["grad_diff"] = float(np.max(np.abs(row["sparse"]["grad"] - ref))
                                 / np.max(np.abs(ref)))
    plan = sparse.structure(circuit).plan
    row["plan"] = plan.stats
    return row


# =============================================================================
# 3. Ring oscillator transient on ngspice's grid
# =============================================================================


def ring_case(n: int) -> dict:
    ro.T_END = RING_T_END
    nodes = ro.stage_nodes(n)
    ng = cm.run_ngspice(ro.ngspice_netlist(n), [f"v({x})" for x in nodes],
                        repeats=3)
    ic = ro.initial_voltages(n)
    ts = jnp.asarray(np.concatenate([[0.0], ng["time"]]))
    v_ng = np.stack([np.concatenate([[ic[x]], ng[f"v({x})"]]) for x in nodes], 1)
    circuit = ro.build_voltax(n)
    z0 = circuit.state(v=ic)
    steps = len(ts) - 1
    row = dict(n=n, S=circuit.size, steps=steps, ngspice=ng.analysis / steps,
               ngspice_total=ng.analysis)
    for name in MODES:
        if name != "sparse" and n > RING_DENSE_MAX:
            continue
        with mode(name) as opts:
            sol, first, steady = cm.time_call(
                lambda: vx.transient(circuit, ts, ic=z0, method="be", options=opts),
                repeats=2)
        v = np.stack([np.asarray(sol.v(x)) for x in nodes], 1)
        row[name] = dict(first=first, steady=steady, per_step=steady / steps,
                         err=float(np.max(np.abs(v - v_ng))),
                         converged=bool(np.all(sol.converged)))
        print(f"  ring {n} {name}: {cm.fmt_s(steady / steps)}/step "
              f"(compile {cm.fmt_s(first - steady)}), err vs ngspice "
              f"{row[name]['err']:.1e} V", flush=True)
    return row


# =============================================================================
# 4. One Newton iteration taken apart
# =============================================================================


def coloring(st: sparse.Structure) -> np.ndarray:
    """Greedy distance-2 column coloring of the Jacobian pattern (columns that
    share a row get different colors), largest columns first."""
    R = sp.csr_matrix((np.ones(st.nnz), st.indices, st.indptr), shape=(st.size,) * 2)
    C = R.tocsc()
    colors = -np.ones(st.size, int)
    for j in np.argsort(-np.diff(C.indptr), kind="stable"):
        taken = set()
        for r in C.indices[C.indptr[j]:C.indptr[j + 1]]:
            taken.update(colors[R.indices[R.indptr[r]:R.indptr[r + 1]]].tolist())
        c = 0
        while c in taken:
            c += 1
        colors[j] = c
    return colors


def newton_parts(circuit: vx.Circuit, z: jnp.ndarray, dense_ok: bool) -> dict:
    st = sparse.structure(circuit)
    plan = st.plan
    h = 1e-12
    S = circuit.size

    # every timed function takes the circuit as an argument, so the loop in
    # `loop_time` perturbs its parameters too (a linear circuit's Jacobian
    # does not depend on z and would otherwise be hoisted out of the loop)
    def residual(c, z):
        return c.q(z) / h + c.f(z, 0.0)

    colors = coloring(st)
    n_colors = int(colors.max()) + 1
    seeds = jnp.asarray(np.eye(n_colors)[colors].T)  # (colors, S)
    rows, cols = st.rows, st.indices

    def colored(c, z):  # compressed JVPs, then pick each entry from its color
        d = jax.vmap(lambda s: jax.jvp(lambda z_: residual(c, z_), (z,), (s,))[1])(
            seeds)
        return d[colors[cols], rows]

    def stamped(c, z):
        return sparse.jacobian(c, z, f=1.0, q=1.0 / h, structure=st)

    J = stamped(circuit, z).at[st.diagonal[: circuit.n_nodes]].add(1e-12)
    b = residual(circuit, z)
    out = dict(S=S, colors=n_colors, residual=loop_time(residual, circuit, z),
               stamped=loop_time(stamped, circuit, z),
               colored=loop_time(colored, circuit, z),
               sparse_lu=loop_time(lambda J, b: sparse.solve(
                   plan, sparse.factor(plan, J), b), J, b))
    if dense_ok:
        out["jacfwd"] = loop_time(jax.jacfwd(residual, argnums=1), circuit, z, n=10)
        out["dense_lu"] = loop_time(lambda J, b: jnp.linalg.solve(st.to_dense(J), b),
                                    J, b, n=10)
    return out


# =============================================================================
# Report
# =============================================================================


def main() -> None:
    cm.header("Sparse vs dense solvers")
    print("\n[1-2] power grid DC + adjoint gradient")
    grids = [grid_case(n) for n in GRIDS]
    print("\n[3] ring oscillator transient")
    rings = [ring_case(n) for n in RINGS]
    print("\n[4] one Newton iteration")
    parts = []
    for n in RINGS:
        c = ro.build_voltax(n)
        z = c.state(v=ro.initial_voltages(n))
        parts.append(dict(kind=f"ring {n}", **newton_parts(c, z, n <= RING_DENSE_MAX)))
        print(f"  {parts[-1]['kind']} done", flush=True)
    for n in GRIDS:
        c = pg.build_voltax(pg.grid_spec(n))
        parts.append(dict(kind=f"grid {n}x{n}",
                          **newton_parts(c, c.zeros(), n <= GRID_DENSE_MAX)))
        print(f"  {parts[-1]['kind']} done", flush=True)

    def t(row, name, key="steady"):
        return cm.fmt_s(row[name][key]) if name in row else "-"

    cm.print_table([{
        "grid": f"{r['n']}x{r['n']}", "unknowns": r["S"],
        "jacfwd+dense": t(r, "dense (jacfwd)"), "dense": t(r, "dense"),
        "sparse": t(r, "sparse"),
        "sparse compile": cm.fmt_s(r["sparse"]["first"] - r["sparse"]["steady"]),
        "dense compile": (cm.fmt_s(r["dense"]["first"] - r["dense"]["steady"])
                          if "dense" in r else "-"),
        "ngspice analysis": cm.fmt_s(r["ngspice"]) if r["ngspice"] else "<1 ms",
        "ngspice wall": cm.fmt_s(r["ngspice_wall"]),
        "sparse err (V)": r["sparse"]["err"],
    } for r in grids], "[1] power-grid DC operating point (steady-state run time)")
    cm.print_table([{
        "grid": f"{r['n']}x{r['n']}", "parameters": r["params"],
        "jacfwd+dense": t(r, "dense (jacfwd)", "gsteady"),
        "dense": t(r, "dense", "gsteady"), "sparse": t(r, "sparse", "gsteady"),
        "sparse vs dense (rel)": r.get("grad_diff", "-"),
    } for r in grids], "[2] adjoint gradient of the worst node voltage wrt every R")
    cm.print_table([{
        "stages": r["n"], "unknowns": r["S"], "steps": r["steps"],
        "jacfwd+dense": t(r, "dense (jacfwd)", "per_step"),
        "dense": t(r, "dense", "per_step"), "sparse": t(r, "sparse", "per_step"),
        "sparse compile": cm.fmt_s(r["sparse"]["first"] - r["sparse"]["steady"]),
        "ngspice": cm.fmt_s(r["ngspice"]),
        "sparse err vs ngspice (V)": r["sparse"]["err"],
    } for r in rings], "[3] ring oscillator transient: time per step")
    cm.print_table([{
        "circuit": p["kind"], "unknowns": p["S"], "colors": p["colors"],
        "residual": cm.fmt_s(p["residual"]),
        "jacfwd": cm.fmt_s(p["jacfwd"]) if "jacfwd" in p else "-",
        "colored": cm.fmt_s(p["colored"]), "stamped": cm.fmt_s(p["stamped"]),
        "dense LU": cm.fmt_s(p["dense_lu"]) if "dense_lu" in p else "-",
        "sparse LU": cm.fmt_s(p["sparse_lu"]),
    } for p in parts], "[4] one Newton iteration: Jacobian and linear solve")
    cm.print_table([{
        "grid": f"{r['n']}x{r['n']}", **{k: (f"{v:.3g}" if isinstance(v, float) else v)
                                        for k, v in r["plan"].items()}}
        for r in grids], "Sparse LU symbolic statistics (power grid)")

    # asymptotic slopes over the larger half of the sizes
    def fit(rows, key, sub="steady"):
        pts = [(r["S"], r[key][sub]) for r in rows if key in r]
        pts = pts[len(pts) // 2 - (len(pts) > 3):] if len(pts) > 2 else pts
        return slope(*zip(*pts)) if len(pts) >= 2 else float("nan")

    print("\nlog-log slopes (time ~ unknowns^slope, larger sizes):")
    for label, rows, sub in (("grid DC", grids, "steady"),
                             ("grid gradient", grids, "gsteady"),
                             ("ring step", rings, "per_step")):
        print(f"  {label}: " + ", ".join(f"{m} {fit(rows, m, sub):.2f}"
                                         for m in MODES))
    ng = [(r["S"], r["ngspice"]) for r in grids[1:] if r["ngspice"] > 0]
    if len(ng) >= 2:
        print(f"  ngspice grid DC: {slope(*zip(*ng)):.2f}")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    styles = {"dense (jacfwd)": "C3s:", "dense": "C1o--", "sparse": "C0o-"}
    for ax, rows, sub, title in (
            (axes[0], grids, "steady", "power-grid DC solve"),
            (axes[1], grids, "gsteady", "power-grid adjoint gradient"),
            (axes[2], rings, "per_step", "ring-oscillator transient, per step")):
        for m in MODES:
            pts = [(r["S"], r[m][sub]) for r in rows if m in r]
            if pts:
                ax.loglog(*zip(*pts), styles[m], label=f"voltax {m}")
        if sub != "gsteady":
            ng = [(r["S"], r["ngspice"]) for r in rows if r["ngspice"] > 0]
            ax.loglog(*zip(*ng), "k^-", label="ngspice (analysis time)")
        ax.set(xlabel="unknowns", ylabel="time (s)", title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    cm.save_figure(fig, "sparse_scaling")


if __name__ == "__main__":
    main()
