"""Power-grid DC operating point: voltax vs ngspice.

An N x N resistive mesh (on-chip power grid) fed by four corner pads at VDD
and loaded by current sinks at a random 20 % of its nodes. The circuit is
linear, so the two simulators must agree to solver precision. For each grid
size we compare every node voltage, the DC runtimes, and finally the adjoint
gradient of the worst-case IR drop with respect to a resistor against a
central finite difference computed with ngspice.

Run:  uv run python benchmarks/benchmark_power_grid.py   (VOLTAX_FAST=1: small grids)
"""

from __future__ import annotations

import _common as cm  # first: configures JAX for single-threaded CPU
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx

SIZES = cm.pick([10, 20], [10, 20, 30, 40])
VDD = 1.0
R_SEGMENT = 0.1  # ohm per grid segment
LOAD_FRACTION = 0.2
LOAD_RANGE = (1e-3, 5e-3)  # A
SEED = 42


def grid_spec(n: int) -> tuple[list, list, list]:
    """``(resistors, loads, pads)`` as plain lists, shared by both simulators.

    resistors: ``(name, node_a, node_b, ohms)``; loads: ``(name, node, amps)``;
    pads: ``(name, node)``.
    """
    rng = np.random.default_rng(SEED)
    node = lambda i, j: f"n{i}_{j}"  # noqa: E731
    resistors = []
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                resistors.append((f"Rh{i}_{j}", node(i, j), node(i, j + 1),
                                  R_SEGMENT))
            if i + 1 < n:
                resistors.append((f"Rv{i}_{j}", node(i, j), node(i + 1, j),
                                  R_SEGMENT))
    n_loads = int(LOAD_FRACTION * n * n)
    picks = rng.choice(n * n, n_loads, replace=False)
    amps = rng.uniform(*LOAD_RANGE, n_loads)
    loads = [(f"I{k}", node(p // n, p % n), a)
             for k, (p, a) in enumerate(zip(picks, amps))]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    pads = [(f"Vpad{k}", node(i, j)) for k, (i, j) in enumerate(corners)]
    return resistors, loads, pads


def build_voltax(spec) -> vx.Circuit:
    resistors, loads, pads = spec
    b = vx.CircuitBuilder()
    for name, a, c, r in resistors:
        b.resistor(a, c, r, name=name)
    for name, a, amps in loads:
        b.isource(a, "0", amps, name=name)  # sink: amps flow a -> ground
    for name, a in pads:
        b.vsource(a, "0", VDD, name=name)
    return b.build()


def ngspice_netlist(spec, scale: dict[str, float] | None = None) -> str:
    """Netlist of `spec`; `scale` multiplies selected resistors (for FD)."""
    resistors, loads, pads = spec
    scale = scale or {}
    lines = ["power grid"]
    lines += [f"{n} {a} {c} {cm.num(r * scale.get(n, 1.0))}"
              for n, a, c, r in resistors]
    lines += [f"{n} {a} 0 DC {cm.num(i)}" for n, a, i in loads]
    lines += [f"{n} {a} 0 DC {VDD}" for n, a in pads]
    lines.append(".op")
    return "\n".join(lines) + "\n"


def run_size(n: int) -> dict:
    spec = grid_spec(n)
    circuit = build_voltax(spec)
    nodes = list(circuit.layout.node_names)
    sol, first, steady = cm.time_call(lambda: vx.dc(circuit), repeats=cm.pick(5, 10))
    ng = cm.run_ngspice(ngspice_netlist(spec), [], repeats=cm.pick(5, 10))
    v_vx = np.asarray(sol.v())
    v_ng = np.array([ng[f"v({name})"] for name in nodes]).ravel()
    abs_err = np.abs(v_vx - v_ng)
    return dict(n=n, circuit=circuit, spec=spec, nodes=nodes, v_vx=v_vx, v_ng=v_ng,
                abs_err=abs_err.max(), rel_err=(abs_err / np.abs(v_ng)).max(),
                drop=VDD - v_ng.min(), first=first, steady=steady, ng=ng,
                converged=bool(sol.converged))


def gradient_check(res: dict) -> dict:
    """d v(worst node) / d log(R_k) by the adjoint vs ngspice central FD."""
    circuit, spec, nodes = res["circuit"], res["spec"], res["nodes"]
    worst = nodes[int(np.argmin(res["v_ng"]))]
    dc_worst = lambda c: vx.dc(c).v(worst)  # noqa: E731
    grad_fn = eqx.filter_jit(eqx.filter_grad(dc_worst))
    grads, first, steady = cm.time_call(lambda: grad_fn(circuit), 3)
    # the most influential resistor, by voltax's gradient
    name, (group, idx) = max(
        ((name, circuit.layout.device(name)) for name, *_ in spec[0]),
        key=lambda item: abs(float(grads.elements[item[1][0]].log_r[item[1][1]])))
    g_vx = float(grads.elements[group].log_r[idx])
    eps = 1e-4
    up = cm.run_ngspice(ngspice_netlist(spec, {name: np.exp(eps)}), [f"v({worst})"])
    dn = cm.run_ngspice(ngspice_netlist(spec, {name: np.exp(-eps)}), [f"v({worst})"])
    g_fd = float((up[f"v({worst})"][0] - dn[f"v({worst})"][0]) / (2 * eps))
    return dict(node=worst, resistor=name, g_vx=g_vx, g_fd=g_fd, first=first,
                steady=steady, n_params=len(spec[0]))


def main() -> None:
    cm.header("Power grid DC operating point (linear)")
    results = [run_size(n) for n in SIZES]
    cm.print_table([
        cm.timing_row(f"{r['n']}x{r['n']}", r["first"], r["steady"], r["ng"],
                      nodes=r["n"] ** 2, **{"IR drop (mV)": f"{r['drop'] * 1e3:.1f}",
                                            "max abs err (V)": r["abs_err"],
                                            "max rel err": r["rel_err"],
                                            "conv": r["converged"]})
        for r in results
    ], "Node voltages: voltax vs ngspice")

    g = gradient_check(results[-1])
    rel = abs(g["g_vx"] - g["g_fd"]) / abs(g["g_fd"])
    cm.print_table([{
        "grid": f"{results[-1]['n']}x{results[-1]['n']}",
        "d v(node)/d log R": f"v({g['node']}) / {g['resistor']}",
        "voltax adjoint": f"{g['g_vx']:.9e}",
        "ngspice central FD": f"{g['g_fd']:.9e}",
        "rel diff": rel,
        "grad wrt all R (1st call)": cm.fmt_s(g["first"]),
        "grad (steady)": cm.fmt_s(g["steady"]),
    }], f"Adjoint gradient check ({g['n_params']} resistor parameters)")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    big = results[-1]
    n = big["n"]
    im = axes[0].imshow(big["v_ng"].reshape(n, n) * 1e3, cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="mV")
    axes[0].set_title(f"{n}x{n} grid: node voltage (ngspice)")
    im = axes[1].imshow(np.abs(big["v_vx"] - big["v_ng"]).reshape(n, n), cmap="magma")
    fig.colorbar(im, ax=axes[1], label="V")
    axes[1].set_title("|voltax - ngspice|")
    ax = axes[2]
    nodes = [r["n"] ** 2 for r in results]
    ax.loglog(nodes, [r["steady"] * 1e3 for r in results], "o-",
              label="voltax dc (steady)")
    ax.loglog(nodes, [r["first"] * 1e3 for r in results], "o:", label="voltax 1st call")
    ax.loglog(nodes, [r["ng"].wall * 1e3 for r in results], "s-", label="ngspice wall")
    ax.loglog(nodes, [r["ng"].analysis * 1e3 for r in results], "s:",
              label="ngspice analysis")
    ax.set(xlabel="nodes", ylabel="time (ms)", title="DC solve runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    cm.save_figure(fig, "power_grid")


if __name__ == "__main__":
    main()
