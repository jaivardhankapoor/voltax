"""Cost of a full gradient vs the number of parameters.

The gradient of a scalar loss with respect to every resistor of an N x N
power grid is computed with voltax's adjoint (one backward pass) and compared
with what finite differences cost: one extra simulation per parameter, i.e.
``(P + 1)`` forward solves, for voltax's own forward solve and for an ngspice
run of the same grid.

Usage: ``uv run python benchmarks/benchmark_gradient_cost.py`` (``VOLTAX_FAST=1`` for
smaller grids). Writes ``benchmarks/figures/gradient_cost.png``.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import numpy as np
from _common import FAST, run_ngspice, save_figure

import voltax as vx

SIZES = [4, 6, 8, 12, 16, 20] if FAST else [4, 6, 8, 12, 16, 20, 26, 32]


def grid(n: int) -> vx.Circuit:
    """N x N grid of 0.1 ohm segments, 1 mA loads, supplied at one corner."""
    b = vx.CircuitBuilder()
    b.vsource("n0_0", "0", 1.0, name="Vdd")
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                b.resistor(f"n{i}_{j}", f"n{i + 1}_{j}", 0.1, group="wires")
            if j + 1 < n:
                b.resistor(f"n{i}_{j}", f"n{i}_{j + 1}", 0.1, group="wires")
            b.isource(f"n{i}_{j}", "0", 1e-3)
    return b.build()


def netlist(n: int) -> str:
    lines = ["* grid", "Vdd n0_0 0 1"]
    k = 0
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                lines.append(f"R{k} n{i}_{j} n{i + 1}_{j} 0.1")
                k += 1
            if j + 1 < n:
                lines.append(f"R{k} n{i}_{j} n{i}_{j + 1} 0.1")
                k += 1
            lines.append(f"I{i}_{j} n{i}_{j} 0 1m")
    return "\n".join(lines + [".op"])


def best_of(fn, repeats: int = 5) -> float:
    jax.block_until_ready(fn())  # compile / warm up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t)
    return min(times)


def loss(c: vx.Circuit) -> jax.Array:
    """Worst-case IR drop, smoothed: a typical sizing objective."""
    v = vx.dc(c, options=vx.Options(gmin_steps=0)).v()
    return jax.nn.logsumexp(50 * (1.0 - v)) / 50


rows = []
for n in SIZES:
    c = grid(n)
    p = c.elements["wires"].size
    forward = eqx.filter_jit(loss)
    gradient = eqx.filter_jit(eqx.filter_grad(loss))
    t_fwd = best_of(lambda: forward(c))
    t_grad = best_of(lambda: gradient(c))
    # one ngspice run per finite-difference evaluation: whole process (wall)
    t_ng = run_ngspice(netlist(n), ["v(n0_1)"], repeats=3).wall
    rows.append((n, p, t_fwd, t_grad, t_ng))
    print(f"N={n:3d}  P={p:5d}  forward {t_fwd * 1e3:8.2f} ms  "
          f"adjoint gradient {t_grad * 1e3:8.2f} ms ({t_grad / t_fwd:5.1f}x)  "
          f"FD would be {(p + 1) * t_fwd:8.2f} s (voltax) / "
          f"{(p + 1) * t_ng:8.2f} s (ngspice)", flush=True)

rows = np.array(rows, dtype=float)
P, t_fwd, t_grad, t_ng = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]

import matplotlib.pyplot as plt  # noqa: E402

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.loglog(P, t_grad, "o-", color="C2", lw=2, label="voltax adjoint gradient")
ax1.loglog(P, (P + 1) * t_fwd, "s--", color="C0", label="finite differences, voltax")
ax1.loglog(P, (P + 1) * t_ng, "^--", color="C1",
           label="finite differences, ngspice runs")
ax1.set(xlabel="number of parameters P", ylabel="time for one full gradient (s)",
        title="a  Gradient cost")
ax1.legend(frameon=False, fontsize=8)
ax2.semilogx(P, t_grad / t_fwd, "o-", color="C2", lw=2)
ax2.axhline(1.0, color="gray", lw=0.8, ls=":")
ax2.set(xlabel="number of parameters P", ylabel="gradient time / forward time",
        title="b  Gradient overhead stays constant", ylim=(0, None))
fig.tight_layout()
save_figure(fig, "gradient_cost")
