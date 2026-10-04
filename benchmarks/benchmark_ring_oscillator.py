"""CMOS ring oscillator: voltax vs ngspice (exact Level 1).

A ring of N inverters (built with `voltax.library.cmos.ring_oscillator`) with
a load capacitor per stage. A ring has no stable DC operating point to start
from, so both simulators start from the same asymmetric initial condition
(``.ic`` + ``UIC`` in ngspice, an explicit state vector in voltax).

Oscillators are a hard accuracy test: any per-step error accumulates into a
phase drift. ngspice runs backward Euler (``method=gear maxord=1``) with tight
tolerances and voltax runs backward Euler on ngspice's accepted time points,
so with the exact Level-1 model (``Level1MOSFET(smooth=0)``) the waveforms
should agree to Newton tolerance over many periods.

Run:  uv run python benchmarks/benchmark_ring_oscillator.py   (VOLTAX_FAST=1: small)
"""

from __future__ import annotations

import functools

import _common as cm  # first: configures JAX for single-threaded CPU
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx
from voltax.library import cmos

VDD = 1.2
WN, WP, L = 0.4e-6, 0.8e-6, 0.13e-6
CLOAD = 5e-15
CGSO = 0.3e-9
NMOS = dict(kp=400e-6, vth=0.4, lam=0.04)
PMOS = dict(kp=200e-6, vth=0.4, lam=0.04)
STAGES = cm.pick([5, 11], [5, 21, 51])
T_END = cm.pick(2e-9, 5e-9)
DT = 1e-12
EXACT_OPTIONS = ".options method=gear maxord=1 reltol=1e-6 vntol=1e-9 abstol=1e-15"


def stage_nodes(n: int) -> list[str]:
    return [f"n{k}" for k in range(1, n + 1)]


def initial_voltages(n: int) -> dict[str, float]:
    """Alternating high/low; with an odd ring two neighbours clash and the
    ring starts oscillating immediately."""
    ic = {node: (0.9 if k % 2 == 0 else 0.1) * VDD
          for k, node in enumerate(stage_nodes(n))}
    return {"vdd": VDD, **ic}


def build_voltax(n: int) -> vx.Circuit:
    b = vx.CircuitBuilder(
        mos_model=functools.partial(vx.Level1MOSFET, smooth=0.0),
        nmos=vx.MOSProcess.nmos(**NMOS, cov=CGSO),
        pmos=vx.MOSProcess.pmos(**PMOS, cov=CGSO),
    )
    b.vsource("vdd", "0", VDD, name="Vdd")
    nodes = stage_nodes(n)
    cmos.ring_oscillator(b, nodes, "vdd", wn=WN, wp=WP)
    for node in nodes:
        b.capacitor(node, "0", CLOAD)
    return b.build()


def ngspice_netlist(n: int) -> str:
    nodes = stage_nodes(n)
    lines = [f"{n}-stage ring oscillator", f"Vdd vdd 0 {VDD}"]
    for k, (a, y) in enumerate(zip(nodes, nodes[1:] + nodes[:1])):
        lines += [f"Mp{k} {y} {a} vdd vdd pch w={WP:.4g} l={L:.4g}",
                  f"Mn{k} {y} {a} 0 0 nch w={WN:.4g} l={L:.4g}",
                  f"C{k} {y} 0 {CLOAD:.4g}"]
    for name, p in (("nch nmos", NMOS), ("pch pmos", PMOS)):
        vto = p["vth"] if "nmos" in name else -p["vth"]
        lines.append(f".model {name} (level=1 kp={p['kp']} vto={vto} "
                     f"lambda={p['lam']} cgso={CGSO} cgdo={CGSO})")
    ic = " ".join(f"v({k})={v!r}" for k, v in initial_voltages(n).items())
    lines += [f".ic {ic}", EXACT_OPTIONS, f".tran {DT} {T_END} 0 {DT} UIC"]
    return "\n".join(lines) + "\n"


def period(t: np.ndarray, v: np.ndarray) -> float:
    """Mean period from rising VDD/2 crossings in the second half of the run."""
    s = v > VDD / 2
    k = np.nonzero(~s[:-1] & s[1:])[0]
    tc = t[k] + (VDD / 2 - v[k]) * (t[k + 1] - t[k]) / (v[k + 1] - v[k])
    tc = tc[tc > t[-1] / 2]
    return float(np.mean(np.diff(tc))) if len(tc) > 1 else float("nan")


def run(n: int) -> dict:
    nodes = stage_nodes(n)
    ng = cm.run_ngspice(ngspice_netlist(n), [f"v({x})" for x in nodes],
                        repeats=cm.pick(3, 5))
    ic = initial_voltages(n)
    # with UIC ngspice does not output t = 0; prepend the initial condition
    ts = np.concatenate([[0.0], ng["time"]])
    v_ng = np.stack([np.concatenate([[ic[x]], ng[f"v({x})"]]) for x in nodes], 1)
    circuit = build_voltax(n)
    z0 = circuit.state(v=ic)
    sol, first, steady = cm.time_call(
        lambda: vx.transient(circuit, ts, ic=z0, method="be"), repeats=cm.pick(3, 5))
    v_vx = np.stack([np.asarray(sol.v(x)) for x in nodes], 1)
    err, rel = cm.errors(v_vx, v_ng)
    p_vx, p_ng = period(ts, v_vx[:, 0]), period(ts, v_ng[:, 0])
    return dict(n=n, ts=ts, v_vx=v_vx, v_ng=v_ng, err=err, rel=err / VDD,
                p_vx=p_vx, p_ng=p_ng, first=first, steady=steady, ng=ng,
                converged=bool(np.all(sol.converged)))


def main() -> None:
    cm.header("CMOS ring oscillator (exact Level 1)")
    results = [run(n) for n in STAGES]
    cm.print_table([
        cm.timing_row(f"{r['n']} stages", r["first"], r["steady"], r["ng"],
                      points=len(r["ts"]),
                      **{"max abs err (V)": r["err"], "rel (/VDD)": r["rel"],
                         "period ngspice": cm.fmt_s(r["p_ng"]),
                         "period rel diff": abs(r["p_vx"] - r["p_ng"]) / r["p_ng"],
                         "conv": r["converged"]})
        for r in results
    ], f"All stage voltages over {cm.fmt_s(T_END)} on ngspice's time grid")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2),
                             squeeze=False)
    for ax, r in zip(axes[0], results):
        t = r["ts"] * 1e9
        for k in range(min(3, r["n"])):
            ax.plot(t, r["v_ng"][:, k], f"C{k}-", lw=3, alpha=0.45,
                    label=f"n{k + 1} ngspice")
            ax.plot(t, r["v_vx"][:, k], f"C{k}--", lw=1.2, label=f"n{k + 1} voltax")
        ax.set(xlabel="t (ns)", ylabel="V",
               title=f"{r['n']} stages: max err {r['err']:.1e} V")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    cm.save_figure(fig, "ring_oscillator")


if __name__ == "__main__":
    main()
