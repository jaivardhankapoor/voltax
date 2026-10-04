"""CMOS inverter chain from one SPICE netlist: voltax vs ngspice.

The *same netlist text* is given to ngspice and to `voltax.parse_netlist`, so
this benchmark also exercises the netlist API. Three model pairings:

* ``level1-exact``: SPICE Level 1 in both. The model card has no ``level``
  (ngspice's default is Level 1) and voltax is told to build
  ``Level1MOSFET(smooth=0)``, the exact square law. ngspice runs backward
  Euler (``method=gear maxord=1``) with tight tolerances, and voltax runs
  backward Euler on ngspice's accepted time points: same model, same
  discretization, so the waveforms must agree to Newton tolerance.
* ``level1-smooth``: the model card says ``level=1``, which `parse_netlist`
  maps to ``Level1MOSFET`` with its default ``smooth=0.02`` V (softplus
  overdrive, differentiable through cutoff). Same ngspice reference; the
  difference is the smoothing.
* ``ekv-vs-bsim4``: voltax's default EKV model against ngspice BSIM4
  (level 54). Different device models: this is a model comparison, not a
  solver check.

Run:  uv run python benchmarks/benchmark_inverter_ngspice.py   (VOLTAX_FAST=1: short)
"""

from __future__ import annotations

import functools

import _common as cm  # first: configures JAX for single-threaded CPU
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx

VDD = 1.2
WN, WP, L = 0.4e-6, 0.8e-6, 0.13e-6
CLOAD = 10e-15
N_STAGES = 3
PERIOD = 1.6e-9
T_END = cm.pick(2e-9, 4 * PERIOD)
DT = 1e-12  # max time step
CGSO = 0.3e-9  # gate overlap capacitance per width (F/m); voltax's default `cov`

LEVEL1 = f"""
.model nch nmos ({{level}}kp=400u vto=0.4 lambda=0.04 cgso={CGSO} cgdo={CGSO})
.model pch pmos ({{level}}kp=200u vto=-0.4 lambda=0.04 cgso={CGSO} cgdo={CGSO})
"""
BSIM4 = """
.model nch nmos level=54 version=4.8 vth0=0.4 k1=0.5 k2=0.0
+ toxe=4e-9 toxp=3e-9 toxm=4e-9 epsrox=3.9 wint=5e-9 lint=5e-9
.model pch pmos level=54 version=4.8 vth0=-0.4 k1=0.5 k2=0.0
+ toxe=4e-9 toxp=3e-9 toxm=4e-9 epsrox=3.9 wint=5e-9 lint=5e-9
"""
EXACT_OPTIONS = ".options method=gear maxord=1 reltol=1e-6 vntol=1e-9 abstol=1e-15"
NODES = ["in"] + [f"n{k}" for k in range(1, N_STAGES + 1)]


def netlist(models: str) -> str:
    lines = [
        f"{N_STAGES}-stage CMOS inverter chain",
        f"Vdd vdd 0 {VDD}",
        f"Vin in 0 PULSE(0 {VDD} 100p 20p 20p {PERIOD / 2 - 20e-12:.4g} {PERIOD:.4g})",
    ]
    for k in range(1, N_STAGES + 1):
        a, y = NODES[k - 1], NODES[k]
        lines += [f"Mp{k} {y} {a} vdd vdd pch w={WP:.4g} l={L:.4g}",
                  f"Mn{k} {y} {a} 0 0 nch w={WN:.4g} l={L:.4g}",
                  f"C{k} {y} 0 {CLOAD:.4g}"]
    return "\n".join(lines) + "\n" + models


def crossings(t: np.ndarray, v: np.ndarray, level: float = VDD / 2) -> np.ndarray:
    """Times where `v` crosses `level` (linear interpolation)."""
    s = np.sign(v - level)
    k = np.nonzero(s[:-1] * s[1:] < 0)[0]
    return t[k] + (level - v[k]) * (t[k + 1] - t[k]) / (v[k + 1] - v[k])


def compare(case: str, text: str, ng_options: str, method: str,
            **parse_kwargs) -> dict:
    """Run `text` in ngspice, then in voltax on ngspice's time points."""
    tran = f".tran {DT} {T_END} 0 {DT}"
    ng = cm.run_ngspice(f"{text}{ng_options}\n{tran}\n",
                        [f"v({n})" for n in NODES], repeats=cm.pick(3, 5))
    ts = ng["time"]
    circuit = vx.parse_netlist(text, title=True, **parse_kwargs)
    sol, first, steady = cm.time_call(
        lambda: vx.transient(circuit, ts, method=method), repeats=cm.pick(3, 5))
    out = NODES[-1]
    v_vx, v_ng = np.asarray(sol.v(out)), ng[f"v({out})"]
    t_vx, t_ng = crossings(ts, v_vx), crossings(ts, v_ng)
    m = min(len(t_vx), len(t_ng))
    all_err = max(cm.errors(sol.v(n), ng[f"v({n})"])[0] for n in NODES[1:])
    return dict(case=case, ts=ts, v_vx={n: np.asarray(sol.v(n)) for n in NODES},
                v_ng={n: ng[f"v({n})"] for n in NODES}, ng=ng, first=first,
                steady=steady, err=all_err, rel=all_err / VDD,
                edge=np.max(np.abs(t_vx[:m] - t_ng[:m])) if m else np.nan,
                converged=bool(np.all(sol.converged)))


def main() -> None:
    cm.header(f"{N_STAGES}-stage CMOS inverter chain (netlist API)")
    exact = functools.partial(vx.Level1MOSFET, smooth=0.0)
    results = [
        compare("level1-exact", netlist(LEVEL1.format(level="")), EXACT_OPTIONS,
                "be", mos_model=exact),
        compare("level1-smooth", netlist(LEVEL1.format(level="level=1 ")),
                EXACT_OPTIONS, "be"),
        compare("ekv-vs-bsim4", netlist(BSIM4), "", "trap"),
    ]
    cm.print_table([
        cm.timing_row(r["case"], r["first"], r["steady"], r["ng"],
                      points=len(r["ts"]), **{"max abs err (V)": r["err"],
                                              "rel (/VDD)": r["rel"],
                                              "max edge shift": cm.fmt_s(r["edge"]),
                                              "conv": r["converged"]})
        for r in results
    ], "Waveforms n1..n3 on ngspice's time grid (edge shift = 50% crossings of "
       "the output)")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, r in zip(axes, results):
        t = r["ts"] * 1e9
        ax.plot(t, r["v_ng"]["in"], color="0.6", lw=1, label="in")
        for k, n in enumerate(NODES[1:]):
            ax.plot(t, r["v_ng"][n], f"C{k}-", lw=2.5, alpha=0.5,
                    label=f"{n} ngspice")
            ax.plot(t, r["v_vx"][n], f"C{k}--", lw=1.2, label=f"{n} voltax")
        ax.set(xlabel="t (ns)", title=f"{r['case']}: max err {r['err']:.2g} V")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("V")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    cm.save_figure(fig, "inverter_chain")


if __name__ == "__main__":
    main()
