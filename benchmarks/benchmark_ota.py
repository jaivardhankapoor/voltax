"""Five-transistor OTA: DC, AC and transient, voltax vs ngspice (Level 1).

The classic 5T operational transconductance amplifier::

        vdd ---+-----------+
               M3 (diode)  M4          PMOS current-mirror load
          d1 --+           +-- out --- Cload
               M1          M2          NMOS differential pair (body at 0,
        inp ---|           |--- inm    so the body effect is active)
               +--- tail --+
                    M5 --- bias        NMOS tail current source

Both simulators use the exact SPICE Level 1 model (``Level1MOSFET(smooth=0)``
in voltax) with body effect (gamma) and gate-overlap capacitances. Three
analyses are compared:

* DC operating point: every node voltage and the supply current.
* AC: differential gain ``v(out)/v(inp - inm)`` on ngspice's frequency grid.
* Transient: a 50 mV differential pulse. ngspice runs backward Euler
  (``method=gear maxord=1``) with tight tolerances; voltax runs backward Euler
  on ngspice's accepted time points.

Run:  uv run python benchmarks/benchmark_ota.py   (VOLTAX_FAST=1: shorter transient)
"""

from __future__ import annotations

import functools

import _common as cm  # first: configures JAX for single-threaded CPU
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx
from voltax import signals

VDD, VBIAS, VCM = 1.8, 0.6, 0.9
W, L = 2e-6, 0.5e-6
CLOAD = 1e-12
CGSO = 0.3e-9
NMOS = dict(kp=200e-6, vth=0.5, lam=0.05, gamma=0.4, phi=0.7)
PMOS = dict(kp=100e-6, vth=0.5, lam=0.05, gamma=0.4, phi=0.7)
PULSE = (0.0, 50e-3, 50e-9, 1e-9, 1e-9, 40e-9, 200e-9)  # v1 v2 td tr tf pw per
T_END = cm.pick(200e-9, 400e-9)
DT = 0.2e-9
FREQS = "dec 20 1k 10g"
EXACT_OPTIONS = ".options method=gear maxord=1 reltol=1e-6 vntol=1e-9 abstol=1e-15"
NODES = ["inp", "inm", "d1", "out", "tail"]


def build_voltax() -> vx.Circuit:
    b = vx.CircuitBuilder(
        mos_model=functools.partial(vx.Level1MOSFET, smooth=0.0),
        nmos=vx.MOSProcess.nmos(**NMOS, cov=CGSO),
        pmos=vx.MOSProcess.pmos(**PMOS, cov=CGSO),
    )
    b.vsource("vdd", "0", VDD, name="Vdd")
    b.vsource("bias", "0", VBIAS, name="Vbias")
    b.vsource("vcm", "0", VCM, name="Vcm")
    b.vsource("inp", "vcm", signals.Pulse(*PULSE), ac=1.0, name="Vdiff")
    b.vcvs("inm", "vcm", "inp", "vcm", -1.0, name="Einm")
    b.nmos("d1", "inp", "tail", "0", w=W, l=L, name="M1")
    b.nmos("out", "inm", "tail", "0", w=W, l=L, name="M2")
    b.pmos("d1", "d1", "vdd", "vdd", w=2 * W, l=L, name="M3")
    b.pmos("out", "d1", "vdd", "vdd", w=2 * W, l=L, name="M4")
    b.nmos("tail", "bias", "0", "0", w=W, l=L, name="M5")
    b.capacitor("out", "0", CLOAD, name="Cload")
    return b.build()


def ngspice_netlist(analysis: str) -> str:
    card = lambda p: " ".join(  # noqa: E731
        f"{k}={v}" for k, v in dict(kp=p["kp"], lambda_=p["lam"], gamma=p["gamma"],
                                    phi=p["phi"], cgso=CGSO, cgdo=CGSO).items()
    ).replace("lambda_", "lambda")
    pulse = " ".join(f"{x:.6g}" for x in PULSE)
    return f"""5-transistor OTA
Vdd vdd 0 {VDD}
Vbias bias 0 {VBIAS}
Vcm vcm 0 {VCM}
Vdiff inp vcm DC 0 AC 1 PULSE({pulse})
Einm inm vcm inp vcm -1
M1 d1 inp tail 0 nch w={W} l={L}
M2 out inm tail 0 nch w={W} l={L}
M3 d1 d1 vdd vdd pch w={2 * W} l={L}
M4 out d1 vdd vdd pch w={2 * W} l={L}
M5 tail bias 0 0 nch w={W} l={L}
Cload out 0 {CLOAD}
.model nch nmos (level=1 vto={NMOS['vth']} {card(NMOS)})
.model pch pmos (level=1 vto={-PMOS['vth']} {card(PMOS)})
{analysis}
"""


def main() -> None:
    cm.header("5-transistor OTA (Level 1 with body effect)")
    circuit = build_voltax()
    reps = cm.pick(3, 5)
    rows = []

    # ------------------------------------------------------------- DC
    ng = cm.run_ngspice(ngspice_netlist(".op"), [], repeats=reps)
    op, first, steady = cm.time_call(lambda: vx.dc(circuit), reps)
    v_vx = np.array([float(op.v(n)) for n in NODES])
    v_ng = np.array([ng[f"v({n})"][0] for n in NODES])
    i_vx, i_ng = float(op.i("Vdd")), float(ng["i(vdd)"][0])
    err, rel = cm.errors(v_vx, v_ng)
    rows.append(cm.timing_row("dc", first, steady, ng, **{
        "max abs err": f"{err:.2e} V", "rel err": f"{rel:.1e}",
        "note": f"I(Vdd) {i_vx * 1e6:.4f} vs {i_ng * 1e6:.4f} uA "
                f"(rel {abs(i_vx - i_ng) / abs(i_ng):.1e})"}))

    # ------------------------------------------------------------- AC
    ng = cm.run_ngspice(ngspice_netlist(f".ac {FREQS}"), ["v(out)"], repeats=reps)
    freqs = ng["frequency"].real
    acs, first, steady = cm.time_call(lambda: vx.ac(circuit, freqs), reps)
    h_vx, h_ng = np.asarray(acs.v("out")), ng["v(out)"]  # input amplitude 2 V/V
    err, rel = cm.errors(h_vx, h_ng)
    db_err = np.max(np.abs(20 * np.log10(np.abs(h_vx / h_ng))))
    ph_err = np.max(np.abs(np.angle(h_vx / h_ng, deg=True)))
    gain_db = 20 * np.log10(np.abs(h_ng[0]) / 2)
    rows.append(cm.timing_row("ac", first, steady, ng, **{
        "max abs err": f"{err:.2e} V/V", "rel err": f"{rel:.1e}",
        "note": f"{len(freqs)} freqs, gain {gain_db:.2f} dB, "
                f"max {db_err:.1e} dB / {ph_err:.1e} deg"}))

    # ------------------------------------------------------ transient
    tran = f"{EXACT_OPTIONS}\n.tran {DT} {T_END} 0 {DT}"
    ng = cm.run_ngspice(ngspice_netlist(tran), [f"v({n})" for n in NODES],
                        repeats=reps)
    ts = ng["time"]
    sol, first, steady = cm.time_call(
        lambda: vx.transient(circuit, ts, method="be"), reps)
    errs = [cm.errors(sol.v(n), ng[f"v({n})"]) for n in NODES]
    err, rel = max(e[0] for e in errs), max(e[1] for e in errs)
    out_vx, out_ng = np.asarray(sol.v("out")), ng["v(out)"]
    rows.append(cm.timing_row("tran", first, steady, ng, **{
        "max abs err": f"{err:.2e} V", "rel err": f"{rel:.1e}",
        "note": f"{len(ts)} points, out swing {np.ptp(out_ng) * 1e3:.1f} mV, "
                f"converged {bool(np.all(sol.converged))}"}))

    cm.print_table(rows, "voltax vs ngspice (rel err = max abs err / max |ref|)")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]
    ax.semilogx(freqs, 20 * np.log10(np.abs(h_ng) / 2), "C0-", lw=3, alpha=0.5,
                label="ngspice")
    ax.semilogx(freqs, 20 * np.log10(np.abs(h_vx) / 2), "C3--", label="voltax")
    ax.set(xlabel="f (Hz)", ylabel="|v(out) / v(inp-inm)| (dB)",
           title="AC differential gain")
    ax.legend()
    ax = axes[1]
    ax.semilogx(freqs, np.angle(h_ng, deg=True), "C0-", lw=3, alpha=0.5)
    ax.semilogx(freqs, np.angle(h_vx, deg=True), "C3--")
    ax.set(xlabel="f (Hz)", ylabel="phase (deg)", title="AC phase")
    ax = axes[2]
    t = ts * 1e9
    ax.plot(t, out_ng, "C0-", lw=3, alpha=0.5, label="v(out) ngspice")
    ax.plot(t, out_vx, "C3--", label="v(out) voltax")
    ax2 = ax.twinx()
    ax2.plot(t, ng["v(inp)"] - ng["v(inm)"], color="0.6", lw=1)
    ax2.set_ylabel("v(inp) - v(inm) (V)", color="0.5")
    ax.set(xlabel="t (ns)", ylabel="V", title=f"Transient (max err {err:.1e} V)")
    ax.legend(loc="center right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cm.save_figure(fig, "ota")


if __name__ == "__main__":
    main()
