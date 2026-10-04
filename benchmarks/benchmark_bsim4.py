"""BSIM4 compiled from Verilog-A (voltax.va) vs ngspice's built-in BSIM4.

The bundled ``bsim4.va`` (BSIM 4.8.0, Verilog-A port) is compiled to a voltax
element and compared with ngspice's C implementation (``level=54``) on the
FreePDK45 / PTM-45nm cards (bundled; rgatemod=1, rbodymod=1, igcmod=1,
igbmod=1, so gate resistance, the substrate network (4 internal nodes per
device) and gate tunneling are all exercised).

Two voltax builds:

* ``va``: the Verilog-A exactly as published.
* ``va+ng``: the same source with `vx.va.NGSPICE_BSIM4_OVERRIDES`, which pins
  the physical constants to the values hard-coded in ngspice's C (kT/q,
  q, eps0, pi). Both builds use `vx.va.NGSPICE_BSIM4_DEFAULTS` for the few
  parameter defaults where the VA and ngspice disagree (``gidlmod`` ...).
  Circuit tests (VTC, transients) use ``va+ng`` only.

Residual differences, explained in docs/concepts/verilog-a.md: gate and bulk
currents flow through the 0.36-15 ohm gate/substrate resistors, so their
values carry float64 quantization ``ulp(V)/R`` (~1e-6 relative of a pA
current at 0.5 V) in *both* simulators; transients agree to ngspice's Newton
tolerance (reltol=1e-6).

Comparisons (all on ngspice's own sweep points / time grid):

1. Id-Vgs at |Vds| = 50 mV and 1 V (log scale: subthreshold), NMOS and PMOS,
   plus Vbs = -0.3 V; gate and bulk currents too.
2. Id-Vds families.
3. gm, gds (voltax: forward-mode AD Jacobian) and Cgg, Cgd, Cgs (voltax:
   AD of the charges) from AC analysis of many biased copies at once.
4. Inverter VTC.
5. Transient: 3-stage inverter chain (backward Euler on ngspice's time
   points) and a 5-stage ring-oscillator frequency.

Run:  uv run python benchmarks/benchmark_bsim4.py      (VOLTAX_FAST=1: fewer points)
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # LAPACK slows badly under load

import _common as cm  # noqa: E402  (first: configures JAX for single-threaded CPU)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx

VDD = 1.0
W_N, W_P, L = 0.5e-6, 1.0e-6, 50e-9
NG_OPTIONS = ".options reltol=1e-9 abstol=1e-18 vntol=1e-12"
OPTS = vx.Options(gmin=0.0, rtol=1e-10, atol=1e-13)  # gmin: see module notes
FLOOR = 1e-14  # A; pointwise relative errors are reported where |I| > FLOOR
TRAN_OPTIONS = ".options method=gear maxord=1 reltol=1e-6 abstol=1e-15 vntol=1e-9"

# --------------------------------------------------------------------- models


def load_cards() -> dict[str, tuple[str, dict]]:
    out = {}
    for pol in ("n", "p"):
        name, kind, params = vx.va.bundled_card(f"freepdk45_{pol}mos.inc")
        params = {**vx.va.NGSPICE_BSIM4_DEFAULTS, **params}
        params.pop("level", None)
        out[pol] = (name, params)
    return out


CARDS = load_cards()


def model_text() -> str:
    lines = []
    for pol, (name, params) in CARDS.items():
        lines.append(f".model {name} {pol}mos level=54")
        lines += [f"+ {k}={cm.num(v)}" for k, v in params.items()]
    return "\n".join(lines)


BUILDS = {
    "va": vx.va.load("bsim4"),
    "va+ng": vx.va.load("bsim4", overrides=vx.va.NGSPICE_BSIM4_OVERRIDES),
}


def classes(build: str) -> dict[str, type]:
    va = BUILDS[build]
    return {pol: va.element(params, polarity=pol, name=name)
            for pol, (name, params) in CARDS.items()}


def width(pol: str) -> float:
    return W_N if pol == "n" else W_P


# ------------------------------------------------------------------- helpers


def rel_err(test, ref, floor: float = FLOOR) -> float:
    """Max pointwise relative error where |ref| > floor."""
    test, ref = np.asarray(test).ravel(), np.asarray(ref).ravel()
    mask = np.abs(ref) > floor
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(test[mask] - ref[mask]) / np.abs(ref[mask])))


def single_device(cls, pol: str, vd: float, vg: float, vb: float = 0.0):
    b = vx.CircuitBuilder()
    b.vsource("d", "0", vd, name="Vd")
    b.vsource("g", "0", vg, name="Vg")
    b.vsource("b", "0", vb, name="Vb")
    b.add(cls(("d", "g", "0", "b"), w=width(pol), l=L), name="M1")
    return b.build()


def ng_single(pol: str, vd: float, vb: float, sweep: str) -> cm.NgspiceResult:
    name = CARDS[pol][0]
    deck = f"""* bsim4 single device
M1 d g 0 b {name} w={cm.num(width(pol))} l={cm.num(L)}
Vd d 0 {cm.num(vd)}
Vg g 0 0
Vb b 0 {cm.num(vb)}
{model_text()}
{NG_OPTIONS}
{sweep}
"""
    return cm.run_ngspice(deck, ["i(vd)", "i(vg)", "i(vb)", "v(g)", "v(d)"])


# ------------------------------------------------------------- 1. Id-Vgs


def idvgs(results: list, figs: dict) -> None:
    n_pts = cm.pick(25, 61)
    for pol in ("n", "p"):
        s = 1.0 if pol == "n" else -1.0
        for vds, vbs in ((0.05, 0.0), (1.0, 0.0), (1.0, -0.3)):
            vg = np.linspace(0.0, 1.2 * VDD, n_pts) * s
            step = (vg[1] - vg[0])
            ng = ng_single(pol, s * vds, s * vbs,
                           f".dc Vg {cm.num(vg[0])} {cm.num(vg[-1] + step / 2)} "
                           f"{cm.num(step)}")
            for build in BUILDS:
                circ = single_device(classes(build)[pol], pol, s * vds, 0.0, s * vbs)
                sol = vx.dc_sweep(circ, jnp.asarray(ng["v(g)"]), "Vg", options=OPTS)
                row = {"test": f"Id-Vgs {pol}mos Vds={s * vds:+.2f} Vbs={s * vbs:+.1f}",
                       "build": build}
                for src, label in (("Vd", "Id"), ("Vg", "Ig"), ("Vb", "Ib")):
                    row[f"{label} rel"] = rel_err(sol.i(src), ng[f"i({src.lower()})"])
                row["conv"] = bool(np.all(sol.converged))
                results.append(row)
                if build == "va+ng" and vbs == 0.0:
                    figs[(pol, vds)] = (ng["v(g)"], -np.asarray(sol.i("Vd")),
                                        -ng["i(vd)"])


# ------------------------------------------------------------- 2. Id-Vds


def idvds(results: list, figs: dict) -> None:
    for pol in ("n", "p"):
        s = 1.0 if pol == "n" else -1.0
        vgs = np.array(cm.pick([0.6, 1.0], [0.4, 0.6, 0.8, 1.0])) * s
        step = cm.pick(0.1, 0.025)
        name = CARDS[pol][0]
        deck = f"""* bsim4 output characteristics
M1 d g 0 0 {name} w={cm.num(width(pol))} l={cm.num(L)}
Vd d 0 0
Vg g 0 0
{model_text()}
{NG_OPTIONS}
.dc Vd 0 {cm.num(s * (VDD + step / 2))} {cm.num(s * step)} Vg {cm.num(vgs[0])} \
{cm.num(vgs[-1] + s * 0.01)} {cm.num(vgs[1] - vgs[0])}
"""
        ng = cm.run_ngspice(deck, ["i(vd)", "v(d)", "v(g)"])
        vd_all, vg_all, id_ng = ng["v(d)"], ng["v(g)"], ng["i(vd)"]
        for build in BUILDS:
            cls = classes(build)[pol]
            mine = np.zeros_like(id_ng)
            for vg in vgs:
                sel = np.isclose(vg_all, vg)
                circ = single_device(cls, pol, 0.0, float(vg))
                sol = vx.dc_sweep(circ, jnp.asarray(vd_all[sel]), "Vd", options=OPTS)
                mine[sel] = np.asarray(sol.i("Vd"))
            results.append({"test": f"Id-Vds {pol}mos ({len(vgs)} Vgs)",
                            "build": build, "Id rel": rel_err(mine, id_ng)})
            if build == "va+ng":
                figs[pol] = (vd_all, vg_all, -mine, -id_ng, vgs)


# ----------------------------------------------- 3. gm, gds, C-V from AC


def small_signal(results: list, figs: dict) -> None:
    """Many independently biased copies in one AC run; excite one terminal
    kind at a time and read every copy's terminal currents."""
    freq = 1e6
    w = 2 * np.pi * freq
    n_pts = cm.pick(13, 31)
    for pol in ("n", "p"):
        s = 1.0 if pol == "n" else -1.0
        vgs = np.linspace(-0.2, 1.2, n_pts) * s
        vds = s * VDD
        name = CARDS[pol][0]
        ys = {}
        for drive in ("g", "d", "s"):
            lines = ["* bsim4 small-signal"]
            for k, vg in enumerate(vgs):
                ac = lambda t: " AC 1" if t == drive else ""  # noqa: E731
                lines += [f"M{k} d{k} g{k} s{k} 0 {name} w={cm.num(width(pol))} "
                          f"l={cm.num(L)}",
                          f"Vd{k} d{k} 0 DC {cm.num(vds)}{ac('d')}",
                          f"Vg{k} g{k} 0 DC {cm.num(vg)}{ac('g')}",
                          f"Vs{k} s{k} 0 DC 0{ac('s')}"]
            deck = "\n".join(lines) + f"\n{model_text()}\n{NG_OPTIONS}\n" \
                f".ac lin 1 {freq} {freq}\n"
            save = [f"i(v{t}{k})" for k in range(len(vgs)) for t in "dgs"]
            ng = cm.run_ngspice(deck, save)
            for build in BUILDS:
                cls = classes(build)[pol]
                b = vx.CircuitBuilder()
                for k, vg in enumerate(vgs):
                    b.vsource(f"d{k}", "0", vds, ac=float(drive == "d"), name=f"Vd{k}")
                    b.vsource(f"g{k}", "0", float(vg), ac=float(drive == "g"),
                              name=f"Vg{k}")
                    b.vsource(f"s{k}", "0", 0.0, ac=float(drive == "s"),
                              name=f"Vs{k}")
                    b.add(cls((f"d{k}", f"g{k}", f"s{k}", "0"), w=width(pol), l=L),
                          name=f"M{k}")
                circ = b.build()
                op = vx.dc(circ, options=OPTS)
                sol = vx.ac(circ, jnp.array([freq]), op=op)
                for t in "dgs":
                    mine = np.array([complex(sol.i(f"V{t}{k}")[0])
                                     for k in range(len(vgs))])
                    ref = np.array([ng[f"i(v{t}{k})"][0] for k in range(len(vgs))])
                    ys[(build, t, drive)] = (mine, ref)
        for build in BUILDS:
            # y_td = d i(V_t) / d v_drive;  i(V) is the current into the + node
            gm = ys[(build, "d", "g")]
            gds = ys[(build, "d", "d")]
            cgg = ys[(build, "g", "g")]
            cgd = ys[(build, "g", "d")]
            cgs = ys[(build, "g", "s")]
            row = {"test": f"small-signal {pol}mos Vds={vds:+.1f}", "build": build}
            row["gm rel"] = rel_err(gm[0].real, gm[1].real, 1e-9)
            row["gds rel"] = rel_err(gds[0].real, gds[1].real, 1e-9)
            row["Cgg rel"] = rel_err(cgg[0].imag / w, cgg[1].imag / w, 1e-19)
            row["Cgd rel"] = rel_err(cgd[0].imag / w, cgd[1].imag / w, 1e-19)
            row["Cgs rel"] = rel_err(cgs[0].imag / w, cgs[1].imag / w, 1e-19)
            results.append(row)
            if build == "va+ng":
                figs[pol] = (vgs, {k: (v[0], v[1]) for k, v in (
                    ("gm", gm), ("gds", gds), ("Cgg", cgg), ("Cgd", cgd),
                    ("Cgs", cgs))}, w)


# ---------------------------------------------------------- 4/5. circuits


def inverter_lines(stages: int, inp: str, ring: bool = False) -> list[str]:
    lines = []
    nodes = [inp] + [f"n{k}" for k in range(1, stages + 1)]
    if ring:
        nodes[-1] = inp
    for k in range(stages):
        a, y = nodes[k], nodes[k + 1]
        geom = f"l={cm.num(L)}"
        lines += [f"Mp{k} {y} {a} vdd vdd {CARDS['p'][0]} w={cm.num(W_P)} {geom}",
                  f"Mn{k} {y} {a} 0 0 {CARDS['n'][0]} w={cm.num(W_N)} {geom}",
                  f"C{k} {y} 0 1f"]
    return lines


def to_voltax(lines: list[str], build: str, b: vx.CircuitBuilder) -> None:
    cls = classes(build)
    for ln in lines:
        t = ln.split()
        if t[0][0] == "M":
            pol = "p" if t[5] == CARDS["p"][0] else "n"
            b.add(cls[pol](tuple(t[1:5]), w=width(pol), l=L), name=t[0])
        elif t[0][0] == "C":
            b.capacitor(t[1], t[2], 1e-15, name=t[0])


def vtc(results: list, figs: dict) -> None:
    step = cm.pick(0.05, 0.01)
    lines = inverter_lines(1, "in")
    deck = "\n".join(["* inverter VTC", f"Vdd vdd 0 {VDD}", "Vin in 0 0", *lines,
                      model_text(), NG_OPTIONS,
                      f".dc Vin 0 {cm.num(VDD + step / 2)} {cm.num(step)}"])
    ng = cm.run_ngspice(deck, ["v(in)", "v(n1)", "i(vdd)"])
    for build in ("va+ng",):
        b = vx.CircuitBuilder()
        b.vsource("vdd", "0", VDD, name="Vdd")
        b.vsource("in", "0", 0.0, name="Vin")
        to_voltax(lines, build, b)
        sol = vx.dc_sweep(b.build(), jnp.asarray(ng["v(in)"]), "Vin", options=OPTS)
        err = cm.errors(sol.v("n1"), ng["v(n1)"])[0]
        results.append({"test": f"inverter VTC ({len(ng['v(in)'])} pts)",
                        "build": build,
                        "max |dV| (V)": err,
                        "Idd rel": rel_err(sol.i("Vdd"), ng["i(vdd)"], 1e-12)})
        if build == "va+ng":
            figs["vtc"] = (ng["v(in)"], np.asarray(sol.v("n1")), ng["v(n1)"])


def chain_transient(results: list, figs: dict) -> None:
    period = 400e-12
    t_end = cm.pick(400e-12, 800e-12)
    lines = inverter_lines(3, "in")
    nodes = ["n1", "n2", "n3"]
    deck = "\n".join([
        "* 3-stage inverter chain", f"Vdd vdd 0 {VDD}",
        f"Vin in 0 PULSE(0 {VDD} 20p 10p 10p {cm.num(period / 2 - 10e-12)} "
        f"{cm.num(period)})", *lines, model_text(),
        TRAN_OPTIONS,
        f".tran 0.2p {cm.num(t_end)} 0 1p"])
    ng = cm.run_ngspice(deck, [f"v({n})" for n in nodes])
    ts = ng["time"]
    for build in ("va+ng",):
        b = vx.CircuitBuilder()
        b.vsource("vdd", "0", VDD, name="Vdd")
        b.vsource("in", "0", vx.signals.Pulse(0.0, VDD, 20e-12, 10e-12, 10e-12,
                                              period / 2 - 10e-12, period),
                  name="Vin")
        to_voltax(lines, build, b)
        circ = b.build()
        sol, first, steady = cm.time_call(
            lambda c=circ: vx.transient(c, jnp.asarray(ts), options=OPTS), repeats=2)
        err = max(cm.errors(sol.v(n), ng[f"v({n})"])[0] for n in nodes)
        t_vx = vx.measure.crossings(jnp.asarray(ts), sol.v("n3"), VDD / 2)
        t_ng = vx.measure.crossings(jnp.asarray(ts), jnp.asarray(ng["v(n3)"]), VDD / 2)
        t_vx, t_ng = np.asarray(t_vx), np.asarray(t_ng)
        ok = np.isfinite(t_vx) & np.isfinite(t_ng)
        shift = float(np.max(np.abs(t_vx[ok] - t_ng[ok]))) if ok.any() else np.nan
        results.append({"test": f"3-inv chain tran ({len(ts)} pts)", "build": build,
                        "max |dV| (V)": err, "edge shift": cm.fmt_s(shift),
                        "voltax 1st": cm.fmt_s(first), "voltax run": cm.fmt_s(steady),
                        "ngspice": cm.fmt_s(ng.wall),
                        "conv": bool(np.all(sol.converged))})
        if build == "va+ng":
            figs["chain"] = (ts, {n: (np.asarray(sol.v(n)), ng[f"v({n})"])
                                  for n in nodes})


def ring_oscillator(results: list, figs: dict) -> None:
    stages = 5
    t_end = cm.pick(150e-12, 400e-12)
    lines = inverter_lines(stages, "n0", ring=True)
    lines = [ln.replace(" n5 ", " n0 ") for ln in lines]
    ics = {"n0": 0.0, "n1": VDD, "n2": 0.0, "n3": VDD, "n4": 0.0}
    deck = "\n".join([
        "* 5-stage ring oscillator", f"Vdd vdd 0 {VDD}", *lines, model_text(),
        ".ic " + " ".join(f"v({k})={v}" for k, v in ics.items()),
        TRAN_OPTIONS,
        f".tran 0.2p {cm.num(t_end)} 0 0.5p uic"])
    ng = cm.run_ngspice(deck, ["v(n0)"])
    ts = ng["time"]
    f_ng = float(vx.measure.frequency(jnp.asarray(ts), jnp.asarray(ng["v(n0)"]),
                                      level=VDD / 2))
    for build in ("va+ng",):
        b = vx.CircuitBuilder()
        b.vsource("vdd", "0", VDD, name="Vdd")
        to_voltax(lines, build, b)
        circ = b.build()
        z0 = circ.state(v={"vdd": VDD, **ics})
        sol = vx.transient(circ, jnp.asarray(ts), ic=z0, options=OPTS)
        f_vx = float(vx.measure.frequency(sol.t, sol.v("n0"), level=VDD / 2))
        err = cm.errors(sol.v("n0"), ng["v(n0)"])[0]
        results.append({"test": f"5-stage RO tran ({len(ts)} pts)", "build": build,
                        "max |dV| (V)": err, "f voltax (GHz)": f_vx / 1e9,
                        "f ngspice (GHz)": f_ng / 1e9,
                        "f rel": abs(f_vx - f_ng) / f_ng,
                        "conv": bool(np.all(sol.converged))})
        if build == "va+ng":
            figs["ro"] = (ts, np.asarray(sol.v("n0")), ng["v(n0)"])


# ------------------------------------------------------------------- plots


def plot(figs: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    for (pol, vds), (vg, mine, ref) in sorted(figs["idvgs"].items()):
        ax.semilogy(np.abs(vg), np.abs(ref), "-", lw=3, alpha=0.4,
                    label=f"{pol}mos |Vds|={vds} ngspice")
        ax.semilogy(np.abs(vg), np.abs(mine), "k--", lw=1)
    ax.set(xlabel="|Vgs| (V)", ylabel="|Id| (A)", title="Id-Vgs (dashed: voltax)")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    vd_all, vg_all, mine, ref, vgs = figs["idvds"]["n"]
    for vg in vgs:
        sel = np.isclose(vg_all, vg)
        ax.plot(vd_all[sel], ref[sel] * 1e6, "-", lw=3, alpha=0.4)
        ax.plot(vd_all[sel], mine[sel] * 1e6, "k--", lw=1)
    ax.set(xlabel="Vds (V)", ylabel="Id (uA)", title="NMOS Id-Vds")
    ax = axes[0, 2]
    vgs, d, w = figs["ss"]["n"]
    # i(V) is the current into the source's + node: gm = -Re y_dg, etc.
    ax.plot(vgs, -d["gm"][1].real * 1e3, "-", lw=3, alpha=0.4, label="gm ngspice")
    ax.plot(vgs, -d["gm"][0].real * 1e3, "k--", lw=1, label="gm voltax (AD)")
    ax.plot(vgs, -d["gds"][1].real * 1e3, "-", lw=3, alpha=0.4, label="gds ngspice")
    ax.plot(vgs, -d["gds"][0].real * 1e3, "k:", lw=1, label="gds voltax (AD)")
    ax.set(xlabel="Vgs (V)", ylabel="mS", title="NMOS gm, gds at Vds=1 V")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for k, sgn in (("Cgg", -1), ("Cgd", 1), ("Cgs", 1)):  # Cgg = dQg/dVg
        ax.plot(vgs, sgn * d[k][1].imag / w * 1e15, "-", lw=3, alpha=0.4,
                label=f"{k} ngspice")
        ax.plot(vgs, sgn * d[k][0].imag / w * 1e15, "k--", lw=1)
    ax.set(xlabel="Vgs (V)", ylabel="fF", title="NMOS C-V at Vds=1 V")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    vin, mine, ref = figs["vtc"]
    ax.plot(vin, ref, "-", lw=3, alpha=0.4, label="ngspice")
    ax.plot(vin, mine, "k--", lw=1, label="voltax")
    ts, chain = figs["chain"]
    ax2 = axes[1, 2]
    for k, (n, (mine, ref)) in enumerate(chain.items()):
        ax2.plot(ts * 1e12, ref, f"C{k}-", lw=3, alpha=0.4, label=f"{n} ngspice")
        ax2.plot(ts * 1e12, mine, "k--", lw=1)
    ax.set(xlabel="Vin (V)", ylabel="Vout (V)", title="Inverter VTC")
    ax.legend(fontsize=7)
    ax2.set(xlabel="t (ps)", ylabel="V", title="3-inverter chain (dashed: voltax)")
    ax2.legend(fontsize=7)
    for a in axes.ravel():
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    cm.save_figure(fig, "bsim4_validation")


def main() -> None:
    cm.header("BSIM4 (Verilog-A via voltax.va) vs ngspice BSIM4 level=54")
    results: list[dict] = []
    figs: dict = {"idvgs": {}, "idvds": {}, "ss": {}}
    idvgs(results, figs["idvgs"])
    idvds(results, figs["idvds"])
    small_signal(results, figs["ss"])
    vtc(results, figs)
    chain_transient(results, figs)
    ring_oscillator(results, figs)
    keys: list[str] = []
    for r in results:
        keys += [k for k in r if k not in keys]
    groups = [("DC currents (max pointwise relative error, |I| > 1e-14 A)",
               ["test", "build", "Id rel", "Ig rel", "Ib rel", "conv"]),
              ("Small signal (AC at 1 MHz; voltax derivatives by AD)",
               ["test", "build", "gm rel", "gds rel", "Cgg rel", "Cgd rel",
                "Cgs rel"]),
              ("Circuits", ["test", "build", "max |dV| (V)", "Idd rel", "edge shift",
                            "f voltax (GHz)", "f ngspice (GHz)", "f rel",
                            "voltax 1st", "voltax run", "ngspice", "conv"])]
    for title, cols in groups:
        rows = [{c: r.get(c, "") for c in cols} for r in results
                if any(c in r for c in cols[2:3])]
        cm.print_table(rows, title)
    plot(figs)


if __name__ == "__main__":
    main()
