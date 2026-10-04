"""General-purpose netlists: the same deck text in voltax and ngspice.

Each case is a realistic deck exercising one group of parser features. The
*identical text* goes to ``ngspice -b`` and to `voltax.parse_netlist`, and
the transients are compared on a matched discretization (ngspice backward
Euler, ``method=gear maxord=1``, max step ``dt``; voltax ``method="be"`` on
ngspice's accepted time points), so differences come only from the device
equations and Newton tolerances.

Cases:

* ``hierarchy``: parameterized subcircuits three levels deep (defaults
  referencing other parameters, ``.param`` in a body, instance overrides
  evaluated in the caller's scope, ``m=`` multipliers, ``.global``,
  ``.func``, ternaries), an RC ladder driven by ``PULSE``. (``m=`` sits on
  a leaf instance: ngspice-42 applies an instance's ``m`` only to the
  devices directly inside it, voltax to the whole hierarchy below it.)
* ``lib-tt`` / ``lib-ff`` / ``lib-ss``: a Level-1 CMOS inverter chain whose
  model cards come from ``.lib models.lib <corner>`` (corner sections
  ``.lib``-include a shared section and ``.include`` a parameter file).
* ``bsource``: nonlinear behavioral circuit: ``B`` sources with
  ``v(a)``, ``v(a,b)``, ``i(Vsense)``, ``time``, user functions; ``E
  vol=``, ``G cur=``, ``TABLE`` and ``POLY(2)`` sources.
* ``sources``: ``EXP``, ``SFFM``, ``AM``, ``PWL r= td=``, ``PULSE``, ``SIN``
  into RC loads.
* ``binned``: Level-1 inverters whose ``nch``/``pch`` models are binned by
  length. ngspice does not bin Level-1 models, so it gets the same deck with
  each device pointing at the bin voltax selected; the bin *selection* is
  checked separately against ngspice BSIM3 (level 8) binning, where each bin
  has a very different ``vth0`` so the drain current reveals the chosen bin.

Run:  uv run python benchmarks/benchmark_netlists.py   (VOLTAX_FAST=1: shorter runs)
"""

from __future__ import annotations

import functools
import re
import tempfile
import warnings
from pathlib import Path

import _common as cm  # first: configures JAX for single-threaded CPU
import numpy as np

import voltax as vx

EXACT = ".options method=gear maxord=1 reltol=1e-6 vntol=1e-9 abstol=1e-15"
LEVEL1 = functools.partial(vx.Level1MOSFET, smooth=0.0)

HIERARCHY = """parameterized hierarchy
.global vref
.param rbase=1k cbase=1p ratio={2*1.5}
.func par(a, b) = a * b / (a + b)
.subckt rcstage in out params: r=1k c=1p
Rs in out {r}
Cs out 0 {c}
.ends rcstage
.subckt section in out r=1k c=1p taps=2
.param rmid = {taps > 1 ? r / 2 : r}
X1 in mid rcstage r={rmid} c={c}
X2 mid out rcstage r={rmid} c={c * 2} m=2
Rload out vref {par(10 * r, 20 * r)}
.ends section
.subckt ladder in out scale=1
Xa in a section r={rbase * scale} c={cbase}
Xb a b section r={rbase * scale * ratio} c={cbase / 2} taps=1
Xc b out section r={rbase} c={cbase}
.ends ladder
Vref vref 0 0.2
Vin in 0 PULSE(0 1 {0.2n} 0.1n 0.1n 5n 10n)
Xl in out ladder scale=0.5
Cout out 0 0.5p
"""

MODELS_LIB = """* corner library
.lib tt
.param dvt=0 kfac=1
.lib 'models.lib' mos
.endl tt
.lib ff
.param dvt=-0.05 kfac=1.2
.lib 'models.lib' mos
.endl ff
.lib ss
.param dvt=0.05 kfac=0.8
.lib 'models.lib' mos
.endl ss
.lib mos
.include 'process.inc'
.model nch nmos (level=1 kp={kp_n*kfac} vto={vt_n+dvt} lambda=0.05
+ cgso=0.2n cgdo=0.2n gamma=0.3 phi=0.7)
.model pch pmos (level=1 kp={kp_p*kfac} vto={-vt_p-dvt} lambda=0.08
+ cgso=0.2n cgdo=0.2n gamma=0.3 phi=0.7)
.endl mos
"""

PROCESS_INC = """* process parameters
.param kp_n=300u kp_p=120u
.param vt_n=0.45 vt_p=0.42
"""

LIB_DECK = """inverter chain, {corner} corner
.lib '{lib}' {corner}
.param wn=1u wp={{2.5*wn}} lch=0.2u
.subckt inv in out vdd
Mp out in vdd vdd pch w={{wp}} l={{lch}}
Mn out in 0 0 nch w={{wn}} l={{lch}}
Cl out 0 2f
.ends
Vdd vdd 0 1.5
Vin in 0 PULSE(0 1.5 0.1n 50p 50p 1n 2n)
X1 in a vdd inv
X2 a b vdd inv
X3 b out vdd inv
"""

BSOURCE = """behavioral nonlinear circuit
.temp 27
.param gm=2m vsat=0.6 is0=1e-14 vth=0.02585
.func sat(x) = vsat * tanh(x / vsat)
Vin in 0 SIN(0 1 20meg)
Bgm 0 a I={gm * sat(v(in))}
Ra a 0 1k
Ca a 0 2p
Bsq b 0 V={0.5 * v(a) * v(a) + 0.1 * v(in, a)}
Rb b c 1k
Vsense c d 0
Dlike d 0 dmod
.model dmod d (is=1e-14 n=1)
Bmirror 0 e I={-2 * i(Vsense) + 1u * sin(6.283185307e7 * time)}
Re e 0 2k
Ce e 0 1p
Esat f 0 vol='v(e) > 0.5 ? 0.5 + 0.1 * (v(e) - 0.5) : v(e)'
Gcur 0 g cur='1m * v(f) * v(f)'
Rg g 0 1k
Etab h 0 TABLE {v(in)} = (-2, -1) (2, 1.5)
Gpol 0 k POLY(2) h 0 g 0 0 1m 0.5m 0.2m
Rk k 0 1k
Ck k 0 1p
"""

SOURCES = """independent source waveforms
Ve e 0 EXP(0.1 1 2n 3n 12n 4n)
Re e e2 1k
Ce e2 0 1p
Vf f 0 SFFM(0.2 1 200meg 1.5 20meg 30 60)
Rf f f2 1k
Cf f2 0 0.2p
Va a 0 AM(1 0.5 25meg 250meg 3n)
Ra a a2 1k
Ca a2 0 0.2p
Vp p 0 PWL(0 0 2n 1 4n 0.2 6n 1) r=2n td=1n
Rp p p2 1k
Cp p2 0 1p
Vq q 0 PULSE(-0.5 1 1n 0.5n 0.2n 3n 7n)
Iq 0 q2 SIN(0 1m 100meg 2n 1e7)
Rq q q2 1k
Cq q2 0 1p
"""

BINNED_MODELS = """
.model nch.1 nmos (level=1 kp=320u vto=0.42 lambda=0.08 cgso=0.2n cgdo=0.2n
+ lmin=0.1u lmax=0.5u wmin=0.1u wmax=100u)
.model nch.2 nmos (level=1 kp=280u vto=0.46 lambda=0.04 cgso=0.2n cgdo=0.2n
+ lmin=0.5u lmax=5u wmin=0.1u wmax=100u)
.model pch.1 pmos (level=1 kp=110u vto=-0.40 lambda=0.10 cgso=0.2n cgdo=0.2n
+ lmin=0.1u lmax=0.5u wmin=0.1u wmax=100u)
.model pch.2 pmos (level=1 kp=95u vto=-0.44 lambda=0.05 cgso=0.2n cgdo=0.2n
+ lmin=0.5u lmax=5u wmin=0.1u wmax=100u)
"""

BINNED = """binned level-1 models
""" + BINNED_MODELS + """
.subckt inv in out vdd l=0.2u
Mp out in vdd vdd pch w={3*l} l={l}
Mn out in 0 0 nch w={1.5*l} l={l}
Cl out 0 3f
.ends
Vdd vdd 0 1.5
Vin in 0 PULSE(0 1.5 0.1n 50p 50p 1.5n 3n)
X1 in a vdd inv l=0.2u
X2 a b vdd inv l=0.6u
X3 b out vdd inv l=0.499u
"""


def transient_case(name: str, deck: str, nodes: list[str], dt: float, t_end: float,
                   base_dir: Path | None = None, **parse_kwargs) -> dict:
    """Run `deck` in both simulators; return a table row of max errors."""
    ng = cm.run_ngspice(f"{deck}\n{EXACT}\n.tran {dt:g} {t_end:g} 0 {dt:g}",
                        [f"v({n})" for n in nodes])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit = vx.parse_netlist(deck, title=True, base_dir=base_dir,
                                   **parse_kwargs)
    ts = ng["time"]
    sol, first, steady = cm.time_call(lambda: vx.transient(circuit, ts, method="be"),
                                      repeats=2)
    worst = (0.0, 0.0, "")
    by_lower = {n.lower(): n for n in circuit.layout.node_names}
    for n in nodes:
        err, rel = cm.errors(sol.v(by_lower[n]), ng[f"v({n})"])
        if err >= worst[0]:
            worst = (err, rel, n)
    return {"case": name, "points": len(ts), "worst node": worst[2],
            "max abs err (V)": worst[0], "rel err": worst[1],
            "converged": bool(np.all(np.asarray(sol.converged))),
            "voltax run": cm.fmt_s(steady), "ngspice wall": cm.fmt_s(ng.wall)}


TABLE_XS = (-1.5, -0.7, -0.2, 0.25, 0.75, 1.5)
TABLE_POINTS = "(-1, -0.5) (0, 0) (0.5, 0.8) (1, 1)"


def table_dc_check() -> dict:
    """``E ... TABLE`` and ``G ... POLY(2)`` at DC. ngspice implements TABLE
    with the XSPICE ``pwl`` code model, which rounds the corners (within
    ~10% of a segment of each breakpoint); voltax interpolates linearly, as
    SPICE defines TABLE, so the inputs stay away from the corners."""
    lines = ["table and poly at dc"]
    for k, x in enumerate(TABLE_XS):
        lines += [f"V{k} x{k} 0 {x}",
                  f"E{k} h{k} 0 TABLE {{v(x{k})}} = {TABLE_POINTS}",
                  f"G{k} 0 k{k} POLY(2) x{k} 0 h{k} 0 0.1m 1m 0.5m 0.2m 0 -0.3m",
                  f"R{k} k{k} 0 1k"]
    deck = "\n".join(lines) + "\n.op"
    nodes = [f"{a}{k}" for k in range(len(TABLE_XS)) for a in "hk"]
    ng = cm.run_ngspice(deck, [f"v({n})" for n in nodes])
    sol = vx.dc(vx.parse_netlist(deck, title=True))
    ref = np.array([ng[f"v({n})"][0] for n in nodes])
    err, rel = cm.errors(np.array([sol.v(n) for n in nodes]), ref)
    return {"case": "TABLE + POLY(2) at DC", "points": len(TABLE_XS),
            "worst node": "", "max abs err (V)": err, "rel err": rel,
            "converged": bool(sol.converged), "voltax run": "", "ngspice wall": ""}


def debinned(deck: str, info) -> str:
    """`deck` with the binned cards renamed and every device pointing at
    the bin voltax chose (for ngspice, which does not bin Level-1 models).
    Works because each subcircuit instance here has its own geometry."""
    lines = []
    for line in deck.splitlines():
        line = re.sub(r"\b(nch|pch)\.(\d)\b", r"\1_\2", line)
        line = re.sub(r"lmin=\S+ lmax=\S+ wmin=\S+ wmax=[^\s)]+", "", line)
        lines.append(line)
    text = "\n".join(lines)
    # one subckt copy per instance, with the chosen bins substituted
    body = re.search(r"\.subckt inv.*?\.ends\n", text, re.DOTALL).group(0)
    copies, text = [], text.replace(body, "")
    for inst in ("X1", "X2", "X3"):
        sub = body.replace(".subckt inv", f".subckt inv_{inst}")
        for dev, model in (("Mp", "pch"), ("Mn", "nch")):
            card = info.devices[f"{inst}.{dev}"].card.replace(".", "_")
            sub = re.sub(rf"^({dev} .*?) {model} ", rf"\1 {card} ", sub, flags=re.M)
        copies.append(sub)
        text = re.sub(rf"^({inst} .*) inv\b", rf"\1 inv_{inst}", text, flags=re.M)
    return text + "\n" + "".join(copies)


def bin_selection_check() -> dict:
    """Bin choice of voltax vs ngspice BSIM3 (level 8) binning, on a grid of
    L/W including values on and within 1 nm of the bin edges.

    Each geometry is simulated once with the binned model name and once per
    bin with that bin's card forced; ngspice's choice is the bin whose
    forced current equals the binned current."""
    edges_l, edges_w = (0.1e-6, 0.5e-6, 5e-6), (0.2e-6, 2e-6, 50e-6)
    cards, bins = [], []
    for i in range(2):
        for j in range(2):
            k = len(bins) + 1
            bins.append(f"nch.{k}")
            cards.append(f".model nch.{k} nmos level=8 version=3.3.0 "
                         f"vth0={0.2 + 0.2 * k:g} lmin={edges_l[i]:g} "
                         f"lmax={edges_l[i + 1]:g} wmin={edges_w[j]:g} "
                         f"wmax={edges_w[j + 1]:g}")
    ls = [0.15e-6, 0.4989e-6, 0.4991e-6, 0.5e-6, 0.5001e-6, 2e-6, 4.9991e-6]
    ws = [0.3e-6, 1.9989e-6, 1.9991e-6, 2e-6, 10e-6]
    geoms = [(l, w) for l in ls for w in ws]
    lines = ["bin selection", *cards, "Vg g 0 1.5"]
    for n, (l, w) in enumerate(geoms):
        lines += [f"Vd{n} d{n} 0 1.0", f"M{n} d{n} g 0 0 nch w={w:.6g} l={l:.6g}"]
    deck = "\n".join(lines) + "\n.op"
    ng = cm.run_ngspice(deck, [f"i(vd{n})" for n in range(len(geoms))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = vx.parse_netlist(deck, title=True, return_info=True)

    forced = ["forced bins", "Vg g 0 1.5"]
    forced += [re.sub(r" lmin=.*", "", c).replace("nch.", "nchb") for c in cards]
    for n, (l, w) in enumerate(geoms):
        for k in range(len(bins)):
            forced += [f"Vf{n}_{k} f{n}_{k} 0 1.0",
                       f"Mf{n}_{k} f{n}_{k} g 0 0 nchb{k + 1} w={w:.6g} l={l:.6g}"]
    names = [f"i(vf{n}_{k})" for n in range(len(geoms)) for k in range(len(bins))]
    ref = cm.run_ngspice("\n".join(forced) + "\n.op", names)

    mismatches = []
    for n, (l, w) in enumerate(geoms):
        i_ng = ng[f"i(vd{n})"][0]
        k = int(np.argmin([abs(ref[f"i(vf{n}_{k})"][0] - i_ng)
                           for k in range(len(bins))]))
        if bins[k] != info.devices[f"M{n}"].card:
            mismatches.append(f"L={l:g} W={w:g}: ngspice {bins[k]}, voltax "
                              f"{info.devices[f'M{n}'].card}")
    for m in mismatches:
        print("  bin mismatch:", m)
    agree = len(geoms) - len(mismatches)
    return {"case": "bin selection vs BSIM3", "points": len(geoms),
            "worst node": "", "max abs err (V)": "", "rel err": "",
            "converged": "", "voltax run": f"{agree}/{len(geoms)} bins agree",
            "ngspice wall": ""}


def main() -> None:
    cm.header("General-purpose netlists: voltax vs ngspice")
    rows = []
    t_scale = cm.pick(0.5, 1.0)

    rows.append(transient_case("hierarchy", HIERARCHY, ["out", "xl.a", "xl.xa.mid"],
                               20e-12, 20e-9 * t_scale))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "models.lib").write_text(MODELS_LIB)
        (tmp / "process.inc").write_text(PROCESS_INC)
        for corner in cm.pick(("tt", "ss"), ("tt", "ff", "ss")):
            deck = LIB_DECK.format(corner=corner, lib=tmp / "models.lib")
            rows.append(transient_case(f"lib-{corner}", deck, ["a", "b", "out"],
                                       5e-12, 4e-9 * t_scale, base_dir=tmp,
                                       mos_model=LEVEL1))

    rows.append(transient_case("bsource", BSOURCE, ["a", "b", "e", "f", "g", "h", "k"],
                               0.25e-9, 100e-9 * t_scale))
    rows.append(table_dc_check())
    rows.append(transient_case("sources", SOURCES, ["e2", "f2", "a2", "p2", "q2"],
                               10e-12, 20e-9 * t_scale))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = vx.parse_netlist(BINNED, title=True, return_info=True)
    chosen = {d: info.devices[d].card for d in sorted(info.devices)}
    print("binned deck, bins chosen by voltax:", chosen)
    flat = debinned(BINNED, info)
    ng_row = transient_case("binned (vs de-binned)", flat, ["a", "b", "out"], 5e-12,
                            6e-9 * t_scale, mos_model=LEVEL1)
    # voltax on the *binned* deck, against ngspice on the de-binned one
    ng = cm.run_ngspice(f"{flat}\n{EXACT}\n.tran 5e-12 {6e-9 * t_scale:g} 0 5e-12",
                        ["v(out)"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binned = vx.parse_netlist(BINNED, title=True, mos_model=LEVEL1)
    sol = vx.transient(binned, ng["time"], method="be")
    err, rel = cm.errors(sol.v("out"), ng["v(out)"])
    ng_row.update({"case": "binned", "worst node": "out", "max abs err (V)": err,
                   "rel err": rel})
    rows.append(ng_row)
    rows.append(bin_selection_check())

    cm.print_table(rows, "Max error over all saved nodes (matched BE grid)")


if __name__ == "__main__":
    main()
