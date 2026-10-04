"""Open-PDK model decks: parse, build and solve in voltax; compare with ngspice.

Uses read-only mirrors of the open PDK model trees under
``~/.cache/voltax/pdk`` (override with ``VOLTAX_PDK_ROOT``), laid out as
``<owner>/<repo>/main/<path>`` like raw.githubusercontent.com:

* sky130: ``google/skywater-pdk-libs-sky130_fd_pr``, ``models/sky130.lib.spice``
  section ``tt``; devices are subcircuits (``sky130_fd_pr__nfet_01v8``,
  W/L in microns through ``.option scale=1.0u``), binned BSIM4 cards.
* gf180mcu: ``google/globalfoundries-pdk-libs-gf180mcu_fd_pr``,
  ``models/ngspice/design.ngspice`` + ``sm141064.ngspice`` section
  ``typical``; binned BSIM4 ``nmos_3p3``/``pmos_3p3``.
* ihp-sg13g2: ``IHP-GmbH/IHP-Open-PDK``, ``.../models/cornerMOSlv.lib``
  section ``mos_tt``; subcircuits with ``.if`` blocks around ``N`` (OSDI)
  devices using PSP 103 Verilog-A models.

For each PDK the script prints the files read, the model mapping report,
and the inverter's output voltage at ``vin = vdd/2`` from voltax (EKV
fallbacks for BSIM4/PSP, so it *differs* from ngspice's BSIM4 by design)
and from ngspice. It then checks voltax's bin selection against ngspice's
(``show <device> : model``) on a grid of geometries.

Finally, for the BSIM4 PDKs (sky130, gf180mcu) the same decks run with the
compiled Verilog-A BSIM4 (`vx.va.ngspice_bsim4_models()`, fetched on first
use) against ngspice's built-in BSIM4: drain currents of NMOS/PMOS at a grid
of (Vgs, Vds) and the inverter VTC. Both PDKs' cards say ``version = 4.5``,
which ngspice runs with its BSIM4.5.0 code while the Verilog-A file is
BSIM4.8.0; the comparison is therefore also repeated against ngspice on a copy
of the cards with ``version = 4.8`` (same equations on both sides).

The sky130 repository contains one line ngspice rejects (a bare ``include``
of a file that does not exist); ngspice runs on a symlinked copy of the tree
with that line commented out. IHP needs the PSP OSDI library in ngspice,
which is not available here, so it is voltax-only.

Run:  uv run python benchmarks/benchmark_pdk_decks.py  (VOLTAX_FAST=1: fewer bins)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # LAPACK slows badly under load

import _common as cm  # noqa: E402  (first: configures JAX for single-threaded CPU)
import jax.numpy as jnp
import numpy as np

import voltax as vx

ROOT = Path(os.environ.get("VOLTAX_PDK_ROOT", Path.home() / ".cache/voltax/pdk"))
SKY = ROOT / "google/skywater-pdk-libs-sky130_fd_pr/main"
GF = ROOT / "google/globalfoundries-pdk-libs-gf180mcu_fd_pr/main/models/ngspice"
IHP = ROOT / "IHP-GmbH/IHP-Open-PDK/main/ihp-sg13g2/libs.tech/ngspice/models"


def sky130_header(root: Path) -> str:
    return f'.lib "{root}/models/sky130.lib.spice" tt\n'


def gf180_header(root: Path) -> str:
    return f'.include "{root}/design.ngspice"\n.lib "{root}/sm141064.ngspice" typical\n'


def ihp_header(root: Path) -> str:
    return f'.lib "{root}/cornerMOSlv.lib" mos_tt\n'


PDKS = {
    "sky130": dict(
        root=SKY, header=sky130_header, vdd=1.8,
        devices="""XMP out in vdd vdd sky130_fd_pr__pfet_01v8 w=2 l=0.15
XMN out in 0 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15
""",
        # (W, L) in microns, both polarities
        grid=[(w, l) for w in (0.42, 0.55, 1.0, 1.65, 3.0, 5.0, 7.0)
              for l in (0.15, 0.18, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)],
        instance="X{k}{p} d{k}{p} g 0 0 sky130_fd_pr__{p}fet_01v8 w={w:g} l={l:g}",
        device="m.x{k}{p}.msky130_fd_pr__{p}fet_01v8",
        voltax_device="X{k}{p}.msky130_fd_pr__{p}fet_01v8",
        iv_size=(1.0, 0.15),
        iv_instance="X{k} d{k} g{k} 0 0 sky130_fd_pr__{p}fet_01v8 w={w:g} l={l:g}",
    ),
    "gf180mcu": dict(
        root=GF, header=gf180_header, vdd=3.3,
        devices="""MP out in vdd vdd pmos_3p3 w=2u l=0.28u
MN out in 0 0 nmos_3p3 w=1u l=0.28u
""",
        grid=[(w, l) for w in (0.22, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
              for l in (0.28, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0)],
        instance="M{k}{p} d{k}{p} g 0 0 {p}mos_3p3 w={w:g}u l={l:g}u",
        device="m{k}{p}",
        voltax_device="M{k}{p}",
        iv_size=(1.0, 0.28),
        iv_instance="M{k} d{k} g{k} 0 0 {p}mos_3p3 w={w:g}u l={l:g}u",
    ),
    "ihp-sg13g2": dict(
        root=IHP, header=ihp_header, vdd=1.2, ngspice=False,
        devices="""XMP out in vdd vdd sg13_lv_pmos w=2u l=0.13u
XMN out in 0 0 sg13_lv_nmos w=1u l=0.13u
""",
    ),
}


NGSPICE_TIGHT = ".options reltol=1e-9 abstol=1e-18 vntol=1e-12"


def ngspice_tree(name: str, cfg: dict, tmp: Path, version: str | None = None) -> Path:
    """Root to give ngspice: the mirror, or a copy (sky130 ~30 MB) with the
    bare ``include`` line commented out and, with ``version``, every BSIM4
    ``version = 4.5`` card rewritten to that version. (Symlinks would not
    do: ngspice resolves relative includes from the link target's directory.)

    ngspice dispatches ``level=54`` cards with ``version = 4.5`` to its
    BSIM4.5.0 code (``BSIM4v5``); the Verilog-A model is BSIM4.8, so an
    apples-to-apples reference needs ``version = 4.8``."""
    if name != "sky130" and version is None:
        return cfg["root"]
    copy = tmp / f"{name}{'-v' + version if version else ''}"
    if copy.exists():
        return copy
    for src in cfg["root"].rglob("*"):
        dst = copy / src.relative_to(cfg["root"])
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(errors="replace")
        if name == "sky130":
            text = re.sub(r"^(\s*include\s)", r"* \1", text, flags=re.MULTILINE)
        if version is not None:
            text = re.sub(r"(\bversion\s*=\s*)4\.50*\b", rf"\g<1>{version}", text,
                          flags=re.IGNORECASE)
        dst.write_text(text)
    return copy


def parse(deck: str) -> tuple[vx.Circuit, vx.netlist.NetlistInfo, float, list]:
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit, info = vx.parse_netlist(deck, return_info=True)
    return circuit, info, time.perf_counter() - start, caught


def inverter(name: str, cfg: dict, tmp: Path) -> dict:
    vdd = cfg["vdd"]
    body = f"Vdd vdd 0 {vdd}\nVin in 0 {vdd / 2}\n{cfg['devices']}"
    circuit, info, seconds, caught = parse(cfg["header"](cfg["root"]) + body)
    sol = vx.dc(circuit)
    print(f"\n{name}: {len(info.files)} files, parsed in {seconds:.1f} s, "
          f"{len(info.models)} model cards, {len(caught)} warnings")
    for d, m in sorted(info.devices.items()):
        print(f"  {d}: {m.card} -> {m.implementation}")
    print("  " + info.model_report().replace("\n", "\n  "))
    row = {"PDK": name, "files": len(info.files), "cards": len(info.models),
           "devices": len(info.devices),
           "voltax v(out) [EKV fallback]": float(sol.v("out")),
           "converged": bool(sol.converged)}
    if cfg.get("ngspice", True):
        root = ngspice_tree(name, cfg, tmp)
        ng = cm.run_ngspice(f"* {name}\n{cfg['header'](root)}{body}.op", ["v(out)"])
        row["ngspice v(out) [BSIM4]"] = float(ng["v(out)"][0])
    else:
        row["ngspice v(out) [BSIM4]"] = "needs psp103va.osdi"
    return row


def bin_check(name: str, cfg: dict, tmp: Path) -> dict:
    """Which bin ngspice and voltax pick for each geometry."""
    grid = cfg["grid"][::3] if cm.FAST else cfg["grid"]
    lines = [f"Vg g 0 {cfg['vdd'] / 2}"]
    keys = []
    for k, (w, l) in enumerate(grid):
        for p in ("n", "p"):
            lines += [f"Vd{k}{p} d{k}{p} 0 0.1",
                      cfg["instance"].format(k=k, p=p, w=w, l=l)]
            keys.append((k, p, w, l))
    body = "\n".join(lines) + "\n"
    _, info, _, _ = parse(cfg["header"](cfg["root"]) + body)
    # ngspice names subcircuit-local cards "x1:name.N" and truncates them in
    # `show` output, so identify the bin by its bounds instead
    shows = "\n".join(f"echo @@dev\nshowmod {cfg['device'].format(k=k, p=p)} : "
                      "lmin lmax wmin wmax" for k, p, _, _ in keys)
    root = ngspice_tree(name, cfg, tmp)
    deck = tmp / f"{name}_bins.sp"
    deck.write_text(f"* bins\n{cfg['header'](root)}{body}.op\n.control\nop\n"
                    f"{shows}\n.endc\n.end\n")
    out = subprocess.run([cm.NGSPICE, "-b", str(deck)], capture_output=True,
                         text=True, errors="replace").stdout
    ng_bounds = []
    for chunk in out.split("@@dev")[1:]:
        found: dict[str, str] = {}  # first match: later output may repeat keys
        for key, value in re.findall(r"^\s*(lmin|lmax|wmin|wmax)\s+(\S+)\s*$",
                                     chunk, re.MULTILINE):
            found.setdefault(key, value)
        ng_bounds.append(tuple(float(found.get(k, "nan"))
                               for k in ("lmin", "lmax", "wmin", "wmax")))
    if len(ng_bounds) != len(keys):
        raise RuntimeError(f"{name}: ngspice reported {len(ng_bounds)} bins for "
                           f"{len(keys)} devices:\n{out[-2000:]}")
    mismatches = []
    for (k, p, w, l), ng in zip(keys, ng_bounds):
        card = info.devices[cfg["voltax_device"].format(k=k, p=p)]
        if not np.allclose(card.bounds, ng, rtol=1e-9):
            mismatches.append(f"{p}fet W={w:g} L={l:g}: ngspice bin {ng}, voltax "
                              f"{card.card} {card.bounds}")
    for m in mismatches:
        print("  bin mismatch:", m)
    return {"PDK": name, "devices": len(keys),
            "bins agree": f"{len(keys) - len(mismatches)}/{len(keys)}"}


def _iv_deck(cfg: dict) -> tuple[str, list[tuple[str, float, float]]]:
    """Independently biased NMOS/PMOS devices on a (Vgs, Vds) grid."""
    vdd = cfg["vdd"]
    fr = [0.2, 0.4, 0.6, 1.0] if cm.FAST else [0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    lines, keys = [], []
    k = 0
    for p, s in (("n", 1.0), ("p", -1.0)):
        for fg in fr:
            for fd in (0.03, 0.5, 1.0):
                vg, vd = s * fg * vdd, s * fd * vdd
                w, l = cfg["iv_size"]
                lines += [f"Vg{k} g{k} 0 {vg:.6g}", f"Vd{k} d{k} 0 {vd:.6g}",
                          cfg["iv_instance"].format(k=k, p=p, w=w, l=l)]
                keys.append((p, vg, vd))
                k += 1
    return "\n".join(lines) + "\n", keys


def bsim4_compare(name: str, cfg: dict, tmp: Path,
                  version: str | None = None) -> list[dict]:
    """Compiled Verilog-A BSIM4 vs ngspice BSIM4 on the PDK's own cards
    (``version``: BSIM4 version forced on the ngspice side, see
    :func:`ngspice_tree`)."""
    try:
        models = vx.va.ngspice_bsim4_models()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"skipping BSIM4 comparison: {e}")
        return []
    root = ngspice_tree(name, cfg, tmp, version)
    label = f"{name} (ngspice {'BSIM' + version if version else 'as is'})"
    opts = vx.Options(gmin=0.0)
    rows = []
    # I-V
    body, keys = _iv_deck(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit = vx.parse_netlist(cfg["header"](cfg["root"]) + body,
                                   models=models)
    sol, first, steady = cm.time_call(lambda: vx.dc(circuit, options=opts),
                                      repeats=1)
    ng = cm.run_ngspice(f"* {name} iv\n{cfg['header'](root)}{body}.op",
                        [f"i(vd{k})" for k in range(len(keys))])
    for p in ("n", "p"):
        idx = [k for k, key in enumerate(keys) if key[0] == p]
        mine = np.array([float(sol.i(f"Vd{k}")) for k in idx])
        ref = np.array([float(ng[f"i(vd{k})"][0]) for k in idx])
        rel = np.abs(mine - ref) / np.abs(ref)
        rows.append({"PDK": label, "test": f"{p}fet Id, {len(idx)} (Vgs,Vds)",
                     "max rel err": float(rel.max()),
                     "|Id| range (A)":
                         f"{np.abs(ref).min():.2g}..{np.abs(ref).max():.2g}",
                     "voltax 1st": cm.fmt_s(first), "conv": bool(sol.converged)})
    # inverter VTC
    vdd = cfg["vdd"]
    step = vdd / (12 if cm.FAST else 36)
    body = f"Vdd vdd 0 {vdd}\nVin in 0 0\n{cfg['devices']}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit = vx.parse_netlist(cfg["header"](cfg["root"]) + body,
                                   models=models)
    # tight tolerances: near the switching point the gain is ~100, and
    # ngspice's default reltol=1e-3 leaves mV-level errors in v(out)
    ng = cm.run_ngspice(f"* {name} vtc\n{cfg['header'](root)}{body}"
                        f"{NGSPICE_TIGHT}\n.dc Vin 0 {vdd + step / 2:.6g} {step:.6g}",
                        ["v(in)", "v(out)"])
    sol, first, steady = cm.time_call(
        lambda: vx.dc_sweep(circuit, jnp.asarray(ng["v(in)"]), "Vin"), repeats=1)
    err = np.abs(np.asarray(sol.v("out")) - ng["v(out)"])
    worst = int(np.argmax(err))
    mid = int(np.argmin(np.abs(ng["v(in)"] - vdd / 2)))
    rows.append({"PDK": label, "test": f"inverter VTC, {len(err)} points",
                 "max |dV| (V)": float(err.max()),
                 "at vin (V)": float(ng["v(in)"][worst]),
                 "v(out) at vdd/2: voltax / ngspice":
                     f"{float(sol.v('out')[mid]):.6f} / {ng['v(out)'][mid]:.6f}",
                 "voltax 1st": cm.fmt_s(first),
                 "conv": bool(np.all(sol.converged))})
    return rows


def main() -> None:
    cm.header("Open-PDK model decks: voltax vs ngspice")
    available = {n: c for n, c in PDKS.items() if c["root"].exists()}
    for n in set(PDKS) - set(available):
        print(f"skipping {n}: {PDKS[n]['root']} not found")
    rows, bins = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, cfg in available.items():
            rows.append(inverter(name, cfg, tmp))
        for name, cfg in available.items():
            if "grid" in cfg:
                bins.append(bin_check(name, cfg, tmp))
        bsim4 = []
        for name, cfg in available.items():
            if "iv_instance" in cfg:
                bsim4 += bsim4_compare(name, cfg, tmp)
                bsim4 += bsim4_compare(name, cfg, tmp, version="4.8")
    cm.print_table(rows, "Inverter at vin = vdd/2 (voltax uses EKV for BSIM4/PSP)")
    cm.print_table(bins, "Bin selection, voltax vs ngspice")
    if bsim4:
        keys: list[str] = []
        for r in bsim4:
            keys += [k for k in r if k not in keys]
        cm.print_table([{k: r.get(k, "") for k in keys} for r in bsim4],
                       "Compiled Verilog-A BSIM4 (ngspice-compatible) vs ngspice "
                       "BSIM4, same decks and cards")


if __name__ == "__main__":
    main()
