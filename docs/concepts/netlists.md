# SPICE netlists

`vx.parse_netlist(text)` turns a SPICE deck into a `Circuit`;
`vx.parse_netlist_file(path)` reads a file (whose first line is a title,
following SPICE). Device and node names are kept as written, so you address
results exactly as in the deck:

```python
import jax.numpy as jnp
import voltax as vx

c = vx.parse_netlist("""
* two-stage RC with a buffer
.param r1=1k
V1 in 0 PULSE(0 1 0 1n 1n 5u 10u) AC 1
R1 in a {r1}
C1 a 0 1n
E1 b 0 a 0 1
R2 b out 2k
C2 out 0 1n
""")
sol = vx.transient(c, jnp.linspace(0, 20e-6, 401))
print(sol.v("out")[-1], vx.ac(c, jnp.array([1e5])).db("out"))
```

The parser is written for real-world decks: foundry model libraries
(`.lib` corners, `.include` trees, thousands of `.param` expressions,
binned model cards) and typical ngspice testbenches. Its results are
checked against ngspice-42 in `benchmarks/benchmark_netlists.py`.

## Supported syntax

| element | syntax |
|---|---|
| resistor, capacitor, inductor | `R1 a b 1k`, `R1 a b r={expr}`, `R1 a b rmod l=10u w=1u` (model `r(rsh)`), `C1 a b 10p`, `L1 a b 1u`; `m=` on all three |
| coupled inductors | `K1 L1 L2 0.99` (merged into a `Transformer`) |
| voltage / current source | `V1 a b [DC] 1.2 [AC mag [phase]] [waveform]`, in any order; waveforms `PULSE SIN EXP PWL SFFM AM`, see [below](#sources) |
| diode | `D1 a k [model] [area]`, model `d(is n cjo tt rs)` |
| BJT | `Q1 c b e [s] [model] [area]`, model `npn`/`pnp` `(is bf br vaf tf tr cje cjc)` |
| MOSFET | `M1 d g s b model [w=] [l=] [m=] [nf=] [ad= as= pd= ps= nrd= nrs= ...]`, model `nmos`/`pmos` `(level kp vto lambda n gamma phi tox cgso cgdo)` |
| VCVS, VCCS | `E1 p n cp cn gain`, `G1 p n cp cn gm` |
| CCCS, CCVS | `F1 p n Vsense gain`, `H1 p n Vsense r` |
| behavioral | `B1 p n V={expr}` / `I={expr}`; `E1 p n value={expr}` (or `vol=`), `G1 p n value={expr}` (or `cur=`); `E`/`G` `TABLE {expr} = (x, y) ...`; `E`/`G`/`F`/`H` `POLY(n) ...` |
| switch | `S1 p n cp cn [model]`, model `sw(ron roff vt vh)` |
| subcircuit | `.subckt name pins... [params:] k=v ...` / `.ends`, instantiated with `X1 nodes... name [k=v ...] [m=]` |
| compiled device | `N1 nodes... model [k=v ...]` (ngspice OSDI / Verilog-A, e.g. `.model nch psp103va`); mapped with [`models=`](#plugging-in-your-own-models-models), MOSFET cards (`type=+1/-1`) otherwise fall back to the builder's model |

| directive | |
|---|---|
| `.param a=1 b={2*a}`, `.func f(x)={x*x}` | parameters and user functions, any order, scoped (below) |
| `.include file`, `.lib file section`, `.lib section` ... `.endl` | file inclusion and library corners |
| `.model name type (k=v ...)`, `name.1`, `name.2`, ... | model cards, with expressions and geometry bins |
| `.if (cond)` / `.elseif` / `.else` / `.endif` | conditional element lines and `.param`s, nested; conditions use the enclosing subcircuit instance's parameters |
| `.global n1 n2` | nodes shared by all subcircuit instances |
| `.options scale= defw= defl= temp= tnom=`, `.temp` | geometry scale, default W/L, temperature |
| `.tran tstep tstop` | supplies SPICE's default `PULSE`/`SIN`/`EXP`/`SFFM` timings |
| `.end` | stops reading |

Analysis and output directives (`.tran`, `.ac`, `.dc`, `.op`, `.print`,
`.meas`, `.control` blocks, ...) are otherwise skipped: run analyses from
Python. Lines continue with a leading `+` or a trailing `\`; comments are
`*` lines and inline `;`, `$ ` and `//`. Tabs and CRLF line endings are
fine. Names are case-insensitive (the first spelling is kept). Values accept
SPICE suffixes (`f p n u m k meg g t mil`) and ignore trailing units
(`10pF`).

Anything the parser does not understand raises `NetlistError` (a
`ValueError`) that starts with the file and line, e.g.
`cells.spice:12: M3: unknown model 'nch_lvt'`. Voltax never silently drops
part of your circuit.

## Expressions and parameters

An expression can appear wherever a value can: element values, model
parameters, instance and subcircuit parameters, source arguments. Write it in
braces `{...}` or single quotes `'...'` (inside `.param` lines plain
`a = b * 2` works too). The language is SPICE's:

* arithmetic `+ - * / %`, power `**` or `^`, comparisons `< <= > >= == !=`,
  logic `&& || !` and the ternary `c ? a : b`;
* functions `sqrt exp log ln log10 pow pwr abs min max sin cos tan asin acos
  atan atan2 sinh cosh tanh floor ceil int nint sgn sign if limit u uramp`
  (`log` is the natural log, as in ngspice), the constant `pi`;
* `agauss`, `gauss`, `aunif`, `unif` evaluate to their nominal value, so
  Monte-Carlo-ready foundry decks parse (sample mismatch in Python instead);
* numbers with SPICE suffixes, e.g. `{2*10u}`.

Parameters may reference each other in any order; a cycle is an error.
`.func` defines functions.

```python
c, info = vx.parse_netlist("""
.param rload = 'rbase * ratio' ratio = {vdd > 1 ? 2 : 1}
.param rbase=500 vdd=1.2
.func par(a, b) = a * b / (a + b)
V1 in 0 {vdd}
R1 in out {par(rload, rload)}
R2 out 0 1k
""", return_info=True)
print(info.params)
print(vx.dc(c).v("out"))
```

prints `{'rload': 1000.0, 'ratio': 2.0, 'rbase': 500.0, 'vdd': 1.2}`: `R1`
is two 1 kΩ in parallel, so `out` sits at 0.8 V.

## Subcircuits with parameters

Subcircuit parameters are declared on the `.subckt` line (with or without
`params:`), and their defaults may use other parameters. An instance
overrides them, evaluated in the *caller's* scope. Lookup goes from the
instance to its parent instance up to the global `.param`s, so a nested
subcircuit sees the parameters of the instance that contains it. `m=` on an
instance means `m` parallel copies of the whole subcircuit: resistances
divide, capacitances and transistor widths multiply.

```python
divider = """
.param rtop=3k
.subckt div top bot out params: ra=1k rb={2*ra}
R1 top out {ra}
R2 out bot {rb}
.ends div
V1 in 0 3
X1 in 0 a div
X2 in 0 b div ra=2k
X3 in 0 c div ra={rtop} rb=1k m=2
"""
c = vx.parse_netlist(divider)
sol = vx.dc(c)
print([float(sol.v(n)) for n in "abc"], float(c.get("X3.R1", "r")))
```

`a` and `b` both divide 3 V by 1:2 (`rb` follows `ra`), `c` by 3:1, and
`X3.R1` is the 3 kΩ resistor halved by `m=2`.

## Includes and libraries

`.include file` inlines a file; `.lib file section` inlines one
`.lib section` ... `.endl` block of a library file (corner selection, e.g.
`tt`, `ff`, `ss`). Relative paths resolve against the including file's
directory (`parse_netlist_file`), or against `base_dir=` for
`parse_netlist`. A file or section is inlined once per definition context:
a repeated top-level include is a no-op, but a file included inside several
`.subckt` bodies (as IHP does with its model cards) is inlined into each.
Recursion raises an error. Parameters are evaluated when used, so a
parameter file may be included before or after the library that uses it.

```python
import tempfile
from pathlib import Path

lib = Path(tempfile.mkdtemp())
(lib / "models.lib").write_text("""
.lib tt
.param dvt=0
.lib models.lib mos
.endl tt
.lib ff
.param dvt=-0.05
.lib models.lib mos
.endl ff
.lib mos
.model nch nmos (level=1 kp=200u vto={0.45 + dvt} lambda=0.05)
.endl mos
""")
deck = """
.lib models.lib {corner}
Vd d 0 1
Vg g 0 0.8
M1 d g 0 0 nch w=2u l=0.5u
"""
for corner in ("tt", "ff"):
    c = vx.parse_netlist(deck.format(corner=corner), base_dir=lib)
    print(corner, -float(vx.dc(c).i("Vd")))
```

prints the drain current in each corner: the `ff` threshold is 50 mV lower,
so it conducts more.

## Models

A MOSFET card with `level=1` uses `Level1MOSFET` (or the builder's
`mos_model` if that is a Level-1 variant, e.g.
`functools.partial(vx.Level1MOSFET, smooth=0.0)` for the exact SPICE
model). Without `level`, the builder's default model (`EKVMOSFET`) is used.
Any other level (BSIM4 `level=54`, PSP `level=103`, ...) falls back to the
builder's default model with a warning, unless you register a model for it
(next section). Parameters that the chosen model does not use trigger a
warning rather than being dropped silently. All devices that reference the
same card share one `MOSProcess`, so its gradient aggregates across them.

`return_info=True` also returns a `NetlistInfo`: the files read, the
parameters, and how every `.model` card maps:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the level-54 card warns
    c, info = vx.parse_netlist("""
.model n1 nmos level=1 kp=100u
.model nbsim nmos level=54 version=4.5 vth0=0.4
.model dd d is=1e-15
V1 d 0 1
M1 d d 0 0 n1
""", return_info=True)
print(info.model_report())
```

```text
type   level  status      cards  implementation
d      -      native          1  Diode
nmos   1      native          1  Level1MOSFET
nmos   54     fallback        1  EKVMOSFET (level 54 not implemented)
```

### Binning

Foundry decks split a model into geometry bins, `nch.1`, `nch.2`, ...,
each valid for `lmin <= L <= lmax`, `wmin <= W <= wmax`. A device that
references `nch` gets the bin its (scaled) geometry falls in, exactly as
ngspice-42 chooses it. Edges have a 1 nm tolerance and the last-defined
matching bin wins, so a device on a shared edge gets the upper bin. W is per
finger (`w/nf`).
`info.devices` records the choice and the device's relative distance to the
nearest edge it shares with another bin:

```python
c, info = vx.parse_netlist("""
.model nch.1 nmos (level=1 kp=300u vto=0.40 lmin=0.1u lmax=1u wmin=0.1u wmax=100u)
.model nch.2 nmos (level=1 kp=250u vto=0.45 lmin=1u lmax=10u wmin=0.1u wmax=100u)
V1 d 0 1
M1 d d 0 0 nch w=2u l=0.5u
M2 d d 0 0 nch w=2u l=2u
""", return_info=True)
for name in ("M1", "M2"):
    print(name, info.devices[name].card, info.devices[name].margin)
```

!!! warning "Binning makes W/L gradients piecewise"
    The bin is chosen once, at parse time. Gradients with respect to a
    device's `w`/`l` are those of the chosen bin's model: they do not see the
    jump to the neighbouring bin's parameters when an optimizer moves the
    geometry across an edge. Check `info.devices[name].margin` (the parser
    warns when a device is within 1% of a shared edge), and re-parse with
    the optimized geometry to pick up the right bin.

### Plugging in your own models (`models=`)

`parse_netlist(text, models={key: factory})` maps model cards to your own
`Element` classes, e.g. a compiled BSIM4. Keys are matched
case-insensitively, most specific first:

1. the model name a device references (`"nch"`) or its bin (`"nch.3"`),
2. `"<type>:<level>:<version>"`, e.g. `"nmos:54:4.5"`,
3. `"<type>:<level>"`, e.g. `"nmos:54"`, `"pmos:54"`, `"d:3"`,
4. `"<type>"`, e.g. `"nmos"`.

The factory is called as `factory.from_netlist(spec)` if it has that method
(e.g. a classmethod), otherwise as `factory(spec)`, and returns one
single-device `Element` whose nodes are `spec.nodes`. `spec` is a
`voltax.netlist.DeviceSpec`:

| field | content |
|---|---|
| `name` | hierarchical device name, `"X1.M3"` |
| `letter` | `"m"`, `"d"`, `"q"` or `"n"` (hooks apply to these elements; `N` requires one) |
| `nodes` | terminal node names: `(d, g, s, b)`, `(a, k)`, `(c, b, e[, s])`, or all nodes of an `N` device |
| `model` | the bin-selected `ModelCard`: `name`, `base`, `type`, `level`, `version`, `params` (all evaluated, lower-case keys), `text` (raw values) |
| `instance` | evaluated instance parameters: MOSFETs get `w`, `l` (and `ad as pd ps`) with `.option scale` applied, `m` (instance times subcircuit multiplier), and everything else as written (`nf`, `nrd`, `sa`, ...); diodes and BJTs get `area` and `m` |
| `options` | `.options` values plus `temp` (°C, default 27) |

All devices that use the same card receive the same `spec.model` object,
so a factory can share process parameters (and their gradients) by caching
on `spec.model.key`:

```python
class MyNMOS(vx.Level1MOSFET):
    processes = {}

    @classmethod
    def from_netlist(cls, spec):
        p = spec.model.params
        key = spec.model.key
        if key not in cls.processes:
            cls.processes[key] = vx.MOSProcess(kp=p["u0"] * 3.45e-11 / p["toxe"],
                                               vth=p["vth0"])
        w = spec.instance["w"] * spec.instance["m"]
        return cls(spec.nodes, w, spec.instance["l"], cls.processes[key], "n")

c = vx.parse_netlist("""
.model nch nmos level=54 version=4.5 vth0=0.42 u0=0.03 toxe=4n
V1 d 0 1
M1 d d 0 0 nch w=1u l=0.2u
M2 d d 0 0 nch w=1u l=0.2u
""", models={"nmos:54": MyNMOS})
print(type(c.elements["MyNMOS_n"]).__name__, c.elements["MyNMOS_n"].size)
```

## Behavioral sources

`B` elements and `E`/`G` with `value=` compile their expression into a
JAX function, a `BehavioralSource` element. The expression may use node
voltages `v(a)` and `v(a, b)`, the current through a voltage source
`i(Vx)`, `time`, parameters and functions. `V=` sources set a voltage (with
a branch current, like `E`), `I=` sources drive a current from `n+` through
the source to `n-`. Parameters stay differentiable:

```python
import equinox as eqx

c = vx.parse_netlist("""
.param gm=2m vsat=0.5
V1 in 0 0.3
B1 0 out I={gm * vsat * tanh(v(in) / vsat)}
R1 out 0 1k
Bx x 0 V={2 * v(out) + 1k * i(V1)}
Rx x 0 1k
""")
group, _ = c.layout.device("B1")
grad = eqx.filter_grad(lambda c: vx.dc(c).v("out"))(c)
print(float(vx.dc(c).v("out")), float(grad.elements[group].params["gm"][0]))
```

`TABLE {expr} = (x1, y1) (x2, y2) ...` interpolates linearly (constant
beyond the end points); `POLY(n)` builds the SPICE polynomial of `n`
controlling voltages (`E`/`G`) or source currents (`F`/`H`).

## Sources

| waveform | parameters (SPICE order) | signal |
|---|---|---|
| `PULSE` | `v1 v2 td tr tf pw per` | `signals.Pulse` |
| `SIN` | `vo va freq td theta phase` | `signals.Sine` |
| `EXP` | `v1 v2 td1 tau1 td2 tau2` | `signals.Exp` |
| `PWL` | `t1 v1 t2 v2 ... [r=] [td=]`, or `PWL FILE=name` (pairs, `#` comments) | `signals.PWL` |
| `SFFM` | `vo va fc mdi fs [phase_c phase_s]` | `signals.SFFM` |
| `AM` | `va vo mf fc td` | `signals.AM` |

Arguments may be expressions, with or without parentheses
(`SIN 0 1 {f0}`). With a `.tran tstep tstop` line, omitted or zero
arguments take SPICE's defaults (`tr = tf = tstep`, `pw = per = tstop`,
`freq = 1/tstop`, `tau = tstep`); without it, `EXP` needs explicit time
constants and `PULSE` falls back to `Pulse`'s defaults. A source has one
waveform; with both a `DC` value and a waveform, the waveform is used (also
at the operating point, as in ngspice's transient). `m=` multiplies current
sources.

## Open-PDK model decks

Point the deck at the PDK's ngspice model library and select a corner,
exactly as for ngspice:

```text
sky130:     .lib "<sky130_fd_pr>/models/sky130.lib.spice" tt
            XM1 d g s b sky130_fd_pr__nfet_01v8 w=1 l=0.15     (W/L in um)
gf180mcu:   .include "<gf180mcu_fd_pr>/models/ngspice/design.ngspice"
            .lib "<gf180mcu_fd_pr>/models/ngspice/sm141064.ngspice" typical
            M1 d g s b nmos_3p3 w=1u l=0.28u
ihp-sg13g2: .lib "<IHP-Open-PDK>/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt
            XM1 d g s b sg13_lv_nmos w=1u l=0.13u
```

and parse with `vx.parse_netlist(deck, return_info=True)`. The whole
hierarchy parses: corner sections, nested includes, thousands of `.param`
expressions (`agauss` at nominal), device subcircuit wrappers with
`.option scale`, IHP's `.if` blocks and `N` (OSDI) devices, and binned
cards. The bin chosen for each device matches ngspice-42; on a grid of
geometries `benchmarks/benchmark_pdk_decks.py` finds 112/112 agreeing sky130
devices and 98/98 gf180mcu devices.

What does *not* exist yet is the transistor physics: the core MOSFETs are
BSIM4 (`level=54`, sky130 and gf180mcu) or PSP 103 (Verilog-A, IHP), and
Voltax falls back to `EKVMOSFET` with a warning, so currents are only
qualitative unless you plug in the compiled BSIM4 (below). `info.model_report()` lists every card as `native` (resistors,
capacitors, BJTs), `fallback` (BSIM4, PSP, level-3 diodes) or `hook`; plug
real models in with [`models=`](#plugging-in-your-own-models-models), e.g.
`{"nmos:54": BSIM4, "pmos:54": BSIM4, "psp103va": PSP}`.

## Differences from ngspice

The parser follows ngspice-42 except where noted here:

* `m=` on a subcircuit instance multiplies everything below it, including
  nested subcircuit instances (HSPICE semantics); ngspice-42 only applies it
  to the devices directly inside the instance.
* `TABLE` interpolates linearly, as SPICE defines it; ngspice implements it
  with the XSPICE `pwl` code model, which rounds the corners.
* Without `.temp`, devices use Voltax's 300 K thermal voltage (`VT_300K`);
  ngspice defaults to 27 °C. Add `.temp 27` to match ngspice exactly.
* A MOSFET card without `level` uses the builder's default model
  (`EKVMOSFET`), not Level 1.
* A dot-less `include "file"` line (found in the sky130_fd_pr repository) is
  treated as `.include` with a warning, and skipped with a warning when the
  file does not exist; ngspice rejects the line.
* An `N` device whose card has `type = +1/-1` (a Verilog-A MOSFET) falls back
  to the builder's MOSFET model when no `models=` entry is given; ngspice
  needs the compiled OSDI library.
* Built-in models ignore junction and layout parameters (`ad as pd ps nrd
  nrs`, temperature coefficients); the parser warns once, listing them.

Not supported (they raise an error): XSPICE `A` devices, transmission
lines, `J`, `W`, `Z`, `U` and `O` elements, Laplace sources, `ddt()`/`idt()`
in expressions, and `N` devices that are neither hooked nor recognizably a
MOSFET.

Foundry BSIM4 cards (`level=54`) can use the compiled Verilog-A BSIM4 through
the `models=` hook, `parse_netlist(text, models=vx.va.ngspice_bsim4_models())`
(BSIM4 configured like ngspice's; the Verilog-A source is fetched on first
use). On the sky130 and gf180mcu decks it reproduces ngspice's drain
currents to ~1e-14. See [Verilog-A compact models](verilog-a.md#pdk-model-decks-sky130-gf180mcu).
