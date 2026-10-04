# Verilog-A compact models

Industry transistor models (BSIM4, BSIM-CMG, PSP, HICUM, ...) are published as
Verilog-A. `voltax.va` compiles a Verilog-A module into an ordinary Voltax
`Element` class, so a foundry-grade model works with every analysis, with
`jit`/`vmap`, and with exact gradients. BSIM4 4.8 is validated against
ngspice's built-in BSIM4 (`level=54`) to round-off, on its own cards and on
the sky130 and gf180mcu PDK decks. Its Verilog-A source is licensed
CC BY-NC 4.0, so it is not part of Voltax: it is
[downloaded on first use](#getting-bsim4-licence).

## Quick start

```python
import jax
import voltax as vx

bsim4 = vx.va.load("bsim4")                  # BSIM4 4.8 Verilog-A (fetched once)
print(bsim4)
name, kind, card = vx.va.bundled_card("freepdk45_nmos.inc")  # PTM 45 nm card
NCH = bsim4.element(card, polarity="n", name=name)           # an Element class
m1 = NCH(("d", "g", "0", "0"), w=90e-9, l=50e-9)
print(m1.internal_names)
```

`load` parses the source once; `element` compiles it for one model card and
returns an `Element` subclass (cached per card). Instances take the
Verilog-A instance parameters (`w`, `l`, `nf`, `ad`, ...). This card enables
a gate resistor (`rgatemod=1`) and the substrate network (`rbodymod=1`), so
the device has four internal nodes, solved like any other unknown.

Use it like a built-in element:

```python
b = vx.CircuitBuilder()
b.vsource("d", "0", 1.0, name="Vd")
b.vsource("g", "0", 1.0, name="Vg")
b.add(m1, name="M1")
c = b.build()
opts = vx.Options(gmin=0.0)
print(f"Id = {-vx.dc(c, options=opts).i('Vd'):.6e} A")

def drain_current(c):
    return -vx.dc(c, options=opts).i("Vd")

grads = jax.grad(drain_current)(c)
print(grads.elements["nmos_vtg_n"].inst["log_w"])   # dId / dlog(W)
```

Real instance parameters are per-device arrays (positive ones in log-space,
like every Voltax parameter), so devices with different sizes still fuse into
one vectorized group and get gradients. Model parameters are compile-time
constants unless you ask for them:

```python
NCH_D = bsim4.element(card, polarity="n", name=name,
                      differentiable=("vth0", "u0", "toxe"))
m2 = NCH_D(("d", "g", "0", "0"), w=90e-9, l=50e-9)
print(sorted(m2.model.physical()))
```

They then live in a shared `VAParams` (like `MOSProcess`): every device built
from the class reads the same values, gradients aggregate over them, and
`element.with_model(vth0=0.45)` returns a modified copy.

### Netlists

`parse_netlist` routes MOSFET cards of a given `level` to a compiled model:

```python
cards = (vx.va.MODELS_DIR / "cards")
net = f"""
Vdd vdd 0 1.0
Vin in 0 0.45
Mp out in vdd vdd pmos_vtg w=180n l=50n
Mn out in 0 0 nmos_vtg w=90n l=50n
{(cards / "freepdk45_nmos.inc").read_text()}
{(cards / "freepdk45_pmos.inc").read_text()}
"""
inv = vx.parse_netlist(net, models=vx.va.level_models(bsim4, level=54))
print(inv.summary())
print(f"Vout = {vx.dc(inv).v('out'):.4f} V")
```

Every `.model` card becomes one element class, so all devices of a card fuse
into one group. Keyword arguments of `level_models` (`differentiable`,
`smooth`, `equality`, `temperature`, `defaults`, `force`) apply to every card.
To reproduce ngspice's BSIM4 (foundry decks), use
`vx.va.ngspice_bsim4_models()`: BSIM4 with ngspice's constants, defaults and
analysis semantics (see [PDK model decks](#pdk-model-decks-sky130-gf180mcu)).

## Getting BSIM4 (licence)

The BSIM4 4.8 Verilog-A (BSIM Group, UC Berkeley; Verilog-A port from
[cogenda/VA-BSIM48](https://github.com/cogenda/VA-BSIM48)) is licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
attribution required, **non-commercial use only**. That is incompatible
with Voltax's MIT licence, so the file is not in the repository or the
package. `vx.va.load("bsim4")` finds it, in order:

1. an explicit path: `vx.va.load("/path/to/bsim4.va")`;
2. the environment variable `VOLTAX_BSIM4_VA` (a path);
3. the cache, `~/.cache/voltax/va/bsim4.va` (`$XDG_CACHE_HOME` respected);
4. otherwise it downloads the file from a pinned commit of
   [dwarning/VA-Models](https://github.com/dwarning/VA-Models)
   (`code/bsim4/vacode/bsim4.va`), checks its SHA-256, caches it, and
   warns once with the licence terms.

`vx.va.fetch("bsim4")` does step 4 explicitly (e.g. before going offline)
and returns the cached path:

```python
path = vx.va.fetch("bsim4")      # no-op when the cached copy verifies
print(path.name, path.parent == vx.va.cache_dir())
```

Offline and without a cached copy, `load` raises an error that explains
how to supply the file. Tests that need BSIM4 skip in that case.

## Your own Verilog-A

Any module in the supported subset compiles the same way, from a file or a
string:

```python
diode_va = """
`include "disciplines.vams"
module dio(a, c);
  inout a, c; electrical a, c;
  parameter real is = 1e-14 from (0:inf);
  parameter real n = 1.0;
  parameter real cj = 0.0 from [0:inf);
  real vd, id;
  analog begin
    vd = V(a, c);
    id = is * (limexp(vd / (n * $vt)) - 1.0);
    I(a, c) <+ id;
    I(a, c) <+ ddt(cj * vd);
  end
endmodule
"""
DIO = vx.va.load(diode_va).element({"is": 1e-15, "cj": 1e-12})
d1 = DIO(("a", "0"))
print(d1.source)
```

The generated code is plain Python: `f` returns the resistive contributions
(currents into each terminal, then internal residuals), `q` the arguments of
`ddt`. Every line points back to its Verilog-A source line, and the source
is registered with `linecache`, so tracebacks show it.

## How the compiler works

The pipeline is `preprocess` (macros, `` `include``, `` `ifdef``, keeping
source line numbers) → `parse` (an AST) → `compile_module` → `VAElement`.

The compiler is an **online partial evaluator**: it *executes* the analog
block with every value either

* **static**, a Python number known at compile time: integer parameters
  (model selectors such as `capmod`, `rgatemod`), `$param_given`,
  `$simparam`, constants, and real model parameters that are not
  `differentiable`; or
* **traced**, a JAX array computed by generated code: terminal voltages,
  internal unknowns, real instance parameters, `differentiable` model
  parameters, and `$temperature` when `temperature=None`.

Operations on static values are evaluated in Python with C semantics, so
selector branches leave no trace in the generated code (BSIM4 with this card:
about 1000 statements in `f`, out of ~9700 source lines). Binding times are
tracked per program point, not per variable: a variable can be static on one
path and traced on another. `for`/`while` loops with static conditions are
unrolled; a loop whose condition also has a traced part is unrolled as nested
traced `if`s while its static part (an iteration counter) allows. Analog
functions are inlined. Dead code (e.g. BSIM4's noise expressions) is removed.

A condition on traced values becomes a `jnp.where` merge (an SSA phi) of both
branches. Every outer value the branch reads goes through
`rt.guard(cond, x) = where(cond, x, stop_gradient(x))`, the **double-where**
guard: in lanes where the branch is not taken it may compute `sqrt(-1)` or
`1/0`, but both its values and its (forward- and reverse-mode) derivatives are
discarded by selects instead of multiplied by zero, so no NaN leaks into
gradients. `V(a, b) <+ 0` under static conditions collapses nodes (BSIM4's
`rdsmod=0` drain/source nodes); other voltage contributions get a
branch-current unknown; remaining internal nodes are internal unknowns.

**Why generated source?** Code generation happens once per (model card,
static instance configuration) and costs about half a second for BSIM4. The
result is readable, diffable and debuggable Python (`element.source`), JAX
traces it like hand-written code, and the static/traced split is decided
before tracing, so `if` on static values is just Python. The functions are
wrapped in `jax.jit`, so repeated traces inside analyses reuse one jaxpr.

## Supported Verilog-A

| construct | support |
|---|---|
| preprocessor | `` `define`` (with arguments, multi-line), `` `undef``, `` `ifdef``/`` `ifndef``/`` `elsif``/`` `else``, `` `include`` (built-in `disciplines.vams`, `constants.vams`) |
| module items | ports, `inout`/`input`/`output`, `electrical`, `ground`, `branch`, `parameter`/`localparam` `real`/`integer`/`string` with `from`/`exclude`, `aliasparam`, `(* type="instance" *)`, `real`/`integer`/`genvar`, `analog function` |
| statements | `begin`/`end` (named, local declarations), `if`/`else`, `case`, `for`, `while`, `repeat`, `@(initial_step)` and friends (run every evaluation; static parts fold), system tasks (`$strobe` ... ignored; `$finish`/`$error` raise if reached statically) |
| contributions | `I(a,b) <+`, `V(a,b) <+`, named branches, `ddt()` in linear combinations, noise sources (dropped) |
| expressions | all Verilog operators, `?:`, `exp ln log sqrt pow abs min max floor ceil` and (inverse/hyperbolic) trig, `hypot atan2 limexp $limit $temperature $vt $simparam $param_given $port_connected $mfactor analysis("noise")` |
| not supported | arrays, `idt`/`ddx`/`laplace_*`/`transition`, `analysis()` of dc/tran/ac, `V(a,b) <+ 0` under a bias-dependent condition, `ddt()` times a bias-dependent factor, digital/event constructs |

Unsupported input raises `VAError` with `file:line`.

## Validation: BSIM4 vs ngspice

`benchmarks/benchmark_bsim4.py` compares the compiled BSIM4 with ngspice 42's
C implementation on the bundled FreePDK45/PTM cards (gate resistor, substrate
network and gate tunneling on), on ngspice's own sweep points and time grid.

| test | as published | with ngspice constants |
|---|---|---|
| Id-Vgs (lin/sat, subthreshold, Vbs=-0.3), NMOS+PMOS | 2.7e-4 | 4.4e-14 |
| Id-Vds families | 1.4e-5 | 6.9e-14 |
| gm, gds (AD Jacobian) | 1.0e-3 | 5.4e-14 |
| Cgg, Cgd, Cgs (AD of charges) | 5.8e-6 | 6.7e-13 |
| gate / bulk current | 1.6e-4 / 1.9e-4 | 1.5e-4 / 1.9e-5 (see below) |
| inverter VTC, 101 points (max \|dV\|) | | 3.7e-10 V |
| 3-inverter chain transient, 2487 steps (max \|dV\|) | | 3.5e-7 V |
| 5-stage ring oscillator frequency | | 4.1e-5 (1.1e-6 with ngspice `reltol=1e-9`) |

(maximum pointwise relative errors). Two kinds of differences were found and
explained:

* **Physical constants.** The Verilog-A computes kT/q as `` `P_K/`P_Q`` with
  the NIST-1998 values of `constants.vams` (8.617343e-5 V/K); ngspice's C
  hard-codes `KboQ = 8.617087e-5`, a 3e-5 relative difference in the thermal
  voltage, which is 2e-4 in subthreshold current. The C also uses
  `Charge_q = 1.60219e-19`, `EPS0 = 8.85418e-12`, `PI = 3.141592654` and,
  in poly-depletion only, `CHARGE = 1.6021766208e-19`.
  `vx.va.NGSPICE_BSIM4_OVERRIDES` pins these via locked macros (the Verilog-A
  source stays untouched): `vx.va.load("bsim4", overrides=...)`.
* **Defaults.** Five parameter defaults differ between `bsim4.va` and
  ngspice's `b4set.c` (`gidlmod` 1 vs 0, `cvchargemod` 1 vs 0, `aigbacc`,
  `lwn`, `lc`); a card that does not set them gives different models.
  `vx.va.NGSPICE_BSIM4_DEFAULTS` holds ngspice's values.

With both, the equations are identical: drain/bulk currents and the AC
small-signal parameters agree to 1e-14...1e-13. Gate currents agree to
1e-6...1.5e-4 (worst for the smallest currents): they flow through the 0.36 Ω
gate resistor, so both simulators carry a
float64 quantization of `ulp(V)/R` (~1e-16 A on a pA current) in the internal
node voltage (and through the 5-15 Ω substrate resistors for bulk currents
under reverse body bias). Transients agree to ngspice's Newton tolerance:
tightening ngspice from `reltol=1e-6` to `1e-9` takes the ring-oscillator
frequency agreement from 4e-5 to 1e-6. (Its waveforms keep a constant phase
offset from the `uic` start, where the two simulators initialize the internal
nodes differently.)

## PDK model decks (sky130, gf180mcu)

The open sky130 and gf180mcu PDKs ship binned BSIM4 cards. With
`models=vx.va.ngspice_bsim4_models()` the netlist parser picks each device's
bin (as ngspice does) and compiles one element class per card used:

```python
import warnings
from pathlib import Path

GF = (Path.home() / ".cache/voltax/pdk/google/globalfoundries-pdk-libs-"
      "gf180mcu_fd_pr/main/models/ngspice")
if GF.exists():                  # a local mirror of the PDK's model files
    deck = f"""
.include "{GF}/design.ngspice"
.lib "{GF}/sm141064.ngspice" typical
Vdd vdd 0 3.3
Vin in 0 1.65
MP out in vdd vdd pmos_3p3 w=2u l=0.28u
MN out in 0 0 nmos_3p3 w=1u l=0.28u
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # unused-parameter notes from the deck
        inv = vx.parse_netlist(deck, models=vx.va.ngspice_bsim4_models())
    print(f"v(out) = {vx.dc(inv, options=vx.Options(gmin=0.0)).v('out'):.6f} V")
```

`benchmarks/benchmark_pdk_decks.py` runs the same decks in ngspice 42
(drain current of NMOS and PMOS on a grid of 21 (Vgs, Vds) points each,
from subthreshold to saturation, and the inverter transfer curve):

| PDK | test | vs ngspice as distributed | vs ngspice with `version = 4.8` |
|---|---|---|---|
| sky130 (`nfet_01v8`, W/L = 1/0.15 µm) | NMOS Id, 21 points (22 pA-0.5 mA) | 3.6e-4 | 1.4e-14 |
| | PMOS Id, 21 points | 1.1e-5 | 2.5e-14 |
| | inverter VTC, 37 points (max \|dV\|) | 3.6e-8 V | 3.6e-8 V |
| | v(out) at vin = 0.9 V (ngspice 0.498914 V) | 0.498914 V | |
| gf180mcu (`nmos_3p3`, W/L = 1/0.28 µm) | NMOS Id, 21 points (26 nA-0.5 mA) | 1.3e-3 | 7.6e-15 |
| | PMOS Id, 21 points | 1.2e-3 | 1.3e-14 |
| | inverter VTC, 37 points (max \|dV\|) | 9.2e-5 V | 4.9e-8 V |
| | v(out) at vin = 1.65 V (ngspice 0.468247 V) | 0.468263 V | |

(Maximum relative errors unless stated. ngspice runs the VTC with
`reltol=1e-9`; at its default `reltol=1e-3` its own error near the
switching point is ~1 mV. Bin selection agrees for all 112 sky130 and 98
gf180mcu test geometries.)

Two things had to be understood to get there:

* **sky130: zero current.** The sky130 cards set `tnoimod=1` (thermal noise
  model 1) with `rdsmod=0`. ngspice creates the internal drain/source prime
  nodes for `tnoimod=1` only when a noise analysis is requested
  (`b4set.c`); the Verilog-A creates them whenever `rdsmod != 0 ||
  tnoimod == 1`, and with `rdsmod=0` their series conductance is zero: the
  channel floats between two internal nodes and the device carries no
  current. `vx.va.NGSPICE_BSIM4_FORCE` (`tnoimod=0`, passed as `force=` by
  `ngspice_bsim4_models`) restores ngspice's DC/AC/transient model (noise
  analysis is not supported anyway), and compiling such a card without it
  warns.
* **BSIM4 version.** Both PDKs' cards say `version = 4.5`, and ngspice
  evaluates them with its BSIM4.5.0 code (`BSIM4v5`), while the Verilog-A is
  BSIM4.8.0. The two versions differ slightly (up to 0.04% in sky130 and 0.13%
  in gf180mcu drain currents here, largest in subthreshold). Rewriting the
  cards to `version = 4.8` for ngspice (the benchmark does this on a copy)
  makes the two simulators agree to round-off; the remaining difference to
  ngspice as distributed is the BSIM4 version, not the compiler.

## Differentiability audit

`voltax.va.audit` and `benchmarks/bsim4_differentiability.py` audit the
compiled model for gradient-based use.

**Static report.** The compiler records every conditional, equality branch,
clamp (`if (x < b) x = b`), `min`/`max`/`abs` and limiter it evaluates, with
its source line, macro and dependencies:

```python
from voltax.va import audit

rep = audit.static_report(bsim4, card)   # all real params + temperature traced
print(rep.summary())
```

With everything traced (868 inputs), BSIM4 on the FreePDK45 card has 99
bias-dependent branch sites and 114 parameter-dependent ones; 320 more
(model selectors, `$param_given`) fold away. In the default build (model
parameters static, W/L traced; ngspice defaults as in the benchmark) 66
bias-dependent sites remain as `where`
merges. The full table, with source lines, is generated as
`benchmarks/figures/bsim4_static_audit.md`. Kinds: `if` (general traced
conditional), `eq` (`if (x == c)`: a single point), `clamp`
(`if (x < b) x = b`), `case`, `min`/`max`/`abs` (none are live in BSIM4's
bias path: the ones in the source sit in noise code, which dead-code
elimination removes), and integer rounding. The report also counts traced
operations that *can* produce NaN/inf (bias-dependent divisions, `sqrt`,
`ln`, `exp`): about 230 sites, all inside guarded branches or well-defined
by construction.

**Empirical probes.** `audit.DeviceProbe` evaluates the compiled model as an
explicit function of bias and parameters (no circuit) and sweeps one
variable at a time. It is compiled with instrumentation, so every switch of
an active branch condition along the sweep is located by bisection to
~1e-15 and the value and slope jumps across it are measured; equality
branches hit exactly on the grid are checked for AD errors at the point:

```python
import numpy as np

intrinsic = {**card, "rbodymod": 0, "rgatemod": 0}   # no internal nodes
probe = audit.DeviceProbe(bsim4, intrinsic, instance={"w": 0.5e-6, "l": 50e-9})
sweep = probe.sweep("d", np.linspace(-0.1, 0.1, 21), {"g": 0.6})
for event in sweep.events:
    if event.kinds():
        print(f"Vds={event.x:+.1e}", sorted(event.kinds()),
              [line for line, _ in event.sites])
```

`benchmarks/bsim4_differentiability.py` runs this on fine sweeps of Vds
through 0, Vgs from -0.6 to 1.5 V, Vbs from -1.5 V into forward bias, the
Gummel-symmetry path (Vd = -Vs = x), and parameter sweeps (VTH0, U0, TOXE,
K1, VSAT, NFACTOR, RDSW, ETA0, DVT0, VOFF, MJS, W, L, temperature), checking
all eight terminal currents and charges. Findings for BSIM4 4.8:

| finding | where (bsim4.va) | effect |
|---|---|---|
| no NaN/inf in values, first or second AD derivatives on any sweep, including exactly Vds = 0, forward-biased junctions and the parameter ranges | all guarded branches | the double-where guard holds |
| AD vs central finite differences | everywhere away from the points below | agree to ≤ 2e-6 |
| `if (Vds == 0.0) Vdseff = 0.0;` and three similar patches | 7400, 9209, 9275, 9315 | **plain AD gives gds = 0 at exactly Vds = 0** (100% error; 30-90% in the charge derivatives). ngspice's hand-written Jacobian keeps the general derivative there. Fixed by `equality="limit"` |
| source/drain mode swap `if (vds >= 0.0)` | 6473, 6809, 9393, 9468 | Id is C1 at Vds = 0, but the gate and bulk currents and all four charges have slope jumps of 75-200%: Cgd, Cgs and the gate-leakage conductances are discontinuous at Vds = 0. On the Gummel path Ig, Qg and Ib are V-shaped (the slope changes sign) |
| channel-length modulation enabled by `diffVds > 1.0e-10` | 7622 | a 1e-10 relative jump in Id at Vds ≈ ±2 nV |
| impact-ionization cutoff `diffVds > beta0 / EXP_THRESHOLD` | 7901 | 9e-11 relative jump in Ib, 170% slope jump (of a ~fA current) at Vgs = 0.47 V, Vds = 1 V |
| on the Gummel path (Vd = -Vs = x) | mode swap | Id is C1 but its second derivative changes sign at x = 0: BSIM4 is not Gummel-symmetric beyond first order (see `benchmarks/figures/bsim4_audit_vds.png`) |
| `if (T != Tnom)` temperature scaling, `if (MJS == 0.5)` junction charge | 3551, 9280 ... | parameter-level equality branches hit exactly at the card's values (the default 27 °C equals TNOM exactly). Empirically benign here: AD at T = TNOM agrees with the one-sided slopes to < 1e-4 |
| junction diodes (exponential + linear extension), `DEXP` limiters, Vbseff and Vdseff smoothing, clamps | 6187-6460, 6490, 7403, 8268 ... | crossed by the sweeps, all C1 to working precision |

Parameter sweeps (at saturation, subthreshold and linear bias, all eight
outputs) found no NaN/inf and AD = FD to ≤ 1e-4, except where a sensitivity
is very weak (VSAT, RDSW, U0 in subthreshold: ≤ 3e-2). U0, TOXE, K1, VSAT,
NFACTOR, RDSW (20-400 Ω), DVT0, VOFF, W and the temperature (233-398 K) are
C1 over their ranges. Three parameter-space kinks remain:

* **VTH0**: the clamp `if (here_BSIM4vtfbphi2 < 0.0)` (line 4981) bites at
  VTH0 ≈ 0.35 V for this card (below its 0.41 V); dId/dVTH0 jumps by 28%
  (saturation) to 99% (linear region) there. An optimizer moving VTH0 across
  it sees a gradient discontinuity.
* **L**: gate-tunneling branches (lines 8275, 8519) switch at L < 33 nm
  (Leff of a few nm, outside any sensible range for this card).
* **ETA0** in subthreshold: the impact-ionization cutoff (line 7901).

Right at Vds = 0 and on the parameter boundaries above, prefer one-sided
reasoning: the derivative is that of one branch.

## Smooth mode and equality branches

Two compile options trade exact Verilog-A semantics for better derivatives;
values are unchanged by the first and changed by at most `w ln 2` by the
second:

* `equality="limit"` (default `"value"`): at `if (x == c)` branches on traced
  values, keep the special-case value but differentiate the general branch,
  the way SPICE's hand-written Jacobians do. This removes the AD point errors
  above (BSIM4: gds and the charge derivatives at exactly Vds = 0). It
  cannot help if the general branch is itself singular at the point
  (then the guard falls back to the special case's derivative).
* `smooth=w` (or `{line: w}`): replace traced `min`, `max`, `abs` and
  recognized clamps `if (x < b) x = b` with softplus-based surrogates of
  width `w` (error at most `w ln 2` in the clamped variable). For BSIM4 this
  only touches the two live bias clamps (`Vdseff > Vds` at line 7403,
  `Voxacc < 0` at 8268), and it is harmful: `Vdseff = min(Vdseff, Vds)` sits
  at its bound by design near Vds = 0, so the surrogate shifts it by up to
  `w ln 2` and creates channel current at Vds = 0:

| mode | max \|ΔId\| / max \|Id\| | Id at Vds = 0, Vgs = 1 V | switch breaks (C0/C1) | AD point errors | Newton iterations from z = 0 (mean / max) | Newton failures |
|---|---|---|---|---|---|---|
| exact (`equality="value"`) | 0 | -6.6e-10 A (gate leakage) | 15 | 4 | 8.0 / 11 | 0 |
| `equality="limit"` | 2e-16 | -6.6e-10 A | 15 | 0 | 7.9 / 10 | 0 |
| limit + `smooth=1e-3` | 6e-3 | -1.3e-6 A | 8 | 0 | 8.0 / 10 | 0 |
| limit + `smooth=1e-2` | 0.13 | -1.3e-5 A | 8 | 0 | 7.9 / 10 | 0 |

(Over the Vds and Vgs sweeps of the audit; Newton counts for 22 DC solves of
an inverter and a 3-inverter chain, all from the all-zero state, using
Voltax's damped Newton.) BSIM4 is already written with smooth `Vgsteff`,
`Vdseff` and `Vbseff` functions, so Newton gains nothing from smoothing, and
even the AD point errors at the all-zero initial state do not hurt
convergence. Recommendation: use `equality="limit"` when gradients at
exactly Vds = 0 matter; leave `smooth` off for BSIM4 (it is meant for
hand-written or simpler models with hard `min`/`max`/clamps), or restrict it
to chosen lines with `smooth={line: width}`.

The breaks that matter for BSIM4, the mode swap at Vds = 0, are genuine
model properties (not clamps), and no compile-time option should paper over
them: they are where the model's charge and gate-current formulations differ
between the forward and reverse modes.

**End to end.** The delay of a 3-inverter BSIM4 chain (full card: gate
resistor, substrate network, gate tunneling; transient on a 0.25 ps grid,
`vx.measure.delay`) differentiated with the adjoint method agrees with
central finite differences to 3e-6 for a transistor width and 4e-7 for the
NMOS VTH0, in both equality modes.

## Limitations

* No `idt`, `ddx`, Laplace/z-domain filters, `transition`, arrays, or
  noise analysis (noise contributions are dropped); `analysis()` only for
  `"noise"` (returns 0).
* Topology must not depend on bias: `V(a,b) <+ 0` (node collapse) only under
  static conditions.
* Only BSIM4 4.8 is available; cards with an older `version` are evaluated
  with 4.8 equations (ngspice uses its older code, see
  [PDK model decks](#pdk-model-decks-sky130-gf180mcu)). Bin selection across
  cards happens in the netlist parser, so a device's bin cannot change
  while W/L is optimized; within a card binning is smooth in 1/L, 1/W.
* Compile time: the BSIM4 Jacobian takes ~5 s of XLA compilation per device
  type, so a first DC call takes ~20-25 s and a transient 45-75 s on a shared
  4-core CPU (later calls are fast: the 2487-step chain transient then runs
  in ~4-7 s, vs ~0.2-3 s for ngspice).
* `bsim4.va` is licensed CC BY-NC 4.0 and therefore not bundled (see
  [Getting BSIM4](#getting-bsim4-licence) and `voltax/va/models/NOTICE`).
