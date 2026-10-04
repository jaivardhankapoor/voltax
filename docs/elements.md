# Device models

All models follow the conventions of [The circuit equations](concepts/formulation.md):
currents are positive *into* the device, \(v_{ab} = v_a - v_b\), and positive
parameters are stored in log-space (`log_<name>`) with a physical-valued
property `<name>`. Builder helpers are listed after each model; full
signatures are in the [API reference](api/elements.md).

## Passive

| model | terminals | equations | builder |
|---|---|---|---|
| `Resistor(r)` | p, n | \(i = v_{pn}/r\) | `b.resistor` |
| `Capacitor(c)` | p, n | \(q = c\,v_{pn}\) | `b.capacitor` |
| `Inductor(l, rs=0)` | p, n | \(l\,\dot\imath + r_s i = v_{pn}\); internal `i` | `b.inductor` |
| `Transformer(l1, l2, k)` | p1, n1, p2, n2 | \(\begin{bmatrix}v_1\\v_2\end{bmatrix} = \frac{d}{dt}\begin{bmatrix}l_1 & M\\ M & l_2\end{bmatrix}\begin{bmatrix}i_1\\i_2\end{bmatrix}\), \(M = k\sqrt{l_1 l_2}\) | `b.transformer` |

## Sources

| model | equations | builder |
|---|---|---|
| `VoltageSource(value, ac, ac_phase)` | \(v_{pn} = \text{value}(t)\); internal `i` (p→n through the source) | `b.vsource` |
| `CurrentSource(value, ac, ac_phase)` | pushes \(\text{value}(t)\) from p through the source to n | `b.isource` |

`value` is a number or a [signal](api/signals.md): `Constant`, `Step`,
`Pulse`, `Sine`, `PWL` (with SPICE `td=`/`r=` delay and repeat), `Exp`,
`SFFM`, `AM`, `Function(fn, params)`, or sums of these
(`Constant(1.0) + Sine(...)`). All signal parameters are differentiable.

## Controlled sources

Four-terminal: output port (p, n), control port (cp, cn). Current-controlled
sources measure the current through a built-in zero-volt sense port
cp → cn (in netlists, `F`/`H` elements are re-wired automatically so the sense
port is in series with the named voltage source).

| model | equation | builder |
|---|---|---|
| `VCVS(gain)` (E) | \(v_{pn} = \text{gain}\; v_{cp,cn}\) | `b.vcvs` |
| `VCCS(gm)` (G) | \(i_{p\to n} = g_m\, v_{cp,cn}\) | `b.vccs` |
| `CCCS(gain)` (F) | \(i_{p\to n} = \text{gain}\; i_{sense}\) | `b.cccs` |
| `CCVS(r)` (H) | \(v_{pn} = r\, i_{sense}\) | `b.ccvs` |

## Op-amps

| model | behaviour | builder |
|---|---|---|
| `IdealOpAmp` | nullor: forces \(v_+ = v_-\) with unlimited output current (needs negative feedback) | `b.ideal_opamp` |
| `OpAmp(a0, gbw, rout, vmin, vmax)` | single pole at \(\text{gbw}/a_0\): \(\tau \dot x + x = \text{clip}(a_0 v_d)\), output through \(r_{out}\); soft rails via `tanh` | `b.opamp` |

Build multi-pole or slew-limited op-amps as a subcircuit of these plus
controlled sources, or as a [custom element](concepts/custom-elements.md).

## Semiconductors

**Diode** `Diode(is_, n, cj, tt)`: \(i = I_s(e^{v/nV_T} - 1)\),
\(q = t_t\, i + c_j v\). Netlist models also accept series resistance `rs`.

**BJT** `BJT(polarity, is_, bf, br, vaf, tf, tr, cje, cjc)`: Ebers–Moll
transport model with Early effect,

\[
i_c = (i_f - i_r)\left(1 + \tfrac{v_{ce}}{V_{AF}}\right) - \frac{i_r}{\beta_R},\quad
i_b = \frac{i_f}{\beta_F} + \frac{i_r}{\beta_R},\quad
i_{f,r} = I_s\left(e^{v_{be,bc}/V_T} - 1\right),
\]

with charges \(q_{be} = t_f i_f + c_{je} v_{be}\), \(q_{bc} = t_r i_r + c_{jc} v_{bc}\).

Exponentials in both models continue linearly above \(40\,V_T\) so Newton
iterates stay finite.

### MOSFETs

Both MOSFET models take `(d, g, s, b)` terminals, per-device `w`, `l` and
threshold offset `dvth` (for mismatch studies), a static `polarity`
(`"n"`/`"p"`), and a shared `MOSProcess(kp, vth, n, lam, vt, gamma, phi, cox, cov)`.
PMOS devices use the same equations in a mirrored frame.

**`EKVMOSFET`** (default) is continuous from weak to strong inversion:

\[
v_p = \frac{v_{gb} - V_{th}}{n},\quad
i_{f,r} = \ln^2\!\left(1 + e^{(v_p - v_{sb,db})/2V_T}\right),\quad
I_{ds} = 2 n\, k_p \tfrac{W}{L} V_T^2 (i_f - i_r)(1 + \lambda |v_{ds}|).
\]

Its charges are the EKV charge-sheet expressions with Ward–Dutton
partitioning (charge-conserving), giving the textbook limits
\(C_{gs} \to \tfrac23 C_{ox}WL\) in saturation and
\(C_{gb} = \tfrac{n-1}{n} C_{ox}WL\) below threshold, plus overlap
capacitance \(C_{ov} W\).

**`Level1MOSFET`** is the Shichman–Hodges square law (SPICE level 1) with
body effect \(V_{th} + \gamma(\sqrt{\phi + v_{sb}} - \sqrt\phi)\) (the square
root is linearized for a forward-biased bulk, \(v_{sb} < 0\), as in SPICE). The
overdrive is smoothed with `softplus` of width `smooth` (default 20 mV) for
differentiability through cutoff; `smooth=0` is the exact SPICE model.
Charges are overlap only.

Use `m.ids(vd, vg, vs, vb)` to evaluate the drain current directly, e.g.
for I-V curves.

## Switches, trainable and behavioral elements

| model | behaviour | builder |
|---|---|---|
| `Switch(ron, roff, vth, vwidth)` | \(\log g\) moves from \(-\log r_{off}\) to \(-\log r_{on}\) along \(\sigma((v_c - v_{th})/v_{width})\) | `b.switch` |
| `Conductance(g, transform, g_min, g_max)` | \(i = g(\theta)\, v\) with `log`/`softplus`/`sigmoid`/`linear` maps (see [Gradients](concepts/gradients.md#parametrizations)) | `b.conductance` |
| `NonlinearResistor(fn, params)` | \(i = \mathrm{fn}(v, \text{params})\) | `b.add(...)` |
| `NonlinearCapacitor(fn, params)` | \(q = \mathrm{fn}(v, \text{params})\) | `b.add(...)` |
| `BehavioralSource(fn, params, output)` | terminals (p, n, controls..., sense ports...); \(\text{value} = \mathrm{fn}(v_{ctrl}, i_{sense}, t, \text{params})\) driven as a current (`output="i"`) or voltage (`"v"`, internal `i`). The netlist `B`, `E`/`G value=` element | `b.add(...)` |

## Library

`voltax.library.cmos` has static CMOS building blocks written as plain
functions over a builder: `inverter`, `nand`, `nor`, `and_`, `or_`, `xor`,
`full_adder`, `ripple_adder` and `ring_oscillator`.
