# The circuit equations

Voltax solves the same equations as SPICE, written in the charge-oriented
form used by modern simulators. Understanding them makes everything else
(analyses, gradients, custom devices) straightforward.

## State

The unknown is a single vector

\[
z = \begin{bmatrix} v \\ x \end{bmatrix},
\]

where \(v\) are the voltages of the non-ground nodes and \(x\) are the
**internal unknowns** of the devices: the branch current of every voltage
source and inductor, the output current of an ideal op-amp, the pole state
of a behavioral op-amp, and so on. Each device class declares how many it
needs (`n_internal`). `circuit.layout.state_names()` lists them all.

## The DAE

Kirchhoff's current law at every node, plus each device's internal equations,
form a differential-algebraic equation

\[
\frac{d}{dt}\, q(z) + f(z, t) = 0 .
\]

- \(f(z, t)\) collects **resistive** terms: currents that depend on voltages
  directly (resistors, diodes, transistor channels, sources), and algebraic
  constraints such as \(v_p - v_n - V(t) = 0\) for a voltage source.
- \(q(z)\) collects **reactive** terms: charges on capacitors and transistor
  gates, and fluxes \(L\,i\) of inductors.

`Circuit.f(z, t)` and `Circuit.q(z)` evaluate the two halves. Every analysis
in Voltax is a few lines on top of them.

!!! note "Why charges and not capacitances?"
    Writing the reactive part as \(dq/dt\), rather than \(C(v)\,dv/dt\),
    guarantees charge conservation for nonlinear capacitors such as MOSFET
    gates. The capacitance matrix is still available as \(C = \partial q /
    \partial z\), which is how AC analysis gets it.

## How devices contribute

Each device only knows about its own terminals. For a group of \(N\) devices
with \(T\) terminals and \(K\) internal unknowns each, the circuit gathers

- `v`: the \((N, T)\) terminal voltages, and
- `x`: the \((N, K)\) internal unknowns,

and calls the device's two methods:

| method | returns | meaning |
|---|---|---|
| `currents(v, x, t)` | `I` \((N,T)\), `F` \((N,K)\) | current flowing *into the device* at each terminal; residual of each internal equation |
| `charges(v, x)` | `Q` \((N,T)\), `P` \((N,K)\) | charge stored at each terminal; the differentiated part of each internal equation |

The circuit adds `I` and `Q` into the KCL rows of the terminal nodes, and
`F` and `P` into the device's own internal rows. Ground (index `-1`) is
handled by appending a zero to \(v\) and dropping the corresponding row, so
device code never special-cases it.

For example, the inductor's internal equation is \(L\,di/dt + r_s i -
(v_p - v_n) = 0\). It returns `I = (+i, -i)`, `F = r_s i - (v_p - v_n)` and
`P = L i`. That is the whole device.

## Discretization

Transient analysis replaces \(dq/dt\) with a finite difference on the time
grid you provide. With step \(h = t_{n+1} - t_n\):

\[
\text{backward Euler:}\quad
\frac{q(z_{n+1}) - q(z_n)}{h} + f(z_{n+1}, t_{n+1}) = 0
\]

\[
\text{trapezoidal:}\quad
\frac{q(z_{n+1}) - q(z_n)}{h} + \tfrac12\left[f(z_{n+1}, t_{n+1}) +
f(z_n, t_n)\right] = 0
\]

Backward Euler (default) is first-order and strongly damped, which makes it
robust for stiff digital circuits. Trapezoidal is second-order and preserves
oscillations, which makes it better for LC circuits and accuracy studies.
Each step is a nonlinear root-finding problem solved by Newton's method (see
[Analyses](analyses.md)).

## DC and AC

- **DC**: drop the reactive part, \(f(z, t) = 0\): capacitors open, inductors
  short.
- **AC**: linearize at the DC point,
  \(G = \partial f/\partial z\), \(C = \partial q/\partial z\), and solve
  \((G + j\omega C)\, \tilde z = -b\) at each frequency, where \(b\) is the
  small-signal stimulus of the sources (their `ac` magnitude and phase).
  The matrices come from `jax.jacfwd`, so every device gets AC analysis
  for free.

## Sign conventions

Voltax follows SPICE:

- A voltage source's current \(i\) flows from \(p\) to \(n\) *through the
  source*, so a battery delivering power reports a negative current.
- A current source `isource(p, n, I)` pushes \(I\) from \(p\) through itself
  to \(n\), i.e. *into* node \(n\).
- Device currents `I` are positive when flowing from the node into the device.
