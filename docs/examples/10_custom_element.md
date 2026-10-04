# 10. Writing your own device: a memristor with an internal state

Every device in voltax, from the resistor to the EKV MOSFET, is an `Element`
subclass with two short methods. This example writes a new one from scratch:
the HP Labs memristor, a resistor whose resistance depends on how much charge
has flowed through it. Nothing else in the library has to change; the new
device works in every analysis and is differentiable like the built-ins.

Voltax circuits are the charge-oriented DAE $\frac{d}{dt}q(z) + f(z, t) = 0$.
An element contributes, per device, terminal currents $I$ and charges $Q$, and
it may own *internal unknowns* $x$ with their own equations
$\frac{d}{dt}P(x) + F(v, x) = 0$. The linear-drift memristor has one internal
state, the normalized width $w \in [0, 1]$ of its doped region:

$$ i = \frac{v}{R_\text{on} w + R_\text{off} (1 - w)}, \qquad
   \frac{dw}{dt} = k\, i\, \big(1 - (2w - 1)^{2p}\big), $$

with $k = \mu_v R_\text{on} / D^2$ and a Joglekar window that keeps $w$ in
range. So `currents` returns $I = (i, -i)$ and $F = -k\,i\,f(w)$, and
`charges` returns $P = w$.

What to look at: under a sine drive the I-V curve is a hysteresis loop
pinched at the origin (the memristor fingerprint) that shrinks as the
frequency rises; the gradient of the loop area with respect to the device
parameters matches finite differences. For a *stateless* nonlinear device
you don't need a subclass at all: use `vx.NonlinearResistor` or
`vx.NonlinearCapacitor` with any `jnp` function.

Run it: `uv run python examples/10_custom_element.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/10_custom_element.py"
```
