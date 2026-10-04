# 03. Designing an active filter by gradient descent on its Bode plot

Filter design is usually done with tables: pick a prototype (Butterworth,
Chebyshev, ...), look up the stage Q factors, and solve for component values.
With a differentiable AC analysis we can instead *fit* the component values
to a desired magnitude response directly, including non-idealities such as
the op-amp's finite gain-bandwidth product that the tables ignore.

The circuit is a 4th-order low-pass made of two unity-gain Sallen-Key stages
built around a behavioral `vx.OpAmp`. The target is a Butterworth response
with corner $f_c = 10$ kHz,

$$ |H(f)|^2 = \frac{1}{1 + (f/f_c)^{8}}, \qquad
   L = \frac{1}{F} \sum_f \big(20\log_{10}|H_\text{sim}(f)| - 20\log_{10}
   |H(f)|\big)^2 . $$

`vx.ac` linearizes the circuit around its DC operating point and solves
$(G + j 2\pi f C)\,z = -b$ for all frequencies at once; it is differentiable,
so `jax.grad` of the dB error gives the gradient with respect to all four
capacitors. The resistors are fixed at 10 kOhm: scaling every R up and every
C down by the same factor leaves the response unchanged, so fixing the
impedance level makes the solution unique.

What to look at: the capacitors converge close to the textbook equal-resistor
Sallen-Key design (C1 = 2Q/(wR), C2 = 1/(2QwR) with Q = 0.541 and 1.307),
but not exactly onto it: the optimized design also corrects for the op-amp's
finite bandwidth and so tracks the target better than the textbook values.
The figure shows the Bode plot before and after.

Run it: `uv run python examples/03_opamp_filter_design.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/03_opamp_filter_design.py"
```
