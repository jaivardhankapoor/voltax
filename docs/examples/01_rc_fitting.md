# 01. Fitting an RC ladder to a measured step response

The simplest useful thing a differentiable simulator does: recover component
values from waveforms. We simulate a two-stage RC low-pass filter, treat its
step response as a "measurement", start from values that are 50% off, and
let a gradient-based optimizer find the true R1, R2, C1, C2.

The loss is the mean squared error between simulated and measured node
voltages,

$$ L(\theta) = \frac{1}{T} \sum_{k,\,n} \big(v_n(t_k; \theta)
   - v_n^\text{meas}(t_k)\big)^2, \qquad \theta = (\log R_1, \log R_2,
   \log C_1, \log C_2), $$

and `jax.grad` differentiates straight through `vx.transient`. Each implicit
time step is differentiated with the adjoint (implicit function theorem), so
the backward pass costs one linear solve per step, independent of how many
Newton iterations the forward pass needed. Optimizing in log-space keeps the
values positive and puts ohms and nanofarads on the same scale.

Identifiability matters: the transfer function to `out` alone has only three
free coefficients (DC gain and two poles), so four parameters cannot be
recovered from `v(out)` by itself. Observing the middle node `n1` as well
pins all four down.

What to look at: the loss falls to round-off level within a few dozen L-BFGS
iterations and the fitted values match the truth; the figure overlays the
initial, fitted and measured waveforms.

Run it: `uv run python examples/01_rc_fitting.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/01_rc_fitting.py"
```
