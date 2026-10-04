# 02. A CMOS inverter chain from a SPICE netlist

Voltax reads ordinary SPICE netlists, so existing designs can be simulated and
differentiated without rewriting them in Python. Here a three-stage CMOS
inverter chain is described as text, parsed into a `Circuit`, and analysed
three ways: a DC transfer curve, a transient pulse response, and the gradient
of the propagation delay with respect to every transistor width.

The transistors use the EKV model, which is smooth from weak to strong
inversion. The propagation delay is measured where the output crosses
$V_{DD}/2$; we locate the crossing sample with a (non-differentiable) search
and then interpolate linearly between the two neighbouring samples,

$$ t_{50} = t_k + (t_{k+1} - t_k)\,
   \frac{V_{DD}/2 - v_k}{v_{k+1} - v_k}, $$

which *is* differentiable in the waveform values, so `jax.grad` gives
$\partial t_{pd} / \partial \log W$ for all six devices in one backward pass.

What to look at: the transfer curve switches near mid-rail; the delay
gradients are negative (wider = faster) and largest for the devices that
drive the critical edges.

Run it: `uv run python examples/02_netlist_cmos_inverter.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/02_netlist_cmos_inverter.py"
```
