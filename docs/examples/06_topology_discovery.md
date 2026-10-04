# 06. Discovering a filter topology from a "circuit soup"

Gradient descent can choose *which components exist*, not just their values.
We start from a "soup": every pair of nodes among {in, out, n2, n3, gnd} is
joined by a resistor and a capacitor (18 candidate components), and ask for a
low-pass response with real poles at 20 kHz and 200 kHz. A sparsity penalty
pushes every component towards "absent" (zero conductance or capacitance)
unless the frequency response needs it, so what survives is a circuit you can
read off and build.

The loss combines an AC-analysis fit in dB with a log-sum sparsity penalty,

$$ L = \overline{\big(20\log_{10}|H(f)| - H_\text{dB}^\star(f)\big)^2}
   + \lambda \sum_k \Big[\log\big(1 + \tfrac{g_k}{g_\text{ref}}\big)
   + \log\big(1 + \tfrac{c_k}{c_\text{ref}}\big)\Big]. $$

Unlike an L1 penalty, whose pull on $\log g$ vanishes as $g \to 0$, the
log-sum penalty pushes every unneeded component down at a constant rate in
log-space until it is negligible. Conductances use `vx.Conductance` with a
sigmoid map between `g_min` (open) and `g_max` (short), the standard
relaxation of a discrete on/off choice.

What to look at: of the 18 candidates only about four survive, forming the
textbook two-section RC ladder; pruning the rest leaves the Bode plot
unchanged.

Run it: `uv run python examples/06_topology_discovery.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/06_topology_discovery.py"
```
