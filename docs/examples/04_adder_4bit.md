# 04. A 200-transistor 4-bit ripple-carry adder from library gates

Real digital blocks are built hierarchically: transistors make gates, gates
make full adders, full adders make an adder. `voltax.library.cmos` provides
the standard static-CMOS gates as functions that add devices to a
`CircuitBuilder`, and `CircuitBuilder.scope` keeps the internal nets of each
instance apart. At build time all 200 MOSFETs are fused into two vectorized
groups (NMOS and PMOS), so the device physics is a handful of array ops.

We check the logic exhaustively with DC operating points, mapping over input
words, and then run a transient on the worst-case input, where a carry must
ripple through all four stages:

$$ 1111_2 + 0000_2 + c_\text{in}: \quad c_\text{in}: 0 \to 1
   \;\Rightarrow\; s = 0000_2,\ c_\text{out} = 1 . $$

Inputs are driven by one named group of voltage sources (`group="inputs"`),
so `with_dc` can set all nine input voltages from a bit vector inside a
JAX transformation.

What to look at: every checked input combination produces the right sum;
the transient shows the sum bits flipping one after another as the carry
ripples, and the printed carry propagation delay.

Run it: `uv run python examples/04_adder_4bit.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/04_adder_4bit.py"
```
