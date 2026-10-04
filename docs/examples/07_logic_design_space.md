# 07. Searching every way to wire two transistors, then sizing the winner

What logic can you build from one PMOS and one NMOS? We put both transistors
on a "breadboard": each of their six pins (drain, gate, source) can be wired
to one of the nets {vdd, a, b, y, gnd}. That is $5^6 = 15\,625$ circuits,
each to be simulated for all four input combinations: 62 500 DC operating
points. Because a circuit is a pytree and the wiring is just an array of
switch conductances, `jax.vmap` evaluates them all as one batched program.

A wiring realizes a Boolean function if, for every input, the output is a
clean logic level ($y < 0.2\,V_{DD}$ or $y > 0.8\,V_{DD}$). The output is
loaded by 100 kOhm to mid-supply, so a floating output reads as an invalid
$V_{DD}/2$ rather than a lucky leakage level.

The search rediscovers the two classic two-transistor circuits: the CMOS
inverter and the transmission gate. We then turn from discrete search to
gradients and size the inverter. Its switching threshold $V_M$ (where
$v_\text{out} = v_\text{in}$) is the DC solution of an inverter whose output
is tied back to its input, so $\partial V_M / \partial \log W_p$ comes from
implicit differentiation of a single DC solve, and a few Newton steps on
$V_M(W_p) = V_{DD}/2$ give the width for symmetric noise margins.

What to look at: the table of realizable functions (no NAND/NOR/XOR, as
expected with two transistors), the four equivalent inverter wirings
(drain/source swaps), and the PMOS width that centres $V_M$.
In fast mode only input `a` is on the breadboard ($4^6 = 4096$ wirings).

Run it: `uv run python examples/07_logic_design_space.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/07_logic_design_space.py"
```
