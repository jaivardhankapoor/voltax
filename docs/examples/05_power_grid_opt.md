# 05. Sizing 760 wires of a power grid with one adjoint solve per step

A chip's power grid is a resistive mesh: supply pads at the edges, current
drawn by logic everywhere in between. The voltage lost along the wires ("IR
drop") slows the logic down, and the fix is wider metal, which costs area.
Deciding *which* of hundreds of wires to widen is a large optimization
problem.

With finite differences, each gradient would need one extra DC solve per
wire (761 solves for a 20x20 grid). Voltax differentiates the DC solution by
the adjoint method instead: for the operating point $F(v^*, g) = 0$,

$$ \frac{dL}{dg} = -\lambda^\top \frac{\partial F}{\partial g},
   \qquad \Big(\frac{\partial F}{\partial v}\Big)^{\!\top} \lambda
   = \frac{\partial L}{\partial v^*}, $$

so every gradient costs one extra (transposed) linear solve, however many
wires there are. The loss trades IR drop against total metal,
$L = \overline{(V_{DD} - v)^2} + \mu\, \overline{g}$, and each wire is a
`vx.Conductance` with a sigmoid parametrization, which bounds its
conductance between a minimum and a maximum manufacturable width.

What to look at: the optimized grid has a much smaller worst-case drop than a
uniform grid using the *same total metal*; the figure shows where the
optimizer put the metal (thick wires carry the load currents to the pads).

Run it: `uv run python examples/05_power_grid_opt.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/05_power_grid_opt.py"
```
