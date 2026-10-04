# 12. Energy per operation vs. supply voltage: power-aware sizing of CMOS logic

Lowering the supply voltage is the most effective way to save energy in
digital logic, and it is paid for in speed. Designers trade the two by
sweeping $V_{DD}$ and resizing transistors; with a differentiable simulator
both knobs can be tuned at once by gradient descent on the *measured* energy
and delay of a transient simulation.

The circuit is a chain of four CMOS inverters driving a 50 fF load. The first
stage is unit-sized (it stands for the gate that drives the chain); the other
three may be resized. The input toggles once up and once down per clock
period $T$, and the energy of one such operation is what the supply delivers,

$$ E_{op} = \int_0^T V_{DD}\, i_{DD}(t)\, dt
   \;\approx\; C_{eff}\, V_{DD}^2 , $$

because every node that switches is charged once from the supply (half of
$C V^2$ is burnt in the PMOS on the way up, the other half in the NMOS on the
way down). Short-circuit current while both devices conduct and the
voltage-dependence of the transistor capacitances make the measured $C_{eff}$
drift slightly with $V_{DD}$. The gate delay follows the alpha-power law

$$ t_{pd} \propto \frac{C\, V_{DD}}{(V_{DD} - V_{th})^{\alpha}} , $$

so energy falls quadratically as $V_{DD}$ drops while delay blows up near
$V_{th}$. The energy-delay product $\mathrm{EDP} = E_{op}\, t_{pd}$ has a
minimum in between ($V_{DD} = 3V_{th}/(3-\alpha)$ for the alpha-power law).

`vx.measure.energy` integrates the power absorbed by the source named "Vdd"
(negative: it delivers energy), `vx.measure.delay` measures the 50% input to
output delay of both edges, and both are differentiable. We use the
trapezoidal rule on a time grid refined at the input edges by `vx.time_grid`:
backward Euler's first-order error overestimates the energy by several
percent on such a grid.

Finally we minimise $E_{op}$ over $(\log V_{DD}, \log s_2, \log s_3,
\log s_4)$ (the per-stage width multipliers) subject to $t_{pd} \le
t_{spec}$, with an augmented Lagrangian and L-BFGS. The gradients are checked
against finite differences first.

What to look at: (a) energy follows $C_{eff} V_{DD}^2$ closely while delay
grows steeply at low $V_{DD}$; (b) the EDP minimum; (c) the optimizer walks
from a fast, power-hungry start onto the delay constraint and then slides
along it to lower energy, ending below the best plain $V_{DD}$ scaling of the
unit-sized chain: tapering the stages buys back the speed that a lower
supply costs.

Run it: `uv run python examples/12_energy_delay_optimization.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/12_energy_delay_optimization.py"
```
