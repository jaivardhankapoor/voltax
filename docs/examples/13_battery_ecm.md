# 13. Battery equivalent-circuit model: identifiability and energy prediction

A battery management system (BMS) cannot see inside a cell. It sees the
terminal voltage while a known current flows, and from that it must infer a
model good enough to predict what matters: available power, voltage sag, and
how much energy turns into heat. The workhorse model is the two-RC Thevenin
equivalent circuit (ECM):

    ocv --[R0]-- n1 --[R1 || C1]-- n2 --[R2 || C2]-- term --> load I(t)
     |                                                 |
    Voc                                               gnd

* $V_\text{oc}$: open-circuit voltage (held at 3.7 V here; a state-of-charge
  dependent $V_\text{oc}(\text{SOC})$ is a `Function` signal or a nonlinear
  element away),
* $R_0$: ohmic resistance (electrolyte, contacts): the instantaneous IR jump,
* $R_1 \| C_1$: charge transfer and double layer, $\tau_1 = R_1 C_1 \approx$
  seconds,
* $R_2 \| C_2$: solid-state diffusion, $\tau_2 = R_2 C_2 \approx$ tens of
  seconds.

The load is an ordinary `CurrentSource` from `term` to ground carrying a
pulse profile like a hybrid pulse power characterization (HPPC) test:
discharge and charge pulses separated by rests. With the SPICE convention
(`b.isource(p, n, I)` pushes $I$ from $p$ through the source into $n$), a
positive $I$ draws current out of the + terminal, i.e. a discharge. For a
current $I(t)$ the terminal voltage is

$$ v_\text{term}(t) = V_\text{oc} - R_0 I(t) - u_1(t) - u_2(t), \qquad
   \dot u_k = \frac{I(t)}{C_k} - \frac{u_k}{R_k C_k}. $$

We simulate this circuit with true parameters, add 2 mV sensor noise,
sample at 2 Hz like a BMS would, and fit $\theta = \log(R_0, R_1, R_2, C_1,
C_2)$ by minimizing

$$ L(\theta) = \frac{1}{N\sigma^2} \sum_k \big(v_\text{term}(t_k; \theta)
   - v^\text{meas}_k\big)^2 $$

with L-BFGS through `vx.transient`, from many random initializations at
once (`jax.vmap` over the whole optimizer).

**Identifiability.** The model is symmetric under swapping the two RC
branches: $(R_1, C_1) \leftrightarrow (R_2, C_2)$ gives exactly the same
terminal voltage. So the data determine the parameters only up to this
relabeling, and different initializations land in either mode. Sorting the
branches by time constant ("fast" and "slow") removes the ambiguity, after
which all five parameters are recovered to about a percent: the pulse test
excites both time scales, so nothing else is degenerate.

**Physical prediction.** The heat dissipated in each resistor is

$$ E_{R} = \int_0^T \frac{v_R(t)^2}{R}\, dt, $$

which `vx.measure.energy(circuit, sol, "R0")` computes from the solved
waveforms. A label-switched fit puts the fast-branch heat into "R2", so the
per-resistor bars disagree with the truth, yet the *total* heat, the quantity
a thermal model needs, is predicted to a fraction of a percent by every fit.
We also check `measure.energy` against $R_0 \int I^2 dt$ (the full load
current flows through $R_0$) and the energy balance

$$ E_\text{delivered by }V_\text{oc} = \sum_R E_R + \Delta E_{C_1}
   + \Delta E_{C_2} + E_\text{load}. $$

What to look at: (a) every fit (blue: branches in the true order, red:
swapped) tracks the noisy data to the noise level, $\chi^2/N \approx 1$;
(b) the raw errors of $R_1, R_2, C_1, C_2$ are tens to hundreds of percent
for swapped fits, while $R_0$, $R_0 + R_1 + R_2$ and the parameters relabeled
fast/slow sit within a percent for every fit; (c) the per-resistor heat bars
of swapped fits trade places, but the total heat agrees with the truth for
every fit.

Run it: `uv run python examples/13_battery_ecm.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/13_battery_ecm.py"
```
