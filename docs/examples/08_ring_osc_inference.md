# 08. Inferring transistor parameters from a ring oscillator with NUTS

A ring oscillator is the classic process monitor: its frequency and waveform
shape depend on how strong the NMOS and PMOS devices came out of the fab.
Here we turn that around and ask for the *posterior distribution* of the
process transconductances $k_n, k_p$ given a noisy measured waveform.

Model, for the voltage at one stage sampled at times $t_k$:

$$ \log k_{n,p} \sim \mathcal{N}(\log k^\text{nom}_{n,p},\ 0.3^2), \qquad
   v^\text{meas}_k \sim \mathcal{N}\big(v(t_k; k_n, k_p),\ \sigma^2\big). $$

The No-U-Turn Sampler (blackjax) needs $\nabla_\theta \log p$, which voltax
provides through `vx.transient` by implicit differentiation. Both parameters
are *shared* leaves: every transistor references the same `MOSProcess`, so
one scalar `log_kp` moves all five NMOS (or PMOS) devices at once and its
gradient sums their contributions.

A ring has no stable DC operating point to start from, so the transient
starts from an explicit, asymmetric initial state built with `circuit.state`.
We observe only about three periods: over many periods a small frequency
error becomes a full phase slip and the likelihood turns multimodal.

What to look at: the posterior mean lands near the true values with a few
percent uncertainty, and the scatter plot shows the $k_n$-$k_p$ correlation
(a stronger NMOS can partly compensate a weaker PMOS).

Run it: `uv run python examples/08_ring_osc_inference.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/08_ring_osc_inference.py"
```
