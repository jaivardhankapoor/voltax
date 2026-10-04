# 09. Gradient-free SMC vs. gradient-based NUTS on an RC filter

Bayesian inference over circuit parameters can be run with or without
gradients. This example compares the two on the smallest interesting
problem: an RC low-pass whose step response is measured with noise.

$$ \log R, \log C \sim \mathcal{N}(\log R_0, 0.7^2) \times
   \mathcal{N}(\log C_0, 0.7^2), \qquad
   v^\text{meas}_k \sim \mathcal{N}\big(1 - e^{-t_k / RC},\ \sigma^2\big). $$

The step response depends on $R$ and $C$ only through $\tau = RC$, so the
posterior is a long thin ridge along the hyperbola $RC = \tau$, bounded only
by the prior. That makes it a good stress test:

* **Tempered SMC with random-walk Metropolis** moves a cloud of particles
  from the prior to the posterior; it needs only log-density *values*.
* **NUTS** follows Hamiltonian trajectories along the ridge using the
  gradient $\nabla \log p$, which voltax supplies by differentiating
  `vx.transient`.

What to look at: both methods recover $\tau$ tightly while $R$ and $C$
individually stay as uncertain as the prior allows; the timing and the
number of log-density evaluations show the cost of each approach.

Run it: `uv run python examples/09_rc_smc_inference.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/09_rc_smc_inference.py"
```
