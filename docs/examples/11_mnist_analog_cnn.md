# 11. Training an analog convolutional network made of resistors and diodes

An analog neural network computes with physics: weights are conductances,
summation is Kirchhoff's current law, and the nonlinearity comes from
devices. Here a small convolutional network for the 8x8 scikit-learn digits
is *a circuit*, and training means finding conductances such that the DC
operating point classifies the input.

Architecture (all conductances, ``g = softplus(theta)`` so they stay
positive):

* 64 input sources (pixel intensity in volts) and one -1 V bias source.
* A 3x3 convolution: each of the 36 hidden nodes is joined to its 3x3 input
  patch by 9 conductances. *Weight sharing* is just indexing: the 324
  conductances read their values from 9 kernel parameters.
* A diode from every hidden node to ground (a soft rectifier).
* A fully connected readout: 36 x 10 conductances into 10 output nodes,
  each loaded to ground.

At DC, node $j$ settles at the conductance-weighted average of its
neighbours, $v_j = \sum_k g_{jk} v_k / \sum_k g_{jk}$ (minus diode current),
so the network is a physically constrained, normalized neural net. The
logits are the output voltages, and the cross-entropy gradient with respect
to all 380 parameters comes from one adjoint solve per image.

What to look at: test accuracy (well above the 10% chance level even in fast
mode), the accuracy after snapping every resistor to the nearest E96
standard value, and the learned kernel / readout maps in the figure.

Run it: `uv run python examples/11_mnist_analog_cnn.py` (add `VOLTAX_FAST=1` for a quick run).

## Source

```python
--8<-- "examples/11_mnist_analog_cnn.py"
```
