"""Trainable and user-defined two-terminal elements.

* `Conductance`: a linear conductance whose parametrization you choose
  (log, softplus, sigmoid-bounded, or raw). Use it when the conductances are
  the *learned weights* of a model, e.g. analog neural networks or topology
  search, where log-space is not the right geometry.
* `NonlinearResistor` / `NonlinearCapacitor`: any ``i = fn(v, params)`` or
  ``q = fn(v, params)`` you can write in `jnp`, with differentiable params.
* `BehavioralSource`: a current or voltage source whose value is any
  function of control-node voltages, sensed branch currents and time (the
  SPICE ``B`` element; the netlist parser compiles ``V={...}`` / ``I={...}``
  expressions into one).
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..element import Element, _is_single, through, two_terminal

Transform = Literal["log", "softplus", "sigmoid", "linear"]


class Conductance(Element):
    """Linear conductance ``i = g(theta) (v_p - v_n)`` with a chosen map
    from the raw parameter ``theta`` to ``g``:

    ============  ==============================================
    ``log``       ``g = exp(theta)``
    ``softplus``  ``g = g_min + softplus(theta)``
    ``sigmoid``   ``g = g_min + (g_max - g_min) sigmoid(theta)``
    ``linear``    ``g = theta`` (unconstrained; may go negative)
    ============  ==============================================

    The sigmoid map is the usual relaxation for "does this edge exist?"
    topology search: ``theta -> -inf`` removes the edge (``g -> g_min``).

    Args:
        nodes: ``(p, n)`` or a list of them.
        g: Initial conductance(s) in siemens; converted to ``theta``.
        transform: Parametrization (static).
        g_min, g_max: Bounds used by ``softplus`` / ``sigmoid`` (static).
    """

    terminals = ("p", "n")
    theta: Array
    transform: Transform = eqx.field(static=True)
    g_min: float = eqx.field(static=True)
    g_max: float = eqx.field(static=True)

    def __init__(self, nodes: Any, g: Any, transform: Transform = "log",
                 g_min: float = 0.0, g_max: float = 1.0):
        if transform not in ("log", "softplus", "sigmoid", "linear"):
            raise ValueError(f"unknown transform {transform!r}")
        self.nodes = self._devices(nodes)
        self.transform, self.g_min, self.g_max = transform, g_min, g_max
        self.theta = self._per_device(self.to_theta(g))

    def to_theta(self, g: Any) -> Array:
        """Inverse map ``g -> theta``."""
        g = jnp.asarray(g, dtype=float)
        if self.transform == "log":
            return jnp.log(g)
        if self.transform == "softplus":
            y = g - self.g_min
            return y + jnp.log(-jnp.expm1(-y))  # inverse softplus
        if self.transform == "sigmoid":
            u = (g - self.g_min) / (self.g_max - self.g_min)
            return jnp.log(u) - jnp.log1p(-u)
        return g

    @property
    def g(self) -> Array:
        if self.transform == "log":
            return jnp.exp(self.theta)
        if self.transform == "softplus":
            return self.g_min + jax.nn.softplus(self.theta)
        if self.transform == "sigmoid":
            return self.g_min + (self.g_max - self.g_min) * jax.nn.sigmoid(self.theta)
        return self.theta

    @property
    def r(self) -> Array:
        return 1.0 / self.g

    def set(self, index: int, **values: Any) -> "Conductance":
        if "g" in values:
            values["theta"] = self.to_theta(values.pop("g"))
        if "r" in values:
            values["theta"] = self.to_theta(1.0 / jnp.asarray(values.pop("r")))
        return super().set(index, **values)

    def currents(self, v, x, t):
        return through(self.g * two_terminal(v)), None


class NonlinearResistor(Element):
    """Two-terminal element with current ``i = fn(v, params)``, ``v = v_p - v_n``.

    `fn` is static (must be hashable, e.g. a module-level function); `params`
    is any pytree of arrays with a leading device axis (added automatically
    for a single device) and is differentiable.

    Example:
        >>> def tanh_i(v, p):
        ...     return p["i_max"] * jnp.tanh(v / p["v0"])
        >>> NonlinearResistor(("a", "0"), tanh_i, {"i_max": 1e-3, "v0": 0.1})
    """

    terminals = ("p", "n")
    params: Any
    fn: Callable[[Array, Any], Array] = eqx.field(static=True)

    def __init__(self, nodes: Any, fn: Callable[[Array, Any], Array],
                 params: Any = None):
        self.nodes = self._devices(nodes)
        self.fn = fn
        self.params = self._per_device_tree(params, _is_single(nodes))

    def currents(self, v, x, t):
        return through(self.fn(two_terminal(v), self.params)), None


class NonlinearCapacitor(Element):
    """Two-terminal element with charge ``q = fn(v, params)`` (see
    `NonlinearResistor` for conventions). Charge-based, so it conserves
    charge for any `fn` (e.g. varactors, ``q = c0 v + c1 v^2 / 2``).
    """

    terminals = ("p", "n")
    params: Any
    fn: Callable[[Array, Any], Array] = eqx.field(static=True)

    def __init__(self, nodes: Any, fn: Callable[[Array, Any], Array],
                 params: Any = None):
        self.nodes = self._devices(nodes)
        self.fn = fn
        self.params = self._per_device_tree(params, _is_single(nodes))

    def currents(self, v, x, t):
        return None, None

    def charges(self, v, x):
        return through(self.fn(two_terminal(v), self.params)), None


class BehavioralSource(Element):
    """Multi-terminal behavioral source (SPICE ``B``, ``E``/``G`` with
    ``value=``): ``value = fn(v_ctrl, i_sense, t, params)``.

    Terminals are ``(p, n, c_0 ... c_{k-1}, s_0p, s_0n, ...)``:

    * ``(p, n)``: the output port.
    * ``c_j``: control nodes; ``fn`` receives their voltages ``v_ctrl``
      ``(N, k)`` (ground is not a terminal: use a constant 0).
    * ``(s_jp, s_jn)``: `n_sense` zero-volt sense ports, each in series with
      a branch whose current ``fn`` reads as ``i_sense[..., j]`` (flowing
      ``s_jp -> s_jn``), as for `CCCS`.

    With ``output="i"`` the source drives current ``value`` from `p` through
    itself to `n` (like `CurrentSource`); with ``output="v"`` it enforces
    ``v_p - v_n = value`` and has a branch-current unknown ``i``.

    `fn` is static (hashable, e.g. a module-level function or a
    `voltax.netlist.CompiledExpr`); `params` is a dict of differentiable
    values with a leading device axis (added for a single device).

    Example:
        >>> def gain(vc, isense, t, p):  # i = gm * v(c)^2
        ...     return p["gm"] * vc[..., 0] ** 2
        >>> BehavioralSource(("out", "0", "c"), gain, {"gm": 1e-3})
    """

    params: Any
    fn: Callable[..., Array] = eqx.field(static=True)
    output: Literal["i", "v"] = eqx.field(static=True)
    n_sense: int = eqx.field(static=True)

    def __init__(self, nodes: Any, fn: Callable[..., Array], params: Any = None,
                 output: Literal["i", "v"] = "i", n_sense: int = 0):
        if output not in ("i", "v"):
            raise ValueError(f"output must be 'i' or 'v', got {output!r}")
        self.nodes = self._devices(nodes)
        if len(self.nodes[0]) < 2 + 2 * n_sense:
            raise ValueError("BehavioralSource needs (p, n, controls..., "
                             f"{n_sense} sense port pairs)")
        self.fn, self.output, self.n_sense = fn, output, n_sense
        self.params = self._per_device_tree({} if params is None else params,
                                            _is_single(nodes))

    @property
    def n_ctrl(self) -> int:
        return len(self.nodes[0]) - 2 - 2 * self.n_sense

    @property
    def terminals(self) -> tuple[str, ...]:  # type: ignore[override]
        sense = tuple(f"s{j}{s}" for j in range(self.n_sense) for s in "pn")
        return ("p", "n", *(f"c{j}" for j in range(self.n_ctrl)), *sense)

    @property
    def n_internal(self) -> int:  # type: ignore[override]
        return self.n_sense + (self.output == "v")

    @property
    def internal_names(self) -> tuple[str, ...]:  # type: ignore[override]
        sense = tuple(f"i_sense{j}" for j in range(self.n_sense))
        return (("i",) if self.output == "v" else ()) + sense

    def value(self, v: Array, x: Array, t: Array) -> Array:
        """The expression value per device, from terminal voltages `v` and
        internal unknowns `x`."""
        k = self.n_ctrl
        off = int(self.output == "v")
        vc = v[..., 2:2 + k]
        isense = x[..., off:off + self.n_sense]
        out = self.fn(vc, isense, t, self.params)
        return jnp.broadcast_to(out, v.shape[:-1])

    def currents(self, v, x, t):
        value = self.value(v, x, t)
        off = int(self.output == "v")
        i_out = x[..., 0] if self.output == "v" else value
        isense = x[..., off:off + self.n_sense]
        zeros = jnp.zeros(v.shape[:-1] + (self.n_ctrl,), v.dtype)
        sense_i = jnp.stack([isense, -isense], axis=-1).reshape(
            v.shape[:-1] + (2 * self.n_sense,))
        I = jnp.concatenate([jnp.stack([i_out, -i_out], -1), zeros, sense_i], -1)
        base = 2 + self.n_ctrl
        sense_eq = v[..., base::2] - v[..., base + 1::2]  # zero-volt ports
        if self.output == "v":
            F = jnp.concatenate([(v[..., 0] - v[..., 1] - value)[..., None],
                                 sense_eq], -1)
        else:
            F = sense_eq
        return I, F
