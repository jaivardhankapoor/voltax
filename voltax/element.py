"""The `Element` base class: the single extension point for device models.

Voltax writes every circuit as the charge-oriented DAE used by SPICE,

    d/dt q(x) + f(x, t) = 0,        x = [node voltages; internal unknowns],

and every device contributes to `f` (currents / algebraic equations) and `q`
(charges / fluxes) through two *device-local* methods:

    currents(v, x, t) -> (I, F)       # resistive part
    charges(v, x)     -> (Q, P)       # reactive part (optional)

* `v`: ``(N, T)`` voltages at the element's ``T`` terminals, for ``N`` devices.
* `x`: ``(N, K)`` the element's own internal unknowns (branch currents, internal
  states). ``K = n_internal`` is declared per class.
* `I`, `Q`: ``(N, T)`` current / charge flowing *into the device* at each
  terminal (i.e. leaving the node).
* `F`, `P`: ``(N, K)`` residuals of the internal equations and their
  time-differentiated parts; the internal equations read ``dP/dt + F = 0``.

An element never sees global node indices. The `Circuit` gathers terminal
voltages, calls the element, and scatters results back, so writing a new
device model is just writing its physics.

One `Element` instance holds ``N`` devices of the same type (struct-of-arrays),
so the physics is vectorized over devices. Write element code with ``[..., k]``
or ``.T`` indexing and elementwise `jnp` ops, and shared (scalar) parameters
broadcast automatically.

Devices must be *independent*: device ``i``'s outputs may depend only on
device ``i``'s own ``v[i]`` and ``x[i]`` (parameters may be shared). The
solvers rely on this to build Jacobians from per-device blocks (see
`voltax.sparse`). An element whose devices interact (e.g. a matrix of mutual
inductances across devices) must set ``local = False``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, ClassVar, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

Node = int | str
"""A node reference: an integer index (``-1`` is ground) or a name."""


def _is_single(nodes: Sequence[Any]) -> bool:
    """`nodes` is one device's terminal tuple, e.g. ``("in", "out")``."""
    return len(nodes) > 0 and isinstance(nodes[0], (int, str, np.integer))


def positive(x: Any) -> Array:
    """`log(x)`, validating that `x` is positive. Used for log-space params."""
    x = jnp.asarray(x, dtype=float)
    if not isinstance(x, jax.core.Tracer) and bool(jnp.any(x <= 0)):
        raise ValueError(f"expected a positive value, got {x}")
    return jnp.log(x)


class Element(eqx.Module):
    """Base class for a vectorized group of ``N`` devices of one type.

    Subclasses declare:

    * ``terminals``: terminal names, e.g. ``("p", "n")``.
    * ``n_internal``: internal unknowns per device (default 0).
    * ``internal_names``: names of those unknowns, e.g. ``("i",)``.
    * ``shared``: names of fields that are shared by all devices (e.g. process
      parameters). They are not stacked when devices are grouped.
    * ``local``: ``True`` (default) if each device's outputs depend only on
      that device's own terminals and internals. Set ``False`` for elements
      that couple their devices; the whole group is then treated as one
      block (dense Jacobian over all its unknowns).

    and implement `currents` (and optionally `charges` and `ac_stimulus`).

    Every positive physical quantity is stored in log-space as ``log_<name>``
    with a matching property ``<name>``. Gradients and optimizers therefore act
    on log-values, which keeps parameters positive and well-scaled.
    """

    nodes: tuple[tuple[Node, ...], ...] = eqx.field(static=True)

    terminals: ClassVar[tuple[str, ...]] = ()
    n_internal: ClassVar[int] = 0
    internal_names: ClassVar[tuple[str, ...]] = ()
    shared: ClassVar[tuple[str, ...]] = ()
    local: ClassVar[bool] = True

    # ------------------------------------------------------------------ physics

    def currents(self, v: Array, x: Array, t: Array) -> tuple[Array, Array | None]:
        """Resistive contributions ``(I, F)``; see module docstring."""
        raise NotImplementedError

    def charges(self, v: Array, x: Array) -> tuple[Array | None, Array | None]:
        """Reactive contributions ``(Q, P)``. Default: none."""
        return None, None

    def ac_stimulus(self) -> tuple[Array | None, Array | None]:
        """Small-signal excitation, as an additive term in ``(I, F)``."""
        return None, None

    # -------------------------------------------------------------- structure

    @property
    def size(self) -> int:
        """Number of devices in this group."""
        return len(self.nodes)

    def node_array(self) -> np.ndarray:
        """``(N, T)`` integer node indices (requires resolved nodes)."""
        return np.asarray(self.nodes, dtype=np.int64).reshape(self.size, -1)

    def with_nodes(self, nodes: Sequence[Sequence[Node]]) -> "Element":
        """Copy of this element with different node references."""
        new = copy.copy(self)  # `nodes` is static, so `eqx.tree_at` can't set it
        object.__setattr__(new, "nodes", tuple(tuple(d) for d in nodes))
        return new

    # ---------------------------------------------------------- per-device I/O

    def get(self, field: str, index: int | slice = slice(None)) -> Array:
        """Physical value of `field` for device(s) `index`.

        ``get("r", 0)`` reads ``exp(log_r[0])`` if the element stores ``log_r``.
        """
        return jnp.asarray(getattr(self, field))[index]

    def set(self, index: int, **values: Any) -> "Element":
        """Return a copy with device `index`'s parameters replaced.

        Values are physical; log-space fields are handled automatically, so
        ``resistors.set(0, r=2e3)`` updates ``log_r[0]``.
        """
        fields = set(_dynamic_fields(self)) - set(self.shared)
        out = self
        for key, value in values.items():
            log_attr = f"log_{key.rstrip('_')}"
            if log_attr in fields:
                attr, value = log_attr, jnp.log(jnp.asarray(value, dtype=float))
            elif key in fields:
                attr = key
            else:
                settable = sorted(f.removeprefix("log_") for f in fields)
                raise AttributeError(f"{type(self).__name__} has no per-device "
                                     f"parameter {key!r}; settable: {settable}")
            new = getattr(out, attr).at[index].set(value)
            out = eqx.tree_at(lambda e, a=attr: getattr(e, a), out, new)
        return out

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _devices(nodes: Sequence[Any]) -> tuple[tuple[Node, ...], ...]:
        """Normalize single-device or batched `nodes` to a tuple of tuples."""
        if _is_single(nodes):
            return (tuple(nodes),)
        return tuple(tuple(d) for d in nodes)

    def _per_device(self, value: Any) -> Array:
        """Broadcast a scalar or ``(N,)`` value to a float ``(N,)`` array."""
        value = jnp.asarray(value, dtype=float)
        if value.ndim == 0:
            return jnp.broadcast_to(value, (self.size,))
        if value.shape != (self.size,):
            raise ValueError(
                f"expected a scalar or shape ({self.size},) value, got {value.shape}"
            )
        return value

    def _log_per_device(self, value: Any) -> Array:
        """`_per_device` in log-space (validated positive)."""
        return self._per_device(positive(value))

    def _per_device_tree(self, tree: Any, single: bool) -> Any:
        """Give every leaf of `tree` (e.g. a `Signal`) a leading device axis.

        Single-device input gains a new axis; batched input has its scalar
        leaves broadcast to ``(N,)``.
        """

        def fix(x: Any) -> Any:
            x = jnp.asarray(x, dtype=float)
            if single:
                return x[None]
            return jnp.broadcast_to(x, (self.size,)) if x.ndim == 0 else x

        return jax.tree.map(fix, tree)


def concatenate(elements: Sequence[Element]) -> Element:
    """Fuse several elements of the same type into one vectorized group.

    Per-device leaves are concatenated along axis 0; fields listed in
    ``shared`` are taken from the first element (callers must make sure they
    agree, see `group_key`).
    """
    first = elements[0]
    if len(elements) == 1:
        return first
    nodes = tuple(d for e in elements for d in e.nodes)

    def stacked(name: str) -> Any:
        if name in first.shared:
            return getattr(first, name)
        return jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *(getattr(e, name) for e in elements),
        )

    out = first.with_nodes(nodes)
    for f in _dynamic_fields(first):
        out = eqx.tree_at(lambda e, f=f: getattr(e, f), out, stacked(f))
    return out


def group_key(element: Element) -> tuple:
    """Hashable key: elements with equal keys can be fused by `concatenate`."""
    static = tuple(
        (f.name, getattr(element, f.name))
        for f in dataclasses.fields(element)
        if f.metadata.get("static", False) and f.name != "nodes"
    )
    shared = tuple(id(getattr(element, s)) for s in element.shared)

    def leaf_sig(x: Any) -> Any:
        return (jnp.shape(x)[1:], jnp.result_type(x)) if eqx.is_array(x) else x

    per_device = tuple(
        (name, jax.tree.structure(getattr(element, name)),
         tuple(leaf_sig(x) for x in jax.tree.leaves(getattr(element, name))))
        for name in _dynamic_fields(element)
        if name not in element.shared
    )
    return (type(element), static, shared, per_device)


def _dynamic_fields(element: Element) -> list[str]:
    """Names of the non-static (pytree) fields."""
    return [f.name for f in dataclasses.fields(element)
            if not f.metadata.get("static", False)]


def two_terminal(v: Array) -> Array:
    """Branch voltage ``v_p - v_n`` of a two-terminal element."""
    return v[..., 0] - v[..., 1]


def through(i: Array) -> Array:
    """Terminal currents ``(+i, -i)`` for a current `i` flowing ``p -> n``."""
    return jnp.stack([i, -i], axis=-1)
