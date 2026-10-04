"""`Circuit`: a set of element groups plus a static layout, viewed as a DAE.

A circuit defines the semi-explicit DAE

    d/dt q(z) + f(z, t) = 0,      z = [node voltages (n_nodes); internals],

where `internals` are the elements' internal unknowns (branch currents,
internal states) laid out group by group. `Circuit.f` and `Circuit.q` evaluate
the two halves; every analysis in `voltax.analysis` is built on them.

Circuits are Equinox modules: all parameters are pytree leaves, the topology is
static. Use `eqx.filter_grad`, `jax.vmap`, `eqx.tree_at`, etc. directly, or the
convenience methods `Circuit.get` / `Circuit.set` for per-device access.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .element import Element


@dataclass(frozen=True)
class Layout:
    """Static (hashable) map between names and state-vector indices."""

    node_names: tuple[str, ...]
    groups: tuple[tuple[str, int, int, int], ...]
    """``(group, n_devices, n_internal_per_device, offset_in_state)``."""
    devices: tuple[tuple[str, str, int], ...]
    """``(device_name, group, index_in_group)``."""
    internal_names: tuple[tuple[str, tuple[str, ...]], ...]
    """``(group, names_of_internal_unknowns)``."""

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def size(self) -> int:
        """Length of the state vector ``z``."""
        return self.n_nodes + sum(n * k for _, n, k, _ in self.groups)

    @cached_property
    def _node_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.node_names)}

    @cached_property
    def _device_index(self) -> dict[str, tuple[str, int]]:
        return {name: (g, i) for name, g, i in self.devices}

    @cached_property
    def _group_info(self) -> dict[str, tuple[int, int, int]]:
        return {g: (n, k, off) for g, n, k, off in self.groups}

    def node(self, name: str) -> int:
        """State index of node `name` (ground is ``-1``)."""
        if name in _GROUND:
            return -1
        try:
            return self._node_index[name]
        except KeyError:
            raise KeyError(f"no node named {name!r}") from None

    def device(self, name: str) -> tuple[str, int]:
        """``(group, index)`` of device `name`."""
        try:
            return self._device_index[name]
        except KeyError:
            raise KeyError(f"no device named {name!r}") from None

    def internal(self, device: str, which: str | None = None) -> int:
        """State index of internal unknown `which` of `device`."""
        group, idx = self.device(device)
        n, k, offset = self._group_info[group]
        names = dict(self.internal_names)[group]
        if k == 0:
            raise ValueError(f"device {device!r} has no internal unknowns")
        if which is None:
            which = names[0]
        if which not in names:
            raise KeyError(f"{device!r} has internals {names}, not {which!r}")
        return offset + idx * k + names.index(which)

    def group_internals(self, group: str) -> np.ndarray:
        """``(N, K)`` state indices of the internal unknowns of `group`."""
        n, k, offset = self._group_info[group]
        return np.arange(offset, offset + n * k).reshape(n, k)

    def state_names(self) -> list[str]:
        """Human-readable label of every entry of ``z``."""
        labels = [f"v({n})" for n in self.node_names]
        names = dict(self.internal_names)
        by_group: dict[str, list[str]] = {}
        for dev, g, _ in self.devices:
            by_group.setdefault(g, []).append(dev)
        for g, n, k, _ in self.groups:
            for dev in by_group.get(g, [f"{g}[{i}]" for i in range(n)]):
                labels += [f"{w}({dev})" for w in names[g][:k]]
        return labels


_GROUND = frozenset({"0", "gnd", "GND", "ground"})


class Circuit(eqx.Module):
    """A compiled circuit. Usually created by `CircuitBuilder.build` or
    `parse_netlist`, rarely by hand.

    Args:
        elements: ``{group_name: element}``; node references must be integer
            indices (``-1`` = ground).
        node_names: Names of nodes ``0 .. n_nodes-1``.
        device_names: Optional ``{device_name: (group, index)}``.
    """

    elements: dict[str, Element]
    layout: Layout = eqx.field(static=True)

    def __init__(
        self,
        elements: Mapping[str, Element],
        node_names: tuple[str, ...] | list[str] | int,
        device_names: Mapping[str, tuple[str, int]] | None = None,
    ):
        if isinstance(node_names, int):
            node_names = tuple(str(i + 1) for i in range(node_names))
        node_names = tuple(node_names)
        n_nodes = len(node_names)
        groups, internals = [], []
        offset = n_nodes
        for name in sorted(elements):  # jax flattens dicts in sorted order
            el = elements[name]
            nodes = el.node_array() if el.size else np.zeros((0, 0), int)
            if el.size and (nodes.shape[1] != len(el.terminals)):
                raise ValueError(f"group {name!r}: expected {len(el.terminals)} "
                                 f"terminals, got {nodes.shape[1]}")
            if nodes.size and (nodes.min() < -1 or nodes.max() >= n_nodes):
                raise ValueError(f"group {name!r} references a node outside "
                                 f"[-1, {n_nodes})")
            if len(el.internal_names) != el.n_internal:
                raise TypeError(f"{type(el).__name__}.internal_names must have "
                                f"n_internal={el.n_internal} entries")
            groups.append((name, el.size, el.n_internal, offset))
            internals.append((name, el.internal_names))
            offset += el.size * el.n_internal
        if device_names is None:
            device_names = {f"{g}[{i}]": (g, i) for g, n, _, _ in groups
                            for i in range(n)}
        devices = tuple((d, g, i) for d, (g, i) in device_names.items())
        self.elements = dict(elements)
        self.layout = Layout(node_names, tuple(groups), devices, tuple(internals))

    # ------------------------------------------------------------- the DAE

    @property
    def n_nodes(self) -> int:
        return self.layout.n_nodes

    @property
    def size(self) -> int:
        """Length of the state vector."""
        return self.layout.size

    def zeros(self) -> Array:
        """An all-zero state vector."""
        return jnp.zeros(self.size)

    def state(self, v: Mapping[str, Any] | None = None,
              x: Mapping[str, Any] | None = None,
              z: Array | None = None) -> Array:
        """A state vector with chosen entries, e.g. for an initial condition:
        ``transient(c, ts, ic=c.state(v={"n1": 1.2}, x={"M1.w": 0.1}))``.

        Args:
            v: ``{node: voltage}``.
            x: ``{"device.unknown": value}`` for internal unknowns (or just
                ``"device"`` for its first one, e.g. an inductor current).
            z: Base state (default zeros).
        """
        z = self.zeros() if z is None else jnp.asarray(z)
        for node, value in (v or {}).items():
            z = z.at[self.node(node)].set(value)
        for key, value in (x or {}).items():
            device, _, which = key.partition(".")
            z = z.at[self.layout.internal(device, which or None)].set(value)
        return z

    def f(self, z: Array, t: Array | float = 0.0) -> Array:
        """Resistive part ``f(z, t)`` of the DAE ``d/dt q(z) + f(z, t) = 0``."""
        return self._assemble(z, lambda el, v, x: el.currents(v, x, t))

    def q(self, z: Array) -> Array:
        """Reactive part ``q(z)`` (charges and fluxes)."""
        return self._assemble(z, lambda el, v, x: el.charges(v, x))

    def ac_stimulus(self) -> Array:
        """Complex small-signal excitation ``b`` added to ``f`` (AC analysis)."""
        z = jnp.zeros(self.size, dtype=complex)
        return self._assemble(z, lambda el, v, x: el.ac_stimulus(), complex)

    def _assemble(self, z: Array, fn: Callable, dtype: Any = None) -> Array:
        nv = self.n_nodes
        dtype = dtype or z.dtype
        v_ext = jnp.concatenate([z[:nv], jnp.zeros(1, z.dtype)])  # [-1] = ground
        nodes_out = jnp.zeros(nv + 1, dtype)
        internal_out = jnp.zeros(self.size - nv, dtype)
        for name, n, k, offset in self.layout.groups:
            if n == 0:
                continue
            el = self.elements[name]
            idx = el.node_array()
            internal_idx = np.arange(offset - nv, offset - nv + n * k).reshape(n, k)
            I, F = fn(el, v_ext[idx], z[nv + internal_idx])
            if I is not None:
                nodes_out = nodes_out.at[idx].add(jnp.broadcast_to(I, idx.shape))
            if F is not None and k:
                internal_out = internal_out.at[internal_idx].add(F)
        return jnp.concatenate([nodes_out[:nv], internal_out])

    # ------------------------------------------------------ named access

    def node(self, name: str) -> int:
        """State index of node `name`."""
        return self.layout.node(name)

    def device(self, name: str) -> tuple[Element, int]:
        """``(element_group, index)`` holding device `name`."""
        group, idx = self.layout.device(name)
        return self.elements[group], idx

    def devices(self, group: str | None = None) -> list[str]:
        """Device names, in index order within `group` (or all devices)."""
        if group is not None and group not in self.elements:
            raise KeyError(f"no group {group!r}; groups: {sorted(self.elements)}")
        found = [(g, i, d) for d, g, i in self.layout.devices
                 if group is None or g == group]
        return [d for _, _, d in sorted(found)]

    def get(self, device: str, param: str) -> Array:
        """Physical value of `param` for `device`, e.g. ``get("R1", "r")``."""
        el, idx = self.device(device)
        return el.get(param, idx)

    def set(self, device: str, **values: Any) -> "Circuit":
        """Copy with `device`'s parameters replaced (physical units).

        ``circuit.set("R1", r=2e3)``. Works under `jit`/`grad`/`vmap`.
        """
        group, idx = self.layout.device(device)
        new = self.elements[group].set(idx, **values)
        return eqx.tree_at(lambda c: c.elements[group], self, new)

    def replace(self, group: str, element: Element) -> "Circuit":
        """Copy with element group `group` replaced (same topology)."""
        return eqx.tree_at(lambda c: c.elements[group], self, element)

    def summary(self) -> str:
        """Human-readable overview of nodes, groups and state layout."""
        lines = [f"Circuit: {self.n_nodes} nodes, state size {self.size}"]
        for name, n, k, _ in self.layout.groups:
            el = self.elements[name]
            extra = f", {k} internal/device" if k else ""
            lines.append(f"  {name}: {n} x {type(el).__name__}{extra}")
        return "\n".join(lines)
