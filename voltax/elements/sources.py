"""Independent voltage and current sources."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..element import Element, _is_single, through, two_terminal
from ..signals import Constant, Signal, as_signal


def _phasor(mag: Array, phase_deg: Array) -> Array:
    return mag * jnp.exp(1j * jnp.deg2rad(phase_deg))


class _Source(Element):
    """Shared helpers for sources whose waveform is a `Signal`."""

    value: Signal

    def with_dc(self, values: Any) -> "_Source":
        """Copy with every device's waveform replaced by DC `values`.

        Typical use: drive a group of input sources from data inside `vmap`,
        ``circuit.replace("inputs", src.with_dc(x))``.
        """
        return eqx.tree_at(lambda e: e.value, self, Constant(self._per_device(values)))

    def set(self, index: int, **values: Any) -> "_Source":
        """Like `Element.set`, plus waveform parameters.

        * ``value=`` sets a DC value. A non-`Constant` waveform is replaced by
          it when the group holds a single device.
        * Any parameter of the waveform, e.g. ``freq=`` for a `Sine` or
          ``delay=`` for a `Pulse`, is set for device `index`.
        """
        wave_fields = {f.name for f in dataclasses.fields(self.value)}
        for key in [k for k in values if k in wave_fields and k != "value"]:
            new = getattr(self.value, key).at[index].set(values.pop(key))
            self = eqx.tree_at(lambda e, k=key: getattr(e.value, k), self, new)
        if "value" in values:
            dc = values.pop("value")
            if isinstance(self.value, Constant):
                new = self.value.value.at[index].set(dc)
                self = eqx.tree_at(lambda e: e.value.value, self, new)
            elif self.size == 1:
                self = self.with_dc(dc)
            else:
                raise TypeError("set(value=...) on a batched non-DC source; "
                                "replace `.value` with eqx.tree_at instead")
        return super().set(index, **values) if values else self


class VoltageSource(_Source):
    """Independent voltage source, ``v_p - v_n = value(t)``.

    Internal unknown: the current ``i`` flowing ``p -> n`` *through the
    source* (SPICE convention: a supply delivering power has ``i < 0``).

    Args:
        nodes: ``(p, n)`` or a list of them.
        value: A number or a `Signal` (per-device leaves when batched).
        ac: Small-signal magnitude for AC analysis.
        ac_phase: Small-signal phase in degrees.
    """

    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("i",)
    value: Signal
    ac: Array
    ac_phase: Array

    def __init__(self, nodes: Any, value: Signal | Any = 0.0, ac: Any = 0.0,
                 ac_phase: Any = 0.0):
        self.nodes = self._devices(nodes)
        self.value = self._per_device_tree(as_signal(value), _is_single(nodes))
        self.ac = self._per_device(ac)
        self.ac_phase = self._per_device(ac_phase)

    def currents(self, v, x, t):
        i = x[..., 0]
        return through(i), (two_terminal(v) - self.value(t))[..., None]

    def ac_stimulus(self):
        return None, -_phasor(self.ac, self.ac_phase)[..., None]


class CurrentSource(_Source):
    """Independent current source driving ``value(t)`` from ``p`` to ``n``
    through the source (SPICE convention), i.e. out of node ``n``.
    """

    terminals = ("p", "n")
    value: Signal
    ac: Array
    ac_phase: Array

    def __init__(self, nodes: Any, value: Signal | Any = 0.0, ac: Any = 0.0,
                 ac_phase: Any = 0.0):
        self.nodes = self._devices(nodes)
        self.value = self._per_device_tree(as_signal(value), _is_single(nodes))
        self.ac = self._per_device(ac)
        self.ac_phase = self._per_device(ac_phase)

    def currents(self, v, x, t):
        i = jnp.broadcast_to(self.value(t), (self.size,))
        return through(i), None

    def ac_stimulus(self):
        return through(_phasor(self.ac, self.ac_phase)), None
