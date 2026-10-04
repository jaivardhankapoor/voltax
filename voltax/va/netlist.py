"""Netlist hook: map ``.model ... level=N`` cards to compiled Verilog-A models.

    circuit = vx.parse_netlist(text, models=vx.va.level_models(bsim4, level=54))

`level_models` returns ``{"nmos:54": factory, "pmos:54": factory}`` for the
parser's ``models=`` hook, which calls ``factory.from_netlist(spec)`` with a
`voltax.netlist.DeviceSpec` for every matching device.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

from ..element import Element
from .model import VAModel


class NetlistFactory:
    """Builds devices of `model` from netlist model cards (one element class
    per card, so devices sharing a card fuse into one group)."""

    def __init__(self, model: VAModel, **element_kwargs: Any):
        self.model = model
        self.element_kwargs = element_kwargs
        self._warned: set[str] = set()

    def __call__(self, card_name: str, kind: str, params: Mapping[str, float],
                 nodes: tuple[str, ...], instance: Mapping[str, float]) -> Element:
        kind = kind.lower()
        if kind not in ("nmos", "pmos"):
            raise ValueError(f"model {card_name!r}: Verilog-A MOSFET cards must be "
                             f"nmos or pmos, got {kind!r}")
        card = {k: v for k, v in params.items() if k != "level"}
        if "level" in self.model.params:
            card["level"] = params.get("level")
        kwargs = {"ranges": "warn", **self.element_kwargs}
        cls = self.model.element(card, polarity=kind[0], name=card_name, **kwargs)
        n_ports = len(self.model.module.ports)
        if len(nodes) != n_ports:
            raise ValueError(f"{self.model.name} has {n_ports} terminals, "
                             f"got {len(nodes)} nodes")
        return cls(tuple(nodes), **dict(instance))

    def from_netlist(self, spec: Any) -> Element:
        """Build one device from a `voltax.netlist.DeviceSpec` (``models=`` hook).

        Instance parameters the Verilog-A module does not declare are dropped
        with a single warning per name; ``m`` (multiplier) is always passed.
        """
        known = self.model.instance_params
        instance = {}
        for key, value in spec.instance.items():
            if key in known or key == "m":
                instance[key] = value
            elif key not in self._warned:
                self._warned.add(key)
                warnings.warn(f"{self.model.name}: instance parameter {key!r} is "
                              "not declared by the Verilog-A module; ignored",
                              stacklevel=2)
        return self(spec.model.name, spec.model.type, spec.model.params,
                    tuple(spec.nodes), instance)


def level_models(model: VAModel, level: int = 54, **element_kwargs: Any
                 ) -> dict[str, NetlistFactory]:
    """``{"nmos:<level>": f, "pmos:<level>": f}`` for `voltax.parse_netlist`'s
    ``models=`` hook.

    `element_kwargs` (``differentiable``, ``smooth``, ``temperature``,
    ``defaults``, ``force``, ...) are passed to `VAModel.element` for every
    card. For ngspice-compatible BSIM4 use `ngspice_bsim4_models`.
    """
    factory = NetlistFactory(model, **element_kwargs)
    return {f"nmos:{int(level)}": factory, f"pmos:{int(level)}": factory}


def ngspice_bsim4_models(**element_kwargs: Any) -> dict[str, NetlistFactory]:
    """``models=`` hook mapping ``level=54`` cards to BSIM4 (fetched on first use),
    configured to reproduce ngspice's built-in BSIM4: its physical constants
    (`NGSPICE_BSIM4_OVERRIDES`), its parameter defaults
    (`NGSPICE_BSIM4_DEFAULTS`) and its analysis semantics
    (`NGSPICE_BSIM4_FORCE`)."""
    from .model import (
        NGSPICE_BSIM4_DEFAULTS,
        NGSPICE_BSIM4_FORCE,
        NGSPICE_BSIM4_OVERRIDES,
        load,
    )

    kwargs = {"defaults": NGSPICE_BSIM4_DEFAULTS, "force": NGSPICE_BSIM4_FORCE,
              **element_kwargs}
    return level_models(load("bsim4", overrides=NGSPICE_BSIM4_OVERRIDES), 54,
                        **kwargs)
