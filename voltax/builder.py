"""`CircuitBuilder`: describe a circuit device by device, with named nodes.

    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Step(0.0, 1.0))
    b.resistor("in", "out", 1e3, name="R1")
    b.capacitor("out", "0", 1e-9)
    circuit = b.build()

Nodes are strings (numbers are converted with ``str``); ``"0"``, ``"gnd"``,
``"GND"`` and ``"ground"`` are ground. At `build`, devices of the same type and
static configuration are fused into one vectorized element group, so a circuit
with 10 000 transistors evaluates its physics in a handful of array ops.

Any `Element` instance can be added with `add`; the typed helpers
(``resistor``, ``nmos``, ...) are thin wrappers around it.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Iterator, Literal

from . import elements as el
from .circuit import _GROUND, Circuit
from .element import Element, concatenate, group_key
from .signals import Constant, Signal

_PREFIX = {
    el.Conductance: "R", el.NonlinearResistor: "B", el.NonlinearCapacitor: "B",
    el.BehavioralSource: "B",
    el.Resistor: "R", el.Capacitor: "C", el.Inductor: "L", el.Transformer: "K",
    el.VoltageSource: "V", el.CurrentSource: "I", el.Diode: "D", el.BJT: "Q",
    el.EKVMOSFET: "M", el.Level1MOSFET: "M", el.VCVS: "E", el.VCCS: "G",
    el.CCCS: "F", el.CCVS: "H", el.IdealOpAmp: "U", el.OpAmp: "U", el.Switch: "S",
}


def _default_group_name(element: Element) -> str:
    """Class name, plus polarity for transistors (``EKVMOSFET_n``, ``BJT_pnp``)
    or the waveform for non-DC sources (``VoltageSource_Pulse``)."""
    name = type(element).__name__
    polarity = getattr(element, "polarity", None)
    if isinstance(polarity, str):
        return f"{name}_{polarity}"
    wave = getattr(element, "value", None)
    if isinstance(wave, Signal) and not isinstance(wave, Constant):
        return f"{name}_{type(wave).__name__}"
    return name


class CircuitBuilder:
    """Incrementally assemble a `Circuit`.

    Args:
        nmos, pmos: Default `MOSProcess` for `nmos` / `pmos` devices. All
            transistors built from the same process share its parameters.
        mos_model: MOSFET model class used by `nmos` / `pmos`.
    """

    def __init__(self, nmos: el.MOSProcess | None = None,
                 pmos: el.MOSProcess | None = None,
                 mos_model: type[el.EKVMOSFET] | type[el.Level1MOSFET] = el.EKVMOSFET):
        self.nmos_process = nmos or el.MOSProcess.nmos()
        self.pmos_process = pmos or el.MOSProcess.pmos()
        self.mos_model = mos_model
        self._devices: list[tuple[str, Element, str | None]] = []
        self._names: set[str] = set()
        self._nodes: dict[str, None] = {}
        self._scope: list[str] = []
        self._counters: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------- nodes

    def node(self, name: str | None = None) -> str:
        """Create a node local to the current `scope` and return its name.

        Without `name`, a fresh unique node is created (for internal nets of
        a subcircuit).
        """
        if name is None:
            self._counters["_node"] += 1
            name = f"_{self._counters['_node']}"
        full = ".".join([*self._scope, name])
        self._nodes.setdefault(full, None)
        return full

    @contextlib.contextmanager
    def scope(self, prefix: str) -> Iterator[None]:
        """Prefix device names and `node` names created inside the block.

        Use it to instantiate a subcircuit several times without name clashes.
        """
        self._scope.append(prefix)
        try:
            yield
        finally:
            self._scope.pop()

    @property
    def nodes(self) -> list[str]:
        """Non-ground node names in index order."""
        return list(self._nodes)

    # ----------------------------------------------------------- devices

    def add(self, element: Element, name: str | None = None,
            group: str | None = None) -> str:
        """Add a single-device `element` whose nodes are names.

        Args:
            element: E.g. ``vx.Resistor(("a", "b"), 1e3)``.
            name: Device name (default: SPICE letter + counter, e.g. ``R3``).
            group: Force the device into a named group, e.g. to keep a set of
                devices separate for optimization. Devices in one group must
                be fusable (same type and static configuration).

        Returns:
            The (scoped) device name.
        """
        if element.size != 1:
            raise ValueError("add() takes single-device elements")
        if name is None:
            prefix = _PREFIX.get(type(element), type(element).__name__)
            self._counters[prefix] += 1
            name = f"{prefix}{self._counters[prefix]}"
        name = ".".join([*self._scope, name])
        if name in self._names:
            raise ValueError(f"duplicate device name {name!r}")
        self._names.add(name)
        terminals = tuple(str(n) for n in element.nodes[0])
        for n in terminals:
            if n not in _GROUND:
                self._nodes.setdefault(n, None)
        self._devices.append((name, element.with_nodes((terminals,)), group))
        return name

    def resistor(self, p, n, r, name=None, group=None) -> str:
        """Resistor of `r` ohms between `p` and `n`."""
        return self.add(el.Resistor((p, n), r), name, group)

    def conductance(self, p, n, g, transform="log", g_min=0.0, g_max=1.0,
                    name=None, group=None) -> str:
        """Trainable conductance (see `Conductance` for `transform`)."""
        dev = el.Conductance((p, n), g, transform, g_min, g_max)
        return self.add(dev, name, group)

    def capacitor(self, p, n, c, name=None, group=None) -> str:
        """Capacitor of `c` farads."""
        return self.add(el.Capacitor((p, n), c), name, group)

    def inductor(self, p, n, l, rs=0.0, name=None, group=None) -> str:
        """Inductor of `l` henries with series resistance `rs`."""
        return self.add(el.Inductor((p, n), l, rs), name, group)

    def transformer(self, p1, n1, p2, n2, l1, l2, k=1.0, name=None, group=None):
        """Coupled inductors (see `Transformer`)."""
        return self.add(el.Transformer((p1, n1, p2, n2), l1, l2, k), name, group)

    def vsource(self, p, n, value: Signal | float = 0.0, ac=0.0, ac_phase=0.0,
                name=None, group=None) -> str:
        """Voltage source ``v_p - v_n = value(t)``."""
        return self.add(el.VoltageSource((p, n), value, ac, ac_phase), name, group)

    def isource(self, p, n, value: Signal | float = 0.0, ac=0.0, ac_phase=0.0,
                name=None, group=None) -> str:
        """Current source pushing `value(t)` from `p` through itself to `n`."""
        return self.add(el.CurrentSource((p, n), value, ac, ac_phase), name, group)

    def diode(self, a, k, name=None, group=None, **params) -> str:
        """Diode from anode `a` to cathode `k` (see `Diode` for params)."""
        return self.add(el.Diode((a, k), **params), name, group)

    def bjt(self, c, b, e, polarity: Literal["npn", "pnp"] = "npn", name=None,
            group=None, **params) -> str:
        """Bipolar transistor (see `BJT` for params)."""
        return self.add(el.BJT((c, b, e), polarity, **params), name, group)

    def nmos(self, d, g, s, b=None, w=1e-6, l=0.13e-6, name=None,
             group=None, process=None, dvth=0.0) -> str:
        """NMOS transistor; bulk defaults to the source. `dvth` is a
        per-device threshold offset (mismatch)."""
        dev = self.mos_model((d, g, s, s if b is None else b), w, l,
                             process or self.nmos_process, "n", dvth)
        return self.add(dev, name, group)

    def pmos(self, d, g, s, b=None, w=1e-6, l=0.13e-6, name=None,
             group=None, process=None, dvth=0.0) -> str:
        """PMOS transistor; bulk defaults to the source."""
        dev = self.mos_model((d, g, s, s if b is None else b), w, l,
                             process or self.pmos_process, "p", dvth)
        return self.add(dev, name, group)

    def vcvs(self, p, n, cp, cn, gain, name=None, group=None) -> str:
        """``v(p, n) = gain * v(cp, cn)``."""
        return self.add(el.VCVS((p, n, cp, cn), gain), name, group)

    def vccs(self, p, n, cp, cn, gm, name=None, group=None) -> str:
        """Current ``gm * v(cp, cn)`` from `p` through the source to `n`."""
        return self.add(el.VCCS((p, n, cp, cn), gm), name, group)

    def cccs(self, p, n, sp, sn, gain, name=None, group=None) -> str:
        """Current ``gain * i_sense`` where ``i_sense`` flows ``sp -> sn``."""
        return self.add(el.CCCS((p, n, sp, sn), gain), name, group)

    def ccvs(self, p, n, sp, sn, r, name=None, group=None) -> str:
        """``v(p, n) = r * i_sense`` where ``i_sense`` flows ``sp -> sn``."""
        return self.add(el.CCVS((p, n, sp, sn), r), name, group)

    def opamp(self, inp, inn, out, name=None, group=None, **params) -> str:
        """Behavioral op-amp (see `OpAmp` for params)."""
        return self.add(el.OpAmp((inp, inn, out), **params), name, group)

    def ideal_opamp(self, inp, inn, out, name=None, group=None) -> str:
        """Ideal (nullor) op-amp."""
        return self.add(el.IdealOpAmp((inp, inn, out)), name, group)

    def switch(self, p, n, cp, cn, name=None, group=None, **params) -> str:
        """Voltage-controlled switch (see `Switch` for params)."""
        return self.add(el.Switch((p, n, cp, cn), **params), name, group)

    # ------------------------------------------------------------- build

    def build(self) -> Circuit:
        """Resolve node names, fuse devices into groups, return a `Circuit`."""
        index = {name: i for i, name in enumerate(self._nodes)}
        resolve = lambda n: -1 if n in _GROUND else index[n]  # noqa: E731

        buckets: dict[Any, list[tuple[str, Element]]] = {}
        for name, dev, group in self._devices:
            dev = dev.with_nodes((tuple(resolve(n) for n in dev.nodes[0]),))
            key = group if group is not None else group_key(dev)
            if group is not None and buckets.get(key):
                if group_key(buckets[key][0][1]) != group_key(dev):
                    raise ValueError(f"devices in group {group!r} cannot be fused")
            buckets.setdefault(key, []).append((name, dev))

        groups: dict[str, Element] = {}
        device_names: dict[str, tuple[str, int]] = {}
        used: dict[str, int] = defaultdict(int)
        # explicit groups first so automatic names never collide with them
        order = sorted(buckets, key=lambda k: not isinstance(k, str))
        for key in order:
            members = buckets[key]
            if isinstance(key, str):
                gname = key
            else:
                base = _default_group_name(members[0][1])
                gname = base
                while gname in groups or gname in buckets:
                    used[base] += 1
                    gname = f"{base}_{used[base] + 1}"
            groups[gname] = concatenate([dev for _, dev in members])
            for i, (name, _) in enumerate(members):
                device_names[name] = (gname, i)
        return Circuit(groups, tuple(self._nodes), device_names)
