"""Emission: flattened `Elem` records -> `CircuitBuilder` calls.

Each element letter has one handler, ``_r``, ``_c``, .... Values go through
`Emitter.value`, so every number may be an expression in the instance's
parameter scope. Cross-element features are resolved up front:

* Current sensing (``F``/``H``, ``i(Vx)`` in behavioral expressions): a
  zero-volt sense port per (sensor, source) pair is inserted in series with
  the sensed voltage source, ``V: p -> V#0``, ``port 1: V#0 -> V#1``, ...,
  ``port k: -> n``.
* ``K`` merges two inductors into one `Transformer`.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .. import elements as el
from .. import signals
from ..builder import CircuitBuilder
from ..circuit import Circuit
from ..element import Element
from .expressions import AST, Compiled, compile_jax, parse_expression
from .lexer import Loc, NetlistError, split_kv, try_number, unquote, warn
from .models import (
    DeviceSpec,
    ModelCard,
    ModelFactory,
    bin_margin,
    call_hook,
    find_hook,
    select_bin,
)
from .preprocess import Elem, Flattener

K_BOLTZMANN_Q = 1.38064852e-23 / 1.6021766208e-19
"""Boltzmann constant over electron charge (V/K), ngspice's values."""

_MOS = {"level", "kp", "vto", "lambda", "n", "gamma", "phi", "tox", "cgso", "cgdo",
        "lmin", "lmax", "wmin", "wmax", "version", "tnom"}
MODEL_PARAMS: dict[str, set[str]] = {
    "nmos": _MOS,
    "pmos": _MOS,
    "d": {"level", "is", "n", "cjo", "cj0", "tt", "rs", "tnom"},
    "npn": {"level", "is", "bf", "br", "vaf", "va", "tf", "tr", "cje", "cjc",
            "tnom"},
    "pnp": {"level", "is", "bf", "br", "vaf", "va", "tf", "tr", "cje", "cjc",
            "tnom"},
    "sw": {"ron", "roff", "vt", "vh"},
    "r": {"rsh", "r", "res", "tnom", "defw", "narrow", "short"},
    "c": {"cj", "cap", "c", "tnom", "defw", "narrow", "short"},
}
"""Model-card parameters Voltax understands, by model kind."""


def _is_level1(model_cls) -> bool:
    """`model_cls` is `Level1MOSFET` or a ``functools.partial`` of it."""
    return getattr(model_cls, "func", model_cls) is el.Level1MOSFET


def _cls_name(obj: Any) -> str:
    obj = getattr(obj, "func", obj)
    return getattr(obj, "__name__", type(obj).__name__)


# =============================================================================
# Parse report
# =============================================================================


@dataclass
class ModelSummary:
    """How a ``.model`` card maps onto Voltax.

    `status` is ``"native"`` (a built-in model implements this card),
    ``"hook"`` (a ``models=`` factory handles it), or ``"fallback"`` (no
    implementation for this level: devices use the builder's default MOSFET
    model, and only its parameters are read from the card).
    """

    name: str
    type: str
    level: str
    version: str | None
    status: str
    implementation: str
    n_params: int
    loc: Loc | None = None
    subckt: str = ""
    """Enclosing subcircuit(s) for a local card (``"a/b"``), else ``""``."""


@dataclass
class DeviceModel:
    """The model card (and bin) chosen for an ``M``/``D``/``Q``/``N`` device."""

    model: str
    card: str
    implementation: str
    bounds: tuple[float, float, float, float] | None = None
    l: float | None = None
    w: float | None = None
    margin: float = math.inf
    """Smallest relative distance of the device's L/W to its bin edges."""


@dataclass
class NetlistInfo:
    """What `parse_netlist(..., return_info=True)` learned about the deck.

    Attributes:
        files: Files read (the deck and every ``.include`` / ``.lib`` file).
        params: Top-level parameters that evaluate to a number.
        options: ``.options`` (numbers where they evaluate) and ``temp``.
        models: Every ``.model`` card (bins included), keyed by name, or
            ``"subckt/name"`` for cards local to a subcircuit.
        devices: Model card / bin chosen for each ``M``, ``D``, ``Q`` device.
        subckts: Names of the top-level subcircuit definitions.
    """

    files: list[str] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    models: dict[str, ModelSummary] = field(default_factory=dict)
    devices: dict[str, DeviceModel] = field(default_factory=dict)
    subckts: list[str] = field(default_factory=list)

    def model_report(self) -> str:
        """Table of model cards grouped by (type, level, status)."""
        groups: dict[tuple, list[str]] = {}
        for s in self.models.values():
            groups.setdefault((s.type, s.level, s.status, s.implementation),
                              []).append(s.name)
        tw = max([4, *(len(t) for t, *_ in groups)])
        lw = max([5, *(len(lv or "-") for _, lv, *_ in groups)])
        lines = [f"{'type':{tw}} {'level':{lw}} {'status':11} {'cards':>5}  "
                 "implementation"]
        for (typ, lv, status, impl), names in sorted(groups.items()):
            lines.append(f"{typ:{tw}} {lv or '-':{lw}} {status:11} "
                         f"{len(names):5d}  {impl}")
        return "\n".join(lines)


# =============================================================================
# Emitter
# =============================================================================


class Emitter:
    """Adds flattened netlist elements to a `CircuitBuilder`."""

    def __init__(self, flat: Flattener, lines: list[Elem], builder: CircuitBuilder,
                 hooks: Mapping[str, ModelFactory] | None, options: dict[str, Any],
                 tran: tuple[float, float] | None, info: NetlistInfo):
        self.fl = flat
        self.lines = lines
        self.b = builder
        self.hooks = dict(hooks or {})
        self.options = options
        self.tran = tran
        self.info = info
        self.scale = float(options.get("scale", 1.0))
        self.temp = float(options.get("temp", 27.0))
        self.vt = (K_BOLTZMANN_Q * (self.temp + 273.15)
                   if "temp" in options else None)
        self.by_name = {e.name.lower(): e for e in lines}
        if len(self.by_name) != len(lines):
            seen: set[str] = set()
            for e in lines:
                if e.name.lower() in seen:
                    raise NetlistError(f"duplicate element name {e.name!r}", e.loc)
                seen.add(e.name.lower())
        self.processes: dict[tuple, el.MOSProcess] = {}
        self.vsource_nodes: dict[str, tuple[str, str]] = {}
        self.sense_ports: dict[tuple[str, str], tuple[str, str]] = {}
        self.compiled: dict[str, tuple[Compiled, str]] = {}
        self.coupled: set[str] = set()
        self.warned: set[str] = set()
        self.ignored_instance: dict[str, list[str]] = {}
        self.bin_sets: dict[int, list[ModelCard]] = {}
        self.near_edges: list[str] = []
        self.ctrl_nodes: list[tuple[Elem, str]] = []

    def run(self) -> Circuit:
        for e in self.lines:
            self._prepare(e)
        self._route_current_sensing()
        self._collect_couplings()
        for e in self.lines:
            handler = getattr(self, f"_{e.letter}", None)
            if handler is None:
                raise NetlistError(f"unsupported element {e.name!r}", e.loc)
            handler(e)
        self._final_warnings()
        nodes = set()
        for _, dev, _ in self.b._devices:
            terms = dev.nodes[0]
            if isinstance(dev, el.BehavioralSource):  # control nodes don't count
                terms = terms[:2] + terms[2 + dev.n_ctrl:]
            nodes.update(terms)
        for e, n in self.ctrl_nodes:
            if n not in nodes:
                raise NetlistError(f"{e.name}: node {n!r} is not connected to "
                                   "any element", e.loc)
        if not self.b.nodes:
            self.warn("netlist defines no nodes")
        return self.b.build()

    # --------------------------------------------------------------- helpers

    def warn(self, message: str, key: str | None = None) -> None:
        """Warn once per `key`, attributed to the caller of `parse_netlist`."""
        if key is not None:
            if key in self.warned:
                return
            self.warned.add(key)
        warn(message)

    def value(self, e: Elem, text: str, what: str = "value") -> float:
        """Number or expression `text` in `e`'s scope."""
        num = try_number(text)
        if num is not None:
            return num
        try:
            return e.scope.eval(text, e.loc)
        except NetlistError as err:
            raise NetlistError(f"{e.name}: cannot parse {what} {text!r}: "
                               f"{err.message}", e.loc) from None

    @staticmethod
    def positional(e: Elem) -> list[str]:
        """Tokens without ``=`` (initial-condition flags dropped)."""
        return [t for t in e.toks if split_kv(t) is None
                and t.lower() not in ("off", "on")]

    def keywords(self, e: Elem) -> dict[str, str]:
        out = {}
        for t in e.toks:
            kv = split_kv(t)
            if kv is not None:
                out[kv[0]] = kv[1]
        return out

    def need(self, e: Elem, n: int, syntax: str) -> list[str]:
        pos = self.positional(e)
        if len(pos) < n:
            raise NetlistError(f"{e.name}: expected `{syntax}`", e.loc)
        return pos

    def _compile(self, e: Elem, ast: AST, text: str) -> Compiled:
        frame = e.frame
        out = compile_jax(
            ast, e.scope, text, e.loc,
            node_name=lambda n: self.fl.node(frame, n),
            device_name=lambda d: self.fl.device(frame, d).lower(),
            ground=lambda n: n == "0",
            constants=e.scope.constants,
        )
        self.ctrl_nodes += [(e, n) for n in out.nodes]
        return out

    # ---------------------------------------------- behavioral pre-compilation

    def _prepare(self, e: Elem) -> None:
        letter = e.letter
        if letter == "b":
            self._prepare_b(e)
        elif letter in "eg" and len(e.toks) > 3 and (
                split_kv(e.toks[3]) or e.toks[3].lower().startswith(("poly", "table"))):
            self._prepare_eg(e)
        elif letter in "fh":
            self._prepare_fh(e)

    def _prepare_b(self, e: Elem) -> None:
        kws = self.keywords(e)
        exprs = {k: v for k, v in kws.items() if k in ("v", "i")}
        if len(exprs) != 1 or len(self.positional(e)) != 3:
            raise NetlistError(f"{e.name}: expected `B<name> n+ n- V={{expr}}` or "
                               "`I={expr}`", e.loc)
        extra = set(kws) - {"v", "i", "m"}
        if extra:
            raise NetlistError(f"{e.name}: unsupported B-source parameters "
                               f"{sorted(extra)}", e.loc)
        (kind, text), = exprs.items()
        ast = parse_expression(text, e.loc)
        if kind == "i":
            ast = self._times_m(e, ast, kws.get("m"))
        self.compiled[e.name.lower()] = (self._compile(e, ast, text), kind)

    def _times_m(self, e: Elem, ast: AST, m: str | None) -> AST:
        mult = e.mult * (self.value(e, m, "m") if m else 1.0)
        return ast if mult == 1.0 else ("bin", "*", ast, ("num", mult))

    def _prepare_eg(self, e: Elem) -> None:
        output = "v" if e.letter == "e" else "i"
        tok = e.toks[3]
        low = tok.lower()
        kv = split_kv(tok)
        if kv is not None and kv[0] in ("value", "vol", "cur"):
            if (kv[0] == "vol" and output != "v") or (kv[0] == "cur" and output != "i"):
                raise NetlistError(f"{e.name}: {kv[0]}= does not fit a "
                                   f"{e.letter.upper()} source", e.loc)
            text = kv[1]
            ast = parse_expression(text, e.loc)
        elif low.startswith("table"):
            text = " ".join(e.toks[3:])
            ast = self._table(e, text)
        elif low.startswith("poly"):
            text = " ".join(e.toks[3:])
            n = int(re.match(r"poly\((\d+)\)", low).group(1))
            pos = e.toks[4:]
            if len(pos) < 2 * n:
                raise NetlistError(f"{e.name}: POLY({n}) needs {2 * n} control nodes",
                                   e.loc)
            nodes, coeffs = e.toks[4:4 + 2 * n], pos[2 * n:]
            # control nodes were renamed by the flattener already
            xs = [("v", nodes[2 * k], nodes[2 * k + 1]) for k in range(n)]
            ast = _poly(xs, [self.value(e, c, "coefficient") for c in coeffs], e)
            compiled = compile_jax(ast, e.scope, text, e.loc, node_name=lambda x: x,
                                   device_name=lambda d: d, ground=lambda x: x == "0",
                                   constants=e.scope.constants)
            self.ctrl_nodes += [(e, x) for x in compiled.nodes]
            self.compiled[e.name.lower()] = (compiled, output)
            return
        else:
            raise NetlistError(f"{e.name}: unsupported syntax {tok!r}", e.loc)
        if output == "i":
            ast = self._times_m(e, ast, self.keywords(e).get("m"))
        self.compiled[e.name.lower()] = (self._compile(e, ast, text), output)

    def _table(self, e: Elem, text: str) -> AST:
        m = re.match(r"table\s*(\{.*?\}|'.*?'|\S+?)\s*=?\s*(\(.*)$", text,
                     re.IGNORECASE | re.DOTALL)
        if not m:
            raise NetlistError(f"{e.name}: expected `TABLE {{expr}} = (x1, y1) ...`",
                               e.loc)
        ast = parse_expression(m.group(1), e.loc)
        pairs = re.findall(r"\(([^()]*)\)", m.group(2))
        xs, ys = [], []
        for p in pairs:
            parts = [x for x in re.split(r"[\s,]+", p.strip()) if x]
            if len(parts) != 2:
                raise NetlistError(f"{e.name}: bad table point ({p})", e.loc)
            xs.append(self.value(e, parts[0]))
            ys.append(self.value(e, parts[1]))
        if len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
            raise NetlistError(f"{e.name}: TABLE needs >= 2 points with increasing x",
                               e.loc)
        return ("table", ast, tuple(xs), tuple(ys))

    def _prepare_fh(self, e: Elem) -> None:
        if len(e.toks) > 3 and e.toks[3].lower().startswith("poly"):
            low = e.toks[3].lower()
            n = int(re.match(r"poly\((\d+)\)", low).group(1))
            if len(e.toks) < 4 + n:
                raise NetlistError(f"{e.name}: POLY({n}) needs {n} controlling "
                                   "sources", e.loc)
            xs = [("i", name) for name in e.toks[4:4 + n]]
            coeffs = [self.value(e, c, "coefficient") for c in e.toks[4 + n:]]
            ast = _poly(xs, coeffs, e)
            output = "i" if e.letter == "f" else "v"
            if output == "i":
                ast = self._times_m(e, ast, None)
            self.compiled[e.name.lower()] = (
                self._compile(e, ast, " ".join(e.toks[3:])), output)

    # ------------------------------------------------------------ pre-passes

    def _sensors(self) -> dict[str, list[str]]:
        """``{vsource: [sensor, ...]}`` for every current-sensing element."""
        users: dict[str, list[str]] = {}
        for e in self.lines:
            key = e.name.lower()
            if key in self.compiled:
                sensed = self.compiled[key][0].sensed
            elif e.letter in "fh":
                if len(e.toks) < 5:
                    raise NetlistError(f"{e.name}: expected `{e.letter.upper()}"
                                       "<name> n+ n- Vsense gain`", e.loc)
                sensed = [self.fl.device(e.frame, e.toks[3]).lower()]
            else:
                continue
            for v in sensed:
                target = self.by_name.get(v)
                if target is None or target.letter != "v":
                    kind = "is not a voltage source" if target else "not found"
                    shown = (target.name if target else v).rsplit(".", 1)[-1]
                    raise NetlistError(f"{e.name}: controlling source {shown!r} "
                                       f"{kind}", e.loc)
                users.setdefault(v, []).append(key)
        return users

    def _route_current_sensing(self) -> None:
        for vname, sensors in self._sensors().items():
            v = self.by_name[vname]
            p, n = v.toks[1], v.toks[2]
            chain = [f"{v.name}#{k}" for k in range(len(sensors))] + [n]
            self.vsource_nodes[vname] = (p, chain[0])
            for k, sensor in enumerate(sensors):
                self.sense_ports[(sensor, vname)] = (chain[k], chain[k + 1])

    def _collect_couplings(self) -> None:
        for e in self.lines:
            if e.letter != "k":
                continue
            pos = self.need(e, 4, "K<name> L1 L2 k")
            if len(pos) != 4:
                raise NetlistError(f"{e.name}: only two coupled inductors per K "
                                   "element are supported", e.loc)
            for ind in pos[1:3]:
                full = self.fl.device(e.frame, ind).lower()
                if full in self.coupled or full not in self.by_name \
                        or self.by_name[full].letter != "l":
                    raise NetlistError(f"{e.name}: inductor {ind!r} missing or "
                                       "coupled twice", e.loc)
                self.coupled.add(full)

    # ------------------------------------------------------- passive elements

    def _two_terminal_value(self, e: Elem, key: str) -> tuple[float | None, dict]:
        """Value of an R/C/L: positional or ``r=``/``c=``/``l=``."""
        pos = self.positional(e)
        kws = self.keywords(e)
        if key in kws:
            return self.value(e, kws.pop(key)), kws
        if len(pos) > 3:
            num = try_number(pos[3])
            if num is not None or pos[3][0] in "{'\"" or e.scope.has(pos[3].lower()):
                return self.value(e, pos[3]), kws
        return None, kws

    def _ignore(self, e: Elem, kws: dict, allowed: set[str], what: str) -> None:
        extra = set(kws) - allowed
        if extra:
            self.warn(f"{e.name}: ignoring {what} parameters {sorted(extra)}",
                      key=f"{e.name}:{what}")

    def _r(self, e: Elem) -> None:
        if len(self.positional(e)) < 3:
            raise NetlistError(f"{e.name}: expected `R<name> n+ n- value`", e.loc)
        r, kws = self._two_terminal_value(e, "r")
        if r is None:  # semiconductor resistor: R n+ n- model l= w=
            r = self._model_resistance(e, kws)
        self._ignore(e, kws, {"m", "l", "w", "r", "temp", "dtemp"}, "resistor")
        m = e.mult * (self.value(e, kws["m"], "m") if "m" in kws else 1.0)
        if r <= 0:
            raise NetlistError(f"{e.name}: resistance must be positive, got {r}",
                               e.loc)
        self.b.resistor(e.toks[1], e.toks[2], r / m, name=e.name)

    def _model_resistance(self, e: Elem, kws: dict) -> float:
        pos = self.positional(e)
        if len(pos) < 4:
            raise NetlistError(f"{e.name}: missing resistance", e.loc)
        card = self._card(e, pos[3], None, None)
        rsh = card.params.get("rsh")
        if rsh is None or "l" not in kws:
            raise NetlistError(f"{e.name}: resistor model {pos[3]!r} needs rsh and "
                               "l= (r = rsh l / w)", e.loc)
        w = self.value(e, kws.get("w", "0"), "w") or card.params.get("defw", 1e-6)
        narrow = card.params.get("narrow", 0.0)
        short = card.params.get("short", 0.0)
        return rsh * (self.value(e, kws["l"], "l") - short) / (w - narrow)

    def _c(self, e: Elem) -> None:
        if len(self.positional(e)) < 3:
            raise NetlistError(f"{e.name}: expected `C<name> n+ n- value`", e.loc)
        c, kws = self._two_terminal_value(e, "c")
        if c is None:
            raise NetlistError(f"{e.name}: missing capacitance", e.loc)
        self._ignore(e, kws, {"m", "ic"}, "capacitor")
        m = e.mult * (self.value(e, kws["m"], "m") if "m" in kws else 1.0)
        self.b.capacitor(e.toks[1], e.toks[2], c * m, name=e.name)

    def _inductance(self, e: Elem) -> float:
        l, kws = self._two_terminal_value(e, "l")
        if l is None:
            raise NetlistError(f"{e.name}: missing inductance", e.loc)
        self._ignore(e, kws, {"m", "ic"}, "inductor")
        return l / (e.mult * (self.value(e, kws["m"], "m") if "m" in kws else 1.0))

    def _l(self, e: Elem) -> None:
        if len(self.positional(e)) < 4:
            raise NetlistError(f"{e.name}: expected `L<name> n+ n- value`", e.loc)
        if e.name.lower() not in self.coupled:  # coupled ones are emitted by K
            self.b.inductor(e.toks[1], e.toks[2], self._inductance(e), name=e.name)

    def _k(self, e: Elem) -> None:
        pos = self.positional(e)
        l1 = self.by_name[self.fl.device(e.frame, pos[1]).lower()]
        l2 = self.by_name[self.fl.device(e.frame, pos[2]).lower()]
        k = self.value(e, pos[3], "coupling")
        if not 0 < abs(k) <= 1:
            raise NetlistError(f"{e.name}: coupling must be in (0, 1], got {k}",
                               e.loc)
        self.b.transformer(l1.toks[1], l1.toks[2], l2.toks[1], l2.toks[2],
                           self._inductance(l1), self._inductance(l2), k,
                           name=e.name)

    # ---------------------------------------------------------------- sources

    def _v(self, e: Elem) -> None:
        if len(e.toks) < 3:
            raise NetlistError(f"{e.name}: expected `V<name> n+ n- [value]`", e.loc)
        value, ac, phase = self._source(e, e.toks[3:], 1.0)
        p, n = self.vsource_nodes.get(e.name.lower(), (e.toks[1], e.toks[2]))
        self.b.vsource(p, n, value, ac, phase, name=e.name)

    def _i(self, e: Elem) -> None:
        if len(e.toks) < 3:
            raise NetlistError(f"{e.name}: expected `I<name> n+ n- [value]`", e.loc)
        kws = self.keywords(e)
        m = e.mult * (self.value(e, kws["m"], "m") if "m" in kws else 1.0)
        toks = [t for t in e.toks[3:] if not (split_kv(t) and split_kv(t)[0] == "m")]
        value, ac, phase = self._source(e, toks, m)
        self.b.isource(e.toks[1], e.toks[2], value, ac * m, phase, name=e.name)

    _WAVES = ("pulse", "sin", "exp", "pwl", "sffm", "am")

    def _source(self, e: Elem, toks: list[str], scale: float
                ) -> tuple[Any, float, float]:
        """``[DC] v [AC mag [phase]] [PULSE|SIN|EXP|PWL|SFFM|AM(...)]`` in any
        order; returns ``(value or Signal, ac_mag, ac_phase)``."""
        dc: float | None = None
        wave: tuple[str, list[str]] | None = None
        pwl_opts: dict[str, str] = {}
        ac, phase = 0.0, 0.0
        i = 0

        def is_value(tok: str) -> bool:
            return try_number(tok) is not None or tok[:1] in "{'\""

        while i < len(toks):
            tok, low = toks[i], toks[i].lower()
            kv = split_kv(tok)
            func = re.match(r"([a-z]+)\s*\((.*)\)$", tok, re.IGNORECASE | re.DOTALL)
            if func and func.group(1).lower() in self._WAVES:
                if wave is not None:
                    raise NetlistError(f"{e.name}: more than one waveform", e.loc)
                wave = (func.group(1).lower(), _split_args(func.group(2)))
            elif low in self._WAVES:  # bare form: SIN 0 1 1k
                if wave is not None:
                    raise NetlistError(f"{e.name}: more than one waveform", e.loc)
                args = []
                def is_file(tok: str) -> bool:
                    return low == "pwl" and (split_kv(tok) or ("",))[0] == "file"

                while i + 1 < len(toks) and (is_value(toks[i + 1])
                                             or is_file(toks[i + 1])):
                    args.append(toks[i + 1])
                    i += 1
                wave = (low, args)
            elif kv is not None:
                key, val = kv
                if key == "dc":
                    dc = self.value(e, val, "DC value")
                elif key == "ac":
                    ac = self.value(e, val, "AC magnitude")
                elif key in ("r", "td") and wave and wave[0] == "pwl":
                    pwl_opts[key] = val
                elif key == "file" and wave and wave[0] == "pwl":
                    wave[1].append(tok)
                else:
                    raise NetlistError(f"{e.name}: unsupported source parameter "
                                       f"{key!r}", e.loc)
            elif low == "dc":
                if i + 1 >= len(toks):
                    raise NetlistError(f"{e.name}: DC needs a value", e.loc)
                dc = self.value(e, toks[i + 1], "DC value")
                i += 1
            elif low == "ac":
                if i + 1 < len(toks) and is_value(toks[i + 1]):
                    ac = self.value(e, toks[i + 1], "AC magnitude")
                    i += 1
                    if i + 1 < len(toks) and is_value(toks[i + 1]):
                        phase = self.value(e, toks[i + 1], "AC phase")
                        i += 1
                else:
                    ac = 1.0
            elif low in ("distof1", "distof2"):
                raise NetlistError(f"{e.name}: distortion inputs are not supported",
                                   e.loc)
            else:
                dc = self.value(e, tok, "source value")
            i += 1
        if wave is None:
            return (dc or 0.0) * scale, ac, phase
        return self._wave(e, wave[0], wave[1], pwl_opts, scale), ac, phase

    def _wave(self, e: Elem, kind: str, raw: list[str], pwl_opts: dict[str, str],
              scale: float) -> signals.Signal:
        if kind == "pwl":
            return self._pwl(e, raw, pwl_opts, scale)
        args = [self.value(e, a, f"{kind.upper()} argument") for a in raw]
        tstep, tstop = self.tran or (None, None)

        def arg(k: int, default: float | None, zero_default: bool = False) -> float:
            if k < len(args) and not (zero_default and args[k] == 0.0):
                return args[k]
            if default is None:
                raise NetlistError(f"{e.name}: {kind.upper()} argument {k + 1} is "
                                   "required (or add a .tran line for SPICE "
                                   "defaults)", e.loc)
            return default

        need = {"pulse": 2, "sin": 2, "exp": 2, "sffm": 2, "am": 4}[kind]
        if len(args) < need:
            raise NetlistError(f"{e.name}: {kind.upper()} needs at least {need} "
                               "arguments", e.loc)
        if kind == "pulse":
            if self.tran:
                rise = arg(3, tstep, True)
                fall = arg(4, tstep, True)
                width = arg(5, tstop, True)
                period = arg(6, tstop, True)
            else:
                rise, fall = arg(3, 1e-9), arg(4, 1e-9)
                width, period = arg(5, 1e-6), arg(6, 2e-6)
            return signals.Pulse(args[0] * scale, args[1] * scale, arg(2, 0.0), rise,
                                 fall, width, period)
        if kind == "sin":
            freq = arg(2, 1.0 / tstop if tstop else None, bool(tstop))
            return signals.Sine(args[0] * scale, args[1] * scale, freq, arg(3, 0.0),
                                arg(4, 0.0), arg(5, 0.0))
        if kind == "exp":
            td1 = arg(2, 0.0)
            tau1 = arg(3, tstep, bool(tstep))
            td2 = arg(4, td1 + tstep if tstep else None, bool(tstep))
            tau2 = arg(5, tstep, bool(tstep))
            return signals.Exp(args[0] * scale, args[1] * scale, td1, tau1, td2, tau2)
        if kind == "sffm":
            f0 = 1.0 / tstop if tstop else None
            return signals.SFFM(args[0] * scale, args[1] * scale, arg(2, f0, bool(f0)),
                                arg(3, 0.0), arg(4, f0, bool(f0)), arg(5, 0.0),
                                arg(6, 0.0))
        return signals.AM(args[0] * scale, args[1], args[2], args[3], arg(4, 0.0))

    def _pwl(self, e: Elem, raw: list[str], opts: dict[str, str],
             scale: float) -> signals.PWL:
        points: list[float] = []
        for a in raw:
            kv = split_kv(a)
            if kv is not None and kv[0] in ("r", "td"):
                opts[kv[0]] = kv[1]
            elif kv is not None and kv[0] == "file":
                points += self._pwl_file(e, unquote(kv[1]))
            elif kv is not None:
                raise NetlistError(f"{e.name}: unsupported PWL option {kv[0]!r}", e.loc)
            else:
                points.append(self.value(e, a.strip("()"), "PWL point"))
        if len(points) < 2 or len(points) % 2:
            raise NetlistError(f"{e.name}: PWL needs time/value pairs", e.loc)
        times, values = points[0::2], [v * scale for v in points[1::2]]
        if any(b < a for a, b in zip(times, times[1:])):
            raise NetlistError(f"{e.name}: PWL times must be non-decreasing", e.loc)
        repeat = self.value(e, opts["r"], "r") if "r" in opts else -1.0
        if repeat >= 0 and not any(abs(t - repeat) <= 1e-12 * max(1.0, abs(t))
                                   for t in times):
            raise NetlistError(f"{e.name}: PWL r={repeat:g} is not one of the "
                               "time points", e.loc)
        delay = self.value(e, opts["td"], "td") if "td" in opts else 0.0
        return signals.PWL(times, values, delay, repeat)

    def _pwl_file(self, e: Elem, name: str) -> list[float]:
        base = Path(e.loc.file).parent if not e.loc.file.startswith("<") \
            else self.fl.base_dir
        path = Path(name) if Path(name).is_absolute() else base / name
        try:
            text = path.read_text()
        except OSError as err:
            raise NetlistError(f"{e.name}: cannot read PWL file {str(path)!r}: "
                               f"{err.strerror}", e.loc) from None
        points: list[float] = []
        for k, line in enumerate(text.splitlines(), start=1):
            line = line.split("#", 1)[0].split(";", 1)[0].strip()
            if not line or line.startswith("*"):
                continue
            parts = [p for p in re.split(r"[\s,]+", line) if p]
            if len(parts) != 2:
                raise NetlistError(f"{path}:{k}: expected `time value`", e.loc)
            try:
                points += [float(try_number(p)) for p in parts]
            except TypeError:
                raise NetlistError(f"{path}:{k}: cannot parse {line!r}",
                                   e.loc) from None
        return points

    # ------------------------------------------------------ controlled sources

    def _behavioral(self, e: Elem) -> None:
        compiled, output = self.compiled[e.name.lower()]
        ports = [n for d in compiled.sensed
                 for n in self.sense_ports[(e.name.lower(), d)]]
        nodes = (e.toks[1], e.toks[2], *compiled.nodes, *ports)
        device = el.BehavioralSource(nodes, compiled.expr, compiled.params,
                                     output=output, n_sense=len(compiled.sensed))
        self.b.add(device, name=e.name)

    def _b(self, e: Elem) -> None:
        self._behavioral(e)

    def _e(self, e: Elem) -> None:
        if e.name.lower() in self.compiled:
            return self._behavioral(e)
        pos = self.need(e, 6, "E<name> n+ n- nc+ nc- gain")
        self.b.vcvs(e.toks[1], e.toks[2], e.toks[3], e.toks[4],
                    self.value(e, pos[5], "gain"), name=e.name)

    def _g(self, e: Elem) -> None:
        if e.name.lower() in self.compiled:
            return self._behavioral(e)
        pos = self.need(e, 6, "G<name> n+ n- nc+ nc- gm")
        kws = self.keywords(e)
        m = e.mult * (self.value(e, kws["m"], "m") if "m" in kws else 1.0)
        self.b.vccs(e.toks[1], e.toks[2], e.toks[3], e.toks[4],
                    self.value(e, pos[5], "gm") * m, name=e.name)

    def _f(self, e: Elem) -> None:
        if e.name.lower() in self.compiled:
            return self._behavioral(e)
        sp, sn = self.sense_ports[(e.name.lower(),
                                   self.fl.device(e.frame, e.toks[3]).lower())]
        gain = self.value(e, e.toks[4], "gain") * e.mult
        self.b.cccs(e.toks[1], e.toks[2], sp, sn, gain, name=e.name)

    def _h(self, e: Elem) -> None:
        if e.name.lower() in self.compiled:
            return self._behavioral(e)
        sp, sn = self.sense_ports[(e.name.lower(),
                                   self.fl.device(e.frame, e.toks[3]).lower())]
        self.b.ccvs(e.toks[1], e.toks[2], sp, sn, self.value(e, e.toks[4], "gain"),
                    name=e.name)

    # ---------------------------------------------------------- model lookup

    def _card(self, e: Elem, name: str, l: float | None, w: float | None
              ) -> ModelCard:
        """Model card `name` as seen from `e`, bin-selected by (l, w)."""
        card, bins = self.fl.model(e.frame, name, e.loc)
        if card is not None:
            return card
        if bins:
            if l is None or w is None:
                raise NetlistError(f"{e.name}: binned model {name!r} needs a "
                                   "device with L and W", e.loc)
            card = select_bin(bins, l, w, e.name, e.loc)
            self.bin_sets[id(card)] = bins
            return card
        raise NetlistError(f"{e.name}: unknown model {name!r}", e.loc)

    def _bare_card(self, e: Elem, name: str | None, default: str) -> ModelCard:
        """A card for a missing or bare-type model name (``nmos``, ``d``)."""
        kind = (name or default).lower()
        return ModelCard(kind, kind, kind, {}, {}, None, None, e.loc)

    def _check_params(self, card: ModelCard, extra: set[str] = frozenset()) -> None:
        """Warn once per card about parameters Voltax does not use."""
        unused = set(card.params) - MODEL_PARAMS.get(card.type, set()) - extra
        if unused:
            self.warn(f"model {card.name!r} ({card.type}): ignoring unsupported "
                      f"parameters {sorted(unused)}", key=f"params:{card.base}")

    def _record(self, e: Elem, ref: str, card: ModelCard, impl: str,
                l: float | None = None, w: float | None = None) -> None:
        binned = card.name.lower() != ref.lower()
        info = DeviceModel(ref, card.name, impl,
                           card.bounds if binned else None, l, w)
        if binned and l is not None and w is not None:
            info.margin = bin_margin(card, l, w, self.bin_sets.get(id(card)))
            if info.margin < 0.01:
                self.near_edges.append(f"{e.name} ({card.name}, margin "
                                       f"{info.margin:.2%})")
        self.info.devices[e.name] = info

    def _hooked(self, e: Elem, ref: str, card: ModelCard, nodes: tuple[str, ...],
                instance: dict[str, float], l=None, w=None) -> bool:
        factory = find_hook(self.hooks, card, ref)
        if factory is None:
            return False
        spec = DeviceSpec(e.name, e.letter, nodes, card, dict(instance),
                          {**self.options, "temp": self.temp})
        device = call_hook(factory, spec)
        if not isinstance(device, Element):
            raise NetlistError(f"{e.name}: model hook {_cls_name(factory)} returned "
                               f"{type(device).__name__}, not an Element", e.loc)
        self.b.add(device, name=e.name)
        self._record(e, ref, card, f"hook: {_cls_name(factory)}", l, w)
        return True

    # ----------------------------------------------------------- semiconductors

    def _d(self, e: Elem) -> None:
        pos = self.need(e, 3, "D<name> n+ n- [model] [area]")
        kws = {k: self.value(e, v, k) for k, v in self.keywords(e).items()}
        ref = pos[3] if len(pos) > 3 else None
        area = kws.pop("area", self.value(e, pos[4], "area") if len(pos) > 4 else 1.0)
        m = e.mult * kws.pop("m", 1.0)
        card = self._card(e, ref, None, None) if ref and ref.lower() != "d" \
            else self._bare_card(e, ref, "d")
        instance = {**kws, "area": area, "m": m}
        if self._hooked(e, ref or "d", card, (e.toks[1], e.toks[2]), instance):
            return
        if card.type != "d":
            raise NetlistError(f"{e.name}: model {ref!r} is a {card.type}, not a "
                               "diode", e.loc)
        if card.level not in (None, 1):
            self.warn(f"model {card.name!r}: diode level {card.level_str()} is not "
                      "implemented; using the Shockley diode", key=f"lv:{card.key}")
        self._check_params(card)
        self._ignore(e, kws, {"off", "ic", "temp", "dtemp"}, "diode")
        p = card.params
        scale = area * m
        cathode = e.toks[2]
        if p.get("rs", 0.0) > 0:  # series resistance via an internal node
            cathode = self.b.node(f"{e.name}#rs")
            self.b.resistor(cathode, e.toks[2], p["rs"] / scale, name=f"{e.name}#rs")
        extra = {"vt": self.vt} if self.vt else {}
        self.b.diode(e.toks[1], cathode, name=e.name, is_=p.get("is", 1e-14) * scale,
                     n=p.get("n", 1.0), cj=p.get("cjo", p.get("cj0", 0.0)) * scale,
                     tt=p.get("tt", 0.0), **extra)
        self._record(e, ref or "d", card, "Diode")

    def _q(self, e: Elem) -> None:
        pos = self.need(e, 4, "Q<name> c b e [s] [model] [area]")
        kws = {k: self.value(e, v, k) for k, v in self.keywords(e).items()}
        n_nodes = 3
        if len(pos) > 5 and self.fl._is_model(e.frame, pos[5]):
            n_nodes = 4
        ref = pos[1 + n_nodes] if len(pos) > 1 + n_nodes else None
        area = kws.pop("area", self.value(e, pos[2 + n_nodes], "area")
                       if len(pos) > 2 + n_nodes else 1.0)
        m = e.mult * kws.pop("m", 1.0)
        if ref is None or ref.lower() in ("npn", "pnp"):
            card = self._bare_card(e, ref, "npn")
        else:
            card = self._card(e, ref, None, None)
        nodes = tuple(e.toks[1:1 + n_nodes])
        instance = {**kws, "area": area, "m": m}
        if self._hooked(e, ref or card.type, card, nodes, instance):
            return
        if card.type not in ("npn", "pnp"):
            raise NetlistError(f"{e.name}: model {ref!r} is not npn/pnp", e.loc)
        if n_nodes == 4:
            self.warn(f"{e.name}: substrate node ignored", key=f"sub:{e.name}")
        self._check_params(card)
        self._ignore(e, kws, {"off", "ic", "temp", "dtemp"}, "BJT")
        p, scale = card.params, area * m
        extra = {"vt": self.vt} if self.vt else {}
        self.b.bjt(e.toks[1], e.toks[2], e.toks[3], card.type, name=e.name,
                   is_=p.get("is", 1e-16) * scale, bf=p.get("bf", 100.0),
                   br=p.get("br", 1.0), vaf=p.get("vaf", p.get("va", float("inf"))),
                   tf=p.get("tf", 0.0), tr=p.get("tr", 0.0),
                   cje=p.get("cje", 0.0) * scale, cjc=p.get("cjc", 0.0) * scale,
                   **extra)
        self._record(e, ref or card.type, card, "BJT")

    def _m(self, e: Elem) -> None:
        pos = self.need(e, 6, "M<name> d g s b model [w=] [l=]")
        ref = pos[5]
        inst = {k: self.value(e, v, k) for k, v in self.keywords(e).items()}
        s = self.scale
        defw = float(self.options.get("defw", 1e-6))
        defl = float(self.options.get("defl", 0.13e-6))
        w = inst.get("w", defw / s) * s
        l = inst.get("l", defl / s) * s
        inst["w"], inst["l"] = w, l
        for k in ("ad", "as"):
            if k in inst:
                inst[k] *= s * s
        for k in ("pd", "ps"):
            if k in inst:
                inst[k] *= s
        inst["m"] = inst.get("m", 1.0) * e.mult
        nf = inst.get("nf", 1.0)
        nodes = tuple(e.toks[1:5])
        if ref.lower() in ("nmos", "pmos") and not self.fl._is_model(e.frame, ref):
            card = self._bare_card(e, ref, "nmos")
        else:
            card = self._card(e, ref, l, w / nf)
        if self._hooked(e, ref, card, nodes, inst, l, w / nf):
            return
        if card.type not in ("nmos", "pmos"):
            raise NetlistError(f"{e.name}: model {ref!r} is a {card.type}, not "
                               "nmos/pmos", e.loc)
        polarity = "p" if card.type == "pmos" else "n"
        cls = self.b.mos_model
        if card.level == 1 and not _is_level1(cls):
            cls = el.Level1MOSFET  # builder's model unless it isn't level 1
        if card.level not in (None, 1):
            self.warn(f"model {card.name!r}: level {card.level_str()} has no Voltax "
                      f"implementation; approximating it with {_cls_name(cls)} "
                      "(register one with parse_netlist(..., models={"
                      f"'{card.type}:{card.level_str()}': ...}}))",
                      key=f"lv:{card.base}")
        self._check_params(card)
        if _is_level1(cls) and "tox" in card.params:
            self.warn(f"model {card.name!r}: Level1MOSFET has overlap capacitance "
                      "only; intrinsic (Meyer) gate capacitance from tox is not "
                      "modeled", key=f"tox:{card.key}")
        ignored = set(inst) - {"w", "l", "m"} - ({"nf"} if nf == 1 else set())
        for key in ignored:
            self.ignored_instance.setdefault(key, []).append(e.name)
        process = self._mos_process(card, polarity)
        device = cls(nodes, w * inst["m"], l, process, polarity)
        self.b.add(device, name=e.name)
        self._record(e, ref, card, _cls_name(cls), l, w / nf)

    def _mos_process(self, card: ModelCard, polarity: str) -> el.MOSProcess:
        """One shared `MOSProcess` per model card (the builder's by default)."""
        key = (card.key, polarity)
        if key in self.processes:
            return self.processes[key]
        p = card.params
        base = self.b.pmos_process if polarity == "p" else self.b.nmos_process
        overrides = {
            "kp": p.get("kp"),
            "vth": abs(p["vto"]) if "vto" in p else None,
            "lam": p.get("lambda"),
            "n": p.get("n"),
            "gamma": p.get("gamma"),
            "phi": p.get("phi"),
            "cox": 3.45e-11 / p["tox"] if "tox" in p else None,
            "cov": p.get("cgso", p.get("cgdo")),
            "vt": self.vt,
        }
        if p.get("cgso", 0.0) != p.get("cgdo", p.get("cgso", 0.0)):
            self.warn(f"model {card.name!r}: cgso != cgdo; using cgso for both "
                      "overlap capacitances", key=f"cgso:{card.key}")
        overrides = {k: v for k, v in overrides.items() if v is not None}
        process = base.replace(**overrides) if overrides else base
        self.processes[key] = process
        return process

    def _n(self, e: Elem) -> None:
        """``N<name> nodes... model [k=v ...]``: a device compiled from
        Verilog-A (ngspice OSDI). Needs a ``models=`` hook."""
        pos = self.positional(e)
        k = next(i for i, t in enumerate(pos[1:], 1) if self.fl._is_model(e.frame, t))
        ref, nodes = pos[k], tuple(pos[1:k])
        inst = {key: self.value(e, v, key) for key, v in self.keywords(e).items()}
        if "w" in inst and "l" in inst:
            inst["w"] *= self.scale
            inst["l"] *= self.scale
        inst["m"] = inst.get("m", 1.0) * e.mult
        card = self._card(e, ref, inst.get("l"), inst.get("w"))
        if self._hooked(e, ref, card, nodes, inst, inst.get("l"), inst.get("w")):
            return
        if not (is_va_mosfet(card) and len(nodes) == 4):
            raise NetlistError(f"{e.name}: model {ref!r} (type {card.type!r}) is a "
                               "compiled (Verilog-A/OSDI) device; register it with "
                               f"parse_netlist(..., models={{'{card.type}': ...}})",
                               e.loc)
        # a Verilog-A MOSFET (PSP, BSIM-CMG, ...): approximate it like an
        # unimplemented .model level until a hook supplies the real model
        polarity = "p" if card.params["type"] < 0 else "n"
        cls = self.b.mos_model
        self.warn(f"model {card.name!r}: {card.type} (Verilog-A MOSFET) has no "
                  f"Voltax implementation; approximating it with {_cls_name(cls)} "
                  f"(register one with parse_netlist(..., models={{'{card.type}': "
                  "...}))", key=f"va:{card.base}")
        defw = float(self.options.get("defw", 1e-6))
        defl = float(self.options.get("defl", 0.13e-6))
        w, l = inst.get("w", defw), inst.get("l", defl)
        m = inst["m"] * inst.get("mult", 1.0)
        for key in set(inst) - {"w", "l", "m", "mult"}:
            self.ignored_instance.setdefault(key, []).append(e.name)
        device = cls(nodes, w * m, l, self._mos_process(card, polarity), polarity)
        self.b.add(device, name=e.name)
        self._record(e, ref, card, f"{_cls_name(cls)} (fallback)", l, w)

    # ---------------------------------------------------------------- switches

    def _s(self, e: Elem) -> None:
        pos = self.need(e, 5, "S<name> n+ n- nc+ nc- [model]")
        ref = pos[5] if len(pos) > 5 and pos[5].lower() not in ("on", "off") else None
        card = self._card(e, ref, None, None) if ref and ref.lower() != "sw" \
            else self._bare_card(e, ref, "sw")
        if card.type != "sw":
            raise NetlistError(f"{e.name}: model {ref!r} is not a sw model", e.loc)
        self._check_params(card)
        p = card.params
        self.b.switch(e.toks[1], e.toks[2], e.toks[3], e.toks[4], name=e.name,
                      ron=p.get("ron", 1.0), roff=p.get("roff", 1e9),
                      vth=p.get("vt", 0.5), vwidth=max(p.get("vh", 0.0), 0.05))

    # --------------------------------------------------------------- warnings

    def _final_warnings(self) -> None:
        if self.ignored_instance:
            devices = {d for ds in self.ignored_instance.values() for d in ds}
            keys = sorted(self.ignored_instance)
            self.warn(f"instance parameters {keys} on {len(devices)} MOSFET(s) "
                      f"(e.g. {min(devices)}) are ignored by the built-in "
                      "EKV/Level-1 models (no junction capacitance, source/drain "
                      "resistance or layout effects; nf > 1 only splits W)")
        if self.near_edges:
            self.warn(f"{len(self.near_edges)} device(s) within 1% of a model-bin "
                      f"edge: {', '.join(self.near_edges[:5])}. The bin is fixed at "
                      "parse time, so W/L gradients ignore the jump to the "
                      "neighbouring bin")


def is_va_mosfet(card: ModelCard) -> bool:
    """A Verilog-A (OSDI) MOSFET card: unknown type with ``type = +-1``."""
    return card.type not in MODEL_PARAMS and card.params.get("type") in (1.0, -1.0)


def _split_args(text: str) -> list[str]:
    """Arguments of ``PULSE(...)``: whitespace/comma separated, braces and
    quotes kept together."""
    out, cur, depth, quote = [], "", 0, None
    for ch in text:
        if quote:
            cur += ch
            quote = None if ch == quote else quote
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if depth == 0 and (ch.isspace() or ch == ","):
            if cur:
                out.append(cur)
            cur = ""
            continue
        cur += ch
    if cur:
        out.append(cur)
    # join key = value split by spaces
    joined: list[str] = []
    i = 0
    while i < len(out):
        if out[i] == "=" and joined and i + 1 < len(out):
            joined[-1] += "=" + out[i + 1]
            i += 2
            continue
        if out[i].endswith("=") and i + 1 < len(out):
            joined.append(out[i] + out[i + 1])
            i += 2
            continue
        joined.append(out[i])
        i += 1
    return joined


def _poly(xs: list[AST], coeffs: list[float], e: Elem) -> AST:
    """SPICE ``POLY(n)`` polynomial: ``p0 + sum p_i x_i + sum p_ij x_i x_j ...``
    with terms ordered as in SPICE (combinations with replacement)."""
    if not coeffs:
        raise NetlistError(f"{e.name}: POLY needs coefficients", e.loc)
    if len(xs) == 1 and len(coeffs) == 1:  # SPICE2: a lone coefficient is p1
        coeffs = [0.0, coeffs[0]]
    terms: list[AST] = []
    k = 0
    degree = 0
    while k < len(coeffs):
        for combo in itertools.combinations_with_replacement(range(len(xs)), degree):
            if k == len(coeffs):
                break
            c = coeffs[k]
            k += 1
            if c == 0.0:
                continue
            term: AST = ("num", c)
            for j in combo:
                term = ("bin", "*", term, xs[j])
            terms.append(term)
        degree += 1
    out: AST = ("num", 0.0)
    for t in terms:
        out = ("bin", "+", out, t)
    return out
