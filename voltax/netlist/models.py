"""Model cards, binning and the device-model hook.

A ``.model`` line is stored unevaluated (`ModelDef`) in the scope where it
appears and evaluated on first use (`ModelCard`), so parameter expressions
can reference ``.param`` values, including subcircuit parameters.

**Binning.** Foundry decks define one model per geometry bin, named
``nch.1``, ``nch.2``, ... with ``lmin/lmax/wmin/wmax``. A device that
references ``nch`` gets a bin with ``lmin <= L <= lmax`` and
``wmin <= W <= wmax`` (W per finger), matching ngspice-42 exactly: edges
have a 1 nm tolerance and bins are tried in reverse definition order, so a
device on a shared edge gets the later-defined bin. The bin is
chosen once, at parse time, from the instance geometry. Gradients with
respect to W/L are therefore those of the chosen bin's model; they do not
"see" the jump to a neighbouring bin, which makes the true W/L dependence
piecewise.

**Hook.** ``parse_netlist(..., models={key: factory})`` maps model cards to
your own `Element` classes. See `DeviceSpec` for the contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .expressions import Scope
from .lexer import Loc, NetlistError, split_fields, split_kv, try_number

BIN_EPS = 1e-9
"""Bin-edge tolerance (m), as in ngspice: ``lmin - eps <= L <= lmax + eps``."""


@dataclass
class ModelDef:
    """An unevaluated ``.model`` line."""

    name: str
    type: str
    params: list[tuple[str, str]]
    loc: Loc


def parse_model_line(text: str, loc: Loc) -> ModelDef:
    """``.model name type [(] k=v ... [)]`` -> `ModelDef` (unevaluated)."""
    m = re.match(r"\.model\s+(\S+)\s+([A-Za-z_]\w*)\s*(.*)$", text, re.IGNORECASE
                 | re.DOTALL)
    if not m:
        raise NetlistError(f"malformed .model line: {text!r}", loc)
    name, kind, rest = m.group(1), m.group(2).lower(), m.group(3).strip()
    if rest.startswith("("):  # parentheses are optional (and often unbalanced)
        rest = rest[1:]
        if rest.rstrip().endswith(")"):
            rest = rest.rstrip()[:-1]
    rest = _commas_to_spaces(rest)
    params: list[tuple[str, str]] = []
    for tok in split_fields(rest):
        kv = split_kv(tok)
        if kv is None:
            if tok.startswith("(") or tok.endswith(")"):
                tok = tok.strip("()")
                if not tok:
                    continue
                kv = split_kv(tok)
            if kv is None:
                raise NetlistError(f".model {name}: expected key=value, got {tok!r}",
                                   loc)
        params.append(kv)
    return ModelDef(name, kind, params, loc)


def _commas_to_spaces(text: str) -> str:
    """Model parameters may be comma separated; keep commas inside brackets."""
    out, depth, quote = [], 0, None
    for ch in text:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            ch = " "
        out.append(ch)
    return "".join(out)


@dataclass(eq=False)
class ModelCard:
    """An evaluated model card.

    Attributes:
        name: Name as defined (for a bin, e.g. ``"nch.3"``).
        base: Name devices use (``"nch"`` for every ``nch.N`` bin).
        type: Model type, lower case (``"nmos"``, ``"pmos"``, ``"d"``,
            ``"npn"``, ``"pnp"``, ``"sw"``, ``"r"``, ...).
        params: Parameter values by lower-case name (expressions evaluated).
        text: Raw (unevaluated) parameter text by name, e.g. for version
            strings such as ``version=4.5``.
        level: ``params["level"]`` if given.
        version: ``version`` parameter as written, if given.
        loc: Where the card is defined.
    """

    name: str
    base: str
    type: str
    params: dict[str, float]
    text: dict[str, str]
    level: float | None
    version: str | None
    loc: Loc

    @property
    def key(self) -> tuple:
        """Hashable identity (one card object per definition and scope)."""
        return (self.name.lower(), id(self))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(lmin, lmax, wmin, wmax)``; missing bounds are 0 / inf."""
        p, inf = self.params, float("inf")
        return (p.get("lmin", 0.0), p.get("lmax", inf), p.get("wmin", 0.0),
                p.get("wmax", inf))

    def level_str(self) -> str:
        if self.level is None:
            return ""
        return str(int(self.level)) if float(self.level).is_integer() \
            else str(self.level)

    def __repr__(self) -> str:
        lv = f" level={self.level_str()}" if self.level is not None else ""
        return f"ModelCard({self.name!r}, {self.type!r}{lv}, {len(self.params)} params)"


def evaluate_model(d: ModelDef, scope: Scope) -> ModelCard:
    """Evaluate a `ModelDef`'s parameter expressions in `scope`."""
    params, text = {}, {}
    for key, value in d.params:
        text[key] = value
        num = try_number(value)
        if num is None:
            try:
                num = scope.eval(value, d.loc)
            except NetlistError:
                if key == "version":  # e.g. version=4.5.0: keep as text only
                    continue
                raise
        params[key] = num
    base = d.name.lower()
    if re.fullmatch(r".+\.\d+", base):
        base = base.rsplit(".", 1)[0]
    version = text.get("version")
    return ModelCard(d.name, base, d.type, params, text, params.get("level"),
                     version.strip("'\"") if version else None, d.loc)


def select_bin(bins: list[ModelCard], l: float, w: float, device: str,
               loc: Loc | None) -> ModelCard:
    """The bin whose ``[lmin, lmax] x [wmin, wmax]`` (+- `BIN_EPS`) contains
    ``(l, w)``; bins are tried last-defined first, as ngspice does."""
    for card in reversed(bins):
        lmin, lmax, wmin, wmax = card.bounds
        if (lmin - BIN_EPS <= l <= lmax + BIN_EPS
                and wmin - BIN_EPS <= w <= wmax + BIN_EPS):
            return card
    ranges = ", ".join(f"{c.name}: L[{c.bounds[0]:g},{c.bounds[1]:g}) "
                       f"W[{c.bounds[2]:g},{c.bounds[3]:g})" for c in bins)
    raise NetlistError(f"{device}: no bin of model {bins[0].base!r} covers "
                       f"L={l:g}, W={w:g} ({ranges})", loc)


def bin_margin(card: ModelCard, l: float, w: float,
               bins: list[ModelCard] | None = None) -> float:
    """Smallest relative distance of ``(l, w)`` to an edge of `card`'s bin
    that it shares with another bin in `bins` (crossing it would switch
    models); ``inf`` if there is none. Without `bins`, every finite edge
    counts."""
    lmin, lmax, wmin, wmax = card.bounds
    out = float("inf")
    others = [b for b in (bins or []) if b is not card]

    def shared(point: tuple[float, float]) -> bool:
        if bins is None:
            return True
        return any(b.bounds[0] <= point[0] <= b.bounds[1]
                   and b.bounds[2] <= point[1] <= b.bounds[3] for b in others)

    for k, (x, edges) in enumerate(((l, (lmin, lmax)), (w, (wmin, wmax)))):
        for e in edges:
            if not 0 < e < float("inf"):
                continue
            beyond = e * (1 - 1e-6) if e == edges[0] else e * (1 + 1e-6)
            point = (beyond, w) if k == 0 else (l, beyond)
            if shared(point):
                out = min(out, abs(x - e) / e)
    return out


# =============================================================================
# The hook
# =============================================================================


@dataclass(frozen=True)
class DeviceSpec:
    """Everything a model hook needs to build one device.

    A hook is registered as ``parse_netlist(text, models={key: factory})``.
    Keys are matched case-insensitively, most specific first:

    1. the model name a device references (``"nch"``) or its bin (``"nch.3"``),
    2. ``"<type>:<level>:<version>"``, e.g. ``"nmos:54:4.5"``,
    3. ``"<type>:<level>"``, e.g. ``"nmos:54"``, ``"pmos:54"``, ``"d:3"``,
    4. ``"<type>"``, e.g. ``"nmos"``.

    The factory is called as ``factory.from_netlist(spec)`` if it has such a
    method (e.g. a classmethod on an `Element` subclass), otherwise as
    ``factory(spec)``. It must return a single-device `voltax.Element` whose
    nodes are ``spec.nodes`` (names); the parser adds it under
    ``spec.name``. Hooks apply to ``M``, ``D``, ``Q`` and ``N`` (ngspice
    OSDI / Verilog-A) elements. Without a hook, an ``N`` device whose card has
    ``type = +1/-1`` (a Verilog-A MOSFET such as PSP) falls back to the
    builder's MOSFET model with a warning; other ``N`` devices raise.

    Devices that reference the same card get the same ``spec.model`` object,
    so a factory can share process-level parameters (and their gradients)
    between them, e.g. by caching on ``spec.model.key``.

    Attributes:
        name: Hierarchical device name (``"X1.M3"``).
        letter: Element letter (``"m"``, ``"d"``, ``"q"``, ``"n"``).
        nodes: Terminal node names as written in the netlist (after
            subcircuit renaming): ``(d, g, s, b)`` for MOSFETs, ``(a, k)``
            for diodes, ``(c, b, e[, s])`` for BJTs, all tokens before the
            model name for ``N`` devices.
        model: The (bin-selected) `ModelCard`; ``model.params`` holds every
            model parameter with expressions evaluated.
        instance: Instance parameters, lower case, evaluated. For MOSFETs
            ``w``/``l`` (and ``ad as pd ps``) already include ``.option
            scale``; ``m`` is the total multiplier (instance ``m`` times
            enclosing subcircuit ``m``); other keys (``nf``, ``nrd``,
            ``nrs``, ``sa``, ``sb``, ...) are passed through. Diodes and
            BJTs get ``area`` (default 1) and ``m``.
        options: ``.options`` values plus ``temp`` (deg C, default 27) and
            ``tnom``.
    """

    name: str
    letter: str
    nodes: tuple[str, ...]
    model: ModelCard
    instance: Mapping[str, float]
    options: Mapping[str, float] = field(default_factory=dict)


ModelFactory = Callable[[DeviceSpec], Any]


def hook_keys(card: ModelCard, referenced: str) -> list[str]:
    """Hook lookup keys for a card, most specific first."""
    keys = [referenced.lower(), card.name.lower()]
    lv = card.level_str()
    if lv:
        if card.version:
            keys.append(f"{card.type}:{lv}:{card.version.lower()}")
        keys.append(f"{card.type}:{lv}")
    keys.append(card.type)
    return list(dict.fromkeys(keys))


def find_hook(hooks: Mapping[str, ModelFactory], card: ModelCard,
              referenced: str) -> ModelFactory | None:
    if not hooks:
        return None
    lowered = {k.lower(): v for k, v in hooks.items()}
    for key in hook_keys(card, referenced):
        if key in lowered:
            return lowered[key]
    return None


def call_hook(factory: ModelFactory, spec: DeviceSpec) -> Any:
    make = getattr(factory, "from_netlist", None)
    return make(spec) if make is not None else factory(spec)
