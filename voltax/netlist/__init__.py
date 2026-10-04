"""SPICE netlist parser.

`parse_netlist` turns a SPICE deck into a `Circuit` via `CircuitBuilder`.
Device and node names are kept as written (first spelling wins; SPICE names
are case-insensitive), so results are addressed exactly as in the deck:
``sol.v("out")``, ``sol.i("Vdd")``.

The parser is split into four stages, one module each:

=================  =========================================================
`lexer`            numbers, logical lines with ``file:line``, fields
`preprocess`       ``.include``/``.lib``, definitions, subcircuit flattening
`expressions`      ``.param`` expression language, scopes, JAX compilation
`emit`             one handler per element letter -> `CircuitBuilder`
=================  =========================================================

(`models` holds model cards, binning and the ``models=`` hook.)

Supported elements: ``R C L K V I E F G H B D Q M S X``; directives
``.param .func .subckt/.ends .model .include .lib/.endl .global .options
.temp .end``; analysis and output directives (``.tran .ac .dc .op .print
.control`` ...) are skipped (``.tran`` supplies SPICE's default source
timings). Anything else raises `NetlistError` (a `ValueError`) with the file
and line number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..builder import CircuitBuilder
from ..circuit import Circuit
from .emit import DeviceModel, Emitter, ModelSummary, NetlistInfo
from .expressions import CONSTANTS, CompiledExpr, Scope, parse_expression
from .lexer import NetlistError, parse_value, read_lines, try_number
from .models import DeviceSpec, ModelCard, ModelFactory, find_hook
from .preprocess import Deck, Flattener, Sources, SubcktDef, collect, expand

__all__ = [
    "CompiledExpr",
    "DeviceModel",
    "DeviceSpec",
    "ModelCard",
    "ModelSummary",
    "NetlistError",
    "NetlistInfo",
    "parse_expression",
    "parse_netlist",
    "parse_netlist_file",
    "parse_value",
]


def parse_netlist(text: str, title: bool = False, *,
                  base_dir: str | Path | None = None,
                  models: Mapping[str, ModelFactory] | None = None,
                  return_info: bool = False, source: str = "<netlist>",
                  **builder_kwargs: Any) -> Circuit | tuple[Circuit, NetlistInfo]:
    """Parse a SPICE netlist into a `Circuit`.

    Args:
        text: Netlist source.
        title: Treat the first line as a title (SPICE file convention).
        base_dir: Directory that relative ``.include`` / ``.lib`` / PWL file
            paths are resolved against (default: the working directory).
        models: Model hook: ``{key: factory}`` mapping model cards to your
            own `Element` classes, e.g. ``{"nmos:54": BSIM4, "pmos:54":
            BSIM4}``. See `voltax.netlist.DeviceSpec` for the key order and
            the factory contract.
        return_info: Also return a `voltax.netlist.NetlistInfo` (files read,
            parameters, model cards and how they map, chosen bins).
        source: Name used for this text in error messages.
        **builder_kwargs: Passed to `CircuitBuilder` (default processes, MOS
            model).

    Raises:
        NetlistError: A `ValueError` whose message starts with
            ``file:line:``.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    sources = Sources()
    lines = expand(read_lines(text, source, title), base, sources)
    deck = collect(lines)
    constants = dict(CONSTANTS)
    flat = Flattener(deck, constants)
    flat.base_dir = base
    options = _options(deck, flat.top_scope)
    constants["temper"] = options.get("temp", 27.0)
    tran = None
    if deck.tran is not None:
        tran = tuple(_eval(flat.top_scope, x, ".tran") for x in deck.tran)
    elements = flat.run()
    builder = CircuitBuilder(**builder_kwargs)
    info = NetlistInfo(files=sources.files, options=options,
                       subckts=sorted(deck.top.subckts))
    circuit = Emitter(flat, elements, builder, models, options, tran, info).run()
    info.params = flat.top_scope.all_values()
    info.models = _model_summaries(deck, flat.top_scope, models or {}, builder)
    return (circuit, info) if return_info else circuit


def parse_netlist_file(path: str | Path, title: bool = True,
                       **kwargs: Any) -> Circuit | tuple[Circuit, NetlistInfo]:
    """Read and parse a netlist file (first line is the title by default).

    Relative ``.include`` / ``.lib`` paths resolve against the file's
    directory. Keyword arguments are those of `parse_netlist`.
    """
    path = Path(path)
    kwargs.setdefault("base_dir", path.resolve().parent)
    return parse_netlist(path.read_text(errors="replace"), title=title,
                         source=str(path), **kwargs)


def _eval(scope: Scope, text: str, what: str) -> float:
    num = try_number(text)
    if num is not None:
        return num
    try:
        return scope.eval(text)
    except NetlistError as e:
        raise NetlistError(f"{what}: {e.message}") from None


def _options(deck: Deck, scope: Scope) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, text in deck.options.items():
        try:
            out[key] = _eval(scope, text, ".options")
        except NetlistError:
            out[key] = text  # e.g. method=gear
    for flag in deck.option_flags:
        out.setdefault(flag, True)
    if deck.temp is not None:
        out["temp"] = _eval(scope, deck.temp, ".temp")
    for key in ("scale", "temp", "tnom", "defw", "defl"):
        if key in out and not isinstance(out[key], float):
            raise NetlistError(f".options {key}={out[key]!r} is not a number")
    return out


def _model_summaries(deck: Deck, scope: Scope, hooks: Mapping[str, ModelFactory],
                     builder: CircuitBuilder) -> dict[str, ModelSummary]:
    """Every ``.model`` in the deck, top level and inside subcircuits.

    Subcircuit-local cards may depend on instance parameters, so only their
    ``level`` and ``version`` are read (literally, or in the top scope).
    """
    out = {}

    def visit(d: SubcktDef, prefix: str) -> None:
        for name, m in d.models.items():
            text = dict(m.params)
            level = None
            if "level" in text:
                level = try_number(text["level"])
                if level is None:
                    try:
                        level = scope.eval(text["level"], m.loc)
                    except NetlistError:
                        level = None
            version = text.get("version")
            card = ModelCard(m.name, name, m.type, {}, text, level,
                             version.strip("'\"") if version else None, m.loc)
            if "." in name and name.rsplit(".", 1)[1].isdigit():
                card.base = name.rsplit(".", 1)[0]
            status, impl = _implementation(card, hooks, builder.mos_model)
            out[prefix + name] = ModelSummary(
                m.name, m.type, card.level_str() if level is not None else
                ("?" if "level" in text else ""), card.version, status, impl,
                len(m.params), m.loc, prefix.rstrip("/"))
        for sub_name, sub in d.subckts.items():
            visit(sub, f"{prefix}{sub_name}/")

    visit(deck.top, "")
    return out


def _implementation(card: ModelCard, hooks: Mapping[str, ModelFactory],
                    mos_model: Any) -> tuple[str, str]:
    factory = find_hook(hooks, card, card.base)
    if factory is not None:
        name = getattr(getattr(factory, "func", factory), "__name__", "factory")
        return "hook", name
    level = card.level
    mos_name = getattr(getattr(mos_model, "func", mos_model), "__name__", "?")
    if card.type in ("nmos", "pmos"):
        if level == 1:
            return "native", "Level1MOSFET"
        if level is None:
            return "native", mos_name
        return "fallback", f"{mos_name} (level {card.level_str()} not implemented)"
    if card.type == "d":
        if level in (None, 1):
            return "native", "Diode"
        return "fallback", f"Diode (level {card.level_str()} not implemented)"
    native = {"npn": "BJT", "pnp": "BJT", "sw": "Switch", "r": "Resistor",
              "c": "Capacitor"}
    if card.type in native:
        return "native", native[card.type]
    if try_number(card.text.get("type", "")) in (1.0, -1.0):  # N device (OSDI)
        return "fallback", f"{mos_name} (needs Verilog-A model {card.type!r})"
    return "unsupported", f"no Voltax model for type {card.type!r}"

