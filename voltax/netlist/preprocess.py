"""Deck structure: includes and libraries, definitions, flattening.

1. `expand` inlines ``.include`` files and ``.lib file section`` sections
   (paths relative to the including file; each file or section is read
   once; recursion raises).
2. `collect` builds the definition tree: the top level and every
   ``.subckt`` (nested definitions allowed) hold their element lines,
   ``.param`` definitions, ``.model`` cards and ``.func`` functions.
3. `flatten` instantiates the hierarchy into a list of `Elem` records with
   hierarchical names (``X1.X2.M3``), renamed nodes, the parameter `Scope`
   of their instance and their subcircuit multiplier ``m``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .expressions import Scope
from .lexer import (
    Line,
    Loc,
    NetlistError,
    read_lines,
    split_fields,
    split_kv,
    unquote,
    warn,
)
from .models import ModelCard, ModelDef, evaluate_model, parse_model_line

GROUND = frozenset({"0", "gnd", "ground"})

IGNORED = {".tran", ".ac", ".dc", ".op", ".print", ".plot", ".save", ".meas",
           ".measure", ".ic", ".nodeset", ".probe", ".title", ".four", ".fourier",
           ".noise", ".tf", ".sens", ".pz", ".disto", ".width", ".csparam",
           ".options_end", ".hdl", ".alter", ".protect", ".unprotect"}
"""Directives that do not describe the circuit (analyses, output): skipped."""


# =============================================================================
# 1. Includes and libraries
# =============================================================================


@dataclass
class Sources:
    """Files read while expanding includes (for reports and caching)."""

    files: list[str] = field(default_factory=list)
    _cache: dict[Path, list[Line]] = field(default_factory=dict)
    _sections: dict[Path, dict[str, list[Line]]] = field(default_factory=dict)

    def lines(self, path: Path, loc: Loc | None) -> list[Line]:
        if path not in self._cache:
            try:
                text = path.read_text(errors="replace")
            except OSError as e:
                raise NetlistError(f"cannot read {path}: {e.strerror}", loc) from None
            self._cache[path] = read_lines(text, str(path), title=False)
            self.files.append(str(path))
        return self._cache[path]

    def section(self, path: Path, name: str, loc: Loc | None) -> list[Line]:
        if path not in self._sections:
            self._sections[path] = _lib_sections(self.lines(path, loc))
        sections = self._sections[path]
        if name.lower() not in sections:
            have = ", ".join(sorted(sections)) or "none"
            raise NetlistError(f"section {name!r} not found in {path} "
                               f"(sections: {have})", loc)
        return sections[name.lower()]


def _lib_sections(lines: list[Line]) -> dict[str, list[Line]]:
    sections: dict[str, list[Line]] = {}
    current: list[Line] | None = None
    for line in lines:
        toks = line.text.split()
        head = toks[0].lower()
        if head == ".lib" and len(toks) == 2 and current is None:
            current = sections.setdefault(toks[1].lower(), [])
        elif head == ".endl" and current is not None:
            current = None
        elif current is not None:
            current.append(line)
    return sections


def _resolve(raw: str, base_dir: Path, loc: Loc, must_exist: bool = True) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(unquote(raw))))
    if not path.is_absolute():
        path = base_dir / path
    if not must_exist:
        return path
    if not path.exists():
        raise NetlistError(f"file not found: {unquote(raw)!r} (looked in {base_dir})",
                           loc)
    return path.resolve()


def expand(lines: list[Line], base_dir: Path, sources: Sources,
           stack: tuple = (), seen: set | None = None,
           context: list[str] | None = None) -> list[Line]:
    """Inline ``.include`` / ``.lib file section``; drop ``.lib name``
    section definitions (they are only used through a reference).

    A file (or section) is inlined once per definition context: twice at the
    top level is a no-op, but inside each ``.subckt`` body it is inlined
    again (IHP includes its model cards inside every device subcircuit).
    `context` is the stack of currently open ``.subckt`` names.
    """
    seen = set() if seen is None else seen
    context = [] if context is None else context
    out: list[Line] = []
    skipping: Loc | None = None
    for line in lines:
        toks = split_fields(line.text)
        head = toks[0].lower()
        if skipping is not None:
            if head == ".endl":
                skipping = None
            continue
        if head == "include" and len(toks) == 2:
            # not SPICE (ngspice rejects it as a malformed current source),
            # but it appears in foundry decks: honour it as .include
            path = _resolve(toks[1], base_dir, line.loc, must_exist=False)
            if not path.exists():
                warn(f"{line.loc}: skipping `include {toks[1]}` (no leading dot, "
                     f"and {path} does not exist)")
                continue
            warn(f"{line.loc}: `include` without a dot treated as .include")
            head = ".include"
        if head in (".include", ".inc", ".incl"):
            if len(toks) != 2:
                raise NetlistError(f"{head} takes one file name", line.loc)
            path = _resolve(toks[1], base_dir, line.loc)
            key = (path, None)
            if key in stack:
                raise NetlistError(f"recursive include of {path}", line.loc)
            if (key, tuple(context)) in seen:
                continue  # include once per context
            seen.add((key, tuple(context)))
            out += expand(sources.lines(path, line.loc), path.parent, sources,
                          stack + (key,), seen, context)
        elif head == ".lib":
            if len(toks) == 2:  # a section definition in this file: skip it
                skipping = line.loc
                continue
            if len(toks) != 3:
                raise NetlistError(".lib takes a file name and a section name",
                                   line.loc)
            path = _resolve(toks[1], base_dir, line.loc)
            key = (path, toks[2].lower())
            if key in stack:
                raise NetlistError(f"recursive .lib {path} {toks[2]}", line.loc)
            if (key, tuple(context)) in seen:
                continue
            seen.add((key, tuple(context)))
            body = sources.section(path, toks[2], line.loc)
            out += expand(body, path.parent, sources, stack + (key,), seen,
                          context)
        elif head == ".endl":
            raise NetlistError(".endl without .lib", line.loc)
        else:
            if head == ".subckt" and len(toks) > 1:
                context.append(toks[1].lower())
            elif head == ".ends" and context:
                context.pop()
            out.append(line)
    if skipping is not None:
        raise NetlistError(".lib section without .endl", skipping)
    return out


# =============================================================================
# 2. Definitions
# =============================================================================


@dataclass
class Conditional:
    """An ``.if`` / ``.elseif`` / ``.else`` / ``.endif`` block.

    `branches` holds ``(condition, loc, items)``; the condition of an
    ``.else`` branch is None. Items are element lines ``(toks, line)``,
    nested `Conditional` blocks, or ``.param`` definitions as
    ``("param", name, text, loc)``. The block is resolved when its
    subcircuit is instantiated, with conditions evaluated in that instance's
    parameter scope.
    """

    loc: Loc
    branches: list[tuple[str | None, Loc, list]] = field(default_factory=list)


@dataclass
class SubcktDef:
    """A ``.subckt`` (or the top level, ``name=""``)."""

    name: str
    pins: list[str]
    loc: Loc | None
    parent: "SubcktDef | None" = None
    defaults: list[tuple[str, str]] = field(default_factory=list)
    params: list[tuple[str, str, Loc]] = field(default_factory=list)
    funcs: list[tuple[str, tuple[str, ...], str, Loc]] = field(default_factory=list)
    models: dict[str, ModelDef] = field(default_factory=dict)
    bins: dict[str, list[ModelDef]] = field(default_factory=dict)
    subckts: dict[str, "SubcktDef"] = field(default_factory=dict)
    body: list = field(default_factory=list)
    """Element lines ``(toks, line)`` and `Conditional` blocks, in order."""

    def find_subckt(self, name: str) -> "SubcktDef | None":
        d: SubcktDef | None = self
        while d is not None:
            if name in d.subckts:
                return d.subckts[name]
            d = d.parent
        return None


@dataclass
class Deck:
    top: SubcktDef
    globals: set[str] = field(default_factory=set)
    options: dict[str, str] = field(default_factory=dict)
    option_flags: set[str] = field(default_factory=set)
    tran: tuple[str, str] | None = None
    temp: str | None = None


def _param_assignments(toks: list[str], loc: Loc, what: str) -> list[tuple[str, str]]:
    out = []
    for tok in toks:
        kv = split_kv(tok)
        if kv is None:
            raise NetlistError(f"{what}: expected name=value, got {tok!r}", loc)
        out.append(kv)
    return out


_FUNC = re.compile(r"([A-Za-z_]\w*)\s*\(([^)]*)\)\s*=?\s*(.+)$", re.DOTALL)


def _func_def(text: str, loc: Loc) -> tuple[str, tuple[str, ...], str]:
    m = _FUNC.match(text.strip())
    if not m:
        raise NetlistError(f"malformed function definition {text!r}", loc)
    args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
    return m.group(1), args, m.group(3).strip()


def collect(lines: list[Line]) -> Deck:
    """Sort lines into the definition tree."""
    top = SubcktDef("", [], None)
    deck = Deck(top)
    current = top
    conds: list[Conditional] = []  # open .if blocks in `current`

    def target() -> list:
        return conds[-1].branches[-1][2] if conds else current.body

    for line in lines:
        toks = split_fields(line.text)
        head = toks[0].lower()
        if not head.startswith("."):
            target().append((toks, line))
            continue
        if head in (".if", ".elseif", ".elif", ".else", ".endif"):
            cond = line.text.split(None, 1)[1].strip() if len(toks) > 1 else None
            if head == ".if":
                if cond is None:
                    raise NetlistError(".if needs a condition", line.loc)
                block = Conditional(line.loc, [(cond, line.loc, [])])
                target().append(block)
                conds.append(block)
            elif not conds:
                raise NetlistError(f"{head} without .if", line.loc)
            elif head == ".endif":
                conds.pop()
            elif conds[-1].branches[-1][0] is None:
                raise NetlistError(f"{head} after .else", line.loc)
            elif head == ".else":
                conds[-1].branches.append((None, line.loc, []))
            else:
                if cond is None:
                    raise NetlistError(f"{head} needs a condition", line.loc)
                conds[-1].branches.append((cond, line.loc, []))
            continue
        if conds:
            if head == ".param":
                for k, v in _param_assignments(toks[1:], line.loc, ".param"):
                    target().append(("param", k, v, line.loc))
                continue
            if head in IGNORED:
                continue
            if head == ".ends":
                raise NetlistError(".ends inside an open .if", line.loc)
            raise NetlistError(f"{toks[0]} inside .if is not supported", line.loc)
        if head == ".subckt":
            if len(toks) < 2:
                raise NetlistError(".subckt needs a name", line.loc)
            pins, defaults = [], []
            for tok in toks[2:]:
                kv = split_kv(tok)
                if kv is not None:
                    defaults.append(kv)
                elif tok.lower() in ("params:", "param:"):
                    continue
                elif defaults:
                    raise NetlistError(f".subckt {toks[1]}: pin {tok!r} after "
                                       "parameters", line.loc)
                else:
                    pins.append(tok)
            sub = SubcktDef(toks[1].lower(), pins, line.loc, current, defaults)
            if sub.name in current.subckts:
                raise NetlistError(f"subcircuit {toks[1]!r} defined twice "
                                   f"(first at {current.subckts[sub.name].loc})",
                                   line.loc)
            current.subckts[sub.name] = sub
            current = sub
        elif head == ".ends":
            if current is top:
                raise NetlistError(".ends without .subckt", line.loc)
            if len(toks) > 1 and toks[1].lower() != current.name:
                raise NetlistError(f".ends {toks[1]} closes .subckt {current.name}",
                                   line.loc)
            current = current.parent
        elif head == ".param":
            body = line.text.split(None, 1)[1] if len(toks) > 1 else ""
            if re.match(r"\s*[A-Za-z_]\w*\s*\(", body):  # .param f(x) = ...
                current.funcs.append((*_func_def(body, line.loc), line.loc))
                continue
            for k, v in _param_assignments(toks[1:], line.loc, ".param"):
                current.params.append((k, v, line.loc))
        elif head == ".func":
            body = line.text.split(None, 1)[1] if len(toks) > 1 else ""
            current.funcs.append((*_func_def(body, line.loc), line.loc))
        elif head == ".model":
            d = parse_model_line(line.text, line.loc)
            name = d.name.lower()
            m = re.fullmatch(r"(.+)\.(\d+)", name)
            if m:
                current.bins.setdefault(m.group(1), []).append(d)
            current.models[name] = d
        elif head == ".global":
            deck.globals.update(t for t in toks[1:])
        elif head in (".option", ".options", ".opt", ".opts"):
            for tok in toks[1:]:
                kv = split_kv(tok)
                if kv is None:
                    deck.option_flags.add(tok.lower())
                else:
                    deck.options[kv[0]] = kv[1]
        elif head == ".temp":
            if len(toks) < 2:
                raise NetlistError(".temp needs a value", line.loc)
            deck.temp = toks[1]
        elif head == ".tran":
            pos = [t for t in toks[1:] if split_kv(t) is None
                   and t.lower() not in ("uic",)]
            if len(pos) >= 2:
                deck.tran = (pos[0], pos[1])
        elif head in IGNORED:
            continue
        else:
            raise NetlistError(f"unsupported directive {toks[0]!r}", line.loc)
    if conds:
        raise NetlistError(".if without .endif", conds[-1].loc)
    if current is not top:
        raise NetlistError(f".subckt {current.name} has no .ends", current.loc)
    return deck


# =============================================================================
# 3. Flattening
# =============================================================================


@dataclass
class Frame:
    """One subcircuit instance during flattening."""

    definition: SubcktDef
    scope: Scope
    prefix: str
    pins: dict[str, str]
    mult: float
    parent: "Frame | None"
    cards: dict[str, ModelCard] = field(default_factory=dict)


@dataclass
class Elem:
    """A flattened element line.

    `toks[0]` is the hierarchical name; node tokens are renamed. `frame`
    gives access to the instance's parameter scope and model cards.
    """

    toks: list[str]
    loc: Loc
    frame: Frame

    @property
    def name(self) -> str:
        return self.toks[0]

    @property
    def letter(self) -> str:
        return self.toks[0].rsplit(".", 1)[-1][0].lower()

    @property
    def scope(self) -> Scope:
        return self.frame.scope

    @property
    def mult(self) -> float:
        return self.frame.mult


class Flattener:
    """Instantiates the definition tree (see module docstring)."""

    MAX_DEPTH = 50

    def __init__(self, deck: Deck, constants: dict[str, float]):
        self.deck = deck
        self.globals = {g.lower() for g in deck.globals}
        self.canon: dict[str, str] = {}
        self.top_scope = Scope(None, "<top>", constants)
        self.out: list[Elem] = []
        self.root = Frame(deck.top, self.top_scope, "", {}, 1.0, None)
        self._define(deck.top, self.top_scope)

    # ----------------------------------------------------------------- scopes

    @staticmethod
    def _define(d: SubcktDef, scope: Scope) -> None:
        for k, v in d.defaults:
            scope.define(k, v, d.loc)
        for k, v, loc in d.params:
            scope.define(k, v, loc)
        for name, args, body, loc in d.funcs:
            scope.define_function(name, args, body, loc)

    # ------------------------------------------------------------------ nodes

    def node(self, frame: Frame, name: str) -> str:
        """Rename a node token used inside `frame`."""
        low = name.lower()
        if low in GROUND:
            return "0"
        if name in frame.pins:
            return frame.pins[name]
        if low in frame.pins:
            return frame.pins[low]
        if low in self.globals or not frame.prefix:
            return self.canon.setdefault(low, name)
        full = frame.prefix + name
        return self.canon.setdefault(full.lower(), full)

    def device(self, frame: Frame, name: str) -> str:
        """Hierarchical name of a device referenced inside `frame`."""
        return frame.prefix + name

    # ----------------------------------------------------------------- models

    def model(self, frame: Frame, name: str, loc: Loc | None
              ) -> tuple[ModelCard | None, list[ModelCard]]:
        """``(card, bins)`` for a model reference: an exact card, or the bins
        ``name.N`` (to be selected by geometry); ``(None, [])`` if unknown."""
        low = name.lower()
        f: Frame | None = frame
        while f is not None:
            d = f.definition
            if low in d.models or low in d.bins:
                if low in d.models:
                    return self._card(f, low), []
                return None, [self._card(f, b.name.lower()) for b in d.bins[low]]
            f = self._lexical_parent(f)
        return None, []

    def _lexical_parent(self, f: Frame) -> Frame | None:
        """The frame whose definition encloses `f`'s definition (models and
        subcircuits are looked up lexically; parameters dynamically)."""
        target = f.definition.parent
        p = f.parent
        while p is not None and p.definition is not target:
            p = p.parent
        if p is None and target is not None:
            return self.root if target is self.deck.top else None
        return p

    def _card(self, f: Frame, low: str) -> ModelCard:
        if low not in f.cards:
            f.cards[low] = evaluate_model(f.definition.models[low], f.scope)
        return f.cards[low]

    # ---------------------------------------------------------------- flatten

    def run(self) -> list[Elem]:
        self._instantiate(self.root, 0)
        return self.out

    def _instantiate(self, frame: Frame, depth: int) -> None:
        if depth > self.MAX_DEPTH:
            raise NetlistError("subcircuit nesting deeper than "
                               f"{self.MAX_DEPTH} (recursive .subckt?)",
                               frame.definition.loc)
        self._items(frame, frame.definition.body, depth)

    def _items(self, frame: Frame, items: list, depth: int) -> None:
        for item in items:
            if isinstance(item, Conditional):
                self._items(frame, self._branch(frame, item), depth)
                continue
            if item[0] == "param":
                _, name, text, loc = item
                frame.scope.define(name, text, loc)
                continue
            toks, line = item
            letter = toks[0][0].lower()
            if letter == "x":
                self._x(frame, toks, line, depth)
                continue
            positions = self._node_positions(frame, toks, line.loc)
            new = [frame.prefix + toks[0]] + [
                self.node(frame, t) if k in positions else t
                for k, t in enumerate(toks[1:], start=1)]
            self.out.append(Elem(new, line.loc, frame))

    @staticmethod
    def _branch(frame: Frame, block: Conditional) -> list:
        """Items of the first branch whose condition holds in `frame`."""
        for cond, loc, items in block.branches:
            if cond is None or frame.scope.eval(cond, loc) != 0:
                return items
        return []

    _FIXED = {"r": 2, "c": 2, "l": 2, "v": 2, "i": 2, "d": 2, "m": 4, "s": 4,
              "b": 2, "f": 2, "h": 2, "k": 0}

    def _node_positions(self, frame: Frame, toks: list[str], loc: Loc) -> set[int]:
        """Token indices holding node names, per element letter."""
        letter = toks[0][0].lower()
        n = self._FIXED.get(letter)
        if letter in ("e", "g"):
            n = 4
            if len(toks) > 3 and (split_kv(toks[3]) or toks[3].lower().startswith(
                    ("poly", "table"))):
                m = re.match(r"poly\((\d+)\)", toks[3].lower())
                if m:  # E n+ n- POLY(k) c1+ c1- ... coefficients
                    return {1, 2} | set(range(4, 4 + 2 * int(m.group(1))))
                n = 2
        elif letter == "q":
            pos = [t for t in toks[1:] if split_kv(t) is None]
            # Q c b e [s] model: a 4th node only if the 5th token is a model
            n = 4 if len(pos) > 4 and self._is_model(frame, pos[4]) else 3
        elif letter == "n":  # OSDI / Verilog-A device: nodes... model params
            n = next((k for k, t in enumerate(toks[1:]) if split_kv(t) is None
                      and self._is_model(frame, t)), None)
            if n is None:
                raise NetlistError(f"{toks[0]}: no .model found among its "
                                   "tokens", loc)
        if n is None:
            raise NetlistError(f"unsupported element {toks[0]!r}", loc)
        if len(toks) < 1 + n:
            raise NetlistError(f"{toks[0]}: expected {n} nodes", loc)
        return set(range(1, 1 + n))

    def _is_model(self, frame: Frame, name: str) -> bool:
        if name.lower() in ("npn", "pnp"):
            return True
        card, bins = self.model(frame, name, None)
        return card is not None or bool(bins)

    def _x(self, frame: Frame, toks: list[str], line: Line, depth: int) -> None:
        pos = [t for t in toks[1:] if split_kv(t) is None
               and t.lower() not in ("params:", "param:")]
        kws = [split_kv(t) for t in toks[1:] if split_kv(t) is not None]
        if not pos:
            raise NetlistError(f"{toks[0]}: missing subcircuit name", line.loc)
        sub_name = pos[-1].lower()
        sub = frame.definition.find_subckt(sub_name) or \
            self.deck.top.find_subckt(sub_name)
        if sub is None:
            raise NetlistError(f"unknown subcircuit {pos[-1]!r} in {toks[0]}",
                               line.loc)
        nodes = pos[:-1]
        if len(nodes) != len(sub.pins):
            raise NetlistError(f"{toks[0]}: {len(nodes)} pins given, subcircuit "
                               f"{sub_name!r} has {len(sub.pins)}", line.loc)
        scope = Scope(frame.scope, frame.prefix + toks[0])
        self._define(sub, scope)
        mult = frame.mult
        known = {k for k, _ in sub.defaults} | {k for k, _, _ in sub.params}
        for key, text in kws:
            value = frame.scope.eval(text, line.loc)  # caller's scope
            if key == "m" and "m" not in known:
                mult *= value
                continue
            if key not in known:
                raise NetlistError(f"{toks[0]}: subcircuit {sub_name!r} has no "
                                   f"parameter {key!r}", line.loc)
            scope.set(key, value)
        pins = {}
        for p, a in zip(sub.pins, nodes):
            pins[p] = self.node(frame, a)
            pins.setdefault(p.lower(), pins[p])
        child = Frame(sub, scope, frame.prefix + toks[0] + ".", pins, mult, frame)
        self._instantiate(child, depth + 1)
