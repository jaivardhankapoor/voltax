"""Verilog-A lexer, AST and recursive-descent parser (the compact-model subset).

Supported (enough for BSIM4 / BSIM-CMG-style models):

* ``module`` with ports, ``inout``/``input``/``output``, ``electrical`` nodes,
  ``ground``, named ``branch`` declarations;
* ``parameter`` / ``localparam`` ``real|integer|string`` with ``from``
  ranges and ``exclude``; ``aliasparam``; ``(* type="instance" *)``
  attributes mark instance parameters;
* ``real`` / ``integer`` / ``genvar`` variables (scalars);
* ``analog function`` definitions (``input`` / ``output`` / ``inout``);
* ``analog`` block statements: ``begin``/``end`` (optionally named, with
  local declarations), assignments, ``if``/``else``, ``case``/``endcase``,
  ``for``, ``while``, ``repeat``, contributions ``I(..) <+`` / ``V(..) <+``,
  system tasks (``$strobe``, ...), event controls ``@(initial_step)`` etc.;
* expressions with the full Verilog operator set and ``?:``.

``discipline`` / ``nature`` blocks are skipped (``electrical`` is built in).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .preprocess import Line, VAError

# =============================================================================
# AST
# =============================================================================


@dataclass
class Node:
    line: int = field(default=0, kw_only=True)
    file: str = field(default="", kw_only=True, repr=False)


# ------------------------------------------------------------------ expressions


@dataclass
class Num(Node):
    value: float | int


@dataclass
class Str(Node):
    value: str


@dataclass
class Name(Node):
    id: str


@dataclass
class Unary(Node):
    op: str
    a: Node


@dataclass
class Binary(Node):
    op: str
    a: Node
    b: Node


@dataclass
class Ternary(Node):
    cond: Node
    a: Node
    b: Node


@dataclass
class Call(Node):
    name: str
    args: list[Node]


# ------------------------------------------------------------------- statements


@dataclass
class Block(Node):
    name: str | None
    decls: dict[str, str]
    stmts: list[Node]


@dataclass
class Assign(Node):
    target: str
    expr: Node


@dataclass
class Contrib(Node):
    kind: str  # "I" or "V"
    nodes: list[str]  # one branch name or one/two node names
    expr: Node


@dataclass
class If(Node):
    cond: Node
    then: Node
    other: Node | None


@dataclass
class Case(Node):
    expr: Node
    items: list[tuple[list[Node] | None, Node]]  # None labels = default


@dataclass
class For(Node):
    init: Assign
    cond: Node
    step: Assign
    body: Node


@dataclass
class While(Node):
    cond: Node
    body: Node


@dataclass
class Repeat(Node):
    count: Node
    body: Node


@dataclass
class SysTask(Node):
    name: str
    args: list[Node]


@dataclass
class Event(Node):
    kind: str
    body: Node


@dataclass
class Nop(Node):
    pass


# ---------------------------------------------------------------------- module


@dataclass
class Param:
    name: str
    type: str  # "real" | "integer" | "string"
    default: Node
    ranges: list[tuple[str, Node, Node, str]] = field(default_factory=list)
    """``(open_bracket, lo, hi, close_bracket)`` from ``from`` clauses."""
    excludes: list[Any] = field(default_factory=list)
    """Excluded values (`Node`) or ranges (tuples like `ranges`)."""
    instance: bool = False
    local: bool = False
    attrs: dict[str, str] = field(default_factory=dict)
    line: int = 0


@dataclass
class Function:
    name: str
    type: str
    args: list[tuple[str, str]]  # (name, direction) in declaration order
    variables: dict[str, str]  # all local names (incl. args) -> type
    body: Node
    line: int = 0


@dataclass
class Module:
    name: str
    ports: list[str]
    nodes: list[str]  # all electrical nodes, ports first
    ground: list[str]
    branches: dict[str, tuple[str, ...]]
    params: dict[str, Param]
    aliases: dict[str, str]
    variables: dict[str, str]
    functions: dict[str, Function]
    analog: list[Node]
    file: str = ""

    @property
    def internal_nodes(self) -> list[str]:
        return [n for n in self.nodes if n not in self.ports and n not in self.ground]


# =============================================================================
# Lexer
# =============================================================================


@dataclass(frozen=True)
class Tok:
    kind: str  # "id" | "sys" | "num" | "str" | "op" | "attr" | "eof"
    value: Any
    file: str
    line: int


_SCALE = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3, "m": 1e-3, "u": 1e-6,
          "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
_NUM = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?([TGMKkmunpfa](?![A-Za-z0-9_]))?")
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_SYS = re.compile(r"\$[A-Za-z_][A-Za-z0-9_$]*")
_OPS = sorted(["<+", "==", "!=", "<=", ">=", "&&", "||", "**", "<<", ">>", "===",
               "!==", "~^", "^~", "+", "-", "*", "/", "%", "<", ">", "!", "~", "&",
               "|", "^", "?", ":", ";", ",", "(", ")", "[", "]", "{", "}", "=", "@",
               "#", "."], key=len, reverse=True)


def tokenize(lines: list[Line]) -> list[Tok]:
    toks: list[Tok] = []
    pending_attr: list[str] | None = None
    attr_line = (None, 0)
    for ln in lines:
        text, i, n = ln.text, 0, len(ln.text)
        while i < n:
            if pending_attr is not None:
                j = text.find("*)", i)
                if j < 0:
                    pending_attr.append(text[i:])
                    break
                pending_attr.append(text[i:j])
                toks.append(Tok("attr", " ".join(pending_attr), *attr_line))
                pending_attr = None
                i = j + 2
                continue
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if text.startswith("(*", i) and not text.startswith("(*)", i):
                pending_attr = []
                attr_line = (ln.file, ln.lineno)
                i += 2
                continue
            if c == '"':
                j = i + 1
                buf = []
                while j < n and text[j] != '"':
                    if text[j] == "\\" and j + 1 < n:
                        buf.append(text[j:j + 2])
                        j += 2
                        continue
                    buf.append(text[j])
                    j += 1
                if j >= n:
                    raise VAError("unterminated string", ln.file, ln.lineno)
                toks.append(Tok("str", "".join(buf), ln.file, ln.lineno))
                i = j + 1
                continue
            if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
                m = _NUM.match(text, i)
                mant, exp, scale = m.group(1), m.group(2), m.group(3)
                if exp is None and scale is None and "." not in mant:
                    value: float | int = int(mant)
                else:
                    value = float(mant + (exp or "")) * _SCALE.get(scale or "", 1.0)
                toks.append(Tok("num", value, ln.file, ln.lineno))
                i = m.end()
                continue
            if c == "\\":  # escaped identifier
                j = i + 1
                while j < n and not text[j].isspace():
                    j += 1
                toks.append(Tok("id", text[i + 1:j], ln.file, ln.lineno))
                i = j
                continue
            m = _ID.match(text, i)
            if m:
                toks.append(Tok("id", m.group(0), ln.file, ln.lineno))
                i = m.end()
                continue
            m = _SYS.match(text, i)
            if m:
                toks.append(Tok("sys", m.group(0), ln.file, ln.lineno))
                i = m.end()
                continue
            for op in _OPS:
                if text.startswith(op, i):
                    toks.append(Tok("op", op, ln.file, ln.lineno))
                    i += len(op)
                    break
            else:
                raise VAError(f"unexpected character {c!r}", ln.file, ln.lineno)
    if pending_attr is not None:
        raise VAError("unterminated (* attribute", *attr_line)
    last = lines[-1] if lines else Line("<eof>", 0, "")
    toks.append(Tok("eof", None, last.file, last.lineno))
    return toks


def _parse_attr(text: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r'(\w+)\s*=\s*("([^"]*)"|[^,\s]+)', text):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(2)
    return out


# =============================================================================
# Parser
# =============================================================================

_BINARY_LEVELS = [
    ["||"], ["&&"], ["|"], ["^", "~^", "^~"], ["&"], ["==", "!=", "===", "!=="],
    ["<", "<=", ">", ">="], ["<<", ">>"], ["+", "-"], ["*", "/", "%"], ["**"],
]
_DECL_TYPES = {"real": "real", "integer": "integer", "genvar": "integer",
               "string": "string"}


class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.pos = 0
        self.attrs: dict[str, str] = {}

    # ------------------------------------------------------------- helpers

    @property
    def tok(self) -> Tok:
        return self.toks[self.pos]

    def peek(self, k: int = 1) -> Tok:
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def error(self, msg: str, tok: Tok | None = None) -> VAError:
        tok = tok or self.tok
        return VAError(f"{msg} (at {tok.value!r})", tok.file, tok.line)

    def next(self) -> Tok:
        t = self.tok
        self.pos += 1
        return t

    def at(self, value: str, kind: str | None = None) -> bool:
        t = self.tok
        return t.value == value and t.kind in ((kind,) if kind else ("op", "id"))

    def accept(self, value: str) -> bool:
        if self.at(value):
            self.pos += 1
            return True
        return False

    def expect(self, value: str) -> Tok:
        if not self.at(value):
            raise self.error(f"expected {value!r}")
        return self.next()

    def ident(self) -> str:
        t = self.tok
        if t.kind != "id":
            raise self.error("expected an identifier")
        self.pos += 1
        return t.value

    def take_attrs(self) -> dict[str, str]:
        out = {}
        while self.tok.kind == "attr":
            out.update(_parse_attr(self.next().value))
        return out

    def _loc(self, node: Node, tok: Tok) -> Node:
        node.line, node.file = tok.line, tok.file
        return node

    # ---------------------------------------------------------------- file

    def parse_modules(self) -> list[Module]:
        modules = []
        while self.tok.kind != "eof":
            self.take_attrs()
            if self.at("module") or self.at("macromodule"):
                modules.append(self.module())
            elif self.at("discipline") or self.at("nature"):
                end = "end" + self.next().value
                while not self.at(end):
                    if self.tok.kind == "eof":
                        raise self.error(f"missing {end}")
                    self.pos += 1
                self.pos += 1
            else:
                raise self.error("expected 'module'")
        return modules

    def module(self) -> Module:
        start = self.next()
        name = self.ident()
        ports: list[str] = []
        if self.accept("("):
            if not self.at(")"):
                ports.append(self.ident())
                while self.accept(","):
                    ports.append(self.ident())
            self.expect(")")
        self.expect(";")
        mod = Module(name, ports, [], [], {}, {}, {}, {}, {}, [], start.file)
        nodes: dict[str, None] = dict.fromkeys(ports)
        while not self.at("endmodule"):
            if self.tok.kind == "eof":
                raise self.error("missing endmodule")
            attrs = self.take_attrs()
            t = self.tok
            word = t.value if t.kind == "id" else None
            if word in ("inout", "input", "output"):
                self.next()
                self.at("electrical") and self.next()
                self._name_list()
            elif word in ("electrical", "voltage", "current"):
                self.next()
                for n in self._name_list():
                    nodes.setdefault(n, None)
            elif word == "ground":
                self.next()
                mod.ground += self._name_list()
            elif word == "branch":
                self.next()
                self.expect("(")
                ends = [self.ident()]
                if self.accept(","):
                    ends.append(self.ident())
                self.expect(")")
                for b in self._name_list():
                    mod.branches[b] = tuple(ends)
            elif word in ("parameter", "localparam"):
                self.next()
                for p in self.param_decl(local=word == "localparam", attrs=attrs):
                    mod.params[p.name] = p
            elif word == "aliasparam":
                self.next()
                alias = self.ident()
                self.expect("=")
                mod.aliases[alias] = self.ident()
                self.expect(";")
            elif word in _DECL_TYPES:
                self.next()
                for v in self._var_list():
                    mod.variables[v] = _DECL_TYPES[word]
            elif word == "analog":
                self.next()
                if self.at("function"):
                    f = self.function()
                    mod.functions[f.name] = f
                else:
                    mod.analog.append(self.statement())
            elif word == "analog_function":
                raise self.error("unsupported item")
            else:
                raise self.error("unsupported module item")
        self.expect("endmodule")
        mod.nodes = list(nodes)
        for p in mod.ports:
            if p not in nodes:
                raise VAError(f"port {p!r} has no discipline", start.file, start.line)
        return mod

    def _name_list(self) -> list[str]:
        names = [self.ident()]
        while self.accept(","):
            names.append(self.ident())
        self.expect(";")
        return names

    def _var_list(self) -> list[str]:
        names = []
        while True:
            names.append(self.ident())
            if self.at("["):
                raise self.error("array variables are not supported")
            if self.accept("="):  # initializer (ignored: VA vars start at 0)
                self.expr()
            if not self.accept(","):
                break
        self.expect(";")
        return names

    def param_decl(self, local: bool, attrs: dict[str, str]) -> list[Param]:
        ptype = "real"
        if self.tok.kind == "id" and self.tok.value in ("real", "integer", "string"):
            ptype = self.next().value
        out = []
        while True:
            line = self.tok.line
            name = self.ident()
            if self.at("["):
                raise self.error("array parameters are not supported")
            self.expect("=")
            default = self.expr()
            p = Param(name, ptype, default, local=local, attrs=dict(attrs),
                      instance=attrs.get("type") == "instance", line=line)
            while self.at("from") or self.at("exclude"):
                kind = self.next().value
                if self.at("[") or self.at("("):
                    lo_b = self.next().value
                    lo = self._range_bound()
                    self.expect(":")
                    hi = self._range_bound()
                    if not (self.at("]") or self.at(")")):
                        raise self.error("expected ']' or ')'")
                    hi_b = self.next().value
                    rng = (lo_b, lo, hi, hi_b)
                    (p.ranges if kind == "from" else p.excludes).append(rng)
                else:
                    if kind == "from":
                        raise self.error("expected a range after 'from'")
                    p.excludes.append(self.expr())
            out.append(p)
            if not self.accept(","):
                break
        self.expect(";")
        return out

    def _range_bound(self) -> Node:
        return self.expr()

    def function(self) -> Function:
        start = self.expect("function")
        ftype = "real"
        if self.tok.kind == "id" and self.tok.value in ("real", "integer"):
            ftype = self.next().value
        name = self.ident()
        self.expect(";")
        args: list[tuple[str, str]] = []
        variables: dict[str, str] = {name: ftype}
        while True:
            self.take_attrs()
            word = self.tok.value if self.tok.kind == "id" else None
            if word in ("input", "output", "inout"):
                self.next()
                for n in self._name_list():
                    args.append((n, word))
                    variables.setdefault(n, "real")
            elif word in _DECL_TYPES:
                self.next()
                for v in self._var_list():
                    variables[v] = _DECL_TYPES[word]
            elif word == "parameter":
                raise self.error("parameters in functions are not supported")
            else:
                break
        body = self.statement()
        self.expect("endfunction")
        return Function(name, ftype, args, variables, body, start.line)

    # ----------------------------------------------------------- statements

    def statement(self) -> Node:
        self.take_attrs()
        t = self.tok
        if t.kind == "op":
            if t.value == ";":
                self.next()
                return self._loc(Nop(), t)
            if t.value == "@":
                self.next()
                self.expect("(")
                depth, kind = 1, []
                while depth:
                    tk = self.next()
                    if tk.value == "(":
                        depth += 1
                    elif tk.value == ")":
                        depth -= 1
                        if not depth:
                            break
                    if tk.kind == "eof":
                        raise self.error("unbalanced @(")
                    kind.append(str(tk.value))
                return self._loc(Event("".join(kind), self.statement()), t)
            raise self.error("unexpected token")
        if t.kind == "sys":
            self.next()
            args = self._call_args() if self.at("(") else []
            self.expect(";")
            return self._loc(SysTask(t.value, args), t)
        if t.kind != "id":
            raise self.error("expected a statement")
        word = t.value
        if word == "begin":
            return self.block()
        if word == "if":
            self.next()
            self.expect("(")
            cond = self.expr()
            self.expect(")")
            then = self.statement()
            other = self.statement() if self.accept("else") else None
            return self._loc(If(cond, then, other), t)
        if word in ("case", "casex", "casez"):
            return self.case()
        if word == "for":
            self.next()
            self.expect("(")
            init = self.assignment()
            self.expect(";")
            cond = self.expr()
            self.expect(";")
            step = self.assignment()
            self.expect(")")
            return self._loc(For(init, cond, step, self.statement()), t)
        if word == "while":
            self.next()
            self.expect("(")
            cond = self.expr()
            self.expect(")")
            return self._loc(While(cond, self.statement()), t)
        if word == "repeat":
            self.next()
            self.expect("(")
            count = self.expr()
            self.expect(")")
            return self._loc(Repeat(count, self.statement()), t)
        if self.peek().value == "(" and self.peek().kind == "op":
            # contribution `I(a, b) <+ expr;` (any access function name)
            save = self.pos
            self.next()
            nodes = self._call_args()
            if self.at("<+"):
                self.next()
                expr = self.expr()
                self.expect(";")
                names = []
                for a in nodes:
                    if not isinstance(a, Name):
                        raise self.error("contribution target must be nodes/branch")
                    names.append(a.id)
                kind = {"I": "I", "V": "V", "flow": "I", "potential": "V"}.get(word)
                if kind is None:
                    raise self.error(f"unknown access function {word!r}")
                return self._loc(Contrib(kind, names, expr), t)
            self.pos = save
            raise self.error("function calls are not statements")
        stmt = self.assignment()
        self.expect(";")
        return stmt

    def assignment(self) -> Assign:
        t = self.tok
        target = self.ident()
        if self.at("["):
            raise self.error("indexed assignment is not supported")
        self.expect("=")
        return self._loc(Assign(target, self.expr()), t)

    def block(self) -> Block:
        t = self.expect("begin")
        name = None
        if self.accept(":"):
            name = self.ident()
        decls: dict[str, str] = {}
        stmts: list[Node] = []
        while not self.at("end"):
            if self.tok.kind == "eof":
                raise self.error("missing 'end'")
            self.take_attrs()
            word = self.tok.value if self.tok.kind == "id" else None
            if word in _DECL_TYPES and self.peek().kind == "id":
                self.next()
                for v in self._var_list():
                    decls[v] = _DECL_TYPES[word]
                continue
            if word == "parameter":
                raise self.error("parameters in named blocks are not supported")
            stmts.append(self.statement())
        self.expect("end")
        return self._loc(Block(name, decls, stmts), t)

    def case(self) -> Case:
        t = self.next()
        self.expect("(")
        expr = self.expr()
        self.expect(")")
        items: list[tuple[list[Node] | None, Node]] = []
        while not self.at("endcase"):
            if self.tok.kind == "eof":
                raise self.error("missing endcase")
            if self.accept("default"):
                self.accept(":")
                items.append((None, self.statement()))
                continue
            labels = [self.expr()]
            while self.accept(","):
                labels.append(self.expr())
            self.expect(":")
            items.append((labels, self.statement()))
        self.expect("endcase")
        return self._loc(Case(expr, items), t)

    # ---------------------------------------------------------- expressions

    def expr(self) -> Node:
        t = self.tok
        cond = self.binary(0)
        if self.accept("?"):
            a = self.expr()
            self.expect(":")
            b = self.expr()
            return self._loc(Ternary(cond, a, b), t)
        return cond

    def binary(self, level: int) -> Node:
        if level == len(_BINARY_LEVELS):
            return self.unary()
        t = self.tok
        ops = _BINARY_LEVELS[level]
        if ops == ["**"]:  # right associative
            a = self.unary()
            if self.tok.kind == "op" and self.tok.value == "**":
                self.next()
                return self._loc(Binary("**", a, self.binary(level)), t)
            return a
        a = self.binary(level + 1)
        while self.tok.kind == "op" and self.tok.value in ops:
            op = self.next().value
            a = self._loc(Binary(op, a, self.binary(level + 1)), t)
        return a

    def unary(self) -> Node:
        t = self.tok
        if t.kind == "op" and t.value in ("-", "+", "!", "~"):
            self.next()
            return self._loc(Unary(t.value, self.unary()), t)
        return self.primary()

    def _call_args(self) -> list[Node]:
        self.expect("(")
        args: list[Node] = []
        if not self.at(")"):
            args.append(self.expr())
            while self.accept(","):
                args.append(self.expr())
        self.expect(")")
        return args

    def primary(self) -> Node:
        t = self.tok
        if t.kind == "num":
            self.next()
            return self._loc(Num(t.value), t)
        if t.kind == "str":
            self.next()
            return self._loc(Str(t.value), t)
        if t.kind == "op" and t.value == "(":
            self.next()
            e = self.expr()
            self.expect(")")
            return e
        if t.kind == "sys":
            self.next()
            args = self._call_args() if self.at("(") else []
            return self._loc(Call(t.value, args), t)
        if t.kind == "id":
            self.next()
            if self.at("(") and self.tok.kind == "op":
                return self._loc(Call(t.value, self._call_args()), t)
            if self.at("["):
                raise self.error("indexing is not supported")
            return self._loc(Name(t.value), t)
        raise self.error("expected an expression")


def parse(lines: list[Line]) -> list[Module]:
    """Parse preprocessed lines into modules."""
    return Parser(tokenize(lines)).parse_modules()
