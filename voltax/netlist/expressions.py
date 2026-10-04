"""The SPICE expression language: parser, parameter scopes, evaluators.

Expressions appear in ``.param`` lines, ``{...}`` / ``'...'`` values,
model cards, instance parameters and behavioral sources. They are parsed by
a small recursive-descent parser (no Python ``eval``) into a tuple AST:

=================================  =========================================
``("num", 1.5)``                   number (SPICE suffixes resolved)
``("id", "w")``                    identifier (parameter, ``time``, ``pi``)
``("call", "max", (a, b))``        function call
``("neg", a)`` / ``("not", a)``    unary minus / logical not
``("bin", "+", a, b)``             binary operator
``("tern", c, a, b)``              ``c ? a : b``
``("v", "a", "b")``                node voltage ``v(a, b)`` (``b`` may be None)
``("i", "Vx")``                    current through voltage source ``Vx``
=================================  =========================================

Precedence, lowest first: ``?:``, ``||``, ``&&``, ``== !=``,
``< <= > >=``, ``+ -``, ``* / %``, unary ``- + !``, ``** ^`` (right
associative, binding tighter than a unary minus on its left, as in Python).

`Scope` resolves parameters lazily (so definition order does not matter)
and reports cycles. `evaluate` computes a float; `compile_jax` turns an
expression with node voltages and branch currents into a hashable
`CompiledExpr` that `voltax.BehavioralSource` evaluates with `jnp`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from .lexer import Loc, NetlistError, scaled

AST = tuple

# =============================================================================
# Tokenizer
# =============================================================================

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[A-Za-zµ]*)
  | (?P<probe>[vViI]\s*\()
  | (?P<id>[A-Za-z_][\w.$#]*)
  | (?P<op>\*\*|==|!=|<=|>=|&&|\|\||[-+*/^%<>!?:(),{}=])
""", re.VERBOSE)


def _tokenize(text: str, loc: Loc | None) -> list[tuple[str, Any]]:
    toks: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise NetlistError(f"unexpected character {text[pos]!r} in expression "
                               f"{text!r}", loc)
        kind = m.lastgroup
        value = m.group(0)
        if kind == "num":
            num = re.match(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", value).group(0)
            toks.append(("num", scaled(num, value[len(num):])))
        elif kind == "probe":
            # v(a) / v(a,b) / i(Vx): the arguments are names, not expressions
            close = text.find(")", m.end())
            if close < 0:
                raise NetlistError(f"unbalanced parenthesis in {text!r}", loc)
            names = [a.strip() for a in text[m.end():close].split(",")]
            toks.append(("probe", (value[0].lower(), names)))
            pos = close + 1
            continue
        elif kind == "id":
            toks.append(("id", value))
        elif kind == "op":
            toks.append(("op", {"{": "(", "}": ")", "=": "=="}.get(value, value)
                         if value in "{}" else value))
        pos = m.end()
    return toks


# =============================================================================
# Parser
# =============================================================================


class _Parser:
    def __init__(self, text: str, loc: Loc | None):
        self.text, self.loc = text, loc
        self.toks = _tokenize(text, loc)
        self.pos = 0

    def error(self, msg: str) -> NetlistError:
        return NetlistError(f"{msg} in expression {self.text!r}", self.loc)

    def peek(self) -> tuple[str, Any] | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def accept(self, *ops: str) -> str | None:
        tok = self.peek()
        if tok and tok[0] == "op" and tok[1] in ops:
            self.pos += 1
            return tok[1]
        return None

    def expect(self, op: str) -> None:
        if not self.accept(op):
            tok = self.peek()
            raise self.error(f"expected {op!r}, got "
                             f"{tok[1] if tok else 'end of expression'!r}")

    def parse(self) -> AST:
        if not self.toks:
            raise self.error("empty expression")
        node = self.ternary()
        if self.pos != len(self.toks):
            raise self.error(f"unexpected {self.toks[self.pos][1]!r}")
        return node

    def ternary(self) -> AST:
        cond = self.binary(0)
        if self.accept("?"):
            a = self.ternary()
            self.expect(":")
            b = self.ternary()
            return ("tern", cond, a, b)
        return cond

    _LEVELS = (("||",), ("&&",), ("==", "!="), ("<", "<=", ">", ">="), ("+", "-"),
               ("*", "/", "%"))

    def binary(self, level: int) -> AST:
        if level == len(self._LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while op := self.accept(*self._LEVELS[level]):
            node = ("bin", op, node, self.binary(level + 1))
        return node

    def unary(self) -> AST:
        if self.accept("-"):
            return ("neg", self.unary())
        if self.accept("+"):
            return self.unary()
        if self.accept("!"):
            return ("not", self.unary())
        return self.power()

    def power(self) -> AST:
        base = self.primary()
        if self.accept("**", "^"):
            return ("bin", "**", base, self.unary())
        return base

    def primary(self) -> AST:
        tok = self.peek()
        if tok is None:
            raise self.error("unexpected end")
        self.pos += 1
        kind, value = tok
        if kind == "num":
            return ("num", value)
        if kind == "probe":
            which, names = value
            if which == "v" and 1 <= len(names) <= 2 and all(names):
                return ("v", names[0], names[1] if len(names) == 2 else None)
            if which == "i" and len(names) == 1 and names[0]:
                return ("i", names[0])
            raise self.error(f"bad probe {which}({', '.join(names)})")
        if kind == "id":
            if self.accept("("):
                args: list[AST] = []
                if not self.accept(")"):
                    args.append(self.ternary())
                    while self.accept(","):
                        args.append(self.ternary())
                    self.expect(")")
                return ("call", value.lower(), tuple(args))
            return ("id", value.lower())
        if value == "(":
            node = self.ternary()
            self.expect(")")
            return node
        raise self.error(f"unexpected {value!r}")


def strip_delimiters(text: str) -> str:
    """``{expr}`` / ``'expr'`` / ``"expr"`` -> ``expr``."""
    text = text.strip()
    if len(text) >= 2 and ((text[0] == "{" and text[-1] == "}")
                           or (text[0] == text[-1] and text[0] in "'\"")):
        return text[1:-1].strip()
    return text


def parse_expression(text: str, loc: Loc | None = None) -> AST:
    """Parse an expression (outer ``{}`` or quotes optional) into an AST."""
    return _Parser(strip_delimiters(text), loc).parse()


# =============================================================================
# Functions
# =============================================================================


def _nint(x):
    return math.floor(x + 0.5)


def _pwr(x, y):
    return math.copysign(abs(x) ** y, x)


def _sign(x, y=None):
    if y is None:
        return float((x > 0) - (x < 0))
    return math.copysign(abs(x), y)


def _limit(x, lo, hi):
    return min(max(x, min(lo, hi)), max(lo, hi))


def _nominal(mean, *_):
    return mean


# name -> (numeric implementation, jnp implementation, allowed arg counts)
FUNCTIONS: dict[str, tuple[Callable, Callable, tuple[int, ...]]] = {
    "sqrt": (math.sqrt, jnp.sqrt, (1,)),
    "exp": (math.exp, jnp.exp, (1,)),
    "ln": (math.log, jnp.log, (1,)),
    "log": (math.log, jnp.log, (1,)),
    "log10": (math.log10, jnp.log10, (1,)),
    "pow": (math.pow, jnp.power, (2,)),
    "pwr": (_pwr, lambda x, y: jnp.sign(x) * jnp.abs(x) ** y, (2,)),
    "abs": (abs, jnp.abs, (1,)),
    "min": (min, lambda *a: _reduce(jnp.minimum, a), (2, 3, 4, 5, 6, 7, 8)),
    "max": (max, lambda *a: _reduce(jnp.maximum, a), (2, 3, 4, 5, 6, 7, 8)),
    "sin": (math.sin, jnp.sin, (1,)),
    "cos": (math.cos, jnp.cos, (1,)),
    "tan": (math.tan, jnp.tan, (1,)),
    "asin": (math.asin, jnp.arcsin, (1,)),
    "acos": (math.acos, jnp.arccos, (1,)),
    "atan": (math.atan, jnp.arctan, (1,)),
    "atan2": (math.atan2, jnp.arctan2, (2,)),
    "sinh": (math.sinh, jnp.sinh, (1,)),
    "cosh": (math.cosh, jnp.cosh, (1,)),
    "tanh": (math.tanh, jnp.tanh, (1,)),
    "asinh": (math.asinh, jnp.arcsinh, (1,)),
    "acosh": (math.acosh, jnp.arccosh, (1,)),
    "atanh": (math.atanh, jnp.arctanh, (1,)),
    "floor": (math.floor, jnp.floor, (1,)),
    "ceil": (math.ceil, jnp.ceil, (1,)),
    "int": (math.trunc, jnp.trunc, (1,)),
    "nint": (_nint, lambda x: jnp.floor(x + 0.5), (1,)),
    "sgn": (_sign, jnp.sign, (1,)),
    "sign": (_sign, lambda x, y=None: jnp.sign(x) if y is None
             else jnp.abs(x) * jnp.where(y >= 0, 1.0, -1.0), (1, 2)),
    "limit": (_limit, lambda x, lo, hi: jnp.clip(x, jnp.minimum(lo, hi),
                                                  jnp.maximum(lo, hi)), (3,)),
    "u": (lambda x: float(x > 0), lambda x: jnp.where(x > 0, 1.0, 0.0), (1,)),
    "uramp": (lambda x: max(x, 0.0), lambda x: jnp.maximum(x, 0.0), (1,)),
    # statistical functions evaluate to their nominal (mean) value
    "agauss": (_nominal, None, (2, 3)),
    "gauss": (_nominal, None, (2, 3)),
    "aunif": (_nominal, None, (2, 3)),
    "unif": (_nominal, None, (2, 3)),
}

CONSTANTS = {"pi": math.pi, "hertz": 0.0}


def _reduce(fn, args):
    out = args[0]
    for a in args[1:]:
        out = fn(out, a)
    return out


_NUMERIC_BIN: dict[str, Callable[[float, float], float]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": math.fmod,
    "**": math.pow,
    "<": lambda a, b: float(a < b),
    "<=": lambda a, b: float(a <= b),
    ">": lambda a, b: float(a > b),
    ">=": lambda a, b: float(a >= b),
    "==": lambda a, b: float(a == b),
    "!=": lambda a, b: float(a != b),
    "&&": lambda a, b: float(bool(a) and bool(b)),
    "||": lambda a, b: float(bool(a) or bool(b)),
}

_JNP_BIN: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": jnp.fmod,
    "**": jnp.power,
    "<": lambda a, b: jnp.where(a < b, 1.0, 0.0),
    "<=": lambda a, b: jnp.where(a <= b, 1.0, 0.0),
    ">": lambda a, b: jnp.where(a > b, 1.0, 0.0),
    ">=": lambda a, b: jnp.where(a >= b, 1.0, 0.0),
    "==": lambda a, b: jnp.where(a == b, 1.0, 0.0),
    "!=": lambda a, b: jnp.where(a != b, 1.0, 0.0),
    "&&": lambda a, b: jnp.where((a != 0) & (b != 0), 1.0, 0.0),
    "||": lambda a, b: jnp.where((a != 0) | (b != 0), 1.0, 0.0),
}


# =============================================================================
# Scopes
# =============================================================================


@dataclass
class UserFunction:
    """``.func name(args) = body``, evaluated in its defining `scope`."""

    args: tuple[str, ...]
    body: AST
    scope: "Scope"
    loc: Loc | None


class Scope:
    """Parameter scope: lazily evaluated definitions plus a parent.

    `define` stores an unevaluated expression; `set` stores a value (e.g.
    an instance override, evaluated in the caller's scope). Lookup order:
    values, definitions, parent. Definitions may reference each other in
    any order; cycles raise `NetlistError`.
    """

    def __init__(self, parent: "Scope | None" = None, name: str = "<top>",
                 constants: dict[str, float] | None = None):
        self.parent = parent
        self.name = name
        self.defs: dict[str, tuple[str, Loc | None]] = {}
        self.values: dict[str, float] = {}
        self.funcs: dict[str, UserFunction] = {}
        self.constants = constants if constants is not None else (
            parent.constants if parent else dict(CONSTANTS))
        self._busy: list[str] = []

    def define(self, name: str, text: str, loc: Loc | None = None) -> None:
        name = name.lower()
        self.defs[name] = (text, loc)
        self.values.pop(name, None)

    def set(self, name: str, value: float) -> None:
        self.values[name.lower()] = value

    def define_function(self, name: str, args: tuple[str, ...], body: str,
                        loc: Loc | None = None) -> None:
        self.funcs[name.lower()] = UserFunction(
            tuple(a.lower() for a in args), parse_expression(body, loc), self, loc)

    def has(self, name: str) -> bool:
        s: Scope | None = self
        while s is not None:
            if name in s.values or name in s.defs:
                return True
            s = s.parent
        return name in self.constants

    def owner(self, name: str) -> "Scope | None":
        """The scope that defines `name`."""
        s: Scope | None = self
        while s is not None:
            if name in s.values or name in s.defs:
                return s
            s = s.parent
        return None

    def lookup(self, name: str, loc: Loc | None = None) -> float:
        name = name.lower()
        s = self.owner(name)
        if s is None:
            if name in self.constants:
                return self.constants[name]
            raise NetlistError(f"unknown parameter {name!r}", loc)
        return s._value(name, loc)

    def _value(self, name: str, loc: Loc | None) -> float:
        if name in self.values:
            return self.values[name]
        if name in self._busy:
            cycle = " -> ".join(self._busy[self._busy.index(name):] + [name])
            raise NetlistError(f"parameter cycle: {cycle}", loc)
        text, def_loc = self.defs[name]
        self._busy.append(name)
        try:
            value = evaluate(parse_expression(text, def_loc), self, def_loc or loc)
        finally:
            self._busy.pop()
        self.values[name] = value
        return value

    def function(self, name: str) -> UserFunction | None:
        s: Scope | None = self
        while s is not None:
            if name in s.funcs:
                return s.funcs[name]
            s = s.parent
        return None

    def eval(self, text: str, loc: Loc | None = None) -> float:
        """Evaluate expression text in this scope."""
        return evaluate(parse_expression(text, loc), self, loc)

    def all_values(self) -> dict[str, float]:
        """Every parameter defined directly in this scope that evaluates
        (errors are skipped)."""
        out = {}
        for name in list(self.defs) + list(self.values):
            try:
                out[name] = self._value(name, None)
            except (NetlistError, ArithmeticError, ValueError):
                pass
        return out


# =============================================================================
# Numeric evaluation
# =============================================================================


def evaluate(node: AST, scope: Scope, loc: Loc | None = None) -> float:
    """Evaluate an AST to a float in `scope` (parameters only, no probes)."""
    kind = node[0]
    try:
        if kind == "num":
            return node[1]
        if kind == "id":
            return scope.lookup(node[1], loc)
        if kind == "neg":
            return -evaluate(node[1], scope, loc)
        if kind == "not":
            return float(not evaluate(node[1], scope, loc))
        if kind == "bin":
            op = node[1]
            a = evaluate(node[2], scope, loc)
            if op == "&&" and not a:
                return 0.0
            if op == "||" and a:
                return 1.0
            return float(_NUMERIC_BIN[op](a, evaluate(node[3], scope, loc)))
        if kind == "tern":
            branch = node[2] if evaluate(node[1], scope, loc) else node[3]
            return evaluate(branch, scope, loc)
        if kind == "call":
            return _call(node, scope, loc)
    except ZeroDivisionError:
        raise NetlistError("division by zero in expression", loc) from None
    except (OverflowError, ValueError) as e:
        if isinstance(e, NetlistError):
            raise
        raise NetlistError(f"math error in expression: {e}", loc) from None
    if kind in ("v", "i"):
        raise NetlistError(f"{kind}(...) is only allowed in behavioral sources", loc)
    raise NetlistError(f"cannot evaluate {node!r}", loc)


def _call(node: AST, scope: Scope, loc: Loc | None) -> float:
    _, name, args = node
    if name == "if":
        if len(args) != 3:
            raise NetlistError("if() takes 3 arguments", loc)
        cond = evaluate(args[0], scope, loc)
        return evaluate(args[1] if cond else args[2], scope, loc)
    user = scope.function(name)
    if user is not None:
        if len(args) != len(user.args):
            raise NetlistError(f"{name}() takes {len(user.args)} arguments", loc)
        local = Scope(user.scope, f"{name}()")
        for a, v in zip(user.args, args):
            local.set(a, evaluate(v, scope, loc))
        return evaluate(user.body, local, loc)
    if name not in FUNCTIONS:
        raise NetlistError(f"unknown function {name!r}", loc)
    fn, _, arity = FUNCTIONS[name]
    if len(args) not in arity:
        raise NetlistError(f"{name}() takes {' or '.join(map(str, arity))} "
                           f"arguments, got {len(args)}", loc)
    return float(fn(*(evaluate(a, scope, loc) for a in args)))


# =============================================================================
# Compilation to JAX (behavioral sources)
# =============================================================================


@dataclass(frozen=True)
class CompiledExpr:
    """A behavioral expression as a hashable callable.

    The AST only contains numbers, parameter leaves ``("p", name)``,
    terminal voltages ``("vt", k)``, sensed currents ``("is", k)``, time
    ``("time",)`` and operators, so it can be a static field (equal
    expressions fuse into one element group) while parameter values stay
    differentiable leaves.

    Called as ``expr(vc, isense, t, params)`` with ``vc`` ``(N, n_ctrl)``,
    ``isense`` ``(N, n_sense)`` and ``params`` a dict of ``(N,)`` arrays.
    """

    ast: AST
    text: str = ""

    def __call__(self, vc, isense, t, params):
        return _jeval(self.ast, vc, isense, t, params)

    def __repr__(self) -> str:
        return f"CompiledExpr({self.text!r})"


def _jeval(node: AST, vc, isense, t, params):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "p":
        return params[node[1]]
    if kind == "vt":
        return vc[..., node[1]]
    if kind == "is":
        return isense[..., node[1]]
    if kind == "time":
        return t
    if kind == "neg":
        return -_jeval(node[1], vc, isense, t, params)
    if kind == "not":
        return jnp.where(_jeval(node[1], vc, isense, t, params) == 0, 1.0, 0.0)
    if kind == "bin":
        return _JNP_BIN[node[1]](_jeval(node[2], vc, isense, t, params),
                                 _jeval(node[3], vc, isense, t, params))
    if kind == "tern":
        return jnp.where(_jeval(node[1], vc, isense, t, params) != 0,
                         _jeval(node[2], vc, isense, t, params),
                         _jeval(node[3], vc, isense, t, params))
    if kind == "table":
        x = _jeval(node[1], vc, isense, t, params)
        return jnp.interp(x, jnp.asarray(node[2]), jnp.asarray(node[3]))
    if kind == "call":
        args = [_jeval(a, vc, isense, t, params) for a in node[2]]
        return FUNCTIONS[node[1]][1](*args)
    raise ValueError(f"cannot evaluate {node!r}")


@dataclass
class Compiled:
    """Result of `compile_jax`."""

    expr: CompiledExpr
    params: dict[str, float]
    nodes: list[str]
    """Control nodes, in terminal order (``("vt", k)`` reads ``nodes[k]``)."""
    sensed: list[str]
    """Sensed device names (``("is", k)`` reads ``sensed[k]``)."""


def compile_jax(node: AST, scope: Scope, text: str, loc: Loc | None,
                node_name: Callable[[str], str], device_name: Callable[[str], str],
                ground: Callable[[str], bool], constants: dict[str, float]
                ) -> Compiled:
    """Compile an expression with probes for `voltax.BehavioralSource`.

    Parameters become differentiable leaves (named by the parameter), node
    names go through `node_name` (subcircuit renaming), sensed device names
    through `device_name`; user functions are inlined.
    """
    params: dict[str, float] = {}
    owners: dict[str, Scope] = {}
    nodes: list[str] = []
    sensed: list[str] = []

    def terminal(name: str) -> AST:
        full = node_name(name)
        if ground(full):
            return ("num", 0.0)
        if full not in nodes:
            nodes.append(full)
        return ("vt", nodes.index(full))

    def param(name: str, sc: Scope) -> AST:
        owner = sc.owner(name)
        key = name
        k = 1
        while key in owners and owners[key] is not owner:
            k += 1
            key = f"{name}#{k}"
        owners[key] = owner
        params[key] = sc.lookup(name, loc)
        return ("p", key)

    def walk(n: AST, sc: Scope, env: dict[str, AST]) -> AST:
        kind = n[0]
        if kind == "num":
            return n
        if kind == "id":
            name = n[1]
            if name in env:
                return env[name]
            if name == "time":
                return ("time",)
            if name in constants and sc.owner(name) is None:
                return ("num", constants[name])
            if not sc.has(name):
                raise NetlistError(f"unknown parameter {name!r} in {text!r}", loc)
            if sc.owner(name) is None:
                return ("num", sc.lookup(name, loc))
            return param(name, sc)
        if kind == "v":
            a = terminal(n[1])
            if n[2] is None:
                return a
            return ("bin", "-", a, terminal(n[2]))
        if kind == "i":
            dev = device_name(n[1])
            if dev not in sensed:
                sensed.append(dev)
            return ("is", sensed.index(dev))
        if kind in ("neg", "not"):
            return (kind, walk(n[1], sc, env))
        if kind == "bin":
            return ("bin", n[1], walk(n[2], sc, env), walk(n[3], sc, env))
        if kind == "tern":
            return ("tern", *(walk(c, sc, env) for c in n[1:]))
        if kind == "table":
            return ("table", walk(n[1], sc, env), n[2], n[3])
        if kind == "call":
            name, args = n[1], n[2]
            if name == "if":
                if len(args) != 3:
                    raise NetlistError("if() takes 3 arguments", loc)
                return ("tern", *(walk(a, sc, env) for a in args))
            user = sc.function(name)
            if user is not None:
                if len(args) != len(user.args):
                    raise NetlistError(f"{name}() takes {len(user.args)} "
                                       "arguments", loc)
                inner = {a: walk(v, sc, env) for a, v in zip(user.args, args)}
                return walk(user.body, user.scope, inner)
            if name not in FUNCTIONS:
                raise NetlistError(f"unknown function {name!r}", loc)
            numeric, jfn, arity = FUNCTIONS[name]
            if len(args) not in arity:
                raise NetlistError(f"{name}() takes {' or '.join(map(str, arity))}"
                                   f" arguments, got {len(args)}", loc)
            compiled = tuple(walk(a, sc, env) for a in args)
            if jfn is None:  # statistical: nominal value
                return compiled[0]
            return ("call", name, compiled)
        raise NetlistError(f"cannot compile {n!r}", loc)

    ast = walk(node, scope, {})
    return Compiled(CompiledExpr(ast, text), params, nodes, sensed)
