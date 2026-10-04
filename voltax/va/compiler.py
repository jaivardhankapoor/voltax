"""Verilog-A -> JAX compiler: an online partial evaluator emitting Python.

The analog block is *executed* symbolically. Every value is either

* `S` - **static**: a concrete Python number, known at compile time
  (integer parameters, model-selector flags, ``$param_given``, constants,
  and any real parameter not declared differentiable), or
* `T` - **traced**: a JAX array computed by generated code (terminal
  voltages, internal unknowns, and differentiable parameters).

Operations on static values are folded in Python (with C semantics), so
model-selector branches (``capMod``, ``mobMod``, ``rdsMod``...) are resolved
at compile time and leave no trace in the generated code. Conditions on
traced values become ``jnp.where`` merges (SSA phi nodes) of both branches,
whose live-in values are read through `runtime.guard` (the "double-where"
NaN guard): the untaken branch can compute ``sqrt(-1)`` or ``1/0`` without
poisoning values *or* gradients. Loops whose condition is static are
unrolled; loops with a traced condition are unrolled as nested traced
``if``s while a static part of the condition (an iteration counter) allows.

Because binding times are tracked per program point (not per variable), a
variable can be static on one path and traced on another; the analysis is
exactly as precise as concrete execution for everything static.

The output is a Python module with two pure functions, ``f`` (resistive
contributions) and ``q`` (reactive, the arguments of ``ddt``), each pruned
by dead-code elimination to what its outputs need. Every statement carries a
``# file:line`` comment pointing back into the Verilog-A source.

The compiler also records every conditional / min / max / abs / clamp /
limiter it evaluates, with its source line and what it depends on ("bias",
"inst", "model", "temp"), for the differentiability audit (`voltax.va.audit`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .parser import (
    Assign,
    Binary,
    Block,
    Call,
    Case,
    Contrib,
    Event,
    For,
    Function,
    If,
    Module,
    Name,
    Nop,
    Num,
    Repeat,
    Str,
    SysTask,
    Ternary,
    Unary,
    While,
)
from .preprocess import VAError

GROUND = "<ground>"

# =============================================================================
# Values
# =============================================================================


@dataclass(frozen=True)
class S:
    """A static (compile-time) value: Python ``int``, ``float`` or ``str``."""

    v: Any

    @property
    def kind(self) -> str:
        if isinstance(self.v, str):
            return "str"
        return "int" if isinstance(self.v, (int, bool, np.integer)) else "real"


@dataclass(frozen=True)
class T:
    """A traced value: the SSA variable `name` in the generated code."""

    name: str
    kind: str  # "real" | "int" | "bool"
    deps: frozenset = frozenset()


Value = S | T

# =============================================================================
# Audit records
# =============================================================================


@dataclass
class Site:
    """One evaluated non-smooth construct (for the differentiability audit)."""

    file: str
    line: int
    kind: str
    """``if``, ``ternary``, ``case``, ``loop``, ``min``, ``max``, ``abs``,
    ``clamp``, ``floor``, ``ceil``, ``limexp``, ``int``, ``sign`` ..."""
    deps: frozenset
    detail: str = ""
    macros: tuple[str, ...] = ()
    cond: str | None = None
    """SSA name of the switching condition (traced sites, instrumented builds)."""
    active: str | None = None
    """SSA name of the path condition under which the site is evaluated (None:
    always), for instrumented builds."""
    result: str | None = None
    """SSA name of the value a ``min``/``max``/``abs``/... site produces."""

    @property
    def category(self) -> str:
        if not self.deps:
            return "static"
        if "bias" in self.deps:
            return "branch-on-bias"
        return "branch-on-param"


@dataclass
class Risk:
    """A traced operation that can produce NaN/inf for some inputs."""

    file: str
    line: int
    op: str
    deps: frozenset
    macros: tuple[str, ...] = ()
    result: str | None = None


# =============================================================================
# Code buffer (SSA statements)
# =============================================================================


@dataclass
class Stmt:
    target: str
    code: str
    uses: tuple[str, ...]
    comment: str


class Code:
    def __init__(self) -> None:
        self.stmts: list[Stmt] = []
        self.counter = 0
        self.depth: dict[str, int] = {}
        self.names: set[str] = set()

    def fresh(self, hint: str) -> str:
        hint = re.sub(r"\W", "_", hint)[:40] or "t"
        if hint[0].isdigit():
            hint = "_" + hint
        self.counter += 1
        return f"{hint}_{self.counter}"

    def emit(self, code: str, uses: Iterable[str], hint: str, depth: int,
             comment: str = "") -> str:
        name = self.fresh(hint)
        self.stmts.append(Stmt(name, code, tuple(uses), comment))
        self.depth[name] = depth
        return name

    def live(self, outputs: Iterable[str]) -> list[Stmt]:
        """Statements needed by `outputs` (dead-code elimination)."""
        needed = set(outputs)
        keep = []
        for st in reversed(self.stmts):
            if st.target in needed:
                keep.append(st)
                needed.update(st.uses)
        return keep[::-1]


def lit(x: Any) -> str:
    """Python source for a static value."""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (int, np.integer)):
        return repr(int(x)) if x >= 0 else f"({int(x)!r})"
    x = float(x)
    if math.isnan(x):
        return "jnp.nan"
    if math.isinf(x):
        return "jnp.inf" if x > 0 else "(-jnp.inf)"
    return repr(x) if x >= 0 else f"({x!r})"


# =============================================================================
# Static (C-semantics) arithmetic
# =============================================================================


def _f64(x: Any) -> np.float64:
    return np.float64(x)


def _static_binary(op: str, a: Any, b: Any) -> Any:
    ints = isinstance(a, int) and isinstance(b, int)
    with np.errstate(all="ignore"):
        if op == "+":
            return a + b if ints else float(_f64(a) + _f64(b))
        if op == "-":
            return a - b if ints else float(_f64(a) - _f64(b))
        if op == "*":
            return a * b if ints else float(_f64(a) * _f64(b))
        if op == "/":
            if ints:
                if b == 0:
                    raise ZeroDivisionError("integer division by zero")
                q = abs(a) // abs(b)
                return q if (a >= 0) == (b >= 0) else -q
            return float(_f64(a) / _f64(b))
        if op == "%":
            if ints:
                return int(math.fmod(a, b))
            return float(np.fmod(_f64(a), _f64(b)))
        if op == "**":
            return float(np.power(_f64(a), _f64(b)))
        if op in ("==", "==="):
            return int(a == b)
        if op in ("!=", "!=="):
            return int(a != b)
        if op == "<":
            return int(a < b)
        if op == "<=":
            return int(a <= b)
        if op == ">":
            return int(a > b)
        if op == ">=":
            return int(a >= b)
        if op == "&&":
            return int(bool(a) and bool(b))
        if op == "||":
            return int(bool(a) or bool(b))
        if op in ("&", "|", "^", "<<", ">>"):
            a, b = int(a), int(b)
            return {"&": a & b, "|": a | b, "^": a ^ b, "<<": a << b,
                    ">>": a >> b}[op]
    raise VAError(f"unsupported operator {op!r}")


_MATH1 = {
    "exp": (np.exp, "jnp.exp"), "ln": (np.log, "jnp.log"),
    "log": (np.log10, "jnp.log10"), "sqrt": (np.sqrt, "jnp.sqrt"),
    "abs": (np.abs, "jnp.abs"), "sin": (np.sin, "jnp.sin"),
    "cos": (np.cos, "jnp.cos"), "tan": (np.tan, "jnp.tan"),
    "asin": (np.arcsin, "jnp.arcsin"), "acos": (np.arccos, "jnp.arccos"),
    "atan": (np.arctan, "jnp.arctan"), "sinh": (np.sinh, "jnp.sinh"),
    "cosh": (np.cosh, "jnp.cosh"), "tanh": (np.tanh, "jnp.tanh"),
    "asinh": (np.arcsinh, "jnp.arcsinh"), "acosh": (np.arccosh, "jnp.arccosh"),
    "atanh": (np.arctanh, "jnp.arctanh"), "floor": (np.floor, "jnp.floor"),
    "ceil": (np.ceil, "jnp.ceil"),
}
_MATH1.update({f"${k}": v for k, v in list(_MATH1.items())})
_MATH1["$log10"] = _MATH1["log"]
_RISKY = {"ln", "log", "sqrt", "asin", "acos", "acosh", "atanh", "pow", "/",
          "**", "exp", "sinh", "cosh"}
_NONSMOOTH1 = {"abs", "floor", "ceil"}

_IGNORED_TASKS = {"$strobe", "$display", "$write", "$monitor", "$debug",
                  "$warning", "$info", "$fstrobe", "$fdisplay", "$fwrite",
                  "$fclose", "$fopen"}
_FATAL_TASKS = {"$finish", "$stop", "$error", "$fatal"}
_NOISE = {"white_noise", "flicker_noise", "noise_table", "noise_table_log"}


# =============================================================================
# Configuration of one specialization
# =============================================================================


@dataclass
class Config:
    """Everything that determines the generated code.

    Attributes:
        static: Values of non-differentiable parameters (given ones; others
            take their defaults). Keys are lower-case parameter names.
        traced_model: Model parameters read from the traced ``P`` dict.
        traced_inst: Instance parameters read from ``P`` (per-device arrays).
        given: Names for which ``$param_given`` is true.
        temperature: Static device temperature (K), or None to make
            ``$temperature`` the traced parameter ``"$temperature"``.
        simparams: ``$simparam`` values.
        smooth: Width of smooth min/max/abs/clamp surrogates: a float for
            every such site, ``{line: width}`` for selected source lines of
            the main file, or None (exact).
        collapse: Node -> representative node (from the zero-voltage
            contributions found in a previous pass).
    """

    static: dict[str, Any] = field(default_factory=dict)
    traced_model: frozenset = frozenset()
    traced_inst: frozenset = frozenset()
    given: frozenset = frozenset()
    temperature: float | None = 300.15
    simparams: dict[str, float] = field(default_factory=dict)
    smooth: float | dict[int, float] | None = None
    collapse: dict[str, str] = field(default_factory=dict)
    observe: tuple[str, ...] = ()
    """Variables whose final values the generated ``f`` also returns (as a
    third output, a dict), e.g. ``("Vth", "Vdsat")`` for debugging."""
    instrument: bool = False
    """Also return every traced site's condition and path condition (keys
    ``"#<i>"`` / ``"@<i>"`` of the third output, ``i`` = index in `sites`)."""
    equality: str = "value"
    """Derivatives at traced equality branches ``if (x == c)``: ``"value"``
    (plain AD of the taken branch) or ``"limit"`` (value of the special case,
    derivative of the general branch, like hand-coded SPICE Jacobians)."""


# =============================================================================
# The partial evaluator
# =============================================================================


class _Frame:
    def __init__(self, cond: T | None, depth: int, guard: bool = True):
        self.cond = cond
        self.depth = depth
        self.guard = guard
        self.guards: dict[str, T] = {}
        self.active: str | None = None


class Evaluator:
    """Symbolically execute a module's analog block under a `Config`."""

    MAX_UNROLL = 512

    def __init__(self, module: Module, config: Config,
                 macros_at: Callable[[str, int], tuple[str, ...]] | None = None):
        self.m = module
        self.cfg = config
        self.code = Code()
        self.frames: list[_Frame] = [_Frame(None, 0)]
        self.sites: list[Site] = []
        self.risks: list[Risk] = []
        self.macros_at = macros_at or (lambda f, n: ())
        self.loc = (module.file, 0)
        self.types: dict[str, str] = dict(module.variables)
        self.env: dict[str, Value] = {}
        self.params: dict[str, Value] = {}
        self.inputs: dict[str, str] = {}  # P key -> SSA name
        self.node_inputs: dict[str, str] = {}  # unknown -> SSA name
        self.zero_v: list[tuple[str, str]] = []  # zero-voltage contributions
        self.vbranches: dict[str, tuple[str, str]] = {}  # name -> (a, b)
        self.used_nodes: set[str] = set()
        self.fatal: list[tuple[str, int, str]] = []
        self.call_depth = 0

    # ---------------------------------------------------------- locations

    def _err(self, msg: str) -> VAError:
        return VAError(msg, *self.loc)

    def _site(self, kind: str, deps: frozenset, detail: str = "",
              cond: T | None = None) -> None:
        f, n = self.loc
        active = self._active() if (cond is not None and self.cfg.instrument) \
            else None
        self.sites.append(Site(f, n, kind, deps, detail, self.macros_at(f, n),
                               cond.name if cond is not None else None, active))

    def _active(self) -> str | None:
        """SSA name of the current path condition (None at top level)."""
        prev = None
        for frame in self.frames[1:]:
            if frame.active is None:
                if prev is None:
                    frame.active = frame.cond.name
                else:
                    frame.active = self.code.emit(
                        f"jnp.logical_and({prev}, {frame.cond.name})",
                        (prev, frame.cond.name), "active", frame.depth)
            prev = frame.active
        return prev

    def _cmp(self, code: str, vals: list) -> T | None:
        """Emit a site condition only for instrumented builds."""
        if not self.cfg.instrument:
            return None
        return self.emit(code, self.uses(*vals), "bool", self.deps(*vals), "sw")

    def _smooth(self) -> float | None:
        """Smooth-mode width at the current source location (None: exact)."""
        sm = self.cfg.smooth
        if isinstance(sm, dict):
            f, n = self.loc
            return sm.get(n) if f == self.m.file else None
        return sm

    def _risk(self, op: str, deps: frozenset) -> None:
        f, n = self.loc
        self.risks.append(Risk(f, n, op, deps, self.macros_at(f, n)))

    def _comment(self) -> str:
        f, n = self.loc
        return f"{f.rsplit('/', 1)[-1]}:{n}"

    # ------------------------------------------------------------- frames

    @property
    def depth(self) -> int:
        return len(self.frames) - 1

    def emit(self, code: str, uses: Iterable[str], kind: str, deps: frozenset,
             hint: str = "t") -> T:
        name = self.code.emit(code, uses, hint, self.depth, self._comment())
        return T(name, kind, frozenset(deps))

    def enter(self, v: Value) -> Value:
        """Read `v` from the current frame, guarding outer traced values."""
        if not isinstance(v, T) or self.depth == 0:
            return v
        d0 = self.code.depth.get(v.name, 0)
        for d in range(d0 + 1, self.depth + 1):
            frame = self.frames[d]
            if not frame.guard:
                continue
            if v.name in frame.guards:
                v = frame.guards[v.name]
                continue
            g = T(self.code.emit(f"rt.guard({frame.cond.name}, {v.name})",
                                 (frame.cond.name, v.name), v.name.rsplit("_", 1)[0]
                                 + "_g", d, self._comment()), v.kind, v.deps)
            frame.guards[v.name] = g
            v = g
        return v

    def _push(self, cond: T, guard: bool = True) -> None:
        self.frames.append(_Frame(cond, self.depth + 1, guard))

    def _pop(self) -> None:
        self.frames.pop()

    # --------------------------------------------------------- conversion

    def ref(self, v: Value) -> str:
        return lit(v.v) if isinstance(v, S) else v.name

    def uses(self, *vals: Value) -> list[str]:
        return [v.name for v in vals if isinstance(v, T)]

    def deps(self, *vals: Value) -> frozenset:
        out: frozenset = frozenset()
        for v in vals:
            if isinstance(v, T):
                out = out | v.deps
        return out

    def as_real(self, v: Value) -> Value:
        if isinstance(v, S):
            if v.kind == "str":
                raise self._err("string used as a number")
            return v
        if v.kind == "bool":
            return self.emit(f"rt.to_real({v.name})", [v.name], "real", v.deps)
        return v

    def as_bool(self, v: Value) -> Value:
        if isinstance(v, S):
            return S(int(bool(v.v)))
        if v.kind == "bool":
            return v
        return self.emit(f"{v.name} != 0", [v.name], "bool", v.deps, "c")

    def convert(self, v: Value, typ: str) -> Value:
        """Assignment conversion to a declared ``real``/``integer`` variable."""
        if typ == "string":
            return v
        if isinstance(v, S):
            if typ == "integer" and not isinstance(v.v, int):
                x = float(v.v)
                if not math.isfinite(x):
                    raise self._err(f"cannot convert {x} to integer")
                return S(int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1))
            if typ == "real" and isinstance(v.v, int):
                return S(float(v.v))
            return v
        if typ == "integer" and v.kind == "bool":
            return T(self.as_real(v).name, "int", v.deps)
        v = self.as_real(v)
        if typ == "integer" and v.kind != "int":
            self._site("int", v.deps, "real -> integer rounding")
            return self.emit(f"rt.to_int({v.name})", [v.name], "int", v.deps)
        if typ == "real" and v.kind == "int":
            return T(v.name, "real", v.deps)
        return v

    # ------------------------------------------------------------ setup

    def setup_params(self) -> None:
        """Bind every parameter to a static value or a traced input."""
        for p in self.m.params.values():
            name = p.name
            key = name.lower()
            self.loc = (self.m.file, p.line)
            if key in self.cfg.traced_inst or key in self.cfg.traced_model:
                dep = "inst" if key in self.cfg.traced_inst else "model"
                ssa = self.code.emit(f"P[{key!r}]", (), f"p_{key}", 0,
                                     self._comment())
                self.inputs[key] = ssa
                kind = "int" if p.type == "integer" else "real"
                self.params[name] = T(ssa, kind, frozenset({dep}))
            elif key in self.cfg.static and key in self.cfg.given:
                val = self.cfg.static[key]
                if p.type == "integer":
                    if float(val) != int(round(float(val))):
                        raise self._err(f"integer parameter {name} = {val}")
                    val = int(round(float(val)))
                elif p.type == "real":
                    val = float(val)
                self.params[name] = S(val)
            else:
                v = self.eval(p.default)
                self.params[name] = self.convert(v, p.type)
        for alias, target in self.m.aliases.items():
            self.params[alias] = self.params[target]

    def temperature(self) -> Value:
        if self.cfg.temperature is None:
            key = "$temperature"
            if key not in self.inputs:
                self.inputs[key] = self.code.emit(f"P[{key!r}]", (), "p_temp", 0)
            return T(self.inputs[key], "real", frozenset({"temp"}))
        return S(float(self.cfg.temperature))

    # ------------------------------------------------------------- nodes

    def rep(self, node: str) -> str:
        if node in self.m.ground:
            return GROUND
        if node not in self.m.nodes:
            raise self._err(f"unknown node {node!r}")
        return self.cfg.collapse.get(node, node)

    def vnode(self, node: str) -> Value:
        r = self.rep(node)
        if r == GROUND:
            return S(0.0)
        self.used_nodes.add(r)
        if r not in self.node_inputs:
            self.node_inputs[r] = self.code.emit(f"V[{r!r}]", (), f"V_{r}", 0)
        return self.enter(T(self.node_inputs[r], "real", frozenset({"bias"})))

    def _branch_nodes(self, names: list[str]) -> tuple[str, str, str]:
        """``(a, b, branch_key)`` for ``V(a,b)`` / ``V(a)`` / ``V(br)``."""
        if len(names) == 1 and names[0] in self.m.branches:
            ends = self.m.branches[names[0]]
            a, b = ends[0], (ends[1] if len(ends) > 1 else None)
            key = names[0]
        elif len(names) == 1:
            a, b = names[0], None
            key = f"{a},"
        elif len(names) == 2:
            a, b = names
            key = f"{a},{b}"
        else:
            raise self._err("access functions take one or two arguments")
        if b is None:
            b = self.m.ground[0] if self.m.ground else "<gnd>"
        return a, b, key

    def probe_v(self, names: list[str]) -> Value:
        a, b, _ = self._branch_nodes(names)
        if b != "<gnd>" and self.rep(a) == self.rep(b):
            return S(0.0)
        va = self.vnode(a)
        vb = S(0.0) if b == "<gnd>" else self.vnode(b)
        return self.binop("-", va, vb)

    def probe_i(self, names: list[str]) -> Value:
        a, b, key = self._branch_nodes(names)
        if key not in self.vbranches:
            raise self._err(f"I({','.join(names)}) probe: only branches with "
                            "voltage contributions can be probed")
        unknown = f"i({key})"
        self.used_nodes.add(unknown)
        if unknown not in self.node_inputs:
            self.node_inputs[unknown] = self.code.emit(f"V[{unknown!r}]", (),
                                                       "I_branch", 0)
        return self.enter(T(self.node_inputs[unknown], "real", frozenset({"bias"})))

    # ------------------------------------------------------- expressions

    def binop(self, op: str, a: Value, b: Value) -> Value:
        if isinstance(a, S) and isinstance(b, S):
            if a.kind == "str" or b.kind == "str":
                if op in ("==", "!="):
                    return S(int((a.v == b.v) == (op == "==")))
                raise self._err("string arithmetic is not supported")
            try:
                return S(_static_binary(op, a.v, b.v))
            except ZeroDivisionError as e:
                raise self._err(str(e)) from None
        deps = self.deps(a, b)
        if op in ("&&", "||"):
            a, b = self.as_bool(a), self.as_bool(b)
            fn = "jnp.logical_and" if op == "&&" else "jnp.logical_or"
            return self.emit(f"{fn}({self.ref(a)}, {self.ref(b)})",
                             self.uses(a, b), "bool", deps, "c")
        a, b = self.as_real(a), self.as_real(b)
        ints = a.kind == "int" and b.kind == "int"
        ra, rb = self.ref(a), self.ref(b)
        uses = self.uses(a, b)
        if op in ("==", "!=", "<", "<=", ">", ">=", "===", "!=="):
            pyop = {"===": "==", "!==": "!="}.get(op, op)
            return self.emit(f"{ra} {pyop} {rb}", uses, "bool", deps, "c")
        kind = "int" if ints else "real"
        if op in ("+", "-", "*"):
            # Fold identities that keep graphs small. ``0 * x -> 0`` differs
            # from C only for x = inf/nan, where the model is broken anyway;
            # it removes the many zero binning/temperature terms.
            if op == "*" and (_is_zero(a) or _is_zero(b)):
                return S(0) if ints else S(0.0)
            if op == "+" and _is_zero(b):
                return a
            if op == "+" and _is_zero(a):
                return b
            if op == "-" and _is_zero(b):
                return a
            if op == "-" and _is_zero(a) and isinstance(b, T):
                return self.emit(f"-{b.name}", [b.name], b.kind, b.deps)
            if op == "*" and _is_one(b):
                return a
            if op == "*" and _is_one(a):
                return b
            return self.emit(f"{ra} {op} {rb}", uses, kind, deps)
        if op == "/":
            if _is_zero(a) and not ints:
                return S(0.0)  # 0 / x (see the note on 0 * x above)
            if ints:
                out = self.emit(f"rt.int_div({ra}, {rb})", uses, "int", deps)
            elif _is_one(b):
                return a
            else:
                out = self.emit(f"{ra} / {rb}", uses, "real", deps)
            if isinstance(b, T):
                self._risk("/", deps)
                self.risks[-1].result = out.name
            return out
        if op == "%":
            return self.emit(f"jnp.fmod({ra}, {rb})", uses, kind, deps)
        if op == "**":
            out = self.emit(f"jnp.power({ra}, {rb})", uses, "real", deps)
            self._risk("pow", deps)
            self.risks[-1].result = out.name
            return out
        raise self._err(f"operator {op!r} on traced values is not supported")

    def unop(self, op: str, a: Value) -> Value:
        if isinstance(a, S):
            if op == "-":
                return S(-a.v)
            if op == "+":
                return a
            if op == "!":
                return S(int(not a.v))
            if op == "~":
                return S(~int(a.v))
        if op == "+":
            return a
        if op == "-":
            a = self.as_real(a)
            return self.emit(f"-{a.name}", [a.name], a.kind, a.deps)
        if op == "!":
            a = self.as_bool(a)
            return self.emit(f"jnp.logical_not({a.name})", [a.name], "bool",
                             a.deps, "c")
        raise self._err(f"operator {op!r} on traced values is not supported")

    def eval(self, e) -> Value:
        loc = self.loc
        if getattr(e, "line", 0):
            self.loc = (e.file or self.m.file, e.line)
        try:
            return self._eval(e)
        finally:
            self.loc = loc

    def _eval(self, e) -> Value:
        if isinstance(e, Num):
            return S(e.value)
        if isinstance(e, Str):
            return S(e.value)
        if isinstance(e, Name):
            return self.lookup(e.id)
        if isinstance(e, Unary):
            return self.unop(e.op, self.eval(e.a))
        if isinstance(e, Binary):
            if e.op in ("&&", "||"):  # short-circuit on static operands
                a = self.as_bool(self.eval(e.a))
                if isinstance(a, S):
                    if e.op == "&&" and not a.v:
                        return S(0)
                    if e.op == "||" and a.v:
                        return S(1)
                    return self.as_bool(self.eval(e.b))
                b = self.as_bool(self.eval(e.b))
                if isinstance(b, S):
                    if e.op == "&&":
                        return a if b.v else S(0)
                    return S(1) if b.v else a
                return self.binop(e.op, a, b)
            return self.binop(e.op, self.eval(e.a), self.eval(e.b))
        if isinstance(e, Ternary):
            c = self.as_bool(self.eval(e.cond))
            if isinstance(c, S):
                return self.eval(e.a if c.v else e.b)
            self._site("ternary", c.deps, cond=c)
            return self.choose(c, lambda: self.eval(e.a), lambda: self.eval(e.b))
        if isinstance(e, Call):
            return self.call(e)
        raise self._err(f"unsupported expression {type(e).__name__}")

    def lookup(self, name: str) -> Value:
        if name in self.env:
            return self.enter(self.env[name])
        if name in self.params:
            return self.enter(self.params[name])
        if name == "inf":
            return S(math.inf)
        raise self._err(f"undefined identifier {name!r}")

    def choose(self, c: T, fa: Callable[[], Value], fb: Callable[[], Value]
               ) -> Value:
        """Traced ``c ? fa() : fb()`` with NaN guards on both sides."""
        self._push(c)
        a = fa()
        self._pop()
        nc = self.emit(f"jnp.logical_not({c.name})", [c.name], "bool", c.deps, "c")
        self._push(nc)
        b = fb()
        self._pop()
        return self.merge(c, a, b)

    def merge(self, c: T, a: Value, b: Value, typ: str | None = None) -> Value:
        if isinstance(a, S) and isinstance(b, S) and a.v == b.v and \
                type(a.v) is type(b.v):
            return a
        if isinstance(a, T) and isinstance(b, T) and a.name == b.name:
            return a
        if typ is not None:
            a, b = self.convert(a, typ), self.convert(b, typ)
        else:
            a, b = self.as_real(a), self.as_real(b)
        kind = "int" if (a.kind == "int" and b.kind == "int") else "real"
        if typ == "string":
            raise self._err("string values cannot depend on traced conditions")
        return self.emit(f"jnp.where({c.name}, {self.ref(a)}, {self.ref(b)})",
                         [c.name] + self.uses(a, b), kind,
                         c.deps | self.deps(a, b), "phi")

    # --------------------------------------------------------------- calls

    def call(self, e: Call) -> Value:
        n = e.name
        if n in ("V", "potential"):
            return self.probe_v(self._names(e))
        if n in ("I", "flow"):
            return self.probe_i(self._names(e))
        if n == "$temperature":
            return self.temperature()
        if n == "$vt":
            temp = self.eval(e.args[0]) if e.args else self.temperature()
            return self.binop("*", S(1.3806503e-23 / 1.602176462e-19), temp)
        if n == "$simparam":
            key = self.eval(e.args[0])
            if not isinstance(key, S):
                raise self._err("$simparam name must be static")
            if key.v in self.cfg.simparams:
                return S(self.cfg.simparams[key.v])
            if len(e.args) > 1:
                return self.eval(e.args[1])
            raise self._err(f"$simparam({key.v!r}) has no value or default")
        if n == "$param_given":
            pname = self._names(e)[0]
            if pname not in self.m.params:
                raise self._err(f"$param_given({pname}): not a parameter")
            return S(int(pname.lower() in self.cfg.given))
        if n == "$port_connected":
            return S(1)
        if n == "$mfactor":
            return S(1.0)
        if n == "analysis":
            kinds = [self.eval(a) for a in e.args]
            names = {k.v for k in kinds if isinstance(k, S)}
            if names & {"dc", "static", "tran", "ic", "ac"}:
                raise self._err("analysis() of dc/tran/ac is not supported "
                                "(element code does not know the analysis)")
            return S(0)
        if n in ("ddt", "idt", "ddx", "idtmod", "laplace_nd", "laplace_zp",
                 "zi_nd", "transition", "slew", "absdelay", "last_crossing"):
            raise self._err(f"{n}() is only supported as a top-level term of a "
                            "contribution" if n == "ddt" else
                            f"{n}() is not supported")
        if n in _NOISE:
            return S(0.0)
        if n in ("$limit",):
            return self.eval(e.args[0])
        if n in self.m.functions:
            return self.call_function(self.m.functions[n], e.args)
        args = [self.eval(a) for a in e.args]
        before, before_r = len(self.sites), len(self.risks)
        out = self.math(n, args)
        if isinstance(out, T):
            for rec in self.sites[before:] + self.risks[before_r:]:
                rec.result = rec.result or out.name
        return out

    def _names(self, e: Call) -> list[str]:
        out = []
        for a in e.args:
            if not isinstance(a, Name):
                raise self._err(f"{e.name}() takes node/branch names")
            out.append(a.id)
        return out

    def math(self, n: str, args: list[Value]) -> Value:
        smooth = self._smooth()
        if n in _MATH1 and len(args) == 1:
            npf, jf = _MATH1[n]
            a = args[0]
            base = n.lstrip("$")
            if isinstance(a, S):
                with np.errstate(all="ignore"):
                    out = npf(np.float64(a.v))
                if base in ("floor", "ceil") or (base == "abs" and a.kind == "int"):
                    return S(int(out)) if math.isfinite(out) else S(float(out))
                return S(float(out))
            a = self.as_real(a)
            if base in _NONSMOOTH1:
                cond = self._cmp(f"{a.name} < 0", [a]) if base == "abs" else None
                self._site(base, a.deps, cond=cond)
            if base in _RISKY:
                self._risk(base, a.deps)
            if base == "abs" and smooth:
                return self.emit(f"rt.sabs({a.name}, {smooth!r})", [a.name], "real",
                                 a.deps, "abs")
            kind = a.kind if base == "abs" else "real"
            return self.emit(f"{jf}({a.name})", [a.name], kind, a.deps, base)
        if n in ("pow", "$pow") and len(args) == 2:
            a, b = args
            if isinstance(a, S) and isinstance(b, S):
                with np.errstate(all="ignore"):
                    return S(float(np.power(np.float64(a.v), np.float64(b.v))))
            a, b = self.as_real(a), self.as_real(b)
            self._risk("pow", self.deps(a, b))
            if isinstance(b, S) and float(b.v) == 1.0:
                return a
            return self.emit(f"jnp.power({self.ref(a)}, {self.ref(b)})",
                             self.uses(a, b), "real", self.deps(a, b), "pow")
        if n in ("min", "max", "$min", "$max") and len(args) == 2:
            a, b = args
            base = n.lstrip("$")
            if isinstance(a, S) and isinstance(b, S):
                pick = (min if base == "min" else max)(a.v, b.v)
                kind_int = a.kind == "int" and b.kind == "int"
                return S(pick if kind_int else float(pick))
            a, b = self.as_real(a), self.as_real(b)
            deps = self.deps(a, b)
            self._site(base, deps, cond=self._cmp(
                f"{self.ref(a)} < {self.ref(b)}", [a, b]))
            kind = "int" if a.kind == b.kind == "int" else "real"
            if smooth:
                fn = "rt.smin" if base == "min" else "rt.smax"
                return self.emit(f"{fn}({self.ref(a)}, {self.ref(b)}, {smooth!r})",
                                 self.uses(a, b), "real", deps, base)
            fn = "jnp.minimum" if base == "min" else "jnp.maximum"
            return self.emit(f"{fn}({self.ref(a)}, {self.ref(b)})",
                             self.uses(a, b), kind, deps, base)
        if n in ("atan2", "$atan2", "hypot", "$hypot") and len(args) == 2:
            a, b = args
            base = n.lstrip("$")
            if isinstance(a, S) and isinstance(b, S):
                f = np.arctan2 if base == "atan2" else np.hypot
                return S(float(f(np.float64(a.v), np.float64(b.v))))
            a, b = self.as_real(a), self.as_real(b)
            jf = "jnp.arctan2" if base == "atan2" else "jnp.hypot"
            return self.emit(f"{jf}({self.ref(a)}, {self.ref(b)})", self.uses(a, b),
                             "real", self.deps(a, b), base)
        if n in ("limexp", "$limexp") and len(args) == 1:
            a = args[0]
            if isinstance(a, S):
                with np.errstate(all="ignore"):
                    return S(float(np.exp(np.float64(a.v))))
            self._site("limexp", a.deps, "exp continued linearly above 80",
                       self._cmp(f"{a.name} > 80.0", [a]))
            return self.emit(f"rt.limexp({a.name})", [a.name], "real", a.deps, "lexp")
        raise self._err(f"unsupported function {n}({len(args)} args)")

    def call_function(self, f: Function, arg_nodes: list) -> Value:
        if len(arg_nodes) != len(f.args):
            raise self._err(f"{f.name}() takes {len(f.args)} arguments")
        if self.call_depth > 32:
            raise self._err("analog function recursion is not supported")
        values = [self.eval(a) if d != "output" else None
                  for a, (_, d) in zip(arg_nodes, f.args)]
        saved_env, saved_types = self.env, self.types
        self.env = {v: S(0) if t == "integer" else S(0.0)
                    for v, t in f.variables.items()}
        self.types = dict(f.variables)
        for (name, d), val in zip(f.args, values):
            if val is not None:
                self.env[name] = self.convert(val, f.variables[name])
        self.call_depth += 1
        try:
            self.exec(f.body)
            result = self.env[f.name]
            outs = [(a, self.env[name]) for a, (name, d) in zip(arg_nodes, f.args)
                    if d in ("output", "inout")]
        finally:
            self.call_depth -= 1
            self.env, self.types = saved_env, saved_types
        for a, val in outs:
            if not isinstance(a, Name):
                raise self._err(f"output argument of {f.name}() must be a variable")
            self.assign(a.id, val)
        return result

    # ----------------------------------------------------------- statements

    def assign(self, target: str, v: Value) -> None:
        if target in self.params and target not in self.env:
            raise self._err(f"cannot assign to parameter {target!r}")
        typ = self.types.get(target)
        if typ is None:
            raise self._err(f"assignment to undeclared variable {target!r}")
        self.env[target] = self.convert(v, typ)

    def exec(self, s) -> None:
        loc = self.loc
        if getattr(s, "line", 0):
            self.loc = (s.file or self.m.file, s.line)
        try:
            self._exec(s)
        finally:
            self.loc = loc

    def _exec(self, s) -> None:
        if isinstance(s, Block):
            for v, t in s.decls.items():
                self.types[v] = t
                self.env.setdefault(v, S(0) if t == "integer" else S(0.0))
            for st in s.stmts:
                self.exec(st)
        elif isinstance(s, Assign):
            self.assign(s.target, self.eval(s.expr))
        elif isinstance(s, If):
            self.exec_if(s)
        elif isinstance(s, Contrib):
            self.contribute(s)
        elif isinstance(s, Case):
            self.exec_case(s)
        elif isinstance(s, For):
            self.exec(s.init)
            self.loop(s.cond, lambda: (self.exec(s.body), self.exec(s.step)))
        elif isinstance(s, While):
            self.loop(s.cond, lambda: self.exec(s.body))
        elif isinstance(s, Repeat):
            n = self.eval(s.count)
            if not isinstance(n, S):
                raise self._err("repeat count must be static")
            for _ in range(int(n.v)):
                self.exec(s.body)
        elif isinstance(s, SysTask):
            if s.name in _FATAL_TASKS:
                if self.depth == 0:
                    raise self._err(f"{s.name} reached during compilation")
                self.fatal.append((*self.loc, s.name))
            elif s.name not in _IGNORED_TASKS:
                raise self._err(f"unsupported system task {s.name}")
        elif isinstance(s, Event):
            kind = s.kind.replace(" ", "")
            if kind in ("initial_step", "initial_instance", "initial_model") or \
                    kind.startswith("initial_step("):
                self.exec(s.body)
            elif kind in ("final_step",) or kind.startswith("final_step("):
                pass
            else:
                raise self._err(f"unsupported event @({s.kind})")
        elif isinstance(s, Nop):
            pass
        else:
            raise self._err(f"unsupported statement {type(s).__name__}")

    def exec_if(self, s: If) -> None:
        c = self.as_bool(self.eval(s.cond))
        if isinstance(c, S):
            self._site("if", frozenset())
            if c.v:
                self.exec(s.then)
            elif s.other is not None:
                self.exec(s.other)
            return
        clamp = _clamp_pattern(s)
        if clamp is not None:
            self._site("clamp", c.deps, clamp[1], cond=c)
            width = self._smooth()
            if width:
                var, kind, bound = clamp
                x = self.lookup(var)
                b = self.eval(bound)
                fn = "rt.smax" if kind == "lower" else "rt.smin"
                x, b = self.as_real(x), self.as_real(b)
                v = self.emit(f"{fn}({self.ref(x)}, {self.ref(b)}, "
                              f"{width!r})", self.uses(x, b), "real",
                              self.deps(x, b), var)
                self.assign(var, v)
                return
        else:
            special = _equality_side(s.cond)
            self._site("eq" if special else "if", c.deps,
                       f"special case in the {special} branch" if special else "",
                       cond=c)
            if special and self.cfg.equality == "limit":
                self.traced_if(c, lambda: self.exec(s.then),
                               (lambda: self.exec(s.other)) if s.other is not None
                               else None, limit=special)
                return
        self.traced_if(c, lambda: self.exec(s.then),
                       (lambda: self.exec(s.other)) if s.other is not None else None)

    def traced_if(self, c: T, then: Callable[[], None],
                  other: Callable[[], None] | None, limit: str | None = None
                  ) -> None:
        """Execute both branches and merge the environments.

        `limit` (``"then"``/``"else"``) names the branch that is a special
        case taken on an equality set: merged values come from it there, but
        derivatives come from the other (general) branch.
        """
        before = dict(self.env)
        # in limit mode the general branch is differentiated on the special
        # lanes too, so its inputs must not be gradient-guarded
        self._push(c, guard=limit != "else")
        then()
        self._pop()
        env_then = self.env
        self.env = dict(before)
        if other is not None:
            nc = self.emit(f"jnp.logical_not({c.name})", [c.name], "bool", c.deps,
                           "c")
            self._push(nc, guard=limit != "then")
            other()
            self._pop()
        env_else = self.env
        merged = {}
        for k in set(env_then) | set(env_else):
            a = env_then.get(k, before.get(k))
            b = env_else.get(k, before.get(k))
            if a is None or b is None:
                # declared inside one branch only (block-local): default 0
                a = S(0.0) if a is None else a
                b = S(0.0) if b is None else b
            typ = self.types.get(k) if not k.startswith("@") else None
            m = self.merge(c, a, b, typ)
            if limit is not None and isinstance(m, T) and m.kind == "real":
                special, general = (a, b) if limit == "then" else (b, a)
                special, general = self.as_real(special), self.as_real(general)
                m = self.emit(f"rt.limit_merge({m.name}, {c.name}, "
                              f"{self.ref(general)}, {limit == 'then'})",
                              [m.name, c.name] + self.uses(general), "real",
                              m.deps | self.deps(general), "phi")
            merged[k] = m
        self.env = merged

    def exec_case(self, s: Case) -> None:
        sel = self.eval(s.expr)
        default = None
        chain: list[tuple[Value, Any]] = []
        for labels, body in s.items:
            if labels is None:
                default = body
                continue
            cond: Value = S(0)
            for lab in labels:
                cond = self._or(cond, self.binop("==", sel, self.eval(lab)))
            if isinstance(cond, S) and cond.v and not chain:
                self._site("case", frozenset())
                self.exec(body)
                return
            if isinstance(cond, S) and not cond.v:
                continue
            chain.append((cond, body))
        if not chain:
            self._site("case", frozenset())
            if default is not None:
                self.exec(default)
            return
        for c, _ in chain:
            cb = self.as_bool(c)
            self._site("case", self.deps(c), cond=cb if isinstance(cb, T) else None)

        def run(i: int) -> None:
            if i == len(chain):
                if default is not None:
                    self.exec(default)
                return
            c, body = chain[i]
            c = self.as_bool(c)
            if isinstance(c, S):
                if c.v:
                    self.exec(body)
                else:
                    run(i + 1)
                return
            self.traced_if(c, lambda: self.exec(body), lambda: run(i + 1))

        run(0)

    def _or(self, a: Value, b: Value) -> Value:
        a, b = self.as_bool(a), self.as_bool(b)
        if isinstance(a, S):
            return S(1) if a.v else b
        if isinstance(b, S):
            return S(1) if b.v else a
        return self.binop("||", a, b)

    def loop(self, cond, body: Callable[[], Any], count: int = 0) -> None:
        while True:
            if count > self.MAX_UNROLL:
                raise self._err("loop does not terminate statically (unroll "
                                f"limit {self.MAX_UNROLL})")
            c = self.as_bool(self.eval(cond))
            if isinstance(c, S):
                if not c.v:
                    return
                body()
                count += 1
                continue
            self._site("loop", c.deps, "traced loop condition (unrolled)", cond=c)
            n = count + 1
            self.traced_if(c, lambda: (body(), self.loop(cond, body, n)), None)
            return

    # -------------------------------------------------------- contributions

    def contribute(self, s: Contrib) -> None:
        if s.kind == "V":
            self.contribute_v(s)
            return
        res, react = self.split(s.expr)
        if s.kind == "I":
            a, b, _ = self._branch_nodes(s.nodes)
            ra = self.rep(a)
            rb = GROUND if b == "<gnd>" else self.rep(b)
            if ra == rb:
                return
            for node, sign in ((ra, 1), (rb, -1)):
                if node == GROUND:
                    continue
                self.used_nodes.add(node)
                for acc, val in ((f"@I:{node}", res), (f"@Q:{node}", react)):
                    if isinstance(val, S) and val.v == 0:
                        continue
                    cur = self.enter(self.env.get(acc, S(0.0)))
                    op = "+" if sign > 0 else "-"
                    self.env[acc] = self.binop(op, cur, self.as_real(val))
            return

    def contribute_v(self, s: Contrib) -> None:
        """``V(a, b) <+ expr``: a branch-current unknown, or a node collapse
        for a static ``V(a, b) <+ 0``."""
        a, b, key = self._branch_nodes(s.nodes)
        known = key in self.vbranches
        if not known:  # register first: the RHS may probe I(a, b)
            self.vbranches[key] = (a, b)
        res, react = self.split(s.expr)
        if isinstance(res, S) and isinstance(react, S) and res.v == 0 and \
                react.v == 0 and not known:
            del self.vbranches[key]
            if self.depth > 0:
                raise self._err("a V() <+ 0 contribution under a traced condition "
                                "(bias-dependent topology) is not supported")
            if f"i({key})" in self.node_inputs:
                raise self._err(f"I({key}) probed on a zero-voltage branch")
            self.zero_v.append((a, b))
            return
        if self.depth > 0 and not known:
            raise self._err("first voltage contribution to a branch must not be "
                            "under a traced condition")
        self.used_nodes.add(f"i({key})")
        for acc, val in ((f"@VF:{key}", res), (f"@VQ:{key}", react)):
            if isinstance(val, S) and val.v == 0:
                continue
            cur = self.enter(self.env.get(acc, S(0.0)))
            self.env[acc] = self.binop("+", cur, self.as_real(val))

    def split(self, e) -> tuple[Value, Value]:
        """Evaluate a contribution expression as ``(resistive, reactive)``,
        where the reactive part collects the arguments of ``ddt``."""
        loc = self.loc
        if getattr(e, "line", 0):
            self.loc = (e.file or self.m.file, e.line)
        try:
            if not _has_ddt(e):
                return self.eval(e), S(0.0)
            if isinstance(e, Call) and e.name == "ddt":
                if len(e.args) != 1:
                    raise self._err("ddt() with tolerance arguments is not "
                                    "supported")
                return S(0.0), self.eval(e.args[0])
            if isinstance(e, Unary) and e.op in "+-":
                r, q = self.split(e.a)
                return self.unop(e.op, r), self.unop(e.op, q)
            if isinstance(e, Binary) and e.op in ("+", "-"):
                ra, qa = self.split(e.a)
                rb, qb = self.split(e.b)
                return self.binop(e.op, ra, rb), self.binop(e.op, qa, qb)
            if isinstance(e, Binary) and e.op == "*":
                if _has_ddt(e.a) and _has_ddt(e.b):
                    raise self._err("product of two ddt() terms")
                k_node, d_node = (e.b, e.a) if _has_ddt(e.a) else (e.a, e.b)
                k = self.eval(k_node)
                if isinstance(k, T) and "bias" in k.deps:
                    raise self._err("ddt() multiplied by a bias-dependent factor "
                                    "is not supported (write ddt(k*q))")
                r, q = self.split(d_node)
                return self.binop("*", k, r), self.binop("*", k, q)
            if isinstance(e, Binary) and e.op == "/" and not _has_ddt(e.b):
                k = self.eval(e.b)
                if isinstance(k, T) and "bias" in k.deps:
                    raise self._err("ddt() divided by a bias-dependent factor is "
                                    "not supported")
                r, q = self.split(e.a)
                return self.binop("/", r, k), self.binop("/", q, k)
            raise self._err("unsupported use of ddt() (only linear combinations "
                            "of ddt() terms are supported)")
        finally:
            self.loc = loc

    # ----------------------------------------------------------- driver

    def run(self) -> None:
        for name, typ in self.types.items():
            self.env[name] = S(0) if typ == "integer" else S(0.0)
        self.setup_params()
        for st in self.m.analog:
            self.exec(st)


def _is_zero(v: Value) -> bool:
    return isinstance(v, S) and v.kind != "str" and v.v == 0


def _is_one(v: Value) -> bool:
    return isinstance(v, S) and v.kind != "str" and v.v == 1


def _has_ddt(e) -> bool:
    if isinstance(e, Call):
        return e.name == "ddt" or any(_has_ddt(a) for a in e.args)
    if isinstance(e, Unary):
        return _has_ddt(e.a)
    if isinstance(e, Binary):
        return _has_ddt(e.a) or _has_ddt(e.b)
    if isinstance(e, Ternary):
        return _has_ddt(e.cond) or _has_ddt(e.a) or _has_ddt(e.b)
    return False


def _strip(e) -> str:
    """Structural fingerprint of an expression (ignores locations)."""
    if isinstance(e, Num):
        return repr(e.value)
    if isinstance(e, Name):
        return e.id
    if isinstance(e, Unary):
        return f"({e.op}{_strip(e.a)})"
    if isinstance(e, Binary):
        return f"({_strip(e.a)}{e.op}{_strip(e.b)})"
    if isinstance(e, Call):
        return f"{e.name}(" + ",".join(_strip(a) for a in e.args) + ")"
    if isinstance(e, Ternary):
        return f"({_strip(e.cond)}?{_strip(e.a)}:{_strip(e.b)})"
    return repr(e)


def _equality_side(cond) -> str | None:
    """``"then"`` for ``if (a == b)`` (or a conjunction of equalities, e.g.
    ``(Rds == 0) && (Lambda == 1)``), ``"else"`` for ``if (a != b)`` (or a
    disjunction of inequalities): the branch taken on a measure-zero set."""
    if isinstance(cond, Binary):
        if cond.op in ("==", "==="):
            return "then"
        if cond.op in ("!=", "!=="):
            return "else"
        if cond.op in ("&&", "||"):
            a, b = _equality_side(cond.a), _equality_side(cond.b)
            want = "then" if cond.op == "&&" else "else"
            if a == b == want:
                return want
    return None


def _clamp_pattern(s: If) -> tuple[str, str, Any] | None:
    """Recognize ``if (x < b) x = b;`` (lower clamp) and ``if (x > b) x = b;``
    (upper clamp), with either operand order and no else branch."""
    if s.other is not None:
        return None
    body = s.then
    if isinstance(body, Block):
        if len(body.stmts) != 1 or body.decls:
            return None
        body = body.stmts[0]
    if not isinstance(body, Assign) or not isinstance(s.cond, Binary):
        return None
    op, a, b = s.cond.op, s.cond.a, s.cond.b
    if op not in ("<", "<=", ">", ">="):
        return None
    x = body.target
    if isinstance(a, Name) and a.id == x and _strip(b) == _strip(body.expr):
        kind = "lower" if op in ("<", "<=") else "upper"
        return x, kind, b
    if isinstance(b, Name) and b.id == x and _strip(a) == _strip(body.expr):
        kind = "lower" if op in (">", ">=") else "upper"
        return x, kind, a
    return None


# =============================================================================
# Driver: collapse fixed point, output assembly, code rendering
# =============================================================================


@dataclass
class Compiled:
    """The result of compiling one module under one `Config`.

    Attributes:
        terminals: Port names (the element's terminals).
        internal: Names of the internal unknowns: internal node voltages
            (node names) and branch currents of voltage contributions
            (``"i(<branch>)"``).
        source: Generated Python source (functions ``f`` and ``q``).
        f, q: The compiled functions ``(V, P) -> (terminal_rows,
            internal_rows)``; `V` maps unknown names to arrays, `P` maps
            traced parameter names to arrays.
        inputs: Keys of `P` the code reads.
        sites, risks: Audit records (see `Site`, `Risk`).
        collapse: Node -> representative after collapsing ``V(a,b) <+ 0``.
        n_statements: Statements in ``f`` and ``q`` after dead-code elimination.
        live: SSA names that reach the outputs of ``f`` or ``q`` (a site whose
            condition/result is not live cannot affect the model).
        kcl_rows: Which internal rows are node KCL equations (scaled by the
            multiplicity ``m``) rather than branch voltage equations.
    """

    module: Module
    config: Config
    terminals: tuple[str, ...]
    internal: tuple[str, ...]
    source: str
    f: Callable
    q: Callable
    inputs: tuple[str, ...]
    sites: list[Site]
    risks: list[Risk]
    collapse: dict[str, str]
    n_statements: tuple[int, int]
    kcl_rows: tuple[bool, ...] = ()
    fatal: list = field(default_factory=list)
    live: frozenset = frozenset()

    def site_live(self, site: Site) -> bool:
        names = [n for n in (site.cond, site.result) if n is not None]
        return not names or any(n in self.live for n in names)


def _collapse_map(module: Module, pairs: list[tuple[str, str]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def rank(x: str) -> int:
        if x in module.ground or x == "<gnd>":
            return 2
        return 1 if x in module.ports else 0

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank(ra) == rank(rb) and rank(ra) > 0:
            raise VAError(f"V({a},{b}) <+ 0 shorts two ports/ground; not supported",
                          module.file)
        if rank(ra) < rank(rb):
            ra, rb = rb, ra
        parent[rb] = ra
    out = {}
    for n in module.nodes:
        r = find(n)
        if r != n:
            out[n] = GROUND if r == "<gnd>" or r in module.ground else r
    return out


def compile_module(module: Module, config: Config,
                   macros_at: Callable[[str, int], tuple[str, ...]] | None = None,
                   name: str = "va") -> Compiled:
    """Partially evaluate `module` under `config` and generate code."""
    collapse: dict[str, str] = {}
    for _ in range(6):
        cfg = Config(**{**config.__dict__, "collapse": collapse})
        ev = Evaluator(module, cfg, macros_at)
        ev.run()
        new = _collapse_map(module, ev.zero_v)
        if new == collapse:
            break
        collapse = new
    else:  # pragma: no cover
        raise VAError("node collapsing did not reach a fixed point", module.file)
    return _assemble(ev, module, cfg, name)


def _assemble(ev: Evaluator, module: Module, cfg: Config, name: str) -> Compiled:
    ev.loc = (module.file, 0)
    ports = tuple(module.ports)
    for p in ports:
        if cfg.collapse.get(p, p) != p:
            raise VAError(f"port {p!r} collapsed into another node", module.file)
    internal_nodes = [n for n in module.internal_nodes
                      if cfg.collapse.get(n, n) == n and n in ev.used_nodes]
    branches = list(ev.vbranches)
    internal = tuple(internal_nodes) + tuple(f"i({k})" for k in branches)

    f_rows: dict[str, Value] = {}
    q_rows: dict[str, Value] = {}
    for node in list(ports) + internal_nodes:
        f_rows[node] = ev.env.get(f"@I:{node}", S(0.0))
        q_rows[node] = ev.env.get(f"@Q:{node}", S(0.0))
    for key in branches:
        a, b = ev.vbranches[key]
        unknown = f"i({key})"
        if unknown not in ev.node_inputs:
            ev.node_inputs[unknown] = ev.code.emit(f"V[{unknown!r}]", (),
                                                   "I_branch", 0)
        x = T(ev.node_inputs[unknown], "real", frozenset({"bias"}))
        for node, sign in ((ev.rep(a), "+"),
                           (GROUND if b == "<gnd>" else ev.rep(b), "-")):
            if node != GROUND:
                f_rows[node] = ev.binop(sign, f_rows.get(node, S(0.0)), x)
        nodes = [key] if key in module.branches else [n for n in key.split(",") if n]
        vab = ev.probe_v(nodes)
        f_rows[unknown] = ev.binop("-", vab, ev.env.get(f"@VF:{key}", S(0.0)))
        q_rows[unknown] = ev.unop("-", ev.env.get(f"@VQ:{key}", S(0.0)))

    order = list(ports) + list(internal)
    observed: dict[str, Value] = {}
    for var in cfg.observe:
        if var not in ev.env:
            raise VAError(f"cannot observe {var!r}: not a variable", module.file)
        observed[var] = ev.as_real(ev.env[var])
    if cfg.instrument:
        for i, site in enumerate(ev.sites):
            if site.cond is not None:
                observed[f"#{i}"] = T(site.cond, "bool", site.deps)
            if site.active is not None:
                observed[f"@{i}"] = T(site.active, "bool", site.deps)
    src_f, n_f = _render("f", ev, [f_rows[k] for k in order], len(ports), observed)
    src_q, n_q = _render("q", ev, [q_rows[k] for k in order], len(ports))
    fname = module.file.rsplit("/", 1)[-1]
    header = (f'"""Generated by voltax.va from {fname} (module {module.name}).\n\n'
              f"Unknowns: {', '.join(order)}\n"
              f"Traced inputs P: {', '.join(sorted(ev.inputs)) or '(none)'}\n"
              '"""\n\nimport jax.numpy as jnp\n\nfrom voltax.va import runtime as rt\n')
    source = header + "\n\n" + src_f + "\n\n\n" + src_q + "\n"
    namespace: dict[str, Any] = {}
    filename = f"<voltax.va:{name}>"
    import linecache

    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    kcl = tuple(not k.startswith("i(") for k in internal)
    rows = list(f_rows.values()) + list(q_rows.values())
    live = frozenset(st.target for st in ev.code.live(
        [v.name for v in rows if isinstance(v, T)]))
    return Compiled(module, cfg, ports, internal, source, namespace["f"],
                    namespace["q"], tuple(sorted(ev.inputs)), ev.sites, ev.risks,
                    dict(cfg.collapse), (n_f, n_q), kcl, ev.fatal, live)


def _render(fname: str, ev: Evaluator, rows: list[Value], n_ports: int,
            observed: dict[str, Value] | None = None) -> tuple[str, int]:
    observed = observed or {}
    outs = [v.name for v in list(rows) + list(observed.values()) if isinstance(v, T)]
    stmts = ev.code.live(outs)
    lines = [f"def {fname}(V, P):"]
    for st in stmts:
        comment = f"  # {st.comment}" if st.comment else ""
        lines.append(f"    {st.target} = {st.code}{comment}")

    def ref(v: Value) -> str:
        return lit(float(v.v)) if isinstance(v, S) else v.name

    def tup(vals: list[Value]) -> str:
        inner = ", ".join(ref(v) for v in vals)
        return f"({inner},)" if len(vals) == 1 else f"({inner})"

    ret = f"    return {tup(rows[:n_ports])}, {tup(rows[n_ports:])}"
    if observed:
        ret += ", {" + ", ".join(f"{k!r}: {ref(v)}" for k, v in observed.items()) + "}"
    lines.append(ret)
    return "\n".join(lines), len(stmts)
