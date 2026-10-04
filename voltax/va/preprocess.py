"""Verilog-A preprocessor: comments, `` `define`` / `` `include`` / `` `ifdef``.

The output is a list of `Line` objects, one per *source* line, so every token
the parser sees carries the file and line it came from (macro expansions stay
on the line of the macro call). Multi-line macro definitions and multi-line
macro calls are joined and padded with empty lines to keep numbering intact.

``disciplines.vams`` and ``constants.vams`` are built in (see `BUILTIN_FILES`):
the standard Accellera definitions, so models compile without the simulator's
include directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


class VAError(Exception):
    """A Verilog-A source error (parse, preprocess or compile), with location."""

    def __init__(self, message: str, file: str | None = None,
                 line: int | None = None):
        self.file, self.line = file, line
        where = f"{file}:{line}: " if file is not None else ""
        super().__init__(where + message)


@dataclass(frozen=True)
class Line:
    file: str
    lineno: int
    text: str
    macros: tuple[str, ...] = ()
    """Macros expanded on this line (outermost first), for diagnostics."""


# Accellera constants.vams (LRM 2.4), selecting the default (NIST 1998) set
# unless one of the PHYSICAL_CONSTANTS_* macros is defined, like the original.
_CONSTANTS = r"""
`define M_E 2.7182818284590452354
`define M_LOG2E 1.4426950408889634074
`define M_LOG10E 0.43429448190325182765
`define M_LN2 0.69314718055994530942
`define M_LN10 2.30258509299404568402
`define M_PI 3.14159265358979323846
`define M_TWO_PI 6.28318530717958647693
`define M_PI_2 1.57079632679489661923
`define M_PI_4 0.78539816339744830962
`define M_1_PI 0.31830988618379067154
`define M_2_PI 0.63661977236758134308
`define M_2_SQRTPI 1.12837916709551257390
`define M_SQRT2 1.41421356237309504880
`define M_SQRT1_2 0.70710678118654752440
`define P_Q_SPICE 1.60219e-19
`define P_Q_OLD 1.6021918e-19
`define P_Q_NIST1998 1.602176462e-19
`define P_Q_NIST2010 1.602176565e-19
`define P_C 2.99792458e8
`define P_K_SPICE 1.38062e-23
`define P_K_OLD 1.3806226e-23
`define P_K_NIST1998 1.3806503e-23
`define P_K_NIST2010 1.3806488e-23
`define P_H_SPICE 6.62620e-34
`define P_H_OLD 6.6260755e-34
`define P_H_NIST1998 6.62606876e-34
`define P_H_NIST2010 6.62606957e-34
`define P_EPS0_SPICE 8.854214871e-12
`define P_EPS0_OLD 8.85418792394420013968e-12
`define P_EPS0_NIST1998 8.854187817e-12
`define P_EPS0_NIST2010 8.854187817e-12
`define P_U0 (4.0e-7 * `M_PI)
`define P_CELSIUS0 273.15
`ifdef PHYSICAL_CONSTANTS_SPICE
`define P_Q `P_Q_SPICE
`define P_K `P_K_SPICE
`define P_H `P_H_SPICE
`define P_EPS0 `P_EPS0_SPICE
`elsif PHYSICAL_CONSTANTS_OLD
`define P_Q `P_Q_OLD
`define P_K `P_K_OLD
`define P_H `P_H_OLD
`define P_EPS0 `P_EPS0_OLD
`elsif PHYSICAL_CONSTANTS_NIST2010
`define P_Q `P_Q_NIST2010
`define P_K `P_K_NIST2010
`define P_H `P_H_NIST2010
`define P_EPS0 `P_EPS0_NIST2010
`else
`define P_Q `P_Q_NIST1998
`define P_K `P_K_NIST1998
`define P_H `P_H_NIST1998
`define P_EPS0 `P_EPS0_NIST1998
`endif
"""

BUILTIN_FILES = {
    "disciplines.vams": "",  # `electrical` is built into the parser
    "discipline.h": "",
    "constants.vams": _CONSTANTS,
    "constants.h": _CONSTANTS,
}
"""Include files provided by the compiler (searched after the user's paths)."""


@dataclass
class Macro:
    name: str
    params: tuple[str, ...] | None  # None: object-like macro
    body: str


@dataclass
class _State:
    macros: dict[str, Macro] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)
    include_dirs: list[Path] = field(default_factory=list)
    depth: int = 0
    used: list[str] = field(default_factory=list)


def strip_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments (outside strings), keeping newlines."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise VAError("unterminated /* comment")
            out.append("\n" * text.count("\n", i, j))
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def _split_args(text: str, start: int) -> tuple[list[str], int]:
    """Parse ``(a, b(c, d), "x,y")`` starting at ``text[start] == '('``.

    Returns the argument strings and the index after the closing paren.
    """
    depth, i, cur, args = 0, start, [], []
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            cur.append(c)
            if c == '"' and text[i - 1] != "\\":
                in_str = False
        elif c == '"':
            in_str = True
            cur.append(c)
        elif c in "([{":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i + 1
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    raise VAError("unbalanced parentheses in macro call")


def _substitute(body: str, params: tuple[str, ...], args: list[str]) -> str:
    """Replace formal parameters (as whole identifiers, outside strings)."""
    mapping = dict(zip(params, args))
    out, i = [], 0
    in_str = False
    while i < len(body):
        c = body[i]
        if in_str:
            out.append(c)
            if c == '"' and body[i - 1] != "\\":
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        m = _IDENT.match(body, i)
        if m and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] in "_$`")):
            word = m.group(0)
            out.append(f"({mapping[word]})" if word in mapping and _needs_parens(
                mapping[word]) else mapping.get(word, word))
            i = m.end()
            continue
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _needs_parens(arg: str) -> bool:
    # Verilog macros substitute text verbatim; keep that semantics (no
    # parenthesization) so expansions read exactly like the original code.
    return False


def expand(text: str, state: _State, depth: int = 0) -> str:
    """Expand every `` `macro`` use in `text` (outside strings)."""
    if "`" not in text:
        return text
    if depth > 64:
        raise VAError("macro expansion too deep (recursive macro?)")
    out, i = [], 0
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if c == '"' and text[i - 1] != "\\":
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c != "`":
            out.append(c)
            i += 1
            continue
        m = _IDENT.match(text, i + 1)
        if not m:
            raise VAError(f"stray backtick in {text.strip()!r}")
        name = m.group(0)
        macro = state.macros.get(name)
        if macro is None:
            raise VAError(f"undefined macro `{name}")
        if name not in state.used:
            state.used.append(name)
        j = m.end()
        if macro.params is not None:
            k = j
            while k < len(text) and text[k] in " \t":
                k += 1
            if k >= len(text) or text[k] != "(":
                raise VAError(f"macro `{name} needs arguments")
            args, j = _split_args(text, k)
            if len(args) != len(macro.params):
                raise VAError(f"macro `{name} takes {len(macro.params)} "
                              f"arguments, got {len(args)}")
            args = [expand(a, state, depth + 1) for a in args]
            body = _substitute(macro.body, macro.params, args)
        else:
            body = macro.body
        out.append(expand(body, state, depth + 1))
        i = j
    return "".join(out)


_DIRECTIVE = re.compile(r"^\s*`(define|undef|ifdef|ifndef|elsif|else|endif|include|"
                        r"timescale|resetall|default_nettype|celldefine|"
                        r"endcelldefine|default_discipline|default_transition)\b"
                        r"(.*)$", re.DOTALL)


def _paren_balance(text: str) -> int:
    depth, in_str = 0, False
    for i, c in enumerate(text):
        if in_str:
            if c == '"' and text[i - 1] != "\\":
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return depth


def _open_string(text: str) -> bool:
    """`text` ends inside a string literal."""
    in_str = False
    for i, c in enumerate(text):
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            in_str = not in_str
    return in_str


def _define(rest: str, state: _State, file: str, lineno: int) -> None:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_$]*)(\(([^)]*)\))?(.*)$", rest, re.DOTALL)
    if not m:
        raise VAError("malformed `define", file, lineno)
    name = m.group(1)
    params = None
    if m.group(2) is not None:
        params = tuple(p.strip() for p in m.group(3).split(",") if p.strip())
    body = m.group(4).strip()
    if name not in state.locked:
        state.macros[name] = Macro(name, params, body)


def preprocess_text(text: str, file: str, state: _State) -> list[Line]:
    """Preprocess `text` (from `file`) with `state` (macros are updated)."""
    if state.depth > 32:
        raise VAError("`include nested too deeply", file)
    raw = strip_comments(text).split("\n")
    out: list[Line] = []
    # conditional stack: (this branch active, some branch taken, parent active)
    cond: list[tuple[bool, bool, bool]] = []
    active = True
    i = 0
    while i < len(raw):
        lineno = i + 1
        line = raw[i]
        m = _DIRECTIVE.match(line)
        if m is None and "`" in line and active:
            # a directive may follow code on the same line only rarely; treat
            # the whole line as code (macro uses) otherwise
            pass
        if m is not None:
            kind, rest = m.group(1), m.group(2)
            consumed = 1
            if kind == "define":
                while rest.rstrip().endswith("\\") and i + consumed < len(raw):
                    rest = rest.rstrip()[:-1] + " " + raw[i + consumed]
                    consumed += 1
                if active:
                    try:
                        _define(rest, state, file, lineno)
                    except VAError as e:
                        raise VAError(str(e), file, lineno) from None
            elif kind == "undef":
                if active:
                    state.macros.pop(rest.strip(), None)
            elif kind in ("ifdef", "ifndef"):
                name = rest.strip().split()[0] if rest.strip() else ""
                defined = name in state.macros
                take = defined if kind == "ifdef" else not defined
                cond.append((active and take, take, active))
                active = active and take
            elif kind == "elsif":
                if not cond:
                    raise VAError("`elsif without `ifdef", file, lineno)
                _, taken, parent = cond.pop()
                name = rest.strip().split()[0]
                take = (not taken) and name in state.macros
                cond.append((parent and take, taken or take, parent))
                active = parent and take
            elif kind == "else":
                if not cond:
                    raise VAError("`else without `ifdef", file, lineno)
                _, taken, parent = cond.pop()
                cond.append((parent and not taken, True, parent))
                active = parent and not taken
            elif kind == "endif":
                if not cond:
                    raise VAError("`endif without `ifdef", file, lineno)
                _, _, parent = cond.pop()
                active = parent
            elif kind == "include":
                if active:
                    name_m = re.match(r'\s*"([^"]+)"', rest)
                    if not name_m:
                        raise VAError("malformed `include", file, lineno)
                    out += _include(name_m.group(1), state, file, lineno)
            # other directives (`timescale, ...) are ignored
            out += [Line(file, lineno + k, "") for k in range(consumed)]
            i += consumed
            continue
        if not active:
            out.append(Line(file, lineno, ""))
            i += 1
            continue
        # join lines while parentheses are open (multi-line macro calls)
        consumed = 1
        while line.rstrip().endswith("\\") and _open_string(line) and \
                i + consumed < len(raw):  # string continued on the next line
            line = line.rstrip()[:-1] + raw[i + consumed].lstrip()
            consumed += 1
        if "`" in line:
            while _paren_balance(line) > 0 and i + consumed < len(raw) and \
                    _DIRECTIVE.match(raw[i + consumed]) is None:
                line = line + " " + raw[i + consumed]
                consumed += 1
        state.used = []
        try:
            text_out = expand(line, state)
        except VAError as e:
            raise VAError(str(e), file, lineno) from None
        out.append(Line(file, lineno, text_out, tuple(state.used)))
        out += [Line(file, lineno + k, "") for k in range(1, consumed)]
        i += consumed
    if cond:
        raise VAError("missing `endif", file)
    return out


def _include(name: str, state: _State, file: str, lineno: int) -> list[Line]:
    base = Path(file).parent if file and not file.startswith("<") else None
    for d in ([base] if base else []) + state.include_dirs:
        path = d / name
        if path.is_file():
            state.depth += 1
            try:
                return preprocess_text(path.read_text(), str(path), state)
            finally:
                state.depth -= 1
    if name in BUILTIN_FILES:
        state.depth += 1
        try:
            lines = preprocess_text(BUILTIN_FILES[name], f"<{name}>", state)
        finally:
            state.depth -= 1
        return [ln for ln in lines if ln.text.strip()]
    raise VAError(f"cannot find include file {name!r}", file, lineno)


def preprocess(text: str, file: str = "<string>",
               defines: dict[str, str] | None = None,
               overrides: dict[str, str] | None = None,
               include_dirs: list[str | Path] | None = None) -> list[Line]:
    """Preprocess Verilog-A source.

    Args:
        text: Source text.
        file: Its name (for messages and relative includes).
        defines: Macros defined before the source (like ``-D``), e.g.
            ``{"PHYSICAL_CONSTANTS_OLD": ""}``.
        overrides: Macros that the source cannot redefine (its own
            `` `define`` of these names is ignored). Useful to pin physical
            constants to another simulator's values.
        include_dirs: Extra include search directories.
    """
    state = _State(include_dirs=[Path(d) for d in (include_dirs or [])])
    for name, body in (defines or {}).items():
        _define(f"{name} {body}", state, "<defines>", 0)
    for name, body in (overrides or {}).items():
        _define(f"{name} {body}", state, "<overrides>", 0)
        state.locked.add(name.split("(")[0])
    return preprocess_text(text, file, state)
