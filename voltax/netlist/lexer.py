"""Lexical layer: numbers, logical lines with source locations, fields.

* `parse_value`: SPICE numbers with scale suffixes (``10f``, ``2meg``).
* `read_lines`: physical text -> logical `Line` objects (comments stripped,
  continuations joined, ``.control`` blocks dropped, stops at ``.end``), each
  carrying its `Loc` (file and line number) for error messages.
* `split_fields`: one logical line -> tokens, respecting ``{}``/``()``/quotes
  and joining ``key = value`` into ``key=value``.
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass

SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "mil": 25.4e-6,
          "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
_EXPONENT = {"t": 12, "g": 9, "meg": 6, "k": 3, "m": -3, "u": -6, "µ": -6, "n": -9,
             "p": -12, "f": -15}
NUMBER = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-zµ]*)$",
                    re.IGNORECASE)


@dataclass(frozen=True)
class Loc:
    """Source location of a logical line."""

    file: str
    line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


class NetlistError(ValueError):
    """A netlist error, prefixed with ``file:line`` when the location is known."""

    def __init__(self, message: str, loc: Loc | None = None):
        self.message, self.loc = message, loc
        super().__init__(f"{loc}: {message}" if loc else message)


def warn(message: str) -> None:
    """`warnings.warn`, attributed to the first caller outside the parser."""
    level, frame = 1, sys._getframe(0)
    while frame.f_back is not None and frame.f_globals.get(
            "__name__", "").startswith("voltax.netlist"):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, stacklevel=level)


@dataclass
class Line:
    """A logical line (continuations joined) and where it starts."""

    text: str
    loc: Loc


def suffix_scale(letters: str) -> float:
    """Scale factor of the letters after a number: ``meg``, ``k``, ``pF``...

    As in SPICE, only the leading scale letter counts and trailing unit
    letters are ignored (``10pF`` = 10e-12, ``5V`` = 5, ``1F`` = 1e-15).
    """
    low = letters.lower()
    if low.startswith("meg"):
        return SUFFIX["meg"]
    if low.startswith("mil"):
        return SUFFIX["mil"]
    return SUFFIX.get(low[:1], 1.0) if low else 1.0


def scaled(mantissa: str, letters: str) -> float:
    """``mantissa`` times the scale of ``letters``, rounded once (``"10"``,
    ``"u"`` -> exactly ``10e-6``, not ``10 * 1e-6``)."""
    low = letters.lower()
    key = "meg" if low.startswith("meg") else "mil" if low.startswith("mil") \
        else low[:1]
    if key in _EXPONENT and "e" not in mantissa.lower():
        return float(f"{mantissa}e{_EXPONENT[key]}")
    return float(mantissa) * suffix_scale(letters)


def try_number(text: str) -> float | None:
    """`text` as a SPICE number, or None if it is not a plain number."""
    m = NUMBER.match(text.strip())
    if not m:
        return None
    return scaled(m.group(1), m.group(2))


def parse_value(text: str) -> float:
    """Parse a SPICE number: ``"10f"``, ``"1.2u"``, ``"4.7k"``, ``"2meg"``,
    ``"10pF"`` (trailing unit letters are ignored, as in SPICE).

    For expressions (``"{2*r}"``, ``"'w/2'"``) use `parse_netlist`, where
    parameters are in scope.
    """
    value = try_number(text)
    if value is None:
        raise ValueError(f"cannot parse value {text!r}")
    return value


# =============================================================================
# Logical lines
# =============================================================================


def _strip_comment(line: str) -> str:
    """Cut inline comments (``;`` anywhere, ``$`` and ``//`` after
    whitespace), ignoring comment characters inside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == ";":
            return line[:i]
        elif ch == "$" and (i == 0 or line[i - 1] == " ") and (
                i + 1 == len(line) or line[i + 1] == " "):
            return line[:i]
        elif ch == "/" and line[i:i + 2] == "//" and (i == 0 or line[i - 1] == " "):
            return line[:i]
    return line


def read_lines(text: str, file: str = "<netlist>", title: bool = False) -> list[Line]:
    """Split netlist text into logical lines.

    Handles CRLF, tabs, ``*`` full-line and ``;``/``$ ``/``//`` inline
    comments, ``+`` continuation lines and trailing-backslash continuations.
    ``.control``/``.endc`` blocks are dropped and reading stops at ``.end``.
    With `title`, the first line is the deck title and is skipped.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    out: list[Line] = []
    in_control = False
    joining = False  # previous physical line ended with a backslash
    for number, raw in enumerate(text.split("\n"), start=1):
        if title and number == 1:
            continue
        stripped = raw.strip()
        if not joining and stripped.startswith("*"):
            continue
        line = _strip_comment(raw).strip()
        low = line.lower()
        if in_control:
            in_control = not low.startswith(".endc")
            continue
        if low.startswith(".control"):
            in_control = True
            continue
        cont_next = line.endswith("\\")
        line = line.rstrip("\\").rstrip()
        if joining or (line.startswith("+") and out):
            if line.startswith("+"):
                line = line[1:]
            if out:
                out[-1].text += " " + line.strip()
            joining = cont_next
            continue
        joining = cont_next
        if not line:
            continue
        if low.split(None, 1)[0] == ".end":
            break
        out.append(Line(line, Loc(file, number)))
    return out


# =============================================================================
# Fields
# =============================================================================

_OPEN = {"(": ")", "{": "}", "[": "]"}
_OPERATOR_END = tuple("+-*/^<>=!&|?:,(")
_OPERATOR_START = tuple("*/^<>=!&|?:),")


def _raw_tokens(text: str) -> list[str]:
    """Whitespace-split at nesting depth 0; a lone ``=`` is its own token."""
    toks: list[str] = []
    cur = ""
    stack: list[str] = []
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            cur += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur += ch
        elif ch in _OPEN:
            stack.append(_OPEN[ch])
            cur += ch
        elif stack and ch == stack[-1]:
            stack.pop()
            cur += ch
        elif not stack and ch.isspace():
            if cur:
                toks.append(cur)
            cur = ""
        elif not stack and ch == "=" and text[i + 1:i + 2] != "=" and (
                not cur or cur[-1] not in "=<>!"):
            if cur:
                toks.append(cur)
            toks.append("=")
            cur = ""
        else:
            cur += ch
        i += 1
    if cur:
        toks.append(cur)
    return toks


def split_fields(text: str) -> list[str]:
    """Tokens of a logical line.

    ``key = value`` becomes ``key=value``; an unbraced expression value that
    contains spaces around operators (``a = b * 2``) stays one token.
    ``PULSE (0 1 ...)`` is joined to ``PULSE(0 1 ...)``.
    """
    raw = _raw_tokens(text)
    out: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok == "=" and out and i + 1 < len(raw):
            value = raw[i + 1]
            i += 2
            # extend an unbraced expression: `a = b + 1`, `a = 2 * c`
            while i < len(raw) and raw[i] != "=" and (
                    value.endswith(_OPERATOR_END) or raw[i].startswith(_OPERATOR_START)
                    or raw[i] in ("+", "-")):
                if i + 1 < len(raw) and raw[i + 1] == "=":
                    break  # raw[i] is the next key
                value += raw[i]
                i += 1
            out[-1] = f"{out[-1]}={value}"
            continue
        if tok.startswith("(") and out and re.fullmatch(r"[A-Za-z_]\w*", out[-1]) \
                and out[-1].lower() in _CALL_WORDS:
            out[-1] += tok
        else:
            out.append(tok)
        i += 1
    return out


_CALL_WORDS = {"pulse", "sin", "exp", "pwl", "sffm", "am", "poly", "dc", "ac",
               "nmos", "pmos", "d", "npn", "pnp", "sw", "csw", "r", "c", "l",
               "table", "value", "file"}


def split_kv(token: str) -> tuple[str, str] | None:
    """``"w=2u"`` -> ``("w", "2u")`` (key lower-cased); None if no ``=``
    outside brackets/quotes."""
    depth, quote = 0, None
    for i, ch in enumerate(token):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0 and i > 0 and token[i - 1] not in "=<>!" \
                and token[i + 1:i + 2] != "=":
            return token[:i].strip().lower(), token[i + 1:].strip()
    return None


def unquote(text: str) -> str:
    """Strip one pair of matching ``"`` or ``'`` quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
