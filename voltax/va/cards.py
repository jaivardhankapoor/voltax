"""SPICE ``.model`` cards for Verilog-A models.

`read_cards` parses the ``.model`` statements of a SPICE text (comments,
``+`` continuations, ``key = value`` with SPICE scale suffixes) into
``{name: (kind, params)}``; `bundled_card` reads the cards shipped in
``voltax/va/models/cards`` (see the NOTICE there).
"""

from __future__ import annotations

import re
from pathlib import Path

from .model import MODELS_DIR

_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "mil": 25.4e-6,
           "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(meg|mil|[tgkmunpfa])?",
                  re.IGNORECASE)


def spice_number(text: str) -> float:
    """``"1.2u"`` -> ``1.2e-6`` (SPICE suffixes; trailing units ignored)."""
    m = _NUM.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse number {text!r}")
    return float(m.group(1)) * _SUFFIX.get((m.group(2) or "").lower(), 1.0)


def read_cards(text: str) -> dict[str, tuple[str, dict[str, float]]]:
    """``{model_name: (kind, {param: value})}`` for every ``.model`` in `text`.

    Names and parameter keys are lower-cased; ``kind`` is e.g. ``"nmos"``.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = re.split(r"[;$]", raw, maxsplit=1)[0].rstrip()
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        if s.startswith("+") and lines:
            lines[-1] += " " + s[1:]
        else:
            lines.append(s)
    out = {}
    for line in lines:
        if not line.lower().startswith(".model"):
            continue
        line = re.sub(r"\s*=\s*", "=", line.replace("(", " ").replace(")", " "))
        toks = line.split()
        if len(toks) < 3:
            raise ValueError(f"malformed .model line: {line!r}")
        name, kind = toks[1].lower(), toks[2].lower()
        params = {}
        for t in toks[3:]:
            if "=" not in t:
                raise ValueError(f".model {name}: expected key=value, got {t!r}")
            k, v = t.split("=", 1)
            params[k.lower()] = spice_number(v)
        out[name] = (kind, params)
    return out


CARDS_DIR = MODELS_DIR / "cards"


def bundled_card(filename: str) -> tuple[str, str, dict[str, float]]:
    """``(name, kind, params)`` of the single model in a bundled card file,
    e.g. ``bundled_card("freepdk45_nmos.inc")``."""
    path = Path(filename)
    if not path.is_file():
        path = CARDS_DIR / filename
    cards = read_cards(path.read_text())
    if len(cards) != 1:
        raise ValueError(f"{filename}: expected one .model, found {list(cards)}")
    (name, (kind, params)), = cards.items()
    return name, kind, params
