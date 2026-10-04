"""Static CMOS logic gates and arithmetic blocks.

Each function adds transistors to a `CircuitBuilder` between existing nodes
and creates any internal nets it needs with `CircuitBuilder.node`, so blocks
compose freely::

    b = vx.CircuitBuilder()
    cmos.nand(b, ["a", "b"], "y", vdd="vdd")
    cmos.ripple_adder(b, a=["a0", "a1"], b=["b0", "b1"], cin="0",
                      s=["s0", "s1"], cout="co", vdd="vdd")

NMOS bulks tie to `gnd`, PMOS bulks to `vdd`. `wn`/`wp` set device widths
(length is the builder default); a series stack of ``k`` devices is widened
``k``-fold so every gate has roughly unit-inverter drive.
"""

from __future__ import annotations

from typing import Sequence

from ..builder import CircuitBuilder

WN, WP = 1e-6, 2e-6
"""Default NMOS / PMOS widths (m) of a unit inverter."""


def inverter(b: CircuitBuilder, a: str, y: str, vdd: str, gnd: str = "0",
             wn: float = WN, wp: float = WP) -> None:
    """``y = not a``."""
    b.pmos(y, a, vdd, vdd, w=wp)
    b.nmos(y, a, gnd, gnd, w=wn)


def nand(b: CircuitBuilder, inputs: Sequence[str], y: str, vdd: str,
         gnd: str = "0", wn: float = WN, wp: float = WP) -> None:
    """``y = not (a and b and ...)``: parallel PMOS, series NMOS."""
    k = len(inputs)
    for a in inputs:
        b.pmos(y, a, vdd, vdd, w=wp)
    lower = gnd
    for i, a in enumerate(inputs):
        upper = y if i == k - 1 else b.node()
        b.nmos(upper, a, lower, gnd, w=k * wn)
        lower = upper


def nor(b: CircuitBuilder, inputs: Sequence[str], y: str, vdd: str,
        gnd: str = "0", wn: float = WN, wp: float = WP) -> None:
    """``y = not (a or b or ...)``: series PMOS, parallel NMOS."""
    k = len(inputs)
    for a in inputs:
        b.nmos(y, a, gnd, gnd, w=wn)
    upper = vdd
    for i, a in enumerate(inputs):
        lower = y if i == k - 1 else b.node()
        b.pmos(lower, a, upper, vdd, w=k * wp)
        upper = lower


def and_(b: CircuitBuilder, inputs: Sequence[str], y: str, vdd: str,
         gnd: str = "0", **sizes: float) -> None:
    """``y = a and b and ...`` (NAND + inverter)."""
    mid = b.node()
    nand(b, inputs, mid, vdd, gnd, **sizes)
    inverter(b, mid, y, vdd, gnd, **sizes)


def or_(b: CircuitBuilder, inputs: Sequence[str], y: str, vdd: str,
        gnd: str = "0", **sizes: float) -> None:
    """``y = a or b or ...`` (NOR + inverter)."""
    mid = b.node()
    nor(b, inputs, mid, vdd, gnd, **sizes)
    inverter(b, mid, y, vdd, gnd, **sizes)


def xor(b: CircuitBuilder, a: str, c: str, y: str, vdd: str, gnd: str = "0",
        **sizes: float) -> None:
    """``y = a xor c`` from four NAND2 gates."""
    n1, n2, n3 = b.node(), b.node(), b.node()
    nand(b, [a, c], n1, vdd, gnd, **sizes)
    nand(b, [a, n1], n2, vdd, gnd, **sizes)
    nand(b, [c, n1], n3, vdd, gnd, **sizes)
    nand(b, [n2, n3], y, vdd, gnd, **sizes)


def full_adder(b: CircuitBuilder, a: str, c: str, cin: str, s: str, cout: str,
               vdd: str, gnd: str = "0", **sizes: float) -> None:
    """One-bit full adder: ``s = a ^ c ^ cin``, ``cout = ac + cin (a ^ c)``."""
    p, g, t = b.node(), b.node(), b.node()
    xor(b, a, c, p, vdd, gnd, **sizes)
    xor(b, p, cin, s, vdd, gnd, **sizes)
    and_(b, [a, c], g, vdd, gnd, **sizes)
    and_(b, [cin, p], t, vdd, gnd, **sizes)
    or_(b, [g, t], cout, vdd, gnd, **sizes)


def ripple_adder(b: CircuitBuilder, a: Sequence[str], c: Sequence[str], cin: str,
                 s: Sequence[str], cout: str, vdd: str, gnd: str = "0",
                 **sizes: float) -> None:
    """``k``-bit ripple-carry adder (LSB first): ``s, cout = a + c + cin``."""
    if not len(a) == len(c) == len(s):
        raise ValueError("a, c and s must have the same width")
    carry = cin
    for i in range(len(a)):
        nxt = cout if i == len(a) - 1 else b.node()
        with b.scope(f"fa{i}"):
            full_adder(b, a[i], c[i], carry, s[i], nxt, vdd, gnd, **sizes)
        carry = nxt


def ring_oscillator(b: CircuitBuilder, nodes: Sequence[str], vdd: str,
                    gnd: str = "0", **sizes: float) -> None:
    """Ring of ``len(nodes)`` inverters (use an odd count):
    ``nodes[i] -> nodes[i+1]``, last back to first."""
    if len(nodes) % 2 == 0:
        raise ValueError("a ring oscillator needs an odd number of stages")
    for i, a in enumerate(nodes):
        inverter(b, a, nodes[(i + 1) % len(nodes)], vdd, gnd, **sizes)
