"""Helpers called by generated Verilog-A code (kept tiny and explicit)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def guard(cond: Array, x: Array) -> Array:
    """``x`` where `cond`, else ``stop_gradient(x)``: the NaN-gradient guard.

    Inside a branch taken only where `cond` holds, every outer value is read
    through `guard`. The final ``where`` selects the other branch in lanes
    where `cond` is false, so values computed there are discarded; `guard`
    additionally stops their (possibly NaN/inf) derivatives from leaking out,
    because both forward- and reverse-mode derivatives of ``where`` are
    selects, which drop the NaNs instead of multiplying them by zero.
    """
    return jnp.where(cond, x, jax.lax.stop_gradient(x))


def to_real(b: Array) -> Array:
    return jnp.where(b, 1.0, 0.0)


def to_bool(x: Array) -> Array:
    return x if jnp.result_type(x) == jnp.bool_ else x != 0


def int_div(a: Array, b: Array) -> Array:
    """C integer division (truncation toward zero)."""
    return jnp.trunc(a / b)


def to_int(x: Array) -> Array:
    """Verilog-A real -> integer conversion: round half away from zero."""
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def ln(x: Array) -> Array:
    return jnp.log(x)


def log10(x: Array) -> Array:
    return jnp.log10(x)


def limexp(x: Array, x_max: float = 80.0) -> Array:
    """``exp(x)`` continued linearly (C1) above `x_max` (Verilog-A ``limexp``).

    Exact at any converged point with ``x <= x_max``; prevents overflow in
    Newton iterates.
    """
    e = jnp.exp(jnp.minimum(x, x_max))
    return jnp.where(x > x_max, e * (1.0 + x - x_max), e)


def limit_merge(merged: Array, cond: Array, general: Array,
                special_when: bool) -> Array:
    """Value of `merged` with the derivative of `general` where the special
    (equality) branch is taken.

    ``if (Vds == 0) Vdseff = 0;`` patches a single point of a function that
    is smooth through it; plain AD of the patch gives a zero slope there. This
    keeps the patched value exactly and differentiates the general formula
    instead (non-finite general values fall back to the patch's derivative).
    """
    on = cond if special_when else jnp.logical_not(cond)
    finite = jnp.isfinite(general)
    tangent = jnp.where(finite, general - jax.lax.stop_gradient(general), 0.0)
    return jnp.where(on & finite, jax.lax.stop_gradient(merged) + tangent, merged)


# ------------------------------------------------------------- smooth mode


def smax(a: Array, b: Array, w: float) -> Array:
    """Smooth ``max(a, b)``: ``b + w softplus((a - b)/w)``; error <= w ln 2."""
    return b + w * jax.nn.softplus((a - b) / w)


def smin(a: Array, b: Array, w: float) -> Array:
    """Smooth ``min(a, b)`` (= ``-smax(-a, -b)``)."""
    return a - w * jax.nn.softplus((a - b) / w)


def sabs(x: Array, w: float) -> Array:
    """Smooth ``|x|``: ``w log(2 cosh(x/w))``; error <= w ln 2 at 0."""
    return w * jnp.logaddexp(x / w, -x / w)
