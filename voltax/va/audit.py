"""Differentiability audit of compiled Verilog-A models.

Two complementary views:

**Static** (`static_report`): the compiler records every conditional,
equality branch, ``min``/``max``/``abs``, clamp (``if (x < b) x = b``) and
limiter it evaluates, with its source line, enclosing macro and what its
condition depends on. Compiling with *everything* traced (all real model
parameters, the temperature and the instance parameters) shows every place
where the model can switch branches as a function of bias
(``branch-on-bias``) or of a parameter (``branch-on-param``); the rest is
resolved at compile time (``static``) and cannot affect gradients. Sites
whose result never reaches the outputs (dead code, e.g. noise) are counted
separately.

**Empirical** (`DeviceProbe`): sweep one variable (a bias direction or a
parameter) on a grid through the compiled code and check

* non-finite values and non-finite first/second AD derivatives;
* AD first derivatives against central finite differences;
* every *branch switch* crossed: the probe is compiled with instrumentation
  (each traced site's condition and path condition are extra outputs), so
  each active switch is located by bisection on its own condition to
  ~1e-15, and the value / slope jumps across it are measured with a
  micro-trapezoid test: for a C1 function
  ``y(x+) - y(x-) = (x+ - x-) (y'(x+) + y'(x-))/2 + O(dx^3)``. A switch is
  then classified as a **C0 break** (value jump), **C1 break** (slope jump:
  a kink) or smooth;
* equality branches (``if (x == c)``) taken at a grid point are checked for
  **AD point errors**: the derivative *at* the point vs the one-sided ones;
* a grid-level trapezoid test flags defects that no recorded switch explains.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .compiler import Compiled, Site
from .model import DEFAULT_TEMPERATURE, VAModel

# =============================================================================
# Static report
# =============================================================================


@dataclass
class SiteSummary:
    """All evaluations of one (line, kind) in the compiled model."""

    file: str
    line: int
    kind: str
    deps: frozenset
    count: int
    macros: tuple[str, ...]
    detail: str
    source: str
    live: bool = True
    """False if no evaluation of this site reaches the model outputs (e.g.
    noise-only code removed by dead-code elimination)."""

    @property
    def category(self) -> str:
        if not self.deps:
            return "static"
        if "bias" in self.deps:
            return "branch-on-bias"
        return "branch-on-param"

    @property
    def depends(self) -> str:
        order = ("bias", "inst", "model", "temp")
        return "+".join(d for d in order if d in self.deps) or "-"


@dataclass
class StaticReport:
    """Aggregated audit records of one compilation."""

    model: str
    sites: list[SiteSummary]
    risks: list[tuple[str, int, str, frozenset, int]]
    n_statements: tuple[int, int]
    traced_params: int

    def counts(self, live: bool | None = True) -> dict[tuple[str, str], int]:
        """``{(kind, category): number of (live) source sites}``."""
        out: dict[tuple[str, str], int] = defaultdict(int)
        for s in self.select(live=live):
            out[(s.kind, s.category)] += 1
        return dict(out)

    def select(self, category: str | None = None,
               kinds: Iterable[str] | None = None,
               live: bool | None = True) -> list[SiteSummary]:
        """Sites filtered by category, kinds and liveness (None: any)."""
        kinds = set(kinds) if kinds is not None else None
        return [s for s in self.sites
                if (category is None or s.category == category)
                and (kinds is None or s.kind in kinds)
                and (live is None or s.live == live)]

    def summary(self) -> str:
        """Markdown summary: counts by kind and category, NaN-risk ops."""
        cats = ("branch-on-bias", "branch-on-param", "static")
        kinds = sorted({s.kind for s in self.select()})
        dead = len(self.select(live=False))
        lines = [f"Static audit of {self.model}: {len(self.select())} live source "
                 f"sites (+{dead} in dead code; {self.traced_params} traced "
                 f"inputs; generated f/q: {self.n_statements[0]}/"
                 f"{self.n_statements[1]} statements)",
                 "",
                 f"| kind | {' | '.join(cats)} |",
                 f"|---|{'---|' * len(cats)}"]
        c = self.counts()
        for k in kinds:
            lines.append(f"| {k} | " + " | ".join(str(c.get((k, cat), 0))
                                                  for cat in cats) + " |")
        risky: dict[str, int] = defaultdict(int)
        for _, _, op, deps, _ in self.risks:
            if "bias" in deps:
                risky[op] += 1
        if risky:
            lines += ["", "Bias-dependent operations that can produce NaN/inf "
                      "(source sites): " + ", ".join(f"{k} {v}" for k, v in
                                                     sorted(risky.items()))]
        return "\n".join(lines)

    def table(self, category: str | None = "branch-on-bias",
              kinds: Iterable[str] | None = None, limit: int | None = None) -> str:
        """Markdown table of live sites (default: the bias-dependent ones)."""
        rows = self.select(category, kinds)[:limit]
        out = ["| line | kind | depends on | macro | source |", "|---|---|---|---|---|"]
        for s in rows:
            src = s.source.strip().replace("|", "\\|")
            if len(src) > 70:
                src = src[:67] + "..."
            out.append(f"| {s.line} | {s.kind} | {s.depends} | "
                       f"{s.macros[0] if s.macros else ''} | `{src}` |")
        return "\n".join(out)


def static_report(model: VAModel, params: Mapping[str, Any] | None = None, *,
                  polarity: str | None = "n", differentiable: Iterable[str] | str =
                  "all", temperature: float | None = None,
                  instance: Mapping[str, Any] | None = None) -> StaticReport:
    """Compile `model` with everything traced and aggregate its audit sites.

    Args:
        model: The loaded Verilog-A model.
        params: Model card (integer selectors decide which code exists).
        polarity: Sets ``type`` like `VAModel.element`.
        differentiable: Model parameters to trace (default ``"all"``).
        temperature: None (default) traces ``$temperature`` too.
        instance: Given instance parameters.
    """
    card = dict(params or {})
    if polarity is not None and "type" in model.params and "type" not in card:
        card["type"] = 1 if polarity == "n" else -1
    c = model.compile(card, differentiable=differentiable, instance=instance,
                      temperature=temperature)
    return summarize(model, c)


def summarize(model: VAModel, c: Compiled) -> StaticReport:
    """Aggregate the audit records of an existing compilation."""
    groups: dict[tuple[str, int, str], list[Site]] = defaultdict(list)
    for s in c.sites:
        groups[(s.file, s.line, s.kind)].append(s)
    sites = []
    for (f, n, kind), ss in sorted(groups.items(), key=lambda kv: (kv[0][1],
                                                                    kv[0][2])):
        deps = frozenset().union(*(s.deps for s in ss))
        live = any(c.site_live(s) for s in ss)
        sites.append(SiteSummary(f, n, kind, deps, len(ss), ss[0].macros,
                                 ss[0].detail, _source_line(model, f, n), live))
    risks: dict[tuple[str, int, str], list] = defaultdict(list)
    for r in c.risks:
        if r.result is None or r.result in c.live:
            risks[(r.file, r.line, r.op)].append(r.deps)
    risk_rows = [(f, n, op, frozenset().union(*ds), len(ds))
                 for (f, n, op), ds in sorted(risks.items(), key=lambda kv: kv[0][1])]
    return StaticReport(model.name, sites, risk_rows, c.n_statements, len(c.inputs))


def _source_line(model: VAModel, file: str, line: int) -> str:
    """The *original* (unexpanded) source line, if the file is readable."""
    cache = model.__dict__.setdefault("_raw_lines", {})
    if file not in cache:
        try:
            with open(file) as fh:
                cache[file] = fh.read().splitlines()
        except OSError:
            cache[file] = []
    lines = cache[file]
    if 0 < line <= len(lines):
        return lines[line - 1]
    for ln in model.lines:  # e.g. source given as a string
        if ln.file == file and ln.lineno == line:
            return ln.text
    return ""


# =============================================================================
# Empirical probe
# =============================================================================


@dataclass
class Defect:
    """A grid interval where the trapezoid-consistency test fails."""

    output: str
    order: int  # 0: value jump, 1: slope jump
    x0: float
    x1: float
    size: float  # relative residual


@dataclass
class SwitchEvent:
    """Branch switches located (by bisection) at one point of a sweep.

    Attributes:
        x: Location (to ~1e-15 relative).
        sites: ``(line, kind)`` of the sites switching there.
        jump0, jump1: Per output, relative jump of the value / first
            derivative across the point (micro-trapezoid residuals; ~1e-16
            for a C1 function).
        point_error: Per output, relative difference between the AD
            derivative *at* the point and the nearer one-sided derivative
            (equality branches ``if (x == c)``; at a kink, AD may return
            either side's slope).
        isolated: True for equality branches (a single point, not a switch).
    """

    x: float
    sites: list[tuple[int, str]]
    jump0: dict[str, float] = field(default_factory=dict)
    jump1: dict[str, float] = field(default_factory=dict)
    point_error: dict[str, float] = field(default_factory=dict)
    isolated: bool = False

    def kinds(self, tol0: float = 1e-12, tol1: float = 1e-7,
              tolp: float = 1e-4) -> set[str]:
        """Subset of ``{"C0 break", "C1 break", "AD point error"}`` (an empty
        set: smooth to first order at working precision). Point errors are
        measured against one-sided slopes 1e-7 away, so curvature leaves
        ~1e-6; real ones are O(1)."""
        out = set()
        if self.jump0 and max(self.jump0.values()) > tol0:
            out.add("C0 break")
        if self.jump1 and max(self.jump1.values()) > tol1:
            out.add("C1 break")
        if self.point_error and max(self.point_error.values()) > tolp:
            out.add("AD point error")
        return out

    def order(self) -> str:
        """The most severe of `kinds` (``"C1"`` if none)."""
        for k in ("C0 break", "C1 break", "AD point error"):
            if k in self.kinds():
                return k
        return "C1"


@dataclass
class SweepResult:
    """Outcome of `DeviceProbe.sweep` (arrays are ``(n_points, n_outputs)``)."""

    name: str
    x: np.ndarray
    outputs: tuple[str, ...]
    y: np.ndarray
    dy: np.ndarray
    d2y: np.ndarray
    dy_fd: np.ndarray
    events: list[SwitchEvent]
    unexplained: list[Defect]
    """Grid defects with no branch switch inside (non-instrumented causes)."""

    @property
    def nonfinite_values(self) -> int:
        return int(np.sum(~np.isfinite(self.y)))

    @property
    def nonfinite_grads(self) -> int:
        return int(np.sum(~np.isfinite(self.dy)) + np.sum(~np.isfinite(self.d2y)))

    def fd_error(self, floor: float = 1e-9) -> np.ndarray:
        """``|AD - FD| / (|AD| + floor * max|AD|)`` per point and output."""
        scale = np.nanmax(np.abs(self.dy), axis=0, keepdims=True)
        return np.abs(self.dy - self.dy_fd) / (np.abs(self.dy) + floor * scale + 1e-300)

    def breaks(self, kind: str) -> list[SwitchEvent]:
        """Events showing one kind of defect (see `SwitchEvent.kinds`)."""
        return [e for e in self.events if kind in e.kinds()]

    def summary_row(self, fd_floor: float = 1e-6) -> dict:
        """One table row: NaNs, AD-vs-FD, switches and breaks."""
        fd = self.fd_error(fd_floor)
        ok = np.ones(len(self.x), bool)
        span = float(np.max(self.x) - np.min(self.x)) if len(self.x) > 1 else 1.0
        for e in self.events:  # an FD stencil straddling a break is meaningless
            if e.order() != "C1":
                ok &= np.abs(self.x - e.x) > 1e-5 * span
        c0, c1, pt = (self.breaks(k) for k in ("C0 break", "C1 break",
                                               "AD point error"))

        def mx(events: list[SwitchEvent], attr: str) -> float:
            vals = [max(getattr(e, attr).values()) for e in events
                    if getattr(e, attr)]
            return max(vals) if vals else 0.0

        lines = sorted({ln for e in c0 + c1 + pt for ln, _ in e.sites})
        return {
            "sweep": self.name,
            "points": len(self.x),
            "NaN/inf y": self.nonfinite_values,
            "NaN/inf AD": self.nonfinite_grads,
            "max |AD-FD|": float(np.nanmax(fd[ok])) if ok.any() else float("nan"),
            "switch pts": len(self.events),
            "C0 breaks": len(c0),
            "max jump": mx(c0, "jump0"),
            "C1 breaks": len(c1),
            "max slope jump": mx(c1, "jump1"),
            "AD pt errors": len(pt),
            "max AD pt err": mx(pt, "point_error"),
            "unexplained": len(self.unexplained),
            "lines": ",".join(str(x) for x in lines[:10]) +
            ("..." if len(lines) > 10 else ""),
        }


class DeviceProbe:
    """Evaluate a compiled model's terminal currents/charges as explicit
    functions of bias and parameters (no circuit, no Newton).

    The model must have no internal unknowns (for BSIM4: ``rdsmod=0,
    rgatemod=0, rbodymod=0``, the intrinsic device).

    Args:
        model: Loaded Verilog-A model.
        params: Model card.
        instance: Instance parameters (``w``, ``l`` ...).
        polarity: ``"n"``/``"p"`` (sets ``type``).
        differentiable: Model parameters that sweeps may vary.
        temperature: Static temperature, or None to allow temperature sweeps.
        smooth: Smooth-mode width (None: exact).
        equality: Derivative mode at equality branches (see `VAModel.element`).
    """

    def __init__(self, model: VAModel, params: Mapping[str, Any], *,
                 instance: Mapping[str, Any] | None = None, polarity: str = "n",
                 differentiable: Iterable[str] | str = (),
                 temperature: float | None = DEFAULT_TEMPERATURE,
                 smooth: float | Mapping[int, float] | None = None,
                 equality: str = "value"):
        card = dict(params)
        if "type" in model.params and "type" not in card:
            card["type"] = 1 if polarity == "n" else -1
        self.model = model
        self.c = model.compile(card, differentiable=differentiable,
                               instance=instance, temperature=temperature,
                               smooth=smooth, instrument=True,
                               equality=equality)
        if self.c.internal:
            raise ValueError(f"DeviceProbe needs a model without internal unknowns; "
                             f"this card has {self.c.internal}")
        defaults = model.static_defaults(card)
        self.P0: dict[str, float] = {}
        for k in self.c.inputs:
            if k == "$temperature":
                self.P0[k] = DEFAULT_TEMPERATURE
            elif instance and k in instance:
                self.P0[k] = float(instance[k])
            else:
                self.P0[k] = float(defaults[k].v)
        self.terminals = self.c.terminals
        self.outputs = tuple(f"I{t}" for t in self.terminals) + \
            tuple(f"Q{t}" for t in self.terminals)
        self._site_keys = [(i, s) for i, s in enumerate(self.c.sites)
                           if s.cond is not None and self.c.site_live(s)]
        self.report = summarize(model, self.c)

    # ------------------------------------------------------------ evaluation

    def evaluate(self, V: Mapping[str, Any], P: Mapping[str, Any] | None = None):
        """``(outputs (2T,), observed dict)`` at bias `V` and parameters `P`."""
        P = {**self.P0, **(P or {})}
        V = {t: jnp.asarray(V.get(t, 0.0), dtype=float) for t in self.terminals}
        it, _, obs = self.c.f(V, P)
        qt, _ = self.c.q(V, P)
        y = jnp.stack([jnp.asarray(a, dtype=float) for a in (*it, *qt)])
        return y, obs

    def _path(self, bias: Mapping[str, float], var: str | Mapping[str, float]
              ) -> Callable[[Any], tuple[Any, Mapping[str, Any]]]:
        """``x -> (V, P)`` for a sweep variable."""
        if isinstance(var, str) and var in self.terminals:
            var = {var: 1.0}
        if isinstance(var, Mapping):
            direction = dict(var)

            def path(x):
                V = {t: bias.get(t, 0.0) + direction.get(t, 0.0) * x
                     for t in self.terminals}
                return V, {}
            return path
        if var not in self.P0:
            raise KeyError(f"{var!r} is not a traced input of this probe; traced: "
                           f"{sorted(self.P0)}")

        def path(x):
            return dict(bias), {var: x}
        return path

    def _compiled_line(self) -> dict[str, Callable]:
        """Jitted, vmapped evaluators along a generic line in (V, P) space,
        ``V = v0 + x vd``, ``P = p0 + x pd`` (compiled once per probe)."""
        if hasattr(self, "_line_fns"):
            return self._line_fns
        keys = list(self.P0)

        def at(x, v0, vd, p0, pd):
            V = {t: v0[k] + vd[k] * x for k, t in enumerate(self.terminals)}
            P = {key: p0[k] + pd[k] * x for k, key in enumerate(keys)}
            return self.evaluate(V, P)

        def f(x, *a):
            return at(x, *a)[0]

        def obs(x, *a):
            return at(x, *a)[1]

        def d1(x, *a):
            return jax.jvp(lambda u: f(u, *a), (x,), (jnp.ones_like(x),))[1]

        def d2(x, *a):
            return jax.jvp(lambda u: d1(u, *a), (x,), (jnp.ones_like(x),))[1]

        axes = (0, None, None, None, None)
        self._line_fns = {k: jax.jit(jax.vmap(fn, in_axes=axes)) for k, fn in
                          (("y", f), ("d1", d1), ("d2", d2), ("obs", obs))}
        return self._line_fns

    def functions(self, var: str | Mapping[str, float],
                  bias: Mapping[str, float] | None = None) -> dict[str, Callable]:
        """Vectorized ``xs -> y, y', y'', observed`` along a sweep path."""
        bias = dict(bias or {})
        keys = list(self.P0)
        v0 = np.array([bias.get(t, 0.0) for t in self.terminals], float)
        vd = np.zeros(len(self.terminals))
        p0 = np.array([self.P0[k] for k in keys], float)
        pd = np.zeros(len(keys))
        if isinstance(var, str) and var in self.terminals:
            var = {var: 1.0}
        if isinstance(var, Mapping):
            for t, wgt in var.items():
                vd[self.terminals.index(t)] = wgt
        else:
            if var not in self.P0:
                raise KeyError(f"{var!r} is not a traced input of this probe; "
                               f"traced: {sorted(self.P0)}")
            p0[keys.index(var)] = 0.0
            pd[keys.index(var)] = 1.0
        args = tuple(jnp.asarray(a) for a in (v0, vd, p0, pd))
        fns = self._compiled_line()
        return {k: (lambda xs, fn=fn: fn(jnp.asarray(xs), *args))
                for k, fn in fns.items()}

    def sweep(self, var: str | Mapping[str, float], values: Sequence[float],
              bias: Mapping[str, float] | None = None, name: str | None = None,
              fd_step: float | None = None, outputs: Sequence[str] | None = None,
              floor: float = 1e-9) -> SweepResult:
        """Sweep `var` (a terminal name, a ``{terminal: weight}`` direction, or
        a traced parameter name) over `values` at fixed `bias`.

        Args:
            fd_step: Central-difference step (default ``1e-6 * max(1, |x|)``
                for biases, ``1e-6 max(|x|, sweep span)`` for parameters).
            outputs: Subset of ``("Id", "Ig", ..., "Qb")`` to analyse
                (default: currents and charges of all terminals).
            floor: Relative floor for normalizations (fraction of the
                largest magnitude along the sweep).
        """
        is_param = isinstance(var, str) and var not in self.terminals
        fns = self.functions(var, bias)
        x = np.asarray(values, dtype=float)
        xs = jnp.asarray(x)
        y = np.asarray(fns["y"](xs))
        dy = np.asarray(fns["d1"](xs))
        d2y = np.asarray(fns["d2"](xs))
        if fd_step is None:
            span = float(np.max(x) - np.min(x)) if len(x) > 1 else 1.0
            h = 1e-6 * (np.maximum(np.abs(x), span) if is_param
                        else np.maximum(1.0, np.abs(x)))
        else:
            h = np.full_like(x, fd_step)
        dy_fd = (np.asarray(fns["y"](jnp.asarray(x + h))) -
                 np.asarray(fns["y"](jnp.asarray(x - h)))) / (2 * h)[:, None]
        conds = jax.device_get(fns["obs"](xs))

        names = self.outputs
        sel = list(range(len(names))) if not outputs else \
            [names.index(o) for o in outputs]
        scale0 = np.nanmax(np.abs(y), axis=0)
        scale1 = np.nanmax(np.abs(dy), axis=0)
        events = self._locate(fns, x, conds, is_param, sel, scale0, scale1, floor)
        defects = []
        for j in sel:
            defects += _defects(names[j], x, y[:, j], dy[:, j], d2y[:, j], floor)
        unexplained = [d for d in defects
                       if not any(d.x0 <= e.x <= d.x1 for e in events)]
        unexplained = _confirm(fns, unexplained, names, floor, scale0, scale1)
        label = name or (var if isinstance(var, str) else
                         "+".join(f"{v:+g}*V{k}" for k, v in var.items()))
        return SweepResult(label, x, tuple(names[j] for j in sel), y[:, sel],
                           dy[:, sel], d2y[:, sel], dy_fd[:, sel], events,
                           unexplained)

    # ------------------------------------------------------------ switches

    def _flips(self, conds: Mapping[str, np.ndarray]) -> list[tuple[int, int, Site]]:
        """``(interval, site_index, site)`` of active condition changes."""
        out = []
        for i, s in self._site_keys:
            if s.kind == "eq":
                continue
            c = np.asarray(conds[f"#{i}"]).astype(bool)
            if c.ndim == 0:
                continue
            change = c[1:] != c[:-1]
            if s.active is not None and f"@{i}" in conds:
                a = np.asarray(conds[f"@{i}"]).astype(bool)
                change &= a[1:] | a[:-1]
            out += [(int(k), i, s) for k in np.nonzero(change)[0]]
        return out

    def _isolated(self, conds: Mapping[str, np.ndarray]) -> list[tuple[int, Site]]:
        """Grid points where an (active) equality branch is taken."""
        out = []
        for i, s in self._site_keys:
            if s.kind != "eq":
                continue
            c = np.asarray(conds[f"#{i}"]).astype(bool)
            if s.detail.endswith("else branch"):
                c = ~c  # `if (x != c)`: the special case is the else branch
            if c.ndim == 0 or c.all():
                continue  # constant along this path: not a point condition
            if s.active is not None and f"@{i}" in conds:
                c = c & np.asarray(conds[f"@{i}"]).astype(bool)
            out += [(int(k), s) for k in np.nonzero(c)[0]]
        return out

    def _locate(self, fns, x, conds, is_param, sel, scale0, scale1, floor
                ) -> list[SwitchEvent]:
        names = self.outputs
        flips = self._flips(conds)
        located: list[tuple[float, float, Site]] = []
        if flips:
            lo = np.array([x[k] for k, _, _ in flips])
            hi = np.array([x[k + 1] for k, _, _ in flips])
            keys = [f"#{i}" for _, i, _ in flips]
            c_lo = np.array([bool(np.asarray(conds[key])[k])
                             for key, (k, _, _) in zip(keys, flips)])
            for _ in range(80):  # bisection on each switching condition
                mid = 0.5 * (lo + hi)
                done = (mid <= lo) | (mid >= hi)
                if done.all():
                    break
                obs = jax.device_get(fns["obs"](jnp.asarray(mid)))
                c_mid = np.array([bool(np.asarray(obs[key])[n])
                                  for n, key in enumerate(keys)])
                same = c_mid == c_lo
                lo = np.where(same & ~done, mid, lo)
                hi = np.where(~same & ~done, mid, hi)
            located = [(float(a), float(b), s) for a, b, (_, _, s) in
                       zip(lo, hi, flips)]
        groups: list[tuple[float, float, list[Site]]] = []
        for a, b, s in sorted(located, key=lambda t: t[1]):
            tol = 1e-12 * (abs(b) if is_param else max(1.0, abs(b)))
            if groups and abs(b - groups[-1][1]) <= tol:
                groups[-1][2].append(s)
            else:
                groups.append((a, b, [s]))
        out: list[SwitchEvent] = []
        if groups:
            lo = np.array([a for a, _, _ in groups])
            hi = np.array([b for _, b, _ in groups])
            unit = np.abs(hi) if is_param else np.maximum(1.0, np.abs(hi))
            # keep each stencil clear of neighbouring switch points (and of
            # equality points, which are exact grid values)
            marks = np.sort(np.concatenate([hi, x[[k for k, _ in
                                                  self._isolated(conds)]]]))
            gap = np.full(len(hi), np.inf)
            for n, b in enumerate(hi):
                other = np.abs(marks - b)
                other = other[other > 1e-15 * unit[n]]
                if other.size:
                    gap[n] = other.min()

            def residuals(rel_delta: float) -> tuple[np.ndarray, np.ndarray]:
                d = np.minimum(rel_delta * unit, 0.25 * gap)
                xm, xp = lo - d, hi + d
                ym, yp = (np.asarray(fns["y"](jnp.asarray(v))) for v in (xm, xp))
                dm, dp = (np.asarray(fns["d1"](jnp.asarray(v))) for v in (xm, xp))
                em, ep = (np.asarray(fns["d2"](jnp.asarray(v))) for v in (xm, xp))
                dx = (xp - xm)[:, None]
                r0 = np.abs(yp - ym - dx * 0.5 * (dp + dm)) / \
                    (np.maximum(np.abs(yp), np.abs(ym)) + floor * scale0 + 1e-300)
                r1 = np.abs(dp - dm - dx * 0.5 * (ep + em)) / \
                    (np.maximum(np.abs(dp), np.abs(dm)) + floor * scale1 + 1e-300)
                return r0, r1

            # a value jump persists as the stencil shrinks; the O(dx) residual
            # that a slope jump leaves in the value test does not
            r0a, r1 = residuals(1e-9)
            r0b, _ = residuals(1e-12)
            r0 = np.minimum(r0a, r0b)
            for n, (_, b, sites) in enumerate(groups):
                out.append(SwitchEvent(
                    b, sorted({(s.line, s.kind) for s in sites}),
                    {names[j]: float(r0[n, j]) for j in sel},
                    {names[j]: float(r1[n, j]) for j in sel}))
        isolated = self._isolated(conds)
        if isolated:
            pts: dict[int, list[Site]] = defaultdict(list)
            for k, s in isolated:
                pts[k].append(s)
            ks = sorted(pts)
            x0 = x[ks]
            delta = 1e-7 * (np.abs(x0) if is_param else np.maximum(1.0, np.abs(x0)))
            d0 = np.asarray(fns["d1"](jnp.asarray(x0)))
            dm = np.asarray(fns["d1"](jnp.asarray(x0 - delta)))
            dp = np.asarray(fns["d1"](jnp.asarray(x0 + delta)))
            err = np.minimum(np.abs(d0 - dm), np.abs(d0 - dp)) / \
                (np.maximum(np.abs(dm), np.abs(dp)) + floor * scale1 + 1e-300)
            for n, k in enumerate(ks):
                out.append(SwitchEvent(
                    float(x[k]), sorted({(s.line, s.kind) for s in pts[k]}),
                    point_error={names[j]: float(err[n, j]) for j in sel},
                    isolated=True))
        return sorted(out, key=lambda e: e.x)


def _defects(name: str, x, y, dy, d2y, floor: float) -> list[Defect]:
    """Grid trapezoid-consistency defects of `y` (order 0) and `dy` (order 1)."""
    out = []
    h = np.diff(x)
    for order, (u, du) in enumerate(((y, dy), (dy, d2y))):
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(du)):
            continue
        scale = np.max(np.abs(u)) if u.size else 0.0
        if scale == 0:
            continue
        resid = np.diff(u) - h * 0.5 * (du[1:] + du[:-1])
        local = np.maximum(np.abs(u[1:]), np.abs(u[:-1])) + floor * scale
        rel = np.abs(resid) / local
        tol = 1e-6 if order == 0 else 1e-4  # O(h^2) for smooth functions
        for k in np.nonzero(rel > tol)[0]:
            out.append(Defect(name, order, float(x[k]), float(x[k + 1]),
                              float(rel[k])))
    return out


def _confirm(fns, defects: list[Defect], names, floor, scale0, scale1
             ) -> list[Defect]:
    """Keep grid defects whose residual does not shrink on a 16x finer grid
    (smooth but rapidly varying regions shrink ~h^2; jumps do not)."""
    out = []
    for d in defects:
        j = names.index(d.output)
        x = np.linspace(d.x0, d.x1, 17)
        y = np.asarray(fns["y"](jnp.asarray(x)))[:, j]
        dy = np.asarray(fns["d1"](jnp.asarray(x)))[:, j]
        d2y = np.asarray(fns["d2"](jnp.asarray(x)))[:, j]
        u, du, scale = (y, dy, scale0[j]) if d.order == 0 else (dy, d2y, scale1[j])
        resid = np.diff(u) - np.diff(x) * 0.5 * (du[1:] + du[:-1])
        local = np.maximum(np.abs(u[1:]), np.abs(u[:-1])) + floor * scale
        if np.max(np.abs(resid) / local) > 0.3 * d.size:
            out.append(d)
    return out


# =============================================================================
# Helpers
# =============================================================================


def gummel_symmetry(probe: DeviceProbe, vg: float, vx_max: float = 0.1,
                    n: int = 401, vb: float = 0.0) -> dict[str, np.ndarray]:
    """Gummel symmetry test: ``Vd = +Vx``, ``Vs = -Vx`` around 0.

    Returns ``Vx`` and ``Ix = Id`` with its first three AD derivatives. A
    symmetric (physically consistent) model gives an odd ``Ix`` whose odd
    derivatives are continuous at ``Vx = 0``.
    """
    x = jnp.linspace(-vx_max, vx_max, n)
    k = probe.outputs.index("Id")
    path = probe._path({"g": vg, "b": vb}, {"d": 1.0, "s": -1.0})

    def f(xv):
        V, P = path(xv)
        return probe.evaluate(V, P)[0][k]

    def deriv(fn):
        return lambda xv: jax.jvp(fn, (xv,), (jnp.ones_like(xv),))[1]

    f1 = deriv(f)
    f2 = deriv(f1)
    f3 = deriv(f2)
    return {"vx": np.asarray(x), **{name: np.asarray(jax.jit(jax.vmap(fn))(x))
                                    for name, fn in (("i", f), ("d1", f1),
                                                     ("d2", f2), ("d3", f3))}}


_NEWTON_FNS: dict[Callable, tuple[Callable, Callable]] = {}


def newton_iterations(residual: Callable, z0, n_nodes: int, args: Any = None,
                      max_steps: int = 200, rtol: float = 1e-6, atol: float = 1e-9,
                      max_dv: float = 1.0) -> tuple[Any, int, bool]:
    """Voltax's damped Newton (`voltax.analysis`), counting iterations.

    `residual(z, args)` should be a module-level function so the jitted
    Jacobian is reused across calls with different `args` (e.g. circuits).
    Returns ``(z, iterations, converged)``; for benchmarking the convergence
    of exact vs smooth models.
    """
    if residual not in _NEWTON_FNS:
        _NEWTON_FNS[residual] = (jax.jit(jax.jacfwd(residual)), jax.jit(residual))
    J, r = _NEWTON_FNS[residual]
    z = jnp.asarray(z0)
    for k in range(1, max_steps + 1):
        dz = jnp.linalg.solve(J(z, args), -r(z, args))
        dv = float(jnp.max(jnp.abs(dz[:n_nodes]), initial=0.0))
        scale = min(1.0, max_dv / (dv + 1e-300))
        z = z + scale * dz
        if not bool(jnp.all(jnp.isfinite(z))):
            return z, k, False
        if scale == 1.0 and bool(jnp.all(jnp.abs(dz) <= atol + rtol * jnp.abs(z))):
            return z, k, True
    return z, max_steps, False


def fmt(x: Any) -> str:
    """Compact number formatting for report tables."""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return "0" if x == 0 else f"{x:.2g}"
    return str(x)
