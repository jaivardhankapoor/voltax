"""Loaded Verilog-A modules and the `Element` classes compiled from them.

    bsim4 = vx.va.load("bsim4")                      # fetched on first use
    NCH = bsim4.element({"vth0": 0.41, ...}, polarity="n", name="nch")
    m1 = NCH(("d", "g", "0", "0"), w=1e-6, l=45e-9)   # one device
    b.add(m1, name="M1")

Binding times (what is static vs traced) are decided per element class:

* integer parameters (model selectors such as ``capmod``, and integer
  instance parameters such as ``nf``) are always **static**;
* real **instance** parameters (``w``, ``l``, ``ad``, ...) are always
  **traced**: they are per-device arrays, differentiable, and devices with
  different values still fuse into one vectorized group;
* real **model** parameters are static unless listed in `differentiable`
  (or ``differentiable="all"``), in which case they live in a shared
  `VAParams` and receive gradients aggregated over all devices of the model.

Static parameters are folded into the generated code, so branches on them
disappear; traced ones keep their ``if``s as NaN-guarded ``where`` merges.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..element import Element
from .compiler import Compiled, Config, Evaluator, S, compile_module
from .parser import Module, Param, parse
from .preprocess import Line, VAError, preprocess

MODELS_DIR = Path(__file__).parent / "models"
"""Directory with bundled Verilog-A sources and model cards (see its NOTICE;
BSIM4 is not bundled but fetched on first use, see `voltax.va.fetch`)."""

DEFAULT_TEMPERATURE = 300.15
"""27 C, the SPICE default circuit temperature (K)."""

NGSPICE_BSIM4_OVERRIDES = {
    "KboQ": "8.617087e-5",
    "P_Q": "1.60219e-19",
    "P_EPS0": "8.85418e-12",
    "M_PI": "3.141592654",
    "BSIM4polyDepletion(phi, ngate, epsgate, coxe, Vgs, Vgs_eff, dVgs_eff_dVg)": (
        "if ((ngate > 1.0e18) && (ngate < 1.0e25) && (Vgs > phi) && (epsgate!=0)) "
        "begin T9 = 1.0e6 * 1.6021766208e-19 * epsgate * ngate / (coxe * coxe); "
        "T8 = Vgs - phi; T4 = sqrt(1.0 + 2.0 * T8 / T9); "
        "T2 = 2.0 * T8 / (T4 + 1.0); T3 = 0.5 * T2 * T2 / T9; "
        "T7 = 1.12 - T3 - 0.05; T6 = sqrt(T7 * T7 + 0.224); "
        "T5 = 1.12 - 0.5 * (T7 + T6); Vgs_eff = Vgs - T5; "
        "dVgs_eff_dVg = 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6); end "
        "else begin Vgs_eff = Vgs; dVgs_eff_dVg = 1.0; end"),
}
"""Macro overrides that make the BSIM4 4.8 ``bsim4.va`` use the same physical
constants as ngspice's C BSIM4 (``b4temp.c``/``b4set.c``/``b4ld.c``):
``KboQ = 8.617087e-5`` (the VA computes ``P_K/P_Q`` = 8.617343e-5 with the
NIST-1998 constants), ``Charge_q = 1.60219e-19``, ``EPS0 = 8.85418e-12``,
``PI = 3.141592654``, and ``CHARGE = 1.6021766208e-19`` in poly depletion.
Use ``vx.va.load("bsim4", overrides=NGSPICE_BSIM4_OVERRIDES)`` to reproduce
ngspice to round-off; the default keeps the Verilog-A source as published."""

NGSPICE_BSIM4_FORCE = {"tnoimod": 0}
"""Parameter values that must override a card to get ngspice's model for
DC/AC/transient analysis. ``tnoimod=1`` (thermal-noise model 1, set by the
sky130 cards) only matters for noise analysis in ngspice, which creates the
drain/source prime nodes for it only when a noise analysis is requested
(``b4set.c``). The Verilog-A creates them whenever ``rdsmod != 0 || tnoimod
== 1``, and with ``rdsmod=0`` their series conductance is 0: the channel then
floats between two internal nodes and carries no current."""

NGSPICE_BSIM4_DEFAULTS = {"gidlmod": 0, "cvchargemod": 0, "aigbacc": 1.36e-2,
                          "lwn": 1.0, "lc": 5e-9}
"""Parameter defaults where ngspice's ``b4set.c`` differs from
``bsim4.va`` (whose defaults are ``gidlmod=1``, ``cvchargemod=1``,
``aigbacc=9.49e-4``, ``lwn=0``, ``lc=0``). Merge under a card,
``{**NGSPICE_BSIM4_DEFAULTS, **card}``, to get ngspice's model for a card
that does not set them. (``vtl`` and ``phig`` also differ but are only
used when given.)"""


# =============================================================================
# Traced model parameters
# =============================================================================


class VAParams(eqx.Module):
    """Differentiable model parameters shared by all devices of a model card.

    Positive parameters (declared ``from (0:...``) are stored as their log,
    under the key ``log_<name>``, like every other Voltax log-space
    parameter. `get` / `replace` use physical values.
    """

    values: dict[str, Array]

    def get(self, name: str) -> Array:
        """Physical value of parameter `name`."""
        name = name.lower()
        if f"log_{name}" in self.values:
            return jnp.exp(self.values[f"log_{name}"])
        return self.values[name]

    def physical(self) -> dict[str, Array]:
        """All parameters, physical values, keyed by name."""
        return {k.removeprefix("log_"): (jnp.exp(v) if k.startswith("log_") else v)
                for k, v in self.values.items()}

    def replace(self, **values: Any) -> "VAParams":
        """Copy with parameters replaced (physical values)."""
        new = dict(self.values)
        for k, v in values.items():
            k = k.lower()
            v = jnp.asarray(v, dtype=float)
            if f"log_{k}" in new:
                new[f"log_{k}"] = jnp.log(v)
            elif k in new:
                new[k] = v
            else:
                raise KeyError(f"{k!r} is not a differentiable parameter of this "
                               f"model; differentiable: {sorted(self.physical())}")
        return VAParams(new)


# =============================================================================
# Loaded module
# =============================================================================


@dataclass
class ParamInfo:
    """A Verilog-A parameter as seen from Python."""

    name: str
    type: str
    instance: bool
    positive: bool
    """Range excludes zero from below (``from (0:...``): stored in log-space."""
    param: Param = field(repr=False)


class VAModel:
    """A parsed Verilog-A module, ready to be specialized into `Element`s.

    Create with `load`. Use `element` to compile an element class for one
    model card, or `compile` for lower-level access to the generated code.
    """

    def __init__(self, module: Module, lines: list[Line], source_hash: str):
        self.module = module
        self.lines = lines
        self.source_hash = source_hash
        self._macros = {(ln.file, ln.lineno): ln.macros for ln in lines if ln.macros}
        self._lower = {p.lower(): p for p in module.params}
        self._classes: dict[Any, type] = {}

    @property
    def name(self) -> str:
        return self.module.name

    def macros_at(self, file: str, line: int) -> tuple[str, ...]:
        return self._macros.get((file, line), ())

    @cached_property
    def params(self) -> dict[str, ParamInfo]:
        """Parameters by lower-case name (aliases included)."""
        out = {}
        for p in self.module.params.values():
            if p.local:
                continue
            out[p.name.lower()] = ParamInfo(p.name, p.type, p.instance,
                                            _is_positive(p), p)
        for alias, target in self.module.aliases.items():
            out[alias.lower()] = out[target.lower()]
        return out

    @cached_property
    def instance_params(self) -> dict[str, ParamInfo]:
        return {k: v for k, v in self.params.items() if v.instance}

    def __repr__(self) -> str:
        m = self.module
        return (f"VAModel({m.name!r}: ports {m.ports}, {len(m.internal_nodes)} "
                f"internal nodes, {len(self.params)} parameters)")

    # ----------------------------------------------------------- compiling

    def compile(self, params: Mapping[str, Any] | None = None, *,
                differentiable: Iterable[str] | str = (),
                instance: Mapping[str, Any] | None = None,
                temperature: float | None = DEFAULT_TEMPERATURE,
                smooth: float | Mapping[int, float] | None = None,
                simparams: Mapping[str, float] | None = None,
                observe: Iterable[str] = (),
                instrument: bool = False,
                equality: str = "value") -> Compiled:
        """Compile with model card `params` (and given `instance` params).

        Returns the `Compiled` artifact (generated source, functions, audit
        records). `differentiable` model parameters, all real instance
        parameters, and the temperature if ``temperature=None`` are traced.
        """
        card = self._normalize(params or {})
        inst = self._normalize(instance or {})
        traced = self._differentiable(differentiable)
        static = {**card, **{k: v for k, v in inst.items()
                             if self.params[k].type != "real"}}
        cfg = Config(static=static, traced_model=traced,
                     traced_inst=frozenset(self._real_instance()),
                     given=frozenset(card) | frozenset(inst),
                     temperature=temperature, smooth=_smooth_arg(smooth),
                     simparams={"gmin": 1e-12, **(simparams or {})},
                     observe=tuple(observe), instrument=instrument,
                     equality=_check_equality(equality))
        return compile_module(self.module, cfg, self.macros_at, self.name)

    def element(self, params: Mapping[str, Any] | None = None, *,
                polarity: Literal["n", "p"] | None = None,
                name: str | None = None,
                differentiable: Iterable[str] | str = (),
                temperature: float | None = DEFAULT_TEMPERATURE,
                smooth: float | Mapping[int, float] | None = None,
                equality: str = "value",
                simparams: Mapping[str, float] | None = None,
                ranges: Literal["error", "warn", "ignore"] = "error",
                defaults: Mapping[str, Any] | None = None,
                force: Mapping[str, Any] | None = None,
                ) -> type["VAElement"]:
        """An `Element` subclass for model card `params`.

        Args:
            params: Model parameters (case-insensitive names; unknown names
                warn). Integer parameters must be integral.
            defaults: Values used for parameters the card does not set
                (e.g. `NGSPICE_BSIM4_DEFAULTS`).
            force: Values that override the card (e.g.
                `NGSPICE_BSIM4_FORCE`).
            ranges: What to do when a model-card value violates the range the
                Verilog-A source declares (``from [0:1]`` ...): ``"error"``
                (default), ``"warn"`` or ``"ignore"``. Foundry cards often
                violate declared ranges that SPICE's C models do not enforce
                (e.g. negative ``lint``); the netlist hook uses ``"warn"``.
            polarity: ``"n"``/``"p"`` sets the ``type`` parameter (+1/-1) of
                models that have one (BSIM, PSP, ...), unless the card does.
            name: Class name (default: the module name); also the default
                group name prefix.
            differentiable: Real model parameters to trace (shared, with
                gradients), or ``"all"``.
            temperature: Device temperature in kelvin (static), or None to
                trace it as the model parameter ``"$temperature"``.
            smooth: Width of the smooth surrogates for ``min``/``max``/
                ``abs`` and ``if (x < b) x = b`` clamps on traced values:
                one width for all such sites, ``{source_line: width}`` for
                selected ones, or None (exact Verilog-A semantics).
            equality: Derivatives at branches on ``x == c`` (e.g. BSIM4's
                ``if (Vds == 0.0) Vdseff = 0.0``): ``"value"`` differentiates
                the branch taken (plain AD; zero slope at that one point),
                ``"limit"`` keeps the value but differentiates the general
                branch, like SPICE's hand-written Jacobians. Values are
                identical either way.
            simparams: ``$simparam`` values (default ``gmin=1e-12``).
        """
        card = {**self._normalize(defaults or {}), **self._normalize(params or {}),
                **self._normalize(force or {})}
        if polarity is not None:
            if polarity not in ("n", "p"):
                raise ValueError(f"polarity must be 'n' or 'p', got {polarity!r}")
            if "type" in self.params and "type" not in card:
                card["type"] = 1 if polarity == "n" else -1
        traced = self._differentiable(differentiable)
        smooth = _smooth_arg(smooth)
        key = (tuple(sorted((k, _hashable(v)) for k, v in card.items())), traced,
               temperature, _hashable(smooth),
               tuple(sorted((simparams or {}).items())), name, polarity,
               _check_equality(equality))
        if key in self._classes:
            return self._classes[key]
        self._check_ranges(card, "model", ranges)
        defaults = self.static_defaults(card)
        model_values = {}
        for k in sorted(traced):
            v = defaults.get(k)
            if not isinstance(v, S):
                raise VAError(f"cannot make {k!r} differentiable: its default "
                              "is not a constant; give it in the model card")
            if self.params[k].positive:
                model_values[f"log_{k}"] = jnp.log(jnp.asarray(float(v.v)))
            else:
                model_values[k] = jnp.asarray(float(v.v))
        if temperature is None:
            model_values["$temperature"] = jnp.asarray(DEFAULT_TEMPERATURE)
        inst_defaults = {}
        for k in self._real_instance():
            v = defaults.get(k)
            inst_defaults[k] = float(v.v) if isinstance(v, S) else None
        cls_name = _identifier(name or self.name)
        attrs = dict(
            va=self,
            card=card,
            traced_model=traced,
            temperature=temperature,
            smooth=smooth,
            equality=equality,
            simparams={"gmin": 1e-12, **(simparams or {})},
            inst_defaults=inst_defaults,
            default_model=VAParams(model_values),
            terminals=tuple(self.module.ports),
            polarity_label=polarity,
            _compiled_cache={},
            __module__=__name__,
            __doc__=f"Verilog-A module {self.name!r} compiled for model card "
                    f"{name or '(unnamed)'} ({len(card)} parameters given).",
        )
        cls = type(cls_name, (VAElement,), attrs)
        self._classes[key] = cls
        return cls

    # ------------------------------------------------------------- helpers

    def _normalize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        unknown = []
        for k, v in params.items():
            key = k.lower()
            if key not in self.params:
                unknown.append(k)
                continue
            info = self.params[key]
            out[info.name.lower()] = v
        if unknown:
            warnings.warn(f"{self.name}: ignoring unknown parameters "
                          f"{sorted(unknown)}", stacklevel=3)
        return out

    def _differentiable(self, names: Iterable[str] | str) -> frozenset:
        if isinstance(names, str):
            if names != "all":
                names = [names]
            else:
                return frozenset(k for k, p in self.params.items()
                                 if p.type == "real" and not p.instance
                                 and p.name.lower() == k)
        out = set()
        for n in names:
            info = self.params.get(n.lower())
            if info is None or info.instance or info.type != "real":
                raise ValueError(f"{n!r} is not a real model parameter of "
                                 f"{self.name}")
            out.add(info.name.lower())
        return frozenset(out)

    def _real_instance(self) -> list[str]:
        return [k for k, p in self.instance_params.items()
                if p.type == "real" and p.name.lower() == k]

    def static_defaults(self, card: Mapping[str, Any]) -> dict[str, Any]:
        """Value of every parameter for model card `card` (static where possible)."""
        cfg = Config(static=dict(card), given=frozenset(card))
        ev = Evaluator(self.module, cfg)
        ev.setup_params()
        return {k.lower(): v for k, v in ev.params.items()}

    def _check_ranges(self, values: Mapping[str, Any], what: str,
                      mode: str = "error") -> None:
        if mode not in ("error", "warn", "ignore"):
            raise ValueError(f"ranges must be 'error', 'warn' or 'ignore', "
                             f"got {mode!r}")
        if mode == "ignore":
            return
        for k, v in values.items():
            info = self.params[k]
            if info.type == "string" or isinstance(v, jax.core.Tracer):
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            msg = _range_violation(info.param, x, self)
            if msg:
                text = f"{self.name}: {what} parameter {k} = {x:g} {msg}"
                if mode == "error":
                    raise ValueError(text)
                warnings.warn(text, stacklevel=4)


def _floating_nodes(c: Compiled, params: Mapping[str, Array]) -> list[str]:
    """Internal nodes without a conductive path to any terminal: not reached
    through nonzero entries of df/dz (evaluated at a generic bias, first
    device). Such nodes make the DC operating point singular."""
    if not c.internal or any(isinstance(v, jax.core.Tracer)
                             for v in jax.tree.leaves(params)):
        return []
    names = list(c.terminals) + list(c.internal)
    z0 = jnp.asarray(np.linspace(0.13, 0.71, len(names)))
    P = {k: (v[:1] if jnp.ndim(v) else v) for k, v in params.items()}

    def rows(fn):
        def g(z):
            V = {n: z[i:i + 1] for i, n in enumerate(names)}
            t, inner = fn(V, P)[:2]
            return jnp.concatenate([jnp.broadcast_to(jnp.asarray(r, float), (1,))
                                    for r in (*t, *inner)])
        return np.asarray(jax.jacfwd(g)(z0))

    with jax.disable_jit():
        A = np.abs(rows(c.f))
    A = (A + A.T) > 0
    reached = set(range(len(c.terminals)))
    frontier = list(reached)
    while frontier:
        i = frontier.pop()
        for j in np.nonzero(A[i])[0]:
            if j not in reached:
                reached.add(int(j))
                frontier.append(int(j))
    return [n for k, n in enumerate(names) if k not in reached
            and not n.startswith("i(")]


def _check_equality(mode: str) -> str:
    if mode not in ("value", "limit"):
        raise ValueError(f"equality must be 'value' or 'limit', got {mode!r}")
    return mode


def _hashable(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def _smooth_arg(smooth: Any) -> float | dict[int, float] | None:
    if smooth is None or isinstance(smooth, (int, float)):
        if smooth is not None and smooth <= 0:
            raise ValueError("smooth width must be positive")
        return None if smooth is None else float(smooth)
    return {int(k): float(v) for k, v in dict(smooth).items()}


def _identifier(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else f"VA_{out}"


def _const(node: Any, model: VAModel | None) -> float | None:
    from .parser import Name, Num, Unary

    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Name) and node.id == "inf":
        return math.inf
    if isinstance(node, Unary) and node.op == "-":
        x = _const(node.a, model)
        return None if x is None else -x
    return None


def _is_positive(p: Param) -> bool:
    for lo_b, lo, _, _ in p.ranges:
        if lo_b == "(" and _const(lo, None) == 0.0:
            return True
    return False


def _range_violation(p: Param, x: float, model: VAModel) -> str | None:
    for lo_b, lo, hi, hi_b in p.ranges:
        a, b = _const(lo, model), _const(hi, model)
        if a is None or b is None:
            continue  # bounds that depend on other parameters: not checked
        ok_lo = x > a if lo_b == "(" else x >= a
        ok_hi = x < b if hi_b == ")" else x <= b
        if not (ok_lo and ok_hi):
            return f"is outside its range {lo_b}{a:g}:{b:g}{hi_b}"
    for ex in p.excludes:
        if isinstance(ex, tuple):
            lo_b, lo, hi, hi_b = ex
            a, b = _const(lo, model), _const(hi, model)
            if a is None or b is None:
                continue
            in_lo = x > a if lo_b == "(" else x >= a
            in_hi = x < b if hi_b == ")" else x <= b
            if in_lo and in_hi:
                return f"is in the excluded range {lo_b}{a:g}:{b:g}{hi_b}"
        else:
            c = _const(ex, model)
            if c is not None and x == c:
                return f"is the excluded value {c:g}"
    return None


# =============================================================================
# Element base class
# =============================================================================


class VAElement(Element):
    """Base class of elements compiled from Verilog-A (see `VAModel.element`).

    Attributes:
        inst: Per-device real instance parameters, ``(N,)`` arrays; positive
            ones under ``log_<name>``.
        mult: Device multiplicity ``m`` (parallel copies), ``(N,)``.
        model: Shared differentiable model parameters (`VAParams`).
        int_inst: Static integer instance parameters ``((name, value), ...)``.
        given_inst: Instance parameters given explicitly (for
            ``$param_given``).
    """

    shared = ("model",)
    inst: dict[str, Array]
    mult: Array
    model: VAParams
    int_inst: tuple = eqx.field(static=True)
    given_inst: frozenset = eqx.field(static=True)

    # set on generated subclasses
    va: ClassVar[Any] = None
    card: ClassVar[Any] = None
    traced_model: ClassVar[Any] = frozenset()
    temperature: ClassVar[Any] = DEFAULT_TEMPERATURE
    smooth: ClassVar[Any] = None
    equality: ClassVar[Any] = "value"
    simparams: ClassVar[Any] = None
    inst_defaults: ClassVar[Any] = None
    default_model: ClassVar[Any] = None
    polarity_label: ClassVar[Any] = None
    _compiled_cache: ClassVar[Any] = None

    def __init__(self, nodes: Any, *, m: Any = 1.0, model: VAParams | None = None,
                 **instance: Any):
        if self.va is None:
            raise TypeError("use VAModel.element(...) to create an element class")
        self.nodes = self._devices(nodes)
        given = self.va._normalize(instance)
        self.va._check_ranges({k: v for k, v in given.items()
                               if not isinstance(v, jax.core.Tracer)
                               and np.ndim(v) == 0}, "instance")
        ints = []
        for k, info in self.va.instance_params.items():
            if info.type != "real" and k in given and info.name.lower() == k:
                val = given[k]
                if float(val) != int(round(float(val))):
                    raise ValueError(f"integer instance parameter {k} = {val}")
                ints.append((k, int(round(float(val)))))
        inst = {}
        for k, default in self.inst_defaults.items():
            val = given.get(k, default)
            if val is None:
                raise ValueError(f"instance parameter {k!r} has no constant default; "
                                 "give it explicitly")
            if self.va.params[k].positive:
                inst[f"log_{k}"] = self._log_per_device(val)
            else:
                inst[k] = self._per_device(val)
        self.inst = inst
        self.mult = self._per_device(m)
        self.model = self.default_model if model is None else model
        self.int_inst = tuple(sorted(ints))
        self.given_inst = frozenset(given)

    # ----------------------------------------------------------- compiled

    @property
    def compiled(self) -> Compiled:
        """The `Compiled` artifact for this group's static configuration."""
        key = (self.int_inst, self.given_inst)
        cache = self._compiled_cache
        if key not in cache:
            cfg = Config(static={**self.card, **dict(self.int_inst)},
                         traced_model=self.traced_model,
                         traced_inst=frozenset(self.inst_defaults),
                         given=frozenset(self.card) | self.given_inst,
                         temperature=self.temperature, smooth=self.smooth,
                         equality=self.equality,
                         simparams=dict(self.simparams))
            c = compile_module(self.va.module, cfg, self.va.macros_at,
                               type(self).__name__)
            cache[key] = (c, jax.jit(c.f), jax.jit(c.q))
            floating = _floating_nodes(c, self.params())
            card = {**self.card, **dict(self.int_inst)}
            if (self.va.module.name.lower().startswith("bsim4")
                    and card.get("tnoimod") == 1 and card.get("rdsmod", 0) == 0
                    and set(c.internal) & {"di", "si"}):
                warnings.warn(
                    f"{type(self).__name__}: BSIM4 card with tnoimod=1 and "
                    "rdsmod=0: the Verilog-A adds drain/source prime nodes "
                    "with zero series conductance, so the channel is "
                    "disconnected from d/s and carries no current (ngspice "
                    "creates them only for noise analysis). Use "
                    "force=vx.va.NGSPICE_BSIM4_FORCE (tnoimod=0), as "
                    "vx.va.ngspice_bsim4_models() does.", stacklevel=3)
            elif floating:
                warnings.warn(
                    f"{type(self).__name__}: internal nodes {floating} have no "
                    "conductive path to any terminal with this card (the "
                    "device will carry no DC current through them). "
                    "Check the topology flags; for BSIM4 cards with tnoimod=1 "
                    "and rdsmod=0 use force=vx.va.NGSPICE_BSIM4_FORCE.",
                    stacklevel=3)
        return cache[key][0]

    @property
    def source(self) -> str:
        """Generated Python source of this group's model code."""
        return self.compiled.source

    @property
    def n_internal(self) -> int:  # type: ignore[override]
        return len(self.compiled.internal)

    @property
    def internal_names(self) -> tuple[str, ...]:  # type: ignore[override]
        return self.compiled.internal

    @property
    def polarity(self) -> str | None:
        """``"n"``/``"p"`` if built with a polarity (used for group names)."""
        return self.polarity_label

    def _fns(self):
        self.compiled  # noqa: B018  (populate the cache)
        return self._compiled_cache[(self.int_inst, self.given_inst)]

    # ------------------------------------------------------------ physics

    def params(self) -> dict[str, Array]:
        """The traced-parameter dict ``P`` passed to the generated code."""
        P = dict(self.model.physical())
        for k, v in self.inst.items():
            P[k.removeprefix("log_")] = jnp.exp(v) if k.startswith("log_") else v
        return P

    def _call(self, fn, v: Array, x: Array):
        c = self.compiled
        V = {name: v[..., k] for k, name in enumerate(c.terminals)}
        V.update({name: x[..., k] for k, name in enumerate(c.internal)})
        rows_t, rows_i = fn(V, self.params())
        shape = (self.size,)
        m = self.mult
        I = jnp.stack([jnp.broadcast_to(r, shape) for r in rows_t], -1) * m[:, None]
        if not c.internal:
            return I, None
        F = jnp.stack([jnp.broadcast_to(r, shape) * (m if kcl else 1.0)
                       for r, kcl in zip(rows_i, c.kcl_rows)], -1)
        return I, F

    def currents(self, v, x, t):
        return self._call(self._fns()[1], v, x)

    def charges(self, v, x):
        I, F = self._call(self._fns()[2], v, x)
        return I, F

    # ------------------------------------------------------- per-device I/O

    def get(self, field: str, index: int | slice = slice(None)) -> Array:
        key = field.lower()
        if f"log_{key}" in self.inst:
            return jnp.exp(self.inst[f"log_{key}"][index])
        if key in self.inst:
            return self.inst[key][index]
        if key == "m":
            return self.mult[index]
        return super().get(field, index)

    def set(self, index: int, **values: Any) -> "VAElement":
        out = self
        for k, v in values.items():
            key = k.lower()
            if key == "m":
                new = out.mult.at[index].set(v)
                out = eqx.tree_at(lambda e: e.mult, out, new)
                continue
            if f"log_{key}" in out.inst:
                name, v = f"log_{key}", jnp.log(jnp.asarray(v, dtype=float))
            elif key in out.inst:
                name = key
            else:
                raise AttributeError(f"{type(self).__name__} has no real instance "
                                     f"parameter {k!r}")
            new = dict(out.inst)
            new[name] = new[name].at[index].set(v)
            out = eqx.tree_at(lambda e: e.inst, out, new)
        return out

    def with_model(self, **values: Any) -> "VAElement":
        """Copy with differentiable model parameters replaced (physical)."""
        return eqx.tree_at(lambda e: e.model, self, self.model.replace(**values))


# =============================================================================
# Loading
# =============================================================================

_LOADED: dict[Any, VAModel] = {}


def load(source: str | Path, module: str | None = None, *,
         defines: Mapping[str, str] | None = None,
         overrides: Mapping[str, str] | None = None,
         include_dirs: Iterable[str | Path] = ()) -> VAModel:
    """Load (preprocess + parse) a Verilog-A module.

    Args:
        source: A path to a ``.va`` file, Verilog-A source text (anything
            containing ``module``), or the name of a known model:
            ``"bsim4"`` is resolved from ``$VOLTAX_BSIM4_VA``, the cache, or
            downloaded from a pinned, checksummed URL on first use (see
            `voltax.va.fetch`; its licence, CC BY-NC 4.0, is shown once).
        module: Module name if the file defines several (default: the last).
        defines: Macros defined before the source (like ``-D``).
        overrides: Macros the source cannot redefine, e.g.
            ``{"KboQ": "8.617087e-5"}`` to pin a physical constant.
        include_dirs: Extra `` `include`` search directories.

    Results are cached by (source text, options).
    """
    text, file = _read(source)
    key = (hashlib.sha256(text.encode()).hexdigest(), module,
           tuple(sorted((defines or {}).items())),
           tuple(sorted((overrides or {}).items())),
           tuple(str(d) for d in include_dirs), file)
    if key in _LOADED:
        return _LOADED[key]
    lines = preprocess(text, file, dict(defines or {}), dict(overrides or {}),
                       list(include_dirs))
    modules = parse(lines)
    if not modules:
        raise VAError("no module found", file)
    if module is None:
        mod = modules[-1]
    else:
        found = [m for m in modules if m.name == module]
        if not found:
            raise VAError(f"no module {module!r}; found "
                          f"{[m.name for m in modules]}", file)
        mod = found[0]
    out = VAModel(mod, lines, key[0])
    _LOADED[key] = out
    return out


def _read(source: str | Path) -> tuple[str, str]:
    if isinstance(source, Path):
        return source.read_text(), str(source)
    if "module" in source and ("\n" in source or ";" in source):
        return source, "<string>"
    path = Path(source)
    if path.is_file():
        return path.read_text(), str(path)
    bundled = MODELS_DIR / f"{source}.va"
    if bundled.is_file():
        return bundled.read_text(), str(bundled)
    from .fetch import REMOTES, resolve

    remote = resolve(source)
    if remote is not None:
        return remote.read_text(), str(remote)
    raise FileNotFoundError(f"no Verilog-A file {source!r} (bundled models: "
                            f"{sorted(p.stem for p in MODELS_DIR.glob('*.va'))}, "
                            f"downloadable: {sorted(REMOTES)})")

