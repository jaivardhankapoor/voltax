"""Verilog-A compact models compiled to Voltax elements.

    import voltax as vx
    bsim4 = vx.va.load("bsim4")                       # fetched on first use
    name, kind, card = vx.va.bundled_card("freepdk45_nmos.inc")
    NCH = bsim4.element(card, polarity="n", name="nch")
    b.add(NCH(("d", "g", "0", "0"), w=90e-9, l=50e-9), name="M1")

Pipeline: `preprocess` (macros, includes) -> `parse` (AST) -> `compile_module`
(online partial evaluation to straight-line JAX code, static branches
resolved, traced branches as NaN-guarded ``where``) -> `VAElement` subclass.
See ``docs/concepts/verilog-a.md`` and `voltax.va.audit` for the
differentiability audit.
"""

from .cards import bundled_card, read_cards, spice_number
from .compiler import Compiled, Config, Risk, Site, compile_module
from .fetch import cache_dir, fetch
from .model import (
    DEFAULT_TEMPERATURE,
    MODELS_DIR,
    NGSPICE_BSIM4_DEFAULTS,
    NGSPICE_BSIM4_FORCE,
    NGSPICE_BSIM4_OVERRIDES,
    VAElement,
    VAModel,
    VAParams,
    load,
)
from .netlist import level_models, ngspice_bsim4_models
from .parser import parse
from .preprocess import VAError, preprocess

__all__ = [
    "Compiled",
    "Config",
    "DEFAULT_TEMPERATURE",
    "MODELS_DIR",
    "NGSPICE_BSIM4_DEFAULTS",
    "NGSPICE_BSIM4_FORCE",
    "NGSPICE_BSIM4_OVERRIDES",
    "Risk",
    "Site",
    "VAElement",
    "VAError",
    "VAModel",
    "VAParams",
    "bundled_card",
    "cache_dir",
    "fetch",
    "compile_module",
    "level_models",
    "ngspice_bsim4_models",
    "load",
    "parse",
    "preprocess",
    "read_cards",
    "spice_number",
]
