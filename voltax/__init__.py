"""Voltax: differentiable circuit simulation on JAX.

Importing voltax enables 64-bit floats in JAX (see `voltax._config`).
"""

from . import (
    _config,  # noqa: F401  (must run before any array is created)
    library,
    measure,
    signals,
    sparse,
    va,
)
from .analysis import (
    ACSolution,
    Options,
    Solution,
    ac,
    dc,
    dc_sweep,
    linearize,
    solve_root,
    time_grid,
    transient,
)
from .builder import CircuitBuilder
from .circuit import Circuit, Layout
from .element import Element, concatenate, positive, through, two_terminal
from .elements import (
    BJT,
    CCCS,
    CCVS,
    EKVMOSFET,
    VCCS,
    VCVS,
    VT_300K,
    BehavioralSource,
    Capacitor,
    Conductance,
    CurrentSource,
    Diode,
    IdealOpAmp,
    Inductor,
    Level1MOSFET,
    MOSProcess,
    NonlinearCapacitor,
    NonlinearResistor,
    OpAmp,
    Resistor,
    Switch,
    Transformer,
    VoltageSource,
)
from .netlist import parse_netlist, parse_netlist_file, parse_value

__version__ = "0.1.0"

__all__ = [
    # circuits
    "Circuit",
    "CircuitBuilder",
    "Layout",
    "parse_netlist",
    "parse_netlist_file",
    "parse_value",
    # analyses
    "ACSolution",
    "Options",
    "Solution",
    "ac",
    "dc",
    "dc_sweep",
    "linearize",
    "solve_root",
    "time_grid",
    "transient",
    # writing elements
    "Element",
    "concatenate",
    "positive",
    "through",
    "two_terminal",
    # elements
    "BJT",
    "BehavioralSource",
    "CCCS",
    "CCVS",
    "Capacitor",
    "Conductance",
    "CurrentSource",
    "Diode",
    "EKVMOSFET",
    "IdealOpAmp",
    "Inductor",
    "Level1MOSFET",
    "MOSProcess",
    "NonlinearCapacitor",
    "NonlinearResistor",
    "OpAmp",
    "Resistor",
    "Switch",
    "Transformer",
    "VCCS",
    "VCVS",
    "VT_300K",
    "VoltageSource",
    # submodules
    "library",
    "measure",
    "signals",
    "sparse",
    "va",
]
