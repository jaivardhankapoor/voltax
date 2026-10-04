# Development

## Layout

```text
voltax/
  element.py        Element base class (the device-model contract)
  circuit.py        Circuit: element groups + static Layout; f(z,t), q(z)
  analysis.py       Newton + implicit differentiation; dc, transient, ac
  builder.py        CircuitBuilder: named nodes, device fusion, scopes
  netlist/          SPICE parser: lexer, preprocess (.include/.lib, subckt
                    flattening), expressions, models (binning, hook), emit
  signals.py        differentiable source waveforms
  elements/         built-in device models
  library/cmos.py   CMOS gates and arithmetic blocks
tests/              pytest suite (physics, gradients, parser, docs)
examples/           numbered, literate example scripts
benchmarks/         accuracy/speed comparisons against ngspice
docs/               this site (MkDocs Material)
```

## Common commands

```bash
uv sync --all-extras                    # .venv from uv.lock
uv run pytest                           # unit tests (~3 min on CPU)
uv run pytest -m extended               # + docs snippets and example smoke runs
uvx ruff check .
uv run mkdocs serve                     # live docs at http://127.0.0.1:8000
VOLTAX_FAST=1 uv run python examples/01_rc_fitting.py
```

## Principles

- **One residual, many analyses.** Devices only describe \(f\) and \(q\).
  New analyses (noise, harmonic balance, sensitivity) should be built on
  `Circuit.f`/`Circuit.q`, not on device internals.
- **Physics in elements, bookkeeping in the circuit.** An element never sees
  global indices; the circuit never knows device equations.
- **Everything differentiable, nothing silent.** Prefer smooth models; raise
  on unsupported input (e.g. unknown netlist elements) instead of skipping.
- **Physical units at the API, log-space in storage.**
- Tests compare against closed forms or finite differences, not against
  previously printed numbers.
