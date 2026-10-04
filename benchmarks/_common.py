"""Shared helpers for the voltax-vs-ngspice benchmarks.

Import this module *before* `voltax` / `jax`: it pins JAX to a single CPU
thread so the runtime comparison with (single-threaded) ngspice is fair.

Contents:

* `FAST` / `pick`: problem-size switch driven by ``VOLTAX_FAST=1``.
* `run_ngspice`: run a netlist in batch mode, return the saved vectors (read
  from a binary rawfile, so values are exact doubles) and timings.
* `parse_raw`: SPICE rawfile reader (binary or ASCII, real or complex).
* `time_call`: first-call (compile + run) and steady-state timing of a JAX call.
* `errors`, `print_table`, `figure_path`: reporting.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
# XLA's CPU LU (and scipy) call OpenBLAS, which has its own thread pool: pin it
# too. On a busy machine its spinning threads can slow a dense LU 100x or more.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Sequence  # noqa: E402

import numpy as np  # noqa: E402

try:  # also enforce it if BLAS was loaded before this module was imported
    from threadpoolctl import threadpool_limits  # noqa: E402

    threadpool_limits(1)
except ImportError:  # pragma: no cover
    pass

FAST = os.environ.get("VOLTAX_FAST", "0") not in ("", "0")
"""True when ``VOLTAX_FAST=1``: benchmarks shrink their problem sizes."""

FIGURES = Path(__file__).resolve().parent / "figures"
NGSPICE = shutil.which("ngspice") or "/usr/bin/ngspice"


def pick(fast: Any, full: Any) -> Any:
    """`fast` in ``VOLTAX_FAST`` mode, else `full`."""
    return fast if FAST else full


def num(x: Any) -> str:
    """Format a number for a netlist with full double precision."""
    return f"{float(x):.17g}"


def header(title: str) -> None:
    mode = "FAST" if FAST else "full"
    print("=" * 72)
    print(f"{title}  [{mode} mode]")
    print("=" * 72)


# =============================================================================
# ngspice
# =============================================================================


@dataclass
class NgspiceResult:
    """Output of `run_ngspice`.

    Attributes:
        vectors: Saved vectors by lower-case name, e.g. ``"time"``, ``"v(out)"``.
        wall: Best (minimum over repeats) wall-clock time of the whole
            ``ngspice -b`` process (s), including start-up and netlist parsing.
        analysis: Best "Total analysis time" reported by ngspice (s); 1 ms
            resolution.
        stats: Other ``rusage`` counters of the last run (timepoints, ...).
    """

    vectors: dict[str, np.ndarray]
    wall: float
    analysis: float
    stats: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.vectors[name.lower()]


def run_ngspice(netlist: str, save: Sequence[str], repeats: int = 1) -> NgspiceResult:
    """Run `netlist` (one analysis directive, no ``.control``/``.end``) with
    ``ngspice -b`` and return the vectors in `save` (e.g. ``["v(out)"]``).

    The rawfile is written in binary, so the values are the doubles ngspice
    computed. The run is repeated `repeats` times for timing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        raw, deck = Path(tmp) / "out.raw", Path(tmp) / "deck.sp"
        deck.write_text(
            f"{netlist.rstrip()}\n"
            ".control\n"
            "set filetype=binary\n"
            "run\n"
            f"write {raw} {' '.join(save)}\n"
            "rusage all\n"
            ".endc\n"
            ".end\n"
        )
        walls, analyses = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            proc = subprocess.run([NGSPICE, "-b", str(deck)], capture_output=True,
                                  text=True, errors="replace")
            walls.append(time.perf_counter() - start)
            log = proc.stdout + proc.stderr
            if proc.returncode != 0 or not raw.exists():
                raise RuntimeError(f"ngspice failed:\n{log[-3000:]}")
            stats = _rusage(log)
            analyses.append(stats.get("Total analysis time (seconds)", float("nan")))
        plots = parse_raw(raw)
    return NgspiceResult(plots[0], min(walls),
                         min(analyses), stats)


def _rusage(log: str) -> dict[str, float]:
    out = {}
    for m in re.finditer(r"^(.+?)\s*=\s*([-+0-9.eE]+)\s*$", log, re.MULTILINE):
        try:
            out[m.group(1).strip()] = float(m.group(2))
        except ValueError:
            pass
    return out


def parse_raw(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Read a SPICE rawfile (binary or ASCII). Returns one dict per plot,
    mapping lower-case vector names to arrays (complex for AC plots)."""
    data = Path(path).read_bytes()
    plots, pos = [], 0
    while pos < len(data):
        meta: dict[str, str] = {}
        names: list[str] = []
        while True:  # header lines up to "Binary:" / "Values:"
            end = data.index(b"\n", pos)
            line = data[pos:end].decode("latin-1")
            pos = end + 1
            if line.startswith("Variables:"):
                n_vars = int(meta["no. variables"])
                for _ in range(n_vars):
                    end = data.index(b"\n", pos)
                    names.append(data[pos:end].decode().split()[1].lower())
                    pos = end + 1
                continue
            if line.startswith(("Binary:", "Values:")):
                break
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
        n_pts = int(meta["no. points"])
        complex_ = "complex" in meta.get("flags", "").lower()
        width = 2 if complex_ else 1
        if line.startswith("Binary:"):
            count = n_pts * len(names) * width
            arr = np.frombuffer(data, "<f8", count, pos)
            pos += 8 * count
        else:
            tokens, needed = [], n_pts * len(names)
            while len(tokens) < needed * width and pos < len(data):
                end = data.find(b"\n", pos)
                end = len(data) if end < 0 else end
                parts = data[pos:end].decode().replace(",", " ").split()
                pos = end + 1
                if not parts:
                    continue
                # each point starts with its index
                tokens += parts[1:] if len(tokens) % (len(names) * width) == 0 \
                    else parts
            arr = np.asarray(tokens[: needed * width], float)
        arr = arr.reshape(n_pts, len(names), width)
        values = arr[..., 0] + 1j * arr[..., 1] if complex_ else arr[..., 0]
        plots.append({name: values[:, k].copy() for k, name in enumerate(names)})
        while pos < len(data) and data[pos:pos + 1] in (b"\n", b"\r"):
            pos += 1
    return plots


# =============================================================================
# voltax timing
# =============================================================================


def time_call(fn: Callable[[], Any], repeats: int = 5) -> tuple[Any, float, float]:
    """Run ``fn()`` once cold, then `repeats` more times.

    Returns:
        ``(result, first_call_s, steady_s)``. The first call includes tracing
        and XLA compilation; `steady_s` is the best warm call (minimum, robust
        to machine noise), so ``first - steady`` approximates the compile time.
    """
    import jax

    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    first = time.perf_counter() - start
    warm = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        warm.append(time.perf_counter() - start)
    return out, first, min(warm)


# =============================================================================
# Reporting
# =============================================================================


def errors(test: Any, ref: Any) -> tuple[float, float]:
    """``(max |test - ref|, max |test - ref| / max |ref|)``."""
    test, ref = np.asarray(test), np.asarray(ref)
    err = float(np.max(np.abs(test - ref)))
    return err, err / max(float(np.max(np.abs(ref))), 1e-300)


def fmt_s(seconds: float) -> str:
    """Seconds as a compact human string (``12.3 ms``)."""
    if seconds != seconds:  # nan
        return "n/a"
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.0f} us"
    if seconds >= 1e-9:
        return f"{seconds * 1e9:.3g} ns"
    return f"{seconds * 1e12:.3g} ps"


def print_table(rows: Sequence[dict[str, Any]], title: str | None = None) -> None:
    """Print a list of dicts as an aligned text table (floats in %.3g)."""
    if not rows:
        return
    cols = list(rows[0])

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)

    table = [[cell(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(r[i]) for r in table)) for i, c in enumerate(cols)]
    if title:
        print(f"\n{title}")
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in table:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def timing_row(name: str, first: float, steady: float, ng: NgspiceResult,
               **extra: Any) -> dict[str, Any]:
    """Standard summary-table row."""
    return {
        "case": name,
        **extra,
        "voltax 1st call": fmt_s(first),
        "voltax compile": fmt_s(max(first - steady, 0.0)),
        "voltax run": fmt_s(steady),
        "ngspice wall": fmt_s(ng.wall),
        "ngspice analysis": fmt_s(ng.analysis) if ng.analysis else "<1 ms",
    }


def figure_path(name: str) -> Path:
    """``benchmarks/figures/<name>.png`` (directory created on demand)."""
    FIGURES.mkdir(exist_ok=True)
    return FIGURES / f"{name}.png"


def save_figure(fig: Any, name: str) -> None:
    path = figure_path(name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {path.relative_to(FIGURES.parent.parent)}")
