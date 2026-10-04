"""Circuit analyses: DC operating point, transient, and small-signal AC.

Every nonlinear solve goes through one function, `solve_root`, which runs
damped Newton in the forward pass and differentiates by the implicit function
theorem in the backward pass. For ``F(z*, p) = 0``::

    dL/dp = -lam^T dF/dp,    with   (dF/dz)^T lam = dL/dz*

so the cost of a gradient is one transposed linear solve per root, regardless
of how many Newton iterations the forward pass took.

Discretizations (for ``d/dt q(z) + f(z, t) = 0``, step ``h``):

* backward Euler: ``(q(z) - q(z_prev))/h + f(z, t) = 0``
* trapezoidal:    ``(q(z) - q(z_prev))/h + (f(z, t) + f(z_prev, t_prev))/2 = 0``

Linear algebra: circuit analyses build Newton Jacobians by stamping per-device
blocks (`voltax.sparse.jacobian`) and solve them either densely (LU) or with
the supernodal sparse LU of `voltax.sparse`, chosen by `Options.solver`.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from . import sparse
from .circuit import Circuit, Layout

# =============================================================================
# Options and results
# =============================================================================


SPARSE_THRESHOLD = 200
"""Default `Options.sparse_threshold` (state size); see benchmarks/README.md."""

DEFAULT_SOLVER = os.environ.get("VOLTAX_SOLVER", "auto")
"""Default `Options.solver`; the ``VOLTAX_SOLVER`` environment variable (read
at import) overrides it, e.g. to run a test suite with the sparse solver."""


class Options(eqx.Module):
    """Newton-solver settings (all static).

    Args:
        max_steps: Maximum Newton iterations per solve.
        rtol, atol: Converged when ``|dz| <= atol + rtol |z|`` elementwise.
        max_dv: Damping: largest node-voltage update per iteration (V).
        gmin: Conductance from every node to ground (S); aids convergence of
            floating / high-impedance nodes. SPICE uses 1e-12.
        gmin_steps: If plain Newton fails in `dc`, retry by gmin stepping:
            solve with ``gmin`` decreasing log-uniformly from `gmin_start`
            to `gmin` over this many solves, warm-starting each from the last.
            0 disables the fallback.
        gmin_start: First gmin of the homotopy (S).
        direct_steps: Newton iterations `dc` tries before falling back to
            gmin stepping.
        solver: Linear solver for the Newton (and adjoint and AC) systems:
            ``"dense"`` (LU of the dense Jacobian), ``"sparse"`` (supernodal
            sparse LU, see `voltax.sparse`), or ``"auto"``: sparse when the
            circuit has at least `sparse_threshold` unknowns.
        sparse_threshold: Smallest state size for which ``"auto"`` picks the
            sparse solver.
    """

    max_steps: int = eqx.field(static=True, default=100)
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-9)
    max_dv: float = eqx.field(static=True, default=1.0)
    gmin: float = eqx.field(static=True, default=1e-12)
    gmin_steps: int = eqx.field(static=True, default=10)
    gmin_start: float = eqx.field(static=True, default=1e-2)
    direct_steps: int = eqx.field(static=True, default=25)
    solver: Literal["auto", "dense", "sparse"] = eqx.field(static=True,
                                                           default=DEFAULT_SOLVER)
    sparse_threshold: int = eqx.field(static=True, default=SPARSE_THRESHOLD)

    def __check_init__(self):
        if self.solver not in ("auto", "dense", "sparse"):
            raise ValueError("solver must be 'auto', 'dense' or 'sparse', "
                             f"got {self.solver!r}")


class Solution(eqx.Module):
    """Result of `dc` or `transient`.

    Attributes:
        t: Time point(s): scalar for DC, ``(T,)`` for transient.
        z: State(s), shape ``(..., state_size)``.
        converged: Whether Newton converged, per time point.
    """

    t: Array
    z: Array
    converged: Array
    layout: Layout = eqx.field(static=True)

    def v(self, node: str | None = None) -> Array:
        """Voltage of `node` (all node voltages ``(..., n_nodes)`` if None)."""
        if node is None:
            return self.z[..., : self.layout.n_nodes]
        idx = self.layout.node(node)
        return jnp.zeros(self.z.shape[:-1]) if idx < 0 else self.z[..., idx]

    def i(self, device: str, which: str | None = None) -> Array:
        """Internal unknown of `device` (default: its first, e.g. its current).

        ``sol.i("V1")`` is the current through source ``V1`` (``p -> n``).
        """
        return self.z[..., self.layout.internal(device, which)]

    def check(self) -> "Solution":
        """Raise if any point failed to converge or is not finite; else return
        self, so you can write ``sol = vx.dc(c).check()``. Call outside `jit`.
        """
        bad = ~(self.converged & jnp.all(jnp.isfinite(self.z), axis=-1))
        if bool(jnp.any(bad)):
            where = "" if jnp.ndim(bad) == 0 else f" at t={self.t[jnp.argmax(bad)]}"
            raise RuntimeError(
                f"Newton did not converge{where}. Common causes: a node without "
                "a DC path to ground, a loop of voltage sources/inductors, or "
                "time steps too coarse for an edge (see docs: Analyses)."
            )
        return self

    def __getitem__(self, name: str) -> Array:
        """``sol["out"]`` is `v`; ``sol["V1.i"]`` is `i`."""
        if "." in name:
            device, which = name.rsplit(".", 1)
            return self.i(device, which)
        return self.v(name)


class ACSolution(eqx.Module):
    """Result of `ac`: complex phasors at each frequency.

    Attributes:
        freqs: ``(F,)`` frequencies in Hz.
        z: ``(F, state_size)`` complex small-signal state.
        op: The DC operating point the circuit was linearized at.
    """

    freqs: Array
    z: Array
    op: Solution
    layout: Layout = eqx.field(static=True)

    def v(self, node: str | None = None) -> Array:
        """Complex node voltage(s), like `Solution.v`."""
        if node is None:
            return self.z[..., : self.layout.n_nodes]
        idx = self.layout.node(node)
        return jnp.zeros(self.z.shape[:-1], complex) if idx < 0 else self.z[..., idx]

    def i(self, device: str, which: str | None = None) -> Array:
        return self.z[..., self.layout.internal(device, which)]

    def db(self, node: str) -> Array:
        """``20 log10 |v(node)|``."""
        return 20 * jnp.log10(jnp.abs(self.v(node)))

    def phase(self, node: str, unwrap: bool = True) -> Array:
        """Phase of `v(node)` in degrees, unwrapped along frequency by default."""
        angle = jnp.angle(self.v(node))
        return jnp.rad2deg(jnp.unwrap(angle) if unwrap else angle)


# =============================================================================
# Newton + implicit differentiation
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _Linear:
    """How Newton forms and solves ``J dz = -r`` (a static argument).

    With ``jacobian=None`` the Jacobian is ``jax.jacfwd`` of the residual
    (works for any residual); otherwise ``jacobian(z, params, args,
    structure)`` returns its stamped values on ``structure``'s pattern, and
    the system is solved with the sparse LU or densely.
    """

    jacobian: Callable | None = None
    structure: sparse.Structure | None = None
    sparse: bool = False

    def matrix(self, residual: Callable, z: Array, params: Any, args: Any) -> Array:
        if self.jacobian is None:
            return jax.jacfwd(lambda z_: residual(z_, params, args))(z)
        data = self.jacobian(z, params, args, self.structure)
        return data if self.sparse else self.structure.to_dense(data)

    def solve(self, J: Array, b: Array, transpose: bool = False) -> Array:
        if self.sparse:
            plan = self.structure.plan
            return sparse.solve(plan, sparse.factor(plan, J), b, transpose)
        return jnp.linalg.solve(J.T if transpose else J, b)


def _newton(residual: Callable, params: Any, args: Any, z0: Array, n_nodes: int,
            opts: Options, lin: _Linear, max_steps: Array | int | None = None
            ) -> tuple[Array, Array]:
    """Damped Newton iteration. Returns ``(z, converged)``.

    `max_steps` (default ``opts.max_steps``) may be a traced value.
    """
    max_steps = opts.max_steps if max_steps is None else max_steps

    def cond(carry):
        k, _, done = carry
        return (k < max_steps) & ~done

    def body(carry):
        k, z, _ = carry
        r = residual(z, params, args)
        dz = lin.solve(lin.matrix(residual, z, params, args), -r)
        dv_max = jnp.max(jnp.abs(dz[:n_nodes]), initial=0.0)
        scale = jnp.minimum(1.0, opts.max_dv / (dv_max + 1e-300))
        z_new = z + scale * dz
        small = jnp.all(jnp.abs(dz) <= opts.atol + opts.rtol * jnp.abs(z_new))
        done = small & (scale == 1.0) & jnp.all(jnp.isfinite(z_new))
        return k + 1, z_new, done

    _, z, done = jax.lax.while_loop(cond, body, (0, z0, jnp.array(False)))
    return z, done


def _gmin_stepping(residual: Callable, params: Any, gmin: Any, guess: Array,
                   n_nodes: int, opts: Options, lin: _Linear
                   ) -> tuple[Array, Array]:
    """Newton with the gmin-stepping fallback of `dc` (``args`` is gmin).

    Stage 0 is a direct attempt of `direct_steps` iterations. If it fails,
    stages 1.. restart from `guess` and step gmin down log-uniformly from
    `gmin_start` to `gmin`, each warm-starting from the last; the final stage
    solves at `gmin` itself. All stages run in one `while_loop`, so Newton and
    its linear solver are compiled once, and the stepping is skipped when the
    direct attempt converged (under `vmap`, only batches containing a failure
    pay for it).
    """
    if opts.gmin_steps == 0:
        return _newton(residual, params, gmin, guess, n_nodes, opts, lin)
    gmins = jnp.concatenate([jnp.asarray(gmin, float)[None], jnp.logspace(
        jnp.log10(opts.gmin_start), jnp.log10(gmin), opts.gmin_steps)])
    steps = jnp.array([opts.direct_steps] + [opts.max_steps] * opts.gmin_steps)

    def body(carry):
        k, z, _ = carry
        z, ok = _newton(residual, params, gmins[k], jnp.where(k == 1, guess, z),
                        n_nodes, opts, lin, steps[k])
        return k + 1, z, ok

    def cond(carry):
        k, _, ok = carry
        return (k <= opts.gmin_steps) & ~(ok & (k == 1))

    _, z, ok = jax.lax.while_loop(cond, body, (0, guess, jnp.array(False)))
    return z, ok


@eqx.filter_custom_vjp
def _root(params, guess, args, residual, n_nodes, opts, lin, stepping):
    solver = _gmin_stepping if stepping else _newton
    return solver(residual, params, args, guess, n_nodes, opts, lin)


@_root.def_fwd
def _root_fwd(perturbed, params, guess, args, residual, n_nodes, opts, lin,
              stepping):
    out = _root.fn(params, guess, args, residual, n_nodes, opts, lin, stepping)
    return out, out[0]


@_root.def_bwd
def _root_bwd(z, g, perturbed, params, guess, args, residual, n_nodes, opts, lin,
              stepping):
    g_z = g[0]
    lam = lin.solve(lin.matrix(residual, z, params, args), g_z, transpose=True)
    _, vjp = eqx.filter_vjp(lambda p: residual(z, p, args), params)
    return vjp(-lam)[0]


def solve_root(residual: Callable, params: Any, guess: Array, args: Any = None,
               n_nodes: int | None = None,
               options: Options = Options()) -> tuple[Array, Array]:
    """Solve ``residual(z, params, args) = 0`` for `z`.

    Differentiable with respect to `params` (any pytree) by implicit
    differentiation; `args` are treated as constants. `residual` must be a
    hashable callable (e.g. a module-level function). `n_nodes` limits the
    voltage damping to the first entries of `z` (default: all).

    Returns:
        ``(z, converged)``.
    """
    n = guess.shape[-1] if n_nodes is None else n_nodes
    guess, args = jax.lax.stop_gradient((guess, args))
    return _root(params, guess, args, residual, n, options, _Linear(), False)


def _solve_circuit(residual: Callable, params: Any, guess: Array, args: Any,
                   circuit: Circuit, options: Options, stepping: bool = False
                   ) -> tuple[Array, Array]:
    """`solve_root` for the circuit residuals below, with stamped Jacobians
    (and optionally the gmin-stepping fallback of `dc`)."""
    guess, args = jax.lax.stop_gradient((guess, args))
    lin = _linear(circuit, residual, options)
    return _root(params, guess, args, residual, circuit.n_nodes, options, lin,
                 stepping)


# =============================================================================
# Residuals (module-level so they are hashable static arguments)
# =============================================================================


def _with_gmin(circuit: Circuit, z: Array, gmin: float) -> Array:
    return jnp.zeros_like(z).at[: circuit.n_nodes].set(gmin * z[: circuit.n_nodes])


# Residuals take (z, params, gmin). Everything that can carry a gradient
# (the circuit, the previous state, the time points) is in `params`.


def _dc_residual(z, params, gmin):
    circuit, t = params
    return circuit.f(z, t) + _with_gmin(circuit, z, gmin)


def _be_residual(z, params, gmin):
    circuit, z_prev, t_prev, t = params
    h = t - t_prev
    dq = circuit.q(z) - circuit.q(z_prev)
    return dq / h + circuit.f(z, t) + _with_gmin(circuit, z, gmin)


def _trap_residual(z, params, gmin):
    circuit, z_prev, t_prev, t = params
    h = t - t_prev
    dq = circuit.q(z) - circuit.q(z_prev)
    f_mid = 0.5 * (circuit.f(z, t) + circuit.f(z_prev, t_prev))
    return dq / h + f_mid + _with_gmin(circuit, z, gmin)


_METHODS = {"be": _be_residual, "trap": _trap_residual}


# Their Jacobians, as stamped values on the circuit's sparsity pattern.


def _add_gmin(J, circuit, st, gmin):
    return J.at[st.diagonal[: circuit.n_nodes]].add(gmin)


def _dc_jacobian(z, params, gmin, st):
    circuit, t = params
    J = sparse.jacobian(circuit, z, t, structure=st)
    return _add_gmin(J, circuit, st, gmin)


def _be_jacobian(z, params, gmin, st):
    circuit, _, t_prev, t = params
    J = sparse.jacobian(circuit, z, t, f=1.0, q=1.0 / (t - t_prev), structure=st)
    return _add_gmin(J, circuit, st, gmin)


def _trap_jacobian(z, params, gmin, st):
    circuit, _, t_prev, t = params
    J = sparse.jacobian(circuit, z, t, f=0.5, q=1.0 / (t - t_prev), structure=st)
    return _add_gmin(J, circuit, st, gmin)


_JACOBIANS = {_dc_residual: _dc_jacobian, _be_residual: _be_jacobian,
              _trap_residual: _trap_jacobian}


def _uses_sparse(circuit: Circuit, opts: Options) -> bool:
    if opts.solver == "auto":
        return circuit.size >= opts.sparse_threshold
    return opts.solver == "sparse"


_JACFWD_BELOW = 16
"""Dense systems with fewer unknowns use ``jax.jacfwd`` (one batched JVP per
unknown), which beats stamping's per-group JVPs and scatter at that size."""


def _linear(circuit: Circuit, residual: Callable, opts: Options) -> _Linear:
    """Stamped Jacobian of a circuit residual, solved per `Options.solver`."""
    if _uses_sparse(circuit, opts):
        return _Linear(_JACOBIANS[residual], sparse.structure(circuit), True)
    if circuit.size < _JACFWD_BELOW:
        return _Linear()
    return _Linear(_JACOBIANS[residual], sparse.structure(circuit), False)


# =============================================================================
# Analyses
# =============================================================================


@eqx.filter_jit
def dc(circuit: Circuit, t: float | Array = 0.0, guess: Array | None = None,
       options: Options = Options()) -> Solution:
    """DC operating point: solve ``f(z, t) = 0`` (capacitors open, inductors
    shorted, sources evaluated at time `t`).

    Runs Newton from `guess`, falling back to gmin stepping if that fails
    (see `Options`). Gradients are exact either way: they only depend on the
    final root, via implicit differentiation.

    Args:
        circuit: The circuit.
        t: Time at which to evaluate sources.
        guess: Initial Newton iterate (default zeros). Pass a nearby solution
            for continuation, e.g. in a sweep.
        options: Solver settings.
    """
    t = jnp.asarray(t, dtype=float)
    guess = circuit.zeros() if guess is None else guess
    z, ok = _solve_circuit(_dc_residual, (circuit, t), guess, options.gmin,
                           circuit, options, stepping=True)
    return Solution(t=t, z=z, converged=ok, layout=circuit.layout)


@eqx.filter_jit
def transient(
    circuit: Circuit,
    ts: Array,
    ic: Literal["dc", "zero"] | Array | Solution = "dc",
    method: Literal["be", "trap"] = "be",
    options: Options = Options(),
) -> Solution:
    """Transient analysis on the time grid `ts` (fixed, possibly non-uniform).

    Args:
        circuit: The circuit.
        ts: ``(T,)`` increasing time points; ``ts[0]`` is the initial time.
        ic: Initial condition: ``"dc"`` (operating point at ``ts[0]``, the
            SPICE default), ``"zero"``, a state vector, or a `Solution`.
        method: ``"be"`` (backward Euler, L-stable, default) or ``"trap"``
            (trapezoidal, 2nd order).
        options: Newton settings for every step.

    Returns:
        `Solution` with ``z`` of shape ``(T, state_size)``.
    """
    ts = jnp.asarray(ts, dtype=float)
    if isinstance(ic, Solution):
        z0 = ic.z
    elif isinstance(ic, str):
        if ic == "dc":
            z0 = dc(circuit, ts[0], options=options).z
        elif ic == "zero":
            z0 = circuit.zeros()
        else:
            raise ValueError(f"ic must be 'dc', 'zero' or an array, got {ic!r}")
    else:
        z0 = jnp.asarray(ic, dtype=float)
    residual = _METHODS[method]

    def step(z_prev, t_pair):
        t_prev, t = t_pair
        params = (circuit, z_prev, t_prev, t)
        z, ok = _solve_circuit(residual, params, z_prev, options.gmin, circuit,
                               options)
        return z, (z, ok)

    _, (zs, oks) = jax.lax.scan(step, z0, jnp.stack([ts[:-1], ts[1:]], axis=1))
    z = jnp.concatenate([z0[None], zs])
    converged = jnp.concatenate([jnp.ones(1, bool), oks])
    return Solution(t=ts, z=z, converged=converged, layout=circuit.layout)


def linearize(circuit: Circuit, z: Array, t: float | Array = 0.0
              ) -> tuple[Array, Array]:
    """Small-signal matrices ``G = df/dz`` and ``C = dq/dz`` at state `z`, as
    dense ``(S, S)`` arrays (`voltax.sparse.jacobian` gives sparse values).
    """
    st = sparse.structure(circuit)
    G = sparse.jacobian(circuit, z, t, f=1.0, q=0, structure=st)
    C = sparse.jacobian(circuit, z, t, f=0, q=1.0, structure=st)
    return st.to_dense(G), st.to_dense(C)


@eqx.filter_jit
def ac(circuit: Circuit, freqs: Array, op: Solution | None = None,
       options: Options = Options()) -> ACSolution:
    """Small-signal AC analysis around the DC operating point.

    Solves ``(G + j 2 pi f C) z = -b`` for every frequency, where ``b`` is the
    AC stimulus set by the sources' ``ac`` / ``ac_phase``.

    Args:
        circuit: The circuit.
        freqs: ``(F,)`` frequencies in Hz.
        op: Operating point (computed with `dc` if None).
        options: Settings for the DC solve; `Options.solver` also picks the
            linear solver of the AC systems.
    """
    op = dc(circuit, options=options) if op is None else op
    st = sparse.structure(circuit)
    G = sparse.jacobian(circuit, op.z, op.t, f=1.0, q=0, structure=st)
    C = sparse.jacobian(circuit, op.z, op.t, f=0, q=1.0, structure=st)
    b = circuit.ac_stimulus()
    use_sparse = _uses_sparse(circuit, options)

    def solve(f):
        A = G + 2j * jnp.pi * f * C
        if use_sparse:
            return sparse.spsolve(st, A, -b)
        return jnp.linalg.solve(st.to_dense(A), -b)

    freqs = jnp.asarray(freqs, dtype=float)
    return ACSolution(freqs=freqs, z=jax.vmap(solve)(freqs), op=op,
                      layout=circuit.layout)


# =============================================================================
# Sweeps and time grids
# =============================================================================


@eqx.filter_jit
def dc_sweep(
    circuit: Circuit,
    values: Array,
    apply: str | Callable[[Circuit, Array], Circuit],
    t: float | Array = 0.0,
    guess: Array | None = None,
    options: Options = Options(),
) -> Solution:
    """DC sweep with continuation (SPICE ``.dc``): each point starts Newton
    from the previous solution.

    Unlike ``jax.vmap(dc)``, the sweep follows one solution branch, so
    circuits with several stable states (Schmitt triggers, latches, SRAM
    cells) show their hysteresis: sweep up and down to see both branches.

    Args:
        circuit: The circuit.
        values: ``(N,)`` sweep values, in sweep order.
        apply: How a value changes the circuit: a source name (sets its DC
            value), or ``fn(circuit, value) -> circuit``.
        t: Time at which to evaluate other sources.
        guess: Starting state for the first point (default zeros).
        options: Solver settings.

    Returns:
        `Solution` with ``z`` of shape ``(N, state_size)``; its ``t`` holds
        the sweep values, so ``sol.t`` is the x-axis.
    """
    if isinstance(apply, str):
        device = apply
        apply = lambda c, v: c.set(device, value=v)  # noqa: E731
    values = jnp.asarray(values, dtype=float)
    z0 = circuit.zeros() if guess is None else guess

    def step(z, value):
        sol = dc(apply(circuit, value), t, guess=z, options=options)
        return sol.z, (sol.z, sol.converged)

    _, (zs, oks) = jax.lax.scan(step, z0, values)
    return Solution(t=values, z=zs, converged=oks, layout=circuit.layout)


def time_grid(circuit: Circuit, t_stop: float, dt_max: float,
              dt_min: float | None = None, ramp_points: int = 10,
              growth: float = 1.2) -> Array:
    """A transient time grid refined at the sources' corners.

    Collects the breakpoints of every source waveform (pulse and step edges,
    PWL corners), puts at least `ramp_points` steps across each ramp, starts
    each interval with a small step and grows it geometrically by `growth`
    up to `dt_max`, so fast edges and the response right after them are
    resolved without a uniformly fine grid.

    Needs concrete (non-traced) source parameters: call it outside `jit` and
    pass the grid in.

    Args:
        circuit: The circuit whose sources set the breakpoints.
        t_stop: End time (the grid starts at 0).
        dt_max: Largest step.
        dt_min: First step after each breakpoint (default: the shortest
            interval between breakpoints divided by `ramp_points`).
        ramp_points: Minimum number of steps between two breakpoints.
        growth: Ratio between consecutive steps after a breakpoint. The
            default 1.2 keeps delay measurements within ~0.1%; larger values
            give fewer points (2.0 is fine for a single RC, not for gate
            chains whose response lasts many steps after the edge).
    """
    from .signals import Signal

    signals = [leaf for el in circuit.elements.values()
               for leaf in jax.tree.leaves(el, is_leaf=lambda x: isinstance(x, Signal))
               if isinstance(leaf, Signal)]
    corners = np.unique(np.concatenate(
        [np.array([0.0, t_stop])] + [s.breakpoints(t_stop) for s in signals]))
    gaps = np.diff(corners)
    gaps = gaps[gaps > 0]
    if dt_min is None:
        dt_min = min(dt_max, gaps.min() / ramp_points) if gaps.size else dt_max
    times = [0.0]
    for a, b in zip(corners[:-1], corners[1:]):
        h = min(dt_min, (b - a) / ramp_points)
        t = a
        while b - t > 1e-9 * h:
            t = min(t + h, b)
            times.append(t)
            h = min(h * growth, dt_max, max((b - a) / ramp_points, dt_min))
    return jnp.asarray(np.unique(times))
