"""Sparse Jacobians and a supernodal sparse LU, in pure JAX.

Circuit Jacobians are very sparse: entry ``(i, j)`` can only be nonzero when
unknowns ``i`` and ``j`` belong to a common device (its terminals plus its
internal unknowns). This module exploits that in two independent steps.

**Stamped Jacobians.** Devices in an element group are independent (see
`Element.local`), so the ``(T + K) x (T + K)`` Jacobian of *every* device in a
group comes from ``T + K`` forward-mode JVPs of that group alone, seeding one
terminal/internal slot of all devices at once. The local blocks are then
scatter-added ("stamped", as in SPICE) into the values of a static sparsity
pattern. That costs a handful of JVPs per group instead of one JVP per
unknown (``jax.jacfwd``). Graph coloring cannot do this well for circuits:
a supply node touching every transistor makes its row dense, which forces
as many colors as unknowns.

**Supernodal LU.** The linear solver is a static-structure sparse LU whose
symbolic analysis runs in numpy at trace time:

1. *Pivot groups.* Unknowns whose diagonal can be zero (voltage-source and
   inductor currents, op-amp outputs, ...) are glued to terminal nodes of
   their device, so they are always eliminated together with a partner in
   one dense block. For two-terminal devices with one internal unknown, a
   bipartite matching picks one partner node; other devices keep all their
   terminals.
2. *Ordering.* Nested dissection on the quotient graph of groups (separators
   from breadth-first level structures), with very high-degree nodes such as
   ``vdd`` moved to a final separator. Leaves and separators become dense
   *supernodes*; the separator tree is the elimination tree.
3. *Schedule.* Supernodes at the same tree height are independent, so each
   level is processed as one or two padded batches.

The numeric phase is a block ``L D U`` factorization: each supernode's
pivot block is inverted (batched LU with *partial pivoting inside the
block*), and its Schur complement is scatter-added into the ancestors'
panels. Pivoting is therefore dynamic within supernodes and static
across them (the usual "static pivoting" compromise of GPU sparse solvers).
Solves (also transposed, for adjoints, and complex, for AC) are batched
matrix-vector products. Everything is jit-, vmap- and GPU-compatible and
never calls back to the host.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from jax import Array

from .element import Element

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Circuit

LEAF_SIZE = int(os.environ.get("VOLTAX_SPARSE_LEAF", 16))
SPLIT_SAVING = 1e6
"""Padded flops a level must save to be factored as two batches, not one."""
"""Largest nested-dissection leaf (unknowns) kept as one dense supernode."""

# =============================================================================
# Sparsity pattern and stamped Jacobians
# =============================================================================


@dataclass(frozen=True, eq=False)
class Structure:
    """Static sparsity structure of a circuit's Jacobians ``df/dz``, ``dq/dz``.

    Values live in a flat ``data`` array aligned with the CSR pattern
    ``(indptr, indices)`` (sorted column indices, diagonal always present).

    Attributes:
        size: Number of unknowns ``S``.
        n_nodes: Number of node voltages (the first entries of ``z``).
        indptr, indices: CSR pattern.
        stamps: Per element group, ``(group, positions, local)``: where each
            local Jacobian entry lands in ``data`` (``nnz`` = dropped, for
            ground). Local groups have ``(N, m, m)`` positions, coupled
            (``local = False``) groups ``(N m, N m)``.
        devices: Per element group, ``(group, unknowns, n_terminals, local)``
            with ``unknowns`` the ``(N, T + K)`` state indices of every
            device's terminals and internals (``-1`` = ground).
    """

    size: int
    n_nodes: int
    indptr: np.ndarray
    indices: np.ndarray
    stamps: tuple[tuple[str, np.ndarray, bool], ...]
    devices: tuple[tuple[str, np.ndarray, int, bool], ...]

    @property
    def nnz(self) -> int:
        return len(self.indices)

    @cached_property
    def rows(self) -> np.ndarray:
        """Row index of every pattern entry."""
        return np.repeat(np.arange(self.size), np.diff(self.indptr))

    @cached_property
    def diagonal(self) -> np.ndarray:
        """Positions of the diagonal entries in ``data``."""
        return _positions(self, np.arange(self.size), np.arange(self.size))

    @cached_property
    def plan(self) -> "Plan":
        """Symbolic factorization (computed on first use, then cached)."""
        return _make_plan(self)

    def to_dense(self, data: Array) -> Array:
        """``(S, S)`` dense matrix with values `data` on the pattern."""
        flat = self.rows * self.size + self.indices
        out = jnp.zeros(self.size * self.size, data.dtype).at[flat].set(data)
        return out.reshape(self.size, self.size)

    def to_scipy(self, data: Any) -> sp.csr_matrix:
        """`scipy.sparse.csr_matrix` with values `data` (for inspection)."""
        return sp.csr_matrix((np.asarray(data), self.indices, self.indptr),
                             shape=(self.size, self.size))

    def matvec(self, data: Array, x: Array) -> Array:
        """``A @ x`` for ``A`` with values `data`."""
        return jax.ops.segment_sum(data * x[self.indices], self.rows,
                                   num_segments=self.size)


def _positions(st: Structure, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Positions of ``(rows, cols)`` in the pattern; ``nnz`` where either is
    ground (-1). Raises if an entry is missing."""
    rows, cols = np.broadcast_arrays(np.asarray(rows), np.asarray(cols))
    keys = st.rows.astype(np.int64) * st.size + st.indices
    want = rows.astype(np.int64) * st.size + cols
    pos = np.searchsorted(keys, want)
    ok = (rows >= 0) & (cols >= 0)
    hit = np.where(ok, keys[np.minimum(pos, len(keys) - 1)] == want, True)
    if not hit.all():  # pragma: no cover - internal invariant
        raise AssertionError("entry outside the sparsity pattern")
    return np.where(ok, pos, len(keys)).astype(np.int32)


def structure(circuit: "Circuit") -> Structure:
    """The (cached) `Structure` of `circuit`'s topology."""
    key = (circuit.layout,
           tuple((name, el.nodes, bool(type(el).local))
                 for name, el in sorted(circuit.elements.items())))
    return _structure(key)


_structure_of = structure  # `jacobian` has an argument named `structure`


@functools.lru_cache(maxsize=64)
def _structure(key: tuple) -> Structure:
    layout, groups = key
    S, nv = layout.size, layout.n_nodes
    offsets = {g: (n, k, off) for g, n, k, off in layout.groups}
    devices, rows, cols = [], [np.arange(S)], [np.arange(S)]
    for name, nodes, local in groups:
        n, k, off = offsets[name]
        if n == 0:
            continue
        idx = np.asarray(nodes, dtype=np.int64).reshape(n, -1)
        allv = np.concatenate([idx, np.arange(off, off + n * k).reshape(n, k)], 1)
        devices.append((name, allv, idx.shape[1], local))
        blocks = allv if local else allv.reshape(1, -1)
        m = blocks.shape[1]
        r = np.repeat(blocks[:, :, None], m, 2).ravel()
        c = np.repeat(blocks[:, None, :], m, 1).ravel()
        keep = (r >= 0) & (c >= 0)
        rows.append(r[keep])
        cols.append(c[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    pat = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(S, S))
    pat.sum_duplicates()
    pat.sort_indices()
    st = Structure(S, nv, pat.indptr.astype(np.int64), pat.indices.astype(np.int64),
                   (), tuple(devices))
    stamps = []
    for name, allv, _, local in devices:
        blocks = allv if local else allv.reshape(1, -1)
        pos = _positions(st, blocks[:, :, None], blocks[:, None, :])
        stamps.append((name, pos if local else pos[0], local))
    object.__setattr__(st, "stamps", tuple(stamps))
    return st


def _is_zero(c: Any) -> bool:
    return isinstance(c, (int, float)) and c == 0


def _group_fn(el: Any, n: int, T: int, k: int, t: Array, cf: Any, cq: Any):
    """``u (N, T+K) -> cf * currents + cq * charges`` of element group `el`."""

    def part(A, B, dtype):
        A = jnp.zeros((n, T), dtype) if A is None else jnp.broadcast_to(A, (n, T))
        B = jnp.zeros((n, k), dtype) if (B is None or k == 0) else B
        return jnp.concatenate([A, B], axis=1)

    def fn(u: Array) -> Array:
        v, x = u[:, :T], u[:, T:]
        out = jnp.zeros((n, T + k), u.dtype)
        if not _is_zero(cf):
            out = out + cf * part(*el.currents(v, x, t), u.dtype)
        if not _is_zero(cq):
            out = out + cq * part(*el.charges(v, x), u.dtype)
        return out

    return fn


def jacobian(circuit: "Circuit", z: Array, t: Array | float = 0.0, f: Any = 1.0,
             q: Any = 0.0, structure: Structure | None = None) -> Array:
    """Stamped values of ``f * df/dz + q * dq/dz`` at state `z` and time `t`,
    aligned with the pattern of `structure` (default ``structure(circuit)``).

    The coefficients may be traced; a literal ``0`` skips that part. Exact
    (forward-mode AD) and costs ``T + K`` JVPs per element group (all
    unknowns of a group with ``local = False``).
    """
    st = _structure_of(circuit) if structure is None else structure
    t = jnp.asarray(t, dtype=float)
    z_ext = jnp.concatenate([z, jnp.zeros(1, z.dtype)])  # [-1] = ground
    data = jnp.zeros(st.nnz, z.dtype)
    for (name, allv, T, local), (_, pos, _) in zip(st.devices, st.stamps):
        el = circuit.elements[name]
        n, m = allv.shape
        cq = 0 if type(el).charges is Element.charges else q  # no reactive part
        if _is_zero(f) and _is_zero(cq):
            continue
        fn = _group_fn(el, n, T, m - T, t, f, cq)
        u0 = z_ext[np.where(allv < 0, z_ext.shape[0] - 1, allv)]
        if local:
            seeds = jnp.broadcast_to(jnp.eye(m, dtype=z.dtype)[:, None, :], (m, n, m))
            d = jax.vmap(lambda s: jax.jvp(fn, (u0,), (s,))[1])(seeds)  # (in, N, out)
            vals = jnp.transpose(d, (1, 2, 0))
        else:
            flat = lambda u: fn(u.reshape(n, m)).ravel()  # noqa: E731
            vals = jax.jacfwd(flat)(u0.ravel())
        data = data.at[pos.ravel()].add(vals.ravel(), mode="drop")
    return data


# =============================================================================
# Symbolic analysis: pivot groups, nested dissection, supernodes
# =============================================================================


def _pivot_groups(st: Structure) -> np.ndarray:
    """Group id of every unknown (see module docstring, step 1)."""
    S = st.size
    parent = np.arange(S)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs_internal, pairs_cands, full = [], [], []
    for _, allv, T, local in st.devices:
        k = allv.shape[1] - T
        if k == 0:
            continue
        if not local:  # coupled devices: one block with all their unknowns
            full.append([int(v) for v in np.unique(allv) if v >= 0])
            continue
        for dev in allv:
            terms = [int(v) for v in dev[:T] if v >= 0]
            internals = [int(v) for v in dev[T:]]
            if T == 2 and k == 1 and terms:
                pairs_internal.append(internals[0])
                pairs_cands.append(terms)
            else:
                full.append(terms + internals)
    if pairs_internal:
        # bipartite matching: internal unknown -> a distinct terminal node
        r = np.repeat(np.arange(len(pairs_internal)), [len(c) for c in pairs_cands])
        c = np.concatenate(pairs_cands)
        g = sp.csr_matrix((np.ones(len(r)), (r, c)),
                          shape=(len(pairs_internal), st.n_nodes))
        match = csg.maximum_bipartite_matching(g, perm_type="column")
        for i, (x, cands) in enumerate(zip(pairs_internal, pairs_cands)):
            if match[i] >= 0:
                union(int(match[i]), x)
            else:
                full.append(cands + [x])
    for members in full:
        for v in members[1:]:
            union(members[0], v)
    roots = np.array([find(a) for a in range(S)])
    return np.unique(roots, return_inverse=True)[1]


def _nested_dissection(adj: sp.csr_matrix, w: np.ndarray, leaf: int,
                       last: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Supernodes (vertex sets of `adj`) in postorder, and their parents.

    `adj` is a symmetric adjacency (no diagonal) with vertex weights `w`;
    vertices in `last` form the final (root) supernode.
    """
    nodes: list[tuple[np.ndarray, int]] = []
    top = -1
    if last.any():
        nodes.append((np.flatnonzero(last), -1))
        top = 0
    stack = [(np.flatnonzero(~last), top)]
    while stack:
        verts, par = stack.pop()
        if len(verts) == 0:
            continue
        sub = adj[verts][:, verts]
        ncomp, lab = csg.connected_components(sub, directed=False)
        if ncomp > 1:
            stack += [(verts[lab == c], par) for c in range(ncomp)]
            continue
        if w[verts].sum() <= leaf:
            nodes.append((verts, par))
            continue
        start = 0  # pseudo-peripheral vertex: repeated farthest-vertex BFS
        for _ in range(4):
            d = csg.shortest_path(sub, unweighted=True, indices=start)
            far = int(np.argmax(d))
            if d[far] <= d[start] or far == start:
                break
            start = far
        d = csg.shortest_path(sub, unweighted=True, indices=start).astype(int)
        depth = d.max()
        if depth < 2:  # (near-)clique: no useful separator
            nodes.append((verts, par))
            continue
        lw = np.bincount(d, weights=w[verts], minlength=depth + 1)
        cum = np.cumsum(lw)
        total = cum[-1]
        cands = [lv for lv in range(1, depth)
                 if cum[lv - 1] >= total / 3 and total - cum[lv] >= total / 3]
        if not cands:
            mid = int(np.searchsorted(cum, total / 2))
            cands = [min(max(mid, 1), depth - 1)]
        lv = min(cands, key=lambda c: lw[c])
        touches = (sub @ (d == lv + 1).astype(float)) > 0
        sep = (d == lv) & touches
        me = len(nodes)
        nodes.append((verts[sep], par))
        stack.append((verts[(d < lv) | ((d == lv) & ~touches)], me))
        stack.append((verts[d > lv], me))
    # postorder (children before parents)
    children: list[list[int]] = [[] for _ in nodes]
    roots = []
    for i, (_, p) in enumerate(nodes):
        (children[p] if p >= 0 else roots).append(i)
    order: list[int] = []
    todo = [(r, False) for r in reversed(roots)]
    while todo:
        i, done = todo.pop()
        if done:
            order.append(i)
        else:
            todo.append((i, True))
            todo += [(c, False) for c in reversed(children[i])]
    new = {old: i for i, old in enumerate(order)}
    sets = [nodes[old][0] for old in order]
    parents = np.array([new[nodes[old][1]] if nodes[old][1] >= 0 else -1
                        for old in order], dtype=np.int64)
    return sets, parents


@dataclass(frozen=True, eq=False)
class Bucket:
    """Supernodes factored together: ``m`` of them, padded to ``kp`` pivots
    and ``rp`` border unknowns; panels stored at ``work[off:]`` as
    ``F11 (m, kp, kp) | F12 (m, kp, rp) | F21 (m, rp, kp)``."""

    m: int
    kp: int
    rp: int
    off: int
    piv: np.ndarray  # (m, kp) pivot unknowns, S = padding
    border: np.ndarray  # (m, rp) border unknowns, S = padding
    valid: np.ndarray  # (m, kp) bool, False on padded pivots
    schur: np.ndarray  # (m, rp, rp) positions in work (size = dropped)


@dataclass(frozen=True, eq=False)
class Plan:
    """Static schedule of the supernodal factorization of a `Structure`."""

    size: int
    """Number of unknowns."""
    work: int
    """Length of the panel storage."""
    buckets: tuple[Bucket, ...]
    a_pos: np.ndarray
    """Where each pattern entry lands in the panel storage."""
    stats: dict = field(default_factory=dict)
    """Supernode count, levels, factor entries, flops (unpadded/padded)."""


def _make_plan(st: Structure, leaf: int | None = None) -> Plan:
    leaf = LEAF_SIZE if leaf is None else leaf
    S = st.size
    gid = _pivot_groups(st)
    G = int(gid.max()) + 1
    E = sp.csr_matrix((np.ones(S), (np.arange(S), gid)), shape=(S, G))
    P = sp.csr_matrix((np.ones(st.nnz), st.indices, st.indptr), shape=(S, S))
    Ps = (P + P.T).tocsr()
    Q = (E.T @ Ps @ E).tocsr()
    Q.setdiag(0)
    Q.eliminate_zeros()
    w = np.bincount(gid, minlength=G).astype(float)
    deg = np.diff(Q.indptr)
    if S <= 2 * leaf:
        last = np.ones(G, bool)  # small: a single dense block
    else:
        last = deg > max(32.0, 8.0 * np.sqrt(G))
    qsets, parents = _nested_dissection(Q, w, leaf, last)
    order = np.argsort(gid, kind="stable")
    starts = np.searchsorted(gid[order], np.arange(G + 1))
    sns = [np.sort(np.concatenate([order[starts[g]:starts[g + 1]] for g in qs]))
           for qs in qsets]
    nsn = len(sns)
    sn_of = np.empty(S, np.int64)
    pos_in = np.empty(S, np.int64)
    for i, s in enumerate(sns):
        sn_of[s] = i
        pos_in[s] = np.arange(len(s))
    children: list[list[int]] = [[] for _ in range(nsn)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    borders: list[np.ndarray] = []
    level = np.zeros(nsn, np.int64)
    for i, s in enumerate(sns):
        adj = Ps.indices[np.concatenate([np.arange(Ps.indptr[v], Ps.indptr[v + 1])
                                         for v in s])]
        cand = np.unique(np.concatenate([adj] + [borders[c] for c in children[i]]))
        borders.append(cand[sn_of[cand] > i])
        if children[i]:
            level[i] = 1 + max(level[c] for c in children[i])
    k = np.array([len(s) for s in sns])
    r = np.array([len(b) for b in borders])

    # --- buckets: one padded batch per level, split in two (by size) when
    # that saves more than SPLIT_SAVING padded flops. Every bucket adds
    # kernels (and a while loop) to compile, ~0.3-0.5 s on CPU.
    def cost(sel: np.ndarray) -> float:
        if len(sel) == 0:
            return 0.0
        K, R = k[sel].max(), r[sel].max()
        return len(sel) * (2.0 * K**3 + 2.0 * K * K * R + 2.0 * K * R * R + 64.0)

    classes: list[np.ndarray] = []
    for lv in range(int(level.max()) + 1):
        sel = np.flatnonzero(level == lv)
        sel = sel[np.argsort(k[sel] ** 3 + k[sel] ** 2 * r[sel] + k[sel] * r[sel] ** 2,
                             kind="stable")]
        cut = min(range(len(sel)), key=lambda c: cost(sel[:c]) + cost(sel[c:]))
        if cost(sel) - cost(sel[:cut]) - cost(sel[cut:]) < SPLIT_SAVING:
            cut = 0
        classes += [part for part in (sel[:cut], sel[cut:]) if len(part)]
    off = 0
    base11 = np.zeros(nsn, np.int64)
    base12 = np.zeros(nsn, np.int64)
    base21 = np.zeros(nsn, np.int64)
    KP = np.zeros(nsn, np.int64)
    RP = np.zeros(nsn, np.int64)
    layout = []
    for sel in classes:
        m, kp, rp = len(sel), int(k[sel].max()), int(r[sel].max())
        j = np.arange(m)
        base11[sel] = off + j * kp * kp
        base12[sel] = off + m * kp * kp + j * kp * rp
        base21[sel] = off + m * kp * kp + m * kp * rp + j * rp * kp
        KP[sel], RP[sel] = kp, rp
        layout.append((sel, m, kp, rp, off))
        off += m * (kp * kp + 2 * kp * rp)
    work = off
    # border position lookup: key (supernode, unknown) -> index in border
    bsn = np.repeat(np.arange(nsn), r)
    bvar = np.concatenate(borders) if nsn else np.zeros(0, np.int64)
    bkeys = bsn * S + bvar
    border_order = np.argsort(bkeys)
    bkeys = bkeys[border_order]
    bidx = np.concatenate([np.arange(x) for x in r])[border_order]

    def bpos(o: np.ndarray, v: np.ndarray) -> np.ndarray:
        if len(bidx) == 0:  # a single supernode: no borders
            return np.zeros(np.shape(o), np.int64)
        q = np.searchsorted(bkeys, o * S + v)
        return bidx[np.minimum(q, len(bidx) - 1)]

    def position(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        su, sv = sn_of[u], sn_of[v]
        o = np.minimum(su, sv)
        same = base11[o] + pos_in[u] * KP[o] + pos_in[v]
        upper = base12[o] + pos_in[u] * RP[o] + bpos(o, v)
        lower = base21[o] + bpos(o, u) * KP[o] + pos_in[v]
        return np.where(su == sv, same, np.where(su < sv, upper, lower))

    a_pos = position(st.rows, st.indices).astype(np.int32)
    buckets = []
    for sel, m, kp, rp, boff in layout:
        piv = np.full((m, kp), S, np.int64)
        brd = np.full((m, rp), S, np.int64)
        schur = np.full((m, rp, rp), work, np.int64)
        for j, i in enumerate(sel):
            piv[j, : k[i]] = sns[i]
            brd[j, : r[i]] = borders[i]
            if r[i]:
                b = borders[i]
                schur[j, : r[i], : r[i]] = position(b[:, None], b[None, :])
        valid = np.arange(kp)[None, :] < k[sel][:, None]
        buckets.append(Bucket(m, kp, rp, boff, piv.astype(np.int32),
                              brd.astype(np.int32), valid, schur.astype(np.int32)))
    flops = float(np.sum(2.0 * k**3 + 2.0 * k * k * r + 2.0 * k * r * r))
    padded = float(sum(m * (2.0 * kp**3 + 2.0 * kp * kp * rp + 2.0 * kp * rp * rp)
                       for _, m, kp, rp, _ in layout))
    stats = dict(supernodes=nsn, levels=int(level.max()) + 1, buckets=len(buckets),
                 factor_entries=int(np.sum(k * k + 2 * k * r)), work=work,
                 flops=flops, padded_flops=padded, largest_supernode=int(k.max()),
                 pivot_groups=G)
    return Plan(S, work, tuple(buckets), a_pos, stats)


# =============================================================================
# Numeric factorization and solves
# =============================================================================


def _inverse(A: Array) -> Array:
    """Batched inverse of ``(m, k, k)`` pivot blocks (LU with partial
    pivoting, i.e. pivoting inside the supernode)."""
    return 1.0 / A if A.shape[-1] == 1 else jnp.linalg.inv(A)


def factor(plan: Plan, data: Array) -> tuple:
    """Numeric factorization of the matrix with pattern values `data`.

    Returns per-bucket ``(W, Ub, Lb)``: pivot-block inverses ``F11^-1``,
    ``F11^-1 F12`` and ``F21 F11^-1``.
    """
    work = jnp.zeros(plan.work, data.dtype).at[plan.a_pos].add(data)
    out = []
    for bk in plan.buckets:
        m, kp, rp, o = bk.m, bk.kp, bk.rp, bk.off
        F11 = work[o: o + m * kp * kp].reshape(m, kp, kp)
        F11 = F11 + jnp.where(bk.valid, 0.0, 1.0)[:, :, None] * jnp.eye(kp)
        W = _inverse(F11)
        if rp:
            o2 = o + m * kp * kp
            F12 = work[o2: o2 + m * kp * rp].reshape(m, kp, rp)
            o3 = o2 + m * kp * rp
            F21 = work[o3: o3 + m * rp * kp].reshape(m, rp, kp)
            Ub, Lb = W @ F12, F21 @ W
            work = work.at[bk.schur].add(-(F21 @ Ub), mode="drop")
        else:
            Ub = Lb = None
        out.append((W, Ub, Lb))
    return tuple(out)


def solve(plan: Plan, factors: tuple, b: Array, transpose: bool = False) -> Array:
    """Solve ``A x = b`` (or ``A^T x = b``) with `factor`'s output."""
    x = b

    def get(idx):
        return x.at[idx].get(mode="fill", fill_value=0)

    pairs = list(zip(plan.buckets, factors))
    for bk, (W, Ub, Lb) in pairs:  # L y = b and D z = y (or U^T, D^T)
        y = get(bk.piv)
        if bk.rp:
            off = Ub if transpose else jnp.swapaxes(Lb, 1, 2)  # (m, kp, rp)
            x = x.at[bk.border].add(-jnp.einsum("mkr,mk->mr", off, y), mode="drop")
        Wop = jnp.swapaxes(W, 1, 2) if transpose else W
        x = x.at[bk.piv].set(jnp.einsum("mjk,mk->mj", Wop, y), mode="drop")
    for bk, (W, Ub, Lb) in reversed(pairs):  # U x = z (or L^T)
        if bk.rp:
            off = jnp.swapaxes(Lb, 1, 2) if transpose else Ub  # (m, kp, rp)
            x = x.at[bk.piv].add(-jnp.einsum("mkr,mr->mk", off, get(bk.border)),
                                 mode="drop")
    return x


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def spsolve(st: Structure, data: Array, b: Array) -> Array:
    """``x = A^-1 b`` for ``A`` with values `data` on ``st``'s pattern.

    Differentiable in `data` and `b` (by the usual adjoint formulas; the
    factorization itself is never differentiated).
    """
    return solve(st.plan, factor(st.plan, data), b)


def _spsolve_fwd(st, data, b):
    factors = factor(st.plan, data)
    x = solve(st.plan, factors, b)
    return x, (factors, x)


def _spsolve_bwd(st, res, g):
    factors, x = res
    gb = solve(st.plan, factors, g, transpose=True)
    gdata = -gb[st.rows] * x[st.indices]
    return gdata, gb


spsolve.defvjp(_spsolve_fwd, _spsolve_bwd)
