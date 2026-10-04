"""Stamped Jacobians and the sparse solver: exactness, agreement with the dense
path, gradients, and jit/vmap behaviour.

The whole suite can also be run with the sparse solver forced on (and tiny
supernodes, to stress static pivoting) with
``VOLTAX_SOLVER=sparse VOLTAX_SPARSE_LEAF=2 pytest``.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_analysis import Memristor, fd_grad, rc_ladder
from test_builder_netlist import NETLIST

import voltax as vx
from voltax import sparse
from voltax.library import cmos

DENSE = vx.Options(solver="dense")
SPARSE = vx.Options(solver="sparse")

# ----------------------------------------------------------------- circuits


def adder(bits=2):
    b = vx.CircuitBuilder()
    b.vsource("vdd", "0", 1.2)
    a = [f"a{i}" for i in range(bits)]
    c = [f"b{i}" for i in range(bits)]
    for k, n in enumerate(a + c):
        b.vsource(n, "0", 1.2 * (k % 2))
    cmos.ripple_adder(b, a, c, "0", [f"s{i}" for i in range(bits)], "co", "vdd")
    return b.build()


def controlled():
    b = vx.CircuitBuilder()
    b.vsource("c", "0", 0.5, ac=1.0)
    b.vcvs("e", "0", "c", "0", 3.0)
    b.resistor("e", "0", 1e3)
    b.vccs("0", "g", "c", "0", 2e-3)
    b.resistor("g", "0", 1e3)
    b.vsource("x", "0", 1.0)
    b.resistor("x", "xs", 1e3)
    b.cccs("0", "f", "xs", "y", 2.0)
    b.resistor("f", "0", 1e3)
    b.ccvs("h", "0", "y", "0", 500.0)
    b.resistor("h", "0", 1e3)
    b.resistor("c", "m", 1e3)  # inverting amplifier with an ideal op-amp
    b.resistor("m", "o1", 4.7e3)
    b.ideal_opamp("0", "m", "o1")
    b.resistor("o1", "p2", 1e3)  # behavioral op-amp follower with an RC load
    b.opamp("p2", "o2", "o2")
    b.capacitor("p2", "0", 1e-9)
    return b.build()


def magnetics():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1e4), ac=1.0)
    b.resistor("in", "p", 10.0)
    b.transformer("p", "0", "s", "0", 1e-3, 4e-3, k=0.99)
    b.resistor("s", "0", 1e3)
    for k in range(4):  # an LC ladder: a chain of inductors sharing nodes
        b.inductor(f"s{k}" if k else "s", f"s{k + 1}", 1e-4, rs=0.1)
        b.capacitor(f"s{k + 1}", "0", 1e-8)
    b.resistor("s4", "0", 50.0)
    b.diode("s4", "d")
    b.resistor("d", "0", 1e3)
    b.switch("d", "0", "s", "0", ron=10.0)
    b.bjt("q", "d", "0")
    b.resistor("in", "q", 1e4)
    return b.build()


def behavioral():
    b = vx.CircuitBuilder()
    b.vsource("in", "0", vx.signals.Sine(0.0, 1.0, freq=1.0))
    b.add(Memristor(("in", "m")), name="M1")
    b.add(vx.NonlinearResistor(("m", "0"), lambda v, p: p * jnp.tanh(v), 1e-3))
    b.add(vx.NonlinearCapacitor(("m", "0"), lambda v, p: p * (v + v**3), 1e-9))
    b.conductance("m", "0", 1e-4, transform="sigmoid", g_max=1e-3)
    return b.build()


class CoupledInductors(vx.Element):
    """N inductors with a full mutual-inductance matrix across devices: its
    devices interact, so it declares ``local = False``."""

    terminals = ("p", "n")
    n_internal = 1
    internal_names = ("i",)
    local = False
    log_l: jax.Array
    k: float = eqx.field(static=True)

    def __init__(self, nodes, l, k=0.5):
        self.nodes = self._devices(nodes)
        self.log_l = self._log_per_device(l)
        self.k = k

    def currents(self, v, x, t):
        i = x[..., 0]
        return vx.through(i), -vx.two_terminal(v)[..., None]

    def charges(self, v, x):
        ll = jnp.exp(self.log_l)
        root = jnp.sqrt(ll)
        n = ll.shape[0]
        M = self.k * jnp.outer(root, root) * (1 - jnp.eye(n)) + jnp.diag(ll)
        return None, (M @ x[..., 0])[..., None]


def coupled():
    # nodes: in=0, a=1, b=2, c=3
    return vx.Circuit({
        "V": vx.VoltageSource((0, -1), vx.signals.Step(0.0, 1.0, delay=1e-6,
                                                       rise=1e-7)),
        "R": vx.Resistor([(0, 1), (2, -1), (3, -1)], jnp.array([10.0, 100.0, 1e3])),
        "K": CoupledInductors([(1, -1), (2, -1), (3, -1)], 1e-3),
    }, ["in", "a", "b", "c"])


def grid(n=10):
    rng = np.random.default_rng(0)
    b = vx.CircuitBuilder()
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                b.resistor(f"n{i}_{j}", f"n{i}_{j + 1}", 0.1 * (1 + rng.random()))
            if i + 1 < n:
                b.resistor(f"n{i}_{j}", f"n{i + 1}_{j}", 0.1 * (1 + rng.random()))
            if rng.random() < 0.2:
                b.isource(f"n{i}_{j}", "0", 1e-3 * (1 + rng.random()))
    for i, j in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
        b.vsource(f"n{i}_{j}", "0", 1.0)
    return b.build()


def ring(n=9):
    b = vx.CircuitBuilder(mos_model=functools.partial(vx.Level1MOSFET, smooth=0.0))
    b.vsource("vdd", "0", 1.2)
    nodes = [f"n{k}" for k in range(n)]
    cmos.ring_oscillator(b, nodes, "vdd")
    for node in nodes:
        b.capacitor(node, "0", 5e-15)
    return b.build(), nodes


CIRCUITS = {
    "rc_ladder": rc_ladder,
    "adder": adder,
    "controlled": controlled,
    "magnetics": magnetics,
    "behavioral": behavioral,
    "coupled": coupled,
    "grid": grid,
    "ring": lambda: ring()[0],
    "netlist": lambda: vx.parse_netlist(NETLIST),
}


def random_state(c, seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(-0.5, 1.5, c.size))


# ------------------------------------------------------- stamped Jacobians


@pytest.mark.parametrize("name", CIRCUITS)
def test_stamped_jacobian_equals_jacfwd(name):
    c = CIRCUITS[name]()
    z, t = random_state(c), jnp.asarray(0.3e-6)
    st = sparse.structure(c)

    @eqx.filter_jit
    def both(c, z):
        G = sparse.jacobian(c, z, t, structure=st)
        C = sparse.jacobian(c, z, t, f=0, q=1, structure=st)
        refs = jax.jacfwd(lambda z_: c.f(z_, t))(z), jax.jacfwd(c.q)(z)
        return (st.to_dense(G), st.to_dense(C)), refs

    # equal up to summation order (e.g. a PMOS with source and bulk on vdd)
    for A, ref in zip(*both(c, z)):
        assert jnp.allclose(A, ref, rtol=1e-13, atol=1e-15 * jnp.abs(ref).max())


def test_pattern_covers_device_blocks_only():
    c = rc_ladder()
    P = sparse.structure(c).to_scipy(np.ones(sparse.structure(c).nnz)).toarray()
    a, out, vin = c.node("a"), c.node("out"), c.layout.internal("Vin")
    assert P[a, out] and P[out, a] and P[vin, c.node("in")]
    assert not P[c.node("in"), out]  # no device connects in and out
    assert np.all(np.diag(P))


def test_structure_is_cached_per_topology():
    assert sparse.structure(rc_ladder()) is sparse.structure(rc_ladder(R1=5.0))


# ---------------------------------------------------- the sparse LU itself


@pytest.mark.parametrize("name,leaf", [("adder", 1), ("controlled", 1), ("grid", 1),
                                       ("magnetics", 16), ("grid", 16)])
def test_factor_solve_matches_dense(name, leaf):
    """Real, transposed and complex solves, with supernodes from single
    unknowns (pivoting only inside pivot groups) to dense blocks."""
    c = CIRCUITS[name]()
    z = vx.dc(c, options=DENSE).z
    st = sparse.structure(c)
    plan = sparse._make_plan(st, leaf)
    G = sparse.jacobian(c, z, structure=st).at[st.diagonal[: c.n_nodes]].add(1e-12)
    C = sparse.jacobian(c, z, f=0, q=1, structure=st)
    b = jnp.asarray(np.random.default_rng(1).standard_normal(c.size))

    @jax.jit
    def solves(A, b):
        fac = sparse.factor(plan, A)
        dense = st.to_dense(A)
        return [(sparse.solve(plan, fac, b, tr),
                 jnp.linalg.solve(dense.T if tr else dense, b)) for tr in (0, 1)]

    for A in (G, G + C / 1e-9, G + 2j * jnp.pi * 1e5 * C):
        for x, ref in solves(A, b.astype(A.dtype)):
            assert jnp.allclose(x, ref, rtol=1e-8, atol=1e-10 * jnp.abs(ref).max())


def test_plan_statistics_and_voltage_source_pairing():
    c = grid(12)
    st = sparse.structure(c)
    plan = st.plan
    assert plan.stats["supernodes"] > 1 and plan.stats["levels"] > 2
    gid = sparse._pivot_groups(st)
    for name in ("V1", "V2", "V3", "V4"):  # each pad current is paired with its node
        group, idx = c.layout.device(name)
        node = int(c.elements[group].node_array()[idx, 0])
        assert gid[c.layout.internal(name)] == gid[node]


def test_spsolve_gradients_match_dense():
    c = magnetics()
    st = sparse.structure(c)
    z = vx.dc(c, options=DENSE).z
    G = sparse.jacobian(c, z, structure=st)
    b = jnp.asarray(np.random.default_rng(2).standard_normal(c.size))

    def loss(solver, data, b):
        x = solver(data, b)
        return jnp.sum(jnp.sin(x)), x

    sp_ = lambda d, b: sparse.spsolve(st, d, b)  # noqa: E731
    de_ = lambda d, b: jnp.linalg.solve(st.to_dense(d), b)  # noqa: E731
    gs = jax.grad(lambda d, b: loss(sp_, d, b)[0], argnums=(0, 1))(G, b)
    gd = jax.grad(lambda d, b: loss(de_, d, b)[0], argnums=(0, 1))(G, b)
    for a, r in zip(gs, gd):
        assert jnp.allclose(a, r, rtol=1e-7, atol=1e-9 * jnp.abs(r).max())


# ------------------------------------------- analyses: sparse == dense


@pytest.mark.parametrize("name", ["adder", "controlled", "magnetics", "grid",
                                  "coupled", "netlist"])
def test_dc_sparse_matches_dense(name):
    c = CIRCUITS[name]()
    zs, zd = vx.dc(c, options=SPARSE), vx.dc(c, options=DENSE)
    assert zs.converged and zd.converged
    assert jnp.allclose(zs.z, zd.z, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name,t_end,method", [
    ("magnetics", 2e-4, "trap"), ("behavioral", 0.5, "be"), ("coupled", 5e-6, "trap"),
    ("netlist", 1e-8, "be")])
def test_transient_sparse_matches_dense(name, t_end, method):
    c = CIRCUITS[name]()
    ts = jnp.linspace(0, t_end, 101)
    ic = c.state() if name == "behavioral" else "dc"
    s = vx.transient(c, ts, ic=ic, method=method, options=SPARSE)
    d = vx.transient(c, ts, ic=ic, method=method, options=DENSE)
    assert s.converged.all() and d.converged.all()
    assert jnp.allclose(s.z, d.z, rtol=1e-9, atol=1e-11)


def test_ring_oscillator_transient_sparse_matches_dense():
    c, nodes = ring(9)
    z0 = c.state(v={"vdd": 1.2, **{n: 1.2 * (k % 2) for k, n in enumerate(nodes)}})
    ts = jnp.linspace(0, 2e-10, 201)
    s = vx.transient(c, ts, ic=z0, options=SPARSE)
    d = vx.transient(c, ts, ic=z0, options=DENSE)
    assert jnp.allclose(s.v(), d.v(), atol=1e-9)


@pytest.mark.parametrize("name", ["controlled", "magnetics", "netlist"])
def test_ac_sparse_matches_dense(name):
    c = CIRCUITS[name]()
    freqs = jnp.logspace(1, 7, 25)
    s, d = vx.ac(c, freqs, options=SPARSE), vx.ac(c, freqs, options=DENSE)
    assert jnp.allclose(s.z, d.z, rtol=1e-9, atol=1e-12)


def test_linearize_is_dense_and_exact():
    c = controlled()
    z = vx.dc(c).z
    G, C = eqx.filter_jit(vx.linearize)(c, z)
    assert G.shape == (c.size, c.size)
    assert jnp.allclose(G, jax.jacfwd(lambda z_: c.f(z_, 0.0))(z), rtol=1e-13)
    assert jnp.allclose(C, jax.jacfwd(c.q)(z), rtol=1e-13)


# ------------------------------------------------------------ gradients


@pytest.mark.parametrize("method", ["be", "trap"])
def test_transient_gradients_sparse_dense_and_fd(method):
    ts = jnp.linspace(0, 5e-6, 41)

    def loss(log_p, options):
        R1, R2, C1, C2 = jnp.exp(log_p)
        sol = vx.transient(rc_ladder(R1, R2, C1, C2), ts, ic="zero", method=method,
                           options=options)
        return jnp.mean(sol.v("out") ** 2)

    x = np.log([1e3, 2e3, 1e-9, 0.5e-9])
    gs = jax.grad(loss)(x, SPARSE)
    gd = jax.grad(loss)(x, DENSE)
    assert jnp.allclose(gs, gd, rtol=1e-9)
    assert np.allclose(gs, fd_grad(lambda x: float(loss(x, SPARSE)), x), rtol=1e-5,
                       atol=1e-10)


def test_dc_gradient_of_power_grid_matches_dense():
    c = grid(8)
    worst = "n4_4"

    def v(circuit, options):
        return vx.dc(circuit, options=options).v(worst)

    gs = eqx.filter_grad(v)(c, SPARSE).elements["Resistor"].log_r
    gd = eqx.filter_grad(v)(c, DENSE).elements["Resistor"].log_r
    assert jnp.allclose(gs, gd, rtol=1e-8, atol=1e-14)
    # spot-check one entry against a central finite difference
    k = int(jnp.argmax(jnp.abs(gs)))
    eps = 1e-5
    name = c.devices("Resistor")[k]
    r = c.get(name, "r")
    up = v(c.set(name, r=r * np.exp(eps)), SPARSE)
    dn = v(c.set(name, r=r * np.exp(-eps)), SPARSE)
    assert np.isclose(gs[k], (up - dn) / (2 * eps), rtol=1e-6)


def test_ac_gradient_sparse_matches_dense():
    def gain(log_r, options):
        c = controlled().set("R8", r=jnp.exp(log_r))
        return vx.ac(c, jnp.array([1e3, 1e5]), options=options).db("o2").sum()

    x = jnp.log(1e3)
    assert jnp.isclose(jax.grad(gain)(x, SPARSE), jax.grad(gain)(x, DENSE), rtol=1e-8)


def test_nonlinear_dc_gradient_through_cmos_adder():
    def s0(vdd, options):
        c = adder().set("V1", value=vdd)
        return vx.dc(c, options=options).v("s0")

    gs = jax.grad(s0)(1.2, SPARSE)
    assert jnp.isclose(gs, jax.grad(s0)(1.2, DENSE), rtol=1e-7)
    assert np.isclose(gs, fd_grad(lambda x: float(s0(x[0], SPARSE)),
                                  np.array([1.2]), 1e-5)[0], rtol=1e-5)


# ------------------------------------------------------- transformations


def test_vmap_and_jit_through_sparse_dc():
    c = grid(6)

    @jax.jit
    def worst(v_pad):
        return vx.dc(c.set("V1", value=v_pad), options=SPARSE).v().min()

    pads = jnp.linspace(0.8, 1.2, 4)
    batched = jax.vmap(worst)(pads)
    looped = jnp.stack([worst(p) for p in pads])
    assert jnp.allclose(batched, looped, rtol=1e-12)


def test_vmap_of_sparse_transient_gradient():
    c, nodes = ring(5)

    def final(scale):
        cs = c.set("V1", value=1.2 * scale)
        z0 = cs.state(v={"vdd": 1.2 * scale, nodes[0]: 1.2 * scale})
        return vx.transient(cs, jnp.linspace(0, 5e-11, 21), ic=z0,
                            options=SPARSE).v(nodes[2])[-1]

    scales = jnp.array([0.9, 1.0, 1.1])
    g_batched = jax.vmap(jax.grad(final))(scales)
    g_loop = jnp.stack([jax.grad(final)(s) for s in scales])
    assert jnp.allclose(g_batched, g_loop, rtol=1e-10)


def test_auto_threshold_and_validation():
    auto = vx.Options(solver="auto")  # (VOLTAX_SOLVER may change the default)
    assert analysis_uses_sparse(grid(15), auto)  # 229 unknowns
    assert not analysis_uses_sparse(rc_ladder(), auto)
    assert analysis_uses_sparse(rc_ladder(),
                                vx.Options(solver="auto", sparse_threshold=1))
    with pytest.raises(ValueError, match="solver"):
        vx.Options(solver="klu")


def analysis_uses_sparse(c, opts):
    from voltax.analysis import _uses_sparse

    return _uses_sparse(c, opts)
