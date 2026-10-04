"""RC step response: voltax vs ngspice vs the analytical solution.

For many random (R, C) pairs we simulate the step response of a first-order
RC low-pass in both simulators and check three things:

1. *Matched discretization.* ngspice runs backward Euler (``method=gear
   maxord=1``) with its own adaptive step control; voltax then integrates with
   backward Euler on exactly ngspice's accepted time points. Both solve the
   same discrete equations, so they should agree to rounding error, and both
   should match the exact backward-Euler recurrence
   ``v[n] = (v[n-1] + h/tau) / (1 + h/tau)``.
2. *Discretization error* of each simulator against ``1 - exp(-t/tau)``.
3. *Batched simulation*: all configurations in one ``jax.vmap`` call on a
   uniform grid (``h = tau/100``), checked against the same recurrence.

Run:  uv run python benchmarks/benchmark_rc.py   (VOLTAX_FAST=1 for a quick run)
"""

from __future__ import annotations

import _common as cm  # first: configures JAX for single-threaded CPU
import jax
import matplotlib.pyplot as plt
import numpy as np

import voltax as vx

N_CONFIGS = cm.pick(10, 100)
SEED = 42
VIN = 1.0
STEPS_PER_TAU = 100
N_TAU = 5


def rc_circuit(r: float, c: float) -> vx.Circuit:
    b = vx.CircuitBuilder()
    b.vsource("in", "0", VIN, name="V1")
    b.resistor("in", "out", r, name="R1")
    b.capacitor("out", "0", c, name="C1")
    return b.build()


def ngspice_netlist(r: float, c: float) -> str:
    tau = r * c
    dt = tau / STEPS_PER_TAU
    return f"""RC step
V1 in 0 DC {VIN}
R1 in out {cm.num(r)}
C1 out 0 {cm.num(c)} IC=0
.options method=gear maxord=1
.tran {cm.num(dt)} {cm.num(N_TAU * tau)} 0 {cm.num(dt)} UIC
"""


def be_recurrence(ts: np.ndarray, tau: float) -> np.ndarray:
    """Exact backward-Euler solution of the RC step on grid `ts`."""
    v = np.zeros(len(ts))
    for k in range(1, len(ts)):
        a = (ts[k] - ts[k - 1]) / tau
        v[k] = (v[k - 1] + a * VIN) / (1 + a)
    return v


def main() -> None:
    cm.header(f"RC step response: {N_CONFIGS} random (R, C) configurations")
    rng = np.random.default_rng(SEED)
    rs = 10 ** rng.uniform(2, 4, N_CONFIGS)  # 100 ohm .. 10 kohm
    cs = 10 ** rng.uniform(-10, -5, N_CONFIGS)  # 100 pF .. 10 uF

    template = rc_circuit(1e3, 1e-9)
    ic = template.state(v={"in": VIN, "out": 0.0})

    rows, waves = [], None
    for k, (r, c) in enumerate(zip(rs, cs)):
        tau = r * c
        ng = cm.run_ngspice(ngspice_netlist(r, c), ["v(out)"], repeats=3)
        # with UIC ngspice does not output t = 0; prepend it (v(out) = 0 there)
        ts = np.concatenate([[0.0], ng["time"]])
        circuit = template.set("R1", r=r).set("C1", c=c)
        sol, first, steady = cm.time_call(
            lambda: vx.transient(circuit, ts, ic=ic, method="be"), repeats=3)
        v_vx, v_ng = np.asarray(sol.v("out")), np.concatenate([[0.0], ng["v(out)"]])
        v_be = be_recurrence(ts, tau)
        v_exact = VIN * (1 - np.exp(-ts / tau))
        rows.append(dict(
            tau=tau, points=len(ts), first=first, steady=steady, ng_wall=ng.wall,
            ng_analysis=ng.analysis,
            vx_vs_ng=cm.errors(v_vx, v_ng)[0], vx_vs_be=cm.errors(v_vx, v_be)[0],
            ng_vs_be=cm.errors(v_ng, v_be)[0], vx_vs_exact=cm.errors(v_vx, v_exact)[0],
            converged=bool(np.all(sol.converged)),
        ))
        if k == 0:
            waves = (ts / tau, v_vx, v_ng, v_exact)

    # ---------------------------------------------------- batched (vmap) run
    ts_norm = np.linspace(0.0, N_TAU, N_TAU * STEPS_PER_TAU + 1)

    @jax.jit
    def batched(r, c):
        def one(r_, c_):
            circ = template.set("R1", r=r_).set("C1", c=c_)
            return vx.transient(circ, ts_norm * r_ * c_, ic=ic, method="be").v("out")
        return jax.vmap(one)(r, c)

    v_batch, b_first, b_steady = cm.time_call(lambda: batched(rs, cs), repeats=3)
    # the recurrence only depends on h/tau, so one reference serves all configs
    batch_err = cm.errors(np.asarray(v_batch), be_recurrence(ts_norm, 1.0)[None])[0]

    # ------------------------------------------------------------- report
    get = lambda key: np.array([row[key] for row in rows])  # noqa: E731
    print(f"\nngspice time points per run: {sorted(set(get('points')))}"
          f"   all voltax solves converged: {all(get('converged'))}")
    cm.print_table([
        {"comparison": "voltax vs ngspice (same BE grid)",
         "max abs err (V)": get("vx_vs_ng").max()},
        {"comparison": "voltax vs exact BE recurrence",
         "max abs err (V)": get("vx_vs_be").max()},
        {"comparison": "ngspice vs exact BE recurrence",
         "max abs err (V)": get("ng_vs_be").max()},
        {"comparison": "voltax vs 1-exp(-t/tau) (discretization)",
         "max abs err (V)": get("vx_vs_exact").max()},
        {"comparison": f"voltax vmap ({N_CONFIGS} configs) vs BE recurrence",
         "max abs err (V)": batch_err},
    ], "Accuracy (worst case over configurations; VIN = 1 V, so abs = rel)")
    print("note: the ~1e-8 V voltax-vs-BE residual is voltax's default node-to-ground"
          "\n      gmin (1e-12 S x R <= 10 kohm); with Options(gmin=0) it is ~1e-14.")
    cm.print_table([
        {"run": "voltax, first call of config 0 (trace + compile)",
         "time": cm.fmt_s(get("first")[0])},
        {"run": "voltax, steady state (median per config)",
         "time": cm.fmt_s(np.median(get("steady")))},
        {"run": "ngspice wall, whole process (median per config)",
         "time": cm.fmt_s(np.median(get("ng_wall")))},
        {"run": "ngspice internal analysis time (median per config)",
         "time": cm.fmt_s(np.median(get("ng_analysis")))},
        {"run": f"voltax vmap over {N_CONFIGS} configs: compile (total)",
         "time": cm.fmt_s(b_first - b_steady)},
        {"run": f"voltax vmap over {N_CONFIGS} configs: steady, per config",
         "time": cm.fmt_s(b_steady / N_CONFIGS)},
    ], "Runtime (same grid for both simulators)")

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    t, v_vx, v_ng, v_exact = waves
    ax = axes[0]
    ax.plot(t, v_exact, "k-", lw=1, label="analytical")
    ax.plot(t, v_ng, "C0-", lw=2.5, alpha=0.6, label="ngspice (BE)")
    ax.plot(t, v_vx, "C3--", lw=1.5, label="voltax (BE, same grid)")
    ax.set(xlabel="t / tau", ylabel="v(out) (V)", title="Step response (config 0)")
    ax.legend()
    ax = axes[1]
    taus = get("tau")
    ax.loglog(taus, np.maximum(get("vx_vs_ng"), 1e-17), "o", label="voltax vs ngspice")
    ax.loglog(taus, get("vx_vs_exact"), "s", mfc="none", label="voltax vs analytical")
    ax.set(xlabel="tau (s)", ylabel="max abs error (V)", title="Accuracy per config")
    ax.legend()
    ax = axes[2]
    ax.loglog(taus, get("steady") * 1e3, "o", label="voltax (steady)")
    ax.loglog(taus, get("ng_wall") * 1e3, "s", label="ngspice wall")
    ax.axhline(b_steady / N_CONFIGS * 1e3, color="C3", ls="--",
               label="voltax vmap, per config")
    ax.set(xlabel="tau (s)", ylabel="time (ms)", title="Runtime per config")
    ax.legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cm.save_figure(fig, "rc")


if __name__ == "__main__":
    main()
