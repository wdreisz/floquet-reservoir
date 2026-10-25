"""Qiskit Fall Fest 2026 (Warsaw) - Track 2, Quantum machine learning.

Interactive demo: a disordered kicked-Ising (Floquet) spin chain as a quantum reservoir computer, compared with an
echo-state network (ESN) and linear autoregression (AR) on memory capacity and on two real time series.
Everything here is classical simulation on this laptop (NumPy density matrices). No quantum hardware is used and no
IBM service is contacted.

Qiskit is NEVER imported here (building Qiskit circuits inside Streamlit threads segfaults); the circuit is a
pre-rendered PNG. Only the NumPy/SciPy modules qreservoir.reservoir and qreservoir.tasks are imported.

Launch:  .venv/bin/python -m streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qreservoir.reservoir import FloquetReservoir  # noqa: E402  (pure NumPy; Qiskit is never imported)
from qreservoir.tasks import capacities  # noqa: E402

ART = ROOT / "artifacts"
FIG = ROOT / "figures"
DOCS = ROOT / "docs"

SIM_TAG = "classical simulation — no quantum hardware used"

# Okabe-Ito palette (colour-blind safe); vermillion matches the theme primary colour.
COL = {
    "quantum_reservoir": "#d55e00",
    "quantum_reservoir_J0": "#e69f00",
    "esn": "#0072b2",
    "ar": "#009e73",
    "persistence": "#999999",
    "observed": "#222222",
    "dtc": "#cc79a7",
    "lags": "#009e73",
}
MODEL_LABEL = {
    "quantum_reservoir": "Quantum reservoir (J = 0.4)",
    "quantum_reservoir_J0": "Quantum reservoir, J = 0 control",
    "esn": "Echo-state network (11 nodes)",
    "ar": "Linear AR",
    "persistence": "Persistence (next = last)",
}
SERIES_LABEL = {"sunspots": "Yearly sunspots", "co2": "Monthly CO₂ change (Mauna Loa)"}
J_LABEL = {0.0: "J = 0 (no interaction, control)", 0.4: "J = 0.4 (used in forecasting)", 0.785: "J = 0.785 ≈ π/4"}
CO2_FIRST_MONTH = pd.Timestamp("1958-03-01")  # first month of statsmodels co2 after resample("MS"); index 0 of the level series

st.set_page_config(page_title="Floquet reservoir - Qiskit Fall Fest 2026", page_icon="⚛", layout="wide")

st.markdown(
    """
<style>
.sim-banner {position: sticky; top: 2.9rem; z-index: 999; background: #fff4e6; border: 1px solid #d55e00;
  color: #8a3b00; border-radius: 6px; padding: 0.45rem 0.9rem; font-weight: 600; letter-spacing: 0.02em;
  font-size: 0.9rem; margin-bottom: 0.8rem;}
.question {font-size: 1.55rem; line-height: 1.3; font-weight: 650; color: #1f1f1f; margin: 0.2rem 0 0.4rem 0;}
.kicker {color: #6b6b6b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;}
.interp {border-left: 4px solid #d55e00; padding: 0.4rem 0.8rem; background: #fafafa; margin: 0.3rem 0;}
</style>
<div class="sim-banner">SIMULATION ONLY — classical simulation on this laptop (NumPy density matrices); no quantum hardware used</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="kicker">Qiskit Fall Fest 2026 · Warsaw · Track 2: Quantum machine learning</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="question">Does time-crystal order give a quantum reservoir better memory — and does any of it beat '
    "classical models on real time series?</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- saved results
@st.cache_data
def load_artifact(name: str) -> dict:
    with open(ART / f"{name}.json") as f:
        return json.load(f)


def med(x) -> float:
    return float(np.median(np.asarray(x, dtype=float)))


def runs_of(e2s: dict, model: str) -> list[dict]:
    return sorted((r for r in e2s["runs"] if r["model"] == model), key=lambda r: r["seed"])


def ar_sig_better(r: dict) -> bool:
    """DM test model vs AR: positive mean loss difference = AR has smaller squared error."""
    d = r.get("dm_vs_ar")
    return bool(d and d["p_value"] < 0.05 and d["mean_loss_diff"] > 0)


def fmt_float(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


S = load_artifact("summary")
E1 = load_artifact("e1_phase_scan")
E2 = load_artifact("e2_forecast")
E3 = load_artifact("e3_shots_noise")
MEM = S["memory"]

tabs = st.tabs(["Overview", "Phase explorer", "Forecasts", "Shots & noise", "Method & honesty"])

# ================================================================ 1. Overview
with tabs[0]:
    fc = S["forecast"]
    best_q = {s: fc[s]["quantum_reservoir"]["median_test_nmse"] for s in fc}
    st.markdown(
        f"""
**Short answer: no, on both counts** ({SIM_TAG}).

* **Memory.** The time-crystal regime (θ = {MEM['dtc_theta_over_pi']:g}π, period-doubling order
  {MEM['dtc_order_at_0.95']:.2f}) has the *least* linear memory of the interacting chain: MC = {MEM['dtc_MC']:.2f},
  barely above the J = 0 control (max {MEM['J0_MC_max']:.2f}). The scrambling regime peaks at θ = {MEM['best_theta_over_pi']:g}π
  with MC = {MEM['best_MC']:.2f}. Across the scan, MC and time-crystal order are anti-correlated
  (r = {MEM['corr_MC_vs_dtc_order']:+.2f}). An 11-node ESN reaches MC {MEM['esn_MC_range'][0]:.1f}–{MEM['esn_MC_range'][1]:.1f};
  11 raw lags reach {MEM['lags11_MC']:.1f}.
* **Forecasting.** Linear AR wins on both series and the ESN is second. The quantum reservoir never beats AR on any seed
  (sunspots median NMSE {best_q['sunspots']:.3f} vs AR {fc['sunspots']['ar_test_nmse']:.3f};
  CO₂ {best_q['co2']:.3f} vs AR {fc['co2']['ar_test_nmse']:.3f}).
* **What the quantum reservoir does do.** Its interactions matter: on CO₂ it clearly beats its own J = 0 control
  (median {best_q['co2']:.3f} vs {fc['co2']['quantum_reservoir_J0']['median_test_nmse']:.3f}).
* **Measurement cost.** With finite shots the scrambling regime's memory collapses (MC at θ = 0.5π:
  {S['shots_MC']['0.5']['exact']:.2f} exact → {S['shots_MC']['0.5']['1000']:.2f} at 1000 shots →
  {S['shots_MC']['0.5']['100']:.2f} at 100 shots).
"""
    )
    st.markdown("**Linear memory capacity MC** (saved E1 scan, iid input, 6 spins, 11 readout features)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Best quantum · θ={MEM['best_theta_over_pi']:g}π", f"{MEM['best_MC']:.2f}", border=True,
              help="Mean over seeds at the best (θ, J, cycles) cell of the scan")
    c2.metric(f"Time crystal · θ={MEM['dtc_theta_over_pi']:g}π", f"{MEM['dtc_MC']:.2f}",
              delta=f"{MEM['dtc_MC'] - MEM['best_MC']:+.2f} vs best", delta_color="normal", border=True)
    c3.metric("ESN, 11 nodes", f"{MEM['esn_MC_range'][0]:.1f}–{MEM['esn_MC_range'][1]:.1f}", border=True)
    c4.metric("11 raw lags", f"{MEM['lags11_MC']:.1f}", border=True,
              help="Upper bound for 11 linear features: each lag stores one delay perfectly")

    for s in ("sunspots", "co2"):
        f = fc[s]
        st.markdown(f"**{SERIES_LABEL[s]}** — one-step test NMSE (lower is better; median over "
                    f"{f['quantum_reservoir']['n_seeds']} seeds)")
        cols = st.columns(5)
        cols[0].metric(f"AR (p={f['ar_p']})", f"{f['ar_test_nmse']:.3f}", border=True)
        cols[1].metric("ESN", f"{f['esn']['median_test_nmse']:.3f}",
                       delta=f"{f['esn']['median_test_nmse'] - f['ar_test_nmse']:+.3f} vs AR", delta_color="inverse",
                       border=True)
        cols[2].metric("Quantum", f"{f['quantum_reservoir']['median_test_nmse']:.3f}",
                       delta=f"{f['quantum_reservoir']['median_test_nmse'] - f['ar_test_nmse']:+.3f} vs AR",
                       delta_color="inverse", border=True,
                       help=f"AR significantly better (DM p < 0.05) on "
                            f"{f['quantum_reservoir']['seeds_where_AR_significantly_better_p05']}/"
                            f"{f['quantum_reservoir']['n_seeds']} seeds; quantum beats AR on "
                            f"{f['quantum_reservoir']['seeds_beating_AR']} seeds")
        cols[3].metric("Quantum J=0", f"{f['quantum_reservoir_J0']['median_test_nmse']:.3f}", border=True)
        cols[4].metric("Persistence", f"{f['persistence_test_nmse']:.3f}", border=True)

    ro = S.get("rolling_origin")
    if ro:
        st.markdown("**Rolling-origin re-evaluation** (saved, several forecast origins instead of one test block)")
        for s, block in ro.items():
            models = sorted(set(block.get("median_test_nmse", {})) | set(block.get("mean_rank", {}))
                            | set(block.get("wins", {})))
            rows = [{"model": MODEL_LABEL.get(m, m),
                     "median test NMSE": block.get("median_test_nmse", {}).get(m),
                     "mean rank (1 = best)": block.get("mean_rank", {}).get(m),
                     "wins": block.get("wins", {}).get(m)} for m in models]
            rows.sort(key=lambda r: (r["mean rank (1 = best)"] is None, r["mean rank (1 = best)"] or 0))
            folds = block.get("folds")
            nf = len(folds) if isinstance(folds, (list, tuple)) else folds
            st.caption(f"{SERIES_LABEL.get(s, s)} — {nf} folds · {SIM_TAG}")
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    left, right = st.columns([3, 2])
    with left:
        if (FIG / "circuit.png").exists():
            st.image(str(FIG / "circuit.png"), width="stretch",
                     caption="One input step, pre-rendered with Qiskit (no circuit is built in this app): reset qubit 0 "
                             "to √(1−u)|0⟩ + √u|1⟩, then Floquet cycles of R_x(θ) kicks, ZZ(J) couplings and random "
                             "Z fields h_i ∈ [−π, π]; read ⟨Z_i⟩ and ⟨Z_i Z_{i+1}⟩ (11 features).")
    with right:
        st.markdown(
            """
**The model.** A chain of 6 spins with random fields. Each time step, one spin is overwritten with the new input
value; then the chain is kicked: every spin is rotated by θ, neighbours interact with strength J.

* **θ ≈ π, J ≠ 0** — a discrete time crystal: spins flip every cycle, magnetisation oscillates with period 2 and is
  robust. Intuition said "stable order = long memory".
* **θ ≈ π/2** — scrambling: information spreads across the chain.
* **J = 0** — no interactions: the spins never talk, so the chain cannot store past inputs beyond the input spin.

A linear ridge readout on the 11 expectation values is the only trained part (same readout for every model).
"""
        )
    st.caption(S["statement"])

# ================================================================ 2. Phase explorer
with tabs[1]:
    df1 = pd.DataFrame([{k: r[k] for k in ("theta_over_pi", "J", "cycles_per_step", "seed", "MC", "QC", "dtc_order")}
                        for r in E1["rows"]])
    Js = sorted(df1["J"].unique())
    cycs = sorted(df1["cycles_per_step"].unique())
    c1, c2, c3 = st.columns(3)
    J_sel = c1.selectbox("Coupling J", Js, index=Js.index(0.4) if 0.4 in Js else 0,
                         format_func=lambda j: J_LABEL.get(j, f"J = {j:g}"), key="e1_J")
    cyc_sel = c2.radio("Floquet cycles per input step", cycs, index=len(cycs) - 1, horizontal=True, key="e1_cyc")
    metric = c3.radio("Capacity", ["MC", "QC"], horizontal=True, key="e1_metric",
                      format_func=lambda m: {"MC": "Linear memory MC", "QC": "Quadratic capacity QC"}[m])

    sub = df1[(df1["J"] == J_sel) & (df1["cycles_per_step"] == cyc_sel)]
    g = sub.groupby("theta_over_pi").agg(m=(metric, "mean"), sd=(metric, "std"), dtc=("dtc_order", "mean"),
                                         n=("seed", "count")).reset_index()
    esn = pd.DataFrame(E1["esn"])
    esn_lo, esn_hi = float(esn[metric].min()), float(esn[metric].max())
    lag_v = E1["lags11"][metric]

    fig = go.Figure()
    x0, x1 = float(g["theta_over_pi"].min()), float(g["theta_over_pi"].max())
    fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[esn_lo, esn_lo, esn_hi, esn_hi], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor="rgba(0,114,178,0.13)",
                             name=f"ESN, 11 nodes (range over {len(esn)} runs: {esn_lo:.1f}–{esn_hi:.1f})",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[lag_v, lag_v], mode="lines", line=dict(color=COL["lags"], dash="dash"),
                             name=f"11 raw lags, linear ({lag_v:.1f})", hovertemplate=f"11 lags: {lag_v:.2f}<extra></extra>"))
    # other J values faintly, for context
    for j in Js:
        if j == J_sel:
            continue
        gj = df1[(df1["J"] == j) & (df1["cycles_per_step"] == cyc_sel)].groupby("theta_over_pi")[metric].mean()
        fig.add_trace(go.Scatter(x=gj.index, y=gj.values, mode="lines", line=dict(color="#bbbbbb", width=1.5),
                                 name=J_LABEL.get(j, f"J = {j:g}") + " (mean)",
                                 hovertemplate="θ/π = %{x}<br>" + metric + " %{y:.2f}<extra>J = " + f"{j:g}</extra>"))
    fig.add_trace(go.Scatter(
        x=g["theta_over_pi"], y=g["m"], mode="lines+markers", name=f"Quantum reservoir, {J_LABEL.get(J_sel, J_sel)}",
        line=dict(color=COL["quantum_reservoir"], width=3),
        error_y=dict(type="data", array=g["sd"].fillna(0), thickness=1, width=3),
        hovertemplate="θ/π = %{x}<br>" + metric + " %{y:.2f} (mean of seeds)<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["theta_over_pi"], y=g["dtc"], mode="lines", yaxis="y2", name="Time-crystal order (right axis)",
                             line=dict(color=COL["dtc"], dash="dot", width=2),
                             hovertemplate="θ/π = %{x}<br>DTC order %{y:.2f}<extra></extra>"))
    fig.add_vrect(x0=0.9, x1=x1 + 0.01, fillcolor=COL["dtc"], opacity=0.06, line_width=0,
                  annotation_text="time-crystal side", annotation_position="bottom left")
    fig.update_layout(
        title=dict(text=f"{metric} vs kick angle θ ({cyc_sel} cycle(s)/step, mean ± sd over {int(g['n'].max())} seeds)",
                   font=dict(size=14)),
        xaxis=dict(title="kick angle θ / π  (1 = perfect spin flip)"),
        yaxis=dict(title=f"{metric} (sum of r² over delays 0–15)", rangemode="tozero"),
        yaxis2=dict(title="time-crystal order", overlaying="y", side="right", range=[-1.05, 1.05], tickvals=[-1, -0.5, 0, 0.5, 1], showgrid=False),
        height=480, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.22, x=0))
    st.plotly_chart(fig, width="stretch")
    corr = float(np.corrcoef(sub["MC"], sub["dtc_order"])[0, 1])
    pk = g.loc[g["m"].idxmax()]
    g_dtc = g.loc[(g["theta_over_pi"] - MEM["dtc_theta_over_pi"]).abs().idxmin()]
    st.markdown(
        f'<div class="interp">At this setting {metric} peaks at θ = {pk["theta_over_pi"]:g}π ({pk["m"]:.2f}); at '
        f"θ = {g_dtc['theta_over_pi']:g}π (time-crystal order {g_dtc['dtc']:+.2f}) it is {g_dtc['m']:.2f}. "
        f"Correlation of MC with time-crystal order across "
        f"seeds and angles here: r = {corr:+.2f}. Even the best quantum setting stays inside or below the ESN band and "
        f"far below 11 plain lags.</div>", unsafe_allow_html=True)

    # memory function heatmap
    lbd = [r for r in E1["rows"] if r["J"] == J_sel and r["cycles_per_step"] == cyc_sel]
    thetas = sorted({r["theta_over_pi"] for r in lbd})
    Z = np.array([np.mean([r["linear_by_delay"] for r in lbd if r["theta_over_pi"] == th], axis=0) for th in thetas])
    hm = go.Figure(go.Heatmap(z=Z.T, x=thetas, y=list(range(Z.shape[1])), colorscale="Oranges", zmin=0, zmax=1,
                              colorbar=dict(title="r²"),
                              hovertemplate="θ/π = %{x}<br>delay k = %{y}<br>r² = %{z:.2f}<extra></extra>"))
    hm.update_layout(title=dict(text="Memory function: how well u(t − k) can be read back, per delay", font=dict(size=14)),
                     xaxis_title="θ / π", yaxis_title="delay k (steps)", height=340, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(hm, width="stretch")
    st.caption(f"Saved E1 scan · {E1['experiment']} · created {E1['provenance']['created_utc']} · {SIM_TAG}.")

    st.divider()
    st.subheader("Live: run the reservoir yourself")
    st.caption("Same protocol as the saved scan: 1500 iid uniform inputs (rng seed 100 + seed), 200-step washout, "
               "ridge readout fitted on 1000 steps, r² measured on the last 300. Computed here, " + SIM_TAG + ".")
    l1, l2, l3, l4 = st.columns(4)
    th_live = l1.slider("θ / π", 0.05, 1.0, 0.5, 0.05, key="live_theta")
    J_live = l2.select_slider("J", options=[0.0, 0.2, 0.4, 0.6, 0.785, 1.0], value=0.4, key="live_J")
    seed_live = l3.slider("Disorder seed", 0, 7, 0, key="live_seed")
    cyc_live = l4.radio("Cycles / step", [1, 2], index=1, horizontal=True, key="live_cyc")

    @st.cache_data(show_spinner=False)
    def live_capacity(theta_over_pi: float, J: float, seed: int, cycles: int) -> dict:
        u = np.random.default_rng(100 + seed).uniform(size=1500)
        t0 = time.perf_counter()
        res = FloquetReservoir(n=6, theta=np.pi * theta_over_pi, J=J, cycles_per_step=cycles, seed=seed)
        c = capacities(res.run(u), u)
        # period-doubling score of the same chain without input (mirrors experiments.dtc_order)
        psi = np.zeros(2**res.n, dtype=complex)
        psi[0] = 1
        s = []
        for t in range(1, 41):
            psi = res.U @ psi
            if t > 20:
                s.append((-1) ** t * float(np.abs(psi) ** 2 @ res.z.mean(axis=1)))
        c["dtc_order"] = float(np.mean(s))
        c["seconds"] = time.perf_counter() - t0
        return c

    with st.spinner(f"Simulating the 6-spin density matrix for 1500 steps ({SIM_TAG})..."):
        live = live_capacity(round(float(th_live), 3), float(J_live), int(seed_live), int(cyc_live))
    saved = df1[(np.isclose(df1["theta_over_pi"], th_live)) & (np.isclose(df1["J"], J_live))
                & (df1["cycles_per_step"] == cyc_live) & (df1["seed"] == seed_live)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linear memory MC", f"{live['MC']:.2f}", border=True,
              help=(f"Saved scan value for this cell: {saved['MC'].iloc[0]:.2f}" if len(saved) else
                    "This cell is not in the saved scan"))
    m2.metric("Quadratic capacity QC", f"{live['QC']:.2f}", border=True)
    m3.metric("Time-crystal order", f"{live['dtc_order']:+.2f}", border=True,
              help="Mean of (−1)^t m_z over cycles 21–40 from all-up, no input. ≈ 1 = robust period doubling")
    m4.metric("Wall time", f"{live['seconds']:.2f} s", border=True)
    bars = go.Figure()
    bars.add_trace(go.Bar(x=list(range(len(live["linear_by_delay"]))), y=live["linear_by_delay"], name="linear r²(k)",
                          marker_color=COL["quantum_reservoir"]))
    bars.add_trace(go.Bar(x=list(range(len(live["quadratic_by_delay"]))), y=live["quadratic_by_delay"],
                          name="quadratic r²(k): u(t−k)·u(t−k−1)", marker_color=COL["quantum_reservoir_J0"]))
    bars.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1.02], title="r²"),
                       xaxis_title="delay k (steps)", legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(bars, width="stretch")
    if len(saved):
        st.caption(f"Matches a saved-scan cell: saved MC = {saved['MC'].iloc[0]:.3f}, live MC = {live['MC']:.3f}.")
    st.caption("Try θ = 0.95π vs 0.5π at J = 0.4, then set J = 0: without interactions the memory is gone at every θ.")

# ================================================================ 3. Forecasts
with tabs[2]:
    c1, c2 = st.columns([2, 1])
    series = c1.radio("Series", list(SERIES_LABEL), format_func=SERIES_LABEL.get, horizontal=True, key="fc_series")
    e2s = E2[series]
    seeds = sorted({r["seed"] for r in e2s["runs"]})
    seed = c2.selectbox("Reservoir seed (disorder / ESN weights)", seeds, key="fc_seed")
    lo, hi = e2s["scale"]
    unscale = lambda z: lo + np.asarray(z, dtype=float) * (hi - lo)  # noqa: E731
    tt = np.asarray(e2s["test_years"])
    if series == "sunspots":
        x = tt + 1  # prediction made at year t is for year t + 1
        xlab, unit = "target year", "sunspot number"
    else:
        x = [CO2_FIRST_MONTH + pd.DateOffset(months=int(i) + 2) for i in tt]  # diff index i+1 ends at month i+2
        xlab, unit = "target month", "CO₂ change vs previous month (ppm)"

    pick = {m: next(r for r in runs_of(e2s, m) if r["seed"] == seed) for m in ("quantum_reservoir", "quantum_reservoir_J0", "esn")}
    true = unscale(pick["quantum_reservoir"]["true"])
    show = st.multiselect("Show", ["ar", "esn", "quantum_reservoir", "quantum_reservoir_J0"],
                          default=["ar", "esn", "quantum_reservoir"], format_func=MODEL_LABEL.get, key="fc_show")
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=x, y=true, name="Observed (test block)", mode="lines+markers",
                            line=dict(color=COL["observed"], width=2), marker=dict(size=4)))
    for m in show:
        pred = e2s["ar"]["pred"] if m == "ar" else pick[m]["pred"]
        ff.add_trace(go.Scatter(x=x, y=unscale(pred), name=MODEL_LABEL[m], mode="lines",
                                line=dict(color=COL[m], width=3 if m == "quantum_reservoir" else 2,
                                          dash="dot" if m == "quantum_reservoir_J0" else None)))
    ff.update_layout(title=dict(text=f"{SERIES_LABEL[series]} — one-step-ahead test forecasts, seed {seed}", font=dict(size=14)),
                     xaxis_title=xlab, yaxis_title=unit, height=440, margin=dict(l=10, r=10, t=50, b=10),
                     legend=dict(orientation="h", y=-0.2, x=0))
    st.plotly_chart(ff, width="stretch")
    st.caption(f"{e2s['description']}. Values are mapped back to original units from the training-block min–max scale "
               f"[{lo:.3g}, {hi:.3g}] (targets are not clipped). AR has no seed (deterministic).")

    rows = [{"model": MODEL_LABEL["ar"], "chosen hyper-parameters": f"p = {e2s['ar']['p']}",
             "val NMSE": e2s["ar"]["val_nmse"], "test NMSE": e2s["ar"]["test_nmse"], "DM p vs AR": None, "verdict vs AR": "—"}]
    for m in ("esn", "quantum_reservoir", "quantum_reservoir_J0"):
        r = pick[m]
        dm = r["dm_vs_ar"]
        verdict = ("AR significantly better" if ar_sig_better(r) else
                   ("significantly better than AR" if dm["p_value"] < 0.05 else "no significant difference"))
        ch = ", ".join(f"{k.replace('_over_pi', '/π')} = {v:g}" for k, v in r["chosen"].items())
        rows.append({"model": MODEL_LABEL[m], "chosen hyper-parameters": ch, "val NMSE": r["val_nmse"],
                     "test NMSE": r["test_nmse"], "DM p vs AR": dm["p_value"], "verdict vs AR": verdict})
    rows.append({"model": MODEL_LABEL["persistence"], "chosen hyper-parameters": "—", "val NMSE": None,
                 "test NMSE": e2s["persistence_test_nmse"], "DM p vs AR": None, "verdict vs AR": "—"})
    st.markdown(f"**Seed {seed}: test NMSE per model** (lower is better; 1.0 = predicting the test mean)")
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"val NMSE": st.column_config.NumberColumn(format="%.3f"),
                                "test NMSE": st.column_config.NumberColumn(format="%.3f"),
                                "DM p vs AR": st.column_config.NumberColumn(format="%.2g")})
    st.caption("Diebold–Mariano test (squared error, horizon 1) on the test block against the same AR forecasts. "
               "Hyper-parameters (θ for the quantum reservoir, spectral radius and input scale for the ESN, lag order p "
               "for AR, ridge α for all) were chosen on the validation block only.")

    # all seeds strip plot
    left, right = st.columns(2)
    with left:
        sp = go.Figure()
        for i, m in enumerate(("esn", "quantum_reservoir", "quantum_reservoir_J0")):
            R = runs_of(e2s, m)
            sp.add_trace(go.Box(y=[r["test_nmse"] for r in R], name=MODEL_LABEL[m], boxpoints="all", jitter=0.4,
                                pointpos=0, marker=dict(color=COL[m], size=8), line=dict(color=COL[m]),
                                fillcolor="rgba(0,0,0,0)", customdata=[r["seed"] for r in R],
                                hovertemplate="seed %{customdata}<br>test NMSE %{y:.3f}<extra></extra>"))
        sp.add_hline(y=e2s["ar"]["test_nmse"], line=dict(color=COL["ar"], dash="dash"),
                     annotation_text=f"AR {e2s['ar']['test_nmse']:.3f}", annotation_position="top right")
        sp.add_hline(y=e2s["persistence_test_nmse"], line=dict(color=COL["persistence"], dash="dot"),
                     annotation_text=f"persistence {e2s['persistence_test_nmse']:.3f}", annotation_position="bottom right")
        sp.update_layout(title=dict(text=f"Test NMSE across all {len(seeds)} seeds", font=dict(size=14)),
                         yaxis_title="test NMSE", showlegend=False, height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(sp, width="stretch")
    with right:
        Q = runs_of(e2s, "quantum_reservoir")
        ths = sorted(Q[0]["val_by_theta"], key=float)
        vt = go.Figure()
        vt.add_trace(go.Scatter(x=[float(t) for t in ths], y=[med([r["val_by_theta"][t] for r in Q]) for t in ths],
                                name="validation (median)", mode="lines+markers", line=dict(color="#666666", dash="dash")))
        vt.add_trace(go.Scatter(x=[float(t) for t in ths], y=[med([r["test_by_theta"][t] for r in Q]) for t in ths],
                                name="test (median)", mode="lines+markers", line=dict(color=COL["quantum_reservoir"], width=3)))
        vt.update_layout(title=dict(text="Quantum reservoir NMSE vs θ (J = 0.4), all seeds", font=dict(size=14)),
                         xaxis_title="θ / π", yaxis_title="NMSE", height=420, margin=dict(l=10, r=10, t=50, b=10),
                         legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(vt, width="stretch")
    nq = len(Q)
    st.markdown(
        f'<div class="interp">Across {nq} seeds on {SERIES_LABEL[series].lower()}: AR is significantly better than the '
        f"quantum reservoir on {sum(ar_sig_better(r) for r in Q)}/{nq} seeds and the quantum reservoir beats AR on "
        f"{sum(r['test_nmse'] < e2s['ar']['test_nmse'] for r in Q)}/{nq}. ESN median {med([r['test_nmse'] for r in runs_of(e2s, 'esn')]):.3f}, "
        f"quantum median {med([r['test_nmse'] for r in Q]):.3f}, J = 0 control median "
        f"{med([r['test_nmse'] for r in runs_of(e2s, 'quantum_reservoir_J0')]):.3f}. The right-hand panel shows θ chosen on "
        "validation does not always transfer to the test block — part of the quantum reservoir's seed-to-seed spread.</div>",
        unsafe_allow_html=True)
    st.caption(f"Saved E2 results · {E2['experiment']} · created {E2['provenance']['created_utc']} · {SIM_TAG}.")

    # ---------------- rolling-origin cross-validation (E4)
    ro_all = S.get("rolling_origin")
    if ro_all and series in ro_all:
        ro = ro_all[series]
        st.divider()
        st.subheader("Rolling-origin check: does the ranking hold on other test blocks?")
        folds = ro.get("folds", [])
        pf = ro.get("per_fold_median_test_nmse", {})
        fkeys = sorted(pf, key=int)
        models_ro = [m for m in ("ar", "esn", "quantum_reservoir", "quantum_reservoir_J0", "persistence")
                     if any(m in pf[k] for k in fkeys)]

        def fold_label(k: str) -> str:
            i = int(k)
            if i < len(folds) and len(folds[i]) == 3:
                _, b, c = folds[i]  # (train_end, val_end, test_end) fractions
                return f"test {b:.0%}–{c:.0%}"
            return f"fold {i}"

        fr = go.Figure()
        for m in models_ro:
            fr.add_trace(go.Scatter(
                x=[fold_label(k) for k in fkeys], y=[pf[k].get(m) for k in fkeys], mode="lines+markers",
                name=MODEL_LABEL.get(m, m),
                line=dict(color=COL.get(m, "#777777"), width=3 if m in ("ar", "quantum_reservoir") else 2,
                          dash="dot" if m in ("quantum_reservoir_J0", "persistence") else None),
                hovertemplate="%{x}<br>median test NMSE %{y:.3f}<extra>" + MODEL_LABEL.get(m, m) + "</extra>"))
        fr.update_layout(title=dict(text=f"{SERIES_LABEL[series]} — median test NMSE per fold (over seeds)",
                                    font=dict(size=14)),
                         xaxis_title="test window (fraction of the series); each fold trains on everything before it",
                         yaxis=dict(title="test NMSE", rangemode="tozero"), height=450,
                         margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3, x=0))
        st.plotly_chart(fr, width="stretch")
        rr = [{"model": MODEL_LABEL.get(m, m),
               "median test NMSE": ro.get("median_test_nmse", {}).get(m),
               "mean rank (1 = best)": ro.get("mean_rank", {}).get(m),
               f"fold wins (of {len(fkeys)})": ro.get("wins", {}).get(m)} for m in models_ro]
        rr.sort(key=lambda r: (r["mean rank (1 = best)"] is None, r["mean rank (1 = best)"] or 0))
        st.dataframe(pd.DataFrame(rr), hide_index=True,
                     column_config={"median test NMSE": st.column_config.NumberColumn(format="%.3f"),
                                    "mean rank (1 = best)": st.column_config.NumberColumn(format="%.2f")})
        fold_txt = "; ".join(f"train → {a:.0%}, val → {b:.0%}, test → {c:.0%}" for a, b, c in folds) if folds else ""
        st.caption(f"{len(fkeys)} expanding-window folds ({fold_txt}). Each fold re-tunes every model on its own "
                   f"validation block. {SIM_TAG}.")
        mr = ro.get("mean_rank", {})
        if mr:
            best_m = min(mr, key=mr.get)
            q_beats = sum(1 for k in fkeys if pf[k].get("quantum_reservoir", np.inf) < pf[k].get("ar", -np.inf))
            j0_beat = sum(1 for k in fkeys if pf[k].get("quantum_reservoir", np.inf) < pf[k].get("quantum_reservoir_J0", -np.inf))
            st.markdown(
                f'<div class="interp">Best mean rank: <b>{MODEL_LABEL.get(best_m, best_m)}</b> ({mr[best_m]:.2f}). '
                f"The quantum reservoir beats AR on {q_beats}/{len(fkeys)} folds and beats its own J = 0 control on "
                f"{j0_beat}/{len(fkeys)} folds (fold medians over seeds).</div>", unsafe_allow_html=True)
        if (FIG / "cv_figure.png").exists():
            with st.expander("Static rolling-origin figure (figures/cv_figure.png)"):
                st.image(str(FIG / "cv_figure.png"), width="stretch")

# ================================================================ 4. Shots & noise
with tabs[3]:
    rows3 = E3["rows"]
    st.markdown("Every number above assumes exact ⟨Z⟩ and ⟨ZZ⟩ at every step. Real hardware estimates them from a finite "
                "number of shots, and noise adds dissipation. Saved E3 results, 5 seeds, J = 0.4, " + SIM_TAG + ".")
    shot_order = [None, 10000, 1000, 100]
    shot_lab = {None: "exact", 10000: "10 000", 1000: "1 000", 100: "100"}
    th_col = {0.5: COL["quantum_reservoir"], 0.95: COL["dtc"]}
    th_lab = {0.5: "θ = 0.5π (scrambling)", 0.95: "θ = 0.95π (time crystal)"}
    left, right = st.columns(2)
    with left:
        fs = go.Figure()
        for th in sorted({r["theta_over_pi"] for r in rows3}):
            R = {r["shots"]: r for r in rows3 if r["theta_over_pi"] == th and r["noise_p"] == 0.0}
            xs = [shot_lab[s] for s in shot_order if s in R]
            fs.add_trace(go.Scatter(x=xs, y=[np.mean(R[s]["MC"]) for s in shot_order if s in R], mode="lines+markers",
                                    name=th_lab.get(th, f"θ = {th:g}π"), line=dict(color=th_col.get(th), width=3),
                                    error_y=dict(type="data", array=[np.std(R[s]["MC"]) for s in shot_order if s in R],
                                                 thickness=1, width=3)))
        fs.update_layout(title=dict(text="Memory capacity vs shots per readout (no gate noise)", font=dict(size=14)),
                         xaxis_title="shots per time step", yaxis=dict(title="MC (mean ± sd, 5 seeds)", rangemode="tozero"),
                         height=400, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fs, width="stretch")
    with right:
        fn = go.Figure()
        for th in sorted({r["theta_over_pi"] for r in rows3}):
            for shots, dash in ((None, None), (1000, "dot")):
                R = sorted((r for r in rows3 if r["theta_over_pi"] == th and r["shots"] == shots), key=lambda r: r["noise_p"])
                if not R:
                    continue
                fn.add_trace(go.Scatter(x=[r["noise_p"] for r in R], y=[np.mean(r["MC"]) for r in R], mode="lines+markers",
                                        name=f"{th_lab.get(th, th)}, {shot_lab[shots]}",
                                        line=dict(color=th_col.get(th), dash=dash, width=2.5)))
        fn.update_layout(title=dict(text="Memory capacity vs depolarizing noise p", font=dict(size=14)),
                         xaxis=dict(title="depolarizing p per qubit per cycle", type="category"),
                         yaxis=dict(title="MC (mean, 5 seeds)", rangemode="tozero"), height=400,
                         margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25, x=0))
        st.plotly_chart(fn, width="stretch")
    sm = S["shots_MC"]
    st.markdown(
        f'<div class="interp">Finite shots erase the scrambling regime\'s memory advantage: at θ = 0.5π MC drops from '
        f"{sm['0.5']['exact']:.2f} (exact) to {sm['0.5']['1000']:.2f} at 1 000 shots and {sm['0.5']['100']:.2f} at 100 — "
        f"down to the time-crystal regime's level ({sm['0.95']['exact']:.2f} exact, {sm['0.95']['100']:.2f} at 100 shots). "
        "The time-crystal regime has little memory to lose. Small depolarizing noise slightly raises exact MC "
        "(dissipation helps fading memory), but does not survive shot noise.</div>", unsafe_allow_html=True)

    tbl = []
    for r in rows3:
        tbl.append({"θ / π": r["theta_over_pi"], "noise p": r["noise_p"], "shots": shot_lab.get(r["shots"], r["shots"]),
                    "MC mean": float(np.mean(r["MC"])), "MC sd": float(np.std(r["MC"])),
                    "sunspot test NMSE (median)": med(r["sunspot_test_nmse"])})
    with st.expander("All E3 rows (saved results)"):
        st.dataframe(pd.DataFrame(tbl), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("MC mean", "MC sd", "sunspot test NMSE (median)")})
        st.caption("Sunspot NMSE here uses a fixed θ (no validation selection), so it differs from the Forecasts tab.")

    st.subheader("What it would cost on hardware")
    hc = E3["hardware_cost"]
    T = E2["sunspots"]["n"]
    h1, h2, h3 = st.columns(3)
    h1.metric("Sunspot series length", f"{T}", border=True)
    h2.metric("Floquet cycles per full sweep", f"{hc['cycles_total_per_shot_sweep']:,}", border=True,
              help="Re-running the sequence up to every readout time: 2 cycles × (1 + 2 + … + T)")
    h3.metric("Circuit executions at 1 000 shots", f"{hc['circuit_runs_at_1000_shots']:,}", border=True,
              help="One circuit per readout time × 1 000 shots, for a single pass over the series")
    st.caption(hc["note"] + " Weak-measurement or windowed-replay protocols (Mujal et al., npj QI 2023) reduce this, "
               "at a cost in memory. Nothing here was run on hardware.")
    if (FIG / "robustness_figure.png").exists():
        with st.expander("Static robustness figure (from the report)"):
            st.image(str(FIG / "robustness_figure.png"), width="stretch")

# ================================================================ 5. Method & honesty
with tabs[4]:
    sp_s, sp_c = E2["sunspots"]["split"], E2["co2"]["split"]
    st.markdown(
        f"""
**What was simulated.** A 6-spin disordered kicked-Ising chain, exact density-matrix simulation in NumPy
(reservoir step verified against Qiskit Aer). Input enters by resetting qubit 0 each step (Fujii & Nakajima 2017);
readout is ⟨Z_i⟩ (6) and ⟨Z_i Z_{{i+1}}⟩ (5) = **11 features**. {SIM_TAG.capitalize()}.

**Matched baselines.** Every model gets **11 features** and the **same ridge readout** (with bias):
an 11-node echo-state network (tanh, sparse random weights), and linear AR on up to 11 lags. A persistence forecast
and a **J = 0 quantum control** (same chain, no interactions) are reported too.

**Splits (time-ordered, no shuffling).** Train 0–65 %, validation 65–80 %, test 80–100 %.
Sunspots: train {sp_s['train']}, val {sp_s['val']}, test {sp_s['test']} (index ranges of {E2['sunspots']['n']} years).
CO₂: train {sp_c['train']}, val {sp_c['val']}, test {sp_c['test']} ({E2['co2']['n']} months). Scaling uses the
training block's min and max only.

**Validation-only tuning.** θ (quantum), spectral radius and input scale (ESN), lag order p (AR) and ridge α (all)
are picked on the validation block; the model is then refit on train + validation and evaluated **once** on test.
Significance: Diebold–Mariano test vs AR on the test block. {len(seeds)} reservoir seeds per model.

**Memory capacity.** 1500 iid uniform inputs, 200 washout, 1000 train, 300 test; MC = Σ_k r²(u_(t−k)), k = 0…15.

**Where classical wins — stated plainly.** Linear AR is the best forecaster on both series; the ESN is second.
11 raw lags have about twice the linear memory of the best quantum setting. The time-crystal regime is the *worst*
memory regime of the interacting chain. No quantum advantage is claimed or shown.

**Limits.** 6 spins only; one J for forecasting; two series; the main forecast uses one test block per series
(the rolling-origin check on the Forecasts tab re-tests on earlier blocks, if present); a hackathon-scale literature scan.
"""
    )
    st.markdown("**Prior work and what is new here**")
    st.markdown(
        "The core idea is **not new**: Zhang et al., *Robust and Efficient Quantum Reservoir Computing with Discrete "
        "Time Crystal*, [arXiv:2508.15230](https://arxiv.org/abs/2508.15230) (2025), use essentially the same "
        "imperfectly π-kicked disordered Ising reservoir on memory, parity and NARMA10 tasks (plus hardware). "
        "This project's angle: **real time series** (sunspots, CO₂), **matched classical baselines** (ESN with the same "
        "feature count and readout, linear AR, a J = 0 control), and the **measurement cost** of reading out every step "
        "with finite shots."
    )
    lit = DOCS / "literature.md"
    if lit.exists():
        with st.expander("Full literature and novelty notes (docs/literature.md)"):
            st.markdown(lit.read_text())
    for name, cap in (("phase_figure", "Phase scan (static figure)"), ("forecast_figure", "Forecasts (static figure)")):
        p = FIG / f"{name}.png"
        if p.exists():
            with st.expander(cap):
                st.image(str(p), width="stretch")
    prov = E2["provenance"]
    st.caption(f"Provenance: Python {prov['python']}, NumPy {prov['numpy']}, statsmodels {prov['statsmodels']}, "
               f"Qiskit {prov['qiskit']} (used offline for verification and the circuit picture only), {prov['platform']}.")

st.divider()
st.caption("SIMULATION ONLY — every number on this page is a classical simulation on this laptop (NumPy density "
           "matrices). No quantum hardware and no IBM service were used. No quantum advantage is claimed.")
