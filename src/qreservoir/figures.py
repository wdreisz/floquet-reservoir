"""Figures + headline summary from saved artifacts only. Run: python -m qreservoir.figures"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
FOOT = "Classical simulation (NumPy density matrices, step checked against Qiskit Aer) — no quantum hardware used."
JCOL = {0.0: ("#9ca3af", "J = 0 (no interactions)"), 0.4: ("#d55e00", "J = 0.4"), 0.785: ("#7c3aed", "J = π/4")}
MODELS = {"ar": ("Linear AR (p lags)", "#009e73"), "esn": ("Echo-state network, 11 nodes", "#0072b2"),
          "quantum_reservoir": ("Quantum reservoir, 6 spins (J = 0.4)", "#d55e00"),
          "quantum_reservoir_J0": ("Quantum reservoir, J = 0", "#9ca3af")}


def _setup():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 190,
                         "legend.fontsize": 9, "axes.titlesize": 12})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.text(0.01, -0.01, FOOT, fontsize=8, color="#6b7280", va="top")
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _load(name):
    return json.loads((ART / name).read_text())


def _scan(e1, cycles=2):
    g = defaultdict(list)
    for r in e1["rows"]:
        if r["cycles_per_step"] == cycles:
            g[(r["J"], r["theta_over_pi"])].append(r)
    out = {}
    for J in sorted({k[0] for k in g}):
        th = sorted(t for j, t in g if j == J)
        rows = [g[(J, t)] for t in th]
        out[J] = {"theta": np.array(th), **{k: (np.array([np.mean([r[k] for r in R]) for R in rows]),
                                               np.array([np.std([r[k] for r in R], ddof=1) for R in rows]))
                                           for k in ("MC", "QC", "dtc_order")}}
    return out


def phase_figure():
    _setup()
    e1 = _load("e1_phase_scan.json")
    S = _scan(e1)
    esn = np.array([r["MC"] for r in e1["esn"]])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"hspace": 0.38, "wspace": 0.26})
    for ax, key, lab in ((axes[0, 0], "MC", "linear memory capacity MC (max 16)"), (axes[0, 1], "QC", "quadratic capacity QC")):
        for J, s in S.items():
            m, sd = s[key]
            col, name = JCOL[J]
            ax.plot(s["theta"], m, "-o", ms=4, color=col, label=name)
            ax.fill_between(s["theta"], m - sd, m + sd, color=col, alpha=0.15)
        ref = esn if key == "MC" else np.array([r["QC"] for r in e1["esn"]])
        ax.axhspan(ref.min(), ref.max(), color="#0072b2", alpha=0.12, label="echo-state network, 11 nodes (range)")
        ax.axhline(e1["lags11"][key], color="#009e73", ls="--", label="11 raw lags (linear AR features)")
        ax.axvspan(0.88, 1.0, color="#fde68a", alpha=0.5, lw=0)
        ax.text(0.94, ax.get_ylim()[1] * 0.92 if key == "MC" else 8.2, "time-\ncrystal", ha="center", fontsize=9)
        ax.set_xlabel("kick angle θ / π")
        ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0, 0].set_title("(a) Memory: scrambling kicks remember, time-crystal kicks forget")
    axes[0, 1].set_title("(b) Nonlinear (product) memory")
    ax = axes[1, 0]
    for J, s in S.items():
        if J == 0.0:
            continue
        col, name = JCOL[J]
        ax.scatter(s["dtc_order"][0], s["MC"][0], color=col, s=30, label=name)
        for t, x, y in zip(s["theta"], s["dtc_order"][0], s["MC"][0], strict=True):
            if t in (0.5, 0.8, 0.9, 0.95, 1.0):
                ax.annotate(f"{t:g}π", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color=col)
    ax.set_xlabel("time-crystal order (period-doubled magnetisation, no input)")
    ax.set_ylabel("memory capacity MC")
    ax.set_title("(c) More time-crystal order → less memory")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    for th, col in ((0.5, "#d55e00"), (0.95, "#f4a582")):
        R = [r for r in e1["rows"] if r["J"] == 0.4 and r["cycles_per_step"] == 2 and r["theta_over_pi"] == th]
        ax.plot(np.mean([r["linear_by_delay"] for r in R], axis=0), "-o", ms=4, color=col, label=f"quantum, θ = {th:g}π")
    R = [r for r in e1["esn"] if r["spectral_radius"] == 0.9]
    ax.plot(np.mean([r["linear_by_delay"] for r in R], axis=0), "-s", ms=4, color="#0072b2", label="echo-state network ρ = 0.9")
    ax.set_xlabel("delay k (steps into the past)")
    ax.set_ylabel("r² of recalling u(t − k)")
    ax.set_title("(d) Memory profile by delay")
    ax.legend(frameon=False)
    _save(fig, "phase_figure")


def forecast_figure():
    _setup()
    e2 = _load("e2_forecast.json")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    x = e2["sunspots"]
    lo, hi = x["scale"]
    yrs = np.array(x["test_years"]) + 1
    un = lambda v: np.array(v) * (hi - lo) + lo  # noqa: E731
    q = sorted([r for r in x["runs"] if r["model"] == "quantum_reservoir"], key=lambda r: r["test_nmse"])
    qm = q[len(q) // 2]
    es = sorted([r for r in x["runs"] if r["model"] == "esn"], key=lambda r: r["test_nmse"])
    em = es[len(es) // 2]
    ax.plot(yrs, un(qm["true"]), color="#111827", lw=2.2, label="observed")
    ax.plot(yrs, un(x["ar"]["pred"]), color=MODELS["ar"][1], label=f"AR({x['ar']['p']})  NMSE {x['ar']['test_nmse']:.2f}")
    ax.plot(yrs, un(em["pred"]), color=MODELS["esn"][1], label=f"echo-state net (median seed)  NMSE {em['test_nmse']:.2f}")
    ax.plot(yrs, un(qm["pred"]), color=MODELS["quantum_reservoir"][1], label=f"quantum reservoir (median seed)  NMSE {qm['test_nmse']:.2f}")
    ax.set_xlabel("year")
    ax.set_ylabel("sunspot number")
    ax.set_title("(a) Held-out sunspot years (last 20 %), one-step-ahead forecasts")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax = fig.add_subplot(gs[1, 0])
    names = ["ar", "esn", "quantum_reservoir", "quantum_reservoir_J0"]
    for i, series in enumerate(("sunspots", "co2")):
        x = e2[series]
        for j, m in enumerate(names):
            pos = j + i * (len(names) + 1.5)
            col = MODELS[m][1]
            if m == "ar":
                ax.scatter([pos], [x["ar"]["test_nmse"]], color=col, marker="D", s=50, zorder=3)
            else:
                v = [r["test_nmse"] for r in x["runs"] if r["model"] == m]
                ax.scatter(pos + np.linspace(-0.2, 0.2, len(v)), v, color=col, s=22, alpha=0.8)
                ax.plot([pos - 0.3, pos + 0.3], [np.median(v)] * 2, color=col, lw=2.5)
        ax.plot([-0.5 + i * (len(names) + 1.5), len(names) - 0.5 + i * (len(names) + 1.5)], [x["persistence_test_nmse"]] * 2,
                color="#6b7280", ls=":", lw=1.2)
        ax.text(i * (len(names) + 1.5) + 1.5, -0.12, "sunspots (yearly)" if series == "sunspots" else "CO₂ (monthly change)",
                ha="center", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("test NMSE (lower is better)")
    ax.set_title("(b) Every seed, both series (dotted: persistence)")
    for m in names:
        ax.scatter([], [], color=MODELS[m][1], label=MODELS[m][0])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[1, 1])
    for series, ls in (("sunspots", "-"), ("co2", "--")):
        R = [r for r in e2[series]["runs"] if r["model"] == "quantum_reservoir"]
        th = sorted(R[0]["val_by_theta"], key=float)
        ax.plot([float(t) for t in th], [np.median([r["val_by_theta"][t] for r in R]) for t in th], ls, marker="o", color="#6b7280",
                label=f"{series}: validation")
        ax.plot([float(t) for t in th], [np.median([r["test_by_theta"][t] for r in R]) for t in th], ls, marker="o", color="#d55e00",
                label=f"{series}: test")
    ax.set_xlabel("kick angle θ / π (J = 0.4)")
    ax.set_ylabel("median NMSE over 8 disorder seeds")
    ax.set_title("(c) Which phase forecasts best?")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "forecast_figure")


def robustness_figure():
    _setup()
    e3 = _load("e3_shots_noise.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for th, col in ((0.5, "#d55e00"), (0.95, "#f4a582")):
        R = [r for r in e3["rows"] if r["theta_over_pi"] == th and r["noise_p"] == 0]
        labels = ["exact" if r["shots"] is None else f"{r['shots']:,}" for r in R]
        axes[0].errorbar(range(len(R)), [np.mean(r["MC"]) for r in R], yerr=[np.std(r["MC"], ddof=1) for r in R], fmt="-o",
                         color=col, capsize=3, label=f"θ = {th:g}π")
        axes[0].set_xticks(range(len(R)), labels)
        R = [r for r in e3["rows"] if r["theta_over_pi"] == th and r["shots"] is None]
        axes[1].errorbar([r["noise_p"] for r in R], [np.mean(r["MC"]) for r in R], yerr=[np.std(r["MC"], ddof=1) for r in R],
                         fmt="-o", color=col, capsize=3, label=f"θ = {th:g}π, exact readout")
        R = [r for r in e3["rows"] if r["theta_over_pi"] == th and r["shots"] == 1000]
        axes[1].errorbar([r["noise_p"] for r in R], [np.mean(r["MC"]) for r in R], yerr=[np.std(r["MC"], ddof=1) for r in R],
                         fmt="--s", color=col, capsize=3, alpha=0.7, label=f"θ = {th:g}π, 1,000 shots")
    axes[0].set_xlabel("shots per readout")
    axes[0].set_ylabel("memory capacity MC (J = 0.4)")
    axes[0].set_title("(e) Finite measurement shots erase the scrambling advantage")
    axes[0].legend(frameon=False)
    axes[1].set_xscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("depolarizing probability per qubit per cycle")
    axes[1].set_ylabel("memory capacity MC")
    axes[1].set_title("(f) Synthetic gate noise")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "robustness_figure")


def cv_figure():
    _setup()
    e4 = _load("e4_rolling_origin.json")
    names = {"ar": MODELS["ar"], "esn": MODELS["esn"], "quantum_reservoir": MODELS["quantum_reservoir"],
             "quantum_reservoir_J0": MODELS["quantum_reservoir_J0"], "persistence": ("Persistence (tomorrow = today)", "#111827")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, series in zip(axes, ("sunspots", "co2"), strict=True):
        for m, (lab, col) in names.items():
            y = [np.median([r["test_nmse"] for r in e4["rows"] if r["series"] == series and r["fold"] == f and r["model"] == m])
                 for f in range(len(e4["folds"]))]
            ax.plot(range(1, len(y) + 1), y, marker="o", color=col, label=lab, ls=":" if m == "persistence" else "-")
        ax.set_xticks(range(1, len(e4["folds"]) + 1), [f"test {int(100 * f[1])}–{int(100 * f[2])} %" for f in e4["folds"]])
        ax.set_ylabel("test NMSE (median over seeds)")
        ax.set_title(f"({'g' if series == 'sunspots' else 'h'}) Rolling-origin CV: {'sunspots' if series == 'sunspots' else 'CO₂ monthly change'}")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "cv_figure")


def circuit_figure():
    """One input step of the reservoir as a Qiskit circuit (drawn here, never inside the app)."""
    from .circuit import step_circuit

    from .reservoir import FloquetReservoir

    h = np.round(FloquetReservoir(n=6, seed=0).h, 2)  # disorder draw of seed 0
    qc = step_circuit(n=6, u=0.3, theta=0.95 * np.pi, J=0.4, h=h, cycles=1)
    fig = qc.draw("mpl", fold=-1, style="clifford")
    fig.suptitle("One input step: reset + encode u on qubit 0, then one Floquet cycle (kick Rx(θ), Ising ZZ, disorder Z)", fontsize=11)
    _save(fig, "circuit")


def summary():
    e1, e2, e3 = _load("e1_phase_scan.json"), _load("e2_forecast.json"), _load("e3_shots_noise.json")
    S = _scan(e1)
    s4 = S[0.4]
    i_best = int(np.argmax(s4["MC"][0]))
    i_dtc = list(s4["theta"]).index(0.95)
    out = {"memory": {"best_theta_over_pi": float(s4["theta"][i_best]), "best_MC": float(s4["MC"][0][i_best]),
                      "dtc_theta_over_pi": 0.95, "dtc_MC": float(s4["MC"][0][i_dtc]), "dtc_order_at_0.95": float(s4["dtc_order"][0][i_dtc]),
                      "J0_MC_max": float(S[0.0]["MC"][0].max()),
                      "esn_MC_range": [float(min(r["MC"] for r in e1["esn"])), float(max(r["MC"] for r in e1["esn"]))],
                      "lags11_MC": e1["lags11"]["MC"],
                      "corr_MC_vs_dtc_order": float(np.corrcoef(np.concatenate([S[J]["dtc_order"][0] for J in (0.4, 0.785)]),
                                                                np.concatenate([S[J]["MC"][0] for J in (0.4, 0.785)]))[0, 1])},
           "forecast": {}}
    for series in ("sunspots", "co2"):
        x = e2[series]
        d = {"ar_p": x["ar"]["p"], "ar_test_nmse": x["ar"]["test_nmse"], "persistence_test_nmse": x["persistence_test_nmse"]}
        for m in ("quantum_reservoir", "quantum_reservoir_J0", "esn"):
            R = [r for r in x["runs"] if r["model"] == m]
            v = [r["test_nmse"] for r in R]
            d[m] = {"median_test_nmse": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n_seeds": len(v),
                    "seeds_where_AR_significantly_better_p05": int(sum(r["dm_vs_ar"]["p_value"] < 0.05 and r["dm_vs_ar"]["mean_loss_diff"] > 0 for r in R)),
                    "seeds_beating_AR": int(sum(t < x["ar"]["test_nmse"] for t in v))}
        R = [r for r in x["runs"] if r["model"] == "quantum_reservoir"]
        d["quantum_reservoir"]["seeds_where_ESN_significantly_better_p05"] = int(sum(r["dm_vs_esn"]["p_value"] < 0.05 and r["dm_vs_esn"]["mean_loss_diff"] > 0 for r in R))
        d["quantum_reservoir"]["chosen_theta_over_pi"] = [r["chosen"]["theta_over_pi"] for r in R]
        d["quantum_median_test_by_theta"] = {t: float(np.median([r["test_by_theta"][t] for r in R])) for t in R[0]["test_by_theta"]}
        d["quantum_median_val_by_theta"] = {t: float(np.median([r["val_by_theta"][t] for r in R])) for t in R[0]["val_by_theta"]}
        out["forecast"][series] = d
    rob = {}
    for th in (0.5, 0.95):
        rob[str(th)] = {("exact" if r["shots"] is None else str(r["shots"])): float(np.mean(r["MC"]))
                        for r in e3["rows"] if r["theta_over_pi"] == th and r["noise_p"] == 0}
    out["shots_MC"] = rob
    out["hardware_cost"] = e3["hardware_cost"]
    if (ART / "e4_rolling_origin.json").exists():
        e4 = _load("e4_rolling_origin.json")
        ro = {}
        for series in ("sunspots", "co2"):
            per = defaultdict(dict)  # fold -> model -> median NMSE over seeds
            for f in range(len(e4["folds"])):
                for m in e4["models"]:
                    per[f][m] = float(np.median([r["test_nmse"] for r in e4["rows"]
                                                 if r["series"] == series and r["fold"] == f and r["model"] == m]))
            ranks = {m: [] for m in e4["models"]}
            for f, d in per.items():
                for k, m in enumerate(sorted(d, key=d.get)):
                    ranks[m].append(k + 1)
            ro[series] = {"folds": e4["folds"], "per_fold_median_test_nmse": {str(f): d for f, d in per.items()},
                          "median_test_nmse": {m: float(np.median([per[f][m] for f in per])) for m in e4["models"]},
                          "mean_rank": {m: float(np.mean(v)) for m, v in ranks.items()},
                          "wins": {m: int(sum(min(d, key=d.get) == m for d in per.values())) for m in e4["models"]}}
        out["rolling_origin"] = ro
    out["statement"] = e1["provenance"]["statement"]
    (ART / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    phase_figure()
    forecast_figure()
    robustness_figure()
    if (ART / "e4_rolling_origin.json").exists():
        cv_figure()
    circuit_figure()
    print(json.dumps(summary(), indent=1))
