"""All experiments (local classical simulation only). Run: python -m qreservoir.experiments"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .reservoir import FloquetReservoir
from .tasks import (ar_features, capacities, diebold_mariano, esn_features, load_series, nmse, one_step,
                    quantum_features, scale, splits)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
STATEMENT = "All results: classical simulation on a laptop (NumPy density matrices; reservoir step verified against Qiskit Aer). No quantum hardware used."
THETAS = np.round(np.linspace(0.05, 1.0, 20), 3)
SEEDS = tuple(range(8))


def provenance() -> dict:
    import qiskit
    import statsmodels

    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
            "qiskit": qiskit.__version__, "statsmodels": statsmodels.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "statement": STATEMENT}


def save(name: str, obj: dict) -> None:
    ART.mkdir(exist_ok=True)

    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (ART / name).write_text(json.dumps({**obj, "provenance": provenance()}, indent=1, default=conv))


def dtc_order(theta: float, J: float, seed: int, cycles: int = 40) -> float:
    """Period-doubling score of the same chain without input: mean over cycles 21-40 of (-1)^t m_z(t) from all-up."""
    r = FloquetReservoir(theta=theta, J=J, seed=seed)
    psi = np.zeros(2**r.n, dtype=complex)
    psi[0] = 1
    s = []
    for t in range(1, cycles + 1):
        psi = r.U @ psi
        if t > cycles // 2:
            s.append((-1) ** t * float(np.abs(psi) ** 2 @ r.z.mean(axis=1)))
    return float(np.mean(s))


def _cap_job(args):
    th, J, cycles, seed = args
    u = np.random.default_rng(100 + seed).uniform(size=1500)
    F = quantum_features(u, np.pi * th, J, seed, cycles=cycles)
    c = capacities(F, u)
    return {"theta_over_pi": th, "J": J, "cycles_per_step": cycles, "seed": seed, "MC": c["MC"], "QC": c["QC"],
            "linear_by_delay": c["linear_by_delay"], "dtc_order": dtc_order(np.pi * th, J, seed)}


def e1_phase_scan() -> None:
    jobs = [(th, J, cyc, s) for th in THETAS for J in (0.0, 0.4, 0.785) for cyc in (1, 2) for s in SEEDS[:5]]
    with ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_cap_job, jobs, chunksize=8))
    esn = []
    for rho in (0.5, 0.9, 1.1):
        for s in SEEDS[:5]:
            u = np.random.default_rng(100 + s).uniform(size=1500)
            c = capacities(esn_features(u, rho, 0.5, s), u)
            esn.append({"spectral_radius": rho, "seed": s, "MC": c["MC"], "QC": c["QC"], "linear_by_delay": c["linear_by_delay"]})
    u = np.random.default_rng(100).uniform(size=1500)
    lag = capacities(ar_features(u, 11), u)
    save("e1_phase_scan.json", {"experiment": "E1 memory capacity vs Floquet kick angle (6 spins, 11 features, iid input)",
                                "rows": rows, "esn": esn, "lags11": {"MC": lag["MC"], "QC": lag["QC"]}})
    print("E1 done", flush=True)


def _forecast_job(args):
    series, kind, seed = args
    _, y, _ = load_series(series)
    sp = splits(len(y))
    z, _ = scale(y, sp)
    if kind == "quantum":
        cands = [("theta", th) for th in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
        res = {th: one_step(quantum_features(z, np.pi * th, 0.4, seed), z, sp) for _, th in cands}
        best = min(res, key=lambda k: res[k]["val_nmse"])
        r = res[best]
        return {"model": "quantum_reservoir", "seed": seed, "chosen": {"theta_over_pi": best}, **{k: r[k] for k in ("val_nmse", "test_nmse", "alpha")},
                "val_by_theta": {str(k): v["val_nmse"] for k, v in res.items()}, "test_by_theta": {str(k): v["test_nmse"] for k, v in res.items()},
                "pred": r["test_pred"], "true": r["test_true"]}
    if kind == "quantum_J0":
        res = {th: one_step(quantum_features(z, np.pi * th, 0.0, seed), z, sp) for th in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)}
        best = min(res, key=lambda k: res[k]["val_nmse"])
        r = res[best]
        return {"model": "quantum_reservoir_J0", "seed": seed, "chosen": {"theta_over_pi": best}, "val_nmse": r["val_nmse"],
                "test_nmse": r["test_nmse"], "pred": r["test_pred"], "true": r["test_true"]}
    if kind == "esn":
        res = {(rho, a): one_step(esn_features(z, rho, a, seed), z, sp) for rho in (0.5, 0.9, 1.1) for a in (0.1, 0.3, 1.0)}
        best = min(res, key=lambda k: res[k]["val_nmse"])
        r = res[best]
        return {"model": "esn", "seed": seed, "chosen": {"spectral_radius": best[0], "input_scale": best[1]}, "val_nmse": r["val_nmse"],
                "test_nmse": r["test_nmse"], "pred": r["test_pred"], "true": r["test_true"]}
    raise ValueError(kind)


def e2_forecast() -> None:
    out = {}
    for series in ("sunspots", "co2"):
        years, y, desc = load_series(series)
        sp = splits(len(y))
        z, (lo, hi) = scale(y, sp)
        jobs = [(series, k, s) for k in ("quantum", "quantum_J0", "esn") for s in SEEDS]
        with ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context("spawn")) as ex:
            runs = list(ex.map(_forecast_job, jobs))
        ar = {p: one_step(ar_features(z, p), z, sp) for p in range(1, 12)}
        p_best = min(ar, key=lambda p: ar[p]["val_nmse"])
        arr = ar[p_best]
        te = slice(sp["test"].start, sp["test"].stop - 1)
        persist = nmse(np.roll(z, -1)[te], z[te])
        # Diebold-Mariano vs AR for every seed
        e_ar = arr["test_true"] - arr["test_pred"]
        for r in runs:
            e = np.array(r["true"]) - np.array(r["pred"])
            r["dm_vs_ar"] = diebold_mariano(e, e_ar)  # positive mean => AR better
        q = [r for r in runs if r["model"] == "quantum_reservoir"]
        es = [r for r in runs if r["model"] == "esn"]
        for rq, re_ in zip(sorted(q, key=lambda r: r["seed"]), sorted(es, key=lambda r: r["seed"]), strict=True):
            rq["dm_vs_esn"] = diebold_mariano(np.array(rq["true"]) - np.array(rq["pred"]), np.array(re_["true"]) - np.array(re_["pred"]))
        out[series] = {"description": desc, "n": int(len(y)), "split": {k: [v.start, v.stop] for k, v in sp.items()},
                       "scale": [float(lo), float(hi)], "test_years": years[te].tolist(), "runs": runs,
                       "ar": {"p": int(p_best), "val_nmse": arr["val_nmse"], "test_nmse": arr["test_nmse"], "pred": arr["test_pred"],
                              "val_by_p": {str(p): ar[p]["val_nmse"] for p in ar}}, "persistence_test_nmse": persist}
        print("E2", series, "AR", round(arr["test_nmse"], 3), "persist", round(persist, 3),
              {m: round(float(np.median([r["test_nmse"] for r in runs if r["model"] == m])), 3) for m in ("quantum_reservoir", "quantum_reservoir_J0", "esn")}, flush=True)
    save("e2_forecast.json", {"experiment": "E2 one-step forecasting, time-ordered train/val/test (65/15/20 %), same readout and feature count", **out})


def e3_shots_noise() -> None:
    _, y, _ = load_series("sunspots")
    sp = splits(len(y))
    z, _ = scale(y, sp)
    rows = []
    for th in (0.5, 0.95):
        for noise in (0.0, 1e-3, 1e-2, 3e-2):
            for shots in (None, 10_000, 1_000, 100):
                if noise > 0 and shots is not None and shots != 1_000:
                    continue
                vals, mcs = [], []
                for s in SEEDS[:5]:
                    F = quantum_features(z, np.pi * th, 0.4, s, noise_p=noise, shots=shots)
                    vals.append(one_step(F, z, sp)["test_nmse"])
                    u = np.random.default_rng(100 + s).uniform(size=1500)
                    mcs.append(capacities(quantum_features(u, np.pi * th, 0.4, s, noise_p=noise, shots=shots), u)["MC"])
                rows.append({"theta_over_pi": th, "noise_p": noise, "shots": shots, "sunspot_test_nmse": vals, "MC": mcs})
                print("E3", th, noise, shots, round(float(np.median(vals)), 3), round(float(np.mean(mcs)), 2), flush=True)
    T = 309
    cost = {"note": "Reading out every step on hardware destroys the state, so the sequence is re-run up to each readout time.",
            "cycles_total_per_shot_sweep": int(2 * T * (T + 1) / 2), "circuit_runs_at_1000_shots": int(T * 1000)}
    save("e3_shots_noise.json", {"experiment": "E3 shot noise and synthetic depolarizing noise (per qubit per cycle)", "rows": rows, "hardware_cost": cost})


ROLL_FOLDS = ((0.5, 0.6, 0.7), (0.6, 0.7, 0.8), (0.7, 0.8, 0.9), (0.8, 0.9, 1.0))


def _roll_job(args):
    series, f, kind, seed = args
    _, y, _ = load_series(series)
    T = len(y)
    a, b, c = (int(x * T) for x in ROLL_FOLDS[f])
    sp = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, c), "fit_final": slice(0, b)}
    z, _ = scale(y, sp)
    if kind in ("quantum_reservoir", "quantum_reservoir_J0"):
        J = 0.4 if kind == "quantum_reservoir" else 0.0
        res = [one_step(quantum_features(z, np.pi * th, J, seed), z, sp) for th in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    elif kind == "esn":
        res = [one_step(esn_features(z, rho, s, seed), z, sp) for rho in (0.5, 0.9, 1.1) for s in (0.1, 0.3, 1.0)]
    elif kind == "ar":
        res = [one_step(ar_features(z, p), z, sp) for p in range(1, 12)]
    else:
        te = slice(b, c - 1)
        return {"series": series, "fold": f, "model": kind, "seed": seed, "test_nmse": nmse(np.roll(z, -1)[te], z[te])}
    best = min(res, key=lambda r: r["val_nmse"])
    return {"series": series, "fold": f, "model": kind, "seed": seed, "test_nmse": best["test_nmse"]}


def e4_rolling_origin() -> None:
    """Rolling-origin cross-validation: 4 folds, each with its own train / validation / test blocks in time order."""
    models = ("ar", "persistence", "esn", "quantum_reservoir", "quantum_reservoir_J0")
    jobs = [(s, f, m, seed) for s in ("sunspots", "co2") for f in range(len(ROLL_FOLDS)) for m in models
            for seed in (SEEDS[:4] if m in ("esn", "quantum_reservoir", "quantum_reservoir_J0") else (0,))]
    with ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_roll_job, jobs, chunksize=2))
    save("e4_rolling_origin.json", {"experiment": "E4 rolling-origin cross-validation (fractions train_end, val_end, test_end per fold)",
                                    "folds": ROLL_FOLDS, "models": models, "rows": rows})
    print("E4 done", flush=True)


def main() -> None:
    t0 = time.time()
    e1_phase_scan()
    print(round(time.time() - t0), "s", flush=True)
    e2_forecast()
    print(round(time.time() - t0), "s", flush=True)
    e4_rolling_origin()
    e3_shots_noise()
    print("all done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
