"""Benchmarks: memory capacity (synthetic, standard), and real time-series forecasting with time-ordered splits.

All models share the same readout (ridge regression with bias), the same number of features
(2N-1 = 11 for N = 6 spins), the same data splits, and hyperparameters chosen on a validation block that lies
strictly before the test block in time.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .reservoir import EchoStateNetwork, FloquetReservoir, lagged

ALPHAS = np.logspace(-8, 0, 9)


def ridge_fit(F: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    A = np.hstack([F, np.ones((len(F), 1))])
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)


def ridge_predict(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([F, np.ones((len(F), 1))]) @ w


def r2(y, yhat) -> float:
    c = np.corrcoef(y, yhat)[0, 1]
    return float(c**2) if np.isfinite(c) else 0.0


# ----------------------------------------------------------------------------- memory capacity

def capacities(F: np.ndarray, u: np.ndarray, washout: int = 200, n_train: int = 1000, max_delay: int = 15, alpha: float = 1e-6) -> dict:
    """Linear short-term memory MC = sum_k r^2(u_{t-k}) and quadratic capacity QC = sum_k r^2(u_{t-k} u_{t-k-1}),
    each r^2 measured on held-out steps after fitting the readout on n_train steps."""
    tr = slice(washout, washout + n_train)
    te = slice(washout + n_train, len(u))
    lin, quad = [], []
    for k in range(max_delay + 1):
        y = np.roll(u, k)
        w = ridge_fit(F[tr], y[tr], alpha)
        lin.append(r2(y[te], ridge_predict(F[te], w)))
        yq = np.roll(u, k) * np.roll(u, k + 1)
        w = ridge_fit(F[tr], yq[tr], alpha)
        quad.append(r2(yq[te], ridge_predict(F[te], w)))
    return {"linear_by_delay": lin, "quadratic_by_delay": quad, "MC": float(sum(lin)), "QC": float(sum(quad))}


# ----------------------------------------------------------------------------- forecasting

def load_series(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    import statsmodels.api as sm

    if name == "sunspots":
        d = sm.datasets.sunspots.load_pandas().data
        return d["YEAR"].to_numpy(), d["SUNACTIVITY"].to_numpy(float), "Yearly sunspot number 1700–2008 (309 values, statsmodels)"
    if name == "co2":
        d = sm.datasets.co2.load_pandas().data.resample("MS").mean().interpolate()
        y = d["co2"].to_numpy(float)
        y = np.diff(y)  # month-to-month change (removes the trend; keeps the seasonal cycle)
        return np.arange(len(y)), y, "Mauna Loa CO2, monthly change 1958–2001 (525 values, statsmodels)"
    raise ValueError(name)


def splits(T: int) -> dict:
    """Time-ordered: train 0–65 %, validation 65–80 %, test 80–100 %."""
    a, b = int(0.65 * T), int(0.80 * T)
    return {"train": slice(0, a), "val": slice(a, b), "test": slice(b, T), "fit_final": slice(0, b)}


def scale(y: np.ndarray, sp: dict) -> tuple[np.ndarray, tuple[float, float]]:
    """Min-max scale with train-block statistics only. Not clipped: later values may leave [0, 1]
    (the quantum encoder clamps its input angle to [0, 1]; targets stay unaltered)."""
    lo, hi = y[sp["train"]].min(), y[sp["train"]].max()
    return (y - lo) / (hi - lo), (lo, hi)


def nmse(y, yhat) -> float:
    return float(np.mean((y - yhat) ** 2) / np.var(y))


def one_step(Fz: np.ndarray, u: np.ndarray, sp: dict, washout: int = 10) -> dict:
    """Predict u_{t+1} from features at t. alpha tuned on validation, then refit on train+val, test once."""
    y = np.roll(u, -1)
    tr = slice(washout, sp["train"].stop)
    best = min(ALPHAS, key=lambda a: nmse(y[sp["val"]], ridge_predict(Fz[sp["val"]], ridge_fit(Fz[tr], y[tr], a))))
    val = nmse(y[sp["val"]], ridge_predict(Fz[sp["val"]], ridge_fit(Fz[tr], y[tr], best)))
    fin = slice(washout, sp["fit_final"].stop)
    te = slice(sp["test"].start, sp["test"].stop - 1)  # last step has no target
    pred = ridge_predict(Fz[te], ridge_fit(Fz[fin], y[fin], best))
    return {"alpha": float(best), "val_nmse": val, "test_nmse": nmse(y[te], pred), "test_pred": pred, "test_true": y[te]}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    """DM test (squared-error loss, horizon 1) for equal predictive accuracy; positive mean => model 2 better."""
    d = e1**2 - e2**2
    n = len(d)
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std(ddof=1) > 0 else 0.0
    return {"mean_loss_diff": float(d.mean()), "dm_stat": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), n - 1)), "n": n}


# ----------------------------------------------------------------------------- model families

def quantum_features(u, theta, J, seed, n=6, cycles=2, noise_p=0.0, shots=None):
    return FloquetReservoir(n=n, theta=theta, J=J, cycles_per_step=cycles, noise_p=noise_p, seed=seed).run(u, shots=shots)


def esn_features(u, rho, a, seed, n_nodes=11):
    return EchoStateNetwork(n_nodes=n_nodes, spectral_radius=rho, input_scale=a, seed=seed).run(u)


def ar_features(u, p):
    return lagged(u, p)
