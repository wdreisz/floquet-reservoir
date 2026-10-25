"""Quantum reservoir = disordered kicked-Ising spin chain (the FloquetForge model), simulated exactly
with density matrices; plus a classical echo-state network with the same number of readout features.

Per input step (Fujii & Nakajima 2017 style injection):
  1. reset qubit 0 into |psi(u)> = sqrt(1-u)|0> + sqrt(u)|1>   (u in [0, 1]); other qubits untouched
  2. apply V Floquet cycles  U_F = exp(-i sum_i h_i Z_i) exp(-i J sum_<ij> Z_i Z_j) prod_i R_x(theta)
  3. (optional) local depolarizing noise p per qubit per cycle
  4. read out <Z_i> (N values) and <Z_i Z_{i+1}> (N-1 values)

theta = pi is a perfect spin flip; theta = pi (1 - eps) with small eps and J != 0 is the time-crystal (period-doubled)
regime; theta ~ pi/2 with J ~ pi/4 is the strongly scrambling (chaotic) kicked-Ising regime; small theta barely mixes.
Bit convention: basis index k has qubit i = (k >> i) & 1 (Qiskit), z_i = 1 - 2 bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_QUBITS = 8


def _z_table(n: int) -> np.ndarray:
    k = np.arange(2**n)[:, None]
    return (1 - 2 * ((k >> np.arange(n)[None, :]) & 1)).astype(float)


@dataclass
class FloquetReservoir:
    n: int = 6
    theta: float = np.pi * 0.95
    J: float = 0.4
    W: float = np.pi
    cycles_per_step: int = 2
    noise_p: float = 0.0
    seed: int = 0
    h: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 2 <= self.n <= MAX_QUBITS:
            raise ValueError(f"n must be 2..{MAX_QUBITS}")
        rng = np.random.default_rng(self.seed)
        self.h = rng.uniform(-self.W, self.W, self.n)
        n = self.n
        z = _z_table(n)
        self.z = z
        zz = sum(z[:, i] * z[:, i + 1] for i in range(n - 1))
        diag = np.exp(-1j * (z @ self.h + self.J * zz))
        c, s = np.cos(self.theta / 2), np.sin(self.theta / 2)
        rx = np.array([[c, -1j * s], [-1j * s, c]])
        R = np.array([[1.0]])
        for _ in range(n):
            R = np.kron(rx, R)
        self.U = diag[:, None] * R  # U_F = D . R
        self.n_features = 2 * n - 1

    def input_state(self, u: float) -> np.ndarray:
        a, b = np.sqrt(max(0.0, 1 - u)), np.sqrt(min(1.0, max(0.0, u)))
        return np.array([[a * a, a * b], [a * b, b * b]], dtype=complex)

    def _reset_q0(self, rho: np.ndarray, u: float) -> np.ndarray:
        d = 2 ** (self.n - 1)
        r = rho.reshape(d, 2, d, 2)
        rest = np.einsum("aibi->ab", r)  # trace out qubit 0 (least significant)
        return np.kron(rest, self.input_state(u))

    def _depolarize(self, rho: np.ndarray) -> np.ndarray:
        p, n = self.noise_p, self.n
        for q in range(n):
            t = rho.reshape((2,) * (2 * n))
            ax, axp = n - 1 - q, 2 * n - 1 - q
            red = np.trace(t, axis1=ax, axis2=axp)  # trace out qubit q
            mixed = np.multiply.outer(red, np.eye(2) / 2)
            # move the new (q, q') axes to their places
            mixed = np.moveaxis(mixed, (2 * n - 2, 2 * n - 1), (ax, axp))
            rho = ((1 - p) * t + p * mixed).reshape(2**n, 2**n)
        return rho

    def features_from_probs(self, probs: np.ndarray) -> np.ndarray:
        zi = probs @ self.z
        zz = np.array([probs @ (self.z[:, i] * self.z[:, i + 1]) for i in range(self.n - 1)])
        return np.concatenate([zi, zz])

    def run(self, u: np.ndarray, shots: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Feature matrix (T, 2n-1). With shots, each step's features are estimated from that many bitstrings
        sampled from the exact measurement distribution (the state itself is not collapsed: this models
        re-running the sequence up to every readout time)."""
        rho = np.zeros((2**self.n, 2**self.n), dtype=complex)
        rho[0, 0] = 1.0
        out = np.empty((len(u), self.n_features))
        Ud = self.U.conj().T
        rng = rng or np.random.default_rng(self.seed + 1)
        for t, ut in enumerate(u):
            rho = self._reset_q0(rho, float(ut))
            for _ in range(self.cycles_per_step):
                rho = self.U @ rho @ Ud
                if self.noise_p > 0:
                    rho = self._depolarize(rho)
            probs = np.clip(np.real(np.diag(rho)), 0, None)
            probs /= probs.sum()
            if shots:
                counts = rng.multinomial(shots, probs)
                probs = counts / shots
            out[t] = self.features_from_probs(probs)
        return out


@dataclass
class EchoStateNetwork:
    """Classical tanh reservoir with n_nodes features (matched to the quantum readout size)."""

    n_nodes: int = 11
    spectral_radius: float = 0.9
    input_scale: float = 0.5
    leak: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        W = rng.normal(size=(self.n_nodes, self.n_nodes)) * (rng.uniform(size=(self.n_nodes, self.n_nodes)) < 0.5)
        rad = max(abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / rad if rad > 0 else 1.0)
        self.w_in = rng.uniform(-1, 1, self.n_nodes) * self.input_scale
        self.b = rng.uniform(-0.2, 0.2, self.n_nodes)
        self.n_features = self.n_nodes

    def run(self, u: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n_nodes)
        out = np.empty((len(u), self.n_nodes))
        for t, ut in enumerate(u):
            x = (1 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.w_in * (2 * ut - 1) + self.b)
            out[t] = x
        return out


def lagged(u: np.ndarray, p: int) -> np.ndarray:
    """Features [u_t, u_{t-1}, ..., u_{t-p+1}] (zeros before the start)."""
    T = len(u)
    out = np.zeros((T, p))
    for k in range(p):
        out[k:, k] = u[: T - k]
    return out
