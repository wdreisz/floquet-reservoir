"""The same reservoir step written as a Qiskit circuit (used for the cross-check test and the circuit figure).

Never import this from the Streamlit app: building Qiskit circuits off the main thread can crash Python.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def step_circuit(n: int, u: float, theta: float, J: float, h: np.ndarray, cycles: int, qc: QuantumCircuit | None = None) -> QuantumCircuit:
    """Append one input step: reset qubit 0, encode sqrt(1-u)|0> + sqrt(u)|1>, then `cycles` Floquet cycles
    U_F = exp(-i sum h_i Z_i) exp(-i J sum Z_i Z_{i+1}) prod R_x(theta) (rightmost first)."""
    qc = qc if qc is not None else QuantumCircuit(n)
    qc.reset(0)
    qc.ry(2 * np.arcsin(np.sqrt(np.clip(u, 0, 1))), 0)
    for _ in range(cycles):
        qc.barrier()
        for q in range(n):
            qc.rx(theta, q)
        for q in range(n - 1):
            qc.rzz(2 * J, q, q + 1)
        for q in range(n):
            qc.rz(2 * h[q], q)
    return qc
