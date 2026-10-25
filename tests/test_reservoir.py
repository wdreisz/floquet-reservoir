import numpy as np
import pytest

from qreservoir.reservoir import EchoStateNetwork, FloquetReservoir, lagged
from qreservoir.tasks import capacities, diebold_mariano, load_series, one_step, scale, splits


def _rho_after(r, u):
    rho = np.zeros((2**r.n, 2**r.n), dtype=complex)
    rho[0, 0] = 1
    for x in u:
        rho = r._reset_q0(rho, x)
        for _ in range(r.cycles_per_step):
            rho = r.U @ rho @ r.U.conj().T
            if r.noise_p:
                rho = r._depolarize(rho)
    return rho


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_density_matrix_stays_physical(noise):
    r = FloquetReservoir(n=4, theta=0.6 * np.pi, J=0.4, noise_p=noise, seed=3)
    rho = _rho_after(r, np.random.default_rng(0).uniform(size=12))
    assert np.isclose(np.trace(rho).real, 1)
    assert np.allclose(rho, rho.conj().T)
    assert np.linalg.eigvalsh(rho).min() > -1e-10


def test_unitary():
    r = FloquetReservoir(n=5, seed=1)
    assert np.allclose(r.U @ r.U.conj().T, np.eye(32))


def test_reset_writes_input_on_qubit0():
    r = FloquetReservoir(n=3, seed=0)
    rho = r._reset_q0(np.eye(8, dtype=complex) / 8, 0.3)
    z0 = np.real(np.diag(rho)) @ r.z[:, 0]
    assert np.isclose(z0, 1 - 2 * 0.3)


def test_perfect_flip_has_no_input_dependence_beyond_qubit0():
    """theta = pi, J arbitrary: Z populations of qubits 1.. never depend on the input (no mixing)."""
    u = np.random.default_rng(1).uniform(size=50)
    F1 = FloquetReservoir(n=4, theta=np.pi, J=0.4, seed=2).run(u)
    F2 = FloquetReservoir(n=4, theta=np.pi, J=0.4, seed=2).run(u[::-1].copy())
    assert np.allclose(F1[:, 1:4], F2[:, 1:4])


def test_matches_qiskit_aer_density_matrix():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from qreservoir.circuit import step_circuit

    n, u = 3, [0.2, 0.9, 0.55]
    r = FloquetReservoir(n=n, theta=0.7 * np.pi, J=0.4, cycles_per_step=2, seed=5)
    F = r.run(np.array(u))
    qc = QuantumCircuit(n)
    for t, x in enumerate(u):
        step_circuit(n, x, r.theta, r.J, r.h, 2, qc)
        qc.save_density_matrix(label=f"s{t}")
    res = AerSimulator(method="density_matrix").run(qc).result().data(0)
    for t in range(len(u)):
        p = np.real(np.diag(np.asarray(res[f"s{t}"])))
        assert np.allclose(r.features_from_probs(p), F[t], atol=1e-8)


def test_reproducible_and_shots_close():
    u = np.random.default_rng(2).uniform(size=40)
    a = FloquetReservoir(n=4, seed=7).run(u)
    assert np.allclose(a, FloquetReservoir(n=4, seed=7).run(u))
    s = FloquetReservoir(n=4, seed=7).run(u, shots=200_000)
    assert np.abs(a - s).max() < 0.02


def test_capacity_sanity():
    u = np.random.default_rng(0).uniform(size=1500)
    assert capacities(lagged(u, 5), u)["MC"] == pytest.approx(5, abs=0.1)
    assert capacities(EchoStateNetwork(seed=0).run(u), u)["MC"] > 3


def test_splits_time_ordered_and_scaling_uses_train_only():
    _, y, _ = load_series("sunspots")
    sp = splits(len(y))
    assert sp["train"].stop <= sp["val"].start and sp["val"].stop <= sp["test"].start
    z, (lo, hi) = scale(y, sp)
    assert lo == y[sp["train"]].min() and hi == y[sp["train"]].max()
    r = one_step(lagged(z, 3), z, sp)
    assert len(r["test_pred"]) == sp["test"].stop - sp["test"].start - 1


def test_dm_sign():
    e = np.random.default_rng(0).normal(size=200)
    assert diebold_mariano(2 * e, e)["mean_loss_diff"] > 0
