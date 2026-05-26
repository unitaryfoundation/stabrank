"""Tests for qutrit Clifford group utilities and gadget checking."""

from __future__ import annotations

import numpy as np
import pytest

from stabrank.qutrit_clifford import (
    H3,
    S3,
    X3,
    Z3,
    SUM_GATE,
    enumerate_single_qutrit_cliffords,
    is_proportional_to_unitary,
    compute_post_measurement_operators,
    check_gadget,
    _symplectic_generators_4,
)
from stabrank.target_functions import (
    qutrit_hadamard_eigenstate,
    qutrit_complex_magic_state,
)

OMEGA: complex = np.exp(2j * np.pi / 3)


@pytest.fixture(scope="module")
def clifford_group() -> list[np.ndarray]:
    return enumerate_single_qutrit_cliffords()


def _pauli(a: int, b: int) -> np.ndarray:
    return np.linalg.matrix_power(X3, a) @ np.linalg.matrix_power(Z3, b)


def _to_pauli_label(M: np.ndarray, tol: float = 1e-9) -> tuple[int, int] | None:
    """Return (a, b) if M is a nonzero scalar multiple of X^a Z^b, else None."""
    for a in range(3):
        for b in range(3):
            P = _pauli(a, b)
            mask = np.abs(P) > tol
            ratios = M[mask] / P[mask]
            if np.allclose(ratios, ratios[0], atol=tol) and np.isclose(
                np.abs(ratios[0]), 1.0, atol=tol
            ):
                return (a, b)
    return None


def _pauli_2q(a1: int, a2: int, b1: int, b2: int) -> np.ndarray:
    """Two-qutrit Pauli X_1^{a1} X_2^{a2} Z_1^{b1} Z_2^{b2}."""
    return np.kron(_pauli(a1, b1), _pauli(a2, b2))


def _to_pauli_label_2q(
    M: np.ndarray, tol: float = 1e-9
) -> tuple[int, int, int, int] | None:
    """Return (a1, a2, b1, b2) if M is a unit-modulus scalar multiple of
    X_1^{a1} X_2^{a2} Z_1^{b1} Z_2^{b2}, else None."""
    for a1 in range(3):
        for a2 in range(3):
            for b1 in range(3):
                for b2 in range(3):
                    P = _pauli_2q(a1, a2, b1, b2)
                    mask = np.abs(P) > tol
                    if not mask.any():
                        continue
                    ratios = M[mask] / P[mask]
                    if np.allclose(ratios, ratios[0], atol=tol) and np.isclose(
                        np.abs(ratios[0]), 1.0, atol=tol
                    ):
                        return (a1, a2, b1, b2)
    return None


_I3 = np.eye(3, dtype=complex)
_TWO_QUTRIT_GENERATORS = [
    ("H1", np.kron(H3, _I3)),
    ("H2", np.kron(_I3, H3)),
    ("S1", np.kron(S3, _I3)),
    ("S2", np.kron(_I3, S3)),
    ("SUM", SUM_GATE),
]


# -----------------------------------------------------------------------
# Single-qutrit Clifford group
# -----------------------------------------------------------------------


class TestSingleQutritCliffordGroup:
    def test_group_order_is_216(self, clifford_group: list[np.ndarray]) -> None:
        assert len(clifford_group) == 216

    def test_generators_are_unitary(self) -> None:
        for name, G in [("H3", H3), ("S3", S3), ("X3", X3)]:
            prod = G.conj().T @ G
            assert np.allclose(prod, np.eye(3), atol=1e-12), (
                f"{name} is not unitary"
            )

    def test_generators_normalize_paulis(self) -> None:
        """Each Clifford generator must send X and Z to phase * X^a Z^b
        under conjugation (the strict normalizer property), not merely to
        some Clifford."""
        for name, G in [("H3", H3), ("S3", S3), ("X3", X3)]:
            for label, P in [("X", X3), ("Z", Z3)]:
                conjugated = G @ P @ G.conj().T
                assert _to_pauli_label(conjugated) is not None, (
                    f"{name} {label} {name}^dag is not a phase * X^a Z^b"
                )

    def test_full_enumeration_normalizes_paulis(
        self, clifford_group: list[np.ndarray]
    ) -> None:
        """Every enumerated Clifford must normalize the full Pauli group:
        for each C and each non-identity X^a Z^b, C X^a Z^b C^dag is a phase
        times some X^a' Z^b'."""
        non_identity = [
            (a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)
        ]
        for i, C in enumerate(clifford_group):
            for a, b in non_identity:
                conjugated = C @ _pauli(a, b) @ C.conj().T
                assert _to_pauli_label(conjugated) is not None, (
                    f"Clifford {i} fails to normalize X^{a} Z^{b}"
                )

    def test_all_elements_are_unitary(
        self, clifford_group: list[np.ndarray]
    ) -> None:
        for i, C in enumerate(clifford_group):
            prod = C.conj().T @ C
            assert np.allclose(prod, np.eye(3), atol=1e-10), (
                f"Element {i} is not unitary"
            )


# -----------------------------------------------------------------------
# Two-qutrit Clifford generators
# -----------------------------------------------------------------------


class TestTwoQutritCliffordGenerators:
    """Regression tests for the (4x4 symplectic, 9x9 unitary) generator
    pairing that underlies the gadget enumeration. The exact symplectic
    deduplication in enumerate_sp4_f3_with_unitaries pairs each symplectic
    generator with a 9x9 unitary; if the unitary's conjugation action does
    not match the declared symplectic action, the deduplication is unsound
    and the Sp(4, F_3) sweep underwriting Theorem~3 of the manuscript
    would silently produce a corrupted certificate.
    """

    def test_generators_are_unitary(self) -> None:
        for name, G in _TWO_QUTRIT_GENERATORS:
            prod = G.conj().T @ G
            assert np.allclose(prod, np.eye(9), atol=1e-12), (
                f"{name} is not unitary"
            )

    def test_generators_normalize_two_qutrit_paulis(self) -> None:
        """Each generator G must send every non-identity two-qutrit Pauli
        to a unit-modulus scalar times another two-qutrit Pauli under
        conjugation."""
        for name, G in _TWO_QUTRIT_GENERATORS:
            for a1 in range(3):
                for a2 in range(3):
                    for b1 in range(3):
                        for b2 in range(3):
                            if (a1, a2, b1, b2) == (0, 0, 0, 0):
                                continue
                            P = _pauli_2q(a1, a2, b1, b2)
                            conjugated = G @ P @ G.conj().T
                            assert _to_pauli_label_2q(conjugated) is not None, (
                                f"{name} fails to normalize "
                                f"X1^{a1} X2^{a2} Z1^{b1} Z2^{b2}"
                            )

    def test_unitary_action_matches_symplectic_generator(self) -> None:
        """For each (F_symp, U) generator pair, the unitary conjugation's
        Pauli image must match F_symp @ v (mod 3) under the standard
        column-vector convention C P(v) C^dag = P(F v)."""
        symp_generators = _symplectic_generators_4()
        assert len(symp_generators) == len(_TWO_QUTRIT_GENERATORS)
        for (name, G), F_symp in zip(_TWO_QUTRIT_GENERATORS, symp_generators):
            for a1 in range(3):
                for a2 in range(3):
                    for b1 in range(3):
                        for b2 in range(3):
                            if (a1, a2, b1, b2) == (0, 0, 0, 0):
                                continue
                            v = np.array([a1, a2, b1, b2], dtype=int)
                            predicted = tuple(int(x) for x in (F_symp @ v) % 3)
                            P = _pauli_2q(a1, a2, b1, b2)
                            conjugated = G @ P @ G.conj().T
                            actual = _to_pauli_label_2q(conjugated)
                            assert actual == predicted, (
                                f"{name}: conjugating X1^{a1} X2^{a2} "
                                f"Z1^{b1} Z2^{b2} yields label {actual}, "
                                f"symplectic prediction was {predicted}"
                            )


# -----------------------------------------------------------------------
# is_proportional_to_unitary
# -----------------------------------------------------------------------


class TestProportionalToUnitary:
    def test_identity(self) -> None:
        assert is_proportional_to_unitary(np.eye(3, dtype=complex))

    def test_scaled_unitary(self) -> None:
        assert is_proportional_to_unitary(2.5 * H3)

    def test_non_unitary_diagonal(self) -> None:
        D = np.diag([1.57, -0.21, -0.21]).astype(complex)
        assert not is_proportional_to_unitary(D)

    def test_zero_matrix(self) -> None:
        assert not is_proportional_to_unitary(np.zeros((3, 3), dtype=complex))


# -----------------------------------------------------------------------
# Gadget check: known-good (SUM + T3)
# -----------------------------------------------------------------------


class TestGadgetSUMWithT3:
    """SUM + |T3> should be a valid gadget injecting diag(1, zeta_9, zeta_9^2)."""

    def test_sum_t3_passes_all_conditions(
        self, clifford_group: list[np.ndarray]
    ) -> None:
        t3 = qutrit_complex_magic_state(1)
        result = check_gadget(SUM_GATE, t3, clifford_group)
        assert result is not None, "SUM + |T3> should be a valid gadget"

    def test_injected_gate_is_t_gate(
        self, clifford_group: list[np.ndarray]
    ) -> None:
        t3 = qutrit_complex_magic_state(1)
        result = check_gadget(SUM_GATE, t3, clifford_group)
        assert result is not None

        U = result["U"]
        zeta9 = np.exp(2j * np.pi / 9)
        expected = np.diag([1.0, zeta9, zeta9**2])

        # U should be proportional to the expected diagonal gate
        nonzero = np.abs(expected) > 1e-10
        ratio = U[nonzero] / expected[nonzero]
        phases = ratio
        assert np.allclose(
            np.abs(phases), np.abs(phases[0]), atol=1e-6
        ), "Injected gate should be proportional to diag(1, zeta_9, zeta_9^2)"


# -----------------------------------------------------------------------
# Gadget check: known-bad (SUM + H3)
# -----------------------------------------------------------------------


class TestGadgetSUMWithH3:
    """SUM + |H3> should fail: the diagonal kickback is non-unitary."""

    def test_sum_h3_fails(self, clifford_group: list[np.ndarray]) -> None:
        h3 = qutrit_hadamard_eigenstate(1)
        result = check_gadget(SUM_GATE, h3, clifford_group)
        # Should fail because E_0 is not proportional to a unitary
        assert result is None, (
            "SUM + |H3> should NOT be a valid gadget "
            "(non-unitary diagonal kickback)"
        )

    def test_e0_is_not_proportional_to_unitary(self) -> None:
        """Directly verify that E_0 from SUM + H3 is non-unitary."""
        h3 = qutrit_hadamard_eigenstate(1)
        ops = compute_post_measurement_operators(SUM_GATE, h3)
        assert not is_proportional_to_unitary(ops[0]), (
            "E_0 from SUM + |H3> should not be proportional to a unitary"
        )
