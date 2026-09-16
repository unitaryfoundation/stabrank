"""The witness parametrisation reaches every stabilizer state and nothing else.

Round trip: enumerate every stabilizer state on two qutrits and on one, two and
three qubits, recover (k, x0, W, Q, l) from the amplitude vector, rebuild it
through the verifier, and require the rebuilt state to be parallel to the
original. The qubit case is the one that matters: with phases in {+1, -1} only,
the Y eigenstates are unreachable, and this test is what fails if the fourth
roots are ever dropped from the verifier again.

The converse: a vector that is not a stabilizer state must be refused rather
than approximated.
"""

import os
import sys

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from qubit_states import all_states  # noqa: E402
from stabrank_verify import stabilizer_vector  # noqa: E402
from to_witness import NotStabilizer, term_from_vector  # noqa: E402

from stabrank.examples.t3_galois_lower_bound import distinct_states  # noqa: E402


def _parallel(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return abs(abs(np.vdot(u, v)) - 1) < 1e-9


def _numeric(term, p, n):
    return np.array([complex(z) for z in stabilizer_vector(term, p, n)])


@pytest.mark.parametrize("n", [1, 2])
def test_qutrit_round_trip(n):
    D = distinct_states(n)
    for j in range(D.shape[1]):
        v = D[:, j]
        term = term_from_vector(v, 3, n)
        assert _parallel(_numeric(term, 3, n), v)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_qubit_round_trip(n):
    D = all_states(n)
    for j in range(D.shape[1]):
        v = D[:, j]
        term = term_from_vector(v, 2, n)
        assert _parallel(_numeric(term, 2, n), v)


def test_y_eigenstate_is_expressible():
    """|+i> = (|0> + i|1>)/sqrt 2 needs l = 1, not a sign."""
    v = np.array([1, 1j]) / np.sqrt(2)
    term = term_from_vector(v, 2, 1)
    assert term["l"] == [1]
    assert _parallel(_numeric(term, 2, 1), v)


@pytest.mark.parametrize("v, p, n", [
    (np.array([0, 1, -1]) / np.sqrt(2), 3, 1),              # Strange: support size 2
    (np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)]), 2, 1),  # H-type: unequal moduli
    (np.array([1, np.exp(2j * np.pi / 9), np.exp(4j * np.pi / 9)]) / np.sqrt(3), 3, 1),
    (np.array([1, 1, 1, -1, 1, 1, 1, 1]) / np.sqrt(8), 2, 3),  # cubic phase
])
def test_non_stabilizer_is_refused(v, p, n):
    with pytest.raises(NotStabilizer):
        term_from_vector(v, p, n)
