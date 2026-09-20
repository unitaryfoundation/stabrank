"""The p-generic dictionary and the ququint T-type orbit.

`dictionary(p, n)` and `clifford_group(p)` now accept any prime. The counts
are checked against p^n prod_{j<=n} (p^j + 1) and p^3 (p^2 - 1), and the
verifier's parametrisation is checked to name exactly the states of the
dictionary: every (flat, coset, quadratic-plus-linear phase) term rebuilt by
`stabilizer_vector` must land in the dictionary, and the dictionary must be
exhausted, for two qutrits and for one ququint. That is the check that the
phase convention w_p^(Q(y) + l.y) is the same object on both sides at p = 3
and at p = 5. The qubit and qutrit paths are unchanged, which the existing
tests cover; here the p = 3 group and dictionary are asserted to have kept
their sizes.
"""

import itertools
import os
import sys

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, ROOT)

from rank_exclusion import clifford_group, dictionary, rank2_search  # noqa: E402
from qutrit_codes import _rref_matrices  # noqa: E402
from stabrank_verify import (ORBIT_P, orbit_state, stabilizer_vector,  # noqa: E402
                             target_vector, verify)


def count(p, n):
    t = p ** n
    for j in range(1, n + 1):
        t *= p ** j + 1
    return t


def _key(v):
    v = np.asarray(v, dtype=complex).ravel()
    v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
    return (np.round(v, 6) + 0.0).tobytes()


def _all_terms(p, n):
    """Every (k, x0, W, Q, l) term with W in reduced row echelon form and x0
    supported off the pivots, which names each state exactly once."""
    for k in range(n + 1):
        for W, pivots, non_pivots in _rref_matrices(n, k, p):
            for x0_free in itertools.product(range(p), repeat=n - k):
                x0 = [0] * n
                for j, c in enumerate(non_pivots):
                    x0[c] = x0_free[j]
                upper = [(i, j) for i in range(k) for j in range(i, k)]
                for qvals in itertools.product(range(p), repeat=len(upper)):
                    Q = [[0] * k for _ in range(k)]
                    for (i, j), q in zip(upper, qvals):
                        Q[i][j] = q
                    for ell in itertools.product(range(p), repeat=k):
                        yield {"k": k, "x0": x0, "W": W.tolist(), "Q": Q, "l": list(ell)}


@pytest.mark.parametrize("p, n", [(5, 1), (5, 2)])
def test_ququint_dictionary_count(p, n):
    D = dictionary(p, n)
    assert D.shape == (p ** n, count(p, n))
    assert D.shape[1] == {1: 30, 2: 3900}[n]
    assert np.allclose(np.linalg.norm(D, axis=0), 1)
    keys = {_key(D[:, i]) for i in range(D.shape[1])}
    assert len(keys) == D.shape[1], "dictionary states are not distinct up to phase"


def test_qutrit_dictionary_unchanged():
    assert dictionary(3, 2).shape == (9, 360)
    assert len(clifford_group(3)) == 216
    assert len(clifford_group(2)) == 24


def test_ququint_clifford_group():
    G = clifford_group(5)
    assert len(G) == 5 ** 3 * (5 ** 2 - 1) == 3000
    for U in G[:50]:
        assert np.allclose(U.conj().T @ U, np.eye(5))


def test_non_prime_rejected():
    with pytest.raises(ValueError):
        dictionary(4, 1)
    with pytest.raises(ValueError):
        clifford_group(6)


@pytest.mark.parametrize("p, n", [(3, 2), (5, 1)])
def test_parametrisation_names_the_dictionary(p, n):
    """The verifier's stabilizer_vector, over every RREF term, hits every
    dictionary state exactly once: the phase convention matches at p = 3 and
    at p = 5."""
    D = dictionary(p, n)
    index = {_key(D[:, i]): i for i in range(D.shape[1])}
    hit = set()
    terms = 0
    for term in _all_terms(p, n):
        v = stabilizer_vector(term, p, n)
        key = _key([complex(sp.N(z, 20)) for z in v])
        assert key in index, f"term {term} rebuilds to a vector not in the dictionary"
        hit.add(index[key])
        terms += 1
    assert terms == count(p, n)
    assert len(hit) == D.shape[1]


def test_t5_orbit_state():
    v = orbit_state("T5")
    assert ORBIT_P["T5"] == 5
    assert v.shape == (5, 1)
    assert sp.simplify(sum(abs(z) ** 2 for z in v) - 1) == 0
    w5 = np.exp(2j * np.pi / 5)
    want = np.array([w5 ** ((x ** 3) % 5) for x in range(5)]) / np.sqrt(5)
    assert np.allclose([complex(sp.N(z, 20)) for z in v], want)
    assert target_vector("T5", 2).shape == (25, 1)
    # not a stabilizer state: no dictionary state is parallel to it
    D = dictionary(5, 1)
    psi = np.array([complex(sp.N(z, 20)) for z in v])
    assert np.max(np.abs(psi.conj() @ D)) < 1 - 1e-6


def test_t5_rank_three_witness_verifies():
    """chi(|T5>) <= 3, and the verifier accepts the witness end to end. The
    third term is the full-support state with phase w_5^(4y^2 + 4y); the
    coefficients were recovered by fit_coeffs.fit and are exact."""
    terms = [{"k": 0, "x0": [0], "W": [], "Q": [], "l": []},
             {"k": 0, "x0": [1], "W": [], "Q": [], "l": []},
             {"k": 1, "x0": [0], "W": [[1]], "Q": [[4]], "l": [4]}]
    coeffs = ["-1/4 + sqrt(5)/4 + I*sqrt(sqrt(5)/40 + 1/8)",
              "1/2 + I*sqrt(1/4 - sqrt(5)/10)",
              "-1/4 + sqrt(5)/4 - I*sqrt(sqrt(5)/8 + 5/8)"]
    sub = {"schema_version": "0.1", "orbit": "T5", "m": 1, "direction": "upper",
           "rank": 3, "witness": {"terms": terms, "coeffs": coeffs},
           "provenance": {"author": "test", "date": "2026-09-20"}}
    r = verify(sub)
    assert r.ok and r.tier == "verified"
    assert abs(r.gamma - float(sp.log(3, 5))) < 1e-12
    # and rank 2 is excluded over the 30 single-ququint states
    D = dictionary(5, 1)
    psi = np.array([complex(sp.N(z, 20)) for z in orbit_state("T5")])
    pairs, worst = rank2_search(psi, D)
    assert pairs == [] and worst < 0.99
