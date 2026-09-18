"""The exhaustive exclusion machinery must still find what exists.

Every certificate built on rank_exclusion.py rests on the search recovering a
decomposition whenever one exists, and on the symmetry reduction being a
genuine symmetry. Both are checked on the small dictionaries: the known rank-2
and rank-3 decompositions at two qutrits and three qubits must be found with
and without symmetry reduction, and the symmetry groups must have the orders
the single-copy Clifford stabilizers predict.
"""

import os
import sys

import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from rank_exclusion import (  # noqa: E402
    dictionary, psi_for, rank2_search, rank3_search, symmetry_orbit_reps,
)


@pytest.fixture(scope="module")
def d2():
    return dictionary(3, 2)


@pytest.fixture(scope="module")
def q3():
    return dictionary(2, 3)


def test_dictionary_counts(d2, q3):
    assert d2.shape == (9, 360)
    assert q3.shape == (8, 1080)


def test_rank2_finds_strange(d2):
    pairs, _ = rank2_search(psi_for("S", 2), d2)
    assert pairs and pairs != "RANK1"


def test_rank2_excludes_norrell(d2):
    pairs, worst = rank2_search(psi_for("N", 2), d2)
    assert pairs == []
    assert worst < 0.99


@pytest.mark.parametrize("orbit, m, p", [("N", 2, 3), ("H3", 2, 3), ("T3", 2, 3),
                                         ("qubit_H", 3, 2), ("qubit_T", 3, 2)])
def test_rank3_found_with_and_without_symmetry(orbit, m, p, d2, q3):
    D = d2 if p == 3 else q3
    psi = psi_for(orbit, m)
    plain = rank3_search(psi, D, workers=1)
    reps, info = symmetry_orbit_reps(orbit, m, D)
    sym = rank3_search(psi, D, workers=1, pivots=reps)
    assert plain["found"] and sym["found"]
    assert info["orbits"] < D.shape[1]


@pytest.mark.parametrize("orbit, local, anti", [
    ("S", 24, True), ("N", 6, True), ("H3", 4, True), ("T3", 3, True),
    ("qubit_H", 2, True), ("qubit_T", 3, True)])
def test_symmetry_group_orders(orbit, local, anti, d2, q3):
    p = 3 if orbit in ("S", "N", "H3", "T3") else 2
    D = d2 if p == 3 else q3
    m = 2 if p == 3 else 3
    _, info = symmetry_orbit_reps(orbit, m, D)
    assert info["local"] == local
    assert info["antiunitary"] == anti
    import math
    assert info["order"] == 2 * local ** m * math.factorial(m)


def test_code_enumerator_matches_dictionary(d2):
    """The phase-code enumerator produces exactly the states of the array path."""
    import numpy as np
    from qutrit_codes import all_codes, encode, count
    codes, ks = all_codes(2)
    assert codes.shape[0] == count(2) == 360
    ref = encode(d2.astype(np.complex128))
    assert {c.tobytes() for c in codes} == {c.tobytes() for c in ref}
    codes3, _ = all_codes(3)
    assert codes3.shape[0] == count(3) == 30240


def test_code_path_finds_known_decompositions(d2):
    from qutrit_codes import all_codes
    from rank_exclusion_codes import (_Geometry, rank2_search_codes, rank3_search_codes,
                                      symmetry_orbit_reps_codes, psi_for as psi_codes)
    c, k = all_codes(2)
    geo = _Geometry(psi_codes("N", 2), c, k)
    reps, _ = symmetry_orbit_reps_codes("N", 2, c, k)
    assert rank3_search_codes(geo, reps, workers=1)["found"]
    pairs, _ = rank2_search_codes(_Geometry(psi_codes("S", 2), c, k))
    assert pairs and pairs != "RANK1"
