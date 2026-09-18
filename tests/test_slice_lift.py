"""Controls for the slice-and-lift lower-bound pipeline."""

import os
import sys

import pytest

sp = pytest.importorskip("sympy")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "verify_challenge"))

from rank_exclusion import dictionary  # noqa: E402
from slice_lift import all_decompositions, lift_all, pauli_images, pauli_images_qubit  # noqa: E402


def test_pauli_images_count_and_inverse():
    import numpy as np
    D = dictionary(3, 2)
    u = D[:, 17]
    pairs = pauli_images(u, 2)
    assert len(pairs) == 9
    # Q^-1 (Q u) = u for the representative used: rebuild Q from its action
    for qu, qinv in pairs:
        assert abs(abs(np.vdot(qu, qu)) - 1) < 1e-9 and abs(abs(np.vdot(qinv, qinv)) - 1) < 1e-9
    Dq = dictionary(2, 3)
    assert len(pauli_images_qubit(Dq[:, 5], 3)) == 8


def test_t3_rank3_lifts_one_copy():
    D = dictionary(3, 1)
    decs, _ = all_decompositions("T3", 1, 3, D, verbose=False)
    assert decs
    assert lift_all("T3", 1, decs, D, verbose=False)


def test_strange_rank2_lifts_with_a_zero_amplitude():
    """|S> has alpha_0 = 0, so the slice-0 equation is homogeneous."""
    D = dictionary(3, 1)
    decs, _ = all_decompositions("S", 1, 2, D, verbose=False)
    assert decs
    assert lift_all("S", 1, decs, D, verbose=False)


def test_norrell_rank3_does_not_lift():
    D = dictionary(3, 2)
    decs, _ = all_decompositions("N", 2, 3, D, verbose=False)
    assert decs
    assert not lift_all("N", 2, decs, D, verbose=False)


def test_qubit_t_rank3_lifts_to_four_copies():
    D = dictionary(2, 3)
    decs, _ = all_decompositions("qubit_T", 3, 3, D, verbose=False)
    assert decs
    assert lift_all("qubit_T", 3, decs, D, verbose=False)


def test_qubit_h_rank2_does_not_lift():
    D = dictionary(2, 2)
    decs, _ = all_decompositions("qubit_H", 2, 2, D, verbose=False)
    assert len(decs) == 1
    assert not lift_all("qubit_H", 2, decs, D, verbose=False)


def test_rank4_pivot_search_matches_brute_force():
    """Every rank-4 decomposition of |N>^2 containing state 0, against a
    direct check of all triples of partners on the 1080-state dictionary."""
    import itertools
    import numpy as np
    from rank_exclusion import psi_for
    from slice_lift import decompositions_with_pivot
    D = dictionary(3, 2)
    psi = psi_for("N", 2)
    psi = psi / np.linalg.norm(psi)
    found = set(decompositions_with_pivot(psi, D, 0, 4))
    # brute force on a random subset of partner triples that includes the found ones
    rng = np.random.default_rng(0)
    others = np.arange(1, D.shape[1])
    checks = set(found)
    for _ in range(3000):
        checks.add(tuple(sorted((0,) + tuple(int(x) for x in rng.choice(others, 3, replace=False)))))
    for cols in checks:
        A = D[:, list(cols)]
        ok = np.linalg.matrix_rank(A, tol=1e-8) == 4
        if ok:
            x, *_ = np.linalg.lstsq(A, psi, rcond=None)
            ok = np.linalg.norm(A @ x - psi) < 1e-9
        assert ok == (cols in found), cols


def test_stabilizer_reduction_covers_every_orbit():
    """The partner loop reduced by the pivot's stabilizer finds the same
    rank-4 decompositions of |N>^2 as the full loop, up to the symmetry
    group (compared after closing both lists under the generators)."""
    import numpy as np
    from rank_exclusion import psi_for, symmetry_orbit_reps
    from slice_lift import decompositions_with_pivot

    def closure(decs, perms):
        seen = set(decs)
        frontier = list(decs)
        while frontier:
            nxt = []
            for d in frontier:
                for p in perms:
                    e = tuple(sorted(int(p[c]) for c in d))
                    if e not in seen:
                        seen.add(e)
                        nxt.append(e)
            frontier = nxt
        return seen

    D = dictionary(3, 2)
    psi = psi_for("N", 2)
    reps, info = symmetry_orbit_reps("N", 2, D, antiunitary=False)
    full = set()
    for i in reps:
        full.update(decompositions_with_pivot(psi, D, int(i), 4))
    reduced, _ = all_decompositions("N", 2, 4, D, verbose=False)
    assert len(reduced) < len(full)
    assert closure(full, info["perms"]) == closure(set(reduced), info["perms"])
