"""Controls for the all-visible base slice matcher (verify_challenge/slice_cover.py)."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))

from slice_cover import (CoverEnumerator, Field, P1, P2, SliceMatcher, TermOptions,  # noqa: E402
                         composite_codes, exact_codes, flats_with_presence)


@pytest.fixture(scope="module")
def enum3():
    return CoverEnumerator(3)


def test_composite_codes_reproduce_random_stabilizer_states():
    """A random six-qubit stabilizer state sliced along three qubits: its
    flat is one of the subspaces with the observed coordinate presence and
    its composite codes are among the structure lemma's assignments."""
    from two_qubit_slice import _random_term
    rng = np.random.default_rng(3)
    F1, F2 = Field(P1), Field(P2)
    n1, n2 = 3, 3
    composite = [x for x in range(1, 1 << n1) if bin(x).count("1") >= 2]
    seen = set()
    for _ in range(80):
        v = _random_term(rng, n1 + n2)
        B = v.reshape(1 << n1, 1 << n2)
        nz = [x for x in range(1 << n1) if np.linalg.norm(B[x]) > 1e-9]
        x0 = nz[int(rng.integers(len(nz)))]
        _, scal = exact_codes(B[x0])
        o = TermOptions(B[x0] / scal, n2, F1, F2)
        W = frozenset(x ^ x0 for x in nz if x != x0)
        code = {x ^ x0: 4 * o.code_of(B[x] / scal)[0] + o.code_of(B[x] / scal)[1] for x in nz if x != x0}
        present = [k for k in range(n1) if (1 << k) in W]
        assert W in flats_with_presence(n1, present)
        rows = composite_codes(o, W, {k: code[1 << k] for k in present}, composite)
        want = np.array([code.get(x, o.absent) for x in composite])
        assert (rows == want[None, :]).all(axis=1).any()
        seen.add(len(nz))
    assert seen == {1, 2, 4, 8}


def test_matcher_recovers_stored_rank4_decompositions_of_h4(enum3):
    """Each stored rank-4 decomposition of |H>^4 with an all-visible slice
    along qubit 0 is recovered from that base slice (independent, dependent
    or repeated base states alike)."""
    from common import load_decompositions
    E = enum3
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    M = SliceMatcher(E, 1)
    decs, _ = load_decompositions("qubit_H", 4, 4)
    tried = 0
    for terms, _ in decs[:12]:
        for x0 in (0, 1):
            slices = [t.reshape(2, 8)[x0] for t in terms]
            if any(np.linalg.norm(s) < 1e-9 for s in slices):
                continue
            cover = tuple(lookup[exact_codes(s)[0].tobytes()] for s in slices)
            hits, st = M.run(cover, x0)
            assert not st["refused"]
            want = sorted(exact_codes(t)[0].tobytes() for t in terms)
            assert any(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == want
                       and h["rank"] == 4 and h["exact"] and h["nonzero"] for h in hits), (cover, x0, st)
            tried += 1
    assert tried >= 4
