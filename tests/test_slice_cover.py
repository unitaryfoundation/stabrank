"""Controls for the all-visible base slice matcher (verify_challenge/slice_cover.py)."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("sympy")   # verify_challenge needs the challenge extra

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


def _native_available():
    from slice_cover import _native_kernel_class
    return _native_kernel_class() is not None


def test_native_kernel_matches_reference_on_the_h6_sample(enum3):
    """The compiled stage A kernel and the Python matcher agree run by run on
    the 160 sampled (cover, x0) pairs of research/h6_rank5/results/sample.json:
    same coordinate-slice solution counts and the same (empty) hit sets."""
    if not _native_available():
        pytest.skip("stabrank_core without SliceMatchKernel")
    import json
    from slice_cover import x0_reps
    E = enum3
    with open(os.path.join(ROOT, "research", "h6_rank5", "results", "sample.json")) as f:
        covers = [tuple(c) for c in json.load(f)["covers"]]
    ref, nat = SliceMatcher(E, 3, native=False), SliceMatcher(E, 3)
    assert nat.native is not None
    native_runs = 0
    for cover in covers:
        for x0 in x0_reps(3):
            h_ref, s_ref = ref.run(cover, x0)
            h_nat, s_nat = nat.run(cover, x0)
            assert s_nat["coord_solutions"] == s_ref["coord_solutions"], (cover, x0)
            assert s_nat["refused"] == s_ref["refused"]
            key = lambda hits: sorted(tuple(sorted(exact_codes(t)[0].tobytes() for t in h["terms"])) for h in hits)
            assert key(h_nat) == key(h_ref), (cover, x0)
            native_runs += s_nat.get("native", False)
    assert native_runs >= 150


def test_native_kernel_matches_reference_hits_for_h4(enum3):
    """On the rank-4 bases of |H>^4 with distinct independent states (one
    sliced qubit) the kernel returns the same hits as the reference."""
    if not _native_available():
        pytest.skip("stabrank_core without SliceMatchKernel")
    from common import load_decompositions
    E = enum3
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    ref, nat = SliceMatcher(E, 1, native=False), SliceMatcher(E, 1)
    decs, _ = load_decompositions("qubit_H", 4, 4)
    compared = with_hits = 0
    for terms, _ in decs[:12]:
        for x0 in (0, 1):
            slices = [t.reshape(2, 8)[x0] for t in terms]
            if any(np.linalg.norm(s) < 1e-9 for s in slices):
                continue
            cover = tuple(lookup[exact_codes(s)[0].tobytes()] for s in slices)
            if len(set(cover)) < len(cover):
                continue
            h_ref, s_ref = ref.run(cover, x0)
            h_nat, s_nat = nat.run(cover, x0)
            if not s_nat.get("native"):
                continue                      # dependent base: the kernel declined
            key = lambda hits: sorted(tuple(sorted(exact_codes(t)[0].tobytes() for t in h["terms"])) for h in hits)
            assert key(h_nat) == key(h_ref), (cover, x0)
            assert s_nat["coord_solutions"] == s_ref["coord_solutions"]
            compared += 1
            with_hits += bool(h_nat)
    assert compared >= 4 and with_hits >= 4


def test_native_cover5_matches_reference_pair_covers():
    """The compiled 5-cover pair kernel and CoverEnumerator.pair_covers list
    the same full covers on the first partner pairs of the sample's pivot."""
    from slice_cover import _native_cover5, _reduce
    if _native_cover5() is None:
        pytest.skip("stabrank_core without cover5_pair")
    ref, nat = CoverEnumerator(3, native=False), CoverEnumerator(3)
    assert ref.native_cover5 is None and nat.native_cover5 is not None
    i = int(ref.reps[3])
    Qi, _ = _reduce(ref.F1, ref.Q1, ref.Q1[i])
    members, partners = ref.pivot_plan(i)
    mask = np.zeros(ref.N, dtype=bool)
    mask[members] = True
    total = 0
    for j in partners[:2]:
        got_ref, n_ref = ref.pair_covers(5, i, int(j), Qi, mask)
        got_nat, n_nat = nat.pair_covers(5, i, int(j), Qi, mask)
        assert got_nat == got_ref, (i, j)
        # the modular candidate counts depend on the random functional (accidental
        # key collisions are rejected by the exact check) and only agree roughly
        assert len(got_ref) <= n_nat and abs(n_nat - n_ref) < 0.05 * n_ref
        total += len(got_ref)
    assert total >= 10
