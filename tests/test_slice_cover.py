"""Controls for the all-visible base slice matcher (verify_challenge/slice_cover.py)."""

import copy
import importlib
import json
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sympy")   # verify_challenge needs the challenge extra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))

from slice_cover import (CoverEnumerator, Field, P1, P2, SliceMatcher, TermOptions,  # noqa: E402
                         UnpinnedFamily, composite_codes, exact_codes, flats_with_presence,
                         reconstruct_block, x0_reps)

H6 = os.path.join(ROOT, "research", "h6_rank5")


@pytest.fixture(scope="module")
def enum3():
    return CoverEnumerator(3)


@pytest.fixture(scope="module")
def h6():
    """The research/h6_rank5 modules, imported under their own `common`
    (research/constructions has a module of the same name)."""
    names = ("common", "driver", "aggregate", "batch")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    sys.path.insert(0, H6)
    try:
        mods = {k: importlib.import_module(k) for k in names}
    finally:
        sys.path.remove(H6)
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**mods)


# ------------------------------------------------- planted instances -------

def _to_field(F, v):
    """A vector with Gaussian-integer entries reduced to F_p."""
    re, im = np.round(v.real).astype(np.int64), np.round(v.imag).astype(np.int64)
    assert np.allclose(v, re + 1j * im, atol=1e-9)
    return (re + F.i * im) % F.p


class PlantedMatcher(SliceMatcher):
    """SliceMatcher run against a planted six-qubit target Psi (a sum of
    stabilizer states with Gaussian-integer coefficients) instead of |H>^6:
    the base slice family is solved against Psi's slice at x0 and every
    slice equation against Psi's slice there. confirm() still measures the
    residual against |H>^6, so hits are compared by their term codes."""

    def __init__(self, E, Psi, x0, n1=3):
        rows = Psi.reshape(1 << n1, -1)
        E2 = copy.copy(E)
        E2.psi = rows[x0]
        E2.psi1, E2.psi2 = _to_field(E.F1, rows[x0]), _to_field(E.F2, rows[x0])
        super().__init__(E2, n1, native=False)
        self._rows = rows

    def rhs(self, x0, x):
        v = self._rows[x]
        return _to_field(self.F1, v), _to_field(self.F2, v), v


def _full_term(rng, m, n1):
    """A random m-qubit stabilizer state, scaled to entries in {0, +-1, +-i},
    with every slice along the first n1 qubits nonzero."""
    from two_qubit_slice import _random_term
    while True:
        v = _random_term(rng, m)
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        B = v.reshape(1 << n1, -1)
        if all(np.linalg.norm(B[x]) > 1e-9 for x in range(1 << n1)):
            return v


def _diagonal_copy(v, n1, sign):
    """v with row y of its (2^n1, 2^n2) slicing multiplied by sign(y) in
    {+1, -1}: a diagonal Clifford on the sliced qubits, so again a
    stabilizer state with the same base slice at any row where sign is +1."""
    B = v.reshape(1 << n1, -1).copy()
    for y in range(1 << n1):
        B[y] *= sign(y)
    return B.ravel()


def _codes_key(terms):
    return sorted(exact_codes(t)[0].tobytes() for t in terms)


def _base_index(E, v, x0, n1):
    return {E.codes[i].tobytes(): i for i in range(E.N)}[exact_codes(v.reshape(1 << n1, -1)[x0])[0].tobytes()]


def _plant(E, rng, n_ordinary, x0=0, n1=3, m=6):
    """n_ordinary random full terms with distinct base slices, plus one
    further full term whose base slice differs from all of them."""
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    terms, bases = [], []
    while len(terms) < n_ordinary + 1:
        v = _full_term(rng, m, n1)
        b = lookup[exact_codes(v.reshape(1 << n1, -1)[x0])[0].tobytes()]
        if b in bases:
            continue
        terms.append(v)
        bases.append(b)
    return terms[:-1], bases[:-1], terms[-1], bases[-1]


def _recover(E, Psi, cover, x0, want):
    M = PlantedMatcher(E, Psi, x0)
    hits, st = M.run(tuple(cover), x0)
    assert not st["refused"], st
    assert st["kappa"] == 0, st
    assert any(_codes_key(h["terms"]) == want for h in hits), (st, [len(h["terms"]) for h in hits])
    return hits, st


def test_reconstruct_block_recovers_copies_cancelling_at_a_slice(enum3):
    """Two copies of one base state with phases differing by a sign at two
    offsets (one coordinate, one composite) and equal coefficients: the
    block vanishes there, the slice equations record no translate class,
    and the reconstruction has to place both copies in one class outside
    the set with net coordinate zero."""
    E = enum3
    rng = np.random.default_rng(11)
    n1, x0 = 3, 0
    for _ in range(3):
        v = _full_term(rng, 6, n1)
        sign = lambda y: -1 if ((y & 1) and not (y >> 1 & 1)) else 1     # Z_0 CZ_01: rows 1 and 5
        w = _diagonal_copy(v, n1, sign)
        u_idx = _base_index(E, v, x0, n1)
        o = TermOptions(E.C[:, u_idx], 3, E.F1, E.F2)
        scal = exact_codes(v.reshape(8, 8)[x0])[1]
        c1, c2 = 1.0, 1.0
        data = []
        for x in [1, 2, 4, 3, 5, 6, 7]:
            tot = (c1 * v + c2 * w).reshape(8, 8)[x] / scal
            a = {}
            if np.linalg.norm(tot) > 1e-9:
                k, l = o.code_of(tot / abs(tot[np.flatnonzero(np.abs(tot) > 1e-9)[0]]))
                a = {k: tot[np.flatnonzero(np.abs(tot) > 1e-9)[0]] / o.vecs[4 * k][np.flatnonzero(np.abs(tot) > 1e-9)[0]]}
            data.append((x, a))
        assert not data[0][1] and not data[4][1], "the planted pair should vanish at offsets 1 and 5"
        out = reconstruct_block(o, 2, c1 + c2, data, n1, rng)
        found = False
        for copies, degenerate in out:
            assert not degenerate
            terms = []
            for cval, cd in copies:
                t = np.zeros((8, 8), dtype=complex)
                t[x0] = o.u
                for x, code in cd.items():
                    t[x0 ^ x] = o.vecs[code]
                terms.append(t.ravel())
            found |= _codes_key(terms) == _codes_key([v, w])
        assert found, [len(out)]


def test_matcher_recovers_planted_pair_cancelling_at_a_slice(enum3):
    """End to end on a planted target: three random full terms plus a pair
    of copies (v, Z_0 CZ_01 v) with equal coefficients, which cancel at
    offsets 1 and 5. Before the repair reconstruct_block marked both copies
    absent there and valid_term_codes rejected the only candidate."""
    E = enum3
    rng = np.random.default_rng(5)
    x0 = 0
    ords, bases, v, u = _plant(E, rng, 3, x0)
    w = _diagonal_copy(v, 3, lambda y: -1 if ((y & 1) and not (y >> 1 & 1)) else 1)
    Psi = sum(ords) + v + w
    cover = sorted(bases + [u, u])
    hits, st = _recover(E, Psi, cover, x0, _codes_key(ords + [v, w]))
    assert st["blocks"] == [2] and st["reconstructions"] >= 1


def test_matcher_recovers_planted_pair_cancelling_at_the_base_point(enum3):
    """The cancel-at-base multiset T + (b, b): copies v and -(Z_0 CZ_01 v)
    agree at the base point and everywhere except offsets 1 and 5, so the
    merged coefficient is zero, the three ordinary terms alone give the
    base slice, and b is dead but exempt in the family."""
    E = enum3
    rng = np.random.default_rng(7)
    x0 = 0
    ords, bases, v, u = _plant(E, rng, 3, x0)
    w = _diagonal_copy(v, 3, lambda y: -1 if ((y & 1) and not (y >> 1 & 1)) else 1)
    Psi = sum(ords) + v - w
    cover = sorted(bases + [u, u])
    # the family over the distinct states has b dead, and exempt
    from slice_cover import Family
    M = PlantedMatcher(E, Psi, x0)
    fam = Family.from_cover(M.E, sorted(set(cover)))
    pos = sorted(set(cover)).index(u)
    assert fam.has_zero_coefficient(()) and not fam.has_zero_coefficient((pos,))
    hits, st = _recover(E, Psi, cover, x0, _codes_key(ords + [v, w]))
    assert st["blocks"] == [2]


def test_dependent_block_translates_raise_instead_of_dropping(enum3, h6):
    """Two blocks whose base states are Pauli translates of each other (a
    (2, 2, 1) base with both blocks in one Pauli orbit) and a slice whose
    chosen translates span one direction twice: the join keeps the state
    (ambiguous split, no pruning) and the final reconstruction's strict
    coordinate solve raises UnpinnedFamily instead of returning None, which
    dropped the state unrecorded; batch.match_cover records such a run as
    undecided."""
    from slice_cover import Block, Family, apply_pauli
    E = enum3
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    u = 5
    o1 = TermOptions(E.C[:, u], 3, E.F1, E.F2)
    u2 = next(lookup[exact_codes(o1.vecs[4 * k])[0].tobytes()] for k in range(1, 8)
              if lookup[exact_codes(o1.vecs[4 * k])[0].tobytes()] != u)
    o2 = TermOptions(E.C[:, u2], 3, E.F1, E.F2)
    blocks = [Block(o1, 2, 0), Block(o2, 2, 1)]
    # a pinned family over the two distinct states against the target u + u2
    E2 = copy.copy(E)
    E2.psi = E.C[:, u] + E.C[:, u2]
    E2.psi1, E2.psi2 = _to_field(E.F1, E2.psi), _to_field(E.F2, E2.psi)
    fam = Family.from_cover(E2, [u, u2])
    assert fam is not None and fam.kappa == 0
    # translate e of u and the class of the same vector among u2's translates
    e = 3
    k2 = o2.code_of(o1.vecs[4 * e])[0]
    res = 2.0 * o1.vecs[4 * e]
    rhs = (_to_field(E.F1, res), _to_field(E.F2, res), res)
    M = SliceMatcher(E2, 3, native=False)
    assert M._block_coordinates(fam, [], blocks, (), ((e,), (k2,)), rhs) is None
    with pytest.raises(UnpinnedFamily, match="dependent"):
        M._block_coordinates(fam, [], blocks, (), ((e,), (k2,)), rhs, strict=True)
    # independent translates: the coordinates come back, strict or not
    a = M._block_coordinates(fam, [], blocks, (), ((e,), ()), rhs, strict=True)
    assert np.allclose(a, [2.0])
    # a residual with no translate chosen, and one outside the chosen span
    with pytest.raises(UnpinnedFamily, match="no translate"):
        M._block_coordinates(fam, [], blocks, (), ((), ()), rhs, strict=True)
    with pytest.raises(UnpinnedFamily, match="outside the span"):
        M._block_coordinates(fam, [], blocks, (), ((1 if e != 1 else 2,), ()), rhs, strict=True)

    class Raising:
        def run(self, cover, x0):
            raise UnpinnedFamily("planted")

    rec = {"undecided": [], "refused": 0, "matched": 0, "native_runs": 0, "coord_solution_hist": {}, "hits": []}
    h6.batch.match_cover(Raising(), (u, u, u2, u2, 7), rec, [])
    assert len(rec["undecided"]) == len(x0_reps(3)) and rec["matched"] == 0
    assert all(d["reason"] == "UnpinnedFamily: planted" for d in rec["undecided"])


def test_unpinned_family_with_a_block_raises_at_the_final_loop(enum3):
    """_complete raises UnpinnedFamily when a state with a block reaches the
    final loop with parameters left in its family, instead of reconstructing
    the copies at a random member: exercised by handing the loop a
    one-parameter family directly."""
    from slice_cover import Block, Family
    E = enum3
    u, v = 5, 7
    o = TermOptions(E.C[:, u], 3, E.F1, E.F2)
    # the family over (u, v, w) with w in span(u, v): psi = u + v, w = the third state u - v is
    # not a stabilizer state in general, so take a dependent triple from the dictionary instead
    E2 = copy.copy(E)
    idx = None
    for w in range(E.N):
        if w in (u, v):
            continue
        if E.rank_mod2((u, v, w), False) == 2:
            idx = w
            break
    if idx is None:
        pytest.skip("no dependent triple with states 5 and 7")
    E2.psi = E.C[:, u] + E.C[:, v]
    E2.psi1, E2.psi2 = _to_field(E.F1, E2.psi), _to_field(E.F2, E2.psi)
    fam = Family.from_cover(E2, [u, v, idx])
    assert fam.kappa == 1
    M = SliceMatcher(E2, 3, native=False)
    blocks = [Block(o, 2, 0)]
    stats = {"unpinned": 0}
    with pytest.raises(UnpinnedFamily, match="parameter coefficient family"):
        M._complete([u, v, idx], 0, [], blocks, [], [], (), fam, [None], [1, 2, 4], [], stats)
    assert stats["unpinned"] == 1


def test_degenerate_covers_list_the_cancel_at_base_multisets(enum3, h6):
    """driver.degenerate_covers lists T + (b, b) for every full 3-cover T of
    |H>^3 and every b outside T and span(T), and the committed delta list
    is exactly that addition."""
    E = enum3
    covers3, _ = E.covers(3)
    assert len(covers3) == 2
    deg = h6.driver.degenerate_covers(E, 5, covers3, [])
    added = set()
    for T in covers3:
        span = {x for x in range(E.N) if x not in T and E.rank_mod2(tuple(T) + (x,), False) == 3}
        for b in range(E.N):
            if b in T or b in span:
                continue
            ms = tuple(sorted(T + (b, b)))
            assert ms in deg, ms
            added.add(ms)
    with open(os.path.join(H6, "degenerate_covers_v2_delta.json")) as f:
        delta = {tuple(c) for c in json.load(f)["covers"]}
    assert delta == added
    assert len(delta) == 2154


def test_aggregate_fails_a_batch_with_refusals(h6):
    """check_batch reports a nonzero refused count (a stored stage C batch
    with one matched run turned into a refusal)."""
    part = h6.common.load_partition()
    geo = h6.common.batch_geometry(part, 173)
    with open(os.path.join(H6, "results", "batch_173.json")) as f:
        rec = json.load(f)
    problems = []
    h6.aggregate.check_batch(rec, part, geo, problems)
    assert not any("refused" in p for p in problems)
    rec["refused"], rec["matched"] = 1, rec["matched"] - 1
    problems = []
    h6.aggregate.check_batch(rec, part, geo, problems)
    assert any("refused" in p for p in problems)


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
