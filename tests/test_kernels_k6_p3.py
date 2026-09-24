"""The compiled kernels of docs/notes/kernels_k6_p3.md against their Python
references: the k = 6 cover kernel (cover6_pair), the p = 3 stage A matcher
(SliceMatch3Kernel), the compiled dense family solve (dense_solve), and the
32-bit key of cover5_pair. Every kernel is checked on planted instances and
on samples of the real censuses (qubit_H, qubit_T, and the qutrit N orbit);
the Python paths stay the oracle."""

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

import slice_cover  # noqa: E402
from slice_cover import (CoverEnumerator, SliceMatcher, TermOptions, _reduce,  # noqa: E402
                         exact_codes, pair_covers6_native, pair_covers6_reference)

Q3 = os.path.join(ROOT, "research", "qutrit_m4_rank5")

pytestmark = pytest.mark.skipif(
    slice_cover._native_symbol("cover6_pair") is None or slice_cover._native_symbol("dense_solve") is None
    or slice_cover._native_symbol("SliceMatch3Kernel") is None,
    reason="stabrank_core lacks the k = 6 and p = 3 kernels")


@pytest.fixture(scope="module")
def enum3():
    return CoverEnumerator(3)


@pytest.fixture(scope="module")
def enumT3():
    return CoverEnumerator(3, orbit="qubit_T")


@pytest.fixture(scope="module")
def q():
    """The research/qutrit_m4_rank5 modules, imported under their own
    `common` (research/constructions and research/h6_rank5 have modules of
    the same name)."""
    names = ("common", "cover_census", "matcher", "driver")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    sys.path.insert(0, Q3)
    try:
        mods = {k: importlib.import_module(k) for k in names}
    finally:
        sys.path.remove(Q3)
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**mods)


@pytest.fixture(scope="module")
def enum_n(q):
    return q.cover_census.CoverEnumerator3("N", 2)


def _plan(E, i):
    Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
    members, _ = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    return Qi, mask


def _codes_key(terms, codes=exact_codes):
    return sorted(codes(t)[0].tobytes() for t in terms)


def _to_field(F, v):
    re, im = np.round(v.real).astype(np.int64), np.round(v.imag).astype(np.int64)
    assert np.allclose(v, re + 1j * im, atol=1e-9)
    return (re + F.i * im) % F.p


# ------------------------------------------------------------- cover6 ------

@pytest.mark.parametrize("pair", [(3, 932), (582, 1025), (1074, 1043)])
def test_cover6_native_matches_reference_on_h3_pairs(enum3, pair):
    """The compiled 6-cover kernel and pair_covers6_reference list the same
    full 6-covers of |H>^3 through a pivot pair, with the same candidate
    count (the two-functional key has the same accidents in both)."""
    E = enum3
    i, j = pair
    Qi, mask = _plan(E, i)
    got, nc = pair_covers6_native(E, E.native_cover6, i, j, mask)
    ref, ncr = pair_covers6_reference(E, i, j, Qi, mask)
    assert got == ref
    assert nc == ncr
    for c in got:
        assert len(set(c)) == 6 and c[0] == min(i, j)
        assert E.is_cover(c) and E.is_full(c)
    # the enumerator dispatches r = 6 to the kernel and to the reference
    assert E.pair_covers(6, i, j, Qi, mask)[0] == got
    E.native_cover6, saved = None, E.native_cover6
    try:
        assert E.pair_covers(6, i, j, Qi, mask)[0] == got
    finally:
        E.native_cover6 = saved


def test_cover6_native_matches_reference_at_t3(enumT3):
    E = enumT3
    i = int(E.reps[1])                       # 3 is not an orbit root of the T symmetry group
    j = int(E.pivot_plan(i)[1][-1])          # a late partner: few members above it
    Qi, mask = _plan(E, i)
    got, nc = pair_covers6_native(E, E.native_cover6, i, j, mask)
    ref, ncr = pair_covers6_reference(E, i, j, Qi, mask)
    assert got == ref and nc == ncr


def test_cover6_native_matches_reference_on_qutrit_pairs(q, enum_n):
    """The same kernel over the two-qutrit dictionary (the Q(w) field): the
    lightest pivot pairs of the |N>^2 census, native against reference,
    through CoverEnumerator3.pair_covers(6)."""
    E = enum_n
    units = E.units()
    assert E.native_cover6 is not None
    for (i, j, M0) in [units[-1], units[-9], units[600]]:
        Qi, mask = _plan(E, i)
        got, nc, M = E.pair_covers(6, i, j, Qi, mask)
        ref, ncr = pair_covers6_reference(E, i, j, Qi, mask)
        assert got == ref and nc == ncr
        assert M <= M0


def test_cover6_recovers_planted_six_term_targets(enum3):
    """Six random three-qubit states with Gaussian-integer coefficients: the
    kernel run from the two smallest members with every member admissible
    lists the planted 6-set as a full cover."""
    E = enum3
    rng = np.random.default_rng(23)
    found = 0
    for _ in range(4):
        idx = tuple(sorted(int(x) for x in rng.choice(E.N, size=6, replace=False)))
        if np.linalg.matrix_rank(E.C[:, list(idx)], tol=1e-8) < 6:
            continue
        coeffs = rng.integers(1, 4, size=6) * rng.choice([1, -1], size=6) + 1j * rng.integers(-2, 3, size=6)
        psi = E.C[:, list(idx)] @ coeffs
        E2 = copy.copy(E)
        E2.psi = psi
        E2.psi1, E2.psi2 = _to_field(E.F1, psi), _to_field(E.F2, psi)
        E2.Q1, _ = _reduce(E.F1, E.U1, E2.psi1)
        mask = np.ones(E.N, dtype=bool)
        got, _ = pair_covers6_native(E2, E2.native_cover6, idx[0], idx[1], mask)
        assert idx in got
        found += 1
    assert found >= 3


def test_cover5_wide_key_lists_the_same_covers_with_fewer_candidates(enum3):
    E = enum3
    for i, j in [(3, 706), (356, 540)]:
        Qi, mask = _plan(E, i)
        E.wide_key = False
        narrow, nc1 = E._pair_covers_native(i, j, mask, 64)
        E.wide_key = True
        wide, nc2 = E._pair_covers_native(i, j, mask, 64)
        E.wide_key = False
        assert narrow == wide
        assert nc2 <= nc1
        assert E.pair_covers(5, i, j, Qi, mask)[0] == narrow


# ------------------------------------------------------- p = 3 stage A ------

def test_stage_a3_recovers_planted_qutrit_decompositions(q, enum_n):
    """Planted five-term and six-term decompositions of random targets with
    distinct independent base slices at x0, every shape of the structure
    lemma allowed: the compiled stage A and the reference recover the planted
    terms and agree on every hit and count."""
    m = q.matcher
    E = enum_n
    Mn = m.Matcher(E.D, 2, E.F1, E.F2, native=True)
    Mr = m.Matcher(E.D, 2, E.F1, E.F2, native=False)
    assert Mn.native_cls is not None
    rng = np.random.default_rng(7)
    done = 0
    for r in (5, 6, 5, 6, 5, 6):
        x0 = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        base = tuple(sorted(int(x) for x in rng.choice(E.N, size=r, replace=False)))
        if np.linalg.matrix_rank(E.C[:, list(base)], tol=1e-8) < r:
            continue
        terms = [q.driver.random_shape_term(Mn.options(u), x0, rng)[0] for u in base]
        T = np.column_stack(terms)
        if np.linalg.matrix_rank(T, tol=1e-8) < r:
            continue
        coeffs = rng.integers(1, 4, size=r) * rng.choice([1, -1], size=r)
        target = m.vector_target(T @ coeffs.astype(complex), 2, E.F1, E.F2)
        hn, sn = Mn.run(base, x0, target)
        hr, sr = Mr.run(base, x0, target)
        assert sn.get("native") and not sr.get("native")
        want = _codes_key(terms, m.exact_codes)
        assert any(_codes_key(h["terms"], m.exact_codes) == want for h in hn), (base, x0, sn)
        assert (sorted(_codes_key(h["terms"], m.exact_codes) for h in hn)
                == sorted(_codes_key(h["terms"], m.exact_codes) for h in hr))
        assert sn["coord_solutions"] == sr["coord_solutions"]
        assert sn["composite_solutions"] == sr["composite_solutions"]
        for h in hn:
            assert h["exact"] and h["residual"] < 1e-7
        done += 1
    assert done >= 4


def test_stage_a3_matches_reference_on_census_covers(q, enum_n):
    """Real full 5-covers of |N>^2 (the first pivot pairs of the census) at
    the base points (0, 0) and (2, 2): hits, refusals, and the solution
    counts that enter the batch hash agree between the kernel and the
    reference, including the reference's convention of an empty
    coord_solutions when a raw coordinate solve is empty."""
    m = q.matcher
    E = enum_n
    Mn = m.Matcher(E.D, 2, E.F1, E.F2, native=True)
    Mr = m.Matcher(E.D, 2, E.F1, E.F2, native=False)
    target = m.psi_target("N", 2, E.F1, E.F2)
    covers = []
    for (i, j, _) in E.units():
        Qi, mask = _plan(E, i)
        got, _, _ = E.pair_covers(5, i, j, Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= 40:
            break
    covers = covers[:40]
    seen_nonempty = False
    for x0 in ((0, 0), (2, 2)):
        for c in covers:
            hn, sn = Mn.run(c, x0, target)
            hr, sr = Mr.run(c, x0, target)
            assert sn.get("native")
            assert sn["refused"] == sr["refused"]
            assert sn["coord_solutions"] == sr["coord_solutions"]
            assert sn["coord_raw"] == sr["coord_raw"]
            assert sn["composite_solutions"] == sr["composite_solutions"]
            assert (sorted(_codes_key(h["terms"], m.exact_codes) for h in hn)
                    == sorted(_codes_key(h["terms"], m.exact_codes) for h in hr))
            seen_nonempty |= bool(sn["coord_solutions"])
    assert seen_nonempty


def test_stage_a3_declines_dependent_and_repeated_bases(q, enum_n):
    """A dependent base (stage B) and a repeated state (stage C) go to the
    reference path: the kernel declines the first and is not consulted for
    the second, and the stats say so."""
    m = q.matcher
    E = enum_n
    Mn = m.Matcher(E.D, 2, E.F1, E.F2, native=True)
    target = m.psi_target("N", 2, E.F1, E.F2)
    with open(os.path.join(Q3, "degenerate_covers_N.json")) as f:
        deg = json.load(f)["covers"]
    dep = next(tuple(c) for c in deg if len(set(c)) == 5)
    rep = next(tuple(c) for c in deg if len(set(c)) < 5)
    _, sd = Mn.run(dep, (2, 2), target)
    assert not sd.get("native") and sd["kappa"] >= 1
    _, sr = Mn.run(rep, (2, 2), target)
    assert not sr.get("native") and sr["blocks"]


# --------------------------------------------------------- dense solve ------

def _run_both(M, cover, x0):
    hn, sn = M.run(cover, x0)
    os.environ["STABRANK_NO_NATIVE"] = "1"
    try:
        hr, sr = M.run(cover, x0)
    finally:
        del os.environ["STABRANK_NO_NATIVE"]
    return (hn, sn), (hr, sr)


def _same(hn, sn, hr, sr):
    return (sorted(_codes_key(h["terms"]) for h in hn) == sorted(_codes_key(h["terms"]) for h in hr)
            and sn["coord_solutions"] == sr["coord_solutions"] and sn["refused"] == sr["refused"]
            and sn["composite_solutions"] == sr["composite_solutions"] and sn["joined"] == sr["joined"])


def test_dense_native_matches_reference_on_stage_b_covers(enum3):
    """Dependent 5-covers of |H>^3 (kappa = 1, the H^5 stage B shape) through
    SliceMatcher(E, 2) at x0 = 00 with the compiled dense solve and with the
    reference: the same hits and the same solution counts."""
    E = enum3
    with open(os.path.join(ROOT, "research", "h6_rank5", "degenerate_covers_v2.json")) as f:
        deg = json.load(f)["covers"]
    covers = [tuple(c) for c in deg if len(set(c)) == 5]
    M = SliceMatcher(E, 2, native=True)
    assert slice_cover._native_dense() is not None
    used = 0
    for c in (covers[0], covers[len(covers) // 3], covers[-1]):
        (hn, sn), (hr, sr) = _run_both(M, c, 0)
        assert sn["kappa"] == 1
        assert _same(hn, sn, hr, sr), (c, sn, sr)
        used += 1
    assert used == 3


class _PlantedMatcher(SliceMatcher):
    """SliceMatcher against a planted five-qubit target Psi with the sliced
    qubits first (n1 = 2): the family and every slice equation are solved
    against Psi's slices."""

    def __init__(self, E, Psi, x0, n1=2):
        rows = Psi.reshape(1 << n1, -1)
        E2 = copy.copy(E)
        E2.psi = rows[x0]
        E2.psi1, E2.psi2 = _to_field(E.F1, rows[x0]), _to_field(E.F2, rows[x0])
        super().__init__(E2, n1, native=False)
        self._rows = rows

    def rhs(self, x0, x):
        v = self._rows[x]
        return _to_field(self.F1, v), _to_field(self.F2, v), v


def test_dense_native_recovers_a_planted_decomposition_over_a_dependent_base(enum3):
    """Five plane terms of |H>^5-shaped stabilizer states whose base slices
    at 00 are a dependent 5-cover's states (kappa = 1 at the base) with
    Gaussian-integer coefficients: the matcher through the compiled dense
    solve and through the reference recover the planted decomposition from
    the one-parameter family and agree on everything."""
    E = enum3
    with open(os.path.join(ROOT, "research", "h6_rank5", "degenerate_covers_v2.json")) as f:
        deg = json.load(f)["covers"]
    rng = np.random.default_rng(3)
    recovered = 0
    for c in [tuple(x) for x in deg if len(set(x)) == 5][:400:133]:
        assert E.rank_mod2(c, False) == 4
        terms = []
        for u in c:
            o = TermOptions(E.C[:, u], 3, E.F1, E.F2)
            c1, c2 = (int(v) for v in rng.integers(0, 32, size=2))
            sign = int(rng.integers(0, 2))
            t = np.zeros((4, 8), dtype=complex)
            t[0], t[1], t[2], t[3] = o.u, o.vecs[c1], o.vecs[c2], o.vecs[o.compose(c1, c2, sign)]
            terms.append(t.ravel())
        coeffs = rng.integers(1, 4, size=5) * rng.choice([1, -1], size=5) + 1j * rng.integers(-1, 2, size=5)
        Psi = np.column_stack(terms) @ coeffs
        M = _PlantedMatcher(E, Psi, 0)
        (hn, sn), (hr, sr) = _run_both(M, c, 0)
        assert sn["kappa"] == 1 and not sn["refused"]
        want = _codes_key(terms)
        assert any(_codes_key(h["terms"]) == want for h in hn), (c, sn)
        assert _same(hn, sn, hr, sr)
        recovered += 1
    assert recovered >= 3


def test_dense_native_kappa_one_on_t_stage_b(enumT3):
    E = enumT3
    with open(os.path.join(ROOT, "research", "t5_rank5", "degenerate5.json")) as f:
        deg = json.load(f)["covers"]
    covers = [tuple(c) for c in deg if len(set(c)) == 5]
    M = SliceMatcher(E, 2, native=True)
    for c in (covers[1], covers[len(covers) // 2]):
        (hn, sn), (hr, sr) = _run_both(M, c, 0)
        assert sn["kappa"] == 1
        assert _same(hn, sn, hr, sr), (c, sn, sr)
