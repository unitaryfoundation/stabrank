"""The stage B6 fresh-term filter of the rank-6 exclusion of |N>^4
(research/n4_rank6/filters6.py) against the reference slice solver on the
bases the 2026-09-25 pod run left undecided: a coordinate-slice solution
that pins the family parameter at the root of the fresh term's coefficient
makes the fresh term contribute nothing, so every one of its 28 codes
solves the slice, and the filter's list must carry all of them as the
reference's does (the run_b6 assertion compares the raw counts). Planted
controls over the same bases: a five-term decomposition that leaves the
fresh term out (the zero-coefficient shape itself) and full six-term
decompositions that must be recovered."""

import importlib
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sympy")   # verify_challenge needs the challenge extra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

import slice_cover  # noqa: E402

N4 = os.path.join(ROOT, "research", "n4_rank6")

pytestmark = pytest.mark.skipif(slice_cover._native_symbol("dense_solve") is None,
                                reason="stabrank_core lacks the compiled dense solve")

# the 24 covers of the 23 undecided batches, with the reference's raw
# coordinate-slice counts (the filter reported 27 fewer per zero-fresh combination)
UNDECIDED = {
    (8, 41, 117, 118, 249, 334): 56,
    (2, 30, 33, 113, 117, 345): 280,
    (8, 32, 113, 117, 154, 335): 60,
    (8, 43, 117, 167, 273, 321): 58,
    (8, 36, 225, 249, 333, 345): 454,
}


@pytest.fixture(scope="module")
def n4():
    """The research/n4_rank6 modules under their own `common` and the
    rank-5 `matcher` (research/constructions and research/h6_rank5 have
    modules of the same names), with sys.path and sys.modules restored."""
    names = ("common", "matcher", "cover_census", "driver", "filters6", "stages", "probe6", "invisible3")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    path0 = list(sys.path)
    sys.path.insert(0, N4)
    try:
        mods = {k: importlib.import_module(k) for k in ("common", "matcher", "filters6", "stages", "probe6", "invisible3")}
    finally:
        sys.path[:] = path0
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**mods)


@pytest.fixture(scope="module")
def env(n4):
    E = n4.common.make_enumerator(native=True)
    M = n4.common.new_matcher(E, native=True)
    target = n4.common.target_of(E)
    Fl = n4.filters6.Filters(M)
    n4.stages.set_max_cand(2_000_000)
    return types.SimpleNamespace(E=E, M=M, target=target, Fl=Fl)


def _reference_slice(n4, env, cover, target, e):
    distinct = sorted(set(cover))
    fam = env.Fl.family(distinct, target, n4.common.X0)
    arrays = [env.M.options(u).arrays() for u in distinct]
    rhs = target.rhs(n4.common.add(n4.common.X0, e))
    return fam, n4.matcher.solve_slice3(arrays, [], fam, rhs, env.M.rng)


def _combos(sols):
    return {tuple(int(c) for c in combo) for combo, _, _ in sols}


@pytest.mark.parametrize("cover,count", sorted(UNDECIDED.items()))
def test_filter_lists_the_zero_fresh_coefficient_codes(n4, env, cover, count):
    """The filter's list at each coordinate slice equals the reference's,
    fresh term's 28 codes at the pinned parameter included, and run_b6
    decides the base by the complete search instead of raising."""
    X0 = n4.common.X0
    for e in (n4.matcher.E1, n4.matcher.E2):
        sols, st = env.Fl.b6_slice(cover, X0, env.target, e)
        fam, ref = _reference_slice(n4, env, cover, env.target, e)
        assert fam.kappa == 1
        assert _combos(sols) == _combos(ref)
        assert len(sols) == len(ref) == count
        j = sorted(set(cover)).index(st["fresh"])
        by5 = {}
        for combo in _combos(sols):
            by5.setdefault(combo[:j] + combo[j + 1:], set()).add(combo[j])
        # at least one five-term combination pins lambda at the root of the
        # fresh coefficient and carries all 28 fresh codes
        assert any(len(codes) == 28 for codes in by5.values())
    hits, st, extra = n4.stages.run_b6(env.M, env.Fl, env.target, cover, 1)
    assert extra["path"] == "filter+reference"
    assert not st["refused"]
    assert st["coord_raw"] == [count, count]
    assert hits == []


def _plant(n4, env, states, rng, seed_terms=None):
    """Random stabilizer terms with base slices the given states at X0 and
    nonzero integer coefficients, retried until independent."""
    X0 = n4.common.X0
    for _ in range(200):
        terms = [n4.probe6.random_shape_term(env.M.options(u), X0, rng)[0] for u in states]
        T = np.column_stack(terms)
        if np.linalg.matrix_rank(T, tol=1e-8) < len(terms):
            continue
        coeffs = (rng.integers(1, 4, size=len(terms)) * rng.choice([1, -1], size=len(terms))).astype(complex)
        vec = (T @ coeffs).reshape(9, -1)
        if np.any(np.linalg.norm(vec, axis=1) < 1e-9):
            continue
        return terms, coeffs
    raise AssertionError("no independent planted instance")


def test_planted_zero_fresh_coefficient_agrees(n4, env):
    """A planted five-term target over the base minus its fresh term: the
    six-set's family passes through the zero fresh coefficient, the
    reference lists the planted combination with all 28 fresh codes, and
    the filter's list is the same."""
    cover = (8, 41, 117, 118, 249, 334)
    X0 = n4.common.X0
    distinct = sorted(cover)
    fam0 = env.Fl.family(distinct, env.target, X0)
    j, _, _ = env.Fl.plan_b6(distinct, fam0)        # the plan depends on the dependency's zero pattern only
    others = [u for i, u in enumerate(distinct) if i != j]
    rng = np.random.default_rng(20260926)
    for _ in range(20):
        terms, coeffs = _plant(n4, env, others, rng)
        tgt = n4.invisible3.planted_target(terms, coeffs, env.M.F1, env.M.F2)
        fam = env.Fl.family(distinct, tgt, X0)
        if fam is not None and fam.kappa == 1 and fam.kappa1 == 1:
            break
    else:
        pytest.fail("no planted five-term target with a one-parameter family over the six-set")
    sols, st = env.Fl.b6_slice(cover, X0, tgt, n4.matcher.E1)
    assert st["fresh"] == distinct[j]
    _, ref = _reference_slice(n4, env, cover, tgt, n4.matcher.E1)
    assert _combos(sols) == _combos(ref)
    by5 = {}
    for combo in _combos(sols):
        by5.setdefault(combo[:j] + combo[j + 1:], set()).add(combo[j])
    assert any(len(codes) == 28 for codes in by5.values())
    hits, st, _ = n4.stages.run_b6(env.M, env.Fl, tgt, cover, 1)
    assert not st["refused"]
    assert st["coord_raw"][0] == len(ref)
    # a five-term decomposition is not a rank-6 hit
    assert hits == []


@pytest.mark.parametrize("cover", [(8, 41, 117, 118, 249, 334), (2, 30, 33, 113, 117, 345),
                                   (8, 36, 225, 249, 333, 345)])
def test_planted_rank6_recovered(n4, env, cover):
    """A planted six-term decomposition over an undecided base is among the
    hits of run_b6 with the filter."""
    rng = np.random.default_rng(sum(cover))
    terms, coeffs = _plant(n4, env, sorted(cover), rng)
    tgt = n4.invisible3.planted_target(terms, coeffs, env.M.F1, env.M.F2)
    hits, st, extra = n4.stages.run_b6(env.M, env.Fl, tgt, cover, 1)
    assert not st["refused"]
    assert extra["path"] == "filter+reference"
    want = n4.common.codes_key(terms)
    assert any(n4.common.codes_key(h["terms"]) == want for h in hits)
