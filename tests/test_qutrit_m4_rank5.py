"""Controls for the stage C path of the qutrit rank-5 pipeline
(research/qutrit_m4_rank5): the strict block reconstruction, the batch
record of a raising run and of a deadline abort, the cover classes behind
the stage C partition, and the cancel-at-base multisets of the list."""

import importlib
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sympy")   # verify_challenge needs the challenge extra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "research", "qutrit_m4_rank5")


@pytest.fixture(scope="module")
def q():
    """The research/qutrit_m4_rank5 modules, imported under their own
    `common` (research/constructions and research/h6_rank5 have modules of
    the same name)."""
    names = ("common", "cover_census", "matcher", "driver", "batch")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    sys.path.insert(0, Q)
    try:
        mods = {k: importlib.import_module(k) for k in names}
    finally:
        sys.path.remove(Q)
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**mods)


@pytest.fixture(scope="module")
def enum_n(q):
    return q.cover_census.CoverEnumerator3("N", 2)


@pytest.fixture(scope="module")
def matcher_n(q, enum_n):
    E = enum_n
    return q.matcher.Matcher(E.D, 2, E.F1, E.F2)


def _rhs(q, M, v):
    return (q.matcher.field_vector(v, M.F1), q.matcher.field_vector(v, M.F2), v)


def test_dependent_block_translates_raise_instead_of_dropping(q, matcher_n):
    """Two blocks whose base states are Pauli translates of each other (a
    (2, 2, 1) base with both blocks in one Pauli orbit) and a slice whose
    chosen translates hold one vector twice: the join keeps the state
    (ambiguous split, no pruning) and the final loop's strict coordinate
    solve raises UnpinnedFamily instead of returning None."""
    m = q.matcher
    M = matcher_n
    u1 = 36
    o1 = M.options(u1)
    u2 = next(k for k in (M.index_of(t) for t in o1.T[1:]) if k != u1)
    o2 = M.options(u2)
    blocks = [m.Block(o1, 2, 0), m.Block(o2, 2, 1)]
    M._proj = m._ProjectorCache(blocks)
    psi = M.C[:, u1] + M.C[:, u2]
    fam = m.family_from(M.U1[[u1, u2]], M.U2[[u1, u2]], M.C[:, [u1, u2]], *_rhs(q, M, psi))
    assert fam is not None and fam.kappa == 0
    e = 3
    k2 = o2.code_of(o1.vecs[3 * e])[0]           # the class of the same vector among u2's translates
    rhs = _rhs(q, M, 2.0 * o1.vecs[3 * e])
    assert M._block_coordinates(fam, [], blocks, (), ((e,), (k2,)), rhs) is None
    with pytest.raises(m.UnpinnedFamily, match="dependent"):
        M._block_coordinates(fam, [], blocks, (), ((e,), (k2,)), rhs, strict=True)
    a = M._block_coordinates(fam, [], blocks, (), ((e,), ()), rhs, strict=True)
    assert np.allclose(a, [2.0])
    with pytest.raises(m.UnpinnedFamily, match="no translate"):
        M._block_coordinates(fam, [], blocks, (), ((), ()), rhs, strict=True)
    with pytest.raises(m.UnpinnedFamily, match="outside the span"):
        M._block_coordinates(fam, [], blocks, (), ((1 if e != 1 else 2,), ()), rhs, strict=True)


def test_batch_records_a_raising_run_and_a_deadline_abort(q):
    """batch.match_cover records a run that raises UnpinnedFamily as
    undecided with the exception's message, and a cover reached after the
    batch's --max-seconds deadline as undecided with the reason "not run"
    without running it."""
    m = q.matcher

    class Raising:
        budget = None

        def run(self, cover, x0, target):
            raise m.UnpinnedFamily("planted")

    rec = {"undecided": [], "refused": 0, "matched": 0, "coord_solution_hist": {}, "hits": []}
    q.batch.match_cover("N", Raising(), None, (1, 1, 2, 3, 4), (0, 0), rec, [])
    assert rec["matched"] == 0 and len(rec["undecided"]) == 1
    assert rec["undecided"][0]["reason"] == "UnpinnedFamily: planted"

    class Expired:
        budget = m.Budget(seconds=0.0)

        def run(self, cover, x0, target):
            raise AssertionError("a cover past the deadline must not run")

    rec = {"undecided": [], "refused": 0, "matched": 0, "coord_solution_hist": {}, "hits": []}
    q.batch.match_cover("N", Expired(), None, (1, 1, 2, 3, 4), (0, 0), rec, [])
    assert rec["undecided"][0]["reason"].startswith("not run")
    assert q.batch.past_deadline(Expired())
    assert not q.batch.past_deadline(Raising())


def test_cover_classes_and_the_cancel_at_base_multisets(q, enum_n, matcher_n):
    """driver.cover_class marks the (2, 2, 1) covers whose blocks admit
    dependent translates and the (2, 1, 1, 1) covers with a dependent
    distinct set, and the stored N list holds T + (b, b) for a state b
    outside the span of a full 3-cover T (the cancel-at-base case)."""
    d = q.driver
    E, M = enum_n, matcher_n
    assert d.cover_class(E, M, (36, 36, 72, 117, 117)) == "(2, 2, 1) dependent"
    assert d.cover_class(E, M, (1, 2, 3, 4, 5)) == "B"
    covers, doc = q.common.load_degenerate("N")
    classes = {c: d.cover_class(E, M, c) for c in covers if q.common.multiplicity_pattern(c) in ((2, 2, 1), (3, 1, 1))}
    assert sum(cls == "(2, 2, 1) dependent" for cls in classes.values()) == 27
    assert all(" dependent" not in cls for c, cls in classes.items() if q.common.multiplicity_pattern(c) == (3, 1, 1))
    covers3 = [tuple(c) for c in q.common.load_census("N")["covers3"]]
    T = covers3[0]
    span = set(E.in_span(T))
    b = next(u for u in range(E.N) if u not in set(T) | span)
    assert tuple(sorted(T + (b, b))) in set(covers)
    assert doc["stage_c_covers"] == 16181 and doc["stage_b_covers"] == 12175
