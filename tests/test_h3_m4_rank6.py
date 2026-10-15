"""Controls for the invisible-flat matcher at p = 3 of the rank-6 exclusion
design of |H3>^4 (research/h3_m4_rank6/invisible_p3.py,
docs/notes/h3_m4_rank6_design.md): the geometry of the base point (0, 0),
planted decompositions of every kind the case split names recovered on
flats of every coordinate class (stage (beta'): kinds a, d, r, h; stage
(gamma): kind a on flat pairs of every shape, the ray kind, and the seven
shared-line kinds same, same1, same2, samec, cancel, cancel1, cancel0),
real census bases dying without a hit or a refusal, and the refusal of
three fresh terms at one point. The whole module runs in under two
minutes on one core."""

import importlib
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sympy")   # verify_challenge needs the challenge extra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H3 = os.path.join(ROOT, "research", "h3_m4_rank6")
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

import slice_cover  # noqa: E402

pytestmark = pytest.mark.skipif(slice_cover._native_symbol("dense_solve") is None,
                                reason="stabrank_core lacks the compiled dense solve")


@pytest.fixture(scope="module")
def h3():
    """The research/h3_m4_rank6 modules under their own `common` and the
    rank-5 `matcher` (research/constructions, research/h6_rank5 and
    research/n4_rank6 have modules of the same names), with sys.path and
    sys.modules restored."""
    names = ("common", "matcher", "cover_census", "invisible_p3", "probe")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    path0 = list(sys.path)
    sys.path.insert(0, H3)
    try:
        mods = {k: importlib.import_module(k) for k in ("common", "matcher", "invisible_p3")}
    finally:
        sys.path[:] = path0
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**mods)


@pytest.fixture(scope="module")
def env(h3):
    E = h3.common.make_enumerator(native=True)
    Mt = h3.common.new_matcher(E, native=True)
    cell = h3.common.CELL
    IM = h3.invisible_p3.InvisibleMatcherP3(Mt, cell)
    lists = h3.common.rank5_lists()
    rng = np.random.default_rng(20261001)
    P = h3.invisible_p3.Plants(E, Mt, cell, lists, rng)
    return types.SimpleNamespace(E=E, Mt=Mt, cell=cell, IM=IM, lists=lists, P=P, target=h3.common.target_of(E))


def _recover(h3, env, base, terms, coeffs, fn, cap=60.0):
    e = h3.invisible_p3.run_plant(env.IM, env.Mt, base, terms, coeffs, fn, cap)
    assert e["status"] == "recovered", e
    return e


def test_geometry(h3):
    cell = h3.common.CELL
    assert cell.x0 == (0, 0)
    assert len(cell.flats) == 16 and len(cell.flat_pairs) == 136
    assert sum(1 for pts in cell.flats.values() if len(pts) == 3) == 8
    for pts in cell.flats.values():
        assert cell.x0 not in pts
    # the slice ratio from (0, 0) is never 1: 1 / (1 + sqrt 3) at the four points with
    # one zero coordinate, 1 / (1 + sqrt 3)^2 at the four with none
    r = sorted({round(h3.common.ratio(y), 9) for y in cell.others})
    assert r == [round(1 / (1 + np.sqrt(3)) ** 2, 9), round(1 / (1 + np.sqrt(3)), 9)]
    c1, c2 = cell.coord_points
    by = [sum(1 for c in (c1, c2) if c in pts) for pts in cell.flats.values()]
    assert (by.count(0), by.count(1), by.count(2)) == (9, 6, 1)
    # the processing order puts the points on no invisible flat first and never offers more
    # than two fresh terms; a two-fresh-term point arises only for two equal flats
    n_scan2 = 0
    for f, g in cell.flat_pairs:
        F, G = cell.flats[f], cell.flats[g]
        order, present, modes = env_order(h3, cell, [F, G])
        assert len(order) == 8 and set(order) == set(cell.others)
        free = [y for y in order if not present[y]]
        assert order[:len(free)] == free and all(modes[y] == "exact" for y in free)
        assert set(modes.values()) <= {"exact", "scan", "scan2"}
        has2 = any(m == "scan2" for m in modes.values())
        assert has2 == (F == G)
        n_scan2 += has2
    assert n_scan2 == 16


def env_order(h3, cell, flats):
    IM = types.SimpleNamespace(cell=cell)
    return h3.invisible_p3.InvisibleMatcherP3.order(IM, flats)


# one flat of each coordinate class: a point missing both coordinate points, a line
# containing one, and the line containing both
BETA_FLATS = ("p22", "L10_011121", "L12_011022")


@pytest.mark.parametrize("flat", BETA_FLATS)
@pytest.mark.parametrize("kind", ("a", "d", "r"))
def test_beta_plants(h3, env, flat, kind):
    assert flat in env.cell.flats
    base, terms, coeffs = env.P.beta(flat, kind)
    e = _recover(h3, env, base, terms, coeffs, lambda tgt, b=base: env.IM.run_one(b, flat, tgt))
    assert e["hits"] >= 1


def test_beta_block_hidden(h3, env):
    """The invisible term's slice at its first point hides in the span of
    a block copy's slice: its coefficient stays a family parameter until
    the reconstruction."""
    flat = "p11"
    base, terms, coeffs = env.P.beta(flat, "h")
    _recover(h3, env, base, terms, coeffs, lambda tgt: env.IM.run_one(base, flat, tgt), cap=90.0)


GAMMA_PAIRS_A = (("p11", "p11"), ("p11", "p22"), ("p12", "L01_101112"), ("p22", "L10_011121"),
                 ("L01_101112", "L12_021120"), ("L11_011220", "L11_011220"))


@pytest.mark.parametrize("pair", GAMMA_PAIRS_A)
def test_gamma_plants_a(h3, env, pair):
    for name in pair:
        assert name in env.cell.flats
    base, terms, coeffs = env.P.gamma(pair, "a")
    _recover(h3, env, base, terms, coeffs, lambda tgt: env.IM.run_two(base, pair, tgt))


def test_gamma_ray(h3, env):
    """A point term at a line term's least point, parallel to the line
    term's slice there: the order births the line term at the line's other
    points first, and the point term's scan at the shared point finds a
    residual parallel to an already born term's slice."""
    cell = env.cell
    pair = None
    for f, g in cell.flat_pairs:
        F, G = cell.flats[f], cell.flats[g]
        if len(F) == 1 and len(G) == 3 and F[0] == min(G):
            pair = (f, g)
            break
    assert "ray" in env.P.gamma_kinds(pair)
    base, terms, coeffs = env.P.gamma(pair, "ray")
    _recover(h3, env, base, terms, coeffs, lambda tgt: env.IM.run_two(base, pair, tgt))


@pytest.mark.parametrize("kind", ("same", "same1", "same2", "samec", "cancel", "cancel1", "cancel0"))
def test_gamma_shared_line(h3, env, kind):
    """Two line terms on one line sharing their slice at the line's first
    point, including the pairs whose coefficients cancel there (the
    residual vanishes at the common first point and the pair is carried
    to the line's second point)."""
    pair = ("L12_011022", "L12_011022")       # the line through both coordinate points
    base, terms, coeffs = env.P.gamma(pair, kind)
    if kind.startswith("cancel"):
        assert abs(coeffs[-1] + coeffs[-2]) < 1e-12
    e = _recover(h3, env, base, terms, coeffs, lambda tgt: env.IM.run_two(base, pair, tgt))
    if kind.startswith("cancel"):
        assert e["hist"].split(",")[-1] != "0"      # a pair solution was formed


def test_gamma_shared_line_other_line(h3, env):
    pair = ("L11_011220", "L11_011220")
    for kind in ("cancel", "samec"):
        base, terms, coeffs = env.P.gamma(pair, kind)
        _recover(h3, env, base, terms, coeffs, lambda tgt, b=base: env.IM.run_two(b, pair, tgt))


def test_real_bases_die_without_hits(h3, env):
    """Real census bases against psi_4 = |H3>^4: no hit, no refusal, and
    every run dies at its first exact point."""
    covers5 = env.lists["covers5"]
    covers4 = env.lists["covers4"]
    for base in (covers5[0], covers5[len(covers5) // 2], covers5[-1]):
        for flat in BETA_FLATS:
            hits, st = env.IM.run_one(tuple(base), flat, env.target)
            assert not hits and not st["refused"], (base, flat, st)
    for base in (covers4[0], covers4[-1]):
        for pair in GAMMA_PAIRS_A[:3]:
            hits, st = env.IM.run_two(tuple(base), pair, env.target)
            assert not hits and not st["refused"], (base, pair, st)


def test_three_fresh_terms_raise(h3, env):
    """Three invisible point terms at one point (a configuration outside
    the design, which Fact B excludes): the matcher raises at that point
    instead of enumerating a rank-3 residual."""
    ip = h3.invisible_p3
    cell, Mt, rng = env.cell, env.Mt, env.P.rng
    base = env.P.pick(env.P.indep4)
    F = cell.flats["p11"]
    while True:
        invs = [ip.invisible_term(Mt, cell, "p11", rng) for _ in range(3)]
        if len({v for _, v in invs}) == 3:
            break
    terms, coeffs = ip.plant(Mt, cell, base, [(t, None) for t, _ in invs], rng)
    tgt = ip.planted_target(terms, coeffs, Mt.F1, Mt.F2)
    with pytest.raises(h3.matcher.UnpinnedFamily, match="3 fresh"):
        env.IM.run(base, [F, F, F], tgt)


def test_flat_through_base_point_raises(h3, env):
    base = tuple(env.lists["covers5"][0])
    with pytest.raises(h3.matcher.UnpinnedFamily):
        env.IM.run(base, [((0, 0), (1, 1), (2, 2))], env.target)
