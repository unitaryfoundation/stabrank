"""Controls for the exact slice-and-lift stage of research/t5_m3_lift."""

import itertools
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "t5_m3_lift"))

from exact import Cyclo, cyclotomic_poly, solve_in_span, stabilizer_form  # noqa: E402
import lift  # noqa: E402
from rank_exclusion import dictionary  # noqa: E402


def test_cyclotomic_polynomials():
    assert cyclotomic_poly(5) == [1, 1, 1, 1, 1]
    assert cyclotomic_poly(9) == [1, 0, 0, 1, 0, 0, 1]
    assert cyclotomic_poly(3) == [1, 1, 1]
    assert cyclotomic_poly(6) == [1, -1, 1]


@pytest.mark.parametrize("n", [3, 5, 9])
def test_field_arithmetic_against_complex(n):
    K = Cyclo(n)
    z = np.exp(2j * np.pi / n)
    rng = np.random.default_rng(n)
    for _ in range(20):
        a = tuple(Fraction(int(x)) for x in rng.integers(-3, 4, size=K.d))
        b = tuple(Fraction(int(x)) for x in rng.integers(-3, 4, size=K.d))
        ca = sum(float(x) * z ** k for k, x in enumerate(a))
        cb = sum(float(x) * z ** k for k, x in enumerate(b))
        assert abs(K.to_complex(K.mul(a, b)) - ca * cb) < 1e-9
        assert abs(K.to_complex(K.conj(a)) - np.conj(ca)) < 1e-9
        if not K.is_zero(a):
            assert K.mul(a, K.inv(a)) == K.one
        n2 = K.norm2(a)
        assert abs(K.to_complex(n2) - abs(ca) ** 2) < 1e-9
    # roots of unity multiply by exponent addition and conjugate to inverses
    for i in range(n):
        for j in range(n):
            assert K.mul(K.root(i), K.root(j)) == K.root(i + j)
        assert K.conj(K.root(i)) == K.root(-i)
        assert K.root_exponent(K.root(i), K.one, n) == i


def test_root_exponent_rejects_unequal_modulus():
    K = Cyclo(5)
    assert K.root_exponent(K.from_int(2), K.one, 5) is None
    assert K.root_exponent(K.mul(K.from_int(3), K.root(2)), K.from_int(3), 5) == 2


def test_solve_in_span_exact():
    K = Cyclo(5)
    cols = [[K.one, K.zero, K.root(1)], [K.zero, K.one, K.root(2)]]
    target = [K.from_int(2), K.root(3), K.add(K.mul(K.from_int(2), K.root(1)), K.root(5))]
    a = solve_in_span(K, cols, target)
    assert a == [K.from_int(2), K.root(3)]
    assert solve_in_span(K, cols, [K.one, K.one, K.one]) is None
    with pytest.raises(ValueError):
        solve_in_span(K, [cols[0], cols[0]], target)


def _flat_state(K, p, m, x0, W, Q, ell, scal):
    k = len(W)
    entries = {}
    for y in itertools.product(range(p), repeat=k):
        x = tuple((x0[c] + sum(y[r] * W[r][c] for r in range(k))) % p for c in range(m))
        q = sum(Q[i][j] * y[i] * y[j] for i in range(k) for j in range(i, k)) + sum(ell[i] * y[i] for i in range(k))
        entries[x] = K.mul(scal, K.root(q % p))
    return entries


def test_stabilizer_form_recognizes_and_rejects():
    K = Cyclo(5)
    W = [(1, 0, 2), (0, 1, 3)]
    Q = [[1, 4], [0, 2]]
    entries = _flat_state(K, 5, 3, (0, 0, 1), W, Q, [3, 0], K.mul(K.from_int(7), K.root(2)))
    form = stabilizer_form(K, 5, 3, entries)
    assert form is not None and form["k"] == 2 and form["W"] == [list(w) for w in W]
    assert form["Q"] == Q and form["l"] == [3, 0]
    # a cubic phase is not a stabilizer state
    cubic = dict(entries)
    for y in itertools.product(range(5), repeat=2):
        x = tuple((0 + y[0] * W[0][c] + y[1] * W[1][c] + (1 if c == 2 else 0)) % 5 for c in range(3))
        cubic[x] = K.mul(cubic[x], K.root(y[0] ** 3 % 5))
    assert stabilizer_form(K, 5, 3, cubic) is None
    # a missing point, an unequal modulus
    short = dict(entries)
    del short[next(iter(short))]
    assert stabilizer_form(K, 5, 3, short) is None
    heavy = dict(entries)
    x = next(iter(heavy))
    heavy[x] = K.mul(heavy[x], K.from_int(2))
    assert stabilizer_form(K, 5, 3, heavy) is None
    # a non-flat support
    notflat = {(0, 0, 0): K.one, (0, 0, 1): K.one, (0, 1, 0): K.one, (1, 1, 1): K.one, (2, 2, 2): K.one}
    assert stabilizer_form(K, 5, 3, notflat) is None


def test_basis_lift_is_the_sector_decomposition():
    """The computational basis of |T5> lifts to the five Z(x)Z sectors of
    |T5>^2 and to nothing else (the patterns are c(z) = c - z)."""
    K = Cyclo(5)
    D = dictionary(5, 1)
    codes = lift.phase_codes(D, 5)
    basis = lift.point_states(codes)
    prob = lift.LiftProblem(5, 1, K, codes, lift.t5_target(K, 1), lift.t5_alpha(K), [tuple(basis)])
    found, stats = prob.search()
    assert stats["patterns_total"] == 3125 and stats["leaf_tests"] == 25 and stats["stabilizer"] == 5
    lifts = prob.lifts(found)
    assert len(lifts) == 1 and prob.check_lift(lifts[0])
    terms = sorted((t["term"]["x0"][1], t["term"]["Q"][0][0], t["term"]["l"][0]) for t in lifts[0])
    assert terms == sorted((c, 3 * c % 5, (-3 * c * c) % 5) for c in range(5))


def test_t5_rank3_decompositions_do_not_lift():
    """The ten rank-3 decompositions of |T5> give no rank-3 decomposition
    of |T5>^2 (whose rank is 5)."""
    K = Cyclo(5)
    D = dictionary(5, 1)
    codes = lift.phase_codes(D, 5)
    target = lift.t5_target(K, 1)
    sets = lift.all_minimal_sets(K, codes, target, 3, 1)
    assert len(sets) == 10
    prob = lift.LiftProblem(5, 1, K, codes, target, lift.t5_alpha(K), sets)
    found, _ = prob.search()
    assert prob.lifts(found) == []


def test_sector_family_product_lifts_and_t5_does_not():
    """From the one census set (the Z(x)Z sectors of |T5>^2): the planted
    product |T5>^2 (x) |+> lifts exactly once (constant patterns), and
    |T5>^2 (x) |T5> does not: every one of the 25 affine patterns fails."""
    K = Cyclo(5)
    D = dictionary(5, 2)
    codes = lift.phase_codes(D, 5)
    target = lift.t5_target(K, 2)
    fam = [lift.HIT]
    plus = [K.one] * 5
    prob = lift.LiftProblem(5, 2, K, codes, target, plus, fam)
    found, stats = prob.search()
    assert stats["leaf_tests"] == 25 and stats["stabilizer"] == 5
    assert all(len({e for _, e in f["choice"]}) == 1 for f in found)
    lifts = prob.lifts(found)
    assert len(lifts) == 1 and prob.check_lift(lifts[0])
    prob = lift.LiftProblem(5, 2, K, codes, target, lift.t5_alpha(K), fam)
    found, stats = prob.search()
    assert stats["leaf_tests"] == 25 and stats["stabilizer"] == 0 and prob.lifts(found) == []


def test_t3_controls_in_q_zeta_9():
    """p = 3 in Q(zeta_9): the basis of |T3> lifts to the carry
    decomposition of |T3>^2; the carry decomposition does not lift to
    |T3>^3 (chi(T3^3) = 8)."""
    K = Cyclo(9)
    D1 = dictionary(3, 1)
    codes1 = lift.phase_codes(D1, 3)
    prob = lift.LiftProblem(3, 1, K, codes1, lift.t3_target(K, 1), lift.t3_alpha(K),
                            [tuple(lift.point_states(codes1))])
    found, _ = prob.search()
    lifts = prob.lifts(found)
    assert len(lifts) == 1 and prob.check_lift(lifts[0])
    carry = tuple(sorted(int(x) for x in D2_index_of_carry(lifts[0], K)))
    D2 = dictionary(3, 2)
    codes2 = lift.phase_codes(D2, 3)
    prob2 = lift.LiftProblem(3, 2, K, codes2, lift.t3_target(K, 2), lift.t3_alpha(K), [carry])
    found2, _ = prob2.search()
    assert prob2.lifts(found2) == []


def D2_index_of_carry(lifted, K):
    """Dictionary indices (two qutrits) of the lifted terms, by matching
    their supports and phases with the dictionary's phase codes."""
    D2 = dictionary(3, 2)
    codes2 = lift.phase_codes(D2, 3)
    out = []
    for f in lifted:
        t = f["term"]
        k = len(t["W"])
        want = {}
        for y in itertools.product(range(3), repeat=k):
            x = tuple((t["x0"][c] + sum(y[r] * t["W"][r][c] for r in range(k))) % 3 for c in range(2))
            q = sum(t["Q"][i][j] * y[i] * y[j] for i in range(k) for j in range(i, k)) + \
                sum(t["l"][i] * y[i] for i in range(k))
            want[x[0] * 3 + x[1]] = q % 3
        for i in range(codes2.shape[0]):
            supp = {int(j): int(codes2[i][j]) - 1 for j in np.flatnonzero(codes2[i])}
            if set(supp) != set(want):
                continue
            base = next(iter(want))
            if all((supp[j] - supp[base]) % 3 == (want[j] - want[base]) % 3 for j in want):
                out.append(i)
                break
    assert len(out) == len(lifted)
    return out
