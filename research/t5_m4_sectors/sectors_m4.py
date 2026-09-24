"""The Z^4 eigensectors of |T5>^4 written down exactly, their single class,
their slices, and the five sub-sector products.

Sector c is the restriction of the phase pattern w^(x1^3 + x2^3 + x3^3 +
x4^3) (entries of 25 |T5>^4) to the hyperplane x1 + x2 + x3 + x4 = c. The
Clifford SUM_{1->4} SUM_{2->4} SUM_{3->4} maps it to u_c (x) |c> with

    u_c(x, y, z) = w^(f_c(x, y, z)),   f_c = x^3 + y^3 + z^3 + (c - x - y - z)^3,

a three-ququint state with full support, 125 entries in Q(zeta_5), and
chi(sector c) = chi(u_c). This script checks, in exact arithmetic on the
exponents mod 5:

1. u_c(x, y, z) is the sector read off the hyperplane (the numerical
   projection of |T5>^4 agrees), and its weight is 1/5.
2. One class: f_{c+1}(x, y, z) - f_c(x - 1, y, z) = 3x^2 - 3x + 1, so
   u_{c+1} = D X_1 u_c with D the diagonal Clifford w^(3x^2 - 3x + 1) and
   X_1 the shift of the first ququint; the five sectors are Clifford
   images of one another and have the same stabilizer rank. The cubic part
   of every f_c is 3(xyz - (x + y + z)(xy + yz + zx)), symmetric under S_3.
3. Slices: u_c(x, y, a) = w^(a^3) g_{c-a}(x, y) with g_d(x, y) = w^(x^3 +
   y^3 + (d - x - y)^3) the Z^3 sector of |T5>^3 at eigenvalue w^d
   (sector_census.sector_codes((1, 1, 1), d)).
4. The five sub-sector products: u_c = sum_{a} (line state on x + y = a
   with phase w^(3a x^2 - 3a^2 x + a^3)) (x) (line state in (z, c - a - z)
   with phase w^(3(c-a) z^2 - 3(c-a)^2 z + (c-a)^3)), read in the (x, y, z)
   coordinates; the five terms are stabilizer states (checked by the
   witness converter), linearly independent, and no four of them span u_c.
5. The same for the two other strings with five rank-one sub-sector terms,
   Z Z Z^2 Z^2 and Z Z Z^4 Z^4 (hyperplanes x1 + x2 + b x3 + b x4 = c, b = 2,
   4): their x4 = a slices are the Z Z Z^b sectors of |T5>^3, and their
   x1 = a slices the Z Z^b Z^b sectors, which are the classes {1,1,b} and
   {1,b,b} of sector_census.py.

Run: sectors_m4.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, HERE)
from sector_census import sector_codes  # noqa: E402

W5 = np.exp(2j * np.pi / 5)


def f_exp(c, x, y, z, b=1):
    """Exponent of sector c of the string Z Z Z^b Z^b at (x, y, z), with
    x4 = (c - x - y) / b - z on the hyperplane x + y + b (z + x4) = c."""
    binv = pow(b, -1, 5)
    x4 = ((c - x - y) * binv - z) % 5
    return (x ** 3 + y ** 3 + z ** 3 + x4 ** 3) % 5


def u_vec(c, b=1):
    return np.array([W5 ** f_exp(c, x, y, z, b) for x, y, z in itertools.product(range(5), repeat=3)])


def check_projection(c, b=1):
    t = np.array([W5 ** ((x ** 3) % 5) for x in range(5)]) / np.sqrt(5)
    psi = t
    for _ in range(3):
        psi = np.kron(psi, t)
    psi = psi.reshape((5,) * 4)
    binv = pow(b, -1, 5)
    u = u_vec(c, b).reshape(5, 5, 5)
    weight = 0.0
    for x, y, z, x4 in itertools.product(range(5), repeat=4):
        on = (x + y + b * z + b * x4) % 5 == c
        if on:
            assert x4 == ((c - x - y) * binv - z) % 5
            assert abs(psi[x, y, z, x4] * 25 - u[x, y, z]) < 1e-12
            weight += abs(psi[x, y, z, x4]) ** 2
    assert abs(weight - 0.2) < 1e-12
    return True


def poly_degree(exp_table):
    """Degree of the exponent table (a function F_5^3 -> F_5) as a polynomial
    of degree at most 4 in each variable, by interpolation over the monomial
    basis; the degree is the largest total degree with a nonzero coefficient."""
    pts = list(itertools.product(range(5), repeat=3))
    monos = list(itertools.product(range(5), repeat=3))
    A = np.array([[pow(x, i, 5) * pow(y, j, 5) * pow(z, k, 5) % 5 for (i, j, k) in monos] for (x, y, z) in pts])
    rhs = np.array([exp_table[p] for p in pts])
    coef = solve_mod5(A, rhs)
    return max((i + j + k for (i, j, k), cf in zip(monos, coef) if cf), default=-1), dict(zip(monos, coef))


def solve_mod5(A, b):
    """Exact solve of the square system A x = b over F_5 (A invertible)."""
    A = [[int(v) % 5 for v in row] + [int(bb) % 5] for row, bb in zip(A, b)]
    n = len(A)
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col])
        A[col], A[piv] = A[piv], A[col]
        inv = pow(A[col][col], -1, 5)
        A[col] = [(v * inv) % 5 for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]
                A[r] = [(v - f * w) % 5 for v, w in zip(A[r], A[col])]
    return [A[r][n] for r in range(n)]


def check_one_class(b=1):
    """f_{c+1}(x, y, z) - f_c(x - 1, y, z) is a quadratic (a Clifford phase)."""
    for c in range(5):
        diff = {}
        for x, y, z in itertools.product(range(5), repeat=3):
            diff[(x, y, z)] = (f_exp((c + 1) % 5, x, y, z, b) - f_exp(c, (x - 1) % 5, y, z, b)) % 5
        deg, coef = poly_degree(diff)
        assert deg <= 2, (c, deg)
        if b == 1:
            want = {(2, 0, 0): 3, (1, 0, 0): 2, (0, 0, 0): 1}       # 3x^2 - 3x + 1
            assert {m: v for m, v in coef.items() if v} == want, coef
    return True


def cubic_part(c, b=1):
    tab = {(x, y, z): f_exp(c, x, y, z, b) for x, y, z in itertools.product(range(5), repeat=3)}
    _, coef = poly_degree(tab)
    return {m: v for m, v in coef.items() if v and sum(m) == 3}


def check_slices(b=1):
    """u_c(x, y, a) = w^(a^3) g(x, y) with g the Z Z Z^b sector at the right
    eigenvalue; u_c(a, y, z) is the Z Z^b Z^b sector."""
    for c in range(5):
        u = u_vec(c, b).reshape(5, 5, 5)
        for a in range(5):
            # x4 slice at z = a: plane x + y + b x4 = c - b a in (x, y, x4)
            g = W5 ** (sector_codes((1, 1, b), (c - b * a) % 5).astype(int) - 1).reshape(5, 5)
            # sector_codes reads the plane by (x, y) with x4 = (d - x - y)/b: our z-slice at
            # z = a leaves x4 = (c - x - y)/b - a, i.e. the plane at d = c - b a; the a^3 factor
            assert np.allclose(u[:, :, a], W5 ** ((a ** 3) % 5) * g)
            # x slice at x = a: plane y + b z + b x4 = c - a in (y, z, x4), string Z Z^b Z^b
            g2 = W5 ** (sector_codes((1, b, b), (c - a) % 5).astype(int) - 1).reshape(5, 5)
            assert np.allclose(u[a, :, :], W5 ** ((a ** 3) % 5) * g2)
    return True


def sub_sector_terms(c, b=1):
    """The five products (Z Z sector a on copies 1, 2) (x) (Z^b Z^b sector on
    copies 3, 4), as vectors in the (x, y, z) coordinates; returns the list."""
    binv = pow(b, -1, 5)
    terms = []
    for a in range(5):
        v = np.zeros(125, dtype=complex)
        for x, y, z in itertools.product(range(5), repeat=3):
            if (x + y) % 5 != a:
                continue
            x4 = ((c - x - y) * binv - z) % 5
            v[25 * x + 5 * y + z] = W5 ** ((x ** 3 + y ** 3 + z ** 3 + x4 ** 3) % 5)
        terms.append(v)
    return terms


def check_sub_sectors(c, b=1):
    from to_witness import term_from_vector
    u = u_vec(c, b)
    terms = sub_sector_terms(c, b)
    assert np.allclose(sum(terms), u)
    for v in terms:
        term_from_vector(v / np.linalg.norm(v), 5, 3)          # raises if not a stabilizer state
    A = np.column_stack(terms)
    assert np.linalg.matrix_rank(A, tol=1e-9) == 5
    for drop in range(5):
        B = np.delete(A, drop, axis=1)
        x, *_ = np.linalg.lstsq(B, u, rcond=None)
        assert np.linalg.norm(B @ x - u) > 1e-3, "four sub-sector terms span the sector"
    return True


def main():
    sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
    for b in (1, 2, 4):
        name = {1: "Z Z Z Z", 2: "Z Z Z^2 Z^2", 4: "Z Z Z^4 Z^4"}[b]
        for c in range(5):
            check_projection(c, b)
            check_sub_sectors(c, b)
        check_one_class(b)
        check_slices(b)
        print(f"{name}: five sectors, each 125 entries in Q(zeta_5) with full support and weight 1/5; "
              f"one class (u_(c+1) = D X_1 u_c with a quadratic D); cubic part of f_0: {cubic_part(0, b)}; "
              f"slices at x4 = a are the Z Z Z^{b} sectors of |T5>^3 and at x1 = a the Z Z^{b} Z^{b} sectors; "
              f"the five sub-sector products are stabilizer states, independent, no four spanning.")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
