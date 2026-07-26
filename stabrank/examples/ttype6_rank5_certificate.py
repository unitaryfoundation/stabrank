"""Exact certificate: the rank-5 stabilizer approximation error of |T>^6.

Proposition (verified here in exact arithmetic). Let |T> be the qubit
Bravyi-Kitaev T-type magic state cos(b)|0> + e^{i pi/4} sin(b)|1> with
cos(2b) = 1/sqrt(3). The 5-dimensional span of the stabilizer states

  w_0 = |+i>|+>|0>|+i>|+>|0>,  w_1, w_2 its cyclic qubit shifts,
  g_0 = the uniform-amplitude state with phases i^{f(x)},
        f(x) = x_1 + x_3 + x_5 + 2 * sum_{(a,b) in E0} x_a x_b,
        E0 = complement of the 6-cycle (0,1,4,5,2,3),
  g_1 = the cyclic shift of g_0,

approximates |T>^6 with least-squares residual exactly

    sqrt(5 (2 - sqrt(3)) / 24) = sqrt(5/6) * sin(pi/12) ~= 0.2362683822.

Conjecture. This is the optimal rank-5 stabilizer approximation of
|T>^6; equivalently chi(|T>^6) = 6, i.e. sub-multiplicativity
chi(|T>^6) <= chi(|T>^4) * chi(|T>^2) = 3 * 2 is tight at six copies.
Support: eleven independent simulated-annealing campaigns (dense
warm-started anneals and cyclic-orbit symmetric-ansatz searches over
three distinct orbit profiles) all terminated at this residual to
fifteen digits and never below it.

The optimal coefficients pair the two blocks orthogonally: the graph
states enter with real coefficients 2/(3 sqrt(3)) and the product
states with equal purely imaginary coefficients of squared modulus
(26 + 15 sqrt(3))/216.

Run:  python ttype6_rank5_certificate.py
Requires sympy (not a package dependency; install separately).
"""
import itertools

import sympy as sp

s2, s3 = sp.sqrt(2), sp.sqrt(3)
cos2 = (sp.Integer(3) + s3) / 6          # cos^2(beta), cos(2 beta) = 1/sqrt(3)
sin2 = (sp.Integer(3) - s3) / 6
cosb, sinb = sp.sqrt(cos2), sp.sqrt(sin2)
eip4 = (1 + sp.I) / s2

BITS = list(itertools.product((0, 1), repeat=6))


def target_amp(bits):
    w = sum(bits)
    return cosb ** (6 - w) * (eip4 * sinb) ** w


SINGLE = {
    "0": (sp.Integer(1), sp.Integer(0)),
    "+": (1 / s2, 1 / s2),
    "i": (1 / s2, sp.I / s2),
}


def product_state(pattern):
    def amp(bits):
        out = sp.Integer(1)
        for q, b in enumerate(bits):
            out *= SINGLE[pattern[q]][b]
        return out
    return amp


# E0 is the complement of the 6-cycle (0,1,4,5,2,3); E1 is its cyclic shift.
E0 = [(0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 5), (2, 4), (3, 4), (3, 5)]
E1 = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)]


def graph_state(lin, edges):
    def amp(bits):
        f = sum(lin[q] * bits[q] for q in range(6))
        f += 2 * sum(bits[a] * bits[b] for a, b in edges)
        return sp.I ** (f % 4) / 8
    return amp


AMPS = [
    graph_state([0, 1, 0, 1, 0, 1], E0),
    graph_state([1, 0, 1, 0, 1, 0], E1),
    product_state("i+0i+0"),
    product_state("0i+0i+"),
    product_state("+0i+0i"),
]


def main():
    gram = sp.zeros(5, 5)
    proj = sp.zeros(5, 1)
    for i in range(5):
        for j in range(5):
            gram[i, j] = sp.simplify(
                sum(sp.conjugate(AMPS[i](b)) * AMPS[j](b) for b in BITS))
        proj[i] = sp.simplify(
            sum(sp.conjugate(AMPS[i](b)) * target_amp(b) for b in BITS))

    residual_sq = sp.simplify(1 - (proj.H * gram.inv() * proj)[0, 0])
    residual_sq = sp.nsimplify(sp.radsimp(residual_sq), [sp.sqrt(3)])
    claimed = 5 * (2 - s3) / 24

    print("residual^2 =", sp.simplify(residual_sq))
    assert sp.simplify(residual_sq - claimed) == 0
    print("exactly equals 5(2 - sqrt(3))/24: verified")


if __name__ == "__main__":
    main()
