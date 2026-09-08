"""Exact certificate: the rank-7 stabilizer approximation error of |S>^6.

Proposition (verified here in exact arithmetic). Let sigma_1, sigma_2 be the
two-qutrit stabilizer states of the paper's m = 2 Strange identity
(canonical form (1/3) sum_y omega^{Q_j(y)} |y> with Q_1 = y0^2 + y0 y1 +
y1^2 and Q_2 = y0^2 + 2 y0 y1 + y1^2), so that |S>^2 lies exactly in their
span. Of the eight product states sigma_a x sigma_b x sigma_c, every
7-element subset approximates |S>^6 with least-squares residual exactly

    sqrt(8/27) = 2 sqrt(2) / (3 sqrt(3)) ~= 0.5443310540.

Conjecture. This is the optimal rank-7 stabilizer approximation of
|S>^6; equivalently chi(|S>^6) = 8, i.e. sub-multiplicativity
chi(|S>^6) <= chi(|S>^2)^3 = 2^3 is tight at six copies. Support: a
warm-started dense anneal and an independent cyclic-orbit symmetric
search both terminated at this residual and never below it.

All inner products factorize over the three copies, so the computation
reduces to exact two-qutrit quantities.

Run:  python strange6_rank7_certificate.py
Requires sympy (not a package dependency; install separately).
"""
import itertools

import sympy as sp

w = sp.Rational(-1, 2) + sp.sqrt(3)*sp.I/2   # exp(2 pi i / 3)


def sigma(qform):
    return [w**qform(y0, y1) / 3 for y0 in range(3) for y1 in range(3)]


def ip(u, v):
    return sp.expand(sum(sp.conjugate(x)*y for x, y in zip(u, v)))


def main():
    s1 = sigma(lambda a, b: (a*a + a*b + b*b) % 3)
    s2 = sigma(lambda a, b: (a*a + 2*a*b + b*b) % 3)
    r = 1/sp.sqrt(2)
    svec = [sp.Integer(0), r, -r]
    s2vec = [sp.expand(svec[a]*svec[b]) for a in range(3) for b in range(3)]

    G2 = {(1, 1): ip(s1, s1), (1, 2): ip(s1, s2),
          (2, 1): sp.conjugate(ip(s1, s2)), (2, 2): ip(s2, s2)}
    T2 = {1: ip(s1, s2vec), 2: ip(s2, s2vec)}

    # Sanity: the paper's m = 2 identity, |S>^2 in span{sigma_1, sigma_2}.
    Gm = sp.Matrix([[G2[(1, 1)], G2[(1, 2)]], [G2[(2, 1)], G2[(2, 2)]]])
    tm = sp.Matrix([T2[1], T2[2]])
    assert sp.simplify(1 - (tm.H * Gm.inv() * tm)[0, 0]) == 0

    labels = list(itertools.product((1, 2), repeat=3))
    for dropped in labels:
        kept = [l for l in labels if l != dropped]
        G = sp.Matrix(7, 7, lambda i, j: sp.expand(
            sp.prod(G2[(kept[i][q], kept[j][q])] for q in range(3))))
        t = sp.Matrix(7, 1, lambda i, _: sp.expand(
            sp.prod(T2[kept[i][q]] for q in range(3))))
        res2 = sp.simplify(sp.expand(1 - (t.H * G.inv() * t)[0, 0]))
        assert sp.simplify(res2 - sp.Rational(8, 27)) == 0, (dropped, res2)
    print("all 8 seven-term subsets: residual^2 = 8/27 exactly, verified")


if __name__ == "__main__":
    main()
