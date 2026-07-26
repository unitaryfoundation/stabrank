"""Exact certificate: the rank-6 stabilizer approximation error of |H3>^4.

Proposition (verified here in exact arithmetic). The 6-dimensional span of
the stabilizer states listed in BASIS_DATA (canonical form: uniform modulus
3^{-k/2} on the given support indices, phases omega^p with omega = e^{2 pi
i/3}) approximates |H3>^4, the four-copy qutrit Hadamard eigenstate, with
least-squares residual exactly

    sqrt((70 - 37 sqrt(3)) / 144) ~= 0.2026580337.

Conjecture. This is the optimal rank-6 stabilizer approximation of
|H3>^4. Support: five independent annealing runs on two engines
terminated at this residual to fifteen digits and never below it.

Run:  python h3m4_rank6_certificate.py
Requires sympy (not a package dependency; install separately).
"""
import sympy as sp

# (k, [(support_index, omega_power), ...]) per state; amplitudes 3^{-k/2} w^p.
BASIS_DATA = [
    (2, [(0, 0), (3, 0), (6, 0), (27, 0), (30, 0), (33, 0), (54, 0), (57, 0), (60, 0)]),
    (0, [(0, 0)]),
    (3, [(0, 0), (1, 2), (2, 2), (3, 0), (4, 2), (5, 2), (6, 0), (7, 2), (8, 2), (9, 2), (10, 1), (11, 1), (12, 2), (13, 1), (14, 1), (15, 2), (16, 1), (17, 1), (18, 2), (19, 1), (20, 1), (21, 2), (22, 1), (23, 1), (24, 2), (25, 1), (26, 1)]),
    (3, [(0, 0), (1, 2), (2, 2), (9, 2), (10, 1), (11, 1), (18, 2), (19, 1), (20, 1), (27, 0), (28, 2), (29, 2), (36, 2), (37, 1), (38, 1), (45, 2), (46, 1), (47, 1), (54, 0), (55, 2), (56, 2), (63, 2), (64, 1), (65, 1), (72, 2), (73, 1), (74, 1)]),
    (3, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0)]),
    (3, [(0, 0), (1, 0), (2, 0), (9, 0), (10, 0), (11, 0), (18, 0), (19, 0), (20, 0), (27, 0), (28, 0), (29, 0), (36, 0), (37, 0), (38, 0), (45, 0), (46, 0), (47, 0), (54, 0), (55, 0), (56, 0), (63, 0), (64, 0), (65, 0), (72, 0), (73, 0), (74, 0)]),
]

w3 = sp.Rational(-1, 2) + sp.sqrt(3)*sp.I/2
s3 = sp.sqrt(3)
c = (s3 - 1)/2
N = sp.sqrt(3 - s3)
H3 = [1/N, c/N, c/N]


def target_amp(idx):
    out = sp.Integer(1)
    for q in range(4):
        out *= H3[(idx // 3**(3 - q)) % 3]
    return out


def main():
    states = [
        {idx: w3**p / sp.sqrt(sp.Integer(3)**k) for idx, p in pairs}
        for k, pairs in BASIS_DATA
    ]
    G = sp.zeros(6, 6)
    t = sp.zeros(6, 1)
    for i in range(6):
        for j in range(6):
            common = set(states[i]) & set(states[j])
            G[i, j] = sp.expand(sum(
                sp.conjugate(states[i][x])*states[j][x] for x in common)) if common else sp.Integer(0)
        t[i] = sp.expand(sum(
            sp.conjugate(states[i][x])*target_amp(x) for x in states[i]))

    res2 = sp.simplify(sp.expand(1 - (t.H * G.inv() * t)[0, 0]))
    res2 = sp.nsimplify(sp.radsimp(res2), [sp.sqrt(3)])
    print("residual^2 =", res2)
    assert sp.simplify(res2 - (70 - 37*s3)/144) == 0
    print("exactly equals (70 - 37 sqrt(3))/144: verified")


if __name__ == "__main__":
    main()
