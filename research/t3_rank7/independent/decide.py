"""Exact decision for candidate class sets: is span(T) inside span(states) over Q(w3)?

rank mod ELL2 = 2^31-1 is exact for values <= 8 (Hadamard: a k x k minor of a matrix of
zeros and roots of unity has modulus <= k^(k/2) < sqrt(ELL2) for k <= 9).  For larger
sets we compute the rank mod a second prime ELL3 as well; if both ranks agree and are
<= 14 the value is exact (a nonzero minor of size <= 15 has norm < 15^15 < ELL2 * ELL3, so
it cannot vanish mod both).  Anything else falls back to Q(w3) fractions.
"""
import numpy as np
from fractions import Fraction

ELL2 = 2 ** 31 - 1
ELL3 = 2 ** 31 - 19        # prime, 1 mod 3
assert ELL3 % 3 == 1


def _root6(ell):
    for g in range(2, ell):
        z = pow(g, (ell - 1) // 6, ell)
        if pow(z, 3, ell) == ell - 1 and pow(z, 2, ell) != 1:
            return z


Z6_2 = _root6(ELL2)
Z6_3 = _root6(ELL3)
for z, l in ((Z6_2, ELL2), (Z6_3, ELL3)):
    w = pow(z, 2, l)
    assert (w * w + w + 1) % l == 0


def to_mod(E, z6, ell):
    tab = np.array([pow(z6, e, ell) for e in range(6)] + [0], dtype=np.int64)
    return tab[E.astype(np.int64)]


def rank_mod(M, ell):
    A = M.copy() % ell
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        nz = np.flatnonzero(A[r:, c])
        if len(nz) == 0:
            continue
        pr = r + nz[0]
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        A[r] = (A[r] * pow(int(A[r, c]), ell - 2, ell)) % ell
        f = A[:, c].copy()
        f[r] = 0
        A = (A - np.outer(f, A[r])) % ell
        r += 1
        if r == rows:
            break
    return r


class Qw:
    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a, self.b = Fraction(a), Fraction(b)

    def __add__(s, o): return Qw(s.a + o.a, s.b + o.b)
    def __sub__(s, o): return Qw(s.a - o.a, s.b - o.b)
    def __mul__(s, o): return Qw(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a - s.b * o.b)

    def inv(s):
        n = s.a * s.a - s.a * s.b + s.b * s.b
        return Qw((s.a - s.b) / n, -s.b / n)

    def iszero(s): return s.a == 0 and s.b == 0


_TAB = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 0), 4: (-1, -1), 5: (0, -1), 6: (0, 0)}


def exact_rank(rows_E):
    M = [[Qw(*_TAB[int(v)]) for v in row] for row in rows_E]
    nr, nc = len(M), len(M[0])
    r = 0
    for c in range(nc):
        p = next((i for i in range(r, nr) if not M[i][c].iszero()), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        iv = M[r][c].inv()
        M[r] = [x * iv for x in M[r]]
        for i in range(nr):
            if i != r and not M[i][c].iszero():
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
        if r == nr:
            break
    return r


class Decider:
    def __init__(self, E, T):
        self.E, self.T = E, T
        self.E2, self.T2 = to_mod(E, Z6_2, ELL2), to_mod(T, Z6_2, ELL2)
        self.E3, self.T3 = to_mod(E, Z6_3, ELL3), to_mod(T, Z6_3, ELL3)
        self.how = {}

    def ranks(self, states):
        """Exact (rank(S), rank(S + T)) over Q(w3)."""
        S2 = self.E2[states]
        r1 = rank_mod(S2, ELL2)
        r2 = rank_mod(np.vstack([S2, self.T2]), ELL2)
        if r1 <= 8 and r2 <= 8:
            self.how["ell2"] = self.how.get("ell2", 0) + 1
            return r1, r2
        S3 = self.E3[states]
        s1 = rank_mod(S3, ELL3)
        s2 = rank_mod(np.vstack([S3, self.T3]), ELL3)
        if r1 == s1 and r2 == s2 and r1 <= 14 and r2 <= 14:
            self.how["two primes"] = self.how.get("two primes", 0) + 1
            return r1, r2
        self.how["Q(w3)"] = self.how.get("Q(w3)", 0) + 1
        rows = [self.E[s] for s in states]
        return exact_rank(rows), exact_rank(rows + [self.T[r] for r in range(self.T.shape[0])])

    def contains(self, states):
        r1, r2 = self.ranks(states)
        return r1 == r2, r1, r2
