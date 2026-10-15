"""Exact arithmetic in the cyclotomic field Q(zeta_n), and the exact
stabilizer test for a vector on m qudits of odd prime dimension p.

Elements of Q(zeta_n) are tuples of `Fraction`s on the power basis 1, z,
..., z^(d - 1) with d = phi(n), reduced modulo the n-th cyclotomic
polynomial, so equality of tuples is equality in the field. Conjugation
sends z to z^(n - 1). Inversion solves the d x d system of multiplication
by the element. Everything here is integer or rational arithmetic; no
floating point enters a decision.

The stabilizer test (`stabilizer_form`) decides whether a vector with
entries in Q(zeta_n), given as a map from points of F_p^m to field
elements, is a nonzero scalar multiple of a stabilizer state: its support
is an affine flat, every entry has the modulus of the first, every ratio
to the first entry is a p-th root of unity, and the exponents form a
quadratic polynomial in the flat's parameters (for odd p the phase of a
stabilizer state is w_p^(quadratic) on an affine flat; Gross 2006,
Hostens, Dehaene, and De Moor 2005). It returns the board's term
parametrization (k, x0, W, Q, l) in the convention of
verify_challenge/to_witness.py, or None.
"""
from __future__ import annotations

import itertools
from fractions import Fraction


def _poly_divmod(num, den):
    """Quotient and remainder of integer polynomials (low to high), den monic."""
    num = list(num)
    q = [0] * max(1, len(num) - len(den) + 1)
    for i in range(len(num) - len(den), -1, -1):
        c = num[i + len(den) - 1]
        if c:
            q[i] = c
            for j, dj in enumerate(den):
                num[i + j] -= c * dj
    rem = num[:len(den) - 1]
    while len(rem) > 1 and rem[-1] == 0:
        rem.pop()
    return q, rem


def cyclotomic_poly(n):
    """Coefficients (low to high) of the n-th cyclotomic polynomial, from
    x^n - 1 = prod_{d | n} Phi_d by exact division."""
    if n == 1:
        return [-1, 1]
    poly = [-1] + [0] * (n - 1) + [1]
    for d in range(1, n):
        if n % d == 0:
            poly, rem = _poly_divmod(poly, cyclotomic_poly(d))
            assert all(r == 0 for r in rem)
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


class Cyclo:
    """Q(zeta_n) with elements as tuples of Fractions of length phi(n)."""

    def __init__(self, n):
        self.n = n
        self.phi = cyclotomic_poly(n)
        self.d = len(self.phi) - 1
        self.zero = tuple(Fraction(0) for _ in range(self.d))
        self.one = self._unit(0)
        # z^k reduced, for k = 0 .. n - 1
        self._pow = []
        for k in range(n):
            if k < self.d:
                self._pow.append(self._unit(k))
            else:
                self._pow.append(self._reduce([0] * k + [1]))

    def _unit(self, k):
        v = [Fraction(0)] * self.d
        v[k] = Fraction(1)
        return tuple(v)

    def _reduce(self, coeffs):
        """A polynomial in z (low to high, length <= 2d - 1) reduced modulo
        Phi_n; for k < n it uses the precomputed powers once available."""
        coeffs = [Fraction(c) for c in coeffs]
        out = [Fraction(0)] * self.d
        for k in range(len(coeffs) - 1, -1, -1):
            c = coeffs[k]
            if c == 0:
                continue
            if k < self.d:
                out[k] += c
            elif k < self.n and k < len(self._pow):
                for i, pi in enumerate(self._pow[k]):
                    out[i] += c * pi
            else:
                # synthetic reduction: z^k = z^(k - d) (-(Phi - z^d))
                for j in range(self.d):
                    coeffs[k - self.d + j] -= c * self.phi[j]
        return tuple(out)

    def root(self, k):
        """zeta_n^k."""
        return self._pow[k % self.n]

    def from_int(self, a):
        v = [Fraction(0)] * self.d
        v[0] = Fraction(a)
        return tuple(v)

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def neg(self, a):
        return tuple(-x for x in a)

    def scale(self, a, c):
        c = Fraction(c)
        return tuple(x * c for x in a)

    def mul(self, a, b):
        if a == self.zero or b == self.zero:
            return self.zero
        conv = [Fraction(0)] * (2 * self.d - 1)
        for i, x in enumerate(a):
            if x == 0:
                continue
            for j, y in enumerate(b):
                if y:
                    conv[i + j] += x * y
        return self._reduce(conv)

    def conj(self, a):
        """Complex conjugation, z -> z^(n - 1) = z^-1."""
        conv = [Fraction(0)] * self.n
        for i, x in enumerate(a):
            if x:
                conv[(self.n - i) % self.n] += x
        return self._reduce(conv)

    def is_zero(self, a):
        return all(x == 0 for x in a)

    def norm2(self, a):
        """|a|^2 = a conj(a), a real element (its tuple is still in the power basis)."""
        return self.mul(a, self.conj(a))

    def inv(self, a):
        """1 / a by solving the multiplication matrix; raises on zero."""
        if self.is_zero(a):
            raise ZeroDivisionError("inverse of zero in Q(zeta_n)")
        cols = [self.mul(a, self._unit(k)) for k in range(self.d)]
        # M y = e_0 with M[i][k] = cols[k][i]
        M = [[cols[k][i] for k in range(self.d)] + [Fraction(1 if i == 0 else 0)] for i in range(self.d)]
        y = solve_square(M, self.d)
        if y is None:
            raise ZeroDivisionError("singular multiplication matrix")
        return tuple(y)

    def div(self, a, b):
        return self.mul(a, self.inv(b))

    def to_complex(self, a):
        import cmath
        z = cmath.exp(2j * cmath.pi / self.n)
        return sum(float(x) * z ** k for k, x in enumerate(a))

    def root_exponent(self, a, b, order):
        """k with a = zeta_order^k b, for a, b elements with |a| = |b| != 0,
        or None. Decided as a conj(b) = zeta^k |b|^2, which also forces
        |a| = |b|."""
        if order <= 0 or self.n % order:
            raise ValueError("order must divide n")
        step = self.n // order
        lhs = self.mul(a, self.conj(b))
        nb = self.norm2(b)
        if self.is_zero(nb):
            return None
        for k in range(order):
            if lhs == self.mul(self.root(step * k), nb):
                return k
        return None


def solve_square(M, d):
    """Gauss-Jordan on an augmented d x (d + 1) Fraction matrix; the solution
    vector, or None when singular."""
    M = [row[:] for row in M]
    for c in range(d):
        piv = next((r for r in range(c, d) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(d):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[r][d] for r in range(d)]


def solve_in_span(K, columns, target):
    """Coefficients a with sum_j a_j columns[j] = target over Q(zeta_n), by
    Gaussian elimination on the (dim x r) system; the columns must be
    linearly independent (else raises) and the system consistent (else
    returns None). Vectors are lists of field elements."""
    r = len(columns)
    dim = len(target)
    rows = [[columns[j][i] for j in range(r)] + [target[i]] for i in range(dim)]
    piv_rows = []
    rank = 0
    for c in range(r):
        piv = next((i for i in range(rank, dim) if not K.is_zero(rows[i][c])), None)
        if piv is None:
            raise ValueError("the columns are linearly dependent")
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = K.inv(rows[rank][c])
        rows[rank] = [K.mul(x, inv) for x in rows[rank]]
        for i in range(dim):
            if i != rank and not K.is_zero(rows[i][c]):
                f = rows[i][c]
                rows[i] = [K.sub(x, K.mul(f, y)) for x, y in zip(rows[i], rows[rank])]
        piv_rows.append(c)
        rank += 1
    for i in range(rank, dim):
        if not K.is_zero(rows[i][r]):
            return None
    return [rows[c][r] for c in range(r)]


# ---------------------------------------------------------- F_p geometry ----

def rref_mod_p(vectors, p):
    """Row-reduced basis (tuples) of the span of integer vectors over F_p,
    and the pivot columns."""
    rows = [[int(x) % p for x in v] for v in vectors]
    n = len(rows[0]) if rows else 0
    basis, pivots = [], []
    rank = 0
    for c in range(n):
        piv = next((i for i in range(rank, len(rows)) if rows[i][c]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = pow(rows[rank][c], p - 2, p)
        rows[rank] = [(x * inv) % p for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                f = rows[i][c]
                rows[i] = [(x - f * y) % p for x, y in zip(rows[i], rows[rank])]
        pivots.append(c)
        rank += 1
    basis = [tuple(rows[i]) for i in range(rank)]
    return basis, pivots


def flat_of(points, p):
    """(x0, W, pivots) of the affine flat spanned by `points` (tuples in
    F_p^m): x0 the least point, W the RREF basis of the differences. Returns
    None when the points are not exactly that flat."""
    pts = sorted(points)
    x0 = pts[0]
    diffs = [tuple((a - b) % p for a, b in zip(q, x0)) for q in pts[1:]]
    W, piv = rref_mod_p(diffs, p) if diffs else ([], [])
    k = len(W)
    if len(pts) != p ** k:
        return None
    seen = set(pts)
    for y in itertools.product(range(p), repeat=k):
        x = tuple((x0[c] + sum(y[r] * W[r][c] for r in range(k))) % p for c in range(len(x0)))
        if x not in seen:
            return None
    return x0, W, piv


def reduce_mod_direction(t, W, piv, p):
    """The canonical representative of t + span(W): the pivot coordinates
    cleared with the RREF rows. Linear in t."""
    t = list(x % p for x in t)
    for row, c in zip(W, piv):
        f = t[c]
        if f:
            t = [(a - f * b) % p for a, b in zip(t, row)]
    return tuple(t)


def stabilizer_form(K, p, m, entries):
    """The term (k, x0, W, Q, l) of a vector proportional to a stabilizer
    state on m qudits of odd prime dimension p, or None. `entries` maps
    points (tuples in F_p^m) with a nonzero entry to field elements of K =
    Q(zeta_n) with p | n; zero entries are absent."""
    if p == 2:
        raise ValueError("odd prime p only (qubit phases are fourth roots with a different form)")
    if not entries:
        return None
    flat = flat_of(list(entries), p)
    if flat is None:
        return None
    x0, W, _ = flat
    k = len(W)
    ref = entries[x0]
    expo = {}
    for y in itertools.product(range(p), repeat=k):
        x = tuple((x0[c] + sum(y[r] * W[r][c] for r in range(k))) % p for c in range(m))
        e = K.root_exponent(entries[x], ref, p)
        if e is None:
            return None
        expo[y] = e

    def unit(i, mult=1):
        y = [0] * k
        y[i] = mult % p
        return tuple(y)

    inv2 = pow(2, -1, p)
    Q = [[0] * k for _ in range(k)]
    ell = [0] * k
    for i in range(k):
        e1, e2 = expo[unit(i)], expo[unit(i, 2)]
        Q[i][i] = ((e2 - 2 * e1) * inv2) % p
        ell[i] = (e1 - Q[i][i]) % p
    for i in range(k):
        for j in range(i + 1, k):
            y = [0] * k
            y[i] = y[j] = 1
            Q[i][j] = (expo[tuple(y)] - expo[unit(i)] - expo[unit(j)]) % p
    for y, e in expo.items():
        q = sum(Q[i][j] * y[i] * y[j] for i in range(k) for j in range(i, k))
        lin = sum(ell[i] * y[i] for i in range(k))
        if (q + lin) % p != e:
            return None
    return {"k": k, "x0": list(x0), "W": [list(r) for r in W], "Q": Q, "l": ell}
