"""Certificate: chi(|sqrtT>^{ot m}) >= min(4, m + 1) for the sqrt(T) state,
with no search.

|sqrtT> = (|0> + zeta |1>)/sqrt 2 with zeta = exp(i pi/8), a primitive
sixteenth root of unity. Every qubit stabilizer state is a positive scalar
times a vector over Z[i] (entries in {0, 1, i, -1, -i}: the verifier's
parametrisation i^(l.y) (-1)^Q(y) on an affine flat), so the span of an
exact decomposition is defined over Q(i) and is mapped to itself by every
automorphism of C fixing Q(i). The target times sqrt(2)^m is t_1^m with
t_k = (1, zeta^k), which lies in Q(zeta), and Gal(Q(zeta)/Q(i)) has order
four: zeta -> zeta^k for k in {1, 5, 9, 13}, the units of Z/16 congruent to
1 mod 4, exactly the ones fixing i = zeta^4. So a stabilizer span containing
t_1^m contains V_m = span{t_1^m, t_5^m, t_9^m, t_13^m}, and chi >= dim V_m.

Descent, as for T3 (cert_t3_galois.py): if t_1^m = sum_j c_j s_j' over C
with s_j' in Z[i]^(2^m), the rank of a matrix is the same over C and over
Q(zeta), so coefficients exist in Q(zeta), and applying sigma_k entrywise
gives t_k^m = sum_j sigma_k(c_j) s_j' in the same span.

dim V_m = min(4, m + 1). Restricted to the coordinates of Hamming weight
0, 1, 2, 3 (one string each) the four vectors are the rows of the
Vandermonde matrix in zeta, zeta^5, zeta^9 = -zeta and zeta^13 = -zeta^5,
four distinct numbers, so the conjugates are independent as soon as m >= 3;
at m = 1, 2 they span the whole symmetric subspace, of dimension m + 1. The
rank of the 2^m x 4 matrix is computed exactly in Q(zeta) (sympy's
algebraic field over QQ, DomainMatrix.rank) for m = 1..6, and the 4 x 4
Vandermonde determinant is checked nonzero in the same field. No
floating-point number stands under the bound, so the bound files declare
`exact: true`.

Consistency checks, not load-bearing: the verifier's exact target
sqrt(2)^m |sqrtT>^m equals t_1^m symbolically, and every entry of the two-,
three- and four-qubit numerical dictionaries is in {0, 1, i, -1, -i} once
the first nonzero entry is normalised, which is the Z[i] fact on the
dictionaries the numerical exclusions used.

This replaces the numerical exclusions of the research survey (rank 2 at
m = 2 over 60 states, rank 3 at m = 3 over 1,080 and at m = 4 over 36,720),
whose m = 3 run came within 0.0019 of the parallel threshold, under
rank_exclusion's margin of 0.01.

Printed claims: CERTIFIED chi(qubit_sqrtT^1) >= 2
                CERTIFIED chi(qubit_sqrtT^2) >= 3
                CERTIFIED chi(qubit_sqrtT^3) >= 4
                CERTIFIED chi(qubit_sqrtT^4) >= 4
                CERTIFIED chi(qubit_sqrtT^5) >= 4
                CERTIFIED chi(qubit_sqrtT^6) >= 4
"""

from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stabrank_verify import target_vector  # noqa: E402

ZETA = sp.exp(sp.I * sp.pi / 8)
CONJ = (1, 5, 9, 13)              # the automorphisms of Q(zeta_16) fixing i
M_MAX = 6


def weight(x, m):
    return sum((x >> j) & 1 for j in range(m))


def conjugate_matrix(K, m):
    """The 2^m x 4 matrix whose columns are t_k^m, k in CONJ, over K = Q(zeta):
    entry (x, k) is zeta^(k |x|) since t_k^m has amplitude zeta^(k|x|) at x."""
    z = K.from_sympy(ZETA)
    rows = [[z ** (k * weight(x, m)) for k in CONJ] for x in range(1 << m)]
    return DomainMatrix(rows, (1 << m, 4), K)


def dictionary_entries_are_gaussian(n):
    from rank_exclusion import dictionary
    D = dictionary(2, n)
    for i in range(D.shape[1]):
        v = D[:, i]
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        if np.abs(v - np.rint(v.real) - 1j * np.rint(v.imag)).max() > 1e-9:
            return False
    return True


def main():
    for k in CONJ:
        if (k * 4) % 16 != 4:
            print(f"zeta -> zeta^{k} does not fix i", file=sys.stderr)
            return 1
    if {k % 16 for k in range(1, 16, 2) if (4 * k) % 16 == 4} != set(CONJ):
        print("the automorphisms fixing i are not the four listed", file=sys.stderr)
        return 1
    print("Gal(Q(zeta_16)/Q(i)) = {zeta -> zeta^k : k = 1, 5, 9, 13}, the units of Z/16 "
          "that fix i = zeta^4")
    K = sp.QQ.algebraic_field(ZETA)
    # the Vandermonde minor on the weights 0..3 of the m = 3 matrix
    z = K.from_sympy(ZETA)
    V = DomainMatrix([[z ** (k * w) for k in CONJ] for w in range(4)], (4, 4), K)
    if V.det() == K.zero:
        print("the 4 x 4 Vandermonde minor is singular", file=sys.stderr)
        return 1
    print("the Vandermonde minor in zeta, zeta^5, -zeta, -zeta^5 has nonzero determinant in Q(zeta)")
    # consistency: the verifier's target is t_1^m up to the scalar sqrt(2)^m
    for m in range(1, 4):
        tgt = target_vector("qubit_sqrtT", m) * sp.sqrt(2) ** m
        want = sp.Matrix([ZETA ** weight(x, m) for x in range(1 << m)])
        if any(sp.simplify(sp.expand(a - b)) != 0 for a, b in zip(tgt, want)):
            print(f"m={m}: the verifier's target is not t_1^m", file=sys.stderr)
            return 1
    print("consistency check: sqrt(2)^m |sqrtT>^m = t_1^m symbolically for m <= 3")
    for n in (2, 3, 4):
        if not dictionary_entries_are_gaussian(n):
            print(f"{n} qubits: a dictionary state is not in Z[i] up to a scalar", file=sys.stderr)
            return 1
    print("consistency check: every two-, three- and four-qubit stabilizer state is a scalar "
          "times a vector over Z[i]\n")

    bounds = {}
    for m in range(1, M_MAX + 1):
        r = conjugate_matrix(K, m).rank()
        want = min(4, m + 1)
        if r != want:
            print(f"m={m}: dim V_m = {r}, expected {want}", file=sys.stderr)
            return 1
        bounds[m] = r
        print(f"  m={m}: dim V_m = {r}, computed exactly in Q(zeta_16)")
    for m, r in bounds.items():
        print(f"CERTIFIED chi(qubit_sqrtT^{m}) >= {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
