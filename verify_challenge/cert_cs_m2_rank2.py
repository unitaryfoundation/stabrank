"""Certificate: chi(|CS>) >= 2 and chi(|CS>^{ot 2}) >= 3 for the two-qubit
controlled-S magic state, exactly, and chi(|CS>^{ot 3}) >= 3 by projection.

|CS> = CS|++> = (|00> + |01> + |10> + i|11>)/2 is a width-2 orbit: m copies
live on 2m qubits. Every amplitude of 2^m |CS>^m is a power of i, and every
n-qubit stabilizer state is a scalar times a vector over Z[i] with entries
in {0, 1, i, -1, -i} (the verifier's own parametrization, i^(l.y) (-1)^Q(y)
on an affine flat), so both questions live in the Gaussian rationals Q(i)
and are decided there: Python integers for the Gaussian integers, Fractions
for the one division. No floating-point number stands anywhere under the
bound, so the bound files declare `exact: true`.

Enumeration. The n-qubit stabilizer states are written down from the
parametrization sum_y i^(l.y) (-1)^Q(y) |x0 + W y> with W in reduced row
echelon form, x0 supported off the pivot columns, Q strictly upper
triangular (a diagonal entry of Q is the same state as l shifted by 2) and
l in Z_4^k. That names each state exactly once; the count is checked
against 2^n prod_{j<=n} (2^j + 1), which is 60 on two qubits and 36,720 on
four, and pairwise distinctness up to phase is checked on the exact
canonical forms.

Rank 1 and rank 2. Write P for the integer vector 2^m |CS>^m and, for a
state s, q(s) = <P,P> s - <P,s> P, the component of s off P scaled to stay
in Z[i]. q(s) = 0 exactly when s is parallel to P. For two distinct states
s, t, P lies in span(s, t) exactly when q(s) and q(t) are parallel: if
P = a s + b t with a, b nonzero then q(t) = -(a/b) q(s), and conversely
q(s) = c q(t) puts s - c t on the line of P with s, t not parallel. So the
pair search is one exact canonicalization per state, dividing q(s) by its
first nonzero entry in Q(i), and two states are a rank-2 decomposition of
P exactly when their canonical forms coincide.

Projection. (I (x) <00|) |CS>^3 = (1/2) |CS>^2, and a computational-basis
slice of a stabilizer state is a stabilizer state or zero, so a rank-r
decomposition of |CS>^3 slices to one of |CS>^2 with at most r terms and
chi(|CS>^3) >= chi(|CS>^2).

Controls, all fatal: on two qubits |++> must be reported as rank 1 (it is
also in the span of many pairs, as any stabilizer state is) and |CS> must
be spanned by the pair {|++>, CZ|++>} (CS = ((1+i)/2) I + ((1-i)/2) CZ)
with no rank-1 hit; on four qubits |CS> (x) |++> must be spanned by a pair
(its rank is 2) and |++++> reported as rank 1. The exact states are also
matched one to one against the numerical dictionary of rank_exclusion.py,
a consistency check nothing below rests on.

Printed claims: CERTIFIED chi(CS^1) >= 2
                CERTIFIED chi(CS^2) >= 3
                CERTIFIED chi(CS^3) >= 3
"""

from __future__ import annotations

import itertools
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

I4 = ((1, 0), (0, 1), (-1, 0), (0, -1))          # i^e as a Gaussian integer (re, im)


def count(n):
    """2^n prod_{j<=n} (2^j + 1), the number of n-qubit stabilizer states."""
    t = 1 << n
    for j in range(1, n + 1):
        t *= (1 << j) + 1
    return t


def rref_matrices(n, k):
    """Every k x n matrix over F_2 in reduced row echelon form, with its
    non-pivot columns (where the coset representative x0 is free)."""
    for pivots in itertools.combinations(range(n), k):
        non_pivots = [j for j in range(n) if j not in pivots]
        free = [(r, c) for r in range(k) for c in non_pivots if c > pivots[r]]
        for vals in itertools.product(range(2), repeat=len(free)):
            W = [[0] * n for _ in range(k)]
            for r in range(k):
                W[r][pivots[r]] = 1
            for (r, c), v in zip(free, vals):
                W[r][c] = v
            yield W, non_pivots


def exact_state(n, k, x0, W, cross, ell):
    """The unnormalized stabilizer state sum_y i^(l.y) (-1)^(sum_{i<j} Q_ij y_i y_j)
    |x0 + W y> as a tuple of 2^n Gaussian integers; `cross` lists the
    strictly-upper entries Q_ij that are 1."""
    v = [(0, 0)] * (1 << n)
    for y in itertools.product(range(2), repeat=k):
        pos = 0
        for c in range(n):
            pos = 2 * pos + (x0[c] + sum(y[r] * W[r][c] for r in range(k))) % 2
        e = sum(ell[r] * y[r] for r in range(k)) + 2 * sum(y[i] * y[j] for i, j in cross)
        v[pos] = I4[e % 4]
    return tuple(v)


def stabilizer_states(n):
    """Every n-qubit stabilizer state, exactly, each named once."""
    out = []
    for k in range(n + 1):
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for W, non_pivots in rref_matrices(n, k):
            for free in itertools.product(range(2), repeat=n - k):
                x0 = [0] * n
                for j, c in enumerate(non_pivots):
                    x0[c] = free[j]
                for qbits in itertools.product(range(2), repeat=len(pairs)):
                    cross = [pq for pq, b in zip(pairs, qbits) if b]
                    for ell in itertools.product(range(4), repeat=k):
                        out.append(exact_state(n, k, x0, W, cross, ell))
    return out


def gmul(x, y):
    return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def kron(u, v):
    return tuple(gmul(a, b) for a in u for b in v)


def canonical(q):
    """q divided by its first nonzero entry, as exact Gaussian rationals; None
    for the zero vector."""
    j = next((i for i, z in enumerate(q) if z != (0, 0)), None)
    if j is None:
        return None
    c, d = q[j]
    nrm = c * c + d * d
    return tuple((Fraction(a * c + b * d, nrm), Fraction(b * c - a * d, nrm)) for a, b in q)


def exact_search(P, states):
    """(indices of states parallel to P, pairs of states whose span contains P),
    decided in Q(i)."""
    pp = sum(a * a + b * b for a, b in P)
    keys, rank1 = {}, []
    for idx, s in enumerate(states):
        ip = (0, 0)                                  # <P, s> = sum conj(P_x) s_x
        for (a, b), (c, d) in zip(P, s):
            ip = (ip[0] + a * c + b * d, ip[1] + a * d - b * c)
        q = tuple((pp * c - (ip[0] * a - ip[1] * b), pp * d - (ip[0] * b + ip[1] * a))
                  for (a, b), (c, d) in zip(P, s))
        key = canonical(q)
        if key is None:
            rank1.append(idx)
        else:
            keys.setdefault(key, []).append(idx)
    pairs = [(g[a], g[b]) for g in keys.values() if len(g) > 1
             for a in range(len(g)) for b in range(a + 1, len(g))]
    return rank1, pairs


def check_against_dictionary(states, n):
    """Consistency check only: the exact list matches the numerical dictionary
    one to one. Not load-bearing for the bound."""
    from rank_exclusion import dictionary
    D = dictionary(2, n)
    index = {}
    for i in range(D.shape[1]):
        v = D[:, i]
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        index[(np.rint(v.real).astype(int).tobytes(), np.rint(v.imag).astype(int).tobytes())] = i
    used = set()
    for s in states:
        c = canonical(s)
        key = (np.array([int(a) for a, _ in c]).tobytes(), np.array([int(b) for _, b in c]).tobytes())
        if key not in index or index[key] in used:
            return False
        used.add(index[key])
    return len(used) == D.shape[1]


def main():
    cs = ((1, 0), (1, 0), (1, 0), (0, 1))            # 2 |CS>
    plus2 = ((1, 0),) * 4                            # 2 |++>
    cz2 = ((1, 0), (1, 0), (1, 0), (-1, 0))          # 2 CZ|++>
    S = {}
    for n in (2, 4):
        S[n] = stabilizer_states(n)
        if len(S[n]) != count(n):
            print(f"{n} qubits: enumerated {len(S[n])} states, expected {count(n)}", file=sys.stderr)
            return 1
        if len({canonical(s) for s in S[n]}) != len(S[n]):
            print(f"{n} qubits: two enumerated states coincide up to phase", file=sys.stderr)
            return 1
        print(f"{n} qubits: {len(S[n])} stabilizer states enumerated from the parametrization, "
              f"distinct up to phase; all arithmetic in Z[i] and Q(i)")
        if not check_against_dictionary(S[n], n):
            print(f"{n} qubits: the exact enumeration does not match the numerical dictionary",
                  file=sys.stderr)
            return 1
        print(f"consistency check: the {len(S[n])} exact states match the numerical dictionary "
              "one to one")

    # controls on two qubits
    rank1, pairs = exact_search(plus2, S[2])
    if rank1 != [S[2].index(plus2)]:
        print(f"rank-1 control FAILED: |++> gave rank1={rank1}", file=sys.stderr)
        return 1
    print("rank-1 control: |++> is reported as parallel to exactly one state, itself")
    i_plus, i_cz = S[2].index(plus2), S[2].index(cz2)
    rank1, pairs = exact_search(cs, S[2])
    want = tuple(sorted((i_plus, i_cz)))
    if rank1 or want not in {tuple(sorted(pq)) for pq in pairs}:
        print(f"positive control FAILED: |CS> gave rank1={rank1}, and the pair "
              f"{{|++>, CZ|++>}} = {want} is not among {pairs}", file=sys.stderr)
        return 1
    print(f"CS m=1: no stabilizer state is parallel to |CS>; {len(pairs)} of the "
          f"{len(S[2]) * (len(S[2]) - 1) // 2} pairs span it, {{|++>, CZ|++>}} among them")
    # controls on four qubits
    rank1, pairs = exact_search(kron(plus2, plus2), S[4])
    if rank1 != [S[4].index(kron(plus2, plus2))]:
        print(f"rank-1 control FAILED: |++++> gave rank1={rank1}", file=sys.stderr)
        return 1
    print("rank-1 control: |++++> is reported as parallel to exactly one state, itself")
    rank1, pairs = exact_search(kron(cs, plus2), S[4])
    if rank1 or not pairs:
        print(f"positive control FAILED: |CS>|++> gave rank1={rank1}, pairs={pairs}", file=sys.stderr)
        return 1
    print(f"positive control: |CS> (x) |++> on four qubits is spanned by {len(pairs)} pairs "
          "and parallel to no state\n")

    rank1, pairs = exact_search(kron(cs, cs), S[4])
    if rank1:
        print(f"CS m=2: |CS>^2 is parallel to stabilizer state(s) {rank1}")
        return 1
    if pairs:
        print(f"CS m=2: rank-2 decomposition FOUND, {pairs[:5]}")
        return 1
    print(f"CS m=2: no state is parallel to |CS>^2 and no pair of the {len(S[4])} four-qubit "
          "stabilizer states spans it; every test was decided exactly in Q(i)")
    print("CERTIFIED chi(CS^1) >= 2")
    print("CERTIFIED chi(CS^2) >= 3")
    # projection: (I (x) <00|) |CS>^3 = |CS>^2 / 2
    print("CERTIFIED chi(CS^3) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
