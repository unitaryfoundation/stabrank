"""Certificate: chi(|T5>) >= 3 for the ququint T-type state, exactly.

|T5> = 5^(-1/2) sum_x w^(x^3) |x> with w = exp(2 pi i / 5). One ququint
carries 30 stabilizer states: the five basis states and the 25 full-support
states sum_y w^(q y^2 + l y) |y> with q, l in F_5. Every amplitude of every
state, and of |T5>, is 0 or a power of w, so the whole question lives in the
ring Z[w], and it is decided there without a single floating-point number.

Rank 1: |T5> is parallel to a state s exactly when every 2 x 2 minor of the
5 x 2 matrix [s | psi] vanishes. Rank 2: two distinct stabilizer states s, t
are linearly independent (they are distinct up to a global phase), so |T5>
lies in their span exactly when the 5 x 3 matrix [s | t | psi] has rank 2,
that is, when all ten 3 x 3 minors vanish. A solution over C of a linear
system with coefficients in Q(w) exists only if one exists over Q(w), so
deciding the span over Q(w) decides it over C.

Arithmetic. An element of Z[w] is stored as five integers, the coefficients
of 1, w, ..., w^4. Products of powers of w add exponents mod 5, so a minor
is a signed sum of monomials and is accumulated exactly. The element
sum_i c_i w^i is zero if and only if c_0 = c_1 = c_2 = c_3 = c_4: the
minimal polynomial of w is 1 + w + w^2 + w^3 + w^4, so the kernel of
Z^5 -> Z[w] is spanned by (1, 1, 1, 1, 1). That is the only zero test used.

The 30 states are enumerated from the parametrisation itself, and the count
is checked against 5 (5 + 1) = 30 and against pairwise distinctness up to
phase; as a consistency check the exact vectors are also matched one to one
against the numerical dictionary of rank_exclusion.py, but nothing below
rests on that match. Every step is exact, so the bound file declares
`exact: true`.

Controls: |0> + |1> has support of size 2, which is not an affine flat of
F_5, so it is not a stabilizer state, and the pair {|0>, |1>} spans it;
the search must report exactly that pair and no rank-1 hit. |+> is a
stabilizer state and must be reported as rank 1.

Printed claim: CERTIFIED chi(T5^1) >= 3
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P = 5


def stabilizer_states():
    """The 30 single-ququint stabilizer states as exponent vectors: entry
    e in 0..4 for the amplitude w^e, None for a zero amplitude."""
    states = [tuple(0 if x == x0 else None for x in range(P)) for x0 in range(P)]
    for q in range(P):
        for l in range(P):
            states.append(tuple((q * y * y + l * y) % P for y in range(P)))
    return states


def t5():
    return tuple((x ** 3) % P for x in range(P))


def same_up_to_phase(a, b):
    """Exact: equal supports and a constant exponent shift on the support."""
    if any((x is None) != (y is None) for x, y in zip(a, b)):
        return False
    shifts = {(y - x) % P for x, y in zip(a, b) if x is not None}
    return len(shifts) == 1


def minor_is_zero(cols, rows):
    """Whether the square minor of the matrix with the given exponent columns,
    restricted to `rows`, is zero in Z[w]. Exact integer arithmetic."""
    k = len(cols)
    coeff = [0] * P
    for perm in itertools.permutations(range(k)):
        sign = 1
        for i in range(k):
            for j in range(i + 1, k):
                if perm[i] > perm[j]:
                    sign = -sign
        e = 0
        for i in range(k):
            a = cols[perm[i]][rows[i]]
            if a is None:
                break
            e += a
        else:
            coeff[e % P] += sign
    return len(set(coeff)) == 1


def in_span(psi, cols):
    """Exact: whether psi lies in the span of the given exponent vectors, all
    of which must be linearly independent (true for distinct stabilizer
    states when there are at most two)."""
    k = len(cols) + 1
    return all(minor_is_zero(list(cols) + [psi], rows)
               for rows in itertools.combinations(range(P), k))


def rank2_search_exact(psi, states):
    """(pairs spanning psi, states parallel to psi)."""
    rank1 = [i for i, s in enumerate(states) if in_span(psi, [s])]
    if rank1:
        return [], rank1
    pairs = [(i, j) for i, j in itertools.combinations(range(len(states)), 2)
             if in_span(psi, [states[i], states[j]])]
    return pairs, []


def check_against_dictionary(states):
    """Consistency check only: the exact list matches the numerical dictionary
    one to one. Not load-bearing for the bound."""
    from rank_exclusion import dictionary
    D = dictionary(P, 1)
    w = np.exp(2j * np.pi / P)
    used = set()
    for s in states:
        v = np.array([0 if e is None else w ** e for e in s])
        v /= np.linalg.norm(v)
        ov = np.abs(v.conj() @ D)
        hits = np.flatnonzero(ov > 1 - 1e-9)
        if len(hits) != 1 or int(hits[0]) in used:
            return False
        used.add(int(hits[0]))
    return len(used) == D.shape[1]


def main():
    states = stabilizer_states()
    if len(states) != P * (P + 1):
        print(f"expected {P * (P + 1)} states, enumerated {len(states)}", file=sys.stderr)
        return 1
    for a, b in itertools.combinations(states, 2):
        if same_up_to_phase(a, b):
            print("two enumerated states coincide up to phase", file=sys.stderr)
            return 1
    print(f"{len(states)} single-ququint stabilizer states, distinct up to phase, "
          f"{len(states) * (len(states) - 1) // 2} pairs; all arithmetic in Z[w_5]")
    if not check_against_dictionary(states):
        print("the exact enumeration does not match the numerical dictionary",
              file=sys.stderr)
        return 1
    print("consistency check: the 30 exact states match the numerical dictionary one to one")

    # controls
    ctrl = (0, 0, None, None, None)                      # |0> + |1>
    pairs, rank1 = rank2_search_exact(ctrl, states)
    want = [(0, 1)]
    if rank1 or pairs != want:
        print(f"positive control FAILED: |0> + |1> gave rank1={rank1}, pairs={pairs}, "
              f"expected exactly the pair {want}", file=sys.stderr)
        return 1
    print("positive control: |0> + |1> is spanned by exactly the pair {|0>, |1>}")
    plus = (0, 0, 0, 0, 0)
    pairs, rank1 = rank2_search_exact(plus, states)
    if len(rank1) != 1:
        print(f"rank-1 control FAILED: |+> reported parallel to {rank1}", file=sys.stderr)
        return 1
    print("rank-1 control: |+> is reported as a stabilizer state\n")

    psi = t5()
    pairs, rank1 = rank2_search_exact(psi, states)
    if rank1:
        print(f"T5 m=1: |T5> is parallel to stabilizer state(s) {rank1}")
        return 1
    if pairs:
        print(f"T5 m=1: rank-2 decomposition FOUND, {pairs}")
        return 1
    print("T5 m=1: no stabilizer state is parallel to |T5> and no pair of the 30 spans it; "
          "every 3 x 3 minor test was decided exactly in Z[w_5]")
    print("CERTIFIED chi(T5^1) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
