"""What a partial-symmetry ansatz costs: the size of the fixed set of an
involution of the copies.

`witness_symmetry.py` shows that every minimal witness the board holds has a
nontrivial set-stabilizer inside U^m semidirect S_m, and that the smallest
symmetry common to all of them is an involution of the copies, sometimes
only after a local Clifford twist. The ansatz "the term set is invariant
under one involution tau" splits a rank-R term set into f terms fixed by tau
and c two-element orbits, with f + 2c = R: the fixed terms come from the
tau-fixed stabilizer states, and each orbit is determined by one free state
of the full dictionary. The search depth falls from R to f + c, and the
question is whether the fixed set is small enough to enumerate, as the
sigma-fixed set was for the m-cycle of 2026-09-26 (66 states at p = 2,
m = 7, and 120 at p = 3, m = 5).

The count below is a lower bound, and an exact one: the tau-fixed stabilizer
states supported on all of F_p^m with x0 = 0 are exactly the states whose
phase function is tau-invariant, and those are counted coefficient by
coefficient. Writing c1 for the number of orbits of tau on the m coordinates
and c2 for the number of orbits on the unordered pairs i < j,

    odd p:  p^(2 c1 + c2)     (l and the diagonal of Q, one value per
                               coordinate orbit, and the off-diagonal Q,
                               one value per pair orbit)
    p = 2:  4^c1 2^c2         (l in Z_4 per coordinate orbit, the strictly
                               upper Q in Z_2 per pair orbit)

States on smaller flats, and on other cosets, only add to this. Against it
the script prints the leaf count of the subset search that
`cyclic_symmetric.py` runs, C(N, R-2), so the two ansatze can be compared on
the same scale.

Usage: involution_cost.py [--control]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

# (label, orbit, p, m, record rank)
CELLS = [
    ("H3 m=4", "H3", 3, 4, 6),
    ("T3 m=4", "T3", 3, 4, 8),
    ("S m=5", "S", 3, 5, 5),
    ("qubit_H m=7", "qubit_H", 2, 7, 6),
]


def involutions(m):
    """One representative per cycle type of involution in S_m."""
    out = []
    for k in range(1, m // 2 + 1):
        pi = list(range(m))
        for j in range(k):
            pi[2 * j], pi[2 * j + 1] = pi[2 * j + 1], pi[2 * j]
        out.append((k, tuple(pi)))
    return out


def orbit_counts(pi, m):
    """(orbits on coordinates, orbits on unordered pairs)."""
    seen, c1 = set(), 0
    for i in range(m):
        if i in seen:
            continue
        c1 += 1
        j = i
        while j not in seen:
            seen.add(j)
            j = pi[j]
    pairs = {frozenset((i, j)) for i in range(m) for j in range(i + 1, m)}
    seen, c2 = set(), 0
    for q in sorted(map(sorted, pairs)):
        q = frozenset(q)
        if q in seen:
            continue
        c2 += 1
        r = q
        while r not in seen:
            seen.add(r)
            r = frozenset(pi[x] for x in r)
    return c1, c2


def full_support_fixed(p, m, pi):
    c1, c2 = orbit_counts(pi, m)
    if p == 2:
        return 4 ** c1 * 2 ** c2, c1, c2
    return p ** (2 * c1 + c2), c1, c2


def dictionary_size(p, n):
    out = p ** n
    for k in range(1, n + 1):
        out *= p ** k + 1
    return out


def report():
    print("%-12s %-3s %-4s %-14s %-12s %-10s %s" %
          ("cell", "m", "rank", "involution", "fixed >=", "dictionary", "C(fixed, R-2) leaves"))
    for label, orbit, p, m, R in CELLS:
        for k, pi in involutions(m):
            n, c1, c2 = full_support_fixed(p, m, pi)
            leaves = math.comb(n, R - 2) if n >= R - 2 else 0
            print("%-12s %-3d %-4d %-14s %-12d %-10.3g %.3g" %
                  (label, m, R, "%d transposition%s" % (k, "s" if k > 1 else ""),
                   n, dictionary_size(p, m), leaves))
    print()
    print("For comparison, 2026-09-26: the m-cycle fixes 66 stabilizer states at")
    print("p = 2, m = 7 (C(66, 4) = %d leaves, 75 s) and 120 at p = 3, m = 5." %
          math.comb(66, 4))


# ------------------------------------------------------------- control ----

def all_stabilizer_states(p, n):
    """Every n-qudit stabilizer state, by brute force over (k, x0, W, Q, l).
    Small n only; used to check the counting formula."""
    sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
    from rank_exclusion import dictionary
    return dictionary(p, n)


def control():
    ok = True

    def show(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  %-56s %-4s (got %s, want %s)" % (name, "ok" if good else "FAIL", got, want))

    for p, n in ((3, 2), (2, 3), (2, 4)):
        D = all_stabilizer_states(p, n)
        show("the dictionary at p=%d, n=%d has the expected size" % (p, n),
             D.shape[1], dictionary_size(p, n))
        pi = tuple([1, 0] + list(range(2, n)))
        base = np.arange(p ** n).reshape((p,) * n)
        idx = np.transpose(base, pi).ravel()
        full = np.abs(D) > 1e-9
        keep = full.all(axis=0)
        sub = D[:, keep]
        moved = sub[idx]
        # projective test; for a full-support state with x0 = 0 the index 0 is
        # fixed by the coordinate permutation, so the phase can only be 1
        fixed = int(np.sum(np.abs(np.einsum("ij,ij->j", sub.conj(), moved)) > 1 - 1e-9))
        want, _, _ = full_support_fixed(p, n, pi)
        # x0 = 0 is the normalization the formula uses: every full-support
        # state has x0 = 0, so the whole count is comparable
        show("p=%d, n=%d: full-support states fixed by the swap" % (p, n), fixed, want)
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.control:
        return control()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
