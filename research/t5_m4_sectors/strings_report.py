"""Sub-sector term counts for every one-Pauli eigensector of |T5>^4.

For a full-support string P = Q1 Q2 Q3 Q4 and a pairing of the copies, say
(12)(34), every joint eigensector of P is the sum of the products (Q1 Q2
sector s1 of |T5>^2) (x) (Q3 Q4 sector s2 of |T5>^2) over the pairs (s1, s2)
whose eigenvalues multiply to the sector's. Each product is a product of
two two-ququint states whose stabilizer ranks are the ranks of their
reduced one-ququint states, decided exactly by brute force over the 30
one-ququint stabilizer states, so the sector's rank is at most the sum of
the rank products over the nonvanishing pairs. This script lists, for each
of the 330 string classes (8 single-site orbits under the order-5 Clifford
stabilizer of |T5>, four copies, order of the copies ignored), the number
of nonvanishing terms and the bound, minimized over the three pairings.

The question it answers: does any string give a sector with fewer than five
sub-sector terms, or five terms of smaller total rank than Z^4's five
stabilizer states? At m = 2 every sector of every string is nonvanishing
(the sector weights are (1/5)(1 + sum_{j=1..4} w^{-sj} <P^j>) with |<P^j>|
<= 1/5 for a Z-free string and 0 for a string with a Z-type site, so no
weight vanishes), hence every pairing gives exactly five nonvanishing terms
for every string, and the bound is 5 exactly when both two-copy sectors
are stabilizer states, which is the Z Z decomposition (unique, section 8 of
t5q_m2_rank5_exclusion.md) and its powers Z^b Z^b.

Uses research/constructions/sectors_p.py; imports research/constructions/
common.py under its own name, so this script must not share a process with
research/t5q_m2_rank5/common.py.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
from rank_exclusion import dictionary  # noqa: E402
from common import target as target_state  # noqa: E402
from sectors_p import sector_vectors, reduce_sectors, local_orbits, label_str  # noqa: E402

P = 5


def main():
    t0 = time.time()
    D1 = dictionary(5, 1)
    N1 = D1.shape[1]

    def rank1(u):
        u = u / np.linalg.norm(u)
        for r in range(1, 6):
            for idx in itertools.combinations(range(N1), r):
                A = D1[:, list(idx)]
                x, *_ = np.linalg.lstsq(A, u, rcond=None)
                if np.linalg.norm(A @ x - u) < 1e-9:
                    return r
        raise AssertionError("no decomposition in five one-ququint states")

    psi2 = target_state("T5", 2)
    psi4 = target_state("T5", 4)
    reps, order, _, _ = local_orbits("T5", P)
    print(f"single-site Pauli orbits under the order-{order} Clifford stabilizer of |T5>: {reps}", flush=True)
    two = {}

    def two_copy(Q):
        Q = tuple(Q)
        if Q not in two:
            secs = sector_vectors(psi2, P, 2, [list(Q)])
            ranks = {s: (0 if u is None else rank1(u)) for s, _, u in reduce_sectors(psi2, P, 2, [list(Q)])}
            two[Q] = (secs, ranks)
        return two[Q]

    rows = []
    hist = {}
    for cls in itertools.combinations_with_replacement(reps, 4):
        Pstr = list(cls)
        secs4 = sector_vectors(psi4, P, 4, [Pstr])
        keys4 = sorted(secs4)
        v4 = [secs4[k] for k in keys4]
        assert min(np.linalg.norm(v) for v in v4) > 1e-9, "a vanishing one-Pauli sector at m = 4"
        best_terms, best_bounds = None, None
        for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            secA, rA = two_copy([Pstr[a], Pstr[b]])
            secB, rB = two_copy([Pstr[c], Pstr[d]])
            terms = [0] * 5
            bound = [0] * 5
            for sA, vA in secA.items():
                if np.linalg.norm(vA) < 1e-9:
                    continue
                for sB, vB in secB.items():
                    if np.linalg.norm(vB) < 1e-9:
                        continue
                    prod = np.kron(vA, vB).reshape((5,) * 4)
                    prod = np.moveaxis(prod, [0, 1, 2, 3], [a, b, c, d]).reshape(-1)
                    # the product is an eigenvector of P: exactly one of its own
                    # sector projections is nonzero, and that names its sector
                    sp = sector_vectors(prod, P, 4, [Pstr])
                    nz = [s for s, v in sp.items() if np.linalg.norm(v) > 1e-9]
                    assert len(nz) == 1, "a product is not in one sector"
                    k = keys4.index(nz[0])
                    terms[k] += 1
                    bound[k] += rA[sA] * rB[sB]
            if best_bounds is None or sum(bound) < sum(best_bounds):
                best_terms, best_bounds = terms, bound
        rows.append((label_str(Pstr), best_terms, best_bounds))
        hist[min(best_bounds)] = hist.get(min(best_bounds), 0) + 1
        print(f"  {label_str(Pstr)}: nonvanishing sub-sector terms per sector {best_terms}, "
              f"bounds {best_bounds}, total {sum(best_bounds)}", flush=True)
    print(f"{len(rows)} string classes [{time.time() - t0:.0f}s]")
    print(f"histogram of the smallest per-sector bound: {dict(sorted(hist.items()))}")
    fives = [r[0] for r in rows if min(r[2]) == 5]
    print(f"strings with a sector of sub-sector bound 5: {fives}")
    fewer = [r[0] for r in rows if min(r[1]) < 5]
    print(f"strings with a sector of fewer than five nonvanishing sub-sector terms: {fewer or 'none'}")
    two_ranks = sorted({(label_str(list(Q)), tuple(sorted(r.values()))) for Q, (_, r) in two.items()})
    stab = [q for q, r in two_ranks if r == (1, 1, 1, 1, 1)]
    print(f"two-copy strings whose five sectors are all stabilizer states: {stab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
