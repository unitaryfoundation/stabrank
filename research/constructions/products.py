"""Tensor products of known decompositions, and whether any two product terms
merge into one stabilizer state.

|M>^(a+b) = |M>^a (x) |M>^b gives the decomposition with terms s_i (x) t_j and
coefficients a_i b_j from decompositions (a_i, s_i) of |M>^a and (b_j, t_j) of
|M>^b. Its coefficients are the unique ones (the terms are independent), so
the only way to shorten it without a new search is a merge: two weighted
terms whose sum is itself a stabilizer state replace two terms by one. A
three-into-two merge needs a rank-2 test in the (a+b)-qudit dictionary and
is not run here.

"Clifford-twisted" products add nothing: every unitary symmetry of |M>^(a+b)
is a local Clifford on each copy times a copy permutation, and it carries a
product decomposition across one bipartition of the copies to a product
decomposition across another. Independent symmetries on the two factors are
already quotiented out by taking one decomposition per orbit on each side,
so the pairs tested here are every product decomposition up to symmetry.

Usage: products.py ORBIT A B   (decompositions of |M>^A and |M>^B from data/)
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORBIT_P, confirm_stabilizer, is_stabilizer_batch, load_decompositions, target  # noqa: E402

RANK = {("N", 1): 2, ("N", 2): 3, ("N", 3): 4, ("H3", 1): 2, ("H3", 2): 3, ("H3", 3): 4,
        ("S", 1): 2, ("S", 2): 2, ("S", 3): 4, ("S", 4): 4, ("T3", 1): 3, ("T3", 2): 3,
        ("qubit_H", 1): 2, ("qubit_H", 3): 3, ("qubit_H", 4): 4, ("qubit_T", 1): 2,
        ("qubit_T", 3): 3, ("qubit_T", 4): 3}


def main(argv):
    orbit, A, B = argv[1], int(argv[2]), int(argv[3])
    p = ORBIT_P[orbit]
    m = A + B
    psi = target(orbit, m)
    decsA, _ = load_decompositions(orbit, A, RANK[(orbit, A)])
    decsB, _ = load_decompositions(orbit, B, RANK[(orbit, B)])
    t0 = time.time()
    n_products = 0
    merges = []
    for ia, (sA, cA) in enumerate(decsA):
        for ib, (sB, cB) in enumerate(decsB):
            terms = [np.kron(s, t) for s in sA for t in sB]
            coeffs = np.array([x * y for x in cA for y in cB])
            V = np.column_stack(terms)
            assert np.linalg.norm(V @ coeffs - psi) < 1e-8
            n_products += 1
            pairs = list(itertools.combinations(range(len(terms)), 2))
            sums = np.array([coeffs[i] * terms[i] + coeffs[j] * terms[j] for i, j in pairs])
            mask = is_stabilizer_batch(sums, p)
            for k in np.flatnonzero(mask):
                if confirm_stabilizer(sums[k], p, m) is not None:
                    merges.append((ia, ib, pairs[k]))
    r = RANK[(orbit, A)] * RANK[(orbit, B)]
    print(f"{orbit} m={m} = {A}+{B}: {len(decsA)} x {len(decsB)} product decompositions of rank {r}, "
          f"{r * (r - 1) // 2} weighted pairs each; {len(merges)} pairs merge into a stabilizer state "
          f"[{time.time() - t0:.0f}s]")
    for ia, ib, pr in merges[:20]:
        print(f"  product ({ia}, {ib}) terms {pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
