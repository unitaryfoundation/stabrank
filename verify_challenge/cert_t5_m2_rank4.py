"""Certificate: chi(|T5>^{ot 2}) >= 5, by excluding ranks 2, 3 and 4.

Two ququints carry 3,900 stabilizer states. Rank 3 (and rank 2 in the same
pass) is excluded by the pivot-and-quotient search of rank_exclusion.py
from one pivot per orbit of the symmetry group of |T5>^2: the order-5
Clifford stabilizer of |T5> on each copy, the swap of the copies and one
antiunitary symmetry, a group of order 100 that splits the dictionary into
66 orbits. Rank 4 is excluded by the pivot-pair search of slice_lift.py
(`decompositions_with_pivot`, rank 4): with the pivot s_i and a partner
s_j fixed, |T5>^2 lies in span(s_i, s_j, s_a, s_b) only if the images of
s_a and s_b modulo span(psi, s_i, s_j) are parallel, which is one
canonicalise-and-sort per (pivot, partner), and every collinear candidate is
confirmed or rejected by a solve in the full space. The pivots are the 98
representatives of the unitary symmetry group (order 50), and for each
pivot the partners are one state per orbit of the pivot's stabilizer,
which is sound because a symmetry fixing the pivot carries decompositions
through the pivot to decompositions through the pivot. Rank 3 is excluded
first so that every rank-4 decomposition is minimal and the listing is
complete.

Both exclusions are numerical and rest on margins: the rank-3 pass reports
the closest approach to parallel among non-parallel quotient directions and
refuses to certify when it clears the threshold by less than
rank_exclusion.MARGIN; the rank-4 pass groups directions by the same
threshold and confirms every candidate by a least-squares residual.

Controls, both run before the claim and both fatal if they fail: the
rank-3 pivot search on one ququint must find the rank-3 decompositions of
|T5> (the board holds one, verified symbolically), and the rank-4 pivot-pair
search from the pivot |00> must find the product decomposition
{|00>, |0+>, |+0>, |++>} of (|+> - |0>)^2, the two-copy Norrell-type state
|N5>^2, whose rank is 4.

Printed claims: CERTIFIED chi(T5^2) >= 5
                CERTIFIED chi(T5^3) >= 5 and chi(T5^4) >= 5 (projection monotonicity)
"""

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import certify_rank3, dictionary  # noqa: E402
from slice_lift import all_decompositions, decompositions_with_pivot  # noqa: E402


def _index(D):
    def key(v):
        v = np.asarray(v, dtype=complex)
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        return (np.round(v, 6) + 0.0).tobytes()
    return key, {key(D[:, i]): i for i in range(D.shape[1])}


def rank4_control(D2):
    """The product decomposition of |N5>^2 must be found from the pivot |00>."""
    key, index = _index(D2)
    zero = np.eye(5, dtype=complex)[0]
    plus = np.ones(5, dtype=complex) / np.sqrt(5)
    n5 = plus - zero
    psi = np.kron(n5, n5)
    pivot = index[key(np.kron(zero, zero))]
    want = tuple(sorted(index[key(np.kron(a, b))] for a in (zero, plus) for b in (zero, plus)))
    decs = decompositions_with_pivot(psi, D2, pivot, 4)
    return want in decs, len(decs)


def main():
    t0 = time.time()
    D1, D2 = dictionary(5, 1), dictionary(5, 2)
    print(f"1 ququint: {D1.shape[1]} stabilizer states; 2 ququints: {D2.shape[1]}")

    decs, _ = all_decompositions("T5", 1, 3, D1, verbose=False)
    if not decs:
        print("positive control FAILED: no rank-3 decomposition of |T5> found, where the "
              "board holds one; the pivot search is broken", file=sys.stderr)
        return 1
    print(f"positive control: {len(decs)} rank-3 decomposition(s) of |T5> found")
    ok, n = rank4_control(D2)
    if not ok:
        print("positive control FAILED: the product decomposition of |N5>^2 was not found "
              "by the rank-4 pivot-pair search; the search is broken", file=sys.stderr)
        return 1
    print(f"positive control: rank-4 search from pivot |00> lists {n} decomposition(s) of "
          f"|N5>^2, the product decomposition among them\n")

    if not certify_rank3("T5", 2, D2, workers=1):
        return 1
    print(f"T5 m=2: rank <= 3 excluded [{time.time() - t0:.0f}s]\n")

    decs, reps = all_decompositions("T5", 2, 4, D2, verbose=True, workers=1)
    if decs:
        print(f"T5 m=2: rank-4 decomposition FOUND, states {decs[0]}")
        return 1
    print(f"T5 m=2: no rank-4 decomposition from any of the {len(reps)} pivot "
          f"representatives [{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(T5^2) >= 5")
    # Projection monotonicity (Lean: stabRankP_powVecP_mono in
    # lean_proofs/LeanProofs/Stabilizer/SliceP.lean): every amplitude of |T5>
    # is nonzero, so slicing carries a rank-r decomposition of T5^(m+1) to one
    # of T5^m, and the m=2 exclusion bounds the m=3 and m=4 cells as well.
    for m in (3, 4):
        print(f"CERTIFIED chi(T5^{m}) >= 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
