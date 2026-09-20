"""Certificate: chi(|S>^{ot 5}) >= 5 for the strange state, by slice-and-lift twice.

chi(|S>^3) = chi(|S>^4) = 4: the rank-2 decomposition of |S>^2 (found here)
tensored with itself gives the upper bounds, the rank-3 exclusion at m=3
(re-run here) gives chi(|S>^3) >= 4, and projection onto <1| on one qutrit
carries it to chi(|S>^4) >= 4.

Slice a rank-4 decomposition of |S>^(m+1) along its first qutrit. The
amplitude of |1> in |S> = (|1> - |2>)/sqrt2 is nonzero, so the slice at
k = 1 is a rank-4 decomposition of |S>^m with independent terms (a
dependent or vanishing term would put |S>^m in the span of three
stabilizer states). Each term's slices are related by a Pauli operator Q
and cube roots of unity, u^(2) = mu Q u^(1) and u^(0) = nu Q^-1 u^(1)
(slice_lift.py, module docstring). So the slice-2 equation sum_i d_i mu_i
Q_i u_i = -|S>^m and the homogeneous slice-0 equation sum_i d_i nu_i Q_i^-1
u_i = 0 must both hold for some choice of 3^m Pauli classes
and phases per term. A unitary symmetry U of |S>^m acts as I (x) U on
|S>^(m+1) and preserves slices, so one decomposition per orbit suffices.

The script lists every rank-4 decomposition of |S>^3 up to the unitary
symmetry group (a pivot-pair search with the partner loop reduced by the
pivot's stabilizer), lifts each to |S>^4 (the product decomposition must
appear, which is the internal positive control), and lifts every resulting
|S>^4 decomposition to |S>^5. Nothing lifts, so chi(|S>^5) >= 5.

External controls: the rank-2 decomposition of |S> lifts to |S>^2 (the
slice-0 equation is homogeneous there too), and the rank-3 decompositions
of |N>^2 do not lift to |N>^3 (chi(N^3) = 4).

Printed claims: CERTIFIED chi(S^5) >= 5
                CERTIFIED chi(S^6) >= 5 (projection monotonicity)
"""

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import certify_rank3, dictionary, psi_for, rank2_search  # noqa: E402
from slice_lift import all_decompositions, lift_all, lifts, confirm_lift  # noqa: E402
from stabrank_verify import orbit_state  # noqa: E402


def main():
    t0 = time.time()
    D1, D2, D3 = dictionary(3, 1), dictionary(3, 2), dictionary(3, 3)
    decs, _ = all_decompositions("S", 1, 2, D1, verbose=False)
    if not decs or not lift_all("S", 1, decs, D1, verbose=False):
        print("positive control FAILED: the rank-2 decomposition of |S> does not lift", file=sys.stderr)
        return 1
    decs, _ = all_decompositions("N", 2, 3, D2, verbose=False)
    if not decs or lift_all("N", 2, decs, D2, verbose=False):
        print("negative control FAILED: a rank-3 decomposition of |N>^2 lifted", file=sys.stderr)
        return 1
    print("controls: |S> lifts to |S>^2, |N>^2 does not lift to |N>^3")
    pairs, _ = rank2_search(psi_for("S", 2), D2)
    if pairs == "RANK1" or not pairs:
        print("no rank-2 decomposition of |S>^2 found", file=sys.stderr)
        return 1
    print(f"chi(S^2) <= 2 ({len(pairs)} pairs), so chi(S^4) <= 4 by tensoring")
    if not certify_rank3("S", 3, D3, workers=1, symmetry=True):
        print("rank-3 exclusion at m=3 did not hold", file=sys.stderr)
        return 1
    print("chi(S^3) >= 4, hence chi(S^4) >= 4 by projection; both cells are exactly 4")
    workers = max(1, min(4, os.cpu_count() or 1))
    decs3, _ = all_decompositions("S", 3, 4, D3, workers=workers)
    if not decs3:
        print("no rank-4 decomposition of |S>^3 found, but one exists", file=sys.stderr)
        return 1
    L4 = lift_all("S", 3, decs3, D3)
    if not L4:
        print("internal control FAILED: no rank-4 decomposition of |S>^3 lifts to |S>^4", file=sys.stderr)
        return 1
    alpha = np.array([complex(x) for x in orbit_state("S")]).ravel()
    psi4 = psi_for("S", 4)
    n5, gap = 0, float("inf")
    for terms in L4:
        A = np.column_stack(terms)
        d, *_ = np.linalg.lstsq(A, psi4, rcond=None)
        if np.linalg.norm(A @ d - psi4) > 1e-9:
            print("a lifted |S>^4 decomposition does not reproduce the target", file=sys.stderr)
            return 1
        L5, (_, g) = lifts(terms, d, alpha, psi4, 4)
        gap = min(gap, g)
        for t5 in L5:
            if confirm_lift(t5, alpha, psi4) > 1e-8:
                print("a lift failed its confirmation", file=sys.stderr)
                return 1
        n5 += len(L5)
    if n5:
        print(f"S m=5: {n5} rank-4 decompositions FOUND by lifting")
        return 1
    print(f"S m=4 -> 5: {len(L4)} decompositions, 0 lifts; every rejected Pauli assignment "
          f"misses the slice equation by at least {gap:.2e}")
    print(f"[{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(S^5) >= 5")
    # projection onto <1| of one qutrit carries a decomposition of S^6 to one of S^5
    print("CERTIFIED chi(S^6) >= 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
