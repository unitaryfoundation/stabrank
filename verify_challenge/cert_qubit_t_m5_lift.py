"""Certificate: chi(|T>^{ot 5}) >= 4 for the BK T-type state, by slice-and-lift.

chi(|T>^4) = 3: the upper bound is the three-term witness of
LeanProofs.QubitTM4 and the lower bound is the rank-2 exclusion re-run here.
Suppose |T>^5 = c_1 s_1 + c_2 s_2 + c_3 s_3 with stabilizer states s_i and
slice along the first qubit, u_i^(k) = (<k| (x) I) s_i. Both amplitudes of
|T> are nonzero, so each slice k gives |T>^4 = sum_i (c_i / alpha_k) u_i^(k);
no u_i^(k) vanishes and no two are parallel, else |T>^4 would have rank at
most 2. So slice 0 is one of the rank-3 decompositions of |T>^4 over the
36720 four-qubit stabilizer states, all of which the pivot search below lists
up to the unitary symmetries of |T>^4 (the order-3 Clifford stabilizer of
|T> on each copy and the permutations of copies, which act on |T>^5 as
I (x) U and preserve slices). For each term, u_i^(1) = mu_i Q_i u_i^(0) with
Q_i a four-qubit Pauli and mu_i a fourth root of unity (slice_lift.py,
`lifts_qubit`), so the slice-1 equation sum_i d_i mu_i Q_i u_i^(0) =
(alpha_1 / alpha_0) |T>^4 must have a solution over the 16 Pauli classes and
4 phases per term. It has none, so chi(|T>^5) >= 4.

Controls: the rank-3 decompositions of |T>^3 lift to |T>^4 (chi stays 3),
and the rank-2 decomposition of |H>^2 does not lift (chi(H^3) = 3).

Printed claim: CERTIFIED chi(qubit_T^5) >= 4
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import MARGIN, PARALLEL, dictionary, psi_for, rank2_search  # noqa: E402
from slice_lift import all_decompositions, lift_all  # noqa: E402


def main():
    t0 = time.time()
    D2, D3, D4 = dictionary(2, 2), dictionary(2, 3), dictionary(2, 4)
    # controls
    decs, _ = all_decompositions("qubit_T", 3, 3, D3, verbose=False)
    if not decs or not lift_all("qubit_T", 3, decs, D3, verbose=False):
        print("positive control FAILED: no rank-3 decomposition of |T>^3 lifts to |T>^4",
              file=sys.stderr)
        return 1
    print(f"positive control: {len(decs)} rank-3 decompositions of |T>^3 up to symmetry, lifts found")
    decs, _ = all_decompositions("qubit_H", 2, 2, D2, verbose=False)
    if not decs or lift_all("qubit_H", 2, decs, D2, verbose=False):
        print("negative control FAILED: a rank-2 decomposition of |H>^2 lifted to |H>^3",
              file=sys.stderr)
        return 1
    print(f"negative control: {len(decs)} rank-2 decomposition(s) of |H>^2, none lifts")
    # chi(T^4) >= 3
    pairs, worst = rank2_search(psi_for("qubit_T", 4), D4)
    if pairs == "RANK1" or pairs or (1 - PARALLEL) - worst < MARGIN:
        print("qubit_T m=4: rank-2 exclusion did not hold", file=sys.stderr)
        return 1
    print(f"qubit_T m=4: no pair spans it; closest approach {worst:.10f}")
    # every rank-3 decomposition of T^4 up to symmetry, and the lift test
    decs, _ = all_decompositions("qubit_T", 4, 3, D4, verbose=False)
    if not decs:
        print("no rank-3 decomposition of |T>^4 found, but the board holds one", file=sys.stderr)
        return 1
    lifted = lift_all("qubit_T", 4, decs, D4)
    if lifted:
        print(f"qubit_T m=5: rank-3 decomposition FOUND by lifting")
        return 1
    print(f"[{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(qubit_T^5) >= 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
