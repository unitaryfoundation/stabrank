"""Certificate: chi(|N>^{ot 4}) >= 5 for the Norrell state, by slice-and-lift.

chi(|N>^3) = 4: the upper bound is the Lean-checked witness on the board
(LeanProofs.M3StabRank) and the lower bound is the rank-3 exclusion re-run
here. Every amplitude of |N> = (|0> + |1> - 2|2>)/sqrt6 is nonzero, so a
rank-4 decomposition of |N>^4 sliced along its first qutrit gives at each
level a rank-4 decomposition of |N>^3 with independent terms, and the
slices of each term are related by a Pauli operator and cube roots of unity
(slice_lift.py, module docstring). The script lists every rank-4
decomposition of |N>^3 up to the unitary symmetry group and tests the slice
equations for every Pauli assignment. None holds, so chi(|N>^4) >= 5.

Controls: the rank-2 decomposition of |S> lifts to |S>^2, and the rank-3
decompositions of |N>^2 do not lift to |N>^3.

Printed claim: CERTIFIED chi(N^4) >= 5
"""

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import certify_rank3, dictionary  # noqa: E402
from slice_lift import all_decompositions, lift_all, lift_chain  # noqa: E402


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
    workers = max(1, min(4, os.cpu_count() or 1))
    if not certify_rank3("N", 3, D3, workers=workers, symmetry=True):
        print("rank-3 exclusion at m=3 did not hold", file=sys.stderr)
        return 1
    print("chi(N^3) >= 4; with the Lean witness, chi(N^3) = 4")
    counts, gap = lift_chain("N", 3, 4, 4, workers=workers)
    if counts[3] == 0:
        print("no rank-4 decomposition of |N>^3 found, but the board holds one", file=sys.stderr)
        return 1
    if counts[4]:
        print(f"N m=4: {counts[4]} rank-4 decompositions FOUND by lifting")
        return 1
    print(f"[{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(N^4) >= 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
