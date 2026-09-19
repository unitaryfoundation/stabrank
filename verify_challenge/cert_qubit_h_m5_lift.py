"""Certificate: chi(|H>^{ot 5}) >= 5 for the qubit H-type state, by slice-and-lift.

chi(|H>^4) = 4: the upper bound is the Lean-checked tensor square of the
two-copy witness (LeanProofs.QubitHStabRank) and the lower bound is the
rank-3 exclusion over the 36720 four-qubit stabilizer states, re-run here.
Both amplitudes of |H> = cos(pi/8)|0> + sin(pi/8)|1> are nonzero, so a
rank-4 decomposition of |H>^5 sliced along its first qubit has slice 0 a
rank-4 decomposition of |H>^4 with independent terms and slice 1 equal,
term by term, to a Pauli image times a fourth root of unity (slice_lift.py,
`lifts_qubit`). The script lists every rank-4 decomposition of |H>^4 up to
the unitary symmetry group (30, up to symmetry) and tests the slice-1
equation for every Pauli assignment. None holds, so chi(|H>^5) >= 5.

Controls: the rank-3 decompositions of |T>^3 lift to |T>^4 (chi stays 3),
and the rank-2 decomposition of |H>^2 does not lift to |H>^3.

Printed claim: CERTIFIED chi(qubit_H^5) >= 5
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
    D2, D3, D4 = dictionary(2, 2), dictionary(2, 3), dictionary(2, 4)
    decs, _ = all_decompositions("qubit_T", 3, 3, D3, verbose=False)
    if not decs or not lift_all("qubit_T", 3, decs, D3, verbose=False):
        print("positive control FAILED: no rank-3 decomposition of |T>^3 lifts to |T>^4", file=sys.stderr)
        return 1
    decs, _ = all_decompositions("qubit_H", 2, 2, D2, verbose=False)
    if not decs or lift_all("qubit_H", 2, decs, D2, verbose=False):
        print("negative control FAILED: a rank-2 decomposition of |H>^2 lifted to |H>^3", file=sys.stderr)
        return 1
    print("controls: |T>^3 lifts to |T>^4, |H>^2 does not lift to |H>^3")
    workers = max(1, min(4, os.cpu_count() or 1))
    if not certify_rank3("qubit_H", 4, D4, workers=workers, symmetry=True):
        print("rank-3 exclusion at m=4 did not hold", file=sys.stderr)
        return 1
    print("chi(qubit_H^4) >= 4; with the Lean witness, chi(qubit_H^4) = 4")
    counts, gap = lift_chain("qubit_H", 4, 4, 5, workers=workers)
    if counts[4] == 0:
        print("no rank-4 decomposition of |H>^4 found, but the board holds one", file=sys.stderr)
        return 1
    if counts[5]:
        print(f"qubit_H m=5: {counts[5]} rank-4 decompositions FOUND by lifting")
        return 1
    print(f"[{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(qubit_H^5) >= 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
