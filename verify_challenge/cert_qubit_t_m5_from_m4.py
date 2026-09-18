"""Certificate: chi(|T>^{ot 5}) >= 3 for the BK T-type state, from m=4 by projection.

Re-runs the rank-2 exclusion over the 36720 four-qubit stabilizer states and
applies projection monotonicity: chi(psi (x) phi) >= chi(psi) whenever phi has
a nonzero computational amplitude, since I (x) <x| carries a decomposition of
psi (x) phi to one of psi with no more terms. The T-type state at m=2, whose
rank-2 decomposition is verified on the board, is the positive control.

Printed claims: CERTIFIED chi(qubit_T^4) >= 3
                CERTIFIED chi(qubit_T^5) >= 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import MARGIN, PARALLEL, dictionary, psi_for, rank2_search  # noqa: E402


def main():
    D2, D4 = dictionary(2, 2), dictionary(2, 4)
    control, _ = rank2_search(psi_for("qubit_T", 2), D2)
    if control == "RANK1" or not control:
        print("positive control FAILED: no rank-2 decomposition of the T-type state at m=2",
              file=sys.stderr)
        return 1
    print(f"positive control: T-type at m=2 found {len(control)} rank-2 pair(s)")
    pairs, worst = rank2_search(psi_for("qubit_T", 4), D4)
    if pairs == "RANK1" or pairs:
        print(f"qubit_T m=4: rank-2 decomposition FOUND, {pairs}")
        return 1
    slack = (1 - PARALLEL) - worst
    print(f"qubit_T m=4: no pair spans it; closest approach {worst:.10f}, clearing by {slack:.4f}")
    if slack < MARGIN:
        print("margin too thin to stand behind", file=sys.stderr)
        return 1
    print("CERTIFIED chi(qubit_T^4) >= 3")
    print("CERTIFIED chi(qubit_T^5) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
