"""Certificate: chi(|T>^{ot 3}) >= 3 for the BK T-type state, excluding rank 2.

A rank-3 decomposition at m=3 was found by the annealer and is verified on the
board, so excluding rank 2 settles the cell at chi = 3 exactly.

psi lies in the span of two stabilizer states exactly when those two have
parallel components off psi, so quotienting by psi once turns the pair search
over the 1080 three-qubit stabilizer states into one canonicalise-and-sort.
Rank 1 is excluded in the same pass. The states are enumerated by closing the
Clifford orbit of |000>, and the count is checked against 2^3 (2+1)(4+1)(8+1).

The exclusion is numerical and rests on a margin: the closest any two
candidate directions come to parallel is reported and must clear the
threshold by at least 0.01, against float error around 1e-15 on these inputs.

The T-type state at m=2 is the positive control: it has a rank-2
decomposition, verified on the board, which the same code must find.

Printed claim: CERTIFIED chi(qubit_T^3) >= 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import MARGIN, PARALLEL, dictionary, psi_for, rank2_search  # noqa: E402


def main():
    D2, D3 = dictionary(2, 2), dictionary(2, 3)
    print(f"2 qubits: {D2.shape[1]} states; 3 qubits: {D3.shape[1]} states")
    control, _ = rank2_search(psi_for("qubit_T", 2), D2)
    if control == "RANK1" or not control:
        print("positive control FAILED: no rank-2 decomposition of the T-type state "
              "at m=2, where one is verified on the board; the search is broken",
              file=sys.stderr)
        return 1
    print(f"positive control: T-type at m=2 found {len(control)} rank-2 pair(s)\n")
    pairs, worst = rank2_search(psi_for("qubit_T", 3), D3)
    if pairs == "RANK1" or pairs:
        print(f"qubit_T m=3: rank-2 decomposition FOUND, {pairs}")
        return 1
    slack = (1 - PARALLEL) - worst
    print(f"qubit_T m=3: no pair spans it. Closest approach to parallel {worst:.10f}, "
          f"clearing the threshold by {slack:.4f}")
    if slack < MARGIN:
        print("margin too thin to stand behind", file=sys.stderr)
        return 1
    print("CERTIFIED chi(qubit_T^3) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
