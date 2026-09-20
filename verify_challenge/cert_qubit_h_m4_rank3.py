"""Certificate: chi(|H>^{ot 4}) >= 4 for the qubit H-type state, excluding rank 3.

The cell sat at 3 <= chi <= 4 after the rank-2 exclusion. Excluding rank 3
settles it at chi = 4, which is the value Bravyi, Smith, and Smolin report.

If psi lies in span(s_i, s_j, s_k) then the images of s_j and s_k in the
quotient C^16 / span(psi, s_i) are parallel (the converse fails only for
dependent triples, which are retested in the full space), so with s_i as a pivot the search
over the 36720 four-qubit stabilizer states is one canonicalise-and-sort per
pivot; rank_exclusion.py documents the method, the symmetry reduction to one pivot
per orbit of the target's Clifford symmetry group, the random projection that
speeds up the sort without being able to lose a configuration, and the margin
the exclusion rests on. Rank 2 is excluded in the same pass. The states are
enumerated by closing the Clifford orbit of |0000>, with the count checked
against 2^4 (2+1)(4+1)(8+1)(16+1).

The H-type state at m=3 is the positive control: its rank-3 decomposition is
verified on the board, the same code must find it, and the script exits
non-zero without printing its claim if it does not.

Printed claim: CERTIFIED chi(qubit_H^4) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import run_certificate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_certificate(cells=[("qubit_H", 4)], controls=[("qubit_H", 3)]))
