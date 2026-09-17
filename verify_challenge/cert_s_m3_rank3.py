"""Certificate: chi(|S>^{ot 3}) >= 4, by excluding every rank-3 decomposition.

This is the m=3 exhaustive value of arXiv:2605.28586, reproduced from scratch
here so that the cell no longer rests on a citation. With the rank-4 upper
bound already machine-checked in Lean, the cell is settled at chi = 4.

If psi lies in span(s_i, s_j, s_k) then the images of s_j and s_k in the
quotient C^27 / span(psi, s_i) are parallel (the converse fails only for
dependent triples, which are retested in the full space), so with s_i as a pivot the search
over pairs (j, k) is one canonicalise-and-sort per pivot rather than a solve
per triple; rank_exclusion.py documents the method, the random projection that
speeds up the sort without being able to lose a configuration, and the margin
the exclusion rests on. Every collinear candidate is then tested in the full
space. Rank 2 is excluded in the same pass, since a state s_j landing in
span(psi, s_i) shows up as a zero quotient direction.

The Norrell state at m=2 is the positive control: its rank-3 decomposition is
proved in Lean, the same code must find it, and the script exits non-zero
without printing its claim if it does not.

Printed claim: CERTIFIED chi(S^3) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import run_certificate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_certificate(cells=[("S", 3)], controls=[("N", 2)]))
