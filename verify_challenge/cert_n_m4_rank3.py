"""Certificate: chi(|N>^{ot 4}) >= 4, by excluding every rank-3 decomposition
over all 7,439,040 four-qutrit stabilizer states.

The dictionary is enumerated as phase codes (qutrit_codes.py, count asserted
against 3^4 prod (3^j + 1)) and never held as complex amplitudes; every
quantity the quotient search needs is an inner product with a stabilizer
state, computed from the codes in chunks (rank_exclusion_codes.py). Rank 2 is
excluded first by one canonicalise-and-sort over the quotient directions,
then rank 3 by the pivot search with one pivot per orbit of the target's
symmetry group, each generator checked at runtime to fix the target and to
permute the dictionary. Every collinear candidate is re-tested in the full
space. The exclusion is numerical and rests on the reported parallelism
margin; the script refuses to claim if it is under 0.01.

Positive control: the same code path must find the known rank-3
decompositions of the Norrell state at m=2.

Printed claims: CERTIFIED chi(N^4) >= 3
                CERTIFIED chi(N^4) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion_codes import certify_m4  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(0 if certify_m4("N") == 3 else 1)
