"""Certificate: chi(|S>^{ot 5}) >= 4, from chi(|S>^{ot 3}) >= 4 by projection.

Re-runs the m=3 rank-3 exclusion and applies projection monotonicity twice:
chi(psi (x) phi) >= chi(psi) whenever phi has a nonzero computational
amplitude, since I (x) <x| carries a decomposition of psi (x) phi to one of
psi with no more terms (each stabilizer state goes to a stabilizer state or
zero). |S> has nonzero amplitude on |1>, so phi = |S>^2 qualifies.

Printed claims: CERTIFIED chi(S^3) >= 4
                CERTIFIED chi(S^5) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import run_certificate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_certificate(cells=[("S", 3)], controls=[("N", 2)],
                                     also=[("S", 5, 4)]))
