"""Certificate: chi(|H>^{ot 6}) >= 4, from chi(|H>^{ot 4}) >= 4 by projection.

The board carried this cell as "projection monotonicity" with the m=4 value
only cited. This script re-runs the m=4 rank-3 exclusion and then applies the
lemma, so the m=6 claim rests on a computation that ran here.

Lemma (projection monotonicity). For any state phi with a nonzero
computational-basis amplitude phi_x, chi(psi (x) phi) >= chi(psi). Proof: take
a decomposition psi (x) phi = sum_j c_j s_j into r stabilizer states and apply
I (x) <x| to both sides. The left side becomes phi_x psi. On the right,
(I (x) <x|) s_j is s_j projected by the stabilizer projector I (x) |x><x| and
stripped of its last factor |x>, which is a stabilizer state or zero. So psi
is a combination of at most r stabilizer states. Here phi = |H>^{ot 2}, whose
amplitude on |00> is cos^2(pi/8) > 0.

Printed claims: CERTIFIED chi(qubit_H^4) >= 4
                CERTIFIED chi(qubit_H^6) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import run_certificate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_certificate(cells=[("qubit_H", 4)], controls=[("qubit_H", 3)],
                                     also=[("qubit_H", 6, 4)]))
