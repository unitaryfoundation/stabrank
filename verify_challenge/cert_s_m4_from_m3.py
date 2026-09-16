"""Certificate: chi(|S>^{ot 4}) >= 4, from chi(|S>^{ot 3}) >= 4 by projection.

The board carried this cell as "projection monotonicity from m=3" with the
m=3 value itself only cited. This script re-runs the m=3 exclusion and then
applies the lemma, so the m=4 claim rests on a computation that ran here.

Lemma (projection monotonicity). For any state phi with a nonzero
computational-basis amplitude phi_x, chi(psi (x) phi) >= chi(psi). Proof: take
a decomposition psi (x) phi = sum_j c_j s_j into r stabilizer states and apply
I (x) <x| to both sides. The left side becomes phi_x psi. On the right,
(I (x) <x|) s_j is the stabilizer state s_j projected by the stabilizer
projector I (x) |x><x| and then stripped of its last factor |x>, which is a
stabilizer state or zero. So psi is a combination of at most r stabilizer
states. Every magic state on the board has a nonzero computational amplitude,
so the lemma applies with phi the single-copy state.

Printed claims: CERTIFIED chi(S^3) >= 4
                CERTIFIED chi(S^4) >= 4
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import run_certificate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_certificate(cells=[("S", 3)], controls=[("N", 2)],
                                     also=[("S", 4, 4)]))
