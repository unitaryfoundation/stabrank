"""Certificate: chi(|T3>^{ot 4}) >= 6, from chi(|T3>^{ot 3}) >= 6 by projection.

The T3 cell at m=4 had no lower bound at all. This script runs the m=3
rank-6 certificate (cert_t3m3_rank6.py, unchanged) and then applies
projection monotonicity, so the m=4 claim rests on that computation having
run here rather than on the board entry for m=3.

Lemma (projection monotonicity). For any state phi with a nonzero
computational-basis amplitude phi_x, chi(psi (x) phi) >= chi(psi). Proof: take
a decomposition psi (x) phi = sum_j c_j s_j into r stabilizer states and apply
I (x) <x| to both sides. The left side becomes phi_x psi. On the right,
(I (x) <x|) s_j is s_j projected by the stabilizer projector I (x) |x><x| and
stripped of its last factor |x>, which is a stabilizer state or zero. So psi
is a combination of at most r stabilizer states. |T3> has all three
computational amplitudes nonzero, so the lemma applies with phi = |T3>.

Printed claims: CERTIFIED chi(T3^3) >= 6
                CERTIFIED chi(T3^4) >= 6
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import cert_t3m3_rank6
    code = cert_t3m3_rank6.main()
    if code != 0:
        raise SystemExit(code)
    print("CERTIFIED chi(T3^4) >= 6")
