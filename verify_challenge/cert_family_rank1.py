"""Certificate: chi >= 2 for family cells, exactly, by showing the state is
not a stabilizer state.

A qubit stabilizer state is |A|^{-1/2} sum_{x in A} i^{l(x)} (-1)^{q(x)} |x>
with A an affine subspace of F_2^n, l linear and q quadratic (the footnote
to Eq. 5 of Qassim, Pashayan, and Gosset, arXiv:2106.07740, citing Dehaene
and De Moor and Van den Nest). Two integer tests refute that form:

  support   A must be closed under x0 + x + x' for x, x' in A, and |A| a
            power of two;
  phase     writing the amplitude at x as a common modulus times i^{e(x)},
            the function e(x0 + x + x') + e(x0) - e(x) - e(x') must be even
            for all x, x' in A, since i^{l(y)}(-1)^{q(y)} satisfies it with
            (-1)^{B(y, y')} for the bilinear part B of q.

Both are decided on the exact data of the target: the support and the
fourth-root exponents come from the sympy vector, not from floats, so no
margin stands anywhere under the claim. A state that passes both tests is
not thereby shown to be a stabilizer state, and the script then refuses to
print the claim for that cell.

The states of the family tracks are the ones this is for: |cat_m> for
m >= 3 (its phase i^{|x|/2} on the even-weight strings is not a Z_4
quadratic form, which is the Pauli-spectrum argument of the paper's
appendix in different clothing). Positive control: |cat_2> and |cat_1> pass
both tests, as stabilizer states must, and the script exits non-zero if
they ever fail.

Printed claims: CERTIFIED chi(cat_3) >= 2
                CERTIFIED chi(cat_4) >= 2
"""

from __future__ import annotations

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stabrank_verify import target_vector  # noqa: E402

CELLS = (("cat", 3), ("cat", 4))
CONTROLS = (("cat", 1), ("cat", 2))


def exact_support_and_phases(vec):
    """(support indices, exponent mod 4 at each) or None if some ratio of
    nonzero amplitudes is not a fourth root of unity (then the state is not
    a stabilizer state for that reason)."""
    supp = [i for i in range(len(vec)) if vec[i] != 0]
    a0 = vec[supp[0]]
    expo = {}
    for i in supp:
        r = sp.nsimplify(sp.simplify(vec[i] / a0))
        for e, root in enumerate((1, sp.I, -1, -sp.I)):
            if sp.simplify(r - root) == 0:
                expo[i] = e
                break
        else:
            return supp, None
    return supp, expo


def refutation(vec):
    """A one-line reason the vector is not a stabilizer state, or None."""
    supp, expo = exact_support_and_phases(vec)
    if expo is None:
        return "a ratio of nonzero amplitudes is not a fourth root of unity"
    S = set(supp)
    if len(S) & (len(S) - 1):
        return f"support has {len(S)} points, not a power of two"
    x0 = supp[0]
    for x in supp:
        for y in supp:
            z = x0 ^ x ^ y
            if z not in S:
                return "support is not an affine subspace of F_2^n"
            if (expo[z] + expo[x0] - expo[x] - expo[y]) % 2:
                return ("phase on the support is not a Z_4 quadratic form with "
                        "even cross terms")
    return None


def main():
    for orbit, m in CONTROLS:
        why = refutation(target_vector(orbit, m))
        if why is not None:
            print(f"positive control FAILED: {orbit}_{m} is a stabilizer state but "
                  f"the test refutes it ({why}); the test is broken", file=sys.stderr)
            return 1
        print(f"positive control: {orbit}_{m} passes both tests, as a stabilizer "
              f"state must")
    ok = True
    for orbit, m in CELLS:
        why = refutation(target_vector(orbit, m))
        if why is None:
            print(f"{orbit}_{m}: both tests pass, so nothing is refuted here",
                  file=sys.stderr)
            ok = False
            continue
        print(f"{orbit}_{m}: {why}")
    if not ok:
        return 1
    for orbit, m in CELLS:
        print(f"CERTIFIED chi({orbit}_{m}) >= 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
