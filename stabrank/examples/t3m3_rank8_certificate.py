"""Certificate that chi(|T3>^3) <= 8 via an explicit stabilizer decomposition.

The bound was stated in Kocia-Sarovar (arXiv:2003.01130) as a
quadratic-Gauss-sum rank, with the stabilizer-rank value itself marked
uncertain ("8?", supported there by a possibly unconverged Monte Carlo
search).  The eight stabilizer states below (supports 3, 3, 9, and five
of full support 27, phases that are cube roots of unity) were found by
the slow-schedule dense simulated-annealing engine of this package and
provide an explicit witness.

Verification levels:
  1. Each embedded state is matched against the exhaustive three-qutrit
     stabilizer dictionary (largest overlap 1 up to 1e-9), proving it is
     a genuine stabilizer state.
  2. The least-squares residual of |T3>^3 on the span of the eight
     states is checked numerically (~1e-16).
  3. The residual is recomputed in exact arithmetic in sympy and shown
     to be identically zero, so |T3>^3 lies exactly in the span.

Each state is encoded as 27 entries, either None (amplitude zero) or an
exponent k meaning w3^k / sqrt(support), w3 = exp(2 pi i / 3), with
basis index x1 x2 x3 read in ternary (x1 most significant).
"""
import sys

import numpy as np
import sympy as sp

from stabrank.stabilizer_extent import enumerate_stabilizer_states
from stabrank.target_functions import qutrit_complex_magic_state

STATES = [
    [2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 0, 1, 0,
     1, 0, 1, 1],
    [None, None, None, None, 1, None, None, None, None, None, None, None,
     None, None, None, None, None, 0, 1, None, None, None, None, None,
     None, None, None],
    [1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2, 2, 2, 0, 2, 1, 2, 1, 0, 1, 2,
     0, 0, 0, 0],
    [2, 0, 2, 0, 0, 1, 2, 1, 1, 0, 0, 1, 0, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2,
     1, 1, 1, 2],
    [None, None, None, None, 1, None, None, None, None, None, None, None,
     None, None, None, None, None, 1, 0, None, None, None, None, None,
     None, None, None],
    [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 2,
     2, 0, 2, 2],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
     0, 1, 0, 0],
    [None, None, 0, None, 0, None, 0, None, None, None, 0, None, 0, None,
     None, None, None, 0, 1, None, None, None, None, 1, None, 1, None],
]


def exact_columns():
    w3 = sp.exp(2 * sp.pi * sp.I / 3)
    cols = []
    for row in STATES:
        supp = sum(1 for e in row if e is not None)
        cols.append([sp.Integer(0) if e is None else w3 ** e / sp.sqrt(supp)
                     for e in row])
    return cols


def numeric_columns():
    w3 = np.exp(2j * np.pi / 3)
    cols = []
    for row in STATES:
        supp = sum(1 for e in row if e is not None)
        cols.append(np.array([0 if e is None else w3 ** e / np.sqrt(supp)
                              for e in row], dtype=complex))
    return cols


def main():
    numeric = numeric_columns()

    if "--skip-dictionary" not in sys.argv:
        S = enumerate_stabilizer_states(3, 3)
        for i, v in enumerate(numeric):
            ov = np.max(np.abs(S.conj() @ v))
            assert ov > 1 - 1e-9, (i, ov)
        print("all 8 states found in the three-qutrit stabilizer dictionary")

    t3 = qutrit_complex_magic_state(3)
    a = np.column_stack(numeric)
    c, *_ = np.linalg.lstsq(a, t3, rcond=None)
    print(f"numerical residual: {np.linalg.norm(a @ c - t3):.2e}")
    assert np.linalg.norm(a @ c - t3) < 1e-12

    w9 = sp.exp(2 * sp.pi * sp.I / 9)
    t = []
    for idx in range(27):
        x1, r = divmod(idx, 9)
        x2, x3 = divmod(r, 3)
        t.append(w9 ** (x1 + x2 + x3) / sp.sqrt(27))

    cols = exact_columns()
    B = sp.Matrix([[cols[j][i] for j in range(8)] for i in range(27)])
    tv = sp.Matrix(t)
    G = (B.conjugate().T * B).applyfunc(sp.nsimplify)
    w = B.conjugate().T * tv
    csol = G.solve(w)
    resid = B * csol - tv
    r2 = sp.expand((resid.conjugate().T * resid)[0])
    r2 = sp.simplify(sp.nsimplify(sp.radsimp(r2), [sp.sqrt(3)]))
    print("exact residual^2 =", r2)
    if r2 != 0:
        # r2 is an algebraic number; it vanishes iff its minimal
        # polynomial over Q is x.
        mp = sp.minimal_polynomial(r2, sp.Symbol("x"))
        print("minimal polynomial:", mp)
        assert mp == sp.Symbol("x"), "residual is not exactly zero"
    print("EXACT: chi(|T3>^3) <= 8 certified.")


if __name__ == "__main__":
    main()
