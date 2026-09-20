"""Controls for common.stabilizer_states_in_span.

  1. Flat counts: sum_k p^(n-k) [n choose k]_p affine flats (184 for three
     qutrits, 307 for four qubits).
  2. |T3>^2 = c_0 l_0 + c_1 l_1 + c_2 l_2: span(psi, l_0, l_1) contains
     exactly three stabilizer states and one of them, l_2, is outside
     span(l_0, l_1). This is the one-state completion test in miniature.
  3. rank2_excluded never excludes a genuine weighted sum of two stabilizer
     states (random generator pairs on eight qubits and six qutrits, and
     random dictionary pairs on three and two qutrits and four qubits,
     including equal-modulus pairs with cancellations). The verdict on the
     three-term class sum of the qubit_T 4+4 product is printed for
     information; the control asserts soundness only.
  4. The span of the rank-4 decomposition of |N>^3 contains exactly its four
     terms (research/constructions/inspan.py: no dictionary state beyond
     the terms), and among the 30 rank-4 decompositions of |H>^4 the extra
     count is 0 or 4, with 17 of the 30 having four extra states (this is
     what inspan.py prints when run; the constructions note says 16, a
     miscount of the same list).

Usage: controls.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mergelib import (flats, load_decompositions, product_terms, rank2_excluded,  # noqa: E402
                      stabilizer_states_in_span, target)
from stabrank import generate_random_stabilizer_state  # noqa: E402


def count_flats(p, n):
    return sum(idx.shape[0] for _, idx in flats(p, n))


def main(argv):
    ok = True
    for p, n, expect in ((3, 3, 184), (2, 4, 307), (3, 2, 9 + 3 * 4 + 1)):
        c = count_flats(p, n)
        print(f"flats of F_{p}^{n}: {c} (expected {expect})")
        ok &= c == expect
    decs, _ = load_decompositions("T3", 2, 3)
    lines, coeffs = decs[0]
    psi = target("T3", 2)
    found, diag = stabilizer_states_in_span(np.column_stack([psi, lines[0], lines[1]]), 3, 2)
    Qk, _ = np.linalg.qr(np.column_stack(lines[:2]))
    outside = [u for u, _ in found if np.linalg.norm(u - Qk @ (Qk.conj().T @ u)) > 1e-6]
    l2 = lines[2] / np.linalg.norm(lines[2])
    hit = any(abs(abs(np.vdot(u, l2)) - 1) < 1e-6 for u in outside)
    print(f"T3^2 completion control: {len(found)} states in span(psi, l0, l1), {len(outside)} outside "
          f"span(l0, l1), l2 recovered: {hit}")
    ok &= len(found) == 3 and len(outside) == 1 and hit
    rng = np.random.default_rng(5)
    bad = 0
    trials = 0
    for pp, nn in ((2, 8), (3, 6)):
        for trial in range(40):
            s1 = generate_random_stabilizer_state(nn, p=pp, seed=int(rng.integers(1 << 30)))
            s2 = generate_random_stabilizer_state(nn, p=pp, seed=int(rng.integers(1 << 30)))
            a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
            exc, why = rank2_excluded(a * s1 + b * s2, pp)
            bad += exc
            trials += 1
    from rank_exclusion import dictionary
    for pp, nn in ((3, 3), (2, 4), (3, 2)):
        D = dictionary(pp, nn)
        for trial in range(300):
            i, j = rng.choice(D.shape[1], size=2, replace=False)
            a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
            if trial % 3 == 0:
                b = a * np.exp(2j * np.pi * rng.integers(4) / 4)   # equal moduli: cancellations possible
            exc, why = rank2_excluded(a * D[:, i] + b * D[:, j], pp)
            bad += exc
            trials += 1
    decsT, _ = load_decompositions("qubit_T", 4, 3)
    termsT, coeffsT = product_terms(decsT[0], decsT[0])
    cls = [i for i in range(9) if abs(coeffsT[i] - coeffsT[0]) < 1e-9]
    vT = sum(termsT[i] for i in cls)
    excT, whyT = rank2_excluded(vT, 2)
    print(f"rank2_excluded control: {bad} of {trials} genuine rank-2 vectors wrongly excluded; qubit_T class "
          f"{cls} sum excluded: {excT} ({whyT})")
    ok &= bad == 0
    decs, _ = load_decompositions("N", 3, 4)
    found, diag = stabilizer_states_in_span(np.column_stack(decs[0][0]), 3, 3)
    print(f"N^3 rank-4 span: {len(found)} stabilizer states (expected 4); {diag['flats']} flats, "
          f"{diag['flagged']} flagged")
    ok &= len(found) == 4
    decs, _ = load_decompositions("qubit_H", 4, 4)
    counts = []
    for u, _ in decs:
        found, _ = stabilizer_states_in_span(np.column_stack(u), 2, 4)
        counts.append(len(found) - 4)
    print(f"H^4 rank-4 spans: extra states per decomposition {sorted(set(counts))}, "
          f"{sum(c == 4 for c in counts)} of {len(counts)} with four extra (expected 17 of 30)")
    ok &= sorted(set(counts)) == [0, 4] and sum(c == 4 for c in counts) == 17
    print("controls", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
