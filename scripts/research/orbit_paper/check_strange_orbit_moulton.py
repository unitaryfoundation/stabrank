"""Brute-force check: does any element of the Strange Clifford orbit
satisfy the subset-sum / Moulton hypothesis?

For every C in the 216-element single-qutrit Clifford group, compute
the computational-basis amplitudes of C |S>, where
    |S> = (|1> - |2>) / sqrt(2).

The Moulton (subset-sum) hypothesis needed by Prop 2 of the paper is:
there exist indices i, j in {0, 1, 2} with a_i != 0, a_j != 0, and
|a_i| / |a_j| >= 2.

If any orbit representative satisfies the hypothesis, the same
Omega(m / log m) lower bound transfers to |S>^{otimes m} by Clifford
invariance of chi_R.

The expected outcome (per the structural F_max(|S>) = 1/2 argument) is
that no representative satisfies the hypothesis.
"""

from __future__ import annotations

import numpy as np

from stabrank.qutrit_clifford import enumerate_single_qutrit_cliffords


def strange_state() -> np.ndarray:
    """|S> = (|1> - |2>) / sqrt(2) as a length-3 complex vector."""
    psi = np.zeros(3, dtype=complex)
    psi[1] = 1.0 / np.sqrt(2.0)
    psi[2] = -1.0 / np.sqrt(2.0)
    return psi


def amplitude_moduli(psi: np.ndarray, tol: float = 1e-10) -> tuple[float, ...]:
    """Return the moduli (|a_0|, |a_1|, |a_2|) with anything below tol set to 0."""
    return tuple(0.0 if abs(a) < tol else abs(a) for a in psi)


def max_modulus_ratio(moduli: tuple[float, ...]) -> tuple[float, int, int]:
    """Max ratio |a_i| / |a_j| over indices i, j with both nonzero. Returns
    (ratio, i, j); ratio = 0.0 if fewer than two nonzero entries exist."""
    nonzero = [(i, m) for i, m in enumerate(moduli) if m > 0]
    if len(nonzero) < 2:
        return 0.0, -1, -1
    best = (0.0, -1, -1)
    for i, mi in nonzero:
        for j, mj in nonzero:
            if i == j:
                continue
            r = mi / mj
            if r > best[0]:
                best = (r, i, j)
    return best


def canonical_phase(psi: np.ndarray) -> np.ndarray:
    """Multiply by a global phase so the first nonzero entry is real positive.
    Used for deduplicating orbit representatives modulo global phase."""
    for a in psi:
        if abs(a) > 1e-10:
            phase = a / abs(a)
            return psi / phase
    return psi


def main() -> int:
    cliffords = enumerate_single_qutrit_cliffords()
    psi_S = strange_state()

    orbit: list[np.ndarray] = []
    moduli_observed: set[tuple[float, ...]] = set()
    max_ratio_overall = 0.0
    max_ratio_witness: dict[str, object] | None = None

    for C in cliffords:
        psi = canonical_phase(C @ psi_S)
        # Deduplicate by approximate equality on the canonicalized vector.
        is_new = True
        for q in orbit:
            if np.allclose(psi, q, atol=1e-9):
                is_new = False
                break
        if not is_new:
            continue
        orbit.append(psi)

        m = amplitude_moduli(psi)
        # Round to 6 decimals for the observed-pattern set.
        m_rounded = tuple(round(x, 6) for x in m)
        moduli_observed.add(m_rounded)

        ratio, i, j = max_modulus_ratio(m)
        if ratio > max_ratio_overall:
            max_ratio_overall = ratio
            max_ratio_witness = {
                "ratio": ratio,
                "i": i,
                "j": j,
                "moduli": m,
                "psi": psi,
            }

    print(f"Single-qutrit Clifford group size: {len(cliffords)}")
    print(f"Distinct (mod global phase) Strange orbit representatives: {len(orbit)}")
    print()
    print("Distinct amplitude-modulus patterns observed in the orbit:")
    for pat in sorted(moduli_observed):
        print(f"  {pat}")
    print()
    print(f"Maximum |a_i|/|a_j| (over orbit reps and nonzero index pairs):"
          f" {max_ratio_overall:.6f}")
    if max_ratio_witness is not None:
        print(f"  witness: moduli = {max_ratio_witness['moduli']}")
        print(f"           indices (i, j) = "
              f"({max_ratio_witness['i']}, {max_ratio_witness['j']})")
    print()
    if max_ratio_overall >= 2.0:
        print("RESULT: Some Clifford-orbit representative satisfies the")
        print("        Moulton hypothesis |a_i| / |a_j| >= 2.")
        return 0
    else:
        print("RESULT: NO Clifford-orbit representative satisfies the")
        print("        Moulton hypothesis. The maximum observed ratio is")
        print(f"        {max_ratio_overall:.6f} < 2.")
        print()
        print("        The Strange orbit is rigidly equal-modulus on its")
        print("        support, consistent with F_max(|S>) = 1/2 forcing each")
        print("        computational-basis amplitude to satisfy |a_k|^2 <= 1/2.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
