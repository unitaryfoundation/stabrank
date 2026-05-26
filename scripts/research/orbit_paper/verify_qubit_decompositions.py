"""Numerical audit of the qubit T-type stabilizer decompositions in the paper.

For each qubit (orbit, m) in Appendix~\\ref{app:qubit-t}, this script:
  1. Encodes the stabilizer-state definitions and coefficients
     exactly as stated in main.tex.
  2. Builds the target |T>^{otimes m} with |T> = cos(beta)|0> +
     e^{i pi/4} sin(beta)|1>, cos(2 beta) = 1/sqrt(3).
  3. Verifies the claimed identity sum_i c_i |s_i> = |T>^{otimes m}
     to floating-point precision.

A PASS means the algebraic identity in the appendix holds exactly. A
FAIL means there's a transcription or proof error in the paper.

Self-contained: only needs numpy.
"""
from __future__ import annotations

import itertools

import numpy as np


def t_state() -> np.ndarray:
    """|T> = cos(beta)|0> + e^{i pi/4} sin(beta)|1>, cos(2 beta) = 1/sqrt(3)."""
    beta = np.arccos(1.0 / np.sqrt(3)) / 2.0
    return np.array(
        [np.cos(beta), np.exp(1j * np.pi / 4) * np.sin(beta)],
        dtype=complex,
    )


def kron_n(v: np.ndarray, n: int) -> np.ndarray:
    out = v
    for _ in range(n - 1):
        out = np.kron(out, v)
    return out


def report(name: str, target: np.ndarray, terms) -> bool:
    s = np.zeros_like(target)
    for c, v in terms:
        s = s + c * v
    res = float(np.linalg.norm(s - target))
    nrm = float(np.linalg.norm(target))
    rel = res / nrm if nrm > 0 else float("inf")
    status = "PASS" if rel < 1e-9 else "FAIL"
    print(f"  [{status}]  {name}:  ||sum - target|| = {res:.3e}  "
          f"(rel {rel:.3e})  chi <= {len(terms)}")
    return rel < 1e-9


def cz_two_qubits() -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def plus_state() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)


def plus_i_state() -> np.ndarray:
    return np.array([1.0, 1j], dtype=complex) / np.sqrt(2)


def verify_qubit_t_m2() -> bool:
    """chi(|T>^2) <= 2 via Eq. (eq:qubit-t-m2-id).

    |T>^2 = cbar * CZ |++> + c * CZ |+i,+i>
    where c = (1 + 1/sqrt(3))/2 + i (1 - 1/sqrt(3))/2.
    """
    CZ = cz_two_qubits()
    pp = np.kron(plus_state(), plus_state())
    pipi = np.kron(plus_i_state(), plus_i_state())
    tau1 = CZ @ pp
    tau2 = CZ @ pipi

    alpha = (1.0 + 1.0 / np.sqrt(3.0)) / 2.0
    delta = (1.0 - 1.0 / np.sqrt(3.0)) / 2.0
    c = alpha + 1j * delta

    target = kron_n(t_state(), 2)
    return report("Qubit T-type m=2 (chi<=2)", target,
                  [(np.conj(c), tau1), (c, tau2)])


def fully_connected_cz(n: int) -> np.ndarray:
    """Diagonal matrix of CZ_{i,j} for every pair i < j on n qubits.

    Diagonal entry at index x is (-1)^{sum_{i<j} x_i x_j}, computed with
    MSB-first bit ordering (matching np.kron and the rest of this file).
    """
    diag = np.empty(1 << n, dtype=complex)
    for idx in range(1 << n):
        bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
        parity = 0
        for i in range(n):
            for j in range(i + 1, n):
                parity += bits[i] * bits[j]
        diag[idx] = (-1.0) ** (parity % 2)
    return np.diag(diag)


def minus_state() -> np.ndarray:
    return np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)


def minus_i_state() -> np.ndarray:
    return np.array([1.0, -1j], dtype=complex) / np.sqrt(2)


def verify_qubit_t_m3() -> bool:
    """chi(|T>^3) <= 3 via Eq. (eq:qubit-t-m3-id).

    |T>^3 = c_0 |000> + c_1 Lambda |--->
                       + c_2 Lambda |-i, -i, -i>
    where Lambda = CZ_{12} CZ_{23} CZ_{13},
          c_0 = (2/sqrt 3) cos(beta),
          c_1 = (2/sqrt 3) sin(beta) e^{-i 5 pi / 6},
          c_2 = (2/sqrt 3) sin(beta) e^{+i 5 pi / 6}.
    """
    beta = np.arccos(1.0 / np.sqrt(3.0)) / 2.0
    Lambda = fully_connected_cz(3)

    s0 = np.zeros(8, dtype=complex)
    s0[0] = 1.0
    minus3 = kron_n(minus_state(), 3)
    minusi3 = kron_n(minus_i_state(), 3)
    s1 = Lambda @ minus3
    s2 = Lambda @ minusi3

    two_over_sq3 = 2.0 / np.sqrt(3.0)
    c0 = two_over_sq3 * np.cos(beta)
    c1 = two_over_sq3 * np.sin(beta) * np.exp(-1j * 5 * np.pi / 6.0)
    c2 = two_over_sq3 * np.sin(beta) * np.exp(+1j * 5 * np.pi / 6.0)

    target = kron_n(t_state(), 3)
    return report("Qubit T-type m=3 (chi<=3)", target,
                  [(c0, s0), (c1, s1), (c2, s2)])


def verify_qubit_t_m4() -> bool:
    """chi(|T>^4) <= 3 via the rank-3 decomposition of Appendix B.2.

    Three 4-qubit stabilizer states:
      |sigma_1> = 2^{-3/2} sum_{x in F_2^4 : x_1 = x_3}
                   i^{x_1} (-1)^{x_2 x_4} |x>
      |sigma_2> = 2^{-2}  sum_{x in F_2^4}
                   i^{x_2 + x_4} (-1)^{x_1 x_3 + x_2 x_4} |x>
      |sigma_3> = 2^{-3/2} sum_{x in F_2^4 : x_2 = x_4}
                   i^{x_1 + x_2 + x_3} (-1)^{x_1 x_3} |x>
    with coefficients c_1 = (2/3) e^{i pi/12}, c_2 = 2/3,
    c_3 = (2/3) e^{-i pi/12}.

    Indexing convention: |x_1 x_2 x_3 x_4> -> index
      8*x_1 + 4*x_2 + 2*x_3 + x_4 (MSB-first, matching np.kron).
    """
    dim = 16
    sigma_1 = np.zeros(dim, dtype=complex)
    sigma_2 = np.zeros(dim, dtype=complex)
    sigma_3 = np.zeros(dim, dtype=complex)
    for x in itertools.product([0, 1], repeat=4):
        x1, x2, x3, x4 = x
        idx = 8 * x1 + 4 * x2 + 2 * x3 + x4
        if x1 == x3:
            sigma_1[idx] = (1j) ** x1 * (-1) ** (x2 * x4)
        sigma_2[idx] = (1j) ** (x2 + x4) * (-1) ** (x1 * x3 + x2 * x4)
        if x2 == x4:
            sigma_3[idx] = (1j) ** (x1 + x2 + x3) * (-1) ** (x1 * x3)
    sigma_1 /= 2.0 ** 1.5
    sigma_2 /= 2.0 ** 2
    sigma_3 /= 2.0 ** 1.5

    c1 = (2.0 / 3.0) * np.exp(1j * np.pi / 12.0)
    c2 = 2.0 / 3.0
    c3 = (2.0 / 3.0) * np.exp(-1j * np.pi / 12.0)

    target = kron_n(t_state(), 4)
    return report("Qubit T-type m=4 (chi<=3)", target,
                  [(c1, sigma_1), (c2, sigma_2), (c3, sigma_3)])


def main() -> int:
    print("Verifying paper appendix qubit T-type decompositions:")
    results = []
    for fn in [verify_qubit_t_m2, verify_qubit_t_m3, verify_qubit_t_m4]:
        try:
            ok = fn()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [ERROR]  {fn.__name__}: {exc}")
            ok = False
        results.append((fn.__name__, ok))

    print("\nSummary:")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name}")
    print(f"\n  {n_pass}/{len(results)} decompositions verified.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
