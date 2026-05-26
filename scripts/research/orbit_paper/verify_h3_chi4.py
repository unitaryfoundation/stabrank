"""Reconstruct the co-author's chi=4 decomposition of |H_3>^otimes 3 from the
markdown spec in wiki/pages/solution_qutrit_real_k4_n3_20260430_170258_decomposition.md
and verify it reconstructs |H_3>^otimes 3 to floating-point precision.

Stabilizer state form:
    |S> = (1/sqrt(3^k)) sum_{y in F_3^k} omega^{Q(y)} |x_0 + W y>
where omega = exp(2pi i / 3).
"""

import itertools
import numpy as np

from stabrank.target_functions import qutrit_hadamard_eigenstate

P = 3
OMEGA3 = np.exp(2j * np.pi / 3)


def trit_index(x_vec, n):
    """Standard kronecker index for an n-qutrit basis state.

    np.kron uses |x_0> as the most significant trit (0 mod 3 = MSB).
    For x_vec = (x_0, ..., x_{n-1}), idx = x_0 * 3^{n-1} + x_1 * 3^{n-2} + ...
    """
    idx = 0
    for trit in x_vec:
        idx = idx * P + int(trit)
    return idx


def build_stabilizer_state(n, k, x_0, W, Q_coeffs):
    """Build (1/sqrt(3^k)) sum_y omega^{Q(y)} |x_0 + W y>.

    Q_coeffs: list of (degree, var_index, coefficient) tuples for the
    polynomial Q(y). For our case Q is purely quadratic in single
    variables: e.g. [(2, 0, 1), (2, 1, 2), (2, 2, 2)] means
    1*y_0^2 + 2*y_1^2 + 2*y_2^2.
    """
    state = np.zeros(P ** n, dtype=complex)
    for y in itertools.product(range(P), repeat=k):
        y = np.asarray(y, dtype=int)
        # Q evaluation
        q_val = 0
        for (deg, var, coef) in Q_coeffs:
            q_val += coef * (int(y[var]) ** deg)
        q_val %= P
        # Indexed basis vector
        x = (np.asarray(x_0, dtype=int) + W @ y) % P
        idx = trit_index(x, n)
        state[idx] += OMEGA3 ** q_val
    return state / np.sqrt(P ** k)


def main():
    n = 3
    target = qutrit_hadamard_eigenstate(n).astype(np.complex128)
    target = target / np.linalg.norm(target)
    print(f"||H_3^otimes {n}|| = {np.linalg.norm(target):.6f}")

    # Co-author's chi=4 decomposition (from the markdown spec)
    W123 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)
    W3 = np.array([[0], [0], [1]], dtype=int)
    W4 = np.array([[0, 1], [1, 0], [0, 0]], dtype=int)
    x0 = np.zeros(3, dtype=int)

    # Basis 1: alpha = 0.2438 + 0.2438i, k=3, Q = y_0^2 + 2y_1^2 + 2y_2^2 mod 3
    s1 = build_stabilizer_state(n, 3, x0, W123,
                                [(2, 0, 1), (2, 1, 2), (2, 2, 2)])
    a1 = 0.2438 + 0.2438j

    # Basis 2: alpha = 0.2438 - 0.2438i, k=3, Q = 2y_0^2 + y_1^2 + y_2^2 mod 3
    s2 = build_stabilizer_state(n, 3, x0, W123,
                                [(2, 0, 2), (2, 1, 1), (2, 2, 1)])
    a2 = 0.2438 - 0.2438j

    # Basis 3: alpha = 0.6661, k=1, Q = 0
    s3 = build_stabilizer_state(n, 1, x0, W3, [])
    a3 = 0.6661 + 0.0j

    # Basis 4: alpha = 0.6661, k=2, Q = 0
    s4 = build_stabilizer_state(n, 2, x0, W4, [])
    a4 = 0.6661 + 0.0j

    print(f"||S_1|| = {np.linalg.norm(s1):.6f}")
    print(f"||S_2|| = {np.linalg.norm(s2):.6f}")
    print(f"||S_3|| = {np.linalg.norm(s3):.6f}")
    print(f"||S_4|| = {np.linalg.norm(s4):.6f}")

    recon = a1 * s1 + a2 * s2 + a3 * s3 + a4 * s4
    err_direct = float(np.linalg.norm(target - recon))
    print(f"\nDirect reconstruction error (with rounded coeffs): {err_direct:.3e}")

    # Now refit coefficients via least-squares for the exact answer
    basis = np.column_stack([s1, s2, s3, s4])
    coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
    refit = basis @ coeffs
    err_refit = float(np.linalg.norm(target - refit))
    print(f"LS-refit reconstruction error:                     {err_refit:.3e}")
    print(f"Refit coeffs: {coeffs}")

    if err_refit < 1e-10:
        print("\n+ Verified: chi(H_3^otimes 3) <= 4 with this 4-state basis.")
    else:
        print("\n!!! reconstruction failed; basis may be wrong or markdown spec mis-parsed.")


if __name__ == "__main__":
    main()
