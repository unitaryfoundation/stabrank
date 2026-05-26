"""Stabilizer fidelity bounds and extent-related numerical diagnostics.

Enumerates all stabilizer states on n qudits of odd prime dimension d,
then computes fidelity bounds and an LP surrogate for finding feasible
low-l1 stabilizer decompositions.

The stabilizer fidelity bound gives: chi >= 1 / max_phi |<phi|psi>|^2.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog


def _tuple_to_index(t: tuple[int, ...] | np.ndarray, d: int) -> int:
    idx = 0
    for v in t:
        idx = idx * d + int(v)
    return idx


def enumerate_stabilizer_states(
    n: int,
    d: int = 3,
) -> np.ndarray:
    """Enumerate all stabilizer states on n qudits of dimension d.

    Each stabilizer state is parametrized by:
      - A k-dimensional subspace (k x n matrix W in RREF over F_d)
      - A coset representative x0 in F_d^n
      - Phase polynomial coefficients (linear, square, mixed quadratic)

    Args:
        n: Number of qudits.
        d: Local dimension (odd prime).

    Returns:
        Complex matrix of shape (N_stab, d^n), each row a normalized state.
    """
    dim = d ** n
    states: list[np.ndarray] = []

    for k in range(n + 1):
        # Enumerate k-dim subspaces via RREF matrices
        for pivots in itertools.combinations(range(n), k):
            non_pivots = [j for j in range(n) if j not in pivots]

            # Free entries in RREF: k rows x (n-k) non-pivot columns
            n_free = k * len(non_pivots)
            for free_vals in itertools.product(range(d), repeat=n_free):
                # Build RREF matrix W (k x n)
                W = np.zeros((k, n), dtype=np.int64)
                idx_f = 0
                for row_i in range(k):
                    W[row_i, pivots[row_i]] = 1
                    for col_j in non_pivots:
                        W[row_i, col_j] = free_vals[idx_f]
                        idx_f += 1

                # Enumerate coset representatives (non-pivot coords of x0)
                for x0_free in itertools.product(range(d), repeat=n - k):
                    x0 = np.zeros(n, dtype=np.int64)
                    for i, col in enumerate(non_pivots):
                        x0[col] = x0_free[i]

                    # Enumerate phase polynomial coefficients
                    n_lin = k
                    n_sq = k  # only for d >= 3
                    n_mix = k * (k - 1) // 2
                    n_phase = n_lin + (n_sq if d >= 3 else 0) + n_mix

                    for phase_vals in itertools.product(range(d), repeat=n_phase):
                        c_lin = list(phase_vals[:n_lin])
                        offset = n_lin
                        if d >= 3:
                            c_sq = list(phase_vals[offset:offset + n_sq])
                            offset += n_sq
                        else:
                            c_sq = [0] * k
                        c_mix = list(phase_vals[offset:offset + n_mix])

                        # Build state vector by evaluating over F_d^k
                        state = np.zeros(dim, dtype=complex)
                        for y_tuple in itertools.product(range(d), repeat=k):
                            y = np.array(y_tuple, dtype=np.int64)

                            # x = x0 + W^T y mod d
                            x = (x0 + W.T @ y) % d

                            # Phase polynomial q(y) / d
                            q = 0.0
                            for i in range(k):
                                q += c_lin[i] * y[i] / d
                                if d >= 3:
                                    q += c_sq[i] * y[i] * y[i] / d
                            mix_idx = 0
                            for s in range(k):
                                for t in range(s + 1, k):
                                    q += c_mix[mix_idx] * y[s] * y[t] / d
                                    mix_idx += 1

                            phase = np.exp(2j * np.pi * q)
                            x_idx = _tuple_to_index(tuple(x), d)
                            state[x_idx] = phase

                        # Normalize
                        norm = np.linalg.norm(state)
                        if norm > 1e-12:
                            state /= norm
                        states.append(state)

    return np.array(states)


def stabilizer_fidelity_bound(
    psi: np.ndarray,
    stab_states: np.ndarray,
) -> dict[str, float]:
    """Compute the stabilizer-fidelity-based lower bound on the extent xi.

    Returns ceil(1/F_max) where F_max = max |<phi|psi>|^2. By
    [BBCCGH2019 Sec. 6.1 Prop. 2], xi(psi) >= 1/F_max(psi). This is
    NOT a rank lower bound: the chain chi >= xi fails for non-orthogonal
    stabilizer decompositions (paper §3.1).

    Args:
        psi: Normalized target state vector.
        stab_states: Matrix of stabilizer states (rows).

    Returns:
        Dict with max_fidelity (F_max) and extent_lb (ceil(1/F_max)).
    """
    psi_normed = psi / np.linalg.norm(psi)
    overlaps = np.abs(stab_states @ psi_normed.conj()) ** 2
    f_max = float(np.max(overlaps))
    extent_lb = int(math.ceil(1.0 / f_max)) if f_max > 1e-15 else 0

    return {
        "max_fidelity": f_max,
        "extent_lb": extent_lb,
    }


def stabilizer_extent_lp(
    psi: np.ndarray,
    stab_states: np.ndarray,
) -> dict[str, float]:
    """Compute a linear-programming surrogate for stabilizer extent.

    This is not the exact stabilizer extent. Exact complex l1
    minimization requires the objective sum_i sqrt(Re(c_i)^2 + Im(c_i)^2),
    which is a conic optimization problem rather than a linear program.

    This helper instead minimizes the split norm
    sum_i |Re(c_i)| + |Im(c_i)| subject to sum c_i |phi_i> = |psi>.
    The resulting coefficients form a valid explicit decomposition, so
    their true complex l1 norm gives an upper bound on the stabilizer
    extent, not a lower bound and not necessarily the optimum.

    The LP is formulated over real variables by splitting c_i = a_i+ - a_i- + j(b_i+ - b_i-).

    Args:
        psi: Normalized target state vector.
        stab_states: Matrix of stabilizer states (rows), shape (N, dim).

    Returns:
        Dict with extent_upper_bound, l1_norm_upper_bound,
        split_l1_objective, and n_nonzero. Note n_nonzero is NOT a rank
        lower bound: it is the support cardinality of an explicit
        decomposition, which is an UPPER bound on chi for that
        decomposition.
    """
    psi_normed = psi / np.linalg.norm(psi)
    N, dim = stab_states.shape

    # Split into real/imag parts.
    # Variables: [a+_1..a+_N, a-_1..a-_N, b+_1..b+_N, b-_1..b-_N], all >= 0
    # c_i = (a+_i - a-_i) + j*(b+_i - b-_i)
    # Constraint: Re(A)*(a+ - a-) - Im(A)*(b+ - b-) = Re(psi)
    #             Im(A)*(a+ - a-) + Re(A)*(b+ - b-) = Im(psi)
    # Minimize: sum(a+) + sum(a-) + sum(b+) + sum(b-)

    A_re = stab_states.real.T  # (dim, N)
    A_im = stab_states.imag.T  # (dim, N)

    # Constraint matrix: 2*dim rows, 4*N columns
    # [A_re, -A_re, -A_im, A_im] * [a+, a-, b+, b-]^T = [Re(psi)]
    # [A_im, -A_im,  A_re, -A_re] * [a+, a-, b+, b-]^T = [Im(psi)]
    A_eq = np.block([
        [A_re, -A_re, -A_im, A_im],
        [A_im, -A_im, A_re, -A_re],
    ])
    b_eq = np.concatenate([psi_normed.real, psi_normed.imag])

    # Objective: minimize sum of all 4N variables (= split real/imag l1 norm).
    # Note: |c_i| <= |a+_i| + |a-_i| + |b+_i| + |b-_i| = a+_i + a-_i + b+_i + b-_i
    # This finds a feasible decomposition using a surrogate objective; it is
    # not the exact complex-l1 extent minimization.
    c_obj = np.ones(4 * N)

    result = linprog(
        c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method='highs',
        options={'presolve': True},
    )

    if not result.success:
        return {
            "extent_upper_bound": float('inf'),
            "l1_norm_upper_bound": float('inf'),
            "split_l1_objective": float('inf'),
            "n_nonzero": 0,
            "status": result.message,
        }

    x = result.x
    a_plus = x[:N]
    a_minus = x[N:2*N]
    b_plus = x[2*N:3*N]
    b_minus = x[3*N:]

    c_real = a_plus - a_minus
    c_imag = b_plus - b_minus
    c_complex = c_real + 1j * c_imag
    c_abs = np.abs(c_complex)

    l1 = float(np.sum(c_abs))
    split_l1 = float(result.fun)
    n_nonzero = int(np.sum(c_abs > 1e-8))
    extent_upper_bound = l1 ** 2

    return {
        "extent_upper_bound": extent_upper_bound,
        "l1_norm_upper_bound": l1,
        "split_l1_objective": split_l1,
        "n_nonzero": n_nonzero,
        "status": "optimal_surrogate",
    }


def cs_ratio_bound_sdp(
    psi: np.ndarray,
    stab_states: np.ndarray,
    solver: str | None = None,
    psi_overlap_threshold: float = 0.0,
) -> dict[str, float]:
    """Cauchy-Schwarz rank lower bound via SDP relaxation (cvxpy).

    See stabrank.cs_ratio_sdp_mosek.cs_ratio_bound_mosek for the more
    efficient direct-Fusion implementation. This cvxpy version is the
    reference implementation used to verify m=1 results.

    Lifts X = c c^* (Hermitian PSD, rank-1 dropped) of the QCQP
        min ||c||_1^2 / ||c||_2^2  s.t.  A c = psi.
    Constraints:
        tr(X) = 1                 ( ||c||_2^2 = 1 )
        tr(M X) = 0               ( Pi_perp A c = 0,  M = A^* Pi_perp A )
        tr(N X) >= eta            ( |<psi|Ac>|^2 >= eta;  N = A^* psi psi^* A )
    Objective: sum_{ij} |X_{ij}|.

    The SDP optimum is a valid lower bound on chi only when
    psi_overlap_threshold <= |<psi|Ac*>|^2 / ||c*||_2^2 at the QCQP
    optimum. Setting eta too high excludes the optimum and over-estimates
    rho_min (INVALID LB on chi); setting eta too small lets X concentrate
    near null(A), giving a near-trivial LB.
    """
    import cvxpy as cp

    psi_normed = psi / np.linalg.norm(psi)
    N = stab_states.shape[0]
    dim = stab_states.shape[1]
    A = stab_states.T

    Pi_perp = np.eye(dim) - np.outer(psi_normed, psi_normed.conj())
    M = (Pi_perp @ A).conj().T @ (Pi_perp @ A)
    N_vec = A.conj().T @ psi_normed
    N_mat = np.outer(N_vec, N_vec.conj())

    X = cp.Variable((N, N), hermitian=True)
    constraints = [
        X >> 0,
        cp.real(cp.trace(X)) == 1,
        cp.real(cp.trace(M @ X)) == 0,
        cp.real(cp.trace(N_mat @ X)) >= psi_overlap_threshold,
    ]
    objective = cp.Minimize(cp.sum(cp.abs(X)))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver or cp.SCS, verbose=False)
    except cp.error.SolverError as e:
        return {"cs_ratio_sdp": float('inf'), "chi_lower_bound": 0,
                "status": f"solver_error: {e}"}

    if prob.value is None or not np.isfinite(prob.value):
        return {"cs_ratio_sdp": float('inf'), "chi_lower_bound": 0,
                "status": prob.status}

    bound = float(prob.value)
    chi_lb = int(math.ceil(bound - 1e-6))
    return {
        "cs_ratio_sdp": bound,
        "chi_lower_bound": chi_lb,
        "status": prob.status,
        "X_value": X.value,
    }
