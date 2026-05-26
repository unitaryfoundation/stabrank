"""Direct MOSEK Fusion implementation of the Cauchy--Schwarz ratio SDP.

The SDP is

    min sum_{ij} |X_{ij}|
    s.t.  X PSD (Hermitian, N x N)
          tr(X) = 1
          tr(M X) = 0      with  M = A^* Pi_perp A
          tr(N X) >= eta   with  N = A^* psi psi^* A

Encoding. A complex Hermitian PSD X = R + i K (R symmetric, K
antisymmetric) is equivalent to the 2N x 2N real symmetric PSD

    Z = [[R, -K], [K, R]]    (Z = Z^T, Z >> 0).

Block constraints are added at the matrix level (not entrywise) so the
model size scales as O(N^2) bookkeeping but only a few O(1) constraints.
The L1 objective uses N quadratic cones for the off-diagonals (batched)
plus 2N linear cones for the diagonals.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np


def cs_ratio_bound_mosek(
    psi: np.ndarray,
    stab_states: np.ndarray,
    psi_overlap_threshold: float = 0.0,
    verbose: bool = False,
) -> dict:
    import mosek.fusion as mf

    psi_normed = psi / np.linalg.norm(psi)
    N = stab_states.shape[0]
    dim = stab_states.shape[1]
    A = stab_states.T  # (dim, N), columns = stabilizer states

    Pi_perp = np.eye(dim) - np.outer(psi_normed, psi_normed.conj())
    M = (Pi_perp @ A).conj().T @ (Pi_perp @ A)
    N_vec = A.conj().T @ psi_normed
    N_mat = np.outer(N_vec, N_vec.conj())

    # Real encoding of complex Hermitian X = R + i K.
    # Z = [[R, -K], [K, R]] is 2N x 2N real symmetric PSD.
    M_R = M.real
    M_I = M.imag
    N_R = N_mat.real
    N_I = N_mat.imag

    # Coefficient matrices for tr(M X), tr(N X) in Z-vars.
    # tr(M X) = sum_{ij}(M_R[i,j] R[i,j] - M_I[i,j] K[i,j]) for Hermitian M, X.
    # In Z: R_{ij} = Z[i,j], K_{ij} = Z[N+i, j].
    coeffs_M = np.zeros((2 * N, 2 * N))
    coeffs_M[:N, :N] = M_R
    coeffs_M[N:, :N] = -M_I
    coeffs_N = np.zeros((2 * N, 2 * N))
    coeffs_N[:N, :N] = N_R
    coeffs_N[N:, :N] = -N_I

    with mf.Model("cs-ratio-bound") as model:
        if verbose:
            model.setLogHandler(sys.stdout)

        Z = model.variable("Z", mf.Domain.inPSDCone(2 * N))

        # Block structure constraints. Z = [[R, -K], [K, R]], so:
        #   Z[0:N, 0:N] - Z[N:2N, N:2N] = 0     (R block equality)
        #   Z[0:N, N:2N] + Z[N:2N, 0:N] = 0     (-K vs K antisymmetry)
        # MOSEK PSD cone enforces Z = Z^T automatically; the second
        # constraint reduces to Z[0:N, N:2N] = -Z[0:N, N:2N]^T, i.e.,
        # the top-right block is antisymmetric.
        R = Z.slice([0, 0], [N, N])
        Rbr = Z.slice([N, N], [2 * N, 2 * N])
        Ktr = Z.slice([0, N], [N, 2 * N])           # = -K  (top-right)
        Kbl = Z.slice([N, 0], [2 * N, N])           # = K   (bottom-left)
        model.constraint("R_eq_Rbr", mf.Expr.sub(R, Rbr), mf.Domain.equalsTo(0.0))
        model.constraint("K_block", mf.Expr.add(Ktr, Kbl), mf.Domain.equalsTo(0.0))

        # tr(R) = 1: matrix-level via the Z-coefficient matrix.
        # tr(R) = sum_i Z[i,i].
        tr_R_coeff = np.zeros((2 * N, 2 * N))
        for i in range(N):
            tr_R_coeff[i, i] = 1.0
        model.constraint("tr_R", mf.Expr.dot(tr_R_coeff, Z), mf.Domain.equalsTo(1.0))

        # tr(M X) = 0  and  tr(N X) >= eta.
        model.constraint("tr_MX", mf.Expr.dot(coeffs_M, Z), mf.Domain.equalsTo(0.0))
        model.constraint(
            "tr_NX",
            mf.Expr.dot(coeffs_N, Z),
            mf.Domain.greaterThan(psi_overlap_threshold),
        )

        # L1 objective: sum_i |R_{ii}| + 2 sum_{i<j} sqrt(R_{ij}^2 + K_{ij}^2).
        # Diagonal: |R_{ii}| <= s_diag_i  via  s_diag_i +- R_{ii} >= 0.
        s_diag = model.variable("s_diag", N, mf.Domain.greaterThan(0.0))

        # Build a matrix selecting Z[i,i] for i in 0..N-1.
        # We can't easily slice diagonals of Z; instead use a coefficient
        # matrix.
        diag_extract = np.zeros((N, 2 * N, 2 * N))
        for i in range(N):
            diag_extract[i, i, i] = 1.0
        # Compute R_ii = sum_{ab} diag_extract[i, a, b] * Z[a, b].
        # In Fusion: use Expr.mulDiag or batched dot. Simpler: just N
        # scalar constraints (still fast since only N of them, not N^2).
        for i in range(N):
            Z_ii = Z.index(i, i)
            model.constraint(
                mf.Expr.sub(s_diag.index(i), Z_ii),
                mf.Domain.greaterThan(0.0),
            )
            model.constraint(
                mf.Expr.add(s_diag.index(i), Z_ii),
                mf.Domain.greaterThan(0.0),
            )

        # Off-diagonal: pairs (i, j) with i < j. There are N(N-1)/2 of
        # them. We stack into a (n_pairs, 3) array and add one batched
        # quadratic-cone constraint.
        n_pairs = N * (N - 1) // 2
        s_off = model.variable("s_off", n_pairs, mf.Domain.greaterThan(0.0))

        # Build the (i, j) index lists.
        ij_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        i_idx = np.array([p[0] for p in ij_pairs])
        j_idx = np.array([p[1] for p in ij_pairs])

        # We need cones of the form (s_ij, R_ij, K_ij) in inQCone.
        # In Fusion, build by hstacking (s_off, Rij_vec, Kij_vec) as a
        # (n_pairs, 3) matrix and applying inQCone over rows.
        # Rij_vec[k] = Z[i, j], Kij_vec[k] = Z[N+i, j].
        # We need to extract these as expressions; use Expr.pick for
        # efficient indexing.

        # mf.Expr.pick on a 2D variable expects (i, j) index pairs.
        R_idx = [[int(i), int(j)] for i, j in zip(i_idx, j_idx)]
        K_idx = [[int(N + i), int(j)] for i, j in zip(i_idx, j_idx)]

        Rij_expr = mf.Expr.pick(Z, R_idx)
        Kij_expr = mf.Expr.pick(Z, K_idx)
        s_off_expr = s_off  # already a Variable of size n_pairs

        # Stack into a (n_pairs, 3) expression where each row is
        # [s_off_k, R_ij_k, K_ij_k].
        cone_block = mf.Expr.hstack(
            mf.Expr.reshape(s_off_expr, [n_pairs, 1]),
            mf.Expr.reshape(Rij_expr, [n_pairs, 1]),
            mf.Expr.reshape(Kij_expr, [n_pairs, 1]),
        )
        model.constraint("L1_offdiag", cone_block, mf.Domain.inQCone(n_pairs, 3))

        objective = mf.Expr.add(
            mf.Expr.sum(s_diag),
            mf.Expr.mul(2.0, mf.Expr.sum(s_off)),
        )
        model.objective(mf.ObjectiveSense.Minimize, objective)

        t0 = time.time()
        model.solve()
        elapsed = time.time() - t0

        sol_status = model.getProblemStatus().name
        if "Primal" in sol_status or "Optimal" in sol_status:
            value = model.primalObjValue()
            return {
                "cs_ratio_sdp": value,
                "chi_lower_bound": int(math.ceil(value - 1e-6)),
                "status": sol_status,
                "time_s": elapsed,
            }
        return {
            "cs_ratio_sdp": float("inf"),
            "chi_lower_bound": 0,
            "status": sol_status,
            "time_s": elapsed,
        }
