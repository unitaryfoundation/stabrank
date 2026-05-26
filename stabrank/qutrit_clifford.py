"""Qutrit Clifford group utilities for gadget construction.

Provides:
- Single-qutrit Clifford generators (H, S, X) and the full group (216 elements)
- Two-qutrit Clifford generators and a streaming BFS iterator
- Gadget acceptance checking (unitarity, non-Clifford, correctable byproducts)
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OMEGA: complex = np.exp(2j * np.pi / 3)

# ---------------------------------------------------------------------------
# Single-qutrit generators (3x3 unitary matrices)
# ---------------------------------------------------------------------------

#: Qutrit Fourier / Hadamard gate
H3: np.ndarray = (1.0 / np.sqrt(3)) * np.array(
    [
        [1, 1, 1],
        [1, _OMEGA, _OMEGA**2],
        [1, _OMEGA**2, _OMEGA],
    ],
    dtype=complex,
)

#: Phase gate diag(1, 1, omega)
S3: np.ndarray = np.diag([1.0, 1.0, _OMEGA]).astype(complex)

#: Cyclic shift |a> -> |a+1 mod 3>
X3: np.ndarray = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)

#: Clock gate diag(1, omega, omega^2)
Z3: np.ndarray = np.diag([1.0, _OMEGA, _OMEGA**2]).astype(complex)

# ---------------------------------------------------------------------------
# Two-qutrit SUM gate  |a,b> -> |a, a+b mod 3>
# ---------------------------------------------------------------------------

def _build_sum_gate() -> np.ndarray:
    SUM = np.zeros((9, 9), dtype=complex)
    for a in range(3):
        for b in range(3):
            row = a * 3 + (a + b) % 3
            col = a * 3 + b
            SUM[row, col] = 1.0
    return SUM


SUM_GATE: np.ndarray = _build_sum_gate()

# ---------------------------------------------------------------------------
# Phase-canonical form for deduplication
# ---------------------------------------------------------------------------

def _canonicalize(mat: np.ndarray) -> np.ndarray:
    """Return a copy of *mat* whose first nonzero entry is real and positive.

    Two unitaries that differ only by a global phase will have the same
    canonical form (within floating-point tolerance).
    """
    flat = mat.ravel()
    for v in flat:
        if np.abs(v) > 1e-12:
            phase = v / np.abs(v)
            return mat / phase
    return mat.copy()


def _reorthogonalize(mat: np.ndarray) -> np.ndarray:
    """Project a nearly-unitary matrix back onto U(n) via polar decomposition.

    Repeated floating-point multiplications cause drift away from unitarity.
    This snaps the matrix back to the nearest unitary.
    """
    U, _, Vt = np.linalg.svd(mat)
    return U @ Vt


def _is_seen(
    mat: np.ndarray,
    group: list[np.ndarray],
    tol: float = 1e-8,
) -> bool:
    """Check if *mat* (phase-canonicalized) is already in *group*."""
    canon = _canonicalize(mat)
    for existing in group:
        if np.allclose(canon, existing, atol=tol):
            return True
    return False

# ---------------------------------------------------------------------------
# Single-qutrit Clifford group enumeration
# ---------------------------------------------------------------------------

def enumerate_single_qutrit_cliffords() -> list[np.ndarray]:
    """Enumerate all 216 single-qutrit Clifford unitaries via BFS.

    Uses direct allclose comparison for deduplication (exact for a group
    of this size). Each element is re-orthogonalized to prevent
    floating-point drift and stored in phase-canonical form.

    Returns a list of 3x3 unitary matrices, one per equivalence class
    modulo global phase.
    """
    generators = [H3, S3, X3]
    group: list[np.ndarray] = []
    queue: deque[np.ndarray] = deque()

    identity = np.eye(3, dtype=complex)
    canon_id = _canonicalize(identity)
    group.append(canon_id)
    queue.append(identity)

    while queue:
        current = queue.popleft()
        for g in generators:
            product = _reorthogonalize(g @ current)
            if not _is_seen(product, group):
                canon = _canonicalize(product)
                group.append(canon)
                queue.append(product)

    return group

# ---------------------------------------------------------------------------
# Clifford checking
# ---------------------------------------------------------------------------

def is_proportional_to_unitary(E: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a 3x3 matrix E is proportional to a unitary."""
    prod = E.conj().T @ E
    diag_val = prod[0, 0]
    if np.abs(diag_val) < tol:
        return False
    return bool(np.allclose(prod / diag_val, np.eye(3), atol=tol))


def normalize_to_unitary(E: np.ndarray) -> np.ndarray:
    """Given E proportional to a unitary, return E / ||E|| (Frobenius-scaled)."""
    norm = np.sqrt(np.abs(np.trace(E.conj().T @ E)) / 3.0)
    return E / norm


def build_clifford_lookup(clifford_group: list[np.ndarray]) -> np.ndarray:
    """Pre-stack canonicalized Cliffords into a (N, 3, 3) array for fast lookup."""
    return np.array([_canonicalize(C) for C in clifford_group])


def is_single_qutrit_clifford(
    U: np.ndarray,
    clifford_group: list[np.ndarray] | np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Check if the 3x3 unitary U is proportional to a known Clifford.

    Accepts either a list of 3x3 matrices or a pre-stacked (N, 3, 3) array
    (from build_clifford_lookup) for best performance.
    """
    canon_U = _canonicalize(U)

    if isinstance(clifford_group, np.ndarray) and clifford_group.ndim == 3:
        # Vectorized: compute max absolute difference against all Cliffords at once
        diffs = np.max(np.abs(clifford_group - canon_U), axis=(1, 2))
        return bool(np.any(diffs < tol))

    # Fallback: sequential comparison
    for C in clifford_group:
        canon_C = _canonicalize(C)
        if np.allclose(canon_U, canon_C, atol=tol):
            return True
    return False

# ---------------------------------------------------------------------------
# Post-measurement operators for injection gadgets
# ---------------------------------------------------------------------------

def compute_post_measurement_operators(
    C_ent: np.ndarray,
    magic_state: np.ndarray,
) -> list[np.ndarray]:
    """Compute the three post-measurement operators E_k for k=0,1,2.

    Given a 9x9 two-qutrit Clifford C_ent and a 3-component magic state
    vector M, the operator E_k acting on the data qutrit is:

        E_k[i, j] = sum_a  C_ent[3*i + k, 3*j + a] * M[a]

    Args:
        C_ent: 9x9 unitary matrix (two-qutrit Clifford).
        magic_state: length-3 state vector.

    Returns:
        List of three 3x3 complex matrices [E_0, E_1, E_2].
    """
    M = magic_state
    operators: list[np.ndarray] = []
    for k in range(3):
        E = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                for a in range(3):
                    E[i, j] += C_ent[3 * i + k, 3 * j + a] * M[a]
        operators.append(E)
    return operators

# ---------------------------------------------------------------------------
# Gadget acceptance check
# ---------------------------------------------------------------------------

def _check_gadget_branch(
    ops: list[np.ndarray],
    success_branch: int,
    clifford_group: list[np.ndarray],
    tol: float,
) -> dict | None:
    """Check whether (ops, success_branch) yields a valid gadget.

    `success_branch` is the post-selection outcome k* whose post-measurement
    operator E_{k*} should be proportional to a non-Clifford unitary U; the
    other two operators E_k (k != k*) must each be proportional to a unitary
    that differs from U by a Clifford byproduct.
    """
    E_star = ops[success_branch]
    if not is_proportional_to_unitary(E_star, tol=tol):
        return None

    U = normalize_to_unitary(E_star)

    if is_single_qutrit_clifford(U, clifford_group, tol=tol):
        return None

    byproducts: list[np.ndarray] = []
    for k in range(3):
        if k == success_branch:
            continue
        Ek = ops[k]
        if not is_proportional_to_unitary(Ek, tol=tol):
            return None
        Uk = normalize_to_unitary(Ek)
        Pk = Uk @ U.conj().T
        if not is_single_qutrit_clifford(Pk, clifford_group, tol=tol):
            return None
        byproducts.append(Pk)

    success_prob = float(np.abs(np.trace(E_star.conj().T @ E_star)) / 3.0)
    return {
        "U": U,
        "byproducts": byproducts,
        "success_prob": success_prob,
        "success_branch": success_branch,
    }


def check_gadget(
    C_ent: np.ndarray,
    magic_state: np.ndarray,
    clifford_group: list[np.ndarray],
    tol: float = 1e-6,
) -> dict | None:
    """Check whether (C_ent, magic_state) forms a valid injection gadget.

    Tries each of the three post-selection outcomes k* in {0, 1, 2} as the
    "success" branch (the one whose post-measurement operator is the
    non-Clifford unitary being injected); the other two branches must be
    correctable Clifford byproducts. Per Lemma "gadget reduction to
    Sp(4, F_3)" of the paper, an exhaustive search over Sp(4, F_3) checked
    at all three branches covers the entire two-qutrit Clifford group.

    Returns a dict with keys 'U', 'byproducts', 'success_prob',
    'success_branch' on success (the smallest k* for which a gadget is
    found), or None on failure.
    """
    ops = compute_post_measurement_operators(C_ent, magic_state)
    for k_star in range(3):
        result = _check_gadget_branch(ops, k_star, clifford_group, tol)
        if result is not None:
            return result
    return None

# ---------------------------------------------------------------------------
# Two-qutrit Clifford group: symplectic representation over F_3
# ---------------------------------------------------------------------------

# Symplectic form J for Sp(4, F_3): J = [[0, I_2], [-I_2, 0]]
_J4: np.ndarray = np.array(
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [2, 0, 0, 0],  # -1 mod 3 = 2
     [0, 2, 0, 0]],
    dtype=np.int64,
)


def _symplectic_generators_4() -> list[np.ndarray]:
    """Return generators for Sp(4, F_3) as 4x4 integer matrices mod 3.

    The symplectic vector is ordered as (x1, x2, z1, z2) where xi, zi are
    the X and Z exponents for qutrit i.

    Generators correspond to the action of H1, H2, S1, S2, SUM on the
    symplectic vector:
      H on qutrit i: (x_i, z_i) -> (z_i, -x_i) = (z_i, 2*x_i mod 3)
      S on qutrit i: (x_i, z_i) -> (x_i, x_i + z_i)
      SUM (ctrl=1, tgt=2): (x1, x2, z1, z2) -> (x1, x1+x2, z1-z2, z2)
                          = (x1, x1+x2, z1+2*z2, z2) mod 3
    """
    # H on qutrit 1: H X1 H^dag = Z1, H Z1 H^dag = X1^{-1}, so
    # (x1, x2, z1, z2) -> (-z1, x2, x1, z2) = (2*z1, x2, x1, z2)
    H1 = np.array([
        [0, 0, 2, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.int64)

    # H on qutrit 2: H X2 H^dag = Z2, H Z2 H^dag = X2^{-1}, so
    # (x1, x2, z1, z2) -> (x1, -z2, z1, x2) = (x1, 2*z2, z1, x2)
    H2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ], dtype=np.int64)

    # S on qutrit 1: (x1, x2, z1, z2) -> (x1, x2, x1+z1, z2)
    S1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.int64)

    # S on qutrit 2: (x1, x2, z1, z2) -> (x1, x2, z1, x2+z2)
    S2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.int64)

    # SUM (ctrl=1, tgt=2): (x1, x2, z1, z2) -> (x1, x1+x2, z1+2*z2, z2)
    SUM_symp = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ], dtype=np.int64)

    return [H1, H2, S1, S2, SUM_symp]


def _is_symplectic(F: np.ndarray) -> bool:
    """Check F^T J F = J (mod 3)."""
    prod = (F.T @ _J4 @ F) % 3
    return np.array_equal(prod, _J4 % 3)


def enumerate_sp4_f3(report_every: int = 10_000) -> list[np.ndarray]:
    """Enumerate all elements of Sp(4, F_3) via BFS over integer matrices.

    This is exact (no floating-point), and the group has exactly 51,840 elements.

    Returns:
        List of 4x4 integer matrices (mod 3).
    """
    generators = _symplectic_generators_4()
    seen: set[bytes] = set()
    group: list[np.ndarray] = []
    queue: deque[np.ndarray] = deque()

    identity = np.eye(4, dtype=np.int64)
    key = identity.tobytes()
    seen.add(key)
    group.append(identity)
    queue.append(identity)

    while queue:
        current = queue.popleft()
        for g in generators:
            product = (g @ current) % 3
            key = product.tobytes()
            if key not in seen:
                seen.add(key)
                group.append(product)
                queue.append(product)

        if report_every and len(group) % report_every == 0:
            print(f"  [Sp4] enumerated {len(group)} elements, queue {len(queue)}")

    return group


def enumerate_sp4_f3_with_unitaries(
    report_every: int = 10_000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Enumerate Sp(4, F_3) and build 9x9 unitaries simultaneously.

    During BFS, we track both the symplectic matrix (for exact dedup) and
    the corresponding 9x9 unitary (by applying the same generator to both).
    This avoids the O(N^2) cost of per-element decomposition.

    Returns:
        List of (F_symplectic, U_unitary) pairs.
    """
    generators_symp = _symplectic_generators_4()
    I3 = np.eye(3, dtype=complex)
    generators_unitary = [
        np.kron(H3, I3),   # H1
        np.kron(I3, H3),   # H2
        np.kron(S3, I3),   # S1
        np.kron(I3, S3),   # S2
        SUM_GATE,           # SUM
    ]

    seen: set[bytes] = set()
    results: list[tuple[np.ndarray, np.ndarray]] = []
    # Queue stores (symplectic_matrix, unitary_matrix)
    queue: deque[tuple[np.ndarray, np.ndarray]] = deque()

    F_id = np.eye(4, dtype=np.int64)
    U_id = np.eye(9, dtype=complex)

    seen.add(F_id.tobytes())
    results.append((F_id, U_id))
    queue.append((F_id, U_id))

    while queue:
        F_curr, U_curr = queue.popleft()
        for g_symp, g_unit in zip(generators_symp, generators_unitary):
            F_new = (g_symp @ F_curr) % 3
            key = F_new.tobytes()
            if key not in seen:
                seen.add(key)
                U_new = g_unit @ U_curr
                results.append((F_new, U_new))
                queue.append((F_new, U_new))

        if report_every and len(results) % report_every == 0:
            print(f"  [Sp4] enumerated {len(results)} elements, queue {len(queue)}")

    return results


def iter_two_qutrit_cliffords(
    report_every: int = 10_000,
) -> Iterator[np.ndarray]:
    """Yield 9x9 unitary matrices for the two-qutrit Clifford group.

    Enumerates Sp(4, F_3) exactly (51,840 elements) and builds the
    corresponding 9x9 unitaries in a single BFS pass. Displacements are
    not enumerated because they don't affect which non-Clifford gate U is
    injected (they only change the Clifford byproduct).

    Args:
        report_every: Print progress every N elements.

    Yields:
        9x9 unitary matrices.
    """
    print("  [Sp4] Enumerating Sp(4, F_3) with unitaries...")
    pairs = enumerate_sp4_f3_with_unitaries(report_every=report_every)
    print(f"  [Sp4] Found {len(pairs)} elements")

    for i, (_, U) in enumerate(pairs):
        yield U

    print(f"  [Sp4] complete: yielded {len(pairs)} unitaries")


