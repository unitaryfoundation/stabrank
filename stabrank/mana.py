"""Discrete Wigner function and mana for odd-prime-dimensional systems.

The mana M(psi) = log_d(||W||_1) quantifies non-stabilizerness and is
additive under tensor products.  For odd prime d the discrete Hudson
theorem guarantees that stabilizer states have non-negative Wigner
functions, so mana is zero iff the state is stabilizer.

Reference: Veitch, Ferrie, Gross, Emerson, New J. Phys. 14, 113011 (2012).
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def _index_to_tuple(idx: int, n: int, d: int) -> tuple[int, ...]:
    """Convert a lexicographic index to an n-digit base-d tuple."""
    digits: list[int] = []
    for _ in range(n):
        digits.append(idx % d)
        idx //= d
    return tuple(reversed(digits))


def _tuple_to_index(t: tuple[int, ...] | np.ndarray, d: int) -> int:
    """Convert an n-digit base-d tuple to a lexicographic index."""
    idx = 0
    for v in t:
        idx = idx * d + int(v)
    return idx


def discrete_wigner_function(
    psi: np.ndarray,
    n: int,
    d: int = 3,
) -> np.ndarray:
    """Compute the discrete Wigner function of a pure state.

    Uses the standard Gross definition for odd-prime d:

        W(q, p) = (1/d^n) sum_{y in F_d^n} omega^{-2 p.y}  psi(q+y) psi*(q-y)

    where arithmetic on the state indices is mod d and omega = exp(2pi i / d).

    Args:
        psi: State vector of length d^n (need not be normalized).
        n:   Number of qudits.
        d:   Local dimension (must be an odd prime, default 3).

    Returns:
        Real array of shape (d^n, d^n) indexed as W[q_idx, p_idx].
    """
    if d % 2 == 0:
        raise ValueError("Discrete Wigner function requires odd prime d.")

    dim = d ** n
    if psi.shape[0] != dim:
        raise ValueError(f"State vector length {psi.shape[0]} != d^n = {dim}.")

    omega = np.exp(2j * np.pi / d)

    # Pre-compute all base-d tuples for speed.
    tuples = np.array(
        list(itertools.product(range(d), repeat=n)), dtype=np.int64
    )  # shape (dim, n)

    W = np.empty((dim, dim), dtype=np.float64)

    for q_idx in range(dim):
        q = tuples[q_idx]
        for p_idx in range(dim):
            p = tuples[p_idx]
            acc = 0.0 + 0.0j
            for y_idx in range(dim):
                y = tuples[y_idx]
                qpy = tuple((q + y) % d)
                qmy = tuple((q - y + d) % d)  # +d to keep positive before mod
                idx_plus = _tuple_to_index(qpy, d)
                idx_minus = _tuple_to_index(qmy, d)
                dot_py = int(np.dot(p, y)) % d
                phase = omega ** (-2 * dot_py)
                acc += phase * psi[idx_plus] * np.conj(psi[idx_minus])
            W[q_idx, p_idx] = (acc / dim).real  # W is guaranteed real

    return W


def compute_mana(
    psi: np.ndarray,
    n: int,
    d: int = 3,
) -> dict[str, float]:
    """Compute the mana and related diagnostics for a pure state.

    The bound sqrt(||W_psi||_1) is a lower bound on the stabilizer extent
    xi(psi) (Pashayan-Wallman-Bartlett 2015 / equivalent). The naive chain
    chi(psi) >= xi(psi) fails for non-orthogonal stabilizer decompositions
    (cf. paper §3.1, with the m=2 Strange counter-example), so this is NOT
    a rigorous lower bound on the stabilizer rank chi. We expose it as
    `extent_lb`, an extent diagnostic only.

    Args:
        psi: State vector of length d^n.
        n:   Number of qudits.
        d:   Local dimension (odd prime, default 3).

    Returns:
        Dictionary with keys:
            wigner_l1:  sum of |W(u)| over phase space.
            wigner_neg: sum of |W(u)| where W(u) < 0 (total negativity).
            mana:       log_d(wigner_l1).
            max_wigner: maximum value of W(u).
            min_wigner: minimum value of W(u).
            extent_lb:  sqrt(wigner_l1), a lower bound on the stabilizer
                        extent xi (NOT a rank lower bound).
    """
    # Normalize the state so that Tr(rho) = 1, i.e. ||psi||=1.
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        raise ValueError("State vector has zero norm.")
    psi_normed = psi / norm

    W = discrete_wigner_function(psi_normed, n, d)

    l1 = float(np.sum(np.abs(W)))
    neg = float(np.sum(np.abs(W[W < 0])))
    mana_val = math.log(l1) / math.log(d) if l1 > 0 else 0.0
    extent_lb = math.sqrt(l1)

    return {
        "wigner_l1": l1,
        "wigner_neg": neg,
        "mana": mana_val,
        "max_wigner": float(np.max(W)),
        "min_wigner": float(np.min(W)),
        "extent_lb": extent_lb,
    }

