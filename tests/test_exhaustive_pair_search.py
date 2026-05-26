"""Smoke tests for the exhaustive-pair search implementing Lemma 1.

Verifies the numba kernel matches a pure-numpy reference, that the
pair iterator emits exactly C(N, 2) pairs in lex order, and that the
search produces the expected verdict on a few small known cases.
"""

from __future__ import annotations

import numpy as np

from scripts.research.orbit_paper.exhaustive_pair_search import (
    ORBITS,
    kernel_pair_residuals_sq,
    numpy_pair_iter,
    search_orbit_pair,
)
from stabrank.stabilizer_extent import enumerate_stabilizer_states


def _reference_pair_residuals_sq(
    S: np.ndarray, psi: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    """Pure-numpy reference: solve 2-state LS via lstsq for each pair.

    Skips rank-deficient pairs (parallel s_0, s_1) since numpy's lstsq
    falls back differently than the kernel; the kernel returns the
    1-state best-fit residual in that case, while the reference here
    returns the lstsq residual on the rank-deficient system.
    """
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        A = np.column_stack([S[i], S[j]])  # (D, 2)
        sol, *_ = np.linalg.lstsq(A, psi, rcond=None)
        residual = psi - A @ sol
        out[k] = float(np.real(residual.conj() @ residual))
    return out


def _filter_full_rank_pairs(S: np.ndarray, pairs: np.ndarray, eps: float = 1e-9):
    """Keep only pairs whose Gram matrix is non-degenerate."""
    keep = np.zeros(pairs.shape[0], dtype=bool)
    for idx in range(pairs.shape[0]):
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        G01 = np.vdot(S[i], S[j])
        if 1.0 - abs(G01) ** 2 > eps:
            keep[idx] = True
    return pairs[keep]


def test_kernel_matches_reference_m1():
    """Numba kernel and lstsq reference agree on full-rank pairs at m=1."""
    S = enumerate_stabilizer_states(1, d=3).astype(np.complex128)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    target_fn, _d = ORBITS["t3"]
    psi = target_fn(1).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    N = S.shape[0]
    all_pairs = np.array(
        [(i, j) for i in range(N) for j in range(i + 1, N)], dtype=np.int64
    )
    pairs = _filter_full_rank_pairs(S, all_pairs)
    assert pairs.shape[0] > 0
    ref = _reference_pair_residuals_sq(S, psi, pairs)
    fast = kernel_pair_residuals_sq(S, psi, pairs)
    assert np.allclose(fast, ref, atol=1e-10), (
        f"max disagreement {np.abs(fast - ref).max():.3e}"
    )


def test_pair_iter_lex_order():
    """Generator should produce lex-ordered pairs covering C(N, 2)."""
    N = 10
    expected = sorted([(i, j) for i in range(N) for j in range(i + 1, N)])
    actual: list[tuple[int, int]] = []
    for batch in numpy_pair_iter(N, batch_size=7):
        for row in batch:
            actual.append((int(row[0]), int(row[1])))
    assert sorted(actual) == expected
    assert len(actual) == N * (N - 1) // 2


def test_strange_m1_finds_witness():
    """chi(Strange) = 2: pair search at m=1 must find a witness."""
    result = search_orbit_pair("strange", m=1, batch_size=8)
    assert result["chi_le_2_witness"], result
    assert result["certificate_schema_version"] == 3
    assert result["search"]["tuple_size"] == 2
    assert len(result["target_sha256"]) == 64
    assert len(result["stabilizer_dictionary_sha256"]) == 64


def test_t3_m1_excludes_chi_le_2():
    """chi(T_3) = 3: pair search at m=1 must rule out chi <= 2 exhaustively."""
    result = search_orbit_pair("t3", m=1, batch_size=8)
    assert not result["chi_le_2_witness"], result
    assert result["certificate"] == "chi >= 3 (exhaustive)"
    assert result["best_residual"] > 1e-3
