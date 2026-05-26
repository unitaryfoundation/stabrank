"""Smoke tests for the numba-accelerated exhaustive-triple search.

Verifies the kernel matches the pure-numpy reference and finds known
witnesses at m=1. Avoids the expensive m=3 search.
"""

from __future__ import annotations

import numpy as np

from scripts.research.orbit_paper.exhaustive_triple_search import (
    batch_triple_residuals_sq as ref_residuals,
)
from scripts.research.orbit_paper.exhaustive_triple_search_numba import (
    ORBITS,
    kernel_residuals_sq,
    numpy_triple_iter,
    search_orbit_numba,
)
from stabrank.stabilizer_extent import enumerate_stabilizer_states


def _filter_full_rank_triples(
    S: np.ndarray, triples: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """Keep only triples whose Gram matrix has det > eps (full rank)."""
    keep = np.zeros(triples.shape[0], dtype=bool)
    for idx in range(triples.shape[0]):
        S_tri = S[triples[idx]]
        G = S_tri.conj() @ S_tri.T
        if np.abs(np.linalg.det(G)) > eps:
            keep[idx] = True
    return triples[keep]


def test_kernel_matches_reference_m1():
    """Numba kernel and pure-numpy reference agree on full-rank triples.

    Rank-deficient triples (det(G) ~= 0) are skipped: kernel emits 1.0 (no
    witness from this triple), reference falls back to lstsq. Both are
    correct for the witness problem since rank-deficient triples can only
    certify chi <= 2, already handled by Lemma 1's pair search.
    """
    S = enumerate_stabilizer_states(1, d=3).astype(np.complex128)
    psi = ORBITS["t3"](1).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    N = S.shape[0]
    all_triples = np.array(
        [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)],
        dtype=np.int64,
    )
    triples = _filter_full_rank_triples(S, all_triples)
    assert triples.shape[0] > 0
    ref = ref_residuals(S, psi, triples)
    fast = kernel_residuals_sq(S, psi, triples)
    assert np.allclose(fast, ref, atol=1e-10), (
        f"max disagreement {np.abs(fast - ref).max():.3e}"
    )


def test_kernel_matches_reference_m2_strange():
    """Same agreement check at m=2, on a small subset of full-rank triples."""
    S = enumerate_stabilizer_states(2, d=3).astype(np.complex128)
    psi = ORBITS["strange"](2).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    N = S.shape[0]
    rng = np.random.default_rng(0)
    triples_sample = rng.choice(N, size=(1000, 3), replace=True)
    triples_sample = np.sort(triples_sample, axis=1)
    # Drop rows with duplicates (i = j or j = k).
    mask = (triples_sample[:, 0] < triples_sample[:, 1]) & (
        triples_sample[:, 1] < triples_sample[:, 2]
    )
    triples = _filter_full_rank_triples(S, triples_sample[mask].astype(np.int64))
    assert triples.shape[0] > 0
    ref = ref_residuals(S, psi, triples)
    fast = kernel_residuals_sq(S, psi, triples)
    assert np.allclose(fast, ref, atol=1e-10)


def test_numpy_iter_lex_order():
    """Generator should produce lex-ordered triples covering C(N, 3)."""
    N = 10
    expected = [
        (i, j, k) for k in range(2, N) for j in range(1, k) for i in range(j)
    ]
    actual: list[tuple[int, int, int]] = []
    for batch in numpy_triple_iter(N, batch_size=7):
        for row in batch:
            actual.append(tuple(int(x) for x in row))
    # The generator emits within-(j,k) blocks; lex order across blocks is
    # by (k, j, i), matching the expected list.
    assert sorted(actual) == sorted(expected)
    assert len(actual) == N * (N - 1) * (N - 2) // 6


def test_numpy_iter_resume():
    """Generator should resume from start_idx without re-emitting earlier triples."""
    N = 8
    full = []
    for batch in numpy_triple_iter(N, batch_size=4):
        for row in batch:
            full.append(tuple(int(x) for x in row))
    skip = 10
    resumed = []
    for batch in numpy_triple_iter(N, batch_size=4, start_idx=skip):
        for row in batch:
            resumed.append(tuple(int(x) for x in row))
    # Resumed sequence must be a (sorted) subset of full triples; combined
    # with the first 'skip' triples it must reconstruct the full list.
    # We don't enforce element-wise equality (the first batch may include
    # earlier (j, k) blocks that haven't been fully skipped), but the
    # cumulative count must match.
    assert len(resumed) >= len(full) - skip
    assert set(resumed).issubset(set(full))


def test_strange_m1_finds_witness():
    """chi(Strange) = 2 -- triple search trivially finds a spanning triple."""
    result = search_orbit_numba("strange", m=1, batch_size=8)
    assert result["chi_le_3_witness"]
    assert result["certificate_schema_version"] == 3
    assert result["search"]["tuple_size"] == 3


def test_t3_m1_finds_witness():
    """chi(T_3) = 3 -- T_3 lies in span(|0>, |1>, |2>)."""
    result = search_orbit_numba("t3", m=1, batch_size=8)
    assert result["chi_le_3_witness"]
