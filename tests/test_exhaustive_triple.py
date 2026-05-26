"""Smoke tests for the exhaustive-triple search.

These tests use small dictionaries (m=1, 2) where the search runs in
milliseconds. The actual interesting cases (m=3) are not tested here
since they take hours-to-days to run; see the script's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.orbit_paper.exhaustive_triple_search import (
    ORBITS,
    batch_triple_residuals_sq,
    search_orbit,
)
from stabrank.stabilizer_extent import enumerate_stabilizer_states


def test_batch_residuals_zero_when_psi_in_span():
    # Build a single-qutrit state in span(|0>, |1>, |2>) = C^3, so any
    # triple of three linearly-independent stabilizer states should give
    # residual ~0 against any target on 1 qutrit.
    S = enumerate_stabilizer_states(1, d=3).astype(np.complex128)
    psi = np.array([0.5, 0.5j, np.sqrt(0.5)], dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    # Just take all triples and verify min residual is ~0.
    N = S.shape[0]
    triples = np.array(
        [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)],
        dtype=np.int64,
    )
    res_sq = batch_triple_residuals_sq(S, psi, triples)
    assert res_sq.min() < 1e-20


def test_strange_m1_finds_witness():
    # chi(Strange) = 2, so Strange in span of two stabilizers; trivially
    # in span of any triple containing those two. Triple search should
    # find a witness immediately.
    result = search_orbit("strange", m=1, batch_size=64)
    assert result["chi_le_3_witness"], (
        f"expected witness for Strange m=1, got residual {result['best_residual']}"
    )
    assert result["best_triple"] is not None


def test_t3_m1_finds_witness():
    # chi(T_3) = 3 (paper Table 3); since dim = 3, T_3 lies in span of
    # the computational-basis stabilizer triple {|0>, |1>, |2>}.
    result = search_orbit("t3", m=1, batch_size=64)
    assert result["chi_le_3_witness"], (
        f"expected witness for T_3 m=1, got residual {result['best_residual']}"
    )
    # Verify the witness coefficients reconstruct T_3.
    S = enumerate_stabilizer_states(1, d=3).astype(np.complex128)
    psi_tgt = ORBITS["t3"](1).astype(np.complex128)
    psi_tgt /= np.linalg.norm(psi_tgt)
    triple = result["best_triple"]
    S_tri = S[triple]
    c, *_ = np.linalg.lstsq(S_tri.T, psi_tgt, rcond=None)
    psi_recon = S_tri.T @ c
    assert np.linalg.norm(psi_recon - psi_tgt) < 1e-9


def test_h3_m1_finds_witness():
    # chi(H_3) = 2, similar to Strange.
    result = search_orbit("h3", m=1, batch_size=64)
    assert result["chi_le_3_witness"]


def test_norrell_m1_finds_witness():
    result = search_orbit("norrell", m=1, batch_size=64)
    assert result["chi_le_3_witness"]


def test_invalid_orbit_raises():
    with pytest.raises(ValueError):
        search_orbit("bogus", m=1)
