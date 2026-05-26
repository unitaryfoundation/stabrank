import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from stabrank.cat6_tensor import (
    all_a2b4_candidates,
    cat6_a2b4_basis,
    cat6_a2b4_decomposition,
    compressed_shifted_cat4_state,
    shifted_cat4_basis,
    solve_decomposition,
)

# `verify_stabilizer.py` lives in scripts/, not on the package path; add it.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from verify_stabilizer import is_stabilizer_state  # noqa: E402


def test_shifted_cat4_factors_have_rank3_decompositions():
    for residue in range(3):
        result = solve_decomposition(
            compressed_shifted_cat4_state(residue),
            shifted_cat4_basis(residue),
        )

        assert result.basis_count == 3
        assert result.residual < 1e-12


def test_a2b4_construction_reconstructs_cat6_for_all_bipartitions():
    for positions_a in itertools.combinations(range(6), 2):
        result = cat6_a2b4_decomposition(positions_a)

        assert result.basis_count == 9
        assert result.residual < 1e-12


def test_a2b4_basis_states_are_stabilizer_states():
    basis = cat6_a2b4_basis((0, 1))

    assert len(basis) == 9
    for state in basis:
        assert np.linalg.norm(state) == pytest.approx(1.0)
        is_stabilizer, _params = is_stabilizer_state(state, p=3)
        assert is_stabilizer


def test_all_a2b4_candidates_deduplicate_global_phases():
    candidates = all_a2b4_candidates()

    assert len(candidates) <= 15 * 9
    assert len(candidates) >= 9
