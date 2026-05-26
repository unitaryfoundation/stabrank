import pytest

from stabrank.stabilizer_extent import (
    enumerate_stabilizer_states,
    stabilizer_extent_lp,
)


def test_stabilizer_extent_lp_reports_surrogate_upper_bound_keys():
    stab_states = enumerate_stabilizer_states(1, d=3)
    result = stabilizer_extent_lp(stab_states[0], stab_states)

    assert result["status"] == "optimal_surrogate"
    assert result["extent_upper_bound"] == pytest.approx(1.0)
    assert result["l1_norm_upper_bound"] == pytest.approx(1.0)
    assert result["split_l1_objective"] == pytest.approx(1.0)
    assert "extent" not in result
    assert "l1_norm" not in result
