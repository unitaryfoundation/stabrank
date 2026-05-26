import numpy as np
import pytest

from stabrank.cat6_repair import iter_a2b4_leave_one_out_seeds, run_repair_search


def test_a2b4_leave_one_out_seed_enumeration():
    seeds = iter_a2b4_leave_one_out_seeds()

    assert len(seeds) == 15 * 9
    assert all(len(seed.basis) == 8 for seed in seeds)
    assert min(seed.residual for seed in seeds) == pytest.approx(1.0 / 3.0)
    assert max(seed.residual for seed in seeds) == pytest.approx(1.0 / 3.0)


def test_rank8_repair_search_smoke():
    result = run_repair_search(
        max_seeds=1,
        initial_temperature=0.2,
        cooling_rate=0.5,
        num_iterations_at_temp=1,
        min_temperature=0.6,
        seed=7,
        num_chains=1,
    )

    assert len(result.attempts) == 1
    assert result.best_attempt.baseline_residual == pytest.approx(1.0 / 3.0)
    assert np.isfinite(result.best_attempt.repaired_residual)
