import numpy as np
import pytest

from stabrank.stabrank_core import (
    apply_random_pauli_string,
    least_squares_solve,
    run_sa_pauli_expansion,
)


def test_least_squares_solve_binding_recovers_exact_coefficients():
    basis_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    basis_1 = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    target = 0.5 * basis_0 + 0.3 * basis_1

    is_representable, coeffs, error = least_squares_solve(
        target, [basis_0, basis_1], 1e-5, 1e-8
    )

    assert is_representable is True
    assert error == pytest.approx(0.0)
    assert coeffs[0] == pytest.approx(0.5 + 0.0j)
    assert coeffs[1] == pytest.approx(0.3 + 0.0j)


def test_apply_random_pauli_string_even_y_constraint_preserved():
    state = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])

    transformed, ops = apply_random_pauli_string(
        state, n=2, p=2, seed=12345, even_y_constraint=True
    )

    assert transformed.shape == state.shape
    assert len(ops) == 2
    assert ops.count("Y") % 2 == 0
    assert np.isfinite(np.linalg.norm(transformed))


@pytest.mark.parametrize("num_chains", [1, 2])
def test_run_sa_binding_keeps_exact_initial_representation(num_chains):
    basis_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    basis_1 = np.array([0.0 + 0.0j, 1.0 + 0.0j])
    target = 0.6 * basis_0 + 0.8 * basis_1

    result = run_sa_pauli_expansion(
        target=target,
        n_orig=1,
        p_prime=2,
        k_subset_size=2,
        initial_basis=[basis_0, basis_1],
        initial_temperature=1.0,
        cooling_rate=0.5,
        num_iterations_at_temp=1,
        min_temperature=0.6,
        rtol=1e-5,
        atol=1e-8,
        two_func_perturb_prob=0.0,
        random_replace_prob=0.0,
        use_real_qubit_moves=False,
        clifford_ratio=0.0,
        early_exit_threshold=1e-12,
        seed=7,
        num_chains=num_chains,
        enable_tracing=(num_chains == 1),
        fixed_dimension=-1,
    )

    k, best_basis, best_coeffs, best_error, best_cost, trace = result

    assert k == 2
    assert len(best_basis) == 2
    assert len(best_coeffs) == 2
    assert best_error < 1e-12
    assert best_cost < 1e-12

    if num_chains == 1:
        assert trace
        assert trace[0]["move_type"] in {0, 1, 2}


def _valid_sa_kwargs(**overrides):
    basis_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    basis_1 = np.array([0.0 + 0.0j, 1.0 + 0.0j])
    kwargs = {
        "target": 0.6 * basis_0 + 0.8 * basis_1,
        "n_orig": 1,
        "p_prime": 2,
        "k_subset_size": 2,
        "initial_basis": [basis_0, basis_1],
        "initial_temperature": 1.0,
        "cooling_rate": 0.5,
        "num_iterations_at_temp": 1,
        "min_temperature": 0.6,
        "rtol": 1e-5,
        "atol": 1e-8,
        "two_func_perturb_prob": 0.0,
        "random_replace_prob": 0.0,
        "use_real_qubit_moves": False,
        "clifford_ratio": 0.0,
        "early_exit_threshold": 1e-12,
        "seed": 7,
        "num_chains": 1,
        "enable_tracing": False,
        "fixed_dimension": -1,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"k_subset_size": 0}, "k_subset_size"),
        ({"initial_basis": [np.array([1.0 + 0.0j, 0.0 + 0.0j])]}, "initial_basis"),
        (
            {
                "initial_basis": [
                    np.array([1.0 + 0.0j, 0.0 + 0.0j]),
                    np.array([1.0 + 0.0j]),
                ]
            },
            "same length as target",
        ),
        ({"fixed_dimension": 2}, "fixed_dimension"),
        ({"cooling_rate": 1.0}, "cooling_rate"),
        ({"target": np.array([1.0 + 0.0j])}, "target length"),
        ({"num_chains": 0}, "num_chains"),
    ],
)
def test_run_sa_binding_rejects_invalid_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_sa_pauli_expansion(**_valid_sa_kwargs(**overrides))


def test_run_sa_binding_fixed_dimension_single_qudit_does_not_hang():
    basis_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    result = run_sa_pauli_expansion(
        **_valid_sa_kwargs(
            target=basis_0,
            k_subset_size=1,
            initial_basis=[basis_0],
            fixed_dimension=1,
            clifford_ratio=1.0,
        )
    )

    assert result[0] == 1
    assert len(result[1]) == 1
