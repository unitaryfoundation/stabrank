import json

import numpy as np
import pytest

import stabrank.ternary_sweep as ternary_sweep
from stabrank.ternary_sweep import (
    breakthrough_chi_threshold,
    gamma_from_chi,
    iter_systematic_codes,
    rank_screen_payload,
    squeeze_code_at_rank,
    squeeze_result_payload,
    systematic_code_from_index,
    total_systematic_codes,
)


def test_breakthrough_chi_thresholds_for_k2_codes():
    assert breakthrough_chi_threshold(6, 2) == 2
    assert breakthrough_chi_threshold(7, 2) == 5
    assert breakthrough_chi_threshold(8, 2) == 8


def test_gamma_from_chi_uses_project_convention():
    assert gamma_from_chi(6, 2, 2) == pytest.approx(np.log(2) / np.log(3) / 2)
    assert gamma_from_chi(7, 2, 5) < 0.5


def test_systematic_code_from_index_decodes_lexicographic_digits():
    code = systematic_code_from_index(6, 2, 1)

    assert code.index == 1
    assert code.a_matrix.shape == (2, 4)
    assert code.a_matrix.tolist() == [[0, 0, 0, 0], [0, 0, 0, 1]]
    assert code.generator.tolist() == [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
    ]


def test_iter_systematic_codes_honors_start_and_limit():
    codes = list(iter_systematic_codes(6, 2, start=5, limit=3))

    assert [code.index for code in codes] == [5, 6, 7]
    assert total_systematic_codes(6, 2) == 3**8


def test_rank_screen_payload_is_json_ready():
    code = systematic_code_from_index(6, 2, 0)

    class Result:
        pass

    result = Result()
    result.code = code
    result.chi = 2
    result.residual = 0.25
    result.exact = False
    result.effective_rank = 18
    result.gamma = gamma_from_chi(6, 2, 2)
    result.seed = 11

    payload = rank_screen_payload(result)

    assert payload["index"] == 0
    assert payload["a_matrix"] == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert payload["effective_rank"] == 18


def test_squeeze_code_repairs_after_non_exact_prune(monkeypatch):
    code = systematic_code_from_index(6, 2, 0)

    monkeypatch.setattr(
        ternary_sweep,
        "qutrit_magic_code_state_compressed",
        lambda generator: np.array([1.0 + 0.0j, 0.0 + 0.0j]),
    )
    monkeypatch.setattr(
        ternary_sweep,
        "generate_random_stabilizer_state",
        lambda n_eff, p, seed: np.array([1.0 + 0.0j, 0.0 + 0.0j]),
    )

    def fake_run_sa_pauli_expansion(**kwargs):
        chi = kwargs["k_subset_size"]
        residual = {3: 0.0, 2: 0.25}[chi]
        basis = [np.array([1.0 + 0.0j, 0.0 + 0.0j]) for _ in range(chi)]
        return chi, basis, np.ones(chi), residual, residual, []

    def fake_prune(target, basis, fn_can_represent, atol=1e-8):
        return basis[:-1], np.int64(2), np.float64(0.5)

    monkeypatch.setattr(
        ternary_sweep,
        "run_sa_pauli_expansion",
        fake_run_sa_pauli_expansion,
    )
    monkeypatch.setattr(
        ternary_sweep,
        "prune_least_significant_basis_function",
        fake_prune,
    )

    result = squeeze_code_at_rank(
        code,
        start_chi=3,
        target_chi=2,
        seed=11,
        early_exit_threshold=1e-9,
    )

    assert result.best_exact_chi == 3
    assert result.reached_target is False
    assert [step.chi for step in result.steps] == [3, 2]
    assert result.steps[0].pruned_chi == 2
    assert result.steps[0].prune_residual == 0.5
    assert result.steps[1].exact is False


def test_squeeze_payload_reports_target_reached_by_exact_prune(monkeypatch):
    code = systematic_code_from_index(6, 2, 0)

    monkeypatch.setattr(
        ternary_sweep,
        "qutrit_magic_code_state_compressed",
        lambda generator: np.array([1.0 + 0.0j, 0.0 + 0.0j]),
    )
    monkeypatch.setattr(
        ternary_sweep,
        "generate_random_stabilizer_state",
        lambda n_eff, p, seed: np.array([1.0 + 0.0j, 0.0 + 0.0j]),
    )

    def fake_run_sa_pauli_expansion(**kwargs):
        chi = kwargs["k_subset_size"]
        basis = [np.array([1.0 + 0.0j, 0.0 + 0.0j]) for _ in range(chi)]
        return chi, basis, np.ones(chi), 0.0, 0.0, []

    def fake_prune(target, basis, fn_can_represent, atol=1e-8):
        return basis[:-1], np.int64(1), np.float64(0.0)

    monkeypatch.setattr(
        ternary_sweep,
        "run_sa_pauli_expansion",
        fake_run_sa_pauli_expansion,
    )
    monkeypatch.setattr(
        ternary_sweep,
        "prune_least_significant_basis_function",
        fake_prune,
    )

    result = squeeze_code_at_rank(
        code,
        start_chi=3,
        target_chi=2,
        seed=11,
        early_exit_threshold=1e-9,
    )
    payload = squeeze_result_payload(result)

    assert payload["best_exact_chi"] == 2
    assert payload["best_effective_rank"] == 18
    assert payload["reached_target"] is True
    assert payload["steps"][0]["prune_exact"] is True
    json.dumps(payload)
