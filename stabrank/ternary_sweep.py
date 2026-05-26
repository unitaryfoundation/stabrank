"""Utilities for systematic ternary code sweeps."""

import math
from dataclasses import dataclass

import numpy as np

from .analysis_utils import prune_least_significant_basis_function
from .linalg_utils import can_represent_as_linear_combination
from .polynomial_utils import generate_random_stabilizer_state
from .stabrank_core import run_sa_pauli_expansion
from .target_functions import qutrit_magic_code_state_compressed


QUTRIT_DIMENSION = 3


@dataclass(frozen=True)
class SystematicCode:
    """One systematic ternary code generator ``G = [I_k | A]``."""

    index: int
    m: int
    k_code: int
    a_matrix: np.ndarray
    generator: np.ndarray


@dataclass(frozen=True)
class RankScreenResult:
    """Result of one fixed-rank SA screen for a code state."""

    code: SystematicCode
    chi: int
    residual: float
    exact: bool
    effective_rank: int
    gamma: float
    seed: int


@dataclass(frozen=True)
class SqueezeStep:
    """One rank attempt in a warm-start squeeze/prune run."""

    chi: int
    seed: int
    residual: float
    exact: bool
    pruned_chi: int
    prune_residual: float
    prune_exact: bool
    removed_index: int


@dataclass(frozen=True)
class SqueezeResult:
    """Result of attempting to squeeze a code state from high chi downward."""

    code: SystematicCode
    start_chi: int
    target_chi: int
    best_exact_chi: int
    best_exact_residual: float
    final_chi: int
    final_residual: float
    reached_target: bool
    seed: int
    steps: list[SqueezeStep]


def total_systematic_codes(m: int, k_code: int, d: int = QUTRIT_DIMENSION) -> int:
    """Return the number of systematic ``[m,k]`` codes over ``F_d``."""

    _validate_code_shape(m, k_code)
    return d ** (k_code * (m - k_code))


def breakthrough_chi_threshold(
    m: int,
    k_code: int,
    d: int = QUTRIT_DIMENSION,
) -> int:
    """Return largest integer chi that would give ``gamma < 1/2``."""

    _validate_code_shape(m, k_code)
    exponent = (m - 2 * k_code) / 2
    if exponent <= 0:
        return 0
    return int(math.ceil(d**exponent) - 1)


def gamma_from_chi(m: int, k_code: int, chi: int, d: int = QUTRIT_DIMENSION) -> float:
    """Return the code-state scaling exponent used in the project notes."""

    _validate_code_shape(m, k_code)
    if chi <= 0:
        raise ValueError("chi must be positive")
    denominator = m - 2 * k_code
    if denominator <= 0:
        return float("inf")
    return math.log(chi, d) / denominator


def systematic_code_from_index(
    m: int,
    k_code: int,
    index: int,
    d: int = QUTRIT_DIMENSION,
) -> SystematicCode:
    """Decode ``index`` into the systematic generator ``[I_k | A]``."""

    _validate_code_shape(m, k_code)
    total = total_systematic_codes(m, k_code, d)
    if index < 0 or index >= total:
        raise ValueError(f"index must be in [0, {total})")

    digit_count = k_code * (m - k_code)
    digits = _index_to_digits(index, digit_count, d)
    a_matrix = np.array(digits, dtype=int).reshape((k_code, m - k_code))
    generator = np.hstack([np.eye(k_code, dtype=int), a_matrix])
    return SystematicCode(
        index=index,
        m=m,
        k_code=k_code,
        a_matrix=a_matrix,
        generator=generator,
    )


def iter_systematic_codes(
    m: int,
    k_code: int,
    *,
    start: int = 0,
    limit=None,
    d: int = QUTRIT_DIMENSION,
):
    """Yield systematic ternary codes in deterministic lexicographic order."""

    total = total_systematic_codes(m, k_code, d)
    if start < 0 or start > total:
        raise ValueError(f"start must be in [0, {total}]")

    stop = total if limit is None else min(total, start + limit)
    for index in range(start, stop):
        yield systematic_code_from_index(m, k_code, index, d)


def rank_screen_payload(result: RankScreenResult) -> dict:
    """Return a JSON-serializable payload for a rank screen result."""

    return {
        "index": result.code.index,
        "m": result.code.m,
        "k_code": result.code.k_code,
        "a_matrix": result.code.a_matrix.tolist(),
        "generator": result.code.generator.tolist(),
        "chi": result.chi,
        "residual": result.residual,
        "exact": result.exact,
        "effective_rank": result.effective_rank,
        "gamma": result.gamma,
        "seed": result.seed,
    }


def squeeze_result_payload(result: SqueezeResult) -> dict:
    """Return a JSON-serializable payload for a squeeze/prune run."""

    best_exact_chi = None if result.best_exact_chi == 0 else result.best_exact_chi
    return {
        "index": result.code.index,
        "m": result.code.m,
        "k_code": result.code.k_code,
        "a_matrix": result.code.a_matrix.tolist(),
        "generator": result.code.generator.tolist(),
        "start_chi": result.start_chi,
        "target_chi": result.target_chi,
        "best_exact_chi": best_exact_chi,
        "best_exact_residual": _json_float(result.best_exact_residual),
        "best_effective_rank": _effective_rank_or_none(
            result.code.k_code,
            result.best_exact_chi,
        ),
        "best_gamma": _gamma_or_none(result.code, result.best_exact_chi),
        "final_chi": result.final_chi,
        "final_residual": _json_float(result.final_residual),
        "reached_target": result.reached_target,
        "seed": result.seed,
        "steps": [_squeeze_step_payload(step) for step in result.steps],
    }


def screen_code_at_rank(
    code: SystematicCode,
    *,
    chi: int,
    seed: int,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.99,
    num_iterations_at_temp: int = 200,
    min_temperature: float = 0.01,
    two_func_perturb_prob: float = 0.3,
    random_replace_prob: float = 0.05,
    clifford_ratio: float = 0.5,
    early_exit_threshold: float = 1e-9,
    num_chains: int = 1,
    fixed_dimension: int = -1,
) -> RankScreenResult:
    """Run one fixed-rank SA screen for a systematic ternary code."""

    if chi <= 0:
        raise ValueError("chi must be positive")

    n_eff = code.m - code.k_code
    target = qutrit_magic_code_state_compressed(code.generator)
    target = target / np.linalg.norm(target)
    initial_basis = [
        generate_random_stabilizer_state(n_eff, p=QUTRIT_DIMENSION, seed=seed + idx)
        for idx in range(chi)
    ]
    result = run_sa_pauli_expansion(
        target=target,
        n_orig=n_eff,
        p_prime=QUTRIT_DIMENSION,
        k_subset_size=chi,
        initial_basis=initial_basis,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        num_iterations_at_temp=num_iterations_at_temp,
        min_temperature=min_temperature,
        rtol=1e-5,
        atol=1e-8,
        two_func_perturb_prob=two_func_perturb_prob,
        random_replace_prob=random_replace_prob,
        use_real_qubit_moves=False,
        clifford_ratio=clifford_ratio,
        early_exit_threshold=early_exit_threshold,
        seed=seed,
        num_chains=num_chains,
        enable_tracing=False,
        fixed_dimension=fixed_dimension,
    )
    residual = float(result[3])
    return RankScreenResult(
        code=code,
        chi=chi,
        residual=residual,
        exact=residual <= early_exit_threshold,
        effective_rank=(QUTRIT_DIMENSION**code.k_code) * chi,
        gamma=gamma_from_chi(code.m, code.k_code, chi),
        seed=seed,
    )


def squeeze_code_at_rank(
    code: SystematicCode,
    *,
    start_chi: int,
    target_chi: int,
    seed: int,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.99,
    num_iterations_at_temp: int = 2000,
    min_temperature: float = 0.001,
    two_func_perturb_prob: float = 0.3,
    random_replace_prob: float = 0.05,
    clifford_ratio: float = 0.5,
    early_exit_threshold: float = 1e-9,
    num_chains: int = 16,
    fixed_dimension: int = -1,
) -> SqueezeResult:
    """Warm-start at ``start_chi`` and repair/prune down toward ``target_chi``."""

    _validate_squeeze_bounds(start_chi, target_chi)

    n_eff = code.m - code.k_code
    target = qutrit_magic_code_state_compressed(code.generator)
    target = target / np.linalg.norm(target)
    basis = [
        generate_random_stabilizer_state(n_eff, p=QUTRIT_DIMENSION, seed=seed + idx)
        for idx in range(start_chi)
    ]
    steps = []
    best_exact_chi = 0
    best_exact_residual = float("inf")
    final_chi = start_chi
    final_residual = float("inf")

    for chi in range(start_chi, target_chi - 1, -1):
        if len(basis) > chi:
            basis = basis[:chi]

        run_seed = seed + chi
        result = run_sa_pauli_expansion(
            target=target,
            n_orig=n_eff,
            p_prime=QUTRIT_DIMENSION,
            k_subset_size=chi,
            initial_basis=basis,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            num_iterations_at_temp=num_iterations_at_temp,
            min_temperature=min_temperature,
            rtol=1e-5,
            atol=1e-8,
            two_func_perturb_prob=two_func_perturb_prob,
            random_replace_prob=random_replace_prob,
            use_real_qubit_moves=False,
            clifford_ratio=clifford_ratio,
            early_exit_threshold=early_exit_threshold,
            seed=run_seed,
            num_chains=num_chains,
            enable_tracing=False,
            fixed_dimension=fixed_dimension,
        )
        basis = list(result[1])
        residual = float(result[3])
        exact = residual <= early_exit_threshold
        final_chi = chi
        final_residual = residual

        if exact and _is_better_exact(chi, residual, best_exact_chi, best_exact_residual):
            best_exact_chi = chi
            best_exact_residual = residual

        pruned_chi = 0
        prune_residual = float("nan")
        prune_exact = False
        removed_index = -1
        if exact and chi > target_chi:
            basis, removed_index, prune_residual = prune_least_significant_basis_function(
                target,
                basis,
                can_represent_as_linear_combination,
                atol=1e-8,
            )
            pruned_chi = chi - 1
            prune_exact = prune_residual <= early_exit_threshold
            if prune_exact and _is_better_exact(
                pruned_chi,
                prune_residual,
                best_exact_chi,
                best_exact_residual,
            ):
                best_exact_chi = pruned_chi
                best_exact_residual = prune_residual

        steps.append(
            SqueezeStep(
                chi=chi,
                seed=run_seed,
                residual=residual,
                exact=bool(exact),
                pruned_chi=pruned_chi,
                prune_residual=float(prune_residual),
                prune_exact=bool(prune_exact),
                removed_index=int(removed_index),
            )
        )

        if not exact or best_exact_chi == target_chi:
            break

    return SqueezeResult(
        code=code,
        start_chi=start_chi,
        target_chi=target_chi,
        best_exact_chi=best_exact_chi,
        best_exact_residual=best_exact_residual,
        final_chi=final_chi,
        final_residual=final_residual,
        reached_target=bool(best_exact_chi == target_chi),
        seed=seed,
        steps=steps,
    )


def _index_to_digits(index: int, digit_count: int, base: int) -> list[int]:
    digits = [0] * digit_count
    remaining = index
    for pos in range(digit_count - 1, -1, -1):
        digits[pos] = remaining % base
        remaining //= base
    return digits


def _validate_code_shape(m: int, k_code: int) -> None:
    if m <= 0:
        raise ValueError("m must be positive")
    if k_code <= 0 or k_code >= m:
        raise ValueError("k_code must satisfy 0 < k_code < m")


def _validate_squeeze_bounds(start_chi: int, target_chi: int) -> None:
    if start_chi <= 0:
        raise ValueError("start_chi must be positive")
    if target_chi <= 0:
        raise ValueError("target_chi must be positive")
    if target_chi > start_chi:
        raise ValueError("target_chi must be less than or equal to start_chi")


def _is_better_exact(
    chi: int,
    residual: float,
    best_chi: int,
    best_residual: float,
) -> bool:
    return best_chi == 0 or chi < best_chi or (chi == best_chi and residual < best_residual)


def _squeeze_step_payload(step: SqueezeStep) -> dict:
    return {
        "chi": step.chi,
        "seed": step.seed,
        "residual": step.residual,
        "exact": bool(step.exact),
        "pruned_chi": None if step.pruned_chi == 0 else step.pruned_chi,
        "prune_residual": _json_float(step.prune_residual),
        "prune_exact": bool(step.prune_exact),
        "removed_index": None if step.removed_index < 0 else step.removed_index,
    }


def _json_float(value: float):
    if math.isinf(value) or math.isnan(value):
        return None
    return value


def _effective_rank_or_none(k_code: int, chi: int):
    if chi == 0:
        return None
    return (QUTRIT_DIMENSION**k_code) * chi


def _gamma_or_none(code: SystematicCode, chi: int):
    if chi == 0:
        return None
    return gamma_from_chi(code.m, code.k_code, chi)
