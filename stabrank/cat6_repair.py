"""Rank-8 repair searches seeded from cat6 A2/B4 tensor decompositions."""

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cat6_tensor import cat6_a2b4_basis, cat6_target, solve_decomposition
from .stabrank_core import run_sa_pauli_expansion


@dataclass(frozen=True)
class RepairSeed:
    """One leave-one-out rank-8 seed from a 9-state A2/B4 decomposition."""

    positions_a: tuple[int, int]
    omitted_index: int
    basis: list[np.ndarray]
    residual: float


@dataclass(frozen=True)
class RepairAttempt:
    """Result of one rank-8 repair attempt."""

    positions_a: tuple[int, int]
    omitted_index: int
    baseline_residual: float
    repaired_residual: float
    exact: bool


@dataclass(frozen=True)
class RepairSearchResult:
    """Summary of a rank-8 repair search."""

    attempts: list[RepairAttempt]
    best_attempt: RepairAttempt
    exact: bool


def iter_a2b4_leave_one_out_seeds(
    positions: "Sequence[tuple[int, int]] | None" = None,
) -> list[RepairSeed]:
    """Enumerate rank-8 seeds from every A2/B4 construction and omission."""

    if positions is None:
        positions = list(itertools.combinations(range(6), 2))

    target = cat6_target()
    seeds = []
    for positions_a in positions:
        basis = cat6_a2b4_basis(positions_a)
        for omitted_index in range(len(basis)):
            seed_basis = basis[:omitted_index] + basis[omitted_index + 1 :]
            residual = solve_decomposition(target, seed_basis).residual
            seeds.append(
                RepairSeed(
                    positions_a=tuple(positions_a),
                    omitted_index=omitted_index,
                    basis=seed_basis,
                    residual=residual,
                )
            )
    return seeds


def run_repair_attempt(
    seed: RepairSeed,
    *,
    initial_temperature: float,
    cooling_rate: float,
    num_iterations_at_temp: int,
    min_temperature: float,
    two_func_perturb_prob: float,
    random_replace_prob: float,
    clifford_ratio: float,
    early_exit_threshold: float,
    rng_seed: int,
    num_chains: int,
    fixed_dimension: int,
) -> RepairAttempt:
    """Run one warm-started SA repair attempt from a leave-one-out basis."""

    target = cat6_target()
    result = run_sa_pauli_expansion(
        target=target,
        n_orig=6,
        p_prime=3,
        k_subset_size=8,
        initial_basis=seed.basis,
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
        seed=rng_seed,
        num_chains=num_chains,
        enable_tracing=False,
        fixed_dimension=fixed_dimension,
    )
    repaired_residual = float(result[3])
    return RepairAttempt(
        positions_a=seed.positions_a,
        omitted_index=seed.omitted_index,
        baseline_residual=seed.residual,
        repaired_residual=repaired_residual,
        exact=repaired_residual <= early_exit_threshold,
    )


def run_repair_search(
    *,
    max_seeds: "int | None" = None,
    initial_temperature: float = 0.2,
    cooling_rate: float = 0.85,
    num_iterations_at_temp: int = 5,
    min_temperature: float = 0.1,
    two_func_perturb_prob: float = 0.3,
    random_replace_prob: float = 0.02,
    clifford_ratio: float = 0.7,
    early_exit_threshold: float = 1e-9,
    seed: int = 20260501,
    num_chains: int = 1,
    fixed_dimension: int = -1,
) -> RepairSearchResult:
    """Run warm-started rank-8 repair over leave-one-out A2/B4 seeds."""

    seeds = iter_a2b4_leave_one_out_seeds()
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    attempts = []
    for index, repair_seed in enumerate(seeds):
        attempt = run_repair_attempt(
            repair_seed,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            num_iterations_at_temp=num_iterations_at_temp,
            min_temperature=min_temperature,
            two_func_perturb_prob=two_func_perturb_prob,
            random_replace_prob=random_replace_prob,
            clifford_ratio=clifford_ratio,
            early_exit_threshold=early_exit_threshold,
            rng_seed=seed + index,
            num_chains=num_chains,
            fixed_dimension=fixed_dimension,
        )
        attempts.append(attempt)
        if attempt.exact:
            break

    if not attempts:
        raise ValueError("repair search requires at least one seed")

    best_attempt = min(attempts, key=lambda item: item.repaired_residual)
    return RepairSearchResult(
        attempts=attempts,
        best_attempt=best_attempt,
        exact=best_attempt.exact,
    )
