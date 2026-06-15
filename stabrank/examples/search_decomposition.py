"""Simulated-annealing search for stabilizer-rank decompositions of a target state.

This is a single, parameterized driver that replaces the previous family of
near-identical per-state example scripts. Pick a target with ``--target`` and the
script runs the same "anneal, then squeeze/prune down" workflow against it.

Available targets:
    qubit_real      Qubit Hadamard eigenstate (real amplitudes; real-restricted moves)
    qubit_ttype     Qubit T-type magic state |T>^n (complex amplitudes)
    qutrit_real     Qutrit Hadamard eigenstate |H3>
    qutrit_complex  Qutrit complex magic state |T3>
    qutrit_strange  Qutrit strange state |S>
    qutrit_norrell  Qutrit Norrell state |N+>

Usage:
    uv run python stabrank/examples/search_decomposition.py --target qutrit_strange
    uv run python stabrank/examples/search_decomposition.py --target qubit_real --n 6 --k-start 6 --seed 6132679
"""
import argparse
import datetime
import random
import sys
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Ensure the parent directory is in the path to import stabrank when run in-place.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qubit_hadamard_eigenstate,
    qubit_t_type_magic_state,
    qutrit_complex_magic_state,
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_strange_state,
    can_represent_as_linear_combination,
    generate_random_stabilizer_state,
    prune_least_significant_basis_function,
)

from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion


@dataclass(frozen=True)
class TargetSpec:
    """Everything that differs between the per-state searches."""

    label: str                       # used in the title and saved-file names
    make_target: Callable[[int], np.ndarray]
    p_prime: int
    use_real_qubit_moves: bool
    num_chains: int
    cooling_rate: float
    num_iterations_at_temp: int
    default_n: int
    default_k_start: int


TARGETS: dict[str, TargetSpec] = {
    "qubit_real": TargetSpec(
        label="qubit_real",
        make_target=qubit_hadamard_eigenstate,
        p_prime=2,
        use_real_qubit_moves=True,  # Hadamard eigenstate has real amplitudes
        num_chains=8,
        cooling_rate=0.995,
        num_iterations_at_temp=8000,
        default_n=6,
        default_k_start=6,
    ),
    "qubit_ttype": TargetSpec(
        label="qubit_ttype",
        make_target=qubit_t_type_magic_state,
        p_prime=2,
        use_real_qubit_moves=False,  # T-type is complex: do NOT restrict to real moves
        num_chains=8,
        cooling_rate=0.995,
        num_iterations_at_temp=8000,
        default_n=3,
        default_k_start=3,
    ),
    "qutrit_real": TargetSpec(
        label="qutrit_real",
        make_target=qutrit_hadamard_eigenstate,
        p_prime=3,
        use_real_qubit_moves=False,  # never use real-qubit moves for qutrits
        num_chains=16,
        cooling_rate=0.99,
        num_iterations_at_temp=10000,
        default_n=3,
        default_k_start=4,
    ),
    "qutrit_complex": TargetSpec(
        label="qutrit_complex",
        make_target=qutrit_complex_magic_state,
        p_prime=3,
        use_real_qubit_moves=False,
        num_chains=16,
        cooling_rate=0.99,
        num_iterations_at_temp=20000,
        default_n=3,
        default_k_start=5,
    ),
    "qutrit_strange": TargetSpec(
        label="qutrit_strange",
        make_target=qutrit_strange_state,
        p_prime=3,
        use_real_qubit_moves=False,
        num_chains=16,
        cooling_rate=0.99,
        num_iterations_at_temp=10000,
        default_n=3,
        default_k_start=8,
    ),
    "qutrit_norrell": TargetSpec(
        label="qutrit_norrell",
        make_target=qutrit_norrell_state,
        p_prime=3,
        use_real_qubit_moves=False,
        num_chains=16,
        cooling_rate=0.99,
        num_iterations_at_temp=10000,
        default_n=3,
        default_k_start=8,
    ),
}

# Shared SA hyper-parameters (identical across every former example script).
INITIAL_TEMPERATURE = 1.0
MIN_TEMPERATURE = 1 / 4000
ATOL = 1e-7
ERROR_THRESHOLD = 1e-9
TWO_FUNC_PERTURB_PROB = 0.3
RANDOM_REPLACE_PROB = 0.05
CLIFFORD_RATIO = 0.5


def _save(spec, target_func, n, current_k, lin_coeffs, funcs, error, timestamp):
    """Persist a found decomposition to a compressed .npz file."""
    filename = f"solution_{spec.label}_k{current_k}_n{n}_{timestamp}.npz"
    np.savez_compressed(
        filename,
        target_function=target_func,
        linear_coeffs=lin_coeffs,
        n=n,
        p=spec.p_prime,
        k=current_k,
        final_error=error,
        **{f"basis_func_{i}": f for i, f in enumerate(funcs)},
    )
    print(f"Saved to {filename}")


def run_search(target: str, n: int, k_start: int, seed: int) -> None:
    spec = TARGETS[target]

    print(f"\n--- Stabilizer-rank search: {spec.label} (n={n}, k_start={k_start}) ---")
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    target_func = spec.make_target(n)

    print(f"Generating initial random basis of {k_start} functions...")
    initial_basis = [
        generate_random_stabilizer_state(n, p=spec.p_prime) for _ in range(k_start)
    ]

    current_best_funcs = initial_basis
    current_k = k_start

    # --- Squeeze loop: anneal at current_k, then prune as far as the error allows ---
    while current_k > 0:
        print(f"\n=== Optimizing for k={current_k} ===")

        _, best_funcs_tuple, best_lin_coeffs, best_error, _best_cost, _trace = (
            cpp_run_sa_pauli_expansion(
                target=target_func,
                n_orig=n,
                p_prime=spec.p_prime,
                k_subset_size=current_k,
                initial_basis=current_best_funcs,
                initial_temperature=INITIAL_TEMPERATURE,
                cooling_rate=spec.cooling_rate,
                num_iterations_at_temp=spec.num_iterations_at_temp,
                min_temperature=MIN_TEMPERATURE,
                atol=ATOL,
                two_func_perturb_prob=TWO_FUNC_PERTURB_PROB,
                random_replace_prob=RANDOM_REPLACE_PROB,
                use_real_qubit_moves=spec.use_real_qubit_moves,
                clifford_ratio=CLIFFORD_RATIO,
                early_exit_threshold=ERROR_THRESHOLD,
                seed=seed,
                num_chains=spec.num_chains,
            )
        )
        current_best_funcs = list(best_funcs_tuple)
        final_error = best_error

        if final_error >= ERROR_THRESHOLD:
            print(f"🛑 Failed to converge for k={current_k}. Best error: {final_error:.2e}")
            break

        print(f"✅ Found representation for k={current_k} (Error: {final_error:.2e})")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save(spec, target_func, n, current_k, best_lin_coeffs, current_best_funcs, final_error, timestamp)

        # Keep pruning as long as the error stays below threshold.
        while current_k > 0:
            print(f"✂️  Pruning from k={current_k} -> k={current_k - 1}")
            next_funcs, idx, err = prune_least_significant_basis_function(
                target_func, current_best_funcs, can_represent_as_linear_combination
            )
            print(f"Removed index {idx}. New error: {err:.2e}")

            current_k -= 1
            current_best_funcs = next_funcs

            if err >= ERROR_THRESHOLD:
                # Pruned basis no longer satisfies the threshold; re-anneal at current_k.
                break

            print(
                f"✨ Pruned basis still satisfies threshold ({err:.2e} < {ERROR_THRESHOLD:.2e})!"
            )
            print(f"✅ Automatically found representation for k={current_k} via pruning.")
            _, new_lin_coeffs, _ = can_represent_as_linear_combination(
                target_func, current_best_funcs
            )
            _save(spec, target_func, n, current_k, new_lin_coeffs, current_best_funcs, err, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="qutrit_strange",
        help="Which target state to search for.",
    )
    parser.add_argument("--n", type=int, default=None, help="Number of qudits (defaults per target).")
    parser.add_argument(
        "--k-start", type=int, default=None, help="Starting stabilizer rank (defaults per target)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random; printed for reproducibility).",
    )
    args = parser.parse_args()

    spec = TARGETS[args.target]
    n = args.n if args.n is not None else spec.default_n
    k_start = args.k_start if args.k_start is not None else spec.default_k_start
    seed = args.seed if args.seed is not None else np.random.randint(0, 10_000_000)

    run_search(args.target, n, k_start, seed)


if __name__ == "__main__":
    main()
