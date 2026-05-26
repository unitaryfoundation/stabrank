"""Simulated annealing search for stabilizer decompositions of the qubit T-type magic state.

The T-type state |T> = cos(beta)|0> + e^{i pi/4} sin(beta)|1> with
cos(2 beta) = 1/sqrt(3) has complex amplitudes, so the search uses
generic (non-real-restricted) Pauli moves.

Usage:
    uv run python stabrank/examples/qubit_t_type_example.py
"""
import datetime
import random
import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qubit_t_type_magic_state,
    can_represent_as_linear_combination,
    generate_random_poly_coeffs,
    evaluate_coeffs_on_subspace,
    prune_least_significant_basis_function,
)

from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion


def run_qubit_t_type_example() -> None:
    """Search for a stabilizer decomposition of |T>^n using simulated annealing."""
    print("\n--- Qubit T-type Magic State Decomposition Example ---")

    # --- SA Parameters ---
    seed = np.random.randint(0, 10_000_000)
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    initial_temperature = 1.0
    cooling_rate = 0.995
    num_iterations_at_temp = 8000
    min_temperature = 1 / 4000
    atol = 1e-7
    ERROR_THRESHOLD = 1e-9

    # --- Problem Setup (Qubits p=2) ---
    n_orig = 3  # Number of qubits
    p_prime = 2

    # Target: complex-valued T-type state
    target_func = qubit_t_type_magic_state(n_orig)

    # Initial stabilizer rank guess
    k_start = 3

    # Full-space parameters for random stabilizer state generation
    k_dim_poly = n_orig
    W_basis = np.eye(n_orig, dtype=int)
    x0_trans = np.zeros(n_orig, dtype=int)

    print(f"Generating initial random basis of {k_start} functions...")
    initial_basis: list[np.ndarray] = []

    def generate_random_func() -> np.ndarray:
        coeffs = generate_random_poly_coeffs(k_dim_poly, p_prime, [0.0])
        func = evaluate_coeffs_on_subspace(coeffs, n_orig, p_prime, x0_trans, W_basis)
        return func

    for _ in range(k_start):
        func = generate_random_func()
        initial_basis.append(func / np.linalg.norm(func))

    current_best_funcs = initial_basis
    current_k = k_start

    # --- Squeeze Loop ---
    while current_k > 0:
        print(f"\n=== Optimizing for k={current_k} ===")
        found_solution = False
        final_lin_coeffs = None
        final_error = float("inf")

        runs = 1
        for i in range(runs):
            print(f"--- SA Run {i + 1}/{runs} ---")

            _, best_funcs_tuple, best_lin_coeffs, best_error, best_cost, _trace = (
                cpp_run_sa_pauli_expansion(
                    target=target_func,
                    n_orig=n_orig,
                    p_prime=p_prime,
                    k_subset_size=current_k,
                    initial_basis=current_best_funcs,
                    initial_temperature=initial_temperature,
                    cooling_rate=cooling_rate,
                    num_iterations_at_temp=num_iterations_at_temp,
                    min_temperature=min_temperature,
                    atol=atol,
                    two_func_perturb_prob=0.3,
                    random_replace_prob=0.05,
                    # T-type has complex amplitudes: do NOT restrict to real moves
                    use_real_qubit_moves=False,
                    clifford_ratio=0.5,
                    early_exit_threshold=ERROR_THRESHOLD,
                    seed=seed + i,
                    num_chains=8,
                )
            )
            best_funcs = list(best_funcs_tuple)

            current_best_funcs = best_funcs
            final_error = best_error
            final_lin_coeffs = best_lin_coeffs

            if final_error < ERROR_THRESHOLD:
                found_solution = True
                break

        if found_solution:
            print(f"Found representation for k={current_k} (Error: {final_error:.2e})")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_qubit_ttype_k{current_k}_n{n_orig}_{timestamp}.npz"
            np.savez_compressed(
                filename,
                target_function=target_func,
                linear_coeffs=final_lin_coeffs,
                n=n_orig,
                p=p_prime,
                k=current_k,
                final_error=final_error,
                **{f"basis_func_{i}": f for i, f in enumerate(current_best_funcs)},
            )
            print(f"Saved to {filename}")

            # Prune loop
            while current_k > 0:
                print(f"Pruning from k={current_k} -> k={current_k - 1}")
                next_funcs, idx, err = prune_least_significant_basis_function(
                    target_func, current_best_funcs, can_represent_as_linear_combination
                )
                print(f"Removed index {idx}. New error: {err:.2e}")

                current_k -= 1
                current_best_funcs = next_funcs

                if err < ERROR_THRESHOLD:
                    print(
                        f"Pruned basis still satisfies threshold "
                        f"({err:.2e} < {ERROR_THRESHOLD:.2e})!"
                    )
                    print(
                        f"Automatically found representation for k={current_k} "
                        f"via pruning."
                    )

                    is_rep, new_lin_coeffs, _ = can_represent_as_linear_combination(
                        target_func, current_best_funcs
                    )

                    filename = (
                        f"solution_qubit_ttype_k{current_k}_n{n_orig}_{timestamp}.npz"
                    )
                    np.savez_compressed(
                        filename,
                        target_function=target_func,
                        linear_coeffs=new_lin_coeffs,
                        n=n_orig,
                        p=p_prime,
                        k=current_k,
                        final_error=err,
                        **{
                            f"basis_func_{i}": f
                            for i, f in enumerate(current_best_funcs)
                        },
                    )
                    print(f"Saved to {filename}")
                else:
                    break
        else:
            print(
                f"Failed to converge for k={current_k}. Best error: {final_error:.2e}"
            )
            break


if __name__ == "__main__":
    run_qubit_t_type_example()
