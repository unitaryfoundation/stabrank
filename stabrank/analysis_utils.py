"""Analysis utilities for stabilizer rank computations."""

import numpy as np

def calculate_sa_cost(error_metric, num_functions_in_subset, lambda_k_penalty=0.0):
    """Cost function for simulated annealing."""
    if error_metric is None or np.isinf(error_metric) or np.isnan(error_metric):
        return float('inf')
    return error_metric

def prune_least_significant_basis_function(
    target_function: np.ndarray,
    basis_funcs: list[np.ndarray],
    fn_can_represent_as_linear_combination: callable,
    atol: float = 1e-8
) -> tuple[list[np.ndarray], int, float]:
    """
    Identifies and removes the basis function whose removal causes the minimal increase in error.
    
    Returns:
        tuple: (new_basis_funcs, removed_index, new_error)
            - new_basis_funcs: The list of basis functions with the best one removed.
            - removed_index: The index of the function that was removed (original index).
            - new_error: The reconstruction error after removal.
    """
    k_current = len(basis_funcs)
    if k_current <= 0:
         return [], -1, float('inf')

    best_new_basis = []
    best_removed_idx = -1
    min_error_after_removal = float('inf')

    # Probe each function
    for i in range(k_current):
        # Create a candidate basis excluding function i
        candidate_basis = basis_funcs[:i] + basis_funcs[i+1:]
        
        # Calculate error
        _, _, error = fn_can_represent_as_linear_combination(
            target_function, candidate_basis, rtol=0.0, atol=atol
        )

        if error < min_error_after_removal:
            min_error_after_removal = error
            best_removed_idx = i
            best_new_basis = candidate_basis
    
    return best_new_basis, best_removed_idx, min_error_after_removal
