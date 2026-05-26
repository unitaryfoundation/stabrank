import numpy as np
import random
import datetime
import sys
import os

# Ensure the parent directory is in the path to import stabrank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qutrit_complex_magic_state,
    can_represent_as_linear_combination,
    prune_least_significant_basis_function
)

from stabrank.polynomial_utils import generate_random_stabilizer_state
from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion

def run_qutrit_complex_example():
    """
    Runs an example of finding a stabilizer rank decomposition for the Qutrit Complex Magic State (|T3>).
    """
    print("\n--- Qutrit Complex Magic State Decomposition Example (|T3>) ---")
    
    # --- SA Parameters ---
    seed = np.random.randint(0, 10000000)
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    
    initial_temperature = 1.0
    cooling_rate = 0.99 
    num_iterations_at_temp = 20000
    min_temperature = 1/4000
    atol = 1e-7
    ERROR_THRESHOLD = 1e-9
    
    # --- Problem Setup (Qutrits p=3) ---
    n_orig = 3   # Number of qutrits
    p_prime = 3  # Qutrits
    
    # Target: Complex magic state
    target_func = qutrit_complex_magic_state(n_orig)
    
    # Start with k=8
    k_start = 5
    
    # Initial Basis
    print(f"Generating initial random basis of {k_start} functions...")
    initial_basis = []
    
    for _ in range(k_start):
        func = generate_random_stabilizer_state(n_orig, p=p_prime)
        initial_basis.append(func)

    current_best_funcs = initial_basis
    current_k = k_start
    
    # --- Squeeze Loop ---
    while current_k > 0:
        print(f"\n=== Optimizing for k={current_k} ===")
        found_solution = False
        final_lin_coeffs = None
        final_error = float('inf')
        
        runs = 1
        
        for i in range(runs):
            print(f"--- SA Run {i+1}/{runs} ---")
            
            _, best_funcs_tuple, best_lin_coeffs, best_error, best_cost, _trace = \
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
                    use_real_qubit_moves=False,  # DO NOT use real qubit moves for qutrits!
                    clifford_ratio=0.5,
                    early_exit_threshold=ERROR_THRESHOLD,
                    seed=seed + i,
                    num_chains=16
                )
            best_funcs = list(best_funcs_tuple)
            
            current_best_funcs = best_funcs
            final_error = best_error
            final_lin_coeffs = best_lin_coeffs
            
            if final_error < ERROR_THRESHOLD:
                found_solution = True
                break
        
        if found_solution:
            print(f"✅ Found representation for k={current_k} (Error: {final_error:.2e})")
            
            # Save
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_qutrit_complex_k{current_k}_n{n_orig}_{timestamp}.npz"
            np.savez_compressed(filename, 
                                target_function=target_func, 
                                linear_coeffs=final_lin_coeffs,
                                n=n_orig, p=p_prime, k=current_k, final_error=final_error,
                                **{f'basis_func_{i}': f for i, f in enumerate(current_best_funcs)})
            print(f"Saved to {filename}")
            
            # Keep pruning as long as the error stays below threshold!
            while current_k > 0:
                print(f"✂️  Pruning from k={current_k} -> k={current_k-1}")
                next_funcs, idx, err = prune_least_significant_basis_function(
                    target_func, current_best_funcs, can_represent_as_linear_combination
                )
                print(f"Removed index {idx}. New error: {err:.2e}")
                
                current_k -= 1
                current_best_funcs = next_funcs
                
                if err < ERROR_THRESHOLD:
                    print(f"✨ Pruned basis still satisfies error threshold ({err:.2e} < {ERROR_THRESHOLD:.2e})!")
                    print(f"✅ Automatically found representation for k={current_k} via pruning.")
                    
                    is_rep, new_lin_coeffs, _ = can_represent_as_linear_combination(target_func, current_best_funcs)
                    
                    filename = f"solution_qutrit_complex_k{current_k}_n{n_orig}_{timestamp}.npz"
                    np.savez_compressed(filename, 
                                        target_function=target_func, 
                                        linear_coeffs=new_lin_coeffs,
                                        n=n_orig, p=p_prime, k=current_k, final_error=err,
                                        **{f'basis_func_{i}': f for i, f in enumerate(current_best_funcs)})
                    print(f"Saved to {filename}")
                else:
                    break
        else:
            print(f"🛑 Failed to converge only k={current_k}. Best error: {final_error:.2e}")
            break

if __name__ == "__main__":
    run_qutrit_complex_example()
