import numpy as np
import random
import datetime
import sys
import os
import math

# Ensure the parent directory is in the path to import stabrank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qubit_magic_code_state_compressed,
    can_represent_as_linear_combination,
    prune_least_significant_basis_function
)

from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion

def run_magic_code_state_example():
    """
    Runs an example of finding a stabilizer rank decomposition for a Magic Code State.
    Specifically, we test the 6-qubit magic cat state, which is the magic code state
    for the 6-bit repetition code (where G = [1, 1, 1, 1, 1, 1]).
    The paper proves this state has a stabilizer rank <= 3. Let's find it!
    """
    print("\n--- Magic Code State Decomposition Example (Cat-6) ---")
    
    # --- SA Parameters ---
    seed = np.random.randint(0, 10000000)
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    
    initial_temperature = 1.0
    cooling_rate = 0.99 
    num_iterations_at_temp = 4000
    min_temperature = 1/4000
    atol = 1e-7
    ERROR_THRESHOLD = 1e-9
    
    # --- Problem Setup ---
    m = 6   # Number of qubits (length of the code)
    k = 1   # Dimension of the code
    p_prime = 2  # Qubits
    
    # Generator matrix for the 6-bit repetition code
    # Shape: (k, m) = (1, 6)
    G = np.array([[1, 1, 0, 0, 0, 0]], dtype=int)
    
    print(f"Code Length (m): {m}")
    print(f"Code Dimension (k): {k}")
    print(f"Generator Matrix G:\n{G}")
    
    # Target: Magic Code State
    target_func = qubit_magic_code_state_compressed(G)
    
    # Let's start with k=5 and prune down to see if we can hit 3.
    k_start = 3
    
    n_eff = m - k

    # Initial Basis
    from stabrank.polynomial_utils import generate_random_stabilizer_state
    print(f"\nGenerating initial random basis of {k_start} functions...")
    initial_basis = []
    for _ in range(k_start):
        func = generate_random_stabilizer_state(n_eff, p=p_prime)
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
                    n_orig=n_eff,
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
                    use_real_qubit_moves=False,
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
            eff_rank = current_k * (2**k)
            if m - 2*k > 0:
                gamma = math.log(current_k, 2) / (m - 2*k)
            else:
                gamma = float('inf')
            print(f"✅ Found representation for chi={current_k} (Error: {final_error:.2e})")
            print(f"   => eff_rank={eff_rank}, gamma={gamma:.4f}")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_magic_code_state_k{current_k}_m{m}_{timestamp}.npz"
            np.savez_compressed(filename, 
                                target_function=target_func, 
                                linear_coeffs=final_lin_coeffs,
                                n=m, p=p_prime, k=current_k, final_error=final_error,
                                **{f'basis_func_{i}': f for i, f in enumerate(current_best_funcs)})
            print(f"Saved to {filename}")
            
            while current_k > 0:
                print(f"✂️  Pruning from k={current_k} -> k={current_k-1}")
                next_funcs, idx, err = prune_least_significant_basis_function(
                    target_func, current_best_funcs, can_represent_as_linear_combination
                )
                print(f"Removed index {idx}. New error: {err:.2e}")
                
                current_k -= 1
                current_best_funcs = next_funcs
                
                if err < ERROR_THRESHOLD:
                    eff_rank = current_k * (2**k)
                    if m - 2*k > 0:
                        gamma = math.log(current_k, 2) / (m - 2*k)
                    else:
                        gamma = float('inf')
                    print(f"✨ Pruned basis still satisfies error threshold ({err:.2e} < {ERROR_THRESHOLD:.2e})!")
                    print(f"✅ Automatically found representation for chi={current_k} via pruning.")
                    print(f"   => eff_rank={eff_rank}, gamma={gamma:.4f}")
                    
                    is_rep, new_lin_coeffs, _ = can_represent_as_linear_combination(target_func, current_best_funcs)
                    
                    filename = f"solution_magic_code_state_k{current_k}_m{m}_{timestamp}.npz"
                    np.savez_compressed(filename, 
                                        target_function=target_func, 
                                        linear_coeffs=new_lin_coeffs,
                                        n=m, p=p_prime, k=current_k, final_error=err,
                                        **{f'basis_func_{i}': f for i, f in enumerate(current_best_funcs)})
                    print(f"Saved to {filename}")
                else:
                    break
        else:
            print(f"🛑 Failed to converge only k={current_k}. Best error: {final_error:.2e}")
            break

if __name__ == "__main__":
    run_magic_code_state_example()
