import numpy as np
import itertools
import datetime
import sys
import os
import csv
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qutrit_magic_code_state_compressed,
    can_represent_as_linear_combination,
    prune_least_significant_basis_function
)
from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion

def run_qutrit_sweep(m: int):
    print(f"--- Sweeping Qutrit Magic Code States for m={m} ---")
    
    # SA Params for Deep Pass
    deep_temp = 1.0
    deep_cooling = 0.99
    deep_iters = 2000
    deep_min_temp = 1/4000
    deep_chains = 16
    
    ERROR_THRESHOLD = 1e-9
    
    total_codes_checked = 0
    all_results = []
    
    timestamp_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sweep_qutrit_results_m{m}_{timestamp_run}.csv"
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'k', 'A_idx', 'G_flat', 'min_chi', 'effective_rank', 'final_error', 'gamma'])
    
    # Mathematical bound for exponent < 0.5 requires k < m/2
    max_k = 2 
    
    # Fixed random seed based on m for reproducibility
    seed = m * 1000000
    
    for k in range(1, max_k):
        # Iterate over all possible k x (m-k) ternary matrices A
        num_A = 3**(k * (m-k))
        print(f"\nEvaluating k={k} (Total systematic ternary codes: {num_A})")
        
        for A_idx, A_flat in enumerate(itertools.product([0, 1, 2], repeat=k * (m-k))):
            A = np.array(A_flat).reshape((k, m-k))
            G = np.hstack([np.eye(k, dtype=int), A])
            
            target_func = qutrit_magic_code_state_compressed(G)
            
            # Start squeezing from m-1
            k_start = 6
            
            # The SA search now occurs in the reduced dimension n_eff = m - k
            n_eff = m - k
            
            # Initial random basis
            from stabrank.polynomial_utils import generate_random_stabilizer_state
            initial_basis = []
            for _ in range(k_start):
                func = generate_random_stabilizer_state(n_eff, p=3)
                initial_basis.append(func)
                
            current_best_funcs = initial_basis
            current_chi = k_start
            
            best_successful_chi = None
            best_successful_error = None
            best_successful_funcs = None
            best_successful_coeffs = None
            
            print(f"  Testing code {A_idx+1}/{num_A} (k={k}) ... ", end="", flush=True)
            
            while current_chi > 0:
                _, best_funcs, best_coeffs, deep_error, _, _trace = cpp_run_sa_pauli_expansion(
                    target=target_func,
                    n_orig=n_eff,
                    p_prime=3,
                    k_subset_size=current_chi,
                    initial_basis=current_best_funcs,
                    initial_temperature=deep_temp,
                    cooling_rate=deep_cooling,
                    num_iterations_at_temp=deep_iters,
                    min_temperature=deep_min_temp,
                    atol=1e-7,
                    two_func_perturb_prob=0.3,
                    random_replace_prob=0.05,
                    use_real_qubit_moves=False,
                    clifford_ratio=0.5,
                    early_exit_threshold=ERROR_THRESHOLD,
                    seed=seed + A_idx + current_chi,
                    num_chains=deep_chains
                )
                
                if deep_error < ERROR_THRESHOLD:
                    # Success!
                    best_successful_chi = current_chi
                    best_successful_error = deep_error
                    best_successful_funcs = best_funcs
                    best_successful_coeffs = best_coeffs
                    
                    # Try to prune
                    next_funcs, _, err = prune_least_significant_basis_function(
                        target_func, best_funcs, can_represent_as_linear_combination
                    )
                    
                    current_chi -= 1
                    current_best_funcs = next_funcs
                    
                    if err < ERROR_THRESHOLD:
                        # Pruning successful
                        best_successful_chi = current_chi
                        best_successful_error = err
                        best_successful_funcs = next_funcs
                        _, new_lin_coeffs, _ = can_represent_as_linear_combination(target_func, next_funcs)
                        best_successful_coeffs = new_lin_coeffs
                        # Continue pruning down
                        while current_chi > 0:
                            next_funcs, _, err = prune_least_significant_basis_function(
                                target_func, current_best_funcs, can_represent_as_linear_combination
                            )
                            if err < ERROR_THRESHOLD:
                                current_chi -= 1
                                current_best_funcs = next_funcs
                                best_successful_chi = current_chi
                                best_successful_error = err
                                best_successful_funcs = next_funcs
                                _, new_lin_coeffs, _ = can_represent_as_linear_combination(target_func, next_funcs)
                                best_successful_coeffs = new_lin_coeffs
                            else:
                                break
                    else:
                        pass
                else:
                    # SA failed to converge for this chi. 
                    break
                    
            if best_successful_chi is not None:
                eff_rank = best_successful_chi * (3**k)
                if m - 2*k > 0:
                    gamma = math.log(best_successful_chi, 3) / (m - 2*k)
                else:
                    gamma = float('inf')
                
                print(f"min_chi={best_successful_chi}, eff_rank={eff_rank}, gamma={gamma:.4f}")
                
                # Write to CSV
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    G_str = ''.join(map(str, G.flatten()))
                    writer.writerow([m, k, A_idx, G_str, best_successful_chi, eff_rank, best_successful_error, gamma])
            else:
                print(f"FAILED to find representation even at chi={k_start}")
                
            total_codes_checked += 1
                    
    print("\n=== SWEEP COMPLETE ===")
    print(f"Total ternary codes checked: {total_codes_checked}")
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    # Test m=5. 
    # For m=5, k<2.5 implies k=1, 2.
    # Total codes = 3^4 + 3^6 = 81 + 729 = 810 codes.
    run_qutrit_sweep(m=5)
