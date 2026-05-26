import numpy as np
import random
import itertools
import datetime
import sys
import os
import csv
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabrank import (
    qubit_magic_code_state_compressed,
    can_represent_as_linear_combination,
    prune_least_significant_basis_function
)
from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion

def run_sweep(m: int, target_effective_rank: int):
    """
    Sweeps over all systematic codes of length m to find magic code states
    that beat or tie the target_effective_rank.
    
    Effective Rank = 2^k * chi(C_m)
    We want to find codes where 2^k * chi(C_m) <= target_effective_rank.
    This means we only need to search for chi(C_m) <= target_effective_rank // 2^k.
    """
    print(f"\n--- Sweeping Magic Code States for m={m} ---")
    print(f"Target Effective Rank: <= {target_effective_rank}")
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    # SA Params for Fast Pass
    fast_temp = 1.0
    fast_cooling = 0.95
    fast_iters = 100
    fast_min_temp = 0.01
    fast_chains = 8
    
    # SA Params for Deep Pass
    deep_temp = 1.0
    deep_cooling = 0.99
    deep_iters = 500
    deep_min_temp = 1/4000
    deep_chains = 16
    
    ERROR_THRESHOLD = 1e-9
    
    total_codes_checked = 0
    all_results = []
    
    timestamp_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sweep_results_m{m}_{timestamp_run}.csv"
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'k', 'A_idx', 'G_flat', 'min_chi', 'effective_rank', 'final_error'])
    
    # Mathematical bound for exponent < 0.5 requires k < m/2
    max_k = (m + 1) // 2 
    
    for k in range(1, max_k):
        # Iterate over all possible k x (m-k) binary matrices A
        num_A = 2**(k * (m-k))
        print(f"\nEvaluating k={k} (Total systematic codes: {num_A})")
        
        for A_idx, A_flat in enumerate(itertools.product([0, 1], repeat=k * (m-k))):
            A = np.array(A_flat).reshape((k, m-k))
            G = np.hstack([np.eye(k, dtype=int), A])
            
            target_func = qubit_magic_code_state_compressed(G)
            
            # Start squeezing from m-1
            k_start = m - 1
            
            # The SA search now occurs in the reduced dimension n_eff = m - k
            n_eff = m - k
            
            # Initial random basis
            from stabrank.polynomial_utils import generate_random_stabilizer_state
            initial_basis = []
            for _ in range(k_start):
                func = generate_random_stabilizer_state(n_eff, p=2)
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
                    p_prime=2,
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
                        # Continue pruning
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
                        # Pruning failed, loop will run SA on current_chi (which is now old_chi - 1)
                        pass
                else:
                    # SA failed to converge for this chi. The best successful chi is the previous one.
                    break
                    
            if best_successful_chi is not None:
                eff_rank = best_successful_chi * (2**k)
                if m - 2*k > 0:
                    gamma = math.log(best_successful_chi, 2) / (m - 2*k)
                else:
                    gamma = float('inf')
                print(f"min_chi={best_successful_chi}, eff_rank={eff_rank}, gamma={gamma:.4f}")
                
                # Write to CSV
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    G_str = ''.join(map(str, G.flatten()))
                    writer.writerow([m, k, A_idx, G_str, best_successful_chi, eff_rank, best_successful_error, gamma])
                
            total_codes_checked += 1
                    
    print("\n=== SWEEP COMPLETE ===")
    print(f"Total codes checked: {total_codes_checked}")
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    # Test m=6. The cat_6 state has effective rank 6 (k=1, chi=3).
    # Let's see if we can find any code that matches or beats 6.
    run_sweep(m=6, target_effective_rank=6)
