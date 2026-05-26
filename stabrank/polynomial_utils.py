"""Polynomial utilities for stabilizer rank computations."""

import numpy as np
import itertools
import re
import random
from numba import njit
from numba.typed import List
from .linalg_utils import get_lex_index

@njit(cache=True)
def _calculate_poly_value_at_point_numba(
    point_y_or_x, k_or_n, p_prime, alpha_coeff,
    c_j0_linear, c_j0_quadratic_mixed,
    c_j0_quadratic_square, c_j1_linear_p2only
):
    """Calculate polynomial value at a point using Numba for performance."""
    poly_value = float(alpha_coeff)
    
    if k_or_n > 0:
        current_sum_j0_lin = 0.0
        for i in range(k_or_n):
            current_sum_j0_lin += point_y_or_x[i] * c_j0_linear[i]
        poly_value += current_sum_j0_lin / float(p_prime)

        if p_prime >= 3:
            current_sum_j0_sq = 0.0
            for i in range(k_or_n):
                current_sum_j0_sq += (point_y_or_x[i]**2) * c_j0_quadratic_square[i]
            poly_value += current_sum_j0_sq / float(p_prime)

    coeff_idx_j0_quad = 0
    for s_idx in range(k_or_n):
        for t_idx in range(s_idx + 1, k_or_n):
            term_val = float(point_y_or_x[s_idx]) * float(point_y_or_x[t_idx]) * \
                       c_j0_quadratic_mixed[coeff_idx_j0_quad]
            poly_value += term_val / float(p_prime)
            coeff_idx_j0_quad += 1
            
    if p_prime == 2 and k_or_n > 0:
        current_sum_j1_lin = 0.0
        for i in range(k_or_n):
            current_sum_j1_lin += point_y_or_x[i] * c_j1_linear_p2only[i]
        poly_value += current_sum_j1_lin / 4.0
        
    res = poly_value % 1.0
    if res < 0:
        res += 1.0
    return res

@njit(cache=True)
def _fill_one_output_on_n_orig_space(
    current_output_on_n_orig_space, y_params_list_of_arrays, n_orig, p_prime,
    x0_translation, W_basis, k_dim, alpha_val, c_j0_lin_arr_k, c_j0_qm_arr_k,
    c_j0_qs_arr_k_eff, c_j1_lin_arr_k_eff
):
    """Fill output array with polynomial evaluations."""
    for i in range(len(y_params_list_of_arrays)):
        y_param_vec = y_params_list_of_arrays[i]
        w_component = np.zeros(n_orig, dtype=np.int64)
        
        if k_dim > 0 and n_orig > 0:
            for j_idx in range(n_orig):
                sum_val = 0
                for i_idx in range(k_dim):
                    sum_val += y_param_vec[i_idx] * W_basis[i_idx, j_idx]
                w_component[j_idx] = sum_val
        
        x_in_orig_space = (x0_translation + w_component) % p_prime
        q_y_val_on_torus = _calculate_poly_value_at_point_numba(
            y_param_vec, k_dim, p_prime, alpha_val,
            c_j0_lin_arr_k, c_j0_qm_arr_k,
            c_j0_qs_arr_k_eff, c_j1_lin_arr_k_eff
        )
        complex_phase = np.exp(2j * np.pi * q_y_val_on_torus)
        
        if n_orig > 0:
            lex_idx_orig = get_lex_index(x_in_orig_space, n_orig, p_prime)
            current_output_on_n_orig_space[lex_idx_orig] = complex_phase
        else:
            current_output_on_n_orig_space[0] = complex_phase

def format_polynomial_string(k_dim, p_prime, alpha_val, c_j0_lin, c_j0_qm, c_j0_qs, c_j1_lin):
    """Format polynomial coefficients into readable string."""
    terms = []
    if alpha_val != 0.0: 
        terms.append(f"{alpha_val:.3g}")
    
    for i in range(k_dim):
        c = c_j0_lin[i]
        if c != 0: 
            terms.append(f"({f'y_{i}' if c == 1 else f'{c}*y_{i}'})/{p_prime}")
    
    if p_prime >= 3:
        for i in range(k_dim):
            c = c_j0_qs[i]
            if c != 0: 
                terms.append(f"({f'y_{i}^2' if c == 1 else f'{c}*y_{i}^2'})/{p_prime}")
    
    coeff_idx = 0
    for s_idx in range(k_dim):
        for t_idx in range(s_idx + 1, k_dim):
            c = c_j0_qm[coeff_idx]
            if c != 0: 
                terms.append(f"({f'y_{s_idx}*y_{t_idx}' if c == 1 else f'{c}*y_{s_idx}*y_{t_idx}'})/{p_prime}")
            coeff_idx += 1
    
    if p_prime == 2:
        for i in range(k_dim):
            c = c_j1_lin[i]
            if c != 0: 
                terms.append(f"({f'y_{i}' if c == 1 else f'{c}*y_{i}'})/4")
    
    return " + ".join(terms) if terms else "0.0"

def generate_distinct_phases_on_param_subspace_v2(
    n_orig, p_prime, x0_translation, W_basis, alpha_values_to_test=None
):
    """Generate distinct phase functions on parameter subspace."""
    if not isinstance(x0_translation, np.ndarray) or x0_translation.shape != (n_orig,):
        if not (n_orig == 0 and x0_translation.shape == (0,)):
             raise ValueError(f"x0_translation must be shape ({n_orig},), got {x0_translation.shape}")
    
    expected_W_shape_1 = n_orig if n_orig > 0 else 0
    if not isinstance(W_basis, np.ndarray) or W_basis.ndim != 2 or W_basis.shape[1] != expected_W_shape_1:
        if W_basis.ndim == 1 and W_basis.shape == (0,) and W_basis.shape[0] == 0:
             W_basis = W_basis.reshape(0, expected_W_shape_1)
        elif W_basis.shape == (0,0) and n_orig > 0 and W_basis.shape[1] != expected_W_shape_1:
            W_basis = np.empty((0, expected_W_shape_1), dtype=W_basis.dtype)
        elif not (W_basis.ndim == 2 and W_basis.shape[1] == expected_W_shape_1):
             raise ValueError(f"W_basis must be shape (k, {expected_W_shape_1}). Got {W_basis.shape}")

    k_dim = W_basis.shape[0]
    if k_dim > 4: 
        raise ValueError("Generator intended for k <= 4.")
    if alpha_values_to_test is None: 
        alpha_values_to_test = [0.0]
    
    y_param_tuples = list(itertools.product(range(p_prime), repeat=k_dim))
    y_params_list_for_numba = List()
    if k_dim == 0:
        y_params_list_for_numba.append(np.array([], dtype=np.int64))
    else:
        for tpl in y_param_tuples:
            y_params_list_for_numba.append(np.array(tpl, dtype=np.int64))
            
    all_generated_functions, all_polynomial_strings = [], []
    coeff_range = range(p_prime)
    
    iter_c_j0_lin_k = list(itertools.product(coeff_range, repeat=k_dim))
    iter_c_j0_qm_k = list(itertools.product(coeff_range, repeat=k_dim * (k_dim - 1) // 2 if k_dim >=2 else 0))
    iter_c_j0_qs_k = list(itertools.product(coeff_range, repeat=k_dim)) if p_prime >= 3 and k_dim > 0 else [()]
    iter_c_j1_lin_k = list(itertools.product(coeff_range, repeat=k_dim)) if p_prime == 2 and k_dim > 0 else [()]

    for alpha_val in alpha_values_to_test:
        for c_j0_lin_tuple in iter_c_j0_lin_k:
            c_j0_lin_arr = np.array(c_j0_lin_tuple, dtype=np.int64)
            for c_j0_qm_tuple in iter_c_j0_qm_k:
                c_j0_qm_arr = np.array(c_j0_qm_tuple, dtype=np.int64)
                for c_j0_qs_tuple in iter_c_j0_qs_k:
                    c_j0_qs_arr = np.array(c_j0_qs_tuple, dtype=np.int64) if p_prime >= 3 and k_dim > 0 else np.zeros(k_dim, dtype=np.int64)
                    for c_j1_lin_tuple in iter_c_j1_lin_k:
                        c_j1_lin_arr = np.array(c_j1_lin_tuple, dtype=np.int64) if p_prime == 2 and k_dim > 0 else np.zeros(k_dim, dtype=np.int64)
                        
                        if k_dim == 0:
                            c_j0_lin_arr = np.array([], dtype=np.int64)
                            c_j0_qm_arr = np.array([], dtype=np.int64)
                            c_j0_qs_arr = np.array([], dtype=np.int64)
                            c_j1_lin_arr = np.array([], dtype=np.int64)

                        output_len = p_prime**n_orig if n_orig > 0 else 1
                        current_output = np.zeros(output_len, dtype=complex)
                        _fill_one_output_on_n_orig_space(
                            current_output, y_params_list_for_numba, n_orig, p_prime,
                            x0_translation, W_basis, k_dim, float(alpha_val),
                            c_j0_lin_arr, c_j0_qm_arr, c_j0_qs_arr, c_j1_lin_arr
                        )
                        all_generated_functions.append(current_output)
                        poly_str = format_polynomial_string(
                            k_dim, p_prime, alpha_val, c_j0_lin_arr,
                            c_j0_qm_arr, c_j0_qs_arr, c_j1_lin_arr
                        )
                        all_polynomial_strings.append(poly_str)
    return all_generated_functions, all_polynomial_strings

def evaluate_poly_string_on_subspace(
    poly_string: str, n_orig: int, p_prime: int, x0_translation: np.ndarray, W_basis: np.ndarray
) -> np.ndarray:
    """Evaluate polynomial string on subspace."""
    k_dim = W_basis.shape[0]
    y_indices_from_string = [int(i) for i in re.findall(r'y_(\d+)', poly_string)]
    if y_indices_from_string and max(y_indices_from_string) >= k_dim:
        raise ValueError(
            f"Polynomial string contains y_{max(y_indices_from_string)}, which is out of bounds for "
            f"subspace dimension k_dim={k_dim}."
        )

    alpha_coeff, c_j0_lin, c_j0_qm, c_j0_qs, c_j1_lin = 0.0, \
        np.zeros(k_dim, dtype=np.int64), \
        np.zeros(k_dim * (k_dim - 1) // 2 if k_dim >= 2 else 0, dtype=np.int64), \
        np.zeros(k_dim, dtype=np.int64), \
        np.zeros(k_dim, dtype=np.int64)

    normalized_poly_string = ' '.join(poly_string.replace("+", " + ").split())
    terms = normalized_poly_string.split(' + ')

    for term in terms:
        term = term.strip().replace('(', '').replace(')', '').replace('|', '')
        if not term: 
            continue

        if 'y' not in term and '/' not in term: 
            try: 
                alpha_coeff = float(term)
            except ValueError: 
                pass 
            continue
        
        if '/' not in term:
            if 'y' in term:
                 raise ValueError(f"Malformed term (missing denominator?): '{term}'")
            else: 
                try: 
                    alpha_coeff = float(term)
                    continue
                except ValueError: 
                    pass 
                continue

        num_part, den_part = term.split('/')
        denominator = int(den_part)
        coeff, vars_part = 1, num_part
        if '*' in num_part:
            parts = num_part.split('*', 1)
            if parts[0].isdigit() or (parts[0].startswith('-') and parts[0][1:].isdigit()):
                coeff, vars_part = int(parts[0]), parts[1]
            else: 
                vars_part = num_part
        elif num_part.startswith('y_'): 
            vars_part = num_part
        
        if '^2' in vars_part:
            match = re.search(r'y_(\d+)', vars_part)
            if match: 
                y_idx = int(match.group(1))
            else: 
                raise ValueError(f"Could not parse y_idx from square term: {vars_part}")
            c_j0_qs[y_idx] = coeff
        elif '*' in vars_part: 
            y_matches = re.findall(r'y_(\d+)', vars_part)
            if len(y_matches) == 2:
                s, t = sorted([int(i) for i in y_matches])
                qm_idx, current_idx_count, found_qm_idx = 0, 0, False
                for i_loop in range(k_dim):
                    for j_loop in range(i_loop + 1, k_dim):
                        if i_loop == s and j_loop == t: 
                            qm_idx, found_qm_idx = current_idx_count, True
                            break
                        current_idx_count += 1
                    if found_qm_idx: 
                        break
                if not found_qm_idx and c_j0_qm.size > 0: 
                     raise ValueError(f"Could not determine qm_idx for s={s}, t={t}")
                if c_j0_qm.size > 0: 
                    c_j0_qm[qm_idx] = coeff
            else: 
                raise ValueError(f"Malformed mixed term: {vars_part}")
        else: 
            match = re.search(r'y_(\d+)', vars_part)
            if match: 
                y_idx = int(match.group(1))
            else: 
                raise ValueError(f"Could not parse y_idx from linear term: {vars_part}")
            if denominator == p_prime: 
                c_j0_lin[y_idx] = coeff
            elif denominator == 4 and p_prime == 2: 
                c_j1_lin[y_idx] = coeff
            else: 
                raise ValueError(f"Unknown denominator {denominator} for linear term with p_prime={p_prime}")
    
    output_array = np.zeros(p_prime**n_orig if n_orig > 0 else 1, dtype=complex)
    y_param_tuples = list(itertools.product(range(p_prime), repeat=k_dim))
    y_params_list_for_numba = List() 
    if k_dim == 0: 
        y_params_list_for_numba.append(np.array([], dtype=np.int64))
    else:
        for tpl in y_param_tuples: 
            y_params_list_for_numba.append(np.array(tpl, dtype=np.int64))
    
    _fill_one_output_on_n_orig_space(
        output_array, y_params_list_for_numba, n_orig, p_prime,
        x0_translation, W_basis, k_dim, alpha_coeff,
        c_j0_lin, c_j0_qm, c_j0_qs, c_j1_lin
    )
    return output_array

def generate_random_poly_coeffs(k_dim_poly: int, p_prime: int, alpha_options: list = [0.0]) -> tuple:
    """Generate random polynomial coefficients."""
    alpha_val = random.choice(alpha_options)
    c_j0_lin = np.random.randint(0, p_prime, size=k_dim_poly, dtype=np.int64)
    
    qm_len = k_dim_poly * (k_dim_poly - 1) // 2 if k_dim_poly >= 2 else 0
    c_j0_qm = np.random.randint(0, p_prime, size=qm_len, dtype=np.int64)
    
    c_j0_qs = np.zeros(k_dim_poly, dtype=np.int64)
    if p_prime >= 3:
        c_j0_qs = np.random.randint(0, p_prime, size=k_dim_poly, dtype=np.int64)
        
    c_j1_lin = np.zeros(k_dim_poly, dtype=np.int64)
    if p_prime == 2:
        c_j1_lin = np.random.randint(0, p_prime, size=k_dim_poly, dtype=np.int64)

    return alpha_val, c_j0_lin, c_j0_qm, c_j0_qs, c_j1_lin

def evaluate_coeffs_on_subspace(
    coeffs_tuple: tuple, n_orig: int, p_prime: int, x0_translation: np.ndarray, W_basis: np.ndarray
) -> np.ndarray:
    """Evaluate polynomial coefficients on subspace."""
    k_dim_poly = W_basis.shape[0]
    alpha_val, c_j0_lin_arr_k, c_j0_qm_arr_k, c_j0_qs_arr_k_eff, c_j1_lin_arr_k_eff = coeffs_tuple

    expected_qm_len = k_dim_poly * (k_dim_poly - 1) // 2 if k_dim_poly >= 2 else 0
    if not (len(c_j0_lin_arr_k) == k_dim_poly and \
            len(c_j0_qs_arr_k_eff) == k_dim_poly and \
            len(c_j1_lin_arr_k_eff) == k_dim_poly and \
            len(c_j0_qm_arr_k) == expected_qm_len):
        raise ValueError(f"Coefficient array dimensions mismatch for k_dim_poly={k_dim_poly}")

    output_len = p_prime**n_orig if n_orig > 0 else 1
    output_array = np.zeros(output_len, dtype=complex)
    
    y_param_tuples = list(itertools.product(range(p_prime), repeat=k_dim_poly))
    y_params_list_for_numba = List()
    if k_dim_poly == 0:
        y_params_list_for_numba.append(np.array([], dtype=np.int64))
    else:
        for tpl in y_param_tuples:
            y_params_list_for_numba.append(np.array(tpl, dtype=np.int64))

    _fill_one_output_on_n_orig_space(
        output_array, y_params_list_for_numba, n_orig, p_prime,
        x0_translation, W_basis, k_dim_poly,
        float(alpha_val), 
        c_j0_lin_arr_k, c_j0_qm_arr_k,
        c_j0_qs_arr_k_eff, c_j1_lin_arr_k_eff
    )
    return output_array

def get_vector_from_index(index: int, n: int, p: int) -> list[int]:
    """
    Converts an index to its base-p vector representation as a list of ints.
    This is the corrected version.
    """
    base_p_string = np.base_repr(index, base=p).zfill(n)
    # Explicitly convert each character digit to an integer
    return [int(digit) for digit in base_p_string]

def get_index_from_vector(vector: list[int], n: int, p: int) -> int:
    index = 0
    for i in range(n):
        index += vector[i] * (p**(n - 1 - i))
    return index

def apply_X_operator(poly_array: np.ndarray, i: int, n: int, p: int) -> np.ndarray:
    """
    Applies the local X operator on index i to a polynomial's array representation.

    Args:
        poly_array: The numpy array of complex values representing the polynomial.
        i: The index of the variable (0 to n-1) to which the operator is applied.
        n: The number of variables.
        p: The prime base.

    Returns:
        A new numpy array representing the transformed polynomial.
    """
    num_elements = p**n
    new_poly_array = np.zeros(num_elements, dtype=poly_array.dtype)

    for target_index in range(num_elements):
        # 1. Get the vector for the current target index
        target_vector = get_vector_from_index(target_index, n, p)

        # 2. Find the source vector by applying the *inverse* operator.
        #    The new value at `target_vector` comes from the old value at the
        #    vector that gets mapped TO it.
        source_vector = list(target_vector)
        source_vector[i] = (source_vector[i] - 1 + p) % p

        # 3. Get the index of the source vector
        source_index = get_index_from_vector(source_vector, n, p)

        # 4. Assign the value
        new_poly_array[target_index] = poly_array[source_index]

    return new_poly_array

def apply_Z_operator(poly_array: np.ndarray, i: int, n: int, p: int) -> np.ndarray:
    """
    Applies the local Z operator on index i to a polynomial's array representation.

    Args:
        poly_array: The numpy array of complex values representing the polynomial.
        i: The index of the variable (0 to n-1) on which the operator acts.
        n: The number of variables.
        p: The prime base.

    Returns:
        A new numpy array representing the transformed polynomial.
    """
    num_elements = p**n
    
    # 1. Define the primitive p-th root of unity
    omega = np.exp(2j * np.pi / p)

    # 2. Generate an array of all x_i values for every index
    # This creates an array [0, 0, ..., 1, 1, ..., etc.] for the i-th component
    indices = np.arange(num_elements)
    # The value of x_i is equivalent to (index // p^(n-1-i)) % p
    xi_values = (indices // (p**(n - 1 - i))) % p
    
    # 3. Calculate the phase factor for every element at once
    phases = omega**xi_values

    # 4. Apply the phases to the polynomial array with a single multiplication
    new_poly_array = poly_array * phases

    return new_poly_array

def apply_Y_operator(poly_array: np.ndarray, i: int, n: int, p: int) -> np.ndarray:
    """
    Applies the local Y operator (Z then X) on index i.

    Args:
        poly_array: The numpy array of complex values.
        i: The index of the variable (0 to n-1) to which the operator is applied.
        n: The number of variables.
        p: The prime base.

    Returns:
        A new numpy array representing the transformed polynomial.
    """
    # 1. Apply the Z operator first
    z_transformed_array = apply_Z_operator(poly_array, i, n, p)
    
    # 2. Apply the X operator to the result
    y_transformed_array = apply_X_operator(z_transformed_array, i, n, p)
    
    return y_transformed_array

def apply_I_operator(poly_array: np.ndarray, i: int, n: int, p: int) -> np.ndarray:
    """
    Applies the Identity operator, which returns the array unmodified.
    The arguments i, n, and p are included for a consistent interface.
    """
    return poly_array

def choose_random_operator() -> str:
    """
    Randomly selects and returns the name of an operator from ('I', 'X', 'Y', 'Z').
    """
    operators = ["I", "X", "Y", "Z"]
    return random.choice(operators)

def apply_random_pauli_string(poly_array: np.ndarray, n: int, p: int, constraint: str = None) -> tuple[np.ndarray, list[str]]:
    """
    Applies a random Pauli string X^a Z^b to the basis function represented by poly_array.
    
    constraint: If 'even_y', ensures the Pauli string has an even number of Y operators (for p=2).
    """
    
    # ... logic for choosing operators ...
    operator_functions = {
        "I": apply_I_operator,
        "X": apply_X_operator,
        "Y": apply_Y_operator,
        "Z": apply_Z_operator,
    }

    max_retries = 100
    chosen_operators = []
    
    # Rejection sampling loop for constraints
    for attempt in range(max_retries):
        chosen_operators = []
        y_count = 0
        
        # Loop through each variable index from 0 to n-1
        for i in range(n):
            op_name = choose_random_operator()
            chosen_operators.append(op_name)
            if op_name == 'Y':
                y_count += 1
        
        if constraint == 'even_y':
            if p != 2:
                 # Constraint only valid for qubits, ignore or warn? For now ignore.
                 break
            if y_count % 2 == 0:
                break # Valid
            # Else retry
        else:
            break # No constraint
            
    else:
        # If we failed to satisfy constraint after max_retries (unlikely), 
        # force fix the last operator.
        if constraint == 'even_y' and y_count % 2 != 0:
            # Flip last operator to change parity
            last_op = chosen_operators[-1]
            if last_op == 'Y':
                chosen_operators[-1] = 'X' # Y -> X (Odd -> Even)
            else:
                chosen_operators[-1] = 'Y' # Not Y -> Y (Even -> Odd)

    current_array = poly_array.copy()
    
    # For p=2, logic is simple sum. For p>2, it's sum of powers.
    # Original code was generic. We keep it generic.
    
    # Applying the standard random Pauli string logic:
    # Projector P = I + phase * PauliString
    # To implement this for stabilizer states:
    # We are usually finding the result of (I + P)|psi>.
    # If |psi> is stabilizer state, (I+P)|psi> is EITHER 0 OR a stabilizer state.
    # The current implementation in this codebase seems to be summing the array + rotated array.
    # Note: simple addition of arrays does NOT generally result in a stabilizer state unless they align.
    # But let's stick to the existing logic structure and just insert the constraint.

    omega = np.exp(2j * np.pi / p) # Phase factor base
    
    # ... Wait, the original code had a specific loop for k in range(p-1). 
    # That loop implements the projection sum: |psi> + omega*P|psi> + ...
    # We should preserve that inner logic.
    
    # Let's just apply the operators we chose.

    # Calculate P|psi>, P^2|psi> ...
    
    # Actually, looking at original code, it does:
    # poly_array += omega**(k+1)*current_array
    # This implies it sums |psi> + w*P|psi> ... which is the projector.
    
    # We rebuild the array update using the CHOSEN operators
    
    # Reset current_array for calculation
    current_array = poly_array.copy()
    
    for k in range(p-1):
        # Apply the full string P once to get P^(k+1)|psi> from P^k|psi>
        # (Wait, no, the loop `for i in range(n)` applies the string P to `current_array`)
        
        # Apply string P to current_array
        for i in range(n):
            op_name = chosen_operators[i]
            op_function = operator_functions[op_name]
            current_array = op_function(current_array, i, n, p)
            
        # Add to accumulation (poly_array is the accumulator initialized with |psi>)
        # Note: The original logic modifies poly_array in place.
        # We need to be careful. The original code was:
        # poly_array += omega**(k+1)*current_array
        # This adds P|psi>, then P^2|psi>...
        
        # YES.
        poly_array += omega**(k+1)*current_array
        
    return poly_array, chosen_operators

def generate_random_stabilizer_state(m: int, p: int = 3, seed=None) -> np.ndarray:
    """
    Generates a uniformly random stabilizer state vector of size p^m.
    Leverages the Numba-compiled _fill_one_output_on_n_orig_space to evaluate
    a random quadratic phase polynomial over a random affine subspace.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    # 1. Pick a random subspace dimension k in [0, m]
    k_dim = random.randint(0, m)
    
    # 2. Generate a random translation vector x0
    x0_translation = np.random.randint(0, p, size=m, dtype=np.int64)
    
    # 3. Generate a random full-rank generator matrix W of size (k_dim, m)
    W_basis = np.zeros((k_dim, m), dtype=np.int64)
    if k_dim > 0:
        W_base = np.zeros((k_dim, m), dtype=np.int64)
        W_base[:, :k_dim] = np.eye(k_dim, dtype=np.int64)
        if m > k_dim:
            W_base[:, k_dim:] = np.random.randint(0, p, size=(k_dim, m - k_dim))
        
        col_indices = np.random.permutation(m)
        W_basis = W_base[:, col_indices]
        
    # 4. Generate random polynomial coefficients
    alpha_val = 0.0
    c_j0_lin = np.random.randint(0, p, size=k_dim, dtype=np.int64) if k_dim > 0 else np.array([], dtype=np.int64)
    
    num_mixed = k_dim * (k_dim - 1) // 2 if k_dim >= 2 else 0
    c_j0_qm = np.random.randint(0, p, size=num_mixed, dtype=np.int64) if num_mixed > 0 else np.array([], dtype=np.int64)
    
    c_j0_qs = np.zeros(k_dim, dtype=np.int64)
    c_j1_lin = np.zeros(k_dim, dtype=np.int64)
    
    if p >= 3 and k_dim > 0:
        c_j0_qs = np.random.randint(0, p, size=k_dim, dtype=np.int64)
    elif p == 2 and k_dim > 0:
        c_j1_lin = np.random.randint(0, 2, size=k_dim, dtype=np.int64)
        
    # 5. Build y_params_list_for_numba
    y_params_list = List()
    if k_dim > 0:
        y_tuples = list(itertools.product(range(p), repeat=k_dim))
        for y in y_tuples:
            y_params_list.append(np.array(y, dtype=np.int64))
    else:
        y_params_list.append(np.array([], dtype=np.int64))
        
    # 6. Evaluate polynomial over subspace
    output_len = p**m if m > 0 else 1
    current_output = np.zeros(output_len, dtype=complex)
    
    _fill_one_output_on_n_orig_space(
        current_output, y_params_list, m, p,
        x0_translation, W_basis, k_dim, float(alpha_val),
        c_j0_lin, c_j0_qm, c_j0_qs, c_j1_lin
    )
    
    # 7. Normalize the state vector
    norm = np.linalg.norm(current_output)
    if norm > 1e-12:
        current_output /= norm
        
    return current_output
