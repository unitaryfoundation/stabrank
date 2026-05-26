"""Linear algebra utilities for stabilizer rank computations."""

import numpy as np
from numba import njit, types
from numba.typed import List

@njit(cache=True)
def get_lex_index(point_x_orig, n_orig, p_prime):
    """Convert point coordinates to lexicographic index."""
    idx = 0
    if n_orig == 0: 
        return 0
    for i in range(n_orig):
        idx = idx * p_prime + point_x_orig[i]
    return idx

@njit(cache=True)
def get_modular_inverse(value, p_prime):
    """Compute modular inverse using Fermat's little theorem."""
    if value == 0:
        raise ValueError("Cannot compute inverse of 0.")
    base = int(value)
    exponent = int(p_prime - 2)
    modulus = int(p_prime)
    if modulus == 1: 
        return 0
    res = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            res = (res * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return res

@njit(cache=True)
def parametrize_affine_subspace_Ax_eq_b_numba_core(A_orig, b_orig, p_prime):
    """Core function for parametrizing affine subspace Ax = b (mod p_prime)."""
    m_orig_shape = A_orig.shape[0]
    if A_orig.ndim == 1 and A_orig.shape[0] == 0:
        A = np.empty((0,0), dtype=np.int64)
        n_orig = 0
    elif A_orig.ndim == 2:
        A = A_orig.copy().astype(np.int64)
        n_orig = A_orig.shape[1]
    else:
        A = np.empty((0,0), dtype=np.int64)
        n_orig = 0
    
    num_rows = A.shape[0]
    b_reshaped = b_orig.copy().astype(np.int64).reshape(m_orig_shape, 1)

    if n_orig == 0:
        is_consistent = True
        for r_idx in range(b_reshaped.shape[0]):
            if (b_reshaped[r_idx, 0] % p_prime) != 0:
                is_consistent = False
                break
        if is_consistent:
            return np.array([], dtype=np.int64), np.empty((0,0), dtype=np.int64), 0, True
        else:
            return np.array([], dtype=np.int64), np.empty((0,0), dtype=np.int64), 0, False

    aug_matrix = np.hstack((A, b_reshaped)) % p_prime
    num_vars = n_orig
    pivot_row_current = 0
    pivot_cols_indices_list = List.empty_list(types.int64)

    for col in range(num_vars):
        if pivot_row_current == num_rows: 
            break
        i = pivot_row_current
        while i < num_rows and aug_matrix[i, col] == 0:
            i += 1
        if i < num_rows:
            aug_matrix_row_pivot_current = aug_matrix[pivot_row_current,:].copy()
            aug_matrix_row_i = aug_matrix[i,:].copy()
            aug_matrix[pivot_row_current,:] = aug_matrix_row_i
            aug_matrix[i,:] = aug_matrix_row_pivot_current
            
            inv_pivot = get_modular_inverse(int(aug_matrix[pivot_row_current, col]), p_prime)
            aug_matrix[pivot_row_current, :] = (aug_matrix[pivot_row_current, :] * inv_pivot) % p_prime
            for r_idx in range(num_rows):
                if r_idx != pivot_row_current:
                    factor = aug_matrix[r_idx, col]
                    aug_matrix[r_idx, :] = (aug_matrix[r_idx, :] - aug_matrix[pivot_row_current, :] * factor) % p_prime
            pivot_cols_indices_list.append(col)
            pivot_row_current += 1
    
    rank = len(pivot_cols_indices_list)

    for r_check in range(rank, num_rows):
        if aug_matrix[r_check, num_vars] != 0:
            return np.empty((num_vars,), dtype=np.int64), \
                   np.empty((0, num_vars), dtype=np.int64), 0, False

    x0_particular = np.zeros(num_vars, dtype=np.int64)
    pivot_cols_indices_arr = np.array(pivot_cols_indices_list, dtype=np.int64)
    
    for r_idx in range(rank):
        pivot_col = pivot_cols_indices_arr[r_idx]
        x0_particular[pivot_col] = aug_matrix[r_idx, num_vars]

    all_col_indices_arr = np.arange(num_vars)
    free_col_indices_arr = np.setdiff1d(all_col_indices_arr, pivot_cols_indices_arr)
    
    k_dimension = len(free_col_indices_arr)
    W_basis_np = np.empty((k_dimension, num_vars), dtype=np.int64)
    for i, free_col_idx_val in enumerate(free_col_indices_arr):
        basis_vec = np.zeros(num_vars, dtype=np.int64)
        basis_vec[free_col_idx_val] = 1
        for r_idx in range(rank):
            pivot_col = pivot_cols_indices_arr[r_idx]
            basis_vec[pivot_col] = (-aug_matrix[r_idx, free_col_idx_val]) % p_prime
        W_basis_np[i,:] = basis_vec
        
    return x0_particular, W_basis_np, k_dimension, True

def parametrize_affine_subspace_Ax_eq_b(A_orig, b_orig, p_prime):
    """
    Parametrize the affine subspace defined by Ax = b (mod p_prime).
    
    Returns:
        x0_particular: Particular solution
        W_basis: Basis for the null space  
        k_dimension: Dimension of the null space
        is_consistent: Whether the system is consistent
    """
    if not isinstance(A_orig, np.ndarray) or not isinstance(b_orig, np.ndarray):
        raise TypeError("Inputs A and b must be NumPy arrays.")
    if A_orig.ndim > 2 or (A_orig.ndim == 1 and A_orig.shape != (0,)):
         raise ValueError("A_matrix must be a 2D NumPy array or an empty 1D array for 0x0 case.")
    if b_orig.ndim != 1:
        raise ValueError("b_vector must be a 1D NumPy array.")
    
    m_orig = A_orig.shape[0] if A_orig.ndim == 2 else 0
    if b_orig.shape[0] != m_orig:
        raise ValueError(f"A_matrix rows ({m_orig}) must match b_vector length ({b_orig.shape[0]}).")

    return parametrize_affine_subspace_Ax_eq_b_numba_core(A_orig, b_orig, p_prime)

def can_represent_as_linear_combination(target_function, list_of_basis_functions, rtol=1e-5, atol=1e-8):
    """
    Check if target function can be represented as linear combination of basis functions.
    
    Returns:
        is_representable: Boolean indicating if representation exists
        coeffs: Coefficients of the linear combination
        reconstruction_error: L2 norm of the reconstruction error
    """
    if not list_of_basis_functions:
        is_target_zero = np.allclose(target_function, 0, atol=atol) if target_function.size > 0 else True
        if is_target_zero: 
            return True, np.array([], dtype=complex), 0.0
        return False, None, np.linalg.norm(target_function) if target_function.size > 0 else 0.0
    
    M = np.array(list_of_basis_functions).T
    try:
        coeffs, residuals, rank, s_values = np.linalg.lstsq(M, target_function, rcond=None)
    except np.linalg.LinAlgError as e:
        if M.shape[1] == 0 and np.allclose(target_function, 0, atol=atol):
            return True, np.array([], dtype=complex), 0.0
        raise e

    reconstructed_target = M @ coeffs
    is_representable = np.allclose(target_function, reconstructed_target, rtol=rtol, atol=atol)
    reconstruction_error = np.linalg.norm(target_function - reconstructed_target) if target_function.size > 0 else 0.0
    
    return (True, coeffs, reconstruction_error) if is_representable else (False, coeffs, reconstruction_error)


def row_reduce_finite_field(matrix, p):
    """Computes the row-reduced echelon form of a matrix over a finite field."""
    M = matrix.copy()
    num_rows, num_cols = M.shape
    pivot_row = 0
    for j in range(num_cols): # Iterate through columns
        if pivot_row == num_rows:
            break
        
        # Find a row with a non-zero entry in the current column
        i = pivot_row
        while i < num_rows and M[i, j] == 0:
            i += 1
        
        if i < num_rows:
            # Swap rows to bring pivot to the top
            M[[pivot_row, i]] = M[[i, pivot_row]]
            
            # Normalize the pivot row
            inv = get_modular_inverse(M[pivot_row, j], p)
            M[pivot_row, :] = (M[pivot_row, :] * inv) % p
            
            # Eliminate other non-zero entries in this column
            for i in range(num_rows):
                if i != pivot_row:
                    factor = M[i, j]
                    M[i, :] = (M[i, :] - factor * M[pivot_row, :]) % p
            
            pivot_row += 1
    return M
