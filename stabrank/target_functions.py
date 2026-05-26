"""Target functions for stabilizer rank computations."""

import numpy as np
import itertools

def qutrit_hadamard_eigenstate(n: int, p_prime: int = 3) -> np.ndarray:
    """
    Computes the tensor product of the +1 eigenvector of the qutrit Hadamard gate.
    |H+> = (1/N) * [1, (sqrt(3)-1)/2, (sqrt(3)-1)/2]^T
    
    This state has purely real amplitudes.
    """
    if p_prime != 3:
        raise ValueError("Hadamard eigenstate defined for qutrits (p=3) only.")
    
    # Define the single-qutrit state vector
    # Unnormalized components:
    c0 = 1.0
    c1 = (np.sqrt(3) - 1) / 2.0
    c2 = c1 # Same as c1
    
    psi_single = np.array([c0, c1, c2], dtype=complex)
    
    # Normalize it
    norm = np.linalg.norm(psi_single)
    psi_single = psi_single / norm
    
    # Compute the n-fold tensor product |psi>^n
    # We can do this efficiently by iterating or using Kronecker products
    if n == 0:
        return np.array([1.0], dtype=complex)
    
    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)
        
    return current_state

def qutrit_strange_state(n: int, p_prime: int = 3) -> np.ndarray:
    """
    Computes the tensor product of the Strange state |S>.
    The single qutrit strange state is the +i eigenstate of the qutrit Hadamard gate,
    possessing maximal Wigner negativity (Mana) and distinguished as a simultaneous
    eigenvector of all symplectic rotations.
    
    |S> = 1/sqrt(2) * (|1> - |2>)
    """
    if p_prime != 3:
        raise ValueError("Strange state is defined for qutrits (p=3) only.")
        
    if n == 0:
        return np.array([1.0], dtype=complex)
        
    psi_single = np.array([0.0, 1.0/np.sqrt(2), -1.0/np.sqrt(2)], dtype=complex)
    
    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)
        
    return current_state

def qutrit_norrell_state(n: int, p_prime: int = 3) -> np.ndarray:
    """
    Computes the tensor product of the Norrell state |N+>.
    The Norrell state has maximal Mana (equal to the Strange state, log_3(5/3)),
    but resides in a distinct Clifford orbit with orbit size 36 (vs 9 for Strange).
    Reconstructed from its discrete Wigner function given in Jain & Prakash (2020) eq. 3.45.
    
    |N+> = 1/sqrt(6) * (|0> + |1> - 2|2>)
    """
    if p_prime != 3:
        raise ValueError("Norrell state is defined for qutrits (p=3) only.")
        
    if n == 0:
        return np.array([1.0], dtype=complex)
        
    psi_single = np.array([1.0, 1.0, -2.0], dtype=complex) / np.sqrt(6)
    
    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)
        
    return current_state

def qutrit_complex_magic_state(n: int, p_prime: int = 3) -> np.ndarray:
    """
    Computes the tensor product of the Complex qutrit magic state |T3>.
    The single qutrit complex magic state is used for standard phase-hierarchy state injection.
    
    |T3> = 1/sqrt(3) * (|0> + e^{2*pi*i/9}|1> + e^{4*pi*i/9}|2>)
    """
    if p_prime != 3:
        raise ValueError("Complex magic state is defined for qutrits (p=3) only.")
        
    if n == 0:
        return np.array([1.0], dtype=complex)
        
    xi = np.exp(2j * np.pi / 9)
    psi_single = np.array([1.0, xi, xi**2], dtype=complex) / np.sqrt(3)
    
    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)
        
    return current_state

def qubit_magic_phase_sum(n: int) -> np.ndarray:
    """
    Computes the n-qubit T-state phase-sum vector.

    The amplitude on x in {0,1}^n is exp(i*pi*|x|/4) / sqrt(2^n).
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be non-negative.")

    output = np.empty(2**n, dtype=complex)
    norm = np.sqrt(2**n)
    for idx, x_tuple in enumerate(itertools.product([0, 1], repeat=n)):
        output[idx] = np.exp(1j * np.pi * sum(x_tuple) / 4.0) / norm
    return output


def qubit_magic_phase_sum_constrained(num_free_variables: int) -> np.ndarray:
    """
    Computes the parity-constrained qubit T-state phase-sum vector.

    This is the compressed repetition-code target obtained by projecting
    |T>^(m) onto the even-parity subspace, where m = num_free_variables + 1.
    """
    if not isinstance(num_free_variables, int) or num_free_variables < 0:
        raise ValueError("num_free_variables must be non-negative.")

    m = num_free_variables + 1
    T_m = qubit_magic_phase_sum(m)
    output = np.zeros(2**num_free_variables, dtype=complex)

    for idx, x_tuple in enumerate(itertools.product([0, 1], repeat=num_free_variables)):
        last_bit = sum(x_tuple) % 2
        full_idx = 0
        for bit in (*x_tuple, last_bit):
            full_idx = (full_idx << 1) | bit
        output[idx] = T_m[full_idx] * np.sqrt(2)

    return output


def qutrit_magic_phase_sum(n: int, p_prime: int = 3) -> np.ndarray:
    """Alias for the standard qutrit complex magic-state phase-sum vector."""
    return qutrit_complex_magic_state(n, p_prime=p_prime)

def qutrit_magic_phase_sum_constrained(num_free_variables: int, p_prime: int = 3) -> np.ndarray:
    r"""
    Computes the compressed qutrit magic code state for the k=1 repetition code.
    Constructs the state by projecting |T3>^{\otimes m} onto the zero-sum subspace.
    Here m = num_free_variables + 1.
    """
    import itertools
    if p_prime != 3:
        raise ValueError("Function is specific to p_prime=3.")
        
    m = num_free_variables + 1
    T_m = qutrit_complex_magic_state(m)
    
    output = np.zeros(3**num_free_variables, dtype=complex)
    
    for idx, x_tuple in enumerate(itertools.product(range(3), repeat=num_free_variables)):
        sum_x = sum(x_tuple)
        last_x = (-sum_x % 3 + 3) % 3
        
        full_idx = 0
        for i, val in enumerate(x_tuple):
            full_idx += val * (3**(m - 1 - i))
        full_idx += last_x
        
        output[idx] = T_m[full_idx] * np.sqrt(3)
        
    return output


def qubit_hadamard_eigenstate(n: int) -> np.ndarray:
    """
    Computes the tensor product of the +1 eigenvector of the Qubit Hadamard gate.
    |H+> = (1/N) * [1, sqrt(2)-1]^T.
    
    This state has purely real amplitudes.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be non-negative.")

    # Define the single-qubit state vector
    # Unnormalized components:
    c0 = 1.0
    c1 = np.sqrt(2) - 1.0
    
    psi_single = np.array([c0, c1], dtype=complex)
    
    # Normalize it
    norm = np.linalg.norm(psi_single)
    psi_single = psi_single / norm
    
    # Compute the n-fold tensor product |psi>^n
    if n == 0:
        return np.array([1.0], dtype=complex)
    
    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)
        
    return current_state

def qubit_t_type_magic_state(n: int) -> np.ndarray:
    """
    Computes the tensor product of the qubit T-type magic state.
    |T> = cos(beta)|0> + e^{i pi/4} sin(beta)|1>,  cos(2 beta) = 1/sqrt(3).

    The T-type state sits at a face-center of the stabilizer octahedron on the
    Bloch sphere, with Bloch vector (1,1,1)/sqrt(3). Its Clifford orbit has
    8 elements (vs 12 for the H-type). Introduced in Bravyi & Kitaev (2005),
    it is the second Clifford-inequivalent family of qubit magic states.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be non-negative.")

    beta = np.arccos(1.0 / np.sqrt(3)) / 2.0
    psi_single = np.array(
        [np.cos(beta), np.exp(1j * np.pi / 4) * np.sin(beta)], dtype=complex
    )

    if n == 0:
        return np.array([1.0], dtype=complex)

    current_state = psi_single
    for _ in range(n - 1):
        current_state = np.kron(current_state, psi_single)

    return current_state

def qubit_magic_code_state(G: np.ndarray) -> np.ndarray:
    r"""
    Computes the Magic Code State |C_m> for a binary linear code C with generator matrix G.
    |C_m> = (1/sqrt(|C|)) \sum_{c \in C} Z^c |T>^m
    
    Args:
        G: k x m binary numpy array representing the generator matrix of the code.
        
    Returns:
        np.ndarray of size 2^m representing the state vector in the computational basis.
    """
    if not isinstance(G, np.ndarray) or G.ndim != 2:
        raise ValueError("G must be a 2D numpy array.")
        
    k, m = G.shape
    state_vector = np.zeros(2**m, dtype=complex)
    
    # Amplitude normalization for x in the dual code C^perp
    # Based on the identity: |C_m> = 2^{(k-m)/2} \sum_{x \in C^\perp} e^{i \pi |x| / 4} |x>
    amp = 2**((k - m) / 2.0)
    
    # Iterate over all 2^m computational basis states
    for x_tuple in itertools.product([0, 1], repeat=m):
        x = np.array(x_tuple)
        
        # Check if x is in the dual code C^perp (i.e., G * x = 0 mod 2)
        if np.all((G @ x) % 2 == 0):
            # Compute integer index for x (standard kronecker order puts x_0 as MSB)
            idx = 0
            for bit in x_tuple:
                idx = (idx << 1) | bit
            
            # Phase is exp(i * (pi/4) * hamming_weight(x))
            hw = np.sum(x)
            state_vector[idx] = amp * np.exp(1j * np.pi * hw / 4.0)
            
    return state_vector

def qutrit_magic_code_state(G: np.ndarray) -> np.ndarray:
    r"""
    Computes the Qutrit Magic Code State |C_m^{(3)}> for a linear code C over F_3 with generator matrix G.
    |C_m^{(3)}> = (1/sqrt(3^k)) \sum_{c \in C} Z^c |T_3>^m
    
    Args:
        G: k x m ternary numpy array (elements in {0, 1, 2}) representing the generator matrix.
        
    Returns:
        np.ndarray of size 3^m representing the state vector in the computational basis.
    """
    if not isinstance(G, np.ndarray) or G.ndim != 2:
        raise ValueError("G must be a 2D numpy array.")
        
    k, m = G.shape
    state_vector = np.zeros(3**m, dtype=complex)
    
    # Amplitude normalization for x in the dual code C^perp over F_3
    # Based on the identity: |C_m^{(3)}> = 3^{(k-m)/2} \sum_{x \in C^\perp} e^{i (2\pi/9) |x|_1} |x>
    amp = 3**((k - m) / 2.0)
    
    # Iterate over all 3^m computational basis states
    for x_tuple in itertools.product([0, 1, 2], repeat=m):
        x = np.array(x_tuple)
        
        # Check if x is in the dual code C^perp over F_3 (i.e., G * x = 0 mod 3)
        if np.all((G @ x) % 3 == 0):
            # Compute integer index for x (standard kronecker order puts x_0 as MSB)
            # For base 3, idx = idx * 3 + bit
            idx = 0
            for trit in x_tuple:
                idx = (idx * 3) + trit
            
            # Phase is exp(i * (2*pi/9) * |x|_1)
            hw = np.sum(x)
            state_vector[idx] = amp * np.exp(2j * np.pi * hw / 9.0)
            
    return state_vector

def qubit_magic_code_state_compressed(G: np.ndarray) -> np.ndarray:
    r"""
    Computes the Clifford-compressed Qubit Magic Code State |\phi> on m-k qubits.
    The original state |C_m> is supported on the dual code C^\perp.
    By applying a CNOT circuit U, we map |C_m> to |0>^k \otimes |\phi>.
    This returns the state vector |\phi> of size 2^{m-k}.
    """
    if not isinstance(G, np.ndarray) or G.ndim != 2:
        raise ValueError("G must be a 2D numpy array.")
        
    k, m = G.shape
    if k >= m:
        raise ValueError("Code dimension k must be less than length m.")
        
    A = G[:, k:]
    
    state_vector = np.zeros(2**(m-k), dtype=complex)
    amp = 2**((k - m) / 2.0)
    
    for y_tuple in itertools.product([0, 1], repeat=m-k):
        y = np.array(y_tuple, dtype=int)
        
        # x is the corresponding vector in C^\perp
        # x = [-A y, y]^T mod 2
        # Since we are in F_2, -A = A
        x_first = (A @ y) % 2
        x = np.concatenate([x_first, y])
        
        # Index in the compressed state is just y
        idx = 0
        for bit in y_tuple:
            idx = (idx << 1) | bit
            
        hw = np.sum(x)
        state_vector[idx] = amp * np.exp(1j * np.pi * hw / 4.0)
        
    return state_vector

def qutrit_magic_code_state_compressed(G: np.ndarray) -> np.ndarray:
    r"""
    Computes the Clifford-compressed Qutrit Magic Code State |\phi> on m-k qutrits.
    The original state |C_m^{(3)}> is supported on the dual code C^\perp over F_3.
    By applying a SUM circuit U, we map it to |0>^k \otimes |\phi>.
    This returns the state vector |\phi> of size 3^{m-k}.
    """
    if not isinstance(G, np.ndarray) or G.ndim != 2:
        raise ValueError("G must be a 2D numpy array.")
        
    k, m = G.shape
    if k >= m:
        raise ValueError("Code dimension k must be less than length m.")
        
    A = G[:, k:]
    
    # Generate the full m-qutrit T3 state
    T_m = qutrit_complex_magic_state(m)
    
    state_vector = np.zeros(3**(m-k), dtype=complex)
    
    # The amplitude scalar due to projection is 3^(k/2).
    # |C_m> = 3^{(k-m)/2} \sum_{x \in C^\perp} e^{i 2\pi |x|_1 / 9} |x>
    # T_m[x] = 3^{-m/2} e^{i 2\pi |x|_1 / 9}
    # So state_vector[y] = T_m[x] * 3^{k/2}
    amp_scale = 3**(k / 2.0)
    
    for y_tuple in itertools.product([0, 1, 2], repeat=m-k):
        y = np.array(y_tuple, dtype=int)
        
        # x is the corresponding vector in C^\perp
        # x = [-A y, y]^T mod 3
        x_first = (-A @ y) % 3
        x = np.concatenate([x_first, y])
        
        # Index in the compressed state is just y
        idx = 0
        for trit in y_tuple:
            idx = (idx * 3) + trit
            
        # Index in the full T_m state is x
        full_idx = 0
        for trit in x:
            full_idx = (full_idx * 3) + trit
            
        state_vector[idx] = T_m[full_idx] * amp_scale
        
    return state_vector
