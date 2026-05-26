"""StabRank: stabilizer-rank search utilities and target states."""

__version__ = "0.1.0"
__author__ = "Farrokh Labib"

from .analysis_utils import (
    calculate_sa_cost,
    prune_least_significant_basis_function,
)
from .linalg_utils import (
    can_represent_as_linear_combination,
    get_lex_index,
    get_modular_inverse,
    parametrize_affine_subspace_Ax_eq_b,
)
from .polynomial_utils import (
    apply_random_pauli_string,
    evaluate_coeffs_on_subspace,
    evaluate_poly_string_on_subspace,
    format_polynomial_string,
    generate_distinct_phases_on_param_subspace_v2,
    generate_random_poly_coeffs,
    generate_random_stabilizer_state,
)
from .target_functions import (
    qubit_hadamard_eigenstate,
    qubit_t_type_magic_state,
    qubit_magic_code_state,
    qubit_magic_code_state_compressed,
    qubit_magic_phase_sum,
    qubit_magic_phase_sum_constrained,
    qutrit_complex_magic_state,
    qutrit_hadamard_eigenstate,
    qutrit_magic_code_state,
    qutrit_magic_code_state_compressed,
    qutrit_magic_phase_sum,
    qutrit_magic_phase_sum_constrained,
    qutrit_norrell_state,
    qutrit_strange_state,
)

__all__ = [
    "apply_random_pauli_string",
    "calculate_sa_cost",
    "can_represent_as_linear_combination",
    "evaluate_coeffs_on_subspace",
    "evaluate_poly_string_on_subspace",
    "format_polynomial_string",
    "generate_distinct_phases_on_param_subspace_v2",
    "generate_random_poly_coeffs",
    "generate_random_stabilizer_state",
    "get_lex_index",
    "get_modular_inverse",
    "parametrize_affine_subspace_Ax_eq_b",
    "prune_least_significant_basis_function",
    "qubit_hadamard_eigenstate",
    "qubit_t_type_magic_state",
    "qubit_magic_code_state",
    "qubit_magic_code_state_compressed",
    "qubit_magic_phase_sum",
    "qubit_magic_phase_sum_constrained",
    "qutrit_complex_magic_state",
    "qutrit_hadamard_eigenstate",
    "qutrit_magic_code_state",
    "qutrit_magic_code_state_compressed",
    "qutrit_magic_phase_sum",
    "qutrit_magic_phase_sum_constrained",
    "qutrit_norrell_state",
    "qutrit_strange_state",
]
