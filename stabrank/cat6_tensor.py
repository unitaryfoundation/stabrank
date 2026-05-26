"""Tensor-product constructions and candidate searches for qutrit cat6."""

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .target_functions import qutrit_magic_code_state

P_QUTRIT = 3
CAT6_LENGTH = 6


@dataclass(frozen=True)
class DecompositionResult:
    """Least-squares certificate for a stabilizer decomposition."""

    basis: list[np.ndarray]
    coefficients: np.ndarray
    residual: float

    @property
    def basis_count(self) -> int:
        return len(self.basis)


@dataclass(frozen=True)
class BeamSearchResult:
    """Best subset found by deterministic beam search."""

    subset: tuple[int, ...]
    residual: float
    coefficients: np.ndarray
    candidates: int
    beam_width: int
    exact: bool


def trits_to_index(trits: Sequence[int], p: int = P_QUTRIT) -> int:
    """Convert a base-p tuple to a lexicographic integer index."""

    index = 0
    for value in trits:
        index = index * p + int(value)
    return index


def index_to_trits(index: int, n: int, p: int = P_QUTRIT) -> tuple[int, ...]:
    """Convert a lexicographic integer index to a base-p tuple."""

    trits = [0] * n
    for pos in range(n - 1, -1, -1):
        trits[pos] = index % p
        index //= p
    return tuple(trits)


def cat6_target() -> np.ndarray:
    """Return the normalized uncompressed qutrit repetition-code cat6 target."""

    generator = np.ones((1, CAT6_LENGTH), dtype=int)
    target = qutrit_magic_code_state(generator)
    return target / np.linalg.norm(target)


def shifted_cat_state(n: int, residue: int) -> np.ndarray:
    """Return normalized phi_n(residue) on n qutrits."""

    omega9 = np.exp(2j * np.pi / 9)
    state = np.zeros(P_QUTRIT**n, dtype=complex)
    for trits in itertools.product(range(P_QUTRIT), repeat=n):
        if sum(trits) % P_QUTRIT == residue:
            state[trits_to_index(trits)] = omega9 ** sum(trits)
    return state / np.linalg.norm(state)


def compressed_shifted_cat4_state(residue: int) -> np.ndarray:
    """Return phi_4(residue), Clifford-compressed to three qutrits."""

    omega9 = np.exp(2j * np.pi / 9)
    state = np.empty(P_QUTRIT**3, dtype=complex)
    for index, y in enumerate(itertools.product(range(P_QUTRIT), repeat=3)):
        last = (residue - sum(y)) % P_QUTRIT
        state[index] = omega9 ** (sum(y) + last)
    return state / np.linalg.norm(state)


def stabilizer_state_from_params(
    n: int,
    x0: Sequence[int],
    w_basis: Sequence[Sequence[int]],
    c_lin: Sequence[int],
    c_sq: Sequence[int],
    c_mix: Sequence[Sequence[int]],
) -> np.ndarray:
    """Build a normalized qutrit stabilizer state from affine/quadratic data."""

    x0_arr = np.array(x0, dtype=int)
    w_arr = np.array(w_basis, dtype=int)
    c_lin_arr = np.array(c_lin, dtype=int)
    c_sq_arr = np.array(c_sq, dtype=int)
    c_mix_arr = np.array(c_mix, dtype=int)
    k_dim = w_arr.shape[0]

    state = np.zeros(P_QUTRIT**n, dtype=complex)
    omega = np.exp(2j * np.pi / P_QUTRIT)
    for y in itertools.product(range(P_QUTRIT), repeat=k_dim):
        y_arr = np.array(y, dtype=int)
        x = (x0_arr + y_arr @ w_arr) % P_QUTRIT
        q_val = int(np.dot(c_lin_arr, y_arr) + np.dot(c_sq_arr, y_arr * y_arr))
        for left in range(k_dim):
            for right in range(left + 1, k_dim):
                q_val += int(c_mix_arr[left, right] * y_arr[left] * y_arr[right])
        state[trits_to_index(x)] = omega ** (q_val % P_QUTRIT)
    return state / np.linalg.norm(state)


SHIFTED_CAT4_DECOMPOSITIONS = {
    0: [
        {
            "x0": [0, 0, 1],
            "w": [[0, 1, 2], [1, 0, 0]],
            "lin": [1, 0],
            "sq": [2, 0],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 0, 0],
            "w": [[0, 1, 2], [1, 0, 0]],
            "lin": [0, 0],
            "sq": [1, 1],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 0, 2],
            "w": [[0, 1, 2], [1, 0, 0]],
            "lin": [0, 1],
            "sq": [0, 2],
            "mix": [[0, 0], [0, 0]],
        },
    ],
    1: [
        {
            "x0": [0, 2, 0],
            "w": [[0, 0, 1], [1, 2, 0]],
            "lin": [0, 0],
            "sq": [0, 0],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 0, 0],
            "w": [[0, 0, 1], [1, 2, 0]],
            "lin": [1, 0],
            "sq": [2, 1],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 1, 0],
            "w": [[0, 0, 1], [1, 2, 0]],
            "lin": [0, 1],
            "sq": [1, 2],
            "mix": [[0, 0], [0, 0]],
        },
    ],
    2: [
        {
            "x0": [0, 0, 0],
            "w": [[0, 1, 0], [1, 0, 2]],
            "lin": [0, 0],
            "sq": [0, 1],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 0, 1],
            "w": [[0, 1, 0], [1, 0, 2]],
            "lin": [1, 1],
            "sq": [2, 2],
            "mix": [[0, 0], [0, 0]],
        },
        {
            "x0": [0, 0, 2],
            "w": [[0, 1, 0], [1, 0, 2]],
            "lin": [0, 0],
            "sq": [1, 0],
            "mix": [[0, 0], [0, 0]],
        },
    ],
}


def shifted_cat4_basis(residue: int) -> list[np.ndarray]:
    """Return a fixed rank-3 stabilizer basis for compressed phi_4(residue)."""

    return [
        stabilizer_state_from_params(
            n=3,
            x0=params["x0"],
            w_basis=params["w"],
            c_lin=params["lin"],
            c_sq=params["sq"],
            c_mix=params["mix"],
        )
        for params in SHIFTED_CAT4_DECOMPOSITIONS[residue]
    ]


def embed_compressed_phi4_state(state: np.ndarray, residue: int) -> np.ndarray:
    """Embed a compressed 3-qutrit phi_4 state into its 4-qutrit coset."""

    embedded = np.zeros(P_QUTRIT**4, dtype=complex)
    for index, value in enumerate(state):
        y = index_to_trits(index, 3)
        last = (residue - sum(y)) % P_QUTRIT
        embedded[trits_to_index((*y, last))] = value
    return embedded / np.linalg.norm(embedded)


def solve_decomposition(target: np.ndarray, basis: Sequence[np.ndarray]) -> DecompositionResult:
    """Fit coefficients for target in the span of basis."""

    matrix = np.column_stack(basis)
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        matrix,
        target,
        rcond=None,
    )
    residual = float(np.linalg.norm(target - matrix @ coefficients))
    return DecompositionResult(list(basis), coefficients, residual)


def product_state_on_positions(
    state_a: np.ndarray,
    positions_a: Sequence[int],
    state_b: np.ndarray,
    positions_b: Sequence[int],
) -> np.ndarray:
    """Tensor product state embedded into the requested cat6 positions."""

    full = np.zeros(P_QUTRIT**CAT6_LENGTH, dtype=complex)
    for full_index in range(full.size):
        trits = index_to_trits(full_index, CAT6_LENGTH)
        a_index = trits_to_index([trits[pos] for pos in positions_a])
        b_index = trits_to_index([trits[pos] for pos in positions_b])
        full[full_index] = state_a[a_index] * state_b[b_index]
    return full


def cat6_a2b4_basis(positions_a: Sequence[int] = (0, 1)) -> list[np.ndarray]:
    """Construct the 9-state A2/B4 tensor-product cat6 basis."""

    positions_a = tuple(positions_a)
    if len(positions_a) != 2:
        raise ValueError("positions_a must contain exactly two positions")
    if len(set(positions_a)) != 2 or any(pos < 0 or pos >= CAT6_LENGTH for pos in positions_a):
        raise ValueError("positions_a must be distinct cat6 positions")

    positions_b = tuple(pos for pos in range(CAT6_LENGTH) if pos not in positions_a)
    basis = []
    for residue_a in range(P_QUTRIT):
        residue_b = (-residue_a) % P_QUTRIT
        state_a = shifted_cat_state(2, residue_a)
        for compressed_b in shifted_cat4_basis(residue_b):
            state_b = embed_compressed_phi4_state(compressed_b, residue_b)
            basis.append(product_state_on_positions(state_a, positions_a, state_b, positions_b))
    return basis


def cat6_a2b4_decomposition(positions_a: Sequence[int] = (0, 1)) -> DecompositionResult:
    """Return the exact 9-state A2/B4 cat6 decomposition certificate."""

    return solve_decomposition(cat6_target(), cat6_a2b4_basis(positions_a))


def normalize_global_phase(state: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Canonicalize a state up to global phase for deduplication."""

    normalized = state / np.linalg.norm(state)
    nonzero = np.flatnonzero(np.abs(normalized) > atol)
    if nonzero.size == 0:
        return normalized
    phase = normalized[nonzero[0]] / abs(normalized[nonzero[0]])
    return normalized / phase


def deduplicate_states(
    states: Iterable[np.ndarray],
    *,
    decimals: int = 12,
) -> list[np.ndarray]:
    """Deduplicate states up to global phase."""

    seen = set()
    unique = []
    for state in states:
        canonical = normalize_global_phase(state)
        rounded = np.round(canonical.real, decimals) + 1j * np.round(
            canonical.imag,
            decimals,
        )
        key = tuple(zip(rounded.real.tolist(), rounded.imag.tolist()))
        if key not in seen:
            seen.add(key)
            unique.append(state / np.linalg.norm(state))
    return unique


def all_a2b4_candidates() -> list[np.ndarray]:
    """Return all deduplicated states from the 15 A2/B4 cat6 bipartitions."""

    candidates = []
    for positions_a in itertools.combinations(range(CAT6_LENGTH), 2):
        candidates.extend(cat6_a2b4_basis(positions_a))
    return deduplicate_states(candidates)


def score_subset(
    target: np.ndarray,
    candidates: Sequence[np.ndarray],
    subset: Sequence[int],
) -> tuple[float, np.ndarray]:
    """Return least-squares residual and coefficients for a candidate subset."""

    matrix = np.column_stack([candidates[idx] for idx in subset])
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        matrix,
        target,
        rcond=None,
    )
    residual = float(np.linalg.norm(target - matrix @ coefficients))
    return residual, coefficients


def beam_search_rank8_candidate(
    *,
    beam_width: int = 4096,
    early_exit_threshold: float = 1e-10,
) -> BeamSearchResult:
    """Search the multi-bipartition A2/B4 candidate pool for an 8-state fit."""

    candidates = all_a2b4_candidates()
    target = cat6_target()
    beam: list[tuple[float, tuple[int, ...]]] = [(float(np.linalg.norm(target)), ())]
    best_subset: tuple[int, ...] = ()
    best_residual = float(np.linalg.norm(target))
    best_coefficients = np.array([], dtype=complex)

    for size in range(1, 9):
        expanded: dict[tuple[int, ...], float] = {}
        for _residual, subset in beam:
            start = subset[-1] + 1 if subset else 0
            for candidate_idx in range(start, len(candidates)):
                trial = (*subset, candidate_idx)
                residual, _coefficients = score_subset(target, candidates, trial)
                if trial not in expanded or residual < expanded[trial]:
                    expanded[trial] = residual

        ranked = sorted(expanded.items(), key=lambda item: item[1])
        beam = [(residual, subset) for subset, residual in ranked[:beam_width]]
        if beam and beam[0][0] < best_residual:
            best_residual, best_subset = beam[0]
            best_residual, best_coefficients = score_subset(target, candidates, best_subset)
        if size == 8 and best_residual <= early_exit_threshold:
            break

    return BeamSearchResult(
        subset=best_subset,
        residual=best_residual,
        coefficients=best_coefficients,
        candidates=len(candidates),
        beam_width=beam_width,
        exact=best_residual <= early_exit_threshold,
    )
