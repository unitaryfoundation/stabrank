import itertools

import numpy as np
import pytest

from stabrank.target_functions import (
    qubit_magic_code_state,
    qubit_magic_code_state_compressed,
    qutrit_magic_code_state,
    qutrit_magic_code_state_compressed,
)


def _lex_index(values: np.ndarray, p: int) -> int:
    idx = 0
    for value in values:
        idx = idx * p + int(value)
    return idx


@pytest.mark.parametrize(
    ("p", "generator", "compressed_generator", "G"),
    [
        (
            2,
            qubit_magic_code_state,
            qubit_magic_code_state_compressed,
            np.array([[1, 0, 1], [0, 1, 1]], dtype=int),
        ),
        (
            3,
            qutrit_magic_code_state,
            qutrit_magic_code_state_compressed,
            np.array([[1, 0, 1], [0, 1, 2]], dtype=int),
        ),
    ],
)
def test_compressed_magic_code_state_matches_full_state_on_support(
    p, generator, compressed_generator, G
):
    full_state = generator(G)
    compressed_state = compressed_generator(G)

    k, m = G.shape
    A = G[:, k:]
    seen_support = set()

    for y_tuple in itertools.product(range(p), repeat=m - k):
        y = np.array(y_tuple, dtype=int)
        if p == 2:
            x_first = (A @ y) % p
        else:
            x_first = (-A @ y) % p
        x = np.concatenate([x_first, y])

        full_idx = _lex_index(x, p)
        compressed_idx = _lex_index(y, p)
        seen_support.add(full_idx)

        assert full_state[full_idx] == pytest.approx(compressed_state[compressed_idx])

    assert np.linalg.norm(full_state) == pytest.approx(1.0)
    assert np.linalg.norm(compressed_state) == pytest.approx(1.0)

    off_support = [
        idx for idx in range(full_state.size) if idx not in seen_support
    ]
    assert np.allclose(full_state[off_support], 0.0)


def test_qutrit_repetition_code_compressed_state_has_expected_support_size():
    G = np.array([[1, 1, 1, 1]], dtype=int)

    full_state = qutrit_magic_code_state(G)
    compressed_state = qutrit_magic_code_state_compressed(G)

    assert full_state.shape == (3**4,)
    assert compressed_state.shape == (3**3,)
    assert np.count_nonzero(np.abs(full_state) > 1e-12) == compressed_state.size
