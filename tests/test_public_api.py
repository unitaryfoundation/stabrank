import numpy as np
import pytest

import stabrank


def test_star_import_exports_only_existing_symbols():
    namespace: dict[str, object] = {}
    exec("from stabrank import *", namespace)

    assert set(stabrank.__all__).issubset(namespace)
    assert all(hasattr(stabrank, name) for name in stabrank.__all__)


def test_qubit_constrained_phase_sum_is_exported_and_normalized():
    target = stabrank.qubit_magic_phase_sum_constrained(3)

    assert target.shape == (8,)
    assert np.linalg.norm(target) == pytest.approx(1.0)
