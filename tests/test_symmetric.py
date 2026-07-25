"""Tests for the permutation-symmetric ansatz search."""
import numpy as np
import pytest

from stabrank.symmetric import (
    SymmetricSAConfig,
    apply_cx,
    apply_cz,
    apply_f,
    apply_s,
    apply_x,
    apply_z,
    orbit_columns,
    periodic_seed,
    run_symmetric_sa,
    shift_qudits,
)
from stabrank.target_functions import (
    qubit_t_type_magic_state,
    qutrit_strange_state,
)


def test_shift_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=81) + 1j * rng.normal(size=81)
    assert np.allclose(shift_qudits(shift_qudits(v, 1, 4, 3), -1, 4, 3), v)


@pytest.mark.parametrize("p,n", [(2, 4), (3, 3)])
def test_gates_are_norm_preserving(p, n):
    rng = np.random.default_rng(1)
    dim = p**n
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    norm = np.linalg.norm(v)
    for out in (
        apply_x(v, 1, n, p),
        apply_z(v, 1, n, p),
        apply_s(v, 0, n, p),
        apply_f(v, n - 1, n, p),
        apply_cz(v, 0, 1, n, p),
        apply_cx(v, 0, n - 1, n, p),
    ):
        assert np.isclose(np.linalg.norm(out), norm)


def test_periodic_seed_is_shift_invariant():
    rng = np.random.default_rng(2)
    for p, n, period in [(2, 6, 2), (3, 6, 3), (2, 4, 1)]:
        s = periodic_seed(n, p, period, rng)
        shifted = shift_qudits(s, period, n, p)
        overlap = abs(np.vdot(shifted, s))
        assert np.isclose(overlap, np.vdot(s, s).real)


def test_orbit_columns_size_and_closure():
    rng = np.random.default_rng(3)
    s = periodic_seed(4, 2, 2, rng)  # invariant under shift by 2 (unit 1, orbit 2)
    cols = orbit_columns(s, 2, 1, 4, 2)
    assert len(cols) == 2
    # Shifting the last column by the unit returns to the first (up to phase).
    back = shift_qudits(cols[-1], 1, 4, 2)
    assert np.isclose(abs(np.vdot(back, cols[0])), np.vdot(cols[0], cols[0]).real)


def test_symmetric_sa_finds_strange_m2_rank2():
    target = qutrit_strange_state(2)
    cfg = SymmetricSAConfig(
        shift_unit=1, orbit_sizes=(1, 1),
        initial_temperature=0.5, min_temperature=1e-3,
        cooling_rate=0.995, iterations_at_temp=200,
    )
    err, cols, coeffs = run_symmetric_sa(target, 2, 3, cfg, seed=1)
    assert err < 1e-9
    a = np.column_stack(cols)
    assert np.linalg.norm(target - a @ coeffs) < 1e-9


def test_config_validation():
    target = qubit_t_type_magic_state(2)
    with pytest.raises(ValueError):
        run_symmetric_sa(
            target, 2, 2,
            SymmetricSAConfig(shift_unit=3, orbit_sizes=(1,)), seed=0)
    with pytest.raises(ValueError):
        run_symmetric_sa(
            target, 2, 2,
            SymmetricSAConfig(shift_unit=1, orbit_sizes=(4,)), seed=0)
