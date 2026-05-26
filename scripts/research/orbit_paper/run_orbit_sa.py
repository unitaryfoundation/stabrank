"""SA upper-bound search for any of the four qutrit magic state orbits.

Usage:
    python run_orbit_sa.py <orbit> <m> <chi_target> [--seeds=<n>] [--iters=<i>]

orbit ∈ {strange, h3, norrell, t3}
"""

import argparse
import datetime
import random
import sys
from pathlib import Path

import numpy as np

from stabrank.target_functions import (
    qutrit_strange_state,
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_complex_magic_state,
)
from stabrank.polynomial_utils import generate_random_stabilizer_state
from stabrank.stabrank_core import run_sa_pauli_expansion as cpp_run_sa_pauli_expansion

try:
    from scripts.research.orbit_paper._paths import SOLUTIONS_DIR
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _paths import SOLUTIONS_DIR


sys.stdout.reconfigure(line_buffering=True)


BUILDERS = {
    "strange": qutrit_strange_state,
    "h3":      qutrit_hadamard_eigenstate,
    "norrell": qutrit_norrell_state,
    "t3":      qutrit_complex_magic_state,
}


def run_sa(target, n, k, seed, iters, num_chains=16, clifford_ratio=0.5):
    np.random.seed(seed)
    random.seed(seed)
    initial = [generate_random_stabilizer_state(n, p=3) for _ in range(k)]
    _, funcs, coeffs, err, cost, _ = cpp_run_sa_pauli_expansion(
        target=target, n_orig=n, p_prime=3, k_subset_size=k,
        initial_basis=initial,
        initial_temperature=1.0, cooling_rate=0.99,
        num_iterations_at_temp=iters,
        min_temperature=1.0 / 4000.0,
        atol=1e-7,
        two_func_perturb_prob=0.3,
        random_replace_prob=0.05,
        use_real_qubit_moves=False,
        clifford_ratio=clifford_ratio,
        early_exit_threshold=1e-9,
        seed=seed, num_chains=num_chains,
        fixed_dimension=-1,
    )
    return list(funcs), np.asarray(coeffs), float(err), float(cost)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("orbit", choices=BUILDERS.keys())
    parser.add_argument("m", type=int)
    parser.add_argument("chi_target", type=int)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--clifford-ratio", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=SOLUTIONS_DIR)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-chain SA logging")
    args = parser.parse_args()

    orbit, m, k = args.orbit, args.m, args.chi_target
    print(f"=== {orbit} SA: m={m} (dim={3**m}), chi_target={k}, seeds={args.seeds} ===", flush=True)

    builder = BUILDERS[orbit]
    target = builder(m).astype(np.complex128)
    target = target / np.linalg.norm(target)

    base_seed = np.random.randint(0, 10_000_000)
    print(f"base_seed={base_seed}", flush=True)

    best_err = float("inf")
    best_cost = float("inf")
    best_funcs, best_coeffs = None, None
    converged_seed = -1
    seeds_used = 0
    if args.quiet:
        # Redirect SA per-chain logs to /dev/null on this script's stdout
        # by capturing via run_sa (which prints to C++ stdout we cannot fully suppress)
        pass

    for s in range(args.seeds):
        seed = base_seed + s * 100003
        funcs, coeffs, err, cost = run_sa(
            target,
            m,
            k,
            seed,
            args.iters,
            num_chains=args.num_chains,
            clifford_ratio=args.clifford_ratio,
        )
        seeds_used = s + 1
        print(f"  seed {s+1}/{args.seeds}: err={err:.3e}", flush=True)
        if err < best_err:
            best_err = err
            best_cost = cost
            best_funcs = funcs
            best_coeffs = coeffs
        if err < 1e-9:
            converged_seed = s
            break

    print(f"best err={best_err:.3e}", flush=True)
    if best_err < 1e-9:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        filename = args.output_dir / f"solution_{orbit}_m{m}_chi{k}_{ts}.npz"
        np.savez_compressed(
            filename,
            target_function=target, linear_coeffs=best_coeffs,
            n=m, p=3, k=k, final_error=best_err, final_cost=best_cost,
            orbit=orbit, timestamp_utc=ts,
            base_seed=base_seed, converged_seed=converged_seed,
            seeds_requested=args.seeds, seeds_used=seeds_used,
            num_iterations_at_temp=args.iters,
            num_chains=args.num_chains,
            clifford_ratio=args.clifford_ratio,
            script="scripts/research/orbit_paper/run_orbit_sa.py",
            **{f"basis_func_{i}": f for i, f in enumerate(best_funcs)},
        )
        print(f"saved {filename}", flush=True)
        gamma = float(np.log(k) / np.log(3) / m) if m > 0 else float("nan")
        print(f"chi({orbit}^{m}) <= {k}, gamma = log_3({k})/{m} = {gamma:.4f}", flush=True)
        return 0
    print(f"FAILED to converge at chi={k} after {args.seeds} seeds.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
