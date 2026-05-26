"""Squeeze SA: starting at chi=k_start, try to find decompositions at decreasing
chi until the search fails. Saves the smallest converged chi.

Usage:
    python squeeze_orbit_sa.py <orbit> <m> --start=<k_start> [--seeds=<n>] [--iters=<i>]
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


def run_sa_once(target, n, k, seed, iters, num_chains=16, clifford_ratio=0.5):
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


def find_chi_at(target, n, k, seeds, iters, num_chains=16, clifford_ratio=0.5):
    base_seed = np.random.randint(0, 10_000_000)
    best_err = float("inf")
    best_cost = float("inf")
    best_funcs, best_coeffs = None, None
    best_seed = -1
    for s in range(seeds):
        seed = base_seed + s * 100003
        funcs, coeffs, err, cost = run_sa_once(
            target,
            n,
            k,
            seed,
            iters,
            num_chains=num_chains,
            clifford_ratio=clifford_ratio,
        )
        if err < best_err:
            best_err = err
            best_cost = cost
            best_funcs = funcs
            best_coeffs = coeffs
            best_seed = seed
        if err < 1e-9:
            return funcs, coeffs, err, cost, s + 1, base_seed, seed
    return best_funcs, best_coeffs, best_err, best_cost, seeds, base_seed, best_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("orbit", choices=BUILDERS.keys())
    parser.add_argument("m", type=int)
    parser.add_argument("--start", type=int, required=True,
                        help="Starting chi value (will decrease)")
    parser.add_argument("--floor", type=int, default=1,
                        help="Stop searching below this chi (default 1)")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--clifford-ratio", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=SOLUTIONS_DIR)
    args = parser.parse_args()

    orbit, m, k_start = args.orbit, args.m, args.start
    print(f"=== {orbit} squeeze: m={m} (dim={3**m}), start={k_start}, floor={args.floor} ===")

    builder = BUILDERS[orbit]
    target = builder(m).astype(np.complex128)
    target = target / np.linalg.norm(target)

    last_converged = None
    last_funcs, last_coeffs = None, None
    last_err = float("inf")
    last_cost = float("inf")
    last_used = 0
    last_base_seed = -1
    last_seed = -1

    for k in range(k_start, args.floor - 1, -1):
        print(f"--- chi={k} ---", flush=True)
        funcs, coeffs, err, cost, used, base_seed, seed = find_chi_at(
            target,
            m,
            k,
            args.seeds,
            args.iters,
            num_chains=args.num_chains,
            clifford_ratio=args.clifford_ratio,
        )
        if err < 1e-9:
            print(f"  CONVERGED at chi={k} (seed {used}, err={err:.2e})", flush=True)
            gamma = float(np.log(k) / np.log(3) / m) if m > 0 else float("nan")
            print(f"  chi({orbit}^{m}) <= {k}, gamma = {gamma:.4f}", flush=True)
            last_converged = k
            last_funcs, last_coeffs = funcs, coeffs
            last_err = err
            last_cost = cost
            last_used = used
            last_base_seed = base_seed
            last_seed = seed
        else:
            print(f"  FAILED at chi={k} (best err={err:.3e}, all {used} seeds tried)", flush=True)
            break

    if last_converged is not None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        filename = args.output_dir / f"solution_{orbit}_m{m}_chi{last_converged}_{ts}.npz"
        np.savez_compressed(
            filename,
            target_function=target, linear_coeffs=last_coeffs,
            n=m, p=3, k=last_converged, final_error=last_err,
            final_cost=last_cost, orbit=orbit, timestamp_utc=ts,
            base_seed=last_base_seed, converged_seed=last_seed,
            seeds_requested=args.seeds, seeds_used=last_used,
            num_iterations_at_temp=args.iters,
            num_chains=args.num_chains,
            clifford_ratio=args.clifford_ratio,
            script="scripts/research/orbit_paper/squeeze_orbit_sa.py",
            **{f"basis_func_{i}": f for i, f in enumerate(last_funcs)},
        )
        print(f"saved {filename}", flush=True)
        return 0
    else:
        print(f"No convergence at any chi from {k_start} down to {args.floor}.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
