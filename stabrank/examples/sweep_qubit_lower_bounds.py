"""Sweep all qubit magic code states for a given (m, k) and compute lower bounds.

Usage:
    uv run python -m stabrank.examples.sweep_qubit_lower_bounds --m 6
    uv run python -m stabrank.examples.sweep_qubit_lower_bounds --m 8 --k 1
    uv run python -m stabrank.examples.sweep_qubit_lower_bounds --m 8 --k 1 --k 2

Enumerates all systematic [m, k] binary codes G = [I_k | A] with A in F_2^{k x (m-k)},
computes the exhaustive stabilizer fidelity lower bound via C++, and saves results to CSV.
"""

import argparse
import csv
import datetime
import itertools
import math
import time

import numpy as np

from stabrank.stabrank_core import max_stabilizer_fidelity
from stabrank.target_functions import qubit_magic_code_state_compressed


def sweep_lower_bounds(m: int, k_values: list[int] | None = None) -> None:
    d = 2

    if k_values is None:
        k_values = list(range(1, m // 2 + 1))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sweep_qubit_lower_bounds_m{m}_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m", "k_code", "A_idx", "G_flat", "n_eff", "dim",
            "f_max", "extent_lb", "effective_extent_lb", "gamma_extent",
            "best_k_stab", "time_seconds",
            *[f"f_max_k{kk}" for kk in range(m)],
        ])

    print(f"=== Sweep Qubit Lower Bounds for m={m} ===")
    print(f"Output: {csv_filename}\n")

    total_codes = 0

    for k in k_values:
        n_eff = m - k
        num_A = d ** (k * n_eff)
        print(f"--- k={k}, n_eff={n_eff}, dim={d**n_eff}, codes to check: {num_A} ---")

        for A_idx, A_flat in enumerate(
            itertools.product(range(d), repeat=k * n_eff)
        ):
            A = np.array(A_flat, dtype=int).reshape((k, n_eff))
            G = np.hstack([np.eye(k, dtype=int), A])

            psi = qubit_magic_code_state_compressed(G)
            psi_normed = psi / np.linalg.norm(psi)

            t0 = time.time()
            result = max_stabilizer_fidelity(psi_normed, n=n_eff, d=d)
            elapsed = time.time() - t0

            f_max = result["f_max"]
            extent_lb = result["extent_lb"]
            f_per_k = list(result["f_max_per_k"])

            eff_extent = extent_lb * (d ** k)
            gamma_extent = (
                math.log(extent_lb, d) / (m - 2 * k)
                if extent_lb > 1 and m - 2 * k > 0
                else 0.0
            )

            best_k_stab = max(range(n_eff + 1), key=lambda kk: f_per_k[kk])

            G_str = "".join(map(str, G.flatten()))
            status = (
                f"  [{A_idx + 1}/{num_A}] G={G_str}  "
                f"xi>={extent_lb}  eff>={eff_extent}  "
                f"gamma_xi={gamma_extent:.4f}  best_k={best_k_stab}  "
                f"({elapsed:.1f}s)"
            )
            print(status)

            f_per_k_padded = f_per_k + [0.0] * (m - len(f_per_k))

            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    m, k, A_idx, G_str, n_eff, d ** n_eff,
                    f_max, extent_lb, eff_extent, f"{gamma_extent:.6f}",
                    best_k_stab, f"{elapsed:.2f}",
                    *[f"{fk:.8f}" for fk in f_per_k_padded],
                ])

            total_codes += 1

    print("\n=== SWEEP COMPLETE ===")
    print(f"Total codes checked: {total_codes}")
    print(f"Results saved to {csv_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep qubit code states and compute stabilizer rank lower bounds."
    )
    parser.add_argument("--m", type=int, required=True, help="Number of qubits m.")
    parser.add_argument(
        "--k", type=int, action="append", default=None,
        help="Code dimension(s) k to sweep. Can be repeated. Default: all k < m/2.",
    )
    args = parser.parse_args()
    sweep_lower_bounds(m=args.m, k_values=args.k)


if __name__ == "__main__":
    main()
