"""Numba-accelerated exhaustive-pair search for chi(M^otimes m) >= 3 vs <= 2.

Implements Lemma 1 of the paper as a deterministic CLI: chi(psi) <= 2 iff
psi lies in span(s_1, s_2) for some pair of stabilizer states. Iterating
over all C(|Stab_m^(3)|, 2) pairs and checking the 2-state least-squares
fit therefore gives a deterministic certificate of either chi <= 2
(with explicit witness pair) or chi >= 3 (exhaustive).

Math (2-state version of the Cramer's-rule projection used in the triple
search):
    For S = [s_0; s_1] with normalized rows, G = S^H S has G_00 = G_11 = 1
    and off-diagonal G_01 = <s_0|s_1>. With b_t = <s_t|psi>,
        ||psi - P_span psi||^2 = 1 - b^H G^-1 b
                               = 1 - (|b_0|^2 + |b_1|^2 - 2 Re(conj(b_0) G_01 b_1))
                                     / (1 - |G_01|^2).
    When |G_01| = 1 (s_0 and s_1 parallel) the pair spans a 1-dim subspace;
    we fall back to the single-state best-fit residual 1 - max(|b_0|^2, |b_1|^2).

USAGE:
  python exhaustive_pair_search.py --orbit t3 --m 1 --output cert-t3-m1.json
  python exhaustive_pair_search.py --orbit h3 --m 2 --output cert-h3-m2.json
  python exhaustive_pair_search.py --orbit strange --m 3 --output cert-strange-m3.json

Output JSON shape:
    {
      "orbit": "t3", "m": 1,
      "n_pairs_total": 66, "n_pairs_processed": 66,
      "best_residual": 0.2796..., "best_pair": [i, j],
      "chi_le_2_witness": false,
      "certificate": "chi >= 3 (exhaustive)",
      "elapsed_seconds": 0.012
    }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from stabrank.stabilizer_extent import enumerate_stabilizer_states
from stabrank.target_functions import (
    qubit_t_type_magic_state,
    qutrit_complex_magic_state,
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_strange_state,
)

try:
    from scripts.research.orbit_paper._certificate_metadata import (
        build_certificate_metadata,
        normalize_rows,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _certificate_metadata import build_certificate_metadata, normalize_rows

# The qudit enumerator in stabrank.stabilizer_extent uses phases
# omega_p^{Q(y)} which for d=2 yields only +/-1 and so misses the |+/-i>
# states. Use the dedicated qubit enumerator with the proper Z/4 phase
# polynomial structure when d == 2.
try:
    from scripts.research.qubit_stabilizer_enum import (
        enumerate_qubit_stabilizer_states,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from qubit_stabilizer_enum import enumerate_qubit_stabilizer_states


def _enumerate_dictionary(n: int, d: int) -> np.ndarray:
    """Dispatch to the correct enumerator for the local dimension."""
    if d == 2:
        return enumerate_qubit_stabilizer_states(n).astype(np.complex128)
    return enumerate_stabilizer_states(n, d=d).astype(np.complex128)

# Each orbit knows its local dimension; the enumerator and target are
# both selected from this table.
ORBITS = {
    "strange":  (qutrit_strange_state,          3),
    "h3":       (qutrit_hadamard_eigenstate,    3),
    "norrell":  (qutrit_norrell_state,          3),
    "t3":       (qutrit_complex_magic_state,    3),
    "qubit_t":  (qubit_t_type_magic_state,      2),
}


@njit(cache=True, parallel=True, fastmath=True)
def kernel_pair_residuals_sq(
    S: np.ndarray, psi: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    """Squared residuals ||psi - P_span(s_i, s_j) psi||^2 for each pair.

    S: (N, D) complex128 stabilizer dictionary, rows L2-normalized.
    psi: (D,) complex128 target state, L2-normalized.
    pairs: (B, 2) int64 row-indices into S with i < j.

    Returns (B,) float64 of clipped non-negative squared residuals.
    """
    B = pairs.shape[0]
    D = S.shape[1]
    out = np.empty(B, dtype=np.float64)

    for b in prange(B):
        i0 = pairs[b, 0]
        i1 = pairs[b, 1]

        # Inner products b_t = <s_t | psi> = sum_d conj(S[t, d]) * psi[d]
        b0 = 0.0 + 0.0j
        b1 = 0.0 + 0.0j
        # Off-diagonal Gram entry G_01 = <s_0 | s_1>
        G01 = 0.0 + 0.0j
        for d in range(D):
            p = psi[d]
            s0 = S[i0, d]
            s1 = S[i1, d]
            b0 += s0.conjugate() * p
            b1 += s1.conjugate() * p
            G01 += s0.conjugate() * s1

        a01 = (G01.conjugate() * G01).real  # |G_01|^2
        det = 1.0 - a01
        bb0 = (b0.conjugate() * b0).real
        bb1 = (b1.conjugate() * b1).real

        if det > 1e-14:
            # adj(G) = [[1, -G_01], [-conj(G_01), 1]]
            # b^H adj(G) b = |b_0|^2 + |b_1|^2 - 2 Re(conj(b_0) G_01 b_1)
            cross = (b0.conjugate() * G01 * b1).real
            proj_num = bb0 + bb1 - 2.0 * cross
            res = 1.0 - proj_num / det
        else:
            # Linearly dependent pair (parallel up to phase): fall back to the
            # 1-state best fit.
            best_single = bb0 if bb0 > bb1 else bb1
            res = 1.0 - best_single

        if res < 0.0:
            res = 0.0
        out[b] = res

    return out


def numpy_pair_iter(N: int, batch_size: int, start_idx: int = 0):
    """Yield (B, 2) int64 batches in lex order over C(N, 2) pairs.

    Lex order: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), ..., i.e.
    grouped by j (the larger index): for j in 1..N-1, for i in 0..j-1.
    """
    cum = 0
    for j in range(1, N):
        n_for_j = j  # pairs (0,j), (1,j), ..., (j-1, j)
        if cum + n_for_j <= start_idx:
            cum += n_for_j
            continue
        i_lo = max(0, start_idx - cum)
        for i_start in range(i_lo, n_for_j, batch_size):
            i_end = min(i_start + batch_size, n_for_j)
            B = i_end - i_start
            pairs = np.empty((B, 2), dtype=np.int64)
            pairs[:, 0] = np.arange(i_start, i_end)
            pairs[:, 1] = j
            yield pairs
        cum += n_for_j


def search_orbit_pair(
    orbit: str,
    m: int,
    tol: float = 1e-10,
    batch_size: int = 200_000,
    log_every: int = 50_000_000,
    early_exit_on_witness: bool = True,
) -> dict:
    """Run the numba-accelerated pair search for one (orbit, m) pair."""
    if orbit not in ORBITS:
        raise ValueError(f"orbit must be one of {sorted(ORBITS)}")

    target_fn, d = ORBITS[orbit]
    psi = target_fn(m).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    print(f"[{orbit} m={m} d={d}] target dim {psi.shape[0]}")

    t0 = time.time()
    S = _enumerate_dictionary(m, d)
    # Normalize rows.
    S = normalize_rows(S)
    N = S.shape[0]
    n_pairs = N * (N - 1) // 2
    print(
        f"[{orbit} m={m} d={d}] enumerated {N} stabilizer states "
        f"({time.time() - t0:.1f}s); {n_pairs:,} pairs"
    )

    # JIT warm-up.
    _ = kernel_pair_residuals_sq(S, psi, np.array([[0, 1]], dtype=np.int64))

    t_start = time.time()
    n_processed = 0
    best_res_sq = float("inf")
    best_pair: tuple[int, int] | None = None
    next_log = log_every

    for batch in numpy_pair_iter(N, batch_size):
        res_sq = kernel_pair_residuals_sq(S, psi, batch)
        idx_local = int(np.argmin(res_sq))
        local_best = float(res_sq[idx_local])
        if local_best < best_res_sq:
            best_res_sq = local_best
            best_pair = tuple(int(x) for x in batch[idx_local])

        n_processed += batch.shape[0]

        if n_processed >= next_log:
            elapsed = time.time() - t_start
            rate = n_processed / max(elapsed, 1e-9)
            print(
                f"[{orbit} m={m} d={d}] {n_processed:,}/{n_pairs:,} "
                f"({n_processed / n_pairs * 100:.3f}%) "
                f"best_resid={np.sqrt(best_res_sq):.3e} "
                f"rate={rate / 1e6:.1f}M/s"
            )
            next_log += log_every

        if early_exit_on_witness and best_res_sq < tol * tol:
            print(
                f"[{orbit} m={m} d={d}] chi <= 2 WITNESS at idx "
                f"{n_processed:,}; residual={np.sqrt(best_res_sq):.3e}"
            )
            break

    elapsed = time.time() - t_start
    found_witness = best_res_sq < tol * tol
    result = dict(
        orbit=orbit,
        m=m,
        n_pairs_total=n_pairs,
        n_pairs_processed=n_processed,
        elapsed_seconds=elapsed,
        best_residual=float(np.sqrt(best_res_sq)),
        best_pair=list(best_pair) if best_pair else None,
        chi_le_2_witness=found_witness,
        certificate=(
            "chi <= 2 (witness)" if found_witness else "chi >= 3 (exhaustive)"
        ),
    )
    result.update(
        build_certificate_metadata(
            target=psi,
            stabilizer_dictionary=S,
            tuple_size=2,
            script="scripts/research/orbit_paper/exhaustive_pair_search.py",
            parameters={
                "tol": tol,
                "batch_size": batch_size,
                "log_every": log_every,
                "early_exit_on_witness": early_exit_on_witness,
            },
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--orbit", required=True, choices=sorted(ORBITS), help="Magic-state orbit."
    )
    parser.add_argument("--m", type=int, required=True, help="Tensor power.")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--no-early-exit", action="store_true")
    parser.add_argument("--log-every", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = search_orbit_pair(
        orbit=args.orbit,
        m=args.m,
        tol=args.tol,
        batch_size=args.batch_size,
        log_every=args.log_every,
        early_exit_on_witness=not args.no_early_exit,
    )

    print()
    print("=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
