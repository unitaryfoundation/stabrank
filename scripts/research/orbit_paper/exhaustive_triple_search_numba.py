"""Numba-accelerated exhaustive-triple search for chi(M^otimes m) >= 4 vs <= 3.

Optimized variant of exhaustive_triple_search.py: explicit Cramer's-rule 3x3
solve in a numba JIT kernel, parallelized across cores via numba prange.
On a Ryzen 12-16 core machine this should run roughly 30-50x faster than
the pure-numpy version, putting m=3 within ~1-2 days per orbit.

Math (same as Lemma 1 extended to triples):
  chi(psi) <= 3  iff  psi in span(s_1, s_2, s_3) for some stabilizer triple.
For each triple, the squared residual ||psi - P_span psi||^2 = 1 - b^H G^-1 b
where G is the 3x3 Hermitian Gram matrix S^H S restricted to the triple, and
b = S^H psi is the (3,) inner-product vector. With det(G) real positive,
b^H G^-1 b = b^H adj(G) b / det(G), all real -- the kernel computes this
without an explicit matrix solve.

USAGE:
  python exhaustive_triple_search_numba.py --orbit strange --m 3 \\
    --batch-size 200000 --checkpoint ck-strange-m3.json
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

ORBITS = {
    "strange": qutrit_strange_state,
    "h3": qutrit_hadamard_eigenstate,
    "norrell": qutrit_norrell_state,
    "t3": qutrit_complex_magic_state,
}


@njit(cache=True, parallel=True, fastmath=True)
def kernel_residuals_sq(
    S: np.ndarray, psi: np.ndarray, triples: np.ndarray
) -> np.ndarray:
    """Squared residuals for each (i, j, k) triple, computed via Cramer's rule.

    S: (N, D) complex128 stabilizer dictionary, rows normalized.
    psi: (D,) complex128 target state, normalized.
    triples: (B, 3) int64 row-indices into S.

    Returns (B,) float64 of clipped non-negative squared residuals.
    """
    B = triples.shape[0]
    D = S.shape[1]
    out = np.empty(B, dtype=np.float64)

    for b in prange(B):
        i0 = triples[b, 0]
        i1 = triples[b, 1]
        i2 = triples[b, 2]

        # Inner products b_t = <s_t | psi> = sum_d conj(S[t, d]) * psi[d]
        b0 = 0.0 + 0.0j
        b1 = 0.0 + 0.0j
        b2 = 0.0 + 0.0j
        for d in range(D):
            p = psi[d]
            b0 += S[i0, d].conjugate() * p
            b1 += S[i1, d].conjugate() * p
            b2 += S[i2, d].conjugate() * p

        # Gram off-diagonals G_ij = <s_i | s_j>; diagonals are 1 (normalized).
        G01 = 0.0 + 0.0j
        G02 = 0.0 + 0.0j
        G12 = 0.0 + 0.0j
        for d in range(D):
            s0 = S[i0, d]
            s1 = S[i1, d]
            s2 = S[i2, d]
            G01 += s0.conjugate() * s1
            G02 += s0.conjugate() * s2
            G12 += s1.conjugate() * s2

        # Squared moduli of off-diagonals (real).
        a01 = (G01.conjugate() * G01).real
        a02 = (G02.conjugate() * G02).real
        a12 = (G12.conjugate() * G12).real

        # det(G) for Hermitian 3x3 with diagonals = 1.
        # det = 1 - |G12|^2 - |G02|^2 - |G01|^2
        #       + 2 * Re(G01 * G12 * conj(G02))
        cross = G01 * G12 * G02.conjugate()
        det = 1.0 - a12 - a02 - a01 + 2.0 * cross.real

        # Adjugate-quadratic-form numerator: b^H adj(G) b.
        # For Hermitian G with G_ii = 1:
        #   adj(G)_00 = 1 - |G12|^2
        #   adj(G)_11 = 1 - |G02|^2
        #   adj(G)_22 = 1 - |G01|^2
        #   adj(G)_01 = G02 * conj(G12) - G01
        #   adj(G)_02 = G01 * G12        - G02
        #   adj(G)_12 = conj(G01) * G02  - G12
        ad00 = 1.0 - a12
        ad11 = 1.0 - a02
        ad22 = 1.0 - a01
        ad01 = G02 * G12.conjugate() - G01
        ad02 = G01 * G12 - G02
        ad12 = G01.conjugate() * G02 - G12

        # b^H adj(G) b = sum_{ij} conj(b_i) adj_ij b_j; real for Hermitian adj.
        # Diagonals contribute |b_i|^2 * adj_ii (real).
        # Off-diagonals contribute 2 * Re(conj(b_i) * adj_ij * b_j).
        bb0 = (b0.conjugate() * b0).real
        bb1 = (b1.conjugate() * b1).real
        bb2 = (b2.conjugate() * b2).real
        cross01 = (b0.conjugate() * ad01 * b1).real
        cross02 = (b0.conjugate() * ad02 * b2).real
        cross12 = (b1.conjugate() * ad12 * b2).real
        proj_num = (
            bb0 * ad00 + bb1 * ad11 + bb2 * ad22
            + 2.0 * (cross01 + cross02 + cross12)
        )

        if det > 1e-14:
            proj_norm_sq = proj_num / det
            res = 1.0 - proj_norm_sq
        else:
            # Linearly dependent triple: span has dim < 3, residual is the
            # 2-state best fit. We don't attempt to recover that here -- the
            # pair search (Lemma 1) already covered chi <= 2.
            res = 1.0

        # Numerical floor: residual^2 >= 0.
        if res < 0.0:
            res = 0.0
        out[b] = res

    return out


def numpy_triple_iter(N: int, batch_size: int, start_idx: int = 0):
    """Yield (B, 3) int64 batches in lex order over C(N, 3) triples.

    Faster than itertools-based generation: emits numpy arrays directly,
    avoiding Python-tuple overhead. Honors a starting linear index for
    checkpoint resume.
    """
    cum = 0  # cumulative triple count up to current outer-k iteration
    for k in range(2, N):
        # Triples with largest index = k: there are C(k, 2) of them.
        n_for_k = k * (k - 1) // 2
        if cum + n_for_k <= start_idx:
            cum += n_for_k
            continue
        # Process (i, j) with i < j < k. Generate all such pairs as numpy
        # arrays, then chunk by batch_size.
        # i_arr, j_arr are length C(k, 2).
        # Use indices_pair_lex(k) for lex ordering: (0,1), (0,2), (1,2),
        # (0,3), (1,3), (2,3), ..., (k-2, k-1).
        # Equivalently: for j in 1..k-1, for i in 0..j-1.
        for j in range(1, k):
            local_count = j  # number of i values for this j
            local_cum = cum + (j - 1) * j // 2  # triples processed so far for this k, all j' < j
            if local_cum + local_count <= start_idx:
                continue
            i_lo = max(0, start_idx - local_cum)
            for i_start in range(i_lo, local_count, batch_size):
                i_end = min(i_start + batch_size, local_count)
                B = i_end - i_start
                triples = np.empty((B, 3), dtype=np.int64)
                triples[:, 0] = np.arange(i_start, i_end)
                triples[:, 1] = j
                triples[:, 2] = k
                yield triples
        cum += n_for_k


def search_orbit_numba(
    orbit: str,
    m: int,
    tol: float = 1e-10,
    batch_size: int = 200_000,
    checkpoint_path: Path | None = None,
    log_every: int = 50_000_000,
    early_exit_on_witness: bool = True,
) -> dict:
    """Run the numba-accelerated triple search for one (orbit, m) pair."""
    if orbit not in ORBITS:
        raise ValueError(f"orbit must be one of {sorted(ORBITS)}")

    psi = ORBITS[orbit](m).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    print(f"[{orbit} m={m}] target dim {psi.shape[0]}")

    t0 = time.time()
    S = enumerate_stabilizer_states(m, d=3).astype(np.complex128)
    S = normalize_rows(S)
    N = S.shape[0]
    n_triples = N * (N - 1) * (N - 2) // 6
    print(
        f"[{orbit} m={m}] enumerated {N} stabilizer states "
        f"({time.time() - t0:.1f}s); {n_triples:,} triples"
    )

    # Resume from checkpoint if present.
    start_idx = 0
    best_res_sq = float("inf")
    best_triple: tuple[int, int, int] | None = None
    if checkpoint_path is not None and checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        if ck.get("orbit") == orbit and ck.get("m") == m:
            start_idx = ck["n_processed"]
            best_res_sq = ck["best_residual_sq"]
            best_triple = (
                tuple(ck["best_triple"]) if ck["best_triple"] else None
            )
            print(f"[{orbit} m={m}] resuming from idx {start_idx:,}")

    # JIT warm-up: kernel needs a small invocation to compile before the
    # main loop, otherwise the first batch absorbs the compile cost.
    print(f"[{orbit} m={m}] warming up numba kernel...")
    _ = kernel_residuals_sq(S, psi, np.array([[0, 1, 2]], dtype=np.int64))
    print(f"[{orbit} m={m}] kernel ready ({time.time() - t0:.1f}s)")

    t_start = time.time()
    n_processed = start_idx
    next_log = ((n_processed // log_every) + 1) * log_every
    last_checkpoint_idx = n_processed

    for batch in numpy_triple_iter(N, batch_size, start_idx=start_idx):
        res_sq = kernel_residuals_sq(S, psi, batch)
        idx_local = int(np.argmin(res_sq))
        local_best = float(res_sq[idx_local])
        if local_best < best_res_sq:
            best_res_sq = local_best
            best_triple = tuple(int(x) for x in batch[idx_local])

        n_processed += batch.shape[0]

        if n_processed >= next_log:
            elapsed = time.time() - t_start
            rate = (n_processed - start_idx) / max(elapsed, 1e-9)
            eta_s = (n_triples - n_processed) / max(rate, 1e-9)
            print(
                f"[{orbit} m={m}] {n_processed:,}/{n_triples:,} "
                f"({n_processed / n_triples * 100:.3f}%) "
                f"best_resid={np.sqrt(best_res_sq):.3e} "
                f"rate={rate / 1e6:.1f}M/s "
                f"eta={eta_s / 3600:.1f}h"
            )
            next_log = ((n_processed // log_every) + 1) * log_every

        # Checkpoint every ~1% of progress (or every batch for small m).
        if (
            checkpoint_path is not None
            and n_processed - last_checkpoint_idx >= max(n_triples // 100, batch_size)
        ):
            ck = dict(
                orbit=orbit,
                m=m,
                n_processed=n_processed,
                best_residual_sq=best_res_sq,
                best_triple=list(best_triple) if best_triple else None,
            )
            checkpoint_path.write_text(json.dumps(ck))
            last_checkpoint_idx = n_processed

        if early_exit_on_witness and best_res_sq < tol * tol:
            print(
                f"[{orbit} m={m}] chi <= 3 WITNESS at idx {n_processed:,}; "
                f"residual={np.sqrt(best_res_sq):.3e}"
            )
            break

    elapsed = time.time() - t_start
    found_witness = best_res_sq < tol * tol
    result = dict(
        orbit=orbit,
        m=m,
        n_triples_total=n_triples,
        n_triples_processed=n_processed,
        elapsed_seconds=elapsed,
        best_residual=float(np.sqrt(best_res_sq)),
        best_triple=list(best_triple) if best_triple else None,
        chi_le_3_witness=found_witness,
        certificate=(
            "chi <= 3 (witness)" if found_witness else "chi >= 4 (exhaustive)"
        ),
    )
    result.update(
        build_certificate_metadata(
            target=psi,
            stabilizer_dictionary=S,
            tuple_size=3,
            script="scripts/research/orbit_paper/exhaustive_triple_search_numba.py",
            parameters={
                "tol": tol,
                "batch_size": batch_size,
                "checkpoint_path": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "log_every": log_every,
                "early_exit_on_witness": early_exit_on_witness,
            },
        )
    )

    # Final checkpoint.
    if checkpoint_path is not None:
        ck = dict(
            orbit=orbit,
            m=m,
            n_processed=n_processed,
            best_residual_sq=best_res_sq,
            best_triple=list(best_triple) if best_triple else None,
        )
        checkpoint_path.write_text(json.dumps(ck))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--orbit", required=True, choices=sorted(ORBITS), help="Magic-state orbit."
    )
    parser.add_argument("--m", type=int, required=True, help="Tensor power.")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-early-exit", action="store_true")
    parser.add_argument("--log-every", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = search_orbit_numba(
        orbit=args.orbit,
        m=args.m,
        tol=args.tol,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
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
