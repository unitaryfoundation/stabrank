"""Exhaustive-triple search: certify chi(M^otimes m) >= 4 vs <= 3.

Extension of Lemma 1 (paper/main.tex) one rank up:
  chi(psi) <= 3  iff  psi in span(s_1, s_2, s_3) for some stabilizer triple.
Iterate over all C(|Stab_m^(3)|, 3) triples, check the 3-state LS fit. If
no triple yields residual < tol, then chi >= 4 with deterministic certainty.

Combined with chi <= 4 from Theorem 3 of the paper, a "no witness" result
upgrades the m=3 entries for Strange / H_3 / Norrell from "<= 4" to "= 4".

Resource estimates (single-core numpy, batched):
  m=1 -- |Stab_1^(3)| =       12 ->     220 triples (~milliseconds)
  m=2 -- |Stab_2^(3)| =      414 ->  11.7 M triples (~minutes)
  m=3 -- |Stab_3^(3)| =   41,580 ->  1.2 e13 triples (~30+ days naive;
                                   feasible with Clifford symmetry
                                   reduction, GPU, or cluster)

USAGE:
  python exhaustive_triple_search.py --orbit strange --m 2
  python exhaustive_triple_search.py --orbit h3 --m 3 --checkpoint ck.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

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


def batch_triple_residuals_sq(
    S: np.ndarray, psi: np.ndarray, triples: np.ndarray
) -> np.ndarray:
    """Squared residuals ||psi - P_span psi||^2 for each triple.

    Args:
        S: (N, D) complex stabilizer dictionary, rows normalized.
        psi: (D,) complex target, normalized.
        triples: (B, 3) int64 row-indices into S.

    Returns:
        (B,) float array of squared residuals.
    """
    # (B, 3, D) batched stabilizer triples
    S_batch = S[triples]
    # (B, 3) inner products <s_t | psi>
    b = np.einsum("btd,d->bt", S_batch.conj(), psi)
    # (B, 3, 3) Gram matrices G_ts = <s_t | s_s>
    G = np.einsum("btd,bsd->bts", S_batch.conj(), S_batch)
    # Solve G c = b for each batch element. Some G are singular when the
    # triple is linearly dependent; use lstsq fallback per-element only there.
    try:
        c = np.linalg.solve(G, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        c = np.empty_like(b)
        for i in range(triples.shape[0]):
            c[i], *_ = np.linalg.lstsq(G[i], b[i], rcond=None)
    # ||P_span psi||^2 = c^H b (real, since G is Hermitian PSD).
    proj_norm_sq = np.real(np.einsum("bt,bt->b", c.conj(), b))
    # ||psi - P_span psi||^2 = ||psi||^2 - ||P_span psi||^2 = 1 - proj_norm_sq
    residuals_sq = 1.0 - proj_norm_sq
    # Numerical floor: residuals_sq can dip slightly below zero from float error.
    return np.clip(residuals_sq, 0.0, None)


def search_orbit(
    orbit: str,
    m: int,
    tol: float = 1e-10,
    batch_size: int = 100_000,
    checkpoint_path: Path | None = None,
    log_every: int = 1_000_000,
    early_exit_on_witness: bool = True,
) -> dict:
    """Run the triple search for one (orbit, m) pair.

    Returns a dict with the result. If chi <= 3 witness found, includes
    the triple indices and coefficients; else best_residual is the minimum
    achieved (a chi >= 4 certificate when best_residual > tol).
    """
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

    # Resume from checkpoint if available.
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
            print(f"[{orbit} m={m}] resuming from triple idx {start_idx:,}")

    triple_iter = itertools.combinations(range(N), 3)
    # Skip already-processed triples on resume.
    for _ in range(start_idx):
        next(triple_iter)

    t_start = time.time()
    n_processed = start_idx
    next_log = ((n_processed // log_every) + 1) * log_every

    while True:
        batch = list(itertools.islice(triple_iter, batch_size))
        if not batch:
            break

        triples = np.asarray(batch, dtype=np.int64)
        res_sq = batch_triple_residuals_sq(S, psi, triples)
        idx_local = int(np.argmin(res_sq))
        local_best = float(res_sq[idx_local])
        if local_best < best_res_sq:
            best_res_sq = local_best
            best_triple = tuple(int(x) for x in triples[idx_local])

        n_processed += len(batch)

        if n_processed >= next_log:
            elapsed = time.time() - t_start
            rate = (n_processed - start_idx) / max(elapsed, 1e-9)
            eta_s = (n_triples - n_processed) / max(rate, 1e-9)
            print(
                f"[{orbit} m={m}] {n_processed:,}/{n_triples:,} "
                f"({n_processed / n_triples * 100:.2f}%) "
                f"best_resid={np.sqrt(best_res_sq):.3e} "
                f"eta={eta_s / 60:.1f}min"
            )
            next_log = ((n_processed // log_every) + 1) * log_every

        if checkpoint_path is not None:
            ck = dict(
                orbit=orbit,
                m=m,
                n_processed=n_processed,
                best_residual_sq=best_res_sq,
                best_triple=list(best_triple) if best_triple else None,
            )
            checkpoint_path.write_text(json.dumps(ck))

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
            script="scripts/research/orbit_paper/exhaustive_triple_search.py",
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--orbit", required=True, choices=sorted(ORBITS), help="Magic-state orbit."
    )
    parser.add_argument("--m", type=int, required=True, help="Tensor power.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Witness tolerance.")
    parser.add_argument(
        "--batch-size", type=int, default=100_000, help="Triples per numpy batch."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="JSON file for resume support."
    )
    parser.add_argument(
        "--no-early-exit",
        action="store_true",
        help="Continue scanning even after a witness is found.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000_000,
        help="Log progress every this many triples.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON file for the final result."
    )
    args = parser.parse_args()

    result = search_orbit(
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
