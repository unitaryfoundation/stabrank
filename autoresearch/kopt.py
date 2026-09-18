"""Exhaustive completion of a plateau configuration: can any k of its terms be
replaced so that the rest plus k dictionary states span the target?

The annealer stops at configurations whose residual is an exact algebraic
number, the same one across seeds: local optima of its exchange landscape.
Its moves change one term at a time, so a configuration from which the target
is reachable only by replacing several terms at once is a dead end for it.
This tool tests every such replacement exactly. Fix the r - k kept terms,
project the target and the whole dictionary onto the orthogonal complement of
their span, and ask whether the projected target lies in the span of k
projected dictionary states: for k = 1 that is a parallelism test, for k = 2
the rank-2 quotient search of rank_exclusion.py, for k = 3 its rank-3 pivot
search. A failure is informative: it certifies that no completion of that
partial basis exists, for every choice of the kept terms tried.

Usage:
    kopt.py ORBIT M plateau.npz [--k 2] [--keep-min 4]

Runs single-threaded at low priority by default; it is meant to be cheap.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))


def load_plateau(path):
    z = np.load(path)
    keys = sorted((k for k in z.files if k.startswith("arr_")), key=lambda s: int(s[4:]))
    return [z[k] for k in keys], float(z["residual"])


def complement_projector(kept):
    """Orthonormal basis of span(kept) and the projector onto its complement."""
    Q, _ = np.linalg.qr(np.column_stack(kept))
    return Q


def _confirm(psi_r, Dr, sets, k):
    """Keep only the candidate sets whose projected states are independent and
    span the projected target. After projecting off the kept terms, distinct
    dictionary states can become parallel, and the quotient tests flag such a
    pair exactly as they flag a genuine completion, so every candidate is
    re-solved here."""
    out = []
    for s in sets:
        A = Dr[:, list(s)]
        if np.linalg.matrix_rank(A, tol=1e-8) < (k or len(s)):
            continue
        x, *_ = np.linalg.lstsq(A, psi_r, rcond=None)
        if np.linalg.norm(A @ x - psi_r) < 1e-9:
            out.append(s)
    return out


def _dedupe_parallel(Dr, idx):
    """One representative per projective direction among unit columns of Dr.

    After projecting off the kept terms, distinct dictionary states can have
    parallel images. The rank-3 search treats a parallel pair as a rank-2
    decomposition and stops, so duplicates are removed first; a completion
    using a removed state is still found through its representative, since
    only the direction matters."""
    rng = np.random.default_rng(3)
    can = rng.normal(size=Dr.shape[0]) + 1j * rng.normal(size=Dr.shape[0])
    key = rng.normal(size=Dr.shape[0]) + 1j * rng.normal(size=Dr.shape[0])
    kk = (key @ Dr) / (can @ Dr)
    order = np.lexsort((kk.imag, kk.real))
    ks = kk[order]
    gap = np.abs(np.diff(ks)) > 1e-8 * (1 + np.abs(ks[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(gap) + 1, [len(ks)]))
    keep = []
    for a, b in zip(starts[:-1], starts[1:]):
        members = order[a:b]
        reps = [members[0]]
        for j in members[1:]:
            if all(abs(np.vdot(Dr[:, r], Dr[:, j])) < 1 - 1e-6 for r in reps):
                reps.append(j)
        keep += reps
    keep = np.array(sorted(keep))
    return Dr[:, keep], idx[keep]


def completions(psi, D, kept, k):
    """Dictionary index sets of size k completing `kept` to a decomposition of psi.

    Returns (list of index tuples, diagnostics)."""
    from rank_exclusion import rank2_search, rank3_search
    Q = complement_projector(kept)
    psi_r = psi - Q @ (Q.conj().T @ psi)
    nr = np.linalg.norm(psi_r)
    if nr < 1e-9:
        return [("already in span",)], {"note": "target lies in the kept span"}
    psi_r /= nr
    Dr = D - Q @ (Q.conj().T @ D)
    nd = np.linalg.norm(Dr, axis=0)
    live = nd > 1e-9
    Dr = Dr[:, live] / nd[live]
    idx = np.flatnonzero(live)
    Dr, idx = _dedupe_parallel(Dr, idx)
    if k == 1:
        ov = np.abs(psi_r.conj() @ Dr)
        hits = idx[ov > 1 - 1e-9]
        return [(int(h),) for h in hits], {"closest": float(ov.max()), "live": int(live.sum())}
    if k == 2:
        pairs, worst = rank2_search(psi_r, Dr)
        if pairs == "RANK1":
            ov = np.abs(psi_r.conj() @ Dr)
            return [(int(idx[h]),) for h in np.flatnonzero(ov > 1 - 1e-9)], \
                {"note": "rank 1 completion"}
        good = _confirm(psi_r, Dr, pairs, 2)
        return [(int(idx[a]), int(idx[b])) for a, b in good], \
            {"closest": float(worst), "live": int(live.sum()), "flagged": len(pairs)}
    if k == 3:
        r = rank3_search(psi_r, Dr, workers=1)
        sets = [tuple(cols) for cols, _ in r["found"]]
        sets += [tuple(c for c in pr if c != "RANK1") for pr in r["rank2"]]
        good = _confirm(psi_r, Dr, sets, None)
        good = [s for s in good if np.linalg.matrix_rank(Dr[:, list(s)], tol=1e-8) == len(s)]
        return [tuple(int(idx[c]) for c in s) for s in good], \
            {"closest": float(r["worst"]), "candidates": r["candidates"],
             "live": int(live.sum()), "flagged": len(sets)}
    raise ValueError("k must be 1, 2 or 3")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit")
    ap.add_argument("m", type=int)
    ap.add_argument("plateau")
    ap.add_argument("--k", type=int, default=2, help="how many terms to replace")
    a = ap.parse_args(argv[1:])
    from rank_exclusion import dictionary, psi_for
    from stabrank_verify import ORBIT_P
    terms, resid = load_plateau(a.plateau)
    r = len(terms)
    psi = psi_for(a.orbit, a.m)
    psi = psi / np.linalg.norm(psi)
    D = dictionary(ORBIT_P[a.orbit], a.m)
    print(f"plateau of {r} terms at residual {resid:.4f}; replacing {a.k} of them, "
          f"{len(list(itertools.combinations(range(r), a.k)))} choices of kept set")
    any_found = False
    best = 0.0
    for drop in itertools.combinations(range(r), a.k):
        kept = [terms[i] for i in range(r) if i not in drop]
        found, diag = completions(psi, D, kept, a.k)
        best = max(best, diag.get("closest", 0.0))
        if found:
            any_found = True
            print(f"COMPLETION drop {drop}: dictionary states {found[:3]}"
                  f"{' ...' if len(found) > 3 else ''}")
    if not any_found:
        print(f"no completion by replacing any {a.k} terms; closest approach to a "
              f"completion {best:.6f} (parallel threshold 1 - 1e-6); every flagged pair "
              f"was a dependent pair of projected states, not a decomposition")
    return 0 if any_found else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
