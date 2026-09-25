"""Rank-8 attempts at |T>^7 inside the KvdWV partial-times-block structure.

The board's chi(|H>^7) <= 9 (`bounds/qubit_H-m7-upper-9.json`) is the
4-to-3 partial decomposition of Kissinger, van de Wetering, and Vilmart
(arXiv:2202.09202): |T>^5 = sqrt 2 sum_j c_j <T|_6 s_j with s_j the three
cat_6 terms, so |T>^7 = sum_j c_j (I^5 (x) |T>^2 <T|) s_j, and |T>^2 <T| is
written through the Choi vector as a rank-3 decomposition of |T>^3, giving
3 x 3 = 9 terms. `kvv_cat.py` uses the board's m=3 witness for that block.
Here the block is varied over every stored rank-3 decomposition of |H>^3
(`data/qubit_H_m3_rank3.json`, one per symmetry orbit, mapped to the T
basis), independently for each of the three partial terms j, and each
resulting nine-term decomposition is tested exactly for a merge: a pair of
terms whose weighted sum is itself a stabilizer state (`term_from_vector`),
which would give rank 8 and the exponent log_2(8)/7 = 0.4286 (the cell
6 <= chi <= 9; a rank-8 witness tightens the cell but does not beat
log_2(3)/4). Two other nine-term shapes are tested the same way: the
product |T>^4 (x) |T>^3 is 4 x 3 = 12 and is skipped; the product of the
partial with the block placed on qubits 1, 2 instead of 6, 7 is the same
set up to a permutation. With --anneal a warm-started rank-8 anneal from
each nine-term set with one term pruned is run (the board's annealer, a
search, not an exclusion).

Usage:
    t7_partial_merge.py [--anneal] [--chains 2] [--iters 3000] [--max-sets N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA, term_vector  # noqa: E402
from kvv_cat import C, S, T, apply_on, cat6_terms, lstsq_check, power  # noqa: E402
from to_witness import NotStabilizer, term_from_vector  # noqa: E402


def block_decompositions():
    """The stored rank-3 decompositions of |H>^3, as term vectors in the T
    basis (the board's H = C T on every qubit)."""
    rec = json.load(open(os.path.join(DATA, "qubit_H_m3_rank3.json")))
    out = []
    for dec in rec["decompositions"]:
        terms = []
        for t in dec:
            v = term_vector(t, 2, 3)
            for q in range(3):
                v = apply_on(C.conj().T, q, 3, v)
            terms.append(v)
        A = np.stack(terms, axis=1)
        c, *_ = np.linalg.lstsq(A, power(T, 3), rcond=None)
        assert np.linalg.norm(A @ c - power(T, 3)) < 1e-9
        out.append([c[i] * terms[i] for i in range(3)])
    return out


def choi_terms(block):
    """(M_l) with sum_l M_l = |T>^2 <T| from a decomposition of |T>^3."""
    out = []
    for v in block:
        ch = apply_on(S.conj().T, 0, 3, v)
        out.append(ch.reshape(2, 4).T)        # M[b, a] from |a>|b>
    return out


def nine_term_sets(blocks):
    """All choices of one block decomposition per cat_6 term j: the terms
    (I^5 (x) M_l) s_j of kvv_cat.partial_product_terms with the block varied,
    with the coefficients fitted so that the nine terms sum to |T>^7."""
    cat = cat6_terms()
    target = power(T, 7)
    chois = [choi_terms(b) for b in blocks]
    for choice in itertools.product(range(len(blocks)), repeat=3):
        raw = []
        for j, s in enumerate(cat):
            s = s.reshape(32, 2)
            for M in chois[choice[j]]:
                raw.append(np.einsum("xa,ba->xb", s, M).reshape(-1))
        A = np.stack(raw, axis=1)
        c, *_ = np.linalg.lstsq(A, target, rcond=None)
        assert np.linalg.norm(A @ c - target) < 1e-9, "nine terms do not span |T>^7"
        yield choice, [c[i] * raw[i] for i in range(9)]


def cat_state():
    from kvv_cat import cat
    return cat(6)


def is_stab(v):
    try:
        term_from_vector(v, 2, 7)
        return True
    except (NotStabilizer, ValueError, IndexError):
        return False


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max-sets", type=int, default=2)
    a = ap.parse_args(argv[1:])
    t0 = time.time()
    blocks = block_decompositions()
    print(f"{len(blocks)} stored rank-3 decompositions of |T>^3 (T basis)")
    target = power(T, 7)
    nsets = 0
    merges = 0
    distinct = set()
    pool = {}
    for choice, terms in nine_term_sets(blocks):
        nsets += 1
        A = np.stack(terms, axis=1)
        assert np.linalg.norm(A.sum(axis=1) - target) < 1e-8, "nine terms do not rebuild |T>^7"
        assert all(is_stab(v) for v in terms), "a term is not a stabilizer state"
        for v in terms:
            pool[(np.round(v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]], 6) + 0.0).tobytes()] = v
        for i, j in itertools.combinations(range(9), 2):
            s = terms[i] + terms[j]
            if np.linalg.norm(s) > 1e-9 and is_stab(s):
                merges += 1
                print(f"  MERGE in set {choice}: terms {i}, {j}")
        distinct.add(tuple(sorted((np.round(np.abs(v), 6)).tobytes() for v in terms)))
    P = np.stack(list(pool.values()), axis=1)
    print(f"{nsets} nine-term sets, {len(distinct)} distinct up to term moduli, "
          f"{merges} pairwise merges; pool of {P.shape[1]} states spans "
          f"{np.linalg.matrix_rank(P, tol=1e-8)} dimensions; {time.time() - t0:.0f}s")
    if a.anneal:
        from stabrank import (can_represent_as_linear_combination,
                              prune_least_significant_basis_function)
        from stabrank.stabrank_core import run_sa_pauli_expansion
        done = 0
        for choice, terms in nine_term_sets(blocks):
            if done >= a.max_sets:
                break
            done += 1
            funcs = [v / np.linalg.norm(v) for v in terms]
            pruned, _, _ = prune_least_significant_basis_function(
                target, funcs, can_represent_as_linear_combination)
            basis = pruned if len(pruned) == 8 else funcs[:8]
            t1 = time.time()
            np.random.seed(1)
            _, _, _, err, _, _ = run_sa_pauli_expansion(
                target=target, n_orig=7, p_prime=2, k_subset_size=8, initial_basis=basis,
                initial_temperature=1.0, cooling_rate=0.99, num_iterations_at_temp=a.iters,
                min_temperature=1 / 4000, atol=1e-7, two_func_perturb_prob=0.3,
                random_replace_prob=0.05, use_real_qubit_moves=False, clifford_ratio=0.5,
                early_exit_threshold=1e-9, seed=1, num_chains=a.chains)
            print(f"  warm rank-8 anneal from set {choice}: residual {err:.4f} ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main(sys.argv)
