"""The T3 m=5 Z-eigensector through the sector contraction of the exact
m=3 and m=4 sector decompositions.

Sectors. With |T3> = 3^(-1/2) sum_x w9^x |x> and Z|T3> = 3^(-1/2) sum_x
w9^(x+3x) |x>, the Z^(x m) eigensector s of |T3>^m is

    c_s^(m)(x) = 3^(-m/2) w9^s w3^((|x| - s)/3)   on |x| = s (mod 3),

|x| the integer sum of the digits, and |T3>^m = sum_s c_s^(m). The basis
permutation x -> (x_1, ..., x_(m-1), |x| mod 3) is a Clifford (SUM gates)
and carries c_s^(m) to |s> (x) psi_s^(m) with psi_s^(m)(x') =
w3^ceil((|x'| - s)/3) on m-1 qutrits, the carry state of
autoresearch/run.py. Its rank r_m gives chi(|T3>^m) <= 3 r_m; the board has
r_3 = r_4 = 3 (exact) and r_5 <= 6, and r_5 <= 5 would give
chi(|T3>^5) <= 15 and exponent 0.493 < 1/2.

Contraction. The three states Z^i|T3> are orthonormal, so the two-qutrit
vector |Phi> = sum_i Z^i|T3> (x) Z^i|T3> = 3 Pi_(ZZ = 1) |T3>^2 is c_0^(2)
up to scale, a stabilizer state (its carry state is w3^(x^2)). Contracting
one qutrit of c_j^(m) with one of c_k^(m') through <Phi| kills the mixed
terms Z^i|T3> (x) Z^(i')|T3>, i != i', and leaves c_(j+k)^(m+m'-2) up to a
scalar. A stabilizer term contracted with a stabilizer bra is a stabilizer
state or zero, so r_(m+m'-2) <= r_m r_(m'), the qutrit analogue of the
cat_(a+b-2) gluing of Qassim, Pashayan, and Gosset. At (m, m') = (3, 4)
this gives nine-term decompositions of every m=5 sector.

What this script does: lists every rank-3 decomposition of psi_s^(3) (full
search over the 360 two-qutrit states) and of psi_0^(4) (search over the
30,240 three-qutrit states with one pivot per S_3 orbit, closed under S_3;
the other two sectors follow by the shift psi_s(x' + e_1) = psi_(s-1)(x')),
forms every contraction (j, k) into every m=5 sector, confirms each
nine-term set reproduces the sector, pools the distinct four-qutrit carry
states the contractions use (closed under the S_4 symmetry of the sector),
compares them with the board's six-term sector decompositions
(bounds/T3-m5-upper-18.json), and tests, under a time budget, whether any
five-term decomposition shares four terms with a contraction (stabilizer
states in span(psi, four kept), one flat at a time). With --rank5 it also
runs an exact rank-5 pivot-pair search inside the closed pool (about 1 s
per pivot pair, 392,602 pairs per sector: a pod job, not a laptop one).

Usage (from the repository root):
    uv run --extra challenge python research/constructions/t3_sector_contraction.py
        [--sector S] [--budget SECONDS] [--skip N] [--rank5]
--skip N resumes the completion test after its first N distinct
four-subsets, so a sector can be finished in chunks under a wall-clock cap
(sector 0 took 17,575 subsets in 334 s and the remaining 2,135 in 47 s).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, ROOT, os.path.join(ROOT, "verify_challenge"), os.path.join(ROOT, "research", "merges"),
           os.path.join(ROOT, "autoresearch")):
    sys.path.insert(0, _p)

from common import term_vector  # noqa: E402
from mergelib import stabilizer_states_in_span  # noqa: E402
from rank_exclusion import dictionary, rank3_search  # noqa: E402
from to_witness import term_from_vector, NotStabilizer  # noqa: E402

W3 = np.exp(2j * np.pi / 3)
W9 = np.exp(2j * np.pi / 9)


def digits(n):
    return np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64)


def sector(m, s):
    """c_s^(m) as an m-qutrit vector, unit norm."""
    x = digits(m)
    tot = x.sum(axis=1)
    v = np.where(tot % 3 == s, W9 ** s * W3 ** ((tot - s) // 3), 0).astype(complex)
    return v / np.linalg.norm(v)


def carry(m, s):
    """psi_s^(m) on m-1 qutrits, unit norm (autoresearch/run.py sector_target)."""
    x = digits(m - 1)
    tot = x.sum(axis=1)
    v = W3 ** np.ceil((tot - s) / 3).astype(int)
    return v / np.linalg.norm(v)


def _sector_index(m, s):
    x = digits(m - 1)
    last = (s - x.sum(axis=1)) % 3
    return np.array([int("".join(map(str, list(r) + [l])), 3) for r, l in zip(x, last)])


def lift(v, m, s):
    """Carry-picture vector on m-1 qutrits -> m-qutrit vector supported on |x| = s."""
    out = np.zeros(3 ** m, dtype=complex)
    out[_sector_index(m, s)] = v
    return out


def compress(V, m, s):
    """m-qutrit vector supported on |x| = s -> carry-picture vector."""
    idx = _sector_index(m, s)
    mask = np.ones(3 ** m, dtype=bool)
    mask[idx] = False
    assert np.abs(V[mask]).max() < 1e-9, "vector leaves the sector"
    return V[idx]


def phi_bra():
    t = np.array([1, W9, W9 ** 2]) / np.sqrt(3)
    zs = [np.diag(W3 ** (i * np.arange(3))) for i in range(3)]
    return sum(np.kron(z @ t, z @ t) for z in zs).conj()


def contract(A, B, m, mp):
    """(<Phi| on qutrits m and m+1)(A (x) B) for A on m and B on mp qutrits."""
    bra = phi_bra()
    v = np.kron(A, B).reshape(3 ** (m - 1), 3, 3, 3 ** (mp - 1))
    return np.einsum("ab,iabj->ij", bra.reshape(3, 3), v).reshape(-1)


def ray_key(v, tol=1e-8):
    v = v / np.linalg.norm(v)
    i = np.flatnonzero(np.abs(v) > tol)[0]
    v = v / (v[i] / abs(v[i]))
    return tuple(np.round(v, 6).view(float))


def distinct(vectors, tol=1e-8):
    seen, out = set(), []
    for v in vectors:
        if np.linalg.norm(v) < tol:
            continue
        k = ray_key(v, tol)
        if k not in seen:
            seen.add(k)
            out.append(v / np.linalg.norm(v))
    return out


def permute_qutrits(v, perm, n):
    return v.reshape([3] * n).transpose(perm).reshape(-1)


def step_down(v):
    """Terms of psi_s to terms of psi_(s-1): since |x' + e_1| is |x'| + 1
    unless x'_1 = 2, where it is |x'| - 2, psi_(s-1)(x') = w3^[x'_1 = 2]
    psi_s(x' + e_1); the map is X^(-1) followed by the phase gate
    diag(1, 1, w3) on the first qutrit, a Clifford."""
    w = np.roll(v.reshape(3, -1), -1, axis=0)
    w[2] *= W3
    return w.reshape(-1)


def s3_orbit_pivots(D, n):
    """One column index per orbit of the qutrit permutations on the dictionary."""
    keys = {ray_key(D[:, i]): i for i in range(D.shape[1])}
    reps = []
    seen = set()
    for i in range(D.shape[1]):
        if i in seen:
            continue
        orbit = {keys[ray_key(permute_qutrits(D[:, i], perm, n))]
                 for perm in itertools.permutations(range(n))}
        seen |= orbit
        reps.append(min(orbit))
    return np.array(reps), keys


def all_rank3(psi, D, n, pivots=None, keys=None):
    res = rank3_search(psi, D, workers=1, pivots=pivots)
    assert not res["rank2"], res["rank2"]
    decs = {tuple(sorted(idx)) for idx, _ in res["found"]}
    if pivots is not None:
        # close under the permutations, which fix psi
        closed = set(decs)
        for idx in decs:
            for perm in itertools.permutations(range(n)):
                closed.add(tuple(sorted(keys[ray_key(permute_qutrits(D[:, i], perm, n))] for i in idx)))
        decs = closed
    return [[D[:, i] for i in idx] for idx in sorted(decs)]


def board_sector_terms():
    """The six four-qutrit carry terms of each sector of bounds/T3-m5-upper-18.json."""
    sub = json.load(open(os.path.join(ROOT, "bounds", "T3-m5-upper-18.json")))
    by_sector = {0: [], 1: [], 2: []}
    for t in sub["witness"]["terms"]:
        v = term_vector(t, 3, 5)
        supp = digits(5)[np.abs(v) > 1e-9]
        sums = set((supp.sum(axis=1) % 3).tolist())
        assert len(sums) == 1, "a board term crosses sectors"
        s = sums.pop()
        by_sector[s].append(compress(v, 5, s))
    for s in range(3):
        psi = carry(5, s)
        A = np.stack(by_sector[s], axis=1)
        c, *_ = np.linalg.lstsq(A, psi, rcond=None)
        assert np.linalg.norm(A @ c - psi) < 1e-9
    return by_sector


def rank5_in_pool(psi, pool, first_pivots, budget):
    """5-subsets of the pool spanning psi, with the first pivot restricted to
    `first_pivots` (one per S_4 orbit of the pool) and the second over the
    whole pool; stops at the time budget and reports coverage."""
    P = np.stack(pool, axis=1)
    n = P.shape[1]
    hits, done, t0 = [], 0, time.time()
    total = len(first_pivots) * (n - 1)
    for a in first_pivots:
        for b in range(n):
            if b == a:
                continue
            if time.time() - t0 > budget:
                return hits, done, total
            Qab, _ = np.linalg.qr(P[:, [a, b]])
            psi2 = psi - Qab @ (Qab.conj().T @ psi)
            done += 1
            if np.linalg.norm(psi2) < 1e-9:
                hits.append(((a, b), "rank2"))
                continue
            D2 = P - Qab @ (Qab.conj().T @ P)
            nrm = np.linalg.norm(D2, axis=0)
            keep = np.flatnonzero(nrm > 1e-9)
            keep = keep[(keep != a) & (keep != b)]
            D2 = D2[:, keep] / nrm[keep]
            # distinct rays only: rank3_search assumes no two dictionary states are parallel
            uniq = []
            seen = set()
            for i in range(D2.shape[1]):
                kk = ray_key(D2[:, i])
                if kk not in seen:
                    seen.add(kk)
                    uniq.append(i)
            uniq = np.array(uniq)
            res = rank3_search(psi2, D2[:, uniq], workers=1)
            cands = [tuple(int(keep[uniq[i]]) for i in idx) for idx, _ in res["found"]]
            cands += [tuple(int(keep[uniq[i]]) for i in r) for r in res["rank2"] if r[0] != "RANK1"]
            for cols in cands:
                Afull = P[:, [a, b] + list(cols)]
                x, *_ = np.linalg.lstsq(Afull, psi, rcond=None)
                if np.linalg.norm(Afull @ x - psi) < 1e-9:
                    hits.append(((a, b), cols))
    return hits, done, total


def main(argv):
    budget = 150.0
    if "--budget" in argv:
        budget = float(argv[argv.index("--budget") + 1])
    sectors = [int(argv[argv.index("--sector") + 1])] if "--sector" in argv else [0, 1, 2]
    skip = int(argv[argv.index("--skip") + 1]) if "--skip" in argv else 0
    do_rank5 = "--rank5" in argv
    t0 = time.time()
    D2 = dictionary(3, 2)
    D3 = dictionary(3, 3)
    print(f"dictionaries: {D2.shape[1]} two-qutrit, {D3.shape[1]} three-qutrit states", flush=True)
    dec3 = {s: all_rank3(carry(3, s), D2, 2) for s in range(3)}
    piv, keys = s3_orbit_pivots(D3, 3)
    print(f"  S_3 orbit representatives on the three-qutrit dictionary: {len(piv)}", flush=True)
    dec4 = {0: all_rank3(carry(4, 0), D3, 3, pivots=piv, keys=keys)}
    for s in (2, 1):
        prev = dec4[(s + 1) % 3]
        cand = [[step_down(v) for v in dec] for dec in prev]
        psi = carry(4, s)
        for dec in cand:
            A = np.stack(dec, axis=1)
            c, *_ = np.linalg.lstsq(A, psi, rcond=None)
            assert np.linalg.norm(A @ c - psi) < 1e-9, "shift map failed"
        dec4[s] = cand
    for s in range(3):
        print(f"  sector {s}: {len(dec3[s])} rank-3 decompositions of psi^(3), "
              f"{len(dec4[s])} of psi^(4)", flush=True)
    print(f"[{time.time() - t0:.0f}s]", flush=True)

    for j in range(3):
        for k in range(3):
            out = contract(sector(3, j), sector(4, k), 3, 4)
            tgt = sector(5, (j + k) % 3)
            ov = abs(np.vdot(tgt, out)) / np.linalg.norm(out)
            assert abs(ov - 1) < 1e-9, (j, k, ov)
    print("  <Phi| contraction maps c_j^(3) (x) c_k^(4) onto c_(j+k)^(5) for all j, k", flush=True)

    board = board_sector_terms()
    for s in sectors:
        psi = carry(5, s)
        nine_sets, pool = [], []
        for j in range(3):
            k = (s - j) % 3
            for A in dec3[j]:
                for B in dec4[k]:
                    terms = []
                    for a in A:
                        for b in B:
                            v = contract(lift(a, 3, j), lift(b, 4, k), 3, 4)
                            if np.linalg.norm(v) < 1e-9:
                                continue
                            term_from_vector(v / np.linalg.norm(v), 3, 5)   # raises if not stabilizer
                            terms.append(compress(v, 5, s))
                    M = np.stack(terms, axis=1)
                    c, *_ = np.linalg.lstsq(M, psi, rcond=None)
                    assert np.linalg.norm(M @ c - psi) < 1e-9
                    nine_sets.append((j, k, terms, np.linalg.matrix_rank(M, tol=1e-8)))
                    pool += terms
        pool = distinct(pool)
        ranks = sorted({r for *_, r in nine_sets})
        sizes = sorted({len(t) for _, _, t, _ in nine_sets})
        print(f"sector {s}: {len(nine_sets)} contractions, term counts {sizes}, span ranks {ranks}; "
              f"pool of {len(pool)} distinct four-qutrit carry states, span rank "
              f"{np.linalg.matrix_rank(np.stack(pool, axis=1), tol=1e-8)}", flush=True)
        bkeys = {ray_key(v) for v in board[s]}
        shared = sum(ray_key(v) in bkeys for v in pool)
        print(f"  board's six-term decomposition of sector {s} shares {shared} states with the pool",
              flush=True)
        # close under S_4 and record orbit representatives
        closed = {}
        reps = []
        for v in pool:
            k0 = ray_key(v)
            if k0 in closed:
                continue
            reps.append(len(closed))
            for perm in itertools.permutations(range(4)):
                w = permute_qutrits(v, perm, 4)
                kw = ray_key(w)
                if kw not in closed:
                    closed[kw] = w
        pool = list(closed.values())
        A = np.stack(pool, axis=1)
        print(f"  S_4 closure of the pool: {len(pool)} states in {len(reps)} orbits, span rank "
              f"{np.linalg.matrix_rank(A, tol=1e-8)}", flush=True)
        bshared = sum(ray_key(v) in {ray_key(w) for w in pool} for v in board[s])
        print(f"  board's six terms inside the closed pool: {bshared}", flush=True)
        if do_rank5:
            t1 = time.time()
            hits, done, total = rank5_in_pool(psi, pool, reps, budget)
            print(f"  exact rank-5 pivot-pair search inside the closed pool: {len(hits)} verified hits, "
                  f"{done} of {total} pivot pairs scanned [{time.time() - t1:.0f}s]", flush=True)
            for h in hits[:10]:
                print("   ", h)
        # completion: five-term decompositions sharing four terms with a contraction
        t1 = time.time()
        found5, tested = 0, set()
        stopped = False
        n_subsets = 0
        for j, k, terms, _ in nine_sets:
            for sub_idx in itertools.combinations(range(len(terms)), 4):
                key = tuple(sorted(ray_key(terms[i]) for i in sub_idx))
                if key in tested:
                    continue
                tested.add(key)
                n_subsets += 1
                if n_subsets <= skip:
                    continue
                if time.time() - t1 > budget:
                    stopped = True
                    break
                kept = np.stack([terms[i] for i in sub_idx], axis=1)
                V = np.column_stack([kept, psi])
                if np.linalg.matrix_rank(V, tol=1e-8) < 5:
                    continue
                states, _ = stabilizer_states_in_span(V, 3, 4)
                Qk, _ = np.linalg.qr(kept)
                new = [v for v, _t in states
                       if np.linalg.norm(v - Qk @ (Qk.conj().T @ v)) > 1e-6 * np.linalg.norm(v)]
                found5 += len(new)
            if stopped:
                break
        # count the distinct four-subsets in full, so coverage is reported against the total
        allkeys = {tuple(sorted(ray_key(terms[i]) for i in sub_idx))
                   for _, _, terms, _ in nine_sets for sub_idx in itertools.combinations(range(9), 4)}
        print(f"  completions: {found5} five-term decompositions sharing four contraction terms; "
              f"distinct four-subsets {skip + 1} to {n_subsets if stopped else len(allkeys)} of "
              f"{len(allkeys)} tested{' (budget hit)' if stopped else ' (complete)'} "
              f"[{time.time() - t1:.0f}s]", flush=True)
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
