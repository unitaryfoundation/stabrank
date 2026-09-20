"""Strict slice-and-lift of one known decomposition, for ranks the
row-table matching of slice_lift.py cannot hold in memory.

A rank-r decomposition of phi^(m+1) with r = chi(phi^m) has every slice a
minimal decomposition of phi^m, all r terms full (slice_lift.py). Taking a
known rank-r decomposition (d_i, u_i) of phi^m as slice b, the lift exists
iff there are Pauli classes Q_i and cube roots mu_i, nu_i with

    sum_i d_i mu_i Q_i u_i      = (alpha_{b+1} / alpha_b) phi^m,
    sum_i d_i nu_i Q_i^-1 u_i   = (alpha_{b+2} / alpha_b) phi^m.

The first equation is matched by meet in the middle on scalar keys: two
random linear functionals of the row sums, so the left half (3^m classes
times 3 phases per term, r/2 terms) and the right half are (3^(m+1))^(r/2)
complex numbers each rather than row tables. Every key coincidence within
the window is checked on the full vectors, so the keys can only add work,
never lose a match. For each match the second equation is a linear system
in nu, solved and checked to have cube-root entries.

This tests lifts of one decomposition (and its orbit under I (x) U, which
preserves slices), not of every rank-r decomposition of phi^m, so a null
result here is not a lower bound.

Usage: strict_lift_big.py ORBIT M BOUND_OR_DATA [--base 0,1,2]
   BOUND_OR_DATA is a bounds/*.json upper-bound file (its witness is the
   decomposition at M copies) or data/<orbit>_m<M>_rank<r>.json (every
   stored decomposition is tried).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORBIT_P, alpha, target, term_vector  # noqa: E402
from slice_lift import pauli_images  # noqa: E402

W3 = np.exp(2j * np.pi / 3)
CUBE = np.array([1, W3, W3 ** 2])
WINDOW = 1e-6


def scalar_keys(rows, f):
    """rows: (n, dim) -> (n,) complex keys."""
    return rows @ f


def sum_keys(key_lists):
    """All sums of one key from each list (mixed radix, first list slowest)."""
    S = key_lists[0]
    for K in key_lists[1:]:
        S = (S[:, None] + K[None, :]).ravel()
    return S


def decode(idx, sizes):
    out = []
    for n in reversed(sizes):
        out.append(idx % n)
        idx //= n
    return list(reversed(out))


def mitm(opts, rhs, log):
    """Assignments (one row per term) with rows summing to rhs, by scalar-key
    meet in the middle on two random functionals."""
    r = len(opts)
    sizes = [len(o) for o in opts]
    half = r // 2
    rng = np.random.default_rng(17)
    dim = len(rhs)
    f1 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    f2 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    kL1 = sum_keys([o @ f1 for o in opts[:half]])
    kL2 = sum_keys([o @ f2 for o in opts[:half]])
    tgt1, tgt2 = rhs @ f1, rhs @ f2
    order = np.argsort(kL1.real, kind="stable")
    sL = kL1.real[order]
    log(f"    left table {len(kL1)} keys sorted")
    kR1 = sum_keys([o @ f1 for o in opts[half:]])
    kR2 = sum_keys([o @ f2 for o in opts[half:]])
    want1 = tgt1 - kR1              # need kL1 == want1
    want2 = tgt2 - kR2
    found = []
    n_cand = 0
    step = 1 << 22
    for lo_i in range(0, len(want1), step):
        w1 = want1[lo_i:lo_i + step]
        lo = np.searchsorted(sL, w1.real - WINDOW, side="left")
        hi = np.searchsorted(sL, w1.real + WINDOW, side="right")
        rows = np.flatnonzero(hi > lo)
        if not len(rows):
            continue
        counts = hi[rows] - lo[rows]
        rs = np.repeat(rows, counts)
        starts = np.repeat(lo[rows], counts)
        offs = np.arange(len(rs)) - np.repeat(np.cumsum(counts) - counts, counts)
        ls = order[starts + offs]
        rs_glob = rs + lo_i
        ok = (np.abs(kL1[ls].imag - want1[rs_glob].imag) <= WINDOW) & \
             (np.abs(kL2[ls] - want2[rs_glob]) <= 10 * WINDOW)
        ls, rs_glob = ls[ok], rs_glob[ok]
        n_cand += len(ls)
        for l, rr in zip(ls, rs_glob):
            cl = decode(int(l), sizes[:half])
            cr = decode(int(rr), sizes[half:])
            combo = cl + cr
            v = sum(opts[i][c] for i, c in enumerate(combo))
            if np.abs(v - rhs).max() < 1e-7:
                found.append(tuple(combo))
    log(f"    {n_cand} key coincidences, {len(found)} full matches")
    return found


def lift_one(orbit, m, u, d, bases, log):
    a = alpha(orbit)
    psi = target(orbit, m)
    r = len(u)
    imgs = [pauli_images(ui, m) for ui in u]
    lifts = []
    for b in bases:
        if abs(a[b]) < 1e-12:
            continue
        f, g = (b + 1) % 3, (b + 2) % 3
        t0 = time.time()
        opts_f = [np.array([d[i] * mu * qu for qu, _ in imgs[i] for mu in CUBE]) for i in range(r)]
        combos = mitm(opts_f, (a[f] / a[b]) * psi, log)
        n_second = 0
        for combo in combos:
            cls = [c // 3 for c in combo]
            Qg = np.column_stack([d[i] * imgs[i][cls[i]][1] for i in range(r)])
            nu, *_ = np.linalg.lstsq(Qg, (a[g] / a[b]) * psi, rcond=None)
            if np.linalg.norm(Qg @ nu - (a[g] / a[b]) * psi) < 1e-7 and \
                    all(min(abs(x - c) for c in CUBE) < 1e-6 for x in nu):
                n_second += 1
                terms = []
                for i in range(r):
                    blocks = [None] * 3
                    blocks[b] = u[i]
                    blocks[f] = CUBE[combo[i] % 3] * imgs[i][cls[i]][0]
                    blocks[g] = nu[i] * imgs[i][cls[i]][1]
                    terms.append(np.concatenate(blocks))
                lifts.append(terms)
        log(f"  base {b}: {len(combos)} matches of the slice-{f} equation, {n_second} also solve "
            f"slice {g} [{time.time() - t0:.0f}s]")
    return lifts


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("orbit")
    ap.add_argument("m", type=int)
    ap.add_argument("source")
    ap.add_argument("--base", default="0,1,2")
    a = ap.parse_args(argv[1:])
    p = ORBIT_P[a.orbit]
    assert p == 3
    psi = target(a.orbit, a.m)
    src = json.load(open(a.source))
    if "witness" in src:
        decs = [src["witness"]["terms"]]
    else:
        decs = src["decompositions"]
    bases = [int(x) for x in a.base.split(",")]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for di, dec in enumerate(decs):
        u = [term_vector(t, 3, a.m) for t in dec]
        U = np.column_stack(u)
        d, *_ = np.linalg.lstsq(U, psi, rcond=None)
        assert np.linalg.norm(U @ d - psi) < 1e-9
        print(f"{a.orbit} m={a.m} -> {a.m + 1}: decomposition {di} of rank {len(u)}", flush=True)
        lifts = lift_one(a.orbit, a.m, u, d, bases, lambda s: print(s, flush=True))
        psi_up = np.kron(alpha(a.orbit), psi)
        for k, terms in enumerate(lifts):
            A = np.column_stack(terms)
            c, *_ = np.linalg.lstsq(A, psi_up, rcond=None)
            res = np.linalg.norm(A @ c - psi_up)
            print(f"  lift {k}: residual {res:.2e}")
            if res < 1e-8:
                total += 1
                path = os.path.join(out_dir, f"strict_{a.orbit}_m{a.m + 1}_rank{len(u)}_dec{di}_hit{k}.json")
                with open(path, "w") as fh:
                    json.dump([[[float(z.real), float(z.imag)] for z in t] for t in terms], fh)
                print(f"  written {path}")
    print(f"{total} lifts of rank {len(u)} at m={a.m + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
