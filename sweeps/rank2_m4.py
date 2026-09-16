"""Rank-2 exclusion at m=4 for the qutrit orbits, without holding the states.

H3, N and T3 at m=4 have no lower bound at all, so each is still a route to the
published exponent. Excluding rank 2 gives chi >= 3 for all three.

The obstacle is memory, not time: there are 7,439,040 stabilizer states on four
qutrits and holding them is 9.6 GB, which enumerate_stabilizer_states doubles by
building a Python list before stacking. So nothing is held. psi lies in the span
of two stabilizer states exactly when those two have parallel components off
psi, and parallel vectors have *identical* canonical keys: with u = q/||q|| and
key = (KEY.u)/(CAN.u), scaling q by lambda scales both numerator and denominator
by lambda/|lambda|. So the key is stored (16 bytes) and the state discarded.

That direction is the one that matters. Equal keys are necessary for parallel,
so a pair that is genuinely parallel cannot be missed; two distinct directions
sharing a key would only add a candidate, and the minimum gap reported below
says whether any came close. No false negatives is the property a lower bound
needs.

The raw enumeration repeats each state under several parametrisations (414
triples give 360 distinct states at m=2), and a repeat has the same quotient
direction and therefore the same key. A rank-2 decomposition needs two DISTINCT
states, so repeats are removed first. Dividing a qutrit stabilizer state by its
first nonzero entry puts every entry in {0, 1, w3, w3^2}, which are separated by
more than 1.7, so the canonical form is exact rather than a rounding. States are
identified by a 128-bit digest of it: at 7.4 million states a 64-bit digest would
carry a one-in-a-million collision chance, and a collision here would silently
merge two distinct states, which is the direction that loses a lower bound.

Positive control: the Strange state at m=2 has a known rank-2 decomposition, so
the same code must find its keys colliding.
"""

import argparse
import hashlib
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank.stabilizer_extent import _tuple_to_index
from stabrank_verify import target_vector


def stream_states(n, d=3):
    """Every stabilizer state on n qudits, one at a time, never all at once."""
    dim = d ** n
    for k in range(n + 1):
        for pivots in itertools.combinations(range(n), k):
            non_pivots = [j for j in range(n) if j not in pivots]
            n_free = k * len(non_pivots)
            for free_vals in itertools.product(range(d), repeat=n_free):
                W = np.zeros((k, n), dtype=np.int64)
                idx_f = 0
                for row_i in range(k):
                    W[row_i, pivots[row_i]] = 1
                    for col_j in non_pivots:
                        W[row_i, col_j] = free_vals[idx_f]
                        idx_f += 1
                for x0_free in itertools.product(range(d), repeat=n - k):
                    x0 = np.zeros(n, dtype=np.int64)
                    for i, col in enumerate(non_pivots):
                        x0[col] = x0_free[i]
                    n_mix = k * (k - 1) // 2
                    n_phase = k + (k if d >= 3 else 0) + n_mix
                    ys = list(itertools.product(range(d), repeat=k))
                    xs = [_tuple_to_index(tuple((x0 + W.T @ np.array(y)) % d), d)
                          for y in ys]
                    for phase_vals in itertools.product(range(d), repeat=n_phase):
                        c_lin = phase_vals[:k]
                        c_sq = phase_vals[k:2 * k] if d >= 3 else [0] * k
                        c_mix = phase_vals[(2 * k if d >= 3 else k):]
                        state = np.zeros(dim, dtype=complex)
                        for y, xi in zip(ys, xs):
                            q = 0.0
                            for i in range(k):
                                q += c_lin[i] * y[i] / d
                                if d >= 3:
                                    q += c_sq[i] * y[i] * y[i] / d
                            mi = 0
                            for s_ in range(k):
                                for t_ in range(s_ + 1, k):
                                    q += c_mix[mi] * y[s_] * y[t_] / d
                                    mi += 1
                            state[xi] = np.exp(2j * np.pi * q)
                        nrm = np.linalg.norm(state)
                        if nrm > 1e-12:
                            yield state / nrm


def keys_for(psis, n, d, report_every=500000):
    """Canonical quotient keys for every stabilizer state, per target."""
    rng = np.random.default_rng(7)
    dim = d ** n
    can = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    key = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    out = [[] for _ in psis]
    digests = []
    rank1 = [False] * len(psis)
    t0, count = time.time(), 0
    for s in stream_states(n, d):
        count += 1
        nz = np.flatnonzero(np.abs(s) > 1e-9)
        canon = np.round(s / s[nz[0]], 6) + 0.0
        digests.append(hashlib.blake2b(canon.tobytes(), digest_size=16).digest())
        for j, psi in enumerate(psis):
            q = s - psi * np.vdot(psi, s)
            nq = np.linalg.norm(q)
            if nq <= 1e-9:
                rank1[j] = True
                continue
            u = q / nq
            c = can @ u
            if abs(c) <= 1e-9:
                continue
            out[j].append((key @ u) / c)
        if count % report_every == 0:
            print(f"    {count:,} states, {time.time()-t0:.0f}s", flush=True)
    print(f"    {count:,} states total in {time.time()-t0:.0f}s", flush=True)
    dig = np.array(digests, dtype="S16")
    _, keep = np.unique(dig, return_index=True)
    print(f"    {len(keep):,} distinct up to phase", flush=True)
    return [np.array(o)[keep] for o in out], rank1, count, len(keep)


def min_gap(k):
    """Smallest distance between any two keys, via one sort."""
    o = np.argsort(k)
    ks = k[o]
    d = np.abs(ks[1:] - ks[:-1])
    i = int(np.argmin(d))
    return float(d[i]), (int(o[i]), int(o[i + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--orbits", default="T3,N,H3")
    ap.add_argument("--control", action="store_true", help="run the m=2 Strange control")
    a = ap.parse_args()

    if a.control:
        print("positive control: Strange at m=2 (a rank-2 decomposition is known)")
        psi = np.array([complex(x) for x in target_vector("S", 2)]).ravel()
        psi /= np.linalg.norm(psi)
        ks, r1, n, nd = keys_for([psi], 2, 3)
        g, pair = min_gap(ks[0])
        verdict = "keys collide, as they must" if g < 1e-9 else "NO COLLISION -- search is broken"
        print(f"  {n} states ({nd} distinct), minimum key gap {g:.3e} at {pair}: {verdict}\n")
        if g >= 1e-9:
            return 1

    orbits = a.orbits.split(",")
    psis = []
    for o in orbits:
        p = np.array([complex(x) for x in target_vector(o, a.m)]).ravel()
        psis.append(p / np.linalg.norm(p))
    print(f"m={a.m}, orbits {orbits}")
    ks, rank1, n, nd = keys_for(psis, a.m, 3)
    ok = True
    for o, k, r1 in zip(orbits, ks, rank1):
        if r1:
            print(f"{o} m={a.m}: target IS a stabilizer state (rank 1)")
            ok = False
            continue
        g, pair = min_gap(k)
        if g < 1e-9:
            print(f"{o} m={a.m}: candidate parallel pair {pair}, gap {g:.3e} -- "
                  f"needs exact confirmation, NOT excluded")
            ok = False
        else:
            print(f"{o} m={a.m}: no two of {len(k):,} keys within {g:.3e}; "
                  f"no rank-2 decomposition")
            print(f"CERTIFIED chi({o}^{a.m}) >= 3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
