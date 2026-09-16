"""Exhaustive rank-2 and rank-3 exclusion by quotienting out the target.

A lower bound chi(psi) >= r + 1 says that no r stabilizer states span psi.
For r = 2 and r = 3 the whole dictionary of stabilizer states is small enough
to settle this by enumeration, if the enumeration is organised so that the
work is linear or quadratic in the dictionary rather than a binomial in it.

The organising fact is one quotient at a time. If psi lies in
span(s_1, ..., s_r) then the images of the s_j in C^d / span(psi) are linearly
dependent. For r = 2 that is two parallel directions, found by canonicalising
every direction and sorting once. For r = 3, fix the first state s_i as a pivot
and quotient by span(psi, s_i) as well: if psi is in span(s_i, s_j, s_k) then
the images of s_j and s_k in that second quotient are parallel, which is the
same sort, done once per pivot. The converse fails only when the three states
are themselves dependent, which is why every candidate is retested in the full
space. Every triple has a smallest index, so pivots only look at states with a
larger index and each triple is examined once.

Sorting finds parallel directions through a key that is constant on a
projective point: with u the unit direction, key = (KEY.u)/(CAN.u) for two fixed
generic functionals is unchanged by u -> lambda u. Parallel directions get equal
keys and land adjacent after sorting. The key is not injective on projective
points, though: it is constant on a whole hyperplane of directions, so an
unrelated direction can share a key exactly and sit between two parallel ones
in the sort order. Comparing only adjacent entries would then split the pair.
So the sorted keys are cut into runs wherever consecutive keys differ by more
than a relative 1e-8, and every pair inside a run is tested by its overlap.
Parallel directions have keys equal to rounding and always share a run; an
intruder can only enlarge a run, never separate its members. Runs are tiny
in practice, so the all-pairs test inside them costs nothing.

Before the sort, directions are pushed through a random linear map to a few
dimensions. A linear map preserves collinearity, so a genuine dependency is
still a dependency after projecting and the search cannot lose one; the map
can only propose extra candidates, and every candidate is then tested in the
full space by least squares. This is the same safeguard the T3 m=3 certificate
in this repository documents, and the projected dimension is kept well above
the value at which that certificate saw accidental near-parallel pairs.

Everything here is numerical and rests on a margin: the closest any two
non-parallel adjacent directions come to the parallel threshold is returned,
and a certificate should refuse to print its claim when that margin is thin.
The dictionaries themselves are exact up to rounding at 1e-16.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARALLEL = 1e-6        # |<u, v>| above 1 - PARALLEL counts as parallel
MARGIN = 0.01          # an exclusion must clear the threshold by at least this
PROJ_DIM = 10          # random projection dimension for the per-pivot sort
RESID = 1e-9           # a candidate spans psi if its residual is below this


def dictionary(p, n):
    """Every n-qudit stabilizer state for p in {2, 3}, one column each,
    distinct up to phase and unit norm."""
    if p == 2:
        from qubit_states import all_states
        D = all_states(n)          # asserts the count itself
    elif p == 3:
        from stabrank.examples.t3_galois_lower_bound import distinct_states
        D = distinct_states(n)
    else:
        raise ValueError("dictionaries are available for p = 2 and p = 3")
    want = p ** n
    for j in range(1, n + 1):
        want *= p ** j + 1
    if D.shape[1] != want:
        # Every exclusion is only as sound as the dictionary is complete.
        raise AssertionError(f"{D.shape[1]} states on {n} qudits of dimension {p}, "
                             f"expected {want}")
    return D / np.linalg.norm(D, axis=0)


def _orth(cols):
    """Orthonormal basis of the span of the given columns."""
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q


def _parallel_groups(U, rng):
    """Groups of mutually parallel unit columns of U, and the closest approach
    to parallel among adjacent non-parallel pairs.

    Returns (groups, worst): groups is a list of index arrays each of size >= 2,
    worst is the largest adjacent overlap that was not counted as parallel.
    """
    n = U.shape[1]
    if n < 2:
        return [], 0.0
    for _attempt in range(8):
        can = rng.normal(size=U.shape[0]) + 1j * rng.normal(size=U.shape[0])
        key = rng.normal(size=U.shape[0]) + 1j * rng.normal(size=U.shape[0])
        c = can @ U
        # A direction the canonicalising functional nearly annihilates has no
        # usable key. Dropping it would silently remove it from every pair,
        # so redraw the functionals instead; a generic draw never hits this.
        if (np.abs(c) > 1e-9).all():
            break
    else:
        raise RuntimeError("could not draw a canonicalising functional that is "
                           "nonzero on every direction")
    kk = (key @ U) / c
    order = np.lexsort((kk.imag, kk.real))     # complex argsort is far slower
    ks = kk[order]
    # Cut the sorted keys into runs of (near-)equal key; parallel directions
    # share a run, and only pairs inside a run can be parallel.
    gap = np.abs(np.diff(ks)) > 1e-8 * (1 + np.abs(ks[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(gap) + 1, [n]))
    # Adjacent overlaps across the whole order: the margin report says how
    # close any two neighbouring directions come to the parallel threshold.
    ov_adj = np.abs(np.sum(U[:, order[:-1]].conj() * U[:, order[1:]], axis=0))
    worst = float(ov_adj[ov_adj <= 1 - PARALLEL].max()) if (ov_adj <= 1 - PARALLEL).any() else 0.0
    groups = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        members = order[a:b]
        V = U[:, members]
        G = np.abs(V.conj().T @ V)
        np.fill_diagonal(G, 0)
        par = G > 1 - PARALLEL
        off = G[~par & (G > 0)]
        if off.size:
            worst = max(worst, float(off.max()))
        # connected components of the parallel relation inside the run
        seen = np.zeros(b - a, bool)
        for s in range(b - a):
            if seen[s]:
                continue
            comp = [s]
            seen[s] = True
            stack = [s]
            while stack:
                v = stack.pop()
                for w in np.flatnonzero(par[v]):
                    if not seen[w]:
                        seen[w] = True
                        comp.append(w)
                        stack.append(w)
            if len(comp) >= 2:
                groups.append(members[comp])
    return groups, worst


def rank2_search(psi, D, seed=7):
    """Pairs (i, j) with psi in span(s_i, s_j), and the closest approach.

    Returns ("RANK1", 1.0) if some state is parallel to psi.
    """
    psi = psi / np.linalg.norm(psi)
    q = D - np.outer(psi, psi.conj() @ D)
    nq = np.linalg.norm(q, axis=0)
    if (nq <= 1e-9).any():
        return "RANK1", 1.0
    groups, worst = _parallel_groups(q / nq, np.random.default_rng(seed))
    pairs = [(int(g[a]), int(g[b])) for g in groups
             for a in range(len(g)) for b in range(a + 1, len(g))]
    return pairs, worst


class _Rank3Worker:
    """Per-process state for the pivot loop; built once per worker."""

    def __init__(self, psi, D, seed):
        self.psi = psi / np.linalg.norm(psi)
        self.D = D
        P = D - np.outer(self.psi, self.psi.conj() @ D)
        self.nP = np.linalg.norm(P, axis=0)
        self.P = P / np.where(self.nP > 0, self.nP, 1)
        rng = np.random.default_rng(seed)
        self.R = (rng.normal(size=(PROJ_DIM, D.shape[0]))
                  + 1j * rng.normal(size=(PROJ_DIM, D.shape[0])))
        self.RP = self.R @ self.P
        self.rng = np.random.default_rng(seed + 1)

    def pivots(self, lo, hi):
        cands, worst, rank2 = [], 0.0, []
        for i in range(lo, hi):
            pi = self.P[:, i]
            rest = slice(i + 1, self.P.shape[1])
            ov = pi.conj() @ self.P[:, rest]
            q = self.RP[:, rest] - np.outer(self.R @ pi, ov)
            # full-space norm of the quotient direction, for the zero test
            nq_full = np.sqrt(np.maximum(1 - np.abs(ov) ** 2, 0))
            zero = nq_full < 1e-7
            if zero.any():
                rank2 += [(i, int(i + 1 + j)) for j in np.flatnonzero(zero)]
            keep = ~zero
            nq = np.linalg.norm(q, axis=0)
            if (nq[keep] <= 1e-12).any():
                # A nonzero quotient direction that the random projection maps
                # to (nearly) zero: probability zero for a generic map, and if
                # it happens the search must not silently proceed without it.
                raise RuntimeError("random projection annihilated a direction; "
                                   "rerun with another seed")
            U = q[:, keep] / nq[keep]
            ids = np.flatnonzero(keep) + i + 1
            groups, w = _parallel_groups(U, self.rng)
            worst = max(worst, w)
            for g in groups:
                g = ids[g]
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        cands.append((i, int(g[a]), int(g[b])))
        return cands, worst, rank2


_WORKER = None


def _init(psi, D, seed):
    global _WORKER
    _WORKER = _Rank3Worker(psi, D, seed)


def _run(rng):
    return _WORKER.pivots(*rng)


def rank3_search(psi, D, workers=None, seed=7):
    """Triples spanning psi, with diagnostics.

    Returns a dict with keys
      found:      list of (indices, coefficients) for triples that span psi
      rank2:      pairs found to span psi along the way (should be empty
                  when rank 2 has already been excluded)
      candidates: number of collinear-in-quotient triples tested
      min_resid:  smallest least-squares residual among candidates that
                  were not counted as decompositions
      worst:      closest approach to parallel among non-parallel directions
    Zero-norm quotient directions at the first stage (a state parallel to psi)
    are reported through rank2 as well, as ("RANK1", j).
    """
    N = D.shape[1]
    psi = psi / np.linalg.norm(psi)
    P = D - np.outer(psi, psi.conj() @ D)
    nP = np.linalg.norm(P, axis=0)
    out = {"found": [], "rank2": [], "candidates": 0, "min_resid": np.inf, "worst": 0.0}
    if (nP <= 1e-9).any():
        out["rank2"] = [("RANK1", int(j)) for j in np.flatnonzero(nP <= 1e-9)]
        return out
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    # Balance work: pivot i costs N - i, so split by area rather than by count.
    cuts = [0]
    total = N * (N - 1) / 2
    for k in range(1, workers):
        target = total * k / workers
        i = int(N - np.sqrt(max(N * N - 2 * target, 0)))
        cuts.append(min(max(i, cuts[-1]), N))
    cuts.append(N)
    ranges = [(cuts[k], cuts[k + 1]) for k in range(workers) if cuts[k] < cuts[k + 1]]
    cands, rank2 = [], []
    if workers == 1:
        _init(psi, D, seed)
        for r in ranges:
            c, w, r2 = _run(r)
            cands += c; rank2 += r2; out["worst"] = max(out["worst"], w)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(psi, D, seed)) as ex:
            for c, w, r2 in ex.map(_run, ranges):
                cands += c; rank2 += r2; out["worst"] = max(out["worst"], w)
    out["rank2"] = rank2
    out["candidates"] = len(cands)
    out["dependent"] = 0
    if rank2:
        return out
    # Candidates are tested in the full space, vectorised over batches by
    # Gram-Schmidt. Most candidates are three stabilizer states that are
    # linearly dependent among themselves (three states in one two-dimensional
    # stabilizer subspace), which look collinear in every quotient. Their span
    # is two-dimensional, and rank 2 has just been excluded above (rank2 is
    # empty), so they cannot contain psi and are counted rather than solved.
    # For the independent ones the residual of psi against the span is exact
    # up to rounding, and anything that passes is re-solved alone.
    cand = np.array(cands, dtype=np.int64).reshape(-1, 3)
    for lo in range(0, len(cand), 200000):
        idx = cand[lo:lo + 200000]
        s1, s2, s3 = (D[:, idx[:, k]].T for k in range(3))     # (batch, dim)
        u1 = s1
        v2 = s2 - np.sum(u1.conj() * s2, axis=1)[:, None] * u1
        n2 = np.linalg.norm(v2, axis=1)
        good = n2 > 1e-9
        u2 = v2 / np.where(good, n2, 1)[:, None]
        v3 = s3 - np.sum(u1.conj() * s3, axis=1)[:, None] * u1 \
                - np.sum(u2.conj() * s3, axis=1)[:, None] * u2
        n3 = np.linalg.norm(v3, axis=1)
        indep = good & (n3 > 1e-9)
        out["dependent"] += int((~indep).sum())
        u3 = v3 / np.where(indep, n3, 1)[:, None]
        r = psi[None, :] - sum(np.sum(u.conj() * psi[None, :], axis=1)[:, None] * u
                               for u in (u1, u2, u3))
        res = np.linalg.norm(r, axis=1)
        res = np.where(indep, res, np.inf)
        hit = res < RESID
        for cols in idx[hit]:
            A1 = D[:, cols]
            x, *_ = np.linalg.lstsq(A1, psi, rcond=None)
            r1 = float(np.linalg.norm(A1 @ x - psi))
            if r1 < RESID:
                out["found"].append(([int(c) for c in cols], x))
            else:
                out["min_resid"] = min(out["min_resid"], r1)
        rest = res[indep & ~hit]
        if rest.size:
            out["min_resid"] = min(out["min_resid"], float(rest.min()))
    return out


def psi_for(orbit, m):
    from stabrank_verify import target_vector
    return np.array([complex(x) for x in target_vector(orbit, m)]).ravel()


def certify_rank3(orbit, m, D, label=None, workers=None):
    """Run the rank-3 exclusion for one cell and print a report.

    Returns True when no rank <= 3 decomposition exists and the margin is
    adequate, False otherwise. The caller decides what claim to print.
    """
    label = label or f"{orbit} m={m}"
    psi = psi_for(orbit, m)
    r = rank3_search(psi, D, workers=workers)
    if r["rank2"]:
        print(f"{label}: rank <= 2 decomposition FOUND via {r['rank2'][:3]}")
        return False
    if r["found"]:
        cols, x = r["found"][0]
        print(f"{label}: rank-3 decomposition FOUND, states {cols}, "
              f"coefficients {np.round(x, 6)}")
        return False
    slack = (1 - PARALLEL) - r["worst"]
    indep = r["candidates"] - r["dependent"]
    resid = (f" (smallest residual {r['min_resid']:.4f})"
             if np.isfinite(r["min_resid"]) else "")
    print(f"{label}: {r['candidates']} collinear candidates, of which "
          f"{r['dependent']} are three states of one two-dimensional span and "
          f"{indep} are independent triples; none spans the target{resid}. "
          f"Closest approach to parallel {r['worst']:.6f}, clearing the "
          f"threshold by {slack:.4f}.")
    if slack < MARGIN:
        print(f"{label}: margin too thin to stand behind", file=sys.stderr)
        return False
    return True


def control_rank3(orbit, m, D, label=None):
    """A cell with a known rank-3 decomposition; the search must find it."""
    label = label or f"{orbit} m={m}"
    r = rank3_search(psi_for(orbit, m), D)
    if not r["found"]:
        print(f"positive control FAILED: no rank-3 decomposition found for "
              f"{label}, where one is known; the search is broken", file=sys.stderr)
        return False
    print(f"positive control: {label} found {len(r['found'])} rank-3 "
          f"decomposition(s), as expected")
    return True


def run_certificate(cells, controls, workers=None, also=()):
    """Drive a certificate script: controls first, then each cell, and print
    one CERTIFIED line per cell that survives.

    cells:    (orbit, m) pairs whose rank-3 exclusion is the claim, printed as
              CERTIFIED chi(orbit^m) >= 4
    controls: (orbit, m) pairs with a known rank-3 decomposition; the search
              must find each or the script exits non-zero without claiming.
    also:     extra (orbit, m, rank) claims that follow from the cells by the
              projection lemma (see cert_*_from_*.py), printed only when every
              cell passed.

    Returns a process exit code.
    """
    dicts = {}

    def dict_for(orbit, m):
        from stabrank_verify import ORBIT_P
        key = (ORBIT_P[orbit], m)
        if key not in dicts:
            dicts[key] = dictionary(*key)
            print(f"{m} qu{'bit' if key[0] == 2 else 'trit'}s: "
                  f"{dicts[key].shape[1]} stabilizer states")
        return dicts[key]

    for orbit, m in controls:
        if not control_rank3(orbit, m, dict_for(orbit, m)):
            return 1
    print()
    ok = True
    for orbit, m in cells:
        if not certify_rank3(orbit, m, dict_for(orbit, m), workers=workers):
            ok = False
            continue
        print(f"CERTIFIED chi({orbit}^{m}) >= 4")
    if ok:
        for orbit, m, rank in also:
            print(f"CERTIFIED chi({orbit}^{m}) >= {rank}")
    return 0 if ok else 1
