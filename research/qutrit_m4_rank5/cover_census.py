"""p = 3 port of verify_challenge/slice_cover.CoverEnumerator for costing:
full r-covers of |M>^n (M = N or H3) over the n-qutrit dictionary, one per
orbit of the unitary symmetry group, modulo P1 with exact re-checks.

Usage: q3cover.py census ORBIT n [--cap SECONDS] [--pairs K] [--out FILE]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from slice_lift import stabilizer_orbit_labels  # noqa: E402

P1 = 65521
P2 = 2013265921
NUM_TOL = 1e-9
W3 = np.exp(2j * np.pi / 3)


class Field3:
    """Q(omega_3, sqrt 3) reduced modulo a prime p = 1 mod 12."""

    def __init__(self, p):
        assert p % 12 == 1
        self.p = p
        for g in range(2, p):
            w = pow(g, (p - 1) // 3, p)
            if w != 1:
                break
        self.w = w
        assert pow(w, 3, p) == 1 and (w * w + w + 1) % p == 0
        for g in range(2, p):
            i = pow(g, (p - 1) // 4, p)
            if (i * i) % p == p - 1:
                break
        self.i = i
        sm3 = (2 * w + 1) % p                       # sqrt(-3)
        assert (sm3 * sm3) % p == p - 3
        self.sqrt3 = sm3 * pow(i, p - 2, p) % p
        assert (self.sqrt3 ** 2) % p == 3
        self.wpow = np.array([1, w, w * w % p], dtype=np.int64)
        self.inv_table = None
        if p < 1 << 20:
            t = np.zeros(p, dtype=np.int64)
            t[1:] = [pow(int(a), p - 2, p) for a in range(1, p)]
            self.inv_table = t

    def inv(self, a):
        a = np.asarray(a, dtype=np.int64) % self.p
        if self.inv_table is not None:
            return self.inv_table[a]
        return np.vectorize(lambda x: pow(int(x), self.p - 2, self.p), otypes=[np.int64])(a)

    def alpha(self, orbit):
        if orbit == "N":
            return np.array([1, 1, self.p - 2], dtype=np.int64)
        if orbit == "H3":
            return np.array([(1 + self.sqrt3) % self.p, 1, 1], dtype=np.int64)
        raise ValueError(orbit)

    def target(self, orbit, n):
        a = self.alpha(orbit)
        v = a
        for _ in range(n - 1):
            v = np.kron(v, a) % self.p
        return v % self.p

    def codes_to_field(self, codes):
        out = self.wpow[(codes - 1) % 3]
        return np.where(codes > 0, out, 0).astype(np.int64)


def patterns(D):
    """Exact phase codes (0 zero, 1..3 = 1, w, w^2) of the dictionary columns
    with the first nonzero entry made 1, shape (N, dim), and the unnormalised
    complex matrix (dim, N)."""
    dim, N = D.shape
    nz = np.abs(D) > 1e-9
    first = np.argmax(nz, axis=0)
    ref = D[first, np.arange(N)]
    W = D / ref[None, :]
    codes = np.zeros((N, dim), dtype=np.int8)
    for c in range(3):
        codes[np.abs(W.T - W3 ** c) < 1e-6] = c + 1
    assert np.array_equal(codes > 0, nz.T), "a dictionary entry is not a cube root of unity"
    C = np.zeros((dim, N), dtype=complex)
    for c in range(3):
        C[codes.T == c + 1] = W3 ** c
    return codes, C


def _reduce(F, rows, v):
    c = int(np.flatnonzero(v)[0])
    f = (rows[:, c] * F.inv(v[c])) % F.p
    res = (rows - f[:, None] * v[None, :]) % F.p
    return np.delete(res, c, axis=1), c


def _canon_rows(F, R):
    nz = R != 0
    has = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = R[np.arange(len(R)), first]
    scale = np.where(has, F.inv(lead), 0)
    return (R * scale[:, None]) % F.p, has


def _groups_by_key(keys):
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    brk = np.flatnonzero(sk[1:] != sk[:-1]) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [len(sk)]))
    return [order[s:e] for s, e in zip(starts, ends) if e - s >= 2]


def rank_mod(M, p):
    """Rank over F_p of an int64 matrix (numpy, p < 2^31)."""
    A = np.array(M, dtype=np.int64) % p
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        if rank == rows:
            break
        piv = None
        for r in range(rank, rows):
            if A[r, c]:
                piv = r
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        A[rank] = (A[rank] * pow(int(A[rank, c]), p - 2, p)) % p
        for r in range(rows):
            if r != rank and A[r, c]:
                A[r] = (A[r] - A[r, c] * A[rank]) % p
        rank += 1
    return rank


class CoverEnumerator3:
    def __init__(self, orbit, n, D=None, verbose=False, seed=17):
        self.orbit, self.n = orbit, n
        self.D = dictionary(3, n) if D is None else D
        self.N = self.D.shape[1]
        self.codes, self.C = patterns(self.D)
        self.F1, self.F2 = Field3(P1), Field3(P2)
        self.U1 = self.F1.codes_to_field(self.codes)
        self.U2 = self.F2.codes_to_field(self.codes)
        self.psi1, self.psi2 = self.F1.target(orbit, n), self.F2.target(orbit, n)
        self.psi = psi_for(orbit, n)
        reps, info = symmetry_orbit_reps(orbit, n, self.D, antiunitary=False)
        self.reps, self.info = np.sort(reps), info
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.Q1, _ = _reduce(self.F1, self.U1, self.psi1)

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    # -- exact checks --
    def rank_mod2(self, idx, with_psi):
        M = [self.U2[k] for k in idx]
        if with_psi:
            M.append(self.psi2)
        return rank_mod(np.array(M, dtype=np.int64), P2)

    def solve(self, idx):
        A = self.C[:, list(idx)]
        d, *_ = np.linalg.lstsq(A, self.psi, rcond=None)
        return d, float(np.linalg.norm(A @ d - self.psi)), len(idx) - np.linalg.matrix_rank(A, tol=1e-8)

    def is_cover(self, idx):
        r0 = self.rank_mod2(idx, False)
        r1 = self.rank_mod2(idx, True)
        _, res, _ = self.solve(idx)
        exact = r0 == r1
        if exact != (res < NUM_TOL):
            raise AssertionError(f"modular and numeric cover tests disagree on {idx}")
        return exact

    def is_full(self, idx):
        A = self.C[:, list(idx)]
        d0, *_ = np.linalg.lstsq(A, self.psi, rcond=None)
        if np.linalg.norm(A @ d0 - self.psi) > NUM_TOL:
            return False
        _, s, vh = np.linalg.svd(A)
        rank = int(np.sum(s > 1e-8))
        K = vh[rank:].conj().T
        dead = (np.abs(d0) < 1e-9) & (np.abs(K).sum(axis=1) < 1e-9 if K.shape[1] else True)
        return not np.any(dead)

    # -- kernels --
    def pivot_plan(self, i):
        roots = self.info["roots"]
        orbit_size = int(np.count_nonzero(roots == i))
        labels, _ = stabilizer_orbit_labels(self.info["perms"], int(i),
                                            stabilizer_order=self.info["order"] // orbit_size)
        cand = np.flatnonzero(roots >= i)
        _, first = np.unique(labels[cand], return_index=True)
        partners = cand[first]
        partners = partners[partners != i]
        return cand, partners

    def units(self):
        out = []
        for i in self.reps:
            members, partners = self.pivot_plan(int(i))
            for j in partners:
                out.append((int(i), int(j), int(np.count_nonzero(members > j))))
        return out

    def covers(self, r, pivots=None):
        F = self.F1
        found, ncand = set(), 0
        for i in (self.reps if pivots is None else pivots):
            i = int(i)
            members, partners = self.pivot_plan(i)
            member_mask = np.zeros(self.N, dtype=bool)
            member_mask[members] = True
            Qi, _ = _reduce(F, self.Q1, self.Q1[i])
            if r == 3:
                keep = member_mask.copy()
                keep[i] = False
                ids = np.flatnonzero(keep)
                Rc, has = _canon_rows(F, Qi[ids])
                ids, Rc = ids[has], Rc[has]
                key = (Rc @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
                for g in _groups_by_key(key):
                    for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                        ncand += 1
                        idx = tuple(sorted((i, a, b)))
                        if self.is_cover(idx) and self.is_full(idx):
                            found.add(idx)
                continue
            for j in partners:
                got, nc, _ = self.pair_covers(r, i, int(j), Qi, member_mask)
                found.update(got)
                ncand += nc
        return sorted(found), ncand

    def pair_covers(self, r, i, j, Qi, member_mask, max_run=64, block=16):
        """Full r-covers (r = 4, 5) with pivot i, partner j, other members
        above j in member_mask. Returns (set, candidates, M)."""
        F = self.F1
        found, ncand = set(), 0
        if not np.any(Qi[j]):
            return found, 0, 0
        R, _ = _reduce(F, Qi, Qi[j])
        keep = member_mask & (np.arange(self.N) > j)
        keep[i] = False
        ids = np.flatnonzero(keep)
        if r == 4:
            Rc, has = _canon_rows(F, R[ids])
            ids2, Rc = ids[has], Rc[has]
            key = (Rc @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
            for g in _groups_by_key(key):
                for a, b in itertools.combinations(sorted(ids2[g].tolist()), 2):
                    ncand += 1
                    idx = tuple(sorted((i, j, a, b)))
                    if self.is_cover(idx) and self.is_full(idx):
                        found.add(idx)
            return found, ncand, len(ids2)
        Rm = R[ids]
        has = Rm.any(axis=1)
        ids, Rm = ids[has], Rm[has]
        M = len(ids)
        if M < 3:
            return found, ncand, M
        D2 = Rm.shape[1]
        first = np.argmax(Rm != 0, axis=1)
        lead = Rm[np.arange(M), first]
        Rk = (Rm * F.inv(lead)[:, None]) % F.p
        f = self.rng.integers(1, F.p, size=D2)
        for k0 in range(0, M - 1, block):
            k1 = min(M - 1, k0 + block)
            K = k1 - k0
            coef = Rm[:, first[k0:k1]].T                                # (K, M)
            res = (Rm[None, :, :] - coef[:, :, None] * Rk[k0:k1, None, :]) % F.p   # (K, M, D2)
            nz = res != 0
            hasres = nz.any(axis=2)
            f2 = np.argmax(nz, axis=2)
            lead2 = np.take_along_axis(res, f2[:, :, None], axis=2)[:, :, 0]
            sc = np.where(hasres, F.inv(lead2), 0)
            res = (res * sc[:, :, None]) % F.p
            key = (res @ f) % F.p                                       # (K, M)
            sent = F.p + np.arange(M)
            bad = ~hasres | (np.arange(M)[None, :] <= (k0 + np.arange(K))[:, None])
            key = np.where(bad, sent[None, :], key)
            order = np.argsort(key, axis=1, kind="stable")
            sk = np.take_along_axis(key, order, axis=1)
            eq = sk[:, 1:] == sk[:, :-1]
            for kk, l in zip(*np.nonzero(eq)):
                if l > 0 and eq[kk, l - 1]:
                    continue
                e = l + 1
                while e < M - 1 and eq[kk, e]:
                    e += 1
                run = order[kk, l:e + 1]
                if len(run) > max_run:
                    raise AssertionError(f"parallel class of size {len(run)} at {i}, {j}, {ids[k0 + kk]}")
                for a, b in itertools.combinations(sorted(ids[run].tolist()), 2):
                    ncand += 1
                    idx = tuple(sorted((i, j, int(ids[k0 + kk]), a, b)))
                    if self.is_cover(idx) and self.is_full(idx):
                        found.add(idx)
        return found, ncand, M

    def in_span(self, idx):
        F = self.F1
        rows = self.U1.copy()
        basis = self.U1[list(idx)].copy()
        for b in range(len(idx)):
            v = basis[b]
            if not np.any(v):
                raise AssertionError("dependent basis in in_span")
            rows, c = _reduce(F, rows, v)
            basis = np.delete((basis - (basis[:, c] * F.inv(v[c]))[:, None] * v[None, :]) % F.p, c, axis=1)
        cand = np.flatnonzero(~rows.any(axis=1))
        out = []
        for x in cand:
            if int(x) in idx:
                continue
            if self.rank_mod2(tuple(idx) + (int(x),), False) == len(idx):
                out.append(int(x))
        return out

    def dependent_covers(self, r, covers3, covers4):
        F = self.F1
        out = set()
        for T in covers3:
            span = self.in_span(T)
            for extra in itertools.combinations(span, r - 3):
                idx = tuple(sorted(T + extra))
                if self.is_full(idx):
                    out.add(idx)
            if r == 5:
                rows = self.U1.copy()
                for b in T:
                    rows, _ = _reduce(F, rows, rows[b])
                Rc, has = _canon_rows(F, rows)
                ids = np.flatnonzero(has)
                key = (Rc[ids] @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
                for g in _groups_by_key(key):
                    for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                        idx = tuple(sorted(T + (a, b)))
                        if self.is_cover(idx) and self.is_full(idx):
                            out.add(idx)
        if r == 5:
            for Cv in covers4:
                for x in self.in_span(Cv):
                    idx = tuple(sorted(Cv + (x,)))
                    if self.is_full(idx):
                        out.add(idx)
        return sorted(out)

    def repeated_covers(self, r, covers3, covers4):
        """Full r-covers as multisets with a repeated state: a (r-1)-cover
        with one of its states repeated, a (r-2)-cover with a pair repeated
        or one state tripled (fullness of the multiset: every distinct state
        has a nonzero coefficient in some solution, the copies sharing it)."""
        out = set()
        base = covers4 if r == 5 else covers3
        for Cv in base:
            for x in Cv:
                out.add(tuple(sorted(Cv + (x,))))
        if r == 5:
            for T in covers3:
                for x in T:
                    out.add(tuple(sorted(T + (x, x))))
                for x, y in itertools.combinations(T, 2):
                    out.add(tuple(sorted(T + (x, y))))
        return sorted(out)


def census(args):
    orbit, n = args[0], int(args[1])
    cap = None
    npairs = None
    out = None
    for k, a in enumerate(args):
        if a == "--cap":
            cap = float(args[k + 1])
        if a == "--pairs":
            npairs = int(args[k + 1])
        if a == "--out":
            out = args[k + 1]
    os.nice(19)
    t0 = time.time()
    E = CoverEnumerator3(orbit, n, verbose=True)
    units = E.units()
    print(f"{orbit} n={n}: N={E.N}, |G|={E.info['order']}, {len(E.reps)} pivot orbits, {len(units)} pivot pairs "
          f"[{time.time() - t0:.1f}s]", flush=True)
    rec = {"orbit": orbit, "n": n, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "pairs": len(units), "rows": []}
    if n == 2:
        for r in (3, 4):
            t = time.time()
            cv, nc = E.covers(r)
            rec[f"covers{r}"] = [list(c) for c in cv]
            print(f"  r={r}: {len(cv)} full covers, {nc} candidates [{time.time() - t:.1f}s]", flush=True)
        c3 = [tuple(c) for c in rec["covers3"]]
        c4 = [tuple(c) for c in rec["covers4"]]
    # r = 5 over the units (all, or a stratified sample of --pairs)
    sel = units
    if npairs is not None and npairs < len(units):
        idx = np.linspace(0, len(units) - 1, npairs).astype(int)
        sel = [units[k] for k in idx]
    plans = {}
    tot_cov, tot_cand = 0, 0
    found5 = set()
    t5 = time.time()
    for u, (i, j, M0) in enumerate(sel):
        if cap is not None and time.time() - t5 > cap:
            print(f"  cap reached after {u} pairs", flush=True)
            break
        if i not in plans:
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            plans[i] = (Qi, mask)
        Qi, mask = plans[i]
        tk = time.time()
        got, nc, M = E.pair_covers(5, i, j, Qi, mask)
        dt = time.time() - tk
        found5.update(got)
        tot_cov += len(got)
        tot_cand += nc
        rec["rows"].append([i, j, M, len(got), nc, dt])
        if n == 3 or (u + 1) % 500 == 0:
            print(f"  pair {u + 1}/{len(sel)} ({i}, {j}): M={M}, {len(got)} covers, {nc} candidates, {dt:.1f}s "
                  f"[total {tot_cov} covers, {time.time() - t0:.0f}s]", flush=True)
    rec["covers5_found"] = int(len(found5))
    rec["covers5"] = [list(c) for c in sorted(found5)] if n == 2 else [list(c) for c in sorted(found5)][:200]
    rec["candidates5"] = int(tot_cand)
    rec["seconds5"] = time.time() - t5
    rec["pairs_run"] = len(rec["rows"])
    print(f"  r=5: {len(found5)} full covers from {len(rec['rows'])} pairs, {tot_cand} candidates "
          f"[{time.time() - t5:.1f}s]", flush=True)
    if n == 2:
        t = time.time()
        dep = E.dependent_covers(5, c3, c4)
        rep = E.repeated_covers(5, c3, c4)
        # multiplicity patterns of the dependent list
        kap = {}
        for c in dep:
            _, _, nul = E.solve(c)
            kap[nul] = kap.get(nul, 0) + 1
        rec["dependent5"] = [list(c) for c in dep]
        rec["repeated5"] = [list(c) for c in rep]
        rec["dependent5_kappa"] = {str(k): v for k, v in kap.items()}
        print(f"  degenerate r=5: {len(dep)} dependent (kappa histogram {kap}), {len(rep)} repeated "
              f"[{time.time() - t:.1f}s]", flush=True)
        # in-span counts of the 3-covers and 4-covers
        rec["inspan3"] = [len(E.in_span(T)) for T in c3]
        rec["inspan4"] = [len(E.in_span(T)) for T in c4]
        print(f"  states in the span of each 3-cover: {rec['inspan3']}; of the 4-covers: "
              f"min {min(rec['inspan4']) if c4 else None}, max {max(rec['inspan4']) if c4 else None}")
    if out:
        with open(out, "w") as f:
            json.dump(rec, f)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "census":
        sys.exit(census(sys.argv[2:]))
