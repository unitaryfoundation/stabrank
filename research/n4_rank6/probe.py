"""Measurements for the choice of the next lower-bound exclusion
(docs/notes/next_exclusion_feasibility_2.md).

    probe.py census6 ORBIT n [--pairs K] [--budget S] [--out FILE]
        Sampled census of the full 6-covers of |M>^n over the n-qudit
        dictionary, one per orbit of the unitary symmetry group as far as
        the pivot and partner reductions of the 5-cover enumerators go:
        pivot i, partner j, two further pivots k < l among the members above
        j, and a pair (a, b) whose images are parallel modulo
        span(target, u_i, u_j, u_k, u_l) over F_65521, keyed by two random
        functionals; every candidate is decided exactly (rank mod
        2013265921 and numerically, the fullness numerically) and its
        coefficient family's dimension kappa is recorded. ORBIT is N or H3
        (qutrits, n = 2, CoverEnumerator3) or qubit_H, qubit_T (qubits,
        CoverEnumerator). The pairs are a stratified sample of the units;
        the totals are projected per pivot.
    probe.py rate6 --census FILE [--count K] [--x0 2,2] [--deg K] [--out FILE]
        The qutrit stage A matcher (research/qutrit_m4_rank5/matcher.py,
        generic in the number of terms) on sampled full 6-covers of |N>^2
        from a census6 record at the base point x0, and on degenerate
        6-multisets built from the stored 5-cover census (a 5-cover plus one
        of its members; a 5-cover plus a state of its span).
    probe.py census5 ORBIT n [--pairs K] [--budget S] [--out FILE]
        Sampled 5-cover census of |M>^n through the qubit enumerator
        (compiled kernel when available): pivot pairs, seconds per pair,
        covers per pair, projected totals. Used for |H>^4 over the 36,720
        four-qubit states.
    probe.py census5t5 [--pairs K] [--budget S] [--plant K] [--plan-only] [--out FILE]
        Sampled 5-cover census of |T5>^2 over the 3,900 two-ququint states
        through the compiled kernel with a Q(zeta_5) field, one pair per
        pivot at least; --plant K plants K five-term targets and requires
        the kernel to recover each from its two smallest members;
        --plan-only prints the pivot pairs and the member-squared cost
        model without running the kernel.
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))

P1 = 65521
P2 = 2013265921


def log(s):
    print(s, flush=True)


def qutrit_orbit(orbit):
    return orbit in ("N", "H3", "S", "T3")


def build(orbit, n):
    if qutrit_orbit(orbit):
        from cover_census import CoverEnumerator3, _reduce
        E = CoverEnumerator3(orbit, n)
        return E, _reduce
    from slice_cover import CoverEnumerator, _reduce
    E = CoverEnumerator(n, orbit=orbit, native=True)
    return E, _reduce


def units_of(E):
    out = []
    for i in E.reps:
        members, partners = E.pivot_plan(int(i))
        for j in partners:
            out.append((int(i), int(j), int(np.count_nonzero(members > j))))
    return out


def stratified(units, K, phase=0.0, limit=None):
    """About K units, at least one per pivot, spread through each pivot's
    partner list at the fractions (t + phase) / k; `limit` then thins the
    list evenly to at most that many units."""
    by = {}
    for u in units:
        by.setdefault(u[0], []).append(u)
    total = len(units)
    sel = []
    for i, lst in by.items():
        k = max(1, int(round(K * len(lst) / total)))
        k = min(k, len(lst))
        idx = np.minimum(len(lst) - 1, ((np.arange(k) + phase) / k * len(lst)).astype(int))
        sel.extend(lst[t] for t in idx)
    if limit is not None and len(sel) > limit:
        pick = np.linspace(0, len(sel) - 1, limit).astype(int)
        sel = [sel[t] for t in pick]
    return sel


# ------------------------------------------------------------- census6 ----

def pair_covers6(E, reduce, i, j, Qi, member_mask, rng, max_run=4096):
    """Full 6-covers with pivot i, partner j and four further members above
    j: (covers, candidates, M). A cover is recorded as (idx, kappa)."""
    F = E.F1
    p = F.p
    found, ncand = {}, 0
    if not np.any(Qi[j]):
        return found, 0, 0
    R, _ = reduce(F, Qi, Qi[j])
    keep = member_mask & (np.arange(E.N) > j)
    keep[i] = False
    ids = np.flatnonzero(keep)
    Rm = R[ids]
    has = Rm.any(axis=1)
    ids, Rm = ids[has], Rm[has]
    M = len(ids)
    if M < 4:
        return found, ncand, M
    D2 = Rm.shape[1]
    cands = []
    for kk in range(M - 3):
        # reduce the rows above k by row k (drop its leading column)
        rk = Rm[kk]
        c = int(np.argmax(rk != 0))
        rows = Rm[kk + 1:]
        g = (rows[:, c] * F.inv(rk[c])) % p
        A = np.delete((rows - g[:, None] * rk[None, :]) % p, c, axis=1)      # (Mk, D2 - 1)
        idk = ids[kk + 1:]
        hasA = A.any(axis=1)
        A, idk = A[hasA], idk[hasA]
        Mk = len(idk)
        if Mk < 3:
            continue
        # for every l: residues of m > l modulo row l, normalized, keyed
        first = np.argmax(A != 0, axis=1)
        lead = A[np.arange(Mk), first]
        Al = (A * F.inv(lead)[:, None]) % p
        coef = A[:, first].T                                                  # (Mk, Mk)
        res = (A[None, :, :] - coef[:, :, None] * Al[:, None, :]) % p        # (Mk, Mk, D2 - 1)
        # drop the leading column of each l from its residues: instead of a
        # per-row delete, normalize by the first nonzero entry (projective key)
        nz = res != 0
        hasres = nz.any(axis=2)
        f0 = np.argmax(nz, axis=2)
        lead2 = np.take_along_axis(res, f0[:, :, None], axis=2)[:, :, 0]
        sc = np.where(hasres, F.inv(lead2), 0)
        res = (res * sc[:, :, None]) % p
        fa = rng.integers(1, p, size=res.shape[2])
        fb = rng.integers(1, p, size=res.shape[2])
        key = ((res @ fa) % p) * p + ((res @ fb) % p)                        # (Mk, Mk)
        sent = np.int64(p) * p + np.arange(Mk)
        bad = ~hasres | (np.arange(Mk)[None, :] <= np.arange(Mk)[:, None])
        key = np.where(bad, sent[None, :], key)
        order = np.argsort(key, axis=1, kind="stable")
        sk = np.take_along_axis(key, order, axis=1)
        eq = sk[:, 1:] == sk[:, :-1]
        for l, t in zip(*np.nonzero(eq)):
            if t > 0 and eq[l, t - 1]:
                continue
            e = t + 1
            while e < Mk - 1 and eq[l, e]:
                e += 1
            run = order[l, t:e + 1]
            if len(run) > max_run:
                raise AssertionError(f"parallel class of size {len(run)} at {i}, {j}, {ids[kk]}, {idk[l]}")
            for a, b in itertools.combinations(sorted(idk[run].tolist()), 2):
                ncand += 1
                cands.append((i, j, int(ids[kk]), int(idk[l]), a, b))
    # batched numerical pre-decision (a 6-set containing a 5-cover is a
    # candidate with a zero coefficient; those dominate), then the exact
    # decision of every survivor
    nfull = 0
    for s0 in range(0, len(cands), 20000):
        chunk = cands[s0:s0 + 20000]
        idx = np.array(chunk, dtype=np.int64)                                  # (C, 6)
        A = np.transpose(E.C[:, idx], (1, 0, 2))                               # (C, dim, 6)
        d = np.einsum("cji,i->cj", np.linalg.pinv(A), E.psi)                   # (C, 6)
        res = np.linalg.norm(np.einsum("cij,cj->ci", A, d) - E.psi, axis=1)
        rank = np.linalg.matrix_rank(A, tol=1e-8)
        ok = res < 1e-6
        indep = ok & (rank == 6) & (np.abs(d) > 1e-7).all(axis=1)
        dep = ok & (rank < 6)
        for t in np.flatnonzero(indep | dep):
            tup = tuple(sorted(chunk[t]))
            if tup in found:
                continue
            if E.is_cover(tup) and E.is_full(tup):
                _, _, nul = E.solve(tup)
                found[tup] = int(nul)
                nfull += 1
    return found, ncand, M


def census6(args):
    orbit, n = args.orbit, args.n
    os.nice(19)
    t0 = time.time()
    E, reduce = build(orbit, n)
    t_build = time.time() - t0
    units = units_of(E)
    log(f"{orbit} n={n}: N={E.N}, |G|={E.info['order']}, {len(E.reps)} pivots, {len(units)} pivot pairs, "
        f"built in {t_build:.1f}s")
    rng = np.random.default_rng(args.seed)
    sel = units if args.pairs is None or args.pairs >= len(units) else stratified(units, args.pairs, args.phase, args.limit)
    plans = {}
    rows = []
    found = {}
    kappa_hist = {}
    t_run = time.time()
    for u, (i, j, M0) in enumerate(sel):
        if args.budget and time.time() - t_run > args.budget:
            log(f"budget reached after {u} pairs")
            break
        if i not in plans:
            Qi, _ = reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            plans[i] = (Qi, mask)
        Qi, mask = plans[i]
        tk = time.time()
        got, nc, M = pair_covers6(E, reduce, i, j, Qi, mask, rng)
        dt = time.time() - tk
        for idx, kap in got.items():
            found[idx] = kap
            kappa_hist[kap] = kappa_hist.get(kap, 0) + 1
        rows.append([i, j, M, len(got), sum(1 for k in got.values() if k == 0), nc, dt])
        if (u + 1) % 10 == 0 or dt > 30:
            log(f"  pair {u + 1}/{len(sel)} ({i}, {j}): M={M}, {len(got)} full 6-covers, {nc} candidates, "
                f"{dt:.1f}s [total {len(found)} covers, {time.time() - t0:.0f}s]")
    # projection per pivot
    by_unit = {}
    for i, j, M0 in units:
        by_unit.setdefault(i, 0)
        by_unit[i] += 1
    proj_cov, proj_ind, proj_sec, per = 0.0, 0.0, 0.0, {}
    for i, cnt in by_unit.items():
        mine = [r for r in rows if r[0] == i]
        if not mine:
            continue
        mc = float(np.mean([r[3] for r in mine]))
        mi = float(np.mean([r[4] for r in mine]))
        ms = float(np.mean([r[6] for r in mine]))
        per[i] = {"pairs": cnt, "sampled": len(mine), "covers_per_pair": mc, "independent_per_pair": mi,
                  "seconds_per_pair": ms, "members": int(np.mean([r[2] for r in mine]))}
        proj_cov += mc * cnt
        proj_ind += mi * cnt
        proj_sec += ms * cnt
    rec = {"orbit": orbit, "n": n, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "pairs": len(units), "pairs_run": len(rows), "build_seconds": t_build,
           "seconds": time.time() - t_run, "rows": rows, "units": [list(u) for u in units],
           "columns": ["pivot", "partner", "members", "full6", "full6_independent", "candidates", "seconds"],
           "found": len(found), "kappa_histogram": {str(k): v for k, v in sorted(kappa_hist.items())},
           "candidates": int(sum(r[5] for r in rows)),
           "projected_full6": proj_cov, "projected_full6_independent": proj_ind,
           "projected_seconds": proj_sec, "per_pivot": per,
           "sample_covers": [[list(k), v] for k, v in list(found.items())[:2000]]}
    log(f"r=6: {len(found)} full 6-covers in {len(rows)} of {len(units)} pairs ({rec['candidates']} candidates, "
        f"{rec['seconds']:.0f}s); kappa histogram {kappa_hist}; projected {proj_cov:.3g} covers "
        f"({proj_ind:.3g} independent) in {proj_sec / 3600:.2f} CPU-hours")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f)
        log(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------- census5 ----

def census5(args):
    orbit, n = args.orbit, args.n
    os.nice(19)
    t0 = time.time()
    E, reduce = build(orbit, n)
    t_build = time.time() - t0
    log(f"{orbit} n={n}: N={E.N}, |G|={E.info['order']}, {len(E.reps)} pivots, built in {t_build:.1f}s; "
        f"native cover5: {E.native_cover5 is not None}")
    t1 = time.time()
    plans, per_pivot = {}, {}
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        per_pivot[i] = (len(partners), len(members))
        plans[i] = (members, partners)
    total_pairs = sum(c for c, _ in per_pivot.values())
    log(f"{total_pairs} pivot pairs planned in {time.time() - t1:.1f}s")
    rng = np.random.default_rng(args.seed)
    rows, covers = [], set()
    t_all = time.time()
    for i, (c, M) in per_pivot.items():
        if args.budget and time.time() - t_all > args.budget:
            log("budget reached")
            break
        members, partners = plans[i]
        k = max(1, int(round(args.pairs * c / total_pairs)))
        sel = rng.choice(partners, size=min(k, len(partners)), replace=False)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        Qi, _ = reduce(E.F1, E.Q1, E.Q1[i])
        for j in sel:
            tk = time.time()
            got, nc = E.pair_covers(5, i, int(j), Qi, mask)
            covers.update(got)
            rows.append([i, int(j), M, len(got), int(nc), time.time() - tk])
            log(f"  pivot {i} partner {int(j)}: {len(got)} covers, {nc} candidates, {rows[-1][5]:.1f}s")
    est_cov, est_sec, per = 0.0, 0.0, {}
    for i, (c, M) in per_pivot.items():
        mine = [r for r in rows if r[0] == i]
        if not mine:
            continue
        mc = float(np.mean([r[3] for r in mine]))
        ms = float(np.mean([r[5] for r in mine]))
        per[i] = {"pairs": c, "members": M, "sampled": len(mine), "covers_per_pair": mc, "seconds_per_pair": ms}
        est_cov += mc * c
        est_sec += ms * c
    rec = {"orbit": orbit, "n": n, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "pairs": total_pairs, "pairs_run": len(rows), "build_seconds": t_build, "rows": rows,
           "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"],
           "covers_found": len(covers), "projected_covers": est_cov, "projected_seconds": est_sec,
           "per_pivot": per, "sample_covers": [list(c) for c in sorted(covers)][:200]}
    log(f"r=5: {len(covers)} covers in {len(rows)} of {total_pairs} pairs; projected {est_cov:.3g} covers, "
        f"{est_sec / 3600:.2f} CPU-hours")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f)
        log(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------- rate6 ----

def rate6(args):
    os.nice(19)
    from cover_census import CoverEnumerator3
    from matcher import Matcher, Field3, psi_target
    import common as qcommon
    t0 = time.time()
    with open(args.census) as f:
        rec = json.load(f)
    orbit = rec["orbit"]
    covers = [tuple(c) for c, kap in rec["sample_covers"] if kap == 0]
    E = CoverEnumerator3(orbit, 2)
    Mt = Matcher(E.D, 2, E.F1, E.F2)
    target = psi_target(orbit, 2, Mt.F1, Mt.F2)
    x0 = tuple(int(v) for v in args.x0.split(","))
    log(f"{orbit}: {len(covers)} independent 6-covers in the record; matcher built in {time.time() - t0:.1f}s; x0={x0}")
    rng = np.random.default_rng(args.seed)
    out = {"orbit": orbit, "x0": list(x0), "groups": {}}

    def run_group(name, lst):
        secs, sols, hits, refused, kap = [], [], 0, 0, []
        t_g = time.time()
        for cover in lst:
            if args.budget and time.time() - t_g > args.budget:
                log(f"  {name}: budget reached after {len(secs)} runs")
                break
            t1 = time.time()
            h, st = Mt.run(cover, x0, target)
            secs.append(time.time() - t1)
            sols.append(list(st["coord_solutions"]))
            kap.append(st["kappa"])
            hits += len(h)
            refused += int(st["refused"])
        if secs:
            log(f"  {name}: {len(secs)} runs, mean {np.mean(secs):.3f}s, median {np.median(secs):.3f}s, "
                f"max {np.max(secs):.2f}s; hits {hits}, refused {refused}; kappa {sorted(set(kap))}; "
                f"coordinate solutions {sols[:8]}")
        out["groups"][name] = {"runs": len(secs), "mean": float(np.mean(secs)) if secs else None,
                               "median": float(np.median(secs)) if secs else None,
                               "max": float(np.max(secs)) if secs else None, "hits": hits, "refused": refused,
                               "kappa": [int(k) if k is not None else None for k in kap],
                               "coord_solutions": sols, "seconds": secs}

    pick = [covers[t] for t in np.linspace(0, len(covers) - 1, min(args.count, len(covers))).astype(int)]
    run_group("A6 independent", pick)
    if args.deg:
        c5 = [tuple(c) for c in qcommon.load_census(orbit)["covers5"]]
        sel = [c5[t] for t in rng.choice(len(c5), size=min(args.deg, len(c5)), replace=False)]
        rep = [tuple(sorted(c + (c[rng.integers(len(c))],))) for c in sel]
        run_group("C6 (2,1,1,1,1): 5-cover plus a member", rep)
        dep = []
        for c in sel:
            sp = E.in_span(c)
            if sp:
                dep.append(tuple(sorted(c + (sp[0],))))
            if len(dep) >= args.deg:
                break
        log(f"  dependent multisets from {len(sel)} sampled 5-covers: {len(dep)} (span states per cover: "
            f"{[len(E.in_span(c)) for c in sel[:10]]})")
        run_group("B6 dependent: 5-cover plus a span state", dep)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
        log(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------ census5t5 ----

class Field5:
    """Q(zeta_5) modulo a prime p = 1 mod 5: enough for the ququint
    stabilizer states (entries fifth roots of unity up to a scalar) and for
    |T5>^n (entries w5^{sum x^3})."""

    def __init__(self, p):
        self.p = p
        assert (p - 1) % 5 == 0
        for g in range(2, p):
            w = pow(g, (p - 1) // 5, p)
            if w != 1:
                break
        self.w = w
        assert pow(w, 5, p) == 1
        self.wpow = np.array([pow(w, k, p) for k in range(5)], dtype=np.int64)
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

    def codes_to_field(self, codes):
        out = np.zeros(codes.shape, dtype=np.int64)
        nz = codes > 0
        out[nz] = self.wpow[codes[nz] - 1]
        return out


def patterns5(D):
    """Phase codes (0 zero, 1..5 = w5^0..w5^4) of the dictionary columns with
    the first nonzero entry made 1, and the unnormalized complex matrix."""
    W5 = np.exp(2j * np.pi / 5)
    dim, N = D.shape
    nz = np.abs(D) > 1e-9
    first = np.argmax(nz, axis=0)
    ref = D[first, np.arange(N)]
    W = D / ref[None, :]
    codes = np.zeros((N, dim), dtype=np.int8)
    for c in range(5):
        codes[np.abs(W.T - W5 ** c) < 1e-6] = c + 1
    assert np.array_equal(codes > 0, nz.T), "a dictionary entry is not a fifth root of unity"
    C = np.zeros((dim, N), dtype=complex)
    for c in range(5):
        C[codes.T == c + 1] = W5 ** c
    return codes, C


class T5Enumerator:
    """The two-ququint dictionary with modular images and the pivot plan of
    the qubit enumerator, for the compiled 5-cover kernel."""

    def __init__(self, n=2, seed=17):
        from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps
        from slice_cover import _reduce
        self.n = n
        self.D = dictionary(5, n)
        self.N = self.D.shape[1]
        self.codes, self.C = patterns5(self.D)
        self.F1, self.F2 = Field5(P1), Field5(P2)
        self.U1 = self.F1.codes_to_field(self.codes)
        self.U2 = self.F2.codes_to_field(self.codes)
        pts = list(itertools.product(range(5), repeat=n))
        tc = np.array([sum(x ** 3 for x in pt) % 5 for pt in pts], dtype=np.int64)
        self.psi1 = self.F1.wpow[tc]
        self.psi2 = self.F2.wpow[tc]
        self.psi = psi_for("T5", n)
        reps, info = symmetry_orbit_reps("T5", n, self.D, antiunitary=False)
        self.reps, self.info = np.sort(reps), info
        self.rng_seed = seed
        self.Q1, _ = _reduce(self.F1, self.U1, self.psi1)
        from stabrank.stabrank_core import cover5_pair
        self.native_cover5 = cover5_pair

    def pivot_plan(self, i):
        from slice_lift import stabilizer_orbit_labels
        roots = self.info["roots"]
        orbit_size = int(np.count_nonzero(roots == i))
        labels, _ = stabilizer_orbit_labels(self.info["perms"], int(i),
                                            stabilizer_order=self.info["order"] // orbit_size)
        cand = np.flatnonzero(roots >= i)
        _, first = np.unique(labels[cand], return_index=True)
        partners = cand[first]
        partners = partners[partners != i]
        return cand, partners

    def solve(self, idx):
        A = self.C[:, list(idx)]
        d, *_ = np.linalg.lstsq(A, self.psi, rcond=None)
        return d, float(np.linalg.norm(A @ d - self.psi)), len(idx) - np.linalg.matrix_rank(A, tol=1e-8)

    def pair_covers_native(self, i, j, member_mask, max_run=4096):
        idx, flags, ncand, members = self.native_cover5(
            self.Q1, self.U1, self.psi1, self.U2, self.psi2, int(i), int(j),
            np.ascontiguousarray(member_mask, dtype=np.uint8), int(max_run), int(self.rng_seed))
        found = []
        for row, (f1, f2) in zip(np.asarray(idx), np.asarray(flags)):
            t = tuple(int(x) for x in row)
            d, res, nul = self.solve(t)
            full_num = res < 1e-7 and (nul > 0 or np.all(np.abs(d) > 1e-7))
            found.append((t, bool(f1), bool(f2), full_num, int(nul)))
        return found, int(ncand), int(members)


def census5t5(args):
    os.nice(19)
    t0 = time.time()
    E = T5Enumerator(2)
    t_build = time.time() - t0
    log(f"T5 n=2: N={E.N}, |G|={E.info['order']}, {len(E.reps)} pivots, built in {t_build:.1f}s; "
        f"w5 mod P1 = {E.F1.w}")
    # sanity: the target lies in the span of the whole dictionary mod P1 and the
    # modular target has the complex pattern
    t1 = time.time()
    per_pivot, plans = {}, {}
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        per_pivot[i] = (len(partners), len(members))
        plans[i] = (members, partners)
    total_pairs = sum(c for c, _ in per_pivot.values())
    log(f"{total_pairs} pivot pairs planned in {time.time() - t1:.1f}s; per pivot (partners/members): "
        + ", ".join(f"{i}:{c}/{M}" for i, (c, M) in list(per_pivot.items())[:12]) + " ...")
    # the members above every partner, for the M^2 cost model
    msq, mlist = 0.0, []
    for i, (members, partners) in plans.items():
        above = np.array([int(np.count_nonzero(members > j)) for j in partners], dtype=float)
        msq += float(np.sum(above ** 2))
        mlist.append(above)
    allm = np.concatenate(mlist)
    log(f"members above the partner over all pairs: mean {allm.mean():.0f}, max {allm.max():.0f}, "
        f"sum of squares {msq:.3g}; at {args.sec_per_m2:.3g} s per member^2 the census is {msq * args.sec_per_m2 / 3600:.1f} CPU-hours")
    if args.plan_only:
        return 0
    rng = np.random.default_rng(args.seed)
    # control: the modular target is the phase pattern of the complex one
    from rank_exclusion import psi_for
    cplx = psi_for("T5", 2)
    W5 = np.exp(2j * np.pi / 5)
    ratio = cplx / cplx[0]
    pts = list(itertools.product(range(5), repeat=2))
    tc = np.array([sum(x ** 3 for x in pt) % 5 for pt in pts])
    assert np.allclose(ratio, W5 ** tc), "the target pattern is not w5^(x^3 + y^3)"
    log("control: the modular target is the phase pattern w5^(x^3 + y^3) of |T5>^2")
    # control: planted targets sum_i c_i s_i over five random states are found
    # from their two smallest members with every other state admissible
    planted_ok = 0
    for t in range(args.plant):
        S = np.sort(rng.choice(E.N, size=5, replace=False))
        coef = np.array([1, 2, 3, 4, 5])
        rng.shuffle(coef)
        t1 = (coef[None, :] * E.U1[S].T).sum(axis=1) % P1
        t2 = (coef[None, :] * E.U2[S].T).sum(axis=1) % P2
        mask = np.ones(E.N, dtype=bool)
        from slice_cover import _reduce as _red
        Qt, _ = _red(E.F1, E.U1, t1)
        idx, flags, nc, mm = E.native_cover5(
            Qt, E.U1, t1, E.U2, t2, int(S[0]), int(S[1]),
            np.ascontiguousarray(mask, dtype=np.uint8), 4096, int(E.rng_seed))
        found_sets = {tuple(int(x) for x in row) for row in np.asarray(idx)}
        planted_ok += tuple(int(x) for x in S) in found_sets
    if args.plant:
        log(f"control: {planted_ok} of {args.plant} planted 5-term targets recovered from their two smallest members")
    rows, covers = [], {}
    t_all = time.time()
    for i, (c, M) in per_pivot.items():
        if args.budget and time.time() - t_all > args.budget:
            log("budget reached")
            break
        members, partners = plans[i]
        k = max(1, int(round(args.pairs * c / total_pairs)))
        sel = rng.choice(partners, size=min(k, len(partners)), replace=False)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        for j in sel:
            if args.budget and time.time() - t_all > args.budget:
                break
            tk = time.time()
            got, nc, Mm = E.pair_covers_native(i, int(j), mask)
            dt = time.time() - tk
            nfull = sum(1 for g in got if g[1] and g[2] and g[3])
            for g in got:
                covers[g[0]] = g[1:]
            rows.append([i, int(j), Mm, len(got), nfull, nc, dt])
            log(f"  pivot {i} partner {int(j)}: members {Mm}, {len(got)} span hits, {nfull} full, {nc} candidates, {dt:.1f}s")
    est_cov, est_full, est_sec, per = 0.0, 0.0, 0.0, {}
    for i, (c, M) in per_pivot.items():
        mine = [r for r in rows if r[0] == i]
        if not mine:
            continue
        mc = float(np.mean([r[3] for r in mine]))
        mf = float(np.mean([r[4] for r in mine]))
        ms = float(np.mean([r[6] for r in mine]))
        per[i] = {"pairs": c, "members": M, "sampled": len(mine), "span_hits_per_pair": mc,
                  "full_per_pair": mf, "seconds_per_pair": ms}
        est_cov += mc * c
        est_full += mf * c
        est_sec += ms * c
    rec = {"orbit": "T5", "n": 2, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "pairs": total_pairs, "pairs_run": len(rows), "build_seconds": t_build, "rows": rows,
           "planted": [planted_ok, args.plant], "units": [[i, c, M] for i, (c, M) in per_pivot.items()],
           "columns": ["pivot", "partner", "members", "span_hits", "full", "candidates", "seconds"],
           "covers_found": len(covers), "projected_span_hits": est_cov, "projected_full": est_full,
           "projected_seconds": est_sec, "per_pivot": per,
           "sample_covers": [[list(k), list(v)] for k, v in list(covers.items())[:200]]}
    log(f"r=5: {len(covers)} 5-sets with the target in their span in {len(rows)} of {total_pairs} pairs "
        f"({sum(1 for v in covers.values() if v[0] and v[1] and v[2])} full); projected {est_full:.3g} full covers, "
        f"{est_cov:.3g} span hits, {est_sec / 3600:.2f} CPU-hours")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f)
        log(f"wrote {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c6 = sub.add_parser("census6")
    c6.add_argument("orbit")
    c6.add_argument("n", type=int)
    c6.add_argument("--pairs", type=int)
    c6.add_argument("--budget", type=float, default=0)
    c6.add_argument("--seed", type=int, default=11)
    c6.add_argument("--phase", type=float, default=0.0)
    c6.add_argument("--limit", type=int)
    c6.add_argument("--out")
    c6.set_defaults(fn=census6)
    c5 = sub.add_parser("census5")
    c5.add_argument("orbit")
    c5.add_argument("n", type=int)
    c5.add_argument("--pairs", type=int, default=20)
    c5.add_argument("--budget", type=float, default=0)
    c5.add_argument("--seed", type=int, default=11)
    c5.add_argument("--out")
    c5.set_defaults(fn=census5)
    t5 = sub.add_parser("census5t5")
    t5.add_argument("--pairs", type=int, default=20)
    t5.add_argument("--budget", type=float, default=0)
    t5.add_argument("--seed", type=int, default=11)
    t5.add_argument("--plant", type=int, default=0)
    t5.add_argument("--plan-only", action="store_true")
    t5.add_argument("--sec-per-m2", type=float, default=7.49e-8)
    t5.add_argument("--out")
    t5.set_defaults(fn=census5t5)
    r6 = sub.add_parser("rate6")
    r6.add_argument("--census", required=True)
    r6.add_argument("--count", type=int, default=40)
    r6.add_argument("--deg", type=int, default=0)
    r6.add_argument("--x0", default="2,2")
    r6.add_argument("--budget", type=float, default=0)
    r6.add_argument("--seed", type=int, default=11)
    r6.add_argument("--out")
    r6.set_defaults(fn=rate6)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
