"""The degenerate base lists of the rank-6 exclusion of |N>^4 at k = 6
(docs/notes/n4_rank6_design.md, sections 2 and 5): the full 6-multisets
of two-qutrit stabilizer states at the base point that are not six
distinct independent states, and their reduction to one representative
per orbit of the unitary symmetry group G_2 of |N>^2 (order 72).

    degenerate6.py [--orbit N] [--out FILE] [--reps FILE] [--skip-c6]

Two lists, both built from the rank-5 census files
(research/qutrit_m4_rank5/results/N/kernel_census.json: the full 3-, 4-,
and 5-covers of distinct independent states, one per G_2 orbit as far as
the pivot and partner reductions go; degenerate_covers_N.json: the
dependent 5-covers).

B6, dependent 6-sets of distinct states. A full 6-set S with a coefficient
family of dimension kappa >= 1 contains an independent full k-cover T
(k <= 5) with rank(T u R) = rank(S) <= 5 for R = S \\ T: move along the
family until a coefficient vanishes, drop the zero coefficients, repeat.
So S is one of: T_5 + a state of span(T_5) (rank 5, kappa 1); T_4 + two
states of span(T_4) (rank 4, kappa 2); T_4 + a pair parallel modulo
span(T_4), both outside it (rank 5, kappa 1); T_3 + three states of its
span (kappa 3); T_3 + a span state + a parallel pair (kappa 2); T_3 + a
triple all parallel modulo span(T_3) (kappa 2); T_3 + a triple whose
images modulo span(T_3) span a plane with no two parallel (kappa 1). A
6-set with an added state outside the span that is parallel to nothing
has that state dead and is not full. Every candidate is decided
numerically (residual, rank, no dead coefficient), the modular
enumeration being a superset.

C6, 6-multisets with a repeated state. Let C be the distinct states whose
merged coefficient is nonzero and Z the states whose copies cancel at the
base point. C is a full |C|-cover of distinct states (independent or
dependent, |C| in {3, 4, 5}), so the multiset is C with multiplicities
plus cancelling blocks of at least two copies of a state outside C, the
total being 6: a 5-cover with a member doubled; a 4-cover with a member
tripled or two doubled, or plus (b, b) for any b outside it; a 3-cover
with multiplicities (4, 1, 1), (3, 2, 1), (2, 2, 2), or a member doubled
plus (b, b), or plus (b, b, b). No fullness test applies beyond C being
full (the blocks are exempt). The same multiset can arise from several
routes (b in span(C) makes C + b a dependent cover); the list is
deduplicated.

Orbit reduction. G_2 acts on the two-qutrit dictionary by permutations
(the closure of the generators of symmetry_orbit_reps, order 72); the
canonical form of a multiset is the lexicographically least image of its
sorted tuple. The note states the lemma that one base per orbit suffices.
"""
from __future__ import annotations

import argparse
import hashlib
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
from cover_census import CoverEnumerator3, P1, _canon_rows, _groups_by_key, _reduce  # noqa: E402
import common as qcommon  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TOL = 1e-8


def log(s):
    print(s, flush=True)


def group_perms(E):
    """The unitary symmetry group as an array (|G|, N) of permutations of
    the dictionary, the closure of the enumerator's generators."""
    N = E.N
    seen = {tuple(range(N))}
    elems = [np.arange(N)]
    frontier = [np.arange(N)]
    while frontier:
        nxt = []
        for g in frontier:
            for p in E.info["perms"]:
                h = p[g]
                t = tuple(h)
                if t not in seen:
                    seen.add(t)
                    elems.append(h)
                    nxt.append(h)
        frontier = nxt
    G = np.array(elems, dtype=np.int64)
    assert len(G) == E.info["order"], (len(G), E.info["order"])
    return G


def encode(sets, N):
    """Sorted rows (n, k) to one int64 per row, base N."""
    code = np.zeros(len(sets), dtype=np.int64)
    for c in range(sets.shape[1]):
        code = code * N + sets[:, c]
    return code


def decode(codes, k, N):
    out = np.zeros((len(codes), k), dtype=np.int64)
    c = codes.copy()
    for j in range(k - 1, -1, -1):
        out[:, j] = c % N
        c //= N
    return out


def canonical_codes(sets, G, N, chunk=20000):
    """The canonical form (least image under G of the sorted tuple) of every
    row of `sets` (n, k), as base-N codes."""
    sets = np.asarray(sets, dtype=np.int64)
    out = np.empty(len(sets), dtype=np.int64)
    for s in range(0, len(sets), chunk):
        blk = sets[s:s + chunk]
        img = G[:, blk]                                 # (|G|, n, k)
        img.sort(axis=2)
        codes = np.zeros(img.shape[:2], dtype=np.int64)
        for c in range(img.shape[2]):
            codes = codes * N + img[:, :, c]
        out[s:s + chunk] = codes.min(axis=0)
    return out


def residues_mod(E, rows_idx):
    """The dictionary reduced modulo the span of the states rows_idx over
    F_P1: (R, zero) with zero the mask of states in the span (P1 superset)."""
    rows = E.U1.copy()
    for b in rows_idx:
        rows, _ = _reduce(E.F1, rows, rows[b])
    zero = ~rows.any(axis=1)
    return rows, zero


def span_states(E, T):
    """States of span(T) other than T's members, over F_P1 (a superset of
    the exact answer, decided numerically by the caller)."""
    _, zero = residues_mod(E, T)
    zero[list(T)] = False
    return np.flatnonzero(zero)


def parallel_groups(E, T, rng):
    """Groups of states (outside span(T)) whose images modulo span(T) are
    parallel over F_P1, and the span states."""
    R, zero = residues_mod(E, T)
    Rc, has = _canon_rows(E.F1, R)
    ids = np.flatnonzero(has)
    f1 = rng.integers(1, P1, size=Rc.shape[1])
    f2 = rng.integers(1, P1, size=Rc.shape[1])
    key = ((Rc[ids] @ f1) % P1) * P1 + ((Rc[ids] @ f2) % P1)
    groups = [ids[g] for g in _groups_by_key(key)]
    zero[list(T)] = False
    return groups, np.flatnonzero(zero)


def decide_batch(E, sets, chunk=20000):
    """Numerical decision of 6-sets (n, 6): (cover, kappa, full) arrays.
    cover: psi in the span; kappa: 6 - rank; full: no coefficient dead on
    the whole family (CoverEnumerator3.is_full, vectorized)."""
    n = len(sets)
    cover = np.zeros(n, dtype=bool)
    kappa = np.zeros(n, dtype=np.int64)
    full = np.zeros(n, dtype=bool)
    for s in range(0, n, chunk):
        idx = sets[s:s + chunk]
        A = np.transpose(E.C[:, idx], (1, 0, 2))           # (c, 9, 6)
        U, sv, Vh = np.linalg.svd(A)
        rank = (sv > 1e-8).sum(axis=1)
        # least-squares coefficients through the SVD
        r = A.shape[2]
        sinv = np.where(sv > 1e-8, 1.0 / np.where(sv > 1e-8, sv, 1.0), 0.0)
        Ub = np.einsum("cij,i->cj", U[:, :, :r].conj(), E.psi)          # (c, 6)
        d0 = np.einsum("cji,cj->ci", Vh.conj(), Ub * sinv)              # (c, 6)
        res = np.linalg.norm(np.einsum("cij,cj->ci", A, d0) - E.psi[None, :], axis=1)
        cov = res < TOL
        # kernel directions: rows of Vh beyond the rank
        dead = np.zeros((len(idx), r), dtype=bool)
        for c in range(len(idx)):
            K = Vh[c, rank[c]:].conj().T                                  # (6, kappa)
            dd = np.abs(d0[c]) < 1e-9
            if K.shape[1]:
                dd &= np.abs(K).sum(axis=1) < 1e-9
            dead[c] = dd
        cover[s:s + chunk] = cov
        kappa[s:s + chunk] = r - rank
        full[s:s + chunk] = cov & ~dead.any(axis=1)
    return cover, kappa, full


def build_b6(E, covers3, covers4, covers5, rng):
    """The dependent 6-sets by route, as (route -> list of sorted tuples)
    before the numerical decision, and the (T_5, x) multiplicity."""
    routes = {k: [] for k in ("R5", "R4a", "R4b", "R3a", "R3b", "R3c", "R3d")}
    t0 = time.time()
    span_hist = {}
    for n, T in enumerate(covers5):
        sp = span_states(E, T)
        span_hist[len(sp)] = span_hist.get(len(sp), 0) + 1
        for x in sp:
            routes["R5"].append(tuple(sorted(T + (int(x),))))
        if (n + 1) % 50000 == 0:
            log(f"  R5: {n + 1}/{len(covers5)} 5-covers, {len(routes['R5'])} sets [{time.time() - t0:.0f}s]")
    log(f"  R5 done: {len(routes['R5'])} (T_5, x) pairs, span-state histogram {dict(sorted(span_hist.items()))} "
        f"[{time.time() - t0:.0f}s]")
    for T in covers4:
        groups, sp = parallel_groups(E, T, rng)
        for a, b in itertools.combinations(sp.tolist(), 2):
            routes["R4a"].append(tuple(sorted(T + (a, b))))
        for g in groups:
            for a, b in itertools.combinations(sorted(g.tolist()), 2):
                routes["R4b"].append(tuple(sorted(T + (a, b))))
    log(f"  R4: {len(routes['R4a'])} span pairs, {len(routes['R4b'])} parallel pairs [{time.time() - t0:.0f}s]")
    for T in covers3:
        groups, sp = parallel_groups(E, T, rng)
        for trip in itertools.combinations(sp.tolist(), 3):
            routes["R3a"].append(tuple(sorted(T + trip)))
        for a in sp.tolist():
            for g in groups:
                for b, c in itertools.combinations(sorted(g.tolist()), 2):
                    routes["R3b"].append(tuple(sorted(T + (a, b, c))))
        for g in groups:
            for trip in itertools.combinations(sorted(g.tolist()), 3):
                routes["R3c"].append(tuple(sorted(T + trip)))
        # collinear triples in general position: residues modulo span(T, a)
        outside = np.flatnonzero(~np.isin(np.arange(E.N), list(T) + sp.tolist()))
        for a in outside.tolist():
            R, zero = residues_mod(E, T + (a,))
            Rc, has = _canon_rows(E.F1, R)
            ids = np.flatnonzero(has & (np.arange(E.N) > a))
            ids = ids[~np.isin(ids, list(T) + sp.tolist())]
            if len(ids) < 2:
                continue
            f1 = rng.integers(1, P1, size=Rc.shape[1])
            f2 = rng.integers(1, P1, size=Rc.shape[1])
            key = ((Rc[ids] @ f1) % P1) * P1 + ((Rc[ids] @ f2) % P1)
            for g in _groups_by_key(key):
                for b, c in itertools.combinations(sorted(ids[g].tolist()), 2):
                    routes["R3d"].append(tuple(sorted(T + (a, b, c))))
    log(f"  R3: {[len(routes[k]) for k in ('R3a', 'R3b', 'R3c', 'R3d')]} [{time.time() - t0:.0f}s]")
    return routes


def build_c6(E, full3, full4, full5, dep4, dep5):
    """The repeated 6-multisets by route (route -> list of sorted tuples)."""
    routes = {}
    N = E.N

    def add(name, ms):
        routes.setdefault(name, []).append(tuple(sorted(ms)))

    for C in full5 + dep5:
        for u in C:
            add("C5+member (2,1,1,1,1)", C + (u,))
    for C in full4 + dep4:
        cs = set(C)
        for u in C:
            add("C4+member^2 (3,1,1,1)", C + (u, u))
        for u, v in itertools.combinations(C, 2):
            add("C4+2 members (2,2,1,1)", C + (u, v))
        for b in range(N):
            if b not in cs:
                add("C4+(b,b) cancel (2,1,1,1,1)", C + (b, b))
    for C in full3:
        cs = set(C)
        for u in C:
            add("C3 (4,1,1)", C + (u, u, u))
        for u, v in itertools.permutations(C, 2):
            add("C3 (3,2,1)", C + (u, u, v))
        add("C3 (2,2,2)", C + C)
        for u in C:
            for b in range(N):
                if b not in cs:
                    add("C3+member+(b,b) cancel (2,2,1,1)", C + (u, b, b))
        for b in range(N):
            if b not in cs:
                add("C3+(b,b,b) cancel (3,1,1,1)", C + (b, b, b))
    return routes


def pattern(ms):
    return tuple(sorted((ms.count(u) for u in set(ms)), reverse=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit", default="N")
    ap.add_argument("--out")
    ap.add_argument("--reps", help="write the orbit representatives (npz)")
    ap.add_argument("--skip-c6", action="store_true")
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args(argv)
    try:
        os.nice(19)
    except OSError:
        pass
    t0 = time.time()
    E = CoverEnumerator3(a.orbit, 2)
    G = group_perms(E)
    rng = np.random.default_rng(a.seed)
    cen = qcommon.load_census(a.orbit)
    covers3 = [tuple(c) for c in cen["covers3"]]
    covers4 = [tuple(c) for c in cen["covers4"]]
    covers5 = [tuple(c) for c in cen["covers5"]]
    deg, _ = qcommon.load_degenerate(a.orbit)
    dep5 = [c for c in deg if len(set(c)) == 5]
    log(f"{a.orbit}: N={E.N}, |G_2|={len(G)}, covers 3/4/5: {len(covers3)}/{len(covers4)}/{len(covers5)}, "
        f"dependent 5-covers {len(dep5)} [{time.time() - t0:.1f}s]")
    # dependent 4-covers: T_3 plus a span state, full
    dep4 = []
    for T in covers3:
        for x in span_states(E, T).tolist():
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                dep4.append(S)
    dep4 = sorted(set(dep4))
    log(f"dependent 4-covers: {len(dep4)}")
    rec = {"orbit": a.orbit, "N": int(E.N), "group_order": int(len(G)), "covers3": len(covers3),
           "covers4": len(covers4), "covers5": len(covers5), "dependent5": len(dep5), "dependent4": len(dep4),
           "census_sha256": cen["sha256"], "B6": {}, "C6": {}}

    # ---- B6
    routes = build_b6(E, covers3, covers4, covers5, rng)
    allsets = sorted(set(itertools.chain.from_iterable(routes.values())))
    arr = np.array(allsets, dtype=np.int64)
    log(f"B6 candidates: {sum(len(v) for v in routes.values())} by route, {len(arr)} distinct sets")
    t1 = time.time()
    cover, kappa, full = decide_batch(E, arr)
    log(f"  decided numerically in {time.time() - t1:.0f}s: covers {int(cover.sum())}, full {int(full.sum())}, "
        f"kappa histogram {dict(zip(*np.unique(kappa[full], return_counts=True)))}")
    keep = full & (kappa >= 1)
    B = arr[keep]
    kB = kappa[keep]
    codes = canonical_codes(B, G, E.N)
    uniq, inv = np.unique(codes, return_inverse=True)
    orbit_kappa = np.zeros(len(uniq), dtype=np.int64)
    orbit_kappa[inv] = kB
    # multiplicity of the R5 route per 6-set and per orbit
    r5 = np.array(sorted(set(routes["R5"])), dtype=np.int64) if routes["R5"] else np.zeros((0, 6), dtype=np.int64)
    r5_codes = canonical_codes(r5, G, E.N) if len(r5) else np.zeros(0, dtype=np.int64)
    r5_pairs = len(routes["R5"])
    rec["B6"] = {
        "by_route_pairs": {k: len(v) for k, v in routes.items()},
        "distinct_sets_candidates": int(len(arr)), "distinct_sets_full_dependent": int(len(B)),
        "kappa_histogram_sets": {str(k): int(v) for k, v in zip(*np.unique(kB, return_counts=True))},
        "orbits": int(len(uniq)),
        "kappa_histogram_orbits": {str(k): int(v) for k, v in zip(*np.unique(orbit_kappa, return_counts=True))},
        "R5_pairs": int(r5_pairs), "R5_distinct_sets": int(len(r5)), "R5_orbits": int(len(np.unique(r5_codes))),
        "orbit_size_histogram": {str(k): int(v) for k, v in zip(*np.unique(np.bincount(inv), return_counts=True))},
        "sha256_orbit_codes": hashlib.sha256(uniq.tobytes()).hexdigest(),
    }
    log(f"B6: {len(B)} full dependent 6-sets, {len(uniq)} G_2 orbits; kappa (orbits) "
        f"{rec['B6']['kappa_histogram_orbits']}; R5 {r5_pairs} pairs, {len(r5)} sets, "
        f"{rec['B6']['R5_orbits']} orbits [{time.time() - t0:.0f}s]")
    reps = {"B6_orbit_codes": uniq, "B6_orbit_kappa": orbit_kappa}

    # ---- C6
    if not a.skip_c6:
        croutes = build_c6(E, covers3, covers4, covers5, dep4, dep5)
        allms = {}
        for name, lst in croutes.items():
            for ms in lst:
                allms.setdefault(ms, set()).add(name)
        keys = sorted(allms)
        arrc = np.array(keys, dtype=np.int64)
        codesc = canonical_codes(arrc, G, E.N)
        uniqc, invc = np.unique(codesc, return_inverse=True)
        pats = [pattern(list(ms)) for ms in keys]
        pat_sets, pat_orbits = {}, {}
        seen_orbit = set()
        for ms, pt, cd in zip(keys, pats, codesc):
            pat_sets[str(pt)] = pat_sets.get(str(pt), 0) + 1
            if cd not in seen_orbit:
                seen_orbit.add(cd)
                pat_orbits[str(pt)] = pat_orbits.get(str(pt), 0) + 1
        # kappa of the distinct-state family per multiset (numerical)
        rec["C6"] = {"by_route_multisets": {k: len(v) for k, v in croutes.items()},
                     "distinct_multisets": int(len(arrc)), "orbits": int(len(uniqc)),
                     "by_pattern_multisets": pat_sets, "by_pattern_orbits": pat_orbits,
                     "sha256_orbit_codes": hashlib.sha256(uniqc.tobytes()).hexdigest()}
        # dependent-state multisets (kappa >= 1 over the distinct states) by pattern
        dist = [tuple(sorted(set(ms))) for ms in keys]
        dep_by_pat_orbit = {}
        seen_orbit = set()
        for ms, dd, pt, cd in zip(keys, dist, pats, codesc):
            if cd in seen_orbit:
                continue
            seen_orbit.add(cd)
            A = E.C[:, list(dd)]
            if np.linalg.matrix_rank(A, tol=1e-8) < len(dd):
                dep_by_pat_orbit[str(pt)] = dep_by_pat_orbit.get(str(pt), 0) + 1
        rec["C6"]["dependent_distinct_states_by_pattern_orbits"] = dep_by_pat_orbit
        log(f"C6: {len(arrc)} distinct multisets, {len(uniqc)} orbits; by pattern (multisets) {pat_sets}; "
            f"by pattern (orbits) {pat_orbits}; dependent distinct states (orbits) {dep_by_pat_orbit} "
            f"[{time.time() - t0:.0f}s]")
        reps["C6_orbit_codes"] = uniqc
    rec["seconds"] = time.time() - t0
    out = a.out or os.path.join(RESULTS, f"degenerate6_{a.orbit}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    log(f"wrote {os.path.relpath(out, ROOT)}")
    if a.reps:
        np.savez_compressed(a.reps, **reps)
        log(f"wrote {a.reps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
