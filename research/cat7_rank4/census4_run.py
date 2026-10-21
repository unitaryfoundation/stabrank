"""Census of the full 4-covers of a base target over the 4-qubit stabilizer
dictionary, up to a symmetry group, in parallel over pivot orbits.

    python census4_run.py TARGET NWORKERS OUT.json [PIVOT_LO PIVOT_HI]

TARGET: 'c5' (slice 00 of c_7), 'c5d' (slice 01 of c_7), or a path to a JSON
exponent row (planted controls). Exact: screens mod P = 2013265921 on
Gaussian-integer quotients with the norm bounds of cert_cat_m7_lift
(|q1| <= 2, |q2| <= 8, |q3| <= 128, minors <= 2 * 128^2 < P), every genuine
candidate re-solved in Gaussian-rational arithmetic.

A candidate 4-set {s1, s2, s3, s4} (pivot s1, partner s2) is one with q3(s3),
q3(s4) nonzero and parallel, where q3 kills span(t, s1, s2). The relation
e3 s3 + e4 s4 = alpha t + beta s1 + gamma s2 has alpha != 0 exactly when the
images of s3, s4 modulo span(s1, s2) alone are not parallel; only those are
kept (the others are plain dependent 4-sets without t).
"""
import os, sys, json, time, itertools
import numpy as np
import multiprocessing as mp
def _find_root():
    """the public repository root: STABRANK_PUB, else the nearest ancestor of
    this file holding verify_challenge/stabrank_verify.py, else the worktree."""
    env = os.environ.get("STABRANK_PUB")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "verify_challenge", "stabrank_verify.py")):
            return d
        d = os.path.dirname(d)
    return "/Users/vincent.russo/Projects/genesis/stabrank-r31j"

PUB = _find_root()
sys.path.insert(0, os.path.join(PUB, "verify_challenge"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cert_cat_m7_lift as L
import cat_slice2 as C
P, POW_I = L.P, L.POW_I
_G = {}
OLD_HASH = os.environ.get("CENSUS_OLD_HASH") == "1"     # the pre-referee functional, for the completeness control only


def dotp(R, h):
    """(R . h) mod P for rows with entries in [0, P): products reduced before
    the sum, so the value is a linear function of the row mod P. The old form
    (R @ h) % P wrapped past 2^63 on 16-term rows and is kept only behind
    CENSUS_OLD_HASH=1 for the control that records its failure."""
    if OLD_HASH:
        return (R @ h) % P
    return ((R * h) % P).sum(axis=-1) % P


def target_row(name):
    """the base target: ('exponent row' or None, mod-P row, exact Gaussian-rational row).
    'c5' and 'c5d' are slices of c_7; a path names a JSON file with either an
    exponent row (list of ints in -1..3) or {"exact": [[re, im], ...]} with
    Gaussian rationals written as strings (planted controls)."""
    if name == "c5":
        e = C.slices(7)[0]
    elif name == "c5d":
        e = C.slices(7)[1]
    else:
        doc = json.load(open(name))
        if isinstance(doc, dict):
            from fractions import Fraction
            ex = [(Fraction(a), Fraction(b)) for a, b in doc["exact"]]
            mod = np.array([C.frac_mod(x) for x in ex], dtype=np.int64)
            return None, mod, ex
        e = np.array(doc, dtype=np.int8)
    mod = (POW_I[np.clip(e, 0, 3)] * (e >= 0)).astype(np.int64)
    return e, mod, C.exact_row(e)


def setup(name, symmetric):
    if _G:
        return _G
    E = L.stabilizer_exponents(4)
    M = L.to_modp(E)
    tb, tbm, tbx = target_row(name)
    if symmetric:
        gens = C.base_symmetry(4)
        ev, od = L.parity_swap_maps(4)
        gens = gens + [("perm", ev, od)]
        C.check_symmetry_slices(gens[:-1], 7)
        roots = L.orbit_reps(E, gens, 4)
    else:
        roots = np.arange(E.shape[0])
    j0 = int(np.flatnonzero(tbm != 0)[0])
    inv_t0 = pow(int(tbm[j0]), P - 2, P)
    Q1 = (M - M[:, j0:j0 + 1] * (tbm[None, :] * inv_t0 % P)) % P
    _G.update(E=E, M=M, tb=tb, tbm=tbm, tbx=tbx, roots=roots, Q1=Q1)
    return _G


def kill(Qrows, qpiv, k):
    """rows -> rows with the k-th coordinate eliminated against qpiv (kills span(qpiv))."""
    return (qpiv[k] * Qrows - Qrows[:, k:k + 1] * qpiv[None, :]) % P


def ratio_keys(fk, gk):
    inv = L.inv_mod(np.where(gk != 0, gk, 1))
    return np.where(gk != 0, fk * inv % P, np.where(fk != 0, P, P + 1))


def census_pivots(args):
    """args = (name, symmetric, pivots, log_every[, subset]); `subset` (control
    runs only) restricts the partner and pool states to the given indices."""
    name, symmetric, pivots, log_every = args[:4]
    subset = set(int(x) for x in args[4]) if len(args) > 4 and args[4] is not None else None
    stride, offset = (args[5] if len(args) > 5 else (1, 0))      # partner positions p with p % stride == offset
    G = setup(name, symmetric)
    E, M, tb, tbm, roots, Q1 = G["E"], G["M"], G["tb"], G["tbm"], G["roots"], G["Q1"]
    rng = np.random.default_rng(2026)
    f, g = rng.integers(1, P, size=16), rng.integers(1, P, size=16)
    order_r = np.argsort(roots, kind="stable")
    sroots = roots[order_r]
    found = set()
    npairs = 0
    t0 = time.time()
    for c, s1 in enumerate(pivots):
        start = int(np.searchsorted(sroots, roots[s1]))
        sub = order_r[start:]
        if int(sub[0]) != int(s1):
            pos = int(np.flatnonzero(sub == s1)[0])
            sub = np.concatenate([[s1], sub[:pos], sub[pos + 1:]])
        if subset is not None:
            sub = np.array([int(s1)] + [int(x) for x in sub[1:] if int(x) in subset])
        q1 = Q1[s1]
        k = int(np.flatnonzero(q1)[0])
        Q2 = kill(Q1[sub], q1, k)                 # kills span(t, s1)
        m1 = M[s1]
        k1 = int(np.flatnonzero(m1)[0])
        R1 = kill(M[sub], m1, k1)                 # kills span(s1)
        F2, G2 = dotp(Q2, f), dotp(Q2, g)         # reduce before the sum (referee, 2026-10-17)
        F1, G1 = dotp(R1, f), dotp(R1, g)
        kcol = np.argmax(Q2 != 0, axis=1); has2 = (Q2 != 0).any(axis=1)
        rcol = np.argmax(R1 != 0, axis=1)
        n = len(sub)
        for p in range(1, n - 1):
            if p % stride != offset or not has2[p]:
                continue
            pool = np.arange(p + 1, n)
            kk = int(kcol[p]); A2 = Q2[p, kk]; B2 = Q2[pool, kk]
            fk = (A2 * F2[pool] - B2 * F2[p]) % P
            gk = (A2 * G2[pool] - B2 * G2[p]) % P
            key3 = ratio_keys(fk, gk)
            rk = int(rcol[p])
            npairs += 1
            order = np.argsort(key3, kind="stable")
            sk = key3[order]
            same = np.flatnonzero(sk[1:] == sk[:-1])
            if same.size == 0:
                continue
            groups = {}
            for a in same:
                groups.setdefault(int(sk[a]), set()).update((int(order[a]), int(order[a + 1])))
            for kv, gset in groups.items():
                idx = sorted(gset)
                if kv == P + 1:
                    # both functionals vanish: drop the rows with q3 = 0 exactly (states in
                    # span(t, s1, s2)); any nonzero rows left are compared exactly below
                    rows_chk = kill(Q2[pool[idx]], Q2[p], kk)
                    nz = (rows_chk != 0).any(axis=1)
                    idx = [i for i, z in zip(idx, nz) if z]
                    if len(idx) < 2:
                        continue
                # every decision below is on full rows modulo P (exact by the norm bounds);
                # the hash keys only formed the group
                rows3 = kill(Q2[pool[idx]], Q2[p], kk)
                W3, _ = L.proj_normal(rows3)
                rowsp = kill(R1[pool[idx]], R1[p], rk)
                Wp, hasp = L.proj_normal(rowsp)
                for a_, b_ in itertools.combinations(range(len(idx)), 2):
                    if not np.array_equal(W3[a_], W3[b_]):
                        continue                      # q3 images not parallel
                    if np.array_equal(Wp[a_], Wp[b_]) or not hasp[a_] or not hasp[b_]:
                        continue                      # dependent modulo span(s1, s2): no t in the relation
                    found.add(tuple(sorted((int(s1), int(sub[p]), int(sub[pool[idx[a_]]]), int(sub[pool[idx[b_]]])))))
        if log_every and (c + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[{os.getpid()}] pivot {c + 1}/{len(pivots)} pairs {npairs} covers {len(found)} {el:.0f}s", flush=True)
    # exact classification
    tg = G["tbx"]
    A, dropped = [], 0
    for quad in sorted(found):
        d = L.solve_exact([C.exact_row(E[i]) for i in quad], tg)
        if d is not None and all(x != (0, 0) for x in d):
            A.append([list(quad), [[str(x[0]), str(x[1])] for x in d]])
        else:
            dropped += 1
    return {"covers": A, "pairs": npairs, "dropped": dropped, "pivots": len(pivots)}


def main():
    name, nw, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    symmetric = name in ("c5", "c5d")
    G = setup(name, symmetric)
    pivots = np.unique(G["roots"])
    if len(sys.argv) > 5:
        pivots = pivots[int(sys.argv[4]):int(sys.argv[5])]
    # work units: each pivot's partner loop is split into STRIDE classes so that the
    # largest orbit (pivot 0, 36,718 partners) does not set the wall clock
    STRIDE = int(os.environ.get("CENSUS_STRIDE", "16")) if nw > 1 else 1
    units = [(name, symmetric, [int(p)], 0, None, (STRIDE, r)) for p in pivots for r in range(STRIDE)]
    print(f"target {name}: {len(pivots)} pivots, {len(units)} work units (stride {STRIDE}), {nw} workers", flush=True)
    t0 = time.time()
    if nw == 1:
        parts = [census_pivots(u) for u in units]
    else:
        with mp.get_context("fork").Pool(nw) as pool:
            parts = pool.map(census_pivots, units, chunksize=1)
    seen = set()
    covers = []
    for part in parts:
        for c in part["covers"]:
            key = tuple(c[0])
            if key not in seen:
                seen.add(key); covers.append(c)
    res = {"target": name, "pivots": int(len(pivots)), "pairs": sum(p["pairs"] for p in parts),
           "dropped": sum(p["dropped"] for p in parts), "covers": covers, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"full 4-covers: {len(covers)} (dropped {res['dropped']}) from {res['pairs']} pairs in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
