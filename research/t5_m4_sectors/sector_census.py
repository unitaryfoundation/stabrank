"""Exact stabilizer rank of the Z-type Pauli eigensectors of |T5>^3, by census
over the two-ququint dictionary, and what it says about |T5>^4.

The Z^4 eigensector of |T5>^4 at eigenvalue w^c is the restriction of the
phase pattern w^(x1^3 + x2^3 + x3^3 + x4^3) to the hyperplane x1 + x2 + x3
+ x4 = c. Slicing it at x4 = a (projecting the fourth ququint onto |a>)
gives w^(a^3) times the Z^3 eigensector of |T5>^3 at eigenvalue w^(c - a),
and slicing carries a stabilizer decomposition term by term to stabilizer
states or zeros, so

    chi(Z^3 sector of |T5>^3) <= chi(Z^4 sector of |T5>^4) <= 5.

The upper bound is the five sub-sector products of constructions_2026_09.md.
So the m = 4 question (rank 4, which would give chi(|T5>^4) <= 20) is
decided at m = 3 whenever the m = 3 sector has rank 5, and the m = 3 sector
is a two-ququint state, where the dictionary has 3,900 states and a
rank-1..4 census is minutes of laptop time.

The m = 3 sector of the string Z^c1 Z^c2 Z^c3 at eigenvalue w^d is the
restriction of w^(x^3 + y^3 + z^3) to the plane c1 x + c2 y + c3 z = d; the
Clifford (multiplier by c3 on the third ququint, then SUM gates from the
first two) maps it to |d> (x) g with

    g(x, y) = w^(x^3 + y^3 + z(x, y)^3),   z(x, y) = (d - c1 x - c2 y) / c3,

a two-ququint state with full support and a cubic phase, so chi(sector) =
chi(g). The five eigenvalues d of one string give Clifford-equivalent g
(translate x by one and correct the quadratic phase, the order-5 symmetry
of |T5> on the first copy), so each string has one sector class. Up to the
order of the sites and the overall power of the string (which does not
change the group and hence not the sectors) the Z-type strings fall in
five classes, {1,1,1}, {1,1,2}, {1,1,3}, {1,1,4}, {1,2,3}; the Z^4 sector
of |T5>^4 slices to {1,1,1}, the Z Z Z^2 Z^2 sector to {1,1,2} and {1,1,3},
and the Z Z Z^4 Z^4 sector to {1,1,4}.

The census (`census`) is the exact k = 1..4 census of research/t5q_m2_rank5
(common.py, `census_low`) with an arbitrary target and an arbitrary unitary
symmetry group: the dictionary is reduced modulo the target over Q(zeta_5)
mod 65521, then modulo a pivot, then modulo a partner; a k-set is a
candidate when the images of its last two members are parallel (a
two-functional key), and every candidate is decided exactly modulo
2013265921 and numerically. Pivots are the orbit roots of the unitary
symmetry group of g (the 25 translations with their phase corrections, and
the permutations of the three copies that fix the string); members have
roots at or above the pivot; partners are one per orbit of the pivot's
stabilizer subgroup (Schreier generators). The reduction is the one proved
in section 1.2 of docs/notes/t5q_m2_rank5_exclusion.md. Every symmetry
generator is checked to fix g up to phase and to permute the dictionary.

Usage:
    sector_census.py census C1 C2 C3 [--d D] [--kmax 4] [--no-symmetry]
                     [--out PATH]
    sector_census.py control            # planted and |N5>^2 controls
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
sys.path.insert(0, os.path.join(ROOT, "research", "t5q_m2_rank5"))
from rank_exclusion import dictionary  # noqa: E402
from slice_lift import stabilizer_orbit_labels  # noqa: E402
from common import (Field5, patterns5, reduce_rows, canon_rows, pair_key, groups_by_key,  # noqa: E402
                    rank_mod, P1, P2, W5, sha256_json)

NUM_TOL = 1e-7
RESULTS = os.path.join(HERE, "results")


# ------------------------------------------------------------ the target ---

def sector_codes(cs, d):
    """Phase codes (exponent + 1) of g(x, y) for the string Z^c1 Z^c2 Z^c3 at
    eigenvalue w^d, indexed 5 x + y."""
    c1, c2, c3 = cs
    inv3 = pow(c3, -1, 5)
    codes = np.zeros(25, dtype=np.int8)
    for x in range(5):
        for y in range(5):
            z = ((d - c1 * x - c2 * y) * inv3) % 5
            codes[5 * x + y] = (x ** 3 + y ** 3 + z ** 3) % 5 + 1
    return codes


def check_sector_reduction(cs, d):
    """The projection of |T5>^3 onto the eigenspace of Z^c1 Z^c2 Z^c3 with
    eigenvalue w^d is supported on the plane c1 x + c2 y + c3 z = d, and
    reading the plane off by (x, y) gives g up to the scalar 5^-3/2."""
    c1, c2, c3 = cs
    t = np.array([W5 ** ((x ** 3) % 5) for x in range(5)]) / np.sqrt(5)
    psi = np.kron(np.kron(t, t), t).reshape(5, 5, 5)
    ev = np.zeros((5, 5, 5), dtype=int)
    for x, y, z in itertools.product(range(5), repeat=3):
        ev[x, y, z] = (c1 * x + c2 * y + c3 * z) % 5
    sec = np.where(ev == d, psi, 0)
    g = W5 ** (sector_codes(cs, d).astype(int) - 1) / 5 ** 1.5
    inv3 = pow(c3, -1, 5)
    for x, y in itertools.product(range(5), repeat=2):
        z = ((d - c1 * x - c2 * y) * inv3) % 5
        assert abs(sec[x, y, z] - g[5 * x + y]) < 1e-12
    assert abs(np.linalg.norm(sec) ** 2 - 0.2) < 1e-12
    return True


class SectorTarget:
    """g in the three forms of the census: mod P1 (and the dictionary reduced
    modulo its span), mod P2, and the unnormalized complex pattern."""

    def __init__(self, D, codes_or_vec, F1, F2, U1, planted=None):
        self.F1, self.F2 = F1, F2
        if planted is None:
            codes = np.asarray(codes_or_vec, dtype=np.int8)
            self.t1 = F1.codes_to_field(codes)
            self.t2 = F2.codes_to_field(codes)
            self.tC = W5 ** (codes.astype(int) - 1)
            self.tC[codes == 0] = 0
        else:
            states, coeffs, U2, C = planted
            c = np.asarray(coeffs, dtype=np.int64)
            idx = [int(x) for x in states]
            self.t1 = (c[:, None] * U1[idx]).sum(axis=0) % P1
            self.t2 = (c[:, None] * U2[idx]).sum(axis=0) % P2
            self.tC = C[:, idx] @ c.astype(complex)
        if not np.any(self.t1):
            raise ValueError("the target vanishes mod P1")
        self.Q1, _ = reduce_rows(F1, U1, self.t1)


# ----------------------------------------------------------- symmetries ---

def _key(v):
    v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
    return (np.round(v, 6) + 0.0).tobytes()


def perms_from_unitaries(D, target, gens):
    """The permutations of the dictionary induced by the unitaries `gens`;
    every one is checked to fix the target up to phase and to permute D."""
    N = D.shape[1]
    index = {_key(D[:, i]): i for i in range(N)}
    if len(index) != N:
        raise AssertionError("dictionary states are not distinct up to phase")
    t = target / np.linalg.norm(target)
    perms = []
    for g in gens:
        img_t = g @ t
        if abs(abs(np.vdot(t, img_t)) - 1) > 1e-9:
            raise AssertionError("a proposed symmetry does not fix the target up to phase")
        img = g @ D
        try:
            pm = np.array([index[_key(img[:, i])] for i in range(N)])
        except KeyError:
            raise AssertionError("a proposed symmetry does not map the dictionary to itself")
        if len(np.unique(pm)) != N:
            raise AssertionError("a proposed symmetry is not a permutation of the dictionary")
        perms.append(pm)
    return perms


def sector_symmetries(cs, d, gC):
    """Unitary symmetries of g as 25 x 25 matrices: the two unit translations
    of (x, y) with their diagonal phase corrections (the order-5 Clifford
    stabilizer of |T5> on the copies, restricted to the plane), and the
    transpositions of copies that fix the string. Returns (gens, order)."""
    c1, c2, c3 = cs
    inv3 = pow(c3, -1, 5)
    gens = []

    def perm_matrix(f):
        M = np.zeros((25, 25), dtype=complex)
        for x, y in itertools.product(range(5), repeat=2):
            x2, y2 = f(x, y)
            M[5 * x2 + y2, 5 * x + y] = 1
        return M

    for a1, a2 in ((1, 0), (0, 1)):
        T = perm_matrix(lambda x, y: ((x + a1) % 5, (y + a2) % 5))
        img = T @ gC
        Dg = np.diag(gC / img)              # restores g exactly: (Dg T g) = g
        gens.append(Dg @ T)
    aut = 1
    if c1 == c2:
        gens.append(perm_matrix(lambda x, y: (y, x)))
        aut *= 2
    if c1 == c3:
        gens.append(perm_matrix(lambda x, y: (((d - c1 * x - c2 * y) * inv3) % 5, y)))
        aut = 6 if aut == 2 else 2
    elif c2 == c3:
        gens.append(perm_matrix(lambda x, y: (x, ((d - c1 * x - c2 * y) * inv3) % 5)))
        aut = 6 if aut == 2 else 2
    return gens, 25 * aut


def orbit_data(perms, N):
    parent = np.arange(N)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pm in perms:
        for a in range(N):
            ra, rb = find(a), find(int(pm[a]))
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(a) for a in range(N)])
    return np.unique(roots), roots


# --------------------------------------------------------------- census ---

class Census:
    def __init__(self, D, codes, C, U1, U2, F1, F2, T, perms, order, roots, reps):
        self.D, self.codes, self.C, self.U1, self.U2 = D, codes, C, U1, U2
        self.F1, self.F2, self.T = F1, F2, T
        self.perms, self.order, self.roots, self.reps = perms, order, roots, reps
        self.N = D.shape[1]

    def decide(self, idx):
        idx = [int(x) for x in idx]
        rows2 = [self.U2[k] for k in idx]
        r0 = rank_mod(rows2, P2)
        r1 = rank_mod(rows2 + [self.T.t2], P2)
        exact = bool(r0 == r1)
        A = self.C[:, idx]
        dvec, *_ = np.linalg.lstsq(A, self.T.tC, rcond=None)
        res = float(np.linalg.norm(A @ dvec - self.T.tC))
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        numeric = bool(res < NUM_TOL)
        return {"states": idx, "exact_mod_p2": exact, "numeric": numeric, "agree": exact == numeric,
                "rank": rank, "independent": rank == len(idx),
                "nonzero": bool(np.all(np.abs(dvec) > 1e-7)), "residual": res,
                "coeffs": [[float(z.real), float(z.imag)] for z in dvec]}

    def run(self, kmax=4, seed=17, log=print, pivots=None):
        F, T = self.F1, self.T
        rng = np.random.default_rng(seed)
        hits = {k: [] for k in range(1, kmax + 1)}
        cands = {k: 0 for k in range(1, kmax + 1)}
        units = {k: 0 for k in range(1, kmax + 1)}
        t0 = time.time()

        def consider(k, idx):
            cands[k] += 1
            dec = self.decide(idx)
            if dec["exact_mod_p2"] or dec["numeric"]:
                hits[k].append(dec)

        for l in np.flatnonzero(~T.Q1.any(axis=1)):
            consider(1, (int(l),))
        units[1] = 1
        pivot_list = self.reps if pivots is None else np.asarray(pivots)
        for n_piv, i in enumerate(pivot_list):
            i = int(i)
            if not T.Q1[i].any():
                continue                                   # a rank-1 hit, recorded above
            members = np.flatnonzero(self.roots >= self.roots[i]) if pivots is None else np.arange(self.N)
            mem = members[members != i]
            Qi, _ = reduce_rows(F, T.Q1, T.Q1[i])
            if kmax >= 2:
                units[2] += 1
                for l in mem[~Qi[mem].any(axis=1)]:
                    consider(2, tuple(sorted((i, int(l)))))
            if kmax >= 3:
                units[3] += 1
                Rc, has = canon_rows(F, Qi[mem])
                ids, Rc = mem[has], Rc[has]
                if len(ids) >= 2:
                    for grp in groups_by_key(pair_key(F, Rc, rng)):
                        for a, b in itertools.combinations(sorted(ids[grp].tolist()), 2):
                            consider(3, tuple(sorted((i, a, b))))
            if kmax >= 4:
                if pivots is None:
                    orbit_size = int(np.count_nonzero(self.roots == self.roots[i]))
                    labels, _ = stabilizer_orbit_labels(self.perms, i, stabilizer_order=self.order // orbit_size)
                    _, first = np.unique(labels[members], return_index=True)
                    partners = members[first]
                    partners = partners[partners != i]
                else:
                    partners = mem
                for j in partners:
                    j = int(j)
                    if not Qi[j].any():
                        continue                           # u_j in span(g, u_i): a rank-2 set
                    units[4] += 1
                    ids = mem[mem > j]
                    if len(ids) < 2:
                        continue
                    R, _ = reduce_rows(F, Qi[ids], Qi[j])
                    Rc, has = canon_rows(F, R)
                    ids, Rc = ids[has], Rc[has]
                    if len(ids) < 2:
                        continue
                    for grp in groups_by_key(pair_key(F, Rc, rng)):
                        for a, b in itertools.combinations(sorted(ids[grp].tolist()), 2):
                            consider(4, tuple(sorted((i, j, a, b))))
            if log and (n_piv % 10 == 0 or n_piv == len(pivot_list) - 1):
                log(f"  pivot {n_piv + 1}/{len(pivot_list)} (state {i}): hits "
                    f"{ {k: len(v) for k, v in hits.items()} }, candidates {cands} [{time.time() - t0:.0f}s]")
        return {"hits": {str(k): v for k, v in hits.items()}, "candidates": {str(k): v for k, v in cands.items()},
                "units": {str(k): v for k, v in units.items()}, "seconds": time.time() - t0}


def build(cs, d, symmetry=True, log=print):
    t0 = time.time()
    D = dictionary(5, 2)
    codes, C = patterns5(D)
    F1, F2 = Field5(P1), Field5(P2)
    U1, U2 = F1.codes_to_field(codes), F2.codes_to_field(codes)
    check_sector_reduction(cs, d)
    gcodes = sector_codes(cs, d)
    T = SectorTarget(D, gcodes, F1, F2, U1)
    if symmetry:
        gens, order = sector_symmetries(cs, d, T.tC)
        perms = perms_from_unitaries(D, T.tC, gens)
        reps, roots = orbit_data(perms, D.shape[1])
    else:
        perms, order = [], 1
        reps, roots = np.arange(D.shape[1]), np.arange(D.shape[1])
    if log:
        log(f"string Z^{cs[0]} Z^{cs[1]} Z^{cs[2]}, eigenvalue w^{d}: dictionary {D.shape[1]} states, "
            f"unitary symmetry group of order {order}, {len(reps)} orbits [{time.time() - t0:.1f}s]")
    return Census(D, codes, C, U1, U2, F1, F2, T, perms, order, roots, reps), gcodes


# ------------------------------------------------------------- controls ---

def control(log=print):
    D = dictionary(5, 2)
    codes, C = patterns5(D)
    F1, F2 = Field5(P1), Field5(P2)
    U1, U2 = F1.codes_to_field(codes), F2.codes_to_field(codes)
    N = D.shape[1]
    index = {_key(D[:, i]): i for i in range(N)}
    ok = True
    # (a) |N5>^2 = (|+> - |0>)^2 has the product decomposition {00, 0+, +0, ++}
    zero = np.eye(5, dtype=complex)[0]
    plus = np.ones(5, dtype=complex)
    n5 = plus - zero                                        # entries 0, 1: codes
    psi = np.kron(n5, n5)
    pcodes = np.where(np.abs(psi) > 0.5, 1, 0).astype(np.int8)
    T = SectorTarget(D, pcodes, F1, F2, U1)
    cen = Census(D, codes, C, U1, U2, F1, F2, T, [], 1, np.arange(N), np.arange(N))
    piv = index[_key(np.kron(zero, zero))]
    rec = cen.run(kmax=4, log=None, pivots=[piv])
    want = sorted(index[_key(np.kron(a, b))] for a in (zero, plus) for b in (zero, plus))
    found = [h["states"] for h in rec["hits"]["4"]]
    got = want in found
    log(f"control |N5>^2 from pivot |00>: {len(found)} rank-4 sets, product decomposition "
        f"{'found' if got else 'MISSING'}; k = 1, 2, 3 hits "
        f"{[len(rec['hits'][k]) for k in '123']}")
    ok &= got
    # (b) planted rank-4 targets: 12 random independent 4-sets, coefficients 1..4 permuted
    rng = np.random.default_rng(5)
    n_ok = 0
    for trial in range(12):
        while True:
            states = sorted(rng.choice(N, 4, replace=False).tolist())
            if np.linalg.matrix_rank(C[:, states], tol=1e-8) == 4:
                break
        coeffs = rng.permutation([1, 2, 3, 4]).tolist()
        T = SectorTarget(D, None, F1, F2, U1, planted=(states, coeffs, U2, C))
        cen = Census(D, codes, C, U1, U2, F1, F2, T, [], 1, np.arange(N), np.arange(N))
        rec = cen.run(kmax=4, log=None, pivots=[states[0]])
        found = [h["states"] for h in rec["hits"]["4"]]
        hit = states in found
        n_ok += hit
        if not hit:
            log(f"  planted {states} x {coeffs}: MISSING (found {found[:3]})")
    log(f"control planted rank-4 targets: {n_ok} of 12 recovered from their smallest member")
    ok &= n_ok == 12
    # (c) the one-ququint cell: |T5> has rank 3, found by the k = 3 census
    D1 = dictionary(5, 1)
    codes1, C1 = patterns5(D1)
    U11, U21 = F1.codes_to_field(codes1), F2.codes_to_field(codes1)
    tcodes = np.array([(x ** 3) % 5 + 1 for x in range(5)], dtype=np.int8)
    T = SectorTarget(D1, tcodes, F1, F2, U11)
    cen = Census(D1, codes1, C1, U11, U21, F1, F2, T, [], 1, np.arange(30), np.arange(30))
    rec = cen.run(kmax=3, log=None, pivots=list(range(30)))
    # with explicit pivots every set is listed once per member, so count sets
    n3 = len({tuple(h["states"]) for h in rec["hits"]["3"]})
    log(f"control |T5>: k = 1, 2 hits {[len(rec['hits'][k]) for k in '12']}, {n3} distinct rank-3 "
        f"decompositions (10 expected, all pivots)")
    ok &= n3 == 10 and not rec["hits"]["1"] and not rec["hits"]["2"]
    log("controls PASSED" if ok else "controls FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------- main ---

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "control"])
    ap.add_argument("cs", nargs="*", type=int)
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[1:])
    try:
        os.nice(19)
    except OSError:
        pass
    if a.cmd == "control":
        return control()
    if len(a.cs) != 3:
        ap.error("census takes the three exponents c1 c2 c3")
    cs = tuple(c % 5 for c in a.cs)
    cen, gcodes = build(cs, a.d % 5, symmetry=not a.no_symmetry)
    rec = cen.run(kmax=a.kmax, seed=a.seed)
    counts = {k: len(v) for k, v in rec["hits"].items()}
    rank = next((int(k) for k in sorted(rec["hits"]) if rec["hits"][k]), None)
    print(f"hits per k: {counts}; candidates {rec['candidates']}; units {rec['units']}; "
          f"{rec['seconds']:.0f}s")
    if rank is None:
        five = a.kmax == 4 and len(set(cs)) < 3       # two equal exponents: five line terms
        print(f"VERDICT: the Z^{cs[0]} Z^{cs[1]} Z^{cs[2]} sector of |T5>^3 has stabilizer rank >= {a.kmax + 1}"
              + (" (= 5 with the five sub-sector products)" if five else ""))
    else:
        print(f"VERDICT: the Z^{cs[0]} Z^{cs[1]} Z^{cs[2]} sector of |T5>^3 has stabilizer rank {rank}; "
              f"{counts[str(rank)]} decomposition(s) listed (one per symmetry orbit at least)")
        for h in rec["hits"][str(rank)][:10]:
            print("   ", h["states"], "exact" if h["exact_mod_p2"] else "", "numeric" if h["numeric"] else "",
                  f"residual {h['residual']:.1e}")
    body = {"string": list(cs), "d": a.d % 5, "target_codes": gcodes.tolist(), "kmax": a.kmax,
            "symmetry": not a.no_symmetry, "group_order": cen.order, "pivots": int(len(cen.reps)),
            "dictionary_size": int(cen.N), "seed": a.seed, "rank": rank if rank is not None else f">{a.kmax}",
            **rec}
    out = a.out or os.path.join(RESULTS, f"census_Z{cs[0]}{cs[1]}{cs[2]}_d{a.d % 5}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    body["sha256"] = sha256_json(body)
    with open(out, "w") as f:
        json.dump(body, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
