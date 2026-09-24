"""Stages (beta') and (gamma) of the rank-6 exclusion of |N>^4 at p = 3: the
geometry of the flats missing the base point (2, 2), the exact-point
solution counts on real bases, and a prototype of the invisible-flat
matcher for the flats that miss both coordinate points
(docs/notes/n4_rank6_design.md, section 6).

    invisible3_probe.py geometry
        The 16 flats of F_3^2 missing (2, 2), which coordinate points each
        contains, the slice ratios, and the 136 flat multisets of stage
        (gamma).
    invisible3_probe.py points [--count K]
        For K full 5-covers and K full 4-covers of |N>^2 (evenly spaced in
        the census lists), the exact slice equation at each of the eight
        other points with every option allowed (the point family, 28
        options per term, meet in the middle and exact decision): solution
        counts per point and seconds per solve. The first exact point of a
        (base, flat) run is one of these.
    invisible3_probe.py beta [--count K] [--plant P]
        The (beta') prototype on K full 5-covers at every flat missing both
        coordinate points (9 of 16): both coordinate points exact, the join,
        the composite exact points over the shapes alive, the birth scan of
        the invisible term at the flat's first point (residual against the
        360-state table), the line's other points with the fresh term's 27
        phased translates, assembly and confirmation. Per-run times and
        where the runs die. P planted instances (a random invisible term on
        each such flat) must be recovered.
    invisible3_probe.py gamma [--count K] [--plant P]
        The same for K full 4-covers at the 45 flat multisets whose two
        flats both miss the coordinate points, with the rank-2 residual
        scan when the two flats share their first point.
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
sys.path.insert(0, HERE)
from cover_census import CoverEnumerator3, P1, P2, _canon_rows  # noqa: E402
from matcher import (COMP, E1, E2, PTS, Family, Matcher, add, exact_codes, family_from, pidx, psi_target,  # noqa: E402
                     rank_mod, restrict, slice_base, slice_system, solve_slice3, vector_target)
from common import qcommon  # noqa: E402  (research/n4_rank6/common.py loads the rank-5 common by file)
from probe6 import random_shape_term  # noqa: E402

RESULTS = os.path.join(HERE, "results")
X0 = (2, 2)
DIRS = [(0, 1), (1, 0), (1, 1), (1, 2)]
ALPHA = {0: 1, 1: 1, 2: -2}


def log(s):
    print(s, flush=True)


def sub(x, y):
    return ((x[0] - y[0]) % 3, (x[1] - y[1]) % 3)


def flats_missing(x0=X0):
    """The affine flats of F_3^2 missing x0: points and lines, as
    (name, kind, points in a fixed order, direction or None)."""
    out = []
    for y in PTS:
        if y != x0:
            out.append((f"p{y[0]}{y[1]}", "point", (y,), None))
    seen = set()
    for v in DIRS:
        for b in PTS:
            pts = tuple(sorted({add(b, (t * v[0] % 3, t * v[1] % 3)) for t in range(3)}))
            if pts in seen or x0 in pts:
                continue
            seen.add(pts)
            out.append((f"L{v[0]}{v[1]}_" + "".join(f"{p[0]}{p[1]}" for p in pts), "line", pts, v))
    assert len(out) == 16 and sum(1 for f in out if f[1] == "line") == 8
    return out


def ratio(y, x0=X0):
    return ALPHA[y[0]] * ALPHA[y[1]] / (ALPHA[x0[0]] * ALPHA[x0[1]])


def coord_points(x0=X0):
    return add(x0, E1), add(x0, E2)


def geometry(a):
    fl = flats_missing()
    c1, c2 = coord_points()
    rec = {"x0": list(X0), "coordinate_points": [list(c1), list(c2)], "flats": []}
    for name, kind, pts, v in fl:
        rec["flats"].append({"name": name, "kind": kind, "points": [list(p) for p in pts],
                             "contains_coordinate_points": [list(p) for p in pts if p in (c1, c2)],
                             "ratios": [ratio(p) for p in pts]})
    both = [f for f in rec["flats"] if not f["contains_coordinate_points"]]
    one = [f for f in rec["flats"] if len(f["contains_coordinate_points"]) == 1]
    two = [f for f in rec["flats"] if len(f["contains_coordinate_points"]) == 2]
    pairs = list(itertools.combinations_with_replacement(range(16), 2))
    both_idx = {i for i, f in enumerate(rec["flats"]) if not f["contains_coordinate_points"]}
    rec["gamma_flat_multisets"] = len(pairs)
    rec["gamma_pairs_missing_both_coordinate_points"] = sum(1 for p in pairs if p[0] in both_idx and p[1] in both_idx)
    rec["flats_missing_both_coordinate_points"] = [f["name"] for f in both]
    rec["flats_with_one_coordinate_point"] = [f["name"] for f in one]
    rec["flats_with_both_coordinate_points"] = [f["name"] for f in two]
    rec["ratios_from_x0"] = {f"{y[0]}{y[1]}": ratio(y) for y in PTS if y != X0}
    log(json.dumps(rec, indent=1))
    with open(os.path.join(RESULTS, "invisible3_geometry.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return 0


class Proto:
    """The (beta') and (gamma) prototype for flats missing both coordinate
    points, over a Matcher's option tables."""

    def __init__(self, Mt, target, seed=13):
        self.M = Mt
        self.target = target
        self.rng = np.random.default_rng(seed)
        # the dictionary table for the birth scan: normalized rows mod P1,
        # keyed by two random functionals
        Rc, has = _canon_rows(Mt.F1, Mt.U1)
        assert has.all()
        self.f1 = self.rng.integers(1, P1, size=9)
        self.f2 = self.rng.integers(1, P1, size=9)
        keys = ((Rc @ self.f1) % P1) * P1 + ((Rc @ self.f2) % P1)
        self.table = {}
        for i, k in enumerate(keys.tolist()):
            self.table.setdefault(k, []).append(i)
        self.Rc = Rc

    def exact_point(self, arrays, fam, y):
        return solve_slice3(arrays, [], fam, self.target.rhs(y), self.rng, stats={}, where=f"exact {y}")

    def birth_scan(self, arrays, d1, d2, dC, y):
        """Residuals rhs_y - sum d_i w_i over every combination of the given
        option arrays; those parallel to a dictionary state (mod P1, then
        exactly over C and mod P2) are returned as (combo, state index,
        coefficient over C, coefficient mod P1, coefficient mod P2)."""
        r1, r2, rC = self.target.rhs(y)
        sizes = [len(o[0]) for o in arrays]
        grids = np.indices(sizes).reshape(len(sizes), -1).T                        # (n, r)
        R = np.tile(r1 % P1, (len(grids), 1))
        for i, o in enumerate(arrays):
            R = (R - int(d1[i]) * o[0][grids[:, i]]) % P1
        Rc, has = _canon_rows(self.M.F1, R)
        keys = ((Rc @ self.f1) % P1) * P1 + ((Rc @ self.f2) % P1)
        out = []
        for n in np.flatnonzero(has):
            for v in self.table.get(int(keys[n]), []):
                if np.any(Rc[n] != self.Rc[v]):
                    continue
                # exact: r = c v over the three fields
                combo = tuple(int(c) for c in grids[n])
                rc = rC - sum(dC[i] * arrays[i][2][combo[i]] for i in range(len(arrays)))
                vC = self.M.C[:, v]
                j = np.flatnonzero(np.abs(vC) > 1e-9)[0]
                cC = rc[j] / vC[j]
                if np.linalg.norm(rc - cC * vC) > 1e-7 * max(1.0, np.linalg.norm(rc)) or abs(cC) < 1e-9:
                    continue
                rr1 = (r1 - sum(int(d1[i]) * arrays[i][0][combo[i]] for i in range(len(arrays)))) % P1
                rr2 = (r2 - sum((int(d2[i]) % P2) * arrays[i][1][combo[i]] % P2 for i in range(len(arrays)))) % P2
                v1, v2 = self.M.U1[v], self.M.U2[v]
                j1 = np.flatnonzero(v1)[0]
                c1 = (int(rr1[j1]) * pow(int(v1[j1]), P1 - 2, P1)) % P1
                c2 = (int(rr2[j1]) * pow(int(v2[j1]), P2 - 2, P2)) % P2
                if np.any((rr1 - c1 * v1) % P1) or np.any((rr2 - c2 * v2) % P2):
                    continue
                out.append((combo, int(v), cC, c1, c2))
        return out, len(grids)

    def rank2_scan(self, arrays, d1, d2, dC, y):
        """Residuals that are c_4 v_4 + c_5 v_5 for two distinct dictionary
        states (projective hash modulo the residual, exact decision), plus
        the count of residuals that are a single state or zero (the
        same-state cases, left to the design)."""
        r1, r2, rC = self.target.rhs(y)
        sizes = [len(o[0]) for o in arrays]
        grids = np.indices(sizes).reshape(len(sizes), -1).T
        R = np.tile(r1 % P1, (len(grids), 1))
        for i, o in enumerate(arrays):
            R = (R - int(d1[i]) * o[0][grids[:, i]]) % P1
        out, same, zero = [], 0, 0
        for n in range(len(grids)):
            r = R[n]
            nz = np.flatnonzero(r)
            if not len(nz):
                zero += 1
                continue
            c = int(nz[0])
            f = (self.M.U1[:, c] * pow(int(r[c]), P1 - 2, P1)) % P1
            res = np.delete((self.M.U1 - f[:, None] * r[None, :]) % P1, c, axis=1)
            Rc, has = _canon_rows(self.M.F1, res)
            same += int((~has).sum())
            ids = np.flatnonzero(has)
            keys = ((Rc[ids] @ self.f1[:8]) % P1) * P1 + ((Rc[ids] @ self.f2[:8]) % P1)
            order = np.argsort(keys, kind="stable")
            sk = keys[order]
            brk = np.flatnonzero(sk[1:] != sk[:-1]) + 1
            for grp in np.split(order, brk):
                if len(grp) < 2:
                    continue
                for a, b in itertools.combinations(sorted(ids[grp].tolist()), 2):
                    combo = tuple(int(c) for c in grids[n])
                    rc = rC - sum(dC[i] * arrays[i][2][combo[i]] for i in range(len(arrays)))
                    A = self.M.C[:, [a, b]]
                    coef, *_ = np.linalg.lstsq(A, rc, rcond=None)
                    if np.linalg.norm(A @ coef - rc) > 1e-7 or np.any(np.abs(coef) < 1e-9):
                        continue
                    M2 = np.vstack([self.M.U2[a], self.M.U2[b]])
                    rr2 = (r2 - sum((int(d2[i]) % P2) * arrays[i][1][combo[i]] % P2 for i in range(len(arrays)))) % P2
                    if rank_mod(np.vstack([M2, rr2[None, :]]), P2) != 2:
                        continue
                    out.append((combo, a, b, coef))
        return out, len(grids), same, zero

    def run_beta(self, cover, flat, plant_terms=None):
        """One (5-cover, flat) run for a flat missing both coordinate points.
        Returns (hits, stats)."""
        M = self.M
        name, kind, pts, v = flat
        st = {"flat": name, "stage": None, "seconds": {}, "coord": None, "joined": 0, "exact_states": [],
              "birth_candidates": 0, "line_states": 0, "hits": 0}
        t0 = time.time()
        distinct = sorted(set(int(u) for u in cover))
        if len(distinct) != len(cover):
            st["stage"] = "repeated base: outside the prototype"
            return [], st
        b1, b2, bC = self.target.rhs(X0)
        fam = family_from(M.U1[distinct], M.U2[distinct], M.C[:, distinct], b1, b2, bC)
        if fam is None or fam.kappa != 0:
            st["stage"] = f"family kappa {None if fam is None else fam.kappa}: outside the prototype"
            return [], st
        opts = [M.options(u) for u in distinct]
        arrays = [o.arrays() for o in opts]
        r = len(opts)
        # the two coordinate points are exact
        s1 = self.exact_point(arrays, fam, add(X0, E1))
        s2 = self.exact_point(arrays, fam, add(X0, E2)) if s1 else []
        st["coord"] = [len(s1), len(s2)]
        st["seconds"]["coord"] = round(time.time() - t0, 3)
        if not s1 or not s2:
            st["stage"] = "coordinate point"
            return [], st
        states = []
        for (c1, _, f1), (c2, _, f2) in itertools.product(s1, s2):
            rows = [opts[i].composite_rows(c1[i], c2[i]) for i in range(r)]
            states.append(([c1, c2], [np.ones(len(R), dtype=bool) for R in rows], rows))
        st["joined"] = len(states)
        # composite exact points: every composite offset whose point is not on the flat
        comp_pts = [(c, x) for c, x in enumerate(COMP) if add(X0, x) not in pts]
        flat_pts = [(c, x) for c, x in enumerate(COMP) if add(X0, x) in pts]
        d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
        for c, x in comp_pts:
            y = add(X0, x)
            new = []
            for cl, alive, rows in states:
                codes = [np.unique(rows[i][alive[i], c]) for i in range(r)]
                arr = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                for combo, _, _ in self.exact_point(arr, fam, y):
                    chosen = tuple(int(codes[i][combo[i]]) for i in range(r))
                    alive2 = [alive[i] & (rows[i][:, c] == chosen[i]) for i in range(r)]
                    new.append((cl + [chosen], alive2, rows))
            states = new
            st["exact_states"].append(len(states))
            if not states:
                st["stage"] = f"composite exact point {y}"
                st["seconds"]["total"] = round(time.time() - t0, 3)
                return [], st
        # the flat's first point: the birth scan
        c_first, x_first = flat_pts[0]
        y_first = add(X0, x_first)
        born = []
        for cl, alive, rows in states:
            codes = [np.unique(rows[i][alive[i], c_first]) for i in range(r)]
            arr = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
            found, ncomb = self.birth_scan(arr, d1, d2, dC, y_first)
            for combo, vidx, cC, c1, c2 in found:
                chosen = tuple(int(codes[i][combo[i]]) for i in range(r))
                alive2 = [alive[i] & (rows[i][:, c_first] == chosen[i]) for i in range(r)]
                born.append((cl + [chosen], alive2, rows, vidx, (c1, c2, cC)))
        st["birth_candidates"] = len(born)
        st["seconds"]["birth"] = round(time.time() - t0, 3)
        if not born:
            st["stage"] = f"birth point {y_first}"
            st["seconds"]["total"] = round(time.time() - t0, 3)
            return [], st
        # the line's other points: the fresh term as an ordinary term with its coefficient pinned
        hits = []
        for cl, alive, rows, vidx, (c1, c2, cC) in born:
            ov = M.options(vidx)
            fam6 = Family([(np.append(d1, c1), np.zeros((r + 1, 0), dtype=np.int64)),
                           (np.append(d2, c2), np.zeros((r + 1, 0), dtype=np.int64)),
                           (np.append(dC, cC), np.zeros((r + 1, 0), dtype=complex))])
            sub_states = [(cl, alive, [])]
            for c, x in flat_pts[1:]:
                y = add(X0, x)
                new = []
                for cl2, alive2, fresh_codes in sub_states:
                    codes = [np.unique(rows[i][alive2[i], c]) for i in range(r)]
                    arr = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                    arr.append((ov.m1[:27], ov.m2[:27], ov.vecs[:27]))
                    for combo, _, _ in self.exact_point(arr, fam6, y):
                        chosen = tuple(int(codes[i][combo[i]]) for i in range(r))
                        alive3 = [alive2[i] & (rows[i][:, c] == chosen[i]) for i in range(r)]
                        new.append((cl2 + [chosen], alive3, fresh_codes + [int(combo[r])]))
                sub_states = new
            st["line_states"] += len(sub_states)
            for cl2, alive2, fresh_codes in sub_states:
                terms = []
                for i in range(r):
                    t = np.zeros((9, 9), dtype=complex)
                    t[pidx(X0)] = opts[i].u
                    t[pidx(add(X0, E1))] = opts[i].vecs[cl2[0][i]]
                    t[pidx(add(X0, E2))] = opts[i].vecs[cl2[1][i]]
                    k = 2
                    for c, x in comp_pts + flat_pts:
                        t[pidx(add(X0, x))] = opts[i].vecs[cl2[k][i]]
                        k += 1
                    terms.append(t.ravel())
                t = np.zeros((9, 9), dtype=complex)
                t[pidx(y_first)] = ov.u
                for (c, x), code in zip(flat_pts[1:], fresh_codes):
                    t[pidx(add(X0, x))] = ov.vecs[code]
                terms.append(t.ravel())
                h = M.confirm(terms, 0, self.target)
                if h["exact"] and h["residual"] < 1e-7 and h["nonzero"] and h["independent"]:
                    hits.append(h)
        st["hits"] = len(hits)
        st["stage"] = "assembled"
        st["seconds"]["total"] = round(time.time() - t0, 3)
        return hits, st


def load_bases():
    cen = qcommon.load_census("N")
    return [tuple(c) for c in cen["covers5"]], [tuple(c) for c in cen["covers4"]]


def points(a):
    try:
        os.nice(19)
    except OSError:
        pass
    E = CoverEnumerator3("N", 2)
    Mt = Matcher(E.D, 2, E.F1, E.F2)
    target = psi_target("N", 2, Mt.F1, Mt.F2)
    P = Proto(Mt, target)
    c5, c4 = load_bases()
    rec = {"x0": list(X0), "bases": {}}
    for name, lst in (("5-covers", c5), ("4-covers", c4)):
        pick = [lst[t] for t in np.linspace(0, len(lst) - 1, min(a.count, len(lst))).astype(int)]
        per_point = {}
        for y in PTS:
            if y == X0:
                continue
            counts, secs = [], []
            for cover in pick:
                distinct = sorted(cover)
                b1, b2, bC = target.rhs(X0)
                fam = family_from(Mt.U1[distinct], Mt.U2[distinct], Mt.C[:, distinct], b1, b2, bC)
                arrays = [Mt.options(u).arrays() for u in distinct]
                t0 = time.time()
                sols = P.exact_point(arrays, fam, y)
                secs.append(time.time() - t0)
                counts.append(len(sols))
            counts = np.array(counts)
            per_point[f"{y[0]}{y[1]}"] = {"ratio": ratio(y), "zero_fraction": float(np.mean(counts == 0)),
                                         "mean_solutions": float(counts.mean()), "max_solutions": int(counts.max()),
                                         "hist": {str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
                                         "ms_per_solve": 1e3 * float(np.mean(secs))}
            log(f"{name} at {y} (ratio {ratio(y):+.2f}): no solution for {100 * np.mean(counts == 0):.1f} % of "
                f"{len(pick)} bases, mean {counts.mean():.2f}, max {counts.max()}, {1e3 * np.mean(secs):.1f} ms per solve")
        rec["bases"][name] = {"count": len(pick), "points": per_point}
    out = os.path.join(RESULTS, "invisible3_points.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


def plant_invisible(Mt, base, flat, rng):
    """A planted six-term target: five visible terms with random flats
    through X0 over the base states, one invisible term on `flat` with a
    random dictionary state at its first point."""
    name, kind, pts, v = flat
    terms = []
    for u in base:
        t, _ = random_shape_term(Mt.options(u), X0, rng)
        terms.append(t)
    vidx = int(rng.integers(0, Mt.N))
    ov = Mt.options(vidx)
    if kind == "point":
        t = np.zeros((9, 9), dtype=complex)
        t[pidx(pts[0])] = ov.u
        inv = t.ravel()
    else:
        kindname = {(1, 0): "line1", (0, 1): "line2", (1, 1): "diag1", (1, 2): "diag2"}[v]
        inv, _ = random_shape_term(ov, pts[0], rng, kind=kindname)
    terms.append(inv)
    T = np.column_stack(terms)
    coeffs = rng.integers(1, 4, size=6) * rng.choice([1, -1], size=6)
    return T, coeffs, vidx


def beta(a):
    try:
        os.nice(19)
    except OSError:
        pass
    E = CoverEnumerator3("N", 2)
    Mt = Matcher(E.D, 2, E.F1, E.F2)
    target = psi_target("N", 2, Mt.F1, Mt.F2)
    P = Proto(Mt, target)
    c1, c2 = coord_points()
    flats = [f for f in flats_missing() if c1 not in f[2] and c2 not in f[2]]
    c5, _ = load_bases()
    rng = np.random.default_rng(a.seed)
    rec = {"x0": list(X0), "flats": [f[0] for f in flats], "planted": {}, "real": {}}
    ok = True
    # planted
    for flat in flats:
        got, tsum = 0, 0.0
        for _ in range(a.plant):
            while True:
                base = c5[int(rng.integers(0, len(c5)))]
                T, coeffs, vidx = plant_invisible(Mt, base, flat, rng)
                if np.linalg.matrix_rank(T, tol=1e-8) == 6:
                    break
            vec = T @ coeffs.astype(complex)
            tgt = vector_target(vec, 2, Mt.F1, Mt.F2)
            Pp = Proto(Mt, tgt, seed=int(rng.integers(1, 1 << 30)))
            t0 = time.time()
            hits, st = Pp.run_beta(base, flat)
            tsum += time.time() - t0
            wkey = sorted(exact_codes(T[:, k])[0].tobytes() for k in range(6))
            found = any(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == wkey for h in hits)
            got += found
            if not found:
                log(f"  planted NOT recovered on {flat[0]}: base {base}, stats {st}")
                ok = False
        rec["planted"][flat[0]] = {"planted": a.plant, "recovered": got, "seconds": tsum / max(1, a.plant)}
        log(f"planted on {flat[0]}: {got}/{a.plant} recovered, {tsum / max(1, a.plant):.2f} s per run")
    # real bases
    pick = [c5[t] for t in np.linspace(0, len(c5) - 1, min(a.count, len(c5))).astype(int)]
    for flat in flats:
        secs, stages, hits_n = [], {}, 0
        for base in pick:
            t0 = time.time()
            hits, st = P.run_beta(base, flat)
            secs.append(time.time() - t0)
            stages[st["stage"]] = stages.get(st["stage"], 0) + 1
            hits_n += len(hits)
        rec["real"][flat[0]] = {"bases": len(pick), "ms_per_run": 1e3 * float(np.mean(secs)),
                                "max_ms": 1e3 * float(np.max(secs)), "died_at": stages, "hits": hits_n}
        log(f"{flat[0]} ({flat[1]}): {len(pick)} real bases, {1e3 * np.mean(secs):.1f} ms per run (max "
            f"{1e3 * np.max(secs):.0f}), died at {stages}, hits {hits_n}")
    rec["pass"] = ok
    out = os.path.join(RESULTS, "invisible3_beta.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"beta prototype: {'PASS' if ok else 'FAIL'}; wrote {os.path.relpath(out, ROOT)}")
    return 0 if ok else 1


def gamma(a):
    """(gamma) on 4-covers: the exact points, the joined states, and the
    birth or rank-2 scan at the first point of the flats; the continuation
    along the flats is the (beta') code applied twice and is not
    prototyped here beyond the scans."""
    try:
        os.nice(19)
    except OSError:
        pass
    E = CoverEnumerator3("N", 2)
    Mt = Matcher(E.D, 2, E.F1, E.F2)
    target = psi_target("N", 2, Mt.F1, Mt.F2)
    P = Proto(Mt, target)
    c1, c2 = coord_points()
    flats = [f for f in flats_missing() if c1 not in f[2] and c2 not in f[2]]
    pairs = list(itertools.combinations_with_replacement(range(len(flats)), 2))
    _, c4 = load_bases()
    pick = [c4[t] for t in np.linspace(0, len(c4) - 1, min(a.count, len(c4))).astype(int)]
    rec = {"x0": list(X0), "pairs": len(pairs), "rows": []}
    t_all = time.time()
    for base in pick:
        distinct = sorted(base)
        b1, b2, bC = target.rhs(X0)
        fam = family_from(Mt.U1[distinct], Mt.U2[distinct], Mt.C[:, distinct], b1, b2, bC)
        if fam is None or fam.kappa != 0:
            continue
        opts = [Mt.options(u) for u in distinct]
        arrays = [o.arrays() for o in opts]
        d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
        t0 = time.time()
        s1 = P.exact_point(arrays, fam, add(X0, E1))
        s2 = P.exact_point(arrays, fam, add(X0, E2)) if s1 else []
        row = {"base": list(base), "coord": [len(s1), len(s2)], "pairs": {}}
        if s1 and s2:
            states = []
            for (cc1, _, _), (cc2, _, _) in itertools.product(s1, s2):
                rows = [opts[i].composite_rows(cc1[i], cc2[i]) for i in range(4)]
                states.append(([cc1, cc2], [np.ones(len(R), dtype=bool) for R in rows], rows))
            for pa, pb in pairs:
                fa, fb = flats[pa], flats[pb]
                on = set(fa[2]) | set(fb[2])
                comp_pts = [(c, x) for c, x in enumerate(COMP) if add(X0, x) not in on]
                st_states = states
                died = None
                for c, x in comp_pts:
                    y = add(X0, x)
                    new = []
                    for cl, alive, rows in st_states:
                        codes = [np.unique(rows[i][alive[i], c]) for i in range(4)]
                        arr = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                        for combo, _, _ in P.exact_point(arr, fam, y):
                            chosen = tuple(int(codes[i][combo[i]]) for i in range(4))
                            new.append((cl + [chosen], [alive[i] & (rows[i][:, c] == chosen[i]) for i in range(4)], rows))
                    st_states = new
                    if not st_states:
                        died = f"exact {y}"
                        break
                scan = None
                if st_states:
                    ya, yb = fa[2][0], fb[2][0]
                    ca = COMP.index(sub(ya, X0))
                    n_b, n_r2, n_same, n_zero = 0, 0, 0, 0
                    for cl, alive, rows in st_states:
                        codes = [np.unique(rows[i][alive[i], ca]) for i in range(4)]
                        arr = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                        if ya == yb:
                            found, _, same, zero = P.rank2_scan(arr, d1, d2, dC, ya)
                            n_r2 += len(found)
                            n_same += same
                            n_zero += zero
                        else:
                            found, _ = P.birth_scan(arr, d1, d2, dC, ya)
                            n_b += len(found)
                    scan = {"states": len(st_states), "birth": n_b, "rank2": n_r2, "same_state": n_same, "zero": n_zero}
                row["pairs"][f"{fa[0]}+{fb[0]}"] = {"died": died, "scan": scan}
        row["seconds"] = time.time() - t0
        rec["rows"].append(row)
        surv = sum(1 for v in row["pairs"].values() if v["died"] is None)
        log(f"4-cover {base}: coordinate solutions {row['coord']}, {surv}/{len(pairs)} flat pairs reach the scans, "
            f"{row['seconds']:.2f} s for all pairs")
    rec["seconds"] = time.time() - t_all
    out = os.path.join(RESULTS, "invisible3_gamma.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    g = sub_.add_parser("geometry")
    g.set_defaults(fn=geometry)
    p = sub_.add_parser("points")
    p.add_argument("--count", type=int, default=100)
    p.set_defaults(fn=points)
    b = sub_.add_parser("beta")
    b.add_argument("--count", type=int, default=100)
    b.add_argument("--plant", type=int, default=2)
    b.add_argument("--seed", type=int, default=5)
    b.set_defaults(fn=beta)
    c = sub_.add_parser("gamma")
    c.add_argument("--count", type=int, default=30)
    c.set_defaults(fn=gamma)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
