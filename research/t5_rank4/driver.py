"""Setup, census, rates, partition and controls for the rank-4 exclusion of
|T>^5 by a two-qubit base slice (docs/notes/t5_rank4_exclusion.md). The
batches run through batch.py and are checked by aggregate.py.

Every rank-4 decomposition of |T>^5 has, along qubits 1, 2, an all-visible
base at x_0 = 00 or x_0 = 01 (Facts 1 and 2 and the (1, 1) exclusion of the
note), so the base is a full 4-cover of |T>^3: four three-qubit stabilizer
states, repeats allowed, whose span contains psi_3 with all coefficients
nonzero. The census (`census --write`, covers4.json) lists every such
cover up to the unitary symmetry of psi_3: the covers of distinct
independent states from CoverEnumerator.covers(4) (kind A), and the
degenerate multisets over the full 3-covers (a state of the span, kind B;
a repeated state, kind C). Every cover is matched at both base points by
verify_challenge/slice_cover.SliceMatcher with orbit qubit_T.

Commands
  tables                    recompute the residual tables behind Fact 1 and the
                            (1, 1) exclusion (tables.py as a subprocess)
  census [--write]          enumerate the full 4-covers of |T>^3 and, with
                            --write, store them as covers4.json with their hash
  control-census            the census against the numeric enumerator of
                            slice_lift.all_decompositions (5,205 rank-4 tuples,
                            the full ones canonicalized under the symmetry group)
                            and the 3-covers against the stored rank-3 list
  control-witness [--reference] [--base K] [--max-cand N]
                            recover the rank-6 witness bounds/qubit_T-m5-upper-6.json
                            from each of its all-visible (pair, base point) slices
  control-planted [--plant K] [--seed S]
                            recover planted four-term instances of five kinds
                            (planes, a diagonal line, a repeated base state, a
                            line with a repeated state, a coordinate line)
  control-m4-pair           recover the rank-3 decomposition of |T>^4 from the
                            full 3-covers of |T>^2 along a qubit pair
  sample [--count N] [--reference]
                            time the matcher on covers of every kind at both base
                            points; the rates go to results/rates.json
  partition [--target-s S] [--pod-factor F]
                            write partition.json over the census by kind

Running a batch: `batch.py K`; checking the results: `aggregate.py`.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (research/t5_rank4/common.py)
from common import (CENSUS, HERE, M, N1, N2, ORBIT, PARTITION, RANK, RATES, RESULTS, ROOT,  # noqa: E402
                    X0S, codes_key, genuine, git_commit, kind_of, make_enumerator)
from slice_cover import (CoverEnumerator, Family, SliceMatcher, UnpinnedFamily, apply_pauli,  # noqa: E402
                         exact_codes, pauli_reps, patterns)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- census ------

def degenerate_covers(E, r, minimal):
    """Full r-multisets over the minimal covers `minimal` (full chi-covers of
    E's target, one per orbit) with r = chi + 1: every multiset T + (x,)
    with x in T (a repeated state) or x in span(T) (four distinct dependent
    states), with a coefficient family in which no unrepeated state is dead.
    Why nothing else (docs/notes/t5_rank4_exclusion.md, section 3): with a
    repeated state the distinct states number chi with merged coefficients;
    a nonzero merged coefficient makes them a full chi-cover T with the
    repeated state in T, a zero one leaves chi - 1 states covering psi,
    against chi(psi) = chi; r distinct dependent states have rank chi and
    contain a full chi-cover T with the last state in span(T); and there is
    no cancel-at-base multiset, since two cancelling copies leave chi - 1
    states covering psi."""
    out = set()
    chi = len(minimal[0]) if minimal else r - 1
    assert r == chi + 1 and all(len(T) == chi for T in minimal)

    def ok(ms):
        distinct = sorted(set(ms))
        fam = Family.from_cover(E, distinct)
        if fam is None:
            return False
        exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
        return not fam.has_zero_coefficient(exempt)

    for T in minimal:
        span = [x for x in range(E.N) if x not in T and E.rank_mod2(tuple(T) + (x,), False) == chi]
        for x in list(T) + span:
            ms = tuple(sorted(T + (x,)))
            if ok(ms):
                out.add(ms)
    return sorted(out)


def census_lists(E, verbose=False):
    """(covers3, covers4, degenerate, bases, kinds): the full 3-covers and
    4-covers of distinct independent states of E's target, the degenerate
    4-multisets over the 3-covers, their union sorted, and the kind (A, B,
    C) of every base."""
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    deg = degenerate_covers(E, 4, covers3)
    bases = sorted(set(covers4) | set(deg))
    kinds = [kind_of(E, c) for c in bases]
    for c, k in zip(bases, kinds):
        if (c in set(covers4)) != (k == "A"):
            raise AssertionError(f"cover {c} of kind {k} and the independent list disagree")
    return covers3, covers4, deg, bases, kinds


def census(args):
    E = make_enumerator()
    t0 = time.time()
    covers3, covers4, deg, bases, kinds = census_lists(E)
    by = {}
    for ms in deg:
        by[str(common.multiplicity_pattern(ms))] = by.get(str(common.multiplicity_pattern(ms)), 0) + 1
    span_states = {T: [x for x in range(E.N) if x not in T and E.rank_mod2(tuple(T) + (x,), False) == 3]
                   for T in covers3}
    dt = time.time() - t0
    n_kind = {k: kinds.count(k) for k in ("A", "B", "C")}
    print(f"{ORBIT} psi_3: {E.N} states, unitary symmetry group of order {E.info['order']}, "
          f"{E.info['orbits']} orbits; {len(covers3)} full 3-covers, {len(covers4)} full 4-covers of distinct "
          f"independent states, {len(deg)} degenerate 4-multisets by pattern {by}, states in span(T): "
          f"{ {str(T): len(v) for T, v in span_states.items()} }; {len(bases)} bases by kind {n_kind} [{dt:.0f}s]")
    if args.write:
        rec = {"orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
               "orbits": int(E.info["orbits"]),
               "covers3": [list(T) for T in covers3], "span_states": {str(list(T)): v for T, v in span_states.items()},
               "independent": len(covers4), "dependent": n_kind["B"], "repeated": n_kind["C"],
               "degenerate_by_pattern": by, "count": len(bases), "seconds": dt, "git": git_commit(),
               "generated": _now(), "covers": [list(c) for c in bases], "kinds": kinds}
        sha = common.write_hashed(CENSUS, rec)
        print(f"wrote {os.path.relpath(CENSUS, ROOT)} (sha256 {sha[:16]})")
    return 0


# ----------------------------------------------------------- symmetry ------

def group_closure(perms, N, order):
    """The permutation group of the dictionary generated by `perms`, by
    closure; its order must equal `order`."""
    gens = [np.asarray(p, dtype=np.int32) for p in perms]
    ident = np.arange(N, dtype=np.int32)
    seen = {ident.tobytes(): ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                k = h[g]
                key = k.tobytes()
                if key not in seen:
                    seen[key] = k
                    nxt.append(k)
        frontier = nxt
    if len(seen) != order:
        raise AssertionError(f"group closure has {len(seen)} elements, expected {order}")
    return list(seen.values())


def canonical(idx, group):
    return min(tuple(sorted(int(x) for x in g[list(idx)])) for g in group)


def control_census(args):
    """The census against the numeric enumerator: slice_lift.all_decompositions
    ("qubit_T", 3, 4) lists 5,205 rank-4 index tuples of |T>^3 (at least one
    per orbit, non-full covers included); the full ones among them, canonicalized
    under the unitary symmetry group of psi_3, must equal the census's kind A
    covers canonicalized, and the numeric rank-3 tuples must equal the 3-covers
    and the stored rank-3 list (qubit_T_m3_rank3.json) up to the group."""
    from slice_lift import all_decompositions
    E = make_enumerator()
    t0 = time.time()
    G = group_closure(E.info["perms"], E.N, E.info["order"])
    covers3, covers4, deg, bases, kinds = census_lists(E)
    t_census = time.time() - t0
    t1 = time.time()
    num4, _ = all_decompositions(ORBIT, 3, 4, D=E.D, verbose=False)
    num3, _ = all_decompositions(ORBIT, 3, 3, D=E.D, verbose=False)
    t_num = time.time() - t1
    num4 = [tuple(int(x) for x in c) for c in num4]
    num3 = [tuple(int(x) for x in c) for c in num3]
    full4 = [c for c in num4 if len(set(c)) == 4 and E.is_cover(c) and E.is_full(c)]
    canon_num4 = {canonical(c, G) for c in full4}
    canon_cen4 = {canonical(c, G) for c in covers4}
    canon_num3 = {canonical(c, G) for c in num3 if E.is_cover(c) and E.is_full(c)}
    canon_cen3 = {canonical(c, G) for c in covers3}
    stored, _ = common.load_decompositions(ORBIT, 3, 3)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    canon_stored3 = {canonical([lookup[exact_codes(t)[0].tobytes()] for t in terms], G) for terms, _ in stored}
    # the census has exactly one representative per orbit
    one_per_orbit = len(canon_cen4) == len(covers4) and len(canon_cen3) == len(covers3)
    # every degenerate multiset is full (a cover with no dead unrepeated state) and none is missed by
    # a second route: the multisets T + x over the numeric 3-covers' orbits agree
    deg_num = degenerate_covers(E, 4, sorted({c for c in num3 if E.is_cover(c) and E.is_full(c)}))
    canon_deg_num = {canonical(c, G) for c in deg_num}
    canon_deg = {canonical(c, G) for c in deg}
    # one_per_orbit is informational: the pivot and partner reductions may leave duplicates
    ok = (canon_num4 == canon_cen4 and canon_num3 == canon_cen3 and canon_stored3 == canon_cen3
          and canon_deg_num == canon_deg)
    rec = {"git": git_commit(), "generated": _now(), "group_order": E.info["order"], "N": E.N,
           "numeric_rank4_tuples": len(num4), "numeric_full_4covers": len(full4),
           "numeric_non_full": len(num4) - len(full4), "numeric_rank3_tuples": len(num3),
           "census_independent": len(covers4), "census_degenerate": len(deg), "census_bases": len(bases),
           "census_kinds": {k: kinds.count(k) for k in ("A", "B", "C")},
           "orbits_numeric_full4": len(canon_num4), "orbits_census4": len(canon_cen4),
           "orbits_numeric3": len(canon_num3), "orbits_census3": len(canon_cen3), "orbits_stored3": len(canon_stored3),
           "one_per_orbit": one_per_orbit, "degenerate_from_numeric_3covers_equal": canon_deg_num == canon_deg,
           "census_seconds": t_census, "numeric_seconds": t_num, "pass": ok}
    print(f"numeric enumerator: {len(num4)} rank-4 tuples ({len(full4)} full covers of distinct states in "
          f"{len(canon_num4)} orbits, {len(num4) - len(full4)} non-full), {len(num3)} rank-3 tuples in "
          f"{len(canon_num3)} orbits [{t_num:.0f}s]; census: {len(covers4)} independent covers in {len(canon_cen4)} "
          f"orbits, {len(covers3)} 3-covers in {len(canon_cen3)} orbits, stored rank-3 list in "
          f"{len(canon_stored3)} orbits, {len(deg)} degenerate multisets [{t_census:.0f}s]; "
          f"{'PASS' if ok else 'FAIL'}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_census.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("control-census:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------- controls ------

def slice_terms(terms, S, x0, m, n1, lookup):
    """Dictionary indices of the base slices of `terms` along the qubits S
    (moved to the front) at x0, or None when a term vanishes there."""
    perm = list(S) + [q for q in range(m) if q not in S]
    out = []
    for v in terms:
        s = v.reshape([2] * m).transpose(perm).reshape(1 << n1, -1)[x0]
        if np.linalg.norm(s) < 1e-9:
            return None
        codes, _ = exact_codes(s)
        out.append(lookup[codes.tobytes()])
    return tuple(out)


def flat_type(v, S, m, n1):
    perm = list(S) + [q for q in range(m) if q not in S]
    sl = v.reshape([2] * m).transpose(perm).reshape(1 << n1, -1)
    return tuple(int(x) for x in range(1 << n1) if np.linalg.norm(sl[x]) > 1e-9)


def control_witness(args):
    """The rank-6 witness of |T>^5 (the product of the rank-3 decomposition
    of |T>^4 with |0> and |1>) sliced along every qubit pair: at every (pair,
    x0) where all six terms are nonzero the base is a 6-cover of psi_3, and
    the matcher (compiled kernel, or the reference with --reference) must
    return the witness itself among its rank-6 decompositions. A base that
    raises is recorded as aborted and fails the control."""
    m, n1, rank = M, N1, 6
    E = make_enumerator()
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    w = json.load(open(os.path.join(ROOT, args.path)))
    terms = [common.term_vector(t, 2, m) for t in w["witness"]["terms"]]
    psi = common.target(ORBIT, m)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    assert np.linalg.norm(np.column_stack(terms) @ c - psi) < 1e-9
    bases, types = {}, {}
    for S in itertools.combinations(range(m), n1):
        types[S] = [flat_type(v, S, m, n1) for v in terms]
        for x0 in range(1 << n1):
            b = slice_terms(terms, S, x0, m, n1, lookup)
            if b is not None:
                bases.setdefault((b, x0), []).append(S)
    items = sorted(bases.items())
    print(f"{len(items)} distinct all-visible (base, x0) pairs over the {len(types)} qubit pairs; flat types: "
          + "; ".join(f"{S}: " + ",".join({4: 'P', 2: 'L' + ''.join(str(p) for p in t), 1: 'pt'}[len(t)]
                                             for t in ts) for S, ts in types.items()))
    Mt = SliceMatcher(E, n1, verbose=args.verbose, max_cand=args.max_cand, native=not args.reference)
    report = {"witness": args.path, "git": git_commit(), "generated": _now(), "max_cand": args.max_cand,
              "matcher": "reference" if Mt.native is None else "native", "bases": []}
    sel = range(len(items)) if args.base is None else [args.base]
    passed = aborted = 0
    for k in sel:
        (cover, x0), Ss = items[k]
        distinct = sorted(set(cover))
        fam = Family.from_cover(E, distinct)
        kappa = None if fam is None else fam.kappa
        t0 = time.time()
        entry = {"base": k, "cover": list(cover), "x0": x0, "pairs": Ss, "distinct": len(distinct), "kappa": kappa}
        try:
            hits, st = Mt.run(cover, x0)
        except Exception as exc:                        # noqa: BLE001
            dt = time.time() - t0
            entry.update({"status": "aborted", "reason": f"{type(exc).__name__}: {str(exc)[:200]}", "seconds": dt})
            aborted += 1
            print(f"  base {k}: cover {cover} x0 {x0:02b} kappa {kappa}: ABORTED {entry['reason']} "
                  f"after {dt:.0f}s", flush=True)
        else:
            good = [h for h in hits if genuine(h, rank)]
            S = Ss[0]
            perm = list(S) + [q for q in range(m) if q not in S]
            wkey = codes_key([v.reshape([2] * m).transpose(perm).reshape(-1) for v in terms])
            same = any(codes_key(h["terms"]) == wkey for h in good)
            dt = time.time() - t0
            entry.update({"status": "recovered" if same else "not recovered", "rank6_decompositions": len(good),
                          "witness_recovered": same, "seconds": dt,
                          "stats": {kk: (v if not isinstance(v, np.generic) else v.item()) for kk, v in st.items()}})
            passed += same
            print(f"  base {k}: cover {cover} x0 {x0:02b} kappa {kappa} ({len(Ss)} pairs): {len(good)} genuine "
                  f"rank-6 decompositions, witness {'recovered' if same else 'NOT among them'}, {dt:.1f}s, "
                  f"coord {st['coord_solutions']}, joined {st['joined']}, blocks {st['blocks']}, "
                  f"native {st['native']}", flush=True)
        report["bases"].append(entry)
    report["passed"] = passed
    report["aborted"] = aborted
    report["total"] = len(report["bases"])
    os.makedirs(RESULTS, exist_ok=True)
    name = "control_witness" + ("_reference" if args.reference else "") + \
        ("" if args.base is None else f"_{args.base}") + ".json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(report, f, indent=1)
    print(f"control-witness ({report['matcher']}): {passed}/{len(report['bases'])} bases recover the witness, "
          f"{aborted} aborted; {'PASS' if passed == len(report['bases']) else 'FAIL'}")
    return 0 if passed == len(report["bases"]) else 1


# -- planted instances --

def to_field(F, v):
    """A vector with Gaussian-integer entries reduced to F_p."""
    re, im = np.round(v.real).astype(np.int64), np.round(v.imag).astype(np.int64)
    if not np.allclose(v, re + 1j * im, atol=1e-9):
        raise AssertionError("vector entries are not Gaussian integers")
    return (re + F.i * im) % F.p


class PlantedMatcher(SliceMatcher):
    """SliceMatcher run against a planted five-qubit target Psi (a sum of
    stabilizer states with Gaussian-integer coefficients) instead of |T>^5:
    the base family is solved against Psi's slice at x0 and every slice
    equation against Psi's slice there. confirm() still measures the
    residual against |T>^5, so hits are compared by their term codes."""

    def __init__(self, E, Psi, x0, n1=N1):
        rows = Psi.reshape(1 << n1, -1)
        E2 = copy.copy(E)
        E2.psi = rows[x0]
        E2.psi1, E2.psi2 = to_field(E.F1, rows[x0]), to_field(E.F2, rows[x0])
        super().__init__(E2, n1, native=False)
        self._rows = rows

    def rhs(self, x0, x):
        v = self._rows[x]
        return to_field(self.F1, v), to_field(self.F2, v), v


def random_term(rng, E, flat):
    """A random five-qubit stabilizer state (entries in {0, +-1, +-i}) whose
    flat along the first two qubits is `flat` (a tuple of points), built by
    the structure lemma from a random three-qubit base state and random
    Pauli classes, phases and quadratic sign."""
    u = E.C[:, rng.integers(E.N)]
    return _term_over(rng, u, flat)


def _term_over(rng, u, flat):
    reps, _ = pauli_reps(u, N2)
    t = np.zeros((4, 8), dtype=complex)
    pts = sorted(flat)
    p0 = pts[0]
    t[p0] = u
    if len(pts) == 4:
        k1, k2 = rng.integers(8), rng.integers(8)
        l1, l2, s = rng.integers(4), rng.integers(4), rng.integers(2)
        (a1, c1), (a2, c2) = reps[k1], reps[k2]
        t[p0 ^ 0b01] = (1j) ** l1 * apply_pauli(u, a1, c1, N2)
        t[p0 ^ 0b10] = (1j) ** l2 * apply_pauli(u, a2, c2, N2)
        t[p0 ^ 0b11] = (-1) ** s * (1j) ** (l1 + l2) * apply_pauli(apply_pauli(u, a1, c1, N2), a2, c2, N2)
    elif len(pts) == 2:
        k, l = rng.integers(8), rng.integers(4)
        a, c = reps[k]
        t[pts[1]] = (1j) ** l * apply_pauli(u, a, c, N2)
    return t.ravel()


def _gauss_coeffs(rng, n):
    return np.array([complex(rng.integers(1, 4)) * (1j) ** rng.integers(4) for _ in range(n)])


def control_planted(args):
    """Planted four-term instances at both base points, the target their sum
    with Gaussian-integer coefficients: (a) four planes; (b) three planes and
    a line on the diagonal through x0; (c) a repeated base state (two plane
    copies with one base slice) and two planes; (d) a repeated base state,
    a plane and a diagonal line; (e) three planes and a term on the
    coordinate line through x0 (excluded by Fact 1, but the matcher must
    find it). Every planted decomposition must be among the hits."""
    E = make_enumerator()
    rng = np.random.default_rng(args.seed)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    PLANE = (0, 1, 2, 3)
    kinds = ["a", "b", "c", "d", "e"]
    report = {"seed": args.seed, "git": git_commit(), "generated": _now(), "plants": []}
    n_ok = total = 0
    for k in range(args.plant):
        for x0 in X0S:
            kind = kinds[k % len(kinds)]
            diag = (x0, x0 ^ 0b11)
            while True:
                if kind == "a":
                    terms = [random_term(rng, E, PLANE) for _ in range(4)]
                elif kind == "b":
                    terms = [random_term(rng, E, PLANE) for _ in range(3)] + [random_term(rng, E, diag)]
                elif kind == "c":
                    v = random_term(rng, E, PLANE)
                    terms = [v, _term_over(rng, v.reshape(4, 8)[x0], PLANE if x0 == 0 else (x0, x0 ^ 1, x0 ^ 2, x0 ^ 3)),
                             random_term(rng, E, PLANE), random_term(rng, E, PLANE)]
                elif kind == "d":
                    v = random_term(rng, E, PLANE)
                    terms = [v, _term_over(rng, v.reshape(4, 8)[x0], PLANE if x0 == 0 else (x0, x0 ^ 1, x0 ^ 2, x0 ^ 3)),
                             random_term(rng, E, PLANE), random_term(rng, E, diag)]
                else:
                    terms = [random_term(rng, E, PLANE) for _ in range(3)] + [random_term(rng, E, (x0, x0 ^ 0b01))]
                A = np.column_stack(terms)
                if np.linalg.matrix_rank(A, tol=1e-8) < 4:
                    continue
                base = tuple(sorted(lookup[exact_codes(t.reshape(4, 8)[x0])[0].tobytes()] for t in terms))
                if kind in ("c", "d"):
                    if len(set(base)) != 3:
                        continue
                    if codes_key([terms[0]]) == codes_key([terms[1]]):
                        continue
                elif len(set(base)) < 4:
                    continue
                coeffs = _gauss_coeffs(rng, 4)
                if kind in ("c", "d") and abs(coeffs[0] + coeffs[1]) < 1e-9:
                    continue
                Psi = A @ coeffs
                Bm = PlantedMatcher(E, Psi, x0)
                fam = Family.from_cover(Bm.E, sorted(set(base)))
                exempt = [i for i, u in enumerate(sorted(set(base))) if base.count(u) > 1]
                if fam is None or fam.has_zero_coefficient(exempt):
                    continue
                break
            t0 = time.time()
            total += 1
            entry = {"kind": kind, "x0": x0, "base": list(base)}
            try:
                hits, st = Bm.run(base, x0)
            except UnpinnedFamily as exc:
                entry.update({"status": "undecided", "reason": str(exc)[:200], "seconds": time.time() - t0})
                print(f"  plant {k} kind {kind} x0 {x0:02b} base {base}: UNDECIDED {str(exc)[:120]}")
            else:
                want = codes_key(terms)
                same = any(codes_key(h["terms"]) == want for h in hits)
                n_ok += same
                entry.update({"status": "recovered" if same else "not recovered", "hits": len(hits),
                              "seconds": time.time() - t0,
                              "stats": {kk: (v if not isinstance(v, np.generic) else v.item()) for kk, v in st.items()}})
                print(f"  plant {k} kind {kind} x0 {x0:02b} base {base}: {len(hits)} hits, planted "
                      f"{'recovered' if same else 'NOT recovered'}, {time.time() - t0:.2f}s, kappa {st['kappa']} "
                      f"blocks {st['blocks']} coord {st['coord_solutions']} recon {st['reconstructions']}",
                      flush=True)
            report["plants"].append(entry)
    report["recovered"] = n_ok
    report["total"] = total
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_planted.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"control-planted: {n_ok}/{total} planted decompositions recovered; {'PASS' if n_ok == total else 'FAIL'}")
    return 0 if n_ok == total else 1


# -- the m = 4 control along a qubit pair --

def control_m4_pair(args):
    """Slice the stored rank-3 decomposition of |T>^4 along a qubit pair:
    every all-visible base is a full 3-cover of |T>^2 over the 60 two-qubit
    states, and the matcher at n_1 = 2 over every such base (independent,
    dependent and repeated, from the enumerator) must recover the stored
    class and nothing outside the stored list."""
    m, n1, rank = 4, 2, 3
    E = CoverEnumerator(2, orbit=ORBIT)
    lookup2 = {E.codes[i].tobytes(): i for i in range(E.N)}
    t0 = time.time()
    covers2, _ = E.covers(2)
    covers3, _ = E.covers(3)
    degenerate = degenerate_covers(E, 3, covers2)
    bases = sorted(set(covers3) | set(degenerate))
    print(f"{len(covers2)} full 2-covers and {len(covers3)} independent full 3-covers of |T>^2, "
          f"{len(degenerate)} dependent or repeated, {len(bases)} bases [{time.time() - t0:.0f}s]")
    D4 = dictionary(2, 4)
    codes4, _ = patterns(D4)
    lookup4 = {codes4[i].tobytes(): i for i in range(D4.shape[1])}
    _, info4 = symmetry_orbit_reps(ORBIT, 4, D4, antiunitary=False)
    group = group_closure(info4["perms"], D4.shape[1], info4["order"])
    stored, _ = common.load_decompositions(ORBIT, m, rank)
    stored_keys, expected = {}, set()
    x0s = list(range(1 << n1)) if args.all_x0 else list(X0S)
    for terms, _ in stored:
        idx = [lookup4[exact_codes(t)[0].tobytes()] for t in terms]
        key = canonical(idx, group)
        stored_keys[key] = terms
        for g in group:
            if key in expected:
                break
            gterms = [D4[:, g[i]] for i in idx]
            for S in itertools.combinations(range(m), n1):
                for x0 in x0s:
                    b = slice_terms(gterms, S, x0, m, n1, lookup2)
                    if b is None:
                        continue
                    distinct = sorted(set(b))
                    fam = Family.from_cover(E, distinct)
                    exempt = [i for i, u in enumerate(distinct) if b.count(u) > 1]
                    if fam is not None and not fam.has_zero_coefficient(exempt):
                        expected.add(key)
    Mt = SliceMatcher(E, n1, verbose=args.verbose, native=not args.reference)
    recovered = {}
    t1 = time.time()
    stats = {"matched": 0, "refused": 0, "hits": 0, "non_genuine": 0, "undecided": 0, "native_runs": 0}
    for cover in bases:
        for x0 in x0s:
            try:
                hits, st = Mt.run(cover, x0)
            except UnpinnedFamily as exc:
                stats["undecided"] += 1
                print(f"  {cover} x0 {x0:02b}: UnpinnedFamily: {str(exc)[:120]}")
                continue
            if st["refused"]:
                stats["refused"] += 1
                continue
            stats["matched"] += 1
            stats["native_runs"] += bool(st.get("native"))
            for h in hits:
                if not genuine(h, rank):
                    stats["non_genuine"] += 1
                    continue
                stats["hits"] += 1
                key = canonical([lookup4[exact_codes(t)[0].tobytes()] for t in h["terms"]], group)
                recovered.setdefault(key, []).append((list(cover), x0))
    dt = time.time() - t1
    unknown = set(recovered) - set(stored_keys)
    missing = expected - set(recovered)
    extra = set(recovered) - expected
    print(f"matched {stats['matched']} (cover, x0) pairs in {dt:.0f}s; {stats['hits']} hits, "
          f"{len(recovered)} classes of rank-{rank} decompositions of |T>^{m}; stored {len(stored_keys)}, "
          f"expected recoverable {len(expected)}; not stored {len(unknown)}, missing {len(missing)}, "
          f"unexpected {len(extra)}; stats {stats}")
    os.makedirs(RESULTS, exist_ok=True)
    ok = not unknown and not missing and not extra and not stats["undecided"] and len(recovered) == 1
    name = "control_m4_pair" + ("_reference" if args.reference else "") + ".json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump({"git": git_commit(), "generated": _now(), "x0": x0s, "bases": len(bases),
                   "independent": len(covers3), "degenerate": len(degenerate), "group_order": info4["order"],
                   "stored": len(stored_keys), "expected": len(expected), "recovered": len(recovered),
                   "unknown": len(unknown), "missing": len(missing), "extra": len(extra), "seconds": dt,
                   "stats": stats, "matcher": "reference" if Mt.native is None else "native", "pass": ok,
                   "recovered_from": {str(k): v for k, v in recovered.items()}}, f, indent=1)
    print("control-m4-pair:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# -- the tables --

def tables(args):
    script = os.path.join(HERE, "tables.py")
    cmd = [sys.executable, script, "--J", str(args.J)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for ln in proc.stdout.strip().splitlines():
        print("   ", ln)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
    print(f"tables: exit {proc.returncode} in {time.time() - t0:.0f}s; {'PASS' if proc.returncode == 0 else 'FAIL'}")
    return proc.returncode


# ------------------------------------------------------------- rates -------

def _spread(lst, n):
    n = min(n, len(lst))
    return [lst[int(round(k * (len(lst) - 1) / max(n - 1, 1)))] for k in range(n)]


def _load_rates():
    if os.path.exists(RATES):
        with open(RATES) as f:
            return json.load(f)
    return {"git": git_commit(), "stages": {}}


def sample(args):
    """Time the matcher on covers of every kind spread through the census,
    both base points, one (cover, x0) run at a time under a per-run cap."""
    import signal
    E = make_enumerator()
    covers, doc = common.load_census()
    Mt = SliceMatcher(E, N1, native=not args.reference)
    by_kind = {k: [c for c, kk in zip(covers, doc["kinds"]) if kk == k] for k in ("A", "B", "C")}
    rates = _load_rates()

    class Deadline(Exception):
        pass

    def alarm(signum, frame):
        raise Deadline()

    signal.signal(signal.SIGALRM, alarm)
    for kind in ("A", "B", "C"):
        lst = by_kind[kind]
        if not lst:
            continue
        sel = _spread(lst, args.count) if len(lst) > args.count else lst
        rec = {"stage": kind, "count": 0, "runs": 0, "seconds": [], "by_pattern": {}, "hits": 0, "refused": 0,
               "undecided": 0, "capped": 0, "hist": {}, "generated": _now(), "git": git_commit(),
               "matcher": "reference" if Mt.native is None else "native"}
        t_all = time.time()
        for cover in sel:
            pat = str(common.multiplicity_pattern(cover))
            for x0 in X0S:
                t1 = time.time()
                signal.alarm(args.cap)
                try:
                    hits, st = Mt.run(cover, x0)
                    if st["refused"]:
                        rec["refused"] += 1
                    else:
                        rec["hits"] += sum(genuine(h, RANK) for h in hits)
                        key = ",".join(str(b) for b in st["coord_solutions"])
                        rec["hist"][key] = rec["hist"].get(key, 0) + 1
                except Deadline:
                    rec["capped"] += 1
                except UnpinnedFamily as exc:
                    rec["undecided"] += 1
                    print(f"  {cover} x0 {x0:02b}: UnpinnedFamily: {exc}")
                finally:
                    signal.alarm(0)
                dt = time.time() - t1
                rec["seconds"].append(dt)
                rec["by_pattern"].setdefault(pat, []).append(dt)
                rec["runs"] += 1
            rec["count"] += 1
        a = np.array(rec["seconds"])
        rec["per_run_mean_s"] = float(a.mean())
        rec["per_run_median_s"] = float(np.median(a))
        rec["per_run_max_s"] = float(a.max())
        rec["per_pattern_mean_s"] = {p: float(np.mean(v)) for p, v in rec["by_pattern"].items()}
        rec["wall_s"] = time.time() - t_all
        print(f"kind {kind} ({rec['matcher']}): {rec['count']} of {len(lst)} covers x {len(X0S)} base points: "
              f"per (cover, x0) mean {a.mean():.4f}s, median {np.median(a):.4f}s, max {a.max():.3f}s; hits "
              f"{rec['hits']}, refused {rec['refused']}, undecided {rec['undecided']}, capped {rec['capped']}; "
              f"histogram {rec['hist']}")
        rates["stages"][kind] = {k: v for k, v in rec.items() if k not in ("seconds", "by_pattern")}
    rates["git"] = git_commit()
    os.makedirs(RESULTS, exist_ok=True)
    with open(RATES, "w") as f:
        json.dump(rates, f, indent=1)
    return 0


# ---------------------------------------------------------- partition ------

def partition(args):
    E = make_enumerator()
    covers, doc = common.load_census()
    rates = _load_rates()["stages"]
    for k in ("A", "B", "C"):
        if k not in rates and any(kk == k for kk in doc["kinds"]):
            raise SystemExit(f"no rate for kind {k} in {RATES}; run `driver.py sample` first")
    F = args.pod_factor
    n_x0 = len(X0S)
    geometry, counts, est = [], {}, {}
    for k in ("A", "B", "C"):
        ids = [i for i, kk in enumerate(doc["kinds"]) if kk == k]
        if not ids:
            counts[k], est[k] = 0, 0.0
            continue
        rate = rates[k]["per_run_mean_s"]
        cost = F * rate * n_x0 * len(ids)
        n = max(1, math.ceil(cost / args.target_s), math.ceil(len(ids) / args.max_covers))
        for b in range(n):
            geometry.append({"index": len(geometry), "stage": k, "cover_ids": ids[b::n]})
        counts[k], est[k] = n, cost
    est["total"] = sum(v for kk, v in est.items() if kk != "total")
    rec = {"orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "x0": list(X0S), "stage_a_batches": counts["A"], "stage_b_batches": counts["B"],
           "stage_c_batches": counts["C"],
           "cost_model": {"pod_factor": F, "target_s": args.target_s, "rates": os.path.relpath(RATES, HERE),
                          "s_per_run": {k: rates[k]["per_run_mean_s"] for k in rates}, "runs_per_cover": n_x0},
           "estimated_s": est,
           "census": {"file": os.path.relpath(CENSUS, HERE), "sha256": doc["sha256"], "count": len(covers),
                      "independent": doc["independent"], "dependent": doc["dependent"], "repeated": doc["repeated"]},
           "batches": len(geometry), "git": git_commit(), "generated": _now(), "batch_geometry": geometry}
    sha = common.write_hashed(PARTITION, rec)
    print(f"{len(covers)} bases: {counts['A']} kind A batches ({est['A'] / 60:.1f} pod min), {counts['B']} kind B "
          f"({est['B'] / 60:.1f}), {counts['C']} kind C ({est['C'] / 60:.1f}); {len(geometry)} batches, "
          f"{est['total'] / 60:.1f} pod CPU-min at {F} x the laptop rates; wrote {PARTITION} (sha256 {sha[:16]})")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--J", type=int, default=2)
    p.set_defaults(fn=tables)
    p = sub.add_parser("census")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=census)
    p = sub.add_parser("control-census")
    p.set_defaults(fn=control_census)
    p = sub.add_parser("control-witness")
    p.add_argument("--path", default="bounds/qubit_T-m5-upper-6.json")
    p.add_argument("--base", type=int, default=None, help="run only the k-th base (0-based)")
    p.add_argument("--max-cand", type=int, default=2_000_000, help="candidate cap of the dense solve")
    p.add_argument("--reference", action="store_true", help="the Python matcher instead of the compiled kernel")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_witness)
    p = sub.add_parser("control-planted")
    p.add_argument("--plant", type=int, default=10, help="planted instances per base point")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(fn=control_planted)
    p = sub.add_parser("control-m4-pair")
    p.add_argument("--all-x0", action="store_true", help="all four base points instead of 00 and 01")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_m4_pair)
    p = sub.add_parser("sample")
    p.add_argument("--count", type=int, default=100, help="covers per kind")
    p.add_argument("--cap", type=int, default=120, help="seconds per (cover, x0) run")
    p.add_argument("--reference", action="store_true")
    p.set_defaults(fn=sample)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=120.0, help="pod seconds per batch")
    p.add_argument("--pod-factor", type=float, default=2.0, help="pod seconds per laptop second")
    p.add_argument("--max-covers", type=int, default=1000, help="at most this many covers per batch")
    p.set_defaults(fn=partition)
    args = ap.parse_args(argv[1:])
    common.lower_priority()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
