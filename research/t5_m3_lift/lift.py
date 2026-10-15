"""Slice-and-lift at exact rank: the lower bound chi(|T5>^3) >= 6 from the
settled cell chi(|T5>^2) = 5 and the census listing of its rank-5
decompositions (docs/notes/t5_m3_lift_design.md).

    lift.py cell [--low-census | --trust-low-census] [--results-dir D]
    lift.py control-basis | control-rank3 | control-product | control-t3-basis
            | control-t3-m2 | control-n-m2 | control-stabtest
    lift.py control-t3-full          (168 M patterns before pruning; not in the chain)
    lift.py plan

The lemma (section 1 of the note). Let p be an odd prime, phi in C^p with
every amplitude alpha_z nonzero, psi_m = phi^(x m), r = chi(psi_m), and let
E be a family of r-sets of m-qudit stabilizer states (distinct up to
phase) closed under the unitary symmetry group of psi_m and containing
every r-set whose span contains psi_m. For a set E in E write a^E(e) for
the coefficient of e in the unique expansion of psi_m over E. A pattern is
a choice, for every z in F_p, of a set E_z in E and a member e_z of E_z;
its vector is

    P = sum_z alpha_z a^(E_z)(e_z) e_z (x) |z>.

Then chi(psi_(m+1)) = r if and only if there are sets E_0, ..., E_(p-1)
in E and r patterns with those sets whose members are a bijection onto
E_z at every z and whose vectors are all (scalar multiples of) stabilizer
states; otherwise chi(psi_(m+1)) >= r + 1. There is no invisible term and
no coincident term: a slice of a rank-r decomposition of psi_(m+1) at z is
a decomposition of psi_m with at most r terms, and chi(psi_m) = r forces
every slice of every term to be nonzero and the r slices to be pairwise
non-parallel (section 1.2 of the note).

The decision is exact over Q(zeta_n) (`exact.py`): the coefficients
a^E(e) are solved by Gaussian elimination in the field, the pattern vector
is built in the field, and the stabilizer test checks the support flat,
the moduli, the root-of-unity ratios, and the quadratic exponents with
rational arithmetic. Patterns are enumerated by a depth-first search over
z with three exact prunes that every stabilizer pattern satisfies (equal
moduli, equal direction space of the slices, translates affine in z);
the pruned patterns are counted and never tested.

Results go to research/t5_m3_lift/results/<command>.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from exact import (Cyclo, flat_of, reduce_mod_direction, rref_mod_p,  # noqa: E402
                   solve_in_span, stabilizer_form)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CENSUS = os.path.join(ROOT, "research", "t5q_m2_rank5")
CLAIM_M3 = "CERTIFIED chi(T5^3) >= 6"
CLAIM_M4 = "CERTIFIED chi(T5^4) >= 6"
HIT = (525, 563, 591, 619, 637)          # the census hit, research/t5q_m2_rank5/results/batch_31.json


# ------------------------------------------------------------- helpers ----

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lower_priority():
    try:
        os.nice(19)
    except OSError:
        pass


def digits(index, p, m):
    index = int(index)
    out = []
    for _ in range(m):
        out.append(index % p)
        index //= p
    return tuple(reversed(out))


def phase_codes(D, p, tol=1e-6):
    """Phase codes (0 zero, 1..p = w_p^0..w_p^(p-1)) of the columns of a
    complex dictionary D with the first nonzero entry made 1; every nonzero
    entry must be a p-th root of unity times the column scalar."""
    dim, N = D.shape
    w = np.exp(2j * np.pi / p)
    nz = np.abs(D) > 1e-9
    first = np.argmax(nz, axis=0)
    ref = D[first, np.arange(N)]
    W = D / ref[None, :]
    codes = np.zeros((N, dim), dtype=np.int16)
    for c in range(p):
        codes[np.abs(W.T - w ** c) < tol] = c + 1
    if not np.array_equal(codes > 0, nz.T):
        raise AssertionError("a dictionary entry is not a p-th root of unity times the column scalar")
    return codes


def codes_to_field(K, codes, step):
    """A code row to a list of field elements (zeta_n^(step (code - 1)), 0 for 0)."""
    return [K.zero if c == 0 else K.root(step * (int(c) - 1)) for c in codes]


# -------------------------------------------------------- the lift search ---

class LiftProblem:
    """psi_m = sum_e a^E(e) e for every set E of the family, the one-qudit
    factor alpha, and the pattern search for lifts to m + 1 qudits.

    p: odd prime local dimension; m: qudits of the sliced level; K:
    Q(zeta_n) with p | n; codes: (N, p^m) phase codes of the dictionary;
    target: psi_m as a list of p^m field elements; alpha: p field elements,
    all nonzero; sets: the family E as sorted tuples of dictionary indices.
    """

    def __init__(self, p, m, K, codes, target, alpha, sets):
        self.p, self.m, self.K = p, m, K
        if K.n % p:
            raise ValueError("p must divide the field order n")
        self.step = K.n // p
        self.codes = codes
        self.dim = p ** m
        self.target = target
        self.alpha = list(alpha)
        if any(K.is_zero(a) for a in self.alpha):
            raise ValueError("every amplitude of the one-qudit factor must be nonzero (the lemma needs it)")
        self.alpha_n2 = [K.norm2(a) for a in self.alpha]
        self.sets = [tuple(sorted(int(x) for x in s)) for s in sets]
        self.r = len(self.sets[0]) if self.sets else 0
        if any(len(s) != self.r for s in self.sets):
            raise ValueError("every set of the family must have the same size r")
        self.vec = {}
        self.geom = {}
        for s in self.sets:
            for e in s:
                if e not in self.vec:
                    self.vec[e] = codes_to_field(K, codes[e], self.step)
                    pts = [digits(i, p, m) for i in np.flatnonzero(codes[e])]
                    flat = flat_of(pts, p)
                    if flat is None:
                        raise AssertionError(f"dictionary state {e} is not supported on an affine flat")
                    x0, W, piv = flat
                    self.geom[e] = {"points": pts, "V": tuple(W), "rep": reduce_mod_direction(x0, W, piv, p),
                                    "size": len(pts)}
        self.coeffs = {}
        for si, s in enumerate(self.sets):
            a = solve_in_span(K, [self.vec[e] for e in s], target)
            if a is None:
                raise AssertionError(f"set {s} does not span the target")
            if any(K.is_zero(x) for x in a):
                raise AssertionError(f"set {s} has a zero coefficient; the family must hold minimal sets")
            self.coeffs[si] = {e: x for e, x in zip(s, a)}

    # -- patterns --
    def options(self, z):
        """(set index, member, modulus of alpha_z a^E(e)) for every set and member."""
        out = []
        for si, s in enumerate(self.sets):
            for e in s:
                n2 = self.K.mul(self.alpha_n2[z], self.K.norm2(self.coeffs[si][e]))
                out.append((si, e, n2))
        return out

    def pattern_vector(self, choice):
        """The entries of P for a pattern [(si, e)] * p, as a map point -> element."""
        K = self.K
        entries = {}
        for z, (si, e) in enumerate(choice):
            scal = K.mul(self.alpha[z], self.coeffs[si][e])
            for i in np.flatnonzero(self.codes[e]):
                x = digits(int(i), self.p, self.m)
                entries[x + (z,)] = K.mul(scal, K.root(self.step * (int(self.codes[e][i]) - 1)))
        return entries

    def search(self, verbose=False):
        """Every pattern whose vector is a stabilizer state, with counts of
        the patterns pruned and tested."""
        p = self.p
        opts = [self.options(z) for z in range(p)]
        stats = {"patterns_total": int(np.prod([len(o) for o in opts])), "leaf_tests": 0,
                 "pruned_modulus": 0, "pruned_direction": 0, "pruned_translate": 0, "stabilizer": 0}
        found = []
        t0 = time.time()

        def rec(z, choice, V, rep0, d, mod):
            if z == p:
                stats["leaf_tests"] += 1
                entries = self.pattern_vector(choice)
                form = stabilizer_form(self.K, p, self.m + 1, entries)
                if form is not None:
                    stats["stabilizer"] += 1
                    found.append({"choice": [(si, e) for si, e in choice], "term": form})
                return
            for si, e, n2 in opts[z]:
                g = self.geom[e]
                if z == 0:
                    rec(1, choice + [(si, e)], g["V"], g["rep"], None, n2)
                    continue
                if n2 != mod:
                    stats["pruned_modulus"] += 1
                    continue
                if g["V"] != V:
                    stats["pruned_direction"] += 1
                    continue
                diff = tuple((a - b) % p for a, b in zip(g["rep"], rep0))
                if z == 1:
                    rec(2, choice + [(si, e)], V, rep0, diff, mod)
                    continue
                if diff != tuple((z * x) % p for x in d):
                    stats["pruned_translate"] += 1
                    continue
                rec(z + 1, choice + [(si, e)], V, rep0, d, mod)

        rec(0, [], None, None, None, None)
        stats["seconds"] = time.time() - t0
        if verbose:
            print(f"  patterns {stats['patterns_total']}, leaf tests {stats['leaf_tests']}, stabilizer "
                  f"{stats['stabilizer']} ({stats['seconds']:.1f}s)", flush=True)
        return found, stats

    def lifts(self, found):
        """r-sets of stabilizer patterns with the same sets at every z whose
        members are a bijection onto E_z at every z: the rank-r
        decompositions of psi_(m+1)."""
        groups = {}
        for f in found:
            key = tuple(si for si, _ in f["choice"])
            groups.setdefault(key, []).append(f)
        out = []
        for key, pats in groups.items():
            n = len(pats)
            if n < self.r:
                continue

            def rec(start, chosen, used):
                if len(chosen) == self.r:
                    out.append([pats[i] for i in chosen])
                    return
                for i in range(start, n):
                    members = [e for _, e in pats[i]["choice"]]
                    if any(e in used[z] for z, e in enumerate(members)):
                        continue
                    rec(i + 1, chosen + [i], [u | {e} for u, e in zip(used, members)])

            rec(0, [], [set() for _ in range(self.p)])
        return out

    def check_lift(self, lift):
        """The lifted terms sum to psi_m (x) alpha exactly."""
        K = self.K
        total = {}
        for f in lift:
            for x, v in self.pattern_vector(f["choice"]).items():
                total[x] = K.add(total.get(x, K.zero), v)
        for i in range(self.dim):
            x = digits(i, self.p, self.m)
            for z in range(self.p):
                want = K.mul(self.target[i], self.alpha[z])
                if total.get(x + (z,), K.zero) != want:
                    return False
        return True


def orbit_closure(sets, perms):
    seen = {tuple(sorted(int(x) for x in s)) for s in sets}
    frontier = list(seen)
    while frontier:
        nxt = []
        for s in frontier:
            for pm in perms:
                t = tuple(sorted(int(pm[x]) for x in s))
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return sorted(seen)


def run_problem(prob, label, expect_lifts=None, verbose=True):
    """Search, lift, check; a record for the result file."""
    found, stats = prob.search(verbose=verbose)
    lifts = prob.lifts(found)
    for L in lifts:
        if not prob.check_lift(L):
            raise AssertionError("a lift does not reproduce the target")
    rec = {"label": label, "p": prob.p, "m": prob.m, "field": prob.K.n, "r": prob.r, "sets": len(prob.sets),
           "stats": stats, "stabilizer_patterns": found, "lifts": [[f["term"] for f in L] for L in lifts],
           "lift_count": len(lifts)}
    if verbose:
        print(f"{label}: {len(prob.sets)} set(s) of size {prob.r}, {stats['patterns_total']} patterns, "
              f"{stats['leaf_tests']} tested, {stats['stabilizer']} stabilizer, {len(lifts)} lift(s)", flush=True)
    if expect_lifts is not None:
        ok = (len(lifts) > 0) if expect_lifts == "some" else (len(lifts) == expect_lifts)
        rec["expected"] = expect_lifts
        rec["passed"] = bool(ok)
    return rec


def write_result(name, rec):
    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(rec)
    rec.update({"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "git_commit": git_commit(), "hostname": socket.gethostname(),
                "python": platform.python_version(), "numpy": np.__version__})
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return path


# ------------------------------------------------------------- the cell ----

def t5_alpha(K):
    """|T5> up to 1/sqrt 5: w^(z^3)."""
    return [K.root(K.n // 5 * (z ** 3 % 5)) for z in range(5)]


def t5_target(K, m):
    return [K.root(K.n // 5 * (sum(x ** 3 for x in pt) % 5)) for pt in itertools.product(range(5), repeat=m)]


def census_listing(results_dir=None, verbose=True):
    """The complete listing of the rank-5 census of |T5>^2 from its stored
    batch records: every record present, consistent with the partition
    (aggregate.check_batch), every unit run, no undecided unit; returns the
    enumerator, the decomposition 5-sets, and the record hashes."""
    sys.path.insert(0, CENSUS)
    import common as census_common  # noqa: E402
    import aggregate as census_aggregate  # noqa: E402
    part = census_common.load_partition()
    E = census_common.make_enumerator(native=False)
    problems = []
    if E.dictionary_sha256() != part["dictionary_sha256"]:
        problems.append("dictionary hash differs from the partition's")
    units = [list(u) for u in E.units()]
    if units != part["units"]:
        problems.append("the partition's units are not the enumerator's")
    rdir = results_dir or census_common.RESULTS
    records, hashes, hits = {}, {}, []
    for geo in part["batch_geometry"]:
        idx = geo["index"]
        path = os.path.join(rdir, f"batch_{idx}.json")
        if not os.path.exists(path):
            problems.append(f"batch {idx} missing")
            continue
        with open(path) as f:
            rec = json.load(f)
        records[idx] = rec
        census_aggregate.check_batch(rec, part, geo, problems)
        hashes[idx] = rec.get("deterministic_sha256")
        for h in rec.get("hits", []):
            d = E.decide(h["states"])
            if d["decomposition"] != h.get("decomposition"):
                problems.append(f"batch {idx}: a hit's stored decision differs from the re-decision")
            if d["decomposition"]:
                hits.append(tuple(sorted(int(x) for x in h["states"])))
    if verbose:
        print(f"census records: {len(records)}/{part['batches']} batches, "
              f"{sum(r.get('units_run', 0) for r in records.values())}/{part['units_count']} units run, "
              f"{sum(r.get('hit_count', 0) for r in records.values())} hit(s), "
              f"{sum(len(r.get('undecided', [])) for r in records.values())} undecided, "
              f"{len(problems)} problem(s)", flush=True)
        for pr in problems[:20]:
            print(f"PROBLEM: {pr}")
    return E, part, sorted(set(hits)), hashes, problems


def cmd_cell(a):
    t0 = time.time()
    E, part, hits, hashes, problems = census_listing(a.results_dir)
    if problems:
        print(f"NOT CERTIFIED: {len(problems)} problem(s) in the census records")
        return 1
    low = None
    if a.low_census:
        t1 = time.time()
        low = {}
        for k in (1, 2, 3, 4):
            r = E.census_low(k)
            low[k] = {"hits": len(r["hits"]), "candidates": r["candidates"], "units": r["units"]}
            if r["hits"]:
                print(f"NOT CERTIFIED: the k = {k} census lists a set with psi_2 in its span")
                return 1
        print(f"low-rank censuses k = 1..4 empty ({time.time() - t1:.0f}s): chi(T5^2) >= 5 exactly", flush=True)
    elif a.trust_low_census:
        ctrl = os.path.join(CENSUS, "results", "control_rank4.json")
        if not os.path.exists(ctrl):
            print("NOT CERTIFIED: --trust-low-census given but control_rank4.json is absent")
            return 1
        print("low-rank censuses trusted from research/t5q_m2_rank5/results/control_rank4.json "
              "(the aggregate re-runs them)", flush=True)
    else:
        print("NOT CERTIFIED: say --low-census (150 s) or --trust-low-census; chi(T5^2) = 5 exactly is an input")
        return 1
    K = Cyclo(5)
    codes = E.codes
    family = orbit_closure(hits, E.info["perms"])
    print(f"census listing: {len(hits)} decomposition 5-set(s) {hits}; orbit closure under the unitary "
          f"symmetry group (order {E.info['order']}): {len(family)} set(s)", flush=True)
    # exact re-decision of every set in the family over Q(zeta_5)
    target = t5_target(K, 2)
    for s in family:
        a_e = solve_in_span(K, [codes_to_field(K, codes[e], 1) for e in s], target)
        if a_e is None or any(K.is_zero(x) for x in a_e):
            print(f"NOT CERTIFIED: set {s} is not an exact full decomposition over Q(zeta_5)")
            return 1
    prob = LiftProblem(5, 2, K, codes, target, t5_alpha(K), family)
    rec = run_problem(prob, "T5 m=2 -> 3", expect_lifts=0)
    rec.update({"census": {"partition_sha256": part["sha256"], "dictionary_sha256": part["dictionary_sha256"],
                           "plan_sha256": part["plan_sha256"], "batches": len(hashes),
                           "deterministic_sha256": hashes, "hits": [list(h) for h in hits],
                           "family": [list(s) for s in family], "group_order": E.info["order"]},
                "low_census": low, "seconds": time.time() - t0})
    write_result("cell", rec)
    if rec["lift_count"]:
        print(f"DECOMPOSITION FOUND: {rec['lift_count']} rank-5 lift(s) of |T5>^2 to |T5>^3")
        return 2
    print("no rank-5 decomposition of |T5>^2 lifts to |T5>^3: every one of the "
          f"{rec['stats']['leaf_tests']} patterns that survive the exact prunes fails the stabilizer test")
    print(CLAIM_M3)
    return 0


# ------------------------------------------------------------- controls ----

def point_states(codes):
    return [int(i) for i in range(codes.shape[0]) if int(np.count_nonzero(codes[i])) == 1]


def all_minimal_sets(K, codes, target, r, step):
    """Every r-set of dictionary states whose span contains the target with
    all coefficients nonzero and the states independent, by brute force."""
    N = codes.shape[0]
    vecs = [codes_to_field(K, codes[i], step) for i in range(N)]
    out = []
    for s in itertools.combinations(range(N), r):
        try:
            a = solve_in_span(K, [vecs[i] for i in s], target)
        except ValueError:
            continue
        if a is not None and all(not K.is_zero(x) for x in a):
            out.append(tuple(s))
    return out


def cmd_control_basis(a):
    """p = 5, m = 1 -> 2 through the computational basis (a 5-set, not
    minimal: chi(T5) = 3). The machinery control: the lift must be exactly
    the Z(x)Z sector decomposition of |T5>^2, with non-constant patterns."""
    K = Cyclo(5)
    D = dictionary(5, 1)
    codes = phase_codes(D, 5)
    basis = point_states(codes)
    prob = LiftProblem(5, 1, K, codes, t5_target(K, 1), t5_alpha(K), [tuple(basis)])
    rec = run_problem(prob, "T5 basis m=1 -> 2", expect_lifts=1)
    with open(os.path.join(ROOT, "bounds", "T5-m2-upper-5.json")) as f:
        want = json.load(f)["witness"]["terms"]
    ok = rec["passed"] and rec["lift_count"] == 1 and \
        sorted(json.dumps(t, sort_keys=True) for t in rec["lifts"][0]) == \
        sorted(json.dumps(t, sort_keys=True) for t in want)
    nonconst = all(len({e for _, e in f["choice"]}) == 5 for f in rec["stabilizer_patterns"])
    rec["terms_match_board"] = bool(ok)
    rec["patterns_nonconstant"] = bool(nonconst)
    rec["passed"] = bool(ok and nonconst)
    write_result("control_basis", rec)
    print(f"control-basis {'PASSED' if rec['passed'] else 'FAILED'}: the lift is the board's five Z(x)Z "
          f"sectors, patterns c(z) = c - z")
    return 0 if rec["passed"] else 1


def cmd_control_rank3(a):
    """p = 5, m = 1 -> 2 through every rank-3 decomposition of |T5> (the
    lemma at exact rank 3): no lift, consistent with chi(T5^2) = 5."""
    K = Cyclo(5)
    D = dictionary(5, 1)
    codes = phase_codes(D, 5)
    target = t5_target(K, 1)
    sets = all_minimal_sets(K, codes, target, 3, 1)
    _, info = symmetry_orbit_reps("T5", 1, D, antiunitary=False)
    closed = orbit_closure(sets, info["perms"])
    print(f"rank-3 decompositions of |T5>: {len(sets)} (closure {len(closed)}, group order {info['order']})")
    prob = LiftProblem(5, 1, K, codes, target, t5_alpha(K), closed)
    rec = run_problem(prob, "T5 rank-3 m=1 -> 2", expect_lifts=0)
    rec["decompositions"] = len(sets)
    rec["passed"] = bool(rec["passed"] and len(sets) == 10 and len(closed) == 10)
    write_result("control_rank3", rec)
    print(f"control-rank3 {'PASSED' if rec['passed'] else 'FAILED'}: 10 rank-3 decompositions, 0 lifts "
          f"(chi(T5^2) >= 4 by the lemma; the cell is 5)")
    return 0 if rec["passed"] else 1


def cmd_control_product(a):
    """p = 5, m = 2 -> 3 with the one-qudit factor a full-support stabilizer
    state beta instead of |T5>: the planted rank-5 decomposition
    sum_c w^(c^3) L_c (x) beta of |T5>^2 (x) beta must be recovered as the
    only lift, from the same census family."""
    K = Cyclo(5)
    E = _enumerator()
    codes = E.codes
    target = t5_target(K, 2)
    family = orbit_closure([HIT], E.info["perms"])
    results = []
    for name, expo in (("plus", [0] * 5), ("w^(z^2)", [z * z % 5 for z in range(5)]),
                       ("w^(2z^2 + 3z)", [(2 * z * z + 3 * z) % 5 for z in range(5)])):
        beta = [K.root(e) for e in expo]
        prob = LiftProblem(5, 2, K, codes, target, beta, family)
        rec = run_problem(prob, f"T5^2 (x) beta, beta = {name}", expect_lifts=1)
        # the recovered terms are L_c (x) beta: lines x + y = c (x) full support in z
        terms_ok = rec["lift_count"] == 1 and all(t["k"] == 2 for t in rec["lifts"][0])
        const = all(len({e for _, e in f["choice"]}) == 1 for f in rec["stabilizer_patterns"])
        rec["terms_k2"] = bool(terms_ok)
        rec["patterns_constant"] = bool(const)
        rec["beta_exponents"] = expo
        rec["passed"] = bool(rec["passed"] and terms_ok and const and rec["stats"]["stabilizer"] == 5)
        results.append(rec)
    passed = all(r["passed"] for r in results)
    write_result("control_product", {"label": "planted product lifts", "plants": results, "passed": passed})
    print(f"control-product {'PASSED' if passed else 'FAILED'}: {len(results)} planted products, each recovered "
          f"as the only lift (5 stabilizer patterns, the constant ones)")
    return 0 if passed else 1


def t3_alpha(K):
    """|T3> up to 1/sqrt 3: (1, zeta_9, zeta_9^2) (stabrank_verify.orbit_state)."""
    return [K.root(K.n // 9 * x) for x in range(3)]


def t3_target(K, m):
    return [K.root(K.n // 9 * (sum(pt) % 9)) for pt in itertools.product(range(3), repeat=m)]


def cmd_control_t3_basis(a):
    """p = 3, m = 1 -> 2, field Q(zeta_9), through the computational basis:
    the carry decomposition of |T3>^2 (three Z(x)Z sectors) is the lift."""
    K = Cyclo(9)
    D = dictionary(3, 1)
    codes = phase_codes(D, 3)
    basis = point_states(codes)
    prob = LiftProblem(3, 1, K, codes, t3_target(K, 1), t3_alpha(K), [tuple(basis)])
    rec = run_problem(prob, "T3 basis m=1 -> 2", expect_lifts=1)
    write_result("control_t3_basis", rec)
    print(f"control-t3-basis {'PASSED' if rec['passed'] else 'FAILED'}: the carry decomposition of |T3>^2")
    return 0 if rec["passed"] else 1


def cmd_control_t3_full(a):
    """p = 3, m = 1 -> 2 through every rank-3 decomposition of |T3> (184
    independent triples of the 12 states): lifts exist (chi(T3^2) = 3)."""
    K = Cyclo(9)
    D = dictionary(3, 1)
    codes = phase_codes(D, 3)
    target = t3_target(K, 1)
    sets = all_minimal_sets(K, codes, target, 3, 3)
    print(f"rank-3 decompositions of |T3>: {len(sets)}")
    prob = LiftProblem(3, 1, K, codes, target, t3_alpha(K), sets)
    rec = run_problem(prob, "T3 all rank-3 m=1 -> 2", expect_lifts="some")
    rec["decompositions"] = len(sets)
    write_result("control_t3_full", rec)
    print(f"control-t3-full {'PASSED' if rec['passed'] else 'FAILED'}: {rec['lift_count']} lifts "
          f"(chi(T3^2) = 3, lifts must exist)")
    return 0 if rec["passed"] else 1


def _numeric_sets(orbit, m, rank, D):
    """Rank-`rank` decompositions of |orbit>^m up to the unitary symmetry
    group by the numerical pivot search of slice_lift, closed under the
    group; the exact solve in LiftProblem re-decides every one."""
    from slice_lift import all_decompositions
    decs, _ = all_decompositions(orbit, m, rank, D, verbose=False)
    _, info = symmetry_orbit_reps(orbit, m, D, antiunitary=False)
    return orbit_closure(decs, info["perms"]), info


def cmd_control_t3_m2(a):
    """p = 3, m = 2 -> 3: the carry decomposition is the only rank-3
    decomposition of |T3>^2 up to symmetry, and it must not lift
    (chi(T3^3) = 8 on the board)."""
    K = Cyclo(9)
    D = dictionary(3, 2)
    codes = phase_codes(D, 3)
    sets, info = _numeric_sets("T3", 2, 3, D)
    print(f"rank-3 decompositions of |T3>^2: {len(sets)} (group order {info['order']})")
    prob = LiftProblem(3, 2, K, codes, t3_target(K, 2), t3_alpha(K), sets)
    rec = run_problem(prob, "T3 m=2 -> 3", expect_lifts=0)
    write_result("control_t3_m2", rec)
    print(f"control-t3-m2 {'PASSED' if rec['passed'] else 'FAILED'}: 0 lifts (chi(T3^3) = 8)")
    return 0 if rec["passed"] else 1


def cmd_control_n_m2(a):
    """p = 3, m = 2 -> 3 for the Norrell state |N> = (1, 1, -2)/sqrt 6 (field
    Q(zeta_3), rational amplitudes): the 48 rank-3 decompositions of |N>^2
    do not lift (chi(N^3) = 4), the negative control of slice_lift.py made
    exact."""
    K = Cyclo(3)
    D = dictionary(3, 2)
    codes = phase_codes(D, 3)
    sets, info = _numeric_sets("N", 2, 3, D)
    print(f"rank-3 decompositions of |N>^2: {len(sets)} (group order {info['order']})")
    alpha = [K.from_int(1), K.from_int(1), K.from_int(-2)]
    target = []
    for pt in itertools.product(range(3), repeat=2):
        v = K.one
        for x in pt:
            v = K.mul(v, alpha[x])
        target.append(v)
    prob = LiftProblem(3, 2, K, codes, target, alpha, sets)
    rec = run_problem(prob, "N m=2 -> 3", expect_lifts=0)
    rec["passed"] = bool(rec["passed"] and len(sets) == 48)
    write_result("control_n_m2", rec)
    print(f"control-n-m2 {'PASSED' if rec['passed'] else 'FAILED'}: 48 decompositions, 0 lifts (chi(N^3) = 4)")
    return 0 if rec["passed"] else 1


def cmd_control_stabtest(a):
    """The exact stabilizer test on random stabilizer states of three
    ququints built from the (flat, quadratic phase) parametrization, and on
    three perturbations of each (one phase changed, one support point
    removed, one modulus changed), which must fail."""
    rng = np.random.default_rng(a.seed)
    K = Cyclo(5)
    p, m = 5, 3
    ok, n = True, 0
    for _ in range(a.count):
        k = int(rng.integers(0, m + 1))
        # a random flat: random k independent directions and a base point
        while True:
            dirs = rng.integers(0, p, size=(k, m))
            W, _ = rref_mod_p(dirs.tolist(), p) if k else ([], [])
            if len(W) == k:
                break
        x0 = tuple(int(x) for x in rng.integers(0, p, size=m))
        Q = rng.integers(0, p, size=(k, k))
        ell = rng.integers(0, p, size=k)
        scal = K.mul(K.root(int(rng.integers(0, 5))), K.from_int(int(rng.integers(1, 4))))
        entries = {}
        for y in itertools.product(range(p), repeat=k):
            x = tuple((x0[c] + sum(y[r] * W[r][c] for r in range(k))) % p for c in range(m))
            q = sum(int(Q[i][j]) * y[i] * y[j] for i in range(k) for j in range(i, k)) + \
                sum(int(ell[i]) * y[i] for i in range(k))
            entries[x] = K.mul(scal, K.root(q % p))
        form = stabilizer_form(K, p, m, entries)
        good = form is not None and form["k"] == k
        if good:
            # the form rebuilds the same entries up to the scalar
            ref = entries[tuple(form["x0"])]
            for y in itertools.product(range(p), repeat=k):
                x = tuple((form["x0"][c] + sum(y[r] * form["W"][r][c] for r in range(k))) % p for c in range(m))
                q = sum(form["Q"][i][j] * y[i] * y[j] for i in range(k) for j in range(i, k)) + \
                    sum(form["l"][i] * y[i] for i in range(k))
                if entries[x] != K.mul(ref, K.root(q % p)):
                    good = False
        n += 1
        pts = sorted(entries)
        # perturbations
        if k >= 1:
            e1 = dict(entries)
            x = pts[int(rng.integers(0, len(pts)))]
            e1[x] = K.mul(e1[x], K.root(int(rng.integers(1, 5))))
            good = good and (stabilizer_form(K, p, m, e1) is None or k == 0)
            e2 = dict(entries)
            del e2[pts[int(rng.integers(0, len(pts)))]]
            good = good and stabilizer_form(K, p, m, e2) is None
        e3 = dict(entries)
        x = pts[int(rng.integers(0, len(pts)))]
        e3[x] = K.mul(e3[x], K.from_int(2))
        good = good and (stabilizer_form(K, p, m, e3) is None or len(pts) == 1)
        ok = ok and good
        if not good:
            print(f"  FAILED on a random state with k = {k}")
    rec = {"label": "exact stabilizer test", "count": n, "seed": a.seed, "passed": bool(ok)}
    write_result("control_stabtest", rec)
    print(f"control-stabtest {'PASSED' if ok else 'FAILED'}: {n} random stabilizer states recognized, their "
          f"perturbations rejected")
    return 0 if ok else 1


def _enumerator():
    sys.path.insert(0, CENSUS)
    import common as census_common  # noqa: E402
    return census_common.make_enumerator(native=False)


def cmd_plan(a):
    """Pattern counts for the cell and the controls, without searching."""
    print("cell T5 m=2 -> 3: family of 1 set (the census hit, fixed by the group), r = 5: 5^5 = 3125 patterns; "
          "the prunes leave the 25 affine patterns c(z) = c0 + d z for the stabilizer test")
    print("control-rank3: 10 sets x 3 members per slice: 30^5 = 24.3 M patterns before pruning")
    print("control-n-m2: 48 sets x 3 members per slice: 144^3 = 2.99 M patterns before pruning")
    print("control-t3-full: 184 sets x 3: 552^3 = 168 M patterns before pruning")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--results-dir", default=None, help="the census records (default research/t5q_m2_rank5/results)")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--low-census", action="store_true", help="re-run the exact k = 1..4 censuses (about 150 s)")
    g.add_argument("--trust-low-census", action="store_true",
                   help="rely on the stored control_rank4.json (the aggregate re-runs the censuses)")
    for name in ("control-basis", "control-rank3", "control-product", "control-t3-basis", "control-t3-full",
                 "control-t3-m2", "control-n-m2", "plan"):
        sub.add_parser(name)
    s = sub.add_parser("control-stabtest")
    s.add_argument("--count", type=int, default=40)
    s.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(argv[1:])
    lower_priority()
    t0 = time.time()
    fn = {"cell": cmd_cell, "control-basis": cmd_control_basis, "control-rank3": cmd_control_rank3,
          "control-product": cmd_control_product, "control-t3-basis": cmd_control_t3_basis,
          "control-t3-full": cmd_control_t3_full, "control-t3-m2": cmd_control_t3_m2,
          "control-n-m2": cmd_control_n_m2, "control-stabtest": cmd_control_stabtest, "plan": cmd_plan}[a.cmd]
    rc = fn(a)
    print(f"[{time.time() - t0:.1f}s]")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
