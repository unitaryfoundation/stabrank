"""Plan, partition, rate sample and controls of the rank-5 exclusion of
|T5>^2 by direct census (docs/notes/t5q_m2_rank5_exclusion.md).

    driver.py plan
        The 155,422 (pivot, partner, members above the partner) units of
        the 5-cover enumeration over the 3,900 two-ququint states, the
        squared-member cost model, the dictionary and plan hashes; written
        to results/plan.json.
    driver.py sample [--count K] [--budget S] [--seed S]
        A stratified sample of units through the compiled kernel (at least
        one per pivot); fits seconds per squared member count and writes
        results/rates.json and results/sample.json.
    driver.py partition [--target-s 600] [--pod-factor 1.3] [--sec-per-m2 X]
        Contiguous batches of units of about --target-s pod seconds each,
        costed at sec_per_m2 x M^2 x pod_factor (sec_per_m2 from
        results/rates.json unless given); writes partition.json, hashed.
    driver.py control-pattern
        The modular target equals the phase pattern w_5^(x^3 + y^3) of the
        complex |T5>^2, and both primes carry a fifth root of unity.
    driver.py control-planted [--count 24] [--reference] [--seed S]
        Planted five-term targets: random independent 5-sets with the
        coefficients a permutation of 1..5, with two equal coefficients,
        five states of one Pauli orbit, a repeated state (a 4-set target),
        and a dependent 5-set (five states spanning four dimensions). The
        kernel must list every independent plant from its two smallest
        members with both fullness flags, with the planted coefficients
        recovered; every degenerate plant must be found by the k = 4 census
        from its two smallest members, and the kernel's 5-sets for it must
        all be non-full supersets of a 4-subset. --reference runs the same
        plants through the Python reference of the kernel.
    driver.py control-rank4
        The k = 1, 2, 3, 4 censuses of psi_2 through the same reductions:
        all must be empty (chi(T5^2) >= 5, agreeing with
        verify_challenge/cert_t5_m2_rank4.py's numerical exclusion), which
        the rank-5 census argument needs.
    driver.py control-m1
        The one-ququint cell: the k = 3 census of |T5> over the 30 states
        against the brute-force list of rank-3 decompositions (ten, in
        orbits under the order-5 unitary group), and k = 1, 2 empty.
    driver.py control-hash
        Re-enumerate the dictionary and the units and compare their hashes
        with the partition's.
    driver.py control-reference [--count 12]
        The compiled kernel against the Python reference on sampled units:
        equal member counts and equal 5-set lists (both expected empty).

Every control writes results/control_<name>.json and exits 1 on failure.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (research/t5q_m2_rank5/common.py)
from common import (HERE, P1, P2, PARTITION, RANK, RATES, RESULTS, ROOT, W5, M, ORBIT, T5Enumerator,  # noqa: E402
                    make_enumerator, git_commit, sha256_json, write_hashed)

PLAN = os.path.join(RESULTS, "plan.json")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(s):
    print(s, flush=True)


def _write(name, rec):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    log(f"wrote {os.path.relpath(path, ROOT)}")


def plan_units(E):
    units = [list(u) for u in E.units()]
    return units, sha256_json(units)


def cost_model(units, sec_per_m2, pod_factor):
    Ms = np.array([u[2] for u in units], dtype=float)
    return {"sum_m2": float(np.sum(Ms ** 2)), "mean_m": float(Ms.mean()), "max_m": float(Ms.max()),
            "sec_per_m2": sec_per_m2, "pod_factor": pod_factor,
            "laptop_s": float(np.sum(Ms ** 2)) * sec_per_m2,
            "pod_s": float(np.sum(Ms ** 2)) * sec_per_m2 * pod_factor}


# ---------------------------------------------------------------- plan ----

def plan(args):
    t0 = time.time()
    E = make_enumerator()
    units, plan_sha = plan_units(E)
    cm = cost_model(units, common.SEC_PER_M2_DEFAULT, 1.3)
    rec = {"orbit": ORBIT, "m": M, "rank": RANK, "N": E.N, "group_order": E.info["order"],
           "pivots": len(E.reps), "units_count": len(units), "dictionary_sha256": E.dictionary_sha256(),
           "plan_sha256": plan_sha, "cost_model": cm, "git": git_commit(), "generated": _now(),
           "seconds": time.time() - t0, "per_pivot": [[int(i), int(np.count_nonzero(np.array(units)[:, 0] == i))]
                                                       for i in E.reps]}
    _write("plan.json", rec)          # the units themselves are stored once, in the partition
    log(f"{E.N} states, group order {E.info['order']}, {len(E.reps)} pivots, {len(units)} units; "
        f"members above the partner: mean {cm['mean_m']:.0f}, max {cm['max_m']:.0f}, sum of squares "
        f"{cm['sum_m2']:.3g}; at {cm['sec_per_m2']:.3g} s per M^2 the census is {cm['laptop_s'] / 3600:.1f} "
        f"laptop CPU-h, {cm['pod_s'] / 3600:.1f} pod CPU-h at factor {cm['pod_factor']}; dictionary sha256 "
        f"{rec['dictionary_sha256'][:16]}, plan sha256 {plan_sha[:16]}")
    return 0


# -------------------------------------------------------------- sample ----

def stratified(units, K, rng):
    """About K units, at least one per pivot, drawn at random within each
    pivot's partner list in proportion to its length."""
    by = {}
    for t, u in enumerate(units):
        by.setdefault(u[0], []).append(t)
    total = len(units)
    sel = []
    for i, lst in by.items():
        k = max(1, int(round(K * len(lst) / total)))
        k = min(k, len(lst))
        sel.extend(int(x) for x in rng.choice(lst, size=k, replace=False))
    return sorted(sel)


def sample(args):
    common.lower_priority()
    E = make_enumerator()
    if E.native_cover5 is None:
        raise SystemExit("the compiled kernel is not available; the rate sample needs it")
    units, plan_sha = plan_units(E)
    rng = np.random.default_rng(args.seed)
    sel = stratified(units, args.count, rng)
    rows, masks = [], {}
    t_all = time.time()
    for n, t in enumerate(sel):
        if args.budget and time.time() - t_all > args.budget:
            log(f"budget reached after {n} units")
            break
        i, j, Mp = units[t]
        if i not in masks:
            masks[i] = E.member_mask(i)
        tk = time.time()
        sets, nc, Mk = E.pair_sets_native(i, j, masks[i])
        dt = time.time() - tk
        rows.append([t, i, j, Mp, Mk, len(sets), nc, dt])
        if sets:
            log(f"  unit {t} ({i}, {j}): {len(sets)} 5-set(s) with psi in their span")
    Mk = np.array([r[4] for r in rows], dtype=float)
    Mp = np.array([r[3] for r in rows], dtype=float)
    secs = np.array([r[7] for r in rows])
    fit_k = float(np.sum(secs * Mk ** 2) / np.sum(Mk ** 4))
    fit_p = float(np.sum(secs * Mp ** 2) / np.sum(Mp ** 4))
    cm = cost_model(units, fit_p, 1.3)
    rec = {"seed": args.seed, "git": git_commit(), "generated": _now(), "units_sampled": len(rows),
           "units": len(units), "plan_sha256": plan_sha, "seconds": float(secs.sum()),
           "sec_per_m2_plan": fit_p, "sec_per_m2_kernel": fit_k,
           "sets": int(sum(r[5] for r in rows)), "candidates": int(sum(r[6] for r in rows)),
           "columns": ["unit", "pivot", "partner", "members_plan", "members_kernel", "sets", "candidates", "seconds"],
           "rows": rows, "projected": cm}
    _write("sample.json", rec)
    rates = {"sec_per_m2": fit_p, "sec_per_m2_kernel": fit_k, "units_sampled": len(rows),
             "seconds": float(secs.sum()), "seed": args.seed, "git": git_commit(), "generated": _now(),
             "hostname": __import__("socket").gethostname()}
    with open(RATES, "w") as f:
        json.dump(rates, f, indent=1)
        f.write("\n")
    log(f"{len(rows)} units in {secs.sum():.0f}s: {rec['sets']} 5-sets, {rec['candidates']} modular candidates; "
        f"fit {fit_p:.3g} s per plan M^2 ({fit_k:.3g} per kernel M^2); census {cm['laptop_s'] / 3600:.1f} "
        f"laptop CPU-h, {cm['pod_s'] / 3600:.1f} pod CPU-h at factor 1.3; wrote {os.path.relpath(RATES, ROOT)}")
    return 0


# ----------------------------------------------------------- partition ----

def partition(args):
    E = make_enumerator()
    units, plan_sha = plan_units(E)
    if args.sec_per_m2 is not None:
        spm = args.sec_per_m2
        src = "command line"
    elif os.path.exists(RATES):
        with open(RATES) as f:
            spm = json.load(f)["sec_per_m2"]
        src = os.path.relpath(RATES, HERE)
    else:
        spm = common.SEC_PER_M2_DEFAULT
        src = "default (the feasibility note's fit)"
    F = args.pod_factor
    costs = [spm * u[2] ** 2 * F for u in units]
    total = sum(costs)
    geometry, start, acc = [], 0, 0.0
    for t, c in enumerate(costs):
        acc += c
        if acc >= args.target_s or t == len(units) - 1:
            geometry.append({"index": len(geometry), "start": start, "end": t + 1,
                             "n_units": t + 1 - start, "est_pod_s": acc})
            start, acc = t + 1, 0.0
    rec = {"orbit": ORBIT, "m": M, "rank": RANK, "N": E.N, "group_order": E.info["order"],
           "pivots": len(E.reps), "units_count": len(units), "dictionary_sha256": E.dictionary_sha256(),
           "plan_sha256": plan_sha,
           "cost_model": {"sec_per_m2": spm, "source": src, "pod_factor": F, "target_s": args.target_s,
                          "sum_m2": float(sum(u[2] ** 2 for u in units))},
           "estimated_pod_s": total, "estimated_laptop_s": total / F,
           "batches": len(geometry), "git": git_commit(), "generated": _now(),
           "batch_geometry": geometry, "units": units}
    sha = write_hashed(PARTITION, rec)
    est = np.array([g["est_pod_s"] for g in geometry])
    log(f"{len(units)} units in {len(geometry)} batches of about {args.target_s:.0f} pod s (min {est.min():.0f}, "
        f"max {est.max():.0f}, {est.mean():.0f} mean; the last batch {est[-1]:.0f}); {total / 3600:.1f} pod CPU-h "
        f"at {spm:.3g} s per M^2 ({src}) x {F}; wrote {os.path.relpath(PARTITION, ROOT)} (sha256 {sha[:16]})")
    return 0


# ------------------------------------------------------------ controls ----

def control_pattern(args):
    E = make_enumerator()
    pts = list(itertools.product(range(5), repeat=2))
    tc = np.array([(x ** 3 + y ** 3) % 5 for x, y in pts])
    ratio = E.psi / E.psi[0]
    ok_pattern = bool(np.allclose(ratio, W5 ** tc, atol=1e-9))
    ok_codes = bool(np.array_equal(E.tcodes.astype(int) - 1, tc))
    ok_mod = bool(np.array_equal(E.psi1, E.F1.wpow[tc]) and np.array_equal(E.psi2, E.F2.wpow[tc]))
    ok_roots = (pow(E.F1.w, 5, P1) == 1 and E.F1.w != 1 and pow(E.F2.w, 5, P2) == 1 and E.F2.w != 1
                and (P1 - 1) % 5 == 0 and (P2 - 1) % 5 == 0)
    # every dictionary column is a phase pattern (patterns5 asserted it); the
    # mod-P1 images of two states are proportional only when the states agree
    nz_ok = bool(np.array_equal(E.U1 != 0, E.codes > 0) and np.array_equal(E.U2 != 0, E.codes > 0))
    ok = ok_pattern and ok_codes and ok_mod and ok_roots and nz_ok
    rec = {"git": git_commit(), "generated": _now(), "pattern": ok_pattern, "codes": ok_codes,
           "modular": ok_mod, "roots_of_unity": ok_roots, "supports": nz_ok, "w5_mod_p1": E.F1.w,
           "w5_mod_p2": E.F2.w, "N": E.N, "group_order": E.info["order"], "pivots": len(E.reps),
           "dictionary_sha256": E.dictionary_sha256(), "passed": ok}
    _write("control_pattern.json", rec)
    log(f"control-pattern: the complex |T5>^2 has the phase pattern w_5^(x^3 + y^3) [{ok_pattern}], the codes "
        f"agree [{ok_codes}], the modular targets agree [{ok_mod}], w_5 = {E.F1.w} mod {P1} and {E.F2.w} mod {P2} "
        f"are fifth roots of unity [{ok_roots}], supports agree [{nz_ok}]: {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def _lookup(E):
    def key(v):
        v = np.asarray(v, dtype=complex)
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        return (np.round(v, 6) + 0.0).tobytes()
    return key, {key(E.C[:, s]): s for s in range(E.N)}


def pauli_image(v, a, b):
    """X^a Z^b on two ququints (a, b in F_5^2) applied to the pattern v
    indexed by 5 x + y: (X^a Z^b v)[x] = w^(b . (x - a)) v[x - a]."""
    out = np.zeros_like(v)
    for x in range(5):
        for y in range(5):
            sx, sy = (x - a[0]) % 5, (y - a[1]) % 5
            out[5 * x + y] = W5 ** ((b[0] * sx + b[1] * sy) % 5) * v[5 * sx + sy]
    return out


def plant_sets(E, rng, count):
    """(kind, states, coefficients) for the planted control; states sorted,
    repeated states allowed for the `repeated` kind."""
    key, lookup = _lookup(E)
    kinds = ["random"] * max(1, count // 2) + ["equal"] * max(1, count // 6) + ["pauli"] * max(1, count // 6)
    kinds += ["repeated"] * max(1, count // 12) + ["dependent"] * max(1, count // 12)
    kinds = kinds[:max(count, 5)]
    out = []
    for kind in kinds:
        if kind in ("random", "equal"):
            while True:
                S = sorted(int(x) for x in rng.choice(E.N, size=5, replace=False))
                if np.linalg.matrix_rank(E.C[:, S], tol=1e-8) == 5:
                    break
            c = np.array([1, 2, 3, 4, 5] if kind == "random" else [1, 1, 2, 3, 4])
            rng.shuffle(c)
        elif kind == "pauli":
            s = int(rng.integers(E.N))
            paulis = [(a, b) for a in itertools.product(range(5), repeat=2) for b in itertools.product(range(5), repeat=2)]
            orbit = sorted({lookup[key(pauli_image(E.C[:, s], a, b))] for a, b in paulis})
            S = sorted(int(x) for x in rng.choice(orbit, size=5, replace=False))
            c = np.array([1, 2, 3, 4, 5])
            rng.shuffle(c)
        elif kind == "repeated":
            while True:
                D4 = sorted(int(x) for x in rng.choice(E.N, size=4, replace=False))
                if np.linalg.matrix_rank(E.C[:, D4], tol=1e-8) == 4:
                    break
            S = sorted(D4 + [D4[int(rng.integers(4))]])
            c = np.array([1, 2, 3, 4, 5])
            rng.shuffle(c)
        else:
            # three point states on the line {(t, 0)} and two line states on
            # it whose restrictions to t = 3, 4 are parallel: five states of
            # rank four (the restriction argument in the note)
            pts = []
            for t in range(3):
                v = np.zeros(25, dtype=complex)
                v[5 * t] = 1
                pts.append(lookup[key(v)])
            q1, l1 = int(rng.integers(5)), int(rng.integers(5))
            q2, l2 = (q1 + 1) % 5, (l1 + 3) % 5
            lines = []
            for q, l in ((q1, l1), (q2, l2)):
                v = np.zeros(25, dtype=complex)
                for t in range(5):
                    v[5 * t] = W5 ** ((q * t * t + l * t) % 5)
                lines.append(lookup[key(v)])
            S = sorted(pts + lines)
            if np.linalg.matrix_rank(E.C[:, S], tol=1e-8) != 4:
                raise AssertionError("the dependent plant does not have rank 4")
            c = np.array([1, 2, 3, 4, 5])
            rng.shuffle(c)
        out.append((kind, [int(x) for x in S], [int(x) for x in c]))
    return out


def control_planted(args):
    common.lower_priority()
    E = make_enumerator()
    if E.native_cover5 is None and not args.reference:
        raise SystemExit("the compiled kernel is not available; use --reference")
    rng = np.random.default_rng(args.seed)
    plants = plant_sets(E, rng, args.count)
    mask = np.ones(E.N, dtype=bool)
    rec = {"seed": args.seed, "git": git_commit(), "generated": _now(), "path": "reference" if args.reference
           else "native", "plants": []}
    n_ok = 0
    t_all = time.time()
    for kind, S, c in plants:
        T = E.planted_target(S, c)
        dist = sorted(set(S))
        i, j = dist[0], dist[1]
        t0 = time.time()
        entry = {"kind": kind, "states": S, "coeffs": c, "pivot": i, "partner": j}
        try:
            sets, nc, Mk = E.pair_sets(i, j, mask, native=not args.reference, target=T)
        except Exception as exc:                                  # noqa: BLE001
            entry.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            rec["plants"].append(entry)
            log(f"  {kind} {S}: ERROR {exc}")
            continue
        found = {s for s, _, _ in sets}
        entry.update({"sets": len(sets), "candidates": nc, "members": Mk})
        if kind in ("random", "equal", "pauli"):
            flags = next(((f1, f2) for s, f1, f2 in sets if s == tuple(S)), None)
            d = E.decide(S, T)
            coeffs = np.array([complex(a, b) for a, b in d["coeffs"]])
            same_c = bool(np.allclose(coeffs, np.array(c, dtype=complex), atol=1e-6))
            ok = flags == (True, True) and d["decomposition"] and d["independent"] and same_c
            entry.update({"listed": flags is not None, "full_flags": list(flags) if flags else None,
                          "decomposition": d["decomposition"], "coefficients_recovered": same_c,
                          "full_sets": sum(f1 and f2 for _, f1, f2 in sets)})
        else:
            # a degenerate plant is a rank-4 target: the k = 4 census from the
            # same pivot and partner must list a 4-subset of its states, the
            # dependent 5-set itself is never a kernel set (two of its members
            # have zero images modulo the span of the other three and the
            # target), and every kernel set's fullness flags must agree with
            # the exact and numerical decision (a target on a line has other
            # genuine full 5-term decompositions, which the kernel lists)
            hits4, nc4 = E.low_sets(4, i, j, mask, target=T)
            subsets4 = [h for h in hits4 if set(h["states"]) <= set(dist) and h["decomposition"]]
            flags_ok, full_sets, genuine = True, 0, 0
            for s, f1, f2 in sets:
                d = E.decide(s, T)
                full_sets += f1 and f2
                genuine += d["decomposition"] and d["independent"] and d["nonzero"]
                if not d["decomposition"] or (f1 and f2) != (d["independent"] and d["nonzero"]):
                    flags_ok = False
            not_listed = tuple(S) not in found
            contains = any(set(h["states"]) <= set(s) for s in found for h in subsets4)
            ok = bool(subsets4) and flags_ok and not_listed and contains
            entry.update({"k4_hits": len(hits4), "k4_candidates": nc4, "k4_subsets": [h["states"] for h in subsets4],
                          "kernel_full_sets": int(full_sets), "kernel_genuine_full": int(genuine),
                          "flags_agree_with_decisions": flags_ok, "planted_set_not_listed": not_listed,
                          "some_set_contains_a_4_subset": contains})
        entry.update({"status": "recovered" if ok else "NOT recovered", "seconds": time.time() - t0})
        n_ok += ok
        rec["plants"].append(entry)
        log(f"  {kind:9s} {S} c={c}: {entry['status']}, {len(sets)} kernel set(s), {nc} candidates, "
            f"M={Mk}, {entry['seconds']:.1f}s")
    rec.update({"recovered": n_ok, "total": len(plants), "passed": n_ok == len(plants),
                "seconds": time.time() - t_all})
    _write("control_planted" + ("_reference" if args.reference else "") + ".json", rec)
    log(f"control-planted ({rec['path']}): {n_ok} of {len(plants)} planted targets recovered in "
        f"{rec['seconds']:.0f}s: {'PASSED' if rec['passed'] else 'FAILED'}")
    return 0 if rec["passed"] else 1


def low_census_all(E, log_fn=log):
    """The k = 1..4 censuses of psi over the pivot plan; (records, passed)."""
    out = {}
    for k in (1, 2, 3, 4):
        t0 = time.time()
        r = E.census_low(k)
        r["seconds"] = time.time() - t0
        out[str(k)] = r
        log_fn(f"  k = {k}: {len(r['hits'])} set(s) with psi in their span, {r['candidates']} candidates over "
               f"{r['units']} unit(s), {r['seconds']:.1f}s")
    passed = all(not r["hits"] for r in out.values())
    return out, passed


def control_rank4(args):
    common.lower_priority()
    E = make_enumerator()
    t0 = time.time()
    out, passed = low_census_all(E)
    rec = {"git": git_commit(), "generated": _now(), "N": E.N, "group_order": E.info["order"],
           "pivots": len(E.reps), "census": out, "passed": passed, "seconds": time.time() - t0}
    _write("control_rank4.json", rec)
    log(f"control-rank4: ranks 1 to 4 {'excluded by the exact census' if passed else 'NOT excluded'} in "
        f"{rec['seconds']:.0f}s (cert_t5_m2_rank4.py excludes them numerically): "
        f"{'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


def control_m1(args):
    common.lower_priority()
    E1 = T5Enumerator(n=1)
    t0 = time.time()
    census = {}
    for k in (1, 2, 3):
        census[k] = E1.census_low(k)
    hits3 = [tuple(h["states"]) for h in census[3]["hits"] if h["decomposition"]]
    brute = []
    for S in itertools.combinations(range(E1.N), 3):
        d = E1.decide(S)
        if not d["agree"]:
            raise AssertionError(f"modular and numeric decisions disagree on {S}")
        if d["decomposition"] and d["independent"] and d["nonzero"]:
            brute.append(S)
    # orbits of the brute-force list under the unitary group's permutations
    perms = E1.info["perms"]
    seen, orbits = set(), []
    for S in brute:
        if S in seen:
            continue
        orb, frontier = {S}, [S]
        while frontier:
            T = frontier.pop()
            for pm in perms:
                U = tuple(sorted(int(pm[x]) for x in T))
                if U not in orb:
                    orb.add(U)
                    frontier.append(U)
        seen |= orb
        orbits.append(sorted(orb))
    census_covers = all(any(h in orb for h in hits3) for orb in orbits)
    census_inside = all(h in brute for h in hits3)
    ok = (len(brute) == 10 and census_covers and census_inside and not census[1]["hits"]
          and not census[2]["hits"])
    rec = {"git": git_commit(), "generated": _now(), "N": E1.N, "group_order": E1.info["order"],
           "pivots": len(E1.reps), "brute_force_rank3": [list(S) for S in brute], "orbits": len(orbits),
           "orbit_sizes": [len(o) for o in orbits], "census_rank3": [list(h) for h in hits3],
           "census_k1_hits": len(census[1]["hits"]), "census_k2_hits": len(census[2]["hits"]),
           "census_k3_candidates": census[3]["candidates"], "every_orbit_hit": census_covers,
           "every_census_hit_genuine": census_inside, "passed": ok, "seconds": time.time() - t0}
    _write("control_m1.json", rec)
    log(f"control-m1: |T5> has {len(brute)} rank-3 decompositions over the {E1.N} one-ququint states in "
        f"{len(orbits)} orbit(s) of the order-{E1.info['order']} unitary group (sizes {rec['orbit_sizes']}); the "
        f"k = 3 census lists {len(hits3)} of them, hitting every orbit [{census_covers}], all genuine "
        f"[{census_inside}]; k = 1, 2 empty [{not census[1]['hits'] and not census[2]['hits']}]: "
        f"{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def control_hash(args):
    part = common.load_partition(args.partition)
    E = make_enumerator()
    units, plan_sha = plan_units(E)
    dsha = E.dictionary_sha256()
    ok = (dsha == part["dictionary_sha256"] and plan_sha == part["plan_sha256"] and units == part["units"]
          and E.N == part["N"] and E.info["order"] == part["group_order"])
    rec = {"git": git_commit(), "generated": _now(), "dictionary_sha256": dsha,
           "partition_dictionary_sha256": part["dictionary_sha256"], "plan_sha256": plan_sha,
           "partition_plan_sha256": part["plan_sha256"], "units": len(units), "passed": ok}
    _write("control_hash.json", rec)
    log(f"control-hash: dictionary {dsha[:16]} {'==' if dsha == part['dictionary_sha256'] else '!='} partition's, "
        f"plan {plan_sha[:16]} {'==' if plan_sha == part['plan_sha256'] else '!='} partition's: "
        f"{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def control_reference(args):
    common.lower_priority()
    E = make_enumerator()
    if E.native_cover5 is None:
        raise SystemExit("the compiled kernel is not available")
    units, _ = plan_units(E)
    rng = np.random.default_rng(args.seed)
    # prefer small units so that the reference stays cheap
    small = [t for t, u in enumerate(units) if u[2] <= args.max_members]
    sel = sorted(int(x) for x in rng.choice(small, size=min(args.count, len(small)), replace=False))
    rows, ok = [], True
    for t in sel:
        i, j, Mp = units[t]
        mask = E.member_mask(i)
        t0 = time.time()
        sn, cn, Mn = E.pair_sets_native(i, j, mask)
        tn = time.time() - t0
        t0 = time.time()
        sr, cr, Mr = E.pair_sets_reference(i, j, mask)
        tr = time.time() - t0
        same = sorted(sn) == sorted(sr) and Mn == Mr
        ok &= same
        rows.append([t, i, j, Mp, Mn, Mr, len(sn), len(sr), cn, cr, tn, tr, same])
        log(f"  unit {t} ({i}, {j}): members {Mn}/{Mr}, sets {len(sn)}/{len(sr)}, candidates {cn}/{cr}, "
            f"{tn:.2f}s/{tr:.2f}s {'agree' if same else 'DIFFER'}")
    rec = {"seed": args.seed, "git": git_commit(), "generated": _now(),
           "columns": ["unit", "pivot", "partner", "members_plan", "members_native", "members_reference",
                       "sets_native", "sets_reference", "candidates_native", "candidates_reference",
                       "seconds_native", "seconds_reference", "agree"],
           "rows": rows, "passed": ok}
    _write("control_reference.json", rec)
    log(f"control-reference: {len(rows)} units, native and reference {'agree on every one' if ok else 'DIFFER'}: "
        f"{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.set_defaults(fn=plan)
    s = sub.add_parser("sample")
    s.add_argument("--count", type=int, default=150)
    s.add_argument("--budget", type=float, default=0)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=sample)
    s = sub.add_parser("partition")
    s.add_argument("--target-s", type=float, default=600.0)
    s.add_argument("--pod-factor", type=float, default=1.3)
    s.add_argument("--sec-per-m2", type=float, default=None)
    s.set_defaults(fn=partition)
    s = sub.add_parser("control-pattern")
    s.set_defaults(fn=control_pattern)
    s = sub.add_parser("control-planted")
    s.add_argument("--count", type=int, default=24)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--reference", action="store_true")
    s.set_defaults(fn=control_planted)
    s = sub.add_parser("control-rank4")
    s.set_defaults(fn=control_rank4)
    s = sub.add_parser("control-m1")
    s.set_defaults(fn=control_m1)
    s = sub.add_parser("control-hash")
    s.add_argument("--partition", default=PARTITION)
    s.set_defaults(fn=control_hash)
    s = sub.add_parser("control-reference")
    s.add_argument("--count", type=int, default=12)
    s.add_argument("--max-members", type=int, default=1200)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=control_reference)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
