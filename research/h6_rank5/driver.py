"""Resumable driver for the rank-5 exclusion of |H>^6 by an all-visible base
slice (docs/notes/h6_rank5_exclusion.md), and its controls.

Stage A: every full 5-cover of |H>^3 with five distinct, linearly
independent base states, one per orbit of the unitary symmetry group of
|H>^3, enumerated per pivot pair (i, j) by
verify_challenge/slice_cover.CoverEnumerator.pair_covers, and matched at
the four base points x_0 (one per Hamming weight) by SliceMatcher.run. The
matcher handles dependent and repeated base states (stages B and C of the
note) through the coefficient family and the block treatment of repeated
copies; the enumeration of those covers is `degenerate`.

Commands
  census                    run the 5-cover kernel alone over every pivot pair and
                            record the covers, candidates and seconds per pair
  partition [--target-s S]  write partition.json: pivot pairs grouped into
                            batches of about S seconds (kernel seconds plus the
                            matcher per cover, from the census when present)
  run BATCH                 run one batch, writing results/batch_BATCH.json
                            (skipped when the file exists: resume by rerunning)
  status                    coverage and totals over the result files
  control-witness           recover the rank-6 witness bounds/qubit_H-m6-upper-6.json
                            from its own all-visible (triple, base point) slices
  control-m4                recover the rank-4 decompositions of |H>^4 from the
                            full 4-covers of |H>^3 (one sliced qubit)
  sample [--count N]        time the matcher on N full 5-covers from the first
                            pivot at the four base points (--reference: Python path)
  fixture                   write the C++ test fixture from the reference matcher
                            on the sample covers
  degenerate [--sample N]   count the dependent and repeated 5-covers (stages B, C)
                            and time N of each multiplicity pattern
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
from slice_cover import (CoverEnumerator, Family, SliceMatcher, x0_reps, _reduce,  # noqa: E402
                         exact_codes, patterns)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402

PARTITION = os.path.join(HERE, "partition.json")
RESULTS = os.path.join(HERE, "results")
N1 = 3


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def hit_record(h, cover, x0):
    return {"cover": list(cover), "x0": x0, "residual": h["residual"], "rank": h["rank"],
            "exact": h["exact"], "independent": h["independent"], "nonzero": h["nonzero"],
            "free_parameters": h["free_parameters"],
            "coeffs": [[z.real, z.imag] for z in h["coeffs"]],
            "terms": [[[z.real, z.imag] for z in t] for t in h["terms"]]}


def genuine(h, rank):
    return h["rank"] == rank and h["exact"] and h["independent"] and h["nonzero"] and h["residual"] < 1e-8


# ------------------------------------------------------------ partition ----

def pairs_of(E):
    """Every (pivot, partner, member count) unit in the pivot order of the
    enumerator."""
    out = []
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        for j in partners:
            j = int(j)
            M = int(np.count_nonzero(members > j)) - (1 if i > j else 0)
            out.append((i, j, M))
    return out


CENSUS = os.path.join(RESULTS, "kernel_census.json")


def census(args):
    """Run the 5-cover kernel alone over every pivot pair and record, per
    unit, the covers found, the modular candidates and the seconds, so the
    partition can balance batches by the matcher's work (proportional to the
    covers) instead of by a kernel cost model."""
    E = CoverEnumerator(N1)
    units = pairs_of(E)
    t0 = time.time()
    rows, plans = [], {}
    for n, (i, j, M) in enumerate(units):
        if i not in plans:
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            plans[i] = (Qi, mask)
        Qi, mask = plans[i]
        tk = time.time()
        covers, nc = E.pair_covers(5, i, j, Qi, mask)
        rows.append([i, j, M, len(covers), nc, time.time() - tk])
        if args.verbose and (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(units)} units, {sum(r[3] for r in rows)} covers [{time.time() - t0:.0f}s]",
                  flush=True)
    rec = {"orbit": "qubit_H", "m": 6, "rank": 5, "n1": N1, "N": E.N, "git": git_commit(),
           "matcher": "native" if E.native_cover5 is not None else "reference",
           "units": len(units), "covers": int(sum(r[3] for r in rows)),
           "candidates": int(sum(r[4] for r in rows)), "seconds": time.time() - t0,
           "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"], "rows": rows}
    os.makedirs(RESULTS, exist_ok=True)
    with open(CENSUS, "w") as f:
        json.dump(rec, f)
    print(f"{len(units)} pivot pairs: {rec['covers']} full 5-covers, {rec['candidates']} candidates, "
          f"{rec['seconds']:.0f}s; wrote {CENSUS}")
    return 0


def partition(args):
    E = CoverEnumerator(N1)
    units = pairs_of(E)
    if os.path.exists(CENSUS) and not args.no_census:
        # measured kernel seconds per unit plus the matcher at its sampled
        # per-cover cost (the four base points)
        with open(CENSUS) as f:
            cen = json.load(f)
        by = {(r[0], r[1]): r for r in cen["rows"]}
        costs = [by[(i, j)][5] + args.match_ms * 1e-3 * by[(i, j)][3] for i, j, _ in units]
        model = {"census": os.path.relpath(CENSUS, HERE), "match_ms_per_cover": args.match_ms,
                 "census_covers": cen["covers"]}
    else:
        # kernel cost model: quadratic in M (the M x M residue array) plus a
        # per-unit floor; the matcher cost is then unknown before the run
        costs = [args.floor_s + args.k_s * (M / 1000.0) ** 2 for _, _, M in units]
        model = {"floor_s": args.floor_s, "k_s_per_M2_over_1e6": args.k_s}
    batches, cur, acc = [], [], 0.0
    for u, c in zip(units, costs):
        cur.append(list(u))
        acc += c
        if acc >= args.target_s:
            batches.append(cur)
            cur, acc = [], 0.0
    if cur:
        batches.append(cur)
    rec = {"orbit": "qubit_H", "m": 6, "rank": 5, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "pivot_orbits": int(E.info["orbits"]), "units": len(units), "batches": len(batches),
           "cost_model": model, "target_s": args.target_s,
           "estimated_s": float(sum(costs)), "git": git_commit(),
           "batch_units": batches}
    with open(PARTITION, "w") as f:
        json.dump(rec, f)
    print(f"{len(units)} pivot pairs, {len(batches)} batches, estimate "
          f"{sum(costs) / 3600:.2f} CPU-h at the stored rates; wrote {PARTITION}")


# ------------------------------------------------------------------ run ----

def match_cover(M, E, cover, rec, rank):
    """Run one cover at the four base points, accumulating into rec."""
    for x0 in x0_reps(N1):
        hits, st = M.run(cover, x0)
        if st["refused"]:
            rec["refused"] += 1
            continue
        rec["matched"] += 1
        k = ",".join(str(b) for b in st["coord_solutions"])
        rec["coord_solution_hist"][k] = rec["coord_solution_hist"].get(k, 0) + 1
        for h in hits:
            rec["hits"].append(hit_record(h, cover, x0))


def run(args):
    with open(PARTITION) as f:
        part = json.load(f)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"batch_{args.batch}.json")
    if os.path.exists(out) and not args.force:
        print(f"{out} exists; skipping")
        return 0
    units = part["batch_units"][args.batch]
    E = CoverEnumerator(N1)
    assert E.N == part["N"] and E.info["order"] == part["group_order"]
    M = SliceMatcher(E, N1)
    t0 = time.time()
    rec = {"batch": args.batch, "units": units, "git": git_commit(), "covers": 0, "candidates": 0,
           "refused": 0, "matched": 0, "hits": [], "coord_solution_hist": {}, "kernel_s": 0.0,
           "match_s": 0.0}
    for i, j, _ in units:
        tk = time.time()
        Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
        members, _ = E.pivot_plan(i)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        covers, nc = E.pair_covers(5, i, j, Qi, mask)
        rec["kernel_s"] += time.time() - tk
        rec["covers"] += len(covers)
        rec["candidates"] += nc
        tm = time.time()
        for cover in sorted(covers):
            match_cover(M, E, cover, rec, 5)
        rec["match_s"] += time.time() - tm
    rec["wall_s"] = time.time() - t0
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    with open(out, "w") as f:
        json.dump(rec, f)
    print(f"batch {args.batch}: {len(units)} units, {rec['covers']} covers, {rec['matched']} matched, "
          f"{rec['refused']} refused, {len(rec['hits'])} hits, "
          f"kernel {rec['kernel_s']:.0f}s, match {rec['match_s']:.0f}s")
    return 2 if rec["hits"] else 0


def status(args):
    with open(PARTITION) as f:
        part = json.load(f)
    done, tot = 0, {"covers": 0, "matched": 0, "refused": 0, "hits": 0, "kernel_s": 0.0, "match_s": 0.0}
    for b in range(part["batches"]):
        p = os.path.join(RESULTS, f"batch_{b}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rec = json.load(f)
        done += 1
        for k in tot:
            tot[k] += len(rec[k]) if k == "hits" else rec[k]
    print(f"{done}/{part['batches']} batches done; {tot}")
    if done:
        print(f"mean kernel {tot['kernel_s'] / done:.0f}s and match {tot['match_s'] / done:.0f}s per batch")


# ------------------------------------------------------------- controls ----

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


def control_witness(args):
    from common import term_vector, target
    m, n1, rank = 6, N1, 6
    E = CoverEnumerator(n1)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    w = json.load(open(os.path.join(ROOT, args.path)))
    terms = [term_vector(t, 2, m) for t in w["witness"]["terms"]]
    psi = target("qubit_H", m)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    assert np.linalg.norm(np.column_stack(terms) @ c - psi) < 1e-9
    bases = {}
    for S in itertools.combinations(range(m), n1):
        for x0 in range(1 << n1):
            b = slice_terms(terms, S, x0, m, n1, lookup)
            if b is not None:
                bases.setdefault((b, x0), []).append(S)
    M = SliceMatcher(E, n1, verbose=args.verbose)
    report = {"witness": args.path, "git": git_commit(), "bases": []}
    passed = 0
    selected = -1
    for (cover, x0), Ss in sorted(bases.items()):
        repeated = len(set(cover)) < len(cover)
        if args.distinct_only and repeated or args.repeated_only and not repeated:
            continue
        selected += 1
        if args.base is not None and selected != args.base:
            continue
        t0 = time.time()
        hits, st = M.run(cover, x0)
        good = [h for h in hits if genuine(h, rank)]
        # is the witness itself among the hits? compare the term patterns after the same permutation
        S = Ss[0]
        perm = list(S) + [q for q in range(m) if q not in S]
        wkey = sorted(exact_codes(v.reshape([2] * m).transpose(perm).reshape(-1))[0].tobytes() for v in terms)
        same = any(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == wkey for h in good)
        dt = time.time() - t0
        print(f"base {cover} x0 {x0:0{n1}b} ({len(Ss)} triples, {'repeated' if repeated else 'distinct'}, "
              f"kappa {st['kappa']}): {len(good)} rank-{rank} decompositions, witness itself "
              f"{'recovered' if same else 'not among them'}, {dt:.0f}s, stats {st}")
        report["bases"].append({"cover": list(cover), "x0": x0, "triples": Ss, "repeated": repeated,
                                "hits": len(good), "witness_recovered": same, "seconds": dt,
                                "stats": {k: v for k, v in st.items()}})
        passed += bool(good)
    os.makedirs(RESULTS, exist_ok=True)
    suffix = "_repeated" if args.repeated_only else ("_distinct" if args.distinct_only else "")
    if args.base is not None:
        suffix += f"_{args.base}"
    with open(os.path.join(RESULTS, f"control_witness{suffix}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"control-witness: {passed}/{len(report['bases'])} bases recover a rank-{rank} decomposition")
    return 0 if passed == len(report["bases"]) else 1


def m4_group(D4):
    """The unitary symmetry group of |H>^4 as permutations of the four-qubit
    dictionary, by closure of the generators."""
    _, info = symmetry_orbit_reps("qubit_H", 4, D4, antiunitary=False)
    gens = [np.asarray(p, dtype=np.int32) for p in info["perms"]]
    ident = np.arange(D4.shape[1], dtype=np.int32)
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
    assert len(seen) == info["order"], (len(seen), info["order"])
    return list(seen.values())


def canonical(idx, group):
    return min(tuple(sorted(int(x) for x in g[list(idx)])) for g in group)


def degenerate_covers(E, r, covers3, covers4):
    """Full r-covers (r = 4, 5) whose distinct states are dependent or which
    repeat a state, as multisets (sorted tuples): every multiset over a
    3-cover or 4-cover plus states of the span, with a coefficient family
    in which no unrepeated state is dead."""
    out = set()

    def ok(ms):
        distinct = sorted(set(ms))
        fam = Family.from_cover(E, distinct)
        if fam is None:
            return False
        exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
        return not fam.has_zero_coefficient(exempt)

    for T in covers3:
        span = [x for x in range(E.N) if x not in T and E.rank_mod2(tuple(T) + (x,), False) == 3]
        pool = list(T) + span
        for extra in itertools.combinations_with_replacement(pool, r - 3):
            ms = tuple(sorted(T + extra))
            if ok(ms):
                out.add(ms)
        if r == 5:
            # a pair x, y outside span(T) with y in span(T, x): the second pivot kernel
            F = E.F1
            rows = E.U1.copy()
            for b in T:
                rows, _ = _reduce(F, rows, rows[b])
            from slice_cover import _canon_rows, _groups_by_key
            Rc, has = _canon_rows(F, rows)
            ids = np.flatnonzero(has)
            key = (Rc[ids] @ E.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
            for g in _groups_by_key(key):
                for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                    ms = tuple(sorted(T + (a, b)))
                    if E.is_cover(ms) and ok(ms):
                        out.add(ms)
    if r == 5:
        for Cv in covers4:
            span = [x for x in range(E.N) if x not in Cv and E.rank_mod2(tuple(Cv) + (x,), False) == 4]
            for x in list(Cv) + span:
                ms = tuple(sorted(Cv + (x,)))
                if ok(ms):
                    out.add(ms)
    return sorted(out)


def control_m4(args):
    from common import load_decompositions
    m, n1, rank = 4, 1, 4
    E = CoverEnumerator(3)
    lookup3 = {E.codes[i].tobytes(): i for i in range(E.N)}
    t0 = time.time()
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    degenerate = degenerate_covers(E, 4, covers3, covers4)
    bases = sorted(set(covers4) | set(degenerate))
    print(f"{len(covers4)} independent full 4-covers, {len(degenerate)} dependent or repeated, "
          f"{len(bases)} bases [{time.time() - t0:.0f}s]")
    D4 = dictionary(2, 4)
    codes4, _ = patterns(D4)
    lookup4 = {codes4[i].tobytes(): i for i in range(D4.shape[1])}
    group = m4_group(D4)
    stored, _ = load_decompositions("qubit_H", m, rank)
    stored_keys = {}
    expected = set()
    for terms, _ in stored:
        key = canonical([lookup4[exact_codes(t)[0].tobytes()] for t in terms], group)
        stored_keys[key] = terms
        for q in range(m):
            for x0 in range(2):
                b = slice_terms(terms, (q,), x0, m, n1, lookup3)
                if b is None:
                    continue
                distinct = sorted(set(b))
                fam = Family.from_cover(E, distinct)
                exempt = [i for i, u in enumerate(distinct) if b.count(u) > 1]
                if fam is not None and not fam.has_zero_coefficient(exempt):
                    expected.add(key)
    M = SliceMatcher(E, n1, verbose=args.verbose)
    recovered = {}
    t1 = time.time()
    stats = {"matched": 0, "refused": 0, "hits": 0, "non_genuine": 0}
    for cover in bases:
        for x0 in range(2):
            hits, st = M.run(cover, x0)
            if st["refused"]:
                stats["refused"] += 1
                continue
            stats["matched"] += 1
            for h in hits:
                if not genuine(h, rank):
                    stats["non_genuine"] += 1
                    continue
                stats["hits"] += 1
                key = canonical([lookup4[exact_codes(t)[0].tobytes()] for t in h["terms"]], group)
                recovered.setdefault(key, []).append((cover, x0))
    dt = time.time() - t1
    unknown = set(recovered) - set(stored_keys)
    missing = expected - set(recovered)
    extra = set(recovered) - expected
    print(f"matched {stats['matched']} (cover, x0) pairs in {dt:.0f}s; {stats['hits']} hits, "
          f"{len(recovered)} classes of rank-{rank} decompositions of |H>^{m}; stored {len(stored_keys)}, "
          f"expected recoverable {len(expected)}; not stored {len(unknown)}, missing {len(missing)}, "
          f"unexpected {len(extra)}; stats {stats}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_m4.json"), "w") as f:
        json.dump({"git": git_commit(), "bases": len(bases), "independent": len(covers4),
                   "degenerate": len(degenerate), "stored": len(stored_keys), "expected": len(expected),
                   "recovered": len(recovered), "unknown": len(unknown), "missing": len(missing),
                   "extra": len(extra), "seconds": dt, "stats": stats,
                   "recovered_from": {str(k): v for k, v in recovered.items()}}, f, indent=1)
    ok = not unknown and not missing and not extra
    print("control-m4:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def sample(args):
    E = CoverEnumerator(N1)
    M = SliceMatcher(E, N1, native=not args.reference)
    i = int(E.reps[args.pivot])
    Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
    members, partners = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    covers = []
    tk = time.time()
    for j in partners:
        got, _ = E.pair_covers(5, i, int(j), Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= args.count:
            break
    covers = covers[:args.count]
    print(f"{len(covers)} covers from pivot {i} in {time.time() - tk:.1f}s of kernel")
    rec = {"covers": 0, "matched": 0, "refused": 0, "hits": [], "coord_solution_hist": {}}
    times = []
    for cover in covers:
        t0 = time.time()
        match_cover(M, E, cover, rec, 5)
        times.append(time.time() - t0)
        rec["covers"] += 1
    times = np.array(times)
    which = "reference" if M.native is None else "native"
    print(f"{which} matcher, per cover (four base points): mean {times.mean():.4f}s, "
          f"median {np.median(times):.4f}s, max {times.max():.4f}s; {rec['matched']} matched, "
          f"{rec['refused']} refused, {len(rec['hits'])} hits; coordinate-slice solution histogram "
          f"{rec['coord_solution_hist']}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, args.out or f"sample_{which}.json"), "w") as f:
        json.dump({"git": git_commit(), "pivot": i, "matcher": which, "covers": [list(c) for c in covers],
                   "seconds": times.tolist(), "matched": rec["matched"], "refused": rec["refused"],
                   "hits": rec["hits"], "coord_solution_hist": rec["coord_solution_hist"]}, f)
    return 0


def fixture(args):
    """Write cpp/tests/data/h6_rank5_sample.txt: the sample covers as base
    state codes with the reference matcher's per-run results, for the C++
    test of the compiled kernel (cpp/tests/test_slice_match.cpp)."""
    E = CoverEnumerator(N1)
    M = SliceMatcher(E, N1, native=False)
    with open(os.path.join(RESULTS, "sample.json")) as f:
        covers = [tuple(c) for c in json.load(f)["covers"]]
    out = os.path.join(ROOT, "cpp", "tests", "data", "h6_rank5_sample.txt")
    lines = ["# x0, five base states as phase codes (0 zero, 1..4 = 1, i, -1, -i), kappa, "
             "cumulative coordinate-slice solutions, hits; reference matcher on results/sample.json"]
    t0 = time.time()
    for cover in covers:
        for x0 in x0_reps(N1):
            hits, st = M.run(cover, x0)
            codes = " ".join("".join(str(int(c)) for c in E.codes[u]) for u in cover)
            sols = ",".join(str(v) for v in st["coord_solutions"]) or "-"
            lines.append(f"{x0} {codes} {st['kappa']} {sols} {len(hits)}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(lines) - 1} runs in {time.time() - t0:.1f}s")
    return 0


def degenerate(args):
    E = CoverEnumerator(N1)
    t0 = time.time()
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    deg = degenerate_covers(E, 5, covers3, covers4)
    by, groups = {}, {}
    for ms in deg:
        pat = tuple(sorted((ms.count(u) for u in set(ms)), reverse=True))
        by[str(pat)] = by.get(str(pat), 0) + 1
        groups.setdefault(str(pat), []).append(ms)
    print(f"{len(deg)} dependent or repeated full 5-covers over the 3-cover and 4-cover classes "
          f"({time.time() - t0:.0f}s); by multiplicity pattern {by}")
    if not args.sample:
        return 0
    # timing sample: --sample covers from each multiplicity pattern, evenly
    # spaced, through the reference matcher (blocks and coefficient families)
    M = SliceMatcher(E, N1)
    rec = {"covers": 0, "matched": 0, "refused": 0, "hits": [], "coord_solution_hist": {}}
    out = {"git": git_commit(), "patterns": {}}
    for pat, lst in sorted(groups.items()):
        picks = [lst[k] for k in np.linspace(0, len(lst) - 1, min(args.sample, len(lst))).astype(int)]
        times, kappas = [], []
        for cover in picks:
            t1 = time.time()
            match_cover(M, E, cover, rec, 5)
            times.append(time.time() - t1)
            kappas.append(M.run(cover, x0_reps(N1)[0])[1]["kappa"])
        times = np.array(times)
        print(f"pattern {pat}: {len(lst)} covers, sampled {len(picks)}: per cover (four base points) "
              f"mean {times.mean():.2f}s, median {np.median(times):.2f}s, max {times.max():.2f}s, "
              f"kappa {sorted(set(kappas))}; projected {len(lst) * times.mean() / 3600:.2f} CPU-h")
        out["patterns"][pat] = {"covers": len(lst), "sampled": len(picks), "seconds": times.tolist(),
                                "kappa": kappas, "projected_s": float(len(lst) * times.mean())}
    out["hits"] = rec["hits"]
    out["coord_solution_hist"] = rec["coord_solution_hist"]
    print(f"{rec['matched']} matched, {rec['refused']} refused, {len(rec['hits'])} hits; "
          f"projected total {sum(v['projected_s'] for v in out['patterns'].values()) / 3600:.2f} CPU-h")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "degenerate_sample.json"), "w") as f:
        json.dump(out, f)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=600.0)
    p.add_argument("--floor-s", type=float, default=0.05)
    p.add_argument("--k-s", type=float, default=8.0, help="kernel seconds per pair at M = 1000 (no census)")
    p.add_argument("--match-ms", type=float, default=1.4, help="matcher milliseconds per cover (census)")
    p.add_argument("--no-census", action="store_true", help="ignore results/kernel_census.json")
    p.set_defaults(fn=partition)
    p = sub.add_parser("census")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=census)
    p = sub.add_parser("run")
    p.add_argument("batch", type=int)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=run)
    p = sub.add_parser("status")
    p.set_defaults(fn=status)
    p = sub.add_parser("control-witness")
    p.add_argument("--path", default="bounds/qubit_H-m6-upper-6.json")
    p.add_argument("--distinct-only", action="store_true", help="skip bases with a repeated state")
    p.add_argument("--repeated-only", action="store_true", help="run only the bases with a repeated state")
    p.add_argument("--base", type=int, default=None, help="run only the k-th selected base (0-based)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_witness)
    p = sub.add_parser("control-m4")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_m4)
    p = sub.add_parser("sample")
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--pivot", type=int, default=0)
    p.add_argument("--reference", action="store_true", help="force the Python matcher")
    p.add_argument("--out", default=None, help="result file name under results/")
    p.set_defaults(fn=sample)
    p = sub.add_parser("fixture")
    p.set_defaults(fn=fixture)
    p = sub.add_parser("degenerate")
    p.add_argument("--sample", type=int, default=0, help="time N covers of each multiplicity pattern")
    p.set_defaults(fn=degenerate)
    args = ap.parse_args(argv[1:])
    os.nice(19)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
