"""Setup, partition and controls for the rank-5 exclusion of |N>^4 and |H3>^4
by a two-qutrit all-visible base slice (docs/notes/qutrit_m4_rank5_exclusion.md).
The batches themselves run through batch.py and are checked by aggregate.py.

Stage A: every full 5-cover of |M>^2 with five distinct, linearly independent
base states, one per orbit of the unitary symmetry group G_2 of |M>^2,
enumerated per pivot pair (i, j) by cover_census.CoverEnumerator3.pair_covers
and matched at the one base point of the cell (x0 = (0, 0) for N, (1, 1) for
H3) by matcher.Matcher.run. Stage B: the full 5-covers of five distinct but
dependent states; stage C: the covers with a repeated state. Both are listed
once by `degenerate --write` and matched through the coefficient family and
the block treatment of repeated copies.

Commands (ORBIT is N or H3)
  census ORBIT              run the 5-cover kernel over every pivot pair, record the
                            covers, candidates and seconds per pair and the hashed
                            lists of full 3-, 4- and 5-covers (results/ORBIT/kernel_census.json)
  degenerate ORBIT [--sample K] [--write]
                            list the dependent and repeated full 5-covers (stages B, C),
                            time K of each multiplicity pattern through the matcher,
                            and with --write store the list with its hash
  partition ORBIT [--target-s S] [--target-bc-s S] [--match-ms X]
                            write partition_ORBIT.json: stage A pivot pairs grouped
                            into batches of about S seconds (census seconds plus X ms
                            per cover), stage B and C covers round-robin into batches
                            of about --target-bc-s seconds at the sampled rates
  sample ORBIT [--count K]  time the matcher on K stage A covers drawn from the census
  control-covers ORBIT      the census's full 3-covers and 4-covers against the stored
                            rank-3 list and slice_lift.all_decompositions at rank 4
  control-planted ORBIT [--count K]
                            planted five-term decompositions recovered from their base
                            slice: generic, with a diagonal line term, dependent and
                            repeated bases
  control-product ORBIT     the product decompositions phi (x) (rank-3 decomposition of
                            |M>^2) recovered from every base point
  control-m3 ORBIT          the rank-4 decompositions of |M>^3 recovered by 2 + 1 slicing
                            from every full 4-cover of |M> by single-qutrit states
  control-witness ORBIT [--base K]
                            the rank-7 N witness (Lean) or the rank-8 H3 witness
                            recovered from its all-visible base points
  export-witness            write N_m4_rank7_witness.json from the Lean term definitions

Running a batch: `batch.py ORBIT K`; checking the results: `aggregate.py ORBIT`.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import HERE, M, N1, N2, RANK, ROOT, X0, git_commit, pairs_of, stage_of  # noqa: E402
from cover_census import (P1, P2, CoverEnumerator3, Field3, _canon_rows, _groups_by_key,  # noqa: E402
                          _reduce)
from matcher import (COMP, E1, E2, PTS, W3, Matcher, Target, TermOpts, add, constructions_common,  # noqa: E402
                     exact_codes, field_vector, move_front, pidx, psi_target, slice_base, vector_target)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402
from slice_cover import Family  # noqa: E402

WITNESS_N = os.path.join(HERE, "N_m4_rank7_witness.json")
WITNESS_H3 = os.path.join(ROOT, "bounds", "H3-m4-upper-8.json")


def genuine(h, rank):
    return h["rank"] == rank and h["exact"] and h["independent"] and h["nonzero"] and h["residual"] < 1e-8


def hit_summary(orbit, h, cover, x0):
    det, num = common.hit_record(orbit, h, cover, x0)
    return {**det, **num}


def enumerator(orbit):
    return CoverEnumerator3(orbit, N2)


def new_matcher(n2=N2, verbose=False):
    return Matcher(dictionary(3, n2), n2, Field3(P1), Field3(P2), verbose=verbose)


# -------------------------------------------------------------- census ----

def census(args):
    """The 5-cover kernel over every pivot pair (the reference kernel of
    cover_census, every candidate re-decided mod P2 and numerically), with
    the full 3-covers and 4-covers, stored with the file's hash."""
    orbit = args.orbit
    E = enumerator(orbit)
    units = pairs_of(E)
    t0 = time.time()
    small = {}
    for r in (3, 4):
        t = time.time()
        cv, nc = E.covers(r)
        small[r] = (cv, nc, time.time() - t)
        print(f"  r={r}: {len(cv)} full covers, {nc} candidates [{time.time() - t:.1f}s]", flush=True)
    rows, plans, found = [], {}, set()
    t5 = time.time()
    for n, (i, j, Mc) in enumerate(units):
        if i not in plans:
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            plans[i] = (Qi, mask)
        Qi, mask = plans[i]
        tk = time.time()
        covers, nc, _ = E.pair_covers(5, i, j, Qi, mask)
        rows.append([i, j, Mc, len(covers), nc, time.time() - tk])
        found.update(covers)
        if args.verbose and (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(units)} units, {sum(r[3] for r in rows)} covers [{time.time() - t0:.0f}s]",
                  flush=True)
    rec = {"orbit": orbit, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "pivot_orbits": int(E.info["orbits"]), "git": git_commit(),
           "units": len(units), "covers": int(sum(r[3] for r in rows)),
           "candidates": int(sum(r[4] for r in rows)), "seconds": time.time() - t5,
           "covers3": [list(c) for c in small[3][0]], "candidates3": int(small[3][1]),
           "covers4": [list(c) for c in small[4][0]], "candidates4": int(small[4][1]),
           "covers5": [list(c) for c in sorted(found)],
           "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"], "rows": rows}
    if len(found) != rec["covers"]:
        raise AssertionError(f"{rec['covers']} covers over the pairs but {len(found)} distinct")
    sha = common.write_hashed(common.census_path(orbit), rec)
    print(f"{orbit}: {len(units)} pivot pairs, {rec['covers']} full 5-covers, {rec['candidates']} candidates, "
          f"{rec['seconds']:.0f}s; {len(small[3][0])} full 3-covers, {len(small[4][0])} full 4-covers; "
          f"wrote {os.path.relpath(common.census_path(orbit), ROOT)} (sha256 {sha[:16]})")
    return 0


def small_covers(orbit, E):
    """The full 3-covers and 4-covers, from the census when present."""
    path = common.census_path(orbit)
    if os.path.exists(path):
        cen = common.load_census(orbit, path)
        return [tuple(c) for c in cen["covers3"]], [tuple(c) for c in cen["covers4"]]
    c3, _ = E.covers(3)
    c4, _ = E.covers(4)
    return c3, c4


# ------------------------------------------------------------ degenerate ----

def degenerate_covers(E, r, covers3, covers4):
    """Full r-covers (r = 4, 5) whose distinct states are dependent or which
    repeat a state, as multisets (sorted tuples): every multiset over a
    3-cover or 4-cover plus states of the span, or a 3-cover plus a pair
    parallel modulo it, with a coefficient family in which no unrepeated
    state is dead (the shape of research/h6_rank5/driver.degenerate_covers)."""
    out = set()

    def ok(ms):
        distinct = sorted(set(ms))
        fam = Family.from_cover(E, distinct)
        if fam is None:
            return False
        exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
        return not fam.has_zero_coefficient(exempt)

    for T in covers3:
        span = E.in_span(T)
        pool = list(T) + span
        for extra in itertools.combinations_with_replacement(pool, r - 3):
            ms = tuple(sorted(T + extra))
            if ok(ms):
                out.add(ms)
        if r == 5:
            F = E.F1
            rows = E.U1.copy()
            for b in T:
                rows, _ = _reduce(F, rows, rows[b])
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
            span = E.in_span(Cv)
            for x in list(Cv) + span:
                ms = tuple(sorted(Cv + (x,)))
                if ok(ms):
                    out.add(ms)
    return sorted(out)


def degenerate(args):
    orbit = args.orbit
    E = enumerator(orbit)
    t0 = time.time()
    covers3, covers4 = small_covers(orbit, E)
    deg = degenerate_covers(E, RANK, covers3, covers4)
    by, groups = {}, {}
    for ms in deg:
        pat = str(common.multiplicity_pattern(ms))
        by[pat] = by.get(pat, 0) + 1
        groups.setdefault(pat, []).append(ms)
    dt = time.time() - t0
    nB = sum(1 for c in deg if stage_of(c) == "B")
    print(f"{orbit}: {len(deg)} dependent or repeated full 5-covers over {len(covers3)} 3-covers and "
          f"{len(covers4)} 4-covers ({dt:.0f}s): {nB} dependent (stage B), {len(deg) - nB} repeated "
          f"(stage C); by multiplicity pattern {by}")
    if args.write:
        rec = {"orbit": orbit, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
               "covers3": len(covers3), "covers4": len(covers4), "count": len(deg),
               "stage_b_covers": nB, "stage_c_covers": len(deg) - nB,
               "by_pattern": by, "seconds": dt, "git": git_commit(),
               "covers": [list(c) for c in deg]}
        sha = common.write_hashed(common.degenerate_path(orbit), rec)
        print(f"wrote {os.path.relpath(common.degenerate_path(orbit), ROOT)} (sha256 {sha[:16]})")
    if not args.sample:
        return 0
    Mt = new_matcher()
    target = psi_target(orbit, N2, Mt.F1, Mt.F2)
    x0 = X0[orbit]
    out = {"orbit": orbit, "x0": list(x0), "git": git_commit(), "sample": args.sample, "cap_s": args.cap_s,
           "patterns": {}, "hits": [], "coord_solution_hist": {}}
    path = common.degenerate_sample_path(orbit)
    os.makedirs(common.results_dir(orbit), exist_ok=True)

    def flush():
        # written after every cover, so that a run killed by an outer wall-clock
        # cap (a single repeated dependent cover can run for hours) leaves the
        # covers measured so far
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)

    # the sample is spread evenly over each pattern's sorted list, the same
    # covers every time; --cap-s bounds the wall time per pattern, so a
    # pattern with a heavy tail keeps the covers it finished
    ts = time.time()
    budget = args.cap_s / len(groups) if args.cap_s else None
    for pat, lst in sorted(groups.items()):
        picks = [lst[k] for k in np.linspace(0, len(lst) - 1, min(args.sample, len(lst))).astype(int)]
        rec = {"covers": len(lst), "sampled": 0, "planned": len(picks), "seconds": [], "kappa": [],
               "refused": 0, "capped": False, "projected_s": None}
        out["patterns"][pat] = rec
        t_pat = time.time()
        for cover in picks:
            if budget is not None and rec["sampled"] and time.time() - t_pat > budget:
                rec["capped"] = True
                break
            t1 = time.time()
            hits, st = Mt.run(cover, x0, target)
            rec["seconds"].append(time.time() - t1)
            rec["kappa"].append(st["kappa"])
            rec["refused"] += st["refused"]
            rec["sampled"] += 1
            k = ",".join(str(b) for b in st["coord_solutions"])
            out["coord_solution_hist"][k] = out["coord_solution_hist"].get(k, 0) + 1
            for h in hits:
                out["hits"].append(hit_summary(orbit, h, cover, x0))
            rec["projected_s"] = float(len(lst) * np.mean(rec["seconds"]))
            flush()
        times = np.array(rec["seconds"])
        print(f"pattern {pat}: {len(lst)} covers, sampled {len(times)} of {len(picks)} planned"
              f"{' (capped)' if rec['capped'] else ''}: per cover mean {times.mean():.2f}s, "
              f"median {np.median(times):.2f}s, max {times.max():.2f}s, kappa {sorted(set(rec['kappa']))}, "
              f"{rec['refused']} refused; projected {rec['projected_s'] / 3600:.2f} CPU-h", flush=True)
    print(f"{len(out['hits'])} hits; projected total "
          f"{sum(v['projected_s'] for v in out['patterns'].values()) / 3600:.2f} CPU-h at the sampled means "
          f"[{time.time() - ts:.0f}s]; wrote {os.path.relpath(path, ROOT)}")
    return 0


# ------------------------------------------------------------- partition ----

def partition(args):
    orbit = args.orbit
    E = enumerator(orbit)
    units = pairs_of(E)
    cen = common.load_census(orbit)
    by = {(r[0], r[1]): r for r in cen["rows"]}
    if sorted(by) != sorted((u[0], u[1]) for u in units):
        raise ValueError("the census's pivot pairs are not the enumerator's")
    costs = [by[(i, j)][5] + args.match_ms * 1e-3 * by[(i, j)][3] for i, j, _ in units]
    batches, cur, acc = [], [], 0.0
    for u, c in zip(units, costs):
        cur.append(list(u))
        acc += c
        if acc >= args.target_s:
            batches.append(cur)
            cur, acc = [], 0.0
    if cur:
        batches.append(cur)
    geometry = [{"index": k, "stage": "A", "units": b} for k, b in enumerate(batches)]
    rec = {"orbit": orbit, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "pivot_orbits": int(E.info["orbits"]), "x0": list(X0[orbit]), "units": len(units),
           "stage_a_batches": len(batches), "stage_b_batches": 0, "stage_c_batches": 0,
           "cost_model": {"census": os.path.relpath(common.census_path(orbit), HERE),
                          "census_sha256": cen["sha256"], "match_ms_per_cover": args.match_ms,
                          "census_covers": cen["covers"]},
           "target_s": args.target_s, "target_bc_s": args.target_bc_s,
           "estimated_s": {"A": float(sum(costs)), "B": 0.0, "C": 0.0}}
    covers, deg = common.load_degenerate(orbit)
    with open(common.degenerate_sample_path(orbit)) as f:
        smp = json.load(f)
    rates = {pat: float(np.mean(v["seconds"])) for pat, v in smp["patterns"].items()}
    rec["cost_model"]["degenerate_sample"] = os.path.relpath(common.degenerate_sample_path(orbit), HERE)
    rec["cost_model"]["s_per_cover_by_pattern"] = rates
    ids = {"B": [], "C": []}
    cost = {"B": 0.0, "C": 0.0}
    for k, c in enumerate(covers):
        st = stage_of(c)
        ids[st].append(k)
        cost[st] += rates[str(common.multiplicity_pattern(c))]
    for st in ("B", "C"):
        if not ids[st]:
            continue
        n = max(1, math.ceil(cost[st] / args.target_bc_s))
        # round robin over the sorted cover list: the cost of a cover
        # correlates with its 3-cover or 4-cover, so contiguous chunks
        # would not balance
        for b in range(n):
            geometry.append({"index": len(geometry), "stage": st, "cover_ids": ids[st][b::n]})
        rec[f"stage_{st.lower()}_batches"] = n
        rec["estimated_s"][st] = cost[st]
    rec["degenerate"] = {"file": os.path.relpath(common.degenerate_path(orbit), HERE), "sha256": deg["sha256"],
                         "count": len(covers), "stage_b_covers": len(ids["B"]),
                         "stage_c_covers": len(ids["C"])}
    rec["estimated_s"]["total"] = sum(rec["estimated_s"].values())
    rec["batches"] = len(geometry)
    rec["git"] = git_commit()
    rec["generated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec["batch_geometry"] = geometry
    sha = common.write_hashed(common.partition_path(orbit), rec)
    e = rec["estimated_s"]
    print(f"{orbit}: {len(units)} pivot pairs in {rec['stage_a_batches']} stage A batches "
          f"({e['A'] / 3600:.2f} CPU-h), {rec['stage_b_batches']} stage B batches ({e['B'] / 3600:.2f} CPU-h), "
          f"{rec['stage_c_batches']} stage C batches ({e['C'] / 3600:.2f} CPU-h); {rec['batches']} batches, "
          f"{e['total'] / 3600:.1f} CPU-h at the stored rates; wrote "
          f"{os.path.relpath(common.partition_path(orbit), ROOT)} (sha256 {sha[:16]})")
    return 0


def sample(args):
    orbit = args.orbit
    cen = common.load_census(orbit)
    covers = [tuple(c) for c in cen["covers5"]]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(covers), size=min(args.count, len(covers)), replace=False)
    Mt = new_matcher()
    target = psi_target(orbit, N2, Mt.F1, Mt.F2)
    x0 = X0[orbit]
    times, hist, hits, refused = [], {}, [], 0
    for k in pick:
        cover = covers[k]
        t = time.time()
        h, st = Mt.run(cover, x0, target)
        times.append(time.time() - t)
        key = ",".join(str(b) for b in st["coord_solutions"])
        hist[key] = hist.get(key, 0) + 1
        refused += st["refused"]
        for hh in h:
            hits.append(hit_summary(orbit, hh, cover, x0))
    times = np.array(times)
    print(f"{orbit} stage A at x0 {x0}: {len(pick)} covers, per cover mean {times.mean():.4f}s, "
          f"median {np.median(times):.4f}s, max {times.max():.4f}s; {refused} refused, {len(hits)} hits; "
          f"histogram {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    os.makedirs(common.results_dir(orbit), exist_ok=True)
    with open(os.path.join(common.results_dir(orbit), "sample.json"), "w") as f:
        json.dump({"orbit": orbit, "git": git_commit(), "x0": list(x0), "covers": [list(covers[k]) for k in pick],
                   "seconds": times.tolist(), "refused": refused, "hits": hits, "coord_solution_hist": hist}, f)
    return 0


# -------------------------------------------------------------- controls ----

def group_closure(info, N):
    """The unitary symmetry group as permutations of the dictionary, by
    closure of the generators."""
    gens = [np.asarray(p, dtype=np.int32) for p in info["perms"]]
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
    assert len(seen) == info["order"], (len(seen), info["order"])
    return list(seen.values())


def canonical(idx, group):
    return min(tuple(sorted(int(x) for x in g[list(idx)])) for g in group)


def write_control(orbit, name, doc):
    os.makedirs(common.results_dir(orbit), exist_ok=True)
    path = os.path.join(common.results_dir(orbit), f"{name}.json")
    with open(path, "w") as f:
        json.dump({"orbit": orbit, "git": git_commit(), **doc}, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def control_covers(args):
    """Control (a): the census's full 3-covers against the stored rank-3
    decompositions of |M>^2 (fullness and G_2-equivalence, both ways), and
    the full 4-covers against slice_lift.all_decompositions(orbit, 2, 4)
    restricted by fullness."""
    from slice_lift import all_decompositions
    orbit = args.orbit
    E = enumerator(orbit)
    t0 = time.time()
    covers3, covers4 = small_covers(orbit, E)
    group = group_closure(E.info, E.N)
    Mt = new_matcher()
    cc = constructions_common()
    stored, rec3 = cc.load_decompositions(orbit, N2, 3)
    stored_idx = [tuple(sorted(Mt.index_of(u) for u in terms)) for terms, _ in stored]
    not_full = [c for c in stored_idx if not E.is_full(c)]
    stored_classes = {canonical(c, group) for c in stored_idx}
    census_classes3 = {canonical(c, group) for c in covers3}
    missing3 = stored_classes - census_classes3
    unknown3 = census_classes3 - stored_classes
    print(f"r=3: {len(covers3)} census covers in {len(census_classes3)} G_2 classes; stored list "
          f"{len(stored)} decompositions in {len(stored_classes)} classes, {len(not_full)} not full; "
          f"missing from the census {len(missing3)}, not in the stored list {len(unknown3)} "
          f"[{time.time() - t0:.0f}s]")
    t1 = time.time()
    decs4, _ = all_decompositions(orbit, N2, 4, D=E.D, verbose=False)
    decs4 = [tuple(sorted(int(x) for x in d)) for d in decs4]
    full4 = [d for d in decs4 if E.is_full(d)]
    ref_classes4 = {canonical(c, group) for c in full4}
    census_classes4 = {canonical(c, group) for c in covers4}
    missing4 = ref_classes4 - census_classes4
    unknown4 = census_classes4 - ref_classes4
    dup4 = len(covers4) - len(census_classes4)
    print(f"r=4: {len(covers4)} census covers in {len(census_classes4)} G_2 classes ({dup4} duplicates); "
          f"slice_lift.all_decompositions {len(decs4)} decompositions, {len(full4)} full, "
          f"{len(ref_classes4)} classes; missing from the census {len(missing4)}, not in the reference "
          f"{len(unknown4)} [{time.time() - t1:.0f}s]")
    ok = not not_full and not missing3 and not unknown3 and not missing4 and not unknown4
    write_control(orbit, "control_covers", {
        "covers3": len(covers3), "classes3": len(census_classes3), "stored3": len(stored),
        "stored3_classes": len(stored_classes), "stored3_not_full": len(not_full),
        "missing3": len(missing3), "unknown3": len(unknown3),
        "covers4": len(covers4), "classes4": len(census_classes4), "reference4": len(decs4),
        "reference4_full": len(full4), "reference4_classes": len(ref_classes4),
        "missing4": len(missing4), "unknown4": len(unknown4), "pass": ok})
    print("control-covers:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def random_shape_term(o, x0, rng, kind=None):
    """A random four-qutrit stabilizer term with base slice o.u at x0, as a
    (9, dim) array of slices, of the given flat kind: 'plane', 'line' along
    e_1, e_2, (1, 1) or (1, 2) ('line1', 'line2', 'diag1', 'diag2') or
    'point'. Every shape of the structure lemma is a stabilizer state (the
    count closes the dictionary), so the term is one."""
    kinds = ["plane", "line1", "line2", "diag1", "diag2", "point"]
    kind = kind or kinds[int(rng.integers(0, len(kinds)))]
    dim = len(o.u)
    t = np.zeros((9, dim), dtype=complex)
    t[pidx(x0)] = o.u
    if kind == "plane":
        k1, k2 = rng.integers(0, o.nclass, size=2)
        a, b, c, d, e = rng.integers(0, 3, size=5)
        for x in PTS:
            if x == (0, 0):
                continue
            q = (a * x[0] * x[0] + b * x[1] * x[1] + c * x[0] * x[1] + d * x[0] + e * x[1]) % 3
            t[pidx(add(x0, x))] = o.vecs[3 * o.cls[k1, k2, pidx(x)] + (q + o.g[k1, k2, pidx(x)]) % 3]
    elif kind != "point":
        v = {"line1": (1, 0), "line2": (0, 1), "diag1": (1, 1), "diag2": (1, 2)}[kind]
        k = int(rng.integers(0, o.nclass))
        a, b = rng.integers(0, 3, size=2)
        t[pidx(add(x0, v))] = o.vecs[3 * k + (a + b) % 3]
        x2 = add(v, v)
        t[pidx(add(x0, x2))] = o.vecs[3 * o.cls[k, 0, pidx((2, 0))] + (o.g[k, 0, pidx((2, 0))] + a + 2 * b) % 3]
    return t.ravel(), kind


def control_planted(args):
    """Control (b): planted five-term decompositions of random targets,
    recovered from their base slice at x0. Bases: generic (distinct
    independent states, random flats), with a diagonal line term, dependent
    (five distinct states from the stage B list) and repeated (a state twice,
    from the stage C list). The planted decomposition must be among the hits
    and every hit must be a genuine decomposition of the target."""
    orbit = args.orbit
    Mt = new_matcher()
    x0 = X0[orbit]
    rng = np.random.default_rng(5)
    cen = common.load_census(orbit)
    covers5 = [tuple(c) for c in cen["covers5"]]
    deg, _ = common.load_degenerate(orbit)
    depB = [c for c in deg if stage_of(c) == "B"]
    repC = [c for c in deg if stage_of(c) == "C"]
    report = {"x0": list(x0), "cases": {}}
    ok_all = True
    for case in ("generic", "diagonal", "dependent", "repeated"):
        done, extra, tsum, spurious = 0, 0, 0.0, 0
        while done < args.count:
            if case in ("generic", "diagonal"):
                base = covers5[int(rng.integers(0, len(covers5)))]
            elif case == "dependent":
                base = depB[int(rng.integers(0, len(depB)))]
            else:
                base = repC[int(rng.integers(0, len(repC)))]
            terms, kinds = [], []
            for n, u in enumerate(base):
                kind = "diag1" if case == "diagonal" and n == 0 else None
                if case == "diagonal" and n == 1:
                    kind = "diag2"
                t, k = random_shape_term(Mt.options(u), x0, rng, kind)
                terms.append(t)
                kinds.append(k)
            T = np.column_stack(terms)
            if np.linalg.matrix_rank(T, tol=1e-8) < len(base):
                continue                                  # a repeated state gave equal terms
            coeffs = rng.integers(1, 4, size=len(base)) * rng.choice([1, -1], size=len(base))
            vec = T @ coeffs.astype(complex)
            target = vector_target(vec, N2, Mt.F1, Mt.F2)
            if slice_base(Mt, T, x0) != tuple(base) and sorted(slice_base(Mt, T, x0)) != sorted(base):
                raise AssertionError("planted base differs from the intended one")
            t0 = time.time()
            hits, st = Mt.run(base, x0, target)
            tsum += time.time() - t0
            if st["refused"]:
                raise AssertionError(f"planted base refused ({case}): {base}, {st}")
            wkey = sorted(exact_codes(c)[0].tobytes() for c in terms)
            same = [h for h in hits if sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == wkey]
            if not same:
                print(f"  {case}: NOT recovered: base {base}, kinds {kinds}, stats {st}")
                ok_all = False
            for h in hits:
                if h["exact"] != (h["residual"] < 1e-7):
                    ok_all = False                        # modular and numeric decisions disagree
                if not h["exact"]:
                    spurious += 1                         # a candidate the final check rejects
            extra += len(hits) - len(same)
            done += 1
        print(f"{case}: {done} planted decompositions recovered, {extra} further decompositions with the same "
              f"base, {spurious} candidates rejected by the final check, {tsum / done:.3f}s per run")
        report["cases"][case] = {"planted": done, "extra": extra, "rejected": spurious, "seconds": tsum}
    report["pass"] = ok_all
    write_control(orbit, "control_planted", report)
    print("control-planted:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def control_product(args):
    """phi (x) |M>^2 with phi a full-support two-qutrit stabilizer state has
    the three-term decompositions phi (x) (rank-3 decomposition of |M>^2);
    the matcher must find them from the base slice at every x0."""
    orbit = args.orbit
    Mt = new_matcher()
    cc = constructions_common()
    decs, _ = cc.load_decompositions(orbit, N2, 3)
    rng = np.random.default_rng(3)
    q = rng.integers(0, 3, size=5)
    phi = np.array([W3 ** ((q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + q[2] * x[0] * x[1] + q[3] * x[0] + q[4] * x[1]) % 3)
                    for x in PTS])
    aC = cc.alpha(orbit)
    vec = np.kron(phi, np.kron(aC, aC))
    a1, a2 = Mt.F1.alpha(orbit), Mt.F2.alpha(orbit)
    T_1 = np.kron(field_vector(phi, Mt.F1), np.kron(a1, a1) % P1) % P1
    T_2 = np.kron(field_vector(phi, Mt.F2), np.kron(a2, a2) % P2) % P2
    target = Target(vec, T_1, T_2, N2)
    nhit, nrun, t0, ok = 0, 0, time.time(), True
    for u, d in decs:
        base = tuple(Mt.index_of(ui) for ui in u)
        for x0 in PTS:
            h, st = Mt.run(base, x0, target)
            nrun += 1
            if st["refused"] or not h:
                print(f"  product decomposition not found at {x0}: {st}")
                ok = False
            nhit += len(h)
    print(f"control-product ({orbit}): {nrun} runs over {len(decs)} stored rank-3 decompositions x 9 base "
          f"points, {nhit} three-term decompositions of phi (x) |M>^2 found [{time.time() - t0:.0f}s]")
    write_control(orbit, "control_product", {"runs": nrun, "hits": nhit, "stored": len(decs), "pass": ok})
    print("control-product:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def control_m3(args):
    """Control (c): the rank-4 decompositions of |M>^3 by 2 + 1 slicing. Every
    4-multiset of single-qutrit stabilizer states whose span contains |M>
    with a full family is matched at every base point against |M>^3; every
    genuine rank-4 hit must be G_3-equivalent to a stored decomposition, and
    every stored decomposition must be recovered."""
    orbit = args.orbit
    m, n2, rank = 3, 1, 4
    E1q = CoverEnumerator3(orbit, n2)
    Mt = new_matcher(n2)
    target = psi_target(orbit, n2, Mt.F1, Mt.F2)
    t0 = time.time()
    bases = []
    for ms in itertools.combinations_with_replacement(range(E1q.N), rank):
        distinct = sorted(set(ms))
        if len(distinct) < 2:
            continue
        fam = Family.from_cover(E1q, distinct)
        if fam is None:
            continue
        exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
        if not fam.has_zero_coefficient(exempt):
            bases.append(tuple(ms))
    print(f"{len(bases)} full 4-multisets of the {E1q.N} single-qutrit states over |{orbit}> "
          f"[{time.time() - t0:.0f}s]")
    D3 = dictionary(3, m)
    M3 = Matcher(D3, m, Mt.F1, Mt.F2)          # for the dictionary lookup of 3-qutrit states
    _, info3 = symmetry_orbit_reps(orbit, m, D3, antiunitary=False)
    group = group_closure(info3, D3.shape[1])
    cc = constructions_common()
    stored, _ = cc.load_decompositions(orbit, m, rank)
    stored_keys = {canonical([M3.index_of(t) for t in terms], group): terms for terms, _ in stored}
    # the stored decompositions' own all-visible (base, x0) pairs, which
    # must all recover their class; then a sample of the enumerated
    # multisets at every base point, which must produce nothing outside
    # the stored classes (--sample 0 runs them all, about an hour)
    own = {}
    expected = set()
    for terms, _ in stored:
        key = canonical([M3.index_of(t) for t in terms], group)
        for S in itertools.combinations(range(m), 2):
            moved = np.column_stack([move_front(t, S, m) for t in terms])
            for x0 in PTS:
                b = slice_base(Mt, moved, x0)
                if b is not None:
                    assert tuple(sorted(b)) in set(bases), "a stored decomposition's base is not enumerated"
                    own.setdefault((tuple(sorted(b)), x0), set()).add(key)
                    expected.add(key)
    runs = sorted(own)
    rng = np.random.default_rng(7)
    pool = [(c, x0) for c in bases for x0 in PTS if (c, x0) not in own]
    if args.sample and args.sample < len(pool):
        pool = [pool[k] for k in sorted(rng.choice(len(pool), size=args.sample, replace=False))]
    runs += pool
    print(f"{len(own)} (base, x0) pairs of the stored decompositions, {len(pool)} further pairs sampled "
          f"from the {len(bases) * len(PTS)} enumerated (multiset, base point) pairs")
    recovered, stats = {}, {"matched": 0, "refused": 0, "hits": 0, "non_genuine": 0}
    own_missed = []
    t1 = time.time()
    for cover, x0 in runs:
        if True:
            hits, st = Mt.run(cover, x0, target)
            if (cover, x0) in own:
                got = {canonical([M3.index_of(t) for t in h["terms"]], group) for h in hits if genuine(h, rank)}
                if not own[(cover, x0)] <= got:
                    own_missed.append([list(cover), list(x0)])
            if st["refused"]:
                stats["refused"] += 1
                continue
            stats["matched"] += 1
            for h in hits:
                if not genuine(h, rank):
                    stats["non_genuine"] += 1
                    continue
                stats["hits"] += 1
                key = canonical([M3.index_of(t) for t in h["terms"]], group)
                recovered.setdefault(key, []).append([list(cover), list(x0)])
    dt = time.time() - t1
    unknown = set(recovered) - set(stored_keys)
    missing = expected - set(recovered)
    print(f"matched {stats['matched']} (cover, x0) pairs in {dt:.0f}s; {stats['hits']} genuine hits, "
          f"{len(recovered)} G_3 classes of rank-{rank} decompositions of |{orbit}>^{m}; stored "
          f"{len(stored_keys)}, expected recoverable {len(expected)}; not stored {len(unknown)}, "
          f"missing {len(missing)}; stats {stats}")
    ok = not unknown and not missing and not own_missed and len(expected) == len(stored_keys)
    print(f"stored decompositions' own bases: {len(own)} pairs, {len(own_missed)} did not recover their class")
    write_control(orbit, "control_m3", {"bases": len(bases), "runs": len(runs), "own_pairs": len(own),
                                        "sampled_pairs": len(pool), "stored": len(stored_keys),
                                        "expected": len(expected), "recovered": len(recovered),
                                        "unknown": len(unknown), "missing": len(missing),
                                        "own_missed": own_missed, "seconds": dt, "stats": stats,
                                        "recovered_from": {str(k): v for k, v in recovered.items()},
                                        "pass": ok})
    print("control-m3:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def norrell_rank7_terms():
    """The seven terms of lean_proofs/LeanProofs/NorrellM4Pointwise.lean as
    complex vectors on four qutrits (index y_0 y_1 y_2 y_3, y_0 the most
    significant digit), unnormalised."""
    def two(y):
        return 1 if y == 2 else 0
    terms = []
    specs = [
        (lambda y: y[1] == 2, lambda y: 2 * two(y[0]) + two(y[2]) + 2 * two(y[3])),
        (lambda y: True, lambda y: 2 * two(y[0]) + 2 * two(y[1]) + two(y[2]) + two(y[3])),
        (lambda y: y[0] == 2 and y[3] == 2, lambda y: two(y[1]) + two(y[2])),
        (lambda y: y[2] == 2 and y[3] == 2, lambda y: 2 * two(y[0]) + two(y[1])),
        (lambda y: y[0] == 2, lambda y: 2 * two(y[1]) + 2 * two(y[2]) + two(y[3])),
        (lambda y: y[1] == 2 and y[2] == 2, lambda y: two(y[0]) + 2 * two(y[3])),
        (lambda y: True, lambda y: 0),
    ]
    for supp, q in specs:
        v = np.zeros(81, dtype=complex)
        for idx, y in enumerate(itertools.product(range(3), repeat=4)):
            if supp(y):
                v[idx] = W3 ** (q(y) % 3)
        terms.append(v)
    return terms


def export_witness(args):
    """Write N_m4_rank7_witness.json: the seven Lean terms as phase codes,
    checked to reproduce |N>^4 with rank 7."""
    cc = constructions_common()
    terms = norrell_rank7_terms()
    psi = cc.target("N", 4)
    A = np.column_stack(terms)
    c, *_ = np.linalg.lstsq(A, psi, rcond=None)
    res = float(np.linalg.norm(A @ c - psi))
    rank = int(np.linalg.matrix_rank(A, tol=1e-8))
    if res > 1e-9 or rank != 7:
        raise AssertionError(f"the Lean terms do not reproduce |N>^4: residual {res:.2e}, rank {rank}")
    D4 = dictionary(3, 4)
    M4 = Matcher(D4, 4, Field3(P1), Field3(P2))
    for t in terms:
        M4.index_of(t)                                   # raises unless a dictionary state
    doc = {"orbit": "N", "m": 4, "rank": 7, "source": "lean_proofs/LeanProofs/NorrellM4Pointwise.lean "
           "(norrell_m4_decomposition), rebuilt numerically by driver.norrell_rank7_terms",
           "index": "y_0 y_1 y_2 y_3 with y_0 the most significant digit",
           "codes": "0 zero, 1..3 = 1, w, w^2 with the first nonzero entry made 1",
           "terms": [exact_codes(t)[0].astype(int).tolist() for t in terms],
           "coeffs": [[float(z.real), float(z.imag)] for z in c], "residual": res}
    with open(WITNESS_N, "w") as f:
        json.dump(doc, f)
        f.write("\n")
    print(f"wrote {os.path.relpath(WITNESS_N, ROOT)}: 7 stabilizer terms, residual {res:.1e}, rank {rank}")
    return 0


def witness_terms(orbit):
    cc = constructions_common()
    if orbit == "N":
        if not os.path.exists(WITNESS_N):
            export_witness(None)
        from matcher import term_from_codes
        with open(WITNESS_N) as f:
            doc = json.load(f)
        return [term_from_codes(c) for c in doc["terms"]], doc["rank"], os.path.relpath(WITNESS_N, ROOT)
    with open(WITNESS_H3) as f:
        w = json.load(f)
    terms = [cc.term_vector(t, 3, M) for t in w["witness"]["terms"]]
    return terms, int(w["rank"]), os.path.relpath(WITNESS_H3, ROOT)


def control_witness(args):
    """Control (d): the rank-7 N witness or the rank-8 H3 witness recovered
    from its own all-visible (qutrit pair, base point) slices, the matcher
    running at rank 7 or 8 through the family and block machinery."""
    orbit = args.orbit
    terms, rank, source = witness_terms(orbit)
    cc = constructions_common()
    psi = cc.target(orbit, M)
    A = np.column_stack(terms)
    c, *_ = np.linalg.lstsq(A, psi, rcond=None)
    assert np.linalg.norm(A @ c - psi) < 1e-9 and np.linalg.matrix_rank(A, tol=1e-8) == rank
    Mt = new_matcher(verbose=args.verbose)
    target = psi_target(orbit, N2, Mt.F1, Mt.F2)
    bases = {}
    for S in itertools.combinations(range(M), 2):
        moved = np.column_stack([move_front(t, S, M) for t in terms])
        for x0 in PTS:
            b = slice_base(Mt, moved, x0)
            if b is not None:
                bases.setdefault((tuple(sorted(b)), x0), []).append((S, b))
    print(f"{orbit} rank-{rank} witness ({source}): {len(bases)} all-visible (base, x0) pairs over the 6 "
          f"qutrit pairs and 9 points")
    report = {"witness": source, "rank": rank, "bases": []}
    passed, selected = 0, -1
    for (cover, x0), Ss in sorted(bases.items()):
        selected += 1
        if args.base is not None and selected != args.base:
            continue
        repeated = len(set(cover)) < len(cover)
        t0 = time.time()
        hits, st = Mt.run(cover, x0, target)
        good = [h for h in hits if genuine(h, rank)]
        S, b = Ss[0]
        wkey = sorted(exact_codes(move_front(v, S, M))[0].tobytes() for v in terms)
        same = any(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == wkey for h in good)
        dt = time.time() - t0
        print(f"base {cover} x0 {x0} ({len(Ss)} pairs, {'repeated' if repeated else 'distinct'}, "
              f"kappa {st['kappa']}): {len(good)} rank-{rank} decompositions, witness itself "
              f"{'recovered' if same else 'not among them'}, {dt:.0f}s, stats {st}", flush=True)
        report["bases"].append({"cover": list(cover), "x0": list(x0), "pairs": [list(s) for s, _ in Ss],
                                "repeated": repeated, "hits": len(good), "witness_recovered": same,
                                "seconds": dt, "stats": st})
        passed += same
        suffix = "" if args.base is None else f"_{args.base}"
        report["pass"] = passed == len(report["bases"]) and bool(report["bases"])
        report["complete"] = selected == len(bases) - 1 or args.base is not None
        write_control(orbit, f"control_witness{suffix}", report)     # after every base: an outer cap may kill the run
    print(f"control-witness: {passed}/{len(report['bases'])} bases recover the rank-{rank} witness")
    return 0 if report["pass"] else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, orbit=True):
        p = sub.add_parser(name)
        if orbit:
            p.add_argument("orbit", choices=common.ORBITS)
        p.add_argument("--verbose", action="store_true")
        p.set_defaults(fn=fn)
        return p

    add("census", census)
    p = add("degenerate", degenerate)
    p.add_argument("--sample", type=int, default=0, help="time K covers of each multiplicity pattern")
    p.add_argument("--cap-s", type=float, default=0.0, help="wall-clock cap for the sampling, split "
                   "evenly over the patterns (0: none)")
    p.add_argument("--write", action="store_true", help="store the list with its hash")
    p = add("partition", partition)
    p.add_argument("--target-s", type=float, default=700.0, help="seconds per stage A batch")
    p.add_argument("--target-bc-s", type=float, default=700.0, help="seconds per stage B or C batch")
    p.add_argument("--match-ms", type=float, default=70.0, help="matcher milliseconds per stage A cover")
    p = add("sample", sample)
    p.add_argument("--count", type=int, default=200)
    add("control-covers", control_covers)
    p = add("control-planted", control_planted)
    p.add_argument("--count", type=int, default=8, help="planted instances per case")
    add("control-product", control_product)
    p = add("control-m3", control_m3)
    p.add_argument("--sample", type=int, default=300,
                   help="further (multiset, base point) pairs to run beyond the stored decompositions' own "
                        "(0: all, about an hour)")
    p = add("control-witness", control_witness)
    p.add_argument("--base", type=int, default=None, help="run only the k-th (base, x0) pair (0-based)")
    add("export-witness", export_witness, orbit=False)
    args = ap.parse_args(argv[1:])
    common.lower_priority()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
