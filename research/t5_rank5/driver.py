"""Lists, rates, partition and controls for the rank-5 exclusion of |T>^5 by
a two-qubit base slice at the single base point 00
(docs/notes/t5_rank5_exclusion.md). The batches run through batch.py and
are checked by aggregate.py.

Every rank-5 decomposition of |T>^5 has, along qubits 1, 2 at 00, three,
four or five visible terms (Fact 2, chi(T^3) = 3), so its base is a full
k-cover of |T>^3 with k in {3, 4, 5} and the 5 - k invisible terms lie on
flats of F_2^2 missing 00. Stage (alpha), k = 5: the kernel census of the
6,115,136 full 5-covers of distinct independent states (stage A, regenerated
per pivot pair) and the 43,773 dependent or repeated 5-multisets of
degenerate5.json (stages B and C, the cancel-at-base multisets included),
through slice_cover.SliceMatcher. Stage (beta'), k = 4: the 4,709 full
4-covers of research/t5_rank4/covers4.json, each with the invisible term on
every one of the six flats, through invisible.InvisibleMatcher.run_one.
Stage (gamma), k = 3: the 4 full 3-covers with every one of the 21 flat
pairs, through InvisibleMatcher.run_two.

Commands
  degenerate [--write]      enumerate the dependent and repeated full
                            5-multisets (stages B and C) and, with --write,
                            store them as degenerate5.json with their hash
  control-census            re-enumerate the 3-covers, 4-covers and degenerate
                            multisets from scratch and compare them by hash with
                            covers4.json and degenerate5.json; check the census
                            file's units against the enumerator's pivot pairs
  sample STAGE [--count N] [--cap S]
                            time the matcher on N covers of a stage (A, B, C,
                            beta or gamma); the rates go to results/rates.json
  partition [--target-s S] [--pod-factor F]
                            write partition.json: stage A pivot pairs grouped
                            into batches of about S pod seconds, stages B, C,
                            beta and gamma round-robin
  control-witness [--all-x0] [--max-cand N] [--reference]
                            recover the rank-6 witness bounds/qubit_T-m5-upper-6.json
                            from each of its all-visible bases at 00 (every
                            base point with --all-x0) through stage (alpha)
  control-planted [--stage beta|gamma|all] [--plant K] [--seed S]
                            recover planted instances: stage (beta') at every
                            one of the six flats, stage (gamma) at every one of
                            the 21 flat pairs, with the repeated-base, ambiguous,
                            same-state and cancelling-pair variants
  control-m4-pair [--all-x0]
                            recover the rank-3 decomposition of |T>^4 from the
                            full 3-covers of |T>^2 along a qubit pair
  tables                    check the stored residual tables the case split
                            leans on (results/rank2_full.json and the rank-4
                            record's tables.json)

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
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (research/t5_rank5/common.py)
from common import (CENSUS, COVERS4, DEGENERATE, FLAT_NAMES, FLAT_PAIRS, FLATS, HERE, M, N1, N2, ORBIT,  # noqa: E402
                    PARTITION, RANK, RATES, RESULTS, ROOT, RUNS_PER_COVER, X0, X0S, codes_key, genuine, git_commit,
                    kind_of, make_enumerator, pairs_of, stage_of)
from invisible import InvisibleMatcher, PlantedInvisibleMatcher, to_field  # noqa: E402
from slice_cover import (CoverEnumerator, Family, SliceMatcher, UnpinnedFamily, _canon_rows, _groups_by_key,  # noqa: E402
                         _reduce, apply_pauli, exact_codes, pauli_reps, patterns)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise Deadline()


# ------------------------------------------------------------- lists -------

def degenerate_covers(E, covers3, covers4):
    """Full 5-multisets of psi_3 whose distinct states are dependent or which
    repeat a state, as sorted tuples: over every full 3-cover T the multisets
    T + (x, y) with x, y in T or span(T), the pairs (x, y) parallel modulo T
    (y in span(T, x)), and the cancel-at-base multisets T + (b, b) with b
    outside T and span(T); over every full 4-cover C the multisets C + (x,)
    with x in C or span(C). Every multiset must have a coefficient family in
    which no unrepeated state is dead. The routes of the H^6 v2 list
    (docs/notes/h6_rank5_stagec_repair.md) and of the H^5 pipeline; the
    completeness argument is section 3 of the exclusion note."""
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
        for extra in itertools.combinations_with_replacement(pool, 2):
            ms = tuple(sorted(T + extra))
            if ok(ms):
                out.add(ms)
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
        in_pool = set(pool)
        for b in range(E.N):
            if b in in_pool:
                continue
            ms = tuple(sorted(T + (b, b)))
            if ok(ms):
                out.add(ms)
    for Cv in covers4:
        for x in list(Cv) + E.in_span(Cv):
            ms = tuple(sorted(Cv + (x,)))
            if ok(ms):
                out.add(ms)
    return sorted(out)


def stored_lists():
    """(covers3, covers4 of kind A, the 12 repeated 4-multisets, doc) from the
    rank-4 record's census."""
    covers, doc = common.load_covers4()
    covers3 = common.covers3_of(doc)
    ind = [c for c, k in zip(covers, doc["kinds"]) if k == "A"]
    rep = [c for c, k in zip(covers, doc["kinds"]) if k == "C"]
    return covers3, ind, rep, doc


def degenerate(args):
    E = make_enumerator()
    t0 = time.time()
    covers3, covers4, _, _ = stored_lists()
    lst = degenerate_covers(E, covers3, covers4)
    dt = time.time() - t0
    by, kappa = {}, {}
    for ms in lst:
        pat = str(common.multiplicity_pattern(ms))
        by[pat] = by.get(pat, 0) + 1
        fam = Family.from_cover(E, sorted(set(ms)))
        key = f"{len(set(ms))} distinct, kappa {fam.kappa}"
        kappa[key] = kappa.get(key, 0) + 1
    n_b = sum(stage_of(c) == "B" for c in lst)
    print(f"{len(lst)} degenerate 5-multisets over {len(covers3)} full 3-covers and {len(covers4)} full 4-covers: "
          f"by pattern {by}, by family {kappa}; stage B {n_b}, stage C {len(lst) - n_b} [{dt:.0f}s]")
    probe = os.path.join(RESULTS, "degenerate5.json")
    if os.path.exists(probe):
        with open(probe) as f:
            old = [tuple(int(x) for x in c) for c in json.load(f)["covers"]]
        print(f"the feasibility probe's list ({len(old)}): {'equal' if old == lst else 'DIFFERENT'}")
    if args.write:
        rec = {"orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
               "covers3": [list(T) for T in covers3], "covers4_sha256": common.load_covers4()[1]["sha256"],
               "count": len(lst), "stage_b": n_b, "stage_c": len(lst) - n_b, "by_pattern": by,
               "by_distinct_kappa": kappa, "seconds": dt, "git": git_commit(), "generated": _now(),
               "covers": [list(c) for c in lst]}
        sha = common.write_hashed(DEGENERATE, rec)
        print(f"wrote {os.path.relpath(DEGENERATE, ROOT)} (sha256 {sha[:16]})")
    return 0


def fresh_lists(E):
    """(covers3, covers4, repeated 4-multisets, degenerate 5-multisets) from
    scratch: the enumerator's 3-covers and 4-covers, the multisets T + (x,)
    with x in T over the 3-covers (the only degenerate 4-multisets, section
    3 of the note), and degenerate_covers."""
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    rep4 = set()
    for T in covers3:
        for x in list(T) + E.in_span(T):
            ms = tuple(sorted(T + (x,)))
            distinct = sorted(set(ms))
            fam = Family.from_cover(E, distinct)
            exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
            if fam is not None and not fam.has_zero_coefficient(exempt):
                rep4.add(ms)
    deg5 = degenerate_covers(E, covers3, covers4)
    return covers3, covers4, sorted(rep4), deg5


def control_census(args):
    """Fresh enumerations of every list against the stored ones, and the
    census file's pivot pairs against the enumerator's."""
    E = make_enumerator()
    t0 = time.time()
    covers3, covers4, rep4, deg5 = fresh_lists(E)
    dt = time.time() - t0
    s3, s4, srep, doc4 = stored_lists()
    stored_bases = [tuple(int(x) for x in c) for c in doc4["covers"]]
    fresh_bases = sorted(set(covers4) | set(rep4))
    fresh_kinds = [kind_of(E, c) for c in fresh_bases]
    ok4 = fresh_bases == stored_bases and fresh_kinds == doc4["kinds"] and covers3 == s3
    deg_stored, ddoc = common.load_degenerate()
    okd = deg5 == deg_stored and ddoc["covers4_sha256"] == doc4["sha256"]
    cen = common.load_census()
    units = pairs_of(E)
    okc = [[r[0], r[1]] for r in cen["rows"]] == [[i, j] for i, j, _ in units] and cen["units"] == len(units) \
        and cen["covers"] == sum(r[3] for r in cen["rows"])
    print(f"fresh: {len(covers3)} full 3-covers, {len(covers4)} full 4-covers of distinct independent states, "
          f"{len(rep4)} repeated 4-multisets, {len(deg5)} degenerate 5-multisets [{dt:.0f}s]")
    print(f"4-cover census {os.path.relpath(COVERS4, ROOT)} (sha256 {doc4['sha256'][:16]}): "
          f"{'equal' if ok4 else 'DIFFERENT'}; degenerate list (sha256 {ddoc['sha256'][:16]}): "
          f"{'equal' if okd else 'DIFFERENT'}; census {os.path.relpath(CENSUS, ROOT)}: {len(cen['rows'])} rows, "
          f"{cen['covers']} covers, pivot pairs {'equal to' if okc else 'DIFFERENT from'} the enumerator's "
          f"{len(units)}")
    ok = ok4 and okd and okc
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_census.json"), "w") as f:
        json.dump({"git": git_commit(), "generated": _now(), "covers3": len(covers3), "covers4": len(covers4),
                   "repeated4": len(rep4), "degenerate5": len(deg5), "covers4_sha256": doc4["sha256"],
                   "degenerate_sha256": ddoc["sha256"], "census_sha256": common.sha256_file(CENSUS),
                   "census_covers": cen["covers"], "units": len(units), "seconds": dt,
                   "covers4_equal": ok4, "degenerate_equal": okd, "census_units_equal": okc, "pass": ok}, f, indent=1)
    print("control-census:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------- rates -------

def census_covers(E, count):
    """The first `count` full 5-covers of the kernel census, pivot pair by
    pivot pair."""
    covers = []
    for i, j, _ in pairs_of(E):
        Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
        members, _ = E.pivot_plan(i)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        got, _ = E.pair_covers(5, i, j, Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= count:
            break
    return covers[:count]


def _spread(lst, n):
    n = min(n, len(lst))
    if n == 0:
        return []
    return [lst[int(round(k * (len(lst) - 1) / max(n - 1, 1)))] for k in range(n)]


def _load_rates():
    if os.path.exists(RATES):
        with open(RATES) as f:
            return json.load(f)
    return {"git": git_commit(), "stages": {}}


def run_units(stage, cover, pairs=None):
    """The runs of one cover: (x0,) for the stage (alpha) kinds, the flats
    for stage beta, the flat pairs (or the selection `pairs`) for stage
    gamma."""
    if stage == "beta":
        return list(FLAT_NAMES)
    if stage == "gamma":
        return list(FLAT_PAIRS) if pairs is None else [p for p in FLAT_PAIRS if "+".join(p) in pairs]
    return list(X0S)


def run_matcher(stage, matcher, cover, unit):
    if stage == "beta":
        return matcher.run_one(cover, unit)
    if stage == "gamma":
        return matcher.run_two(cover, unit)
    return matcher.run(cover, unit)


def hist_key(stage, st):
    if stage == "beta":
        return f"{'/'.join(str(v) for v in st['exact_solutions'])},{st['scan_solutions']},{st['second_solutions']}"
    if stage == "gamma":
        return f"{'/'.join(str(v) for v in st['exact_solutions'])},{st['scan_solutions']},{st['scan2_solutions']}," \
               f"{st['pair_solutions']}"
    return ",".join(str(b) for b in st["coord_solutions"])


def sample(args):
    """Time the matcher of one stage on covers spread through its list, every
    run of the cover (base point, flat or flat pair), one run at a time
    under a per-run cap."""
    E = make_enumerator()
    stage = args.stage
    rec = {"stage": stage, "count": 0, "runs": 0, "seconds": [], "by_pattern": {}, "by_unit": {}, "hits": 0,
           "refused": 0, "undecided": 0, "capped": 0, "hist": {}, "generated": _now(), "git": git_commit()}
    if stage == "A":
        Mt = SliceMatcher(E, N1, native=not args.reference)
        covers = census_covers(E, args.count)
        rec["matcher"] = "native" if Mt.native is not None else "reference"
    elif stage in ("B", "C"):
        Mt = SliceMatcher(E, N1, native=False)
        all_covers, _ = common.load_degenerate()
        covers = [c for c in all_covers if stage_of(c) == stage]
        if stage == "C":
            by = {}
            for c in covers:
                by.setdefault(common.multiplicity_pattern(c), []).append(c)
            covers = []
            for pat, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
                covers += _spread(lst, args.count if len(lst) > 12 else len(lst))
        else:
            covers = _spread(covers, args.count)
        rec["matcher"] = "reference"
    elif stage == "beta":
        Mt = InvisibleMatcher(E)
        _, ind, rep, _ = stored_lists()
        covers = _spread(ind, args.count) + rep
        rec["matcher"] = "invisible"
    else:
        Mt = InvisibleMatcher(E)
        covers3, _, _, _ = stored_lists()
        covers = covers3[args.first:args.first + args.count]
        rec["matcher"] = "invisible"
    signal.signal(signal.SIGALRM, _alarm)
    t_all = time.time()
    for cover in covers:
        pat = str(common.multiplicity_pattern(cover))
        t_cover = 0.0
        for unit in run_units(stage, cover, set(args.pairs.split(",")) if args.pairs else None):
            t1 = time.time()
            signal.alarm(args.cap)
            try:
                hits, st = run_matcher(stage, Mt, cover, unit)
                if st["refused"]:
                    rec["refused"] += 1
                else:
                    rec["hits"] += sum(genuine(h, RANK) for h in hits)
                    key = hist_key(stage, st)
                    rec["hist"][key] = rec["hist"].get(key, 0) + 1
            except Deadline:
                rec["capped"] += 1
                print(f"  {cover} {unit}: capped at {args.cap}s", flush=True)
            except UnpinnedFamily as exc:
                rec["undecided"] += 1
                print(f"  {cover} {unit}: UnpinnedFamily: {exc}", flush=True)
            finally:
                signal.alarm(0)
            dt = time.time() - t1
            t_cover += dt
            rec["by_unit"].setdefault(str(unit), []).append(dt)
            rec["runs"] += 1
        rec["seconds"].append(t_cover)
        rec["by_pattern"].setdefault(pat, []).append(t_cover)
        rec["count"] += 1
    a = np.array(rec["seconds"])
    rec["per_cover_mean_s"] = float(a.mean())
    rec["per_cover_median_s"] = float(np.median(a))
    rec["per_cover_max_s"] = float(a.max())
    rec["per_pattern_mean_s"] = {p: float(np.mean(v)) for p, v in rec["by_pattern"].items()}
    rec["per_unit_mean_s"] = {u: float(np.mean(v)) for u, v in rec["by_unit"].items()}
    rec["per_unit_max_s"] = {u: float(np.max(v)) for u, v in rec["by_unit"].items()}
    rec["wall_s"] = time.time() - t_all
    rec["covers"] = [list(c) for c in covers]
    print(f"stage {stage} ({rec['matcher']}): {rec['count']} covers, {rec['runs']} runs: per cover mean "
          f"{a.mean():.4f}s, median {np.median(a):.4f}s, max {a.max():.3f}s; by pattern "
          f"{ {p: round(v, 4) for p, v in rec['per_pattern_mean_s'].items()} }; per unit "
          f"{ {u: round(v, 4) for u, v in rec['per_unit_mean_s'].items()} }; hits {rec['hits']}, refused "
          f"{rec['refused']}, undecided {rec['undecided']}, capped {rec['capped']}; histogram "
          f"{dict(sorted(rec['hist'].items(), key=lambda kv: -kv[1])[:10])}")
    os.makedirs(RESULTS, exist_ok=True)
    rates = _load_rates()
    rates["stages"][stage] = {k: v for k, v in rec.items() if k not in ("seconds", "by_pattern", "by_unit", "covers")}
    rates["git"] = git_commit()
    with open(RATES, "w") as f:
        json.dump(rates, f, indent=1)
    with open(os.path.join(RESULTS, f"sample_{stage}.json"), "w") as f:
        json.dump(rec, f)
    return 0


# ---------------------------------------------------------- partition ------

def partition(args):
    E = make_enumerator()
    units = pairs_of(E)
    cen = common.load_census()
    rates = _load_rates()["stages"]
    for st in ("A", "B", "C", "beta", "gamma"):
        if st not in rates:
            raise SystemExit(f"no rate for stage {st} in {RATES}; run `driver.py sample {st}` first")
    F = args.pod_factor
    by = {(r[0], r[1]): r for r in cen["rows"]}
    if len(by) != len(units) or any((i, j) not in by for i, j, _ in units):
        raise SystemExit("the census rows are not the enumerator's pivot pairs")
    a_rate = rates["A"]["per_cover_mean_s"]
    costs = [F * (by[(i, j)][5] + a_rate * by[(i, j)][3]) for i, j, _ in units]
    batches_a, cur, acc = [], [], 0.0
    for u, c in zip(units, costs):
        cur.append(list(u))
        acc += c
        if acc >= args.target_s:
            batches_a.append(cur)
            cur, acc = [], 0.0
    if cur:
        batches_a.append(cur)
    geometry = [{"index": k, "stage": "A", "units": b} for k, b in enumerate(batches_a)]
    est = {"A": float(sum(costs))}
    covers, deg = common.load_degenerate()
    covers4, doc4 = common.load_covers4()
    covers3 = common.covers3_of(doc4)
    pat_rate = {}
    for st in ("B", "C", "beta", "gamma"):
        pat_rate[st] = {p: v * F for p, v in rates[st]["per_pattern_mean_s"].items()}
    ids = {"B": [], "C": []}
    cost = {"B": 0.0, "C": 0.0, "beta": 0.0, "gamma": 0.0}
    for k, c in enumerate(covers):
        st = stage_of(c)
        ids[st].append(k)
        cost[st] += pat_rate[st][str(common.multiplicity_pattern(c))]
    for c in covers4:
        cost["beta"] += pat_rate["beta"][str(common.multiplicity_pattern(c))]
    for c in covers3:
        cost["gamma"] += pat_rate["gamma"][str(common.multiplicity_pattern(c))]
    ids["beta"] = list(range(len(covers4)))
    ids["gamma"] = list(range(len(covers3)))
    counts = {}
    for st in ("B", "C", "beta", "gamma"):
        n = max(1, math.ceil(cost[st] / args.target_s))
        if st in ("B", "C"):
            n = max(n, math.ceil(len(ids[st]) / args.max_covers))
        for b in range(n):
            geometry.append({"index": len(geometry), "stage": st, "cover_ids": ids[st][b::n]})
        counts[st] = n
        est[st] = cost[st]
    est["total"] = sum(v for k, v in est.items())
    rec = {"orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "x0": list(X0S), "flats": {k: list(v) for k, v in FLATS.items()}, "flat_pairs": [list(p) for p in FLAT_PAIRS],
           "runs_per_cover": RUNS_PER_COVER, "pivot_orbits": int(E.info["orbits"]), "units": len(units),
           "stage_a_batches": len(batches_a), "stage_b_batches": counts["B"], "stage_c_batches": counts["C"],
           "stage_beta_batches": counts["beta"], "stage_gamma_batches": counts["gamma"],
           "cost_model": {"pod_factor": F, "target_s": args.target_s, "rates": os.path.relpath(RATES, HERE),
                          "a_s_per_cover": a_rate, "s_per_cover_by_pattern": pat_rate},
           "estimated_s": est,
           "census": {"file": os.path.relpath(CENSUS, HERE), "sha256": common.sha256_file(CENSUS),
                      "covers": cen["covers"], "units": cen["units"]},
           "degenerate": {"file": os.path.relpath(DEGENERATE, HERE), "sha256": deg["sha256"], "count": len(covers),
                          "stage_b_covers": len(ids["B"]), "stage_c_covers": len(ids["C"])},
           "covers4": {"file": os.path.relpath(COVERS4, HERE), "sha256": doc4["sha256"], "count": len(covers4),
                       "independent": doc4["independent"], "repeated": doc4["repeated"]},
           "covers3": {"count": len(covers3), "covers": [list(T) for T in covers3]},
           "batches": len(geometry), "git": git_commit(), "generated": _now(), "batch_geometry": geometry}
    sha = common.write_hashed(PARTITION, rec)
    print(f"{len(units)} pivot pairs in {len(batches_a)} stage A batches ({est['A'] / 3600:.2f} pod CPU-h), "
          f"{counts['B']} stage B ({est['B'] / 3600:.2f}), {counts['C']} stage C ({est['C'] / 3600:.2f}), "
          f"{counts['beta']} stage beta ({est['beta'] / 3600:.2f}), {counts['gamma']} stage gamma "
          f"({est['gamma'] / 3600:.3f}); {len(geometry)} batches, {est['total'] / 3600:.1f} pod CPU-h at {F} x the "
          f"laptop rates; wrote {PARTITION} (sha256 {sha[:16]})")
    return 0


# ------------------------------------------------------------ controls -----

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
    of |T>^4 with |0> and |1> on qubit 5) sliced along every qubit pair: at
    every (pair, x0) where all six terms are nonzero the base is a 6-cover
    of psi_3, and the stage (alpha) matcher must return the witness itself
    among its rank-6 decompositions. Every base is reported; a base that
    raises or exceeds the candidate cap is recorded as aborted and fails
    the control, never counted as passed. By default the bases at 00, the
    pipeline's base point; --all-x0 runs every base point for the record."""
    m, n1, rank = M, N1, 6
    E = make_enumerator()
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    w = json.load(open(os.path.join(ROOT, args.path)))
    terms = [common.term_vector(t, 2, m) for t in w["witness"]["terms"]]
    psi = common.target(ORBIT, m)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    assert np.linalg.norm(np.column_stack(terms) @ c - psi) < 1e-9
    x0s = list(range(1 << n1)) if args.all_x0 else list(X0S)
    bases, types = {}, {}
    for S in itertools.combinations(range(m), n1):
        types[S] = [flat_type(v, S, m, n1) for v in terms]
        for x0 in x0s:
            b = slice_terms(terms, S, x0, m, n1, lookup)
            if b is not None:
                bases.setdefault((b, x0), []).append(S)
    items = sorted(bases.items())
    print(f"{len(items)} distinct all-visible (base, x0) pairs at x0 in {[f'{x:02b}' for x in x0s]} over the "
          f"{len(types)} qubit pairs; flat types: "
          + "; ".join(f"{S}: " + ",".join({4: 'P', 2: 'L' + ''.join(str(p) for p in t), 1: 'pt'}[len(t)]
                                             for t in ts) for S, ts in types.items()))
    Mt = SliceMatcher(E, n1, verbose=args.verbose, max_cand=args.max_cand, native=not args.reference)
    report = {"witness": args.path, "git": git_commit(), "generated": _now(), "max_cand": args.max_cand,
              "x0": x0s, "matcher": "reference" if Mt.native is None else "native", "bases": []}
    passed = aborted = 0
    for k, ((cover, x0), Ss) in enumerate(items):
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
    name = "control_witness" + ("_reference" if args.reference else "") + ("_all_x0" if args.all_x0 else "") + \
        (f"_cap{args.max_cand}" if args.max_cand != 2_000_000 else "") + ".json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(report, f, indent=1)
    ok = passed == len(report["bases"]) and aborted == 0
    print(f"control-witness ({report['matcher']}, cap {args.max_cand}): {passed}/{len(report['bases'])} bases "
          f"recover the witness, {aborted} aborted; {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# -- planted instances --

PLANE = (0, 1, 2, 3)
VISIBLE_LINES = [(0, 3), (0, 1), (0, 2)]          # A, C0, D0: the lines through 00


def term_over(rng, u, flat):
    """A five-qubit stabilizer state (entries in {0, +-1, +-i}) with the
    slice u at the first point of `flat` (a tuple of points) and, by the
    structure lemma, random phased Pauli translates of u at the others."""
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


def random_term(rng, E, flat):
    return term_over(rng, E.C[:, rng.integers(E.N)], flat)


def random_visible(rng, E, line_prob=0.3):
    """A visible term: a plane, or with probability line_prob a line
    through 00."""
    if rng.random() < line_prob:
        return random_term(rng, E, VISIBLE_LINES[rng.integers(3)])
    return random_term(rng, E, PLANE)


def translate_of(rng, u, phase=True):
    """A random phased Pauli translate of u (a phased translate with a
    random class; the identity class is allowed)."""
    reps, _ = pauli_reps(u, N2)
    a, c = reps[rng.integers(8)]
    return (1j) ** (rng.integers(4) if phase else 0) * apply_pauli(u, a, c, N2)


def _gauss_coeffs(rng, n):
    return np.array([complex(rng.integers(1, 4)) * (1j) ** rng.integers(4) for _ in range(n)])


def _planted_enum(E, Psi):
    rows = Psi.reshape(4, 8)
    E2 = copy.copy(E)
    E2.psi = rows[X0]
    E2.psi1, E2.psi2 = to_field(E.F1, rows[X0]), to_field(E.F2, rows[X0])
    return E2


def plant_beta(rng, E, lookup, flat, kind):
    """A planted stage (beta') instance at the flat `flat`: (a) four visible
    terms, planes or lines through 00; (b) three planes and a line through
    00; (c) a repeated base state (two plane copies with one base slice)
    and two visible terms; (e) a repeated base state whose copies share a
    Pauli class with the invisible term at the first point of its flat
    (the residual there lies in the span of the block's translates, the
    ambiguous case decided by the joint reconstruction). Returns (terms,
    coeffs, base) with the invisible term last."""
    F = FLATS[flat]
    while True:
        if kind == "a":
            vis = [random_visible(rng, E) for _ in range(4)]
        elif kind == "b":
            vis = [random_term(rng, E, PLANE) for _ in range(3)] + [random_term(rng, E, VISIBLE_LINES[rng.integers(3)])]
        else:
            v = random_term(rng, E, PLANE)
            vis = [v, term_over(rng, v.reshape(4, 8)[X0], PLANE), random_visible(rng, E), random_visible(rng, E)]
        if kind == "e":
            w1 = vis[0].reshape(4, 8)[F[0]]
            u = translate_of(rng, w1)
        else:
            u = E.C[:, rng.integers(E.N)]
        inv = term_over(rng, u, F)
        terms = vis + [inv]
        A = np.column_stack(terms)
        if np.linalg.matrix_rank(A, tol=1e-8) < 5:
            continue
        base = tuple(sorted(lookup[exact_codes(t.reshape(4, 8)[X0])[0].tobytes()] for t in vis))
        if kind in ("c", "e"):
            if len(set(base)) != 3 or codes_key([vis[0]]) == codes_key([vis[1]]):
                continue
        elif len(set(base)) < 4:
            continue
        coeffs = _gauss_coeffs(rng, 5)
        if kind in ("c", "e") and abs(coeffs[0] + coeffs[1]) < 1e-9:
            continue
        Psi = A @ coeffs
        fam = Family.from_cover(_planted_enum(E, Psi), sorted(set(base)))
        exempt = [i for i, u in enumerate(sorted(set(base))) if base.count(u) > 1]
        if fam is None or fam.kappa or fam.has_zero_coefficient(exempt):
            continue
        return terms, coeffs, base


def plant_gamma(rng, E, lookup, pair, kind):
    """A planted stage (gamma) instance at the flat pair: (a) random
    invisible terms on the two flats; (same) two line terms on one line
    with the same slice at its first point and different classes at the
    second; (same1) the same with one class and different phases; (samec)
    the same with one class, phases and coefficients cancelling at the
    second point; (cancel) the same slice at the first point with
    cancelling coefficients and different classes at the second; (cancel1)
    the same with one class and different phases at the second; (ray) a
    point term whose slice is the ray of the line term's slice at that
    point. Returns (terms, coeffs, base) with the invisible terms last."""
    F4, F5 = FLATS[pair[0]], FLATS[pair[1]]
    while True:
        vis = [random_visible(rng, E) for _ in range(3)]
        coeffs = _gauss_coeffs(rng, 5)
        if kind == "a":
            inv = [random_term(rng, E, F4), random_term(rng, E, F5)]
        elif kind in ("same", "same1", "samec", "cancel", "cancel1"):
            assert F4 == F5 and len(F4) == 2
            u = E.C[:, rng.integers(E.N)]
            reps, _ = pauli_reps(u, N2)
            if kind in ("same1", "samec", "cancel1"):
                k = rng.integers(8)
                k1 = k2 = k
                l1 = rng.integers(4)
                l2 = (l1 + rng.integers(1, 4)) % 4
            else:
                k1, k2 = rng.choice(8, size=2, replace=False)
                l1, l2 = rng.integers(4), rng.integers(4)
            t4, t5 = np.zeros((4, 8), dtype=complex), np.zeros((4, 8), dtype=complex)
            t4[F4[0]] = t5[F4[0]] = u
            t4[F4[1]] = (1j) ** l1 * apply_pauli(u, *reps[k1], N2)
            t5[F4[1]] = (1j) ** l2 * apply_pauli(u, *reps[k2], N2)
            inv = [t4.ravel(), t5.ravel()]
            if kind in ("cancel", "cancel1"):
                coeffs[4] = -coeffs[3]
            if kind == "samec":
                coeffs[4] = -coeffs[3] * (1j) ** ((l1 - l2) % 4)
        else:
            assert len(F4) == 1 and F4[0] in F5
            t5 = random_term(rng, E, F5)
            t4 = np.zeros((4, 8), dtype=complex)
            t4[F4[0]] = (1j) ** rng.integers(4) * t5.reshape(4, 8)[F4[0]]
            inv = [t4.ravel(), t5]
        terms = vis + inv
        A = np.column_stack(terms)
        if np.linalg.matrix_rank(A, tol=1e-8) < 5:
            continue
        base = tuple(sorted(lookup[exact_codes(t.reshape(4, 8)[X0])[0].tobytes()] for t in vis))
        if len(set(base)) < 3:
            continue
        Psi = A @ coeffs
        fam = Family.from_cover(_planted_enum(E, Psi), sorted(base))
        if fam is None or fam.kappa or fam.has_zero_coefficient():
            continue
        return terms, coeffs, base


def control_planted(args):
    """Planted instances for every flat (stage (beta')) and every flat pair
    (stage (gamma)), the target their sum with Gaussian-integer
    coefficients, the matcher run against the target's slices. Every
    planted decomposition must be among the hits."""
    E = make_enumerator()
    rng = np.random.default_rng(args.seed)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    plants = []
    only = set(args.only.split(",")) if args.only else None
    kinds = set(args.kinds.split(",")) if args.kinds else None
    if args.stage in ("beta", "all"):
        for flat in FLAT_NAMES:
            for rep in range(args.plant):
                for kind in ("a", "b", "c", "e"):
                    plants.append(("beta", flat, kind))
    if args.stage in ("gamma", "all"):
        for pair in FLAT_PAIRS:
            for rep in range(args.plant):
                plants.append(("gamma", pair, "a"))
                F4, F5 = FLATS[pair[0]], FLATS[pair[1]]
                if F4 == F5 and len(F4) == 2:
                    plants += [("gamma", pair, k) for k in ("same", "same1", "samec", "cancel", "cancel1")]
                if len(F4) == 1 and len(F5) == 2 and F4[0] in F5:
                    plants.append(("gamma", pair, "ray"))
    if only is not None:
        plants = [p for p in plants if (p[1] if isinstance(p[1], str) else "+".join(p[1])) in only]
    if kinds is not None:
        plants = [p for p in plants if p[2] in kinds]
    report = {"seed": args.seed, "git": git_commit(), "generated": _now(), "plants": []}
    run_meta = {"id": f"{_now()} {args.stage} {args.only} {args.kinds}", "stage": args.stage, "only": args.only,
                "kinds": args.kinds, "seed": args.seed, "git": git_commit(), "generated": _now()}
    n_ok = total = 0
    t_all = time.time()
    for stage, unit, kind in plants:
        if stage == "beta":
            terms, coeffs, base = plant_beta(rng, E, lookup, unit, kind)
        else:
            terms, coeffs, base = plant_gamma(rng, E, lookup, unit, kind)
        Psi = np.column_stack(terms) @ coeffs
        Bm = PlantedInvisibleMatcher(E, Psi, verbose=args.verbose)
        t0 = time.time()
        total += 1
        entry = {"stage": stage, "unit": list(unit) if isinstance(unit, tuple) else unit, "kind": kind,
                 "base": list(base)}
        try:
            hits, st = Bm.run_one(base, unit) if stage == "beta" else Bm.run_two(base, unit)
        except UnpinnedFamily as exc:
            entry.update({"status": "undecided", "reason": str(exc)[:200], "seconds": time.time() - t0})
            print(f"  {stage} {unit} kind {kind} base {base}: UNDECIDED {str(exc)[:120]}", flush=True)
        else:
            want = codes_key(terms)
            same = any(codes_key(h["terms"]) == want for h in hits)
            n_ok += same
            entry.update({"status": "recovered" if same else "not recovered", "hits": len(hits),
                          "seconds": time.time() - t0,
                          "stats": {kk: (v if not isinstance(v, np.generic) else v.item()) for kk, v in st.items()}})
            extra = (f"exact {st['exact_solutions']} scan {st['scan_solutions']} second {st['second_solutions']} "
                     f"ambiguous {st['ambiguous']} brute {st['brute']}" if stage == "beta" else
                     f"order {st['order']} exact {st['exact_solutions']} scan {st['scan_solutions']} scan2 "
                     f"{st['scan2_solutions']} pair {st['pair_solutions']}")
            print(f"  {stage} {unit} kind {kind} base {base}: {len(hits)} hits, planted "
                  f"{'recovered' if same else 'NOT recovered'}, {time.time() - t0:.2f}s, {extra}", flush=True)
        report["plants"].append(entry)
        _merge_planted(report, run_meta, time.time() - t_all, final=False)
    print(f"control-planted ({args.stage}{', ' + args.only if args.only else ''}"
          f"{', ' + args.kinds if args.kinds else ''}): {n_ok}/{total} planted decompositions recovered in "
          f"{time.time() - t_all:.0f}s; {'PASS' if n_ok == total else 'FAIL'}")
    merged = _merge_planted(report, run_meta, time.time() - t_all, final=True)
    print(f"control-planted record: {merged['recovered']}/{merged['total']} recovered over {len(merged['runs'])} "
          f"run(s) (beta {merged['by_stage']['beta']}, gamma {merged['by_stage']['gamma']}); "
          f"{'PASS' if merged['pass'] else 'FAIL'}")
    return 0 if n_ok == total else 1


def _merge_planted(report, run_meta, seconds, final):
    """The planted record merges the runs (the slow plant kinds run one at
    a time under the cap, and a run killed at the cap keeps what it
    finished): one entry per (stage, unit, kind), the latest run replacing
    an earlier one; rewritten after every plant."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "control_planted.json")
    merged = {"plants": [], "runs": []}
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
    keys = {(e["stage"], str(e["unit"]), e["kind"]) for e in report["plants"]}
    merged["plants"] = [e for e in merged["plants"] if (e["stage"], str(e["unit"]), e["kind"]) not in keys] \
        + report["plants"]
    merged["runs"] = [r for r in merged["runs"] if r.get("id") != run_meta["id"]]
    merged["runs"].append({**run_meta, "recovered": sum(e["status"] == "recovered" for e in report["plants"]),
                           "total": len(report["plants"]), "seconds": seconds, "complete": final})
    merged["recovered"] = sum(e["status"] == "recovered" for e in merged["plants"])
    merged["total"] = len(merged["plants"])
    merged["by_stage"] = {st: [sum(1 for e in merged["plants"] if e["stage"] == st and e["status"] == "recovered"),
                               sum(1 for e in merged["plants"] if e["stage"] == st)] for st in ("beta", "gamma")}
    merged["pass"] = merged["recovered"] == merged["total"]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, path)
    return merged


# -- the m = 4 control along a qubit pair --

def group_closure(perms, N, order):
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


def degenerate_minimal(E, r, minimal):
    """Full r-multisets over the minimal covers with r = chi + 1: T + (x,)
    with x in T or span(T), no unrepeated state dead (as the rank-4
    record)."""
    out = set()
    for T in minimal:
        for x in list(T) + E.in_span(T):
            ms = tuple(sorted(T + (x,)))
            distinct = sorted(set(ms))
            fam = Family.from_cover(E, distinct)
            exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
            if fam is not None and not fam.has_zero_coefficient(exempt):
                out.add(ms)
    return sorted(out)


def control_m4_pair(args):
    """Slice the stored rank-3 decomposition of |T>^4 along a qubit pair:
    every all-visible base is a full 3-cover of |T>^2 over the 60 two-qubit
    states, and the stage (alpha) matcher at n_1 = 2 over every such base
    (independent, dependent and repeated, from the enumerator) must recover
    the stored class wherever a member of it has an all-visible base at a
    base point, and nothing outside the stored list."""
    m, n1, rank = 4, 2, 3
    E = CoverEnumerator(2, orbit=ORBIT)
    lookup2 = {E.codes[i].tobytes(): i for i in range(E.N)}
    t0 = time.time()
    covers2, _ = E.covers(2)
    covers3, _ = E.covers(3)
    degenerate_ = degenerate_minimal(E, 3, covers2)
    bases = sorted(set(covers3) | set(degenerate_))
    print(f"{len(covers2)} full 2-covers and {len(covers3)} independent full 3-covers of |T>^2, "
          f"{len(degenerate_)} dependent or repeated, {len(bases)} bases [{time.time() - t0:.0f}s]")
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
          f"expected recoverable at x0 in {x0s}: {len(expected)}; not stored {len(unknown)}, missing "
          f"{len(missing)}, unexpected {len(extra)}; stats {stats}")
    os.makedirs(RESULTS, exist_ok=True)
    ok = not unknown and not missing and not extra and not stats["undecided"] and len(expected) == 1
    name = "control_m4_pair" + ("_all_x0" if args.all_x0 else "") + ".json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump({"git": git_commit(), "generated": _now(), "x0": x0s, "bases": len(bases),
                   "independent": len(covers3), "degenerate": len(degenerate_), "group_order": info4["order"],
                   "stored": len(stored_keys), "expected": len(expected), "recovered": len(recovered),
                   "unknown": len(unknown), "missing": len(missing), "extra": len(extra), "seconds": dt,
                   "stats": stats, "matcher": "reference" if Mt.native is None else "native", "pass": ok,
                   "recovered_from": {str(k): v for k, v in recovered.items()}}, f, indent=1)
    print("control-m4-pair:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# -- the tables --

def tables(args):
    """The stored residual tables the case split leans on: the rank-2 table
    of the feasibility note (no one-qubit slice of a rank-5 decomposition
    of |T>^5 has exactly three visible terms: 0 exact, stabilizer and
    rank-2 residuals at tau^{+-1} over all 65^3 combinations) and the
    rank-4 record's tables for the 8 rank-3 decompositions of |T>^3 at
    tau^{+-2} (0 exact rows: the configurations P A A B B and
    P P A p01 p10). Consistency checks on the stage (gamma) runs these say
    are empty; they are not inputs of the exclusion."""
    ok = True
    out = {"git": git_commit(), "generated": _now(), "checks": {}}
    path = os.path.join(RESULTS, "rank2_full.json")
    with open(path) as f:
        r2 = json.load(f)
    for j in ("1", "-1"):
        r = r2["ratios"][j]
        good = r["combos"] == r2["combinations_per_ratio"] == 65 ** 3 and r["exact"] == r["stabilizer"] == r["rank2"] == 0
        out["checks"][f"rank2_j{j}"] = {"combos": r["combos"], "exact": r["exact"], "stabilizer": r["stabilizer"],
                                        "rank2": r["rank2"], "pass": good}
        print(f"rank-2 table j = {j}: {r['combos']} combinations, exact {r['exact']}, stabilizer {r['stabilizer']}, "
              f"rank <= 2: {r['rank2']} ({r['collisions']} collisions decided): {'OK' if good else 'FAIL'}")
        ok &= good
    path = os.path.join(ROOT, "research", "t5_rank4", "results", "tables.json")
    with open(path) as f:
        t = json.load(f)["runs"]["t3"]
    tot = t["totals"]
    good = t["decompositions"] == 8 and tot["2"][0] == 0 and tot["-2"][0] == 0
    out["checks"]["t3_j2"] = {"decompositions": t["decompositions"], "totals": tot, "pass": good}
    print(f"rank-3 decompositions of |T>^3 ({t['decompositions']}) at tau^j: (exact, stabilizer) = "
          f"{ {j: tuple(v) for j, v in tot.items()} }: exact at j = +-2 {'zero, OK' if good else 'FAIL'}")
    ok &= good
    out["pass"] = ok
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_tables.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("tables:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("degenerate")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=degenerate)
    p = sub.add_parser("control-census")
    p.set_defaults(fn=control_census)
    p = sub.add_parser("sample")
    p.add_argument("stage", choices=["A", "B", "C", "beta", "gamma"])
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--cap", type=int, default=120, help="seconds per run")
    p.add_argument("--reference", action="store_true", help="stage A through the Python matcher")
    p.add_argument("--first", type=int, default=0, help="stage gamma: the first 3-cover to sample")
    p.add_argument("--pairs", default=None, help="stage gamma: comma-separated flat pairs as F+G")
    p.set_defaults(fn=sample)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=600.0, help="pod seconds per batch")
    p.add_argument("--pod-factor", type=float, default=2.0, help="pod seconds per laptop second")
    p.add_argument("--max-covers", type=int, default=2000, help="at most this many covers per stage B or C batch")
    p.set_defaults(fn=partition)
    p = sub.add_parser("control-witness")
    p.add_argument("--path", default="bounds/qubit_T-m5-upper-6.json")
    p.add_argument("--all-x0", action="store_true", help="every base point, not only 00")
    p.add_argument("--max-cand", type=int, default=2_000_000, help="candidate cap of the dense solve")
    p.add_argument("--reference", action="store_true", help="the Python matcher instead of the compiled kernel")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_witness)
    p = sub.add_parser("control-planted")
    p.add_argument("--stage", choices=["beta", "gamma", "all"], default="all")
    p.add_argument("--plant", type=int, default=1, help="repetitions of every (flat, kind)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--only", default=None, help="comma-separated flats (beta) or pairs as F+G (gamma)")
    p.add_argument("--kinds", default=None, help="comma-separated plant kinds")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_planted)
    p = sub.add_parser("control-m4-pair")
    p.add_argument("--all-x0", action="store_true", help="all four base points instead of 00")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_m4_pair)
    p = sub.add_parser("tables")
    p.set_defaults(fn=tables)
    args = ap.parse_args(argv[1:])
    common.lower_priority()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
