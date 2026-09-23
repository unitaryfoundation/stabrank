"""Setup, partition, rates and controls for the rank-5 exclusion of |H>^5 by
a two-qubit base slice (docs/notes/h5_rank5_exclusion.md). The batches run
through batch.py and are checked by aggregate.py.

Stage (alpha): an all-visible base at x_0 in {00, 01}; the base is a full
5-cover of |H>^3, the object of the H^6 census, which is reused by hash:
stage A is the 5,939,465 covers of distinct independent states regenerated
per pivot pair by the compiled kernel (research/h6_rank5/results/
kernel_census.json), stages B and C the 28,396 dependent and repeated
covers of research/h6_rank5/degenerate_covers_v2.json (the cancel-at-base
multisets included). Stage (beta): four visible terms at x_0 and one
invisible line term on the diagonal missing x_0; the base is a full
4-cover of |H>^3, listed here (`beta-covers --write`: 3,460 covers of
distinct independent states and 6 multisets of pattern (2, 1, 1)) and
matched by beta.BetaMatcher.

Commands
  beta-covers [--write]     enumerate the stage (beta) bases and, with --write,
                            store them as beta_covers.json with their hash
  sample STAGE [--count N] [--cap S]
                            time the matcher on N covers of a stage (A, B, C or
                            beta) at both base points; the rates go to
                            results/rates.json, which `partition` reads
  partition [--target-s S] [--pod-factor F]
                            write partition.json: stage A pivot pairs grouped
                            into batches of about S pod seconds (F times the
                            laptop rates), stages B, C and beta round-robin
  control-witness [--base K] [--max-cand N]
                            recover the rank-6 witness bounds/qubit_H-m5-upper-6.json
                            from each of its 24 all-visible (pair, base point)
                            slices through the stage (alpha) matcher
  control-beta [--plant K] [--seed S]
                            recover planted stage (beta) instances (independent
                            bases, a repeated base state, a visible term on a
                            coordinate line, both base points)
  control-m4-pair           recover the rank-4 decompositions of |H>^4 from the
                            full 4-covers of |H>^2 along a qubit pair
  tables [--fact1]          re-run the residual tables behind the (2, 2)
                            exclusion (ratio t^{+-2} for the rank-3 decompositions
                            of |H>^3) and, with --fact1, Fact 1 (ratio t^{+-1}
                            for the 30 rank-4 decompositions of |H>^4, 5 minutes)

Running a batch: `batch.py K`; checking the results: `aggregate.py`.
"""
from __future__ import annotations

import argparse
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
import common  # noqa: E402  (research/h5_rank5/common.py)
from common import (BETA_COVERS, CENSUS, DEGENERATE, HERE, M, N1, N2, PARTITION, RANK, RATES,  # noqa: E402
                    RESULTS, ROOT, X0S, genuine, git_commit, pairs_of, stage_of)
from beta import BetaMatcher, PlantedBetaMatcher, codes_key  # noqa: E402
from slice_cover import (CoverEnumerator, Family, SliceMatcher, UnpinnedFamily, _canon_rows,  # noqa: E402
                         _groups_by_key, _reduce, apply_pauli, exact_codes, pauli_reps, patterns)
from rank_exclusion import dictionary, symmetry_orbit_reps  # noqa: E402


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- lists -------

def degenerate_covers(E, r, covers3, covers4):
    """Full r-covers (r = 4, 5) whose distinct states are dependent or which
    repeat a state, as multisets (sorted tuples): every multiset over a
    3-cover or 4-cover plus states of the span, the pairs parallel modulo a
    3-cover, and the cancel-at-base multisets T + (b, b), with a
    coefficient family in which no unrepeated state is dead. The same
    function as research/h6_rank5/driver.degenerate_covers (the v2 list);
    kept here so that nothing from research/h6_rank5 is imported, and
    checked against that list's hash by the aggregate."""
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
    if r == 5:
        for Cv in covers4:
            span = [x for x in range(E.N) if x not in Cv and E.rank_mod2(tuple(Cv) + (x,), False) == 4]
            for x in list(Cv) + span:
                ms = tuple(sorted(Cv + (x,)))
                if ok(ms):
                    out.add(ms)
    return sorted(out)


def beta_bases(E, covers3, covers4):
    """The stage (beta) bases: the full 4-covers of distinct independent
    states (one per orbit, from the enumerator) and the degenerate
    4-multisets over the 3-cover classes (a state of the span, or a repeated
    state, with no unrepeated state dead). Returns (sorted list, the
    degenerate ones)."""
    deg = degenerate_covers(E, 4, covers3, covers4)
    bases = sorted(set(covers4) | set(deg))
    return bases, deg


def beta_covers(args):
    E = CoverEnumerator(N2)
    t0 = time.time()
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    bases, deg = beta_bases(E, covers3, covers4)
    by = {}
    for ms in deg:
        by[str(common.multiplicity_pattern(ms))] = by.get(str(common.multiplicity_pattern(ms)), 0) + 1
    span_states = {T: [x for x in range(E.N) if x not in T and E.rank_mod2(tuple(T) + (x,), False) == 3]
                   for T in covers3}
    dt = time.time() - t0
    print(f"{len(covers3)} full 3-covers, {len(covers4)} full 4-covers of distinct independent states, "
          f"{len(deg)} degenerate 4-multisets by pattern {by}, states in span(T): "
          f"{ {str(T): len(v) for T, v in span_states.items()} }; {len(bases)} stage (beta) bases [{dt:.0f}s]")
    if args.write:
        rec = {"orbit": "qubit_H", "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
               "covers3": [list(T) for T in covers3], "span_states": {str(list(T)): v for T, v in span_states.items()},
               "independent": len(covers4), "degenerate": len(deg), "degenerate_by_pattern": by,
               "count": len(bases), "seconds": dt, "git": git_commit(),
               "covers": [list(c) for c in bases]}
        sha = common.write_hashed(BETA_COVERS, rec)
        print(f"wrote {BETA_COVERS} (sha256 {sha[:16]})")
    return 0


# ------------------------------------------------------------- rates -------

def census_covers(E, count, pivot=0):
    i = int(E.reps[pivot])
    Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
    members, partners = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    covers = []
    for j in partners:
        got, _ = E.pair_covers(5, i, int(j), Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= count:
            break
    return covers[:count]


def _spread(lst, n):
    n = min(n, len(lst))
    return [lst[int(round(k * (len(lst) - 1) / max(n - 1, 1)))] for k in range(n)]


def _load_rates():
    if os.path.exists(RATES):
        with open(RATES) as f:
            return json.load(f)
    return {"git": git_commit(), "stages": {}}


def sample(args):
    """Time the matcher of one stage on covers spread through its list, both
    base points, one (cover, x0) run at a time under a per-run cap."""
    E = CoverEnumerator(N2)
    stage = args.stage
    rec = {"stage": stage, "count": 0, "runs": 0, "seconds": [], "by_pattern": {}, "hits": 0, "refused": 0,
           "undecided": 0, "capped": 0, "hist": {}, "generated": _now(), "git": git_commit()}
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
                covers += _spread(lst, args.count if len(lst) > 6 else len(lst))
        else:
            covers = _spread(covers, args.count)
        rec["matcher"] = "reference"
    else:
        Mt = BetaMatcher(E)
        bases, _ = common.load_beta()
        ind = [c for c in bases if len(set(c)) == 4]
        deg = [c for c in bases if len(set(c)) < 4]
        covers = _spread(ind, args.count) + deg
        rec["matcher"] = "beta"
    import signal

    class Deadline(Exception):
        pass

    def alarm(signum, frame):
        raise Deadline()

    signal.signal(signal.SIGALRM, alarm)
    t_all = time.time()
    for cover in covers:
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
                    key = (",".join(str(b) for b in st["coord_solutions"]) if stage != "beta"
                           else f"{st['b_solutions']},{st['a1_solutions']},{st['a2_solutions'] + st['a2_point']}")
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
    rec["covers"] = [list(c) for c in covers]
    print(f"stage {stage} ({rec['matcher']}): {rec['count']} covers x {len(X0S)} base points: per (cover, x0) "
          f"mean {a.mean():.4f}s, median {np.median(a):.4f}s, max {a.max():.3f}s; by pattern "
          f"{ {p: round(v, 4) for p, v in rec['per_pattern_mean_s'].items()} }; hits {rec['hits']}, refused "
          f"{rec['refused']}, undecided {rec['undecided']}, capped {rec['capped']}; histogram {rec['hist']}")
    os.makedirs(RESULTS, exist_ok=True)
    rates = _load_rates()
    rates["stages"][stage] = {k: v for k, v in rec.items() if k not in ("seconds", "by_pattern", "covers")}
    rates["git"] = git_commit()
    with open(RATES, "w") as f:
        json.dump(rates, f, indent=1)
    with open(os.path.join(RESULTS, f"sample_{stage}.json"), "w") as f:
        json.dump(rec, f)
    return 0


# ---------------------------------------------------------- partition ------

def partition(args):
    E = CoverEnumerator(N2)
    units = pairs_of(E)
    cen = common.load_census()
    rates = _load_rates()["stages"]
    for st in ("A", "B", "C", "beta"):
        if st not in rates:
            raise SystemExit(f"no rate for stage {st} in {RATES}; run `driver.py sample {st}` first")
    F = args.pod_factor
    n_x0 = len(X0S)
    # stage A: kernel seconds plus the native matcher per (cover, x0), from the census
    by = {(r[0], r[1]): r for r in cen["rows"]}
    if len(by) != len(units) or any((i, j) not in by for i, j, _ in units):
        raise SystemExit("the census rows are not the enumerator's pivot pairs")
    a_rate = rates["A"]["per_run_mean_s"]
    costs = [F * (by[(i, j)][5] + a_rate * n_x0 * by[(i, j)][3]) for i, j, _ in units]
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
    # stages B and C: the degenerate list by pattern rate; stage beta: its list
    covers, deg = common.load_degenerate()
    beta, bdoc = common.load_beta()
    pat_rate = {}
    for st in ("B", "C", "beta"):
        pat_rate.update({p: v * n_x0 * F for p, v in rates[st]["per_pattern_mean_s"].items()})
    ids = {"B": [], "C": []}
    cost = {"B": 0.0, "C": 0.0, "beta": 0.0}
    for k, c in enumerate(covers):
        st = stage_of(c)
        ids[st].append(k)
        cost[st] += pat_rate[str(common.multiplicity_pattern(c))]
    for k, c in enumerate(beta):
        cost["beta"] += pat_rate[str(common.multiplicity_pattern(c))]
    ids["beta"] = list(range(len(beta)))
    counts = {}
    for st in ("B", "C", "beta"):
        n = max(1, math.ceil(cost[st] / args.target_s))
        for b in range(n):
            geometry.append({"index": len(geometry), "stage": st, "cover_ids": ids[st][b::n]})
        counts[st] = n
        est[st] = cost[st]
    est["total"] = sum(est.values())
    rec = {"orbit": "qubit_H", "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "x0": list(X0S), "pivot_orbits": int(E.info["orbits"]), "units": len(units),
           "stage_a_batches": len(batches_a), "stage_b_batches": counts["B"], "stage_c_batches": counts["C"],
           "stage_beta_batches": counts["beta"],
           "cost_model": {"pod_factor": F, "target_s": args.target_s, "rates": os.path.relpath(RATES, HERE),
                          "a_s_per_run": a_rate, "s_per_run_by_pattern": pat_rate, "runs_per_cover": n_x0},
           "estimated_s": est,
           "census": {"file": os.path.relpath(CENSUS, HERE), "sha256": common.sha256_file(CENSUS),
                      "covers": cen["covers"], "units": cen["units"]},
           "degenerate": {"file": os.path.relpath(DEGENERATE, HERE), "sha256": deg["sha256"],
                          "count": len(covers), "stage_b_covers": len(ids["B"]), "stage_c_covers": len(ids["C"])},
           "beta": {"file": os.path.relpath(BETA_COVERS, HERE), "sha256": bdoc["sha256"], "count": len(beta),
                    "independent": bdoc["independent"], "degenerate": bdoc["degenerate"]},
           "batches": len(geometry), "git": git_commit(), "generated": _now(), "batch_geometry": geometry}
    sha = common.write_hashed(PARTITION, rec)
    print(f"{len(units)} pivot pairs in {len(batches_a)} stage A batches ({est['A'] / 3600:.2f} pod CPU-h), "
          f"{counts['B']} stage B ({est['B'] / 3600:.2f}), {counts['C']} stage C ({est['C'] / 3600:.2f}), "
          f"{counts['beta']} stage beta ({est['beta'] / 3600:.2f}); {len(geometry)} batches, "
          f"{est['total'] / 3600:.1f} pod CPU-h at {F} x the laptop rates; wrote {PARTITION} (sha256 {sha[:16]})")
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
    """The rank-6 witness of |H>^5 sliced along every qubit pair: at every
    (pair, x0) where all six terms are nonzero the base is a 6-cover of
    psi_3, and the stage (alpha) matcher must return the witness itself
    among its rank-6 decompositions. Every base is reported; a base that
    raises or exceeds the candidate cap is recorded as aborted and fails
    the control, never counted as passed."""
    m, n1, rank = M, N1, 6
    E = CoverEnumerator(N2)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    w = json.load(open(os.path.join(ROOT, args.path)))
    terms = [common.term_vector(t, 2, m) for t in w["witness"]["terms"]]
    psi = common.target("qubit_H", m)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    assert np.linalg.norm(np.column_stack(terms) @ c - psi) < 1e-9
    bases = {}
    types = {}
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
    Mt = SliceMatcher(E, n1, verbose=args.verbose, max_cand=args.max_cand)
    report = {"witness": args.path, "git": git_commit(), "generated": _now(), "max_cand": args.max_cand,
              "bases": []}
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
    name = "control_witness.json" if args.base is None else f"control_witness_{args.base}.json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(report, f, indent=1)
    print(f"control-witness: {passed}/{len(report['bases'])} bases recover the witness, {aborted} aborted; "
          f"{'PASS' if passed == len(report['bases']) else 'FAIL'}")
    return 0 if passed == len(report["bases"]) else 1


# -- planted stage (beta) instances --

def random_term(rng, E, flat, x0_line=None):
    """A random five-qubit stabilizer state (entries in {0, +-1, +-i}) whose
    flat along the first two qubits is `flat` (a tuple of points), built by
    the structure lemma from a random three-qubit base state and random
    Pauli classes, phases and quadratic sign."""
    u = E.C[:, rng.integers(E.N)]
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


def _pauli_copy(rng, E, v, x0):
    """A second copy of the plane term v: the same base slice at x0, fresh
    Pauli classes, phases and sign elsewhere."""
    u = v.reshape(4, 8)[x0]
    reps, _ = pauli_reps(u, N2)
    t = np.zeros((4, 8), dtype=complex)
    t[x0] = u
    k1, k2 = rng.integers(8), rng.integers(8)
    l1, l2, s = rng.integers(4), rng.integers(4), rng.integers(2)
    (a1, c1), (a2, c2) = reps[k1], reps[k2]
    t[x0 ^ 0b01] = (1j) ** l1 * apply_pauli(u, a1, c1, N2)
    t[x0 ^ 0b10] = (1j) ** l2 * apply_pauli(u, a2, c2, N2)
    t[x0 ^ 0b11] = (-1) ** s * (1j) ** (l1 + l2) * apply_pauli(apply_pauli(u, a1, c1, N2), a2, c2, N2)
    return t.ravel()


def _gauss_coeffs(rng, n):
    """Nonzero Gaussian-integer coefficients (small)."""
    return np.array([complex(rng.integers(1, 4)) * (1j) ** rng.integers(4) for _ in range(n)])


def control_beta(args):
    """Planted stage (beta) instances at both base points: (a) three planes,
    a line on the diagonal through x0 and the invisible line on the other
    diagonal; (b) two planes and two lines through x0; (c) a repeated base
    state (two plane copies with one base slice); (d) a visible term on a
    coordinate line through x0 (absent at the second coordinate point and
    at x_b), which Fact 1 excludes but the matcher must find; (e) a repeated
    base state whose first copy shares a Pauli class with the invisible term
    at the first coordinate point (the ambiguous case that the joint
    reconstruction decides). Every planted decomposition must be among the
    hits."""
    E = CoverEnumerator(N2)
    rng = np.random.default_rng(args.seed)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    PLANE = (0, 1, 2, 3)
    kinds = ["a", "b", "c", "d", "e"]
    report = {"seed": args.seed, "git": git_commit(), "generated": _now(), "plants": []}
    n_ok = 0
    total = 0
    for k in range(args.plant):
        for x0 in X0S:
            kind = kinds[k % len(kinds)]
            diag = (x0, x0 ^ 0b11)
            other = (x0 ^ 0b01, x0 ^ 0b10)
            while True:
                if kind == "a":
                    vis = [random_term(rng, E, PLANE) for _ in range(3)] + [random_term(rng, E, diag)]
                elif kind == "b":
                    vis = [random_term(rng, E, PLANE) for _ in range(2)] + [random_term(rng, E, diag) for _ in range(2)]
                elif kind == "c":
                    v = random_term(rng, E, PLANE)
                    vis = [v, _pauli_copy(rng, E, v, x0), random_term(rng, E, PLANE), random_term(rng, E, diag)]
                elif kind == "d":
                    vis = [random_term(rng, E, PLANE) for _ in range(2)] + [random_term(rng, E, diag),
                                                                            random_term(rng, E, (x0, x0 ^ 0b01))]
                else:
                    # a repeated base state whose first copy and the invisible line term share a
                    # Pauli class at the first coordinate point: the residual there lies in the
                    # span of two translates of the repeated state, the ambiguous case of beta.py
                    v = random_term(rng, E, PLANE)
                    vis = [v, _pauli_copy(rng, E, v, x0), random_term(rng, E, PLANE), random_term(rng, E, diag)]
                if kind == "e":
                    w1 = v.reshape(4, 8)[x0 ^ 0b01]
                    reps, _ = pauli_reps(w1, N2)
                    a, c = reps[rng.integers(8)]
                    inv = np.zeros((4, 8), dtype=complex)
                    inv[x0 ^ 0b01] = (1j) ** rng.integers(4) * w1
                    inv[x0 ^ 0b10] = (1j) ** rng.integers(4) * apply_pauli(w1, a, c, N2)
                    inv = inv.ravel()
                else:
                    inv = random_term(rng, E, other)
                terms = vis + [inv]
                A = np.column_stack(terms)
                if np.linalg.matrix_rank(A, tol=1e-8) < 5:
                    continue
                base = tuple(sorted(lookup[exact_codes(t.reshape(4, 8)[x0])[0].tobytes()] for t in vis))
                if kind not in ("c", "e") and len(set(base)) < 4:
                    continue
                if kind in ("c", "e") and len(set(base)) != 3:
                    continue
                coeffs = _gauss_coeffs(rng, 5)
                if kind in ("c", "e") and abs(coeffs[0] + coeffs[1]) < 1e-9:
                    continue
                Psi = A @ coeffs
                # the visible terms' base coefficients must be a point with no dead state
                fam = Family.from_cover(_planted_enum(E, Psi, x0), sorted(set(base)))
                if fam is None or fam.kappa:
                    continue
                break
            Bm = PlantedBetaMatcher(E, Psi, x0)
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
                      f"{'recovered' if same else 'NOT recovered'}, {time.time() - t0:.2f}s, b {st['b_solutions']} "
                      f"a1 {st['a1_solutions']} a2 {st['a2_solutions']} recon {st['reconstructions']} "
                      f"ambiguous {st['ambiguous']} brute {st['brute']}", flush=True)
            report["plants"].append(entry)
    report["recovered"] = n_ok
    report["total"] = total
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "control_beta.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"control-beta: {n_ok}/{total} planted decompositions recovered; {'PASS' if n_ok == total else 'FAIL'}")
    return 0 if n_ok == total else 1


def _planted_enum(E, Psi, x0):
    import copy
    from beta import to_field
    rows = Psi.reshape(4, 8)
    E2 = copy.copy(E)
    E2.psi = rows[x0]
    E2.psi1, E2.psi2 = to_field(E.F1, rows[x0]), to_field(E.F2, rows[x0])
    return E2


# -- the m = 4 control along a qubit pair --

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


def control_m4_pair(args):
    """Slice the 30 stored rank-4 decompositions of |H>^4 along a qubit
    pair: every all-visible base is a full 4-cover of |H>^2 over the 60
    two-qubit states, and the stage (alpha) matcher at n_1 = 2 over every
    such base (independent, dependent and repeated, from the enumerator)
    must recover every stored class that has such a base and nothing
    outside the stored list."""
    m, n1, rank = 4, 2, 4
    E = CoverEnumerator(2)
    lookup2 = {E.codes[i].tobytes(): i for i in range(E.N)}
    t0 = time.time()
    covers3, _ = E.covers(3)
    covers4, _ = E.covers(4)
    degenerate = degenerate_covers(E, 4, covers3, covers4)
    bases = sorted(set(covers4) | set(degenerate))
    print(f"{len(covers4)} independent full 4-covers of |H>^2, {len(degenerate)} dependent or repeated, "
          f"{len(bases)} bases [{time.time() - t0:.0f}s]")
    D4 = dictionary(2, 4)
    codes4, _ = patterns(D4)
    lookup4 = {codes4[i].tobytes(): i for i in range(D4.shape[1])}
    group = m4_group(D4)
    stored, _ = common.load_decompositions("qubit_H", m, rank)
    stored_keys = {}
    expected = set()
    x0s = list(range(1 << n1)) if args.all_x0 else list(X0S)
    for terms, _ in stored:
        key = canonical([lookup4[exact_codes(t)[0].tobytes()] for t in terms], group)
        stored_keys[key] = terms
        for S in itertools.combinations(range(m), n1):
            for x0 in x0s:
                b = slice_terms(terms, S, x0, m, n1, lookup2)
                if b is None:
                    continue
                distinct = sorted(set(b))
                fam = Family.from_cover(E, distinct)
                exempt = [i for i, u in enumerate(distinct) if b.count(u) > 1]
                if fam is not None and not fam.has_zero_coefficient(exempt):
                    expected.add(key)
    Mt = SliceMatcher(E, n1, verbose=args.verbose)
    recovered = {}
    t1 = time.time()
    stats = {"matched": 0, "refused": 0, "hits": 0, "non_genuine": 0, "undecided": 0}
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
          f"{len(recovered)} classes of rank-{rank} decompositions of |H>^{m}; stored {len(stored_keys)}, "
          f"expected recoverable {len(expected)}; not stored {len(unknown)}, missing {len(missing)}, "
          f"unexpected {len(extra)}; stats {stats}")
    os.makedirs(RESULTS, exist_ok=True)
    ok = not unknown and not missing and not extra and not stats["undecided"]
    with open(os.path.join(RESULTS, "control_m4_pair.json"), "w") as f:
        json.dump({"git": git_commit(), "generated": _now(), "x0": x0s, "bases": len(bases),
                   "independent": len(covers4), "degenerate": len(degenerate), "stored": len(stored_keys),
                   "expected": len(expected), "recovered": len(recovered), "unknown": len(unknown),
                   "missing": len(missing), "extra": len(extra), "seconds": dt, "stats": stats, "pass": ok,
                   "recovered_from": {str(k): v for k, v in recovered.items()}}, f, indent=1)
    print("control-m4-pair:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# -- the residual tables --

def tables(args):
    """Re-run the residual tables of research/constructions/two_qubit_slice.py
    that the case split rests on: the rank-3 decompositions of |H>^3 at the
    ratios t^j, |j| <= 2 (the (2, 2) configuration needs an exact
    combination at j = +-2), and with --fact1 the rank-4 decompositions of
    |H>^4 at |j| <= 1 (Fact 1 needs (0, 0) at j = +-1). The output lines are
    stored under results/tables.json."""
    script = os.path.join(ROOT, "research", "constructions", "two_qubit_slice.py")
    runs = [("h3_j2", ["--m", "5", "--n1", "2", "--ratio-tables", "2"])]
    if args.fact1:
        runs.append(("h4_j1", ["--m", "6", "--n1", "2", "--ratio-tables", "1"]))
    out = {"git": git_commit(), "generated": _now(), "runs": {}}
    ok = True
    for name, extra in runs:
        cmd = [sys.executable, script] + extra
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        lines = proc.stdout.strip().splitlines()
        print(f"{name}: {' '.join(extra)} ({time.time() - t0:.0f}s, exit {proc.returncode})")
        for ln in lines:
            print("   ", ln)
        totals = next((ln for ln in lines if "totals" in ln), "")
        # parse "j=-2:(a, b)" entries
        parsed = {}
        for tok in totals.split("j=")[1:]:
            j, rest = tok.split(":", 1)
            a, b = rest.strip().lstrip("(").split(")")[0].split(",")
            parsed[int(j)] = (int(a), int(b))
        out["runs"][name] = {"command": extra, "exit": proc.returncode, "seconds": time.time() - t0,
                             "lines": lines, "totals": {str(j): v for j, v in parsed.items()}}
        if name == "h3_j2":
            good = proc.returncode == 0 and parsed.get(2, (1,))[0] == 0 and parsed.get(-2, (1,))[0] == 0
            print(f"    (2, 2) exclusion: exact combinations at j = +-2 are {parsed.get(-2)} and {parsed.get(2)}: "
                  f"{'OK' if good else 'FAIL'}")
        else:
            good = proc.returncode == 0 and parsed.get(1) == (0, 0) and parsed.get(-1) == (0, 0)
            print(f"    Fact 1: (exact, stabilizer) at j = +-1 are {parsed.get(-1)} and {parsed.get(1)}: "
                  f"{'OK' if good else 'FAIL'}")
        out["runs"][name]["pass"] = good
        ok &= good
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "tables.json")
    prev = {}
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f).get("runs", {})
    prev.update(out["runs"])
    out["runs"] = prev
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("tables:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("beta-covers")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=beta_covers)
    p = sub.add_parser("sample")
    p.add_argument("stage", choices=["A", "B", "C", "beta"])
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--cap", type=int, default=120, help="seconds per (cover, x0) run")
    p.add_argument("--reference", action="store_true", help="stage A through the Python matcher")
    p.set_defaults(fn=sample)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=600.0, help="pod seconds per batch")
    p.add_argument("--pod-factor", type=float, default=2.0, help="pod seconds per laptop second")
    p.set_defaults(fn=partition)
    p = sub.add_parser("control-witness")
    p.add_argument("--path", default="bounds/qubit_H-m5-upper-6.json")
    p.add_argument("--base", type=int, default=None, help="run only the k-th base (0-based)")
    p.add_argument("--max-cand", type=int, default=2_000_000, help="candidate cap of the dense solve")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_witness)
    p = sub.add_parser("control-beta")
    p.add_argument("--plant", type=int, default=8, help="planted instances per base point")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(fn=control_beta)
    p = sub.add_parser("control-m4-pair")
    p.add_argument("--all-x0", action="store_true", help="all four base points instead of 00 and 01")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=control_m4_pair)
    p = sub.add_parser("tables")
    p.add_argument("--fact1", action="store_true", help="also the 5-minute Fact 1 tables")
    p.set_defaults(fn=tables)
    args = ap.parse_args(argv[1:])
    common.lower_priority()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
