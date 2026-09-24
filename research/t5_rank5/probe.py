"""Feasibility probes for the rank-5 exclusion of |T>^5 by a two-qubit base
slice (docs/notes/t5_rank5_feasibility.md). Every subcommand writes a
record under results/ and prints a summary; run each one through run.py
(nice 19, wall-clock cap).

  flats                    the flat configurations of five terms along a
                           qubit pair allowed by Fact 2 (at most two absent
                           terms at any point), by the number of terms
                           invisible at 00 and at the best point
  census5 [--pairs K]      the pivot pairs of the 5-cover enumeration of
                           |T>^3, the compiled kernel on a sample of K
                           pairs, and the extrapolated census size and time
  census5 --full [--pivots a,b]
                           the whole census (per-pair counts only), one
                           pivot at a time, appending to results/census5_full.json
  degenerate5              the dependent and repeated-state full 5-multisets
                           over the 4 full 3-covers and the 4,697 full
                           4-covers of |T>^3 (the stage B and C lists)
  rates [--count N]        SliceMatcher(E, 2) at orbit qubit_T on sampled
                           5-covers of each kind at x_0 in {00, 01, 11}
  beta-rate [--count N]    the H^5 stage (beta) matcher (diagonal fifth
                           term) run at orbit qubit_T on sampled 4-covers
  census4 [--pairs K]      CoverEnumerator(4, qubit_T): construction time,
                           pivot pairs, the Python 4-cover kernel on a sample
                           of pairs, and the extrapolated census of the
                           rank-4 decompositions of |T>^4
  rank2 [--count K]        the one-qubit slice with exactly three visible
                           terms: the residual of the rank-3 decomposition
                           of |T>^4 at ratio tau^{+-1} over a sample of the
                           65^3 code combinations, tested for stabilizer
                           rank <= 2 over the 36,720 four-qubit states
                           (modular hashing, exact re-decision), with the
                           projected time for the full table
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import itertools
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from slice_cover import (P1, P2, CoverEnumerator, Family, Field, SliceMatcher, TermOptions,  # noqa: E402
                         UnpinnedFamily, _affine_solve_mod, _canon_rows, _groups_by_key, _rank_mod, _reduce,
                         exact_codes, patterns)
from rank_exclusion import dictionary  # noqa: E402

ORBIT = "qubit_T"
N2 = 3
N1 = 2
COVERS4 = os.path.join(ROOT, "research", "t5_rank4", "covers4.json")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write(name, rec):
    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(rec)
    rec.setdefault("generated", _now())
    rec.setdefault("git", git_commit())
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def logger():
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)
    return log


def load_covers4():
    with open(COVERS4) as f:
        doc = json.load(f)
    covers = [tuple(int(x) for x in c) for c in doc["covers"]]
    kinds = doc["kinds"]
    covers3 = [tuple(int(x) for x in T) for T in doc["covers3"]]
    return covers3, [c for c, k in zip(covers, kinds) if k == "A"], [c for c, k in zip(covers, kinds) if k == "C"]


def spread(lst, n):
    n = min(n, len(lst))
    if n == 0:
        return []
    return [lst[int(round(k * (len(lst) - 1) / max(n - 1, 1)))] for k in range(n)]


# ------------------------------------------------------------ flats -------

FLAT_NAMES = {
    frozenset({0, 1, 2, 3}): "P",
    frozenset({0, 3}): "A", frozenset({1, 2}): "B",
    frozenset({0, 1}): "C0", frozenset({2, 3}): "C1",     # x1 = 0, x1 = 1
    frozenset({0, 2}): "D0", frozenset({1, 3}): "D1",     # x2 = 0, x2 = 1
    frozenset({0}): "p00", frozenset({1}): "p01", frozenset({2}): "p10", frozenset({3}): "p11",
}


def flats(args):
    """Points of F_2^2 are 2 x1 + x2. Every nonempty subset of a 2-element
    set or the whole plane is an affine flat: 4 points, 6 lines, the plane."""
    all_flats = sorted(FLAT_NAMES, key=lambda f: (-len(f), FLAT_NAMES[f]))
    rows = []
    by_min, by_00 = {}, {}
    inv00 = {}
    gamma = []
    for combo in itertools.combinations_with_replacement(range(len(all_flats)), 5):
        fl = [all_flats[i] for i in combo]
        a = [sum(1 for f in fl if x not in f) for x in range(4)]
        if max(a) > 2:
            continue
        names = [FLAT_NAMES[f] for f in fl]
        m = min(a)
        by_min[m] = by_min.get(m, 0) + 1
        by_00[a[0]] = by_00.get(a[0], 0) + 1
        inv = tuple(sorted(FLAT_NAMES[f] for f in fl if 0 not in f))
        inv00[inv] = inv00.get(inv, 0) + 1
        rows.append({"flats": names, "absent": a, "min": m, "invisible_at_00": list(inv)})
        if m == 2:
            gamma.append({"flats": names, "absent": a})
    rec = {"configurations": len(rows), "by_min_absent": {str(k): v for k, v in sorted(by_min.items())},
           "by_absent_at_00": {str(k): v for k, v in sorted(by_00.items())},
           "invisible_multisets_at_00": {" ".join(k) if k else "(none)": v for k, v in sorted(inv00.items())},
           "min_absent_2": gamma, "rows": rows}
    print(f"{len(rows)} flat configurations with at most two absent terms at every point; "
          f"by min absent {rec['by_min_absent']}; by absent at 00 {rec['by_absent_at_00']}")
    print(f"{len(inv00)} invisible multisets at 00: {rec['invisible_multisets_at_00']}")
    print(f"{len(gamma)} configurations with two absent terms at every point:")
    for g in gamma:
        print("  " + " ".join(g["flats"]) + f"  absent {g['absent']}")
    write("flats.json", rec)
    return 0


# ---------------------------------------------------------- census 5 ------

def pivot_pairs(E):
    """[(pivot, partner, members)] over every pivot of E, plus the per-pivot plans."""
    units, plans = [], {}
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
        plans[i] = (Qi, mask)
        for j in partners:
            units.append((i, int(j), int(len(members))))
    return units, plans


def census5(args):
    log = logger()
    E = CoverEnumerator(N2, orbit=ORBIT)
    log(f"{ORBIT} psi_3: N = {E.N}, group order {E.info['order']}, {E.info['orbits']} orbits, "
        f"kernel {'native' if E.native_cover5 is not None else 'reference'}")
    units, plans = pivot_pairs(E)
    per_pivot = {}
    for i, j, M in units:
        per_pivot.setdefault(i, [0, M])
        per_pivot[i][0] += 1
    log(f"{len(units)} pivot pairs over {len(per_pivot)} pivots: "
        + ", ".join(f"{i}:{c} partners/{M} members" for i, (c, M) in per_pivot.items()))
    rng = np.random.default_rng(args.seed)
    if args.full:
        path = os.path.join(RESULTS, "census5_full.json")
        rec = {"orbit": ORBIT, "n2": N2, "rank": 5, "N": E.N, "units": len(units), "rows": [],
               "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"]}
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
        done = {(r[0], r[1]) for r in rec["rows"]}
        sel = [int(x) for x in args.pivots.split(",")] if args.pivots else sorted(per_pivot)
        covers_seen = set()
        for i in sel:
            t1 = time.time()
            n_new = 0
            for (pi, j, M) in units:
                if pi != i or (pi, j) in done:
                    continue
                Qi, mask = plans[i]
                tk = time.time()
                got, nc = E.pair_covers(5, i, j, Qi, mask)
                covers_seen.update(got)
                rec["rows"].append([i, j, M, len(got), int(nc), time.time() - tk])
                n_new += 1
            tot = sum(r[3] for r in rec["rows"] if r[0] == i)
            log(f"pivot {i}: {n_new} new pairs, {tot} covers at this pivot [{time.time() - t1:.0f}s]")
            os.makedirs(RESULTS, exist_ok=True)
            with open(path, "w") as f:
                json.dump(rec, f)
        rows = rec["rows"]
        rec["covers"] = int(sum(r[3] for r in rows))
        rec["candidates"] = int(sum(r[4] for r in rows))
        rec["seconds"] = float(sum(r[5] for r in rows))
        rec["pairs_done"] = len(rows)
        rec["distinct_covers_this_run"] = len(covers_seen)
        with open(path, "w") as f:
            json.dump(rec, f)
        log(f"{len(rows)}/{len(units)} pairs done: {rec['covers']} covers, {rec['candidates']} candidates, "
            f"{rec['seconds']:.0f} kernel seconds; wrote {os.path.relpath(path, ROOT)}")
        return 0
    # a stratified sample: for each pivot a share of the K pairs proportional to its partners
    K = args.pairs
    sample = []
    for i, (c, M) in per_pivot.items():
        mine = [u for u in units if u[0] == i]
        k = max(1, int(round(K * len(mine) / len(units))))
        idx = rng.choice(len(mine), size=min(k, len(mine)), replace=False)
        sample.extend(mine[t] for t in sorted(idx))
    rows = []
    covers = set()
    for i, j, M in sample:
        Qi, mask = plans[i]
        tk = time.time()
        got, nc = E.pair_covers(5, i, j, Qi, mask)
        covers.update(got)
        rows.append([i, j, M, len(got), int(nc), time.time() - tk])
    est_cov, est_sec, per = 0.0, 0.0, {}
    for i, (c, M) in per_pivot.items():
        mine = [r for r in rows if r[0] == i]
        if not mine:
            continue
        mc = float(np.mean([r[3] for r in mine]))
        ms = float(np.mean([r[5] for r in mine]))
        per[i] = {"pairs": c, "sampled": len(mine), "covers_per_pair": mc, "seconds_per_pair": ms,
                  "est_covers": mc * c, "est_seconds": ms * c}
        est_cov += mc * c
        est_sec += ms * c
    rec = {"orbit": ORBIT, "n2": N2, "rank": 5, "N": E.N, "group_order": E.info["order"],
           "orbits": int(E.info["orbits"]), "units": len(units), "per_pivot_pairs": {str(k): v for k, v in per_pivot.items()},
           "sampled_pairs": len(rows), "sampled_covers": int(sum(r[3] for r in rows)),
           "sampled_distinct_covers": len(covers), "sampled_candidates": int(sum(r[4] for r in rows)),
           "sampled_seconds": float(sum(r[5] for r in rows)),
           "estimate": {"covers": est_cov, "kernel_seconds": est_sec, "per_pivot": {str(k): v for k, v in per.items()}},
           "kernel": "native" if E.native_cover5 is not None else "reference",
           "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"], "rows": rows,
           "sample_covers": sorted(covers)[:2000]}
    log(f"sample of {len(rows)} pairs: {rec['sampled_covers']} covers ({len(covers)} distinct), "
        f"{rec['sampled_seconds']:.1f}s; estimate {est_cov:.3g} covers and {est_sec:.0f} kernel seconds "
        f"over {len(units)} pairs")
    write("census5_sample.json", rec)
    return 0


# ------------------------------------------------------- degenerate 5 -----

def degenerate5(args):
    """The full 5-multisets of psi_3 whose distinct states are dependent or
    which repeat a state, as in research/h5_rank5/driver.degenerate_covers
    (the H^6 v2 routes), with the span states through the modular in_span."""
    log = logger()
    E = CoverEnumerator(N2, orbit=ORBIT)
    covers3, covers4, c4rep = load_covers4()
    log(f"{len(covers3)} full 3-covers, {len(covers4)} full 4-covers of distinct independent states, "
        f"{len(c4rep)} 4-multisets with a repeated state")

    def ok(ms):
        distinct = sorted(set(ms))
        fam = Family.from_cover(E, distinct)
        if fam is None:
            return False
        exempt = [i for i, u in enumerate(distinct) if ms.count(u) > 1]
        return not fam.has_zero_coefficient(exempt)

    out = {}

    def add(ms, route):
        ms = tuple(sorted(ms))
        if ms not in out:
            out[ms] = route

    for T in covers3:
        span = E.in_span(T)
        log(f"3-cover {T}: {len(span)} states in span(T)")
        pool = list(T) + span
        for extra in itertools.combinations_with_replacement(pool, 2):
            ms = tuple(sorted(T + extra))
            if ok(ms):
                add(ms, "T+pool2")
        F = E.F1
        rows = E.U1.copy()
        for b in T:
            rows, _ = _reduce(F, rows, rows[b])
        Rc, has = _canon_rows(F, rows)
        ids = np.flatnonzero(has)
        key = (Rc[ids] @ E.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
        n_pairs = 0
        for g in _groups_by_key(key):
            for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                n_pairs += 1
                ms = tuple(sorted(T + (a, b)))
                if E.is_cover(ms) and ok(ms):
                    add(ms, "T+parallel-pair")
        in_pool = set(pool)
        n_cancel = 0
        for b in range(E.N):
            if b in in_pool:
                continue
            ms = tuple(sorted(T + (b, b)))
            if ok(ms):
                add(ms, "T+(b,b)")
                n_cancel += 1
        log(f"  parallel pairs tested {n_pairs}; cancel-at-base multisets {n_cancel}")
    n_span4 = 0
    for k, Cv in enumerate(covers4):
        span = E.in_span(Cv)
        n_span4 += len(span)
        for x in list(Cv) + span:
            ms = tuple(sorted(Cv + (x,)))
            if ok(ms):
                add(ms, "C4+x" if x in Cv else "C4+span")
        if (k + 1) % 1000 == 0:
            log(f"  {k + 1}/{len(covers4)} 4-covers, {len(out)} multisets so far")
    log(f"states in the spans of the 4-covers: {n_span4} in all")
    lst = sorted(out)
    by_pattern, by_route = {}, {}
    for ms in lst:
        pat = str(tuple(sorted((ms.count(u) for u in set(ms)), reverse=True)))
        by_pattern[pat] = by_pattern.get(pat, 0) + 1
        by_route[out[ms]] = by_route.get(out[ms], 0) + 1
    kappa = {}
    for ms in lst:
        distinct = sorted(set(ms))
        fam = Family.from_cover(E, distinct)
        key = f"{len(distinct)} distinct, kappa {fam.kappa}"
        kappa[key] = kappa.get(key, 0) + 1
    rec = {"orbit": ORBIT, "n2": N2, "rank": 5, "count": len(lst), "by_pattern": by_pattern, "by_route": by_route,
           "by_distinct_kappa": kappa, "covers": [list(ms) for ms in lst], "routes": [out[ms] for ms in lst]}
    log(f"{len(lst)} degenerate 5-multisets by pattern {by_pattern}, by route {by_route}, by family {kappa}")
    write("degenerate5.json", rec)
    return 0


# --------------------------------------------------------------- rates ----

class Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise Deadline()


def _time_runs(Mt, covers, x0s, cap, log, rank=5, label=""):
    signal.signal(signal.SIGALRM, _alarm)
    rec = {"runs": 0, "seconds": [], "by_x0": {}, "hits": 0, "refused": 0, "undecided": 0, "capped": 0,
           "hist": {}, "native_runs": 0}
    for cover in covers:
        for x0 in x0s:
            t1 = time.time()
            signal.alarm(cap)
            try:
                hits, st = Mt.run(cover, x0)
                if st["refused"]:
                    rec["refused"] += 1
                else:
                    rec["hits"] += sum(1 for h in hits if h["rank"] == rank and h["exact"] and h["independent"]
                                       and h["nonzero"] and h["residual"] < 1e-8)
                    key = ",".join(str(b) for b in st.get("coord_solutions", []))
                    rec["hist"][key] = rec["hist"].get(key, 0) + 1
                    rec["native_runs"] += int(bool(st.get("native")))
            except Deadline:
                rec["capped"] += 1
            except UnpinnedFamily as exc:
                rec["undecided"] += 1
                log(f"  {cover} x0 {x0:02b}: UnpinnedFamily: {exc}")
            finally:
                signal.alarm(0)
            dt = time.time() - t1
            rec["seconds"].append(dt)
            rec["by_x0"].setdefault(str(x0), []).append(dt)
            rec["runs"] += 1
    a = np.array(rec["seconds"]) if rec["seconds"] else np.zeros(1)
    rec["per_run_mean_s"], rec["per_run_median_s"], rec["per_run_max_s"] = float(a.mean()), float(np.median(a)), float(a.max())
    rec["per_x0_mean_s"] = {k: float(np.mean(v)) for k, v in rec["by_x0"].items()}
    log(f"{label}: {len(covers)} covers x {len(x0s)} base points: per run mean {a.mean():.4f}s, median "
        f"{np.median(a):.4f}s, max {a.max():.3f}s; per x0 {rec['per_x0_mean_s']}; hits {rec['hits']}, refused "
        f"{rec['refused']}, undecided {rec['undecided']}, capped {rec['capped']}; native {rec['native_runs']}; "
        f"histogram {dict(sorted(rec['hist'].items(), key=lambda kv: -kv[1])[:8])}")
    rec.pop("seconds")
    rec.pop("by_x0")
    return rec


def rates(args):
    log = logger()
    E = CoverEnumerator(N2, orbit=ORBIT)
    x0s = [int(x) for x in args.x0s.split(",")]
    Mt = SliceMatcher(E, N1)
    log(f"matcher {'native' if Mt.native is not None else 'reference'}; base points {[f'{x:02b}' for x in x0s]}")
    out = {"orbit": ORBIT, "x0s": x0s, "stages": {}}
    # kind A: 5-covers from the sample file (or freshly from the first pivots)
    path = os.path.join(RESULTS, "census5_sample.json")
    if os.path.exists(path):
        with open(path) as f:
            sample = [tuple(c) for c in json.load(f)["sample_covers"]]
    else:
        units, plans = pivot_pairs(E)
        sample = []
        for i, j, M in units[:200]:
            got, _ = E.pair_covers(5, i, j, *plans[i])
            sample.extend(sorted(got))
            if len(sample) >= args.count:
                break
    covers = spread(sample, args.count)
    out["stages"]["A"] = _time_runs(Mt, covers, x0s, args.cap, log, label="kind A (distinct independent, native)")
    write("rates.json", out)
    # kinds B and C from degenerate5.json
    path = os.path.join(RESULTS, "degenerate5.json")
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
        deg = [tuple(c) for c in doc["covers"]]
        by = {}
        for ms in deg:
            pat = tuple(sorted((ms.count(u) for u in set(ms)), reverse=True))
            by.setdefault(pat, []).append(ms)
        for pat, lst in sorted(by.items(), reverse=True):
            n = len(lst) if pat == (3, 1, 1) else args.count_deg
            sel = spread(lst, n)
            r = _time_runs(Mt, sel, x0s, args.cap, log, label=f"pattern {pat} ({len(lst)} in the list, reference)")
            r["count"] = len(sel)
            r["list_size"] = len(lst)
            out["stages"][str(pat)] = r
            write("rates.json", out)
    else:
        log("no degenerate5.json; run degenerate5 first for the kind B and C rates")
    write("rates.json", out)
    return 0


def beta_rate(args):
    log = logger()
    spec = importlib.util.spec_from_file_location("h5_beta", os.path.join(ROOT, "research", "h5_rank5", "beta.py"))
    beta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(beta)
    E = CoverEnumerator(N2, orbit=ORBIT)
    covers3, covers4, c4rep = load_covers4()
    B = beta.BetaMatcher(E)
    x0s = [int(x) for x in args.x0s.split(",")]
    out = {"orbit": ORBIT, "x0s": x0s, "stages": {}}
    sel = spread(covers4, args.count)
    out["stages"]["independent"] = _time_runs(B, sel, x0s, args.cap, log, label=f"beta independent ({len(covers4)} in the list)")
    out["stages"]["repeated"] = _time_runs(B, c4rep, x0s, args.cap, log, label=f"beta repeated ({len(c4rep)} in the list)")
    write("beta_rates.json", out)
    return 0


# ---------------------------------------------------------- census 4 ------

def census4(args):
    log = logger()
    t1 = time.time()
    E4 = CoverEnumerator(4, orbit=ORBIT, native=False)
    t_build = time.time() - t1
    log(f"CoverEnumerator(4, {ORBIT}): N = {E4.N}, group order {E4.info['order']}, {E4.info['orbits']} orbits, "
        f"{len(E4.reps)} pivots, built in {t_build:.1f}s")
    rng = np.random.default_rng(args.seed)
    per_pivot, plans = {}, {}
    t1 = time.time()
    for i in E4.reps:
        i = int(i)
        members, partners = E4.pivot_plan(i)
        per_pivot[i] = (len(partners), len(members))
        plans[i] = (members, partners)
    t_plan = time.time() - t1
    total_pairs = sum(c for c, M in per_pivot.values())
    log(f"{total_pairs} pivot pairs in {t_plan:.1f}s: "
        + ", ".join(f"{i}:{c}/{M}" for i, (c, M) in per_pivot.items()))
    rows = []
    covers = set()
    K = args.pairs
    t_all = time.time()
    for i, (c, M) in per_pivot.items():
        if time.time() - t_all > args.budget:
            log("sample budget reached")
            break
        members, partners = plans[i]
        k = max(1, int(round(K * c / total_pairs)))
        sel = rng.choice(partners, size=min(k, len(partners)), replace=False)
        mask = np.zeros(E4.N, dtype=bool)
        mask[members] = True
        Qi, _ = _reduce(E4.F1, E4.Q1, E4.Q1[i])
        for j in sel:
            tk = time.time()
            got, nc = E4.pair_covers(4, i, int(j), Qi, mask)
            covers.update(got)
            rows.append([i, int(j), M, len(got), int(nc), time.time() - tk])
    est_cov, est_sec, per = 0.0, 0.0, {}
    for i, (c, M) in per_pivot.items():
        mine = [r for r in rows if r[0] == i]
        if not mine:
            continue
        mc = float(np.mean([r[3] for r in mine]))
        ms = float(np.mean([r[5] for r in mine]))
        per[i] = {"pairs": c, "members": M, "sampled": len(mine), "covers_per_pair": mc, "seconds_per_pair": ms,
                  "est_covers": mc * c, "est_seconds": ms * c}
        est_cov += mc * c
        est_sec += ms * c
    sampled_frac = sum(per[i]["pairs"] for i in per) / total_pairs
    rec = {"orbit": ORBIT, "n": 4, "rank": 4, "N": E4.N, "group_order": E4.info["order"], "orbits": int(E4.info["orbits"]),
           "build_seconds": t_build, "plan_seconds": t_plan, "pivot_pairs": total_pairs,
           "per_pivot": {str(i): {"pairs": c, "members": M} for i, (c, M) in per_pivot.items()},
           "sampled_pairs": len(rows), "sampled_covers": int(sum(r[3] for r in rows)),
           "sampled_distinct_covers": len(covers), "sampled_candidates": int(sum(r[4] for r in rows)),
           "sampled_seconds": float(sum(r[5] for r in rows)), "pivots_sampled_fraction_of_pairs": sampled_frac,
           "estimate": {"covers_from_sampled_pivots": est_cov, "python_seconds_from_sampled_pivots": est_sec,
                        "covers_scaled_to_all_pairs": est_cov / max(sampled_frac, 1e-9),
                        "python_seconds_scaled_to_all_pairs": est_sec / max(sampled_frac, 1e-9),
                        "per_pivot": {str(k): v for k, v in per.items()}},
           "columns": ["pivot", "partner", "members", "covers", "candidates", "seconds"], "rows": rows,
           "sample_covers": sorted(covers)[:500]}
    log(f"sample of {len(rows)} pairs: {rec['sampled_covers']} covers ({len(covers)} distinct), "
        f"{rec['sampled_candidates']} candidates, {rec['sampled_seconds']:.1f}s; estimate over the sampled pivots "
        f"({sampled_frac:.0%} of the pairs) {est_cov:.3g} covers and {est_sec:.0f} s; scaled to all pairs "
        f"{rec['estimate']['covers_scaled_to_all_pairs']:.3g} covers and "
        f"{rec['estimate']['python_seconds_scaled_to_all_pairs'] / 3600:.2f} CPU-hours")
    write("census4_sample.json", rec)
    return 0


def products4(args):
    """Full 4-covers of |T>^4 that are products of two full 2-covers of
    |T>^2 (qubits 12 times qubits 34): they exist, so the census of rank-4
    decompositions of |T>^4 is nonempty; checked with the n = 4 enumerator's
    exact cover and fullness tests."""
    log = logger()
    E2 = CoverEnumerator(2, orbit=ORBIT, native=False)
    covers2, _ = E2.covers(2)
    log(f"{len(covers2)} full 2-covers of |T>^2 up to symmetry (group order {E2.info['order']}): {covers2}")
    E4 = CoverEnumerator(4, orbit=ORBIT, native=False)
    key_rows = {E4.codes[k].tobytes(): k for k in range(E4.N)}
    found = []
    for A in covers2:
        for Bc in covers2:
            idx = []
            for a in A:
                for b in Bc:
                    v = np.kron(E2.C[:, a], E2.C[:, b])
                    idx.append(key_rows[exact_codes(v)[0].tobytes()])
            idx = tuple(sorted(idx))
            ok = E4.is_cover(idx) and E4.is_full(idx)
            found.append({"covers2": [list(A), list(Bc)], "indices": [int(k) for k in idx], "full_cover": bool(ok),
                          "orbit_roots": [int(E4.info["roots"][k]) for k in idx]})
    n_ok = sum(f["full_cover"] for f in found)
    log(f"{len(found)} product 4-tuples, {n_ok} full 4-covers of |T>^4; orbit roots of their members: "
        f"{sorted({r for f in found for r in f['orbit_roots']})}")
    write("products4.json", {"orbit": ORBIT, "covers2": [list(c) for c in covers2], "products": found})
    return 0


# ------------------------------------------------------------- rank 2 -----

def _vec_inv(a, p):
    """Modular inverse of an int64 array, elementwise, by Fermat (p < 2^31)."""
    a = np.asarray(a, dtype=np.int64) % p
    result = np.ones_like(a)
    base = a.copy()
    e = p - 2
    while e:
        if e & 1:
            result = (result * base) % p
        base = (base * base) % p
        e >>= 1
    return result


def rank2(args):
    """Residuals tau^j psi_4 - sum_i d_i w_i over code combinations of the
    rank-3 decomposition of |T>^4, each tested for stabilizer rank <= 2 over
    the four-qubit dictionary: v, w span the residual R iff their images
    in F^16 / span(R) are parallel, so their projective keys (two ratios of
    random functionals) agree; every key collision is decided exactly mod
    P2 and over C."""
    sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
    from common import load_decompositions  # noqa: E402  (research/constructions/common.py)
    log = logger()
    D = dictionary(2, 4)
    codes, C = patterns(D)
    N, dim = codes.shape
    F1, F2 = Field(P1, ORBIT), Field(P2, ORBIT)
    U1, U2 = F1.codes_to_field(codes), F2.codes_to_field(codes)
    psi1, psi2 = F1.target(4), F2.target(4)
    decs, rec4 = load_decompositions(ORBIT, 4, 3)
    assert len(decs) == 1
    u, d = decs[0]
    key_rows = {codes[k].tobytes(): k for k in range(N)}
    idx = [key_rows[exact_codes(np.asarray(ui, dtype=complex))[0].tobytes()] for ui in u]
    s1 = _affine_solve_mod(U1[idx].T, psi1, P1)
    s2 = _affine_solve_mod(U2[idx].T, psi2, P2)
    assert s1 is not None and s2 is not None and s1[1].shape[1] == 0 and s2[1].shape[1] == 0
    d1, d2 = s1[0], s2[0]
    psiC = C[:, 0] * 0
    # the complex target scaled as the field does: entry x is tau^|x|
    tau = np.exp(1j * np.pi / 4) * (np.sqrt(3) - 1) / np.sqrt(2)
    psiC = np.array([tau ** bin(x).count("1") for x in range(dim)])
    dC, *_ = np.linalg.lstsq(C[:, idx], psiC, rcond=None)
    assert np.linalg.norm(C[:, idx] @ dC - psiC) < 1e-9
    opts = [TermOptions(C[:, k], 4, F1, F2) for k in idx]
    n_opt = opts[0].m1.shape[0]
    log(f"decomposition indices {idx}, {n_opt} options per term, {n_opt ** 3} combinations per ratio")
    rng = np.random.default_rng(args.seed)
    combos_all = np.array(list(itertools.product(range(n_opt), repeat=3)), dtype=np.int64)
    chunk = args.start is not None
    if chunk:
        combos = combos_all[args.start:args.stop]
    elif args.count < len(combos_all):
        pick = np.sort(rng.choice(len(combos_all), size=args.count, replace=False))
        combos = combos_all[pick]
    else:
        combos = combos_all
    f1, f2, f3 = (rng.integers(1, P1, size=dim) for _ in range(3))
    out = {"decomposition_indices": [int(k) for k in idx], "options_per_term": int(n_opt),
           "combinations_per_ratio": int(n_opt ** 3), "sampled": int(len(combos)), "ratios": {}}
    out["chunk"] = [args.start, args.stop] if chunk else None
    for j in ((args.j,) if args.j else (1, -1)):
        rhs1 = (psi1 * pow(int(F1.ratio), j % (P1 - 1), P1)) % P1
        rhs2 = (psi2 * pow(int(F2.ratio), j % (P2 - 1), P2)) % P2
        rhsC = psiC * tau ** j
        t1 = time.time()
        n_zero, n_stab, n_rank2, n_coll, n_keyzero = 0, 0, 0, 0, 0
        hits = []
        B = args.batch
        for s in range(0, len(combos), B):
            cb = combos[s:s + B]
            R = rhs1[None, :].copy()
            R = (R - sum((d1[i] * opts[i].m1[cb[:, i]]) % P1 for i in range(3))) % P1     # (b, dim)
            nz = R != 0
            zero_rows = ~nz.any(axis=1)
            n_zero += int(zero_rows.sum())
            keep = np.flatnonzero(~zero_rows)
            if len(keep) == 0:
                continue
            R = R[keep]
            cb = cb[keep]
            c = np.argmax(R != 0, axis=1)                                   # (b,)
            Rc = R[np.arange(len(R)), c]
            inv_rc = F1.inv(Rc)                                              # (b,)
            # images of the dictionary modulo span(R): V - (V[:, c] / R_c) R
            Vc = U1[:, c].T                                                  # (b, N)
            coef = (Vc * inv_rc[:, None]) % P1                               # (b, N)
            img = (U1[None, :, :] - coef[:, :, None] * R[:, None, :]) % P1   # (b, N, dim)
            k1 = (img @ f1) % P1                                             # (b, N)
            k2 = (img @ f2) % P1
            k3 = (img @ f3) % P1
            img_zero = ~img.any(axis=2)                                      # v in span(R): R is a stabilizer state
            n_stab += int(img_zero.any(axis=1).sum())
            # projective key of the image: (k2/k1, k3/k1) when k1 != 0; parallel
            # images share k1 = 0, so those rows are keyed by k3/k2 in a
            # disjoint range; k1 = k2 = 0 (probability 1/p^2 per row) is
            # brute-forced below
            kz = (k1 == 0) & ~img_zero
            kzz = kz & (k2 == 0)
            n_keyzero += int(kzz.sum())
            inv1 = F1.inv(np.where(k1 == 0, 1, k1))
            inv2 = F1.inv(np.where(k2 == 0, 1, k2))
            key = ((k2 * inv1) % P1) * P1 + (k3 * inv1) % P1                 # (b, N)
            key = np.where(kz, P1 * P1 + (k3 * inv2) % P1, key)
            key = np.where(img_zero | kzz, -1 - np.arange(N)[None, :], key)  # unique sentinels
            kz = kzz
            order = np.argsort(key, axis=1, kind="stable")
            sk = np.take_along_axis(key, order, axis=1)
            eq = (sk[:, 1:] == sk[:, :-1]) & (sk[:, 1:] >= 0)
            for b, l in zip(*np.nonzero(eq)):
                n_coll += 1
                v, w = int(order[b, l]), int(order[b, l + 1])
                # exact decision mod P2 and over C
                R2 = (rhs2 - sum((d2[i] * opts[i].m2[cb[b, i]]) % P2 for i in range(3))) % P2
                r_mod = _rank_mod(np.array([R2, U2[v], U2[w]], dtype=np.int64), P2)
                RC = rhsC - sum(dC[i] * opts[i].vecs[cb[b, i]] for i in range(3))
                r_num = int(np.linalg.matrix_rank(np.column_stack([RC, C[:, v], C[:, w]]), tol=1e-8))
                if (r_mod <= 2) != (r_num <= 2):
                    raise AssertionError(f"modular and numeric rank disagree on combo {cb[b].tolist()}, states {v}, {w}")
                if r_mod <= 2:
                    n_rank2 += 1
                    hits.append({"combo": cb[b].tolist(), "states": [v, w]})
            # keys equal to zero for the first functional: decide those rows by brute force
            for b in np.flatnonzero(kz.any(axis=1)):
                R2 = (rhs2 - sum((d2[i] * opts[i].m2[cb[b, i]]) % P2 for i in range(3))) % P2
                for v in np.flatnonzero(kz[b]):
                    rows = (U2[None, :, :] - 0) % P2
                    for w in range(N):
                        if w == v:
                            continue
                        if _rank_mod(np.array([R2, U2[v], U2[w]], dtype=np.int64), P2) <= 2:
                            n_rank2 += 1
                            hits.append({"combo": cb[b].tolist(), "states": [int(v), int(w)], "via": "keyzero"})
            if args.verbose and (s // B) % 20 == 0:
                log(f"  j={j}: {s + len(cb)}/{len(combos)} combos, collisions {n_coll}, rank2 {n_rank2}")
        dt = time.time() - t1
        per = dt / len(combos)
        out["ratios"][str(j)] = {"exact": n_zero, "stabilizer": n_stab, "rank2": n_rank2, "collisions": n_coll,
                                 "key_zero_rows": n_keyzero, "seconds": dt, "seconds_per_combo": per,
                                 "projected_full_seconds": per * n_opt ** 3, "hits": hits[:50]}
        log(f"j = {j}: {len(combos)} combos in {dt:.1f}s ({per * 1e3:.2f} ms each; full table "
            f"{per * n_opt ** 3 / 60:.1f} min): exact {n_zero}, stabilizer {n_stab}, rank <= 2: {n_rank2} "
            f"(collisions decided {n_coll}, key-zero rows {n_keyzero})")
    if chunk:
        write(f"rank2_j{args.j}_{args.start}_{args.stop}.json", out)
    else:
        write("rank2_sample.json" if args.count < n_opt ** 3 else "rank2_full.json", out)
    return 0


def rank2_aggregate(args):
    """Sum the chunk records rank2_j*_*.json into rank2_full.json and check
    that the chunks tile the 65^3 combinations for both ratios."""
    import glob
    recs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "rank2_j*_*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    if not recs:
        print("no chunk records")
        return 1
    total = recs[0]["combinations_per_ratio"]
    out = {"combinations_per_ratio": total, "ratios": {}, "chunks": len(recs)}
    for j in ("1", "-1"):
        mine = [r for r in recs if j in r["ratios"]]
        spans = sorted(tuple(r["chunk"]) for r in mine)
        covered = 0
        pos = 0
        for a, b in spans:
            if a != pos:
                raise AssertionError(f"j = {j}: chunks do not tile, gap before {a}")
            pos = b
            covered += b - a
        if pos < total:
            raise AssertionError(f"j = {j}: chunks stop at {pos} of {total}")
        agg = {k: sum(r["ratios"][j][k] for r in mine) for k in ("exact", "stabilizer", "rank2", "collisions",
                                                                 "key_zero_rows", "seconds")}
        agg["hits"] = [h for r in mine for h in r["ratios"][j]["hits"]]
        agg["combos"] = covered
        out["ratios"][j] = agg
        print(f"j = {j}: {covered} combos in {agg['seconds'] / 60:.1f} min: exact {agg['exact']}, stabilizer "
              f"{agg['stabilizer']}, rank <= 2: {agg['rank2']} (collisions decided {agg['collisions']})")
    write("rank2_full.json", out)
    return 0


# ---------------------------------------------------------------- main ----

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("flats").set_defaults(fn=flats)
    p = sub.add_parser("census5")
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--full", action="store_true")
    p.add_argument("--pivots", default=None)
    p.add_argument("--seed", type=int, default=5)
    p.set_defaults(fn=census5)
    sub.add_parser("degenerate5").set_defaults(fn=degenerate5)
    p = sub.add_parser("rates")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--count-deg", type=int, default=6)
    p.add_argument("--x0s", default="0,1,3")
    p.add_argument("--cap", type=int, default=120)
    p.set_defaults(fn=rates)
    p = sub.add_parser("beta-rate")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--x0s", default="0,1")
    p.add_argument("--cap", type=int, default=120)
    p.set_defaults(fn=beta_rate)
    p = sub.add_parser("census4")
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--budget", type=float, default=300.0, help="seconds of sampling before stopping")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=census4)
    sub.add_parser("products4").set_defaults(fn=products4)
    p = sub.add_parser("rank2")
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--j", type=int, default=None, help="one ratio exponent only (1 or -1)")
    p.add_argument("--start", type=int, default=None, help="chunk of the 65^3 combinations [start, stop)")
    p.add_argument("--stop", type=int, default=None)
    p.set_defaults(fn=rank2)
    sub.add_parser("rank2-aggregate").set_defaults(fn=rank2_aggregate)
    a = ap.parse_args(argv[1:])
    try:
        os.nice(19)
    except OSError:
        pass
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
