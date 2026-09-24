"""Lists, rates, partition and controls for the rank-6 exclusion of |N>^4 by
a two-qutrit base slice at the single base point (2, 2)
(docs/notes/n4_rank6_design.md, docs/notes/n4_rank6_exclusion.md). The
batches run through batch.py and are checked by aggregate.py.

Every rank-6 decomposition of |N>^4 has, along qutrits 1, 2 at (2, 2),
four, five or six visible terms (Fact B: at least four terms are nonzero
at every two-qutrit point), so its base is a full k-multiset of |N>^2 with
k in {4, 5, 6} and the 6 - k invisible terms lie on flats of F_3^2 missing
(2, 2). Stage A6, k = 6 distinct independent states: the kernel census of
the 37,201,212 full 6-covers (cover6_pair per pivot pair) through the
compiled stage A matcher. Stage B6, k = 6 distinct dependent states: the
G_2 orbit representatives of the dependent 6-sets (the fresh-term filter
of filters6.py for kappa 1, the reference matcher at a raised candidate
cap for kappa 2 and 3). Stage C6, a repeated state: the orbit
representatives of the repeated 6-multisets through the reference
matcher's block paths. Stage (beta'), k = 5: the orbit representatives of
the full 5-multisets, each with the invisible term on every one of the 16
flats, through invisible3.InvisibleMatcher3.run_one. Stage (gamma), k = 4:
the orbit representatives of the full 4-multisets with every one of the
136 flat multisets, through run_two.

Commands
  lists [--write]           build the orbit-representative lists (B6, C6, k5, k4)
                            from the rank-5 census files and, with --write, store
                            them as reps_N.json with their hash
  control-lists [--census]  re-enumerate the lists from scratch and compare them by
                            hash with reps_N.json; check the 6-cover census file's
                            pivot pairs against the enumerator's (and, with
                            --census, re-run cover6_pair over every pair)
  sample STAGE [--count N] [--budget S]
                            time the matcher on N items of a stage (A6, B6, C6,
                            beta or gamma) on real items; the rates go to
                            results/rates.json
  partition [--target-s S]  write partition.json: batches of about S pod seconds
                            at the sampled rates and the pod factors (compiled
                            1.3, Python 4)
  control-planted [--stage b6|c6|beta|gamma|all] [--plant K] [--seed S]
                            recover planted six-term decompositions: B6 kappa 1
                            through the filter path and kappa 2 at the raised
                            cap, C6 by pattern, stage (beta') at every one of
                            the 16 flats (independent, dependent, repeated and
                            block-hidden bases), stage (gamma) at every one of
                            the 136 flat multisets with the same-state and
                            cancelling kinds on the eight shared lines
  control-witness [--cap S] the rank-7 Lean witness of |N>^4 from every
                            all-visible base at (2, 2), each base reported,
                            an aborted base recorded as aborted
  control-orbit [--count K] the orbit lemma on planted decompositions: a
                            decomposition with base S maps to one with base
                            U S under G_2, found by the matcher, with the
                            same canonical form
  control-m3                the rank-4 decompositions of |N>^3 along a qutrit
                            pair at (2, 2) through the same matchers (n_2 = 1)

Running a batch: `batch.py K`; checking the results: `aggregate.py`.
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
import common  # noqa: E402  (research/n4_rank6/common.py)
from common import (CENSUS6, COMP, E1, E2, FLAT_NAMES, FLAT_PAIRS, FLATS, HERE, M, N1, N2, OFFSETS, PARTITION,  # noqa: E402
                    PTS, POD_FACTOR_COMPILED, POD_FACTOR_PYTHON, RANK, RATES, REPS, RESULTS, ROOT,
                    RUNS_PER_ITEM, X0, add, codes_key, genuine, git_commit, pairs_of, pidx, qcommon)
from invisible3 import InvisibleMatcher3, hist_key, planted_target  # noqa: E402
from matcher import (Budget, BudgetExceeded, Matcher, UnpinnedFamily, exact_codes, family_from,  # noqa: E402
                     move_front, psi_target, slice_base, term_from_codes)
from probe6 import random_shape_term  # noqa: E402
import degenerate6  # noqa: E402
from stages import a6_kernel, run_a6, run_b6, run_c6, set_max_cand  # noqa: E402


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(s):
    print(s, flush=True)


# ------------------------------------------------------------- lists -------

def build_lists(E, seed=23, verbose=True):
    """The orbit-representative lists of stages B6, C6, (beta') and (gamma)
    from the rank-5 census files, each as the sorted array of canonical
    codes (the least image of the sorted tuple under G_2, base 360) with
    the counts by route; the B6 list with the family dimension kappa of
    every orbit."""
    t0 = time.time()
    G = degenerate6.group_perms(E)
    rng = np.random.default_rng(seed)
    cen = qcommon.load_census("N")
    covers3 = [tuple(c) for c in cen["covers3"]]
    covers4 = [tuple(c) for c in cen["covers4"]]
    covers5 = [tuple(c) for c in cen["covers5"]]
    deg, ddoc = qcommon.load_degenerate("N")
    dep5 = [c for c in deg if len(set(c)) == 5]
    rep5 = [c for c in deg if len(set(c)) < 5]
    dep4 = []
    for T in covers3:
        for x in degenerate6.span_states(E, T).tolist():
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                dep4.append(S)
    dep4 = sorted(set(dep4))
    rep4 = sorted({tuple(sorted(T + (u,))) for T in covers3 for u in T})
    out = {"orbit": "N", "N": int(E.N), "group_order": int(len(G)), "census5_sha256": cen["sha256"],
           "degenerate5_sha256": ddoc["sha256"],
           "sources": {"covers3": len(covers3), "covers4": len(covers4), "covers5": len(covers5),
                       "dependent5": len(dep5), "repeated5": len(rep5), "dependent4": len(dep4),
                       "repeated4": len(rep4)}}
    # B6
    routes = degenerate6.build_b6(E, covers3, covers4, covers5, rng) if verbose else \
        _quiet(lambda: degenerate6.build_b6(E, covers3, covers4, covers5, rng))
    allsets = sorted(set(itertools.chain.from_iterable(routes.values())))
    arr = np.array(allsets, dtype=np.int64)
    cover, kappa, full = degenerate6.decide_batch(E, arr)
    keep = full & (kappa >= 1)
    B = arr[keep]
    codesB = degenerate6.canonical_codes(B, G, E.N)
    uniqB, inv = np.unique(codesB, return_inverse=True)
    kB = np.zeros(len(uniqB), dtype=np.int64)
    kB[inv] = kappa[keep]
    # every member of an orbit has the same kappa: check
    chk = np.full(len(uniqB), -1, dtype=np.int64)
    for c, k in zip(inv, kappa[keep]):
        if chk[c] not in (-1, k):
            raise AssertionError("two members of a G_2 orbit have different family dimensions")
        chk[c] = k
    out["B6"] = {"count": int(len(uniqB)), "codes": [int(c) for c in uniqB], "kappa": [int(k) for k in kB],
                 "by_route_pairs": {k: len(v) for k, v in routes.items()},
                 "distinct_sets_candidates": int(len(arr)), "distinct_sets_full_dependent": int(len(B)),
                 "kappa_histogram_orbits": {str(k): int(v) for k, v in zip(*np.unique(kB, return_counts=True))}}
    if verbose:
        log(f"B6: {len(B)} full dependent 6-sets, {len(uniqB)} orbits, kappa {out['B6']['kappa_histogram_orbits']} "
            f"[{time.time() - t0:.0f}s]")
    # C6
    croutes = degenerate6.build_c6(E, covers3, covers4, covers5, dep4, dep5)
    keys = sorted(set(itertools.chain.from_iterable(croutes.values())))
    arrc = np.array(keys, dtype=np.int64)
    uniqC = np.unique(degenerate6.canonical_codes(arrc, G, E.N))
    pats = {}
    for row in common.decode(uniqC, 6):
        p = str(common.multiplicity_pattern(row.tolist()))
        pats[p] = pats.get(p, 0) + 1
    out["C6"] = {"count": int(len(uniqC)), "codes": [int(c) for c in uniqC],
                 "by_route_multisets": {k: len(v) for k, v in croutes.items()},
                 "distinct_multisets": int(len(arrc)), "by_pattern_orbits": pats}
    if verbose:
        log(f"C6: {len(arrc)} distinct multisets, {len(uniqC)} orbits, by pattern {pats} [{time.time() - t0:.0f}s]")
    # k5, k4
    all5 = sorted(set(covers5) | set(dep5) | set(rep5))
    uniq5 = np.unique(degenerate6.canonical_codes(np.array(all5, dtype=np.int64), G, E.N))
    all4 = sorted(set(covers4) | set(dep4) | set(rep4))
    uniq4 = np.unique(degenerate6.canonical_codes(np.array(all4, dtype=np.int64), G, E.N))
    out["k5"] = {"count": int(len(uniq5)), "codes": [int(c) for c in uniq5], "multisets": len(all5)}
    out["k4"] = {"count": int(len(uniq4)), "codes": [int(c) for c in uniq4], "multisets": len(all4)}
    if verbose:
        log(f"k5: {len(all5)} multisets, {len(uniq5)} orbits; k4: {len(all4)} multisets, {len(uniq4)} orbits "
            f"[{time.time() - t0:.0f}s]")
    out["seconds"] = time.time() - t0
    return out


def _quiet(fn):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def lists(args):
    E = common.make_enumerator()
    out = build_lists(E)
    out["git"] = git_commit()
    out["generated"] = _now()
    if args.write:
        sha = common.write_hashed(REPS, out)
        log(f"wrote {os.path.relpath(REPS, ROOT)} (sha256 {sha[:16]}): B6 {out['B6']['count']}, C6 {out['C6']['count']}, "
            f"k5 {out['k5']['count']}, k4 {out['k4']['count']}")
    return 0


def control_lists(args):
    """Fresh lists against the stored ones by content; the 6-cover census
    file's pivot pairs against the enumerator's; with --census the kernel
    census re-run pair by pair."""
    E = common.make_enumerator()
    t0 = time.time()
    fresh = build_lists(E, verbose=False)
    stored, doc = common.load_reps()
    ok = True
    rec = {"git": git_commit(), "generated": _now(), "reps_sha256": doc["sha256"], "lists": {}}
    for key in ("B6", "C6", "k5", "k4"):
        same = fresh[key]["codes"] == doc[key]["codes"]
        if key == "B6":
            same = same and fresh["B6"]["kappa"] == doc["B6"]["kappa"]
        rec["lists"][key] = {"fresh": fresh[key]["count"], "stored": doc[key]["count"], "equal": same}
        ok &= same
        log(f"{key}: fresh {fresh[key]['count']} orbits, stored {doc[key]['count']}: {'equal' if same else 'DIFFERENT'}")
    cen = common.load_census6()
    units = pairs_of(E)
    okc = [[r[0], r[1]] for r in cen["rows"]] == [[i, j] for i, j, _ in units] and cen["pairs"] == len(units) \
        and cen["full6"] == sum(r[3] for r in cen["rows"]) and cen["full6"] == cen["full6_independent"]
    rec["census"] = {"file_sha256": common.sha256_file(CENSUS6), "pairs": len(cen["rows"]), "covers": cen["full6"],
                     "units_equal": okc}
    log(f"census {os.path.relpath(CENSUS6, ROOT)}: {len(cen['rows'])} rows, {cen['full6']} full 6-covers, pivot pairs "
        f"{'equal to' if okc else 'DIFFERENT from'} the enumerator's {len(units)}")
    ok &= okc
    if args.census:
        t1 = time.time()
        mism = _recensus(E, cen)
        rec["census"]["recount_mismatches"] = mism
        rec["census"]["recount_seconds"] = time.time() - t1
        log(f"census re-run: {len(cen['rows'])} pairs in {time.time() - t1:.0f}s, {len(mism)} pairs with a different "
            f"full 6-cover count")
        ok &= not mism
    rec["seconds"] = time.time() - t0
    rec["pass"] = ok
    _write_control("control_lists", rec)
    log(f"control-lists: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _recensus(E, cen):
    """cover6_pair over every pivot pair; the pairs whose full 6-cover
    count differs from the stored census row."""
    from cover_census import _reduce
    mism = []
    plans = {}
    for row in cen["rows"]:
        i, j = int(row[0]), int(row[1])
        if i not in plans:
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            plans[i] = (Qi, mask)
        Qi, mask = plans[i]
        covers, _, _ = E.pair_covers(6, i, j, Qi, mask)
        if len(covers) != row[3]:
            mism.append([i, j, len(covers), row[3]])
    return mism


def _write_control(name, doc):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    log(f"wrote {os.path.relpath(path, ROOT)}")


# ------------------------------------------------------------- rates -------

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


def stage_items(stage, lists):
    """The items of a list stage as tuples, with their cost classes."""
    if stage == "B6":
        items = [tuple(int(u) for u in row) for row in lists["B6"]]
        classes = [f"kappa {int(k)}" for k in lists["B6_kappa"]]
        return items, classes
    key = {"C6": "C6", "beta": "k5", "gamma": "k4"}[stage]
    items = [tuple(int(u) for u in row) for row in lists[key]]
    return items, None


def classify(E, stage, items, classes=None):
    if classes is not None:
        return classes
    return [common.item_class(E, it, stage) for it in items]


def sample(args):
    """Time the matcher of one stage on items spread through its list (or
    on covers of random pivot pairs for A6), every run of the item, under a
    wall-clock budget; the per-class means go to results/rates.json."""
    E = common.make_enumerator()
    Mt = common.new_matcher(E, native=True)
    target = common.target_of(E)
    stage = args.stage
    rec = {"stage": stage, "count": 0, "runs": 0, "hits": 0, "refused": 0, "undecided": 0, "hist": {},
           "by_class": {}, "generated": _now(), "git": git_commit(), "budget_s": args.budget}
    t_all = time.time()
    if stage == "A6":
        cen = common.load_census6()
        rng = np.random.default_rng(args.seed)
        rows = [cen["rows"][t] for t in rng.choice(len(cen["rows"]), size=min(args.pairs, len(cen["rows"])), replace=False)]
        covers = []
        from cover_census import _reduce
        tk = time.time()
        for row in rows:
            i, j = int(row[0]), int(row[1])
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            got, _, _ = E.pair_covers(6, i, j, Qi, mask)
            covers.extend(sorted(got))
            if len(covers) >= args.count:
                break
        covers = covers[:args.count]
        rec["kernel_s_per_cover"] = (time.time() - tk) / max(1, len(covers))
        k = a6_kernel(Mt, target)
        for c in covers[:400]:                       # warm the option cache
            run_a6(Mt, k, target, c)
        secs, agree = [], 0
        t0 = time.time()
        for c in covers:
            hits, st = run_a6(Mt, k, target, c)
            rec["runs"] += 1
            rec["hits"] += sum(genuine(h, RANK) for h in hits)
            rec["refused"] += int(st["refused"])
            key = ",".join(str(b) for b in st["coord_raw"])
            rec["hist"][key] = rec["hist"].get(key, 0) + 1
        dt = time.time() - t0
        rec["count"] = len(covers)
        rec["per_item_mean_s"] = dt / max(1, len(covers))
        # agreement of the lean loop with Matcher.run on the first items
        for c in covers[:args.reference]:
            h1, s1 = run_a6(Mt, k, target, c)
            h2, s2 = Mt.run(c, X0, target)
            agree += (sorted(codes_key(h["terms"]) for h in h1) == sorted(codes_key(h["terms"]) for h in h2)
                      and s1["coord_raw"] == s2["coord_raw"] and s1["refused"] == s2["refused"])
        rec["reference_checked"] = min(args.reference, len(covers))
        rec["reference_agree"] = agree
        rec["by_class"] = {"A6": {"items": len(covers), "mean_s": rec["per_item_mean_s"], "dense_mean_s": 0.0}}
        log(f"A6: {len(covers)} covers from {len(rows)} random pairs: {1e3 * rec['per_item_mean_s']:.3f} ms per cover, "
            f"kernel {1e3 * rec['kernel_s_per_cover']:.3f} ms per cover; hits {rec['hits']}, refused {rec['refused']}; "
            f"Matcher.run agrees on {agree}/{rec['reference_checked']}; hist {dict(list(rec['hist'].items())[:5])}")
    else:
        lists, _ = common.load_reps()
        items, classes = stage_items(stage, lists)
        classes = classify(E, stage, items, classes)
        groups = {}
        for it, cl in zip(items, classes):
            groups.setdefault(cl, []).append(it)
        IM = InvisibleMatcher3(Mt) if stage in ("beta", "gamma") else None
        Fl = None
        if stage == "B6":
            from filters6 import Filters
            Fl = Filters(Mt)
        per_class_budget = args.budget / max(1, len(groups))
        for cl, lst in sorted(groups.items()):
            n = args.count if stage != "gamma" else args.count
            picks = _spread(lst, n)
            secs, dsecs, first = [], [], {}
            t_cl = time.time()
            for it in picks:
                if secs and time.time() - t_cl > per_class_budget:
                    break
                t1 = time.time()
                dense = 0
                units = [None] if stage in ("B6", "C6") else (FLAT_NAMES if stage == "beta" else FLAT_PAIRS)
                # a per-item cap: an item past it is a tail item (counted, excluded from the rate the
                # partition uses, left to the batch guard), not a stall of the whole sample
                Mt.budget = Budget(seconds=args.item_cap) if args.item_cap else None
                if IM is not None:
                    IM.deadline = time.time() + args.item_cap if args.item_cap else None
                for unit in units:
                    try:
                        if stage == "B6":
                            kap = int(cl.split()[1])
                            cap = 2_000_000 if kap == 1 else 200_000_000
                            set_max_cand(cap)
                            hits, st, extra = run_b6(Mt, Fl, target, it, kap)
                            dense += extra.get("dense_raw_s", 0.0)
                        elif stage == "C6":
                            hits, st = run_c6(Mt, target, it)
                        elif stage == "beta":
                            hits, st = IM.run_one(it, unit, target)
                        else:
                            hits, st = IM.run_two(it, unit, target)
                    except BudgetExceeded as exc:
                        rec["capped"] = rec.get("capped", 0) + 1
                        log(f"  {it} {unit}: capped at {args.item_cap}s ({str(exc)[:80]})")
                        break
                    except UnpinnedFamily as exc:
                        rec["undecided"] += 1
                        log(f"  {it} {unit}: {type(exc).__name__}: {str(exc)[:100]}")
                        continue
                    rec["runs"] += 1
                    if st["refused"]:
                        rec["refused"] += 1
                        continue
                    rec["hits"] += sum(genuine(h, RANK) for h in hits)
                    key = hist_key(stage, st) if stage in ("beta", "gamma") else ",".join(str(b) for b in st["coord_raw"])
                    rec["hist"][key] = rec["hist"].get(key, 0) + 1
                secs.append(time.time() - t1)
                dsecs.append(dense)
                rec["count"] += 1
            Mt.budget = None
            if IM is not None:
                IM.deadline = None
            if not secs:
                continue
            a = np.array(secs)
            cut = min(60.0, args.item_cap * 0.9) if args.item_cap else 60.0
            body = a[a <= cut]                       # the estimate the partition uses; the tail is guarded by --max-seconds
            rec["by_class"][cl] = {"items": len(lst), "sampled": len(secs), "mean_s": float(a.mean()),
                                   "median_s": float(np.median(a)), "max_s": float(a.max()),
                                   "trimmed_mean_s": float(body.mean()) if len(body) else float(a.mean()),
                                   "tail_items": int((a > cut).sum()), "tail_cut_s": cut,
                                   "dense_mean_s": float(np.mean(dsecs))}
            log(f"{stage} {cl}: {len(lst)} items, {len(secs)} sampled: per item mean {a.mean():.4f}s, median "
                f"{np.median(a):.4f}s, max {a.max():.3f}s, trimmed mean {rec['by_class'][cl]['trimmed_mean_s']:.4f}s "
                f"({rec['by_class'][cl]['tail_items']} tail items)"
                + (f", dense {np.mean(dsecs):.4f}s" if stage == "B6" else "") + f" [{time.time() - t_cl:.0f}s]")
        log(f"{stage}: {rec['count']} items, {rec['runs']} runs, hits {rec['hits']}, refused {rec['refused']}, undecided "
            f"{rec['undecided']}; hist {dict(sorted(rec['hist'].items(), key=lambda kv: -kv[1])[:8])}")
    rec["wall_s"] = time.time() - t_all
    os.makedirs(RESULTS, exist_ok=True)
    rates = _load_rates()
    rates["stages"][stage] = rec
    rates["git"] = git_commit()
    with open(RATES, "w") as f:
        json.dump(rates, f, indent=1)
    with open(os.path.join(RESULTS, f"sample_{stage}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return 0


# ---------------------------------------------------------- partition ------

def pod_seconds(cls_rate, compiled_share_s=0.0):
    """Laptop seconds to pod seconds: the compiled share at 1.3, the rest
    at 4 (docs/notes/n4_rank6_design.md, section 7)."""
    return POD_FACTOR_COMPILED * compiled_share_s + POD_FACTOR_PYTHON * max(0.0, cls_rate - compiled_share_s)


def partition(args):
    E = common.make_enumerator()
    units = pairs_of(E)
    cen = common.load_census6()
    rates = _load_rates()["stages"]
    for st in common.STAGES:
        if st not in rates:
            raise SystemExit(f"no rate for stage {st} in {RATES}; run `driver.py sample {st}` first")
    lists, rdoc = common.load_reps()
    by = {(r[0], r[1]): r for r in cen["rows"]}
    if [(i, j) for i, j, _ in units] != [(r[0], r[1]) for r in cen["rows"]]:
        raise SystemExit("the census rows are not the enumerator's pivot pairs")
    a_rate = rates["A6"]["per_item_mean_s"]
    costs = [POD_FACTOR_COMPILED * (by[(i, j)][6] + a_rate * by[(i, j)][3]) for i, j, _ in units]
    geometry, cur, acc = [], [], 0.0
    for u, c in zip(units, costs):
        cur.append(list(u))
        acc += c
        if acc >= args.target_s:
            geometry.append({"index": len(geometry), "stage": "A6", "units": cur, "estimated_s": round(acc, 1)})
            cur, acc = [], 0.0
    if cur:
        geometry.append({"index": len(geometry), "stage": "A6", "units": cur, "estimated_s": round(acc, 1)})
    est = {"A6": float(sum(costs))}
    counts = {"A6": len(geometry)}
    cost_model = {"pod_factor_compiled": POD_FACTOR_COMPILED, "pod_factor_python": POD_FACTOR_PYTHON,
                  "target_s": args.target_s, "rates": os.path.relpath(RATES, HERE), "a6_s_per_cover": a_rate,
                  "pod_s_per_item_by_class": {}}
    list_meta = {}
    for stage in ("B6", "C6", "beta", "gamma"):
        items, classes = stage_items(stage, lists)
        classes = classify(E, stage, items, classes)
        cls_rate = {}
        for cl, v in rates[stage]["by_class"].items():
            # the mean over the sampled items of at most 60 s: a tail item (the (2, 2, 1, 1) class of
            # C6 had one at 325 s among 40) is left to the batch guard and the resume, as in the
            # rank-5 partition, instead of inflating every batch of its class
            cls_rate[cl] = pod_seconds(v.get("trimmed_mean_s", v["mean_s"]),
                                       v.get("dense_mean_s", 0.0) if stage == "B6" else 0.0)
        missing = sorted(set(classes) - set(cls_rate))
        if missing:
            raise SystemExit(f"stage {stage}: no sampled rate for classes {missing}")
        cost_model["pod_s_per_item_by_class"][stage] = cls_rate
        groups = {}
        for k, cl in enumerate(classes):
            groups.setdefault(cl, []).append(k)
        est[stage] = 0.0
        counts[stage] = 0
        list_meta[stage] = {"count": len(items), "by_class": {cl: len(ids) for cl, ids in groups.items()}}
        for cl, ids in sorted(groups.items()):
            rate = cls_rate[cl]
            total = rate * len(ids)
            est[stage] += total
            if stage == "B6" and cl == "kappa 3":
                for k in ids:
                    geometry.append({"index": len(geometry), "stage": stage, "item_ids": [k], "class": cl,
                                     "max_cand": args.max_cand_high, "estimated_s": round(rate, 1)})
                    counts[stage] += 1
                continue
            n = max(1, math.ceil(total / args.target_s))
            if stage in ("B6", "C6"):
                n = max(n, math.ceil(len(ids) / args.max_items))
            for b in range(n):
                geo = {"index": len(geometry), "stage": stage, "item_ids": ids[b::n], "class": cl,
                       "estimated_s": round(rate * len(ids[b::n]), 1)}
                if stage == "B6":
                    geo["max_cand"] = 2_000_000 if cl == "kappa 1" else args.max_cand_high
                geometry.append(geo)
            counts[stage] += n
    est["total"] = sum(v for k, v in est.items())
    rec = {"orbit": "N", "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "x0": list(X0), "flats": {k: [list(p) for p in v] for k, v in FLATS.items()},
           "flat_pairs": [list(p) for p in FLAT_PAIRS], "runs_per_item": RUNS_PER_ITEM,
           "pivot_orbits": int(E.info["orbits"]), "units": len(units),
           "stage_batches": counts, "cost_model": cost_model, "estimated_s": est,
           "census": {"file": os.path.relpath(CENSUS6, HERE), "sha256": common.sha256_file(CENSUS6),
                      "covers": cen["full6"], "pairs": cen["pairs"]},
           "reps": {"file": os.path.relpath(REPS, HERE), "sha256": rdoc["sha256"],
                    **{k: rdoc[k]["count"] for k in ("B6", "C6", "k5", "k4")},
                    "B6_kappa_histogram": rdoc["B6"]["kappa_histogram_orbits"], "lists": list_meta},
           "batches": len(geometry), "git": git_commit(), "generated": _now(), "batch_geometry": geometry}
    sha = common.write_hashed(PARTITION, rec)
    log(f"{len(units)} pivot pairs in {counts['A6']} stage A6 batches ({est['A6'] / 3600:.2f} pod CPU-h), "
        f"{counts['B6']} B6 ({est['B6'] / 3600:.2f}), {counts['C6']} C6 ({est['C6'] / 3600:.2f}), {counts['beta']} beta "
        f"({est['beta'] / 3600:.2f}), {counts['gamma']} gamma ({est['gamma'] / 3600:.3f}); {len(geometry)} batches, "
        f"{est['total'] / 3600:.1f} pod CPU-h; wrote {os.path.relpath(PARTITION, ROOT)} (sha256 {sha[:16]})")
    return 0


# ------------------------------------------------------------ controls -----

KIND = {(1, 0): "line1", (0, 1): "line2", (1, 1): "diag1", (1, 2): "diag2"}


def flat_direction(pts):
    if len(pts) == 1:
        return None
    d = ((pts[1][0] - pts[0][0]) % 3, (pts[1][1] - pts[0][1]) % 3)
    for v in KIND:
        if d == v or d == (2 * v[0] % 3, 2 * v[1] % 3):
            return v
    raise AssertionError("not a line")


def dbl(o, k):
    return int(o.cls[k, 0, pidx((2, 0))])


def invisible_term(Mt, flat, rng, v=None):
    """A random stabilizer term on the flat (a point term, or a line term
    along the flat with random class and phases), with the dictionary
    state v (random when None) at the flat's first point."""
    pts = FLATS[flat]
    v = int(rng.integers(0, Mt.N)) if v is None else int(v)
    ov = Mt.options(v)
    if len(pts) == 1:
        t = np.zeros((9, 9), dtype=complex)
        t[pidx(pts[0])] = ov.u
        return t.ravel(), v
    t, _ = random_shape_term(ov, pts[0], rng, kind=KIND[flat_direction(pts)])
    return t, v


def line_pair(Mt, flat, rng, kind):
    """Two line terms on the same line flat sharing their slice at the
    line's first point (the point the matcher processes first): kinds
    same (different classes at the second point), same1 (one class,
    different phases), same2 (equal at the first two points, different
    phases at the third), cancel, cancel1, cancel0 (the same three shapes
    with cancelling coefficients), samec (one class at the second point
    with phases and coefficients cancelling there). Returns (t4, t5,
    coefficient relation) with the relation a function of c_4 giving c_5."""
    pts = FLATS[flat]
    y0 = min(pts)
    others = sorted(p for p in pts if p != y0)
    v = int(rng.integers(0, Mt.N))
    ov = Mt.options(v)
    K = ov.nclass
    if kind in ("same", "cancel"):
        k1, k2 = rng.choice(K, size=2, replace=False)
        l1, l2 = rng.integers(3), rng.integers(3)
    else:
        k1 = k2 = int(rng.integers(K))
        l1 = int(rng.integers(3))
        l2 = (l1 + int(rng.integers(1, 3))) % 3 if kind not in ("same2", "cancel0") else l1
    m1, m2 = int(rng.integers(3)), int(rng.integers(3))
    if kind in ("same2", "cancel0"):
        m2 = (m1 + int(rng.integers(1, 3))) % 3
    t4, t5 = np.zeros((9, 9), dtype=complex), np.zeros((9, 9), dtype=complex)
    t4[pidx(y0)] = t5[pidx(y0)] = ov.u
    t4[pidx(others[0])] = ov.vecs[3 * int(k1) + int(l1)]
    t5[pidx(others[0])] = ov.vecs[3 * int(k2) + int(l2)]
    t4[pidx(others[1])] = ov.vecs[3 * dbl(ov, int(k1)) + m1]
    t5[pidx(others[1])] = ov.vecs[3 * dbl(ov, int(k2)) + m2]
    W = np.exp(2j * np.pi / 3)
    if kind.startswith("cancel"):
        rel = lambda c4: -c4                                # noqa: E731
    elif kind == "samec":
        rel = lambda c4: -c4 * W ** ((int(l1) - int(l2)) % 3)     # noqa: E731
    else:
        rel = None
    return t4.ravel(), t5.ravel(), rel


def plant(Mt, base, invisible, rng, hidden=None):
    """Planted terms over the base multiset (random shapes with the base
    slices at X0, a block's copies distinct) plus the invisible terms, with
    integer coefficients (a coefficient relation applied when given);
    retried until the six terms are independent. `hidden`: a block state
    whose copies' slice at the invisible term's first point spans the
    invisible term's slice there (the block-hidden kind)."""
    inv_terms = [t for t, _ in invisible]
    for _ in range(200):
        vis = [random_shape_term(Mt.options(u), X0, rng)[0] for u in base]
        terms = vis + inv_terms
        T = np.column_stack(terms)
        if np.linalg.matrix_rank(T, tol=1e-8) < len(terms):
            continue
        coeffs = (rng.integers(1, 4, size=len(terms)) * rng.choice([1, -1], size=len(terms))).astype(complex)
        for t, rel in invisible:
            if rel is not None:
                j = [id(x) for x in terms].index(id(t))
                coeffs[j] = rel(coeffs[j - 1])
        vec = (T @ coeffs).reshape(9, -1)
        if np.any(np.linalg.norm(vec, axis=1) < 1e-9):
            continue                # a zero slice: no real target has one, and the block-only path assumes none
        if sorted(slice_base(Mt, np.column_stack(vis), X0)) != sorted(base):
            raise AssertionError("planted base differs from the intended one")
        return terms, coeffs
    raise PlantFailed("no independent planted instance found")


class PlantFailed(Exception):
    pass


def control_planted(args):
    """Planted six-term decompositions recovered by the stage matchers
    (module note). Every planted decomposition must be among the hits."""
    E = common.make_enumerator()
    Mt = common.new_matcher(E, native=True)
    rng = np.random.default_rng(args.seed)
    lists, _ = common.load_reps()
    stages = ("b6", "c6", "beta", "gamma") if args.stage == "all" else (args.stage,)
    only = set(args.only.split(",")) if args.only else None
    kinds_only = set(args.kinds.split(",")) if args.kinds else None
    report = {"seed": args.seed, "git": git_commit(), "generated": _now(), "plants": [], "only": args.only,
              "kinds": args.kinds, "cap_s": args.cap}
    n_ok = total = 0
    t_all = time.time()

    def record(stage, unit, kind, base, terms, coeffs, fn):
        nonlocal n_ok, total
        tgt = planted_target(terms, coeffs, Mt.F1, Mt.F2)
        t0 = time.time()
        total += 1
        entry = {"stage": stage, "unit": unit, "kind": kind, "base": [int(u) for u in base]}
        Mt.budget = Budget(seconds=args.cap)
        IM.deadline = time.time() + args.cap
        try:
            hits, st = fn(tgt)
        except (UnpinnedFamily, BudgetExceeded) as exc:
            entry.update({"status": "undecided", "reason": str(exc)[:200], "seconds": time.time() - t0})
            log(f"  {stage} {unit} {kind} base {base}: UNDECIDED {str(exc)[:120]}")
        else:
            want = codes_key(terms)
            same = any(codes_key(h["terms"]) == want for h in hits)
            n_ok += same
            entry.update({"status": "recovered" if same else "not recovered", "hits": len(hits),
                          "seconds": time.time() - t0,
                          "stats": {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool, list))}})
            log(f"  {stage} {unit} {kind} base {base}: {len(hits)} hits, planted {'recovered' if same else 'NOT recovered'}, "
                f"{time.time() - t0:.2f}s")
        report["plants"].append(entry)
        Mt.budget = None

    IM = InvisibleMatcher3(Mt)
    if "b6" in stages:
        from filters6 import Filters
        Fl = Filters(Mt)
        kB = lists["B6_kappa"]
        for kap, count in ((1, args.plant * 3), (2, args.plant)):
            reps = [tuple(int(u) for u in lists["B6"][i]) for i in np.flatnonzero(kB == kap)]
            for _ in range(count):
                base = reps[int(rng.integers(len(reps)))]
                terms, coeffs = plant(Mt, base, [], rng)
                set_max_cand(2_000_000 if kap == 1 else 200_000_000)
                record("b6", f"kappa {kap}", "filter" if kap == 1 else "reference", base, terms, coeffs,
                       lambda tgt: run_b6(Mt, Fl, tgt, base, kap)[:2])
        set_max_cand(2_000_000)
    if "c6" in stages:
        pats = {}
        for row in lists["C6"]:
            pats.setdefault(common.item_class(E, row, "C6"), []).append(tuple(int(u) for u in row))
        for cl, reps in sorted(pats.items()):
            if cl.startswith(("(2, 2, 1, 1)", "(3, 2, 1)", "(4, 1, 1)")):
                # two blocks, or a block of four copies: a planted target keeps every slice
                # alive (the blocks span most of the slice space) and the run exceeds the cap,
                # as the rank-5 note's two-block plants did; the real bases of these classes
                # run at sub-second rates and any raise is recorded as undecided. Not planted.
                report["plants"].append({"stage": "c6", "unit": cl, "kind": "reference", "status": "not planted",
                                         "reason": "two blocks or a block of four: the planted target keeps every "
                                                   "slice alive past the cap; the real bases are run and any raise "
                                                   "is recorded as undecided"})
                continue
            for _ in range(args.plant):
                base = reps[int(rng.integers(len(reps)))]
                terms, coeffs = plant(Mt, base, [], rng)
                record("c6", cl, "reference", base, terms, coeffs, lambda tgt: run_c6(Mt, tgt, base))
    if "beta" in stages:
        k5 = [tuple(int(u) for u in row) for row in lists["k5"]]
        cls5 = {}
        for it in k5:
            cls5.setdefault(common.item_class(E, it, "beta"), []).append(it)
        indep = cls5["(1, 1, 1, 1, 1)"]
        dep = cls5["(1, 1, 1, 1, 1) dependent"]
        rep = [it for cl, lst in cls5.items() if cl.startswith("(2, 1, 1, 1)") for it in lst]
        for flat in FLAT_NAMES:
            if only is not None and flat not in only:
                continue
            for _ in range(args.plant):
                for kind in ("a", "d", "r", "h"):
                    if kinds_only is not None and kind not in kinds_only:
                        continue
                    if kind == "a":
                        base = indep[int(rng.integers(len(indep)))]
                        inv = invisible_term(Mt, flat, rng)
                        terms, coeffs = plant(Mt, base, [(inv[0], None)], rng)
                    elif kind == "d":
                        base = dep[int(rng.integers(len(dep)))]
                        inv = invisible_term(Mt, flat, rng)
                        terms, coeffs = plant(Mt, base, [(inv[0], None)], rng)
                    elif kind == "r":
                        base = rep[int(rng.integers(len(rep)))]
                        inv = invisible_term(Mt, flat, rng)
                        terms, coeffs = plant(Mt, base, [(inv[0], None)], rng)
                    else:
                        # block-hidden: the invisible term's slice at its first point is a
                        # translate of a block copy's slice there
                        base = rep[int(rng.integers(len(rep)))]
                        b = next(u for u in base if base.count(u) > 1)
                        for _try in range(50):
                            vis = [random_shape_term(Mt.options(u), X0, rng)[0] for u in base]
                            copy = vis[base.index(b)].reshape(9, 9)
                            y0 = min(FLATS[flat])
                            s = copy[pidx(y0)]
                            if np.linalg.norm(s) < 1e-9:
                                continue
                            vh = Mt.index_of(s)
                            inv = invisible_term(Mt, flat, rng, v=vh)
                            terms = vis + [inv[0]]
                            if np.linalg.matrix_rank(np.column_stack(terms), tol=1e-8) == 6:
                                break
                        else:
                            continue
                        coeffs = (rng.integers(1, 4, size=6) * rng.choice([1, -1], size=6)).astype(complex)
                    record("beta", flat, kind, base, terms, coeffs, lambda tgt: IM.run_one(base, flat, tgt))
    if "gamma" in stages:
        k4 = [tuple(int(u) for u in row) for row in lists["k4"]]
        indep4 = [it for it in k4 if common.item_class(E, it, "gamma") == "(1, 1, 1, 1)"]
        for pair in FLAT_PAIRS:
            if only is not None and "+".join(pair) not in only:
                continue
            F4, F5 = FLATS[pair[0]], FLATS[pair[1]]
            kinds = ["a"]
            if F4 == F5 and len(F4) == 3:
                kinds += ["same", "same1", "same2", "samec", "cancel", "cancel1", "cancel0"]
            if len(F4) == 1 and len(F5) == 3 and F4[0] in F5:
                kinds.append("ray")
            for kind in kinds:
                if kinds_only is not None and kind not in kinds_only:
                    continue
                base = indep4[int(rng.integers(len(indep4)))]
                if kind == "a":
                    i4, i5 = invisible_term(Mt, pair[0], rng), invisible_term(Mt, pair[1], rng)
                    invs = [(i4[0], None), (i5[0], None)]
                elif kind == "ray":
                    t5, v5 = invisible_term(Mt, pair[1], rng)
                    t4 = np.zeros((9, 9), dtype=complex)
                    t4[pidx(F4[0])] = t5.reshape(9, 9)[pidx(F4[0])]
                    invs = [(t4.ravel(), None), (t5, None)]
                else:
                    t4, t5, rel = line_pair(Mt, pair[0], rng, kind)
                    invs = [(t4, None), (t5, rel)]
                try:
                    terms, coeffs = plant(Mt, base, invs, rng)
                except PlantFailed:
                    continue                          # two equal random terms: not an instance
                record("gamma", "+".join(pair), kind, base, terms, coeffs, lambda tgt: IM.run_two(base, pair, tgt))
    IM.deadline = None
    report["recovered"], report["total"] = n_ok, total
    report["by_stage"] = {st: [sum(1 for e in report["plants"] if e["stage"] == st and e["status"] == "recovered"),
                               sum(1 for e in report["plants"] if e["stage"] == st)] for st in ("b6", "c6", "beta", "gamma")}
    report["seconds"] = time.time() - t_all
    report["pass"] = n_ok == total
    name = "control_planted" if args.stage == "all" else f"control_planted_{args.stage}"
    if args.kinds:
        name += "_" + "".join(ch for ch in args.kinds if ch.isalnum())
    _write_control(name, report)
    log(f"control-planted ({args.stage}): {n_ok}/{total} planted decompositions recovered in {time.time() - t_all:.0f}s; "
        f"by stage {report['by_stage']}; {'PASS' if n_ok == total else 'FAIL'}")
    return 0 if n_ok == total else 1


def control_witness(args):
    """The rank-7 Lean witness sliced along every qutrit pair at (2, 2):
    every base is all-visible (every fixed coordinate of the witness's
    terms is 2), a 7-multiset; the matcher at rank 7 must return the
    witness among its hits. Every base is reported; a base that raises or
    exceeds the cap is recorded as aborted and fails the control."""
    E = common.make_enumerator()
    terms, doc = common.witness_terms()
    cc = common.qcommon  # noqa: F841
    from matcher import constructions_common
    psi = constructions_common().target("N", M)
    A = np.column_stack(terms)
    c, *_ = np.linalg.lstsq(A, psi, rcond=None)
    assert np.linalg.norm(A @ c - psi) < 1e-9 and np.linalg.matrix_rank(A, tol=1e-8) == 7
    Mt = common.new_matcher(E, native=True)
    target = common.target_of(E)
    bases = {}
    for S in itertools.combinations(range(M), 2):
        moved = np.column_stack([move_front(t, S, M) for t in terms])
        b = slice_base(Mt, moved, X0)
        if b is None:
            visible = [k for k in range(7) if np.linalg.norm(moved[pidx(X0) * 9:(pidx(X0) + 1) * 9, k]) > 1e-9]
            bases.setdefault(("partial", len(visible)), []).append(S)
            continue
        bases.setdefault(tuple(sorted(b)), []).append(S)
    report = {"witness": os.path.relpath(common.WITNESS7, ROOT), "git": git_commit(), "generated": _now(),
              "cap_s": args.cap, "x0": list(X0), "bases": []}
    passed = aborted = 0
    items = sorted(bases.items(), key=lambda kv: str(kv[0]))
    for k, (cover, Ss) in enumerate(items):
        if cover[0] != "partial":
            distinct = sorted(set(cover))
            b1, b2, bC = target.rhs(X0)
            fam = family_from(Mt.U1[distinct], Mt.U2[distinct], Mt.C[:, distinct], b1, b2, bC)
            log(f"base {k}: {cover} pairs {[list(S) for S in Ss]} distinct {len(distinct)} kappa "
                f"{None if fam is None else fam.kappa}")

    def flush():
        n_full = sum(1 for e in report["bases"] if "cover" in e)
        report.update({"passed": passed, "aborted": aborted, "all_visible_bases": n_full,
                       "complete": len(report["bases"]) == len(items),
                       "pass": passed == n_full and aborted == 0 and len(report["bases"]) == len(items)})
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, "control_witness.json")
        with open(path + ".tmp", "w") as f:
            json.dump(report, f, indent=1)
        os.replace(path + ".tmp", path)

    for k, (cover, Ss) in enumerate(items):
        if cover[0] == "partial":
            report["bases"].append({"base": k, "pairs": [list(S) for S in Ss], "status": "not all-visible",
                                    "visible": cover[1]})
            flush()
            continue
        distinct = sorted(set(cover))
        b1, b2, bC = target.rhs(X0)
        fam = family_from(Mt.U1[distinct], Mt.U2[distinct], Mt.C[:, distinct], b1, b2, bC)
        kappa = None if fam is None else int(fam.kappa)
        entry = {"base": k, "cover": list(cover), "pairs": [list(S) for S in Ss], "distinct": len(distinct),
                 "multiplicities": sorted((cover.count(u) for u in distinct), reverse=True), "kappa": kappa}
        if kappa is not None and kappa >= 3:
            # the three-parameter dense solve runs in the Python reference for minutes per slice
            # and does not reach a deadline check inside one solve; recorded as not run
            entry.update({"status": "not run", "reason": f"kappa {kappa}: the Python dense solve"})
            report["bases"].append(entry)
            flush()
            log(f"  base {k}: {cover} kappa {kappa}: not run (three-parameter family)")
            continue
        Mt.budget = Budget(seconds=args.cap, max_rss_gb=args.max_rss_gb)
        set_max_cand(args.max_cand)
        t0 = time.time()
        try:
            hits, st = Mt.run(cover, X0, target)
        except Exception as exc:                          # noqa: BLE001
            dt = time.time() - t0
            entry.update({"status": "aborted", "reason": f"{type(exc).__name__}: {str(exc)[:200]}", "seconds": dt,
                          "stats": _plain(Mt.last_stats)})
            aborted += 1
            log(f"  base {k}: {cover} kappa {kappa}: ABORTED {entry['reason'][:100]} after {dt:.0f}s")
        else:
            good = [h for h in hits if genuine(h, 7)]
            S = Ss[0]
            wkey = codes_key([move_front(v, S, M) for v in terms])
            same = any(codes_key(h["terms"]) == wkey for h in good)
            dt = time.time() - t0
            entry.update({"status": "recovered" if same else "not recovered", "rank7_decompositions": len(good),
                          "witness_recovered": same, "seconds": dt, "stats": _plain(st)})
            passed += same
            log(f"  base {k}: {cover} kappa {kappa} ({len(Ss)} pairs): {len(good)} genuine rank-7 decompositions, "
                f"witness {'recovered' if same else 'NOT among them'}, {dt:.1f}s, coord {st.get('coord_raw')}")
        report["bases"].append(entry)
        flush()
    Mt.budget = None
    set_max_cand(2_000_000)
    flush()
    n_full = report["all_visible_bases"]
    log(f"control-witness: {passed}/{n_full} all-visible bases recover the witness, {aborted} aborted; "
        f"{'PASS' if report['pass'] else 'FAIL (recorded, not required)'}; wrote results/control_witness.json")
    return 0 if report["pass"] else 1


def _plain(st):
    if st is None:
        return None
    out = {}
    for k, v in st.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {str(a): (b if isinstance(b, (int, float, str, bool)) else str(b)) for a, b in v.items()}
        elif isinstance(v, np.generic):
            out[k] = v.item()
    return out


def g2_unitaries(E):
    """The unitary symmetry group G_2 of |N>^2 as 9 x 9 unitaries up to
    phase (the closure of the single-copy Clifford stabilizer of |N> on
    either copy and the swap), and the dictionary permutation each one
    induces; the permutations must be the census's group."""
    from rank_exclusion import clifford_group
    from stabrank_verify import orbit_state
    psi1 = np.array([complex(x) for x in orbit_state("N")]).ravel()
    psi1 /= np.linalg.norm(psi1)
    uni = [U for U in clifford_group(3) if abs(abs(np.vdot(psi1, U @ psi1)) - 1) < 1e-9]
    I3 = np.eye(3, dtype=complex)
    swap = np.zeros((9, 9), dtype=complex)
    for a in range(3):
        for b in range(3):
            swap[3 * b + a, 3 * a + b] = 1
    gens = [np.kron(U, I3) for U in uni] + [swap]

    def key(U):
        v = U.ravel()
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        return (np.round(v, 6) + 0.0).tobytes()

    elems = {key(np.eye(9, dtype=complex)): np.eye(9, dtype=complex)}
    frontier = [np.eye(9, dtype=complex)]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                k = h @ g
                kk = key(k)
                if kk not in elems:
                    elems[kk] = k
                    nxt.append(k)
        frontier = nxt
    perms = []
    for g in elems.values():
        perm = np.zeros(E.N, dtype=np.int64)
        for i in range(E.N):
            perm[i] = E_index(E, g @ E.C[:, i])
        perms.append(perm)
    return list(elems.values()), np.array(perms)


def E_index(E, v):
    codes, _ = exact_codes(v)
    key = codes.tobytes()
    lookup = getattr(E, "_lookup", None)
    if lookup is None:
        lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
        E._lookup = lookup
    return lookup[key]


def control_orbit(args):
    """The orbit lemma on planted decompositions: g = I_9 (x) U for U in G_2
    carries a six-term decomposition with base S at (2, 2) to one with base
    U S and the same invisible flats; the matcher on U S finds the image,
    and S and U S have the same canonical form (the least image under the
    72 permutations, which are exactly G_2's)."""
    E = common.make_enumerator()
    Mt = common.new_matcher(E, native=True)
    rng = np.random.default_rng(args.seed)
    Us, perms = g2_unitaries(E)
    G = degenerate6.group_perms(E)
    same_group = {tuple(p) for p in perms} == {tuple(g) for g in G}
    log(f"G_2: {len(Us)} unitaries up to phase, induced permutations {'equal to' if same_group else 'DIFFERENT from'} "
        f"the census's group of order {len(G)}")
    lists, _ = common.load_reps()
    IM = InvisibleMatcher3(Mt)
    cen = qcommon.load_census("N")
    c5 = [tuple(c) for c in cen["covers5"]]
    k5 = [tuple(int(u) for u in row) for row in lists["k5"]]
    rows = []
    ok = same_group

    def monomial(U):
        nz = np.abs(U) > 1e-9
        vals = U[nz]
        return (np.all(nz.sum(axis=0) == 1) and np.all(nz.sum(axis=1) == 1)
                and np.all(np.abs(vals ** 3 - 1) < 1e-6))

    # the matcher part needs the image target's amplitudes in Z[w], so it
    # draws U from the monomial elements (permutation matrices with cube-root
    # entries: the 0 <-> 1 swap of |N> on either copy and the copy swap); the
    # stabilizer, base and canonical-form checks run over every element
    mono = [i for i, U in enumerate(Us) if monomial(U)]
    log(f"{len(mono)} of the {len(Us)} unitaries are monomial with cube-root entries")
    for n in range(args.count):
        kind = "alpha" if n % 2 == 0 else "beta"
        if kind == "alpha":
            # six distinct visible terms over a census 5-cover plus a random sixth state
            while True:
                base = tuple(sorted(c5[int(rng.integers(len(c5)))] + (int(rng.integers(E.N)),)))
                if len(set(base)) == 6:
                    break
            terms, coeffs = plant(Mt, base, [], rng)
            flats = None
        else:
            base = k5[int(rng.integers(len(k5)))]
            if len(set(base)) < 5:
                base = c5[int(rng.integers(len(c5)))]
            flat = FLAT_NAMES[int(rng.integers(len(FLAT_NAMES)))]
            inv = invisible_term(Mt, flat, rng)
            terms, coeffs = plant(Mt, base, [(inv[0], None)], rng)
            flats = [flat]
        # every element: the images are stabilizer states with the permuted base and the same canonical form
        all_ok = True
        for ui in rng.choice(len(Us), size=min(12, len(Us)), replace=False):
            U = Us[int(ui)]
            g_terms = [(t.reshape(9, 9) @ U.T).ravel() for t in terms]
            for t in g_terms:
                exact_codes(t)                                # raises unless the entries are roots of unity up to a scalar
            vis = [t for t in g_terms if np.linalg.norm(t.reshape(9, 9)[pidx(X0)]) > 1e-9]
            g_base = tuple(sorted(slice_base(Mt, np.column_stack(vis), X0)))
            pred = tuple(sorted(int(perms[int(ui)][u]) for u in base))
            canon_same = int(degenerate6.canonical_codes(np.array([base], dtype=np.int64), G, E.N)[0]) == \
                int(degenerate6.canonical_codes(np.array([g_base], dtype=np.int64), G, E.N)[0])
            all_ok &= (g_base == pred) and canon_same and len(vis) == len(base)
        # a monomial element: the matcher on the image base against the image target finds the image
        ui = int(mono[int(rng.integers(len(mono)))])
        U = Us[ui]
        g_terms = [(t.reshape(9, 9) @ U.T).ravel() for t in terms]
        vis = [t for t in g_terms if np.linalg.norm(t.reshape(9, 9)[pidx(X0)]) > 1e-9]
        g_base = tuple(sorted(slice_base(Mt, np.column_stack(vis), X0)))
        pred = tuple(sorted(int(perms[ui][u]) for u in base))
        tgt = planted_target(g_terms, coeffs, Mt.F1, Mt.F2)
        if flats is None:
            hits, st = Mt.run(g_base, X0, tgt)
        else:
            hits, st = IM.run_one(g_base, flats[0], tgt)
        found = any(codes_key(h["terms"]) == codes_key(g_terms) for h in hits)
        row = {"kind": kind, "base": list(base), "image_base": list(g_base), "predicted_image": list(pred),
               "image_equals_prediction": g_base == pred, "twelve_elements_consistent": bool(all_ok),
               "image_found": found, "flats": flats}
        ok &= found and all_ok and g_base == pred
        rows.append(row)
        log(f"  {kind} base {base} -> {g_base}: image {'found' if found else 'NOT found'}; 12 random elements "
            f"{'consistent' if all_ok else 'INCONSISTENT'} (stabilizer images, permuted base, canonical form)")
    _write_control("control_orbit", {"git": git_commit(), "generated": _now(), "group_order": len(Us),
                                     "permutations_equal": same_group, "rows": rows, "pass": ok})
    log(f"control-orbit: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def control_m3(args):
    """The stored rank-4 decompositions of |N>^3 sliced along every qutrit
    pair at (2, 2): the visible terms' slices are single-qutrit states (the
    n_2 = 1 geometry), the invisible terms lie on flats missing (2, 2), and
    the stage matcher of the configuration (four visible: the qutrit
    Matcher; three or two visible: InvisibleMatcher3 with one or two
    flats) must recover the decomposition."""
    from cover_census import Field3, P1, P2
    from rank_exclusion import dictionary
    from matcher import constructions_common
    cc = constructions_common()
    stored, _ = cc.load_decompositions("N", 3, 4)
    D1 = dictionary(3, 1)
    Mt = Matcher(D1, 1, Field3(P1), Field3(P2), native=False)
    target = psi_target("N", 1, Mt.F1, Mt.F2)
    IM = InvisibleMatcher3(Mt)
    flat_of = {}
    for name, pts in FLATS.items():
        flat_of[frozenset(pts)] = name
    rows = []
    ok = True
    t0 = time.time()
    for d, (terms, _) in enumerate(stored):
        for S in itertools.combinations(range(3), 2):
            moved = [move_front(t, S, 3) for t in terms]
            sl = [t.reshape(9, 3) for t in moved]
            vis = [k for k in range(4) if np.linalg.norm(sl[k][pidx(X0)]) > 1e-9]
            inv = [k for k in range(4) if k not in vis]
            flats = []
            for k in inv:
                pts = frozenset(y for y in PTS if np.linalg.norm(sl[k][pidx(y)]) > 1e-9)
                if pts not in flat_of:
                    rows.append({"decomposition": d, "pair": list(S), "status": "invisible term not on a flat"})
                    ok = False
                    break
                flats.append(flat_of[pts])
            else:
                base = tuple(sorted(Mt.index_of(sl[k][pidx(X0)]) for k in vis))
                want = codes_key(moved)
                try:
                    if not flats:
                        hits, st = Mt.run(base, X0, target)
                    else:
                        hits, st = IM.run(base, [FLATS[f] for f in flats], target)
                except (UnpinnedFamily, BudgetExceeded) as exc:
                    rows.append({"decomposition": d, "pair": list(S), "base": list(base), "flats": flats,
                                 "status": f"undecided: {str(exc)[:120]}"})
                    ok = False
                    continue
                found = any(codes_key(h["terms"]) == want for h in hits)
                ok &= found and not st["refused"]
                rows.append({"decomposition": d, "pair": list(S), "base": list(base), "flats": flats,
                             "visible": len(vis), "hits": len(hits), "status": "recovered" if found else "NOT recovered"})
                log(f"  decomposition {d} pair {S}: base {base} flats {flats}: {len(hits)} hits, "
                    f"{'recovered' if found else 'NOT recovered'}")
    _write_control("control_m3", {"git": git_commit(), "generated": _now(), "stored": len(stored), "rows": rows,
                                  "seconds": time.time() - t0, "pass": ok})
    log(f"control-m3: {len(rows)} (decomposition, pair) runs, {'PASS' if ok else 'FAIL'} [{time.time() - t0:.0f}s]")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lists")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=lists)
    p = sub.add_parser("control-lists")
    p.add_argument("--census", action="store_true", help="re-run cover6_pair over every pivot pair (about 7 minutes)")
    p.set_defaults(fn=control_lists)
    p = sub.add_parser("sample")
    p.add_argument("stage", choices=list(common.STAGES))
    p.add_argument("--count", type=int, default=40, help="items per class (A6: covers)")
    p.add_argument("--pairs", type=int, default=12, help="A6: random pivot pairs to draw covers from")
    p.add_argument("--reference", type=int, default=50, help="A6: covers also run through Matcher.run")
    p.add_argument("--budget", type=float, default=480.0, help="wall-clock seconds over all classes")
    p.add_argument("--item-cap", type=float, default=0.0, help="seconds per item (0: none); a capped item is a tail item")
    p.add_argument("--seed", type=int, default=3)
    p.set_defaults(fn=sample)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=600.0, help="pod seconds per batch")
    p.add_argument("--max-items", type=int, default=5000, help="at most this many items per B6 or C6 batch")
    p.add_argument("--max-cand-high", type=int, default=200_000_000, help="dense candidate cap for kappa 2 and 3")
    p.set_defaults(fn=partition)
    p = sub.add_parser("control-planted")
    p.add_argument("--stage", choices=["b6", "c6", "beta", "gamma", "all"], default="all")
    p.add_argument("--plant", type=int, default=1)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--cap", type=float, default=60.0, help="seconds per planted instance")
    p.add_argument("--only", default=None, help="comma-separated flats (beta) or pairs as F+G (gamma)")
    p.add_argument("--kinds", default=None, help="comma-separated plant kinds")
    p.set_defaults(fn=control_planted)
    p = sub.add_parser("control-witness")
    p.add_argument("--cap", type=float, default=120.0, help="seconds per base")
    p.add_argument("--max-rss-gb", type=float, default=8.0)
    p.add_argument("--max-cand", type=int, default=200_000_000)
    p.set_defaults(fn=control_witness)
    p = sub.add_parser("control-orbit")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(fn=control_orbit)
    p = sub.add_parser("control-m3")
    p.set_defaults(fn=control_m3)
    args = ap.parse_args(argv[1:])
    common.lower_priority()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
