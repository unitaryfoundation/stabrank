"""Measurements for the design of the rank-6 exclusion of |H3>^4
(docs/notes/h3_m4_rank6_design.md): the geometry of the flats missing the
base point (0, 0), the base lists and their G_2 orbit counts, the rate of
the invisible-flat matcher on real bases, a sample of the 6-cover census
of |H3>^2 through the compiled cover6_pair, and the stage A6 kernel rate
at (0, 0). Records under research/h3_m4_rank6/results/.

    probe.py geometry
        The 16 flats of F_3^2 missing (0, 0), which coordinate points each
        contains, the slice ratios alpha_y / alpha_(0, 0), and the 136 flat
        multisets of stage (gamma) by kind.
    probe.py lists
        The stage (beta') bases (the attested rank-5 lists of |H3>^2: full
        5-covers, dependent 5-covers, repeated 5-multisets) and the stage
        (gamma) bases (full 4-covers, dependent 4-covers T_3 + span state,
        repeated 4-multisets T_3 + member), each with its G_2 orbit count;
        the span-state counts of sampled 5-covers for the B6 estimate.
    probe.py rate [--count K] [--budget S]
        InvisibleMatcherP3.run_one on K orbit-spread bases of every (beta')
        class at every one of the 16 flats and run_two on K bases of every
        (gamma) class at every one of the 136 flat multisets: seconds per
        run, where the runs die, hits, refusals.
    probe.py census6 [--pairs K] [--budget S] [--all]
        cover6_pair on K stratified pivot pairs of the 6-cover enumeration
        of |H3>^2 (every pair with --all, resumable): full 6-covers,
        candidates, seconds, and the projection by the sum of M^3.
    probe.py a6rate [--pairs K] [--count N]
        Matcher.run at (0, 0) through SliceMatch3Kernel on N real full
        6-covers from K random pivot pairs: milliseconds per cover.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common  # noqa: E402
from common import CELL, X0  # noqa: E402
from cover_census import _reduce  # noqa: E402
from invisible_p3 import InvisibleMatcherP3, hist_key  # noqa: E402
from matcher import BudgetExceeded, UnpinnedFamily  # noqa: E402

RESULTS = common.RESULTS


def log(s):
    print(s, flush=True)


def write(name, rec):
    out = os.path.join(RESULTS, name)
    rec = dict(rec)
    rec["git"] = common.git_commit()
    common.write_record(out, rec)
    log(f"wrote {os.path.relpath(out, common.ROOT)}")


# ----------------------------------------------------------- geometry ------

def geometry(a):
    cell = CELL
    c1, c2 = cell.coord_points
    rows = []
    for name, pts in cell.flats.items():
        rows.append({"flat": name, "points": [list(p) for p in pts], "kind": cell.flat_kind(name),
                     "contains_e1": c1 in pts, "contains_e2": c2 in pts,
                     "ratios": [round(common.ratio(p), 6) for p in pts]})
    by_coord = {"neither": 0, "one": 0, "both": 0}
    for r in rows:
        n = int(r["contains_e1"]) + int(r["contains_e2"])
        by_coord[("neither", "one", "both")[n]] += 1
    pairs = {"point+point (equal)": 0, "point+point": 0, "point+line (point on line)": 0,
             "point+line": 0, "line+line (equal)": 0, "line+line": 0}
    first_shared = 0
    for f, g in cell.flat_pairs:
        F, G = cell.flats[f], cell.flats[g]
        if len(F) == 1 and len(G) == 1:
            pairs["point+point (equal)" if F == G else "point+point"] += 1
        elif len(F) == 1:
            pairs["point+line (point on line)" if F[0] in G else "point+line"] += 1
        else:
            pairs["line+line (equal)" if F == G else "line+line"] += 1
        # the first processed point of each flat under the matcher's order: shared?
        IM_order = InvisibleMatcherP3.order
        _, _, modes = IM_order(type("S", (), {"cell": cell})(), [F, G])
        first_shared += any(m == "scan2" for _, m in modes.items())
    ratios = sorted({round(common.ratio(y), 6) for y in cell.others})
    rec = {"x0": list(cell.x0), "alpha": {str(k): v for k, v in common.ALPHA_C.items()}, "ratios": ratios,
           "flats": rows, "flats_by_coordinate_points": by_coord, "flat_pairs": len(cell.flat_pairs),
           "pairs_by_kind": pairs, "pairs_with_two_fresh_terms_at_a_point": first_shared}
    log(f"base point {cell.x0}; ratios {ratios}; flats by coordinate points {by_coord}; pairs {pairs}; "
        f"{first_shared} of 136 flat multisets reach a two-fresh-term point")
    write("geometry.json", rec)
    return 0


# -------------------------------------------------------------- lists ------

def group_perms(E):
    """The unitary symmetry group as an array (|G|, N) of permutations of
    the dictionary, the closure of the enumerator's generators."""
    N = E.N
    seen = {tuple(range(N))}
    elems = [np.arange(N)]
    frontier = [np.arange(N)]
    while frontier:
        nxt = []
        for g in frontier:
            for p in E.info["perms"]:
                h = p[g]
                t = tuple(h)
                if t not in seen:
                    seen.add(t)
                    elems.append(h)
                    nxt.append(h)
        frontier = nxt
    G = np.array(elems, dtype=np.int64)
    assert len(G) == E.info["order"], (len(G), E.info["order"])
    return G


def canonical_codes(sets, G, N, chunk=20000):
    """The canonical form (least image under G of the sorted tuple) of every
    row of `sets` (n, k), as base-N codes."""
    sets = np.asarray(sets, dtype=np.int64)
    out = np.empty(len(sets), dtype=np.int64)
    for s in range(0, len(sets), chunk):
        blk = sets[s:s + chunk]
        img = G[:, blk]
        img.sort(axis=2)
        codes = np.zeros(img.shape[:2], dtype=np.int64)
        for c in range(img.shape[2]):
            codes = codes * N + img[:, :, c]
        out[s:s + chunk] = codes.min(axis=0)
    return out


def span_states(E, T):
    """States of span(T) other than T's members, over F_P1 (a superset of
    the exact answer, decided numerically by the caller)."""
    rows = E.U1.copy()
    for b in T:
        rows, _ = _reduce(E.F1, rows, rows[b])
    zero = ~rows.any(axis=1)
    zero[list(T)] = False
    return np.flatnonzero(zero)


def orbit_reps(lst, G, N):
    """One representative (the least member) per orbit, in list order."""
    if not lst:
        return []
    codes = canonical_codes(np.array(lst, dtype=np.int64), G, N)
    _, first = np.unique(codes, return_index=True)
    return [lst[i] for i in sorted(first)]


def lists(a):
    common.lower_priority()
    t0 = time.time()
    E = common.make_enumerator()
    G = group_perms(E)
    L = common.rank5_lists()
    rec = {"group_order": int(len(G)), "k5": {}, "k4": {}, "covers3": len(L["covers3"]),
           "census_sha256": L["census_sha256"], "degenerate_sha256": L["degenerate_sha256"]}
    rep5 = L["rep5"]
    by_pat = {}
    for c in rep5:
        by_pat[str(common.multiplicity_pattern(c))] = by_pat.get(str(common.multiplicity_pattern(c)), 0) + 1
    cancel_at_base = [c for c in rep5 if common.multiplicity_pattern(c) == (2, 1, 1, 1)
                      and not E.is_cover(tuple(sorted(set(c))))]
    for name, lst in (("full 5-covers (census)", L["covers5"]), ("dependent 5-covers", L["dep5"]),
                      ("repeated 5-multisets", rep5)):
        reps = orbit_reps(lst, G, E.N)
        rec["k5"][name] = {"multisets": len(lst), "orbits": len(reps)}
        log(f"{name}: {len(lst)} -> {len(reps)} orbits")
    rec["k5"]["repeated by pattern"] = by_pat
    rec["k5"]["cancel-at-base T_3 + (b, b)"] = len(cancel_at_base)
    all5 = L["covers5"] + L["dep5"] + rep5
    rec["k5"]["all"] = {"multisets": len(all5), "orbits": len(orbit_reps(all5, G, E.N))}
    dep4 = []
    for T in L["covers3"]:
        for x in span_states(E, T).tolist():
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                dep4.append(S)
    dep4 = sorted(set(dep4))
    rep4 = sorted({tuple(sorted(T + (u,))) for T in L["covers3"] for u in T})
    for name, lst in (("full 4-covers (census)", L["covers4"]), ("dependent 4-covers", dep4),
                      ("repeated 4-multisets", rep4)):
        reps = orbit_reps(lst, G, E.N)
        rec["k4"][name] = {"multisets": len(lst), "orbits": len(reps)}
        log(f"{name}: {len(lst)} -> {len(reps)} orbits")
    all4 = L["covers4"] + dep4 + rep4
    rec["k4"]["all"] = {"multisets": len(all4), "orbits": len(orbit_reps(all4, G, E.N))}
    # the B6 and C6 size estimates: span states per 5-cover (T_5 + span state is the
    # dominant B6 route) on an evenly spaced sample
    sample = [L["covers5"][i] for i in np.linspace(0, len(L["covers5"]) - 1, a.sample).astype(int)]
    counts = []
    for T in sample:
        xs = span_states(E, T).tolist()
        n = 0
        for x in xs:
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                n += 1
        counts.append(n)
    rec["b6_span_states_per_5cover"] = {"sample": len(sample), "mean": float(np.mean(counts)),
                                         "min": int(min(counts)), "max": int(max(counts)),
                                         "histogram": {str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}}
    rec["b6_t5_plus_span_pairs_estimate"] = int(round(np.mean(counts) * len(L["covers5"])))
    rec["c6_estimates"] = {"(2, 1, 1, 1, 1) from independent 5-covers": 5 * len(L["covers5"]),
                           "(2, 1, 1, 1, 1) from dependent 5-covers": 5 * len(L["dep5"]),
                           "4-cover plus (b, b), b outside": len(L["covers4"]) * (E.N - 4),
                           "(3, 1, 1, 1), (2, 2, 1, 1) from 4-covers": 4 * len(L["covers4"]) + 6 * len(L["covers4"])}
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"B6 route T_5 + span state: {np.mean(counts):.1f} span states per 5-cover in {len(sample)} sampled, "
        f"about {rec['b6_t5_plus_span_pairs_estimate']:,} (T_5, x) pairs")
    write("lists.json", rec)
    return 0


# --------------------------------------------------------------- rate ------

def _classes(E, L, G):
    """Orbit-spread bases per class for the rate sample."""
    out = {}
    out["beta (1, 1, 1, 1, 1)"] = orbit_reps(L["covers5"], G, E.N)
    out["beta (1, 1, 1, 1, 1) dependent"] = orbit_reps(L["dep5"], G, E.N)
    rep = orbit_reps(L["rep5"], G, E.N)
    for c in rep:
        out.setdefault("beta " + common.item_class(E, c), []).append(c)
    out["gamma (1, 1, 1, 1)"] = orbit_reps(L["covers4"], G, E.N)
    dep4 = []
    for T in L["covers3"]:
        for x in span_states(E, T).tolist():
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                dep4.append(S)
    out["gamma (1, 1, 1, 1) dependent"] = orbit_reps(sorted(set(dep4)), G, E.N)
    out["gamma (2, 1, 1)"] = orbit_reps(sorted({tuple(sorted(T + (u,))) for T in L["covers3"] for u in T}), G, E.N)
    return out


def rate(a):
    common.lower_priority()
    t_all = time.time()
    E = common.make_enumerator()
    Mt = common.new_matcher(E, native=True)
    target = common.target_of(E)
    cell = CELL
    IM = InvisibleMatcherP3(Mt, cell)
    G = group_perms(E)
    L = common.rank5_lists()
    classes = _classes(E, L, G)
    rec = {"x0": list(cell.x0), "count": a.count, "classes": {}}
    deadline = time.time() + a.budget
    for cl, reps in classes.items():
        if a.only and not cl.startswith(a.only):
            continue
        n = min(a.count, len(reps))
        sel = [reps[i] for i in np.linspace(0, len(reps) - 1, n).astype(int)] if n else []
        stage = cl.split()[0]
        units = cell.flat_names if stage == "beta" else cell.flat_pairs
        secs, hist, hits, refused, undecided, reasons = [], {}, 0, 0, 0, {}
        per_base = []
        t_cl = time.time()
        # warm the shape tables of the first base, outside the timing
        if sel:
            try:
                (IM.run_one if stage == "beta" else IM.run_two)(sel[0], units[0], target)
            except (UnpinnedFamily, BudgetExceeded):
                pass
        for base in sel:
            if time.time() > deadline:
                break
            tb = time.time()
            for u in units:
                t0 = time.time()
                try:
                    h, st = (IM.run_one if stage == "beta" else IM.run_two)(base, u, target)
                except UnpinnedFamily as exc:
                    undecided += 1
                    reasons[str(exc)[:80]] = reasons.get(str(exc)[:80], 0) + 1
                except BudgetExceeded as exc:
                    undecided += 1
                    reasons["budget: " + str(exc)[:70]] = reasons.get("budget: " + str(exc)[:70], 0) + 1
                else:
                    hits += len(h)
                    refused += int(st["refused"])
                    k = hist_key(st)
                    hist[k] = hist.get(k, 0) + 1
                secs.append(time.time() - t0)
            per_base.append(round(time.time() - tb, 3))
        rec["classes"][cl] = {"orbits": len(reps), "bases_run": len(per_base), "runs": len(secs),
                              "s_per_run_mean": float(np.mean(secs)) if secs else None,
                              "s_per_run_median": float(np.median(secs)) if secs else None,
                              "s_per_run_max": float(np.max(secs)) if secs else None,
                              "s_per_base": per_base, "hits": hits, "refused": refused, "undecided": undecided,
                              "undecided_reasons": reasons,
                              "hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12]),
                              "seconds": round(time.time() - t_cl, 1)}
        r = rec["classes"][cl]
        log(f"{cl}: {len(reps)} orbits, {len(per_base)} bases x {len(units)} units: "
            f"{(r['s_per_run_mean'] or 0) * 1e3:.1f} ms per run (median {(r['s_per_run_median'] or 0) * 1e3:.1f}, "
            f"max {(r['s_per_run_max'] or 0) * 1e3:.0f}), hits {hits}, refused {refused}, undecided {undecided}"
            f"{' ' + str(reasons) if reasons else ''}; hist {list(r['hist'].items())[:4]} [{r['seconds']}s]")
    rec["seconds"] = round(time.time() - t_all, 1)
    write("rate_invisible.json", rec)
    return 0


# ------------------------------------------------------------- census ------

def _plan(E, i):
    Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
    members, partners = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    return Qi, mask, partners


def kernel_pair(E, i, j, mask, max_run=4096):
    """cover6_pair on one pair: (6-sets, ranks mod P2, candidates, M)."""
    idx, flags, ranks, ncand, members = E.native_cover6(
        E.Q1, E.U1, E.psi1, E.U2, E.psi2, int(i), int(j), np.ascontiguousarray(mask, dtype=np.uint8),
        int(max_run), int(E.rng_seed))
    idx = np.asarray(idx).reshape(-1, 6)
    flags = np.asarray(flags).reshape(-1, 2)
    ranks = np.asarray(ranks).reshape(-1)
    f1, f2 = flags[:, 0] != 0, flags[:, 1] != 0
    keep = f1 & f2
    for t in np.flatnonzero(f1 ^ f2):
        keep[t] = E.is_full(tuple(int(x) for x in idx[t]))
    return idx[keep], ranks[keep], int(ncand), int(members)


def census6(a):
    common.lower_priority()
    E = common.make_enumerator()
    if E.native_cover6 is None:
        raise SystemExit("cover6_pair is not available")
    out = os.path.join(RESULTS, "census6_H3_full.json" if a.all else "census6_H3_sample.json")
    rec = {"orbit": common.ORBIT, "n": 2, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "columns": ["pivot", "partner", "members", "full6", "full6_independent", "candidates", "seconds"],
           "rows": []}
    if a.all and os.path.exists(out):
        with open(out) as f:
            rec = json.load(f)
    done = {(r[0], r[1]) for r in rec["rows"]}
    units = E.units()
    rec["pairs"] = len(units)
    rec["sum_M3"] = int(sum(m ** 3 for _, _, m in units))
    rec["sum_M2"] = int(sum(m ** 2 for _, _, m in units))
    if a.all:
        todo = [u for u in units if (u[0], u[1]) not in done]
    else:
        # stratified: every pivot's partner list at the fraction `phase`, plus the heaviest pairs
        by_pivot = {}
        for u in units:
            by_pivot.setdefault(u[0], []).append(u)
        todo = []
        for i, lst in by_pivot.items():
            lst = sorted(lst, key=lambda u: -u[2])
            k = max(1, round(a.pairs * len(lst) / len(units)))
            picks = sorted({int(round(q)) for q in np.linspace(0, len(lst) - 1, k)})
            todo.extend(lst[q] for q in picks)
    t_start = time.time()
    plans = {}
    for (i, j, m) in todo:
        if time.time() - t_start > a.budget:
            log("budget reached")
            break
        if i not in plans:
            plans[i] = _plan(E, i)
        t0 = time.time()
        idx, ranks, ncand, members = kernel_pair(E, i, j, plans[i][1])
        dt = time.time() - t0
        rec["rows"].append([int(i), int(j), int(members), int(len(idx)), int(np.sum(ranks == 6)), ncand, round(dt, 4)])
        if a.all:
            rec["pairs_done"] = len(rec["rows"])
            with open(out + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(out + ".tmp", out)
    rows = rec["rows"]
    rec["pairs_done"] = len(rows)
    full6 = sum(r[3] for r in rows)
    dep = sum(r[3] - r[4] for r in rows)
    secs = sum(r[6] for r in rows)
    m3 = sum(r[2] ** 3 for r in rows)
    rec["full6"] = int(full6)
    rec["full6_independent"] = int(full6 - dep)
    rec["dependent_from_kernel"] = int(dep)
    rec["kernel_seconds"] = round(secs, 2)
    if not a.all and rows:
        # project by M^3 within the sample (covers and seconds both scale with the member count)
        rec["projection"] = {"covers_per_M3": full6 / m3, "seconds_per_M3": secs / m3,
                             "covers_projected": int(round(full6 / m3 * rec["sum_M3"])),
                             "seconds_projected": round(secs / m3 * rec["sum_M3"], 1)}
        # and per pivot: the mean covers per pair of each sampled pivot times its pair count
        per_piv = {}
        for r in rows:
            per_piv.setdefault(r[0], []).append(r[3])
        counts = {}
        for u in units:
            counts[u[0]] = counts.get(u[0], 0) + 1
        rec["projection"]["covers_projected_per_pivot_means"] = int(round(sum(
            np.mean(v) * counts[i] for i, v in per_piv.items())))
        log(f"{len(rows)} pairs of {len(units)}: {full6:,} full 6-covers ({dep} dependent from the kernel) in {secs:.1f} "
            f"kernel seconds; projected {rec['projection']['covers_projected']:,} covers (by M^3) or "
            f"{rec['projection']['covers_projected_per_pivot_means']:,} (per-pivot means), "
            f"{rec['projection']['seconds_projected']:.0f} kernel seconds for the census")
    else:
        log(f"{len(rows)} pairs of {len(units)}: {full6:,} full 6-covers in {secs:.1f} kernel seconds")
    with open(out, "w") as f:
        json.dump(rec, f)
    log(f"wrote {os.path.relpath(out, common.ROOT)}")
    return 0


def a6rate(a):
    common.lower_priority()
    E = common.make_enumerator()
    Mt = common.new_matcher(E, native=True)
    target = common.target_of(E)
    x0 = X0
    units = E.units()
    rng = np.random.default_rng(a.seed)
    sel = [units[t] for t in rng.choice(len(units), size=min(a.pairs, len(units)), replace=False)]
    covers, dep = [], 0
    plans = {}
    for (i, j, _) in sel:
        if i not in plans:
            plans[i] = _plan(E, i)
        idx, ranks, _, _ = kernel_pair(E, i, j, plans[i][1])
        dep += int(np.sum(ranks < 6))
        covers.extend(tuple(int(x) for x in row) for row in idx[ranks == 6])
        if len(covers) >= a.count:
            break
    covers = covers[:a.count]
    for c in covers[:200]:
        Mt.run(c, x0, target)
    t0 = time.time()
    hist, hits, refused, nat = {}, 0, 0, 0
    for c in covers:
        h, st = Mt.run(c, x0, target)
        k = ",".join(str(v) for v in st["coord_raw"])
        hist[k] = hist.get(k, 0) + 1
        hits += len(h)
        refused += int(st["refused"])
        nat += int(bool(st.get("native")))
    dt = time.time() - t0
    rec = {"x0": list(x0), "covers": len(covers), "pairs": len(sel), "dependent_from_kernel": dep,
           "ms_per_cover": 1e3 * dt / max(1, len(covers)), "native_runs": nat, "hits": hits, "refused": refused,
           "coord_raw_hist": dict(sorted(hist.items(), key=lambda kv: -kv[1]))}
    log(f"A6 at {x0}: {rec['ms_per_cover']:.3f} ms per cover ({nat} of {len(covers)} through the kernel), hits {hits}, "
        f"refused {refused}; coord_raw histogram {list(rec['coord_raw_hist'].items())[:6]}")
    write("rate_a6_x00.json", rec)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.set_defaults(fn=geometry)
    ls = sub.add_parser("lists")
    ls.add_argument("--sample", type=int, default=200)
    ls.set_defaults(fn=lists)
    r = sub.add_parser("rate")
    r.add_argument("--count", type=int, default=20)
    r.add_argument("--budget", type=float, default=400.0)
    r.add_argument("--only", default=None)
    r.set_defaults(fn=rate)
    c = sub.add_parser("census6")
    c.add_argument("--pairs", type=int, default=60)
    c.add_argument("--budget", type=float, default=300.0)
    c.add_argument("--all", action="store_true")
    c.set_defaults(fn=census6)
    r6 = sub.add_parser("a6rate")
    r6.add_argument("--pairs", type=int, default=12)
    r6.add_argument("--count", type=int, default=5000)
    r6.add_argument("--seed", type=int, default=3)
    r6.set_defaults(fn=a6rate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
