"""The census of full 6-covers of |N>^2 through the compiled cover6_pair,
pair by pair with a resumable record, and the stage A6 matcher rate
(SliceMatch3Kernel through Matcher.run) on real 6-covers at the base point
(docs/notes/n4_rank6_design.md, section 4).

    census6_full.py census [--budget S] [--out FILE]
        Every pivot pair of CoverEnumerator3("N", 2) not yet in the record:
        the kernel's 6-sets with their rank mod P2 (independent when 6),
        the candidate count, the member count M and the seconds. The record
        is rewritten after every pair; rerunning resumes.
    census6_full.py rate [--pairs K] [--count N] [--reference R] [--x0 2,2]
        Matcher.run at x0 on the full 6-covers of the first K pivot pairs
        (up to N covers): milliseconds per cover through the kernel, the
        coordinate-slice histogram, hits and refusals; the first R covers
        also through the reference matcher for agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from cover_census import CoverEnumerator3, _reduce  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def log(s):
    print(s, flush=True)


def plan(E, i):
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
    for t in np.flatnonzero(f1 ^ f2):                          # full modulo one prime only: decide numerically
        keep[t] = E.is_full(tuple(int(x) for x in idx[t]))
    return idx[keep], ranks[keep], int(ncand), int(members)


def census(a):
    try:
        os.nice(19)
    except OSError:
        pass
    E = CoverEnumerator3("N", 2)
    if E.native_cover6 is None:
        raise SystemExit("cover6_pair is not available")
    out = a.out or os.path.join(RESULTS, "census6_N_full.json")
    rec = {"orbit": "N", "n": 2, "N": int(E.N), "order": int(E.info["order"]), "pivots": len(E.reps),
           "columns": ["pivot", "partner", "members", "full6", "full6_independent", "candidates", "seconds"],
           "rows": []}
    if os.path.exists(out):
        with open(out) as f:
            rec = json.load(f)
    done = {(r[0], r[1]) for r in rec["rows"]}
    units = E.units()
    rec["pairs"] = len(units)
    log(f"N n=2: {len(units)} pivot pairs, {len(done)} done")
    t0 = time.time()
    plans = {}
    for (i, j, M0) in units:
        if (i, j) in done:
            continue
        if a.budget and time.time() - t0 > a.budget:
            log("budget reached")
            break
        if i not in plans:
            plans[i] = plan(E, i)
        Qi, mask, _ = plans[i]
        t1 = time.time()
        idx, ranks, ncand, M = kernel_pair(E, i, j, mask)
        dt = time.time() - t1
        rec["rows"].append([int(i), int(j), M, int(len(idx)), int(np.sum(ranks == 6)), ncand, dt])
        done.add((i, j))
        if len(rec["rows"]) % 50 == 0 or dt > 3:
            tot = sum(r[3] for r in rec["rows"])
            log(f"  pair {len(rec['rows'])}/{len(units)} ({i}, {j}): M={M}, {len(idx)} full 6-covers "
                f"({int(np.sum(ranks < 6))} dependent), {ncand} candidates, {dt:.1f}s [total {tot}, "
                f"{time.time() - t0:.0f}s]")
        with open(out + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(out + ".tmp", out)
    rows = rec["rows"]
    rec["pairs_done"] = len(rows)
    rec["full6"] = int(sum(r[3] for r in rows))
    rec["full6_independent"] = int(sum(r[4] for r in rows))
    rec["candidates"] = int(sum(r[5] for r in rows))
    rec["kernel_seconds"] = float(sum(r[6] for r in rows))
    rec["sum_M3"] = float(sum(r[2] ** 3 for r in rows))
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(out + ".tmp", out)
    log(f"{len(rows)} of {len(units)} pairs: {rec['full6']} full 6-covers ({rec['full6_independent']} independent), "
        f"{rec['candidates']} candidates, kernel {rec['kernel_seconds']:.0f}s, sum M^3 {rec['sum_M3']:.3g}")
    return 0 if len(rows) == len(units) else 2


def rate(a):
    try:
        os.nice(19)
    except OSError:
        pass
    from matcher import Matcher, exact_codes, psi_target
    E = CoverEnumerator3("N", 2)
    Mn = Matcher(E.D, 2, E.F1, E.F2, native=True)
    Mr = Matcher(E.D, 2, E.F1, E.F2, native=False)
    target = psi_target("N", 2, E.F1, E.F2)
    x0 = tuple(int(v) for v in a.x0.split(","))
    covers, ranks_dep = [], 0
    units = E.units()
    rng = np.random.default_rng(a.seed)
    sel = [units[t] for t in rng.choice(len(units), size=min(a.pairs, len(units)), replace=False)]
    plans = {}
    for (i, j, _) in sel:
        if i not in plans:
            plans[i] = plan(E, i)
        idx, ranks, _, _ = kernel_pair(E, i, j, plans[i][1])
        ranks_dep += int(np.sum(ranks < 6))
        covers.extend(tuple(int(x) for x in row) for row in idx[ranks == 6])
        if len(covers) >= a.count:
            break
    covers = covers[:a.count]
    log(f"{len(covers)} independent full 6-covers from {len(sel)} random pivot pairs ({ranks_dep} dependent 6-sets "
        f"returned by the kernel); x0 {x0}")
    # warm the option cache, then time
    for c in covers[:400]:
        Mn.run(c, x0, target)
    t0 = time.time()
    hist, hits, refused, nat = {}, 0, 0, 0
    for c in covers:
        h, st = Mn.run(c, x0, target)
        k = ",".join(str(v) for v in st["coord_raw"])
        hist[k] = hist.get(k, 0) + 1
        hits += len(h)
        refused += int(st["refused"])
        nat += int(bool(st.get("native")))
    dt = time.time() - t0
    mism = 0
    tr0 = time.time()
    for c in covers[:a.reference]:
        hn, sn = Mn.run(c, x0, target)
        hr, sr = Mr.run(c, x0, target)
        kn = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hn)
        kr = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hr)
        same = kn == kr and sn["coord_solutions"] == sr["coord_solutions"] and sn["refused"] == sr["refused"] \
            and sn["composite_solutions"] == sr["composite_solutions"]
        mism += not same
    tr = time.time() - tr0
    rec = {"x0": list(x0), "covers": len(covers), "pairs": len(sel), "dependent_from_kernel": ranks_dep,
           "ms_per_cover": 1e3 * dt / len(covers), "native_runs": nat, "hits": hits, "refused": refused,
           "coord_raw_hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
           "reference_checked": min(a.reference, len(covers)), "reference_mismatches": mism,
           "reference_ms_per_cover": 1e3 * tr / max(1, min(a.reference, len(covers)))}
    out = a.out or os.path.join(RESULTS, f"rate_a6_x{x0[0]}{x0[1]}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"A6 at {x0}: {rec['ms_per_cover']:.3f} ms per cover ({nat} through the kernel), hits {hits}, refused "
        f"{refused}; coord_raw histogram {list(rec['coord_raw_hist'].items())[:6]}; reference {mism} mismatches in "
        f"{rec['reference_checked']} at {rec['reference_ms_per_cover']:.1f} ms; wrote {os.path.relpath(out, ROOT)}")
    return 0 if mism == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--budget", type=float, default=0)
    c.add_argument("--out")
    c.set_defaults(fn=census)
    r = sub.add_parser("rate")
    r.add_argument("--pairs", type=int, default=12)
    r.add_argument("--count", type=int, default=20000)
    r.add_argument("--reference", type=int, default=50)
    r.add_argument("--x0", default="2,2")
    r.add_argument("--seed", type=int, default=3)
    r.add_argument("--out")
    r.set_defaults(fn=rate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
