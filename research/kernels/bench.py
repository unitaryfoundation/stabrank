"""Rates of the compiled kernels against their Python references
(docs/notes/kernels_k6_p3.md). Every subcommand runs the reference where it
is affordable, compares the results item by item, and writes a record under
research/kernels/results/.

    bench.py cover6 ORBIT n --pairs i,j [i,j ...] [--reference] [--out FILE]
        The k = 6 cover kernel (cover6_pair) on the given pivot pairs of the
        n-qudit dictionary: covers, candidates, and seconds per pair, and with
        --reference the same through slice_cover.pair_covers6_reference with
        an equality check of the cover sets.
    bench.py cover5-key ORBIT n --pairs i,j [i,j ...] [--out FILE]
        cover5_pair with the 16-bit key (one functional) and the 32-bit key
        (two functionals) on the given pairs: the covers must agree; the
        candidate counts and the seconds are the accident rate.
    bench.py stage-a3 ORBIT [--count K] [--x0 a,b ...] [--out FILE]
        The p = 3 stage A kernel (SliceMatch3Kernel) against
        research/qutrit_m4_rank5/matcher.Matcher on the first K full 5-covers
        of |M>^2 from the census enumeration, at the given base points: hits,
        coordinate and composite solution counts must agree; per-cover times
        of both paths.
    bench.py dense3 ORBIT [--count K] [--x0 a,b] [--out FILE]
        The compiled dense family solve inside the qutrit matcher on K
        dependent 6-covers of |M>^2 of the B6 kind (a full 5-cover of the
        census enumeration plus a state of its span, kappa = 1) at the base
        point, against the reference with STABRANK_NO_NATIVE set for the
        second run: hits and solution counts must agree; per-cover times.
    bench.py dense ORBIT [--count K] [--x0 X] [--out FILE]
        The compiled dense family solve on K dependent 5-covers (stage B) of
        psi_3 at qubit_H (the H^6 degenerate list) or qubit_T (the T^5 list)
        through slice_cover.SliceMatcher(E, 2), against the reference with
        STABRANK_NO_NATIVE set for the second run: hits and solution counts
        must agree; per-cover times.

Everything is single-process; the caller wraps it in research/t5_rank5/run.py
for the nice level and the wall-clock cap.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
RESULTS = os.path.join(HERE, "results")


def log(s):
    print(s, flush=True)


def write(rec, path):
    rec["host"] = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()}
    rec["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    log(f"wrote {os.path.relpath(path, ROOT)}")


def qutrit_orbit(orbit):
    return orbit in ("N", "H3", "S", "T3")


def build(orbit, n, **kw):
    if qutrit_orbit(orbit):
        from cover_census import CoverEnumerator3, _reduce
        return CoverEnumerator3(orbit, n, **kw), _reduce
    from slice_cover import CoverEnumerator, _reduce
    return CoverEnumerator(n, orbit=orbit, **kw), _reduce


def plan(E, reduce, i):
    Qi, _ = reduce(E.F1, E.Q1, E.Q1[i])
    members, _ = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    return Qi, mask


def parse_pairs(items, E=None):
    """Pairs as i,j, or u:K for the K-th pivot pair of the enumerator's plan."""
    out = []
    units = None
    for a in items:
        if a.startswith("u:"):
            if units is None:
                units = [(int(i), int(j)) for i in E.reps for j in E.pivot_plan(int(i))[1]]
            out.append(units[int(a[2:])])
        else:
            out.append(tuple(int(v) for v in a.split(",")))
    return out


# ---------------------------------------------------------------- cover6 ----

def cover6(a):
    from slice_cover import pair_covers6_native, pair_covers6_reference
    t0 = time.time()
    E, reduce = build(a.orbit, a.n)
    log(f"{a.orbit} n={a.n}: N={E.N}, built in {time.time() - t0:.1f}s; native cover6 {E.native_cover6 is not None}")
    if E.native_cover6 is None:
        raise SystemExit("cover6_pair is not available in stabrank_core")
    rows = []
    for i, j in parse_pairs(a.pairs, E):
        Qi, mask = plan(E, reduce, i)
        t = time.time()
        got, nc = pair_covers6_native(E, E.native_cover6, i, j, mask)
        tn = time.time() - t
        M = int(np.count_nonzero(reduce(E.F1, Qi, Qi[j])[0][mask & (np.arange(E.N) > j) & (np.arange(E.N) != i)].any(axis=1))) if np.any(Qi[j]) else 0
        row = {"pivot": i, "partner": j, "members": M, "covers": len(got), "candidates": int(nc), "native_s": tn}
        if a.reference:
            t = time.time()
            ref, ncr = pair_covers6_reference(E, i, j, Qi, mask)
            row.update(reference_s=time.time() - t, reference_covers=len(ref), reference_candidates=int(ncr),
                       equal=bool(ref == got))
        rows.append(row)
        log(f"  pair {(i, j)}: M={M}, {len(got)} covers, {nc} candidates, {tn:.2f}s"
            + (f"; reference {row['reference_covers']} covers, {row['reference_candidates']} candidates, "
               f"{row['reference_s']:.1f}s, equal {row['equal']}" if a.reference else ""))
    rec = {"kernel": "cover6_pair", "orbit": a.orbit, "n": a.n, "N": int(E.N), "rows": rows}
    write(rec, a.out or os.path.join(RESULTS, f"cover6_{a.orbit}_n{a.n}.json"))
    return 0 if all(r.get("equal", True) for r in rows) else 1


# ------------------------------------------------------------ cover5-key ----

def cover5_key(a):
    t0 = time.time()
    E, reduce = build(a.orbit, a.n)
    log(f"{a.orbit} n={a.n}: N={E.N}, built in {time.time() - t0:.1f}s; native cover5 {E.native_cover5 is not None}")
    if E.native_cover5 is None:
        raise SystemExit("cover5_pair is not available in stabrank_core")
    rows = []
    ok = True
    for i, j in parse_pairs(a.pairs, E):
        _, mask = plan(E, reduce, i)
        row = {"pivot": i, "partner": j}
        got = {}
        for width in (1, 2):
            E.wide_key = width == 2
            t = time.time()
            got[width], nc = E._pair_covers_native(i, j, mask, 64)
            row[f"key{width}_s"] = time.time() - t
            row[f"key{width}_candidates"] = int(nc)
            row[f"key{width}_covers"] = len(got[width])
        row["equal"] = bool(got[1] == got[2])
        ok &= row["equal"]
        rows.append(row)
        log(f"  pair {(i, j)}: 16-bit key {row['key1_candidates']} candidates {row['key1_s']:.1f}s, "
            f"32-bit key {row['key2_candidates']} candidates {row['key2_s']:.1f}s, {row['key1_covers']} covers, "
            f"equal {row['equal']}")
    rec = {"kernel": "cover5_pair", "orbit": a.orbit, "n": a.n, "N": int(E.N), "rows": rows}
    write(rec, a.out or os.path.join(RESULTS, f"cover5_key_{a.orbit}_n{a.n}.json"))
    return 0 if ok else 1


# -------------------------------------------------------------- stage-a3 ----

def stage_a3(a):
    from cover_census import CoverEnumerator3, _reduce
    from matcher import Matcher, exact_codes, psi_target
    x0s = [tuple(int(v) for v in x.split(",")) for x in a.x0]
    E = CoverEnumerator3(a.orbit, 2)
    Mn = Matcher(E.D, 2, E.F1, E.F2, native=True)
    Mr = Matcher(E.D, 2, E.F1, E.F2, native=False)
    if Mn.native_cls is None:
        raise SystemExit("SliceMatch3Kernel is not available in stabrank_core")
    target = psi_target(a.orbit, 2, E.F1, E.F2)
    covers = []
    for (i, j, _) in E.units():
        Qi, mask = plan(E, _reduce, i)
        got, _, _ = E.pair_covers(5, i, j, Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= a.count:
            break
    covers = covers[:a.count]
    log(f"{a.orbit}: {len(covers)} full 5-covers from the first pivot pairs")
    rows = []
    ok = True
    for x0 in x0s:
        tn = tr = 0.0
        mism = 0
        hist = {}
        for c in covers:
            t = time.time()
            hn, sn = Mn.run(c, x0, target)
            tn += time.time() - t
            t = time.time()
            hr, sr = Mr.run(c, x0, target)
            tr += time.time() - t
            kn = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hn)
            kr = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hr)
            same = (kn == kr and sn["coord_solutions"] == sr["coord_solutions"] and sn["refused"] == sr["refused"]
                    and sn["composite_solutions"] == sr["composite_solutions"] and bool(sn.get("native")))
            mism += not same
            k = ",".join(str(v) for v in sn["coord_solutions"])
            hist[k] = hist.get(k, 0) + 1
        ok &= mism == 0
        rows.append({"x0": list(x0), "covers": len(covers), "native_ms_per_cover": 1e3 * tn / len(covers),
                     "reference_ms_per_cover": 1e3 * tr / len(covers), "mismatches": mism, "coord_hist": hist})
        log(f"  x0 {x0}: native {rows[-1]['native_ms_per_cover']:.2f} ms per cover, reference "
            f"{rows[-1]['reference_ms_per_cover']:.1f} ms, mismatches {mism}")
    rec = {"kernel": "SliceMatch3Kernel", "orbit": a.orbit, "rows": rows}
    write(rec, a.out or os.path.join(RESULTS, f"stage_a3_{a.orbit}.json"))
    return 0 if ok else 1


# ----------------------------------------------------------------- dense ----

def dense(a):
    import slice_cover
    from slice_cover import CoverEnumerator, SliceMatcher, exact_codes
    path = (os.path.join(ROOT, "research", "h6_rank5", "degenerate_covers_v2.json") if a.orbit == "qubit_H"
            else os.path.join(ROOT, "research", "t5_rank5", "degenerate5.json"))
    with open(path) as f:
        doc = json.load(f)
    covers = [tuple(c) for c in doc["covers"] if len(set(c)) == 5]
    step = max(1, len(covers) // a.count)
    sample = covers[::step][:a.count]
    if slice_cover._native_dense() is None:
        raise SystemExit("dense_solve is not available in stabrank_core")
    E = CoverEnumerator(3, orbit=a.orbit)
    M = SliceMatcher(E, 2, native=True)
    log(f"{a.orbit}: {len(covers)} dependent 5-covers in {os.path.relpath(path, ROOT)}, {len(sample)} sampled")
    rows = []
    tn = tr = 0.0
    ok = True
    for c in sample:
        t = time.time()
        hn, sn = M.run(c, a.x0)
        dn = time.time() - t
        os.environ["STABRANK_NO_NATIVE"] = "1"
        t = time.time()
        hr, sr = M.run(c, a.x0)
        dr = time.time() - t
        del os.environ["STABRANK_NO_NATIVE"]
        kn = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hn)
        kr = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hr)
        same = (kn == kr and sn["coord_solutions"] == sr["coord_solutions"] and sn["refused"] == sr["refused"]
                and sn["composite_solutions"] == sr["composite_solutions"] and sn["joined"] == sr["joined"])
        ok &= same
        tn += dn
        tr += dr
        rows.append({"cover": list(c), "kappa": sn["kappa"], "coord_solutions": sn["coord_solutions"],
                     "hits": len(hn), "native_s": dn, "reference_s": dr, "dense_raw": sn.get("dense_raw", 0),
                     "native_candidates": sn["candidates"], "reference_candidates": sr["candidates"], "equal": same})
        log(f"  {c}: coord {sn['coord_solutions']}, hits {len(hn)}, native {dn:.2f}s, reference {dr:.2f}s, "
            f"{'equal' if same else 'MISMATCH'}")
    rec = {"kernel": "dense_solve", "orbit": a.orbit, "x0": a.x0, "n1": 2, "list": os.path.relpath(path, ROOT),
           "rows": rows, "native_s_per_cover": tn / len(sample), "reference_s_per_cover": tr / len(sample)}
    log(f"native {rec['native_s_per_cover']:.2f} s per cover, reference {rec['reference_s_per_cover']:.2f} s")
    write(rec, a.out or os.path.join(RESULTS, f"dense_{a.orbit}.json"))
    return 0 if ok else 1


def dense3(a):
    from cover_census import CoverEnumerator3, _reduce
    from matcher import Matcher, exact_codes, psi_target
    x0 = tuple(int(v) for v in a.x0.split(","))
    E = CoverEnumerator3(a.orbit, 2)
    M = Matcher(E.D, 2, E.F1, E.F2, native=True)
    target = psi_target(a.orbit, 2, E.F1, E.F2)
    bases = []
    for (i, j, _) in E.units():
        Qi, mask = plan(E, _reduce, i)
        got, _, _ = E.pair_covers(5, i, j, Qi, mask)
        for c in sorted(got):
            for x in E.in_span(c):
                six = tuple(sorted(c + (x,)))
                if E.is_full(six):
                    bases.append(six)
                if len(bases) >= a.count:
                    break
            if len(bases) >= a.count:
                break
        if len(bases) >= a.count:
            break
    log(f"{a.orbit}: {len(bases)} B6 bases (a full 5-cover plus a state of its span) at x0 {x0}")
    rows = []
    tn = tr = 0.0
    ok = True
    for c in bases:
        t = time.time()
        hn, sn = M.run(c, x0, target)
        dn = time.time() - t
        os.environ["STABRANK_NO_NATIVE"] = "1"
        t = time.time()
        hr, sr = M.run(c, x0, target)
        dr = time.time() - t
        del os.environ["STABRANK_NO_NATIVE"]
        kn = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hn)
        kr = sorted(sorted(exact_codes(v)[0].tobytes() for v in h["terms"]) for h in hr)
        same = (kn == kr and sn["coord_solutions"] == sr["coord_solutions"] and sn["refused"] == sr["refused"]
                and sn["composite_solutions"] == sr["composite_solutions"] and sn["coord_raw"] == sr["coord_raw"])
        ok &= same
        tn += dn
        tr += dr
        rows.append({"cover": list(c), "kappa": sn["kappa"], "coord_raw": sn["coord_raw"],
                     "coord_solutions": sn["coord_solutions"], "hits": len(hn), "native_s": dn, "reference_s": dr,
                     "dense_raw": sn.get("dense_raw", 0), "equal": same})
        log(f"  {c}: kappa {sn['kappa']}, raw {sn['coord_raw']}, coord {sn['coord_solutions']}, hits {len(hn)}, "
            f"native {dn:.2f}s, reference {dr:.2f}s, {'equal' if same else 'MISMATCH'}")
    rec = {"kernel": "dense_solve (qutrit matcher)", "orbit": a.orbit, "x0": list(x0), "kind": "B6", "rows": rows,
           "native_s_per_cover": tn / max(1, len(bases)), "reference_s_per_cover": tr / max(1, len(bases))}
    log(f"native {rec['native_s_per_cover']:.2f} s per cover, reference {rec['reference_s_per_cover']:.2f} s")
    write(rec, a.out or os.path.join(RESULTS, f"dense3_{a.orbit}_B6.json"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cover6")
    p.add_argument("orbit")
    p.add_argument("n", type=int)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cover6)
    p = sub.add_parser("cover5-key")
    p.add_argument("orbit")
    p.add_argument("n", type=int)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cover5_key)
    p = sub.add_parser("stage-a3")
    p.add_argument("orbit")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--x0", nargs="+", default=["0,0", "2,2"])
    p.add_argument("--out")
    p.set_defaults(fn=stage_a3)
    p = sub.add_parser("dense3")
    p.add_argument("orbit")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--x0", default="2,2")
    p.add_argument("--out")
    p.set_defaults(fn=dense3)
    p = sub.add_parser("dense")
    p.add_argument("orbit")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--x0", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(fn=dense)
    a = ap.parse_args(argv[1:])
    try:
        os.nice(19)
    except OSError:
        pass
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
