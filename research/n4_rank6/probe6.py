"""Measurements for the rank-6 exclusion design of |N>^4
(docs/notes/n4_rank6_design.md): the first-slice filters of filters6.py
on the G_2 orbit representatives of degenerate6.py, planted controls, and
the reference matcher's rates on the shapes the filters do not cover.

    probe6.py planted --reps NPZ [--count K] [--x0 2,2]
        Planted six-term decompositions over kappa = 1 dependent bases (B6)
        and over one-block bases (C6, patterns (2,1,1,1,1) and (3,1,1,1)):
        the filter's first-slice solution list must contain the planted
        codes and equal the reference solve_slice3's list.
    probe6.py rate --reps NPZ [--count K] [--x0 2,2] [--reference R]
        The filters on sampled real orbit representatives: B6 kappa = 1,
        C6 (2,1,1,1,1) with kappa 0 and 1, C6 (3,1,1,1); per-item times and
        survivor counts; the first R items of each class also through the
        reference's solve_slice3 for agreement on the solution list.
    probe6.py kappa2 --reps NPZ [--count K] [--x0 2,2]
        The reference Matcher.run on B6 representatives with kappa = 2 (the
        2-parameter dense solve), per-item time.
    probe6.py two-blocks --reps NPZ [--count K] [--x0 2,2]
        The reference Matcher.run on C6 (2,2,1,1) representatives.
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from cover_census import CoverEnumerator3  # noqa: E402
from matcher import (E1, PTS, Block, Matcher, add, exact_codes, pidx, psi_target, slice_base, solve_slice3,  # noqa: E402
                     vector_target)
from degenerate6 import decode  # noqa: E402
from filters6 import Filters  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def log(s):
    print(s, flush=True)


def random_shape_term(o, x0, rng, kind=None):
    """research/qutrit_m4_rank5/driver.random_shape_term: a random
    four-qutrit stabilizer term with base slice o.u at x0."""
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


def setup(a):
    try:
        os.nice(19)
    except OSError:
        pass
    E = CoverEnumerator3("N", 2)
    Mt = Matcher(E.D, 2, E.F1, E.F2, native=True)
    Fl = Filters(Mt)
    reps = np.load(a.reps)
    B6 = decode(reps["B6_orbit_codes"], 6, E.N)
    kB = reps["B6_orbit_kappa"]
    C6 = decode(reps["C6_orbit_codes"], 6, E.N) if "C6_orbit_codes" in reps else None
    x0 = tuple(int(v) for v in a.x0.split(","))
    return E, Mt, Fl, B6, kB, C6, x0


def pattern(ms):
    ms = list(ms)
    return tuple(sorted((ms.count(u) for u in set(ms)), reverse=True))


def classes_c6(C6, Mt):
    """Orbit representatives of C6 by class: (2,1,1,1,1) with kappa 0 and 1,
    (3,1,1,1), (2,2,1,1), and the rest; kappa from the numerical rank of the
    distinct states."""
    out = {"(2,1,1,1,1) k0": [], "(2,1,1,1,1) k1": [], "(3,1,1,1)": [], "(2,2,1,1)": [], "rest": []}
    for ms in C6:
        pt = pattern(ms)
        if pt == (2, 1, 1, 1, 1):
            d = sorted(set(int(u) for u in ms))
            dep = np.linalg.matrix_rank(Mt.C[:, d], tol=1e-8) < len(d)
            out["(2,1,1,1,1) k1" if dep else "(2,1,1,1,1) k0"].append(tuple(int(u) for u in ms))
        elif pt == (3, 1, 1, 1):
            out["(3,1,1,1)"].append(tuple(int(u) for u in ms))
        elif pt == (2, 2, 1, 1):
            out["(2,2,1,1)"].append(tuple(int(u) for u in ms))
        else:
            out["rest"].append(tuple(int(u) for u in ms))
    return out


def reference_first_slice(Mt, cover, x0, target):
    """The reference's first coordinate slice: solve_slice3 against the
    initial family, as the set of (combo, Ssel) with the block treatment."""
    distinct = sorted(set(int(u) for u in cover))
    mult = {u: list(cover).count(u) for u in distinct}
    b1, b2, bC = target.rhs(x0)
    from matcher import family_from
    fam = family_from(Mt.U1[distinct], Mt.U2[distinct], Mt.C[:, distinct], b1, b2, bC)
    blocks = [Block(Mt.options(u), mult[u], i) for i, u in enumerate(distinct) if mult[u] > 1]
    ords = [i for i in range(len(distinct)) if mult[distinct[i]] == 1]
    arrays = [Mt.options(distinct[i]).arrays() for i in ords]
    rhs = target.rhs(add(x0, E1))
    t0 = time.time()
    sols = solve_slice3(arrays, blocks, fam, rhs, Mt.rng, stats={}, where="reference first slice")
    return {(tuple(int(c) for c in combo), tuple(tuple(S) for S in Ssel)) for combo, Ssel, _ in sols}, time.time() - t0


def filter_first_slice(Fl, cover, x0, target):
    distinct = sorted(set(int(u) for u in cover))
    if len(distinct) == 6:
        sols, st = Fl.b6_slice(cover, x0, target, E1)
    else:
        sols, st = Fl.c6_slice(cover, x0, target, E1)
    if sols is None:
        return None, st
    return {(tuple(int(c) for c in combo), tuple(tuple(S) for S in Ssel)) for combo, Ssel, _ in sols}, st


def planted(a):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    rng = np.random.default_rng(a.seed)
    cl = classes_c6(C6, Mt)
    cases = {"B6 kappa1": [tuple(int(u) for u in B6[i]) for i in np.flatnonzero(kB == 1)],
             "C6 (2,1,1,1,1) k1": cl["(2,1,1,1,1) k1"]}
    rec = {"x0": list(x0), "cases": {}}
    ok_all = True
    for name, lst in cases.items():
        done, agree, found, tf, tr = 0, 0, 0, 0.0, 0.0
        while done < a.count:
            base = lst[int(rng.integers(0, len(lst)))]
            terms = []
            for u in base:
                t, _ = random_shape_term(Mt.options(u), x0, rng)
                terms.append(t)
            T = np.column_stack(terms)
            if np.linalg.matrix_rank(T, tol=1e-8) < len(base):
                continue
            coeffs = rng.integers(1, 4, size=len(base)) * rng.choice([1, -1], size=len(base))
            vec = T @ coeffs.astype(complex)
            target = vector_target(vec, 2, Mt.F1, Mt.F2)
            if sorted(slice_base(Mt, T, x0)) != sorted(base):
                raise AssertionError("planted base differs from the intended one")
            # the planted codes at x0 + e1 per distinct state (ordinary terms) and the block's classes
            distinct = sorted(set(base))
            t1 = time.time()
            fs, st = filter_first_slice(Fl, base, x0, target)
            tf += time.time() - t1
            if fs is None:
                raise AssertionError(f"{name}: shape outside the filter: {st}")
            rs, dt = reference_first_slice(Mt, base, x0, target)
            tr += dt
            same = fs == rs
            agree += same
            # planted solution present: the planted combo over the ordinary terms
            pl_combo = []
            for i, u in enumerate(distinct):
                if base.count(u) == 1:
                    o = Mt.options(u)
                    v = terms[base.index(u)][pidx(add(x0, E1)) * 9:(pidx(add(x0, E1)) + 1) * 9]
                    pl_combo.append(o.absent if np.linalg.norm(v) < 1e-9 else 3 * o.code_of(v)[0] + o.code_of(v)[1])
            present = any(tuple(c) == tuple(pl_combo) for c, _ in fs) if len(distinct) < 6 else \
                any(tuple(c) == tuple(pl_combo) for c, _ in fs)
            found += present
            if not same or not present:
                log(f"  {name}: base {base}: filter {len(fs)} solutions, reference {len(rs)}, planted present {present}")
                ok_all = False
            done += 1
        rec["cases"][name] = {"planted": done, "agree": agree, "planted_found": found, "filter_s": tf / done,
                              "reference_s": tr / done}
        log(f"{name}: {done} planted, {agree} first-slice lists equal to the reference, {found} planted "
            f"solutions found; filter {tf / done:.3f} s, reference {tr / done:.3f} s per instance")
    rec["pass"] = ok_all
    out = a.out or os.path.join(RESULTS, "probe6_planted.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"planted: {'PASS' if ok_all else 'FAIL'}; wrote {os.path.relpath(out, ROOT)}")
    return 0 if ok_all else 1


def rate(a):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    target = psi_target("N", 2, Mt.F1, Mt.F2)
    rng = np.random.default_rng(a.seed)
    cl = classes_c6(C6, Mt)
    cases = {"B6 kappa1": [tuple(int(u) for u in B6[i]) for i in np.flatnonzero(kB == 1)],
             "C6 (2,1,1,1,1) k1": cl["(2,1,1,1,1) k1"]}
    plain = {"C6 (2,1,1,1,1) k0": cl["(2,1,1,1,1) k0"], "C6 (3,1,1,1)": cl["(3,1,1,1)"]}
    rec = {"x0": list(x0), "cases": {}}
    ok_all = True
    for name, lst in cases.items():
        pick = [lst[t] for t in np.linspace(0, len(lst) - 1, min(a.count, len(lst))).astype(int)]
        secs, dsecs, surv, survp, sols, agree, checked, tref, parts, slack = [], [], [], [], [], 0, 0, 0.0, [], []
        t_case = time.time()
        for n, base in enumerate(pick):
            if a.budget and time.time() - t_case > a.budget:
                log(f"  {name}: budget reached after {n} items")
                break
            t1 = time.time()
            fs, st = filter_first_slice(Fl, base, x0, target)
            secs.append(time.time() - t1)
            if fs is None:
                log(f"  {name}: outside the filter: {st}")
                ok_all = False
                continue
            dsecs.append(st["seconds_dense"])
            surv.append(st.get("survivors5", st.get("survivors")))
            survp.append(st["survivors_p1"])
            sols.append(len(fs))
            parts.append(st.get("parts", st.get("g", 0) + 1))
            slack.append(st.get("slack", 0))
            if n < a.reference:
                rs, dt = reference_first_slice(Mt, base, x0, target)
                tref += dt
                checked += 1
                agree += fs == rs
                if fs != rs:
                    log(f"  {name}: MISMATCH on {base}: filter {len(fs)}, reference {len(rs)}")
                    ok_all = False
        rec["cases"][name] = {"items": len(lst), "run": len(secs), "mean_s": float(np.mean(secs)),
                              "median_s": float(np.median(secs)), "max_s": float(np.max(secs)),
                              "dense_mean_s": float(np.mean(dsecs)), "survivors_mean": float(np.mean(surv)),
                              "survivors_max": int(np.max(surv)), "survivors_p1_mean": float(np.mean(survp)),
                              "with_solutions": int(np.sum(np.array(sols) > 0)),
                              "solutions_hist": {str(k): int(v) for k, v in zip(*np.unique(sols, return_counts=True))},
                              "parts_hist": {str(k): int(v) for k, v in zip(*np.unique(parts, return_counts=True))},
                              "slack_hist": {str(k): int(v) for k, v in zip(*np.unique(slack, return_counts=True))},
                              "reference_checked": checked, "reference_agree": agree,
                              "reference_mean_s": tref / max(1, checked), "seconds": secs}
        log(f"{name}: {len(lst)} orbit reps, {len(secs)} run: filter mean {np.mean(secs):.4f} s (dense "
            f"{np.mean(dsecs):.4f}), median {np.median(secs):.4f}, max {np.max(secs):.3f}; survivors mean "
            f"{np.mean(surv):.1f} max {np.max(surv)}, after the P1 whole-vector test {np.mean(survp):.2f}; items "
            f"with first-slice solutions {rec['cases'][name]['with_solutions']} (hist "
            f"{rec['cases'][name]['solutions_hist']}); parts {rec['cases'][name]['parts_hist']}, slack "
            f"{rec['cases'][name]['slack_hist']}; reference {agree}/{checked} equal at {tref / max(1, checked):.3f} s")
    for name, lst in plain.items():
        pick = [lst[t] for t in np.linspace(0, len(lst) - 1, min(a.count, len(lst))).astype(int)]
        secs, tsl, raws = [], [], []
        t_case = time.time()
        for n, base in enumerate(pick):
            if a.budget and time.time() - t_case > a.budget:
                log(f"  {name}: budget reached after {n} items")
                break
            rs, dt = reference_first_slice(Mt, base, x0, target)
            tsl.append(dt)
            t1 = time.time()
            hits, st = Mt.run(base, x0, target)
            secs.append(time.time() - t1)
            raws.append(st.get("coord_raw", [0])[0])
        rec["cases"][name] = {"items": len(lst), "run": len(secs), "reference_run_mean_s": float(np.mean(secs)),
                              "reference_run_median_s": float(np.median(secs)), "reference_run_max_s": float(np.max(secs)),
                              "reference_first_slice_mean_s": float(np.mean(tsl)),
                              "first_slice_raw_hist": {str(k): int(v) for k, v in zip(*np.unique(raws, return_counts=True))}}
        log(f"{name}: {len(lst)} orbit reps, {len(secs)} run through the reference: Matcher.run mean "
            f"{np.mean(secs):.4f} s, median {np.median(secs):.4f}, max {np.max(secs):.3f}; first slice alone "
            f"{np.mean(tsl):.4f} s; first-slice solutions {rec['cases'][name]['first_slice_raw_hist']}")
    out = a.out or os.path.join(RESULTS, f"probe6_rate_x{x0[0]}{x0[1]}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"rate: {'consistent' if ok_all else 'INCONSISTENT'}; wrote {os.path.relpath(out, ROOT)}")
    return 0 if ok_all else 1


def run_reference(a, name, lst):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    target = psi_target("N", 2, Mt.F1, Mt.F2)
    pick = [lst[t] for t in np.linspace(0, len(lst) - 1, min(a.count, len(lst))).astype(int)]
    rows = []
    t_all = time.time()
    for base in pick:
        if a.budget and time.time() - t_all > a.budget:
            log("  budget reached")
            break
        t1 = time.time()
        try:
            hits, st = Mt.run(base, x0, target)
            err = None
        except Exception as ex:                      # BudgetExceeded, UnpinnedFamily, AssertionError
            hits, st, err = [], Mt.last_stats, f"{type(ex).__name__}: {ex}"
        dt = time.time() - t1
        rows.append({"cover": list(base), "seconds": dt, "kappa": st.get("kappa"), "coord_raw": st.get("coord_raw"),
                     "coord_solutions": st.get("coord_solutions"), "hits": len(hits), "error": err,
                     "dense_raw": st.get("dense_raw")})
        log(f"  {base}: {dt:.1f} s, kappa {st.get('kappa')}, coord_raw {st.get('coord_raw')}, hits {len(hits)}"
            f"{', ' + err if err else ''}")
    rec = {"class": name, "x0": list(x0), "items": len(lst), "rows": rows,
           "mean_s": float(np.mean([r["seconds"] for r in rows])) if rows else None}
    out = a.out or os.path.join(RESULTS, f"probe6_{name}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"{name}: {len(rows)} runs, mean {rec['mean_s']} s; wrote {os.path.relpath(out, ROOT)}")
    return 0


def kappa2(a):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    lst = [tuple(int(u) for u in B6[i]) for i in np.flatnonzero(kB >= 2)]
    log(f"B6 with kappa >= 2: {len(lst)} orbit reps (kappa histogram {dict(zip(*np.unique(kB[kB >= 2], return_counts=True)))})")
    if a.max_cand:
        import functools
        import matcher as mmod
        mmod.solve_slice3 = functools.partial(mmod.solve_slice3, max_cand=a.max_cand)
        log(f"dense candidate cap raised to {a.max_cand}")
    return run_reference(a, "kappa2", lst)


def c6k1(a):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    cl = classes_c6(C6, Mt)
    return run_reference(a, "c6k1", cl["(2,1,1,1,1) k1"])


def two_blocks(a):
    E, Mt, Fl, B6, kB, C6, x0 = setup(a)
    cl = classes_c6(C6, Mt)
    return run_reference(a, "two_blocks", cl["(2,2,1,1)"] + cl["rest"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("planted", planted), ("rate", rate), ("kappa2", kappa2), ("two-blocks", two_blocks),
                     ("c6k1", c6k1)):
        p = sub.add_parser(name)
        p.add_argument("--reps", required=True)
        p.add_argument("--max-cand", type=int, default=0)
        p.add_argument("--count", type=int, default=20)
        p.add_argument("--reference", type=int, default=5)
        p.add_argument("--x0", default="2,2")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--budget", type=float, default=0)
        p.add_argument("--out")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
