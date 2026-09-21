"""Every irreducible (minimal) rank-5 decomposition of |M>^m up to the
unitary symmetry group, with the slice-and-lift test on each, as
resumable work units for a many-core machine.

Why: any rank-5 decomposition of |M>^(m+1) with no minimal slice has, at
each slice, an irreducible rank-5 decomposition of |M>^m; the constructions
note showed every candidate at the record cells must be of that shape. If
some irreducible rank-5 decomposition of |M>^m lifts (slice_lift.lifts),
|M>^(m+1) has rank 5; if none does, chi(|M>^(m+1)) >= 6 follows once the
list is complete.

Completeness. A minimal five-set {i, j, a, b, c}: move the member of the
lowest symmetry orbit to that orbit's representative i (so every member
lies in an orbit with root >= i), then apply an element of the stabilizer
of i so that the least index among the other four is minimal in its
stabilizer orbit; call it j. Projecting psi and the dictionary off s_j,
{i, a, b, c} is a minimal rank-4 decomposition of the projected target (a
three-subset spanning it would give a four-subset of the five-set spanning
psi), which the compiled pivot-pair search lists completely for pivot i
with members restricted to orbits at or above i. So the units are (i, j)
with i over orbit representatives and j over stabilizer-orbit minima among
states with root >= i, and every minimal five-set appears in at least one
unit. Hits are re-solved in the full space and kept only if minimal.

Usage:
    rank5_full.py ORBIT --m 3 --out DIR [--workers 16] [--pivots 0,1,2]
                  [--max-units N]
Each unit writes DIR/unit_<i>_<j>.json (skipped if present, so a rerun
resumes); DIR/summary.json is rewritten after every unit.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker process; the pool supplies the parallelism

import argparse
import json
import multiprocessing as mp
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from slice_lift import (decompositions_with_pivot, lifts, lifts_qubit,  # noqa: E402
                        stabilizer_orbit_labels, confirm_lift)
from stabrank_verify import ORBIT_P, orbit_state  # noqa: E402

_G = {}


def minimal(D, psi, cols):
    A = D[:, list(cols)]
    if np.linalg.matrix_rank(A, tol=1e-8) < len(cols):
        return False
    x, *_ = np.linalg.lstsq(A, psi, rcond=None)
    if np.linalg.norm(A @ x - psi) > 1e-9:
        return False
    for drop in cols:
        sub = [c for c in cols if c != drop]
        B = D[:, sub]
        y, *_ = np.linalg.lstsq(B, psi, rcond=None)
        if np.linalg.norm(B @ y - psi) < 1e-9:
            return False
    return True


def unit(args):
    i, j, members = args
    D, psi, m, p, alpha = _G["D"], _G["psi"], _G["m"], _G["p"], _G["alpha"]
    t0 = time.time()
    e = D[:, j] / np.linalg.norm(D[:, j])
    psi_p = psi - e * (e.conj() @ psi)
    Dp = D - np.outer(e, e.conj() @ D)
    mem = np.array([k for k in members if k != j])
    quads = decompositions_with_pivot(psi_p / np.linalg.norm(psi_p), Dp, int(i), 4, members=mem)
    found = set()
    for q in quads:
        cols = tuple(sorted(set(int(c) for c in q) | {int(j)}))
        if len(cols) == 5 and minimal(D, psi, cols):
            found.add(cols)
    lifter = lifts if p == 3 else lifts_qubit
    lifted = []
    for cols in sorted(found):
        states = [D[:, c] for c in cols]
        d, *_ = np.linalg.lstsq(np.column_stack(states), psi, rcond=None)
        L, (nm, gap) = lifter(states, d, alpha, psi, m)
        ok = [confirm_lift(t, alpha, psi) for t in L]
        lifted.append({"cols": list(cols), "lifts": len(L), "slice_matches": nm, "gap": gap,
                       "confirm": [float(r) for r in ok]})
    return {"i": int(i), "j": int(j), "found": [list(c) for c in sorted(found)], "lift": lifted,
            "seconds": round(time.time() - t0, 2)}


def init(D, psi, m, p, alpha):
    _G.update(D=D, psi=psi, m=m, p=p, alpha=alpha)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--pivots", default="", help="comma-separated positions in the sorted representative list")
    ap.add_argument("--max-units", type=int, default=0)
    a = ap.parse_args(argv[1:])
    os.makedirs(a.out, exist_ok=True)
    p = ORBIT_P[a.orbit]
    t0 = time.time()
    D = dictionary(p, a.m)
    psi = psi_for(a.orbit, a.m)
    psi = psi / np.linalg.norm(psi)
    alpha = np.array([complex(x) for x in orbit_state(a.orbit)]).ravel()
    reps, info = symmetry_orbit_reps(a.orbit, a.m, D, antiunitary=False)
    roots = info["roots"]
    reps = np.sort(reps)
    chosen = [int(x) for x in a.pivots.split(",") if x != ""] if a.pivots else list(range(len(reps)))
    units = []
    for pos in chosen:
        i = int(reps[pos])
        orbit_size = int(np.count_nonzero(roots == i))
        labels, _ = stabilizer_orbit_labels(info["perms"], i, stabilizer_order=info["order"] // orbit_size)
        cand = np.flatnonzero(roots >= i)
        _, first = np.unique(labels[cand], return_index=True)
        partners = [int(j) for j in cand[first] if j != i]
        for j in partners:
            if not os.path.exists(os.path.join(a.out, f"unit_{i}_{j}.json")):
                units.append((i, j, cand))
    if a.max_units:
        units = units[:a.max_units]
    print(f"{a.orbit}^{a.m}: {D.shape[1]} states, group order {info['order']}, {len(reps)} pivot orbits; "
          f"{len(units)} units to run on {a.workers} workers, host {socket.gethostname()} "
          f"[setup {time.time() - t0:.0f}s]", flush=True)
    done = 0
    total_found = 0
    total_lifts = 0
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(D, psi, a.m, p, alpha)) as pool:
        for res in pool.imap_unordered(unit, units, chunksize=1):
            with open(os.path.join(a.out, f"unit_{res['i']}_{res['j']}.json"), "w") as f:
                json.dump(res, f)
            done += 1
            total_found += len(res["found"])
            nl = sum(x["lifts"] for x in res["lift"])
            total_lifts += nl
            if res["found"] or done % 50 == 0:
                print(f"  unit ({res['i']}, {res['j']}): {len(res['found'])} irreducible rank-5, "
                      f"{nl} lifts [{res['seconds']}s]; done {done}/{len(units)}, "
                      f"found {total_found}, lifts {total_lifts}, {time.time() - t0:.0f}s", flush=True)
            if nl:
                print(f"LIFT FOUND: rank-5 decomposition of {a.orbit}^{a.m + 1} from unit ({res['i']}, {res['j']}): "
                      f"{[x for x in res['lift'] if x['lifts']]}", flush=True)
            with open(os.path.join(a.out, "summary.json"), "w") as f:
                json.dump({"orbit": a.orbit, "m": a.m, "units_total": len(units), "units_done": done,
                           "found": total_found, "lifts": total_lifts, "seconds": time.time() - t0}, f)
    print(f"finished {done} units: {total_found} irreducible rank-5 decompositions (unit representatives), "
          f"{total_lifts} lifts, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
