"""Does any annealing schedule recover a decomposition known to be in-family?

The rank-9 product decomposition of T3^{ot 4} reconstructs to 1.3e-15 through
orbit_columns at shift_unit=2, orbit_sizes=(2,2,2,1,1,1), so the target sits
inside the ansatz exactly. 138 anneals at the default schedule all plateaued
near 0.26 and none came within 0.23 of it. Either some schedule finds it, in
which case that schedule is what a rank-8 attempt should use, or none does and
the annealer needs algorithmic work rather than more cores.

Configurations are parametrised by total proposal budget rather than by
cooling rate, since budget is what costs wall-clock: steps to cool from T0 to
Tmin is log(Tmin/T0)/log(rate), and budget is that times iterations_at_temp.
Two schedules with the same budget but different shapes (slow cooling with few
iterations per temperature, or the reverse) explore differently, and both
shapes appear at each budget.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank.symmetric import SymmetricSAConfig, run_symmetric_sa
from stabrank_verify import target_vector

TMIN = 1e-4
SOLVED = 1e-10


def budget(t0, rate, iters):
    return int(math.log(TMIN / t0) / math.log(rate)) * iters


def configs():
    """(label, SymmetricSAConfig) across schedule shape and budget."""
    out = []
    for t0 in (0.1, 0.5, 2.0):
        for rate, iters in ((0.9985, 600), (0.99985, 600), (0.9985, 6000),
                            (0.999985, 600), (0.99985, 6000), (0.9985, 60000)):
            for reseed in (0.02, 0.15):
                b = budget(t0, rate, iters)
                out.append((
                    f"T0={t0} rate={rate} iters={iters} reseed={reseed} budget={b/1e6:.1f}M",
                    dict(initial_temperature=t0, cooling_rate=rate,
                         iterations_at_temp=iters, reseed_prob=reseed)))
    return out


ANSATZ = [(2, (2, 2, 2, 1, 1, 1)),      # the shape the known solution lives in
          (1, (2, 2, 2, 2, 1)),
          (1, (4, 4, 1))]


def one(job):
    (ci, label, kw), (g, sizes), seed, rank, cap = job
    psi = np.array([complex(x) for x in target_vector("T3", 4)]).ravel()
    psi /= np.linalg.norm(psi)
    cfg = SymmetricSAConfig(shift_unit=g, orbit_sizes=sizes, min_temperature=TMIN,
                            early_exit=SOLVED, **kw)
    t0 = time.time()
    try:
        err, _, _ = run_symmetric_sa(psi, 4, 3, cfg, seed=seed, engine="cpp")
    except Exception as exc:
        return dict(label=label, g=g, sizes=list(sizes), seed=seed, rank=rank,
                    error=str(exc)[:120], secs=time.time() - t0)
    return dict(label=label, g=g, sizes=list(sizes), seed=seed, rank=rank,
                residual=float(err), solved=bool(err < SOLVED),
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rank", type=int, default=9)
    ap.add_argument("--max-secs", type=float, default=1800, help="skip configs slower than this")
    ap.add_argument("--out", default="sweep.jsonl")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="offset so hosts explore different seeds")
    a = ap.parse_args()

    cfgs = [(i, lab, kw) for i, (lab, kw) in enumerate(configs())]
    jobs = [(c, an, a.seed_base + s, a.rank, a.max_secs)
            for c in cfgs for an in ANSATZ for s in range(a.seeds)]
    # cheapest schedules first, so a truncated run still covers every shape
    jobs.sort(key=lambda j: budget(j[0][2]["initial_temperature"],
                                   j[0][2]["cooling_rate"],
                                   j[0][2]["iterations_at_temp"]))
    print(f"{len(jobs)} jobs, {a.workers} workers, writing {a.out}", flush=True)

    deadline = time.time() + a.hours * 3600
    done = solved = 0
    with open(a.out, "a") as f, ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            if r.get("solved"):
                solved += 1
                print(f"SOLVED  {r['label']}  ansatz g={r['g']} {r['sizes']} "
                      f"seed={r['seed']} in {r['secs']}s", flush=True)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)} done, {solved} solved", flush=True)
            if time.time() > deadline:
                print("time budget reached", flush=True)
                break
    print(f"FINISHED {done} runs, {solved} solved", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
