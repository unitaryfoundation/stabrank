"""Resumable driver for the rank-5 exclusion of |H>^6 by an all-visible base
slice (docs/notes/h6_rank5_exclusion.md).

Stage A (this driver): every full 5-cover of |H>^3 with five distinct,
linearly independent base states, one per orbit of the unitary symmetry
group of |H>^3, enumerated per pivot pair (i, j) by
verify_challenge/slice_cover.CoverEnumerator.pair_covers, and matched at
the four base points x_0 (one per Hamming weight) by SliceMatcher.run.

Stages B (dependent base states) and C (repeated base states) are not
implemented; the note says what they need. Nothing this driver produces is
a certificate until B and C exist and the controls pass.

Commands
  partition [--target-s S]  write partition.json: pivot pairs grouped into
                            batches of about S seconds at the measured rates
  run BATCH                 run one batch, writing results/batch_BATCH.json
                            (skipped when the file exists: resume by rerunning)
  status                    coverage and totals over the result files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from slice_cover import CoverEnumerator, SliceMatcher, x0_reps, confirm_decomposition, _reduce  # noqa: E402

PARTITION = os.path.join(HERE, "partition.json")
RESULTS = os.path.join(HERE, "results")
N1 = 3


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def pairs_of(E):
    """Every (pivot, partner, member count) unit in the pivot order of the
    enumerator."""
    out = []
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        for j in partners:
            j = int(j)
            M = int(np.count_nonzero(members > j)) - (1 if i > j else 0)
            out.append((i, j, M))
    return out


def partition(args):
    E = CoverEnumerator(N1)
    units = pairs_of(E)
    # cost model: kernel time is quadratic in M (residue array M x M), plus a
    # per-unit floor; the matcher cost is proportional to the covers found,
    # which is unknown before the run, so the partition is by kernel cost only
    # and the per-batch matcher time is recorded in the results.
    costs = [args.floor_s + args.k_s * (M / 1000.0) ** 2 for _, _, M in units]
    batches, cur, acc = [], [], 0.0
    for u, c in zip(units, costs):
        cur.append(list(u))
        acc += c
        if acc >= args.target_s:
            batches.append(cur)
            cur, acc = [], 0.0
    if cur:
        batches.append(cur)
    rec = {"orbit": "qubit_H", "m": 6, "rank": 5, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "pivot_orbits": int(E.info["orbits"]), "units": len(units), "batches": len(batches),
           "cost_model": {"floor_s": args.floor_s, "k_s_per_M2_over_1e6": args.k_s},
           "estimated_kernel_s": float(sum(costs)), "git": git_commit(),
           "batch_units": batches}
    with open(PARTITION, "w") as f:
        json.dump(rec, f)
    print(f"{len(units)} pivot pairs, {len(batches)} batches, kernel estimate "
          f"{sum(costs) / 3600:.1f} CPU-h at the stored rates; wrote {PARTITION}")


def run(args):
    with open(PARTITION) as f:
        part = json.load(f)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"batch_{args.batch}.json")
    if os.path.exists(out) and not args.force:
        print(f"{out} exists; skipping")
        return 0
    units = part["batch_units"][args.batch]
    E = CoverEnumerator(N1)
    assert E.N == part["N"] and E.info["order"] == part["group_order"]
    M = SliceMatcher(E, N1)
    t0 = time.time()
    rec = {"batch": args.batch, "units": units, "git": git_commit(), "covers": 0, "candidates": 0,
           "dependent": 0, "matched": 0, "hits": [], "basis_solution_hist": {}, "kernel_s": 0.0,
           "match_s": 0.0}
    for i, j, _ in units:
        tk = time.time()
        Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
        members, _ = E.pivot_plan(i)
        mask = np.zeros(E.N, dtype=bool)
        mask[members] = True
        covers, nc = E.pair_covers(5, i, j, Qi, mask)
        rec["kernel_s"] += time.time() - tk
        rec["covers"] += len(covers)
        rec["candidates"] += nc
        tm = time.time()
        for cover in sorted(covers):
            for x0 in x0_reps(N1):
                try:
                    hits, st = M.run(cover, x0)
                except ValueError:
                    rec["dependent"] += 1            # stage B territory: recorded, not decided
                    continue
                rec["matched"] += 1
                k = ",".join(str(b) for b in st["basis_solutions"])
                rec["basis_solution_hist"][k] = rec["basis_solution_hist"].get(k, 0) + 1
                for h in hits:
                    res, c = confirm_decomposition(h, N1 + E.n)
                    rec["hits"].append({"cover": list(cover), "x0": x0, "residual": res,
                                        "coeffs": [[z.real, z.imag] for z in c],
                                        "terms": [[[z.real, z.imag] for z in t] for t in h]})
        rec["match_s"] += time.time() - tm
    rec["wall_s"] = time.time() - t0
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    with open(out, "w") as f:
        json.dump(rec, f)
    print(f"batch {args.batch}: {len(units)} units, {rec['covers']} covers, {rec['matched']} matched, "
          f"{rec['dependent']} dependent (undecided), {len(rec['hits'])} hits, "
          f"kernel {rec['kernel_s']:.0f}s, match {rec['match_s']:.0f}s")
    return 2 if rec["hits"] else 0


def status(args):
    with open(PARTITION) as f:
        part = json.load(f)
    done, tot = 0, {"covers": 0, "matched": 0, "dependent": 0, "hits": 0, "kernel_s": 0.0, "match_s": 0.0}
    for b in range(part["batches"]):
        p = os.path.join(RESULTS, f"batch_{b}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rec = json.load(f)
        done += 1
        for k in tot:
            tot[k] += len(rec[k]) if k == "hits" else rec[k]
    print(f"{done}/{part['batches']} batches done; {tot}")
    if done:
        print(f"mean kernel {tot['kernel_s'] / done:.0f}s and match {tot['match_s'] / done:.0f}s per batch")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partition")
    p.add_argument("--target-s", type=float, default=1800.0)
    p.add_argument("--floor-s", type=float, default=0.05)
    p.add_argument("--k-s", type=float, default=8.0, help="kernel seconds per pair at M = 1000")
    p.set_defaults(fn=partition)
    p = sub.add_parser("run")
    p.add_argument("batch", type=int)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=run)
    p = sub.add_parser("status")
    p.set_defaults(fn=status)
    args = ap.parse_args(argv[1:])
    os.nice(19)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
