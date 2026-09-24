"""One batch of the rank-5 exclusion of |M>^4 (M = N or H3) by a two-qutrit
all-visible base slice.

    batch.py ORBIT K [--partition P] [--out-dir D] [--force] [--verbose]

Batch K of partition_ORBIT.json is either a stage A batch (a list of pivot
pairs (i, j) of the 5-cover enumeration of |M>^2: the reference kernel of
cover_census lists the full 5-covers of the pair and the matcher runs each
at the cell's base point x0), a stage B batch (a list of indices into
degenerate_covers_ORBIT.json: full 5-covers of five distinct but dependent
states, matched through the coefficient family) or a stage C batch (covers
with a repeated state, matched through the block treatment). See
docs/notes/qutrit_m4_rank5_exclusion.md and research/qutrit_m4_rank5/README.md.

Every hit the matcher returns (a candidate decomposition of |M>^4 whose nine
slices all satisfy the modular equations) is re-decided here exactly: the
terms are rebuilt from their phase codes and |M>^4 is tested against their
span mod 2013265921 and in floating point (common.decide_terms). A hit with
|M>^4 in the span is a decomposition with at most five terms, and the batch
exits 2 with DECOMPOSITION FOUND on its last stdout line; a run that raises
(UnpinnedFamily from the block reconstruction included), or a hit on which
the two tests disagree, is recorded under `undecided`, which fails the batch
(exit 1). Exit 0 when every cover was matched and no hit is a decomposition.

--max-seconds S is the batch's wall-clock guard against the stage C tail
(one cover has run for hours): the matcher checks the deadline inside its
join and composite loops and aborts the running cover with BudgetExceeded,
and every cover not yet started is recorded as undecided with the reason
"not run", so the batch ends with a record that names the covers left over
instead of running on. 0 (the default) means no guard. The aborted batch
exits 1 and fails the aggregate; re-run it with --force and a larger cap
or without one.

Output: <out-dir>/batch_<K>.json (default results/ORBIT/), skipped when it
exists (resume by rerunning; --force overwrites). The record's
deterministic part (geometry, partition and degenerate-list hashes, cell,
counts, the coordinate-slice solution histogram, the hits as phase codes
with their decisions, `undecided`) is hashed as `deterministic_sha256`, so
that a re-run on any machine can be compared bit for bit; timing, host and
version fields follow, then `sha256` over the whole record.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import M, N1, N2, RANK, X0  # noqa: E402
from cover_census import P1, P2, CoverEnumerator3, Field3, _reduce  # noqa: E402
from matcher import Budget, Matcher, psi_target  # noqa: E402


def past_deadline(matcher):
    return matcher.budget is not None and matcher.budget.deadline is not None \
        and time.time() > matcher.budget.deadline


def match_cover(orbit, matcher, target, cover, x0, rec, numerics):
    """One cover at the base point, accumulated into the record. A cover
    reached after the batch's --max-seconds deadline is not run and is
    recorded as undecided with that reason."""
    if past_deadline(matcher):
        rec["undecided"].append({"cover": [int(u) for u in cover], "x0": list(x0),
                                 "reason": "not run: the batch passed its --max-seconds deadline"})
        return
    try:
        hits, st = matcher.run(cover, x0, target)
    except Exception as exc:                          # noqa: BLE001
        rec["undecided"].append({"cover": [int(u) for u in cover], "x0": list(x0),
                                 "reason": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc().splitlines()[-3:]})
        return
    if st["refused"]:
        rec["refused"] += 1
        return
    rec["matched"] += 1
    k = ",".join(str(b) for b in st["coord_solutions"])
    rec["coord_solution_hist"][k] = rec["coord_solution_hist"].get(k, 0) + 1
    for h in hits:
        det, num = common.hit_record(orbit, h, cover, x0)
        rec["hits"].append(det)
        numerics.append(num)
        if not det["agree"]:
            rec["undecided"].append({"cover": det["cover"], "x0": det["x0"],
                                     "reason": "modular and numeric decisions of a hit disagree",
                                     "hit": len(rec["hits"]) - 1})


def run_batch(orbit, part, index, verbose=False, max_seconds=0.0):
    """Run batch `index` of the partition. Returns (deterministic record,
    extra fields), the extra fields not part of the deterministic hash.
    max_seconds > 0 sets the batch's wall-clock guard (module note)."""
    geo = common.batch_geometry(part, index)
    stage = geo["stage"]
    x0 = tuple(part["x0"])
    assert x0 == X0[orbit], "the partition's base point is not the cell's"
    E = CoverEnumerator3(orbit, N2)
    assert E.N == part["N"] and E.info["order"] == part["group_order"], "dictionary differs from the partition's"
    budget = Budget(seconds=max_seconds) if max_seconds else None
    matcher = Matcher(E.D, N2, E.F1, E.F2, verbose=verbose, budget=budget)
    target = psi_target(orbit, N2, E.F1, E.F2)
    rec = {"batch": {"index": index, "stage": stage},
           "orbit": orbit, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "x0": list(x0), "partition_sha256": part["sha256"], "degenerate_sha256": None,
           "covers": 0, "matched": 0, "refused": 0, "coord_solution_hist": {},
           "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"candidates": 0, "kernel_s": 0.0, "match_s": 0.0, "max_seconds": max_seconds}
    t0 = time.time()
    if stage == "A":
        units = geo["units"]
        rec["batch"]["units"] = units
        rec["batch"]["n_units"] = len(units)
        plans = {}
        for i, j, _ in units:
            tk = time.time()
            if i not in plans:
                Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
                members, _ = E.pivot_plan(i)
                mask = np.zeros(E.N, dtype=bool)
                mask[members] = True
                plans[i] = (Qi, mask)
            Qi, mask = plans[i]
            covers, nc, _ = E.pair_covers(5, i, j, Qi, mask)
            extra["kernel_s"] += time.time() - tk
            extra["candidates"] += int(nc)
            rec["covers"] += len(covers)
            tm = time.time()
            for cover in sorted(covers):
                match_cover(orbit, matcher, target, cover, x0, rec, numerics)
            extra["match_s"] += time.time() - tm
    else:
        deg_path = os.path.join(common.HERE, part["degenerate"]["file"])
        all_covers, deg = common.load_degenerate(orbit, deg_path, part["degenerate"]["sha256"])
        rec["degenerate_sha256"] = deg["sha256"]
        ids = geo["cover_ids"]
        rec["batch"]["cover_ids"] = ids
        rec["batch"]["n_covers"] = len(ids)
        tm = time.time()
        for k in ids:
            cover = all_covers[k]
            assert common.stage_of(cover) == stage, (k, cover, stage)
            rec["covers"] += 1
            match_cover(orbit, matcher, target, cover, x0, rec, numerics)
        extra["match_s"] += time.time() - tm
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["aborted_at_deadline"] = past_deadline(matcher) and bool(rec["undecided"])
    extra["wall_s"] = time.time() - t0
    extra["hit_numerics"] = numerics
    extra["matcher"] = "native" if matcher.native_cls is not None else "reference"
    return rec, extra


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", choices=common.ORBITS)
    ap.add_argument("batch", type=int, help="the batch index")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-clock guard for the batch: the running cover is aborted and the covers not yet "
                         "started are recorded as undecided (0: none)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv[1:])
    common.lower_priority()
    orbit = a.orbit
    part = common.load_partition(orbit, a.partition)
    index = a.batch
    geo = common.batch_geometry(part, index)
    out_dir = a.out_dir or common.results_dir(orbit)
    path = os.path.join(out_dir, f"batch_{index}.json")
    if os.path.exists(path) and not a.force:
        print(f"{os.path.relpath(path, common.ROOT)} exists; skipping (--force to redo)")
        return 0
    size = (f"{len(geo['units'])} pivot pairs" if geo["stage"] == "A"
            else f"{len(geo['cover_ids'])} covers")
    print(f"{orbit} batch {index} of {part['batches']}: stage {geo['stage']}, {size}, x0 {tuple(part['x0'])}",
          flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(orbit, part, index, verbose=a.verbose, max_seconds=a.max_seconds)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "candidates": extra["candidates"], "kernel_s": extra["kernel_s"], "match_s": extra["match_s"],
        "max_seconds": extra["max_seconds"], "aborted_at_deadline": extra["aborted_at_deadline"],
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system,
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "git_commit": common.git_commit(),
        "matcher": extra["matcher"],
        "kernel_version": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
        "hit_numerics": extra["hit_numerics"],
    })
    rec["sha256"] = common.sha256_json(rec)
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    print(f"{orbit} batch {index}: {rec['covers']} covers, {rec['matched']} matched, {rec['refused']} refused, "
          f"{rec['hit_count']} hits, {rec['decompositions']} decompositions, "
          f"{len(rec['undecided'])} undecided; kernel {rec['kernel_s']:.0f}s, match {rec['match_s']:.0f}s, "
          f"wall {rec['wall_s']:.0f}s; deterministic sha256 {rec['deterministic_sha256'][:16]}",
          flush=True)
    if rec["decompositions"]:
        h = next(h for h in rec["hits"] if h["decomposition"])
        print(f"DECOMPOSITION FOUND {orbit} batch {index}: |{orbit}>^4 lies in the span of {h['terms_count']} "
              f"stabilizer states (numerical rank {h['rank']}) from base cover {h['cover']} at "
              f"x0 {tuple(h['x0'])}; chi({orbit}^4) <= {h['rank']}; {rec['decompositions']} such hit(s)",
              flush=True)
        return 2
    if rec["undecided"]:
        if rec["aborted_at_deadline"]:
            not_run = sum(1 for u in rec["undecided"] if u["reason"].startswith("not run"))
            print(f"{orbit} batch {index} ABORTED at the --max-seconds {a.max_seconds:.0f}s deadline: "
                  f"{len(rec['undecided']) - not_run} cover(s) aborted while running, {not_run} not run; "
                  f"see {os.path.relpath(path, common.ROOT)}", flush=True)
        raise RuntimeError(f"{orbit} batch {index}: {len(rec['undecided'])} undecided cover(s); see {path}")
    print(f"{orbit} batch {index} complete: no hit is a decomposition of |{orbit}>^4; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
