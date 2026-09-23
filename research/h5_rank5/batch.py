"""One batch of the rank-5 exclusion of |H>^5 by a two-qubit base slice.

    batch.py K [--partition P] [--out-dir D] [--force | --resume]
             [--native | --no-native] [--max-seconds S] [--verbose]

Batch K of the partition is a stage A batch (pivot pairs (i, j) of the
5-cover enumeration of |H>^3: the compiled kernel lists the full 5-covers
of the pair and the stage (alpha) matcher runs each at the base points 00
and 01), a stage B batch (indices into the H^6 degenerate list: dependent
5-covers of five distinct states, through the coefficient family), a stage
C batch (covers with a repeated state, through the block treatment), or a
stage beta batch (indices into beta_covers.json: full 4-covers of |H>^3,
through beta.BetaMatcher). See docs/notes/h5_rank5_exclusion.md.

Every hit the matcher returns is re-decided here exactly from the phase
codes of its terms (psi_5 against their span mod 2013265921 and in floating
point, common.decide_terms). A hit with psi_5 in the span is a decomposition
with at most five terms, and the batch exits 2 with DECOMPOSITION FOUND on
its last stdout line. A (cover, x0) run that raises (UnpinnedFamily, the
candidate cap, anything else), a hit on which the two decisions disagree,
and every cover not run before --max-seconds elapsed are recorded under
`undecided`, which fails the batch (exit 1) and the aggregate. Exit 0 when
every cover was matched and no hit is a decomposition.

Output: <out-dir>/batch_<K>.json. An existing complete record is skipped
(resume by rerunning); with --resume a record that has undecided runs from
a deadline is redone from scratch; --force redoes any record. The record's
deterministic part (geometry, partition and list hashes, counts, the
solution histogram, the hits as phase codes with their decisions,
`undecided`) is hashed as `deterministic_sha256`, so that a re-run on any
machine, with the compiled kernels or with --no-native, is compared bit for
bit; the modular candidate count, timing, host, matcher and version fields
follow outside it, then `sha256` over the whole record.

--no-native (or STABRANK_NO_NATIVE=1) keeps the 5-cover enumeration and the
stage A matcher in the Python reference of verify_challenge/slice_cover.py.
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
import common  # noqa: E402  (research/h5_rank5/common.py)
from common import M, N1, N2, ORBIT, RANK, X0S  # noqa: E402
from beta import BetaMatcher  # noqa: E402
from slice_cover import CoverEnumerator, SliceMatcher, _reduce  # noqa: E402


class Deadline(Exception):
    pass


def hist_key(stage, st):
    if stage == "beta":
        return f"{st['b_solutions']},{st['a1_solutions']},{st['a2_solutions'] + st['a2_point']}"
    return ",".join(str(b) for b in st["coord_solutions"])


def match_cover(matcher, cover, rec, numerics, stage="A", deadline=None):
    """One cover at both base points, accumulated into the record."""
    for x0 in X0S:
        if deadline is not None and time.time() > deadline:
            rec["undecided"].append({"cover": [int(u) for u in cover], "x0": int(x0),
                                     "reason": "deadline: not run"})
            continue
        try:
            hits, st = matcher.run(cover, x0)
        except Exception as exc:                      # noqa: BLE001
            rec["undecided"].append({"cover": [int(u) for u in cover], "x0": int(x0),
                                     "reason": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc().splitlines()[-3:]})
            continue
        if st["refused"]:
            rec["refused"] += 1
            continue
        rec["matched"] += 1
        rec["native_runs"] += bool(st.get("native"))
        k = hist_key(stage, st)
        rec["coord_solution_hist"][k] = rec["coord_solution_hist"].get(k, 0) + 1
        for h in hits:
            det, num = common.hit_record(h, cover, x0)
            rec["hits"].append(det)
            numerics.append(num)
            if not det["agree"]:
                rec["undecided"].append({"cover": det["cover"], "x0": det["x0"],
                                         "reason": "modular and numeric decisions of a hit disagree",
                                         "hit": len(rec["hits"]) - 1})


def run_batch(part, index, native, max_seconds=None, verbose=False):
    """Run batch `index` of the partition. Returns (deterministic record,
    extra fields) with the extra fields not part of the deterministic hash."""
    geo = common.batch_geometry(part, index)
    stage = geo["stage"]
    E = CoverEnumerator(N2, native=native)
    assert E.N == part["N"] and E.info["order"] == part["group_order"], "dictionary differs from the partition's"
    matcher = BetaMatcher(E, verbose=verbose) if stage == "beta" else SliceMatcher(E, N1, native=native, verbose=verbose)
    rec = {"batch": {"index": index, "stage": stage},
           "orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "x0": list(X0S), "N": E.N, "group_order": E.info["order"],
           "partition_sha256": part["sha256"], "degenerate_sha256": None, "beta_sha256": None,
           "covers": 0, "matched": 0, "refused": 0, "native_runs": 0, "coord_solution_hist": {},
           "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"candidates": 0, "kernel_s": 0.0, "match_s": 0.0, "deadline_hit": False}
    t0 = time.time()
    deadline = None if max_seconds is None else t0 + max_seconds
    if stage == "A":
        units = geo["units"]
        rec["batch"]["units"] = units
        rec["batch"]["n_units"] = len(units)
        for i, j, _ in units:
            if deadline is not None and time.time() > deadline:
                rec["undecided"].append({"unit": [int(i), int(j)], "reason": "deadline: pivot pair not run"})
                extra["deadline_hit"] = True
                continue
            tk = time.time()
            Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
            members, _ = E.pivot_plan(i)
            mask = np.zeros(E.N, dtype=bool)
            mask[members] = True
            covers, nc = E.pair_covers(5, i, j, Qi, mask)
            extra["kernel_s"] += time.time() - tk
            extra["candidates"] += int(nc)
            rec["covers"] += len(covers)
            tm = time.time()
            for cover in sorted(covers):
                match_cover(matcher, cover, rec, numerics, stage, deadline)
            extra["match_s"] += time.time() - tm
    else:
        if stage == "beta":
            path = os.path.join(common.HERE, part["beta"]["file"])
            all_covers, doc = common.load_beta(path, part["beta"]["sha256"])
            rec["beta_sha256"] = doc["sha256"]
        else:
            path = os.path.join(common.HERE, part["degenerate"]["file"])
            all_covers, doc = common.load_degenerate(path, part["degenerate"]["sha256"])
            rec["degenerate_sha256"] = doc["sha256"]
        ids = geo["cover_ids"]
        rec["batch"]["cover_ids"] = ids
        rec["batch"]["n_covers"] = len(ids)
        tm = time.time()
        for k in ids:
            cover = all_covers[k]
            assert common.stage_of(cover) == stage, (k, cover, stage)
            rec["covers"] += 1
            match_cover(matcher, cover, rec, numerics, stage, deadline)
        extra["match_s"] += time.time() - tm
    extra["deadline_hit"] |= any(u.get("reason", "").startswith("deadline") for u in rec["undecided"])
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["wall_s"] = time.time() - t0
    extra["hit_numerics"] = numerics
    extra["matcher"] = ("beta" if stage == "beta" else
                        "native" if getattr(matcher, "native", None) is not None else "reference")
    extra["cover5"] = "native" if E.native_cover5 is not None else "reference"
    return rec, extra


def record_state(path):
    """'missing', 'deadline' (a stored record with deadline undecided runs),
    or 'complete'."""
    if not os.path.exists(path):
        return "missing"
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return "deadline"
    if any(str(u.get("reason", "")).startswith("deadline") for u in rec.get("undecided", [])):
        return "deadline"
    return "complete"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch", type=int, help="the batch index")
    ap.add_argument("--partition", default=common.PARTITION)
    ap.add_argument("--out-dir", default=common.RESULTS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--force", action="store_true", help="overwrite an existing result")
    g.add_argument("--resume", action="store_true",
                   help="redo a stored record that a deadline left incomplete; skip complete ones")
    g2 = ap.add_mutually_exclusive_group()
    g2.add_argument("--native", dest="native", action="store_true", default=None,
                    help="use the compiled kernels (default unless STABRANK_NO_NATIVE is set)")
    g2.add_argument("--no-native", dest="native", action="store_false",
                    help="Python reference for the 5-cover kernel and the stage A matcher")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop matching after S seconds; covers not run are recorded as undecided")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv[1:])
    common.lower_priority()
    native = a.native
    if native is None:
        native = not os.environ.get("STABRANK_NO_NATIVE")
    if not native:
        os.environ["STABRANK_NO_NATIVE"] = "1"
    part = common.load_partition(a.partition)
    index = a.batch
    geo = common.batch_geometry(part, index)
    path = os.path.join(a.out_dir, f"batch_{index}.json")
    state = record_state(path)
    if state == "complete" and not a.force:
        print(f"{os.path.relpath(path, common.ROOT)} exists and is complete; skipping (--force to redo)")
        return 0
    if state == "deadline" and not (a.force or a.resume):
        print(f"{os.path.relpath(path, common.ROOT)} exists with deadline-undecided runs; --resume or --force "
              "to redo it", flush=True)
        return 1
    size = (f"{len(geo['units'])} pivot pairs" if geo["stage"] == "A" else f"{len(geo['cover_ids'])} covers")
    print(f"batch {index} of {part['batches']}: stage {geo['stage']}, {size}, "
          f"{'compiled kernels' if native else 'Python reference'}"
          + (f", deadline {a.max_seconds:.0f}s" if a.max_seconds else ""), flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(part, index, native, a.max_seconds, verbose=a.verbose)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "candidates": extra["candidates"], "kernel_s": extra["kernel_s"], "match_s": extra["match_s"],
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system, "max_seconds": a.max_seconds,
        "deadline_hit": extra["deadline_hit"],
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "git_commit": common.git_commit(),
        "native": native, "matcher": extra["matcher"], "cover5": extra["cover5"],
        "kernel_version": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
        "hit_numerics": extra["hit_numerics"],
    })
    rec["sha256"] = common.sha256_json(rec)
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    print(f"batch {index}: {rec['covers']} covers, {rec['matched']} matched, {rec['refused']} refused, "
          f"{rec['hit_count']} hits, {rec['decompositions']} decompositions, "
          f"{len(rec['undecided'])} undecided; kernel {rec['kernel_s']:.0f}s, match {rec['match_s']:.0f}s, "
          f"wall {rec['wall_s']:.0f}s; deterministic sha256 {rec['deterministic_sha256'][:16]}",
          flush=True)
    if rec["decompositions"]:
        h = next(h for h in rec["hits"] if h["decomposition"])
        print(f"DECOMPOSITION FOUND batch {index}: psi_5 lies in the span of {h['terms_count']} "
              f"stabilizer states (numerical rank {h['rank']}) from base cover {h['cover']} at "
              f"x0 {h['x0']}; chi(qubit_H^5) <= {h['rank']}; {rec['decompositions']} such hit(s)",
              flush=True)
        return 2
    if rec["undecided"]:
        n_dead = sum(str(u.get("reason", "")).startswith("deadline") for u in rec["undecided"])
        print(f"batch {index}: {len(rec['undecided'])} undecided run(s), {n_dead} of them not run before the "
              f"deadline; see {os.path.relpath(path, common.ROOT)}"
              + (" (rerun with --resume)" if n_dead else ""), flush=True)
        return 1
    print(f"batch {index} complete: no hit is a decomposition of |H>^5; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
