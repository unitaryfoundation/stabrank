"""One batch of the rank-4 exclusion of |T>^5 by a two-qubit base slice.

    batch.py K [--partition P] [--out-dir D] [--force | --resume]
             [--native | --no-native] [--max-seconds S] [--verbose]

Batch K of the partition is a list of indices into the census covers4.json
(every full 4-cover of |T>^3 up to the unitary symmetry of psi_3: kind A,
distinct independent states; kind B, distinct dependent states; kind C, a
repeated state). Each cover is matched at the base points 00 and 01 by
verify_challenge/slice_cover.SliceMatcher with orbit qubit_T (kind A
through the compiled kernel unless --no-native, kinds B and C through the
coefficient family and the block treatment of the Python reference). See
docs/notes/t5_rank4_exclusion.md.

Every hit the matcher returns is re-decided here exactly from the phase
codes of its terms (psi_5 = |T>^5 against their span mod 2013265921 in the
Q(zeta_24) field and in floating point, common.decide_terms). A hit with
psi_5 in the span is a decomposition with at most four terms, and the batch
exits 2 with DECOMPOSITION FOUND on its last stdout line. A (cover, x0) run
that raises (UnpinnedFamily, the candidate cap, anything else), a hit on
which the two decisions disagree, and every cover not run before
--max-seconds elapsed are recorded under `undecided`, which fails the batch
(exit 1) and the aggregate. Exit 0 when every cover was matched and no hit
is a decomposition.

Output: <out-dir>/batch_<K>.json. An existing complete record is skipped
(resume by rerunning); with --resume a record that has undecided runs from
a deadline is redone from scratch; --force redoes any record. The record's
deterministic part (geometry, partition and census hashes, counts, the
solution histogram, the hits as phase codes with their decisions,
`undecided`) is hashed as `deterministic_sha256`, so that a re-run on any
machine, with the compiled kernel or with --no-native, is compared bit for
bit; the count of runs through the compiled kernel, timing, host, matcher
and version fields follow outside it, then `sha256` over the whole record.
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
import common  # noqa: E402  (research/t5_rank4/common.py)
from common import M, N1, ORBIT, RANK, X0S  # noqa: E402
from slice_cover import SliceMatcher  # noqa: E402


def hist_key(st):
    return ",".join(str(b) for b in st["coord_solutions"])


def match_cover(matcher, cover, rec, numerics, deadline=None, extra=None):
    """One cover at both base points, accumulated into the record."""
    if extra is None:
        extra = {"native_runs": 0}
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
        extra["native_runs"] += bool(st.get("native"))       # a path count, outside the deterministic part
        k = hist_key(st)
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
    E = common.make_enumerator(native=native)
    assert E.N == part["N"] and E.info["order"] == part["group_order"], "dictionary differs from the partition's"
    matcher = SliceMatcher(E, N1, native=native, verbose=verbose)
    path = os.path.join(common.HERE, part["census"]["file"])
    covers, doc = common.load_census(path, part["census"]["sha256"])
    rec = {"batch": {"index": index, "stage": stage, "cover_ids": geo["cover_ids"], "n_covers": len(geo["cover_ids"])},
           "orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "x0": list(X0S), "N": E.N, "group_order": E.info["order"],
           "partition_sha256": part["sha256"], "census_sha256": doc["sha256"],
           "covers": 0, "matched": 0, "refused": 0, "coord_solution_hist": {},
           "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"deadline_hit": False, "native_runs": 0}
    t0 = time.time()
    deadline = None if max_seconds is None else t0 + max_seconds
    for k in geo["cover_ids"]:
        cover = covers[k]
        assert doc["kinds"][k] == stage, (k, cover, stage)
        rec["covers"] += 1
        match_cover(matcher, cover, rec, numerics, deadline, extra)
    extra["match_s"] = time.time() - t0
    extra["deadline_hit"] = any(u.get("reason", "").startswith("deadline") for u in rec["undecided"])
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["wall_s"] = time.time() - t0
    extra["hit_numerics"] = numerics
    extra["matcher"] = "native" if matcher.native is not None else "reference"
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
                    help="use the compiled kernel (default unless STABRANK_NO_NATIVE is set)")
    g2.add_argument("--no-native", dest="native", action="store_false",
                    help="Python reference for the kind A matcher")
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
    print(f"batch {index} of {part['batches']}: kind {geo['stage']}, {len(geo['cover_ids'])} covers, "
          f"{'compiled kernel' if native else 'Python reference'}"
          + (f", deadline {a.max_seconds:.0f}s" if a.max_seconds else ""), flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(part, index, native, a.max_seconds, verbose=a.verbose)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "native_runs": extra["native_runs"],
        "match_s": extra["match_s"], "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system,
        "max_seconds": a.max_seconds, "deadline_hit": extra["deadline_hit"],
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "git_commit": common.git_commit(),
        "native": native, "matcher": extra["matcher"],
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
          f"{len(rec['undecided'])} undecided; match {rec['match_s']:.0f}s, wall {rec['wall_s']:.0f}s; "
          f"deterministic sha256 {rec['deterministic_sha256'][:16]}", flush=True)
    if rec["decompositions"]:
        h = next(h for h in rec["hits"] if h["decomposition"])
        print(f"DECOMPOSITION FOUND batch {index}: psi_5 lies in the span of {h['terms_count']} "
              f"stabilizer states (numerical rank {h['rank']}) from base cover {h['cover']} at "
              f"x0 {h['x0']}; chi(qubit_T^5) <= {h['rank']}; {rec['decompositions']} such hit(s)",
              flush=True)
        return 2
    if rec["undecided"]:
        n_dead = sum(str(u.get("reason", "")).startswith("deadline") for u in rec["undecided"])
        print(f"batch {index}: {len(rec['undecided'])} undecided run(s), {n_dead} of them not run before the "
              f"deadline; see {os.path.relpath(path, common.ROOT)}"
              + (" (rerun with --resume)" if n_dead else ""), flush=True)
        return 1
    print(f"batch {index} complete: no hit is a decomposition of |T>^5; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
