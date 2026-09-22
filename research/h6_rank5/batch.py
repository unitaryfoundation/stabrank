"""One batch of the rank-5 exclusion of |H>^6 by an all-visible base slice.

    batch.py K [--partition P] [--out-dir D] [--force] [--native | --no-native]
             [--verbose]

Batch K of the partition is either a stage A batch (a list of pivot pairs
(i, j) of the 5-cover enumeration of |H>^3: the compiled kernel lists the
full 5-covers of the pair and the matcher runs each at the four base
points), a stage B batch (a list of indices into degenerate_covers.json:
full 5-covers of five distinct but dependent states, matched through the
coefficient family) or a stage C batch (covers with a repeated state,
matched through the block treatment). See docs/notes/h6_rank5_exclusion.md
sections 2 to 4 and research/h6_rank5/README.md.

Every hit the matcher returns (a candidate decomposition of |H>^6 whose
slices all satisfy the modular equations) is re-decided here exactly: the
five terms are rebuilt from their phase codes and psi_6 is tested against
their span mod 2013265921 and in floating point (common.decide_terms). A hit
with psi_6 in the span is a decomposition with at most five terms, and the
batch exits 2 with DECOMPOSITION FOUND on its last stdout line; a hit on
which the two tests disagree is recorded under `undecided`, which fails the
batch (exit 1). Exit 0 when every cover was matched and no hit is a
decomposition.

Output: <out-dir>/batch_<K>.json, skipped when it exists (resume by
rerunning; --force overwrites). The record's deterministic part (geometry,
partition hash, counts, the coordinate-slice solution histogram, the hits
as phase codes with their decisions, `undecided`) is hashed as
`deterministic_sha256`, so that a re-run on any machine, with the compiled
kernels or with --no-native, can be compared bit for bit; timing, host,
matcher and version fields follow, then `sha256` over the whole record.

--no-native (or STABRANK_NO_NATIVE=1 in the environment) keeps the 5-cover
enumeration and the stage A matcher in the Python reference of
verify_challenge/slice_cover.py; a stage A batch is then about 80 times
slower.
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
from common import M, N1, ORBIT, RANK  # noqa: E402
from slice_cover import CoverEnumerator, SliceMatcher, _reduce, x0_reps  # noqa: E402


def match_cover(matcher, cover, rec, numerics):
    """One cover at the four base points, accumulated into the record."""
    for x0 in x0_reps(N1):
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
        k = ",".join(str(b) for b in st["coord_solutions"])
        rec["coord_solution_hist"][k] = rec["coord_solution_hist"].get(k, 0) + 1
        for h in hits:
            det, num = common.hit_record(h, cover, x0)
            rec["hits"].append(det)
            numerics.append(num)
            if not det["agree"]:
                rec["undecided"].append({"cover": det["cover"], "x0": det["x0"],
                                         "reason": "modular and numeric decisions of a hit disagree",
                                         "hit": len(rec["hits"]) - 1})


def run_batch(part, index, native, verbose=False):
    """Run batch `index` of the partition. Returns (deterministic record,
    extra fields) with the extra fields not part of the deterministic hash."""
    geo = common.batch_geometry(part, index)
    stage = geo["stage"]
    E = CoverEnumerator(N1, native=native)
    assert E.N == part["N"] and E.info["order"] == part["group_order"], "dictionary differs from the partition's"
    matcher = SliceMatcher(E, N1, native=native, verbose=verbose)
    rec = {"batch": {"index": index, "stage": stage},
           "orbit": ORBIT, "m": M, "rank": RANK, "n1": N1, "N": E.N, "group_order": E.info["order"],
           "partition_sha256": part["sha256"], "degenerate_sha256": None,
           "covers": 0, "matched": 0, "refused": 0, "native_runs": 0, "coord_solution_hist": {},
           "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"candidates": 0, "kernel_s": 0.0, "match_s": 0.0}
    t0 = time.time()
    if stage == "A":
        units = geo["units"]
        rec["batch"]["units"] = units
        rec["batch"]["n_units"] = len(units)
        for i, j, _ in units:
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
                match_cover(matcher, cover, rec, numerics)
            extra["match_s"] += time.time() - tm
    else:
        deg_path = os.path.join(common.HERE, part["degenerate"]["file"])
        all_covers, deg = common.load_degenerate(deg_path, part["degenerate"]["sha256"])
        rec["degenerate_sha256"] = deg["sha256"]
        ids = geo["cover_ids"]
        rec["batch"]["cover_ids"] = ids
        rec["batch"]["n_covers"] = len(ids)
        tm = time.time()
        for k in ids:
            cover = all_covers[k]
            assert common.stage_of(cover) == stage, (k, cover, stage)
            rec["covers"] += 1
            match_cover(matcher, cover, rec, numerics)
        extra["match_s"] += time.time() - tm
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["wall_s"] = time.time() - t0
    extra["hit_numerics"] = numerics
    extra["matcher"] = "native" if matcher.native is not None else "reference"
    extra["cover5"] = "native" if E.native_cover5 is not None else "reference"
    return rec, extra


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch", type=int, help="the batch index")
    ap.add_argument("--partition", default=common.PARTITION)
    ap.add_argument("--out-dir", default=common.RESULTS)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--native", dest="native", action="store_true", default=None,
                   help="use the compiled kernels (default unless STABRANK_NO_NATIVE is set)")
    g.add_argument("--no-native", dest="native", action="store_false",
                   help="Python reference for the 5-cover kernel and the stage A matcher")
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
    if os.path.exists(path) and not a.force:
        print(f"{os.path.relpath(path, common.ROOT)} exists; skipping (--force to redo)")
        return 0
    size = (f"{len(geo['units'])} pivot pairs" if geo["stage"] == "A"
            else f"{len(geo['cover_ids'])} covers")
    print(f"batch {index} of {part['batches']}: stage {geo['stage']}, {size}, "
          f"{'compiled kernels' if native else 'Python reference'}", flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(part, index, native, verbose=a.verbose)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "candidates": extra["candidates"], "kernel_s": extra["kernel_s"], "match_s": extra["match_s"],
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system,
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
        print(f"DECOMPOSITION FOUND batch {index}: psi_6 lies in the span of {h['terms_count']} "
              f"stabilizer states (numerical rank {h['rank']}) from base cover {h['cover']} at "
              f"x0 {h['x0']}; chi(qubit_H^6) <= {h['rank']}; {rec['decompositions']} such hit(s)",
              flush=True)
        return 2
    if rec["undecided"]:
        raise RuntimeError(f"batch {index}: {len(rec['undecided'])} undecided (cover, x0) run(s); "
                           f"see {path}")
    print(f"batch {index} complete: no hit is a decomposition of |H>^6; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
