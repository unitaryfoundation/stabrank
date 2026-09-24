"""One batch of the rank-5 census of |T5>^2.

    batch.py K [--partition P] [--out-dir D] [--force | --resume]
             [--native | --no-native] [--max-seconds S]

Batch K of the partition is a contiguous range of (pivot, partner) units
of the 5-cover enumeration over the 3,900 two-ququint stabilizer states.
For every unit the kernel (cpp/src/cover5.cpp through
stabrank_core.cover5_pair, or its Python reference with --no-native) lists
every 5-set through the pivot and partner, with the other three members
above the partner inside the pivot's member mask, whose span contains
psi_2 = |T5>^2 modulo 2013265921. Every such set is a hit: it is
re-decided here exactly (the rank test modulo 2013265921 in the Q(zeta_5)
field, independently of the kernel) and numerically (least squares over
C), and recorded with the kernel's fullness flags and both decisions. A
hit on which both decisions agree that psi_2 lies in the span is a
decomposition of |T5>^2 with at most five terms, and the batch exits 2
with DECOMPOSITION FOUND on its last stdout line. A hit on which the two
decisions disagree, a unit on which the kernel raises (a parallel class
above max_run), and every unit not run before --max-seconds elapsed are
recorded under `undecided`, which fails the batch (exit 1) and the
aggregate. Exit 0 when every unit ran and no hit exists.

Output: <out-dir>/batch_<K>.json. An existing complete record is skipped;
with --resume a record that a deadline left incomplete is redone from
scratch; --force redoes any record. The record's deterministic part (the
geometry, the units, the dictionary, plan and partition hashes, the
kernel's member count per unit, the hits with their flags and decisions,
`undecided`) is hashed as `deterministic_sha256`, so that a re-run on any
machine, through the compiled kernel or the Python reference, is compared
bit for bit; the modular candidate count and the count of units run
through the compiled kernel (path counts, which depend on the kernel's
random functional), timing, host and version fields follow outside it,
then `sha256` over the whole record.
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
import common  # noqa: E402  (research/t5q_m2_rank5/common.py)
from common import M, ORBIT, RANK  # noqa: E402


def run_batch(part, index, native, max_seconds=None):
    """Run batch `index` of the partition. Returns (deterministic record,
    extra fields), the extra fields outside the deterministic hash."""
    geo = common.batch_geometry(part, index)
    E = common.make_enumerator(native=native)
    if E.N != part["N"] or E.info["order"] != part["group_order"]:
        raise AssertionError("dictionary size or symmetry order differs from the partition's")
    dsha = E.dictionary_sha256()
    if dsha != part["dictionary_sha256"]:
        raise AssertionError("the dictionary differs from the partition's (hash of the phase codes)")
    units = [list(u) for u in part["units"][geo["start"]:geo["end"]]]
    rec = {"batch": {"index": index, "start": geo["start"], "end": geo["end"], "n_units": len(units),
                     "units": units},
           "orbit": ORBIT, "m": M, "rank": RANK, "N": E.N, "group_order": E.info["order"],
           "dictionary_sha256": dsha, "plan_sha256": part["plan_sha256"], "partition_sha256": part["sha256"],
           "units_run": 0, "members": [], "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"candidates": 0, "native_runs": 0, "kernel_s": 0.0, "deadline_hit": False}
    t0 = time.time()
    deadline = None if max_seconds is None else t0 + max_seconds
    masks = {}
    use_native = native and E.native_cover5 is not None
    for i, j, Mp in units:
        if deadline is not None and time.time() > deadline:
            rec["undecided"].append({"unit": [int(i), int(j)], "reason": "deadline: pivot pair not run"})
            rec["members"].append(None)
            extra["deadline_hit"] = True
            continue
        if i not in masks:
            masks[i] = E.member_mask(i)
        tk = time.time()
        try:
            sets, nc, Mk = E.pair_sets(i, j, masks[i], native=use_native)
        except Exception as exc:                      # noqa: BLE001
            rec["undecided"].append({"unit": [int(i), int(j)], "reason": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc().splitlines()[-3:]})
            rec["members"].append(None)
            extra["kernel_s"] += time.time() - tk
            continue
        extra["kernel_s"] += time.time() - tk
        extra["candidates"] += int(nc)
        extra["native_runs"] += int(use_native)
        rec["units_run"] += 1
        rec["members"].append(int(Mk))
        for idx, f1, f2 in sorted(sets):
            det, num = common.hit_record(E, idx, f1, f2)
            det["unit"] = [int(i), int(j)]
            rec["hits"].append(det)
            numerics.append(num)
            if not det["agree"]:
                rec["undecided"].append({"unit": [int(i), int(j)], "hit": len(rec["hits"]) - 1,
                                         "reason": "modular and numeric decisions of a hit disagree"})
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["wall_s"] = time.time() - t0
    extra["hit_numerics"] = numerics
    extra["cover5"] = "native" if use_native else "reference"
    return rec, extra


def record_state(path):
    """'missing', 'deadline' (a stored record with deadline-undecided units
    or an unreadable file), or 'complete'."""
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
                    help="the Python reference of the 5-cover kernel")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop after S seconds; units not run are recorded as undecided")
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
        print(f"{os.path.relpath(path, common.ROOT)} exists with deadline-undecided units; --resume or --force "
              "to redo it", flush=True)
        return 1
    print(f"batch {index} of {part['batches']}: units {geo['start']}..{geo['end'] - 1} ({geo['n_units']} pivot "
          f"pairs, about {geo['est_pod_s']:.0f} pod s), {'compiled kernel' if native else 'Python reference'}"
          + (f", deadline {a.max_seconds:.0f}s" if a.max_seconds else ""), flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(part, index, native, a.max_seconds)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "native_runs": extra["native_runs"], "candidates": extra["candidates"], "kernel_s": extra["kernel_s"],
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system, "max_seconds": a.max_seconds,
        "deadline_hit": extra["deadline_hit"],
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "git_commit": common.git_commit(),
        "native": native, "cover5": extra["cover5"],
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
    print(f"batch {index}: {rec['units_run']} of {rec['batch']['n_units']} units run, {rec['candidates']} modular "
          f"candidates, {rec['hit_count']} hits, {rec['decompositions']} decompositions, "
          f"{len(rec['undecided'])} undecided; kernel {rec['kernel_s']:.0f}s, wall {rec['wall_s']:.0f}s; "
          f"deterministic sha256 {rec['deterministic_sha256'][:16]}", flush=True)
    if rec["decompositions"]:
        h = next(h for h in rec["hits"] if h["decomposition"])
        print(f"DECOMPOSITION FOUND batch {index}: psi_2 = |T5>^2 lies in the span of the {h['terms_count']} "
              f"stabilizer states {h['states']} (numerical rank {h['rank']}, coefficients "
              f"{'all nonzero' if h['nonzero'] else 'with a zero'}); chi(T5^2) <= {h['rank']}; "
              f"{rec['decompositions']} such hit(s)", flush=True)
        return 2
    if rec["undecided"]:
        n_dead = sum(str(u.get("reason", "")).startswith("deadline") for u in rec["undecided"])
        print(f"batch {index}: {len(rec['undecided'])} undecided unit(s), {n_dead} of them not run before the "
              f"deadline; see {os.path.relpath(path, common.ROOT)}"
              + (" (rerun with --resume)" if n_dead else ""), flush=True)
        return 1
    print(f"batch {index} complete: no 5-set of its units has |T5>^2 in its span; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
