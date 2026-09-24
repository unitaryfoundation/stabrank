"""One batch of the rank-6 exclusion of |N>^4 by a two-qutrit base slice at
the base point (2, 2).

    batch.py K [--partition P] [--out-dir D] [--force | --resume]
             [--native | --no-native] [--max-seconds S] [--verbose]

Batch K of the partition is a stage A6 batch (pivot pairs (i, j) of the
6-cover enumeration of |N>^2: the compiled kernel cover6_pair lists the
full 6-covers of the pair and each runs through the compiled stage A
matcher at (2, 2)), a stage B6 batch (indices into the B6 list of
reps_N.json: G_2 orbit representatives of dependent 6-sets, through the
fresh-term filter for kappa 1 and the reference matcher at the batch's
candidate cap otherwise), a stage C6 batch (representatives of the
repeated 6-multisets, through the block paths), a stage beta batch
(representatives of the full 5-multisets, each run with the invisible term
on every one of the 16 flats missing (2, 2), through
invisible3.InvisibleMatcher3.run_one), or a stage gamma batch
(representatives of the full 4-multisets, each run with every one of the
136 flat multisets, through run_two). See docs/notes/n4_rank6_exclusion.md.

Every hit the matcher returns is re-decided here exactly from the phase
codes of its terms (psi_4 = |N>^4 against their span mod 2013265921 and in
floating point, common.decide_terms). A hit with psi_4 in the span is a
decomposition with at most six terms, and the batch exits 2 with
DECOMPOSITION FOUND on its last stdout line. A run that raises
(UnpinnedFamily, the candidate cap, anything else), a hit on which the two
decisions disagree, and every run not made before --max-seconds elapsed
are recorded under `undecided`, which fails the batch (exit 1) and the
aggregate. Exit 0 when every run was made and no hit is a decomposition.

Output: <out-dir>/batch_<K>.json. An existing complete record is skipped
(resume by rerunning); with --resume a record that has undecided runs from
a deadline is redone from scratch; --force redoes any record. The record's
deterministic part (geometry, partition and list hashes, counts, the
solution histogram, the hits as phase codes with their decisions,
`undecided`) is hashed as `deterministic_sha256`, so that a re-run on any
machine, with the compiled kernels or with --no-native, is compared bit for
bit; the counts of runs through the compiled kernels and through the
filter, the modular candidate and raw feature-zero counts, the filter
fallbacks, timing, host, matcher and version fields follow outside it,
then `sha256` over the whole record.

--no-native (or STABRANK_NO_NATIVE=1) keeps the 6-cover enumeration, the
stage A matcher and the dense solve in the Python references; stage B6
then runs every base through the reference matcher.
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
import common  # noqa: E402  (research/n4_rank6/common.py)
from common import FLAT_NAMES, FLAT_PAIRS, M, N1, RANK, RUNS_PER_ITEM, X0  # noqa: E402
from invisible3 import InvisibleMatcher3, hist_key  # noqa: E402
from matcher import Budget  # noqa: E402
from stages import a6_kernel, coord_key, run_a6, run_b6, run_c6, set_max_cand  # noqa: E402


def run_units(stage):
    if stage == "beta":
        return [(f, f) for f in FLAT_NAMES]
    if stage == "gamma":
        return [(list(p), p) for p in FLAT_PAIRS]
    return [(None, None)]


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


class Runner:
    def __init__(self, part, geo, native, max_seconds, verbose):
        self.stage = geo["stage"]
        self.E = common.make_enumerator(native=native)
        assert self.E.N == part["N"] and self.E.info["order"] == part["group_order"], \
            "dictionary differs from the partition's"
        self.M = common.new_matcher(self.E, native=native)
        self.M.verbose = verbose
        self.target = common.target_of(self.E)
        self.t0 = time.time()
        self.deadline = None if max_seconds is None else self.t0 + max_seconds
        self.kernel = a6_kernel(self.M, self.target) if self.stage == "A6" else None
        self.Fl = None
        if self.stage == "B6" and native and not os.environ.get("STABRANK_NO_NATIVE"):
            try:
                from filters6 import Filters
                self.Fl = Filters(self.M)
            except RuntimeError:
                self.Fl = None
        self.IM = InvisibleMatcher3(self.M, verbose=verbose) if self.stage in ("beta", "gamma") else None
        if self.IM is not None:
            self.IM.deadline = self.deadline
        if self.deadline is not None:
            self.M.budget = Budget(seconds=max_seconds)
        set_max_cand(geo.get("max_cand", 2_000_000))
        if self.IM is not None:
            self.IM.max_cand = geo.get("max_cand", 2_000_000)

    def past_deadline(self):
        return self.deadline is not None and time.time() > self.deadline

    def one(self, item, unit, kappa=None):
        """(hits, stats, extra) of one run."""
        extra = {}
        if self.stage == "A6":
            hits, st = run_a6(self.M, self.kernel, self.target, item)
        elif self.stage == "B6":
            hits, st, extra = run_b6(self.M, self.Fl, self.target, item, kappa)
        elif self.stage == "C6":
            hits, st = run_c6(self.M, self.target, item)
        elif self.stage == "beta":
            hits, st = self.IM.run_one(item, unit, self.target)
        else:
            hits, st = self.IM.run_two(item, unit, self.target)
        return hits, st, extra


def match_item(R, item, rec, numerics, extra, kappa=None):
    """One item through every one of its runs, accumulated into the record."""
    stage = R.stage
    for label, unit in run_units(stage):
        if R.past_deadline():
            rec["undecided"].append({"cover": [int(u) for u in item], "unit": label, "reason": "deadline: not run"})
            extra["deadline_hit"] = True
            continue
        try:
            hits, st, ex = R.one(item, unit, kappa)
        except Exception as exc:                      # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            if "deadline" in msg:
                msg = "deadline: aborted while running (" + msg[:160] + ")"
                extra["deadline_hit"] = True
            rec["undecided"].append({"cover": [int(u) for u in item], "unit": label, "reason": msg[:400],
                                     "traceback": traceback.format_exc().splitlines()[-3:]})
            continue
        if st["refused"]:
            rec["refused"] += 1
            continue
        rec["matched"] += 1
        extra["native_runs"] += bool(st.get("native"))
        extra["candidates"] += int(st.get("candidates", 0))
        extra["dense_raw"] += int(st.get("dense_raw", 0)) + int(ex.get("dense_raw", 0))
        if ex.get("path"):
            extra["b6_paths"][ex["path"]] = extra["b6_paths"].get(ex["path"], 0) + 1
        if ex.get("fallback"):
            extra["filter_fallbacks"].append({"cover": [int(u) for u in item], "reason": ex["fallback"]})
        k = hist_key(stage, st) if stage in ("beta", "gamma") else coord_key(st)
        rec["coord_solution_hist"][k] = rec["coord_solution_hist"].get(k, 0) + 1
        flats = None if stage in ("A6", "B6", "C6") else ([unit] if stage == "beta" else list(unit))
        for h in hits:
            det, num = common.hit_record(h, item, X0, flats)
            rec["hits"].append(det)
            numerics.append(num)
            if not det["agree"]:
                rec["undecided"].append({"cover": det["cover"], "unit": label,
                                         "reason": "modular and numeric decisions of a hit disagree",
                                         "hit": len(rec["hits"]) - 1})


def run_batch(part, index, native, max_seconds=None, verbose=False):
    """Run batch `index` of the partition. Returns (deterministic record,
    extra fields) with the extra fields not part of the deterministic hash."""
    geo = common.batch_geometry(part, index)
    stage = geo["stage"]
    R = Runner(part, geo, native, max_seconds, verbose)
    E = R.E
    rec = {"batch": {"index": index, "stage": stage},
           "orbit": common.ORBIT, "m": M, "rank": RANK, "n1": N1, "x0": list(X0), "N": E.N,
           "group_order": E.info["order"], "runs_per_item": RUNS_PER_ITEM[stage],
           "partition_sha256": part["sha256"], "census_sha256": None, "reps_sha256": None,
           "max_cand": geo.get("max_cand", 2_000_000),
           "items": 0, "matched": 0, "refused": 0, "coord_solution_hist": {},
           "hits": [], "hit_count": 0, "decompositions": 0, "undecided": []}
    numerics = []
    extra = {"candidates": 0, "native_runs": 0, "dense_raw": 0, "b6_paths": {}, "filter_fallbacks": [],
             "kernel_s": 0.0, "match_s": 0.0, "deadline_hit": False}
    if stage == "A6":
        from cover_census import _reduce
        rec["census_sha256"] = part["census"]["sha256"]
        units = geo["units"]
        rec["batch"]["units"] = units
        rec["batch"]["n_units"] = len(units)
        plans = {}
        for i, j, _ in units:
            if R.past_deadline():
                rec["undecided"].append({"unit": [int(i), int(j)], "reason": "deadline: pivot pair not run"})
                extra["deadline_hit"] = True
                continue
            tk = time.time()
            if i not in plans:
                Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
                members, _ = E.pivot_plan(i)
                mask = np.zeros(E.N, dtype=bool)
                mask[members] = True
                plans[i] = (Qi, mask)
            Qi, mask = plans[i]
            covers, nc, _ = E.pair_covers(6, i, j, Qi, mask)
            extra["kernel_s"] += time.time() - tk
            extra["candidates"] += int(nc)
            rec["items"] += len(covers)
            tm = time.time()
            for cover in sorted(covers):
                match_item(R, cover, rec, numerics, extra)
            extra["match_s"] += time.time() - tm
    else:
        lists, doc = common.load_reps(os.path.join(common.HERE, part["reps"]["file"]), part["reps"]["sha256"])
        rec["reps_sha256"] = doc["sha256"]
        key = {"B6": "B6", "C6": "C6", "beta": "k5", "gamma": "k4"}[stage]
        ids = geo["item_ids"]
        rec["batch"]["item_ids"] = ids
        rec["batch"]["n_items"] = len(ids)
        rec["batch"]["class"] = geo.get("class")
        tm = time.time()
        for k in ids:
            item = tuple(int(u) for u in lists[key][k])
            kappa = int(lists["B6_kappa"][k]) if stage == "B6" else None
            rec["items"] += 1
            match_item(R, item, rec, numerics, extra, kappa)
        extra["match_s"] += time.time() - tm
    extra["deadline_hit"] |= any(str(u.get("reason", "")).startswith("deadline") for u in rec["undecided"])
    rec["hit_count"] = len(rec["hits"])
    rec["decompositions"] = sum(h["decomposition"] for h in rec["hits"])
    rec["deterministic_sha256"] = common.sha256_json(rec)
    extra["wall_s"] = time.time() - R.t0
    extra["hit_numerics"] = numerics
    extra["matcher"] = ("invisible3" if stage in ("beta", "gamma") else
                        "native" if R.M.native_cls is not None else "reference")
    extra["filter"] = "dense_solve" if R.Fl is not None else None
    extra["cover6"] = "native" if E.native_cover6 is not None else "reference"
    return rec, extra


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
                    help="Python references for the 6-cover kernel, the stage A matcher and the dense solve")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop after S seconds; the running item is aborted and runs not made are recorded as "
                         "undecided")
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
    size = (f"{len(geo['units'])} pivot pairs" if geo["stage"] == "A6"
            else f"{len(geo['item_ids'])} items ({geo.get('class')})")
    print(f"batch {index} of {part['batches']}: stage {geo['stage']}, {size}, {RUNS_PER_ITEM[geo['stage']]} run(s) "
          f"per item, {'compiled kernels' if native else 'Python references'}, candidate cap "
          f"{geo.get('max_cand', 2_000_000)}" + (f", deadline {a.max_seconds:.0f}s" if a.max_seconds else ""),
          flush=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    rec, extra = run_batch(part, index, native, a.max_seconds, verbose=a.verbose)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "native_runs": extra["native_runs"], "candidates": extra["candidates"], "dense_raw": extra["dense_raw"],
        "b6_paths": extra["b6_paths"], "filter_fallbacks": extra["filter_fallbacks"],
        "kernel_s": extra["kernel_s"], "match_s": extra["match_s"],
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system, "max_seconds": a.max_seconds,
        "deadline_hit": extra["deadline_hit"],
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "git_commit": common.git_commit(),
        "native": native, "matcher": extra["matcher"], "filter": extra["filter"], "cover6": extra["cover6"],
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
    print(f"batch {index}: {rec['items']} items, {rec['matched']} matched, {rec['refused']} refused, "
          f"{rec['hit_count']} hits, {rec['decompositions']} decompositions, "
          f"{len(rec['undecided'])} undecided; kernel {rec['kernel_s']:.0f}s, match {rec['match_s']:.0f}s, "
          f"wall {rec['wall_s']:.0f}s; deterministic sha256 {rec['deterministic_sha256'][:16]}", flush=True)
    if rec["decompositions"]:
        h = next(h for h in rec["hits"] if h["decomposition"])
        print(f"DECOMPOSITION FOUND batch {index}: psi_4 lies in the span of {h['terms_count']} stabilizer states "
              f"(numerical rank {h['rank']}) from base cover {h['cover']} at x0 {h['x0']}"
              + (f" with invisible flats {h['flats']}" if "flats" in h else "")
              + f"; chi(N^4) <= {h['rank']}; {rec['decompositions']} such hit(s)", flush=True)
        return 2
    if rec["undecided"]:
        n_dead = sum(str(u.get("reason", "")).startswith("deadline") for u in rec["undecided"])
        print(f"batch {index}: {len(rec['undecided'])} undecided run(s), {n_dead} of them at the deadline; see "
              f"{os.path.relpath(path, common.ROOT)}" + (" (rerun with --resume)" if n_dead else ""), flush=True)
        return 1
    print(f"batch {index} complete: no hit is a decomposition of |N>^4; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
