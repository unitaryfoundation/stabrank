"""Certificate-side check of the batch outputs of the rank-6 exclusion of |N>^4.

    aggregate.py [--partition P] [--results-dir D] [--dry-run] [--partial]
                 [--no-reenumerate] [--no-reenumerate-census]
                 [--recheck N] [--recheck-seed S] [--recheck-dir D]

Checks, in order: the partition's own hash; that the stage A6 units of the
partition are exactly the pivot pairs of the enumerator in its order and
that the 6-cover census file has the hash the partition names and the same
pivot pairs; that the orbit-representative lists (reps_N.json) have the
hash and counts the partition names and, unless --no-reenumerate (about
two minutes), equal a fresh enumeration from the rank-5 census files (the
dependent 6-sets by every route with their family dimensions, the repeated
6-multisets, the full 5-multisets and the full 4-multisets, each reduced
to the least element of its G_2 orbit by driver.build_lists); unless
--no-reenumerate-census (about seven minutes), that cover6_pair re-run over
every pivot pair gives the census's full 6-cover count on every pair; that
the B6, C6, beta and gamma item ids tile their lists exactly once with the
right class and candidate cap; every batch file present; each file's two
hashes (deterministic part, whole record); geometry, partition hash, list
hashes and cell equal to the partition's; run counts consistent (matched +
refused + undecided = runs per item x items, plus one per pivot pair not
run); every `undecided` list empty and every `refused` count zero; every
hit re-decided here from its phase codes (psi_4 = |N>^4 against the span
of the terms mod 2013265921 and numerically) with no hit a decomposition;
and, when every stage A6 batch is present, the stage A6 cover total equal
to the census's 37,201,212.

Then, unless --dry-run, it re-runs N batches from scratch (indices drawn
from numpy's default_rng(S) over the batches present, S printed on a
`seed:` line) into a scratch directory and compares the deterministic hash
of each with the stored one, bit for bit. When the stored checks pass it
writes batch_manifest.json (one entry per batch with its parameters, output
path and SHA-256, the file `certificate.attested.batches` names), or
batch_manifest.partial.json while batches are missing.

Prints `CERTIFIED chi(N^4) >= 7` only when every batch is present, every
check above holds, no hit is a decomposition and the re-runs match.
--dry-run reports the same checks without the re-runs and never certifies.
--partial reports progress over the batches present instead of failing on
the missing ones. Exit 0 when certified or (dry run) when every stored check
passes, 2 when a stored batch reports a decomposition, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (research/n4_rank6/common.py)
from common import CLAIM, M, N1, RANK, RUNS_PER_ITEM, STAGES, X0  # noqa: E402

DETERMINISTIC_KEY = "deterministic_sha256"
FULL_KEY = "sha256"
LIST_KEY = {"B6": "B6", "C6": "C6", "beta": "k5", "gamma": "k4"}


def split_record(rec):
    """(deterministic part, full part) of a batch record, by key order."""
    det, full = {}, {}
    before_det = True
    for k, v in rec.items():
        if k == FULL_KEY:
            break
        full[k] = v
        if k == DETERMINISTIC_KEY:
            before_det = False
        elif before_det:
            det[k] = v
    return det, full


def check_batch(rec, part, geo, problems):
    """Append every discrepancy of one stored batch to `problems`."""
    idx = geo["index"]
    stage = geo["stage"]
    det, full = split_record(rec)
    if common.sha256_json(det) != rec.get(DETERMINISTIC_KEY):
        problems.append(f"batch {idx}: deterministic hash does not match its content")
    if common.sha256_json(full) != rec.get(FULL_KEY):
        problems.append(f"batch {idx}: record hash does not match its content")
    for key in ("native_runs", "candidates", "dense_raw", "b6_paths", "filter_fallbacks"):
        if key in det:
            problems.append(f"batch {idx}: `{key}` sits inside the deterministic part")
    b = rec.get("batch", {})
    if b.get("index") != idx or b.get("stage") != stage:
        problems.append(f"batch {idx}: geometry index/stage {b.get('index')}/{b.get('stage')}, "
                        f"partition says {idx}/{stage}")
    if stage == "A6":
        if b.get("units") != geo["units"]:
            problems.append(f"batch {idx}: pivot pairs differ from the partition's")
        if rec.get("census_sha256") != part["census"]["sha256"]:
            problems.append(f"batch {idx}: ran against a different 6-cover census")
    else:
        if b.get("item_ids") != geo["item_ids"]:
            problems.append(f"batch {idx}: item ids differ from the partition's")
        if rec.get("reps_sha256") != part["reps"]["sha256"]:
            problems.append(f"batch {idx}: ran against a different list of orbit representatives")
        if rec.get("items") != len(geo["item_ids"]):
            problems.append(f"batch {idx}: {rec.get('items')} items, partition lists {len(geo['item_ids'])}")
    if rec.get("max_cand") != geo.get("max_cand", 2_000_000):
        problems.append(f"batch {idx}: candidate cap {rec.get('max_cand')}, partition says {geo.get('max_cand', 2_000_000)}")
    if rec.get("partition_sha256") != part["sha256"]:
        problems.append(f"batch {idx}: ran against a different partition")
    for key, want in (("orbit", common.ORBIT), ("m", M), ("rank", RANK), ("n1", N1), ("x0", list(X0)),
                      ("N", part["N"]), ("group_order", part["group_order"]),
                      ("runs_per_item", RUNS_PER_ITEM[stage])):
        if rec.get(key) != want:
            problems.append(f"batch {idx}: {key} = {rec.get(key)}, expected {want}")
    und = rec.get("undecided", [])
    n_pairs = sum("cover" not in u for u in und)
    if rec.get("matched", 0) + rec.get("refused", 0) + len(und) \
            != RUNS_PER_ITEM[stage] * rec.get("items", 0) + n_pairs:
        problems.append(f"batch {idx}: matched + refused + undecided runs do not equal "
                        f"{RUNS_PER_ITEM[stage]} x items")
    if sum(rec.get("coord_solution_hist", {}).values()) != rec.get("matched", 0):
        problems.append(f"batch {idx}: the solution histogram does not sum to matched")
    if und:
        problems.append(f"batch {idx}: {len(und)} undecided run(s) ({und[0].get('reason', '')[:80]})")
    if rec.get("refused"):
        # a refusal is a dead ordinary coefficient, which every stage list
        # excludes; one in a batch means the matcher and the list disagree
        problems.append(f"batch {idx}: {rec['refused']} refused run(s)")
    if rec.get("hit_count") != len(rec.get("hits", [])):
        problems.append(f"batch {idx}: hit_count disagrees with the hits list")
    if rec.get("decompositions") != sum(h.get("decomposition", False) for h in rec.get("hits", [])):
        problems.append(f"batch {idx}: decompositions disagrees with the hits' decisions")


def redecide_hits(records, problems):
    """Every stored hit re-decided from its phase codes. Returns the list of
    (batch index, hit) that are decompositions of psi_4."""
    found = []
    for idx, rec in sorted(records.items()):
        for h_i, h in enumerate(rec.get("hits", [])):
            d = common.decide_terms(h["terms"])
            for key in ("exact_mod_p2", "numeric", "decomposition", "rank", "terms_count"):
                if d[key] != h.get(key):
                    problems.append(f"batch {idx} hit {h_i}: stored {key} = {h.get(key)}, re-decided {d[key]}")
            if not d["agree"]:
                problems.append(f"batch {idx} hit {h_i}: modular and numeric decisions disagree "
                                f"(residual {d['residual']:.2e})")
            if d["decomposition"]:
                found.append((idx, {**h, "residual": d["residual"]}))
    return found


def check_stage_a(part, problems, recensus):
    E = common.make_enumerator()
    units = common.pairs_of(E)
    stored = [u for geo in part["batch_geometry"] if geo["stage"] == "A6" for u in geo["units"]]
    if units != stored:
        problems.append("the stage A6 units of the partition are not the enumerator's pivot pairs")
    if E.N != part["N"] or E.info["order"] != part["group_order"]:
        problems.append("dictionary size or symmetry order differs from the partition's")
    path = os.path.join(common.HERE, part["census"]["file"])
    cen = None
    try:
        cen = common.load_census6(path, part["census"]["sha256"])
        if cen["full6"] != part["census"]["covers"] or len(cen["rows"]) != len(units) \
                or cen["full6"] != sum(r[3] for r in cen["rows"]) or cen["full6"] != cen["full6_independent"]:
            problems.append("the census file's totals differ from the partition's, or a 6-set is dependent")
        if [[r[0], r[1]] for r in cen["rows"]] != [[i, j] for i, j, _ in units]:
            problems.append("the census rows are not the enumerator's pivot pairs")
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"census: {exc}")
    if recensus and cen is not None:
        from driver import _recensus
        t0 = time.time()
        mism = _recensus(E, cen)
        if mism:
            problems.append(f"cover6_pair re-run: {len(mism)} pivot pair(s) with a different full 6-cover count, "
                            f"first {mism[:3]}")
        print(f"6-cover census re-run over {len(cen['rows'])} pivot pairs in {time.time() - t0:.0f}s: "
              f"{'every count equal' if not mism else 'DIFFERENT counts'}")
    return E


def check_lists(part, reenumerate, problems, E):
    """The stored lists against their hashes and counts, the partition's
    tiling of them, and (optionally) a fresh enumeration."""
    try:
        lists, doc = common.load_reps(os.path.join(common.HERE, part["reps"]["file"]), part["reps"]["sha256"])
    except (OSError, ValueError) as exc:
        problems.append(str(exc))
        return None
    for key in ("B6", "C6", "k5", "k4"):
        if doc[key]["count"] != part["reps"][key]:
            problems.append(f"list {key} has {doc[key]['count']} orbits, partition says {part['reps'][key]}")
    for stage in ("B6", "C6", "beta", "gamma"):
        n = len(lists[LIST_KEY[stage]])
        seen = []
        for geo in part["batch_geometry"]:
            if geo["stage"] != stage:
                continue
            for k in geo["item_ids"]:
                if not 0 <= k < n:
                    problems.append(f"batch {geo['index']}: item id {k} outside the {stage} list")
                    continue
                seen.append(k)
                if stage == "B6":
                    kap = int(lists["B6_kappa"][k])
                    if geo.get("class") != f"kappa {kap}":
                        problems.append(f"batch {geo['index']}: item {k} has kappa {kap}, the batch class is {geo.get('class')}")
                    want_cap = 2_000_000 if kap == 1 else part["batch_geometry"][geo["index"]].get("max_cand")
                    if kap >= 2 and geo.get("max_cand", 2_000_000) <= 2_000_000:
                        problems.append(f"batch {geo['index']}: kappa {kap} item {k} at the default candidate cap")
                    del want_cap
                else:
                    cls = common.item_class(E, lists[LIST_KEY[stage]][k], stage)
                    if geo.get("class") != cls:
                        problems.append(f"batch {geo['index']}: item {k} is of class {cls}, the batch class is {geo.get('class')}")
        if sorted(seen) != list(range(n)):
            problems.append(f"the stage {stage} item ids do not tile the {LIST_KEY[stage]} list exactly once")
    if reenumerate:
        from driver import build_lists
        t0 = time.time()
        fresh = build_lists(E, verbose=False)
        for key in ("B6", "C6", "k5", "k4"):
            same = fresh[key]["codes"] == doc[key]["codes"]
            if key == "B6":
                same = same and fresh["B6"]["kappa"] == doc["B6"]["kappa"]
            if not same:
                problems.append(f"fresh enumeration of the {key} list ({fresh[key]['count']} orbits) differs from "
                                f"the stored one ({doc[key]['count']})")
        print(f"lists re-enumerated in {time.time() - t0:.0f}s: B6 {fresh['B6']['count']}, C6 {fresh['C6']['count']}, "
              f"k5 {fresh['k5']['count']}, k4 {fresh['k4']['count']} orbits "
              f"({'all equal to' if not any('fresh enumeration' in p for p in problems) else 'DIFFERENT from'} "
              "the stored lists)")
    return lists


def write_manifest(path, part, records, results_dir, complete):
    """The batch manifest behind the attested bound: one entry per stored
    batch with its parameters, output path and file hash (the fields
    verify_challenge/stabrank_verify.load_batch_manifest requires)."""
    entries = []
    for idx in sorted(records):
        geo = common.batch_geometry(part, idx)
        out = os.path.join(results_dir, f"batch_{idx}.json")
        params = {"stage": geo["stage"]}
        if geo["stage"] == "A6":
            params["pivot_pairs"] = len(geo["units"])
        else:
            params["items"] = len(geo["item_ids"])
            params["class"] = geo.get("class")
        params["runs_per_item"] = RUNS_PER_ITEM[geo["stage"]]
        params["max_cand"] = geo.get("max_cand", 2_000_000)
        params["matched"] = records[idx]["matched"]
        params["hits"] = records[idx]["hit_count"]
        entries.append({"id": f"batch_{idx}", "params": params,
                        "output": os.path.relpath(out, common.ROOT), "sha256": common.sha256_file(out)})
    doc = {"partition": os.path.relpath(common.PARTITION, common.ROOT), "partition_sha256": part["sha256"],
           "census_sha256": part["census"]["sha256"], "reps_sha256": part["reps"]["sha256"],
           "complete": complete, "batches": entries}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, common.ROOT)}: {len(entries)} batch(es)" + ("" if complete else " (partial)"))


def rerun(part_path, index, out_dir):
    cmd = ["nice", "-n", "19", sys.executable, os.path.join(common.HERE, "batch.py"), str(index),
           "--partition", part_path, "--out-dir", out_dir, "--force"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=common.ROOT, capture_output=True, text=True)
    return proc, time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--partition", default=common.PARTITION)
    ap.add_argument("--results-dir", default=common.RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="stored checks only, no re-runs")
    ap.add_argument("--partial", action="store_true", help="report over the batches present")
    ap.add_argument("--no-reenumerate", action="store_true",
                    help="trust the stored orbit-representative lists by their hashes alone")
    ap.add_argument("--no-reenumerate-census", action="store_true",
                    help="trust the 6-cover census file by its hash alone (skip the cover6_pair re-run)")
    ap.add_argument("--manifest", default=None,
                    help="where to write the batch manifest (default batch_manifest.json, or "
                         "batch_manifest.partial.json when batches are missing)")
    ap.add_argument("--no-manifest", action="store_true", help="do not write a manifest")
    ap.add_argument("--recheck", type=int, default=2, help="batches to re-run from scratch")
    ap.add_argument("--recheck-seed", type=int, default=20260925)
    ap.add_argument("--recheck-dir", default=None)
    a = ap.parse_args(argv[1:])
    common.lower_priority()
    t_all = time.time()
    problems = []
    try:
        part = common.load_partition(a.partition)
    except (OSError, ValueError) as exc:
        print(f"NOT CERTIFIED: {exc}")
        return 1
    sb = part["stage_batches"]
    print(f"partition {os.path.relpath(a.partition, common.ROOT)}: {common.ORBIT} m={M} rank={RANK} n1={N1} "
          f"x0={part['x0']}, {part['batches']} batches (" + ", ".join(f"{s} {sb[s]}" for s in STAGES) +
          f"), sha256 {part['sha256'][:16]}")
    E = check_stage_a(part, problems, not a.no_reenumerate_census)
    check_lists(part, not a.no_reenumerate, problems, E)
    geometry = part["batch_geometry"]
    records, missing = {}, []
    for geo in geometry:
        idx = geo["index"]
        path = os.path.join(a.results_dir, f"batch_{idx}.json")
        if not os.path.exists(path):
            missing.append(idx)
            continue
        with open(path) as f:
            records[idx] = json.load(f)
        check_batch(records[idx], part, geo, problems)
    present = set(records)
    B = len(geometry)
    by_stage = {s: [geo["index"] for geo in geometry if geo["stage"] == s] for s in STAGES}
    tot = {s: {"batches": 0, "items": 0, "matched": 0, "refused": 0, "hits": 0, "undecided": 0, "cpu_s": 0.0,
               "wall_s": 0.0, "kernel_s": 0.0, "match_s": 0.0} for s in STAGES}
    for idx, r in records.items():
        s = r["batch"]["stage"]
        tot[s]["batches"] += 1
        for k in ("items", "matched", "refused"):
            tot[s][k] += r.get(k, 0)
        tot[s]["hits"] += r.get("hit_count", 0)
        tot[s]["undecided"] += len(r.get("undecided", []))
        for k in ("cpu_s", "wall_s", "kernel_s", "match_s"):
            tot[s][k] += r.get(k, 0.0)
    if all(i in present for i in by_stage["A6"]):
        if tot["A6"]["items"] != part["census"]["covers"]:
            problems.append(f"stage A6 matched {tot['A6']['items']} covers, the census counted {part['census']['covers']}")
    found = redecide_hits(records, problems)
    print(f"batches present {len(records)}/{B}" + (f", missing {len(missing)}" if missing else ""))
    stage_total = {"A6": part["census"]["covers"], "B6": part["reps"]["B6"], "C6": part["reps"]["C6"],
                   "beta": part["reps"]["k5"], "gamma": part["reps"]["k4"]}
    for s in STAGES:
        t = tot[s]
        if not t["batches"]:
            continue
        n = len(by_stage[s])
        rate = t["cpu_s"] / max(1, t["items"])
        proj = rate * stage_total[s]
        est_s = part["estimated_s"][s]
        print(f"stage {s}: {t['batches']}/{n} batches, {t['items']} items, {t['matched']} matched, "
              f"{t['refused']} refused, {t['undecided']} undecided, {t['hits']} hits; {t['cpu_s'] / 3600:.2f} CPU-h "
              f"({t['cpu_s'] / t['batches']:.0f}s per batch, {rate * 1e3:.2f} ms per item); "
              f"projected stage total {proj / 3600:.2f} CPU-h at this rate (partition estimate {est_s / 3600:.2f} pod)")
    for p in problems[:50]:
        print(f"PROBLEM: {p}")
    if len(problems) > 50:
        print(f"... {len(problems) - 50} more problems")
    if found:
        idx, h = found[0]
        print(f"DECOMPOSITION FOUND: psi_4 lies in the span of {h['terms_count']} stabilizer states "
              f"(batch {idx}, numerical rank {h['rank']}, base cover {h['cover']} at x0 {h['x0']}, "
              f"residual {h['residual']:.2e}); chi(N^4) <= {h['rank']}; {len(found)} such hit(s)")
        return 2
    if missing and not a.partial:
        print(f"NOT CERTIFIED: {len(missing)} batch(es) missing, first {missing[:10]}")
        return 1
    if problems:
        print(f"NOT CERTIFIED: {len(problems)} problem(s)")
        return 1
    if missing:
        print(f"partial: every stored check passes for the {len(records)} batches present; {len(missing)} to go")
    else:
        print("every stored check passes: all batches present, hashes verify, geometry matches the partition, "
              "the stage A6 units are the enumerator's pivot pairs, the census and the lists match their hashes "
              "and are tiled exactly once, no refused or undecided run, no hit is a decomposition of |N>^4")
    if records and not a.no_manifest:
        manifest = a.manifest or os.path.join(common.HERE, "batch_manifest.json" if not missing
                                              else "batch_manifest.partial.json")
        write_manifest(manifest, part, records, a.results_dir, complete=not missing)
    if a.dry_run:
        print("dry run: re-runs skipped; NOT CERTIFIED")
        return 1 if missing else 0
    pool = sorted(present)
    n = min(a.recheck, len(pool))
    rng = np.random.default_rng(a.recheck_seed)
    picks = sorted(int(pool[x]) for x in rng.choice(len(pool), size=n, replace=False)) if n else []
    out_dir = a.recheck_dir or tempfile.mkdtemp(prefix="n4_rank6_recheck_")
    print(f"seed: {a.recheck_seed}")
    print(f"re-running {n} batch(es) from scratch: {picks} -> {out_dir}")
    mismatches = []
    for idx in picks:
        proc, dt = rerun(a.partition, idx, out_dir)
        path = os.path.join(out_dir, f"batch_{idx}.json")
        if proc.returncode not in (0, 2) or not os.path.exists(path):
            mismatches.append(idx)
            print(f"  batch {idx}: re-run failed (exit {proc.returncode}); stderr tail: "
                  f"{(proc.stderr or '').strip().splitlines()[-1:]}")
            continue
        with open(path) as f:
            new = json.load(f)
        same = new.get(DETERMINISTIC_KEY) == records[idx].get(DETERMINISTIC_KEY)
        print(f"  batch {idx} (stage {new['batch']['stage']}): {dt:.0f}s, deterministic hash "
              f"{'matches' if same else 'DIFFERS'} ({new.get(DETERMINISTIC_KEY, '')[:16]})")
        if not same:
            mismatches.append(idx)
    if mismatches:
        print(f"NOT CERTIFIED: re-run mismatch on batch(es) {mismatches}")
        return 1
    if missing:
        print(f"partial: the {n} re-run(s) match; NOT CERTIFIED ({len(missing)} batches missing); "
              f"{time.time() - t_all:.0f}s")
        return 1
    print(f"aggregate complete in {time.time() - t_all:.0f}s")
    print(CLAIM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
