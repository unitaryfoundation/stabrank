"""Certificate-side check of the batch outputs of the rank-4 exclusion of |T>^5.

    aggregate.py [--partition P] [--results-dir D] [--dry-run] [--partial]
                 [--no-reenumerate] [--recheck N] [--recheck-seed S] [--recheck-dir D]

Checks, in order: the partition's own hash; that the census covers4.json
has the hash the partition names and, unless --no-reenumerate (about three
minutes), equals a fresh enumeration of the full 4-covers of |T>^3 (the
covers of distinct independent states from CoverEnumerator.covers(4) and
the degenerate multisets over the full 3-covers, with the same kinds); that
the batches' cover ids tile the census exactly once with the right kind;
every batch file present; each file's two hashes (deterministic part, whole
record); geometry, partition hash, census hash and cell equal to the
partition's; run counts consistent (matched + refused + undecided = 2 x
covers); every `undecided` list empty and every `refused` count zero; and
every hit re-decided here from its phase codes (psi_5 = |T>^5 against the
span of the terms mod 2013265921 in the Q(zeta_24) field and numerically)
with no hit a decomposition.

Then, unless --dry-run, it re-runs N batches from scratch (indices drawn
from numpy's default_rng(S) over the batches present, S printed on a
`seed:` line) into a scratch directory and compares the deterministic hash
of each with the stored one, bit for bit. When the stored checks pass it
writes batch_manifest.json (one entry per batch with its parameters, output
path and SHA-256, the file `certificate.attested.batches` names), or
batch_manifest.partial.json while batches are missing.

Prints `CERTIFIED chi(qubit_T^5) >= 5` only when every batch is present,
every check above holds, no hit is a decomposition and the re-runs match.
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
import common  # noqa: E402  (research/t5_rank4/common.py)
from common import CLAIM, M, N1, ORBIT, RANK, STAGES, X0S  # noqa: E402

DETERMINISTIC_KEY = "deterministic_sha256"
FULL_KEY = "sha256"


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
    det, full = split_record(rec)
    if common.sha256_json(det) != rec.get(DETERMINISTIC_KEY):
        problems.append(f"batch {idx}: deterministic hash does not match its content")
    if common.sha256_json(full) != rec.get(FULL_KEY):
        problems.append(f"batch {idx}: record hash does not match its content")
    b = rec.get("batch", {})
    if b.get("index") != idx or b.get("stage") != geo["stage"]:
        problems.append(f"batch {idx}: geometry index/stage {b.get('index')}/{b.get('stage')}, "
                        f"partition says {idx}/{geo['stage']}")
    if b.get("cover_ids") != geo["cover_ids"]:
        problems.append(f"batch {idx}: cover ids differ from the partition's")
    if rec.get("census_sha256") != part["census"]["sha256"]:
        problems.append(f"batch {idx}: ran against a different census")
    if rec.get("covers") != len(geo["cover_ids"]):
        problems.append(f"batch {idx}: {rec.get('covers')} covers, partition lists {len(geo['cover_ids'])}")
    if rec.get("partition_sha256") != part["sha256"]:
        problems.append(f"batch {idx}: ran against a different partition")
    for key, want in (("orbit", ORBIT), ("m", M), ("rank", RANK), ("n1", N1), ("x0", list(X0S)),
                      ("N", part["N"]), ("group_order", part["group_order"])):
        if rec.get(key) != want:
            problems.append(f"batch {idx}: {key} = {rec.get(key)}, expected {want}")
    if rec.get("matched", 0) + rec.get("refused", 0) + len(rec.get("undecided", [])) \
            != len(X0S) * rec.get("covers", 0):
        problems.append(f"batch {idx}: matched + refused + undecided runs do not equal {len(X0S)} x covers")
    if sum(rec.get("coord_solution_hist", {}).values()) != rec.get("matched", 0):
        problems.append(f"batch {idx}: the solution histogram does not sum to matched")
    if rec.get("undecided"):
        problems.append(f"batch {idx}: {len(rec['undecided'])} undecided run(s) "
                        f"({rec['undecided'][0].get('reason', '')[:80]})")
    if rec.get("refused"):
        # a refusal is a dead ordinary coefficient, which the census excludes;
        # one in a batch means the matcher and the census disagree
        problems.append(f"batch {idx}: {rec['refused']} refused (cover, x0) run(s)")
    if rec.get("hit_count") != len(rec.get("hits", [])):
        problems.append(f"batch {idx}: hit_count disagrees with the hits list")
    if rec.get("decompositions") != sum(h.get("decomposition", False) for h in rec.get("hits", [])):
        problems.append(f"batch {idx}: decompositions disagrees with the hits' decisions")


def redecide_hits(records, problems):
    """Every stored hit re-decided from its phase codes. Returns the list of
    (batch index, hit) that are decompositions of psi_5."""
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


def check_census(part, reenumerate, problems):
    """The stored census against its hash, the partition's tiling of it by
    kind, the enumerator's dictionary and group, and (optionally) a fresh
    enumeration."""
    E = common.make_enumerator()
    if E.N != part["N"] or E.info["order"] != part["group_order"]:
        problems.append("dictionary size or symmetry order differs from the partition's")
    path = os.path.join(common.HERE, part["census"]["file"])
    try:
        covers, doc = common.load_census(path, part["census"]["sha256"])
    except (OSError, ValueError) as exc:
        problems.append(str(exc))
        return E, None
    kinds = doc["kinds"]
    if len(covers) != part["census"]["count"]:
        problems.append(f"census has {len(covers)} covers, partition says {part['census']['count']}")
    for k, key in (("A", "independent"), ("B", "dependent"), ("C", "repeated")):
        if kinds.count(k) != part["census"][key]:
            problems.append(f"census has {kinds.count(k)} kind {k} covers, partition says {part['census'][key]}")
    seen = []
    for geo in part["batch_geometry"]:
        for k in geo["cover_ids"]:
            if not 0 <= k < len(covers):
                problems.append(f"batch {geo['index']}: cover id {k} outside the census")
                continue
            if kinds[k] != geo["stage"]:
                problems.append(f"batch {geo['index']}: cover {k} is of kind {kinds[k]}, not {geo['stage']}")
            seen.append(k)
    if sorted(seen) != list(range(len(covers))):
        problems.append("the batches' cover ids do not tile the census exactly once")
    if reenumerate:
        from driver import census_lists
        t0 = time.time()
        covers3, covers4, deg, bases, fresh_kinds = census_lists(E)
        same = [tuple(c) for c in bases] == covers and fresh_kinds == kinds
        if not same:
            problems.append(f"fresh enumeration lists {len(bases)} bases ({len(covers4)} independent, {len(deg)} "
                            f"degenerate), the stored census {len(covers)}, or they differ")
        print(f"census re-enumerated in {time.time() - t0:.0f}s: {len(covers3)} full 3-covers, {len(covers4)} "
              f"independent full 4-covers, {len(deg)} degenerate multisets, {len(bases)} bases "
              f"({'equal to' if same else 'DIFFERENT from'} the stored census)")
    return E, covers


def write_manifest(path, part, records, results_dir, complete):
    """The batch manifest behind the attested bound: one entry per stored
    batch with its parameters, output path and file hash (the fields
    verify_challenge/stabrank_verify.load_batch_manifest requires)."""
    entries = []
    for idx in sorted(records):
        geo = common.batch_geometry(part, idx)
        out = os.path.join(results_dir, f"batch_{idx}.json")
        params = {"stage": geo["stage"], "covers": len(geo["cover_ids"]),
                  "matched": records[idx]["matched"], "hits": records[idx]["hit_count"]}
        entries.append({"id": f"batch_{idx}", "params": params,
                        "output": os.path.relpath(out, common.ROOT), "sha256": common.sha256_file(out)})
    doc = {"partition": os.path.relpath(common.PARTITION, common.ROOT), "partition_sha256": part["sha256"],
           "census_sha256": part["census"]["sha256"], "complete": complete, "batches": entries}
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
    ap.add_argument("--no-reenumerate", action="store_true", help="trust the stored census by its hash alone")
    ap.add_argument("--manifest", default=None,
                    help="where to write the batch manifest (default batch_manifest.json, or "
                         "batch_manifest.partial.json when batches are missing)")
    ap.add_argument("--no-manifest", action="store_true", help="do not write a manifest")
    ap.add_argument("--recheck", type=int, default=2, help="batches to re-run from scratch")
    ap.add_argument("--recheck-seed", type=int, default=20260924)
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
    print(f"partition {os.path.relpath(a.partition, common.ROOT)}: {ORBIT} m={M} rank={RANK} n1={N1} "
          f"x0={part['x0']}, {part['batches']} batches (A {part['stage_a_batches']}, B {part['stage_b_batches']}, "
          f"C {part['stage_c_batches']}), sha256 {part['sha256'][:16]}")
    check_census(part, not a.no_reenumerate, problems)
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
    tot = {s: {"batches": 0, "covers": 0, "matched": 0, "refused": 0, "hits": 0, "undecided": 0, "cpu_s": 0.0,
               "wall_s": 0.0, "match_s": 0.0} for s in STAGES}
    for idx, r in records.items():
        s = r["batch"]["stage"]
        tot[s]["batches"] += 1
        for k in ("covers", "matched", "refused"):
            tot[s][k] += r.get(k, 0)
        tot[s]["hits"] += r.get("hit_count", 0)
        tot[s]["undecided"] += len(r.get("undecided", []))
        for k in ("cpu_s", "wall_s", "match_s"):
            tot[s][k] += r.get(k, 0.0)
    found = redecide_hits(records, problems)
    print(f"batches present {len(records)}/{B}" + (f", missing {len(missing)}" if missing else ""))
    stage_total = {"A": part["census"]["independent"], "B": part["census"]["dependent"],
                   "C": part["census"]["repeated"]}
    for s in STAGES:
        t = tot[s]
        if not t["batches"]:
            continue
        n = len(by_stage[s])
        rate = t["cpu_s"] / max(1, t["covers"])
        proj = rate * stage_total[s]
        est_s = part["estimated_s"][s]
        print(f"kind {s}: {t['batches']}/{n} batches, {t['covers']} covers, {t['matched']} matched, "
              f"{t['refused']} refused, {t['undecided']} undecided, {t['hits']} hits; {t['cpu_s'] / 60:.1f} CPU-min "
              f"({t['cpu_s'] / t['batches']:.0f}s per batch, {rate * 1e3:.1f} ms per cover); "
              f"projected kind total {proj / 60:.1f} CPU-min at this rate (partition estimate {est_s / 60:.1f})")
    for p in problems[:50]:
        print(f"PROBLEM: {p}")
    if len(problems) > 50:
        print(f"... {len(problems) - 50} more problems")
    if found:
        idx, h = found[0]
        print(f"DECOMPOSITION FOUND: psi_5 lies in the span of {h['terms_count']} stabilizer states "
              f"(batch {idx}, numerical rank {h['rank']}, base cover {h['cover']} at x0 {h['x0']}, "
              f"residual {h['residual']:.2e}); chi(qubit_T^5) <= {h['rank']}; {len(found)} such hit(s)")
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
              "the census matches its hash and is tiled exactly once, no refused or undecided run, no hit is a "
              "decomposition of |T>^5")
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
    out_dir = a.recheck_dir or tempfile.mkdtemp(prefix="t5_rank4_recheck_")
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
        print(f"  batch {idx} (kind {new['batch']['stage']}): {dt:.0f}s, deterministic hash "
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
