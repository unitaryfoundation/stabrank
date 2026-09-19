"""Certificate-side check of the batch outputs of the rank-7 scan.

    aggregate.py [--partition P] [--batches-dir D] [--dry-run] [--partial]
                 [--recheck N] [--recheck-seed S]

Checks, in order: the partition's own hash; every batch file present; each
file's two hashes (deterministic part, whole record); geometry, partition
hash, setup hash and cell equal to the partition's; nominal step count equal
to the partition's; the j-ranges of every block tiling [start + 1, N); the
step counts summing to the partition total; every `undecided` list empty; and
the `found` lists. Then, unless --dry-run, it re-runs a deterministic subset
of N batches from scratch (symmetry cache bypassed, indices drawn from
numpy's default_rng(S), S printed) into a scratch directory and compares the
deterministic hash of each with the stored one, bit for bit.

Prints `CERTIFIED chi(T3^3) >= 8` only when the partition is the m=3 rank-7
one, every check above holds, no class set contains V_3 and the re-runs
match. --dry-run reports the same checks without the re-runs and never
certifies. --partial reports progress over the batches present instead of
failing on the missing ones. Exit 0 when certified or (dry run) when every
stored check passes, 2 when a stored batch reports a decomposition, 1
otherwise.
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
import common  # noqa: E402

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
    for key in ("index", "block", "j_lo", "j_hi", "pairs", "steps"):
        if b.get(key) != geo[key]:
            problems.append(f"batch {idx}: geometry {key} = {b.get(key)}, partition says {geo[key]}")
    if rec.get("partition_sha256") != part["sha256"]:
        problems.append(f"batch {idx}: ran against a different partition")
    if rec.get("setup_sha256") != part["setup"]["setup_sha256"]:
        problems.append(f"batch {idx}: setup hash differs from the partition's")
    for key in ("m", "rank", "need"):
        if rec.get(key) != part[key]:
            problems.append(f"batch {idx}: {key} = {rec.get(key)}, partition says {part[key]}")
    if rec.get("steps_nominal") != geo["steps"]:
        problems.append(f"batch {idx}: {rec.get('steps_nominal')} nominal steps, partition says "
                        f"{geo['steps']}")
    if rec.get("pairs_scanned") != geo["pairs"]:
        problems.append(f"batch {idx}: {rec.get('pairs_scanned')} pairs scanned, partition says "
                        f"{geo['pairs']}")
    if rec.get("candidates") != rec.get("decided", 0) + rec.get("spurious_count", 0) \
            + rec.get("oversize_count", 0):
        problems.append(f"batch {idx}: candidates do not split into decided, spurious and oversize")
    if rec.get("undecided"):
        problems.append(f"batch {idx}: {len(rec['undecided'])} undecided class set(s)")
    if rec.get("found_count", 0) != len(rec.get("found", [])) and rec.get("found_count", 0) <= 256:
        problems.append(f"batch {idx}: found_count disagrees with the found list")


def coverage(part, present):
    """Blocks whose present batches do not tile [start + 1, N)."""
    N = part["setup"]["N"]
    by_block = {}
    for geo in part["batches"]:
        if geo["index"] in present:
            by_block.setdefault(geo["block"], []).append(geo)
    bad = []
    for blk in part["setup"]["blocks"]:
        ranges = sorted(by_block.get(blk["block"], []), key=lambda g: g["j_lo"])
        lo = blk["start"] + 1
        ok = bool(ranges)
        for g in ranges:
            if g["j_lo"] != lo:
                ok = False
            lo = g["j_hi"]
        if lo != N:
            ok = False
        if not ok:
            bad.append(blk["block"])
    return bad


def rerun(part_path, index, m, rank, out_dir):
    cmd = ["nice", "-n", "19", sys.executable, os.path.join(common.HERE, "batch.py"), "T3",
           str(m), str(rank), "--seeds", "1", "--seed0", str(index), "--partition", part_path,
           "--out-dir", out_dir, "--no-cache"]
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[v] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=common.ROOT, env=env, capture_output=True, text=True)
    return proc, time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--partition", default=os.path.join(common.HERE, "partition.json"))
    ap.add_argument("--batches-dir", default=os.path.join(common.HERE, "batches"))
    ap.add_argument("--dry-run", action="store_true", help="every check except the re-runs")
    ap.add_argument("--partial", action="store_true",
                    help="report progress over the batches present; never certifies")
    ap.add_argument("--recheck", type=int, default=3, help="batches to re-run from scratch")
    ap.add_argument("--recheck-seed", type=int, default=20260919)
    ap.add_argument("--recheck-dir", default=None,
                    help="where the re-runs write (default: a fresh temporary directory)")
    a = ap.parse_args(argv[1:])
    part = common.load_partition(a.partition)
    B = part["n_batches"]
    problems, missing, records = [], [], {}
    for geo in part["batches"]:
        path = os.path.join(a.batches_dir, f"batch_{geo['index']}.json")
        if not os.path.exists(path):
            missing.append(geo["index"])
            continue
        with open(path) as f:
            rec = json.load(f)
        records[geo["index"]] = rec
        check_batch(rec, part, geo, problems)
    present = set(records)
    steps = sum(r["steps_nominal"] for r in records.values())
    steps_done = sum(r.get("steps_done", 0) for r in records.values())
    expected_steps = sum(g["steps"] for g in part["batches"] if g["index"] in present)
    if steps != expected_steps:
        problems.append(f"stored step counts sum to {steps}, the partition's batches to "
                        f"{expected_steps}")
    if not missing and steps != part["total_steps"]:
        problems.append(f"step total {steps} differs from the partition total {part['total_steps']}")
    bad_blocks = coverage(part, present)
    if not missing and bad_blocks:
        problems.append(f"blocks whose j-ranges do not tile their range: {bad_blocks}")
    found = [(idx, f) for idx, r in records.items() for f in r.get("found", [])]
    found_total = sum(r.get("found_count", 0) for r in records.values())
    spurious = sum(r.get("spurious_count", 0) for r in records.values())
    candidates = sum(r.get("candidates", 0) for r in records.values())
    hist = {}
    for r in records.values():
        for k, v in r.get("histogram", {}).items():
            hist[k] = hist.get(k, 0) + v
    kernel_s = sum(r.get("kernel_s", 0.0) for r in records.values())
    cpu_s = sum(r.get("cpu_s", 0.0) for r in records.values())
    wall_s = sum(r.get("wall_s", 0.0) for r in records.values())
    m, rank = part["m"], part["rank"]
    print(f"partition {os.path.relpath(a.partition, common.ROOT)}: m={m} rank={rank}, {B} batches, "
          f"{part['total_steps']:.4e} inner steps, sha256 {part['sha256'][:16]}")
    print(f"batches present {len(records)}/{B}" + (f", missing {len(missing)}" if missing else ""))
    if records:
        ns = 1e9 * kernel_s / max(1, steps_done)
        print(f"stored: {steps:.4e} steps ({100 * steps / part['total_steps']:.2f}% of the total), "
              f"{candidates} candidate class sets, {found_total} containing V_{m}, {spurious} "
              f"spurious, {cpu_s / 3600:.2f} CPU-h, {wall_s / 3600:.2f} h wall, {ns:.1f} ns/step; "
              f"projected total {ns * part['total_steps'] / 3.6e12:.0f} CPU-h at this rate")
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:12]
        print("histogram (zero members, parallel members): "
              + ", ".join(f"{k}: {v}" for k, v in top) + (" ..." if len(hist) > 12 else ""))
    for p in problems[:50]:
        print(f"PROBLEM: {p}")
    if len(problems) > 50:
        print(f"... {len(problems) - 50} more problems")
    if found:
        best = min(found, key=lambda t: (t[1].get("min_terms") or 99, t[1]["rank"]))
        idx, f = best
        print(f"DECOMPOSITION FOUND: V_{m} lies in the span of {f.get('min_terms')} stabilizer "
              f"states (batch {idx}, class rank {f['rank']}, members {f['members']}); "
              f"chi(T3^{m}) <= {f.get('min_terms')}; {found_total} class set(s) in total contain V_{m}")
        return 2
    if missing and not a.partial:
        print(f"NOT CERTIFIED: {len(missing)} batch(es) missing, first {missing[:10]}")
        return 1
    if problems:
        print(f"NOT CERTIFIED: {len(problems)} problem(s)")
        return 1
    if missing:
        print(f"partial: every stored check passes for the {len(records)} batches present; "
              f"{len(missing)} to go")
    else:
        print("every stored check passes: all batches present, hashes verify, geometry and step "
              "counts match the partition, ranges tile every block, no undecided class set, no "
              f"class set contains V_{m}")
    if a.dry_run:
        print("dry run: re-runs skipped; NOT CERTIFIED")
        return 1 if missing else 0
    pool = sorted(present)
    n = min(a.recheck, len(pool))
    rng = np.random.default_rng(a.recheck_seed)
    picks = sorted(int(pool[x]) for x in rng.choice(len(pool), size=n, replace=False))
    out_dir = a.recheck_dir or tempfile.mkdtemp(prefix="t3_rank7_recheck_")
    print(f"re-running {n} batch(es) from scratch, seed {a.recheck_seed}: {picks} -> {out_dir}")
    mismatches = []
    for idx in picks:
        proc, dt = rerun(a.partition, idx, m, rank, out_dir)
        path = os.path.join(out_dir, f"batch_{idx}.json")
        if proc.returncode not in (0, 2) or not os.path.exists(path):
            mismatches.append(idx)
            print(f"  batch {idx}: re-run failed (exit {proc.returncode}); stderr tail: "
                  f"{(proc.stderr or '').strip().splitlines()[-1:]}")
            continue
        with open(path) as f:
            new = json.load(f)
        same = new.get(DETERMINISTIC_KEY) == records[idx].get(DETERMINISTIC_KEY)
        print(f"  batch {idx}: {dt:.0f}s, deterministic hash "
              f"{'matches' if same else 'DIFFERS'} ({new.get(DETERMINISTIC_KEY, '')[:16]})")
        if not same:
            mismatches.append(idx)
    if mismatches:
        print(f"NOT CERTIFIED: re-run mismatch on batch(es) {mismatches}")
        return 1
    if missing:
        print(f"partial: the {n} re-run(s) match; NOT CERTIFIED ({len(missing)} batches missing)")
        return 1
    if m == 3 and rank == 7:
        print("CERTIFIED chi(T3^3) >= 8")
    else:
        print(f"no rank-{rank} decomposition of |T3>^{m} over the stabilizer dictionary "
              f"(control run, not a board claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
