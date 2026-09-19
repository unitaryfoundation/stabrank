"""Write the fixed batch partition of the three-pivot scan.

A batch is (block a, range of second pivots [j_lo, j_hi)) in the orbit-block
labelling of common.py. The ranges of a block tile [start + 1, N) exactly, so
the aggregator can check coverage from the ranges alone. Batches are cut
greedily at about `--target-steps` inner steps each; the exact step count of
every batch is recorded and the sum equals the total of count_steps.py
(`blocked_steps_with_stab_ij` in results/step_counts.json at m=3).

    uv run --extra challenge python research/t3_rank7/make_partition.py
    uv run --extra challenge python research/t3_rank7/make_partition.py --m 2 \
        --target-steps 2e6 --out research/t3_rank7/partition_m2.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(part, part_path, path, cap_s):
    """A loop.py manifest with one job per block: the seeds are the block's
    batch indices, all in one round, so the batches run block by block in
    priority order (block 0 first). Nothing stops on a decomposition; a batch
    that finds one exits 2, which the loop logs as a miss, with the
    DECOMPOSITION FOUND line as its last stdout line."""
    rel = os.path.relpath(part_path, common.ROOT)
    out_dir = os.path.relpath(os.path.join(
        common.HERE, "batches" if part["m"] == 3 else f"batches_m{part['m']}"), common.ROOT)
    jobs = []
    nblocks = len(part["per_block"])
    for pb in part["per_block"]:
        jobs.append({"id": f"T3-m{part['m']}-r{part['rank']}-block-{pb['block']:02d}",
                     "orbit": "T3", "m": part["m"], "rank": part["rank"],
                     "seed0": pb["first_batch"], "seeds": pb["batches"],
                     "seeds_per_round": pb["batches"], "priority": nblocks - pb["block"],
                     "note": f"orbit of size {pb['orbit_size']} (representative {pb['rep']}), "
                             f"{pb['pairs']} pivot pairs, {pb['steps']:.3e} inner steps"})
    man = {"name": os.path.splitext(os.path.basename(path))[0],
           "description": f"batches of the three-pivot scan excluding rank {part['rank']} for "
                          f"|T3>^{part['m']}; run with --runner research/t3_rank7/batch.py; the "
                          f"seed is the batch index into {rel}",
           "defaults": {"seeds": 1, "stop_on_solve": False, "cap_s": cap_s,
                        "extra_args": ["--partition", rel, "--out-dir", out_dir]},
           "jobs": jobs}
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(f"{len(jobs)} jobs written {os.path.relpath(path, common.ROOT)}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--rank", type=int, default=7)
    ap.add_argument("--target-steps", type=float, default=3e10,
                    help="inner steps per batch; a batch closes at the first j that reaches it")
    ap.add_argument("--out", default=os.path.join(common.HERE, "partition.json"))
    ap.add_argument("--manifest", default=None,
                    help="loop.py manifest to write (default manifest_rank<RANK>.json at m=3, "
                         "manifest_m<M>.json otherwise)")
    ap.add_argument("--cap-s", type=int, default=3 * 3600, help="wall-clock cap per batch")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rewrite the manifest from the existing partition file")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv[1:])
    manifest = a.manifest or os.path.join(
        common.HERE, f"manifest_rank{a.rank}.json" if a.m == 3 else f"manifest_m{a.m}.json")
    if a.manifest_only:
        write_manifest(common.load_partition(a.out), a.out, manifest, a.cap_s)
        return 0
    t0 = time.time()
    S = common.Setup(a.m, cache=not a.no_cache)
    print(f"[{time.time() - t0:.0f}s] m={a.m}: {S.N} states, group order {S.group_order}, "
          f"{len(S.blocks)} blocks, {S.nfree} states inside V_{a.m}", flush=True)
    target = int(a.target_steps)
    batches = []
    per_block = []
    total = 0
    for b in S.blocks:
        stab = S.stabilizer(b["block"])
        js = S.second_pivots(b["block"], stab)
        steps_j = np.array([S.pair_steps(j, S.third_pivot_mask(j, stab)) for j in js], dtype=np.int64)
        nontrivial = sum(1 for j in js if S.third_pivot_mask(j, stab) is not None)
        block_steps = int(steps_j.sum())
        total += block_steps
        per_block.append({"block": b["block"], "rep": b["rep"], "orbit_size": b["orbit_size"],
                          "start": b["start"], "pairs": int(len(js)),
                          "pairs_with_nontrivial_stab_ij": nontrivial, "steps": block_steps,
                          "first_batch": len(batches)})
        j_lo = b["start"] + 1
        acc = 0
        n_pairs = 0
        for t, j in enumerate(js):
            acc += int(steps_j[t])
            n_pairs += 1
            last = t == len(js) - 1
            if acc >= target or last:
                j_hi = S.N if last else int(js[t + 1])
                batches.append({"index": len(batches), "block": b["block"], "j_lo": int(j_lo),
                                "j_hi": j_hi, "pairs": n_pairs, "steps": acc})
                j_lo, acc, n_pairs = j_hi, 0, 0
        per_block[-1]["batches"] = len(batches) - per_block[-1]["first_batch"]
        print(f"[{time.time() - t0:.0f}s] block {b['block']} (orbit {b['orbit_size']}, start "
              f"{b['start']}): {len(js)} pairs, {block_steps:.3e} steps, "
              f"{per_block[-1]['batches']} batches", flush=True)
    assert sum(x["steps"] for x in batches) == total
    body = {"m": a.m, "rank": a.rank, "need": a.rank - 3, "target_steps": target,
            "total_steps": total, "n_batches": len(batches), "setup": S.describe(),
            "per_block": per_block, "batches": batches,
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit()}
    body["sha256"] = common.sha256_json(body)
    with open(a.out, "w") as f:
        json.dump(body, f, indent=1)
        f.write("\n")
    print(f"{len(batches)} batches, {total} inner steps ({total:.4e}), largest batch "
          f"{max(x['steps'] for x in batches):.3e}, written {os.path.relpath(a.out, common.ROOT)}")
    write_manifest(body, a.out, manifest, a.cap_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
