"""Combinatorial check of the orbit-block pivot order and its masks.

For random 7-sets of states (none inside V_m) it verifies that some group
element moves the set into the form the batch kernel enumerates: the least
label is the first label of its block (the block's representative), the
second least is an admissible second pivot of that block (minimal in its
Stab(i)-orbit), and the third least is an admissible third pivot for that
pair (minimal in its Stab(i, j)-orbit). A random 7-set has no member inside
span(V_m, s_i, s_j) other than by accident, so this checks the ordering
argument of docs/notes/t3_rank7_exclusion.md, section 2, on the masks that
common.py actually produces; the treatment of the members inside
span(V_m, s_i, s_j) is that of scan3.py's kernel3, validated there.

    uv run --extra challenge python research/t3_rank7/check_order.py [--m 3] [--sets 2000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--sets", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv[1:])
    t0 = time.time()
    S = common.Setup(a.m)
    print(f"[{time.time() - t0:.0f}s] m={a.m}: {S.N} states, {len(S.blocks)} blocks, group order "
          f"{S.group_order}", flush=True)
    block_of_start = np.full(S.N, -1, dtype=np.int64)
    for b in S.blocks:
        block_of_start[b["start"]] = b["block"]
    stabs = {b["block"]: S.stabilizer(b["block"]) for b in S.blocks}
    jok = {}
    for b in S.blocks:
        mask = np.zeros(S.N, dtype=bool)
        mask[S.second_pivots(b["block"], stabs[b["block"]])] = True
        jok[b["block"]] = mask
    kok_cache = {}
    rng = np.random.default_rng(a.seed)
    labels = np.flatnonzero(S.isfree_b == 0)
    images = S.perms_b          # images[g, t] = label of g(state t)
    for n in range(a.sets):
        sub = rng.choice(labels, size=7, replace=False)
        imgs = np.sort(images[:, sub], axis=1)          # |G| x 7, sorted labels
        ok = False
        for g in np.flatnonzero(block_of_start[imgs[:, 0]] >= 0):
            i, j, k = int(imgs[g, 0]), int(imgs[g, 1]), int(imgs[g, 2])
            blk = int(block_of_start[i])
            if not jok[blk][j]:
                continue
            if (blk, j) not in kok_cache:
                kok_cache[(blk, j)] = S.third_pivot_mask(j, stabs[blk])
            kok = kok_cache[(blk, j)]
            if kok is not None and not kok[k]:
                continue
            ok = True
            break
        assert ok, f"7-set {sorted(int(S.old_of[x]) for x in sub)} has no admissible form"
        if n % 500 == 499:
            print(f"[{time.time() - t0:.0f}s] {n + 1} sets checked", flush=True)
    print(f"every one of {a.sets} random 7-sets has an admissible canonical form under the "
          f"orbit-block order with the Stab(i, j) mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
