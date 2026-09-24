"""Compare the deterministic hashes of re-run batches (reproduce_hashes.sh)
with the stored records, and write research/kernels/results/hash_check.json.

    compare_hashes.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

PAIRS = [
    ("T^5 stage A batch 4", "research/t5_rank5/results/batch_4.json", "t5/batch_4.json"),
    ("T^5 stage B batch 11", "research/t5_rank5/results/batch_11.json", "t5/batch_11.json"),
    ("N^4 rank 5 stage A batch 0", "research/qutrit_m4_rank5/results/N/batch_0.json", "N/batch_0.json"),
]


def main(argv):
    out_dir = argv[1]
    rows = []
    ok = True
    for label, stored, rerun in PAIRS:
        sp = os.path.join(ROOT, stored)
        rp = os.path.join(out_dir, rerun)
        if not os.path.exists(rp):
            rows.append({"batch": label, "stored": stored, "status": "not run"})
            ok = False
            continue
        with open(sp) as f:
            s = json.load(f)
        with open(rp) as f:
            r = json.load(f)
        same = s["deterministic_sha256"] == r["deterministic_sha256"]
        ok &= same
        row = {"batch": label, "stored": stored, "stored_sha256": s["deterministic_sha256"],
               "rerun_sha256": r["deterministic_sha256"], "equal": same,
               "stored_matched": s.get("matched"), "rerun_matched": r.get("matched"),
               "stored_covers": s.get("covers"), "rerun_covers": r.get("covers"),
               "stored_undecided": len(s.get("undecided", [])), "rerun_undecided": len(r.get("undecided", [])),
               "stored_match_s": s.get("match_s"), "rerun_match_s": r.get("match_s"),
               "stored_kernel_s": s.get("kernel_s"), "rerun_kernel_s": r.get("kernel_s"),
               "stored_host": s.get("hostname"), "stored_matcher": s.get("matcher"), "rerun_matcher": r.get("matcher"),
               "rerun_native_runs": r.get("native_runs")}
        rows.append(row)
        print(f"{label}: {'EQUAL' if same else 'DIFFERENT'} ({s['deterministic_sha256'][:16]} vs "
              f"{r['deterministic_sha256'][:16]}); covers {s.get('covers')} vs {r.get('covers')}, "
              f"match {s.get('match_s', 0):.0f}s (stored, {s.get('hostname')}) vs {r.get('match_s', 0):.0f}s (rerun)")
    rec = {"rows": rows, "all_equal": ok, "host": platform.node(), "machine": platform.machine(),
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.join(HERE, "results", "hash_check.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("all equal" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
