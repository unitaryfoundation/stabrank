"""The batched rank-7 scan: partition consistency, manifest scheduling, and the
m=2 control batch through batch.py and aggregate.py."""

import json
import os
import sys

import pytest

pytest.importorskip("sympy")
pytest.importorskip("numba")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "research", "t3_rank7")
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))
sys.path.insert(0, HERE)

import loop  # noqa: E402
import common  # noqa: E402


def test_partition_matches_count_steps_and_tiles_every_block():
    part = common.load_partition(os.path.join(HERE, "partition.json"))
    counts = json.load(open(os.path.join(HERE, "results", "step_counts.json")))
    assert part["total_steps"] == counts["blocked_steps_with_stab_ij"]
    assert sum(b["steps"] for b in part["batches"]) == part["total_steps"]
    assert sum(b["pairs"] for b in part["per_block"]) == counts["blocked_pairs"]
    assert [b["index"] for b in part["batches"]] == list(range(part["n_batches"]))
    N = part["setup"]["N"]
    for blk in part["setup"]["blocks"]:
        ranges = sorted((b for b in part["batches"] if b["block"] == blk["block"]),
                        key=lambda b: b["j_lo"])
        lo = blk["start"] + 1
        for b in ranges:
            assert b["j_lo"] == lo
            lo = b["j_hi"]
        assert lo == N
    ref = {r["rep"]: r["steps_stab"] for r in counts["per_rep_blocked"]}
    for pb in part["per_block"]:
        assert pb["steps"] == ref[pb["rep"]]


def test_manifest_runs_every_batch_once_in_block_order():
    man = loop.load_manifest(os.path.join(HERE, "manifest_rank7.json"))
    part = common.load_partition(os.path.join(HERE, "partition.json"))
    seq = loop.schedule(man["jobs"], {})
    seeds = [seed for _, seed in seq]
    assert seeds == list(range(part["n_batches"]))
    blocks = [int(jid.rsplit("-", 1)[1]) for jid, _ in seq]
    assert blocks == sorted(blocks)
    cmd = loop.command(man["jobs"][0], 0, os.path.join(HERE, "batch.py"))
    assert cmd[5:9] == ["T3", "3", "7", "--seeds"]
    assert "--partition" in cmd and not man["jobs"][0]["stop_on_solve"]


def test_m2_control_batch_finds_v2_and_aggregate_reports_it(tmp_path):
    import batch
    import aggregate
    part_path = os.path.join(HERE, "partition_m2.json")
    out = str(tmp_path / "batches_m2")
    code = batch.main(["batch.py", "T3", "2", "7", "--seeds", "1", "--seed0", "4", "--chains", "8",
                       "--partition", part_path, "--out-dir", out])
    assert code == 2
    rec = json.load(open(os.path.join(out, "batch_4.json")))
    assert rec["found_count"] == rec["candidates"] > 0 and not rec["undecided"]
    assert min(f["min_terms"] for f in rec["found"] if f["min_terms"]) == 3
    det, full = aggregate.split_record(rec)
    assert common.sha256_json(det) == rec["deterministic_sha256"]
    assert common.sha256_json(full) == rec["sha256"]
    code = aggregate.main(["aggregate.py", "--partition", part_path, "--batches-dir", out,
                           "--dry-run", "--partial"])
    assert code == 2
