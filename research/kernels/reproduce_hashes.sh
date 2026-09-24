#!/bin/sh
# Re-run stored batches of the certified exclusions through the rebuilt
# kernels into a scratch directory and compare the deterministic hashes with
# the stored records: T^5 stage A batch 4 (cover5_pair and SliceMatchKernel,
# unchanged kernels after the refactor onto cpp/src/modular_detail.hpp), T^5
# stage B batch 11 (the Python family path, now through the compiled dense
# solve), and the qutrit N stage A batch 0 (the reference matcher's record
# against the compiled p = 3 stage A). One process at a time, from the
# repository root; OUT is the scratch directory.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${1:-/tmp/stabrank-hash-check}
mkdir -p "$OUT/t5" "$OUT/N"
RUN="uv run --extra challenge --extra test python research/t5_rank5/run.py --max-seconds 600 --"
$RUN research/t5_rank5/batch.py 4 --out-dir "$OUT/t5" --force
$RUN research/t5_rank5/batch.py 11 --out-dir "$OUT/t5" --force
$RUN research/qutrit_m4_rank5/batch.py N 0 --out-dir "$OUT/N" --force
uv run --extra challenge python research/kernels/compare_hashes.py "$OUT"
