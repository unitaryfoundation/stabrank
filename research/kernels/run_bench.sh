#!/bin/sh
# The benchmark sequence behind docs/notes/kernels_k6_p3.md: every step
# through research/t5_rank5/run.py (nice 19, its own session, a 600 s cap),
# one process at a time, from the repository root.
set -u
cd "$(dirname "$0")/../.." || exit 1
RUN="uv run --extra challenge --extra test python research/t5_rank5/run.py --max-seconds 600 --"
B=research/kernels/bench.py
R=research/kernels/results
$RUN $B cover6 qubit_H 3 --pairs 3,932 811,924 582,1025 1074,1043 --reference --out $R/cover6_qubit_H_n3_reference.json
$RUN $B cover6 qubit_H 3 --pairs 3,706 356,540 3,354 3,932 811,924 --out $R/cover6_qubit_H_n3_heavy.json
$RUN $B cover6 qubit_T 3 --pairs u:4671 u:4200 --reference --out $R/cover6_qubit_T_n3_reference.json
$RUN $B cover6 N 2 --pairs u:1208 u:1200 u:1190 u:1150 u:1100 u:600 --reference --out $R/cover6_N_n2_reference.json
$RUN $B cover6 N 2 --pairs u:0 u:1 u:16 u:17 u:52 u:200 --out $R/cover6_N_n2_heavy.json
$RUN $B stage-a3 N --count 200 --x0 0,0 2,2
$RUN $B stage-a3 H3 --count 200 --x0 1,1 0,0
$RUN $B dense qubit_H --count 8
$RUN $B dense qubit_T --count 8
$RUN $B cover5-key qubit_H 3 --pairs 3,706 356,540 3,354 u:0 u:1000
$RUN $B cover5-key qubit_H 4 --pairs 215,22355 541,10359
$RUN $B dense3 N --count 6 --x0 2,2
$RUN $B dense3 N --count 4 --x0 0,0
