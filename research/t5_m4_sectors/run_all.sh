#!/bin/sh
# The whole laptop run, one process at a time, each step at nice 19 under the
# 600 s cap of research/t5_rank5/run.py. Outputs go under results/.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run --extra challenge python research/t5_rank5/run.py --max-seconds 600"
OUT=research/t5_m4_sectors/results
mkdir -p "$OUT"
$RUN --log "$OUT/control.log" -- research/t5_m4_sectors/sector_census.py control
$RUN --log "$OUT/sectors_m4.log" -- research/t5_m4_sectors/sectors_m4.py
$RUN --log "$OUT/census_Z111.log" -- research/t5_m4_sectors/sector_census.py census 1 1 1
$RUN --log "$OUT/census_Z112.log" -- research/t5_m4_sectors/sector_census.py census 1 1 2
$RUN --log "$OUT/census_Z113.log" -- research/t5_m4_sectors/sector_census.py census 1 1 3
$RUN --log "$OUT/census_Z114.log" -- research/t5_m4_sectors/sector_census.py census 1 1 4
$RUN --log "$OUT/census_Z123.log" -- research/t5_m4_sectors/sector_census.py census 1 2 3
$RUN --log "$OUT/census_Z111_d1.log" -- research/t5_m4_sectors/sector_census.py census 1 1 1 --d 1 --seed 23
$RUN --log "$OUT/strings.log" -- research/t5_m4_sectors/strings_report.py
