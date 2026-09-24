#!/bin/sh
# The full rank-2 residual table of probe.py rank2 in ten runs of one
# ratio and one fifth of the 65^3 = 274,625 code combinations each, every
# run under the 600 s cap of run.py, one process at a time, then the
# aggregate. Run from the repository root.
set -e
HERE=$(dirname "$0")
LOG="$HERE/results/rank2_loop.log"
for J in 1 -1; do
  for START in 0 54925 109850 164775 219700; do
    STOP=$((START + 54925))
    if [ "$STOP" -gt 274625 ]; then STOP=274625; fi
    python3 "$HERE/run.py" --max-seconds 600 --log "$LOG" -- "$HERE/probe.py" rank2 --j $J --start $START --stop $STOP --batch 8
  done
done
python3 "$HERE/run.py" --max-seconds 120 --log "$LOG" -- "$HERE/probe.py" rank2-aggregate
