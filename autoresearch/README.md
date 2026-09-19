# Autoresearch loop

One iteration of the search-verify-record loop, with the record made
unavoidable.

```
uv run --extra challenge python autoresearch/run.py qubit_H 4 4 --seeds 2
uv run --extra challenge python autoresearch/summary.py
```

`run.py ORBIT M RANK` anneals at a fixed rank for a number of seeds. Every
run, successful or not, appends one line to `runs.jsonl` with its
configuration, seed, wall-clock, process CPU time and final residual. A run
that reaches the rank is converted to the witness format by
`verify_challenge/to_witness.py`, refit in exact arithmetic, and, if the
refit succeeds, written as a submission under `bounds/` whose
`provenance.compute` is the sum over every run logged for that cell, failed
ones included. A decomposition that does not refit exactly is logged as a
near miss and not submitted. Pass `--llm MODEL` when a model chose the cell
or the configuration, so the ledger records it.

`summary.py` joins the run log with `docs/ledger.json` and prints, per cell,
runs, CPU-hours, exact solutions and CPU-hours per solution. That last
column is what a scaling or cost-trend figure is drawn from.

The log is committed. It is the raw data behind any claim about what a
discovery cost, so a line is never edited or removed; a wrong run is
recorded as a wrong run.

## The unattended loop

`loop.py` runs `run.py` for days from a manifest, one job at a time.

```
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/example.json
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/plateau_cells.json --max-hours 72
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/plateau_cells.json --resume
uv run --extra challenge python autoresearch/loop.py replay autoresearch/state/plateau_cells.json
uv run --extra challenge python autoresearch/loop.py status --since 72
```

A manifest is a JSON file with a `jobs` list and an optional `defaults`
block. Each job names a cell (`orbit`, `m`, `rank`, where `orbit` may be
`T3sector<s>` exactly as `run.py` accepts it), a seed range (`seed0`,
`seeds`), the annealer settings (`chains`, `iters`, `cooling`), a
`priority` (larger runs first within a round, default 0), a wall-clock cap
per job in seconds (`cap_s`, default four hours) and optionally
`seeds_per_round`, `llm`, `warm_from`, `save_plateaus`, `extra_args` and
`stop_on_solve` (default true: a cell whose rank is reached gets no more
seeds). `manifests/example.json` finishes in seconds and is what the tests
and a first smoke run use; `manifests/plateau_cells.json` is the four open
cells with seed ranges that do not overlap the runs already in the log.

Every job is one seed of one cell: `nice -n 19 python run.py ORBIT M RANK
--seeds 1 --seed0 SEED ...` with `OMP_NUM_THREADS=1`, so the loop never holds
more than one core on a shared machine. The order is round-robin: the first
seed of every cell in priority order, then the second seed of every cell,
and so on, so no cell waits for another to exhaust its seeds. The order is a
fixed function of the manifest, which is what makes a resumed loop continue
exactly where a fresh one would have been.

After every job the loop writes `state/<manifest name>.json`: the manifest's
jobs as run, which seeds of each cell are done with their outcomes, and the
full sequence of jobs with timestamps. `run --resume` continues from it,
`replay STATE` re-runs the recorded sequence (from the snapshot inside the
state file, so a later edit of the manifest does not change what is
replayed) into `state/<name>-replay.json`, and `--dry-run` prints the
schedule of either without running anything.

Each job also appends one JSON line to `loop.log` with start and end time in
UTC, wall clock, the child's CPU time from `getrusage`, exit code, the last
eight lines of stderr, the residual parsed from stdout and an outcome read
from the exit code and stderr: `discovery` (exit 0, submission written and
verified, or a sector decomposition saved), `miss` (exit 2), `near_miss`
(reached the rank numerically, exact refit failed), `verifier_refusal` (a
submission was written and the verifier rejected it), `timeout` (the job's
cap), `oom`, `exception:<Type>`, `infrastructure:<Type>` (import, spawn and
OS errors) and `killed:<SIG>`. Infrastructure failures, out-of-memory kills
and unknown exit codes are retried once; a miss, a near miss, a timeout and
an exception inside the search never are. `loop.log` also carries a `start`
and a `stop` line per session with the stop reason.

`--max-hours H` stops the loop before the first job whose cap would not fit
in the time left, so a job is never killed to meet the limit. SIGINT or
SIGTERM terminates the running child, records it as `interrupted` (not
done, so `--resume` runs that seed again), checkpoints and exits with code
130. `status [--since HOURS]` summarises the log: jobs and sessions,
outcomes and failures by class, discoveries, wall and CPU hours, and an
`operability` line with the hours covered from first start to last end, the
fraction of that time a job was running and the longest gap between
consecutive jobs. That line is the evidence for the 72-hour autonomous-loop
metric.

`run.py` still appends every annealing run to `runs.jsonl`, so the cost
ledger is unchanged; the loop only adds the scheduling and diagnosis layer
on top of it.
