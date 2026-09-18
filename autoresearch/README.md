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
