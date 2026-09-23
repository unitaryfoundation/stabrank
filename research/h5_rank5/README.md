# Rank-5 exclusion of |H>^5 by a two-qubit base slice

Design and status: `docs/notes/h5_rank5_exclusion.md` (the feasibility
study is `docs/notes/next_exclusion_feasibility.md`). The pipeline mirrors
`research/h6_rank5/` and reuses its census and degenerate list by hash;
the new piece is stage (beta), `beta.py`, for a base point with four
visible terms and one invisible line term. Nothing here is a bound until
the batches have run on the pod and `aggregate.py` prints
`CERTIFIED chi(qubit_H^5) >= 6`.

## Files

- `common.py`: the cell (n_1 = 2, base points 00 and 01), the paths to the
  H^6 census (`research/h6_rank5/results/kernel_census.json`) and the
  degenerate list (`research/h6_rank5/degenerate_covers_v2.json`), the
  stage (beta) list `beta_covers.json`, canonical hashing, the loaders with
  their hash checks, and `decide_terms`, the exact re-decision of a
  candidate decomposition of psi_5 from its phase codes.
- `beta.py`: `BetaMatcher`, the stage (beta) matcher (module docstring for
  the equations, what it assumes, what it raises), and
  `PlantedBetaMatcher` for the planted control.
- `driver.py`: `beta-covers --write` (the 3,460 + 6 stage (beta) bases),
  `sample STAGE` (rates, to `results/rates.json`), `partition`,
  `control-witness`, `control-beta`, `control-m4-pair`, `tables`.
- `batch.py`: one batch by index (`--max-seconds`, `--resume`, exit 2 on
  `DECOMPOSITION FOUND`, exit 1 on any undecided run).
- `aggregate.py`: the certificate-side check and the batch manifest
  (`--dry-run --partial` during the run, `--recheck 2 --recheck-seed
  20260923` at the end).
- `partition.json`: the batch geometry, hashed, with the census, degenerate
  and beta list hashes.
- `results/`: the controls (`control_witness.json`, `control_beta.json`,
  `control_m4_pair.json`), the rate samples (`sample_*.json`,
  `rates.json`), the tables (`tables.json`), and `batch_<K>.json` for
  every batch run.
- `qubit_H-m5-lower-6.json.draft`: the bound file to fill in and move to
  `bounds/` after the run; `verify_challenge/cert_qubit_h_m5_rank5_attested.py`
  is its certificate.
- `prototype.py`: the measurements of the feasibility note (superseded by
  `driver.py sample`).

## Running

From the repository root after `uv sync --extra challenge` (compiles
`stabrank_core`). One batch:

```
nice -n 19 uv run --extra challenge python research/h5_rank5/batch.py K
```

A finished record is skipped on rerun; a record that a deadline left with
undecided covers is redone with `--resume`; `--force` redoes any record.
Progress and the final check:

```
nice -n 19 uv run --extra challenge python research/h5_rank5/aggregate.py --dry-run --partial
nice -n 19 uv run --extra challenge python research/h5_rank5/aggregate.py --recheck 2 --recheck-seed 20260923
```

The pod commands (15 processes through xargs, `setsid nohup`, resumable)
are in section 8 of the note.
