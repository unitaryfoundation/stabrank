# Rank-5 exclusion of |H>^6 by an all-visible base slice

Design and status: `docs/notes/h6_rank5_exclusion.md`. The argument is
complete and both positive controls pass on every base (section 5 of the
note); the 5-cover enumeration is complete (5,939,465 full covers of
distinct independent states, `results/kernel_census.json`, plus 26,242
dependent or repeated covers, `degenerate_covers.json`); the whole
exclusion is partitioned into 190 batches (`partition.json`) that
`batch.py` runs one at a time and `aggregate.py` checks, and the
certificate `verify_challenge/cert_qubit_h_m6_rank5_attested.py` behind the
draft bound `research/h6_rank5/qubit_H-m6-lower-6.json.draft` (attested tier) prints
`CERTIFIED chi(qubit_H^6) >= 6` once every batch is stored and clean. The
full run, about 36 CPU-hours, has not been launched; three test batches
(one per stage) are stored under `results/`.

## Files

- `common.py`: paths, canonical JSON hashing, the partition and
  degenerate-cover loaders with their hash checks, the pivot-pair units of
  stage A (`pairs_of`), and `decide_terms`, the exact re-decision of a
  candidate decomposition from its phase codes (psi_6 against the span of
  the terms mod 2013265921 and numerically).
- `driver.py`: setup and controls. `census` (the 5-cover kernel over every
  pivot pair, 459 s), `degenerate --write` (the 26,242 dependent or
  repeated full 5-covers over the 3-cover and 4-cover classes, 189 s,
  stored with their hash), `partition` (below), `control-m4` and
  `control-witness` (the two positive controls), `sample` and `fixture`
  (matcher timing and the C++ test fixture).
- `partition.json`: the batch geometry, hashed. Stage A: the 14,280 pivot
  pairs of the 5-cover enumeration grouped greedily into 15 batches of
  about 600 s (kernel seconds from the census plus 1.4 ms per cover for
  the compiled matcher). Stage B: the 12,390 dependent covers of five
  distinct states, round-robin over 158 batches of 78 or 79 covers (about
  700 s at the sampled 8.9 s per cover). Stage C: the 13,852 covers with a
  repeated state, round-robin over 17 batches of 814 or 815 covers (the
  sample mean of 0.82 s per cover is carried by rare 9 s covers; the test
  batch ran at 0.17 s per cover, so these batches take 2 to 4 minutes).
  Batch indices: 0 to 14 stage A, 15 to 172 stage B, 173 to 189 stage C.
- `degenerate_covers.json`: the stage B and C covers as sorted index
  tuples into `dictionary(2, 3)`, with the count by multiplicity pattern
  and the file's hash, which the partition and every stage B or C batch
  record.
- `degenerate_covers_v2.json`, `degenerate_covers_v2_delta.json`,
  `partition_stage_c_v2.json`: the stage C repair
  (`docs/notes/h6_rank5_stagec_repair.md`): the list regenerated with the
  cancel-at-base route of `degenerate_covers` (28,396 covers, the 2,154
  added ones in the delta file, stage B unchanged) and the partition of
  its 16,006 stage C covers into batches 190 to 196, which supersede
  batches 173 to 189. `batch.py K --partition
  research/h6_rank5/partition_stage_c_v2.json` runs one; `aggregate.py`
  picks the repair partition up when the file exists.
- `batch.py`: one batch by index (below).
- `aggregate.py`: the certificate-side check (below).
- `results/`: `batch_<K>.json` for every batch run so far; the controls
  `control_m4.json`, `control_witness.json`, `control_witness_repeated.json`;
  the timing samples `sample.json`, `sample_native.json`,
  `degenerate_sample.json`; and `kernel_census.json`.
- `batch_manifest.json` (written by `aggregate.py` once every batch is
  present and clean; `batch_manifest.partial.json` before that): one entry
  per batch with its parameters, output path and SHA-256, the file
  `certificate.attested.batches` names.

The machinery lives in `verify_challenge/slice_cover.py`
(`CoverEnumerator`, `Family`, `Block`, `solve_slice`, `SliceMatcher`), with
the two hot paths compiled in `cpp/src/cover5.cpp` (the 5-cover pair
kernel) and `cpp/src/slice_match.cpp` (stage A matching for distinct,
independent base states), bound in `python/bindings.cpp` and tested
against the Python reference in `tests/test_slice_cover.py`,
`cpp/tests/test_cover5.cpp` and `cpp/tests/test_slice_match.cpp`.

## Batch

`batch.py K` runs batch K of `partition.json` and writes
`results/batch_K.json`; an existing file is skipped (`--force` redoes it),
so a loop over the indices resumes. A stage A batch runs the compiled
5-cover kernel on each of its pivot pairs and the matcher on every cover
found, at the four base points x_0 (one per Hamming weight); a stage B or
C batch loads its covers from `degenerate_covers.json` (hash checked
against the partition) and runs the reference matcher, through the
coefficient family for dependent states and the block treatment for
repeated ones. Every hit the matcher returns is re-decided exactly
(`common.decide_terms`); a hit with psi_6 in the span of its terms is a
decomposition with at most five terms, and the batch exits 2 with
`DECOMPOSITION FOUND` on its last line. A (cover, x_0) run that raises, or
a hit on which the modular and numeric decisions disagree, is recorded
under `undecided` and fails the batch (exit 1). Exit 0 otherwise.

`--no-native`, or `STABRANK_NO_NATIVE=1` in the environment, keeps the
5-cover kernel and the stage A matcher in Python (about 80 times slower on
stage A; stages B and C run in Python either way).

The record: `batch` (index, stage, the pivot pairs or cover ids), the cell
(`orbit`, `m`, `rank`, `n1`, `N`, `group_order`), the partition and
degenerate-list hashes, `covers`, `matched`, `refused`, `native_runs`,
`coord_solution_hist` (runs by their coordinate-slice solution counts),
`hits` (base cover and point, the terms as phase codes, the exact
decision), `hit_count`, `decompositions`, `undecided`; then
`deterministic_sha256` over everything above, which a re-run on any machine
or with `--no-native` must reproduce; then `candidates` (the kernel's
modular candidates), timing (`kernel_s`, `match_s`, `wall_s`, `cpu_s`,
timestamps), `hostname`, `git_commit`, `native`, `matcher`, `cover5`,
version strings and `hit_numerics` (residual and coefficients per hit);
then `sha256` over the whole record.

## Aggregate

`aggregate.py` checks the partition's hash; that its stage A units are
exactly the enumerator's pivot pairs in order; the stored degenerate list
against the hash the partition names and, unless `--no-reenumerate`,
against a fresh enumeration (about 190 s); that the stage B and C cover ids
tile the list exactly once with the right multiplicity; every batch file
present with both hashes, geometry, partition and degenerate hashes and
cell equal to the partition's, run counts consistent, no `undecided`
entry; every stored hit re-decided from its phase codes with no hit a
decomposition; and, once every stage A batch is present, the stage A cover
total equal to the census's 5,939,465. It writes the manifest, then (unless
`--dry-run`) re-runs `--recheck N` batches from scratch (indices drawn
from `numpy.random.default_rng(--recheck-seed)` over the batches present,
the seed printed on a `seed:` line) and compares their
`deterministic_sha256` with the stored ones. `--partial` reports over the
batches present while the run is in progress. Exit 0 when certified (or,
in a dry run, when every stored check passes), 2 when a stored batch
reports a decomposition, 1 otherwise.

## Launch, resume, aggregate

From the repository root after `uv sync --extra challenge` (which compiles
`stabrank_core` with the two kernels; `uv run --extra challenge python -c
"import stabrank.stabrank_core as c; print(c.cover5_pair, c.SliceMatchKernel)"`
confirms they are present). One batch:

```
nice -n 19 uv run --extra challenge python research/h6_rank5/batch.py K
```

All 190 batches on a 16-vCPU pod, 15 at a time, one log per batch:

```
mkdir -p research/h6_rank5/results
seq 0 189 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/h6_rank5/batch.py "$0" \
   > research/h6_rank5/results/batch_"$0".log 2>&1'
```

Rerunning the same command resumes (finished batches are skipped). The
stage B batches (about 700 s each, 158 of them) dominate: about 36
CPU-hours in all, about 2.5 hours of wall time on 15 cores. Progress and
the final check:

```
nice -n 19 uv run --extra challenge python research/h6_rank5/aggregate.py --dry-run --partial
nice -n 19 uv run --extra challenge python research/h6_rank5/aggregate.py --recheck 2 --recheck-seed 20260921
```

The second command is what the certificate runs (through
`verify_challenge/cert_qubit_h_m6_rank5_attested.py`, with a scratch
directory for the re-runs and `--no-manifest`); it writes
`batch_manifest.json`, whose per-file hashes the bound's `attested` block
points at. After the run, fill the placeholders in
`research/h6_rank5/qubit_H-m6-lower-6.json.draft` (compute hours, hardware, date, the
`--no-native` cross-check) and rename it to `bounds/qubit_H-m6-lower-6.json`;
the submissions workflow verifies every touched bound, so the draft keeps
its suffix until the manifest exists.

## What is validated

- Partition: `aggregate.py` rebuilds the enumerator's pivot pairs and the
  degenerate cover list (26,242 in 185 s) and finds both equal to the
  stored ones.
- Test batches, one core at nice 19 on an 18-core laptop at load average
  about 10: stage A batch 14 (7,795 pivot pairs, the smallest stage A
  batch), 134,822 covers, 539,288 matched runs, 0 hits, 220 s (kernel 120
  s, match 99 s; the partition estimated 301 s); stage B batch 15, 79
  covers, 316 matched runs, 0 hits, 663 s (8.4 s per cover; estimate 700
  s); stage C batch 173, 815 covers, 3,260 matched runs, 0 hits, 135 s
  (0.17 s per cover; estimate 700 s). No run refused, no hit, nothing
  undecided.
- Aggregate under `--partial`: every stored check passes on the three
  batches, the partial manifest is written, and the seeded re-run of batch
  14 from scratch reproduces its deterministic hash.
- Controls and kernels: section 5 of the note and `tests/test_slice_cover.py`;
  nothing in the batch pipeline changes the matcher.
