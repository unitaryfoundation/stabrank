# Rank-7 exclusion for |T3>^3: batch pipeline

Design note: `docs/notes/t3_rank7_exclusion.md`. The full scan ran on
2026-09-19 and 2026-09-20 (four `loop.py` instances over
`manifest_rank7_p0.json` to `manifest_rank7_p3.json`, disjoint quarters of
every block); all 459 batch outputs are stored under `batches/` and listed
with their hashes in `batch_manifest.json`. `aggregate.py` prints
`CERTIFIED chi(T3^3) >= 8` after checking them and re-running a seeded
subset from scratch; `verify_challenge/cert_t3m3_rank8_attested.py` is the
certificate behind `bounds/T3-m3-lower-8.json` (attested tier). The result:
1,213,458,815 candidate class sets, none containing V_3, none spurious or
undecided, 80.1 CPU-hours.

## Files

- `common.py`: shared setup. Dictionary, descent to V_m, reduction mod
  65521, the projection of the quotient to F_ell^6 (seed 2024), the
  symmetry group of V_m and the dictionary permutations it induces, and the
  orbit-block relabelling of the note: states inside V_m first, then every
  G-orbit as a contiguous block in increasing order of size. The first pivot
  of a block is its first label; the admissible second pivots j are the
  labels above it minimal in their Stab(i)-orbit; the admissible third
  pivots k > j are minimal in their Stab(i, j)-orbit. Also the partition
  loader and the canonical-JSON hash used everywhere.
- `make_partition.py`: writes `partition.json` (the fixed batch geometry:
  for every batch its block, j-range, pair count and exact inner-step count)
  and `manifest_rank7.json` for `autoresearch/loop.py`. The inner-step count
  of a pair (i, j) is the sum over admissible k of N - k - 1, the quantity
  `count_steps.py` reports; the sum over batches is 13,125,302,318,795, the
  `blocked_steps_with_stab_ij` total of `results/step_counts.json`, and the
  per-block totals match `count_steps.py` block by block. 459 batches of
  about 3e10 steps (median 3.0e10, smallest 8.3e8, largest 3.04e10; block 0
  is a single batch of 1.59e10).
- `batch.py`: one batch. Accepts the `run.py` command line the loop builds,
  with `--seed0` as the batch index and the annealer flags ignored. Runs the
  scan for the batch's (block, j-range) in the compiled kernel `scan_pairs`
  and writes `batches/batch_<K>.json`.
- `aggregate.py`: the certificate-side check (below).
- `check_order.py`: combinatorial check of the pivot order: random 7-sets
  have a canonical form the masks admit.
- `scan3.py`, `count_steps.py`, `results/`: the prototype and the counts the
  note is based on, unchanged. `scan3.py`'s `kernel3` is the reference the
  batch kernel is compared against.
- `rare_pivot_m2.py`: the finite checks behind
  `docs/notes/t3_rank7_rare_pivot.md` (the rare-type pivot lemma cannot be
  obtained from the slice structure): restriction of V_3 to every
  coordinate slice is V_2, the two-qutrit spanning configurations of V_2 of
  sizes 3, 4, 5 and 7 named in that note, and the lift table from
  two-qutrit orbits to three-qutrit orbits. Exact arithmetic throughout.
  Writes `results/rare_pivot_m2.json`; about one minute, run with
  `uv run --extra challenge python research/t3_rank7/rare_pivot_m2.py`.
- `batches/`: the stored outputs of the m=3 batches run so far (three
  validation batches at present); `batches_m2/`: the m=2 control outputs.
- `partition_m2.json`, `manifest_m2.json`: the m=2 miniature (7 batches,
  3.9e6 steps) used as a control.

## Kernel

`scan_pairs` in `batch.py` is `kernel3` of `scan3.py` with three changes.
The per-j third-pivot mask (`kok_index`, `kok_rows`) implements the Stab(i,
j) reduction; the nominal step count is accumulated for every admissible k
whether or not the kernel skips it as a member of span(V_m, s_i, s_j), so
it equals the partition's count exactly; and the class lists are replaced by
an in-place exact decision. For every candidate class set the kernel gathers
the members (three pivots, the states above j inside span(V_m, s_i, s_j),
the states above k with zero image modulo span(q_i, q_j, q_k), one group of
mutually parallel images there, and the states inside V_m), reduces their
rows to echelon form mod 2^31 - 1 and, when the rank is at most 7, reduces
the three descent vectors psi_r against that row space. A rank of at most 8
mod 2^31 - 1 is the rank over Q(w3) (Hadamard, as in the certificate), and
rank(class with V_m) = rank(class) is the same statement as "the psi_r
reduce to zero", so the decision is exact. A class of rank at least 8 cannot
lie in the 7-dimensional preimage of its image space and is a projection
collision; the kernel stores it and `batch.py` re-splits it in Python
against the unprojected quotient images modulo the three pivots, deciding
each subclass by the rank pair, or by its 7-subsets if a subclass still has
rank at least 8. Anything left is an `undecided` entry, which fails the
batch. Only counts, a histogram by (zero members, parallel members) and the
exceptions are stored.

The kernel reproduces `kernel3` on the five pairs of
`results/m3_pair_timings.json` (same candidate counts, histograms and inner
steps, in the original labelling with trivial masks).

`cpp/src/t3_scan.cpp` is a C++ port of the same kernel (same hash, same
table, fused reduction and grouping, compile-time modulus, one array of
structs for the table), bound as `stabrank_core.t3_scan_pairs` with the
argument list and output buffers of the numba kernel, and tested in
`cpp/tests/test_t3_scan.cpp` (planted rank-7 configuration found and
decided, generic target in no class span, third-pivot mask honoured,
determinism). `batch.py --engine cpp` selects it. Its records are identical
to the numba kernel's (same `deterministic_sha256` on m=3 batch 15 and m=2
batch 4, same counts on the five recorded pairs) but it is not faster: 18
to 21 ns per step against 15 to 21 for numba back to back on the same load,
so numba is the default. The note's 10 ns estimate assumed 32-bit
coordinates and a direct-indexed count array in place of the hash table,
which this port does not do.

## Batch output

`batches/batch_<K>.json`: the geometry (`batch`: index, block, j_lo, j_hi,
pairs, steps, first pivot as an original index, orbit size), m, rank, need,
the partition, setup and dictionary hashes, `steps_nominal` and
`steps_done`, pairs scanned and pairs with a nontrivial Stab(i, j), Z-member
and parallel-to-pivot totals, `candidates`, `decided`, `histogram`,
`found_count` and `found` (class sets whose span contains V_m, as original
indices with their rank and the smallest spanning subset, up to 256
stored), `spurious_count` and `spurious` (re-split records), `oversize_count`,
`undecided`; then `deterministic_sha256` over everything above; then wall,
CPU, setup and kernel times, ns per step, timestamps, hostname, git commit,
kernel version string, interpreter versions; then `sha256` over everything
above it.

Exit codes: 0 when the batch completes and no class set contains V_m
(`loop.py` logs this as `discovery`, its word for exit 0); 2 when one does
(`loop.py` logs `miss`; the last stdout line begins `DECOMPOSITION FOUND`
and names the class); a traceback (exit 1) when a class set is undecided,
the geometry or step count disagrees with the partition, or the setup hash
differs from the partition's. The loop retries only infrastructure
failures, so a batch that fails on its own content is logged once as
`exception:<Type>`.

## Launch, resume, aggregate

From the repository root, one core, `nice -n 19` applied by the loop:

```
uv run --extra challenge python autoresearch/loop.py run research/t3_rank7/manifest_rank7.json \
    --runner research/t3_rank7/batch.py --max-hours 60
uv run --extra challenge python autoresearch/loop.py run research/t3_rank7/manifest_rank7.json \
    --runner research/t3_rank7/batch.py --resume --max-hours 60
uv run --extra challenge python autoresearch/loop.py status --since 72
uv run --extra challenge python research/t3_rank7/aggregate.py --dry-run --partial
```

The manifest has one job per block (45 jobs), the block's batch indices as
its seeds in a single round, priority decreasing with the block index, so
the batches run block 0 first, then block 1, and so on; `stop_on_solve` is
false and `cap_s` is 10800 (three hours; the largest batch is 3.04e10
steps, about 9 minutes at the measured 18 ns per step and 23 at the note's
45 ns, so a timeout means the machine was starved, and `--resume` runs that
batch again). `extra_args` carries the partition
path and the output directory. The loop's state file
`autoresearch/state/manifest_rank7.json` records which batches are done;
`--resume` continues from it, and a batch that was interrupted is run again.
The batch files themselves are the audit trail and can be regenerated one at
a time with `batch.py T3 3 7 --seeds 1 --seed0 K --partition
research/t3_rank7/partition.json`.

Several loops can run at once on different machines by giving each a
manifest that covers a subset of the jobs, provided they write to different
state files (the manifest `name`) and the batch files are collected into one
directory before aggregating.

`aggregate.py` checks the partition's hash, that every batch file is
present, both hashes of every file, that the geometry, partition hash, setup
hash, cell and nominal step count match the partition, that the j-ranges of
every block tile [start + 1, N), that the step counts sum to the partition
total, that every `undecided` list is empty and every `found` list empty.
Then it re-runs `--recheck N` batches from scratch (symmetry cache bypassed,
indices drawn from `numpy.random.default_rng(--recheck-seed)`, seed and
indices printed) into a scratch directory and compares their
`deterministic_sha256` with the stored ones. `--dry-run` skips the re-runs
and never certifies; `--partial` reports over the batches present (and
re-runs among them) while the scan is in progress. Exit 0 when certified
(or, in a dry run, when every stored check passes), 2 when a stored batch
reports a decomposition, 1 otherwise.

## What is validated

- Partition: the sum of the batch step counts equals `count_steps.py`'s
  total for the orbit-block order with the Stab(i, j) mask,
  13,125,302,318,795, and every block's total matches `per_rep_blocked`
  in `results/step_counts.json`. Every batch asserts that its kernel's
  nominal count equals its partition entry.
- Kernel against the prototype: `scan_pairs` and `kernel3` agree on the five
  recorded m=3 pairs (0 to 4602 candidate classes each).
- Pivot order: `check_order.py` finds an admissible canonical form for 3000
  random 7-sets at m=2 and 2000 at m=3.
- End to end at m=3: batches 15, 60 and 69 (the three smallest, 8.3e8 to
  1.4e9 steps, 2435 to 5494 candidate classes) run clean through `batch.py`,
  aggregate under `--dry-run --partial`, and their from-scratch re-runs
  match bit for bit under `--partial --recheck`.
- Control at m=2 through the same code path: `partition_m2.json` and
  `manifest_m2.json`, run by `loop.py` with `--runner
  research/t3_rank7/batch.py`, then `aggregate.py --partition
  research/t3_rank7/partition_m2.json --batches-dir
  research/t3_rank7/batches_m2`. Every one of the 1,930,049 class sets
  contains V_2 (the three states inside V_2 are members of every class),
  the aggregator reports the smallest spanning subset as 3 states, so
  chi(T3^2) <= 3, and dim V_2 = 3 gives equality. This exercises the found
  path; nothing at m=2 exercises a negative outcome.
- Timing: 15 to 21 ns per inner step for the numba kernel at load average
  about 5 on 18 cores (the note measured 42 to 45 ns at load 27 to 120);
  30 s of setup per batch with the symmetry cache, 60 s without. Projected
  total at 18 ns: 66 CPU-hours for the 1.31e13 steps, plus 4 hours of
  setup over 459 batches, about 9 minutes per median batch; at the note's
  45 ns, 164 CPU-hours. The decision cost is negligible at the observed
  candidate rate (about 3e-6 per step).

## What is not validated

- The full scan has not been run; the three m=3 batches are 0.02 percent of
  the steps and their j-ranges are the cheap tails of their blocks, so the
  candidate flood at small j (the note's 1e-5 classes per step) has been
  seen only in `kernel3`'s pair timings, not in a stored batch.
- No spurious class (rank at least 8 mod 2^31 - 1) has occurred, so the
  re-split path of `batch.py` has not run on real data; it is exercised
  only by reading.
- The negative (`CERTIFIED`) branch of `aggregate.py` has not been reached
  by any run, since m=2 always finds V_2 and the m=3 set is partial.
- Case B of the note has no separate script; the three-pivot scan covers
  it.
- The C++ kernel has been compared with the numba kernel on two batches
  and five pairs, not on a batch with a spurious class or an oversize
  class, where the two record paths could differ in ordering.
