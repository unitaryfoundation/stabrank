# Rank-5 exclusion at N m=4 and H3 m=4 by a two-qutrit all-visible base slice

Design and costing: `docs/notes/qutrit_m4_rank5_exclusion.md` (the run plan
is its section 7). The argument: every rank-5 decomposition of |M>^4
(M = N or H3) is carried by a monomial symmetry of |M> on qutrits 1, 2 to
one whose slice at x_0 = (0, 0) (N) or (1, 1) (H3) along those qutrits has
all five terms nonzero, so that slice is a full 5-cover of |M>^2 by
two-qutrit stabilizer states, and the two-qutrit slice structure lemma
fixes the eight other slices of every term up to Pauli classes and cube
roots. Every full 5-cover of |M>^2 is enumerated up to the unitary
symmetry group G_2 of |M>^2 (stage A: distinct independent states; stage
B: dependent; stage C: a repeated state) and matched at x_0 modulo 65521
with exact re-decision mod 2013265921 and in floating point. Everything is
Python; there is no compiled kernel.

## Files

- `cover_census.py`: the p = 3 enumerator (`CoverEnumerator3`, the port of
  `verify_challenge/slice_cover.CoverEnumerator`), `Field3` (Q(zeta_12)
  modulo a prime 1 mod 12), the phase-code patterns of the dictionary.
- `matcher.py`: the matcher (`Matcher.run(cover, x0, target)`), generic in
  the number of terms and in the base dimension: the coefficient family for
  dependent states and the block treatment of repeated states
  (`slice_cover.Family`, `solve_slice`, `slice_system` reused; `Block`,
  `reconstruct_block`, `_refine_split` and the flat types redone for p = 3),
  the term options (27 phased Pauli translates and absent), the composite
  shapes of the structure lemma (plane, coordinate line, point or diagonal
  line by the absence pattern), targets (`psi_target`, `vector_target`) and
  the exact re-decision of a hit. It replaces `slice_cover._affine_solve_C`
  by a truncated-SVD solve (module note), which the block reconstruction
  needs.
- `common.py`: the two cells (`ORBITS`, `X0`), paths per cell, canonical
  hashing, the partition, census and degenerate-list loaders with hash
  checks, `pairs_of`, `stage_of`, and `decide_terms` / `hit_record`.
- `driver.py`: `census`, `degenerate [--sample K --cap-s S] [--write]`,
  `partition`, `sample`, the controls `control-covers`, `control-planted`,
  `control-product`, `control-m3`, `control-witness`, and `export-witness`
  (the seven Lean terms of the rank-7 N witness to
  `N_m4_rank7_witness.json`).
- `partition_N.json`, `partition_H3.json`: the batch geometry per cell,
  hashed. N: 1,209 pivot pairs in 37 stage A batches, 12,175 dependent
  covers in 128 stage B batches, 5,910 repeated covers in 39 stage C
  batches (204 batches: 0 to 36 A, 37 to 164 B, 165 to 203 C). H3: 2,390
  pivot pairs in 28 stage A batches, 6,112 dependent covers in 57 stage B
  batches, 4,888 repeated covers in 8 stage C batches (93 batches: 0 to 27
  A, 28 to 84 B, 85 to 92 C). Targets of 540 s per batch at the laptop
  rates (census seconds plus 126 ms (N) or 100 ms (H3) per stage A cover;
  the sampled seconds per cover by multiplicity pattern for B and C, round
  robin over the sorted lists).
- `degenerate_covers_N.json`, `degenerate_covers_H3.json`: the stage B and
  C covers as sorted index tuples into `dictionary(3, 2)`, with the counts
  by multiplicity pattern and the file's hash.
- `results/N/`, `results/H3/`: `kernel_census.json` (per pivot pair the
  covers, candidates and seconds, plus the hashed lists of full 3-, 4- and
  5-covers), `degenerate_sample.json`, `sample.json`, the controls
  `control_*.json`, and `batch_K.json` for every batch run so far.
- `batch.py ORBIT K`, `aggregate.py ORBIT`: below.
- `batch_manifest_N.json`, `batch_manifest_H3.json` (written by the
  aggregate once every batch is present and clean; `.partial.json` before
  that): one entry per batch with parameters, output path and SHA-256, the
  file `certificate.attested.batches` names.
- `N-m4-lower-6.json.draft`, `H3-m4-lower-6.json.draft`: the attested-tier
  bound files with placeholders for compute hours, hardware and date; they
  move to `bounds/` only once the manifests exist, since the submissions
  workflow verifies every touched bound and `check_attested` fails on a
  missing manifest.
- `match_prototype.py`: the stage A prototype behind the note's
  measurements; not used by the pipeline.

The certificates are `verify_challenge/cert_n_m4_rank5_attested.py` and
`verify_challenge/cert_h3_m4_rank5_attested.py` (`CERTIFIED chi(N^4) >= 6`,
`CERTIFIED chi(H3^4) >= 6`, seed 20260922, two re-runs).

## Batch

`batch.py ORBIT K` runs batch K of `partition_ORBIT.json` and writes
`results/ORBIT/batch_K.json`; an existing file is skipped (`--force` redoes
it), so a loop over the indices resumes. A stage A batch runs the reference
5-cover kernel on each of its pivot pairs (every candidate re-decided mod
2013265921 and numerically) and the matcher on every cover found at the
cell's base point; a stage B or C batch loads its covers from the
degenerate list (hash checked against the partition). Every hit the matcher
returns is re-decided exactly (`common.decide_terms`); a hit with |M>^4 in
the span of its terms is a decomposition with at most five terms and the
batch exits 2 with `DECOMPOSITION FOUND` on its last line. A cover whose
run raises, or a hit on which the modular and numeric decisions disagree,
is recorded under `undecided` and fails the batch (exit 1). Exit 0
otherwise. The record's deterministic part (geometry, cell, base point,
partition and degenerate-list hashes, counts, the coordinate-slice solution
histogram, the hits as phase codes with their decisions, `undecided`) is
hashed as `deterministic_sha256`; timing, host, git commit and version
fields follow, then `sha256` over the whole record.

## Aggregate

`aggregate.py ORBIT` checks the partition's hash; that its stage A units
are exactly the enumerator's pivot pairs in order and its base point the
cell's; the stored degenerate list against the hash the partition names
and, unless `--no-reenumerate`, against a fresh enumeration (seconds); that
the stage B and C cover ids tile the list exactly once with the right
multiplicity; every batch file present with both hashes, geometry, hashes
and cell equal to the partition's, run counts consistent, no `undecided`
entry; every stored hit re-decided from its phase codes with no hit a
decomposition; and, once every stage A batch is present, the stage A cover
total equal to the census's (197,440 for N, 188,451 for H3). It writes the
manifest, then (unless `--dry-run`) re-runs `--recheck N` batches from
scratch (indices drawn from `numpy.random.default_rng(--recheck-seed)` over
the batches present, the seed printed on a `seed:` line) and compares
their `deterministic_sha256` with the stored ones. `--partial` reports over
the batches present while the run is in progress. Exit 0 when certified
(or, in a dry run, when every stored check passes), 2 when a stored batch
reports a decomposition, 1 otherwise.

## Launch, resume, aggregate

From the repository root after `uv sync --extra challenge`. One batch:

```
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py N K
```

All batches of one cell on a 16-vCPU pod, 15 at a time, one log per batch
(204 batches for N, 93 for H3):

```
mkdir -p research/qutrit_m4_rank5/results/N research/qutrit_m4_rank5/results/H3
seq 0 203 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py N "$0" \
   > research/qutrit_m4_rank5/results/N/batch_"$0".log 2>&1'
seq 0 92 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py H3 "$0" \
   > research/qutrit_m4_rank5/results/H3/batch_"$0".log 2>&1'
```

Rerunning the same command resumes (finished batches are skipped).
Progress and the final check, per cell:

```
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --recheck 2 --recheck-seed 20260922
```

The second command is what the certificate runs (through
`verify_challenge/cert_n_m4_rank5_attested.py`, with a scratch directory for
the re-runs and `--no-manifest`); it writes `batch_manifest_N.json`, whose
per-file hashes the bound's `attested` block points at. After the run,
fill the placeholders in `N-m4-lower-6.json.draft` (compute hours,
hardware, date) and move it to `bounds/N-m4-lower-6.json`; the same for
H3.

## Controls

Section 5 of the note lists them; `results/ORBIT/control_*.json` hold the
outcomes. Run with `driver.py control-covers N`, `control-planted N`,
`control-product N`, `control-m3 N`, `control-witness N` (and H3).
