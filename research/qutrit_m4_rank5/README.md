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
  needs. A base with no ordinary term (every distinct state repeated: the
  m = 3 control and the rank-8 H3 witness bases) goes to
  `BlockOnlyMatcher` instead of the family path: every slice is solved on
  its own for the translate selections whose span contains it
  (`block_only_slice`, dependent selections included with their
  coordinate family; a slice proportional to an earlier one reuses its
  selections), the two points of each of the four lines through x0 are
  paired by the class map k -> 2k, each independent paired selection
  keeps the block configurations that are consistent in the family
  parameter (`block_line_configs`, `_theta_combos`), three lines at a
  time are joined (the two with the fewest surviving selections drive,
  the third is looked up through the class relation of the structure
  lemma, `block_relation`, and a per-block lambda prefilter), and the
  fourth line is completed (`_complete_line`: plane copies from the
  completion table, line copies absent, copies absent on all three lines
  read off the residual) before the exact confirmation. It finds every
  decomposition whose selection is independent on at least three of the
  four lines; a decomposition dependent on two or more lines is outside
  it (`stats["dependent_lines_limit"]`).
- `common.py`: the two cells (`ORBITS`, `X0`), paths per cell, canonical
  hashing, the partition, census and degenerate-list loaders with hash
  checks, `pairs_of`, `stage_of`, and `decide_terms` / `hit_record`.
- `driver.py`: `census`, `degenerate [--sample K --cap-s S] [--write]`,
  `partition`, `sample`, the controls `control-covers`, `control-planted`,
  `control-product`, `control-m3`, `control-witness` (one base per process,
  below), and `export-witness` (the seven Lean terms of the rank-7 N
  witness to `N_m4_rank7_witness.json`, each recognised as a stabilizer
  state by `stabilizer_spec`, affine support and quadratic phase, and
  stored with its (k, x0, W, Q, l) spec; the earlier version tested
  membership in `dictionary(3, 4)`, 7,439,040 states of dimension 81, whose
  construction with its phase patterns and field images takes tens of GB).
- `N_m4_rank7_witness.json`: the exported witness (phase codes, specs,
  coefficients).
- `results/controls/witness_ORBIT_base_K.json`: one record per
  `control-witness` base: the base, its counts (distinct states,
  multiplicities, kappa), the caps, the outcome (`pass`, `fail`,
  `aborted`) with its reason, seconds, peak RSS and the matcher's stats
  (partial after an abort); `results/ORBIT/control_witness.json` is their
  collection (`--summary`).
- `partition_N.json`, `partition_H3.json`: the batch geometry per cell,
  hashed (note, section 8). N: 1,209 pivot pairs in 17 stage A batches
  (0 to 16), 12,175 dependent covers in 98 stage B batches (17 to 114,
  124 or 125 covers), 16,181 repeated covers in 56 stage C batches: 29
  shared batches (115 to 143, 554 to 562 covers) and 27 single-cover
  batches (144 to 170) for the (2, 2, 1) covers whose two blocks admit
  dependent translates, the stage C tail; 171 batches. H3: 2,390 pivot
  pairs in 9 stage A batches (0 to 8), 6,112 dependent covers in 44 stage
  B batches (9 to 52, 138 or 139 covers), 7,024 repeated covers in 4
  stage C batches (53 to 56, 1,748 to 1,764 covers); 57 batches. Targets
  of 600 s per batch at the pod rates (`--match-ms 54` and `21`; `--rates`
  by cover class: stage B 4.8 and 4.3 s, (2, 1, 1, 1) 0.8 and 0.3 s, 25
  and 6 s with kappa >= 1, (3, 1, 1) 7 and 0.15 s, (2, 2, 1) 10 and 5 s,
  dependent (2, 2, 1) a nominal 3,600 s). Stage B is round robin over the
  sorted list; stage C is dealt longest first onto the least loaded batch
  (`driver.cover_class` gives the class), a cover at or above the target
  alone in its batch, the cheap covers of a batch first.
- `degenerate_covers_N.json`, `degenerate_covers_H3.json`: the stage B and
  C covers as sorted index tuples into `dictionary(3, 2)`, with the counts
  by multiplicity pattern and the file's hash; N 28,356 covers (12,175
  stage B, 16,181 stage C), H3 13,136 (6,112, 7,024), including the
  cancel-at-base multisets T + (b, b) with b outside span(T).
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
- `tests/test_qutrit_m4_rank5.py` (repository `tests/`): the strict block
  coordinate solve (dependent translates of two blocks, a residual with
  no translate, a residual outside the span raise `UnpinnedFamily`;
  independent translates return the coordinates), the batch record of a
  raising run and of a deadline abort, `cover_class` on the 27 dependent
  (2, 2, 1) N covers, and the cancel-at-base multisets in the N list.

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
run raises (`UnpinnedFamily` from the block reconstruction included), or
a hit on which the modular and numeric decisions disagree, is recorded
under `undecided` and fails the batch (exit 1). Exit 0 otherwise.
`--max-seconds S` guards the batch against the stage C tail: the running
cover is aborted at the deadline (`BudgetExceeded`, recorded as
undecided) and every cover not yet started is recorded as undecided with
the reason "not run"; the batch prints `ABORTED at the --max-seconds`,
exits 1, and is re-run with `--force` and a larger cap or none. The
record's deterministic part (geometry, cell, base point,
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

The first batch of each stage first (N 0, 17, 115 and H3 0, 9, 53, one
process each), then the partial aggregates, to check the rates:

```
mkdir -p research/qutrit_m4_rank5/results/N research/qutrit_m4_rank5/results/H3
printf 'N 0\nN 17\nN 115\nH3 0\nH3 9\nH3 53\n' | xargs -P 6 -L 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py "$0" "$1" --max-seconds 3600 \
   > research/qutrit_m4_rank5/results/"$0"/batch_"$1".log 2>&1'
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --dry-run --partial
```

All batches on a 16-vCPU pod, 15 at a time, one log per batch: the
shared batches under a 3,600 s guard, the 27 single-cover N batches (the
dependent (2, 2, 1) covers, the stage C tail) without one:

```
seq 0 143 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py N "$0" --max-seconds 3600 \
   > research/qutrit_m4_rank5/results/N/batch_"$0".log 2>&1'
seq 0 56 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py H3 "$0" --max-seconds 3600 \
   > research/qutrit_m4_rank5/results/H3/batch_"$0".log 2>&1'
seq 144 170 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py N "$0" \
   > research/qutrit_m4_rank5/results/N/batch_"$0".log 2>&1'
```

Rerunning the same command resumes (finished batches are skipped). A
batch that hit the guard is listed by `grep -l ABORTED
research/qutrit_m4_rank5/results/*/batch_*.log` and re-run with `--force`
and no `--max-seconds`. Projected: N 24 CPU-hours for the shared batches
plus the 27 single-cover batches (a nominal hour each; one such cover
took two hours on the pod), H3 9.3 CPU-hours. Progress and the final
check, per cell (the seeded recheck re-runs N batches 121 and 169 and H3
batches 39 and 56 once every batch is present):

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
`control-product N`, `control-m3 N` (and H3). Status after the review
fixes and the two matcher changes of 2026-09-23 (the stale
second-coordinate-slice index keyed on `id(sols)`, which had cost the
product control 8 of its 270 hits, and the single-block reconstruction
with the coefficient family's parameter as an unknown; note, section 8):
`control-planted` recovers all 32 planted decompositions per cell with 0
candidates rejected, `control-product` 270 of 270 (N) and 81 of 81 (H3),
`control-m3 N --sample 200` 9 genuine hits in 201 runs, 0 refused, the
stored class recovered (63 s). `control-m3 H3` is still the unfinished
item of section 7.

### control-witness

The rank-7 N witness (the Lean terms) and the rank-8 H3 witness
(`bounds/H3-m4-upper-8.json`) recovered from their all-visible (qutrit
pair, base point) slices with the matcher at rank 7 or 8. N has 6 bases,
H3 has 3 (`--list-bases` prints them with their distinct-state count,
multiplicities and kappa). Each base runs in its own process and writes
`results/controls/witness_ORBIT_base_K.json`; a base passes when the
witness is recovered exactly once as a genuine rank-7 (rank-8)
decomposition. The caps `--max-rss-gb` (default 8; also refuses a dense
solve whose feature matrices would exceed it, before allocating),
`--max-states` (200,000 joined states), `--max-solutions` (2,000,000 per
slice equation, the `_dense` candidate cap included) and `--max-seconds`
abort the base with the reason recorded instead of letting the process be
killed; 0 disables a cap.

```
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness N --list-bases
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness N --base 3 --max-rss-gb 100 --max-states 0 --max-seconds 0
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness N --summary
```

Status (2026-09-22, one core at nice 19 on the loaded laptop, 10-minute
cap per base; section 7 of the note has the diagnosis; the H3 entries
below describe the family path, replaced for block-only bases on
2026-09-23, see the next paragraph): no base has passed yet on the
laptop. N base 0 (seven distinct states, kappa 3) is refused at the
first coordinate slice after 25 s, 2,084,934 hash candidates from the
3-parameter dense solve; bases 1, 2 and 5 have the same shape and were
not run. N base 3 (three repeated pairs and one ordinary state, kappa 1):
80,759 solutions per coordinate slice (42 s and 28 s), 6,785 states after
the first, the join of the second still running at the deadline after
3.9 million pairs (1.2 million dropped by a dead coefficient, 2.7 million
by the split tracking), peak RSS 1.5 GB. N base 4 (one repeated pair,
kappa 2): 87,766 solutions per slice (160 s and 171 s), 3,689 states
after the first, more than 200,338 joined states after the second,
peak RSS 0.8 GB. H3 base 0 (four repeated pairs, no ordinary state,
kappa 1): 458,564 pinned states per coordinate slice (274 s and 123 s,
from 54,260 translate-set solutions), 5,420 after the first slice's
split tracking, more than 458,564 after the join, peak RSS 6.3 GB. H3
bases 1 (eight distinct states, kappa 4) and 2 (four repeated pairs) were
not run. On the pod (2026-09-22, one base per process, `--max-rss-gb 12`)
N base 4 PASSED: the rank-7 witness recovered exactly once, 67 other
genuine rank-7 decompositions, 4,489 s, 1.1 GB; the five kappa >= 3
bases aborted at the candidate cap; H3 base 2 reached the join of the
second coordinate slice (11,565,760 solutions, 1,958,256 states, 1,346 s,
8.8 GB) and was killed in the composite stage.

H3 block-only bases (2026-09-23, `BlockOnlyMatcher`, base 2 on the
laptop under the 10-minute, 8 GB cap): the family path could not have
recovered the witness there at all (its own selection at x0 + e_2 and
x0 + 2 e_2 is the four base states, of rank 3, which the block-only solve
rejected as dependent, and no block ever pins the family since the two
copies t_i (x) |0>, t_i (x) |+> never occupy two classes), so the memory
blow-up was not the obstacle. With the new path: the eight slices of
|H3>^4 along a qutrit pair are all proportional to |H3>^2, so one solve
(60,941 selections, 6,681 dependent, 17 to 19 s, 0.5 GB) serves all
eight; per line 54,260 independent paired selections, of which 4,100
(e_1), 4,100 (e_2), 100 ((1, 1)) and 100 ((1, 2)) admit a
lambda-consistent configuration (36 to 42 s per line); the passes over
three lines: [e_2, (1, 1), (1, 2)] and [e_1, (1, 1), (1, 2)] driven by
the two diagonal lines, 10,000 pairs each, 2,736 lambda-feasible, 1,392
triples, 156 s each; [e_1, e_2, (1, 2)] driven by e_1 and (1, 2), 410,000
pairs, 19,664 feasible, 7,256 triples, 56 s; the fourth pass
[e_1, e_2, (1, 1)] started at 369 s and had not finished at the 560 s
deadline (record: `aborted`, peak RSS 0.60 GB, 5,184 distinct genuine
rank-8 decompositions confirmed by then). The witness's own triple
(lines e_1, (1, 1), (1, 2), the dependent e_2 line completed) gives 433
block leaves, 338 combinations, 2,592 completions and 1,296 distinct
genuine rank-8 decompositions with the witness exactly once, in 61 s
(the base admits one rank-2 stabilizer decomposition of |H3> per block
independently). The whole run needs about 750 s on the loaded laptop,
so the pass criterion is not yet met under the local cap; an aborted run
now records the hits found before the abort. To finish the H3
block-only bases on the pod (bases 0 and 2; base 1, kappa 4 with eight
ordinary states, stays refused at the dense-solve cap):

```
mkdir -p research/qutrit_m4_rank5/results/controls
for k in 0 2; do
  nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness H3 --base $k \
    --max-rss-gb 12 --max-states 0 --max-solutions 0 --max-seconds 0 --verbose \
    > research/qutrit_m4_rank5/results/controls/witness_H3_base_$k.log 2>&1
done
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness H3 --summary
```

The earlier all-bases command, one base per process, 15 in parallel:

```
mkdir -p research/qutrit_m4_rank5/results/controls
{ seq 0 5 | sed 's/^/N /'; seq 0 2 | sed 's/^/H3 /'; } | xargs -P 15 -L 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness "$0" --base "$1" \
   --max-rss-gb 100 --max-states 20000000 --max-solutions 20000000 --max-seconds 0 --verbose \
   > research/qutrit_m4_rank5/results/controls/witness_"$0"_base_"$1".log 2>&1'
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness N --summary
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness H3 --summary
```

Nine processes in all (six N, three H3), each within the pod's 124 GB
through `--max-rss-gb 100` (the cap is checked before every translate-set
selection and after every slice, and a dense solve is refused before it
allocates when its estimate exceeds the cap). The kappa-3 and kappa-4
bases will still abort at `--max-solutions` unless it is raised further;
their first coordinate slice has millions of exact solutions.
