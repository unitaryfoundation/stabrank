# Stage C repair of the rank-5 exclusion for |H>^6

Status (2026-09-23). The review of the qutrit rank-5 pipeline
(`docs/notes/qutrit_m4_rank5_review.md`, sections 2.3, 2.4, and 3) found
three defects in the stage C path of the matcher and one in the aggregate,
all of which the H^6 pipeline shares, since the qutrit code was ported from
`verify_challenge/slice_cover.py` and `research/h6_rank5/driver.py`. This
note records the defects as they stand in the H^6 code, the fixes, what
the regenerated cover list adds, what the stored stage C batches would
have done under the fixed code, and the pod run that replaces stage C.
The attested bound `bounds/qubit_H-m6-lower-6.json` and the stored batch
outputs under `research/h6_rank5/results/` are not modified here; the
bound stays at the attested tier once stage C has been re-run, with a new
manifest and a certificate run.

Notation is that of `docs/notes/h6_rank5_exclusion.md`: psi_3 = |H>^3 is
the target of the base slice, x_0 the base point in F_2^3, u_i the base
state of the i-th term, c_i its coefficient, and T a full 3-cover of psi_3
(three independent stabilizer states whose span contains psi_3 with all
three coefficients nonzero). A block is a base state that appears g >= 2
times in the base multiset; its merged coefficient is the sum of its
copies' coefficients. A translate of a base state u is a vector Q u with Q
a Pauli on the three unsliced qubits, one representative per class modulo
the stabilizer of u; the eight classes are a basis of C^8.

## 1. The defects in the H^6 code

Defect 1, the cover list (`research/h6_rank5/driver.py`,
`degenerate_covers`). The stage C list was built from three routes: a full
3-cover T plus two states drawn from T and span(T), a full 3-cover plus a
pair of states parallel modulo T, and a full 4-cover plus a state of its
span. A rank-5 decomposition whose base multiset has pattern (2, 1, 1, 1)
with the two copies cancelling at x_0 (c_1 = -c_2, merged coefficient
zero) has its three ordinary terms alone covering psi_3, so they form a
full 3-cover T, and the repeated state b can be any state at all. For b in
T or span(T) the first route lists T + (b, b). For b outside span(T)
nothing did: the pool of the first route is T and span(T), the parallel
pairs are pairs of distinct states, and T + b is a 4-cover that `is_full`
rejects because the coefficient of b is zero, so the third route never
sees it. Property P of PR #87 does not exclude the configuration: it
takes the invisible terms at a slice to vanish there one by one, and two
copies present at x_0 with cancelling coefficients are not a vanishing
term. The two copies can agree on the whole four-point plane through x_0
and differ elsewhere, so the configuration is a live case for the
exclusion.

Defect 2, the block reconstruction (`verify_challenge/slice_cover.py`,
`reconstruct_block`). The translate set that a slice equation chooses for
a block records only the classes with nonzero net coordinate. Two copies
in one class at a slice with c_1 i^{l_1} + c_2 i^{l_2} = 0 contribute
nothing there, the set omits their class, and the reconstruction assigned
copies only to the classes of the set, so both copies were marked absent
at that slice. Absent at a slice where the copy is in fact present is not
a valid shape, `valid_term_codes` rejected the only candidate, and the
decomposition was lost without a record.

Defect 3, the final loop of `SliceMatcher._complete` and
`_block_coordinates`. Two silent drops. (a) When a state with a block
reached the final loop with parameters left in its coefficient family,
the merged coefficient was drawn at one random member of the family and
the residual coordinates at a fresh random member per offset, so the data
handed to `reconstruct_block` did not belong to one decomposition and the
true copies failed the linear solve. (b) `_block_coordinates` returned
None when the chosen translates of two blocks were dependent (two blocks
in one Pauli orbit, the split of the residual between them a family), or
when the residual missed the span at the solve tolerance, and the loop
did `ok = False; continue`: the state was dropped and nothing recorded it.

Defect 4, the aggregate (`research/h6_rank5/aggregate.py`,
`check_batch`). A refused run (a family with a dead ordinary coefficient)
was counted and printed but did not fail the batch. The stage lists
exclude such multisets, so a refusal means the list and the matcher
disagree on a cover; it should be a problem, not a statistic.

## 2. The fixes

All four follow the qutrit commit 51ea8ff with the same semantics.

1. `driver.degenerate_covers` adds, for every full 3-cover T and every
   state b outside T and span(T), the multiset T + (b, b), subject to the
   same `ok` test as the other routes (no unrepeated state dead on the
   family; the repeated state is exempt). Nothing else in the enumeration
   changes, so the stage B covers are the same.
2. `slice_cover.reconstruct_block` lets the copies use, at every offset,
   classes outside the chosen translate set, each by at least two copies
   with net coordinate zero, as long as the classes used number at most
   g. Once a copy's three coordinate codes are fixed, its composite codes
   are restricted to the shapes of the structure lemma for them
   (`copy_composite_rows`, the union of `composite_codes` over every flat
   with that coordinate presence, cached per base state), which keeps the
   extra freedom small; `valid_term_codes` still decides every copy at
   the end. The function now requires the data in offset order
   (coordinate offsets first) and asserts it.
3. `slice_cover.UnpinnedFamily` is a new exception. `_complete` raises it
   when a state with a block reaches the final loop with `f.kappa > 0`,
   counting the event in `stats["unpinned"]`, and `_block_coordinates`
   takes a `strict` flag, set at the final loop, under which the three
   None cases (no translate chosen but a nonzero residual, the residual
   outside the span at the solve tolerance, the translates of the blocks
   dependent) raise instead of returning None. In the join the flag is
   off and None still means no pruning. `batch.match_cover` already
   catches every exception into the record's `undecided` list, the batch
   exits 1, and the aggregate fails on a nonempty list, so a run that
   meets either case is visible.
4. `aggregate.check_batch` appends a problem for a nonzero `refused`
   count.

Two pipeline additions carry the repair: `driver.py degenerate --write
--out FILE` writes the list to a file other than `degenerate_covers.json`,
and `driver.py partition-stage-c` writes a stage C repair partition
(section 5); `aggregate.py` reads such a partition when present (or with
`--repair PATH`, ignored with `--no-repair`), and `common.batch_geometry`
looks a batch up by index rather than by position.

Tests (`tests/test_slice_cover.py`, all in the qubit setting with n_1 = 3
against a planted six-qubit target, since a planted rank-5 instance for
|H>^6 is impossible if chi(H^6) = 6): `reconstruct_block` recovers two
copies (v, Z_0 CZ_01 v) with equal coefficients from data that omit the
two offsets where they cancel; the matcher recovers the same pair end to
end among three random full terms; the matcher recovers the cancel-at-base
pair (v, -Z_0 CZ_01 v) from the multiset T + (b, b) with b dead but exempt
in the family; the strict `_block_coordinates` raises on dependent
translates of two blocks in one Pauli orbit, on a residual with no
translate, and on a residual outside the span, returns coordinates on
independent translates, and `batch.match_cover` records a raising run as
undecided at every base point; `_complete` raises on a one-parameter
family with a block; `degenerate_covers` lists T + (b, b) for both
3-covers and every b outside span(T), and the committed delta file is
exactly that set; `check_batch` flags a stored stage C record with one
refusal. The five earlier tests are unchanged and pass.

## 3. The regenerated list

`driver.py degenerate --write --out degenerate_covers_v2.json` (185 s on
one laptop core at nice 19):

| list | file | sha256 (first 16) | covers | (1,1,1,1,1) | (2,1,1,1) | (2,2,1) | (3,1,1) |
|---|---|---|---|---|---|---|---|
| stored | `degenerate_covers.json` | `d58a1e5d366c5b4d` | 26,242 | 12,390 | 13,840 | 6 | 6 |
| regenerated | `degenerate_covers_v2.json` | `8d4fc6934d72398c` | 28,396 | 12,390 | 15,994 | 6 | 6 |

The delta (`degenerate_covers_v2_delta.json`, sha256 `20f60eff7b7dd587`,
hashed like the lists): 2,154 covers, all of pattern (2, 1, 1, 1), 1,077
for each of the two full 3-covers (3, 352, 912) and (75, 353, 749), that
is, every one of the 1,077 dictionary states outside T (span(T) contains
no further stabilizer state), every one accepted by `ok`. The stored list
is a subset of the regenerated one, and the stage B covers are identical.
The stage C part grows from 13,852 to 16,006 covers. The stored list's
hash sits in `partition.json` and in every stage B and C record, so the
stored records still verify against it; the fresh enumeration no longer
equals it, which is why the aggregate compares the fresh enumeration with
the repair list once a repair partition exists (section 5).

## 4. The stored stage C batches under the fixed code

Fixes 2 and 3 change the matcher only where a state reaches the final
reconstruction loop of `_complete`, that is, after all seven slice
equations have a solution. To count the stored stage C covers that would
now behave differently without re-deciding them, the fixed matcher was run
over the 13,852 stage C covers of the stored list at the four base points
(one laptop core at nice 19, load average 10 to 13, 1,839 s in six capped
chunks), with `UnpinnedFamily` caught and counted per run instead of
aborting, and `reconstruct_block` wrapped to count calls and results that
use a class outside the translate set.

| quantity | stored stage C (13,852 covers) | delta (2,154 covers) |
|---|---|---|
| (cover, x_0) runs | 55,408 | 8,616 |
| refused | 0 | 0 |
| `UnpinnedFamily` | 0 | 0 |
| runs with all three coordinate slices solved | 2,646 | 238 |
| runs reaching a block reconstruction | 0 | 0 |
| `reconstruct_block` calls, results, results using an extra class | 0, 0, 0 | 0, 0, 0 |
| raw hits | 0 | 0 |
| seconds per cover (mean, max) | 0.13, 46 | 0.035, 0.44 |

The 2,646 runs that get through the coordinate slices agree exactly with
the stored records' `coord_solution_hist` (2,646 keys of three nonzero
entries over the 17 stage C batches), and every one of them dies at a
composite slice before the final loop. So no stored stage C run took the
old reconstruction path at all: none of the 190 stored batches would
raise `UnpinnedFamily`, none would enter the new `reconstruct_block`
branch, and the 17 stage C records would come out with the same
deterministic content under the fixed matcher. The defect that changes
the exclusion is the first one: 2,154 covers were never matched. Their
laptop run above (8,616 runs, 238 through the coordinate slices, 0
reconstructions, 0 hits, 75 s) is a preview, not a record; the pod run
of section 5 produces the records.

Stage B has no blocks, so `_complete` ends in `confirm` for every state
and neither `reconstruct_block` nor the strict `_block_coordinates` is on
its path; its 158 records are unaffected without a run. Stage A runs the
compiled kernel on distinct independent states and is unaffected.

The rank-6 witness control on its two repeated bases
(`results/control_witness_repeated.json`, 1,062 s and 1,067 s per base,
296 block reconstructions) was not re-run here (above the per-run cap).
On that control several ordinary base states are Pauli translates of the
repeated state, so the strict `_block_coordinates` could now raise where
the old code dropped a state, and the control needs a re-run on the pod
(section 5); it is a control, not part of the exclusion.

## 5. The pod run

The repair partition `research/h6_rank5/partition_stage_c_v2.json`
(`driver.py partition-stage-c`, sha256 `be4ff271d4281721`) holds the
16,006 stage C covers of `degenerate_covers_v2.json` round-robin over 7
batches of 2,286 or 2,287 covers, indices 190 to 196 (190 to 193 have
2,287 covers, 194 to 196 have 2,286), sized at the pod's 2026-09-22 stage
C rate of 0.26 s per cover (3,598 CPU-s over 13,852 covers): about 595 s
per batch, 1.16 CPU-hours in all, about 10 minutes of wall time with the
seven in parallel. It names `partition.json` by hash as the partition it
repairs and batches 173 to 189 as superseded. The record files land in
`results/` beside the stored ones (no index clash) and carry the repair
partition's hash and the new list's hash.

On the pod (`/root/stabrank-h6`, the clone with the compiled kernels;
stages B and C run in Python either way), after this branch is pushed:

```
ssh -i ~/.ssh/id_ed25519 -p 40096 root@157.157.221.30
cd /root/stabrank-h6
git fetch origin h6-stagec-repair && git checkout h6-stagec-repair
uv sync --extra challenge
uv run --extra challenge python -m pytest tests/test_slice_cover.py -q
for K in 190 191 192 193 194 195 196; do
  setsid nohup nice -n 19 uv run --extra challenge python \
    research/h6_rank5/batch.py $K \
    --partition research/h6_rank5/partition_stage_c_v2.json \
    > research/h6_rank5/results/batch_$K.log 2>&1 < /dev/null &
done
disown -a
```

Each batch prints its summary line and exits 0 (clean), 1 (an undecided
run, which includes `UnpinnedFamily`), or 2 (a decomposition). Check
progress and then aggregate; the aggregate picks the repair partition up
by its file name:

```
tail -n 2 research/h6_rank5/results/batch_19[0-6].log
nice -n 19 uv run --extra challenge python research/h6_rank5/aggregate.py \
  --dry-run --partial
nice -n 19 uv run --extra challenge python research/h6_rank5/aggregate.py \
  --recheck 2 --recheck-seed 20260921
```

The second aggregate is what the certificate runs. With the batch pool now
0 to 172 and 190 to 196, the seed picks batches 10 (stage A, about 760 s
on the pod) and 140 (stage B, about 1,300 to 1,600 s) for the re-runs, so
with the 210 s re-enumeration the certificate takes about 2,300 to 2,600 s
against its 3,600 s budget. It writes `batch_manifest.json` with 180
entries (15 stage A, 158 stage B, 7 stage C) and the repair partition's
and list's hashes, and prints `CERTIFIED chi(qubit_H^6) >= 6` when every
check holds. Then the witness control on its repeated bases (about 2 x
1,100 s laptop time, more on the pod):

```
nice -n 19 uv run --extra challenge python research/h6_rank5/driver.py \
  control-witness --repeated-only
```

Copy back `results/batch_19[0-6].json`, `results/batch_19[0-6].log`,
`batch_manifest.json`, and `results/control_witness_repeated.json` with
rsync over port 40096 and commit them.

If a repair batch exits 1 with `UnpinnedFamily` in its `undecided` list,
the aggregate does not certify and the case has to be decided: either the
joint reconstruction (the family parameter or the block split carried as
unknowns shared by the blocks, as the qutrit `BlockOnlyMatcher` does for
its coordinate families) is implemented and the batch re-run, or the
cover is settled by a separate argument. Section 4 makes this unlikely
for the stored covers (no run reaches the loop) and the laptop preview
found none among the added ones.

## 6. The bound afterwards

Run record (2026-09-23). The seven repair batches ran on the pod as in
section 5 (records' clock 09:17 to 09:32 UTC, seven in parallel): 16,006
covers, 64,024 matched runs, 0 refused, 0 hits, 0 undecided, 4,507 CPU-s,
529 to 864 s of wall time per batch. `aggregate.py --dry-run --partial`
passed every stored check over the 180 batches, and `aggregate.py
--recheck 2 --recheck-seed 20260921` re-ran batches 10 (574 s) and 140
(949 s) with matching deterministic hashes and printed `CERTIFIED
chi(qubit_H^6) >= 6` in 1,793 s; the 180-entry `batch_manifest.json` and
the records `results/batch_190.json` to `batch_196.json` are committed.
The witness control on its repeated bases has not been re-run. Before the
records existed the certificate did not pass (with the repair partition
present the aggregate reported the seven batches missing; without it,
`--no-repair`, the fresh enumeration differed from the stored list).

The edits that followed the run:

- `research/h6_rank5/batch_manifest.json` is the new manifest (180
  entries; the 17 superseded stage C records stay in `results/` for the
  history but are not in it). `certificate.attested.batches` in the bound
  keeps pointing at it.
- `bounds/qubit_H-m6-lower-6.json`: update `attested.compute_hours`,
  `attested.hardware`, `attested.note`, `provenance.compute`, and
  `provenance.date` for the stage C re-run (7 batches over 16,006 covers
  with the repaired matcher, the 17 original stage C batches superseded),
  and add to `notes` the cancel-at-base case as a dependency of the
  argument: the base multiset T + (b, b) with b outside span(T) is not
  excluded by property P and is now in the stage C list. The tier stays
  attested; the certificate script and its claim line do not change.
- `docs/notes/h6_rank5_exclusion.md`: section 4 (the block treatment now
  allows cancelling copies and raises on an unpinned or ambiguous
  reconstruction), section 6 (the stage C list and partition), and a new
  section 9 with the run record of the repair; section 7's totals get
  the new stage C figures.
- `research/h6_rank5/README.md`: the file list (the v2 list, the delta,
  the repair partition) and the repair mode of the aggregate.
- `verify_challenge/stabrank_verify.py` runs on the bound as before; the
  manifest format is unchanged apart from the added repair fields.
