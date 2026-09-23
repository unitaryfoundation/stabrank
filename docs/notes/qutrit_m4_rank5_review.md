# Review of the rank-5 exclusion pipeline for |N>^4 and |H3>^4

Status (2026-09-23). Review of `research/qutrit_m4_rank5/` at the head of
`qutrit-m4-rank5-pipeline` (commit e548e5a) against the argument of
`docs/notes/qutrit_m4_rank5_exclusion.md`, sections 1 to 7. The question
is whether a pod run of `batch.py` over the two partitions, aggregated
clean, excludes every rank-5 stabilizer decomposition of |M>^4, M = N or
H3. Answer: not as committed. Three defects let a rank-5 decomposition
with a repeated base state pass unrecorded, and two of them are
exhibited on planted instances below; all three are in the stage C path
(covers with a repeated state). Stages A and B are complete as
implemented. The second commit on this branch fixes what can be fixed
locally and makes the rest fail loudly; the degenerate cover lists and
both partitions then have to be regenerated before any batch is run.

Notation is that of the exclusion note: psi_2 = |M>^2, x_0 the base point
along qutrits 1, 2, u_i^(x) the slice of the term s_i at x, c_i its
coefficient, and G_2 the unitary symmetry group of psi_2 on qutrits 3, 4.
A block is a base state that appears g >= 2 times in the base multiset;
its merged coefficient is the sum of its copies' coefficients.

## 1. Which path each stage runs

`batch.py` builds one `matcher.Matcher` (no `Budget`: every cap is None)
and calls `Matcher.run(cover, x0, target)` on every cover of its batch,
through `match_cover`. Inside `run` the path is chosen by the multiset,
not by the stage:

- Stage A (full 5-covers of five distinct independent states, from
  `CoverEnumerator3.pair_covers`): `family_from` returns a point family
  (kappa = 0), `blocks` is empty, `opts` has five ordinary terms. The two
  coordinate slices go through `solve_slice3` with `_mitm` (meet in the
  middle mod 65521, then the whole equation mod 65521, then the exact
  `restrict`), the join through `_compatible` (dictionary lookup on the
  pinned coefficient vector), the six composite points through
  `_complete`, and every surviving state through `confirm`.
- Stage B (five distinct dependent states, kappa = 1 or 2 from
  `degenerate_covers_ORBIT.json`): the same path with `_dense` (the
  Laplace-feature superset hash) wherever the family still has a
  parameter mod 65521, and `_compatible`'s annihilator test for an
  unpinned first-slice state. No blocks.
- Stage C (a repeated state): the same `Matcher.run` with one or two
  `Block`s. The block enters each slice equation through a translate set
  `Ssel` (all subsets of the nine Pauli classes of size at most g), the
  ordinary terms are solved against the projection onto the annihilator
  of the chosen translates, `_join_blocks` prunes by `_refine_split`, and
  the copies are rebuilt at the end of `_complete` by `reconstruct_block`
  from the residual coordinates at the eight offsets.

`BlockOnlyMatcher` runs only when `opts` is empty, that is, when every
distinct state is repeated. A full 5-multiset with that property has at
most two distinct states, and chi(psi_2) = 3 needs three distinct states
with nonzero merged coefficient at x_0, so no rank-5 cover reaches it; the
stage lists contain no such multiset (patterns (2, 1, 1, 1), (2, 2, 1),
(3, 1, 1) only). Neither `dependent_lines_limit` nor
`_block_only_solutions`'s rejection of dependent selections is on any
rank-5 path. The family path has its own analogue of the second, which
section 2.4 describes.

The `_pin_by_blocks` step that section 7 of the exclusion note describes
(the family pinned by a block whose copies occupy two classes) was
removed in commit e548e5a and is not in the current `matcher.py`; the
note still presents it as part of the family path. Within the family path
the coefficient family is now pinned by the slice restrictions alone.

## 2. Completeness at rank 5

Take a rank-5 decomposition psi_4 = sum_i c_i s_i, all c_i nonzero, and by
section 2 of the exclusion note (Facts A and B and the monomial symmetry)
assume every s_i has a nonzero slice at x_0, so the base multiset
(u_1^(x_0), ..., u_5^(x_0)) satisfies sum_i c_i u_i^(x_0) = alpha_{x_0}
psi_2. Facts A and B are used only there; the matcher allows every flat
through x_0, as the note says.

### 2.1 Stage A

Five distinct independent base states: the coefficient vector is the
unique solution and every entry is nonzero, so the base is a full 5-cover
in the sense of `is_full`, and it is G_2-equivalent to one listed by the
census (G_2 acts on qutrits 3, 4, so it commutes with the slicing and
carries decompositions to decompositions). Matching at one representative
per orbit suffices. The enumeration's completeness rests on the pivot and
partner logic ported from the H^6 enumerator; the r = 3 and r = 4
cross-checks against `slice_lift.all_decompositions` (`control-covers`,
nothing missing, nothing extra) are the evidence for the port, and the
aggregate re-derives the pivot pairs and the stage A cover total. I found
no gap here.

### 2.2 Stage B

Five distinct dependent states with the true coefficient vector c in the
affine family and every c_i nonzero, so `ok(ms)` accepts the multiset
whenever it is generated. The three routes of `driver.degenerate_covers`
(a full 3-cover T plus two states of T or span(T), a full 3-cover plus a
pair parallel modulo it, a full 4-cover plus a state of its span) cover
every case, by the following count. Let k be a dependency of the five
states and r_i = c_i / k_i on its support.

- Rank 3: every independent 3-subset T spans span(S), the other two states
  are in span(T), and T is a full 3-cover because psi_2 in the span of two
  states would contradict chi(psi_2) = 3. Route one.
- Rank 4, |supp k| = 3 or 4 with a ratio value that appears once at index
  j: S minus j is an independent 4-cover with all four coefficients
  nonzero (moving along the family to kill c_j kills nothing else), and
  u_j lies in its span. Route three. All ratios equal is impossible: the
  support would sum to zero and psi_2 would lie in the span of at most
  two states.
- Rank 4, |supp k| = 4 with ratio pattern (2, 2), and |supp k| = 5 with
  pattern (2, 3) or (3, 2): killing the pair with equal ratio leaves an
  independent full 3-cover T, and that pair is parallel modulo T since
  its k-combination lies in span(T). Route two. Patterns with a singleton
  value fall under the previous case, and all five ratios equal is
  impossible as above.

Stage B never has a block, so the final step of `_complete` is `confirm`
on the five fully determined terms, and a family left unpinned there
cannot hide anything (five dependent four-qutrit terms would give a rank
at most 4). Complete as implemented.

### 2.3 Stage C, the base lists

Let D be the set of distinct base states and D' those with nonzero merged
coefficient. Then |D'| >= 3.

- (2, 2, 1) and (3, 1, 1): |D| = 3, so every merged coefficient is
  nonzero, D is an independent full 3-cover T, and the multiset is
  T + (x, y) or T + (x, x) with x, y in T. Generated by route one, and
  `ok` accepts it (no ordinary state dead).
- (2, 1, 1, 1), block b with nonzero merged coefficient: D is a full
  4-cover (route three, x = b) or has rank 3, in which case D contains an
  independent full 3-cover T with the fourth state in span(T) and either
  b in T or b the fourth state; both are in the pool of route one.
- (2, 1, 1, 1), block b with merged coefficient zero, c_1 = -c_2: the
  three ordinary terms alone give alpha_{x_0} psi_2, so they are an
  independent full 3-cover T and the multiset is T + (b, b) with b any of
  the other states. For b in span(T) route one lists it. For b outside
  span(T) nothing does: the pool of route one is T and span(T), route two
  produces pairs of distinct states, and T + b is a 4-cover that `is_full`
  rejects (the coefficient of b is zero), so route three never sees it.
  `ok` would accept the multiset (b is exempt), but it is never proposed.

The last case is not excluded by Fact B. PR #86 (`two_qutrit_slice.py`,
case M) takes the R - 3 invisible terms at a minimal slice to vanish at
x_0 one by one ("the remaining R - 3 terms vanish at x_0, so each is a
line-type term on one of the 8 lines missing x_0 or a point term"). Two
copies with the same nonzero slice u_b at x_0 and coefficients c and -c
are present at x_0 and cancel there, which is a different configuration,
and the base-point argument of section 2 does not remove it either: the
copies can agree on the whole four-point orbit {00, 01, 10, 11} and differ
elsewhere (two plane copies with the same coordinate codes and quadratics
differing by a (x_1^2 - x_1) + b (x_2^2 - x_2) agree at those four points
and differ on the lines x_1 = 2 or x_2 = 2). So the base T + (b, b), b
outside T and span(T), is a live case for the exclusion, and it was
absent from both stage lists.

Planted instance (case B of the review script, seed 1): two plane copies
of base state 271 with quadratics differing by x_1^2 - x_1, coefficients 1
and -1, plus three random plane terms on states 168, 183, and 342. The
base (168, 183, 271, 271, 342) is not in `degenerate_covers_N.json`, and
the matcher at head, run on that base directly, returned 0 hits.

### 2.4 Stage C, the matcher on a listed base

Three places in the family path decide a state by something other than
the slice equations, and each can drop a genuine decomposition.

(a) `reconstruct_block` and `_join_blocks`, cancelling copies at a slice.
The translate set `Ssel` of a slice equation records only classes with
nonzero net coordinate, and `_join_blocks` drops a pinned state whose
chosen translates include one with coordinate zero ("the residual must
use every chosen translate"). That is sound for the equation, since the
smaller set is also enumerated. But `reconstruct_block` then assigns the
copies only to the classes of the set, and a class with no coordinate
means every copy absent there. Two copies in the same class at a slice
with c_1 w^{l_1} + c_2 w^{l_2} = 0 contribute nothing to that slice, the
set omits their class, and the reconstruction marks both copies absent,
which is not a valid shape for a plane copy, so `valid_codes` rejects the
only candidate and the decomposition is gone. No refusal is recorded.
Planted instance (case A, seeds 1 and 2): two plane copies of one base
state with the same classes and phases differing by w^{x_1}, coefficients
1 and -w^2, so the block vanishes at (1, 0), (1, 1), and (1, 2) while its
merged coefficient at x_0 is 1 - w^2. Base (168, 183, 271, 271, 342) and
(39, 93, 107, 107, 299): 0 hits at head, the planted decomposition
recovered (3 genuine hits each) after the fix.

(b) The final loop of `_complete`, a family left unpinned with a block.
If `f.kappa > 0` when a state reaches the reconstruction, the code draws
one random member of the family for the merged coefficient D
(`f.coefficients`) and a fresh random member inside every
`_block_coordinates` call, one per offset, so the coordinates handed to
`reconstruct_block` do not even belong to one decomposition. The
reconstruction of the true codes then fails the linear solve and the
state produces nothing, silently. The commit message of e548e5a records
exactly this failure for the block-only bases ("nothing ever pinned the
coefficient family and the reconstruction used a random point of it")
and fixes it there by carrying the family parameter as an unknown; the
family path keeps the random point. Whether a rank-5 decomposition of
|M>^4 can leave a (2, 1, 1, 1) family with kappa = 1 unpinned through all
eight slice equations is open: it needs sum_i K_i u_i^(x) in the span of
the chosen translates at every x, which the random covers I probed never
satisfied (no probed state reached a reconstruction at all), but the
m = 3 control does reach it, so the configuration is not hypothetical
(section 4).

(c) The final loop of `_complete`, dependent translates across blocks.
For a (2, 2, 1) base the two blocks' chosen translates at a slice can be
dependent (translates of two base states in one Pauli orbit coincide),
the coordinate solve in `_block_coordinates` then has a free direction,
the function returns None, and the loop did `ok = False; continue`: the
state was dropped without a record. This is the family path's version of
the block-only matcher's dependent-selection limit, and unlike that one
it was not recorded anywhere.

## 3. Refusals, caps, and exact re-decision

- `stats["refused"]` is set when `family_from` finds no family (cannot
  happen for a listed cover) or when `has_zero_coefficient(exempt)` finds
  an ordinary state dead on the whole family. The second is a sound
  negative given chi(|M>^4) >= 5: a decomposition with that base would
  use the state with coefficient zero and have rank 4. The stage lists
  exclude such multisets (`ok`), so a refusal in a batch means the list
  and the matcher disagree. `batch.py` counted refusals and exited 0;
  `aggregate.py` printed them and did not fail. The H^6 aggregate does
  the same (its `check_batch` fails on `undecided` only), so the premise
  that the H^6 convention fails on refusals is not what the code does.
  The second commit makes the qutrit aggregate report a refused cover as
  a problem.
- Caps. `batch.py` passes no `Budget`, so `--max-states`,
  `--max-solutions`, `--max-rss-gb`, and `check_dense` are inactive in
  batches. The one cap left is `max_cand = 2,000,000` in `_dense`, which
  raises `BudgetExceeded`; `match_cover` catches every exception into
  `undecided`, the batch exits 1, and the aggregate fails. Recorded.
  `pair_covers` raises `AssertionError` on a parallel class above
  `max_run = 64`; that happens outside `match_cover`, the batch dies with
  no output file, and the aggregate reports the batch missing. Loud,
  though not as a record.
- Tolerances. `has_zero_coefficient` (1e-9 on the complex family),
  `is_full` in the census (same), the `_affine_solve_C` fit tolerance in
  `restrict`'s fast path and in `_block_coordinates` (1e-7 relative), and
  `_refine_split` (1e-7) are floating-point decisions that prune, and none
  is re-decided modulo 2013265921. A false "dead" or a false "outside the
  span" would drop a genuine decomposition. The solves are on 9-row
  systems with unit-modulus entries, so the rounding error is many orders
  below the thresholds, and the exact modular checks sit next to them in
  `restrict`; I rate this low, but `has_zero_coefficient` could cheaply
  require the mod 2013265921 family to agree before pruning.
- Hits. Every raw hit is re-decided in `common.decide_terms` mod
  2013265921 and numerically, both stored, a disagreement is `undecided`,
  and the aggregate re-decides every stored hit from its phase codes.
  No hit is discarded before recording except by `_dedupe` on the code
  set. Sound.
- What the batch never sees. A state dropped inside `Matcher.run` by (a),
  (b), or (c) above leaves no trace in the record, which is the failure
  mode the audit was for; the per-slice solution histogram does not
  reveal it.
- The witness controls in the repository (`results/controls/`,
  `results/*/control_witness.json`) are the laptop records at commit
  b449a56 with `passed: 0`. The pod results quoted in the exclusion note
  (N base 4 passed, H3 bases 0 and 2 recovered once each) are not
  committed, so the review could not check them.

## 4. Verdict and changes

Not complete as committed, for stage C. Stages A and B are complete.

Second commit on this branch (all under the review's compute cap):

1. `matcher.reconstruct_block` allows, at every offset, classes outside
   the translate set used by at least two copies with net coordinate
   zero, as long as the classes used number at most g, and restricts a
   copy's composite codes to the structure lemma's shapes for its
   coordinate codes (`composite_rows`), which keeps the search small (a
   plane copy has one class and at most three phases per composite
   point). Cases A and B of the planted script are recovered. The planted
   control (`control-planted N`, `control-planted H3`) still recovers all
   32 planted decompositions per cell with 0 candidates rejected; the
   repeated case for N now reports 27 further genuine decompositions with
   the same base where it reported 9, which are cancelling-copy variants
   the old reconstruction could not produce, and H3 stays at 15.
2. `Matcher._complete` raises `UnpinnedFamily` (a new exception) instead
   of reconstructing at a random family member when a state with a block
   reaches the final loop with `f.kappa > 0`, and `_block_coordinates`
   in strict mode raises it when the residual split between blocks is a
   family or the residual does not fit. `match_cover` records the cover
   as `undecided`, so a pod run either never meets the case or fails
   visibly. `control-m3 N --sample 200` now aborts within its first
   minute with `UnpinnedFamily` (kappa = 1, one block of two): the
   configuration occurs at m = 3, and that control's earlier pass rested
   on the random-point reconstruction for such states. The proper treatment is the one
   `BlockOnlyMatcher` uses, the family parameter carried into the
   reconstruction as an unknown shared by the blocks; it is not
   implemented here.
3. `driver.degenerate_covers` adds, for every full 3-cover T, the
   multisets T + (b, b) with b outside T and span(T). Dry run: N goes
   from 5,910 to 16,181 stage C covers (16,007 of pattern (2, 1, 1, 1)),
   H3 from 4,888 to 7,024 (6,988). At the sampled (2, 1, 1, 1) rate of
   0.1 to 0.9 s per cover on this machine the addition is about 1 to 3
   CPU-hours for N and well under one for H3. The stored lists and the
   partitions are not regenerated in this commit.
4. `aggregate.check_batch` reports a nonzero `refused` count as a problem.

Before a pod run can claim chi >= 6:

- Regenerate `degenerate_covers_N.json` and `degenerate_covers_H3.json`
  (`driver.py degenerate ORBIT --write`) and both partitions
  (`driver.py partition ...`); every batch record carries the list and
  partition hashes, so nothing already run survives, and `--reenumerate`
  in the aggregate must equal the new lists.
- Decide what to do with `UnpinnedFamily` at rank 5: either the run
  never raises it, in which case the record shows it, or implement the
  joint reconstruction and add a planted instance that exercises it.
  A planted instance for case (c) (two blocks in one Pauli orbit) is
  worth adding either way.
- Section 7 of the exclusion note should drop `_pin_by_blocks` from the
  description of the family path and add the cancelling-copy case to the
  stage C description and to the dependencies listed for the bound files
  (Fact B does not cover it).
- The same three defects are in `research/h6_rank5/driver.degenerate_covers`
  (the same pool logic, no T + (b, b) route) and in
  `verify_challenge/slice_cover.reconstruct_block`, `_join_blocks`, and
  `SliceMatcher._complete`, from which the qutrit code was ported. The
  attested H^6 bound inherits them and needs the same review.

Review script and probes (not committed): planted cancelling pairs at a
coordinate slice and at the base point, and about 60 N and 30 H3 stage C
runs spread over the lists with a per-cover deadline (0 hits, 0 refusals,
no state reached a reconstruction; (2, 2, 1) covers run 3 to 8 s or past
40 s at the composite stage, the others 0.1 to 1.2 s).
