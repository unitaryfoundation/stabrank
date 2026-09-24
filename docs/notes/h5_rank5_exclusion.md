# Excluding rank 5 for |H>^5: design note

Status (2026-09-23). Pipeline built and controlled on the laptop; the pod
run has not been launched, so nothing here is a bound. The cell is
5 <= chi(H^5) <= 6 (`bounds/qubit_H-m5-lower-5.json`, slice and lift;
`bounds/qubit_H-m5-upper-6.json`, the Bravyi, Smith, and Smolin witness at
the Lean tier). A clean run of the batches of `research/h5_rank5/`
settles chi(H^5) = 6. The design was chosen in
`docs/notes/next_exclusion_feasibility.md` (section 4); this note is the
argument as implemented, written so that it can be checked from the files
alone.

Notation. psi_m = |H>^m with |H> = cos(pi/8)|0> + sin(pi/8)|1>, t =
tan(pi/8). N_3 = 1080 three-qubit stabilizer states in the order of
`dictionary(2, 3)`; G_3 the unitary symmetry group of psi_3 (order 48). A
rank-5 decomposition is psi_5 = sum_{i=1}^5 c_i s_i with stabilizer states
s_i and all c_i nonzero. Slicing along a set S of n_1 qubits at x in
F_2^{n_1}: u_i^(x) = (<x| (x) I) s_i and sum_i c_i u_i^(x) = alpha_x
psi_{5 - n_1} with alpha_x = cos(pi/8)^{n_1 - |x|} sin(pi/8)^{|x|}, never
zero; the ratio between the slices at x and x_0 is t^{|x| - |x_0|}. Each
term is nonzero exactly on an affine flat of F_2^{n_1} (the projection of
its support). The slice structure lemma (`research/constructions/
two_qubit_slice.py`, PR #87) describes the slices of one term on its flat
from the slice at one point: along each basis direction a fourth root of
unity times a Pauli on the unsliced qubits, on composite points the class
product with one quadratic sign per basis pair. Everything below runs at
n_1 = 2 along qubits 1, 2; by the copy symmetry the pair is general.

## 1. The two facts and how each is established

Fact 1 (every term is full along every qubit). Slice along one qubit at
value k in {0, 1}. The visible terms decompose alpha_k psi_4, so there are
at least chi(H^4) = 4 of them (`bounds/qubit_H-m4-lower-4.json`). If
exactly four are visible at k, they are distinct and independent
(otherwise psi_4 lies in the span of at most three stabilizer states), so
they are one of the 30 minimal decompositions of |H>^4 up to its unitary
symmetry (`research/constructions/data/qubit_H_m4_rank4.json`; the symmetry
acts on the unsliced qubits and preserves slices), and the fifth term is
|k'> (x) v with v a four-qubit stabilizer state. The equation at k' says
that alpha_{k'} psi_4 minus the four visible terms' slices at k' (Pauli
translates of their slices at k, or zero) is a nonzero multiple of a
stabilizer state: a k = 1 row of the residual tables of PR #87 at the ratio
t^{+-1}. Those tables, over all 65^4 code combinations of the four terms
for each of the 30 decompositions, have (exact, stabilizer) = (0, 0) at j =
-1 and j = 1 (`two_qubit_slice.py --m 6 --n1 2 --ratio-tables 1`, about 5
minutes; recomputed here by `driver.py tables --fact1`, `results/
tables.json`). So no one-qubit slice has exactly four visible terms, every
term is nonzero at both values of every qubit, and no term is a product
across any qubit. Fact 1 rests on the completeness of the 30-element list
(the same dependency the H^6 bound declares) and on PR #87's shape
enumeration, which has its own controls.

Fact 2 (at most two absent terms at any two-qubit point). The visible
terms at a point of F_2^2 decompose a nonzero multiple of psi_3, and
chi(H^3) = 3 (`bounds/qubit_H-m3-lower-3.json`).

## 2. The case split

Flat lemma. By Fact 1 the flat of every term along qubits 1, 2 projects
onto both coordinates, so it is the plane F_2^2 or one of the two diagonal
lines A = {00, 11} and B = {01, 10}; the coordinate lines and the points
are products across a qubit. Let n_A and n_B be the numbers of line terms
on A and on B. By Fact 2, n_A <= 2 and n_B <= 2 (a line term on A is
absent at 01 and at 10).

Base point.

- n_B = 0: at x_0 = 00 all five terms are visible.
- n_A = 0 and n_B >= 1: at x_0 = 01 all five are visible.
- n_A >= n_B >= 1: at x_0 = 00 the planes and the A lines are visible and
  the n_B terms on B are invisible. With n_B = 1 four terms are visible.
  With n_B = 2 also n_A = 2: one plane, two lines on each diagonal.
- n_B > n_A >= 1: n_A = 1 and n_B = 2, and at x_0 = 01 four terms are
  visible (the plane terms and the two B lines), the A line invisible.

The (2, 2) configuration does not occur. At 00 the plane and the two A
lines are visible and form a rank-3 decomposition of alpha_{00} psi_3
(distinct and independent, since psi_3 in the span of two stabilizer states
contradicts chi(H^3) = 3), so up to the symmetry of psi_3 it is one of the 6
stored rank-3 decompositions of |H>^3 (`qubit_H_m3_rank3.json`). At 11 the
same three terms are present with slices that are phased Pauli translates
of their slices at 00, the two B lines are absent, and the ratio is t^2: an
exact (k = 0) row of PR #87's tables at j = 2. Those tables for the 6
decompositions have 0 exact combinations at j = -2 and j = 2 (`two_qubit_
slice.py --m 5 --n1 2 --ratio-tables 2`, about a second; `driver.py
tables`, `results/tables.json`).

What remains are two stages, each at the two base points 00 and 01 (01 and
10 are exchanged by the swap of the sliced qubits, a symmetry of psi_5):

- Stage (alpha): all five terms visible at x_0. The base
  (u_1^(x_0), ..., u_5^(x_0)) is a full 5-cover of psi_3: five stabilizer
  states, repeats allowed, whose span contains psi_3 with all coefficients
  d_i = c_i / alpha_{x_0} nonzero. This is exactly the object of the H^6
  exclusion (`docs/notes/h6_rank5_exclusion.md`), whose census and
  degenerate lists are reused unchanged (section 3).
- Stage (beta): four terms visible at x_0 (three planes and a line on the
  diagonal through x_0, or two and two) and one line term on the diagonal
  missing x_0. The base is a full 4-cover of psi_3. The point x_0 + 11 has
  the four visible terms present and the fifth absent; the two coordinate
  points x_0 + 01 and x_0 + 10 have the fifth term present with slices
  c_5 v and c_5 i^l Q v (v a three-qubit stabilizer state, Q a Pauli).

The matcher of stage (alpha) assumes nothing beyond the base point: every
term may have any of the five subspaces of F_2^2 through x_0 as its flat
(the absence option is in every coordinate-slice option list). The matcher
of stage (beta) assumes the flat of the invisible term (Fact 1: a diagonal
line, so present at both coordinate points and absent at x_0 + 11) and
nothing about the visible terms' flats; it also accepts a zero residual at
one coordinate point, so a product fifth term at the other coordinate point
would be found as well. Facts 1 and 2 are otherwise used only to show that
the enumerated bases are complete.

## 3. The bases

Stage (alpha): the full 5-covers of psi_3, the same set for any n_1. From
the H^6 run, by hash:

| list | file | count | content |
|---|---|---|---|
| stage A | `research/h6_rank5/results/kernel_census.json` | 5,939,465 | covers of five distinct independent states, one per G_3 orbit as far as the pivot and partner reductions go, regenerated per pivot pair (14,280 pairs) by the compiled kernel `cover5_pair` |
| stages B, C | `research/h6_rank5/degenerate_covers_v2.json` (sha256 `8d4fc6934d72398c`) | 28,396 | 12,390 dependent covers of five distinct states, pattern (1, 1, 1, 1, 1), kappa = 1 (stage B); 16,006 with a repeated state (stage C): 15,994 of pattern (2, 1, 1, 1), including the 2,154 cancel-at-base multisets T + (b, b) with b outside span(T), 6 of (2, 2, 1), 6 of (3, 1, 1) |

The partition records the census file's SHA-256 and the list's hash; the
aggregate re-enumerates the degenerate list (`driver.degenerate_covers`,
the same routes as the H^6 v2 list, kept in `research/h5_rank5/driver.py`
so that nothing is imported from `research/h6_rank5`) and requires
equality. Completeness of the census is the H^6 argument's (section 3 of
that note): pivot one per orbit, partner minimal in its stabilizer orbit,
members in orbits at or above the pivot's, the residue kernel mod 65521
listing a superset, every candidate decided mod 2013265921 and
numerically. G_3 acts on the three unsliced qubits, commutes with the
slicing, and carries rank-5 decompositions of psi_5 to rank-5
decompositions, so one base per orbit suffices at n_1 = 2 as at n_1 = 3.

Stage (beta): the full 4-covers of psi_3 (`research/h5_rank5/beta_covers.json`,
sha256 `51b531daec2d6804`, `driver.py beta-covers --write`, 178 s):

| kind | count | content |
|---|---|---|
| distinct independent | 3,460 | `CoverEnumerator.covers(4)`, one per G_3 orbit (cross-checked at H^6 against the numeric enumerator: its 3,731 tuples are these plus 271 non-full ones) |
| degenerate | 6 | pattern (2, 1, 1): one state of a full 3-cover T doubled, three per 3-cover class (T = (3, 352, 912) and (75, 353, 749)) |

Why nothing else: four visible terms with a repeated state have three
distinct states with merged coefficients; if the block's merged
coefficient is nonzero the three states are a full 3-cover T and the
repeated one is in T (the 6 listed); if it is zero (two copies cancelling
at x_0) the two other states cover psi_3, against chi(H^3) = 3. Four
distinct dependent states contain a 3-cover T with the fourth state in
span(T), and span(T) holds no further dictionary state for either class
(enumerated: 0 and 0). Patterns (2, 2) and (3, 1) have two distinct states.
The aggregate re-enumerates this list too (`driver.beta_bases`, 3-covers,
4-covers and the degenerate multisets over the 3-cover classes with the
same `ok` test as the H^6 lists: no unrepeated state dead on the family).

## 4. The matchers

Stage (alpha) runs `verify_challenge/slice_cover.SliceMatcher(E, 2)`
unchanged, at x_0 in {00, 01}: the two coordinate slices at ratio t^{+-1}
and the composite slice at ratio t^{+-2} (x_0 = 00) or 1 (x_0 = 01). The
class is generic in n_1; the compiled kernel `SliceMatchKernel` accepts
n_1 = 2 and handles stage A (distinct independent bases), stages B and C
run the Python reference through the coefficient family and the block
treatment with the 2026-09-23 repairs (cancelling copies reconstructed,
`UnpinnedFamily` raised instead of a silent drop, `docs/notes/
h6_rank5_stagec_repair.md`). Everything section 4 of the H^6 note says
about exactness holds here with psi_5 in place of psi_6.

Stage (beta) is `research/h5_rank5/beta.py`, `BetaMatcher.run(cover, x0)`;
its module docstring is the specification. In short, for a base with
coefficients d (a point: the bases are independent or a 3-cover with one
state doubled) and a block for a repeated state:

1. x_b = x_0 + 11, exact: `solve_slice` over 33 options per visible term
   (32 phased translates and absent) and the block's translate sets, meet
   in the middle mod 65521, every candidate decided over C and mod
   2013265921; the block's coordinates on its chosen translates and the
   admissible coefficient splits of the pair are recorded.
2. x_a1 = x_0 + 01, residual: all 33^4 (or 33^2 x 37 with a block) option
   combinations at once mod 65521; the residual after the visible terms,
   projected onto the annihilator of the block's translates, normalised
   and looked up in the table of the normalised projected dictionary
   states by a random functional (states inside the block's span are
   excluded from the table; their zero pattern mod 65521 must agree with
   the one over C, else the run raises). Every match is re-decided: the
   system (translates | v)(a, c_5) = residual is solved over C, mod 65521
   and mod 2013265921 and must have a unique solution with every entry
   nonzero. A zero residual is kept as "fifth term absent here".
3. x_a2 = x_0 + 10, exact given (v, c_5): the visible terms' options are
   those compatible with their codes at x_a1 and x_b (a plane term's code
   is fixed up to the quadratic sign by composition, a term present at
   exactly one of the two points is a line term and absent, a term absent
   at both is free), the fifth term is one of the 32 phased translates of
   v or absent, and `solve_slice` decides the equation with the fifth term
   carried as a fifth ordinary term of known coefficient in all three
   fields. When the fifth term was absent at x_a1 the residual test of
   step 2 runs at x_a2 instead (a point term there).
4. Assembly: each visible term's codes must pass `valid_term_codes` (a
   subspace through x_0 with the structure lemma's composite code), the
   block's copies are rebuilt by `reconstruct_block` from the coordinates
   of the three slices (cancelling copies allowed), the fifth term is
   built from v and its translate, and every hit goes through
   `SliceMatcher.confirm` (residual against psi_5, rank, independence,
   nonzero coefficients, span mod 2013265921); `batch.py` re-decides every
   hit again from its phase codes.

The ambiguous case. With a block, the residual at a coordinate point can
lie in the span of the block's chosen translates with no fifth term
needed there; the fifth term's slice could then be a dictionary state
inside that span (a translate of the repeated state, or another
stabilizer state in the span of two translates), and the slice equation
cannot separate it from the copies. This happens on the real bases (base
(75, 353, 749, 749) at x_0 = 01 in the sample). `BetaMatcher._pair_brute`
decides it by enumeration: for every dictionary state v inside the span,
every assignment of the two copies' codes at the three points, every code
of the fifth term at x_a2 and every admissible option of the visible
terms there, the three coefficients (c_1, c_2, c_5) are the solution of the
linear system of the three slice equations in the translate basis of the
repeated state together with c_1 + c_2 = D, solved in batches offset by
offset; a survivor whose system leaves a coefficient free raises
`UnpinnedFamily`. The same ambiguity for a point term at x_a2 (a product
term, excluded by Fact 1) is counted (`point_ambiguous`) and not searched.
The enumeration is written for one pair block, which is all the list
contains; anything else raises.

Refusals and undecided runs. A base whose family is empty or has a dead
ordinary coefficient is refused; the lists exclude such multisets, so a
refusal fails the aggregate. `UnpinnedFamily` (any of the cases above),
the candidate cap of the dense solve, a hit on which the modular and
numeric decisions disagree, and a cover not run before `--max-seconds`
are recorded as undecided; the batch exits 1 and the aggregate fails.

## 5. Controls

All on one laptop core at nice 19 with load average 45 to 60 on 18 cores
(other sessions), so every time below overstates an unloaded core.

Control 1, the rank-6 witness (`driver.py control-witness`,
`results/control_witness.json`). `bounds/qubit_H-m5-upper-6.json` sliced
along every qubit pair has 24 distinct all-visible (base, x_0) pairs: 22
with six distinct independent states (kappa = 0, the native path), one
with a repeated state (kappa = 1, base 4), one with six distinct dependent
states (kappa = 2, base 5). Its flat types are planes everywhere except
one line term along the pair (0, 4) and two line terms on one diagonal
along (1, 2), so the witness exercises stage (alpha) only; stage (beta) is
controlled by planted instances. Result: 24 of 24 bases return the
witness itself among their genuine rank-6 decompositions (1 to 6 per
base, rank 6, exact mod 2013265921, all coefficients nonzero). The 22
kappa = 0 bases run through the compiled kernel in 0.1 to 0.8 s each;
base 4 (cover (74, 74, 908, 350, 180, 166) at 00, one pair block) takes
47 s through the block path (coordinate slices 1,181 and 14,731
solutions, 3,515 joined states); base 5 (cover (74, 183, 908, 350, 180,
166) at 11, six distinct dependent states, kappa = 2) exceeds the
matcher's default candidate cap of 2,000,000 at its first coordinate slice
(a 2-parameter dense solve; the run aborts after 30 s with the cap's
`AssertionError`, which the control records as aborted and counts as a
failure, `control-witness: 23/24, 1 aborted, FAIL` at the default cap) and
finishes under `--max-cand 10000000` in 264 s (coordinate slices 246 and
910 solutions, 814 joined states, the witness the single hit). The stored
record `results/control_witness.json` is the run at the raised cap; the
cap is a `SliceMatcher` parameter and the batches run at the default,
where a run over the cap is recorded as undecided. A 2-parameter family of
six distinct states is a rank-6 shape and lies on no rank-5 path (stage B
has five states and kappa = 1).

Control 2, planted stage (beta) instances (`driver.py control-beta
--plant 10`, `results/control_beta.json`): random five-term configurations
of five kinds at both base points, the target their sum with
Gaussian-integer coefficients, the matcher run against the target's
slices: (a) three planes, a line on the diagonal through x_0, the invisible
line; (b) two planes and two lines; (c) a repeated base state (two plane
copies with one base slice); (d) a visible term on a coordinate line
through x_0 (excluded by Fact 1, but the matcher must find it); (e) a
repeated base state whose first copy shares a Pauli class with the
invisible term at the first coordinate point, the ambiguous case of
section 4. Result: 20 of 20 planted decompositions recovered (two per
kind per base point; every instance a genuine hit of the planted target
with the planted term codes), in 0.2 to 9 s each; the (e) instances
exercise `_pair_brute` (1 to 4 ambiguous translate sets, up to 3.7 million
batched systems, 1 to 1.3 s after the restructuring of its third stage,
which matches the x_a2 equation with the coefficients pinned by x_b and
x_a1 instead of solving 33^3 systems per visible-term combination). The
(d) instance at x_0 = 00 returns 48 hits, all genuine decompositions of
the planted target with the same base: with a visible coordinate-line term
the point term at x_a2 can be chosen in many ways.

Control 3, the rank-4 decompositions of |H>^4 along a qubit pair
(`driver.py control-m4-pair`, `results/control_m4_pair.json`): the stage
(alpha) matcher at n_1 = 2 over every full 4-cover of |H>^2 (the 60
two-qubit states) must recover every stored class of `qubit_H_m4_rank4.json`
that has an all-visible base at 00 or 01 and nothing outside the stored
list. Bases: 67,336 independent full 4-covers of |H>^2 (one per orbit of
the symmetry of psi_2) and 2,144 dependent or repeated ones, 138,960
(cover, x_0) runs in 96 s (native path), 216 genuine rank-4 hits forming 23
classes under the unitary symmetry group of psi_4 (order 384). Result: the
23 stored classes are all expected (each has a member with an all-visible
base at 00 or 01 with a full family; the expectation is computed over the
whole class, since the enumerator's bases are orbit representatives) and
all 23 are recovered, nothing missing, nothing unexpected, nothing outside
the stored list, no refusal, no undecided run. A first run that computed
the expectation from the stored representative alone found 22 expected
and 23 recovered; the extra class was a stored representative none of
whose own pair slices at 00 or 01 is all-visible, while another member of
its class has one.

Control 4, the tables (`driver.py tables --fact1`, `results/tables.json`,
the committed `two_qubit_slice.py` run as a subprocess and its totals
parsed): the 6 rank-3 decompositions of |H>^3 at |j| <= 2 give (exact,
stabilizer) = (0, 0) at j = -2 and j = 2, (0, 15) at j = +-1, (6, 594) at
j = 0 (0.6 s); the 30 rank-4 decompositions of |H>^4 at |j| <= 1 give
(0, 0) at j = -1 and at j = 1 and (30, 9169) at j = 0 (290 s). Both agree
with the feasibility note's recomputation.

## 6. Rates and partition

Measured by `driver.py sample STAGE` (`results/rates.json`), per (cover,
x_0) run, one process at nice 19 on the loaded laptop:

| stage | covers sampled | matcher | per (cover, x_0): mean | median | max | hits, refused, undecided |
|---|---|---|---|---|---|---|
| A | 300 from the first pivot | native | 0.4 ms | 0.3 ms | 9 ms | 0, 0, 0 |
| B, pattern (1, 1, 1, 1, 1) | 8 spread through the 12,390 | reference (1-parameter dense solve at every base point) | 3.54 s | 3.99 s | 4.63 s | 0, 0, 0 |
| C, (2, 1, 1, 1) | 12 spread through the 15,994 | reference (block) | 0.076 s | | | 0, 0, 0 |
| C, (2, 2, 1) | all 6 | reference | 1.18 s | | 6.3 s | 0, 0, 0 |
| C, (3, 1, 1) | all 6 | reference | 0.078 s | | | 0, 0, 0 |
| beta, independent | 30 spread through the 3,460 | BetaMatcher | 0.178 s | 0.07 s | 1.6 s | 0, 0, 0 |
| beta, (2, 1, 1) | all 6 | BetaMatcher (block, joint reconstruction) | 3.58 s | | 42 s (base (75, 353, 749, 749) at 01, 874 x_a1 solutions, 4 ambiguous sets) | 0, 0, 0 |

Stage A's solution histogram over the 600 runs: 458 die at the first
coordinate slice, the rest have (1, 1) to (12, 144) solutions on the two
coordinate slices, as in the feasibility note. The three quarters of
stage beta runs at x_0 = 00 have no x_b solution (ratio t^2); at x_0 = 01
the trivial x_b solution (every term as at x_0, ratio 1) always exists and
the x_a1 residual scan decides the run.

The partition (`driver.py partition --target-s 600 --pod-factor 2.0`,
`partition.json`) sizes every batch at about 600 s of pod time, taking the
pod at twice the laptop rates (the H^6 stage B ran 2.3 times slower on the
shared pod than in the laptop sample): stage A pivot pairs grouped
greedily by the census's kernel seconds plus the sampled matcher time per
cover at two base points; stages B, C and beta round-robin over their
sorted lists at the sampled rate per multiplicity pattern.

`partition.json` (sha256 `f6afa1c778771af5`, 323 batches, git de2540c):

| stage | batches | indices | per batch | estimated pod CPU-hours |
|---|---|---|---|---|
| A | 16 | 0 to 15 | 37 to 5,100 pivot pairs (26,336 to 446,924 covers; kernel 6 to 106 s plus 0.35 ms per run), about 300 laptop s each, batch 15 about 60 s | 2.6 |
| B | 293 | 16 to 308 | 42 or 43 covers at 14.1 pod s per cover | 48.7 |
| C | 9 | 309 to 317 | 1,778 or 1,779 covers at 0.30 pod s (the (2, 2, 1) covers at 4.7 s) | 1.4 |
| beta | 5 | 318 to 322 | 693 or 694 covers at 0.71 pod s (the six (2, 1, 1) bases at 14.3 s) | 0.7 |
| total | 323 | | | 53.4 |

Wall time on 15 processes: about 3.6 hours, dominated by stage B (293
batches of about 600 s). If the pod runs closer to the laptop rate the
batches finish sooner and the partition is still valid; if stage B is
slower than the factor 2, the first stage B batch shows it
(`aggregate.py --dry-run --partial` prints the projected total per stage
at the observed rate) and the loop simply takes longer, since nothing in
the argument depends on the batch size. Compiling the 1-parameter dense
solve would remove most of stage B's 48 hours, as noted for H^6; not
needed to run.

Pipeline test (one laptop core at nice 19, load average 45 to 60). One
batch of each stage into `results/`, all clean (0 refused, 0 hits, 0
undecided): stage A batch 15 (5,100 pivot pairs, 26,336 covers, 52,672
runs, kernel 82 s, match 18 s, 100 s wall); stage beta batch 322 (693
covers including the (2, 1, 1) base (75, 353, 749, 749), 1,386 runs, 200
s); stage C batch 309 (1,779 covers, 3,558 runs, 59 s); stage B batch 16
(43 covers, 86 runs, 272 s, 3.2 s per run against the sampled 3.5). The
deadline path: batch 322 with `--max-seconds 5` into a scratch directory
stops after 36 runs, records the 1,350 runs not made as undecided
("deadline: not run"), exits 1, and a rerun without `--resume` refuses to
skip it (exit 1) while `--resume` redoes it. `aggregate.py --dry-run
--partial` re-enumerated both lists (28,396 and 3,466, equal to the
stored ones, 278 s), passed every stored check on the four records, wrote
`batch_manifest.partial.json`, and projected the stage totals at the
observed rates (stage B 19 CPU-hours at the laptop rate, against the
partition's 49 pod hours).

## 7. Soundness checklist

From `docs/notes/qutrit_m4_rank5_review.md` and the H^6 stage C repair;
status of each item for this pipeline.

1. The stage C list carries the cancel-at-base multisets T + (b, b): yes,
   the v2 list by hash; the aggregate's fresh enumeration must equal it.
2. `reconstruct_block` allows classes outside the translate set used by
   two copies with net coordinate zero and restricts composite codes to
   the structure lemma's shapes: yes, the repaired `slice_cover.py` is
   used by both matchers.
3. A block state reaching the final loop with an unpinned family or with
   dependent block translates raises `UnpinnedFamily`; the batch records
   the run as undecided and the aggregate fails: yes (stage (alpha)
   unchanged; stage (beta) raises on every case its matcher cannot decide,
   section 4).
4. A refused cover fails the aggregate: yes (`aggregate.check_batch`).
5. Floating-point pruning backed by exact checks: stage (beta)'s residual
   lookup is done mod 65521 (a superset) and every survivor re-decided
   over C and mod 2013265921 (step 2 of section 4); the ambiguous-case
   enumeration prunes by complex least squares at tolerance 1e-7 on
   systems with unit-modulus entries, and its survivors are confirmed
   exactly, as `reconstruct_block`'s solves are in the H^6 pipeline. The
   tolerances of `has_zero_coefficient`, `is_full`, `restrict`, and
   `_refine_split` are as at H^6.
6. Every hit re-decided from its phase codes mod 2013265921 and numerically
   (`common.decide_terms`), a disagreement undecided, the aggregate
   re-deciding every stored hit: yes.
7. The deterministic hash excludes the modular candidate count: yes
   (`candidates` follows `deterministic_sha256` in the record). The
   `--no-native` replay of one stage A batch is a pod task (section 8);
   open until run.
8. Every batch record carries the partition, census, degenerate-list and
   beta-list hashes; the aggregate checks them: yes.
9. No `id()`-keyed caches: the stage (beta) tables are keyed by the block
   state's dictionary index and the translate set, a pure function of the
   key; `SliceMatcher.cache` and `BetaMatcher._fifth` are keyed by
   dictionary index.
10. `--max-seconds`: covers not run are recorded as undecided (a stage A
    pivot pair not run is recorded as a unit), the batch exits 1, and
    `--resume` redoes such a record.
11. The matcher assumes nothing beyond the base point: stage (alpha) yes;
    stage (beta) assumes Fact 1 for the invisible term's flat (present at
    both coordinate points, absent at x_0 + 11) and nothing for the
    visible terms (the absent option is in every option list, the
    presence pattern is filtered to subspaces through x_0 at assembly).
12. Controls re-run after the last matcher change: the witness control,
    the planted control and the m = 4 pair control were run at the commit
    that carries the final `beta.py` and `slice_cover.py` (the only change
    to `slice_cover.py` is the `max_cand` parameter of `SliceMatcher`);
    the witness control's base 5 aborts at the default candidate cap and
    is recorded as aborted (a control failure, never a pass), and passes
    at the raised cap (section 5).

## 8. The pod run

The pod checkout is `/root/stabrank-h6`; the branch is fetched, never
`git checkout -f` (it would overwrite result files); runs go under `setsid
nohup ... < /dev/null & disown`; the batch loop is resumable (a finished
record is skipped, one left incomplete by the deadline is redone with
`--resume`).

```
ssh -i ~/.ssh/id_ed25519 -p 40096 root@157.157.221.30
cd /root/stabrank-h6
git fetch origin h5-rank5-pipeline && git checkout h5-rank5-pipeline && git pull --ff-only
uv sync --extra challenge
uv run --extra challenge python -c "import stabrank.stabrank_core as c; print(c.cover5_pair, c.SliceMatchKernel)"
uv run --extra challenge python -m pytest tests/test_slice_cover.py -q
mkdir -p research/h5_rank5/results /root/logs
```

Rate check first (one batch of each stage, in the foreground of a
`setsid nohup` shell, about ten minutes), then the aggregate's `--partial`
projection tells whether the pod factor of 2 holds before the whole
partition is launched:

```
setsid nohup sh -c 'for K in 0 16 309 318; do
  nice -n 19 uv run --extra challenge python research/h5_rank5/batch.py $K --max-seconds 3600 \
    > research/h5_rank5/results/batch_$K.log 2>&1; done' < /dev/null > /root/logs/h5_probe.log 2>&1 & disown
nice -n 19 uv run --extra challenge python research/h5_rank5/aggregate.py --dry-run --partial --no-reenumerate
```

(0, 16, 309, 318 are the first batches of stages A, B, C, and beta;
batches 15, 16, 309, and 322 were run on the laptop as the pipeline test
and are skipped by the loop if their records are copied over.) The full
run, 15 processes over the 323 batches:

```
setsid nohup sh -c 'seq 0 322 | xargs -P 15 -n 1 sh -c \
  "nice -n 19 uv run --extra challenge python research/h5_rank5/batch.py \"\$0\" --resume --max-seconds 3600 \
   > research/h5_rank5/results/batch_\"\$0\".log 2>&1"' < /dev/null > /root/logs/h5_run.log 2>&1 & disown
```

Rerunning the same command resumes. Progress, the final check (what the
certificate runs), and the optional `--no-native` replay of the smallest
stage A batch (about 80 times its native time; a pod-hours task, not
needed for the aggregate):

```
grep -l "DECOMPOSITION FOUND\|undecided run" research/h5_rank5/results/batch_*.log
nice -n 19 uv run --extra challenge python research/h5_rank5/aggregate.py --dry-run --partial
nice -n 19 uv run --extra challenge python research/h5_rank5/aggregate.py --recheck 2 --recheck-seed 20260923
setsid nohup sh -c 'STABRANK_NO_NATIVE=1 nice -n 19 uv run --extra challenge python \
  research/h5_rank5/batch.py 15 --out-dir research/h5_rank5/results/nonative --force' \
  < /dev/null > /root/logs/h5_nonative.log 2>&1 & disown
```

Copy back `research/h5_rank5/results/batch_*.json`, the logs,
`batch_manifest.json`, and the no-native record with rsync over port
40096, commit them, fill the placeholders of
`research/h5_rank5/qubit_H-m5-lower-6.json.draft` (compute hours,
hardware, dates, the batch indices of the re-runs) and move it to
`bounds/qubit_H-m5-lower-6.json`; the submissions workflow verifies every
touched bound, so the draft keeps its suffix until the manifest exists.
Then update the board and project the bound to no other cell (the m = 6
cell already has chi = 6; m >= 7 cells hold chi >= 6 from it).

### 8.1 The run (2026-09-23 to 2026-09-24)

All 323 batches ran on the RunPod pod: the stage A probe batch 0 at 21:56
UTC (446,924 covers, 222 s), then the 323-batch loop fifteen at a time from
22:02 to 00:28 UTC, with the pod's two anneal loops paused for it. Totals
from the aggregate: stage A 16 batches over the 5,939,465 full 5-covers of
|H>^3 (326 s per batch, 0.9 ms per cover), stage B 293 batches over the
12,390 dependent covers (431 s per batch, 10.2 s per cover), stage C 9
batches over the 16,006 covers with a repeated state (82 s per batch), and
stage beta 5 batches over the 3,466 full 4-covers (286 s per batch); every
cover matched at both base points, 0 hits, 0 refused, 0 undecided, 37.2
CPU-hours in all (the partition estimated 53). The aggregate with
`--recheck 2 --recheck-seed 20260923` re-enumerated the degenerate covers
and the stage beta bases (equal to the stored lists, 214 s), verified every
stored batch, re-ran batches 19 (stage B, 249 s) and 319 (stage beta,
227 s) from scratch with matching deterministic hashes, wrote
`batch_manifest.json`, and printed `CERTIFIED chi(qubit_H^5) >= 6` in
691 s. With the Lean rank-6 witness the cell is chi(H^5) = 6, filed as
`bounds/qubit_H-m5-lower-6.json` at the attested tier. The `--no-native`
replay of stage A batch 15 (item 7 of the checklist) was started on the pod
after the run; its record goes under `results/nonative/` when it finishes.

### 8.2 The no-native cross-check (2026-09-24)

Stage A batch 15 (5,100 pivot pairs, the smallest stage A batch) was
re-run on the pod with `STABRANK_NO_NATIVE=1`, so both the 5-cover kernel
and the matcher were the Python reference implementations: 8,312 s
(kernel 8,096 s, matching 216 s) against 424 s compiled.
`results/nonative/batch_15.json` agrees with the stored
`results/batch_15.json` on every exact quantity: 26,336 covers, 52,672
matched runs, the same coordinate-slice solution histogram, 0 hits, 0
refused, 0 undecided. Two fields differ. The modular candidate count
(86,864,687 against 86,719,704) sits outside the deterministic part and
counts the superset each kernel's random hash functional lets through
before exact re-decision, so it depends on the implementation. The field
`native_runs` (0 against 52,672) sits inside the deterministic part, which
is a defect of this runner inherited from `research/h6_rank5/batch.py`:
it counts how many matcher runs took the compiled path, so the
deterministic hash of a reference replay can never equal the stored one,
and the cross-check has to be read field by field, as here. The rank-4
pipeline for |T>^5 (`research/t5_rank4/batch.py`) keeps that field outside
the hash; this runner is left as run, since changing the hash definition
after the fact would invalidate the 323 stored hashes and the manifest.

## 9. What is proved, what is assumed, what is open

Proved by table or argument here: Fact 1 (given the 30-element list),
Fact 2 (a board bound), the flat lemma, the base-point case split, the
exclusion of the (2, 2) configuration, the completeness of the stage
(beta) list, the reuse of the census. Assumed from earlier work: the
completeness of the 30 rank-4 decompositions of |H>^4 up to symmetry (as
the H^6 bound), the H^6 census and degenerate enumeration (hashed,
re-enumerable), PR #87's shape enumeration. Open before the bound: the pod
run itself, its aggregate, the `--no-native` replay of one stage A batch,
and the pod rate of stage B (the partition takes twice the laptop rate;
the first stage B batch on the pod decides whether the 3.6 hours of wall
time hold). Two limitations of the stage (beta) matcher are recorded
rather than closed, both harmless for the exclusion as listed: the joint
reconstruction is written for one pair block (the list has no other
block shape) and raises otherwise, and a point term at x_a2 whose slice
lies inside a block's span (a product fifth term, excluded by Fact 1) is
counted in `point_ambiguous` and not searched.
