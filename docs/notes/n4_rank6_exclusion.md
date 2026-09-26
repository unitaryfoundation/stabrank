# Excluding rank 6 for |N>^4: the pipeline as built

Status (2026-09-26). Pipeline built and controlled on the laptop
(2026-09-24), run on the pod 2026-09-25 to 2026-09-26 (section 8.2): all
1,057 batches, 0 hits, 0 refused, 0 undecided after the stage B6 filter fix
of PR #153 and the rerun of 23 batches; the aggregate certified
chi(N^4) >= 7, filed as `bounds/N-m4-lower-7.json` at the attested tier.
With `bounds/N-m4-upper-7.json` (the Lean witness) the cell is
chi(N^4) = 7; before the run it stood at 6 <= chi(N^4) <= 7
(`bounds/N-m4-lower-6.json`, the rank-5 exclusion of
`docs/notes/qutrit_m4_rank5_exclusion.md`). The design was chosen and
measured in `docs/notes/n4_rank6_design.md`; this note is the argument as
implemented, written so that it can be checked from the files alone.

Notation follows the design note: psi_4 = |N>^4 with |N> = (1, 1, -2) /
sqrt 6, psi_2 = |N>^2, the 360 two-qutrit stabilizer states in the order
of `dictionary(3, 2)`, G_2 the unitary symmetry group of psi_2 (order 72,
acting on the dictionary by permutations), w = exp(2 pi i / 3). Slicing
along qutrits 1, 2 at x in F_3^2 gives sum_i c_i u_i^(x) = alpha_x psi_2
with alpha_x = a_{x_1} a_{x_2}, a = (1, 1, -2); from the base point
X0 = (2, 2) the ratio alpha_x / alpha_{X0} is 1/4 at the four points with
both coordinates in {0, 1} and -1/2 at the four points with one coordinate
2. Each term is nonzero exactly on an affine flat of F_3^2: a point, one of
the 12 lines, or the plane. The two-qutrit slice structure lemma (PR #86,
`research/constructions/two_qutrit_slice.py`) describes the slices of one
term on its flat from the slice at one point: a plane term has
u(X0 + x) = w^{q(x)} Q_2^{x_2} Q_1^{x_1} u(X0) with two-qutrit Paulis Q_1,
Q_2 and a quadratic q with q(0) = 0; a line term along a has
u(y_0 + t a) = w^{q(t)} Q^t u(y_0). The arithmetic is Q(omega_3) modulo
P1 = 65521 and P2 = 2013265921 (both 1 mod 3); every modular kernel lists
a superset and every candidate is decided mod P2 and numerically.

## 1. The one fact and why there is no Fact A

Fact B (at least four of the six terms are nonzero at every two-qutrit
point): a point sees at least chi(N^2) = 3 terms, and PR #86's case M
search (`docs/notes/constructions_2026_09.md`: no rank-5 or rank-6
decomposition of |N>^4 has a two-qutrit slice with exactly three nonzero
terms, 90 (decomposition, x_0) pairs and 945 coverage patterns for N, 0
exact completions) closes three. This is the only structural input.

Fact A of the rank-5 argument (every term full along every qutrit) does
not hold at rank 6: a one-qutrit slice may see five terms, a non-minimal
rank-5 decomposition of |N>^3, whose census is out of reach. So the design
drops the flat lemma and the base-point case split of the rank-5
exclusion and works at one base point with the invisible terms' flats as
parameters, as the T^5 exclusion does (`docs/notes/t5_rank5_exclusion.md`).

## 2. The case split

Base point X0 = (2, 2) for every decomposition. Suppose psi_4 = sum_{i=1}^6
c_i s_i with pairwise distinct stabilizer states and every c_i nonzero (a
repeat or a zero coefficient is rank at most 5, excluded by the board). By
Fact B the terms visible at X0 number k in {4, 5, 6}. Their base slices
u_i^(X0) form a full k-multiset of psi_2: k stabilizer states, repeats
allowed, whose span contains psi_2 with all coefficients d_i = c_i /
alpha_{X0} nonzero (a repeated state's copies may cancel there, their
merged coefficient being exempt). The 6 - k invisible terms have flats
missing X0: one of the 8 other points or one of the 8 lines not through X0.

Lemma (one base per G_2 orbit). Let U be a unitary symmetry of psi_2, so
U psi_2 = e^{i theta} psi_2 and U permutes the dictionary up to phases, and
let g = I_9 (x) U act on qutrits 3, 4. Then g psi_4 = e^{i theta} psi_4, g
carries stabilizer states to stabilizer states, and slicing along qutrits
1, 2 commutes with g: (g s)^(x) = U s^(x) for every x. So g carries a rank-6
decomposition with base multiset S at X0 and invisible flats F to one with
base U S and the same flats, and the decompositions with base S biject
with those with base U S. A matcher that finds every decomposition with a
given base need run on one base per G_2 orbit of multisets. The group acts
through the unsliced qutrits only; a symmetry of qutrits 1, 2 would move X0.
The lists hold the least element of each orbit (the least image of the
sorted tuple under the 72 permutations, `degenerate6.canonical_codes`), and
the aggregate recomputes them. Control 5 checks the lemma on planted
decompositions.

Hence every rank-6 decomposition is found by a matcher that, given a full
k-multiset at X0 and the flats of the 6 - k invisible terms, enumerates
every decomposition with that base and those flats, run over every full
k-multiset up to G_2 and every choice of flats:

- Stage A6, k = 6 distinct independent states: `matcher.Matcher.run` at
  X0 through the compiled `SliceMatch3Kernel`, called directly by the
  batch (`stages.run_a6`).
- Stage B6, k = 6 distinct dependent states: the fresh-term filter of
  `filters6.py` for kappa = 1, then the reference `Matcher.run`.
- Stage C6, a repeated state: the reference `Matcher.run` (block paths;
  `BlockOnlyMatcher` for the (2, 2, 2) multisets).
- Stage (beta'), k = 5: `invisible3.InvisibleMatcher3.run_one(cover, F)`
  for every flat F of the 16.
- Stage (gamma), k = 4: `run_two(cover, (F_4, F_5))` for every multiset of
  two flats, 136 in all.

The 16 flats (`common.FLATS`, named as in
`results/invisible3_geometry.json`): the points p00, p01, p02, p10, p11,
p12, p20, p21 and the lines L01_000102 = {x_1 = 0}, L01_101112 = {x_1 = 1},
L10_001020 = {x_2 = 0}, L10_011121 = {x_2 = 1}, L11_011220 =
{(0, 1), (1, 2), (2, 0)}, L11_021021 = {(0, 2), (1, 0), (2, 1)},
L12_001221 = {(0, 0), (1, 2), (2, 1)}, and L12_021120 =
{(0, 2), (1, 1), (2, 0)}. Nine miss both coordinate points (0, 2) and
(2, 0), six contain one, and L12_021120 contains both. The stages carry
exactly these 16 and 136 cases; Fact B also bounds the invisible terms at
every other point by two, but the matchers do not use that.

## 3. The bases

All lists come from the attested rank-5 census files
(`research/qutrit_m4_rank5/results/N/kernel_census.json`: 29 full
3-covers, 1,403 full 4-covers, 197,440 full 5-covers of distinct
independent states, one per G_2 orbit as far as the pivot and partner
reductions go; `degenerate_covers_N.json`: 12,175 dependent 5-covers and
16,181 repeated 5-multisets) and from the compiled 6-cover census.

| stage | list | count | file |
|---|---|---|---|
| A6 | full 6-covers of distinct independent states | 37,201,212 | `results/census6_N_full.json`: the per-pivot-pair counts of `cover6_pair` over the 1,209 pivot pairs of `CoverEnumerator3("N", 2)` (702,716,072 modular candidates, 413 kernel seconds); every 6-set of rank 6 mod P2; the covers themselves are regenerated per pivot pair by the batches |
| B6 | G_2 orbits of the dependent full 6-sets | 259,655 (kappa 1: 256,970; kappa 2: 2,683; kappa 3: 2) | `reps_N.json` (`driver.py lists --write`): 2,701,615 distinct full dependent 6-sets from the routes of the design note, section 2.1 (T_5 plus a span state; T_4 plus two span states or a parallel pair; T_3 plus three span states, a span state and a parallel pair, a parallel triple, or a collinear triple), enumerated mod P1 and decided numerically, canonicalized |
| C6 | G_2 orbits of the repeated full 6-multisets | 321,553 ((2, 1, 1, 1, 1) 315,229; (3, 1, 1, 1) 2,277; (2, 2, 1, 1) 3,981; (4, 1, 1) 20; (3, 2, 1) 38; (2, 2, 2) 8) | same: 1,602,177 distinct multisets from the merged-coefficient routes (a 5-cover with a member doubled; a 4-cover with a member tripled, two doubled, or plus (b, b) for any other state b; a 3-cover with multiplicities (4, 1, 1), (3, 2, 1), (2, 2, 2), or a member doubled plus (b, b), or plus (b, b, b)), canonicalized |
| beta' | G_2 orbits of the full 5-multisets | 54,488 (independent 50,719; dependent 1,452; repeated 2,317) | same: the 225,796 multisets of the rank-5 run, canonicalized |
| gamma | G_2 orbits of the full 4-multisets | 505 (independent 478; dependent 7; repeated 20) | same: the 1,403 census 4-covers, the 31 dependent 4-covers (T_3 plus a span state, full) and the 87 repeated 4-multisets (T_3 plus a member); no cancel-at-base 4-multiset exists since chi(N^2) = 3 leaves no 2-cover |

Completeness. A6 rests on the pivot and partner reductions of the census
(the H^6 argument: pivot one per orbit, partner minimal in its stabilizer
orbit, members above the partner, the residue kernel listing a superset,
every candidate decided mod P2 and numerically) and on the kernel's
`is_full` test, unchanged by the rank. B6 rests on the kappa reduction of
the design note: a full 6-set S with kappa >= 1 contains an independent
full k-cover T with k <= 5 and rank(S) <= 5, so S is T plus states of
span(T) or parallel modulo it in one of the seven routes, every route
enumerated over F_P1 (a superset) and decided numerically. C6 rests on the
merged-coefficient argument: the distinct states with nonzero merged
coefficient form a full cover C of 3, 4, or 5 distinct states and the rest
are cancelling blocks of two or more copies of any state outside C, no
fullness test applying to the blocks. The k = 5 and k = 4 lists are the
attested rank-5 lists (the cancel-at-base multisets T_3 + (b, b) included)
plus the dependent and repeated 4-multisets. `reps_N.json` records the
hashes of the two rank-5 files it was built from; the aggregate rebuilds
every list from them and requires equality (control 4).

## 4. The matchers

Stages A6, B6, C6 run the qutrit `Matcher` of the rank-5 pipeline
(`research/qutrit_m4_rank5/matcher.py`, unchanged on this branch) at X0:
the two coordinate slices against the initial coefficient family, the join,
the six composite points over the shapes of the structure lemma still
alive, the block reconstruction with a single block's family parameter as
an unknown, `UnpinnedFamily` for the shapes it does not cover. Stage A6
calls the compiled kernel directly and confirms its hits in Python
(`stages.run_a6`), reproducing `Matcher._run_native`'s counts; a cover the
kernel declines goes through `Matcher.run`.

Stage B6 with kappa = 1 runs `filters6.Filters.b6_slice` at the first
coordinate slice (design note, section 3.1): in the translate basis of the
fresh term's base slice the equation U + lambda V is supported on one
coordinate, so on each of three parts of three coordinates the
complementary six-coordinate projection is a 1-parameter dense solve over
the five ordinary terms (`dense_solve`), the union over the parts is a
superset, and every survivor is completed with the fresh term's code and
decided by `Family.restrict`, the reference's own decision: the list
equals the reference's first-slice list (planted control 1, and the
agreement runs of the design note). An empty list decides the base; a
nonempty one is filtered the same way at the second coordinate slice
(against the initial family, as the reference does); a base with solutions
at both goes through `Matcher.run`, whose raw counts must equal the
filter's or the run raises. A base the filter cannot take (no fresh term
with a nonzero dependency modulo both primes, kappa outside 1) falls back
to `Matcher.run`, counted outside the deterministic part
(`filter_fallbacks`). Kappa = 2 and 3 run `Matcher.run` with the dense
candidate cap raised to 2e8 (`stages.set_max_cand`, recorded in the
partition and in every record as `max_cand`); the kappa 3 bases take the
Python `_dense` (three parameters). The deterministic per-run key of the
three stages is `coord_raw` ([n_1], or [n_1, n_2] with both counted
against the initial family), which the filter and the reference produce
alike, so a `--no-native` replay hashes the same.

Stages (beta') and (gamma) are `research/n4_rank6/invisible3.py`; its
module docstring is the specification. The flats of the invisible terms
are parameters; nothing is assumed about the visible terms' flats: each
visible term's codes over the eight nonzero offsets range over the full
shape table of the structure lemma for its base slice (20,008 rows: 19,683
planes, four lines through X0 with 81 shapes each, the point X0), the rows
alive are restricted by every code chosen, and the codes offered at a
point are the values the alive rows take there, so the presence pattern
and the composite codes are the lemma's by construction (the p = 3 form of
the T^5 matcher's `compatible_codes`, exact rather than necessary). The
points are processed exact points first (no invisible term present), then
by fewest fresh terms; a born invisible term's coefficient joins the
coefficient family as a new coordinate restricted by the slice equation
at its birth point, so pinned bases pin it at once and dependent bases
carry it. The flats containing a coordinate point are handled by this
order without a separate matcher: their first exact point is the other
coordinate point or a composite point.

- exact (no fresh term): `matcher.solve_slice3` over the visible options,
  the block's translate sets and the born terms' options (27 phased
  translates at the second point of a line term, the three phases of the
  doubled class at the third); meet in the middle mod P1 for a pinned
  family, the compiled dense solve for one or two parameters, the Python
  reference beyond; every candidate decided over the three fields.
- scan (one fresh term): with a pinned family, all option combinations at
  once mod P1, the residual projected onto the annihilator of the block's
  translates, normalized and looked up in the table of the normalized
  projected dictionary states; with parameters, the states whose
  projection lies in the span of the residual's affine family mod P1.
  Every candidate is decided by the exact restriction of the extended
  family. A zero projected residual with no translate chosen is another
  flat's configuration (the fresh term absent at a point of its flat) and
  is skipped; with translates chosen, the states inside their span are
  candidates whose coefficient stays a family parameter, decided by the
  block reconstruction (the ambiguous case of the T^5 matcher, handled
  here by the family instead of a joint brute force; planted kind h of
  control 1).
- scan2 (two fresh terms, stage (gamma) with two equal flats): the rank-2
  residual scan of the T^5 matcher at p = 3 (projective hash of the
  dictionary modulo the residual, every collision decided mod P2 and over
  C with agreement required). For two equal line flats the states may
  coincide: r = D v is carried as two born terms with the same state and
  a free split, decided at the line's other points by the dense solve;
  r = 0 (a cancelling pair, the common state unknown) is carried to the
  line's second point, where the residual is scanned for a stabilizer
  residual (both copies in one class with distinct phases), a rank-2
  residual whose states lie in one Pauli orbit with coefficients
  cancelling up to a cube root (the common slice at the first point runs
  over the nine class translates and three phases), or zero again (the
  pair agrees at two points and differs at the third, where a stabilizer
  residual fixes it). Two equal point flats with equal states or a zero
  residual are one term twice or two cancelling copies and are rejected.

Every surviving state is assembled into six terms (the copies of a block
through `matcher.reconstruct_block` from the residual coordinates at the
eight offsets, `_block_coordinates` strict) and confirmed
(`Matcher.confirm`: residual against psi_4, rank, independence, nonzero
coefficients, the span condition mod P2); `batch.py` re-decides every hit
from its phase codes (`common.decide_terms`).

Refusals and undecided runs. A base whose family is empty or has a dead
ordinary coefficient is refused; the lists exclude such multisets, so a
refusal fails the aggregate. `UnpinnedFamily` (two fresh terms at a point
of a base with a block or an unpinned family; a family left with
parameters at the end on a base with two or more blocks; the strict
coordinate solve of the reconstruction; a residual whose complex and
modular decisions disagree; an invisible term not determined on its whole
flat; any configuration the enumeration does not cover), `BudgetExceeded`
(the dense candidate cap, the batch deadline), a hit on which the two
decisions disagree, and a run not made before `--max-seconds` are
recorded as undecided; the batch exits 1 and the aggregate fails.

## 5. Controls

All on one laptop core at nice 19 (Apple M5 Pro, 18 cores, load from
other sessions), through `research/t5_rank5/run.py` with a 600 s cap, one
process at a time. Every control is a `driver.py` command with its record
under `results/` and its log under `results/logs/`.

Control 1, planted instances (`driver.py control-planted --stage STAGE`,
`results/control_planted_{b6,c6,beta_adr,beta_h,gamma}.json`): the target
is the sum of the planted terms with integer coefficients, the matcher runs
against the target's slices, and the planted decomposition must be among
the hits. Stage B6: three kappa = 1 orbit representatives through the
filter path (first slice, second slice, then the reference) and one kappa =
2 representative at the raised cap, 4 of 4 recovered in 3 to 7 s. Stage C6
by class: (2, 1, 1, 1, 1) independent and dependent, (2, 2, 2) through
`BlockOnlyMatcher`, (3, 1, 1, 1) independent and dependent: 4 of 5
recovered (one (3, 1, 1, 1) dependent instance capped at 120 s); the
(2, 2, 1, 1), (3, 2, 1) and (4, 1, 1) classes are not planted, since a
planted target with two blocks or a block of four keeps every slice alive
past the cap (the rank-5 note met the same limit); their real bases run in
the sample and the test batches below. Stage (beta') at every one of the
16 flats, four kinds each: (a) an independent 5-cover with five random
visible shapes and a random invisible state on the flat (a point term, or
a line term with random class and phases), (d) a dependent (kappa 1)
5-cover, (r) a repeated (2, 1, 1, 1) base, (h) a repeated base whose block
copy shares its slice at the flat's first point with the invisible term
(the block-hidden case, the invisible coefficient a family parameter until
the reconstruction): 64 of 64 recovered, kinds a, d, r in 0.1 to 2 s and
kind h in 1 to 57 s. Stage (gamma) at every one of the 136 flat multisets
(kind a, random invisible terms), the six (point, line) multisets with the
point on the line (ray: the point term's slice the ray of the line term's
slice there), and for the eight shared lines the kinds same (equal slices
at the first point, different classes at the second), same1 (one class,
different phases), same2 (equal at the first two points, different phases
at the third), samec (one class with phases and coefficients cancelling at
the second point), cancel, cancel1 and cancel0 (the same three shapes with
cancelling coefficients): 216 of 216 recovered in 21 s. One defect was
found and fixed by these plants before the record: the cancelling pair's
rank-2 continuation took the common first-point slice as the dictionary
state itself, dropping the phase that the second-point codes carry, and
missed the cancel kind on seven of the eight lines.

Control 2, the rank-7 witness (`driver.py control-witness --cap 25`,
`results/control_witness.json`). `bounds/N-m4-upper-7.json` sliced along
every qutrit pair has every term present at (2, 2) (each fixed coordinate
of its terms is 2), so its six bases are all-visible 7-multisets: base 0
with three doubled states and kappa 1, base 1 with two doubled states and
kappa 2, and four bases of seven distinct states with kappa 3. Bases 0 and
1 abort at the 25 s cap at the first coordinate slice (the seven-term
dense solve with blocks; the design note's control-witness of the rank-5
pipeline passed one base and refused the kappa 3 ones at 8 GB), and the
four kappa 3 bases are not run (the three-parameter dense solve is the
Python reference, minutes per slice). Recorded as 0 of 6, 2 aborted, 4
not run, a control failure at this cap, never a pass; the seven-term
shapes with kappa 2 and 3 are on no rank-6 path (stage B6 has kappa 1
bases through the filter and kappa 2 and 3 bases through the same
reference, which the planted kappa 2 instance and the test batches
exercise).

Control 3, the rank-4 decompositions of |N>^3 along a qutrit pair
(`driver.py control-m3`, `results/control_m3.json`): the stored
decomposition (`research/constructions/data/N_m3_rank4.json`) sliced along
each of the three qutrit pairs at (2, 2) has all four terms visible (base
(2, 3, 8, 10) of single-qutrit states), and the qutrit `Matcher` at n_2 = 1
recovers it three times out of three; no configuration with an invisible
term arises, so this control does not reach the invisible matcher (the
design note named it as a control of the same code; the rank-5
decompositions of |N>^3 that would exercise the invisible flats have no
stored list).

Control 4, the lists (`driver.py control-lists`, `results/control_lists.json`,
70 s): fresh enumerations of the B6, C6, k5 and k4 orbit representatives
(with the B6 family dimensions) equal `reps_N.json`; the census file's
1,209 rows are the enumerator's pivot pairs in its order, sum to 37,201,212
and hold no dependent 6-set. The aggregate's dry run below re-ran
`cover6_pair` over every pair with every count equal (516 s) and
re-enumerated the lists again (75 s).

Control 5, the orbit lemma (`driver.py control-orbit`,
`results/control_orbit.json`): the 72 unitaries of G_2 (the closure of the
0 <-> 1 swap of |N> on either copy and the copy swap, built from
`rank_exclusion.clifford_group`) induce exactly the census's permutation
group of order 72; for six planted decompositions (three all-visible over
a census 5-cover plus a sixth state, three with one invisible term on a
random flat over a k5 representative), twelve random elements each carry
the terms to stabilizer states whose base is the permuted base with the
same canonical form, and for a monomial element (8 of the 72 have cube-root
entries, so the image target's amplitudes stay in Z[w]) the matcher on the
image base against the image target finds the image decomposition, 6 of 6.

Control 6, the A6 loop against `Matcher.run` (`driver.py sample A6`,
`results/sample_A6.json`): on 100 of the 20,000 covers of the rate sample
the direct kernel loop and `Matcher.run` agree on hits, refusals and
`coord_raw`; on all 20,000 both die at the first coordinate slice.

## 6. Rates, partition, and the pipeline test

Measured by `driver.py sample STAGE` (`results/rates.json`,
`results/sample_*.json`), per item on real items, one process at nice 19
with a load average of 9 to 10 from other sessions (the design note's
rates were taken at a lower load; stage B6 runs at twice its rate here).
The sample warms the option and shape tables on a first pass and times a
second, since a batch of thousands of items runs with them warm; a
per-item cap of 40 s marks tail items, which the partition's rate excludes
and the batch guard covers.

| stage, class | items | per item (laptop) | pod seconds per item (compiled 1.3, Python 4) | test batch: items, seconds |
|---|---|---|---|---|
| A6 | 37,201,212 covers | 0.63 ms through the kernel loop (kernel 0.07 ms, `Matcher.run` 0.68 ms in the design note) | 0.82 ms, plus 1.3 x the census kernel seconds per pair | batch 0 (pivot pair (117, 0)): 753,064 covers, 499 s |
| B6 kappa 1 | 256,970 | 0.55 s (dense 0.21 s; 0.27 s in the design note) | 1.66 | batch 54: 362 items, 187 s |
| B6 kappa 2 | 2,683 | 5.2 s at cap 2e8 | 21.0 | batch 765: 29 items, 180 s |
| B6 kappa 3 | 2 | 397 s (Python dense, cap 2e8) | 1,588 | batch 859: 1 item, 434 s |
| C6 (2, 1, 1, 1, 1) | 307,822 | 0.048 s (median 0.008 s) | 0.19 | batch 861: 3,142 items, 38 s (0.012 s per item) |
| C6 (2, 1, 1, 1, 1) dependent | 7,407 | 0.92 s | 3.7 | |
| C6 (2, 2, 1, 1) | 3,945 | 0.21 s trimmed (1 of 10 past 40 s; 325 s once in an earlier sample) | 0.84 | |
| C6 (2, 2, 1, 1) dependent | 36 | both sampled items past 40 s | 160 (the cap) | |
| C6 (2, 2, 2), (3, 1, 1, 1), (3, 1, 1, 1) dependent, (3, 2, 1), (4, 1, 1) | 8, 2,253, 24, 38, 20 | 0.35, 0.014, 5.5, 6.7, 0.31 s | 1.4, 0.06, 22, 27, 1.2 | |
| beta', 16 flats | 50,719 independent; 1,452 dependent; 2,253 + 24 + 20 + 20 repeated | 0.040 s; 0.79 s; 0.08 to 2.5 s | 0.16; 3.2; 0.3 to 10 | batch 1027: 3,623 items (57,968 runs), 177 s |
| gamma, 136 flat multisets | 478; 7; 20 | 0.078 s; 0.41 s; 0.42 s | 0.31; 1.6; 1.7 | batch 1054: 478 items (65,008 runs), 51 s |

Every sampled run and every test-batch run died without a hit: no
refusal, no undecided run. Stage (beta') and (gamma) runs die at their
first exact point in every case ('0,0,0,0' histogram); a few gamma runs of
the repeated bases had three solutions at their first exact point and died
at the second. B6 kappa 1 bases had first-slice solutions in 19 of 70
sampled (2 to 158), all dying at the second slice or the composite stage;
C6 bases had up to 245 first-slice solutions.

The partition (`driver.py partition --target-s 600`, `partition.json`,
sha256 `597eca221ac6d6b3`): stage A6 pivot pairs grouped by 1.3 x (census
kernel seconds + 0.63 ms x covers), a batch closing before a pair would
carry it past 600 pod seconds (the pivot 117 pairs sit alone at about 500
s); stages B6, C6, beta' and gamma round-robin within each cost class at
the pod rate per item above, B6 kappa 3 as single-item batches, with the
candidate cap 2e8 recorded in the geometry of every kappa 2 and 3 batch.

| stage | batches | indices | per batch | estimated pod CPU-hours |
|---|---|---|---|---|
| A6 | 54 | 0 to 53 | 1 to 190 pivot pairs | 8.6 |
| B6 kappa 1 | 711 | 54 to 764 | 361 or 362 items | 118.4 |
| B6 kappa 2 | 94 | 765 to 858 | 28 or 29 items | 15.7 |
| B6 kappa 3 | 2 | 859, 860 | 1 item | 0.9 |
| C6 | 166 | 861 to 1026 | by class: 98 batches of 3,141 or 3,142 (2, 1, 1, 1, 1) items, 46 of 161 dependent ones, 6 + 10 of the (2, 2, 1, 1) classes, 6 for the rest | 26.9 |
| beta' | 27 | 1027 to 1053 | by class: 14 batches of 3,622 or 3,623 independent bases, 8 of 181 or 182 dependent, 5 for the repeated classes | 3.8 |
| gamma | 3 | 1054 to 1056 | 478, 7, 20 bases | 0.05 |
| total | 1,057 | | | 174 |

The B6 kappa 1 estimate is the whole uncertainty: at the sampled laptop
rate with the compiled factor on the dense share and the Python factor on
the rest it is 118 pod CPU-hours; at the compiled factor throughout it
would be 48, and at the design note's lower-load rate 25 to 39. The
first B6 batches on the pod settle it before the rest is launched, as
the T^5 note's section 8 does for its stage B. The C6 estimate carries
the cold-cache inflation of the (2, 1, 1, 1, 1) sample (0.048 s against
0.012 s in the test batch) and the two tail classes at the cap; the test
batch projects C6 at about 5 pod CPU-hours. Wall time on 15 processes: 6
to 12 hours.

Pipeline test (one laptop core at nice 19, load 9 to 10), the first batch
of every stage and class into `results/`: batch 0 (A6, 753,064 covers,
kernel 5 s, match 493 s), 54 (B6 kappa 1, 362 items, 187 s), 765 (kappa 2,
29 items, 180 s), 859 (kappa 3, 1 item, 434 s), 861 (C6 (2, 1, 1, 1, 1),
3,142 items, 38 s), 1027 (beta' independent, 3,623 bases, 57,968 runs,
177 s), 1054 (gamma independent, 478 bases, 65,008 runs, 51 s): 0 hits, 0
refused, 0 undecided in every one, every B6 kappa 1 item decided by the
filter (no fallback). `aggregate.py --dry-run --partial`
(`results/logs/aggregate_dry.log`) re-enumerated the lists (equal), passed
every stored check on the seven records, wrote
`batch_manifest.partial.json`, and projected the stage totals at the
observed laptop rates: A6 6.8 CPU-hours, B6 147 (the three batches over
all three kappa classes), C6 1.1, beta' 0.7, gamma 0.01. The same dry run
against the previous partition re-ran the 6-cover census with every pair's
count equal (516 s); it is skipped here with `--no-reenumerate-census` to
stay under the 600 s cap and runs by default on the pod. The laptop records
of the seven batches are committed under `results/`, so the pod loop skips
them.

## 7. Soundness checklist

From `docs/notes/qutrit_m4_rank5_review.md` and the T^5 note's section 7;
status of each item for this pipeline.

1. The base-point reduction: every rank-6 decomposition has a full 4-, 5-
   or 6-multiset base at (2, 2) with the invisible terms on flats missing
   (2, 2). From Fact B (PR #86's tables, declared, not re-run) and the flat
   enumeration; the 16 flats and 136 flat multisets are exactly the cases
   the stages carry. Closed.
2. The k = 6 lists are complete: A6 from the census (the H^6 argument), B6
   from the kappa reduction with every route enumerated mod P1 and decided
   numerically, C6 from the merged-coefficient argument with cancelling
   blocks of any state outside the full cover. The aggregate re-runs the
   census and re-enumerates the lists (control 4). Closed given the
   enumerator's pivot and partner reductions.
3. Orbit representatives: the canonical form is the least image under the
   closure of the census's generators (order 72 asserted), the lemma is
   section 2, the aggregate recomputes the representatives, and control 5
   checks the lemma on planted decompositions with the unitaries of G_2
   inducing the census's permutations. Closed.
4. The B6 filter is a necessary condition on the first coordinate slice
   whose survivors are decided by the reference's `restrict`, the second
   slice is filtered the same way, a base surviving both runs the unchanged
   reference, and the reference's `coord_raw` must equal the filter's:
   planted control 1 (kappa 1), the design note's 6 of 6 planted and real
   agreement, and the test batch. Closed.
5. The candidate cap is raised to 2e8 for kappa 2 and 3, recorded in the
   partition and every record; a base refused at the cap is undecided and
   fails the aggregate; the kappa 3 bases run the Python dense solve in
   single-item batches (434 s each on the laptop). Closed.
6. Stages (beta') and (gamma) assume nothing about the visible terms'
   flats beyond the base point (the full shape table), allow the absent
   option everywhere, decide the scans mod P1 as supersets with the exact
   restriction of the extended family, and enumerate the same-state and
   cancelling cases of two line flats (planted kinds same, same1, same2,
   samec, cancel, cancel1, cancel0 on all eight lines). Closed for the
   listed configurations; two fresh terms at a point of a base with a
   block (the 20 repeated 4-multisets on the 16 same-flat multisets) or
   with an unpinned family raise, and none of the 2,720 and 952 such runs
   of the sample and the test batch reached a scan2 point (every run died
   at its first exact point).
7. Repeated states, cancelling copies, dependent translates and unpinned
   families take the rank-5 matcher's paths (`reconstruct_block` with the
   family parameter for one block, the strict coordinate solve,
   `BlockOnlyMatcher`); a raise is recorded as undecided and fails the
   aggregate. The two-block bases with dependent translates ((2, 2, 1, 1)
   dependent, 36 orbits) are the tail of stage C6 and can raise as the 27
   rank-5 covers could; none did in the sample. Open until the run.
8. Every hit is re-decided from its phase codes mod P2 and numerically,
   distinct terms and rank 6 required by `genuine`; the aggregate
   re-decides every stored hit. Closed.
9. Deterministic hashes exclude the kernel-run, filter-path, candidate,
   raw feature-zero and fallback counts, and the histogram key of stages
   A6, B6, C6 is `coord_raw`, which the kernel, the filter and the
   reference produce alike; the aggregate fails a record that carries any
   of those counts inside the deterministic part. The `--no-native` replay
   of one A6 batch and one B6 batch is a pod task (section 8); open until
   run.
10. `--max-seconds` records runs not made as undecided with a deadline
    reason, the running item is aborted at the deadline through the
    matcher's budget and the invisible matcher's deadline, and `--resume`
    redoes such a record. Closed (the batch runner; not exercised on a
    real deadline here).
11. Controls: planted instances for B6 (kappa 1 and 2), C6 (five of nine
    classes), every one of the 16 flats (four kinds) and every one of the
    136 flat multisets (with the shared-line kinds); the witness at its six
    all-visible bases recorded as aborted or not run (control 2, on no
    rank-6 path); the m = 3 control on the all-visible bases only; the
    lists and census by hash; the orbit lemma. Every control ran at the
    commit that carries the final `invisible3.py` except the b6 and
    beta_adr plants, which ran before the cancelling-pair fix, a branch
    they do not touch.
12. No `id()`-keyed caches: the scan tables are keyed by the block states
    and translate sets, the shape and option tables by dictionary index,
    the reference's compatibility index by the solution list object.
13. The census's `is_full` decisions and the lists' numerical decisions are
    the rank-5 pipeline's, decided mod P2 and numerically with agreement
    where the two are compared (`is_cover`), and the aggregate's fresh
    enumeration repeats them.

## 8. The pod run

The pod checkout is `/root/stabrank-h6`; the branch is fetched and checked
out, never `git checkout -f` (it would overwrite result files); the pod's
default g++ 9 lacks the C++20 headers, so a fresh build of the extension
runs with `CC=gcc-10 CXX=g++-10`; runs go under `setsid nohup ... <
/dev/null & disown`; the batch loop is resumable (a finished record is
skipped, one left incomplete by the deadline is redone with `--resume`).

```
ssh -i ~/.ssh/id_ed25519 -p 40096 root@157.157.221.30
cd /root/stabrank-h6
git fetch origin n4-rank6-pipeline && git checkout n4-rank6-pipeline && git pull --ff-only
CC=gcc-10 CXX=g++-10 /root/.local/bin/uv sync --reinstall-package stabrank --extra challenge
/root/.local/bin/uv run --extra challenge python -c "import stabrank.stabrank_core as c; print(c.cover6_pair, c.SliceMatch3Kernel, c.dense_solve)"
/root/.local/bin/uv run --extra challenge python -m pytest tests/test_kernels_k6_p3.py -q -x
mkdir -p research/n4_rank6/results /root/logs
```

Rate check first (the second batch of each stage and class, the first ones
having run on the laptop as the pipeline test), in a `setsid nohup` shell,
about half an hour; then the aggregate's `--partial` projection tells
whether the B6 factor is the compiled or the Python one before the whole
partition is launched:

```
setsid nohup sh -c 'for K in 1 55 766 860 862 1028 1055; do
  nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/batch.py $K --max-seconds 3600 \
    > research/n4_rank6/results/batch_$K.log 2>&1; done' < /dev/null > /root/logs/n4_probe.log 2>&1 & disown
nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/aggregate.py --dry-run --partial --no-reenumerate --no-reenumerate-census
```

The full run, 15 processes over the 1,057 batches (the laptop records of
batches 0, 54, 765, 859, 861, 1027 and 1054 are committed under `results/`
and are skipped by the loop; the guard of 3,600 s is six times the batch
estimate, and a batch that hits it is redone by the same loop with
`--resume`):

```
setsid nohup sh -c 'seq 0 1056 | xargs -P 15 -n 1 sh -c \
  "nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/batch.py \"\$0\" --resume --max-seconds 3600 \
   > research/n4_rank6/results/batch_\"\$0\".log 2>&1"' < /dev/null > /root/logs/n4_run.log 2>&1 & disown
```

Rerunning the same command resumes. Progress, the batches that hit the
guard (rerun each with `--force` and no guard), the final check (what the
certificate runs), and the `--no-native` replays of the smallest A6 batch
(53, the light pairs) and of one B6 kappa 1 batch (checklist item 9; the
Python references for the 6-cover kernel, the stage A matcher and the
dense solve, the B6 replay through the Python `_dense` about 20 times
slower):

```
grep -l "DECOMPOSITION FOUND\|undecided run" research/n4_rank6/results/batch_*.log
grep -l "deadline" research/n4_rank6/results/batch_*.log
nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/aggregate.py --dry-run --partial
nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/aggregate.py --recheck 2 --recheck-seed 20260925
setsid nohup sh -c 'STABRANK_NO_NATIVE=1 nice -n 19 /root/.local/bin/uv run --extra challenge python \
  research/n4_rank6/batch.py 53 --out-dir research/n4_rank6/results/nonative --force; \
  STABRANK_NO_NATIVE=1 nice -n 19 /root/.local/bin/uv run --extra challenge python \
  research/n4_rank6/batch.py 764 --out-dir research/n4_rank6/results/nonative --force' \
  < /dev/null > /root/logs/n4_nonative.log 2>&1 & disown
```

Copy back `research/n4_rank6/results/batch_*.json`, the logs,
`batch_manifest.json` and the no-native records with rsync over port
40096, commit them, fill the placeholders of
`research/n4_rank6/N-m4-lower-7.json.draft` (compute hours, hardware,
dates, the batch indices of the re-runs) and move it to
`bounds/N-m4-lower-7.json`; the submissions workflow verifies every touched
bound, so the draft keeps its suffix until the manifest exists. Then run
`verify_challenge/cert_n_m4_rank6_attested.py` and update the board:
chi(N^4) = 7.

### 8.1 The undecided B6 items of the first pod run

The full run of 2026-09-25 finished all 1,057 batches with 0 hits, and 24
stage B6 runs (kappa 1) in 23 batches (101, 131, 191, 202, 231, 233, 250,
278, 281, 293, 302, 304, 326, 380, 404, 423 with two, 437, 451, 453, 470,
503, 551, 583) ended undecided with the run_b6 assertion `filter and
reference disagree on the coordinate-slice counts`, for instance the cover
(8, 41, 117, 118, 249, 334) with the filter's [2, 2] against the
reference's [56, 56]. The runner recorded them as undecided, so the
exclusion was incomplete on exactly those 24 items.

Cause. On these bases every coordinate-slice solution pins the family
parameter lambda at the root of the fresh term's coefficient, d_j(lambda)
= 0: the six-set is dependent and the target's slice is a combination of
the other five states alone. The fresh term then contributes nothing, so
for a five-term combination solving the slice every one of the fresh
term's 28 codes (27 phased translates and absent) is a solution, and the
reference `solve_slice3` lists all 28 (its join drops the zero
coefficient afterwards, which is why `zero_coefficient` equals the raw
count on these items). The filter's whole-vector test skipped a
vanishing fresh coefficient with `if mu == 0 or dj == 0: continue`,
keeping only the absent code, so its list was short by 27 per such
combination: 56 = 2 + 2 x 27, 58 = 4 + 2 x 27, 60 = 6 + 2 x 27, 280 =
10 + 10 x 27, 454 = 22 + 16 x 27 on the 24 items. The skip was sound as
an exclusion (a decomposition with a zero coefficient has rank 5), but it
broke the contract of section 3.1 of the design note, that the filter's
list is the reference's first-slice list, and the deterministic key
`coord_raw` is that raw count. The filter was the wrong side; the
reference and the assertion were right.

Fix (`filters6.Filters.b6_slice`): when d_j(lambda) = 0 mod P1 and the
residual coordinate mu is 0, the filter adds all three phases at that
coordinate (and the absent code as before), each decided by
`Family.restrict` as every other candidate; mu = 0 with d_j != 0 is still
no solution. The assertion stays. `tests/test_n4_rank6_b6_filter.py`
checks five of the 24 covers against the reference list at both slices
(all 28 fresh codes present, run_b6 deciding with the reference's counts),
a planted five-term target over a base minus its fresh term (the
zero-coefficient shape itself, filter and reference agreeing), and planted
six-term decompositions over three of the bases recovered through the
fixed filter.

Local check at the fix. All 24 items decide on the laptop through
`stages.run_b6` on the compiled path (`filter+reference`, 2.2 to 2.5 s
each, counts equal to the reference's, 0 hits; the survivors of the
second slice are dropped at the join as zero-coefficient states, the
remaining joined states end in the composite stage with no hit), two of
them again with `STABRANK_NO_NATIVE=1` with the same counts and 0 hits.
The laptop record of the unaffected batch 54 (B6 kappa 1, 362 items, 39
`filter+reference` runs) rerun with `--force` reproduces its committed
`deterministic_sha256`: the fix changes no decided record, only the
formerly undecided ones.

Pod rerun. On the pod with the fix applied, one process per batch at nice
19 (twelve at a time), then the final aggregate:

```
git fetch origin && git checkout n4-b6-filter-fix && git pull --ff-only
setsid nohup sh -c 'for K in 101 131 191 202 231 233 250 278 281 293 302 304 326 380 404 423 437 451 453 470 503 551 583; do
  nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/batch.py $K --force \
    > research/n4_rank6/results/batch_$K.log 2>&1; done' < /dev/null > /root/logs/n4_rerun.log 2>&1 & disown
grep -l "DECOMPOSITION FOUND\|undecided run" research/n4_rank6/results/batch_*.log
nice -n 19 /root/.local/bin/uv run --extra challenge python research/n4_rank6/aggregate.py --recheck 2 --recheck-seed 20260925
```

The 23 records replace the undecided ones (each about 6 to 7 minutes),
the aggregate's recheck re-runs two batches of the whole pool from
scratch at the fixed seed and compares their deterministic hashes as
before, and the manifest hashes every record file. As run, the pod
checkout stayed at the pipeline commit `a844cec` with the fix
(`filters6.py` of PR #153, byte for byte) applied to the working tree, so
all 1,057 records carry `a844cec` in `git_commit`; the 23 rerun records
are told apart by `started` (06:12 UTC onward) and `max_seconds` (7200
against 3600).

### 8.2 The run (2026-09-25 to 2026-09-26)

All 1,057 batches ran on the RunPod pod (16 vCPU AMD EPYC 4564P on a
shared host, gcc-10 kernels, Python 3.12, numpy 2.3.5), fifteen at a time
under `batch.py K --resume --max-seconds 3600`, from 21:04:22 to 05:40:30
UTC; the seven pipeline-test batches (0, 54, 765, 859, 861, 1027, 1054)
were regenerated by the loop and reproduce their laptop deterministic
hashes. No batch hit the guard and no stage B6 base fell back from the
filter to the reference. Totals from the aggregate:

| stage | batches | items | matched runs | CPU-h | per batch | per item |
|---|---|---|---|---|---|---|
| A6 | 54 | 37,201,212 covers | 37,201,212 | 12.07 | 804 s | 1.17 ms |
| B6 | 807 | 259,655 | 259,655 | 95.27 | 425 s | 1.32 s |
| C6 | 166 | 321,553 | 321,553 | 12.31 | 267 s | 138 ms |
| beta' | 27 | 54,488 | 871,808 (16 flats) | 1.87 | 249 s | 123 ms |
| gamma | 3 | 505 | 68,680 (136 flat multisets) | 0.03 | 38 s | 228 ms |

121.5 CPU-hours in all (the partition estimated 174 at the pod factors:
A6 ran slower than estimated, B6 and C6 faster), 0 hits, 0 refused. The
first aggregate (`--recheck 2 --recheck-seed 20260925`, 05:40 to 05:50
UTC, `results/logs/n4_agg.log`) re-ran the 6-cover census over the 1,209
pivot pairs (536 s, every count equal), re-enumerated the four lists (76 s,
equal to the stored ones), found every stored hash and geometry in order,
and refused on the 24 undecided B6 runs of section 8.1 (`NOT CERTIFIED:
23 problem(s)`). The 23 batches were rerun with the fix (`--force
--max-seconds 7200`, twelve at a time, 06:12:12 to 06:26:18 UTC, 2.5
CPU-hours, all 24 items decided with the reference's counts, 0 hits). The
second aggregate (06:26 to 06:46 UTC, `results/logs/n4_agg2.log`) repeated
the census re-run (596 s) and the list re-enumeration (83 s), verified
every stored batch (all present, hashes and geometry matching the
partition, the stage A6 units the enumerator's pivot pairs, the census and
the lists tiled exactly once, no refused or undecided run, no hit), re-ran
batches 168 (stage B6, 255 s) and 799 (stage B6, 283 s) from scratch with
matching deterministic hashes, wrote `batch_manifest.json` (1,057
batches), and printed `CERTIFIED chi(N^4) >= 7` in 1,223 s. With the Lean
rank-7 witness `bounds/N-m4-upper-7.json` the cell is chi(N^4) = 7, filed
as `bounds/N-m4-lower-7.json` at the attested tier (the declared budget is
the schema's cap of 3,600 s; the certificate's own run is the second
aggregate without the manifest write).

### 8.3 The no-native cross-check (2026-09-26)

Batches 53 (stage A6, 245 pivot pairs, 451,695 covers, the light class)
and 764 (stage B6, kappa 1, 361 items) were re-run on the pod with
`STABRANK_NO_NATIVE=1`, both started at 05:50 UTC when the first
aggregate refused and finished at 08:02 and 09:22 UTC: for batch 53 the Python 6-cover
enumerator and the Python stage A matcher (7,927 s, kernel 1,241 s and
matching 6,685 s, against 528 s compiled); for batch 764 the Python
reference matcher on every item with no fresh-term filter at all
(`b6_paths` `{'reference': 361}` against the stored `{'filter': 320,
'filter+reference': 41}`, 12,720 s against 420 s), since the no-native
switch drops the compiled dense solve the filter is built on rather than
replacing it by a Python one. Both records agree with the stored ones on
every one of the 22 fields of the deterministic part and reproduce the
stored `deterministic_sha256` exactly (`c7ee0d82c5576fb0` and
`3355d307f18b64e4`): the same items and matched counts, the same
coordinate-slice solution histograms, 0 hits, 0 refused, 0 undecided. For
batch 764 this is also a whole-batch check of the fresh-term filter after
the fix of section 8.1: the filter's path and the reference's path give
the same deterministic record on all 361 bases. The fields outside the
deterministic part differ as they must: the modular candidate count
(3,336,686,577 against 3,332,836,501 on batch 53, the superset each
kernel's random hash functional lets through; 1,926 against 215,190,216 on
batch 764, the filter's exact slice solutions against the reference's
modular candidates), `dense_raw` (0 without the filter), `native_runs` (0
against 451,695 and 320), and the timings. Unlike the H^5 runner
(`docs/notes/h5_rank5_exclusion.md`, section 8.2), this runner keeps
`native_runs` and the path counts outside the hash, so the replay hashes
can be compared directly. The records and their logs are
`results/nonative/batch_53.json`, `batch_764.json`, and the two `.log`
files; the pod's chain log is `results/logs/n4_chain.log`.

## 9. What is proved, what is assumed, what is open

Proved by argument or table here: the base-point reduction (every rank-6
decomposition has a full 4-, 5- or 6-multiset base at (2, 2) with the
invisible terms on flats missing (2, 2)), the orbit lemma and the
completeness of the 16 and 136 invisible cases, the completeness of the B6
and C6 lists given the rank-5 census files, the k5 and k4 lists given the
attested rank-5 lists. Assumed from earlier work: Fact B (PR #86's case M
tables), the pivot and partner reductions of the enumerator (the H^6
argument) and the compiled `cover6_pair` against its Python reference
(`docs/notes/kernels_k6_p3.md`), the two-qutrit slice structure lemma
(PR #86), the rank-5 matcher with its block repairs
(`docs/notes/qutrit_m4_rank5_review.md`), the fresh-term filter's lemma
(design note, section 3.1, with the vanishing-coefficient case of section
8.1). Closed by the run (section 8.2): the pod run itself, its aggregate,
and the B6 pod rate (95 CPU-hours against the 50 to 120 the first batches
left open); the `--no-native` replays are section 8.3. Recorded rather
than closed: the
witness bases (aborted at the cap, or not run at kappa 3, on no rank-6
path); the (2, 2, 1, 1) tail of stage C6 (two items past 40 s of 12
sampled in the independent class, both sampled dependent items past the
cap), which the guard and the resume carry and which can raise the strict
coordinate solve as the rank-5 tail could; the two-fresh-term scan on a
repeated base (raises; no such run reached it); the C6 classes not
planted ((2, 2, 1, 1), (3, 2, 1), (4, 1, 1)); the m = 3 control reaching
the all-visible matcher only.
