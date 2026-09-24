# Excluding rank 5 for |T>^5: design note

Status (2026-09-24). Pipeline built and controlled on the laptop; the pod
run has not been launched, so nothing here is a bound. The cell is
5 <= chi(T^5) <= 6 (`bounds/qubit_T-m5-lower-5.json`, the rank-4 exclusion
of `docs/notes/t5_rank4_exclusion.md`; `bounds/qubit_T-m5-upper-6.json`,
the product witness at the Lean tier). A clean run of the batches of
`research/t5_rank5/` settles chi(T^5) = 6 and, by projection monotonicity,
chi(T^6) = 6 (that cell holds 5 <= chi <= 6 from the projected lower bound
and its own Lean witness). The design was chosen in
`docs/notes/t5_rank5_feasibility.md`; this note is the argument as
implemented, written so that it can be checked from the files alone.

Notation follows the rank-4 note: |T> = cos(beta)|0> + e^{i pi/4}
sin(beta)|1> with cos(2 beta) = 1/sqrt 3, psi_m = |T>^m, tau = e^{i pi/4}
tan(beta) the ratio between the amplitudes of |T>, N_3 = 1080 the
three-qubit stabilizer states in the order of `dictionary(2, 3)`, G_3 the
unitary symmetry group of psi_3 (order 162, 20 orbits on the dictionary).
A rank-5 decomposition is psi_5 = sum_{i=1}^5 c_i s_i with pairwise
distinct stabilizer states s_i and every c_i nonzero (a decomposition with
two equal states or a zero coefficient has at most four terms, which
`qubit_T-m5-lower-5.json` excludes). Slicing along qubits 1, 2 at x in
F_2^2 gives sum_i c_i u_i^(x) = alpha_x psi_3 with alpha_x = tau^{|x|}
alpha_00 (never zero) and u_i^(x) = (<x| (x) I) s_i; the points are
numbered x = 2 x_1 + x_2, so 1 = 01, 2 = 10, 3 = 11. Each term is nonzero
exactly on an affine flat of F_2^2: a point, one of the six lines, or the
plane P. The lines are A = {00, 11} and B = {01, 10} (the diagonals), C0 =
{00, 01} and C1 = {10, 11} (x_1 fixed), D0 = {00, 10} and D1 = {01, 11}
(x_2 fixed); p_x is the point x. The slice structure lemma
(`research/constructions/two_qubit_slice.py`, PR #87) describes the slices
of one term on its flat from the slice at one point: along each direction
a fourth root of unity times a Pauli on the unsliced qubits, on a composite
point the class product with one quadratic sign. The arithmetic is the
Q(zeta_24) field of `slice_cover.Field(p, "qubit_T")` mod 65521 and
2013265921, as in the rank-4 exclusion (both primes are 1 mod 24, every
modular kernel lists a superset, every candidate is decided mod
2013265921 and numerically).

## 1. The one fact and why there is no Fact 1

Fact 2 (at most two absent terms at any point of F_2^2): the visible terms
at a point decompose a nonzero multiple of psi_3, and chi(T^3) = 3
(`bounds/qubit_T-m3-lower-3.json`). This is the only structural input.

The H^5 argument's Fact 1 (every term full along every qubit) does not
transfer: chi(T^4) = 3, so a one-qubit slice of a rank-5 decomposition may
see three terms (closed by the rank-2 table of the feasibility note, section
1) or four (a non-minimal rank-4 decomposition of |T>^4, whose census over
the 36,720 four-qubit states is not run). The design therefore drops the
flat lemma and the base-point case split and works at one base point with
the invisible terms' flats as parameters.

## 2. The case split

Base point 00 for every decomposition. By Fact 2 the terms visible at 00
number k in {3, 4, 5}. Their base slices (u_i^(00)) form a full k-cover of
psi_3: k stabilizer states, repeats allowed, whose span contains psi_3
with all coefficients d_i = c_i / alpha_00 nonzero. The 5 - k invisible
terms have flats missing 00: one of the six flats p_01, p_10, p_11, B, C1,
D1. G_3 acts on the unsliced qubits, commutes with the slicing, and carries
rank-5 decompositions of psi_5 to rank-5 decompositions, so one base per
G_3 orbit suffices. Hence every rank-5 decomposition is found by a matcher
that, given a full k-cover at 00 and the flats of the 5 - k invisible
terms, enumerates every decomposition with that base and those flats, run
over every full k-cover up to G_3 and every choice of flats. No flat lemma
and no choice of base point is needed, and the swap symmetry of the sliced
qubits is not used.

The flat configurations Fact 2 allows (`research/t5_rank5/probe.py flats`,
`results/flats.json`): 124 multisets of five flats with at most two
absent terms at every point; by the number absent at 00 they split 16
(all visible), 45 (one invisible), 63 (two invisible), and the invisible
multiset at 00 takes 28 values: none, each of the six flats alone, and each
of the 21 pairs. The stages carry exactly these 28 cases:

- Stage (alpha), k = 5: the base is a full 5-cover of psi_3 (kinds A, B,
  C: distinct independent, distinct dependent, a repeated state, the
  cancel-at-base multisets T + (b, b) included). Matcher:
  `slice_cover.SliceMatcher(E, 2)` at 00, unchanged; it assumes nothing
  beyond the base point (every subspace of F_2^2 through 00 is allowed for
  every term).
- Stage (beta'), k = 4: the base is one of the full 4-covers of psi_3; one
  invisible term with flat F in {p_01, p_10, p_11, B, C1, D1}. Matcher:
  `invisible.InvisibleMatcher.run_one(cover, F)`.
- Stage (gamma), k = 3: the base is one of the full 3-covers; two invisible
  terms with flats (F_4, F_5), a multiset of two of the six flats, 21 in
  all. Matcher: `InvisibleMatcher.run_two(cover, (F_4, F_5))`. Three visible
  terms are always distinct and independent (a repeated or dependent
  triple would put psi_3 in the span of two stabilizer states), so the
  coefficients are a point.

Twelve configurations have two absent terms at every point; the tables of
the feasibility note (sections 1 and 5) say which stage (gamma) runs must
be empty (P A A B B and P P A p_01 p_10 by the rank-3 tables at tau^{+-2};
the six configurations with a one-qubit slice of three terms by the rank-2
table). Stage (gamma) runs every flat pair regardless, and `driver.py
tables` checks the stored tables as a consistency control (section 6).

## 3. The bases

Stage (alpha): the full 5-covers of psi_3 up to G_3, the enumeration of
the feasibility note (section 3), by hash.

| kind | pattern | count | list |
|---|---|---|---|
| A, distinct independent | (1, 1, 1, 1, 1), kappa = 0 | 6,115,136 | `results/census5_full.json`: the per-pivot-pair counts of `CoverEnumerator.covers(5)` at orbit qubit_T through the compiled kernel `cover5_pair` (20 pivots, 4,672 pivot pairs, 345,706,953 modular candidates, 291 kernel seconds); the covers themselves are regenerated per pivot pair by the batches |
| B, distinct dependent | (1, 1, 1, 1, 1), kappa = 1 | 20,653 | `degenerate5.json` (`driver.py degenerate --write`): a full 3-cover T plus a pair parallel modulo T, 4,488; a full 4-cover plus a state of its span, 16,165 |
| C, a repeated state | (2, 1, 1, 1), kappa = 0 | 23,096 | same: a full 4-cover plus one of its members, 18,788; the cancel-at-base multisets T + (b, b) with b outside T and span(T), 4,308 |
| C | (2, 2, 1), (3, 1, 1) | 12, 12 | same: T plus two of its members |

The routes are those of the H^6 v2 list (`docs/notes/h6_rank5_stagec_repair.md`)
and of the H^5 pipeline, with the span states through the modular
`in_span`; every multiset passes the `ok` test (a coefficient family with
no unrepeated state dead). Completeness of the census rests on the H^6
argument (pivot one per orbit, partner minimal in its stabilizer orbit,
members in orbits at or above the pivot's, the residue kernel listing a
superset, every candidate decided mod 2013265921 and numerically) and on
the degenerate-list argument of the H^6 stage C repair, both unchanged by
the orbit. The rank-4 record cross-checked the 3-covers and 4-covers
against the numeric enumerator orbit by orbit (`research/t5_rank4/results/control_census.json`).

Stage (beta'): the 4,709 full 4-covers of `research/t5_rank4/covers4.json`
(sha256 `e83212833195240a`, reused by hash): 4,697 of distinct independent
states and 12 of pattern (2, 1, 1) (a state of a full 3-cover doubled); no
dependent 4-cover exists (span(T) holds no further dictionary state for
any of the four 3-covers) and no cancel-at-base 4-multiset (two cancelling
copies would leave two states covering psi_3).

Stage (gamma): the 4 full 3-covers (12, 352, 856), (33, 124, 158), (143,
565, 749), (309, 358, 873), stored in `covers4.json` and in the partition.

The partition records the census file's sha256, the degenerate list's and
the 4-cover census's hashes and the 3-covers; the aggregate re-enumerates
the 3-covers, the 4-covers with their repeated multisets and the degenerate
5-multisets (`driver.fresh_lists`, about 100 s) and requires equality.

## 4. The matchers

Stage (alpha) runs `slice_cover.SliceMatcher(E, 2)` at 00: the two
coordinate slices at ratio tau and the composite slice at ratio tau^2. Kind
A runs through the compiled kernel `SliceMatchKernel` (orbit-agnostic:
the target slices mod both primes are its inputs); kinds B and C run the
Python reference through the coefficient family and the block treatment
with the 2026-09-23 repairs (cancelling copies reconstructed,
`UnpinnedFamily` raised instead of a silent drop). Everything section 4 of
the H^6 note says about exactness holds here with psi_5 = |T>^5 and the
Q(zeta_24) field.

Stages (beta') and (gamma) are `research/t5_rank5/invisible.py`; its module
docstring is the specification. The flats of the invisible terms are
parameters; nothing is assumed about the visible terms' flats: at every
one of the three other points each visible term is absent or one of the
32 phased Pauli translates of its base slice, its codes at a point are
restricted to those compatible with the codes already fixed at the other
two points (a term present at two points is a plane whose third code is
the composition up to the quadratic sign, a term present at exactly one is
a line through 00 and absent at the third, a term absent at both is
free), and the presence pattern is checked at assembly to be a subspace
through 00 with the structure lemma's composite code (`valid_term_codes`).
The points are processed in an order that keeps the number of invisible
terms whose slice is still unknown ("fresh") at each point as small as
possible: the points where no invisible term is present first, then the
others.

- exact (no fresh term): `solve_slice` over the visible options, the
  block's translate sets and the 32 phased translates of every invisible
  term already known (meet in the middle mod 65521, every candidate
  decided over C and mod 2013265921); in stage (gamma) with known terms a
  batched variant meets the visible residuals with the known terms' options
  through a random functional and decides every candidate in the three
  fields (`_exact_known`).
- scan (one fresh term): the residual after the visible terms, the block
  and the known terms must be c v for a dictionary state v: all option
  combinations at once mod 65521, the residual projected onto the
  annihilator of the block's translates, normalized and looked up in the
  table of the normalized projected dictionary states, every match
  re-decided exactly (the system (translates | v)(a, c) = r over C, mod
  65521 and mod 2013265921, unique with every entry nonzero). A zero
  residual means the fresh term is absent at a point of its flat, another
  flat's configuration, and is not a solution here.
- scan2 (two fresh terms, stage (gamma) only, for two equal point flats or
  two equal line flats): the residual must be c_4 v_4 + c_5 v_5. Distinct
  states are found by a projective hash (v and w span r exactly when their
  images in F^8 / span(r) are parallel; every pair inside a run of equal
  keys is taken), every collision decided exactly mod 2013265921 and over
  C with the two required to agree. Two equal point flats with equal
  states or a zero residual would be the same term twice and are rejected.
  For two line terms the states may coincide: r = D v (D = c_4 + c_5, the
  split unknown) or r = 0 (a cancelling pair, c_5 = -c_4, v unknown). Both
  are carried to the line's other point, where the pair contributes a
  vector of the span of at most two translates of the common state: with v
  known, the coordinates in the translate basis of v (decided over C and
  mod both primes, which must agree on their zero pattern) give the two
  classes, one class with two phases, or an empty support (the pair
  cancels there inside one class), and the split is solved from the phase
  relations numerically and mod 2013265921; with v unknown, the residual
  is scanned for a stabilizer residual (both copies in one class, phases
  distinct) or a rank-2 residual whose two states lie in one Pauli orbit
  with coefficients cancelling up to a fourth root of unity, and the
  common slice runs over the 32 phased translates.

The ambiguous case of stage (beta') with a block (the 12 repeated bases):
at the scan point the residual can lie in the span of the block's chosen
translates, so the fifth term's slice could be a dictionary state inside
that span and the slice equation cannot separate it from the copies.
`_pair_brute` decides it by enumeration, as the H^5 stage (beta) did for
its diagonal flat: for every dictionary state v inside the span, every
assignment of the two copies' codes at the three points, every code of the
fifth term at the second point of its flat and every admissible option of
the visible terms there, the coefficients (c_1, c_2, c_5) solve the slice
equations in the translate basis of the repeated state together with c_1 +
c_2 = D, point by point in batches; a survivor whose system leaves a
coefficient free raises `UnpinnedFamily`.

Every surviving state is assembled into five terms (the copies of a block
through `reconstruct_block`) and confirmed as at H^6 (`SliceMatcher.confirm`:
residual against psi_5, rank, independence, nonzero coefficients, the span
condition mod 2013265921); `batch.py` re-decides every hit again from its
phase codes (`common.decide_terms`).

Refusals and undecided runs. A base whose family is empty or has a dead
ordinary coefficient is refused; the lists exclude such multisets, so a
refusal fails the aggregate. `UnpinnedFamily` (a base with a coefficient
family, a block other than one pair, a residual whose complex and modular
decisions disagree, a joint reconstruction that leaves a coefficient free,
any configuration the enumeration does not cover), the candidate cap of the
dense solve, a hit on which the modular and numeric decisions disagree, and
a run not made before `--max-seconds` are recorded as undecided; the batch
exits 1 and the aggregate fails.

## 5. Controls

All on one laptop core at nice 19 (Apple M5 Pro, 18 cores; the load
average from other sessions fell from about 17 to about 3 during the
session, so the times below are not all on the same footing, and the
feasibility note's rates, taken at load 30 to 40, are about twice these).
Every control is a `driver.py` command with its record under `results/`
and its log under `results/logs/`.

Control 1, planted instances (`driver.py control-planted`,
`results/control_planted.json`, merged over capped runs): the target is the
sum of the planted terms with Gaussian-integer coefficients, the matcher
runs against the target's slices (`invisible.PlantedInvisibleMatcher`), and
the planted decomposition must be among the hits. Stage (beta') at every
one of the six flats, four kinds each: (a) four visible terms, planes or
lines through 00; (b) three planes and a line through 00; (c) a repeated
base state (two plane copies with one base slice) and two visible terms;
(e) a repeated base state whose copies share a Pauli class with the
invisible term at the first point of its flat, the ambiguous case decided
by the joint reconstruction (`_pair_brute`, exercised in all six (e)
instances and in two (c) ones). Stage (gamma) at every one of the 21 flat
pairs with random invisible terms (kind a), the six (point, line) pairs
with the point on the line and the point term's slice the ray of the line
term's slice there (ray), and for the shared-line pairs the same-state
variants (same: equal slices at the first point, different classes at the
second; same1: one class, different phases; cancel: cancelling
coefficients with different classes at the second point; cancel1:
cancelling coefficients, one class): cancel at all three lines, cancel1 at
(B, B) and (C1, C1), same and same1 at (C1, C1) and (D1, D1). Result: 60
of 60 planted decompositions recovered (24 stage (beta'), 36 stage
(gamma)); the (beta') instances run in 0.4 to 5.6 s, the (gamma) kind a
and ray instances in 0.4 to 46 s, and the same-state and cancelling
instances in 3 to 320 s, since the planted target's slice at the shared
line's first point is a low-rank vector and the rank-2 scan then lists
tens of thousands of stabilizer-state pairs through it (real bases show
nothing of the kind, section 6). Two defects were found and fixed by these plants
before the record: the rank-2 scan took only adjacent pairs inside a run
of three or more states with equal projective keys (the (B, B) kind a
instance was missed), and the cancelling pair with distinct classes was
matched with c_4 + c_5 = 0 instead of c_4 + mu c_5 = 0 for a fourth root of
unity mu (the two terms' slices at the shared point are the same ray up to
that phase). Recovered in runs killed at the cap before the record was
made incremental, and so not in the record: the kind samec instances (the
pair cancelling at the second point inside one class) at all three lines,
and same and same1 at (B, B); not run to the end: cancel1 at (D1, D1).

Control 2, the rank-6 witness (`driver.py control-witness`,
`results/control_witness.json`; `--all-x0`, `results/control_witness_all_x0.json`).
`bounds/qubit_T-m5-upper-6.json` sliced along every qubit pair has 4
distinct all-visible bases at 00 (14 over the four base points): at 00
two with six distinct independent states (kappa = 0) and two with six
distinct dependent states (kappa = 2). Result: the two kappa = 0 bases
return the witness among their 4 genuine rank-6 decompositions in 1.3 s
each through the compiled kernel; the two kappa = 2 bases abort after 8 s
at the 2,000,000-candidate cap of the dense 2-parameter solve at their
first coordinate slice and are recorded as aborted, a control failure at
that cap, never a pass (`control-witness: 2/4, 2 aborted, FAIL`). Over all
four base points: 10 of 14 recovered, the same 4 kappa = 2 bases aborted.
This is the rank-4 record's result (its raised-cap run did not finish
within a 600 s cap either); a 2-parameter family of six distinct
dependent states is a rank-6 shape on no rank-5 path (stage B has five
states and kappa = 1), so the exclusion does not rest on it, and the
control is recorded as incomplete rather than passed. Along a pair
containing qubit 5 the witness has three invisible terms at every point
(its terms are point terms along qubit 5), so it controls no stage with
invisible terms; those are controlled by the plants.

Control 3, the rank-3 decomposition of |T>^4 along a qubit pair
(`driver.py control-m4-pair`, `results/control_m4_pair.json`): the stage
(alpha) matcher at n_1 = 2 over every full 3-cover of |T>^2 (206
independent covers, 5 dependent or repeated) at 00 must recover the stored
class wherever a member of it has an all-visible base at 00 with a full
family, and nothing outside the stored list: 211 runs in 1 s, 5 genuine
rank-3 hits forming the one stored class, nothing missing, nothing
unexpected, no refusal, no undecided run.

Control 4, the lists (`driver.py control-census`, `results/control_census.json`):
fresh enumerations of the 3-covers (4), the 4-covers of distinct
independent states (4,697) with the repeated 4-multisets (12) and their
kinds, and the degenerate 5-multisets (43,773) equal the stored
`covers4.json` and `degenerate5.json` (38 s); the census file's 4,672 rows
are the enumerator's pivot pairs in its order and sum to 6,115,136. The
degenerate list also equals the feasibility probe's, which built the same
multisets with a second implementation of the span states. The aggregate
repeats the re-enumeration on every run.

Control 5, the tables (`driver.py tables`, `results/control_tables.json`):
the rank-2 table of the feasibility note (0 exact, 0 stabilizer, 0 rank-2
residuals over all 274,625 combinations at each of tau^{+-1}, 76,971
collisions decided) and the rank-4 record's tables for the 8 rank-3
decompositions of |T>^3 (0 exact rows at tau^{+-2}). They are consistency
checks on the stage (gamma) runs they say are empty, not inputs of the
exclusion.

## 6. Rates, partition, and the pipeline test

Measured by `driver.py sample STAGE` (`results/rates.json`,
`results/sample_*.json`), per cover, one process at nice 19 with a load
average of 3 to 6 from other sessions:

| stage | covers sampled | matcher | per cover: mean | median | max | hits, refused, undecided |
|---|---|---|---|---|---|---|
| A | 200 from the first pivot pairs | native | 0.17 ms | 0.2 ms | 1 ms | 0, 0, 0 |
| B, pattern (1, 1, 1, 1, 1) | 200 spread through the 20,653 | reference (1-parameter dense solve) | 2.0 s | 1.95 s | 12.8 s | 0, 0, 0 |
| C, (2, 1, 1, 1) | 200 spread through the 23,096 | reference (block) | 0.053 s | | | 0, 0, 0 |
| C, (2, 2, 1) | all 12 | reference | 4.3 s | | 12.2 s | 0, 0, 0 |
| C, (3, 1, 1) | all 12 | reference | 0.014 s | | | 0, 0, 0 |
| beta', independent, six flats | 40 spread through the 4,697 | InvisibleMatcher | 0.19 s | 0.01 s | | 0, 0, 0 |
| beta', (2, 1, 1), six flats | all 12 | InvisibleMatcher (block) | 12.2 s | | 47.6 s | 0, 0, 0 |
| gamma, 21 pairs | all 4 | InvisibleMatcher | 15.5 s | 1.0 s | 59.9 s | 0, 0, 0 |

Stage A's histogram over the 200 runs: 87 die at the first coordinate
slice, then (1, 1) 44, (7, 49) 22, (2, 4) 22, (4, 16) 15, up to (16, 256).
Stage beta' by flat: the three flats missing 11 (p_01, p_10, B) cost 1 to
6 ms per cover, since the exact equation at 11 (ratio tau^2) almost never
has a solution and the run dies there; p_11 costs 2 ms (two exact
coordinate points first); C1 and D1 cost 1.5 s per cover on average (24 s
at most, on the repeated bases), since their exact point is a coordinate
point (ratio tau, a few solutions) and the scan at the other coordinate
point then runs over 33^4 combinations. Stage gamma by pair: the pairs
with an exact point at 11 die at once; (B, C1) and (B, D1) cost 20 s at
most (two scans of 33^3 combinations, the second over about a hundred
states from the first: the residual at a coordinate point is a stabilizer
state for about a hundred visible combinations of a 3-cover, as the
rank-4 record's tables say), (C1, C1) and (D1, D1) 8 s.

The partition (`driver.py partition --target-s 600 --pod-factor 2.0`,
`partition.json`, sha256 `41d4cdad723391d2`): stage A pivot pairs grouped
greedily by the census's kernel seconds plus the sampled matcher time per
cover; stages B, C, beta' and gamma round-robin over their lists at the
sampled rate per multiplicity pattern (B and C with at most 2,000 covers
per batch). The pod is taken at twice the laptop rate, as in the H^5
partition; the laptop rates here were measured at a lower load than the
H^5 ones, so the factor may be optimistic, and the first batch of every
stage on the pod decides (section 8).

| stage | batches | indices | per batch | estimated pod CPU-hours |
|---|---|---|---|---|
| A | 5 | 0 to 4 | 144, 389, 292, 989, 2,858 pivot pairs | 0.75 |
| B | 138 | 5 to 142 | 149 or 150 covers at 4.0 pod s | 22.9 |
| C | 12 | 143 to 154 | 1,926 or 1,927 covers at 0.11 pod s (the (2, 2, 1) covers at 8.6 s) | 0.71 |
| beta' | 4 | 155 to 158 | 1,177 or 1,178 covers at 0.37 pod s (the twelve (2, 1, 1) bases at 24 s) | 0.57 |
| gamma | 1 | 159 | 4 covers at 31 pod s | 0.03 |
| total | 160 | | | 25.0 |

Wall time on 15 processes: about 1.7 hours, dominated by stage B (138
batches of about 600 s). Stage B is 92 percent of the cost; compiling the
1-parameter dense solve, the saving noted at H^6 and H^5, would bring the
whole run under 3 pod CPU-hours. Not needed to run.

Pipeline test (one laptop core at nice 19, load average 3 to 6). One
batch of each stage into `results/`, the first batch of every stage:

| batch | stage | content | runs | time | result |
|---|---|---|---|---|---|
| 0 | A | 144 pivot pairs, 1,559,276 covers (the heavy pivots come first) | 1,559,276 | kernel 17 s, match 250 s, 267 s wall | 0 hits, 0 refused, 0 undecided |
| 5 | B | 150 dependent covers | 150 | 285 s (1.9 s per cover) | 0, 0, 0 |
| 143 | C | 1,927 covers with a repeated state, one of pattern (2, 2, 1) | 1,927 | 128 s (67 ms per cover) | 0, 0, 0 |
| 155 | beta' | 1,178 full 4-covers, three of them repeated, six flats each | 7,068 | 200 s (0.17 s per cover) | 0, 0, 0 |
| 159 | gamma | the 4 full 3-covers, 21 flat pairs each | 84 | 62 s | 0, 0, 0 |

`aggregate.py --dry-run --partial` (`results/logs/aggregate_dry.log`, 38 s)
re-enumerated the three lists (equal to the stored ones), passed every
stored check on the five records, wrote `batch_manifest.partial.json`, and
projected the stage totals at the observed laptop rates: A 0.29 CPU-hours,
B 10.9, C 0.43, beta' 0.22, gamma 0.02, against the partition's 25 pod
hours at the factor 2. The deadline path: batch 159 with `--max-seconds 8`
into a scratch directory stops after its first runs, records the runs not
made as undecided ("deadline: not run"), exits 1, a rerun without
`--resume` refuses to skip it (exit 1), and `--resume` redoes it with the
deterministic hash of the committed record.

The laptop records of the five batches are committed under `results/`
(`batch_K.json`, `batch_K.log`), so the pod loop skips them.

## 7. Soundness checklist

From `docs/notes/qutrit_m4_rank5_review.md` and the H^6 stage C repair;
status of each item for this pipeline.

1. The stage C list carries the cancel-at-base multisets T + (b, b): yes,
   4,308 of them in `degenerate5.json`; the aggregate's fresh enumeration
   must equal the list (control 4).
2. `reconstruct_block` allows classes outside the translate set used by
   two copies with net coordinate zero and restricts composite codes to
   the structure lemma's shapes: yes, the repaired `slice_cover.py` is
   used by both matchers (stage (beta') rebuilds its block through it).
3. A block state reaching the final loop with an unpinned family or with
   dependent block translates raises `UnpinnedFamily`; the batch records
   the run as undecided and the aggregate fails: yes (stage (alpha)
   unchanged; stages (beta') and (gamma) raise on every case their
   enumeration does not cover, section 4).
4. A refused cover fails the aggregate: yes (`aggregate.check_batch`).
5. Floating-point pruning backed by exact checks: the scan lookups are
   done mod 65521 (a superset) and every survivor is re-decided over C and
   mod 2013265921 with the two required to agree (a disagreement raises);
   the rank-2 scan's collisions, the translate-basis coordinates and the
   pair splits are decided mod 2013265921 and numerically with agreement
   required; the joint reconstruction prunes by complex least squares at
   tolerance 1e-7 on systems with unit-modulus entries and its survivors
   are confirmed exactly, as at H^5. The tolerances of
   `has_zero_coefficient`, `is_full`, `restrict` and `_refine_split` are as
   at H^6, and the kinds of the census are decided mod 2013265921 and
   numerically with agreement (`common.kind_of`).
6. Every hit re-decided from its phase codes mod 2013265921 and
   numerically (`common.decide_terms`, the T field), a disagreement
   undecided, the aggregate re-deciding every stored hit: yes.
7. The deterministic hash excludes the kernel-run count (`native_runs`)
   and the modular candidate count (`candidates`): yes, both follow
   `deterministic_sha256` in the record, and `aggregate.check_batch`
   fails a record that carries either inside the deterministic part. The
   `--no-native` replay of one stage A batch is a pod task (section 8);
   open until run.
8. Every batch record carries the partition, census, degenerate-list and
   4-cover-census hashes; the aggregate checks them, and checks that the
   degenerate list was built over the 4-cover census it names: yes.
9. No `id()`-keyed caches: the scan tables are keyed by the block state's
   dictionary index and the translate set; `SliceMatcher.cache`,
   `InvisibleMatcher._inv`, `_tbasis_cache` and the Pauli-orbit table are
   keyed by dictionary index.
10. `--max-seconds`: runs not made are recorded as undecided (a stage A
    pivot pair not run is recorded as a unit), the batch exits 1, and
    `--resume` redoes such a record.
11. The matchers assume nothing beyond the base point and the invisible
    flats they are given: stage (alpha) yes; stages (beta') and (gamma)
    enumerate every flat and flat pair missing 00 and assume nothing about
    the visible terms' flats (the absent option is in every option list,
    the compatibility filter is a necessary condition of the structure
    lemma, the presence pattern is checked at assembly). The 28 invisible
    multisets at 00 that Fact 2 allows are exactly the stages' cases.
12. Controls re-run after the last matcher change: every instance of the
    planted record (60), the witness, m = 4 pair, list and table controls,
    the rate samples and the pipeline test were run at the commit that
    carries the final `invisible.py`; the (B, B) same and same1 and the
    samec instances were recovered before the last change (the light
    option tables and the Pauli-orbit table, which touch only the cost)
    and are not in the record. `slice_cover.py` is unchanged on this
    branch.
13. The T field is a ring homomorphism from Q(zeta_24) sharing i with the
    compiled kernels: as in the rank-4 record (asserted in
    `Field.__init__`, exercised by the witness control through the kernel).
14. Terms are pairwise distinct: stage (gamma) rejects two equal point
    terms, two equal invisible terms and equal-phase copies on a shared
    line; the joint reconstruction rejects equal copies; `confirm` reports
    the rank and independence of every hit and `genuine` requires rank 5.

## 8. The pod run

The pod checkout is `/root/stabrank-h6` (on branch `h5-rank5-pipeline`);
the new branch is fetched and checked out, never `git checkout -f` (it
would overwrite result files); runs go under `setsid nohup ... < /dev/null
& disown`; the batch loop is resumable (a finished record is skipped, one
left incomplete by the deadline is redone with `--resume`).

```
ssh -i ~/.ssh/id_ed25519 -p 40096 root@157.157.221.30
cd /root/stabrank-h6
git fetch origin t5-rank5-pipeline && git checkout t5-rank5-pipeline && git pull --ff-only
/root/.local/bin/uv sync --extra challenge
/root/.local/bin/uv run --extra challenge python -c "import stabrank.stabrank_core as c; print(c.cover5_pair, c.SliceMatchKernel)"
/root/.local/bin/uv run --extra challenge python -m pytest tests/test_slice_cover.py -q
mkdir -p research/t5_rank5/results /root/logs
```

Rate check first (the second batch of each of stages A, B, C and beta',
the first ones having run on the laptop as the pipeline test; the gamma
batch 159 ran on the laptop too), in the foreground of a `setsid nohup`
shell, about fifteen minutes; then the aggregate's `--partial` projection
tells whether the pod factor of 2 holds before the whole partition is
launched:

```
setsid nohup sh -c 'for K in 1 6 144 156; do
  nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5_rank5/batch.py $K --max-seconds 3600 \
    > research/t5_rank5/results/batch_$K.log 2>&1; done' < /dev/null > /root/logs/t5_probe.log 2>&1 & disown
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5_rank5/aggregate.py --dry-run --partial --no-reenumerate
```

The full run, 15 processes over the 160 batches (the laptop records of
batches 0, 5, 143, 155 and 159 are committed under `results/` and are
skipped by the loop):

```
setsid nohup sh -c 'seq 0 159 | xargs -P 15 -n 1 sh -c \
  "nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5_rank5/batch.py \"\$0\" --resume --max-seconds 3600 \
   > research/t5_rank5/results/batch_\"\$0\".log 2>&1"' < /dev/null > /root/logs/t5_run.log 2>&1 & disown
```

Rerunning the same command resumes. Progress, the final check (what the
certificate runs), and the `--no-native` replay of the smallest stage A
batch (checklist item 7; the Python reference for the 5-cover kernel and
the stage A matcher, about 80 times the native time):

```
grep -l "DECOMPOSITION FOUND\|undecided run" research/t5_rank5/results/batch_*.log
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5_rank5/aggregate.py --dry-run --partial
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5_rank5/aggregate.py --recheck 2 --recheck-seed 20260924
setsid nohup sh -c 'STABRANK_NO_NATIVE=1 nice -n 19 /root/.local/bin/uv run --extra challenge python \
  research/t5_rank5/batch.py 0 --out-dir research/t5_rank5/results/nonative --force' \
  < /dev/null > /root/logs/t5_nonative.log 2>&1 & disown
```

Copy back `research/t5_rank5/results/batch_*.json`, the logs,
`batch_manifest.json` and the no-native record with rsync over port
40096, commit them, fill the placeholders of
`bounds/qubit_T-m5-lower-6.json` and
`bounds/qubit_T-m6-lower-6.json` (compute hours, hardware, dates, the batch
indices of the re-runs) and move them to `bounds/`; the submissions
workflow verifies every touched bound, so the drafts keep their suffix
until the manifest exists. Then run
`verify_challenge/cert_qubit_t_m5_rank5_attested.py` and update the board:
chi(T^5) = 6 and chi(T^6) = 6.

### 8.1 The run (2026-09-24)

All 160 batches ran on the RunPod pod with the anneal loops paused: the
gamma smoke batch 159 at 07:37 UTC (4 covers, 84 runs, 105 s), then the
160-batch loop fifteen at a time from 07:40 to 10:04 UTC. Totals from the
aggregate: stage A 5 batches over the 6,115,136 distinct independent full
5-covers of |T>^3 (507 s per batch, 0.4 ms per cover), stage B 138 batches
over the 20,653 dependent covers (887 s per batch, 5.9 s per cover, about
four times the laptop rate against the factor two the partition assumed),
stage C 12 batches over the 23,120 multisets with a repeated state (371 s
per batch), stage beta' 4 batches over the 4,709 full 4-covers with all six
invisible flats (421 s per batch), and stage gamma 1 batch over the 4 full
3-covers with all 21 flat pairs (106 s); 0 hits, 0 refused, 0 undecided,
36.4 CPU-hours in all (the partition estimated 25). The aggregate with
`--recheck 2 --recheck-seed 20260924` re-enumerated the 3-covers, the
4-covers, and the degenerate list (equal to the stored ones, 54 s),
verified every stored batch, re-ran batches 10 (550 s) and 58 (565 s) from
scratch with matching deterministic hashes, wrote `batch_manifest.json`,
and printed `CERTIFIED chi(qubit_T^5) >= 6` in 1,169 s. With the Lean-tier
rank-6 witnesses the cells are chi(T^5) = 6 and chi(T^6) = 6, filed as
`bounds/qubit_T-m5-lower-6.json` and `bounds/qubit_T-m6-lower-6.json` at
the attested tier. The `--no-native` replay of stage A batch 4 was started
on the pod after the run; its record goes under `results/nonative/`.

## 9. What is proved, what is assumed, what is open

Proved by argument or table here: Fact 2 (a board bound), the base-point
reduction (every rank-5 decomposition has a full 3-, 4- or 5-cover base at
00 with the invisible terms on flats missing 00), the completeness of the
28 invisible cases, the completeness of the degenerate list and the
4-cover and 3-cover lists given the enumerator (with the rank-4 record's
numeric cross-check), the emptiness of the stage (gamma) runs the tables
cover. Assumed from earlier work: the pivot and partner reductions of the
enumerator (the H^6 argument), the slice structure lemma (PR #87), the
stage C repair of `slice_cover.py`. Open before the bound: the pod run
itself, its aggregate, the `--no-native` replay of one stage A batch, and
the pod rate of stage B (the partition takes twice the laptop rate at a
low load; the first stage B batch on the pod decides whether the 1.7 hours
of wall time hold). Recorded rather than closed: the two kappa = 2 witness
bases, aborted at the default candidate cap (control 2, on no rank-5
path); the same-state and cancelling planted kinds are in the record for
the pairs listed in control 1 and not for the others (samec at no line,
same and same1 at (B, B), cancel1 at (D1, D1)); the joint reconstruction
of stage (beta') is written for one pair block (the list has no other
block shape) and raises otherwise.
