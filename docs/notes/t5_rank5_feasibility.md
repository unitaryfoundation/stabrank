# Excluding rank 5 for |T>^5: feasibility and design

Status (2026-09-24). Design and costing only; nothing here is a bound.
The cell is 5 <= chi(T^5) <= 6 (`bounds/qubit_T-m5-lower-5.json`, the
rank-4 exclusion of `docs/notes/t5_rank4_exclusion.md`;
`bounds/qubit_T-m5-upper-6.json`, the product witness at the Lean tier).
Excluding rank 5 settles chi(T^5) = 6 and, by projection monotonicity,
chi(T^6) = 6 (that cell holds 5 <= chi <= 6 from the projected lower bound
and its own Lean witness).

Verdict: feasible, as a pod run of about 60 CPU-hours (about 4 hours of
wall time on 15 processes), with one structural input, chi(T^3) = 3, and
no analogue of the H^5 argument's Fact 1. The H^5 design (an all-visible
or one-invisible base point chosen through the flat lemma) does not carry
over, because chi(T^4) = 3 lets a one-qubit slice see three or four
terms, and establishing that every term is full along every qubit would
need the census of the non-minimal rank-4 decompositions of |T>^4 over
the 36,720 four-qubit stabilizer states (section 1). The design here
(section 2) drops Fact 1 and the base-point case split: the base point is
00 along qubits 1, 2 for every decomposition, the visible terms at 00
number three, four, or five by chi(T^3) = 3, and the matcher carries the
invisible terms with every flat of F_2^2 that misses 00. The bases are
then the full 3-covers, 4-covers, and 5-covers of |T>^3 over the 1080
three-qubit states, all enumerated here (section 3): 4, 4,709, and
6,158,909 (6,115,136 of distinct independent states by the compiled
kernel in 291 s, plus 43,773 dependent or repeated-state multisets). The
cost is dominated by the 20,653 dependent 5-covers through the Python
reference matcher at about 4 s each (section 4). Scripts and records are
under `research/t5_rank5/` (`probe.py`, `run.py`, `results/`).

Notation follows `docs/notes/t5_rank4_exclusion.md`: |T> = cos(beta)|0> +
e^{i pi/4} sin(beta)|1> with cos(2 beta) = 1/sqrt 3, psi_m = |T>^m, tau =
e^{i pi/4} tan(beta) the ratio between the amplitudes of |T>, N_3 = 1080
the three-qubit stabilizer states in the order of `dictionary(2, 3)`, and
G_3 the unitary symmetry group of psi_3 (order 162, 20 orbits on the
dictionary). A rank-5 decomposition is psi_5 = sum_{i=1}^5 c_i s_i with
stabilizer states s_i and every c_i nonzero. Slicing along qubits 1, 2 at
x in F_2^2 gives sum_i c_i u_i^(x) = alpha_x psi_3 with alpha_x =
a_0^{2 - |x|} a_1^{|x|} (never zero) and u_i^(x) = (<x| (x) I) s_i; the
ratio between the slices at x and x_0 is tau^{|x| - |x_0|}. Each term is
nonzero exactly on an affine flat of F_2^2: one of the four points, the
six lines, or the plane. The lines are named A = {00, 11} and B = {01, 10}
(the diagonals), C0 = {00, 01} and C1 = {10, 11} (x_1 fixed), D0 =
{00, 10} and D1 = {01, 11} (x_2 fixed); P is the plane and p_x the point
x. The number of terms absent at a point x is written a(x). For a
one-qubit slice at value k of some qubit, k' = 1 - k is the other value.
The coefficient family of a base is the affine set of coefficient vectors
d with sum_i d_i u_i = psi_3 over the distinct base states, and kappa is
its dimension (0 for independent states). Everything about slices,
translates, and the structure lemma is as in the earlier notes. The arithmetic is the Q(zeta_24) field of `slice_cover.Field(p,
"qubit_T")` mod 65521 and 2013265921, as in the rank-4 exclusion.

All timings are from one laptop core at nice 19 with a load average of 30
to 40 on 18 cores from other sessions, under a 600 s cap per run
(`research/t5_rank5/run.py`), so every rate overstates an unloaded core.

## 1. Why Fact 1 does not transfer

Fact 1 of the H^5 and T^5 rank-4 arguments says that every term is
nonzero at both values of every qubit. Its proof slices along one qubit
and counts visible terms: they decompose alpha_k psi_4, so there are at
least chi(T^4) of them. For rank 4 at T this gave exactly three visible
terms in the only bad case, the unique rank-3 decomposition of |T>^4, and
a table killed it. For rank 5 a one-qubit slice sees three, four, or
five terms, not four or five: chi(T^4) = 3, so three visible terms are
possible. Of the two bad cases, the three-visible one closes by a table
computed here and the four-visible one does not.

Three visible terms at value k of qubit q. They are the rank-3
decomposition of |T>^4 (one class), and the two other terms are point
terms |k'> (x) v_4 and |k'> (x) v_5. The equation at k' says that
tau^{+-1} psi_4 minus the visible terms' slices at k' (phased Pauli
translates of their slices at k, or zero, 65 options each) is c_4 v_4 +
c_5 v_5: a residual of stabilizer rank at most 2 over the 36,720
four-qubit states, for one of the 65^3 = 274,625 code combinations at
each ratio. `probe.py rank2` decides this by a projective hash: v and w
span R exactly when their images in F^16 / span(R) are parallel, so the
keys (f_2 v' / f_1 v', f_3 v' / f_1 v') of random functionals agree; each
key collision is re-decided by rank mod 2013265921 and over C, and the
two must agree. Measured on 1,000 random combinations per ratio: 7.1 ms
per combination, 0 exact, 0 stabilizer, 0 rank-2 residuals, 127 and 141
accidental collisions decided. The full table (both ratios, 549,250
combinations) is about 65 minutes; it was run in ten capped chunks
(`rank2_loop.sh`, `results/rank2_full.json`), with the totals in section
5. A zero table says: no one-qubit slice of a rank-5 decomposition of
|T>^5 has exactly three visible terms, or equivalently, no two terms are
point terms at the same value of the same qubit.

Four visible terms at value k. They are a rank-4 decomposition of |T>^4
with all coefficients nonzero, which is not minimal, and the fifth term
is a point term |k'> (x) v; the equation at k' says that tau^{+-1} psi_4
minus the four visible terms' translates is a nonzero multiple of a
stabilizer state, a stabilizer-residual row over 65^4 = 17.85 million
code combinations per decomposition and ratio (about 10 s each with the
committed `residual_table` at H). The blocker is the list of those
decompositions: every full 4-cover of |T>^4 by four of the 36,720 states
(distinct independent, distinct dependent, or with a repeated state) up
to the symmetry group G_4 of psi_4 (order 1,944). `probe.py census4`
measured what `CoverEnumerator(4, orbit="qubit_T").covers(4)` would cost:

| quantity | value |
|---|---|
| construction of the enumerator (dictionary, field images, orbits) | 5.0 s |
| orbits of G_4 on the 36,720 states (pivots) | 76 |
| pivot pairs (pivot, partner) over all pivots | 404,583 (planned in 36 s) |
| sample: pairs, stratified over the 76 pivots | 161 |
| full 4-covers found in the sample | 0 |
| modular candidates in the sample | 437,536 (2,717 per pair) |
| seconds per pair (Python `pair_covers(4)`) | 0.34 |
| projected census time at this rate | 39 CPU-hours |

The rate is set by accidental key collisions, not by covers: the r = 4
path hashes about 36,000 canonical residue rows into F_65521 with one
random functional, so thousands of pairs collide by chance at every
pivot pair and each is decided exactly. The enumerator was written for
N_3 = 1080, where this does not happen. A second functional or the
larger prime for the key would bring the census to a few CPU-hours; it
is a code change in `slice_cover.py`, not a design problem. The census
is not empty: the products of two full 2-covers of |T>^2 (two classes,
(10, 21) and (12, 21), under the symmetry of psi_2 of order 18) are full
4-covers of |T>^4 of distinct independent states (`probe.py products4`:
four product tuples, all full covers, every member in the orbit with root
6164). Zero covers in 161 of 404,583 pairs bounds the census loosely:
with a Poisson bound of three covers per 161 pairs, fewer than about
7,500 covers counted with multiplicity across pairs, most likely a few
hundred to a few thousand up to G_4. Even so the route through Fact 1
costs the re-hashed census, then 20 s of residual tables per decomposition
(two ratios), then the rank-2 table above, and it delivers only the flat
lemma, which the design below does without. It is not run here.

## 2. The design: one base point, three stages

Fact 2 (at most two absent terms at any point of F_2^2): the visible
terms at a point decompose a nonzero multiple of psi_3, and chi(T^3) = 3
(`bounds/qubit_T-m3-lower-3.json`). This is the only structural input.

Base point 00 for every decomposition. At x_0 = 00 the visible terms
number k in {3, 4, 5}; their base slices (u_i^(00)) form a full k-cover of
psi_3, that is, k three-qubit stabilizer states, repeats allowed, whose
span contains psi_3 with all coefficients d_i = c_i / alpha_00 nonzero
(the actual coefficients are one such assignment). The 5 - k invisible
terms have flats that miss 00: one of the six flats p_01, p_10, p_11, B,
C1, D1. G_3 acts on the unsliced qubits, commutes with the slicing, and
carries rank-5 decompositions of psi_5 to rank-5 decompositions
(psi_5 = psi_2 (x) psi_3), so one base per G_3 orbit suffices. Hence every
rank-5 decomposition is found by a matcher that, given a full k-cover base
at 00 and the flats of the 5 - k invisible terms, enumerates every
decomposition with that base and those flats, run over every full k-cover
up to G_3 and every choice of flats. No flat lemma and no choice of base
point is needed, and the swap symmetry of the sliced qubits is not used.
The base point 00 is also the cheapest of the three inequivalent points
(00, 01 with 10, and 11; |T> has no monomial symmetry, so 00 and 11 are
not exchanged): its other points have ratios tau, tau, and tau^2, none
equal to 1, while from 01 the point 10 has ratio 1, where the trivial
translate always solves the exact equation and the residual scans always
run (0.36 s against 0.013 s per cover in the stage (beta) measurement of
section 4).

Stages by the number of visible terms.

- Stage (alpha), k = 5: base a full 5-cover of psi_3 (kinds A, B, C as at
  H^6: distinct independent, distinct dependent, a repeated state,
  including the cancel-at-base multisets T + (b, b)). Matcher:
  `slice_cover.SliceMatcher(E, 2)` at x_0 = 00 with `E = CoverEnumerator(3,
  orbit="qubit_T")`, unchanged; it assumes nothing beyond the base point
  (every subspace of F_2^2 through 00 is allowed for every term).
- Stage (beta'), k = 4: base one of the 4,709 full 4-covers of psi_3 of the
  rank-4 census (`research/t5_rank4/covers4.json`: 4,697 of distinct
  independent states, 12 with a repeated state, no dependent ones); one
  invisible term with flat F in {p_01, p_10, p_11, B, C1, D1}. The H^5
  stage (beta) matcher (`research/h5_rank5/beta.py`) is the case F = B:
  exact at 11, residual scan at 01 (the residual after the visible terms
  is c_5 v for a dictionary state v), exact at 10 with the fifth term as a
  known phased translate of v. The general version takes F as a
  parameter: at a point y not in F the equation is exact over the visible
  terms' options; at the first point of F it is a residual scan giving
  (v, c_5); at the second point of F (a line) it is exact with the fifth
  term among the 32 phased translates of v. The visible terms keep the
  absent option everywhere and the presence pattern is filtered to
  subspaces through 00 at assembly, as now. The block treatment of a
  repeated base state (the 12 bases) is the existing one.
- Stage (gamma), k = 3: base one of the 4 full 3-covers of psi_3 (the 8
  stored rank-3 decompositions fall into these 4 orbits); two invisible
  terms with flats (F_4, F_5), a multiset of two of the six flats missing
  00, 21 in all. At a point y the equation has zero, one, or two invisible
  terms present: exact, a residual scan, or a rank-2 residual (the
  residual is c_4 v_4 + c_5 v_5 for dictionary states, decided by the
  same projective hash as section 1 over the 1080 three-qubit states,
  where it is instant). A term present at two points of its line has
  phased Pauli translates as slices. With four bases, 21 flat pairs, and
  33^3 visible options, the stage is minutes of work however it is
  written. Three visible terms are always distinct and independent (a
  repeated or dependent triple would put psi_3 in the span of two
  stabilizer states), so the coefficients are a point.

The flat configurations that Fact 2 allows (`probe.py flats`,
`results/flats.json`): 124 multisets of five flats with a(x) <= 2 at every
point; by a(00) they split 16 (all visible), 45 (one invisible), 63 (two
invisible); the invisible multiset at 00 takes 28 values: none, each of
the six flats alone, and each of the 21 pairs. So stages (beta') and
(gamma) must indeed carry every flat and every pair. Twelve configurations
have two absent terms at every point (the only ones where no point sees
more than three terms):

| configuration | one-qubit view | closed by |
|---|---|---|
| P A A B B | | the 8 rank-3 decompositions at ratio tau^{+-2}: 0 exact rows (section 5) |
| P P A p_01 p_10 | | the same table |
| P C0 C0 C1 C1, P D0 D0 D1 D1 | qubit 1 (or 2) at value 0 sees three terms | the rank-2 table of section 1 |
| P P C0 p_10 p_11, P P C1 p_00 p_01, P P D0 p_01 p_11, P P D1 p_00 p_10 | a one-qubit slice sees three terms | the rank-2 table of section 1 |
| P A B C0 C1, P A B D0 D1, P C0 C1 D0 D1, P P B p_00 p_11 | | stage (gamma) |

Stage (gamma) runs over all 21 flat pairs regardless; the tables are
cross-checks that also say which of its runs must be empty.

## 3. The bases

Stage (alpha): the full 5-covers of psi_3 up to G_3.

| kind | pattern | count | how | record |
|---|---|---|---|---|
| A, distinct independent | (1, 1, 1, 1, 1), kappa = 0 | 6,115,136 | `CoverEnumerator.covers(5)` with orbit qubit_T through the compiled kernel `cover5_pair` (modular targets as inputs): 20 pivots, 4,672 pivot pairs, 345,706,953 modular candidates, 291 kernel seconds | `results/census5_full.json` (per-pair counts) |
| B, distinct dependent | (1, 1, 1, 1, 1), kappa = 1 | 20,653 | a full 3-cover T plus a pair (x, y) with y in span(T, x): 4,488; a full 4-cover plus a state of its span: 16,165 (36,935 span states over the 4,697 covers); T plus two states of span(T): 0, since span(T) holds no dictionary state for any of the 4 covers | `results/degenerate5.json` |
| C, a repeated state | (2, 1, 1, 1), kappa = 0 | 23,096 | a full 4-cover plus one of its members: 18,788 = 4 x 4,697; the cancel-at-base multisets T + (b, b) with b outside T: 4,308 = 4 x 1,077 | same |
| C | (2, 2, 1), (3, 1, 1) | 12, 12 | T plus two of its members | same |

The routes are those of the H^6 v2 list (`research/h5_rank5/driver.py`,
`degenerate_covers`), with the span states through the modular `in_span`;
every multiset passes the `ok` test (a coefficient family with no
unrepeated state dead). The per-pivot census: 12: 1,713,973; 143:
1,319,070; 158: 1,460,754; 198: 310,955; 230: 576,657; 312: 357,136; 441:
184,090; 758: 74,369; 817: 25,093; 821: 14,129; 834: 18,456; 873: 58,438;
877: 1,198; 986: 666; 1019: 83; 1020: 46; 1025: 23; 1021, 1042, 1073: 0.
The corresponding H^6 numbers are 5,939,465 (14,280 pivot pairs, 459 s),
12,390, 15,994 (2,154 cancel-at-base), 6, 6. Completeness rests on the
H^6 census argument (pivot one per orbit, partner minimal in its
stabilizer orbit, members in orbits at or above the pivot's, the residue
kernel listing a superset, every candidate decided mod 2013265921 and
numerically) and on the degenerate-list argument of the H^6 stage C
repair, both unchanged by the orbit.

Stage (beta'): the 4,709 full 4-covers of `covers4.json` (sha256
`e83212833195240a`), cross-checked in the rank-4 record against the
numeric enumerator.

Stage (gamma): the 4 full 3-covers (12, 352, 856), (33, 124, 158), (143,
565, 749), (309, 358, 873).

## 4. Matcher rates and projected cost

Measured with `probe.py rates` (`results/rates.json`) and `probe.py
beta-rate` (`results/beta_rates.json`), per (cover, x_0) run, at x_0 in
{00, 01, 11} for the record though only 00 is needed:

| stage | list | covers sampled | matcher | per run at 00 | mean over the three points | max | hits, refused, undecided |
|---|---|---|---|---|---|---|---|
| (alpha) A | 6,115,136 | 200 spread through the sample | native `SliceMatchKernel` | 0.40 ms | 0.36 ms | 3.7 ms | 0, 0, 0 |
| (alpha) B, (1, 1, 1, 1, 1) | 20,653 | 4 spread | reference, 1-parameter dense solve | 4.3 s | 4.9 s | 12.0 s | 0, 0, 0 |
| (alpha) C, (2, 1, 1, 1) | 23,096 | 4 spread | reference, block | 0.51 s | 0.22 s | 1.4 s | 0, 0, 0 |
| (alpha) C, (2, 2, 1) | 12 | 4 (and all 12 in a first run) | reference | 8.4 s | 5.8 s (7.5 s over all 12) | 24 s (40 s) | 0, 0, 0 |
| (alpha) C, (3, 1, 1) | 12 | all 12 | reference | 0.03 s | 0.02 s | 0.06 s | 0, 0, 0 |
| (beta), F = B | 4,697 | 30 spread | H^5 `BetaMatcher` at orbit qubit_T | 0.013 s | 0.19 s (0.36 s at 01) | 1.2 s | 0, 0, 0 |
| (beta), F = B, repeated | 12 | all 12 | same, block | 0.005 s | 7.0 s (14 s at 01) | 44 s | 0, 0, 0 |

Kind A's solution histogram over the 600 runs: 508 die at the first
coordinate slice, then (1, 1) 28, (7, 49) 22, (2, 4) 14, (4, 16) 13, up
to (17, 289). The H^5 `BetaMatcher` runs at T without change (it takes
the orbit from the enumerator), which is why its rate could be measured;
the other five flats are estimated from it: a flat that misses 11 leaves
11 exact at ratio tau^2, which almost always has no solution (the 0.013 s
runs); a flat containing 11 makes 11 a residual scan of 33^4 options at
once (about 0.3 s at n_2 = 3), followed by exact solves at the coordinate
points. About 1 s per cover over the six flats, 3 to 5 s for the twelve
bases with a block.

Projected cost at the laptop rates and with the pod at twice the laptop
rate (the H^5 stage B ran at 5.1 s per run on the pod against 3.5 s in
the laptop sample; the H^6 factor was 2.3):

| stage | runs | rate | laptop CPU-hours | pod CPU-hours |
|---|---|---|---|---|
| 5-cover kernel | 4,672 pivot pairs | 291 s in all | 0.08 | 0.2 |
| (alpha) A, native | 6,115,136 | 0.40 ms | 0.7 | 1.4 |
| (alpha) B, reference | 20,653 | 4.3 s | 24.7 | 49 |
| (alpha) C, reference | 23,120 | 0.51 s, the (2, 2, 1) at 8.4 s | 3.3 | 6.6 |
| (beta'), six flats | 4,709 | about 1 s per cover | 1.3 | 2.6 |
| (gamma) | 4 x 21 | seconds | 0.1 | 0.1 |
| total | | | about 30 | about 60 |

About 4 hours of wall time on 15 processes, in the H^5 partition scheme
(about 600 s per batch, stage B round-robin, stage A by pivot pairs
weighted by the census). Stage B is 80 percent of it; compiling the
1-parameter dense solve, noted at H^6 and H^5 as the obvious saving, would
bring the whole run under 15 pod CPU-hours. For comparison, a design
that first established Fact 1 and then ran the H^5 stages at the two base
points 00 and 01 would double stage B and add the T^4 census and its
residual tables.

## 5. Tables

From the rank-4 record (`research/t5_rank4/results/tables.json`, the
committed `two_qubit_slice.residual_table` at the T target), counts of
(exact, stabilizer) residuals over all code combinations of the visible
terms:

| decompositions | ratio tau^j | exact | stabilizer |
|---|---|---|---|
| the 8 rank-3 decompositions of |T>^3, 65^3 combinations each | j = -2, 2 | 0 | 0 |
| | j = -1, 1 | 1 | 111 |
| | j = 0 | 8 | 768 |
| the rank-3 decomposition of |T>^4 | j = -2, -1, 1, 2 | 0 | 0 |
| | j = 0 | 1 | 192 |

What closes by table: the configurations P A A B B and P P A p_01 p_10
(three terms visible at 00, the same three present at 11 with nothing
else, ratio tau^2, an exact row at j = 2: none). The rank-2 table of
section 1 closes the six configurations in which a one-qubit slice sees
exactly three terms: over all 274,625 code combinations at each of j = 1
and j = -1 there are 0 exact, 0 stabilizer, and 0 rank-2 residuals
(38,303 and 38,668 accidental key collisions, all decided exactly mod
2013265921 and over C with agreement; `results/rank2_full.json`, 30.7 and
29.6 minutes). So no two terms of a rank-5 decomposition of |T>^5 are
point terms at the same value of the same qubit, and every one-qubit
slice sees four or five terms. Nothing else closes by table;
the remaining four two-invisible configurations and every one-invisible
configuration go to stages (gamma) and (beta'). The 111 stabilizer
residuals at j = +-1 for the rank-3 decompositions of |T>^3 bound the
branching of the residual scans in stage (gamma) at the coordinate
points; the single exact row at j = +-1 is the rank-3 decomposition of
|T>^4 seen along one qubit, as expected.

## 6. Controls

1. The rank-6 witness `bounds/qubit_T-m5-upper-6.json` (the product of the
   rank-3 decomposition of |T>^4 with |0> and |1> on qubit 5) through
   stage (alpha) from its all-visible bases; at x_0 = 00 along pairs among
   qubits 1 to 4 it has four (from the rank-4 record's
   `control_witness_default_cap.json`): two with six distinct independent
   states, recovered in about 3 s each, and two with six distinct
   dependent states (kappa = 2), which abort at the default candidate cap
   of the dense solve and did not finish at a raised cap within a 600 s
   run; they lie on no rank-5 path (stage B has five states and kappa =
   1) and are recorded as aborted, never passed. The other ten bases are
   at 01, 10, or 11 and can be run for the record since the matcher is
   generic in x_0. Along a pair containing qubit 5 the witness has three
   invisible terms at every point (its terms are point terms along qubit
   5), so it controls no stage with invisible terms.
2. Planted instances, as in the H^5 control 2, for stage (beta') at each
   of the six flats and for stage (gamma) at each of the 21 flat pairs,
   with the block cases for the repeated bases, the target the sum of the
   planted terms with Gaussian-integer coefficients; also instances with
   a visible term on a coordinate line or point through 00, which the
   matcher must find or reject but never mis-assign.
3. The rank-3 decomposition of |T>^4 along a qubit pair (the rank-4
   record's control 5, 422 runs, the stored class recovered): the stage
   (alpha) geometry at n_1 = 2 with hits, unchanged.
4. The census against a fresh enumeration in the aggregate (the
   degenerate list and the stage (beta') and (gamma) lists re-enumerated
   and compared by hash; the kernel census regenerated per pivot pair as
   at H^6), and the `--no-native` replay of one stage A batch.
5. The rank-2 table of section 1 and the tables of section 5 as
   consistency checks on the stage (gamma) runs they say are empty.

## 7. What is left to build, and the blockers

There is no structural blocker. The engineering:

1. The general invisible-flat matcher: stage (beta') with the flat as a
   parameter (six cases, the F = B case being `beta.py`), including the
   block path for the 12 repeated bases at every flat, and stage (gamma)
   with two invisible terms and the rank-2 residual over the 1080 states.
   Every survivor re-decided mod 2013265921 and over C, every hit through
   `SliceMatcher.confirm` and `decide_terms`, as now.
2. The pipeline `research/t5_rank5/` in the shape of `research/h5_rank5/`
   with one base point, three stages, the census by pivot pairs (the
   per-pair record of `results/census5_full.json` gives the partition its
   weights), the degenerate list by hash, and the certificate
   `verify_challenge/cert_qubit_t_m5_rank5_attested.py` printing
   `CERTIFIED chi(qubit_T^5) >= 6` and, by projection, `CERTIFIED
   chi(qubit_T^6) >= 6`.
3. Optional: the compiled 1-parameter dense solve for stage B (a factor of
   about 4 in the total).

Sizes of what is not done: the T^4 rank-4 census (404,583 pivot pairs, 39
CPU-hours at the present hashing, a few with a wider key; at most a few
thousand covers) and its residual tables are not needed; the rank-2 table
of section 1 is run.

## 8. What was run

All through `research/t5_rank5/run.py` (nice 19, own session, 600 s cap),
one process at a time, about 2.1 CPU-hours in all. One exception: the
first, slow version of the rank-2 probe survived its cap (macOS refused
the group signal with EPERM; the runner now falls back to killing the
group's members one by one) and ran for 23 minutes, overlapping the
matcher-rate, T^4 census, and product-cover runs; its 3,000 combinations
at j = 1 gave 0 rank-2 residuals as well. The overlapped rates are on a
machine already at load 30 to 40, so the effect is within their scatter.

| step | command | time | result |
|---|---|---|---|
| flat configurations | `probe.py flats` | 1 s (73 s with the environment build) | 124 configurations, 12 with two absent terms everywhere, 28 invisible multisets at 00 |
| 5-cover census, sample | `probe.py census5 --pairs 300` | 34 s | 4,672 pivot pairs; 363,673 covers in 300 pairs; estimate 5.69e6 covers, 276 kernel seconds |
| 5-cover census, full | `probe.py census5 --full` | 296 s | 6,115,136 covers, 345,706,953 candidates, 291 kernel seconds |
| degenerate 5-multisets | `probe.py degenerate5` | 19 s | 43,773: 20,653 dependent, 23,096 + 12 + 12 with a repeated state |
| matcher rates | `probe.py rates --count 200 --count-deg 4 --x0s 0,1,3` | 135 s (a first attempt with more degenerate covers was killed at the cap) | section 4 |
| stage (beta) rate | `probe.py beta-rate --count 30 --x0s 0,1` | 181 s | 0.013 s at 00, 0.36 s at 01; blocks 14 s at 01 |
| T^4 census probe | `probe.py census4 --pairs 150 --budget 200` | 97 s | 404,583 pairs, 0 covers in 161, 0.34 s per pair |
| product covers of T^4 | `probe.py products4` | 6 s | 4 product full 4-covers exist |
| rank-2 table, sample | `probe.py rank2 --count 1000` | 18 s (a first version brute-forced the key-zero rows and was killed) | 7.1 ms per combination, 0 rank-2 residuals in 2,000 |
| rank-2 table, full | `rank2_loop.sh` (ten capped runs of 54,925 combinations, 350 to 383 s each, and the aggregate) | 60.3 min | 0 exact, 0 stabilizer, 0 rank-2 residuals at j = 1 and at j = -1; 76,971 collisions decided exactly, 1 key-zero row brute-forced |
