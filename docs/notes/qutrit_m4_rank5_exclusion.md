# Excluding rank 5 for |N>^4 and |H3>^4: design and costing note

Status (2026-09-22; section 8 added 2026-09-23 with the launch plan after
the review, which supersedes section 7's lists, partitions, and pod
commands). Design and costing only; nothing here is a bound. The
cells stand at 5 <= chi(N^4) <= 7 (`bounds/N-m4-lower-5.json`,
`bounds/N-m4-upper-7.json`, the Lean witness `norrell_m4_stabRank_le_seven`)
and 5 <= chi(H3^4) <= 8 (`bounds/H3-m4-lower-5.json`,
`bounds/H3-m4-upper-8.json`). There is no rank-6 witness at either cell: the
"rank 6" phrases in the two upper-bound files record where numerical rank-6
searches plateaued (a floor of sqrt(211/4043) for N, sqrt(70 - 37 sqrt 3)/12
for H3), not decompositions. Excluding rank 5 therefore moves the cells to
6 <= chi <= 7 and 6 <= chi <= 8, and does not close either.

Two routes are costed. The route the H^6 exclusion suggests, a one-qutrit
base slice that is a full 5-cover of |M>^3 over the 30,240-state
three-qutrit dictionary (section 3), is out of reach: its pivot-pair
enumeration is four to five orders of magnitude more kernel work than the
H^6 census (sum of M^2 over the pivot pairs 8e13 for N and 2.4e14 for H3
against about 2e9), of order 10^4 CPU-hours with a compiled kernel. The route this note recommends, a two-qutrit base slice that is a
full 5-cover of |M>^2 over the 360-state two-qutrit dictionary (section 4),
is free by the same structural results, and its enumeration has already
been run to completion in Python (197,440 covers for N, 188,451 for H3,
about 10 minutes each), with the stage A matcher ported and measured at
50 to 70 ms per cover in Python. The whole exclusion projects to about 20
CPU-hours for N and 12 for H3 without any compiled kernel (section 6).

Notation. |M> is |N> = (|0> + |1> - 2|2>)/sqrt 6 or |H3>, the Fourier
eigenvector with amplitudes proportional to (1 + sqrt 3, 1, 1); every
amplitude is nonzero. psi_m = |M>^m, alpha_k the single-copy amplitudes.
N_n is the number of n-qutrit stabilizer states up to phase, in the order
of `dictionary(3, n)`: N_1 = 12, N_2 = 360, N_3 = 30,240, N_4 = 7,439,040.
G_n is the unitary symmetry group of psi_n (the single-copy Clifford
stabilizer of |M> on each copy, 6 elements for N and 4 for H3, and the
permutations of the copies): |G_2| = 72 (N), 32 (H3); |G_3| = 1296 (N),
384 (H3). w = exp(2 pi i / 3).

## 1. Slice structure lemmas for p = 3

One qutrit (the lemma of `verify_challenge/slice_lift.py`). A four-qutrit
stabilizer term s sliced along one qutrit, u^(k) = (<k| (x) I) s, is either
full (all three slices nonzero) or local (s = |k_0> (x) v). For a full term
the stabilizer group contains lambda X Z^b (x) Q with Q a three-qutrit Pauli
and lambda a cube root of unity, so u^(k+1) = lambda w^{bk} Q u^(k), i.e.

    u^(k_0 + t) = w^{q(t)} Q^t u^(k_0),    q(t) = a t^2 + b t,

with the two phases (q(1), q(2)) = (a + b, a + 2b) ranging freely over
F_3^2 as (a, b) does. The class of Q^2 modulo Stab(u) is determined by the
class of Q, so a full term with base slice u has 27 classes x 9 phase
patterns = 243 shapes and a local term at k_0 has one. The count closes the
four-qutrit dictionary exactly: 30,240 x 243 + 3 x 30,240 = 30,240 x 246 =
7,439,040 = 3^4 (3 + 1)(9 + 1)(27 + 1)(81 + 1).

Two qutrits (the lemma of PR #86, `research/constructions/two_qutrit_slice.py`
and `docs/notes/constructions_2026_09.md`, section 2026-09-21). Along
qutrits 1, 2 the nonzero slices of a term form an affine flat pi(F) of
F_3^2 of dimension 2, 1 or 0. A nine-slice term has

    u^(x_0 + x) = w^{q(x)} Q_2^{x_2} Q_1^{x_1} u^(x_0)

with two-qutrit Paulis Q_1, Q_2 and q a quadratic on F_3^2 with q(0) = 0
(243 of them); the class map x -> class(Q_2^{x_2} Q_1^{x_1}) is linear (81),
so 19,683 shapes per base slice. A line term along direction a has slices
w^{q(t)} Q^t u with 81 shapes per direction (4 directions), a point term
one. The count closes the dictionary: 360 x 19,683 + 12 x 360 x 81 + 9 x
360 = 7,439,040. Both lemmas are the p = 3 form of the qubit lemma used for
H^6: fourth roots and a Z_4-valued form become cube roots and an F_3
quadratic, and the class map is again linear.

## 2. Every rank-5 decomposition has an all-visible base slice, at 1 + 3 and at 2 + 2

Suppose psi_4 = sum_{i=1}^5 c_i s_i. Both facts below are from earlier
sessions and are not re-derived here.

Fact A (one-qutrit slices, the "no minimal slice" result,
`docs/notes/constructions_2026_09.md`, section 1 and "sharpest negative
facts" item 3; script `research/constructions/relaxed_lift.py`). A
one-qutrit slice of a rank-5 decomposition has at most 5 nonzero terms and
at least 4, since chi(psi_3) = 4 (`bounds/N-m3-lower-4.json`,
`bounds/H3-m3-lower-4.json`) and every alpha_k is nonzero. A slice with
exactly 4 nonzero terms is a minimal decomposition of psi_3, hence the
unique rank-4 decomposition up to G_3 (`research/constructions/data/N_m3_rank4.json`,
`H3_m3_rank4.json`), and case [A] of `relaxed_lift.py` (one local term,
two Pauli-matched slices, a stabilizer residual on the third) found 0
lifts for N and for H3 at R = 5. So every one-qutrit slice, along every
qutrit and at every value k, has all five terms nonzero: every term is full
along every qutrit.

Fact B (two-qutrit slices, PR #86). A two-qutrit slice has at least 3
nonzero terms (chi(psi_2) = 3), and PR #86 showed that no rank-5 or rank-6
decomposition of psi_4 has a two-qutrit slice with exactly 3 nonzero terms
(27 (decomposition, x_0) pairs for H3, 90 for N, 0 exact completions). So
every two-qutrit slice point has at least 4 of the 5 terms nonzero.

Consequence at 1 + 3. By Fact A, at every qutrit and every value k the
slice sum_i c_i u_i^(k) = alpha_k psi_3 has all five terms nonzero: the base
slice at any (qutrit, k) is a full 5-cover of psi_3 by three-qutrit
stabilizer states, and the other two slices are w^{q(t)} Q_i^t u_i termwise
by section 1. There is nothing to choose: 4 qutrits x 3 values = 12 base
points, all all-visible, and one suffices (the monomial symmetry of |M>,
the swap 0 <-> 1 for N and 1 <-> 2 for H3, identifies two of the three
values). This is the counting lemma of the H^6 note with the counting
removed.

Consequence at 2 + 2. Fix qutrits 1, 2. By Fact A every term is full along
qutrit 1 and along qutrit 2, so pi(F_i) is not a point and its direction
is not (0, 1) or (1, 0): each term is a nine-slice term or a line term
along a diagonal direction (1, 1) or (1, 2). A line term is absent at 6 of
the 9 points; two line terms would be absent together at some point (two
6-sets in a 9-set meet), leaving at most 3 visible terms there, against
Fact B. So at most one term is a line term along qutrits 1, 2. If there is
none, all nine points are all-visible; if there is one, the three points of
its line are. Every diagonal line of F_3^2 meets the point (0, 0), or an
image of it under the monomial symmetry of |M> acting on qutrits 1 and 2
and the swap of those qutrits: for N the orbit of (0, 0) is
{00, 01, 10, 11} and for H3 the orbit of (1, 1) is {11, 12, 21, 22}, and
each of the six diagonal lines meets the respective orbit (checked by
listing them: for N the lines {00, 11, 22}, {01, 12, 20}, {02, 10, 21},
{00, 12, 21}, {01, 10, 22}, {02, 11, 20} contain 00, 01, 10, 00, 01, 11;
for H3 they contain 11, 12, 21, 12, 22, 11). Those symmetries act on
qutrits 1, 2 only, so they carry rank-5 decompositions to rank-5
decompositions and commute with G_2 on qutrits 3, 4. Hence: every rank-5
decomposition is G-equivalent to one whose slice at x_0 = (0, 0) (N) or
x_0 = (1, 1) (H3) along qutrits 1, 2 has all five terms nonzero, and that
slice is a full 5-cover of psi_2 by two-qutrit stabilizer states, with the
eight other slices given by the two-qutrit lemma. One base point per cover,
and the covers are listed up to G_2.

The matcher itself should not assume Facts A and B beyond the choice of
base point, as the H^6 matcher does not assume property P: it allows every
flat through x_0 (the plane, the two coordinate lines, the two diagonal
lines, the point) and lets the equations decide. Facts A and B only
guarantee that the enumerated bases are complete.

Exactness. All amplitudes lie in Q(w, sqrt 3) = Q(zeta_12) (the states in
Q(w), |N> in Q, |H3> in Q(sqrt 3) after dropping the overall
normalisation). A prime p splits this field iff p = 1 mod 12; both
65521 = 1 + 12 x 5460 and 2013265921 = 1 + 12 x 167,772,160 do (both are
1 mod 48, so w, i and sqrt 3 = (2w + 1)/i all exist mod p, and
`Field3` in `research/qutrit_m4_rank5/cover_census.py` checks
w^2 + w + 1 = 0 and (sqrt 3)^2 = 3). The same two primes as for H^6 serve,
with the same superset logic: the modular kernel and the modular matcher
list supersets, every candidate is re-decided mod the second prime and in
floating point, and a hit is confirmed as a decomposition of psi_4.

## 3. Route 1 + 3: full 5-covers of psi_3 over 30,240 states (out of reach)

Counts (`research/qutrit_m4_rank5/cover_census.py`; the setup ran in 25 s
per orbit). Pivot orbits of G_3 on the 30,240 states, partners (one per
Stab(pivot)-orbit among the members in orbits at or above the pivot's, as
in `slice_lift.all_decompositions` and the H^6 enumerator), and the number
M of members above the partner, which sets the cost of a pivot pair:

| cell | states | \|G_3\| | pivot orbits | pivot pairs | mean M | sum of M^2 |
|---|---|---|---|---|---|---|
| H^6 base, |H>^3 (for comparison) | 1,080 | 48 | 48 | 14,280 | about 1,000 | about 2e9 (459 s at 0.21 s per (M/1000)^2) |
| N, |N>^3 | 30,240 | 1,296 | 88 | 386,109 | 12,455 | 8.07e13 |
| H3, |H3>^3 | 30,240 | 384 | 170 | 1,248,112 | 11,784 | 2.41e14 |

Cost. The compiled 5-cover kernel (`cpp/src/cover5.cpp`) costs
0.21 s (M/1000)^2 per pivot pair at residue dimension 5 (the H^6 note,
section 3), and its inner loop is linear in the residue dimension, which
here is 27 - 4 = 23. Extrapolated: N 0.21 x 8.07e7 x 23/5 = 7.8e7 s, about
21,000 CPU-hours; H3 2.3e8 s, about 65,000 CPU-hours; without the dimension
factor, 4,700 and 14,000 CPU-hours. A 16-vCPU pod gives 384 CPU-hours per
day. The Python reference could not even finish one pivot pair at
M = 30,239 inside the 20-minute sampling cap (the first pair of the N
census ran for 2.6 hours before being killed), so the number of full
5-covers of psi_3 is not known; whatever it is, the enumeration alone rules
this route out. The per-cover matching would be `slice_lift.lifts` with
five terms (81 options per term on slice k_0 + 1, a 81^2 x 81^3 meet in the
middle in C^27, then 3^5 phases on slice k_0 + 2 by a linear solve), of
order a second per cover in Python and tens of milliseconds compiled, and
is not the bottleneck. This is the estimate; nothing at 1 + 3 was measured
beyond the pair census above.

What would have to change: a restriction of the three-qutrit dictionary
strong enough to cut sum M^2 by three orders of magnitude. Facts A and B
give a little (at most one base state is a product |y> (x) v along each of
the three unsliced qutrits, which removes at most about 3,000 of the 30,240
states as candidates for four of the five terms), nowhere near enough. The
2 + 2 route is the change.

## 4. Route 2 + 2: full 5-covers of psi_2 over 360 states (measured)

Enumeration (`cover_census.py census ORBIT 2`, one core at nice 19; the
Python reference kernel, no compiled code). Full r-covers of psi_2 by
distinct independent two-qutrit stabilizer states, one per G_2 orbit as far
as the pivot and partner reductions go, decided mod 65521 with every
candidate re-decided mod 2013265921 and in floating point (`is_cover`,
`is_full`):

| cell | pivot orbits | pivot pairs | full 3-covers | full 4-covers | full 5-covers | 5-cover candidates | time |
|---|---|---|---|---|---|---|---|
| N | 17 | 1,209 | 29 | 1,403 | 197,440 | 5,101,468 | 526 s |
| H3 | 23 | 2,390 | 6 | 1,211 | 188,451 | 8,654,981 | 904 s |

The time is dominated by the numerical re-decision of the candidates (about
100 microseconds each); the kernel itself is seconds. The 3-cover counts
are the rank-3 decompositions of psi_2 up to G_2; the stored lists
(`research/constructions/data/N_m2_rank3.json`, 30 entries;
`H3_m2_rank3.json`, 9 entries) come from `slice_lift.all_decompositions`,
which lists one member per orbit "and possibly more", so 29 <= 30 and
6 <= 9 are the expected direction of the difference, but the
reconciliation (every stored decomposition full and G_2-equivalent to a
listed one) has not been run and is step 1 of the plan.

Degenerate covers, listed as in the H^6 note from the 3-covers and 4-covers
(a 3-cover plus two states of its span or a pair parallel modulo it, a
4-cover plus a state of its span; multisets with a repeated state):

| cell | dependent, distinct states | of which kappa = 1 / 2 | repeated | states in the span of a 3-cover | 4-covers with an empty span |
|---|---|---|---|---|---|
| N | 12,175 | 12,155 / 20 | 5,786 | 0, 1 or 4 | 78 of 1,403 |
| H3 | 6,112 | 6,112 / 0 | 4,880 | 1 | 8 of 1,211 |

For comparison the H^6 census had 5,939,465 full 5-covers from 1,080
states and 14,280 pivot pairs, with 12,390 dependent and 13,852 repeated
covers. Here the covers are thirty times fewer and the degenerate lists
about the same size, so the degenerate stages weigh more in proportion.

Matching (`research/qutrit_m4_rank5/match_prototype.py`, a port of stage A
of `verify_challenge/slice_cover.py` to p = 3 and a two-qutrit base; distinct
independent base states only). Per term the options at the two coordinate
slices x_0 + e_1, x_0 + e_2 are the 27 vectors w^l Q u (9 Pauli classes
modulo Stab(u), 3 cube roots) and absent, 28 in all, against 33 for H^6.
The coefficient family is a point (the base states are independent), so
each coordinate slice is one meet in the middle on a random functional mod
65521 (28^2 hashed, 28^3 probed, every collision decided on the whole
9-vector mod 65521, then mod 2013265921 and over C), and the joined states
are the product of the two solution lists. The absence pattern at the two
coordinate slices fixes the flat type of each term: present at both, a
plane; present at one, that coordinate line; absent at both, one of the
point and the two diagonal lines. The six composite points x_0 + x,
x in {11, 20, 02, 12, 21, 22}, are solved point by point over the
structure lemma's codes: a plane term with coordinate codes (k_1, l_1),
(k_2, l_2) has class(Q_2^{x_2} Q_1^{x_1}) at x (a product table computed
once per base state) and phase q(x) + g(x) with q ranging over the 27
quadratics that agree with l_1, l_2 on the coordinate directions; a
coordinate-line term has 3 phases at its third point; the absent-at-both
type has 1 + 81 + 81 rows (point, two diagonal lines, each 27 codes at the
first point and 3 phases at the second). Every raw hit is confirmed as a
decomposition of psi_4 (residual, rank 5, nonzero coefficients, and the
rank test mod 2013265921 agreeing with the numerical one).

Measured (300 covers per cell drawn uniformly from the census, one base
point each, one core at nice 19 on the laptop):

| cell | x_0 | mean per cover | median | max | coordinate-slice solutions (both slices agree) | joined states | composite candidates | hits |
|---|---|---|---|---|---|---|---|---|
| N | (0, 0) | 70 ms | 51 ms | 0.80 s | 1 to 40 per slice; modes (2, 2) 70, (3, 3) 48, (1, 1) 46, (9, 9) 31 | 18,028 | 6.9e6 | 0 |
| H3 | (1, 1) | 50 ms | 44 ms | 0.66 s | modes (3, 3) 95, (1, 1) 82, (2, 2) 52, (9, 9) 42; max 34 | 6,405 | 3.0e6 | 0 |

Unlike H^6, where 146 of 160 runs died on the first coordinate slice, every
sampled cover here has solutions on both coordinate slices (the slice
ratios alpha_x / alpha_{x_0} are 1 for several x, so the base cover itself
is always a solution up to phases), and the composite stage does the
excluding. The cost is still small because the composite equations are
solved point by point with at most 3^5 phase combinations per point for
five plane terms.

Controls run on the prototype (both pass): (i) planted instances, 8 random
five-term decompositions of a random target by random four-qutrit
stabilizer states with small integer coefficients, recovered from their
base slice at a random all-visible x_0 with no spurious hit, 0.13 s per run;
(ii) the product decompositions phi (x) (rank-3 decomposition of psi_2) of
phi (x) psi_2 for a random full-support two-qutrit stabilizer state phi, 5
stored decompositions x 2 base points for each of N and H3, all 20 found,
none missing. The second control caught an int64 overflow in the mod
2013265921 re-check (products of two reduced residues exceed 2^63 unless
reduced pairwise), which is the kind of defect the H^6 pipeline's
`--no-native` cross-check exists for.

## 5. Controls the pipeline needs

1. Enumeration cross-check: the 29 (N) and 6 (H3) full 3-covers against
   the stored rank-3 lists (fullness of each stored decomposition and
   G_2-equivalence), and the full 4-covers against
   `slice_lift.all_decompositions(orbit, 2, 4)` restricted by `is_full`,
   as the H^6 note did at r = 3 and 4.
2. Planted instances and product decompositions, as run on the prototype,
   extended to bases with a diagonal line term (plant a four-qutrit term
   whose flat projects to a diagonal line) and to dependent and repeated
   bases once stages B and C exist.
3. m = 3 control, the analogue of the H^6 control 1: slice the unique
   rank-4 decompositions of psi_3 along two qutrits. Each has one point
   term (k = 0 at (2, 2, 2) for N, (0, 0, 0) for H3) and three full-support
   terms, so at the point term's two-qutrit projection all four terms are
   visible and the base is a 4-cover of |M> by single-qutrit states.
   Four vectors in C^3 are dependent, so this control exercises the
   affine-family path (kappa >= 1), not stage A; the expected output is the
   stored class (and any other rank-4 decomposition of psi_3 with the same
   base), nothing outside `N_m3_rank4.json` and `H3_m3_rank4.json`.
4. Witness controls at m = 4. The rank-6 witnesses named in the task do
   not exist (see the status paragraph). The available witnesses are the
   rank-8 witness of `bounds/H3-m4-upper-8.json` (eight terms in the JSON:
   along qutrits 1, 2 at x_0 = (0, 0) all eight are visible, two of them
   as point terms, so the base is an 8-multiset in C^9, dependent) and the
   rank-7 Lean witness of |N>^4 (`lean_proofs/LeanProofs/M4StabRank.lean`:
   two 3-flats y_1 = 2 and y_0 = 2, three planes with two coordinates fixed
   at 2, two full-support states; along qutrits 0, 1 all seven are visible
   at (2, 2), with line terms along both coordinate directions). Both need
   the matcher at r = 7 and 8 with the family and block machinery of
   stages B and C, and the N witness first needs its seven terms exported
   from the Lean file to a witness JSON. They are the counterpart of the
   H^6 control 2 and should be run before the exclusion is trusted.
5. The `--no-native` style cross-check is moot if the pipeline stays in
   Python; if a compiled matcher is written for stage B, one batch per
   stage must be replayed by the reference.

## 6. Verdict and plan

Costs, one base point per cover, from the measured stage A rate and the
H^6 degenerate rates scaled to the option count (28^2 x 28^3 = 1.7e7
feature pairs per 1-parameter dense solve against 39e6 for H^6, so about
half the 8.9 s of H^6, taken as 4 s per dependent cover; repeated covers
at the H^6 test-batch rate of 0.2 s):

| cell | stage A (measured rate) | stage B, dependent (estimate) | stage C, repeated (estimate) | total, Python |
|---|---|---|---|---|
| N | 197,440 x 70 ms = 3.8 CPU-h | 12,175 x 4 s = 13.5 CPU-h | 5,786 x 0.2 s = 0.3 CPU-h | about 18 CPU-h |
| H3 | 188,451 x 50 ms = 2.6 CPU-h | 6,112 x 4 s = 6.8 CPU-h | 4,880 x 0.2 s = 0.3 CPU-h | about 10 CPU-h |

Verdict. The 1 + 3 route does not fit a 16-vCPU pod in a day by two orders
of magnitude even compiled (section 3). The 2 + 2 route fits in Python
alone: about 30 CPU-hours for both cells together, two to three hours of
wall time on 16 vCPUs, with no compiled kernel; an 80x compiled stage A
would save 6 CPU-hours and is not worth writing unless the stage B
estimate is badly off (it is the least certain number here, being scaled
from H^6 rather than measured; the first measured stage B batch settles
it). The enumeration is already done and takes 15 minutes to redo.

Plan, in order.

1. `verify_challenge/slice_cover3.py`: the p = 3 enumerator of
   `research/qutrit_m4_rank5/cover_census.py` moved next to
   `slice_cover.py` (or `slice_cover.py` parametrised by p), with the
   r = 3, 4 cross-checks of section 5 item 1 as tests, and the degenerate
   lists written once with their hash.
2. The matcher: the stage A port of `match_prototype.py`, plus the affine
   family (kappa >= 1, the Laplace-feature dense solve of `slice_cover.py`
   with 28 options per term) and the blocks for repeated states, ported
   from `slice_cover.py`; the composite stage keeps the six-point,
   point-by-point solve with the three flat types of the absent-at-both
   pattern. Generic in r so that the witness controls at r = 7, 8 run.
3. Controls of section 5: items 1 to 4, with the N witness exported from
   Lean first.
4. `research/qutrit_m4_rank5/`: partition (stage A by the 1,209 and 2,390
   pivot pairs, stages B and C round robin over the sorted degenerate
   lists, target about 600 s per batch), `batch.py K` with hashed
   deterministic records, `aggregate.py --recheck 2`, and an attested
   certificate per cell, as in `research/h6_rank5/`; base point (0, 0) for
   N and (1, 1) for H3 with the argument of section 2 stated in the bound
   files' notes.
5. Run both cells on the pod (about 30 CPU-hours), aggregate, file
   `bounds/N-m4-lower-6.json` and `bounds/H3-m4-lower-6.json` at the
   attested tier with dependencies declared: Fact A (`relaxed_lift.py`,
   case [A], and the rank-3 exclusions at m = 3), Fact B (PR #86's tables,
   104 s for H3 and 431 s for N, not re-run), the two slice lemmas, the
   base-point argument of section 2, the modular supersets with exact
   re-decision, and the controls.

Everything above depends on PR #86 and on the one-qutrit result only
through the base-point argument of section 2; the matcher does not use
them.

## 7. Run plan (2026-09-22, pipeline built)

Code: `research/qutrit_m4_rank5/` (`matcher.py`, `common.py`, `driver.py`,
`batch.py`, `aggregate.py`, README), the certificates
`verify_challenge/cert_n_m4_rank5_attested.py` and
`cert_h3_m4_rank5_attested.py` (`CERTIFIED chi(N^4) >= 6`,
`CERTIFIED chi(H3^4) >= 6`, seed 20260922, two re-runs), and the draft
bound files `N-m4-lower-6.json.draft`, `H3-m4-lower-6.json.draft` next to
the code (attested tier, placeholders for compute hours, hardware and
date; they move to `bounds/` once the manifests exist). Everything is the
Python reference; there is no compiled kernel. All times below are one
core at nice 19 on an 18-core laptop at load average about 20 (other
sessions), so they overstate an unloaded pod.

Matcher. The port of `slice_cover.SliceMatcher` to p = 3 and a two-qutrit
base with the coefficient family (kappa >= 1) and the block treatment of
repeated states, the composite points solved one at a time over the
shapes still alive, generic in the number of terms and the base dimension.
Two defects found by the controls and fixed: the block reconstruction
inherited `slice_cover._affine_solve_C`, which solves with numpy's
relative singular-value cutoff, so the sum-zero direction of two copies
hit by equal phases (an A K of order 1e-16 rather than 0) was inverted
into a garbage pin instead of a consistency condition, producing
reconstructions that violated their own per-slice constraints (planted
repeated bases: 26 such candidates in 8 runs for H3 before the fix, 0
after; the planted decomposition was recovered in every run either way);
the matcher now uses a truncated-SVD solve with the scale max(1, s_0),
patched into `slice_cover` for `Family.restrict` as well. And a slice
equation with no ordinary term (every distinct state repeated, which the
m = 3 control produces since chi(|M>) = 2) is decided directly.

Census (`driver.py census`, the reference kernel with every candidate
re-decided; hashed lists of the full 3-, 4- and 5-covers in
`results/ORBIT/kernel_census.json`): N 1,209 pivot pairs, 197,440 full
5-covers, 5,101,468 candidates, 629 s; H3 2,390 pairs, 188,451 covers,
8,654,981 candidates, 1,128 s. Both equal the section 4 counts.

Degenerate covers (`driver.py degenerate --write`, the H^6 criterion: a
multiset over a 3-cover or 4-cover with a coefficient family in which no
unrepeated state is dead): N 12,175 dependent (all kappa = 1) and 5,910
repeated (5,736 of pattern (2, 1, 1, 1), 87 of (2, 2, 1), 87 of (3, 1, 1));
H3 6,112 dependent and 4,888 repeated (4,852 / 18 / 18). The dependent
counts equal section 4's; the repeated counts exceed the prototype's
(5,786 and 4,880) because the prototype listed multisets without the
fullness test of the family.

Controls (`results/ORBIT/control_*.json`; items 1 and 2 pass for both
cells, item 3 passes for N and is unfinished for H3, item 4 was not run).

1. `control-covers`: the 29 (N) and 6 (H3) full 3-covers fall into 8 and
   3 G_2 classes, exactly the classes of the stored rank-3 lists (30 and 9
   decompositions, every one full); the 1,403 and 1,211 full 4-covers fall
   into 478 and 554 classes, exactly the classes of the full members of
   `slice_lift.all_decompositions(orbit, 2, 4)` (1,403 of 1,909 and 1,211
   of 1,306 decompositions full). Nothing missing, nothing extra.
2. `control-planted`, 8 instances per case, both cells: generic bases with
   random flats, bases with two diagonal line terms, dependent bases from
   the stage B list, repeated bases from the stage C list; every planted
   decomposition recovered from its base slice, 0 candidates rejected by
   the final check (further genuine decompositions with the same base: N
   0, 0, 1, 9; H3 0, 0, 0, 15). `control-product`: the 30 (N) and 9 (H3)
   stored rank-3 decompositions of |M>^2, tensored with a random
   full-support two-qutrit stabilizer state, recovered at all 9 base
   points (270 and 81 runs).
3. `control-m3` (2 + 1 slicing of |M>^3 against the full 4-multisets of
   the 12 single-qutrit states, every base dependent): for N, 924
   multisets; the stored rank-4 decomposition is recovered from its own
   all-visible base, and 200 further sampled (multiset, base point) pairs
   produce genuine rank-4 decompositions only in the stored G_3 class (9
   genuine hits in 201 runs, 0 refused, 287 s; 144 s with the same
   counts after the solver changes made for the witness control): PASS.
   For H3 the same
   command (`control-m3 H3 --sample 200`) ran for 60 minutes without
   finishing and was killed; it logged no partial counts (the multiset
   enumeration and the first runs print nothing until the end), so the H3
   case is UNFINISHED. The H3 bases are more expensive because many of
   them have every distinct state repeated or kappa = 2, and the block
   translate sets multiply; the control should be rerun with a smaller
   `--sample` and its per-run timing printed.
4. `control-witness`: run one base per process under a 10-minute cap;
   no base passed yet. Four of the nine bases were run (N bases 0, 3, 4,
   H3 base 0); each aborted with its reason recorded, none was killed.
   The kappa-3 base is refused at the first coordinate slice (millions of
   exact solutions), the bases with repeated states finish both
   coordinate slices but their join exceeds 200,000 states or the cap.

Rates (per cover, one base point). Stage A (`driver.py sample`, 300
census covers): N 126 ms mean, 100 ms median, 1.13 s max; H3 100 ms, 82
ms, 0.98 s; the coordinate-slice solution histograms have the modes of
section 4 exactly ((2, 4) 70, (3, 9) 48, (1, 1) 46, (9, 81) 31 of 300
for N). Stages B and C (`driver.py degenerate --sample 8 --cap-s 440`,
evenly spaced over each pattern's list, measured before the
reconstruction fix, which does not touch the pinned-family solves that
dominate): N (1, 1, 1, 1, 1) 5.66 s mean, 3.85 s median, 11.2 s max;
(2, 1, 1, 1) 2.48 s; (2, 2, 1) 66.9 s (2 covers, capped); (3, 1, 1) 6.93 s.
H3 5.00 s, 0.61 s, 58.3 s (2, capped), 0.13 s. The stage B rate is the
number section 6 could not measure: 5 to 6 s per cover against the 4 s
scaled from H^6. Stage C has a heavy tail: a separate 12-cover sample of
the N stage C list had median 1.7 s but one kappa = 1 cover with a
repeated state at 234 s (32,092 second-slice solutions), and the first
sampling attempt spent two hours on one cover before it was killed; the
(2, 2, 1) covers (two blocks, 46^2 translate-set combinations per slice)
are a minute each. A stage C batch can therefore run much longer than
its estimate.

Partition (`partition_N.json`, `partition_H3.json`, `driver.py partition
--target-s 540 --target-bc-s 540 --match-ms 126|100`). N: 37 stage A
batches (batches 0 to 36, 1 to 243 pivot pairs each, 7.1 CPU-h at the
census seconds plus the sampled matcher rate), 128 stage B batches (37 to
164, 95 or 96 covers, 19.2 CPU-h), 39 stage C batches (165 to 203, 151 or
152 covers, 5.7 CPU-h): 204 batches, 32.0 CPU-h. H3: 28 stage A batches
(0 to 27, 1 to 1,407 pairs, 5.6 CPU-h), 57 stage B batches (28 to 84, 107
or 108 covers, 8.5 CPU-h), 8 stage C batches (85 to 92, 611 covers, 1.1
CPU-h): 93 batches, 15.2 CPU-h. About 47 CPU-hours for both cells at the
loaded-laptop rates, against the 30 of section 6 (the stage B rate and
the stage A load account for the difference); on 15 cores of an unloaded
pod, three to four hours of wall time.

Pod commands (16-vCPU Linux, the H^6 shape; from the repository root
after `uv sync --extra challenge`; rerunning resumes, finished batches
are skipped):

```
mkdir -p research/qutrit_m4_rank5/results/N research/qutrit_m4_rank5/results/H3
seq 0 203 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py N "$0" \
   > research/qutrit_m4_rank5/results/N/batch_"$0".log 2>&1'
seq 0 92 | xargs -P 15 -n 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py H3 "$0" \
   > research/qutrit_m4_rank5/results/H3/batch_"$0".log 2>&1'
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --recheck 2 --recheck-seed 20260922
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --recheck 2 --recheck-seed 20260922
```

The last two write `batch_manifest_N.json` and `batch_manifest_H3.json`
and print the claim lines; then fill the placeholders of the two draft
bound files and move them to `bounds/`.

Status of what the plan asked for and this session did not finish (the
session ran under a 10-minute cap per job after the first degenerate
sampling attempt hung for two hours on one stage C cover):

- Measured batches: NOT RUN. The rates above come from the samples; the
  first stage A, B and C batch of each cell should be run on the pod
  before the rest (`batch.py N 0`, `batch.py N 37`, `batch.py N 165`,
  `batch.py H3 0`, `batch.py H3 28`, `batch.py H3 85`) and the partition
  regenerated with `--match-ms` and the sample rates corrected if they are
  far off; the partition and degenerate-list hashes then change, so no
  batch may be kept across a repartition.
- The aggregate has not been exercised on stored batches of these cells;
  it is the H^6 aggregate with the cell, base point and per-cell paths
  added.

The witness control (2026-09-22, `driver.py control-witness`, one base per
process, 10-minute cap per run, one core at nice 19 on the loaded
laptop). The rank-7 N witness (the seven Lean terms, exported by
`export-witness`) has 6 all-visible (base, x0) pairs, all at x0 = (2, 2)
and each from one qutrit pair: bases 0, 1, 2 and 5 have seven distinct
states with kappa = 3, base 3 has three repeated pairs and one ordinary
state (multiplicities (2, 2, 2, 1), kappa = 1), base 4 one repeated pair
and five ordinary states ((2, 1, 1, 1, 1, 1), kappa = 2). The rank-8 H3
witness (`bounds/H3-m4-upper-8.json`, four m = 3 terms tensored with a
rank-2 decomposition of |H3>) has 3 pairs at x0 = (0, 0): bases 0 and 2
are four repeated pairs with no ordinary state ((2, 2, 2, 2), kappa = 1
on the four distinct states), base 1 eight distinct states with
kappa = 4. The earlier attempts (the N run killed with exit 137 after 63
minutes on the 124 GB pod, the H3 run stopped after 4 hours without
output) are explained by three separate costs, each now bounded with an
explicit failure.

1. Memory: `export-witness` checked that the seven Lean terms are
   stabilizer states by membership in `dictionary(3, 4)`, which has
   7,439,040 states of dimension 81; the dictionary, its phase patterns,
   the two field images and the lookup table run to tens of GB and this
   is the most likely place the pod run died (the export runs first, and
   the matcher on any base uses less than 7 GB, below). The terms are now
   recognised directly (`stabilizer_spec`: affine support of size 3^k,
   quadratic phase exponent on its coordinates, fitted exactly mod 3) and
   stored with their (k, x0, W, Q, l) specs; the export takes seconds.
2. The dense coefficient-family solve with many ordinary states. N base 0
   (kappa = 3, seven ordinary states, 28 options each) hits the
   2,000,000-candidate cap of `slice_cover._dense` at the first
   coordinate slice after 25 s (peak RSS 1.4 GB): a 3-parameter family of
   seven terms in a 9-dimensional slice has millions of exact solutions at
   one slice, and the feature product itself is small (614,656 x 21,952
   with 70 Laplace features, about 0.4 GB). H3 base 1 (kappa = 4, eight
   ordinary states) is the 4-hour run: 28^4 x 28^4 sides with 252
   features, chunks of 32 rows against 614,656, about 1e14 flops before
   the first candidate is decided, and the same candidate explosion after
   it. These bases are refused, not computed: `Budget.check_dense`
   estimates the feature matrices from the side products and the
   parameter count before the solve allocates, and the candidate cap
   raises `BudgetExceeded` with the count. Bases 1, 2 and 5 of N have
   the same shape as base 0 and were not run.
3. The product over the blocks' translate sets when several states repeat.
   With no ordinary state (H3 bases 0 and 2) the old path enumerated
   46^4 = 4,477,456 translate-set selections per slice equation, each
   with three annihilators computed in Python; and nothing pins the
   1-parameter family before the reconstruction, so the states after the
   two coordinate slices are the full product of the two solution lists.
   Both are replaced. The selections are now found by a meet in the
   middle over two halves of the blocks: per column count (a, b) of the
   halves the dependence of [left translates | right translates | slice]
   under a random projection to a + b + 1 coordinates mod 65521 is a
   Laplace expansion into minors of the two sides, a feature product as
   in `_dense`; every candidate is decided exactly (independent
   translates over F_65521, the slice in their span over the three fields,
   every coordinate nonzero, the rule `_join_blocks` already applied to a
   pinned family). And a block whose two copies occupy two translate
   classes at a slice contributes c_1 w^{l_1} Q_{k_1} u + c_2 w^{l_2}
   Q_{k_2} u, so its merged coefficient c_1 + c_2 is one of the nine values
   a_{k_1} w^{-l_1} + a_{k_2} w^{-l_2} of the slice's coordinates a on the
   translates, each a linear condition that pins the family exactly over
   the three fields (`_pin_by_blocks`, the split-tracking idea of the H^6
   control applied before the family is pinned; with no ordinary state the
   coordinates do not depend on the family member, so the condition is
   exact; this step was removed in commit e548e5a, when the block-only
   bases moved to `BlockOnlyMatcher`, and the family path pins the family
   by the slice restrictions alone). The second coordinate slice is now solved once against the
   initial family and joined with the first slice's states by family
   compatibility (pinned against pinned by the coefficient vector mod
   65521, a dictionary lookup; otherwise by the annihilator of the
   direction space, then the exact restriction), which gives the same
   states as solving it once per state. The block projectors are cached
   per selection and the matmul mod 2013265921 runs in int64 (16-bit
   split) instead of object arithmetic. Per-run caps: `--max-rss-gb`
   (checked before every selection and after every slice from
   /proc/self/statm on Linux, the peak on macOS), `--max-states`,
   `--max-solutions`, `--max-seconds`; every abort writes the base's
   record with the reason and the partial stats.

Regression: the planted control and the m = 3 control for N give the
same counts as before the changes (9 genuine hits in 201 runs, 0
refused; 144 s against 287 s); every solver change is exact or a superset
hash followed by the exact decision, and the m = 3 control exercises the
no-ordinary-state path throughout since chi(|M>) = 2.

Per base (`results/controls/witness_ORBIT_base_K.json`,
`results/ORBIT/control_witness.json`; "solutions" are against the initial
family, "states" after the dead-coefficient and split-tracking filters):

- N base 0 (kappa 3): ABORTED at the first coordinate slice, 2,084,934
  hash candidates, 25 s, peak RSS 1.4 GB.
- N base 3 (kappa 1, blocks 2, 2, 2, one ordinary state): 80,759
  solutions per coordinate slice (42 s and 28 s), 6,785 states after the
  first, the join of the second still running at the deadline (575 s)
  after 3.9 million pairs, 1,201,872 dropped by a dead coefficient and
  2,738,863 by the split tracking, peak RSS 1.5 GB. The pairs come from
  the first-slice states the ordinary state does not pin (kappa still 1),
  each compatible with every pinned second-slice solution on the family
  line; the join runs at about 8,000 pairs per second here, so the base
  needs roughly 15 to 30 minutes on this machine before its composite
  stage, and is the most likely N base to pass on the pod.
- N base 4 (kappa 2, one block, five ordinary states): 87,766 solutions
  per slice (160 s and 171 s), 3,689 states after the first, more than
  200,338 after the join (the states cap; 602,372 pairs dropped by a
  dead coefficient and 449,434 by the split tracking by then), peak RSS
  0.8 GB. Each joined state costs a six-slice composite stage over 46
  translate sets, so the base is a matter of hours on the pod with the
  cap raised.
- H3 base 0 (kappa 1, blocks 2, 2, 2, 2, no ordinary state): 54,260
  translate-set solutions per slice from 1,983,517 hash candidates,
  458,564 pinned states after `_pin_by_blocks` (274 s and 123 s), 5,420
  after the first slice's split tracking (453,144 pruned), more than
  458,564 after the join (the states cap), peak RSS 6.3 GB (the projector
  cache and the candidate lists). The join size comes from the first-slice
  solutions no block pins (no block with two classes moving along the
  family), which are the whole family line and therefore compatible with
  every pinned second-slice state; those pairs are pruned only when the
  second slice's splits are compared with the composite slices.
- N bases 1, 2, 5 and H3 bases 1, 2: not run (bases 1, 2, 5 and H3 base 1
  have the shape refused under item 2; H3 base 2 has the shape of H3 base
  0).

None of the four bases passed within the cap, so the control is still
open; the pipeline is not declared ready on this count. What the runs do
establish: no base exceeds 7 GB, so the 124 GB pod holds 15 at a time;
every abort is recorded with its reason; and the remaining cost is the
number of states after the two coordinate slices at rank 7 and 8, which
the rank-5 batches never approach (their kappa is at most 1 and the
sampled covers have at most 32,092 second-slice solutions). To finish on
the pod (one base per process, all nine at once, each within 100 GB):

```
mkdir -p research/qutrit_m4_rank5/results/controls
{ seq 0 5 | sed 's/^/N /'; seq 0 2 | sed 's/^/H3 /'; } | xargs -P 15 -L 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness "$0" --base "$1" \
   --max-rss-gb 100 --max-states 20000000 --max-solutions 20000000 --max-seconds 0 --verbose \
   > research/qutrit_m4_rank5/results/controls/witness_"$0"_base_"$1".log 2>&1'
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness N --summary
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/driver.py control-witness H3 --summary
```

The kappa-3 and kappa-4 bases (N 0, 1, 2, 5 and H3 1) will abort at the
20,000,000-candidate cap unless it is raised further, and their first
slice then has millions of states; a pass on those bases needs a
different order of solving (two slices jointly, or the family pinned by
the composite points first), not more memory.

Pod outcome (2026-09-22 night, one base per process, `--max-rss-gb 12`):
N base 4 PASSED (the rank-7 witness recovered exactly once, 67 other
genuine rank-7 decompositions, 4,489 s, 1.1 GB); the five kappa >= 3
bases aborted at the candidate cap; H3 base 2 logged "slice (1, 0):
458,564 solutions, 5,420 states after the join, 31 s, 5.6 GB", then
"slice (0, 1): 11,565,760 solutions, 1,958,256 states after the join,
1,346 s, 8.8 GB", and died in the composite stage with no further output
(the resident-set guard only ran at stage boundaries); H3 base 0 has the
same shape.

The block-only matcher (2026-09-23). Two facts about the H3 witness at
bases 0 and 2, checked directly, show that the family path could not
have recovered it however much memory it had. (i) The witness's own
translate selection at x0 + e_2 and x0 + 2 e_2 is the four base states
themselves: at x_3 = 1, 2 the |0> copy of every block vanishes and the
|+> copy is the base slice up to a scalar. The four base states have
rank 3 (that is the kappa = 1), so `_block_only_solutions` rejected the
selection as dependent, and no other selection produces the witness's
codes. (ii) The two copies t_i (x) |0> and t_i (x) |+> of a block never
occupy two classes at any slice (the single-qutrit factor only
contributes a scalar or zero), so `_pin_by_blocks` never pinned the
family, `_join_blocks` skipped every state with kappa = 1, and the
reconstruction used a random point of the family line, at which the
equal-phase condition of slice (1, 0) fails. The 2-million-state join
was therefore going to finish with the witness absent.

A base with no ordinary term now takes a separate route
(`matcher.BlockOnlyMatcher`; the rank-5 batches never reach it, since a
full 5-multiset with every state repeated does not exist, but the m = 3
control does through its (2, 2) multisets). Every one of the eight
slices is solved on its own for the translate selections whose span
contains it (`block_only_slice`: the meet-in-the-middle Laplace hash of
the old path, whose candidates are decided in batches by the singular
values of the stacked column matrices; dependent selections are kept
with their coordinate family a = a0 + N mu; a slice proportional to an
earlier one reuses its selections with scaled coordinates, which for
|H3>^4 along any qutrit pair is all eight, since every slice is a
multiple of |H3>^2). The two points x, 2x of each of the four lines
through x0 are paired: a copy in class k at x is in class 2k at 2x
whatever its shape, so the pairing is exact and by a dictionary lookup.
On a line an independent paired selection gives every block a small
set of configurations of its two copies (both present in one class with
two phase pairs, one present, or one in each of two classes) with the
constraint on (c_1, c_2, lambda) in closed form: a copy alone in its
class keeps its modulus along the line, so its phase at 2x is its phase
at x plus the cube-root shift of the coordinate ratio, and two copies in
one class with distinct phase differences are a 2 x 2 solve; the
configurations pin lambda to a value or leave it free, and a selection
survives its line only if the blocks agree on some lambda (hashing on
the pinned values). Three lines are joined at a time: the two with the
fewest survivors drive (the 4,100 x 4,100 product of the coordinate
lines never runs; the diagonal lines have 100 survivors each here), a
pair passes only if the blocks' lambda sets meet, the third line is
looked up through the per-block class relation of the structure lemma
(`block_relation`, a table over the 118 one-copy class patterns), and the
fourth line is completed: a copy present on two or more of the three
lines is a plane whose codes there follow from a table
(`line_completion`), a copy present on one line is a line term absent
there, and a copy absent on all three is a point or a line along the
fourth direction, read off the residual of the fourth line's two slices
once every other coefficient is pinned (a small direct selection search
over the blocks involved, then phases from the coordinates). Each
distinct completed code set is confirmed exactly once. What this finds:
every decomposition whose selection is independent on at least three of
the four lines. A dependent selection is never searched, since on its
own line it constrains almost nothing (with the slice's parameter free
every configuration of every block is consistent, and 36 percent of the
6,681 dependent selections of an H3 slice survive the line with up to
1e5 combinations each), so a decomposition dependent on two or more
lines is outside the matcher; the record carries
`dependent_lines_limit` and the per-slice dependent counts. The resident
set and the deadline are now checked inside the line, join and
completion loops, and inside the family path's join and composite loops.

H3 base 2 on the laptop under the 10-minute, 8 GB cap: one slice solve
of 17 to 19 s (60,941 selections, 6,681 dependent, 991,762 hash
candidates), seven reuses; per line 54,260 independent paired selections
of which 4,100 (e_1), 4,100 (e_2), 100 ((1, 1)) and 100 ((1, 2)) survive,
36 to 42 s per line; passes [e_2, (1, 1), (1, 2)] and
[e_1, (1, 1), (1, 2)] (driven by the diagonals): 10,000 pairs, 2,736
lambda-feasible, 1,392 triples, 156 s each; [e_1, e_2, (1, 2)] (driven
by e_1 and (1, 2)): 410,000 pairs, 19,664 feasible, 7,256 triples, 56 s;
the fourth pass [e_1, e_2, (1, 1)] started at 369 s and was still running
at the 560 s deadline (record `aborted`, peak RSS 0.60 GB; 597,024 block
leaves, 114,338 combinations, 10,368 completions, 5,184 distinct genuine
rank-8 decompositions confirmed by then). The witness's own triple
(lines e_1, (1, 1), (1, 2) with the dependent e_2 line completed; the
witness has three independent lines) gives 433 block leaves, 338
combinations, 2,592 completions and 1,296 distinct genuine rank-8
decompositions with the witness exactly once, in 61 s: the base admits a
rank-2 stabilizer decomposition of |H3> per block independently, hence
thousands of genuine decompositions, most of the run's time being their
enumeration. The full run needs about 750 s here, so the base does not
pass under the local cap; on the pod (`--max-seconds 0`, the command in
the README) it should complete in well under an hour per base. H3 base
0 has the identical shape. H3 base 1 (kappa 4, eight ordinary states)
is unchanged and refused at the dense-solve cap.

Regression after the change (both under the cap): `control-m3 N
--sample 200` gives the same counts as before (924 multisets, 9 genuine
hits in 201 runs, 0 refused, the stored class recovered from its own
base; 99 s against 144 s), and `control-planted N` recovers all 32
planted decompositions with the same further-decomposition counts
(0, 0, 1, 9) and 0 candidates rejected by the final check.

## 8. Launch plan (2026-09-23, after the review)

What changed since section 7. The review
(`docs/notes/qutrit_m4_rank5_review.md`) found three defects in the stage
C path and one in the aggregate; its second commit fixed the block
reconstruction for cancelling copies, made the final loop of `_complete`
raise `UnpinnedFamily` instead of reconstructing at a random member of an
unpinned family, added the cancel-at-base multisets T + (b, b) to the
degenerate enumeration, and made the aggregate fail on a refusal. The
`_pin_by_blocks` step that section 7 describes was removed in commit
e548e5a: the family is pinned by the slice restrictions alone, and the
block-only bases take `BlockOnlyMatcher`. This session added the
following.

- `Matcher._compatible` cached its index over the second coordinate
  slice's solutions by `id(sols)`. The list is freed at the end of each
  run and the next run's list often gets the same address, so the stale
  index was served and compatible pairs were dropped with no record. The
  product control lost 8 of its 270 hits this way (one solution per
  slice, zero states after the join); a batch of any stage could lose a
  state the same way. The index is keyed on the list object now, with a
  reference kept, and reset per run. Every control below was re-run after
  this fix.
- A single block whose coefficient family still has kappa parameters at
  the final loop is reconstructed with the parameter as an unknown:
  `_block_coordinates` returns the residual coordinates as a0 + A lambda
  over d = d0 + K lambda, and `reconstruct_block` carries lambda next to
  the copy coefficients through its solves, so the copies are decided on
  the whole family at once. This is the configuration `control-m3 N`
  reaches (kappa = 1, one block of two), which aborted with
  `UnpinnedFamily` after the review's fix and passes again (924
  multisets, 9 genuine hits in 201 runs, 0 refused, the stored class
  recovered from its own base, 63 s). `UnpinnedFamily` remains for
  several blocks with kappa > 0 (the parameter would have to be shared
  across the blocks' reconstructions), and for the strict coordinate
  solve: dependent translates of two blocks, a nonzero residual with no
  translate chosen, a residual outside the span at the solve tolerance.
- `batch.py --max-seconds S`: the matcher aborts the running cover at
  the deadline (`BudgetExceeded`, recorded as undecided) and every cover
  not yet started is recorded as undecided with the reason "not run", so
  a batch caught by the stage C tail ends with a record naming what is
  left instead of running for hours. The batch exits 1 and fails the
  aggregate; the record carries `max_seconds` and `aborted_at_deadline`
  outside the deterministic part.
- `driver.cover_class` and the partition by class (below).

Where `UnpinnedFamily` can fire at rank 5. The kappa > 0 case needs a
dependent set of distinct states, and among the stage C covers those are
the (2, 1, 1, 1) covers T + (x, y) with x in span(T): 124 for N and 8 for
H3, one block each, so the multi-block kappa > 0 raise cannot occur on a
listed cover and the single-block case is reconstructed rather than
raised. The dependent-translates raise needs two blocks whose chosen
translates (at most two each) are linearly dependent: `cover_class`
finds 27 such (2, 2, 1) covers for N (blocks in one Pauli orbit, or a
translate of one block in the span of two of the other, as |0+> lies in
the span of |00> and |01>) and none for H3. The two tolerance raises can
fire on any block cover in principle; they did not fire on any cover run
below.

Dry pass (this laptop, one core at nice 19, load average 12 to 16, every
cover under a 90 or 120 s deadline; `UnpinnedFamily` caught and counted).
Every kappa >= 1 stage C cover of both cells was run, all 27 dependent
(2, 2, 1) N covers were attempted (6 before the sample was cut), and the
rest was sampled evenly over each list. The N pass ran before the index
fix; 59 of its 124 kappa >= 1 covers were re-run after it with the same
outcomes (0 hits, 0 reconstructions, 17.6 s median).

| cell | class | covers | run | outcome | reached the final loop | seconds per cover (mean, median, max) |
|---|---|---|---|---|---|---|
| N | (2, 1, 1, 1) kappa >= 1 | 124 | 124 | all decided, 0 hits | 0 | 27.4, 38.0, 51.2 |
| N | (2, 1, 1, 1) kappa = 0 | 15,883 | 147 | all decided, 0 hits | 0 | 0.77, 0.29, 7.7 |
| N | (2, 2, 1) independent | 60 | 3 | all decided, 0 hits | 0 | 5.5, 4.6, 9.2 |
| N | (2, 2, 1) dependent | 27 | 6 | all 6 past the 90 s deadline at a composite slice | 0 | above 90 |
| N | (3, 1, 1) | 87 | 0 | (sampled in section 7: 6.9 s) | | |
| H3 | (2, 1, 1, 1) kappa >= 1 | 8 | 8 | all decided, 0 hits | 0 | 6.4, 5.2, 10.5 |
| H3 | (2, 1, 1, 1) kappa = 0 | 6,980 | 295 | all decided, 0 hits | 0 | 0.24, 0.10, 3.8 |
| H3 | (2, 2, 1) | 18 | 1 | decided, 0 hits | 0 | 4.5 |
| H3 | (3, 1, 1) | 18 | 4 | all decided, 0 hits | 0 | 0.03 |

No run raised `UnpinnedFamily`, no run was refused, and no state reached
the final loop at all (every state dies at a composite slice, as in the
H^6 dry pass of `docs/notes/h6_rank5_stagec_repair.md`). The 27
dependent (2, 2, 1) covers of N are the stage C tail that section 7 met
(the 234 s cover, the two-hour cover, the 1,661 s batch): they cannot be
decided on the laptop under the cap and they are the only listed covers
on which the dependent-translates raise can fire. The decision is to keep
the raise and run those 27 covers as single-cover batches without a
deadline (below); if one raises, the record names it, the aggregate does
not certify, and the follow-up is the reconstruction with the residual
split between the two blocks carried as unknowns, or a separate argument
for that cover. Everything else in stage C is decided by the matcher as
it stands.

Planted (2, 2, 1) instance with two blocks in one Pauli orbit: not
testable end to end under the cap. At n2 = 1 (three-qutrit terms, the
m = 3 geometry) all 8 planted instances (random shapes, small integer
coefficients, 0 to 5 of the 8 slices with dependent block columns) passed
45 s at a composite slice: two blocks of two copies span the whole
three-dimensional slice space, so the coordinate equations are vacuous
and the joined states are the full product of the option lists. At
n2 = 2 one instance (base (30, 224, 224, 332, 332)) passed 470 s at
composite slice (1, 2). The case is covered instead by a unit test on the
strict coordinate solve (`tests/test_qutrit_m4_rank5.py`: two blocks in
one Pauli orbit with a shared translate raise "dependent", independent
translates return the coordinates, the other two strict raises fire on
their inputs, and `batch.match_cover` records a raising run as
undecided), and the covers it affects are exactly the 27 single-cover
batches.

Lists (`driver.py degenerate --write`, 17 s and 3 s):

| cell | covers | stage B | stage C | (2, 1, 1, 1) | of which kappa >= 1 | (2, 2, 1) | of which dependent | (3, 1, 1) | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| N | 28,356 | 12,175 | 16,181 | 16,007 | 124 | 87 | 27 | 87 | `a505dcbbc00a3e32` |
| H3 | 13,136 | 6,112 | 7,024 | 6,988 | 8 | 18 | 0 | 18 | `8b41e2740a03c196` |

The aggregate's fresh enumeration equals both lists. The stage B covers
are unchanged from section 7; the addition is the T + (b, b) route,
10,271 covers for N and 2,136 for H3, all of pattern (2, 1, 1, 1) with
kappa = 0.

Rates and partition (`driver.py partition ORBIT --target-s 600
--target-bc-s 600 --match-ms M --rates JSON`). Stage A and B rates are
the pod's measured ones (54 and 21 ms per stage A cover, 4.8 and 4.3 s
per stage B cover). Stage C rates by class, from the dry pass scaled by
the pod-to-laptop ratio of stage B (0.85) and rounded up: N 0.8 s per
(2, 1, 1, 1) cover, 25 s with kappa >= 1, 7 s per (3, 1, 1), 10 s per
independent (2, 2, 1), and a nominal 3,600 s per dependent (2, 2, 1)
cover (the pod's 10.9 s stage C mean over the old list of 5,910 covers,
64,400 CPU-s, leaves about 2,000 s per dependent cover once the other
classes are accounted for at these rates); H3 0.3, 6, 0.15, 5 s. The
stage C covers are dealt longest first onto the least loaded batch, a
cover at or above the target gets its own batch, and the cheap covers of
a batch run first so that a deadline abort leaves the fewest not run.

| cell | stage A | stage B | stage C, shared batches | stage C, single-cover batches | batches | projected CPU-h |
|---|---|---|---|---|---|---|
| N | 17 (0 to 16), 3.1 h | 98 (17 to 114), 124 or 125 covers, 16.2 h | 29 (115 to 143), 554 to 562 covers, about 590 s each, 4.7 h | 27 (144 to 170), 27.0 h nominal | 171 | 51.1 (24.1 without the nominal tail) |
| H3 | 9 (0 to 8), 1.4 h | 44 (9 to 52), 138 or 139 covers, 7.3 h | 4 (53 to 56), 1,748 to 1,764 covers, about 560 s each, 0.6 h | 0 | 57 | 9.3 |

Every shared stage C batch of N holds 4 or 5 kappa >= 1 covers and 0 to
3 independent (2, 2, 1) covers; every H3 stage C batch holds 2 kappa >= 1
and 4 or 5 (2, 2, 1) covers. Partition hashes `455303990789bf9c` (N) and
`d6b847ec0efda241` (H3). Both hashes and both list hashes differ from
section 7's, so the six sample batches run on the pod against the old
partitions (N 0, 37, 165 and H3 0, 28, 85) do not verify against the
new geometry and are not reused; the corresponding first batches of each
stage are N 0, 17, 115 and H3 0, 9, 53.

Controls after the changes (this laptop): `control-planted N` and `H3`
recover all 32 planted decompositions per cell with 0 candidates
rejected (further decompositions with the same base N 0, 0, 1, 20 and
H3 0, 0, 0, 15; the repeated case draws from the new list, so the N
count differs from the review's 27); `control-product N` 270 of 270 and
`H3` 81 of 81; `control-m3 N --sample 200` as above. `control-covers`
does not touch the matcher and was not re-run. `aggregate.py N --dry-run
--partial` and the H3 counterpart pass every stored check against the
empty results directories (fresh enumeration equal to the list, geometry
tiles the list once). `batch.py H3 53 --max-seconds 20` exercised the
guard: 42 covers matched, 1 aborted at a composite slice, 1,705 not run,
exit 1.

Pod commands (16-vCPU Linux, from the repository root after `uv sync
--extra challenge`; rerunning any loop resumes, finished batches are
skipped). First the six sample batches, one process each, and the
partial aggregates, to check the rates before the rest:

```
mkdir -p research/qutrit_m4_rank5/results/N research/qutrit_m4_rank5/results/H3
printf 'N 0\nN 17\nN 115\nH3 0\nH3 9\nH3 53\n' | xargs -P 6 -L 1 sh -c \
  'nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/batch.py "$0" "$1" --max-seconds 3600 \
   > research/qutrit_m4_rank5/results/"$0"/batch_"$1".log 2>&1'
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --dry-run --partial
```

Then everything, 15 processes at a time: the shared batches under a
3,600 s guard (six times their estimate), the 27 single-cover N batches
with no guard:

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

A batch that hits the guard prints `ABORTED at the --max-seconds` and
exits 1; list them with `grep -l ABORTED
research/qutrit_m4_rank5/results/*/batch_*.log`, and re-run each with
`--force` and no guard (`batch.py N K --force`). A single-cover batch
that exits 1 with `UnpinnedFamily` in its record's `undecided` list is
the dependent-translates case above. Then the aggregates, the dry run
first and then the certificate's command, which re-runs two batches
drawn with seed 20260922 from the batches present (N batches 121 and
169 when all 171 are present, a shared stage C batch and a single-cover
one; H3 batches 39 and 56):

```
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --dry-run --partial
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py N --recheck 2 --recheck-seed 20260922
nice -n 19 uv run --extra challenge python research/qutrit_m4_rank5/aggregate.py H3 --recheck 2 --recheck-seed 20260922
```

The last two write `batch_manifest_N.json` and `batch_manifest_H3.json`
and print the claim lines; then fill the placeholders of the two draft
bound files and move them to `bounds/`. Wall time on 15 cores: H3 about
45 minutes; N about 2 hours for the 144 shared batches plus whatever the
27 single-cover batches take, which is the one open cost (two hours for
one of them on the pod in section 7).

Dependencies of the bound files to record with the run, beyond section
2: the cancel-at-base case (a base multiset T + (b, b) with b outside
span(T) and the two copies cancelling at x_0, which Fact B does not
exclude and which the stage C list now holds), and, for N, the 27
dependent (2, 2, 1) covers decided by the matcher with the strict
reconstruction rather than a separate argument.
