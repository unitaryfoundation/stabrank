# Excluding rank 5 for |N>^4 and |H3>^4: design and costing note

Status (2026-09-22). Design and costing only; nothing here is a bound. The
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
