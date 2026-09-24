# Excluding rank 6 for |N>^4: design note

Status (2026-09-24). Design, prototypes, and measurements; nothing here is
a bound. The cell is 6 <= chi(N^4) <= 7 (`bounds/N-m4-lower-6.json`, the
rank-5 exclusion of `docs/notes/qutrit_m4_rank5_exclusion.md`;
`bounds/N-m4-upper-7.json`, the Lean witness), so excluding rank 6
settles chi(N^4) = 7. `docs/notes/next_exclusion_feasibility_2.md`
(section 5) chose the base point (2, 2) and measured the census of full
6-covers of |N>^2 at 3.2e7 to 3.8e7; `docs/notes/kernels_k6_p3.md` wrote
the compiled kernels and re-projected the run at about 750 laptop
CPU-hours, with the dependent 6-multisets of stage B6 at about 2e6 bases
times 1.24 s dominating, and left two ideas: carry the span state's
coefficient as the one unknown through a hashed first point, or reduce the
6-sets under the symmetry group of |N>^2. This note does both, prototypes
the remaining stages, and projects the run at 34 to 57 laptop CPU-hours,
70 to 110 on the pod if the compiled dense solve earns the compiled
factor and 130 to 210 if it earns the Python one (section 7). The scripts are `research/n4_rank6/`
(`degenerate6.py`, `filters6.py`, `probe6.py`, `census6_full.py`,
`invisible3_probe.py`, `orbits5.py`, and `diag/`) and their records under
`research/n4_rank6/results/`. Everything ran single-process at nice 19 on
the 18-core laptop under load from other sessions, through
`research/t5_rank5/run.py` with a 600 s cap, about 1.5 CPU-hours in all
(section 10).

Notation follows the two notes above. psi_4 = |N>^4 with |N> = (1, 1, -2)
/ sqrt 6; psi_2 = |N>^2; the 360 two-qutrit stabilizer states in the order
of `dictionary(3, 2)`; G_2 the unitary symmetry group of psi_2 (the
single-copy Clifford stabilizer of |N> on each copy and the swap of the
copies, order 72, acting on the dictionary by permutations). Slicing along
qutrits 1, 2 at x in F_3^2 gives sum_i c_i u_i^(x) = alpha_x psi_2 with
alpha_x = a_{x_1} a_{x_2}, a = (1, 1, -2), so from the base point
x_0 = (2, 2) the ratio alpha_x / alpha_{x_0} is 1/4 at the four points with
both coordinates in {0, 1} and -1/2 at the four points with one coordinate
2. The coordinate points are (0, 2) = x_0 + e_1 and (2, 0) = x_0 + e_2,
both of ratio -1/2. A full k-cover is a multiset of k dictionary states
whose span contains psi_2 with every coefficient nonzero; kappa is the
dimension of the coefficient family over the distinct states. P1 = 65521,
P2 = 2013265921; every modular step lists a superset and every candidate
is decided mod P2 and over C, as in every census of this project.

## 1. The argument

Suppose psi_4 = sum_{i=1}^6 c_i s_i with pairwise distinct stabilizer
states s_i and every c_i nonzero (a repeat or a zero coefficient is rank at
most 5, excluded by the board).

Fact B (at least four of the six terms are nonzero at every two-qutrit
point). A point sees at least chi(N^2) = 3 terms, and PR #86's case M
search (`docs/notes/constructions_2026_09.md`: no rank-5 or rank-6
decomposition of |N>^4 has a two-qutrit slice with exactly three nonzero
terms, 90 (decomposition, x_0) pairs and 945 coverage patterns for N, 0
exact completions) closes three. Fact A of the rank-5 argument (every term
full along every qutrit) does not hold at rank 6: a one-qutrit slice may
see five terms, a non-minimal rank-5 decomposition of |N>^3, whose census
is out of reach (the feasibility note, section 5.1). Nothing below uses
it.

The base point. At x_0 = (2, 2) the visible terms number k in {4, 5, 6}
and their base slices form a full k-cover of psi_2; the 6 - k invisible
terms have flats missing x_0, and every affine flat of F_3^2 missing a
point is one of 8 points or 8 lines (12 lines in all, 4 through x_0). The
matcher for a base and a choice of invisible flats enumerates every
decomposition with that base and those flats and assumes nothing else
(every flat through x_0 is allowed for every visible term). This is the
T^5 design (`docs/notes/t5_rank5_exclusion.md`, section 2) with Fact B in
place of Fact 2 and 16 flats in place of 6. No case split on the base
point and no monomial symmetry of |N> is used; (2, 2) is chosen because
no other point has ratio 1 from it, which the feasibility note measured as
a factor five to twenty over (0, 0) on every base kind.

Lemma (one base per G_2 orbit). Let U be a unitary symmetry of psi_2, so
U psi_2 = e^{i theta} psi_2 and U permutes the two-qutrit dictionary up to
phases, and let g = I_9 (x) U act on four qutrits, U on qutrits 3, 4. Then
g psi_4 = e^{i theta} psi_4, g carries stabilizer states to stabilizer
states, and slicing along qutrits 1, 2 commutes with g: (g s)^(x) = U
s^(x) for every x. So g carries a rank-6 decomposition of psi_4 with base
multiset S at x_0 (visible terms' slices, with multiplicity) and invisible
flats F to one with base multiset U S at x_0 and the same flats F, and the
coefficients go along up to the common phase. Hence the decompositions
with base S biject with those with base U S, and a matcher that finds
every decomposition with a given base need run on one base per G_2 orbit
of multisets, the flats being untouched. The group must act on the whole
decomposition through the unsliced qutrits; a symmetry acting on qutrits
1, 2 would move x_0, which is why only G_2 is used and why the
antiunitary symmetry, which the census also leaves out, is not needed.
The pivot and partner reductions of every census rest on the same lemma
(`rank_exclusion.symmetry_orbit_reps`); what is new here is applying it
to the degenerate lists, which the rank-5 run listed without orbit
reduction, by a canonical form (the least image of the sorted tuple under
the 72 permutations, `degenerate6.canonical_codes`).

## 2. The lists

All lists are built from the rank-5 census files
(`research/qutrit_m4_rank5/results/N/kernel_census.json`: 29 full
3-covers, 1,403 full 4-covers, 197,440 full 5-covers of distinct
independent states, one per G_2 orbit as far as the pivot and partner
reductions go; `degenerate_covers_N.json`: 12,175 dependent 5-covers and
16,181 repeated 5-multisets) and from the compiled 6-cover census.

### 2.1 k = 6: A6, B6, C6

A6, six distinct independent states (kappa = 0): the census of full
6-covers through `cover6_pair` over the 1,209 pivot pairs; section 4.

B6, six distinct dependent states (kappa >= 1). Completeness: a full
6-set S with kappa >= 1 contains an independent full k-cover T with k <= 5
and rank(S) <= 5 (move along the family until a coefficient vanishes, drop
the zero coefficients, repeat), so S is T_5 plus a state of span(T_5)
(kappa 1); T_4 plus two states of its span (kappa 2) or plus a pair
parallel modulo span(T_4) with both outside it (kappa 1); or T_3 plus
three states of its span (kappa 3), a span state and a parallel pair
(kappa 2), a triple all parallel modulo the span (kappa 2), or a triple
whose images modulo span(T_3) span a plane with no two parallel (kappa 1).
An added state outside the span parallel to nothing is dead, and the
6-set is not full. `degenerate6.py` enumerates these routes over F_P1,
decides every candidate numerically (residual, rank, no dead
coefficient), deduplicates, and canonicalizes
(`results/degenerate6_N.json`):

| route | (T, R) pairs | distinct full 6-sets | G_2 orbits |
|---|---|---|---|
| T_5 + span state | 3,222,224 (0 to 35 span states per 5-cover, 46,238 covers with 35) | 1,608,435 | 251,028 |
| T_4 + two span states, T_4 + parallel pair | 58,932; 847,501 | | |
| T_3 + three span states, + span state and parallel pair, + parallel triple, + collinear triple | 72; 23,220; 12,410; 1,715,580 | | |
| all routes | 5,879,939 | 2,701,615 (of 3,611,918 candidates; kappa 1: 2,672,788, kappa 2: 28,820, kappa 3: 7) | 259,655 (kappa 1: 256,970, kappa 2: 2,683, kappa 3: 2) |

The feasibility note's "about 2e6 6-sets" was the (T_5, x) count with a
mean of 12 span states; the mean is 16.3 and the distinct sets 1.6e6 by
that route and 2.7e6 by all routes. The orbit reduction is a factor 10.4:
a kappa = 1 set has up to six 5-subsets that are full 5-covers, each with
one or more census representatives in its orbit, so its orbit appears
once per such subset. The dependency vector K of a kappa = 1 orbit
representative has support 3, 4, 5, or 6 for 82,706, 72,520, 58,115, and
43,629 of the 256,970 representatives (`diag/suppk.py`); the support
size sets the filter's cost in section 3.

C6, a repeated state. Let C be the distinct states whose merged
coefficient (the sum over copies) is nonzero at x_0 and Z the states
whose copies cancel there. C is a full |C|-cover of distinct states,
independent or dependent, |C| in {3, 4, 5}, so the multiset is C with
multiplicities plus cancelling blocks of at least two copies of a state
outside C, six in all: a 5-cover with a member doubled (the 197,440
independent and 12,175 dependent 5-covers); a 4-cover (1,403 independent,
31 dependent) with a member tripled, two members doubled, or plus (b, b)
for any of the 356 other states; a 3-cover with multiplicities (4, 1, 1),
(3, 2, 1), (2, 2, 2), or a member doubled plus (b, b), or plus (b, b, b).
No fullness test applies beyond C being full (the blocks are exempt, as in
the H^6 v2 list and the rank-5 lists), and the same multiset arises from
several routes when b lies in span(C). Deduplicated and canonicalized:

| pattern | distinct multisets | G_2 orbits | of which the distinct states are dependent (orbits) |
|---|---|---|---|
| (2, 1, 1, 1, 1) | 1,546,463 | 315,229 | 7,407 |
| (3, 1, 1, 1) | 16,007 | 2,277 | 24 |
| (2, 2, 1, 1) | 39,417 | 3,981 | 36 |
| (4, 1, 1), (3, 2, 1), (2, 2, 2) | 87, 174, 29 | 20, 38, 8 | 0 |
| all | 1,602,177 | 321,553 | 7,467 |

### 2.2 k = 5 and k = 4

Stage (beta'), k = 5: the bases are the 225,796 full 5-multisets of the
rank-5 run (197,440 + 12,175 + 16,181, which include the cancel-at-base
multisets T_3 + (b, b)), each with one invisible term on one of 16 flats.
Stage (gamma), k = 4: the 1,403 full 4-covers, the 31 dependent 4-covers
(T_3 plus a span state, full), and the 87 repeated 4-multisets (T_3 plus a
member); there is no cancel-at-base 4-multiset, since chi(N^2) = 3 leaves
no 2-cover. Each with two invisible terms on one of the 136 flat
multisets. The G_2 orbit counts of these lists are in section 6.

## 3. Stage B6: the decision that closes

### 3.1 The fresh-term filter (option (i), made exact)

The feasibility note's idea was to carry the span state's coefficient as
the one unknown through the first coordinate point. Made precise it is a
change of basis, not a hash. Let S be a kappa = 1 base of six distinct
states with family d(lambda) = d_0 + lambda K, and pick a term j with
K_j != 0 (the fresh term); the other five are independent, whatever
route S came from. In the basis B of the nine Pauli translates Q_k s_j of
the fresh term's base slice (a basis of C^9), write

    U = B^-1 (sum_{i != j} d_{0i} w_i - rhs),    V = B^-1 sum_{i != j} K_i w_i,

where w_i is term i's option at the coordinate slice (27 phased
translates or absent, 28 options) and rhs = -psi_2 / 2. The fresh term's
slice there is (d_{0j} + lambda K_j) w^l Q_k s_j or zero, so U + lambda V
is supported on at most one coordinate k. Partition the nine coordinates
into parts P_1, P_2, P_3 of three; k lies in one part, so on the
complementary six coordinates G the vector U_G lies in the span of V_G.
Each of the three conditions is a 1-parameter dense solve over the five
ordinary terms (sides 28^3 x 28^2, 1.7e7 feature pairs, against 4.8e8 for
the six-term solve of `kernels_k6_p3.md`), run through `dense_solve` on
the projected option tables mod P1 and P2, and the union over the three
parts is a superset of the exact first-slice solutions. Every surviving
combination is completed with the fresh term's code (the class k and the
phase from the residual coordinate, or absent, from the whole 9-vector
mod P1) and decided by `Family.restrict`, the reference's own decision,
so the filter's solution list is the reference's first-slice list, not a
superset. The fresh term and the partition are chosen per base to
minimize the structural slack (options whose support in the translate
basis lies inside one part, which that part's group cannot see); a
two-part split (4 + 5) leaves tens of thousands of structural survivors
per base because the projected options of two-qutrit stabilizer states
have support 3 or 9 in a translate basis, and three parts leave slack 0
on 59 of 60 sampled bases. The code is `filters6.Filters.b6_slice`, the
lemma is the paragraph above, and the survivors' second coordinate slice
is the same filter at e_2; a base surviving both goes to the reference
`Matcher.run`, which recomputes everything.

Why the "hash against the dictionary" reading does not work: the
condition U + lambda V in span(e_k) is a parallelism, not a sum, and the
projection that turns it into a lookup depends on V, which depends on the
options; with two random functionals it is a 2 x 2 determinant, bilinear
across the two sides, which is what the dense product computes.

Controls (`probe6.py planted`, `results/probe6_planted.json`): six
planted six-term decompositions of random targets over kappa = 1 orbit
representatives with random flats and integer coefficients; the filter's
first-slice list equals the reference `solve_slice3` list and contains the
planted codes in 6 of 6; and on real bases the two lists agree wherever
compared (6 of 6 in the rate runs).

Rates (`probe6.py rate`, `results/probe6_rate_x22.json`, 60 orbit
representatives evenly spaced; the reference is `solve_slice3` on the same
slice; "survivors" are the combinations the three dense products return
after the kernel's own whole-vector checks):

| quantity | value |
|---|---|
| filter per base, mean / median / max | 0.271 s / 0.234 s / 0.556 s (dense products 0.204 s) |
| reference first slice per base | 1.387 s (the whole `Matcher.run` was 1.24 s in `kernels_k6_p3.md`, measured at a lower load) |
| raw feature zeros per base by support of K (3, 4, 5, 6) | 2.0e6, 2.1e5, 4.3e4, 1.0e4; dense seconds 0.39, 0.18, 0.16, 0.15 |
| survivors after the kernel's checks, mean / max | 36.6 / 382; after the P1 whole-vector test 1.0 |
| bases with first-slice solutions | 9 of 60 (1 to 25 solutions) |

The raw feature zeros are structural: when the dependency has support 3
(x = a t_1 + b t_2, a third of the representatives) the vector V involves
two ordinary terms, and the combinations where those two translates
cancel (the same Pauli on the sub-dependency, about 28 of 22k) make V_G
vanish for every choice of the other three terms, 28 x 22k feature zeros
per part, each decided by the kernel's whole-vector check at about 0.1
microseconds. The floor without them is 0.15 s per base; a kernel option
that skips combinations with V_G = 0 unless U_G = 0 (those are the
lambda-free solutions, decided separately by a meet in the middle) would
bring the support-3 class to the floor and save about a third of the
stage. Not done here.

### 3.2 Orbit reduction (option (ii))

Section 1's lemma and section 2.1's counts: 2,701,615 full dependent
6-sets become 259,655 orbit representatives. The aggregate must rebuild
the full list by the routes, canonicalize, and compare the representative
list's hash, as it re-enumerates the degenerate lists today.

### 3.3 Both, and the projection of the stage

kappa = 1 (256,970 representatives): 0.27 s per base for the first slice;
15 percent survive to the second slice (another 0.27 s), and the fraction
surviving both is not measured (0 of 25 bases in the variant run
`diag/variants_b6.py` had first-slice solutions at all); those go to
`Matcher.run` at 1.4 s. Projected: 256,970 x 0.27 s + 0.15 x 256,970 x
0.27 s + (at most 0.15 x 256,970 x 1.4 s) = 19.3 + 2.9 + at most 15
laptop CPU-hours, so 22 to 37, with the survivor share the only
uncertainty.

kappa = 2 (2,683 representatives): the reference's 2-parameter dense
solve refused 3 of 4 at the default candidate cap of 2,000,000 (structural
feature zeros of the 3 x 3 determinant, the same effect as above); with
the cap at 2e8 all 6 sampled bases decided, 2.8 to 11.6 s, mean 6.9 s, 4
of 6 with coordinate solutions (up to 602) and 0 hits after the composite
stage (`probe6.py kappa2 --max-cand 200000000`, `results/probe6_kappa2.json`).
Projected 2,683 x 6.9 s = 5.1 laptop CPU-hours. The batch must run these
with the raised cap and record `dense_raw`.

kappa = 3 (2 representatives, T_3 plus three states of its span): the
compiled kernel takes kappa <= 2; the Python `_dense` with 70 Laplace
features over 28^3 x 28^3 pairs is about 3e10 flops per slice, minutes
per base, and the candidate cap must be raised as above. Two bases; run
them as single-base batches. Not measured.

## 4. Stage A6 and the census

The census (`census6_full.py census`, `results/census6_N_full.json`,
resumable, 413 kernel seconds in all): `cover6_pair` over all 1,209 pivot
pairs of `CoverEnumerator3("N", 2)` lists 37,201,212 full 6-covers, every
one of rank 6 mod P2 (the kernel returned no dependent 6-set on any pair),
from 702,716,072 modular candidates; the heaviest pair (117, 8) has
M = 348 members above the partner and 5,286,661 covers in 7.1 s, two pairs
exceed a million covers, and the time is 4.7e-8 s x M^3 (sum of M^3 over
the pairs 8.79e9). The feasibility note's stratified samples projected
3.18e7 and 3.76e7; the exact count is 3.72e7, 188 times the rank-5 census.
Per pivot: 117 carries 10,686,095, 281 6,957,470, 288 3,587,042, 285
3,118,532, 279 3,063,219, 310 2,803,910, 297 2,606,843, 306 1,987,350, 313
1,132,513, 325 802,638, and the seven remaining pivots 455,600 together.
The kernel's 6-sets are all distinct independent bases by construction (a
6-set with a 5-subset covering psi_2 has a member with a zero residue and
is skipped), and the record's per-pair independent count confirms it on
every pair; dependent 6-sets are stage B6's list.

The matcher (`census6_full.py rate`, `results/rate_a6_x22.json`): 20,000
covers from 12 random pivot pairs through `Matcher.run` at (2, 2), warm
option cache, 0.68 to 0.70 ms per cover including the Python wrapper (the
kernel alone was 0.10 ms per 5-cover in `kernels_k6_p3.md`), every run
dying at the first coordinate slice (`coord_raw` 0 in 20,000 of 20,000),
no hit, no refusal, and the reference matcher agreeing on hits, counts,
and refusals for the first 50 (56 ms each). Projected: 3.72e7 x 0.68 ms =
7.0 laptop CPU-hours through `Matcher.run`, 1 to 2 with a batch loop that
calls the kernel directly and confirms only its hits. The census 5-covers
are 197,440 multisets in 50,719 G_2 orbits (`orbits5.py`), so the 6-cover
census is also about four times its orbit count; canonicalizing 3.7e7
sets is minutes of numpy and would cut A6 to about 2 hours through
`Matcher.run`, an option rather than a need.

## 5. Stage C6

The reference matcher's block path is cheap at (2, 2) and the filter is
not needed except where the family has a parameter; measured on orbit
representatives (`results/probe6_rate_x22.json`, `probe6_c6k1.json`,
`probe6_two_blocks.json`):

| class | orbits | matcher | per base: mean / median / max | first-slice solutions | projected laptop CPU-hours |
|---|---|---|---|---|---|
| (2, 1, 1, 1, 1), kappa 0 | 307,822 | `Matcher.run` (block path, meet in the middle per translate set) | 0.008 / 0.008 / 0.009 s | 0 in 60 | 0.7 |
| (2, 1, 1, 1, 1), kappa 1 | 7,407 | `Matcher.run` (1-parameter dense per translate set) | 1.05 / 0.4 / 6.1 s (8 bases) | 5 to 511 in 6 of 8, all dying at the composite stage | 2.2 |
| (3, 1, 1, 1) | 2,277 | `Matcher.run` | 0.18 / 0.014 / 10.0 s | 245 in 1 of 60 | 0.1 |
| (2, 2, 1, 1) and the rest | 3,981 + 66 | `Matcher.run` (two blocks; `BlockOnlyMatcher` for (2, 2, 2)) | 0.58 / 0.3 / 1.6 s (5 bases) | 6 and 1 in 2 of 5 | 0.7 |

The kappa = 1 block filter of `filters6.c6_slice` (the residual in the
block's translate basis has at most two nonzero coordinates, a dense
solve per coordinate group) lists the reference's solutions exactly (6
of 6 planted, 2 of 2 real) but is fifteen times slower than the
reference's own first slice on this shape (2.0 s against 0.14 s, with
14,000 structural survivors per base); it is not used. The (2, 1, 1, 1)
kappa >= 1 covers that cost 27 s each in the rank-5 run were the
five-term shape at (0, 0); here the four ordinary terms and the ratio
-1/2 leave the same class at about a second.

## 6. Stages (beta') and (gamma) at p = 3

Geometry (`invisible3_probe.py geometry`, `results/invisible3_geometry.json`).
The 16 flats missing (2, 2): the 8 points and the 8 lines
{x_1 = 0}, {x_1 = 1}, {x_2 = 0}, {x_2 = 1}, {(0, 1), (1, 2), (2, 0)},
{(0, 2), (1, 0), (2, 1)}, {(0, 0), (1, 2), (2, 1)}, and
{(0, 2), (1, 1), (2, 0)}. By the coordinate points they contain: 9 miss
both (6 points and the lines {x_1 = 1}, {x_2 = 1}, {(0, 0), (1, 2),
(2, 1)}), 6 contain one (the points (0, 2) and (2, 0) and the lines
{x_1 = 0}, {x_2 = 0}, {(0, 1), (1, 2), (2, 0)}, {(0, 2), (1, 0), (2, 1)}),
and one contains both ({(0, 2), (1, 1), (2, 0)}). Stage (gamma) has 136
flat multisets, 45 of them pairs of flats missing both coordinate
points. Every flat leaves at least five exact points (points of F_3^2 on
no invisible flat other than x_0), and the slice ratio at every point
other than x_0 is 1/4 or -1/2.

The visible terms the base carries: Fact B gives k >= 4 at every point, so
k in {4, 5, 6}; k = 6 is section 2.1, k = 5 has one invisible term, k = 4
two. Fact B also bounds the invisible terms at every other point by two,
but the matcher does not use that: it allows every option, absent
included, for every visible term at every point, as the T^5 matcher does.

Exact points (`invisible3_probe.py points`, `results/invisible3_points.json`,
60 full 5-covers and 60 full 4-covers evenly spaced in the census lists,
the slice equation at each of the eight other points with every option
allowed, the point family, meet in the middle mod P1 and exact decision):
no solution at any of the eight points for any of the 120 bases; 1.9 to
2.1 ms per solve for five terms, 0.1 to 0.2 ms for four. So a (base,
flat) run of either stage dies at its first exact point in every sampled
case, whatever the flat, and the stages cost one meet in the middle per
run.

The (beta') prototype for the 9 flats missing both coordinate points
(`invisible3_probe.py beta`, `Proto.run_beta`): both coordinate points
exact, the join (the family is a point), the composite exact points over
the shapes the structure lemma leaves alive (`TermOpts.composite_rows`,
the matcher's own tables), the birth scan at the flat's first point (the
residual after the visible terms normalized and looked up in the table of
the 360 normalized dictionary states mod P1, every match decided over C
and mod P2; a zero residual is another flat's configuration and is
skipped), the line's other two points with the fresh term as an ordinary
term with its coefficient pinned and 27 phased translates, assembly, and
`Matcher.confirm`. The dictionary the birth condition scans is the
two-qutrit one, 360 states; the 1,080 of the task's question is the
three-qubit count. Planted (`results/invisible3_beta.json`): two random
targets per flat, five visible terms with random flats through x_0 over a
census 5-cover and one invisible term with a random dictionary state at
the flat's first point (a point term, or a line term along the flat with
random class and quadratic), recovered in 18 of 18 by the prototype, 0.01
to 0.05 s per run. Real bases: 40 census 5-covers at each of the 9 flats,
every run dying at the first coordinate point, 2.1 ms per run (the first
flat's 71 ms is the cold option cache), 0 hits.

The 7 flats containing a coordinate point need the order of points of the
T^5 matcher (exact points first, the codes at later points restricted to
those compatible with the codes fixed earlier, `compatible_codes` of
`research/t5_rank5/invisible.py`, ported to the p = 3 shapes: a term
present at two points fixes its class and phase at the third up to the
quadratic, a term present at one point is a line through x_0 and absent
at the third, a term absent at both is free), since only one coordinate
point is exact and the composite exact points come before the second
coordinate point. Not written. Its first exact point is the other
coordinate point or a composite point, where the sample above has no
solutions.

Stage (gamma) (`invisible3_probe.py gamma`): the same exact-point pass on
4-covers over the 45 flat pairs missing both coordinate points, then the
birth scan (different first points) or the rank-2 residual scan (a shared
first point: the residual r must be c_4 v_4 + c_5 v_5 with v_4 != v_5,
found by the projective hash of the 360 states modulo r, every collision
decided over C and mod P2; the same-state cases r = D v and r = 0 of two
line flats sharing a point are counted and left to the design, as at T^5).
On 10 census 4-covers no flat pair reaches a scan: the two coordinate
points have no solution for any of them (`results/invisible3_gamma.json`),
and all 45 pairs of a base are decided in under 10 ms together. The scans
themselves ran only inside the planted (beta') runs (the birth scan) and
are otherwise untested; the rank-2 scan is exercised by no instance here.

Costs. Independent 5-covers: 197,440 x 16 flats x about 2.5 ms (the exact
solve plus the option-table lookups) is 2.2 laptop CPU-hours; the 12,175
dependent 5-covers need the 1-parameter dense solve at their first exact
point (five terms, 4.8e8 pairs through the six-term kernel or the
fresh-term filter's 5.1e7, 0.05 to 0.6 s) for 2.7 to 32 CPU-hours over 16
flats before orbit reduction, and the 16,181 repeated 5-multisets the
block path at about 0.01 s, 0.7 CPU-hours; stage (gamma) is 1,521 x 136 x
0.2 ms, a minute. The k = 5 lists reduce under G_2 (`orbits5.py`,
`results/orbits5_N.json`): the 197,440 census 5-covers to 50,719 orbits,
the 12,175 dependent to 1,452, the 16,181 repeated to 2,317, 54,488 in
all; the k = 4 lists from 1,521 to 505 (478 + 7 + 20). With the reduction
stage (beta') is 50,719 x 16 x 2.1 ms + 1,452 x 16 x (0.05 to 0.6 s) +
2,317 x 16 x 0.01 s, 1 to 4 laptop CPU-hours, and stage (gamma) 505 x 136
x 0.2 ms, a quarter of a minute. The dependent 5-cover bases are the one
place where the six-term dense solve would run per (base, flat); the
fresh-term filter of section 3.1 applies to them unchanged (five distinct
states with kappa = 1 and one fresh term), which is the 0.05 s figure.

## 7. Projection

Laptop CPU-hours at the rates above; pod factors as measured on the T^5
run: 1.3 for the compiled stages, 4 for the Python stages. The B6 filter
is three compiled dense products plus Python bookkeeping (0.20 of 0.27 s
in the kernel), so it is shown at both factors.

| stage | items | rate | laptop CPU-hours | pod, factor 1.3 | pod, factor 4 |
|---|---|---|---|---|---|
| census of full 6-covers (`cover6_pair`) | 1,209 pivot pairs | 4.7e-8 s x M^3 | 0.11 | 0.15 | |
| A6 (`SliceMatch3Kernel`) | 37,201,212 covers | 0.68 ms through `Matcher.run`, about 0.15 ms kernel | 2 to 7 | 3 to 9 | |
| B6, kappa 1 (filter, then the reference on survivors) | 256,970 | 0.27 s + survivors | 22 to 37 | 29 to 48 | 88 to 148 |
| B6, kappa 2 (reference, cap 2e8) | 2,683 | 6.9 s | 5.1 | | 20 |
| B6, kappa 3 (Python dense) | 2 | minutes | 0.1 | | 0.4 |
| C6 (2, 1, 1, 1, 1) kappa 0 | 307,822 | 0.008 s | 0.7 | | 2.8 |
| C6 (2, 1, 1, 1, 1) kappa 1 | 7,407 | 1.05 s | 2.2 | | 8.6 |
| C6 (3, 1, 1, 1), (2, 2, 1, 1), rest | 6,324 | 0.18 to 0.58 s | 0.8 | | 3.2 |
| (beta'), 16 flats, orbit-reduced | 54,488 bases | 2.1 ms to 0.6 s per (base, flat) | 1 to 4 | | 4 to 16 |
| (gamma), 136 flat multisets, orbit-reduced | 505 bases | 0.2 ms | 0.004 | | 0.02 |
| total | | | 34 to 57 | | |

Pod total: the compiled stages (census and A6) are 3 to 9 pod CPU-hours;
the Python-led stages (B6 kappa 2 and 3, C6, (beta'), (gamma)) about 10
to 13 laptop CPU-hours, 40 to 50 on the pod at factor 4; B6 kappa 1 adds
29 to 48 if the compiled dense solve earns the compiled factor and 88 to
148 if it does not. So 70 to 110 pod CPU-hours in the favorable case and
130 to 210 in the unfavorable one, five to fourteen hours of wall time on
15 processes. The first B6 batch on the pod settles the factor before the
partition is trusted, as the T^5 note's section 8 does for its stage B.
The pod factor of 4 was measured on the Python `_dense` of the T^5 run;
`dense_solve` is compiled and its per-pair rate on the pod is not yet
measured, so the favorable case is the expected one and the unfavorable
one is the bound.

## 8. Soundness checklist

In the shape of the T^5 note's section 7; status for this design.

1. The base-point reduction: every rank-6 decomposition has a full 4-, 5-,
   or 6-multiset base at (2, 2) with the invisible terms on flats missing
   (2, 2). From Fact B (PR #86's tables) and the flat enumeration; the 16
   flats and 136 flat multisets are exactly the cases the stages carry.
2. The k = 6 lists are complete: A6 from the census (the H^6 census
   argument, unchanged by the orbit or by k = 6); B6 from the kappa
   reduction of section 2.1 with every route enumerated over F_P1 and
   decided numerically (a modular superset); C6 from the merged-coefficient
   argument, no fullness test beyond C being full, cancelling blocks of
   any state outside C. The aggregate re-enumerates all three and compares
   hashes.
3. Orbit representatives: the canonical form is computed from the full
   lists under the closure of the census's generators (order 72 asserted);
   the lemma of section 1 is stated in the bound file's notes; the
   aggregate recomputes the representatives.
4. The B6 filter is a necessary condition on the first coordinate slice,
   its survivors are completed and decided by the reference's `restrict`,
   and a base surviving both coordinate slices runs through the unchanged
   reference matcher; the planted control and the real-base agreement
   checks are re-run at the commit that carries the final `filters6.py`.
5. The candidate cap of the dense solve is raised for kappa >= 2 and the
   raw count recorded; a base refused at the cap fails the aggregate.
6. Stage (beta') and (gamma) matchers assume nothing about the visible
   terms' flats beyond the base point, allow the absent option everywhere,
   and check the presence pattern at assembly; the birth and rank-2 scans
   are mod P1 supersets decided over C and mod P2 with agreement required;
   the same-state cases of two line flats sharing a point are enumerated,
   not dropped.
7. Repeated states, cancelling copies, dependent translates, and unpinned
   families take the rank-5 matcher's paths (`reconstruct_block`,
   `UnpinnedFamily`, `BlockOnlyMatcher`); a raise is recorded as
   undecided and fails the aggregate.
8. Every hit is re-decided from its phase codes mod P2 and numerically;
   distinct terms and rank 6 are required (`genuine`).
9. Deterministic hashes exclude the kernel-run, candidate, and raw-zero
   counts; one A6 batch and one B6 batch are replayed with
   `STABRANK_NO_NATIVE=1` (the B6 replay through the Python `_dense` is
   about 20 times slower and must be a small batch).
10. `--max-seconds` records runs not made as undecided; `--resume` redoes
    them.
11. Controls: planted instances for every stage (B6 kappa 1 and 2, C6 by
    pattern, (beta') on every one of the 16 flats, (gamma) on the 136 flat
    multisets including same-state and cancelling kinds), the rank-7 Lean
    witness of |N>^4 at its all-visible bases at (2, 2) (its bases have
    kappa up to 3 and blocks; `control-witness` of the rank-5 pipeline
    passed one base and refused the kappa 3 ones, so this control is
    recorded rather than required, as at T^5), and the m = 3 control
    (rank-4 decompositions of |N>^3 at 2 + 1) through the same code.

## 9. What remains to build before a launch

1. The (beta') matcher for the 7 flats containing a coordinate point (the
   compatible-codes order of points), and the (gamma) continuation along
   two flats with the same-state and cancelling cases; the 9-flat prototype
   here covers the rest.
2. The batch pipeline in the shape of `research/t5_rank5/`: plan and
   partition over the census pairs (A6), the B6 representatives by kappa,
   the C6 representatives by class, the (beta') bases by flat, and the
   (gamma) bases; batch records with the filter's statistics; the
   aggregate with the fresh enumeration and canonicalization of every
   list.
3. The kappa = 3 bases (2) through the Python dense solve with the cap
   raised, or a direct argument.
4. Optional: the V_G = 0 skip in `dense_solve` (a third of B6).
5. The controls of item 11 above, and the first batch of every stage on
   the pod for the rates.

## 10. What was run

All through `research/t5_rank5/run.py` (nice 19, own session, 600 s cap),
one process at a time, about 1.5 CPU-hours in all on the loaded laptop.
Records under `research/n4_rank6/results/`; the orbit representatives
(`degenerate6.py --reps FILE`, a 1.4 MB npz) are not committed and are
regenerated by the script in 80 s.

| step | command | time | result |
|---|---|---|---|
| degenerate lists and orbits | `degenerate6.py --reps ...` | 81 s | section 2.1 (`degenerate6_N.json`) |
| planted filter control | `probe6.py planted --count 6` | 38 s | 6 of 6 and 6 of 6 (`probe6_planted.json`) |
| filter and reference rates | `probe6.py rate --count 60 --reference 2 --budget 100` | 143 s | sections 3 and 5 (`probe6_rate_x22.json`) |
| filter variants and supports | `diag/variants_b6.py`, `diag/suppk.py`, `diag/diag_b6.py` | 75 s, 14 s, 20 s | section 3.1 |
| kappa 2 | `probe6.py kappa2 --count 6 --max-cand 200000000` | 42 s | 6 of 6 decided (`probe6_kappa2.json`); at the default cap 3 of 4 refused |
| two blocks | `probe6.py two-blocks --count 5` | 8 s | `probe6_two_blocks.json` |
| C6 kappa 1 | `probe6.py c6k1 --count 8` | 13 s | `probe6_c6k1.json` |
| flats | `invisible3_probe.py geometry` | 0 s | `invisible3_geometry.json` |
| exact points | `invisible3_probe.py points --count 60` | 8 s | `invisible3_points.json` |
| census | `census6_full.py census --budget 540` (a first pass of 470 s was lost to a flag-masking bug and rerun) | 416 s | 1,209 pairs, 37,201,212 covers (`census6_N_full.json`) |
| A6 rate | `census6_full.py rate --pairs 12 --count 20000 --reference 50` | 19 s | 0.68 ms per cover, 50 of 50 equal (`rate_a6_x22.json`) |
| (beta') prototype | `invisible3_probe.py beta --count 40 --plant 2` | 8 s | 18 of 18 planted recovered; every real run dies at the first coordinate point (`invisible3_beta.json`) |
| (gamma) probe | `invisible3_probe.py gamma --count 10` | 2 s | no flat pair reaches a scan (`invisible3_gamma.json`) |
| k = 5 and k = 4 orbits | `orbits5.py` | 3 s | 54,488 and 505 orbits (`orbits5_N.json`) |
