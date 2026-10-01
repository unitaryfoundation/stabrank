# Excluding rank 6 for |H3>^4: design note

Status (2026-10-01). Design and controls only; nothing here is a bound.
The cell is 6 <= chi(H3^4) <= 8 (`bounds/H3-m4-lower-6.json`, the rank-5
exclusion of `docs/notes/qutrit_m4_rank5_exclusion.md`;
`bounds/H3-m4-upper-8.json`, the Lean witness, the four terms of the
rank-4 decomposition of |H3>^3 tensored with the rank-2 decomposition of
|H3>), so excluding rank 6 moves it to 7 <= chi(H3^4) <= 8 and settles
nothing exactly. `docs/notes/next_exclusion_feasibility_2.md` (section 5)
costed the N^4 and H3^4 rank-6 exclusions together and named three pieces
of engineering; `docs/notes/n4_rank6_design.md` and
`docs/notes/n4_rank6_exclusion.md` built and ran the N^4 pipeline
(chi(N^4) = 7, 2026-09-26), which left the H3 exclusion with one missing
component: the invisible-flat matcher at p = 3 for the H3 cell, whose base
point, slice ratios, flats and symmetry group differ from N's. This note
is the stage plan for H3^4 in the shape of the N^4 design note, the case
split for the invisible terms with a completeness sketch per class, the
matcher (`research/h3_m4_rank6/invisible_p3.py`) with its planted
controls, and a measured cost table. The scripts are
`research/h3_m4_rank6/` (`common.py`, `invisible_p3.py`, `probe.py`) and
their records under `research/h3_m4_rank6/results/`; everything ran
single-process at nice 19 on the 18-core laptop at a load average of 10
to 12 from other sessions (section 9).

Notation follows the N^4 notes. psi_4 = |H3>^4 with |H3> the eigenvector
of the qutrit Fourier transform with eigenvalue 1, amplitudes proportional
to (1 + sqrt 3, 1, 1), all nonzero; psi_2 = |H3>^2; the 360 two-qutrit
stabilizer states in the order of `dictionary(3, 2)`; G_2 the unitary
symmetry group of psi_2 (the 4-element Clifford stabilizer of |H3> on each
copy and the swap of the copies, order 32, acting on the dictionary by
permutations; N's is 72). w = exp(2 pi i / 3). Slicing along qutrits 1, 2
at x in F_3^2 gives sum_i c_i u_i^(x) = alpha_x psi_2 with alpha_x =
a_{x_1} a_{x_2}, a = (1 + sqrt 3, 1, 1). Each term is nonzero exactly on
an affine flat of F_3^2 (a point, one of the 12 lines, or the plane) and
its slices on the flat are phased Pauli translates of one another by the
two-qutrit slice structure lemma (PR #86). The arithmetic is
Q(w, sqrt 3) = Q(zeta_12) modulo P1 = 65521 and P2 = 2013265921 (both 1
mod 12, so w and sqrt 3 exist in both fields; `cover_census.Field3`); the
dictionary states have entries in Z[w] and only the target needs sqrt 3.
Every modular step lists a superset and every candidate is decided mod P2
and over C, as in every census and matcher of this project. A full
k-multiset of psi_2 is a multiset of k dictionary states whose span
contains psi_2 with every coefficient nonzero (a repeated state's copies
may cancel, their merged coefficient being exempt); kappa is the dimension
of the coefficient family over the distinct states.

## 1. The argument

Suppose psi_4 = sum_{i=1}^6 c_i s_i with pairwise distinct stabilizer
states s_i and every c_i nonzero (a repeat or a zero coefficient is rank at
most 5, excluded by the board).

Fact B (at least four of the six terms are nonzero at every two-qutrit
point). A point sees at least chi(H3^2) = 3 terms
(`bounds/H3-m2-lower-3.json`), and PR #86's case M search
(`docs/notes/constructions_2026_09.md`, "Case M": no rank-5 or rank-6
decomposition of |N>^4 or |H3>^4 has a two-qutrit slice with exactly three
nonzero terms; for H3, 27 (decomposition, x_0) pairs over the 9 rank-3
decompositions of |H3>^2, 25,515 coverage patterns, 51 live, 315
candidates, 66 rank-2 survivors, 0 exact completions) closes three. This
is the only structural input, as for N.

Fact A of the rank-5 argument (every term full along every qutrit) does
not hold at rank 6: a one-qutrit slice sees at least chi(H3^3) = 4 terms
(`bounds/H3-m3-lower-4.json`), exactly four is the unique rank-4
decomposition of |H3>^3 plus two point terms, which `relaxed_lift.py` case
[A] at R = 6 excludes (the same note, sharpest negative fact 3: no rank-5
or rank-6 decomposition of |H3>^4 has a minimal one-qutrit slice), but a
slice with five terms is a non-minimal rank-5 decomposition of |H3>^3,
whose census (full 5-covers of |H3>^3 over 30,240 states) is out of reach.
So the design drops the flat lemma and the base-point case split of the
rank-5 exclusion and works at one base point with the invisible terms'
flats as parameters, exactly as the N^4 design does.

The base point. At X0 = (0, 0) the visible terms number k in {4, 5, 6} by
Fact B, their base slices form a full k-multiset of psi_2, and the 6 - k
invisible terms have flats missing X0: one of the 8 other points or one of
the 8 lines not through X0 (12 lines in all, 4 through X0). The slice
ratio alpha_x / alpha_{X0} from (0, 0) is 1 / (1 + sqrt 3) = 0.366 at the
four points with one zero coordinate and 1 / (1 + sqrt 3)^2 = 0.134 at the
four points with none; it is never 1. The rank-5 run used x_0 = (1, 1),
forced by its case split (every diagonal line meets the orbit of (1, 1)
under the 1 <-> 2 swap), and from (1, 1) three points have ratio 1, where
the trivial translate always solves the exact slice equation and keeps
states alive; the feasibility note measured the analogous choice for N
((2, 2) against (0, 0)) as a factor five to twenty on every base kind.
(0, 0) is to H3 what (2, 2) is to N: the point whose coordinate value
carries the distinguished amplitude. No case split on the base point and
no monomial symmetry of |H3> is used; the matcher allows every flat
through X0 for every visible term.

Lemma (one base per G_2 orbit). The lemma of the N^4 notes, verbatim with
H3 for N: a unitary symmetry U of psi_2 acting as I_9 (x) U on qutrits 3,
4 fixes psi_4 up to a phase, permutes the dictionary, commutes with
slicing along qutrits 1, 2, and carries a rank-6 decomposition with base
multiset S at X0 and invisible flats F to one with base U S and the same
flats, the coefficients going along. So a matcher that finds every
decomposition with a given base need run on one base per G_2 orbit of
multisets, the flats untouched. The group acts through the unsliced
qutrits only; the antiunitary symmetry, which the census also leaves out,
is not used. The canonical form is the least image of the sorted tuple
under the 32 permutations (`probe.canonical_codes`, the N^4
`degenerate6.canonical_codes` with the H3 group).

Hence every rank-6 decomposition of psi_4 is found by a matcher that,
given a full k-multiset at X0 and the flats of the 6 - k invisible terms,
enumerates every decomposition with that base and those flats, run over
every full k-multiset up to G_2 and every choice of flats. The stages are
those of the N^4 design:

- Stage A6, k = 6 distinct independent states: the census of full
  6-covers of psi_2 through the compiled `cover6_pair`, matched by the
  compiled `SliceMatch3Kernel` through `matcher.Matcher.run` at X0.
- Stage B6, k = 6 distinct dependent states: the fresh-term filter of
  `research/n4_rank6/filters6.py` for kappa = 1, then the reference
  `Matcher.run`; kappa 2 and 3 through the reference at the raised cap.
- Stage C6, a repeated state: the reference `Matcher.run` (block paths;
  `BlockOnlyMatcher` for a base with no ordinary term).
- Stage (beta'), k = 5: `invisible_p3.InvisibleMatcherP3.run_one(cover,
  F)` for every flat F of the 16.
- Stage (gamma), k = 4: `run_two(cover, (F_4, F_5))` for every multiset
  of two flats, 136 in all.

Stages A6, B6, C6 reuse the qutrit kernels and matchers unchanged: the
`Matcher`, `SliceMatch3Kernel`, `dense_solve`, `cover6_pair`, and the
B6 filter's lemma are generic in the orbit and the base point (they take
the enumerator's fields, the target, and x0 as inputs; the rank-5
pipeline ran the matcher on H3 at (1, 1), and section 7 runs the kernel
and the census on H3 at (0, 0)). The N^4 wrappers `stages.py` and
`filters6.py` read the base point from the N cell's `common` and need
the import-level change of section 10. Stages (beta') and (gamma) need
the new matcher only because the N^4 `invisible3.py` reads its base
point and flats from module constants of the N cell; `invisible_p3.py`
is that matcher with the geometry carried by a `common.Cell` (base
point, flats, offset indices), so that the same code runs at (0, 0) for
H3 and would run at (2, 2) for N.

## 2. The 16 flats and the 136 flat multisets at (0, 0)

`probe.py geometry` (`results/geometry.json`). The flats missing (0, 0),
named p{x_1}{x_2} for a point and L{d_1}{d_2}_{points} for a line of
direction (d_1, d_2): the 8 points p01, p02, p10, p11, p12, p20, p21, p22
and the 8 lines L01_101112 = {x_1 = 1}, L01_202122 = {x_1 = 2}, L10_011121
= {x_2 = 1}, L10_021222 = {x_2 = 2}, L11_011220 = {(0, 1), (1, 2), (2, 0)},
L11_021021 = {(0, 2), (1, 0), (2, 1)}, L12_011022 = {(0, 1), (1, 0),
(2, 2)}, and L12_021120 = {(0, 2), (1, 1), (2, 0)}. The coordinate points
are (1, 0) = X0 + e_1 and (0, 1) = X0 + e_2. By the coordinate points they
contain: 9 flats miss both (6 points and the lines {x_1 = 2}, {x_2 = 2},
{(0, 2), (1, 1), (2, 0)}), 6 contain one (the points p10 and p01 and the
lines {x_1 = 1}, {x_2 = 1}, {(0, 1), (1, 2), (2, 0)}, {(0, 2), (1, 0),
(2, 1)}), and one contains both ({(0, 1), (1, 0), (2, 2)}). Every flat
leaves at least five exact points (points of F_3^2 other than X0 on no
invisible flat), and every point other than X0 has ratio 0.366 or 0.134.

The 136 flat multisets of stage (gamma) by kind: 8 pairs of one point
twice, 28 of two distinct points, 24 of a point on a line, 40 of a point
off a line, 8 of one line twice, 28 of two distinct lines. Under the
matcher's processing order (section 4) a two-fresh-term point arises for
exactly the multisets whose two flats share their first processed point:
the 8 equal-point pairs and the 8 equal-line pairs, 16 multisets in all
(`pairs_with_two_fresh_terms_at_a_point` in the record). A point on a
line is never such a point: the line's two other points carry one fresh
term and are processed before the shared point, which carries two.

## 3. The lists

All lists come from the attested rank-5 census files of H3
(`research/qutrit_m4_rank5/results/H3/kernel_census.json`: 6 full
3-covers, 1,211 full 4-covers, 188,451 full 5-covers of distinct
independent states, one per G_2 orbit as far as the pivot and partner
reductions go; `degenerate_covers_H3.json`: 6,112 dependent 5-covers and
7,024 repeated 5-multisets, of which 2,136 are the cancel-at-base
multisets T_3 + (b, b) for the 6 3-covers and the 356 states b outside
each) and from the 6-cover census of section 7. The
lists of stages (beta') and (gamma), with their G_2 orbit counts
(`probe.py lists`, `results/lists.json`):

| stage | list | multisets | G_2 orbits |
|---|---|---|---|
| beta' | full 5-covers (census) | 188,451 | 57,456 |
| beta' | dependent 5-covers | 6,112 | 1,653 |
| beta' | repeated 5-multisets ((2, 1, 1, 1) 4,852 + 2,136 cancel-at-base, (3, 1, 1) 18, (2, 2, 1) 18) | 7,024 | 2,304 ((2, 1, 1, 1) 2,283, (3, 1, 1) 9, (2, 2, 1) 9; (2, 1, 1, 1) dependent 3) |
| beta' | all | 201,587 | 61,413 |
| gamma | full 4-covers (census) | 1,211 | 554 |
| gamma | dependent 4-covers (T_3 plus a span state, full) | 2 | 2 |
| gamma | repeated 4-multisets (T_3 plus a member) | 18 | 9 |
| gamma | all | 1,231 | 565 |

There is no cancel-at-base 4-multiset: chi(H3^2) = 3 leaves no 2-cover.
The orbit reduction is about a factor 3.3 on the 5-covers (N's
was 3.9 with a group of order 72); the H3 group is smaller, so more of the
census's representatives are orbit representatives already.

The k = 6 lists are N's by construction with the H3 inputs:

- A6: the census of full 6-covers of psi_2 (`cover6_pair` over the 2,390
  pivot pairs of `CoverEnumerator3("H3", 2)`, 23 pivots): 36,368,678
  covers, section 7.
- B6: the dependent full 6-sets by the seven routes of the N^4 design
  (T_5 plus a span state; T_4 plus two span states or a parallel pair; T_3
  plus three span states, a span state and a parallel pair, a parallel
  triple, or a collinear triple), enumerated mod P1 and decided
  numerically (`research/n4_rank6/degenerate6.py --orbit H3`, not run
  here). The dominant route is T_5 plus a span state: 19.8 span
  states per census 5-cover on a sample of 200 evenly spaced covers
  (min 4, max 42), about 3.74 million
  (T_5, x) pairs before deduplication and orbit reduction; N's 3.2 million
  pairs became 1.6 million distinct sets and 251,028 orbits by this route.
  With H3's smaller group the orbit reduction is weaker and the B6 list is
  expected at 3e5 to 7e5 orbit representatives, the single largest cost
  of the exclusion (section 8).
- C6: the repeated full 6-multisets by the merged-coefficient routes: a
  5-cover with a member doubled (5 x 188,451 + 5 x 6,112 before
  deduplication), a 4-cover with a member tripled, two doubled, or plus
  (b, b) for any of the 356 other states (1,211 x 356 = 431,116), a
  3-cover with multiplicities (4, 1, 1), (3, 2, 1), (2, 2, 2), or plus
  (b, b) and a member, or plus (b, b, b).

Completeness of the lists is the N^4 argument unchanged: A6 by the pivot
and partner reductions of the census (the H^6 argument) and the kernel's
`is_full` test; B6 by the kappa reduction (a full 6-set with kappa >= 1
contains an independent full k-cover with k <= 5 and rank at most 5, so
it is that cover plus states of its span or parallel modulo it); C6 by
the merged-coefficient argument (the distinct states with nonzero merged
coefficient form a full cover C of 3, 4 or 5 states, the rest are
cancelling blocks of two or more copies of any state outside C, no
fullness test applying to the blocks); the k = 5 and k = 4 lists are the
attested rank-5 lists, which include the cancel-at-base multisets, plus
the dependent and repeated 4-multisets built from the 3-covers.

## 4. The invisible terms at p = 3: the case split

Let the decomposition have k visible terms at X0 and 6 - k invisible
ones, 6 - k in {0, 1, 2} by Fact B. An invisible term is nonzero exactly
on a flat F missing X0, a point or a line; at the first processed point y
of F its slice is c v for a dictionary state v, and along a line the two
other slices are w^{q(t)} Q^t v: 27 phased translates of v at the second
point (code 3 k + l, class k of Q, phase l) and the three phases of the
class of Q^2 at the third. The matcher processes the eight points other
than X0 in the order: points on no invisible flat first ("exact"), then
by the fewest invisible terms whose slice is still unknown ("fresh"),
ties broken by the point's coordinates. At every point the visible terms'
codes range over the values their alive shapes take there (the full shape
table of the structure lemma, 20,008 rows per base slice, restricted by
every code chosen before), the absent code included; a block's copies
enter through a translate set; the coefficient family carries the
visible coefficients (pinned, or kappa parameters for a dependent base)
and gains one coordinate per born invisible term.

### 4.1 One invisible term (stage (beta'), k = 5)

Class I.1, the flat misses both coordinate points (9 flats). The two
coordinate points and at least three composite points are exact; the
family is restricted at each (meet in the middle mod P1 for a pinned
family, the dense solve for a dependent one), the surviving states reach
the flat's first point with a residual r = rhs - (visible terms) - (block
contribution), and r must be c v for a dictionary state v: the scan
(normalized residual against the table of normalized projected dictionary
states mod P1, every match decided by the exact restriction of the
extended family over the three fields). A line continues at its second
and third points with the born term as an ordinary term of 27 and then 3
options. Completeness: every decomposition with this base and flat has
its visible terms' codes in the shape table, so the state carrying its
codes survives every exact point (the slice equation holds exactly, hence
mod P1, hence it is a candidate of the hash and passes the exact
decision), its residual at the flat's first point equals c v exactly, so
v is in the scan's superset and passes, and its codes at the later points
are among the options offered. A zero residual at the flat's first point
with no translate chosen means the invisible term vanishes there, which
contradicts the flat; such a state is dropped (`absent_at_flat`), and the
decomposition, if it exists, is a configuration of a smaller flat run by
the pipeline. A zero projected residual with translates chosen (the
term's slice inside the block's span) keeps the states inside that span
as candidates with the coefficient left as a family parameter, which the
block reconstruction decides (planted kind h).

Class I.2, the flat contains one coordinate point (6 flats). The other
coordinate point and at least four composite points are exact and come
first; the scan happens at the flat's first processed point, which for a
point flat is the coordinate point itself and for a line is its least
point, and the remaining points of the line follow. The T^5 matcher
needed a separate "compatible codes" order for this class because its
composite options were derived from the two coordinate codes; here the
shape table makes every point's options the restriction of the alive
shapes, so the class needs no separate code, only the order. Completeness
as in I.1.

Class I.3, the flat contains both coordinate points (the line
{(0, 1), (1, 0), (2, 2)}). The five composite points off the line are
exact and processed first; the line's least point (0, 1) is the scan
point, (1, 0) and (2, 2) follow with the born term's options. The
coordinate codes of the visible terms are then chosen at scan points,
not exact points, which costs nothing in completeness (the shape table is
restricted by whatever code is chosen wherever it is chosen) and little
in time (section 6: this flat's runs are not slower than the others').

### 4.2 Two invisible terms (stage (gamma), k = 4)

Class II.1, distinct first processed points (96 of the 136 multisets: two
distinct points, a point off a line, two distinct lines). Each term is born
alone at its own first point by the scan of 4.1; the exact points come first,
and a point where one term is born and the other is already born is a scan
with the born term's options among the ordinary ones. Completeness as in I.1,
term by term.

Class II.2, two equal point flats (8 multisets). Both terms are born at
the same point: the residual r must be c_4 v_4 + c_5 v_5 with v_4 != v_5
(two point terms with the same state are one term, and a zero residual is
two cancelling copies of one term; both are rank-5 configurations and are
rejected). The rank-2 scan: v and w span r exactly when their images in
F^9 / span(r) are parallel, so a projective hash of the dictionary modulo
r (three random functionals mod P1) lists the colliding pairs, each
decided mod P2 and over C with agreement required and both coefficients
nonzero. Completeness: a decomposition of this class has r = c_4 v_4 +
c_5 v_5 exactly, so (v_4, v_5) collide and pass.

Class II.3, a point flat on a line flat (the 24 point-on-line
multisets). The order puts the line's two other points, each with one
fresh term, before the shared point, so the line term is born alone by
the scan of 4.1 at the line's least other point, continues at the third
point with its options, and reaches the shared point born; the point term
is then the only fresh term there and is born by a scan with the line
term's three third-point phases among the ordinary options. The point
term's slice may be parallel to the line term's slice at that point
(planted kind "ray"): the scan then finds the residual a multiple of the
same state and the extended family carries both coefficients, which the
later exact restriction pins (the two terms are distinct stabilizer
states of psi_4, since their flats differ, so the decomposition has rank
6 and `confirm` accepts it). The residual at the shared point cannot be
zero for a decomposition of this class, since the point term is nonzero
there. Completeness as in II.1, with the two births in this order.

Class II.4, two equal line flats (8 multisets). At the line's first
processed point y_0 the residual r = c_4 v_4 + c_5 v_5 and the states may
coincide, so there are three cases, all enumerated:

- r of rank 2 (v_4 != v_5): the rank-2 scan births both terms with
  pinned coefficients; the line's second point is exact with 27 x 27
  options for the pair, its third with 3 x 3 (planted kind a on an
  equal-line multiset, two random line terms whose states at y_0
  differ; the kinds below plant v_4 = v_5).
- r = D v, D = c_4 + c_5 != 0 (the same state at y_0, "same-state"):
  carried as two born terms with the same state v whose split (c_4, c_5)
  is a family parameter, decided at the line's second point, where the
  two copies have codes (k_4, l_4) and (k_5, l_5) with either different
  classes (kind "same") or one class with different phases (kind
  "same1"), or equal codes again and different phases at the third
  point (kind "same2"); the dense solve over the one-parameter family
  handles each. The copies may also cancel at the second point with a
  common class and phases l_4 != l_5 and c_5 = -w^{l_4 - l_5} c_4 (kind
  "samec"); the one-parameter family restricted by the second point's
  equation then pins the split at that root and the third point decides.
- r = 0 (a cancelling pair, c_5 = -c_4, the common state v unknown):
  the configuration a hidden-term matcher that births a term at the
  first nonzero residual never sees (section 4.4). It is modeled here
  explicitly, as the N^4 matcher models it: the state is carried to the line's
  second point with a "cancel" marker and no born term, and there the residual
  after the visible terms is scanned for (i) a stabilizer residual a w (both
  copies in one Pauli class of v at that point with phases l_4 != l_5: w = Q v
  up to a phase, a = c_4 (w^{l_4} - w^{l_5}), so c_4 = a / (w^{l_4} - w^{l_5})
  for each of the six ordered phase pairs, and the common slice at y_0 is one
  of the nine class translates s of w with the phases of the second-point
  codes read off from s; kind "cancel1"); (ii) a rank-2 residual c_v v' + c_w
  w' with v', w' in one Pauli orbit and c_v + w^l c_w = 0 for some cube root
  (the copies in different classes at the second point: v' = Q_4 s and w' =
  w^{-l} Q_5 s for a class translate s of v' and a phase m carried into the
  codes; kind "cancel"); (iii) zero again (the pair agrees at the first two
  points and must differ at the third, else it is one term twice: the marker
  becomes "cancel0" and the third point is scanned for a stabilizer residual
  as in (i), with the second-point class fixed by the doubled class of the
  third; kind "cancel0"). In each case both terms are born with pinned
  coefficients (c_4, -c_4) and the remaining point of the line, if any, is
  exact with the born terms' options. A state whose marker survives the last
  point of its line undetermined raises (`UnpinnedFamily`) rather than being
  dropped. Completeness: for a decomposition of this class the two invisible
  terms are c_4 (s, Q_4 s up to phase, Q_4^2 s up to phase) and -c_4 (s, Q_5 s
  up to phase, Q_5^2 s up to phase) on the line; at the second point their sum
  is a stabilizer residual, a rank-2 residual of two states in the Pauli orbit
  of s with coefficients cancelling up to the phase ratio, or zero, which is
  the trichotomy above, and in the first two cases the enumeration over the
  nine class translates and the three phases recovers s and both codes; in the
  third the same trichotomy at the third point has no zero branch (the terms
  would be equal) and the second-point class is determined by the third-point
  class through the doubled-class map, which is a bijection on classes at p =
  3 (k -> 2k).

Two fresh terms at a point of a base with a repeated state, or with an
unpinned family, or with a born term already present, raise
`UnpinnedFamily` (the rank-2 scan is written for a pinned family without
blocks); the stage (gamma) bases with a block are the 18
repeated 4-multisets, and the dependent 4-covers have kappa 1. Section 6
measures how often such a run reaches a two-fresh-term point at all (in
the N^4 run, never: every run died at its first exact point). Three
fresh terms at one point are outside the design (Fact B) and raise.

### 4.3 Assembly and confirmation

Every surviving state is assembled into six terms (the visible terms
from their unique alive shape, the invisible terms from their codes,
which must cover their whole flat, the copies of a block through
`matcher.reconstruct_block` from the residual coordinates at the eight
offsets with the strict coordinate solve) and confirmed by
`Matcher.confirm` (residual against psi_4 over C, rank, independence,
nonzero coefficients, the span condition mod P2); a batch runner would
re-decide every hit from its phase codes (`common.decide_terms`). A hit
is a decomposition only if it is `genuine` (rank 6, exact, independent,
nonzero).

### 4.4 The soundness lesson applied

A review of a hidden-term matcher elsewhere in this project found that
a matcher which births a hidden term at the first point where the
residual is nonzero does not see two hidden terms that cancel at their
common first point, and that the completeness argument written for it
assumed without saying so that the first nonzero residual is the first
point of each hidden flat. The matcher here never uses the first nonzero
residual as a birth criterion: the flats are parameters, so the matcher
knows at which point each invisible term is first present, and at that
point it enumerates the residual's structure, zero included. A zero
residual is decided as follows: in class I it is another flat's
configuration (the term is absent at a point of its flat) and that flat
is run; in II.2 two cancelling point terms are one term, rank 5; in II.3
the order births the line term before the shared point and the point
term's residual there is nonzero by definition of its flat; in II.4 the
cancelling pair is carried explicitly to the line's next point. The
planted kinds cancel, cancel1, cancel0, and samec (section 6) exercise
the carried case on every one of the eight lines.

## 5. The matcher

`research/h3_m4_rank6/invisible_p3.py`; its module docstring is the
specification. `InvisibleMatcherP3(M, cell)` takes a qutrit
`matcher.Matcher` over the two-qutrit dictionary (its option tables,
fields, block routines, `solve_slice3` and `confirm` are reused) and a
`common.Cell` (the base point, the 16 flats and 136 multisets, the offset
index of every point). `run(cover, flats, target)` is the generic entry
(a list of flats of any length with the fresh-term enumeration up to
two), `run_one` and `run_two` the stage entries by flat name. The code is
the N^4 `invisible3.InvisibleMatcher3` with the geometry threaded through
the cell and two additions: a check that no invisible flat contains the
base point, and an explicit raise at three or more fresh terms. The
exact arithmetic is the project's: dictionary states and the planted
targets have entries in Z[w] and go through `Field3` mod P1 and P2
(`field_vector`), the real target's sqrt 3 through `Field3.sqrt3`; every
hash is a superset mod P1 and every decision is mod P2 and over C with
agreement required.

The planted-control generator (`Plants`) draws bases from the attested
rank-5 lists of H3 (independent, dependent and (2, 1, 1, 1) repeated
5-multisets; independent 4-covers), random visible shapes with those base
slices at (0, 0) (`random_shape_term`: a plane, a line through X0 in one
of the four directions, or the point), integer coefficients, and the
invisible terms of the kind asked for: a random point or line term on the
flat (`invisible_term`), the block-hidden term (`plant_hidden`), the
(point, line) ray, and the seven shared-line kinds (`line_pair`: same,
same1, same2, samec, cancel, cancel1, cancel0 with the coefficient
relation c_5 = -c_4 or c_5 = -w^{l_4 - l_5} c_4). `invisible_p3.py
controls` runs them; `tests/test_h3_m4_rank6.py` runs a subset of every
kind in under two minutes.

## 6. Controls

All on one laptop core at nice 19 through `nice -n 19 uv run`, load 10
to 12 from other sessions; records under `research/h3_m4_rank6/results/`.

Planted instances (`invisible_p3.py controls --stage beta --kinds a,d,r`,
Planted instances (`invisible_p3.py controls --stage beta --kinds a,d,r`,
`--stage beta --kinds h`, `--stage gamma`;
`results/control_planted_beta_adr.json`, `control_planted_beta_h.json`,
`control_planted_gamma.json`): the target is the sum of the planted terms with
integer coefficients, the matcher runs against the target's slices on the
planted base and flats, and the planted decomposition must be among the hits.

| stage | kinds | plants | recovered | seconds per plant: median / mean / max |
|---|---|---|---|---|
| beta', every one of the 16 flats | a (independent 5-cover), d (dependent, kappa 1), r (repeated (2, 1, 1, 1)) | 48 | 48 | 0.18 / 2.2 / 94 (one kind-r plant at p12, base (7, 7, 36, 118, 119): 46 to 322 exact solutions at every exact point, the block of two copies of state 7 keeping the states alive; 6 hits, one of them the plant) |
| beta', every flat | h (block-hidden: the invisible term's first slice a translate of a block copy's) | 16 | 16 | 1.35 / 107 / 1,629 (one plant at L10_021222, base (0, 0, 117, 118, 131): 54,891 exact solutions at the last composite point before the scan, 6 hits, one of them the plant) |
| gamma, every one of the 136 flat multisets | a (two random invisible terms) | 136 | 136 | 0.05 / 0.09 / 1.2 |
| gamma, the 24 point-on-line multisets | ray (the point term's slice the ray of the line term's slice there) | 24 | 24 | within the row above |
| gamma, the 8 equal-line multisets | same, same1, same2, samec, cancel, cancel1, cancel0 | 56 | 56 | within the row above; every cancel plant formed at least one pair solution |
| all | | 280 | 280 | |

The 1,629 s plant is a planted target, not a real one (its block of two
copies of state 0 spans most of the slice space at every point, the
effect the N^4 note's two-block plants showed); the per-plant cap of 90 s
did not stop it because the deadline is checked between points and the
run's last exact point is one long `solve_slice3`. On real bases the
repeated class runs in milliseconds (below). The test module's kind-h
instance is on a point flat and takes about a second.

Real bases (`probe.py rate --count 12`, `results/rate_invisible.json`):
12 orbit-spread bases of every class at every one of the 16 flats
(beta') or 136 flat multisets (gamma) against psi_4, with the shape
tables warm:

| class | orbits | bases x units | ms per run: mean / median / max | hits | refused | undecided | where the runs die |
|---|---|---|---|---|---|---|---|
| beta' (1, 1, 1, 1, 1) | 57,456 | 12 x 16 | 23.8 / 2.6 / 548 (the mean carries the shape tables of 60 new base states, built once per state; the median is the warm rate) | 0 | 0 | 0 | first exact point, 192 of 192 |
| beta' (1, 1, 1, 1, 1) dependent | 1,653 | 12 x 16 | 56.0 / 51.1 / 451 | 0 | 0 | 0 | first exact point in 176 of 192; 6 to 18 solutions at the first one to three exact points in 16, dying at the next |
| beta' (2, 1, 1, 1) | 2,283 | 12 x 16 | 7.6 / 5.2 / 117 | 0 | 0 | 0 | first exact point, 192 of 192 |
| beta' (2, 1, 1, 1) dependent | 3 | 3 x 16 | 19.0 / 20.0 / 33 | 0 | 0 | 0 | 18 or 55 solutions at the first two or three exact points in 21 of 48, dying at the next |
| beta' (3, 1, 1) | 9 | 9 x 16 | 9.3 / 9.3 / 11 | 0 | 0 | 0 | first exact point, 144 of 144 |
| beta' (2, 2, 1) | 9 | 9 x 16 | 171 / 173 / 268 | 0 | 0 | 0 | first exact point, 144 of 144 |
| gamma (1, 1, 1, 1) | 554 | 12 x 136 | 0.9 / 0.6 / 221 | 0 | 0 | 0 | first exact point, 1,632 of 1,632 |
| gamma (1, 1, 1, 1) dependent | 2 | 2 x 136 | 6.1 / 2.3 / 697 | 0 | 0 | 0 | first exact point in 143 of 272; 2 solutions at the first one to three exact points in 126, dying at the next |
| gamma (2, 1, 1) | 9 | 9 x 136 | 3.2 / 3.1 / 5 | 0 | 0 | 0 | first exact point, 1,224 of 1,224 |

Every run died without a hit, refusal or undecided result; no run of any
class reached a scan or scan2 point: the first exact point has no
solution for every sampled base at every flat (the '0,0,0,0' histogram),
as in the N^4 run. The two-fresh-term restriction (no blocks, pinned
family) is therefore never exercised by a real base in the sample, and
the planted gamma kinds are the only exercise of the scan2 paths.

Tests (`tests/test_h3_m4_rank6.py`, `uv run --extra challenge --extra test
python -m pytest tests/test_h3_m4_rank6.py`, 17 s): the
geometry (16 flats, 136 multisets, ratios never 1, 9/6/1 by coordinate
points, the points on no invisible flat first in every order, at most
two fresh terms, and a two-fresh-term point for the 16 equal-flat
multisets only),
beta kinds a, d, r on a flat of each coordinate class and kind h on a
point flat, gamma kind a on six multisets of every shape, the ray kind,
the seven shared-line kinds on the line through both coordinate points
and cancel and samec on a diagonal line, real census bases dying without
a hit, and the raises on three fresh terms and on a flat through the base
point.

## 7. The 6-cover census and the stage A6 rate

`probe.py census6 --all` (`results/census6_H3_full.json`, resumable, 670
kernel seconds in all): `cover6_pair` over all 2,390 pivot pairs of
`CoverEnumerator3("H3", 2)` (23 pivots; the sum of M^3 over the pairs is
1.69e10, N's was 8.79e9 over 1,209 pairs) lists 36,368,678 full 6-covers
of psi_2, every one of rank 6 mod P2 (the kernel returned no dependent
6-set on any pair, as for N); the heaviest pair (117, 0) has M = 356
members above the partner and 2,054,452 covers. The count is 193 times
the rank-5 census (188,451) and 0.98 times N's 6-cover census
(37,201,212). The stratified sample run first (`probe.py census6 --pairs
80`, `results/census6_H3_sample.json`: 79 pairs, 2,794,019 covers, 27 s)
projected 7.5e7 by M^3 and 9.9e7 by per-pivot means, a factor two above
the exact count; the sample took its pairs at even ranks of each pivot's
partner list sorted by member count and overweighted the heavy pairs.
The exact count supersedes it.

`probe.py a6rate --pairs 12 --count 5000` (`results/rate_a6_x00.json`):
`Matcher.run` at (0, 0) through `SliceMatch3Kernel` on 5,000 real full
6-covers, 0.61 ms per cover (N's was 0.68 ms through `Matcher.run`, 0.63
through the direct kernel loop), every run dying at the first coordinate
slice (`coord_raw` 0 in 5,000 of 5,000), no hit, no refusal.

## 8. Projection

Laptop CPU-hours at the rates above and, for the stages not measured
here, at the N^4 run's measured rates on the same code (B6, C6: the
exclusion note's section 6, taken at a load of 9 to 10), with the pod
factors of the T^5 and N^4 runs: 1.3 for the compiled stages, 4 for the
Python stages, the B6 filter shown at both. The N^4 B6 kappa-1 rate on
the pod came in at the Python factor (118 pod CPU-hours for 256,970
items), which is the figure used here.

| stage | items (H3) | basis of the count | rate | laptop CPU-h | pod CPU-h |
|---|---|---|---|---|---|
| census of full 6-covers | 2,390 pivot pairs | run | 670 s in all | 0.19 | 0.24 |
| A6 | 36,368,678 covers | census | 0.61 ms | 6.2 | 8 |
| B6 kappa 1 | 3e5 to 7e5 orbits | N's 251,028 orbits by the T_5 route scaled by the (T_5, x) pair count ratio 1.16 and the weaker orbit reduction (32 against 72) | 0.55 s laptop, 1.66 s pod (N, filter) | 46 to 107 | 140 to 320 |
| B6 kappa 2, 3 | 3e3 to 7e3 | N's 2,685 scaled as above | 5.2 s laptop, 21 s pod at cap 2e8 (N) | 5 to 10 | 19 to 42 |
| C6 | 3e5 to 7e5 orbits | N's 321,553 scaled by the 5-cover count (0.95) and the orbit factor | 0.012 to 0.05 s (N, (2, 1, 1, 1, 1)); the tail classes at seconds | 3 to 10 | 10 to 40 |
| beta', 16 flats | 61,413 orbits | measured | 2.6 to 23.8 ms per run (independent), 56 ms (dependent), 7.6 ms (repeated) | 1.2 to 6.6 | 5 to 26 |
| gamma, 136 flat multisets | 565 orbits | measured | 0.9 ms per run (6 ms dependent, 3 ms repeated) | 0.02 to 0.13 | 0.1 to 0.5 |
| total | | | | 62 to 140 | 180 to 440 |

Error bars. The census and A6 counts are exact and their rates measured
on 5,000 covers; beta' and gamma carry the spread of the rate samples
(the max-to-mean ratios in section 6, a factor 2 on beta'). The
B6 and C6 items are the extrapolation: N's lists were built by
`degenerate6.py` in 81 s and the same script with `--orbit H3` would
replace the range by a count in minutes; the range is the (T_5, x) pair
count measured here times N's deduplication ratio (0.5) and an orbit
reduction between N's 10.4 and N's scaled by the group orders, 10.4 x
Error bars. The measured stages (census, A6, beta', gamma) carry the sampling
error of the census projection (20 percent) and of the rate samples (the
max-to-mean ratios in section 6, a factor 2 on beta'). The B6 and C6 items are
the extrapolation: N's lists were built by `degenerate6.py` in 81 s and the
same script with `--orbit H3` would replace the range by a count in minutes;
the range is the (T_5, x) pair count measured here times N's deduplication
ratio (0.5) and an orbit reduction between N's 10.4 and N's scaled by the
group orders, 10.4 x 32 / 72 = 4.6, which gives 3e5 to 7e5. Stage B6 is the
whole uncertainty of the run, as it was for N (118 of 174 pod CPU-hours).

Pod budget. At the pod rates above the run is 180 to 440 pod CPU-hours,
dominated by B6: on 32 cores (30 processes) 6 to 15 hours of wall time, on
64 cores (60 processes) 3 to 7.5 hours. The N^4 run took 174 pod CPU-hours
on a 16-vCPU shared host over about 20 hours of wall time including the
B6 rerun; H3 is 1.0 to 2.5 times that. Nothing is out of
reach: every stage has a measured or N-measured rate and no stage exceeds
a few hundred pod CPU-hours. The one stage without an H3 measurement of
its own is B6 (the filter and the reference on H3 bases at (0, 0)), and
the first B6 batch on the pod would settle its factor before the
partition is trusted, as the T^5 and N^4 notes did.

## 9. What was run

All on the laptop at nice 19, one process at a time, about 1.3 CPU-hours
in all (the planted controls 0.5 of it, the census 0.2). Records under
`research/h3_m4_rank6/results/`.

| step | command | time | result |
|---|---|---|---|
| planted controls, beta a, d, r | `invisible_p3.py controls --stage beta --kinds a,d,r --cap 120` | 107 s | 48 of 48 (`control_planted_beta_adr.json`) |
| planted controls, beta h | `... --kinds h --cap 90` | 1,714 s | 16 of 16 (`control_planted_beta_h.json`) |
| planted controls, gamma | `... --stage gamma --cap 120` | 29 s | 216 of 216 (`control_planted_gamma.json`) |
| tests | `pytest tests/test_h3_m4_rank6.py` | 17 s | 29 passed |
| geometry | `probe.py geometry` | 1 s | section 2 (`geometry.json`) |
| lists and orbits | `probe.py lists --sample 200` | 30 s | section 3 (`lists.json`) |
| matcher rates on real bases | `probe.py rate --count 12 --budget 420` | 52 s | section 6 (`rate_invisible.json`) |
| 6-cover census sample | `probe.py census6 --pairs 80 --budget 300` | 40 s | section 7 (`census6_H3_sample.json`) |
| 6-cover census | `probe.py census6 --all --budget 1500` | 690 s | 36,368,678 covers (`census6_H3_full.json`) |
| A6 rate | `probe.py a6rate --pairs 12 --count 5000` | 25 s | section 7 (`rate_a6_x00.json`) |

## 10. What remains to build before a launch

1. The lists: `degenerate6.py --orbit H3` for B6 and C6 (minutes) and an
   `orbits` record of every list with hashes; the 6-cover census is run
   (section 7).
2. The pipeline in the shape of `research/n4_rank6/` (`driver.py lists`,
   `sample`, `partition`, `batch.py`, `aggregate.py`, the certificate
   script) with `common.py` of this directory as the cell; `stages.py`
   and `filters6.py` of the N^4 pipeline import their cell's `common`
   and would need the same `Cell` treatment as the matcher got here.
3. The B6 rate on H3 bases at (0, 0) (filter and reference), which fixes
   the dominant cost; the C6 tail classes ((2, 2, 1, 1) dependent) at a
   cap.
4. The remaining controls of the N^4 checklist: the orbit lemma on
   planted decompositions with the 32 unitaries, the lists by hash, the
   rank-8 Lean witness at its all-visible bases (recorded, not required),
   and the m = 3 control (the rank-4 decomposition of |H3>^3 by 2 + 1
   slicing).
