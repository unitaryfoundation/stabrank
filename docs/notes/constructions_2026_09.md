# Structured constructions at the exponent-moving cells (2026-09-20)

Scripts and stored data: `research/constructions/` (README there says what
each script validates). Everything below was run single-process at nice 19
on the shared machine; no annealing.

## Cells and what would beat the literature

| cell | board | rank that beats the published exponent (gamma) |
|---|---|---|
| N m=4 | 5 <= chi <= 7 | 6 (0.4077 < 0.4206), 5 (0.3662) |
| H3 m=4 | 5 <= chi <= 8 | 6 (0.4077); 7 improves the cell only (0.4428) |
| T3 m=4 | 6 <= chi <= 9 on the board | 8 (0.4732 < 0.5); rank 7 is excluded, see below |
| S m=5 | 5 <= chi <= 8 | 5 (0.2930); 6 (0.3261) does not beat 0.3155, 6 and 7 tighten the cell |
| S m=6 | 5 <= chi <= 8 | 7 (0.2952) |
| qubit_H m=6 | 4 <= chi <= 6 | 5 (0.3870 < 0.3963) |
| qubit_T m=6 | 4 <= chi <= 6 | 5 (0.3870) |

Correction to the target list: at S m=5 rank 6 gives log_3(6)/5 = 0.3261,
above the published 0.3155, so only rank 5 moves the exponent there (rank 6
and 7 would still tighten the cell).

T3 m=4 rank 7 is closed by a bound already on the board: with
`bounds/T3-m3-lower-8.json` (attested rank-7 exclusion) chi(|T3>^3) = 8, and
projection monotonicity (I (x) <x| on the fourth qutrit, as in
`cert_t3_m4_from_m3.py`) gives chi(|T3>^4) >= 8. The board's
`T3-m4-lower-6.json` predates the m=3 result and could be raised to 8 by the
same certificate pattern (at the attested tier). Rank 8 at m=4 is the only
T3 target left.

## Outcomes

No exact witness was found. Per approach and cell:

### 1. Slice-and-lift as a constructor

Setting (from `verify_challenge/slice_lift.py`): slicing a rank-R
decomposition of |M>^(m+1) along one qudit gives, at each level k with
alpha_k != 0, an R-term (or shorter) decomposition of |M>^m. Every term is
either full (all p slices nonzero, slice k+1 = phase * Q * slice k for one
Pauli Q) or local (|k> (x) v, one nonzero slice). This dichotomy holds for
every stored minimal decomposition and every witness on the board
(`structure.py`: "non-dichotomy 0" everywhere), as the lemma says.

A slice with exactly r = chi(|M>^m) nonzero terms is a minimal
decomposition, hence one of the stored lists up to the slice-preserving
symmetry I (x) U. `relaxed_lift.py` enumerates every configuration with at
least one minimal slice:

- [A] R = r+1, one local term: two slices minimal and Pauli-matched, the
  third slice's residual must be a stabilizer state.
- [B1] R = r+2, two local terms on one slice: two minimal matched slices,
  third residual of stabilizer rank <= 2.
- [B2] R = r+2, two local terms on different slices: one minimal slice, the
  other two residuals each a stabilizer state (a full scan over
  (3^m * 3)^r Pauli-and-phase assignments).

Control: from the 30 rank-3 decompositions of |N>^2 the script recovers
rank-4 decompositions of |N>^3 (18 hits in case A; the known rank-4
decomposition has exactly one local term along every qutrit), and a hit
converts to an exact witness with `to_witness.py`.

| cell | stored list one copy down | result |
|---|---|---|
| N m=4, R=5 | the unique rank-4 decomposition of N^3 | Pauli matching between slices: only the trivial identity match between the two equal-amplitude slices (alpha_0 = alpha_1), in either direction; no assignment matches slice 2 (ratio -2). Residual at the free slice over all 81 phase choices: never a stabilizer state. 0 lifts. |
| N m=4, R=6 | same | B1: 162 residuals tested for rank <= 2 in the 3-qutrit dictionary, none. B2: 43,046,721 assignments per base; the only stabilizer residuals at the first free slice are the 8 trivial ones (a rescaled existing term, from the identity match with a phase), and none of them leaves a stabilizer residual at the second free slice. 0 lifts. |
| H3 m=4, R=5 | the unique rank-4 decomposition of H3^3 | same picture with the roles of the slices moved: the only matches are the identity between the two equal-amplitude slices 1 and 2; slice 0 (ratio a/b) never matches. 0 stabilizer residuals. |
| H3 m=4, R=6 | same | B1: 162 residuals, none of rank <= 2. B2: 8 trivial stabilizer residuals at one slice, none at both. 0 lifts. |
| S m=5, R=5 | the 69 rank-4 decompositions of S^4 (all lifts of the S^3 list) | With alpha_0 = 0 the slice-0 equation is homogeneous. For every decomposition and base slice: 0 Pauli matches between slices 1 and 2 (ratio -1) and 0 solutions of the homogeneous slice-0 equation; hence 0 stabilizer residuals, 0 lifts. |
| S m=5, R=6 | same | B1 is dead with the matches (none). B2 needs (81 * 3)^4 = 3.5e9 assignments per base and decomposition (138 of them, each 81-dimensional); one did not finish in ten minutes, so B2 was not run at S. |
| qubit_H m=5, R=5 | the 30 rank-4 decompositions of H^4 | (needed first, since rank 5 at m=6 forces chi(H^5) = 5 by slicing) 30 x 2 bases x 16,777,216 assignments: 0 strict matches, 0 stabilizer residuals. No rank-5 decomposition of |H>^5 has a slice that is a minimal decomposition of |H>^4. |
| T3 m=4, R=8 | the board's rank-8 witness for T3^3 (`strict_lift_big.py`, scalar-key meet in the middle over 81^4 = 43,046,721 keys per side) | 0 key coincidences at any of the three base slices: the known rank-8 decomposition of |T3>^3 is not a slice of any rank-8 decomposition of |T3>^4. Controls: |T3> lifts to |T3>^2 (3 lifts), |N>^2 does not lift to |N>^3. |

What the stored lists cannot reach, and why: every configuration with no
minimal slice. For R = r+1 that is "all terms full", where each slice is an
(r+1)-term decomposition of |M>^m. `inspan.py` shows that the minimal
decompositions of N^3, H3^3 and S^3 have no dictionary state in their span
beyond their own terms, so an (r+1)-term slice there has r+1 independent
terms (an irreducible 5-term decomposition of a rank-4 state), and listing
those is a rank-5 pivot search over 30240 states that was not attempted.
For qubit_H^4, 16 of the 30 minimal decompositions have 4 further
dictionary states in their span, so a 5-term slice of a rank-5 |H>^5
decomposition can be minimal-plus-in-span there; the coefficients then
carry one free parameter and the slice-1 equation is 64^5 assignments per
five-set, which was not run. The known rank-6 witnesses at qubit_H m=5 and
m=6 and the rank-8 witness at T3 m=3 are all-full along every qudit
(`structure.py`), which is the shape the constructor cannot see.

### 2. Symmetric ansatz

The starting fact is the opposite of what the earlier sessions recorded
about cyclic shifts: the known minimal decompositions are very symmetric as
sets. `symmetric.py` (orbit of the term set under the group generated by
the local Clifford stabilizer of |M> on each copy and the copy
permutations):

| decomposition | unitary group | set-stabilizer order | in S_m alone |
|---|---|---|---|
| N^3 rank 4 (unique) | 1296 | 144 | 6 (all of S_3) |
| H3^3 rank 4 (unique) | 384 | 96 | 6 (all of S_3) |
| S^3 rank 4 (15 listed) | 82944 | 72 to 576 | 1 or 2 |
| qubit_H^4 rank 4 (30) | 384 | 8 to 128 | 2 to 8 |
| qubit_T^4 rank 3 (unique) | 1944 | 216 | 8 |
| T3^3 rank 8 (board witness) | 162 | 2 | 2 |

So an S_m-invariant union of orbits is the right ansatz for N and H3 (it
contains the rank-4 decompositions one copy down) and the wrong one for S
and T3. `perm_symmetric.py` runs it exactly: stabilizer states are
generated flat by flat (RREF basis and coset, then all phase forms), flats
without a large enough permutation stabilizer are skipped without
generating their states, each surviving state's S_m-stabilizer is read off
its phase exponents, and the S_m-symmetrisation of each state with trivial
stabilizer character is an orbit vector of cost |orbit|. A decomposition
with an S_m-invariant term set is a combination of orbit vectors with total
cost equal to its rank, so the search is a cost-pruned pivot search for
subsets of orbit vectors spanning the target. Control: at N m=3 it returns
exactly one S_3-invariant decomposition with at most 4 terms, the known one
(four fully symmetric states).

Results at the target cells (four qutrits: 2452 affine flats, of which 392
have an S_4-stabilizer of order >= 4; 6,235,884 states on them; S_4-orbits
of size <= 6 with trivial character: 48 of size 1, 144 of size 3, 1392 of
size 4, 2832 of size 6; about 100 s to enumerate):

| cell | R | orbit vectors | pivot sets scanned (by number of orbits) | S_m-invariant decompositions |
|---|---|---|---|---|
| N m=4 | 6 | 4416 | 2: 1, 3: 1584, 4: 8040, 5: 17296, 6: 194580 | none |
| H3 m=4 | 7 | 4416 | 2: 1, 3: 1584, 4: 74856, 5: 179728, 6: 194580, 7: 1712304 | none |
| T3 m=4 | 8 | 4572 (156 more of size 8) | 2: 1, 3: 4416; unions of 4 to 8 orbits not scanned (the uncapped run passed ten minutes in the 4-orbit stage and was stopped) | none among unions of at most 3 orbits |

So the symmetry that the rank-4 decompositions of N^3 and H3^3 have does
not extend: no S_4-invariant set of at most 6 (N) or 7 (H3) stabilizer
states spans the four-copy state.

For T3 the single-orbit and 3-orbit stages also show that |T3>^4 is not in
the span of the orbit vectors of size 1 and 3 together (it is once size-4
orbits are included), so an S_4-invariant rank-8 decomposition would need
at least one orbit of size 4, 6 or 8, and the unscanned stages are exactly
the 4-, 5-, 6-, 7- and 8-orbit unions with cost 8.

Qubits (`perm_symmetric_qubit.py`; controls at m=3 recover S_3-invariant
rank-3 decompositions of |H>^3 and |T>^3, and at |H>^4 the S_4-invariant
rank-4 ones): at m=6 with R = 5 only flats with an S_6-stabilizer of order
at least 144 can carry a term, there are 6 such flats, and of the
136,314,886 phase forms on them exactly 18 states have an orbit of size at
most 5 (all 18 are fully S_6-symmetric). Neither |H>^6 nor |T>^6 lies in
the span of those 18 states, so there is no S_6-invariant decomposition of
either with at most 5 terms. Each of these two scans took about 15 minutes
of one core, over the ten-minute guideline; the phase-form scan is the
whole cost and is not resumable in the current script.

Not done: subgroups of S_m other than S_m itself, and subgroups involving
the local Cliffords. The set-stabilizers above show which would be needed
(order-8 subgroups of S_4 for the qubit T-type). The obstacle for a
general subgroup H is not group theory but enumeration: the H-orbit vectors
of cost <= R must be listed, and for H small most of the 7,439,040
four-qutrit states have small H-orbits, so the search is over millions of
orbit vectors. The flat-pruning trick above only works because S_4 is
large.

### 3. Pauli eigensectors (the cat construction beyond T3)

`sectors.py`: for every Pauli string P with full support on the m copies
(one per orbit under the local stabilizer of |M> and the copy
permutations), |M>^m = sum_s Pi_s |M>^m, each sector living in the code
{P = w^s}, Clifford-equivalent to |s'> (x) (an (m-1)-qutrit state) whose
rank is decided in the (m-1)-qutrit dictionary. The bound is
chi(|M>^m) <= sum_s chi(sector s).

| cell | Pauli-string classes | sectors of rank <= 2 | best total |
|---|---|---|---|
| N m=4 | 5 (local stabilizer of order 6 has 2 orbits on the 8 Paulis) | none | >= 9 |
| H3 m=4 | 5 | none | >= 9 |
| S m=4 | 1 | none | >= 9 |
| T3 m=4 | 35 | none | >= 9 (the known sector rank 3 each) |
| T3 m=3 | 20 | X Z^2 (x) X Z^2 (x) X Z^2 and X^2 Z^2 (x)^3 have one rank-2 sector | 8, equal to chi(T3^3) |
| N m=3 | 4 | X^2 (x) X^2 (x) X^2 has one | 8 > 4 |

Every sector at m=4 has rank at least 3, so no Pauli sector decomposition
gets below 9 at any of the m=4 cells; the route needs a sector of rank 2
and there is none. S m=5 was not run: the sectors are 4-qutrit states and
the rank-2 test needs the 7.4-million-state dictionary (9.6 GB dense).
Non-Pauli Cliffords (order-3 elements outside the Pauli group) have
eigenspaces that are not stabilizer codes, so their sectors would need the
full m-qutrit dictionary as well.

### 4. Products with a twist

Every unitary symmetry of |M>^(a+b) is a local Clifford on each copy times a
copy permutation, and it maps a product decomposition across one
bipartition to a product decomposition across another; there is no
Clifford twist that produces a non-product decomposition from a product
one. What remains is merging: two weighted product terms whose sum is a
stabilizer state. `products.py` over every pair of stored decompositions
(one per symmetry orbit on each side):

| cell | split | products tested | weighted pairs each | merges |
|---|---|---|---|---|
| T3 m=4 | 2+2 | 1 | 36 | 0 |
| N m=4 | 2+2, 3+1 | 900, 6 | 36, 28 | 0 |
| H3 m=4 | 3+1, 2+2 | 5, 81 | 28, 36 | 0 |
| S m=5 | 4+1, 3+2 | 69, 15 | 28 | 0 |

Three-into-two merges need a rank-2 test in the (a+b)-qutrit dictionary and
were not run at m >= 4.

## Sharpest negative facts

1. T3 m=4: rank 7 is impossible given chi(T3^3) = 8 (attested); the known
   rank-8 decomposition of T3^3 is not a slice of any rank-8 decomposition
   of T3^4 (43 million Pauli assignments per side, no key coincidence at
   any base slice).
2. qubit_H: a rank-5 decomposition of |H>^6 would force chi(|H>^5) = 5 by
   slicing (both slices are decompositions of |H>^5 with at most 5 terms,
   and chi(|H>^5) >= 5), and no rank-5 decomposition of |H>^5 has a
   minimal-slice configuration; any that exists is all-full along every
   qubit with irreducible or minimal-plus-in-span 5-term slices.
3. N and H3 at m=4: any rank-5 or rank-6 decomposition has no minimal
   slice along any qutrit, i.e. every slice is an irreducible 5- or 6-term
   decomposition of the 3-copy state, since the unique rank-4
   decompositions admit no nontrivial Pauli matching between slices and no
   stabilizer or rank-2 residual.
4. Pauli sectors at m=4 all have rank at least 3 for N, H3, S and T3.
5. No pair of product terms merges at any of the cells.
6. No decomposition with an S_4-invariant term set has at most 6 terms for
   |N>^4 or at most 7 for |H3>^4, although the unique rank-4
   decompositions one copy down are S_3-invariant; no decomposition with an
   S_6-invariant term set has at most 5 terms for |H>^6 or |T>^6 (only 18
   six-qubit stabilizer states have an S_6-orbit of size at most 5, and the
   targets are outside their span).

## What is worth doing next, if anything

- The rank-5 (irreducible) decompositions of N^3 and H3^3: a pivot search
  one rank above `slice_lift.all_decompositions`, which would either close
  the all-full configurations at m=4 through the same lift test or supply
  them.
- H3 m=4 at rank 7 and S m=5 at rank 7: these tighten cells, not exponents.
- Raising `T3-m4-lower` to 8 from the attested m=3 result.

## 2026-09-21: two-qutrit slicing at N m=4 and H3 m=4

Script: `research/constructions/two_qutrit_slice.py` (controls with
`--control`). Single process at nice 19; H3 took 104 s, N 431 s.

### The structure lemma for two-qutrit slices

Slice a decomposition psi^4 = sum_i c_i s_i along qutrits 1 and 2:
u_i^(x) = (<x| (x) I) s_i for x in F_3^2, so sum_i c_i u_i^(x) =
alpha_x psi^2 with alpha_x = alpha_{x_1} alpha_{x_2}. The flat F_i of s_i
projects to an affine flat pi(F_i) of F_3^2 of dimension 0, 1 or 2, so a
term has 1, 3 or 9 nonzero slices. The stabilizer group of s_i projected to
the symplectic coordinates of qutrits 1, 2 has a kernel of at most 9
elements (those of the form I (x) Q), hence an image of order 9 or 81
whose X-part is the direction space of pi(F_i). Reading off the group
elements above the two unit directions gives, for a nine-slice term,

    u^(x_0 + x) = w^{q(x)} Q_2^{x_2} Q_1^{x_1} u^(x_0)

with Q_1, Q_2 two-qutrit Paulis and q a quadratic polynomial on F_3^2 with
q(0) = 0; conversely every such expression is a stabilizer state (it is
C_2 C_1 (phi_q (x) u) with C_j = sum_t |t><t| (x) Q_j^t a controlled Pauli,
which is Clifford because (X^a Z^c)^t = w^{h(t)} X^{at} Z^{ct} with h
quadratic, and phi_q the full-support state with phases w^q). The nine
Pauli translates of u form an orthonormal basis, so each slice is one of
nine basis states times a cube root of unity, the class map is linear (81
maps) and the phases are quadratic (243), giving 19683 nine-slice shapes per
base slice; a line-type term (pi(F) a line x_0 + <a>) has slices
w^{q(t)} Q^t u with 81 shapes per direction, and a point term one. The
counts close the dictionary exactly: 360 x 19683 + 12 x 360 x 81 + 9 x 360
= 7,439,040. This is the generalisation asked for: the one-qutrit lemma's
single X Z^b (x) Q becomes two elements commuting up to phase, and the cube
roots become a quadratic phase pattern.

### Why "all nine slices nonzero for every term" is not the cheap case

With every term nine-slice, every slice is a six-term, non-minimal
decomposition of psi^2, so no slice fixes the coefficients c_i, and the
lift equations are the full rank-6 problem over the 7,085,880 nine-slice
states with six free complex coefficients (about 10^41 configurations
before symmetry). The one-qutrit lemma is finite only because a minimal
slice pins the coefficients and reduces each term to a Pauli class and a
phase; the two-qutrit analogue of that is a slice with exactly three
nonzero terms, and that is the case searched.

### Case M: some two-qutrit slice has exactly three nonzero terms

By copy permutations the bipartition {1,2} | {3,4} is general, and the
monomial Clifford symmetries of |M> (the affine permutations of F_3 fixing
|M>: the swap 0 <-> 1 for N, the swap 1 <-> 2 for H3) with the swap of
qutrits 1 and 2 reduce the base slice x_0 to three classes per orbit. At x_0
the three visible terms form a minimal decomposition of psi^2, one of the
stored lists (30 for N, 9 for H3) up to the unitary symmetry of psi^2 on
qutrits 3, 4, and c_i = alpha_{x_0} d_i. Each visible term has one of the
20008 shapes; each of the R - 3 <= 3 invisible terms vanishes at x_0, so it
is a line-type term on one of the 8 lines missing x_0 or a point term at one
of the 8 other points, with a free coefficient. For every multiset of
invisible shapes (945 distinct coverage vectors k_x over ranks 3 to 6), the
visible terms must leave at each slice x != x_0 a residual
alpha_x psi^2 - sum_visible of stabilizer rank at most k_x; every pattern
leaves at least five slices with k_x <= 1. The search tabulates, per slice,
the 28^3 visible combos (27 present codes plus absent) whose residual is
zero or a stabilizer state, joins candidates from the two most constraining
slices (bucketed shapes, 27 shapes per term pinned at two independent
slices), checks the remaining k <= 1 slices by table lookup and the k = 2
slices by a rank-2 test against the 360-state dictionary, and runs an
exact completion of the invisible terms (private slices pin a term's
slice; the other two slices on its line run over 81 options with pruning;
a shared slice is split over the dictionary pairs) on every survivor.

Per-slice allowed sets are tiny. N at x_0 = (0,0): the three slices with
alpha ratio 1 admit 1 exact and 9 stabilizer-residual combos each, the four
slices with ratio -2 admit at most 1 stabilizer residual, and the slice
(2,2) (ratio 4) admits none, so it must be covered twice; nine of the 30
decompositions (indices 0, 1, 3, 4, 5, 6, 19, 20, 26) keep 17 live patterns
there, the other 21 none; at x_0 = (0,2) and (2,2) every pattern dies on
the slices with ratio -1/2 or 1/4. H3 at
x_0 = (0,0) and (0,1) admits nothing at any slice for any of the nine
decompositions; at (1,1) three decompositions (indices 3, 7, 8) leave the
slices (1,2), (2,1), (2,2) (ratio 1) with 1 exact and 9 stabilizer combos
and the others with at most 1.

| cell | (decomposition, x_0) pairs | coverage patterns | live patterns | pinned pairs | candidates | rank-2 survivors | exact completions |
|---|---|---|---|---|---|---|---|
| H3 m=4, ranks 3..6 | 27 | 25515 | 51 | 51 | 315 | 66 | 0 |
| N m=4, ranks 3..6 | 90 | 85050 | 153 | 153 | 945 | 216 | 0 |

Result: no rank-5 or rank-6 decomposition of |N>^4 or |H3>^4 has a
two-qutrit slice with exactly three nonzero terms (along any 2 + 2
bipartition). Combined with the one-qutrit result above (no minimal
one-qutrit slice), any such decomposition has at least five nonzero terms
at every one-qutrit slice and at least four at every two-qutrit slice, i.e.
every two-qutrit slice is a four-, five- or six-term decomposition of
psi^2 in which the terms are phased Pauli translates of the base states.

Not covered: two-qutrit slices with four or more nonzero terms everywhere
(the "all nine slices nonzero" case sits here), which has no finite
reduction of this kind; S m=5 (slices are decompositions of |S>^3, rank 4,
so a minimal two-qutrit slice has four visible terms with 177147 nine-slice
shapes each on three qutrits, and the per-slice tables become 244^4; the
join would have to be a meet in the middle per slice, not written); T3 m=4
at rank 8 (three visible and five invisible terms cover every slice, so the
residual-rank pruning is empty); the qubit cells (two-qubit slicing of
|H>^6 has four slices that are decompositions of |H>^4, rank 4, and a
minimal slice leaves one invisible term at rank 5, a smaller finite case
than the qutrit ones and a candidate for the next session).
