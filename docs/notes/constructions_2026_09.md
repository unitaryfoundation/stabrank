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

## 2026-09-21: two-qubit slicing at qubit_H m=6 (rank 5) and the m=7 cell

Script: `research/constructions/two_qubit_slice.py` (controls with
`--control`, `--witness`, and the m=4 run below). Single process at nice
19; the rank-5 search over the 30 stored decompositions took 501 s, the
witness control 836 s, the m = 4 control 8 s.

### The structure lemma for qubit slices

Slice a decomposition psi^m = sum_i c_i s_i of |H>^m along n_1 qubits:
u_i^(x) = (<x| (x) I) s_i for x in F_2^{n_1}, a vector on the n_2 = m - n_1
remaining qubits, with sum_i c_i u_i^(x) = alpha_x psi^{n_2} and
alpha_x = cos(pi/8)^{n_1 - |x|} sin(pi/8)^{|x|}, never zero. A term is
nonzero exactly on an affine flat x_0 + V of F_2^{n_1}, and reading the
stabilizer group elements above a basis v_1, ..., v_j of V gives Paulis
Q_1, ..., Q_j on the n_2 qubits with

    u^(x_0 + t_1 v_1 + ... + t_j v_j) = i^{l.t} (-1)^{q(t)} Q_j^{t_j} ... Q_1^{t_1} u^(x_0),

l in Z_4^j and q a quadratic form over F_2 without diagonal. Conversely
every such expression is C_j ... C_1 (phi (x) u) with C_k a controlled
Pauli and phi the full-support stabilizer state on V with those phases, so
it is a stabilizer state; changing a class representative only moves
(l, q). With the 2^{n_2} Pauli classes mod Stab(u) as an orthonormal basis
of translates, a term with base slice u at x_0 has
sum_j [subspaces of dim j] (2^{n_2})^j 4^j 2^{j(j-1)/2} shapes: 8385 for
two sliced qubits and four remaining (8192 four-slice, 192 line, 1 point),
and the count closes the six-qubit dictionary exactly,
36720 x 8192 + 6 x 36720 x 64 + 4 x 36720 = 315,057,600. The qutrit
lemma's cube roots and quadratic phase become fourth roots and a
Z_4-valued form, and the class map is again linear.

### Case M at m=6: a four-qubit slice with exactly four nonzero terms

Rank 5 would give exponent log_2(5)/6 = 0.387 < 0.3963. By copy symmetry
the bipartition {1,2} | {3,4,5,6} is general and x_0 is determined by its
Hamming weight (|H> has no monomial symmetry): 00, 01, 11. At x_0 the four
visible terms are one of the 30 stored rank-4 decompositions of |H>^4 (up
to the unitary symmetry, which acts on qubits 3 to 6 and preserves slices)
with c_i = alpha_{x_0} d_i; the single invisible term is a line term on one
of the 3 lines missing x_0 or a point term at one of the 3 other points, so
at least one other slice is exact (k = 0) and the rest have k <= 1. The
slice ratio alpha_x / alpha_{x_0} is tan(pi/8)^{|x| - |x_0|}, and the
per-slice tables over all 65^4 = 17,850,625 code combinations (64 codes
plus absent per visible term) give, for every one of the 30
decompositions:

| ratio | exact combos | stabilizer-residual combos |
|---|---|---|
| 1 | 1 | 260 to 360 (266 for 14 of the 30) |
| tan(pi/8), cot(pi/8) | 0 | 0 |
| tan(pi/8)^2, cot(pi/8)^2 | 0 | 0 |

The zero at ratio tan(pi/8) is the slice-and-lift exclusion behind
`bounds/qubit_H-m5-lower-5.json`, recomputed here rather than assumed; the
zero in the stabilizer column at ratio tan(pi/8) is new and is what closes
the case: a slice at Hamming distance one from x_0 can carry neither an
exact match (k = 0) nor a single invisible term (k = 1), so it needs k >= 2,
and one invisible term covers each slice at most once. Every coverage
pattern is dead at the table stage for every decomposition and every x_0:
90 (decomposition, x_0) pairs, 630 coverage patterns over ranks 4 and 5
(the rank-4 pattern with no invisible term included), 630 dead, no pinned
pair, no candidate.

Result: no rank-5 decomposition of |H>^6 has a four-qubit slice (any 2 + 4
bipartition) with exactly four nonzero terms. Since chi(|H>^4) = 4, every
four-qubit slice of a rank-5 decomposition has all five terms nonzero, so
every term's flat projects onto F_2^2 for every pair of qubits: the dual of
each term's direction space has minimum distance at least 3 (no term is a
line or point term along any pair of qubits). The same tables say more
about any rank-R decomposition with a minimal four-qubit slice at x_0:
every slice whose Hamming weight differs from |x_0| (all three others for
x_0 = 00 or 11, two of them for x_0 = 01) carries at least two invisible
terms.

### Controls

- `--control`: 300 random six-qubit, 200 four-qubit, 100 (n_1 = 3) and 6
  (n_1 = 4) random stabilizer states (random `(k, x0, W, Q, l)` through
  `common.term_vector`) all have their code pattern in the enumeration for
  their own base slice; the shape counts match the lemma at every (n_1,
  n_2); 30 random sums of one or two invisible terms are recovered exactly
  by the completion at m = 6 and at m = 4.
- m = 4 from the rank-2 slice of |H>^2 (`--m 4 --n1 2 --rank 4 --x0 all`,
  8 s): the single rank-2 decomposition of |H>^2 up to symmetry, sliced at
  x_0 = 01 or 10, yields 33 + 33 = 66 exact rank-4 decompositions of |H>^4
  (two invisible terms, tables with exact rank-2 classes on two qubits);
  at x_0 = 00 and 11 every pattern dies at the table stage. For scale, 57
  of the 180 (decomposition, qubit pair) slicings of the stored rank-4
  list have a two-term slice, so the control has genuine targets.
- The rank-6 witness `bounds/qubit_H-m6-upper-6.json` sliced along qubits
  (i, j), i, j <= 4, has exactly four nonzero terms at x_0 = 01 and 10 (the
  pairs containing qubit 5 have five), and all 20 such (pair, x_0) cases
  give the identical base decomposition (same coefficients and slices,
  checked). The two invisible terms are line terms on the complementary
  line, which is the one pattern with a single constrained slice; the
  dedicated path (129^4 = 276,922,881 assignments after pinning at x_0 and
  the exact slice, the rank-2 support filter on both residuals, the joint
  moduli filter, the exact rank-2 test, the completion) recovers the
  witness: the exact slice 01 (ratio 1) admits one exact combination (the
  identity classes), 332,608 of the 276,922,881 assignments pass the
  filters, 112 pass the exact rank-2 test on both slices, and exactly one
  completes, reproducing psi^6 to 1e-15, in 836 s.

### Not covered, and the m=7 cell

- A two-qubit slice with exactly two nonzero terms (slicing four qubits,
  three invisible terms over 15 slices): the visible terms have 4,703,985
  shapes each and the 2.3 million multisets of invisible flats make the
  pattern-by-pattern join too slow at this budget. The two-qubit tables
  (289 combinations) are cheap and say: ratio 1 admits 1 exact and 32
  stabilizer combos, ratio tan(pi/8)^{+-1} 0 exact and 4 stabilizer,
  ratios tan(pi/8)^{+-2}, tan(pi/8)^{+-3}, and tan(pi/8)^{+-4} 0 and 0
  (`--m 4 --n1 2 --ratio-tables 4`). For x_0 of weight 0 or 4 this
  alone kills the case by counting (every other slice needs k >= 1, the
  eleven at weight >= 2 need k >= 2, and three flats give at most 24
  coverage slots for 26 needed); weights 1 to 3 are open.
- |H>^7 at rank 8 (exponent 0.4286 < 0.4308) along a 3 + 4 bipartition
  with the three-qubit slice minimal: 71,113,185 shapes per visible term
  and five invisible terms that can cover all 15 other slices twice, so
  the residual-rank pruning is empty for those patterns; not attempted.
  With the four-qubit slice minimal instead (four visible, four invisible
  over 7 slices) the four-qubit tables transfer, extended to ratio
  tan(pi/8)^{+-3} (`--m 6 --n1 2 --ratio-tables 3`, 648 s): for all 30
  decompositions the ratios tan(pi/8)^j with |j| = 1, 2, 3 admit neither an
  exact nor a stabilizer-residual combination, so in any rank-8
  decomposition of |H>^7 with a four-qubit slice of exactly four nonzero
  terms at x_0, every slice whose Hamming weight differs from |x_0| carries
  at least two of the four invisible terms (all seven other slices for
  x_0 = 000, five of them otherwise). Four flats of at most four points
  give 16 coverage slots against 14 needed at x_0 = 000, so counting does
  not close it, and the join (2,154,945 shapes per visible term, 73,815
  multisets of invisible flats) was not run.
- Slices with more than r nonzero terms everywhere (all terms four-slice),
  as for the qutrits.

## 2026-09-22: one structured attempt per record-capable family

Scripts: `research/constructions/kvv_cat_m9.py`,
`research/constructions/product_witness.py`,
`research/constructions/t3_sector_contraction.py`. Everything ran as one
process at nice 19; no run longer than ten minutes except the S m=8 anneal
(below, one run under a 1200 s cap). Bound files written today are listed
at the end; none moves an exponent. The literature update of the same day
is section 6 of `literature_sweep_2026_09.md` (nothing new).

### (a) qubit_H at m=9: rank 18, the cell was empty

Every route through the cat-state machinery gives 18 terms at nine copies,
and the routes are distinct as term sets:

| route | terms | distinct rays |
|---|---|---|
| project the glued nine-term cat_10 onto \|0> and onto \|1> on qubit 10 (cat_9 and cat_9^- = (T^9 - T_perp^9)/sqrt 2, nine terms each) | 18 | 18 |
| glue cat_6 with cat_5 = sqrt 2 (I (x) <0\|) cat_6 through <cat_2\| | 18 | 18 |
| 4-to-3 partial decomposition times the board's m=5 witness (3 chi(T^5)) | 18 | 18 |
| products T^6 x T^3 and T^7 x T^2 | 18, 18 | 18, 18 |

Why nothing lower comes out of these pieces: chi(T^9) <= chi(T^10) by
projection, and 18 is also 3 chi(T^5) = 6 chi(T^3) = 9 chi(T^2), so every
factorisation of nine copies into the known pieces lands on the same
number. The cat_5 partial route (|T>^4 = sqrt 2 (I (x) <T|) cat_5, whose
three terms contract to |0^4>/sqrt 2 and two Clifford images of
(stabilizer (x) |H>)) gives chi(T^{4+r}) <= chi(T^r) + 2 chi(T^{r+1}),
which is 18 at r = 5 and 16 > 12 at r = 4, so it never beats the 4-to-3
partial. The five routes use 81 distinct nine-qubit stabilizer states
spanning a 66-dimensional space, and greedy and randomised pruning inside
that pool found nothing shorter than 18 (a heuristic, not an exclusion).
Filed as `bounds/qubit_H-m9-upper-18.json` (projected cat_10, method
"structured construction", exponent log_2(18)/9 = 0.4633, verified
symbolically in 162 s). The rank-11 target (0.3844) is not in reach of any
combination of published pieces. A rank-17 anneal warm-started from the
witness was not run (compute budget; the verification reruns below used
it).

### (b) qubit_T at m=7 to 10: the Clifford-transfer premise is false; two products filed

The brief asked whether the KvdWV witnesses for |H>^m transfer term by term
to the qubit_T orbit through "the single-qubit Clifford that maps H to T".
No such Clifford exists: the H-type (edge-centre) and T-type (face-centre)
states are distinct orbits of the single-qubit Clifford group, which acts
on the Bloch sphere as the octahedral rotation group, and the board's own
values already separate them (chi(F^4) = 3 against chi(H^4) = 4). The
Clifford C = HSH in `kvv_cat.py` maps (|0> + e^{i pi/4}|1>)/sqrt 2 to the
H-type state; both are edge states. So nothing transfers, and the QPG cat
construction for the face state F (their explicit three-term cat_6(F))
has to be redone in the F orbit. Doing so gains nothing at m <= 10:
cat_2(F) = |00> + i|11> is a stabilizer state with F_perp = sin beta|0> -
e^{i pi/4} cos beta|1>, so the gluing identity holds verbatim and
cat_10(F) has at most nine terms, giving chi(F^10) <= 18, and the partial
route gives chi(F^{5+r}) <= 3 chi(F^{r+1}); with chi(F^4) = 3 already on
the board, the plain products are at least as good at every m: 3 x 3 = 9
at m=7 (partial: 3 chi(F^3) = 9), 3 x 3 = 9 at m=8 (partial:
3 chi(F^4) = 9; the brief's "12" is the H-type value), 3 x 6 = 18 at m=9,
3 x 3 x 2 = 18 at m=10.

`product_witness.py` builds product witnesses exactly (x0 and l
concatenated, W and Q block diagonal, coefficients multiplied in sympy).
Filed: `bounds/qubit_T-m8-upper-9.json` (m=4 x m=4, exponent 0.3962,
exactly the baseline; verified symbolically in 44 s) and
`bounds/qubit_T-m10-upper-18.json` (m=4 x m=4 x m=2, 0.4170; 88 s). Not
filed: the m=7 (m=4 x m=3, rank 9) and m=9 (m=4 x m=5, rank 18) products.
They are correct by construction, but the m=3 and m=5 factors have
coefficients in the nested field Q(sqrt 2, sqrt 3, i, sqrt(3 + sqrt 3))
(cos beta = sqrt((3 + sqrt 3)/6) is a degree-4 number, and at odd m the
target amplitudes are odd in it), and the verifier's per-entry
sympy.simplify fails slowly on products of several distinct nested atoms
before the 60-digit fallback decides each entry: the m=7 file did not
finish in 400 s in two runs, so it would not fit the 900 s budget, and the
m=9 file has four times the entries and twice the terms. The board's own
m=5 witness is "checked to 60 significant digits" for the same reason. A
canonical rewrite of the coefficients as p + a q with a = sqrt(sqrt(3)/6 +
1/2) and p, q in Q(sqrt 2, sqrt 3, i) was started and not finished; it is
what would make the odd-m face-state cells verifiable, and is the one
piece of tooling this session leaves open. The merge anneal from the m=8
product minus one term was already run on 2026-09-20
(`merge_recipes_2026_09.md`, section 2: plateaus 0.37 to 0.39) and was not
repeated.

### (c) T3 at m=5 through the sector contraction: rank 9 per sector, no five-term completion

The qutrit analogue of the cat_{a+b-2} gluing. With |T3> = 3^{-1/2} sum_x
w9^x |x>, the Z^{(x)m} eigensector s of |T3>^m is c_s^{(m)}(x) = 3^{-m/2}
w9^s w3^{(|x| - s)/3} on |x| = s (mod 3), |x| the integer digit sum, and
the SUM-gate permutation x -> (x_1, ..., x_{m-1}, |x| mod 3) carries it to
|s> (x) psi_s^{(m)}, the carry state of `autoresearch/run.py`. The three
states Z^i|T3> are orthonormal, so |Phi> = sum_i Z^i|T3> (x) Z^i|T3> =
3 Pi_{ZZ = 1}|T3>^2 is the m=2 sector, a stabilizer state (carry state
w3^{x^2}), and contracting one qutrit of c_j^{(m)} with one of c_k^{(m')}
through <Phi| kills the mixed terms and leaves c_{j+k}^{(m+m'-2)} up to a
scalar; a stabilizer term contracted with a stabilizer bra is a stabilizer
state or zero, so r_{m+m'-2} <= r_m r_{m'}. This is "the Clifford-twisted
maximally entangled state" of the earlier notes, made explicit; the script
checks the identity for all nine (j, k) numerically to 1e-9.

Inputs. psi_s^{(3)} (two qutrits) has 55 rank-3 decompositions per sector
(full pivot search over the 360 states). psi_0^{(4)} (three qutrits) has
exactly 3 rank-3 decompositions (pivot search over the 30,240 states with
one pivot per S_3 orbit, 5784 pivots, closed under S_3; 94 s), one S_3
orbit, and sectors 1 and 2 follow by the Clifford X^{-1} then
diag(1, 1, w3) on the first carry qutrit, which maps terms of psi_s to
terms of psi_{s-1} (the phase gate corrects the carry when the digit
wraps). The board's `T3-m4-upper-9.json` is exactly the union of the three
sector decompositions (each of its nine terms lies in one sector).

Output at m=5. For each sector, 3 x 55 x 3 = 495 contractions; every one
has nine nonzero terms, all stabilizer states, all linearly independent,
so the ansatz gives r_5 <= 9 (chi(|T3>^5) <= 27), worse than the annealed
r_5 = 6 already on the board and than the rank-15 target (r_5 = 5). The
495 nine-term sets use 621 distinct four-qutrit carry states spanning 63
of the 81 dimensions; closed under the S_4 permutations of the carry
qutrits (a symmetry of psi_s^{(5)}) they are 1935 states in 203 orbits
spanning all 81. The board's six-term sector decomposition shares no state
with either pool.

Exact tests on sector 0 (sectors 1 and 2 are its Galois conjugates, so
the same statements hold there):

| test | result |
|---|---|
| five-term decompositions of psi_0^{(5)} sharing four terms with some contraction (stabilizer states in span(psi, four kept), `mergelib.stabilizer_states_in_span` over the 2452 four-qutrit flats) | all 19,710 distinct four-subsets of the 495 nine-term sets tested (two chunks, 334 s and 47 s); 0 completions |
| exact rank-5 pivot-pair search inside the closed pool (first pivot over the 203 orbit representatives, second over the 1935 states, `rank3_search` on the quotient with parallel quotient states merged and every hit re-solved in the full space) | about 1 s per pivot pair, 392,602 pairs, roughly 4.5 CPU-days: not run beyond a 60 s probe (68 pairs, no verified hit); the `--rank5` flag runs it |

So the contraction ansatz is loose by three terms per sector, its terms are
disjoint from the annealed rank-6 decomposition, and no rank-5 sector
decomposition contains four contraction terms. Exact: the enumeration of
the m=3 and m=4 sector decompositions (up to the 1e-9 residual of the
pivot search), the stabilizer tests, the span lists (1e-8 eigenvalue
tolerance). Nothing here is a lower bound.

### (d) S at m=7 and m=8: products, one merge anneal

Products of the board's exact witnesses: m=7 from m=4 x m=2 x m=1
(4 x 2 x 2 = 16; the m=3 file carries a Lean proof and no witness, so the
m=7 product goes through m=2 and m=1; exponent log_3(16)/7 = 0.3605) and
m=8 from m=4 x m=4 (16, exponent log_3(16)/8 = 0.3155, exactly the
baseline, since the m=4 witness is itself the square of the m=2 rank-2
carry decomposition). Both cells were empty. Both witnesses are sixteen
full-support terms (k = m) with quadratic phases, the tensor cubes and
squares of the two m=2 terms, with coefficients in Q(sqrt 3, i).

Both verify symbolically, in 1 s (m=7) and 4 s (m=8): the coefficients stay in
Q(sqrt 3, i) and the entries reduce without the numeric fallback.

Merge anneal (`run.py S 8 15 --warm-from bounds/S-m8-upper-16.json
--seeds 1 --chains 2 --iters 4000`, 6561 amplitudes, logged in
`autoresearch/runs.jsonl`): the warm start prunes the sixteen-term product to fifteen by dropping
the least significant term, and the annealer's cost at the start is
0.4444. At 6561 amplitudes and rank 15 the two chains had reached
temperature 0.0997 of the schedule when the 1200 s cap killed the process,
and the best cost never left 0.4444 (the current cost wandered near 1.0).
Since run.py did not return, the run is logged by hand in
`autoresearch/runs.jsonl` (seed 1, `killed_s` 1200, residual 0.4444, cpu_s
estimated as the wall time). Same signature as the T3 sector product
(`merge_recipes_2026_09.md`, section 1): the product minus one term is a
strict local optimum of the exchange landscape, and a real attempt at S
m=8 rank 15 is the exact one-state completion test on span(psi, fifteen
kept) at eight qutrits (`research/merges/complete1.py`, which would have
to enumerate the eight-qutrit flats; not run), not more annealing.

### Bound files written today

| file | rank | exponent | tier | method |
|---|---|---|---|---|
| `qubit_H-m9-upper-18.json` | 18 | 0.4633 | verified (symbolic, 162 s) | projected glued cat_10 |
| `qubit_T-m8-upper-9.json` | 9 | 0.3962 | verified (symbolic, 44 s) | product m=4 x m=4 |
| `qubit_T-m10-upper-18.json` | 18 | 0.4170 | verified (symbolic, 88 s) | product m=4 x m=4 x m=2 |
| `S-m7-upper-16.json` | 16 | 0.3605 | verified (symbolic, 1 s) | product m=4 x m=2 x m=1 |
| `S-m8-upper-16.json` | 16 | 0.3155 | verified (symbolic, 4 s) | product m=4 x m=4 |

### Sharpest facts of the session

1. qubit_H m=9: every combination of the published cat pieces gives 18;
   the pool of 81 states they use spans 66 dimensions and prunes to nothing
   shorter.
2. qubit_T and qubit_H are not Clifford-equivalent orbits, so no H-type
   witness transfers to the face state; the face-state cat machinery of QPG
   reproduces the products and nothing better at m <= 10, and the odd-m
   product witnesses are correct but too slow for the verifier as written.
3. T3 m=5: the sector contraction of the exact m=3 and m=4 sector
   decompositions gives 9 per sector, its 621 states are disjoint from the
   annealed rank-6 decomposition, and no rank-5 sector decomposition shares
   four terms with any of the 495 contractions.
4. S m=8: the warm-started rank-15 anneal never left its start cost in 1200 s,
   so the product minus one term is a strict local optimum of the
   annealer, as at the T3 sector; the cell now holds the product at the
   baseline exponent.

## 2026-09-23: T5 products, a structured rank-7 search at T5 m=2, qutrit products

Scripts: `research/constructions/product_witness.py` (extended),
`research/constructions/t5_m2_merge.py`. Everything ran as one process at
nice 19 under a hard process-group cap (15 minutes per run), about 1.3
CPU-hours in all. The literature update of the same day is section 7 of
`literature_sweep_2026_09.md` (nothing new). Bound files written today are
listed at the end; none moves a published exponent.

### (a) T5 at m=3, 4, 5: products of the board's own witnesses

The T5 orbit is measured against the single-copy product bound
log_5(3) = 0.6826 (no exponent is published for any p = 5 state), and the
m=2 cell already sits below it (rank 8, 0.6460). Products of the m=1
(rank 3) and m=2 (rank 8) witnesses therefore land below the baseline at
every m; the site does not count that as a beaten exponent, and the notes
of each file say so.

| cell | factors | rank | exponent | verification |
|---|---|---|---|---|
| T5 m=3 | m=1 x m=2 | 24 | log_5(24)/3 = 0.6582 | symbolic, 35 s |
| T5 m=4 | m=2 x m=2 | 64 | log_5(64)/4 = 0.6460 | symbolic, 238 s |
| T5 m=5 | m=2 x m=2 x m=1 | 192 | log_5(192)/5 = 0.6533 | not finished: killed at the 900 s cap |

The m=5 file (192 terms on 3125 amplitudes, coefficients in
Q(w5, sqrt(1/8 - sqrt 5/40), sqrt(1/4 - sqrt 5/10))) is correct by
construction but the verifier's per-entry simplification did not finish
under the 900 s budget (`validate_bounds.py` caps T5 at m <= 5 precisely
because 5^5 amplitudes are the edge of that budget), so it is not filed.
Scaling from m=4 (64 x 625 term entries in 238 s) puts it near an hour.
A witness for that cell would need either a cheaper exact check for
Q(w5) coefficients (the analogue of `_fit_cyclotomic` for T3, deciding
each entry in the number field instead of by `simplify`) or a smaller
decomposition.

`product_witness.py` now takes a factor from the stored decomposition
lists under `research/constructions/data/` when the cell's bound file has
no witness (the Lean and literature entries at N, H3, and T3, m <= 3),
with the coefficients fitted exactly by `fit_coeffs.fit`, and its notes
carry the baseline sentence for orbits without a published exponent.

### (b) T5 at m=2: no rank-7 decomposition keeps five product terms

Cell 5 <= chi(|T5>^2) <= 8; the product of two rank-3 single-ququint
decompositions has nine terms. |T5> has ten rank-3 decompositions over the
30 single-ququint states, not five as `bounds/T5-m1-upper-3.json` says:
one for each pair of basis states {|a>, |b>}, completed by one
full-support state (the pivot search of 2026-09-20 counted them up to the
order-5 Clifford stabilizer, which does not act freely on the pairs). The
55 unordered products (the copy swap identifies (a, b) with (b, a)) give
55 nine-term decompositions of |T5>^2.

Search. Fix five of the nine product terms with free coefficients and ask
for two two-ququint stabilizer states t1, t2 with |T5>^2 in span(five
fixed, t1, t2). Every amplitude in sight is 0 or a power of w5 up to the
normalisation, so the whole problem reduces exactly mod the prime ideal
above q = 10061 = 1 (mod 5), with w5 sent to an element of order 5 in F_q.
For each fixed set the quotient of F_q^25 by span(five fixed, psi) has
dimension 19 (checked for every set; a drop would have been re-decided
exactly as a lower-rank hit); t1, t2 can complete the decomposition only
if their images in the quotient are parallel or one of them vanishes, and
parallel images are found by hashing the canonically scaled quotient
vectors of all 3,900 states in one pass, with no pair loop. Reduction mod
a prime preserves every dependency over Q(w5), so the pass over-reports
and never misses; every report is re-decided over Q(w5) by exact rank
(sympy `DomainMatrix` over `QQ.algebraic_field(w5)`).

| fixed product terms | fixed sets | in-span singles (all the fixed terms themselves) | parallel pairs re-decided exactly | exact completions | wall |
|---|---|---|---|---|---|
| 5 (rank <= 7) | 55 x C(9, 5) = 6930 | 34,650 | 227,700 | 0 | 174 s |
| 6 (rank <= 8) | 55 x C(9, 6) = 4620 | 27,720 | 257,400 | 0 | 211 s |

The parallel pairs are genuine three-term dependencies among stabilizer
states modulo the fixed span (states on a common flat, for instance) that
do not involve the target; none survives the exact test. Controls
(`--control`, 7 s): fixing any seven of the nine product terms recovers
the dropped two (36 of 36); fixing any six of the eight terms of
`bounds/T5-m2-upper-8.json` recovers the other two (28 of 28); fixing
seven witness terms and asking for one state returns exactly the dropped
term (8 of 8) and nothing else, so no rank-7 decomposition shares seven
terms with the rank-8 witness either.

Result: no seven-term decomposition of |T5>^2 contains five terms of any
product of two rank-3 single-ququint decompositions, and no eight-term
one contains six. The rank-8 witness on the board (three points, four
lines, one plane) shares 4 terms with one of the 55 products, 3 with
another, 2 with six, 1 with fourteen, and none with the other 33, so it is
consistent with the six-term statement and shows that a rank-8
decomposition can keep four product terms. A rank-7 decomposition sharing
four product terms would need a pivot loop over the third new state
(3,900 quotient searches per fixed set, about 8 minutes each in the
present Python; not run). Nothing here is a lower bound; the cell stays
5 <= chi <= 8.

### (c) N, H3, and T3 at m=5 and m=6: products from the stored exact decompositions

The brief's product values (N m=5 <= 21, N m=6 <= 49, H3 m=5 <= 24,
H3 m=6 <= 64, T3 m=6 <= 64) come from the witness-bearing bound files
alone; the board's m <= 3 cells at N, H3, and T3 are Lean or literature
entries without a witness, and the stored minimal-decomposition lists
under `research/constructions/data/` hold smaller factors: the rank-3
decompositions of |N>^2 and |H3>^2, the unique rank-4 decompositions of
|N>^3 and |H3>^3, and the three-line rank-3 decomposition of |T3>^2.
`product_witness.py` now fits those exactly (`fit_coeffs.fit`, the
cyclotomic route for T3) and multiplies them, so the products filed are
the best the board's own values imply:

| cell | factors | rank | exponent | verification |
|---|---|---|---|---|
| N m=5 | m=2 (rank 3, decomposition 0 of 30) x m=3 (rank 4) | 12 | log_3(12)/5 = 0.4524 | symbolic, 1 s |
| N m=6 | m=3 x m=3 | 16 | log_3(16)/6 = 0.4206 (the baseline) | symbolic, 1 s |
| H3 m=5 | m=2 (rank 3, decomposition 0 of 30) x m=3 (rank 4) | 12 | 0.4524 | symbolic, 389 s |
| H3 m=6 | m=3 x m=3 | 16 | 0.4206 (the baseline) | symbolic, 2 s |
| T3 m=6 | m=2 x m=2 x m=2 (rank 3 each) | 27 | log_3(27)/6 = 0.5000 (the baseline) | symbolic, 13 s |

All five cells were empty. The N m=6 and H3 m=6 witnesses are the
sixteen-term products whose merge anneals are in
`merge_recipes_2026_09.md`, section 3, now filed as exact witnesses. The
verification times split by the parity of m for H3, as they did for the
face state on 2026-09-22: at odd m the target amplitudes are odd in the
nested radical sqrt((3 + sqrt 3)/6), and sympy's per-entry simplification
is two orders of magnitude slower (389 s for 243 entries at m=5 against
2 s for 729 entries at m=6). The H3 m=5 file is the only one of the five
that came near the ten-minute limit.

Not filed: T3 m=5 stays at the annealed rank 18 (the product m=2 x m=3
would be 24); the S cells at m=5 and m=6 already hold witnesses.

### Bound files written today

| file | rank | exponent | tier | method |
|---|---|---|---|---|
| `T5-m3-upper-24.json` | 24 | 0.6582 | verified (symbolic, 35 s) | product m=1 x m=2 |
| `T5-m4-upper-64.json` | 64 | 0.6460 | verified (symbolic, 238 s) | product m=2 x m=2 |
| `N-m5-upper-12.json` | 12 | 0.4524 | verified (symbolic, 1 s) | product m=2 x m=3 from the stored lists |
| `N-m6-upper-16.json` | 16 | 0.4206 | verified (symbolic, 1 s) | product m=3 x m=3 |
| `H3-m5-upper-12.json` | 12 | 0.4524 | verified (symbolic, 389 s) | product m=2 x m=3 |
| `H3-m6-upper-16.json` | 16 | 0.4206 | verified (symbolic, 2 s) | product m=3 x m=3 |
| `T3-m6-upper-27.json` | 27 | 0.5000 | verified (symbolic, 13 s) | product m=2 x m=2 x m=2 |

The T5 m=5 product (192 terms, 0.6533) was built and not filed: its
verification was killed at the 900 s cap.

### Sharpest facts of the session

1. T5 m=2: no seven-term decomposition of |T5>^2 keeps five terms of any
   of the 55 products of two rank-3 single-ququint decompositions, and no
   eight-term one keeps six; the exact mod-q quotient search decides each
   fixed set in about 25 ms and its three positive controls pass in full.
   |T5> has ten rank-3 decompositions, not five.
2. T5 products land below the orbit's single-copy baseline at m=3 and
   m=4 (0.6582 and 0.6460) because the m=2 cell already does; that is
   the product bound, not a beaten exponent, and the files say so.
3. The board's own minimal decompositions give N and H3 at m=5 rank 12
   and T3 at m=6 rank 27, below the products the witness-bearing bound
   files alone imply (21, 24, and 64).
4. Verification cost for H3 (as for the face state) is governed by the
   parity of m, not by the size of the target: 389 s at m=5 against 2 s
   at m=6.
