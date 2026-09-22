# Excluding rank 5 for |H>^6: design note

Status (2026-09-21). Not run to completion, and not certifiable in its
present form. The enumeration and matching machinery exists
(`verify_challenge/slice_cover.py`, `research/h6_rank5/driver.py`), both
positive controls pass (section 5: the rank-4 decompositions of |H>^4 from
the full 4-covers of |H>^3, and the rank-6 witness from its own base
slices with distinct states; the repeated-state bases of the witness have
not finished), the per-cover cost is measured (section 4) and the full
5-cover run is projected at 100 to 400 CPU-hours in the present Python,
far outside the local budget. The
cell stays at 5 <= chi(H^6) <= 6, with the lower bound the projection of
`bounds/qubit_H-m5-lower-5.json` and the upper bound the QPG cat witness
`bounds/qubit_H-m6-upper-6.json`. Excluding rank 5 would also settle
chi(H^5) = 5 or 6 only indirectly (a rank-5 decomposition of |H>^5 need not
lift), so the two open cells stay coupled but distinct.

Notation. psi_m = |H>^m with |H> = cos(pi/8)|0> + sin(pi/8)|1>, both
amplitudes nonzero. N_3 = 1080 is the number of three-qubit stabilizer
states, one per state up to phase, in the order of `dictionary(2, 3)`. G_3
is the unitary symmetry group of psi_3 (the Clifford stabilizer {I, H} of
|H> on each qubit and the permutations of the three qubits), of order 48,
with 48 orbits on the 1080 states. t = tan(pi/8).

## 1. The restricted dictionary

Six-qubit stabilizer states: 315,057,600, in 4,922,775 stabilizer groups
(Lagrangian subspaces of F_2^12) of 64 states each. A state has support
x_0 + V with V a subspace of F_2^6 of dimension j; the pure-Z part of its
stabilizer group is V^perp.

Property P (PR #87, `research/constructions/two_qubit_slice.py`): in a
rank-5 decomposition of psi_6, every four-qubit slice along every 2 + 4
bipartition has all five terms nonzero (a slice with exactly four nonzero
terms would be one of the 30 rank-4 decompositions of psi_4 and none of
them extends; 501 s of tables). So every term's V projects onto F_2^2 for
every pair of coordinates: V^perp has no word of weight 1 or 2. This
depends on PR #87's computation; nothing here re-derives it.

Property P* (closure). The set of states that can appear as a term is
invariant under the unitary symmetry group G_6 of psi_6 (Hadamards on any
subset of qubits and permutations, order 2^6 6! = 46,080), because G_6
carries rank-5 decompositions to rank-5 decompositions. A Hadamard on qubit
k swaps the X and Z coordinates of qubit k in the stabilizer group, so P
after every element of G_6 says: the stabilizer group contains no
non-identity element of weight at most 2 whose non-identity factors are
all X or Z (elements with a Y factor are allowed). P* is G_6-invariant, P
is not (the all-|+> state has P and H_1 maps it to |0>|+>^5, which does
not), so P* is the right set to pivot on.

Counts (`research/h6_rank5/restricted_dictionary.py`, 11 s, exact by
enumeration of the 2825 subspaces V and the 2^{j(j+1)/2} symmetric forms
on each):

| set                         | states      | stabilizer groups |
|-----------------------------|-------------|-------------------|
| all six-qubit states        | 315,057,600 | 4,922,775         |
| property P                  | 233,889,792 | 3,654,528         |
| property P*                 | 119,608,128 | 1,868,877         |

By support dimension, P* has 1920 groups at j = 3 (30 subspaces, all of
them [6,3,3] codes for V^perp), 120,490 at j = 4, 737,544 at j = 5,
1,008,923 at j = 6; nothing below j = 3, since a [6, k, 3] code needs
k <= 3.

Orbits of G_6 on the P* states: not computed. The exact union-find over
the 1.2e8 states (hash all states, image under 8 generators, connected
components) hashed the states in 523 s but was killed after 104 minutes
in the generator pass as over budget. The count is at least
119,608,128 / 46,080 = 2596 orbits, and the orbit sizes divide 46,080. A
Burnside count over the 65 conjugacy classes of the hyperoctahedral group,
with fixed points counted on stabilizer groups and their 64 sign patterns,
would be the cheap way to finish it; it was not needed for the route below,
which never pivots on six-qubit states.

Why the direct pivot search is out of reach even on P*: a rank-5 search
over N = 1.2e8 states is a pivot pair and three further members, at least
N^2 per pair; with 2596 pivot orbits and about N/|Stab| partners each it is
of order N^3 / |G_6| = 4e19 steps. No kernel makes that a computation.

## 2. The argument: a base slice with every term visible

Slice a rank-5 decomposition psi_6 = sum c_i s_i along a set S of three
qubits: u_i^(x) = (<x| (x) I) s_i for x in F_2^3, and sum_i c_i u_i^(x) =
alpha_x psi_3 with alpha_x = cos(pi/8)^{3 - |x|} sin(pi/8)^|x|. Each term
is nonzero on an affine flat of F_2^3 whose direction W_i is the projection
of V_i onto the coordinates S. By P, W_i has dimension 3 (a full term, all
eight slices nonzero) or 2 (a plane term, four slices nonzero): a
projection of dimension <= 1 would give V_i^perp a word of weight <= 2
supported in S.

Lemma (an all-visible slice exists). There are S and x_0 in F_2^3 at which
all five terms are nonzero. Proof: a term is a plane term along S exactly
when V_i^perp has a word of weight 3 supported on S, and V_i^perp is a code
of length 6 with minimum distance 3 and dimension at most 3, hence at most
7 nonzero words, so each term is a plane term along at most 7 of the 20
triples S. Five terms give at most 35 (term, S) incidences. If every S had
at least two plane terms there would be at least 40. So some S has at most
one plane term, and any x_0 on that term's plane (any x_0 at all if there
is none) sees all five terms.

At such (S, x_0), by the structure lemma of PR #87, with v_1, v_2, v_3 the
coordinate directions,

  (i)  the base slice is a "full 5-cover" of psi_3: five stabilizer states
       u_i = u_i^(x_0), with sum_i d_i u_i = psi_3, d_i = c_i / alpha_{x_0}
       all nonzero;
  (ii) at x_0 + v_k the term is i^{l_k} Q_k u_i (one of 8 Pauli classes mod
       Stab(u_i) times a fourth root of unity, 32 options) or zero when v_k
       is not in W_i; at x_0 + v_a + v_b it is a sign times the product of
       the two class representatives (the sign absorbs the quadratic form
       and the change of representative); at x_0 + v_1 + v_2 + v_3 the sign
       is the product of the three pair signs (checked in the note's
       derivation: with representatives Q'_k = Q_k g_k, g_k in Stab(u), the
       triple conversion factor is the product of the pair factors).
       A plane term missing a coordinate direction has its other basis
       direction free (32 options) and one pair sign.

Slice ratios: the equation at x_0 + e is sum_i d_i w_i = t^{|x_0 + e| - |x_0|}
psi_3. The three coordinate slices carry ratio t^{+-1}; by the permutation
symmetry of S only four x_0 need to be run (one per Hamming weight).

Exactness. All amplitudes lie in Q(zeta_16); the primes 65521 and
2013265921 are 1 mod 16, so both give ring maps Z[zeta_16, 1/2] -> F_p.
Linear dependencies reduce, so a modular kernel lists a superset of the
true covers and a modular match is a superset of the true matches; every
candidate is re-decided mod the second prime and in floating point, and a
hit is confirmed as a decomposition of psi_6. The coefficients d form an
affine family (a point when the base states are distinct and independent)
computed in both fields and over C; its dimension is the complex one,
checked against the second prime, and a larger dimension mod 65521 (a
modular rank accident) only enlarges the hashing superset.

## 3. Enumeration of covers of psi_3

`CoverEnumerator.covers(r)` lists full r-covers up to G_3 by pivot (one
per orbit), partner (minimal in its Stab(pivot)-orbit, members in orbits
at or above the pivot's, as in `slice_lift.all_decompositions`) and, for
r = 5, a third pivot k with the last two members parallel modulo
span(psi_3, u_i, u_j, u_k) (an (M, M, 5) residue array per pair mod
65521). "Full" means some solution of sum d u = psi_3 has every
coefficient nonzero, decided from the affine solution space (a coordinate
is dead iff it vanishes on the particular solution and on the kernel).

Measured on one low-priority core (load average 40 to 50 on 18 cores):

| r | full covers (tuples) | G_3 classes | candidates | time  |
|---|----------------------|-------------|------------|-------|
| 3 | 2                    | 2           | 3,850      | 0.3 s |
| 4 | 3,460                | 2,406       | 1,617,832  | 93 s  |
| 5 | >= 1,239,946 after pivots 0 to 3 of 48 | not computed | 36,597,459 | 3227 s (killed) |

Cross-check at r = 3 and 4 against `slice_lift.all_decompositions`
(numeric, independent code): identical classes at r = 3; at r = 4 the
reference lists 3731 tuples (2671 classes) and this kernel 3460 (2406
classes), a difference of 271 tuples (265 classes). Every one of the 271
fails the fullness test and contains a rank-3 decomposition of psi_3 as a
3-subset (`is_full` false, some 3-subset is a cover): the reference's
rank-4 kernel drops states whose image vanishes mod span(psi, s_i, s_j)
but not a fourth state that is merely independent of a 3-cover, so it
lists 4-sets in which one coefficient is forced to zero. Such a 4-set
cannot be a base slice of a rank-4 decomposition one copy up (its fourth
term would have coefficient zero), so the two lists agree on what matters,
and the reference is the superset. The check is
`is_full` on each of the 271 and `is_cover` on their 3-subsets, both mod
2013265921 and in floating point.

The r = 5 count is the problem: at least 1.24 million full 5-covers after
four of 48 pivots (the first pivots have the largest member sets, so the
total is perhaps 3 to 10 million tuples), and 36.6 million modular
candidates re-decided one by one in Python at about 90 microseconds each.
Per pivot pair the residue array costs about 20 ms in numpy for M near
1000; the exact re-checks dominate.

## 4. Matching, and its cost

`SliceMatcher.run(cover, x_0)` handles any multiset cover. The distinct
states of the cover get an affine coefficient family d = d_0 + K lambda
(`Family.from_cover`, solved mod 65521, mod 2013265921 and over C; the
exact dimension kappa is the complex one, checked against the second prime,
and the dimension mod 65521 may exceed it by a modular rank accident, in
which case that family is a superset and only weakens the hashing). A state
repeated g times becomes a block (`Block`): at every slice the copies
together contribute an arbitrary vector of the span of at most g Pauli
translates of the shared base state, and since the 8 translates are a
basis of C^8 the slice equation is projected onto the annihilator of the
chosen translate set (37 sets for a pair, 93 for a triple). The copies'
coefficients, classes and phases are reconstructed at the end from the
residuals (`reconstruct_block`, a depth-first assignment over the seven
slices with the coefficient family of the copies, each copy checked to be a
stabilizer state through `valid_term_codes`); for a pair the admissible
splits (c_1, c_2) with c_1 + c_2 = D are also tracked from slice to slice
(`_refine_split`: a slice on two translates pins them up to phases, a slice
on one translate constrains them once pinned), which is what removes the
multiplicity of section 5.

Per slice (`solve_slice`): when the family is a point, meet in the middle
on a random functional mod 65521 over the per-term option lists (33 per
term on a coordinate slice), then the whole equation mod 65521 on every
collision at once, then the exact check over C and mod 2013265921. When
kappa >= 1 parameters remain, the condition is that u = sum d_0i w_i - rhs
lie in the span of v_j = sum K_ij w_i, tested as det = 0 mod 65521 for the
(kappa + 1) x (kappa + 1) matrix of kappa + 1 random functionals applied to
(u, v_1, ..., v_kappa); the determinant is Laplace-expanded into C(2 kappa
+ 2, kappa + 1) features of the two halves of the terms (6 for kappa = 1,
20 for kappa = 2) and the halves are joined by a dense float64 product
(exact below 2^53), 1.3e9 entries and about 40 s for six terms, 39
million and about a second for five. Every accepted solution restricts the
family, so after the first slice the family is usually a point and the
later slices are meet-in-the-middle passes. The absence pattern on the
coordinate slices fixes each ordinary term's flat among all 16 subspaces of
F_2^3 (the matcher does not assume property P), the composite slices are
solved point by point over the option codes generated by the structure
lemma (`composite_codes`: class products with one free sign per basis pair,
the product of the pair signs on the triple, 32 free codes on a basis point
that is not a coordinate direction), and each hit is confirmed against
psi_6 in floating point and mod 2013265921 (`confirm`: residual, rank,
independence, nonzero coefficients).

Measured cost (`driver.py sample --count 40`, one core at nice 19 on a
machine with load average 18 on 18 cores): 40 full 5-covers from the first
pivot, four base points each, 160 (cover, x_0) runs, no hit. Per cover
(all four base points) mean 0.110 s, median 0.019 s, maximum 0.94 s. The
first coordinate slice has no solution in 146 of the 160 runs; the rest
have solution counts (1, 1, 1) on the three coordinate slices (6 runs),
(2, 4, 8) (4 runs) and (9, 81, 729) (4 runs, the expensive ones). The
kernel produced the 40 covers in 43 s.

Degenerate covers (`driver.py degenerate`, 208 s): 26,242 full 5-covers
whose base states are dependent or repeated, over the two 3-cover classes
and the 3,460 4-cover classes without further symmetry reduction: 12,390
with five distinct dependent states, 13,840 with one state repeated
(3,460 x 4), 6 with a triple, 6 with two pairs. Their per-cover cost was
not sampled; the dependent ones start with a 1-parameter dense solve
(about a second for five terms) and the repeated ones with 37 projected
meet-in-the-middle passes per slice, so a few seconds per cover is the
expectation, and the whole degenerate list is of order 10 to 30 CPU-hours.

Projection for the full run. Stage A matching at the sampled mean of
0.110 s per cover over the 3 to 10 million full 5-covers the kernel is
expected to produce (section 3) is 90 to 300 CPU-hours; at the median it
would be 15 to 50. The kernel itself is 15 to 50 CPU-hours at the measured
rates (the partition's calibrated estimate is 8.1 CPU-hours over 14,280
pivot pairs in 17 batches). Stages B and C add the degenerate list above.
So the full exclusion is 100 to 400 CPU-hours in the present Python, all
of it per-candidate work that a compiled kernel would cut by one to two
orders of magnitude.

## 5. Controls

Control 1, m = 4 from psi_3 (`driver.py control-m4`; one sliced qubit,
rank 4, base slice a full 4-cover): passes. Bases: the 3,460 independent
full 4-covers and the 6 dependent or repeated ones (a 3-cover plus a state
of its span or one of its own states), 6,932 (cover, x_0) runs in 9 s, 95
hits, all genuine rank-4 decompositions of psi_4, forming 23 classes under
the unitary symmetry group of psi_4 (order 384; the stored list of 30 in
`research/constructions/data/qubit_H_m4_rank4.json` also collapses to 23
classes under that group). All 23 stored classes have an all-visible slice
along some qubit with a full base cover, so all 23 were expected, and 23
were recovered: nothing missing, nothing unexpected, nothing outside the
stored list. Enumerating the bases took 101 s (the 4-cover kernel).

Control 2, the rank-6 witness `bounds/qubit_H-m6-upper-6.json`
(`driver.py control-witness`): at every triple S and base point x_0 at
which all six terms are nonzero, the base is one of four (cover, x_0)
pairs, each shared by 10 triples: six distinct states of rank 4 at x_0 =
000 and at 111 (the triples inside qubits 0 to 4), or five distinct states
of rank 4 with one repeated (the triples containing qubit 5). Every base
has kappa = 2. On the two distinct bases the control passes: from
(1035, 0, 619, 908, 349, 1) at 000 and from (368, 242, 65, 180, 166, 460)
at 111 the matcher returns exactly one genuine rank-6 decomposition (rank
6, exact mod 2013265921, all coefficients nonzero, residual 4e-15), and it
is the witness itself (same six term patterns after the slicing
permutation). Cost 373 s and 396 s per base (`--distinct-only`, one core at
nice 19 on the loaded machine): the first coordinate slice is a 2-parameter
dense solve (80 million hash candidates over the run), the coordinate
slices then have 4,930, 3,650 and 120,544 solutions, all joined states have
a pinned family, and the composite stage over 240,992 flat-type selections
leaves the single hit.
On the repeated bases the matcher is sound but has not finished: the
projected (span-of-translates) treatment of the pair is weak on this
witness because several ordinary base states are Pauli translates of the
repeated state, so the residuals lie in the translate span for structural
reasons; slice 0 gave 10,192 solutions and 403 states, slice 1 151,861
solutions and 15,333 states, and the run was stopped at the 9-minute cap
before slice 2. The split tracking of section 4 was added after that
measurement and its effect on these bases has not been measured.

Gap (a), repeated base states, and gap (b), dependent base states, of the
earlier revision are implemented (blocks, coefficient family). What remains
on the control side is the repeated-base run above.

## 6. What would close it

1. Run `control-witness` without `--distinct-only` to completion on the two
   repeated bases (or bound the state growth there, for instance by solving
   two coordinate slices jointly for the ordinary terms).
2. Move the 5-cover kernel's candidate decision into compiled code (rank
   pairs mod 2013265921 as in `research/t3_rank7/batch.py`) and batch the
   matcher's option tables per base state (they depend only on u_i).
3. Add the degenerate covers (`degenerate_covers`) to the partition as
   stages B and C, run the partition on idle cores, aggregate, and write the
   certificate on the attested tier (`bounds/T3-m3-lower-8.json` pattern)
   if the total exceeds the 3600 s budget, which at the measured rates it
   will.

Everything above depends on PR #87's property P only through the lemma of
section 2 (an all-visible slice exists); the matcher itself does not use
P. A certificate should either re-run those tables (501 s) or declare the
dependency.
