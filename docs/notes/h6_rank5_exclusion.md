# Excluding rank 5 for |H>^6: design note

Status (2026-09-21). Not run to completion, and not certifiable in its
present form: the enumeration and matching machinery exists
(`verify_challenge/slice_cover.py`, `research/h6_rank5/driver.py`), the
counts below are measured, neither mandatory control has passed (the m = 4
control did not finish, the rank-6 witness control exposes two cases the
argument does not yet cover, section 5), and the 5-cover enumeration is
far outside the local budget (section 3). The
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
hit is confirmed as a decomposition of psi_6. Coefficients d are exact
reductions when the base system is nonsingular mod p (Cramer), and a
singular reduction is detected and refused rather than trusted.

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

`SliceMatcher.run(cover, x_0)`: for each coordinate slice, meet in the
middle over the 33 options per term (32 codes and absent) on a random
functional mod 65521 (1089 against 35937 sums), full checks mod both
primes and numerically; the products of the three solution lists are
typed (absent coordinate directions determine the plane), completed with
the pair signs and the free plane directions, and checked on the four
remaining slices; hits are confirmed against psi_6.

Measured cost: not obtained. A sample of 40 full 5-covers from pivot 0
(11,580 covers came out of that pivot's first partners in 77 s) at the
four base points did not finish in 51 minutes on one loaded core and was
killed. An earlier attempt on the same sample crashed inside the
completion stage, which means at least one cover had solutions on all
three coordinate slices, so the combination stage ran: it enumerates the
sign choices of every term (8 per full term, 32 x 32 x 2 for an
even-plane term) as a Cartesian product over the five terms before
checking the four remaining slices, up to 8^5 = 32,768 products per basis
solution and far more with plane terms, in Python. That stage must become
a slice-by-slice matching (each remaining slice is again a sum condition
on per-term option lists, hashable) before the per-cover cost is
meaningful. The coordinate-slice matching alone (three meet-in-the-middle
passes of 1089 against 35,937 sums) is of order 30 ms per (cover, x_0).

Projected total for stage A (distinct, independent base states) in the
present Python: kernel about 48 pivots at 0.3 to 1 hour each, 15 to 50
CPU-hours; matching about (3 to 10) x 10^6 covers x 4 base points x the
per-run cost above. Both are dominated by Python per-candidate work and
would drop by one to two orders of magnitude in a compiled kernel with the
exact re-check batched; the pattern is the T3 rank-7 scan. The driver
`research/h6_rank5/driver.py` partitions the pivot pairs into batches
(`partition`), runs a batch resumably (`run B`, results in
`results/batch_B.json` with counts, the basis-solution histogram, every
hit and the wall time) and reports coverage (`status`).

## 5. Controls, and the gap they expose

Control 1, m = 4 from psi_3 (one sliced qubit, rank 4, base slice a full
4-cover, one other slice at ratio t^{+-1}, point terms allowed as
"absent"): not completed. The 3460 full 4-covers (no dependent ones) were
enumerated and the matcher ran over them at both base points until it hit
a bug in the completion stage (an unbound variable when there are no
composite slices), fixed afterwards; the rerun was queued behind the
timing sample above and never started. The comparison it was to make (the
recovered rank-4 decompositions of psi_4, canonicalised under the order-384
unitary symmetry group, against the stored 30) is written in the scratch
script and is what would establish that the enumeration plus matching is
complete at one rank down. Until it passes, the pipeline has no positive
control.

Control 2, the rank-6 witness `bounds/qubit_H-m6-upper-6.json`: at every
triple S and every base point x_0 at which all six terms are nonzero,
either two terms have the same base slice up to phase (the witness has
base slices (1035, 0, 619, 622, 20, 0) at x_0 = 000 for five of the ten
triples containing qubits 5 and 6, and (368, 242, 65, 536, 479, 242) at
x_0 = 111) or the six base states are linearly dependent (S = {2, 3, 4}).
The matcher refuses both, by design: it assumes distinct, independent base
states. So the witness is not recovered, and the control fails in the
informative way: it shows two cases the argument of section 2 does not
cover.

Gap (a), repeated base states. Two terms with proportional slices at x_0
are a multiset cover: the merged coefficient on the shared state can be
anything, including zero (then the base slice sees a 3-cover plus a
cancelling pair, 2 x 1080 configurations). The split of the merged
coefficient is a free parameter in every slice equation. With one free
parameter and the pair the only parametrised terms, the slice equation
reads (rho psi_3 - sum_{others} d_i w_i) parallel to (w_a - w_b), which is
still hashable (canonical directions on both sides, 35937 against 1089).
The general multiset (patterns 2+2+1, 3+1+1, 2+1+1+1 with a nonzero merged
coefficient) needs the same treatment with the merged 4- or 3-cover
coefficient family. Not implemented.

Gap (b), dependent base states. A full 5-cover whose states have rank 4
or 3 has a 1- or 2-parameter coefficient family; rebasing to another x_0
helps only if some slice is independent. The dependent full 5-covers are
enumerable (a full 4-cover plus a state in its span, a 3-cover plus a pair
of states parallel modulo its span, a 3-cover plus two states of its
span; `dependent_covers`), their count was not measured. Matching with a
1-parameter family where the parameter touches all five terms is not a
sum condition; the fallback is a batched least-squares over the 33^5 code
assignments per (cover, slice), about 10 s each in numpy.

Until (a) and (b) are implemented and control 2 recovers the witness, no
negative statement about rank 5 follows from stage A, whatever it finds.

## 6. What would close it

1. Implement the multiset and dependent base cases (section 5) and rerun
   control 2 until the witness is recovered from at least one (S, x_0).
2. Move the 5-cover kernel's candidate decision into compiled code (rank
   pairs mod 2013265921 as in `research/t3_rank7/batch.py`) and batch the
   matcher's option tables per base state (they depend only on u_i).
3. Run the partition on idle cores, aggregate, and write the certificate
   on the attested tier (`bounds/T3-m3-lower-8.json` pattern) if the total
   exceeds the 3600 s budget, which at the measured rates it will.

Everything above depends on PR #87's property P; a certificate should
either re-run those tables (501 s) or declare the dependency.
