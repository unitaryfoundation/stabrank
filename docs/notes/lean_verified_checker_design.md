# Bringing the scan-based lower bounds to the Lean tier: design note

Status (2026-09-22). Design, plus the first step of section 3 (the
throughput measurement, section 5) and the first pieces of the dictionary
completeness lemma (`lean_proofs/LeanProofs/Stabilizer/Reparam.lean`,
`QutritDict2.lean`; section 5). The question is
what it would take to move the lower bounds that now sit on the
`reproduced` and `attested` tiers (CONTRIBUTING, "Five tiers") to the
`lean` tier, where a bound is a theorem about `stabRankP` in
`lean_proofs/`. Three families of argument are on the board:

- the m=3 rank-4 pivot exclusions `S-m3-lower-4`, `N-m3-lower-4`,
  `H3-m3-lower-4` (`verify_challenge/rank_exclusion.py`), and the same
  method at four qubits behind `qubit_H-m4-lower-4` and `qubit_T-m4-lower-3`;
- the m=5 slice-and-lift exclusions `qubit_H-m5-lower-5`,
  `qubit_T-m5-lower-4`, `S-m5-lower-5` (`verify_challenge/slice_lift.py`),
  and the m=4 lifts `N-m4-lower-5`, `H3-m4-lower-5`;
- the two attested scans, `T3-m3-lower-8` (research/t3_rank7, 1.31e13
  inner steps) and `qubit_H-m6-lower-6` (research/h6_rank5, 190 batches),
  with the verified two-pivot scan `T3-m3-lower-7` as their small sibling.

Summary. Every family splits into pure mathematics that Lean does not have
yet and a finite computation. Four pieces of the mathematics are shared by
all three families and dominate the cost: a completeness lemma for the
dictionary of stabilizer states (every `IsStabP` vector is a scalar multiple
of a listed normal form), a reduction lemma from complex-linear dependence
to dependence over a prime field, the covering lemma behind the symmetry
reduction, and the exactness of the candidate decisions. Once those exist,
the m=3 rank-4 cells are a few times 1e7 to 4e8 field operations each,
which the Lean kernel can run by `decide` only as hundreds to thousands
of small theorems, since its memory grows with every operation (section
5), or which `native_decide` runs in minutes; their certificates are a
few thousand exact witnesses. The m=5 lifts are within
reach for `qubit_T^5` and `S^5` and need compiled evaluation for
`qubit_H^5`. The two attested scans are out of reach for the kernel at
any cost and would need weeks of formalisation plus hundreds of CPU-hours
of `native_decide` even to attempt.

Notation. p is the local dimension (2 or 3), n the number of qudits, N the
number of n-qudit stabilizer states up to phase (1080 on three qubits,
36720 on four, 30240 on three qutrits), psi the target vector. D is the
phase period (p for odd p, 4 for p = 2), zeta_D = exp(2 pi i / D), and K
the smallest cyclotomic field containing zeta_D and the amplitudes of psi:
Q(w3) for S, N, T3, Q(zeta_12) for H3 (its amplitudes are proportional to
(sqrt3 + 1, 1, 1)), Q(zeta_16) for the qubit H-type state (cos and sin of
pi/8), Q(zeta_24) for the qubit T-type state (a phase zeta_8 and the ratio
sqrt2 (sqrt3 + 1) / 2). ell is a prime with ell = 1 mod M when
K = Q(zeta_M), so that the cyclotomic polynomial Phi_M has a root mod ell
and the integers of K reduce to F_ell; 65521 is 1 mod 48 and serves every
cell. A field operation is one multiply-and-reduce mod ell. Engineering
days are estimates for one person fluent in Lean 4 and Mathlib; they are
guesses within a factor of two.

## 0. What Lean has now, and what every family needs

Present in `lean_proofs/` (README): `stabRank` and `stabRankP p` against
the predicate `IsStabP p` (a nonzero multiple of `stabVecP` with an
injective affine support parametrisation), the reduction lemma
`stabRank_gt_of_no_decomp_le` (no set of at most k `IsStabP` vectors spans
psi, and some decomposition exists, gives `stabRankP psi > k`), the
computational basis as a stabilizer basis (`decompCardsP_nonempty`), the
tensor bound, and two lower bounds that avoid enumeration: the affine
support argument of `StrangeM2Lower` and the Galois descent of
`T3GaloisDescent` and `T3GaloisM` (a span of vectors over Z[w3] that
contains |T3>^m contains all three conjugates, so `stabRankP 3 |T3>^m > 2`).
Nothing in Lean enumerates the dictionary, decides membership in a span by
computation, or uses a symmetry group.

Four lemmas are missing and are shared by all three families.

Dictionary completeness. `IsStabP p v` is an existential over
(c, k, x0, W, Q, l). To decide anything by enumeration Lean needs: every
such v is a nonzero multiple of a term in an explicit finite list, the
list of normal forms that `cert_t3m3_rank7.enumerate_terms` produces (W
in reduced row echelon form, x0 zero on the pivot columns, Q upper
triangular, l free). The proof is the reparametrisation y -> A y + b by
an element of GL_k(F_p) and a translation, which carries W to echelon
form, moves x0 off the pivot columns, and turns the phase into another
quadratic-plus-linear form plus a constant absorbed into c. Injectivity
forces k <= n. For odd p the phase bookkeeping is a direct computation;
for p = 2 the phase is 2 Q(y) + l.y mod 4 and the substitution produces
carry terms 2 y_i t_i that have to be moved into the quadratic part. This
lemma cannot be replaced by a finite check: the raw parameter space at
p = 3, n = 3, k = 3 has 27 * 3^9 * 3^9 * 27 = 2.8e11 points. Mathlib has
no reduced row echelon form, so an executable elimination with a proof
that it preserves the affine image and reaches echelon form has to be
written. Estimate: 10 to 15 days for odd p, 5 more for p = 2. It is the
single largest item and the prerequisite for every enumeration-based
lower bound at n >= 2.

Reduction to a prime field. Every checker below works mod ell and relies
on: if vectors with entries in the ring of integers of K are linearly
dependent over C, their reductions mod a prime above ell are dependent
over F_ell. The route that avoids algebraic number theory: descend
coefficients from C to K by `mem_span_descend` (already in
`T3GaloisDescent`), write elements of K = Q(zeta_M) as f(zeta_M)/g(zeta_M)
with f, g in Q[X] (`IntermediateField.mem_adjoin_simple_iff`), clear
denominators to Z[X], observe that dependence makes every maximal minor
vanish, so each minor polynomial D(X) in Z[X] has D(zeta_M) = 0, hence
the cyclotomic polynomial Phi_M divides D in Z[X] (`cyclotomic_eq_minpoly`,
monic divisor), hence D(z) = 0 in F_ell for any root z of Phi_M mod ell.
Then a matrix over a field whose maximal minors all vanish has dependent
rows (independent rows give independent columns to pick, whose square
submatrix has nonzero determinant). Everything computable is over ZMod
ell; the complex side is polynomial identities. Estimate: 4 days. The
same lemma gives the exactness direction the Python scans state as
"reduction can only add candidates".

Covering lemma for symmetries. A C-linear or conjugate-linear bijection g
with g(psi) = c psi and g(s_j) = c_j s_{pi(j)} for every dictionary state
carries decompositions to decompositions, so if some decomposition
contains s_j and g moves s_j onto its orbit representative, a
decomposition containing the representative exists. The pure lemma is
twenty lines. What costs is verifying the generators: the copy
permutations, complex conjugation, and diagonal or permutation Cliffords
act on exponent tables by relabelling and are checked by `decide`; the
qutrit Fourier transform and the qubit Hadamard (the only symmetry of
|H> besides copy permutations) are not monomial, and their action on a
state is a Gauss sum that has to be evaluated exactly in Z[zeta_12] or
Z[zeta_8] per dictionary element (30240 * 27 * 3 ring operations per
generator, a few times 1e6, fine for the kernel). No Clifford covariance
theorem is needed: completeness of the dictionary plus a per-element
exact check that g(s_j) is a scalar times s_{pi(j)} is enough, and the
orbit forest (for each j a parent and a generator label, N entries) is
part of the certificate. Estimate: 4 days. Without symmetry the m=3
qutrit sieve is C(30240, 2) = 4.57e8 pivot pairs, which excludes the
kernel outright (section 2), so this lemma is not optional for route (a).
The measured price of using only the monomial part of each group is in
the table at the end (74 pivots instead of 12 for S, 563 instead of 74
for N, 563 instead of 116 for H3, 1548 instead of 186 for the qubit
H-type at m=4): a factor 5 to 8 in kernel time, which is why the exact
verification of the non-monomial generators is worth its four days.

Exact decisions. The sieve mod ell lists a superset of the true
configurations; the survivors must be decided exactly. Two options.
First, exact arithmetic in K, represented as Z[X] mod Phi_M with
rational coefficients, cheap when the survivors are few thousand (family
1) and the witnesses are small (a relation s_k = a s_i + b s_j, or a dual
functional phi with phi(s) = 0 on the candidate set and phi(psi) != 0,
which is the general certificate that psi is outside a span). Second, the
modular exactness the T3 scans use: a rank computed mod ell2 = 2^31 - 1
is exact up to 7 because a nonzero (r x r) minor with root-of-unity
entries has field norm at most (r!)^2 by the Leibniz formula (the
Hadamard bound the Python uses is sharper but harder to formalise), and
(8!)^2 = 1.6e9 < ell2, so a minor that vanishes mod ell2 vanishes. This
works for K of degree 2 (Z[w3], Z[i]); for Z[zeta_16] the norm has eight
factors and (5!)^8 = 4.3e16 exceeds 2013265921, so the H^6 decisions
would need exact K-arithmetic or a 64-bit prime. Estimate: 3 days for
the functional-witness form, 5 for the modular-rank form.

Two representational choices apply throughout. States are exponent
tables (per digit string, either "absent" or an exponent in Z/D), which
is what `term_exponents` produces and what `stabVecP` evaluates to on
the support; the evaluation lemma linking `stabVecP` to its table is one
day. Certificates are shipped as compact `Array Nat` literals or parsed
from a string by a verified parser, because Lean elaborates a list
literal of 1e5 elements slowly; nothing below needs more than about 1e6
numbers.

## 1. Statement chains and the finite computations

### 1.1 The m=3 rank-4 pivot exclusions

Claim: `stabRankP 3 psi > 3` for psi = |S>^3, |N>^3, |H3>^3.

Pure mathematics: the reduction lemma (present); dictionary completeness
(missing); the pivot-and-quotient lemma, that if psi lies in
span(s_i, s_j, s_k) then the images of s_j and s_k in C^27 / span(psi, s_i)
are parallel or one of them vanishes (five lines); the covering lemma
with the symmetry group of psi (order 165888, 2592, 768); the reduction
to F_ell; and, for the dependent candidates, that a set of three states
of rank two spans what two of them span, so psi in it contradicts the
rank-2 exclusion (two lines).

Finite computation, measured this session with the certificate scripts
(each 5 to 7 s on eight cores):

| cell | pivots | partner steps (pivots x (N-1)) | candidates | independent |
|------|--------|--------------------------------|------------|-------------|
| S^3  | 12     | 362,868                        | 1,404      | 0           |
| N^3  | 74     | 2,237,686                      | 8,658      | 0           |
| H3^3 | 116    | 3,507,724                      | 13,572     | 0           |

One partner step is: reduce a 27-vector against the two pivot rows
(psi, s_i) mod ell, canonicalise the residue (one inverse, 27 products),
and insert or compare its key; about 1e2 field operations. The rank-2
pass (all N residues modulo psi pairwise non-parallel) is one more
canonicalise-and-sort of 30240 vectors. Every candidate was a dependent
triple (three states in one two-dimensional stabilizer subspace), so the
exact decisions are 1,404 to 13,572 relations s_k = a s_i + b s_j with
a, b in Q(w3) (the states are over Z[w3] whatever the target; the target's
field matters only for the sieve), each checked in 27 ring operations.
Totals: about 4e7 (S), 2.5e8 (N), 4e8 (H3) field operations for the
sieve, plus sorting, plus under 1e6 for the decisions.

The same method at four qubits (dimension 16, N = 36720) feeds family 2:
`qubit_H^4` rank 3 (`qubit_H-m4-lower-4`), 186 pivots, 6,829,734 partner
steps (about 5e8 field operations). The lower bound chi(T^4) >= 3 behind
`qubit_T-m4-lower-3` is a rank-2 exclusion (`cert_qubit_rank2.py`), one
canonicalise-and-sort of the 36720 residues modulo psi, about 4e6
operations; the rank-3 sieve at `qubit_T^4` (62 pivots, 2,276,578 steps,
about 1.6e8 operations) is the enumeration of its rank-3 decompositions
for family 2, not an exclusion. Candidate counts at m=4 were not
re-measured here; the certificates report them.

### 1.2 The m=5 slice-and-lift exclusions

Claim: `stabRankP p phi^(m+1) > r` where r = chi(phi^m) is known exactly
(the upper bound is a Lean witness in every case on the board: the
tensor square for |H>^4, `QubitTM4StabRank` for |T>^4, `M3StabRank` and
the tensor square for |S>^3 and |S>^4).

Pure mathematics: the slice-and-lift lemma of `slice_lift.py`, in two
parts. Part (a), that a slice of a rank-r decomposition at a coordinate
value with nonzero amplitude is a rank-r decomposition of phi^m with no
vanishing term (from minimality: a vanishing or dependent term would
give phi^m a shorter decomposition), is short. Part (b), that the
slices of one stabilizer term are related by a Pauli operator and a
D-th root of unity, is where the work is: the Python proof uses the
stabilizer group, which the Lean predicate does not have, but the same
statement follows from the parametrisation directly. If the first
column of W is zero the term is a product with a single nonzero slice,
excluded by (a); otherwise the slices are hyperplane sections of the
y-space that differ by a fixed translation t, and the phase difference
Q(y + t) - Q(y) is linear in y, which is a Z-type Pauli, while the
support shift W^T t is an X-type Pauli. For p = 2 the carry terms in
l.(y + t) mod 4 add a further Z factor. Estimate: 5 days for odd p, 3
more for p = 2. Then: completeness of the list of rank-r decompositions
of phi^m up to the unitary symmetry group (the covering lemma again,
now on decompositions, using that a symmetry I (x) U preserves slices;
one day), the reduction to F_ell for the lift equation (the equation has
finitely many discrete unknowns and exact coefficients, so a complex
solution reduces to a solution mod ell), and minimality of every rank-r
decomposition when chi = r (immediate).

Finite computations, from the bound files and this session's counts:

| cell | list to enumerate | pivots | pivot pairs | inner steps | lift options per term | lift work per decomposition |
|------|-------------------|--------|-------------|-------------|-----------------------|-----------------------------|
| qubit_T^5 >= 4 | rank-3 decompositions of |T>^4: 1 class | 76 | none (rank 3: every other state is a partner) | 2.8e6 | 2^4 classes x 4 phases = 64 | 64^3 = 2.6e5 brute force on 16-vectors |
| S^5 >= 5 | rank-4 decompositions of |S>^3: 5 classes, lifting to 27 of |S>^4 | 12 | 9,204 | 1.82e8 | 81 at m=3, 243 at m=4 | meet in the middle, 81^2 = 6,561 sums then 243^2 = 59,049 sums on 27- and 81-vectors |
| qubit_H^5 >= 5 | rank-4 decompositions of |H>^4: 30 classes | 246 | 1,929,617 | 2.79e10 | 64 | 64^2 = 4,096 sums on 16-vectors |

An inner step of the rank-4 enumeration is a reduction of a 16- or
27-vector against three pivot rows, a canonicalisation, and a hash;
about 1e2 field operations. So the enumerations are about 3e8 (T^4,
rank 3), 2e10 (S^3, rank 4), and 3e12 (H^4, rank 4) field operations.
The lift checks are negligible against them: the largest is S^4 -> S^5,
27 decompositions times two tables of 59,049 sums of 81-vectors, about
3e8 operations in all. The m=4 lifts `N-m4-lower-5` and `H3-m4-lower-5`
have the same shape as S^3 -> S^4 (rank-4 lists at m=3, 88 and 170
unitary pivot orbits), not measured here.

### 1.3 The big scans

T3^3 rank 7 (`T3-m3-lower-8`). Pure mathematics: Galois descent to
V_3 = span of the three conjugates (present in Lean as
`tConjM_mem_span_of_overZomega3`); Lemma 1 of `t3_rank7_exclusion.md`,
that a rank-7 decomposition is seven independent states whose images
modulo V_3 span four dimensions (dimension counting, one day);
completeness of the three-pivot order with the orbit-block relabelling
and the Stab(i, j) minimality of the third pivot (the argument of
section 2 of that note; a careful case analysis, about 10 days, and the
part most likely to hide a gap since `check_order.py` only samples it);
the exactness of the rank pair mod 2^31 - 1 (section 0, Leibniz form
suffices since ranks up to 8 are what is used); and the re-splitting of
spurious classes. Finite computation: 1.3125e13 inner steps (each a row
reduction of a 3-vector mod 65521 and a hash, about 10 field operations)
and 1,213,458,815 candidate class sets, each decided by a rank pair on
up to about ten 27-vectors mod 2^31 - 1 (about 1e3 operations). About
1.3e14 field operations for the sieve and 1e12 for the decisions;
288,520 CPU-seconds in numba at 21 ns per step.

T3^3 rank 6 (`T3-m3-lower-7`, verified tier) is the same argument one
rank down with two pivots: 326,368 pivot pairs, at most 6.2e9 inner
steps (from the recorded three-pivot total, the mean number of states
above the second pivot is at most 18,900), 259,423 class sets, 270
CPU-seconds. About 6e10 field operations plus 3e8 for the decisions.

H^6 rank 5 (`qubit_H-m6-lower-6`). Pure mathematics: property P (a
finite computation dressed as a lemma: none of the 30 rank-4
decompositions of |H>^4 extends across a two-qubit slice, 501 s of
tables over 65^4 code combinations per decomposition and ratio, of order
1e10 operations); its closure P* under the symmetry group; the
all-visible slice lemma (a counting argument: a term is a plane term
along at most 7 of the 20 triples since a [6, <= 3, 3] code has at most
7 nonzero words, so five terms give at most 35 incidences against the 40
needed to block every triple; two days once codes are in place); the
slice structure lemma for three sliced qubits (the multi-qubit version
of part (b) above, with the pair signs and the triple sign; 8 days);
completeness of the 5-cover enumeration up to G_3 (three pivots, 48
orbits, the same order argument as the T3 scan; 8 days); and
completeness of the matcher's option sets (33 per term per coordinate
slice, the composite codes, the coefficient family for dependent bases,
the block treatment of repeated bases; 15 days, and the repeated-base
case has no structural argument, only an enumeration). Finite
computation, from the 190 stored records: 834,686,307 cover-kernel
candidates over 14,280 pivot pairs (about 7e10 operations), 23,862,828
matched runs over 5,965,707 covers (stage A at about 1e4 to 1e5
operations per run, 1e12 in all; stage B's 12,390 dependent covers each
start with a dense solve over 3.9e7 pairs, 2e12 in all); 264,249
CPU-seconds. About 3e12 field operations, and the decisions are exact
only through two primes plus floating point, which Lean would replace
by exact Q(zeta_16) arithmetic.

## 2. Three routes for the finite part

Throughput. Measured in section 5 (this paragraph originally assumed 1e5
to 1e6 field operations per second and a ceiling near 1e9 operations per
theorem; both were wrong). `decide +kernel` on a reflected row reduction
over `List Nat` runs at about 1.6e4 operations per second and, what
matters more, allocates about 15 KB per operation that is not released
until the declaration is checked, so a theorem's operation count is capped
by memory: about 2.5e5 operations under a 4 GB limit in a module with no
Mathlib import, and about 5e4 in a module that imports Mathlib, whose
mapped `.olean` files alone occupy 3.3 GB. Compiled evaluation
(`native_decide`, interpreted, no `precompileModules`) runs at about
6.4e6 operations per second. The estimates in (a) and (b) below are
restated with these figures.

(a) Full kernel computation. Write the checker as a Lean function on
`List Nat`, prove it sound against `stabRankP`, and close the finite
claim by `decide +kernel`. With the measured figures the unit of work is
a theorem of at most about 2.5e5 operations (15 s, 4 GB), and it has to
live in a module that imports nothing, with the soundness link to
`stabRankP` in a separate Mathlib module that only restates the
computed facts. Family 1 is then 4e7 to 4e8 operations per qutrit cell,
that is 160 (S^3) to 1600 (H3^3) modules of 15 s each, 40 minutes to 7
CPU-hours per cell, and 1.6e8 to 5e8 per qubit cell at m=4, 640 to 2000
modules; `lake` builds them in parallel, and the raw-recursor style of
`Bench/KernelBenchRec.lean` halves both figures. This is feasible for
S^3 and N^3 and heavy for H3^3 and `qubit_H^4`. Family 2: qubit_T^5 at
3e8 is 1200 modules, at the edge; S^5 at 2e10 is 8e4 modules and not
sensible; qubit_H^5 at 3e12 is not feasible. Family 3: 1.3e14 and 3e12
operations are not feasible by five to eight orders of magnitude, nor is
the two-pivot T3 scan at 6e10.

(b) Compiled evaluation. `native_decide` closes the same claims by
running the compiled checker and adds the axiom `Lean.ofReduceBool`,
which trusts the Lean compiler, the C compiler, and the runtime in
addition to the kernel. Measured at 6.4e6 operations per second through
the interpreter (section 5); `precompileModules` would run the compiled
C and should gain a factor of five to twenty. At 6.4e6: family 1 in one
to two minutes per cell, qubit_T^5 in a minute, S^5 in an hour,
qubit_H^5 in five days, the two-pivot T3 scan in three hours, H^6 in
five days, and the three-pivot T3 scan in 2e7 seconds, about 5700
CPU-hours, against numba's 80; with precompilation these fall to the
figures originally guessed here, family 1 in seconds, qubit_H^5 in hours
and the three-pivot scan in a few hundred CPU-hours, if the compiled Lean
gets within a factor of five to ten of numba's 21 ns per step. The repository has no policy on
`native_decide`. The trade-off: the mathematics (dictionary
completeness, the descent, the covering lemma, the soundness of the
checker) is kernel-checked either way, and only the finite evaluation
is trusted to the compiler; the `lean` tier as defined in CONTRIBUTING
says "a Lean module builds and its theorem is the bound as stated",
which a `native_decide` proof satisfies literally while resting on a
larger trusted base than every present Lean bound. If it is used, the
receipt should record `#print axioms` and the board should mark the
bound distinctly (a `lean-native` marker, say), so that a later
kernel-checked proof replaces it under the equal-rank rule the way
`reproduced` replaces `attested`.

(c) Verified checker with an emitted certificate. Lean checks a
certificate that the Python scan writes. For non-existence claims over
a dictionary the certificate cannot shrink the sieve: the claim that no
two of N residues are parallel has no witness shorter than the residues
themselves, and computing them is the search. What a certificate does
remove is everything that is not the sieve: the exact decisions (a
relation or a dual functional per candidate instead of exact linear
algebra), the symmetry (verified generators and an orbit forest instead
of a group computation), the decomposition lists (exact coefficients
instead of a solve), and the sort (a permutation to verify instead of a
sort to perform, a marginal saving). Sizes and checking cost per family:

- Family 1: candidates with a relation over K, 1,404 to 13,572 entries
  of two K-numbers (two to four integers each), under 1e5 integers; the
  orbit forest, N entries; the generators as permutations of exponent
  tables, a few times N entries. Checking cost is the sieve, 4e7 to 5e8
  operations, plus under 1e7 for the certificate. Kernel-feasible.
- Family 2: the decomposition list with exact coefficients (1 to 30
  entries of r indices and r K-numbers), dual functionals for the
  candidate r-sets that are not decompositions (count not measured;
  the S^3 rank-4 search reports 2160 decompositions from 5
  representatives, and candidates are a small multiple), the orbit
  forest. Checking cost is the enumeration sieve, 3e8 (T^4) to 3e12
  (H^4) operations, plus the lift tables, under 3e8. Kernel-feasible
  for qubit_T^5, marginal for S^5, compiled only for qubit_H^5.
- Family 3: the T3 scan's 1.2e9 class sets would be about 1e10 integers
  as a certificate, tens of gigabytes, and they are a by-product of the
  sieve in any case; the H^6 run's certificate is empty (no hits) and
  every one of the 2.4e7 matched runs has to be recomputed. Checking
  cost equals search cost, 1e14 and 3e12 operations. Not kernel-feasible;
  compiled only, with the caveats of (b).

So route (c) is not an alternative to (a) or (b) but the shape both
take: a reflected checker whose inputs are the dictionary, the target,
and a small certificate, closed by `decide +kernel` where the count
allows and by `native_decide` where it does not.

## 3. Recommended path

Ordered by payoff per day. Days are cumulative estimates for one
person; the shared library (steps 1 to 4) is about 30 days and every
cell after the first is a few days.

0. Measure kernel throughput (half a day). Done, section 5: 1.6e4
   operations per second and 15 KB per operation, so the binding
   constraint is memory, not time, and family 1 needs theorems of at
   most 2.5e5 operations in Mathlib-free modules from the start; the
   kernel route stays open for S^3 and N^3 and is heavy for the rest.

1. Shared library, part one (about 8 days): exponent tables and the
   evaluation lemma for `stabVecP` (1 day); the cyclotomic reduction
   lemma of section 0 (4 days); the dual-functional and relation
   witnesses for "psi is outside a span", exact over K as Z[X] mod
   Phi_M (3 days).

2. Dictionary completeness for p = 3 (10 to 15 days): executable
   elimination over ZMod 3 with the reparametrisation lemma, and the
   proof that the list generated the way `enumerate_terms` generates it
   contains every normal form. Started, section 5: the reparametrisation
   lemma for every prime and every k, the full-support normal form for
   k = n, and the complete dictionary at two qutrits (360 tables, checked
   against the Python count) are in `lean_proofs/`. What remains is the
   reduced row echelon form for 0 < k < n at general n, which the
   three-qutrit dictionary needs; the phase bookkeeping for p = 2 is
   already generic in `PhaseP.lean`, so the extra five days estimated
   for qubits shrink to the carry-free relabelling of the tables.

3. Symmetry (4 days): exact per-element verification of the generators
   (monomial ones by relabelling, the Fourier transform by a Gauss sum
   in Z[zeta_12]), the orbit forest check, the covering lemma. The pure
   covering lemma is in `Stabilizer/Covering.lean` (section 5).

4. The family-1 checker and the first cell, S^3 (about 7 days): the
   pivot sieve over ZMod ell with the rank-2 pass, its soundness through
   the pivot-and-quotient lemma and the reduction lemma, the candidate
   decisions, one theorem per pivot, and the glue to
   `stabRankP 3 |S>^3 > 3`. S^3 first because it has 12 pivots and
   1,404 candidates (about 4e7 operations, minutes of kernel time) and
   its target is over Z. Then N^3 (2 days) and H3^3 (3 days, the target
   brings sqrt3 and K = Q(zeta_12)). Payoff: three cells at the Lean
   tier in both directions, since their upper bounds are already there,
   and the first Lean-checked exhaustive lower bound in the repository.

5. Dictionary completeness for p = 2 (5 days), the rank-3 exclusion
   `qubit_H^4 >= 4` (2 days; 5e8 operations, 186 pivots), and the rank-2
   exclusion `qubit_T^4 >= 3` (1 day; one sort of 36720 residues). These
   settle chi(H^4) = 4 and chi(T^4) = 3 in Lean and are the base of
   family 2.

6. Slice-and-lift (about 15 days): part (b) of the lemma from the
   parametrisation for odd p (5 days) and p = 2 (3 days); the
   decomposition-list checker (the rank-3 and rank-4 sieves with
   parallel groups, exact coefficients for the listed decompositions,
   witnesses for the rest; 5 days); the lift checker over ZMod ell (2
   days). First cell `qubit_T^5 >= 4`: one decomposition class, a
   2.8e6-step enumeration, a 64^3 brute-force lift; kernel-feasible.
   Then `S^5 >= 5` (S^3 rank-4 enumeration at 2e10 operations, the two
   lift stages under 3e8; per-pivot theorems at the kernel ceiling, or
   `native_decide` if step 0 says so), and `N^4 >= 5`, `H3^4 >= 5` by
   the same route. `qubit_H^5 >= 5` last: its enumeration is 3e12
   operations and needs (b) whatever the throughput; the decision to
   accept `native_decide` gates it.

7. `T3^3 >= 7` (about 10 days after the above, compiled only): the
   two-pivot sieve modulo V_3 with Z members, Lemma 1, and the modular
   rank exactness in Leibniz form; 6e10 operations, minutes natively.
   Worth doing as the dress rehearsal for the pivot-order proof if the
   three-pivot scan is ever attempted, and not otherwise.

What the symmetry reduction needs in Lean, stated once: the dictionary
completeness lemma (so that "g maps the dictionary to itself" is a
finite check), exact evaluation of each generator on each exponent
table (monomial generators by relabelling, non-monomial ones by exact
cyclotomic arithmetic), an orbit forest from the certificate, and the
covering lemma. The alternative that avoids symmetry, every pivot with
every partner above it, is 4.57e8 pairs at three qutrits and 6.74e8 at
four qubits, about 5e10 operations per cell: compiled only, and it buys
nothing once the four days for symmetry are spent.

## 4. Out of reach at reasonable cost

The three-pivot T3 scan (`T3-m3-lower-8`). 1.3e14 field operations is
five orders of magnitude above the kernel ceiling and cannot be split
into theorems small enough (459 batches of 3e10 steps are each already
a hundred times the ceiling). Compiled evaluation would take an
estimated 1500 to 3000 CPU-hours of Lean-compiled code, run once,
against 80 CPU-hours in numba, and the completeness of the pivot order
with the Stab(i, j) mask would need about ten days of case analysis to
formalise, after the shared library. The certificate route does not
apply: 1.2e9 class sets cannot be shipped and are not the expensive
part. The bound stays `attested` until either the kernel is a thousand
times faster or a new mathematical reduction shrinks the search, and
the note that produced it already looked for such reductions and found
none (`t3_rank7_exclusion.md`, section 4; `t3_rank7_rare_pivot.md`).

The all-visible base slice argument (`qubit_H-m6-lower-6`). The
computation, 3e12 operations, is compiled-only but not absurd (about
ten hours at 1e8 per second). The mathematics is the obstacle: property
P is itself a finite computation over the 30 rank-4 decompositions of
|H>^4 whose list is a 3e12-operation enumeration (section 1.2), so the
argument has two large enumerations stacked; the slice structure lemma
for three sliced qubits, the cover enumeration order, and the matcher's
completeness over dependent and repeated bases are together 30 to 40
days of formalisation with no reusable payoff beyond this cell; and the
exact decisions rest on two primes plus floating point rather than a
single bound, so Lean would need exact Q(zeta_16) arithmetic in the
matcher. Sixty or more days for one cell, all of it under
`native_decide`. If chi(H^5) is ever settled at 5 or 6 by a rank-5
decomposition or its exclusion, the m=5 -> 6 step becomes a
slice-and-lift of family 2 (the H^5 rank-5 list lifted one copy), and
that would be the route to take instead.

Anything with a dictionary above 36720 states by direct pivot search:
five qubits have 2,423,520 states, four qutrits 7,439,040, and a pair
sieve is N^2 / |G|. Nothing on the board asks for it, and the slice
arguments exist to avoid it.

## 5. Measured, 2026-09-22

Kernel throughput (`lean_proofs/LeanProofs/Bench/`, Lean 4.29.1, Apple
silicon laptop, `nice -n 19`, one `lake build` per module,
`/usr/bin/time -l` for peak RSS). The checker is a Gaussian elimination
mod 65521 over `List (List Nat)` written with ordinary structural
recursion, on pseudo-random `N x N` matrices; it returns its own
operation count (one operation is one `(a + m * b) % 65521` or
`x * m % 65521`), and the theorem asserts the returned triple.

| module | operations | kernel time | peak RSS | route |
|--------|------------|-------------|----------|-------|
| `KernelBench4` | 30,360 (N = 45) | 1.83 s | 0.88 GB | `decide +kernel` |
| `KernelBench` | 100,232 (N = 67) | 6.39 s | 1.94 GB | `decide +kernel` |
| `KernelBenchRec` | 90,000 (300 row operations on 300 entries) | 2.87 s | 1.12 GB | `decide +kernel`, raw `List.rec` and `Nat.rec` |
| `NativeBench` | 100,232 | below 0.1 s | | `native_decide` |
| `NativeBench` | 995,280 (N = 144) | 0.16 s | | `native_decide` |
| `NativeBench` | 10,026,640 (N = 311) | 1.56 s | 0.74 GB (whole module) | `native_decide` |

The baseline RSS of a `lean` process with no imports is 0.4 GB, so the
two kernel sizes give 15 KB of resident memory per operation and 1.6e4
operations per second; the raw-recursor variant gives 8 KB and 3.1e4.
The memory is the kernel's caches (whnf and defeq results and the
instantiated `brecOn` bodies), which are not released until the
declaration is checked, and it is why the 1e6 and 1e7 kernel sizes asked
for in step 0 were not run: at 15 KB per operation they need about 15 GB
and 150 GB, against a 4 GB limit per module here. Their extrapolated
times are about one minute and ten minutes. `native_decide` through the
interpreter (this project does not set `precompileModules`) runs at
6.4e6 operations per second.

Two further figures from the dictionary work: a module that imports
Mathlib starts at 3.3 GB resident from the mapped `.olean` files, which
leaves about 0.7 GB, or 5e4 operations, for kernel work under a 4 GB
limit; and `List.Nodup` on 360 natural-number literals (64,620
comparisons) takes 5.2 s and 1.1 GB in the kernel, so about 1 KB per
elementary kernel step on top of the arithmetic.

Consequences for section 2. The kernel route (a) is not limited by time
but by memory: the unit is a theorem of at most about 2.5e5 operations in
a module that imports nothing, with the soundness link to `stabRankP`
proved once in a Mathlib module that only restates the computed facts
(the pattern of `QutritDict2Keys.lean` and `QutritDict2.lean`). Family 1
is 160 to 2000 such modules per cell; S^3 and N^3 are reasonable, H3^3
and the qubit m=4 cells are heavy, and nothing in families 2 and 3
except `qubit_T^5` is in reach of the kernel. `native_decide` covers
family 1 and the smaller family-2 cells at once and the T3 two-pivot scan
in hours; the three-pivot scan and H^6 need precompiled code and hundreds
to thousands of CPU-hours either way.

Dictionary completeness, first pieces (`Stabilizer/Reparam.lean`,
`QutritDict2.lean`, `QutritDict2Keys.lean`). For every prime p, every n
and k: `stabVecP_reparam`, an affine bijection `y = b + A z` of the flat
coordinates turns `stabVecP x0 W Q l` into `zeta^C` times the `stabVecP`
with base point `x0 + W^T b`, generators `A^T W`, and phase data from
`zeta_pow_quadPhaseP_comp`, injectivity carried along; and
`stabVecP_full_normal`, every full-support term (k = n) is `zeta^C
zeta^(Q'(x) + l'.x)` with `x0 = 0`, `W = I`, by choosing the preimages
of 0 and of the unit vectors. `tableOfP` is the exponent table of a
term with pivot columns and `stabVecP_eq_tableVal` ties it to the
amplitudes. At two qutrits, `isStabP_two_qutrits`: every `IsStabP 3`
vector is a nonzero multiple of one of the 360 tables of `dict2` (9
points, 108 lines with W in reduced row echelon form and x0 zero on the
pivot column, 243 full-support states with Q upper triangular), by a
case split on k <= 2 with the explicit 1 x 1 change of coordinates for
lines and a symbolic folding of Q for full support; `dict2_nodup`, the
360 tables are pairwise distinct, so the list is `dictionary(3, 2)` of
`rank_exclusion.py` (`3^2 (3 + 1)(3^2 + 1) = 360`). The module builds in
5 s with a 3.4 GB peak (3.3 GB of it the import baseline). Remaining
gaps: the reduced row echelon form for 0 < k < n at general n (the three
qutrit dictionary, 30240 states, needs k = 1 and k = 2 at n = 3), and
the statement that two distinct tables are not scalar multiples of one
another, which the normalisation of the exponent at the base point makes
true but which is not proved.

Covering lemma (`Stabilizer/Covering.lean`). `exists_decomp_mem_of_symm`:
for a predicate closed under nonzero rescaling and a linear automorphism
g preserving it with `g psi = c psi`, `c != 0`, a decomposition of size
at most r containing s gives one of size at most r containing s'
whenever `g s = a s'`, `a != 0`. The per-generator hypotheses (that the
monomial Cliffords and copy permutations preserve `IsStabP`, by
relabelling `x0`, `W` and reparametrising the phase as in `SliceP.lean`)
and the semilinear form for the antiunitary generator are not proved.

## Numbers

Measured this session (`verify_challenge/rank_exclusion.py` and
`slice_lift.py` at low priority on an idle 18-core laptop; the group
orders count the antiunitary element, "unitary" excludes it, "monomial"
keeps only the diagonal and permutation local Cliffords):

| cell | N | full group, pivots | unitary group, pivots | monomial, pivots | partner steps (rank-3 sieve) | rank-4 enumeration pairs, steps |
|------|---|--------------------|-----------------------|------------------|------------------------------|---------------------------------|
| S^3 | 30240 | 165888, 12 | 82944, 12 | 2592, 74 | 362,868 | 9,204, 1.82e8 |
| N^3 | 30240 | 2592, 74 | 1296, 88 | 96, 563 | 2,237,686 | |
| H3^3 | 30240 | 768, 116 | 384, 170 | 96, 563 | 3,507,724 | |
| qubit_H^4 | 36720 | 768, 186 | 384, 246 | 48, 1548 | 6,829,734 | 1,929,617, 2.79e10 |
| qubit_T^4 | 36720 | 3888, 62 | 1944, 76 | 48, 1324 | 2,276,578 | 404,659, 6.59e9 |
| qubit_H^3 | 1080 | 96, 40 | 48, 48 | 12, 185 | 43,160 | 14,328, 6.07e6 |

Rank-3 exclusion candidates at m=3: 1,404 (S), 8,658 (N), 13,572 (H3),
all dependent triples, margins 0.063, 0.048, 0.047. From the stored
batch records: T3^3 rank 7, 459 batches, 13,125,291,486,703 inner
steps, 1,213,458,815 class sets decided, 0 spurious, 288,520
CPU-seconds; H^6 rank 5, 190 batches, 5,965,707 covers, 23,862,828
matched runs, 834,686,307 cover-kernel candidates, 0 hits, 264,249
CPU-seconds. From `research/t3_rank7/results/step_counts.json`: 326,368
pivot pairs in the certificate order, 171,832 in orbit-block order.
