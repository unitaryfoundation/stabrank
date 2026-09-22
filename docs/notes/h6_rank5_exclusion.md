# Excluding rank 5 for |H>^6: design note

Status (2026-09-21). Not run to completion, and not certifiable until it
is. The enumeration and matching machinery exists
(`verify_challenge/slice_cover.py`, `research/h6_rank5/driver.py`, with
the two hot paths compiled in `cpp/src/cover5.cpp` and
`cpp/src/slice_match.cpp`), both positive controls pass on every base
(section 5: the rank-4 decompositions of |H>^4 from the full 4-covers of
|H>^3, and the rank-6 witness from all four of its all-visible bases,
distinct and repeated), the 5-cover enumeration has been run to completion
(5,939,465 full covers of distinct independent states, section 3, and
26,242 dependent or repeated covers, section 4), and the whole exclusion is
partitioned into 190 batches with a runner, an aggregator and a
certificate on the attested tier (section 6): stage A is 2.4 CPU-hours
with the compiled kernels, the degenerate stages B and C about 30
CPU-hours in Python, about 36 CPU-hours in all, about 2.5 hours of wall
time on a 16-vCPU pod. One batch of each stage has been run as a test; the
full run has not been launched. The
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
| 5 | >= 1,239,946 after pivots 0 to 3 of 48 | not computed | 36,597,459 | 3227 s (killed; the compiled count is below) |

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

In Python the r = 5 count was out of reach: 36.6 million modular
candidates after four pivots, re-decided one by one at about 90
microseconds each. The same search compiled (`cpp/src/cover5.cpp`, bound
as `stabrank_core.cover5_pair` and dispatched from
`CoverEnumerator.pair_covers` unless `STABRANK_NO_NATIVE` is set) decides
a candidate by row reduction mod 2013265921 in about half a microsecond:
the span condition by the consistency of the 8 x 6 system, the fullness by
the dead-coordinate test on its reduced form, the latter mod both primes
(a cover full modulo exactly one prime, a modular accident that never
occurred, would be decided numerically in Python). At M near 1000 a pivot
pair costs 0.2 s (0.21 s (M/1000)^2 fitted over 14 pairs), against 44 s in
Python; the cover sets agree pair by pair (`tests/test_slice_cover.py`,
and `cpp/tests/test_cover5.cpp` against a brute force on a planted
instance). The complete census (`driver.py census`, 459 s on one core,
`results/kernel_census.json`):

| r | full covers (tuples) | candidates  | time  |
|---|----------------------|-------------|-------|
| 5 | 5,939,465            | 835,507,077 | 459 s |

The count is one tuple per G_3 orbit as far as the pivot and partner
reductions go (pivot one per orbit, partner minimal in its stabilizer
orbit, members in orbits at or above the pivot's); the residual symmetry
among the last three members is not quotiented.

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

Measured cost in Python (`driver.py sample --count 40 --reference`, one
core at nice 19 on a machine with load average 12 to 18 on 18 cores): 40
full 5-covers from the first pivot, four base points each, 160 (cover,
x_0) runs, no hit. Per cover (all four base points) mean 0.110 s, median
0.019 s, maximum 0.94 s. The first coordinate slice has no solution in 146
of the 160 runs; the rest have solution counts (1, 1, 1) on the three
coordinate slices (6 runs), (2, 4, 8) (4 runs) and (9, 81, 729) (4 runs,
the expensive ones).

The compiled matcher (`cpp/src/slice_match.cpp`, bound as
`stabrank_core.SliceMatchKernel`, dispatched from `SliceMatcher.run` for a
base of distinct states unless `STABRANK_NO_NATIVE` is set) covers stage
A: the coefficient family is a
point, so each coordinate slice is solved once by meet in the middle on a
fixed random functional mod 65521 (two terms hashed, three probed, every
collision decided on the whole 8-vector mod 65521 and then mod
2013265921), the joined states are the product of the three solution
lists, and the composite slices are solved point by point over the
structure lemma's code sets with the row consistency of every term
enforced depth first. It declines (status 2) when the base states are
dependent modulo either prime, and Python runs the reference path; the
hits come back as phase codes and are confirmed in floating point and mod
2013265921 by the same `confirm` as the reference. On the same 160 runs
(`driver.py sample --count 40`, `results/sample_native.json`): mean 1.4 ms
per cover, median 0.8 ms, maximum 7 ms, the same solution counts run by
run and the same (empty) hit set, a factor 79 on the mean and 130 on the
maximum. The agreement is tested three ways: `tests/test_slice_cover.py`
compares the two matchers on the 160 sample runs and on the rank-4 bases
of |H>^4 (where there are hits), and `cpp/tests/test_slice_match.cpp`
recovers planted five-term decompositions of a random target from their
base slice, checks every hit exactly mod 2013265921, and replays the 160
sample runs against the reference's per-run results
(`cpp/tests/data/h6_rank5_sample.txt`, written by `driver.py fixture`).

Degenerate covers (`driver.py degenerate --sample 12`, 187 s to
enumerate): 26,242 full 5-covers whose base states are dependent or
repeated, over the two 3-cover classes and the 3,460 4-cover classes
without further symmetry reduction: 12,390 with five distinct dependent
states (kappa = 1), 13,840 with one state repeated (3,460 x 4, kappa = 0),
6 with a triple, 6 with two pairs. Sampled per-cover cost in Python (the
compiled matcher declines these), twelve covers per pattern evenly spaced
through the list, four base points each, no hit
(`results/degenerate_sample.json`): the dependent covers 8.9 s mean (9.6
median, 12.5 maximum; every base point starts with a 1-parameter dense
solve of 39 million pairs), the single-repeat covers 0.82 s mean (0.02
median, 9.1 maximum), the two-pair covers 9.3 s mean over all six, the
triple covers 1.0 s mean over all six. Projected: stage B (dependent)
30.6 CPU-hours, stage C (repeated) 3.2 CPU-hours.

Projection for the full run. The 5-cover enumeration is done (459 s,
section 3). Stage A matching at 1.4 ms per cover over the 5,939,465 covers
is 2.3 CPU-hours; `partition.json` now groups the 14,280 pivot pairs by the
census's covers and seconds per pair (`driver.py partition`, cost per pair
= kernel seconds + 1.4 ms x covers) into 15 batches of about 600 s, 2.44
CPU-hours in all. Stages B and C add about 34 CPU-hours in the present
Python, dominated by the 12,390 dependent covers; a compiled 1-parameter
dense solve would cut that by roughly the same factor as stage A but has
not been written. So the whole exclusion is about 36 CPU-hours, of which
34 are the degenerate list, or about 2.5 hours of wall time on 16 vCPUs.

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
000 and at 111 (the triples inside qubits 0 to 4, kappa = 2), or five
distinct states of rank 4 with one repeated (the triples containing qubit
5, kappa = 1 on the distinct states plus the pair block). On the two
distinct bases the control passes: from
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

On the two repeated bases, (368, 242, 65, 536, 479, 242) at 111 and
(1035, 0, 619, 622, 20, 0) at 000, the control also passes
(`--repeated-only`, `results/control_witness_repeated.json`): 1062 s and
1067 s per base, each returning the witness itself and two further genuine
rank-6 decompositions of psi_6 with the same base slice (51 and 24 raw
hits, the rest of rank below 6 or with a zero coefficient, so no
decomposition is reported twice). The projected (span-of-translates)
treatment of the pair is weak on this witness because several ordinary
base states are Pauli translates of the repeated state, so the residuals
lie in the translate span for structural reasons; without the split
tracking of section 4 the first two coordinate slices gave 403 and 15,333
states and the run was stopped at 9 minutes. With it the same slices give
131 and 2,678 states (10,192 and 138,744 solutions), the third 1,862,920
solutions and 45,469 joined states, 315,696 states are dropped by the
split tracking and 1,744,173 by a dead coefficient, and the composite
stage over 49,689 flat-type selections runs 296 block reconstructions.

Gap (a), repeated base states, and gap (b), dependent base states, of the
earlier revision are implemented (blocks, coefficient family) and both
controls now pass on every base. Whether a rank-5 decomposition can have a
repeated base state at an all-visible base point was not settled: minimal
decompositions can in general (control 1 recovers rank-4 decompositions
of |H>^4 from the repeated bases (3, 3, 352, 912) and (75, 353, 749, 749)
along one qubit), so an argument would have to use property P, and none
was found; the repeated covers stay in the exclusion as stage C.

## 6. Run plan and certificate

Partition (`research/h6_rank5/partition.json`, written by `driver.py
partition`, hashed, 190 batches). Stage A: the 14,280 pivot pairs of the
5-cover enumeration grouped greedily into 15 batches of about 600 s (the
census's kernel seconds plus 1.4 ms per cover for the compiled matcher;
batches 0 to 14, the last one 301 s). Stage B: the 12,390 full 5-covers of
five distinct but dependent states (multiplicity pattern (1, 1, 1, 1, 1),
kappa = 1), assigned round-robin over the sorted list to 158 batches of 78
or 79 covers, about 700 s each at the sampled 8.9 s per cover (batches 15
to 172). Stage C: the 13,852 covers with a repeated state (13,840 of
pattern (2, 1, 1, 1), 6 of (2, 2, 1), 6 of (3, 1, 1)), round-robin over 17
batches of 814 or 815 covers (batches 173 to 189); the partition sizes
them at the sample mean of 0.82 s per cover, but that mean is carried by
rare 9 s covers (median 0.02 s) and the test batch ran at 0.17 s per cover,
so these batches take 2 to 4 minutes and stage C about 0.6 CPU-hours
rather than 3.2. The degenerate covers are enumerated once (`driver.py
degenerate --write`, 189 s, `degenerate_covers.json` with its hash) and
every stage B or C batch loads them by hash; the aggregate re-enumerates
them. Round robin rather than contiguous chunks because the cost of a
cover correlates with its 3-cover or 4-cover, which the sorted order
groups.

Batch (`batch.py K`, resumable, `--native`/`--no-native` with
`STABRANK_NO_NATIVE=1` honoured). A stage A batch runs the compiled 5-cover
kernel on its pivot pairs and the matcher on each cover at the four base
points; a stage B or C batch runs the reference matcher on its covers.
Every hit is re-decided exactly from the phase codes of its five terms
(psi_6 against their span mod 2013265921 and numerically,
`common.decide_terms`); a hit with psi_6 in the span is a decomposition
with at most five terms and the batch exits 2 with `DECOMPOSITION FOUND`;
a run that raises, or a hit on which the two decisions disagree, is
recorded under `undecided` and fails the batch. The record's deterministic
part (geometry, partition and degenerate-list hashes, counts, the
coordinate-slice solution histogram, the hits as phase codes with their
decisions, `undecided`) is hashed as `deterministic_sha256`, so a re-run on
another machine or with `--no-native` is compared bit for bit; timing,
host, git commit, matcher and version fields follow, then `sha256` over
the whole record.

Aggregate (`aggregate.py`). Checks the partition's hash, that its stage A
units are the enumerator's pivot pairs, the degenerate list against its
hash and a fresh enumeration, that the stage B and C cover ids tile the
list exactly once, every batch file with both hashes and its geometry,
counts and `undecided` list, every stored hit re-decided, and the stage A
cover total against the census; writes `batch_manifest.json` (one entry
per batch with parameters, output path and SHA-256); then re-runs
`--recheck N` batches from scratch chosen from `--recheck-seed` and
compares deterministic hashes. `--partial` reports over the batches
present; `--dry-run` skips the re-runs. It prints `CERTIFIED chi(qubit_H^6)
>= 6` only when every batch is present and clean and the re-runs match.

Certificate. `verify_challenge/cert_qubit_h_m6_rank5_attested.py` prints
`seed: 20260921`, runs the aggregate with two re-runs into a scratch
directory and requires its claim line; the draft bound
`research/h6_rank5/qubit_H-m6-lower-6.json.draft` declares `certificate.attested` on
`research/h6_rank5/batch_manifest.json` with `recomputed: 2` and a 3600 s
budget (the re-enumeration is about 190 s and two batches at most about
1400 s, so the certificate fits with margin on one core), and states the
argument's dependencies: PR #87's property P tables
(`research/constructions/two_qubit_slice.py`, 501 s, not re-run), the
all-visible slice lemma of section 2, the slice structure lemma, the
mod-65521 supersets with exact re-decision of every cover and hit, and the
two positive controls of section 5. Compute hours, hardware, date and the
`--no-native` cross-check are placeholders until the run; the file keeps
its `.draft` suffix until the manifest exists, because the submissions
workflow verifies every touched `bounds/*.json` and `check_attested` fails
on a missing manifest.

Test (2026-09-21, one core at nice 19 on an 18-core laptop at load
average about 10). Stage A batch 14 (7,795 pivot pairs, 134,822 covers,
539,288 matched runs): 220 s, kernel 120 s and match 99 s, 0 hits. Stage B
batch 15 (79 covers, 316 matched runs): 663 s, 8.4 s per cover, 0 hits.
Stage C batch 173 (815 covers, 3,260 matched runs): 135 s, 0.17 s per
cover, 0 hits. Nothing refused, nothing undecided. The aggregate under
`--partial` re-enumerated the degenerate covers in 185 s, found them equal
to the stored list, passed every stored check on the three batches, wrote
the partial manifest, and re-ran batch 14 from scratch in 213 s with the
same deterministic hash. Projected totals at the test rates: stage A 2.7
CPU-hours, stage B 28.9, stage C 0.6, about 32 CPU-hours in all (the
partition's estimate is 36.1).

What remains.

1. Run the 190 batches on a 16-vCPU pod (`research/h6_rank5/README.md`
   has the commands: `batch.py K` per batch, 15 at a time through xargs,
   about 2.5 hours of wall time), aggregate, and confirm that no hit is a
   decomposition.
2. Cross-check one stage A batch with `--no-native` (about 80 times slower
   than the 600 s native batch, so batch 14 at about 5 hours) and record
   that its deterministic hash matched; the compiled kernels' agreement
   with the Python reference is otherwise checked only on the sample, the
   controls and the planted instances of the C++ tests.
3. Fill the placeholders of the draft bound (compute hours, hardware,
   date, the cross-check), rename it to `bounds/qubit_H-m6-lower-6.json`,
   and run `verify_challenge/stabrank_verify.py` on it.
4. Optional, if stage B's 30 CPU-hours matter: compile the 1-parameter
   dense solve (the 2 x 2 Laplace features and the 39-million-pair product)
   the way stage A was compiled.

Everything above depends on PR #87's property P only through the lemma of
section 2 (an all-visible slice exists); the matcher itself does not use
P. The certificate declares the dependency rather than re-running the
tables.

## 7. The run (2026-09-22)

All 190 batches ran on the RunPod CPU pod between 01:37 and 08:31 UTC,
ten at a time at nice 19 on a shared 16-vCPU host (load 30 to 45), except
batches 14, 15, and 173, which were the pipeline test on the laptop the
day before and were kept by the resumable runner. Totals from the
aggregate: stage A 15 batches, 5,939,465 covers, 23,757,860 matched runs,
2.77 CPU-hours (1.7 ms per cover); stage B 158 batches, 12,390 covers,
49,560 matched runs, 69.64 CPU-hours (20.2 s per cover, about 2.3 times
the laptop sample rate under the shared host); stage C 17 batches, 13,852
covers, 55,408 matched runs, 1.00 CPU-hours. No refusal, no undecided
run, no hit. `aggregate.py --recheck 2 --recheck-seed 20260921` verified
every stored batch, re-enumerated the 26,242 degenerate covers (211 s,
equal to the stored list), wrote `batch_manifest.json`, and re-ran
batches 11 and 148 from scratch (439 s and 918 s) with matching
deterministic hashes, in 1,569 s total. Together with the QPG cat witness
this gives chi(H^6) = 6, filed as `bounds/qubit_H-m6-lower-6.json` at the
attested tier.
