# Compiled kernels for the next exclusions: k = 6 covers, p = 3 stage A, the dense family solve

Status (2026-09-24). Engineering only; nothing here is a bound. The three
kernels that `docs/notes/next_exclusion_feasibility_2.md` (section 5.5)
named as the blockers of the rank-6 exclusion of |N>^4 are written,
bound into `stabrank_core`, and tested against the Python references,
which stay the oracle:

1. `cover6_pair` (`cpp/src/cover6.cpp`): the full 6-covers of a target
   through a pivot pair, the k = 6 analogue of `cover5_pair`, with a
   32-bit projective key.
2. `SliceMatch3Kernel` (`cpp/src/slice_match3.cpp`): stage A of the
   qutrit matcher `research/qutrit_m4_rank5/matcher.py` for a two-qutrit
   base slice of distinct independent states.
3. `dense_solve` (`cpp/src/dense_solve.cpp`): the Laplace-feature dense
   solve of `verify_challenge/slice_cover._dense` for a coefficient family
   with one or two parameters, with the whole-vector re-decision folded
   in; it serves the stage B path of every slice pipeline (H^5, T^5, the
   qutrit cells) through `slice_cover.solve_slice` and
   `matcher.solve_slice3`.

`cover5_pair` gains a `key_functionals` option (a 32-bit key); its default
is the 16-bit key of the H^6, H^5, and T^5 runs, and its outputs are
unchanged (section 6). Every kernel decides modulo 65521 and 2013265921
and returns a superset of the exact answer; the Python side re-decides
exactly, as before. Measurements ran single-process at nice 19 on the
18-core laptop under load from other sessions (load average 6 to 8),
through `research/t5_rank5/run.py` with a 600 s cap, about 1.5 CPU-hours
in all; the records are under `research/kernels/results/`, the scripts are
`research/kernels/bench.py`, `run_bench.sh`, `reproduce_hashes.sh`, and
`compare_hashes.py`.

Notation follows the two feasibility notes: a full k-cover is a multiset
of k dictionary states whose span contains the target with a coefficient
assignment in which every coefficient is nonzero; kappa is the dimension
of the coefficient family; M is the number of members above the partner
with a nonzero image modulo span(target, u_i, u_j); P1 = 65521 and P2 =
2013265921.

## 1. The k = 6 cover kernel

What it computes. `cover6_pair(Q, U1, psi1, U2, psi2, i, j, members,
max_run, seed)` lists every 6-set {i, j, k, l, a, b} with k < l < a < b
among the admissible members above the partner j whose span contains the
target modulo P2, as `pair_covers6_reference` in `slice_cover.py` does
(the port of the probe's `pair_covers6`): the dictionary is reduced modulo
span(target, u_i), then u_j, the members with a nonzero image are kept,
and for each third pivot k the rows above k are reduced by row k; for
each fourth pivot l the residues of the rows above l modulo row l are
normalized projectively (first nonzero entry 1) and keyed; a pair (a, b)
with equal keys is a candidate, decided by the reduced row echelon form of
[u_i, ..., u_b | target] mod P2 (span) and the fullness test of
`CoverEnumerator.is_full` mod both primes. The result carries the sorted
6-set, the two fullness flags, and the rank mod P2. The Python wrapper
`pair_covers6_native` keeps a 6-set full modulo both primes and decides a
6-set full modulo exactly one of them numerically, as the 5-cover wrapper
does. `CoverEnumerator.pair_covers(6, ...)` and
`CoverEnumerator3.pair_covers(6, ...)` dispatch to the kernel when
`stabrank_core` provides it and `STABRANK_NO_NATIVE` is not set;
`covers(6)` runs the whole enumeration.

Key width and accident rate. The key is two random functionals mod P1
applied to the normalized residue, packed as h_1 P1 + h_2 (32 bits), so
two non-parallel residues collide with probability 1/P1^2 = 2.3e-10, that
is, about M^2 / (2 x 65521^2) accidental candidates per (k, l) against
M^2 / (2 x 65521) for the 16-bit key of `cover5_pair`. The keys of one
fourth pivot go into an open-addressing table with stamps (O(M) per l
instead of a sort), and the parallel classes come out of its chains. On
every pair measured the kernel and the reference (which uses the same
two-functional key) report the same candidate count, and the counts equal
the probe's (`next_exclusion_feasibility_2.md`, section 2.2 and 5.3), so
neither key produced an accident on these pairs.

Rates (`results/cover6_qubit_H_n3_reference.json`,
`cover6_qubit_H_n3_heavy.json`, `cover6_qubit_T_n3_reference.json`,
`cover6_N_n2_reference.json`, `cover6_N_n2_heavy.json`):

| dictionary | pair (i, j) | M | full 6-covers | candidates | kernel | Python reference |
|---|---|---|---|---|---|---|
| H^3 (1080) | (3, 932) | 147 | 2,628 | 90,970 | 0.06 s | 7.3 s |
| H^3 | (811, 924) | 148 | 19 | 172,802 | 0.08 s | 13.1 s |
| H^3 | (582, 1025) | 54 | 0 | 3,363 | 0.00 s | 0.3 s |
| H^3 | (3, 706) | 373 | 206,294 | 3,532,228 | 2.0 s | 66.6 s (probe, batched numerics) |
| H^3 | (356, 540) | 535 | 40,789 | 25,903,049 | 9.1 s | 256 s (probe) |
| H^3 | (3, 354) | 725 | 2,415,058 | 38,418,570 | 19.6 s | not finished in 600 s (probe) |
| T^3 (1080) | (1019, 717) | 220 | 356 | 981,792 | 0.44 s | 72.2 s |
| N^2 (360) | (297, 111) | 166 | 11,698 | 216,427 | 0.16 s | 25.7 s |
| N^2 | (117, 0) | 358 | 753,064 | 5,622,158 | 4.8 s | 417 s for 24 pairs (probe sample) |
| N^2 | (117, 2) | 356 | 819,910 | 5,892,868 | 5.3 s | |
| N^2 | (279, 0) | 357 | 45,446 | 3,908,796 | 2.0 s | |
| N^2 | (281, 150) | 205 | 17,004 | 460,394 | 0.32 s | |

The cover sets agree with the reference on every pair where the reference
ran (4 pairs at H^3, 2 at T^3, 6 at N^2, the light ones), and the cover
counts of the heavy pairs equal the probe's. The kernel is 100 to 170
times the per-candidate Python reference and about 30 times the probe's
batched numerics; its time is roughly 5e-8 s x M^3 at H^3 (D_3 = 3
residue coordinates) and 1.0e-7 s x M^3 at N^2 (D_3 = 4, more candidates
per pair), the candidate decisions (an 8 x 7 or 9 x 7 echelon form mod
P2, about 0.5 microseconds each) taking the larger share on the heavy
pairs.

## 2. The p = 3 stage A kernel

What it computes. `SliceMatch3Kernel(codes, target1, target2, seed)` holds
the n_2-qutrit dictionary as phase codes (0 zero, 1..3 = 1, w, w^2) and
the target's nine slices along the two sliced qutrits over F_P1 and F_P2
(row 3 x_1 + x_2). `run(cover, x0)` is `Matcher.run(cover, x0, target)`
for a cover of distinct states whose coefficient family at x_0 is a point
(status 2 otherwise, status 1 when the target is outside the span or a
coefficient vanishes modulo a prime; the Python matcher then runs and
decides): the options of a term at a slice are the 3^{n_2} Pauli classes
of its base state times the three cube roots (code 3 k + l) plus absent,
built as `TermOpts` builds them (the same class order, the same class and
phase tables of Q_{k_2}^{x_2} Q_{k_1}^{x_1} u); the two coordinate slices
x_0 + e_1, x_0 + e_2 are solved by meet in the middle on a random
functional mod P1 with every collision decided on the whole slice mod P1
and mod P2, and joined (every pair of solutions, the family being a
point); the absence pattern fixes the shapes of the structure lemma
(`TermOpts.composite_rows`: a plane with 27 quadratics, a coordinate line
with 3 phases at its third point, a point or one of the two diagonal
lines with 1 + 2 x 3^{n_2} x 9 shapes); the six composite points are
solved one at a time in the matcher's order over the codes of the shapes
still alive, each solution restricting the shapes. Every hit comes back as
the r terms' phase codes over the nine slices and is confirmed in Python
(`Matcher.confirm`: residual, rank, span mod P2). The reference's stats
conventions are reproduced, including `coord_solutions` left empty when a
raw coordinate solve is empty (the batch hash key) and `coord_raw` holding
the raw counts.

Key width. The meet in the middle hashes on one functional mod P1 (16
bits) and decides every collision on the whole vector, so the key width
only sets the number of whole-vector checks (about n_A n_B / 65521 per
slice, a few hundred at r = 6 with 28 options per term), never the result.

Rates (`results/stage_a3_N.json`, `stage_a3_H3.json`; 200 full 5-covers
of |M>^2 from the first pivot pairs of the census, both base points, the
reference `Matcher(native=False)` with a warm option cache):

| orbit | x_0 | kernel per cover | reference per cover | coordinate-slice histogram |
|---|---|---|---|---|
| N | (0, 0) | 0.64 ms | 39.7 ms | (2, 4) 41 %, (8, 64) 16 %, (40, 1600) 6 %, (17, 289) 4 %, ... |
| N | (2, 2) | 0.10 ms | 1.8 ms | every run dies at the first slice |
| H3 | (1, 1) | 0.29 ms | 30.5 ms | |
| H3 | (0, 0) | 0.10 ms | 1.8 ms | every run dies at the first slice |

Hits (none), refusals, `coord_solutions`, `coord_raw`, and
`composite_solutions` agree on all 400 (cover, x_0) runs of each orbit.
The 0.088 s per 6-cover that the feasibility note measured at (2, 2)
included the option setup of six fresh states; with the cache warm the
reference is 1.8 ms and the kernel 0.1 ms, and in a batch the cache is
warm after the first few covers (360 states).

## 3. The compiled dense family solve

What it computes. `dense_solve(opts1, opts2, d01, K1, d02, K2, rhs1,
rhs2, sideL, sideR, max_cand, seed)` is `slice_cover._dense` for a
family d = d_0 + K lambda with kappa_1 = 1 or 2 parameters mod P1: the
(kappa_1 + 1) random functionals, the per-option matrices, the Laplace
expansion of the determinant of (u, v_1, ..., v_kappa) into the same
features of the two sides (6 for one parameter, 20 for two), and the
product over the two sides, in exact int64 arithmetic (features below
2^16, at most 20 products per pair, one reduction). Every feature zero is
then decided on the whole projected slice equation: u = sum_i d_{0i} w_i -
rhs in the span of v_b = sum_i K_{ib} w_i mod P1 (an echelon form of dim x
(kappa_1 + 1)), and again mod P2 with the family's data there (d_{02},
K_2). Only the survivors are returned; `solve_slice` and `solve_slice3`
pass them to `Family.restrict` (mod P1, over C, mod P2) exactly as they
pass the reference's candidates, so the solutions, the restricted
families, and everything downstream are the reference's. The feature-zero
count (the reference's candidate count) is reported as `dense_raw` in the
stats, and more than `max_cand` of them raise as the reference does. The
reference's candidate list is a superset of the exact solutions because a
dependency over the number field reduces to one mod every prime; the
kernel's whole-vector checks are the first two steps of `restrict`, so
its output is the same superset after those steps.

Where the time went. On a stage B cover (five distinct dependent states,
kappa = 1, 33 options per term, sides of 33^3 and 33^2) the reference
spends its 2 to 5 s in three places: the numpy feature product over
3.9e7 pairs in float64 chunks, the `fmod` and `nonzero` over the same
array, and the Python `restrict` of every one of the 1e5 to 1e6 raw
candidates, most of them structural zeros (a combination with v = 0 mod
P1, which the determinant cannot see) or the 1/65521 accidents. The kernel
does the product and the two whole-vector checks in place, so `restrict`
runs only on genuine modular solutions.

Rates (`results/dense_qubit_H.json`, `dense_qubit_T.json`; 8 dependent
5-covers evenly spaced through each list, `SliceMatcher(E, 2)` at x_0 =
00, the H^5 and T^5 stage B shape; the reference run with
`STABRANK_NO_NATIVE=1`):

| orbit | list | kernel per cover | reference per cover | agreement |
|---|---|---|---|---|
| qubit_H | `research/h6_rank5/degenerate_covers_v2.json` (12,390 dependent) | 0.13 s | 2.00 s | 8 of 8: hits, coordinate and composite counts, joined states |
| qubit_T | `research/t5_rank5/degenerate5.json` (20,653 dependent) | 0.18 s | 2.60 s | 8 of 8 |

The per-cover gain is 14 to 15 times on this shape. A cover whose first
coordinate slice has many solutions (T's (33, 48, 158, 174, 1006) with
309 then 343) keeps 0.36 s against 4.9 s: the survivors' `restrict` and
the `_mitm` of the pinned states stay in Python.

B6 at N, the 5-cover-plus-span-state bases the feasibility note costed
at 18 s each (`results/dense3_N_B6.json`; the bases built here from the
first full 5-covers of the census enumeration and the states of their
spans, six terms with 28 options each, sides of 28^3 and 28^3, so 4.8e8
feature products per slice; the qutrit matcher's `solve_slice3` through
`dense_solve`):

| x_0 | bases | kernel per cover | reference per cover | agreement |
|---|---|---|---|---|
| (2, 2) | 6 | 1.24 s (1.1 to 1.4) | 26.4 s (13.5 to 33.3) | 6 of 6: every run dies at the first coordinate slice in both |
| (0, 0) | 4 | 5.1 s (3.8 to 6.4) | 68.9 s | 4 of 4: raw solutions (1,106 or 3,102) and joined counts (14,394 or 38,016) equal |

At (2, 2) the whole run is the two dense products of the coordinate
slices (the second is solved against the initial family as well before
the join), 1.2 s for 9.6e8 exact int64 feature products, a factor of 21
on the reference. At (0, 0) the thousands of genuine solutions each go
through `restrict` in Python, which is now the larger share.

## 4. The 16-bit key of `cover5_pair`

`Cover5Inputs.key_functionals` (binding `key_functionals`, enumerator
`wide_key=True`) draws a second functional after the first and packs the
key as h_1 P1 + h_2; with the default 1 the code path, the random draws,
and the outputs are those of the H^6, H^5, and T^5 runs (section 6). The
covers are the same either way, since every candidate is decided exactly;
only the candidate count changes. Measured on the |H>^4 census pairs the
feasibility note timed (`results/cover5_key_qubit_H_n4.json`; 36,720
states; the members above the partner number about 14,000 and 26,000):

| pair (i, j) | 16-bit key: candidates, seconds | 32-bit key: candidates, seconds | covers |
|---|---|---|---|
| (215, 22355) | 9,052,099, 15.8 s | 1,646,817, 13.1 s | 0 |
| (541, 10359) | 55,245,349, 61.3 s | 9,457,457, 45.0 s | 0 |

The wider key removes about 80 % of the candidates and 20 to 25 % of the
time: on these pairs the cost floor is the M^2 residue-and-sort work per
third pivot, not the accidental decisions, which is what the note's own
second estimate said ("leaves the hashing at about M^2 log M per pair, of
order 1e3 CPU-hours over 1.9 million pairs"). The 1.65 million candidates
left with the 32-bit key are almost all structural (parallel classes of
the four-qubit dictionary modulo the span), not accidents (about 100
expected). At H^3 (`results/cover5_key_qubit_H_n3.json`) the two keys
differ by about 2 % of the candidates and by nothing measurable in time.

## 5. Tests

C++ (Catch2, `cpp/tests/`, run by `ctest` or `build/cpp/stabrank_tests`):

- `test_cover6.cpp`: a planted instance of 50 states in dimension 9 with a
  planted full 6-cover, a dependent 6-set, a 6-set with a dead
  coefficient, and a 2-cover; the kernel's full covers equal a brute
  force over every 6-set through the pivot pair for five pivot pairs
  including i > j; every reported set spans the target mod P2 with the
  right flags and rank; the partner order and the member mask are
  honored; a dimension with no residue coordinates left gives nothing.
- `test_slice_match3.cpp`: the two-qutrit dictionary generated in the test
  has 360 states and the field's cube root is primitive; planted five-term
  and six-term decompositions of random four-qutrit stabilizer states
  (random flats, x_0 random) are recovered from their base slice, every
  hit satisfies the target equation mod P2 with the returned coefficients;
  a wrong base state gives no planted hit; a repeated state is rejected.
- `test_dense_solve.cpp`: integer instances with a planted solution at
  kappa = 1 and kappa = 2, one-term sides, against a brute force over
  every combination decided mod both primes; the candidate cap raises.

Python (`tests/test_kernels_k6_p3.py`, skipped when `stabrank_core` lacks
the kernels):

- `cover6_pair` against `pair_covers6_reference` on H^3 pairs, a T^3 pair,
  and the lightest N^2 pairs through both enumerators' `pair_covers(6)`,
  with equal cover sets and candidate counts; planted six-term targets of
  random three-qubit states recovered from their two smallest members.
- `cover5_pair` with the 32-bit key lists the same covers as the 16-bit
  key with no more candidates.
- `SliceMatch3Kernel` against `Matcher` on planted five-term and six-term
  qutrit decompositions (every shape of the structure lemma) and on 40
  census covers at (0, 0) and (2, 2): hits, refusals, `coord_solutions`,
  `coord_raw`, `composite_solutions`; dependent and repeated bases go to
  the reference path.
- `dense_solve` inside `SliceMatcher(E, 2)` on stage B covers of the H and
  T lists and on planted decompositions over a dependent base (five plane
  terms whose base slices are a kappa = 1 cover, Gaussian-integer
  coefficients), against the reference with `STABRANK_NO_NATIVE=1`.

Also run: the existing `tests/test_slice_cover.py` and
`tests/test_qutrit_m4_rank5.py`, whose native-kernel and matcher controls
exercise the changed modules (`slice_cover.solve_slice` now routes
kappa = 1 and 2 through `dense_solve`; `matcher.Matcher` runs the compiled
stage A by default). Both pass unchanged, with the new suite 13 tests in
about 7 minutes, most of it the Python references.

## 6. The stored hashes

`research/kernels/reproduce_hashes.sh` re-runs three stored batches
through the rebuilt extension into a scratch directory and
`compare_hashes.py` compares the `deterministic_sha256` of each record
with the stored one (`results/hash_check.json`):

| batch | what it exercises | stored record | result |
|---|---|---|---|
| T^5 stage A batch 4 (308,655 covers through `cover5_pair` and `SliceMatchKernel`) | the two existing kernels after the refactor of `cover5.cpp` onto `cpp/src/modular_detail.hpp` and the `key_functionals` field | `research/t5_rank5/results/batch_4.json` (pod, 2026-09-24) | equal (69be84ae6f12908f); kernel 59 s and match 48 s here against 101 s and 133 s on the loaded pod |
| T^5 stage B batch 11 (150 dependent covers through the family path) | `dense_solve` inside `solve_slice` | `research/t5_rank5/results/batch_11.json` | equal (88a0061ad1625956); match 23 s here against 733 s on the pod, 0.15 s per cover against 4.9 s |
| N^4 rank 5 stage A batch 0 (18,215 covers) | `SliceMatch3Kernel` inside `Matcher.run` against the reference matcher's record | `research/qutrit_m4_rank5/results/N/batch_0.json` (pod, 2026-09-23, `matcher: reference`) | equal (a67063c614687cdd); match 8 s here against 964 s on the pod, every cover through the kernel |

All three deterministic hashes reproduce. The stage A batch confirms that
the refactor left `cover5_pair` and `SliceMatchKernel` bit-identical on
308,655 real covers (the record includes the coordinate-slice solution
histogram and would change on any difference in a hit or a count); the
stage B batch confirms that the compiled dense solve leaves the family
path's results unchanged on 150 real dependent covers; the N batch
confirms the p = 3 kernel against the reference matcher on 18,215 real
covers with the batch's own hash key (the `coord_solutions` histogram).
The `candidates` and `native_runs` fields, which do change, are outside
the deterministic part by design.

## 7. Re-projected costs

Laptop rates as measured here (single process, under load); the pod ran
the compiled stages at about 1.3 times the laptop rate and the Python
stages at about 4 times in the T^5 run.

### 7.1 |N>^4 rank 6 (`next_exclusion_feasibility_2.md`, section 5)

| stage | bases | old rate (Python) | old CPU-hours | new rate | new CPU-hours |
|---|---|---|---|---|---|
| (alpha) census of full 6-covers of |N>^2 | 1,209 pivot pairs, 3.2e7 to 3.8e7 covers | about 4 CPU-hours | 4 | 1.0e-7 s x M^3 (sum of M^3 over the pairs 8.9e9) | about 0.3 |
| (alpha) A6 matching at (2, 2) | 3.2e7 to 3.8e7 | 0.088 s (cold cache; 1.8 ms warm) | 800 to 930 | 0.10 ms | 0.9 to 1.1 |
| (alpha) B6, a 5-cover plus a span state, kappa = 1 | about 2e6 | 18 s | about 10,000 | section 7.3 | |
| (alpha) C6, a 5-cover plus a member (block path, Python) | about 1.5e6 | 0.12 s | 50 | unchanged | 50 |
| (beta') one invisible term, 16 flats | 225,796 x 16 | no p = 3 invisible matcher | | not written | |
| (gamma) two invisible terms | about 1,500 x 136 | | | not written | |

The census and the A6 matching drop from about 900 laptop CPU-hours to
about 1.5. B6 stays the dominant stage (section 7.3), and the invisible-
flat matcher at p = 3 is still unwritten.

### 7.2 |H>^7 rank 6 (`next_exclusion_feasibility_2.md`, section 2)

The 4 + 3 route's census, the full 6-covers of |H>^3 over the 14,280
pivot pairs of the H^6 census (sum of M^3 over the pairs 2.44e12, with M
the members above the partner), projects to 2.44e12 x 5e-8 s, about 34
laptop CPU-hours with `cover6_pair`, against about 1,100 in Python; the
count stays 1.6e9 to 1.9e9 bases. The matching does not exist for that
route: `SliceMatchKernel` accepts n_1 at most 3, the route needs n_1 = 4
(16 points, up to three invisible terms on the five 8-point flats missing
0000), and at the qubit kernel's 0.3 to 0.6 ms per base a hypothetical
n_1 = 4 stage A alone would be 1.7e9 x 0.5 ms, about 240 CPU-hours,
before the invisible-term stages. The 3 + 4 route's 5-cover census of
|H>^4 (1.9 million pivot pairs) improves by the 20 % of section 4 only:
the M^2 residue work per third pivot at M near 14,000 to 26,000 is the
floor, and a 6-cover kernel over the 36,720 states would multiply it by M.
H^7 rank 6 stays out of reach; what changed is that its 4 + 3 census is
now a day of laptop time rather than six weeks.

### 7.3 B6 with the compiled dense solve

About 2e6 B6 bases at 1.24 s each are about 700 laptop CPU-hours, against
the 10,000 the note projected at 18 s; on the pod's Python factor of 4
that is about 2,800 pod CPU-hours, still far above the 100 the note asked
for. The remaining time is the dense product itself: two slices of 4.8e8
pairs each at about 1.3 ns per pair. Getting B6 to the target needs the
different treatment the note named rather than a faster product: a base
that is a full 5-cover plus a state of its span has a unique coefficient
vector on the five once the sixth's coefficient is fixed, so the one
parameter can be carried through the first exact point as an unknown
with a hashed meet in the middle over the five (about 28^3 x 28^2 pairs
against 4.8e8), or the 2e6 bases can be cut by the G_2 orbits of the
6-sets, which the route from the census does not yet use (the 197,440
5-covers are one per orbit, but the span states are not reduced). Either
is a design change beyond this note. With B6 at 700 CPU-hours the whole
N^4 rank 6 exclusion projects to about 750 laptop CPU-hours plus the
unwritten stages (beta') and (gamma), against about 11,000 before.

Summary of the re-projection (laptop CPU-hours):

| stage | before | after |
|---|---|---|
| census of full 6-covers | 4 | 0.3 |
| A6 | 800 to 930 | 0.9 to 1.1 |
| B6 | about 10,000 | about 700 |
| C6 | 50 | 50 |
| (beta'), (gamma) | unwritten | unwritten |
| total of the written stages | about 11,000 | about 750 |

## 8. What was run

All through `research/t5_rank5/run.py` (nice 19, own session, 600 s cap),
one process at a time, on the laptop at load average 6 to 8 from other
sessions; about 1.5 CPU-hours of measurement plus the builds and the test
suites. The commands are `research/kernels/run_bench.sh` and
`reproduce_hashes.sh`.

| step | command | time | result |
|---|---|---|---|
| cover6 against the reference, light H^3 pairs | `bench.py cover6 qubit_H 3 --pairs 3,932 811,924 582,1025 1074,1043 --reference` | 21 s | 4 of 4 equal (`cover6_qubit_H_n3_reference.json`) |
| cover6, heavy H^3 pairs | `... --pairs 3,706 356,540 3,354 3,932 811,924` | 32 s | the probe's counts; (3, 354) in 19.6 s (`cover6_qubit_H_n3_heavy.json`) |
| cover6 at T^3 | `bench.py cover6 qubit_T 3 --pairs u:4671 u:4200 --reference` | 73 s | 2 of 2 equal; a third pair (817, 448) with M near 700 was killed at the cap in the reference (`cover6_qubit_T_n3_reference.json`) |
| cover6 at N^2, light pairs | `bench.py cover6 N 2 --pairs u:1208 u:1200 u:1190 u:1150 u:1100 u:600 --reference` | 27 s | 6 of 6 equal (`cover6_N_n2_reference.json`) |
| cover6 at N^2, heavy pairs | `... --pairs u:0 u:1 u:16 u:17 u:52 u:200` | 17 s | 4.8 to 5.3 s at M near 357 (`cover6_N_n2_heavy.json`) |
| p = 3 stage A at N | `bench.py stage-a3 N --count 200 --x0 0,0 2,2` | 20 s | 400 of 400 runs equal (`stage_a3_N.json`) |
| p = 3 stage A at H3 | `bench.py stage-a3 H3 --count 200 --x0 1,1 0,0` | 15 s | 400 of 400 equal (`stage_a3_H3.json`) |
| dense solve, H stage B | `bench.py dense qubit_H --count 8` | 18 s | 8 of 8 equal, 0.13 s against 2.00 s (`dense_qubit_H.json`) |
| dense solve, T stage B | `bench.py dense qubit_T --count 8` | 24 s | 8 of 8 equal, 0.18 s against 2.60 s (`dense_qubit_T.json`) |
| dense solve, N B6 | `bench.py dense3 N --count 6 --x0 2,2`, `--count 4 --x0 0,0` | 170 s, 300 s | 10 of 10 equal (`dense3_N_B6.json`) |
| cover5 key widths, H^3 | `bench.py cover5-key qubit_H 3 --pairs 3,706 356,540 3,354 u:0 u:1000` | 2 s | same covers (`cover5_key_qubit_H_n3.json`) |
| cover5 key widths, H^4 | `bench.py cover5-key qubit_H 4 --pairs 215,22355 541,10359` | 135 s | same covers, 80 % fewer candidates, 20 to 25 % less time (`cover5_key_qubit_H_n4.json`) |
| hash reproduction | `reproduce_hashes.sh` | 107 s, 23 s, 17 s | three of three equal (`hash_check.json`) |
| C++ suite | `build/cpp/stabrank_tests` | 2 min | 65 test cases, three of them new |
| Python suites | `pytest tests/test_slice_cover.py tests/test_qutrit_m4_rank5.py tests/test_kernels_k6_p3.py` | 7 min | 34 passed |

Not done here, and named in the feasibility note as the third piece for
D: the p = 3 invisible-flat matcher (stages (beta') and (gamma) of the
N^4 rank 6 design). Not changed: the defaults of the certified pipelines
(`cover5_pair` keeps its 16-bit key unless `wide_key=True`; the T^5, H^5,
and H^6 batch runners are untouched); the qutrit `batch.py` records
`matcher: native` from now on, outside the deterministic part.
