# Excluding rank 4 for |T>^5: design note

Status (2026-09-24). Pipeline built, controlled, run and aggregated on
the laptop: every full 4-cover of |T>^3 matched at both base points, 0
hits, 0 refused, 0 undecided, `CERTIFIED chi(qubit_T^5) >= 5` printed by
the aggregate and by the certificate script; the draft bound
`research/t5_rank4/qubit_T-m5-lower-5.json.draft` waits for the
submissions workflow. The cell was 4 <= chi(T^5) <= 6
(`bounds/qubit_T-m5-lower-4.json`, the rank-3 exclusion by slice and lift;
`bounds/qubit_T-m5-upper-6.json`, the product witness at the Lean tier).
Excluding rank 4 gives chi(T^5) >= 5 and, by projection, chi(T^6) >= 5 (the
m = 6 cell holds 4 <= chi <= 6 from the projection of the m = 5 lower bound
and its own Lean witness). The design was sketched as candidate A' in
`docs/notes/next_exclusion_feasibility.md` (section 2.2); this note is the
argument as implemented in `research/t5_rank4/`, in the shape of
`docs/notes/h5_rank5_exclusion.md`, written so that it can be checked from
the files alone.

Notation. |T> = cos(beta)|0> + e^{i pi/4} sin(beta)|1> with cos(2 beta) =
1/sqrt 3 (the face state, `orbit_state("qubit_T")`), psi_m = |T>^m, and
tau = a_1 / a_0 = e^{i pi/4} tan(beta), tan(beta) = (sqrt 3 - 1) / sqrt 2,
the ratio between the amplitudes of |T>. N_3 = 1080 three-qubit stabilizer
states in the order of `dictionary(2, 3)`; G_3 the unitary symmetry group
of psi_3 (order 162: the order-3 Clifford stabilizer of |T> on each qubit
and the permutations of the three qubits; 20 orbits on the dictionary). A
rank-4 decomposition is psi_5 = sum_{i=1}^4 c_i s_i with stabilizer states
s_i and all c_i nonzero. Slicing along a set S of n_1 qubits at x in
F_2^{n_1}: u_i^(x) = (<x| (x) I) s_i and sum_i c_i u_i^(x) = alpha_x
psi_{5 - n_1} with alpha_x = a_0^{n_1 - |x|} a_1^{|x|}, never zero; the
ratio between the slices at x and x_0 is tau^{|x| - |x_0|}. Each term is
nonzero exactly on an affine flat of F_2^{n_1}, and the slice structure
lemma (`research/constructions/two_qubit_slice.py`, PR #87) describes the
slices of one term on its flat from the slice at one point: along each
basis direction a fourth root of unity times a Pauli on the unsliced
qubits, on composite points the class product with one quadratic sign per
basis pair. The lemma is a statement about stabilizer states and does not
depend on the target. Everything below runs at n_1 = 2 along qubits 1, 2;
by the copy symmetry the pair is general.

Arithmetic. The amplitudes of psi_m divided by cos(beta)^m lie in
Q(zeta_24) = Q(i, sqrt 2, sqrt 3): the entry at x is tau^{|x|}. Both
primes of `verify_challenge/slice_cover.py`, 65521 and 2013265921, are
1 mod 24, so the same modular machinery serves with a Q(zeta_24) field
(`Field(p, "qubit_T")`): zeta_8 is the same primitive eighth root as in
the H field (so i, and with it every phase code of every stabilizer state,
is the same element of F_p in both fields and in the compiled kernels),
sqrt 2 = zeta_8 + zeta_8^{-1}, and sqrt 3 = -i (w - w^2) for a primitive
cube root of unity w. Any choice of w is a ring homomorphism Q(zeta_24) ->
F_p, so a dependency over Q(zeta_24) reduces to one mod p and the modular
kernels list supersets, as at H; the exact decision mod 2013265921 is
cross-checked numerically at every step. Scaling the target by
cos(beta)^{-m} changes no span, cover or slice equation. The qubit_H path
of `slice_cover.py` is unchanged by the parameter (`tests/test_slice_cover.py`
runs the H tests as before and adds the T ones: the rank-3 decompositions
of |T>^3 and |T>^4 recovered from their one-qubit slices through the
reference and the compiled kernel, the 2-covers and 3-covers against the
stored lists up to symmetry, and kernel against reference on 4-covers of
|T>^3 at n_1 = 2).

## 1. The two facts and how each is established

Fact 1 (every term is full along every qubit). Slice along one qubit at
value k in {0, 1}. The visible terms decompose alpha_k psi_4, so there are
at least chi(T^4) = 3 of them (`bounds/qubit_T-m4-lower-3.json`). If
exactly three are visible at k, they are distinct and independent
(otherwise psi_4 lies in the span of at most two stabilizer states), so
they are the unique rank-3 decomposition of |T>^4 up to its unitary
symmetry (`research/constructions/data/qubit_T_m4_rank3.json`, one class;
the symmetry acts on the unsliced qubits and preserves slices), and the
fourth term is |k'> (x) v with v a four-qubit stabilizer state. The
equation at k' says that alpha_{k'} psi_4 minus the three visible terms'
slices at k' (Pauli translates of their slices at k, or zero) is a nonzero
multiple of a stabilizer state: a stabilizer-residual row of PR #87's
tables at the ratio tau^{+-1}. Those tables, over all 65^3 code
combinations of the three terms (`research/t5_rank4/tables.py`, the
committed `residual_table` of `two_qubit_slice.py` run with the T
target and ratio; `results/tables.json`, 4 s):

| ratio tau^j | exact | stabilizer |
|---|---|---|
| j = -2, -1, 1, 2 | 0 | 0 |
| j = 0 | 1 | 192 |

So no one-qubit slice has exactly three visible terms, every term is
nonzero at both values of every qubit, and no term is a product across
any qubit. Fact 1 rests on the completeness of the one-element list (the
dependency `qubit_T-m5-lower-4.json` already declares) and on PR #87's
shape enumeration, which has its own controls.

Fact 2 (at most one absent term at any two-qubit point). The visible
terms at a point of F_2^2 decompose a nonzero multiple of psi_3, and
chi(T^3) = 3 (`bounds/qubit_T-m3-lower-3.json`).

## 2. The case split

Flat lemma. By Fact 1 the flat of every term along qubits 1, 2 projects
onto both coordinates, so it is the plane F_2^2 or one of the two diagonal
lines A = {00, 11} and B = {01, 10}; the coordinate lines and the points
are products across a qubit. Let n_A and n_B be the numbers of line terms
on A and on B. By Fact 2, n_A <= 1 and n_B <= 1 (a line term on A is
absent at 01 and at 10, so two of them would leave two absent terms
there).

Base point.

- n_B = 0: at x_0 = 00 all four terms are visible.
- n_A = 0 and n_B = 1: at x_0 = 01 all four are visible.
- n_A = n_B = 1: two planes, one A line, one B line. At 00 the planes and
  the A line are visible and form a rank-3 decomposition of alpha_00 psi_3
  (distinct and independent, since psi_3 in the span of two stabilizer
  states contradicts chi(T^3) = 3), so up to G_3 it is one of the 8 stored
  rank-3 decompositions of |T>^3 (`qubit_T_m3_rank3.json`, 4 orbits under
  G_3). At 11 the same three terms are present with slices that are phased
  Pauli translates of their slices at 00, the B line is absent, and the
  ratio is tau^2: an exact row of the tables at j = 2 (from the base point
  11, j = -2). `tables.py` for the 8 decompositions at |j| <= 2:

  | ratio tau^j | exact | stabilizer |
  |---|---|---|
  | j = -2, 2 | 0 | 0 |
  | j = -1, 1 | 1 | 111 |
  | j = 0 | 8 | 768 |

  So this configuration does not occur.

What remains is one stage at the two base points 00 and 01 (01 and 10 are
exchanged by the swap of the sliced qubits, a symmetry of psi_5): all four
terms visible at x_0, so the base (u_1^(x_0), ..., u_4^(x_0)) is a full
4-cover of psi_3, four stabilizer states, repeats allowed, whose span
contains psi_3 with all coefficients d_i = c_i / alpha_{x_0} nonzero.
There is no stage (beta): at H^5 it carried the case of one invisible
line term, which here is the n_A = n_B = 1 configuration the tables
exclude.

The matcher (`slice_cover.SliceMatcher(E, 2)`) assumes nothing beyond the
base point: every term may have any of the five subspaces of F_2^2
through x_0 as its flat (the absence option is in every coordinate-slice
option list, the presence pattern is filtered to subspaces through x_0 at
assembly). Facts 1 and 2 are used only to show that the enumerated bases
are complete.

## 3. The census

The full 4-covers of psi_3 up to G_3 (`driver.py census --write`,
`research/t5_rank4/covers4.json`, sha256 `e83212833195240a`, 62 s):

| kind | count | content |
|---|---|---|
| A, distinct independent | 4,697 | `CoverEnumerator.covers(4)` with orbit qubit_T: pivot one per G_3 orbit, partner minimal in its stabilizer orbit, members in orbits at or above the pivot's, residue kernel mod 65521 listing a superset, every candidate decided mod 2013265921 and numerically (2,372 orbits; the reductions leave about two representatives per orbit) |
| B, distinct dependent | 0 | span(T) holds no further dictionary state for any of the 4 full 3-covers T (enumerated: 0, 0, 0, 0) |
| C, a repeated state | 12 | the multisets T + (x,), x in T, over the 4 full 3-covers (12, 352, 856), (33, 124, 158), (143, 565, 749), (309, 358, 873); pattern (2, 1, 1) |

Why nothing else. Four visible terms with a repeated state have three
distinct states with merged coefficients; if the block's merged
coefficient is nonzero the three states are a full 3-cover T and the
repeated one is in T (the 12 listed); if it is zero (two copies
cancelling at x_0) the two other states cover psi_3, against chi(T^3) =
3, so there is no cancel-at-base multiset. Four distinct dependent states
have rank 3 (rank 2 would put psi_3 in the span of two states), contain
three independent states that span the same space and hence form a full
3-cover T (a zero coefficient would again leave two states), and the
fourth state lies in span(T), which holds none. Patterns (2, 2) and
(3, 1) have two distinct states. G_3 acts on the three unsliced qubits,
commutes with the slicing, and carries rank-4 decompositions of psi_5 to
rank-4 decompositions, so one base per orbit suffices; the census carries
duplicates rather than fewer.

Census control (`driver.py control-census`, `results/control_census.json`).
The numeric enumerator `slice_lift.all_decompositions("qubit_T", 3, 4)`
lists 5,205 rank-4 index tuples of |T>^3 (at least one per orbit, non-full
covers included, 1 s); 4,697 of them are full covers of distinct states and
fall into 2,372 G_3 orbits, exactly the orbits of the census's kind A
covers; the 8 rank-3 tuples and the stored rank-3 list fall into the 4
orbits of the enumerator's 3-covers; and the degenerate multisets built
over the numeric 3-covers equal the census's 12 up to G_3.

The partition records the census hash and the aggregate re-enumerates the
census (`driver.census_lists`, about 60 s) and requires equality of the
lists and the kinds.

## 4. The matcher

Every base runs through `verify_challenge/slice_cover.SliceMatcher(E, 2)`
with `E = CoverEnumerator(3, orbit="qubit_T")`, at x_0 in {00, 01}: the
two coordinate slices at ratio tau^{+-1} and the composite slice at ratio
tau^{+-2} (x_0 = 00) or 1 (x_0 = 01). Kind A runs through the compiled
kernel `SliceMatchKernel`, which takes the target slices mod both primes
as inputs and is orbit-agnostic (it converts phase codes with the same i
the T field uses); kind C runs the Python reference through the
coefficient family and the block treatment with the 2026-09-23 repairs
(cancelling copies reconstructed, `UnpinnedFamily` raised instead of a
silent drop, `docs/notes/h6_rank5_stagec_repair.md`). Everything section 4
of the H^6 note says about exactness holds here with psi_5 = |T>^5 and the
Q(zeta_24) field.

Refusals and undecided runs. A base whose family is empty or has a dead
ordinary coefficient is refused; the census excludes such multisets, so a
refusal fails the aggregate. `UnpinnedFamily`, the candidate cap of the
dense solve, a hit on which the modular and numeric decisions disagree,
and a cover not run before `--max-seconds` are recorded as undecided; the
batch exits 1 and the aggregate fails.

## 5. Controls

All on one laptop core at nice 19 with load average 30 to 45 on 18 cores
(other sessions), so every time below overstates an unloaded core.

Control 1, the tables (`driver.py tables`, `results/tables.json`): section
1 and section 2, both PASS.

Control 2, the census (`driver.py control-census`): section 3, PASS.

Control 3, the rank-6 witness (`driver.py control-witness`, and
`--reference`; `results/control_witness.json`,
`results/control_witness_reference.json`). `bounds/qubit_T-m5-upper-6.json`
is the product of the rank-3 decomposition of |T>^4 with |0> and |1> on
qubit 5, so its terms are point terms along qubit 5 and it has all-visible
bases only along pairs among qubits 1 to 4 (its k = 2 term is a line
term along the pairs (1, 2) and (3, 4) and a plane along the other four):
14 distinct all-visible (base, x_0) pairs, 10 with six distinct
independent states (kappa = 0) and 4 with six distinct dependent states
(kappa = 2; the bases (31, 446, 33, 452, 84, 739) and (33, 452, 84, 739,
31, 446) at 00 and 11). Result at the default candidate cap
(`results/control_witness_default_cap.json`, and the same through the
reference, `results/control_witness_reference_default_cap.json`): the 10
kappa = 0 bases return the witness itself among their genuine rank-6
decompositions (1 or 4 per base, exact mod 2013265921, all coefficients
nonzero) in 0.7 to 3.0 s each, the kernel and the reference agreeing base
by base; the 4 kappa = 2 bases abort after 13 s at the 2,000,000-candidate
cap of the dense 2-parameter solve at their first coordinate slice and are
recorded as aborted, a control failure at that cap, never a pass
(`control-witness: 10/14, 4 aborted, FAIL`). Re-run with `--max-cand
20000000`, base 4 did not finish inside the 600 s wall-clock cap of this
session and was killed (the exact re-decision of millions of modular
candidates runs in Python); no raised-cap record exists. A 2-parameter
family of six distinct dependent states is a rank-6 shape: the census has
no dependent base at all (kind B is empty), and its kind C bases have
three distinct states with a pinned family, so the dense parametric solve
is on no rank-4 path of this exclusion. The H^5 control had the same shape
at one base and finished under the raised cap in 264 s.

Control 4, planted four-term instances (`driver.py control-planted
--plant 10`, `results/control_planted.json`): random configurations of
five kinds at both base points, the target their sum with
Gaussian-integer coefficients, the matcher run against the target's
slices: (a) four planes; (b) three planes and a line on the diagonal
through x_0; (c) a repeated base state (two plane copies with one base
slice) and two planes; (d) a repeated base state, a plane and a diagonal
line; (e) three planes and a term on the coordinate line through x_0
(excluded by Fact 1, but the matcher must find it). Result: 20 of 20
planted decompositions recovered (two per kind per base point, every one a
hit with the planted term codes), in 0.1 to 0.8 s each; the (c) and (d)
instances run through the block path (one pair block, 8 reconstructions
in all).

Control 5, the rank-3 decomposition of |T>^4 along a qubit pair
(`driver.py control-m4-pair`, `results/control_m4_pair.json`): the matcher
at n_1 = 2 over every full 3-cover of |T>^2 (the 60 two-qubit states;
independent covers from the enumerator, dependent and repeated ones from
the same degenerate routes over the full 2-covers) must recover the
stored class wherever some member of it has an all-visible base at 00 or
01 with a full family, and nothing outside the stored list. Bases: 206
independent full 3-covers of |T>^2 (one per orbit of the symmetry of
psi_2, order 18) and 5 dependent or repeated ones (over the full 2-covers),
422 (cover, x_0) runs in 2 s (412 through the compiled kernel), 6 genuine
rank-3 hits forming one class under the unitary symmetry group of psi_4
(order 1,944), the stored class, which is the one expected; nothing
missing, nothing unexpected, nothing outside the stored list, no refusal,
no undecided run.

## 6. Rates, partition and the run

Measured by `driver.py sample --count 100` (`results/rates.json`), per
(cover, x_0) run:

| kind | covers sampled | matcher | mean | median | max | hits, refused, undecided |
|---|---|---|---|---|---|---|
| A | 100 spread through the 4,697 | native | 42 us | 26 us | 0.42 ms | 0, 0, 0 |
| C | all 12 | reference (one pair block) | 9.4 ms | 3.8 ms | 35 ms | 0, 0, 0 |

Kind A's solution histogram over the 200 runs: 180 die at the first
coordinate slice, the rest have (1, 1) to (4, 16) solutions on the two
coordinate slices or die at the second. The whole exclusion is about
4,697 x 2 x 42 us plus 12 x 2 x 9.4 ms, well under a minute of matching;
it is a laptop job, and the partition exists for the record's shape (a
hashed geometry, seeded re-runs), not for the cost.

The partition (`driver.py partition --target-s 120 --pod-factor 2.0
--max-covers 1000`, `partition.json`, sha256 `1f73f0ba38274b8a`, census
sha256 `e83212833195240a`): six batches, the kind A covers round-robin
into five batches of 939 or 940 (indices 0 to 4) and the 12 kind C covers
in batch 5; the `--max-covers` bound, not the cost, sets the count.

The run (2026-09-24, 02:25 UTC, one laptop core at nice 19, Apple M5 Pro,
macOS 26.5.1, load average 30 to 45 from other sessions; `batch.py K` for
K = 0..5 at commit 68aa938, `results/batch_K.json` and logs): 4,709
covers, 9,418 (cover, x_0) runs, 9,394 of them through the compiled
kernel and 24 (kind C) through the block path of the reference; 0 hits, 0
refused, 0 undecided; 6.2 CPU-seconds and 4.4 s of wall time in all (about
1 s per batch, dominated by start-up; matching 0.1 s per kind A batch and
0.3 s for kind C). Kind A's solution histogram over the 9,394 runs: 8,431
die at the first coordinate slice, the rest have (1, 0) to (4, 16)
solutions on the two coordinate slices; kind C: 16 die at the first slice,
4 have (8, 0) and 4 have (8, 8). The aggregate (`aggregate.py --recheck 2
--recheck-seed 20260924`, `results/aggregate.log`, 68 s) re-enumerated
the census (61 s, equal to the stored one), verified every stored batch,
re-ran batches 0 and 1 from scratch with matching deterministic hashes
(`3d67b74e7d68676c`, `3e6cd157ee251285`), wrote `batch_manifest.json`
and printed `CERTIFIED chi(qubit_T^5) >= 5`; the certificate script
`verify_challenge/cert_qubit_t_m5_rank4_attested.py` (`results/
certificate.log`, 67 s) printed the same. The `--no-native` replay of
batch 0 (`STABRANK_NO_NATIVE=1 batch.py 0 --out-dir results/nonative
--force`, 4.6 s of matching through the Python reference against 0.05 s
native) has the same deterministic hash as the native record, the same
histogram and the same (empty) hit list. A first version of the record
carried the count of kernel runs inside the deterministic part, and the
replay differed there alone; the count now follows the hash with the
timing fields, and every batch, the aggregate and the replay were redone
at commit 68aa938.

No pod run is needed. For the record, the pod commands would be those of
the H^5 note's section 8 with `research/t5_rank4` for `research/h5_rank5`,
`seq 0 5` for the batch indices and `--recheck-seed 20260924`.

## 7. Soundness checklist

From `docs/notes/qutrit_m4_rank5_review.md` and the H^6 stage C repair;
status of each item for this pipeline.

1. The degenerate list carries every multiset a rank-4 decomposition can
   produce, cancel-at-base multisets included: yes; here no cancel-at-base
   multiset exists (section 3), the 12 listed are all there are, and the
   aggregate's fresh enumeration must equal the census.
2. `reconstruct_block` allows classes outside the translate set used by
   two copies with net coordinate zero and restricts composite codes to
   the structure lemma's shapes: yes, the repaired `slice_cover.py`.
3. A block state reaching the final loop with an unpinned family or with
   dependent block translates raises `UnpinnedFamily`; the batch records
   the run as undecided and the aggregate fails: yes.
4. A refused cover fails the aggregate: yes (`aggregate.check_batch`).
5. Floating-point pruning backed by exact checks: the tolerances of
   `has_zero_coefficient`, `is_full`, `restrict` and `_refine_split` are as
   at H^6, the modular checks sit next to them, and the independence used
   to classify the census kinds is decided mod 2013265921 and numerically
   with the two required to agree (`common.kind_of`).
6. Every hit re-decided from its phase codes mod 2013265921 and
   numerically (`common.decide_terms`, the T field), a disagreement
   undecided, the aggregate re-deciding every stored hit: yes.
7. The deterministic hash excludes the timing, host and matcher-path
   fields, the kernel-run count included: yes, after the repair of section
   6; the `--no-native` replay of batch 0 reproduces the native record's
   deterministic hash (`results/nonative/batch_0.json`).
8. Every batch record carries the partition and census hashes; the
   aggregate checks them: yes.
9. No `id()`-keyed caches: `SliceMatcher.cache` is keyed by dictionary
   index.
10. `--max-seconds`: covers not run are recorded as undecided, the batch
    exits 1, and `--resume` redoes such a record.
11. The matcher assumes nothing beyond the base point: yes (every flat
    through x_0, absence in every option list).
12. Controls re-run after the last matcher change: the only change to
    `slice_cover.py` on this branch is the orbit parameter (commit
    ed917ba); every control and every batch ran at or after that commit.
13. The T field is a ring homomorphism from Q(zeta_24) and shares i with
    the compiled kernels: asserted in `Field.__init__` (sqrt 2^2 = 2,
    sqrt 3^2 = 3, tau^2 = i (2 - sqrt 3)), tested against the H field, and
    exercised end to end by the witness control through the kernel and the
    reference with the same result.

## 8. What is proved, what is assumed, what is open

Proved by table or argument here: Fact 1 (given the one-element rank-3
list of |T>^4), Fact 2 (a board bound), the flat lemma, the base-point
case split, the exclusion of the (1, 1) configuration, the completeness
of the census given the enumerator (with the numeric cross-check).
Assumed from earlier work: the completeness of the rank-3 list of |T>^4
up to symmetry (as `qubit_T-m5-lower-4.json`), PR #87's shape enumeration,
the pivot and partner reductions of the enumerator (the H^6 argument,
cross-checked here against the numeric enumerator orbit by orbit).
Not closed: the four kappa = 2 witness bases, aborted at the default
candidate cap and unfinished at the raised cap within this session's
600 s cap (section 5, control 3); they lie on no rank-4 path, so the
exclusion does not rest on them, but the control is recorded as
incomplete rather than passed. Nothing else is open before the bound: the
run, its aggregate with two seeded re-runs, the `--no-native` replay and
the certificate script all ran on the laptop (section 6).
