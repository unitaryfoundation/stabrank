# Excluding rank 7 for |T3>^3: design note

Outcome (2026-09-20). The scan described here ran to completion and
excluded rank 7: chi(T3^3) = 8. 459 batches, 1.31e13 inner steps,
1,213,458,815 candidate class sets decided exactly, none containing V_3,
none spurious or undecided; 80.1 CPU-hours over 22 hours of wall time on
four low-priority cores of an 18-core laptop, 21 ns per step averaged
(the design estimate below was 66 CPU-hours at 18 ns on idle cores). The
bound is `bounds/T3-m3-lower-8.json` on the attested tier (PR #62): the
certificate re-runs two batches from scratch and hashes the rest, and took
58 minutes on the 4-core CI runner against the 3600 s budget. A full
aggregation with three re-runs (batches 94, 109, 363) also matched bit for
bit. The rare-type pivot lemma of section 4 turned out false
(`t3_rank7_rare_pivot.md`), so this was the unreduced scan. What would lift
the tier: a kernel near 10 ns per step with a re-run of a larger subset, or
an independent re-run of every batch on other hardware.

Independent exclusion (2026-09-26). A second scan, designed and run apart
from this pipeline (`research/t3_rank7/independent/`), excludes rank 7 as
well: the certificate's pivot order with Stab(i, j)-minimal third pivots
(5.52e13 inner steps against the 1.31e13 of the orbit-block order), a
different kernel and task partition, a case split by the projected geometry
(six coplanar images, five, or neither, the first two excluded by two-pivot
scans), 378 CPU-hours on a RunPod Xeon pod over 92.9 hours of wall time plus
a laptop supplement. Its checks found that its own third-pivot masks had been
computed over the nontrivial stabilizer elements only, which skipped 3.9
percent of the canonical triples; the supplement scanned exactly those. The
two enumerations share the dictionary, the descent, the projection, the group
and the exact decision, and nothing else. This is the "independent re-run on
other hardware" of the paragraph above in substance; under CONTRIBUTING's
definitions it does not change the tier, which is fixed by what the
certificate re-runs under its budget.

Status of the cell before the scan: 7 <= chi(T3^3) <= 8, lower bound from
`verify_challenge/cert_t3m3_rank7.py` (PR #41), upper bound from the
eight-term witness in `bounds/T3-m3-upper-8.json`. Excluding rank 7 settles
chi(T3^3) = 8, the cell named by goal G4. Prototype code and measured
numbers are under `research/t3_rank7/`; its README says what is validated.

Notation. N = 30240 is the number of three-qutrit stabilizer states, one
per state up to phase, indexed in the order of `build_dictionary(3)`. V_3 =
span{psi_0, psi_1, psi_2} is the Galois space of the certificate, a
3-dimensional subspace of Q(w3)^27, and every stabilizer state is a vector
over Z[w3]. For a state s, its image q_s is its class in the quotient
Q(w3)^27 / V_3, a space of dimension 24. G is the stabilizer of V_3 in the
Clifford group extended by complex conjugation, |G| = 2916 with 45 orbits
on the states; #41 records that no larger symmetry exists. Stab(i) and
Stab(i, j) are the subgroups fixing the state i, or both i and j.

## 1. What a rank-7 configuration is

Lemma 1. chi(T3^3) <= 7 if and only if there are seven linearly
independent stabilizer states s_1, ..., s_7 whose images q_1, ..., q_7 span
exactly four dimensions.

Proof. If chi <= 7, a decomposition with r <= 7 terms has V_3 in its span
(Galois descent). If its states are dependent, a basis of their span has
at most six states and still spans V_3, contradicting chi >= 7. So the
seven states are independent, dim span(s) = 7, and dim(span(s) + V_3) = 3
+ dim span(q) forces dim span(q) = 4. Conversely, seven independent states
with dim span(q) = 4 give dim(span(s) + V_3) = 7 = dim span(s), so V_3 lies
in span(s).

So the task is: for every 4-dimensional subspace W of the 24-dimensional
quotient, the states with image in W are linearly dependent whenever there
are seven of them. Equivalently, in the language of the certificate, for
every class set C (all states with image in some W, up to the pivot
order), rank(C) <= 6, and the rank pair rank(C) versus rank(C together
with the psi_r) decides it exactly mod 2^31 - 1 by the same Hadamard
argument (rank values up to 8 are exact; a class set of rank at least 8
cannot lie in the 7-dimensional preimage of W and is a projection
collision, to be split, see section 3).

Case split by the geometry of the seven image points in the projective
space P(W), which is a P^3. "Coplanar" means lying in a 3-dimensional
linear subspace of W.

Case A, six coplanar. Let C be six states whose images span a
3-dimensional W', P the preimage of W' (dimension 6), and s_7 the
remaining state. If q_7 lies in W' all seven states lie in P and are
dependent. Otherwise write v = u + c s_7 for v in V_3 with u in span(C);
reducing modulo P gives c = 0, so V_3 lies in span(C): a rank <= 6
decomposition. Excluded by `cert_t3m3_rank7.py`; no further computation.
The lemma is not written down in the repository, only in PR #41's text.

Case B, five coplanar and no six. Let C be the five states, W' and P as
above (P has dimension 6), and s_6, s_7 the other two. Since all seven
images span four dimensions, q_6 and q_7 are congruent up to a scalar
modulo W': t = s_6 - lambda s_7 lies in P for some lambda. Any v in V_3
written over the seven states has its s_6, s_7 part in P, hence
proportional to t, so V_3 lies in span(C) + <t>. This has dimension at
most rank(C) + 1, so dim(V_3 meet span(C)) >= 2, that is

    rank(C with psi_0, psi_1, psi_2) <= rank(C) + 1.

A certificate for case B: run the two-pivot kernel of the certificate
with `need = 3` (five states in a 3-dimensional image space, pivots being
two of the five, Z members allowed), keep the class sets passing the rank
filter above (exact mod 2^31 - 1), and for each survivor hash all N images
modulo W' to find the groups of states parallel modulo W'; for each group
H, decide rank(C with H with the psi_r) = rank(C with H) exactly. C with H
lies in the 7-dimensional P + <s_6>, so one rank pair per (survivor,
group) covers every choice of pair. #41 says this case was
excluded exactly in follow-up work; nothing of it is in the repository
(grep for coplanar finds only the rank-6 script), so it counts as open
until a script exists. Its cost is that of the `need = 3` flood (not
measured here) plus N hashing steps per survivor.

Case C, no five coplanar. This is what #41 calls general position. It
contains two sub-cases: C4, some four images coplanar (a size-4 circuit
in the matroid sense), and C0, every four images independent, hence no
three collinear either. C0 is general position in the circuit sense of
#41's second paragraph. Neither sub-case reduces the scan of section 2;
both are covered by it.

The three-pivot scan covers A, B and C at once: it lists every 7-set with
image span at most 4, so the split is only useful for intermediate
milestones, not for saving work.

## 2. Cost

Pivot order. Move a configuration by G so that one member becomes an orbit
representative i. Then choose h in Stab(i) minimising the least index
among the other six members; that member j is minimal in its
Stab(i)-orbit and the other five have index above j. Then among the
remaining members whose image lies outside span(q_i, q_j) (at least two,
since the total span is 4), choose h in Stab(i, j) minimising the least
index k; Stab(i, j) preserves span(q_i, q_j), so this minimisation keeps
the earlier conditions. The members inside span(V_3, s_i, s_j), the Z
members, have index above j; the others have index above k. The kernel
enumerates, per (i, j, k), the states above k whose images modulo
span(q_i, q_j, q_k) vanish or are mutually parallel, and needs Z members +
vanishing + parallel >= 4. One inner step is one row reduction and one
hash of a state l above k.

(a), (b). With j minimal under Stab(i), k any state above j, l any state
above k, the count is the sum over pivot pairs (i, j) of C(N - j - 1, 2).
Averaged over j this is N^3 / (6 |Stab(i)|) per representative, and
summing over representatives gives N^4 / (6 |G|) = 4.78e13. So #41's
estimate already contains (a) and (b); they are not further reductions.
The exact count from `count_steps.py` is larger, 5.83e13 over 326,368
pairs, because the pairs are not evenly spread in j. Restricting k to
Stab(i, j)-orbit minima (the Schreier route of `stabilizer_orbit_labels`
applied to the pair) touches 21,219 pairs and brings the count to
5.52e13, a 5 percent saving.

Orbit-block order, not used by #41. Relabel the states so that each
G-orbit is a contiguous block, blocks in increasing order of orbit size.
Move the configuration so that a member of its lowest block becomes the
pivot; every other member then lies in a block at or above the pivot's,
so the scan for a representative in block a runs on the states with
label at or above the block start (M_a of them), and j, k, l are taken with respect to
the new labels (the Stab(i)-minimality of j is with respect to the new
labels, which is legitimate because minimality is defined by whatever
order the scan uses). The count becomes the sum over blocks of n_a M_a^3
/ (6 |G|) with n_a the orbit size, which for a fine partition approaches
N^4 / (24 |G|), a factor of four. With the actual 45 orbit sizes (three of
2916, seven of 1458, two of 972, seven of 729, four each of 486, 243, 162,
54, one of 108, three each of 81, 27, 9) the exact count from
`count_steps.py` is 1.50e13 over 171,832 pairs, and 1.31e13 with the
Stab(i, j) mask on k: a factor of 3.9, or 4.4, over the certificate
order.

(c) The circuit argument. Every seven points in a 4-dimensional space
contain a dependent subset of at most five. Size-2 circuits (two parallel
images) do not exist at m = 3 (asserted by the certificate). Size-3
circuits (three images spanning two dimensions) are the Z members of the
two-pivot kernel; for each such triple the remaining four images must
span at most two dimensions modulo the triple's plane, which is a
four-collinear search costing N^2 / 2 per triple, so this route is cheap
only if the triples are few (their number is not recorded; the two-pivot
histogram of the certificate has it). Size-4 circuits are the `need = 2`
classes of the two-pivot kernel; for each, the other three images are
parallel modulo the class's 3-space, one hashing pass of N steps per
class. Neither reduces case C0, where every 5-subset is a circuit and the
three-pivot scan is the enumeration of those circuits. So (c) gives no
reduction of the general-position count; it only certifies A, B and C4
separately, which the full scan does anyway.

Measured per-step cost. `kernel3` in `research/t3_rank7/scan3.py` (numba,
same primitives as the certificate) on the real m = 3 data, single pivot
pairs, machine load average between 27 and 120 on 18 cores, `nice -n 19`:
42 to 45 ns per inner step on pairs of 3e7 to 2.5e8 steps
(`results/m3_pair_timings.json`). This is an upper bound for this kernel;
an idle core should be near 25 ns, and a C++ kernel with 32-bit
coordinates, table inverses and a direct-indexed counting array in place
of the hash table should reach about 10 ns (not built; the estimate is
from the operation count: three multiply-reduce steps, one table lookup,
two multiply-reduce steps, one hash or index, one memory probe).

Candidate flood. The pairs tried produced 0 to 4602 candidate classes,
about 1e-5 classes per inner step for the expensive small-j pairs, all
rejected exactly and none spurious. Extrapolated, the full scan produces
of order 1e8 to 1e9 class sets, mostly four parallel members in a 4-space
with three pivots ("0,4" in the histogram) plus some 4-spaces holding ten
or more states ("0,7", "0,9"). The exact decision must therefore run in
compiled code inside each batch (about 5 microseconds per rank pair in
numba, so 1e3 to 1e4 CPU-seconds in total) and the batches must store
counts and exceptions, not the class lists, which would be gigabytes.

CPU-hours. At the measured 45 ns per step:

| pivot order                             | inner steps | CPU-h at 45 ns | at 25 ns | at 10 ns |
|-----------------------------------------|-------------|----------------|----------|----------|
| certificate order, k > j (a, b)         | 5.83e13     | 729            | 405      | 162      |
| plus Stab(i, j) on k                    | 5.52e13     | 690            | 383      | 153      |
| orbit-block order                       | 1.50e13     | 187            | 104      | 42       |
| orbit-block order plus Stab(i, j)       | 1.31e13     | 164            | 91       | 36       |

The first row at 45 ns reproduces #41's "500 to 770 CPU-hours in this
kernel". The recommended configuration is the last row: about 160 CPU-h
with the present numba kernel on a loaded machine, about 90 on idle
cores, and about 36 with a C++ kernel at the estimated 10 ns; on four
idle cores that is a weekend to two weeks of wall clock depending on the
kernel, and it runs unattended in batches.

## 3. Prototype

`research/t3_rank7/scan3.py` implements `kernel3` (the three-pivot scan
per first pivot with the certificate's pivot order, Z members, free
members, count-only mode) and `decide` (one rank pair per class set mod
2^31 - 1, spurious classes counted). Validation, all at m = 2 unless
stated, all passing:

- Exhaustive cross-check on random sub-dictionaries of 22 to 26 states
  with planted targets (a random 3-dimensional subspace of the span of
  seven chosen states) and with V_2: every 7-subset with projected image
  rank <= 4 lies in some class set. The plant is found.
- Planted full-dictionary test (360 states): the plant's class set is
  produced and accepted mod ell.
- Every class set of the certificate's two-pivot kernel (six-state
  configurations) is contained in a class set of the three-pivot kernel
  for the same pivot.
- Symmetry: on a G-invariant sub-dictionary of 57 of the m = 2 states (one
  orbit of 54 under the 2-copy group of order 324, plus the three states in
  V_2), the 77,091 image spaces found with one pivot per orbit and a
  Stab(i)-minimal second pivot, closed under the group, equal those found
  with trivial symmetry. (The full m = 2 dictionary floods: with a
  6-dimensional quotient, 7 images in a 4-space is a codimension-2
  condition and the scan for a planted target reports 24 million class
  sets; this is why m = 2 is a control for the kernel logic and not a model
  for the m = 3 flood, and why the symmetry check runs on a sub-dictionary.)
- m = 3, real data: five pivot pairs scanned and decided exactly, no
  class set contains V_3, no spurious class. This is a timing run.

Not implemented: the orbit-block order, the Stab(i, j) mask, the
re-splitting of spurious classes (re-hash the class members with a second
independent projection, or reduce them exactly in the full 21-dimensional
quotient; expected frequency about 1e-3 per full scan from the collision
probability N^2 / ell^3 per triple), and the compiled decision.

## 4. Structural shortcuts

Slice constraint. For a coordinate slice x_p = c, the restriction of
|T3>^3 is w9^c |T3>^2 and the restriction of V_3 is V_2 (each psi_r
restricts to a scalar multiple of the corresponding psi'_r of two
copies). So the restrictions of the seven states to each of the nine
slices span a space containing V_2. Lemma: at every slice at least three
restrictions are nonzero, and if exactly three they are the three
stabilizer states inside V_2. Proof: chi(T3^2) = 3, and three states
spanning a space that contains a 3-dimensional V_2 span exactly V_2, and
exactly three states lie in V_2 (asserted by the certificate). Why it
does not decide anything: a full-support state (k = 3, 19683 of the 30240
states, five of the eight witness terms) meets every slice, so any
configuration with three full-support states satisfies the counting part
trivially, and the spanning part is a codimension-2 condition per slice on
the restricted images, a filter on candidates but not a constraint on
pivots, which is where the cost is. The same applies to the three coset
planes x1 + x2 + x3 = r, where the restriction is psi_r alone (rank 3 on
that plane, the `T3sector` cells).

Support counting. Supports are affine flats of sizes 1, 3, 9, 27, and the
only requirements are that their union is F_3^3 and that no state lies in
V_3 (already used as `isfree`). One full-support state covers everything.
No lemma.

Pauli spectrum and extent. The stabilizer fidelity of |T3> is
(1 + 2 cos 40 deg)^2 / 9 = 0.712, so extent-type bounds give chi(T3^3) >=
1 / 0.712^3 = 2.8. Far from 7; no route.

Slice-and-lift. The lemma of `slice_lift.py` applies to a rank-r
decomposition of phi^(m+1) with r = chi(phi^m), where the slice
decompositions are minimal and their coefficients unique, so lifting is a
finite matching. Here r = 7 against chi(T3^2) = 3: the slice
decompositions are non-minimal, their coefficients form an affine family
of dimension four, and the discrete lift choices (Pauli class and two
phases per term) number 81^7 per slice configuration. It fails as a
finite enumeration; the linear-algebra version of the condition is the
three-pivot scan again.

Larger symmetry. #41 computed the stabilizer of V_3 in the Clifford group
with conjugation to be exactly 2916, and the Galois automorphisms fixing
Q(w3) act trivially on V_3 and on the dictionary. Nothing to gain.

Rare-type pivot. If a lemma forced every rank-7 configuration to contain a
state of a rare type (a line or point state, 1080 of 30240), the first
pivot could be restricted to those orbits and the count would drop by the
fraction of such states, about 28x. The rank-8 witness contains two line
states and the excluded rank-5 configuration contained only line and plane
states, so the pattern is not implausible, but no argument is known and
none was found here. Worth an hour of thought before the scan runs, not a
dependency. Follow-up: `t3_rank7_rare_pivot.md` shows that the slice
structure cannot give it (five full-support two-qutrit states already span
V_2, and the best pivot restriction the slice route can yield is a factor
of 1.38), so the scan should run without it.

The one reduction that is both sound and unused is the orbit-block order
of section 2.

## 5. Recommendation

Run the three-pivot scan in the orbit-block order with the Stab(i, j)
mask, in a compiled kernel, as idle-time batches. The case split does not
reduce the work, and the structural routes give filters, not lemmas.
The rare-type lemma was examined in `t3_rank7_rare_pivot.md` and gives
nothing by the slice route; do not wait for it.

Kernel. Port `kernel3` to C++ next to `cpp/src/pivot_pair.cpp` (or keep
numba if 25 ns per step on idle cores is acceptable): coordinates as
32-bit integers mod 65521, inverse table, per (i, j) precompute the rows
above j reduced modulo span(q_i, q_j) with a projection to F_ell^3 (so a
direction has one canonical residue and the grouping is a direct-indexed
count array of 65521 stamps rather than a hash table), verify the
candidate buckets in the 4-coordinate version to remove the roughly ten
spurious 4-collisions per third pivot, and decide every candidate class
inside the batch mod 2^31 - 1 with the Hadamard bound, storing only
counts, a histogram, any class of rank at least 8 (spurious, to be
re-split) and any class whose span contains V_3.

Batches. Work is the sum over pivot pairs (i, j) of C(N - j - 1, 2) in
the block labelling, so a batch is (block a, range of j) with the range
chosen for about 15 to 30 minutes at 10 ns per step, giving roughly 300
to 600 batches. The batch runner accepts the `run.py` command line so that
`autoresearch/loop.py run MANIFEST --runner research/t3_rank7/batch.py`
schedules it unchanged: `ORBIT M RANK --seeds 1 --seed0 SEED` with the
seed as the batch index and the annealer flags ignored, `stop_on_solve`
false, `cap_s` a few hours, one job per batch. Exit 0 on completion
(loop.py logs it as "discovery", a misnomer that costs nothing) and write
`research/t3_rank7/results/batch_<index>.json` with the code hash, the
dictionary hash, the projection seed, the (a, j) range, the step count,
the histogram, the exceptions and the wall and CPU time. The loop's
checkpoint gives resume; the result files give the audit trail.

Aggregator and certificate. A script `cert_t3m3_rank8.py` that under the
3600 s budget rebuilds the dictionary, the descent, the projection and
the symmetry (about a minute), checks that the batch files cover every
(a, j) exactly once with matching code and data hashes, re-decides every
stored exception exactly, re-runs a fixed deterministic subset of batches
live (the cheapest ones and a fixed spread of j values across the
blocks, whatever fits in the budget) and compares their counts and
histograms bit for bit, and prints `CERTIFIED chi(T3^3) >= 8` only if
every batch reports zero classes containing V_3 and zero unresolved
spurious classes.

What that certificate must be honest about. The enumeration cannot be
re-run under any budget the board allows; its completeness rests on the
stored batch outputs, which anyone can regenerate with the committed
runner in the recorded CPU time. CONTRIBUTING's `verified` tier asks for
exactness throughout, and the arithmetic here is exact throughout (exact
hashing mod 65521 for the superset, exact ranks mod 2^31 - 1 with the
Hadamard bound), but `verified` also means the pipeline confirmed the
argument, and here it confirms the decision and a sample of the
enumeration, not the enumeration. The submission should say so in
`notes` in those words, record the offline cost in `provenance.compute`
(CPU-hours, hardware, dates, git hash of the runner), and declare
`budget_s: 3600` for the aggregator. Whether that earns `verified`,
`reproduced` or a new marker for attested offline enumeration is a
maintainer decision; the note's recommendation is a new marker, since
neither existing tier describes it and quietly claiming `verified` would
misstate what the pipeline checked. A submission that ships the runner,
the manifest, the result files and the aggregator is reproducible in the
strong sense: every step is deterministic (fixed projection seed, fixed
dictionary order, fixed block order) and any batch can be re-run and
compared bit for bit.

Decided 2026-09-19: the new marker exists as the `attested` tier
(CONTRIBUTING, "Attested offline enumerations"). The submission declares
`certificate.attested` with the manifest path, the number of batches the
aggregator re-runs, the offline cost and hardware, and a note on what
completeness rests on; the verifier checks every stored output against
the manifest's hashes before the script runs, and the tier holds records
below `reproduced`.

## Numbers

From `research/t3_rank7/results/step_counts.json` (exact counts under the
symmetry of V_3) and `results/m3_pair_timings.json` (five pivot pairs,
numba kernel, loaded machine):

| quantity                                        | value    |
|-------------------------------------------------|----------|
| states N                                        | 30240    |
| group order, orbits                             | 2916, 45 |
| pivot pairs, certificate order                  | 326,368  |
| pairs with nontrivial Stab(i, j)                | 21,219   |
| inner steps, certificate order                  | 5.83e13  |
| inner steps, plus Stab(i, j)                    | 5.52e13  |
| pivot pairs, orbit-block order                  | 171,832  |
| inner steps, orbit-block order                  | 1.50e13  |
| inner steps, orbit-block order plus Stab(i, j)  | 1.31e13  |
| measured ns per step (numba, load 27 to 120)    | 42 to 45 |
| CPU-h, recommended order, at 45 / 25 / 10 ns    | 164 / 91 / 36 |
| candidate classes per step (small-j pairs)      | about 1e-5 |

The orbit-block order alone divides the count by 3.9, and with the
Stab(i, j) mask by 4.4, against the 5 percent that Stab(i, j) gives on
its own. The block order costs nothing to implement beyond a relabelling
of the dictionary before the scan.
