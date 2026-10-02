# chi(|T5>^3) >= 6 by slice-and-lift at exact rank: design, controls, and the run

Status (2026-10-02). Done: the pod chain ran on 2026-10-02 (01:30 to
03:19 UTC) and `bounds/T5-m3-lower-6.json` is filed at the attested tier,
with `bounds/T5-m4-lower-6.json` by projection; section 10 has the run.
The pipeline is `research/t5_m3_lift/` (exact lift stage, controls, pod
chain, bound draft), the certificate is
`verify_challenge/cert_t5_m3_lift_attested.py`, and the census it rests on
is the one of `research/t5q_m2_rank5` that found the rank-5 decomposition
of |T5>^2 on 2026-09-24 (`docs/notes/t5q_m2_rank5_exclusion.md`). Sections
1 to 9 are the design and the laptop controls as written before the run.

Notation. w = exp(2 pi i / 5), |T5> = 5^{-1/2} sum_z w^{z^3} |z>, psi_m =
|T5>^m, alpha_z = w^{z^3} / sqrt 5 the amplitudes of |T5>, all nonzero.
The dictionary at m = 2 is the 3,900 two-ququint stabilizer states,
distinct up to phase; every entry of every state is 0 or a fifth root of
unity times a column scalar. G is the unitary symmetry group of psi_2
(order 50). L_c, for c in F_5, is the line x + y = c with phase
w^{3c x^2 - 3c^2 x} in the parameter x, dictionary indices 525, 563, 591,
619, 637 for c = 0, ..., 4; psi_2 = sum_c (w^{c^3} / sqrt 5) L_c is the
rank-5 decomposition the census found. The board's cell is 5 <= chi(psi_3)
<= 15 (`bounds/T5-m3-lower-5.json`, the projection of the m = 2 lower
bound; `bounds/T5-m3-upper-15.json`, the product).

## 1. The argument

### 1.1 The two inputs

chi(psi_2) = 5 exactly: `bounds/T5-m2-upper-5.json` (Lean tier) and
`bounds/T5-m2-lower-5.json`, whose rank-4 exclusion the census pipeline
re-derives exactly by its k = 1..4 censuses (`aggregate.py` runs them,
about 150 s; `control-rank4` stored them).

The listing. The census of `research/t5q_m2_rank5` was built as an
exclusion but is a listing: every one of the 155,422 (pivot, partner)
units ran, every 5-set the kernel returned was re-decided and recorded
(74 records, 0 undecided, 2,795,734,903 modular candidates), and section
1.2 of its note proves that every G-orbit of 5-sets whose span contains
psi_2 contains a listed set. The records hold exactly one hit, the set
{L_0, ..., L_4}, and `decomposition.py` checks that every generator of G
fixes that set. So the family of 5-sets with psi_2 in their span is, up to
the census's modular assumption, the single set {L_0, ..., L_4}. Re-running
the census would reproduce these records bit for bit (the deterministic
hash covers the hits) and is not needed for the listing; the pod chain
offers it as an option and compares every hash.

### 1.2 The lemma, with its case split

Let p be an odd prime, phi in C^p with every amplitude alpha_z nonzero,
psi_m = phi^{(x) m}, r = chi(psi_m), and let E be a family of r-sets of
m-qudit stabilizer states, closed under the unitary symmetry group of
psi_m, that contains every r-set whose span contains psi_m. For E in E
write a^E(e) for the coefficient of e in the expansion of psi_m over E
(unique, since an r-set with psi_m in its span is independent when
chi(psi_m) = r: a dependent set would be spanned by r - 1 of its members).
A pattern is a choice, for every z in F_p, of a set E_z in E and a member
e_z of E_z, and its vector is

    P = sum_z alpha_z a^{E_z}(e_z) e_z (x) |z>.

Lemma. chi(psi_{m+1}) = r if and only if there are sets E_0, ..., E_{p-1}
in E and r patterns with those sets whose members are a bijection onto
E_z at every z and whose vectors are all scalar multiples of stabilizer
states. Otherwise chi(psi_{m+1}) >= r + 1.

Proof. Projection monotonicity (Lean: `stabRankP_powVecP_mono`) gives
chi(psi_{m+1}) >= chi(psi_m) = r, since phi has a nonzero amplitude.

If the patterns exist, sum over the r patterns: at each z the members run
over E_z once, so the sum is sum_z alpha_z (sum_{e in E_z} a^{E_z}(e) e)
(x) |z> = sum_z alpha_z psi_m (x) |z> = psi_{m+1}, a combination of r
stabilizer states. Hence chi(psi_{m+1}) <= r, so = r.

Conversely let psi_{m+1} = sum_{i=1}^r c_i s_i with stabilizer states
s_i. Fix z and slice: u_i = (I (x) <z|) s_i is zero or a scalar multiple
of an m-qudit stabilizer state (the slice of an affine flat is an affine
flat or empty, and a quadratic phase restricts to a quadratic phase), and
alpha_z psi_m = sum_i c_i u_i. Three cases for the slice at z:

(a) some u_i = 0: then psi_m is a combination of at most r - 1 stabilizer
states, so chi(psi_m) <= r - 1, a contradiction;

(b) two of the u_i are parallel: collect them, and psi_m is again a
combination of at most r - 1 distinct states, a contradiction;

(c) every u_i is nonzero and the u_i are pairwise non-parallel: then
{u_1, ..., u_r} up to scalars is an r-set E_z with psi_m in its span, so
E_z is in E, and it is independent, so c_i u_i = alpha_z a^{E_z}(e_z^i)
e_z^i with e_z^i the member of E_z parallel to u_i.

Only (c) occurs, at every z, and the map i -> e_z^i is a bijection onto
E_z at every z. Then c_i s_i = sum_z c_i u_i (x) |z> is the pattern vector
of (E_z, e_z^i)_z, a scalar multiple of the stabilizer state s_i. That is
the data the lemma asks for. QED.

The case split is the whole story of invisible terms here: at exact rank
there are none. Case (a) is the invisible term and case (b) the
coincident term of the N^4 and T^5 designs
(`docs/notes/n4_rank6_design.md`, `docs/notes/t5_rank5_exclusion.md`),
where they occur because the rank being excluded exceeds the rank of the
sliced cell by one (six terms over chi(N^3) = 5). Excluding rank 6 at m =
3 for T5 would need them again (a slice could show four or five visible
terms, with a term on a flat missing the slice value or two terms
coinciding there), and nothing in this note claims it.

Three more things the lemma needs, all satisfied here: every alpha_z is
nonzero (|alpha_z| = 1 / sqrt 5), so no slice equation is homogeneous and
no phase has to be brute-forced; the family is closed under the unitary
symmetry group (the lift stage closes the listed sets under the group's
generators, since the census lists one set per orbit), and the
antiunitary symmetry is not used (it mixes slices, as recorded in
`slice_lift.py`); p is odd, so stabilizer phases are p-th roots of unity
with quadratic exponents (for p = 2 the phases are fourth roots with a
different form, and the stabilizer test would have to change).

### 1.3 What the pattern formulation replaces

`verify_challenge/slice_lift.py` tests lifts by a Pauli relation between
consecutive slices (u^{(z+1)} = mu Q u^{(z)}) and a numerical matching.
At exact rank the pattern formulation is both simpler and exact: the slice
coefficient alpha_z a^{E_z}(e_z) is forced by the uniqueness of the
expansion, so each term is a known vector, and the only question is
whether that vector is a stabilizer state, which is decided by inspecting
its support and phases. The Pauli structure is a consequence (a stabilizer
state's slices are related by a Pauli) and is not assumed.

## 2. The cell

The family is the single set {L_0, ..., L_4}, so a pattern is a function
c: F_5 -> F_5 (the line at slice z), 5^5 = 3,125 patterns, and its vector
is

    P_c = sum_z alpha_z (w^{c(z)^3} / sqrt 5) L_{c(z)} (x) |z>,

the restriction of psi_3 to the set A_c = {(x, y, z) : x + y = c(z)}
(every entry of psi_3 is w^{x^3 + y^3 + z^3} / 5^{3/2} and psi_2 restricted
to the line x + y = c is (w^{c^3} / sqrt 5) L_c). P_c is a stabilizer
state only if A_c is an affine flat, which holds iff c is affine, c(z) =
c_0 + d z: 25 patterns. On x + y = c_0 + d z the exponent x^3 + (c_0 + d z
- x)^3 + z^3 has cubic part (1 + d^3) z^3 + 3 d z x^2 - 3 d^2 z^2 x, and
the monomials z^3, z x^2, z^2 x are independent functions on F_5^2
(degree below 5 in each variable), so it vanishes only if d = 0 and 1 +
d^3 = 0, which is impossible. So no P_c is a stabilizer state, no lift
exists, and chi(psi_3) >= 6. This is the m = 3 half of the sector
observation in section 8.2 of the census note (no plane a x + b y + c z = d
works), now as a lower bound rather than a remark about one construction.

The machine decision does not use the hand computation. `lift.py cell`
reads the 74 records (every check of `aggregate.check_batch`, every unit
run, no undecided unit), takes the hits, closes them under the generators
of G (one set), re-decides the set exactly over Q(zeta_5) (the five
coefficients w^{c^3}, all nonzero, the states independent), and runs the
pattern search. The search is a depth-first enumeration over z with three
exact prunes that every stabilizer pattern satisfies: the moduli
|alpha_z a^{E_z}(e_z)| agree across z (the entries of a stabilizer state
have one modulus), the slices' supports have the same direction space (a
flat whose slices are all nonempty has slices that are translates of one
space), and the translates are affine in z (the slices are t_0 + z d + V).
Pruned patterns are counted, not tested. On the cell 3,100 patterns are
pruned and 25 reach the stabilizer test, which rejects all 25 on the
quadratic check. The whole run, records included, takes under a minute on
the laptop; the search itself is 0.05 s.

## 3. Exactness

Field. `exact.py` represents Q(zeta_n) as tuples of `Fraction`s on the
power basis modulo the n-th cyclotomic polynomial (computed by exact
division of x^n - 1), so equality of tuples is equality in the field.
Conjugation is z -> z^{n-1}; inversion solves the multiplication matrix;
the span solve is Gauss-Jordan elimination over the field. The cell uses
n = 5; the qutrit controls use n = 9 (|T3> has ninth-root amplitudes) and
n = 3 (|N> is rational). No floating point enters any decision; the
dictionary enters through its phase codes (`phase_codes` asserts that
every entry is a p-th root of unity times the column scalar within 1e-6,
the same assertion `patterns5` makes for the census), which is the one
place a tolerance appears, and it appears as an assertion on data the
dictionary enumerator produced exactly.

Stabilizer test. A vector with entries in Q(zeta_n) is a scalar multiple
of a stabilizer state on m qudits of odd prime dimension p iff its support
is an affine flat of F_p^m, every entry has the modulus of the first, every
ratio to the first entry is a p-th root of unity, and the exponents form
a polynomial of degree at most two in the flat's parameters. `stabilizer_
form` checks the flat by row reduction over F_p (size p^k and every point
of the flat present), decides each ratio as v conj(v_0) = w^k |v_0|^2
(which also forces equal moduli), fits the quadratic from the values at
0, e_i, 2 e_i, and e_i + e_j, and verifies the fit at every point. It
returns the board's (k, x0, W, Q, l) in the convention of
`to_witness.term_from_vector`, and the basis control compares its output
with the terms of `bounds/T5-m2-upper-5.json` entry by entry.

Inherited assumption. The census is complete up to the modular caveat of
section 1.4 of its note (a member whose image is nonzero over Q(zeta_5)
could reduce to zero modulo 65521 and be dropped; probability about
65521^{-22} per such event). The lift stage adds nothing to that
assumption and removes nothing from it; the bound inherits it, as every
kernel-census bound of the project does.

## 4. Controls

All through `lift.py`, one process at a time at nice 19, records under
`research/t5_m3_lift/results/`. Times are for the whole command, dictionary
construction included.

| control | what it checks | result | time |
|---|---|---|---|
| `control-stabtest` | 40 random three-ququint stabilizer states built from (flat, quadratic) data are recognized with the right k and rebuilt from the returned term; one phase changed, one point removed, or one modulus changed is rejected | 40 of 40 | 1.5 s |
| `control-basis` | p = 5, m = 1 -> 2 through the computational basis of \|T5> (a machinery control: the basis is a 5-set, not a minimal one): the lift must be exactly the five Z(x)Z sectors of \|T5>^2 with the non-constant patterns c(z) = c - z | 3,125 patterns, 25 tested, 5 stabilizer, 1 lift; terms equal to the board's witness | 10 s |
| `control-rank3` | p = 5, m = 1 -> 2 through every rank-3 decomposition of \|T5> (brute force over the 4,060 triples of the 30 states: 10 sets, 2 orbits of 5): the lemma at exact rank 3 must give no lift, since chi(T5^2) = 5 | 24,300,000 patterns, 101,600 tested, 20 stabilizer patterns, 0 lifts | 107 s |
| `control-product` | p = 5, m = 2 -> 3 from the census set with the one-ququint factor replaced by a full-support stabilizer state beta (three choices: \|+>, w^{z^2}, w^{2z^2 + 3z}): the planted rank-5 decomposition sum_c w^{c^3} L_c (x) beta must be the only lift | each: 25 tested, 5 stabilizer (the constant patterns), 1 lift, terms of k = 2 | 1.1 s |
| `control-t3-basis` | p = 3, Q(zeta_9), m = 1 -> 2 through the basis of \|T3>: the carry decomposition of \|T3>^2 (three Z(x)Z sectors) | 27 patterns, 9 tested, 3 stabilizer, 1 lift | 0.3 s |
| `control-t3-m2` | p = 3, m = 2 -> 3: the carry decomposition is the only rank-3 decomposition of \|T3>^2 up to symmetry (numerical listing, exact re-decision) and must not lift, since chi(T3^3) = 8 | 27 patterns, 9 tested, 0 stabilizer, 0 lifts | 0.6 s |
| `control-n-m2` | p = 3, Q(zeta_3), m = 2 -> 3 for \|N> = (1, 1, -2)/sqrt 6: the 48 rank-3 decompositions of \|N>^2 (closure of the 30 listed) must not lift, since chi(N^3) = 4; the negative control of `slice_lift.py`, exact | 2,985,984 patterns, 11,090 tested, 159 stabilizer patterns, no bijective triple, 0 lifts | 4.5 s |
| `cell --low-census` | the records, the exact k = 1..4 censuses, the closure, the exact re-decision, the search | 74 records, every unit run, k = 1..4 empty, one set, closure one set, 3,125 patterns, 25 tested, 0 stabilizer, 0 lifts; prints the m = 3 claim | 114 s (112 s of it the censuses) |
| `tests/test_t5_m3_lift.py` | the field against complex arithmetic for n = 3, 5, 9, the span solve, the stabilizer test, the basis lift, the rank-3 sets, the census set with the planted product and with \|T5>, the qutrit controls | 11 of 11 | 128 s |

`control-t3-full` (p = 3, m = 1 -> 2 through all 184 independent triples
of the 12 single-qutrit states, 168 M patterns before pruning) was started
and stopped after ten minutes; it is not in the chain. The positive case
it would cover is covered by `control-t3-basis`, and the pruning behavior
on a large family by `control-n-m2` and `control-rank3`.

On `control-n-m2`: 159 pattern vectors are stabilizer states but no three
of them form a bijection at every slice, so the lift fails at the matching
rather than at the stabilizer test, which is the failure mode
`slice_lift.py` reports numerically for the same cell.

## 5. Costs

Census. Already run: 13.4 pod CPU-hours on 2026-09-24 (1.05e-7 s per
squared member count on the 16-vCPU shared pod), 155,422 units, 74
records. Laptop rate re-measured on 2026-10-01 by re-running batch 73
(13,747 units, the smallest batch) from scratch through the compiled
kernel built from this checkout under a 900 s cap: 3,934 of the 13,747
units ran (3,906,815 modular candidates, 0 hits), at 4.9e-7 s per squared
member count over the units run, with the laptop at load 10 to 12 from
other sessions and the process at nice 19. That is 6.6 times the 7.4e-8
measured on 2026-09-24 on the same laptop and 4.7 times the pod's 1.05e-7;
the difference is the load, the scheduling at nice 19, or the build flags
of the scratch venv, and this note does not resolve it. The projection
that matters is the pod's own rate, which the chain's probe re-measures
before anything long starts. The pod chain re-runs
batch 73 the same way as its rate probe and, with `WITH_CENSUS=1`, every
batch again into a scratch directory with every deterministic hash
compared with the committed record (about 13 to 15 CPU-hours; 35 to 45
minutes of wall time at 24 workers on the 32-vCPU pod, if its per-core
speed matches the old pod's).

Lift stage. The search on the cell is 25 stabilizer tests, 0.05 s; the
command is dominated by reading the records and building the enumerator
(about 15 s) and the exact k = 1..4 censuses (about 150 s). The controls
in the chain total about two minutes. The certificate as the verifier
runs it is the listing aggregate (plan 2 s, low censuses 150 s, record
checks 10 s, two seeded batch re-runs of about 600 pod seconds each) plus
the cell command: about 25 minutes, under the 3,600 s budget it declares.

Kernel build on the pod. The kernel needs C++20 library headers that
gcc 9.4's libstdc++ lacks: `cpp/include/stabrank/linalg.hpp` and
`polynomial.hpp` include `<span>`, and `cpp/src/clifford.cpp`,
`polynomial.cpp`, and `fidelity.cpp` include `<numbers>`, both first
shipped with GCC 10. The chain's setup installs gcc-10 and g++-10 and
builds with `CC=gcc-10 CXX=g++-10 uv sync --extra challenge`, as the N^4
run did (`docs/notes/n4_rank6_exclusion.md`).

## 6. Soundness checklist

1. chi(psi_2) = 5 exactly. Closed: the Lean witness and the exact k =
   1..4 censuses (`aggregate.py --list` and `lift.py cell --low-census`
   both run them).
2. The listing is complete up to G. Closed by section 1.2 of the census
   note and the records (every unit run, no undecided), up to the modular
   caveat (item 8).
3. The family fed to the lift is the full G-closure of the listed sets.
   Closed: `orbit_closure` under the generators that
   `symmetry_orbit_reps` checks to fix psi_2 up to phase and to permute
   the dictionary; antiunitary symmetry not used.
4. The lemma's hypotheses: every alpha_z nonzero (asserted), p odd
   (asserted by the stabilizer test), the sets independent with nonzero
   coefficients (asserted by the exact solve).
5. The prunes drop no stabilizer pattern. Closed by the three facts of
   the proof (one modulus, slices are translates of one direction space,
   translates affine in z); the planted controls exercise non-constant
   patterns (basis, t3-basis) and constant ones (product).
6. The stabilizer test is exact and complete for odd p. Closed by the
   characterization (affine flat, one modulus, p-th-root ratios,
   quadratic exponents) and `control-stabtest`.
7. The lift matching finds every bijective r-set of stabilizer patterns.
   Closed: exhaustive backtracking within each group of patterns sharing
   the set sequence (E_z)_z; `control-n-m2` reaches this stage with 159
   patterns and none match, `control-basis` and `control-product` match.
8. The modular caveat of the census (a member dropped by a zero image
   modulo 65521 that is nonzero over Q(zeta_5)). Open by design, inherited,
   stated in the draft's attested note.
9. The certificate re-runs what the budget allows and hashes the rest.
   Closed: `aggregate.py --list --recheck 2 --recheck-seed 20261001` with
   the seed printed, the manifest hashed by the verifier before the script
   runs.
10. The projection to m = 4. Closed: every amplitude of |T5> is nonzero;
    the certificate prints the m = 4 line after the m = 3 one; a
    `T5-m4-lower-6` draft is not written here and would inherit the tier
    as `qubit_T-m6-lower-6.json` does.

## 7. The pod chain and the bound file

`research/t5_m3_lift/pod/pod_chain.sh`, launched by the user when the pod
is free (its header has the setup and launch lines). The branch
`t5-m3-lift` is local and never pushed, so it reaches the pod as a bundle
on top of the public main: `git bundle create /tmp/t5-m3-lift.bundle
origin/main..t5-m3-lift`, copied with `/tmp/podscp.sh`, fetched into a
fresh clone of the public repository, and checked out. The chain: kernel
import (`T5M3_KERNEL_OK`), the six census controls and seven lift
controls (`T5M3_CONTROLS_DONE`), the batch-73 probe with its hash compared
to the committed record and the rate projected (`T5M3_PROBE_DONE`),
optionally the whole census again with every hash compared
(`T5M3_CENSUS_DONE`), the listing aggregate with two seeded re-runs,
writing `research/t5q_m2_rank5/batch_manifest.json` (`T5M3_AGG_EXIT`),
`lift.py cell --low-census` (`T5M3_LIFT_EXIT`), `fill_draft.py` and the
certificate under `stabrank_verify.py` on the filled draft
(`T5M3_CERT_EXIT`), then `T5M3_SUMMARY` and `T5M3_CHAIN_DONE`. Every
marker is anchored at the start of a line and no waiting echo contains a
marker's text.

The bound file. `research/t5_m3_lift/T5-m3-lower-6.json.draft` is the
submission at the attested tier: `certificate.script` is
`cert_t5_m3_lift_attested.py` with `expect` `CERTIFIED chi(T5^3) >= 6`
and `budget_s` 3600; `attested.batches` is the census manifest the
listing aggregate writes, `recomputed` 2, `compute_hours` 13.4 with the
2026-09-24 pod as hardware, and the note says in the tier's words what
the completeness rests on. The tier is attested, not reproduced or
verified, because the enumeration behind the listing is the stored
census, which no budget re-runs; the lift decision itself runs in full
under the budget and is exact. `fill_draft.py` fills the date and the
lift stage's compute block from the stored records and validates the
result against the schema; moving it to `bounds/` is a hand step after
the chain, together with `batch_manifest.json` and the chain's records,
so that the board PR carries the manifest the verifier hashes.

## 8. What was run on the laptop

One process at a time at nice 19, 18-core Apple silicon laptop under load
from other sessions. The repository's `.venv` extension predates the
5-cover kernel (`cover5_pair` is absent from it), so a fresh venv was
built from this checkout in the session's scratch directory for the
batch probe; the lift stage needs no compiled code.

| step | command | time | result |
|---|---|---|---|
| exact stabilizer test | `lift.py control-stabtest` | 1.5 s | 40 of 40 |
| basis lift | `lift.py control-basis` | 10 s | the five Z(x)Z sectors, terms equal to the board's |
| rank-3 sets | `lift.py control-rank3` | 107 s | 10 sets, 0 lifts |
| planted products | `lift.py control-product` | 1.1 s | 3 of 3 recovered as the only lift |
| T3 basis | `lift.py control-t3-basis` | 0.3 s | the carry decomposition |
| T3 m = 2 | `lift.py control-t3-m2` | 0.6 s | 0 lifts |
| N m = 2 | `lift.py control-n-m2` | 4.5 s | 48 sets, 0 lifts |
| tests | `pytest tests/test_t5_m3_lift.py` | 128 s | 11 of 11 |
| the cell | `lift.py cell --low-census` | 114 s | 0 lifts, `CERTIFIED chi(T5^3) >= 6` printed |
| listing aggregate | `aggregate.py --list --dry-run --no-low-census` | 12 s | every stored check passes, 1 set listed, 13.34 kernel CPU-h recorded at 1.05e-7 s per M^2 |
| rate probe | `batch.py 73 --out-dir <scratch> --force --max-seconds 900` | 901 s | 3,934 of 13,747 units before the deadline, 3,906,815 modular candidates, 0 hits; 4.9e-7 s per squared member count over the units run |

## 9. Proved, measured, assumed, open

Proved in this note: the lemma at exact rank with its three-way case
split and the absence of invisible or coincident terms (section 1.2); the
pattern vector of the cell as a restriction of psi_3 and the hand
obstruction (the cubic part on every plane x + y = c_0 + d z).

Measured: the controls of section 4; the pattern counts (3,125, 25
tested, 0 stabilizer) and the search time; the batch-73 rate.

Assumed: the census's modular caveat (inherited); the exactness of the
dictionary's phase codes (asserted on the enumerator's output).

Open: a rank below 15 at m = 3 and below 25 at m = 4; excluding rank 6
at m = 3, which would need the invisible-term machinery and a census of
the rank-5 decompositions of psi_3 or a different route.

## 10. The pod run (2026-10-02)

The chain ran with `WITH_CENSUS=1` at 20 workers on the 32-vCPU pod
(AMD EPYC 9655P, Ubuntu 20.04, kernels built with gcc-10, uv's Python
3.13.5, numpy 2.4.4), commit b894242, log
`research/t5_m3_lift/results/pod/t5m3_chain.log`.

| stage | marker (UTC) | result |
|---|---|---|
| kernel import | 01:30:02 | `cover5_pair` imports |
| six census controls, seven lift controls | 01:36:35 | every one rc = 0 (`results/pod/*.log`, records under `results/` and `research/t5q_m2_rank5/results/control_*.json`) |
| batch-73 probe | 01:43:21 | 13,747 units, 6,012,090 modular candidates, 0 hits, 404 kernel s, 1.18e-7 s per squared member count, deterministic hash equal to the committed record |
| census re-run, 74 batches | 02:32:26 | 155,422 units, 2,795,734,903 modular candidates, 1 hit (batch 31, the sector set), 0 undecided, 14.49 CPU-hours (14.57 kernel hours, 1.15e-7 s per squared member count), 49 minutes of wall time; the deterministic hash of every one of the 74 records equals the committed one (`results/pod/census/`) |
| listing aggregate | 02:54:36, exit 0 | plan equal, k = 1..4 censuses empty, 74 records consistent, hit re-decided, re-runs of batches 12 and 47 (seed 20261001) with equal hashes (591 s, 571 s), manifest written, `LISTING COMPLETE: 1 decomposition 5-set(s)`; 1,329 s |
| `lift.py cell --low-census` | 02:57:24, exit 0 | 3,125 patterns, 25 tested, 0 stabilizer, 0 lifts, `CERTIFIED chi(T5^3) >= 6`; 168 s |
| `fill_draft.py`, then `stabrank_verify.py` on the filled bound | 03:19:38, exit 0 | `PASS T5-m3-lower-6.json chi(\|T5>^3) >= 6, tier=attested`, 2 of 74 batches re-run from seed 20261001; 22 minutes |

What the census re-run compared. The deterministic part of a record is
the batch geometry, its units, the dictionary, plan, and partition hashes,
the kernel's member count for every unit, every hit with its flags and
both decisions, and the undecided list; the candidate count, the kernel
run count, timing, host, and version fields sit outside it. All 74
deterministic hashes agree with the 2026-09-24 records, and the modular
candidate count agrees too (it is seeded). What differed between the two
runs: the pod (16-vCPU EPYC 4564P against 32-vCPU EPYC 9655P), the
Python and numpy versions (3.12.14 and 2.3.5 against 3.13.5 and 2.4.4),
the kernel binary (the 5-cover source was refactored after the first run
in 561c82a and compiled here with gcc-10), and the worker count. What was
shared: `batch.py`, `common.py`, the partition, the dictionary enumerator,
and the kernel's functional seed. The re-run is recorded in the bound's
compute block and does not change the tier: the tier is about the budget,
and a second run of the same enumeration is still an enumeration no
certificate re-runs in full.

The bound files. `bounds/T5-m3-lower-6.json` (attested to
`research/t5q_m2_rank5/batch_manifest.json`, 74 entries, `recomputed` 2;
compute block: 28.9 CPU-hours over both census runs and the chain, 2.9
hours of wall) and `bounds/T5-m4-lower-6.json` (projection, same
certificate, inherits the tier). The board's T5 column is now 3, 5, 6 to
15, 6 to 25.
