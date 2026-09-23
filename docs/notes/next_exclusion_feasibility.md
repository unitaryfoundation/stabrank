# The next lower-bound exclusion on the pod: feasibility and design

Status (2026-09-23). Design and costing only; nothing here is a bound.
Three candidates were compared for the next 16-vCPU run: the rank-5
exclusion of |T>^6 (A), the rank-8 exclusion of |T3>^4 (B), and the
rank-5 exclusion of |H>^5 (C). The pick is C: it reuses the H^6 census,
its degenerate lists, and both compiled kernels without change, its two
structural facts are proved by tables that exist and were recomputed here
in a few minutes, its one new case (an invisible line term) is a small
prototype that recovers planted instances, and the whole run projects to
about 25 CPU-hours at the rates measured on this laptop (about 45 on the
shared pod by the H^6 experience), two to three hours of wall time on
15 processes. A is not what the task assumed: the board has
4 <= chi(T^6) <= 6, rank 4 is open at m = 5 and m = 6, and property P does
not transfer to T because chi(T^4) = 3; the cheap prerequisite, excluding
rank 4 at |T>^5, is a laptop job once the exact arithmetic supports the
T-state field, and it is sketched in section 2. B fails on the census
alone: the two-qutrit base slices of a rank-8 decomposition of |T3>^4 are
about 6.7e13 eight-subsets of the 360 two-qutrit states on a common
hyperplane through the Galois space, seven orders of magnitude above the
H^6 census, and the base-point argument has no Fact B to rest on.

Everything measured below ran single-process at nice 19 on an 18-core
laptop at load average 35 to 80 (other sessions), under a 10-minute cap
per run, so every rate overstates an unloaded core. Scripts:
`research/h5_rank5/prototype.py` (the C measurements and the
invisible-line prototype) and the probes named in section 6.

Notation follows `docs/notes/h6_rank5_exclusion.md`: psi_m = |H>^m with
|H> = cos(pi/8)|0> + sin(pi/8)|1>, t = tan(pi/8), N_3 = 1080 three-qubit
stabilizer states in the order of `dictionary(2, 3)`, G_3 the unitary
symmetry group of psi_3 (order 48). Slicing a decomposition of psi_m
along n_1 qubits at x in F_2^{n_1} gives sum_i c_i u_i^(x) = alpha_x
psi_{m - n_1} with alpha_x = cos(pi/8)^{n_1 - |x|} sin(pi/8)^{|x|}, and
the slice ratio between x and x_0 is t^{|x| - |x_0|}. For the T-state,
|T> = cos(beta)|0> + e^{i pi/4} sin(beta)|1> with cos(2 beta) = 1/sqrt 3,
so the ratio a_1 / a_0 = e^{i pi/4} tan(beta) with tan(beta) =
sqrt(2 - sqrt 3) = (sqrt 3 - 1)/sqrt 2.

## 1. The three cells

| candidate | cell | board | files | what the exclusion gives |
|---|---|---|---|---|
| A | qubit_T m = 6, rank 5 | 4 <= chi <= 6 | `qubit_T-m6-lower-4.json` (projection of m = 5), `qubit_T-m6-upper-6.json` (Lean) | nothing by itself: rank 4 is open, so the lower bound stays 4 |
| A' | qubit_T m = 5, rank 4 | 4 <= chi <= 6 | `qubit_T-m5-lower-4.json` (slice and lift), `qubit_T-m5-upper-6.json` (Lean) | chi(T^5) >= 5 and, by projection, chi(T^6) >= 5 |
| B | T3 m = 4, rank 8 | 8 <= chi <= 9 | `T3-m4-lower-8.json` (projection of the attested m = 3 exclusion), `T3-m4-upper-9.json` (Lean) | chi(T3^4) = 9 |
| C | qubit_H m = 5, rank 5 | 5 <= chi <= 6 | `qubit_H-m5-lower-5.json` (slice and lift), `qubit_H-m5-upper-6.json` (Lean) | chi(H^5) = 6 |

The task's premise for A, "qubit_T m = 6 currently 5 <= chi", is not what
the board says: `qubit_T-m5-lower-4.json` excludes rank 3 at m = 5 (every
rank-3 decomposition of |T>^4 fails to lift) and `qubit_T-m6-lower-4.json`
is its projection. No rank-4 exclusion exists at either cell. A rank-5
exclusion at m = 6 would therefore have to exclude ranks 4 and 5 together,
or be preceded by A'.

## 2. Candidate A: the T-state cells

### 2.1 Property P does not transfer

Property P for H^6 (`docs/notes/h6_rank5_exclusion.md`, section 1) says
that in a rank-5 decomposition of psi_6 every four-qubit slice along every
2 + 4 bipartition has all five terms nonzero. Its proof: chi(H^4) = 4, so a
slice point with fewer than four visible terms is impossible, and a slice
point with exactly four is one of the 30 minimal decompositions of |H>^4,
none of which extends (the tables of `research/constructions/two_qubit_slice.py`, PR #87).
Property P is what makes the counting lemma work (every term's dual code
has distance at least 3, so some triple sees all five terms), and the
counting lemma is what makes the enumerated bases complete.

For T the first step fails: chi(T^4) = 3, so a 2 + 4 slice point of a
rank-5 decomposition of |T>^6 can have three, four, or five visible terms.

Three visible terms. They form the unique minimal decomposition of |T>^4
(`research/constructions/data/qubit_T_m4_rank3.json`, one class), and the
two invisible terms are on flats of F_2^2 missing x_0. The residual tables
of `two_qubit_slice.ratio_tables` for that decomposition (all 65^3 code
combinations of the three visible terms, `t_probe.py ratios qubit_T 4 2`,
3 s):

| ratio (a_1 / a_0)^j | exact combinations | stabilizer-residual combinations |
|---|---|---|
| j = -2, -1, 1, 2 | 0 | 0 |
| j = 0 | 1 | 192 |

So a point at ratio different from 1 needs both invisible terms present
(k = 2). With x_0 = 00 or 11 all three other points have ratio different
from 1, and two terms present at three points of F_2^2 would be present on
a non-flat: impossible. With x_0 = 01 the points 00 and 11 need both
invisible terms, so both are line terms on {00, 11}, and the point 10
(ratio 1) is exact. The three-visible case therefore reduces to one
configuration, decidable by a rank-2 residual test at 00 and 11 over the
36,720 four-qubit states (`residual_table` with `rank2="exact"`), not run
here.

Four visible terms. They form a rank-4 decomposition of |T>^4 that is not
minimal, and no list of those exists: they are the full 4-covers of |T>^4
over 36,720 states, an enumeration two orders of magnitude beyond the
1080-state censuses (the pivot pairs alone are about 3.5e5 with M near
36,720). This is the blocker. The invisible term is absent at x_0 and at
some second point x_1, where the same four terms are visible with Pauli
translates and ratio (a_1 / a_0)^{|x_1| - |x_0|}. When |x_1| - |x_0| = +-1
the two slices are a rank-4 decomposition of |T>^5: the sliced factor is
|0> + (a_1 / a_0)|1> = |T> up to normalization, or |0> + (a_0 / a_1)|1>,
which is X|T>, so A' excludes both. When |x_1| = |x_0| (the invisible
term is a line term on {00, 11} and x_0, x_1 are 01, 10) the two slices are
a rank-4 decomposition of |+> (x) |T>^4, which exists trivially, so no
projection argument removes this configuration. When |x_1| - |x_0| = +-2
the sliced factor is cos^2(beta)|0> + i sin^2(beta)|1>, a non-stabilizer
state whose four-copy product with |T> has unknown rank; A' run with that
factor as the sliced qubit would decide it (the census does not change,
only the slice ratios).

Conclusion: even after A', a term of a rank-5 decomposition of |T>^6 may
be a diagonal-line term along a pair of qubits, so its dual code may have
weight-2 words, and the counting lemma of the H^6 note does not apply. The
T^6 exclusion needs either a new all-visible lemma or a matcher that
carries invisible terms (the route section 4.4 builds for one invisible
line at m = 5, extended to F_2^3 with up to two invisible terms: a
4-cover base with one invisible flat at 4 x 35 flats per cover, a 3-cover
base with two). It is not the next pod run.

### 2.2 A': excluding rank 4 at |T>^5 (a laptop job, once the field is there)

Fact (every term full along every qubit). A one-qubit slice of a rank-4
decomposition of |T>^5 has at least chi(T^4) = 3 visible terms. With
exactly three, they are the unique minimal decomposition of |T>^4 and the
fourth term is local, present at the other value k' with a stabilizer
slice; the equation at k' is then a k = 1 row of the table above at ratio
(a_1 / a_0)^{+-1}, of which there are none. So every one-qubit slice has
all four terms visible.

Base slice along qubits 1, 2 (n_1 = 2, n_2 = 3). By the fact, every term's
flat in F_2^2 is the plane or a diagonal line ({00, 11} or {01, 10}); by
chi(T^3) = 3 every point has at most one absent term, so at most one line
term per diagonal. If no line term is on {01, 10}, x_0 = 00 sees all four;
if none is on {00, 11}, x_0 = 01 does. If there is one line term on each
diagonal, x_0 = 00 sees three (two planes and the {00, 11} line), a rank-3
decomposition of |T>^3, and the point 11 is exact at ratio (a_1 / a_0)^2
with the same three terms present; the tables for the 8 stored rank-3
decompositions of |T>^3 (`t_probe.py ratios qubit_T 3 2`) have 0 exact
combinations at j = +-2, so this configuration does not occur. Hence
every rank-4 decomposition of |T>^5 has an all-visible base at x_0 = 00 or
x_0 = 01, and the base is a full 4-cover of |T>^3.

Census. The numeric enumerator `slice_lift.all_decompositions("qubit_T",
3, 4)` lists 5,205 rank-4 decompositions of |T>^3 up to symmetry in 1.6 s
(its list is a superset of the full covers, as at H where it lists 3,731
against 3,460 full ones). The degenerate 4-multisets over the rank-3
covers are a few (for H, 6); no cancel-at-base multiset exists among four
visible terms, since the other two would be a 2-cover of |T>^3.

Cost. At the measured stage A rate (section 4.4) about 5,000 covers at
two base points is seconds compiled and a minute in Python, plus the
degenerate handful.

Blocker, code only. `verify_challenge/slice_cover.py` hardcodes |H>:
`Field` is Q(zeta_16) with cos(pi/8) and sin(pi/8), `CoverEnumerator`
takes `psi_for("qubit_H", n)` and the qubit_H symmetry group,
`SliceMatcher.rhs` uses tan(pi/8), and `confirm_decomposition` builds the
qubit_H target. The T amplitudes divided by cos(beta)^m lie in
Q(zeta_24) (tan(beta) in Q(sqrt 2, sqrt 3), the phase e^{i pi/4}); both
primes 65521 and 2013265921 are 1 mod 24, so the same two primes serve
with a Q(zeta_24) `Field`. The compiled kernels take the modular target
slices as inputs and are orbit-agnostic. The change is confined to the
four places named, plus the orbit as a constructor argument.

## 3. Candidate B: rank 8 at |T3>^4

Fact A is free: a one-qutrit slice of a rank-8 decomposition of |T3>^4 is a
decomposition of |T3>^3 with at most 8 terms and at least chi(T3^3) = 8
(`bounds/T3-m3-lower-8.json`, attested), so every term is full along every
qutrit. Fact B is not available: a two-qutrit slice point has at least
chi(T3^2) = 3 visible terms and up to five invisible, and the residual
pruning of PR #86 is empty at this rank (`docs/notes/constructions_2026_09.md`:
"three visible and five invisible terms cover every slice"). Along a
qutrit pair every term is a plane or a diagonal-line term, and with line
terms on two distinct parallel lines no point sees every term; the best
point sees planes plus max(n_a) plus max(n_d) over the three lines of each
diagonal direction, so up to four terms can be invisible at the best point
(six line terms, one per line). The base-slice argument would have to carry
up to four invisible terms of unknown slices.

The census kills it independently. An 8-cover of |T3>^2 has a
Q(w_3)-rational span containing psi_2 and hence its Galois conjugates, so
its eight states lie in a hyperplane of C^9 containing V_2 =
span{t_1^2, t_4^2, t_7^2}, a hyperplane of the 6-dimensional quotient.
Three of the 360 two-qutrit states lie in V_2 itself (they are the unique
rank-3 decomposition, consistent with chi(T3^2) = 3) and belong to every
such hyperplane. Sampling 40,000 random 5-subsets of the images
(`t3_probe.py`, 10 s) and weighting each hyperplane by 1 / C(n_H, 5):

| states on the hyperplane n_H | estimated hyperplanes |
|---|---|
| 8 | 2.5e6 |
| 9 to 15 | 4.7e6 |
| 16 to 30 | 5.6e5 |
| 31 to 51 | 2.1e4 |
| 55, 63, 65, 80, 87 | 613, 106, 416, 87, 126 |
| 136 | 22 |

Estimated 8-subsets on a common hyperplane, the stage A census before
fullness and independence: 6.7e13 (the 22 hyperplanes with 136 states
contribute 22 x C(136, 8) = 3e13 on their own; 7-subsets 5.1e12). The
H^6 census was 5.9e6 covers with a further 2.8e4 degenerate ones. No
symmetry reduction (a factor of at most a few hundred from the symmetry
group of |T3>^2) and no kernel closes seven orders of magnitude, and the
1 + 3 route (8-covers of
|T3>^3 over 30,240 states) is worse. The Galois scan that settled m = 3
(seven independent states with images spanning four dimensions of a
24-dimensional quotient, 1.3e13 steps) becomes eight states spanning five
dimensions of a 78-dimensional quotient over 7,439,040 states at m = 4,
also out of reach. B is not a pod run.

## 4. The pick: rank 5 at |H>^5

### 4.1 The argument

Suppose psi_5 = sum_{i=1}^5 c_i s_i with all c_i nonzero. Slice along
qubits 1, 2 (n_1 = 2, n_2 = 3); the copy symmetry makes this pair general,
and there is no monomial symmetry of |H>, so x_0 is not reducible beyond
the swap of the two sliced qubits (01 and 10 are equivalent).

Fact 1 (every term is full along every qubit). A one-qubit slice of the
decomposition has at least chi(H^4) = 4 visible terms. With exactly four
at value k, the four states are distinct and independent (otherwise psi_4
lies in the span of at most three stabilizer states), so they are a
minimal decomposition of |H>^4, one of the 30 stored ones up to the
unitary symmetry of psi_4 (which acts on the unsliced qubits and preserves
slices), with c_i = alpha_k d_i; the fifth term is local, |k'> (x) v with
v a four-qubit stabilizer state, and the equation at k' says that
alpha_{k'} psi_4 - sum_i c_i (Pauli translate of u_i) is proportional to
a stabilizer state, allowing absent translates. That is a k = 1 row of the
residual tables of PR #87 at ratio t^{+-1}, and those tables are empty.
Recomputed here for all 30 decompositions (`two_qubit_slice.py
--ratio-tables 1`, 65^4 = 17.8 million code combinations per decomposition
and ratio, about 10 s each, 5.2 minutes in all): every decomposition has
(exact, stabilizer) = (0, 0) at j = -1 and at j = 1, and (1, 266 to 360)
at j = 0. Fact 1 follows: no term is absent at any one-qubit slice point,
so no term is a product across any qubit.

Fact 2 (at most two absent terms at any two-qubit point): chi(H^3) = 3
(`bounds/qubit_H-m3-lower-3.json`).

Lemma (flat types). By Fact 1 the flat of every term along qubits 1, 2
projects onto each coordinate, so it is the plane or one of the two
diagonal lines {00, 11}, {01, 10}. By Fact 2 at most two terms are line
terms on each diagonal. Let n_A and n_B be their numbers on {00, 11} and
{01, 10}.

Base point. If n_B = 0, x_0 = 00 sees all five terms. If n_A = 0 and
n_B >= 1, x_0 = 01 does. If n_A >= n_B >= 1, take x_0 = 00: the planes and
the {00, 11} lines are visible, n_B in {1, 2} terms are invisible. If
n_B > n_A >= 1 take x_0 = 01 with n_A = 1 invisible. The case n_A = n_B =
2 has one plane and two lines on each diagonal; at x_0 = 00 three terms
are visible, they are a rank-3 decomposition of |H>^3 (distinct and
independent by chi(H^3) = 3), and at 11 the same three are present, the
two {01, 10} lines absent, and the ratio is t^2: the residual tables of
the 6 stored rank-3 decompositions of |H>^3 (`t_probe.py ratios qubit_H 3
2`, 0.4 s) have 0 exact combinations at j = +-2, so this case does not
occur. What remains:

- Stage (alpha): an all-visible base at x_0 in {00, 01}; the base is a
  full 5-cover of psi_3 (five states, repeats allowed, with coefficients
  d_i = c_i / alpha_{x_0} all nonzero), exactly the object of the H^6
  census, and the other three slices are given by the structure lemma with
  n_1 = 2 (two coordinate slices at ratio t^{+-1}, one composite slice at
  ratio t^{+-2} or 1).
- Stage (beta): four visible terms at x_0 in {00, 01} (three planes and a
  line on the diagonal through x_0, or two and two) and one invisible line
  term on the diagonal missing x_0. The base is a full 4-cover of psi_3.
  The point x_0 + 11 has the four visible terms present and the invisible
  one absent, so it is exact at ratio t^{+-2} (x_0 = 00) or 1 (x_0 = 01).
  At the two coordinate points the visible planes are present, the visible
  line is absent, and the residual is c_5 v at one and c_5 i^l Q v at the
  other, with v a three-qubit stabilizer state and Q a Pauli: the residual
  must be a nonzero multiple of a dictionary state at each, and the two
  residuals must be Pauli translates of each other with the same
  coefficient.

The matcher of stage (alpha) does not assume Facts 1 and 2 (it allows all
five subspaces of F_2^2 for every term, as at H^6); Facts 1 and 2 only
guarantee that the enumerated bases are complete. The prototype of stage
(beta) does use the flat lemma at the second coordinate point (a term
present at the first coordinate point is taken to be a plane, its code at
the second fixed by composition up to a sign); the pipeline version should
allow the absent option there too, which costs one more option per term,
so that stage (beta) also assumes nothing beyond the base point.

### 4.2 Enumeration: nothing new

The stage (alpha) bases are the full 5-covers of psi_3, the same set for
any n_1. The H^6 census stands: 5,939,465 covers of distinct independent
states over 14,280 pivot pairs (`research/h6_rank5/results/kernel_census.json`,
459 s compiled) and the degenerate list `degenerate_covers_v2.json`
(28,396 covers: 12,390 dependent of pattern (1, 1, 1, 1, 1), all kappa =
1; 15,994 of pattern (2, 1, 1, 1), including the 2,154 cancel-at-base
multisets T + (b, b); 6 of (2, 2, 1); 6 of (3, 1, 1)). Both are hashed and
would be loaded by hash as at H^6; the aggregate re-enumerates the
degenerate list.

The stage (beta) bases are the full 4-covers of psi_3: 3,460 of distinct
independent states (`CoverEnumerator.covers(4)`, 146 s in Python here,
cross-checked at H^6 against the numeric enumerator) and 6 degenerate
multisets of pattern (2, 1, 1) (a 3-cover with one state doubled; span(T)
holds no further state for either 3-cover class, so no dependent 4-set of
distinct states exists, `beta_degenerate.py`). There is no cancel-at-base
multiset with four visible terms: two cancelling copies leave two states
covering psi_3, against chi(H^3) = 3.

### 4.3 Exactness

As at H^6: everything is in Q(zeta_16), both primes are 1 mod 16, the
modular kernels and the meet-in-the-middle hashes list supersets, every
candidate is re-decided mod 2013265921 and over C, and every hit is
confirmed as a decomposition of psi_5 (residual, rank, independence,
nonzero coefficients, the span test mod 2013265921 agreeing with the
numerical one). Stage (beta)'s stabilizer-residual test is a lookup of the
normalized residual's phase codes in the 1080-state table after a modulus
and support check; it is a floating-point decision that prunes, so the
pipeline version should re-decide the survivors' equations mod
2013265921, as `restrict` does for stage (alpha).

### 4.4 Matcher and measured cost

Stage (alpha) runs `slice_cover.SliceMatcher(E, 2)` unchanged: the class
is generic in n_1 (the H^6 control-m4 ran it at n_1 = 1), and the compiled
`SliceMatchKernel` accepts n_1 in 1..3 and n_2 in 1..3, so the native stage
A path is available. Measured (`prototype.py sampleA`, 200 covers from
the first pivot, both base points):

| path | per (cover, x_0) at 00 | at 01 | max | coordinate-slice solutions (400 runs) |
|---|---|---|---|---|
| native | 0.60 ms mean, 0.35 median | 0.52 ms mean, 0.33 median | 19 ms | 0 in 305 runs; (1, 1) 57; (2, 4) 15; (4, 16) 10; (9, 81) 5; up to (12, 144) |
| reference (40 covers) | 28.8 ms mean | 8.7 ms mean | 148 ms | the same histogram on its 80 runs |

No hit. Three quarters of the runs die at the first coordinate slice, as
at H^6.

Stages B and C run the Python reference (`prototype.py sampleBC`, four
covers evenly spaced through each pattern, two base points each, none
capped, no hit, no state reached a block reconstruction):

| pattern | covers in the list | per (cover, x_0) mean | median | max |
|---|---|---|---|---|
| (1, 1, 1, 1, 1), kappa = 1 | 12,390 | 3.02 s | 3.35 s | 4.47 s |
| (2, 1, 1, 1) | 15,994 | 0.15 s | 0.02 s | 1.11 s |
| (2, 2, 1) | 6 | 1.71 s | 0.69 s | 7.56 s |
| (3, 1, 1) | 6 | 0.12 s | 0.01 s | 0.79 s |

Stage (beta) is the prototype `BetaMatcher` in `prototype.py`: the
coefficients are the unique solution over the four base states; the exact
point x_0 + 11 is solved by meet in the middle over 32 options per term
(the 8 Pauli classes times 4 phases, no absence) on a random functional,
every collision decided on the whole 8-vector; for each solution the
first coordinate point is scanned over all 33^4 = 1,185,921 option
combinations at once (numpy), the residuals normalized and looked up in
the sorted code table of the 1080 states; for each stabilizer residual the
second coordinate point's codes follow from the structure lemma (a term
absent at the first point is absent at the second, a present term's class
is the quotient of its 11-class by its first class and its phase is fixed
up to a sign), 16 combinations at most; the two residuals must be Pauli
translates with a fourth-root phase and equal coefficient; every survivor
is assembled into five terms and confirmed (residual, rank 5, nonzero
coefficients, every term a phase pattern with power-of-two support).
Measured (`prototype.py beta --count 30 --plant 4`):

| base point | 11-slice ratio | per cover | 11-solutions | residual candidates | stabilizer residuals | pairs | hits |
|---|---|---|---|---|---|---|---|
| 00 | t^2 | 0.010 s mean, 0.023 max | 0 in 30 covers | 0 | 0 | 0 | 0 |
| 01 | 1 | 0.51 s mean, 0.29 median, 1.62 max | 47 over 30 covers | 5.6e7 | 467 | 6,936 | 0 |

The planted control: four random five-term configurations of the (beta)
shape (three plane terms, one line on {00, 11}, one line on {01, 10},
random three-qubit base states, Pauli classes, phases, and small
coefficients, sum taken as the target), each recovered from its base at
x_0 = 00 as the unique hit in 0.4 s.

Witness control (the counterpart of the H^6 note's control 2). The rank-6
witness `bounds/qubit_H-m5-upper-6.json` has 24 distinct all-visible
(base, x_0) pairs over the ten qubit pairs: 22 with six distinct
independent states (kappa = 0, the native path), one with a repeated state
(kappa = 1, base 4), one with six distinct dependent states (kappa = 2,
base 5). Its flat types: planes everywhere except one line term along the
pair (0, 4) and two line terms on the same diagonal along (1, 2), so the
witness exercises stage (alpha) and not stage (beta), which is why the
(beta) control is planted. `prototype.py witness`: 23 of the 24 bases
return the witness itself among their genuine rank-6 decompositions (1 to
6 per base, 0.1 to 1.2 s native; base 4 in 36 s through the block path
with coordinate solutions 1,181 and 14,731 and 3,515 joined states); base
5 aborts at the 2,000,000-candidate cap of the dense 2-parameter solve at
its first coordinate slice after 28 s. A 2-parameter family of six
distinct states is a rank-6 shape and is on no rank-5 path (stage B has
five states and kappa = 1); the H^6 control had the same shape and passed
within 373 s, so raising the cap for the control alone would probably
finish it, and the pipeline should record the abort rather than treat the
base as covered.

### 4.5 Projected cost

At the rates above, two base points per cover, and the H^6 partition
scheme (about 600 s per batch):

| stage | runs | rate | CPU-hours (laptop rates) | batches |
|---|---|---|---|---|
| 5-cover kernel | 14,280 pivot pairs | census seconds | 0.13 | with stage A |
| A, native | 5,939,465 x 2 | 0.56 ms | 1.85 | about 12 |
| B, reference | 12,390 x 2 | 3.0 s | 20.8 | about 125 (100 runs each) |
| C, reference | 16,006 x 2 | 0.15 s ((2, 2, 1) at 1.7 s) | 1.4 | about 8 |
| (beta), independent | 3,460 x 2 | 0.01 s at 00, 0.51 s at 01 | 0.5 | 1 |
| (beta), degenerate | 6 x 2 | unmeasured (needs the pair block) | small | with (beta) |
| total | | | about 25 | about 150 |

The H^6 run saw stage B at 2.3 times the laptop sample rate on the shared
pod (20.2 s per cover against 8.9), so plan on 45 to 50 CPU-hours, three
to three and a half hours of wall time on 15 processes, dominated by stage
B. Compiling the 1-parameter dense solve (the 2 x 2 Laplace features and
the 39-million-pair product, item 4 of the H^6 note's "what remains")
would cut stage B by roughly the stage A factor and bring the whole run
under 5 CPU-hours; it is not needed to run.

### 4.6 Degenerate lists

- Stage B: the 12,390 dependent covers of five distinct states (kappa = 1),
  as listed in `degenerate_covers_v2.json`. Unchanged.
- Stage C: the 16,006 covers with a repeated state, including the 2,154
  cancel-at-base multisets T + (b, b) with b outside span(T), which
  property-P-type arguments do not exclude and the H^6 stage C repair
  added. Unchanged.
- Stage (beta): the 6 multisets of pattern (2, 1, 1) over the two 3-cover
  classes; no dependent 4-sets, no cancel-at-base multisets (section 4.2).
  The prototype's point-family path does not cover them: the pair block
  contributes an arbitrary vector of the span of at most two Pauli
  translates of its state at every slice, so the exact 11-slice becomes a
  projected equation as in `slice_cover` and the residual test at the
  coordinate points must allow the block's contribution. Six bases at two
  base points can equally be handed to a direct search: two copies with
  coefficients summing to the merged value, 33 options each at the three
  slices, joined with the two ordinary terms' options by meet in the
  middle, then the same residual and pairing tests.

### 4.7 Soundness checklist (from `docs/notes/qutrit_m4_rank5_review.md` and the H^6 stage C repair)

1. The stage C list carries the cancel-at-base multisets T + (b, b) (it
   does, v2), and the aggregate's fresh enumeration equals the list.
2. `reconstruct_block` allows classes outside the translate set used by two
   copies with net coordinate zero (fixed 2026-09-23) and restricts a
   copy's composite codes to the structure lemma's shapes.
3. A state with a block that reaches the final loop with an unpinned
   family, or with dependent block translates, raises `UnpinnedFamily`;
   the batch records the cover as undecided and the aggregate fails.
4. A refused cover (an ordinary state dead on the whole family) is a
   list-matcher disagreement and fails the aggregate.
5. Floating-point pruning (`has_zero_coefficient`, `is_full`, the fit
   tolerance of `restrict`, `_refine_split`) is backed by the modular
   checks that sit next to it; stage (beta)'s residual lookup needs the
   same exact re-decision of its survivors before it is trusted.
6. Every hit is re-decided from its phase codes mod 2013265921 and
   numerically (`common.decide_terms`), a disagreement is undecided, and
   the aggregate re-decides every stored hit.
7. The deterministic hash of a batch record excludes the modular candidate
   count (the H^6 `--no-native` cross-check differed only there); one
   stage A batch is replayed with `STABRANK_NO_NATIVE=1` and the exact
   content compared.
8. Every batch record carries the partition and list hashes; no batch
   survives a repartition.
9. `Matcher._compatible`-style caches keyed by object id are not used
   (the H^6 `slice_cover` has no such cache; the qutrit matcher had one).
10. Batches run with `--max-seconds`; a deadline abort records the covers
    not run as undecided rather than running for hours.
11. The matcher assumes nothing beyond the base point (stage (alpha)
    already; stage (beta) after the absent option is added at the second
    coordinate point, section 4.1).
12. Controls re-run after any matcher change, and the witness control's
    base 5 recorded as aborted, not passed.

### 4.8 Controls the pipeline needs

1. The witness control above, all 24 bases, with base 5 either finished
   under a raised cap or recorded as aborted.
2. The m = 4 control at n_1 = 2: slice the 30 stored rank-4 decompositions
   of |H>^4 along a qubit pair (base a 4-cover of |H>^2 over the 60
   two-qubit states, every all-visible base) and recover every class,
   nothing outside the stored list. This exercises the n_1 = 2 geometry
   with hits, as control-m4 did for n_1 = 1; the bases are dependent
   (four states in C^4 can be independent, so both paths appear).
3. Planted stage (beta) instances, as run (4 of 4), extended to bases with
   a repeated state once the pair block is written, and to instances where
   a visible term is absent at the second coordinate point (a coordinate
   line), which the matcher must reject or find but never mis-assign.
4. One stage A batch replayed with `--no-native`.
5. The stage B rate measured on the pod's first batch before the partition
   is trusted (H^6: 2.3 times the laptop rate).

### 4.9 Pipeline plan

`research/h5_rank5/` mirroring `research/h6_rank5/`: `common.py` with
N1 = 2, M = 5, RANK = 5, the two base points, and the census and
degenerate-list paths pointing at the H^6 files by hash; `driver.py`
with `partition` (stage A by pivot pairs at 0.56 ms per cover and two base
points, stages B and C round-robin at the sampled rates, stage (beta) as
one or two batches over the 4-cover list, which the driver enumerates and
hashes), `control-witness`, `control-m4-pair`, `control-beta`, `sample`;
`batch.py K` with hashed deterministic records excluding the candidate
count; `aggregate.py --recheck 2`; the certificate
`verify_challenge/cert_qubit_h_m5_rank5_attested.py` printing `CERTIFIED
chi(qubit_H^5) >= 6`; the draft bound `qubit_H-m5-lower-6.json.draft`
declaring the dependencies: Fact 1 through PR #87's tables (recomputed
here), Fact 2 through `qubit_H-m3-lower-3.json`, the t^{+-2} tables for
the rank-3 decompositions of |H>^3, the flat lemma and the base-point case
split of section 4.1, the H^6 census and degenerate lists by hash, the
cancel-at-base multisets, the modular supersets with exact re-decision,
and the controls. Pod commands as in the H^6 README (15 processes through
xargs, resumable, `--max-seconds 3600` on the shared batches, then
`aggregate.py --dry-run --partial`, then `--recheck 2`).

### 4.10 What is proved, what is assumed, what is open

Proved by table or argument in this note: Fact 1 (30 decompositions,
(0, 0) at j = +-1, recomputed), Fact 2 (a board bound), the flat lemma,
the base-point case split, the exclusion of the (2, 2) configuration
(0 exact at j = +-2 for the 6 rank-3 decompositions of |H>^3), the absence
of cancel-at-base multisets among four visible terms, the reuse of the
census. Assumed from earlier work: the completeness of the 30-element
rank-4 list of |H>^4 up to its symmetry (the same assumption the H^6 bound
makes), the H^6 census and degenerate enumeration (hashed, re-enumerable),
`two_qubit_slice.py`'s shape enumeration (its own controls). Open before
the run: the pair block for the 6 degenerate (beta) bases; the absent
option at the second coordinate point in stage (beta); the exact
re-decision of stage (beta) survivors; the witness control's base 5; the
pod rate of stage B.

## 5. Verdict

Run C. It closes an open board cell (chi(H^5) = 6 if no hit) with the
machinery that closed H^6, at about half the H^6 run's cost, and every
structural input is either a board bound or a table that takes minutes.
A' is worth doing on the laptop afterwards for its own sake (it lifts
qubit_T m = 5 and m = 6 to 5 <= chi <= 6), once `slice_cover` takes the
orbit and its field as parameters; A proper and B are not pod runs in
their present form, for the reasons of sections 2.1 and 3.

## 6. What was run

All at nice 19 on the loaded laptop, through a wrapper with a 600 s cap
(`start_new_session`, `killpg`), about 20 minutes of CPU in all.

| step | command | time | result |
|---|---|---|---|
| environment | `prototype.py` imports | 8 s | `stabrank_core` provides `cover5_pair` and `SliceMatchKernel`; `SliceMatcher(E, 2)` binds the native kernel |
| B census sample | `t3_probe.py 40000` | 10 s | section 3 table; 6.7e13 eight-subsets |
| witness bases | `prototype.py witness --list` | 1 s | 24 bases, flat types |
| witness control | `prototype.py witness --cap 120` | 72 s | 23 of 24 recover the witness; base 5 aborts at the dense cap |
| stage A rate | `prototype.py sampleA --count 200`, `--count 40 --reference` | 4 s each | 0.56 ms native, 19 ms reference per (cover, x_0) |
| stages B, C | `prototype.py sampleBC --count 4 --cap 40` | 42 s | 3.0 s, 0.15 s, 1.7 s, 0.12 s per run by pattern |
| stage (beta) | `prototype.py beta --plant 4 --count 30` | 165 s (146 s of 4-cover enumeration) | 4 of 4 planted recovered; 0.01 s and 0.51 s per cover |
| (beta) degenerates | `beta_degenerate.py` | 1 s | 6 multisets, all (2, 1, 1) |
| T field | `t_probe.py field` | 1 s | ratio e^{i pi/4} tan(beta), Q(zeta_24), both primes 1 mod 24 |
| T tables | `t_probe.py ratios qubit_T 4 2`, `ratios qubit_T 3 2` | 4 s, 1 s | (0, 0) at j = +-1, +-2 for the |T>^4 decomposition; 0 exact at j = +-2 for the 8 of |T>^3 |
| T rank-4 list | `t_probe.py covers4 qubit_T 4` | 2 s | 5,205 rank-4 decompositions of |T>^3 (numeric enumerator) |
| H tables, n_2 = 3 | `t_probe.py ratios qubit_H 3 2` | 1 s | 0 exact at j = +-2 for the 6 rank-3 decompositions |
| H tables, n_2 = 4 | `two_qubit_slice.py --ratio-tables 1` | 5.2 min | (0, 0) at j = +-1 for all 30 rank-4 decompositions |

The probe scripts `t3_probe.py`, `t_probe.py`, and `beta_degenerate.py`
are session scratch (their content is described where it is used);
`prototype.py` is committed under `research/h5_rank5/`.
