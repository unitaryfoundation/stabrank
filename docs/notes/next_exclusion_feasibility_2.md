# The next lower-bound exclusion, second round: feasibility and design

Status (2026-09-24). Design and costing only; nothing here is a bound.
Five candidates were compared for the next pod run: a rank-6 exclusion of
|H>^7 (A), the T-state analogue at m = 7 (B), an S cell (C), a rank-6
exclusion of |N>^4 or |H3>^4 (D), and the T3 and T5 cells (E). The pick is
E in the form of the rank-5 exclusion of |T5>^2: a direct census of the
full 5-covers of the two-ququint target over the 3,900 two-ququint
stabilizer states through the existing compiled kernel, no slicing, no
matcher, no structural fact, about 10 CPU-hours on the laptop rates
measured here and about 12 on the pod, an hour of wall time on 15
processes. It moves the cell from 5 <= chi(T5^2) <= 8 to 6 <= chi <= 8
and, by projection, puts the first lower bounds on the T5 m = 3 and m = 4
cells. It settles nothing exactly; no candidate that settles a cell is
within reach, and the measurements below say why. D is the only candidate
whose exclusion closes a cell (chi(N^4) = 7), and it is the one to build
toward: its census of full 6-covers of |N>^2 is 3.2e7 to 3.8e7, 170 times
the rank-5 census, its dependent-base list is about 2e6 multisets at 18 s
each through the present Python dense solve, and its invisible-flat
matcher at p = 3 is not written. A is dead on every slicing route (the
6-cover census of |H>^3 is about 1.6e9 to 1.9e9; the 5-cover census of
|H>^4 alone is 1.9 million pivot pairs at 16 to 137 s each through the
compiled kernel). B has no board cell and the same censuses. C moves one
cell by one step at a cost above A's.

Everything measured below ran single-process at nice 19 on the 18-core
laptop under load from other sessions, through
`research/t5_rank5/run.py` with a 600 s cap, about 0.9 CPU-hours in all
(section 8). The scripts are `research/n4_rank6/probe.py` and its records
under `research/n4_rank6/results/`.

Notation follows `docs/notes/t5_rank5_exclusion.md` and
`docs/notes/qutrit_m4_rank5_exclusion.md`. A full k-cover of a target is a
multiset of k dictionary states whose span contains the target with a
coefficient assignment in which every coefficient is nonzero; kappa is the
dimension of the coefficient family (0 for distinct independent states).
N_n is the n-qudit dictionary size: 60, 1080, 36,720, and 2,423,520 for
two, three, four, and five qubits; 360 and 30,240 for two and three
qutrits; 3,900 for two ququints. G_n is the unitary symmetry group of the n-copy
target (its order is quoted where it matters).

## 1. The cells

| candidate | cell | board (this checkout) | what the exclusion gives |
|---|---|---|---|
| A | qubit_H m = 7, rank 6 | 6 <= chi <= 9 (`qubit_H-m7-lower-6.json`, projection of m = 6; `qubit_H-m7-upper-9.json`, literature) | chi(H^7) >= 7; closes the m = 7 route below log2(3)/4 |
| B | qubit_T m = 7, rank 6 | no cell on the board; m = 5 and m = 6 are 5 <= chi <= 6 in this checkout | chi(T^7) >= 7 once the cell exists |
| C | S m = 5, rank 5 | 5 <= chi <= 8 (`S-m5-lower-5.json`, slice and lift; `S-m5-upper-8.json`, product) | 6 <= chi(S^5) <= 8 |
| D | N m = 4, rank 6 | 6 <= chi <= 7 (`N-m4-lower-6.json`, attested; `N-m4-upper-7.json`, Lean witness) | chi(N^4) = 7 |
| D' | H3 m = 4, rank 6 | 6 <= chi <= 8 | 7 <= chi(H3^4) <= 8 |
| E | T5 m = 2, rank 5 | 5 <= chi <= 8 (`T5-m2-lower-5.json`, rank-4 exclusion; `T5-m2-upper-8.json`, annealed and refit) | 6 <= chi(T5^2) <= 8, and chi(T5^3), chi(T5^4) >= 6 by projection (those cells have product upper bounds 24 and 64 and no lower bound) |
| E' | T3 m = 5 | 8 <= chi <= 18 | nothing within reach: the gap is ten and the rank-8 exclusion at m = 4 was already rejected on census size |

The rank-5 exclusion of |T>^5 was certified on the pod on 2026-09-24
(160 batches, 36.5 CPU-hours, 0 hits; stage B ran at about four times the
laptop rate, the compiled stages at about 1.3 times); its bound PR was
pending when this note was written, so the qubit_T m = 5 and m = 6 files
in this checkout still say 5 <= chi <= 6. With it, chi(T^5) = chi(T^6) =
6, and B's prerequisite is in place; B fails for other reasons (section
3).

## 2. Candidate A: rank 6 at |H>^7

### 2.1 The structural facts are free

Suppose psi_7 = sum_{i=1}^6 c_i s_i with every c_i nonzero.

Fact 1 (every term is full along every qubit). A one-qubit slice sees at
least chi(H^6) = 6 terms, so all six.

Fact 2 (every term is full along every qubit pair). A two-qubit slice
point sees at least chi(H^5) = 6 terms, so all six: every term's flat
along any pair is the plane.

Fact 3 (at most two absent terms at any three-qubit point): chi(H^4) = 4.

Flat lemma along a qubit triple. By Fact 2 the flat of a term in F_2^3
projects onto the plane along every coordinate pair, so its direction
space contains none of e_1, e_2, e_3; a line or a point fails, and among
the seven planes only the even-weight subspace avoids every e_k. So every
term is an 8-point term, or a 4-point term on the even-weight coset E or
the odd-weight coset O. By Fact 3 at most two terms lie on O and at most
two on E. Along a qubit quadruple the same argument gives directions of
dimension 3 or 4 only (a 2-dimensional direction would contain a vector of
weight at most 2), that is, 16-point terms and 8-point terms on the cosets
of the five kernels ker f with f of weight 3 or 4; five of those cosets
miss 0000. At most three terms are absent at any four-qubit point, since
chi(H^3) = 3.

All three facts are board bounds, and the flat lemmas are the arguments
above; nothing needs a table. The case split is then as at T^5: one base
point, every invisible flat as a parameter.

### 2.2 The bases, and why every route fails

| route | base point in | base is a full k-cover of | over | k | invisible terms |
|---|---|---|---|---|---|
| 2 + 5 | F_2^2 | psi_5 | N_5 = 2,423,520 | 6 | none (Fact 2) |
| 3 + 4 | F_2^3 | psi_4 | N_4 = 36,720 | 6, 5, 4 | 0, 1, 2 terms on O |
| 4 + 3 | F_2^4 | psi_3 | N_3 = 1080 | 6, 5, 4, 3 | up to 3 terms on the five 8-point flats missing 0000 |

The 3 + 4 route's 4-cover bases are the 30 minimal decompositions of
|H>^4 (a full 4-cover of psi_4 by four states is independent, since
chi(H^4) = 4), so that stage is small. The 5-cover and 6-cover censuses of
|H>^4 are not. Measured (`probe.py census5 qubit_H 4`, the compiled
`cover5_pair` through `CoverEnumerator(4, orbit="qubit_H")`):

| quantity | value |
|---|---|
| states, group order, pivots | 36,720; 384; 246 |
| pivot pairs | 1,929,371 (planned in 27 s) |
| pairs run in 330 s | 4: (4, 1990) 1 cover, 1.19e8 modular candidates, 123 s; (70, 2670) 0, 1.15e8, 137 s; (215, 22355) 0, 9.1e6, 16 s; (541, 10359) 0, 5.5e7, 62 s |

The per-pair time is the accidental collisions of the 16-bit key at M
near 36,000 members (about M^2 / (2 x 65521) per third pivot, decided one
by one), the same effect the T^4 probe saw in Python. At 60 to 130 s for
the heavy pairs the 5-cover census alone is of order 1e4 to 1e5
CPU-hours; a two-functional key removes the accidents but leaves the
hashing at about M^2 log M per pair, of order 1e3 CPU-hours over 1.9
million pairs. The 6-cover census, which the all-visible stage needs, has
no kernel (cover5.cpp is written for r = 5: pivot, partner, third pivot,
parallel pair) and would be a further factor of M per pair. The 2 + 5
route is worse by the dictionary size.

The 4 + 3 route keeps the small dictionary and pays in the number of
covers. Measured with a Python r = 6 pair kernel (`probe.py census6
qubit_H 3`: pivot, partner, two further pivots, a parallel pair modulo a
5-dimensional span, keyed by two functionals, every candidate decided
numerically in batches and every survivor exactly):

| pair (i, j) | members above j | full 6-covers | candidates | seconds |
|---|---|---|---|---|
| (3, 706) | 373 | 206,294 | 3,532,228 | 66.6 |
| (356, 540) | 535 | 40,789 | 25,903,049 | 256.0 |
| (3, 932) | 147 | 2,628 | 90,970 | 1.4 |
| (811, 924) | 148 | 19 | 172,802 | 1.7 |
| (582, 1025), (1074, 1043), (1074, 1075) | 54, 6, 1 | 0 | 3,363, 0, 0 | 0 |
| (3, 354) | about 1,000 | not finished in 600 s | | |

Over the 14,280 pivot pairs (1,515 with more than 800 members above the
partner, 5,424 with more than 500) a fit of the cover count against the
member count gives 1.6e9 (cubic) to 1.9e9 (quartic) full 6-covers of
|H>^3 of distinct independent states, against 5,939,465 full 5-covers,
and the Python census alone projects to about 1,100 CPU-hours. Every one
of those bases would then be matched over fifteen other points with up
to three invisible terms, with `SliceMatchKernel` accepting n_1 at most
3. Not a pod run in any form; the compiled 5-cover kernel does not extend
to k = 6 and a k = 6 kernel would change the census time, not the count.

## 3. Candidate B: rank 6 at |T>^7

There is no qubit_T m = 7 cell (the board has m = 2 to 6, 8, and 10). The
facts of section 2.1 transfer once chi(T^5) = chi(T^6) = 6 is filed
(chi(T^4) = 3 leaves up to three absent terms at a three-qubit point
instead of two), and the bases are the same censuses over the same
dictionaries with the Q(zeta_24) field: 6-covers of |T>^3 (the 5-cover
census was 6,115,136, larger than H's) or 5-covers and 6-covers of |T>^4.
B fails as A does.

## 4. Candidate C: the S cells

|S> = (|1> - |2>) / sqrt 2 (the Strange state) has chi(S^1) = chi(S^2) = 2,
chi(S^3) = chi(S^4) = 4, and 5 <= chi(S^5) <= 8, 5 <= chi(S^6) <= 8. The
only exclusion that moves a cell is rank 5 at m = 5, to 6 <= chi(S^5) <=
8; the gap stays three, and chi(S^5) >= 6 gives the exponent nothing
(`docs/notes/constructions_2026_09.md`: rank 6 at S m = 5 is above the
published exponent).

The 2 + 3 slice: a two-qutrit point sees at least chi(S^3) = 4 terms, so
the bases are full 4-covers (the 15 rank-4 decompositions of |S>^3 up to
symmetry, from the slice-and-lift record) and full 5-covers of |S>^3 over
the 30,240 three-qutrit states, with at most one invisible term. The
5-cover census of |S>^3 is the size class of the |H>^4 measurement above
(30,240 states, a group of order 6^3 x 6 or so, of order 1e6 pivot pairs
with M near 30,000), that is, of order 1e4 compiled CPU-hours at the
present key. The 3 + 2 slice has 27 points with at most three absent
terms each (chi(S^2) = 2) and 156 flats missing the base point, and the
zero amplitude of |S> makes every slice at a point with a 0 coordinate a
homogeneous equation, the pitfall the slice-and-lift record notes. C is
not a pod run, and its payoff is one step on one cell.

## 5. Candidate D: rank 6 at |N>^4

### 5.1 Which facts survive at rank 6

Suppose psi_4 = |N>^4 = sum_{i=1}^6 c_i s_i with distinct s_i and every
c_i nonzero (fewer distinct terms or a zero coefficient is rank 5,
excluded by `N-m4-lower-6.json`).

One-qutrit slices. A slice at value k of qutrit q sees at least chi(N^3)
= 4 terms. Exactly four is the unique minimal rank-4 decomposition of
|N>^3 plus two point terms at the other values, and
`docs/notes/constructions_2026_09.md` (item 3 of the sharpest negative
facts, `relaxed_lift.py` case [A] at R = 5 and R = 6) closes it: no
rank-5 or rank-6 decomposition of |N>^4 or |H3>^4 has a minimal one-qutrit
slice. So every one-qutrit slice sees five or six terms, and along every
qutrit at most one term is a point term. Fact A of the rank-5 argument
(every term full along every qutrit) does not follow: a slice with five
visible terms is a non-minimal rank-5 decomposition of |N>^3 plus a point
term, and excluding it needs the census of full 5-covers of |N>^3 over
30,240 states, the 1 + 3 route the rank-5 note put out of reach and the
|H>^4 measurement of section 2.2 puts at 1e4 compiled CPU-hours. Fact A
is lost at rank 6 for the same reason Fact 1 was lost at T^5.

Two-qutrit slices. A point sees at least chi(N^2) = 3 terms, and PR #86's
case M search (the same note, "no rank-5 or rank-6 decomposition of |N>^4
or |H3>^4 has a two-qutrit slice with exactly three nonzero terms", 90
(decomposition, x_0) pairs and 945 coverage patterns for N, 0 exact
completions) closes three. So every two-qutrit point sees at least four
of the six terms: Fact B survives at rank 6, and at most two terms are
invisible at any point.

### 5.2 The design that remains

The T^5 design carries over with Fact B in place of Fact 2: one base
point along qutrits 1, 2 for every decomposition, the visible terms there
number k in {4, 5, 6}, their base slices form a full k-cover of |N>^2, the
6 - k invisible terms have flats missing the base point (8 points and 8
lines of F_3^2, 16 flats), and G_2 (order 72) acting on the unsliced
qutrits reduces the bases to one per orbit. No case split and no monomial
symmetry of |N> is used. The base point should be x_0 = (2, 2): |N> =
(1, 1, -2) / sqrt 6, so alpha_x is 1 at the four points with both
coordinates in {0, 1}, -2 at the four points with one coordinate 2, and 4
at (2, 2); from (2, 2) the other eight points have ratios 1/4 and -1/2,
never 1, while from (0, 0) three points have ratio 1, where the trivial
translate always solves the exact equation. The rank-5 run used (0, 0),
forced by its case split; the measurement below shows the factor.

- Stage (alpha), k = 6: full 6-covers of |N>^2 (kinds A, B, C as before).
  Matcher: `research/qutrit_m4_rank5/matcher.py`, which is generic in the
  number of terms (`Matcher.run(cover, x0, target)` takes any multiset).
- Stage (beta'), k = 5: the 197,440 full 5-covers of the rank-5 census
  plus its 12,175 dependent and 16,181 repeated-state multisets, each
  with one invisible term on one of 16 flats. Matcher: a p = 3 port of
  `research/t5_rank5/invisible.py` (points with no fresh term first,
  exact solves, a residual scan at the first point of the invisible
  term's flat, the second and third points of a line as phased Pauli
  translates), not written.
- Stage (gamma), k = 4: the 1,403 full 4-covers and the handful of
  degenerate 4-multisets, each with two invisible terms on one of 136
  flat multisets, with a rank-2 residual scan over the 360 states where
  both are fresh.

### 5.3 The census, measured

`probe.py census6 N 2`, the Python r = 6 pair kernel of section 2.2 over
`CoverEnumerator3("N", 2)` (17 pivots, 1,209 pivot pairs), two stratified
samples of 24 pairs (one to several per pivot, at the start and at the
middle of every pivot's partner list):

| sample | pairs | full 6-covers found | candidates | seconds | projected full 6-covers | projected census seconds |
|---|---|---|---|---|---|---|
| partner lists at the start | 24 | 1,138,899 | 21,377,937 | 417 | 3.18e7 | 15,100 |
| partner lists at the middle | 24 | 1,223,608 | 17,001,528 | 403 | 3.76e7 | 14,500 |

Per pivot (pairs in the census; full 6-covers per sampled pair in the two
samples): 117 (16; 753,064 and 648,920), 279 (36; 45,446 and 60,623), 281
(354; 13,894 and 14,025), 285 (114; 24,240 and 18,826), 288 (38; 59,499
and 202,339), 297 (55; 32,569 and 22,256), 306 (26; 47,131 and 73,014),
310 (140; 19,940 and 37,634), 313 (119; 6,494 and 11,616), 325 (65;
15,912 and 4,184), 332 (43; 3,905 and 3,379), 342 (31; 3,452 and 2,181),
345 (29; 5,046 and 248), 348 (65; 513 and 16), 357 (47; 1,141 and 120),
358 (11; 324 and 48), 359 (20; 81 and 0). Every cover found has kappa =
0, as it must: a 6-set of distinct states with a coefficient family
contains a 5-cover, whose fifth state has a zero residue modulo the
four-state span and is skipped by the kernel, so the dependent 6-sets
come from the degenerate routes, not from the kernel. The rank-5 census
was 197,440 full 5-covers; the rank-6 census is 160 to 190 times that,
and its Python enumeration is about 4 CPU-hours.

The degenerate lists at rank 6, by the routes of the H^6 v2 list:

| kind | route | size |
|---|---|---|
| B, dependent, kappa = 1 | a full 5-cover plus a state of its span | 197,440 x (2 to 35 span states in the twelve sampled 5-covers, mean about 12): about 2e6 6-sets, fewer as unordered sets |
| B | a full 4-cover plus a pair parallel modulo it; a 3-cover plus three states of its span | unmeasured; the rank-5 analogues were 12,175 in all |
| C, (2, 1, 1, 1, 1) | a full 5-cover plus one of its members | 987,200 |
| C, cancel at base | a full 4-cover T plus (b, b) with b outside span T | 1,403 x about 355: about 5e5 |
| C, (2, 2, 1, 1), (3, 1, 1, 1), and the rest | a 4-cover or 3-cover plus members | about 1.4e4 |

### 5.4 Matcher rates, measured

`probe.py rate6` ran the rank-5 pipeline's matcher on 40 independent
6-covers spread through the first sample and on degenerate multisets
built from sampled 5-covers of the stored census, at both base points:

| base point | kind | runs | per run: mean | median | max | coordinate slices |
|---|---|---|---|---|---|---|
| (2, 2) | A6, independent | 40 | 0.088 s | 0.092 s | 0.18 s | every run dies at the first (ratio 1/4) |
| (2, 2) | C6, 5-cover plus a member | 8 | 0.116 s | 0.110 s | 0.20 s | every run dies at the first |
| (2, 2) | B6, 5-cover plus a span state (kappa = 1) | 8 | 18.3 s | 14.7 s | 35.6 s | the 1-parameter dense solve; one run reached (15, 117) |
| (0, 0) | A6 | 40 | 0.475 s | 0.140 s | 5.17 s | (6, 36) typical, up to (18, 324) |
| (0, 0) | C6 | 4 | 2.37 s | 1.88 s | 5.32 s | (27, 513) to (180, 7,920) |
| (0, 0) | B6 | 4 | 32.8 s | 29.1 s | 47.5 s | (962, 1,004) to (22,397, 208,269) |

No hit, no refusal. The base point (2, 2) is five to twenty times cheaper
than (0, 0) on every kind, and the 0.088 s of an independent run at
(2, 2) is the option setup for six terms, not the search.

### 5.5 Projected cost, and what would make D a pod run

At the laptop rates above, with the pod at four times the laptop rate for
the Python paths (the T^5 stage B experience):

| stage | bases | rate | laptop CPU-hours | pod CPU-hours |
|---|---|---|---|---|
| (alpha) A6 | 3.2e7 to 3.8e7 | 0.088 s | 800 to 930 | 3,200 to 3,700 |
| (alpha) B6 | about 2e6 | 18 s | about 10,000 | about 40,000 |
| (alpha) C6 | about 1.5e6 | 0.12 s | 50 | 200 |
| (beta') | 225,796 x 16 flats | unmeasured (no p = 3 invisible matcher); at the T^5 rate of about 0.03 s per (cover, flat) where the run dies at an exact point, more for the 12,175 dependent bases | 30 plus the dependent tail | 120 plus |
| (gamma) | about 1,500 x 136 | unmeasured, small | | |

D is not a pod run in Python at any wall time the pod offers. Three
pieces of engineering would change that, in order of leverage: a
compiled p = 3 stage A kernel in the shape of `slice_match.cpp` (the
qubit kernel runs 0.2 to 0.4 ms per cover, two hundred times the Python
rate; A6 becomes about 5 pod CPU-hours), a compiled or restructured
1-parameter dense solve (the saving noted at H^6, H^5, and T^5; B6 at 2e6
bases needs a factor of a hundred to reach about 100 pod CPU-hours, or a
different treatment of a base that is a 5-cover plus a span state, for
instance carrying the 5-cover's unique coefficients and the span state's
coefficient as the one parameter through the first exact point, where
almost every run dies), and the p = 3 invisible-flat matcher with its
block paths. With all three, D projects to 150 to 300 pod CPU-hours and
settles chi(N^4) = 7; H3 has the same shape one census up (188,451
5-covers) and settles nothing exactly. The pod's anneal loops have been
searching for rank 6 at N^4 and H3^4 since 2026-09-21 without a hit,
which is consistent with chi(N^4) = 7 and does not shorten the exclusion.

## 6. The pick: rank 5 at |T5>^2

### 6.1 The argument

|T5> = 5^{-1/2} sum_x w_5^{x^3} |x> and psi_2 = |T5>^2, whose entries are
w_5^{x^3 + y^3} / 5. A rank-5 decomposition is psi_2 = sum_{i=1}^5 c_i s_i
over the 3,900 two-ququint stabilizer states, and rank 4 is excluded
(`bounds/T5-m2-lower-5.json`: rank 3 by pivot and quotient from the 66
representatives of the full symmetry group of order 100, rank 4 by the
pivot-pair search of `slice_lift.py` from the 98 representatives of the
unitary group of order 50). So every rank-5 decomposition is a 5-set of
distinct states whose span contains psi_2, and the 5 states are
independent: five dependent states span at most four dimensions, and a
target in the span of a rank-4 set is in the span of four of them. There
is no slicing, no base point, no invisible term, no coefficient family,
and no degenerate list: the whole exclusion is the statement that the
census of 5-sets with psi_2 in their span, one per orbit of the unitary
symmetry group as far as the pivot and partner reductions go, is empty.

Completeness rests on the census argument of the H^6 note, unchanged by
the orbit or by the absence of slicing: a symmetry carries a decomposition
to a decomposition (the coefficients go along), so the member with the
lowest orbit root can be moved to that root i as the pivot, every other
member then has a root at or above i (the member mask), and the pivot's
stabilizer moves the second member to one representative per stabilizer
orbit (the partner); the residue kernel lists a superset of the 5-sets
containing i and j whose span contains psi_2 (a parallel pair modulo
span(psi_2, u_i, u_j, u_k) over F_65521 for a third pivot k), and every
candidate is decided by ranks mod 2013265921 in the kernel and
numerically in the batch. The antiunitary symmetry is not used (it would
be sound here, since there is no slice to mix, but the unitary group's 98
pivots cost nothing).

### 6.2 Exactness

The states are phase patterns w_5^{q(x)} on affine flats of F_5^2 up to a
scalar, and the target is a phase pattern; every entry is a fifth root of
unity or zero after the column scalar is removed, which `patterns5`
asserts for every one of the 3,900 columns and `census5t5` checks for the
target (the control below). The field is Q(zeta_5) reduced modulo a prime
that is 1 mod 5: 65521 = 1 + 5 x 13,104 and 2013265921 = 1 + 5 x
402,653,184, so the same two primes serve, with w_5 a fixed nontrivial
fifth root of unity mod p (21009 mod 65521). Span membership and fullness
are scalar-invariant, so the unnormalized patterns stand for the states.
The kernel's superset property is the usual one: a 5-set whose span
contains psi_2 over Q(zeta_5) has vanishing 6 x 6 minors, which vanish
mod every prime, so it is listed unless the mod 65521 reduction drops the
rank of a pivot span, a coincidence of probability about p^{-codimension}
per pair that is negligible at codimension 20 and above; the same
statement underlies every census of this project.

### 6.3 The census, measured

`probe.py census5t5` (`results/census5_T5_sample.json`): the two-ququint
dictionary through `rank_exclusion.dictionary(5, 2)` (3,900 states, the
count asserted), the unitary symmetry group through
`symmetry_orbit_reps("T5", 2, D, antiunitary=False)`, the pivot plan of
`slice_cover.CoverEnumerator`, and the compiled `cover5_pair` with the
modular target patterns as inputs. Every 5-set the kernel returns is
re-decided numerically (residual, rank, coefficients) in the probe.

| quantity | value |
|---|---|
| states, group order, pivots | 3,900; 50; 98 |
| pivot pairs | 155,422 (planned in 0.2 s) |
| members above the partner, over all pairs | mean 1,435, max 3,899, sum of squares 4.58e11 |
| pairs run | 98, one per pivot at a random partner |
| 5-sets with psi_2 in their span | 0 (so 0 full covers) |
| modular candidates | 1,710,465 in the 98 pairs (about 1.2e5 at M = 3,600, the 16-bit key's accidents, all decided in the kernel) |
| seconds per pair | 7.49e-8 M^2 by a fit through the 98 pairs: 0.99 s at M = 3,616, 0.1 s at M near 1,100 |
| projected census | 4.58e11 x 7.49e-8 s = 9.5 laptop CPU-hours |

Controls run in the same command, both passed: the modular target is the
phase pattern w_5^{x^3 + y^3} of the complex |T5>^2 (the ratios to the
first entry agree with the codes), and eight planted targets sum_i c_i s_i
over five random states with coefficients a permutation of 1 to 5 (mod
both primes and over C) were each recovered by the kernel as a 5-set from
their two smallest members with every state admissible, 8 of 8.

### 6.4 Projected cost

| item | laptop | pod (compiled factor 1.3) |
|---|---|---|
| the census over 155,422 pairs | 9.5 CPU-hours | about 12 CPU-hours |
| numerical re-decision of the kernel's 5-sets | none expected; each is microseconds | |
| wall time on 15 processes | | about one hour |
| the `--no-native` replay of one batch through the Python reference `pair_covers(5)` | about 80 times the compiled time for its pairs (the H^5 replay ratio) | hours for one small batch |

A two-functional key in the kernel would remove the accidental candidates
and cut the census several fold; it is not needed at this size.

### 6.5 Pipeline plan

`research/t5_m2_rank5/` in the shape of `research/t5_rank5/`, smaller:

- `common.py`: `Field5`, `patterns5`, and `T5Enumerator` as in the probe
  (dictionary, modular images, target patterns, symmetry reps, pivot
  plan, the compiled kernel and the Python reference `pair_covers(5)` of
  `slice_cover` ported to the ququint arrays for the replay), the
  numerical decision of a 5-set, and the deterministic hash of a batch
  record (the kernel-run and candidate counts outside it, as at T^5).
- `driver.py plan`: the 155,422 units with the member count above every
  partner, hashed; `partition --target-s 600 --pod-factor 1.3`: units
  grouped greedily by 7.49e-8 M^2 into about 80 batches; `sample`: the
  rate fit; `control-planted N`: the planted 5-term targets (50, from
  random states and from states in one Pauli orbit); `control-pattern`:
  the target pattern; `control-m1`: the ten rank-3 decompositions of |T5>
  recovered by the r = 3 path of the same enumerator on one ququint
  (chi(T5) = 3, the board's exact cell).
- `batch.py K [--max-seconds S] [--resume]`: every unit of the batch
  through the kernel; every returned 5-set decided numerically and
  recorded with its flags (a modular-only set is a record, never a
  bound); units not run recorded as undecided; exit 1 on any 5-set, any
  undecided unit, or any kernel exception (a parallel class above
  `max_run`).
- `aggregate.py --recheck 2`: the plan and partition hashes, every
  record, two seeded re-runs from scratch with matching deterministic
  hashes, the manifest, and `CERTIFIED chi(T5^2) >= 6`.
- `verify_challenge/cert_t5_m2_rank5_attested.py`, the draft bounds
  `T5-m2-lower-6.json.draft`, `T5-m3-lower-6.json.draft`, and
  `T5-m4-lower-6.json.draft` (projection monotonicity, as
  `qubit_T-m6-lower-5.json` does), and a README with the pod commands.

Pod commands as in the T^5 note's section 8: fetch the branch into
`/root/stabrank-h6`, `uv sync --extra challenge`, check `cover5_pair`
imports, run the second batch of the partition in the foreground for the
rate (the first as the laptop pipeline test), then the 15-process xargs
loop with `--resume --max-seconds 3600`, then `aggregate.py --recheck 2`,
then the `--no-native` replay of the smallest batch into
`results/nonative/`.

### 6.6 Soundness checklist

From `docs/notes/qutrit_m4_rank5_review.md` and the T^5 note, reduced to
what a direct census has.

1. The dictionary is complete: `dictionary(5, 2)` asserts 3,900 =
   5^2 (5 + 1)(5^2 + 1) and collapses repeats up to a phase; a duplicate
   would only cost time, a missing state would be fatal, and the count
   check is the guard.
2. The phase codes are exact: every nonzero entry of every column is a
   fifth root of unity times the column scalar within 1e-6, asserted for
   all 3,900 columns and for the target; a state that failed the assertion
   would abort the plan rather than be mis-coded.
3. The field: both primes are 1 mod 5; w_5 satisfies w_5^5 = 1, w_5 != 1;
   the target's modular pattern equals the complex pattern's codes
   (control).
4. The kernel lists a superset and decides mod 2013265921; the batch
   re-decides every returned 5-set numerically (residual below 1e-7, rank
   5, every coefficient nonzero); a set that is a cover in one decision
   and not the other is recorded as undecided and fails the aggregate.
5. The reductions: pivots are the orbit roots of the unitary group only
   (`antiunitary=False`); the member mask is roots at or above the pivot;
   partners are one per orbit of the pivot's stabilizer through
   `stabilizer_orbit_labels`, whose Schreier generators give orbits at
   least as fine as the stabilizer's (more partners, never fewer).
6. Distinct terms: the kernel's members are above the partner and exclude
   the pivot, so every 5-set is five distinct states; a repeated state
   or a zero coefficient is rank at most 4, excluded by the board.
7. A parallel class above `max_run` raises in the kernel; the batch
   records the unit as undecided rather than skipping it.
8. Deterministic hashes exclude the candidate count and the kernel-run
   count; one batch replayed with `STABRANK_NO_NATIVE=1` through the
   Python reference must agree on every 5-set (expected none) and on the
   unit list.
9. `--max-seconds` records units not run as undecided; the batch exits 1;
   `--resume` redoes such a record.
10. Controls re-run at the commit that carries the final `common.py`:
    planted (50 of 50 required), pattern, m = 1 (the ten rank-3
    decompositions of |T5>, nothing outside them), and the rate sample.

### 6.7 Controls

1. Planted 5-term targets (8 of 8 here; 50 in the pipeline, including
   targets whose five states share a Pauli orbit and targets with a
   coefficient pattern of small height chosen so that two coefficients
   are equal).
2. The target pattern (passed here).
3. The one-ququint census: chi(T5) = 3 with ten rank-3 decompositions
   (`docs/notes/literature_sweep_2026_09.md`'s sweep note; the m = 1
   bound file lists five, which the control should reconcile by listing
   them up to the unitary group).
4. The rank-4 census of |T5>^2 by the same enumerator's r = 4 path must
   be empty, agreeing with `cert_t5_m2_rank4.py`.
5. The `--no-native` replay of one batch (checklist item 8).
6. The rate of the second batch on the pod before the partition is
   trusted.

### 6.8 What the result is worth

chi(T5^2) >= 6 moves one cell by one step and creates two lower bounds
(m = 3 and m = 4, now empty). It does not settle the cell: the upper
bound 8 is an annealed decomposition, a local anneal campaign at T5 m = 2
is running in the main checkout (the untracked `T5-m2-upper-8.json.local-
draft` and `autoresearch/manifests/t5_m2.json`), and if that campaign
finds rank 6 the exclusion would settle chi(T5^2) = 6; at rank 7 it would
leave 6 <= chi <= 7. The census itself is reusable: any later rank-6 work
at T5^2 (a two-copy census one rank up, or a slice of a three-copy
decomposition) starts from the list of full 5-covers, which this run
shows to be empty, and from the rate measured here.

## 7. Verdict

Run E. It is the only candidate whose cost is measured rather than
extrapolated (9.5 laptop CPU-hours from a fit through 98 pairs and the
exact sum of squared member counts), it needs no structural fact, no
matcher, and no degenerate list, its two controls pass, and its
pipeline is a reduction of the T^5 one. Build toward D afterwards: it is
the only exact-value target, its census (3.2e7 to 3.8e7) and its
degenerate lists (about 3.5e6) are known, its base point is chosen, and
what it needs is a compiled p = 3 stage A kernel, a hundredfold on the
1-parameter dense solve or a different treatment of the 5-cover-plus-span
bases, and the p = 3 invisible-flat matcher. A, B, and C are not pod runs
in any form measured here.

Proved by table, argument, or board bound in this note: A's three facts
and flat lemmas; D's surviving facts (no minimal one-qutrit slice at rank
6, at least four visible terms at every two-qutrit point, both from PR
#86's tables as recorded in `constructions_2026_09.md`) and the loss of
Fact A; E's reduction to an empty census and its exactness. Measured: the
censuses of section 2.2, 5.3, and 6.3, the matcher rates of 5.4, the
member-count model of 6.3. Assumed from earlier work: the pivot and
partner reductions (the H^6 census argument), the numerical tolerances
of `is_cover` and `is_full` and their modular backing. Extrapolated: the
cover counts of |H>^3 at k = 6 (a fit through seven pairs, none of them
the heaviest), the |H>^4 5-cover census time (four pairs), D's stage
(beta') rate (from T^5's), and the pod factor 1.3 for E's compiled census.

Blockers: none for E. For D, the three pieces of engineering of section
5.5. For A, B, and C, the censuses.

## 8. What was run

All through `research/t5_rank5/run.py` (nice 19, own session, 600 s cap),
one process at a time, about 0.9 CPU-hours in all, on the laptop at load
from other sessions. Records under `research/n4_rank6/results/`.

| step | command | time | result |
|---|---|---|---|
| N 6-cover census, first attempt | `probe.py census6 N 2 --pairs 40` | killed at 600 s | the per-candidate decision in Python did not finish one pair; the batched numerical pre-decision was added |
| N 6-cover census, start of the partner lists | `probe.py census6 N 2 --pairs 17 --budget 480` | 418 s | 24 pairs, 1,138,899 covers, projected 3.18e7 (`census6_N_sample.json`) |
| N 6-cover census, middle of the partner lists | `... --phase 0.5` | 404 s | 24 pairs, 1,223,608 covers, projected 3.76e7 (`census6_N_sample_mid.json`) |
| matcher rates at (2, 2) | `probe.py rate6 --census census6_N_sample.json --count 40 --deg 8 --x0 2,2` | 152 s | section 5.4 (`rate6_N_x22.json`) |
| matcher rates at (0, 0) | `... --deg 4 --x0 0,0` | 160 s | section 5.4 (`rate6_N_x00.json`) |
| H^3 6-cover census, heavy pairs | `probe.py census6 qubit_H 3 --pairs 48 --limit 4 --phase 0.5` | killed at 600 s | the first pair (3, 354), about 1,000 members, did not finish |
| H^3 6-cover census, light pairs | `... --limit 3 --phase 0.97` | 2 s | 3 pairs, 2,628 covers (`census6_H3_light.json`) |
| H^3 6-cover census, middle pairs | `... --limit 4 --phase 0.85` | 325 s | 4 pairs, 247,102 covers (`census6_H3_p85.json`); the fits of section 2.2 |
| T5 census sample | `probe.py census5t5 --pairs 24 --budget 400` | 17 s | 98 pairs, 0 span hits, 8.9 CPU-hours projected by per-pivot means |
| T5 census sample with controls | `probe.py census5t5 --pairs 60 --plant 8` | 23 s | 8 of 8 planted recovered; pattern control passed (`census5_T5_sample.json`) |
| T5 cost model | `probe.py census5t5 --plan-only` | 1 s | sum of squared member counts 4.58e11, 9.5 CPU-hours at 7.49e-8 s per member squared |
| H^4 5-cover census sample | `probe.py census5 qubit_H 4 --pairs 24 --budget 330` | 369 s | 1,929,371 pivot pairs; 4 pairs at 16 to 137 s (`census5_H4_sample.json`) |
