# Merge recipes at the exponent-moving product cells (2026-09-20)

Scripts and warm-start files: `research/merges/` (README there). Every
anneal went through `autoresearch/run.py --warm-from` and is in
`autoresearch/runs.jsonl` (seeds 101, 200 to 208, 304, 400 to 407, 500,
600 to 614, 700 to 705); everything ran single-process at nice 19 on the
shared machine, no run longer than six minutes.

The three recipes of `literature_sweep_2026_09.md` section 2 start from a
product of two stored minimal decompositions and ask whether one term can
be saved. Three exact tests were run on every product before any annealing,
all without a dictionary:

- Class sums. Terms with equal coefficient form a class; the class sum is
  tested for stabilizer-ness directly on the vector (unit modulus on an
  affine flat, quadratic-plus-linear phase, by `to_witness.term_from_vector`),
  and for rank 2 by the modulus multiset (`mergelib.rank2_excluded`: two
  stabilizer states on flats F_1, F_2 give one modulus on F_1 minus F_2, one
  on F_2 minus F_1, and on the intersection the values |m_1 + m_2 e^{i phi}
  w^j|, j in the phase group, for one phi; every configuration of flat
  dimensions, intersection size and coefficient moduli is checked against
  the observed multiplicities; controls: 980 genuine two-term sums, none
  excluded).
- Stabilizer states in a span. `mergelib.stabilizer_states_in_span` lists
  every stabilizer state inside a subspace, flat by flat: the vectors of V
  vanishing off a flat F are the eigenvalue-1 eigenspace of V_F^H V_F, and
  on a flat with a d-dimensional such space the state is fixed by its
  phases at d points, so p^(d-1) patterns are tried. 53,968 flats at five
  qutrits, 1,996,024 at six, 7,866,259 at eight qubits; a few seconds per
  subspace. Controls: recovers the third line of |T3>^2 from
  span(psi, l_0, l_1); the span of the rank-4 decomposition of |N>^3 holds
  exactly its four terms and 17 of the 30 rank-4 decompositions of |H>^4
  hold four more (the same counts `research/constructions/inspan.py`
  prints; the constructions note says 16 for H^4).
- One-state completion. An R-term decomposition containing R minus 2 given
  terms of an (R+1)-term one exists iff span(psi, kept) contains a
  stabilizer state outside span(kept). This is `kopt.py --k 1` without a
  dictionary and covers every 2-into-1 and 3-into-1 merge and every
  one-term exchange at once (`research/merges/complete1.py`).

No exact witness was found at any of the three cells.

## 1. T3 at m=6 through Z-eigensectors, sector rank 8

Target: the sector s of |T3>^6 is the five-qutrit carry state
`sector_target(6, s)`; rank 8 in each sector gives chi(T3^6) <= 24 and
gamma 0.4821.

The nine-term product. The rank-3 decomposition of |T3>^2 is three line
states l_0, l_1, l_2 with l_j supported on {x_1 + x_2 = j}, so each l_j is
in one Z^(x2) eigensector, and (|T3>^2)^(x3) splits its 27 products by
sigma_1 + sigma_2 + sigma_3 mod 3 into nine per sector; the nine-term sum
reproduces `sector_target(6, s)` to 4e-16 and every term is a 3-flat state
on the five carry qutrits (`t3_sector_product.py`). The literature sweep's
phrase "the m=3 sector decomposition with itself" refers to this product;
the sector of |T3>^3 is a two-qutrit state and its square is a four-qutrit
state, not the five-qutrit target. The coefficient of a term is
c_{sigma_1} c_{sigma_2} c_{sigma_3} with c_j = 3^(-1/2) e^{2 pi i j / 9}, so
it depends on the integer weight |sigma|: for s = 0 the classes are weight 0
(one term, sigma = 000), weight 3 (seven terms: the six permutations of 012
and 111) and weight 6 (one term, 222); all nine coefficients have modulus
1/3 and the nine terms are orthonormal.

Exact tests on sector 0:

| test | result |
|---|---|
| class sums | the weight-3 sum of seven terms is not a stabilizer state (support 189, one modulus); the singletons are |
| stabilizer states in span(nine terms) | exactly the nine terms (22 of 53,968 flats have a nonzero null space, all trivial) |
| one-state completion, every one of the 36 dropped pairs | span(psi, seven kept) contains exactly the seven kept states, so no eight-term decomposition contains seven of the nine product terms (138 s for all 36) |

Anneals (`run.py T3sector0 5 8 --warm-from research/merges/warm/T3sector0_m5_drop<i>.json`,
4 chains, cooling 0.995):

| dropped term | iterations | seed | residual | wall |
|---|---|---|---|---|
| 0 (sigma 000) | 2000 | 101 | 0.333333 | 32 s |
| 0 | 8000 | 200 | 0.333333 | 126 s |
| 8 (222) | 8000 | 208 | 0.333333 | 135 s |
| 4 (111) | 8000 | 204 | 0.333333 | 127 s |
| 1 (012) | 8000 | 201 | 0.333333 | 131 s |
| 4 | 20000 | 304 | 0.333333 | 332 s |

The residual 1/3 is the starting value: dropping one of nine orthonormal
terms with coefficient modulus 1/3 leaves exactly 1/3, and no chain ever
found a lower configuration (the trace shows the current cost wandering
between 0.45 and 0.98 and the best staying at the start). The eight-term
product with any term dropped is a strict local optimum of the annealer's
exchange landscape, and the exact completion test says why: no single
stabilizer state replaces the dropped term. The random-start plateaus of
the sweep (0.4453, 0.5217) are higher than this warm start, so the
warm start is the better configuration but not a route. Three of the
five-qutrit plateau bases are under `autoresearch/state/plateaus_merges/`
(ignored by git) for `kopt.py --k 2` or `complete1.py`.

What would settle rank 8 here: a two-state replacement (`kopt.py --k 2`)
needs the five-qutrit dictionary (5.4e9 states) or a flat-by-flat version
of the rank-2 quotient search, which `stabilizer_states_in_span` does not
provide; the exact statement available is that any rank-8 sector
decomposition shares at most six terms with the product.

## 2. Qubit T at m=8, rank 8

Product of `bounds/qubit_T-m4-upper-3.json` with itself: nine terms on
eight qubits, flat dimensions 4, 6, 6, 6, 8, 8, 6, 8, 8, factor coefficients
of modulus 2/3 and phases 0, pi/6, -pi/6 (the sweep says pi/12; the stored
witness has pi/6), so the nine coefficients have modulus 4/9 and fall into
five phase classes: phase 0 with three terms (dims 4, 8, 8), +pi/6 and
-pi/6 with two terms each (dims 6, 6), +pi/3 and -pi/3 singletons (dim 8).

| class | stabilizer? | rank 2 | states in span(class) |
|---|---|---|---|
| phase 0, size 3 | no (support 208 of 256, four moduli with multiplicities 128, 66, 8, 6) | excluded (four moduli plus 48 zeros fit no pair of flats: two flats of dimension 8 would need at most four distinct values including zero; nested and disjoint configurations fail the multiplicities) | the three terms only |
| +pi/6, size 2 | no (support 110, three moduli) | n/a | the two terms only |
| -pi/6, size 2 | no | n/a | the two terms only |
| singletons | yes | | |

span(all nine terms) contains exactly the nine terms (67 flagged flats of
7.9 million, 65,722 phase patterns tried on them).

Anneals (`run.py qubit_T 8 8 --warm-from research/merges/warm/qubit_T_m8_drop<i>.json`,
4 chains, 8000 iterations, about 140 s each):

| dropped term (of the size-3 class) | seed | residual |
|---|---|---|
| 0 (dim 4) | 400 | 0.3905 |
| 5 (dim 8) | 405 | 0.3764 |
| 7 (dim 8) | 407 | 0.3694 |

The start residual (least squares of the target on the eight kept terms;
the terms are not orthogonal, largest overlap 1/4) is 0.4000 for each of
the three; the annealer improves on it but stops far from zero at three
different values, so these are not exact plateaus of one configuration.
The plateau bases are saved for completion tests; `complete1.py` on eight
qubits would scan 7.9 million flats per dropped term, about ten minutes per
plateau, and was not run.

## 3. N and H3 at m=6, rank 15

Products of the m=3 rank-4 witnesses with themselves: sixteen terms on six
qutrits, flat dimensions 0, 3, 3, 3, 3, 6, 6, 6, 3, 6, 6, 6, 3, 6, 6, 6 for
both orbits.

N: factor coefficients (-0.6124, 0.3536, 0.6124 i, -0.6124 i) on the terms
(|2,2,2>, |+>^3, the two full-support quadratic states); classes by equal
coefficient have sizes 3, 2, 2, 2, 2, 2, 2, 1 (by modulus alone 9, 6, 1).
The size-3 class is {|222222>, and the two products of the two
full-support states with their Galois conjugates} with coefficient 0.375.

| N class | stabilizer? | rank 2 | states in span(class) |
|---|---|---|---|
| size 3 (dims 0, 6, 6), coefficient 0.375 | no (support 729, three moduli: 468, 260, 1) | excluded (no pair of flats and moduli matches) | the three terms only |
| six pairs (dims 3,3 or 6,6) | no | n/a | the two terms only |
| singleton |+>^6 | yes | | |

H3: factor coefficients (0.6661, 0.6661, 0.3448 e^{-3 pi i/4}, 0.3448 e^{3 pi i/4});
classes of sizes 4, 4, 4, 2, 1, 1 (by modulus 8, 4, 4).

| H3 class | stabilizer? | rank 2 | states in span(class) |
|---|---|---|---|
| size 4 (dims 0, 3, 3, 6), coefficient 0.4436 | no (support 729, three moduli 676, 52, 1) | excluded (no pair of flats and moduli matches) | the four terms only |
| size 4 (dims 3, 6, 3, 6), e^{-3 pi i/4} | no (five moduli 432, 244, 36, 16, 1) | excluded | the four terms only |
| size 4 (dims 3, 6, 3, 6), e^{+3 pi i/4} | no | excluded | the four terms only |
| size 2 (dims 6, 6) | no | n/a | the two terms only |
| singletons | yes | | |

For both orbits span(all sixteen terms) was scanned over every flat except
the whole space (a 16-dimensional null space there means 3^15 phase
patterns, skipped): no stabilizer state beyond the terms on any proper
flat. The nine plane-times-plane terms named in the sweep as the first
merge candidates for N do not share a coefficient (the plane coefficients
are 0.6124 with three different phases), so they are not a class; the
size-3 class above is the one equal-coefficient triple.

Anneals (`run.py <orbit> 6 15 --warm-from research/merges/warm/<orbit>_m6_drop<i>.json`,
4 chains, 6000 iterations, about five minutes each; a 1000-iteration probe
on N drop 0, seed 500, gave 0.3327 in 53 s):

| orbit | dropped term (of the largest class) | seed | residual | wall |
|---|---|---|---|---|
| N | 0 (|222222>, dim 0) | 600 | 0.2068 | 318 s |
| N | 11 (dim 6) | 611 | 0.2135 | 313 s |
| N | 14 (dim 6) | 614 | 0.2508 | 323 s |
| H3 | 0 (dim 0) | 700 | 0.3199 | 362 s |
| H3 | 1 (dim 3) | 701 | 0.3260 | 339 s |
| H3 | 4 (dim 3) | 704 | 0.2852 | 282 s |
| H3 | 5 (dim 6) | 705 | 0.2890 | 302 s |

The start residuals (least squares of the target on the fifteen kept
terms; the product terms overlap by up to 0.19) are 0.3333 for N and 0.3943
for H3 whichever term of the class is dropped; every run improved on its start and
none came near zero, with a different final value per dropped term, so the
seven runs are not repeated exact plateaus of one configuration. Fifteen
terms at 729 dimensions cost about five minutes per 6000 iterations, so
more seeds at this depth or a 20000-iteration run are affordable one at a
time but were not run within the budget. Plateau bases are saved for
`complete1.py` (about two minutes per dropped term at six qutrits when the
whole-space flat is skipped).

## 4. Irreducible rank-5 decompositions of N^3 and H3^3

Question (constructions note): any rank-5 or rank-6 decomposition of |N>^4
or |H3>^4 has no minimal slice, so every slice is a 5- or 6-term
decomposition of the three-copy state with independent terms. Do such
irreducible rank-5 decompositions of |N>^3 and |H3>^3 exist, and do any
contain a term of the unique rank-4 decomposition?

Method (`rank5_pairs.py`): fix the pair (i, j), project psi and the whole
30240-state dictionary off s_j, and run the compiled rank-4 pivot kernel
with pivot i on the projected problem; {i, j, a, b, c} spans psi iff
{i, a, b, c} spans the projected psi. Every hit is re-solved in the full
space and kept only if no four of its five states span psi. The pivot is
the point state of the stored rank-4 decomposition (dim-0 term, in the
stored Clifford frame both orbits have a point plus three full-support
states); partners run over one representative per orbit of a subgroup of
the pivot's stabilizer in the unitary symmetry group (Schreier generators,
`slice_lift.stabilizer_orbit_labels`), 1035 orbits for both orbits. One
partner orbit costs 49 to 51 s of one core (a full 30240-partner rank-4
scan in the 26-dimensional quotient).

| orbit | partner orbits done (of 1035) | minimal rank-5 decompositions containing the pair | wall |
|---|---|---|---|
| N | 15 (orbit sizes 6 to 24) | 0 | 735 s |
| H3 | 15 (orbit sizes 3 to 24) | 2 orbit representatives (partners s_28 and s_29, orbit size 3 each): (0, 28, 9828, 11016, 11151) and (0, 29, 9828, 11016, 11151) | 758 s |

The H3 hits are genuine: term matrix of rank 5, residual 1e-15, no
four-subset spans |H3>^3, flat dimensions 0, 1, 2, 3, 3 (the point, a line,
a plane and two full-support states; coefficient moduli 0.6661 three times
and 0.3448 twice), and their five-dimensional span contains seven
stabilizer states. So irreducible rank-5 decompositions of |H3>^3 exist
and contain the point term of the rank-4 decomposition; the question for
H3^4 at rank 5 is therefore open on the constructor side, and the lift test
of `relaxed_lift.py` (two Pauli-matched slices plus a stabilizer residual)
can be run on these five-term slices exactly as on minimal ones. For N no
rank-5 decomposition through the point term was found in the 15 orbits
scanned.

Cost of finishing: one pivot (the point term) is 1035 partner orbits at 50 s,
about 14 hours of one core per orbit type; the three remaining terms of the
rank-4 decomposition (the full-support states, with smaller stabilizers and
hence more partner orbits) would be at least as much each. The count of all
irreducible rank-5 decompositions (pivots over all 74 or 116 dictionary
orbit representatives, partners over orbits of each pivot's stabilizer) is
of the order of 100 pivots times 1000 partners times 50 s, about 60 core-days
in the current kernel. A rank-5 kernel that quotients by (psi, s_i, s_j, s_a)
once per partner and finds parallel pairs, i.e. the present rank-4 kernel
with one more fixed vector, would not change the count of partner scans;
the saving would have to come from symmetry on the second member as well
(one pair per orbit of pairs, not per orbit of second members for each
first), which the present code does not implement.

## Plateau residuals, in one place

| cell | start (product minus one term) | annealed |
|---|---|---|
| T3 sector 0 (five qutrits) rank 8 | 0.3333 | 0.3333, six runs, never left the start |
| qubit_T m=8 rank 8 | 0.4000 | 0.3905, 0.3764, 0.3694 |
| N m=6 rank 15 | 0.3333 (every dropped term of the class) | 0.2068, 0.2135, 0.2508 |
| H3 m=6 rank 15 | 0.3943 (every dropped term of the class) | 0.3199, 0.3260, 0.2852, 0.2890 |

## What is exact and what is not

Exact: the class-sum stabilizer tests, the rank-2 exclusions, the
stabilizer-states-in-span lists (up to a 1e-8 eigenvalue tolerance on
orthonormal data and the 1e-7 phase tolerance of `term_from_vector`), and
the T3 one-state completion over all 36 pairs. Numerical evidence only: the
anneal residuals. None of this is a lower bound on any cell.

## Follow-up: lifting the irreducible rank-5 decompositions of H3^3

The two irreducible rank-5 decompositions found through the point term,
(0, 28, 9828, 11016, 11151) and (0, 29, 9828, 11016, 11151), were run
through `slice_lift.lifts` as five-term slice-1 decompositions of |H3>^3.
Three Pauli assignments satisfy the slice-2 equation and none satisfies
the slice-0 equation, so neither is a slice of a rank-5 decomposition of
|H3>^4. A rank-5 decomposition of |H3>^4 would need an irreducible rank-5
slice not in this list; the full irreducible rank-5 enumeration (about 60
core-days in the current kernel) is what would settle that, and is a job
for a pod rather than this machine.
