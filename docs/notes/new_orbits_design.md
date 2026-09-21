# New orbits beyond the six single-qudit cells: design note

Status (2026-09-20). Design for extending the board past its six orbits
(four qutrit, two qubit), with the literature values that would seed each
new cell, what breaks in the pipeline for each kind of orbit, and the
smallest change that makes the first new orbit verifiable. The code slice
implemented alongside this note is the ququint T-type orbit `T5`:
`orbit_state("T5")`, `ORBIT_P["T5"] = 5`, a prime-generic
`rank_exclusion.dictionary(p, n)` and `clifford_group(p)`, and tests
(`tests/test_ququint_dictionary.py`). No bound files are added; the
schema enum and the site constants are untouched, and the changes needed
there are listed in section 5.

Three facts computed while writing this, none of them in the literature
as far as the search in section 1 found, drive the recommendation:

- chi(|CS>^2) = 3, where |CS> = CS|++> is the two-qubit controlled-S magic
  state. Rank 2 is excluded over the 36,720 four-qubit stabilizer states
  (closest approach to parallel 0.763 against a threshold of 1 - 1e-6),
  and the pivot search finds sixteen rank-3 decompositions. One of them
  is, exactly,

      |CS>^2 = (1/2) |0000> - ((1+i)/2) |+i>^4 - ((1-i)/2) (CZ (x) CZ) |-i>^4,

  with |+i> = (|0> + i|1>)/sqrt(2) and |-i> its conjugate; sympy confirms
  the identity symbolically. The per-copy exponent is log_2(3)/2 = 0.7925,
  against the trivial 1 from chi(|CS>) = 2 and against 3 log_2(3)/4 = 1.19
  from synthesising CS with three T gates.
- chi(|T5>) = 3 for the ququint T-type state |T5> = 5^(-1/2) sum_x
  w_5^(x^3) |x>: rank 2 is excluded over the 30 single-ququint states and
  a three-term decomposition (two basis states and one full-support state
  with phase w_5^(4y^2 + 4y)) verifies symbolically through the existing
  `verify_upper`. So chi(|T5>) = 3 < 5, unlike T3, where chi(|T3>) = 3 = p.
  At two copies rank 3 is excluded over the 3,900 two-ququint states in
  half a second with the p-generic symmetry reduction (group of order 100,
  66 orbits, margin 0.066), so 4 <= chi(|T5>^2) <= 9.
- For the ququint Norrell-type state |N5> = |H,-1> of Jain and Prakash,
  proportional to |+> - |0>, chi(|N5>) = 2 and chi(|N5>^2) = 4 (rank 3
  excluded over 3,900 states, margin 0.045), so the m = 2 cell is tight at
  the product bound and the exponent log_5(2) = 0.4307 stands until m = 3.

The counts in the task brief for `dictionary(5, n)` are off by a factor
of five: 5^n prod_{j<=n} (5^j + 1) is 30, 3,900 and 2,457,000 for
n = 1, 2, 3, not 150, 19,500 and 12,187,500 (the latter count the five
global phases separately). The enumeration confirms 30 and 3,900 distinct
states, and the feasibility estimates below use the correct numbers.

## 1. Literature values

Searched: arXiv listings and full texts for CCZ, Toffoli, CS, hypergraph
and qudit stabilizer rank, 2016 to 2026. The values below are the ones
that would seed cells; where a paper was read only through its abstract
that is said.

### Multi-qubit magic states

| state | fact | source |
|---|---|---|
| CCZ | \|CCZ> = CCZ\|+++> = 8^(-1/2) sum_{abc} (-1)^(abc) \|abc> (eq. 106); F(CCZ) = 9/16; xi(CCZ) = 16/9 (eq. 31); eight-term sum-over-Cliffords decomposition (eqs. 29, 108); chi_delta(CCZ^t) <= delta^-2 (16/9)^t, i.e. exponent log_2(16/9) = 0.830 per copy for the approximate rank; stabilizer fidelity multiplicative, F(CCZ^t) = F(CCZ)^t (Corollary 3); synthesising CCZ with four T gates gives the worse 1.884^t. The open question of computing exact low-rank decompositions of Z/CZ/CCZ diagonal circuits applied to \|+>^n from the transversal number of the CCZ hypergraph is stated in their conclusion. No exact rank of \|CCZ>^m beyond the trivial 2^m is given. | Bravyi, Browne, Calpin, Campbell, Gosset, Howard, arXiv:1808.00128 |
| CCZ | chi(\|CCZ>) = 2 follows from \|CCZ> = \|+++> - 2^(-1/2) \|111>, and \|CCZ> is not a stabilizer state (its phase (-1)^(abc) is cubic). Confirmed here: 15 rank-2 decompositions over the 1,080 three-qubit states, none of rank 1. | direct |
| T (for comparison) | chi(T^6) <= 6, exponent log_2(3)/4 = 0.3963 (cat-state construction); no CCZ or Toffoli content (checked the full text). | Qassim, Pashayan, Gosset, arXiv:2106.07740 |
| T | chi(T^12) <= 47, exponent 0.463; no CCZ content. | Kocia, arXiv:2012.11739 |
| two-qubit states | a two-qubit state psi_alpha with chi(psi_alpha) = 2 and chi(psi_alpha^2) = 4 for transcendental alpha (Theorem 4.1); chi(T^n) >= (n+1)/(4 log_2(n+1)) (Corollary 3.3). A search snippet attributes to this paper a claim that D\|+>^n with D built from Z, CZ, CCZ has chi = 2 for n <= 5; the read of the paper did not confirm it, so treat it as unverified. | Lovitz and Steffan, arXiv:2110.07781 |
| hypergraph states | stabilizer Renyi entropy of hypergraph states, and of 3-uniform hypergraph states via a matrix rank; nothing on exact stabilizer rank. | Chen, Yan, Zhou, arXiv:2308.01886; arXiv:2602.23687 |
| CS | no stabilizer-rank result found. F(CS) = 5/8 over the 60 two-qubit stabilizer states, so xi(\|CS>) = 8/5 by the Clifford-magic-state theorem of arXiv:1808.00128, giving an approximate-rank exponent log_2(8/5) = 0.678 per copy. chi(\|CS>) = 2 from CS = ((1+i)/2) I + ((1-i)/2) CZ. chi(\|CS>^2) = 3 computed here. | direct |
| Toffoli | Toffoli\|+++> = \|+++> is a stabilizer state; the Toffoli magic state is Toffoli\|++0> = (I (x) I (x) H)\|CCZ>, local-Clifford equivalent to \|CCZ>, so it is the same orbit and not a new cell. | direct |

Hypergraph states in general. For a 3-uniform hypergraph H on n vertices,
|H> = prod_{e in E} CCZ_e |+>^n has chi(|H>) <= 2^tau(H), with tau the
transversal (vertex cover) number: fix the qubits of a transversal to
computational-basis values and every hyperedge loses a vertex, so the
residual is a graph state. This is the construction the conclusion of
arXiv:1808.00128 anticipates. For m disjoint hyperedges tau = m, which is
the trivial product bound for |CCZ>^m; the CS identity above shows the
analogous 2^m bound for |CS>^m is not tight at m = 2, so nothing forces
2^m to be tight for CCZ either. Hypergraph states other than |CCZ>^m and
|CS>^m are a family, not an orbit, and do not fit the board's
"one state, m copies" model; they are out of scope for the orbit table.

### Higher-prime single-qudit states

| state | fact | source |
|---|---|---|
| T5 (face/T-type) | for p > 3 the qudit pi/8 gate is U_v = sum_k w_p^(v_k) \|k><k\| with v_k = (1/12) k (gamma' + k(6z' + (2k-3) gamma')) + k eps' (eq. 25), a cubic in k with cubic coefficient gamma'/6; the magic state is U_v\|+>. The p = 5 example (eq. 26) is v = (0, 3, 4, 2, 1). | Howard and Vala, arXiv:1206.1598 |
| T5 | M = diag(w^3, w, w^-1, w^-2, w^-1) for d = 5 (eq. 49), from lambda_j = C(j,3) - 2j + 3; the magic states \|M_k> = M\|+_k> are eigenstates of the Clifford C_M = M X M^dagger. | Campbell, Anwar, Browne, arXiv:1205.3104 |
| T5 | for any prime p > 3 the magic state is p^(-1/2) sum_x w_p^(P(x)) \|x> with P a cubic, and chi of its n-th tensor power is Omega(n) (Theorem 1.1). No exact values. | Labib, arXiv:2107.10551 |
| ququint catalogue | eight non-degenerate Clifford eigenstates and three degenerate families for p = 5. \|H,-1> = (10 - 2 sqrt 5)^(-1/2) [(1 - sqrt 5)\|0> + \|1> + \|2> + \|3> + \|4>] (Fourier eigenvalue -1); \|H,i>; \|B,-1> (most symmetric, orbit size 250); \|B,-e^(2 pi i/3)> (most magic, mana 0.748); \|A, +-w^2>; \|XV_s, 1> = \|0> + \|1> + w^3\|2> + \|3> + w^2\|4>. No state is a simultaneous eigenvector of all symplectic rotations for p > 3, so the strange state has no analogue. No stabilizer rank is discussed. | Jain and Prakash, arXiv:2003.07164 |
| qudit approximate rank | approximate stabilizer rank and weak simulation for odd prime qudits; scaling with the number of magic states, no exact ranks. | Huang and Love, arXiv:1808.02406 |

All the T-type candidates are one orbit. Every cubic phase over F_5 is
Clifford-equivalent to x -> x^3: the quadratic and linear parts of the
phase are Clifford (products of the phase gate and Z), and the
permutation x -> a x, which is the symplectic diag(a, a^-1), scales the
cubic coefficient by a^3, which ranges over all of F_5^* since cubing is
a bijection on F_5^*. So the Howard-Vala family (cubic coefficient
gamma'/6), the Campbell-Anwar-Browne M state (cubic coefficient 1/6 = 1
mod 5) and Jain and Prakash's |XV_s, 1> (interpolating its phases
(0, 0, 3, 0, 2) gives x^3 + x^2 + 3x, and the phase sum is 0 mod 5 so
there is no quartic term) all lie in the orbit of

    |T5> = 5^(-1/2) sum_{x in F_5} w_5^(x^3) |x> = 5^(-1/2) (1, w_5, w_5^3, w_5^2, w_5^4).

This is the vector `orbit_state("T5")` returns. Two facts about it
matter for the pipeline. Its amplitudes lie in Q(w_5), the same field as
the stabilizer states, so the Galois argument that gives chi(T3^m) >= 3
for free has no analogue; in exchange `fit_coeffs.fit` takes the generic
route without a cyclotomic special case and `is_zero` decides the
identities symbolically. Its local Clifford stabilizer has order 5 (the
Campbell-Anwar-Browne C_M), with an antiunitary partner, so the symmetry
group at m copies has order 2 * 5^m * m!, larger per copy than T3's 3.

The other named p = 5 candidates: |N5> = |H,-1> is proportional to
|+> - |0> (rank 2, local stabilizer of order 8, the Norrell analogue in
the sense that |N> is proportional to |+> - sqrt(3)|2>); |H,i> was not
reconstructed here (the amplitude pattern read from the paper did not
pass the Fourier eigenvector check, so its rank is unrecorded); the +1
eigenspace of the ququint Fourier transform is two-dimensional, so there
is no single H3 analogue. Nothing with p = 7 is worth a cell yet:
7^n prod (7^j + 1) is 56, 19,600 and 47,196,800 for n = 1, 2, 3, so
m = 2 exclusions are still cheap, but the m = 3 dictionary is six times
the four-qutrit one and there is no literature value at all.

## 2. Multi-qubit orbits: what "copies" means and what breaks

A multi-qubit orbit is a state on n_0 qubits (n_0 = 2 for CS, 3 for
CCZ). m copies is the n_0 m qubit state |M>^m, gamma = log_2(rank)/m is
per copy, and the board ranks gamma exactly as it does now. The
comparison exponents are then also per copy: the trivial 1 for both CS
and CCZ (chi = 2 at one copy), the approximate-rank exponents 0.678 (CS)
and 0.830 (CCZ) from the extent, and the T-synthesis exponents 1.19 (CS,
three T) and 1.585 (CCZ, four T). The exact exponent beats the
T-synthesis ones already at m = 1, which is the point of treating these
as orbits rather than as T-count.

What breaks. Everything that derives the qubit count from m assumes
n = m:

- `stabrank_verify.verify_upper` calls `stabilizer_vector(t, p, m)` and
  allocates `sp.zeros(p ** m, 1)`; both need n = n_0 m. `target_vector`
  is already right (the Kronecker power of the n_0-qubit vector has
  length 2^(n_0 m)). `implied_gamma(p, rank, m)` stays per copy.
- `rank_exclusion.symmetry_orbit_reps` builds the local symmetry from
  `clifford_group(p)` on one qudit and permutes copies with
  `_copy_permutation(perm, p, m)`, which permutes single qudits. For a
  width-n_0 orbit the local group is the n_0-qubit Clifford group mod
  phase (11,520 elements for n_0 = 2, closable from H_i, S_i, CNOT in
  seconds; 92,897,280 for n_0 = 3, which is not closable and must be
  replaced by generators of the stabilizer of |CCZ>, which include the
  permutations of the three qubits and the three C-type Cliffords
  X_j CZ_kl of eq. 107 of arXiv:1808.00128; a generator that does not
  fix the target or permute the dictionary raises at runtime, as now,
  so an incomplete generating set only weakens the reduction rather than
  the bound). `_copy_permutation` needs a block width.
- `slice_lift`: the lemma is stated for a single-qudit slice of
  phi^(m+1) = phi (x) phi^m, and its part (a) uses that every slice of
  phi is a nonzero multiple of phi^m. For CCZ a single-qubit slice of
  |CCZ> is |++>/sqrt(2) or CZ|++>/sqrt(2), a stabilizer state, so the
  slices of |CCZ>^(m+1) are (two-qubit stabilizer) (x) |CCZ>^m and the
  two slices are not proportional; `lifts_qubit` assumes they are. The
  clean generalisation slices along the whole first copy: the 2^(n_0)
  slices are then alpha_x |M>^m with every alpha_x nonzero (for CCZ all
  eight amplitudes are +-8^(-1/2); for CS all four are nonzero), so
  every slice is a minimal decomposition of |M>^m and no term's slice
  vanishes. Part (b) becomes: for a stabilizer state s_i on n_0 + n_0 m
  qubits, the slices u_i^(x) are supported on a coset of the X-part A_i
  of the image of Stab(s_i) in the symplectic coordinates of the block,
  and are Pauli images of one another (up to fourth roots of unity)
  inside it; since no slice vanishes, A_i = F_2^(n_0) for every term and
  all 2^(n_0) slices of each term are Pauli-related. The lift test is the
  same meet-in-the-middle over Pauli classes and phases as now, with
  2^(n_0) - 1 slice equations instead of 2, on n_0 m qubits. Roughly
  fifty lines in `slice_lift.py` (a `lifts_qubit_block(width)` and a
  block-aware `all_decompositions`), and the decompositions one copy down
  are exactly what the pivot search already lists.
- `to_witness.term_from_vector(v, p, n)` and `fit_coeffs.fit` take n
  explicitly or through m; `witness_from_vectors(orbit, m, ...)` and
  `fit(orbit, m, ...)` need n_0 m. `autoresearch/run.py` passes
  `n_orig=m` to the annealer; needs n_0 m. `cert_rank1_moduli.py` argues
  from unequal moduli, which fails for CS and CCZ (all moduli equal); the
  exact rank-1 exclusion for them is that the phase i^(x_1 x_2) or
  (-1)^(abc) is not a Z_4 quadratic form with even cross terms, which
  `term_from_vector` decides exactly by raising `NotStabilizer`.
- Schema: `orbit` enum, and the m cap. `m <= 8` is a verification
  budget for single qudits; for CS at m = 4 the target has 2^8 = 256
  entries, fine, and at m = 8 it has 65,536, which sympy will not verify
  in the budget. Cap CS at m <= 4 and CCZ at m <= 2 (64 entries; m = 3 is
  512, still fine, but no tool reaches it). `validate_bounds.py` should
  carry the per-orbit cap since the schema's `maximum` is global.
- Site: `ORBIT_ORDER`, `SYSTEM` (a new value, "2 qubits" / "3 qubits"),
  `BASELINE`, `ORBIT_TEX`, `BASE_TEX`, `ORBIT_DEF`, `COLOR`; `ket()` and
  `next_target()` take m and rank and are per copy already. CONTRIBUTING
  table.

Reachable cells. Dictionaries: 60 (2 qubits), 1,080 (3), 36,720 (4),
2,423,520 (5), 315,057,600 (6). Exhaustive exclusions run to four qubits
in seconds and to five qubits with the phase-code style of enumeration
(the six-qubit dictionary is out of reach for any exclusion).

| orbit | m | qubits | lower | upper | tool |
|---|---|---|---|---|---|
| CS | 1 | 2 | 2 (phase not a Z_4 form, exact) | 2 (CS = ((1+i)/2) I + ((1-i)/2) CZ) | settled |
| CS | 2 | 4 | 3 (rank-2 exclusion, seconds, margin 0.24) | 3 (witness above, verified) | settled here |
| CS | 3 | 6 | 3 by monotonicity; 4 if none of the 16 rank-3 decompositions of \|CS>^2 lifts under the block slice-and-lift (seconds once written) | 6 (product) | annealer for rank 5 (gamma 0.774) or 4 (0.667, below the extent exponent, unlikely); short probes (three seeds, 14 s each) plateau at residual 0.28 to 0.32 at rank 5 and 0.45 at rank 4 |
| CS | 4 | 8 | monotonicity | 9 (3 x 3) | annealer only |
| CCZ | 1 | 3 | 2 (exact) | 2 | settled |
| CCZ | 2 | 6 | 2 by monotonicity; 3 by the block lift from the 15 rank-2 decompositions of \|CCZ> (seconds once written) | 4 (product) | annealer for rank 3 (gamma 0.79); two seeds plateau at residual 0.4516 |

The single-qubit-slice route (targets (stabilizer) (x) |M>^m over
n_0 m + 1 qubits, 2,423,520 states at five qubits for CS m = 2 and CCZ
m = 1) is strictly more work than the block slice and is not needed.

## 3. Ququint orbits: what the verifier needs

`stabilizer_vector(term, p, n)` is already generic in p: the phase is
w_p^((Q(y) + l.y) mod p) for odd p with l read mod p, and `_w(5)` returns
exp(2 pi i / 5), which `is_zero` handles symbolically for the T5
identities (the m = 1 witness verifies "symbolically" in 0.6 s). The
Lean predicate `IsStabP` in `lean_proofs/LeanProofs/Stabilizer/IsStabP.lean`
takes any prime with `stabPeriod p = p` for odd p, so a Lean statement of
a T5 bound needs no new predicate, only a pointwise identity over
`zeta 5`.

What was missing, and is now in place:

- `rank_exclusion.dictionary(p, n)` for an odd prime p >= 5, through
  `stabrank.stabilizer_extent.enumerate_stabilizer_states(n, p)` (which
  lists (flat, coset, phase-polynomial) triples with repeats: 4,400 rows
  for two ququints) collapsed up to a global phase to the 3,900 distinct
  states, checked against p^n prod (p^j + 1). The qubit and qutrit paths
  are unchanged.
- `rank_exclusion.clifford_group(p)` for any odd prime from F, the phase
  gate diag(w^(j(j-1)/2)) and X, with p^3 (p^2 - 1) elements mod phase
  (216 and 3000 for p = 3, 5). For p = 3 the generators are the same
  matrices as before, so the closure is identical.
- `orbit_state("T5")`, `ORBIT_P["T5"] = 5`, `ORBIT_LABEL["T5"]`.
- `symmetry_orbit_reps`, `certify_rank3`, `rank2_search`, `rank3_search`,
  `to_witness.term_from_vector`, `fit_coeffs.fit` and `verify` all run
  for T5 without modification; the certificate driver's unit label is
  now looked up rather than assumed to be bit or trit.

Tests (`tests/test_ququint_dictionary.py`): dictionary counts 30 and
3,900 and distinctness up to phase; the Clifford group has 3000 elements
and the qutrit group and dictionary keep their sizes; non-primes are
rejected; the verifier's parametrisation, enumerated over every RREF
term, names each dictionary state exactly once at p = 3, n = 2 (360
states) and at p = 5, n = 1 (30 states), which is the check that the
phase convention w_p^(Q + l.y) is the same object in the verifier and in
the dictionaries at both primes; the T5 vector; and the rank-3 witness
for |T5> verifying end to end with the rank-2 exclusion beside it.

Reachable cells for p = 5. Dictionaries 30, 3,900, 2,457,000 for one,
two, three ququints; four ququints is 7.7e9 and out of reach.

| orbit | m | lower | upper | tool |
|---|---|---|---|---|
| T5 | 1 | 3 (rank 2 excluded, margin) | 3 (verified witness) | settled here |
| T5 | 2 | 4 (`certify_rank3` in 0.5 s, group of order 100, margin 0.066) | 9 (product) | rank-4 exclusion over 3,900 states with `decompositions_with_pivot(rank=4)` and the 66 symmetry representatives is feasible in minutes; annealer for ranks 4 to 8. Short probes (4 chains, 3,000 iterations per temperature, two seeds, 6 to 22 s each) plateau at residuals 0.14 to 0.17 (rank 8), 0.227 (7), 0.31 to 0.32 (6), 0.311 (5) and 0.385 (4); that says nothing about the cell until a run with the loop's settings is done |
| T5 | 3 | 4 by monotonicity; 5 by slice-and-lift once chi(T5^2) is exact | 27 (product) | exclusions at three ququints need `qutrit_codes.all_codes` generalised to p = 5 (2.46 M states at 125 bytes each is 300 MB, the same size class as four qutrits); the phase-code enumeration is already written for general p except for its `p != 3` guard and the monomial basis, which is the same y_i, y_i^2, y_s y_t for every odd prime |
| N5 | 1 | 2 | 2 | settled |
| N5 | 2 | 4 (rank 3 excluded, margin 0.045) | 4 (product) | settled here |
| N5 | 3 | 4 | 8 | the interesting cell: rank 7 gives gamma 0.403 < log_5 2 = 0.431 |

Effort for the remaining T5 plumbing (schema enum, site constants,
CONTRIBUTING row, two certificates `cert_t5_m1_rank3.py` and
`cert_t5_m2_rank3.py` on the pattern of `cert_t3m1_rank2.py` and
`cert_n_m3_rank3.py`, and the `--orbit` choices of `to_witness`, which
follow `ORBIT_P` automatically): about two hours. The autoresearch loop
runs for T5 as is, since `generate_random_stabilizer_state(m, p=5)` and
`run_sa_pauli_expansion(p_prime=5)` accept p = 5 (the probes above ran
through them).

## 4. Recommendation

First CS, then T5. Both add a cell whose exact value is new and cheap to
certify, and they exercise the two different generalisations the
pipeline needs (block width and local dimension), so doing one of each
leaves the code ready for CCZ and for p = 7 without further design.

CS first because it has the most to show per unit of work: chi(|CS>) = 2
and chi(|CS>^2) = 3 are both exact today, the exponent 0.7925 beats the
trivial 1 at the first non-trivial cell, the comparison exponents from
the literature are concrete (extent 0.678, T-synthesis 1.19), and the
m = 3 lower bound 4 is one block-lift away. It also settles a small
question on the way: the transversal-number bound 2^tau for hypergraph
states is not tight even for two disjoint edges of size two, so there is
no reason to expect 2^m to be tight for |CCZ>^m either, which is what
makes the CCZ m = 2 cell worth an annealer campaign afterwards.

T5 second because it opens a dimension with no published exact rank at
all, its one- and two-copy cells are settled or bounded by the tools as
they stand (this branch), and the m = 2 cell 4 <= chi <= 9 is the
cheapest open cell on the whole board to move: the dictionary has 3,900
states, a rank-4 exclusion is minutes, and any rank below 9 found by the
annealer is a new exponent. Its weakness is the absence of a literature
exponent to compare against; the honest baseline is the single-copy
value log_5(3) = 0.683, labelled as such.

CCZ third: the only new information available cheaply is chi(|CCZ>^2)
>= 3 by the block lift, and the upper side needs a six-qubit annealer
campaign. N5 is settled at m <= 2 and its m = 3 cell needs three-ququint
exclusions or the annealer; it is a natural fourth once the T5 plumbing
exists, since it costs only an `orbit_state` entry.

## 5. Concrete code changes

For T5 (this branch has the first three items):

1. `verify_challenge/stabrank_verify.py`: `orbit_state("T5")`,
   `ORBIT_P`, `ORBIT_LABEL`. Done.
2. `verify_challenge/rank_exclusion.py`: `dictionary` and
   `clifford_group` for any odd prime, `_is_prime`,
   `_distinct_up_to_phase`. Done.
3. `tests/test_ququint_dictionary.py`. Done.
4. `schema/bound.schema.json`: add `"T5"` to the `orbit` enum and its
   description.
5. `site_challenge/build.py`: `ORBIT_ORDER`, `SYSTEM["T5"] = "ququint"`,
   `BASELINE["T5"] = (math.log(3, 5), "log₅3")` with the orbit text
   saying it is the single-copy value and not a published exponent,
   `ORBIT_TEX`, `BASE_TEX`, `ORBIT_DEF`, `COLOR`.
6. `verify_challenge/cert_t5_m1_rank3.py` (rank-2 exclusion over 30
   states plus the exact non-stabilizer argument) and
   `cert_t5_m2_rank3.py` (`run_certificate([("T5", 2)], controls=[("T5", 1)])`
   with a positive control that the rank-3 decomposition of |T5> is
   found). Both run in seconds.
7. `verify_challenge/qutrit_codes.py`: lift the `p != 3` guard in
   `all_codes` (the monomial basis and the RREF enumeration are already
   written in p) so that three-ququint exclusions have a dictionary;
   `rank_exclusion_codes.py` follows. Needed only for T5 and N5 at m = 3.
8. CONTRIBUTING.md orbit table.

For CS (and then CCZ with width 3):

1. `stabrank_verify.py`: `ORBIT_WIDTH = {"CS": 2, "CCZ": 3}` with default
   1; `orbit_state("CS")` returns the 4-vector (1, 1, 1, i)/2 and
   `orbit_state("CCZ")` the 8-vector with -1 at |111>; `verify_upper`
   uses `n = ORBIT_WIDTH.get(orbit, 1) * m` in `stabilizer_vector` and
   in the accumulator.
2. `rank_exclusion.py`: `symmetry_orbit_reps` takes the local group from
   a new `local_clifford_group(orbit)` (closure of H_i, S_i, CNOT for
   width 2; explicit generators for width 3) and `_copy_permutation`
   takes a block width. `psi_for` is unchanged.
3. `slice_lift.py`: `lifts_qubit_block` and a width-aware
   `all_decompositions` / `lift_all`, as in section 2.
4. `to_witness.py`, `fit_coeffs.py`, `autoresearch/run.py`,
   `autoresearch/kopt.py`: replace `m` by `ORBIT_WIDTH * m` wherever it
   stands for the qubit count.
5. `validate_bounds.py`: per-orbit m cap (CS 4, CCZ 2); schema enum.
6. Certificates: `cert_cs_m1.py` (exact non-stabilizer argument via
   `term_from_vector`), `cert_cs_m2_rank3.py` (rank-2 exclusion over
   36,720 states, seconds), `cert_cs_m3_lift.py` (block lift from the
   sixteen rank-3 decompositions of |CS>^2), and the `m = 2` upper bound
   as the verified witness above.
7. Site constants and CONTRIBUTING as for T5, with `SYSTEM["CS"] = "two
   qubits"` and per-copy wording in the orbit text.

Estimated effort: CS end to end (items 1 to 7 including the block lift)
one to two days; T5 remaining plumbing two hours, plus half a day for the
three-ququint code dictionary if the m = 3 cells are wanted.


## Update 2026-09-21: chi(T5^2) >= 5

A rank-4 exclusion over the 3,900 two-ququint stabilizer states (pivot-pair
search from the 98 orbit representatives of the unitary symmetry group of
order 50, 82 s on a laptop core) finds no rank-4 decomposition of |T5>^2,
so with the rank-3 exclusion above and the product bound the cell is
5 <= chi(T5^2) <= 9. Not yet a board cell: the orbit enum, site constants
and a certificate are still to be added (section 5).
