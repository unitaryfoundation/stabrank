# New orbits beyond the six single-qudit cells: design note

Status (2026-09-20). Design for extending the board past its six orbits
(four qutrit, two qubit), with the literature values that would seed each
new cell, what breaks in the pipeline for each kind of orbit, and the
smallest change that makes the first new orbit verifiable. The code slice
implemented alongside this note is the ququint T-type orbit `T5`:
`orbit_state("T5")`, `ORBIT_P["T5"] = 5`, a prime-generic
`rank_exclusion.dictionary(p, n)` and `clifford_group(p)`, and tests
(`tests/test_ququint_dictionary.py`). As of 2026-09-21 the schema enum,
the site constants, two certificates and four bound files are in place
too, so T5 is a board orbit; section 5 records what was added and what
remains (the three-ququint dictionary).

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
| T5 | 2 | 5 (`certify_rank3` in 1 s, group of order 100, margin 0.066; then the rank-4 pivot-pair exclusion from the 98 unitary-symmetry representatives, 50 s) | 8 (annealer witness of 2026-09-21, verified) | annealer for ranks 5 to 7. Short probes (4 chains, 3,000 iterations per temperature, two seeds, 6 to 22 s each) plateau at residuals 0.14 to 0.17 (rank 8), 0.227 (7), 0.31 to 0.32 (6), 0.311 (5) and 0.385 (4); that says nothing about the cell until a run with the loop's settings is done |
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

For T5, everything but item 7 is done as of 2026-09-21:

1. `verify_challenge/stabrank_verify.py`: `orbit_state("T5")`,
   `ORBIT_P`, `ORBIT_LABEL`. Done.
2. `verify_challenge/rank_exclusion.py`: `dictionary` and
   `clifford_group` for any odd prime, `_is_prime`,
   `_distinct_up_to_phase`. Done.
3. `tests/test_ququint_dictionary.py`. Done; it now also checks that the
   four T5 bound files validate and that the certificates' printed claims
   match their `expect` fields.
4. `schema/bound.schema.json`: `"T5"` in the `orbit` enum. Done.
   `validate_bounds.py` caps T5 at `m <= 5`, since the schema's global
   cap of 8 is a 3^8-amplitude budget.
5. `site_challenge/build.py`: `ORBIT_ORDER`, `SYSTEM`, `BASELINE["T5"] =
   (log_5 3, "log₅3")`, `UNPUBLISHED = {"T5"}` with `base_word` and
   `base_note` so every page that shows the baseline calls it a
   single-copy product bound, `ORBIT_TEX`, `BASE_TEX`, `ORBIT_DEF`,
   `COLOR`, `ORB_TEX_NAME`; `report.py` shows "none; single-copy product
   bound" in the reference column. Done.
6. Certificates. `cert_t5_m1_rank2.py` is exact: all 30 states have
   amplitudes 0 or a power of w_5, so every 3 x 3 minor of [s | t | psi]
   is decided in Z[w_5] by integer arithmetic, and the bound file declares
   `exact: true`. `cert_t5_m2_rank4.py` runs `certify_rank3` (1 s) and then
   the rank-4 pivot-pair search of `slice_lift.all_decompositions` from
   the 98 unitary-symmetry representatives (50 s), with the rank-3
   decompositions of |T5> and the product decomposition of |N5>^2 as
   positive controls. Done.
7. `verify_challenge/qutrit_codes.py`: lift the `p != 3` guard in
   `all_codes` (the monomial basis and the RREF enumeration are already
   written in p) so that three-ququint exclusions have a dictionary;
   `rank_exclusion_codes.py` follows. Needed only for T5 and N5 at m = 3.
   Not done.
8. CONTRIBUTING.md orbit table and the baseline convention for orbits
   with no published exponent. Done.

Bound files: `T5-m1-upper-3.json` (verified), `T5-m1-lower-3.json`
(exact certificate, verified), `T5-m2-upper-8.json` (annealer witness,
verified) and `T5-m2-lower-5.json` (reproduced).

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
so with the rank-3 exclusion above the cell was 5 <= chi(T5^2) <= 9. The
same day the annealer (4 chains, 8000 iterations per temperature, seed
7002) found a rank-8 decomposition that refits exactly and verifies
symbolically, so the cell is 5 <= chi(T5^2) <= 8 and T5 is a board orbit
with four cells; section 5 lists what was added.

## 6. Magic cat states: a track indexed by the qubit count (2026-09-24)

The board's seven orbits are all of the form "one state, m copies". The
magic cat states of Qassim, Pashayan, and Gosset (arXiv:2106.07740, QPG
below) are the first track that is not: |cat_m> is one m-qubit state per
m, not a tensor power of anything, and the family is what carries the
published qubit exponent. This section fixes the definition the board
adopts, how the columns apply, and which cells the two public sources
fill. Every value comes from QPG or from Kissinger, van de Wetering, and
Vilmart (arXiv:2202.09202, KvdWV below); nothing in this section is a new
bound.

### 6.1 Definition

QPG Eq. (3), with |T> = 2^{-1/2}(|0> + e^{i pi/4}|1>) and |T_perp> = Z|T>:

    |cat_m> = 2^{-1/2} (|T>^{(x) m} + |T_perp>^{(x) m}).

Expanding, the odd-weight strings cancel and the even-weight ones double:

    |cat_m> = 2^{-(m-1)/2} sum_{x in F_2^m, |x| even} i^{|x|/2} |x>.

KvdWV (Section 4.1) write the same state as 2^{-1/2}(I + Z^{(x) n})|T>^{(x) n}
and as two ZX spiders with phases pi/4 and 5 pi/4, normalized by
1/sqrt 2^{n+1}; the three forms agree on the nose, and
`tests/test_families.py` checks the closed form against the tensor-power
definition symbolically for m <= 6. The amplitudes are fourth roots of
unity times the one scalar (1/sqrt 2)^{m-1}, so the target lives in Q(i)
up to that scalar, a subfield of the Q(zeta_8) that the qubit witnesses
already use. The two-qubit case |cat_2> = 2^{-1/2}(|00> + i|11>) is a
stabilizer state and |cat_1> = |0>, so the track starts at m = 2.

Orbit key `cat`, local dimension 2, and `m` is the number of qubits. The
target constructor `target_vector("cat", m)` builds the closed form
directly; `orbit_state("cat")` raises, since there is no single-qubit
state to take powers of, and the code that assumed one (the symmetry
reduction of `rank_exclusion`, the slice lemma in `slice_lift`) is
listed in 6.4 as not yet adapted.

### 6.2 How the board's columns apply

Rank. chi(cat_m) per m, upper and lower, exactly as for every other cell.

Exponent. A cat cell has no per-copy exponent of its own, but every cat
cell bounds the qubit exponent through the gluing identity that QPG use
in the proof of their Theorem 1 and that KvdWV restate in ZX (their
Section 4.2). With <cat_2| = 2^{-1/2}(<00| - i<11|) applied to the last
qubit of |cat_a> and the first of |cat_b>, the inner products are

    <cat_2|(|T>|T>)           = 2^{-1/2}(1/2 - i e^{i pi/2}/2) = 2^{-1/2},
    <cat_2|(|T_perp>|T_perp>) = 2^{-1/2}(1/2 - i e^{i pi/2}/2) = 2^{-1/2},
    <cat_2|(|T>|T_perp>)      = 2^{-1/2}(1/2 - i (-i)/2)     = 0,

and the fourth vanishes in the same way, so

    (I (x) <cat_2| (x) I)(|cat_a> (x) |cat_b>) = (1/2) |cat_{a+b-2}>.

A bra that is a stabilizer state does not increase the number of terms,
so chi(cat_{a+b-2}) <= chi(cat_a) chi(cat_b). Chaining l copies of
|cat_m> gives chi(cat_{l(m-2)+2}) <= chi(cat_m)^l, and QPG Eq. (4),

    chi(T^{(x) m}) / 2 <= chi(cat_m) <= chi(T^{(x) m}),

(upper side: |cat_m> = 2^{-1/2}(I + Z^{(x) m})|T>^{(x) m} and the projector
is a stabilizer projector; lower side: |T><T| = (I + A)/2 with A the
Clifford e^{-i pi/4} S X, so |T>^{(x) m} is proportional to |cat_m> +
(A (x) I)|cat_m>) turns that into chi(T^{(x) (l(m-2)+2)}) <= 2 chi(cat_m)^l.
Hence

    limsup_n log_2 chi(T^{(x) n}) / n <= log_2 chi(cat_m) / (m - 2)

for every m >= 3. This is QPG's Theorem 4 at k = 1 (the repetition code),
made constructive: the chain replaces the hypothesis (their Eq. 14) that
chi(T^{(x) n}) has a well-defined exponent. At m = 6 and rank 3 it is the
published log_2(3)/4 = 0.3963.

The board therefore shows, in the exponent column of a cat cell,

    gamma_H(cat_m, r) = log_2(r) / (m - 2)        (m >= 3; none at m = 2),

labeled as the implied H-type exponent, and measures the track against
the published log_2(3)/4 like the two qubit orbits. `implied_exponent`
in `stabrank_verify.py` carries the rule, and `next_target` uses m - 2 in
place of m. By the board's usual rule (the largest rank below the threshold
at each m, then the lowest exponent among them) it names chi(cat_7) <= 3
(0.3170); the cells nearest the known values that would beat the exponent
are chi(cat_8) <= 5 (0.3870; it would give chi(T^{(x) 8}) <= 10 through
Eq. 4) and chi(cat_10) <= 8 (0.3750; QPG's glued |cat_10> has 9 terms). Since
|T> and the board's |H> = cos(pi/8)|0> + sin(pi/8)|1> are one Clifford
orbit, chi(T^{(x) m}) is the `qubit_H` cell at m.

Relation to the `qubit_H` cells. Eq. (4) ties every cat cell to the H cell
at the same m in both directions, so the board's intervals must satisfy
cat_lower <= H_upper and H_lower <= 2 cat_upper at every m; the test
checks this on the committed bound files. Nothing new follows in either
direction today: from the attested chi(H^5) = chi(H^6) = 6 one gets
chi(cat_5), chi(cat_6) >= 3, which QPG already prove; from chi(H^7) >= 6
one gets chi(cat_7) >= 3, which QPG's monotonicity gives; and
chi(cat_8) <= 6 gives chi(H^8) <= 12, already on the board from KvdWV. So
no repo-derived cat cell is filed, and none of the cat cells changes an H
cell.

### 6.3 Cells and tiers

QPG Table 1 (bottom row), checked against their text and appendix:

| m | chi(cat_m) | where | tier here |
|---|---|---|---|
| 2 | = 1 | Eq. (5): a stabilizer state; lower bound trivial | verified (one-term witness) |
| 3 | = 2 | upper: <0|_4 of the cat_4 decomposition (appendix; KvdWV Sec. 4.1 in ZX); lower: not a stabilizer state, <cat_3|XXI|cat_3> = 1/2 (appendix) | verified upper (projected terms), verified lower (exact rank-1 exclusion) |
| 4 | = 2 | upper: i|E> + ((1-i)/2) 2^{-1/2}(|0^4> - i|1^4>), E the even-weight uniform state (appendix; KvdWV Sec. 4.1); lower: monotone from cat_3 | verified upper, verified lower (exact rank-1 exclusion of cat_4 itself) |
| 5 | = 3 | upper: <0|_6 of the cat_6 decomposition (appendix; KvdWV Sec. 4.1); lower: Lemma 3 of the appendix, chi(cat_5) > 2, a canonical-form reduction plus a computer comparison of Pauli spectra | verified upper, cited lower |
| 6 | = 3 | upper: Eq. (5), 2^{-3/2}(|0^6> - i|1^6>) + 2^{-1/2} e^{3 i pi/4}(|E_6> + i|K_6>), K_6 = prod_{i<j} CZ_ij E_6; lower: monotone from cat_5 | verified upper, cited lower |
| 7 | 3 <= . <= 6 | upper: <0|_8 of the cat_8 construction; lower: monotone from cat_5 | verified upper, cited lower |
| 8 | 3 <= . <= 6 | upper: <cat_2|_{4,5}(|cat_4> (x) |cat_6>), 2 x 3 terms (appendix); lower: monotone | verified upper, cited lower |

The verified upper bounds are mechanical replays of constructions the
papers state, not new decompositions: `research/constructions/qpg_cat.py`
builds the terms numerically exactly as written (the two cat_4 terms, the
three cat_6 terms, their contraction through <cat_2|, and the <0|
projections that QPG and KvdWV both state as the way to smaller m),
converts them with `to_witness.witness_from_vectors`, and the verifier
checks the identity symbolically. The provenance of every file names the
paper, and the method is `literature`. The contraction at m = 8 gives six
nonzero, linearly independent terms and each projection keeps every term,
so the witnesses have exactly the ranks the papers claim; had a projection
collapsed two terms the file would have had to stay cited at the paper's
value rather than record a smaller one.

The lower bounds at m = 3 and m = 4 are exact: `cert_family_rank1.py`
shows the state is not a stabilizer state in integer arithmetic (support
not an affine subspace of F_2^m, or the phase function on it not a Z_4
quadratic form with even cross terms, decided on the exact fourth-root
exponents), which is the rank-1 exclusion. QPG's own argument is the Pauli
spectrum, an equivalent test. The lower bounds at m = 5 to 8 stay cited:
QPG's Lemma 3 rests on a computer comparison of Pauli spectra that is
theirs, not the pipeline's, and the cheapest machine check here would be a
rank-2 exclusion over the 2,423,520 five-qubit stabilizer states (a
dictionary of about 0.6 GB in memory), which is offline work rather than a
certificate under the budget. Monotonicity chi(cat_m) >= chi(cat_{m-1})
(from <0|_m |cat_m> proportional to |cat_{m-1}>) then carries a five-qubit
result to m = 6, 7, 8 by the same projection lemma the qubit_H cells use.

Two points the papers leave to the reader. QPG's appendix opens with "we
have already shown chi(cat_2) <= 2", where Eq. (6) and the table say
chi(cat_2) = 1; |cat_2> is written out as a stabilizer state, so 1 is the
value. And the phase on |K_6>: prod_{i<j} CZ_ij on an even-weight string
gives (-1)^{C(|x|,2)} = (-1)^{|x|/2}, which is the sign the construction
here uses and which reproduces Eq. (5) to machine precision.

### 6.4 Machinery

Ready now: `verify_upper`, `fit_coeffs.fit`, and `to_witness` work from
`target_vector("cat", m)` unchanged, so a new decomposition of any
|cat_m> for m <= 10 (the schema cap; 1024 amplitudes) is checkable today,
and `validate_bounds.py` refuses m < 2. `cert_family_rank1.py` decides
rank 1 exactly for any family cell.

Not adapted: `rank_exclusion.symmetry_orbit_reps` builds its group from
`orbit_state`; for cat the natural group is the qubit permutations, the
Clifford G = (X + Y)/sqrt 2 on any single qubit (it fixes |T> and negates
|T_perp>, so it maps |cat_m> to the sign-flipped cat, which the search
would have to treat as a second target or as a symmetry up to a Clifford
of the dictionary), and complex conjugation composed with Z^{(x) m}. The
slice lemma in `slice_lift` uses that every single-qubit slice of the
target is proportional to the target one size down; for cat the two
slices are <0|_m |cat_m> proportional to |cat_{m-1}> and <1|_m |cat_m>
proportional to G_1|cat_{m-1}>, so both are Clifford images of the
smaller cat and the lift test goes through with one extra Clifford in
the meet-in-the-middle, the same shape of change as the block lift in
section 2. Lean: the reflection route fits, since the target is
(1/sqrt 2)^{m-1} times a vector with entries in {0, 1, i, -1, -i}, all in
the basis B4 = {1, sqrt 2, i, i sqrt 2} of the qubit_H modules; what is
missing is a `cat` branch in `gen_witness_lean.make_orbit` (target
entry i^{|x|/2} on even strings, scalar s2^{m-1}/2^{m-1}, a statistic on
the weight rather than on the count of zeros) and a `catVec m` with its
`catVec_eq_ev` lemma on the Lean side.

### 6.5 Files touched

1. `verify_challenge/stabrank_verify.py`: `FAMILY`, `target_vector` for
   `cat`, `ORBIT_P["cat"] = 2`, `ORBIT_LABEL`, `implied_exponent`, and
   `exponent_copies` (m - 2 for cat); `verify`, `verify_upper`, and
   `verify_lean` report the implied exponent through it.
2. `schema/bound.schema.json`: `"cat"` in the `orbit` enum.
3. `verify_challenge/validate_bounds.py`: `M_MIN["cat"] = 2`.
4. `verify_challenge/cert_family_rank1.py`: exact rank-1 exclusion for
   cat_3 and cat_4.
5. `research/constructions/qpg_cat.py`: the seven witnesses, with a row in
   `research/constructions/README.md`.
6. `bounds/cat-m{2..8}-upper-*.json`, `bounds/cat-m{3..8}-lower-*.json`,
   with `certs/` receipts for the verified ones.
7. `site_challenge/build.py`: `ORBIT_ORDER`, `SYSTEM`, `BASELINE`,
   `ORBIT_TEX`, `BASE_TEX`, `ORBIT_DEF`, `COLOR`, `FAMILY_KET` for the
   ket rendering, `next_target` on `exponent_copies`, and the `--no-verify`
   path classifies pure citations without a receipt.
   `site_challenge/report.py`: `PUBLISHED_REF["cat"]`.
8. `CONTRIBUTING.md` orbit table; `docs/refs.bib` unchanged (both sources
   already present).
9. `tests/test_families.py`.
