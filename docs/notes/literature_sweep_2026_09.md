# Literature and technique sweep, 2015 to September 2026

Date of sweep: 2026-09-20. Question: is there anything in the published or
preprint literature that lowers a stabilizer-rank upper bound on a board
cell, or that is worth a new board orbit. Throughout, chi(|M>^m) is the
exact stabilizer rank of m copies of the orbit state |M>, and
gamma = log_p(chi)/m is the per-copy exponent the board ranks, with p = 2
for qubits and p = 3 for qutrits.

Board state at the time of the sweep (`docs/ledger.json`, `bounds/`):

| orbit | published gamma | best cell on board | record-capable open cells |
|---|---|---|---|
| S | 0.3155 (m=2, rank 2) | matches | m=5 rank 5 or 6 (cell 5 <= chi <= 8); m=6 rank 7 (0.2952, plateau at sqrt(8/27)) |
| N | 0.4206 (m=3, rank 4) | matches | m=4 rank 5 or 6 (cell 5 <= chi <= 7) |
| H3 | 0.4206 (m=3, rank 4) | matches | m=4 rank 5 or 6 (cell 5 <= chi <= 8) |
| T3 | 0.5000 (m=2, rank 3) | matches | m=4 rank 7 or 8 (cell 6 <= chi <= 9); m=5 rank <= 15 (cell 6 <= chi <= 18) |
| qubit_H | 0.3962 (asymptotic) | 0.4308 (m=6, rank 6) | m=6 rank 5 (0.3870); nothing at m=7, 8 |
| qubit_T | 0.3962 (m=4, rank 3) | matches | m=6 rank 5 (0.3870); nothing at m=7, 8 |

How the sweep was done. arXiv API full-text queries for "stabilizer rank",
"stabilizer extent", "stabilizer decomposition(s)" sorted by date (all hits
2015 to 2026-09-17 read by title, 35 read in full); the Semantic Scholar
citation graphs of arXiv:2106.07740, 1808.00128, 2110.07781, 2106.03214,
2305.10277 and 2003.01130 restricted to 2024 onward (254 citing papers,
titles screened, 20 read); targeted web searches for each name and topic in
the brief. Every value below was read off the paper text, not an abstract or
a search summary, unless marked unconfirmed.

## 1. Sources

| source | arXiv | what it contributes | on the board | usable here |
|---|---|---|---|---|
| Bravyi, Smith, Smolin 2016 | 1506.01396 | chi(H^m) <= 2, 2, 3, 4, 6, 7 for m = 1..6 by Monte Carlo; Conjecture 1 (magic states minimize rank among single-qubit states); Omega(sqrt m) lower bound | yes (m <= 6) | no new cells; Conjecture 1 is refuted in part by chi(F^4) = 3 < chi(H^4) = 4 on the board |
| Bravyi, Gosset 2016 | 1601.07601 | approximate rank 2^{0.23 m} for |H>^m, stabilizer extent | no | approximate only |
| Bravyi, Browne, Calpin, Campbell, Gosset, Howard 2019 | 1808.00128 | chi(psi^m) <= C(2^n + m - 1, m) for m <= 5 (symmetric subspace, Lemma 5), so chi(psi^m) <= m + 1 for single-qubit psi and m <= 5; numerics chi(T^7) = 12; |CCZ> = (2/9)(1 + CZ_{12}X_3)(1 + CZ_{13}X_2)(1 + CZ_{23}X_1)|+^3>, extent xi(CCZ) = 16/9 with stabilizer fidelity 9/16 at |+^3>; hypergraph-transversal remark for Clifford+CCZ states (open question) | m + 1 bound gives nothing below the board's values | the symmetric-subspace bound at p = 3 gives C(m + 2, 2), worse than every board value |
| Qassim, Pashayan, Gosset 2021 | 2106.07740 | cat_m = (|T>^m + |T_perp>^m)/sqrt 2 with chi(T^m)/2 <= chi(cat_m) <= chi(T^m); chi(cat_2) = 1, chi(cat_3) = chi(cat_4) = 2, chi(cat_5) = chi(cat_6) = 3 (exact, appendix), chi(cat_7) <= chi(cat_8) <= 6 via <cat_2|_{4,5}|cat_4>|cat_6> ∝ |cat_8>; contraction of cat_6 chains gives chi(cat_{4l+2}) <= 3^l and gamma <= log_2(3)/4; Table 1: chi(T^m) <= 2, 3, 4, 6, 6, 12, 12 for m = 2..8; same for the face state F with an explicit 3-term cat_6(F) (Eq. 8-9); Theorem 4: any [m, k] linear code L with k < m/2 gives gamma <= log_2 chi(L_hat)/(m - 2k); open question whether any code beats log_2(3)/4 | m=6 rank 6 (qubit_H) as a verified witness; qubit_T m=6 rank 6 by product | cat_8 <= 5 would give chi(T^8) <= 10 (0.4152 at m=8, asymptotic 0.3870); the code-state question is a direct search target (section 2) |
| Kocia 2020 | 2012.11739 | chi(T^12) <= 47, exponent 0.4629; Gauss-sum rank as an asymptotic lower bound for single-Pauli measurement cost | no (m = 12 exceeds the cap) | nothing below 0.3962 |
| Kocia, Sarovar 2021 | 2003.01130 | qutrit T: chi_1 = chi_2 = 3, chi_3 = 8 (numerical, flagged as possibly unconverged); Gauss-sum ranks xi_k of the Wigner function of |T3>^k: 3, 3, 8, 9, 24, 24, <= 72, 72, <= 216, 216, <= 486, 486 for k = 1..12, and "3^{0.482 t}" (k = 6), "3^{0.469 t}" (k = 12) | m = 1, 2, 3 | xi_k is not a stabilizer rank and is not claimed to bound chi (xi_5 = 24 > 18 = the board's chi(T3^5) upper bound shows it is not a lower bound either); but xi_6 = 24 = 27 - 3 = 3 x 8 is a specific hint that each Z-eigensector of |T3>^6 might have rank 8 (recipe R1) |
| Kocia, Tulloch 2022 | 2202.01233 | weak-simulation prefactor; quotes chi_4 = 4, chi_8 = 12, chi_16 = 108 from QPG | no | nothing |
| Peleg, Shpilka, Volk 2022 | 2106.03214 | chi(H^n) = Omega(n) (constant about 1/100), approximate Omega(sqrt n / log n) | cited | asymptotic; at m <= 8 it is below every board lower bound |
| Labib 2022 | 2107.10551 | chi(|psi_U>^n) = Omega(n) for the Howard-Vala state at every prime p (p = 3: the T3 orbit) | cited | asymptotic only; the argument needs the nonclassical cubic phase, so it does not cover S, N, H3 |
| Lovitz, Steffan 2022 | 2110.07781 | chi(T^n) >= (n+1)/(4 log_2(n+1)) via exponentially increasing amplitude subsequences (Theorem 3.2); two-qubit states psi_alpha ∝ e00 + alpha(e01 + e10) with chi = 2 and chi(psi ⊗ psi) = 4; chi(phi ⊗ phi) <= 3 for every single-qubit phi; generic rank chi_n = O(2^{n/2}) and it suffices to work over the reals for it (Prop. 5.4) | cited | Theorem 3.2 gives < 1 at m = 8; the amplitude-modulus hypothesis fails for S (all nonzero amplitudes equal), as arXiv:2605.28586 notes |
| Heimendahl, Montealegre-Mora, Vallentin, Gross 2021 | 2007.04363 | stabilizer extent is not multiplicative in general (generic high-dimensional states); multiplicative for products of 1, 2, 3-qubit states (from BBCCGH) | no | extent, not rank |
| Mehraban, Tahmasbi 2024 | 2305.10277 | approximate rank Omega~(n^2) | cited | asymptotic |
| Kalra, Sinha 2026 (Quantum 10, 2179) | 2503.04101 | Theorem 2: if chi(psi) = k and F_S(phi) is the stabilizer fidelity of phi then |<phi|psi>|/sqrt(F_S) <= sqrt(e)(2k)^{(2k+1)/2}; Barnes-Wall norm N: multiplicative, N(|CS>) = 2, N(phi) divides 2^m for CS-count m; the approximate norm N_{delta+delta_0} is bounded linearly in the approximate rank chi_delta (Theorem 7); fidelity amplification; product states of rank 2^n are dense; open problems 56-59 | cited | Theorem 2 at m <= 8 is trivial (k = 1 already satisfies it for |H>^6); the norm N is a candidate certificate primitive for |CS>-type orbits, not for the board's single-qudit orbits |
| Labib, Russo 2026 | 2605.28586 | the board's baselines; open directions: asymptotic lower bounds for S, qutrit analogues of the approximate-rank bounds, a Clifford conversion protocol for S | yes | the baselines |
| Kliuchnikov, Schönnenbeck 2024 | 2404.17677 | minimal vectors of Barnes-Wall lattices are stabilizer states; post-selected stabilizer circuits over dyadic cyclotomic fields | no | foundation for Kalra-Sinha; qubit only |
| de Silva, Yin, Strelchuk 2024 | 2311.17384 | the space of linear dependencies of n-qubit stabilizer states has a basis of 3-term dependencies (Theorem 3, via a splitting lemma |s> = (t_1 + t_2)/2 on half supports); every decomposition of any state is reachable from the computational-basis one by adding dependent triples; extent computed for 6-qubit |C5Z> and Dicke states | no | a structured alternative to annealing: search over 3-term moves. Qubit only; the qutrit analogue (splitting along a hyperplane into three third-support states) is an easy lemma to write and is what a structured merge search on the qutrit board would rest on |
| Kissinger, van de Wetering 2022 | 2109.01076 | ZX-driven decompositions in chunks of 2 to 6 T states using BBCCGH and QPG terms | no | consumes known decompositions |
| Kissinger, van de Wetering, Vilmart 2022 | 2202.09202 | cat_4 has rank 2 (2^{0.25 t} when present); a 5-T-state decomposition into 3 terms leaving one T (4-to-3, chi(T^t) <= 3 chi(T^{t-4})); the cat_{4k+2} gluing of QPG in ZX form | yes, finite m | chi(H^7) <= 9 (beats QPG's table value 12), chi(H^8) <= 12, chi(H^10) <= 18 (0.4170); witnesses built and verified 2026-09-21 (`bounds/qubit_H-m7-upper-9.json`, `-m8-upper-12`, `-m10-upper-18`); the asymptotic exponent is unchanged |
| Koch, Yeung, Wang 2023 | 2307.01803 | tensored ZX "star edges" (the triangle node, which has stabilizer rank 2) admit decompositions with 2^{0.774 t} terms instead of 2^t; benchmark on Clifford+T+CCZ circuits | no | the only sub-product bound for a CCZ-like resource found; whether it transfers to chi(|CCZ>^k) < 2^k is unconfirmed (candidate orbit C1) |
| Codsi, Laakkonen 2026 | 2603.06377 | tree-width and rank-width simulation bounds with T^{tw} scaling; the CCZ transversal remark of BBCCGH restated | no | nothing for exact rank |
| de Colnet, Geerts, Hai, Laarman, Lee, Pérez 2026 | 2605.29944 | quadratic sums-of-powers dynamic program; uses the QPG 0.3963 decomposition as a black box | no | nothing |
| Camillo, Peres, Heinrich, Bermejo-Vega 2025 | 2510.18977 | real, diagonal and real-diagonal unitaries have optimal extent decompositions inside the corresponding Clifford subgroups; extents up to 7 qubits | no | extent, but the symmetry reduction is reusable for an extent baseline on board cells |
| Arunachalam, Dutt 2026 | 2606.07425 | tomography of bounded-extent states; quotes xi(psi) <= O((2k)^{(2k+1)/2}) for rank k | no | nothing |
| Bu, Gu, Jaffe 2025 | 2508.15908 | quantum uniformity norms characterize the Clifford hierarchy; explicit values for CCZ at every prime d | no | nothing for rank |
| Jain, Prakash 2020 | 2003.07164 | complete list of single-qutrit and single-ququint Clifford eigenstates: qutrit S, |H,1>, N+, |XV_S> = xi^5|0> + xi^4|1> + |2> (xi = e^{2 pi i/9}, the T3 orbit) plus two degenerate families; ququint: 8 non-degenerate states, |B',-e^{±2 pi i/3}> most magic (conjectured), |B,-1> most symmetric, |XV_S,1> = |0> + |1> + w5^3|2> + |3> + w5^2|4>; no strange-state analogue for any prime p > 3 (section 6) | the four qutrit orbits | the source list for p = 5 orbits (section 3) |
| Howard, Vala 2012 | 1206.1598 | qudit pi/8 gates U_upsilon for every prime p, their eigenstates |psi_U>, robustness tables for p = 2, 3, 5, 7 | T3 is the p = 3 case | the p = 5, 7 "T-type" states |
| Campbell, Anwar, Browne 2012 | 1205.3104 | Reed-Muller distillation in all prime dimensions; the states distilled are the Howard-Vala eigenstates | no | same states |
| Huang, Love 2019 | 1808.02406 | qudit approximate rank: a d-term orbit decomposition |M_d> = sum of d stabilizer states under powers of a Clifford; approximate exponents 2^{0.23 t}, 3^{0.32 t}, 5^{0.41 t}, 7^{0.40 t} for d = 2, 3, 5, 7 | no | seeds the approximate-rank column for p = 5, 7 orbits; the d-term orbit decomposition is the trivial chi <= p |
| Prakash 2020 | 2003.02717 | Golay-code distillation of S and N | no | names and motivation only |
| Seddon, Regula, Pashayan, Ouyang, Campbell 2021 | 2002.06181 | dyadic negativity, mixed-state extent, generalized robustness | no | nothing for exact rank |
| Khesin 2025 (thesis) | 2501.17959 | graph-formalism representation of the QPG decompositions; chi(U|psi>) <= 2 for U in the third Clifford level on a stabilizer state | no | restates known values |
| Vollmeier 2025 (thesis) | 2503.03798 | multi-control Toffoli gate state has rank 2 (star-edge decomposition D1) | no | trivial |
| Wan, Zhong (2025); Wan, Zhong, Zapirain (2026) | 2509.01224, 2509.08658 | d = 3 and d = 5 magic-state-cultivation circuits as sums of 4 and 8 Clifford ZX diagrams | no | circuit specific |
| Ohta, Sakurai 2025 | 2506.11725 | maximal-magic states from E8, BW16, E6 shells; the 45 one-qutrit maximal-magic states split into two Clifford orbits | no | no rank content |
| Zhu, Mao, Yi 2024 | 2410.13575 | third moments of qudit Clifford orbits; exact qutrit 3-designs from a magic orbit | no | nothing for rank |
| dring-05, "qutrit-t3-structural-obstructions" (GitHub, 2026-09) | none | two restricted theorems on |T3>^4: (i) any decomposition whose supports all lie in fibers of one nonzero linear functional has >= 9 terms; (ii) an 8-term decomposition with exactly one support crossing the total-sum fibers must have that support equal to F_3^4; explicitly does not prove chi(T3^4) >= 9 | no | unconfirmed here (repository, not a paper; not re-run); if correct it is a search constraint for the T3 m=4 rank-8 cell (recipe R5) |
| machine-search papers (AlphaEvolve, FunSearch, RL, LLM-guided) on stabilizer decompositions, 2025-2026 | | none found | | |

Not rank-relevant after reading, listed so the next sweep skips them:
2609.14252, 2609.16929, 2604.27058 (Clifft), 2606.07425, 2512.20787,
2602.22201, 2511.05478, 2607.18400, 2608.14798, 2604.00766 (coherent-state
rank, bosonic analogue), 2410.24202, 2510.05890.

## 2. Constructions to try on existing cells

Every recipe names the state, the cell, the rank that would move it, the
exponent, and the computation that decides it. Ranks and exponents were
recomputed here; "product" means the sub-multiplicative bound from the
board's own record cells.

R1. T3 at m=6 by Z-eigensectors, sector rank 8. State: the three
Z^{⊗6}-eigensector projections c_j of |T3>^6, each a 5-qutrit carry state
psi_j(x) = w3^{ceil((sum x - j)/3)} on F_3^5 (`autoresearch/run.py`,
`sector_target`, orbit name `T3sector0` with m = 5). Cell: T3 m=6, which the
board does not hold; the product from m=2 gives 27 (gamma 0.5000). Target:
rank 8 in each sector gives chi(T3^6) <= 24 and gamma = log_3(24)/6 =
0.4821, the first value below 1/2 for this orbit at a finite m; by the
sector contraction c_j ⊗ c_k -> c_{j+k} it gives chi(T3^10) <= 192 (0.4778)
and asymptotically log_3(8)/4 = 0.4732. Evidence: Kocia and Sarovar's
Wigner-function count for |T3>^6 is xi_6 = 24 = 27 - 3, obtained from "two
sets of quadratic Gauss sums evaluating to the same value" on top of the
9-per-sector product; that is a merge of exactly one term per sector at the
Gauss-sum level. It is not a stabilizer-rank statement (their xi_5 = 24
exceeds the board's chi(T3^5) <= 18), so it is a hint about where to look,
not a bound. The sector product decomposition has 9 terms (three m=2 carry
lines with sum sigma_1 + sigma_2 + sigma_3 ≡ j), with equal coefficients on
terms of equal integer weight |sigma|, so the merge to test first is inside
a weight class. What to compute: `run.py T3sector0 5 8` warm-started from
the 9-term sector product with one term pruned (a witness file for the
sector product has to be written first; `--warm-from` expects one), and the
same from random starts at the manifest's 20000 iterations; the run log has
no entry for this cell. Three random-start seeds run here (4 chains, one at
3000 and two at 8000 iterations, 48 to 126 s wall each, not logged to
`runs.jsonl`) stopped at residuals 0.4453, 0.4453 and 0.5217; the repeated
value at two different iteration counts is the signature of an exact
plateau, the same behaviour the m=5 sector shows at rank 5 (five plateau
values between 0.448 and 0.464). So the random-start annealer is not the
tool here either; the warm start from the 9-term product and a merge test
inside a weight class are. If a sector reaches rank 8, lift with the cat
construction as in `bounds/T3-m5-upper-18.json` and refit exactly over
Q(w9).

R2. qubit_T (face state F) at m=8, rank 8. Product from the m=4 rank-3
witness (`bounds/qubit_T-m4-upper-3.json`) gives 9 at m=8, gamma 0.3962;
rank 8 gives 0.3750, below the published exponent, on a cell the board does
not hold. Two routes. (a) Merge search from the 9-term product on 256
amplitudes: the terms are pairwise products of the three m=4 terms, which
have flats of dimension 3, 4, 3 with coefficients of equal modulus 2/3 and
phases 0, pi/12, -pi/12, so the nine product coefficients fall into five
phase classes of sizes 3, 2, 2, 1, 1 and the size-3 class (the three terms
of phase 0) is the first merge to test; the annealer warm-started from the
9-term product with one term pruned is the direct test. (b) QPG's cat and code states for F: QPG give
cat_6(F) explicitly (Eq. 8-9 of arXiv:2106.07740) and their Theorem 4
holds verbatim for F, so any [m, k] code L with k < m/2 and
log_2 chi(L_hat(F))/(m - 2k) < log_2(3)/4 beats the exponent. The
compressed code state is a 2^{m-k}-dimensional annealer target;
`stabrank.target_functions.qubit_magic_code_state_compressed(G)` builds it
for the T phase e^{i pi |x|/4} and needs a one-line variant for the F
amplitudes (cos beta, e^{i pi/4} sin beta with cos 2 beta = 1/sqrt 3). The
cheapest codes that would beat 0.3962 are [9, 2] with chi <= 3 (128-dim
target), [10, 2] with chi <= 5 (256-dim), [8, 1] = cat_8 with chi <= 5
(128-dim), [12, 3] with chi <= 5.

R3. N and H3 at m=6, rank 15. Product from m=3 rank 4 gives 16 (gamma
0.4206, the published value); rank 15 gives 0.4108, and the board holds no
m=6 cell for either orbit. Structure of the products: the N m=3 witness is
the three planes y_i = 2 with phases built from the indicator [y = 2] =
2y^2 + y, plus |+>^3; the H3 m=3 witness is two full-support quadratic-phase
states (Galois conjugates 2y_0^2 + 2y_1^2 + y_2^2 and y_0^2 + y_1^2 +
2y_2^2), the line |0,0,+> and the plane |+,+,0>
(`bounds/N-m3-upper-4.json`, `bounds/H3-m3-upper-4.json`, notes). In the
16 products, the 9 plane-times-plane terms for N and the 4
full-support-times-full-support terms for H3 share coefficients up to the
orbit's phase and are the first merge candidates. What to compute: the
annealer at 729 dimensions warm-started from the 16-term product with one
term pruned; and, independently, the Z^{⊗6}-eigensector decomposition of
|N>^6 and |H3>^6 (Z|N> and Z|H3> are in the respective orbits since Z is
Clifford, so the sectors are 5-qutrit code states as for T3); sector rank
<= 5 in all three sectors gives <= 15.

R4. qubit_H at m=8 through cat_8 and code states. QPG record chi(cat_7) <=
chi(cat_8) <= 6 with only chi >= 3 known below (monotonicity from
chi(cat_6) = 3, which they prove exactly), so chi(cat_8) in {3, 4, 5, 6} is
open. cat_8 rank 5 gives chi(H^8) <= 10 (0.4152 at m=8, a new record cell
but still above 0.3962 within the cap) and gamma <= log_2(5)/6 = 0.3870
asymptotically by the QPG contraction; cat_8 rank 4 would give 0.3333.
cat_7 rank 4 gives chi(H^7) <= 8 (0.4286) and asymptotically log_2(4)/5 =
0.4, which does not beat. The compressed cat_8 target is 128-dimensional
(`qubit_magic_code_state_compressed` with the [8, 1] repetition code), so a
rank-4 or rank-5 annealing run is cheap, and a rank-4 exclusion inside the
128-dimensional code by the board's `rank_exclusion_codes.py` machinery is
the companion. This target is already in the project's open list; the
literature adds nothing beyond QPG's <= 6.

R5. T3 at m=4, rank 8, with the dring-05 constraints. Cell 6 <= chi <= 9;
rank 8 gives 0.4732, rank 7 gives 0.4428. The annealer plateaus at rank 8
on seven distinct exact residuals between 0.333 and 0.366 (M5 draft,
section 3). Two structural facts narrow the search: the board's own note
that the Z-eigensectors of |T3>^4 have rank exactly 3, so a sector-aligned
decomposition cannot beat 9 (`bounds/T3-m5-upper-18.json`, notes); and the
unconfirmed dring-05 theorems, which if correct say a rank-8 decomposition
cannot have all supports inside the fibers of a single linear functional,
and that if exactly one support crosses the total-sum fibers it must be all
of F_3^4. What to compute: first re-derive the two dring-05 statements from
the repository's own exact machinery (they are finite enumerations over 40
projective directions and the quadratic phases on each fiber); then
constrain the annealer's flats so that at least two terms cross every
sector, and run the `kopt.py` completion test on the plateau
configurations with the crossing terms kept.

R6. S: nothing in the literature. |S>^m is supported on {1, 2}^m with
amplitudes ±2^{-m/2}, so every technique that uses amplitude growth
(Lovitz-Steffan, Bravyi-Smith-Smolin's Omega(sqrt m), Kalra-Sinha's
fidelity bound with F_S(S) = 1/2) is either inapplicable or trivial at
m <= 8, and no paper gives an upper-bound construction for a support-2
qutrit state. The open cells S m=5 rank 5 (0.2930) and m=6 rank 7 (0.2952)
stay as they are: structure, not literature.

R7. Lower bounds. No published technique certifies anything at m <= 8
beyond what the board's exhaustive and slice-and-lift certificates already
hold: Peleg-Shpilka-Volk, Labib, Lovitz-Steffan, Mehraban-Tahmasbi and
Kalra-Sinha are asymptotic and evaluate below 2 at every board cell. The one
transferable idea is de Silva-Yin-Strelchuk's 3-term dependency basis: a
qutrit version (split a stabilizer state along a hyperplane of its flat into
three states of a third the support, related by the Z-type phases) would
let a merge search move through dependent triples exactly rather than by
annealing, and would double as a completeness argument for a lower bound
inside a restricted class. Not in the literature for p = 3; it is a lemma to
write, not a citation.

## 3. Candidate new orbits

The verifier today takes a single-qudit orbit state (`orbit_state` in
`verify_challenge/stabrank_verify.py`), tensors it m times, and rebuilds
each term from (k, x0, W, Q, l) with p in {2, 3}. A multi-qudit orbit needs
`orbit_state` to return an n-qudit vector and `target_vector` to tensor it
with dimension p^{nm}; the m <= 8 cap would have to shrink to keep exact
sympy verification feasible (2^{3m} amplitudes for a 3-qubit orbit). A p = 5
orbit needs the odd-prime term formula with 3 replaced by 5 (the same
w_p^{Q(y) + l.y} form; no extra roots of unity as in the qubit case), a
5-adic canonical-form dictionary for the exhaustive certificates (30 states
at n = 1, 3900 at n = 2, about 2.5 million at n = 3, about 7.7 billion at
n = 4, so exhaustive certificates stop at n = 3), and coefficients over
Q(w5) or its extensions.

C1. |CCZ> (3 qubits, p = 2). |CCZ> = 2^{-3/2} sum_{xyz} (-1)^{xyz}|xyz> =
|+++> - 2^{-1/2}|111>, so chi(|CCZ>) = 2 exactly (it is not a stabilizer
state: its stabilizer fidelity is 9/16, arXiv:1808.00128 Eq. 33), and
chi(|CCZ>^k) <= 2^k. Extent 16/9 per copy (multiplicative for 3-qubit
factors), giving approximate-rank exponent log_2(16/9) = 0.83 per copy.
The |Toffoli> state is Clifford-equivalent (Hadamard on the target), and
|CS> = (|00> + |01> + |10> + i|11>)/2 = |++> + ((i - 1)/2)|11> is the
2-qubit analogue with chi = 2 and Barnes-Wall norm N(|CS>) = 2
(arXiv:2503.04101). No paper states chi(|CCZ>^2) or any sub-product bound
for tensored CCZ states directly; the closest is Koch, Yeung and Wang's
2^{0.774 t} for t tensored ZX star edges (arXiv:2307.01803, Section 3.2),
each of rank 2, which is a CCZ-like resource but not literally |CCZ>^{⊗t};
transferring it is the first thing to check. Lower bounds: none published
for this state; the Peleg-Shpilka-Volk directional-derivative argument is
built for quadratic phases on affine subspaces and should give Omega(k) for
the classical cubic (-1)^{sum x_i y_i z_i}, but that is a derivation, not a
citation. Seed values: chi(CCZ) = 2 (m=1, verified by the two-term
identity); chi(CCZ^2) <= 4 (product); a rank-3 witness at m=2 would give
gamma = log_2(3)/2 = 0.7925 per copy against the trivial 1. Verifier needs:
a 3-qubit `orbit_state`, p = 2 terms as today, cap m <= 4 or 5 (2^{15}
amplitudes). The 64-dimensional m=2 target is small enough for an
exhaustive rank-3 exclusion over the 6-qubit dictionary only with the
board's pivot machinery; the annealer settles rank 3 in seconds either way.

C2. p = 5 Howard-Vala state |XV_S, 1> = (|0> + |1> + w5^3|2> + |3> +
w5^2|4>)/sqrt 5 (Jain-Prakash Eq. 4.23; the p = 5 member of the family
whose p = 3 member is T3). Literature values: chi <= 5 at m=1 (the
Huang-Love orbit decomposition into d = 5 stabilizer states, or the
computational basis); no exact value at any m; approximate-rank exponent
5^{0.41 t} (arXiv:1808.02406, Table I); exact lower bound Omega(m)
(arXiv:2107.10551, Theorem 1.1, valid at every prime); distillation by
Reed-Muller codes (arXiv:1205.3104) and robustness tables (arXiv:1206.1598).
The target vector lives in Q(w5) (the phases are fifth roots of unity, no
denominators as in the T3 case, since the cubic w5^{x^3}-type phase is
classical for p > 3), so coefficients stay in Q(w5). Cells to seed: m=1
(chi = 5 or less: exhaustive over 30 states settles it in a second), m=2
(product 25; exhaustive rank-2 and rank-3 exclusion over 3900 states is
feasible; the annealer for rank 4 to 24 on 25 amplitudes is trivial). The
carry trick of Kocia-Sarovar does not apply (the phase is not a lift to
w25), so m=2 is a genuine question.

C3. Other p = 5 Clifford eigenstates (Jain-Prakash section 4): |B', -1>
(most symmetric, real entries), |B', -e^{±2 pi i/3}> (largest mana,
conjectured most magic; real algebraic entries kappa_pm, eta_pm), |H, ±1>,
|H, ±i>-type Fourier eigenstates. Jain-Prakash prove no strange-state
analogue exists for p > 3, so there is no p = 5 orbit with the S state's
orbit size 9 and support 2. No rank literature for any of them; seed values
would be the trivial chi <= 5 and whatever the m=1, 2 exhaustive runs
return. Worth an orbit only after C2, and only if C2 shows a non-trivial
m=2 value.

C4. p = 7 Howard-Vala state: same as C2 with approximate exponent
7^{0.40 t}; dictionary 56 at n = 1, 7^2 (8)(50) = 19600 at n = 2. Lowest
priority.

C5. Hypergraph states (3-uniform complete hypergraph states and relatives):
no stabilizer-rank literature at all; the 2025-2026 papers on them compute
stabilizer Rényi entropy and local stabilizers (arXiv:2602.23687,
2511.15911). Nothing to seed; not recommended.

## 4. Nothing there

Checked and yielding nothing for the board, so the next sweep can skip
them:

- Approximate-rank and extent papers: Bravyi-Gosset 2016, Heimendahl et al.
  2021, Seddon et al. 2021, Camillo et al. 2025, Arunachalam-Dutt 2026.
  None carries an exact rank.
- Asymptotic lower bounds: Peleg-Shpilka-Volk, Labib, Lovitz-Steffan,
  Mehraban-Tahmasbi, Kalra-Sinha. All evaluate below 2 at m <= 8.
- Kocia 2020 (m = 12, 47 terms, 0.4629) and Kocia-Tulloch 2022: above
  0.3962 and beyond the cap. The Kocia-Sarovar qutrit "3^{0.469 t}" is a
  Gauss-sum count for the Wigner function, not a stabilizer-rank bound; the
  board's published T3 exponent of 1/2 is correct as stated.
- BBCCGH Theorem 2 (chi(psi^m) <= m + 1 for m <= 5, single qubit): equal to
  or above every board value; the qutrit symmetric-subspace analogue
  C(m + 2, 2) is far above.
- QPG Table 1 at m = 7, 8 (chi(T^7) <= 12, chi(T^8) <= 12): equal to the
  products 6 x 2 already implied by the board's m=6 cell. Corrected after
  the sweep: the partial decomposition of Kissinger, van de Wetering, and
  Vilmart gives chi(T^7) <= 9, and the board now carries verified witnesses
  at m = 7, 8, 10 (9, 12, 18) from that paper.
- ZX-calculus simulation papers (Kissinger-van de Wetering 2022;
  Sutcliffe-Kissinger 2024; Wan-Zhong 2025, 2026): they consume the QPG and
  BBCCGH decompositions; the new objects are the star-edge decomposition of
  Koch-Yeung-Wang (kept for C1) and the 4-to-3 partial decomposition of
  Kissinger-van de Wetering-Vilmart 2022, now on the board at m = 7, 8.
- Simulation frameworks of 2026 (Clifft, Codsi-Laakkonen, de Colnet et
  al., Tsim, SymFT, quEStab, Quokka#): use decompositions, do not produce
  them.
- Clifford-hierarchy and universality papers (Bu-Gu-Jaffe 2025, Xu-Wang
  2026, Borda-Rincón-Galindo 2025): no rank content.
- Magic-monotone and design papers (Ohta-Sakurai 2025, Zhu-Mao-Yi 2024,
  Gross-Nezami-Walter 2021, the stabilizer-entropy literature): no rank
  content.
- T-count synthesis (Amy-Mosca, Amy-Glaudell-Ross, Glaudell-Ross-van de
  Wetering-Yeh): the only link to rank is chi(psi) <= chi(T^{T-count}),
  which never beats a direct decomposition.
- Machine-search for stabilizer decompositions (AlphaEvolve, FunSearch,
  RL, LLM-guided), 2025-2026: no paper found. The only automated search in
  the literature remains Monte Carlo and simulated annealing (BSS 2016,
  BBCCGH 2019, Kocia-Sarovar 2021, this repository).
- Bravyi-Smith-Smolin Conjecture 1 (a single-qubit magic state has the
  smallest rank among non-stabilizer single-qubit states, and the H and F
  orbits share it): the second clause is already refuted on the board at
  m = 4 (chi(F^4) = 3, chi(H^4) = 4).
- Dicke states: stabilizer extent at 6 qubits (de Silva et al.) and
  Monte Carlo rank searches; not magic states in the board's sense.
