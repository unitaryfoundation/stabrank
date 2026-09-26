# Structured constructions (2026-09)

Scripts behind `docs/notes/constructions_2026_09.md`: non-search attempts at
the cells where a smaller rank would beat a published exponent. Run from the
repository root with `uv run --extra challenge python research/constructions/<script>`.

| script | what it does |
|---|---|
| `common.py` | numeric target, term rebuild from `(k, x0, W, Q, l)`, stored-list loader, fast stabilizer filter |
| `export_data.py` | writes `data/*.json` (the minimal-decomposition lists) as witness terms; `--small` recomputes the m <= 2 qutrit and m <= 3 qubit lists in seconds |
| `structure.py` | per qudit, how many terms of each stored decomposition and board witness are local (`k` (x) v) versus full |
| `relaxed_lift.py` | slice-and-lift with R = r+1 or r+2 terms at m+1, every configuration in which some slice is a minimal decomposition (cases A, B1, B2 in its docstring) |
| `strict_lift_big.py` | strict lift (R = r) of one known decomposition by scalar-key meet in the middle, for r = 8 |
| `sectors.py` | Pauli eigensector decomposition; rank of each sector decided inside its code by the rank-2/rank-3 searches |
| `products.py` | product decompositions from the stored lists and pairwise merge tests |
| `inspan.py` | dictionary states inside the span of each minimal decomposition |
| `symmetric.py` | set-stabilizer order of each stored decomposition under the symmetry group of the target |
| `perm_symmetric_qubit.py` | the same for the qubit orbits (phase group Z_4), scanning the phase forms of each kept flat against generators of its permutation stabilizer |
| `perm_symmetric.py` | exact search for decompositions whose term set is invariant under all copy permutations, from orbit vectors generated flat by flat without the full dictionary |
| `kvv_cat.py` | the cat-state witnesses of `bounds/qubit_H-m7-upper-9.json`, `-m8-upper-12.json`, `-m10-upper-18.json` (Kissinger, van de Wetering, and Vilmart, arXiv:2202.09202): the three-term |cat_6>, the cat_{4k+2} gluing at k = 2, and the 4-to-3 partial decomposition of |T>^5, built numerically in the T basis and converted with `to_witness.py`; also regenerates the m = 6 control |
| `two_qutrit_slice.py` | two-qutrit slicing at N m=4 and H3 m=4 (2026-09-21): every four-qutrit stabilizer state with a given slice at a base point enumerated as per-slice Pauli-class and phase codes (19683 nine-slice, 324 line, 1 point), then the exact search for rank <= 6 decompositions with a minimal two-qutrit slice, by per-slice residual-rank tables, a join from the two most constraining slices, and an exact completion of the invisible terms; `--control` runs its three controls |
| `two_qubit_slice.py` | the qubit analogue at qubit_H m=6 (2026-09-21): every m-qubit stabilizer state with a given slice at a base point of F_2^{n_1} enumerated as per-slice Pauli-class and phase codes (8385 per base slice for two sliced qubits and four remaining), per-slice residual tables over 65^4 code combinations, the join and the exact completion as in the qutrit script, plus a dedicated path for two invisible terms on one line (the shape of the known rank-6 decomposition); `--control`, `--witness bounds/qubit_H-m6-upper-6.json --rank 6`, and `--m 4 --n1 2 --rank 4` are its controls, `--ratio-tables J` prints the per-ratio exact and stabilizer counts |
| `kvv_cat_m9.py` | the nine-copy qubit_H witness `bounds/qubit_H-m9-upper-18.json` (2026-09-22): the glued cat_10 projected onto |0> and |1> on the last qubit, plus the four other cat routes to nine copies (glue cat_6 with cat_5, 3 chi(T^5), T^6 x T^3, T^7 x T^2), all 18 terms, with a pruning pass over the pool of states they use |
| `qpg_cat.py` | the cat-track witnesses `bounds/cat-m{2..8}-upper-*.json` (2026-09-24): the decompositions Qassim, Pashayan, and Gosset write out for |cat_2>, |cat_4>, and |cat_6> (arXiv:2106.07740, Eq. 5 and appendix), the <cat_2| contraction of the cat_4 and cat_6 terms for |cat_8>, and the <0| projections to m = 3, 5, 7, built numerically as written and converted with `to_witness.py`; asserts that the contraction gives six independent terms and that no projection collapses a term, so the filed ranks are the papers' |
| `product_witness.py` | exact tensor-product witnesses from bound files already on the board (x0 and l concatenated, W and Q block diagonal, coefficients multiplied in sympy); wrote `qubit_T-m8-upper-9`, `qubit_T-m10-upper-18`, `S-m7-upper-16`, `S-m8-upper-16` (2026-09-22); since 2026-09-23 a factor whose cell has no witness-bearing bound file (the Lean and literature entries at N, H3, T3 m <= 3) is taken from the stored lists under `data/` with coefficients fitted exactly by `fit_coeffs.fit`, `--date` sets the provenance date, and the notes name the single-copy product baseline for T5; wrote `T5-m3-upper-24`, `T5-m4-upper-64`, `N-m5-upper-12`, `N-m6-upper-16`, `H3-m5-upper-12`, `H3-m6-upper-16`, `T3-m6-upper-27` (2026-09-23; the T5 m=5 product of 192 terms was built and not filed because its verification exceeded the 900 s budget) |
| `t5_m2_merge.py` | the structured rank-7 search at T5 m=2 (2026-09-23): fixes `--keep` of the nine terms of a product of two rank-3 single-ququint decompositions (all 55 unordered products of the 10 decompositions) and finds every pair of two-ququint stabilizer states completing them to |T5>^2, exactly over F_q for a prime q = 1 (mod 5) by hashing the quotient images of the 3,900 states (no pair loop), with every collision re-decided over Q(w5) by DomainMatrix rank; `--control` reruns the three positive controls (seven product terms recover the dropped two, six witness terms recover the other two, seven witness terms admit only the dropped term) |
| `t3_sector_contraction.py` | the qutrit cat gluing for the T3 Z-eigensectors (2026-09-22): all rank-3 decompositions of the m=3 and m=4 sector carry states, every contraction through the m=2 sector bra into the m=5 sectors (nine terms each), the pool of states they use, and the exact tests for a five-term sector decomposition sharing four contraction terms (`--sector`, `--skip`, `--budget`, `--rank5`) |
| `sectors_p.py` | the sector route at every prime (2026-09-24): eigensector decompositions of |M>^m for groups generated by one or two commuting Pauli strings at p = 2, 3, 5, the Clifford reduction of each sector to an (m-k)-qudit state by symplectic gate rules applied to vectors, exact ranks 1, 2, 3 in the (m-k)-qudit dictionary where it fits (five qubits, three qutrits, two ququints; `SECTORS_DICT_CACHE` caches the 1.2 GB five-qubit dictionary on disk), a Fourier-formula screen of the nonvanishing-sector count for every commuting pair, canonical pair classes under the local stabilizer, copy permutations, and GL_2(F_p), and `--anneal R` for sectors without a dictionary; `--control` checks the reduction densely at p = 2, 3, 5 and reproduces the T5 m=2, T3 m=3, and N m=3 sector facts |
| `clifford_sectors.py` | eigensector decompositions of |M>^m under non-Pauli single-qudit Cliffords (2026-09-25): one generator per cyclic subgroup of the single-qudit Clifford group mod phase that is neither a Pauli nor a symmetry of |M>, deduped under the local stabilizer; the sectors (1/n) sum_j w_n^{-jk} |C^j M>^m are decided exactly in the m-qudit dictionary where it fits, else bounded below by single-qudit slices, `--anneal R` for upper bounds; `--control` reproduces chi(cat_6) = 3 through the Z sectors of |T>^6 and checks the sector sum against the target |
| `code_states_p.py` | the magic code states of QPG's Theorem 4 at p = 2, 3, 5 (2026-09-25): for an [m, k]_p code L the restriction of |M>^m to L^perp compressed to m - k qudits, one code per monomial-equivalence class (multisets of points of PG(k-1, p) up to PGL_k(p)), exact rank in the (m-k)-qudit dictionary or slice lower bounds, `--anneal` at the rank that would beat the baseline; `--unbiased` lists the Pauli eigenbases in which each orbit state is unbiased (the hypothesis of the theorem), `--twocat M` the ranks of the two-translate states (|M>^M + |P M>^M)/sqrt 2 |
| `t7_partial_merge.py` | the KvdWV partial decomposition of |T>^5 times a rank-3 block for |T>^7 (2026-09-25): the block varied over the stored rank-3 decompositions of |H>^3 independently per partial term, every nine-term set tested exactly for a pairwise merge, `--anneal` for a warm-started rank-8 anneal from a pruned nine-term set |
| `cyclic_symmetric.py` | exact search for decompositions whose term set is invariant under one m-cycle of the copies (2026-09-26): at rank below m every term is fixed exactly, so the script enumerates every sigma-fixed stabilizer state (invariant flats are cosets of cyclic codes; invariant phases are enumerated orbit by orbit, over the quadratic coefficients, or by the coordinate symmetry on the whole-space flat) and runs an exhaustive minimal-subset search over them, the leaf test quotiented to a parallel-pair lookup; p = 2, 3, and 5; `--control` reruns the six positive controls |
| `ledger_closure.py` | the product closure of `docs/ledger.json`: the best upper bound at each cell reachable by splitting m into parts the board already holds, flagged against the bound filed there and against the cells with no bound at all |

## What is validated

- Every decomposition in `data/` is re-checked on load to reproduce the target
  to 1e-9 with independent terms. The lists at `m = 3` (N, H3, S) and
  `qubit_H m = 4`, `qubit_T m = 4` are the outputs of
  `verify_challenge/slice_lift.all_decompositions` from the 2026-09-18
  enumerations (one member per unitary-symmetry orbit; the certificates
  `cert_n_m4_lift.py`, `cert_h3_m4_lift.py`, `cert_s_m5_lift.py`,
  `cert_qubit_h_m5_lift.py`, `cert_qubit_t_m5_lift.py` regenerate them). The
  `S m = 4` list is `lift_all` applied to the `m = 3` list, complete up to
  symmetry for the same reason. The smaller lists are recomputed by
  `export_data.py --small`.
- `relaxed_lift.py` positive control: from the 30 rank-3 decompositions of
  |N>^2 it recovers rank-4 decompositions of |N>^3 (18 hits, case A), and
  `to_witness.py` converts a hit to an exact witness.
- `strict_lift_big.py` controls: the rank-3 decompositions of |T3> lift to
  |T3>^2; the rank-3 decompositions of |N>^2 do not lift to |N>^3.
- `sectors.py` control: for T3 and P = Z^(x 3), Z^(x 4) the sectors have rank
  at least 3, as the exhaustive results behind `bounds/T3-m5-upper-18.json`
  say. The Clifford reduction asserts that each sector lands in one block.
- `sectors_p.py --control`: the gate-rule Clifford reduction agrees with the
  dense conjugation of random commuting generators at p = 2, 3, 5; the Z Z
  sectors of |T5>^2 are five stabilizer states; the Z^3 sectors of |T3>^3
  have rank 3; X^2 X^2 X^2 on |N>^3 has one rank-2 sector; the
  Fourier-formula sector weights match explicit projections. Run
  separately: the Y^6 sectors of |H>^6 have exact rank >= 3 in the
  five-qubit dictionary and the annealer finds rank 3 (chi(cat_6) = 3 of
  Qassim, Pashayan, and Gosset).
- `perm_symmetric.py` controls: at N m=3 and H3 m=3 with R = 4 it returns
  exactly the known S_3-invariant rank-4 decomposition; at S m=3 it returns
  none, as the set-stabilizers from `symmetric.py` (order <= 2 in S_3)
  predict. `perm_symmetric_qubit.py` controls: S_3-invariant rank-3
  decompositions of |H>^3 (4) and |T>^3 (6), and S_4-invariant rank-4
  decompositions of |H>^4 (10), matching the set-stabilizer orders.
- `two_qutrit_slice.py --control`: 400 random four-qutrit stabilizer states
  (random `(k, x0, W, Q, l)` through `common.term_vector`) all have their
  slice code pattern in the enumeration for their own base slice (211
  nine-slice, 104 line, 85 point); 40 random sums of one to three
  line/point terms with random coefficients are recovered exactly by the
  completion; the driver finds the three-term lift of phi (x) |N>^2 for a
  random full-support two-qutrit stabilizer state phi. The shape counts
  also match the dictionary: 360 x 19683 + 12 x 360 x 81 + 9 x 360 =
  7,439,040 four-qutrit states.
- `two_qubit_slice.py` controls: 300 random six-qubit, 200 four-qubit,
  100 (three sliced qubits) and 6 (four sliced qubits) random stabilizer
  states all have their slice code pattern in the enumeration for their own
  base slice, and the shape counts match the lemma's formula at every
  (n_1, n_2) (315,057,600 six-qubit states for n_1 = 2); 30 random sums of
  one or two invisible terms are recovered exactly by the completion at
  m = 6 and m = 4; from the single rank-2 decomposition of |H>^2 the m = 4
  run returns 66 exact rank-4 decompositions of |H>^4; the rank-6 witness of
  |H>^6 is recovered from its minimal four-qubit slice (all 20 such slices
  give the same base decomposition) through the shared-line path.
- `t3_sector_contraction.py` controls: the <Phi| contraction maps c_j^(3) (x)
  c_k^(4) onto c_(j+k)^(5) for all nine (j, k); every contracted term is
  confirmed a stabilizer state by `term_from_vector`; every nine-term set
  reproduces its sector to 1e-9; the three sector decompositions of the
  m=4 carry state are the sectors of `bounds/T3-m4-upper-9.json`.
  `kvv_cat_m9.py` checks every route against |T>^9 to 1e-14 before the
  exact conversion. `product_witness.py` writes exact terms, so its check
  is the verifier itself.
- `t5_m2_merge.py --control`: fixing any seven of the nine product terms
  recovers the two dropped ones (36 of 36 fixed sets); fixing any six of
  the eight terms of `bounds/T5-m2-upper-8.json` recovers the other two
  (28 of 28); fixing seven witness terms and asking for one completing
  state returns exactly the dropped term (8 of 8) and no other. Reduction
  mod the prime ideal above q = 10061 preserves every dependency over
  Q(w5), so the mod-q pass can only over-report, and each report is
  re-decided exactly; the search also checks that the fixed terms and the
  target stay independent mod q (a dependency there would be re-decided
  exactly as a lower-rank hit).
- `cyclic_symmetric.py --control`: the six positive controls return exactly
  the board's exact values and nothing below them. At cat m=6 it recovers a
  rank-3 cyclically invariant decomposition (Qassim, Pashayan, and Gosset's
  three terms are individually permutation invariant), at cat m=5 rank 3, at
  qubit_H m=3 and qubit_T m=3 rank 3, and at N m=3 and H3 m=3 rank 4, the
  four fully symmetric states `perm_symmetric.py` also returns. Run
  separately as negative controls: qubit_H m=5 and qubit_T m=5 return
  nothing at rank 5 or below, as chi = 6 there requires. Every state the
  enumeration keeps is confirmed by `to_witness.term_from_vector`, and a
  flat whose invariant phase functions exceed the cap is reported as skipped
  rather than dropped silently.

- Every hit any script reports is checked numerically against the target and
  written as amplitude vectors under `results/` (not committed) for
  `verify_challenge/to_witness.py`, which is the exact step.

Null results here are about the configurations searched, which the note lists
per cell; none of them is a lower bound.
