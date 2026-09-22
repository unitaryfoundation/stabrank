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
| `product_witness.py` | exact tensor-product witnesses from bound files already on the board (x0 and l concatenated, W and Q block diagonal, coefficients multiplied in sympy); wrote `qubit_T-m8-upper-9`, `qubit_T-m10-upper-18`, `S-m7-upper-16`, `S-m8-upper-16` (2026-09-22) |
| `t3_sector_contraction.py` | the qutrit cat gluing for the T3 Z-eigensectors (2026-09-22): all rank-3 decompositions of the m=3 and m=4 sector carry states, every contraction through the m=2 sector bra into the m=5 sectors (nine terms each), the pool of states they use, and the exact tests for a five-term sector decomposition sharing four contraction terms (`--sector`, `--skip`, `--budget`, `--rank5`) |

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
- Every hit any script reports is checked numerically against the target and
  written as amplitude vectors under `results/` (not committed) for
  `verify_challenge/to_witness.py`, which is the exact step.

Null results here are about the configurations searched, which the note lists
per cell; none of them is a lower bound.
