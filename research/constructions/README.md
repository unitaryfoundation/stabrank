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
| `perm_symmetric.py` | exact search for decompositions whose term set is invariant under all copy permutations, from orbit vectors generated flat by flat without the full dictionary |

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
  predict.
- Every hit any script reports is checked numerically against the target and
  written as amplitude vectors under `results/` (not committed) for
  `verify_challenge/to_witness.py`, which is the exact step.

Null results here are about the configurations searched, which the note lists
per cell; none of them is a lower bound.
