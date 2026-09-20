# Merge recipes (2026-09)

Scripts behind `docs/notes/merge_recipes_2026_09.md`: the three product-and-merge
recipes of the September literature sweep (T3 six-copy sectors at rank 8,
qubit T at m=8 rank 8, N and H3 at m=6 rank 15) and the rank-5 pivot-pair
search for the three-copy qutrit states. Run from the repository root with
`uv run --extra challenge python research/merges/<script>`. Annealing goes
through `autoresearch/run.py --warm-from <warm file>` so that every run lands
in `autoresearch/runs.jsonl`.

| script | what it does |
|---|---|
| `mergelib.py` | flat enumeration over F_p^n, `stabilizer_states_in_span` (every stabilizer state inside a subspace, exact, no dictionary), coefficient classes, modulus-pattern rank-2 obstruction, warm-start file writer |
| `controls.py` | flat counts; the one-state completion recovers the third line of `|T3>^2` from `span(psi, l0, l1)`; in-span counts match `research/constructions/inspan.py` on N^3 and H^4 |
| `t3_sector_product.py` | nine-term product decomposition of each Z-eigensector of `|T3>^6` in carry coordinates, weight classes, warm files, stabilizer states in the span, one-state completion for every dropped pair |
| `class_sums.py` | product of the stored minimal decompositions for one orbit (qubit_T 4+4, N 3+3, H3 3+3), coefficient classes, exact stabilizer and rank-2 tests on class sums, stabilizer states in class spans and in the whole span, warm files, optional completion |
| `complete1.py` | exact one-state completion of a plateau (`run.py --save-plateaus` output) or warm file: dictionary-free `kopt.py --k 1` |
| `rank5_pairs.py` | minimal rank-5 decompositions of N^3 or H3^3 containing a fixed pair, through the compiled rank-4 pivot kernel on the dictionary projected off the second state, partners taken one per orbit of the pivot's stabilizer |
| `warm/` | the warm-start files (bounds-style `witness.terms`; not submissions) |

## The subspace test

A stabilizer state supported on the affine flat F lies in a subspace V exactly
when some vector of V vanishes off F and has unit modulus and a quadratic
phase on F. With V orthonormal, the vectors vanishing off F are the
eigenvalue-1 eigenspace of V_F^H V_F (rows of V on F), one small Hermitian
matrix per flat, batched over all flats of one pivot shape; on a flat with a
d-dimensional null space the state is fixed by its phases at d independent
points, so p^(d-1) (4^(d-1) for qubits) patterns are tried and each survivor
is confirmed by `to_witness.term_from_vector`. Five qutrits have 53,968 flats,
six qutrits 1,996,024, eight qubits 7,866,259; a scan of an 8-dimensional
subspace at five qutrits takes a few seconds.

Consequences used in the note: an R-term decomposition that contains R-2 of
the terms of a known (R+1)-term decomposition exists iff `span(psi, kept)`
contains a stabilizer state outside `span(kept)`; that covers every 2-into-1
and 3-into-1 merge and every one-term exchange at once.

## What is validated

- `controls.py` passes: 184, 307 and 22 flats for F_3^3, F_2^4, F_3^2; the
  T3^2 completion control; N^3 span holds exactly its four terms; 17 of the
  30 rank-4 decompositions of H^4 have four more stabilizer states in their
  span (the same count `inspan.py` prints when run; the constructions note
  says 16).
- Every product decomposition is checked to reproduce its target to 1e-9
  before anything is tested on it, and every warm file is rebuilt from
  witness terms by `run.py` itself.
- `rank5_pairs.py`: every reported five-set is re-solved in the full space
  (residual < 1e-9, term matrix of rank 5) and checked to have no spanning
  four-subset; the two H3 hits were re-verified outside the script.

Null results here are about the configurations searched, which the note lists
per cell; none of them is a lower bound.
