# Rank-5 exclusion of |H>^6 by an all-visible base slice

Design and status: `docs/notes/h6_rank5_exclusion.md`. Nothing here is a
certificate; see section 5 of the note for the two cases the argument does
not yet cover.

- `restricted_dictionary.py`: exact counts of the six-qubit stabilizer
  states with property P (PR #87) and its symmetry closure P*, by support
  dimension. `--orbits` starts the exact orbit count under the symmetry
  group of |H>^6 (about 10 minutes to hash the 1.2e8 states; the generator
  pass was not run to completion).
- `driver.py`: resumable batches over the pivot pairs of the 5-cover
  enumeration of |H>^3 (`partition`, `run B`, `status`), each batch
  matching its covers at the four base points and recording counts, the
  basis-solution histogram and any hit in `results/batch_B.json`.
- `partition.json`: 14,280 pivot pairs in 17 batches at the calibrated
  kernel rate.

The machinery lives in `verify_challenge/slice_cover.py`
(`CoverEnumerator`, `SliceMatcher`).
