# Rank-5 exclusion of |H>^6 by an all-visible base slice

Design and status: `docs/notes/h6_rank5_exclusion.md`. Nothing here is a
certificate: both positive controls pass (section 5 of the note), the
per-cover cost is measured (section 4), and the full run is projected at
100 to 400 CPU-hours in the present Python.

- `restricted_dictionary.py`: exact counts of the six-qubit stabilizer
  states with property P (PR #87) and its symmetry closure P*, by support
  dimension. `--orbits` starts the exact orbit count under the symmetry
  group of |H>^6 (about 10 minutes to hash the 1.2e8 states; the generator
  pass was not run to completion).
- `driver.py`: resumable batches over the pivot pairs of the 5-cover
  enumeration of |H>^3 (`partition`, `run B`, `status`), each batch
  matching its covers at the four base points and recording counts, the
  coordinate-slice solution histogram and any hit in `results/batch_B.json`;
  the controls `control-m4` (rank 4 of |H>^4 from the full 4-covers of
  |H>^3, passes: 23 of 23 classes) and `control-witness` (the rank-6
  witness from its own all-visible bases; `--distinct-only` skips the
  bases with a repeated state); `sample --count N` (timing on N covers of
  the first pivot, 0.110 s mean and 0.019 s median per cover over four base
  points) and `degenerate` (26,242 dependent or repeated full 5-covers,
  stages B and C).
- `partition.json`: 14,280 pivot pairs in 17 batches at the calibrated
  kernel rate.

The machinery lives in `verify_challenge/slice_cover.py`
(`CoverEnumerator`, `Family`, `Block`, `solve_slice`, `SliceMatcher`);
`tests/test_slice_cover.py` has the fast controls.
