# Rank-5 exclusion of |H>^6 by an all-visible base slice

Design and status: `docs/notes/h6_rank5_exclusion.md`. Nothing here is a
certificate: both positive controls pass on every base (section 5 of the
note), the 5-cover enumeration is complete (5,939,465 full covers,
`results/kernel_census.json`), every stage's per-cover cost is measured
(section 4), and the full run is projected at about 36 CPU-hours (2.4 for
stage A with the compiled kernels, about 34 for the degenerate stages B
and C in Python). It has not been launched.

- `restricted_dictionary.py`: exact counts of the six-qubit stabilizer
  states with property P (PR #87) and its symmetry closure P*, by support
  dimension. `--orbits` starts the exact orbit count under the symmetry
  group of |H>^6 (about 10 minutes to hash the 1.2e8 states; the generator
  pass was not run to completion).
- `driver.py`: resumable batches over the pivot pairs of the 5-cover
  enumeration of |H>^3 (`census`, `partition`, `run B`, `status`), each
  batch matching its covers at the four base points and recording counts,
  the coordinate-slice solution histogram and any hit in
  `results/batch_B.json`; the controls `control-m4` (rank 4 of |H>^4 from
  the full 4-covers of |H>^3, passes: 23 of 23 classes) and
  `control-witness` (the rank-6 witness from its own all-visible bases,
  passes on all four; `--distinct-only` and `--repeated-only` select the
  bases, `--base K` one of them); `sample --count N` (timing on N covers of
  the first pivot: 1.4 ms mean per cover over four base points with the
  compiled matcher, 0.110 s with `--reference`); `fixture` (the C++ test's
  copy of the sample with the reference matcher's results); and
  `degenerate [--sample N]` (26,242 dependent or repeated full 5-covers,
  stages B and C, with a timing sample per multiplicity pattern).
- `partition.json`: 14,280 pivot pairs in 15 batches of about 600 s, by the
  census's kernel seconds and covers per pair at 1.4 ms per cover.
- `results/`: `control_m4.json`, `control_witness.json` (distinct bases),
  `control_witness_repeated.json`, `sample.json` (reference matcher),
  `sample_native.json`, `kernel_census.json`, `degenerate_sample.json`.

The machinery lives in `verify_challenge/slice_cover.py`
(`CoverEnumerator`, `Family`, `Block`, `solve_slice`, `SliceMatcher`), with
the two hot paths compiled in `cpp/src/cover5.cpp` (the 5-cover pair
kernel) and `cpp/src/slice_match.cpp` (stage A matching for distinct,
independent base states), bound in `python/bindings.cpp` and tested
against the Python reference in `tests/test_slice_cover.py`,
`cpp/tests/test_cover5.cpp` and `cpp/tests/test_slice_match.cpp`;
`STABRANK_NO_NATIVE=1` keeps everything in Python.
