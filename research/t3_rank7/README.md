# Rank-7 exclusion for |T3>^3: prototype

Design note: `docs/notes/t3_rank7_exclusion.md`. Nothing in this directory
is a certificate. The full scan was not run.

## Files

- `scan3.py`: the three-pivot kernel (`kernel3`, numba), the exact
  class-set decision (`decide`), and the validation entry points listed
  below. It imports `verify_challenge/cert_t3m3_rank7.py` for the
  dictionary, the descent, the reduction mod 65521, the projection, the
  hashing primitives and the symmetry group.
- `count_steps.py`: exact inner-step counts of the m=3 scan under the
  symmetry of V_3, for the pivot order of the merged certificate and for
  the orbit-block order of the note, with and without the Stab(i, j)
  reduction of the third pivot. Writes `results/step_counts.json`.
- `results/step_counts.json`: output of `count_steps.py`.
- `results/m3_pair_timings.json`: per-step timings of `kernel3` on real
  m=3 data for a few pivot pairs, measured on a machine under load (see
  the note for the load figures).

Run from the repository root with `uv run --extra challenge python
research/t3_rank7/scan3.py {small,sym,m3,m3big}` and
`uv run --extra challenge python research/t3_rank7/count_steps.py`.

## What is validated

- `brute_force_check`: on random sub-dictionaries of 22 to 26 two-qutrit
  states, every 7-subset whose projected images span at most four
  dimensions mod 65521 (found by enumerating all 7-subsets and computing
  ranks) lies inside a class set of the three-pivot scan with trivial
  symmetry. Run with planted targets (a random 3-dimensional subspace of
  the span of seven chosen states) and with the true V_2.
- `planted_check`: on the full m=2 dictionary (360 states) with a planted
  target, the scan for the plant's two least members produces a class set
  containing the plant, and the mod-ell rank pair accepts it.
- `m2_kernel2_consistency`: at m=2 with V_2, every class set of the
  two-pivot kernel of the merged certificate (six-state configurations) is
  contained in a class set of the three-pivot kernel for the same pivot.
- `m2_symmetry_check`: at m=2 with V_2, on a G-invariant sub-dictionary
  (57 states: one orbit of 54 under the 2-copy symmetry group of V_2, of
  order 324, plus the three states inside V_2), the set of projected image
  spaces of the class sets found with one first pivot per orbit and a
  stabilizer-minimal second pivot, closed under the group, equals the set
  found with trivial symmetry (77,091 image spaces either way). This
  checks the pivot-order argument, not the orbit-block order of the note,
  which is only counted, not implemented. Larger sub-dictionaries flood
  (the run on 80 states was stopped after six minutes in the Python-side
  class assembly).
- `time_m3_pairs`: `kernel3` runs on the real m=3 data (30240 states, V_3)
  for single pivot pairs, every candidate class is decided exactly mod
  2^31 - 1, and no class set contains V_3 for the pairs tried. This is a
  timing measurement on a few pairs out of 326,368, not evidence about
  rank 7.

## What is not validated

- The orbit-block pivot order (the factor of about three in the note) is
  counted exactly by `count_steps.py` but not implemented in `kernel3`,
  which uses the pivot order of the merged certificate.
- The Stab(i, j) reduction of the third pivot is counted, not implemented.
- The re-splitting of spurious class sets (rank at least 8 mod 2^31 - 1,
  which a projection collision could produce) is described in the note and
  counted by `decide`, not implemented; none occurred in the pairs tried.
- Case B of the note (five coplanar images) has no script here; the merged
  certificate's kernel with `need = 3` plus the rank filter of the note
  would be the starting point.
- The batch runner, the manifest and the aggregator of the note's
  recommendation do not exist yet.
- The per-step timing was measured on a machine with a load average
  between 30 and 120 on 18 cores at `nice -n 19`; treat it as an upper
  bound on the cost per step of this numba kernel.
