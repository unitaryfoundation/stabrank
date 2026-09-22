# Rank-5 exclusion at N m=4 and H3 m=4: costing prototypes

Design note: `docs/notes/qutrit_m4_rank5_exclusion.md`. These two scripts
are the Python prototypes behind its measurements, not the pipeline.

- `cover_census.py census ORBIT n [--pairs K] [--cap S] [--out FILE]`: the
  p = 3 port of `verify_challenge/slice_cover.CoverEnumerator` (full
  r-covers of |M>^n up to the unitary symmetry group, modulo 65521 with
  exact re-decision). At n = 2 it lists the full 3-, 4-, and 5-covers and
  the dependent and repeated 5-covers (about 10 to 15 minutes per orbit);
  at n = 3 it only reports the pivot-pair census, since one pivot pair at
  M near 30,000 does not finish in Python.
- `match_prototype.py`: stage A of the H^6 matcher for a two-qutrit base
  (distinct independent base states), with the controls `planted` and
  `product ORBIT`, the timing command `sample ORBIT CENSUS.json --count K
  --x0 a,b`, and `lifts13` for the 1 + 3 route's matcher cost.

Run from the repository root with
`uv run --extra challenge --extra test python research/qutrit_m4_rank5/<script> ...`.
