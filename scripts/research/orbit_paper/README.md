# Orbit-comparison paper research scripts

Computational artifacts for the paper *Stabilizer rank bounds for
qutrit magic-state orbits* (`paper/main.tex`, `paper/main.pdf`).

## Layout

- `_paths.py` — shared path constants (`SOLUTIONS_DIR`, `DATA_DIR`)
  resolved from this script's location, so the scripts work from any
  CWD.
- `run_orbit_sa.py` — generic single-shot SA driver for any of the
  four magic-state orbits (`strange`, `h3`, `norrell`, `t3`). Produced
  the $\chi = 4$ decompositions in Appendix A.
- `squeeze_orbit_sa.py` — multi-chi squeeze loop wrapper around
  `run_orbit_sa.py`.
- `sweep_orbit_lower_bounds.py` — mana + exhaustive-fidelity LB
  sweep across all four orbits. Generates Tables 2 and 3 in §5.
- `verify_h3_chi4.py` — independent reconstruction of the
  co-author's $H_3$ $m = 3$ $\chi = 4$ decomposition from the
  canonical $(k, x_0, W, Q, \alpha)$ spec, validating Appendix A.3.
- `extract_decomposition.py` — decoder that recovers the canonical
  $(k, x_0, W, Q)$ tuple from any saved basis-state amplitude pattern.
  Used to populate Appendix A.

## Inputs

The scripts load from `paper/solutions/solution_*.npz` via the
`_paths.SOLUTIONS_DIR` constant. `sweep_orbit_lower_bounds.py` writes
to `paper/data/orbit_lower_bounds*.json`.

`run_orbit_sa.py` and `squeeze_orbit_sa.py` also save new converged
decompositions to `paper/solutions/` by default. Use `--output-dir` to
write elsewhere. Saved `.npz` files include the base seed, converged
seed, iteration count, chain count, Clifford-move ratio, final residual,
and generating script path.

## Generic utilities (in `scripts/`)

- `scripts/compute_lower_bounds.py` — generic LB CLI (used by
  multiple wiki pages, not orbit-paper-specific).
- `scripts/verify_stabilizer.py` — generic stabilizer-state
  validation utility (used by `tests/test_cat6_tensor.py` and
  `wiki/pages/qutrit_stabilizer_rank_bounds.md`).

## Open-problem search scripts (removed)

The repo previously contained six experimental SA scripts targeting
the open problems in §8 (push $\Hth^{\otimes 4} \le 6$, push
$\Strange^{\otimes 3} = 3$, etc.). All produced stalled residuals
that are documented in §8 prose; the scripts have been removed since
they didn't yield any tight bound and were not load-bearing for the
paper. The generic `run_orbit_sa.py` and `squeeze_orbit_sa.py` plus
the seeded-init pattern documented in §7 are sufficient for any
follow-up attempt.
