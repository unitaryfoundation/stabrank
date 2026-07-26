# Examples

End-to-end search scripts using the `stabrank_core` C++ engine.

Run any of them as `uv run python stabrank/examples/<script>.py`.

## `search_decomposition.py` — single-state stabilizer-rank search

One parameterized driver for searching the stabilizer-rank decomposition of a
single target state. It replaces the former family of near-identical per-state
scripts (`qubit_real_example.py`, `qutrit_strange_example.py`, …): the only thing
that differed between them was the target state and a few SA hyper-parameters, all
of which now live in the `TARGETS` registry at the top of the file.

```bash
# Use a target's built-in defaults:
uv run python stabrank/examples/search_decomposition.py --target qutrit_strange

# Override size / starting rank / seed (e.g. to reproduce a specific run):
uv run python stabrank/examples/search_decomposition.py --target qubit_real --n 6 --k-start 6 --seed 6132679
```

Available `--target` values:

| Target | State | p |
| --- | --- | --- |
| `qubit_real` | Qubit Hadamard eigenstate (real-restricted moves) | 2 |
| `qubit_ttype` | Qubit T-type magic state \|T>^n | 2 |
| `qutrit_real` | Qutrit Hadamard eigenstate \|H3> | 3 |
| `qutrit_complex` | Qutrit complex magic state \|T3> | 3 |
| `qutrit_strange` | Qutrit strange state \|S> | 3 |
| `qutrit_norrell` | Qutrit Norrell state \|N+> | 3 |

Each run anneals at `k_start`, saves any decomposition that meets the error
threshold, then prunes the least-significant basis function as far as the error
allows before re-annealing at the smaller rank. Solutions are written to
`solution_<target>_k<k>_n<n>_<timestamp>.npz`.

## Notes

- Every script seeds via `np.random` and prints the seed, so runs are reproducible
  post-hoc; `search_decomposition.py` additionally accepts `--seed` to pin a run.
- The `sys.path.append(...)` boilerplate at the top of each script is only needed
  when running in-place without `uv pip install -e .`; it is harmless otherwise.

## `ttype6_rank5_certificate.py` — exact rank-5 approximation certificate

Standalone exact-arithmetic verification (requires `sympy`, not a package
dependency) that an explicit 5-dimensional stabilizer span approximates
the six-copy BK T-type state with residual exactly
`sqrt(5(2 - sqrt(3))/24)`. Accompanies the conjecture on the project page
that this is optimal, i.e. that `chi(|T>^6) = 6`.

## `strange6_rank7_certificate.py`, `h3m4_rank6_certificate.py` — plateau certificates

Standalone exact-arithmetic verifications (require `sympy`, not a package
dependency) of the conjectured optimal low-rank approximation errors:
every 7-element subset of the m = 2 Strange product basis approximates
`|S>^6` at exactly `sqrt(8/27)`, and an explicit 6-dimensional span
approximates `|H3>^4` at exactly `sqrt(70 - 37 sqrt(3))/12`. See the
project page's conjecture table.
