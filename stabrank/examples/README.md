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

## Sweep scripts

These iterate searches over families of codes rather than a single state:

| Script | What it sweeps |
| --- | --- |
| `sweep_code_states.py` | All systematic `[m, k]` qubit codes |
| `sweep_qutrit_code_states.py` | All systematic `[m, k]` qutrit codes |
| `sweep_qutrit_code_states_guided.py` | Qutrit code sweep with a guided start |
| `sweep_qubit_lower_bounds.py` | Cauchy-Schwarz lower bounds over qubit codes |
| `sweep_qutrit_lower_bounds.py` | Cauchy-Schwarz lower bounds over qutrit codes |

`construct_t3_6.py` is a standalone construction script for the qutrit `T3,6` state.

## Notes

- Every script seeds via `np.random` and prints the seed, so runs are reproducible
  post-hoc; `search_decomposition.py` additionally accepts `--seed` to pin a run.
- The `sys.path.append(...)` boilerplate at the top of each script is only needed
  when running in-place without `uv pip install -e .`; it is harmless otherwise.
