# Examples

Five end-to-end search scripts using the `stabrank_core` C++ engine.

| Script | Problem | Expected runtime (Apple Silicon, current `main`) | Status |
| --- | --- | ---: | --- |
| `qutrit_magic_code_state_example.py` | Qutrit cat5 (m=5, [5,1] rep code) | ~30 min | works; reproduces χ=6 |
| `magic_code_state_example.py` | Qubit cat6 (m=6, [6,1] rep code) | ~2 min | works; should reproduce χ≤3 if the iteration budget is enough |
| `qubit_real_example.py` | Qubit Hadamard eigenstate, restricted to real-amplitude moves (`use_real_qubit_moves=True`) | a few min | works; demonstrates the `use_real_qubit_moves` flag |
| `sweep_qutrit_code_states.py` | Sweep over all systematic [m, k] qutrit codes | hours (default `m=6`) | useful as a template; default m=6 will spend most time on codes that don't converge — see notes below |
| `sweep_code_states.py` | Same as above for qubits | minutes for `m=6` | works; `target_effective_rank` arg is currently unused |

Run any of them as `uv run python stabrank/examples/<script>.py`.

## What each one demonstrates

### `qutrit_magic_code_state_example.py` — cat5 baseline + futile prune

Starts SA at `k_start=6`, the *known optimum*, finds a χ=6 representation, then attempts to prune to `k=5` (which fails for cat5 with current SA at any reasonable budget — confirmed today with both `cooling=0.995, 1.65M` iterations and the smaller `cooling=0.99, 825k` iterations sweep budget). The prune-down attempt ends up dominating wall time even though it never produces a result.

Useful as a workflow demonstration; **less useful as a search**, because it spends most of its budget trying to push past a barrier the wiki already documents. Two improvements worth considering:
- Add a CLI flag that skips the `k=5` attempt and just verifies / saves the χ=6 result.
- Or move the futile-push logic to a separate "stress test" example so users running `qutrit_magic_code_state_example.py` get a 30-second sanity-check rather than a 30-minute search.

### `magic_code_state_example.py` — qubit cat6

The qubit analogue of the script above, targeting the 6-bit repetition code's magic cat state. The paper proves χ ≤ 3 for cat6, and SA should find it. Uses `num_iterations_at_temp=200` (notably smaller than the qutrit version's 1000) because qubit cat6 lives on a 32-dim space and convergence is fast.

### `qubit_real_example.py` — restricted Pauli moves

Showcases the `use_real_qubit_moves=True` knob: SA proposals are restricted to Pauli strings with an even number of `Y` operators, which preserves real-valued amplitudes. Target is the qubit Hadamard eigenstate.

Notes:
- `seed = 6132679` is hardcoded on line 30, immediately overriding the random seed selected on line 29. This is not a bug — it pins the example to a specific reproducible run — but it is non-obvious. The hardcoded seed should be a CLI flag or a clearly-flagged constant.
- Initial basis is `np.ones(2**n_orig) / norm` (uniform `|+⟩^n`), not random stabilizer states. The `generate_random_func()` helper above the loop is *defined but unused* — leftover from when initialization was switched. Safe to delete or wire back in.
- Imports `calculate_sa_cost` but never calls it. Safe to remove.

### `sweep_qutrit_code_states.py` — systematic ternary code sweep

Iterates over every systematic-form `[m, k]` ternary code generator matrix `G = [I_k | A]` and searches for the smallest χ for each. This is the methodology described in `wiki/pages/ternary_code_sweep.md`.

Notes:
- Default in `__main__` is `run_qutrit_sweep(m=6)`. Today's empirical signal (see `wiki/pages/cat6_sa_baseline.md`) is that brute-force SA at `k_start=8` does not converge for the cat6 rep code. By extension the m=6 sweep will spend most of its time on codes that fail. **For practical use, prefer m=4 or m=5 first** — pass `m` via a CLI flag or edit `__main__`.
- `fast_temp` / `fast_cooling` / `fast_iters` / `fast_chains` are defined but unused. Looks like a planned two-stage (fast pre-screen + deep search) structure that was never wired up. Safe to remove or implement.
- The `prune_least_significant_basis_function` recovery logic decrements `current_chi` even on a failed prune and lets the outer loop re-run SA at the smaller chi from the pruned start — a useful pattern, mirrored in `run_5_1_sweep.py`.

### `sweep_code_states.py` — qubit code sweep

Qubit analogue. Default `m=6` with `target_effective_rank=6`.

Notes:
- The `target_effective_rank` argument is **defined but never checked** in the loop. The script just records every code's min-chi to CSV regardless. Either wire it up as an early-out filter or drop the parameter.
- Same `fast_*` dead code as the qutrit sweep.

## Cross-cutting observations

- Every example has `sys.path.append(os.path.dirname(...))` boilerplate at the top. This is only necessary if running the file from inside the repo without `uv pip install -e .`. With the package installed it is a no-op — but it's also harmless, so removing it is purely cosmetic.
- Every example inlines the same ~80-line "SA + prune-down" squeeze loop. If a third or fourth search workflow gets added, this is worth extracting into `stabrank/examples/_squeeze.py` or similar.
- All examples seed via `np.random.randint(...)` and print the seed, so individual runs are reproducible *post-hoc*. None take a `--seed` CLI flag to reproduce a specific run intentionally.
- None document expected wall time. The table at the top of this file is the first place that information lives.

## Recommended cleanups (none of them blocking)

1. **Remove dead code**: unused imports (`calculate_sa_cost` in `qubit_real_example.py`), unused functions (`generate_random_func` in same file), unused parameter blocks (`fast_*` in both sweep scripts), unused arguments (`target_effective_rank` in `sweep_code_states.py`).
2. **Add a `--m` CLI flag to the sweep scripts** so picking a smaller `m` doesn't require editing the file.
3. **Either skip the futile prune step in `qutrit_magic_code_state_example.py`** or split it into a separate stress-test script. As-is, the example's wall time is dominated by a step that never produces a result.
4. **Consider extracting the squeeze loop** if a sixth example gets added.

These are surface-level — the underlying logic in each script works. The C++ engine they call is the same one exercised by `tests/` and `benchmarks/`, so functional regressions show up in CI before they reach an example.
