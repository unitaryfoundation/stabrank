# H^6 matcher: caches keyed by object identity

Date: 2026-09-23. Checked at main 702e270 (includes PR #111, the stage C
repair).

## Question

The qutrit matcher (research/qutrit_m4_rank5/matcher.py before a59102c)
cached the compatibility index of the second coordinate slice under
`id(sols)`. The solution list is freed at the end of a run, the next run's
list often lands at the same address, and the stale index was served for
it, which dropped 8 of the 270 product-control hits with no record. The
H^6 matcher (verify_challenge/slice_cover.py, `SliceMatcher`) underlies
the attested chi(H^6) >= 6 in research/h6_rank5, and batch.py builds one
`SliceMatcher` per batch and runs every cover of the batch at the four
base points through it (`run_batch`, `match_cover`), so any state that
survives a `run` call is shared across covers. This note lists that state
and decides, for each item, whether a stale entry can drop a genuine hit.

## State that survives a `run` call

slice_cover.py contains no call to `id()`, no `functools` cache, and no
module-level memo. The dictionaries and sets it does keep are the
following.

1. `SliceMatcher.cache` (`options`, line 1333). A dict from the
   dictionary index of a base state (a Python int, from
   `sorted(set(cover))`) to its `TermOptions`. The key is a value, not an
   address, and the value is a deterministic function of the key and the
   enumerator's dictionary. `TermOptions` is not mutated after
   construction: `arrays()` hands out `m1`, `m2`, and `vecs` by reference,
   and every consumer either indexes them (fancy indexing copies) or
   writes them into a fresh array (`slice_system`, `_block_coordinates`,
   the term assembly in `_complete`, `Block.__init__`); `_mitm` and
   `_dense` build new arrays from `popts`. Verdict: cannot go stale.

2. `TermOptions._composite_rows` (`copy_composite_rows`, line 1091), a
   memo on the options object keyed by `(n1, tuple(composite),
   tuple(int(c) for c in ccodes))`, all ints. The value depends only on
   the key and on the options object it hangs off, and `reconstruct_block`
   reads it without writing. Verdict: cannot go stale.

3. `SliceMatcher.rng`, one `numpy` generator per matcher, consumed across
   runs. It draws the random functionals of `_mitm` and `_dense` and the
   generic family member of `Family.coefficients`. Neither draw can lose a
   genuine solution. A genuine slice solution satisfies the slice
   congruence mod P1 exactly, so it satisfies any linear functional of it,
   and a genuine dependent tuple has vanishing determinant for any choice
   of functionals; both filters are supersets, and every candidate is then
   decided exactly by `fam.restrict` over F_P1, C, and F_P2 (`solve_slice`,
   line 969). `coefficients(rng)` is called in the join only when
   `fam.kappa == 0`, where it returns `d0` (`_join_blocks`, the guard at
   line 1421 and the draw at line 1428, and `_block_coordinates` at line
   1449, reached from the join only past that guard); in `_complete` a
   family with parameters and a
   block raises `UnpinnedFamily` before any reconstruction, and for a hit
   without blocks the coefficients are recomputed by `confirm`. The
   generator state changes which spurious candidates appear, never which
   exact solutions survive. Verdict: no soundness effect.

4. The compiled stage A kernel (cpp/src/slice_match.cpp, held in
   `SliceMatcher.native`). `Impl::cache` is a vector of `StateOptions`
   indexed by dictionary index and returned as a const reference;
   `functional` is drawn once from the seed and fixed for the kernel's
   life; the meet-in-the-middle table `head`, `stamp`, `entries` is reused
   between `solve_point` calls and invalidated by incrementing
   `cur_stamp`, with a bucket treated as empty unless `stamp[key] ==
   cur_stamp` and `entries` cleared per call (lines 332 to 361). Position
   keys and a correct stamp reset: no stale bucket can be probed. (The
   `int32_t` stamp wraps only after 2^31 calls, far beyond a batch, and
   the code refills on zero.) Verdict: cannot go stale.

Everything else is rebuilt per run: `stats`, the `Family` objects (every
`restrict` allocates new arrays), the `Block` objects with their translate
matrices (copies of `TermOptions` rows), and the per-state tuples, which
grow by concatenation (`cl + [combo]`, `bl + [Ssel]`) rather than in
place. `_refine_split` returns its input unchanged or a new list.
`_dedupe`, `pauli_reps`, and `reconstruct_block` keep `seen` sets local to
the call.

## Why the qutrit pattern has no analogue here

The qutrit `_compatible` built an index over the second slice's solution
list and memoised it on the matcher, so a run needed the index from an
earlier point in the same run and could receive one built in an earlier
run. The H^6 `run` (line 1345) has no cross-slice index at all: for each
state it calls `solve_slice` against the state's own restricted family,
iterates the returned list directly, and restricts the family per
solution. No structure built from a solution list outlives the loop that
consumes it. There is no gap to fix.

## Controls

All under `nice -n 19`, one process, killed at 600 s.

- tests/test_slice_cover.py: the planted-pair tests, the block
  reconstruction test, the H^4 stored-decomposition test, and the
  undecided-path tests pass (9 passed). The three native-versus-reference
  comparisons skip: this machine's venv extension predates
  `SliceMatchKernel` and `cover5_pair`, and under pytest the source tree
  shadows the installed package anyway.
- `driver.py control-m4`: 3466 bases (3460 independent full 4-covers, 6
  dependent or repeated), 6932 (cover, x0) pairs matched in 8 s, 95 hits,
  23 classes of rank-4 decompositions recovered, 23 stored, 23 expected
  recoverable, 0 missing, 0 unexpected, 0 refused. PASS.
- `driver.py sample --count 40`: through the Python matcher (the kernel is
  unavailable in this venv build), 160 matched, 0 refused, 0 hits,
  coordinate-slice histogram {'1,1,1': 6, '0': 146, '9,81,729': 4,
  '2,4,8': 4}, identical to the stored results/sample.json and
  results/sample_native.json on the same 40 covers.
- Fresh versus shared matcher (scratch script, not committed): each
  (cover, x0) run on a matcher that had processed all earlier covers and
  on a matcher built for that run alone, comparing the hit code sets and
  the deterministic stats (kappa, coordinate solutions, joined, types,
  composite solutions, prunes, reconstructions). 40 stage A covers (160
  runs, reference matcher), 7 stage B covers (28 runs, dependent distinct
  states, the `_dense` path), 16 stage C covers over the patterns
  (2, 1, 1, 1), (2, 2, 1), and (3, 1, 1) (64 runs): 0 differences, 0
  raised.

## Stored records

The 197 records (15 A, 158 B, 17 C, and the 7 stage C v2 batches) are
unaffected: the analysis above finds no identity-keyed or otherwise
address-dependent cache in either matcher path, and the controls show no
run-order dependence. Every B and C record has `native_runs` 0, so those
stages ran the Python matcher exactly as analysed; stage A ran the kernel
(23,757,860 native runs). No re-run is needed. For reference, a re-run of
the Python-matcher stages would cost about 70 CPU-hours for B (12,390
covers) and 2.3 CPU-hours for C (29,858 covers), from the `cpu_s` fields
of the records.
