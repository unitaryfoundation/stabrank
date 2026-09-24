# research/t5_rank4: the rank-4 exclusion of |T>^5

The pipeline behind `bounds/qubit_T-m5-lower-5.json` (draft here until the
manifest exists): every rank-4 decomposition of |T>^5 has, along qubits 1, 2,
an all-visible base at 00 or 01, the base is a full 4-cover of |T>^3, and
none of the 4,709 such covers (up to the unitary symmetry of |T>^3) extends
to a decomposition of |T>^5. The argument, the facts it rests on, the
controls and the soundness checklist are in
`docs/notes/t5_rank4_exclusion.md`.

Files

| file | role |
|---|---|
| `common.py` | the cell (n_1 = 2, base points 00 and 01, orbit qubit_T), hashing, loaders, the exact re-decision of a hit |
| `tables.py` | the residual tables behind Fact 1 and the (1, 1) exclusion (`results/tables.json`) |
| `driver.py` | `tables`, `census`, `control-census`, `control-witness`, `control-planted`, `control-m4-pair`, `sample`, `partition` |
| `covers4.json` | the census: every full 4-cover of \|T>^3 with its kind (A distinct independent, B distinct dependent, C a repeated state), hashed |
| `partition.json` | the batches over the census, hashed |
| `batch.py K` | one batch: every cover of the batch at both base points through `verify_challenge/slice_cover.SliceMatcher` |
| `aggregate.py` | the certificate-side check: hashes, tiling, re-enumeration of the census, re-decision of every hit, seeded re-runs; prints `CERTIFIED chi(qubit_T^5) >= 5` |
| `results/` | `tables.json`, `control_*.json`, `rates.json`, `batch_K.json` and logs |

Commands, in the order they were run (all at nice 19, one process at a
time, through a wrapper with a 600 s cap):

```
uv run --extra challenge python research/t5_rank4/driver.py tables
uv run --extra challenge python research/t5_rank4/driver.py census --write
uv run --extra challenge python research/t5_rank4/driver.py control-census
uv run --extra challenge python research/t5_rank4/driver.py control-witness            # default cap: 4 kappa = 2 bases abort
uv run --extra challenge python research/t5_rank4/driver.py control-witness --reference
uv run --extra challenge python research/t5_rank4/driver.py control-witness --max-cand 20000000
uv run --extra challenge python research/t5_rank4/driver.py control-planted --plant 10
uv run --extra challenge python research/t5_rank4/driver.py control-m4-pair
uv run --extra challenge python research/t5_rank4/driver.py sample --count 100
uv run --extra challenge python research/t5_rank4/driver.py partition --target-s 120 --max-covers 1000
for K in $(seq 0 N); do uv run --extra challenge python research/t5_rank4/batch.py $K; done
uv run --extra challenge python research/t5_rank4/aggregate.py --recheck 2 --recheck-seed 20260924
uv run --extra challenge python verify_challenge/cert_qubit_t_m5_rank4_attested.py
```

`STABRANK_NO_NATIVE=1` (or `batch.py K --no-native`) keeps the kind A
matcher in the Python reference of `verify_challenge/slice_cover.py`; the
deterministic hash of a batch record does not depend on the path.
