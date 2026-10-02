# research/h3_m4_rank6: the rank-6 exclusion of |H3>^4

The pipeline behind `bounds/H3-m4-lower-7.json` (chi(H3^4) >= 7 at the
attested tier; with the Lean rank-8 witness `bounds/H3-m4-upper-8.json`,
7 <= chi(H3^4) <= 8), run on the pod 2026-10-01 to 2026-10-02
(`docs/notes/h3_m4_rank6_design.md`, section 11). Every rank-6 decomposition of
|H3>^4 has, along qutrits 1, 2 at the base point (0, 0), four, five or six
visible terms (Fact B of PR #86), so its base is a full 4-, 5- or
6-multiset of |H3>^2 and the invisible terms lie on the 16 affine flats of
F_3^2 missing (0, 0). Stage A6 runs the 36,368,678 full 6-covers of
distinct independent states through the compiled stage A kernel; stage B6
the 378,843 G_2 orbit representatives of the dependent 6-sets through the
fresh-term filter and the reference matcher; stage C6 the 354,012
representatives of the repeated 6-multisets through the block paths; stage
(beta') the 61,413 representatives of the full 5-multisets with the
invisible term on each of the 16 flats and stage (gamma) the 565
representatives of the full 4-multisets with each of the 136 flat
multisets through `invisible_p3.InvisibleMatcherP3`. The design, the case
split for the invisible terms, the controls and the cost table are
`docs/notes/h3_m4_rank6_design.md`. The code is the N^4 pipeline
(`research/n4_rank6/`, `docs/notes/n4_rank6_exclusion.md`) with the cell
(base point, orbit, flats) read from this directory's `common.py`.

Files

| file | role |
|---|---|
| `common.py` | the cell (n_1 = 2, base point (0, 0), orbit H3, |G_2| = 32), the 16 flats and 136 flat multisets as a `Cell`, hashing, the loaders of the census, the representative lists and the partition, the exact re-decision of a hit |
| `invisible_p3.py` | `InvisibleMatcherP3.run_one` (stage beta') and `run_two` (stage gamma): the N^4 invisible matcher with the geometry threaded through the cell; the planted generators of every kind (`Plants`) and `invisible_p3.py controls` |
| `filters6.py`, `stages.py`, `degenerate6.py` | the N^4 fresh-term filter, per-item stage runs and degenerate-list routes on this cell |
| `driver.py` | `lists --write`, `control-lists`, `sample STAGE`, `partition [--tiny]`, `control-planted`, `control-witness`, `control-orbit`, `control-m3` |
| `batch.py K` | one batch (`--resume`, `--max-seconds`, exit 2 on `DECOMPOSITION FOUND`, exit 1 on any undecided run) |
| `aggregate.py` | the certificate-side check and the batch manifest |
| `fill_draft.py`, `H3-m4-lower-7.json.draft` | the bound file, filled from the records and the manifest after the aggregate certifies |
| `probe.py` | the design note's measurements: geometry, lists and orbits, matcher rates, the 6-cover census, the A6 rate |
| `pod/pod_chain.sh` | the pod chain (setup and launch lines in its header; markers `H3M4_*`) |
| `reps_H3.json` | the orbit-representative lists (B6 with kappa, C6, k5, k4) as base-360 codes, hashed (`driver.py lists --write`, the pod's build) |
| `results/` | the 1,021 pod records `batch_K.json` with their `batch_K.log`, `census6_H3_full.json` (the 6-cover census per pivot pair), the design records, `control_*.json`, `rates.json` and `sample_*.json` (the pod's rates), `dryrun/` (the laptop dry run of the chain at the smallest scale), and `pod/` (the chain's logs, the aggregate log, the two recheck records) |
| `partition.json`, `batch_manifest.json` | the pod's partition (1,021 batches, hashed) and the aggregate's manifest of the 1,021 records with their SHA-256, what `bounds/H3-m4-lower-7.json` attests to |
| `verify_challenge/cert_h3_m4_rank6_attested.py` | the certificate (runs `aggregate.py --recheck 2`) |

The pod run is `pod/pod_chain.sh`: kernel import, the test module and the
planted controls, the lists (compared with the committed `reps_H3.json`),
the remaining controls, the rates and the partition, the first B6 kappa-1
batch timed with the stage projected at its rate, the batch loop at
`WORKERS` processes with a resume pass, the aggregate with two seeded
re-runs, optionally the `--no-native` replays, the filled draft and the
certificate under the verifier. The laptop dry run of the same steps at
the smallest scale (one census pair, one item per list class, one flat,
one flat multiset) is `driver.py partition --tiny` followed by `batch.py K
--partition results/dryrun/partition_tiny.json --out-dir results/dryrun`
and `aggregate.py --partition ... --results-dir ...`; its records are
under `results/dryrun/`.
