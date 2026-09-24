# research/t5q_m2_rank5: the rank-5 census of |T5>^2

Built to exclude rank 5; the census instead found the rank-5
decomposition. The pod run of 2026-09-24 (74 batches, 155,422 units, 13.4
CPU-hours, 0 undecided) returned exactly one 5-set, the states [525, 563,
591, 619, 637] of batch 31: the five Z(x)Z eigensectors of |T5>^2, each a
stabilizer state because the cubic x^3 + y^3 is quadratic on every line
x + y = c. It is `bounds/T5-m2-upper-5.json` (Lean tier), and with
`bounds/T5-m2-lower-5.json` the cell is settled at chi(|T5>^2) = 5;
`decomposition.py` works the set out and section 8 of
`docs/notes/t5q_m2_rank5_exclusion.md` records the outcome. The exclusion
drafts and `verify_challenge/cert_t5_m2_rank5_attested.py` are deleted.
The rest of this file describes the pipeline as it was built and run.

With ranks 1 to 4 excluded, a rank-5 decomposition of psi_2 = |T5>^2 over the 3,900
two-ququint stabilizer states is a 5-set of distinct independent states
whose span contains psi_2, with every coefficient nonzero; one member of
every orbit under the unitary symmetry group (order 50) contains one of
the 98 pivots and one of the pivot's partners, with the other three
members above the partner. The compiled 5-cover kernel
(`cpp/src/cover5.cpp`, `stabrank_core.cover5_pair`) lists a superset of
those sets for each of the 155,422 (pivot, partner) units over Q(zeta_5)
modulo 65521 and decides each modulo 2013265921; every listed set is
re-decided exactly and numerically in the batch. The exclusion is the
emptiness of the census. No slicing, base point, invisible term,
coefficient family or degenerate list enters. The argument, the controls
and the soundness checklist are in `docs/notes/t5q_m2_rank5_exclusion.md`;
the design and its measurements in `docs/notes/next_exclusion_feasibility_2.md`
(section 6).

Files

| file | role |
|---|---|
| `common.py` | the cell (orbit T5, m = 2, rank 5), `Field5` and the phase codes over Q(zeta_5), `T5Enumerator` (dictionary, modular images, target, unitary symmetry group, pivot plan, the compiled kernel and its Python reference `pair_sets_reference`, the exact and numerical decision of a set, the k = 1..4 censuses `census_low`), hashing, the partition loader |
| `driver.py` | `plan`, `sample`, `partition`, and the controls `control-pattern`, `control-planted [--reference]`, `control-rank4`, `control-m1`, `control-hash`, `control-reference` |
| `partition.json` | the 155,422 units (pivot, partner, members above the partner) in enumerator order and their tiling into 74 contiguous batches of about 600 pod seconds, hashed, with the dictionary and plan hashes |
| `batch.py K` | one batch (`--resume`, `--max-seconds`, `--no-native`; exit 2 on `DECOMPOSITION FOUND`, exit 1 on any undecided unit) |
| `aggregate.py` | the certificate-side check and the batch manifest (`--dry-run --partial` during the run, `--recheck 2 --recheck-seed 20260925` at the end; `--no-low-census` skips the 100 s k = 1..4 censuses) |
| `results/` | `plan.json`, `rates.json`, `sample.json`, `control_*.json`, `batch_K.json` and `batch_K.log` |
| `decomposition.py` | the hit of batch 31 rebuilt from the dictionary: the five lines x + y = c with phase w^(3c x^2 - 3c^2 x), the Z[w] identity, the coefficients w^(c^3)/sqrt 5 against `fit_coeffs.fit`, the symbolic verification, the orbit of the 5-set under the symmetry group (size one), and the common stabilizer <Z(x)Z> |

Commands, in the order they were run on the laptop (all at nice 19, one
process at a time, through `research/t5_rank5/run.py` with a 600 s cap:
`uv run --extra challenge python research/t5_rank5/run.py --max-seconds 600 -- research/t5q_m2_rank5/driver.py CMD`):

```
driver.py plan
driver.py control-pattern
driver.py control-m1
driver.py control-planted --count 24
driver.py control-rank4
driver.py sample --count 150 --budget 300
driver.py control-reference --count 12
driver.py partition --target-s 600 --pod-factor 1.3
batch.py 73 --max-seconds 560                 # the smallest batch, as the pipeline test
aggregate.py --dry-run --partial
driver.py control-planted --count 8 --reference
```

Measured on the laptop (18-core Apple silicon under load from other
sessions, nice 19): 7.42e-8 s per squared member count over 163 sampled
units, 7.16e-8 over batch 73 (13,747 units, 245 s). The census is 4.58e11
squared members, 9.1 to 9.5 laptop CPU-hours, 12.3 pod CPU-hours at the
compiled factor 1.3, about 50 minutes of wall time on 15 processes.

## The pod run

The pod checkout is `/root/stabrank-h6` (on branch `t5-rank5-pipeline`
after the T^5 run); the new branch is fetched and checked out, never `git
checkout -f` (it would overwrite result files); runs go under `setsid
nohup ... < /dev/null & disown`; the batch loop is resumable (a finished
record is skipped, one left incomplete by the deadline is redone with
`--resume`). Check with `ps aux | grep loop.py` whether the anneal loops
are running before launching, and pause them for the run as before.

```
ssh -i ~/.ssh/id_ed25519 -p 40096 root@157.157.221.30
cd /root/stabrank-h6
git fetch origin t5q-m2-rank5 && git checkout t5q-m2-rank5 && git pull --ff-only
/root/.local/bin/uv sync --extra challenge
/root/.local/bin/uv run --extra challenge python -c "from stabrank.stabrank_core import cover5_pair; print(cover5_pair)"
mkdir -p research/t5q_m2_rank5/results /root/logs
```

Controls at the pod's commit (a few minutes in all; every one must print
PASSED), then the rate check on batch 1 in the foreground of a `setsid
nohup` shell (about ten minutes; batch 73 is the laptop record), then the
aggregate's `--partial` projection tells whether the pod factor 1.3 holds
before the whole partition is launched:

```
for C in control-pattern control-hash control-m1 control-planted control-rank4 control-reference; do
  nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/driver.py $C > /root/logs/t5q_$C.log 2>&1 || echo "$C FAILED"
done
setsid nohup sh -c 'nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/batch.py 1 --max-seconds 3600 \
  > research/t5q_m2_rank5/results/batch_1.log 2>&1' < /dev/null > /root/logs/t5q_probe.log 2>&1 & disown
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/aggregate.py --dry-run --partial --no-low-census
```

The full run, 15 processes over the 74 batches (the laptop record of
batch 73 is committed under `results/` and is skipped by the loop):

```
setsid nohup sh -c 'seq 0 73 | xargs -P 15 -n 1 sh -c \
  "nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/batch.py \"\$0\" --resume --max-seconds 3600 \
   > research/t5q_m2_rank5/results/batch_\"\$0\".log 2>&1"' < /dev/null > /root/logs/t5q_run.log 2>&1 & disown
```

Rerunning the same command resumes. Progress, the final check (what the
certificate runs), and the `--no-native` replay of batch 73 (checklist
item 8; the Python reference of the kernel, about 13 times the compiled
time, about an hour):

```
grep -l "DECOMPOSITION FOUND\|undecided unit" research/t5q_m2_rank5/results/batch_*.log
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/aggregate.py --dry-run --partial
nice -n 19 /root/.local/bin/uv run --extra challenge python research/t5q_m2_rank5/aggregate.py --recheck 2 --recheck-seed 20260925
setsid nohup sh -c 'STABRANK_NO_NATIVE=1 nice -n 19 /root/.local/bin/uv run --extra challenge python \
  research/t5q_m2_rank5/batch.py 73 --out-dir research/t5q_m2_rank5/results/nonative --force' \
  < /dev/null > /root/logs/t5q_nonative.log 2>&1 & disown
```

Copy back `research/t5q_m2_rank5/results/batch_*.json`, the logs,
`batch_manifest.json`, the control records and the no-native record
(`rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p 40096" root@157.157.221.30:/root/stabrank-h6/research/t5q_m2_rank5/ research/t5q_m2_rank5/`),
and commit them. That is what happened on 2026-09-24 (12:32 to 13:36 UTC,
15 processes, 13.4 CPU-hours); `aggregate.py --recheck 2` re-derived the
plan and the k = 1..4 censuses, found the 74 records complete with 0
undecided units, and stopped with exit 2 at the hit of batch 31 before its
two seeded re-runs, so the no-native replay and the re-runs were not made.
The hit is `bounds/T5-m2-upper-5.json`; see `decomposition.py` and section
8 of the note.
