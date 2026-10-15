# Rank-7 exclusion for |T3>^3: independent scan

A second exclusion of rank 7, run independently of the batch pipeline in the
parent directory, on other hardware, with a different pivot order and kernel
and a case split the batch scan did not use. It reaches the same conclusion,
chi(T3^3) = 8, and is recorded here as evidence next to the attested bound
`bounds/T3-m3-lower-8.json`. It does not change the bound's tier: the
board's tiers are defined by what a certificate re-runs under its budget
(CONTRIBUTING.md), and this scan is 1.36e6 CPU-seconds.

What is shared with the batch pipeline: the dictionary (`build_dictionary`),
the Galois descent to V_3, the reduction modulo 65521 with the projection of
seed 2024 to F_ell^6, the symmetry group of order 2916, and the exact rank
decision modulo 2^31 - 1 under the Hadamard bound (`cert7.py` is a verbatim
copy of `verify_challenge/cert_t3m3_rank7.py`, used as a library). What is
not: the pivot order (the certificate's labelling with Stab(i)-minimal second
and Stab(i, j)-minimal third pivots, 5.52e13 inner steps, against the
orbit-block order's 1.31e13), the kernel (`k3.kernel3`, direct-indexed
counting on a coordinate ratio, class sets returned and decided in Python by
`decide.py` and `caseB.ranks_batch`), the partition into 326,323 pivot-pair
tasks, the case split below, the hardware (a RunPod Xeon pod and this
laptop against the batch scan's laptop), and the numba versions.

## Argument

After the descent a rank-7 decomposition is seven independent stabilizer
states whose images modulo V_3 span four dimensions. The seven projected
images (in F_ell^6) fall into one of three cases:

- A: six of them in a 3-dimensional linear subspace;
- B: five but not six;
- C: no five.

`caseB.py` excludes B, and with it A: the two-pivot kernel of the certificate
with `need = 3` lists every canonically labelled five-set with projected
images in a 3-dimensional space (24,510,983 class sets, 25,899,498
five-subsets), and every such five-set has exact rank at most 4, whereas five
states of a rank-7 configuration are independent. `caseA.py` (from a stored
class dump) and `caseA_scan.py` (self-contained, regenerating the
certificate's 259,423 rank-6 class sets) exclude A on their own: no coplanar
six-subset has the exact rank pair (6, 7). `caseC.py` excludes C with the
three-pivot kernel `k3.kernel3` over the canonical triples (i, j, k): each
emitted class set contains every case-C configuration with that canonical
labelling, and every class set is rejected by an exact rank pair. The full
argument, including the canonical-labelling lemma, the one projected-geometry
fact it needs (a state on the projected line through two canonical pivots is
on the exact line; checked exhaustively), and the exactness of every step, is
in the file `REPORT-rank7.md` of the working directory this was run from
(stabrank-work/rank8 in the genesis checkout); this README records what was
run and what was checked.

## Runs

| run | where | wall | CPU | result |
|---|---|---|---|---|
| `sym_exact.py` | laptop | 68 s | 68 s | stabilizer of V_3 of order 2916, 45 orbits, exact |
| `cost_c.py` | laptop | 5 min | 5 min | 326,323 canonical pivot pairs, 21,177 with nontrivial Stab(i, j) |
| `caseA.py 8` | laptop; pod | 22 s; 76 s | 157 s; 403 s | 1,711,129 six-subsets, rank pairs (3, 6): 1,597,050, (4, 6): 114,079 |
| `caseA_scan.py 4` | laptop | 88 s | 305 s | 748,765 six-subsets, (3, 6): 640,796, (4, 6): 107,969 |
| `caseB.py 8` | laptop | 78 s | 602 s | 25,899,498 five-subsets, none of rank 5 |
| `caseC.py 6`, session 1 | RunPod pod 3 (Xeon Gold 6342, cgroup quota 7.65 cores shared with another job), nice 10 | 643 s | 2,828 s | 612 tasks, then a kernel buffer overflow; `caseC.scan` now halves the third-pivot range on overflow (`test_split.py`) |
| `caseC.py 6`, session 2 | same | 334,351 s | 1,359,655 s | 325,711 tasks; 1,714,877,884 class sets; 0 containing V_3 |
| `caseC_supp.py 14` | laptop, nice 10 | 2,717 s (45 min) | 36,533 s (10.1 h) | 21,173 tasks; 220,971,842 class sets; 0 containing V_3 |

Rank-pair histogram (rank S, rank S + V_3) of the pod scan: (4, 7)
1,713,075,489; (5, 7) 1,464,994; (5, 8) 336,320; (6, 7) 1,061; (6, 8) 15;
(6, 9) 5. Of the supplement: (4, 7): 220,516,777; (5, 7): 437,895; (5, 8): 17,159; (6, 7): 8; (6, 8): 3. Every pair is strict. A value of 9
is exact only as a lower bound (the Hadamard bound covers values up to 8),
which is all the strict inequality needs.

## The defect found by the checks, and its repair

`cost_c.kok_of` computed the mask of admissible third pivots for a pair with
nontrivial Stab(i, j) as the states fixed by the minimum over the nontrivial
elements of Stab(i, j), which excludes every orbit minimum that no nontrivial
element fixes. The masks the pod scan used are subsets of the correct ones
(orbit minima under the full Stab(i, j)); on 21,173 pairs, 158,231,569
canonical third pivots carrying 2,137,587,169,048 inner steps (3.9 percent of
the corrected total 55,184,899,475,797) were not scanned. `check_caseC.py`
found this by recomputing the masks from the exact group action; `kok_fix.py`
computes the correct and the missing masks; `caseC_supp.py` scanned exactly
the missing triples with the same kernel and decision. The union of the two
scans visits every canonical triple once. The pod scan alone did not exclude
rank 7; the pod scan and the supplement together do.

## Checks (`check_caseC.py`, `logs/check_caseC.log`)

- coverage: the pod records are the task list, each task once, full range;
- step accounting: recorded steps plus the steps of the third pivots the
  kernel skips because they lie on the projected line through the pivots
  (which it does not count) equal the exact prediction, task by task, and the
  same for the supplement against the corrected total;
- decisions: every rank pair strict, no candidates, histograms consistent,
  every class set of more than 8 states decided at the first arithmetic level;
- symmetry data: first pivots are the 45 orbit minima, second-pivot masks
  equal the certificate's, third-pivot masks recomputed (the defect above);
- projected geometry: no state has zero projected image, none is projected-
  parallel to a first pivot, and all 2,107 states on a projected line through
  a canonical pivot pair are on the exact line.

## Files

- Scripts as run: `cert7.py`, `setup_m3.py`, `decide.py`, `sym_exact.py`,
  `symact.py`, `cost_c.py` (with the defect, unchanged), `k3.py`, `caseA.py`,
  `caseB.py`, `caseC.py`, `test_split.py`, `run_caseC_pod.sh`, `status_c.py`,
  the controls `controlB_planted.py`, `controlB_witness.py`,
  `controlC_planted.py`, `control_m2.py`.
- Added on 2026-09-26: `kok_fix.py`, `caseC_supp.py`, `check_caseC.py`,
  `caseA_scan.py`.
- `logs/`: `caseB.json`, `caseB_stage1.json`, `caseB.log`, `sym_exact.log`,
  `costC.log`, `control_m2.log`, `controlC_planted.log`,
  `caseA.json.laptop`, `caseA_scan.json`, `caseC_supp.json`,
  `caseC_supp_progress.jsonl.xz` (the 21,173 supplement records; uncompressed
  SHA-256 076d77e9b27cf7bcd5736f408465b09d2296c278f3c2d5e5feb30c16338df0f1),
  `check_caseC.log`, `check_caseC.json`.
- `logs/pod/`: `caseA.json`, `caseA.log`, `caseC.json`,
  `caseC_progress.jsonl.xz` (the 326,323 per-task records; the uncompressed
  file has SHA-256
  9a264923e4c8ba5c3fbd536afa6199ade88211648aa2efd870f2d8c69030ff85),
  `compute_pod_stabrank.jsonl` (the pod's compute-ledger records of the two
  sessions), `pod_md5.txt` (MD5 of every script and data file on the pod as
  it ran).
- Not shipped (rebuilt by the scripts, or too large): `logs/setup_m3.npz`
  (58 s), `logs/sym_exact.npz` (68 s), `logs/costC.npz` and
  `logs/costC_kok.pkl` (5 min, 640 MB), the stored rank-6 class dump that
  `caseA.py` reads (`caseA_scan.py` replaces it).

## Reproduction

From this directory, with a Python that has numpy and numba,
`OMP_NUM_THREADS=1`:

```
python sym_exact.py            # 68 s: the group, writes logs/sym_exact.npz
python cost_c.py               # 5 min: the as-run masks, logs/costC.npz, logs/costC_kok.pkl
python kok_fix.py              # 75 s: the corrected masks and the 2.1376e12 missing steps
python caseA_scan.py 8         # under 90 s: case A
python caseB.py 8              # 78 s wall on 8 workers: case B
python test_split.py           # 6 s: the overflow halving is lossless
xz -dk logs/pod/caseC_progress.jsonl.xz logs/caseC_supp_progress.jsonl.xz
python check_caseC.py logs/pod/caseC_progress.jsonl     # about 4 min: the checks
python caseC.py 6              # the full case-C scan, 378 CPU-hours, resumable
python caseC_supp.py 14        # the supplement, 2,717 s (45 min) on 14 workers
```
