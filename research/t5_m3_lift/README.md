# research/t5_m3_lift: chi(|T5>^3) >= 6 by slice-and-lift at exact rank

Status: design and controls only; nothing here is a bound yet. The
argument, the lemma with its case split, the costs, and the launch plan
are in `docs/notes/t5_m3_lift_design.md`.

The cell chi(|T5>^2) = 5 is settled and the census of
`research/t5q_m2_rank5` (2026-09-24, 74 batches, every unit run) is the
complete listing of the rank-5 decompositions of |T5>^2 up to the unitary
symmetry group: one set, the five Z(x)Z eigensectors, fixed by the group.
A rank-5 decomposition of |T5>^3 sliced at the third ququint must show
that set at every slice value, every term visible (chi(|T5>^2) = 5 leaves
no room for a zero or a coincident slice), so each lifted term is
determined by the pattern of sectors it slices to. `lift.py` enumerates
the patterns and decides exactly over Q(zeta_5) whether any pattern vector
is a stabilizer state. None is, so no rank-5 decomposition of |T5>^3
exists.

Files

| file | role |
|---|---|
| `exact.py` | Q(zeta_n) with rational arithmetic (tuples of `Fraction`s modulo the cyclotomic polynomial), the exact span solve, F_p flats, and the exact stabilizer test returning the board's (k, x0, W, Q, l) |
| `lift.py` | `LiftProblem` (the family of sets, the exact coefficients, the pruned pattern search, the lift matching), the `cell` command (reads the census records through `aggregate.check_batch`, closes the hits under the symmetry group, re-decides them exactly, runs the lift, prints `CERTIFIED chi(T5^3) >= 6`), the controls, and `plan` |
| `fill_draft.py` | fills `T5-m3-lower-6.json.draft` (date, compute block) into a submission file once the manifest exists |
| `T5-m3-lower-6.json.draft` | the bound file at the attested tier (the census is the stored enumeration; the lift runs under the budget) |
| `pod/pod_chain.sh` | the pod chain: kernel import, census and lift controls, the batch-73 rate probe against the committed record, optionally the whole census again with every hash compared, `aggregate.py --list --recheck 2`, `lift.py cell --low-census`, the certificate under the verifier; anchored `T5M3_*` markers |
| `results/` | one JSON record per command |
| `../t5q_m2_rank5/aggregate.py --list` | the listing mode of the census aggregate: a stored decomposition is reported rather than failed, the manifest is written, and the two seeded re-runs are made |
| `../../verify_challenge/cert_t5_m3_lift_attested.py` | the certificate: `aggregate.py --list --recheck 2 --recheck-seed 20261001`, then `lift.py cell --trust-low-census`; prints the m = 3 claim and the m = 4 projection |
| `../../tests/test_t5_m3_lift.py` | the field, the stabilizer test, and the lift on the basis, the rank-3 sets, the census set with a planted product, and the qutrit controls |

Commands, as run on the laptop (one process at a time, nice 19; the
repository's `.venv` extension predates the 5-cover kernel, so the batch
probe used a fresh venv built from this checkout):

```
python research/t5_m3_lift/lift.py control-stabtest
python research/t5_m3_lift/lift.py control-basis
python research/t5_m3_lift/lift.py control-rank3
python research/t5_m3_lift/lift.py control-product
python research/t5_m3_lift/lift.py control-t3-basis
python research/t5_m3_lift/lift.py control-t3-m2
python research/t5_m3_lift/lift.py control-n-m2
python research/t5_m3_lift/lift.py cell --low-census
python research/t5q_m2_rank5/aggregate.py --list --dry-run --no-low-census
python research/t5q_m2_rank5/batch.py 73 --out-dir <scratch> --force    # the rate probe
```

The pod launch commands are in the header of `pod/pod_chain.sh`.
