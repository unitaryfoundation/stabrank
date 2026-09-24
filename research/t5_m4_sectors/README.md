# research/t5_m4_sectors: the Z^4 sectors of |T5>^4 have stabilizer rank 5

Question (constructions_2026_09.md, 2026-09-24): the Z^4 eigensector of
|T5>^4 is a three-ququint state of rank at most 5 (five stabilizer
sub-sector products); rank 4 would give chi(|T5>^4) <= 20 and the per-copy
exponent log_5(20)/4 = 0.4654. Answer: the rank is exactly 5, for every
eigenvalue, so the one-Pauli sector route at m = 4 gives exactly the
product 25 and nothing below it.

The argument needs no three-ququint dictionary. Slicing the sector at
x4 = a (projecting the fourth ququint onto |a>) gives w^(a^3) times the
Z^3 sector of |T5>^3 at eigenvalue w^(c - a), and slicing carries a
stabilizer decomposition term by term to stabilizer states or zeros, so
chi(m = 3 sector) <= chi(m = 4 sector) <= 5. The m = 3 sector is a
two-ququint state (the Clifford reduction of the plane x + y + z = d to its
(x, y) coordinates), and an exact census over the 3,900 two-ququint
stabilizer states excludes its ranks 1 to 4 in 20 s. The same census, run
for the four other classes of Z-type strings at m = 3, shows that every
Z-type sector of |T5>^3 has rank exactly 5, so every Z-type one-Pauli
sector of |T5>^4 has rank >= 5 as well (a Z-type string at m = 4 slices to
a Z-type string at m = 3 on the copies left).

Files

| file | role |
|---|---|
| `sector_census.py` | the exact k = 1..4 census of a two-ququint target over Q(zeta_5) modulo 65521 (candidates) and 2013265921 (decisions), plus the numerical decision; the sector targets `g(x, y) = w^(x^3 + y^3 + z(x, y)^3)` for a string `Z^c1 Z^c2 Z^c3` at eigenvalue `w^d`; the unitary symmetry group of the sector (25 translations with their quadratic phase corrections, times the copy permutations fixing the string), checked to fix the target up to phase and to permute the dictionary; pivots one per orbit, partners one per orbit of the pivot's stabilizer subgroup, the reduction of section 1.2 of `docs/notes/t5q_m2_rank5_exclusion.md`; `control` runs the planted rank-4 targets, the product decomposition of \|N5>^2, and the ten rank-3 decompositions of \|T5> |
| `sectors_m4.py` | the five Z^4 sectors of \|T5>^4 written down exactly (`u_c(x, y, z) = w^(f_c)`, `f_c = x^3 + y^3 + z^3 + (c - x - y - z)^3`), the one-class statement (`u_(c+1) = D X_1 u_c` with `D = w^(3x^2 - 3x + 1)`), the cubic part `3(xyz - (x + y + z)(xy + yz + zx))`, the slices, and the five sub-sector products (stabilizer states, independent, no four spanning); the same for `Z Z Z^2 Z^2` and `Z Z Z^4 Z^4`, the two other strings with five rank-one sub-sector terms |
| `strings_report.py` | for all 330 classes of full-support one-Pauli strings on \|T5>^4, the number of nonvanishing sub-sector terms per sector and the sub-sector rank bound (exact one-ququint ranks), minimized over the three pairings of the copies |
| `run_all.sh` | the whole laptop run, one process at a time through `research/t5_rank5/run.py` (nice 19, 600 s cap) |
| `results/` | the census records (`census_Z*_d*.json`, sha256 over the body) and the logs |

Census records: `hits` per k (empty), `candidates` per k (the modular
candidates decided), `units` (pivots for k = 2, 3; pivot-partner pairs for
k = 4), the target's phase codes, the group order, the pivot count, the
seed. The k = 4 census of one class runs in 20 to 90 s on the laptop.

What remains open for the m = 4 cell: a rank below 25 needs something
other than one-Pauli sectors (the two-Pauli totals are >= 60, the
sub-sector bound of every one-Pauli sector is >= 5, and the Z-type
sectors are exactly 5).
