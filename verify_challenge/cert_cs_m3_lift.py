"""Certificate: chi(|CS>^{ot 3}) >= 4 for the two-qubit controlled-S magic
state, by slice-and-lift along a whole copy.

chi(|CS>^2) = 3: the three-term witness CS-m2-upper-3.json and the exact
rank-2 exclusion of cert_cs_m2_rank2.py, re-run here numerically. Suppose
|CS>^3 = c_1 s_1 + c_2 s_2 + c_3 s_3 with stabilizer states s_i on six
qubits, and slice along the first copy (two qubits): u_i^(x) = (<x| (x) I)
s_i for x in F_2^2. All four amplitudes of |CS> are nonzero, so every slice
x gives |CS>^2 = sum_i (c_i / alpha_x) u_i^(x); no u_i^(x) vanishes and no
two are parallel, else chi(|CS>^2) <= 2. The argument then has three steps.

(i) The base. Slice 00 is one of the rank-3 decompositions of |CS>^2 over
the 36,720 four-qubit stabilizer states. There are 16 of them, one orbit of
the unitary symmetry group of |CS>^2 (the 24 two-qubit Cliffords fixing
|CS> up to phase on each copy and the swap of the two copies, order 1152),
each consisting of one computational basis state and two full-support
states; the pivot search lists one member from each of the three pivot
orbits the class meets, three index triples that are conjugate under the
group. The group acts on |CS>^3 as I (x) U and preserves slices, so one
member would suffice.

(ii) The slice equations. For each term the other three slices are Pauli
images of the first: the x with u_i^(x) != 0 form a coset of the X-part of
the image of Stab(s_i) in the symplectic coordinates of the first copy, four
nonzero slices force that image to be all of F_2^2, so for each a there is
lambda X^a Z^c (x) Q_i^(a) in Stab(s_i) and u_i^(a) = mu_i^(a) Q_i^(a)
u_i^(00) with mu_i^(a) a fourth root of unity (slice_lift.lifts_qubit_block
has the derivation). The three slice equations

    sum_i d_i mu_i^(a) Q_i^(a) u_i = (alpha_a / alpha_00) |CS>^2,   a in {01, 10, 11},

are each a matching over the 16 Pauli classes and 4 phases per term, solved
by meet in the middle. Because the amplitude ratios of |CS> are 1, 1 and i,
all fourth roots of unity, the identity assignment Q_i = I with mu_i = 1, 1,
i solves every slice equation of every base decomposition, so the slice
equations by themselves exclude nothing. The matcher finds that this is the
only solution: one match per slice, class 0 on every term.

(iii) The stabilizer test. The only slice-compatible candidate for each term
is therefore s_i = |CS> (x) u_i, and that is not a stabilizer state, since
|CS> is not and a computational-basis slice of a stabilizer state is a
stabilizer state or zero. to_witness.term_from_vector rejects every
assembled term (its phase is not a Z_4 quadratic form with even cross
terms). This step is where the bound is decided; the per-decomposition
lines in the log show the match counts of (ii) and the rejections of (iii).
Hence chi(|CS>^3) >= 4.

The argument is numerical and rests on margins: the pivot listing groups
quotient directions at the parallel threshold 1 - 1e-6, the rank-2 exclusion
must clear it by rank_exclusion.MARGIN, and the matchings accept within 1e-7
and reject every other Pauli assignment outside a 1e-4 search window. The
bound is therefore reproduced, not verified.

Controls, all fatal. Width one: the block matcher with width 1 must agree
with `lifts_qubit` in that the rank-3 decompositions of |T>^3 lift to |T>^4
(chi stays 3) and the rank-2 decomposition of |H>^2 does not lift to |H>^3
(chi(H^3) = 3). Width two, positive: a single-qubit magic state tensored
with |+>, read as a two-qubit block state, has rank 2 at one and at two
copies (chi((psi|+>)^2) = chi(psi^2) = 2), so some rank-2 decomposition of
it must lift; this is run for the Bravyi-Kitaev T-type state (3 of its 15
rank-2 decompositions lift) and for the edge state (|0> + e^{i pi/4}|1>)/
sqrt 2 (1 of 15). Width two, negative: no rank-2 decomposition of |CS> may
lift to |CS>^2, since chi(|CS>^2) = 3; here too every slice equation has only
the identity solution and every assembled candidate fails the stabilizer
test.

Printed claim: CERTIFIED chi(CS^3) >= 4
"""

from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import MARGIN, PARALLEL, dictionary, psi_for, rank2_search  # noqa: E402
from slice_lift import (all_decompositions, confirm_lift, decompositions_with_pivot,  # noqa: E402
                        lift_all, lifts_qubit, lifts_qubit_block)
from stabrank_verify import orbit_state  # noqa: E402


def _alpha(orbit):
    return np.array([complex(x) for x in orbit_state(orbit)]).ravel()


def _coeffs(states, psi):
    A = np.column_stack(states)
    d, *_ = np.linalg.lstsq(A, psi, rcond=None)
    if np.linalg.norm(A @ d - psi) > 1e-9:
        raise AssertionError("a listed decomposition does not reproduce its target")
    return d


def width_one_controls(D2, D3):
    """The block matcher at width 1 agrees with lifts_qubit on two known cells."""
    for orbit, m, rank, D, want in (("qubit_T", 3, 3, D3, True), ("qubit_H", 2, 2, D2, False)):
        alpha, psi = _alpha(orbit), psi_for(orbit, m)
        decs, _ = all_decompositions(orbit, m, rank, D, verbose=False)
        if not decs:
            return f"no rank-{rank} decomposition of |{orbit}>^{m} listed"
        n_plain = n_block = 0
        for cols in decs:
            states = [D[:, c] for c in cols]
            d = _coeffs(states, psi)
            n_plain += len(lifts_qubit(states, d, alpha, psi, m)[0])
            n_block += len(lifts_qubit_block(states, d, alpha, psi, m, 1)[0])
        if (n_plain > 0) != want or n_plain != n_block:
            return (f"{orbit} m={m}: lifts_qubit found {n_plain} lifts, the block matcher "
                    f"{n_block}, expected {'some' if want else 'none'} and agreement")
        print(f"width-1 control: {len(decs)} rank-{rank} decomposition(s) of |{orbit}>^{m}, "
              f"{n_plain} lift(s) by both matchers ({'expected' if want else 'none expected'})")
    return None


def block_rank2_lifts(alpha, D2):
    """Lift every rank-2 decomposition of the two-qubit state `alpha` to two
    copies with the block matcher. Returns (decompositions, lifts, assembled
    candidates, candidates rejected by the stabilizer test, whether every
    slice equation had only the identity solution)."""
    decs = set()
    for i in range(D2.shape[1]):
        decs.update(decompositions_with_pivot(alpha, D2, i, 2))
    lifts = cands = rejected = 0
    identity_only = True
    for cols in sorted(decs):
        states = [D2[:, c] for c in cols]
        d = _coeffs(states, alpha)
        st = {}
        L, _ = lifts_qubit_block(states, d, alpha, alpha, 2, 2, stats=st)
        for terms in L:
            if confirm_lift(terms, alpha, alpha) > 1e-8:
                raise AssertionError("a block lift failed its confirmation")
        lifts += len(L)
        cands += st["candidates"]
        rejected += len({c for c, _, _ in st["rejected"]})
        identity_only &= st["identity_only"]
    return len(decs), lifts, cands, rejected, identity_only


def main():
    t0 = time.time()
    D2, D3, D4 = dictionary(2, 2), dictionary(2, 3), dictionary(2, 4)
    print(f"2 qubits: {D2.shape[1]} states; 3 qubits: {D3.shape[1]}; 4 qubits: {D4.shape[1]}")

    err = width_one_controls(D2, D3)
    if err:
        print(f"width-1 control FAILED: {err}", file=sys.stderr)
        return 1
    plus = np.ones(2) / np.sqrt(2)
    edge = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    for name, psi1 in (("Bravyi-Kitaev T-type state (face)", _alpha("qubit_T")),
                       ("edge state (|0> + e^{i pi/4}|1>)/sqrt 2", edge)):
        n_dec, n_lift, n_cand, n_rej, ident = block_rank2_lifts(np.kron(psi1, plus), D2)
        if not n_dec or not n_lift:
            print(f"positive control FAILED: {name} (x) |+> has {n_dec} rank-2 decompositions "
                  f"and {n_lift} block lifts to two copies, where chi stays 2", file=sys.stderr)
            return 1
        print(f"positive control: {name} (x) |+> has {n_dec} rank-2 decompositions; "
              f"{n_cand} candidates assembled, {n_rej} rejected by the stabilizer test, "
              f"{n_lift} lift(s)")
    n_dec, n_lift, n_cand, n_rej, ident = block_rank2_lifts(_alpha("CS"), D2)
    if not n_dec or n_lift:
        print(f"negative control FAILED: |CS> has {n_dec} rank-2 decompositions and "
              f"{n_lift} block lifts to |CS>^2, where chi is 3", file=sys.stderr)
        return 1
    print(f"negative control: |CS> has {n_dec} rank-2 decompositions; slice equations solved "
          f"{'only by the identity assignment' if ident else 'also by non-identity assignments'}, "
          f"{n_cand} candidates assembled and all {n_rej} rejected by the stabilizer test; "
          "none lifts to |CS>^2\n")

    # chi(|CS>^2) >= 3 (exact in cert_cs_m2_rank2.py; numerically here)
    pairs, worst = rank2_search(psi_for("CS", 2), D4)
    if pairs == "RANK1" or pairs or (1 - PARALLEL) - worst < MARGIN:
        print("CS m=2: rank-2 exclusion did not hold", file=sys.stderr)
        return 1
    print(f"CS m=2: no pair of the {D4.shape[1]} four-qubit states spans |CS>^2; closest "
          f"approach to parallel {worst:.6f}")
    decs, reps = all_decompositions("CS", 2, 3, D4, verbose=False)
    if not decs:
        print("no rank-3 decomposition of |CS>^2 listed, but the board holds one", file=sys.stderr)
        return 1
    print(f"CS m=2: {len(decs)} rank-3 decomposition(s) listed from {len(reps)} pivot orbits "
          f"(the 16 decompositions form one orbit of the unitary symmetry group) "
          f"[{time.time() - t0:.0f}s]")
    stats = []
    lifted = lift_all("CS", 2, decs, D4, stats_out=stats)
    if lifted:
        print(f"CS m=3: {len(lifted)} rank-3 decomposition(s) FOUND by lifting")
        return 1
    if len(stats) != len(decs) or any(len(st["matches"]) < 3 for st in stats):
        print("CS m=3: some slice equation had no solution; that is not the expected "
              "shape of this exclusion, see the lines above", file=sys.stderr)
        return 1
    ident = all(st["identity_only"] for st in stats)
    n_cand = sum(st["candidates"] for st in stats)
    n_rej = sum(len({c for c, _, _ in st["rejected"]}) for st in stats)
    if n_rej != n_cand:
        print("CS m=3: an assembled candidate survived the stabilizer test but no lift was "
              "returned; inconsistent", file=sys.stderr)
        return 1
    print(f"CS m=3: every slice equation of every listed decomposition is solved "
          f"{'only by the identity assignment (mu = 1, 1, i), so the only candidate terms are |CS> (x) u_i' if ident else 'by the identity assignment and by others'}; "
          f"all {n_cand} assembled candidates fail the stabilizer test, which is the step that "
          f"decides the bound [{time.time() - t0:.0f}s]")
    print("CERTIFIED chi(CS^3) >= 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
