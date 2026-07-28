"""Certificate that chi(|T3>^2) = 3, with carry-construction witnesses.

Upper bound (exact identity).  Writing w9 = exp(2 pi i / 9) and grouping
the amplitudes of |T3>^2 by sigma = (x1 + x2) mod 3 gives the rank-3
identity of Kocia-Sarovar (arXiv:2003.01130),

    3 |T3>^2 = v0 + w9 v1 + w9^2 v2,

where each carry state is a stabilizer state supported on a line:

    v0 = |00> + w3|12> + w3|21>,
    v1 = |01> + |10> + w3|22>,
    v2 = |02> + |11> + |20>.

Lower bound (exhaustive).  Sweeping all pairs from the two-qutrit
stabilizer dictionary (360 distinct states) leaves a minimum
least-squares residual of 0.4645, so no two-term decomposition exists.

The same carry blocks give explicit product witnesses downstream:
nine terms for |T3>^3 (v-block tensor computational basis) and nine
terms for |T3>^4 (v-block tensor v-block), the latter being the
witness behind the m = 4 entry chi(|T3>^4) <= 9 of the SoA table.
"""
import numpy as np

from stabrank.stabilizer_extent import enumerate_stabilizer_states
from stabrank.target_functions import qutrit_complex_magic_state

w3 = np.exp(2j * np.pi / 3)
w9 = np.exp(2j * np.pi / 9)


def carry_block():
    v0 = np.zeros(9, complex)
    v0[0], v0[5], v0[7] = 1, w3, w3
    v1 = np.zeros(9, complex)
    v1[1], v1[3], v1[8] = 1, 1, w3
    v2 = np.zeros(9, complex)
    v2[2], v2[4], v2[6] = 1, 1, 1
    return [v / np.linalg.norm(v) for v in (v0, v1, v2)]


def lsq_residual(target, basis):
    a = np.column_stack(basis)
    c, *_ = np.linalg.lstsq(a, target, rcond=None)
    return np.linalg.norm(a @ c - target)


def main():
    block = carry_block()

    # Upper bound: the three-term carry identity at m = 2.
    t2 = qutrit_complex_magic_state(2)
    recon = sum(w9 ** s * np.sqrt(3) * block[s] for s in range(3)) / 3
    err = np.linalg.norm(recon - t2)
    print(f"three-term identity residual: {err:.2e}")
    assert err < 1e-12

    # Lower bound: exhaustive pair sweep over the full dictionary.
    S = enumerate_stabilizer_states(2, 3)
    t = t2 / np.linalg.norm(t2)
    canon = set()
    for v in S:
        idx = np.argmax(np.abs(v) > 1e-10)
        canon.add(tuple(np.round(v / v[idx], 6)))
    assert len(canon) == 360, len(canon)

    c = S.conj() @ t
    G = S.conj() @ S.T
    iu, ju = np.triu_indices(S.shape[0], k=1)
    g = G[iu, ju]
    den = 1.0 - np.abs(g) ** 2
    ci, cj = c[iu], c[ju]
    num = (np.abs(ci) ** 2 + np.abs(cj) ** 2
           - 2.0 * np.real(np.conj(ci) * g * cj))
    proj2 = np.where(den > 1e-12, num / np.maximum(den, 1e-300),
                     np.maximum(np.abs(ci) ** 2, np.abs(cj) ** 2))
    r2min = float(np.min(1.0 - proj2))
    print(f"pairs checked: {len(iu)}; min pair residual: {np.sqrt(r2min):.12f}")
    assert r2min > 0.2
    print("CERTIFIED: chi(|T3>^2) = 3")

    # Product witnesses at m = 3 and m = 4.
    t3 = qutrit_complex_magic_state(3)
    basis1 = [np.eye(3, dtype=complex)[i] for i in range(3)]
    warm3 = [np.kron(a, b) for a in block for b in basis1]
    print(f"m=3 nine-term product residual: {lsq_residual(t3, warm3):.2e}")
    assert lsq_residual(t3, warm3) < 1e-12

    t4 = qutrit_complex_magic_state(4)
    warm4 = [np.kron(a, b) for a in block for b in block]
    print(f"m=4 nine-term product residual: {lsq_residual(t4, warm4):.2e}")
    assert lsq_residual(t4, warm4) < 1e-12
    print("explicit witnesses: chi(|T3>^3) <= 9 (product), chi(|T3>^4) <= 9")


if __name__ == "__main__":
    main()
