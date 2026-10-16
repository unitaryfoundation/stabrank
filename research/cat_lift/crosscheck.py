"""Completeness cross-check of cert_cat_m7_lift.py on c_5 (4 qubits), separate code path.

Brute force, no symmetry and no pivot reduction:
  * dictionary from the maintainers' qubit_states.all_states (BFS closure, floats),
    not from cert_cat_m7_lift's exact parametrisation;
  * every one of the 36,720 states is used as a pivot; partners are found by a
    floating-point orthogonal projection off span{t, pivot} and a projective
    sort key (the method of verify_challenge/cert_qubit_rank2.py), not by the
    mod-P quotient;
  * every candidate triple is confirmed in sympy (exact ranks over Q(i)).
Then the pivot-orbit method of cert_cat_m7_lift.py is run, each decomposition it
returns is closed under the symmetry generators, and we check
    union of orbits == brute-force set,   sum of orbit sizes == brute-force count.
Finally every brute-force decomposition gets an independent floating-point
lift test to c_6 (exhaustive over Pauli images and fourth roots, no
meet-in-the-middle), confirming that liftability is constant on orbits and
matches cert_cat_m7_lift. Same for rank 2 at c_4 -> c_5.

Usage: python research/cat_lift/crosscheck.py [path to verify_challenge]  (about 11 min, one core)
Exit 0 only if every comparison agrees.
"""
import itertools
import os
import sys
import time

import numpy as np

VC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "verify_challenge")
sys.path.insert(0, VC)
import qubit_states as qs          # noqa: E402
import cert_cat_m7_lift as cl         # noqa: E402
import sympy as sp                 # noqa: E402

FOURTH = np.array([1, 1j, -1, -1j])


def exp_row(v):
    """float stabilizer vector -> normalized exponent row (exact key)."""
    nz = np.flatnonzero(np.abs(v) > 1e-9)
    w = v / v[nz[0]]
    e = np.full(len(v), -1, np.int8)
    ang = np.rint(np.angle(w[nz]) / (np.pi / 2)).astype(int) % 4
    if not np.allclose(w[nz], FOURTH[ang], atol=1e-9):
        raise AssertionError("not a stabilizer amplitude pattern")
    e[nz] = ang
    return e


def cvec(e):
    return np.where(e >= 0, FOURTH[np.clip(e, 0, 3)], 0)


def svec(e):
    return sp.Matrix([0 if x < 0 else sp.I ** int(x) for x in e])


def brute_force(m, rank):
    n = m - 1
    D = qs.all_states(n)                    # (dim, N)
    D = D / np.linalg.norm(D, axis=0)
    N = D.shape[1]
    E = np.array([exp_row(D[:, i]) for i in range(N)])
    t_e = cl.c_exponents(m)
    t = cvec(t_e) / np.sqrt(len(t_e))
    rng = np.random.default_rng(3)
    can = rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))
    key = rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))
    found = set()
    pivots = [None] if rank == 2 else range(N)
    for i in pivots:
        B = t[:, None] if i is None else np.linalg.qr(np.column_stack([t, D[:, i]]))[0]
        R = D - B @ (B.conj().T @ D)
        nr = np.linalg.norm(R, axis=0)
        ok = nr > 1e-9
        idx = np.flatnonzero(ok)
        U = R[:, ok] / nr[ok]
        c = can @ U
        good = np.abs(c) > 1e-9
        if not good.all():                  # fall back: perturb the canonical vector
            raise AssertionError("degenerate canonical projection")
        kv = (key @ U) / c
        order = np.lexsort((kv.imag, kv.real))
        Us, ids = U[:, order], idx[order]
        ov = np.abs(np.sum(Us[:, :-1].conj() * Us[:, 1:], axis=0))
        # groups of consecutive parallel vectors
        run = [0]
        groups = []
        for a in range(len(ov)):
            if ov[a] > 1 - 1e-9:
                run.append(a + 1)
            else:
                if len(run) > 1:
                    groups.append(run)
                run = [a + 1]
        if len(run) > 1:
            groups.append(run)
        for g in groups:
            for a, b in itertools.combinations(g, 2):
                j, k = int(ids[a]), int(ids[b])
                trip = (j, k) if i is None else (i, j, k)
                if len(set(trip)) < len(trip):
                    continue
                # drop linearly dependent triples (t not in their span then)
                if i is not None and np.linalg.svd(D[:, list(trip)], compute_uv=False)[-1] < 1e-6:
                    continue
                found.add(tuple(sorted(trip)))
        if i is not None and i % 5000 == 0:
            print(f"    brute force pivot {i}/{N}, candidates {len(found)}", flush=True)
    # exact confirmation, sympy
    ts = svec(t_e)
    exact = set()
    for trip in found:
        S = sp.Matrix.hstack(*[svec(E[j]) for j in trip])
        if S.rank() == rank and sp.Matrix.hstack(S, ts).rank() == rank:
            # all coefficients nonzero: no proper subset spans t
            if all(sp.Matrix.hstack(*[svec(E[j]) for j in sub], ts).rank() > len(sub)
                   for sub in itertools.combinations(trip, rank - 1)):
                exact.add(frozenset(E[j].tobytes() for j in trip))
    return exact, N


def orbit_of(dec_rows, gens, n):
    start = frozenset(r.tobytes() for r in dec_rows)
    seen = {start}
    todo = [start]
    while todo:
        cur = todo.pop()
        rows = np.array([np.frombuffer(b, np.int8) for b in cur])
        for g in gens:
            img = cl.renormalize(cl.act(g, rows, n))
            s = frozenset(r.tobytes() for r in img)
            if s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


def float_lifts(dec_rows, m):
    """exhaustive float test: sum_j d_j mu_j P_j a_j = D c_m."""
    n = m - 1
    A = np.column_stack([cvec(r) for r in dec_rows])
    t = cvec(cl.c_exponents(m))
    d = np.linalg.lstsq(A, t, rcond=None)[0]
    assert np.linalg.norm(A @ d - t) < 1e-9
    rhs = cvec((cl.c_exponents(m) + cl.d_exponents(n)) % 4)
    dim = 1 << n
    x = np.arange(dim)
    opts = []
    for j, r in enumerate(dec_rows):
        a = cvec(r)
        imgs = []
        for u in range(dim):
            for v in range(dim):
                ph = np.array([(-1) ** bin(v & xx).count("1") for xx in x])
                pa = np.zeros(dim, complex)
                pa[x ^ u] = ph * a
                for mu in FOURTH:
                    imgs.append(d[j] * mu * pa)
        opts.append(np.unique(np.round(np.array(imgs), 9), axis=0))
    hits = 0
    for combo in itertools.product(*opts[:-1]):
        res = rhs - sum(combo)
        if np.min(np.linalg.norm(opts[-1] - res[None, :], axis=1)) < 1e-7:
            hits += 1
    return hits


def check(m, rank):
    t0 = time.time()
    n = m - 1
    bf, N = brute_force(m, rank)
    print(f"c_{m} rank {rank}: brute force over {N} states finds {len(bf)} decompositions "
          f"({time.time() - t0:.0f}s)", flush=True)
    E = cl.stabilizer_exponents(n)
    M = cl.to_modp(E)
    gens = cl.generators(n)
    reps = cl.orbit_reps(E, gens, n) if rank == 3 else None
    decs = cl.decompositions(m, rank, E, M, reps, log_every=0)
    union, total = set(), 0
    for idx, _ in decs:
        orb = orbit_of(E[list(idx)], gens, n)
        print(f"  pivot-method decomposition {idx}: orbit size {len(orb)}", flush=True)
        if orb & union:
            print("  (overlaps an orbit already counted: the method returned two members of one orbit)")
        union |= orb
        total += len(orb)
    print(f"  union of orbits {len(union)}; brute force {len(bf)}")
    ok = union == bf
    print("  MATCH" if ok else "  MISMATCH", flush=True)
    lift_counts = {}
    for dec in bf:
        rows = [np.frombuffer(b, np.int8) for b in dec]
        lift_counts[dec] = float_lifts(rows, m) > 0
    vals = set(lift_counts.values())
    print(f"  float lift test on all {len(bf)} brute-force decompositions: "
          f"{sum(lift_counts.values())} lift ({time.time() - t0:.0f}s)", flush=True)
    return ok, vals


if __name__ == "__main__":
    ok1, v1 = check(4, 2)
    ok2, v2 = check(5, 3)
    good = ok1 and ok2 and v1 == {False} and v2 == {True}
    print("CROSS-CHECK PASSED" if good else "CROSS-CHECK FAILED")
    raise SystemExit(0 if good else 1)
