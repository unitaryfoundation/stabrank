"""Shared setup for the census pipeline of the rank-5 exclusion of |T5>^2
(docs/notes/t5q_m2_rank5_exclusion.md): the two-ququint dictionary (3,900
stabilizer states) with its phase codes and modular images over Q(zeta_5)
reduced modulo 65521 and 2013265921, the target psi_2 = |T5>^2 as the phase
pattern w_5^(x^3 + y^3), the unitary symmetry group (order 50, 98 orbit
representatives), the pivot plan (pivot, partner, member mask) of the
5-cover enumerators, the compiled kernel `cover5_pair` and its Python
reference, the exact and numerical decision of a 5-set, the low-rank
censuses (k = 1 to 4) through the same reductions, canonical hashing, and
the partition loaders.

The exclusion is the emptiness of the census of 5-sets of distinct states
whose span contains psi_2: with ranks 1 to 4 excluded, every rank-5
decomposition is such a 5-set (its states independent, its coefficients
nonzero), and one member of every symmetry orbit of such 5-sets is listed
by the kernel from its pivot and partner.

This module is imported as `common` by driver.py, batch.py and aggregate.py
from research/t5q_m2_rank5 only; no other research directory is put on the
path.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from slice_lift import stabilizer_orbit_labels  # noqa: E402

ORBIT = "T5"
M = 2                        # copies of |T5>
RANK = 5                     # the rank excluded
P1 = 65521                   # 1 + 5 x 13,104: the hashing prime
P2 = 2013265921              # 1 + 5 x 402,653,184: the deciding prime
N_STATES = 3900              # 5^2 (5 + 1)(5^2 + 1)
GROUP_ORDER = 50             # the unitary symmetry group of psi_2: 5^2 local, times the copy swap
DEFAULT_SEED = 17            # the kernel's functional seed (slice_cover's default)
MAX_RUN = 4096               # the kernel raises on a parallel class above this
NUM_TOL = 1e-7               # residual tolerance for a numerical span decision (norms are 5)
SEC_PER_M2_DEFAULT = 7.49e-8  # laptop seconds per squared member count (the feasibility note's fit)
PARTITION = os.path.join(HERE, "partition.json")
RESULTS = os.path.join(HERE, "results")
RATES = os.path.join(RESULTS, "rates.json")
CLAIM = "CERTIFIED chi(T5^2) >= 6"
CLAIM_M3 = "CERTIFIED chi(T5^3) >= 6"
CLAIM_M4 = "CERTIFIED chi(T5^4) >= 6"


# ------------------------------------------------------------- hashing ----

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def lower_priority():
    """nice 19 for this process; a no-op when the priority is already at the
    floor (macOS raises PermissionError there, Linux clamps)."""
    try:
        os.nice(19)
    except OSError:
        pass


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_hashed(path, body):
    """Write `body` with its own sha256 appended, atomically."""
    body = dict(body)
    body.pop("sha256", None)
    body["sha256"] = sha256_json(body)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f)
        f.write("\n")
    os.replace(tmp, path)
    return body["sha256"]


def load_hashed(path, what):
    with open(path) as f:
        doc = json.load(f)
    body = {k: v for k, v in doc.items() if k != "sha256"}
    if sha256_json(body) != doc.get("sha256"):
        raise ValueError(f"{what} {path}: sha256 does not match its content")
    return doc


def load_partition(path=PARTITION):
    part = load_hashed(path, "partition")
    for key in ("batch_geometry", "dictionary_sha256", "plan_sha256", "units", "N", "group_order"):
        if key not in part:
            raise ValueError(f"{path} lacks `{key}`; rerun driver.py partition")
    return part


def batch_geometry(part, index):
    geos = part["batch_geometry"]
    if 0 <= index < len(geos) and geos[index]["index"] == index:
        return geos[index]
    for geo in geos:
        if geo["index"] == index:
            return geo
    raise IndexError(f"batch index {index} not in the partition (0..{len(geos) - 1})")


# --------------------------------------------------------------- field ----

class Field5:
    """Q(zeta_5) modulo a prime p = 1 mod 5: enough for the ququint
    stabilizer states (entries fifth roots of unity up to a scalar) and for
    |T5>^n (entries w_5^{sum x^3}). w is the smallest-base element of order
    5 (a fixed choice; any of the four nontrivial fifth roots would do)."""

    def __init__(self, p):
        self.p = p
        assert (p - 1) % 5 == 0, f"{p} is not 1 mod 5"
        for g in range(2, p):
            w = pow(g, (p - 1) // 5, p)
            if w != 1:
                break
        self.w = w
        assert pow(w, 5, p) == 1 and w != 1
        self.wpow = np.array([pow(w, k, p) for k in range(5)], dtype=np.int64)
        self.inv_table = None
        if p < 1 << 20:
            t = np.zeros(p, dtype=np.int64)
            t[1:] = [pow(int(a), p - 2, p) for a in range(1, p)]
            self.inv_table = t

    def inv(self, a):
        a = np.asarray(a, dtype=np.int64) % self.p
        if self.inv_table is not None:
            return self.inv_table[a]
        return np.vectorize(lambda x: pow(int(x), self.p - 2, self.p), otypes=[np.int64])(a)

    def codes_to_field(self, codes):
        """Phase codes (0 zero, 1..5 = w^0..w^4) to field elements."""
        out = np.zeros(codes.shape, dtype=np.int64)
        nz = codes > 0
        out[nz] = self.wpow[np.asarray(codes[nz], dtype=np.int64) - 1]
        return out


def reduce_rows(F, rows, v):
    """rows - rows[:, c] v / v[c] with c the first nonzero coordinate of v,
    column c removed; returns (residues, c). A residue is zero exactly when
    the row lies in span(v) plus the span already quotiented out."""
    c = int(np.flatnonzero(v)[0])
    f = (rows[:, c] * F.inv(v[c])) % F.p
    res = (rows - f[:, None] * v[None, :]) % F.p
    return np.delete(res, c, axis=1), c


def canon_rows(F, R):
    """Rows scaled so the first nonzero entry is 1; returns (rows, has) with
    `has` false on zero rows (left unchanged)."""
    nz = R != 0
    has = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = R[np.arange(len(R)), first]
    scale = np.where(has, F.inv(lead), 0)
    return (R * scale[:, None]) % F.p, has


def pair_key(F, Rc, rng):
    """A two-functional key of canonical rows over F_p (about p^2 values):
    equal rows have equal keys, and an accidental collision of unequal rows
    has probability about p^-2 per pair instead of the p^-1 of one
    functional. Used by the low-rank censuses, where every candidate is
    decided one by one in Python."""
    fa = rng.integers(1, F.p, size=Rc.shape[1])
    fb = rng.integers(1, F.p, size=Rc.shape[1])
    return ((Rc @ fa) % F.p) * F.p + ((Rc @ fb) % F.p)


def groups_by_key(keys):
    """Indices grouped by equal key, groups of size >= 2 only."""
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    brk = np.flatnonzero(sk[1:] != sk[:-1]) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [len(sk)]))
    return [order[s:e] for s, e in zip(starts, ends) if e - s >= 2]


def rank_mod(rows, p):
    """Rank of an integer matrix over F_p (Python ints, exact)."""
    A = [[int(x) % p for x in row] for row in rows]
    n, m = len(A), (len(A[0]) if A else 0)
    rank = 0
    for c in range(m):
        piv = None
        for r in range(rank, n):
            if A[r][c]:
                piv = r
                break
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = pow(A[rank][c], p - 2, p)
        A[rank] = [(x * inv) % p for x in A[rank]]
        for r in range(n):
            if r != rank and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % p for x, y in zip(A[r], A[rank])]
        rank += 1
    return rank


# ---------------------------------------------------------- dictionary ----

W5 = np.exp(2j * np.pi / 5)


def patterns5(D):
    """Phase codes (0 zero, 1..5 = w_5^0..w_5^4) of the dictionary columns
    with the first nonzero entry made 1, and the unnormalized complex
    pattern matrix. Every nonzero entry of every column must be a fifth
    root of unity times the column scalar within 1e-6, else this raises."""
    dim, N = D.shape
    nz = np.abs(D) > 1e-9
    first = np.argmax(nz, axis=0)
    ref = D[first, np.arange(N)]
    W = D / ref[None, :]
    codes = np.zeros((N, dim), dtype=np.int8)
    for c in range(5):
        codes[np.abs(W.T - W5 ** c) < 1e-6] = c + 1
    if not np.array_equal(codes > 0, nz.T):
        raise AssertionError("a dictionary entry is not a fifth root of unity times the column scalar")
    C = np.zeros((dim, N), dtype=complex)
    for c in range(5):
        C[codes.T == c + 1] = W5 ** c
    return codes, C


def target_codes(n):
    """The phase codes of psi_n = |T5>^n: w_5^(sum_q x_q^3), code = exponent + 1."""
    pts = itertools.product(range(5), repeat=n)
    return np.array([sum(x ** 3 for x in pt) % 5 + 1 for pt in pts], dtype=np.int8)


def _native_cover5():
    if os.environ.get("STABRANK_NO_NATIVE"):
        return None
    try:
        from stabrank.stabrank_core import cover5_pair
    except ImportError:
        return None
    return cover5_pair


class Target:
    """A target vector in the three forms the pipeline uses: mod P1 (with
    the dictionary reduced modulo its span, the kernel's Q input), mod P2,
    and as an unnormalized complex vector."""

    def __init__(self, E, t1, t2, tC):
        self.t1 = np.asarray(t1, dtype=np.int64) % P1
        self.t2 = np.asarray(t2, dtype=np.int64) % P2
        self.tC = np.asarray(tC, dtype=complex)
        if not np.any(self.t1):
            raise ValueError("the target vanishes mod P1")
        self.Q1, _ = reduce_rows(E.F1, E.U1, self.t1)


class T5Enumerator:
    """The n-ququint dictionary with its modular images, the target, the
    unitary symmetry group and the pivot plan of the qubit 5-cover
    enumerator (slice_cover.CoverEnumerator), for the compiled kernel and
    its Python reference."""

    def __init__(self, n=M, native=True, seed=DEFAULT_SEED):
        self.n = n
        self.D = dictionary(5, n)                     # asserts the count 5^n prod (5^j + 1)
        self.N = self.D.shape[1]
        self.dim = self.D.shape[0]
        self.codes, self.C = patterns5(self.D)
        self.F1, self.F2 = Field5(P1), Field5(P2)
        self.U1 = self.F1.codes_to_field(self.codes)
        self.U2 = self.F2.codes_to_field(self.codes)
        self.tcodes = target_codes(n)
        self.psi1 = self.F1.codes_to_field(self.tcodes)
        self.psi2 = self.F2.codes_to_field(self.tcodes)
        self.psi = psi_for(ORBIT, n)                  # the complex target, unit norm
        ratio = self.psi / self.psi[0]
        if not np.allclose(ratio, W5 ** (self.tcodes.astype(int) - 1), atol=1e-9):
            raise AssertionError("the target is not the phase pattern w_5^(sum x^3)")
        self.psiC = W5 ** (self.tcodes.astype(int) - 1)  # the unnormalized complex pattern
        reps, info = symmetry_orbit_reps(ORBIT, n, self.D, antiunitary=False)
        self.reps, self.info = np.sort(reps), info
        self.rng_seed = seed
        self.rng = np.random.default_rng(seed)
        self.target = Target(self, self.psi1, self.psi2, self.psiC)
        self.Q1 = self.target.Q1                      # (N, dim - 1), mod span(psi)
        self.native_cover5 = _native_cover5() if native else None

    def planted_target(self, states, coeffs):
        """The target sum_k c_k u_{s_k} with integer coefficients, in the
        three forms (the plant of the controls)."""
        c = np.asarray(coeffs, dtype=np.int64)
        idx = [int(x) for x in states]
        t1 = (c[:, None] * self.U1[idx]).sum(axis=0) % P1
        t2 = (c[:, None] * self.U2[idx]).sum(axis=0) % P2
        tC = self.C[:, idx] @ c.astype(complex)
        return Target(self, t1, t2, tC)

    # -- identity --
    def dictionary_sha256(self):
        """Hash of the phase codes in dictionary order: the state list."""
        return sha256_bytes(np.ascontiguousarray(self.codes).tobytes())

    # -- plan --
    def pivot_plan(self, i):
        """(members, partners) of pivot i: the states whose orbit root is at
        or above i, and one state per orbit of (a subgroup of) the pivot's
        stabilizer among them, the pivot excluded."""
        roots = self.info["roots"]
        orbit_size = int(np.count_nonzero(roots == i))
        labels, _ = stabilizer_orbit_labels(self.info["perms"], int(i),
                                            stabilizer_order=self.info["order"] // orbit_size)
        cand = np.flatnonzero(roots >= i)
        _, first = np.unique(labels[cand], return_index=True)
        partners = cand[first]
        partners = partners[partners != i]
        return cand, partners

    def units(self):
        """Every (pivot, partner, members above the partner) unit of the
        5-cover enumeration, in pivot order then partner order; the member
        count is the plan's (roots at or above the pivot, index above the
        partner, the pivot itself not counted)."""
        out = []
        for i in self.reps:
            i = int(i)
            members, partners = self.pivot_plan(i)
            for j in partners:
                j = int(j)
                Mc = int(np.count_nonzero(members > j)) - (1 if i > j else 0)
                out.append((i, j, Mc))
        return out

    def member_mask(self, i):
        members, _ = self.pivot_plan(int(i))
        mask = np.zeros(self.N, dtype=bool)
        mask[members] = True
        return mask

    # -- decisions --
    def decide(self, idx, target=None):
        """Exact and numerical decision of the set `idx` of states: whether
        the target (psi unless given) lies in their span mod P2, whether it
        does numerically (residual of the least-squares fit below NUM_TOL),
        the numerical rank, the coefficients, and whether every coefficient
        is nonzero."""
        T = self.target if target is None else target
        idx = [int(x) for x in idx]
        rows2 = [self.U2[k] for k in idx]
        r0 = rank_mod(rows2, P2)
        r1 = rank_mod(rows2 + [T.t2], P2)
        exact = bool(r0 == r1)
        A = self.C[:, idx]
        d, *_ = np.linalg.lstsq(A, T.tC, rcond=None)
        res = float(np.linalg.norm(A @ d - T.tC))
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        numeric = bool(res < NUM_TOL)
        return {"exact_mod_p2": exact, "numeric": numeric, "agree": exact == numeric,
                "decomposition": exact and numeric, "rank": rank, "rank_mod_p2": int(r0),
                "terms_count": len(idx), "independent": rank == len(idx),
                "nonzero": bool(np.all(np.abs(d) > 1e-7)),
                "residual": res, "coeffs": [[float(z.real), float(z.imag)] for z in d]}

    # -- the 5-cover pair kernel --
    def pair_sets_native(self, i, j, mask, max_run=MAX_RUN, target=None):
        """The compiled kernel: (sets, candidates, members) with every set a
        sorted 5-tuple through i and j whose span contains the target (psi
        unless given) mod P2, with its fullness flags mod P1 and mod P2 as
        the kernel decided them."""
        T = self.target if target is None else target
        idx, flags, ncand, members = self.native_cover5(
            T.Q1, self.U1, T.t1, self.U2, T.t2, int(i), int(j),
            np.ascontiguousarray(mask, dtype=np.uint8), int(max_run), int(self.rng_seed))
        sets = [(tuple(int(x) for x in row), bool(f1), bool(f2))
                for row, (f1, f2) in zip(np.asarray(idx).reshape(-1, 5), np.asarray(flags).reshape(-1, 2))]
        return sets, int(ncand), int(members)

    def pair_sets_reference(self, i, j, mask, max_run=MAX_RUN, rng=None, target=None):
        """The Python reference of the kernel (slice_cover.pair_covers, r = 5,
        ported to the ququint arrays): pivot i, partner j, third pivot k
        over the members above j with a nonzero image mod span(psi, u_i,
        u_j), and a pair (a, b) above k whose images mod span(psi, u_i, u_j,
        u_k) are nonzero and parallel, keyed by a random functional over
        F_65521; every candidate decided mod P2 (span) with fullness flags
        mod both primes. Same return shape as the native kernel."""
        F = self.F1
        p = F.p
        rng = self.rng if rng is None else rng
        T = self.target if target is None else target
        i, j = int(i), int(j)
        if i == j:
            raise ValueError("bad pivot pair")
        if not np.any(T.Q1[i]):
            raise ValueError("the pivot lies in the span of the target")
        Qi, _ = reduce_rows(F, T.Q1, T.Q1[i])
        if not np.any(Qi[j]):
            return [], 0, 0
        R, _ = reduce_rows(F, Qi, Qi[j])
        keep = np.asarray(mask, dtype=bool) & (np.arange(self.N) > j)
        keep[i] = False
        ids = np.flatnonzero(keep)
        Rm = R[ids]
        has = Rm.any(axis=1)
        ids, Rm = ids[has], Rm[has]
        Mm = len(ids)
        if Mm < 3:
            return [], 0, Mm
        f = rng.integers(1, p, size=Rm.shape[1])
        first = np.argmax(Rm != 0, axis=1)
        lead = Rm[np.arange(Mm), first]
        Rk = (Rm * F.inv(lead)[:, None]) % p
        coef = Rm[:, first].T                                          # coef[k, l] = Rm[l, first[k]]
        res = (Rm[None, :, :] - coef[:, :, None] * Rk[:, None, :]) % p   # (M, M, D2)
        nz = res != 0
        hasres = nz.any(axis=2)
        f2 = np.argmax(nz, axis=2)
        lead2 = np.take_along_axis(res, f2[:, :, None], axis=2)[:, :, 0]
        sc = np.where(hasres, F.inv(lead2), 0)
        res = (res * sc[:, :, None]) % p
        key = (res @ f) % p
        sent = p + np.arange(Mm)
        bad = ~hasres | (np.arange(Mm)[None, :] <= np.arange(Mm)[:, None])
        key = np.where(bad, sent[None, :], key)
        order = np.argsort(key, axis=1, kind="stable")
        sk = np.take_along_axis(key, order, axis=1)
        eq = sk[:, 1:] == sk[:, :-1]
        sets, ncand = [], 0
        for k, l in zip(*np.nonzero(eq)):
            if l > 0 and eq[k, l - 1]:
                continue
            e = l + 1
            while e < Mm - 1 and eq[k, e]:
                e += 1
            run = order[k, l:e + 1]
            if len(run) > max_run:
                raise RuntimeError(f"parallel class of size {len(run)} at pivot {i}, {j}, {ids[k]}")
            for a, b in itertools.combinations(sorted(ids[run].tolist()), 2):
                ncand += 1
                idx = tuple(sorted((i, j, int(ids[k]), int(a), int(b))))
                rows2 = [self.U2[t] for t in idx]
                if rank_mod(rows2 + [T.t2], P2) != rank_mod(rows2, P2):
                    continue
                sets.append((idx, self._full_mod(idx, self.U1, T.t1, P1),
                             self._full_mod(idx, self.U2, T.t2, P2)))
        return sets, ncand, Mm

    @staticmethod
    def _full_mod(idx, U, psi, p):
        """Some solution of sum_t d_t u_t = psi mod p has every d_t nonzero
        (cover5.cpp's is_full): false when the system is inconsistent, or a
        pivot coordinate is forced to zero with no free column in its row."""
        r = len(idx)
        A = [[int(U[t][x]) % p for t in idx] + [int(psi[x]) % p] for x in range(len(psi))]
        n = len(A)
        rank, pivcol = 0, []
        for c in range(r):
            piv = None
            for t in range(rank, n):
                if A[t][c]:
                    piv = t
                    break
            if piv is None:
                continue
            A[rank], A[piv] = A[piv], A[rank]
            inv = pow(A[rank][c], p - 2, p)
            A[rank] = [(x * inv) % p for x in A[rank]]
            for t in range(n):
                if t != rank and A[t][c]:
                    g = A[t][c]
                    A[t] = [(x - g * y) % p for x, y in zip(A[t], A[rank])]
            pivcol.append(c)
            rank += 1
        if any(A[t][r] for t in range(rank, n)):
            return False
        is_piv = [False] * r
        for c in pivcol:
            is_piv[c] = True
        for t in range(rank):
            if A[t][r] != 0:
                continue
            if not any(A[t][c] for c in range(r) if not is_piv[c]):
                return False
        return True

    def pair_sets(self, i, j, mask, native=None, max_run=MAX_RUN, target=None):
        use = self.native_cover5 is not None if native is None else (native and self.native_cover5 is not None)
        if use:
            return self.pair_sets_native(i, j, mask, max_run, target=target)
        return self.pair_sets_reference(i, j, mask, max_run, target=target)

    # -- low-rank censuses --
    def low_sets(self, k, i, j, mask, target=None, rng=None):
        """The k-set candidates (k = 2, 3, 4) through the pivot i (and, for
        k = 4, the partner j) over the members in `mask`, by the same
        reductions as the 5-cover kernel: k = 2, members whose image mod
        span(target, u_i) vanishes; k = 3, parallel pairs of members mod
        span(target, u_i); k = 4, parallel pairs above j mod span(target,
        u_i, u_j), keyed by two functionals. Every candidate is decided mod
        P2 and numerically. Returns (hits, candidates): the sets whose span
        contains the target under either decision, with their decisions."""
        F = self.F1
        rng = self.rng if rng is None else rng
        T = self.target if target is None else target
        i = int(i)
        mask = np.asarray(mask, dtype=bool).copy()
        mask[i] = False
        hits, cands = [], 0

        def consider(idx):
            nonlocal cands
            cands += 1
            d = self.decide(idx, T)
            if d["exact_mod_p2"] or d["numeric"]:
                hits.append({"states": [int(x) for x in idx], **{kk: d[kk] for kk in (
                    "exact_mod_p2", "numeric", "agree", "decomposition", "rank", "nonzero", "residual")}})

        if not np.any(T.Q1[i]):
            raise ValueError("the pivot lies in the span of the target")
        Qi, _ = reduce_rows(F, T.Q1, T.Q1[i])
        if k == 2:
            for l in np.flatnonzero(mask & ~Qi.any(axis=1)):
                consider(tuple(sorted((i, int(l)))))
            return hits, cands
        if k == 3:
            ids = np.flatnonzero(mask)
            Rc, has = canon_rows(F, Qi[ids])
            ids, Rc = ids[has], Rc[has]
            if len(ids) >= 2:
                for g in groups_by_key(pair_key(F, Rc, rng)):
                    for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                        consider(tuple(sorted((i, a, b))))
            return hits, cands
        if k != 4:
            raise ValueError("low_sets takes k in 2..4")
        j = int(j)
        if not np.any(Qi[j]):
            return hits, cands                                 # u_j in span(target, u_i): rank <= 2
        R, _ = reduce_rows(F, Qi, Qi[j])
        ids = np.flatnonzero(mask & (np.arange(self.N) > j))
        Rc, has = canon_rows(F, R[ids])
        ids, Rc = ids[has], Rc[has]
        if len(ids) >= 2:
            for g in groups_by_key(pair_key(F, Rc, rng)):
                for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                    consider(tuple(sorted((i, j, a, b))))
        return hits, cands

    def census_low(self, k, pivots=None, rng=None):
        """The k-set census of psi for k = 1 to 4 over the pivot plan: k = 1,
        states parallel to psi (a zero row of Q1); k = 2 and 3 through
        low_sets at every pivot over its members; k = 4 through low_sets at
        every (pivot, partner). Returns a record with the sets whose span
        contains psi (expected none), the candidate count and the unit
        count. Complete for rank k once ranks below k are excluded (a
        member skipped for a zero image would give a lower rank), one per
        symmetry orbit as far as the pivot and partner reductions go."""
        rng = self.rng if rng is None else rng
        hits, cands, units = [], 0, 0
        if k == 1:
            for l in np.flatnonzero(~self.Q1.any(axis=1)):
                cands += 1
                d = self.decide((int(l),))
                if d["exact_mod_p2"] or d["numeric"]:
                    hits.append({"states": [int(l)], **{kk: d[kk] for kk in (
                        "exact_mod_p2", "numeric", "agree", "decomposition", "rank", "nonzero", "residual")}})
            return {"k": 1, "hits": hits, "candidates": cands, "units": 1}
        if k not in (2, 3, 4):
            raise ValueError("census_low takes k in 1..4")
        for i in (self.reps if pivots is None else pivots):
            i = int(i)
            members, partners = self.pivot_plan(i)
            mask = np.zeros(self.N, dtype=bool)
            mask[members] = True
            if k in (2, 3):
                units += 1
                h, c = self.low_sets(k, i, None, mask, rng=rng)
                hits += h
                cands += c
                continue
            for j in partners:
                units += 1
                h, c = self.low_sets(4, i, int(j), mask, rng=rng)
                hits += h
                cands += c
        return {"k": k, "hits": hits, "candidates": cands, "units": units}


def make_enumerator(native=True, n=M):
    return T5Enumerator(n=n, native=native)


def hit_record(E, idx, f1, f2, target=None):
    """(deterministic part, numeric part) of one kernel set: the states, the
    kernel's fullness flags, and the exact and numerical decisions."""
    d = E.decide(idx, target)
    det = {"states": [int(x) for x in idx], "full_mod_p1": bool(f1), "full_mod_p2": bool(f2),
           **{k: d[k] for k in ("exact_mod_p2", "numeric", "agree", "decomposition", "rank",
                                "rank_mod_p2", "terms_count", "independent", "nonzero")}}
    return det, {"residual": d["residual"], "coeffs": d["coeffs"]}
