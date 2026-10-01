"""Action of the exact V_3 stabilizer G' (logs/sym_exact.npz, from sym_exact.py) on the
dictionary: permutations of elements, orbit data, stabilizers of states and pairs."""
import itertools, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
VEC = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int64)
WTS = np.array([243, 81, 27, 9, 3, 1])
ADD = ((VEC[:, None, :] + VEC[None, :, :]) % 3 * WTS).sum(2).astype(np.int16)
A_, B_ = VEC[:, :3], VEC[:, 3:]
OMEGA = (((A_[:, None, :] * B_[None, :, :]).sum(2) - (A_[None, :, :] * B_[:, None, :]).sum(2)) % 3).astype(np.int8)
DCONJ = (np.concatenate([A_, (-B_) % 3], axis=1) * WTS).sum(1).astype(np.int16)


class Group:
    def __init__(self, path=os.path.join(HERE, "logs", "sym_exact.npz")):
        z = np.load(path)
        self.S, self.t, self.k = z["S"], z["t"], z["k"]
        self.order = int(z["order"]); assert self.order == len(self.S)
        self.gens = list(zip(z["gens_S"], z["gens_t"], z["gens_k"]))
        self.roots = z["roots"]
        self.Lc, self.Le = z["Lc"].astype(np.int64), z["Le"].astype(np.int64)
        self.N = self.Lc.shape[0]
        self.KEY = self._pack(self.Lc, self.Le)
        self.RND = np.random.default_rng(7).integers(0, 2**63 - 1, size=729 * 3, dtype=np.int64)
        self.HASH = self.RND[self.KEY].sum(axis=1)
        self.hord = np.argsort(self.HASH); self.HS = self.HASH[self.hord]
        assert len(np.unique(self.HS)) == self.N

    @staticmethod
    def _pack(Lc, Le):
        order = np.argsort(Lc, axis=-1, kind="stable")
        Ls = np.take_along_axis(Lc, order, axis=-1).astype(np.int64)
        Es = np.take_along_axis(Le, order, axis=-1).astype(np.int64)
        return Ls * 3 + Es

    def _transform(self, S, t, k, Lc, Le):
        Lnew = ((Lc[..., None] // WTS) % 3) if False else VEC[Lc]          # (..., 27, 6)
        Lnew = ((Lnew @ S.T) % 3 * WTS).sum(-1)
        Enew = (Le + OMEGA[t, Lnew].astype(np.int64)) % 3
        if k:
            Lnew = DCONJ[Lnew].astype(np.int64); Enew = (-Enew) % 3
        return Lnew, Enew

    def act(self, S, t, k):
        """Exact permutation of the dictionary induced by K^k W(t) C_S."""
        Lnew, Enew = self._transform(S, int(t), int(k), self.Lc, self.Le)
        key = self._pack(Lnew, Enew)
        h = self.RND[key].sum(axis=1)
        pos = np.searchsorted(self.HS, h)
        assert np.all(pos < self.N) and np.all(self.HS[pos] == h)
        P = self.hord[pos]
        assert np.array_equal(self.KEY[P], key)
        return P

    def stabilizer_indices(self, s, subset=None):
        """Indices (into the element list) of the elements fixing state s (optionally among subset)."""
        idx = np.arange(self.order) if subset is None else np.asarray(subset)
        L, e = self.Lc[s], self.Le[s]
        out = []
        for c0 in range(0, len(idx), 4096):
            ch = idx[c0:c0 + 4096]
            V = VEC[L]                                                   # (27, 6)
            Lnew = ((np.einsum('gij,kj->gki', self.S[ch], V) % 3) * WTS).sum(-1)   # (g, 27)
            Enew = (e[None, :] + OMEGA[self.t[ch][:, None], Lnew].astype(np.int64)) % 3
            kk = self.k[ch].astype(bool)
            Lnew = np.where(kk[:, None], DCONJ[Lnew].astype(np.int64), Lnew)
            Enew = np.where(kk[:, None], (-Enew) % 3, Enew)
            key = self._pack(Lnew, Enew)
            h = self.RND[key].sum(axis=1)
            hit = np.flatnonzero(h == self.HASH[s])
            for g in hit:
                assert np.array_equal(key[g], self.KEY[s])
            out.extend(ch[hit].tolist())
        return np.array(out, dtype=np.int64)

    def element(self, n):
        return self.S[n], int(self.t[n]), int(self.k[n])
