#include "stabrank/cover6.hpp"

#include "modular_detail.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

namespace stabrank {

namespace {

using modular::P1;
using modular::P2;
using modular::Rref;
using modular::inv_mod;
using modular::is_full;
using modular::mod;
using modular::rref_aug;

// Open-addressing table over the projective keys of one fourth pivot,
// reused across pivots through a stamp; chains link the members with an
// equal key in ascending order of insertion.
struct KeyTable {
    std::vector<int64_t> key;
    std::vector<int32_t> first, last, stamp, next;
    std::vector<int32_t> groups;   // slots that hold at least two members
    int32_t cur = 0;
    size_t mask = 0;

    void reset(size_t members) {
        size_t T = 16;
        while (T < 2 * members + 16) T <<= 1;
        if (T != key.size()) {
            key.assign(T, 0);
            first.assign(T, -1);
            last.assign(T, -1);
            stamp.assign(T, 0);
            cur = 0;
        }
        mask = T - 1;
        if (next.size() < members) next.resize(members);
    }
    void begin() {
        if (++cur == 0) { std::fill(stamp.begin(), stamp.end(), 0); cur = 1; }
        groups.clear();
    }
    void insert(int64_t k, int32_t m) {
        size_t h = static_cast<size_t>(static_cast<uint64_t>(k) * 0x9E3779B97F4A7C15ull >> 20) & mask;
        while (true) {
            if (stamp[h] != cur) {
                stamp[h] = cur;
                key[h] = k;
                first[h] = last[h] = m;
                next[m] = -1;
                return;
            }
            if (key[h] == k) {
                if (first[h] == last[h]) groups.push_back(static_cast<int32_t>(h));
                next[last[h]] = m;
                last[h] = m;
                next[m] = -1;
                return;
            }
            h = (h + 1) & mask;
        }
    }
};

}  // namespace

Cover6Result cover6_pair(const Cover6Inputs& in) {
    Cover6Result out;
    const int64_t N = in.N, D = in.D, dim = in.dim;
    if (in.i < 0 || in.i >= N || in.j < 0 || in.j >= N || in.i == in.j)
        throw std::invalid_argument("cover6: bad pivot pair");
    // modulo span(target, u_i), then span(target, u_i, u_j)
    std::vector<int64_t> Q(in.Q, in.Q + N * D);
    bool zi = true;
    for (int64_t d = 0; d < D; ++d) if (Q[in.i * D + d] != 0) { zi = false; break; }
    if (zi) throw std::invalid_argument("cover6: the pivot lies in the span of the target");
    std::vector<int64_t> Qi = modular::reduce_p1(Q, N, D, &Q[in.i * D], "cover6");
    const int64_t D1 = D - 1;
    bool zj = true;
    for (int64_t d = 0; d < D1; ++d) if (Qi[in.j * D1 + d] != 0) { zj = false; break; }
    if (zj) return out;                                   // u_j in span(target, u_i)
    std::vector<int64_t> R = modular::reduce_p1(Qi, N, D1, &Qi[in.j * D1], "cover6");
    const int64_t D2 = D1 - 1;
    if (D2 < 2) return out;                               // no room for two further pivots and a pair
    // members above j with a nonzero image, ascending
    std::vector<int64_t> ids;
    for (int64_t l = in.j + 1; l < N; ++l) {
        if (l == in.i || in.members[l] == 0) continue;
        bool z = true;
        for (int64_t d = 0; d < D2; ++d) if (R[l * D2 + d] != 0) { z = false; break; }
        if (!z) ids.push_back(l);
    }
    const int64_t M = static_cast<int64_t>(ids.size());
    out.members = M;
    if (M < 4) return out;
    const std::vector<int64_t>& inv = modular::inverses_p1();
    std::mt19937_64 rng(in.seed);
    std::uniform_int_distribution<int64_t> U(1, P1 - 1);
    const int64_t D3 = D2 - 1;
    std::vector<int64_t> fa(D3), fb(D3);
    for (auto& v : fa) v = U(rng);
    for (auto& v : fb) v = U(rng);
    // the rows of the members, packed (M x D2)
    std::vector<int64_t> Rm(M * D2);
    for (int64_t t = 0; t < M; ++t) std::copy(&R[ids[t] * D2], &R[ids[t] * D2] + D2, &Rm[t * D2]);
    std::vector<int64_t> A;          // the rows above k reduced by row k: Mk x D3
    std::vector<int64_t> idk;        // their positions in ids
    std::vector<int64_t> res(D3);
    std::vector<int32_t> group;
    KeyTable table;
    for (int64_t kk = 0; kk + 3 < M; ++kk) {
        const int64_t* rk = &Rm[kk * D2];
        int64_t c = 0;
        while (rk[c] == 0) ++c;
        const int64_t ic = inv[rk[c]];
        A.clear();
        idk.clear();
        for (int64_t ll = kk + 1; ll < M; ++ll) {
            const int64_t* rl = &Rm[ll * D2];
            const int64_t g = mod(rl[c] * ic, P1);
            bool nz = false;
            size_t base = A.size();
            A.resize(base + D3);
            int64_t col = 0;
            for (int64_t d = 0; d < D2; ++d) {
                if (d == c) continue;
                const int64_t v = mod(rl[d] - g * rk[d], P1);
                A[base + col++] = v;
                nz |= v != 0;
            }
            if (!nz) { A.resize(base); continue; }         // u_l in span(target, u_i, u_j, u_k)
            idk.push_back(ll);
        }
        const int64_t Mk = static_cast<int64_t>(idk.size());
        if (Mk < 3) continue;
        // per row of A: leading column and inverse leading entry
        table.reset(static_cast<size_t>(Mk));
        for (int64_t l = 0; l + 2 < Mk; ++l) {
            const int64_t* al = &A[l * D3];
            int64_t cl = 0;
            while (al[cl] == 0) ++cl;
            const int64_t il = inv[al[cl]];
            table.begin();
            for (int64_t m = l + 1; m < Mk; ++m) {
                const int64_t* am = &A[m * D3];
                const int64_t g = mod(am[cl] * il, P1);
                int64_t first = -1;
                for (int64_t d = 0; d < D3; ++d) {
                    res[d] = mod(am[d] - g * al[d], P1);
                    if (first < 0 && res[d] != 0) first = d;
                }
                if (first < 0) continue;                  // u_m in span(target, u_i, u_j, u_k, u_l)
                const int64_t s = inv[res[first]];
                int64_t h1 = 0, h2 = 0;
                for (int64_t d = first; d < D3; ++d) {
                    const int64_t rd = mod(res[d] * s, P1);
                    h1 = mod(h1 + fa[d] * rd, P1);
                    h2 = mod(h2 + fb[d] * rd, P1);
                }
                table.insert(h1 * P1 + h2, static_cast<int32_t>(m));
            }
            for (int32_t slot : table.groups) {
                group.clear();
                for (int32_t m = table.first[slot]; m >= 0; m = table.next[m]) group.push_back(m);
                if (static_cast<int64_t>(group.size()) > in.max_run)
                    throw std::runtime_error("cover6: parallel class of size " + std::to_string(group.size())
                                             + " at pivot " + std::to_string(in.i) + ", " + std::to_string(in.j)
                                             + ", " + std::to_string(ids[kk]) + ", " + std::to_string(ids[idk[l]]));
                std::sort(group.begin(), group.end());
                for (size_t a = 0; a < group.size(); ++a) {
                    for (size_t b = a + 1; b < group.size(); ++b) {
                        ++out.candidates;
                        std::array<int64_t, 6> idx = {in.i, in.j, ids[kk], ids[idk[l]],
                                                      ids[idk[group[a]]], ids[idk[group[b]]]};
                        std::sort(idx.begin(), idx.end());
                        std::vector<const int64_t*> c2(6), c1(6);
                        for (int t = 0; t < 6; ++t) {
                            c2[t] = in.U2 + idx[t] * dim;
                            c1[t] = in.U1 + idx[t] * dim;
                        }
                        Rref R2 = rref_aug(c2, dim, in.psi2, P2);
                        if (!R2.consistent) continue;    // target outside the span
                        Rref R1 = rref_aug(c1, dim, in.psi1, P1);
                        out.covers.push_back({idx, R1.consistent && is_full(R1), is_full(R2), R2.rank});
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace stabrank
