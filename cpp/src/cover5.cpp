#include "stabrank/cover5.hpp"

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

std::vector<int64_t> reduce(const std::vector<int64_t>& rows, int64_t n, int64_t D, const int64_t* v) {
    return modular::reduce_p1(rows, n, D, v, "cover5");
}

}  // namespace

Cover5Result cover5_pair(const Cover5Inputs& in) {
    Cover5Result out;
    const int64_t N = in.N, D = in.D, dim = in.dim;
    if (in.i < 0 || in.i >= N || in.j < 0 || in.j >= N || in.i == in.j)
        throw std::invalid_argument("cover5: bad pivot pair");
    // modulo span(target, u_i), then span(target, u_i, u_j)
    std::vector<int64_t> Q(in.Q, in.Q + N * D);
    bool zi = true;
    for (int64_t d = 0; d < D; ++d) if (Q[in.i * D + d] != 0) { zi = false; break; }
    if (zi) throw std::invalid_argument("cover5: the pivot lies in the span of the target");
    std::vector<int64_t> Qi = reduce(Q, N, D, &Q[in.i * D]);
    const int64_t D1 = D - 1;
    bool zj = true;
    for (int64_t d = 0; d < D1; ++d) if (Qi[in.j * D1 + d] != 0) { zj = false; break; }
    if (zj) return out;                                   // u_j in span(target, u_i)
    std::vector<int64_t> R = reduce(Qi, N, D1, &Qi[in.j * D1]);
    const int64_t D2 = D1 - 1;
    // members above j with a nonzero image
    std::vector<int64_t> ids;
    for (int64_t l = in.j + 1; l < N; ++l) {
        if (l == in.i || in.members[l] == 0) continue;
        bool z = true;
        for (int64_t d = 0; d < D2; ++d) if (R[l * D2 + d] != 0) { z = false; break; }
        if (!z) ids.push_back(l);
    }
    const int64_t M = static_cast<int64_t>(ids.size());
    out.members = M;
    if (M < 3) return out;
    std::mt19937_64 rng(in.seed);
    std::uniform_int_distribution<int64_t> U(1, P1 - 1);
    std::vector<int64_t> f(D2);
    for (auto& v : f) v = U(rng);
    // the wide key: a second functional drawn after the first, so that the
    // first is the one the narrow key has always used
    if (in.key_functionals != 1 && in.key_functionals != 2)
        throw std::invalid_argument("cover5: key_functionals must be 1 or 2");
    std::vector<int64_t> f2(in.key_functionals == 2 ? D2 : 0);
    for (auto& v : f2) v = U(rng);
    // per member: leading column and inverse leading entry
    std::vector<int64_t> lead(M), ilead(M);
    for (int64_t t = 0; t < M; ++t) {
        const int64_t* row = &R[ids[t] * D2];
        int64_t c = 0;
        while (row[c] == 0) ++c;
        lead[t] = c;
        ilead[t] = inv_mod(row[c], P1);
    }
    std::vector<int64_t> key(M), res(D2);
    std::vector<int64_t> order;
    std::vector<int64_t> group;
    for (int64_t kk = 0; kk < M; ++kk) {
        const int64_t k = ids[kk];
        const int64_t* rk = &R[k * D2];
        const int64_t c = lead[kk], ic = ilead[kk];
        order.clear();
        for (int64_t ll = kk + 1; ll < M; ++ll) {
            const int64_t* rl = &R[ids[ll] * D2];
            const int64_t g = mod(rl[c] * ic, P1);
            int64_t first = -1;
            for (int64_t d = 0; d < D2; ++d) {
                res[d] = mod(rl[d] - g * rk[d], P1);
                if (first < 0 && res[d] != 0) first = d;
            }
            if (first < 0) continue;                      // u_l in span(target, u_i, u_j, u_k)
            const int64_t s = inv_mod(res[first], P1);
            int64_t h = 0, h2 = 0;
            for (int64_t d = 0; d < D2; ++d) {
                const int64_t rd = mod(res[d] * s, P1);
                h = mod(h + f[d] * rd, P1);
                if (!f2.empty()) h2 = mod(h2 + f2[d] * rd, P1);
            }
            key[ll] = f2.empty() ? h : h * P1 + h2;
            order.push_back(ll);
        }
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return key[a] != key[b] ? key[a] < key[b] : a < b;
        });
        for (size_t s = 0; s < order.size();) {
            size_t e = s + 1;
            while (e < order.size() && key[order[e]] == key[order[s]]) ++e;
            if (e - s >= 2) {
                if (static_cast<int64_t>(e - s) > in.max_run)
                    throw std::runtime_error("cover5: parallel class of size " + std::to_string(e - s)
                                             + " at pivot " + std::to_string(in.i) + ", " + std::to_string(in.j)
                                             + ", " + std::to_string(k));
                group.assign(order.begin() + s, order.begin() + e);
                std::sort(group.begin(), group.end());
                for (size_t a = 0; a < group.size(); ++a) {
                    for (size_t b = a + 1; b < group.size(); ++b) {
                        ++out.candidates;
                        std::array<int64_t, 5> idx = {in.i, in.j, k, ids[group[a]], ids[group[b]]};
                        std::sort(idx.begin(), idx.end());
                        std::vector<const int64_t*> c2(5), c1(5);
                        for (int t = 0; t < 5; ++t) {
                            c2[t] = in.U2 + idx[t] * dim;
                            c1[t] = in.U1 + idx[t] * dim;
                        }
                        Rref R2 = rref_aug(c2, dim, in.psi2, P2);
                        if (!R2.consistent) continue;    // target outside the span
                        Rref R1 = rref_aug(c1, dim, in.psi1, P1);
                        out.covers.push_back({idx, R1.consistent && is_full(R1), is_full(R2)});
                    }
                }
            }
            s = e;
        }
    }
    return out;
}

}  // namespace stabrank
