#include "stabrank/dense_solve.hpp"

#include "modular_detail.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

namespace stabrank {

namespace {

using modular::P1;
using modular::P2;
using modular::mod;

// det of an n x n matrix (n <= 3) given as rows of values mod p.
inline int64_t det_small(const int64_t* M, int n, int64_t p) {
    if (n == 0) return 1;
    if (n == 1) return mod(M[0], p);
    if (n == 2) return mod(mod(M[0] * M[3], p) - mod(M[1] * M[2], p), p);
    int64_t d = 0;
    d = mod(d + mod(M[0] * mod(mod(M[4] * M[8], p) - mod(M[5] * M[7], p), p), p), p);
    d = mod(d - mod(M[1] * mod(mod(M[3] * M[8], p) - mod(M[5] * M[6], p), p), p), p);
    d = mod(d + mod(M[2] * mod(mod(M[3] * M[7], p) - mod(M[4] * M[6], p), p), p), p);
    return d;
}

// The Laplace features (S, C) with |S| = |C| = kk over kk = 0..n, in the
// Python's order (itertools.combinations, S outer, C inner), with the
// complementary sets and the sign (-1)^(sum S + sum C).
struct Feature {
    std::vector<int> S, C, Sc, Cc;
    int sign;
};

void combos(int n, int k, int start, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
    if (static_cast<int>(cur.size()) == k) { out.push_back(cur); return; }
    for (int a = start; a < n; ++a) {
        cur.push_back(a);
        combos(n, k, a + 1, cur, out);
        cur.pop_back();
    }
}

std::vector<Feature> features(int n) {
    std::vector<Feature> out;
    for (int kk = 0; kk <= n; ++kk) {
        std::vector<std::vector<int>> subsets;
        std::vector<int> cur;
        combos(n, kk, 0, cur, subsets);
        for (auto& S : subsets)
            for (auto& C : subsets) {
                Feature f;
                f.S = S;
                f.C = C;
                for (int a = 0; a < n; ++a) {
                    if (std::find(S.begin(), S.end(), a) == S.end()) f.Sc.push_back(a);
                    if (std::find(C.begin(), C.end(), a) == C.end()) f.Cc.push_back(a);
                }
                int s = 0;
                for (int a : S) s += a;
                for (int a : C) s += a;
                f.sign = (s % 2) ? -1 : 1;
                out.push_back(f);
            }
    }
    return out;
}

// det of the submatrix M[rows, cols] of an n x n matrix.
inline int64_t minor_det(const int64_t* M, int n, const std::vector<int>& rows, const std::vector<int>& cols,
                         int64_t p) {
    const int k = static_cast<int>(rows.size());
    int64_t sub[9];
    for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) sub[a * k + b] = M[rows[a] * n + cols[b]];
    return det_small(sub, k, p);
}

// u in span(V) mod p for u (dim) and V (dim x k, given as k column vectors).
bool in_span(const std::vector<int64_t>& u, const std::vector<std::vector<int64_t>>& V, int64_t dim, int64_t p) {
    std::vector<const int64_t*> cols;
    for (auto& v : V) cols.push_back(v.data());
    if (cols.empty()) {
        for (int64_t y = 0; y < dim; ++y) if (mod(u[y], p) != 0) return false;
        return true;
    }
    return modular::rref_aug(cols, dim, u.data(), p).consistent;
}

}  // namespace

DenseSolveResult dense_solve(const DenseSolveInputs& in) {
    DenseSolveResult out;
    const int r = in.r, n = in.kappa1 + 1;
    const int64_t dim = in.dim;
    if (in.kappa1 < 1 || in.kappa1 > 2) throw std::invalid_argument("dense_solve: kappa1 must be 1 or 2");
    if (in.nL + in.nR != r) throw std::invalid_argument("dense_solve: the sides do not partition the terms");
    std::mt19937_64 rng(in.seed);
    std::uniform_int_distribution<int64_t> U(1, P1 - 1);
    std::vector<int64_t> f(static_cast<size_t>(n) * dim);
    for (auto& v : f) v = U(rng);
    // coef[i] = (d0_i, K_i1, ..., K_ik); frhs[a] = f_a . rhs
    std::vector<int64_t> frhs(n, 0);
    for (int a = 0; a < n; ++a)
        for (int64_t y = 0; y < dim; ++y) frhs[a] = mod(frhs[a] + f[a * dim + y] * mod(in.rhs1[y], P1), P1);
    // T[i][o] = n x n matrix (f_a . w_o) coef_i[b]
    std::vector<std::vector<int64_t>> T(r);
    for (int i = 0; i < r; ++i) {
        const int64_t sz = in.sizes[i];
        T[i].assign(static_cast<size_t>(sz) * n * n, 0);
        int64_t coef[3] = {mod(in.d01[i], P1), 0, 0};
        for (int b = 0; b < in.kappa1; ++b) coef[b + 1] = mod(in.K1[i * in.kappa1 + b], P1);
        for (int64_t o = 0; o < sz; ++o) {
            const int64_t* w = in.opts1[i] + o * dim;
            for (int a = 0; a < n; ++a) {
                int64_t fw = 0;
                for (int64_t y = 0; y < dim; ++y) fw = mod(fw + f[a * dim + y] * mod(w[y], P1), P1);
                for (int b = 0; b < n; ++b) T[i][(o * n + a) * n + b] = mod(fw * coef[b], P1);
            }
        }
    }
    const std::vector<Feature> feats = features(n);
    const int NF = static_cast<int>(feats.size());
    // one side: every combination of its terms' options, its summed matrix
    // and its feature vector
    auto build = [&](const int* side, int ns, bool left, std::vector<int64_t>& F, std::vector<int32_t>& idx) {
        int64_t total = 1;
        for (int t = 0; t < ns; ++t) total *= in.sizes[side[t]];
        F.assign(static_cast<size_t>(total) * NF, 0);
        idx.assign(static_cast<size_t>(total) * std::max(ns, 1), 0);
        std::vector<int64_t> M(n * n);
        for (int64_t c = 0; c < total; ++c) {
            std::fill(M.begin(), M.end(), 0);
            int64_t rem = c;
            for (int t = 0; t < ns; ++t) {
                const int i = side[t];
                const int64_t o = rem % in.sizes[i];
                rem /= in.sizes[i];
                idx[c * std::max(ns, 1) + t] = static_cast<int32_t>(o);
                const int64_t* Ti = &T[i][o * n * n];
                for (int e = 0; e < n * n; ++e) M[e] += Ti[e];
            }
            for (int e = 0; e < n * n; ++e) M[e] = mod(M[e], P1);
            if (left)
                for (int a = 0; a < n; ++a) M[a * n] = mod(M[a * n] - frhs[a], P1);
            for (int t = 0; t < NF; ++t) {
                const Feature& ft = feats[t];
                int64_t v = left ? minor_det(M.data(), n, ft.S, ft.C, P1)
                                 : mod(ft.sign * minor_det(M.data(), n, ft.Sc, ft.Cc, P1), P1);
                F[c * NF + t] = v;
            }
        }
        return total;
    };
    std::vector<int64_t> FL, FR;
    std::vector<int32_t> idxL, idxR;
    const int64_t nL = build(in.sideL, in.nL, true, FL, idxL);
    const int64_t nR = build(in.sideR, in.nR, false, FR, idxR);
    // whole-vector decision of one combination
    std::vector<int> combo(r);
    std::vector<int64_t> u1(dim), u2(dim);
    std::vector<std::vector<int64_t>> V1(in.kappa1, std::vector<int64_t>(dim));
    std::vector<std::vector<int64_t>> V2(in.kappa2, std::vector<int64_t>(dim));
    auto decide = [&]() {
        for (int64_t y = 0; y < dim; ++y) u1[y] = mod(-in.rhs1[y], P1);
        for (auto& v : V1) std::fill(v.begin(), v.end(), 0);
        for (int i = 0; i < r; ++i) {
            const int64_t* w = in.opts1[i] + static_cast<int64_t>(combo[i]) * dim;
            const int64_t d0 = mod(in.d01[i], P1);
            for (int64_t y = 0; y < dim; ++y) u1[y] = mod(u1[y] + d0 * w[y], P1);
            for (int b = 0; b < in.kappa1; ++b) {
                const int64_t kb = mod(in.K1[i * in.kappa1 + b], P1);
                if (kb == 0) continue;
                for (int64_t y = 0; y < dim; ++y) V1[b][y] = mod(V1[b][y] + kb * w[y], P1);
            }
        }
        if (!in_span(u1, V1, dim, P1)) return false;
        ++out.pass1;
        for (int64_t y = 0; y < dim; ++y) u2[y] = mod(-in.rhs2[y], P2);
        for (auto& v : V2) std::fill(v.begin(), v.end(), 0);
        for (int i = 0; i < r; ++i) {
            const int64_t* w = in.opts2[i] + static_cast<int64_t>(combo[i]) * dim;
            const int64_t d0 = mod(in.d02[i], P2);
            for (int64_t y = 0; y < dim; ++y) u2[y] = mod(u2[y] + mod(d0 * mod(w[y], P2), P2), P2);
            for (int b = 0; b < in.kappa2; ++b) {
                const int64_t kb = mod(in.K2[i * in.kappa2 + b], P2);
                if (kb == 0) continue;
                for (int64_t y = 0; y < dim; ++y) V2[b][y] = mod(V2[b][y] + mod(kb * mod(w[y], P2), P2), P2);
            }
        }
        return in_span(u2, V2, dim, P2);
    };
    // the dense product: features are below P1 < 2^16, so NF <= 20 products
    // below 2^32 sum to below 2^37 and one reduction per pair suffices
    const int strideL = std::max(in.nL, 1), strideR = std::max(in.nR, 1);
    for (int64_t a = 0; a < nL; ++a) {
        const int64_t* fl = &FL[a * NF];
        for (int64_t b = 0; b < nR; ++b) {
            const int64_t* fr = &FR[b * NF];
            int64_t acc = 0;
            for (int t = 0; t < NF; ++t) acc += fl[t] * fr[t];
            if (acc % P1 != 0) continue;
            if (++out.raw > in.max_cand)
                throw std::runtime_error(std::to_string(out.raw) + " slice candidates: structural degeneracy of the family");
            for (int t = 0; t < in.nL; ++t) combo[in.sideL[t]] = idxL[a * strideL + t];
            for (int t = 0; t < in.nR; ++t) combo[in.sideR[t]] = idxR[b * strideR + t];
            if (!decide()) continue;
            ++out.passed;
            out.combos.insert(out.combos.end(), combo.begin(), combo.end());
        }
    }
    return out;
}

}  // namespace stabrank
