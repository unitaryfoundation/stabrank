#include "stabrank/cover5.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

namespace stabrank {

namespace {

constexpr int64_t P1 = 65521;
constexpr int64_t P2 = 2013265921;

inline int64_t mod(int64_t x, int64_t p) {
    x %= p;
    return x < 0 ? x + p : x;
}

int64_t inv_mod(int64_t a, int64_t p) {
    int64_t t = 0, newt = 1, r = p, newr = mod(a, p);
    while (newr != 0) {
        const int64_t q = r / newr;
        int64_t tmp = t - q * newt;
        t = newt;
        newt = tmp;
        tmp = r - q * newr;
        r = newr;
        newr = tmp;
    }
    return t < 0 ? t + p : t;
}

// rows (n x D) minus the multiple of v that clears v's first nonzero
// coordinate, that coordinate removed: n x (D - 1).
std::vector<int64_t> reduce(const std::vector<int64_t>& rows, int64_t n, int64_t D, const int64_t* v) {
    int64_t c = -1;
    for (int64_t d = 0; d < D; ++d) if (v[d] != 0) { c = d; break; }
    if (c < 0) throw std::invalid_argument("cover5: reduction by a zero vector");
    const int64_t iv = inv_mod(v[c], P1);
    std::vector<int64_t> out(n * (D - 1));
    for (int64_t s = 0; s < n; ++s) {
        const int64_t* row = &rows[s * D];
        const int64_t f = mod(row[c] * iv, P1);
        int64_t col = 0;
        for (int64_t d = 0; d < D; ++d) {
            if (d == c) continue;
            out[s * (D - 1) + col++] = mod(row[d] - f * v[d], P1);
        }
    }
    return out;
}

// Reduced row echelon form of the augmented system [A | b] (dim x (r + 1))
// mod p; returns the rank of A and fills pivcol; the last column is
// consistent iff no row beyond the rank has a nonzero last entry.
struct Rref {
    std::vector<int64_t> M;
    int64_t rows, cols;          // cols = r + 1
    std::vector<int64_t> pivcol; // per rank row
    int64_t rank = 0;
    bool consistent = true;
};

Rref rref_aug(const std::vector<const int64_t*>& colvecs, int64_t dim, const int64_t* b, int64_t p) {
    Rref R;
    const int64_t r = static_cast<int64_t>(colvecs.size());
    R.rows = dim;
    R.cols = r + 1;
    R.M.resize(dim * (r + 1));
    for (int64_t x = 0; x < dim; ++x) {
        for (int64_t c = 0; c < r; ++c) R.M[x * (r + 1) + c] = mod(colvecs[c][x], p);
        R.M[x * (r + 1) + r] = mod(b[x], p);
    }
    int64_t rank = 0;
    for (int64_t c = 0; c < r && rank < dim; ++c) {
        int64_t pr = -1;
        for (int64_t t = rank; t < dim; ++t) if (R.M[t * (r + 1) + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        if (pr != rank)
            for (int64_t d = 0; d <= r; ++d) std::swap(R.M[rank * (r + 1) + d], R.M[pr * (r + 1) + d]);
        const int64_t f = inv_mod(R.M[rank * (r + 1) + c], p);
        for (int64_t d = 0; d <= r; ++d) R.M[rank * (r + 1) + d] = mod(R.M[rank * (r + 1) + d] * f, p);
        for (int64_t t = 0; t < dim; ++t) {
            if (t == rank || R.M[t * (r + 1) + c] == 0) continue;
            const int64_t g = R.M[t * (r + 1) + c];
            for (int64_t d = 0; d <= r; ++d)
                R.M[t * (r + 1) + d] = mod(R.M[t * (r + 1) + d] - g * R.M[rank * (r + 1) + d], p);
        }
        R.pivcol.push_back(c);
        ++rank;
    }
    R.rank = rank;
    for (int64_t t = rank; t < dim; ++t)
        if (R.M[t * (r + 1) + r] != 0) { R.consistent = false; break; }
    return R;
}

// Some solution of A d = b has every coordinate nonzero: coordinate c is
// dead iff it is a pivot column whose particular value is zero and whose
// row has no free-column entry (CoverEnumerator.is_full).
bool is_full(const Rref& R) {
    const int64_t r = R.cols - 1;
    std::vector<char> is_piv(r, 0);
    for (int64_t t = 0; t < R.rank; ++t) is_piv[R.pivcol[t]] = 1;
    for (int64_t t = 0; t < R.rank; ++t) {
        const int64_t* row = &R.M[t * R.cols];
        if (row[r] != 0) continue;
        bool dead = true;
        for (int64_t c = 0; c < r && dead; ++c)
            if (!is_piv[c] && row[c] != 0) dead = false;
        if (dead) return false;
    }
    return true;
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
            int64_t h = 0;
            for (int64_t d = 0; d < D2; ++d) h = mod(h + f[d] * mod(res[d] * s, P1), P1);
            key[ll] = h;
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
