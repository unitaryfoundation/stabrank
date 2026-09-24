#pragma once

// Modular linear algebra shared by the cover kernels (cover5.cpp, cover6.cpp),
// the dense family solve (dense_solve.cpp) and the p = 3 stage A matcher
// (slice_match3.cpp): the two primes of verify_challenge/slice_cover.py, the
// inverse, the row reduction modulo one vector, the reduced row echelon form
// of an augmented system and the fullness test of CoverEnumerator.is_full.
// Internal to cpp/src; every function is a plain int64 routine with the
// arrays row-major.

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stabrank {
namespace modular {

constexpr int64_t P1 = 65521;
constexpr int64_t P2 = 2013265921;

inline int64_t mod(int64_t x, int64_t p) {
    x %= p;
    return x < 0 ? x + p : x;
}

inline int64_t inv_mod(int64_t a, int64_t p) {
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

// The inverses 1..P1 - 1 modulo P1, computed once.
inline const std::vector<int64_t>& inverses_p1() {
    static const std::vector<int64_t> table = [] {
        std::vector<int64_t> t(P1, 0);
        for (int64_t a = 1; a < P1; ++a) t[a] = inv_mod(a, P1);
        return t;
    }();
    return table;
}

// rows (n x D) minus the multiple of v that clears v's first nonzero
// coordinate, that coordinate removed: n x (D - 1), modulo P1
// (slice_cover._reduce).
inline std::vector<int64_t> reduce_p1(const std::vector<int64_t>& rows, int64_t n, int64_t D, const int64_t* v,
                                      const char* who) {
    int64_t c = -1;
    for (int64_t d = 0; d < D; ++d) if (v[d] != 0) { c = d; break; }
    if (c < 0) throw std::invalid_argument(std::string(who) + ": reduction by a zero vector");
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
// mod p; the rank of A, the pivot column per rank row, and whether the last
// column is consistent (no row beyond the rank has a nonzero last entry).
struct Rref {
    std::vector<int64_t> M;
    int64_t rows = 0, cols = 0;   // cols = r + 1
    std::vector<int64_t> pivcol;  // per rank row
    int64_t rank = 0;
    bool consistent = true;
};

inline Rref rref_aug(const std::vector<const int64_t*>& colvecs, int64_t dim, const int64_t* b, int64_t p) {
    Rref R;
    const int64_t r = static_cast<int64_t>(colvecs.size());
    R.rows = dim;
    R.cols = r + 1;
    R.M.resize(dim * (r + 1));
    for (int64_t x = 0; x < dim; ++x) {
        for (int64_t c = 0; c < r; ++c) R.M[x * (r + 1) + c] = mod(colvecs[c][x], p);
        R.M[x * (r + 1) + r] = b == nullptr ? 0 : mod(b[x], p);
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
inline bool is_full(const Rref& R) {
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

// Rank mod p of the dim x k matrix with the given columns.
inline int64_t rank_of(const std::vector<const int64_t*>& colvecs, int64_t dim, int64_t p) {
    return rref_aug(colvecs, dim, nullptr, p).rank;
}

}  // namespace modular
}  // namespace stabrank
