#include "stabrank/cover6.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace stabrank;

namespace {

constexpr int64_t P1 = 65521;
constexpr int64_t P2 = 2013265921;

int64_t mod(int64_t x, int64_t p) { x %= p; return x < 0 ? x + p : x; }

int64_t pow_mod(int64_t a, int64_t e, int64_t p) {
    int64_t r = 1;
    a = mod(a, p);
    while (e > 0) {
        if (e & 1) r = mod(r * a, p);
        a = mod(a * a, p);
        e >>= 1;
    }
    return r;
}

// slice_cover._reduce mod P1
std::vector<int64_t> reduce(const std::vector<int64_t>& rows, int64_t n, int64_t D, const std::vector<int64_t>& v) {
    int64_t c = 0;
    while (v[c] == 0) ++c;
    const int64_t iv = pow_mod(v[c], P1 - 2, P1);
    std::vector<int64_t> out(n * (D - 1));
    for (int64_t s = 0; s < n; ++s) {
        const int64_t f = mod(rows[s * D + c] * iv, P1);
        int64_t col = 0;
        for (int64_t d = 0; d < D; ++d) {
            if (d == c) continue;
            out[s * (D - 1) + col++] = mod(rows[s * D + d] - f * v[d], P1);
        }
    }
    return out;
}

// Reference decision of one 6-set mod p: the rank of the states, whether
// psi is in their span, and whether some solution has every coefficient
// nonzero (CoverEnumerator.is_cover and is_full, both mod p).
struct Decision { int64_t rank; bool cover, full; };

Decision decide(const std::vector<int64_t>& U, int64_t dim, const std::vector<int64_t>& psi,
                const std::array<int64_t, 6>& S, int64_t p) {
    const int64_t r = 6, cols = r + 1;
    std::vector<int64_t> M(dim * cols);
    for (int64_t x = 0; x < dim; ++x) {
        for (int64_t c = 0; c < r; ++c) M[x * cols + c] = mod(U[S[c] * dim + x], p);
        M[x * cols + r] = mod(psi[x], p);
    }
    std::vector<int64_t> piv;
    int64_t rank = 0;
    for (int64_t c = 0; c < r && rank < dim; ++c) {
        int64_t pr = -1;
        for (int64_t t = rank; t < dim; ++t) if (M[t * cols + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        for (int64_t d = 0; d < cols; ++d) std::swap(M[rank * cols + d], M[pr * cols + d]);
        const int64_t f = pow_mod(M[rank * cols + c], p - 2, p);
        for (int64_t d = 0; d < cols; ++d) M[rank * cols + d] = mod(M[rank * cols + d] * f, p);
        for (int64_t t = 0; t < dim; ++t) {
            if (t == rank || M[t * cols + c] == 0) continue;
            const int64_t g = M[t * cols + c];
            for (int64_t d = 0; d < cols; ++d) M[t * cols + d] = mod(M[t * cols + d] - g * M[rank * cols + d], p);
        }
        piv.push_back(c);
        ++rank;
    }
    Decision D{rank, true, true};
    for (int64_t t = rank; t < dim; ++t) if (M[t * cols + r] != 0) D.cover = false;
    if (!D.cover) { D.full = false; return D; }
    std::vector<char> is_piv(r, 0);
    for (int64_t c : piv) is_piv[c] = 1;
    for (int64_t t = 0; t < rank; ++t) {
        if (M[t * cols + r] != 0) continue;
        bool dead = true;
        for (int64_t c = 0; c < r; ++c) if (!is_piv[c] && M[t * cols + c] != 0) dead = false;
        if (dead) D.full = false;
    }
    return D;
}

// Fifty states with small integer entries in dimension 9 and a planted
// target psi = 2 u3 + 3 u11 - u20 + 5 u27 + 7 u33 + 4 u41; u45 = u3 - 2 u11
// + 3 u20 + u27 (a dependency without psi), u47 = psi - 2 u11 + 3 u20 + u27
// + u33 (a 6-set {3, 11, 20, 27, 33, 47} in which the coefficient of u3
// vanishes), and u49 = 2 u3 + 3 u11 - u20 + 5 u27 + 7 u33 (a 5-cover
// {3, 11, 20, 27, 33, 49} of psi minus ... : psi = u49 + 4 u41, so
// {41, 49} is a 2-cover and every 6-set containing it is dependent or has
// dead coefficients).
struct Instance {
    int64_t N = 50, dim = 9, D = 8;
    std::vector<int64_t> E, U1, U2, psi, psi1, psi2, Q;
    std::vector<uint8_t> members;
    std::array<int64_t, 6> full = {3, 11, 20, 27, 33, 41};
    std::array<int64_t, 6> dead = {3, 11, 20, 27, 33, 47};
    std::array<int64_t, 6> dependent = {3, 11, 20, 27, 33, 45};
};

Instance make_instance(unsigned seed) {
    Instance I;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> small(-3, 3);
    I.E.assign(I.N * I.dim, 0);
    for (auto& x : I.E) x = small(rng);
    I.psi.assign(I.dim, 0);
    const int64_t c[6] = {2, 3, -1, 5, 7, 4};
    for (int t = 0; t < 6; ++t)
        for (int64_t d = 0; d < I.dim; ++d) I.psi[d] += c[t] * I.E[I.full[t] * I.dim + d];
    for (int64_t d = 0; d < I.dim; ++d) {
        auto e = [&](int64_t s) { return I.E[s * I.dim + d]; };
        I.E[45 * I.dim + d] = e(3) - 2 * e(11) + 3 * e(20) + e(27);
        I.E[47 * I.dim + d] = I.psi[d] - 2 * e(11) + 3 * e(20) + e(27) + e(33);
        I.E[49 * I.dim + d] = 2 * e(3) + 3 * e(11) - e(20) + 5 * e(27) + 7 * e(33);
    }
    I.U1.resize(I.N * I.dim);
    I.U2.resize(I.N * I.dim);
    for (int64_t k = 0; k < I.N * I.dim; ++k) { I.U1[k] = mod(I.E[k], P1); I.U2[k] = mod(I.E[k], P2); }
    I.psi1.resize(I.dim);
    I.psi2.resize(I.dim);
    for (int64_t d = 0; d < I.dim; ++d) { I.psi1[d] = mod(I.psi[d], P1); I.psi2[d] = mod(I.psi[d], P2); }
    I.Q = reduce(I.U1, I.N, I.dim, I.psi1);
    I.members.assign(I.N, 1);
    return I;
}

Cover6Result run(const Instance& I, int64_t i, int64_t j) {
    Cover6Inputs in{I.Q.data(), I.N, I.D, I.U1.data(), I.psi1.data(), I.U2.data(), I.psi2.data(), I.dim,
                    i, j, I.members.data(), 4096, 17};
    return cover6_pair(in);
}

using Set6 = std::set<std::array<int64_t, 6>>;

Set6 full_covers(const Cover6Result& r) {
    Set6 out;
    for (auto& c : r.covers) if (c.full1 && c.full2) out.insert(c.idx);
    return out;
}

// Brute force: every 6-set through i and j with the other members above j
// and admissible, independent mod P2, spanning psi, with all coefficients
// nonzero.
Set6 brute_full(const Instance& I, int64_t i, int64_t j) {
    Set6 out;
    std::vector<int64_t> pool;
    for (int64_t l = j + 1; l < I.N; ++l) if (l != i && I.members[l]) pool.push_back(l);
    const size_t n = pool.size();
    for (size_t a = 0; a < n; ++a)
        for (size_t b = a + 1; b < n; ++b)
            for (size_t c = b + 1; c < n; ++c)
                for (size_t d = c + 1; d < n; ++d) {
                    std::array<int64_t, 6> S = {i, j, pool[a], pool[b], pool[c], pool[d]};
                    std::sort(S.begin(), S.end());
                    auto dec = decide(I.U2, I.dim, I.psi2, S, P2);
                    if (dec.rank == 6 && dec.cover && dec.full) out.insert(S);
                }
    return out;
}

}  // namespace

TEST_CASE("the six-cover pair kernel lists exactly the full independent covers through a pivot pair") {
    Instance I = make_instance(3);
    for (auto [i, j] : std::vector<std::pair<int64_t, int64_t>>{{3, 11}, {11, 3}, {3, 20}, {20, 27}, {0, 1}}) {
        auto res = run(I, i, j);
        auto want = brute_full(I, i, j);
        CHECK(full_covers(res) == want);
        for (auto& c : res.covers) {
            auto d2 = decide(I.U2, I.dim, I.psi2, c.idx, P2);
            auto d1 = decide(I.U1, I.dim, I.psi1, c.idx, P1);
            CHECK(d2.cover);
            CHECK(c.full2 == d2.full);
            CHECK(c.full1 == d1.full);
            CHECK(c.rank2 == d2.rank);
        }
    }
    auto res = run(I, 3, 11);
    auto found = full_covers(res);
    CHECK(found.count(I.full) == 1);
    CHECK(found.count(I.dead) == 0);
    CHECK(found.count(I.dependent) == 0);
    bool dead_reported = false;
    for (auto& c : res.covers) if (c.idx == I.dead) { dead_reported = true; CHECK_FALSE(c.full2); }
    CHECK(dead_reported);
    CHECK(res.candidates >= static_cast<int64_t>(res.covers.size()));
    CHECK(brute_full(I, 3, 11).size() >= 2);
}

TEST_CASE("the six-cover kernel honours the partner order and the member mask") {
    Instance I = make_instance(3);
    CHECK(full_covers(run(I, 3, 20)).count(I.full) == 0);   // 11 < 20 would be the partner
    CHECK(full_covers(run(I, 11, 3)).count(I.full) == 1);   // i > j is allowed
    I.members[41] = 0;
    auto found = full_covers(run(I, 3, 11));
    CHECK(found.count(I.full) == 0);
    CHECK(found == brute_full(I, 3, 11));
}

TEST_CASE("the six-cover kernel finds nothing when the residue space is too small") {
    // dimension 4: after the target and four pivots no coordinate is left
    Instance I;
    I.N = 12;
    I.dim = 4;
    I.D = 3;
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int64_t> small(-2, 2);
    I.E.assign(I.N * I.dim, 0);
    for (auto& x : I.E) x = small(rng);
    I.psi = {1, 2, 3, 4};
    I.U1.resize(I.N * I.dim);
    I.U2.resize(I.N * I.dim);
    for (int64_t k = 0; k < I.N * I.dim; ++k) { I.U1[k] = mod(I.E[k], P1); I.U2[k] = mod(I.E[k], P2); }
    I.psi1 = I.psi2 = I.psi;
    I.Q = reduce(I.U1, I.N, I.dim, I.psi1);
    I.members.assign(I.N, 1);
    auto res = run(I, 0, 1);
    CHECK(res.covers.empty());
    CHECK(res.candidates == 0);
}
