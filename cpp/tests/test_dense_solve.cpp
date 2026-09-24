#include "stabrank/dense_solve.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

// Rank mod p of the columns (each dim long).
int64_t rank_mod(const std::vector<std::vector<int64_t>>& cols, int64_t dim, int64_t p) {
    const int64_t r = static_cast<int64_t>(cols.size());
    std::vector<int64_t> M(dim * r);
    for (int64_t x = 0; x < dim; ++x)
        for (int64_t c = 0; c < r; ++c) M[x * r + c] = mod(cols[c][x], p);
    int64_t rank = 0;
    for (int64_t c = 0; c < r && rank < dim; ++c) {
        int64_t pr = -1;
        for (int64_t t = rank; t < dim; ++t) if (M[t * r + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        for (int64_t d = 0; d < r; ++d) std::swap(M[rank * r + d], M[pr * r + d]);
        const int64_t f = pow_mod(M[rank * r + c], p - 2, p);
        for (int64_t d = 0; d < r; ++d) M[rank * r + d] = mod(M[rank * r + d] * f, p);
        for (int64_t t = 0; t < dim; ++t) {
            if (t == rank || M[t * r + c] == 0) continue;
            const int64_t g = M[t * r + c];
            for (int64_t d = 0; d < r; ++d) M[t * r + d] = mod(M[t * r + d] - g * M[rank * r + d], p);
        }
        ++rank;
    }
    return rank;
}

// An integer instance: r terms with small integer option vectors, an
// integer family d0 + K lambda (kappa columns), and a right-hand side
// built from a planted combination at an integer lambda, so that the
// planted combination solves the equation over Z and hence mod both
// primes. Everything is handed to the kernel reduced mod P1 and mod P2.
struct Instance {
    int r, kappa;
    int64_t dim;
    std::vector<int64_t> sizes;
    std::vector<std::vector<int64_t>> W;          // per term: sizes[i] x dim integers
    std::vector<int64_t> d0, K;                   // r, r x kappa
    std::vector<int64_t> rhs;                     // dim
    std::vector<int> planted;
    std::vector<std::vector<int64_t>> W1, W2;
    std::vector<int64_t> d01, K1, d02, K2, rhs1, rhs2;
    std::vector<const int64_t*> p1, p2;
    std::vector<int> sideL, sideR;
};

Instance make_instance(unsigned seed, int r, int kappa, int64_t dim, int64_t nopt) {
    Instance I;
    I.r = r;
    I.kappa = kappa;
    I.dim = dim;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> small(-2, 2), coef(1, 5), opt(0, nopt - 1);
    I.sizes.assign(r, nopt);
    I.W.resize(r);
    for (int i = 0; i < r; ++i) {
        I.W[i].resize(nopt * dim);
        for (auto& v : I.W[i]) v = small(rng);
    }
    I.d0.resize(r);
    I.K.resize(r * kappa);
    for (auto& v : I.d0) v = coef(rng) * (rng() & 1 ? 1 : -1);
    for (auto& v : I.K) v = small(rng);
    std::vector<int64_t> lambda(kappa);
    for (auto& v : lambda) v = coef(rng);
    I.planted.resize(r);
    for (auto& o : I.planted) o = static_cast<int>(opt(rng));
    I.rhs.assign(dim, 0);
    for (int i = 0; i < r; ++i) {
        int64_t di = I.d0[i];
        for (int b = 0; b < kappa; ++b) di += I.K[i * kappa + b] * lambda[b];
        for (int64_t y = 0; y < dim; ++y) I.rhs[y] += di * I.W[i][I.planted[i] * dim + y];
    }
    I.W1.resize(r);
    I.W2.resize(r);
    for (int i = 0; i < r; ++i) {
        I.W1[i].resize(nopt * dim);
        I.W2[i].resize(nopt * dim);
        for (int64_t k = 0; k < nopt * dim; ++k) { I.W1[i][k] = mod(I.W[i][k], P1); I.W2[i][k] = mod(I.W[i][k], P2); }
        I.p1.push_back(I.W1[i].data());
        I.p2.push_back(I.W2[i].data());
    }
    for (int i = 0; i < r; ++i) { I.d01.push_back(mod(I.d0[i], P1)); I.d02.push_back(mod(I.d0[i], P2)); }
    for (auto v : I.K) { I.K1.push_back(mod(v, P1)); I.K2.push_back(mod(v, P2)); }
    for (auto v : I.rhs) { I.rhs1.push_back(mod(v, P1)); I.rhs2.push_back(mod(v, P2)); }
    for (int i = 0; i < r; ++i) (i < (r + 1) / 2 ? I.sideL : I.sideR).push_back(i);
    return I;
}

DenseSolveResult run(const Instance& I, int64_t max_cand = 1'000'000) {
    DenseSolveInputs in{I.r, I.p1.data(), I.p2.data(), I.sizes.data(), I.dim, I.d01.data(), I.K1.data(), I.kappa,
                        I.d02.data(), I.K2.data(), I.kappa, I.rhs1.data(), I.rhs2.data(),
                        I.sideL.data(), static_cast<int>(I.sideL.size()), I.sideR.data(),
                        static_cast<int>(I.sideR.size()), max_cand, 7};
    return dense_solve(in);
}

// Exact decision of one combination mod p: u = sum d0_i w_i - rhs lies in
// the span of v_b = sum K_ib w_i.
bool solves(const Instance& I, const std::vector<int>& combo, int64_t p) {
    std::vector<std::vector<int64_t>> V(I.kappa, std::vector<int64_t>(I.dim, 0));
    std::vector<int64_t> u(I.dim);
    for (int64_t y = 0; y < I.dim; ++y) u[y] = mod(-I.rhs[y], p);
    for (int i = 0; i < I.r; ++i)
        for (int64_t y = 0; y < I.dim; ++y) {
            const int64_t w = mod(I.W[i][combo[i] * I.dim + y], p);
            u[y] = mod(u[y] + mod(mod(I.d0[i], p) * w, p), p);
            for (int b = 0; b < I.kappa; ++b) V[b][y] = mod(V[b][y] + mod(mod(I.K[i * I.kappa + b], p) * w, p), p);
        }
    const int64_t rv = rank_mod(V, I.dim, p);
    V.push_back(u);
    return rank_mod(V, I.dim, p) == rv;
}

std::set<std::vector<int>> brute(const Instance& I) {
    std::set<std::vector<int>> out;
    std::vector<int> combo(I.r, 0);
    while (true) {
        if (solves(I, combo, P2) && solves(I, combo, P1)) out.insert(combo);
        int i = 0;
        while (i < I.r) {
            if (++combo[i] < I.sizes[i]) break;
            combo[i] = 0;
            ++i;
        }
        if (i == I.r) break;
    }
    return out;
}

std::set<std::vector<int>> as_set(const DenseSolveResult& res, int r) {
    std::set<std::vector<int>> out;
    for (int64_t k = 0; k < res.passed; ++k)
        out.insert(std::vector<int>(res.combos.begin() + k * r, res.combos.begin() + (k + 1) * r));
    return out;
}

}  // namespace

TEST_CASE("the dense family solve returns exactly the combinations that solve the slice equation, kappa = 1") {
    for (unsigned seed : {1u, 2u, 3u, 4u}) {
        Instance I = make_instance(seed, 5, 1, 6, 6);
        auto res = run(I);
        auto want = brute(I);
        CHECK(as_set(res, I.r) == want);
        CHECK(want.count(I.planted) == 1);
        CHECK(res.raw >= res.passed);
        CHECK(res.pass1 >= res.passed);
    }
}

TEST_CASE("the dense family solve with two parameters, and with a right side of one term") {
    Instance I = make_instance(11, 4, 2, 7, 5);
    auto res = run(I);
    CHECK(as_set(res, I.r) == brute(I));
    CHECK(as_set(res, I.r).count(I.planted) == 1);
    // one side holding a single term
    Instance J = make_instance(12, 3, 1, 6, 7);
    J.sideL = {0, 1};
    J.sideR = {2};
    auto res2 = run(J);
    CHECK(as_set(res2, J.r) == brute(J));
}

TEST_CASE("the dense family solve raises past the candidate cap") {
    Instance I = make_instance(21, 4, 1, 2, 6);    // dimension 2: almost every combination is a candidate
    CHECK_THROWS(run(I, 3));
}
