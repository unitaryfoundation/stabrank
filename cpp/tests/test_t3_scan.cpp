#include "stabrank/t3_scan.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

using namespace stabrank;

namespace {

constexpr int64_t ELL = 65521;
constexpr int64_t ELL2 = 2147483647;

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

// Row-major matrix mod p with reduced row echelon form; returns the pivot columns.
std::vector<int64_t> rref(std::vector<int64_t>& M, int64_t rows, int64_t cols, int64_t p) {
    std::vector<int64_t> piv;
    int64_t r = 0;
    for (int64_t c = 0; c < cols && r < rows; ++c) {
        int64_t pr = -1;
        for (int64_t t = r; t < rows; ++t) if (M[t * cols + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        for (int64_t d = 0; d < cols; ++d) std::swap(M[r * cols + d], M[pr * cols + d]);
        const int64_t f = pow_mod(M[r * cols + c], p - 2, p);
        for (int64_t d = 0; d < cols; ++d) M[r * cols + d] = mod(M[r * cols + d] * f, p);
        for (int64_t t = 0; t < rows; ++t) {
            if (t == r || M[t * cols + c] == 0) continue;
            const int64_t g = M[t * cols + c];
            for (int64_t d = 0; d < cols; ++d) M[t * cols + d] = mod(M[t * cols + d] - g * M[r * cols + d], p);
        }
        piv.push_back(c);
        ++r;
    }
    return piv;
}

struct Instance {
    int64_t N, dim, D;
    std::vector<int64_t> E, E2, T2, PD, inv;
    std::vector<int8_t> isfree;
};

// N random rows with small integer entries, a target of three rows inside the
// span of the seven planted rows (or of no rows at all when plant is false),
// and the quotient images projected to F_ell^6 exactly as
// cert_t3m3_rank7.quotient_projection does.
Instance make_instance(unsigned seed, bool plant, const std::vector<int64_t>& planted) {
    Instance I;
    I.N = 40; I.dim = 12; I.D = 6;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> small(0, 5), big(0, ELL - 1), coef(1, 30);
    I.E.resize(I.N * I.dim);
    for (auto& x : I.E) x = small(rng);
    std::vector<int64_t> T(3 * I.dim, 0);
    if (plant) {
        for (int64_t r = 0; r < 3; ++r)
            for (int64_t s : planted) {
                const int64_t c = coef(rng);
                for (int64_t d = 0; d < I.dim; ++d) T[r * I.dim + d] += c * I.E[s * I.dim + d];
            }
    } else {
        for (auto& x : T) x = small(rng) * 7 + 1;
    }
    I.E2.resize(I.N * I.dim);
    for (int64_t k = 0; k < I.N * I.dim; ++k) I.E2[k] = mod(I.E[k], ELL2);
    I.T2.resize(3 * I.dim);
    for (int64_t k = 0; k < 3 * I.dim; ++k) I.T2[k] = mod(T[k], ELL2);
    // quotient mod ell: reduce the rows of E against the RREF of T, drop the pivot columns
    std::vector<int64_t> Tl(3 * I.dim);
    for (int64_t k = 0; k < 3 * I.dim; ++k) Tl[k] = mod(T[k], ELL);
    auto piv = rref(Tl, 3, I.dim, ELL);
    REQUIRE(piv.size() == 3);
    std::vector<int64_t> Vq(I.N * I.dim);
    for (int64_t k = 0; k < I.N * I.dim; ++k) Vq[k] = mod(I.E[k], ELL);
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t s = 0; s < I.N; ++s) {
            const int64_t c = Vq[s * I.dim + piv[r]];
            if (c == 0) continue;
            for (int64_t d = 0; d < I.dim; ++d) Vq[s * I.dim + d] = mod(Vq[s * I.dim + d] - c * Tl[r * I.dim + d], ELL);
        }
    std::vector<int64_t> keep;
    for (int64_t d = 0; d < I.dim; ++d)
        if (d != piv[0] && d != piv[1] && d != piv[2]) keep.push_back(d);
    std::vector<int64_t> R(I.D * keep.size());
    for (auto& x : R) x = big(rng);
    I.PD.assign(I.N * I.D, 0);
    for (int64_t s = 0; s < I.N; ++s)
        for (int64_t a = 0; a < I.D; ++a) {
            int64_t acc = 0;
            for (size_t t = 0; t < keep.size(); ++t) acc = mod(acc + Vq[s * I.dim + keep[t]] * R[a * keep.size() + t], ELL);
            I.PD[s * I.D + a] = acc;
        }
    I.isfree.assign(I.N, 0);
    for (int64_t s = 0; s < I.N; ++s) {
        bool z = true;
        for (int64_t a = 0; a < I.D; ++a) if (I.PD[s * I.D + a] != 0) z = false;
        if (z) I.isfree[s] = 1;
    }
    I.inv.assign(ELL, 0);
    for (int64_t x = 1; x < ELL; ++x) I.inv[x] = pow_mod(x, ELL - 2, ELL);
    return I;
}

struct Buffers {
    static constexpr int64_t HIST = 64, MAXF = 64, MAXS = 64, MAXO = 8, MAXM = 96;
    std::vector<int64_t> hist, found_buf, found_len, found_meta, spur_buf, spur_len, spur_meta,
        over_meta, counters;
    Buffers()
        : hist(HIST * HIST, 0), found_buf(MAXF * MAXM, 0), found_len(MAXF, 0), found_meta(MAXF * 3, 0),
          spur_buf(MAXS * MAXM, 0), spur_len(MAXS, 0), spur_meta(MAXS * 3, 0), over_meta(MAXO * 3, 0),
          counters(10, 0) {}
    T3ScanOutputs out() {
        return T3ScanOutputs{hist.data(), HIST, found_buf.data(), found_len.data(), found_meta.data(),
                             MAXF, spur_buf.data(), spur_len.data(), spur_meta.data(), MAXS,
                             over_meta.data(), MAXO, MAXM, counters.data()};
    }
};

T3ScanInputs inputs(const Instance& I, int64_t i, const std::vector<int64_t>& jlist,
                    const std::vector<int64_t>& kok_index, const std::vector<int8_t>& kok_rows,
                    int64_t need) {
    return T3ScanInputs{I.PD.data(), I.N, I.D, I.inv.data(), ELL, I.E2.data(), I.dim, I.T2.data(), 3,
                        ELL2, i, jlist.data(), static_cast<int64_t>(jlist.size()), kok_index.data(),
                        kok_rows.data(), I.isfree.data(), need};
}

}  // namespace

TEST_CASE("the scan finds a planted rank-7 configuration and decides it exactly") {
    const std::vector<int64_t> plant{4, 9, 13, 20, 27, 31, 38};
    Instance I = make_instance(11, true, plant);
    for (auto f : I.isfree) REQUIRE(f == 0);   // generic rows: nothing inside the target space
    Buffers B;
    std::vector<int64_t> jlist{9}, kok_index{-1};
    std::vector<int8_t> kok_rows(I.N, 0);
    t3_scan_pairs(inputs(I, 4, jlist, kok_index, kok_rows, 4), B.out());
    // nominal steps: sum over k > 9 of N - k - 1 = C(30, 2)
    CHECK(B.counters[0] == 435);
    CHECK(B.counters[9] == 1);
    REQUIRE(B.counters[4] >= 1);                 // at least one class contains the target
    bool covers = false;
    for (int64_t o = 0; o < B.counters[4] && o < Buffers::MAXF; ++o) {
        std::vector<int64_t> mem(B.found_buf.begin() + o * Buffers::MAXM,
                                 B.found_buf.begin() + o * Buffers::MAXM + B.found_len[o]);
        bool all = true;
        for (int64_t s : plant) {
            bool in = false;
            for (int64_t x : mem) if (x == s) in = true;
            all = all && in;
        }
        if (all) { covers = true; CHECK(B.found_meta[o * 3 + 2] == 7); }
    }
    CHECK(covers);
    CHECK(B.counters[5] == 0);                   // no spurious class
    CHECK(B.counters[6] == 0);
    // the members of every class lie in a space of dimension at most 7
    CHECK(B.counters[2] == B.counters[3]);
}

TEST_CASE("a generic target is in no class span and the third-pivot mask restricts k") {
    Instance I = make_instance(5, false, {});
    Buffers B;
    std::vector<int64_t> jlist{2, 7}, kok_index{-1, 0};
    std::vector<int8_t> kok_rows(I.N, 0);
    for (int64_t k = 8; k < I.N; k += 3) kok_rows[k] = 1;      // admissible k for j = 7
    t3_scan_pairs(inputs(I, 0, jlist, kok_index, kok_rows, 4), B.out());
    int64_t expect = 0;
    for (int64_t k = 3; k < I.N; ++k) expect += I.N - k - 1;
    for (int64_t k = 8; k < I.N; k += 3) expect += I.N - k - 1;
    CHECK(B.counters[0] == expect);
    CHECK(B.counters[1] <= expect);
    CHECK(B.counters[9] == 2);
    CHECK(B.counters[4] == 0);
    CHECK(B.counters[2] == B.counters[3] + B.counters[5] + B.counters[6]);
}

TEST_CASE("the scan is deterministic") {
    const std::vector<int64_t> plant{1, 3, 6, 10, 15, 21, 28};
    Instance I = make_instance(3, true, plant);
    Buffers B1, B2;
    std::vector<int64_t> jlist{3, 5, 8}, kok_index{-1, -1, -1};
    std::vector<int8_t> kok_rows(I.N, 0);
    t3_scan_pairs(inputs(I, 1, jlist, kok_index, kok_rows, 4), B1.out());
    t3_scan_pairs(inputs(I, 1, jlist, kok_index, kok_rows, 4), B2.out());
    CHECK(B1.counters == B2.counters);
    CHECK(B1.hist == B2.hist);
    CHECK(B1.found_buf == B2.found_buf);
}
