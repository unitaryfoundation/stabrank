#include "stabrank/slice_match3.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace stabrank;

namespace {

int64_t mod(int64_t x, int64_t p) { x %= p; return x < 0 ? x + p : x; }

// Codes (0 zero, 1..3 = 1, w, w^2) of the qutrit stabilizer state
// sum_t w^{t^T Q t + l . t} |x0 + t G> on n qutrits, G a k x n generator
// matrix over F_3 (rows as digit vectors, most significant qutrit first),
// Q upper triangular with diagonal, l in F_3^k.
std::vector<int8_t> stabilizer_codes3(int n, int k, const std::vector<std::vector<int>>& G,
                                      const std::vector<int>& x0, const std::vector<std::vector<int>>& Q,
                                      const std::vector<int>& l) {
    int dim = 1;
    for (int q = 0; q < n; ++q) dim *= 3;
    std::vector<int8_t> v(dim, 0);
    int total = 1;
    for (int a = 0; a < k; ++a) total *= 3;
    std::vector<int> t(k);
    for (int ti = 0; ti < total; ++ti) {
        int rem = ti;
        for (int a = 0; a < k; ++a) { t[a] = rem % 3; rem /= 3; }
        int idx = 0, ph = 0;
        for (int q = 0; q < n; ++q) {
            int d = x0[q];
            for (int a = 0; a < k; ++a) d += t[a] * G[a][q];
            idx = 3 * idx + d % 3;
        }
        for (int a = 0; a < k; ++a) {
            ph += l[a] * t[a];
            for (int b = a; b < k; ++b) ph += Q[a][b] * t[a] * t[b];
        }
        v[idx] = static_cast<int8_t>(1 + ph % 3);
    }
    return v;
}

std::vector<int8_t> normalise3(const std::vector<int8_t>& v) {
    std::vector<int8_t> out(v.size(), 0);
    int shift = -1;
    for (size_t x = 0; x < v.size(); ++x) {
        if (v[x] == 0) continue;
        if (shift < 0) shift = v[x] - 1;
        out[x] = static_cast<int8_t>(1 + (((v[x] - 1) - shift + 3) % 3));
    }
    return out;
}

// Rank over F_3 of the rows.
bool independent3(const std::vector<std::vector<int>>& rows, int n) {
    std::vector<std::vector<int>> M = rows;
    int rank = 0;
    for (int c = 0; c < n && rank < static_cast<int>(M.size()); ++c) {
        int pr = -1;
        for (int t = rank; t < static_cast<int>(M.size()); ++t) if (M[t][c] % 3 != 0) { pr = t; break; }
        if (pr < 0) continue;
        std::swap(M[rank], M[pr]);
        const int inv = M[rank][c] % 3 == 1 ? 1 : 2;
        for (int d = 0; d < n; ++d) M[rank][d] = (M[rank][d] * inv) % 3;
        for (int t = 0; t < static_cast<int>(M.size()); ++t) {
            if (t == rank) continue;
            const int g = M[t][c] % 3;
            for (int d = 0; d < n; ++d) M[t][d] = ((M[t][d] - g * M[rank][d]) % 3 + 3) % 3;
        }
        ++rank;
    }
    return rank == static_cast<int>(rows.size());
}

// Every n-qutrit stabilizer state, one per phase, as normalised codes.
std::vector<std::vector<int8_t>> dictionary3(int n) {
    std::set<std::vector<int8_t>> seen;
    int dim = 1;
    for (int q = 0; q < n; ++q) dim *= 3;
    for (int k = 0; k <= n; ++k) {
        int nG = 1;
        for (int a = 0; a < k * n; ++a) nG *= 3;
        for (int gi = 0; gi < nG; ++gi) {
            std::vector<std::vector<int>> G(k, std::vector<int>(n));
            int rem = gi;
            for (int a = 0; a < k; ++a) for (int q = 0; q < n; ++q) { G[a][q] = rem % 3; rem /= 3; }
            if (!independent3(G, n)) continue;
            int nQ = 1, nl = 1;
            for (int a = 0; a < k * (k + 1) / 2; ++a) nQ *= 3;
            for (int a = 0; a < k; ++a) nl *= 3;
            for (int x = 0; x < dim; ++x) {
                std::vector<int> x0(n);
                int rx = x;
                for (int q = n - 1; q >= 0; --q) { x0[q] = rx % 3; rx /= 3; }
                for (int qi = 0; qi < nQ; ++qi) {
                    std::vector<std::vector<int>> Q(k, std::vector<int>(k, 0));
                    int rq = qi;
                    for (int a = 0; a < k; ++a) for (int b = a; b < k; ++b) { Q[a][b] = rq % 3; rq /= 3; }
                    for (int li = 0; li < nl; ++li) {
                        std::vector<int> l(k);
                        int rl = li;
                        for (int a = 0; a < k; ++a) { l[a] = rl % 3; rl /= 3; }
                        seen.insert(normalise3(stabilizer_codes3(n, k, G, x0, Q, l)));
                    }
                }
            }
        }
    }
    return std::vector<std::vector<int8_t>>(seen.begin(), seen.end());
}

std::vector<int8_t> random_state3(int n, std::mt19937_64& rng, int kmin) {
    std::uniform_int_distribution<int> kd(kmin, n), td(0, 2);
    const int k = kd(rng);
    std::vector<std::vector<int>> G(k, std::vector<int>(n));
    do { for (auto& row : G) for (auto& g : row) g = td(rng); } while (!independent3(G, n));
    std::vector<int> x0(n), l(k);
    for (auto& v : x0) v = td(rng);
    for (auto& v : l) v = td(rng);
    std::vector<std::vector<int>> Q(k, std::vector<int>(k, 0));
    for (int a = 0; a < k; ++a) for (int b = a; b < k; ++b) Q[a][b] = td(rng);
    return stabilizer_codes3(n, k, G, x0, Q, l);
}

struct Dict {
    std::vector<std::vector<int8_t>> states;
    std::map<std::vector<int8_t>, int> index;
    std::vector<int8_t> flat;
    Dict() : states(dictionary3(2)) {
        for (size_t i = 0; i < states.size(); ++i) {
            index[states[i]] = static_cast<int>(i);
            flat.insert(flat.end(), states[i].begin(), states[i].end());
        }
    }
};

}  // namespace

TEST_CASE("the two-qutrit dictionary has 360 states and the field has a primitive cube root") {
    Dict D;
    REQUIRE(D.states.size() == 360);
    Field3 F1(SM3_P1), F2(SM3_P2);
    CHECK(F1.w != 1);
    CHECK(F1.pow(F1.w, 3) == 1);
    CHECK(F2.w != 1);
    CHECK(F2.pow(F2.w, 3) == 1);
    CHECK(mod(F1.w * F1.w + F1.w + 1, SM3_P1) == 0);
}

TEST_CASE("planted five-term and six-term qutrit decompositions are recovered from an all-visible base slice") {
    Dict D;
    Field3 F1(SM3_P1), F2(SM3_P2);
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int> cd(1, 6), xd(0, 8);
    for (int r : {5, 6}) {
        int planted_found = 0, runs = 0;
        while (planted_found < 5) {
            const int x0 = xd(rng);
            std::vector<std::vector<int8_t>> terms;
            std::vector<int> cover, coeffs;
            std::set<int> used;
            bool ok = true;
            for (int i = 0; i < r && ok; ++i) {
                auto s = random_state3(4, rng, 2);
                std::vector<int8_t> base(s.begin() + x0 * 9, s.begin() + x0 * 9 + 9);
                if (std::all_of(base.begin(), base.end(), [](int8_t c) { return c == 0; })) { ok = false; break; }
                auto it = D.index.find(normalise3(base));
                REQUIRE(it != D.index.end());
                if (!used.insert(it->second).second) { ok = false; break; }
                cover.push_back(it->second);
                terms.push_back(s);
                coeffs.push_back(cd(rng));
            }
            if (!ok) continue;
            std::vector<int64_t> T1(81, 0), T2(81, 0);
            for (int i = 0; i < r; ++i)
                for (int z = 0; z < 81; ++z) {
                    T1[z] = (T1[z] + coeffs[i] * F1.of_code(terms[i][z])) % SM3_P1;
                    T2[z] = (T2[z] + coeffs[i] * F2.of_code(terms[i][z])) % SM3_P2;
                }
            SliceMatch3Kernel K(D.flat.data(), 360, 2, T1.data(), T2.data(), 11);
            auto res = K.run(cover, x0);
            if (res.status != 0) continue;   // dependent base states: not this kernel's case
            ++runs;
            REQUIRE(res.nhits >= 1);
            std::set<std::vector<int8_t>> want;
            for (auto& t : terms) want.insert(normalise3(t));
            bool found = false;
            for (int64_t h = 0; h < res.nhits; ++h) {
                std::set<std::vector<int8_t>> got;
                for (int i = 0; i < r; ++i) {
                    const int8_t* t = &res.hits[(h * r + i) * 81];
                    got.insert(normalise3(std::vector<int8_t>(t, t + 81)));
                }
                for (int z = 0; z < 81; ++z) {
                    int64_t acc = 0;
                    for (int i = 0; i < r; ++i)
                        acc = (acc + res.coeffs2[i] * F2.of_code(res.hits[(h * r + i) * 81 + z])) % SM3_P2;
                    REQUIRE(acc == T2[z]);
                }
                if (got == want) found = true;
            }
            CHECK(found);
            planted_found += found;
            REQUIRE(res.coord_solutions.size() == 2);
        }
        CHECK(runs >= 5);
    }
}

TEST_CASE("a wrong qutrit base state gives no planted hit, and a repeated state is rejected") {
    Dict D;
    Field3 F1(SM3_P1), F2(SM3_P2);
    std::mt19937_64 rng(9);
    std::vector<std::vector<int8_t>> terms;
    std::vector<int> cover;
    std::set<int> used;
    while (terms.size() < 6) {
        auto s = random_state3(4, rng, 4);   // full support: every slice nonzero
        std::vector<int8_t> base(s.begin(), s.begin() + 9);
        const int idx = D.index.at(normalise3(base));
        if (!used.insert(idx).second) continue;
        terms.push_back(s);
        cover.push_back(idx);
    }
    std::vector<int64_t> T1(81, 0), T2(81, 0);
    for (int i = 0; i < 5; ++i)
        for (int z = 0; z < 81; ++z) {
            T1[z] = (T1[z] + (i + 1) * F1.of_code(terms[i][z])) % SM3_P1;
            T2[z] = (T2[z] + (i + 1) * F2.of_code(terms[i][z])) % SM3_P2;
        }
    SliceMatch3Kernel K(D.flat.data(), 360, 2, T1.data(), T2.data(), 11);
    std::vector<int> good(cover.begin(), cover.begin() + 5);
    auto res = K.run(good, 0);
    if (res.status == 0) CHECK(res.nhits >= 1);
    std::vector<int> wrong = {cover[5], cover[1], cover[2], cover[3], cover[4]};
    auto bad = K.run(wrong, 0);
    CHECK((bad.status != 0 || bad.nhits == 0));
    std::vector<int> rep = {cover[0], cover[0], cover[2], cover[3], cover[4]};
    CHECK_THROWS(K.run(rep, 0));
}
