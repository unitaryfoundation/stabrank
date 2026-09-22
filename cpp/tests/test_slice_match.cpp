#include "stabrank/slice_match.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace stabrank;

namespace {

int popcount(int x) { return __builtin_popcount(static_cast<unsigned>(x)); }

// Codes of the stabilizer state sum_t i^{l . t} (-1)^{t^T Q t} |x0 + t G>
// on n qubits, with G a k x n generator matrix (rows as bit masks) and Q
// strictly upper triangular (bit a * k + b for a < b).
std::vector<int8_t> stabilizer_codes(int n, int k, const std::vector<int>& G, int x0,
                                     const std::vector<int>& l, uint64_t Q) {
    std::vector<int8_t> v(1 << n, 0);
    for (int t = 0; t < (1 << k); ++t) {
        int x = x0, ph = 0;
        for (int a = 0; a < k; ++a) {
            if (!((t >> a) & 1)) continue;
            x ^= G[a];
            ph += l[a];
            for (int b = a + 1; b < k; ++b)
                if (((t >> b) & 1) && ((Q >> (a * k + b)) & 1)) ph += 2;
        }
        v[x] = static_cast<int8_t>(1 + (ph & 3));
    }
    return v;
}

std::vector<int8_t> normalise(const std::vector<int8_t>& v) {
    std::vector<int8_t> out(v.size(), 0);
    int shift = -1;
    for (size_t x = 0; x < v.size(); ++x) {
        if (v[x] == 0) continue;
        if (shift < 0) shift = v[x] - 1;
        out[x] = static_cast<int8_t>(1 + (((v[x] - 1) - shift + 4) & 3));
    }
    return out;
}

bool independent(const std::vector<int>& rows) {
    std::vector<int> basis;
    for (int r : rows) {
        int x = r;
        for (int b : basis) x = std::min(x, x ^ b);
        if (x == 0) return false;
        basis.push_back(x);
    }
    return true;
}

// Every three-qubit stabilizer state, one per phase, as normalised codes.
std::vector<std::vector<int8_t>> dictionary3() {
    std::set<std::vector<int8_t>> seen;
    const int n = 3;
    for (int k = 0; k <= n; ++k) {
        std::vector<int> G(k, 1);
        while (true) {
            if (independent(G)) {
                for (int x0 = 0; x0 < (1 << n); ++x0)
                    for (int lc = 0; lc < (1 << (2 * k)); ++lc) {
                        std::vector<int> l(k);
                        for (int a = 0; a < k; ++a) l[a] = (lc >> (2 * a)) & 3;
                        for (uint64_t Q = 0; Q < (1ull << (k * k)); ++Q) {
                            bool upper = true;
                            for (int a = 0; a < k && upper; ++a)
                                for (int b = 0; b <= a; ++b)
                                    if ((Q >> (a * k + b)) & 1) { upper = false; break; }
                            if (!upper) continue;
                            seen.insert(normalise(stabilizer_codes(n, k, G, x0, l, Q)));
                        }
                    }
            }
            int a = 0;
            while (a < k) {
                if (++G[a] < (1 << n)) break;
                G[a] = 1;
                ++a;
            }
            if (a == k) break;
        }
    }
    return std::vector<std::vector<int8_t>>(seen.begin(), seen.end());
}

std::vector<int8_t> random_state(int n, std::mt19937_64& rng, int kmin) {
    std::uniform_int_distribution<int> kd(kmin, n), vd(1, (1 << n) - 1), xd(0, (1 << n) - 1), ld(0, 3);
    const int k = kd(rng);
    std::vector<int> G(k);
    do { for (auto& g : G) g = vd(rng); } while (!independent(G));
    std::vector<int> l(k);
    for (auto& v : l) v = ld(rng);
    uint64_t Q = 0;
    for (int a = 0; a < k; ++a)
        for (int b = a + 1; b < k; ++b)
            if (rng() & 1) Q |= 1ull << (a * k + b);
    return stabilizer_codes(n, k, G, xd(rng), l, Q);
}

struct Dict {
    std::vector<std::vector<int8_t>> states;
    std::map<std::vector<int8_t>, int> index;
    std::vector<int8_t> flat;
    Dict() : states(dictionary3()) {
        for (size_t i = 0; i < states.size(); ++i) {
            index[states[i]] = static_cast<int>(i);
            flat.insert(flat.end(), states[i].begin(), states[i].end());
        }
    }
};

// The slices of |H>^6 along its first three qubits over F_p.
std::vector<int64_t> h6_target(const SliceField& F) {
    std::vector<int64_t> T(64);
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            const int w = popcount(x) + popcount(y);
            T[x * 8 + y] = F.pow(F.cos, 6 - w) * F.pow(F.sin, w) % F.p;
        }
    return T;
}

}  // namespace

TEST_CASE("the three-qubit dictionary has 1080 states and the fields agree with slice_cover") {
    Dict D;
    REQUIRE(D.states.size() == 1080);
    SliceField F1(SM_P1), F2(SM_P2);
    // slice_cover.Field(65521): zeta = 2^((p - 1)/16) has order 16
    CHECK(F1.pow(F1.zeta, 8) == SM_P1 - 1);
    CHECK(F2.pow(F2.zeta, 8) == SM_P2 - 1);
    CHECK(F1.i == F1.pow(F1.zeta, 4));
    CHECK((2 * F1.cos * F1.sin % SM_P1) * (2 * F1.cos * F1.sin % SM_P1) % SM_P1 == F1.inv(2));
}

TEST_CASE("planted five-term decompositions are recovered from an all-visible base slice") {
    Dict D;
    SliceField F1(SM_P1), F2(SM_P2);
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int> cd(1, 6), xd(0, 7);
    int planted_found = 0, runs = 0;
    while (planted_found < 6) {
        const int x0 = xd(rng);
        std::vector<std::vector<int8_t>> terms;
        std::vector<int> cover, coeffs;
        std::set<int> used;
        bool ok = true;
        for (int i = 0; i < 5 && ok; ++i) {
            auto s = random_state(6, rng, 3);
            std::vector<int8_t> base(s.begin() + x0 * 8, s.begin() + x0 * 8 + 8);
            if (std::all_of(base.begin(), base.end(), [](int8_t c) { return c == 0; })) { ok = false; break; }
            auto it = D.index.find(normalise(base));
            REQUIRE(it != D.index.end());
            if (!used.insert(it->second).second) { ok = false; break; }
            cover.push_back(it->second);
            terms.push_back(s);
            coeffs.push_back(cd(rng));
        }
        if (!ok) continue;
        std::vector<int64_t> T1(64, 0), T2(64, 0);
        for (int i = 0; i < 5; ++i)
            for (int z = 0; z < 64; ++z) {
                T1[z] = (T1[z] + coeffs[i] * F1.of_code(terms[i][z])) % SM_P1;
                T2[z] = (T2[z] + coeffs[i] * F2.of_code(terms[i][z])) % SM_P2;
            }
        SliceMatchKernel K(D.flat.data(), 1080, 3, 3, T1.data(), T2.data(), 11);
        auto res = K.run(cover, x0);
        if (res.status != 0) continue;   // dependent base states: not this kernel's case
        ++runs;
        REQUIRE(res.nhits >= 1);
        std::set<std::vector<int8_t>> want;
        for (auto& t : terms) want.insert(normalise(t));
        bool found = false;
        for (int64_t h = 0; h < res.nhits; ++h) {
            std::set<std::vector<int8_t>> got;
            for (int i = 0; i < 5; ++i) {
                const int8_t* t = &res.hits[(h * 5 + i) * 64];
                got.insert(normalise(std::vector<int8_t>(t, t + 64)));
                // every hit satisfies the target equation exactly mod P2
            }
            for (int z = 0; z < 64; ++z) {
                int64_t acc = 0;
                for (int i = 0; i < 5; ++i)
                    acc = (acc + res.coeffs2[i] * F2.of_code(res.hits[(h * 5 + i) * 64 + z])) % SM_P2;
                REQUIRE(acc == T2[z]);
            }
            if (got == want) found = true;
        }
        CHECK(found);
        planted_found += found;
        REQUIRE(res.coord_solutions.size() == 3);
    }
    CHECK(runs >= 6);
}

TEST_CASE("a base point where a planted term vanishes, or a wrong base state, gives no planted hit") {
    Dict D;
    SliceField F1(SM_P1), F2(SM_P2);
    std::mt19937_64 rng(9);
    // five full-support states and a target from them; a sixth state replaces one base state
    std::vector<std::vector<int8_t>> terms;
    std::vector<int> cover;
    std::set<int> used;
    while (terms.size() < 6) {
        auto s = random_state(6, rng, 6);
        std::vector<int8_t> base(s.begin(), s.begin() + 8);
        const int idx = D.index.at(normalise(base));
        if (!used.insert(idx).second) continue;
        terms.push_back(s);
        cover.push_back(idx);
    }
    std::vector<int64_t> T1(64, 0), T2(64, 0);
    for (int i = 0; i < 5; ++i)
        for (int z = 0; z < 64; ++z) {
            T1[z] = (T1[z] + (i + 1) * F1.of_code(terms[i][z])) % SM_P1;
            T2[z] = (T2[z] + (i + 1) * F2.of_code(terms[i][z])) % SM_P2;
        }
    SliceMatchKernel K(D.flat.data(), 1080, 3, 3, T1.data(), T2.data(), 11);
    std::vector<int> good(cover.begin(), cover.begin() + 5);
    auto res = K.run(good, 0);
    REQUIRE(res.status == 0);
    CHECK(res.nhits >= 1);
    std::vector<int> wrong = {cover[5], cover[1], cover[2], cover[3], cover[4]};
    auto bad = K.run(wrong, 0);
    // the target is generically outside the span of the wrong base: refused, or no hit
    CHECK((bad.status == 1 || bad.nhits == 0));
}

TEST_CASE("the |H>^6 sample runs agree with the Python reference matcher") {
    // research/h6_rank5/driver.py fixture writes this file from the reference
    // (STABRANK_NO_NATIVE=1) run of the 160 sample (cover, x0) pairs.
    std::ifstream in(std::string(STABRANK_TEST_DATA_DIR) + "/h6_rank5_sample.txt");
    REQUIRE(in.good());
    Dict D;
    SliceField F1(SM_P1), F2(SM_P2);
    auto T1 = h6_target(F1), T2 = h6_target(F2);
    SliceMatchKernel K(D.flat.data(), 1080, 3, 3, T1.data(), T2.data(), 11);
    std::string line;
    int runs = 0, nonempty = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int x0, kappa, nhits;
        std::string u[5], sols;
        ss >> x0 >> u[0] >> u[1] >> u[2] >> u[3] >> u[4] >> kappa >> sols >> nhits;
        std::vector<int> cover;
        for (int i = 0; i < 5; ++i) {
            REQUIRE(u[i].size() == 8);
            std::vector<int8_t> c(8);
            for (int y = 0; y < 8; ++y) c[y] = static_cast<int8_t>(u[i][y] - '0');
            cover.push_back(D.index.at(c));
        }
        std::vector<int64_t> want;
        for (size_t p = 0, q; p < sols.size(); p = q + 1) {
            q = sols.find(',', p);
            if (q == std::string::npos) q = sols.size();
            want.push_back(std::stoll(sols.substr(p, q - p)));
        }
        auto res = K.run(cover, x0);
        ++runs;
        if (kappa > 0) { CHECK(res.status == 2); continue; }
        REQUIRE(res.status == 0);
        CHECK(res.coord_solutions == want);
        CHECK(res.nhits == nhits);
        nonempty += !want.empty() && want.back() > 0;
    }
    CHECK(runs == 160);
    CHECK(nonempty >= 10);
}
