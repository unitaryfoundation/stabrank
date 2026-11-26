"""Replay the five-copy Pauli-projection exclusion for the H7/H9 certificates.

Write T = (|0> + omega |1>)/sqrt(2), omega = exp(i*pi/4). The Clifford
(I-iX)/sqrt(2) sends X to X and Y to Z, so it maps the +1 eigenstate
of (X+Y)/sqrt(2) to the +1 eigenstate of the Hadamard, the H target.

All spans are tested exactly over F_65537 with i = 256. Gaussian representatives
for stabilizer rays have entries 0 or powers of i. A complex span of k<=5
such rays containing the T^5 amplitude vector also contains its two Gaussian
even/odd columns: apply the field automorphism omega -> -omega over Q(i)
to a solution over Q(omega), which exists by Gaussian elimination. A nonzero
k-column minor stays nonzero modulo the prime, since its Gaussian norm is
at most k^k <= 3125. Any true target containment therefore survives reduction.
For optional four-qubit positive controls, one added target column has entries
of modulus <=2, giving norm bound 4*(k+1)^(k+1) <=12500 for k<=4.

For each signed stabilizer of a proposed term, the opposite Pauli projection
removes that term. Four-qubit projected stabilizers or zero must then span
the projected target in <=4 terms. The first kernel exhausts these branch
tests by inspecting inverse images of points and lines in the quotient by
the two target columns. Dimension-one target planes are allowed through
the filter. Physical symmetries preserving the projected target plane
reduce anchor tests; no hash equality is accepted without full-key equality.
The next kernel retains exactly the five-qubit rays passing all opposite
branches. The last kernel exhausts rank-three quotient extensions of the
target among the survivors. A five-term solution must occupy one such
five-dimensional extension. Smaller solutions could be extended using the
retained computational basis. Each extension is checked for five independent
contained rays.

Run with Python 3, a GNU-compatible C++17 compiler, and OpenMP. The kernels
regenerate both intermediate tables in a fresh temporary directory. The
two-thread classifier may discover equivalent positive witnesses in different
orders. Its table, the final exclusion, and the output of this script are
deterministic. No downloaded tables or packages are required.
"""

import os
from pathlib import Path
import subprocess
import tempfile

DICTIONARY = r'''#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace certificate {

using Word = std::uint32_t;
using Wide = std::uint64_t;
constexpr Word prime = 65537;
constexpr std::array<Word, 4> fourth_root{1, 256, 65536, 65281};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline Word product(Word a, Word b) {
    return static_cast<Word>((Wide(a) * b) % prime);
}
inline Word difference(Word a, Word b) {
    return a >= b ? a - b : a + prime - b;
}
inline std::array<Word, prime> inverse_table() {
    std::array<Word, prime> table{};
    table[1] = 1;
    for (Word i = 2; i < prime; ++i)
        table[i] = prime - product(prime / i, table[prime % i]);
    return table;
}
inline const auto inverses = inverse_table();
inline int weight(Word x) { return __builtin_popcount(x); }
inline int parity(Word x) { return __builtin_parity(x); }
inline int first_bit(Word x) { return __builtin_ctz(x); }
inline Wide mix(Wide x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

// A nonzero amplitude has phase i^(low + 2 high). Bit j addresses coordinate j.
struct Ray {
    Word support, low, high;
    bool operator==(const Ray& other) const {
        return support == other.support && low == other.low && high == other.high;
    }
    int phase(int j) const {
        if (!(support >> j & 1u)) return -1;
        return int(low >> j & 1u) + 2 * int(high >> j & 1u);
    }
    Word amplitude(int j) const {
        int exponent = phase(j);
        return exponent < 0 ? 0 : fourth_root[exponent];
    }
};

struct RayHash {
    std::size_t operator()(const Ray& s) const {
        return mix((Wide(s.support) << 32 | s.low) ^ mix(s.high));
    }
};

inline Ray canonical(Ray s) {
    require(s.support != 0, "zero ray");
    int rotation = (-s.phase(first_bit(s.support))) & 3;
    if (rotation & 1) {
        s.high ^= s.low;
        s.low ^= s.support;
    }
    if (rotation & 2) s.high ^= s.support;
    return s;
}

inline Ray from_phases(const std::array<int, 32>& phases) {
    Ray s{};
    for (int j = 0; j < 32; ++j) {
        if (phases[j] < 0) continue;
        Word bit = Word(1) << j;
        s.support |= bit;
        if (phases[j] & 1) s.low |= bit;
        if (phases[j] & 2) s.high |= bit;
    }
    return canonical(s);
}

// Generate every affine support in F_2^n and every linear/quadratic phase on it.
// The origin of each affine chart is assigned amplitude 1.
inline std::vector<Ray> dictionary(int n) {
    const int dimension = 1 << n;
    std::vector<Word> linear_spaces{1};
    std::unordered_set<Word> seen{1};
    for (std::size_t i = 0; i < linear_spaces.size(); ++i) {
        Word space = linear_spaces[i];
        for (int x = 1; x < dimension; ++x) {
            if (space >> x & 1u) continue;
            Word enlarged = space;
            for (Word k = space; k; k &= k - 1)
                enlarged |= Word(1) << (first_bit(k) ^ x);
            if (seen.insert(enlarged).second) linear_spaces.push_back(enlarged);
        }
    }
    std::vector<Ray> result;
    for (Word space : linear_spaces) {
        std::vector<int> chart{0};
        Word spanned = 1;
        for (int x = 1; x < dimension; ++x) {
            if (!(space >> x & 1u) || (spanned >> x & 1u)) continue;
            int old_size = chart.size();
            for (int j = 0; j < old_size; ++j) {
                int y = chart[j] ^ x;
                chart.push_back(y);
                spanned |= Word(1) << y;
            }
        }
        const int linear_dimension = first_bit(chart.size());
        Word covered = 0;
        for (int origin = 0; origin < dimension; ++origin) {
            if (covered >> origin & 1u) continue;
            Word support = 0;
            std::vector<Word> linear(linear_dimension);
            for (int j = 0; j < int(chart.size()); ++j) {
                Word bit = Word(1) << (origin ^ chart[j]);
                support |= bit;
                for (int b = 0; b < linear_dimension; ++b)
                    if (j >> b & 1) linear[b] |= bit;
            }
            covered |= support;
            auto phase_generators = linear;
            for (int j = 0; j < linear_dimension; ++j)
                for (int k = 0; k < j; ++k)
                    phase_generators.push_back(linear[j] & linear[k]);
            std::vector<Word> signs(1u << phase_generators.size());
            for (unsigned j = 1; j < signs.size(); ++j)
                signs[j] = signs[j & (j - 1)] ^ phase_generators[first_bit(j)];
            for (int j = 0; j < (1 << linear_dimension); ++j)
                for (Word high : signs) result.push_back({support, signs[j], high});
        }
    }
    require(result.size() == (n == 4 ? 36720u : 2423520u), "dictionary size mismatch");
    return result;
}

using Index = std::unordered_map<Ray, int, RayHash>;
inline Index index_rays(const std::vector<Ray>& rays) {
    Index index;
    index.reserve(rays.size());
    for (int i = 0; i < int(rays.size()); ++i)
        require(index.emplace(rays[i], i).second, "duplicate stabilizer ray");
    return index;
}

template <std::size_t N>
using Vector = std::array<Word, N>;

template <std::size_t N>
Vector<N> amplitudes(const Ray& s) {
    Vector<N> v{};
    for (std::size_t j = 0; j < N; ++j) v[j] = s.amplitude(j);
    return v;
}

template <std::size_t N>
struct Basis {
    std::array<Vector<N>, N> rows{};
    std::vector<int> pivots;

    Vector<N> reduce(Vector<N> v) const {
        for (int i : pivots) {
            Word scale = v[i];
            if (!scale) continue;
            for (int j = i; j < int(N); ++j)
                v[j] = difference(v[j], product(scale, rows[i][j]));
        }
        return v;
    }
    bool insert(Vector<N> v) {
        v = reduce(v);
        int pivot = 0;
        while (pivot < int(N) && !v[pivot]) ++pivot;
        if (pivot == int(N)) return false;
        Word scale = inverses[v[pivot]];
        for (int j = pivot; j < int(N); ++j) v[j] = product(v[j], scale);
        rows[pivot] = v;
        pivots.insert(std::lower_bound(pivots.begin(), pivots.end(), pivot), pivot);
        return true;
    }
};

template <std::size_t N>
bool is_zero(const Vector<N>& v) {
    return std::all_of(v.begin(), v.end(), [](Word x) { return !x; });
}

// A physical symmetry is either a transposition, or local C operations followed
// by an optional global parity. Global phases have no effect on rays.
struct Symmetry {
    int first = -1, second = -1;
    Word local_c = 0;
    bool global_z = false;
};

inline Ray transform(const Ray& s, const Symmetry& symmetry) {
    std::array<int, 32> phases;
    phases.fill(-1);
    for (Word x = s.support; x; x &= x - 1) {
        int before = first_bit(x), after = before;
        int phase = s.phase(before);
        if (symmetry.first >= 0) {
            int a = symmetry.first, b = symmetry.second;
            if (((before >> a) ^ (before >> b)) & 1)
                after ^= (1 << a) | (1 << b);
        } else {
            after ^= symmetry.local_c;
            phase += 3 * weight(symmetry.local_c & ~Word(after));
            if (symmetry.global_z) phase += 2 * parity(before);
        }
        phases[after] = phase & 3;
    }
    return from_phases(phases);
}

template <class Apply>
std::vector<int> orbit_owners(const std::vector<Ray>& rays,
                              const std::vector<Symmetry>& generators, Apply apply) {
    auto index = index_rays(rays);
    std::vector<int> owner(rays.size(), -1);
    for (int i = 0; i < int(rays.size()); ++i) {
        if (owner[i] >= 0) continue;
        std::vector<int> queue{i};
        owner[i] = i;
        for (std::size_t k = 0; k < queue.size(); ++k) {
            for (const auto& g : generators) {
                int j = index.at(apply(rays[queue[k]], g));
                if (owner[j] < 0) {
                    owner[j] = i;
                    queue.push_back(j);
                }
            }
        }
    }
    return owner;
}
inline std::vector<int> physical_owners(const std::vector<Ray>& rays) {
    std::vector<Symmetry> generators;
    for (int i = 0; i < 4; ++i) generators.push_back({i, i + 1});
    for (int i = 0; i < 5; ++i) generators.push_back({-1, -1, Word(1) << i});
    auto result = orbit_owners(rays, generators, transform);
    int orbit_count = 0;
    for (int i = 0; i < int(result.size()); ++i) orbit_count += result[i] == i;
    require(orbit_count == 1581, "five-qubit orbit count mismatch");
    return result;
}

}  // namespace certificate
'''

CLASSIFY = r'''#include "stabilizer_dictionary.hpp"

#include <atomic>
#include <fstream>
#include <mutex>

using namespace certificate;
using Column = Vector<16>;

struct Branch {
    int equatorial, diagonal, outcome;
    Word x_mask, z_mask;
    std::array<int, 16> chart{};
    std::array<Column, 2> target{};

    Branch(int a, int b, int sign) : equatorial(a), diagonal(b), outcome(sign) {
        x_mask = (1u << a) - 1;
        z_mask = ((1u << b) - 1) << a;
        for (int y = 0; y < 16; ++y) {
            int x = 2 * y;
            if (!a) x |= parity(z_mask & x) ^ (sign < 0);
            chart[y] = x;
            for (int endpoint = 0; endpoint < (a ? 2 : 1); ++endpoint) {
                int w = weight(endpoint ? x ^ x_mask : x);
                int negative = endpoint && (parity(z_mask & x) ^ (sign < 0));
                Word value = fourth_root[(w / 2 + 2 * negative) & 3];
                auto& entry = target[w & 1][y];
                entry = (entry + value) % prime;
            }
        }
    }
    Ray embed(const Ray& s) const {
        std::array<int, 32> p;
        p.fill(-1);
        for (int y = 0; y < 16; ++y) {
            int x = chart[y], value = s.phase(y);
            p[x] = value;
            if (equatorial && value >= 0)
                p[x ^ x_mask] = (value + 2 * (parity(z_mask & x) ^ (outcome < 0))) & 3;
        }
        return from_phases(p);
    }
    Ray extract(const Ray& s) const {
        std::array<int, 32> p;
        p.fill(-1);
        for (int y = 0; y < 16; ++y) p[y] = s.phase(chart[y]);
        return from_phases(p);
    }
    std::vector<Symmetry> generators() const {
        std::vector<Symmetry> g;
        int a = equatorial, b = diagonal;
        for (int i = 0; i < 4; ++i)
            if (i + 1 != a && i + 1 != a + b) g.push_back({i, i + 1});
        for (int i = a + b; i < 5; ++i) g.push_back({-1, -1, Word(1) << i});
        for (int i = a; i + 1 < a + b; ++i) g.push_back({-1, -1, Word(3) << i});
        if (!(a & 1)) g.push_back({-1, -1, 0, true});
        else if (b) g.push_back({-1, -1, Word(1) << a, true});
        return g;
    }
};

inline int leading(const Column& column) {
    int j = 0;
    while (j < 16 && !column[j]) ++j;
    return j;
}
inline Column projective(Column x) {
    int j = leading(x);
    if (j == 16) return x;
    Word scale = inverses[x[j]];
    for (int k = j; k < 16; ++k) x[k] = product(x[k], scale);
    return x;
}
struct ColumnHash {
    std::size_t operator()(const Column& c) const {
        Wide h = 0;
        for (int j = 0; j < 16; j += 2)
            h = mix(h ^ (Wide(c[j]) << 32 | c[j + 1]));
        return h;
    }
};
using Buckets = std::unordered_map<Column, std::vector<int>, ColumnHash>;

// The caller provides all rays of a preimage of known dimension k. If their
// span reaches k the selected columns necessarily contain the target plane.
std::vector<int> witness(const std::vector<Column>& columns,
                         const std::vector<int>& common,
                         const std::vector<int>& fiber, int k) {
    if (common.size() + fiber.size() < std::size_t(k)) return {};
    Basis<16> basis;
    std::vector<int> selected;
    for (const auto* group : {&common, &fiber}) {
        for (int i : *group) {
            if (basis.insert(columns[i])) selected.push_back(i);
            if (int(selected.size()) == k) return selected;
        }
    }
    return {};
}

struct Result {
    int rank_or_lower;
    int anchors;
    std::vector<int> witness;
};

Result classify(const Branch& branch, const std::vector<Ray>& rays,
                const std::vector<Column>& columns) {
    Basis<16> target;
    target.insert(branch.target[0]);
    target.insert(branch.target[1]);
    if (target.pivots.size() == 1) return {0, 0, {}};  // conservatively unfiltered
    require(target.pivots.size() == 2, "zero or malformed projection");

    std::vector<Column> quotient;
    std::vector<int> zero;
    Buckets proportional;
    quotient.reserve(rays.size());
    for (int i = 0; i < int(rays.size()); ++i) {
        auto point = projective(target.reduce(columns[i]));
        quotient.push_back(point);
        if (is_zero(point)) zero.push_back(i);
        else proportional[point].push_back(i);
    }
    auto hit = witness(columns, {}, zero, 2);
    if (!hit.empty()) return {2, 0, hit};
    for (const auto& entry : proportional) {
        hit = witness(columns, zero, entry.second, 3);
        if (!hit.empty()) return {3, 0, hit};
    }
    auto owners = orbit_owners(rays, branch.generators(), [&](const Ray& s, const Symmetry& g) {
        return branch.extract(transform(branch.embed(s), g));
    });
    std::vector<int> anchors;
    for (int i = 0; i < int(rays.size()); ++i)
        if (owners[i] == i && !is_zero(quotient[i])) anchors.push_back(i);

    std::atomic<bool> found{false};
    std::mutex result_mutex;
    #pragma omp parallel for schedule(dynamic)
    for (int it = 0; it < int(anchors.size()); ++it) {
        if (found.load(std::memory_order_relaxed)) continue;
        int a = anchors[it], pivot = leading(quotient[a]);
        const auto& point = quotient[a];

        // Exact point coordinates are stored as the hash-table key. Equality
        // compares all 16 field elements for every hash collision.
        Buckets lines;
        lines.reserve(rays.size());
        auto common = zero;
        for (int i = 0; i < int(rays.size()); ++i) {
            if (is_zero(quotient[i])) continue;
            auto residual = quotient[i];
            Word scale = residual[pivot];
            if (scale) {
                for (int j = pivot; j < 16; ++j)
                    residual[j] = difference(residual[j], product(scale, point[j]));
            }
            int h = leading(residual);
            if (h == 16) common.push_back(i);
            else lines[projective(residual)].push_back(i);
        }
        for (const auto& entry : lines) {
            auto candidate = witness(columns, common, entry.second, 4);
            if (candidate.empty()) continue;
            std::lock_guard<std::mutex> lock(result_mutex);
            if (!found.load()) {
                hit = std::move(candidate);
                found.store(true);
            }
            break;
        }
    }
    return {found.load() ? 4 : 5, int(anchors.size()), hit};
}

int main() try {
    const auto rays = dictionary(4);
    std::vector<Column> columns;
    for (const auto& s : rays) columns.push_back(amplitudes<16>(s));
    std::ofstream table("branch_table.txt");
    require(bool(table), "could not create branch_table.txt");
    int count = 0, excluded = 0;
    for (int a = 0; a <= 5; ++a) {
        for (int b = 0; a + b <= 5; ++b) {
            if (!a && !b) continue;
            for (int sign : {1, -1}) {
                if (b && sign < 0) continue;
                Branch branch(a, b, sign);
                auto result = classify(branch, rays, columns);
                ++count;
                excluded += result.rank_or_lower == 5;
                table << a << ' ' << b << ' ' << sign << ' ' << result.rank_or_lower << '\n';
                std::cout << "a=" << a << " b=" << b << " outcome=" << sign
                          << " rank_or_lower=" << result.rank_or_lower
                          << " anchors=" << result.anchors << " witness";
                for (int id : result.witness) std::cout << ' ' << id;
                std::cout << std::endl;
                if ((!a && b == 1) || (a == 1 && !b))
                    require(result.rank_or_lower == 4, "single-site positive control failed");
                if (!a && b == 2)
                    require(result.rank_or_lower == 3, "two-site positive control failed");
            }
        }
    }
    require(count == 25, "incomplete Pauli branch enumeration");
    require(excluded == 9, "unexpected number of excluded branches");
    require(bool(table), "failed to write branch_table.txt");
    std::cout << "Classified " << count << " branches. Excluded " << excluded << ".\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Branch search failed: " << error.what() << '\n';
    return 1;
}
'''

FILTER = r'''#include "stabilizer_dictionary.hpp"

#include <fstream>
#include <set>

using namespace certificate;

struct SignedPauli {
    Word x, z;
    int negative;
};

std::vector<SignedPauli> stabilizers_of(const Ray& s) {
    std::array<int, 32> phase;
    for (int j = 0; j < 32; ++j) phase[j] = s.phase(j);
    std::vector<SignedPauli> result;
    for (Word x = 0; x < 32; ++x) {
        for (Word z = 0; z < 32; ++z) {
            if (!x && !z) continue;
            int eigenphase = -1;
            bool stabilizes = true;
            for (Word sites = s.support; sites; sites &= sites - 1) {
                int y = first_bit(sites);
                if (phase[y ^ x] < 0) {
                    stabilizes = false;
                    break;
                }
                // Hermitian P(x,z)=i^|x&z| X^x Z^z.
                int value = (phase[y] + weight(x & z) + 2 * parity(y & z)
                             - phase[y ^ x]) & 3;
                if ((value & 1) || (eigenphase >= 0 && value != eigenphase)) {
                    stabilizes = false;
                    break;
                }
                eigenphase = value;
            }
            if (stabilizes) result.push_back({x, z, eigenphase / 2});
        }
    }
    require(result.size() == 31, "wrong signed stabilizer group order");
    return result;
}

struct BranchTable {
    int rank[6][6][2]{};
    BranchTable() {
        std::ifstream in("branch_table.txt");
        require(bool(in), "missing branch_table.txt; run classify_branches first");
        std::set<std::array<int, 3>> seen;
        for (int a = 0; a <= 5; ++a)
            for (int b = 0; b <= 5; ++b)
                for (int sign = 0; sign <= 1; ++sign) rank[a][b][sign] = -1;
        int a, b, sign, value;
        while (in >> a >> b >> sign >> value) {
            require(a >= 0 && b >= 0 && a + b >= 1 && a + b <= 5, "invalid Pauli weights");
            require((sign == 1 || sign == -1) && !(b && sign < 0), "invalid branch sign");
            require(value == 0 || (value >= 2 && value <= 5), "unexpected branch result");
            require(seen.insert({a, b, sign}).second, "duplicate branch");
            require((value == 0) == (a == 0 && b == 5), "unexpected degenerate plane");
            rank[a][b][sign < 0] = value;
            if (b) rank[a][b][1] = value;
        }
        require(in.eof() && seen.size() == 25, "incomplete or malformed branch table");
        for (int a = 0; a <= 5; ++a)
            for (int b = 0; a + b <= 5; ++b)
                if (a + b)
                    for (int negative = 0; negative <= 1; ++negative)
                        require(rank[a][b][negative] >= 0, "missing branch");
    }
    int opposite(const SignedPauli& p) const {
        return rank[weight(p.x)][weight(p.z & ~p.x)][1 - p.negative];
    }
};

int main() try {
    BranchTable table;
    const auto rays = dictionary(5);
    const auto owner = physical_owners(rays);
    std::vector<bool> allowed(rays.size());
    int orbit_count = 0;
    for (int i = 0; i < int(rays.size()); ++i) {
        if (owner[i] != i) continue;
        auto group = stabilizers_of(rays[i]);
        bool keep = std::all_of(group.begin(), group.end(), [&](const SignedPauli& p) {
            return table.opposite(p) < 5;
        });
        allowed[i] = keep;
        orbit_count += keep;
    }
    std::ofstream out("vocabulary.txt");
    require(bool(out), "could not create vocabulary.txt");
    int total = 0, computational = 0;
    for (int i = 0; i < int(rays.size()); ++i) {
        if (!allowed[owner[i]]) continue;
        out << i << ' ' << owner[i] << '\n';
        ++total;
        computational += weight(rays[i].support) == 1;
    }
    require(bool(out), "failed to write vocabulary.txt");
    require(total == 224 && orbit_count == 7 && computational == 32, "vocabulary count mismatch");
    std::cout << "All 2,423,520 rays form 1,581 orbits. Retained " << total
              << " rays in " << orbit_count << " orbits, including all " << computational
              << " computational basis rays.\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Vocabulary filter failed: " << error.what() << '\n';
    return 1;
}
'''

EXCLUDE = r'''#include "stabilizer_dictionary.hpp"

#include <fstream>
#include <set>

using namespace certificate;
using Column = Vector<32>;

int main() try {
    std::ifstream in("vocabulary.txt");
    require(bool(in), "missing vocabulary.txt; run filter_vocabulary first");
    const auto rays = dictionary(5);
    const auto owner = physical_owners(rays);
    std::vector<int> ids, anchors;
    std::set<int> unique, representatives;
    int id, claimed_owner;
    while (in >> id >> claimed_owner) {
        require(id >= 0 && id < int(rays.size()) && owner[id] == claimed_owner,
                "invalid vocabulary index or owner");
        require(unique.insert(id).second, "duplicate retained ray");
        representatives.insert(claimed_owner);
        if (id == claimed_owner) anchors.push_back(ids.size());
        ids.push_back(id);
    }
    require(in.eof() && ids.size() == 224 && anchors.size() == 7 && representatives.size() == 7,
            "incomplete or malformed vocabulary");
    for (int i = 0; i < int(rays.size()); ++i)
        require(bool(representatives.count(owner[i])) == bool(unique.count(i)),
                "a retained orbit is incomplete");
    std::vector<Column> columns;
    int computational = 0;
    for (int i : ids) {
        columns.push_back(amplitudes<32>(rays[i]));
        computational += weight(rays[i].support) == 1;
    }
    require(computational == 32, "the retained rays must span at least five dimensions");

    std::array<Column, 2> conjugates{};
    for (Word y = 0; y < 32; ++y) {
        int w = weight(y);
        conjugates[w & 1u][y] = fourth_root[w / 2];
    }
    Basis<32> target;
    target.insert(conjugates[0]);
    target.insert(conjugates[1]);
    require(target.pivots.size() == 2, "target plane dimension mismatch");

    int tests = 0, maximum = 0;
    for (int a : anchors) {
        auto first = target;
        if (!first.insert(columns[a])) continue;
        for (int b = 0; b < int(columns.size()); ++b) {
            if (b == a) continue;
            auto second = first;
            if (!second.insert(columns[b])) continue;
            for (int c = b + 1; c < int(columns.size()); ++c) {
                if (c == a) continue;
                auto five_space = second;
                if (!five_space.insert(columns[c])) continue;
                ++tests;
                Basis<32> contained;
                for (const auto& v : columns)
                    if (is_zero(five_space.reduce(v))) contained.insert(v);
                maximum = std::max(maximum, int(contained.pivots.size()));
                require(maximum < 5, "five independent atoms found in a candidate space");
            }
        }
        std::cout << "Anchor " << ids[a] << " cumulative tests " << tests
                  << " maximum contained rank " << maximum << '\n';
    }
    require(tests == 173223 && maximum == 4, "flat count or maximum changed");
    std::cout << "Certified: no five-term stabilizer decomposition of T^5. "
              << tests << " tested spaces; maximum contained rank " << maximum << ".\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Five-atom test failed: " << error.what() << '\n';
    return 1;
}
'''


KERNELS = (
    ("classify_branches", CLASSIFY, "Classified 25 branches. Excluded 9."),
    ("filter_vocabulary", FILTER,
     "All 2,423,520 rays form 1,581 orbits. Retained 224 rays in 7 orbits, "
     "including all 32 computational basis rays."),
    ("exclude_five", EXCLUDE,
     "Certified: no five-term stabilizer decomposition of T^5. "
     "173223 tested spaces; maximum contained rank 4."),
)


def main():
    compiler = os.environ.get("CXX", "g++")
    with tempfile.TemporaryDirectory(prefix="h5-gaussian-") as directory:
        build = Path(directory)
        env = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE",
                   TMPDIR=directory)
        (build / "stabilizer_dictionary.hpp").write_text(DICTIONARY, encoding="utf-8")
        for name, source, expected in KERNELS:
            path = build / (name + ".cpp")
            path.write_text(source, encoding="utf-8")
            flags = ["-std=c++17", "-O3"]
            if name == "classify_branches":
                flags.append("-fopenmp")
            executable = build / name
            subprocess.run([compiler, *flags, str(path), "-o", str(executable)],
                           cwd=build, env=env, check=True)
            result = subprocess.run([str(executable)], cwd=build, env=env,
                                    stdout=subprocess.PIPE, text=True, check=True)
            if expected not in result.stdout.splitlines():
                raise RuntimeError(f"{name} did not report its exact verification counts")
            print(expected, flush=True)
    print("CERTIFIED chi(H_5) >= 6", flush=True)


if __name__ == "__main__":
    main()
