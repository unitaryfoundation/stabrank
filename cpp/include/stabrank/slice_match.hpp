#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace stabrank {

// Compiled stage A of verify_challenge/slice_cover.py::SliceMatcher.run for
// a base slice of distinct, linearly independent states (coefficient family
// a point). The dictionary is given as phase codes (0 zero, 1..4 = 1, i, -1,
// -i) of the n2-qubit stabilizer states, the target as its 2^n1 slices over
// F_P1 and F_P2, and a run returns every decomposition of the target with
// the given base slice at the given base point, decided modulo 65521 and
// re-decided modulo 2013265921. The caller confirms each hit in floating
// point (slice_cover.SliceMatcher.confirm); the Python matcher stays the
// reference for repeated or dependent base states.

constexpr int64_t SM_P1 = 65521;
constexpr int64_t SM_P2 = 2013265921;

// Q(zeta_16) reduced modulo a prime p = 1 mod 16, with the same choice of
// root as slice_cover.Field (the first g >= 2 whose (p - 1)/16-th power has
// order 16).
struct SliceField {
    int64_t p, zeta, i, cos, sin, tan;
    int64_t ipow[4];
    explicit SliceField(int64_t prime);
    int64_t of_code(int8_t c) const { return c == 0 ? 0 : ipow[(c - 1) & 3]; }
    int64_t pow(int64_t a, int64_t e) const;
    int64_t inv(int64_t a) const { return pow(a, p - 2); }
};

struct SliceMatchResult {
    // 0: run complete; 1: not a cover, or a coefficient vanishes (refused);
    // 2: the base states are dependent modulo one of the primes (kappa > 0
    // or a modular rank accident), which this kernel does not handle.
    int status = 0;
    std::vector<int64_t> coord_solutions;   // cumulative, as the Python stats
    int64_t joined = 0, types = 0, composite_solutions = 0, candidates = 0;
    std::vector<int64_t> coeffs2;           // the r base coefficients mod P2
    int64_t nhits = 0;
    std::vector<int8_t> hits;               // nhits x r x (2^n1 dim) codes, term layout x * dim + y
};

class SliceMatchKernel {
public:
    // codes: N x dim (dim = 2^n2) phase codes; target1, target2: 2^n1 x dim
    // residues of the target's slices (row x is the slice at the sliced
    // qubits' value x).
    SliceMatchKernel(const int8_t* codes, int64_t N, int n2, int n1,
                     const int64_t* target1, const int64_t* target2, uint64_t seed);
    ~SliceMatchKernel();
    SliceMatchKernel(const SliceMatchKernel&) = delete;
    SliceMatchKernel& operator=(const SliceMatchKernel&) = delete;

    SliceMatchResult run(const std::vector<int>& cover, int x0);

    int n1() const { return n1_; }
    int n2() const { return n2_; }
    int64_t N() const { return N_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n1_, n2_;
    int64_t N_;
};

}  // namespace stabrank
