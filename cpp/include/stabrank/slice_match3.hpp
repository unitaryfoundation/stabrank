#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace stabrank {

// Compiled stage A of research/qutrit_m4_rank5/matcher.py::Matcher.run for a
// two-qutrit base slice (n_1 = 2, the nine points of F_3^2) of distinct,
// linearly independent n_2-qutrit stabilizer states, whose coefficient
// family is a point. The dictionary is given as phase codes (0 zero, 1..3 =
// 1, w, w^2 with w = exp(2 pi i / 3)) of the n_2-qutrit states, the target
// as its nine slices over F_P1 and F_P2 (row 3 x_1 + x_2 is the slice at
// the sliced qutrits' value (x_1, x_2)), and a run returns every
// decomposition of the target whose slice at the base point x_0 is the
// given cover, decided modulo 65521 and re-decided modulo 2013265921. The
// options of a term at a slice are the 3^{n_2} Pauli classes times the
// three cube roots (code 3 k + l) and absent (code 3^{n_2 + 1}); the two
// coordinate slices x_0 + e_1, x_0 + e_2 are solved by meet in the middle on
// a random functional and joined, the absence pattern fixes the shapes of
// the structure lemma (a plane with 27 quadratics, a coordinate line with 3
// phases at its third point, a point or one of the two diagonal lines), and
// the six composite points are solved one at a time over the shapes still
// alive. The caller confirms every hit in floating point; the Python
// matcher stays the reference for repeated or dependent base states.

constexpr int64_t SM3_P1 = 65521;
constexpr int64_t SM3_P2 = 2013265921;

// Q(w) reduced modulo a prime p = 1 mod 3, with the same primitive cube root
// as research/qutrit_m4_rank5/cover_census.Field3 (the (p - 1)/3-th power of
// the first g >= 2 for which it is not 1).
struct Field3 {
    int64_t p, w;
    int64_t wpow[3];
    explicit Field3(int64_t prime);
    int64_t of_code(int8_t c) const { return c == 0 ? 0 : wpow[(c - 1) % 3]; }
    int64_t pow(int64_t a, int64_t e) const;
    int64_t inv(int64_t a) const { return pow(a, p - 2); }
};

struct SliceMatch3Result {
    // 0: run complete; 1: not a cover, or a coefficient vanishes (refused);
    // 2: the base states are dependent modulo one of the primes (kappa > 0
    // or a modular rank accident), which this kernel does not handle.
    int status = 0;
    std::vector<int64_t> coord_solutions;   // cumulative, as the Python stats
    int64_t joined = 0, composite_solutions = 0, candidates = 0;
    std::vector<int64_t> coeffs2;           // the r base coefficients mod P2
    int64_t nhits = 0;
    std::vector<int8_t> hits;               // nhits x r x (9 dim) codes, term layout (3 x_1 + x_2) dim + y
};

class SliceMatch3Kernel {
public:
    // codes: N x dim (dim = 3^n2) phase codes; target1, target2: 9 x dim.
    SliceMatch3Kernel(const int8_t* codes, int64_t N, int n2,
                      const int64_t* target1, const int64_t* target2, uint64_t seed);
    ~SliceMatch3Kernel();
    SliceMatch3Kernel(const SliceMatch3Kernel&) = delete;
    SliceMatch3Kernel& operator=(const SliceMatch3Kernel&) = delete;

    // x0 = 3 x_1 + x_2 in 0..8.
    SliceMatch3Result run(const std::vector<int>& cover, int x0);

    int n2() const { return n2_; }
    int64_t N() const { return N_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n2_;
    int64_t N_;
};

}  // namespace stabrank
