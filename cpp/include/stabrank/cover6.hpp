#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace stabrank {

// The full 6-covers of a target through a pivot pair: the k = 6 analogue of
// cover5_pair (verify_challenge/slice_cover.CoverEnumerator.pair_covers for
// r = 6 is the Python reference). The 6-sets containing the pivot i and the
// partner j, with the other four members above j inside the member mask,
// are found as a third pivot k, a fourth pivot l > k, and a pair (a, b)
// above l whose images are parallel modulo span(target, u_i, u_j, u_k, u_l)
// over F_65521. The parallel test is a projective key of two random
// functionals (32 bits, so about M^2 / (2 x 65521^2) accidental candidates
// per (k, l) at M members, against M^2 / (2 x 65521) for the 16-bit key of
// cover5_pair), and every candidate is decided exactly: the span condition
// by the rank mod 2013265921, the fullness (some solution with every
// coefficient nonzero) mod both primes. Every array is row-major int64.
struct Cover6Inputs {
    const int64_t* Q;        // N x D: the dictionary reduced modulo the target, mod 65521
    int64_t N, D;
    const int64_t* U1;       // N x dim states mod 65521
    const int64_t* psi1;     // dim
    const int64_t* U2;       // N x dim states mod 2013265921
    const int64_t* psi2;     // dim
    int64_t dim;
    int64_t i, j;
    const uint8_t* members;  // N: admissible further members
    int64_t max_run;         // largest parallel class accepted (an error above)
    uint64_t seed;
};

struct Cover6Found {
    std::array<int64_t, 6> idx;   // sorted
    bool full1, full2;            // fullness mod 65521 and mod 2013265921
    int64_t rank2;                // rank of the six states mod 2013265921 (6 - kappa)
};

struct Cover6Result {
    std::vector<Cover6Found> covers;   // the 6-sets whose span contains the target mod 2013265921
    int64_t candidates = 0;            // parallel pairs decided
    int64_t members = 0;               // M, the members above j with a nonzero image
};

Cover6Result cover6_pair(const Cover6Inputs& in);

}  // namespace stabrank
