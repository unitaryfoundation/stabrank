#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace stabrank {

// Compiled verify_challenge/slice_cover.CoverEnumerator.pair_covers for
// r = 5: the full 5-covers of the target containing the pivot i and the
// partner j, with the other three members above j inside the member mask,
// found as a third pivot k and a pair (a, b) whose images are parallel
// modulo span(target, u_i, u_j, u_k) over F_65521, then decided exactly:
// the span condition by ranks mod 2013265921, the fullness (some solution
// with every coefficient nonzero) mod both primes. Every array is
// row-major int64.
struct Cover5Inputs {
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
    // 1: the projective key is one random functional mod 65521 (16 bits,
    // about M^2 / (2 x 65521) accidental candidates per third pivot at M
    // members, the rate docs/notes/kernels_k6_p3.md measures); 2: two
    // functionals, a 32-bit key. The covers reported are the same either
    // way (every candidate is decided exactly); only `candidates` changes.
    int key_functionals = 1;
};

struct Cover5Found {
    std::array<int64_t, 5> idx;   // sorted
    bool full1, full2;            // fullness mod 65521 and mod 2013265921
};

struct Cover5Result {
    std::vector<Cover5Found> covers;   // the 5-sets whose span contains the target mod 2013265921
    int64_t candidates = 0;            // parallel pairs decided
    int64_t members = 0;               // M, the members with a nonzero image
};

Cover5Result cover5_pair(const Cover5Inputs& in);

}  // namespace stabrank
