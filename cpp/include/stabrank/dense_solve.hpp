#pragma once

#include <cstdint>
#include <vector>

namespace stabrank {

// Compiled verify_challenge/slice_cover._dense with the whole-vector
// re-decision folded in: the option combinations (one option per ordinary
// term) for which u = sum_i d0_i w_i - rhs lies in the span of the family
// directions v_j = sum_i K_ij w_i, that is, for which the slice equation
// sum_i d_i w_i = rhs has a solution d in the affine coefficient family
// d0 + K lambda. The necessary condition that the (k + 1) x (k + 1) matrix
// of k + 1 random functionals applied to (u, v_1, ..., v_k) be singular mod
// 65521 is expanded by Laplace into features of the two sides of the terms
// (exactly as the Python) and evaluated over the product of the sides in
// exact int64 arithmetic; every zero is then decided on the whole vector
// mod 65521 (u in span(v) by rank) and again mod 2013265921 with the
// family's data there, and only the survivors are returned. The result is
// a superset of the exact solutions (a dependency over the number field
// reduces to one mod every prime); the Python caller re-decides each
// survivor over C and both primes with Family.restrict, as it does for the
// reference path. k = kappa1 is 1 or 2 (n = k + 1 is 2 or 3).
struct DenseSolveInputs {
    int r;                        // ordinary terms
    const int64_t* const* opts1;  // per term: sizes[i] x dim option vectors mod 65521 (projected)
    const int64_t* const* opts2;  // per term: sizes[i] x dim option vectors mod 2013265921
    const int64_t* sizes;         // r
    int64_t dim;
    const int64_t* d01;           // r: the particular coefficients mod 65521
    const int64_t* K1;            // r x kappa1: the family directions mod 65521 (independent columns)
    int kappa1;
    const int64_t* d02;           // r
    const int64_t* K2;            // r x kappa2 mod 2013265921
    int kappa2;
    const int64_t* rhs1;          // dim
    const int64_t* rhs2;          // dim
    const int* sideL;             // the terms of the left side (slice_cover._split_sides()[0])
    int nL;
    const int* sideR;             // the right side
    int nR;
    int64_t max_cand;             // largest number of feature zeros accepted (an error above)
    uint64_t seed;
};

struct DenseSolveResult {
    std::vector<int32_t> combos;  // passed x r option indices, in the order found
    int64_t passed = 0;
    int64_t raw = 0;              // feature zeros (the Python candidate count)
    int64_t pass1 = 0;            // zeros that survive the whole-vector test mod 65521
};

DenseSolveResult dense_solve(const DenseSolveInputs& in);

}  // namespace stabrank
