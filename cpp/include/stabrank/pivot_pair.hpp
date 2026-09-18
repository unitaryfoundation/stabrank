#pragma once

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <vector>

namespace stabrank {

// Settings of the rank-4 pivot-pair search. The defaults match
// verify_challenge/rank_exclusion.py (PARALLEL, PROJ_DIM, RESID) and the
// numpy reference in verify_challenge/slice_lift.py.
struct PivotPairConfig {
    double parallel = 1e-6;      // |<u|v>| > 1 - parallel counts as parallel (unit vectors)
    double zero = 1e-7;          // a quotient image below this norm is zero
    double key_rel = 1e-8;       // relative gap that separates runs of canonical keys
    double rank_tol = 1e-8;      // smallest singular value for four independent states
    double confirm_resid = 1e-9; // residual of the exact solve for a decomposition
    int proj_dim = 10;           // random projection used for the sort
    uint64_t seed = 11;
};

// Every rank-4 decomposition {i, j, a, b} of psi over the dictionary D
// (one unit-norm state per column) that contains the pivot i and a partner
// j from `partners`, with a and b drawn from the columns marked in
// `allowed` and of index larger than j (so j is the least non-pivot
// member; a decomposition is reported once per such partner). Quotient by span(psi, s_i, s_j): a decomposition's other two
// members have parallel images there, and a parallel pair whose images
// modulo span(s_i, s_j) alone are not parallel is a decomposition (the psi
// coefficient is nonzero). Both tests run on the same random projection,
// and every surviving pair is confirmed by an exact solve in the full
// space, so the projection can lose speed but not decompositions.
std::vector<std::array<int, 4>> rank4_pivot_partners(
    const Eigen::MatrixXcd& D,
    const Eigen::VectorXcd& psi,
    int i,
    const std::vector<int>& partners,
    const std::vector<uint8_t>& allowed,
    const PivotPairConfig& cfg = PivotPairConfig{});

}  // namespace stabrank
