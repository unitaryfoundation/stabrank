#pragma once

#include "stabrank/types.hpp"

#include <cstdint>
#include <vector>

namespace stabrank {

// Configuration for the cyclic-orbit symmetric-ansatz SA engine, the C++
// counterpart of stabrank.symmetric. The basis is the union of seed orbits
// under the cyclic qudit shift by `shift_unit`; seed j has orbit size
// orbit_sizes[j] and is kept invariant under the shift by
// shift_unit * orbit_sizes[j] by construction. Coefficients are free and
// recovered by (incremental) least squares.
struct SymmetricConfig {
    int shift_unit = 1;
    std::vector<int> orbit_sizes;
    double initial_temperature = 0.5;
    double min_temperature = 1e-4;
    double cooling_rate = 0.999;
    int iterations_at_temp = 400;
    double reseed_prob = 0.02;
    double two_seed_prob = 0.2;
    double early_exit = 1e-12;
    int audit_every = 5000;
    int num_chains = 1;
};

struct SymmetricResult {
    double best_error = 0.0;
    std::vector<ComplexVec> best_columns;  // expanded orbit columns (rank k)
    ComplexVec best_coeffs;
};

// Cyclically shift qudit positions by `shift` (semantics match
// stabrank.symmetric.shift_qudits).
ComplexVec shift_qudits(const ComplexVec& state, int shift, int n, int p);

SymmetricResult run_symmetric_sa(
    const SymmetricConfig& config,
    const ComplexVec& target,
    int n,
    int p,
    uint64_t base_seed);

}  // namespace stabrank
