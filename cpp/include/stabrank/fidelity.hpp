#pragma once

#include "types.hpp"

#include <cstdint>
#include <vector>

namespace stabrank {

struct FidelityResult {
    double f_max;                       // Global max fidelity F_max
    int extent_lb;                      // ceil(1/F_max), a LOWER BOUND on
                                        // the stabilizer extent xi (BBCCGH 2019).
                                        // NOT a rank lower bound: chi >= xi
                                        // fails for non-orthogonal stabilizer
                                        // decompositions (paper §3.1).
    std::vector<double> f_max_per_k;    // Best fidelity per subspace dim k
    std::vector<int64_t> states_per_k;  // Number of states checked per k
    int64_t total_states;               // Total states checked
    double elapsed_seconds;             // Wall-clock time
};

// Exhaustive search over all stabilizer states on n qudits of dimension d.
// Returns the maximum fidelity F_max, per-dimension fidelities, and the
// extent lower bound ceil(1/F_max). Uses multithreading to parallelize
// the phase polynomial loop.
FidelityResult max_stabilizer_fidelity(
    const ComplexVec& target,
    int n,
    int d);

}  // namespace stabrank
