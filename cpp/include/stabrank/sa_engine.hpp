#pragma once

#include "types.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace stabrank {

// Run SA with Pauli-expansion moves.
// When config.num_chains > 1, launches that many independent SA chains
// in parallel threads (each seeded from base_seed + chain_index) and
// returns the result with the lowest cost across all chains.
// generator_func is an optional callback to produce fresh random basis
// functions for the "random reset" move type.
SAResult run_sa_pauli_expansion(
    const SAConfig& config,
    const ComplexVec& target,
    int n_orig,
    int p_prime,
    int k_subset_size,
    std::vector<ComplexVec> initial_basis,
    uint64_t base_seed,
    std::function<ComplexVec(std::mt19937_64&)> generator_func = nullptr);

}  // namespace stabrank

