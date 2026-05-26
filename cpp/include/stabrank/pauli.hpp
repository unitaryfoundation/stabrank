#pragma once

#include "types.hpp"

#include <random>
#include <utility>
#include <vector>

namespace stabrank {

// Apply the generalized X (shift) operator on qudit index i.
ComplexVec apply_X(const ComplexVec& state, int qudit, int n, int p);

// Apply the generalized Z (phase) operator on qudit index i.
ComplexVec apply_Z(const ComplexVec& state, int qudit, int n, int p);

// Apply Y = Z then X on qudit index i.
ComplexVec apply_Y(const ComplexVec& state, int qudit, int n, int p);

// Apply a random Pauli string and the stabilizer projector sum,
// returning the transformed state and the chosen operator labels.
// If even_y_constraint is true (qubits only), ensures an even number of Y ops.
std::pair<ComplexVec, std::vector<char>> apply_random_pauli_string(
    const ComplexVec& state,
    int n,
    int p,
    std::mt19937_64& rng,
    bool even_y_constraint = false);

}  // namespace stabrank
