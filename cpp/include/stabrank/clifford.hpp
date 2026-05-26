#pragma once

#include "stabrank/types.hpp"
#include <random>

namespace stabrank {

// Applies a random single-gate Clifford (H, S, or CNOT/SUM) to a random target qudit(s).
ComplexVec apply_random_single_gate_clifford(
    const ComplexVec& state,
    int n,
    int p,
    std::mt19937_64& rng);

ComplexVec apply_clifford_H(const ComplexVec& state, int qudit, int n, int p);
ComplexVec apply_clifford_S(const ComplexVec& state, int qudit, int n, int p);
ComplexVec apply_clifford_CX(const ComplexVec& state, int control, int target, int n, int p);
ComplexVec apply_clifford_CZ(const ComplexVec& state, int control, int target, int n, int p);

}  // namespace stabrank
