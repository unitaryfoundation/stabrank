#pragma once

#include "types.hpp"

#include <cstdint>
#include <random>
#include <span>

namespace stabrank {

// Evaluate a polynomial at a single point y in GF(p)^k, returning
// the fractional value q(y) mod 1.0.
double evaluate_poly_at_point(
    const PolyCoeffs& coeffs,
    std::span<const int64_t> point,
    int k_dim,
    int p);

// Evaluate polynomial coefficients on the full original space,
// returning a complex vector of length p^n_orig.
ComplexVec evaluate_coeffs_on_subspace(
    const PolyCoeffs& coeffs,
    int n_orig,
    int p,
    std::span<const int64_t> x0_translation,
    const IntMatrix& W_basis);

// Generate random polynomial coefficients.
PolyCoeffs generate_random_coeffs(
    int k_dim, int p, std::mt19937_64& rng);

}  // namespace stabrank
