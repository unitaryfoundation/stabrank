#pragma once

#include "types.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <span>
#include <vector>

namespace stabrank {

struct LeastSquaresWorkspace {
    Eigen::MatrixXcd matrix;
    Eigen::VectorXcd target;
    double target_norm = 0.0;
};

// Convert an n-digit base-p vector to its lexicographic index.
int lex_index(std::span<const int64_t> point, int n, int p);

// Modular inverse via Fermat's little theorem: val^(p-2) mod p.
int64_t mod_inverse(int64_t val, int64_t p);

// Solve min ||target - M*x||_2 via Eigen's BDCSVD.
// M is built from basis_funcs as columns.
LeastSquaresResult least_squares_solve(
    const ComplexVec& target,
    const std::vector<ComplexVec>& basis_funcs,
    double rtol = 1e-5,
    double atol = 1e-8);

// Allocate and cache the target-side least-squares data for repeated solves.
LeastSquaresWorkspace make_least_squares_workspace(
    const ComplexVec& target,
    int basis_size);

// Replace one matrix column in the cached workspace.
void set_least_squares_basis_column(
    LeastSquaresWorkspace& workspace,
    int column,
    const ComplexVec& basis_func);

// Solve min ||target - M*x||_2 using a pre-populated workspace matrix.
LeastSquaresResult least_squares_solve(
    const LeastSquaresWorkspace& workspace,
    double rtol = 1e-5,
    double atol = 1e-8);

}  // namespace stabrank
