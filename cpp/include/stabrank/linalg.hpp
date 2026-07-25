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

// Incrementally-updatable least-squares state for the SA hot loop.
//
// Maintains the basis matrix A together with its Gram matrix G = A^H A and
// projected target b = A^H t. Replacing one column refreshes one Gram
// row/column in O(m k) instead of the O(m k^2) full refactorization the
// QR path pays per proposal, and a solve reduces to the k x k system
// G x = b. Every Gram entry is recomputed from the current columns on
// update, so the cached system does not drift across arbitrarily many
// replacements.
//
// Two escape hatches keep the answers as honest as the QR path:
//  - Near-rank-deficient G (relative LDLT pivot ratio below 1e-14, i.e.
//    singular-value ratio ~1e-7) routes the solve through the existing
//    guarded QR/SVD path, which also covers the underdetermined m < k case.
//  - The normal-equation residual formula subtracts O(1) quantities and
//    cannot resolve residuals below ~1e-8 for unit-norm targets; once the
//    estimate drops under 1e-6 the residual is recomputed explicitly as
//    ||t - A x||, so machine-precision decompositions report exactly.
class IncrementalLeastSquares {
public:
    IncrementalLeastSquares(const ComplexVec& target, int basis_size);

    // Replace one basis column and refresh its Gram row/column. O(m k).
    void set_column(int column, const ComplexVec& basis_func);

    // Solve min ||target - A x||_2 from the cached Gram system.
    LeastSquaresResult solve(double rtol = 1e-5, double atol = 1e-8);

private:
    Eigen::MatrixXcd matrix_;   // m x k basis columns
    Eigen::VectorXcd target_;   // m
    Eigen::MatrixXcd gram_;     // k x k, A^H A
    Eigen::VectorXcd proj_;     // k, A^H t
    double target_norm_ = 0.0;

    // Reused solver storage (avoids per-proposal allocation).
    Eigen::LDLT<Eigen::MatrixXcd> ldlt_;
    Eigen::VectorXcd solution_;
    Eigen::VectorXcd residual_scratch_;
};

}  // namespace stabrank
