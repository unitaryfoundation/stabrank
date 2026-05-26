#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace stabrank {

using ComplexVec = std::vector<std::complex<double>>;

// Polynomial coefficients over GF(p).
// Represents q(y) = alpha + sum(c_j0_lin[i]*y_i)/p + sum(c_j0_qs[i]*y_i^2)/p
//                  + sum(c_j0_qm[idx]*y_s*y_t)/p + sum(c_j1_lin[i]*y_i)/4
struct PolyCoeffs {
    double alpha = 0.0;
    std::vector<int64_t> c_j0_lin;      // linear coeffs, size = k_dim
    std::vector<int64_t> c_j0_qm;       // mixed quadratic, size = k*(k-1)/2
    std::vector<int64_t> c_j0_qs;       // square quadratic, size = k_dim (p>=3 only)
    std::vector<int64_t> c_j1_lin;      // second-level linear, size = k_dim (p==2 only)
};

// Result of a least-squares solve.
struct LeastSquaresResult {
    bool is_representable = false;
    ComplexVec coeffs;
    double reconstruction_error = 0.0;
    // True when the basis matrix is near-rank-deficient (smallest absolute
    // QR-diagonal entry is below a relative threshold against the largest).
    // In this case the QR-implied residual can underreport the actual
    // ||target - M*x|| due to catastrophic cancellation in the triangular
    // solve, so we recompute the residual explicitly and store it here.
    bool degeneracy_detected = false;
};

// Represents a single step in the SA trajectory
struct SATraceStep {
    int iteration;
    double temperature;
    double current_cost;
    double best_cost;
    bool accepted;
    int move_type; // 0=RandomReset, 1=Cluster, 2=Single
    std::vector<int> k_values;
};

// Configuration for the SA engine.
struct SAConfig {
    double initial_temperature = 1.0;
    double cooling_rate = 0.99;
    int num_iterations_at_temp = 1000;
    double min_temperature = 1e-5;
    double rtol = 1e-5;
    double atol = 1e-8;
    double two_func_perturb_prob = 0.1;
    double random_replace_prob = 0.01;
    bool use_real_qubit_moves = false;
    double clifford_ratio = 0.0;
    int fixed_dimension = -1;
    double early_exit_threshold = 1e-9;
    int num_chains = 1;
    bool enable_tracing = false;
};

// Result of an SA run.
struct SAResult {
    int k;
    std::vector<ComplexVec> best_basis_funcs;
    ComplexVec best_lin_coeffs;
    double best_error = 0.0;
    double best_cost = 0.0;
    std::vector<SATraceStep> trace;
};

// Integer matrix type for subspace bases (row-major, k_dim x n_orig).
using IntMatrix = std::vector<std::vector<int64_t>>;

}  // namespace stabrank
