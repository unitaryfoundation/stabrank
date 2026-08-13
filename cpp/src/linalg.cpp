#include "stabrank/linalg.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>

namespace stabrank {

namespace {

double vector_norm(const ComplexVec& vec) {
    double norm_sq = 0.0;
    for (const auto& value : vec) {
        norm_sq += std::norm(value);
    }
    return std::sqrt(norm_sq);
}

}  // namespace

int lex_index(std::span<const int64_t> point, int n, int p) {
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        idx = idx * p + static_cast<int>(point[i]);
    }
    return idx;
}

int64_t mod_inverse(int64_t val, int64_t p) {
    // Fermat's little theorem: val^(p-2) mod p
    int64_t base = ((val % p) + p) % p;
    int64_t exponent = p - 2;
    int64_t result = 1;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = (result * base) % p;
        }
        exponent /= 2;
        base = (base * base) % p;
    }
    return result;
}

LeastSquaresResult least_squares_solve(
    const ComplexVec& target,
    const std::vector<ComplexVec>& basis_funcs,
    double rtol,
    double atol) {
    if (basis_funcs.empty()) {
        LeastSquaresResult result;
        double norm = 0.0;
        for (auto& v : target) norm += std::norm(v);
        norm = std::sqrt(norm);
        bool is_zero = (norm < atol);
        result.is_representable = is_zero;
        result.reconstruction_error = norm;
        return result;
    }

    auto workspace = make_least_squares_workspace(
        target, static_cast<int>(basis_funcs.size()));
    for (int j = 0; j < static_cast<int>(basis_funcs.size()); ++j) {
        set_least_squares_basis_column(workspace, j, basis_funcs[static_cast<size_t>(j)]);
    }
    return least_squares_solve(workspace, rtol, atol);
}

LeastSquaresWorkspace make_least_squares_workspace(
    const ComplexVec& target,
    int basis_size) {

    LeastSquaresWorkspace workspace;
    const int rows = static_cast<int>(target.size());
    workspace.matrix.resize(rows, basis_size);
    workspace.target.resize(rows);
    for (int i = 0; i < rows; ++i) {
        workspace.target(i) = target[static_cast<size_t>(i)];
    }
    workspace.target_norm = vector_norm(target);
    return workspace;
}

void set_least_squares_basis_column(
    LeastSquaresWorkspace& workspace,
    int column,
    const ComplexVec& basis_func) {

    for (int row = 0; row < workspace.matrix.rows(); ++row) {
        workspace.matrix(row, column) = basis_func[static_cast<size_t>(row)];
    }
}

LeastSquaresResult least_squares_solve(
    const LeastSquaresWorkspace& workspace,
    double rtol,
    double atol) {

    LeastSquaresResult result;

    if (workspace.matrix.cols() == 0) {
        bool is_zero = (workspace.target_norm < atol);
        result.is_representable = is_zero;
        result.reconstruction_error = workspace.target_norm;
        return result;
    }

    const Eigen::Index m = workspace.matrix.rows();
    const Eigen::Index k = workspace.matrix.cols();

    if (m < k) {
        // Underdetermined: HouseholderQR cannot solve this. Fall back to
        // SVD which gives the minimum-norm least-squares solution.
        auto svd = workspace.matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXcd x = svd.solve(workspace.target);
        Eigen::VectorXcd residual = workspace.target - workspace.matrix * x;
        const double error = residual.norm();
        result.is_representable = (error <= atol + rtol * workspace.target_norm);
        result.reconstruction_error = error;
        result.coeffs.resize(static_cast<size_t>(k));
        for (Eigen::Index j = 0; j < k; ++j) result.coeffs[static_cast<size_t>(j)] = x(j);
        return result;
    }

    // The SA hot path uses tall-thin matrices (m = p^n_orig, k = basis size)
    // that are full rank in practice because mutation moves producing a
    // zero column are rejected upstream. Householder QR avoids the SVD work
    // we previously did on every proposal, and lets us recover the residual
    // norm from the trailing entries of Q* y for free instead of multiplying
    // M * x out a second time.
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(workspace.matrix);
    Eigen::VectorXcd c = qr.householderQ().adjoint() * workspace.target;
    Eigen::VectorXcd x = qr.matrixQR()
        .topLeftCorner(k, k)
        .triangularView<Eigen::Upper>()
        .solve(c.head(k));
    double error = (m > k) ? c.tail(m - k).norm() : 0.0;

    // Degeneracy guard: if R has near-zero diagonal entries the triangular
    // solve above amplifies floating-point noise and produces an x with huge
    // magnitude that nearly cancels when multiplied by M. The QR-implied
    // residual (the trailing-norm computation) does NOT see this — it only
    // captures the component orthogonal to the column span. So a rank-
    // deficient basis can pass through silently with a tiny "error" but
    // catastrophically wrong coefficients (||x|| ~ 1e15 with cancellation
    // down to ||M*x - target|| ~ 1).
    //
    // We detect this via the relative ratio of |R_ii|. If the smallest
    // diagonal entry is much smaller than the largest, the basis is
    // near-singular; we recompute the residual via the explicit M*x to get
    // an honest answer that the SA cost function can act on.
    constexpr double kRelativeRankTol = 1e-10;
    double max_abs_diag = 0.0;
    double min_abs_diag = std::numeric_limits<double>::infinity();
    for (Eigen::Index j = 0; j < k; ++j) {
        const double v = std::abs(qr.matrixQR()(j, j));
        if (v > max_abs_diag) max_abs_diag = v;
        if (v < min_abs_diag) min_abs_diag = v;
    }
    const bool degenerate =
        (max_abs_diag > 0.0) && (min_abs_diag < kRelativeRankTol * max_abs_diag);
    if (degenerate) {
        // The triangular solve's x can be enormous here, and the explicit
        // reconstruction ||t - M x|| then carries absolute error of order
        // ||M|| ||x|| eps, which UNDERREPORTS the true span distance (a
        // duplicated column drove ||x|| ~ 1e15 and produced a stable-looking
        // but bogus residual that SA then chased). Solve instead by
        // rank-truncated SVD: the minimal-norm solution has modest ||x||, so
        // both it and its residual are honest.
        Eigen::BDCSVD<Eigen::MatrixXcd> svd(
            workspace.matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(kRelativeRankTol);
        x = svd.solve(workspace.target);
        error = (workspace.target - workspace.matrix * x).norm();
    }

    result.is_representable = !degenerate &&
        (error <= atol + rtol * workspace.target_norm);
    result.reconstruction_error = error;
    result.degeneracy_detected = degenerate;
    result.coeffs.resize(static_cast<size_t>(k));
    for (Eigen::Index j = 0; j < k; ++j) {
        result.coeffs[static_cast<size_t>(j)] = x(j);
    }

    return result;
}

IncrementalLeastSquares::IncrementalLeastSquares(
    const ComplexVec& target,
    int basis_size) {

    const int rows = static_cast<int>(target.size());
    matrix_ = Eigen::MatrixXcd::Zero(rows, basis_size);
    target_.resize(rows);
    for (int i = 0; i < rows; ++i) {
        target_(i) = target[static_cast<size_t>(i)];
    }
    target_norm_ = target_.norm();
    gram_ = Eigen::MatrixXcd::Zero(basis_size, basis_size);
    proj_ = Eigen::VectorXcd::Zero(basis_size);
}

void IncrementalLeastSquares::set_column(
    int column,
    const ComplexVec& basis_func) {

    for (int row = 0; row < matrix_.rows(); ++row) {
        matrix_(row, column) = basis_func[static_cast<size_t>(row)];
    }
    const auto col = matrix_.col(column);
    for (Eigen::Index j = 0; j < matrix_.cols(); ++j) {
        // G(j, c) = a_j^H a_c; recomputed exactly from the current columns.
        const std::complex<double> g = matrix_.col(j).dot(col);
        gram_(j, column) = g;
        gram_(column, j) = std::conj(g);
    }
    proj_(column) = col.dot(target_);
}

LeastSquaresResult IncrementalLeastSquares::solve(double rtol, double atol) {
    const Eigen::Index m = matrix_.rows();
    const Eigen::Index k = matrix_.cols();

    if (k == 0) {
        LeastSquaresResult result;
        result.is_representable = (target_norm_ < atol);
        result.reconstruction_error = target_norm_;
        return result;
    }

    ldlt_.compute(gram_);

    // The eigenvalues of G are the squared singular values of A, so the QR
    // path's 1e-10 relative diagonal threshold corresponds to 1e-20 here.
    // Use a more conservative 1e-14 (singular-value ratio ~1e-7): the Gram
    // route loses accuracy earlier than QR, so marginal cases are routed to
    // the guarded QR/SVD solver instead of risking a bad solve. That
    // fallback also covers the underdetermined m < k case, where G is
    // always singular.
    constexpr double kGramRankTol = 1e-14;
    const Eigen::VectorXd d = ldlt_.vectorD().real().cwiseAbs();
    const double max_d = d.maxCoeff();
    const double min_d = d.minCoeff();
    const bool degenerate = (ldlt_.info() != Eigen::Success) ||
        !(max_d > 0.0) || (m < k) || (min_d < kGramRankTol * max_d);

    if (degenerate) {
        LeastSquaresWorkspace workspace;
        workspace.matrix = matrix_;
        workspace.target = target_;
        workspace.target_norm = target_norm_;
        return least_squares_solve(workspace, rtol, atol);
    }

    solution_ = ldlt_.solve(proj_);

    // r^2 = ||t||^2 - 2 Re(b^H x) + x^H G x, clamped against fp noise.
    double r_sq = target_norm_ * target_norm_
        - 2.0 * std::real(proj_.dot(solution_))
        + std::real(solution_.dot(gram_ * solution_));
    if (r_sq < 0.0) r_sq = 0.0;
    double error = std::sqrt(r_sq);

    // Below the cancellation floor of the subtraction above the estimate is
    // meaningless; recompute explicitly so near-exact decompositions report
    // machine-precision residuals just like the QR path does.
    const double explicit_floor = 1e-6 * std::max(1.0, target_norm_);
    if (error < explicit_floor) {
        residual_scratch_ = target_ - matrix_ * solution_;
        error = residual_scratch_.norm();
    }

    LeastSquaresResult result;
    result.is_representable = (error <= atol + rtol * target_norm_);
    result.reconstruction_error = error;
    result.coeffs.resize(static_cast<size_t>(k));
    for (Eigen::Index j = 0; j < k; ++j) {
        result.coeffs[static_cast<size_t>(j)] = solution_(j);
    }
    return result;
}

}  // namespace stabrank
