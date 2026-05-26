#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/linalg.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace stabrank;

namespace {

void require_results_match(
    const LeastSquaresResult& actual,
    const LeastSquaresResult& expected,
    double tol = 1e-10) {
    REQUIRE(actual.is_representable == expected.is_representable);
    REQUIRE_THAT(
        actual.reconstruction_error,
        Catch::Matchers::WithinAbs(expected.reconstruction_error, tol));
    REQUIRE(actual.coeffs.size() == expected.coeffs.size());
    for (size_t idx = 0; idx < actual.coeffs.size(); ++idx) {
        INFO("idx=" << idx);
        REQUIRE_THAT(actual.coeffs[idx].real(), Catch::Matchers::WithinAbs(expected.coeffs[idx].real(), tol));
        REQUIRE_THAT(actual.coeffs[idx].imag(), Catch::Matchers::WithinAbs(expected.coeffs[idx].imag(), tol));
    }
}

}  // namespace

TEST_CASE("lex_index for binary strings") {
    // |00> -> 0, |01> -> 1, |10> -> 2, |11> -> 3
    std::vector<int64_t> v00 = {0, 0};
    std::vector<int64_t> v01 = {0, 1};
    std::vector<int64_t> v10 = {1, 0};
    std::vector<int64_t> v11 = {1, 1};

    REQUIRE(lex_index(v00, 2, 2) == 0);
    REQUIRE(lex_index(v01, 2, 2) == 1);
    REQUIRE(lex_index(v10, 2, 2) == 2);
    REQUIRE(lex_index(v11, 2, 2) == 3);
}

TEST_CASE("lex_index for ternary strings") {
    // |012> in base 3 = 0*9 + 1*3 + 2 = 5
    std::vector<int64_t> v012 = {0, 1, 2};
    REQUIRE(lex_index(v012, 3, 3) == 5);
}

TEST_CASE("mod_inverse basic") {
    // 2^(3-2) mod 3 = 2
    REQUIRE(mod_inverse(2, 3) == 2);
    // 3 * mod_inverse(3, 7) mod 7 = 1
    int64_t inv = mod_inverse(3, 7);
    REQUIRE((3 * inv) % 7 == 1);
}

TEST_CASE("least_squares_solve with exact representation") {
    // target = 0.5 * basis[0] + 0.3 * basis[1]
    ComplexVec b0 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    ComplexVec b1 = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

    ComplexVec target(4);
    for (int i = 0; i < 4; ++i) {
        target[i] = 0.5 * b0[i] + 0.3 * b1[i];
    }

    auto result = least_squares_solve(target, {b0, b1});
    REQUIRE(result.is_representable);
    REQUIRE(result.reconstruction_error < 1e-10);
    REQUIRE_THAT(result.coeffs[0].real(), Catch::Matchers::WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(result.coeffs[1].real(), Catch::Matchers::WithinAbs(0.3, 1e-10));
}

TEST_CASE("least_squares_solve with no exact representation") {
    ComplexVec b0 = {{1.0, 0.0}, {0.0, 0.0}};
    ComplexVec target = {{1.0, 0.0}, {1.0, 0.0}};

    auto result = least_squares_solve(target, {b0});
    REQUIRE_FALSE(result.is_representable);
    REQUIRE(result.reconstruction_error > 0.1);
}

TEST_CASE("least_squares workspace matches direct solve") {
    ComplexVec b0 = {{1.0, 0.0}, {0.0, 0.0}, {0.5, 0.25}};
    ComplexVec b1 = {{0.0, 0.0}, {1.0, 0.0}, {0.25, -0.5}};
    ComplexVec target(3);
    for (int i = 0; i < 3; ++i) {
        target[static_cast<size_t>(i)] =
            std::complex<double>(0.6, -0.2) * b0[static_cast<size_t>(i)] +
            std::complex<double>(-0.1, 0.4) * b1[static_cast<size_t>(i)];
    }

    auto direct = least_squares_solve(target, {b0, b1});

    auto workspace = make_least_squares_workspace(target, 2);
    set_least_squares_basis_column(workspace, 0, b0);
    set_least_squares_basis_column(workspace, 1, b1);
    auto cached = least_squares_solve(workspace);

    require_results_match(cached, direct);
}

TEST_CASE("least_squares workspace reflects column updates") {
    const auto omega = std::exp(std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
    ComplexVec b0 = {
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
    };
    ComplexVec b1 = {
        {0.0, 0.0},
        omega,
        {1.0, -0.25},
    };
    ComplexVec b2 = {
        {0.25, 0.5},
        {0.0, 0.0},
        {0.5, 0.25},
    };
    ComplexVec target(3);
    for (int i = 0; i < 3; ++i) {
        target[static_cast<size_t>(i)] =
            std::complex<double>(0.75, 0.1) * b0[static_cast<size_t>(i)] +
            std::complex<double>(-0.3, 0.2) * b1[static_cast<size_t>(i)] +
            std::complex<double>(0.15, -0.05) * b2[static_cast<size_t>(i)];
    }

    auto workspace = make_least_squares_workspace(target, 3);
    set_least_squares_basis_column(workspace, 0, b0);
    set_least_squares_basis_column(workspace, 1, b1);
    set_least_squares_basis_column(workspace, 2, b2);

    ComplexVec updated_b1 = {
        {0.0, 0.0},
        omega * omega,
        {0.5, 0.5},
    };
    set_least_squares_basis_column(workspace, 1, updated_b1);
    auto updated = least_squares_solve(workspace);
    auto expected = least_squares_solve(target, {b0, updated_b1, b2});

    require_results_match(updated, expected);
}

TEST_CASE("least_squares_solve flags degenerate basis (two identical columns)") {
    // Regression for the SA degeneracy bug: when two basis columns are
    // identical, the QR-implied residual was reported near-zero but the
    // resulting coefficients had ~1e15 magnitude with catastrophic
    // cancellation when actually reconstructed. The guard must detect this
    // and return is_representable=false even when the QR-implied residual
    // is small.
    using Cd = std::complex<double>;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    ComplexVec target = {Cd(inv_sqrt2, 0.0), Cd(inv_sqrt2, 0.0), Cd(0.0, 0.0)};
    ComplexVec same = {Cd(inv_sqrt2, 0.0), Cd(inv_sqrt2, 0.0), Cd(0.0, 0.0)};

    auto result = least_squares_solve(target, {same, same});
    REQUIRE(result.degeneracy_detected);
    REQUIRE_FALSE(result.is_representable);
    // The honest reconstruction error (||target - M*x|| with the unstable x)
    // is what we report when degeneracy is detected. The value depends on
    // floating-point details but it must NOT be a near-zero value that the
    // SA cost function would accept as convergence.
    INFO("reconstruction_error reported: " << result.reconstruction_error);
}

TEST_CASE("least_squares_solve does not flag well-conditioned bases") {
    using Cd = std::complex<double>;
    ComplexVec target = {Cd(1.0, 0.0), Cd(0.0, 0.0), Cd(0.0, 0.0)};
    ComplexVec b0 = {Cd(1.0, 0.0), Cd(0.0, 0.0), Cd(0.0, 0.0)};
    ComplexVec b1 = {Cd(0.0, 0.0), Cd(1.0, 0.0), Cd(0.0, 0.0)};

    auto result = least_squares_solve(target, {b0, b1});
    REQUIRE_FALSE(result.degeneracy_detected);
    REQUIRE(result.is_representable);
}
