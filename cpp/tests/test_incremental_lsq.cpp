#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/linalg.hpp"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace stabrank;

namespace {

ComplexVec random_state(int dim, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    ComplexVec state(static_cast<size_t>(dim));
    double norm_sq = 0.0;
    for (auto& v : state) {
        v = std::complex<double>(dist(rng), dist(rng));
        norm_sq += std::norm(v);
    }
    const double norm = std::sqrt(norm_sq);
    for (auto& v : state) v /= norm;
    return state;
}

}  // namespace

TEST_CASE("incremental solve matches batch QR across random column churn",
          "[linalg][incremental]") {
    std::mt19937_64 rng(42);
    const int m = 64;
    const int k = 5;

    const ComplexVec target = random_state(m, rng);
    std::vector<ComplexVec> basis;
    for (int j = 0; j < k; ++j) basis.push_back(random_state(m, rng));

    IncrementalLeastSquares inc(target, k);
    for (int j = 0; j < k; ++j) inc.set_column(j, basis[static_cast<size_t>(j)]);

    std::uniform_int_distribution<int> idx_dist(0, k - 1);
    for (int step = 0; step < 200; ++step) {
        const int idx = idx_dist(rng);
        basis[static_cast<size_t>(idx)] = random_state(m, rng);
        inc.set_column(idx, basis[static_cast<size_t>(idx)]);

        if (step % 20 != 0) continue;
        const auto incremental = inc.solve(1e-5, 1e-8);
        const auto batch = least_squares_solve(target, basis, 1e-5, 1e-8);
        INFO("step=" << step);
        REQUIRE_THAT(
            incremental.reconstruction_error,
            Catch::Matchers::WithinAbs(batch.reconstruction_error, 1e-8));
        REQUIRE(incremental.coeffs.size() == batch.coeffs.size());
        for (size_t j = 0; j < batch.coeffs.size(); ++j) {
            INFO("coeff " << j);
            REQUIRE_THAT(
                std::abs(incremental.coeffs[j] - batch.coeffs[j]),
                Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
    }
}

TEST_CASE("incremental solve resolves exact decompositions to machine precision",
          "[linalg][incremental]") {
    std::mt19937_64 rng(7);
    const int m = 81;
    const int k = 4;

    std::vector<ComplexVec> basis;
    for (int j = 0; j < k; ++j) basis.push_back(random_state(m, rng));

    // target = sum_j c_j basis_j exactly.
    std::vector<std::complex<double>> cs = {
        {0.3, 0.1}, {-0.7, 0.4}, {0.05, -0.9}, {1.1, 0.0}};
    ComplexVec target(static_cast<size_t>(m), {0.0, 0.0});
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < m; ++i) {
            target[static_cast<size_t>(i)] +=
                cs[static_cast<size_t>(j)] *
                basis[static_cast<size_t>(j)][static_cast<size_t>(i)];
        }
    }

    IncrementalLeastSquares inc(target, k);
    for (int j = 0; j < k; ++j) inc.set_column(j, basis[static_cast<size_t>(j)]);

    const auto result = inc.solve(1e-5, 1e-8);
    // The Gram residual formula bottoms out near 1e-8; the explicit-residual
    // tier must kick in and report machine precision.
    REQUIRE(result.reconstruction_error < 1e-12);
    REQUIRE(result.is_representable);
}

TEST_CASE("incremental solve routes degenerate bases through the guarded path",
          "[linalg][incremental]") {
    std::mt19937_64 rng(3);
    const int m = 32;
    const int k = 4;

    const ComplexVec target = random_state(m, rng);
    std::vector<ComplexVec> basis;
    for (int j = 0; j < k - 1; ++j) basis.push_back(random_state(m, rng));
    basis.push_back(basis[0]);  // duplicated column -> rank deficient

    IncrementalLeastSquares inc(target, k);
    for (int j = 0; j < k; ++j) inc.set_column(j, basis[static_cast<size_t>(j)]);

    const auto incremental = inc.solve(1e-5, 1e-8);
    const auto batch = least_squares_solve(target, basis, 1e-5, 1e-8);
    REQUIRE(incremental.degeneracy_detected == batch.degeneracy_detected);
    REQUIRE_THAT(
        incremental.reconstruction_error,
        Catch::Matchers::WithinAbs(batch.reconstruction_error, 1e-8));
}

TEST_CASE("incremental solve handles underdetermined m < k via fallback",
          "[linalg][incremental]") {
    std::mt19937_64 rng(5);
    const int m = 4;
    const int k = 6;

    const ComplexVec target = random_state(m, rng);
    std::vector<ComplexVec> basis;
    for (int j = 0; j < k; ++j) basis.push_back(random_state(m, rng));

    IncrementalLeastSquares inc(target, k);
    for (int j = 0; j < k; ++j) inc.set_column(j, basis[static_cast<size_t>(j)]);

    const auto incremental = inc.solve(1e-5, 1e-8);
    const auto batch = least_squares_solve(target, basis, 1e-5, 1e-8);
    REQUIRE_THAT(
        incremental.reconstruction_error,
        Catch::Matchers::WithinAbs(batch.reconstruction_error, 1e-8));
}
