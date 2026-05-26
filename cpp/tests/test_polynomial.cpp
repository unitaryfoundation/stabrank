#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/polynomial.hpp"

#include <cmath>
#include <numbers>
#include <random>

using namespace stabrank;

TEST_CASE("evaluate_poly_at_point - constant polynomial") {
    PolyCoeffs coeffs;
    coeffs.alpha = 0.5;
    coeffs.c_j0_lin = {};
    coeffs.c_j0_qm = {};
    coeffs.c_j0_qs = {};
    coeffs.c_j1_lin = {};

    std::vector<int64_t> point = {};
    double val = evaluate_poly_at_point(coeffs, point, 0, 2);
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(0.5, 1e-12));
}

TEST_CASE("evaluate_poly_at_point - linear term p=3") {
    // q(y) = y_0 / 3, evaluate at y_0 = 1 -> 1/3
    PolyCoeffs coeffs;
    coeffs.alpha = 0.0;
    coeffs.c_j0_lin = {1};
    coeffs.c_j0_qm = {};
    coeffs.c_j0_qs = {0};
    coeffs.c_j1_lin = {0};

    std::vector<int64_t> point = {1};
    double val = evaluate_poly_at_point(coeffs, point, 1, 3);
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(1.0 / 3.0, 1e-12));
}

TEST_CASE("evaluate_coeffs_on_subspace - trivial case") {
    // n_orig=1, p=2, k_dim=1, W=[[1]], x0=[0]
    // Only linear term c_j0_lin[0]=1. 
    // y=0: q=0, phase=exp(0)=1 -> x=0, output[0]=1
    // y=1: q=1/2, phase=exp(i*pi)=-1 -> x=1, output[1]=-1
    PolyCoeffs coeffs;
    coeffs.alpha = 0.0;
    coeffs.c_j0_lin = {1};
    coeffs.c_j0_qm = {};
    coeffs.c_j0_qs = {0};
    coeffs.c_j1_lin = {0};

    std::vector<int64_t> x0 = {0};
    IntMatrix W = {{1}};

    auto output = evaluate_coeffs_on_subspace(coeffs, 1, 2, x0, W);
    REQUIRE(output.size() == 2);
    REQUIRE_THAT(output[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(output[1].real(), Catch::Matchers::WithinAbs(-1.0, 1e-12));
}

TEST_CASE("generate_random_coeffs dimensions") {
    std::mt19937_64 rng(42);

    auto c2 = generate_random_coeffs(3, 2, rng);
    REQUIRE(c2.c_j0_lin.size() == 3);
    REQUIRE(c2.c_j0_qm.size() == 3);  // 3*2/2
    REQUIRE(c2.c_j0_qs.size() == 3);
    REQUIRE(c2.c_j1_lin.size() == 3);

    auto c3 = generate_random_coeffs(3, 3, rng);
    REQUIRE(c3.c_j0_lin.size() == 3);
    REQUIRE(c3.c_j0_qm.size() == 3);
    // p>=3, so qs should be random (not all zeros)
    REQUIRE(c3.c_j0_qs.size() == 3);
    // p!=2, so j1_lin should be zeros
    bool all_zero = true;
    for (auto v : c3.c_j1_lin) if (v != 0) all_zero = false;
    REQUIRE(all_zero);
}
