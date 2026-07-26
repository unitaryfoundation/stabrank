#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/symmetric_engine.hpp"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace stabrank;

namespace {

ComplexVec random_vec(size_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    ComplexVec v(dim);
    for (auto& z : v) z = {dist(rng), dist(rng)};
    return v;
}

// The qutrit strange state |S> = (|1> - |2>)/sqrt(2), two copies.
ComplexVec strange_two_copies() {
    const double r = 1.0 / std::sqrt(2.0);
    ComplexVec s = {{0.0, 0.0}, {r, 0.0}, {-r, 0.0}};
    ComplexVec out(9);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            out[static_cast<size_t>(3 * a + b)] = s[static_cast<size_t>(a)] * s[static_cast<size_t>(b)];
    return out;
}

}  // namespace

TEST_CASE("shift_qudits round trip and small-case correctness", "[symmetric]") {
    const auto v = random_vec(81, 11);
    const auto w = shift_qudits(shift_qudits(v, 1, 4, 3), -1, 4, 3);
    for (size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(std::abs(v[i] - w[i]), Catch::Matchers::WithinAbs(0.0, 1e-14));
    }

    // Two qubits: shifting by 1 swaps the qubits, i.e. transposes |ab> -> |ba>.
    ComplexVec x = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const auto y = shift_qudits(x, 1, 2, 2);
    REQUIRE(y[0] == x[0]);
    REQUIRE(y[1] == x[2]);
    REQUIRE(y[2] == x[1]);
    REQUIRE(y[3] == x[3]);
}

TEST_CASE("symmetric SA finds the strange m=2 rank-2 decomposition", "[symmetric]") {
    SymmetricConfig config;
    config.shift_unit = 1;
    config.orbit_sizes = {1, 1};
    config.initial_temperature = 0.5;
    config.min_temperature = 1e-3;
    config.cooling_rate = 0.99;
    config.iterations_at_temp = 300;
    config.num_chains = 4;

    const auto target = strange_two_copies();
    const auto result = run_symmetric_sa(config, target, 2, 3, /*base_seed=*/7);
    REQUIRE(result.best_error < 1e-9);
    REQUIRE(result.best_columns.size() == 2);
    REQUIRE(result.best_coeffs.size() == 2);
}

TEST_CASE("symmetric SA validates configuration", "[symmetric]") {
    const auto target = strange_two_copies();
    SymmetricConfig config;
    config.orbit_sizes = {1};
    config.shift_unit = 3;  // does not divide n = 2
    REQUIRE_THROWS(run_symmetric_sa(config, target, 2, 3, 1));
    config.shift_unit = 1;
    config.orbit_sizes = {4};  // does not divide the group order
    REQUIRE_THROWS(run_symmetric_sa(config, target, 2, 3, 1));
}
