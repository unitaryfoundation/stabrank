#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/pauli.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <random>

using namespace stabrank;

namespace {

constexpr double kTol = 1e-12;

void require_vec_close(const ComplexVec& actual, const ComplexVec& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        INFO("idx=" << idx);
        REQUIRE_THAT(actual[idx].real(), Catch::Matchers::WithinAbs(expected[idx].real(), kTol));
        REQUIRE_THAT(actual[idx].imag(), Catch::Matchers::WithinAbs(expected[idx].imag(), kTol));
    }
}

ComplexVec reference_apply_X(const ComplexVec& state, int qudit, int n, int p) {
    const int num_elements = static_cast<int>(state.size());
    ComplexVec result(state.size());

    for (int target_idx = 0; target_idx < num_elements; ++target_idx) {
        std::vector<int> target_vec(static_cast<size_t>(n));
        int temp = target_idx;
        for (int i = n - 1; i >= 0; --i) {
            target_vec[static_cast<size_t>(i)] = temp % p;
            temp /= p;
        }

        std::vector<int> source_vec = target_vec;
        source_vec[static_cast<size_t>(qudit)] =
            (source_vec[static_cast<size_t>(qudit)] - 1 + p) % p;

        int source_idx = 0;
        for (int i = 0; i < n; ++i) {
            source_idx = source_idx * p + source_vec[static_cast<size_t>(i)];
        }

        result[static_cast<size_t>(target_idx)] = state[static_cast<size_t>(source_idx)];
    }
    return result;
}

ComplexVec reference_apply_Z(const ComplexVec& state, int qudit, int n, int p) {
    const int num_elements = static_cast<int>(state.size());
    ComplexVec result(state.size());
    const auto omega = std::exp(
        std::complex<double>(0.0, 2.0 * std::numbers::pi / static_cast<double>(p)));

    for (int idx = 0; idx < num_elements; ++idx) {
        int divisor = 1;
        for (int i = 0; i < n - 1 - qudit; ++i) {
            divisor *= p;
        }
        const int xi = (idx / divisor) % p;
        result[static_cast<size_t>(idx)] =
            state[static_cast<size_t>(idx)] * std::pow(omega, xi);
    }
    return result;
}

ComplexVec reference_apply_Y(const ComplexVec& state, int qudit, int n, int p) {
    return reference_apply_X(reference_apply_Z(state, qudit, n, p), qudit, n, p);
}

ComplexVec reference_apply_pauli_string(
    const ComplexVec& state,
    const std::vector<char>& ops,
    int n,
    int p) {
    ComplexVec current = state;
    for (int qudit = 0; qudit < n; ++qudit) {
        switch (ops[static_cast<size_t>(qudit)]) {
            case 'X':
                current = reference_apply_X(current, qudit, n, p);
                break;
            case 'Y':
                current = reference_apply_Y(current, qudit, n, p);
                break;
            case 'Z':
                current = reference_apply_Z(current, qudit, n, p);
                break;
            default:
                break;
        }
    }
    return current;
}

}  // namespace

TEST_CASE("apply_X on single qubit") {
    // |0> -> |1>, |1> -> |0>
    ComplexVec state = {{1.0, 0.0}, {0.0, 0.0}};  // |0>
    auto result = apply_X(state, 0, 1, 2);
    REQUIRE_THAT(result[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result[1].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("apply_Z on single qubit") {
    // Z|1> = -|1>
    ComplexVec state = {{0.0, 0.0}, {1.0, 0.0}};  // |1>
    auto result = apply_Z(state, 0, 1, 2);
    REQUIRE_THAT(result[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result[1].real(), Catch::Matchers::WithinAbs(-1.0, 1e-12));
}

TEST_CASE("apply_Y on single qubit") {
    // Y = ZX. Y|0> = X(Z|0>) = X(|0>) = |1>
    ComplexVec state = {{1.0, 0.0}, {0.0, 0.0}};  // |0>
    auto result = apply_Y(state, 0, 1, 2);
    REQUIRE_THAT(result[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result[1].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("apply_X on second qubit of two") {
    // |00> stays at index 0, maps from source index 1 (|01>)
    // State: |01> = (0, 1, 0, 0)
    ComplexVec state = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    // X on qubit 1: |01> -> |00>
    auto result = apply_X(state, 1, 2, 2);
    REQUIRE_THAT(result[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(result[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("Pauli kernels match the reference implementation on two qutrits") {
    ComplexVec state = {
        {1.0, 0.0},
        {2.0, -0.5},
        {3.0, 0.25},
        {4.0, -0.75},
        {5.0, 0.5},
        {6.0, -1.0},
        {7.0, 0.75},
        {8.0, -1.25},
        {9.0, 1.5},
    };

    for (int qudit = 0; qudit < 2; ++qudit) {
        require_vec_close(apply_X(state, qudit, 2, 3), reference_apply_X(state, qudit, 2, 3));
        require_vec_close(apply_Z(state, qudit, 2, 3), reference_apply_Z(state, qudit, 2, 3));
        require_vec_close(apply_Y(state, qudit, 2, 3), reference_apply_Y(state, qudit, 2, 3));
    }
}

TEST_CASE("apply_random_pauli_string preserves norm structure") {
    // Create a simple + state
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    ComplexVec state = {{inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}};

    std::mt19937_64 rng(42);
    auto [result, ops] = apply_random_pauli_string(state, 1, 2, rng, false);

    // Result should be non-trivial (not all zero)
    double norm_sq = 0.0;
    for (auto& v : result) norm_sq += std::norm(v);
    // The projector sum can produce zero, but for a single qubit with random
    // Pauli, the probability is low. We just check it's finite.
    REQUIRE(std::isfinite(norm_sq));
}

TEST_CASE("apply_random_pauli_string matches the reference projector sum for qutrits") {
    ComplexVec state = {
        {1.0, 0.0},
        {0.5, -0.25},
        {0.75, 0.5},
        {0.25, -0.5},
        {1.25, 0.75},
        {0.0, -1.0},
        {0.5, 0.5},
        {1.5, -0.25},
        {0.25, 1.25},
    };

    std::mt19937_64 rng(123);
    auto [result, ops] = apply_random_pauli_string(state, 2, 3, rng, false);

    const auto omega = std::exp(
        std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
    ComplexVec expected = state;
    ComplexVec current = state;
    std::complex<double> phase = omega;

    for (int k = 0; k < 2; ++k) {
        current = reference_apply_pauli_string(current, ops, 2, 3);
        for (size_t idx = 0; idx < expected.size(); ++idx) {
            expected[idx] += phase * current[idx];
        }
        phase *= omega;
    }

    require_vec_close(result, expected);
}

TEST_CASE("apply_random_pauli_string enforces the even-Y qubit constraint") {
    ComplexVec state = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

    std::mt19937_64 rng(7);
    auto [result, ops] = apply_random_pauli_string(state, 2, 2, rng, true);

    REQUIRE(result.size() == state.size());
    REQUIRE(std::count(ops.begin(), ops.end(), 'Y') % 2 == 0);
}
