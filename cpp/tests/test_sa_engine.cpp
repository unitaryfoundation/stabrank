#include <catch2/catch_test_macros.hpp>

#include "stabrank/sa_engine.hpp"
#include "stabrank/polynomial.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

using namespace stabrank;

// Build the 2-qubit |H+>^2 target state for a small convergence test.
static ComplexVec make_qubit_hadamard_eigenstate(int n) {
    // |H+> = (1/N) * [1, sqrt(2)-1]
    double c0 = 1.0;
    double c1 = std::sqrt(2.0) - 1.0;
    double norm = std::sqrt(c0 * c0 + c1 * c1);
    c0 /= norm;
    c1 /= norm;

    // n-fold tensor product
    ComplexVec state = {{c0, 0.0}, {c1, 0.0}};
    for (int i = 1; i < n; ++i) {
        ComplexVec next;
        next.reserve(state.size() * 2);
        for (auto& s : state) {
            next.push_back(s * std::complex<double>(c0, 0.0));
            next.push_back(s * std::complex<double>(c1, 0.0));
        }
        state = std::move(next);
    }
    return state;
}

TEST_CASE("SA engine converges on small qubit problem") {
    int n = 2;
    int p = 2;
    int k = 4;

    auto target = make_qubit_hadamard_eigenstate(n);

    // Generate random initial basis
    std::mt19937_64 rng(12345);
    std::vector<ComplexVec> initial_basis;
    for (int i = 0; i < k; ++i) {
        auto coeffs = generate_random_coeffs(n, p, rng);
        std::vector<int64_t> x0(n, 0);
        IntMatrix W(n, std::vector<int64_t>(n, 0));
        for (int j = 0; j < n; ++j) W[j][j] = 1;  // identity
        auto func = evaluate_coeffs_on_subspace(coeffs, n, p, x0, W);

        double norm = 0.0;
        for (auto& v : func) norm += std::norm(v);
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (auto& v : func) v /= norm;
        }
        initial_basis.push_back(std::move(func));
    }

    SAConfig config;
    config.initial_temperature = 1.0;
    config.cooling_rate = 0.9;
    config.num_iterations_at_temp = 500;
    config.min_temperature = 0.001;
    config.use_real_qubit_moves = true;

    auto result = run_sa_pauli_expansion(
        config, target, n, p, k, initial_basis, /*base_seed=*/12345, nullptr);

    // For a 2-qubit problem with k=4, SA should find a very good
    // decomposition. We use a generous threshold since the test
    // needs to be deterministic and fast.
    REQUIRE(result.best_error < 0.1);
}

namespace {

// Build the qutrit magic cat-state target for the [m, 1] repetition code:
//
//     |C_m^(3)> = 3^((1-m)/2) * sum_{x in C^perp} exp(i 2*pi/9 * |x|_1) |x>
//
// where C^perp = { x in F_3^m : sum x_i == 0 mod 3 } and |x|_1 is the
// integer digit sum (not reduced mod 3, so the phase varies across the
// support).
ComplexVec make_qutrit_cat_target(int m) {
    int dim = 1;
    for (int i = 0; i < m; ++i) dim *= 3;

    ComplexVec target(static_cast<size_t>(dim), {0.0, 0.0});
    const double normalization = 1.0 / std::pow(
        3.0, static_cast<double>(m - 1) / 2.0);
    const double phase_step = 2.0 * std::numbers::pi / 9.0;

    for (int idx = 0; idx < dim; ++idx) {
        int sum_int = 0;
        int temp = idx;
        for (int i = 0; i < m; ++i) {
            sum_int += temp % 3;
            temp /= 3;
        }
        if (sum_int % 3 == 0) {
            target[static_cast<size_t>(idx)] = normalization * std::exp(
                std::complex<double>(0.0, phase_step * sum_int));
        }
    }
    return target;
}

ComplexVec make_random_normalized_basis_func(
    int n_orig, int p, std::mt19937_64& rng) {
    auto coeffs = generate_random_coeffs(n_orig, p, rng);
    std::vector<int64_t> x0(static_cast<size_t>(n_orig), 0);
    IntMatrix W(static_cast<size_t>(n_orig),
                std::vector<int64_t>(static_cast<size_t>(n_orig), 0));
    for (int j = 0; j < n_orig; ++j) W[j][j] = 1;
    auto func = evaluate_coeffs_on_subspace(coeffs, n_orig, p, x0, W);

    double norm_sq = 0.0;
    for (auto& v : func) norm_sq += std::norm(v);
    const double norm = std::sqrt(norm_sq);
    if (norm > 1e-12) {
        for (auto& v : func) v /= norm;
    }
    return func;
}

}  // namespace

TEST_CASE("SA finds a quality decomposition of the qutrit cat2 state") {
    // Integration-level regression test for SA search quality on a real
    // qutrit problem. The kernel-level reference-comparison tests catch
    // numerical bugs in individual operators; this test catches the
    // failure mode they cannot -- "the kernel is still correct, but the
    // SA loop now explores the polytope worse." If a future optimization
    // silently degrades search quality (subtly biased proposals,
    // mis-rounded phases, etc.), the run on this fixed seed and budget
    // will fail to reach the threshold.
    //
    // We use the m=2 cat target (dim 9, three support points) rather than
    // m=3 (dim 27, nine support points): the smaller polytope still
    // exercises every qutrit code path through SA but does not have the
    // deep platform-sensitive local minimum that makes m=3 a knife-edge
    // problem -- a single floating-point sign difference between macOS
    // arm64 and Linux x86_64 routinely sends a single SA chain on m=3
    // into a stuck basin near error ~0.5 on one platform while reaching
    // machine zero on the other.
    constexpr int n_orig = 2;
    constexpr int p = 3;
    constexpr int k_subset_size = 4;

    const auto target = make_qutrit_cat_target(n_orig);

    std::mt19937_64 rng(2026);
    std::vector<ComplexVec> initial_basis;
    initial_basis.reserve(static_cast<size_t>(k_subset_size));
    for (int i = 0; i < k_subset_size; ++i) {
        initial_basis.push_back(
            make_random_normalized_basis_func(n_orig, p, rng));
    }

    SAConfig config;
    config.initial_temperature = 1.0;
    config.cooling_rate = 0.85;
    config.num_iterations_at_temp = 300;
    config.min_temperature = 0.01;
    config.rtol = 1e-5;
    config.atol = 1e-8;
    config.two_func_perturb_prob = 0.3;
    config.random_replace_prob = 0.05;
    config.use_real_qubit_moves = false;
    config.early_exit_threshold = 1e-9;

    auto result = run_sa_pauli_expansion(
        config, target, n_orig, p, k_subset_size,
        initial_basis, /*base_seed=*/2026, nullptr);

    // On the current optimized main, the m=2 cat target reliably triggers
    // SA early-exit (best_error ~1e-15) regardless of platform. The
    // threshold is set well above the early-exit boundary so it absorbs
    // any residual FP variance while still catching mild search-quality
    // regressions of the kind algebraic-invariant tests cannot detect
    // (e.g. a phase-LUT sign error that lets the kernels pass property
    // tests but biases SA proposals).
    INFO("best_error = " << result.best_error);
    REQUIRE(result.best_error < 1e-6);
}
