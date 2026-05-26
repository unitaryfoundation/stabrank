// Property tests for the C++ kernels.
//
// These tests assert algebraic invariants that must hold for any correct
// implementation of the Pauli and Clifford kernels regardless of internal
// representation: norm preservation, P^p = (phase) * I, Hadamard / phase /
// CX / CZ powers, etc. They are deliberately implementation-agnostic so
// they remain valid as the kernels get optimized.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "stabrank/clifford.hpp"
#include "stabrank/linalg.hpp"
#include "stabrank/pauli.hpp"
#include "stabrank/polynomial.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

using namespace stabrank;

namespace {

constexpr double kTol = 1e-10;

ComplexVec random_state(int n, int p, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    int dim = 1;
    for (int i = 0; i < n; ++i) dim *= p;

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

double l2_norm(const ComplexVec& vec) {
    double s = 0.0;
    for (const auto& v : vec) s += std::norm(v);
    return std::sqrt(s);
}

void require_close(
    const ComplexVec& actual,
    const ComplexVec& expected,
    double tol = kTol) {
    REQUIRE(actual.size() == expected.size());
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        INFO("idx=" << idx);
        REQUIRE_THAT(actual[idx].real(), Catch::Matchers::WithinAbs(expected[idx].real(), tol));
        REQUIRE_THAT(actual[idx].imag(), Catch::Matchers::WithinAbs(expected[idx].imag(), tol));
    }
}

void require_proportional(
    const ComplexVec& actual,
    const ComplexVec& expected,
    std::complex<double> ratio,
    double tol = kTol) {
    REQUIRE(actual.size() == expected.size());
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        const auto target = expected[idx] * ratio;
        INFO("idx=" << idx);
        REQUIRE_THAT(actual[idx].real(), Catch::Matchers::WithinAbs(target.real(), tol));
        REQUIRE_THAT(actual[idx].imag(), Catch::Matchers::WithinAbs(target.imag(), tol));
    }
}

}  // namespace

TEST_CASE("Pauli kernels preserve the L2 norm") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 11}, {4, 2, 13}, {5, 2, 17},
        {2, 3, 19}, {3, 3, 23}, {4, 3, 29},
    };

    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        const double norm = l2_norm(state);

        for (int q = 0; q < c.n; ++q) {
            REQUIRE_THAT(l2_norm(apply_X(state, q, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
            REQUIRE_THAT(l2_norm(apply_Z(state, q, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
            REQUIRE_THAT(l2_norm(apply_Y(state, q, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
        }
    }
}

TEST_CASE("X^p == I and Z^p == I on random states") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {4, 2, 101}, {5, 2, 103}, {3, 3, 107}, {4, 3, 109},
    };

    for (const auto& c : cases) {
        const auto initial = random_state(c.n, c.p, c.seed);

        for (int q = 0; q < c.n; ++q) {
            ComplexVec via_x = initial;
            ComplexVec via_z = initial;
            for (int k = 0; k < c.p; ++k) {
                via_x = apply_X(via_x, q, c.n, c.p);
                via_z = apply_Z(via_z, q, c.n, c.p);
            }
            require_close(via_x, initial);
            require_close(via_z, initial);
        }
    }
}

TEST_CASE("Y^p collapses to a global scalar (per qudit)") {
    // Y == apply_X . apply_Z, so on |x>:
    //   Y|x> = omega^x |x+1 mod p>
    //   Y^p|x> = omega^(0 + 1 + ... + (p-1) + p*x) |x>
    //         = omega^(p(p-1)/2 + p*x) |x>
    //   For p=2: omega^(1 + 2x) = -1; so Y^2 = -I.
    //   For p=3: omega^(3 + 3x) = omega^0 = 1; so Y^3 = I.
    struct Case { int n; int p; uint64_t seed; std::complex<double> expected; };
    const Case cases[] = {
        {4, 2, 211, std::complex<double>(-1.0, 0.0)},
        {5, 2, 213, std::complex<double>(-1.0, 0.0)},
        {3, 3, 217, std::complex<double>(1.0, 0.0)},
        {4, 3, 219, std::complex<double>(1.0, 0.0)},
    };

    for (const auto& c : cases) {
        const auto initial = random_state(c.n, c.p, c.seed);
        for (int q = 0; q < c.n; ++q) {
            ComplexVec via = initial;
            for (int k = 0; k < c.p; ++k) {
                via = apply_Y(via, q, c.n, c.p);
            }
            require_proportional(via, initial, c.expected);
        }
    }
}

TEST_CASE("X and Z anti-commute on qubits, omega-commute on qutrits") {
    // For qudits with omega = exp(2*pi*i / p):
    //   ZX |x> = omega^(x+1) |x+1>
    //   XZ |x> = omega^x |x+1>
    // so ZX = omega * XZ on every basis state.
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {{3, 2, 311}, {3, 3, 313}};

    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi /
                                          static_cast<double>(c.p)));

        for (int q = 0; q < c.n; ++q) {
            const auto zx = apply_Z(apply_X(state, q, c.n, c.p), q, c.n, c.p);
            const auto xz = apply_X(apply_Z(state, q, c.n, c.p), q, c.n, c.p);
            require_proportional(zx, xz, omega);
        }
    }
}

TEST_CASE("Clifford kernels preserve the L2 norm") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 401}, {4, 2, 403},
        {3, 3, 405}, {4, 3, 407},
    };

    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        const double norm = l2_norm(state);

        for (int q = 0; q < c.n; ++q) {
            REQUIRE_THAT(l2_norm(apply_clifford_H(state, q, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
            REQUIRE_THAT(l2_norm(apply_clifford_S(state, q, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
        }
        if (c.n >= 2) {
            REQUIRE_THAT(l2_norm(apply_clifford_CX(state, 0, 1, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
            REQUIRE_THAT(l2_norm(apply_clifford_CZ(state, 0, 1, c.n, c.p)),
                         Catch::Matchers::WithinAbs(norm, kTol));
        }
    }
}

TEST_CASE("Hadamard satisfies F^4 == I (qubits also satisfy F^2 == I)") {
    // The Hadamard kernel implements the qudit Fourier transform, so F^4 == I
    // on every qudit dimension. For p=2 it additionally satisfies F^2 == I.
    {
        const auto initial = random_state(4, 2, 501);
        for (int q = 0; q < 4; ++q) {
            const auto h2 = apply_clifford_H(
                apply_clifford_H(initial, q, 4, 2), q, 4, 2);
            require_close(h2, initial);
        }
    }
    {
        const auto initial = random_state(3, 3, 503);
        for (int q = 0; q < 3; ++q) {
            ComplexVec h4 = initial;
            for (int k = 0; k < 4; ++k) {
                h4 = apply_clifford_H(h4, q, 3, 3);
            }
            require_close(h4, initial);
        }
    }
}

TEST_CASE("S^4 == I on qubits and S^3 == I on qutrits") {
    {
        const auto initial = random_state(3, 2, 601);
        for (int q = 0; q < 3; ++q) {
            ComplexVec s4 = initial;
            for (int k = 0; k < 4; ++k) {
                s4 = apply_clifford_S(s4, q, 3, 2);
            }
            require_close(s4, initial);
        }
    }
    {
        const auto initial = random_state(3, 3, 603);
        for (int q = 0; q < 3; ++q) {
            ComplexVec s3 = initial;
            for (int k = 0; k < 3; ++k) {
                s3 = apply_clifford_S(s3, q, 3, 3);
            }
            require_close(s3, initial);
        }
    }
}

TEST_CASE("CX^p == I and CZ^p == I for both qubits and qutrits") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {{3, 2, 701}, {3, 3, 703}, {4, 3, 705}};

    for (const auto& c : cases) {
        const auto initial = random_state(c.n, c.p, c.seed);
        for (int ctrl = 0; ctrl < c.n; ++ctrl) {
            for (int tgt = 0; tgt < c.n; ++tgt) {
                if (ctrl == tgt) continue;
                ComplexVec cx_p = initial;
                ComplexVec cz_p = initial;
                for (int k = 0; k < c.p; ++k) {
                    cx_p = apply_clifford_CX(cx_p, ctrl, tgt, c.n, c.p);
                    cz_p = apply_clifford_CZ(cz_p, ctrl, tgt, c.n, c.p);
                }
                require_close(cx_p, initial);
                require_close(cz_p, initial);
            }
        }
    }
}

namespace {

// Slow-but-explicit reference implementations of the Clifford and polynomial
// kernels. These are intentionally written to look like the math: every
// amplitude is decoded into per-qudit digits, the operator's effect is
// computed in those terms, and the result is encoded back. Optimization
// passes (stride-based iteration, phase look-up tables, integer accumulators)
// must produce the same output as these references on every test case.

ComplexVec reference_apply_clifford_H(const ComplexVec& state, int qudit, int n, int p) {
    int num_elements = static_cast<int>(state.size());
    ComplexVec result(static_cast<size_t>(num_elements), {0.0, 0.0});

    int divisor = 1;
    for (int i = 0; i < n - 1 - qudit; ++i) divisor *= p;

    if (p == 2) {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        for (int idx = 0; idx < num_elements; ++idx) {
            const int x_i = (idx / divisor) % 2;
            if (x_i == 0) {
                const int idx1 = idx + divisor;
                const auto a0 = state[static_cast<size_t>(idx)];
                const auto a1 = state[static_cast<size_t>(idx1)];
                result[static_cast<size_t>(idx)] = (a0 + a1) * inv_sqrt2;
                result[static_cast<size_t>(idx1)] = (a0 - a1) * inv_sqrt2;
            }
        }
    } else if (p == 3) {
        const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        for (int idx = 0; idx < num_elements; ++idx) {
            const int x_i = (idx / divisor) % 3;
            if (x_i == 0) {
                const int idx1 = idx + divisor;
                const int idx2 = idx + 2 * divisor;
                const auto a0 = state[static_cast<size_t>(idx)];
                const auto a1 = state[static_cast<size_t>(idx1)];
                const auto a2 = state[static_cast<size_t>(idx2)];
                result[static_cast<size_t>(idx)] = (a0 + a1 + a2) * inv_sqrt3;
                result[static_cast<size_t>(idx1)] =
                    (a0 + a1 * omega + a2 * omega * omega) * inv_sqrt3;
                result[static_cast<size_t>(idx2)] =
                    (a0 + a1 * omega * omega + a2 * omega) * inv_sqrt3;
            }
        }
    }
    return result;
}

ComplexVec reference_apply_clifford_S(const ComplexVec& state, int qudit, int n, int p) {
    ComplexVec result = state;
    int divisor = 1;
    for (int i = 0; i < n - 1 - qudit; ++i) divisor *= p;

    if (p == 2) {
        const std::complex<double> i_phase(0.0, 1.0);
        for (size_t idx = 0; idx < state.size(); ++idx) {
            const int x_i = (static_cast<int>(idx) / divisor) % 2;
            if (x_i == 1) result[idx] *= i_phase;
        }
    } else if (p == 3) {
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        for (size_t idx = 0; idx < state.size(); ++idx) {
            const int x_i = (static_cast<int>(idx) / divisor) % 3;
            if (x_i != 0) result[idx] *= omega;
        }
    }
    return result;
}

ComplexVec reference_apply_clifford_CX(
    const ComplexVec& state, int control, int target, int n, int p) {
    int num_elements = static_cast<int>(state.size());
    ComplexVec result(static_cast<size_t>(num_elements));

    int divisor_c = 1;
    for (int i = 0; i < n - 1 - control; ++i) divisor_c *= p;
    int divisor_t = 1;
    for (int i = 0; i < n - 1 - target; ++i) divisor_t *= p;

    for (int idx = 0; idx < num_elements; ++idx) {
        const int x_c = (idx / divisor_c) % p;
        const int x_t = (idx / divisor_t) % p;
        const int source_x_t = (x_t - x_c + p) % p;
        const int source_idx = idx - x_t * divisor_t + source_x_t * divisor_t;
        result[static_cast<size_t>(idx)] = state[static_cast<size_t>(source_idx)];
    }
    return result;
}

ComplexVec reference_apply_clifford_CZ(
    const ComplexVec& state, int control, int target, int n, int p) {
    int num_elements = static_cast<int>(state.size());
    ComplexVec result = state;

    int divisor_c = 1;
    for (int i = 0; i < n - 1 - control; ++i) divisor_c *= p;
    int divisor_t = 1;
    for (int i = 0; i < n - 1 - target; ++i) divisor_t *= p;

    if (p == 2) {
        for (int idx = 0; idx < num_elements; ++idx) {
            const int x_c = (idx / divisor_c) % 2;
            const int x_t = (idx / divisor_t) % 2;
            if (x_c == 1 && x_t == 1) result[static_cast<size_t>(idx)] *= -1.0;
        }
    } else if (p == 3) {
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        for (int idx = 0; idx < num_elements; ++idx) {
            const int x_c = (idx / divisor_c) % 3;
            const int x_t = (idx / divisor_t) % 3;
            const int exponent = (x_c * x_t) % 3;
            if (exponent == 1) {
                result[static_cast<size_t>(idx)] *= omega;
            } else if (exponent == 2) {
                result[static_cast<size_t>(idx)] *= omega * omega;
            }
        }
    }
    return result;
}

ComplexVec reference_evaluate_coeffs_on_subspace(
    const PolyCoeffs& coeffs,
    int n_orig,
    int p,
    std::span<const int64_t> x0_translation,
    const IntMatrix& W_basis) {

    const int k_dim = static_cast<int>(W_basis.size());
    int output_len = 1;
    for (int i = 0; i < n_orig; ++i) output_len *= p;
    ComplexVec output(static_cast<size_t>(output_len), {0.0, 0.0});

    int num_y_points = 1;
    for (int i = 0; i < k_dim; ++i) num_y_points *= p;

    for (int y_idx = 0; y_idx < num_y_points; ++y_idx) {
        std::vector<int64_t> y_point(static_cast<size_t>(k_dim));
        int temp = y_idx;
        for (int i = k_dim - 1; i >= 0; --i) {
            y_point[static_cast<size_t>(i)] = temp % p;
            temp /= p;
        }

        std::vector<int64_t> x_orig(static_cast<size_t>(n_orig));
        for (int j = 0; j < n_orig; ++j) {
            int64_t sum_val = 0;
            for (int i = 0; i < k_dim; ++i) {
                sum_val += y_point[static_cast<size_t>(i)] *
                           W_basis[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
            x_orig[static_cast<size_t>(j)] =
                ((x0_translation[j] + sum_val) % p + p) % p;
        }

        const double q_val = evaluate_poly_at_point(coeffs, y_point, k_dim, p);
        const auto phase = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi * q_val));

        if (n_orig > 0) {
            const int idx = lex_index(x_orig, n_orig, p);
            output[static_cast<size_t>(idx)] = phase;
        } else {
            output[0] = phase;
        }
    }
    return output;
}

}  // namespace

TEST_CASE("apply_clifford_H matches its reference implementation") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 801}, {4, 2, 803}, {5, 2, 805},
        {2, 3, 807}, {3, 3, 809}, {4, 3, 811},
    };
    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        for (int q = 0; q < c.n; ++q) {
            INFO("n=" << c.n << " p=" << c.p << " q=" << q);
            require_close(apply_clifford_H(state, q, c.n, c.p),
                          reference_apply_clifford_H(state, q, c.n, c.p));
        }
    }
}

TEST_CASE("apply_clifford_S matches its reference implementation") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 821}, {4, 2, 823}, {5, 2, 825},
        {2, 3, 827}, {3, 3, 829}, {4, 3, 831},
    };
    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        for (int q = 0; q < c.n; ++q) {
            INFO("n=" << c.n << " p=" << c.p << " q=" << q);
            require_close(apply_clifford_S(state, q, c.n, c.p),
                          reference_apply_clifford_S(state, q, c.n, c.p));
        }
    }
}

TEST_CASE("apply_clifford_CX matches its reference implementation") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 841}, {4, 2, 843}, {3, 3, 845}, {4, 3, 847},
    };
    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        for (int ctrl = 0; ctrl < c.n; ++ctrl) {
            for (int tgt = 0; tgt < c.n; ++tgt) {
                if (ctrl == tgt) continue;
                INFO("n=" << c.n << " p=" << c.p
                          << " ctrl=" << ctrl << " tgt=" << tgt);
                require_close(
                    apply_clifford_CX(state, ctrl, tgt, c.n, c.p),
                    reference_apply_clifford_CX(state, ctrl, tgt, c.n, c.p));
            }
        }
    }
}

TEST_CASE("apply_clifford_CZ matches its reference implementation") {
    struct Case { int n; int p; uint64_t seed; };
    const Case cases[] = {
        {3, 2, 861}, {4, 2, 863}, {3, 3, 865}, {4, 3, 867},
    };
    for (const auto& c : cases) {
        const auto state = random_state(c.n, c.p, c.seed);
        for (int ctrl = 0; ctrl < c.n; ++ctrl) {
            for (int tgt = 0; tgt < c.n; ++tgt) {
                if (ctrl == tgt) continue;
                INFO("n=" << c.n << " p=" << c.p
                          << " ctrl=" << ctrl << " tgt=" << tgt);
                require_close(
                    apply_clifford_CZ(state, ctrl, tgt, c.n, c.p),
                    reference_apply_clifford_CZ(state, ctrl, tgt, c.n, c.p));
            }
        }
    }
}

TEST_CASE("evaluate_coeffs_on_subspace matches its reference implementation") {
    std::mt19937_64 rng(881);
    for (int p : {2, 3}) {
        for (int n : {2, 3, 4}) {
            for (int k_dim : {1, 2, std::min(n, 3)}) {
                auto coeffs = generate_random_coeffs(k_dim, p, rng);

                std::vector<int64_t> x0(static_cast<size_t>(n));
                std::uniform_int_distribution<int> dist(0, p - 1);
                for (auto& v : x0) v = dist(rng);

                IntMatrix W(static_cast<size_t>(k_dim),
                            std::vector<int64_t>(static_cast<size_t>(n), 0));
                for (int j = 0; j < k_dim && j < n; ++j) W[j][j] = 1;

                INFO("p=" << p << " n=" << n << " k_dim=" << k_dim);
                require_close(
                    evaluate_coeffs_on_subspace(coeffs, n, p, x0, W),
                    reference_evaluate_coeffs_on_subspace(
                        coeffs, n, p, x0, W));
            }
        }
    }
}

TEST_CASE("evaluate_coeffs_on_subspace yields a phase vector on its support") {
    // The polynomial-evaluation kernel encodes a pure-phase amplitude at the
    // support points of the affine subspace and zero elsewhere. After
    // normalization it should be a unit-norm state, and every non-zero entry
    // should have unit magnitude.
    std::mt19937_64 rng(11);
    for (int p : {2, 3}) {
        for (int n : {2, 3, 4}) {
            const int k = std::min(n, 3);
            auto coeffs = generate_random_coeffs(k, p, rng);

            std::vector<int64_t> x0(static_cast<size_t>(n), 0);
            IntMatrix W(static_cast<size_t>(k),
                        std::vector<int64_t>(static_cast<size_t>(n), 0));
            for (int j = 0; j < k; ++j) W[j][j] = 1;

            const auto vec = evaluate_coeffs_on_subspace(coeffs, n, p, x0, W);

            int support_count = 0;
            for (const auto& v : vec) {
                const double mag = std::abs(v);
                if (mag > 1e-12) {
                    ++support_count;
                    REQUIRE_THAT(mag, Catch::Matchers::WithinAbs(1.0, 1e-12));
                }
            }

            int expected_support = 1;
            for (int i = 0; i < k; ++i) expected_support *= p;
            REQUIRE(support_count == expected_support);
        }
    }
}
