#include "stabrank/polynomial.hpp"
#include "stabrank/linalg.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace stabrank {

double evaluate_poly_at_point(
    const PolyCoeffs& coeffs,
    std::span<const int64_t> point,
    int k_dim,
    int p) {

    double poly_value = coeffs.alpha;

    // Linear terms: sum(c_j0_lin[i] * y_i) / p
    if (k_dim > 0) {
        double lin_sum = 0.0;
        for (int i = 0; i < k_dim; ++i) {
            lin_sum += static_cast<double>(point[i]) *
                       static_cast<double>(coeffs.c_j0_lin[i]);
        }
        poly_value += lin_sum / static_cast<double>(p);
    }

    // Square terms (p >= 3 only): sum(c_j0_qs[i] * y_i^2) / p
    if (p >= 3 && k_dim > 0) {
        double sq_sum = 0.0;
        for (int i = 0; i < k_dim; ++i) {
            double yi = static_cast<double>(point[i]);
            sq_sum += (yi * yi) * static_cast<double>(coeffs.c_j0_qs[i]);
        }
        poly_value += sq_sum / static_cast<double>(p);
    }

    // Mixed quadratic terms: sum(c_j0_qm[idx] * y_s * y_t) / p
    int qm_idx = 0;
    for (int s = 0; s < k_dim; ++s) {
        for (int t = s + 1; t < k_dim; ++t) {
            double term = static_cast<double>(point[s]) *
                          static_cast<double>(point[t]) *
                          static_cast<double>(coeffs.c_j0_qm[qm_idx]);
            poly_value += term / static_cast<double>(p);
            ++qm_idx;
        }
    }

    // Second-level linear (p == 2 only): sum(c_j1_lin[i] * y_i) / 4
    if (p == 2 && k_dim > 0) {
        double j1_sum = 0.0;
        for (int i = 0; i < k_dim; ++i) {
            j1_sum += static_cast<double>(point[i]) *
                      static_cast<double>(coeffs.c_j1_lin[i]);
        }
        poly_value += j1_sum / 4.0;
    }

    // Reduce mod 1.0 to [0, 1)
    double res = std::fmod(poly_value, 1.0);
    if (res < 0.0) res += 1.0;
    return res;
}

namespace {

// All non-alpha terms in the polynomial encoding live over a common rational
// denominator D: D = 4 for p == 2 (because of the c_j1_lin / 4 contribution)
// and D = p otherwise. Pre-build a phase look-up table keyed on the integer
// numerator mod D so the inner loop replaces std::fmod + std::exp with a
// single LUT load. The constant alpha offset is folded into every entry so
// no extra multiply is needed in the hot path.
std::vector<std::complex<double>> build_phase_table(double alpha, int p) {
    const int D = (p == 2) ? 4 : p;
    std::vector<std::complex<double>> table(static_cast<size_t>(D));
    const auto base = std::exp(
        std::complex<double>(0.0, 2.0 * std::numbers::pi * alpha));
    const double inv_d = 1.0 / static_cast<double>(D);
    for (int k = 0; k < D; ++k) {
        const auto phase = std::exp(
            std::complex<double>(
                0.0, 2.0 * std::numbers::pi * static_cast<double>(k) * inv_d));
        table[static_cast<size_t>(k)] = base * phase;
    }
    return table;
}

}  // namespace

ComplexVec evaluate_coeffs_on_subspace(
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

    // Common denominator setup: D = 4 when p == 2 (c_j1_lin / 4 dominates),
    // D = p otherwise. The /p terms need an extra factor of D / p when p == 2.
    const int D = (p == 2) ? 4 : p;
    const int64_t scale_p = (p == 2) ? 2 : 1;

    const auto phase_table = build_phase_table(coeffs.alpha, p);

    // Reused per-iteration buffers: a base-p counter for y_point and an
    // x_orig output digit vector. Allocating once outside the loop keeps
    // the fast path allocation-free.
    std::vector<int64_t> y_point(static_cast<size_t>(k_dim), 0);
    std::vector<int64_t> x_orig(static_cast<size_t>(n_orig), 0);

    for (int y_idx = 0; y_idx < num_y_points; ++y_idx) {
        // Accumulate the polynomial value as an integer numerator over D.
        int64_t numerator = 0;

        // Linear / p
        for (int i = 0; i < k_dim; ++i) {
            numerator += scale_p * coeffs.c_j0_lin[static_cast<size_t>(i)] *
                         y_point[static_cast<size_t>(i)];
        }
        // Square / p (qutrit and higher)
        if (p >= 3) {
            for (int i = 0; i < k_dim; ++i) {
                const int64_t y = y_point[static_cast<size_t>(i)];
                numerator += scale_p * coeffs.c_j0_qs[static_cast<size_t>(i)] *
                             y * y;
            }
        }
        // Mixed quadratic / p
        size_t qm_idx = 0;
        for (int s = 0; s < k_dim; ++s) {
            const int64_t y_s = y_point[static_cast<size_t>(s)];
            for (int t = s + 1; t < k_dim; ++t) {
                numerator += scale_p * coeffs.c_j0_qm[qm_idx] * y_s *
                             y_point[static_cast<size_t>(t)];
                ++qm_idx;
            }
        }
        // Second-level linear / 4 (qubits only). D / 4 == 1 in that case.
        if (p == 2) {
            for (int i = 0; i < k_dim; ++i) {
                numerator += coeffs.c_j1_lin[static_cast<size_t>(i)] *
                             y_point[static_cast<size_t>(i)];
            }
        }

        const int phase_idx = static_cast<int>(((numerator % D) + D) % D);
        const auto phase = phase_table[static_cast<size_t>(phase_idx)];

        // x = x0 + W^T * y (mod p)
        for (int j = 0; j < n_orig; ++j) {
            int64_t sum_val = 0;
            for (int i = 0; i < k_dim; ++i) {
                sum_val += y_point[static_cast<size_t>(i)] *
                           W_basis[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
            x_orig[static_cast<size_t>(j)] =
                ((x0_translation[j] + sum_val) % p + p) % p;
        }

        if (n_orig > 0) {
            const int idx = lex_index(x_orig, n_orig, p);
            output[static_cast<size_t>(idx)] = phase;
        } else {
            output[0] = phase;
        }

        // Increment y_point as a base-p counter (LSB at index k_dim - 1).
        for (int i = k_dim - 1; i >= 0; --i) {
            ++y_point[static_cast<size_t>(i)];
            if (y_point[static_cast<size_t>(i)] < p) break;
            y_point[static_cast<size_t>(i)] = 0;
        }
    }

    return output;
}

PolyCoeffs generate_random_coeffs(int k_dim, int p, std::mt19937_64& rng) {
    PolyCoeffs coeffs;
    coeffs.alpha = 0.0;

    std::uniform_int_distribution<int64_t> dist(0, p - 1);

    coeffs.c_j0_lin.resize(k_dim);
    for (auto& c : coeffs.c_j0_lin) c = dist(rng);

    int qm_len = (k_dim >= 2) ? k_dim * (k_dim - 1) / 2 : 0;
    coeffs.c_j0_qm.resize(qm_len);
    for (auto& c : coeffs.c_j0_qm) c = dist(rng);

    coeffs.c_j0_qs.resize(k_dim, 0);
    if (p >= 3) {
        for (auto& c : coeffs.c_j0_qs) c = dist(rng);
    }

    coeffs.c_j1_lin.resize(k_dim, 0);
    if (p == 2) {
        for (auto& c : coeffs.c_j1_lin) c = dist(rng);
    }

    return coeffs;
}

}  // namespace stabrank
