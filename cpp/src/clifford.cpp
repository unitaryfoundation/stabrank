#include "stabrank/clifford.hpp"

#include <cmath>
#include <complex>
#include <numbers>

namespace stabrank {

namespace {

// Stride for a given qudit (lex_index reads the last digit fastest, so the
// last qudit has stride 1, the second-to-last has stride p, and so on).
size_t qudit_stride(int qudit, int n, int p) {
    size_t stride = 1;
    for (int i = 0; i < n - 1 - qudit; ++i) {
        stride *= static_cast<size_t>(p);
    }
    return stride;
}

}  // namespace

ComplexVec apply_clifford_H(const ComplexVec& state, int qudit, int n, int p) {
    const size_t num_elements = state.size();
    // Match the original behavior of returning an all-zero result when the
    // qudit dimension falls outside the supported {2, 3} cases.
    ComplexVec result(num_elements, std::complex<double>{0.0, 0.0});

    const size_t stride = qudit_stride(qudit, n, p);
    const size_t block_span = stride * static_cast<size_t>(p);

    if (p == 2) {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        for (size_t block_start = 0; block_start < num_elements;
             block_start += block_span) {
            for (size_t inner = 0; inner < stride; ++inner) {
                const size_t i0 = block_start + inner;
                const size_t i1 = i0 + stride;
                const auto a0 = state[i0];
                const auto a1 = state[i1];
                result[i0] = (a0 + a1) * inv_sqrt2;
                result[i1] = (a0 - a1) * inv_sqrt2;
            }
        }
    } else if (p == 3) {
        const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        const auto omega2 = omega * omega;
        for (size_t block_start = 0; block_start < num_elements;
             block_start += block_span) {
            for (size_t inner = 0; inner < stride; ++inner) {
                const size_t i0 = block_start + inner;
                const size_t i1 = i0 + stride;
                const size_t i2 = i1 + stride;
                const auto a0 = state[i0];
                const auto a1 = state[i1];
                const auto a2 = state[i2];
                result[i0] = (a0 + a1 + a2) * inv_sqrt3;
                result[i1] = (a0 + a1 * omega + a2 * omega2) * inv_sqrt3;
                result[i2] = (a0 + a1 * omega2 + a2 * omega) * inv_sqrt3;
            }
        }
    }
    return result;
}

ComplexVec apply_clifford_S(const ComplexVec& state, int qudit, int n, int p) {
    ComplexVec result = state;

    const size_t stride = qudit_stride(qudit, n, p);
    const size_t block_span = stride * static_cast<size_t>(p);

    if (p == 2) {
        constexpr std::complex<double> i_phase{0.0, 1.0};
        for (size_t block_start = 0; block_start < state.size();
             block_start += block_span) {
            // Only the digit-1 slab picks up a phase.
            const size_t base = block_start + stride;
            for (size_t inner = 0; inner < stride; ++inner) {
                result[base + inner] *= i_phase;
            }
        }
    } else if (p == 3) {
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        for (size_t block_start = 0; block_start < state.size();
             block_start += block_span) {
            // Digits 1 and 2 each pick up the same omega factor.
            for (size_t digit = 1; digit < static_cast<size_t>(p); ++digit) {
                const size_t base = block_start + digit * stride;
                for (size_t inner = 0; inner < stride; ++inner) {
                    result[base + inner] *= omega;
                }
            }
        }
    }
    return result;
}

ComplexVec apply_clifford_CX(const ComplexVec& state, int control, int target, int n, int p) {
    int num_elements = static_cast<int>(state.size());
    ComplexVec result(num_elements);

    int divisor_c = 1;
    for (int i = 0; i < n - 1 - control; ++i) divisor_c *= p;

    int divisor_t = 1;
    for (int i = 0; i < n - 1 - target; ++i) divisor_t *= p;

    for (int idx = 0; idx < num_elements; ++idx) {
        int x_c = (idx / divisor_c) % p;
        int x_t = (idx / divisor_t) % p;

        int source_x_t = (x_t - x_c + p) % p;
        int source_idx = idx - x_t * divisor_t + source_x_t * divisor_t;
        result[idx] = state[source_idx];
    }
    return result;
}

ComplexVec apply_clifford_CZ(const ComplexVec& state, int control, int target, int n, int p) {
    int num_elements = static_cast<int>(state.size());
    ComplexVec result = state;

    int divisor_c = 1;
    for (int i = 0; i < n - 1 - control; ++i) divisor_c *= p;
    int divisor_t = 1;
    for (int i = 0; i < n - 1 - target; ++i) divisor_t *= p;

    if (p == 2) {
        for (int idx = 0; idx < num_elements; ++idx) {
            int x_c = (idx / divisor_c) % 2;
            int x_t = (idx / divisor_t) % 2;
            if (x_c == 1 && x_t == 1) {
                result[static_cast<size_t>(idx)] *= -1.0;
            }
        }
    } else if (p == 3) {
        // Same branched structure as before; only ~2/3 of amplitudes need a
        // phase, and copying state then multiplying in-place beats reading,
        // multiplying, and writing every amplitude unconditionally. The
        // single useful tweak is hoisting omega^2 out of the inner loop.
        const auto omega = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi / 3.0));
        const auto omega2 = omega * omega;
        for (int idx = 0; idx < num_elements; ++idx) {
            int x_c = (idx / divisor_c) % 3;
            int x_t = (idx / divisor_t) % 3;
            int exponent = (x_c * x_t) % 3;
            if (exponent == 1) {
                result[static_cast<size_t>(idx)] *= omega;
            } else if (exponent == 2) {
                result[static_cast<size_t>(idx)] *= omega2;
            }
        }
    }
    return result;
}

ComplexVec apply_random_single_gate_clifford(
    const ComplexVec& state,
    int n,
    int p,
    std::mt19937_64& rng) {

    if (n == 0) return state;

    std::uniform_int_distribution<int> gate_dist(0, 2);
    std::uniform_int_distribution<int> qudit_dist(0, n - 1);

    int gate = gate_dist(rng);
    if (n == 1 && gate == 2) {
        std::uniform_int_distribution<int> gate_dist_1q(0, 1);
        gate = gate_dist_1q(rng);
    }

    if (gate == 0) {
        int q = qudit_dist(rng);
        return apply_clifford_H(state, q, n, p);
    } else if (gate == 1) {
        int q = qudit_dist(rng);
        return apply_clifford_S(state, q, n, p);
    } else {
        int c = qudit_dist(rng);
        int t = qudit_dist(rng);
        while (t == c) {
            t = qudit_dist(rng);
        }
        return apply_clifford_CX(state, c, t, n, p);
    }
}

}  // namespace stabrank
