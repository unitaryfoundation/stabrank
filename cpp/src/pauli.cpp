#include "stabrank/pauli.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace stabrank {
namespace {

size_t qudit_stride(int qudit, int n, int p) {
    size_t stride = 1;
    for (int i = 0; i < n - 1 - qudit; ++i) {
        stride *= static_cast<size_t>(p);
    }
    return stride;
}

std::vector<std::complex<double>> make_phase_table(int p) {
    std::vector<std::complex<double>> phases(static_cast<size_t>(p), {1.0, 0.0});
    const auto omega = std::exp(
        std::complex<double>(0.0, 2.0 * std::numbers::pi / static_cast<double>(p)));
    for (int i = 1; i < p; ++i) {
        phases[static_cast<size_t>(i)] = phases[static_cast<size_t>(i - 1)] * omega;
    }
    return phases;
}

void apply_x_into(const ComplexVec& source, ComplexVec& target, size_t stride, int p) {
    const size_t num_elements = source.size();
    const size_t block_span = stride * static_cast<size_t>(p);

    for (size_t block_start = 0; block_start < num_elements; block_start += block_span) {
        for (int target_digit = 0; target_digit < p; ++target_digit) {
            const size_t target_offset =
                block_start + static_cast<size_t>(target_digit) * stride;
            const int source_digit = target_digit == 0 ? p - 1 : target_digit - 1;
            const size_t source_offset =
                block_start + static_cast<size_t>(source_digit) * stride;
            std::copy_n(
                source.begin() + static_cast<std::ptrdiff_t>(source_offset),
                static_cast<std::ptrdiff_t>(stride),
                target.begin() + static_cast<std::ptrdiff_t>(target_offset));
        }
    }
}

void apply_z_into(
    const ComplexVec& source,
    ComplexVec& target,
    size_t stride,
    const std::vector<std::complex<double>>& phases) {

    const size_t num_elements = source.size();
    const size_t p = phases.size();
    const size_t block_span = stride * p;

    for (size_t block_start = 0; block_start < num_elements; block_start += block_span) {
        for (size_t digit = 0; digit < p; ++digit) {
            const auto phase = phases[digit];
            const size_t offset = block_start + digit * stride;
            for (size_t inner = 0; inner < stride; ++inner) {
                target[offset + inner] = source[offset + inner] * phase;
            }
        }
    }
}

void apply_y_into(
    const ComplexVec& source,
    ComplexVec& target,
    size_t stride,
    int p,
    const std::vector<std::complex<double>>& phases) {

    const size_t num_elements = source.size();
    const size_t block_span = stride * static_cast<size_t>(p);

    for (size_t block_start = 0; block_start < num_elements; block_start += block_span) {
        for (int target_digit = 0; target_digit < p; ++target_digit) {
            const int source_digit = target_digit == 0 ? p - 1 : target_digit - 1;
            const size_t target_offset =
                block_start + static_cast<size_t>(target_digit) * stride;
            const size_t source_offset =
                block_start + static_cast<size_t>(source_digit) * stride;
            const auto phase = phases[static_cast<size_t>(source_digit)];
            for (size_t inner = 0; inner < stride; ++inner) {
                target[target_offset + inner] = source[source_offset + inner] * phase;
            }
        }
    }
}

void apply_pauli_string_into(
    const ComplexVec& source,
    ComplexVec& target,
    const std::vector<char>& ops,
    const std::vector<size_t>& strides,
    int p,
    const std::vector<std::complex<double>>& phases) {

    const size_t num_elements = source.size();
    const size_t p_size = static_cast<size_t>(p);

    for (size_t source_idx = 0; source_idx < num_elements; ++source_idx) {
        size_t target_idx = source_idx;
        std::complex<double> phase = {1.0, 0.0};

        for (size_t qudit = 0; qudit < ops.size(); ++qudit) {
            const size_t stride = strides[qudit];
            const size_t digit = (source_idx / stride) % p_size;

            switch (ops[qudit]) {
                case 'X':
                    if (digit + 1 < p_size) {
                        target_idx += stride;
                    } else {
                        target_idx -= (p_size - 1) * stride;
                    }
                    break;
                case 'Y':
                    phase *= phases[digit];
                    if (digit + 1 < p_size) {
                        target_idx += stride;
                    } else {
                        target_idx -= (p_size - 1) * stride;
                    }
                    break;
                case 'Z':
                    phase *= phases[digit];
                    break;
                default: break;
            }
        }

        target[target_idx] = source[source_idx] * phase;
    }
}

}  // namespace

ComplexVec apply_X(const ComplexVec& state, int qudit, int n, int p) {
    ComplexVec result(state.size());
    apply_x_into(state, result, qudit_stride(qudit, n, p), p);
    return result;
}

ComplexVec apply_Z(const ComplexVec& state, int qudit, int n, int p) {
    ComplexVec result(state.size());
    apply_z_into(state, result, qudit_stride(qudit, n, p), make_phase_table(p));
    return result;
}

ComplexVec apply_Y(const ComplexVec& state, int qudit, int n, int p) {
    ComplexVec result(state.size());
    apply_y_into(
        state,
        result,
        qudit_stride(qudit, n, p),
        p,
        make_phase_table(p));
    return result;
}

std::pair<ComplexVec, std::vector<char>> apply_random_pauli_string(
    const ComplexVec& state,
    int n,
    int p,
    std::mt19937_64& rng,
    bool even_y_constraint) {

    static constexpr char ops[] = {'I', 'X', 'Y', 'Z'};
    std::uniform_int_distribution<int> op_dist(0, 3);

    std::vector<char> chosen_ops(n);

    // Rejection sampling for even-Y constraint
    constexpr int max_retries = 100;
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        int y_count = 0;
        for (int i = 0; i < n; ++i) {
            chosen_ops[i] = ops[op_dist(rng)];
            if (chosen_ops[i] == 'Y') ++y_count;
        }

        if (!even_y_constraint || p != 2 || y_count % 2 == 0) {
            break;
        }

        // On last attempt, force-fix parity
        if (attempt == max_retries - 1 && y_count % 2 != 0) {
            if (chosen_ops[n - 1] == 'Y') {
                chosen_ops[n - 1] = 'X';
            } else {
                chosen_ops[n - 1] = 'Y';
            }
        }
    }

    // Apply the operators to build P|psi>
    // Then compute the projector sum: |psi> + omega*P|psi> + omega^2*P^2|psi> + ...
    const auto phases = make_phase_table(p);
    std::vector<size_t> strides(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        strides[static_cast<size_t>(i)] = qudit_stride(i, n, p);
    }

    ComplexVec accumulated = state;  // starts as |psi>
    ComplexVec current = state;      // tracks P^k|psi>
    ComplexVec scratch(state.size());
    std::complex<double> phase = phases[1];

    for (int k = 0; k < p - 1; ++k) {
        apply_pauli_string_into(current, scratch, chosen_ops, strides, p, phases);
        current.swap(scratch);
        // accumulated += omega^(k+1) * current
        for (size_t j = 0; j < accumulated.size(); ++j) {
            accumulated[j] += phase * current[j];
        }
        phase *= phases[1];
    }

    return {accumulated, chosen_ops};
}

}  // namespace stabrank
