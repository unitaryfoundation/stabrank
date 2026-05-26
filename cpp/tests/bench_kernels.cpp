// C++-level micro-benchmarks for the stabrank native kernels.
//
// These exist on top of `scripts/benchmark_core.py` because the Python
// harness measures the full nanobind-bound call path, which can mask small
// kernel-level deltas (especially for the qutrit hot path). This binary
// links directly against `stabrank_core_lib` and times the kernels with
// no binding overhead.
//
// Run with: `./stabrank_bench` (built via the cpp_bench CMake target).

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "stabrank/clifford.hpp"
#include "stabrank/linalg.hpp"
#include "stabrank/pauli.hpp"
#include "stabrank/polynomial.hpp"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace stabrank;

namespace {

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

std::vector<ComplexVec> random_basis(int n, int p, int count, uint64_t seed) {
    std::vector<ComplexVec> basis;
    basis.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        basis.push_back(random_state(n, p, seed + static_cast<uint64_t>(i)));
    }
    return basis;
}

}  // namespace

TEST_CASE("Pauli kernels - qubit n=8") {
    constexpr int n = 8;
    constexpr int p = 2;
    const auto state = random_state(n, p, 1001);

    BENCHMARK("apply_X qubit n=8") {
        return apply_X(state, 3, n, p);
    };
    BENCHMARK("apply_Z qubit n=8") {
        return apply_Z(state, 3, n, p);
    };
    BENCHMARK("apply_Y qubit n=8") {
        return apply_Y(state, 3, n, p);
    };
}

TEST_CASE("Pauli kernels - qutrit n=6") {
    constexpr int n = 6;
    constexpr int p = 3;
    const auto state = random_state(n, p, 1003);

    BENCHMARK("apply_X qutrit n=6") {
        return apply_X(state, 2, n, p);
    };
    BENCHMARK("apply_Z qutrit n=6") {
        return apply_Z(state, 2, n, p);
    };
    BENCHMARK("apply_Y qutrit n=6") {
        return apply_Y(state, 2, n, p);
    };
}

TEST_CASE("Pauli kernels - qutrit n=7") {
    constexpr int n = 7;
    constexpr int p = 3;
    const auto state = random_state(n, p, 1005);

    BENCHMARK("apply_X qutrit n=7") {
        return apply_X(state, 2, n, p);
    };
    BENCHMARK("apply_Z qutrit n=7") {
        return apply_Z(state, 2, n, p);
    };
    BENCHMARK("apply_Y qutrit n=7") {
        return apply_Y(state, 2, n, p);
    };
}

TEST_CASE("Pauli string projector - qubit n=8") {
    constexpr int n = 8;
    constexpr int p = 2;
    auto state = random_state(n, p, 2001);

    std::mt19937_64 rng(7);
    BENCHMARK("apply_random_pauli_string qubit n=8") {
        return apply_random_pauli_string(state, n, p, rng, false);
    };
}

TEST_CASE("Pauli string projector - qutrit n=6") {
    constexpr int n = 6;
    constexpr int p = 3;
    auto state = random_state(n, p, 2003);

    std::mt19937_64 rng(11);
    BENCHMARK("apply_random_pauli_string qutrit n=6") {
        return apply_random_pauli_string(state, n, p, rng, false);
    };
}

TEST_CASE("Pauli string projector - qutrit n=7") {
    constexpr int n = 7;
    constexpr int p = 3;
    auto state = random_state(n, p, 2005);

    std::mt19937_64 rng(13);
    BENCHMARK("apply_random_pauli_string qutrit n=7") {
        return apply_random_pauli_string(state, n, p, rng, false);
    };
}

TEST_CASE("Clifford kernels - qubit n=8") {
    constexpr int n = 8;
    constexpr int p = 2;
    const auto state = random_state(n, p, 3001);

    BENCHMARK("apply_clifford_H qubit n=8") {
        return apply_clifford_H(state, 3, n, p);
    };
    BENCHMARK("apply_clifford_S qubit n=8") {
        return apply_clifford_S(state, 3, n, p);
    };
    BENCHMARK("apply_clifford_CX qubit n=8") {
        return apply_clifford_CX(state, 0, 3, n, p);
    };
    BENCHMARK("apply_clifford_CZ qubit n=8") {
        return apply_clifford_CZ(state, 0, 3, n, p);
    };
}

TEST_CASE("Clifford kernels - qutrit n=6") {
    constexpr int n = 6;
    constexpr int p = 3;
    const auto state = random_state(n, p, 3003);

    BENCHMARK("apply_clifford_H qutrit n=6") {
        return apply_clifford_H(state, 2, n, p);
    };
    BENCHMARK("apply_clifford_S qutrit n=6") {
        return apply_clifford_S(state, 2, n, p);
    };
    BENCHMARK("apply_clifford_CX qutrit n=6") {
        return apply_clifford_CX(state, 0, 2, n, p);
    };
    BENCHMARK("apply_clifford_CZ qutrit n=6") {
        return apply_clifford_CZ(state, 0, 2, n, p);
    };
}

TEST_CASE("Clifford kernels - qutrit n=7") {
    constexpr int n = 7;
    constexpr int p = 3;
    const auto state = random_state(n, p, 3005);

    BENCHMARK("apply_clifford_H qutrit n=7") {
        return apply_clifford_H(state, 2, n, p);
    };
    BENCHMARK("apply_clifford_S qutrit n=7") {
        return apply_clifford_S(state, 2, n, p);
    };
    BENCHMARK("apply_clifford_CX qutrit n=7") {
        return apply_clifford_CX(state, 0, 2, n, p);
    };
    BENCHMARK("apply_clifford_CZ qutrit n=7") {
        return apply_clifford_CZ(state, 0, 2, n, p);
    };
}

TEST_CASE("Polynomial subspace evaluator") {
    std::mt19937_64 rng(5005);

    {
        constexpr int n = 7;
        constexpr int p = 2;
        constexpr int k = 7;
        auto coeffs = generate_random_coeffs(k, p, rng);
        std::vector<int64_t> x0(n, 0);
        IntMatrix W(k, std::vector<int64_t>(n, 0));
        for (int j = 0; j < k; ++j) W[j][j] = 1;

        BENCHMARK("evaluate_coeffs_on_subspace qubit n=7 k=7") {
            return evaluate_coeffs_on_subspace(coeffs, n, p, x0, W);
        };
    }

    {
        constexpr int n = 6;
        constexpr int p = 3;
        constexpr int k = 4;
        auto coeffs = generate_random_coeffs(k, p, rng);
        std::vector<int64_t> x0(n, 0);
        IntMatrix W(k, std::vector<int64_t>(n, 0));
        for (int j = 0; j < k; ++j) W[j][j] = 1;

        BENCHMARK("evaluate_coeffs_on_subspace qutrit n=6 k=4") {
            return evaluate_coeffs_on_subspace(coeffs, n, p, x0, W);
        };
    }

    {
        constexpr int n = 7;
        constexpr int p = 3;
        constexpr int k = 4;
        auto coeffs = generate_random_coeffs(k, p, rng);
        std::vector<int64_t> x0(n, 0);
        IntMatrix W(k, std::vector<int64_t>(n, 0));
        for (int j = 0; j < k; ++j) W[j][j] = 1;

        BENCHMARK("evaluate_coeffs_on_subspace qutrit n=7 k=4") {
            return evaluate_coeffs_on_subspace(coeffs, n, p, x0, W);
        };
    }
}

TEST_CASE("Least squares - direct solve") {
    {
        constexpr int n = 7;
        constexpr int p = 2;
        const auto target = random_state(n, p, 6001);
        const auto basis = random_basis(n, p, /*count=*/8, 6100);
        BENCHMARK("least_squares_solve direct qubit n=7 k=8") {
            return least_squares_solve(target, basis, 1e-5, 1e-8);
        };
    }
    {
        constexpr int n = 6;
        constexpr int p = 3;
        const auto target = random_state(n, p, 6003);
        const auto basis = random_basis(n, p, /*count=*/8, 6300);
        BENCHMARK("least_squares_solve direct qutrit n=6 k=8") {
            return least_squares_solve(target, basis, 1e-5, 1e-8);
        };
    }
    {
        constexpr int n = 7;
        constexpr int p = 3;
        const auto target = random_state(n, p, 6005);
        const auto basis = random_basis(n, p, /*count=*/8, 6500);
        BENCHMARK("least_squares_solve direct qutrit n=7 k=8") {
            return least_squares_solve(target, basis, 1e-5, 1e-8);
        };
    }
    {
        // Scaling-only case: dim 3^8 = 6561. Single bench point so a future
        // optimization PR can see how the solver behaves at the next-up
        // qutrit size without paying for a full set of n=8 cases here.
        constexpr int n = 8;
        constexpr int p = 3;
        const auto target = random_state(n, p, 6007);
        const auto basis = random_basis(n, p, /*count=*/8, 6700);
        BENCHMARK("least_squares_solve direct qutrit n=8 k=8") {
            return least_squares_solve(target, basis, 1e-5, 1e-8);
        };
    }
}

TEST_CASE("Least squares - workspace path mimicking SA inner loop") {
    // The SA inner loop pattern: keep the workspace alive, mutate one column,
    // resolve. This is the path the next optimization step will attack, so
    // having a stable benchmark on it is important.
    {
        constexpr int n = 7;
        constexpr int p = 2;
        constexpr int k = 8;
        const auto target = random_state(n, p, 7001);
        const auto basis = random_basis(n, p, k, 7100);
        const auto replacement = random_state(n, p, 7200);

        auto workspace = make_least_squares_workspace(target, k);
        for (int j = 0; j < k; ++j) {
            set_least_squares_basis_column(workspace, j, basis[static_cast<size_t>(j)]);
        }

        BENCHMARK("least_squares_solve workspace qubit n=7 k=8") {
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };

        BENCHMARK("least_squares update one column qubit n=7 k=8") {
            set_least_squares_basis_column(workspace, 0, replacement);
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };
    }
    {
        constexpr int n = 6;
        constexpr int p = 3;
        constexpr int k = 8;
        const auto target = random_state(n, p, 7003);
        const auto basis = random_basis(n, p, k, 7300);
        const auto replacement = random_state(n, p, 7400);

        auto workspace = make_least_squares_workspace(target, k);
        for (int j = 0; j < k; ++j) {
            set_least_squares_basis_column(workspace, j, basis[static_cast<size_t>(j)]);
        }

        BENCHMARK("least_squares_solve workspace qutrit n=6 k=8") {
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };

        BENCHMARK("least_squares update one column qutrit n=6 k=8") {
            set_least_squares_basis_column(workspace, 0, replacement);
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };
    }
    {
        constexpr int n = 7;
        constexpr int p = 3;
        constexpr int k = 8;
        const auto target = random_state(n, p, 7005);
        const auto basis = random_basis(n, p, k, 7500);
        const auto replacement = random_state(n, p, 7600);

        auto workspace = make_least_squares_workspace(target, k);
        for (int j = 0; j < k; ++j) {
            set_least_squares_basis_column(workspace, j, basis[static_cast<size_t>(j)]);
        }

        BENCHMARK("least_squares_solve workspace qutrit n=7 k=8") {
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };

        BENCHMARK("least_squares update one column qutrit n=7 k=8") {
            set_least_squares_basis_column(workspace, 0, replacement);
            return least_squares_solve(workspace, 1e-5, 1e-8);
        };
    }
}
