#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "stabrank/linalg.hpp"
#include "stabrank/pauli.hpp"
#include "stabrank/polynomial.hpp"
#include "stabrank/sa_engine.hpp"
#include "stabrank/types.hpp"
#include "stabrank/clifford.hpp"
#include "stabrank/fidelity.hpp"

#include <complex>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Helpers to convert between numpy arrays and C++ vectors.
static stabrank::ComplexVec ndarray_to_complexvec(
    nb::ndarray<std::complex<double>, nb::ndim<1>> arr) {
    stabrank::ComplexVec vec(arr.shape(0));
    for (size_t i = 0; i < arr.shape(0); ++i) {
        vec[i] = arr(i);
    }
    return vec;
}

static nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>
complexvec_to_ndarray(const stabrank::ComplexVec& vec) {
    size_t n = vec.size();
    auto* data = new std::complex<double>[n];
    std::copy(vec.begin(), vec.end(), data);
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<std::complex<double>*>(p);
    });
    return nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>(
        data, {n}, owner);
}

static size_t checked_dimension(int n_orig, int p_prime) {
    size_t dim = 1;
    const auto base = static_cast<size_t>(p_prime);
    for (int i = 0; i < n_orig; ++i) {
        if (dim > std::numeric_limits<size_t>::max() / base) {
            throw nb::value_error("p_prime ** n_orig is too large.");
        }
        dim *= base;
    }
    return dim;
}

static bool is_probability(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

NB_MODULE(stabrank_core, m) {
    m.doc() = "C++ accelerated stabrank core library";

    // --- run_sa_pauli_expansion ---
    m.def("run_sa_pauli_expansion",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> target_arr,
           int n_orig, int p_prime, int k_subset_size,
           nb::list initial_basis_list,
           double initial_temperature,
           double cooling_rate,
           int num_iterations_at_temp,
           double min_temperature,
           double rtol, double atol,
           double two_func_perturb_prob,
           double random_replace_prob,
           bool use_real_qubit_moves,
           double clifford_ratio,
           double early_exit_threshold,
           uint64_t seed,
           int num_chains,
           bool enable_tracing,
           int fixed_dimension) {

            if (n_orig < 1) {
                throw nb::value_error("n_orig must be positive.");
            }
            if (p_prime < 2) {
                throw nb::value_error("p_prime must be at least 2.");
            }
            if (k_subset_size < 1) {
                throw nb::value_error("k_subset_size must be positive.");
            }
            if (fixed_dimension < -1 || fixed_dimension > n_orig) {
                throw nb::value_error(
                    "fixed_dimension must be -1 or between 0 and n_orig.");
            }
            if (num_chains < 1) {
                throw nb::value_error("num_chains must be positive.");
            }
            if (!std::isfinite(initial_temperature) || initial_temperature <= 0.0) {
                throw nb::value_error("initial_temperature must be positive.");
            }
            if (!std::isfinite(min_temperature) || min_temperature < 0.0) {
                throw nb::value_error("min_temperature must be non-negative.");
            }
            if (!std::isfinite(cooling_rate) ||
                cooling_rate <= 0.0 || cooling_rate >= 1.0) {
                throw nb::value_error("cooling_rate must be in the open interval (0, 1).");
            }
            if (num_iterations_at_temp < 1) {
                throw nb::value_error("num_iterations_at_temp must be positive.");
            }
            if (!std::isfinite(rtol) || rtol < 0.0 ||
                !std::isfinite(atol) || atol < 0.0) {
                throw nb::value_error("rtol and atol must be non-negative finite values.");
            }
            if (!is_probability(two_func_perturb_prob) ||
                !is_probability(random_replace_prob) ||
                two_func_perturb_prob + random_replace_prob > 1.0) {
                throw nb::value_error(
                    "move probabilities must be in [0, 1] and sum to at most 1.");
            }
            if (!is_probability(clifford_ratio)) {
                throw nb::value_error("clifford_ratio must be in [0, 1].");
            }
            if (!std::isfinite(early_exit_threshold) || early_exit_threshold < 0.0) {
                throw nb::value_error("early_exit_threshold must be non-negative.");
            }

            const size_t expected_dim = checked_dimension(n_orig, p_prime);
            if (target_arr.shape(0) != expected_dim) {
                throw nb::value_error(
                    "target length must equal p_prime ** n_orig.");
            }

            // Convert target
            auto target = ndarray_to_complexvec(target_arr);

            // Convert initial basis
            std::vector<stabrank::ComplexVec> initial_basis;
            const size_t initial_basis_len = nb::len(initial_basis_list);
            if (initial_basis_len != 0 &&
                initial_basis_len != static_cast<size_t>(k_subset_size)) {
                throw nb::value_error(
                    "initial_basis must be empty or have k_subset_size entries.");
            }
            for (size_t i = 0; i < nb::len(initial_basis_list); ++i) {
                auto arr = nb::cast<nb::ndarray<std::complex<double>, nb::ndim<1>>>(
                    initial_basis_list[i]);
                if (arr.shape(0) != expected_dim) {
                    throw nb::value_error(
                        "every initial_basis entry must have the same length as target.");
                }
                initial_basis.push_back(ndarray_to_complexvec(arr));
            }

            // Build config
            stabrank::SAConfig config;
            config.initial_temperature = initial_temperature;
            config.cooling_rate = cooling_rate;
            config.num_iterations_at_temp = num_iterations_at_temp;
            config.min_temperature = min_temperature;
            config.rtol = rtol;
            config.atol = atol;
            config.two_func_perturb_prob = two_func_perturb_prob;
            config.random_replace_prob = random_replace_prob;
            config.use_real_qubit_moves = use_real_qubit_moves;
            config.clifford_ratio = clifford_ratio;
            config.early_exit_threshold = early_exit_threshold;
            config.num_chains = num_chains;
            config.enable_tracing = enable_tracing;
            config.fixed_dimension = fixed_dimension;

            // Generator function for random resets: generates a random
            // stabilizer state using polynomial coefficients.
            // Captured by value so it is safe to use from multiple threads.
            int gen_n = n_orig;
            int gen_p = p_prime;
            int gen_fixed_dim = fixed_dimension;
            auto generator = [gen_n, gen_p, gen_fixed_dim](std::mt19937_64& gen) -> stabrank::ComplexVec {
                int d = (gen_fixed_dim >= 0) ? gen_fixed_dim : gen_n;
                auto coeffs = stabrank::generate_random_coeffs(d, gen_p, gen);
                std::vector<int64_t> x0(gen_n, 0);
                stabrank::IntMatrix W(d, std::vector<int64_t>(gen_n, 0));
                for (int j = 0; j < d; ++j) W[j][j] = 1;
                auto func = stabrank::evaluate_coeffs_on_subspace(
                    coeffs, gen_n, gen_p, x0, W);
                
                if (gen_fixed_dim >= 0 && gen_n > 0) {
                    std::uniform_int_distribution<int> q_dist(0, gen_n - 1);
                    const bool can_entangle = gen_n >= 2;
                    std::uniform_int_distribution<int> op_dist(0, can_entangle ? 1 : 0);
                    int num_shuffles = 2 * gen_n * gen_n;
                    for (int i = 0; i < num_shuffles; ++i) {
                        int op = op_dist(gen);
                        if (op == 0 || !can_entangle) {
                            func = stabrank::apply_X(func, q_dist(gen), gen_n, gen_p);
                        } else {
                            int c = q_dist(gen);
                            int t = q_dist(gen);
                            while (t == c) t = q_dist(gen);
                            func = stabrank::apply_clifford_CX(func, c, t, gen_n, gen_p);
                        }
                    }
                }
                
                double norm = 0.0;
                for (auto& v : func) norm += std::norm(v);
                norm = std::sqrt(norm);
                if (norm > 1e-12) {
                    for (auto& v : func) v /= norm;
                }
                return func;
            };

            stabrank::SAResult result;
            {
                // Release the GIL so C++ threads can run freely.
                nb::gil_scoped_release release;
                result = stabrank::run_sa_pauli_expansion(
                    config, target, n_orig, p_prime, k_subset_size,
                    std::move(initial_basis), seed, generator);
            }

            // Convert results back to Python
            nb::list out_basis;
            for (auto& f : result.best_basis_funcs) {
                out_basis.append(complexvec_to_ndarray(f));
            }
            auto out_coeffs = complexvec_to_ndarray(result.best_lin_coeffs);

            nb::list out_trace;
            for (const auto& step : result.trace) {
                nb::dict d;
                d["iteration"] = step.iteration;
                d["temperature"] = step.temperature;
                d["current_cost"] = step.current_cost;
                d["best_cost"] = step.best_cost;
                d["accepted"] = step.accepted;
                d["move_type"] = step.move_type;
                nb::list k_vals;
                for (int k : step.k_values) {
                    k_vals.append(k);
                }
                d["k_values"] = k_vals;
                out_trace.append(d);
            }

            return nb::make_tuple(
                result.k, out_basis, out_coeffs,
                result.best_error, result.best_cost, out_trace);
        },
        "target"_a, "n_orig"_a, "p_prime"_a, "k_subset_size"_a,
        "initial_basis"_a,
        "initial_temperature"_a = 1.0,
        "cooling_rate"_a = 0.99,
        "num_iterations_at_temp"_a = 1000,
        "min_temperature"_a = 1e-5,
        "rtol"_a = 1e-5, "atol"_a = 1e-8,
        nb::arg("two_func_perturb_prob") = 0.1,
        nb::arg("random_replace_prob") = 0.01,
        nb::arg("use_real_qubit_moves") = false,
        nb::arg("clifford_ratio") = 0.0,
        nb::arg("early_exit_threshold") = 1e-9,
        "seed"_a = 42,
        "num_chains"_a = 1,
        "enable_tracing"_a = false,
        "fixed_dimension"_a = -1,
        "Run SA with Pauli expansion moves (C++ accelerated).");

    // --- least_squares_solve ---
    m.def("least_squares_solve",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> target_arr,
           nb::list basis_list,
           double rtol, double atol) {
            auto target = ndarray_to_complexvec(target_arr);
            std::vector<stabrank::ComplexVec> basis;
            for (size_t i = 0; i < nb::len(basis_list); ++i) {
                auto arr = nb::cast<nb::ndarray<std::complex<double>, nb::ndim<1>>>(
                    basis_list[i]);
                basis.push_back(ndarray_to_complexvec(arr));
            }
            auto result = stabrank::least_squares_solve(target, basis, rtol, atol);
            auto out_coeffs = complexvec_to_ndarray(result.coeffs);
            return nb::make_tuple(result.is_representable, out_coeffs,
                                  result.reconstruction_error);
        },
        "target"_a, "basis_funcs"_a, "rtol"_a = 1e-5, "atol"_a = 1e-8,
        "Solve least-squares for target = sum(c_i * basis_i).");

    // --- apply_random_pauli_string ---
    m.def("apply_random_pauli_string",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> state_arr,
           int n, int p, uint64_t seed, bool even_y_constraint) {
            auto state = ndarray_to_complexvec(state_arr);
            std::mt19937_64 rng(seed);
            auto [result, ops] = stabrank::apply_random_pauli_string(
                state, n, p, rng, even_y_constraint);
            auto out = complexvec_to_ndarray(result);
            std::string ops_str(ops.begin(), ops.end());
            return nb::make_tuple(out, ops_str);
        },
        "state"_a, "n"_a, "p"_a, "seed"_a = 42,
        "even_y_constraint"_a = false,
        "Apply a random Pauli string projector to a state vector.");

    // --- max_stabilizer_fidelity ---
    m.def("max_stabilizer_fidelity",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> target_arr,
           int n, int d) {
            auto target = ndarray_to_complexvec(target_arr);
            auto result = stabrank::max_stabilizer_fidelity(target, n, d);

            nb::dict out;
            out["f_max"] = result.f_max;
            out["extent_lb"] = result.extent_lb;
            out["total_states"] = result.total_states;
            out["elapsed_seconds"] = result.elapsed_seconds;

            nb::list f_per_k, states_per_k;
            for (size_t i = 0; i < result.f_max_per_k.size(); ++i) {
                f_per_k.append(result.f_max_per_k[i]);
                states_per_k.append(result.states_per_k[i]);
            }
            out["f_max_per_k"] = f_per_k;
            out["states_per_k"] = states_per_k;
            return out;
        },
        "target"_a, "n"_a, "d"_a = 3,
        "Exhaustive search for max |<phi|psi>|^2 over all stabilizer states.");
}
