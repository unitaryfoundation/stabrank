#include "stabrank/sa_engine.hpp"
#include "stabrank/linalg.hpp"
#include "stabrank/pauli.hpp"
#include "stabrank/clifford.hpp"

#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace stabrank {

namespace {

bool normalize_state(ComplexVec& func, double min_norm) {
    double norm_sq = 0.0;
    for (const auto& value : func) {
        norm_sq += std::norm(value);
    }
    const double norm = std::sqrt(norm_sq);
    if (norm <= min_norm) {
        return false;
    }
    for (auto& value : func) {
        value /= norm;
    }
    return true;
}

}  // namespace

// ---- Single-chain SA worker (completely self-contained) ----
static SAResult run_single_chain(
    const SAConfig& config,
    const ComplexVec& target,
    int n_orig,
    int p_prime,
    int k_subset_size,
    std::vector<ComplexVec> initial_basis,
    uint64_t seed,
    std::function<ComplexVec(std::mt19937_64&)> generator_func,
    int chain_id,
    bool verbose) {

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::uniform_int_distribution<int> idx_dist(0, k_subset_size - 1);

    // If no initial basis was provided, generate one using the generator.
    if (initial_basis.empty() && generator_func) {
        initial_basis.reserve(k_subset_size);
        for (int i = 0; i < k_subset_size; ++i) {
            auto func = generator_func(rng);
            normalize_state(func, 1e-12);
            initial_basis.push_back(std::move(func));
        }
    }

    // Initial state
    auto current_funcs = initial_basis;
    auto ls_workspace = make_least_squares_workspace(target, k_subset_size);
    for (int idx = 0; idx < k_subset_size; ++idx) {
        set_least_squares_basis_column(
            ls_workspace, idx, current_funcs[static_cast<size_t>(idx)]);
    }
    auto ls_result = least_squares_solve(ls_workspace, config.rtol, config.atol);
    double current_error = ls_result.reconstruction_error;
    double current_cost = current_error;  // cost = error

    auto best_funcs = current_funcs;
    auto best_lin_coeffs = ls_result.coeffs;
    double best_error = current_error;
    double best_cost = current_cost;

    double temperature = config.initial_temperature;
    double last_log_temp = std::floor(std::log10(temperature));

    std::vector<SATraceStep> chain_trace;
    int total_iterations = 0;

    auto get_k_values = [&](const std::vector<ComplexVec>& funcs) {
        std::vector<int> ks;
        ks.reserve(funcs.size());
        for (const auto& f : funcs) {
            int support = 0;
            for (const auto& val : f) {
                if (std::abs(val) > 1e-5) support++;
            }
            if (support > 0) {
                ks.push_back(std::round(std::log(support) / std::log(p_prime)));
            } else {
                ks.push_back(0);
            }
        }
        return ks;
    };

    auto apply_mutation = [&](const ComplexVec& func) -> ComplexVec {
        if (config.fixed_dimension >= 0) {
            std::uniform_int_distribution<int> op_dist(0, n_orig >= 2 ? 4 : 2);
            int op = op_dist(rng);
            std::uniform_int_distribution<int> q_dist(0, n_orig - 1);
            
            
            if (op == 0) return apply_X(func, q_dist(rng), n_orig, p_prime);
            if (op == 1) return apply_Z(func, q_dist(rng), n_orig, p_prime);
            if (op == 2) return apply_clifford_S(func, q_dist(rng), n_orig, p_prime);
            
            int c = q_dist(rng);
            int t = q_dist(rng);
            while (t == c) t = q_dist(rng);
            
            if (op == 3) return apply_clifford_CX(func, c, t, n_orig, p_prime);
            return apply_clifford_CZ(func, c, t, n_orig, p_prime);
        } else {
            std::uniform_real_distribution<double> coin_dist(0.0, 1.0);
            if (config.clifford_ratio > 0.0 && coin_dist(rng) < config.clifford_ratio) {
                return apply_random_single_gate_clifford(func, n_orig, p_prime, rng);
            } else {
                auto [nf, _ops] = apply_random_pauli_string(func, n_orig, p_prime, rng, config.use_real_qubit_moves);
                return nf;
            }
        }
    };

    if (verbose) {
        std::cerr << "SA Pauli Expansion Initial (k=" << k_subset_size
                  << "): Temp=" << temperature
                  << ", Cost=" << current_cost
                  << ", Error=" << current_error << "\n";
    }

    while (temperature > config.min_temperature) {
        for (int iter = 0; iter < config.num_iterations_at_temp; ++iter) {
            std::vector<int> mutated_indices;
            std::vector<ComplexVec> previous_funcs;
            mutated_indices.reserve(2);
            previous_funcs.reserve(2);

            auto apply_candidate_update = [&](int idx, ComplexVec new_func, double min_norm) {
                if (!normalize_state(new_func, min_norm)) {
                    return;
                }
                mutated_indices.push_back(idx);
                previous_funcs.push_back(current_funcs[static_cast<size_t>(idx)]);
                current_funcs[static_cast<size_t>(idx)] = std::move(new_func);
                set_least_squares_basis_column(
                    ls_workspace, idx, current_funcs[static_cast<size_t>(idx)]);
            };

            double rand_val = uniform_01(rng);
            int move_type = -1;

            // Move Type A: Random Reset
            if (generator_func && rand_val < config.random_replace_prob) {
                move_type = 0;
                int idx = idx_dist(rng);
                auto new_func = generator_func(rng);
                apply_candidate_update(idx, std::move(new_func), 1e-12);

            // Move Type B: Cluster Move (2 functions)
            } else if (k_subset_size >= 2 &&
                       rand_val < (config.random_replace_prob + config.two_func_perturb_prob)) {
                move_type = 1;
                // Pick 2 distinct indices
                int i1 = idx_dist(rng);
                int i2 = idx_dist(rng);
                while (i2 == i1) i2 = idx_dist(rng);

                for (int idx : {i1, i2}) {
                    ComplexVec new_func = apply_mutation(current_funcs[static_cast<size_t>(idx)]);
                    apply_candidate_update(idx, std::move(new_func), 1e-9);
                }

            // Move Type C: Single move
            } else {
                move_type = 2;
                int idx = idx_dist(rng);
                ComplexVec new_func = apply_mutation(current_funcs[static_cast<size_t>(idx)]);
                apply_candidate_update(idx, std::move(new_func), 1e-9);
            }

            // Evaluate
            auto new_ls = least_squares_solve(ls_workspace, config.rtol, config.atol);
            double new_error = new_ls.reconstruction_error;
            double new_cost = new_error;

            // Metropolis acceptance
            double cost_delta = new_cost - current_cost;
            bool accepted = false;
            if (cost_delta < 0.0 || uniform_01(rng) < std::exp(-cost_delta / temperature)) {
                current_error = new_error;
                current_cost = new_cost;
                accepted = true;

                if (new_cost < best_cost) {
                    best_funcs = current_funcs;
                    best_lin_coeffs = new_ls.coeffs;
                    best_error = new_error;
                    best_cost = new_cost;
                }
            } else {
                for (size_t i = 0; i < mutated_indices.size(); ++i) {
                    const int idx = mutated_indices[i];
                    current_funcs[static_cast<size_t>(idx)] = std::move(previous_funcs[i]);
                    set_least_squares_basis_column(
                        ls_workspace, idx, current_funcs[static_cast<size_t>(idx)]);
                }
            }

            if (config.enable_tracing) {
                SATraceStep step;
                step.iteration = total_iterations;
                step.temperature = temperature;
                step.current_cost = current_cost;
                step.best_cost = best_cost;
                step.accepted = accepted;
                step.move_type = move_type;
                step.k_values = get_k_values(current_funcs);
                chain_trace.push_back(std::move(step));
            }
            total_iterations++;

            if (best_cost <= config.early_exit_threshold) break;
        }

        double current_log_temp = std::floor(std::log10(temperature));
        if (verbose && current_log_temp < last_log_temp) {
            std::cerr << "  Temp=" << temperature
                      << ": CurrentCost=" << current_cost
                      << ", BestCost=" << best_cost << "\n";
            last_log_temp = current_log_temp;
        }

        if (best_cost <= config.early_exit_threshold) {
             if (verbose) {
                 std::cerr << "  Reached early exit threshold (" << best_cost << " <= " << config.early_exit_threshold << "). Ending SA early!\n";
             }
             break;
        }

        temperature *= config.cooling_rate;
    }

    if (verbose) {
        std::cerr << "Finished SA for k=" << k_subset_size
                  << ". Best cost: " << best_cost
                  << ", Best error: " << best_error << "\n";
    }

    SAResult result;
    result.k = k_subset_size;
    result.best_basis_funcs = std::move(best_funcs);
    result.best_lin_coeffs = std::move(best_lin_coeffs);
    result.best_error = best_error;
    result.best_cost = best_cost;
    result.trace = std::move(chain_trace);
    return result;
}

// ---- Public entry point: dispatches to single or multi-chain ----
SAResult run_sa_pauli_expansion(
    const SAConfig& config,
    const ComplexVec& target,
    int n_orig,
    int p_prime,
    int k_subset_size,
    std::vector<ComplexVec> initial_basis,
    uint64_t base_seed,
    std::function<ComplexVec(std::mt19937_64&)> generator_func) {

    if (target.empty()) {
        throw std::invalid_argument("target must be non-empty.");
    }
    if (n_orig < 1) {
        throw std::invalid_argument("n_orig must be positive.");
    }
    if (p_prime < 2) {
        throw std::invalid_argument("p_prime must be at least 2.");
    }
    if (k_subset_size < 1) {
        throw std::invalid_argument("k_subset_size must be positive.");
    }
    if (config.fixed_dimension < -1 || config.fixed_dimension > n_orig) {
        throw std::invalid_argument(
            "fixed_dimension must be -1 or between 0 and n_orig.");
    }
    if (!initial_basis.empty() &&
        initial_basis.size() != static_cast<size_t>(k_subset_size)) {
        throw std::invalid_argument(
            "initial_basis must be empty or have k_subset_size entries.");
    }
    for (const auto& basis_func : initial_basis) {
        if (basis_func.size() != target.size()) {
            throw std::invalid_argument(
                "every initial_basis entry must have the same length as target.");
        }
    }
    if (!std::isfinite(config.initial_temperature) ||
        config.initial_temperature <= 0.0) {
        throw std::invalid_argument("initial_temperature must be positive.");
    }
    if (!std::isfinite(config.min_temperature) || config.min_temperature < 0.0) {
        throw std::invalid_argument("min_temperature must be non-negative.");
    }
    if (!std::isfinite(config.cooling_rate) ||
        config.cooling_rate <= 0.0 || config.cooling_rate >= 1.0) {
        throw std::invalid_argument("cooling_rate must be in the open interval (0, 1).");
    }
    if (config.num_iterations_at_temp < 1) {
        throw std::invalid_argument("num_iterations_at_temp must be positive.");
    }
    if (!std::isfinite(config.rtol) || config.rtol < 0.0 ||
        !std::isfinite(config.atol) || config.atol < 0.0) {
        throw std::invalid_argument("rtol and atol must be non-negative finite values.");
    }
    if (!std::isfinite(config.two_func_perturb_prob) ||
        config.two_func_perturb_prob < 0.0 ||
        config.two_func_perturb_prob > 1.0 ||
        !std::isfinite(config.random_replace_prob) ||
        config.random_replace_prob < 0.0 ||
        config.random_replace_prob > 1.0 ||
        config.two_func_perturb_prob + config.random_replace_prob > 1.0) {
        throw std::invalid_argument(
            "move probabilities must be in [0, 1] and sum to at most 1.");
    }
    if (!std::isfinite(config.clifford_ratio) ||
        config.clifford_ratio < 0.0 || config.clifford_ratio > 1.0) {
        throw std::invalid_argument("clifford_ratio must be in [0, 1].");
    }
    if (!std::isfinite(config.early_exit_threshold) ||
        config.early_exit_threshold < 0.0) {
        throw std::invalid_argument("early_exit_threshold must be non-negative.");
    }
    if (config.num_chains < 1) {
        throw std::invalid_argument("num_chains must be positive.");
    }

    int num_chains = config.num_chains;

    if (num_chains == 1) {
        // Single chain: run directly (verbose output to stderr)
        return run_single_chain(
            config, target, n_orig, p_prime, k_subset_size,
            initial_basis, base_seed, generator_func, 0, /*verbose=*/true);
    }

    // Multi-chain: launch each chain as an async task
    std::cerr << "Launching " << num_chains << " parallel SA chains for k="
              << k_subset_size << "...\n";

    std::vector<std::future<SAResult>> futures;
    futures.reserve(num_chains);

    for (int i = 0; i < num_chains; ++i) {
        // Each chain gets a distinct seed and its own deep copy of the initial basis
        uint64_t chain_seed = base_seed + static_cast<uint64_t>(i) * 1000003ULL;
        auto basis_copy = initial_basis;  // deep copy per chain

        futures.push_back(std::async(std::launch::async,
            [&config, &target, n_orig, p_prime, k_subset_size,
             basis_copy = std::move(basis_copy), chain_seed,
             generator_func, i]() mutable {
                // Only chain 0 prints verbose logs to avoid interleaved output
                return run_single_chain(
                    config, target, n_orig, p_prime, k_subset_size,
                    std::move(basis_copy), chain_seed, generator_func,
                    i, /*verbose=*/(i == 0));
            }));
    }

    // Collect results and pick the best
    SAResult best_result;
    best_result.best_cost = std::numeric_limits<double>::infinity();

    for (int i = 0; i < num_chains; ++i) {
        auto result = futures[i].get();
        std::cerr << "  Chain " << i << " finished: cost=" << result.best_cost
                  << ", error=" << result.best_error << "\n";
        if (result.best_cost < best_result.best_cost) {
            best_result = std::move(result);
        }
    }

    std::cerr << "Best across " << num_chains << " chains: cost="
              << best_result.best_cost << ", error=" << best_result.best_error << "\n";

    return best_result;
}

}  // namespace stabrank
