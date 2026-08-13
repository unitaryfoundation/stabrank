#include "stabrank/symmetric_engine.hpp"

#include "stabrank/clifford.hpp"
#include "stabrank/linalg.hpp"
#include "stabrank/pauli.hpp"

#include <cmath>
#include <complex>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>

namespace stabrank {

namespace {

size_t int_pow(int base, int exp) {
    size_t out = 1;
    for (int i = 0; i < exp; ++i) out *= static_cast<size_t>(base);
    return out;
}

// Permutation table for the cyclic qudit shift: out[i] = in[table[i]].
std::vector<size_t> shift_table(int shift, int n, int p) {
    const size_t dim = int_pow(p, n);
    std::vector<size_t> stride(static_cast<size_t>(n));
    for (int q = 0; q < n; ++q) stride[static_cast<size_t>(q)] = int_pow(p, n - 1 - q);

    std::vector<size_t> table(dim);
    std::vector<int> digits(static_cast<size_t>(n));
    for (size_t out = 0; out < dim; ++out) {
        size_t rem = out;
        for (int q = 0; q < n; ++q) {
            digits[static_cast<size_t>(q)] = static_cast<int>(rem / stride[static_cast<size_t>(q)]);
            rem %= stride[static_cast<size_t>(q)];
        }
        size_t in = 0;
        for (int k = 0; k < n; ++k) {
            const int src_axis = ((k - shift) % n + n) % n;
            in += static_cast<size_t>(digits[static_cast<size_t>(k)]) * stride[static_cast<size_t>(src_axis)];
        }
        table[out] = in;
    }
    return table;
}

ComplexVec apply_table(const ComplexVec& state, const std::vector<size_t>& table) {
    ComplexVec out(state.size());
    for (size_t i = 0; i < state.size(); ++i) out[i] = state[table[i]];
    return out;
}

ComplexVec kron(const ComplexVec& a, const ComplexVec& b) {
    ComplexVec out(a.size() * b.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[i * b.size() + j] = a[i] * b[j];
    return out;
}

void normalize(ComplexVec& v) {
    double norm_sq = 0.0;
    for (const auto& z : v) norm_sq += std::norm(z);
    const double norm = std::sqrt(norm_sq);
    for (auto& z : v) z /= norm;
}

// Random stabilizer state on `period` qudits via a Clifford walk from |0...0>.
ComplexVec random_block(int period, int p, std::mt19937_64& rng) {
    ComplexVec block(int_pow(p, period), {0.0, 0.0});
    block[0] = {1.0, 0.0};
    std::uniform_int_distribution<int> gate(0, period >= 2 ? 5 : 3);
    std::uniform_int_distribution<int> site(0, period - 1);
    const int steps = 6 * period + 6;
    for (int i = 0; i < steps; ++i) {
        const int q = site(rng);
        switch (gate(rng)) {
            case 0: block = apply_clifford_H(block, q, period, p); break;
            case 1: block = apply_clifford_S(block, q, period, p); break;
            case 2: block = apply_X(block, q, period, p); break;
            case 3: block = apply_Z(block, q, period, p); break;
            case 4: {
                int t = site(rng);
                while (t == q) t = site(rng);
                block = apply_clifford_CX(block, q, t, period, p);
                break;
            }
            default: {
                int t = site(rng);
                while (t == q) t = site(rng);
                block = apply_clifford_CZ(block, q, t, period, p);
                break;
            }
        }
    }
    normalize(block);
    return block;
}

ComplexVec periodic_seed(int n, int p, int period, std::mt19937_64& rng) {
    const ComplexVec block = random_block(period, p, rng);
    ComplexVec state = block;
    for (int i = 0; i < n / period - 1; ++i) {
        state = kron(state, block);
    }
    return state;
}

bool is_periodic(const ComplexVec& state, const std::vector<size_t>& period_table) {
    std::complex<double> overlap{0.0, 0.0};
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.size(); ++i) {
        overlap += std::conj(state[period_table[i]]) * state[i];
        norm_sq += std::norm(state[i]);
    }
    return std::abs(std::abs(overlap) - norm_sq) < 1e-9 * std::max(1.0, norm_sq);
}

// One symmetry-preserving move: a gate replicated over period translates.
ComplexVec mutate_seed(
    const ComplexVec& seed, int period, int n, int p, std::mt19937_64& rng) {

    const int reps = n / period;
    std::uniform_int_distribution<int> kind_dist(0, 5);
    std::uniform_int_distribution<int> site(0, period - 1);
    int kind = kind_dist(rng);
    if (kind == 5 && period < 2) kind = kind_dist(rng) % 5;

    ComplexVec out = seed;
    if (kind <= 3) {
        const int q0 = site(rng);
        for (int t = 0; t < reps; ++t) {
            const int q = q0 + t * period;
            switch (kind) {
                case 0: out = apply_X(out, q, n, p); break;
                case 1: out = apply_Z(out, q, n, p); break;
                case 2: out = apply_clifford_S(out, q, n, p); break;
                default: out = apply_clifford_H(out, q, n, p); break;
            }
        }
    } else if (kind == 4) {
        const int q0 = site(rng);
        std::uniform_int_distribution<int> off(1, n - 1);
        const int offset = off(rng);
        std::vector<std::pair<int, int>> pairs;
        for (int t = 0; t < reps; ++t) {
            int a = (q0 + t * period) % n;
            int b = (q0 + t * period + offset) % n;
            if (a > b) std::swap(a, b);
            if (a == b) continue;
            bool dup = false;
            for (const auto& pr : pairs) dup = dup || (pr.first == a && pr.second == b);
            if (!dup) pairs.emplace_back(a, b);
        }
        for (const auto& pr : pairs) out = apply_clifford_CZ(out, pr.first, pr.second, n, p);
    } else {
        const int q0 = site(rng);
        int q1 = site(rng);
        while (q1 == q0) q1 = site(rng);
        for (int t = 0; t < reps; ++t) {
            out = apply_clifford_CX(out, q0 + t * period, q1 + t * period, n, p);
        }
    }
    return out;
}

SymmetricResult run_single_chain(
    const SymmetricConfig& config,
    const ComplexVec& target,
    int n,
    int p,
    uint64_t seed_value,
    bool verbose) {

    std::mt19937_64 rng(seed_value);
    const int num_seeds = static_cast<int>(config.orbit_sizes.size());
    int k_total = 0;
    for (int o : config.orbit_sizes) k_total += o;

    std::vector<int> periods(static_cast<size_t>(num_seeds));
    std::vector<int> offsets(static_cast<size_t>(num_seeds));
    {
        int off = 0;
        for (int j = 0; j < num_seeds; ++j) {
            periods[static_cast<size_t>(j)] = config.shift_unit * config.orbit_sizes[static_cast<size_t>(j)];
            offsets[static_cast<size_t>(j)] = off;
            off += config.orbit_sizes[static_cast<size_t>(j)];
        }
    }

    const auto unit_table = shift_table(config.shift_unit, n, p);
    std::vector<std::vector<size_t>> period_tables;
    period_tables.reserve(static_cast<size_t>(num_seeds));
    for (int j = 0; j < num_seeds; ++j) {
        period_tables.push_back(shift_table(periods[static_cast<size_t>(j)], n, p));
    }

    std::vector<ComplexVec> seeds(static_cast<size_t>(num_seeds));
    for (int j = 0; j < num_seeds; ++j) {
        seeds[static_cast<size_t>(j)] = periodic_seed(n, p, periods[static_cast<size_t>(j)], rng);
    }

    IncrementalLeastSquares ils(target, k_total);
    std::vector<ComplexVec> columns(static_cast<size_t>(k_total));

    auto expand_seed = [&](int j) {
        const int o = config.orbit_sizes[static_cast<size_t>(j)];
        const int off = offsets[static_cast<size_t>(j)];
        columns[static_cast<size_t>(off)] = seeds[static_cast<size_t>(j)];
        for (int t = 1; t < o; ++t) {
            columns[static_cast<size_t>(off + t)] =
                apply_table(columns[static_cast<size_t>(off + t - 1)], unit_table);
        }
        for (int t = 0; t < o; ++t) {
            ils.set_column(off + t, columns[static_cast<size_t>(off + t)]);
        }
    };
    for (int j = 0; j < num_seeds; ++j) expand_seed(j);

    auto result0 = ils.solve();
    double current_err = result0.reconstruction_error;
    double best_err = current_err;
    std::vector<ComplexVec> best_columns = columns;
    ComplexVec best_coeffs = result0.coeffs;

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> seed_pick(0, num_seeds - 1);

    double temperature = config.initial_temperature;
    double last_log_temp = std::floor(std::log10(temperature));
    long long iteration = 0;

    while (temperature > config.min_temperature && best_err > config.early_exit) {
        for (int it = 0; it < config.iterations_at_temp; ++it) {
            ++iteration;
            int n_mut = (num_seeds > 1 && uni(rng) < config.two_seed_prob) ? 2 : 1;
            std::vector<int> picks;
            picks.push_back(seed_pick(rng));
            if (n_mut == 2) {
                int j2 = seed_pick(rng);
                while (j2 == picks[0]) j2 = seed_pick(rng);
                picks.push_back(j2);
            }
            std::vector<ComplexVec> saved;
            for (int j : picks) {
                saved.push_back(seeds[static_cast<size_t>(j)]);
                if (uni(rng) < config.reseed_prob) {
                    seeds[static_cast<size_t>(j)] =
                        periodic_seed(n, p, periods[static_cast<size_t>(j)], rng);
                } else {
                    seeds[static_cast<size_t>(j)] = mutate_seed(
                        seeds[static_cast<size_t>(j)], periods[static_cast<size_t>(j)], n, p, rng);
                }
                expand_seed(j);
            }

            const auto res = ils.solve();
            const double delta = res.reconstruction_error - current_err;
            if (delta < 0.0 || uni(rng) < std::exp(-delta / temperature)) {
                current_err = res.reconstruction_error;
                if (current_err < best_err) {
                    best_err = current_err;
                    best_columns = columns;
                    best_coeffs = res.coeffs;
                    for (int j = 0; j < num_seeds; ++j) {
                        if (!is_periodic(seeds[static_cast<size_t>(j)],
                                         period_tables[static_cast<size_t>(j)])) {
                            throw std::runtime_error(
                                "symmetry audit failed: seed lost its periodicity");
                        }
                    }
                    if (best_err <= config.early_exit) break;
                }
            } else {
                for (size_t i = 0; i < picks.size(); ++i) {
                    seeds[static_cast<size_t>(picks[i])] = std::move(saved[i]);
                    expand_seed(picks[i]);
                }
            }

            if (config.audit_every > 0 && iteration % config.audit_every == 0) {
                for (int j = 0; j < num_seeds; ++j) {
                    if (!is_periodic(seeds[static_cast<size_t>(j)],
                                     period_tables[static_cast<size_t>(j)])) {
                        throw std::runtime_error(
                            "symmetry audit failed: seed lost its periodicity");
                    }
                }
            }
        }

        const double log_temp = std::floor(std::log10(temperature));
        if (verbose && log_temp < last_log_temp) {
            std::cerr << "  T=" << temperature << ": current=" << current_err
                      << ", best=" << best_err << "\n";
            last_log_temp = log_temp;
        }
        temperature *= config.cooling_rate;
    }

    SymmetricResult out;
    out.best_error = best_err;
    out.best_columns = std::move(best_columns);
    out.best_coeffs = std::move(best_coeffs);
    return out;
}

}  // namespace

ComplexVec shift_qudits(const ComplexVec& state, int shift, int n, int p) {
    return apply_table(state, shift_table(shift, n, p));
}

SymmetricResult run_symmetric_sa(
    const SymmetricConfig& config,
    const ComplexVec& target,
    int n,
    int p,
    uint64_t base_seed) {

    if (target.empty()) throw std::invalid_argument("target must be non-empty");
    if (n < 1 || p < 2) throw std::invalid_argument("n and p must be positive (p >= 2)");
    if (config.shift_unit < 1 || n % config.shift_unit != 0) {
        throw std::invalid_argument("shift_unit must divide n");
    }
    if (config.orbit_sizes.empty()) {
        throw std::invalid_argument("orbit_sizes must be non-empty");
    }
    const int group_order = n / config.shift_unit;
    for (int o : config.orbit_sizes) {
        if (o < 1 || group_order % o != 0) {
            throw std::invalid_argument("orbit sizes must divide the group order");
        }
    }
    if (target.size() != int_pow(p, n)) {
        throw std::invalid_argument("target length must equal p^n");
    }
    if (config.num_chains < 1) throw std::invalid_argument("num_chains must be positive");

    if (config.num_chains == 1) {
        return run_single_chain(config, target, n, p, base_seed, /*verbose=*/true);
    }

    std::cerr << "Launching " << config.num_chains
              << " symmetric SA chains (k=";
    int k_total = 0;
    for (int o : config.orbit_sizes) k_total += o;
    std::cerr << k_total << ")...\n";

    std::vector<std::future<SymmetricResult>> futures;
    futures.reserve(static_cast<size_t>(config.num_chains));
    for (int i = 0; i < config.num_chains; ++i) {
        const uint64_t chain_seed = base_seed + static_cast<uint64_t>(i) * 1000003ULL;
        futures.push_back(std::async(std::launch::async, [&, chain_seed, i]() {
            return run_single_chain(config, target, n, p, chain_seed, /*verbose=*/(i == 0));
        }));
    }

    SymmetricResult best;
    best.best_error = std::numeric_limits<double>::infinity();
    for (int i = 0; i < config.num_chains; ++i) {
        auto result = futures[static_cast<size_t>(i)].get();
        std::cerr << "  chain " << i << ": best=" << result.best_error << "\n";
        if (result.best_error < best.best_error) best = std::move(result);
    }
    std::cerr << "Best across chains: " << best.best_error << "\n";
    return best;
}

}  // namespace stabrank
