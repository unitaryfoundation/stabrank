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
#include "stabrank/symmetric_engine.hpp"
#include "stabrank/pivot_pair.hpp"
#include "stabrank/t3_scan.hpp"
#include "stabrank/slice_match.hpp"
#include "stabrank/cover5.hpp"
#include "stabrank/cover6.hpp"
#include "stabrank/slice_match3.hpp"
#include "stabrank/dense_solve.hpp"

#include <algorithm>
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

    // --- run_symmetric_sa (cyclic-orbit symmetric-ansatz engine) ---
    m.def("run_symmetric_sa",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> target_arr,
           int n, int p, int shift_unit, std::vector<int> orbit_sizes,
           double initial_temperature, double min_temperature,
           double cooling_rate, int iterations_at_temp,
           double reseed_prob, double two_seed_prob,
           double early_exit, int audit_every,
           uint64_t seed, int num_chains) {

            stabrank::SymmetricConfig config;
            config.shift_unit = shift_unit;
            config.orbit_sizes = std::move(orbit_sizes);
            config.initial_temperature = initial_temperature;
            config.min_temperature = min_temperature;
            config.cooling_rate = cooling_rate;
            config.iterations_at_temp = iterations_at_temp;
            config.reseed_prob = reseed_prob;
            config.two_seed_prob = two_seed_prob;
            config.early_exit = early_exit;
            config.audit_every = audit_every;
            config.num_chains = num_chains;

            const auto target = ndarray_to_complexvec(target_arr);
            stabrank::SymmetricResult result;
            {
                nb::gil_scoped_release release;
                result = stabrank::run_symmetric_sa(config, target, n, p, seed);
            }

            nb::list columns;
            for (const auto& col : result.best_columns) {
                columns.append(complexvec_to_ndarray(col));
            }
            return nb::make_tuple(
                result.best_error, columns,
                complexvec_to_ndarray(result.best_coeffs));
        },
        "target"_a, "n"_a, "p"_a, "shift_unit"_a, "orbit_sizes"_a,
        "initial_temperature"_a = 0.5, "min_temperature"_a = 1e-4,
        "cooling_rate"_a = 0.999, "iterations_at_temp"_a = 400,
        "reseed_prob"_a = 0.02, "two_seed_prob"_a = 0.2,
        "early_exit"_a = 1e-12, "audit_every"_a = 5000,
        "seed"_a = 0, "num_chains"_a = 1,
        "Cyclic-orbit symmetric-ansatz SA; returns (best_error, columns, coeffs).");

    m.def("shift_qudits",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>> state,
           int shift, int n, int p) {
            return complexvec_to_ndarray(
                stabrank::shift_qudits(ndarray_to_complexvec(state), shift, n, p));
        },
        "state"_a, "shift"_a, "n"_a, "p"_a,
        "Cyclically shift qudit positions by `shift`.");

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
    m.def("rank4_pivot_partners",
          [](nb::ndarray<const std::complex<double>, nb::ndim<2>, nb::c_contig> D,
             nb::ndarray<const std::complex<double>, nb::ndim<1>, nb::c_contig> psi,
             int i,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> partners,
             nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig> allowed,
             int proj_dim, uint64_t seed) {
              const size_t dim = D.shape(0), N = D.shape(1);
              Eigen::MatrixXcd Dm(dim, N);
              for (size_t r = 0; r < dim; ++r)
                  for (size_t c = 0; c < N; ++c) Dm(r, c) = D(r, c);
              Eigen::VectorXcd pv(psi.shape(0));
              for (size_t r = 0; r < psi.shape(0); ++r) pv[r] = psi(r);
              std::vector<int> part(partners.shape(0));
              for (size_t k = 0; k < partners.shape(0); ++k) part[k] = static_cast<int>(partners(k));
              std::vector<uint8_t> allow(allowed.shape(0));
              for (size_t k = 0; k < allowed.shape(0); ++k) allow[k] = allowed(k);
              stabrank::PivotPairConfig cfg;
              cfg.proj_dim = proj_dim;
              cfg.seed = seed;
              auto found = stabrank::rank4_pivot_partners(Dm, pv, i, part, allow, cfg);
              const size_t n = found.size();
              auto* data = new int64_t[n * 4];
              for (size_t k = 0; k < n; ++k)
                  for (int t = 0; t < 4; ++t) data[k * 4 + t] = found[k][t];
              nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
              return nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>(data, {n, 4}, owner);
          },
          "D"_a, "psi"_a, "i"_a, "partners"_a, "allowed"_a, "proj_dim"_a = 10, "seed"_a = 11,
          "Every rank-4 decomposition of psi over the dictionary D (dim x N, unit columns) "
          "containing pivot i and a partner from `partners`, with the other two members among "
          "the columns marked in `allowed`; sorted index quadruples, one per row.");

    // --- t3_scan_pairs ---
    m.def("t3_scan_pairs",
          [](nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> PD,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> inv, int64_t ell,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> E2,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> T2, int64_t ell2, int64_t i,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> jlist,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> kok_index,
             nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig> kok_rows,
             nb::ndarray<const int8_t, nb::ndim<1>, nb::c_contig> isfree, int64_t need,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> hist,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> found_buf,
             nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> found_len,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> found_meta,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> spur_buf,
             nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> spur_len,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> spur_meta,
             nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> over_meta,
             nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> counters) {
              const int64_t N = static_cast<int64_t>(PD.shape(0));
              if (static_cast<int64_t>(E2.shape(0)) != N || static_cast<int64_t>(isfree.shape(0)) != N
                  || static_cast<int64_t>(kok_rows.shape(1)) != N
                  || E2.shape(1) != T2.shape(1) || jlist.shape(0) != kok_index.shape(0)
                  || hist.shape(0) != hist.shape(1) || found_buf.shape(1) != spur_buf.shape(1)
                  || found_meta.shape(1) != 3 || spur_meta.shape(1) != 3 || over_meta.shape(1) != 3
                  || found_len.shape(0) != found_buf.shape(0) || spur_len.shape(0) != spur_buf.shape(0)
                  || found_meta.shape(0) != found_buf.shape(0) || spur_meta.shape(0) != spur_buf.shape(0)
                  || counters.shape(0) < 10 || inv.shape(0) != static_cast<size_t>(ell))
                  throw nb::value_error("t3_scan_pairs: array shapes disagree");
              stabrank::T3ScanInputs in{PD.data(), N, static_cast<int64_t>(PD.shape(1)), inv.data(), ell,
                                        E2.data(), static_cast<int64_t>(E2.shape(1)), T2.data(),
                                        static_cast<int64_t>(T2.shape(0)), ell2, i, jlist.data(),
                                        static_cast<int64_t>(jlist.shape(0)), kok_index.data(),
                                        kok_rows.data(), isfree.data(), need};
              stabrank::T3ScanOutputs out{hist.data(), static_cast<int64_t>(hist.shape(0)),
                                          found_buf.data(), found_len.data(), found_meta.data(),
                                          static_cast<int64_t>(found_buf.shape(0)), spur_buf.data(),
                                          spur_len.data(), spur_meta.data(),
                                          static_cast<int64_t>(spur_buf.shape(0)), over_meta.data(),
                                          static_cast<int64_t>(over_meta.shape(0)),
                                          static_cast<int64_t>(found_buf.shape(1)), counters.data()};
              stabrank::t3_scan_pairs(in, out);
          },
          "PD"_a, "inv"_a, "ell"_a, "E2"_a, "T2"_a, "ell2"_a, "i"_a, "jlist"_a, "kok_index"_a,
          "kok_rows"_a, "isfree"_a, "need"_a, "hist"_a, "found_buf"_a, "found_len"_a, "found_meta"_a,
          "spur_buf"_a, "spur_len"_a, "spur_meta"_a, "over_meta"_a, "counters"_a,
          "One three-pivot scan of research/t3_rank7/batch.py with the exact in-place decision; "
          "same arguments and output buffers as the numba kernel scan_pairs.");


    // --- slice_match_kernel ---
    nb::class_<stabrank::SliceMatchKernel>(m, "SliceMatchKernel",
        "Compiled stage A of verify_challenge/slice_cover.SliceMatcher.run: every decomposition "
        "of the target with a given base slice of distinct, independent states at a given base "
        "point, decided mod 65521 and re-decided mod 2013265921 (cpp/src/slice_match.cpp).")
        .def("__init__",
             [](stabrank::SliceMatchKernel* self,
                nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig> codes, int n1,
                nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> target1,
                nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> target2, uint64_t seed) {
                 const size_t dim = codes.shape(1);
                 int n2 = 0;
                 while ((size_t{1} << n2) < dim) ++n2;
                 if ((size_t{1} << n2) != dim) throw nb::value_error("codes: the row length must be a power of two");
                 if (target1.shape(0) != (size_t{1} << n1) || target1.shape(1) != dim
                     || target2.shape(0) != target1.shape(0) || target2.shape(1) != dim)
                     throw nb::value_error("target arrays must be 2^n1 x dim");
                 new (self) stabrank::SliceMatchKernel(codes.data(), static_cast<int64_t>(codes.shape(0)), n2, n1,
                                                       target1.data(), target2.data(), seed);
             },
             "codes"_a, "n1"_a, "target1"_a, "target2"_a, "seed"_a = 11)
        .def("run",
             [](stabrank::SliceMatchKernel& self, const std::vector<int>& cover, int x0) {
                 auto res = self.run(cover, x0);
                 nb::dict out;
                 out["status"] = res.status;
                 out["coord_solutions"] = res.coord_solutions;
                 out["joined"] = res.joined;
                 out["types"] = res.types;
                 out["composite_solutions"] = res.composite_solutions;
                 out["candidates"] = res.candidates;
                 out["coeffs2"] = res.coeffs2;
                 const size_t r = cover.size();
                 const size_t T = (size_t{1} << self.n1()) * (size_t{1} << self.n2());
                 const size_t nh = static_cast<size_t>(res.nhits);
                 auto* data = new int8_t[std::max<size_t>(1, nh * r * T)];
                 std::copy(res.hits.begin(), res.hits.end(), data);
                 nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
                 out["hits"] = nb::ndarray<nb::numpy, int8_t, nb::ndim<3>>(data, {nh, r, T}, owner);
                 return out;
             },
             "cover"_a, "x0"_a,
             "status (0 complete, 1 refused, 2 dependent base: use the Python matcher), the stats "
             "and the hits as (nhits, r, 2^n1 dim) phase codes in the term layout x * dim + y.");


    // --- cover5_pair ---
    m.def("cover5_pair",
          [](nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> Q,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> U1,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> psi1,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> U2,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> psi2,
             int64_t i, int64_t j,
             nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig> members,
             int64_t max_run, uint64_t seed, int key_functionals) {
              const int64_t N = static_cast<int64_t>(Q.shape(0));
              if (static_cast<int64_t>(U1.shape(0)) != N || static_cast<int64_t>(U2.shape(0)) != N
                  || static_cast<int64_t>(members.shape(0)) != N || U1.shape(1) != U2.shape(1)
                  || psi1.shape(0) != U1.shape(1) || psi2.shape(0) != U2.shape(1))
                  throw nb::value_error("cover5_pair: array shapes disagree");
              stabrank::Cover5Inputs in{Q.data(), N, static_cast<int64_t>(Q.shape(1)), U1.data(), psi1.data(),
                                        U2.data(), psi2.data(), static_cast<int64_t>(U1.shape(1)), i, j,
                                        members.data(), max_run, seed, key_functionals};
              auto res = stabrank::cover5_pair(in);
              const size_t n = res.covers.size();
              auto* idx = new int64_t[std::max<size_t>(1, n * 5)];
              auto* flags = new int8_t[std::max<size_t>(1, n * 2)];
              for (size_t k = 0; k < n; ++k) {
                  for (int t = 0; t < 5; ++t) idx[k * 5 + t] = res.covers[k].idx[t];
                  flags[k * 2] = res.covers[k].full1;
                  flags[k * 2 + 1] = res.covers[k].full2;
              }
              nb::capsule o1(idx, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
              nb::capsule o2(flags, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
              return nb::make_tuple(nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>(idx, {n, 5}, o1),
                                    nb::ndarray<nb::numpy, int8_t, nb::ndim<2>>(flags, {n, 2}, o2),
                                    res.candidates, res.members);
          },
          "Q"_a, "U1"_a, "psi1"_a, "U2"_a, "psi2"_a, "i"_a, "j"_a, "members"_a, "max_run"_a = 64, "seed"_a = 17,
          "key_functionals"_a = 1,
          "slice_cover.CoverEnumerator.pair_covers for r = 5 (cpp/src/cover5.cpp): the sorted 5-sets "
          "through the pivot i and partner j whose span contains the target mod 2013265921, their "
          "fullness flags mod 65521 and mod 2013265921, the candidate count and the member count. "
          "key_functionals = 2 selects the 32-bit projective key (fewer accidental candidates, the "
          "same covers).");

    // --- cover6_pair ---
    m.def("cover6_pair",
          [](nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> Q,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> U1,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> psi1,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> U2,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> psi2,
             int64_t i, int64_t j,
             nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig> members,
             int64_t max_run, uint64_t seed) {
              const int64_t N = static_cast<int64_t>(Q.shape(0));
              if (static_cast<int64_t>(U1.shape(0)) != N || static_cast<int64_t>(U2.shape(0)) != N
                  || static_cast<int64_t>(members.shape(0)) != N || U1.shape(1) != U2.shape(1)
                  || psi1.shape(0) != U1.shape(1) || psi2.shape(0) != U2.shape(1))
                  throw nb::value_error("cover6_pair: array shapes disagree");
              stabrank::Cover6Inputs in{Q.data(), N, static_cast<int64_t>(Q.shape(1)), U1.data(), psi1.data(),
                                        U2.data(), psi2.data(), static_cast<int64_t>(U1.shape(1)), i, j,
                                        members.data(), max_run, seed};
              auto res = stabrank::cover6_pair(in);
              const size_t n = res.covers.size();
              auto* idx = new int64_t[std::max<size_t>(1, n * 6)];
              auto* flags = new int8_t[std::max<size_t>(1, n * 2)];
              auto* rank = new int64_t[std::max<size_t>(1, n)];
              for (size_t k = 0; k < n; ++k) {
                  for (int t = 0; t < 6; ++t) idx[k * 6 + t] = res.covers[k].idx[t];
                  flags[k * 2] = res.covers[k].full1;
                  flags[k * 2 + 1] = res.covers[k].full2;
                  rank[k] = res.covers[k].rank2;
              }
              nb::capsule o1(idx, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
              nb::capsule o2(flags, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
              nb::capsule o3(rank, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
              return nb::make_tuple(nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>(idx, {n, 6}, o1),
                                    nb::ndarray<nb::numpy, int8_t, nb::ndim<2>>(flags, {n, 2}, o2),
                                    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(rank, {n}, o3),
                                    res.candidates, res.members);
          },
          "Q"_a, "U1"_a, "psi1"_a, "U2"_a, "psi2"_a, "i"_a, "j"_a, "members"_a, "max_run"_a = 4096, "seed"_a = 17,
          "slice_cover.CoverEnumerator.pair_covers for r = 6 (cpp/src/cover6.cpp): the sorted 6-sets "
          "through the pivot i and partner j whose span contains the target mod 2013265921, their "
          "fullness flags mod 65521 and mod 2013265921, their rank mod 2013265921, the candidate count "
          "and the member count.");

    // --- slice_match3_kernel ---
    nb::class_<stabrank::SliceMatch3Kernel>(m, "SliceMatch3Kernel",
        "Compiled stage A of research/qutrit_m4_rank5/matcher.Matcher.run: every decomposition of "
        "the target with a given two-qutrit base slice of distinct, independent states at a given "
        "base point, decided mod 65521 and re-decided mod 2013265921 (cpp/src/slice_match3.cpp).")
        .def("__init__",
             [](stabrank::SliceMatch3Kernel* self,
                nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig> codes,
                nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> target1,
                nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> target2, uint64_t seed) {
                 const size_t dim = codes.shape(1);
                 int n2 = 0;
                 size_t d = 1;
                 while (d < dim) { d *= 3; ++n2; }
                 if (d != dim) throw nb::value_error("codes: the row length must be a power of three");
                 if (target1.shape(0) != 9 || target1.shape(1) != dim
                     || target2.shape(0) != 9 || target2.shape(1) != dim)
                     throw nb::value_error("target arrays must be 9 x dim");
                 new (self) stabrank::SliceMatch3Kernel(codes.data(), static_cast<int64_t>(codes.shape(0)), n2,
                                                        target1.data(), target2.data(), seed);
             },
             "codes"_a, "target1"_a, "target2"_a, "seed"_a = 11)
        .def("run",
             [](stabrank::SliceMatch3Kernel& self, const std::vector<int>& cover, int x0) {
                 auto res = self.run(cover, x0);
                 nb::dict out;
                 out["status"] = res.status;
                 out["coord_solutions"] = res.coord_solutions;
                 out["joined"] = res.joined;
                 out["composite_solutions"] = res.composite_solutions;
                 out["candidates"] = res.candidates;
                 out["coeffs2"] = res.coeffs2;
                 const size_t r = cover.size();
                 size_t dim = 1;
                 for (int k = 0; k < self.n2(); ++k) dim *= 3;
                 const size_t T = 9 * dim;
                 const size_t nh = static_cast<size_t>(res.nhits);
                 auto* data = new int8_t[std::max<size_t>(1, nh * r * T)];
                 std::copy(res.hits.begin(), res.hits.end(), data);
                 nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
                 out["hits"] = nb::ndarray<nb::numpy, int8_t, nb::ndim<3>>(data, {nh, r, T}, owner);
                 return out;
             },
             "cover"_a, "x0"_a,
             "status (0 complete, 1 refused, 2 dependent base: use the Python matcher), the stats and "
             "the hits as (nhits, r, 9 dim) phase codes in the term layout (3 x_1 + x_2) dim + y; x0 is "
             "3 x_1 + x_2.");

    // --- dense_solve ---
    m.def("dense_solve",
          [](const std::vector<nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig>>& opts1,
             const std::vector<nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig>>& opts2,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> d01,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> K1,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> d02,
             nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> K2,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> rhs1,
             nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> rhs2,
             const std::vector<int>& sideL, const std::vector<int>& sideR, int64_t max_cand, uint64_t seed) {
              const int r = static_cast<int>(opts1.size());
              const int64_t dim = static_cast<int64_t>(rhs1.shape(0));
              if (static_cast<int>(opts2.size()) != r || static_cast<int64_t>(d01.shape(0)) != r
                  || static_cast<int64_t>(d02.shape(0)) != r || static_cast<int64_t>(K1.shape(0)) != r
                  || static_cast<int64_t>(K2.shape(0)) != r || static_cast<int64_t>(rhs2.shape(0)) != dim)
                  throw nb::value_error("dense_solve: array shapes disagree");
              std::vector<const int64_t*> p1(r), p2(r);
              std::vector<int64_t> sizes(r);
              for (int i = 0; i < r; ++i) {
                  if (static_cast<int64_t>(opts1[i].shape(1)) != dim || static_cast<int64_t>(opts2[i].shape(1)) != dim
                      || opts1[i].shape(0) != opts2[i].shape(0))
                      throw nb::value_error("dense_solve: option arrays disagree with the right-hand side");
                  p1[i] = opts1[i].data();
                  p2[i] = opts2[i].data();
                  sizes[i] = static_cast<int64_t>(opts1[i].shape(0));
              }
              stabrank::DenseSolveInputs in{r, p1.data(), p2.data(), sizes.data(), dim, d01.data(), K1.data(),
                                            static_cast<int>(K1.shape(1)), d02.data(), K2.data(),
                                            static_cast<int>(K2.shape(1)), rhs1.data(), rhs2.data(),
                                            sideL.data(), static_cast<int>(sideL.size()), sideR.data(),
                                            static_cast<int>(sideR.size()), max_cand, seed};
              auto res = stabrank::dense_solve(in);
              const size_t n = static_cast<size_t>(res.passed);
              auto* data = new int32_t[std::max<size_t>(1, n * r)];
              std::copy(res.combos.begin(), res.combos.end(), data);
              nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
              return nb::make_tuple(nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(data, {n, static_cast<size_t>(r)}, owner),
                                    res.raw, res.pass1);
          },
          "opts1"_a, "opts2"_a, "d01"_a, "K1"_a, "d02"_a, "K2"_a, "rhs1"_a, "rhs2"_a, "sideL"_a, "sideR"_a,
          "max_cand"_a = 2000000, "seed"_a = 7,
          "slice_cover._dense with the whole-vector re-decision folded in (cpp/src/dense_solve.cpp): the "
          "option combinations whose slice equation has a solution in the coefficient family mod 65521 "
          "and mod 2013265921, the count of feature zeros (the Python candidate count) and the count "
          "surviving mod 65521.");

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
