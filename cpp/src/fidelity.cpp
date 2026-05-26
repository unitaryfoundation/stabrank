#include "stabrank/fidelity.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <mutex>
#include <numbers>
#include <thread>
#include <vector>

namespace stabrank {

static int64_t ipow(int base, int exp) {
  int64_t result = 1;
  for (int i = 0; i < exp; ++i)
    result *= base;
  return result;
}

static int tuple_to_index(const int *t, int n, int d) {
  int idx = 0;
  for (int i = 0; i < n; ++i)
    idx = idx * d + t[i];
  return idx;
}

// Evaluate all phase polynomials in range [pc_start, pc_end) for a fixed
// (W, x0) configuration and update the shared best fidelity.
//
// The phase polynomial loop is embarrassingly parallel: each pc independently
// defines a stabilizer state |phi_pc> and computes |<phi_pc|target>|^2.
// We split the pc range across threads to utilize all cores.
static void
eval_phase_range(const ComplexVec &target, int n, int d, int k,
                 const std::vector<int> &W,  // k*n flat row-major
                 const std::vector<int> &x0, // length n
                 int64_t pc_start, int64_t pc_end,
                 int n_lin, int n_sq, int n_mix, int n_j1_lin,
                 std::atomic<double> &best_fidelity,
                 std::atomic<double> &best_fidelity_k,
                 std::atomic<int64_t> &states_checked,
                 std::mutex &print_mutex) {

  int64_t dk = ipow(d, k);
  double inv_sqrt_dk = 1.0 / std::sqrt(static_cast<double>(dk));

  // Thread-local buffers
  std::vector<int> y_buf(k);
  std::vector<int> x_buf(n);
  std::vector<int> c_lin(k), c_sq(k);
  std::vector<int> c_mix(std::max(n_mix, 1));
  std::vector<int> c_j1(k);  // second-level linear (d=2 only)

  // For d=2: phase = exp(2pi*i * Q) where Q = c_lin*y/2 + c_mix*y_s*y_t/2 + c_j1*y/4
  // Multiply Q by 4: 4Q = 2*c_lin*y + 2*c_mix*y_s*y_t + c_j1*y (integer)
  // phase = i^{4Q mod 4} where i = exp(2pi*i/4)
  //
  // For d>=3: phase = omega_d^{q_num mod d}
  // where q_num = c_lin*y + c_sq*y^2 + c_mix*y_s*y_t

  // Pre-compute phase table
  // For d>=3: omega_d^j for j=0..d-1
  // For d==2: i^j for j=0..3
  int phase_table_size = (d == 2) ? 4 : d;
  std::vector<std::complex<double>> phase_table(phase_table_size);
  for (int j = 0; j < phase_table_size; ++j)
    phase_table[j] = std::exp(
        std::complex<double>(0.0, 2.0 * std::numbers::pi * j / phase_table_size));

  // Number of values each coefficient type takes
  // For d>=3: c_lin in {0..d-1}, c_sq in {0..d-1}, c_mix in {0..d-1}
  // For d==2: c_lin in {0,1}, c_mix in {0,1}, c_j1 in {0,1,2,3}
  int j1_base = (d == 2) ? 4 : 1;  // c_j1_lin coeffs range over {0..j1_base-1}

  int64_t local_count = 0;  // thread-local counter, flushed every 100K iters

  for (int64_t pc = pc_start; pc < pc_end; ++pc) {
    // Decode phase coefficients from the flat index pc.
    // Layout: [c_lin (d values each)] [c_sq (d values, d>=3 only)]
    //         [c_mix (d values each)] [c_j1 (4 values each, d==2 only)]
    int64_t tmp_pc = pc;
    for (int i = 0; i < n_lin; ++i) {
      c_lin[i] = static_cast<int>(tmp_pc % d);
      tmp_pc /= d;
    }
    for (int i = 0; i < n_sq; ++i) {
      c_sq[i] = static_cast<int>(tmp_pc % d);
      tmp_pc /= d;
    }
    for (int i = 0; i < n_mix; ++i) {
      c_mix[i] = static_cast<int>(tmp_pc % d);
      tmp_pc /= d;
    }
    for (int i = 0; i < n_j1_lin; ++i) {
      c_j1[i] = static_cast<int>(tmp_pc % j1_base);
      tmp_pc /= j1_base;
    }

    // <state|target> = (1/sqrt(dk)) sum_y conj(phase(y)) target[x0 + W^T y]
    std::complex<double> overlap(0.0, 0.0);

    for (int64_t yi = 0; yi < dk; ++yi) {
      int64_t tmp_yi = yi;
      for (int i = k - 1; i >= 0; --i) {
        y_buf[i] = static_cast<int>(tmp_yi % d);
        tmp_yi /= d;
      }

      // x = x0 + W^T * y mod d
      for (int j = 0; j < n; ++j) {
        int sum_val = x0[j];
        for (int i = 0; i < k; ++i)
          sum_val += y_buf[i] * W[i * n + j];
        x_buf[j] = ((sum_val % d) + d) % d;
      }

      int phase_idx;
      if (d == 2) {
        // 4Q = 2*c_lin*y + 2*c_mix*y_s*y_t + c_j1*y
        int q4 = 0;
        for (int i = 0; i < k; ++i)
          q4 += 2 * c_lin[i] * y_buf[i] + c_j1[i] * y_buf[i];
        int mix_i = 0;
        for (int s = 0; s < k; ++s)
          for (int t = s + 1; t < k; ++t)
            q4 += 2 * c_mix[mix_i++] * y_buf[s] * y_buf[t];
        phase_idx = ((q4 % 4) + 4) % 4;
      } else {
        // q_num = c_lin*y + c_sq*y^2 + c_mix*y_s*y_t (mod d)
        int q_num = 0;
        for (int i = 0; i < k; ++i) {
          q_num += c_lin[i] * y_buf[i];
          q_num += c_sq[i] * y_buf[i] * y_buf[i];
        }
        int mix_i = 0;
        for (int s = 0; s < k; ++s)
          for (int t = s + 1; t < k; ++t)
            q_num += c_mix[mix_i++] * y_buf[s] * y_buf[t];
        phase_idx = ((q_num % d) + d) % d;
      }

      int x_flat = tuple_to_index(x_buf.data(), n, d);
      overlap += std::conj(phase_table[phase_idx]) * target[x_flat];
    }

    overlap *= inv_sqrt_dk;
    double fid = std::norm(overlap);

    // Atomically update per-k best
    double cur_k = best_fidelity_k.load(std::memory_order_relaxed);
    while (fid > cur_k) {
      if (best_fidelity_k.compare_exchange_weak(cur_k, fid,
                                                std::memory_order_relaxed))
        break;
    }

    // Atomically update global best
    double current_best = best_fidelity.load(std::memory_order_relaxed);
    while (fid > current_best) {
      if (best_fidelity.compare_exchange_weak(current_best, fid,
                                              std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(print_mutex);
        int chi_lb = static_cast<int>(std::ceil(1.0 / fid));
        std::cerr << "  New best F=" << fid << " (k=" << k
                  << ") => chi>=" << chi_lb << "\n";
        break;
      }
    }

    // Update progress counter periodically to keep the progress bar live
    ++local_count;
    if (local_count >= 100000) {
      states_checked.fetch_add(local_count, std::memory_order_relaxed);
      local_count = 0;
    }
  }
  // Flush remaining count
  states_checked.fetch_add(local_count, std::memory_order_relaxed);
}

FidelityResult max_stabilizer_fidelity(const ComplexVec &target, int n, int d) {

  unsigned int n_threads = std::thread::hardware_concurrency();
  if (n_threads == 0)
    n_threads = 4;

  std::atomic<double> best_fidelity{0.0};
  std::vector<std::atomic<double>> best_fidelity_per_k(n + 1);
  for (int k = 0; k <= n; ++k)
    best_fidelity_per_k[k].store(0.0, std::memory_order_relaxed);
  std::atomic<int64_t> states_checked{0};
  std::vector<int64_t> states_per_k(n + 1, 0);
  std::mutex print_mutex;

  std::cerr << "Exhaustive stabilizer fidelity search (n=" << n << ", d=" << d
            << ", threads=" << n_threads << ")\n";

  // Pre-compute total number of stabilizer states for progress reporting
  int64_t total_states = 0;
  for (int k = 0; k <= n; ++k) {
    int n_lin = k;
    int n_sq = (d >= 3) ? k : 0;
    int n_mix = k * (k - 1) / 2;
    int n_j1 = (d == 2) ? k : 0;

    // Phase combos: d^n_lin * d^n_sq * d^n_mix * 4^n_j1
    int64_t n_phase = ipow(d, n_lin + n_sq + n_mix) * ipow((d == 2) ? 4 : 1, n_j1);

    // Count subspaces * cosets (Gaussian binomial * d^{n-k})
    int64_t n_structural = 1;
    // C(n,k) pivot combos * d^{k*(n-k)} RREF entries * d^{n-k} cosets
    // = C(n,k) * d^{(k+1)*(n-k)}
    // Easier: just accumulate from the enumeration formula
    int64_t n_pivots = 1; // C(n,k) - compute iteratively
    for (int i = 0; i < k; ++i)
      n_pivots = n_pivots * (n - i) / (i + 1);
    int64_t n_free = ipow(d, k * (n - k));
    int64_t n_coset = ipow(d, n - k);
    n_structural = n_pivots * n_free * n_coset;

    total_states += n_structural * n_phase;
  }
  std::cerr << "  Total states to check: " << total_states << "\n";

  // Pre-compute omega table (kept for compatibility but phase_table is
  // built per-thread in eval_phase_range now)

  // Launch progress reporter thread only for large searches
  auto t_start = std::chrono::steady_clock::now();
  std::atomic<bool> done{false};
  bool use_progress = (total_states > 1000000);
  std::thread progress_thread;
  if (use_progress) {
    progress_thread = std::thread([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (done.load(std::memory_order_relaxed))
          break;

        int64_t checked = states_checked.load(std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        double pct = 100.0 * checked / total_states;
        double rate = checked / elapsed;
        double eta = (total_states - checked) / std::max(rate, 1.0);
        double best = best_fidelity.load(std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "  [" << static_cast<int>(pct) << "%] " << checked << "/"
                  << total_states << "  " << static_cast<int64_t>(rate)
                  << " states/s"
                  << "  ETA " << static_cast<int>(eta / 60) << "m"
                  << static_cast<int>(eta) % 60 << "s"
                  << "  best F=" << best << "\n";
      }
    });
  }

  // For each subspace dimension k, enumerate all (pivot, RREF, coset)
  // configurations and parallelize the phase polynomial loop across threads.
  for (int k = 0; k <= n; ++k) {
    int n_lin = k;
    int n_sq = (d >= 3) ? k : 0;
    int n_mix = k * (k - 1) / 2;
    int n_j1 = (d == 2) ? k : 0;
    int64_t n_phase_combos = ipow(d, n_lin + n_sq + n_mix)
                           * ipow((d == 2) ? 4 : 1, n_j1);

    // Enumerate pivot combinations
    std::vector<int> pivots(k);
    for (int i = 0; i < k; ++i)
      pivots[i] = i;

    auto next_combination = [&]() -> bool {
      int i = k - 1;
      while (i >= 0 && pivots[i] == n - k + i)
        --i;
      if (i < 0)
        return false;
      ++pivots[i];
      for (int j = i + 1; j < k; ++j)
        pivots[j] = pivots[j - 1] + 1;
      return true;
    };

    bool first_combo = true;
    do {
      if (!first_combo && !next_combination())
        break;
      first_combo = false;

      // Compute non-pivot columns
      std::vector<int> non_pivots;
      for (int j = 0; j < n; ++j) {
        bool is_pivot = false;
        for (int p = 0; p < k; ++p)
          if (pivots[p] == j) {
            is_pivot = true;
            break;
          }
        if (!is_pivot)
          non_pivots.push_back(j);
      }
      int n_np = static_cast<int>(non_pivots.size());

      // Enumerate RREF free entries
      int n_free = k * n_np;
      int64_t n_free_combos = ipow(d, n_free);

      for (int64_t fc = 0; fc < n_free_combos; ++fc) {
        // Build W
        std::vector<int> W(k * n, 0);
        int64_t tmp = fc;
        for (int row = 0; row < k; ++row) {
          W[row * n + pivots[row]] = 1;
          for (int ci = 0; ci < n_np; ++ci) {
            W[row * n + non_pivots[ci]] = static_cast<int>(tmp % d);
            tmp /= d;
          }
        }

        // Enumerate cosets
        int64_t n_coset = ipow(d, n_np);
        for (int64_t cc = 0; cc < n_coset; ++cc) {
          std::vector<int> x0(n, 0);
          int64_t tmp_cc = cc;
          for (int ci = 0; ci < n_np; ++ci) {
            x0[non_pivots[ci]] = static_cast<int>(tmp_cc % d);
            tmp_cc /= d;
          }

          // --- Parallel phase loop ---
          // Split [0, n_phase_combos) across n_threads threads.
          int64_t chunk = (n_phase_combos + n_threads - 1) / n_threads;
          std::vector<std::thread> threads;

          for (unsigned int t = 0; t < n_threads; ++t) {
            int64_t start = t * chunk;
            int64_t end = std::min(start + chunk, n_phase_combos);
            if (start >= n_phase_combos)
              break;

            threads.emplace_back(
                eval_phase_range, std::cref(target), n, d, k, std::cref(W),
                std::cref(x0),
                start, end,
                n_lin, n_sq, n_mix, n_j1,
                std::ref(best_fidelity),
                std::ref(best_fidelity_per_k[k]),
                std::ref(states_checked),
                std::ref(print_mutex));
          }

          for (auto &t : threads)
            t.join();
        }
      }
    } while (true);

    int64_t total = states_checked.load();
    int64_t prev = 0;
    for (int j = 0; j < k; ++j) prev += states_per_k[j];
    states_per_k[k] = total - prev;
    std::cerr << "  k=" << k << " done (" << states_per_k[k]
              << " states, best F=" << best_fidelity_per_k[k].load()
              << ")\n";
  }

  done.store(true, std::memory_order_relaxed);
  if (use_progress)
    progress_thread.join();

  auto t_end = std::chrono::steady_clock::now();
  double total_time = std::chrono::duration<double>(t_end - t_start).count();

  FidelityResult result;
  result.f_max = best_fidelity.load();
  result.extent_lb = (result.f_max > 1e-15)
      ? static_cast<int>(std::ceil(1.0 / result.f_max)) : 0;
  result.total_states = states_checked.load();
  result.elapsed_seconds = total_time;
  result.f_max_per_k.resize(n + 1);
  result.states_per_k.resize(n + 1);
  for (int k = 0; k <= n; ++k) {
    result.f_max_per_k[k] = best_fidelity_per_k[k].load();
    result.states_per_k[k] = states_per_k[k];
  }

  std::cerr << "Done. Checked " << result.total_states << " states in "
            << static_cast<int>(total_time) << "s."
            << " F_max=" << result.f_max
            << " => xi>=" << result.extent_lb << "\n";

  return result;
}

} // namespace stabrank
