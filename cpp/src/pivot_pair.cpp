#include "stabrank/pivot_pair.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace stabrank {

namespace {

using cd = std::complex<double>;

Eigen::VectorXcd gaussian(std::mt19937_64& rng, Eigen::Index n) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXcd v(n);
    for (Eigen::Index k = 0; k < n; ++k) v[k] = cd(nd(rng), nd(rng));
    return v;
}

// Orthonormal basis of the span of the given columns (Gram-Schmidt with
// reorthogonalisation), dropping dependent columns.
Eigen::MatrixXcd orthonormal_basis(const Eigen::MatrixXcd& cols) {
    Eigen::MatrixXcd B(cols.rows(), cols.cols());
    Eigen::Index k = 0;
    for (Eigen::Index c = 0; c < cols.cols(); ++c) {
        Eigen::VectorXcd v = cols.col(c);
        for (int pass = 0; pass < 2; ++pass)
            for (Eigen::Index r = 0; r < k; ++r) v -= B.col(r) * (B.col(r).adjoint() * v)(0, 0);
        double n = v.norm();
        if (n > 1e-10) B.col(k++) = v / n;
    }
    return B.leftCols(k);
}

// Groups of mutually parallel unit columns of U (each group has at least
// two members), by canonical-key sort into runs and an overlap test inside
// each run. Mirrors rank_exclusion._parallel_groups.
std::vector<std::vector<int>> parallel_groups(const Eigen::MatrixXcd& U, std::mt19937_64& rng,
                                              const PivotPairConfig& cfg) {
    std::vector<std::vector<int>> groups;
    const Eigen::Index n = U.cols();
    if (n < 2) return groups;
    Eigen::VectorXcd can, key;
    Eigen::RowVectorXcd c;
    bool ok = false;
    for (int attempt = 0; attempt < 8 && !ok; ++attempt) {
        can = gaussian(rng, U.rows());
        key = gaussian(rng, U.rows());
        c = can.transpose() * U;
        ok = (c.cwiseAbs().array() > 1e-9).all();
    }
    if (!ok) throw std::runtime_error("could not draw a canonicalising functional nonzero on every direction");
    Eigen::RowVectorXcd kk = (key.transpose() * U).cwiseQuotient(c);
    struct Keyed { double re, im; int idx; };
    std::vector<Keyed> sorted(n);
    for (Eigen::Index k = 0; k < n; ++k) sorted[k] = {kk[k].real(), kk[k].imag(), static_cast<int>(k)};
    std::sort(sorted.begin(), sorted.end(), [](const Keyed& a, const Keyed& b) {
        return a.re != b.re ? a.re < b.re : a.im < b.im;
    });
    std::vector<int> order(n);
    for (Eigen::Index k = 0; k < n; ++k) order[k] = sorted[k].idx;
    Eigen::Index start = 0;
    while (start < n) {
        Eigen::Index end = start + 1;
        while (end < n &&
               std::abs(kk[order[end]] - kk[order[end - 1]]) <= cfg.key_rel * (1.0 + std::abs(kk[order[end - 1]])))
            ++end;
        if (end - start >= 2) {
            const Eigen::Index g = end - start;
            Eigen::MatrixXcd V(U.rows(), g);
            for (Eigen::Index t = 0; t < g; ++t) V.col(t) = U.col(order[start + t]);
            Eigen::MatrixXd G = (V.adjoint() * V).cwiseAbs();
            std::vector<char> seen(g, 0);
            for (Eigen::Index s = 0; s < g; ++s) {
                if (seen[s]) continue;
                std::vector<int> comp{static_cast<int>(s)};
                seen[s] = 1;
                std::vector<Eigen::Index> stack{s};
                while (!stack.empty()) {
                    Eigen::Index v = stack.back();
                    stack.pop_back();
                    for (Eigen::Index w = 0; w < g; ++w) {
                        if (w != v && !seen[w] && G(v, w) > 1.0 - cfg.parallel) {
                            seen[w] = 1;
                            comp.push_back(static_cast<int>(w));
                            stack.push_back(w);
                        }
                    }
                }
                if (comp.size() >= 2) {
                    std::vector<int> members;
                    for (int t : comp) members.push_back(order[start + t]);
                    groups.push_back(std::move(members));
                }
            }
        }
        start = end;
    }
    return groups;
}

bool confirm(const Eigen::MatrixXcd& D, const Eigen::VectorXcd& psi, const std::array<int, 4>& cols,
             const PivotPairConfig& cfg) {
    Eigen::MatrixXcd A(D.rows(), 4);
    for (int t = 0; t < 4; ++t) A.col(t) = D.col(cols[t]);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.singularValues().minCoeff() < cfg.rank_tol) return false;
    Eigen::VectorXcd x = svd.solve(psi);
    return (A * x - psi).norm() < cfg.confirm_resid;
}

}  // namespace

std::vector<std::array<int, 4>> rank4_pivot_partners(
    const Eigen::MatrixXcd& D, const Eigen::VectorXcd& psi_in, int i,
    const std::vector<int>& partners, const std::vector<uint8_t>& allowed,
    const PivotPairConfig& cfg) {
    const Eigen::Index dim = D.rows(), N = D.cols();
    if (psi_in.size() != dim) throw std::invalid_argument("psi has the wrong dimension");
    if (static_cast<Eigen::Index>(allowed.size()) != N) throw std::invalid_argument("allowed has the wrong length");
    if (i < 0 || i >= N) throw std::invalid_argument("pivot out of range");
    Eigen::VectorXcd psi = psi_in / psi_in.norm();
    std::mt19937_64 rng(cfg.seed);
    Eigen::MatrixXcd R(cfg.proj_dim, dim);
    for (int r = 0; r < cfg.proj_dim; ++r) R.row(r) = gaussian(rng, dim).transpose();

    Eigen::MatrixXcd two(dim, 2);
    two.col(0) = psi;
    two.col(1) = D.col(i);
    Eigen::MatrixXcd B1 = orthonormal_basis(two);
    Eigen::MatrixXcd q1 = D - B1 * (B1.adjoint() * D);
    Eigen::VectorXd n1 = q1.colwise().norm();
    Eigen::VectorXcd e_i = D.col(i) / D.col(i).norm();
    Eigen::MatrixXcd p1 = D - e_i * (e_i.adjoint() * D);
    Eigen::VectorXd np1 = p1.colwise().norm();
    Eigen::MatrixXcd RQ1 = R * q1, RP1 = R * p1;

    std::set<std::array<int, 4>> found;
    Eigen::VectorXcd can = gaussian(rng, cfg.proj_dim), key = gaussian(rng, cfg.proj_dim);

    for (int j : partners) {
        if (j < 0 || j >= N) throw std::invalid_argument("partner out of range");
        if (j == i || n1[j] <= cfg.zero) continue;
        Eigen::VectorXcd v = q1.col(j) / n1[j];
        Eigen::RowVectorXcd w = v.adjoint() * q1;                 // components along the partner
        Eigen::VectorXcd Rv = R * v;
        std::vector<int> ids;
        ids.reserve(N);
        // The other two members carry indices above the partner: with the
        // partner the least index of the non-pivot members (the caller makes
        // it minimal in its orbit under the pivot's stabilizer, and the two
        // conditions hold together, see slice_lift.all_decompositions).
        for (Eigen::Index k = j + 1; k < N; ++k) {
            if (k == i || !allowed[k]) continue;
            double nq2sq = n1[k] * n1[k] - std::norm(w[k]);
            if (nq2sq > cfg.zero * cfg.zero) ids.push_back(static_cast<int>(k));
        }
        if (ids.size() < 2) continue;
        Eigen::MatrixXcd U(cfg.proj_dim, ids.size());
        for (size_t t = 0; t < ids.size(); ++t) {
            U.col(t) = RQ1.col(ids[t]) - Rv * w[ids[t]];
            U.col(t) /= U.col(t).norm();
        }
        auto groups = parallel_groups(U, rng, cfg);
        if (groups.empty()) continue;
        // second quotient, modulo span(s_i, s_j) only
        Eigen::VectorXcd f = p1.col(j) / np1[j];
        Eigen::VectorXcd Rf = R * f;
        for (const auto& g : groups) {
            const size_t gs = g.size();
            Eigen::MatrixXcd V2(cfg.proj_dim, gs);
            std::vector<double> n2(gs);
            std::vector<char> small(gs);
            std::vector<cd> k2(gs);
            for (size_t t = 0; t < gs; ++t) {
                int col = ids[g[t]];
                cd wf = (f.adjoint() * p1.col(col))(0, 0);
                V2.col(t) = RP1.col(col) - Rf * wf;
                n2[t] = V2.col(t).norm();
                small[t] = n2[t] < cfg.zero * std::max(1.0, np1[col]);
                cd denom = small[t] ? cd(1.0, 0.0) : (can.transpose() * V2.col(t))(0, 0);
                k2[t] = (key.transpose() * V2.col(t))(0, 0) / denom;
            }
            for (size_t a = 0; a < gs; ++a) {
                if (small[a]) continue;
                for (size_t b = a + 1; b < gs; ++b) {
                    if (small[b]) continue;
                    bool same_key = std::abs(k2[a] - k2[b]) < 1e-6 * (1.0 + std::abs(k2[a]));
                    if (same_key) {
                        double ov = std::abs((V2.col(a).adjoint() * V2.col(b))(0, 0));
                        if (std::abs(ov - n2[a] * n2[b]) < 1e-6 * n2[a] * n2[b]) continue;  // dependent quadruple
                    }
                    std::array<int, 4> cols{i, j, ids[g[a]], ids[g[b]]};
                    std::sort(cols.begin(), cols.end());
                    if (found.count(cols)) continue;
                    if (confirm(D, psi, cols, cfg)) found.insert(cols);
                }
            }
        }
    }
    return std::vector<std::array<int, 4>>(found.begin(), found.end());
}

}  // namespace stabrank
