#include "stabrank/pivot_pair.hpp"

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <random>

using namespace stabrank;
using cd = std::complex<double>;

namespace {

Eigen::MatrixXcd random_unit_columns(int dim, int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd;
    Eigen::MatrixXcd D(dim, N);
    for (int c = 0; c < N; ++c) {
        for (int r = 0; r < dim; ++r) D(r, c) = cd(nd(rng), nd(rng));
        D.col(c) /= D.col(c).norm();
    }
    return D;
}

}  // namespace

TEST_CASE("pivot-pair search finds the planted rank-4 decomposition and nothing else") {
    const int dim = 12, N = 60;
    Eigen::MatrixXcd D = random_unit_columns(dim, N, 3);
    // psi = combination of columns 5, 17, 23, 41; generic random columns give no other quadruple
    Eigen::VectorXcd psi = cd(0.3, 0.1) * D.col(5) + cd(-0.7, 0.2) * D.col(17)
                         + cd(0.5, -0.4) * D.col(23) + cd(0.2, 0.9) * D.col(41);
    std::vector<uint8_t> allowed(N, 1);
    std::vector<int> partners(N);
    for (int k = 0; k < N; ++k) partners[k] = k;
    auto found = rank4_pivot_partners(D, psi, 17, partners, allowed);
    REQUIRE(found.size() == 1);
    CHECK(found[0] == std::array<int, 4>{5, 17, 23, 41});
    // the partner must be the least non-pivot member: 23 and 41 as partners find nothing
    CHECK(rank4_pivot_partners(D, psi, 17, {23, 41}, allowed).empty());
    CHECK(rank4_pivot_partners(D, psi, 17, {5}, allowed).size() == 1);
    // restricting the partners away from the decomposition finds nothing
    std::vector<int> others;
    for (int k = 0; k < N; ++k) if (k != 5 && k != 23 && k != 41) others.push_back(k);
    CHECK(rank4_pivot_partners(D, psi, 17, others, allowed).empty());
    // and a pivot outside the decomposition finds nothing
    CHECK(rank4_pivot_partners(D, psi, 0, partners, allowed).empty());
}

TEST_CASE("a dependent quadruple without psi is not reported") {
    const int dim = 10, N = 40;
    Eigen::MatrixXcd D = random_unit_columns(dim, N, 7);
    // column 30 lies in span(D_2, D_9, D_14): a dependent quadruple containing the pivot 2
    D.col(30) = cd(1.0, 0.5) * D.col(2) + cd(-0.3, 0.2) * D.col(9) + cd(0.4, 0.4) * D.col(14);
    D.col(30) /= D.col(30).norm();
    // psi decomposes over 2, 21, 25, 33
    Eigen::VectorXcd psi = cd(1.0, 0.0) * D.col(2) + cd(0.5, 0.5) * D.col(21)
                         + cd(-0.6, 0.1) * D.col(25) + cd(0.3, -0.8) * D.col(33);
    std::vector<uint8_t> allowed(N, 1);
    std::vector<int> partners(N);
    for (int k = 0; k < N; ++k) partners[k] = k;
    auto found = rank4_pivot_partners(D, psi, 2, partners, allowed);
    REQUIRE(found.size() == 1);
    CHECK(found[0] == std::array<int, 4>{2, 21, 25, 33});
}
