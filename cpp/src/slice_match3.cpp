#include "stabrank/slice_match3.hpp"

#include "modular_detail.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <stdexcept>

namespace stabrank {

namespace {

using modular::mod;

// The points of F_3^2 as (x_1, x_2), the coordinate offsets and the six
// composite offsets in the order research/qutrit_m4_rank5/matcher.py solves
// them (COMP), pidx(x) = 3 x_1 + x_2.
constexpr int NCOMP = 6;
constexpr int COMP[NCOMP][2] = {{1, 1}, {2, 0}, {0, 2}, {1, 2}, {2, 1}, {2, 2}};
constexpr int E1[2] = {1, 0}, E2[2] = {0, 1};

inline int pidx(int x1, int x2) { return 3 * ((x1 % 3 + 3) % 3) + ((x2 % 3 + 3) % 3); }
inline int addp(int x0, const int* v) { return pidx(x0 / 3 + v[0], x0 % 3 + v[1]); }

// ------------------------------------------------------------ options ----

// The slice options of one base state u on n qutrits: the 3 K vectors
// w^l T_k (Pauli class k in first-seen order over X^a Z^c, cube root l,
// option code 3 k + l) plus 'absent' (code 3 K), as phase codes and
// residues, and the class tables cls[k1][k2][x], g[k1][k2][x] of
// Q_{k2}^{x_2} Q_{k1}^{x_1} u = w^g T_cls (matcher.TermOpts).
struct StateOptions3 {
    int n = 0, dim = 0, K = 0, nopt = 0, absent = 0;
    std::vector<int8_t> u;                 // dim
    std::vector<int8_t> codes;             // nopt x dim
    std::vector<int64_t> f1, f2;           // nopt x dim
    std::vector<int64_t> key1;             // nopt: random functional mod P1
    std::vector<int16_t> cls, g;           // K x K x 9
    std::vector<std::vector<int>> reps_a, reps_c;   // K: the digits of a and c
    std::map<int, std::vector<std::array<int16_t, NCOMP>>> rows_cache;   // (c1, c2) -> composite rows

    const int8_t* code(int o) const { return &codes[static_cast<size_t>(o) * dim]; }
    const int64_t* res1(int o) const { return &f1[static_cast<size_t>(o) * dim]; }
    const int64_t* res2(int o) const { return &f2[static_cast<size_t>(o) * dim]; }
    int16_t cl(int k1, int k2, int x) const { return cls[(static_cast<size_t>(k1) * K + k2) * 9 + x]; }
    int16_t gg(int k1, int k2, int x) const { return g[(static_cast<size_t>(k1) * K + k2) * 9 + x]; }
};

// The digits of an index on n qutrits, most significant first.
inline void digits_of(int t, int n, int* d) {
    for (int k = n - 1; k >= 0; --k) { d[k] = t % 3; t /= 3; }
}

// (X^a Z^c u)[y + a] = w^{c . y} u[y], in phase codes.
std::vector<int8_t> apply_pauli3(const std::vector<int8_t>& v, const std::vector<int>& a, const std::vector<int>& c,
                                 int n, int dim) {
    std::vector<int8_t> out(dim, 0);
    int d[16];
    for (int t = 0; t < dim; ++t) {
        if (v[t] == 0) continue;
        digits_of(t, n, d);
        int y = 0, ph = 0;
        for (int k = 0; k < n; ++k) {
            y = 3 * y + (d[k] + a[k]) % 3;
            ph += c[k] * d[k];
        }
        out[y] = static_cast<int8_t>(1 + ((v[t] - 1) + ph) % 3);
    }
    return out;
}

// Codes with the first nonzero entry made 1.
std::vector<int8_t> normalise3(const std::vector<int8_t>& v) {
    std::vector<int8_t> out(v.size(), 0);
    int shift = -1;
    for (size_t x = 0; x < v.size(); ++x) {
        if (v[x] == 0) continue;
        if (shift < 0) shift = v[x] - 1;
        out[x] = static_cast<int8_t>(1 + (((v[x] - 1) - shift + 3) % 3));
    }
    return out;
}

// (class, phase) with v = w^phase T[class], or {-1, -1}.
std::array<int, 2> code_of3(const StateOptions3& o, const std::vector<int8_t>& v) {
    for (int k = 0; k < o.K; ++k) {
        const int8_t* t = o.code(3 * k);
        int l = -1;
        bool ok = true;
        for (int x = 0; x < o.dim && ok; ++x) {
            if ((t[x] == 0) != (v[x] == 0)) { ok = false; break; }
            if (t[x] == 0) continue;
            const int d = ((v[x] - 1) - (t[x] - 1) + 3) % 3;
            if (l < 0) l = d;
            else if (d != l) ok = false;
        }
        if (ok) return {k, l};
    }
    return {-1, -1};
}

StateOptions3 make_options3(const int8_t* ucodes, int n, const Field3& F1, const Field3& F2,
                            const int64_t* functional) {
    StateOptions3 o;
    o.n = n;
    o.dim = 1;
    for (int k = 0; k < n; ++k) o.dim *= 3;
    o.K = o.dim;
    o.nopt = 3 * o.K + 1;
    o.absent = 3 * o.K;
    o.u.assign(ucodes, ucodes + o.dim);
    // the first nonzero entry of a dictionary state is 1 (patterns()), as
    // the class tables and the option phases assume
    for (int x = 0; x < o.dim; ++x) {
        if (o.u[x] == 0) continue;
        if (o.u[x] != 1) throw std::runtime_error("slice_match3: base state codes are not normalized");
        break;
    }
    std::vector<std::vector<int8_t>> imgs;
    std::map<std::vector<int8_t>, int> seen;
    std::vector<int> a(n), c(n);
    // a outer, c inner, both in the lexicographic order of itertools.product
    const int total = o.dim;
    for (int ai = 0; ai < total; ++ai) {
        digits_of(ai, n, a.data());
        for (int ci = 0; ci < total; ++ci) {
            digits_of(ci, n, c.data());
            auto qu = apply_pauli3(o.u, a, c, n, o.dim);
            auto key = normalise3(qu);
            if (seen.count(key)) continue;
            seen[key] = static_cast<int>(imgs.size());
            imgs.push_back(qu);
            o.reps_a.push_back(a);
            o.reps_c.push_back(c);
        }
    }
    if (static_cast<int>(imgs.size()) != o.K)
        throw std::runtime_error("slice_match3: wrong number of Pauli classes of a base state");
    for (int k = 0; k < n; ++k)
        if (o.reps_a[0][k] != 0 || o.reps_c[0][k] != 0)
            throw std::runtime_error("slice_match3: the identity is not the first Pauli class");
    o.codes.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.f1.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.f2.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.key1.assign(o.nopt, 0);
    for (int k = 0; k < o.K; ++k) {
        for (int l = 0; l < 3; ++l) {
            const int op = 3 * k + l;
            int8_t* row = &o.codes[static_cast<size_t>(op) * o.dim];
            for (int x = 0; x < o.dim; ++x) {
                const int8_t v = imgs[k][x];
                row[x] = v == 0 ? 0 : static_cast<int8_t>(1 + ((v - 1) + l) % 3);
                o.f1[static_cast<size_t>(op) * o.dim + x] = F1.of_code(row[x]);
                o.f2[static_cast<size_t>(op) * o.dim + x] = F2.of_code(row[x]);
            }
        }
    }
    for (int op = 0; op < o.nopt; ++op) {
        int64_t h = 0;
        for (int x = 0; x < o.dim; ++x) h = mod(h + functional[x] * o.res1(op)[x], SM3_P1);
        o.key1[op] = h;
    }
    o.cls.assign(static_cast<size_t>(o.K) * o.K * 9, 0);
    o.g.assign(static_cast<size_t>(o.K) * o.K * 9, 0);
    for (int k1 = 0; k1 < o.K; ++k1) {
        for (int k2 = 0; k2 < o.K; ++k2) {
            for (int x1 = 0; x1 < 3; ++x1) {
                for (int x2 = 0; x2 < 3; ++x2) {
                    std::vector<int8_t> v = o.u;
                    for (int t = 0; t < x1; ++t) v = apply_pauli3(v, o.reps_a[k1], o.reps_c[k1], n, o.dim);
                    for (int t = 0; t < x2; ++t) v = apply_pauli3(v, o.reps_a[k2], o.reps_c[k2], n, o.dim);
                    auto kl = code_of3(o, v);
                    if (kl[0] < 0) throw std::runtime_error("slice_match3: class product is not a translate");
                    const size_t at = (static_cast<size_t>(k1) * o.K + k2) * 9 + pidx(x1, x2);
                    o.cls[at] = static_cast<int16_t>(kl[0]);
                    o.g[at] = static_cast<int16_t>(kl[1]);
                }
            }
        }
    }
    return o;
}

// matcher.TermOpts.composite_rows: the codes at the six composite offsets
// of every shape consistent with the coordinate codes c1 (at e_1) and c2
// (at e_2); a plane (27 quadratics), a coordinate line (3 phases at its
// third point), or the point and the two diagonal lines.
const std::vector<std::array<int16_t, NCOMP>>& composite_rows3(StateOptions3& o, int c1, int c2) {
    const int key = c1 * (o.nopt + 1) + c2;
    auto it = o.rows_cache.find(key);
    if (it != o.rows_cache.end()) return it->second;
    const int16_t A = static_cast<int16_t>(o.absent);
    std::vector<std::array<int16_t, NCOMP>> rows;
    auto comp_index = [](int x1, int x2) {
        for (int q = 0; q < NCOMP; ++q) if (COMP[q][0] == x1 && COMP[q][1] == x2) return q;
        return -1;
    };
    if (c1 != o.absent && c2 != o.absent) {
        const int k1 = c1 / 3, l1 = c1 % 3, k2 = c2 / 3, l2 = c2 % 3;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c) {
                    std::array<int16_t, NCOMP> row;
                    for (int q = 0; q < NCOMP; ++q) {
                        const int x1 = COMP[q][0], x2 = COMP[q][1];
                        const int qv = ((a * x1 * x1 + b * x2 * x2 + c * x1 * x2 + (l1 - a + 3) * x1
                                         + (l2 - b + 3) * x2) % 3 + 3) % 3;
                        const int x = pidx(x1, x2);
                        row[q] = static_cast<int16_t>(3 * o.cl(k1, k2, x) + (qv + o.gg(k1, k2, x)) % 3);
                    }
                    rows.push_back(row);
                }
    } else if (c1 != o.absent) {
        const int k1 = c1 / 3, x = pidx(2, 0), q = comp_index(2, 0);
        for (int l = 0; l < 3; ++l) {
            std::array<int16_t, NCOMP> row;
            row.fill(A);
            row[q] = static_cast<int16_t>(3 * o.cl(k1, 0, x) + (o.gg(k1, 0, x) + l) % 3);
            rows.push_back(row);
        }
    } else if (c2 != o.absent) {
        const int k2 = c2 / 3, x = pidx(0, 2), q = comp_index(0, 2);
        for (int l = 0; l < 3; ++l) {
            std::array<int16_t, NCOMP> row;
            row.fill(A);
            row[q] = static_cast<int16_t>(3 * o.cl(0, k2, x) + (o.gg(0, k2, x) + l) % 3);
            rows.push_back(row);
        }
    } else {
        std::array<int16_t, NCOMP> pt;
        pt.fill(A);
        rows.push_back(pt);                              // point term
        const int pairs[2][2][2] = {{{1, 1}, {2, 2}}, {{1, 2}, {2, 1}}};
        const int x20 = pidx(2, 0);
        for (auto& pr : pairs) {
            const int qf = comp_index(pr[0][0], pr[0][1]), qs = comp_index(pr[1][0], pr[1][1]);
            for (int k = 0; k < o.K; ++k)
                for (int l = 0; l < 3; ++l)
                    for (int l2 = 0; l2 < 3; ++l2) {
                        std::array<int16_t, NCOMP> row;
                        row.fill(A);
                        row[qf] = static_cast<int16_t>(3 * k + l);
                        row[qs] = static_cast<int16_t>(3 * o.cl(k, 0, x20) + (o.gg(k, 0, x20) + l2) % 3);
                        rows.push_back(row);
                    }
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return o.rows_cache.emplace(key, std::move(rows)).first->second;
}

// ----------------------------------------------------- linear algebra ----

// Unique solution of A d = b over F_p (columns cols[i], dim rows); 0 with d,
// 1 when inconsistent, 2 when the solution is not unique.
int solve_unique3(const std::vector<const int64_t*>& cols, int dim, const int64_t* b, int64_t p,
                  std::vector<int64_t>& d) {
    const int r = static_cast<int>(cols.size());
    modular::Rref R = modular::rref_aug(cols, dim, b, p);
    if (!R.consistent) return 1;
    if (R.rank < r) return 2;
    d.assign(r, 0);
    for (int t = 0; t < R.rank; ++t) d[R.pivcol[t]] = R.M[t * R.cols + r];
    return 0;
}

}  // namespace

// ------------------------------------------------------------- field -----

Field3::Field3(int64_t prime) : p(prime) {
    if (p % 3 != 1) throw std::invalid_argument("slice_match3: the prime must be 1 mod 3");
    w = 0;
    for (int64_t g = 2; g < p; ++g) {
        const int64_t z = pow(g, (p - 1) / 3);
        if (z != 1) { w = z; break; }
    }
    if (w == 0 || pow(w, 3) != 1 || mod(w * w + w + 1, p) != 0)
        throw std::runtime_error("slice_match3: no primitive cube root");
    for (int k = 0; k < 3; ++k) wpow[k] = pow(w, k);
}

int64_t Field3::pow(int64_t a, int64_t e) const {
    int64_t r = 1;
    a = mod(a, p);
    e = mod(e, p - 1);
    while (e > 0) {
        if (e & 1) r = mod(r * a, p);
        a = mod(a * a, p);
        e >>= 1;
    }
    return r;
}

// -------------------------------------------------------------- kernel ----

struct SliceMatch3Kernel::Impl {
    int n2, dim;
    int64_t N;
    Field3 F1{SM3_P1}, F2{SM3_P2};
    std::vector<int8_t> codes;               // N x dim
    std::vector<int64_t> target1, target2;   // 9 x dim
    std::vector<int64_t> tkey1;              // 9
    std::vector<int64_t> functional;         // dim
    std::vector<std::unique_ptr<StateOptions3>> cache;

    // meet-in-the-middle table indexed by the functional value
    std::vector<int32_t> head, stamp;
    int32_t cur_stamp = 0;
    struct Entry { int32_t next; int64_t combo; };
    std::vector<Entry> entries;

    explicit Impl(int n2_) : n2(n2_), head(SM3_P1, -1), stamp(SM3_P1, 0) {
        dim = 1;
        for (int k = 0; k < n2; ++k) dim *= 3;
    }

    StateOptions3& options(int idx) {
        if (!cache[idx])
            cache[idx].reset(new StateOptions3(make_options3(&codes[static_cast<size_t>(idx) * dim], n2,
                                                             F1, F2, functional.data())));
        return *cache[idx];
    }

    // Every combination of options (one code per term from its list) with
    // sum_i d_i w_i = the target slice at point x, decided mod P1 on the
    // whole vector and re-decided mod P2 (slice_cover._mitm and the exact
    // decision of solve_slice3 for a pinned family).
    std::vector<std::vector<int16_t>> solve_point(const std::vector<StateOptions3*>& opts,
                                                  const std::vector<std::vector<int16_t>>& lists,
                                                  const std::vector<int64_t>& d1, const std::vector<int64_t>& d2,
                                                  int x, int64_t& candidates) {
        const int r = static_cast<int>(opts.size());
        const int64_t* rhs1 = &target1[static_cast<size_t>(x) * dim];
        const int64_t* rhs2 = &target2[static_cast<size_t>(x) * dim];
        std::vector<std::vector<int16_t>> out;
        std::vector<std::vector<int64_t>> hk(r);
        for (int i = 0; i < r; ++i) {
            hk[i].resize(lists[i].size());
            for (size_t o = 0; o < lists[i].size(); ++o)
                hk[i][o] = mod(d1[i] * opts[i]->key1[lists[i][o]], SM3_P1);
        }
        // balanced split by descending size (slice_cover._split_sides)
        std::vector<int> order(r);
        for (int i = 0; i < r; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return lists[a].size() > lists[b].size(); });
        std::vector<int> side[2];
        double prods[2] = {1.0, 1.0};
        for (int i : order) {
            const int s = prods[0] <= prods[1] ? 0 : 1;
            side[s].push_back(i);
            prods[s] *= static_cast<double>(lists[i].size());
        }
        const int A = prods[0] <= prods[1] ? 0 : 1, B = 1 - A;   // A is hashed, B probed
        auto side_size = [&](int s) {
            int64_t n = 1;
            for (int i : side[s]) n *= static_cast<int64_t>(lists[i].size());
            return n;
        };
        const int64_t nA = side_size(A), nB = side_size(B);
        ++cur_stamp;
        if (cur_stamp == 0) { std::fill(stamp.begin(), stamp.end(), 0); cur_stamp = 1; }
        entries.clear();
        std::vector<int> idx(r, 0);
        for (int64_t ca = 0; ca < nA; ++ca) {
            int64_t rem = ca, key = 0;
            for (int i : side[A]) {
                const int64_t sz = static_cast<int64_t>(lists[i].size());
                key += hk[i][static_cast<size_t>(rem % sz)];
                rem /= sz;
            }
            key %= SM3_P1;
            if (stamp[key] != cur_stamp) { stamp[key] = cur_stamp; head[key] = -1; }
            entries.push_back({head[key], ca});
            head[key] = static_cast<int32_t>(entries.size() - 1);
        }
        const int64_t tkey = tkey1[x];
        std::vector<int64_t> acc1(dim), acc2(dim);
        for (int64_t cb = 0; cb < nB; ++cb) {
            int64_t rem = cb, key = 0;
            for (int i : side[B]) {
                const int64_t sz = static_cast<int64_t>(lists[i].size());
                const int o = static_cast<int>(rem % sz);
                rem /= sz;
                idx[i] = o;
                key += hk[i][o];
            }
            const int64_t need = mod(tkey - key, SM3_P1);
            if (stamp[need] != cur_stamp) continue;
            for (int32_t e = head[need]; e >= 0; e = entries[e].next) {
                ++candidates;
                int64_t ra = entries[e].combo;
                for (int i : side[A]) {
                    const int64_t sz = static_cast<int64_t>(lists[i].size());
                    idx[i] = static_cast<int>(ra % sz);
                    ra /= sz;
                }
                std::fill(acc1.begin(), acc1.end(), 0);
                for (int i = 0; i < r; ++i) {
                    const int64_t* w = opts[i]->res1(lists[i][idx[i]]);
                    for (int y = 0; y < dim; ++y) acc1[y] += d1[i] * w[y];
                }
                bool ok = true;
                for (int y = 0; y < dim && ok; ++y) ok = mod(acc1[y], SM3_P1) == rhs1[y];
                if (!ok) continue;
                std::fill(acc2.begin(), acc2.end(), 0);
                for (int i = 0; i < r; ++i) {
                    const int64_t* w = opts[i]->res2(lists[i][idx[i]]);
                    for (int y = 0; y < dim; ++y) acc2[y] = mod(acc2[y] + mod(d2[i] * w[y], SM3_P2), SM3_P2);
                }
                for (int y = 0; y < dim && ok; ++y) ok = acc2[y] == rhs2[y];
                if (!ok) continue;
                std::vector<int16_t> sol(r);
                for (int i = 0; i < r; ++i) sol[i] = lists[i][idx[i]];
                out.push_back(sol);
            }
        }
        return out;
    }

    // matcher.Matcher._complete for one joined state: the composite points
    // one at a time over the codes of the shapes still alive per term, each
    // solution restricting the shapes (depth first).
    void complete(SliceMatch3Result& res, const std::vector<StateOptions3*>& opts, int x0,
                  const std::vector<int16_t>& cl1, const std::vector<int16_t>& cl2,
                  const std::vector<int64_t>& d1, const std::vector<int64_t>& d2) {
        const int r = static_cast<int>(opts.size());
        std::vector<const std::vector<std::array<int16_t, NCOMP>>*> rows(r);
        for (int i = 0; i < r; ++i) rows[i] = &composite_rows3(*opts[i], cl1[i], cl2[i]);
        // alive[c][i]: the rows of term i still consistent before point c
        std::vector<std::vector<std::vector<int32_t>>> alive(NCOMP + 1, std::vector<std::vector<int32_t>>(r));
        for (int i = 0; i < r; ++i) {
            alive[0][i].resize(rows[i]->size());
            for (size_t t = 0; t < rows[i]->size(); ++t) alive[0][i][t] = static_cast<int32_t>(t);
        }
        std::vector<std::vector<std::vector<int16_t>>> sols(NCOMP);
        std::vector<size_t> pos(NCOMP, 0);
        std::vector<std::vector<int16_t>> lists(r);
        std::vector<std::vector<int16_t>> pick(NCOMP, std::vector<int16_t>(r));   // pick[q][i]: code of term i at point q
        int c = 0;
        bool fresh = true;
        while (c >= 0) {
            if (c == NCOMP) {
                emit_hit(res, opts, x0, cl1, cl2, pick);
                --c;
                fresh = false;
                continue;
            }
            if (fresh) {
                for (int i = 0; i < r; ++i) {
                    lists[i].clear();
                    for (int32_t t : alive[c][i]) lists[i].push_back((*rows[i])[t][c]);
                    std::sort(lists[i].begin(), lists[i].end());
                    lists[i].erase(std::unique(lists[i].begin(), lists[i].end()), lists[i].end());
                }
                sols[c] = solve_point(opts, lists, d1, d2, addp(x0, COMP[c]), res.candidates);
                res.composite_solutions += static_cast<int64_t>(sols[c].size());
                pos[c] = 0;
            }
            if (pos[c] >= sols[c].size()) {
                --c;
                fresh = false;
                continue;
            }
            const auto& sol = sols[c][pos[c]++];
            bool ok = true;
            for (int i = 0; i < r && ok; ++i) {
                alive[c + 1][i].clear();
                for (int32_t t : alive[c][i])
                    if ((*rows[i])[t][c] == sol[i]) alive[c + 1][i].push_back(t);
                ok = !alive[c + 1][i].empty();
            }
            if (!ok) continue;
            for (int i = 0; i < r; ++i) pick[c][i] = sol[i];
            ++c;
            fresh = true;
        }
    }

    void emit_hit(SliceMatch3Result& res, const std::vector<StateOptions3*>& opts, int x0,
                       const std::vector<int16_t>& cl1, const std::vector<int16_t>& cl2,
                       const std::vector<std::vector<int16_t>>& pick) {
        const int r = static_cast<int>(opts.size());
        const size_t T = static_cast<size_t>(9) * dim;
        std::vector<int8_t> term(T);
        for (int i = 0; i < r; ++i) {
            std::fill(term.begin(), term.end(), 0);
            const StateOptions3& o = *opts[i];
            std::copy(o.u.begin(), o.u.end(), term.begin() + static_cast<size_t>(x0) * dim);
            const int8_t* c1 = o.code(cl1[i]);
            std::copy(c1, c1 + dim, term.begin() + static_cast<size_t>(addp(x0, E1)) * dim);
            const int8_t* c2 = o.code(cl2[i]);
            std::copy(c2, c2 + dim, term.begin() + static_cast<size_t>(addp(x0, E2)) * dim);
            for (int q = 0; q < NCOMP; ++q) {
                const int8_t* c = o.code(pick[q][i]);
                std::copy(c, c + dim, term.begin() + static_cast<size_t>(addp(x0, COMP[q])) * dim);
            }
            res.hits.insert(res.hits.end(), term.begin(), term.end());
        }
        ++res.nhits;
    }
};

SliceMatch3Kernel::SliceMatch3Kernel(const int8_t* codes, int64_t N, int n2,
                                     const int64_t* target1, const int64_t* target2, uint64_t seed)
    : impl_(new Impl(n2)), n2_(n2), N_(N) {
    if (n2 < 1 || n2 > 3) throw std::invalid_argument("slice_match3: n2 must be 1..3");
    Impl& I = *impl_;
    I.N = N;
    I.codes.assign(codes, codes + N * I.dim);
    for (int8_t c : I.codes)
        if (c < 0 || c > 3) throw std::invalid_argument("slice_match3: phase codes must be 0..3");
    I.target1.resize(static_cast<size_t>(9) * I.dim);
    I.target2.resize(static_cast<size_t>(9) * I.dim);
    for (size_t k = 0; k < I.target1.size(); ++k) {
        I.target1[k] = mod(target1[k], SM3_P1);
        I.target2[k] = mod(target2[k], SM3_P2);
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> U(1, SM3_P1 - 1);
    I.functional.resize(I.dim);
    for (auto& f : I.functional) f = U(rng);
    I.tkey1.assign(9, 0);
    for (int x = 0; x < 9; ++x)
        for (int y = 0; y < I.dim; ++y)
            I.tkey1[x] = mod(I.tkey1[x] + I.functional[y] * I.target1[static_cast<size_t>(x) * I.dim + y], SM3_P1);
    I.cache.resize(N);
}

SliceMatch3Kernel::~SliceMatch3Kernel() = default;

SliceMatch3Result SliceMatch3Kernel::run(const std::vector<int>& cover, int x0) {
    Impl& I = *impl_;
    SliceMatch3Result res;
    const int r = static_cast<int>(cover.size());
    if (r < 1 || r > 9) throw std::invalid_argument("slice_match3: cover size must be 1..9");
    if (x0 < 0 || x0 >= 9) throw std::invalid_argument("slice_match3: base point out of range");
    for (size_t a = 0; a < cover.size(); ++a) {
        if (cover[a] < 0 || cover[a] >= I.N) throw std::invalid_argument("slice_match3: state index out of range");
        for (size_t b = 0; b < a; ++b)
            if (cover[a] == cover[b]) throw std::invalid_argument("slice_match3: repeated base state");
    }
    std::vector<StateOptions3*> opts(r);
    for (int i = 0; i < r; ++i) opts[i] = &I.options(cover[i]);
    // the coefficient family at the base point, a point when the states are independent
    std::vector<const int64_t*> c1(r), c2(r);
    for (int i = 0; i < r; ++i) {
        c1[i] = opts[i]->res1(0);   // option 0 is class 0, phase 0: the state itself
        c2[i] = opts[i]->res2(0);
    }
    std::vector<int64_t> d1, d2;
    const int s1 = solve_unique3(c1, I.dim, &I.target1[static_cast<size_t>(x0) * I.dim], SM3_P1, d1);
    const int s2 = solve_unique3(c2, I.dim, &I.target2[static_cast<size_t>(x0) * I.dim], SM3_P2, d2);
    if (s1 == 1 || s2 == 1) { res.status = 1; return res; }
    if (s1 == 2 || s2 == 2) { res.status = 2; return res; }
    for (int i = 0; i < r; ++i)
        if (d1[i] == 0 || d2[i] == 0) { res.status = 1; return res; }
    res.coeffs2 = d2;
    // the two coordinate slices, each solved once against the point family
    std::vector<std::vector<int16_t>> full(r);
    for (int i = 0; i < r; ++i) {
        full[i].resize(opts[i]->nopt);
        for (int o = 0; o < opts[i]->nopt; ++o) full[i][o] = static_cast<int16_t>(o);
    }
    const int xe1 = addp(x0, E1), xe2 = addp(x0, E2);
    std::vector<std::vector<int16_t>> sols1 = I.solve_point(opts, full, d1, d2, xe1, res.candidates);
    res.coord_solutions.push_back(static_cast<int64_t>(sols1.size()));
    if (sols1.empty()) return res;
    std::vector<std::vector<int16_t>> sols2;
    const bool same = std::equal(&I.target1[static_cast<size_t>(xe1) * I.dim],
                                 &I.target1[static_cast<size_t>(xe1) * I.dim] + I.dim,
                                 &I.target1[static_cast<size_t>(xe2) * I.dim])
                      && std::equal(&I.target2[static_cast<size_t>(xe1) * I.dim],
                                    &I.target2[static_cast<size_t>(xe1) * I.dim] + I.dim,
                                    &I.target2[static_cast<size_t>(xe2) * I.dim]);
    sols2 = same ? sols1 : I.solve_point(opts, full, d1, d2, xe2, res.candidates);
    const int64_t joined = static_cast<int64_t>(sols1.size()) * static_cast<int64_t>(sols2.size());
    res.coord_solutions.push_back(joined);
    if (sols2.empty()) return res;
    res.joined = joined;
    for (const auto& cl1 : sols1)
        for (const auto& cl2 : sols2) I.complete(res, opts, x0, cl1, cl2, d1, d2);
    return res;
}

}  // namespace stabrank
