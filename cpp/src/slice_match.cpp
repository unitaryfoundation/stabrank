#include "stabrank/slice_match.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>

namespace stabrank {

namespace {

inline int64_t mod(int64_t x, int64_t p) {
    x %= p;
    return x < 0 ? x + p : x;
}

inline int popcount(int x) { return __builtin_popcount(static_cast<unsigned>(x)); }

// ------------------------------------------------------------ options ----

// The slice options of one base state u: the 4 K vectors i^l Q_k u (Pauli
// class k in first-seen order over X^a Z^c, phase l, option code 4 k + l)
// plus 'absent' (code 4 K), as phase codes and residues, and the class
// products Q_b Q_a u = i^l Q_c u.
struct StateOptions {
    int dim = 0, K = 0, nopt = 0, absent = 0;
    std::vector<int8_t> u;                 // dim
    std::vector<int8_t> codes;             // nopt x dim
    std::vector<int64_t> f1, f2;           // nopt x dim
    std::vector<int64_t> key1;             // nopt: random functional mod P1
    std::vector<std::array<int, 2>> prod;  // K x K: (class, phase)
    std::vector<std::array<int, 2>> reps;  // K: (a, c)

    const int8_t* code(int o) const { return &codes[static_cast<size_t>(o) * dim]; }
    const int64_t* res1(int o) const { return &f1[static_cast<size_t>(o) * dim]; }
    const int64_t* res2(int o) const { return &f2[static_cast<size_t>(o) * dim]; }

    int compose(int code_a, int code_b, int sign) const {
        const int ka = code_a / 4, la = code_a % 4, kb = code_b / 4, lb = code_b % 4;
        const auto& pr = prod[static_cast<size_t>(ka) * K + kb];
        return 4 * pr[0] + ((la + lb + pr[1] + 2 * sign) & 3);
    }
};

// X^a Z^c applied to a phase pattern: out[x ^ a] = (-1)^{c . x} v[x].
std::vector<int8_t> apply_pauli(const std::vector<int8_t>& v, int a, int c, int dim) {
    std::vector<int8_t> out(dim, 0);
    for (int x = 0; x < dim; ++x) {
        if (v[x] == 0) continue;
        const int par = popcount(c & x) & 1;
        out[x ^ a] = static_cast<int8_t>(1 + (((v[x] - 1) + 2 * par) & 3));
    }
    return out;
}

// Codes with the first nonzero entry made 1.
std::vector<int8_t> normalise(const std::vector<int8_t>& v) {
    std::vector<int8_t> out(v.size(), 0);
    int shift = -1;
    for (size_t x = 0; x < v.size(); ++x) {
        if (v[x] == 0) continue;
        if (shift < 0) shift = v[x] - 1;
        out[x] = static_cast<int8_t>(1 + (((v[x] - 1) - shift + 4) & 3));
    }
    return out;
}

// (class, phase) with v = i^phase imgs[class], or {-1, -1}.
std::array<int, 2> code_of(const StateOptions& o, const std::vector<int8_t>& v) {
    for (int k = 0; k < o.K; ++k) {
        const int8_t* w = o.code(4 * k);
        int l = -1;
        bool ok = true;
        for (int x = 0; x < o.dim && ok; ++x) {
            if ((w[x] == 0) != (v[x] == 0)) { ok = false; break; }
            if (w[x] == 0) continue;
            const int d = ((v[x] - 1) - (w[x] - 1) + 4) & 3;
            if (l < 0) l = d;
            else if (d != l) ok = false;
        }
        if (ok) return {k, l};
    }
    return {-1, -1};
}

StateOptions make_options(const int8_t* ucodes, int n2, const SliceField& F1, const SliceField& F2,
                          const int64_t* functional) {
    StateOptions o;
    o.dim = 1 << n2;
    o.K = o.dim;
    o.nopt = 4 * o.K + 1;
    o.absent = 4 * o.K;
    o.u.assign(ucodes, ucodes + o.dim);
    std::vector<std::vector<int8_t>> imgs;
    std::map<std::vector<int8_t>, int> seen;
    for (int a = 0; a < o.dim; ++a) {
        for (int c = 0; c < o.dim; ++c) {
            auto qu = apply_pauli(o.u, a, c, o.dim);
            auto key = normalise(qu);
            if (seen.count(key)) continue;
            seen[key] = static_cast<int>(imgs.size());
            imgs.push_back(qu);
            o.reps.push_back({a, c});
        }
    }
    if (static_cast<int>(imgs.size()) != o.K)
        throw std::runtime_error("slice_match: wrong number of Pauli classes of a base state");
    o.codes.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.f1.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.f2.assign(static_cast<size_t>(o.nopt) * o.dim, 0);
    o.key1.assign(o.nopt, 0);
    for (int k = 0; k < o.K; ++k) {
        for (int l = 0; l < 4; ++l) {
            const int op = 4 * k + l;
            int8_t* row = &o.codes[static_cast<size_t>(op) * o.dim];
            for (int x = 0; x < o.dim; ++x) {
                const int8_t v = imgs[k][x];
                row[x] = v == 0 ? 0 : static_cast<int8_t>(1 + (((v - 1) + l) & 3));
                o.f1[static_cast<size_t>(op) * o.dim + x] = F1.of_code(row[x]);
                o.f2[static_cast<size_t>(op) * o.dim + x] = F2.of_code(row[x]);
            }
        }
    }
    for (int op = 0; op < o.nopt; ++op) {
        int64_t h = 0;
        for (int x = 0; x < o.dim; ++x) h = mod(h + functional[x] * o.res1(op)[x], SM_P1);
        o.key1[op] = h;
    }
    o.prod.assign(static_cast<size_t>(o.K) * o.K, {0, 0});
    for (int ka = 0; ka < o.K; ++ka) {
        for (int kb = 0; kb < o.K; ++kb) {
            auto v = apply_pauli(apply_pauli(o.u, o.reps[ka][0], o.reps[ka][1], o.dim),
                                 o.reps[kb][0], o.reps[kb][1], o.dim);
            auto kl = code_of(o, v);
            if (kl[0] < 0) throw std::runtime_error("slice_match: class product is not a translate");
            o.prod[static_cast<size_t>(ka) * o.K + kb] = kl;
        }
    }
    return o;
}

// ----------------------------------------------------- linear algebra ----

// Unique solution of A d = b over F_p (A dim x r, column-major as columns[i]);
// returns 0 with d, 1 when inconsistent, 2 when the solution is not unique.
int solve_unique(const std::vector<const int64_t*>& cols, int dim, const int64_t* b, int64_t p,
                 std::vector<int64_t>& d) {
    const int r = static_cast<int>(cols.size());
    std::vector<int64_t> M(static_cast<size_t>(dim) * (r + 1));
    for (int x = 0; x < dim; ++x) {
        for (int i = 0; i < r; ++i) M[x * (r + 1) + i] = mod(cols[i][x], p);
        M[x * (r + 1) + r] = mod(b[x], p);
    }
    std::vector<int> pivcol;
    int rank = 0;
    for (int c = 0; c < r && rank < dim; ++c) {
        int pr = -1;
        for (int t = rank; t < dim; ++t)
            if (M[t * (r + 1) + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        if (pr != rank)
            for (int j = 0; j <= r; ++j) std::swap(M[rank * (r + 1) + j], M[pr * (r + 1) + j]);
        // inverse by Fermat
        int64_t a = M[rank * (r + 1) + c], e = p - 2, inv = 1;
        while (e > 0) {
            if (e & 1) inv = mod(inv * a, p);
            a = mod(a * a, p);
            e >>= 1;
        }
        for (int j = 0; j <= r; ++j) M[rank * (r + 1) + j] = mod(M[rank * (r + 1) + j] * inv, p);
        for (int t = 0; t < dim; ++t) {
            if (t == rank || M[t * (r + 1) + c] == 0) continue;
            const int64_t g = M[t * (r + 1) + c];
            for (int j = 0; j <= r; ++j)
                M[t * (r + 1) + j] = mod(M[t * (r + 1) + j] - g * M[rank * (r + 1) + j], p);
        }
        pivcol.push_back(c);
        ++rank;
    }
    for (int t = rank; t < dim; ++t)
        if (M[t * (r + 1) + r] != 0) return 1;
    if (rank < r) return 2;
    d.assign(r, 0);
    for (int t = 0; t < rank; ++t) d[pivcol[t]] = M[t * (r + 1) + r];
    return 0;
}

// ------------------------------------------------------------- flats -----

struct Flats {
    int n1 = 0;
    std::vector<int> points;                      // nonzero points of F_2^{n1}
    std::vector<int> composite;                   // points of weight >= 2, ascending
    std::vector<int> subspaces;                   // masks over points 1..2^n1-1
    std::vector<std::vector<int>> by_presence;    // coordinate-presence mask -> subspaces

    explicit Flats(int n) : n1(n) {
        const int P = 1 << n1;
        for (int x = 1; x < P; ++x) {
            points.push_back(x);
            if (popcount(x) >= 2) composite.push_back(x);
        }
        std::vector<std::pair<std::pair<int, std::vector<int>>, int>> found;
        for (int mask = 0; mask < (1 << (P - 1)); ++mask) {
            // mask bit (x - 1) set means x in W
            auto in = [&](int x) { return x == 0 || (mask >> (x - 1)) & 1; };
            bool closed = true;
            for (int a = 0; a < P && closed; ++a)
                for (int b = 0; b < P; ++b)
                    if (in(a) && in(b) && !in(a ^ b)) { closed = false; break; }
            if (!closed) continue;
            std::vector<int> pts;
            for (int x = 1; x < P; ++x) if (in(x)) pts.push_back(x);
            found.push_back({{static_cast<int>(pts.size()), pts}, mask});
        }
        std::sort(found.begin(), found.end());
        for (auto& f : found) subspaces.push_back(f.second);
        by_presence.assign(1 << n1, {});
        for (int W : subspaces) {
            int pres = 0;
            for (int k = 0; k < n1; ++k)
                if ((W >> ((1 << k) - 1)) & 1) pres |= 1 << k;
            by_presence[pres].push_back(W);
        }
    }
    static bool contains(int W, int x) { return x == 0 || ((W >> (x - 1)) & 1); }
};

}  // namespace

// ------------------------------------------------------------- field -----

SliceField::SliceField(int64_t prime) : p(prime) {
    if (p % 16 != 1) throw std::invalid_argument("slice_match: the prime must be 1 mod 16");
    zeta = 0;
    for (int64_t g = 2; g < p; ++g) {
        const int64_t z = pow(g, (p - 1) / 16);
        if (pow(z, 8) != 1) { zeta = z; break; }
    }
    if (zeta == 0) throw std::runtime_error("slice_match: no primitive 16th root");
    const int64_t zi = inv(zeta);
    i = pow(zeta, 4);
    const int64_t inv2 = inv(2);
    cos = mod((zeta + zi) * inv2, p);
    sin = mod(mod(mod(zeta - zi, p) * inv2, p) * inv(i), p);
    tan = mod(sin * inv(cos), p);
    for (int k = 0; k < 4; ++k) ipow[k] = pow(i, k);
    if (mod(cos * cos + sin * sin, p) != 1) throw std::runtime_error("slice_match: cos^2 + sin^2 != 1");
}

int64_t SliceField::pow(int64_t a, int64_t e) const {
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

struct SliceMatchKernel::Impl {
    int n1, n2, dim, P;                      // P = 2^n1 slices
    int64_t N;
    SliceField F1{SM_P1}, F2{SM_P2};
    std::vector<int8_t> codes;               // N x dim
    std::vector<int64_t> target1, target2;   // P x dim
    std::vector<int64_t> tkey1;              // P: functional applied to the target slices
    std::vector<int64_t> functional;         // dim
    Flats flats;
    std::vector<std::unique_ptr<StateOptions>> cache;

    // meet-in-the-middle table indexed by the functional value
    std::vector<int32_t> head, stamp;
    int32_t cur_stamp = 0;
    struct Entry { int32_t next; int64_t combo; };
    std::vector<Entry> entries;

    Impl(int n1_, int n2_) : n1(n1_), n2(n2_), dim(1 << n2_), P(1 << n1_), flats(n1_),
                             head(SM_P1, -1), stamp(SM_P1, 0) {}

    const StateOptions& options(int idx) {
        if (!cache[idx])
            cache[idx].reset(new StateOptions(make_options(&codes[static_cast<size_t>(idx) * dim], n2,
                                                           F1, F2, functional.data())));
        return *cache[idx];
    }

    // Every combination of options (one code per term from its list) with
    // sum_i d_i w_i = rhs, decided mod P1 on the whole vector and re-decided
    // mod P2.
    std::vector<std::vector<int>> solve_point(const std::vector<const StateOptions*>& opts,
                                              const std::vector<std::vector<int>>& lists,
                                              const std::vector<int64_t>& d1, const std::vector<int64_t>& d2,
                                              int x, int64_t& candidates) {
        const int r = static_cast<int>(opts.size());
        const int64_t* rhs1 = &target1[static_cast<size_t>(x) * dim];
        const int64_t* rhs2 = &target2[static_cast<size_t>(x) * dim];
        std::vector<std::vector<int>> out;
        // per-term hashed keys
        std::vector<std::vector<int64_t>> hk(r);
        for (int i = 0; i < r; ++i) {
            hk[i].resize(lists[i].size());
            for (size_t o = 0; o < lists[i].size(); ++o)
                hk[i][o] = mod(d1[i] * opts[i]->key1[lists[i][o]], SM_P1);
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
        for (int s = 0; s < 2; ++s)
            if (lists.empty() || side[s].empty()) { /* one side empty is fine */ }
        const int A = prods[0] <= prods[1] ? 0 : 1, B = 1 - A;   // A is hashed, B probed
        auto side_size = [&](int s) {
            int64_t n = 1;
            for (int i : side[s]) n *= static_cast<int64_t>(lists[i].size());
            return n;
        };
        const int64_t nA = side_size(A), nB = side_size(B);
        // hash side A
        ++cur_stamp;
        if (cur_stamp == 0) { std::fill(stamp.begin(), stamp.end(), 0); cur_stamp = 1; }
        entries.clear();
        std::vector<int> idx(r, 0);
        for (int64_t ca = 0; ca < nA; ++ca) {
            int64_t rem = ca, key = 0;
            for (int i : side[A]) {
                const int64_t sz = static_cast<int64_t>(lists[i].size());
                const int o = static_cast<int>(rem % sz);
                rem /= sz;
                key += hk[i][o];
            }
            key %= SM_P1;
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
            const int64_t need = mod(tkey - key, SM_P1);
            if (stamp[need] != cur_stamp) continue;
            for (int32_t e = head[need]; e >= 0; e = entries[e].next) {
                ++candidates;
                int64_t ra = entries[e].combo;
                for (int i : side[A]) {
                    const int64_t sz = static_cast<int64_t>(lists[i].size());
                    idx[i] = static_cast<int>(ra % sz);
                    ra /= sz;
                }
                // whole equation mod P1
                std::fill(acc1.begin(), acc1.end(), 0);
                for (int i = 0; i < r; ++i) {
                    const int64_t* w = opts[i]->res1(lists[i][idx[i]]);
                    for (int y = 0; y < dim; ++y) acc1[y] += d1[i] * w[y];
                }
                bool ok = true;
                for (int y = 0; y < dim && ok; ++y) ok = mod(acc1[y], SM_P1) == rhs1[y];
                if (!ok) continue;
                // exact re-decision mod P2
                std::fill(acc2.begin(), acc2.end(), 0);
                for (int i = 0; i < r; ++i) {
                    const int64_t* w = opts[i]->res2(lists[i][idx[i]]);
                    for (int y = 0; y < dim; ++y) acc2[y] = mod(acc2[y] + mod(d2[i] * w[y], SM_P2), SM_P2);
                }
                for (int y = 0; y < dim && ok; ++y) ok = acc2[y] == rhs2[y];
                if (!ok) continue;
                std::vector<int> sol(r);
                for (int i = 0; i < r; ++i) sol[i] = lists[i][idx[i]];
                out.push_back(sol);
            }
        }
        return out;
    }

    // slice_cover.composite_codes: every assignment of option codes at the
    // composite points for a term with flat W and coordinate codes ccode
    // (ccode[k] = code at e_k, or -1 when absent there).
    std::vector<std::vector<int8_t>> composite_codes(const StateOptions& o, int W, const std::vector<int>& ccode) {
        const int P2n = 1 << n1;
        std::vector<int> basis, free;
        std::vector<char> span(P2n, 0);
        span[0] = 1;
        for (int k = 0; k < n1; ++k) {
            if (ccode[k] < 0) continue;
            basis.push_back(1 << k);
            for (int s = 0; s < P2n; ++s) if (span[s]) span[s ^ (1 << k)] = 1;
        }
        for (int pnt : flats.points) {
            if (!Flats::contains(W, pnt) || span[pnt]) continue;
            basis.push_back(pnt);
            free.push_back(pnt);
            for (int s = 0; s < P2n; ++s) if (span[s]) span[s ^ pnt] = 1;
        }
        const int j = static_cast<int>(basis.size());
        std::vector<std::pair<int, int>> pairs;
        for (int a = 0; a < j; ++a) for (int b = a + 1; b < j; ++b) pairs.push_back({a, b});
        const int nfree = static_cast<int>(free.size());
        int64_t nf = 1;
        for (int t = 0; t < nfree; ++t) nf *= o.absent;   // range(o.absent): 4 K codes per free point
        std::vector<std::vector<int8_t>> rows;
        std::vector<int> bcode(P2n, -1), code(P2n, -1);
        for (int k = 0; k < n1; ++k) if (ccode[k] >= 0) bcode[1 << k] = ccode[k];
        const int npairs = static_cast<int>(pairs.size());
        for (int64_t fc = 0; fc < nf; ++fc) {
            int64_t rem = fc;
            for (int t = 0; t < nfree; ++t) { bcode[free[t]] = static_cast<int>(rem % o.absent); rem /= o.absent; }
            for (int signs = 0; signs < (1 << npairs); ++signs) {
                std::fill(code.begin(), code.end(), -1);
                for (int t = 1; t < (1 << j); ++t) {
                    int pnt = 0, c = -1, s = 0;
                    for (int a = 0; a < j; ++a) {
                        if (!((t >> a) & 1)) continue;
                        pnt ^= basis[a];
                        c = c < 0 ? bcode[basis[a]] : o.compose(c, bcode[basis[a]], 0);
                    }
                    for (int q = 0; q < npairs; ++q)
                        if (((t >> pairs[q].first) & 1) && ((t >> pairs[q].second) & 1)) s ^= (signs >> q) & 1;
                    if (s) c = 4 * (c / 4) + ((c % 4 + 2) & 3);
                    code[pnt] = c;
                }
                std::vector<int8_t> row(flats.composite.size());
                for (size_t q = 0; q < flats.composite.size(); ++q) {
                    const int cc = code[flats.composite[q]];
                    row[q] = static_cast<int8_t>(cc < 0 ? o.absent : cc);
                }
                rows.push_back(row);
            }
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    }

    void emit_hit(SliceMatchResult& res, const std::vector<const StateOptions*>& opts, int x0,
                  const std::vector<std::vector<int>>& cl, const std::vector<int>& comp_codes) {
        const int r = static_cast<int>(opts.size());
        const size_t T = static_cast<size_t>(P) * dim;
        std::vector<int8_t> term(T);
        for (int i = 0; i < r; ++i) {
            std::fill(term.begin(), term.end(), 0);
            const StateOptions& o = *opts[i];
            std::copy(o.u.begin(), o.u.end(), term.begin() + static_cast<size_t>(x0) * dim);
            for (int k = 0; k < n1; ++k) {
                const int8_t* c = o.code(cl[k][i]);
                std::copy(c, c + dim, term.begin() + static_cast<size_t>(x0 ^ (1 << k)) * dim);
            }
            for (size_t q = 0; q < flats.composite.size(); ++q) {
                const int8_t* c = o.code(comp_codes[q * r + i]);
                std::copy(c, c + dim, term.begin() + static_cast<size_t>(x0 ^ flats.composite[q]) * dim);
            }
            res.hits.insert(res.hits.end(), term.begin(), term.end());
        }
        ++res.nhits;
    }

    // slice_cover.SliceMatcher._complete for a point family: the composite
    // slices point by point over the per-point code sets, then the row
    // consistency of every term.
    void complete(SliceMatchResult& res, const std::vector<const StateOptions*>& opts, int x0,
                  const std::vector<std::vector<int>>& cl, const std::vector<int>& Wsel,
                  const std::vector<int64_t>& d1, const std::vector<int64_t>& d2) {
        const int r = static_cast<int>(opts.size());
        const int NC = static_cast<int>(flats.composite.size());
        std::vector<std::vector<std::vector<int8_t>>> rows(r);
        std::vector<int> ccode(n1);
        for (int i = 0; i < r; ++i) {
            for (int k = 0; k < n1; ++k) ccode[k] = cl[k][i] == opts[i]->absent ? -1 : cl[k][i];
            rows[i] = composite_codes(*opts[i], Wsel[i], ccode);
        }
        if (NC == 0) {
            emit_hit(res, opts, x0, cl, {});
            return;
        }
        std::vector<std::vector<std::vector<int>>> sols(NC);
        for (int c = 0; c < NC; ++c) {
            std::vector<std::vector<int>> lists(r);
            for (int i = 0; i < r; ++i) {
                for (auto& rw : rows[i]) lists[i].push_back(rw[c]);
                std::sort(lists[i].begin(), lists[i].end());
                lists[i].erase(std::unique(lists[i].begin(), lists[i].end()), lists[i].end());
            }
            sols[c] = solve_point(opts, lists, d1, d2, x0 ^ flats.composite[c], res.candidates);
            res.composite_solutions += static_cast<int64_t>(sols[c].size());
            if (sols[c].empty()) return;
        }
        // depth-first over the composite points with the surviving rows per term
        std::vector<int> chosen(static_cast<size_t>(NC) * r, 0);
        std::vector<std::vector<std::vector<int>>> alive(NC + 1, std::vector<std::vector<int>>(r));
        for (int i = 0; i < r; ++i) {
            alive[0][i].resize(rows[i].size());
            for (size_t t = 0; t < rows[i].size(); ++t) alive[0][i][t] = static_cast<int>(t);
        }
        std::vector<size_t> pos(NC, 0);
        int c = 0;
        while (c >= 0) {
            if (c == NC) {
                emit_hit(res, opts, x0, cl, chosen);
                --c;
                continue;
            }
            if (pos[c] >= sols[c].size()) {
                pos[c] = 0;
                --c;
                continue;
            }
            const auto& sol = sols[c][pos[c]++];
            bool ok = true;
            for (int i = 0; i < r && ok; ++i) {
                alive[c + 1][i].clear();
                for (int t : alive[c][i])
                    if (rows[i][t][c] == sol[i]) alive[c + 1][i].push_back(t);
                ok = !alive[c + 1][i].empty();
            }
            if (!ok) continue;
            for (int i = 0; i < r; ++i) chosen[static_cast<size_t>(c) * r + i] = sol[i];
            ++c;
        }
    }
};

SliceMatchKernel::SliceMatchKernel(const int8_t* codes, int64_t N, int n2, int n1,
                                   const int64_t* target1, const int64_t* target2, uint64_t seed)
    : impl_(new Impl(n1, n2)), n1_(n1), n2_(n2), N_(N) {
    if (n1 < 1 || n1 > 3 || n2 < 1 || n2 > 3) throw std::invalid_argument("slice_match: n1, n2 must be 1..3");
    Impl& I = *impl_;
    I.N = N;
    I.codes.assign(codes, codes + N * I.dim);
    for (int8_t c : I.codes)
        if (c < 0 || c > 4) throw std::invalid_argument("slice_match: phase codes must be 0..4");
    I.target1.resize(static_cast<size_t>(I.P) * I.dim);
    I.target2.resize(static_cast<size_t>(I.P) * I.dim);
    for (size_t k = 0; k < I.target1.size(); ++k) {
        I.target1[k] = mod(target1[k], SM_P1);
        I.target2[k] = mod(target2[k], SM_P2);
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> U(1, SM_P1 - 1);
    I.functional.resize(I.dim);
    for (auto& f : I.functional) f = U(rng);
    I.tkey1.assign(I.P, 0);
    for (int x = 0; x < I.P; ++x)
        for (int y = 0; y < I.dim; ++y)
            I.tkey1[x] = mod(I.tkey1[x] + I.functional[y] * I.target1[static_cast<size_t>(x) * I.dim + y], SM_P1);
    I.cache.resize(N);
}

SliceMatchKernel::~SliceMatchKernel() = default;

SliceMatchResult SliceMatchKernel::run(const std::vector<int>& cover, int x0) {
    Impl& I = *impl_;
    SliceMatchResult res;
    const int r = static_cast<int>(cover.size());
    if (r < 1 || r > 8) throw std::invalid_argument("slice_match: cover size must be 1..8");
    if (x0 < 0 || x0 >= I.P) throw std::invalid_argument("slice_match: base point out of range");
    for (size_t a = 0; a < cover.size(); ++a) {
        if (cover[a] < 0 || cover[a] >= I.N) throw std::invalid_argument("slice_match: state index out of range");
        for (size_t b = 0; b < a; ++b)
            if (cover[a] == cover[b]) throw std::invalid_argument("slice_match: repeated base state");
    }
    std::vector<const StateOptions*> opts(r);
    for (int i = 0; i < r; ++i) opts[i] = &I.options(cover[i]);
    // the coefficient family at the base point, a point when the states are independent
    std::vector<const int64_t*> c1(r), c2(r);
    for (int i = 0; i < r; ++i) {
        c1[i] = opts[i]->res1(0);   // option 0 is class 0, phase 0: the state itself
        c2[i] = opts[i]->res2(0);
    }
    std::vector<int64_t> d1, d2;
    const int s1 = solve_unique(c1, I.dim, &I.target1[static_cast<size_t>(x0) * I.dim], SM_P1, d1);
    const int s2 = solve_unique(c2, I.dim, &I.target2[static_cast<size_t>(x0) * I.dim], SM_P2, d2);
    if (s1 == 1 || s2 == 1) { res.status = 1; return res; }
    if (s1 == 2 || s2 == 2) { res.status = 2; return res; }
    for (int i = 0; i < r; ++i)
        if (d1[i] == 0 || d2[i] == 0) { res.status = 1; return res; }
    res.coeffs2 = d2;
    // coordinate slices, each solved once (the family is a point)
    std::vector<std::vector<std::vector<int>>> sols(I.n1);
    std::vector<std::vector<int>> full(r);
    for (int i = 0; i < r; ++i) {
        full[i].resize(opts[i]->nopt);
        for (int o = 0; o < opts[i]->nopt; ++o) full[i][o] = o;
    }
    int64_t cum = 1;
    for (int k = 0; k < I.n1; ++k) {
        const int x = x0 ^ (1 << k);
        int same = -1;
        for (int kk = 0; kk < k && same < 0; ++kk) {
            const int xx = x0 ^ (1 << kk);
            if (std::equal(&I.target1[static_cast<size_t>(x) * I.dim], &I.target1[static_cast<size_t>(x) * I.dim] + I.dim,
                           &I.target1[static_cast<size_t>(xx) * I.dim])
                && std::equal(&I.target2[static_cast<size_t>(x) * I.dim], &I.target2[static_cast<size_t>(x) * I.dim] + I.dim,
                              &I.target2[static_cast<size_t>(xx) * I.dim]))
                same = kk;
        }
        if (same >= 0) sols[k] = sols[same];
        else sols[k] = I.solve_point(opts, full, d1, d2, x, res.candidates);
        cum *= static_cast<int64_t>(sols[k].size());
        res.coord_solutions.push_back(cum);
        if (sols[k].empty()) return res;
    }
    res.joined = cum;
    // joined states: the product of the coordinate solutions
    std::vector<size_t> pos(I.n1, 0);
    std::vector<std::vector<int>> cl(I.n1);
    std::vector<int> pres(r), Wsel(r);
    while (true) {
        for (int k = 0; k < I.n1; ++k) cl[k] = sols[k][pos[k]];
        for (int i = 0; i < r; ++i) {
            pres[i] = 0;
            for (int k = 0; k < I.n1; ++k)
                if (cl[k][i] != opts[i]->absent) pres[i] |= 1 << k;
        }
        // every selection of a flat per term with the observed presence
        std::vector<size_t> tpos(r, 0);
        while (true) {
            for (int i = 0; i < r; ++i) Wsel[i] = I.flats.by_presence[pres[i]][tpos[i]];
            ++res.types;
            I.complete(res, opts, x0, cl, Wsel, d1, d2);
            int i = 0;
            while (i < r) {
                if (++tpos[i] < I.flats.by_presence[pres[i]].size()) break;
                tpos[i] = 0;
                ++i;
            }
            if (i == r) break;
        }
        int k = 0;
        while (k < I.n1) {
            if (++pos[k] < sols[k].size()) break;
            pos[k] = 0;
            ++k;
        }
        if (k == I.n1) break;
    }
    return res;
}

}  // namespace stabrank
