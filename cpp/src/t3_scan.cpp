#include "stabrank/t3_scan.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stabrank {

namespace {

// Same table geometry and hash as verify_challenge/cert_t3m3_rank7.py
// (_hash, TSIZE), so the grouping and therefore the stored records agree
// with the numba kernel bit for bit.
constexpr int64_t TSIZE = 1 << 17;
constexpr int64_t TMASK = TSIZE - 1;
constexpr uint64_t MULT[8] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull,
                              0x27D4EB2F165667C5ull, 0x94D049BB133111EBull, 0xBF58476D1CE4E5B9ull,
                              0x2545F4914F6CDD1Dull, 0xD6E8FEB86659FD93ull};

enum Counter { C_STEPS_NOMINAL, C_STEPS_DONE, C_CAND, C_DECIDED, C_FOUND, C_SPUR, C_OVER, C_NPAR,
               C_NZ, C_PAIRS };

inline int64_t mod(int64_t x, int64_t p) {
    x %= p;
    return x < 0 ? x + p : x;
}

// The scan's prime is fixed at compile time so that the reductions in the
// inner loop compile to multiply-shift sequences instead of divisions.
constexpr int64_t ELL = 65521;
inline int64_t mod_ell(int64_t x) {
    x %= ELL;
    return x < 0 ? x + ELL : x;
}

inline uint64_t hash_vec(const int64_t* v, int64_t n) {
    uint64_t h = 0x51ED270B27C6D0F5ull;
    for (int64_t d = 0; d < n; ++d) {
        h ^= (static_cast<uint64_t>(v[d]) + 0x9E37ull) * MULT[d];
        h = (h ^ (h >> 29)) * 0xBF58476D1CE4E5B9ull;
    }
    return h ^ (h >> 32);
}

int64_t inv_mod(int64_t a, int64_t p) {
    int64_t t = 0, newt = 1, r = p, newr = mod(a, p);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = t - q * newt;
        t = newt;
        newt = tmp;
        tmp = r - q * newr;
        r = newr;
        newr = tmp;
    }
    return t < 0 ? t + p : t;
}

// Reduced row echelon form mod p of the first n rows of M (n x dim, row-major),
// in place; returns the rank and the pivot columns.
int64_t rref_inplace(int64_t* M, int64_t n, int64_t dim, int64_t p, int64_t* pivcol) {
    int64_t r = 0;
    for (int64_t c = 0; c < dim && r < n; ++c) {
        int64_t pr = -1;
        for (int64_t t = r; t < n; ++t)
            if (M[t * dim + c] != 0) { pr = t; break; }
        if (pr < 0) continue;
        if (pr != r)
            for (int64_t d = 0; d < dim; ++d) std::swap(M[r * dim + d], M[pr * dim + d]);
        const int64_t f = inv_mod(M[r * dim + c], p);
        for (int64_t d = 0; d < dim; ++d) M[r * dim + d] = mod(M[r * dim + d] * f, p);
        for (int64_t t = 0; t < n; ++t) {
            if (t == r || M[t * dim + c] == 0) continue;
            const int64_t g = M[t * dim + c];
            for (int64_t d = 0; d < dim; ++d) M[t * dim + d] = mod(M[t * dim + d] - g * M[r * dim + d], p);
        }
        pivcol[r++] = c;
    }
    return r;
}

bool in_rowspace(const int64_t* M, int64_t r, const int64_t* pivcol, const int64_t* T2, int64_t nT,
                 int64_t dim, int64_t p, int64_t* buf) {
    for (int64_t s = 0; s < nT; ++s) {
        for (int64_t d = 0; d < dim; ++d) buf[d] = mod(T2[s * dim + d], p);
        for (int64_t t = 0; t < r; ++t) {
            const int64_t g = buf[pivcol[t]];
            if (g == 0) continue;
            for (int64_t d = 0; d < dim; ++d) buf[d] = mod(buf[d] - g * M[t * dim + d], p);
        }
        for (int64_t d = 0; d < dim; ++d)
            if (buf[d] != 0) return false;
    }
    return true;
}

}  // namespace

void t3_scan_pairs(const T3ScanInputs& in, const T3ScanOutputs& out) {
    const int64_t N = in.N, D = in.D, D1 = D - 1, dim = in.dim, maxm = out.maxm;
    if (D < 4) throw std::invalid_argument("t3_scan_pairs needs at least four projected coordinates");
    if (in.ell != ELL) throw std::invalid_argument("t3_scan_pairs is compiled for ell = 65521");
    if (out.hist_n < 1) throw std::invalid_argument("empty histogram");
    const int64_t HIST = out.hist_n;
    int64_t* counters = out.counters;

    // one table entry per cache line probe: stamp, key and count together
    struct Slot { int64_t stamp; uint64_t key; int64_t cnt; };
    std::vector<uint64_t> keys(N, 0);
    std::vector<Slot> table(TSIZE, Slot{0, 0, 0});
    std::vector<int64_t> slot_of(N, 0);
    int64_t stamp = 0;

    // Q1: images reduced modulo q_i (N x D1)
    const int64_t* qi = in.PD + in.i * D;
    int64_t a = -1;
    for (int64_t d = 0; d < D; ++d) if (qi[d] != 0) { a = d; break; }
    if (a < 0) throw std::invalid_argument("first pivot has zero image");
    const int64_t ia = in.inv[qi[a]];
    std::vector<int64_t> Q1(N * D1);
    for (int64_t s = 0; s < N; ++s) {
        const int64_t* row = in.PD + s * D;
        const int64_t c = mod_ell(row[a] * ia);
        int64_t col = 0;
        for (int64_t d = 0; d < D; ++d) {
            if (d == a) continue;
            Q1[s * D1 + col++] = mod_ell(row[d] - c * qi[d]);
        }
    }
    std::vector<int8_t> skip1(in.isfree, in.isfree + N);
    std::vector<int64_t> fixed(N);
    int64_t nfixed = 0, npar = 0;
    for (int64_t s = 0; s < N; ++s) {
        if (in.isfree[s] != 0) {
            fixed[nfixed++] = s;
        } else if (s != in.i) {
            bool z = true;
            for (int64_t d = 0; d < D1; ++d) if (Q1[s * D1 + d] != 0) { z = false; break; }
            if (z) { skip1[s] = 1; fixed[nfixed++] = s; ++npar; }
        }
    }
    skip1[in.i] = 1;
    counters[C_NPAR] = npar;

    const int64_t D2 = D1 - 1, D3 = D1 - 2;
    std::vector<int64_t> Q2(N * D2);
    std::vector<int8_t> skip2(N);
    std::vector<int64_t> zbuf(N), members(maxm), work(maxm * dim), pivcol(maxm), tbuf(dim);
    int64_t q3[8], v[8];

    for (int64_t t = 0; t < in.nj; ++t) {
        const int64_t j = in.jlist[t];
        const int64_t ki = in.kok_index[t];
        const int8_t* kok = ki < 0 ? nullptr : in.kok_rows + ki * N;
        for (int64_t k = j + 1; k < N; ++k)
            if (!kok || kok[k] != 0) counters[C_STEPS_NOMINAL] += N - k - 1;
        counters[C_PAIRS] += 1;
        if (skip1[j] != 0) continue;
        const int64_t* qj = &Q1[j * D1];
        int64_t b = -1;
        for (int64_t d = 0; d < D1; ++d) if (qj[d] != 0) { b = d; break; }
        const int64_t ib = in.inv[qj[b]];
        skip2 = skip1;
        int64_t nz = 0;
        for (int64_t l = j + 1; l < N; ++l) {
            if (skip1[l] != 0) continue;
            const int64_t* ql = &Q1[l * D1];
            const int64_t c = mod_ell(ql[b] * ib);
            int64_t col = 0;
            bool zero = true;
            for (int64_t d = 0; d < D1; ++d) {
                if (d == b) continue;
                const int64_t val = mod_ell(ql[d] - c * qj[d]);
                Q2[l * D2 + col++] = val;
                if (val != 0) zero = false;
            }
            if (zero) { zbuf[nz++] = l; skip2[l] = 1; }
        }
        counters[C_NZ] += nz;
        const int64_t nfree_j = nfixed + nz;
        for (int64_t k = j + 1; k < N; ++k) {
            if (kok && kok[k] == 0) continue;
            if (skip2[k] != 0) continue;
            counters[C_STEPS_DONE] += N - k - 1;
            const int64_t* qk = &Q2[k * D2];
            int64_t b2 = -1;
            for (int64_t d = 0; d < D2; ++d) if (qk[d] != 0) { b2 = d; break; }
            const int64_t ib2 = in.inv[qk[b2]];
            ++stamp;
            int64_t nzero = 0;
            // fused: reduce modulo q_k, canonicalise, hash and count
            for (int64_t l = k + 1; l < N; ++l) {
                if (skip2[l] != 0) { slot_of[l] = -1; continue; }
                const int64_t* ql = &Q2[l * D2];
                const int64_t c = mod_ell(ql[b2] * ib2);
                int64_t col = 0, lead = -1;
                for (int64_t d = 0; d < D2; ++d) {
                    if (d == b2) continue;
                    q3[col] = mod_ell(ql[d] - c * qk[d]);
                    if (lead < 0 && q3[col] != 0) lead = col;
                    ++col;
                }
                if (lead < 0) { ++nzero; slot_of[l] = -2; keys[l] = 0; continue; }
                const int64_t s = in.inv[q3[lead]];
                for (int64_t d = 0; d < D3; ++d) v[d] = mod_ell(q3[d] * s);
                uint64_t h = hash_vec(v, D3);
                if (h == 0) h = 1;
                keys[l] = h;
                int64_t slot = static_cast<int64_t>(h & static_cast<uint64_t>(TMASK));
                while (true) {
                    Slot& e = table[slot];
                    if (e.stamp != stamp) { e.stamp = stamp; e.key = h; e.cnt = 1; break; }
                    if (e.key == h) { e.cnt += 1; break; }
                    slot = (slot + 1) & TMASK;
                }
                slot_of[l] = slot;
            }
            const int64_t nzero_total = nzero + nfree_j;
            for (int pass = 0; pass < 2; ++pass) {
                if (pass == 0 && nzero_total < in.need) continue;
                const int64_t lo = pass == 0 ? k : k + 1, hi = pass == 0 ? k + 1 : N;
                for (int64_t l = lo; l < hi; ++l) {
                    int64_t ngroup = 0;
                    uint64_t h = 0;
                    if (pass == 1) {
                        const int64_t s = slot_of[l];
                        if (s < 0) continue;
                        if (!(table[s].cnt > 0 && table[s].cnt + nzero_total >= in.need)) continue;
                        ngroup = table[s].cnt;
                        table[s].cnt = -1;
                        h = keys[l];
                    }
                    counters[C_CAND] += 1;
                    const int64_t a0 = nzero_total < HIST ? nzero_total : HIST - 1;
                    const int64_t a1 = ngroup < HIST ? ngroup : HIST - 1;
                    out.hist[a0 * HIST + a1] += 1;
                    int64_t n = 3;
                    bool over = false;
                    members[0] = in.i; members[1] = j; members[2] = k;
                    auto push = [&](int64_t x) { if (n < maxm) members[n] = x; else over = true; ++n; };
                    for (int64_t ll = k + 1; ll < N; ++ll) {
                        const int64_t sl = slot_of[ll];
                        if (sl == -2 || (ngroup > 0 && sl >= 0 && keys[ll] == h)) push(ll);
                    }
                    for (int64_t u = 0; u < nz; ++u) push(zbuf[u]);
                    for (int64_t u = 0; u < nfixed; ++u) push(fixed[u]);
                    if (over) {
                        const int64_t o = counters[C_OVER];
                        if (o < out.max_over) { out.over_meta[o * 3] = j; out.over_meta[o * 3 + 1] = k; out.over_meta[o * 3 + 2] = n; }
                        counters[C_OVER] += 1;
                        continue;
                    }
                    for (int64_t u = 0; u < n; ++u)
                        for (int64_t d = 0; d < dim; ++d) work[u * dim + d] = in.E2[members[u] * dim + d];
                    const int64_t r = rref_inplace(work.data(), n, dim, in.ell2, pivcol.data());
                    if (r >= 8) {
                        const int64_t o = counters[C_SPUR];
                        if (o < out.max_spur) {
                            for (int64_t u = 0; u < n; ++u) out.spur_buf[o * maxm + u] = members[u];
                            out.spur_len[o] = n;
                            out.spur_meta[o * 3] = j; out.spur_meta[o * 3 + 1] = k; out.spur_meta[o * 3 + 2] = r;
                        }
                        counters[C_SPUR] += 1;
                        continue;
                    }
                    counters[C_DECIDED] += 1;
                    if (in_rowspace(work.data(), r, pivcol.data(), in.T2, in.nT, dim, in.ell2, tbuf.data())) {
                        const int64_t o = counters[C_FOUND];
                        if (o < out.max_found) {
                            for (int64_t u = 0; u < n; ++u) out.found_buf[o * maxm + u] = members[u];
                            out.found_len[o] = n;
                            out.found_meta[o * 3] = j; out.found_meta[o * 3 + 1] = k; out.found_meta[o * 3 + 2] = r;
                        }
                        counters[C_FOUND] += 1;
                    }
                }
            }
        }
    }
}

}  // namespace stabrank
