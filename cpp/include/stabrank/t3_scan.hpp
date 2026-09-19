#pragma once

#include <cstdint>

namespace stabrank {

// Buffers of one three-pivot scan (research/t3_rank7/batch.py::scan_pairs).
// Every array is caller-owned, row-major, int64 unless stated. The kernel
// fills hist, the found/spurious/oversize records and counters exactly as
// the numba kernel does, so a batch record is identical whichever runs.
struct T3ScanInputs {
    const int64_t* PD;        // N x D projected quotient images mod ell
    int64_t N, D;
    const int64_t* inv;       // inverses mod ell, inv[0] unused
    int64_t ell;
    const int64_t* E2;        // N x dim state rows mod ell2
    int64_t dim;
    const int64_t* T2;        // nT x dim target rows mod ell2
    int64_t nT;
    int64_t ell2;
    int64_t i;                // first pivot
    const int64_t* jlist;     // second pivots
    int64_t nj;
    const int64_t* kok_index; // per j: -1 (every k > j admissible) or a row of kok_rows
    const int8_t* kok_rows;   // R x N admissibility masks of the third pivot
    const int8_t* isfree;     // N: states inside the target space
    int64_t need;             // members beyond the three pivots a class needs
};

struct T3ScanOutputs {
    int64_t* hist;            // HIST x HIST counts by (zero members, parallel members)
    int64_t hist_n;
    int64_t* found_buf;       // max_found x maxm member labels
    int64_t* found_len;       // max_found
    int64_t* found_meta;      // max_found x 3: j, k, rank
    int64_t max_found;
    int64_t* spur_buf;        // max_spur x maxm
    int64_t* spur_len;
    int64_t* spur_meta;       // max_spur x 3: j, k, rank
    int64_t max_spur;
    int64_t* over_meta;       // max_over x 3: j, k, members
    int64_t max_over;
    int64_t maxm;             // member capacity per class
    int64_t* counters;        // 10 counters, see batch.py
};

void t3_scan_pairs(const T3ScanInputs& in, const T3ScanOutputs& out);

}  // namespace stabrank
