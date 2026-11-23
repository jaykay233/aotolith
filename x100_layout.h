// X100 int8 GEMM packing, shared by the board benchmark and a host-side check.
//
// One vmadot on X100 (VLEN=256, vlenb=32) is 2 rows x 4 columns x 8 depth = 64
// MACs. From probe_x100_map.cpp, with A supplying 32 bytes of a row pair and B 32
// bytes of a column group:
//
//     lane (rp*4 + cin) = sum_d A[rp*8 + d] * B[cin*8 + d]
//
// so lanes 0..3 are row 0 columns 0..3 and lanes 4..7 are row 1 columns 0..3, and
// A's bytes 16..31 are never read.
//
// The kernel keeps 8 accumulators, (row pair, column group), which covers 4 rows x
// 16 columns. Each 8-deep step needs 2 A registers and 4 B registers, so the packed
// layouts place exactly those adjacent and one pointer walk serves both:
//
//   A   4 rows x K  ->  per depth step 64 B:  [rowpair][32 B]
//       element (r,k) at (k/8)*64 + (r/2)*32 + (r%2)*8 + (k%8)
//   B  16 cols x K  ->  per depth step 128 B: [colgroup][32 B]
//       element (c,k) at (k/8)*128 + (c/4)*32 + (c%4)*8 + (k%8)
//
// That is 192 B fed per 512 MAC, 0.375 B/MAC, the shape probe_x100_gemm.cpp
// measured at 76.7 GMAC/s on a single core.
//
// This header is deliberately free of inline assembly so it compiles on the host:
// the packing arithmetic and the geometry are the parts that go wrong silently, and
// they can both be checked on a laptop against exact integer dots.
#ifndef X100_LAYOUT_H
#define X100_LAYOUT_H

#include <cstdint>
#include <cstring>

namespace x100 {

const int ROWS = 4;      // rows per A block
const int COLS = 16;     // columns per B block
const int DEPTH = 8;     // k per vmadot step
const int ASTEP = 64;    // bytes of A per depth step
const int BSTEP = 128;   // bytes of B per depth step

// Pack `rows` rows of `a` (row stride `lda`, `k` columns) into blocks of ROWS.
// `k` must be a multiple of DEPTH and `rows` a multiple of ROWS.
inline void pack_a(int8_t * dst, const int8_t * a, int64_t rows, int64_t k, int64_t lda) {
    const int64_t nstep = k / DEPTH;
    int64_t p = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += ROWS) {
        for (int64_t ds = 0; ds < nstep; ++ds) {
            int8_t * blk = dst + p; p += ASTEP;
            for (int ri = 0; ri < ROWS; ++ri) {
                const int rp = ri / 2, rr = ri % 2;
                const int8_t * src = a + (r0 + ri) * lda + ds * DEPTH;
                int8_t * out = blk + rp * 32 + rr * 8;
                for (int d = 0; d < DEPTH; ++d) out[d] = src[d];
            }
        }
    }
}

// Pack `ncols` columns of `b` (row stride `ldb`, `k` columns) into blocks of COLS.
// b is row-major n x k, i.e. one row per output column.
inline void pack_b(int8_t * dst, const int8_t * b, int64_t ncols, int64_t k, int64_t ldb) {
    const int64_t nstep = k / DEPTH;
    int64_t p = 0;
    for (int64_t c0 = 0; c0 < ncols; c0 += COLS) {
        for (int64_t ds = 0; ds < nstep; ++ds) {
            int8_t * blk = dst + p; p += BSTEP;
            for (int ci = 0; ci < COLS; ++ci) {
                const int cg = ci / 4, cin = ci % 4;
                const int8_t * src = b + (c0 + ci) * ldb + ds * DEPTH;
                int8_t * out = blk + cg * 32 + cin * 8;
                for (int d = 0; d < DEPTH; ++d) out[d] = src[d];
            }
        }
    }
}

// Software model of the kernel: walk the packed blocks the way the asm does (one
// 64-byte A step and one 128-byte B step per iteration) and apply the lane map.
// out is ROWS x COLS int32, row-major. This is the oracle the board result is
// compared against, so it must mirror the asm exactly rather than share code.
inline void sim_gemm(int32_t * out, const int8_t * ablk, const int8_t * bblk, int64_t k) {
    int32_t acc[2][4][8];
    memset(acc, 0, sizeof acc);
    const int64_t nstep = k / DEPTH;
    const int8_t * ap = ablk;
    const int8_t * bp = bblk;
    for (int64_t s = 0; s < nstep; ++s) {
        const int8_t * a0 = ap;          // row pair 0
        const int8_t * a1 = ap + 32;     // row pair 1
        for (int cg = 0; cg < 4; ++cg) {
            const int8_t * b = bp + cg * 32;
            for (int rp = 0; rp < 2; ++rp) {
                const int8_t * a = rp ? a1 : a0;
                for (int lane = 0; lane < 8; ++lane) {
                    const int rr = lane / 4, cin = lane % 4;
                    int32_t v = 0;
                    for (int d = 0; d < DEPTH; ++d) v += a[rr * 8 + d] * b[cin * 8 + d];
                    acc[rp][cg][lane] += v;
                }
            }
        }
        ap += ASTEP;
        bp += BSTEP;
    }
    for (int rp = 0; rp < 2; ++rp)
        for (int cg = 0; cg < 4; ++cg)
            for (int lane = 0; lane < 8; ++lane) {
                const int r = rp * 2 + lane / 4;
                const int c = cg * 4 + lane % 4;
                out[r * COLS + c] = acc[rp][cg][lane];
            }
}

}  // namespace x100

#endif
