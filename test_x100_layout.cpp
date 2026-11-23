// Host-side check of the X100 packing maths, before it touches the board.
//
// x100_layout.h holds the byte-level layout that the vmadot kernel depends on and
// a software model of the kernel. Neither needs RISC-V intrinsics, so both run
// here: a wrong stride or a swapped row/column index produces a wrong dot product
// rather than a crash, and this is far cheaper to find on a laptop than through a
// board session.
#include "x100_layout.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main() {
    using namespace x100;
    const int64_t rows = 4 * 3, cols = 16 * 2, k = 8 * 5;   // several blocks, odd depth count
    int8_t * a = (int8_t *) aligned_alloc(64, rows * k);
    int8_t * b = (int8_t *) aligned_alloc(64, cols * k);
    int8_t * pa = (int8_t *) calloc(1, rows * k * 2);
    int8_t * pb = (int8_t *) calloc(1, cols * k * 2);
    int32_t * out = (int32_t *) calloc(ROWS * COLS, 4);
    int32_t * naive = (int32_t *) calloc(ROWS * COLS, 4);

    srand(7);
    for (int64_t i = 0; i < rows * k; ++i) a[i] = (int8_t) (rand() % 19 - 9);
    for (int64_t i = 0; i < cols * k; ++i) b[i] = (int8_t) (rand() % 19 - 9);

    pack_a(pa, a, rows, k, k);
    pack_b(pb, b, cols, k, k);

    int bad = 0;
    // block (0,0) and one further in, to prove the block strides as well
    struct { int r0, c0; } blocks[] = { {0, 0}, {ROWS, COLS}, {2 * ROWS, 0} };
    for (auto & bl : blocks) {
        const int64_t ab = (bl.r0 / ROWS) * (k / DEPTH) * ASTEP;
        const int64_t bb = (bl.c0 / COLS) * (k / DEPTH) * BSTEP;
        sim_gemm(out, pa + ab, pb + bb, k);
        for (int r = 0; r < ROWS; ++r)
            for (int c = 0; c < COLS; ++c) {
                int32_t s = 0;
                for (int64_t d = 0; d < k; ++d)
                    s += a[(bl.r0 + r) * k + d] * b[(bl.c0 + c) * k + d];
                naive[r * COLS + c] = s;
                if (out[r * COLS + c] != s) {
                    if (bad < 5)
                        printf("  mismatch block(%d,%d) r=%d c=%d: model %d exact %d\n",
                               bl.r0, bl.c0, r, c, out[r * COLS + c], s);
                    ++bad;
                }
            }
    }

    // A[16..31] is never consumed by vmadot, so the packing must leave it unused.
    int unused_nonzero = 0;
    for (int64_t blk = 0; blk < rows / ROWS; ++blk)
        for (int64_t ds = 0; ds < k / DEPTH; ++ds) {
            const int8_t * p = pa + (blk * (k / DEPTH) + ds) * ASTEP;
            for (int i = 16; i < 32; ++i)
                if (p[i] != 0 && p[32 + i] != 0) ++unused_nonzero;
        }

    printf("packed A %.0f B/block, B %.0f B/block, %lld blocks checked\n",
           (double) ASTEP, (double) BSTEP, (long long) (sizeof blocks[0] / sizeof blocks[0]) * 3);
    printf("mac per block %d, bytes per block %d -> %.3f B/MAC\n",
           ROWS * COLS * DEPTH, ASTEP + BSTEP, (double) (ASTEP + BSTEP) / (ROWS * COLS * DEPTH));
    printf("A bytes 16..31 written: %d (should be 0)\n", unused_nonzero);
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad != 0;
}
