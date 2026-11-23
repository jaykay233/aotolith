// Ground-truth map of gemm_kernel_i8i4 (the kernel we are allowed to beat):
// feed it a single one-hot weight nibble and read the output matrix back. This
// avoids guessing at vmadotsu/vmadotu semantics, since whatever the vendor
// kernel does is by definition what the silicon does.
//
// Known from earlier probes: lane = row*8 + col, four accumulators cover
// columns 0-7 / 8-15 / 16-23 / 24-31, and A within a k block is 4 rows x 32 int8
// contiguous (at r*32 + k).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

extern "C" {
size_t   ggml_type_size(int) { return 0; }
long long ggml_nelements(const void *) { return 0; }
long long ggml_nrows(const void *) { return 0; }
bool     ggml_can_repeat(const void *, const void *) { return false; }
bool     ggml_are_same_shape(const void *, const void *) { return false; }
bool     ggml_is_contiguous_rows(const void *) { return true; }
bool     ggml_is_contiguous(const void *) { return true; }
bool     ggml_is_transposed(const void *) { return false; }
size_t   ggml_element_size(const void *) { return 1; }
void ggml_abort(const char * f, int l, const char * m, ...) { abort(); }
}

// One k block of 32. A record: [4 f32 scale][4 i16 sum][4x32 int8] = 152 B.
// B block: [32 f16 scales][32 cols x 16 B] = 576 B.
static const int KB = 16;                       // 16 k blocks = 512 weights

static void run_one(uint8_t * arec, uint8_t * bblk, float * c) {
    memset(c, 0, 4 * 32 * sizeof(float));
    spacemit_kernels::ime2::gemm_kernel_i8i4(32, arec, bblk, nullptr, c, 4, 32, (size_t) KB, 32);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    static uint8_t arec[KB * 152];
    static uint8_t bblk[KB * 576];
    static float c[4 * 32];

    // A: every value = 1, scale 1, sum 0 (bias handled by the kernel)
    memset(arec, 0, sizeof arec);
    for (int blk = 0; blk < KB; ++blk) {
        uint8_t * r = arec + blk * 152;
        for (int i = 0; i < 4; ++i) ((float *) r)[i] = 1.0f;
        for (int i = 0; i < 4; ++i) ((int16_t *) (r + 16))[i] = 0;
        memset(r + 24, 1, 128);
    }

    printf("== A all 1, B all zero: C must be 0 ==\n");
    memset(bblk, 0, sizeof bblk);
    run_one(arec, bblk, c);
    printf("  c[0..7]:"); for (int i = 0; i < 8; ++i) printf(" %.1f", c[i]); printf("\n");

    printf("\n== B all nibbles 1 (0x11), per k block ==\n");
    memset(bblk, 0, sizeof bblk);
    for (int blk = 0; blk < KB; ++blk) {
        uint8_t * r = bblk + blk * 576;
        for (int i = 0; i < 32; ++i) ((_Float16 *) r)[i] = (_Float16) 1.0f;
        memset(r + 64, 0x11, 512);
    }
    run_one(arec, bblk, c);
    printf("  c[0..7]:"); for (int i = 0; i < 8; ++i) printf(" %.1f", c[i]);
    printf("\n  (all 16 k blocks of 32 weights lit -> 512 if both nibbles count)\n");

    printf("\n== one weight nibble lit: byte m of the col-0 16 B group ==\n");
    for (int m = 0; m < 16; ++m) {
        for (int half = 0; half < 2; ++half) {
            memset(bblk, 0, sizeof bblk);
            for (int blk = 0; blk < KB; ++blk) {
                uint8_t * r = bblk + blk * 576;
                for (int i = 0; i < 32; ++i) ((_Float16 *) r)[i] = (_Float16) 1.0f;
                r[64 + m] = (uint8_t) (half ? 0x10 : 0x01);   // column 0 only
            }
            run_one(arec, bblk, c);
            printf("  b[%2d] %s -> C rows:", m, half ? "hi" : "lo");
            for (int rr = 0; rr < 4; ++rr) printf(" %.1f", c[rr * 32 + 0]);
            printf("   (all 16 k blocks lit)\\n");
        }
    }
    return 0;
}
