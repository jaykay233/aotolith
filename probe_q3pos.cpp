// Is every (row, k) element read back from the right place?
//
// One nonzero at a time: tile[j][k0] = 4, everything else zero. The k16 block that
// contains k0 gets step 4/3 and reproduces 4 exactly; every other block has
// max 0, so its scale is 0 and contributes nothing. C[j] is therefore exactly 4
// for all 32 rows and every k0. A permuted dimension index, a shifted sign bit or
// a mis-strided superblock shows up as a specific k0 failing, which the aggregate
// row-sum tests cannot localise.
//
//   probe_q3pos            (both signs, all 256 positions)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include "ggml_stubs.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}}
namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i3k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                         float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

static const int Q3_K = 256;
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;

static void pack(uint8_t * blk, const float tile[32][Q3_K]) {
    int8_t *   scl = (int8_t *) blk;
    uint16_t * hms = (uint16_t *) (blk + 512);
    uint8_t *  qs  = blk + 512 + 1024;
    _Float16 * s16 = (_Float16 *) (blk + 512 + 1024 + 2048);
    float step[32][16], smax[32] = {0};
    for (int i = 0; i < 32; ++i) {
        for (int kii = 0; kii < 16; ++kii) {
            float mp = 0, mn = 0;
            for (int bi = 0; bi < 16; ++bi) {
                const float v = tile[i][kii * 16 + bi];
                if (v > mp) mp = v;
                if (-v > mn) mn = -v;
            }
            step[i][kii] = fmaxf(mp / 3.0f, mn / 4.0f);
            if (step[i][kii] > smax[i]) smax[i] = step[i][kii];
        }
    }
    for (int i = 0; i < 32; ++i) {
        const float fs = smax[i] > 0 ? smax[i] / 127.0f : 1.0f;
        s16[i] = (_Float16) fs;
        for (int kii = 0; kii < 16; ++kii) {
            int isc = (int) lrintf(step[i][kii] / fs);
            if (isc < 1) isc = 1;
            if (isc > 127) isc = 127;
            scl[kii * 32 + i] = (int8_t) isc;
            const float eff = (float) isc * fs;
            uint16_t mask = 0;
            for (int bi = 0; bi < 16; ++bi) {
                const float v = tile[i][kii * 16 + bi];
                int q3 = eff > 0 ? (int) lrintf(v / eff) : 0;
                if (q3 < -4) q3 = -4;
                if (q3 > 3) q3 = 3;
                const int u = q3 < 0 ? q3 + 4 : q3;
                const int64_t byte = (int64_t) (kii / 4) * 512 + i * 16 + bi;
                qs[byte] = (uint8_t) ((qs[byte] & ~(3 << ((kii % 4) * 2))) | ((u & 3) << ((kii % 4) * 2)));
                if (q3 >= 0) mask |= (uint16_t) (1u << bi);
            }
            hms[kii * 32 + i] = mask;
        }
    }
}

int main(void) {
    {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8, &s);
        sched_setaffinity(0, sizeof s, &s);
    }
    static uint8_t blk[Q3_BLK], arec[4096];
    static float tile[32][Q3_K], a[Q3_K], c[64];
    for (int k = 0; k < Q3_K; ++k) a[k] = 1.0f;
    memset(arec, 0, sizeof arec);
    spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a, Q3_K, arec);

    int bad = 0;
    for (int sign = 0; sign < 2; ++sign)
        for (int k0 = 0; k0 < Q3_K; ++k0) {
            const float v0 = sign ? -4.0f : 4.0f;
            memset(tile, 0, sizeof tile);
            for (int j = 0; j < 32; ++j) tile[j][k0] = v0;
            memset(blk, 0, sizeof blk);
            pack(blk, tile);
            memset(c, 0, sizeof c);
            spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blk, nullptr, c, 1, 32, 1, 32);
            for (int j = 0; j < 32; ++j)
                if (fabs((double) c[j] - v0) > 0.02) {
                    if (bad < 12)
                        printf("k0=%3d sign%s row %2d: want %5.2f got %8.4f\n",
                               k0, sign ? "-" : "+", j, v0, c[j]);
                    ++bad;
                }
        }
    printf("single-nonzero scan: %d of %d (position,row) reads wrong\n", bad, 2 * 256 * 32);
    return 0;
}
