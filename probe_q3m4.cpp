// Does gemm_kernel_i8i3k accept count_m = 4, and with what A record pitch?
//
// Prefill's vendor path runs four activation rows per B read (that is most of why
// it reaches 132 tok/s), and it currently hard-codes the 4-bit kernel and the
// 576-byte Q4_0 block stride. To run the 3-bit blob there instead, the 3-bit
// kernel has to take the same four rows. Its one-row A record is 292 B per K256
// superblock; this measures whether four consecutive records are what it expects.
//
// A row r is one-hot at k = kr, so C[r][j] must equal tile[j][kr]: any wrong pitch
// or row-major/row-minor swap puts another row's weight there.
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
    setvbuf(stdout, nullptr, _IONBF, 0);
    {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8, &s);
        sched_setaffinity(0, sizeof s, &s);
    }
    static uint8_t blk[Q3_BLK], arec[8192];
    static float tile[32][Q3_K], a[4][Q3_K], c[4 * 64];
    unsigned seed = 4242;
    for (int j = 0; j < 32; ++j)
        for (int k = 0; k < Q3_K; ++k) {
            seed = seed * 1103515245u + 12345u;
            tile[j][k] = ((float) ((seed >> 16) & 0x7fff) / 32767.0f * 2.0f - 1.0f) * 3.0f;
        }
    memset(blk, 0, sizeof blk);
    pack(blk, tile);

    const int kr[4] = { 3, 71, 140, 251 };
    for (int r = 0; r < 4; ++r) {
        for (int k = 0; k < Q3_K; ++k) a[r][k] = 0.0f;
        a[r][kr[r]] = 1.0f;
    }

    for (int pitch = 288; pitch <= 296; pitch += 2) {
        memset(arec, 0, sizeof arec);
        for (int r = 0; r < 4; ++r)
            spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a[r], Q3_K, arec + r * pitch);
        memset(c, 0, sizeof c);
        spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blk, nullptr, c, 4, 32, 1, 32);
        double worst = 0;
        for (int r = 0; r < 4; ++r)
            for (int j = 0; j < 32; ++j) {
                const double e = fabs((double) c[r * 32 + j] - (double) tile[j][kr[r]]);
                if (e > worst) worst = e;
            }
        printf("count_m=4 A pitch %3d: worst |C - tile| = %.4f%s\n",
               pitch, worst, worst < 0.15 ? "   <- rows read correctly" : "");
        if (pitch == 292) {
            for (int r = 0; r < 4; ++r) {
                printf("  m=%d want tile[%d..%d][k=%d] = ", r, 0, 3, kr[r]);
                for (int j = 0; j < 3; ++j) printf("%7.3f ", tile[j][kr[r]]);
                printf(" | got C[%d][0..2] = ", r);
                for (int j = 0; j < 3; ++j) printf("%7.3f ", c[r * 32 + j]);
                printf("\n");
            }
        }
    }
    return 0;
}
