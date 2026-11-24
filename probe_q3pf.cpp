// The real 4-row 3-bit prefill pair: quantize_a_4row_i8k + gemm_kernel_i8i3k_m4.
//
// Prefill currently hard-codes the 4-bit hp kernel and the 576-byte Q4_0 block
// stride, so a -q3 model feeds a 4-bit kernel a 3-bit blob. gemm_kernel_i8i3k is
// a one-row kernel (four rows come back all zero), and the vendor ships a
// separate m4 entry point that takes no zero-point operand. Check it reproduces
// the same weights, and pin down its A record pitch the way probe_q3kern did for
// the one-row kernel.
//
// Each activation row is one-hot at a different k, so C[r][j] must equal
// tile[j][kr[r]] for all four rows.
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
void quantize_a_4row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}}
namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i3k_m4(size_t blk_len, const uint8_t * a, const uint8_t * b,
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
    for (int nsup = 1; nsup <= 8; nsup *= 8) {
        const int k = nsup * Q3_K;
        static uint8_t blob[Q3_BLK], arec[32768];
        static float tile[32][Q3_K], a[4][2048], c[4 * 64];
        for (int j = 0; j < 32; ++j)
            for (int kk = 0; kk < Q3_K; ++kk)
                tile[j][kk] = 0.25f * (float) (j + 1) - 3.0f * ((kk >> 3) & 1);
        for (int s = 0; s < nsup; ++s) pack(blob + s * Q3_BLK, tile);
        // last superblock holds the one-hots, so the record stride is exercised
        const int kr[4] = { 3, 71, 140, 251 };
        for (int r = 0; r < 4; ++r) {
            for (int kk = 0; kk < k; ++kk) a[r][kk] = 0.0f;
            a[r][(nsup - 1) * Q3_K + kr[r]] = 1.0f;
        }
        // one straight 4-row record per superblock; find the stride the kernel expects
        for (int pitch = 290; pitch <= 300; ++pitch) {
            memset(arec, 0, sizeof arec);
            for (int s = 0; s < nsup; ++s) {
                float rows[4][2048];
                for (int r = 0; r < 4; ++r)
                    for (int kk = 0; kk < Q3_K; ++kk) rows[r][kk] = a[r][s * Q3_K + kk];
                spacemit_kernels::rvv::quantize_a_4row_i8k(Q3_K, &rows[0][0], Q3_K, arec + s * pitch);
            }
            memset(c, 0, sizeof c);
            spacemit_kernels::ime2::gemm_kernel_i8i3k_m4(Q3_K, arec, blob, c, 4, 32, (size_t) nsup, 32);
            double worst = 0;
            for (int r = 0; r < 4; ++r)
                for (int j = 0; j < 32; ++j) {
                    const double e = fabs((double) c[r * 32 + j] - (double) tile[j][kr[r]]);
                    if (e > worst) worst = e;
                }
            printf("nsup %d  A pitch %3d: worst |C - tile| = %7.4f%s\n",
                   nsup, pitch, worst, worst < 0.05 ? "   <- m4 reads it right" : "");
        }
    }
    return 0;
}
