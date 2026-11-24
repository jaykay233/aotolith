// Does the kernel walk the B blob the way the packer lays it out, once there are
// several 32-column groups and several superblocks in one call?
//
// Real decode calls the kernel with count_n = kstep*32 = 128 columns and k_blks =
// k/256 = 8 or 24. The packer stores [group][superblock], so group g starts at
// g*(k/256)*Q3_BLK. A kernel that instead assumes [superblock][group] gets group 0
// right and every other group wrong, which is invisible in a one-group probe and
// in single-nonzero scans.
//
// Each group's tile is a constant c_g, A is all ones, so C[j] must be
// sum_g c_g for every row j.
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
    setvbuf(stdout, nullptr, _IONBF, 0);
    static float tile[32][Q3_K], c[2048];
    const int cases[][2] = { {1, 1}, {2, 1}, {4, 1}, {4, 8}, {8, 8}, {32, 24} };
    for (int ci = 0; ci < 6; ++ci) {
        const int ng = cases[ci][0], nsup = cases[ci][1];
        const int k = nsup * Q3_K;
        static uint8_t blob[32 * 24 * Q3_BLK];
        static uint8_t arec[64 * 1024];
        static float a[24 * Q3_K];
        memset(blob, 0, sizeof blob);
        memset(arec, 0, sizeof arec);
        for (int i = 0; i < k; ++i) a[i] = 1.0f;
        for (int s = 0; s < nsup; ++s)
            spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a + s * Q3_K, Q3_K, arec + s * 292);
        // A constant per (group, superblock), so a wrong group or a wrong
        // superblock shows a specific value rather than a plausible sum.
        // Output column j belongs to column group j/32, and that group reads
        // nsup superblocks of its own tile, so C[j] must be v_(j/32) * nsup * 256.
        for (int g = 0; g < ng; ++g) {
            for (int s = 0; s < nsup; ++s) {
                const float v = (float) (g + 1) * 0.25f;
                for (int j = 0; j < 32; ++j)
                    for (int kk = 0; kk < Q3_K; ++kk) tile[j][kk] = v;
                pack(blob + ((int64_t) g * nsup + s) * Q3_BLK, tile);
            }
        }
        memset(c, 0, sizeof c);
        spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blob, nullptr, c,
                                                 1, (size_t) (ng * 32), (size_t) nsup, (size_t) (ng * 32));
        double worst = 0;
        int worst_j = -1;
        for (int j = 0; j < ng * 32; ++j) {
            const double want = (double) (j / 32 + 1) * 0.25 * nsup * Q3_K;
            const double e = fabs((double) c[j] - want) / want;
            if (e > worst) { worst = e; worst_j = j; }
        }
        printf("groups %2d superblocks %2d: worst rel %.4f (col %d)%s\n",
               ng, nsup, worst, worst_j, worst < 0.02 ? "   <- group-major walk ok" : "   <-- WRONG");
        if (ng > 1) {
            printf("    per-group first rows: ");
            for (int g = 0; g < ng; ++g) printf("%.2f ", c[g * 32]);
            printf("\n");
        }
    }
    return 0;
}
