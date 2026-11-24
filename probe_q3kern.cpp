// Does the vendor i8i3k kernel read the same 3-bit block layout the packer writes?
//
// The packer and my reading of the layout comment agree (probe_q3pack), which
// only proves self-consistency. This makes the kernel the oracle: a tile whose
// only nonzero per output row j is the value (j+1) at k=7, an activation of all
// ones, so  C[j]  must come back as  j+1  for every j. Any permutation of the
// dimension index, a sign-convention mismatch, or a mis-strided block shows up
// as the wrong row getting the wrong value.
//
//   probe_q3kern
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "ggml_stubs.h"
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}}
namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i3k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                         float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

static const int Q3_K = 256;
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;

// copy of repack_b_q3's arithmetic for one 32x256 superblock
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
    // The IME2 kernels only execute on the A cluster with the AI thread flag set;
    // without it the first vector instruction is an illegal instruction.
    {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8, &s);
        sched_setaffinity(0, sizeof s, &s);
    }
    static uint8_t blk[Q3_BLK];
    static float tile[32][Q3_K];
    static float a[Q3_K];
    static float c[64];
    memset(blk, 0, sizeof blk);
    memset(c, 0, sizeof c);
    for (int j = 0; j < 32; ++j)
        for (int k = 0; k < Q3_K; ++k) tile[j][k] = (k == 7) ? (float) (j + 1) : 0.0f;
    for (int k = 0; k < Q3_K; ++k) a[k] = 1.0f;
    pack(blk, tile);

    static uint8_t arec[4096];
    memset(arec, 0, sizeof arec);
    spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a, Q3_K, arec);
    spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blk, nullptr, c, 1, 32, 1, 32);

    printf("row  want  got\n");
    double worst = 0;
    for (int j = 0; j < 32; ++j) {
        const double e = fabs((double) c[j] - (double) (j + 1));
        if (e > worst) worst = e;
        if (j < 6 || e > 0.01) printf("%3d  %4d  %8.4f%s\n", j, j + 1, c[j], e > 0.01 ? "   <-- mismatch" : "");
    }
    printf("sparse: max absolute error over 32 rows: %.4g\n", worst);

    // Dense tile: with A all ones the kernel must return the row sum, so this
    // compares the packed-and-read weights against the values that went in, at the
    // scale of the whole row rather than one element.
    memset(c, 0, sizeof c);
    unsigned seed = 999;
    for (int j = 0; j < 32; ++j)
        for (int k = 0; k < Q3_K; ++k) {
            seed = seed * 1103515245u + 12345u;
            tile[j][k] = ((float) ((seed >> 16) & 0x7fff) / 32767.0f * 2.0f - 1.0f) * 4.0f;
        }
    memset(blk, 0, sizeof blk);
    pack(blk, tile);
    spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blk, nullptr, c, 1, 32, 1, 32);
    double wrel = 0, wrms = 0;
    for (int j = 0; j < 32; ++j) {
        double s = 0;
        for (int k = 0; k < Q3_K; ++k) s += tile[j][k];
        const double e = fabs((double) c[j] - s);
        const double rel = e / (fabs(s) > 1e-6 ? fabs(s) : 1e-6);
        wrms += e * e;
        if (rel > wrel) wrel = rel;
        if (j < 4) printf("dense row %2d  want %10.4f  got %10.4f\n", j, s, c[j]);
    }
    printf("dense: worst row-sum relative error %.4f, rms %.4f\n", wrel, sqrt(wrms / 32));

    // Two superblocks along k. Block 0 holds zeros, block 1 holds the sparse
    // pattern, and A is all ones over both, so every C[j] must be exactly j+1 with
    // the value coming from k=256+7. If the A record pitch used to walk from one
    // superblock to the next is off by even two bytes, block 1 reads shifted
    // activations and this fails while the one-block test above still passes.
    {
        static uint8_t blk2[2 * Q3_BLK];
        static float tile2[32][Q3_K];
        static float a2[2 * Q3_K];
        static float c2[64];
        for (int pitch = 288; pitch <= 296; pitch += 2) {
            memset(blk2, 0, sizeof blk2);
            memset(a2, 0, sizeof a2);
            for (int k = 0; k < 2 * Q3_K; ++k) a2[k] = 1.0f;
            for (int j = 0; j < 32; ++j)
                for (int k = 0; k < Q3_K; ++k) tile2[j][k] = (k == 7) ? (float) (j + 1) : 0.0f;
            pack(blk2 + Q3_BLK, tile2);      // block 1 only
            static uint8_t arec2[4096];
            memset(arec2, 0, sizeof arec2);
            spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a2, Q3_K, arec2);
            spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a2 + Q3_K, Q3_K, arec2 + pitch);
            memset(c2, 0, sizeof c2);
            spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec2, blk2, nullptr, c2, 1, 32, 2, 32);
            double e = 0;
            for (int j = 0; j < 32; ++j) {
                const double d = fabs((double) c2[j] - (double) (j + 1));
                if (d > e) e = d;
            }
            printf("two superblocks, A pitch %d B: max error %.4g%s\n",
                   pitch, e, e < 0.05 ? "   <- correct" : "");
        }
    }
    // Sign convention. The sparse test above is all-positive, so an inverted hmask
    // bit would pass it and still corrupt every negative weight.
    for (int pass = 0; pass < 3; ++pass) {
        memset(c, 0, sizeof c);
        memset(blk, 0, sizeof blk);
        for (int j = 0; j < 32; ++j)
            for (int k = 0; k < Q3_K; ++k) {
                if (pass == 0)      tile[j][k] = (k == 7) ? -(float) (j + 1) : 0.0f;
                else if (pass == 1) tile[j][k] = (k == 7) ? -(float) (j + 1) : (k == 23 ? (float) (j + 1) : 0.0f);
                else                tile[j][k] = ((k & 1) ? -1.0f : 1.0f) * (float) (j + 1);
            }
        pack(blk, tile);
        spacemit_kernels::ime2::gemm_kernel_i8i3k(Q3_K, arec, blk, nullptr, c, 1, 32, 1, 32);
        double e = 0;
        int bad = -1;
        for (int j = 0; j < 32; ++j) {
            double want = 0;
            for (int k = 0; k < Q3_K; ++k) want += tile[j][k];
            const double d = fabs((double) c[j] - want);
            if (d > e) { e = d; bad = j; }
        }
        printf("sign test %d (%s): max error %.4g%s\n", pass,
               pass == 0 ? "negative only" : pass == 1 ? "negative and positive in one k16" : "alternating",
               e, e < 0.5 ? "" : "   <-- WRONG");
        if (bad >= 0 && e >= 0.5) printf("   row %d want %.4f got %.4f\n", bad, (double) 0, c[bad]);
    }
    return 0;
}
