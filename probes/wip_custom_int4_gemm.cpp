// my_gemm4: int4 GEMM with the scale/epilogue hoisted out of the k loop.
//
// The vendor kernel (gemm_kernel_i8i4_m4) applies a per-32-weight scale plus a
// register shuffle on every k block -- about 16 of its ~34 instructions -- and
// leaves the dot unit idle at 45 GMAC/s of the 413 GMAC/s it can sustain alone.
// With one weight scale per 256-weight group the int32 accumulators can run
// across eight k blocks, so the epilogue runs once per group instead.
//
// Correctness is checked against the vendor kernel on identical data, so the
// exact nibble/lane packing only has to be matched, not understood.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
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
void ggml_abort(const char * f, int l, const char * m, ...) { fprintf(stderr, "abort\n"); abort(); }
}

using namespace spacemit_kernels;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static const int KGRP = 8;             // k blocks per scale group (256 weights)

static void my_gemm4(const int8_t * A, const uint8_t * B, const _Float16 * S,
                     float * out, size_t ldc, size_t k_blks, const float * a_scale) {
    const size_t ngroups = k_blks / KGRP;
    asm volatile(
        "vsetvli t0, x0, e32, m1          \n\t"
        "vxor.vv v28, v28, v28            \n\t"
        "vxor.vv v29, v29, v29            \n\t"
        "vxor.vv v30, v30, v30            \n\t"
        "vxor.vv v31, v31, v31            \n\t"
        "mv     t3, %[NG]                 \n\t"
        "GRP_LOOP%=:                      \n\t"
        "vxor.vv v16, v16, v16            \n\t"
        "vxor.vv v18, v18, v18            \n\t"
        "vxor.vv v20, v20, v20            \n\t"
        "vxor.vv v22, v22, v22            \n\t"
        "mv     t4, %[KGRP]               \n\t"
        "BLK_LOOP%=:                      \n\t"
        "vsetvli t0, x0, e8, m1           \n\t"
        "vl1r.v v0, (%[A])                \n\t"
        "addi   %[A], %[A], 128           \n\t"
        "vl4r.v v4, (%[B])                \n\t"
        "addi   %[B], %[B], 512           \n\t"
        "vsrl.vi    v1, v0, 4             \n\t"
        "vnpack4.vv v12, v0, v1, 3        \n\t"
        "vupack.vv  v2, v12, v12, 2       \n\t"
        "vsetvli t0, x0, e32, m1          \n\t"
        "vmadotsu v16, v3, v4, i4         \n\t"
        "vmadotsu v18, v3, v5, i4         \n\t"
        "vmadotsu v20, v3, v6, i4         \n\t"
        "vmadotsu v22, v3, v7, i4         \n\t"
        "vsll.vi  v16, v16, 4             \n\t"
        "vsll.vi  v18, v18, 4             \n\t"
        "vsll.vi  v20, v20, 4             \n\t"
        "vsll.vi  v22, v22, 4             \n\t"
        "vmadotu  v16, v2, v4, i4         \n\t"
        "vmadotu  v18, v2, v5, i4         \n\t"
        "vmadotu  v20, v2, v6, i4         \n\t"
        "vmadotu  v22, v2, v7, i4         \n\t"
        "addi   t4, t4, -1                \n\t"
        "bgtz   t4, BLK_LOOP%=            \n\t"
        "vsetvli t0, x0, e16, mf2         \n\t"
        "vle16.v v8, (%[S])               \n\t"
        "addi    %[S], %[S], 64           \n\t"
        "vsetvli t0, x0, e32, m1          \n\t"
        "vfwcvt.f.f.v v14, v8             \n\t"
        "vfcvt.f.x.v  v16, v16            \n\t"
        "vfcvt.f.x.v  v18, v18            \n\t"
        "vfcvt.f.x.v  v20, v20            \n\t"
        "vfcvt.f.x.v  v22, v22            \n\t"
        "vfmacc.vv    v28, v16, v14       \n\t"
        "vfmacc.vv    v29, v18, v14       \n\t"
        "vfmacc.vv    v30, v20, v14       \n\t"
        "vfmacc.vv    v31, v22, v14       \n\t"
        "addi   t3, t3, -1                \n\t"
        "bgtz   t3, GRP_LOOP%=            \n\t"
        : [A] "+r"(A), [B] "+r"(B), [S] "+r"(S)
        : [NG] "r"(ngroups), [KGRP] "r"(KGRP)
        : "t0", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v12", "v14",
          "v16", "v18", "v20", "v22", "v28", "v29", "v30", "v31", "memory");

    const float s0 = a_scale[0], s1 = a_scale[1], s2 = a_scale[2], s3 = a_scale[3];
    asm volatile(
        "vsetvli t0, x0, e32, m1     \n\t"
        "vfmul.vf v28, v28, %[s0]    \n\t"
        "vfmul.vf v29, v29, %[s1]    \n\t"
        "vfmul.vf v30, v30, %[s2]    \n\t"
        "vfmul.vf v31, v31, %[s3]    \n\t"
        "vse32.v  v28, (%[O])        \n\t"
        "add  t2, %[LDC], %[O]       \n\t"
        "vse32.v  v29, (t2)          \n\t"
        "add  t3, %[LDC], t2         \n\t"
        "vse32.v  v30, (t3)          \n\t"
        "add  t2, %[LDC], t3         \n\t"
        "vse32.v  v31, (t2)          \n\t"
        :
        : [O] "r"(out), [LDC] "r"(ldc * 4), [s0] "f"(s0), [s1] "f"(s1), [s2] "f"(s2), [s3] "f"(s3)
        : "t0", "t2", "t3", "v28", "v29", "v30", "v31", "memory");
}

static void quant_a_rows(const float * a, int rows, int64_t K, int8_t * out, float * scale) {
    for (int r = 0; r < rows; ++r) {
        float mx = 0;
        for (int64_t k = 0; k < K; ++k) mx = fmaxf(mx, fabsf(a[(int64_t) r * K + k]));
        const float s = mx > 0 ? mx / 127.0f : 1.0f;
        scale[r] = s;
        for (int64_t k = 0; k < K; ++k) {
            int v = (int) lrintf(a[(int64_t) r * K + k] / s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[(int64_t) r * K + k] = (int8_t) v;
        }
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    const int64_t K = 2048, KB = K / 32;
    const int NCOL = 32;

    static float a[4 * 8192];
    for (int r = 0; r < 4; ++r)
        for (int64_t k = 0; k < K; ++k) a[r * K + k] = sinf(0.013f * (float) (k + 31 * r)) * 3.0f;
    static int8_t aq[4 * 8192];
    float ascale[4];
    quant_a_rows(a, 4, K, aq, ascale);
    // A as the dot unit wants it: per k block, 4 rows x 32 int8 contiguous, so the
    // record stride the kernel sees is 128 B (no scale/sum prefix).
    static int8_t apack[64 * 128];
    for (int64_t kb = 0; kb < KB; ++kb)
        for (int r = 0; r < 4; ++r)
            memcpy(apack + kb * 128 + r * 32, aq + r * K + kb * 32, 32);

    static float w[32][2048];
    for (int c = 0; c < NCOL; ++c)
        for (int64_t k = 0; k < K; ++k) w[c][k] = sinf(0.0007f * (float) (k * (c + 3) + 17)) * 1.7f;

    static uint8_t b_nib[64 * 512];
    static _Float16 b_sc[8 * 32];
    memset(b_nib, 0, sizeof b_nib);
    for (int64_t kb = 0; kb < KB; ++kb) {
        const int64_t g = kb / KGRP;
        if (kb % KGRP == 0)
            for (int c = 0; c < NCOL; ++c) {
                float gm = 0;
                for (int64_t k = g * 256; k < g * 256 + 256; ++k) gm = fmaxf(gm, fabsf(w[c][k]));
                b_sc[g * 32 + c] = (_Float16) (gm > 0 ? gm / 7.0f : 1.0f);
            }
        for (int c = 0; c < NCOL; ++c) {
            const float s = (float) b_sc[g * 32 + c];
            uint8_t * d = b_nib + kb * 512 + c * 16;
            for (int j = 0; j < 16; ++j) {
                int lo = (int) lrintf(w[c][kb * 32 + j] / s);
                int hi = (int) lrintf(w[c][kb * 32 + j + 16] / s);
                lo = lo < -8 ? -8 : (lo > 7 ? 7 : lo);
                hi = hi < -8 ? -8 : (hi > 7 ? 7 : hi);
                d[j] = (uint8_t) ((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }

    // vendor-format inputs: A 152 B per k block, B [32 f16 scales][32 x 16 B] per k block
    static uint8_t va[64 * 152];
    static uint8_t vb[64 * 576];
    memset(va, 0, sizeof va);
    memset(vb, 0, sizeof vb);
    for (int64_t kb = 0; kb < KB; ++kb) {
        uint8_t * rec = va + kb * 152;
        float * sc = (float *) rec;
        int16_t * sum = (int16_t *) (rec + 16);
        int8_t * dat = (int8_t *) (rec + 24);
        for (int r = 0; r < 4; ++r) {
            sc[r] = ascale[r];
            int s = 0;
            for (int j = 0; j < 32; ++j) {
                dat[r * 32 + j] = aq[r * K + kb * 32 + j];
                s += aq[r * K + kb * 32 + j];
            }
            sum[r] = (int16_t) (-s);
        }
        for (int c = 0; c < NCOL; ++c) {
            *(_Float16 *) (vb + kb * 576 + c * 2) = b_sc[(kb / KGRP) * 32 + c];
            memcpy(vb + kb * 576 + 64 + c * 16, b_nib + kb * 512 + c * 16, 16);
        }
    }

    static float mine[4 * 32], ref[4 * 32];
    memset(mine, 0, sizeof mine);
    memset(ref, 0, sizeof ref);
    my_gemm4(apack, b_nib, b_sc, mine, 32, KB, ascale);
    ime2::gemm_kernel_i8i4(32, va, vb, nullptr, ref, 4, 32, (size_t) KB, 32);

    printf("mine (row0, cols 0..5):");
    for (int c = 0; c < 6; ++c) printf(" %9.4f", mine[c]);
    printf("\nref  (row0, cols 0..5):");
    for (int c = 0; c < 6; ++c) printf(" %9.4f", ref[c]);
    printf("\n");

    double worst = 0, peak = 0;
    int wr = -1, wc = -1;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 32; ++c) {
            const double d = fabs((double) mine[r * 32 + c] - (double) ref[r * 32 + c]);
            if (d > worst) { worst = d; wr = r; wc = c; }
            peak = fmax(peak, fabs((double) ref[r * 32 + c]));
        }
    printf("worst |mine-ref| = %.4e at (r=%d,c=%d), peak %.4f, relative %.2e\n",
           worst, wr, wc, peak, peak > 0 ? worst / peak : 0.0);

    // throughput, both kernels, same data
    const int iters = 2000;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = now_s();
        for (int it = 0; it < iters; ++it) {
            if (pass == 0) {
                const int8_t * A2 = aq; const uint8_t * B2 = b_nib; const _Float16 * S2 = b_sc;
                my_gemm4(A2, B2, S2, mine, 32, KB, ascale);
            } else {
                const uint8_t * A2 = va; const uint8_t * B2 = vb;
                ime2::gemm_kernel_i8i4(32, A2, B2, nullptr, ref, 4, 32, (size_t) KB, 32);
            }
        }
        const double dt = now_s() - t0;
        const double macs = (double) iters * 4.0 * 32.0 * (double) K;
        printf("%s: %.1f GMAC/s  (%.1f MAC/cycle@1.6G)\n", pass == 0 ? "mine " : "vendor",
               macs / dt / 1e9, macs / dt / 1.6e9);
    }
    return 0;
}
