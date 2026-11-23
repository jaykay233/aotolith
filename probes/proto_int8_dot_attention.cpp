// int8 dot-product attention for 10K prefill, outside the megakernel first.
//
// Rationale: the A100 only executes `vmadot` at full width (512 MAC per
// instruction, 121.9 GMAC/s/core on 4 rows x 64 cols); every FP vector op runs
// on a 128-bit datapath and costs 10-30 cycles. So both attention halves must
// move to the dot unit.
//
// Constraints that shape the layouts, all established by probe:
//   * accumulator destination must be even -> 8 accumulators max
//   * one vmadot reads A bytes [0,64) and B bytes [0,128)
//   * lane = row*8 + col, A[row*16 + d], B[col*16 + d], 16 products per lane
// So QK computes 4 queries x 64 positions per pass over K, and PV contracts 16
// positions per step with 4 queries x 64 dims live. Queries are therefore
// handled 16 at a time: 4 QK passes, 2 PV dim passes, and 16 is a whole number
// of both the 64-position K blocks and the 16-position V blocks.
//
// Scores are written row-major (4 strided stores per accumulator) because both
// softmax and the PV activation packing want one row contiguous.
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>


static inline vfloat32m1_t exp2f_poly(vfloat32m1_t t, size_t vl) {
    vint32m1_t ie = __riscv_vfcvt_x_f_v_i32m1(t, vl);
    const vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, __riscv_vfcvt_f_x_v_f32m1(ie, vl), vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t ex = __riscv_vadd_vx_i32m1(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m1(ex, 23, vl);
    return __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_i32m1_f32m1(ex), vl);
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

enum { HD = 128, CHUNK = 16, QG = 4 };

// ---------------- QK: 4 queries x 64 positions, contract 128 dims
// out[j][r*8 + c] = sum_d q[r][d] * k[64-block pos j*8+c][d]
static void qk4(int8_t * const * acc_out, const int8_t * q8g, const int8_t * kblk, int nks) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "vxor.vv v24, v24, v24          \n\t"
        "vxor.vv v26, v26, v26          \n\t"
        "vxor.vv v28, v28, v28          \n\t"
        "vxor.vv v30, v30, v30          \n\t"
        "mv     t3, %[NK]               \n\t"
        "1:                             \n\t"
        "vsetvli t0, x0, e8, m1         \n\t"
        "vl1r.v  v0,  (%[Q])            \n\t"
        "vl8r.v  v8,  (%[K])            \n\t"
        "addi    %[Q], %[Q], 64         \n\t"
        "addi    %[K], %[K], 1024       \n\t"
        "vsetvli t0, x0, e32, m1        \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v18, v0, v9            \n\t"
        "vmadot  v20, v0, v10           \n\t"
        "vmadot  v22, v0, v11           \n\t"
        "vmadot  v24, v0, v12           \n\t"
        "vmadot  v26, v0, v13           \n\t"
        "vmadot  v28, v0, v14           \n\t"
        "vmadot  v30, v0, v15           \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        "vse32.v v16, (%[O0])           \n\t"
        "vse32.v v18, (%[O1])           \n\t"
        "vse32.v v20, (%[O2])           \n\t"
        "vse32.v v22, (%[O3])           \n\t"
        "vse32.v v24, (%[O4])           \n\t"
        "vse32.v v26, (%[O5])           \n\t"
        "vse32.v v28, (%[O6])           \n\t"
        "vse32.v v30, (%[O7])           \n\t"
        : [Q] "+r"(q8g), [K] "+r"(kblk)
        : [NK] "r"(nks),
          [O0] "r"(acc_out[0]), [O1] "r"(acc_out[1]), [O2] "r"(acc_out[2]), [O3] "r"(acc_out[3]),
          [O4] "r"(acc_out[4]), [O5] "r"(acc_out[5]), [O6] "r"(acc_out[6]), [O7] "r"(acc_out[7])
        : "t0", "t3", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v18", "v20", "v22", "v24", "v26", "v28", "v30", "memory");
}

// ---------------- PV: 4 queries x 64 dims, contract 16 positions
// out[db][r*8 + dc] += sum_p w[r][p] * v[dim 64-block db*8+dc][p]
static void pv4(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "vxor.vv v24, v24, v24          \n\t"
        "vxor.vv v26, v26, v26          \n\t"
        "vxor.vv v28, v28, v28          \n\t"
        "vxor.vv v30, v30, v30          \n\t"
        "mv     t3, %[NP]               \n\t"
        "mv     t4, %[W]                \n\t"
        "mv     t5, %[V]                \n\t"
        "1:                             \n\t"
        "vsetvli t0, x0, e8, m1         \n\t"
        "vl1r.v  v0,  (t4)              \n\t"
        "vl8r.v  v8,  (t5)              \n\t"
        "addi    t4, t4, 128            \n\t"
        "addi    t5, t5, 1024           \n\t"
        "vsetvli t0, x0, e32, m1        \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v18, v0, v9            \n\t"
        "vmadot  v20, v0, v10           \n\t"
        "vmadot  v22, v0, v11           \n\t"
        "vmadot  v24, v0, v12           \n\t"
        "vmadot  v26, v0, v13           \n\t"
        "vmadot  v28, v0, v14           \n\t"
        "vmadot  v30, v0, v15           \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        "vse32.v v16, (%[O0])           \n\t"
        "vse32.v v18, (%[O1])           \n\t"
        "vse32.v v20, (%[O2])           \n\t"
        "vse32.v v22, (%[O3])           \n\t"
        "vse32.v v24, (%[O4])           \n\t"
        "vse32.v v26, (%[O5])           \n\t"
        "vse32.v v28, (%[O6])           \n\t"
        "vse32.v v30, (%[O7])           \n\t"
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t4", "t5",
          "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v18", "v20", "v22", "v24", "v26", "v28", "v30", "memory");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    const int ctx = 4096;
    const int npos = ctx;
    const int npb = npos / 64;              // K blocks of 64
    const int npb16 = npos / 16;            // V blocks of 16

    float * Q = (float *) aligned_alloc(64, (size_t) CHUNK * HD * 4);
    float * K = (float *) aligned_alloc(64, (size_t) npos * HD * 4);
    float * V = (float *) aligned_alloc(64, (size_t) npos * HD * 4);
    for (int r = 0; r < CHUNK; ++r)
        for (int d = 0; d < HD; ++d) Q[r * HD + d] = sinf(0.013f * (float) (r * 131 + d));
    for (int p = 0; p < npos; ++p)
        for (int d = 0; d < HD; ++d) {
            K[p * HD + d] = sinf(0.0071f * (float) (p * 37 + d * 11));
            V[p * HD + d] = cosf(0.0053f * (float) (p * 17 + d * 29));
        }

    // ---- quantise into the dot layouts
    int8_t * k8 = (int8_t *) aligned_alloc(2 << 20, (size_t) npb * 8 * 1024);
    float kmx = 0;
    for (int i = 0; i < npos * HD; ++i) kmx = fmaxf(kmx, fabsf(K[i]));
    const float ksc = kmx > 0 ? kmx / 127.0f : 1.0f;
    for (int p = 0; p < npos; ++p)
        for (int d = 0; d < HD; ++d) {
            const int v = (int) lrintf(K[p * HD + d] / ksc);
            k8[((p / 64) * 8 + d / 16) * 1024 + (p % 64) * 16 + (d % 16)] = (int8_t) v;
        }
    int8_t * v8 = (int8_t *) aligned_alloc(2 << 20, (size_t) npb16 * 2048);
    float vmx = 0;
    for (int i = 0; i < npos * HD; ++i) vmx = fmaxf(vmx, fabsf(V[i]));
    const float vsc = vmx > 0 ? vmx / 127.0f : 1.0f;
    for (int p = 0; p < npos; ++p)
        for (int d = 0; d < HD; ++d) {
            const int v = (int) lrintf(V[p * HD + d] / vsc);
            v8[(p / 16) * 2048 + d * 16 + (p % 16)] = (int8_t) v;
        }

    const int nqg = CHUNK / QG;
    int8_t * q8 = (int8_t *) aligned_alloc(64, (size_t) nqg * 8 * 64);
    float * qsc = (float *) aligned_alloc(64, (size_t) CHUNK * 4);
    for (int r = 0; r < CHUNK; ++r) {
        float mx = 0;
        for (int d = 0; d < HD; ++d) mx = fmaxf(mx, fabsf(Q[r * HD + d]));
        const float s = mx > 0 ? mx / 127.0f : 1.0f;
        qsc[r] = s;
        for (int d = 0; d < HD; ++d) {
            const int v = (int) lrintf(Q[r * HD + d] / s);
            q8[(r / 4 * 8 + d / 16) * 64 + (r % 4) * 16 + (d % 16)] = (int8_t) v;
        }
    }

    static float sc[CHUNK][4096];
    static int32_t raw[8][32];
    static int8_t w8[CHUNK / QG][4096 / 16 * 64 + 64];
    static float wsc[CHUNK];

    const double t_all0 = now_s();

    // ---- QK, 4 queries at a time
    const double tq0 = now_s();
    for (int rep = 0; rep < 8; ++rep)
    for (int g = 0; g < nqg; ++g) {
        float * row[QG] = { sc[g * QG + 0], sc[g * QG + 1], sc[g * QG + 2], sc[g * QG + 3] };
        for (int pb = 0; pb < npb; ++pb) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
            qk4(op, q8 + g * 8 * 64, k8 + pb * 8 * 1024, 8);
            // lane = r*8 + c -> row r, position pb*64 + j*8 + c.
            // Each row's 8 lanes are contiguous in the accumulator, so one
            // 8-lane convert+scale+store per (accumulator, row).
            for (int r = 0; r < QG; ++r) {
                const float s_r = qsc[g * QG + r] * ksc;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&raw[j][r * 8], 8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, 8), s_r, 8);
                    __riscv_vse32_v_f32m1(row[r] + pb * 64 + j * 8, f, 8);
                }
            }
        }
    }
    const double tq1 = now_s();

    // ---- softmax + pack activations for PV
    const double ts0 = now_s();
    for (int rep = 0; rep < 8; ++rep)
    for (int r = 0; r < CHUNK; ++r) {
        float * row = sc[r];
        vfloat32m1_t vm = __riscv_vfmv_v_f_f32m1(-INFINITY, 32);
        for (int i = 0; i < npos; i += 32)
            vm = __riscv_vfmax_vv_f32m1(vm, __riscv_vle32_v_f32m1(row + i, 32), 32);
        const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(
            vm, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), 32));
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 32);
        for (int i = 0; i < npos; i += 32) {
            const vfloat32m1_t t = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), 1.4426950408889634f, 32);
            const vfloat32m1_t e = exp2f_poly(__riscv_vfmax_vf_f32m1(t, -126.0f, 32), 32);
            __riscv_vse32_v_f32m1(row + i, e, 32);
            vs = __riscv_vfadd_vv_f32m1(vs, e, 32);
        }
        const float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
            vs, __riscv_vfmv_v_f_f32m1(0.0f, 1), 32));
        const float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (int i = 0; i < npos; i += 32)
            __riscv_vse32_v_f32m1(row + i, __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), inv, 32), 32);
        // quantise weights: relative to the row max so small weights keep 8 bits
        float wm = 0;
        for (int i = 0; i < npos; ++i) wm = fmaxf(wm, row[i]);
        const float s = wm > 0 ? wm / 127.0f : 1.0f;
        wsc[r] = s;
        const int g = r / QG, rr = r % QG;
        const float rs = 1.0f / s;
        static int8_t tmp8[4096];
        for (int p = 0; p < npos; p += 64) {
            const vfloat32m4_t v = __riscv_vle32_v_f32m4(row + p, 64);
            const vfloat32m4_t q = __riscv_vfmul_vf_f32m4(v, rs, 64);
            const vint16m2_t h = __riscv_vfncvt_x_f_w_i16m2(q, 64);
            const vint8m1_t b = __riscv_vncvt_x_x_w_i8m1(h, 64);
            __riscv_vse8_v_i8m1(tmp8 + p, b, 64);
        }
        for (int p = 0; p < npos; p += 16)
            memcpy(w8[g] + (p / 16) * 64 + rr * 16, tmp8 + p, 16);
    }
    const double ts1 = now_s();

    // ---- PV: 2 dim passes of 64, contract 16 positions per step
    const double tv0 = now_s();
    static float out[CHUNK][HD];
    static int32_t praw[8][32];
    for (int rep = 0; rep < 8; ++rep)
    for (int g = 0; g < nqg; ++g) {
        for (int half = 0; half < 2; ++half) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) praw[j];
            pv4(op, w8[g], v8 + half * 64 * 16, npb16);
            // lane = r*8 + dc, dim = half*64 + j*8 + dc
            for (int r = 0; r < QG; ++r) {
                const float s_r = wsc[g * QG + r] * vsc;
                float * dst = out[g * QG + r] + half * 64;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&praw[j][r * 8], 8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, 8), s_r, 8);
                    __riscv_vse32_v_f32m1(dst + j * 8, f, 8);
                }
            }
        }
    }
    const double tv1 = now_s();
    const double t_all1 = now_s();

    // ---- scalar reference for every row/dim
    double worst = 0, peak = 0;
    int wr = -1, wd = -1;
    for (int r = 0; r < CHUNK; ++r) {
        double mx = -INFINITY;
        static double s[4096];
        for (int p = 0; p < npos; ++p) {
            double a = 0;
            for (int d = 0; d < HD; ++d) a += (double) Q[r * HD + d] * (double) K[p * HD + d];
            s[p] = a; if (a > mx) mx = a;
        }
        double sum = 0;
        for (int p = 0; p < npos; ++p) { s[p] = exp(s[p] - mx); sum += s[p]; }
        for (int d = 0; d < HD; ++d) {
            double a = 0;
            for (int p = 0; p < npos; ++p) a += s[p] * (double) V[p * HD + d];
            a /= sum;
            const double dd = fabs(a - (double) out[r][d]);
            peak = fmax(peak, fabs(a));
            if (dd > worst) { worst = dd; wr = r; wd = d; }
        }
    }

    const double macs = (double) npos * HD * 2.0 * CHUNK * 8.0;
    printf("ctx=%d chunk=%d: %.3f s -> %.1f GMAC/s  [qk %.3f sm %.3f pv %.3f]\n",
           ctx, CHUNK, t_all1 - t_all0, macs / (t_all1 - t_all0) / 1e9,
           tq1 - tq0, ts1 - ts0, tv1 - tv0);
    printf("  qk %.1f  sm %.1f  pv %.1f GMAC/s\n",
           (double) npos * HD * CHUNK * 8.0 / (tq1 - tq0) / 1e9,
           (double) npos * CHUNK * 8.0 / (ts1 - ts0) / 1e9,
           (double) npos * HD * CHUNK * 8.0 / (tv1 - tv0) / 1e9);
    printf("correctness: worst |diff| %.3e at (row %d, dim %d), peak %.3f\n", worst, wr, wd, peak);
    const double tri = 10240.0 * 10241.0 * 0.5 * HD * 2.0 * 42.0 * 16.0;
    printf("pp10240 attention at this rate: %.2f s (%.2f TMAC)\n",
           tri / (macs / (t_all1 - t_all0)), tri / 1e12);
    return 0;
}
