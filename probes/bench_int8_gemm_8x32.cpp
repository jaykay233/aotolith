// How fast can the int8 dot unit run a GEMM once the weight tile is amortised
// over more rows?
//
// The 4-row form reaches 94.8 GMAC/s/core, and its loop is dominated by the
// 512-byte B load being re-paid for every 4 rows. Doubling to 8 rows halves that
// per-row cost. This matters because the DRAM measurement (32 GB/s aggregate)
// leaves only ~11.7 s of the 51.2 s budget for attention, which needs 9.01 TMAC
// -- i.e. 770 GMAC/s on 8 cores. The 4-row form gives 758. The 8-row form is the
// question.
//
// Layout, from the verified map: lane = row*8 + col, A[row*16 + d],
// B[col*16 + d]. Each vmadot sees 4 rows (A's first 64 bytes) x 8 cols (B's
// 128 bytes), 16 k deep. So 8 rows needs two A registers: load the 128-byte
// block at +0 (rows 0-3 in bytes 0-63) and again at +64 (rows 4-7 in its 0-63).
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

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// 8 rows x 32 cols, K = ksteps*16
static void g8(const int8_t * A, const int8_t * B, float * C, int64_t ksteps) {
    const int8_t * A64 = A + 64;
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
        "mv     t3, %[KS]               \n\t"
        "KS_LOOP%=:                     \n\t"
        "vsetvli t0, x0, e8, m1         \n\t"
        "vl1r.v  v0,  (%[A])            \n\t"   // rows 0-3 in bytes 0-63
        "vl1r.v  v1,  (%[A64])          \n\t" // rows 4-7 in bytes 0-63
        "vl4r.v  v4,  (%[B])            \n\t"
        "addi    %[A], %[A], 128        \n\t"
        "addi    %[A64], %[A64], 128    \n\t"
        "addi    %[B], %[B], 512        \n\t"
        "vsetvli t0, x0, e32, m1        \n\t"
        "vmadot  v16, v0, v4            \n\t"
        "vmadot  v18, v0, v5            \n\t"
        "vmadot  v20, v0, v6            \n\t"
        "vmadot  v22, v0, v7            \n\t"
        "vmadot  v24, v1, v4            \n\t"
        "vmadot  v26, v1, v5            \n\t"
        "vmadot  v28, v1, v6            \n\t"
        "vmadot  v30, v1, v7            \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, KS_LOOP%=          \n\t"
        "vfcvt.f.x.v v16, v16           \n\t"
        "vfcvt.f.x.v v18, v18           \n\t"
        "vfcvt.f.x.v v20, v20           \n\t"
        "vfcvt.f.x.v v22, v22           \n\t"
        "vfcvt.f.x.v v24, v24           \n\t"
        "vfcvt.f.x.v v26, v26           \n\t"
        "vfcvt.f.x.v v28, v28           \n\t"
        "vfcvt.f.x.v v30, v30           \n\t"
        "vse32.v v16, (%[C0])           \n\t"
        "vse32.v v18, (%[C1])           \n\t"
        "vse32.v v20, (%[C2])           \n\t"
        "vse32.v v22, (%[C3])           \n\t"
        "vse32.v v24, (%[C4])           \n\t"
        "vse32.v v26, (%[C5])           \n\t"
        "vse32.v v28, (%[C6])           \n\t"
        "vse32.v v30, (%[C7])           \n\t"
        : [A] "+r"(A), [A64] "+r"(A64), [B] "+r"(B)
        : [KS] "r"(ksteps),
          [C0] "r"(C + 0),   [C1] "r"(C + 32),  [C2] "r"(C + 64),  [C3] "r"(C + 96),
          [C4] "r"(C + 128), [C5] "r"(C + 160), [C6] "r"(C + 192), [C7] "r"(C + 224)
        : "t0", "t3", "v0", "v1", "v4", "v5", "v6", "v7",
          "v16", "v18", "v20", "v22", "v24", "v26", "v28", "v30", "memory");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(8, &s);
    if (sched_setaffinity(0, sizeof(s), &s) != 0) { printf("bind failed\n"); return 1; }

    enum { KS = 128 };
    const int64_t K = KS * 16;
    static int8_t a[KS * 128];
    static int8_t b[KS * 512];
    static float c[8 * 32];
    for (int64_t i = 0; i < KS * 128; ++i) a[i] = (int8_t) ((i * 37) % 61 - 30);
    for (int64_t i = 0; i < KS * 512; ++i) b[i] = (int8_t) ((i * 53) % 61 - 30);

    const int64_t macs = 8 * 32 * K;
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
        memset(c, 0, sizeof c);
        const double t0 = now_s();
        const int reps = 20;
        for (int i = 0; i < reps; ++i) g8(a, b, c, KS);
        const double dt = (now_s() - t0) / reps;
        if (dt < best) best = dt;
    }
    printf("int8 dot GEMM 8x32x%lld: %.1f GMAC/s/core  (%.0f MAC/cycle@1.6G)\n",
           (long long) K, (double) macs / best / 1e9, (double) macs / best / 1.6e9);
    printf("  8 cores -> %.0f GMAC/s; attention 9.01 TMAC would take %.2f s\n",
           8 * (double) macs / best / 1e9, 9.01e12 / (8 * (double) macs / best));

    // correctness over all 8 rows x 32 cols
    int bad = 0;
    double worst = 0;
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 32; ++col) {
            long long ref = 0;
            for (int ks = 0; ks < KS; ++ks)
                for (int d = 0; d < 16; ++d)
                    ref += (long long) a[ks * 128 + row * 16 + d] * (long long) b[ks * 512 + col * 16 + d];
            const int rg = row / 4, r4 = row % 4, group = col / 8, lc = col % 8;
            const float got = c[rg * 128 + group * 32 + r4 * 8 + lc];
            const double dd = fabs((double) got - (double) ref);
            if (dd > 0) ++bad;
            if (dd > worst) worst = dd;
        }
    printf("correctness: %d / 256 wrong, worst |diff| %.0f\n", bad, worst);
    return 0;
}
