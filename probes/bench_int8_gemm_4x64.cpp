// The int8 dot GEMM loop is load-issue bound, not dot bound.
//
// 4 rows x 32 cols: 93-95 GMAC/s. Doubling rows changed nothing, so amortising
// the 512-byte B load over more rows is not the lever. Per k-step the loop pays
// three vector loads (~21 cycles each, measured earlier) = ~62 cycles for 4096
// MAC, i.e. 66 MAC/cycle. Each vmadot itself is only ~1 cycle.
//
// The lever is loading B once as 1024 bytes (vl8r, 64 columns) so a k-step costs
// two loads (128 B of A + 1024 B of B) instead of three, while still issuing 8
// dots. That should be ~156 GMAC/s/core, and 8 cores then cover the 9.01 TMAC of
// 10K attention in well under the 11.7 s left by the 32 GB/s DRAM stream.
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

// 4 rows x 64 cols, K = ksteps*16
static void g64(const int8_t * A, const int8_t * B, float * C, int64_t ksteps) {
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
        "vl1r.v  v0,  (%[A])            \n\t"   // 4 rows x 16 in bytes 0-63
        "vl8r.v  v8,  (%[B])            \n\t"   // 64 cols x 16 = 1024 B
        "addi    %[A], %[A], 64         \n\t"
        "addi    %[B], %[B], 1024       \n\t"
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
        : [A] "+r"(A), [B] "+r"(B)
        : [KS] "r"(ksteps),
          [C0] "r"(C + 0),   [C1] "r"(C + 32),  [C2] "r"(C + 64),  [C3] "r"(C + 96),
          [C4] "r"(C + 128), [C5] "r"(C + 160), [C6] "r"(C + 192), [C7] "r"(C + 224)
        : "t0", "t3", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
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
    static int8_t a[KS * 64];
    static int8_t b[KS * 1024];
    static float c[4 * 64];
    for (int64_t i = 0; i < KS * 64; ++i) a[i] = (int8_t) ((i * 37) % 61 - 30);
    for (int64_t i = 0; i < KS * 1024; ++i) b[i] = (int8_t) ((i * 53) % 61 - 30);

    const int64_t macs = 4 * 64 * K;
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
        memset(c, 0, sizeof c);
        const double t0 = now_s();
        for (int i = 0; i < 20; ++i) g64(a, b, c, KS);
        const double dt = (now_s() - t0) / 20;
        if (dt < best) best = dt;
    }
    printf("int8 dot GEMM 4x64x%lld: %.1f GMAC/s/core (%.0f MAC/cycle@1.6G)\n",
           (long long) K, (double) macs / best / 1e9, (double) macs / best / 1.6e9);
    printf("  8 cores -> %.0f GMAC/s; 9.01 TMAC attention in %.2f s (budget 11.7 s)\n",
           8 * (double) macs / best / 1e9, 9.01e12 / (8 * (double) macs / best));

    int bad = 0; double worst = 0;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 64; ++col) {
            long long ref = 0;
            for (int ks = 0; ks < KS; ++ks)
                for (int d = 0; d < 16; ++d)
                    ref += (long long) a[ks * 64 + row * 16 + d] * (long long) b[ks * 1024 + col * 16 + d];
            const int group = col / 8, lc = col % 8;
            const float got = c[group * 32 + row * 8 + lc];
            const double dd = fabs((double) got - (double) ref);
            if (dd > 0) ++bad;
            if (dd > worst) worst = dd;
        }
    printf("correctness: %d / 256 wrong, worst |diff| %.0f\n", bad, worst);
    return 0;
}
