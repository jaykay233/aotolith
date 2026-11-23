// Two numbers decide whether 200 tok/s prefill is reachable on this chip.
//
// 1. DRAM bandwidth actually achievable. Q4_0 weights are 1.265 GB and prefill
//    must stream them once per token batch, so 51.2 s (200 tok/s at 10K) needs
//    24.7 GB/s sustained. Earlier runs hinted at a ~26 GB/s ceiling; if the real
//    ceiling is higher there is headroom, if lower the target is impossible
//    without fewer weight bytes.
//
// 2. Whether int8 `vmadot` really does 512 MAC per instruction. The operand map
//    was verified bit-exactly (A[row*16+d], B[col*16+d], lane = row*8+col), and
//    a single dot then costs 0.628 ns, which at 512 MAC would be 815 GMAC/s on
//    one core. A correctness-checked GEMM settles it.
#define _GNU_SOURCE
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

static int g_cpu = 8;
static void bind1(int cpu) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

// ---------------- 1. DRAM bandwidth
static void bench_bw(int nthreads, int64_t mb) {
    const int64_t n = mb << 20;
    uint8_t * src = (uint8_t *) aligned_alloc(2 << 20, (size_t) n);
    uint8_t * dst = (uint8_t *) aligned_alloc(2 << 20, (size_t) n);
    memset(src, 1, (size_t) n);
    memset(dst, 2, (size_t) n);

    // single-thread read+write
    const double t0 = now_s();
    memcpy(dst, src, (size_t) n);
    const double dt = now_s() - t0;
    printf("  memcpy 1 thread, %lld MB: %.1f GB/s (counts read+write)\n",
           (long long) mb, 2.0 * (double) n / dt / 1e9);

    // read-only, vectorised 32-byte loads
    for (int nt = 1; nt <= nthreads; nt *= 2) {
        const double t1 = now_s();
        double acc = 0;
        #pragma omp parallel for reduction(+:acc) num_threads(nt)
        for (int64_t i = 0; i + 32 <= n; i += 32) {
            const vint8m1_t v = __riscv_vle8_v_i8m1((const int8_t *) (src + i), 32);
            const vint32m1_t sq = __riscv_vwredsum_vs_i8m1_i32m1(
                v, __riscv_vmv_v_x_i32m1(0, 32), 32);
            const vint32m1_t acc_v = __riscv_vreinterpret_v_i32m1_i32m1(sq);
            acc += (double) __riscv_vmv_x_s_i32m1_i32(acc_v);
        }
        const double d1 = now_s() - t1;
        printf("  read %lld MB, %d threads: %.1f GB/s  (acc %g)\n",
               (long long) mb, nt, (double) n / d1 / 1e9, acc);
    }
    free(src); free(dst);
}

// ---------------- 2. int8 dot GEMM, 4 rows x 32 cols
// A packed: per k step of 16, 4 rows x 16 int8 at row*16 (64 B)
// B packed: per k step, 32 cols x 16 int8 at col*16 (512 B = 4 register groups)
static void g4(const int8_t * A, const int8_t * B, float * C, int64_t ldc, int64_t ksteps,
               const float * as, const float * bs) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "mv     t3, %[KS]               \n\t"
        "KS_LOOP%=:                     \n\t"
        "vsetvli t0, x0, e8, m1         \n\t"
        "vl1r.v  v0,  (%[A])            \n\t"   // 4 rows x 16
        "vl4r.v  v4,  (%[B])            \n\t"   // 32 cols x 16
        "addi    %[A], %[A], 64         \n\t"
        "addi    %[B], %[B], 512        \n\t"
        "vsetvli t0, x0, e32, m1        \n\t"
        "vmadot  v16, v0, v4            \n\t"
        "vmadot  v18, v0, v5            \n\t"
        "vmadot  v20, v0, v6            \n\t"
        "vmadot  v22, v0, v7            \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, KS_LOOP%=          \n\t"
        // epilogue: lane = row*8 + col, col = group*8 + lane%8
        "vfcvt.f.x.v v16, v16           \n\t"
        "vfcvt.f.x.v v18, v18           \n\t"
        "vfcvt.f.x.v v20, v20           \n\t"
        "vfcvt.f.x.v v22, v22           \n\t"
        "vse32.v v16, (%[C0])           \n\t"
        "vse32.v v18, (%[C1])           \n\t"
        "vse32.v v20, (%[C2])           \n\t"
        "vse32.v v22, (%[C3])           \n\t"
        : [A] "+r"(A), [B] "+r"(B)
        : [KS] "r"(ksteps), [C0] "r"(C), [C1] "r"(C + 32), [C2] "r"(C + 64), [C3] "r"(C + 96)
        : "t0", "t3", "v0", "v4", "v5", "v6", "v7", "v16", "v18", "v20", "v22", "memory");
    (void) ldc; (void) as; (void) bs;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int cpu = argc > 1 ? atoi(argv[1]) : 8;
    bind1(cpu);
    size_t vlenb = 0; asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    printf("cpu %d vlenb=%zu\n", cpu, vlenb);

    const int64_t KS = 128;            // 128 steps x 16 = 2048 k
    const int64_t K = KS * 16;

    static int8_t a[128 * 64];
    static int8_t b[128 * 512];
    static float c[4 * 32];
    // plausible values
    for (int64_t i = 0; i < KS * 64; ++i) a[i] = (int8_t) ((i * 37) % 251 - 125);
    for (int64_t i = 0; i < KS * 512; ++i) b[i] = (int8_t) ((i * 53) % 251 - 125);
    memset(c, 0, sizeof c);

    const int64_t macs = 4 * 32 * K;

    // warm + time
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
        memset(c, 0, sizeof c);
        const double t0 = now_s();
        const int reps = 20;
        for (int i = 0; i < reps; ++i) g4(a, b, c, 32, KS, nullptr, nullptr);
        const double dt = (now_s() - t0) / reps;
        if (dt < best) best = dt;
    }
    printf("int8 dot GEMM 4x32x%lld: %.1f GMAC/s  (%.1f MAC/cycle@1.6G)\n",
           (long long) K, (double) macs / best / 1e9, (double) macs / best / 1.6e9);

    // correctness: rebuild the same product with plain scalar C
    {
        // rebuild activations/weights in plain form from the packed layout
        static double ref[4][32];
        memset(ref, 0, sizeof ref);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 32; ++col) {
                long long s = 0;
                for (int64_t ks = 0; ks < KS; ++ks)
                    for (int d = 0; d < 16; ++d) {
                        const int8_t av = a[ks * 64 + row * 16 + d];
                        const int8_t bv = b[ks * 512 + col * 16 + d];
                        s += (long long) av * bv;
                    }
                ref[row][col] = (double) s;
            }
        // kernel output: c[group*8 + lane], lane = row*8 + col%8
        double worst = 0; int wr = -1, wc = -1;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 32; ++col) {
                const int group = col / 8, lc = col % 8;
                const float got = c[group * 32 + row * 8 + lc];
                const double d = fabs((double) got - ref[row][col]);
                if (d > worst) { worst = d; wr = row; wc = col; }
            }
        printf("correctness: worst |diff| = %.0f at (r=%d,c=%d), ref[0][0]=%.0f got=%.0f\n",
               worst, wr, wc, ref[0][0],
               (double) c[0 * 8 + 0 * 8 + 0]);
        printf("  -> %s\n", worst == 0 ? "BIT EXACT" : "MISMATCH");
    }

    if (argc > 2 && argv[2][0] == 'b') bench_bw(8, 512);
    return 0;
}
