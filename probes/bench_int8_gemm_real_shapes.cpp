// Measure the int8 4x64 dot GEMM on the real MiniCPM5-2B matrix shapes, 8 cores,
// to get the one number that decides whether prefill 200 tok/s is reachable.
//
// Arithmetic density decides int8 over int4 here, which is counter-intuitive:
// the 413 GMAC/s int4 figure measured earlier is raw i4 x i4 on the same
// operand, but real int8-activation x int4-weight needs vupack + vsll + two
// macOS per 1024 products, i.e. 341 products per instruction. int8 x int8 does
// 512 products per instruction with a single vmadot. Since prefill at 10K is
// compute bound (weights stream in 44 ms, GEMM takes tens of seconds), density
// wins over compactness.
//
// Shapes: q/k/v/o are 2048x2048, gate/up are 6144x2048, down is 2048x6144.
// Per token that is 2.03 GMAC, so 10240 tokens is 20.8 TMAC.
//
// One core per process, 8 processes, aggregate reported the same way the DRAM
// measurement was done.
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


static uint32_t g_gather[32];

// 4 rows x 64 cols, K = ksteps*16. A[row*16+d], B[col*16+d], lane = row*8+col.
// B is pre-packed so 64 columns sit contiguously in 1024 bytes.
static void gemm4x64(const int8_t * A, const int8_t * B, float * C, int64_t ksteps, float asc) {
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
        "KSLOOP%=:                      \n\t"
        "vl1re32.v v0,  (%[A])           \n\t"
        "vl8re32.v v8,  (%[B])           \n\t"
        "addi    %[A], %[A], 64         \n\t"
        "addi    %[B], %[B], 1024       \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v18, v0, v9            \n\t"
        "vmadot  v20, v0, v10           \n\t"
        "vmadot  v22, v0, v11           \n\t"
        "vmadot  v24, v0, v12           \n\t"
        "vmadot  v26, v0, v13           \n\t"
        "vmadot  v28, v0, v14           \n\t"
        "vmadot  v30, v0, v15           \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, KSLOOP%=           \n\t"
        // epilogue: scale, convert, store 64 columns
        "vfcvt.f.x.v v16, v16           \n\t"
        "vfcvt.f.x.v v18, v18           \n\t"
        "vfcvt.f.x.v v20, v20           \n\t"
        "vfcvt.f.x.v v22, v22           \n\t"
        "vfcvt.f.x.v v24, v24           \n\t"
        "vfcvt.f.x.v v26, v26           \n\t"
        "vfcvt.f.x.v v28, v28           \n\t"
        "vfcvt.f.x.v v30, v30           \n\t"
        // lane = row*8+col: gather to [col][row] so each column's 4 rows are
        // contiguous, then a single 32-float store per accumulator
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


int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int cpu = argc > 1 ? atoi(argv[1]) : 8;
    const int64_t M = argc > 2 ? atoll(argv[2]) : 2048;   // rows (tokens)
    const int64_t N = argc > 3 ? atoll(argv[3]) : 2048;   // output cols
    const int64_t K = argc > 4 ? atoll(argv[4]) : 2048;   // reduction

    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("cpu %d bind failed\n", cpu); return 1; }

    for (int p = 0; p < 32; ++p) g_gather[p] = (uint32_t) ((p % 4) * 8 + p / 4);

    // weight tile: N columns x K, packed as [nblk][kstep][8 cols x 16] int8
    const int64_t nblk = N / 64, kst = K / 16;
    int8_t * B = (int8_t *) aligned_alloc(2 << 20, (size_t) nblk * kst * 1024);
    for (int64_t b = 0; b < nblk; ++b)
        for (int64_t s = 0; s < kst; ++s)
            for (int c = 0; c < 64; ++c)
                for (int d = 0; d < 16; ++d)
                    B[(b * kst + s) * 1024 + c * 16 + d] =
                        (int8_t) (((b * 7 + s * 3 + c * 5 + d * 11) % 61) - 30);

    // activation rows: 4 rows x K, packed as [kstep][4 rows x 16] int8
    const int64_t mblk = M / 4;
    int8_t * A = (int8_t *) aligned_alloc(2 << 20, (size_t) kst * 64);
    for (int64_t s = 0; s < kst; ++s)
        for (int r = 0; r < 4; ++r)
            for (int d = 0; d < 16; ++d)
                A[s * 64 + r * 16 + d] = (int8_t) (((s * 5 + r * 3 + d * 7) % 61) - 30);

    float * C = (float *) aligned_alloc(64, 4 * 64 * 4);

    // time a full M x N x K product over all row groups and column blocks
    const int64_t macs_total = M * N * K;
    const int reps = 1;
    fprintf(stderr, "warmup...\n");
    gemm4x64(A, B, C, kst, 1.0f);
    fprintf(stderr, "warmup done, C[0]=%f\n", (double) C[0]);
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r)
        for (int64_t nb = 0; nb < nblk; ++nb)
            for (int64_t mb = 0; mb < mblk; ++mb)
                gemm4x64(A, B + nb * kst * 1024, C, kst, 1.0f);
    const double dt = (now_s() - t0) / reps;

    printf("cpu %d  M=%lld N=%lld K=%lld : %.3f s -> %.1f GMAC/s/core\n",
           cpu, (long long) M, (long long) N, (long long) K, dt,
           (double) macs_total / dt / 1e9);

    // spot check one output element against a scalar product
    {
        const int64_t nb = 0;
        static float first[512];
        gemm4x64(A, B + nb * kst * 1024, first, kst, 1.0f);
        // C layout after gather: 64 cols x 4 rows
        long long ref = 0;
        for (int64_t s = 0; s < kst; ++s)
            for (int d = 0; d < 16; ++d)
                ref += (long long) A[s * 64 + 0 * 16 + d] * (long long) B[(nb * kst + s) * 1024 + 0 * 16 + d];
        printf("  check r0/c0: got %.0f ref %lld -> %s\n",
               (double) first[0], ref, ((long long) first[0] == ref) ? "OK" : "MISMATCH");
    }
    return 0;
}
