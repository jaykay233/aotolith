// Do GEMM and attention share the A cluster's dot units, or only its cores?
//
// This decides the largest remaining prefill lever. The layer chain is strictly
// serial (qkv GEMM -> QK -> softmax -> PV -> o GEMM -> gate/up -> down), so the
// only way to overlap GEMM with attention is across rows: split the chunk and run
// attention on one half while GEMM runs on the other. That pays only if the IME2
// path (Q4_0 GEMM) and the vmadot path (QK/PV) use different execution ports.
//
// The two throughputs are close enough to be suspicious in either direction --
// IME2 measures 47.9 GMAC/s per core, vmadot's QK loop 74.6 -- and the code
// comment in mega3.cpp says they are "the same dot units". Three phases, same
// thread, same buffers:
//
//   ime2   : Z calls of gemm_kernel_i8i4_hp_m4
//   vmadot : Z calls of vm_4x64
//   both   : Z of each, interleaved one for one
//
// If `both` costs the sum, they serialize. If it costs the max, they co-issue and
// row-pipelining GEMM against attention is worth building.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#include "ggml_stubs.h"

namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i4_hp_m4(size_t blk_len, const uint8_t * a, const uint8_t * b,
                              const uint8_t * bzp, float * c, size_t count_m,
                              size_t count_n, size_t k_blks, size_t ldc);
}}

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}
static void open_a100(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
}

// 4 rows x 64 positions x 16 dims per step: the QK/PV kernel, verbatim from
// probe_qk_real.cpp.
static void vm_4x64(const int8_t * A, const int8_t * B, int32_t * C, int64_t ksteps) {
    asm volatile(
        "vsetvli t0, x0, e32, m1   \n\t"
        "vxor.vv v16, v16, v16     \n\t"
        "vxor.vv v18, v18, v18     \n\t"
        "vxor.vv v20, v20, v20     \n\t"
        "vxor.vv v22, v22, v22     \n\t"
        "vxor.vv v24, v24, v24     \n\t"
        "vxor.vv v26, v26, v26     \n\t"
        "vxor.vv v28, v28, v28     \n\t"
        "vxor.vv v30, v30, v30     \n\t"
        "mv     t3, %[KS]          \n\t"
        "I8L%=:                    \n\t"
        "vl1re32.v v0, (%[A])       \n\t"
        "vl8re32.v v8, (%[B])       \n\t"
        "addi   %[A], %[A], 64      \n\t"
        "addi   %[B], %[B], 1024    \n\t"
        "vmadot v16, v0, v8         \n\t"
        "vmadot v18, v0, v9         \n\t"
        "vmadot v20, v0, v10        \n\t"
        "vmadot v22, v0, v11        \n\t"
        "vmadot v24, v0, v12        \n\t"
        "vmadot v26, v0, v13        \n\t"
        "vmadot v28, v0, v14        \n\t"
        "vmadot v30, v0, v15        \n\t"
        "addi   t3, t3, -1          \n\t"
        "bgtz   t3, I8L%=           \n\t"
        "vse32.v v16, (%[C0])       \n\t"
        "vse32.v v18, (%[C1])       \n\t"
        "vse32.v v20, (%[C2])       \n\t"
        "vse32.v v22, (%[C3])       \n\t"
        "vse32.v v24, (%[C4])       \n\t"
        "vse32.v v26, (%[C5])       \n\t"
        "vse32.v v28, (%[C6])       \n\t"
        "vse32.v v30, (%[C7])       \n\t"
        : [A] "+r"(A), [B] "+r"(B)
        : [KS] "r"(ksteps),
          [C0] "r"(C), [C1] "r"(C + 32), [C2] "r"(C + 64), [C3] "r"(C + 96),
          [C4] "r"(C + 128), [C5] "r"(C + 160), [C6] "r"(C + 192), [C7] "r"(C + 224)
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

static int K = 2048, N = 1024, NT = 1, REPS = 5;
static uint8_t * gB, * gA4;
static float * gC;
static int8_t * gVA, * gVB;
static int32_t * gVC;

// IME2 side: one call is 4 rows x 32 columns x K, reading one 576-byte column
// block per K32 step and 1160 bytes of A per K256 superblock.
static double run_ime2(long calls) {
    const int64_t kb = K / 32;
    const size_t acol = (size_t) (K / 256) * 1160;
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) {
        const int64_t a = (c * 37) % (N / 32);
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
            256, gA4, gB + a * kb * 576, nullptr, gC, 4, 32, (size_t) (kb / 8), 2048);
    }
    (void) acol;
    return now_s() - t0;
}

// vmadot side: one call is 4 rows x 64 positions x K.
static double run_vm(long calls) {
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) vm_4x64(gVA, gVB, gVC, K / 16);
    return now_s() - t0;
}

static double run_both(long calls) {
    const int64_t kb = K / 32;
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) {
        const int64_t a = (c * 37) % (N / 32);
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
            256, gA4, gB + a * kb * 576, nullptr, gC, 4, 32, (size_t) (kb / 8), 2048);
        vm_4x64(gVA, gVB, gVC, K / 16);
    }
    return now_s() - t0;
}

struct Job { int tid; double a, b, c; };
static long g_calls;

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    double ba = 1e9, bb = 1e9, bc = 1e9;
    for (int r = 0; r < REPS; ++r) {
        double d = run_ime2(g_calls); if (d < ba) ba = d;
        d = run_vm(g_calls);          if (d < bb) bb = d;
        d = run_both(g_calls);        if (d < bc) bc = d;
    }
    j->a = ba; j->b = bb; j->c = bc;
    return nullptr;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")) REPS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")) g_calls = atol(argv[++i]);
    }
    if (!g_calls) g_calls = 4000;
    open_a100();
    const int64_t kb = K / 32;
    gB  = (uint8_t *) aligned_alloc(4096, (size_t) (N / 32) * kb * 576 + 4096);
    gA4 = (uint8_t *) aligned_alloc(4096, (size_t) (K / 256) * 1160 + 4096);
    gC  = (float *)   aligned_alloc(4096, (size_t) NT * 8192 * 4 + 4096);
    gVA = (int8_t *)  aligned_alloc(4096, (size_t) (K / 16) * 64 + 4096);
    gVB = (int8_t *)  aligned_alloc(4096, (size_t) (K / 16) * 1024 + 4096);
    gVC = (int32_t *) aligned_alloc(4096, 1024);
    if (!gB || !gA4 || !gC || !gVA || !gVB || !gVC) { printf("oom\n"); return 1; }
    memset(gB, 1, (size_t) (N / 32) * kb * 576);
    memset(gA4, 1, (size_t) (K / 256) * 1160);
    memset(gC, 0, (size_t) NT * 8192 * 4);
    memset(gVA, 2, (size_t) (K / 16) * 64);
    memset(gVB, 3, (size_t) (K / 16) * 1024);
    memset(gVC, 0, 1024);

    pthread_t t[8];
    Job jobs[8] = {};
    for (long i = 0; i < NT; ++i) { jobs[i].tid = (int) i; pthread_create(&t[i], nullptr, th, &jobs[i]); }
    for (int i = 0; i < NT; ++i) pthread_join(t[i], nullptr);

    double a = 0, b = 0, c = 0;
    for (int i = 0; i < NT; ++i) {
        if (jobs[i].a > a) a = jobs[i].a;
        if (jobs[i].b > b) b = jobs[i].b;
        if (jobs[i].c > c) c = jobs[i].c;
    }
    // MACs as each kernel actually computes them
    const double mac_ime2 = (double) g_calls * 4.0 * 32.0 * K;
    const double mac_vm   = (double) g_calls * 4.0 * 64.0 * K;
    const double tot      = mac_ime2 + mac_vm;
    printf("K=%-5d N=%-5d thr=%d calls=%ld\n", K, N, NT, g_calls);
    printf("  ime2    %7.3f s  %6.1f GMAC/s\n", a, mac_ime2 / a / 1e9);
    printf("  vmadot  %7.3f s  %6.1f GMAC/s\n", b, mac_vm / b / 1e9);
    printf("  sum     %7.3f s   (if they serialize)\n", a + b);
    printf("  max     %7.3f s   (if they co-issue)\n", a > b ? a : b);
    printf("  both    %7.3f s  %6.1f GMAC/s   overlap %.0f%% of the ideal gain\n",
           c, tot / c / 1e9, 100.0 * ((a + b) - c) / ((a + b) - (a > b ? a : b)));
    return 0;
}
