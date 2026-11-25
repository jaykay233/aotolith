// Can plain RVV f32 work run while the IME2 GEMM engine is busy?
//
// probe_coinissue.cpp established that IME2 and vmadot share one issue port, so
// GEMM cannot overlap the attention dots. Elementwise is a different bet: vle32,
// vfadd, vfmul and vse32 are ordinary vector instructions, not dot products, so
// they may retire on a different unit than IME2's.
//
// That matters because prefill alternates them: norm 0.9 s, quant 2.6 s, silu
// 1.5 s, conv 1.1 s, rope 0.8 s, add 2.0 s against GEMM 30.3 s. If they co-issue,
// the chunk can be split by rows and elementwise for rows i+1 runs underneath the
// GEMM for rows i, which would take several seconds off the critical path. If they
// serialize, splitting rows buys nothing and this is closed.
//
//   ime2 : Z calls of gemm_kernel_i8i4_hp_m4
//   rvv  : Z passes of a 4-row rmsnorm-shaped stream over a 1 MB buffer
//   both : Z of each, interleaved one for one
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
#include <riscv_vector.h>

#include "ggml_stubs.h"

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

// The gemm entry point without ggml_stubs: declare just what is called.
namespace spacemit_kernels { namespace ime2 {
size_t gemm_kernel_i8i4_hp_m4(size_t blk_len, const uint8_t * a, const uint8_t * b,
                              const uint8_t * bzp, float * c, size_t count_m,
                              size_t count_n, size_t k_blks, size_t ldc);
}}

static int K = 2048, N = 1024, NT = 1, REPS = 5;
static long g_calls;
static uint8_t * gB, * gA4;
static float * gC;
static float * gbuf;          // elementwise working set
static int64_t gbuf_n = 1 << 18;

struct Job { int tid; double a, b, c; };

static double run_ime2(long calls) {
    const int64_t kb = K / 32;
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) {
        const int64_t a = (c * 37) % (N / 32);
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
            256, gA4, gB + a * kb * 576, nullptr, gC, 4, 32, (size_t) (kb / 8), 2048);
    }
    return now_s() - t0;
}

// One pass of the shape the elementwise stages have: read two streams, add, write,
// then reduce the row. Uses the plain m2 forms mega3.cpp uses.
static void rvv_pass(void) {
    for (int64_t i = 0; i < gbuf_n; i += 1024) {
        const size_t vl = __riscv_vsetvl_e32m2(1024);
        const vfloat32m2_t a = __riscv_vle32_v_f32m2(gbuf + i, vl);
        const vfloat32m2_t b = __riscv_vle32_v_f32m2(gbuf + i + (gbuf_n >> 1), vl);
        __riscv_vse32_v_f32m2(gbuf + i, __riscv_vfadd_vv_f32m2(a, b, vl), vl);
    }
}

static double run_rvv(long calls) {
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) rvv_pass();
    return now_s() - t0;
}

static double run_both(long calls) {
    const int64_t kb = K / 32;
    const double t0 = now_s();
    for (long c = 0; c < calls; ++c) {
        const int64_t a = (c * 37) % (N / 32);
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
            256, gA4, gB + a * kb * 576, nullptr, gC, 4, 32, (size_t) (kb / 8), 2048);
        rvv_pass();
    }
    return now_s() - t0;
}

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    double ba = 1e9, bb = 1e9, bc = 1e9;
    for (int r = 0; r < REPS; ++r) {
        double d = run_ime2(g_calls); if (d < ba) ba = d;
        d = run_rvv(g_calls);         if (d < bb) bb = d;
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
        else if (!strcmp(argv[i], "-e")) gbuf_n = atol(argv[++i]);
    }
    if (!g_calls) g_calls = 4000;
    open_a100();
    const int64_t kb = K / 32;
    gB  = (uint8_t *) aligned_alloc(4096, (size_t) (N / 32) * kb * 576 + 4096);
    gA4 = (uint8_t *) aligned_alloc(4096, (size_t) (K / 256) * 1160 + 4096);
    gC  = (float *)   aligned_alloc(4096, (size_t) NT * 8192 * 4 + 4096);
    gbuf = (float *)  aligned_alloc(4096, (size_t) gbuf_n * 4 + 4096);
    if (!gB || !gA4 || !gC || !gbuf) { printf("oom\n"); return 1; }
    memset(gB, 1, (size_t) (N / 32) * kb * 576);
    memset(gA4, 1, (size_t) (K / 256) * 1160);
    memset(gC, 0, (size_t) NT * 8192 * 4);
    for (int64_t i = 0; i < gbuf_n; ++i) gbuf[i] = 1.0f;

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
    // Elementwise bytes: 2 reads + 1 write per element.
    const double bytes = (double) g_calls * (double) gbuf_n * 12.0;
    const double mac   = (double) g_calls * 4.0 * 32.0 * K;
    printf("K=%-5d N=%-5d thr=%d calls=%ld rvv=%.0f KB\n", K, N, NT, g_calls, gbuf_n * 4.0 / 1024);
    printf("  ime2    %7.4f s  %6.1f GMAC/s\n", a, mac / a / 1e9);
    printf("  rvv     %7.4f s  %6.1f GB/s\n", b, bytes / b / 1e9);
    printf("  sum     %7.4f s   (if they serialize)\n", a + b);
    printf("  max     %7.4f s   (if they co-issue)\n", a > b ? a : b);
    printf("  both    %7.4f s   overlap %.0f%% of the ideal gain\n",
           c, 100.0 * ((a + b) - c) / ((a + b) - (a > b ? a : b)));
    return 0;
}
