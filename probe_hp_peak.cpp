// What is the IME2 kernel's own ceiling, with the stream taken out of it?
//
// Prefill runs gemm_kernel_i8i4_hp_m4 at ~47.7 GMAC/s per A core at N=6144,
// which is the number that decides whether 200 tok/s is reachable: the
// combined A+X100 rate is 592 GMAC/s and the GEMM needs 990. So the question
// this probe answers is whether 47.7 is the engine or the environment.
//
// Four configurations, one process, same A record (built by the same vendor
// quantiser prefill uses):
//   * N=32, one column block       -- B is 36 KB, fits L1/L2: pure issue rate
//   * N=1024                       -- B is 1.2 MB: L2, no DRAM
//   * N=6144                       -- B is 7 MB: production shape, DRAM
//   * row-group reuse: G groups through one B column block, i.e. the prefill
//     loop order, at the production N
//
// GMAC/s is computed from 4 rows * N * K, so a run that under-uses the engine
// shows up as a low number rather than as a different unit.
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
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b,
                        const uint8_t * bzp, float * c, size_t count_m,
                        size_t count_n, size_t k_blks, size_t ldc);
}}
namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k,
                           uint8_t * quant_a_ptr);
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k,
                        uint8_t * quant_a_ptr);
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


// The vmadot 4x64 kernel prefill uses on the int8 path, copied verbatim so the
// two can be compared under identical B residency.
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
static int32_t g_vmc[8][256];
static int8_t  g_vmA[8 * 4096];

static int K = 2048, N = 6144, NT = 1, G = 1, MODE = 0;
static const int M = 4;
static uint8_t * gB, * gA4, * gA4_32;
static float * gA, * gC;

struct Job { int tid; double dt; };

// One thread: every 32-column block in its slice, then G row groups through it.
static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    const int64_t nblkm = (MODE == 2) ? (N / 64) : (N / 32);
    const int64_t kst_1024 = (int64_t) (K / 16) * 1024;
    const int64_t nb = nblkm, kb = K / 32;
    const int64_t per = (nb + NT - 1) / NT;
    const int64_t n0 = per * j->tid, n1 = (j->tid == NT - 1) ? nb : per * (j->tid + 1);
    float * c = gC + (int64_t) j->tid * 4096;
    double best = 1e9;
    if (MODE == 0) {
        for (int rep = 0; rep < 6; ++rep) {
            const double t0 = now_s();
            for (int64_t a = n0; a < n1; ++a) {
                const uint8_t * bcol = gB + a * kb * 576;
                for (int g = 0; g < G; ++g)
                    spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
                        256, gA4 + (size_t) g * ((size_t) (K / 256) * 1160), bcol, nullptr,
                        c + (size_t) g * (2 << 12), 4, 32, (size_t) (kb / 8), (size_t) 2048);
            }
            const double d = now_s() - t0;
            if (rep >= 2 && d < best) best = d;
        }
    } else if (MODE == 2) {
        // vmadot 4x64: B is (N/64) blocks of kst*1024 bytes, A is 64 B per 16 k
        for (int rep = 0; rep < 6; ++rep) {
            const double t0 = now_s();
            for (int64_t a = n0; a < n1; ++a) {
                const int8_t * bcol = (const int8_t *) gB + a * kst_1024;
                for (int g = 0; g < G; ++g)
                    vm_4x64(g_vmA + (size_t) g * (K / 16) * 64, bcol,
                            g_vmc[j->tid], K / 16);
            }
            const double d = now_s() - t0;
            if (rep >= 2 && d < best) best = d;
        }
    } else {
        // K32 legacy kernel, for the per-k-block comparison
        for (int rep = 0; rep < 6; ++rep) {
            const double t0 = now_s();
            for (int64_t a = n0; a < n1; ++a) {
                const uint8_t * bcol = gB + a * kb * 576;
                for (int g = 0; g < G; ++g)
                    spacemit_kernels::ime2::gemm_kernel_i8i4(
                        32, gA4_32 + (size_t) g * ((size_t) kb * 152), bcol, nullptr,
                        c + (size_t) g * (2 << 12), 4, 32, (size_t) kb, (size_t) 2048);
            }
            const double d = now_s() - t0;
            if (rep >= 2 && d < best) best = d;
        }
    }
    j->dt = best;
    return nullptr;
}

static double run(void) {
    static Job j[8]; static pthread_t t[8];
    for (int i = 0; i < NT; ++i) { j[i] = Job{i, 0}; }
    for (int i = 1; i < NT; ++i) pthread_create(&t[i], nullptr, th, &j[i]);
    th(&j[0]);
    for (int i = 1; i < NT; ++i) pthread_join(t[i], nullptr);
    double worst = 0;
    for (int i = 0; i < NT; ++i) if (j[i].dt > worst) worst = j[i].dt;
    return worst;
}

int main(int argc, char ** argv) {
    int ks[16], ns[16], gs[16], ms[16], nt[16], ncase = 0;
    // k, n, g, mode, threads
    const int def[][5] = {
        { 2048,   32, 1, 0, 1 },
        { 2048, 1024, 1, 0, 1 },
        { 2048, 6144, 1, 0, 1 },
        { 2048, 6144, 4, 0, 1 },
        { 2048, 6144, 1, 0, 8 },
        { 2048, 6144, 4, 0, 8 },
        { 6144, 6144, 1, 0, 8 },   // down's shape: K long, X100's worst case
        { 2048, 6144, 1, 1, 8 },   // K32 legacy for reference
        { 2048,   32, 1, 1, 1 },
        { 2048,   64, 1, 2, 1 },   // vmadot, B=128 KB hot: engine ceiling
        { 2048,   64, 8, 2, 1 },   // vmadot, B reused over 8 row groups
        { 2048, 6144, 1, 2, 1 },   // vmadot, production shape
        { 2048, 6144, 8, 2, 1 },   // vmadot, production with 8-group reuse
        { 2048, 6144, 1, 2, 8 },   // vmadot, 8 threads
        { 6144, 6144, 1, 2, 8 },   // vmadot, down's shape, 8 threads
    };
    ncase = (int) (sizeof(def) / sizeof(def[0]));
    for (int i = 0; i < ncase; ++i) {
        ks[i] = def[i][0]; ns[i] = def[i][1]; gs[i] = def[i][2];
        ms[i] = def[i][3]; nt[i] = def[i][4];
    }

    // one allocation big enough for the largest case
    const int KMAX = 8192, NMAX = 8192, GMAX = 8;
    gB     = (uint8_t *) aligned_alloc(64, (size_t) NMAX * KMAX);
    gA     = (float *) aligned_alloc(64, (size_t) GMAX * M * KMAX * 4);
    gA4    = (uint8_t *) aligned_alloc(64, (size_t) GMAX * (KMAX / 256) * 1160 + 64);
    gA4_32 = (uint8_t *) aligned_alloc(64, (size_t) GMAX * (KMAX / 32) * 152 + 64);
    gC     = (float *) aligned_alloc(64, (size_t) 8 * 8192 * 4);
    if (!gB || !gA || !gA4 || !gA4_32 || !gC) { printf("oom\n"); return 1; }
    srand(11);
    for (int64_t i = 0; i < (int64_t) NMAX * KMAX; ++i) gB[i] = (uint8_t) (rand() & 0xFF);
    for (int64_t i = 0; i < (int64_t) GMAX * M * KMAX; ++i)
        gA[i] = (float) ((rand() % 200) - 100) * 0.01f;
    open_a100();

    printf("%-6s %-7s %-3s %-6s %-4s %9s %9s\n",
           "K", "N", "G", "kernel", "thr", "ms", "GMAC/s");
    for (int i = 0; i < ncase; ++i) {
        K = ks[i]; N = ns[i]; G = gs[i]; MODE = ms[i]; NT = nt[i];
        memset(g_vmA, 3, sizeof(g_vmA));
        for (int g = 0; g < G; ++g) {
            const float * a = gA + (int64_t) g * M * K;
            if (MODE == 2) break;
            if (MODE == 0) {
                memset(gA4 + (size_t) g * ((size_t) (K / 256) * 1160), 0, (size_t) (K / 256) * 1160);
                spacemit_kernels::rvv::quantize_a_4row_i8_hp(
                    256, a, (size_t) K, gA4 + (size_t) g * ((size_t) (K / 256) * 1160));
            } else {
                memset(gA4_32 + (size_t) g * ((size_t) (K / 32) * 152), 0, (size_t) (K / 32) * 152);
                spacemit_kernels::rvv::quantize_a_4row_i8(
                    32, a, (size_t) K, gA4_32 + (size_t) g * ((size_t) (K / 32) * 152));
            }
        }
        const double dt = run();
        const double mac = (double) M * N * K * G;
        printf("%-6d %-7d %-3d %-6s %-4d %9.3f %9.1f\n",
               K, N, G, MODE == 2 ? "vmadot" : (MODE ? "K32" : "K256hp"), NT,
               dt * 1e3, mac / dt / 1e9);
    }
    return 0;
}
