// Is a native count_m=4 IME2 call cheaper than four count_m=1 calls?
//
// Verifying m drafts flips the GEMM from stream-bound to MAC-bound: at m=1 the
// decode GEMM sits at the 25 GB/s DRAM ceiling and only exercises 45 GMAC/s, so
// arithmetic is free, but each extra row adds a flat ~28 ms of MACs. If the engine
// can process four rows per pass natively instead of four times over, that 28 ms
// could collapse.
//
// ime2_kernels.cpp dispatches to gemm_kernel_i8i4_m4 when count_m >= 4. The A tile
// for m rows, per 32-k block, is
//     [f32 scale x m][int16 sum x m][int8 data x m x 32]
// with scale = amax/127 and sum = -sum(q), matching quantize_a_row_i8.
//
// Both paths are compared numerically as well as for speed, so this cannot silently
// report a wrong-but-fast kernel.
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
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const int N_GPU = 8;
static void open_a100(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
}
static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static int K = 2048, N = 65536;
static const int M = 4;
static uint8_t * gB;
static float * gA;
static uint8_t * gA1;   // m1-packed, M rows side by side
static uint8_t * gA4;   // m4-packed
static float * gC1, * gC4;

// scale = amax/127, sum = -sum(q), exactly as quantize_a_row_i8 does it
static void pack_m1(const float * a, int64_t k, uint8_t * dst) {
    for (int64_t b = 0; b < k; b += 32) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(a[b + i]));
        const float sc = amax / 127.0f;
        const float rs = sc ? 1.0f / sc : 0.0f;
        int16_t sum = 0;
        int8_t q[32];
        for (int i = 0; i < 32; ++i) {
            q[i] = (int8_t) lrintf(a[b + i] * rs);
            sum += q[i];
        }
        uint8_t * d = dst + (b / 32) * 38;
        memcpy(d, &sc, 4);
        const int16_t nsum = (int16_t) -sum;
        memcpy(d + 4, &nsum, 2);
        memcpy(d + 6, q, 32);
    }
}

// m4 tile: per k-block [f32 x4][i16 x4][i8 x4 x32] = 152 bytes
static void pack_m4(const float * a, int64_t k, int64_t stride, uint8_t * dst) {
    for (int64_t b = 0; b < k; b += 32) {
        uint8_t * d = dst + (b / 32) * 152;
        float sc[4]; int16_t sum[4]; int8_t q[4][32];
        for (int r = 0; r < M; ++r) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(a[r * stride + b + i]));
            sc[r] = amax / 127.0f;
            const float rs = sc[r] ? 1.0f / sc[r] : 0.0f;
            sum[r] = 0;
            for (int i = 0; i < 32; ++i) {
                q[r][i] = (int8_t) lrintf(a[r * stride + b + i] * rs);
                sum[r] += q[r][i];
            }
        }
        memcpy(d, sc, 16);
        for (int r = 0; r < M; ++r) {
            const int16_t nsum = (int16_t) -sum[r];
            memcpy(d + 16 + 2 * r, &nsum, 2);
        }
        for (int r = 0; r < M; ++r) memcpy(d + 24 + 32 * r, q[r], 32);
    }
}

struct Job { int tid; int mode; double dt; };   // mode 0 = m1 x4, 1 = native m4

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    const int64_t nb = N / 32, kb = K / 32;
    const int64_t per = (nb + N_GPU - 1) / N_GPU;
    const int64_t n0 = per * j->tid, n1 = (j->tid == N_GPU - 1) ? nb : per * (j->tid + 1);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (int64_t a = n0; a < n1; ++a) {
            const uint8_t * bcol = gB + a * kb * 576;
            if (j->mode == 0) {
                for (int r = 0; r < M; ++r)
                    spacemit_kernels::ime2::gemm_kernel_i8i4(
                        32, gA1 + (int64_t) r * kb * 38, bcol, nullptr,
                        gC1 + (int64_t) r * N + a * 32, 1, 32, kb, N);
            } else {
                spacemit_kernels::ime2::gemm_kernel_i8i4(
                    32, gA4, bcol, nullptr, gC4 + a * 32, M, 32, kb, N);
            }
        }
        const double d = now_s() - t0;
        if (rep >= 2 && d < best) best = d;
    }
    j->dt = best;
    return nullptr;
}

static double run_mode(int mode) {
    Job j[N_GPU]; pthread_t t[N_GPU];
    for (int i = 0; i < N_GPU; ++i) j[i] = Job{i, mode, 0};
    for (int i = 1; i < N_GPU; ++i) pthread_create(&t[i], nullptr, th, &j[i]);
    th(&j[0]);
    for (int i = 1; i < N_GPU; ++i) pthread_join(t[i], nullptr);
    double worst = 0;
    for (int i = 0; i < N_GPU; ++i) if (j[i].dt > worst) worst = j[i].dt;
    return worst;
}

int main(int argc, char ** argv) {
    if (argc > 1) N = atoi(argv[1]);
    gB  = (uint8_t *) aligned_alloc(64, (size_t) N * K);
    gA  = (float *) aligned_alloc(64, (size_t) M * K * 4);
    gA1 = (uint8_t *) aligned_alloc(64, (size_t) M * (K / 32) * 38 + 64);
    gA4 = (uint8_t *) aligned_alloc(64, (size_t) (K / 32) * 152 + 64);
    gC1 = (float *) aligned_alloc(64, (size_t) M * N * 4);
    gC4 = (float *) aligned_alloc(64, (size_t) M * N * 4);
    if (!gB || !gA || !gA1 || !gA4 || !gC1 || !gC4) { printf("oom\n"); return 1; }

    srand(7);
    for (int i = 0; i < N * K; ++i) gB[i] = (uint8_t) (rand() & 0xFF);
    for (int i = 0; i < M * K; ++i) gA[i] = (float) ((rand() % 200) - 100) * 0.01f;

    for (int r = 0; r < M; ++r) pack_m1(gA + (int64_t) r * K, K, gA1 + (int64_t) r * (K / 32) * 38);
    pack_m4(gA, K, K, gA4);

    open_a100();

    const double t1 = run_mode(0);
    const double t4 = run_mode(1);

    // numerical agreement between the two packings
    double maxabs = 0, maxrel = 0;
    for (int r = 0; r < M; ++r)
        for (int n = 0; n < N; ++n) {
            const double a = gC1[(int64_t) r * N + n], b = gC4[(int64_t) r * N + n];
            maxabs = fmax(maxabs, fabs(a - b));
            maxrel = fmax(maxrel, fabs(a - b) / (fabs(a) + 1e-9));
        }

    const double mac = (double) M * N * K;
    printf("K=%d N=%d (%.1f MB) rows=%d\n", K, N, (double) N * K / 1e6, M);
    printf("  4x m1 calls : %8.3f ms  %6.1f GMAC/s\n", t1 * 1e3, mac / t1 / 1e9);
    printf("  native m=4  : %8.3f ms  %6.1f GMAC/s\n", t4 * 1e3, mac / t4 / 1e9);
    printf("  speedup     : %.2fx\n", t1 / t4);
    printf("  agreement   : maxabs %.3e  maxrel %.3e  %s\n", maxabs, maxrel,
           (maxabs < 1e-3 || maxrel < 1e-5) ? "OK" : "MISMATCH");
    return 0;
}
