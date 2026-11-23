// What does it cost to verify m draft tokens instead of decoding one?
//
// Speculative decoding only pays if m rows of the target model cost close to what
// one row costs, because at m=1 the weight stream is the whole bill: 1265 MB per
// token at 24 GB/s is 50 ms, and no amount of arithmetic changes that. If m rows
// reuse the same B tile while it is in L1, the stream is amortised and the extra
// rows cost only their MACs.
//
// This measures exactly that, using the same vendor calls and the same n-outer /
// m-inner nesting that prefill uses. A (the activations) is quantized once per row
// and reused, which is what happens naturally in a decode step too: the m rows
// share one set of weights, not one set of activations.
//
// Contract, from ime2_kernels.cpp's mrow reference:
//   A per 32-k block, per row: [f32 scale][i16 sum][i8 x32] = 38 bytes
//   B per 32-n group, per 32-k block: [f16 scale x32][16 B x32] = 576 bytes
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <pthread.h>

#include "ggml_stubs.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const int N_GPU = 8;
// /proc/set_ai_thread unlocks the A100 cluster, but writing it while another
// thread is inside pthread_create faults at NULL+8 in libc's lock path. So it is
// written exactly once, in main, before any thread exists -- never from a worker.
static void open_a100(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
}

static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

// One real matrix shape from the model: FFN gate, 2048 -> 6144.
// N must be large enough that the weight plane cannot sit in cache: the FFN gate
// alone (12.6 MB) benchmarks at 51 GB/s here because it is L2-resident, which hides
// the very effect being measured. At N = 65536 the plane is 134 MB and every byte
// comes from DRAM, which is the regime the real 1265 MB decode stream runs in.
static int K = 2048;
static int N = 65536;
static const int MROWS = 8;
static uint8_t * gB;          // packed weights, (N/32)*(K/32)*576
static float * gA;            // MROWS x K activations
static uint8_t * gAq;         // MROWS x (K/32)*38
static float * gC;            // MROWS x N

struct Job { int tid; int m; double dt; double bytes; };

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    const int nb = N / 32, kb = K / 32;
    const int per = (nb + N_GPU - 1) / N_GPU;
    const int n0 = per * j->tid, n1 = (j->tid == N_GPU - 1) ? nb : per * (j->tid + 1);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        // n outer, m inner: the B tile is loaded once and used for all m rows.
        for (int a = n0; a < n1; ++a) {
            const uint8_t * bcol = gB + (int64_t) a * kb * 576;
            for (int r = 0; r < j->m; ++r) {
                spacemit_kernels::ime2::gemm_kernel_i8i4(32, gAq + (int64_t) r * kb * 38, bcol, nullptr,
                                       gC + (int64_t) r * N + a * 32,
                                       1, 32, kb, N);
            }
        }
        const double d = now_s() - t0;
        if (rep >= 2 && d < best) best = d;
    }
    j->dt = best;
    j->bytes = (double) (n1 - n0) * kb * 576;
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc > 1) N = atoi(argv[1]);
    gB = (uint8_t *) aligned_alloc(64, (size_t) N * K);
    gA = (float *) aligned_alloc(64, (size_t) MROWS * K * 4);
    gAq = (uint8_t *) aligned_alloc(64, (size_t) MROWS * (K / 32) * 38 + 64);
    gC = (float *) aligned_alloc(64, (size_t) MROWS * N * 4);
    if (!gB || !gA || !gAq || !gC) { printf("oom\n"); return 1; }
    open_a100();
    srand(5);
    for (int i = 0; i < N * K; ++i) gB[i] = (uint8_t) (rand() & 0xFF);
    for (int i = 0; i < MROWS * K; ++i) gA[i] = (float) ((rand() % 200) - 100) * 0.01f;
    for (int r = 0; r < MROWS; ++r)
        spacemit_kernels::rvv::quantize_a_row_i8(32, gA + (int64_t) r * K, K,
                                                 gAq + (int64_t) r * (K / 32) * 38);

    printf("FFN gate shape: K=%d N=%d, weights %.1f MB\n", K, N, N * (double) K / 1e6);
    printf("%-3s %-10s %-10s %-12s %-10s %s\n",
           "m", "time ms", "GB/s", "GMAC/s", "vs m=1", "tokens/s from this matrix");

    double base = 0;
    for (int m = 1; m <= MROWS; ++m) {
        Job j[N_GPU];
        pthread_t t[N_GPU];
        for (int i = 0; i < N_GPU; ++i) j[i] = Job{i, m, 0, 0};
        for (int i = 1; i < N_GPU; ++i) pthread_create(&t[i], nullptr, th, &j[i]);
        th(&j[0]);
        for (int i = 1; i < N_GPU; ++i) pthread_join(t[i], nullptr);
        double worst = 0; double bytes = 0;
        for (int i = 0; i < N_GPU; ++i) { if (j[i].dt > worst) worst = j[i].dt; bytes += j[i].bytes; }
        const double mac = (double) m * N * K;
        if (m == 1) base = worst;
        printf("%-3d %-10.3f %-10.1f %-12.0f %-10.2f %.1f\n",
               m, worst * 1e3, bytes / worst / 1e9, mac / worst / 1e9,
               worst / base, (double) m / worst);
    }
    printf("\nReading: the 'vs m=1' column is the whole point. If m rows cost close to\n"
           "one row, the weight stream is being amortised and speculative decoding pays.\n");
    return 0;
}
