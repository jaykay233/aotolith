// X100 prefill GEMM at realistic shapes, with weights streamed from DRAM.
//
// probe_x100_gemm.cpp measured the arithmetic at 76.7 GMAC/s/core with both
// operands L1-resident. Real prefill streams 1.26 GB of weights, so this asks the
// question that actually decides the port: does that rate survive when B comes
// from DRAM through the packed layout?
//
// The layout and the geometry live in x100_layout.h, together with a software
// model of the kernel (sim_gemm) that mirrors the asm rather than sharing code with
// it. The correctness check here is that model against the asm, which is where the
// bugs that only show up on hardware come from.
//
// Two assembler constraints this file obeys, both established the hard way:
//   * vector loads and stores reject non-zero displacement -- only (ptr) parses,
//     so every access is a zero-displacement load off a pointer advanced in between
//   * vmadot destinations must be even -- v16, v18, ..., v30
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#include "x100_layout.h"

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// K = 2048 keeps the B plane at 8 MB for N = 4096, well past L2, which is the
// regime prefill actually runs in.
static const int M = 1024, N = 4096, K = 2048;

static int8_t * gA;      // packed (M/ROWS) * (K/DEPTH) * ASTEP
static int8_t * gB;      // packed (N/COLS) * (K/DEPTH) * BSTEP
static int8_t * rawA;    // unpacked, for the scalar reference
static int8_t * rawB;
static int32_t * gScratch;   // 8 accumulators x 8 lanes, laid out [cg][lane]

// One call covers 4 rows x 16 columns.
//
// Per 8-deep step: 2 A registers (one row pair each) and 4 B registers (one column
// group each) feed 8 vmadots. That is 192 B for 512 MAC, 0.375 B/MAC, which is the
// measured operating point rather than an estimate.
static void x100_gemm(int32_t * out, const int8_t * arow, const int8_t * bcol, int64_t nstep) {
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
        "mv     t3, %[NS]               \n\t"
        "mv     t4, %[A]                \n\t"
        "mv     t5, %[B]                \n\t"
        "1:                             \n\t"
        "vle32.v v0, (t4)               \n\t"   // A: rows 0-1
        "addi    t4, t4, 32             \n\t"
        "vle32.v v1, (t4)               \n\t"   // A: rows 2-3
        "addi    t4, t4, 32             \n\t"
        "vle32.v v2, (t5)               \n\t"   // B: columns 0-3
        "addi    t5, t5, 32             \n\t"
        "vle32.v v3, (t5)               \n\t"   // B: columns 4-7
        "addi    t5, t5, 32             \n\t"
        "vle32.v v4, (t5)               \n\t"   // B: columns 8-11
        "addi    t5, t5, 32             \n\t"
        "vle32.v v5, (t5)               \n\t"   // B: columns 12-15
        "addi    t5, t5, 32             \n\t"
        "vmadot  v16, v0, v2            \n\t"   // rows 0-1, cols 0-3
        "vmadot  v18, v0, v3            \n\t"   // rows 0-1, cols 4-7
        "vmadot  v20, v0, v4            \n\t"   // rows 0-1, cols 8-11
        "vmadot  v22, v0, v5            \n\t"   // rows 0-1, cols 12-15
        "vmadot  v24, v1, v2            \n\t"   // rows 2-3, cols 0-3
        "vmadot  v26, v1, v3            \n\t"   // rows 2-3, cols 4-7
        "vmadot  v28, v1, v4            \n\t"   // rows 2-3, cols 8-11
        "vmadot  v30, v1, v5            \n\t"   // rows 2-3, cols 12-15
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        // 8 lanes per accumulator: lane = row_in_pair*4 + col_in_group, so the
        // stores are one 32-byte block per column group.
        "mv     t1, %[O]                \n\t"
        "vse32.v v16, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v18, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v20, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v22, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v24, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v26, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v28, (t1)              \n\t"
        "addi    t1, t1, 32             \n\t"
        "vse32.v v30, (t1)              \n\t"
        : /* no outputs */
        : [NS] "r"(nstep), [A] "r"(arow), [B] "r"(bcol), [O] "r"(out)
        : "t0", "t1", "t3", "t4", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// Kernel output is [colgroup][lane]; the scalar model writes [row][col].
static void unpack(int32_t * out, const int32_t * buf) {
    for (int r = 0; r < x100::ROWS; ++r)
        for (int c = 0; c < x100::COLS; ++c)
            out[r * x100::COLS + c] = buf[(c / 4) * 8 + (r % 2) * 4 + (c % 4)];
}

static int g_nthread = 8;
static double g_dt[8];

static void * th(void * p) {
    const int tid = (int) (intptr_t) p;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(tid, &s);
    sched_setaffinity(0, sizeof(s), &s);

    static int32_t acc[8][64];
    const int64_t nstep = K / x100::DEPTH;
    const int nb = N / x100::COLS, mb = M / x100::ROWS;
    const int per = (nb + g_nthread - 1) / g_nthread;
    const int n0 = per * tid, n1 = (tid == g_nthread - 1) ? nb : per * (tid + 1);

    const double t0 = now_s();
    // n outer, m inner: each 128-byte B step is loaded once and reused across all
    // row blocks, which is the whole reason this shape reaches DRAM bandwidth.
    for (int n = n0; n < n1; ++n) {
        const int8_t * bp = gB + (int64_t) n * nstep * x100::BSTEP;
        for (int m = 0; m < mb; ++m) {
            const int8_t * ap = gA + (int64_t) m * nstep * x100::ASTEP;
            x100_gemm(acc[tid], ap, bp, nstep);
        }
    }
    g_dt[tid] = now_s() - t0;
    return nullptr;
}

int main(int argc, char ** argv) {
    g_nthread = argc > 1 ? atoi(argv[1]) : 8;
    const int64_t nstep = K / x100::DEPTH;
    gA = (int8_t *) aligned_alloc(64, (size_t) (M / x100::ROWS) * nstep * x100::ASTEP);
    gB = (int8_t *) aligned_alloc(64, (size_t) (N / x100::COLS) * nstep * x100::BSTEP);
    rawA = (int8_t *) aligned_alloc(64, (size_t) M * K);
    rawB = (int8_t *) aligned_alloc(64, (size_t) N * K);
    gScratch = (int32_t *) aligned_alloc(64, 64 * 4);
    if (!gA || !gB || !rawA || !rawB || !gScratch) { printf("oom\n"); return 1; }
    srand(1234);
    for (int i = 0; i < M * K; ++i) rawA[i] = (int8_t) (rand() % 17 - 8);
    for (int i = 0; i < N * K; ++i) rawB[i] = (int8_t) (rand() % 17 - 8);

    x100::pack_a(gA, rawA, M, K, K);
    x100::pack_b(gB, rawB, N, K, K);

    // Exact check: asm against the scalar dot on one 4x16 block.
    int32_t ref[64], got[64];
    x100::sim_gemm(ref, gA, gB, K);
    x100_gemm(gScratch, gA, gB, nstep);
    unpack(got, gScratch);
    int bad = 0;
    for (int r = 0; r < x100::ROWS; ++r)
        for (int c = 0; c < x100::COLS; ++c) {
            int32_t exact = 0;
            for (int k = 0; k < K; ++k) exact += rawA[r * K + k] * rawB[c * K + k];
            if (got[r * x100::COLS + c] != exact) ++bad;
            if (ref[r * x100::COLS + c] != exact) ++bad;
        }
    printf("4x16 x K=%d vs exact scalar: %d/%d wrong (model+asm)\n",
           K, bad, 2 * x100::ROWS * x100::COLS);
    printf("B %.1f MB, A %.1f MB, %d threads, %d depth steps\n",
           N * (double) K / 1e6, M * (double) K / 1e6, g_nthread, (int) nstep);

    for (int rep = 0; rep < 3; ++rep) {
        pthread_t t[8];
        for (int i = 1; i < g_nthread; ++i)
            pthread_create(&t[i], nullptr, th, (void *) (intptr_t) i);
        th(nullptr);
        for (int i = 1; i < g_nthread; ++i) pthread_join(t[i], nullptr);
        double worst = 0;
        for (int i = 0; i < g_nthread; ++i) if (g_dt[i] > worst) worst = g_dt[i];
        const double mac = (double) M * N * K;
        printf("  run %d: %7.1f ms  aggregate %5.0f GMAC/s  (%.1f per core)\n",
               rep, worst * 1e3, mac / worst / 1e9, mac / worst / 1e9 / g_nthread);
    }
    return 0;
}
