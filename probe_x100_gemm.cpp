// What dot rate can the X100 cluster actually reach for a GEMM-shaped loop?
//
// probe_x100_map.cpp established the geometry: one vmadot is 2 rows x 4 columns
// x 8 depth = 64 MACs, with
//     L0..L3 = A[0..7]  x B[8i..8i+7]      (row 0, cols 4i..4i+3)
//     L4..L7 = A[8..15] x B[8i..8i+7]      (row 1, cols 4i..4i+3)
// so a B register covers one column group of four, and A's low 16 bytes are the
// only ones read.
//
// That shape makes the operator 8x narrower than on A100 (64 vs 512 MAC per
// instruction), so the question is not "does it work" but "what does the
// per-instruction operand traffic cost". Two A registers plus four B registers
// feed eight vmadots = 512 MACs, i.e. 0.375 B/MAC, against 0.265 on A100. If the
// achieved rate lands near 79% of the cluster peak (52.5 GMAC/s/core) the port is
// worth building; if it comes out near the A100's own 103 it is not.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static long vlenb() { long v = 0; asm volatile("csrr %0, vlenb" : "=r"(v)); return v; }
static int bind_x100() {
    cpu_set_t s; CPU_ZERO(&s);
    for (int i = 0; i < 8; ++i) CPU_SET(i, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

// kstep iterations of: 2 A loads, 4 B loads, 8 vmadots.
//   acc v16..v23 = 8 accumulators = 2 row pairs x 4 column groups
//   v0, v1       = A for row pair 0 and row pair 1
//   v2..v5       = B for column groups 0..3
// A advances by 32 B per row pair (two rows of 8 depth each ... one register of
// 32 bytes holds both rows), B by 32 B per column group.
static void gemm_loop(int64_t kstep, const int8_t * A0, const int8_t * A1, const int8_t * B,
                      int32_t * out) {
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
        "mv     t3, %[NK]               \n\t"
        "mv     t4, %[A0]               \n\t"
        "mv     t5, %[A1]               \n\t"
        "mv     t6, %[B]                \n\t"
        "1:                             \n\t"
        "vle32.v v0, (t4)               \n\t"
        "vle32.v v1, (t5)               \n\t"
        "vle32.v v2, (t6)               \n\t"
        "addi    t6, t6, 32             \n\t"
        "vle32.v v3, (t6)               \n\t"
        "addi    t6, t6, 32             \n\t"
        "vle32.v v4, (t6)               \n\t"
        "addi    t6, t6, 32             \n\t"
        "vle32.v v5, (t6)               \n\t"
        "addi    t6, t6, 32             \n\t"
        "addi    t4, t4, 32             \n\t"
        "addi    t5, t5, 32             \n\t"
        "vmadot  v16, v0, v2            \n\t"
        "vmadot  v18, v0, v3            \n\t"
        "vmadot  v20, v0, v4            \n\t"
        "vmadot  v22, v0, v5            \n\t"
        "vmadot  v24, v1, v2            \n\t"
        "vmadot  v26, v1, v3            \n\t"
        "vmadot  v28, v1, v4            \n\t"
        "vmadot  v30, v1, v5            \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        "vse32.v v16, (%[O0])           \n\t"
        "vse32.v v18, (%[O1])           \n\t"
        "vse32.v v20, (%[O2])           \n\t"
        "vse32.v v22, (%[O3])           \n\t"
        "vse32.v v24, (%[O4])           \n\t"
        "vse32.v v26, (%[O5])           \n\t"
        "vse32.v v28, (%[O6])           \n\t"
        "vse32.v v30, (%[O7])           \n\t"
        : /* no outputs */
        : [NK] "r"(kstep), [A0] "r"(A0), [A1] "r"(A1), [B] "r"(B),
          [O0] "r"(out), [O1] "r"(out + 8), [O2] "r"(out + 16), [O3] "r"(out + 24),
          [O4] "r"(out + 32), [O5] "r"(out + 40), [O6] "r"(out + 48), [O7] "r"(out + 56)
        : "t0", "t3", "t4", "t5", "t6",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// 8 vmadots per iteration, 64 MACs each. 4 rows x 16 cols per pass over 8 depth.
static const int VPER_ITER = 8;
static const int MAC_PER_VMADOT = 64;

// Both operands fit in L1 at these sizes, which is the regime the real GEMM runs
// in (A is the activation block, B the 576-byte weight column tile).
static int8_t gA[4096] __attribute__((aligned(64)));
static int8_t gB[4096] __attribute__((aligned(64)));
static int32_t gO[64];

// Per-thread weight region: 8 threads must not share one B buffer, or the test
// measures L1 broadcast instead of what eight cores streaming eight slices see.
// 256KB + 4KB per slice, and each thread starts at a different offset inside
// its slice. A power-of-two stride with everyone starting at offset 0 makes all
// eight cores touch the same L2 sets on every iteration, which showed up as a
// bimodal 620 / 400 GMAC/s depending on how the allocator happened to page it.
static int8_t gBbig[8][260 * 1024] __attribute__((aligned(64)));
static int32_t gObig[8][64];

struct Job { int tid; int kstep; int64_t iters; double dt; };

static void * th(void * p) {
    Job * j = (Job *) p;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->tid, &s);
    sched_setaffinity(0, sizeof(s), &s);
    const int8_t * b = gBbig[j->tid] + j->tid * 4160;   // stagger the start
    gemm_loop(j->kstep, gA, gA + 32, b, gObig[j->tid]);       // warm
    const double t0 = now_s();
    for (int64_t c = 0; c < j->iters; ++c) gemm_loop(j->kstep, gA, gA + 32, b, gObig[j->tid]);
    j->dt = now_s() - t0;
    return nullptr;
}

int main() {
    if (bind_x100()) { printf("bind failed\n"); return 1; }
    memset(gA, 1, sizeof gA);
    memset(gB, 1, sizeof gB);
    const long vb = vlenb();
    printf("vlenb=%ld cores=0-7\n", vb);
    // sanity: 8 vmadots of all-ones must report 8 MACs per lane, 8 lanes
    gemm_loop(1, gA, gA + 32, gB, gO);
    long chk = 0;
    for (int i = 0; i < 64; ++i) chk += gO[i];
    printf("all-ones check: sum of 64 lanes = %ld (expect %d)\n", chk, 64 * 8);

    // kstep of 16 == one full 128-deep dot, the real inner-block size.
    // Repeat the block many times so the timer sees steady state.
    printf("\n%-6s %-10s %-12s %s\n", "kstep", "iters", "dots", "GMAC/s/core");
    const int ksteps[] = {8, 16, 32};
    for (int t = 0; t < 3; ++t) {
        const int ks = ksteps[t];
        const int64_t iters = 4000000 / ks;
        gemm_loop(ks, gA, gA + 32, gB, gO);            // warm
        const double t0 = now_s();
        for (int64_t c = 0; c < iters; ++c) gemm_loop(ks, gA, gA + 32, gB, gO);
        const double dt = now_s() - t0;
        const double dots = (double) iters * ks * VPER_ITER;
        printf("%-6d %-10lld %-12.0f %.1f\n", ks, (long long) iters, dots,
               dots * MAC_PER_VMADOT / dt / 1e9);
    }
    // eight cores, each with its own weight slice
    printf("\n=== 8 threads, per-thread B region ===\n");
    for (int i = 0; i < 8; ++i) memset(gBbig[i], 1, sizeof gBbig[i]);
    {
        Job j[8]; pthread_t t[8];
        for (int i = 0; i < 8; ++i) j[i] = Job{i, 32, 128000, 0};
        for (int i = 1; i < 8; ++i) pthread_create(&t[i], nullptr, th, &j[i]);
        j[0].tid = 0; th(&j[0]);
        for (int i = 1; i < 8; ++i) pthread_join(t[i], nullptr);
        double worst = 0, sum = 0;
        for (int i = 0; i < 8; ++i) { if (j[i].dt > worst) worst = j[i].dt; sum += j[i].dt; }
        const double dots = 128000.0 * 32.0 * 8.0;
        printf("per core %.1f GMAC/s   aggregate %.0f GMAC/s   (worst %.4f s)\n",
               dots * 64 / worst / 1e9, dots * 8 * 64 / worst / 1e9, worst);
    }
    return 0;
}
