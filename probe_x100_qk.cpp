// Can the X100 cluster do QK, and at what rate?
//
// QK is 4 queries x (many positions) with K = 128 dims, i.e. the same 4-row GEMM
// shape the X100 offload already runs, but with the shortest possible K. The
// X100 kernel spends 8 vxor + 8 stores + setup outside the loop, so a K=128 call
// has 16 loop iterations to pay that off against, where a 6144-wide projection
// has 768. This probe measures the rate at each K so the answer is a number
// instead of an extrapolation from the long-K result.
//
// The production A100 QK rate is 41.4 GMAC/s per core at 10K (contended with PV),
// 57.8 with PV removed, and 131.7 for the loop alone with K hot. X100 has to beat
// 41.4 to be worth the port.
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

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

// 4 rows x 16 columns, K = nstep*8. Verbatim from mega3.cpp.
static void x100_gemm_4x16(int32_t * out, const int8_t * A, const int8_t * B, int64_t nstep) {
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
        "mv     t3, %[NS]          \n\t"
        "mv     t4, %[A]           \n\t"
        "mv     t5, %[B]           \n\t"
        "1:                        \n\t"
        "vle32.v v0, (t4)          \n\t"
        "addi    t6, t4, 16        \n\t"
        "vle32.v v1, (t6)          \n\t"
        "addi    t4, t4, 32        \n\t"
        "vle32.v v2, (t5)          \n\t"
        "addi    t5, t5, 32        \n\t"
        "vle32.v v3, (t5)          \n\t"
        "addi    t5, t5, 32        \n\t"
        "vle32.v v4, (t5)          \n\t"
        "addi    t5, t5, 32        \n\t"
        "vle32.v v5, (t5)          \n\t"
        "addi    t5, t5, 32        \n\t"
        "vmadot  v16, v0, v2       \n\t"
        "vmadot  v18, v0, v3       \n\t"
        "vmadot  v20, v0, v4       \n\t"
        "vmadot  v22, v0, v5       \n\t"
        "vmadot  v24, v1, v2       \n\t"
        "vmadot  v26, v1, v3       \n\t"
        "vmadot  v28, v1, v4       \n\t"
        "vmadot  v30, v1, v5       \n\t"
        "addi    t3, t3, -1        \n\t"
        "bgtz    t3, 1b            \n\t"
        "mv     t1, %[O]           \n\t"
        "vse32.v v16, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v18, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v20, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v22, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v24, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v26, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v28, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v30, (t1)         \n\t"
        : /* no outputs */
        : [NS] "r"(nstep), [A] "r"(A), [B] "r"(B), [O] "r"(out)
        : "t0", "t1", "t3", "t4", "t5", "t6",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

static int8_t * g_kp;      // K plane: [posblk][nstep][128]  (16 positions x K dims)
static int8_t * g_qp;      // Q records: [qgroup][K/8][32]
static int32_t g_out[64];

static int K = 128;        // contraction length
static int NPB = 160;      // position blocks per call-set (16 positions each)
static int NG = 16;        // query groups of 4
static int NT = 8;

static double g_tm[8];

static void * th(void * p) {
    const long id = (long) p;
    bind_cpu(id);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (int pb = id; pb < NPB; pb += NT) {
            const int8_t * b = g_kp + (size_t) pb * (K / 8) * 128;
            for (int g = 0; g < NG; ++g)
                x100_gemm_4x16(g_out, g_qp + (size_t) g * (K / 8) * 32, b, K / 8);
        }
        const double d = now_s() - t0;
        if (rep >= 2 && d < best) best = d;
    }
    g_tm[id] = best;
    return nullptr;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) NPB = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")) NG = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) NT = atoi(argv[++i]);
    }
    g_kp = (int8_t *) aligned_alloc(4096, (size_t) NPB * (K / 8) * 128 + 4096);
    g_qp = (int8_t *) aligned_alloc(4096, (size_t) NG * (K / 8) * 32 + 4096);
    if (!g_kp || !g_qp) { printf("oom\n"); return 1; }
    memset(g_kp, 3, (size_t) NPB * (K / 8) * 128);
    memset(g_qp, 5, (size_t) NG * (K / 8) * 32);
    pthread_t t[8];
    for (long i = 0; i < NT; ++i) pthread_create(&t[i], nullptr, th, (void *) i);
    for (int i = 0; i < NT; ++i) pthread_join(t[i], nullptr);
    double tm = 0;
    for (int i = 0; i < NT; ++i) if (g_tm[i] > tm) tm = g_tm[i];
    // every thread does ceil((NPB-id)/NT) blocks, each NG groups of 4x16xK
    const double calls = (double) NG * (double) NPB;
    const double mac = calls * 4.0 * 16.0 * (double) K;
    printf("K=%-5d npb=%-4d ng=%-3d thr=%d: %.3f ms  %.1f GMAC/s aggregate  %.1f per core\n",
           K, NPB, NG, NT, tm * 1e3, mac / tm / 1e9, mac / tm / 1e9 / NT);
    return 0;
}
