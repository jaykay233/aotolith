// Why do the attention dots run at 22-27 cycles while the GEMM's identical loop
// runs at 9 and a fixed-address version reaches the 7-cycle peak?
//
// The three attention call sites differ from the GEMM in one measurable way: how
// many loop iterations each asm invocation performs. qk4 is called with
// nks = HEAD_DIM/16 = 8, so every call pays the prologue (vsetvli + 8 vxor) and
// epilogue (8 vse32) for only 64 dots. pv4 gets 640, gemm gets 128. This probe
// runs the same body at each trip count with and without the pointer advance, so
// prologue overhead and memory pattern are separated instead of confounded.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <pthread.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int cur_cpu() {
    cpu_set_t s; CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof(s), &s)) return -1;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &s)) return i;
    return -1;
}
static long vlenb() { long v = 0; asm volatile("csrr %0, vlenb" : "=r"(v)); return v; }

static int bind_c(int cpu) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

// trip iterations, 8 dots each. stride is a runtime value so 0 (fixed address,
// everything resident in L1) and 1024 (the real walk) use the same instruction
// stream -- an earlier comparison across two different functions was confounded
// by code layout and produced a wrong 13x.
static void dot_loop(int64_t trip, const int8_t * A, const int8_t * B, int32_t * C,
                     int64_t stride) {
    asm volatile(
        "vsetvli t0, x0, e32, m1\n\t"
        "vxor.vv v16, v16, v16\n\t"
        "vxor.vv v18, v18, v18\n\t"
        "vxor.vv v20, v20, v20\n\t"
        "vxor.vv v22, v22, v22\n\t"
        "vxor.vv v24, v24, v24\n\t"
        "vxor.vv v26, v26, v26\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vxor.vv v30, v30, v30\n\t"
        "mv     t3, %[N]\n\t"
        "mv     t4, %[A]\n\t"
        "mv     t5, %[B]\n\t"
        "L%=:\n\t"
        "vl1re32.v v0, (t4)\n\t"
        "vl8re32.v v8, (t5)\n\t"
        "addi   t4, t4, 64\n\t"
        "add    t5, t5, %[S]\n\t"
        "vmadot v16, v0, v8\n\t"
        "vmadot v18, v0, v9\n\t"
        "vmadot v20, v0, v10\n\t"
        "vmadot v22, v0, v11\n\t"
        "vmadot v24, v0, v12\n\t"
        "vmadot v26, v0, v13\n\t"
        "vmadot v28, v0, v14\n\t"
        "vmadot v30, v0, v15\n\t"
        "addi   t3, t3, -1\n\t"
        "bgtz   t3, L%=\n\t"
        "vse32.v v16, (%[O0])\n\t"
        "vse32.v v18, (%[O1])\n\t"
        "vse32.v v20, (%[O2])\n\t"
        "vse32.v v22, (%[O3])\n\t"
        "vse32.v v24, (%[O4])\n\t"
        "vse32.v v26, (%[O5])\n\t"
        "vse32.v v28, (%[O6])\n\t"
        "vse32.v v30, (%[O7])\n\t"
        : /* no outputs */
        : [N] "r"(trip), [A] "r"(A), [B] "r"(B), [S] "r"(stride),
          [O0] "r"(C), [O1] "r"(C + 32), [O2] "r"(C + 64), [O3] "r"(C + 96),
          [O4] "r"(C + 128), [O5] "r"(C + 160), [O6] "r"(C + 192), [O7] "r"(C + 224)
        : "t0", "t3", "t4", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// B plane large enough that a 640-trip walk is 640 KB and an 8-thread run does
// not have every thread in the same cache sets.
static int8_t BPLANE[8][1024 * 1024] __attribute__((aligned(4096)));
static int8_t APLANE[64] __attribute__((aligned(64)));
static int32_t CPLANE[64];

struct Job { int trip; int64_t calls; int64_t stride; int tid; double s; };

static double measure(int trip, int64_t calls, int64_t stride, int tid) {
    const int8_t * a = APLANE + tid * 0;
    const int8_t * b = BPLANE[tid];
    dot_loop(trip, a, b, CPLANE, stride);      // warm
    const double t0 = now_s();
    for (int64_t c = 0; c < calls; ++c) dot_loop(trip, a, b, CPLANE, stride);
    return now_s() - t0;
}

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_c(8 + j->tid);
    char name[32]; snprintf(name, sizeof name, "t%d", j->tid);
    j->s = measure(j->trip, j->calls, j->stride, j->tid);
    (void) name;
    return nullptr;
}

int main(int argc, char ** argv) {
    const bool mt = argc > 1 && !strcmp(argv[1], "mt");
    if (!mt) bind_c(8);
    const long vb = vlenb();
    memset(BPLANE, 1, sizeof BPLANE);
    memset(APLANE, 1, sizeof APLANE);
    printf("vlenb=%ld cpu=%d\n", vb, cur_cpu());
    printf("%-8s %-6s %-7s %-9s %-12s %s\n",
           "trip", "calls", "stride", "dots/call", "cycles/dot", "GMAC/s/core");
    const int trips[] = {8, 16, 32, 128, 640};
    const int64_t stride = 1024;
    for (int t = 0; t < 5; ++t) {
        const int trip = trips[t];
        // keep total dots roughly constant so run time stays comparable
        int64_t calls = 6400000 / trip;
        if (calls > 400000) calls = 400000;
        const double dt = measure(trip, calls, stride, 0);
        const double dots = (double) trip * calls;
        const double cyc = 2.0e9;      // A100 policy8 runs at 1.8 GHz; use 1.8
        const double gmac = dots * 512.0 / dt / 1e9;
        printf("%-8d %-6lld %-7lld %-9d %-12.2f %.1f\n",
               trip, (long long) calls, (long long) stride, trip * 8,
               cyc * dt / dots, gmac);
    }
    // fixed address: same instruction stream, stride 0
    {
        const int trip = 640;
        const int64_t calls = 10000;
        const double dt = measure(trip, calls, 0, 0);
        const double dots = (double) trip * calls;
        printf("%-8d %-6lld %-7d %-9d %-12.2f %.1f   <- stride 0\n",
               trip, (long long) calls, 0, trip * 8, 1.8e9 * dt / dots,
               dots * 512.0 / dt / 1e9);
    }
    if (mt) {
        // eight threads, each in its own 1 MB plane: the real attention shape
        Job j[8];
        pthread_t ths[8];
        for (int i = 0; i < 8; ++i) {
            j[i] = Job{640, 5000, 1024, i, 0};
        }
        for (int i = 1; i < 8; ++i) pthread_create(&ths[i], nullptr, th, &j[i]);
        bind_c(8);
        j[0].s = measure(640, 5000, 1024, 0);
        double worst = 0;
        for (int i = 1; i < 8; ++i) pthread_join(ths[i], nullptr);
        for (int i = 0; i < 8; ++i) if (j[i].s > worst) worst = j[i].s;
        const double dots = 640.0 * 5000.0;
        printf("8 threads own planes : %.1f GMAC/s/core (worst %.3fs)  agg %.0f\n",
               dots * 512.0 / worst / 1e9, worst, dots * 512.0 / worst / 1e9 * 8);
    }
    return 0;
}
