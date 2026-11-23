// Resolve the last unknown for moving attention onto the dot unit: the exact
// operand map of `vmadot` (int8 x int8, both operands one register).
//
// Hypothesis from the int4 probes: lane = row*8 + col with A at row*16+d and B
// at col*16+d. micro12 tested B at d*8+col and failed, so the B stride is what
// was wrong. Test every (row, col, d) against this map.
//
// Also measures throughput with 8 accumulators instead of 4: the int4 form went
// from 229 to 413 GMAC/s when given 8 independent accumulators, so the 115
// GMAC/s measured for int8 may just be latency bound, not the real rate.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static void dot8(const uint8_t * a, const uint8_t * b, int32_t * o) {
    asm volatile(
        "vsetvli t0, x0, e8, m1    \n\t"
        "vl1r.v  v0, (%[A])         \n\t"
        "vl1r.v  v4, (%[B])         \n\t"
        "vsetvli t0, x0, e32, m1    \n\t"
        "vxor.vv v16, v16, v16      \n\t"
        "vmadot  v16, v0, v4        \n\t"
        "vse32.v v16, (%[O])        \n\t"
        : : [A] "r"(a), [B] "r"(b), [O] "r"(o)
        : "t0", "v0", "v4", "v16", "memory");
}

// 8 independent accumulators, B groups v4..v7 like the vendor's m4 form.
static double bench8(int64_t iters) {
    const double macs = (double) iters * 8.0 * 128.0;   // 8 dots x 128 MAC
    asm volatile(
        "vsetvli t0, x0, e8, m1     \n\t"
        "vxor.vv v16, v16, v16      \n\t"
        "vxor.vv v18, v18, v18      \n\t"
        "vxor.vv v20, v20, v20      \n\t"
        "vxor.vv v22, v22, v22      \n\t"
        "vxor.vv v24, v24, v24      \n\t"
        "vxor.vv v26, v26, v26      \n\t"
        "vxor.vv v28, v28, v28      \n\t"
        "vxor.vv v30, v30, v30      \n\t"
        "1:                         \n\t"
        "vmadot v16, v0, v4         \n\t"
        "vmadot v18, v0, v5         \n\t"
        "vmadot v20, v0, v6         \n\t"
        "vmadot v22, v0, v7         \n\t"
        "vmadot v24, v0, v8         \n\t"
        "vmadot v26, v0, v9         \n\t"
        "vmadot v28, v0, v10        \n\t"
        "vmadot v30, v0, v11        \n\t"
        "addi  %[N], %[N], -1       \n\t"
        "bgtz  %[N], 1b             \n\t"
        : [N] "+r"(iters)
        :
        : "t0", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v16", "v18", "v20", "v22", "v24", "v26", "v28", "v30", "memory");
    return macs;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    static uint8_t a[128], b[128];
    static int32_t o[32];

    printf("== hypothesis: A[row*16+d], B[col*16+d], lane = row*8+col ==\n");
    int ok = 0, bad = 0;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 8; ++col) {
            const int lane = row * 8 + col;
            for (int d = 0; d < 16; ++d) {
                memset(a, 0, sizeof a); memset(b, 0, sizeof b); memset(o, 0, sizeof o);
                a[row * 16 + d] = 1;
                b[col * 16 + d] = 1;
                dot8(a, b, o);
                if (o[lane] == 1) ++ok; else {
                    if (bad < 10) printf("  miss r%d c%d d%d lane %d got %d\n", row, col, d, lane, o[lane]);
                    ++bad;
                }
            }
        }
    printf("  %d/%d exact\n", ok, ok + bad);

    printf("\n== full 16-dim dot, row0/col0, A=B=1 : lane0 should be 16 ==\n");
    memset(a, 0, sizeof a); memset(b, 0, sizeof b);
    for (int d = 0; d < 16; ++d) { a[0 * 16 + d] = 1; b[0 * 16 + d] = 1; }
    memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("  lane0=%d lane1=%d lane8=%d\n", o[0], o[1], o[8]);

    printf("\n== which A bytes matter at all ==\n");
    for (int j = 0; j < 128; ++j) {
        memset(a, 0, sizeof a); memset(b, 1, sizeof b); memset(o, 0, sizeof o);
        a[j] = 1;
        dot8(a, b, o);
        int n = 0;
        for (int i = 0; i < 32; ++i) if (o[i]) ++n;
        if (!n) printf("  A[%3d] unused\n", j);
    }

    printf("\n== i8 dot throughput, 8 accumulators ==\n");
    bench8(1000);
    const int64_t N = 2000000;
    const double t0 = now_s();
    const double macs = bench8(N);
    const double dt = now_s() - t0;
    printf("  %.1f GMAC/s  (%.1f MAC/cycle@1.6G, %.3f ns/instr)\n",
           macs / dt / 1e9, macs / dt / 1.6e9, dt / (double) N / 8.0 * 1e9);
    return 0;
}
