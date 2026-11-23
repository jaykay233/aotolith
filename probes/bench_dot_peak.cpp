// Pure vmadotsu throughput on one A100 core. The vendor's int4 GEMM spends
// roughly 37 non-multiply vector instructions per 8 vmadotsu, so if the dot
// unit can run far faster than the 45 GMAC/s the vendor kernel achieves, a
// hand-written kernel is worth it -- that is the only way prefill can approach
// 200 tok/s at 10K.
//
// vmadotsu e32,m1 with an i4 operand: 32 int32 lanes, each consuming 4 bytes of
// each source, i.e. 8 int4 products per lane = 256 MAC per instruction.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

#define NACC 8

// NACC independent accumulators, one vmadotsu each per iteration.
static double bench_i4(int64_t iters, int nacc) {
    const int64_t macs = (double) iters * (double) nacc * 256.0;
    double t0, dt;
    if (nacc == 8) {
        asm volatile(
            "vsetvli t0, x0, e32, m1       \n\t"
            "vxor.vv v16, v16, v16         \n\t"
            "vxor.vv v18, v18, v18         \n\t"
            "vxor.vv v20, v20, v20         \n\t"
            "vxor.vv v22, v22, v22         \n\t"
            "vxor.vv v24, v24, v24         \n\t"
            "vxor.vv v26, v26, v26         \n\t"
            "vxor.vv v28, v28, v28         \n\t"
            "vxor.vv v30, v30, v30         \n\t"
            "1:                            \n\t"
            "vmadotsu v16, v3, v4, i4      \n\t"
            "vmadotsu v18, v3, v5, i4      \n\t"
            "vmadotsu v20, v3, v6, i4      \n\t"
            "vmadotsu v22, v3, v7, i4      \n\t"
            "vmadotsu v24, v3, v8, i4      \n\t"
            "vmadotsu v26, v3, v9, i4      \n\t"
            "vmadotsu v28, v3, v10, i4     \n\t"
            "vmadotsu v30, v3, v11, i4     \n\t"
            "addi %[N], %[N], -1           \n\t"
            "bgtz %[N], 1b                 \n\t"
            : [N] "+r"(iters)
            :
            : "t0", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v18", "v20", "v22", "v24", "v26",
              "v28", "v30", "memory");
    } else {
        asm volatile(
            "vsetvli t0, x0, e32, m1       \n\t"
            "vxor.vv v16, v16, v16         \n\t"
            "vxor.vv v18, v18, v18         \n\t"
            "vxor.vv v20, v20, v20         \n\t"
            "vxor.vv v22, v22, v22         \n\t"
            "1:                            \n\t"
            "vmadotsu v16, v3, v4, i4      \n\t"
            "vmadotsu v18, v3, v5, i4      \n\t"
            "vmadotsu v20, v3, v6, i4      \n\t"
            "vmadotsu v22, v3, v7, i4      \n\t"
            "addi %[N], %[N], -1           \n\t"
            "bgtz %[N], 1b                 \n\t"
            : [N] "+r"(iters)
            :
            : "t0", "v3", "v4", "v5", "v6", "v7", "v16", "v18", "v20", "v22", "memory");
    }
    return 0;
}

// Same, but int8 x int8 (no i4), i.e. 4 bytes x 4 bytes = 128 MAC per instr.
static void bench_i8(int64_t iters) {
    asm volatile(
        "vsetvli t0, x0, e32, m1       \n\t"
        "vxor.vv v16, v16, v16         \n\t"
        "vxor.vv v18, v18, v18         \n\t"
        "vxor.vv v20, v20, v20         \n\t"
        "vxor.vv v22, v22, v22         \n\t"
        "1:                            \n\t"
        "vmadot v16, v3, v4            \n\t"
        "vmadot v18, v3, v5            \n\t"
        "vmadot v20, v3, v6            \n\t"
        "vmadot v22, v3, v7            \n\t"
        "addi %[N], %[N], -1           \n\t"
        "bgtz %[N], 1b                 \n\t"
        : [N] "+r"(iters)
        :
        : "t0", "v3", "v4", "v5", "v6", "v7", "v16", "v18", "v20", "v22", "memory");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    int cpu = argc > 1 ? atoi(argv[1]) : 8;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("cpu %d: bind FAILED\n", cpu); return 1; }
    size_t vlenb = 0; asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    printf("cpu %d: vlenb=%zu\n", cpu, vlenb);

    // warm
    bench_i4(1000, 8);
    bench_i4(1000, 4);

    const int64_t N = 2000000;
    for (int acc = 4; acc <= 8; acc += 4) {
        const double t0 = now_s();
        bench_i4(N, acc);
        const double dt = now_s() - t0;
        const double macs = (double) N * acc * 256.0;
        printf("vmadotsu i4, %d indep accumulators: %.1f GMAC/s  = %.1f MAC/cycle@1.6G  (%.2f ns per instr)\n",
               acc, macs / dt / 1e9, macs / dt / 1.6e9, dt / (double) N / acc * 1e9);
    }
    {
        const int64_t N8 = 2000000;
        const double t0 = now_s();
        bench_i8(N8);
        const double dt = now_s() - t0;
        const double macs = (double) N8 * 4 * 128.0;
        printf("vmadot   i8, 4 indep accumulators: %.1f GMAC/s  = %.1f MAC/cycle@1.6G  (%.2f ns per instr)\n",
               macs / dt / 1e9, macs / dt / 1.6e9, dt / (double) N8 / 4 * 1e9);
    }
    return 0;
}
