// Sustained DRAM read bandwidth, single-threaded, run N times in parallel.
//
// Two earlier attempts were wrong for two different reasons: the first put a
// dependent vector add after every 32-byte load (so it measured a 2.5 GB/s ALU
// chain), and the second tripped the known -O1/RVV/pthread codegen bug and
// aborted. This version is one thread, 512-byte e8,m4 loads, four independent
// accumulators, and no pthreads; parallelism comes from running the binary
// several times at once.
//
// Why it matters: Q4_0 weights are 1.265 GB and prefill streams them once, so
// 200 tok/s at 10K (51.2 s total) needs 24.7 GB/s sustained.
#include <riscv_vector.h>
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

static int8_t g_sink[512];

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int cpu = argc > 1 ? atoi(argv[1]) : 8;
    const int64_t mb = argc > 2 ? atoll(argv[2]) : 1024;
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof(s), &s) != 0) { printf("cpu %d bind failed\n", cpu); return 1; }

    const int64_t bytes = mb << 20;
    uint8_t * src = (uint8_t *) aligned_alloc(2 << 20, (size_t) bytes);
    if (!src) { printf("alloc failed\n"); return 1; }
    for (int64_t i = 0; i < bytes; i += 4096) src[i] = (uint8_t) i;

    const int64_t nchunk = bytes / 2048;
    vint8m4_t a0 = __riscv_vmv_v_x_i8m4(0, 512), a1 = a0, a2 = a0, a3 = a0;
    const double t0 = now_s();
    for (int64_t c = 0; c < nchunk; ++c) {
        const int8_t * q = (const int8_t *) (src + c * 2048);
        a0 = __riscv_vadd_vv_i8m4(a0, __riscv_vle8_v_i8m4(q + 0 * 512, 512), 512);
        a1 = __riscv_vadd_vv_i8m4(a1, __riscv_vle8_v_i8m4(q + 1 * 512, 512), 512);
        a2 = __riscv_vadd_vv_i8m4(a2, __riscv_vle8_v_i8m4(q + 2 * 512, 512), 512);
        a3 = __riscv_vadd_vv_i8m4(a3, __riscv_vle8_v_i8m4(q + 3 * 512, 512), 512);
    }
    const double dt = now_s() - t0;
    a0 = __riscv_vadd_vv_i8m4(a0, a1, 512);
    a2 = __riscv_vadd_vv_i8m4(a2, a3, 512);
    a0 = __riscv_vadd_vv_i8m4(a0, a2, 512);
    __riscv_vse8_v_i8m4(g_sink, a0, 512);
    printf("cpu %d: %.1f GB/s  (%lld MB)\n", cpu, (double) bytes / dt / 1e9, (long long) mb);
    return 0;
}
