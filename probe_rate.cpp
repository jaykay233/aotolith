// Comparable int8 dot rate per core on both clusters. The instruction is the
// same; only VLEN differs, so measure MAC/instr at runtime and time a loop of
// back-to-back dots.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

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

static int bind_c(const char * mode) {
    cpu_set_t s; CPU_ZERO(&s);
    if (!strcmp(mode, "a100")) {
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        for (int i = 8; i < 16; ++i) CPU_SET(i, &s);
    } else {
        for (int i = 0; i < 8; ++i) CPU_SET(i, &s);
    }
    return sched_setaffinity(0, sizeof(s), &s);
}

// 8 dots per iteration, same shape as gemm_i8d_4x64's inner loop.
#define DOTS_PER_ITER 8
static void dot_loop(int64_t iters, const int8_t * A, const int8_t * B, int32_t * C) {
    asm volatile(
        "vsetvli t0, x0, e32, m1      \n\t"
        "vxor.vv v16, v16, v16        \n\t"
        "vxor.vv v18, v18, v18        \n\t"
        "vxor.vv v20, v20, v20        \n\t"
        "vxor.vv v22, v22, v22        \n\t"
        "vxor.vv v24, v24, v24        \n\t"
        "vxor.vv v26, v26, v26        \n\t"
        "vxor.vv v28, v28, v28        \n\t"
        "vxor.vv v30, v30, v30        \n\t"
        "mv     t3, %[N]              \n\t"
        "L%=:                         \n\t"
        "vl1re32.v v0, (%[A])         \n\t"
        "vl8re32.v v8, (%[B])         \n\t"
        "vmadot v16, v0, v8           \n\t"
        "vmadot v18, v0, v9           \n\t"
        "vmadot v20, v0, v10          \n\t"
        "vmadot v22, v0, v11          \n\t"
        "vmadot v24, v0, v12          \n\t"
        "vmadot v26, v0, v13          \n\t"
        "vmadot v28, v0, v14          \n\t"
        "vmadot v30, v0, v15          \n\t"
        "addi   t3, t3, -1            \n\t"
        "bgtz   t3, L%=               \n\t"
        "mv     t1, %[C]              \n\t"
        "vse32.v v16, (t1)            \n\t"
        : [A] "+r"(A), [B] "+r"(B)
        : [N] "r"(iters), [C] "r"(C)
        : "t0", "t1", "t3",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

static void one_dot(int32_t * out, const int8_t * A, const int8_t * B) {
    asm volatile(
        "vsetvli t0, x0, e32, m1      \n\t"
        "vl1re32.v v0, (%[A])         \n\t"
        "vl8re32.v v8, (%[B])         \n\t"
        "vxor.vv v16, v16, v16        \n\t"
        "vmadot v16, v0, v8           \n\t"
        "mv     t1, %[O]              \n\t"
        "vse32.v v16, (t1)            \n\t"
        :
        : [A] "r"(A), [B] "r"(B), [O] "r"(out)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "vl", "memory");
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "x100";
    if (bind_c(mode)) { printf("bind failed\n"); return 1; }
    const long vb = vlenb();
    static int8_t A[4096], B[8192];
    static int32_t out[64];
    memset(A, 1, sizeof(A)); memset(B, 1, sizeof(B));
    memset(out, 0, sizeof(out));
    one_dot(out, A, B);
    long mpi = 0;
    for (long i = 0; i < vb / 4 && i < 32; ++i) mpi += out[i];

    const int64_t iters = 2'000'000;
    dot_loop(iters / 10, A, B, out);            // warm up
    const double t0 = now_s();
    dot_loop(iters, A, B, out);
    const double dt = now_s() - t0;
    const double instr = (double) iters * DOTS_PER_ITER;
    printf("%-5s cpu=%d vlenb=%-4ld MAC/dot=%-4ld  %.0f Mdot/s  %.1f GMAC/s/core\n",
           strcmp(mode, "a100") ? "X100" : "A100", cur_cpu(), vb, mpi,
           instr / dt / 1e6, instr * (double) mpi / dt / 1e9);
    return 0;
}
