// Which cluster can actually run the int8 dot kernels, and how fast?
// X100 (cpu 0-7) and A100 (cpu 8-15) differ in VLEN, and every geometry in
// mega3 assumes 32 e32 lanes per m1 register.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <signal.h>
#include <setjmp.h>

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

static long vlenb() {
    long v = 0;
    asm volatile("csrr %0, vlenb" : "=r"(v));
    return v;
}

static int bind_x100() {
    cpu_set_t s; CPU_ZERO(&s);
    for (int i = 0; i < 8; ++i) CPU_SET(i, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}
static int bind_a100() {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s);
    for (int i = 8; i < 16; ++i) CPU_SET(i, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static sigjmp_buf g_jb;
static volatile sig_atomic_t g_sig;
static void on_sig(int s) { g_sig = s; siglongjmp(g_jb, 1); }

static void gemm_i8d_4x64(const int8_t * A, const int8_t * B, int32_t * C, int64_t ksteps) {
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
        "mv     t3, %[KS]          \n\t"
        "I8L%=:                    \n\t"
        "vl1re32.v v0, (%[A])       \n\t"
        "vl8re32.v v8, (%[B])       \n\t"
        "addi   %[A], %[A], 64      \n\t"
        "addi   %[B], %[B], 1024    \n\t"
        "vmadot v16, v0, v8         \n\t"
        "vmadot v18, v0, v9         \n\t"
        "vmadot v20, v0, v10        \n\t"
        "vmadot v22, v0, v11        \n\t"
        "vmadot v24, v0, v12        \n\t"
        "vmadot v26, v0, v13        \n\t"
        "vmadot v28, v0, v14        \n\t"
        "vmadot v30, v0, v15        \n\t"
        "addi   t3, t3, -1          \n\t"
        "bgtz   t3, I8L%=           \n\t"
        "vse32.v v16, (%[C0])       \n\t"
        "vse32.v v18, (%[C1])       \n\t"
        "vse32.v v20, (%[C2])       \n\t"
        "vse32.v v22, (%[C3])       \n\t"
        "vse32.v v24, (%[C4])       \n\t"
        "vse32.v v26, (%[C5])       \n\t"
        "vse32.v v28, (%[C6])       \n\t"
        "vse32.v v30, (%[C7])       \n\t"
        : [A] "+r"(A), [B] "+r"(B)
        : [KS] "r"(ksteps),
          [C0] "r"(C), [C1] "r"(C + 32), [C2] "r"(C + 64), [C3] "r"(C + 96),
          [C4] "r"(C + 128), [C5] "r"(C + 160), [C6] "r"(C + 192), [C7] "r"(C + 224)
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// A: 4 rows x 16 dims per k-step, row-major inside the 16-byte step.
// B: 64 columns x 16 dims per k-step, dim-major inside the 1024-byte block.
// lane r*8+c of the 32-lane accumulator = row r, column c (kernel's own map).
static int run_case(const char * tag, int ks) {
    alignas(64) static int8_t A[64 * 64];
    alignas(64) static int8_t B[64 * 1024];
    alignas(64) static int32_t C[256];
    for (int t = 0; t < ks; ++t)
        for (int r = 0; r < 4; ++r)
            for (int d = 0; d < 16; ++d)
                A[t * 64 + r * 16 + d] = (int8_t)(((t * 7 + r * 13 + d * 3) % 21) - 10);
    for (int t = 0; t < ks; ++t)
        for (int j = 0; j < 8; ++j)
            for (int d = 0; d < 128; ++d)
                B[t * 1024 + j * 128 + d] = (int8_t)(((t * 5 + j * 11 + d * 7) % 19) - 9);

    // scalar reference for the kernel's documented lane map
    static int32_t ref[256];
    for (int j = 0; j < 8; ++j)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c) {
                const int col = j * 8 + c;
                int64_t s = 0;
                for (int t = 0; t < ks; ++t)
                    for (int d = 0; d < 16; ++d)
                        s += (int)A[t * 64 + r * 16 + d] * (int)B[t * 1024 + (col / 8) * 128 + d * 8 + (col % 8)];
                ref[j * 32 + r * 8 + c] = (int32_t) s;
            }

    g_sig = 0;
    if (sigsetjmp(g_jb, 1)) {
        printf("%-6s vlenb=%-4ld ILLEGAL/FAULT signal %d -> int8 dot kernel unusable\n",
               tag, vlenb(), (int) g_sig);
        return 2;
    }
    gemm_i8d_4x64(A, B, C, ks);
    // partial results are as bad as wrong ones on a narrower VLEN: report how
    // many of the 256 accumulator slots the hardware actually produced
    int match = 0, nonzero = 0;
    for (int i = 0; i < 256; ++i) { if (C[i] == ref[i]) match++; if (C[i]) nonzero++; }
    printf("%-6s vlenb=%-4ld lanes/m1=%-3ld  exact %3d/256  nonzero %3d/256\n",
           tag, vlenb(), vlenb() / 4, match, nonzero);

    const int reps = 200;
    const double t0 = now_s();
    for (int i = 0; i < reps; ++i) gemm_i8d_4x64(A, B, C, ks);
    const double dt = now_s() - t0;
    // 4 rows x 64 cols x 16*ks MACs each, 512 per vmadot
    const double macs = (double) reps * 4 * 64 * 16 * ks;
    printf("%-6s %6.2f GMAC/s single core (%.3f s for %d reps)\n",
           tag, macs / dt / 1e9, dt, reps);
    return match == 256;
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "x100";
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sig; sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr); sigaction(SIGBUS, &sa, nullptr);

    printf("start cpu=%d vlenb=%ld\n", cur_cpu(), vlenb());
    const int rc = strcmp(mode, "a100") ? bind_x100() : bind_a100();
    printf("bind(%s) rc=%d -> cpu=%d vlenb=%ld\n", mode, rc, cur_cpu(), vlenb());
    if (rc) { printf("bind failed, cannot test %s\n", mode); return 1; }
    run_case(strcmp(mode, "a100") ? "X100" : "A100", 64);
    return 0;
}
