// Is qk4's K load the reason int8 attention runs at 8.3 GMAC/s?
//
// qk4 (the A100 QK inner loop used by both prefill attention and this tree's
// int8 decode path) loads 1 KB of K per step with a single `vl8re32.v`, then
// issues eight `vmadot`. Measured through the production path that loop runs at
// 8.3 GMAC/s aggregate -- about 1.0 GMAC/s per core, while a vmadot is 512 MAC,
// so the dots cannot be the limit. This probe holds the dot count and the byte
// count fixed and changes only how the K bytes arrive:
//
//   variant 0  no K load at all, registers held      (pure dot throughput)
//   variant 1  vl1re32.v + vl8re32.v                 (what qk4 does today)
//   variant 2  vl1re32.v + 8x vle32.v immediate      (same bytes, plain loads)
//
// Same operands, same eight vmadot, same 8 KiB of K per call. If variant 1 is
// far behind variant 2 the whole-register load is trap-emulated on this core,
// which would be the same class of trap as the strided `vlse32` gather already
// documented, and attention would have a ~10x on the table.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <time.h>

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static long vlenb() { long v = 0; asm volatile("csrr %0, vlenb" : "=r"(v)); return v; }

static int bind_a100() {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s);
    for (int i = 8; i < 16; ++i) CPU_SET(i, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

// 8 iterations of the qk4 shape: A advances 64 B, K advances 1024 B.
#define BODY_LOAD1 \
    "vl1re32.v v0, (%[Q])           \n\t" \
    "vl8re32.v v8, (%[K])           \n\t" \
    "addi %[Q], %[Q], 64            \n\t" \
    "addi %[K], %[K], 1024          \n\t"
#define BODY_LOAD2 \
    "vl1re32.v v0, (%[Q])           \n\t" \
    "mv t2, %[K]                    \n\t" \
    "mv t4, %[K]                    \n\t" \
    "addi t4, t4, 512               \n\t" \
    "vle32.v v8, (t2)               \n\t" \
    "vle32.v v12, (t4)              \n\t" \
    "addi t2, t2, 128               \n\t" \
    "addi t4, t4, 128               \n\t" \
    "vle32.v v9, (t2)               \n\t" \
    "vle32.v v13, (t4)              \n\t" \
    "addi t2, t2, 128               \n\t" \
    "addi t4, t4, 128               \n\t" \
    "vle32.v v10, (t2)              \n\t" \
    "vle32.v v14, (t4)              \n\t" \
    "addi t2, t2, 128               \n\t" \
    "addi t4, t4, 128               \n\t" \
    "vle32.v v11, (t2)              \n\t" \
    "vle32.v v15, (t4)              \n\t" \
    "addi %[Q], %[Q], 64            \n\t" \
    "addi %[K], %[K], 1024          \n\t"
#define BODY_NOLOAD \
    "vl1re32.v v0, (%[Q])           \n\t" \
    "addi %[Q], %[Q], 64            \n\t"
#define DOTS \
    "vmadot  v16, v0, v8            \n\t" \
    "vmadot  v18, v0, v9            \n\t" \
    "vmadot  v20, v0, v10           \n\t" \
    "vmadot  v22, v0, v11           \n\t" \
    "vmadot  v24, v0, v12           \n\t" \
    "vmadot  v26, v0, v13           \n\t" \
    "vmadot  v28, v0, v14           \n\t" \
    "vmadot  v30, v0, v15           \n\t"

static void qk_step(int variant, const int8_t * Q, const int8_t * K, int32_t * out) {
    const int8_t * q = Q;
    const int8_t * k = K;
    if (variant == 0) {
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
            "mv t3, %[N]               \n\t"
            "1:                        \n\t"
            BODY_NOLOAD
            DOTS
            "addi t3, t3, -1           \n\t"
            "bgtz t3, 1b               \n\t"
            "vse32.v v16, (%[O])       \n\t"
            : [Q] "+r"(q), [K] "+r"(k)
            : [N] "r"(8), [O] "r"(out)
            : "t0", "t2", "t3", "t4", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
    } else if (variant == 1) {
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
            "mv t3, %[N]               \n\t"
            "1:                        \n\t"
            BODY_LOAD1
            DOTS
            "addi t3, t3, -1           \n\t"
            "bgtz t3, 1b               \n\t"
            "vse32.v v16, (%[O])       \n\t"
            : [Q] "+r"(q), [K] "+r"(k)
            : [N] "r"(8), [O] "r"(out)
            : "t0", "t2", "t3", "t4", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
    } else {
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
            "mv t3, %[N]               \n\t"
            "1:                        \n\t"
            BODY_LOAD2
            DOTS
            "addi t3, t3, -1           \n\t"
            "bgtz t3, 1b               \n\t"
            "vse32.v v16, (%[O])       \n\t"
            : [Q] "+r"(q), [K] "+r"(k)
            : [N] "r"(8), [O] "r"(out)
            : "t0", "t2", "t3", "t4", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
    }
}

int main() {
    if (bind_a100()) { printf("bind failed\n"); return 1; }
    printf("vlenb=%ld (lanes e32m1 = %ld)\n", vlenb(), vlenb() / 4);

    // One qk_step consumes 512 B of A and 8 KiB of K. 64 KiB K and 4 KiB A keep
    // both inside L1 across a whole run of steps, so this measures instruction
    // cost rather than DRAM.
    static int8_t A[4096] __attribute__((aligned(1024)));
    static int8_t K[65536] __attribute__((aligned(1024)));
    static int32_t out[64];
    memset(A, 1, sizeof(A));
    memset(K, 1, sizeof(K));

    const char * names[3] = { "no K load (dot ceiling)", "vl8re32.v (qk4 today)", "8x vle32.v immediate" };
    for (int variant = 0; variant < 3; ++variant) {
        // warm up
        for (int i = 0; i < 200; ++i) qk_step(variant, A, K, out);
        const long reps = 200000;
        const double t0 = now_s();
        for (long i = 0; i < reps; ++i) qk_step(variant, A, K, out);
        const double dt = now_s() - t0;
        // one qk_step = 8 steps x 8 vmadot x 512 MAC
        const double mac = (double) reps * 8 * 8 * 512;
        printf("  %-26s %8.1f ns/call  %7.1f GMAC/s  (out[0]=%d)\n",
               names[variant], dt / reps * 1e9, mac / dt / 1e9, out[0]);
    }
    return 0;
}
