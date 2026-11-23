// QK with Q held in registers across position blocks: the honest version.
//
// The previous attempt reported 186 GMAC/s but only did a quarter of the work
// (4 accumulators = 32 columns, 4 of 8 k-steps). Full coverage needs:
//   B   8 registers, one 8-aligned group for a 1024-byte load (64 positions)
//   acc 8 registers, must be even-numbered -> spans v16..v30 even
//   Q   8 registers, one 64-byte slice per k-step
// B takes v8..v15 and acc takes the even slots v16..v30, so Q has to live in
// v0..v7 -- including v0, which is the mask register. This probe checks whether
// v0 is usable as a plain source operand for vmadot (writing it crashed earlier,
// reading may be fine). If not, the fallback is to reload Q every block and keep
// the 122 GMAC/s loop rather than 186.
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// One call: 4 rows x 64 positions x 128 k = 32768 MAC, from Q resident in v0..v7.
static void qk_q0(const int8_t * A, const int8_t * B, int32_t * out) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vl1re32.v v0,  (%[A])            \n\t"
        "vl1re32.v v1,  (%[A1])           \n\t"
        "vl1re32.v v2,  (%[A2])           \n\t"
        "vl1re32.v v3,  (%[A3])           \n\t"
        "vl1re32.v v4,  (%[A4])           \n\t"
        "vl1re32.v v5,  (%[A5])           \n\t"
        "vl1re32.v v6,  (%[A6])           \n\t"
        "vl1re32.v v7,  (%[A7])           \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "vxor.vv v24, v24, v24          \n\t"
        "vxor.vv v26, v26, v26          \n\t"
        "vxor.vv v28, v28, v28          \n\t"
        "vxor.vv v30, v30, v30          \n\t"
        // k-step 0
        "vl8re32.v v8,  (%[B0])           \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v18, v0, v9            \n\t"
        "vmadot  v20, v0, v10           \n\t"
        "vmadot  v22, v0, v11           \n\t"
        "vmadot  v24, v0, v12           \n\t"
        "vmadot  v26, v0, v13           \n\t"
        "vmadot  v28, v0, v14           \n\t"
        "vmadot  v30, v0, v15           \n\t"
        // k-step 1
        "vl8re32.v v8,  (%[B1])           \n\t"
        "vmadot  v16, v1, v8            \n\t"
        "vmadot  v18, v1, v9            \n\t"
        "vmadot  v20, v1, v10           \n\t"
        "vmadot  v22, v1, v11           \n\t"
        "vmadot  v24, v1, v12           \n\t"
        "vmadot  v26, v1, v13           \n\t"
        "vmadot  v28, v1, v14           \n\t"
        "vmadot  v30, v1, v15           \n\t"
        // k-step 2
        "vl8re32.v v8,  (%[B2])           \n\t"
        "vmadot  v16, v2, v8            \n\t"
        "vmadot  v18, v2, v9            \n\t"
        "vmadot  v20, v2, v10           \n\t"
        "vmadot  v22, v2, v11           \n\t"
        "vmadot  v24, v2, v12           \n\t"
        "vmadot  v26, v2, v13           \n\t"
        "vmadot  v28, v2, v14           \n\t"
        "vmadot  v30, v2, v15           \n\t"
        // k-step 3
        "vl8re32.v v8,  (%[B3])           \n\t"
        "vmadot  v16, v3, v8            \n\t"
        "vmadot  v18, v3, v9            \n\t"
        "vmadot  v20, v3, v10           \n\t"
        "vmadot  v22, v3, v11           \n\t"
        "vmadot  v24, v3, v12           \n\t"
        "vmadot  v26, v3, v13           \n\t"
        "vmadot  v28, v3, v14           \n\t"
        "vmadot  v30, v3, v15           \n\t"
        // k-step 4
        "vl8re32.v v8,  (%[B4])           \n\t"
        "vmadot  v16, v4, v8            \n\t"
        "vmadot  v18, v4, v9            \n\t"
        "vmadot  v20, v4, v10           \n\t"
        "vmadot  v22, v4, v11           \n\t"
        "vmadot  v24, v4, v12           \n\t"
        "vmadot  v26, v4, v13           \n\t"
        "vmadot  v28, v4, v14           \n\t"
        "vmadot  v30, v4, v15           \n\t"
        // k-step 5
        "vl8re32.v v8,  (%[B5])           \n\t"
        "vmadot  v16, v5, v8            \n\t"
        "vmadot  v18, v5, v9            \n\t"
        "vmadot  v20, v5, v10           \n\t"
        "vmadot  v22, v5, v11           \n\t"
        "vmadot  v24, v5, v12           \n\t"
        "vmadot  v26, v5, v13           \n\t"
        "vmadot  v28, v5, v14           \n\t"
        "vmadot  v30, v5, v15           \n\t"
        // k-step 6
        "vl8re32.v v8,  (%[B6])           \n\t"
        "vmadot  v16, v6, v8            \n\t"
        "vmadot  v18, v6, v9            \n\t"
        "vmadot  v20, v6, v10           \n\t"
        "vmadot  v22, v6, v11           \n\t"
        "vmadot  v24, v6, v12           \n\t"
        "vmadot  v26, v6, v13           \n\t"
        "vmadot  v28, v6, v14           \n\t"
        "vmadot  v30, v6, v15           \n\t"
        // k-step 7
        "vl8re32.v v8,  (%[B7])           \n\t"
        "vmadot  v16, v7, v8            \n\t"
        "vmadot  v18, v7, v9            \n\t"
        "vmadot  v20, v7, v10           \n\t"
        "vmadot  v22, v7, v11           \n\t"
        "vmadot  v24, v7, v12           \n\t"
        "vmadot  v26, v7, v13           \n\t"
        "vmadot  v28, v7, v14           \n\t"
        "vmadot  v30, v7, v15           \n\t"
        "vse32.v v16, (%[O0])           \n\t"
        "vse32.v v18, (%[O1])           \n\t"
        "vse32.v v20, (%[O2])           \n\t"
        "vse32.v v22, (%[O3])           \n\t"
        "vse32.v v24, (%[O4])           \n\t"
        "vse32.v v26, (%[O5])           \n\t"
        "vse32.v v28, (%[O6])           \n\t"
        "vse32.v v30, (%[O7])           \n\t"
        :
        : [A] "r"(A), [A1] "r"(A + 64), [A2] "r"(A + 128), [A3] "r"(A + 192),
          [A4] "r"(A + 256), [A5] "r"(A + 320), [A6] "r"(A + 384), [A7] "r"(A + 448),
          [B0] "r"(B), [B1] "r"(B + 1024), [B2] "r"(B + 2048), [B3] "r"(B + 3072),
          [B4] "r"(B + 4096), [B5] "r"(B + 5120), [B6] "r"(B + 6144), [B7] "r"(B + 7168),
          [O0] "r"(out), [O1] "r"(out + 32), [O2] "r"(out + 64), [O3] "r"(out + 96),
          [O4] "r"(out + 128), [O5] "r"(out + 160), [O6] "r"(out + 192), [O7] "r"(out + 224)
        : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v18", "v20", "v22", "v24", "v26", "v28", "v30", "memory");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    static int8_t q[8 * 64];
    static int8_t k[8 * 1024];
    static int32_t out[8 * 32];
    for (int i = 0; i < 8 * 64; ++i) q[i] = (int8_t) ((i * 37) % 61 - 30);
    for (int i = 0; i < 8 * 1024; ++i) k[i] = (int8_t) ((i * 53) % 61 - 30);

    const int64_t macs = 4LL * 64 * 128;
    const int reps = 300000;
    qk_q0(q, k, out);
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_s();
        for (int i = 0; i < reps; ++i) qk_q0(q, k, out);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    printf("QK Q-in-v0..v7, 8 acc, 8 k-steps: %6.1f GMAC/s  %.1f ns/block\n",
           (double) macs * reps / best / 1e9, best / reps * 1e9);

    int wrong = 0;
    for (int row = 0; row < 4; ++row)
        for (int j = 0; j < 8; ++j)
            for (int c = 0; c < 8; ++c) {
                const int pos = j * 8 + c;
                long long ref = 0;
                for (int ks = 0; ks < 8; ++ks)
                    for (int d = 0; d < 16; ++d)
                        ref += (long long) q[ks * 64 + row * 16 + d]
                             * (long long) k[ks * 1024 + pos * 16 + d];
                if ((long long) out[j * 32 + row * 8 + c] != ref) ++wrong;
            }
    printf("correctness: %d / 256 wrong  -> %s\n", wrong, wrong ? "BAD" : "BIT EXACT");
    return 0;
}
