// Definitive operand-map experiment for `vmadotsu ... i4` and the vendor's
// vnpack4/vupack shuffles. Nothing here is guesswork: every probe zeroes both
// operands, sets exactly one nibble or one byte, and reads all 32 accumulator
// lanes back.
//
// Ground truth to match, from gemm_kernel_i8i4_m4 in ime2_kernels.cpp:
//   vl1r.v v0, (A)        A = 128 B = 4 rows x 32 int8
//   vl4r.v v4, (B)        B = 512 B = 32 cols x 16 B of packed int4
//   vsrl.vi  v1, v0, 4    high nibbles of A
//   vnpack4.vv v12, v0, v1, 3
//   vupack.vv  v2, v12, v12, 2
//   vmadotsu v16, v3, v4, i4      (high 4 bits of the weights)
//   vmadotu  v16, v2, v4, i4      (low 4 bits, after vsll.vi v16, 4)
// so v2 and v3 together are the two halves of A in the form the dot unit wants,
// and the 4 accumulators v16/v18/v20/v22 are four rows x 32 columns.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

// A: 128 B in one register. B: 128 B in one register (8 columns x 16 B).
// Result: 32 int32 lanes.
static void dot_i4(const uint8_t * a128, const uint8_t * b128, int32_t * o, int op) {
    if (op == 0) {
        asm volatile(
            "vsetvli t0, x0, e8, m1     \n\t"
            "vl1r.v  v0, (%[A])          \n\t"
            "vl1r.v  v4, (%[B])          \n\t"
            "vsetvli t0, x0, e32, m1     \n\t"
            "vxor.vv v16, v16, v16       \n\t"
            "vmadotsu v16, v0, v4, i4    \n\t"
            "vse32.v v16, (%[O])         \n\t"
            : : [A] "r"(a128), [B] "r"(b128), [O] "r"(o)
            : "t0", "v0", "v4", "v16", "memory");
    } else if (op == 1) {
        // the vendor's A preparation, then dot with the low half
        asm volatile(
            "vsetvli t0, x0, e8, m1     \n\t"
            "vl1r.v  v0, (%[A])          \n\t"
            "vl1r.v  v4, (%[B])          \n\t"
            "vsrl.vi    v1, v0, 4        \n\t"
            "vnpack4.vv v12, v0, v1, 3   \n\t"
            "vupack.vv  v2, v12, v12, 2  \n\t"
            "vsetvli t0, x0, e32, m1     \n\t"
            "vxor.vv v16, v16, v16       \n\t"
            "vmadotsu v16, v3, v4, i4    \n\t"
            "vse32.v v16, (%[O])         \n\t"
            : : [A] "r"(a128), [B] "r"(b128), [O] "r"(o)
            : "t0", "v0", "v1", "v2", "v3", "v4", "v12", "v16", "memory");
    } else if (op == 2) {
        // dump the two registers that vnpack4/vupack produce
        asm volatile(
            "vsetvli t0, x0, e8, m1     \n\t"
            "vl1r.v  v0, (%[A])          \n\t"
            "vsrl.vi    v1, v0, 4        \n\t"
            "vnpack4.vv v12, v0, v1, 3   \n\t"
            "vupack.vv  v2, v12, v12, 2  \n\t"
            "vse8.v  v2, (%[O])          \n\t"
            "vse8.v  v3, (%[O])          \n\t"
            : : [A] "r"(a128), [O] "r"(o)
            : "t0", "v0", "v1", "v2", "v3", "v12", "memory");
    } else {
        // vnpack4 output alone
        asm volatile(
            "vsetvli t0, x0, e8, m1     \n\t"
            "vl1r.v  v0, (%[A])          \n\t"
            "vsrl.vi    v1, v0, 4        \n\t"
            "vnpack4.vv v12, v0, v1, 3   \n\t"
            "vse8.v  v12, (%[O])         \n\t"
            : : [A] "r"(a128), [O] "r"(o)
            : "t0", "v0", "v1", "v12", "memory");
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    uint8_t a[128], b[128];
    int32_t o[32];
    uint8_t raw[256];

    printf("== P0: A=1 all, B=1 nibble all (int4 weights = 1) ==\n");
    memset(a, 1, sizeof a);
    memset(b, 0x11, sizeof b);
    memset(o, 0, sizeof o);
    dot_i4(a, b, o, 0);
    printf("lanes:");
    for (int i = 0; i < 32; ++i) printf(" %d", o[i]);
    printf("\n");

    printf("\n== P1: which B nibble feeds which lane (A all 1) ==\n");
    for (int j = 0; j < 16; ++j) {
        for (int half = 0; half < 2; ++half) {
            memset(a, 1, sizeof a);
            memset(b, 0, sizeof b);
            b[j] = half ? 0x10 : 0x01;
            memset(o, 0, sizeof o);
            dot_i4(a, b, o, 0);
            printf("  B[%2d] %s:", j, half ? "hi" : "lo");
            for (int i = 0; i < 32; ++i) if (o[i]) printf(" %d=%d", i, o[i]);
            printf("\n");
        }
    }

    printf("\n== P2: which A byte feeds which lane (B all nibbles 1) ==\n");
    for (int j = 0; j < 128; ++j) {
        memset(a, 0, sizeof a);
        a[j] = 1;
        memset(b, 0x11, sizeof b);
        memset(o, 0, sizeof o);
        dot_i4(a, b, o, 0);
        int n = 0;
        for (int i = 0; i < 32; ++i) if (o[i]) ++n;
        if (n) {
            printf("  A[%3d] -> %d lanes:", j, n);
            for (int i = 0; i < 32; ++i) if (o[i]) printf(" %d", i);
            printf("\n");
        } else {
            printf("  A[%3d] -> none\n", j);
        }
    }

    printf("\n== P3: A preparation, dump v2 and v3 (128 B each) ==\n");
    for (int i = 0; i < 128; ++i) a[i] = (uint8_t) i;      // distinct byte values
    memset(raw, 0, sizeof raw);
    dot_i4(a, b, (int32_t *) raw, 2);
    printf("  in  A[0..15] :");
    for (int i = 0; i < 16; ++i) printf(" %3d", a[i]);
    printf("\n  v2  [0..15]  :");
    for (int i = 0; i < 16; ++i) printf(" %3d", raw[i]);
    printf("\n  v2  [64..79] :");
    for (int i = 0; i < 16; ++i) printf(" %3d", raw[64 + i]);
    printf("\n");

    memset(raw, 0, sizeof raw);
    dot_i4(a, b, (int32_t *) raw, 3);
    printf("  vnpack4 out [0..15]:");
    for (int i = 0; i < 16; ++i) printf(" %3d", raw[i]);
    printf("\n");

    printf("\n== P4: vendor-form dot after A preparation, A byte j only ==\n");
    for (int j = 0; j < 40; ++j) {
        memset(a, 0, sizeof a);
        a[j] = 1;
        memset(b, 0x11, sizeof b);
        memset(o, 0, sizeof o);
        dot_i4(a, b, o, 1);
        int n = 0;
        for (int i = 0; i < 32; ++i) if (o[i]) ++n;
        printf("  A[%2d] -> %d lanes:", j, n);
        for (int i = 0; i < 32; ++i) if (o[i]) printf(" %d=%d", i, o[i]);
        printf("\n");
    }
    return 0;
}
