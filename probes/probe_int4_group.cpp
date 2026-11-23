// Resolve the weight-operand map of `vmadotsu ... i4` when vs1 is the m4 group
// the vendor kernel actually uses (vl4r.v v4 loads 4 registers = 512 B = 32
// columns x 16 B). The A side is already known: lane = row*8 + col, and A bytes
// [row*16 .. row*16+15] feed that row's 8 lanes.
//
// A single load of one register gave a degenerate map, which says the i4 operand
// is consumed as a register group, so the probe has to load the whole group.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

// a128: 4 rows x 32 int8 (row r at a[r*32]). b512: 32 cols x 16 B.
// Also reports the 4-accumulator form the vendor uses: v16,v18,v20,v22.
static void dot4(const uint8_t * a128, const uint8_t * b512, int32_t * o4x32, int high) {
    if (high) {
        asm volatile(
            "vsetvli t0, x0, e8, m1      \n\t"
            "vl1r.v  v0, (%[A])           \n\t"
            "vl4r.v  v4, (%[B])           \n\t"
            "vsetvli t0, x0, e32, m1      \n\t"
            "vxor.vv v16, v16, v16        \n\t"
            "vxor.vv v18, v18, v18        \n\t"
            "vxor.vv v20, v20, v20        \n\t"
            "vxor.vv v22, v22, v22        \n\t"
            "vmadotsu v16, v0, v4, i4     \n\t"
            "vmadotsu v18, v0, v5, i4     \n\t"
            "vmadotsu v20, v0, v6, i4     \n\t"
            "vmadotsu v22, v0, v7, i4     \n\t"
            "vse32.v v16, (%[O])          \n\t"
            "vse32.v v18, (%[O1])       \n\t"
            "vse32.v v20, (%[O2])         \n\t"
            "vse32.v v22, (%[O3])         \n\t"
            : : [A] "r"(a128), [B] "r"(b512), [O] "r"(o4x32), [O1] "r"(o4x32 + 32), [O2] "r"(o4x32 + 64), [O3] "r"(o4x32 + 96)
            : "t0", "v0", "v4", "v5", "v6", "v7", "v16", "v18", "v20", "v22", "memory");
    } else {
        asm volatile(
            "vsetvli t0, x0, e8, m1      \n\t"
            "vl1r.v  v0, (%[A])           \n\t"
            "vl4r.v  v4, (%[B])           \n\t"
            "vsetvli t0, x0, e32, m1      \n\t"
            "vxor.vv v16, v16, v16        \n\t"
            "vxor.vv v18, v18, v18        \n\t"
            "vxor.vv v20, v20, v20        \n\t"
            "vxor.vv v22, v22, v22        \n\t"
            "vmadotu v16, v0, v4, i4      \n\t"
            "vmadotu v18, v0, v5, i4      \n\t"
            "vmadotu v20, v0, v6, i4      \n\t"
            "vmadotu v22, v0, v7, i4      \n\t"
            "vse32.v v16, (%[O])          \n\t"
            "vse32.v v18, (%[O1])       \n\t"
            "vse32.v v20, (%[O2])         \n\t"
            "vse32.v v22, (%[O3])         \n\t"
            : : [A] "r"(a128), [B] "r"(b512), [O] "r"(o4x32), [O1] "r"(o4x32 + 32), [O2] "r"(o4x32 + 64), [O3] "r"(o4x32 + 96)
            : "t0", "v0", "v4", "v5", "v6", "v7", "v16", "v18", "v20", "v22", "memory");
    }
}

static void show(const char * tag, const int32_t * o) {
    printf("%s", tag);
    for (int r = 0; r < 4; ++r) {
        printf(" | r%d:", r);
        for (int c = 0; c < 8; ++c) printf(" %d", o[r * 32 + c]);
    }
    printf("\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    static uint8_t a[128], b[512];
    static int32_t o[4 * 32];

    printf("== all ones: expect 16 per lane if each lane sums 16 int4 products ==\n");
    memset(a, 1, sizeof a);
    memset(b, 0x11, sizeof b);
    memset(o, 0, sizeof o);
    dot4(a, b, o, 1);
    show("su:", o);
    memset(o, 0, sizeof o);
    dot4(a, b, o, 0);
    show("u :", o);

    printf("\n== one weight nibble set at a time (A all ones) ==\n");
    for (int m = 0; m < 32; ++m) {
        for (int half = 0; half < 2; ++half) {
            memset(a, 1, sizeof a);
            memset(b, 0, sizeof b);
            b[m] = half ? 0x10 : 0x01;
            memset(o, 0, sizeof o);
            dot4(a, b, o, half ? 1 : 0);
            int n = 0;
            for (int k = 0; k < 128; ++k) if (o[k]) ++n;
            printf("  b[%2d] %s -> %2d hits:", m, half ? "hi" : "lo", n);
            for (int k = 0; k < 128 && n <= 8; ++k)
                if (o[k]) printf(" a%d/r%dc%d=%d", k / 32, (k % 32) / 8, (k % 32) % 8, o[k]);
            printf("\n");
        }
    }

    printf("\n== A row separation check: A[r*32+0]=1 for one r ==\n");
    for (int r = 0; r < 4; ++r) {
        memset(a, 0, sizeof a);
        a[r * 32 + 0] = 1;
        memset(b, 0x11, sizeof b);
        memset(o, 0, sizeof o);
        dot4(a, b, o, 1);
        printf("  A[%2d] (row %d, k 0) ->", r * 32, r);
        for (int k = 0; k < 128; ++k) if (o[k]) printf(" a%d/%d", k / 32, (k % 32) % 8);
        printf("\n");
    }
    return 0;
}
