// Lane map of `vmadot` (int8 x int8) with both operands one register (128 B).
//
// Earlier probes showed vector FP ops costing ~8-30 cycles while vmadotsu runs
// at 1 cycle, which is what a 1024-bit register file behind a 128-bit FP
// datapath looks like. If so, only the dot unit is full width, and attention has
// to be rebuilt on int8 dots -- so its exact lane map matters for both QK and PV.
//
// Hypothesis from micro9/micro10: A is 4 rows x 16 int8 at a[row*16+d]; B is
// 16 k x 8 cols with byte k*8+c; output lane = row*8 + col.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

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

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) { printf("bind failed\n"); return 1; }

    static uint8_t a[128], b[128];
    static int32_t o[32];

    memset(a, 0, sizeof a); memset(b, 0, sizeof b); memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("zero: all zero = %s\n", o[0] == 0 ? "yes" : "no");

    memset(a, 1, sizeof a); memset(b, 1, sizeof b); memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("all ones: lane0 %d lane1 %d lane31 %d\n", o[0], o[1], o[31]);

    // A = one byte at (row, dim); B = one byte at (dim, col): lane (row,col) = 1
    printf("\nA[row*16+d]=1 with B[k*8+c]=1 (matching dim): expect lane row*8+c == 1\n");
    int bad = 0;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 8; ++col) {
            for (int d = 0; d < 16; d += 5) {
                memset(a, 0, sizeof a); memset(b, 0, sizeof b);
                a[row * 16 + d] = 1;
                b[d * 8 + col] = 1;
                memset(o, 0, sizeof o);
                dot8(a, b, o);
                const int lane = row * 8 + col;
                if (o[lane] != 1) {
                    printf("  MISMATCH row%d col%d d%d: lane %d = %d\n", row, col, d, lane, o[lane]);
                    ++bad;
                }
            }
        }
    printf("  hypothesis holds for %s\n", bad == 0 ? "every (row,col,dim) tested" : "NOT all");

    // Confirm no cross-talk and that 16 is the k depth
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    for (int d = 0; d < 16; ++d) { a[0 * 16 + d] = 1; b[d * 8 + 0] = 1; }
    memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("\nfull 16-dim dot on row0/col0: lane0 = %d (expect 16), lane1 = %d (expect 0)\n", o[0], o[1]);

    // signedness: negative x negative should be positive
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    a[0] = (uint8_t) (int8_t) -3;
    b[0] = (uint8_t) (int8_t) -5;
    memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("(-3)*(-5): lane0 = %d (expect 15 if signed)\n", o[0]);

    // saturation check: 127*127*16 would be 258064
    memset(a, 127, sizeof a);
    memset(b, 127, sizeof b);
    memset(o, 0, sizeof o);
    dot8(a, b, o);
    printf("127*127*16: lane0 = %d (expect 258064, saturate, or wrap)\n", o[0]);
    return 0;
}
