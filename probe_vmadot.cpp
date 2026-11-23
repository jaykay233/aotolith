// What does one vmadot actually compute at a given VLEN?
// Feed all-ones operands: each accumulator lane then reports how many MACs it
// performed, which is the whole question for whether the 256-bit X100 cluster is
// worth porting the dot kernels to.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

static int cur_cpu() {
    cpu_set_t s; CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof(s), &s)) return -1;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &s)) return i;
    return -1;
}
static long vlenb() { long v = 0; asm volatile("csrr %0, vlenb" : "=r"(v)); return v; }

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

// One vmadot into v16 from v0 (A, one whole register) and v8..v15 (B, 8 whole
// registers), then dump v16's lanes. Everything outside vmadot is plain RVV so
// the operands really land where the kernel puts them.
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
    const int rc = strcmp(mode, "a100") ? bind_x100() : bind_a100();
    const long vb = vlenb();
    const long lanes = vb / 4;
    printf("[%s] cpu=%d bind_rc=%d vlenb=%ld lanes(e32,m1)=%ld\n", mode, cur_cpu(), rc, vb, lanes);
    if (rc) return 1;

    static int8_t A[4096], B[8192];
    static int32_t out[256];
    memset(A, 1, sizeof(A));    // every A byte 1
    memset(B, 1, sizeof(B));    // every B byte 1
    memset(out, 0, sizeof(out));
    one_dot(out, A, B);
    printf("  all-ones per-lane MAC count:");
    for (long i = 0; i < lanes && i < 32; ++i) printf(" %d", out[i]);
    long tot = 0;
    for (long i = 0; i < lanes && i < 32; ++i) tot += out[i];
    printf("\n  MACs per vmadot at this VLEN = %ld  (A reg = %ld bytes, B field = %ld bytes)\n",
           tot, vb, vb * 8);

    // which B byte feeds lane 0: single hot input
    printf("  lane0 response to a single hot B byte (index: count):");
    for (int j = 0; j < 64; ++j) {
        memset(B, 0, sizeof(B)); B[j] = 1;
        memset(out, 0, sizeof(out));
        one_dot(out, A, B);
        if (out[0]) printf(" %d:%d", j, out[0]);
    }
    printf("\n");

    // peak rate: back-to-back dots with the registers held
    printf("  lane0..3 on single-hot A byte:");
    memset(B, 1, sizeof(B));
    for (int j = 0; j < 32; ++j) {
        memset(A, 0, sizeof(A)); A[j] = 1;
        memset(out, 0, sizeof(out));
        one_dot(out, A, B);
        if (out[0] || out[1]) printf(" %d:%d/%d", j, out[0], out[1]);
    }
    printf("\n");
    return 0;
}
