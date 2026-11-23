// X100 lane mapping for vmadot.
//
// On A100 (VLEN=1024, vlenb=128) one vmadot computes 512 MACs: 32 e32 lanes,
// lane = row*8 + col, and lane (r,c) sums A[r*16+d]*B[c*16+d] over d = 0..15.
// X100 has vlenb=32, so a whole register is 32 bytes and an m1 accumulator has
// 8 lanes -- the same kernel produces 0 of 256 correct lanes there, so the
// geometry has to be re-derived instead of guessed.
//
// Method: A all ones, B all zeros except one byte j. Each lane's accumulator then
// reports whether B byte j feeds it. Repeat for every byte, then swap roles to get
// the A side. That yields the exact bipartite map without assuming anything.
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

static void one_dot(int32_t * out, const int8_t * A, const int8_t * B) {
    asm volatile(
        "vsetvli t0, x0, e32, m1\n\t"
        "vl1re32.v v0, (%[A])\n\t"
        "vl8re32.v v8, (%[B])\n\t"
        "vxor.vv v16, v16, v16\n\t"
        "vmadot v16, v0, v8\n\t"
        "mv     t1, %[O]\n\t"
        "vse32.v v16, (t1)\n\t"
        :
        : [A] "r"(A), [B] "r"(B), [O] "r"(out)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "vl", "memory");
}

int main() {
    if (bind_x100()) { printf("bind failed\n"); return 1; }
    const long vb = vlenb();
    const long lanes = vb / 4;
    printf("vlenb=%ld lanes=%ld cpu=%d\n", vb, lanes, cur_cpu());

    // vl8re32 reads 8 whole registers = 8*vb bytes; vl1re32 reads one = vb bytes.
    static int8_t A[1024], B[1024];
    static int32_t out[64];

    memset(A, 1, sizeof A); memset(B, 1, sizeof B);
    one_dot(out, A, B);
    printf("all-ones: ");
    for (long i = 0; i < lanes && i < 16; ++i) printf("%d ", out[i]);
    printf("\n");

    // B side: which B byte feeds which lane
    printf("\nB byte -> lanes (B=%ld bytes read)\n", vb * 8);
    for (int j = 0; j < (int) (vb * 8) && j < 256; ++j) {
        memset(B, 0, sizeof B); B[j] = 1;
        one_dot(out, A, B);
        int n = 0; long sum = 0;
        for (long i = 0; i < lanes; ++i) { if (out[i]) { ++n; sum += out[i]; } }
        if (n) {
            printf("  B[%3d] ->", j);
            for (long i = 0; i < lanes; ++i) if (out[i]) printf(" L%ld(%d)", i, out[i]);
            printf("\n");
        }
    }

    // A side
    memset(B, 1, sizeof B);
    printf("\nA byte -> lanes (A=%ld bytes read)\n", vb);
    for (int i2 = 0; i2 < (int) vb; ++i2) {
        memset(A, 0, sizeof A); A[i2] = 1;
        one_dot(out, A, B);
        int n = 0;
        for (long i = 0; i < lanes; ++i) if (out[i]) ++n;
        if (n) {
            printf("  A[%2d] ->", i2);
            for (long i = 0; i < lanes; ++i) if (out[i]) printf(" L%ld(%d)", i, out[i]);
            printf("\n");
        }
    }
    return 0;
}
