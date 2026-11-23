// How fast is the attention softmax (int16 scores -> int8 weights) on each
// cluster? Only the A100 cluster can run vmadot, but this stage is plain RVV, so
// the idle X100 cores could take it -- worth wiring only if X100 is within ~2x.
//
// Two shapes of the same maths:
//   vl32 : 32 positions per step, m1 (what the A100 path uses today; needs
//          VLEN>=1024 and the slidedown double-store for the 64-byte w8 block)
//   vl8  :  8 positions per step, m1 (VLMAX on X100, one store per block)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <riscv_vector.h>

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
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        for (int i = 8; i < 16; ++i) CPU_SET(i, &s);
    } else {
        for (int i = 0; i < 8; ++i) CPU_SET(i, &s);
    }
    return sched_setaffinity(0, sizeof(s), &s);
}

static inline vfloat32m1_t exp2_poly(vfloat32m1_t t, size_t vl) {
    vint32m1_t ie = __riscv_vfcvt_x_f_v_i32m1(t, vl);
    const vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, __riscv_vfcvt_f_x_v_f32m1(ie, vl), vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t ex = __riscv_vadd_vx_i32m1(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m1(ex, 23, vl);
    return __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_i32m1_f32m1(ex), vl);
}

static const float PSA_SC = 128.0f;

static float row_max_f(const int16_t * rw, int len) {
    int mx = -32768;
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        const size_t vl = __riscv_vsetvl_e16m1(16);
        const vint16m1_t v = __riscv_vle16_v_i16m1(rw + i, vl);
        const vint16m1_t m = __riscv_vredmax_vs_i16m1_i16m1(
            v, __riscv_vmv_s_x_i16m1((int16_t) -32768, 1), vl);
        const int t = (int) __riscv_vmv_x_s_i16m1_i16(m);
        if (t > mx) mx = t;
    }
    for (; i < len; ++i) if (rw[i] > mx) mx = rw[i];
    return (float) mx / PSA_SC;
}

// The exp + int8 quantise core, one step of `vl` positions.
static inline vint8mf4_t sm_step(const int16_t * src, int p, size_t vl, float mx,
                                vfloat32m1_t * acc) {
    __riscv_vsetvl_e32m1(vl);
    const vint32m1_t ivs = __riscv_vsext_vf2_i32m1(__riscv_vle16_v_i16mf2(src + p, vl), vl);
    vfloat32m1_t v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(ivs, vl), 1.0f / PSA_SC, vl);
    v = __riscv_vfsub_vf_f32m1(v, mx, vl);
    v = __riscv_vfmul_vf_f32m1(v, 1.4426950408889634f, vl);
    v = exp2_poly(__riscv_vfmax_vf_f32m1(v, -126.0f, vl), vl);
    *acc = __riscv_vfadd_vv_f32m1(*acc, v, vl);
    const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, 127.0f, vl);
    const vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(q, vl);
    const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(iv, vl);
    return __riscv_vncvt_x_x_w_i8mf4(h16, vl);
}

// 32 positions per step: one store per 16-position block, plus the slidedown
// double-store that keeps 4 rows x 16 positions contiguous for pv4.
static void sm_vl32(const int16_t * rw, int len, float mx, int8_t * wb, int r, float * sump) {
    const size_t vlz = __riscv_vsetvl_e32m1(32);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlz);
    int p = 0;
    for (; p + 32 <= len; p += 32) {
        const vint8mf4_t b8 = sm_step(rw, p, 32, mx, &acc);
        __riscv_vse8_v_i8mf4(wb + (p / 16) * 64 + r * 16, b8, 16);
        __riscv_vse8_v_i8mf4(wb + (p / 16 + 1) * 64 + r * 16, __riscv_vslidedown_vx_i8mf4(b8, 16, 32), 16);
    }
    for (; p < len; p += 16) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) ((len - p < 16) ? (len - p) : 16));
        const vint8mf4_t b8 = sm_step(rw, p, vl, mx, &acc);
        __riscv_vse8_v_i8mf4(wb + (p / 16) * 64 + r * 16, b8, vl);
    }
    *sump = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
        acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlz));
}

// 8 positions per step: VLMAX on X100, and every step lands inside one block.
static void sm_vl8(const int16_t * rw, int len, float mx, int8_t * wb, int r, float * sump) {
    const size_t vlz = __riscv_vsetvl_e32m1(8);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlz);
    int p = 0;
    for (; p + 8 <= len; p += 8) {
        const vint8mf4_t b8 = sm_step(rw, p, 8, mx, &acc);
        __riscv_vse8_v_i8mf4(wb + (p / 16) * 64 + r * 16 + (p % 16), b8, 8);
    }
    for (; p < len; ++p) {
        const float v = exp2f(((float) rw[p] / PSA_SC - mx) * 1.4426950408889634f);
        wb[(p / 16) * 64 + r * 16 + (p % 16)] = (int8_t) (v * 127.0f);
        sump[0] += v;
    }
    sump[0] += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
        acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlz));
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "a100";
    const int len = argc > 2 ? atoi(argv[2]) : 10240;
    if (bind_c(mode)) { printf("bind failed\n"); return 1; }

    static int16_t rw[20000];
    static int8_t wa[20000 * 4 + 256], wb[20000 * 4 + 256];
    for (int i = 0; i < len; ++i) rw[i] = (int16_t) ((i * 37) % 4000 - 2000);
    const float mx = row_max_f(rw, len);

    float s1 = 0, s2 = 0;
    const int reps = 200;
    fprintf(stderr, "[p] mx=%.6g len=%d\n", mx, len); fflush(stderr);
    memset(wa, 0, sizeof wa);
    double t0 = now_s();
    fprintf(stderr, "[p] enter vl32 timing\n"); fflush(stderr);
    for (int i = 0; i < reps; ++i) sm_vl32(rw, len, mx, wa, i % 4, &s1);
    const double d1 = now_s() - t0;
    fprintf(stderr, "[p] vl32 done s1=%.6g\n", s1); fflush(stderr);

    memset(wb, 0, sizeof wb);
    t0 = now_s();
    for (int i = 0; i < reps; ++i) sm_vl8(rw, len, mx, wb, i % 4, &s2);
    const double d2 = now_s() - t0;

    // both shapes must agree on the quantised weights (bit exact)
    memset(wa, 0, sizeof wa); memset(wb, 0, sizeof wb);
    s1 = s2 = 0;
    sm_vl32(rw, len, mx, wa, 2, &s1);
    sm_vl8(rw, len, mx, wb, 2, &s2);
    int diff = 0;
    for (int i = 0; i < (len / 16) * 64 + 64; ++i) if (wa[i] != wb[i]) diff++;
    printf("%-4s cpu=%d vlenb=%-4ld len=%-6d vl32 %6.3f ns/elem   vl8 %6.3f ns/elem   "
           "diff=%d sum=%.6g/%.6g\n",
           strcmp(mode, "a100") ? "X100" : "A100", cur_cpu(), vlenb(), len,
           d1 / (double) reps / len * 1e9, d2 / (double) reps / len * 1e9, diff, s1, s2);
    return 0;
}
