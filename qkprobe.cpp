// The real QK inner-loop shape, with KT=32 and KT=64.
//
// QK is issue bound: reading one 8 KB tile repeatedly costs the same as reading
// the whole 215 MB cache, and fmaprobe shows f32m1 and f32m2 deliver identical
// MAC/s. So the only way a 64-wide K tile can win is by halving the fixed cost
// per MAC -- 4 loads, 4 widening converts and 32 broadcast scalars per group of
// four dims. This measures exactly that loop and nothing else.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <pthread.h>
#include <riscv_vector.h>

static double now_s(void) {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static _Float16 g_kt[32768] __attribute__((aligned(64)));   // two 64-wide tiles' worth
static float g_q[2048] __attribute__((aligned(64)));
static float g_sink;
static float g_out[512] __attribute__((aligned(64)));
static const int REP = 2000000;

// KT=32: one tile is 128 dims x 32 positions of f16; a load covers 32 positions.
static void * w32(void *) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.f, vl), a1 = a0, a2 = a0, a3 = a0;
    vfloat32m1_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
#define A(H) \
                a##H = __riscv_vfmacc_vf_f32m1(a##H, g_q[(H) * 128 + d + 0], k0, vl); \
                a##H = __riscv_vfmacc_vf_f32m1(a##H, g_q[(H) * 128 + d + 1], k1, vl); \
                a##H = __riscv_vfmacc_vf_f32m1(a##H, g_q[(H) * 128 + d + 2], k2, vl); \
                a##H = __riscv_vfmacc_vf_f32m1(a##H, g_q[(H) * 128 + d + 3], k3, vl);
    for (int r = 0; r < REP; ++r) {
        const _Float16 * kb = g_kt + (r & 1) * 4096;
        for (int d = 0; d < 128; d += 4) {
            vfloat32m1_t k0 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (d + 0) * 32, vl), vl);
            vfloat32m1_t k1 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (d + 1) * 32, vl), vl);
            vfloat32m1_t k2 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (d + 2) * 32, vl), vl);
            vfloat32m1_t k3 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (d + 3) * 32, vl), vl);
                A(0) A(1) A(2) A(3) A(4) A(5) A(6) A(7)
        }
    }
    __riscv_vse32_v_f32m1(g_out, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(a0, a1, vl),
                                                        __riscv_vfadd_vv_f32m1(a2, a3, vl), vl), vl);
    g_sink = __riscv_vfmv_f_s_f32m1_f32(a4) + __riscv_vfmv_f_s_f32m1_f32(a5)
           + __riscv_vfmv_f_s_f32m1_f32(a6) + __riscv_vfmv_f_s_f32m1_f32(a7);
#undef A
    return nullptr;
}

// KT=64: same four dims, but each load covers 64 positions and the accumulators
// are f32m2. Half the iterations for the same MACs.
static void * w64(void *) {
    const size_t vl = __riscv_vsetvl_e32m2(64);
    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.f, vl), a1 = a0, a2 = a0, a3 = a0;
    vfloat32m2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
#define A(H) \
                a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 0], k0, vl); \
                a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 1], k1, vl); \
                a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 2], k2, vl); \
                a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 3], k3, vl);
    for (int r = 0; r < REP; ++r) {
        const _Float16 * kb = g_kt + (r & 1) * 8192;
        for (int d = 0; d < 128; d += 4) {
            vfloat32m2_t k0 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 0) * 64, vl), vl);
            vfloat32m2_t k1 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 1) * 64, vl), vl);
            vfloat32m2_t k2 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 2) * 64, vl), vl);
            vfloat32m2_t k3 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 3) * 64, vl), vl);
                A(0) A(1) A(2) A(3) A(4) A(5) A(6) A(7)
        }
    }
    __riscv_vse32_v_f32m2(g_out, __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a0, a1, vl),
                                                        __riscv_vfadd_vv_f32m2(a2, a3, vl), vl), vl);
#undef A
    return nullptr;
}

int main(int argc, char ** argv) {
    const int w = argc > 1 ? atoi(argv[1]) : 32;
    for (int i = 0; i < 32768; ++i) g_kt[i] = (_Float16) ((i & 63) * 0.01f);
    for (int i = 0; i < 2048; ++i) g_q[i] = 1.0f + (float) (i & 15) * 1e-3f;
    const double t0 = now_s();
    pthread_t th;
    if (w == 32) pthread_create(&th, nullptr, w32, nullptr);
    else         pthread_create(&th, nullptr, w64, nullptr);
    pthread_join(th, nullptr);
    const double dt = now_s() - t0;
    // one rep = 8 heads * 128 dims * (32 or 64) positions
    const double macs = (double) REP * 8.0 * 128.0 * w;
    printf("  KT=%d: %.3f s  %.2f GMAC/s  %.1f MAC/cycle @1.8GHz  (sink %.1f)\n",
           w, dt, macs / dt / 1e9, macs / dt / 1.8e9, (double) g_sink);
    return 0;
}
