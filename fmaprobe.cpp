// Does f32m2 FMA deliver 2x the MACs of f32m1 on this core?
//
// QK is issue bound (reading one tile repeatedly costs the same as reading the
// whole cache), so the only way to make it faster is fewer instructions per MAC,
// which means a wider LMUL. PV's m2 and m4 variants both measure slower in situ,
// which could be FMA throughput or could be register pressure. This isolates it:
// 8 accumulators and one broadcast scalar, nothing else live.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <pthread.h>
#include <riscv_vector.h>

static double now_s(void) {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static float g_a[4096] __attribute__((aligned(64)));
static volatile float g_sc = 1.0f;

static void * w1(void *) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.f, vl), a1 = a0, a2 = a0, a3 = a0;
    vfloat32m1_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const vfloat32m1_t v = __riscv_vle32_v_f32m1(g_a, vl);
    const float s = g_sc;
    for (int i = 0; i < 20000000; ++i) {
        a0 = __riscv_vfmacc_vf_f32m1(a0, s, v, vl);
        a1 = __riscv_vfmacc_vf_f32m1(a1, s, v, vl);
        a2 = __riscv_vfmacc_vf_f32m1(a2, s, v, vl);
        a3 = __riscv_vfmacc_vf_f32m1(a3, s, v, vl);
        a4 = __riscv_vfmacc_vf_f32m1(a4, s, v, vl);
        a5 = __riscv_vfmacc_vf_f32m1(a5, s, v, vl);
        a6 = __riscv_vfmacc_vf_f32m1(a6, s, v, vl);
        a7 = __riscv_vfmacc_vf_f32m1(a7, s, v, vl);
    }
    const vfloat32m1_t t = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(a0, a1, vl),
                                                  __riscv_vfadd_vv_f32m1(a2, a3, vl), vl);
    g_a[0] = __riscv_vfmv_f_s_f32m1_f32(t) + __riscv_vfmv_f_s_f32m1_f32(a4)
           + __riscv_vfmv_f_s_f32m1_f32(a5) + __riscv_vfmv_f_s_f32m1_f32(a6)
           + __riscv_vfmv_f_s_f32m1_f32(a7);
    return nullptr;
}

static void * w2(void *) {
    const size_t vl = __riscv_vsetvl_e32m2(64);
    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.f, vl), a1 = a0, a2 = a0, a3 = a0;
    vfloat32m2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const vfloat32m2_t v = __riscv_vle32_v_f32m2(g_a, vl);
    const float s = g_sc;
    for (int i = 0; i < 20000000; ++i) {
        a0 = __riscv_vfmacc_vf_f32m2(a0, s, v, vl);
        a1 = __riscv_vfmacc_vf_f32m2(a1, s, v, vl);
        a2 = __riscv_vfmacc_vf_f32m2(a2, s, v, vl);
        a3 = __riscv_vfmacc_vf_f32m2(a3, s, v, vl);
        a4 = __riscv_vfmacc_vf_f32m2(a4, s, v, vl);
        a5 = __riscv_vfmacc_vf_f32m2(a5, s, v, vl);
        a6 = __riscv_vfmacc_vf_f32m2(a6, s, v, vl);
        a7 = __riscv_vfmacc_vf_f32m2(a7, s, v, vl);
    }
    __riscv_vse32_v_f32m2(g_a, __riscv_vfadd_vv_f32m2(
        __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a0, a1, vl),
                               __riscv_vfadd_vv_f32m2(a2, a3, vl), vl),
        __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a4, a5, vl),
                               __riscv_vfadd_vv_f32m2(a6, a7, vl), vl), vl), vl);
    return nullptr;
}

int main(int argc, char ** argv) {
    const int lm = argc > 1 ? atoi(argv[1]) : 1;
    for (int i = 0; i < 4096; ++i) g_a[i] = 1.0f + (float) i * 1e-6f;
    const double t0 = now_s();
    pthread_t th;
    if (lm == 1) pthread_create(&th, nullptr, w1, nullptr);
    else         pthread_create(&th, nullptr, w2, nullptr);
    pthread_join(th, nullptr);
    const double dt = now_s() - t0;
    const double lanes = lm == 1 ? 32.0 : 64.0;
    const double macs = 20000000.0 * 8.0 * lanes;
    printf("  m%d: %d FMA x 8 acc, %d lanes -> %.3f s  %.2f GMAC/s  %.2f MAC/cycle @1.8GHz\n",
           lm, 20000000, (int) lanes, dt, macs / dt / 1e9, macs / dt / 1.8e9);
    printf("  (sink %.1f)\n", g_a[0]);
    return 0;
}
