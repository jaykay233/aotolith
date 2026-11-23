// Cache-behaviour probe for the prefill QK tile kernel.
// Same inner loop as rvv_qk_dot_tile_f16_x4, but K comes from NT tiles (NT=1 -> 32 KB,
// fits L1; NT=256 -> 8 MB, cold every pass) and 1..8 threads run concurrently.
// Compares 4 rows/pass (current) against 8 rows/pass (halves K bytes per MAC).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>
#include <sched.h>
#include <riscv_vector.h>

static inline uint64_t rdtime(){ uint64_t t; __asm__ __volatile__("rdtime %0":"=r"(t)); return t; }

#ifndef NT
#define NT 1
#endif
#ifndef VAR
#define VAR 1
#endif
#ifndef THREADS
#define THREADS 8
#endif

alignas(64) static _Float16 kbuf[(size_t)NT * 128 * 128 + 4096];
alignas(64) static _Float16 q_all[16][8 * 128 + 64];
alignas(64) static float    dst_all[16][8 * 128 + 64];

static double g_ns[16];

#if VAR == 1
// x4 wide (current): kv=128, f16m2 load, 4 x f32m4   -> 65536 MAC / 32 KB K
static double body(int ith, int reps){
    const _Float16 *q = q_all[ith];
    float *dst = dst_all[ith];
    const size_t vl = __riscv_vsetvl_e16m2(128);
    uint64_t t0 = rdtime();
    for (int r = 0; r < reps; ++r) {
        for (int t = 0; t < NT; ++t) {
            const _Float16 *kp = kbuf + (size_t)t * 128 * 128;
            vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0, vl), a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < 128; ++d) {
                vfloat16m2_t kv = __riscv_vle16_v_f16m2(kp + d * 128, vl);
                a0 = __riscv_vfwmacc_vf_f32m4(a0, q[0*128+d], kv, vl);
                a1 = __riscv_vfwmacc_vf_f32m4(a1, q[1*128+d], kv, vl);
                a2 = __riscv_vfwmacc_vf_f32m4(a2, q[2*128+d], kv, vl);
                a3 = __riscv_vfwmacc_vf_f32m4(a3, q[3*128+d], kv, vl);
            }
            __riscv_vse32_v_f32m4(dst+0*128, a0, vl);
            __riscv_vse32_v_f32m4(dst+1*128, a1, vl);
            __riscv_vse32_v_f32m4(dst+2*128, a2, vl);
            __riscv_vse32_v_f32m4(dst+3*128, a3, vl);
        }
    }
    uint64_t t1 = rdtime();
    g_ns[ith] = (double)(t1-t0)*1000.0/24.0;
    return (double)reps * NT * 4.0 * 128.0 * 128.0;
}
#elif VAR == 2
// x8 narrow: kv=64, f16m1 load, 8 x f32m2       -> 131072 MAC / 32 KB K
static double body(int ith, int reps){
    const _Float16 *q = q_all[ith];
    float *dst = dst_all[ith];
    const size_t vl = __riscv_vsetvl_e16m1(64);
    uint64_t t0 = rdtime();
    for (int r = 0; r < reps; ++r) {
        for (int t = 0; t < NT; ++t) {
            const _Float16 *kp = kbuf + (size_t)t * 128 * 128;
            for (int half = 0; half < 2; ++half) {
                vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0, vl), a1 = a0, a2 = a0, a3 = a0,
                              a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                for (int d = 0; d < 128; ++d) {
                    vfloat16m1_t kv = __riscv_vle16_v_f16m1(kp + d * 128 + half * 64, vl);
                    a0 = __riscv_vfwmacc_vf_f32m2(a0, q[0*128+d], kv, vl);
                    a1 = __riscv_vfwmacc_vf_f32m2(a1, q[1*128+d], kv, vl);
                    a2 = __riscv_vfwmacc_vf_f32m2(a2, q[2*128+d], kv, vl);
                    a3 = __riscv_vfwmacc_vf_f32m2(a3, q[3*128+d], kv, vl);
                    a4 = __riscv_vfwmacc_vf_f32m2(a4, q[4*128+d], kv, vl);
                    a5 = __riscv_vfwmacc_vf_f32m2(a5, q[5*128+d], kv, vl);
                    a6 = __riscv_vfwmacc_vf_f32m2(a6, q[6*128+d], kv, vl);
                    a7 = __riscv_vfwmacc_vf_f32m2(a7, q[7*128+d], kv, vl);
                }
                __riscv_vse32_v_f32m2(dst+0*64, a0, vl); __riscv_vse32_v_f32m2(dst+1*64, a1, vl);
                __riscv_vse32_v_f32m2(dst+2*64, a2, vl); __riscv_vse32_v_f32m2(dst+3*64, a3, vl);
                __riscv_vse32_v_f32m2(dst+4*64, a4, vl); __riscv_vse32_v_f32m2(dst+5*64, a5, vl);
                __riscv_vse32_v_f32m2(dst+6*64, a6, vl); __riscv_vse32_v_f32m2(dst+7*64, a7, vl);
            }
        }
    }
    uint64_t t1 = rdtime();
    g_ns[ith] = (double)(t1-t0)*1000.0/24.0;
    return (double)reps * NT * 2.0 * 8.0 * 64.0 * 128.0;
}
#elif VAR == 3
// x4 narrow: kv=64, f16m1 load, 4 x f32m2       -> 65536 MAC / 32 KB K
static double body(int ith, int reps){
    const _Float16 *q = q_all[ith];
    float *dst = dst_all[ith];
    const size_t vl = __riscv_vsetvl_e16m1(64);
    uint64_t t0 = rdtime();
    for (int r = 0; r < reps; ++r) {
        for (int t = 0; t < NT; ++t) {
            const _Float16 *kp = kbuf + (size_t)t * 128 * 128;
            for (int half = 0; half < 2; ++half) {
                vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0, vl), a1 = a0, a2 = a0, a3 = a0;
                for (int d = 0; d < 128; ++d) {
                    vfloat16m1_t kv = __riscv_vle16_v_f16m1(kp + d * 128 + half * 64, vl);
                    a0 = __riscv_vfwmacc_vf_f32m2(a0, q[0*128+d], kv, vl);
                    a1 = __riscv_vfwmacc_vf_f32m2(a1, q[1*128+d], kv, vl);
                    a2 = __riscv_vfwmacc_vf_f32m2(a2, q[2*128+d], kv, vl);
                    a3 = __riscv_vfwmacc_vf_f32m2(a3, q[3*128+d], kv, vl);
                }
                __riscv_vse32_v_f32m2(dst+0*64, a0, vl); __riscv_vse32_v_f32m2(dst+1*64, a1, vl);
                __riscv_vse32_v_f32m2(dst+2*64, a2, vl); __riscv_vse32_v_f32m2(dst+3*64, a3, vl);
            }
        }
    }
    uint64_t t1 = rdtime();
    g_ns[ith] = (double)(t1-t0)*1000.0/24.0;
    return (double)reps * NT * 2.0 * 4.0 * 64.0 * 128.0;
}
#else
// x8 wide: kv=128, f16m2 load, 8 x f32m4 (4 KB of accumulators: spills)
static double body(int ith, int reps){
    const _Float16 *q = q_all[ith];
    float *dst = dst_all[ith];
    const size_t vl = __riscv_vsetvl_e16m2(128);
    uint64_t t0 = rdtime();
    for (int r = 0; r < reps; ++r) {
        for (int t = 0; t < NT; ++t) {
            const _Float16 *kp = kbuf + (size_t)t * 128 * 128;
            vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0, vl), a1 = a0, a2 = a0, a3 = a0,
                          a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int d = 0; d < 128; ++d) {
                vfloat16m2_t kv = __riscv_vle16_v_f16m2(kp + d * 128, vl);
                a0 = __riscv_vfwmacc_vf_f32m4(a0, q[0*128+d], kv, vl);
                a1 = __riscv_vfwmacc_vf_f32m4(a1, q[1*128+d], kv, vl);
                a2 = __riscv_vfwmacc_vf_f32m4(a2, q[2*128+d], kv, vl);
                a3 = __riscv_vfwmacc_vf_f32m4(a3, q[3*128+d], kv, vl);
                a4 = __riscv_vfwmacc_vf_f32m4(a4, q[4*128+d], kv, vl);
                a5 = __riscv_vfwmacc_vf_f32m4(a5, q[5*128+d], kv, vl);
                a6 = __riscv_vfwmacc_vf_f32m4(a6, q[6*128+d], kv, vl);
                a7 = __riscv_vfwmacc_vf_f32m4(a7, q[7*128+d], kv, vl);
            }
            for (int i = 0; i < 8; ++i) {
                vfloat32m4_t a = i==0?a0:i==1?a1:i==2?a2:i==3?a3:i==4?a4:i==5?a5:i==6?a6:a7;
                __riscv_vse32_v_f32m4(dst+i*128, a, vl);
            }
        }
    }
    uint64_t t1 = rdtime();
    g_ns[ith] = (double)(t1-t0)*1000.0/24.0;
    return (double)reps * NT * 8.0 * 128.0 * 128.0;
}
#endif

static const char * vname(void){
#if VAR == 1
    return "x4 wide kv128 (current)";
#elif VAR == 2
    return "x8 narrow kv64";
#elif VAR == 3
    return "x4 narrow kv64";
#else
    return "x8 wide kv128";
#endif
}

int main(int argc, char ** argv){
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    for (size_t i = 0; i < sizeof(kbuf)/sizeof(kbuf[0]); ++i) kbuf[i] = (_Float16)((i % 17) * 0.003f);
    for (int th = 0; th < 16; ++th)
        for (int i = 0; i < 8*128+64; ++i) { q_all[th][i] = (_Float16)((i % 11) * 0.01f); dst_all[th][i] = 0.f; }

    std::vector<std::thread> ts;
    for (int i = 0; i < THREADS; ++i) {
        ts.emplace_back([i, reps]{
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(i, &set);
            sched_setaffinity(0, sizeof(set), &set);
            body(i, reps);
        });
    }
    for (auto & t : ts) t.join();

    double worst = 0;
    // per-thread rates (all threads do identical work)
    double mac = (double)reps * NT * (
#if VAR == 1
        4.0 * 128.0 * 128.0
#elif VAR == 2
        2.0 * 8.0 * 64.0 * 128.0
#elif VAR == 3
        2.0 * 4.0 * 64.0 * 128.0
#else
        8.0 * 128.0 * 128.0
#endif
    );
    printf("VAR=%s NT=%d threads=%d : ", vname(), NT, THREADS);
    double sum = 0;
    for (int i = 0; i < THREADS; ++i) {
        double rate = mac / g_ns[i];           // GMAC/s (mac in MMAC, ns in ns -> G/s)
        printf("%.1f ", rate);
        sum += rate;
        if (i == 0 || g_ns[i] > worst) worst = g_ns[i];
    }
    printf("| agg %.1f GMAC/s  slowest %.1f GMAC/s/thread\n", sum, mac / worst);
    return 0;
}
