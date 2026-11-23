// exp2 accuracy/speed probe: the six-term polynomial against the
// three-instruction mantissa trick and a table-corrected variant.
//
// The mantissa trick: vfcvt.x.f of t*2^23 leaves floor(t) in the exponent and
// the fractional part in the mantissa, so reinterpreting (that + 127<<23) is
// 2^floor(t)*(1+f). That overestimates 2^t by up to 6.2% at f=0.5288, which is
// the whole question: the softmax denominator uses the same weights, so how much
// does the distortion actually cost, and is a 32-entry vrgather correction
// worth four more instructions.
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sched.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline vfloat32m1_t poly(vfloat32m1_t t, size_t vl) {
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

// three instructions: mul, cvt, add
static inline vfloat32m1_t fast3(vfloat32m1_t t, size_t vl) {
    const vint32m1_t i = __riscv_vfcvt_x_f_v_i32m1(
        __riscv_vfmul_vf_f32m1(t, 8388608.0f, vl), vl);
    return __riscv_vreinterpret_v_i32m1_f32m1(
        __riscv_vadd_vx_i32m1(i, 127 << 23, vl));
}

// +4 instructions: correct the linear mantissa with a 32-entry table of
// 2^f/(1+f) indexed by the top five bits of f.
static inline vfloat32m1_t fast3_tab(vfloat32m1_t t, const vfloat32m1_t * tab, int nb, size_t vl) {
    const vint32m1_t i = __riscv_vfcvt_x_f_v_i32m1(
        __riscv_vfmul_vf_f32m1(t, 8388608.0f, vl), vl);
    const vfloat32m1_t y = __riscv_vreinterpret_v_i32m1_f32m1(
        __riscv_vadd_vx_i32m1(i, 127 << 23, vl));
    const vuint32m1_t low = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(i), 0x7FFFFF, vl);
    const vuint32m1_t idx = __riscv_vsrl_vx_u32m1(low, 23 - (unsigned) __builtin_ctz((unsigned) nb), vl);
    return __riscv_vfmul_vv_f32m1(y, __riscv_vrgather_vv_f32m1(*tab, idx, vl), vl);
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        const int cpu = atoi(argv[1]);
        // the A cluster only becomes bindable after the vendor handshake on
        // /proc/set_ai_thread; without it sched_setaffinity returns EINVAL
        for (int a = 0; a < 200; ++a) {
            const int fd = open("/proc/set_ai_thread", O_WRONLY);
            if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
            if (sched_setaffinity(0, sizeof s, &s) == 0) break;
            sched_yield();
        }
        long vb = 0; asm volatile("csrr %0, vlenb" : "=r"(vb));
        printf("cpu=%d vlenb=%ld\n", sched_getcpu(), vb);
    }
    const int N = 4096;
    static float in[N], out[N];
    for (int i = 0; i < N; ++i) in[i] = -30.0f * (float) i / (float) (N - 1);   // [-30, 0]

    size_t vl = __riscv_vsetvl_e32m1(32);
    printf("vl=%zu\n", vl);

    // reference
    static double ref[N];
    for (int i = 0; i < N; ++i) ref[i] = pow(2.0, (double) in[i]);

    // table: 2^f/(1+f) at bucket centres, f = (k+0.5)/32
    // one f32m1 register holds VLMAX entries; the narrow cluster only has eight
    static float tabbuf[32];
    const int nb = (int) vl;
    for (int k = 0; k < 32; ++k)
        tabbuf[k] = k < nb ? (float) (pow(2.0, (k + 0.5) / nb) / (1.0 + (k + 0.5) / nb)) : 1.0f;
    printf("table buckets %d\n", nb);
    vfloat32m1_t tab = __riscv_vle32_v_f32m1(tabbuf, vl);

    volatile float sink = 0;
    for (int mode = 0; mode < 3; ++mode) {
        double worst = 0; float at = 0;
        for (int i = 0; i + (int) vl <= N; i += (int) vl) {
            const vfloat32m1_t t = __riscv_vle32_v_f32m1(in + i, vl);
            vfloat32m1_t y;
            if (mode == 0) y = poly(t, vl);
            else if (mode == 1) y = fast3(t, vl);
            else y = fast3_tab(t, &tab, nb, vl);
            __riscv_vse32_v_f32m1(out + i, y, vl);
        }
        for (int i = 0; i < N; ++i) {
            const double err = fabs((double) out[i] - ref[i]) / ref[i];
            if (err > worst) { worst = err; at = in[i]; }
        }
        // speed: 4M lanes of exp
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            const double t0 = now_s();
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (int it = 0; it < 8192; ++it) {
                // walk the whole range so nothing can be hoisted
                const vfloat32m1_t t = __riscv_vle32_v_f32m1(in + (it * (int) vl) % (N - (int) vl), vl);
                vfloat32m1_t y;
                if (mode == 0) y = poly(t, vl);
                else if (mode == 1) y = fast3(t, vl);
                else y = fast3_tab(t, &tab, nb, vl);
                acc = __riscv_vfadd_vv_f32m1(acc, y, vl);
            }
            sink = __riscv_vfmv_f_s_f32m1_f32(acc);
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("%-10s max rel err %.3e at %.3f | %.2f ns per %zu lanes (%.2f cycles/vector) sink=%.3f\n",
               mode == 0 ? "poly6" : mode == 1 ? "fast3" : "fast3+tab",
               worst, at, best / 8192.0 * 1e9, vl, best / 8192.0 * 1.8e9, (double) sink);
    }
    return 0;
}
