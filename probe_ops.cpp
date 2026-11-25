// Per-instruction cost of the primitives the decode attention kernels are built
// from, measured on the core it is pinned to. A vector instruction that the local
// implementation does not provide is trap-emulated, which costs hundreds of ns
// instead of a few -- that shows up here as a per-op number two orders of
// magnitude above the FMA baseline, and it is the only way to tell an emulated op
// from a merely narrow one.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <riscv_vector.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void pin(int cpu) {
    if (cpu >= 8) {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    }
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static const int N = 4096;
static float buf_a[N * 8] __attribute__((aligned(64)));
static float buf_b[N * 8] __attribute__((aligned(64)));
static _Float16 buf_h[N * 8] __attribute__((aligned(64)));

int main(int argc, char ** argv) {
    const int cpu = argc > 1 ? atoi(argv[1]) : 0;
    pin(cpu);
    long vb = 0;
    asm volatile("csrr %0, vlenb" : "=r"(vb));
    for (int i = 0; i < N * 8; ++i) {
        buf_a[i] = (float) (i % 17) * 0.0625f;
        buf_b[i] = (float) (i % 13) * 0.125f;
        buf_h[i] = (_Float16) ((i % 11) * 0.0625f);
    }
    const size_t vl = __riscv_vsetvl_e32m1((size_t) 1 << 20);
    const long iters = 400000;

    // f32 FMA, eight independent accumulators so the chain is not latency-bound
    {
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(1.0f, vl), a1 = a0, a2 = a0, a3 = a0;
        vfloat32m1_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const vfloat32m1_t u = __riscv_vle32_v_f32m1(buf_b, vl);
        double t = now_s();
        for (long i = 0; i < iters; ++i) {
            a0 = __riscv_vfmacc_vv_f32m1(a0, u, u, vl);
            a1 = __riscv_vfmacc_vv_f32m1(a1, u, u, vl);
            a2 = __riscv_vfmacc_vv_f32m1(a2, u, u, vl);
            a3 = __riscv_vfmacc_vv_f32m1(a3, u, u, vl);
            a4 = __riscv_vfmacc_vv_f32m1(a4, u, u, vl);
            a5 = __riscv_vfmacc_vv_f32m1(a5, u, u, vl);
            a6 = __riscv_vfmacc_vv_f32m1(a6, u, u, vl);
            a7 = __riscv_vfmacc_vv_f32m1(a7, u, u, vl);
        }
        const double d = now_s() - t;
        vfloat32m1_t s = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(a0, a1, vl),
                                                __riscv_vfadd_vv_f32m1(a2, a3, vl), vl);
        s = __riscv_vfadd_vv_f32m1(s, __riscv_vfadd_vv_f32m1(
                __riscv_vfadd_vv_f32m1(a4, a5, vl), __riscv_vfadd_vv_f32m1(a6, a7, vl), vl), vl);
        __riscv_vse32_v_f32m1(buf_a, s, vl);
        printf("%-28s %8.2f ns/op  (%5.1f GMAC/s)\n", "f32m1 fma",
               d / (iters * 8) * 1e9, (double) (iters * 8) * (double) vl / d / 1e9);
    }
    // f32 load + add
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        double t = now_s();
        for (long i = 0; i < iters; ++i) {
            const vfloat32m1_t x = __riscv_vle32_v_f32m1(buf_a + ((i * 8) & 8191), vl);
            acc = __riscv_vfadd_vv_f32m1(acc, x, vl);
        }
        const double d = now_s() - t;
        __riscv_vse32_v_f32m1(buf_b, acc, vl);
        printf("%-28s %8.2f ns/op\n", "f32m1 load+add", d / iters * 1e9);
    }
    // f16 load + widening convert to f32m1 (the QK inner loop's K read)
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        double t = now_s();
        for (long i = 0; i < iters; ++i) {
            const vfloat16mf2_t h = __riscv_vle16_v_f16mf2(buf_h + ((i * 8) & 8191), vl);
            const vfloat32m1_t f = __riscv_vfwcvt_f_f_v_f32m1(h, vl);
            acc = __riscv_vfadd_vv_f32m1(acc, f, vl);
        }
        const double d = now_s() - t;
        __riscv_vse32_v_f32m1(buf_b, acc, vl);
        printf("%-28s %8.2f ns/op\n", "f16 load+vwcvt", d / iters * 1e9);
    }
    // int8 load + widening convert (the int8 KV path's K read)
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        const int8_t * p = (const int8_t *) buf_h;
        double t = now_s();
        for (long i = 0; i < iters; ++i) {
            const vint8mf4_t b = __riscv_vle8_v_i8mf4(p + ((i * 8) & 8191), vl);
            const vfloat32m1_t f = __riscv_vfcvt_f_x_v_f32m1(
                __riscv_vwcvt_x_x_v_i32m1(__riscv_vwcvt_x_x_v_i16mf2(b, vl), vl), vl);
            acc = __riscv_vfadd_vv_f32m1(acc, f, vl);
        }
        const double d = now_s() - t;
        __riscv_vse32_v_f32m1(buf_b, acc, vl);
        printf("%-28s %8.2f ns/op\n", "int8 load+3 cvt", d / iters * 1e9);
    }
    // f32 store
    {
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(buf_a, vl);
        double t = now_s();
        for (long i = 0; i < iters; ++i)
            __riscv_vse32_v_f32m1(buf_b + ((i * 8) & 8191), v, vl);
        const double d = now_s() - t;
        printf("%-28s %8.2f ns/op\n", "f32m1 store", d / iters * 1e9);
    }
    // reduction
    {
        vfloat32m1_t acc = __riscv_vle32_v_f32m1(buf_a, vl);
        double t = now_s();
        float sink = 0;
        for (long i = 0; i < iters / 4; ++i)
            sink += __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m1_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));
        const double d = now_s() - t;
        if (sink == 1234.5f) printf("!");
        printf("%-28s %8.2f ns/op\n", "f32m1 vfredusum", d / (iters / 4) * 1e9);
    }
    // f32 -> int convert (exp2f's exponent extraction)
    {
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(buf_a, vl);
        vint32m1_t acc = __riscv_vfcvt_x_f_v_i32m1(v, vl);
        double t = now_s();
        for (long i = 0; i < iters; ++i)
            acc = __riscv_vfcvt_x_f_v_i32m1(__riscv_vfmul_vf_f32m1(v, 1.5f, vl), vl);
        const double d = now_s() - t;
        __riscv_vse32_v_i32m1((int32_t *) buf_b, acc, vl);
        printf("%-28s %8.2f ns/op\n", "f32m1 mul+fcvt.x.f", d / iters * 1e9);
    }
    // gather (used by the table form of exp2f)
    {
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(buf_a, vl);
        const vuint32m1_t idx = __riscv_vreinterpret_v_f32m1_u32m1(v);
        vfloat32m1_t acc = v;
        double t = now_s();
        for (long i = 0; i < iters / 4; ++i)
            acc = __riscv_vfadd_vv_f32m1(acc, __riscv_vrgather_vv_f32m1(v, idx, vl), vl);
        const double d = now_s() - t;
        __riscv_vse32_v_f32m1(buf_b, acc, vl);
        printf("%-28s %8.2f ns/op\n", "f32m1 vrgather", d / (iters / 4) * 1e9);
    }
    printf("cpu %d vlenb %ld -> e32m1 vl %zu\n", cpu, vb, vl);
    return 0;
}
