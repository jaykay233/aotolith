// Isolates the QK score-write epilogue. Same vmadot inner loop in all variants;
// only what happens to the eight 32-lane accumulators changes. PV reaches
// 100 GMAC/s on this core and QK only 35, and the two inner loops are the same
// eight vmadots over the same 1 KB of B, so the gap has to live here.
//
//   A  full epilogue as shipped: vle32, cvt, scale, cvt, narrow, 3 slidedown,
//      4 x vse16 (4 rows x 8 positions per store)
//   B  no float scale: vle32, vnsra, 3 slidedown, 4 x vse16
//   C  interleaved 4-row layout: vle32, vnsra, 1 x vse16 (no slidedown)
//   D  no epilogue at all: 8 x vse32 of the raw accumulators
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <riscv_vector.h>

static const int N_HEAD = 16, HEAD_DIM = 128, N_KV = 2;
static const int MAXCTX = 10240;
static const int I8_ROWS = 64;
static const int NPB = 160;          // 64-position blocks covering 10K

static double now_s() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

static void qk4(int8_t * const * acc_out, const int8_t * q8g, const int8_t * kblk, int nks) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "vxor.vv v24, v24, v24          \n\t"
        "vxor.vv v26, v26, v26          \n\t"
        "vxor.vv v28, v28, v28          \n\t"
        "vxor.vv v30, v30, v30          \n\t"
        "mv     t3, %[NK]               \n\t"
        "1:                             \n\t"
        "vl1re32.v v0, (%[Q])           \n\t"
        "vl8re32.v v8, (%[K])           \n\t"
        "addi    %[Q], %[Q], 64         \n\t"
        "addi    %[K], %[K], 1024       \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v18, v0, v9            \n\t"
        "vmadot  v20, v0, v10           \n\t"
        "vmadot  v22, v0, v11           \n\t"
        "vmadot  v24, v0, v12           \n\t"
        "vmadot  v26, v0, v13           \n\t"
        "vmadot  v28, v0, v14           \n\t"
        "vmadot  v30, v0, v15           \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        "vse32.v v16, (%[O0])           \n\t"
        "vse32.v v18, (%[O1])           \n\t"
        "vse32.v v20, (%[O2])           \n\t"
        "vse32.v v22, (%[O3])           \n\t"
        "vse32.v v24, (%[O4])           \n\t"
        "vse32.v v26, (%[O5])           \n\t"
        "vse32.v v28, (%[O6])           \n\t"
        "vse32.v v30, (%[O7])           \n\t"
        : [Q] "+r"(q8g), [K] "+r"(kblk)
        : [NK] "r"(nks),
          [O0] "r"(acc_out[0]), [O1] "r"(acc_out[1]), [O2] "r"(acc_out[2]), [O3] "r"(acc_out[3]),
          [O4] "r"(acc_out[4]), [O5] "r"(acc_out[5]), [O6] "r"(acc_out[6]), [O7] "r"(acc_out[7])
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// One tile: 16 row groups x NPB position blocks, exactly what pfa_qk_i8 does.
static void tile(int variant, const int8_t * q8i, const int8_t * k8i,
                 int16_t (*sc)[MAXCTX], int8_t * rawout) {
    alignas(64) int32_t raw[8][32];
    const size_t vl32 = __riscv_vsetvl_e32m1(32);
    const size_t vl8 = __riscv_vsetvl_e16mf2(8);
    for (int gg = 0; gg < 16; ++gg) {
        int16_t * row16[4] = { sc[gg * 4 + 0], sc[gg * 4 + 1], sc[gg * 4 + 2], sc[gg * 4 + 3] };
        // variant 4 (P): 4 rows x 16 positions blocks, the pv4 A layout
        static uint16_t gidx[64] __attribute__((aligned(64)));
        static bool init = false;
        if (!init) {
            for (int d = 0; d < 64; ++d) {
                const int r = d / 16, p = d % 16;
                gidx[d] = (uint16_t) ((p / 8) * 32 + r * 8 + (p % 8));
            }
            init = true;
        }
        for (int pb = 0; pb < NPB; ++pb) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
            qk4(op, q8i + gg * 512, k8i + (size_t) pb * 1024, HEAD_DIM / 16);
            if (variant == 3) {
                for (int j = 0; j < 8; ++j)
                    __riscv_vse32_v_i32m1((int32_t *) (rawout + (size_t) pb * 1024 + j * 128),
                                          __riscv_vle32_v_i32m1(raw[j], vl32), vl32);
                continue;
            }
            if (variant == 4) {          // interleaved blocks, but keep the float scale
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl32), 0.0137f, vl32);
                    const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(
                        __riscv_vfcvt_x_f_v_i32m1(f, vl32), vl32);
                    __riscv_vse16_v_i16mf2((int16_t *) (rawout + ((size_t) pb * 8 + j) * 64),
                                           h16, __riscv_vsetvl_e16mf2(32));
                }
                continue;
            }
            if (variant == 5) {          // 2 j at a time -> vrgather -> 128 B store
                for (int j = 0; j < 8; j += 2) {
                    const size_t vl64 = __riscv_vsetvl_e32m2(64);
                    const vint32m2_t a = __riscv_vle32_v_i32m2(raw[j], vl64);
                    const vint16m1_t h = __riscv_vnsra_wx_i16m1(a, 6, vl64);
                    const size_t vh = __riscv_vsetvl_e16m1(64);
                    const vint16m1_t g = __riscv_vrgather_vv_i16m1(
                        h, __riscv_vle16_v_u16m1(gidx, vh), vh);
                    __riscv_vse16_v_i16m1((int16_t *) (rawout + ((size_t) pb * 8 + j) * 64),
                                          g, vh);
                }
                continue;
            }
            for (int j = 0; j < 8; ++j) {
                const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                if (variant == 0) {
                    const float s = 0.0137f;
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl32), s, vl32);
                    const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(
                        __riscv_vfcvt_x_f_v_i32m1(f, vl32), vl32);
                    for (int r = 0; r < 4; ++r) {
                        const vint16mf2_t sl = r
                            ? __riscv_vslidedown_vx_i16mf2(h16, (size_t) (r * 8), vl8) : h16;
                        __riscv_vse16_v_i16mf2(row16[r] + pb * 64 + j * 8, sl, vl8);
                    }
                } else {
                    const vint16mf2_t h16 = __riscv_vnsra_wx_i16mf2(a, 6, vl32);
                    if (variant == 1) {
                        for (int r = 0; r < 4; ++r) {
                            const vint16mf2_t sl = r
                                ? __riscv_vslidedown_vx_i16mf2(h16, (size_t) (r * 8), vl8) : h16;
                            __riscv_vse16_v_i16mf2(row16[r] + pb * 64 + j * 8, sl, vl8);
                        }
                    } else {
                        // interleaved [4 rows x 8 positions] block, 64 B
                        __riscv_vse16_v_i16mf2((int16_t *) (rawout + ((size_t) pb * 8 + j) * 64),
                                               h16, __riscv_vsetvl_e16mf2(32));
                    }
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 4;
    const int8_t * q8i = (const int8_t *) aligned_alloc(4096, (size_t) 16 * 512 + 4096);
    const int8_t * k8i = (const int8_t *) aligned_alloc(4096, (size_t) NPB * 1024);
    int16_t (*sc)[MAXCTX] = (int16_t (*)[MAXCTX]) aligned_alloc(64, (size_t) I8_ROWS * MAXCTX * 2);
    int8_t * rawout = (int8_t *) aligned_alloc(4096, (size_t) NPB * 1024);
    if (!q8i || !k8i || !sc || !rawout) { fprintf(stderr, "oom\n"); return 1; }
    memset((void *) q8i, 3, (size_t) 16 * 512);
    memset((void *) k8i, 5, (size_t) NPB * 1024);
    memset(sc, 0, (size_t) I8_ROWS * MAXCTX * 2);

    const double macs = (double) 16 * 4 * NPB * 64 * HEAD_DIM;
    const char * nm[6] = { "A full epilogue", "B vnsra+slide", "C interleaved 1 store",
                           "D no epilogue", "E interleaved+scale", "F gather->[r][16]" };
    double base = 0;
    for (int v = 0; v < 6; ++v) {
        tile(v, q8i, k8i, sc, rawout);   // warm
        const double t0 = now_s();
        for (int i = 0; i < reps; ++i) tile(v, q8i, k8i, sc, rawout);
        const double dt = (now_s() - t0) / reps;
        const double g = macs / dt / 1e9;
        if (v == 0) base = g;
        printf("%-22s %7.2f ms  %6.1f GMAC/s   x%.2f\n", nm[v], dt * 1e3, g, g / base);
    }
    return 0;
}
