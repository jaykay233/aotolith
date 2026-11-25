// Rewrite the X100 A-copy quantiser's inner step: four rows at a time, one
// vrgather, one 64-byte store.
//
// The routine mega3.cpp uses now walks rows outermost and does, per 16 k values,
// two vsetvl round-trips and two 8-byte stores. Its block layout is
//   [r0 k0-7][r1 k0-7][r2 k0-7][r3 k0-7][r0 k8-15][r1 k8-15][r2 k8-15][r3 k8-15]
// which is a 4x16 byte transpose of the four rows. `vcreate` would assemble the
// tile from fractional-LMUL pieces, but this toolchain's header has no vcreate, so
// the tile is built in assembly instead, where the lane offsets are explicit:
// narrowing into v4, v5, v6, v7 at e8/mf2 puts each row's sixteen bytes at lanes
// 0, 32, 64, 96 of the e8/m4 group, which is what the index vector below gathers
// from.
//
// Two more wins come for free in that form: vnclip.wi saturates on the way down,
// so it replaces each (vmax, vmin, vncvt) triple with one instruction, and the
// four rows share a single vsetvli round-trip per step.
//
//   -k  contraction length (default 6144)
//   -g  row groups (default 64)
//   -n  repeats, best kept (default 5)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <riscv_vector.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static int64_t K = 6144, GRP = 64;
static int REPS = 5;
static float * g_src;
static int8_t * g_out;
static int64_t g_stride;

// Source lane for each of the block's first 64 bytes. Row r's sixteen bytes sit at
// lane r*32 in the e8/m4 tile; byte l takes k=l%8 of row l/8 in the low half and
// k=8+l%8 in the high half.
static uint8_t g_idx[128] __attribute__((aligned(64)));
static void build_idx(void) {
    memset(g_idx, 0, sizeof g_idx);
    for (int l = 0; l < 64; ++l) {
        const int r = (l & 31) / 8;
        const int j = (l & 7) + ((l & 32) ? 8 : 0);
        g_idx[l] = (uint8_t) (r * 32 + j);
    }
}

// ---- reference: what mega3.cpp does now -------------------------------------
static void qx_base(int64_t g, const float * rowmax) {
    const int64_t k = K, kst = k / 16;
    int8_t * xbase = g_out + g * g_stride;
    float xr[4];
    for (int r = 0; r < 4; ++r)
        xr[r] = 1.0f / (rowmax[r] > 0 ? rowmax[r] / 127.0f : 1.0f);
    for (int r = 0; r < 4; ++r) {
        const float * a = g_src + (g * 4 + r) * k;
        for (int64_t t = 0; t < kst; ++t) {
            const size_t vl = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(a + t * 16, vl);
            vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(__riscv_vfmul_vf_f32m1(v, xr[r], vl), vl);
            q = __riscv_vmax_vx_i32m1(q, -127, vl);
            q = __riscv_vmin_vx_i32m1(q, 127, vl);
            const vint8mf4_t b8 =
                __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, vl), vl);
            const size_t v8 = __riscv_vsetvl_e8mf4(8);
            int8_t * xo = xbase + t * 64 + r * 8;
            __riscv_vse8_v_i8mf4(xo, b8, v8);
            __riscv_vse8_v_i8mf4(xo + 32,
                __riscv_vslidedown_vx_i8mf4(b8, 8, (size_t) 16), v8);
            __riscv_vsetvl_e32m1(16);
        }
    }
}

// ---- candidate: four rows per step, slideups, two 32-byte stores ------------
// vrgather.vv on SEW=8 measured 100 M elem/s here, five times slower than the
// code it would replace, so the transpose is done with vslideup instead. Each
// slideup is a register-to-register shift, and at level 1 the destination holds
// exactly the 32 bytes of one half-block.
//
// v20..v23 hold row r's sixteen quantised bytes in lanes 0..15 of their own
// register. Reading them as e8/m1 with vl=32 puts row r's low eight bytes at
// lane 0, so a slideup by 8r lands them at the right place in the 32-byte half.
static void qx_slide(int64_t g, const float * rowmax) {
    const int64_t k = K, kst = k / 16;
    int8_t * dst = g_out + g * g_stride;
    float xr[4];
    for (int r = 0; r < 4; ++r)
        xr[r] = 1.0f / (rowmax[r] > 0 ? rowmax[r] / 127.0f : 1.0f);
    const float * a0 = g_src + (g * 4 + 0) * k;
    const float * a1 = g_src + (g * 4 + 1) * k;
    const float * a2 = g_src + (g * 4 + 2) * k;
    const float * a3 = g_src + (g * 4 + 3) * k;
    asm volatile(
        "mv      t2, %[NK]            \n\t"
        "1:                           \n\t"
        "vsetivli x0, 16, e32, m2     \n\t"
        "vle32.v v8,  (%[A0])         \n\t"
        "vle32.v v10, (%[A1])         \n\t"
        "vle32.v v12, (%[A2])         \n\t"
        "vle32.v v14, (%[A3])         \n\t"
        "addi    %[A0], %[A0], 64     \n\t"
        "addi    %[A1], %[A1], 64     \n\t"
        "addi    %[A2], %[A2], 64     \n\t"
        "addi    %[A3], %[A3], 64     \n\t"
        "vfmul.vf v8,  v8,  %[X0]     \n\t"
        "vfmul.vf v10, v10, %[X1]     \n\t"
        "vfmul.vf v12, v12, %[X2]     \n\t"
        "vfmul.vf v14, v14, %[X3]     \n\t"
        "vfcvt.x.f.v v8,  v8          \n\t"
        "vfcvt.x.f.v v10, v10         \n\t"
        "vfcvt.x.f.v v12, v12         \n\t"
        "vfcvt.x.f.v v14, v14         \n\t"
        "vsetivli x0, 16, e16, m1     \n\t"
        "vnclip.wi v20, v8,  0        \n\t"
        "vnclip.wi v21, v10, 0        \n\t"
        "vnclip.wi v22, v12, 0        \n\t"
        "vnclip.wi v23, v14, 0        \n\t"
        "vsetivli x0, 16, e8, mf2     \n\t"
        "vnclip.wi v20, v20, 0        \n\t"
        "vnclip.wi v21, v21, 0        \n\t"
        "vnclip.wi v22, v22, 0        \n\t"
        "vnclip.wi v23, v23, 0        \n\t"
        "li      t1, 32               \n\t"
        "vsetvli zero, t1, e8, m1     \n\t"
        "vslidedown.vi v24, v20, 8    \n\t"
        "vslidedown.vi v25, v21, 8    \n\t"
        "vslidedown.vi v26, v22, 8    \n\t"
        "vslidedown.vi v27, v23, 8    \n\t"
        "vmv1r.v  v4, v20             \n\t"
        "vslideup.vi v4, v21, 8       \n\t"
        "vslideup.vi v4, v22, 16      \n\t"
        "vslideup.vi v4, v23, 24      \n\t"
        "vmv1r.v  v5, v24             \n\t"
        "vslideup.vi v5, v25, 8       \n\t"
        "vslideup.vi v5, v26, 16      \n\t"
        "vslideup.vi v5, v27, 24      \n\t"
        "vse8.v   v4, (%[DST])        \n\t"
        "addi    t3, %[DST], 32       \n\t"
        "vse8.v   v5, (t3)           \n\t"
        "addi    %[DST], %[DST], 64   \n\t"
        "addi    t2, t2, -1           \n\t"
        "bgtz    t2, 1b               \n\t"
        : [A0] "+r"(a0), [A1] "+r"(a1), [A2] "+r"(a2), [A3] "+r"(a3), [DST] "+r"(dst)
        : [NK] "r"(kst),
          [X0] "f"(xr[0]), [X1] "f"(xr[1]), [X2] "f"(xr[2]), [X3] "f"(xr[3])
        : "t1", "t2", "t3",
          "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "vl", "memory");
}

// The layout, written out one element at a time.
static void expect_step(const float * src, int64_t g, int64_t t, const float * rowmax,
                        int8_t * blk) {
    float xr[4];
    for (int r = 0; r < 4; ++r)
        xr[r] = 1.0f / (rowmax[r] > 0 ? rowmax[r] / 127.0f : 1.0f);
    for (int r = 0; r < 4; ++r)
        for (int j = 0; j < 16; ++j) {
            const float x = src[(g * 4 + r) * K + t * 16 + j];
            int b = (int) lrintf(x * xr[r]);
            if (b > 127) b = 127;
            if (b < -127) b = -127;
            blk[((j < 8) ? (r * 8 + j) : (32 + r * 8 + (j - 8)))] = (int8_t) b;
        }
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k")) K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-g")) GRP = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-n")) REPS = atoi(argv[++i]);
    }
    {
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
    }
    bind_cpu(8);
    {
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { ssize_t w2 = write(fd, "0", 1); (void) w2; close(fd); }
    }
    printf("vlen: e8m1=%zu e8m2=%zu e8m4=%zu e32m1=%zu e32m2=%zu e8mf2=%zu e8mf4=%zu\n",
           __riscv_vsetvl_e8m1(9999), __riscv_vsetvl_e8m2(9999), __riscv_vsetvl_e8m4(9999),
           __riscv_vsetvl_e32m1(9999), __riscv_vsetvl_e32m2(9999),
           __riscv_vsetvl_e8mf2(9999), __riscv_vsetvl_e8mf4(9999));
    build_idx();
    g_stride = 4 * K;
    g_src = (float *) aligned_alloc(4096, (size_t) GRP * 4 * K * 4 + 4096);
    g_out = (int8_t *) aligned_alloc(4096, (size_t) GRP * g_stride + 4096);
    for (int64_t i = 0; i < GRP * 4 * K; ++i) g_src[i] = sinf((float) i * 0.013f) * 2.7f;

    float * rowmax = (float *) aligned_alloc(64, (size_t) GRP * 4 * sizeof(float));
    for (int64_t g = 0; g < GRP; ++g)
        for (int r = 0; r < 4; ++r) {
            float mx = 0;
            for (int64_t i = 0; i < K; ++i) mx = fmaxf(mx, fabsf(g_src[(g * 4 + r) * K + i]));
            rowmax[g * 4 + r] = mx;
        }

    const int64_t nsteps = K / 16;
    for (int w = 0; w < 2; ++w) {
        int bad = 0;
        for (int64_t g = 0; g < (GRP > 4 ? 4 : GRP); ++g) {
            memset(g_out + g * g_stride, 0, (size_t) g_stride);
            if (w == 0) qx_base(g, rowmax + g * 4); else qx_slide(g, rowmax + g * 4);
            int8_t blk[64];
            for (int64_t t = 0; t < nsteps; t += (nsteps > 8 ? nsteps / 8 : 1)) {
                memset(blk, 0, 64);
                expect_step(g_src, g, t, rowmax + g * 4, blk);
                for (int i = 0; i < 64; ++i)
                    if (blk[i] != g_out[g * g_stride + t * 64 + i]) ++bad;
            }
        }
        printf("%s vs layout spec: %s\n", w ? "slide" : "base", bad ? "MISMATCH" : "exact");
    }

    const double elems = (double) GRP * 4.0 * (double) K;
    for (int w = 0; w < 2; ++w) {
        double best = 1e9;
        for (int rep = 0; rep < REPS; ++rep) {
            const double t0 = now_s();
            for (int64_t g = 0; g < GRP; ++g) {
                if (w == 0) qx_base(g, rowmax + g * 4); else qx_slide(g, rowmax + g * 4);
            }
            const double d = now_s() - t0;
            if (rep >= 1 && d < best) best = d;
        }
        printf("  %-22s %7.4f s  %7.1f M elem/s\n",
               w ? "slide (4 rows, 2x32B)" : "base (per row, 8B)", best, elems / best / 1e6);
    }
    return 0;
}
