// Why does the deployed QK run at 42 GMAC/s when the identical vmadot loop in PV
// reaches 78 and probe_qk_epi's best variant reaches 93.8?
//
// pfa_qk_i8 and pfa_pv_i8 use the same 8-accumulator asm. The differences are
// (a) QK writes a 16-position-block score buffer as it goes, PV writes nothing
// until the range ends, and (b) QK re-reads the same 8 KB K block once per row
// group while scores stream out between those reads. This probe isolates those
// two: same tiling, same 16 groups, real-sized K plane, store on/off.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <riscv_vector.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int bind_a(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof s, &s);
}

static const int NTH = 8;
static const int HEAD_DIM = 128;
static const int CTX = 10240;
static const int NTILE = CTX / 64;      // 64-row tiles
static const int G = 16;                // row groups of 4
static const int NPB = (CTX + 63) / 64; // 64-position K blocks
static const int SC_SHIFT = 6;
static const int SCMAX = 1280;          // blocks per group in the deployed layout

static int8_t * g_k8i;                  // [NPB][8][1024]
static int8_t * g_q8i;                  // [tiles][G][8][64]
static int16_t * g_sc;                  // per-thread score tile

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

static int g_store = 1, g_groups = G, g_nslots = 1, g_spread = 1;
static double g_tm[NTH];
static long g_mac[NTH];

static void * work(void * arg) {
    const int tid = (int) (intptr_t) arg;
    bind_a(8 + tid);
    // Production rotates 24 score tiles of a full MAXCTX-worth of blocks, so a
    // tile's writes never land on the same lines twice. g_nslots=1 models the
    // earlier probe, which reused one buffer and so kept every store in L1.
    const size_t slot_sz = (size_t) G * NPB * 32;
    int16_t * base = g_sc + (size_t) tid * (size_t) g_nslots * slot_sz;
    alignas(64) int32_t raw[8][32];
    const size_t vl32 = __riscv_vsetvl_e32m1(32);
    const size_t vl16 = __riscv_vsetvl_e16mf2(32);
    double t0 = now_s();
    long mac = 0;
    for (int t = tid; t < NTILE; t += NTH) {
        const int npb = (t * 64 + 64 + 63) / 64;
        const int8_t * q8t = g_q8i + (size_t) t * G * 8 * 64;
        int16_t * sc = base + (size_t) (t % g_nslots) * slot_sz;
        for (int pb = 0; pb < npb; ++pb) {
            const int8_t * kb = g_k8i + (size_t) pb * 8 * 1024;
            int16_t * blk = sc + (size_t) pb * 32;
            for (int gg = 0; gg < g_groups; ++gg) {
                int8_t * op[8];
                for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
                qk4(op, q8t + gg * 512, kb, HEAD_DIM / 16);
                int16_t * dst = blk + (size_t) gg * (g_spread ? (SCMAX * 32) : (NPB * 32));
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                    const vint16mf2_t h = __riscv_vnsra_wx_i16mf2(a, SC_SHIFT, vl32);
                    if (g_store)
                        __riscv_vse16_v_i16mf2(dst + j * 32, h, vl16);
                    else
                        mac += (long) __riscv_vmv_x_s_i16mf2_i16(h);
                }
                mac += 4 * 64 * HEAD_DIM;
            }
        }
    }
    g_tm[tid] = now_s() - t0;
    g_mac[tid] = mac;
    return nullptr;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s")) g_store = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")) g_groups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) g_nslots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")) g_spread = atoi(argv[++i]);
    }
    g_k8i = (int8_t *) aligned_alloc(4096, (size_t) NPB * 8 * 1024);
    g_q8i = (int8_t *) aligned_alloc(4096, (size_t) NTILE * G * 8 * 64);
    const size_t slot_sz = (size_t) G * SCMAX * 32;
    g_sc  = (int16_t *) aligned_alloc(4096, (size_t) NTH * (size_t) g_nslots * slot_sz * 2);
    memset(g_k8i, 3, (size_t) NPB * 8 * 1024);
    memset(g_q8i, 5, (size_t) NTILE * G * 8 * 64);
    memset(g_sc, 0, (size_t) NTH * (size_t) g_nslots * slot_sz * 2);
    pthread_t th[NTH];
    for (long i = 0; i < NTH; ++i) pthread_create(&th[i], nullptr, work, (void *) i);
    double tm = 0; long mac = 0;
    for (int i = 0; i < NTH; ++i) pthread_join(th[i], nullptr);
    for (int i = 0; i < NTH; ++i) { if (g_tm[i] > tm) tm = g_tm[i]; mac += g_mac[i]; }
    printf("store=%d groups=%d slots=%d spread=%d: %.3f s, qk %.1f GMAC/s aggregate (%.1f per core)\n",
           g_store, g_groups, g_nslots, g_spread, tm, (double) mac / tm / 1e9, (double) mac / tm / 1e9 / NTH);
    return 0;
}
