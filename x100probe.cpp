// QK inner loop on whichever cluster the thread is bound to.
//
// cpu 0..7 are X100 (vlenb=32, e32m2 = 16 lanes), cpu 8..15 are A100
// (vlenb=128, e32m2 = 64 lanes) and need the /proc/set_ai_thread handshake.
// Decode attention currently binds only the A100s; this asks what an X100 core
// would add on the same loop. It cannot add bandwidth (one bus), but QK measured
// as issue bound -- one L1-resident tile instead of the whole cache leaves qk
// unchanged -- and issue is per core.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <riscv_vector.h>

static double now_s(void) {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static _Float16 g_kt[65536] __attribute__((aligned(64)));
static float g_q[2048] __attribute__((aligned(64)));
static float g_out[512] __attribute__((aligned(64)));
static int g_cpu = 0;
static long g_rep = 2000000;
static double g_dt = 0;
static size_t g_vl = 0;
static long g_vb = 0;

static void * worker(void *) {
    if (g_cpu >= 8) {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(g_cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s) != 0) { perror("bind"); return nullptr; }
    cpu_set_t now; CPU_ZERO(&now);
    sched_getaffinity(0, sizeof now, &now);
    if (!CPU_ISSET(g_cpu, &now)) { fprintf(stderr, "  cpu %d: bind lost\n", g_cpu); return nullptr; }

    asm volatile("csrr %0, vlenb" : "=r"(g_vb));
    const size_t vl = __riscv_vsetvl_e32m2((size_t) 4096);
    g_vl = vl;

    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.f, vl), a1 = a0, a2 = a0, a3 = a0;
    vfloat32m2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const double t0 = now_s();
    for (long r = 0; r < g_rep; ++r) {
        const _Float16 * kb = g_kt + (r & 1) * 16384;
        for (int d = 0; d < 128; d += 4) {
            vfloat32m2_t k0 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 0) * 64, vl), vl);
            vfloat32m2_t k1 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 1) * 64, vl), vl);
            vfloat32m2_t k2 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 2) * 64, vl), vl);
            vfloat32m2_t k3 = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(kb + (d + 3) * 64, vl), vl);
#define A(H) \
            a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 0], k0, vl); \
            a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 1], k1, vl); \
            a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 2], k2, vl); \
            a##H = __riscv_vfmacc_vf_f32m2(a##H, g_q[(H) * 128 + d + 3], k3, vl);
            A(0) A(1) A(2) A(3) A(4) A(5) A(6) A(7)
#undef A
        }
    }
    g_dt = now_s() - t0;
    __riscv_vse32_v_f32m2(g_out, __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a0, a1, vl),
                                                        __riscv_vfadd_vv_f32m2(a2, a3, vl), vl), vl);
    return nullptr;
}

int main(int argc, char ** argv) {
    for (int i = 0; i < 65536; ++i) g_kt[i] = (_Float16) ((i & 63) * 0.01f);
    for (int i = 0; i < 2048; ++i) g_q[i] = 1.0f + (float) (i & 15) * 1e-3f;
    if (argc > 1) g_rep = atol(argv[1]);
    for (int cpu = 0; cpu < 8; ++cpu) {
        g_cpu = cpu;
        pthread_t th;
        pthread_create(&th, nullptr, worker, nullptr);
        pthread_join(th, nullptr);
        if (!g_vl) continue;
        // one rep = 8 heads * 128 dims * vl positions
        const double macs = (double) g_rep * 8.0 * 128.0 * (double) g_vl;
        printf("  cpu %d: vlenb=%3ld  e32m2=%2zu lanes  %.3f s  %.2f GMAC/s  %.1f MAC/cycle@2.2GHz\n",
               cpu, g_vb, g_vl, g_dt, macs / g_dt / 1e9, macs / g_dt / 2.2e9);
    }
    return 0;
}
