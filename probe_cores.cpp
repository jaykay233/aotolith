// Per-core QK throughput as a function of how many cores are busy.
//
// x100probe ran one core at a time and reported 37.47 GMAC/s for the QK loop on
// an A100 core, while the production kernel gets ~15.4 GMAC/s per thread. The
// suspicion is clock: one core alone can boost, eight cannot. This runs the same
// inner loop on a caller-chosen set of cores at once and reports aggregate and
// per-core rates, so the two effects -- cluster and occupancy -- separate.
//
//   ./probe_cores 8            one A100 core
//   ./probe_cores 8,9,10,11,12,13,14,15   all eight
//   ./probe_cores 0            one X100 core
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
static float g_out[32][512] __attribute__((aligned(64)));
static long g_rep = 400000;
static double g_dt[32];
static size_t g_vl[32];
static long g_vb[32];
static int g_cpu[32];
static int g_n = 0;

static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    const int cpu = g_cpu[id];
    if (cpu >= 8) {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s) != 0) { perror("bind"); return nullptr; }
    cpu_set_t now; CPU_ZERO(&now);
    sched_getaffinity(0, sizeof now, &now);
    if (!CPU_ISSET(cpu, &now)) { fprintf(stderr, "  cpu %d: bind lost\n", cpu); return nullptr; }

    asm volatile("csrr %0, vlenb" : "=r"(g_vb[id]));
    const size_t vl = __riscv_vsetvl_e32m2((size_t) 4096);
    g_vl[id] = vl;

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
    g_dt[id] = now_s() - t0;
    __riscv_vse32_v_f32m2(g_out[id], __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a0, a1, vl),
                                                             __riscv_vfadd_vv_f32m2(a2, a3, vl), vl), vl);
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s cpu[,cpu...] [rep]\n", argv[0]); return 1; }
    char buf[256];
    snprintf(buf, sizeof buf, "%s", argv[1]);
    for (char * t = strtok(buf, ","); t && g_n < 32; t = strtok(nullptr, ",")) g_cpu[g_n++] = atoi(t);
    if (argc > 2) g_rep = atol(argv[2]);
    for (int i = 0; i < 65536; ++i) g_kt[i] = (_Float16) ((i & 63) * 0.01f);
    for (int i = 0; i < 2048; ++i) g_q[i] = 1.0f + (float) (i & 15) * 1e-3f;

    pthread_t th[32];
    for (int i = 0; i < g_n; ++i) pthread_create(&th[i], nullptr, worker, (void *) (intptr_t) i);
    for (int i = 0; i < g_n; ++i) pthread_join(th[i], nullptr);

    double best = 0, tot = 0;
    for (int i = 0; i < g_n; ++i) {
        if (!g_vl[i]) continue;
        const double macs = (double) g_rep * 8.0 * 128.0 * (double) g_vl[i];
        const double rate = macs / g_dt[i] / 1e9;
        if (rate > best) best = rate;
        tot += rate;
    }
    printf("%d core(s) [%s]: aggregate %.2f GMAC/s, per-core best %.2f GMAC/s, vlenb %ld\n",
           g_n, argv[1], tot, best, g_n ? g_vb[0] : 0L);
    return 0;
}
