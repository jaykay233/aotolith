// What is the real streaming read ceiling, per core count, on a contiguous
// buffer with huge pages?
//
// The earlier cap (1 core 6.8, 8 cores 17.5, 16 cores 19.3 GB/s) came from a
// strided/irregular pattern and it contradicts the decode GEMM: 1265 MB of
// weights in 40 ms at short context is 31.6 GB/s with 8 threads. If a plain
// contiguous stream really does 30+ GB/s, then the 10K GEMM (23.8 GB/s) and
// attention (qk 15.4, pv 14.3 GB/s) are leaving 1.3-2x on the table and the
// fix is kernel efficiency, not fewer bytes.
//
//   bwprobe [MB] [threads]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <riscv_vector.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static uint8_t * g_buf;
static int64_t g_bytes;
static int g_nt;
static double g_t[16];
static volatile float g_sink[16];

// Each thread walks its own contiguous share with full-width vector loads, so
// every byte of the buffer is actually read.
static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    bind_cpu((id < 8) ? (8 + id) : id);   // A cluster first, then X100
    const int64_t per = g_bytes / g_nt;
    const int64_t lo = per * id;
    const int64_t hi = (id == g_nt - 1) ? g_bytes : per * (id + 1);
    // Four independent accumulator chains, 128 bytes per load: one chain with a
    // loop-carried add is latency bound and reports ~half the real ceiling.
    vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, 32), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    // MEGA_STRIDE=N reads N*32 bytes out of every N*128 bytes: the decode PV loop
    // walks V as 128 useful bytes of each 512-byte position, so this isolates
    // whether the stride, rather than the byte count, is what costs bandwidth.
    const int st = getenv("MEGA_STRIDE") ? atoi(getenv("MEGA_STRIDE")) : 0;
    const double t0 = now_s();
    for (int rep = 0; rep < 3; ++rep)
        for (int64_t i = lo; i < hi; i += 4096)
            for (int j = 0; j < 4096; j += 512) {
                if (st) {
                    const int q = j / 128;
                    if ((q & (st - 1)) != 0) continue;
                }
                const size_t vl = __riscv_vsetvl_e32m1((size_t) 32);
                const float * p = (const float *) (g_buf + i + j);
                acc0 = __riscv_vfadd_vv_f32m1(acc0, __riscv_vle32_v_f32m1(p, vl), vl);
                acc1 = __riscv_vfadd_vv_f32m1(acc1, __riscv_vle32_v_f32m1(p + 32, vl), vl);
                acc2 = __riscv_vfadd_vv_f32m1(acc2, __riscv_vle32_v_f32m1(p + 64, vl), vl);
                acc3 = __riscv_vfadd_vv_f32m1(acc3, __riscv_vle32_v_f32m1(p + 96, vl), vl);
            }
    g_t[id] = now_s() - t0;
    const vfloat32m1_t s01 = __riscv_vfadd_vv_f32m1(acc0, acc1, 32);
    g_sink[id] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
        __riscv_vfadd_vv_f32m1(s01, __riscv_vfadd_vv_f32m1(acc2, acc3, 32), 32),
        __riscv_vfmv_v_f_f32m1(0.0f, 1), 32));
    return nullptr;
}

int main(int argc, char ** argv) {
    const int mb = argc > 1 ? atoi(argv[1]) : 2048;
    g_nt = argc > 2 ? atoi(argv[2]) : 8;
    g_bytes = (int64_t) mb << 20;
    if (argc > 3) {
        // Stream the real weight blob instead of a synthetic buffer: same bytes,
        // same page source (a file in tmpfs) as the decode GEMM reads.
        const int fd = open(argv[3], O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        off_t sz = lseek(fd, 0, SEEK_END);
        if (sz < g_bytes) g_bytes = sz;
        g_buf = (uint8_t *) mmap(nullptr, (size_t) g_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (g_buf == MAP_FAILED) { perror("mmap"); return 1; }
        fprintf(stderr, "file %s, %lld MB, %d threads\n",
                argv[3], (long long) (g_bytes >> 20), g_nt);
    } else {
        g_buf = (uint8_t *) aligned_alloc(2 << 20, (size_t) g_bytes);
        if (!g_buf) { fprintf(stderr, "oom\n"); return 1; }
        for (int64_t i = 0; i < g_bytes; i += 4096) g_buf[i] = (uint8_t) (i * 7);
        fprintf(stderr, "buffer %d MB, %d threads\n", mb, g_nt);
    }
    madvise(g_buf, (size_t) g_bytes, MADV_HUGEPAGE);
    madvise(g_buf, (size_t) g_bytes, MADV_COLLAPSE);

    pthread_t th[16];
    const double t0 = now_s();
    for (int i = 0; i < g_nt; ++i) pthread_create(&th[i], nullptr, worker, (void *) (intptr_t) i);
    for (int i = 0; i < g_nt; ++i) pthread_join(th[i], nullptr);
    const double dt = now_s() - t0;
    const int st = getenv("MEGA_STRIDE") ? atoi(getenv("MEGA_STRIDE")) : 1;
    printf("  nt=%-2d stride %d  %d MB in %.3f s  ->  %.1f GB/s used (aggregate, 3 passes)\n",
           g_nt, st, mb, dt, (double) g_bytes * 3 / st / dt / 1e9);
    return 0;
}
