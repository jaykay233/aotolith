// The vendor quantiser runs at 1277 M elem/s per core on a 32 KB array that sits
// in L1. In prefill it delivers about 8.5 GB/s across eight cores, where the board
// sustains 19.3 GB/s. This probe asks whether that gap is the streaming shape or
// the call structure: same function, same record, but a buffer the size prefill
// actually walks, and a thread split that matches task_pf_quant's.
//
//   -b  bytes of source to walk per call (rows * k * 4)
//   -t  threads, each striding over its own groups the way task_pf_quant does
//   -g  groups per thread between stride jumps (1 = g += nthreads, as production)
//
// Reported in GB/s counting source read plus record write, which is what the
// kernel's cost is.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#include "ggml_stubs.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k,
                           uint8_t * quant_a_ptr);
}}

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}
static void open_a100(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
}

static int64_t K = 5632, ROWS = 2560;
static int NT = 8, REPS = 5, GRP = 1;
static float * gsrc;
static uint8_t * gdst;
static int64_t g_rowstride;
static double g_t[16];

struct Job { int tid; };

static void * th(void * p) {
    Job * j = (Job *) p;
    bind_cpu(8 + j->tid);
    const int64_t mg = ROWS / 4;
    double best = 1e9;
    for (int rep = 0; rep < REPS; ++rep) {
        const double t0 = now_s();
        // Same order task_pf_quant uses: thread tid takes group tid, tid+NT, ...
        for (int64_t g = j->tid; g < mg; g += NT * GRP)
            for (int64_t q = 0; q < GRP && g + q * NT < mg; ++q) {
                const int64_t gg = g + q * NT;
                spacemit_kernels::rvv::quantize_a_4row_i8_hp(
                    256, gsrc + gg * 4 * K, (size_t) K, gdst + gg * g_rowstride);
            }
        const double d = now_s() - t0;
        if (rep >= 1 && d < best) best = d;
    }
    g_t[j->tid] = best;
    return nullptr;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k")) K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-r")) ROWS = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-t")) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")) GRP = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) REPS = atoi(argv[++i]);
    }
    open_a100();
    const size_t srcb = (size_t) ROWS * (size_t) K * 4;
    const int64_t per = (K / 256) * 1160;
    g_rowstride = per;
    gsrc = (float *) aligned_alloc(4096, srcb + 4096);
    gdst = (uint8_t *) aligned_alloc(4096, (size_t) (ROWS / 4) * (size_t) per + 4096);
    if (!gsrc || !gdst) { printf("oom\n"); return 1; }
    for (size_t i = 0; i < srcb / 4; ++i) gsrc[i] = (float) ((i * 37) % 251) * 0.01f - 1.2f;
    memset(gdst, 0, (size_t) (ROWS / 4) * (size_t) per);

    pthread_t t[16];
    Job jobs[16] = {};
    for (long i = 0; i < NT; ++i) { jobs[i].tid = (int) i; pthread_create(&t[i], nullptr, th, &jobs[i]); }
    for (int i = 0; i < NT; ++i) pthread_join(t[i], nullptr);
    double tm = 0;
    for (int i = 0; i < NT; ++i) if (g_t[i] > tm) tm = g_t[i];

    const double elems = (double) ROWS * (double) K;
    const double bytes = elems * 4.0 + (double) (ROWS / 4) * (double) per;
    printf("k=%-5lld rows=%-5lld thr=%-2d grp=%d: %.4f s  %.1f GB/s  %.1f M elem/s/core\n",
           (long long) K, (long long) ROWS, NT, GRP, tm, bytes / tm / 1e9,
           elems / tm / 1e6 / NT);
    return 0;
}
