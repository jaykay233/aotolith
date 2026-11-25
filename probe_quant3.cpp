// Is a 4-row hp record for one K256 superblock independent of how many superblocks
// the call covers? The fused silu+quant staging path depends on it: it hands the
// vendor kernel 1024 k at a time and places each slice's records by superblock
// index. If a slice's bytes are not a prefix-preserving copy of the whole-width
// record, that placement is wrong and the fused path must be dropped.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>

#include "ggml_stubs.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k,
                           uint8_t * quant_a_ptr);
}}

static int bind_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

int main(int argc, char ** argv) {
    int64_t K = 6144, SLICE = 1024;
    if (argc > 1) K = atoll(argv[1]);
    if (argc > 2) SLICE = atoll(argv[2]);
    {
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { ssize_t w = write(fd, "0", 1); (void) w; close(fd); }
    }
    bind_cpu(8);
    const int64_t rec = (K / 256) * 1160;
    float * x = (float *) aligned_alloc(64, (size_t) 4 * K * 4 + 256);
    uint8_t * whole = (uint8_t *) aligned_alloc(4096, (size_t) rec + 4096);
    uint8_t * sliced = (uint8_t *) aligned_alloc(4096, (size_t) rec + 4096);
    memset(whole, 0xAA, (size_t) rec);
    memset(sliced, 0xAA, (size_t) rec);
    for (int64_t i = 0; i < 4 * K; ++i) x[i] = sinf((float) i * 0.017f) * 3.1f;

    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x, (size_t) K, whole);

    for (int64_t k0 = 0; k0 < K; k0 += SLICE) {
        const int64_t kn = (K - k0 < SLICE) ? (K - k0) : SLICE;
        // stage the slice the way the fused band does: 4 rows, stride kn
        static float stage[4 * 4096];
        for (int r = 0; r < 4; ++r)
            memcpy(stage + r * kn, x + r * K + k0, (size_t) kn * 4);
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, stage, (size_t) kn,
                                                     sliced + (k0 / 256) * 1160);
    }

    int diff = 0, first = -1;
    for (int64_t i = 0; i < rec; ++i)
        if (whole[i] != sliced[i]) { if (first < 0) first = (int) i; ++diff; }
    printf("K=%lld slice=%lld rec=%lld: %d of %lld bytes differ", (long long) K,
           (long long) SLICE, (long long) rec, diff, (long long) rec);
    if (diff) {
        const int64_t sb = first / 1160, off = first % 1160;
        printf("  first at %d (superblock %lld, offset %lld)", first, (long long) sb, (long long) off);
    }
    printf("\n");
    return 0;
}
