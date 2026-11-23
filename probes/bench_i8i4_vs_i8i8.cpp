// Same shapes, same A handling, only the weight format changes: int4 (Q4_0
// repack, 18 B per 32-column block) versus int8 (34 B). The i8i8 kernel has a
// much shorter inner loop (no vnpack4/vupack unpacking), and the GEMM is not
// DRAM bound here (weights stream at ~7 GB/s against a 26 GB/s ceiling), so the
// wider format may still win. A tile lives in TCM exactly like the megakernel.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>

#include "spine_tcm.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i8(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

extern "C" {
size_t   ggml_type_size(int) { return 0; }
long long ggml_nelements(const void *) { return 0; }
long long ggml_nrows(const void *) { return 0; }
bool     ggml_can_repeat(const void *, const void *) { return false; }
bool     ggml_are_same_shape(const void *, const void *) { return false; }
bool     ggml_is_contiguous_rows(const void *) { return true; }
bool     ggml_is_contiguous(const void *) { return true; }
bool     ggml_is_transposed(const void *) { return false; }
size_t   ggml_element_size(const void *) { return 1; }
void ggml_abort(const char * f, int l, const char * m, ...) {
    fprintf(stderr, "ggml_abort %s:%d\n", f, l); abort();
}
}

using namespace spacemit_kernels;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    spine_tcm_open_handle(nullptr);
    uint8_t * tcm = (uint8_t *) spine_tcm_mem_get(0);
    spine_tcm_mem_info_t mi;
    int64_t tcm_size = 0;
    if (spine_tcm_mem_info(&mi) == 0) tcm_size = (int64_t) mi.blk_size;
    printf("tcm buffer %p size %lld\n", (void *) tcm, (long long) tcm_size);

    // q/k/v: 2048->2048, o: 2048->2048, gate/up: 2048->6144, down: 6144->2048
    struct { const char * name; int k, n; } cls[] = {
        {"qkv", 2048, 2048}, {"o", 2048, 2048}, {"up", 2048, 6144}, {"down", 6144, 2048},
    };
    const int64_t RT = 32;   // rows per tile, as in the megakernel

    for (int ci = 0; ci < 4; ++ci) {
        const int k = cls[ci].k, n = cls[ci].n;
        const int64_t kb = k / 32;
        const int64_t b4_col = kb * 18, b8_col = kb * 34;
        uint8_t * b4 = (uint8_t *) aligned_alloc(4096, (size_t) n * b4_col);
        uint8_t * b8 = (uint8_t *) aligned_alloc(4096, (size_t) n * b8_col);
        // fill so both look like plausible weights
        for (int64_t i = 0; i < (int64_t) n * b4_col; ++i) b4[i] = (uint8_t) (i * 37);
        for (int64_t i = 0; i < (int64_t) n * b8_col; ++i) b8[i] = (uint8_t) (i * 37);
        for (int64_t c = 0; c < n; ++c) {   // f16 scale fields
            for (int64_t x = 0; x < kb; ++x) {
                *(_Float16 *) (b4 + c * b4_col + x * 18) = (_Float16) 0.01f;
                *(_Float16 *) (b8 + c * b8_col + x * 34) = (_Float16) 0.01f;
            }
        }

        // activations for 4 rows, quantized once, A tile in TCM
        static float act[4 * 8192];
        for (int r = 0; r < 4; ++r)
            for (int kk = 0; kk < k; ++kk) act[r * k + kk] = sinf(0.003f * (float) (kk + 11 * r)) * 2.0f;
        const int64_t astride = kb * 152;
        uint8_t * aq = (uint8_t *) (tcm && (int64_t) 4 * astride <= tcm_size ? tcm : aligned_alloc(64, (size_t) 4 * astride));
        memset(aq, 0, (size_t) 4 * astride);
        rvv::quantize_a_4row_i8(32, act, (size_t) k, aq);

        static float c[6144 * RT];
        memset(c, 0, sizeof c);

        const int64_t nb = n / 32;
        const int passes = 6;
        const double macs = (double) passes * (double) RT * (double) k * (double) n;

        // i8i4
        for (int r = 0; r < 2; ++r)
            for (int64_t a = 0; a < nb; ++a)
                for (int64_t g = 0; g < RT / 4; ++g)
                    ime2::gemm_kernel_i8i4(32, aq, b4 + a * b4_col, nullptr,
                                           c + g * 4 * n + a * 32, 4, 32, (size_t) kb, (size_t) n);
        double t0 = now_s();
        for (int p = 0; p < passes; ++p)
            for (int64_t a = 0; a < nb; ++a)
                for (int64_t g = 0; g < RT / 4; ++g)
                    ime2::gemm_kernel_i8i4(32, aq, b4 + a * b4_col, nullptr,
                                           c + g * 4 * n + a * 32, 4, 32, (size_t) kb, (size_t) n);
        const double g4 = macs / (now_s() - t0) / 1e9;

        // i8i8
        for (int r = 0; r < 2; ++r)
            for (int64_t a = 0; a < nb; ++a)
                for (int64_t g = 0; g < RT / 4; ++g)
                    ime2::gemm_kernel_i8i8(32, aq, b8 + a * b8_col, nullptr,
                                           c + g * 4 * n + a * 32, 4, 32, (size_t) kb, (size_t) n);
        t0 = now_s();
        for (int p = 0; p < passes; ++p)
            for (int64_t a = 0; a < nb; ++a)
                for (int64_t g = 0; g < RT / 4; ++g)
                    ime2::gemm_kernel_i8i8(32, aq, b8 + a * b8_col, nullptr,
                                           c + g * 4 * n + a * 32, 4, 32, (size_t) kb, (size_t) n);
        const double g8 = macs / (now_s() - t0) / 1e9;

        printf("%-5s k=%4d n=%4d  i8i4 %6.1f GMAC/s   i8i8 %6.1f GMAC/s   (%+.0f%%)  A_in_%s\n",
               cls[ci].name, k, n, g4, g8, 100.0 * (g8 / g4 - 1.0), aq == tcm ? "TCM" : "heap");
        free(b4);
        free(b8);
        if (aq != tcm) free(aq);
    }
    return 0;
}
