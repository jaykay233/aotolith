// What exactly does quantize_a_4row_i8_hp write, and can a plain RVV loop
// reproduce it byte for byte?
//
// The hp record is 1160 B per 4-row group per 256 k: eight subblocks of
// [4 x fp16 row scale][4 x 32 int8], then a 72 B trailer. That much is in
// CHAIN.md; what is not written down is the rounding the scales and the int8 use,
// and what the trailer's 32 values actually are. Both decide whether a faster
// quantiser is a free swap or a numerics change, so this probe reads the vendor
// record back out and compares it against a candidate implementation computed
// here.
//
// The candidate is the obvious one: per row, max |x| over the subblock, scale =
// that / 127 rounded to fp16, q = round-half-even(x / scale) clamped to +-127,
// trailer = the integer sum of q per subblock per row.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <unistd.h>
#include <sched.h>
#include <riscv_vector.h>

#include "ggml_stubs.h"

namespace spacemit_kernels { namespace rvv {
void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k,
                           uint8_t * quant_a_ptr);
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k,
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

static const int ROWS = 4;

// Candidate implementation: one subblock at a time, RVV where it is exact.
// Returns the same 136 B subblock plus its four i16 sums.
static void cand_subblock(const float * x, int64_t k, int64_t k0,
                          uint8_t * sub, int16_t * sums) {
    _Float16 sc[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        const float * p = x + (int64_t) r * k + k0;
        float mx = 0;
        for (int i = 0; i < 32; ++i) mx = fmaxf(mx, fabsf(p[i]));
        sc[r] = (_Float16) (mx / 127.0f);
    }
    memcpy(sub, sc, sizeof sc);
    for (int r = 0; r < ROWS; ++r) {
        const float * p = x + (int64_t) r * k + k0;
        const float inv = sc[r] != (_Float16) 0 ? 1.0f / (float) sc[r] : 0.0f;
        int32_t s = 0;
        int8_t * q = (int8_t *) sub + 8 + r * 32;
        for (int i = 0; i < 32; ++i) {
            float v = nearbyintf(p[i] * inv);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[i] = (int8_t) v;
            s += q[i];
        }
        sums[r] = (int16_t) s;
    }
}

int main(int argc, char ** argv) {
    int64_t K = 2048;
    int reps = 8;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k")) K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-r")) reps = atoi(argv[++i]);
    }
    bind_cpu(8);
    const int64_t nsb = K / 32;          // subblocks of 32 k
    const int64_t nsup = K / 256;        // 256-k superblocks
    const size_t rec = (size_t) nsup * 1160;
    float * x = (float *) aligned_alloc(64, (size_t) ROWS * K * 4 + 64);
    uint8_t * va = (uint8_t *) aligned_alloc(64, rec + 64);
    uint8_t * ca = (uint8_t *) aligned_alloc(64, rec + 64);
    if (!x || !va || !ca) { printf("oom\n"); return 1; }

    // Realistic activations: per-row scale spread plus a few large outliers, the
    // shape post-rmsnorm activations have.
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int r = 0; r < ROWS; ++r) {
        const float s = 0.4f + 0.6f * (float) r;
        for (int64_t i = 0; i < K; ++i) x[r * K + i] = nd(rng) * s;
        for (int t = 0; t < 3; ++t) x[r * K + (rng() % K)] *= 6.0f;
    }

    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x, (size_t) K, va);
    memset(ca, 0, rec);
    int16_t sums[ROWS];
    for (int64_t sb = 0; sb < nsb; ++sb) {
        const int64_t sup = sb / 8, sub = sb % 8;
        uint8_t * dst = ca + (size_t) sup * 1160 + sub * 136;
        cand_subblock(x, K, sb * 32, dst, sums);
        int16_t * tr = (int16_t *) (ca + (size_t) sup * 1160 + 1088) + sub * ROWS;
        memcpy(tr, sums, sizeof sums);
    }

    // Where do the two records differ?
    int diff = 0;
    for (size_t i = 0; i < rec; ++i)
        if (va[i] != ca[i]) ++diff;
    printf("K=%lld  record %zu B   vendor vs candidate: %d differing bytes of %zu\n",
           (long long) K, rec, diff, rec);

    // Decode the vendor record on its own terms and report what it implies.
    int bad_scale = 0, bad_q = 0, bad_sum = 0, have_sum = 0;
    double worst_q = 0, worst_sum = 0;
    for (int64_t sb = 0; sb < nsb; ++sb) {
        const int64_t sup = sb / 8, sub = sb % 8;
        const uint8_t * src = va + (size_t) sup * 1160 + sub * 136;
        _Float16 sc[ROWS];
        memcpy(sc, src, sizeof sc);
        const int16_t * tr = (const int16_t *) (va + (size_t) sup * 1160 + 1088) + sub * ROWS;
        for (int r = 0; r < ROWS; ++r) {
            const float * p = x + (int64_t) r * K + sb * 32;
            float mx = 0;
            for (int i = 0; i < 32; ++i) mx = fmaxf(mx, fabsf(p[i]));
            const _Float16 want = (_Float16) (mx / 127.0f);
            if ((float) want != (float) sc[r]) ++bad_scale;
            const int8_t * q = (const int8_t *) src + 8 + r * 32;
            const float inv = (float) sc[r] != 0 ? 1.0f / (float) sc[r] : 0.0f;
            int32_t s = 0;
            for (int i = 0; i < 32; ++i) {
                s += q[i];
                float v = nearbyintf(p[i] * inv);
                if (v > 127) v = 127; if (v < -127) v = -127;
                const double e = fabs((double) q[i] - (double) v);
                if (e > 0.5) { ++bad_q; if (e > worst_q) worst_q = e; }
            }
            if (tr[r] != 0) ++have_sum;
            const double es = fabs((double) tr[r] - (double) s);
            if (es > 0.5) { ++bad_sum; if (es > worst_sum) worst_sum = es; }
        }
    }
    printf("vendor decode: scales differing from max/127 -> %d, int8 differing from "
           "round(x/scale) -> %d (worst %.3f)\n", bad_scale, bad_q, worst_q);
    printf("trailer: %d of %lld non-zero entries, sums differing from sum(q) -> %d "
           "(worst %.1f)\n", have_sum, (long long) nsb * ROWS, bad_sum, worst_sum);

    // Speed: vendor against candidate, one core, same buffer.
    double tv = 1e9, tc = 1e9;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x, (size_t) K, va);
        double d = now_s() - t0; if (d < tv) tv = d;
        t0 = now_s();
        for (int64_t sb = 0; sb < nsb; ++sb) {
            const int64_t sup = sb / 8, sub = sb % 8;
            cand_subblock(x, K, sb * 32, ca + (size_t) sup * 1160 + sub * 136, sums);
        }
        d = now_s() - t0; if (d < tc) tc = d;
    }
    const double elems = (double) ROWS * (double) K;
    printf("vendor    %.3f us  %.1f M elem/s/core\n", tv * 1e6, elems / tv / 1e6);
    printf("candidate %.3f us  %.1f M elem/s/core  (%.2fx)\n",
           tc * 1e6, elems / tc / 1e6, tv / tc);
    return 0;
}
