// Is the 2-bit B packing right?
//
// repack_b_q2 has to reproduce llama.cpp's block_q2_K semantics inside the
// vendor's nrow_block_q2_k<32> interleave, and the only authority for that
// interleave is repack_q2_k_to_q2_k_32_bl in the SpacemiT ggml fork. So this
// probe quantises a synthetic 32 x 256 tile with the same math, packs it, runs
// gemm_kernel_i8i2k against a quantised A row, and compares with a float dot of
// the same tile. A layout mistake of any kind shows up as a huge relative error
// rather than as plausible-looking logits.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

// IME2 only exists on the eight A100 cores (8-15); anywhere else the vmadot in
// the kernel is an illegal instruction.
static void bind_a100(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(8, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
size_t gemm_kernel_i8i2k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                         float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
}}

// Stubs for the ggml entry points rvv.o and ime2.o were compiled against.
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
}

// The kernel objects were built for the ggml backend and reference its abort.
extern "C" void ggml_abort(const char * f, int l, const char * m, ...) {
    fprintf(stderr, "ggml_abort %s:%d %s\n", f, l, m);
    abort();
}

static const int Q2_K = 256;
static const size_t Q2_BLK = 32 * 16 + 32 * 64 + 32 * 2 + 32 * 2;   // 2688

static int g_layout = 0;   // 0 = split planes, 1 = both nibbles in the scale byte
static int g_sign   = 0;   // dmin sign
static int g_qsrev = 0;

// Same math as repack_b_q2, straight from a float tile.
static void pack(float * W, uint8_t * blk) {
    uint8_t *  scl = blk;
    _Float16 * s16 = (_Float16 *) (blk + 512 + 2048);
    _Float16 * z16 = (_Float16 *) (blk + 512 + 2048 + 64);
    memset(scl, 0, 512);
    memset(blk + 512, 0, 2048);
    for (int i = 0; i < 32; ++i) {
        float step[16], zoff[16], smax = 0, zmax = 0;
        for (int j = 0; j < 16; ++j) {
            float mn = W[i * 256 + j * 16], mx = mn;
            for (int b2 = 1; b2 < 16; ++b2) {
                const float v = W[i * 256 + j * 16 + b2];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (mn > 0) mn = 0;
            if (mx < 0) mx = 0;
            step[j] = (mx - mn) / 3.0f;
            zoff[j] = -mn;
            if (step[j] > smax) smax = step[j];
            if (zoff[j] > zmax) zmax = zoff[j];
        }
        const float d  = smax > 0 ? smax / 15.0f : 0.0f;
        const float dm = zmax > 0 ? zmax / 15.0f : 0.0f;
        s16[i] = (_Float16) d;
        z16[i] = (_Float16) (g_sign ? -dm : dm);
        for (int j = 0; j < 16; ++j) {
            int ns = d  > 0 ? (int) lrintf(step[j] / d)  : 0;
            int nz = dm > 0 ? (int) lrintf(zoff[j] / dm) : 0;
            if (ns < 0) ns = 0; if (ns > 15) ns = 15;
            if (nz < 0) nz = 0; if (nz > 15) nz = 15;
            if (g_layout == 0) {
                scl[j * 32 + i] |= (uint8_t) ns;
                scl[16 * i + j] |= (uint8_t) (nz << 4);
            } else {
                scl[j * 32 + i] = (uint8_t) (ns | (nz << 4));
            }
            const float eff = step[j];
            for (int b2 = 0; b2 < 16; ++b2) {
                const float v = W[i * 256 + j * 16 + b2] + zoff[j];
                int q2 = eff > 0 ? (int) lrintf(v / eff) : 0;
                if (q2 < 0) q2 = 0;
                if (q2 > 3) q2 = 3;
                int jj = g_qsrev ? (15 - j) : j;
                uint8_t * byte = blk + 512 + (int64_t) (jj / 4) * 32 * 16 + i * 16 + b2;
                *byte |= (uint8_t) (q2 << ((jj % 4) * 2));
            }
        }
    }
}

int main(int argc, char ** argv) {
    g_layout = argc > 1 ? atoi(argv[1]) : 0;
    g_sign   = argc > 2 ? atoi(argv[2]) : 0;
    g_qsrev  = argc > 3 ? atoi(argv[3]) : 0;

    bind_a100();
    static float W[32 * 256], A[256];
    uint32_t s = 12345;
    for (int i = 0; i < 32 * 256; ++i) {
        s = s * 1664525u + 1013904223u;
        W[i] = ((float) (s >> 8) / 16777216.0f - 0.5f) * 0.08f;   // N(0,~0.023)
    }
    for (int i = 0; i < 256; ++i) {
        s = s * 1664525u + 1013904223u;
        A[i] = ((float) (s >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }

    static uint8_t B[Q2_BLK];
    memset(B, 0, sizeof(B));
    pack(W, B);

    static uint8_t AQ[292];
    spacemit_kernels::rvv::quantize_a_row_i8k(256, A, 256, AQ);

    static float c[32];
    spacemit_kernels::ime2::gemm_kernel_i8i2k(256, AQ, B, nullptr, c, 1, 32, 1, 32);

    // Reference: the tile the packer itself reconstructed (so quantisation error
    // cancels and only layout error remains), and the raw float dot.
    double num = 0, den = 0, num2 = 0, den2 = 0;
    for (int i = 0; i < 32; ++i) {
        double ref = 0, refq = 0;
        for (int k = 0; k < 256; ++k) ref += (double) A[k] * W[i * 256 + k];
        // dequant from the packed bytes the same way the vendor kernel reads them
        for (int j = 0; j < 16; ++j) {
            int sb = j;
            int ns, nz;
            if (g_layout == 0) { ns = B[sb * 32 + i] & 0xF; nz = B[16 * i + sb] >> 4; }
            else               { ns = B[sb * 32 + i] & 0xF; nz = B[sb * 32 + i] >> 4; }
            const double d  = (double) ((_Float16 *) (B + 512 + 2048))[i];
            const double dm = (double) ((_Float16 *) (B + 512 + 2048 + 64))[i];
            for (int b2 = 0; b2 < 16; ++b2) {
                int jj = g_qsrev ? (15 - j) : j;
                const uint8_t byte = B[512 + (jj / 4) * 512 + i * 16 + b2];
                const int q = (byte >> ((jj % 4) * 2)) & 3;
                refq += (double) A[j * 16 + b2] * (d * ns * q - dm * nz);
            }
        }
        num += (c[i] - ref) * (c[i] - ref); den += ref * ref;
        num2 += (c[i] - refq) * (c[i] - refq); den2 += refq * refq;
    }
    printf("layout %d sign %d qsrev %d | vs float dot: rel %.4f | vs dequant: rel %.6f\n",
           g_layout, g_sign, g_qsrev, sqrt(num / den), sqrt(num2 / den2));
    printf("  c[0..3] %.5f %.5f %.5f %.5f\n", c[0], c[1], c[2], c[3]);
    return 0;
}
