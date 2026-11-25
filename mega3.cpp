// Attention rewrite: position-split across the 8 A100 threads, K stored
// dim-major, V stored position-major.
//
// Why this shape:
//  * The previous version gave each thread two query heads, so the same KV head
//    was streamed 8 times per layer (GQA 8:1). At 10K that is 3.5 GB of redundant
//    traffic per token, i.e. >130 ms. Splitting positions instead means each KV
//    element is read exactly once.
//  * QK needs sum_d q_d k_pd for many p. With K dim-major a 32-position block is
//    contiguous, so the reduction happens across lanes with no per-position
//    horizontal reduce (vredsum is microcoded and was what made attention 40x
//    slow before).
//  * PV needs out[d] += sum_p w_p v_pd. With V position-major the 64-lane
//    accumulator per head lets all 16 heads stay resident (2 x m2 x 16 = 4 KB),
//    and each v element is read once.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <pthread.h>
#include <atomic>
#include <sys/mman.h>
#include "spine_tcm.h"
#include <sys/stat.h>
#include <riscv_vector.h>
#include <riscv_vector.h>

// ---------------------------------------------------------------- vendor kernels
namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i4_hp(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                           float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i4_hp_m4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                              float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i3k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                         float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i2k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
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
    fprintf(stderr, "ggml_abort %s:%d %s\n", f, l, m); abort();
}
}

// ------------------------------------------------------------------- model shape
static const int N_LAYER = 42, N_EMBD = 2048, N_FF = 6144, N_HEAD = 16, N_KV = 2, HEAD_DIM = 128;// 12288, not 10240. A "10K" prompt is 10241 ids in practice, so a 10240-slot
// cache has its last prompt position one past the end: the V cache is
// [layer][pos][kv][dim] with a MAXCTX stride, so that position lands on the next
// layer's position 0, and the rope table's last entry runs 128 floats off the
// end. Every 10K number taken before this was measured on a corrupted cache.
static const int N_GPU = 8, MAXCTX = 12288, VOCAB = 130560;
static const int NH_PER_KV = N_HEAD / N_KV;
static const float RMS_EPS = 1e-6f;
static const float ROPE_BASE = 5000000.0f;

// --------------------------------------------------------------- stage profiling
enum Stage { S_EMB, S_NORM, S_QUANT, S_GEMM, S_ATTN, S_SILU, S_ROPE, S_HEAD, S_MISC, S_NSTAGE };
static double g_stage[S_NSTAGE];
static double g_snap[S_NSTAGE];
static double g_snap_rope[2];
static const char * g_stage_name[S_NSTAGE] = {
    "embed", "rmsnorm", "quantA", "gemm", "attn", "silu", "rope", "lm_head+argmax", "misc" };

// noinline + a memory clobber: with the call inlined, -O2 merged the TICK and
// TOCK reads of the clock into one call (the sysroot declares clock_gettime in a
// way GCC treats as loop-invariant), so every stage accumulator stayed exactly
// zero while the total time kept being right.
static __attribute__((noinline)) double now_s(void) {
    struct timespec t;
    __asm__ __volatile__("" ::: "memory");
    clock_gettime(CLOCK_MONOTONIC, &t);
    __asm__ __volatile__("" ::: "memory");
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#define TICK_DECL double _t0
#define TICK() _t0 = now_s()
#define TOCK(st) g_stage[st] += now_s() - _t0

static void use_huge(void * p, size_t n) { if (p) madvise(p, n, MADV_HUGEPAGE); }

// THP collapse: madvise(MADV_HUGEPAGE) only makes an area eligible, and the
// kernel splits 2 MB pages back into 4 KB ones when memory gets tight (the 2.2
// GB int8 weight copy did exactly that). A 1 GB streaming read at 4 KB pages
// loses about a quarter of its bandwidth to TLB misses, which is why the same
// blob measured 42 ms and 54 ms per token on different runs. Collapsing after
// the big temporary is gone makes the number repeatable.
static void huge_collapse(void * p, size_t n) {
    if (!p || !n) return;
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
    for (size_t off = 0; off < n; off += (2 << 20))
        madvise((uint8_t *) p + off, (2 << 20) < n - off ? (2 << 20) : n - off, MADV_COLLAPSE);
}

static void report_huge(const char * tag) {
    FILE * f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "AnonHugePages:", 14)) { printf("  [%s]%s", tag, line); break; }
    fclose(f);
}

// exp(x) for x <= 0, vectorised: 2^f on [0,1) plus an integer exponent field.
static inline vfloat32m1_t vexp_neg(vfloat32m1_t x, size_t vl) {
    vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, 1.4426950408889634f, vl);
    t = __riscv_vfmax_vf_f32m1(t, -126.0f, vl);
    vint32m1_t ie = __riscv_vfcvt_x_f_v_i32m1(t, vl);
    vfloat32m1_t ei = __riscv_vfcvt_f_x_v_f32m1(ie, vl);
    vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, ei, vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t ex = __riscv_vadd_vx_i32m1(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m1(ex, 23, vl);
    return __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_i32m1_f32m1(ex), vl);
}

// ------------------------------------------------------------------ gguf reading
static uint8_t * g_map; static size_t g_map_sz;

static uint64_t rd_q(const uint8_t ** p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static uint32_t rd_u32(const uint8_t ** p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }

static void skip_kv(const uint8_t ** p, uint32_t t) {
    if (t == 8) { const uint64_t n = rd_q(p); *p += n; }
    else if (t == 9) {
        const uint32_t et = rd_u32(p); const uint64_t n = rd_q(p);
        for (uint64_t j = 0; j < n; ++j) {
            if (et == 8) { const uint64_t sl = rd_q(p); *p += sl; }
            else if (et == 6 || et == 4 || et == 5) *p += 4;
            else if (et == 10 || et == 11 || et == 12) *p += 8;
            else if (et == 0 || et == 1 || et == 7) *p += 1;
            else if (et == 2 || et == 3) *p += 2;
            else { fprintf(stderr, "array elem %u\n", et); exit(1); }
        }
    }
    else if (t == 0 || t == 1 || t == 7) *p += 1;
    else if (t == 2 || t == 3) *p += 2;
    else if (t == 4 || t == 5 || t == 6) *p += 4;
    else if (t == 10 || t == 11 || t == 12) *p += 8;
    else { fprintf(stderr, "kv type %u\n", t); exit(1); }
}

struct Entry { char name[64]; const uint8_t * data; int64_t k, n; uint32_t type; };
static Entry * g_entries; static int g_n_entries;

static void gguf_index(void) {
    const uint8_t * p = g_map + 4; rd_u32(&p);
    const uint64_t nt = rd_q(&p), nkv = rd_q(&p);
    for (uint64_t i = 0; i < nkv; ++i) {
        const uint64_t kl = rd_q(&p); p += kl;
        const uint32_t kt = rd_u32(&p); skip_kv(&p, kt);
    }
    const uint8_t * hdr = p;
    const uint8_t * q = p;
    for (uint64_t i = 0; i < nt; ++i) {
        const uint64_t nl = rd_q(&q); q += nl;
        const uint32_t nd = rd_u32(&q);
        for (uint32_t d = 0; d < nd; ++d) rd_q(&q);
        rd_u32(&q); rd_q(&q);
    }
    const uint64_t data_off = ((uint64_t) (q - g_map) + 31) / 32 * 32;
    g_entries = (Entry *) calloc(nt, sizeof(Entry));
    g_n_entries = (int) nt;
    p = hdr;
    for (uint64_t i = 0; i < nt; ++i) {
        const uint64_t nl = rd_q(&p);
        Entry & e = g_entries[i];
        const uint64_t cp = nl < sizeof(e.name) - 1 ? nl : sizeof(e.name) - 1;
        memcpy(e.name, p, cp); e.name[cp] = 0;
        p += nl;
        const uint32_t nd = rd_u32(&p);
        uint64_t dims[4] = {1,1,1,1};
        for (uint32_t d = 0; d < nd && d < 4; ++d) dims[d] = rd_q(&p);
        e.type = rd_u32(&p);
        const uint64_t off = rd_q(&p);
        e.data = g_map + data_off + off;
        e.k = (int64_t) dims[0]; e.n = (int64_t) dims[1];
    }
}
static const Entry * find(const char * nm) {
    for (int i = 0; i < g_n_entries; ++i) if (strcmp(g_entries[i].name, nm) == 0) return &g_entries[i];
    return nullptr;
}

// ------------------------------------------------------------------ weight repack
// 3-bit weights. The vendor i8i3k kernel consumes ggml's Q3_K superblock layout
// (verified against its own reference implementation):
//   int8 scales[16][32], uint16 hmask[16][32], uint8 qs[4][32][16], fp16 scales16[32]
// per 32 output columns x 256 K elements, i.e. 3.5625 bit/param versus Q4_0's
// 4.5. Each stored value is a signed 3-bit integer in [-4,3] whose sign is
// carried by the hmask bit, so the effective weight is
//   q3 * int8_scale[row][k16] * fp16_scale[row]
// (no extra 1/16: that factor lives in the Q3_K reference path only).
static int g_bit3 = 0;
// 2-bit weights on the same K256 superblock as the 3-bit path, laid out as the
// vendor i8i2k kernel's nrow_block_q2_k<32>:
//   uint8 scales[16][32]   low nibble = scale nibble, high nibble = zero-point
//                          nibble, both indexed [sub-block][row]
//   uint8  qs[32][64]      [sub-block/4][byte within 16][row], 2-bit fields at
//                          shift 2*(sub-block%4)
//   fp16   scales16[32], zeros16[32]
// per 32 output rows x 256 K = 2688 B, i.e. 2.625 bit per weight against Q3_K's
// 3.5625 and Q4_0's 4.5.
static int g_bit2 = 0;
// Decode with the vendor K256 hp kernel, i.e. the same arithmetic llama.cpp's
// spacemit backend runs for Q4_0: its kernel table maps Q4_0 to
// q4_0_32x256_q8_0, which is gemm_kernel_i8i4_hp, and that kernel's A operand is
// the hp record (one fp16 superblock scale, one fp16 sum per K32 subblock, 290 B
// per 256 k for a single row) rather than the K32 m1 record the old decode path
// fed it. A one-row A record is 290 B: 256 int8 + 8*2 subblock sums + 2 blk scale.
static int g_hpdec = 0;
static void dequant_row(float * dst, const uint8_t * row, int64_t k);
// Which ggml type the weight tensors in this file actually are. Decode streams
// every weight byte, so the byte count is the whole cost: Q2_K is 2.625 bit and
// Q3_K 3.5625 against Q4_0's 4.5, and at the measured 25 GB/s stream ceiling that
// is 31 or 41 ms of GEMM instead of 53. Feeding a real Q2_K/Q3_K file also means
// the reference for alignment is llama.cpp running the same file.
static int g_src_type = 2;                    // ggml type id, 2 = Q4_0
static int g_src_mixed = 0;
static inline float f16_to_f32(const uint8_t * p) {
    _Float16 h; memcpy(&h, p, 2); return (float) h;
}
static int64_t src_row_bytes(int64_t k) {
    switch (g_src_type) {
        case 0:  return k * 4;                // F32
        case 1:  return k * 2;                // F16
        case 2:  return k / 32 * 18;          // Q4_0
        case 8:  return k / 32 * 34;          // Q8_0
        case 10: return k / 256 * 84;         // Q2_K
        case 11: return k / 256 * 110;        // Q3_K
        case 12: return k / 256 * 144;        // Q4_K
        case 13: return k / 256 * 176;        // Q5_K
        case 14: return k / 256 * 210;        // Q6_K
        default: return k / 32 * 18;
    }
}
static int64_t src_row_bytes_t(int64_t k, int t) {
    const int save = g_src_type;
    g_src_type = t;
    const int64_t r = src_row_bytes(k);
    g_src_type = save;
    return r;
}
static const int Q2_K = 256;
static const size_t Q2_BLK = 32 * 16 + 32 * 64 + 32 * 2 + 32 * 2;   // 2688 B
static const int Q3_K = 256;                 // K elements per superblock
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;   // 3648 B
static size_t g_blk_bytes = 576;

__attribute__((noinline))
static void repack_b_q3(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t rs = src_row_bytes(k), nsup = k / Q3_K;
    const int64_t sbytes = rs / nsup;
    static float tile[32][Q3_K];
    for (int64_t g = 0; g < rows; g += 32) {
        const int64_t grp = g / 32;
        for (int64_t sup = 0; sup < nsup; ++sup) {
            uint8_t * blk = dst + (grp * nsup + sup) * Q3_BLK;
            int8_t *   scl = (int8_t *) blk;
            uint16_t * hms = (uint16_t *) (blk + 512);
            uint8_t *  qs  = blk + 512 + 1024;
            _Float16 * s16 = (_Float16 *) (blk + 512 + 1024 + 2048);
            // dequantize this 32 x 256 tile out of whatever the file stores
            for (int i = 0; i < 32; ++i) {
                const int64_t sr = perm ? (int64_t) perm[g + i] : (g + i);
                dequant_row(tile[i], src + sr * rs + sup * sbytes, Q3_K);
            }
            // per (row, k16) step, then a per-row fp16 normalisation so the int8
            // scale keeps full resolution
            float step[32][16], smax[32] = {0};
            for (int i = 0; i < 32; ++i) {
                for (int kii = 0; kii < 16; ++kii) {
                    float mp = 0, mn = 0;
                    for (int bi = 0; bi < 16; ++bi) {
                        const float v = tile[i][kii * 16 + bi];
                        if (v > mp) mp = v;
                        if (-v > mn) mn = -v;
                    }
                    step[i][kii] = fmaxf(mp / 3.0f, mn / 4.0f);
                    if (step[i][kii] > smax[i]) smax[i] = step[i][kii];
                }
            }
            for (int i = 0; i < 32; ++i) {
                const float fs = smax[i] > 0 ? smax[i] / 127.0f : 1.0f;
                s16[i] = (_Float16) fs;
                for (int kii = 0; kii < 16; ++kii) {
                    int isc = (int) lrintf(step[i][kii] / fs);
                    if (isc < 1) isc = 1;
                    if (isc > 127) isc = 127;
                    scl[kii * 32 + i] = (int8_t) isc;
                    const float eff = (float) isc * fs;
                    uint16_t mask = 0;
                    for (int bi = 0; bi < 16; ++bi) {
                        const float v = tile[i][kii * 16 + bi];
                        int q3 = eff > 0 ? (int) lrintf(v / eff) : 0;
                        if (q3 < -4) q3 = -4;
                        if (q3 > 3) q3 = 3;
                        const int u = q3 < 0 ? q3 + 4 : q3;
                        const int64_t byte = (int64_t) (kii / 4) * 512 + i * 16 + bi;
                        qs[byte] = (uint8_t) ((qs[byte] & ~(3 << ((kii % 4) * 2))) | ((u & 3) << ((kii % 4) * 2)));
                        if (q3 >= 0) mask |= (uint16_t) (1u << bi);
                    }
                    hms[kii * 32 + i] = mask;
                }
            }
        }
    }
}

// Same source, 2-bit destination. The encoding is llama.cpp's Q2_K scheme
// without its make_qkx2_quants search: per 16-value sub-block take min and max,
// clamp them so the range straddles zero, and let
//   step = (max - min) / 3,  z = -min
//   d = max_j step / 15,     dm = max_j z / 15
//   ns = round(15 * step / d),  nz = round(15 * z / dm),  q = round((x + z)/step)
// so the kernel's `q * ns * d - nz * dm` reproduces the value. Note the effective
// multiplier is ns*d with no 1/16 anywhere: the vendor kernel's factor_scale of
// 1/16 is undone by its refactor_scale of 16 in the epilogue.
__attribute__((noinline))
// Least-squares (scale, min, levels) for one 16-element group: a port of
// llama.cpp's make_qkx2_quants as quantize_row_q2_K_ref calls it. The max/min
// split this replaces spent the whole 4-bit scale range on one outlier and left
// the rest of the group on two levels; the fit walks sixteen (rmin, rdelta) probes
// and keeps the closed-form least-squares solution with the lowest |x|-weighted
// absolute error. That is what puts llama.cpp's own Q2_K within a few tokens of
// its Q4_0 continuation, which our first attempt was nowhere near.
static float qkx2_16(const float * x, uint8_t * L, float * the_min) {
    float w[16], min = x[0], max = x[0], sum_w = 0, sum_x = 0;
    for (int i = 0; i < 16; ++i) {
        w[i] = fabsf(x[i]);
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < 16; ++i) L[i] = 0;
        *the_min = -min;
        return 0.0f;
    }
    float iscale = 3.0f / (max - min);
    float scale = 1.0f / iscale;
    float best_error = 0;
    for (int i = 0; i < 16; ++i) {
        int l = (int) lrintf(iscale * (x[i] - min));
        if (l < 0) l = 0; if (l > 3) l = 3;
        L[i] = (uint8_t) l;
        best_error += w[i] * fabsf(scale * l + min - x[i]);
    }
    for (int is = 0; is <= 15; ++is) {
        iscale = (-0.5f + 0.1f * (float) is + 3.0f) / (max - min);
        uint8_t Laux[16];
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < 16; ++i) {
            int l = (int) lrintf(iscale * (x[i] - min));
            if (l < 0) l = 0; if (l > 3) l = 3;
            Laux[i] = (uint8_t) l;
            sum_l += w[i] * l; sum_l2 += w[i] * l * l; sum_xl += w[i] * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float ts = (sum_w * sum_xl - sum_x * sum_l) / D;
            float tm = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (tm > 0) { tm = 0; ts = sum_xl / sum_l2; }
            float ce = 0;
            for (int i = 0; i < 16; ++i) ce += w[i] * fabsf(ts * Laux[i] + tm - x[i]);
            if (ce < best_error) {
                memcpy(L, Laux, 16);
                best_error = ce;
                scale = ts;
                min = tm;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static void repack_b_q2(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t rs = src_row_bytes(k), nsup = k / Q2_K;
    const int64_t sbytes = rs / nsup;
    static float tile[32][Q2_K];
    for (int64_t g = 0; g < rows; g += 32) {
        const int64_t grp = g / 32;
        for (int64_t sup = 0; sup < nsup; ++sup) {
            uint8_t *  blk = dst + (grp * nsup + sup) * Q2_BLK;
            uint8_t *  scl = blk;
            _Float16 * s16 = (_Float16 *) (blk + 512 + 2048);
            _Float16 * z16 = (_Float16 *) (blk + 512 + 2048 + 64);
            memset(scl, 0, 512);
            memset(blk + 512, 0, 2048);
            for (int i = 0; i < 32; ++i) {
                const int64_t sr = perm ? (int64_t) perm[g + i] : (g + i);
                dequant_row(tile[i], src + sr * rs + sup * sbytes, Q2_K);
            }
            for (int i = 0; i < 32; ++i) {
                uint8_t L[Q2_K];
                float step[16], zoff[16], smax = 0, zmax = 0;
                for (int j = 0; j < 16; ++j) {
                    step[j] = qkx2_16(tile[i] + j * 16, L + j * 16, &zoff[j]);
                    if (step[j] > smax) smax = step[j];
                    if (zoff[j] > zmax) zmax = zoff[j];
                }
                s16[i] = (_Float16) (smax > 0 ? smax / 15.0f : 0.0f);
                z16[i] = (_Float16) (zmax > 0 ? zmax / 15.0f : 0.0f);
                // Re-derive the levels from the scale the kernel will actually
                // reconstruct (fp16 d times the 4-bit nibble), not from the ideal
                // fit: the nibble rounding is up to a third of a step on the small
                // groups, and leaving it out of the requantisation is a per-group
                // gain error that compounds over 42 layers.
                const float d  = (float) s16[i];
                const float dm = (float) z16[i];
                for (int j = 0; j < 16; ++j) {
                    int ns = smax > 0 ? (int) lrintf(step[j] / (smax / 15.0f)) : 0;
                    int nz = zmax > 0 ? (int) lrintf(zoff[j] / (zmax / 15.0f)) : 0;
                    if (ns < 0) ns = 0; if (ns > 15) ns = 15;
                    if (nz < 0) nz = 0; if (nz > 15) nz = 15;
                    // The vendor's 512-byte scale plane is two overlapping
                    // permutations of the same bytes, not two separate arrays:
                    // scale nibbles column-major by sub-block (scales[j*32+i] low
                    // half) and zero points row-major (scales[16*i+j] high half).
                    // probe_q2pack.cpp measures the packed block against a rebuild
                    // from the same bytes: this layout lands at 0.07% and every
                    // other arrangement is off by 30% or more.
                    scl[j * 32 + i] |= (uint8_t) ns;
                    scl[16 * i + j] |= (uint8_t) (nz << 4);
                    const float dq = d * (float) ns;
                    const float dmq = dm * (float) nz;
                    for (int b2 = 0; b2 < 16; ++b2) {
                        int q2 = dq > 0 ? (int) lrintf((tile[i][j * 16 + b2] + dmq) / dq) : 0;
                        if (q2 < 0) q2 = 0;
                        if (q2 > 3) q2 = 3;
                        uint8_t * byte = blk + 512 + (int64_t) (j / 4) * 32 * 16 + i * 16 + b2;
                        *byte |= (uint8_t) (q2 << ((j % 4) * 2));
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ weight repack
static void repack_b(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t nb = k / 32, rs = nb * 18;
    uint8_t * d = dst;
    for (int64_t b = 0; b < rows; b += 32) {
        for (int64_t x = 0; x < nb; ++x) {
            _Float16 * dd = (_Float16 *) d;
            uint8_t *  qs = d + 64;
            for (int i = 0; i < 32; ++i) {
                const int64_t sr = perm ? (int64_t) perm[b + i] : (b + i);
                const uint8_t * s = src + sr * rs + x * 18;
                _Float16 sc; memcpy(&sc, s, 2);
                dd[i] = sc;
                const uint8_t * sq = s + 2;
                uint8_t * dq = qs + i * 16;
                for (int j = 0; j < 8; ++j) {
                    dq[j]     = (uint8_t) ((sq[2 * j] & 0x0F) | ((sq[2 * j + 1] & 0x0F) << 4));
                    dq[8 + j] = (uint8_t) ((sq[2 * j] >> 4)  | ((sq[2 * j + 1] & 0xF0)));
                }
            }
            d += 576;
        }
    }
}

// ------------------------------------------------------------------- thread pool
static std::atomic<int> g_bar_gen{0};
static std::atomic<int> g_bar_cnt{0};
static std::atomic<int> g_quit{0};
static int g_participants = N_GPU;

static inline void barrier(void) {
    const int g = g_bar_gen.load(std::memory_order_acquire);
    if (g_bar_cnt.fetch_add(1, std::memory_order_acq_rel) == g_participants - 1) {
        g_bar_cnt.store(0, std::memory_order_relaxed);
        g_bar_gen.fetch_add(1, std::memory_order_release);
    } else {
        // Back off between polls: seven cores hammering one cache line starve
        // whatever thread is still doing the work.
        while (g_bar_gen.load(std::memory_order_acquire) == g) sched_yield();
    }
}

typedef void (*task_fn)(int tid);
static task_fn g_task;

static int g_cpu0 = 8;
static uint8_t * g_tcm[16];
static int64_t   g_tcm_size;
static void tcm_attach(int id) {
    static bool opened = false;
    if (!opened) { spine_tcm_open_handle(nullptr); opened = true; }
    if (!spine_tcm_is_available()) return;
    spine_tcm_mem_info_t mi;
    if (spine_tcm_mem_info(&mi) == 0) g_tcm_size = (int64_t) mi.blk_size;
    g_tcm[id] = (uint8_t *) spine_tcm_mem_get(id);
}
static inline long vlenb_self(void) { long v = 0; asm volatile("csrr %0, vlenb" : "=r"(v)); return v; }

// Every int8 kernel here assumes 32 e32 lanes per m1 register (VLEN=1024), which
// only exists on the A100 cluster. A thread that fails to bind therefore does not
// just run slowly, it computes nonsense -- and a lost bind used to show up only as
// a 2x swing in the benchmark. Serialise the vendor handshake, retry it, and treat
// a thread that misses its cluster as fatal.
static int bind_cpu(int cpu) {
    static std::atomic_flag gate = ATOMIC_FLAG_INIT;
    while (gate.test_and_set(std::memory_order_acquire)) sched_yield();
    int rc = -1;
    for (int attempt = 0; attempt < 200; ++attempt) {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
        rc = sched_setaffinity(0, sizeof(s), &s);
        if (rc == 0) break;
        sched_yield();
    }
    gate.clear(std::memory_order_release);
    cpu_set_t now; CPU_ZERO(&now);
    if (sched_getaffinity(0, sizeof(now), &now) == 0) {
        const int on_a = cpu >= 8;
        const long vb = vlenb_self();
        if (CPU_ISSET(cpu, &now) == 0 || (on_a && vb != 128)) {
            fprintf(stderr, "[FATAL] bind to cpu %d failed (rc=%d, vlenb=%ld): a thread on "
                            "the wrong cluster computes garbage\n", cpu, rc, vb);
            _exit(3);
        }
    }
    return rc;
}

static std::atomic<int> g_ready[16];

static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    bind_cpu(g_cpu0 + id);
    tcm_attach(id);
    g_ready[id].store(1, std::memory_order_release);
    for (;;) {
        barrier();
        if (g_quit.load(std::memory_order_acquire)) break;
        g_task(id);
        barrier();
    }
    return nullptr;
}

static void run(task_fn fn) {
    g_task = fn;
    barrier();
    fn(0);
    barrier();
}

static void (*g_pf)(int64_t, int64_t);
static int64_t g_pf_n;
static thread_local int g_pf_tid;
static void task_pf(int tid) {
    const int64_t per = (g_pf_n + g_participants - 1) / g_participants;
    int64_t lo = per * tid, hi = lo + per;
    if (hi > g_pf_n) hi = g_pf_n;
    g_pf_tid = tid;
    if (lo < hi) g_pf(lo, hi);
}
static void parallel_for(int64_t n, void (*fn)(int64_t, int64_t)) {
    g_pf = fn; g_pf_n = n;
    run(task_pf);
}

// ------------------------------------------------- X100 elementwise work pool
// The eight X100 cores (0-7) have no IME2 and are idle while prefill runs its
// GEMMs, but rmsnorm, the activation quantiser, silu and the KV int8 conversion
// are pure elementwise RVV anywhere on the chip. Handing them sixteen bands
// instead of eight keeps every result bit-identical -- the bands only change who
// writes which element -- so this is the one way to use the second cluster
// without touching the greedy chain.
static std::atomic<int> g_el_gen{0}, g_el_done{0};
static void (*g_el_fn)(int) = nullptr;
static int g_el_parts = 8;
static int g_el_last = 8;               // how many bands the last run_el used
static int g_el_on = 0;                 // set only while the X100 threads exist

static void run_el(void (*fn)(int)) {
    if (!g_el_on) { g_el_last = 8; run(fn); return; }
    g_el_last = 16;
    g_el_fn = fn;
    g_el_parts = 16;
    g_el_done.store(0, std::memory_order_relaxed);
    g_el_gen.fetch_add(1, std::memory_order_release);
    run(fn);                                   // A cluster takes ids 0..7
    while (g_el_done.load(std::memory_order_acquire) < 8) sched_yield();
    g_el_parts = 8;
}
// An A100 core is VLEN=1024 at 1.8 GHz and an X100 core VLEN=256 at 2.2 GHz, so an
// even sixteen-way split leaves the A cluster waiting on bands that are four times
// slower: silu got *worse* (1.52 -> 1.78 s) before the weights were added. A
// thread's share is 3 units on the A cluster and 1 on X100.
// conv is a streaming pass whose cost follows thread count rather than per-thread
// speed, so it takes an even sixteen-way split: 6.4 s -> 4.4 s. The compute-bound
// passes (rmsnorm, silu, the vendor quantiser) stay on the A cluster, where a
// weighted attempt measured them slower than the plain eight-way split -- an X100
// core's f32 vector throughput is much worse than a quarter of an A100 core's.
static const int CV_W[16] = { 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1 };
static inline void band_w(const int * W, int tid, int64_t n, int64_t * lo, int64_t * hi) {
    if (g_el_parts != 16) { *lo = n * tid / g_el_parts; *hi = n * (tid + 1) / g_el_parts; return; }
    int64_t tot = 0, pre = 0, mine = W[tid];
    for (int t = 0; t < 16; ++t) tot += W[t];
    for (int t = 0; t < tid; ++t) pre += W[t];
    *lo = n * pre / tot; *hi = n * (pre + mine) / tot;
}

// A 16-band parallel_for. `wa` and `wx` are the weights one A100 and one X100
// thread get, so 3:1 asks an A core for three times an X100 core's share --
// which is the ratio the two cores' f32 vector throughput has. With g_el_on
// clear it degrades to the plain eight-way split.
static void (*g_pfw_fn)(int64_t, int64_t);
static int64_t g_pfw_n;
static int g_pfw_w[16];
static int g_elq = 0;   // MEGA_ELQ: 1 = weighted 3:1, 2 = even, for the quantizer
// rmsnorm and the residual add measured 60.2 -> 58.9 s on sixteen bands with an
// even split (0.85 + 0.84 against 1.50 + 1.50 s), so the pool is the default for
// them. silu and the activation quantiser stay on the A cluster: silu gets slower
// (1.50 -> 1.85) and the quantiser's gain comes from the aliasing below, not from
// the extra bands. Both are still selectable so the comparison can be re-run.
static int g_eln = 2;   // MEGA_ELN: 0 off, 1 weighted 3:1, 2 even
static int g_els = 0;   // MEGA_ELS: same for silu, which loses on sixteen bands
static void pfw_task(int id) {
    int64_t lo, hi;
    band_w(g_pfw_w, id, g_pfw_n, &lo, &hi);
    if (lo < hi) g_pfw_fn(lo, hi);
}
static void parallel_for_w(int64_t n, void (*fn)(int64_t, int64_t), int wa, int wx) {
    if (!g_el_on) { parallel_for(n, fn); return; }
    g_pfw_fn = fn; g_pfw_n = n;
    for (int t = 0; t < 8; ++t) g_pfw_w[t] = wa;
    for (int t = 8; t < 16; ++t) g_pfw_w[t] = wx;
    run_el(pfw_task);
}
static inline void el_split(int mode, int * wa, int * wx) {
    if (mode == 2) { *wa = 1; *wx = 1; }
    else           { *wa = 3; *wx = 1; }
}

// ---------------------------------------------------------------- model buffers
static uint8_t * g_wblob;
static size_t g_wblob_sz = 0;
struct WRef { const uint8_t * b; int64_t k, n; const uint8_t * b8; int type; };
static WRef g_wo, g_wq[N_LAYER], g_wk[N_LAYER], g_wv[N_LAYER], g_wov[N_LAYER];
static WRef g_wg[N_LAYER], g_wu[N_LAYER], g_wd[N_LAYER];
static const Entry * g_emb;
static const float * g_attn_norm[N_LAYER], * g_ffn_norm[N_LAYER], * g_out_norm;

static float * g_x, * g_xa, * g_gate, * g_up, * g_logits;
static float * g_q, * g_k, * g_v, * g_attn_out;
static _Float16 * g_kdim;       // [layer][kv][pos_blk][dim][32], tiled
static _Float16 * g_vpos;       // [layer][kv][pos][dim]
static uint8_t * g_aq;
static float * g_rope_tab;                      // [pos][2][64] cos,sin
static int   g_perm[N_EMBD];                    // planar rotation-pair order

static int g_pos, g_layer;

// Int8 copies of both decode caches, same layouts and strides as the f16 ones so
// the attention loops keep their access shape and only the load width changes.
// Decode reads every KV byte once per token, so the halved width is the whole
// point; the scales are one float per layer per kv head, taken from prefill's
// own max-abs scan, so decode adds no rescaling pass.
static int8_t * g_k8d = nullptr;
static int8_t * g_v8d = nullptr;
// The quantisation grid of each layer's int8 K/V plane. A four-row decode round
// appends positions to those planes, and prefill's own attention dequantises with
// the globals g_k8isc/g_v8isc, so the grid has to be recorded while prefill runs
// rather than looked up afterwards. Fixed size: two kv heads per layer.
static float g_k8i_sc[N_LAYER * N_KV];
static float g_v8i_sc[N_LAYER * N_KV];
// int8 V is stored in PV order, [layer][kv][half][pos][64], not [layer][pos][kv][128].
// PV's inner loop is over positions with the head dimension fixed, so the old
// pos-major layout made it read the 128-byte half of every 512-byte position
// stride, four passes over the same range. A strided-read probe puts that pattern
// at 9.0 GB/s against 17.3 for a sequential stream, and PV measured exactly 9.0.
// Keeping the half as the outer index makes each pass one contiguous run.
static inline int64_t v8_off(int kv, int half, int64_t pos) {
    return (((int64_t) kv * 2 + half) * MAXCTX + pos) * 64;
}
static float g_k8dsc[N_LAYER][N_KV], g_v8dsc[N_LAYER][N_KV];
static int g_k8dec = 0;

static const int64_t KV_LAYER_STRIDE = (int64_t) N_KV * HEAD_DIM * MAXCTX;   // V: pos-major
// K is tiled [kv][pos_blk][dim][32]. A dim-major K with a MAXCTX stride put each
// of the 128 dims on its own page, so both the per-token append and the QK read
// paid a TLB miss per element (~145 us per layer). A tile keeps a 32-position
// window in 16 KB, i.e. four pages.
static const int KT = 32;
static const int NKT = (MAXCTX + KT - 1) / KT;
static const int64_t K_LAYER_STRIDE = (int64_t) N_KV * NKT * HEAD_DIM * KT;
static inline _Float16 * ktile(int layer, int kv, int pblk) {
    return g_kdim + (int64_t) layer * K_LAYER_STRIDE + (((int64_t) kv * NKT + pblk) * HEAD_DIM) * KT;
}

// attention scratch: per-thread score strip, per-thread partial accumulators
static int g_pvprobe = 0;                     // MEGA_PVPROBE: PV without the score loads
static const int SMAX = MAXCTX / 4 + 64;         // a thread may own up to 1/4 of ctx
static float g_sa[N_GPU][N_HEAD][SMAX];
static float g_acc[N_GPU][N_HEAD][HEAD_DIM];
static float g_lmax[N_GPU][N_HEAD], g_lsum[N_GPU][N_HEAD];
static float g_gmax[N_HEAD], g_gsum[N_HEAD];
static int g_dbg = 0;
static double g_aw[4];   // real work inside each attn task, excluding barrier wait

// ------------------------------------------------------------------ fused pieces
static const float * rn_src, * rn_w;
static float * rn_dst;
static double rn_ss[N_GPU];
static float rn_r;

static void rn_sum_fn(int64_t lo, int64_t hi) {
    vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(0.0f, 32);
    int64_t i = lo;
    for (; i + 32 <= hi; i += 32) {
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(rn_src + i, 32);
        a = __riscv_vfmacc_vv_f32m1(a, x, x, 32);
    }
    rn_ss[g_pf_tid] = (double) __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(0.0f, 1), 32));
    for (; i < hi; ++i) rn_ss[g_pf_tid] += 0.0;
}

static void rn_scale_fn(int64_t lo, int64_t hi) {
    int64_t i = lo;
    for (; i + 32 <= hi; i += 32) {
        const size_t vl = 32;
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(rn_src + i, vl);
        const vfloat32m1_t w = __riscv_vle32_v_f32m1(rn_w + i, vl);
        __riscv_vse32_v_f32m1(rn_dst + i, __riscv_vfmul_vf_f32m1(__riscv_vfmul_vv_f32m1(x, w, vl), rn_r, vl), vl);
    }
    for (; i < hi; ++i) rn_dst[i] = rn_src[i] * rn_r * rn_w[i];
}

static void rmsnorm_par(float * dst, const float * src, const float * w, int n) {
    rn_src = src; rn_dst = dst; rn_w = w;
    for (int t = 0; t < N_GPU; ++t) rn_ss[t] = 0;
    parallel_for(n, rn_sum_fn);
    double ss = 0;
    for (int t = 0; t < N_GPU; ++t) ss += rn_ss[t];
    rn_r = 1.0f / sqrtf((float) (ss / n) + RMS_EPS);
    parallel_for(n, rn_scale_fn);
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else {
        *d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t) ((q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4));
    }
}

static void dequant_row_t(float * dst, const uint8_t * row, int64_t k, int type) {
    if (type == 12) {                             // Q4_K
        for (int64_t i = 0; i < k / 256; ++i) {
            const uint8_t * blk = row + i * 144;
            const float d = f16_to_f32(blk), mn = f16_to_f32(blk + 2);
            const uint8_t * sc = blk + 4, * q = blk + 16;
            float * y = dst + i * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                get_scale_min_k4(is + 0, sc, &s1, &m1);
                get_scale_min_k4(is + 1, sc, &s2, &m2);
                const float d1 = d * (float) s1, b1 = mn * (float) m1;
                const float d2 = d * (float) s2, b2 = mn * (float) m2;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (float) (q[l] & 0xF) - b1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (float) (q[l] >> 4) - b2;
                q += 32; is += 2;
            }
        }
        return;
    }
    if (type == 13) {                             // Q5_K
        for (int64_t i = 0; i < k / 256; ++i) {
            const uint8_t * blk = row + i * 176;
            const float d = f16_to_f32(blk), mn = f16_to_f32(blk + 2);
            const uint8_t * sc = blk + 4, * qh = blk + 16, * ql = blk + 48;
            float * y = dst + i * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                get_scale_min_k4(is + 0, sc, &s1, &m1);
                get_scale_min_k4(is + 1, sc, &s2, &m2);
                const float d1 = d * (float) s1, b1 = mn * (float) m1;
                const float d2 = d * (float) s2, b2 = mn * (float) m2;
                for (int l = 0; l < 32; ++l)
                    *y++ = d1 * (float) ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - b1;
                for (int l = 0; l < 32; ++l)
                    *y++ = d2 * (float) ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - b2;
                ql += 32; is += 2; u1 = (uint8_t) (u1 << 2); u2 = (uint8_t) (u2 << 2);
            }
        }
        return;
    }
    if (type == 14) {                             // Q6_K
        for (int64_t i = 0; i < k / 256; ++i) {
            const uint8_t * blk = row + i * 210;
            const uint8_t * ql = blk;
            const uint8_t * qh = blk + 128;
            const int8_t * sc = (const int8_t *) (blk + 192);
            const float d = f16_to_f32(blk + 208);
            float * y = dst + i * 256;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0]  = d * (float) sc[is + 0] * (float) q1;
                    y[l + 32] = d * (float) sc[is + 2] * (float) q2;
                    y[l + 64] = d * (float) sc[is + 4] * (float) q3;
                    y[l + 96] = d * (float) sc[is + 6] * (float) q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        return;
    }
    if (type == 11) {                       // Q3_K, ggml reference decoding
        static const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        for (int64_t i = 0; i < k / 256; ++i) {
            const uint8_t * blk = row + i * 110;
            const float d_all = f16_to_f32(blk + 108);
            const uint8_t * q = blk + 32;          // qs
            const uint8_t * hm = blk;              // hmask
            uint32_t aux[4];
            memcpy(aux, blk + 96, 12);             // scales
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const int8_t * sc = (const int8_t *) aux;
            float * y = dst + i * 256;
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    const float dl = d_all * (float) (sc[is++] - 32);
                    for (int l = 0; l < 16; ++l)
                        *y++ = dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    const float dl2 = d_all * (float) (sc[is++] - 32);
                    for (int l = 0; l < 16; ++l)
                        *y++ = dl2 * (float) ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    shift += 2;
                    m = (uint8_t) (m << 1);
                }
                q += 32;
            }
        }
        return;
    }
    if (type == 10) {                       // Q2_K
        for (int64_t i = 0; i < k / 256; ++i) {
            const uint8_t * blk = row + i * 84;
            const float d = f16_to_f32(blk + 80);
            const float dmin = f16_to_f32(blk + 82);
            const uint8_t * q = blk + 16;
            const uint8_t * scales = blk;
            float * y = dst + i * 256;
            int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t s = scales[is++];
                    float dl = d * (float) (s & 0xF), ml = dmin * (float) (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float) ((q[l] >> shift) & 3) - ml;
                    s = scales[is++];
                    dl = d * (float) (s & 0xF); ml = dmin * (float) (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float) ((q[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
        return;
    }
    if (type == 8) {                        // Q8_0
        for (int64_t b = 0; b < k / 32; ++b) {
            const float sc = f16_to_f32(row + b * 34);
            const int8_t * q = (const int8_t *) (row + b * 34 + 2);
            for (int j = 0; j < 32; ++j) dst[b * 32 + j] = sc * (float) q[j];
        }
        return;
    }
    if (type == 1) {                        // F16
        for (int64_t i = 0; i < k; ++i) dst[i] = (_Float16) 0 + f16_to_f32(row + i * 2);
        return;
    }
    if (type == 0) {                        // F32
        memcpy(dst, row, (size_t) k * 4);
        return;
    }
    const int64_t nb = k / 32;
    for (int64_t b = 0; b < nb; ++b) {
        _Float16 sc; memcpy(&sc, row + b * 18, 2);
        const uint8_t * q = row + b * 18 + 2;
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j]      = (float) sc * (float) ((int) (q[j] & 0x0F) - 8);
            dst[b * 32 + 16 + j] = (float) sc * (float) ((int) (q[j] >> 4) - 8);
        }
    }
}
static void dequant_row(float * dst, const uint8_t * row, int64_t k) {
    dequant_row_t(dst, row, k, g_src_type);
}

// ---- multi-GEMM (unchanged, verified against llama.cpp)
struct MJob { const uint8_t * b; int64_t k, n; float * c; };
static MJob g_mj[8];
static int g_mjn;
static int g_kstep = 4;

static void task_mgemm(int tid) {
    int64_t n0[8], n1[8], step_max = 0;
    for (int j = 0; j < g_mjn; ++j) {
        const int64_t ng = g_mj[j].n / 32;
        const int64_t per = (ng + g_participants - 1) / g_participants;
        n0[j] = per * tid;
        n1[j] = (tid == g_participants - 1) ? ng : per * (tid + 1);
        if (n1[j] > ng) n1[j] = ng;
        if (n0[j] > ng) n0[j] = ng;
        const int64_t st = (n1[j] - n0[j] + g_kstep - 1) / g_kstep;
        if (st > step_max) step_max = st;
    }
    for (int64_t s = 0; s < step_max; ++s)
        for (int j = 0; j < g_mjn; ++j) {
            const int64_t a = n0[j] + s * g_kstep;
            if (a >= n1[j]) continue;
            int64_t b2 = a + g_kstep; if (b2 > n1[j]) b2 = n1[j];
            const int64_t kb = g_mj[j].k / (g_bit3 || g_bit2 ? Q3_K : g_hpdec ? 256 : 32);
            // A column tile is one packed block per output-column group, so its byte
            // stride is (number of blocks along k) * block size. That is k/32 for the
            // Q4_0 blob (576 B per 32 columns x 32 k, and the hp kernel reads the same
            // blob), but k/256 for the q2/q3 blobs, whose block spans a whole
            // superblock. Striding those by k/32 walked eight times too far and read
            // zeroed padding, which is why every -q2/-q3 run produced near-zero
            // logits while thread 0's a=0 tile looked fine.
            const uint8_t * bp = g_mj[j].b + a *
                ((g_bit3 || g_bit2) ? (g_mj[j].k / Q2_K) * g_blk_bytes
                                    : (g_mj[j].k / 32) * g_blk_bytes);
            if (g_bit2)
                spacemit_kernels::ime2::gemm_kernel_i8i2k(
                    Q2_K, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
            else if (g_bit3)
                spacemit_kernels::ime2::gemm_kernel_i8i3k(
                    Q3_K, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
            else if (g_hpdec)
                // llama.cpp's Q4_0 kernel table maps to q4_0_32x256_q8_0 in every
                // case, and that kernel picks its one-row body when count_m < 4.
                // Padding one row to four does NOT reproduce it: the one-row and
                // four-row entry points quantize A with different record layouts,
                // which is a 0.6-logit difference on a knife-edge prompt.
                spacemit_kernels::ime2::gemm_kernel_i8i4_hp(
                    256, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
            else
                spacemit_kernels::ime2::gemm_kernel_i8i4(
                    32, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
        }
}

// 0 = vmadot, 1 = vendor K32 i8i4, 2 = vendor K256 i8i4 hp
//
// The default is the hp path: llama.cpp's own spacemit prefill runs the K256 hp
// kernel, so matching it is what keeps the greedy chain on the reference ids.
// At the 5-token prompt the K32 path emits `8181 350` while both the reference
// and this path emit `8181 33`. 10K: 41.1 s of GEMM against 59.0 s.
static int g_pf_vendor = 2;

// ------------------------------------------------------- four-row decode GEMM
// A speculation round has four rows and one weight stream. The hp kernel that
// prefill already runs takes count_m = 4 and reads the same compact Q4_0 blob
// decode has, so a round costs one pass over 1265 MB instead of four.
//
// The A record is what the kernel expects: per 256-k superblock, four scales,
// four bias sums and four 256-byte payloads, i.e. (k/256) * 1160 bytes. That is
// 4 MAC per weight byte at four rows against 2 at one, so the pass stays on the
// DRAM floor rather than becoming compute bound.
static uint8_t * g_aq4d = nullptr;               // A records, k=2048 then k=6144
static float   * g_c4 = nullptr;                 // 4 x N_FF output scratch
static int64_t g_aq4d_foff = 0;                  // byte offset of the k=6144 record
static inline int64_t m4_aqsz(int64_t k) { return (k / 256) * 1160; }
static void * hugemem(size_t n);

// The drafter runs the same kernel at a third width, its fused feature input.
static int64_t g_aq4d_dsoff = 0, g_aq4d_mkoff = 0;
static const int64_t DS_IN_K = 5 * N_EMBD;
static const int64_t DS_RANK_K = 256;

static void m4_alloc(void) {
    if (g_aq4d) return;
    g_aq4d_foff = m4_aqsz(N_EMBD);
    g_aq4d_dsoff = m4_aqsz(N_EMBD) + m4_aqsz(N_FF);
    g_aq4d_mkoff = g_aq4d_dsoff + m4_aqsz(DS_IN_K);
    const size_t an = (size_t) (m4_aqsz(N_EMBD) + m4_aqsz(N_FF) + m4_aqsz(DS_IN_K)
                                + m4_aqsz(DS_RANK_K)) + (2 << 20);
    const size_t cn = (size_t) 4 * N_FF * 4 + (2 << 20);
    g_aq4d = (uint8_t *) hugemem(an);
    g_c4   = (float *) hugemem(cn);
    if (!g_aq4d || !g_c4) { fprintf(stderr, "oom m4 gemm\n"); return; }
    use_huge(g_aq4d, an);
    use_huge(g_c4, cn);
}

static inline uint8_t * m4_aq(int64_t k) {
    return g_aq4d + (k == N_FF ? g_aq4d_foff : 0);
}

static inline uint8_t * m4_aq2(int64_t k) {
    if (k == DS_IN_K) return g_aq4d + g_aq4d_dsoff;
    if (k == DS_RANK_K) return g_aq4d + g_aq4d_mkoff;
    return m4_aq(k);
}



// Same job list and column split as task_mgemm; the difference is the kernel and
// the row stride, which has to be the matrix width because row r starts at r*n.
static void task_mgemm_m4(int tid) {
    int64_t n0[8], n1[8], step_max = 0;
    for (int j = 0; j < g_mjn; ++j) {
        const int64_t ng = g_mj[j].n / 32;
        const int64_t per = (ng + g_participants - 1) / g_participants;
        n0[j] = per * tid;
        n1[j] = (tid == g_participants - 1) ? ng : per * (tid + 1);
        if (n1[j] > ng) n1[j] = ng;
        if (n0[j] > ng) n0[j] = ng;
        const int64_t st = (n1[j] - n0[j] + g_kstep - 1) / g_kstep;
        if (st > step_max) step_max = st;
    }
    for (int64_t s = 0; s < step_max; ++s)
        for (int j = 0; j < g_mjn; ++j) {
            const int64_t a = n0[j] + s * g_kstep;
            if (a >= n1[j]) continue;
            int64_t b2 = a + g_kstep; if (b2 > n1[j]) b2 = n1[j];
            const int64_t kb = g_mj[j].k / 32;
            const uint8_t * bp = g_mj[j].b + a * (g_mj[j].k / 32) * 576;
            spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
                256, m4_aq2(g_mj[j].k), bp, nullptr, g_mj[j].c + a * 32,
                4, (size_t) ((b2 - a) * 32), (size_t) (kb / 8), (size_t) g_mj[j].n);
        }
}

// What one four-row round costs in GEMM. Runs the exact matrix sequence of the
// decode loop with the m4 kernel and no attention, so it isolates the weight
// stream: one pass over 1265 MB whether it feeds one row or four.
static void m4_gemm_bench(int iters) {
    m4_alloc();
    if (!g_aq4d) return;
    float * x4 = (float *) hugemem((size_t) 4 * N_FF * 4 + (2 << 20));
    if (!x4) return;
    for (int64_t i = 0; i < 4 * N_FF; ++i) x4[i] = sinf(0.001f * (float) (i % 4096)) * 2.0f;
    double best = 1e9, macs = 0;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_s();
        for (int L = 0; L < N_LAYER; ++L) {
            spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x4, (size_t) N_EMBD, m4_aq(N_EMBD));
            g_mjn = 3;
            g_mj[0] = { g_wq[L].b, g_wq[L].k, g_wq[L].n, g_c4 };
            g_mj[1] = { g_wk[L].b, g_wk[L].k, g_wk[L].n, g_c4 };
            g_mj[2] = { g_wv[L].b, g_wv[L].k, g_wv[L].n, g_c4 };
            run(task_mgemm_m4);
            spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x4, (size_t) N_EMBD, m4_aq(N_EMBD));
            g_mjn = 1; g_mj[0] = { g_wov[L].b, g_wov[L].k, g_wov[L].n, g_c4 };
            run(task_mgemm_m4);
            spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x4, (size_t) N_EMBD, m4_aq(N_EMBD));
            g_mjn = 2;
            g_mj[0] = { g_wg[L].b, g_wg[L].k, g_wg[L].n, g_c4 };
            g_mj[1] = { g_wu[L].b, g_wu[L].k, g_wu[L].n, g_c4 };
            run(task_mgemm_m4);
            spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, x4, (size_t) N_FF, m4_aq(N_FF));
            g_mjn = 1; g_mj[0] = { g_wd[L].b, g_wd[L].k, g_wd[L].n, g_c4 };
            run(task_mgemm_m4);
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        macs = 4.0 * (double) N_LAYER * ((double) N_EMBD * N_EMBD * 2 +
                                         (double) N_EMBD * N_FF * 3);
        printf("m4gemm: iter %d %.2f ms (%.1f GMAC/s)\n", it, dt * 1e3, macs / dt / 1e9);
    }
    printf("m4gemm: best %.2f ms for 42 layers x 4 rows -> %.1f GMAC/s\n",
           best * 1e3, macs / best / 1e9);
    free(x4);
}

// ------------------------------------------------------------------ prefill GEMM
// The m=4 vendor path needs A as, per k_blk, [f32 scale x 4][int8 data 4x32]
// = 144 bytes; the grouped quantizer fills that. k_blks must be a multiple of 16.
// Loop order is n-outer / m-inner so the 576-byte B column tile is loaded once
// from DRAM and reused for every row group in the chunk: at m=1 the tile was
// re-read per row, which is what kept prefill pinned to DRAM bandwidth.
static int64_t g_pf_rows;
static const char * g_pf_file = nullptr;   // prompt token ids, whitespace separated
static int64_t g_pf_chunk = 2560;
static int64_t g_pf_nb = 1;          // n-blocks per gemm call
static int g_pf_mt = 4;              // row groups (4 rows each) per shared-B tile
static int g_pf_reps = 0;            // repeat q_proj to isolate compute peak
static uint8_t * g_pfa;              // quantized A for the whole chunk
static int64_t g_pfa_rowstride;      // bytes per 4-row group

// Every job in a phase reads the same activation rows with the same k (q, k and v
// are all k=2048; gate and up likewise), so the per-job copies were three
// quantisations of identical bytes. g_pfa_jslot says which copy a job reads; the
// quantiser writes copy 0 once per phase and the three jobs alias it.
static int g_pfa_jslot[4] = { 0, 1, 2, 3 };

static int g_x100;              // defined with the X100 section below
struct PfJob {
    const uint8_t * b; int64_t k, n; float * c; int64_t ldc;
    int64_t nlim;   // A100 stops here; the tail columns belong to the X100 cluster
};
static PfJob g_pfj[4];
static int g_pfjn;

static void task_pf_gemm(int tid) {
    const int64_t mgroups = g_pf_rows / 4;
    for (int j = 0; j < g_pfjn; ++j) {
        // nlim is the A100's share of the columns when the X100 cluster takes the
        // tail; reading n here made both clusters compute every column, so the
        // X100 work was duplicate and the two wrote the same C elements.
        const int64_t nb = (g_x100 && g_pfj[j].nlim > 0 ? g_pfj[j].nlim : g_pfj[j].n) / 32;
        const int64_t kb = g_pfj[j].k / 32;
        const int64_t ldc = g_pfj[j].ldc;
        const uint8_t * b = g_pfj[j].b;
        float * c = g_pfj[j].c;
        // 16 rows per tile (four kernel row groups), tiles round-robin over threads.
        int64_t G = g_pf_mt; if (G < 1) G = 1;
        if (G > mgroups) G = mgroups;
        const int64_t mtiles = mgroups / G;
        const int64_t atile = G * g_pfa_rowstride;
        const bool use_tcm = g_tcm[tid] != nullptr && atile <= g_tcm_size;
        for (int64_t mt = tid; mt < mtiles; mt += g_participants) {
            const uint8_t * asrc = g_pfa + (g_pfa_jslot[j] * mgroups + mt * G) * g_pfa_rowstride;
            const uint8_t * abase = asrc;
            if (use_tcm) {
                memcpy(g_tcm[tid], asrc, (size_t) atile);
                abase = g_tcm[tid];
            }
            for (int64_t a = 0; a < nb; ++a) {
                // MEGA_BHOT=1 keeps one 36 KB B column block for every column:
                // the results are garbage, but it isolates whether the GEMM is
                // limited by streaming B or by the kernel's own issue rate.
                const uint8_t * bcol = b + (getenv("MEGA_BHOT") ? 0 : a * kb * 576);
                for (int64_t g = 0; g < G; ++g) {
                    float * dst = c + (mt * G + g) * 4 * ldc + a * 32;
                    const uint8_t * ap = abase + g * g_pfa_rowstride;
                    if (g_pf_vendor == 2)
                        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m4(
                            256, ap, bcol, nullptr, dst, 4, 32, (size_t) (kb / 8), (size_t) ldc);
                    else
                        spacemit_kernels::ime2::gemm_kernel_i8i4(
                            32, ap, bcol, nullptr, dst, 4, 32, (size_t) kb, (size_t) ldc);
                }
            }
        }
    }
}

// The activation quantizer is a quarter of prefill time when it runs on one
// core, so it gets the same worker pool as the GEMM.
static const float * g_qa_src;
static int64_t       g_qa_rows, g_qa_k;
static int64_t       g_qa_j;
static void pf_quant_band(int64_t lo, int64_t hi) {
    for (int64_t g = lo; g < hi; ++g) {
        void * dst = g_pfa + (g_qa_j * (g_qa_rows / 4) + g) * g_pfa_rowstride;
        const float * src = g_qa_src + g * 4 * g_qa_k;
        if (g_pf_vendor == 2)
            spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, src, (size_t) g_qa_k, (uint8_t *) dst);
        else
            spacemit_kernels::rvv::quantize_a_4row_i8(32, src, (size_t) g_qa_k, (uint8_t *) dst);
    }
}
static void task_pf_quant(int tid) {
    const int64_t mg = g_qa_rows / 4;
    for (int64_t g = tid; g < mg; g += g_participants) pf_quant_band(g, g + 1);
}
static void quant_a_4row(const float * a, int64_t rows, int64_t k, int64_t j) {
    // One 4-row record per k block is 152 B: [4 x f32 a_scale][4 x i16 a_sum]
    // [4 x 32 int8]. The vendor asm advances A by 16+8+128, so 144 overlaps the
    // next record by 8 bytes.
    //
    // The high-precision variant works on K256 superblocks instead: 8 subblocks
    // of 136 B (4 x fp16 scale + 4 x 32 int8) and a 72 B trailer of per-subblock
    // sums, 1160 B per 4-row group per 256 k. It keeps the fp32 accumulators in
    // v28-v31 for the whole superblock and applies each K32 scale inside, which
    // measures 47.7 GMAC/s against the K32 kernel's 33.3 - same A layout on both
    // sides of the accumulator, one third of the block-loop overhead.
    g_pfa_rowstride = g_pf_vendor == 2 ? (k / 256) * 1160 : (k / 32) * 152;
    g_qa_src = a;
    g_qa_rows = rows;
    g_qa_k = k;
    g_qa_j = j;
    if (g_elq) { int wa, wx; el_split(g_elq, &wa, &wx);
                 parallel_for_w(g_qa_rows / 4, pf_quant_band, wa, wx); }
    else run(task_pf_quant);
}

// ------------------------------------------------------------------ int8 dot GEMM
// Prefill at 10K is compute bound, not bandwidth bound: the 1.36 GB of Q4_0
// weights stream in ~43 ms while the GEMM itself takes tens of seconds. So the
// weight format should optimise products-per-instruction, not bytes per weight.
//
// `vmadot` (int8 x int8) does 512 products per instruction on this core, while
// the real int8xint4 form needs vupack + vsll + two macs for 1024 products, i.e.
// 341. Density wins, so prefill uses an int8 copy of the weights.
//
// Numerics, measured on blk.0.ffn_gate.weight against the F16 master:
//   Q4_0 weights  -> SNR 20.98 dB vs F16
//   int8 per-row  -> SNR 20.88 dB vs F16
// i.e. requantising adds -42 dB of its own noise on top of Q4_0's, so the swap
// is numerically neutral. Per-row scales (one fp16 per output row, one f32 per
// activation row) are what make this cheap: the scale is constant over K, so the
// kernel is a pure dot loop with a single epilogue.
//
// Operand map, verified bit-exact over all 512 one-hot pairs:
//   lane = row*8 + col, A at row*16+d, B at col*16+d, 16 products per lane.
// The destination must be an even register, so 8 accumulators / 64 columns is
// the hard ceiling and 4 rows x 64 cols is the widest shape available.
static int g_i8w = 0;                        // prefill uses int8 weights
static uint8_t * g_blob8 = nullptr;          // the 2.2 GB int8 copy, prefill only
static int64_t g_i8_blk(int64_t k) { return k / 16; }

static void repack_b_i8(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t nb = k / 32, rs = src_row_bytes(k), kst = k / 16;
    int8_t *   pay = (int8_t *) dst;
    // Scales are stored pre-expanded to 32 lanes: a lane holds (row, col) as
    // row*8+col, so the same 8 column scales repeat four times. Doing it here
    // removes a per-iteration gather from the GEMM epilogue.
    float *    scl = (float *) (dst + (size_t) rows * k);
    static float row[8192];
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t sr = perm ? (int64_t) perm[r] : r;
        dequant_row(row, src + sr * rs, k);
        float mx = 0;
        for (int64_t i = 0; i < k; ++i) mx = fmaxf(mx, fabsf(row[i]));
        const float sc = mx > 0 ? mx / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        const int64_t cb = r / 64, cc = r % 64;
        // column r is block nb=r/64, group cg=(r%64)/8, position cc=r%8; the
        // lane is row*8+col so lanes with lane%8 == cc carry this scale
        {
            const int64_t sg = (cb * 8 + cc / 8) * 32;
            for (int lane = (int) (cc % 8); lane < 32; lane += 8) scl[sg + lane] = sc;
        }
        for (int64_t i = 0; i < k; ++i) {
            int v = (int) lrintf(row[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            pay[((cb * kst + i / 16) << 10) + cc * 16 + (i % 16)] = (int8_t) v;
        }
    }
}

// The prefill int8 copy has to be built from whatever the decode path actually
// uses, otherwise the two halves of the chain run different numbers. With -q2
// that source is the 2-bit blob, so this walks it back to fp32 first.
static void repack_b_i8_q2(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k) {
    const int64_t nsup = k / Q2_K, kst = k / 16;
    int8_t * pay = (int8_t *) dst;
    float *  scl = (float *) (dst + (size_t) rows * k);
    static float row[8192];
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t grp = r / 32, i = r % 32;
        for (int64_t sup = 0; sup < nsup; ++sup) {
            const uint8_t * blk = src + (grp * nsup + sup) * Q2_BLK;
            const float d  = (float) ((const _Float16 *) (blk + 512 + 2048))[i];
            const float dm = (float) ((const _Float16 *) (blk + 512 + 2048 + 64))[i];
            for (int j = 0; j < 16; ++j) {
                const float s = (float) (blk[j * 32 + i] & 0x0F) * d;
                const float z = (float) (blk[16 * i + j] >> 4) * dm;
                for (int b2 = 0; b2 < 16; ++b2) {
                    const int q = (blk[512 + (j / 4) * 512 + i * 16 + b2] >> ((j % 4) * 2)) & 3;
                    row[sup * 256 + j * 16 + b2] = (float) q * s - z;
                }
            }
        }
        float mx = 0;
        for (int64_t x = 0; x < k; ++x) mx = fmaxf(mx, fabsf(row[x]));
        const float sc = mx > 0 ? mx / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        const int64_t cb = r / 64, cc = r % 64;
        {
            const int64_t sg = (cb * 8 + cc / 8) * 32;
            for (int lane = (int) (cc % 8); lane < 32; lane += 8) scl[sg + lane] = sc;
        }
        for (int64_t x = 0; x < k; ++x) {
            int v = (int) lrintf(row[x] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            pay[((cb * kst + x / 16) << 10) + cc * 16 + (x % 16)] = (int8_t) v;
        }
    }
}

// The 3-bit analogue of the walker above. Without it a -q3 model has no prefill:
// the vendor helper gemm_kernel_i8i3k is a one-row kernel (four rows come back
// zero, and count_m = 4 corrupts even row 0), while the K256 hp kernel that
// prefill actually calls is 4-bit and strides B by the 576-byte Q4_0 block. Feed
// it a 3-bit blob and every prompt token is computed from garbage weights, which
// then poisons the KV cache that decode reads. Walking the 3-bit blob back to
// int8 keeps prefill and decode on the same weights and costs 21% more B traffic
// than the 3-bit blob would.
static void repack_b_i8_q3(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k) {
    const int64_t nsup = k / Q3_K, kst = k / 16;
    int8_t * pay = (int8_t *) dst;
    float *  scl = (float *) (dst + (size_t) rows * k);
    static float row[8192];
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t grp = r / 32, i = r % 32;
        for (int64_t sup = 0; sup < nsup; ++sup) {
            const uint8_t * blk = src + (grp * nsup + sup) * Q3_BLK;
            const float s16 = (float) ((const _Float16 *) (blk + 512 + 1024 + 2048))[i];
            for (int j = 0; j < 16; ++j) {
                const float s  = (float) ((const int8_t *) blk)[j * 32 + i] * s16;
                const uint16_t hm = ((const uint16_t *) (blk + 512))[j * 32 + i];
                for (int b2 = 0; b2 < 16; ++b2) {
                    const int u = (blk[512 + 1024 + (j / 4) * 512 + i * 16 + b2] >> ((j % 4) * 2)) & 3;
                    const int q3 = (hm >> b2) & 1 ? u : u - 4;
                    row[sup * 256 + j * 16 + b2] = (float) q3 * s;
                }
            }
        }
        float mx = 0;
        for (int64_t x = 0; x < k; ++x) mx = fmaxf(mx, fabsf(row[x]));
        const float sc = mx > 0 ? mx / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        const int64_t cb = r / 64, cc = r % 64;
        {
            const int64_t sg = (cb * 8 + cc / 8) * 32;
            for (int lane = (int) (cc % 8); lane < 32; lane += 8) scl[sg + lane] = sc;
        }
        for (int64_t x = 0; x < k; ++x) {
            int v = (int) lrintf(row[x] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            pay[((cb * kst + x / 16) << 10) + cc * 16 + (x % 16)] = (int8_t) v;
        }
    }
}

// 4 rows x 64 cols, K = ksteps*16. Both loads are whole-register (EEW-explicit,
// because `vl8r.v` is an e8 alias and the dots need SEW=e32).
static void gemm_i8d_4x64(const int8_t * A, const int8_t * B, int32_t * C, int64_t ksteps) {
    asm volatile(
        "vsetvli t0, x0, e32, m1   \n\t"
        "vxor.vv v16, v16, v16     \n\t"
        "vxor.vv v18, v18, v18     \n\t"
        "vxor.vv v20, v20, v20     \n\t"
        "vxor.vv v22, v22, v22     \n\t"
        "vxor.vv v24, v24, v24     \n\t"
        "vxor.vv v26, v26, v26     \n\t"
        "vxor.vv v28, v28, v28     \n\t"
        "vxor.vv v30, v30, v30     \n\t"
        "mv     t3, %[KS]          \n\t"
        "I8L%=:                    \n\t"
        "vl1re32.v v0, (%[A])       \n\t"
        "vl8re32.v v8, (%[B])       \n\t"
        "addi   %[A], %[A], 64      \n\t"
        "addi   %[B], %[B], 1024    \n\t"
        "vmadot v16, v0, v8         \n\t"
        "vmadot v18, v0, v9         \n\t"
        "vmadot v20, v0, v10        \n\t"
        "vmadot v22, v0, v11        \n\t"
        "vmadot v24, v0, v12        \n\t"
        "vmadot v26, v0, v13        \n\t"
        "vmadot v28, v0, v14        \n\t"
        "vmadot v30, v0, v15        \n\t"
        "addi   t3, t3, -1          \n\t"
        "bgtz   t3, I8L%=           \n\t"
        "vse32.v v16, (%[C0])       \n\t"
        "vse32.v v18, (%[C1])       \n\t"
        "vse32.v v20, (%[C2])       \n\t"
        "vse32.v v22, (%[C3])       \n\t"
        "vse32.v v24, (%[C4])       \n\t"
        "vse32.v v26, (%[C5])       \n\t"
        "vse32.v v28, (%[C6])       \n\t"
        "vse32.v v30, (%[C7])       \n\t"
        : [A] "+r"(A), [B] "+r"(B)
        : [KS] "r"(ksteps),
          [C0] "r"(C), [C1] "r"(C + 32), [C2] "r"(C + 64), [C3] "r"(C + 96),
          [C4] "r"(C + 128), [C5] "r"(C + 160), [C6] "r"(C + 192), [C7] "r"(C + 224)
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

static int g_x8_want = 0;    // the pending GEMM has a matching X100 job
static int8_t * g_a8x;      // activations in X100 order, 4*k bytes per group
static int8_t * g_xb[8];    // packed weights for each X100 job
static size_t   g_xb_cap[8];
static float  * g_a8xs;     // 4 row scales per 4-row group
static int64_t  g_x_rowstride;

// Activation packing: one record per 4-row group = [4 x f32 scale][4 rows x K int8]
// laid out per 16-element k-step as 4x16 = 64 bytes.
static const float * g_a8_src;
static int64_t g_a8_rows, g_a8_k, g_a8_j;
static int64_t g_a8_stride;
// One activation scale per 256 k-elements per row. Per-row scaling had to cover
// 2048 values including outliers, so its step was ~10x coarser than llama.cpp's
// per-block scheme and V at layer 0 came out -0.0365 against a -0.0025 reference.
// 256 (16 vmadot steps) measured 0.0048 against per-32's 0.0021 and per-row's
// 0.0364, i.e. most of the accuracy for a flush every 16 steps instead of 2.
static const int A_BLK = 256, A_BSTEP = A_BLK / 16;
// lane = row*8 + col, so the four row scales repeat eight times each
static const uint32_t ROWIDX[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                     2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
// Same map widened to 128 lanes: the four cg groups of a 64-column tile repeat
// the one 32-lane pattern, so a single m4 gather expands four row scales.
static const uint32_t * rowidx4(void) {
    static uint32_t t[128];
    static int init = 0;
    if (!init) { for (int i = 0; i < 128; ++i) t[i] = ROWIDX[i & 31]; init = 1; }
    return t;
}
static int64_t g_a8_nblk;
static void task_pf_quant_i8(int tid) {
    const int64_t k = g_a8_k, kst = k / 16, mg = g_a8_rows / 4;
    const int64_t nblk = k / A_BLK;
    for (int64_t g = tid; g < mg; g += g_participants) {
        uint8_t * rec = g_pfa + (g_a8_j * mg + g) * g_a8_stride;
        int8_t *  pay = (int8_t *) rec;
        float *   sc  = (float *) (rec + kst * 64);      // [block][row], 4 per block
        float rowmax[4], rowsc[4];
        for (int r = 0; r < 4; ++r) rowmax[r] = 0;
        // pass 1: one max per (row, 256-element block); the row max falls out too
        for (int64_t b = 0; b < nblk; ++b) {
            for (int r = 0; r < 4; ++r) {
                const float * a = g_a8_src + ((g * 4 + r) * k) + b * A_BLK;
                float mx = 0;
                for (int64_t i = 0; i < A_BLK; ) {
                    const size_t vl = __riscv_vsetvl_e32m4((size_t) (A_BLK - i));
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(a + i, vl);
                    v = __riscv_vfabs_v_f32m4(v, vl);
                    const vfloat32m1_t red =
                        __riscv_vfredmax_vs_f32m4_f32m1(v, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
                    const float m = __riscv_vfmv_f_s_f32m1_f32(red);
                    if (m > mx) mx = m;
                    i += (int64_t) vl;
                }
                sc[b * 4 + r] = mx > 0 ? mx / 127.0f : 1.0f;
                if (mx > rowmax[r]) rowmax[r] = mx;
            }
        }
        for (int r = 0; r < 4; ++r) {
            rowsc[r] = rowmax[r] > 0 ? rowmax[r] / 127.0f : 1.0f;
            if (g_x8_want && g_a8_j == 0) g_a8xs[g * 4 + r] = rowsc[r];
        }
        // pass 2: quantise against the block's own scale
        for (int r = 0; r < 4; ++r) {
            const float * a = g_a8_src + ((g * 4 + r) * k);
            const float xr = rowsc[r] > 0 ? 1.0f / rowsc[r] : 1.0f;   // X100 keeps a row scale
            for (int64_t t = 0; t < kst; ++t) {
                const size_t vl = __riscv_vsetvl_e32m1(16);
                const int64_t b = (t / A_BSTEP);
                const float s = sc[b * 4 + r];
                vfloat32m1_t v = __riscv_vle32_v_f32m1(a + t * 16, vl);
                vfloat32m1_t qf = __riscv_vfmul_vf_f32m1(v, 1.0f / s, vl);
                vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(qf, vl);
                q = __riscv_vmax_vx_i32m1(q, -127, vl);
                q = __riscv_vmin_vx_i32m1(q, 127, vl);
                const vint16mf2_t b16 = __riscv_vncvt_x_x_w_i16mf2(q, vl);
                const vint8mf4_t b8 = __riscv_vncvt_x_x_w_i8mf4(b16, vl);
                __riscv_vse8_v_i8mf4(pay + t * 64 + r * 16, b8, vl);
                if (g_x8_want && g_a8_j == 0) {
                    // The X100 tail columns keep one scale per row, so requantise
                    // there rather than reusing the block-scaled bytes.
                    vfloat32m1_t xf = __riscv_vfmul_vf_f32m1(v, xr, vl);
                    vint32m1_t xq = __riscv_vfcvt_x_f_v_i32m1(xf, vl);
                    xq = __riscv_vmax_vx_i32m1(xq, -127, vl);
                    xq = __riscv_vmin_vx_i32m1(xq, 127, vl);
                    const vint8mf4_t x8 =
                        __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(xq, vl), vl);
                    int8_t * xo = g_a8x + g * g_x_rowstride + t * 64 + r * 8;
                    const size_t v8 = __riscv_vsetvl_e8mf4(8);
                    __riscv_vse8_v_i8mf4(xo, x8, v8);
                    __riscv_vse8_v_i8mf4(xo + 32,
                        __riscv_vslidedown_vx_i8mf4(x8, 8, (size_t) 16), v8);
                    __riscv_vsetvl_e32m1((size_t) 16);
                }
            }
        }
    }
    (void) kst;
}
static inline void quant_a_i8d(const float * a, int64_t rows, int64_t k, int64_t j) {
    // Payload, then 4 x 32 pre-expanded row scales (one 32-lane copy per row of
    // the 4-row group) = 512 B. Reserving only 128 B let each group's scale
    // block overwrite the first 384 B of the next group's payload, i.e. the
    // first 96 k-elements of every group were garbage on top of quantisation.
    // Payload is (k/16) 64-byte steps; the scales are one f32 per (4-row group,
    // k-block) pair, i.e. 16 B per block. The old fixed 512 B reserved room for
    // only 32 blocks, so a 6144-wide down projection at A_BLK=64 wrote 1.5 KB
    // past the record and every following group's payload was destroyed.
    g_a8_stride = (k / 16) * 64 + (k / A_BLK) * 16 + 64;
    g_a8_nblk = k / A_BLK;
    g_a8_src = a; g_a8_rows = rows; g_a8_k = k; g_a8_j = j;
    run(task_pf_quant_i8);
}
// Vendor-precision prefill: the same m4 i8i4 kernel the decode path runs, which
// keeps the Q4_0 block scales exactly and quantises activations per 32 k
// elements. The vmadot path re-quantises the dequantised weights to int8 with a
// per-row scale, which measured 2.0% on q/k/v and 9.9% on the activation side,
// and a four-row chunk then drifted 17.8% away from the f32 reference by layer
// 41 -- enough to move the argmax. Exactness first, then win the speed back.
// The vendor hp quantiser writes only the A100 record, so the X100 A copy needs
// its own pass. Layout is the one x100_gemm_4x16 reads (probe_x100_map): row r of
// a four-row group lands at r*8 within each 64-byte step, and the 16 k values of
// one step split into two 8-byte halves at +0 and +32.
static const float * g_xq_src;
static int64_t g_xq_k, g_xq_rows;
static void task_x8_quant(int tid) {
    const int64_t k = g_xq_k, kst = k / 16, mg = g_xq_rows / 4;
    for (int64_t g = tid; g < mg; g += g_participants) {
        float rowsc[4];
        for (int r = 0; r < 4; ++r) {
            const float * a = g_xq_src + (g * 4 + r) * k;
            float mx = 0;
            for (int64_t i = 0; i < k; ) {
                const size_t vm = __riscv_vsetvl_e32m4((size_t) (k - i));
                const vfloat32m4_t v =
                    __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(a + i, vm), vm);
                const vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(
                    v, __riscv_vfmv_s_f_f32m1(0.0f, 1), vm);
                const float m = __riscv_vfmv_f_s_f32m1_f32(red);
                if (m > mx) mx = m;
                i += (int64_t) vm;
            }
            rowsc[r] = mx > 0 ? mx / 127.0f : 1.0f;
            g_a8xs[g * 4 + r] = rowsc[r];
        }
        int8_t * xbase = g_a8x + g * g_x_rowstride;
        for (int r = 0; r < 4; ++r) {
            const float * a = g_xq_src + (g * 4 + r) * k;
            const float xr = 1.0f / rowsc[r];
            for (int64_t t = 0; t < kst; ++t) {
                const size_t vl = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v = __riscv_vle32_v_f32m1(a + t * 16, vl);
                vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(__riscv_vfmul_vf_f32m1(v, xr, vl), vl);
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
}
static void x100_quant_a(const float * a, int64_t rows, int64_t k) {
    g_xq_src = a; g_xq_k = k; g_xq_rows = rows;
    run(task_x8_quant);
}

static inline void pf_quant(const float * a, int64_t rows, int64_t k, int64_t j) {
    // set here -- on the single thread that dispatches -- rather than in
    // x100_prepare, which ran after the quantiser and left the first layer writing
    // every row group through a stale (zero) stride.
    if (g_x8_want) g_x_rowstride = 4 * k;
    if (g_pf_vendor) quant_a_4row(a, rows, k, j);
    else quant_a_i8d(a, rows, k, j);
    // The vendor quantiser emits only the A100 record; the vmadot one already
    // writes the X100 copy inline. Building it here for the vendor path is what
    // makes -x100 produce a result at all -- it used to read untouched hugemem.
    if (g_x8_want && j == 0 && g_pf_vendor) x100_quant_a(a, rows, k);
}


// One GEMM over a weight tile: for each 64-column block, sweep every 4-row group
// so the 1 KB weight tile is fetched once and reused. That loop order is worth
// 2x versus n-inner (measured 151 vs 74 GMAC/s).
//
// The epilogue is 27% of prefill GEMM time if left scalar: the kernel writes 256
// int32 per (row group, column block) and the naive version moves them one at a
// time. Vectorised it is 8 cvt + 8 mul + 4x8 lane moves, and the per-lane scale
// products are hoisted out because they only depend on the column block.
static int32_t g_i8c[8][256];
static float   g_i8f[8][256];   // float accumulator between scale blocks
static float   g_i8t[8][256];
static float   g_a8s[8][4];
static const int64_t CH = 8;              // row groups per scheduling chunk
// One counter per (job, column block). Sharing an index across jobs silently
// starved every job after the first: the counter for column block 0 was already
// past mgroups when job 1 started, so K and V came out all zeros while Q was fine.
static const int NCHUNK = 512;
static std::atomic<int64_t> g_chunk[NCHUNK];
static int g_noepi = 0;
static int64_t g_pf_tile = 1 << 24;  // row groups per thread before re-reading B

static void task_pf_gemm_i8(int tid) {
    const int64_t mgroups = g_pf_rows / 4;
    int64_t cbase = 0;
    for (int j = 0; j < g_pfjn; ++j) {
        const int64_t nown = g_pfj[j].nlim ? g_pfj[j].nlim : g_pfj[j].n;
        const int64_t nblk = nown / 64, kst = g_pfj[j].k / 16;
        const int64_t cj = cbase;
        cbase += nblk;
        if (g_dbg) fprintf(stderr, "[a100] j=%d nown=%lld nblk=%lld\n", j, (long long) nown, (long long) nblk);
        const int64_t ldc = g_pfj[j].ldc;
        const uint8_t * b = g_pfj[j].b;
        float * c = g_pfj[j].c;
        // Work is handed out in row-group chunks through a per-column-block atomic
        // counter rather than sliced evenly. The two clusters are not equal: cores
        // 0-3 (A100) measure about 75 GMAC/s each and cores 4-7 (X100) about 99, so
        // an even split leaves the fast cluster waiting at the barrier. Dynamic
        // chunks let the faster cores simply take more, with no calibration.
        // Column splitting was tried and is much worse (226 vs 447 GMAC/s): each
        // thread then owns its own B block, so the weight stream stops being shared
        // through L2 and every thread re-reads the activations.
        for (int64_t nb = 0; nb < nblk; ++nb) {
            const int8_t * bcol = (const int8_t *) b + nb * kst * 1024;
            const float * bsv = (const float *) (b + (size_t) g_pfj[j].n * g_pfj[j].k) + nb * 256;
            float * crow = c + nb * 64;
            for (;;) {
                const int64_t rt = g_chunk[cj + nb].fetch_add(CH, std::memory_order_relaxed);
                if (rt >= mgroups) break;
                int64_t rend = rt + CH;
                if (rend > mgroups) rend = mgroups;
                for (int64_t g = rt; g < rend; ++g) {
                    const uint8_t * arec = g_pfa + (g_pfa_jslot[j] * mgroups + g) * g_a8_stride;
                    const float * asc = (const float *) (arec + kst * 64);
                    if (g_noepi) {
                        gemm_i8d_4x64((const int8_t *) arec, bcol, g_i8c[tid], kst);
                        c[(g * 4) + nb] = (float) g_i8c[tid][0];
                        continue;
                    }
                    // One vmadot run per scale block, flushed into f32 with that
                    // block's four row scales. The scale vector repeats each row's
                    // scale over its eight columns (lane = row*8 + col).
                    float * facc = g_i8f[tid];
                    memset(facc, 0, 256 * 4);
                    const vuint32m1_t ridx = __riscv_vle32_v_u32m1(ROWIDX, 32);
                    for (int64_t b = 0; b < kst; b += A_BSTEP) {
                        const int64_t ns = (kst - b < A_BSTEP) ? (kst - b) : A_BSTEP;
                        gemm_i8d_4x64((const int8_t *) arec + b * 64, bcol + b * 1024,
                                      g_i8c[tid], ns);
                        // four contiguous row scales -> 32 lanes of (row*8+col) order
                        const vfloat32m1_t s4 = __riscv_vle32_v_f32m1(asc + (b / A_BSTEP) * 4, 4);
                        // The flush is charged once per scale block, so at A_BLK=64 it
                        // runs four times as often as at 256 and the m1 version (6 ops
                        // per 32 lanes, 48 per flush) cost 9.3x the whole GEMM. Same
                        // arithmetic in m4: 128 lanes per op, 12 ops per flush.
                        const size_t vl4 = __riscv_vsetvl_e32m4(128);
                        const vfloat32m4_t sv4 = __riscv_vrgather_vv_f32m4(
                            __riscv_vlmul_ext_v_f32m1_f32m4(s4),
                            __riscv_vle32_v_u32m4(rowidx4(), vl4), vl4);
                        for (int h4 = 0; h4 < 2; ++h4) {
                            const vint32m4_t iv = __riscv_vle32_v_i32m4(g_i8c[tid] + h4 * 128, vl4);
                            vfloat32m4_t fv = __riscv_vfcvt_f_x_v_f32m4(iv, vl4);
                            fv = __riscv_vfmul_vv_f32m4(fv, sv4, vl4);
                            const vfloat32m4_t acc = __riscv_vle32_v_f32m4(facc + h4 * 128, vl4);
                            __riscv_vse32_v_f32m4(facc + h4 * 128, __riscv_vfadd_vv_f32m4(acc, fv, vl4), vl4);
                        }
                    }
                    float * base = crow + (g * 4) * ldc;
                    for (int cg = 0; cg < 8; ++cg) {
                        size_t vl = __riscv_vsetvl_e32m1(32);
                        const vfloat32m1_t fv = __riscv_vfmul_vv_f32m1(
                            __riscv_vle32_v_f32m1(facc + cg * 32, vl),
                            __riscv_vle32_v_f32m1(bsv + cg * 32, vl), vl);
                        for (int r = 0; r < 4; ++r) {
                            const vfloat32m1_t sh = r ? __riscv_vslidedown_vx_f32m1(fv, (size_t) (r * 8), vl) : fv;
                            vl = __riscv_vsetvl_e32m1(8);
                            __riscv_vse32_v_f32m1(base + r * ldc + cg * 8, sh, vl);
                            vl = __riscv_vsetvl_e32m1(32);
                        }
                    }
                }
            }
        }
    }
}


// ------------------------------------------------------------------ X100 GEMM
// Cores 0-7 (VLEN=256) sit idle while prefill runs its GEMMs on the A100 cluster.
// Their vmadot is 2 rows x 4 cols x 8 depth = 64 MAC in eight lanes, a quarter of
// A100's shape, so bench_x100_gemm measures ~260 GMAC/s aggregate against A100's
// ~840. Not a competitor, but a fifth of the machine doing nothing.
//
// Geometry, from probe_x100_map and re-checked on every build:
//   A byte  0..7  -> lanes 0..3, first row of the pair
//   A byte  8..15 -> lanes 4..7, second row
//   A byte 16..31 -> never read
// so one 32-byte A register carries two rows x 8 depth, and the kernel reads two
// overlapping 32-byte windows per step: rows 0-1 at +0, rows 2-3 at +16.
//
// Because the A100 record is already int8 with one scale per output row, the port
// needs no requantisation at all: both copies are byte permutations of the data
// the A100 path just built, and both are the same size. That keeps the numerics
// identical in kind (per-row activation scale, per-row weight scale) instead of
// introducing a second quantisation scheme.
// g_x100 is declared above the prefill GEMM (it decides how many columns the A
// cluster runs); -Y turns it on.
// One percentage shared by every matrix is the wrong shape. X100's own rate
// depends on k (measured ~198 GMAC/s at k=6144, ~287 at k=2048) while the A100
// cluster holds ~486-509 on all four, so the equal-time split is a different
// column share for each projection: qkv 35, o 33, gate+up 37, down 28. A single
// 40 made `down` slower than a single 23 did (10.68 s against 8.92 s) because
// X100 became its critical path.
static int g_x100_pct = 23;     // X100 takes this share of the columns
static int g_xpct_explicit = 0; // -x100 N was given, so it wins over the table
static int g_x100_pct_tab[4] = { 35, 33, 37, 28 };   // qkv, o, gate+up, down
static int g_xpct_j[4] = { 23, 23, 23, 23 };         // per-job share for x100_prepare
// Weight source for the X100 columns: the plain int8 copy, not the vendor record
// the A100 kernel reads. x100_prepare repacks it into X100 order.
static const uint8_t * g_xsrc[4];

static void task_pf_gemm_x100(int id);
static void * hugemem(size_t n);


struct XJ {
    const int8_t * b;           // packed weights for this matrix's X100 columns
    int64_t nsteps, nb, ldc, ja, c0;   // c0 = first output column this cluster owns
    float * c;
    const float * bsc;          // nb * 32 pre-duplicated column scales
};
static XJ g_xj[8];
static int g_xjn;
static std::atomic<int64_t> g_xchunk[4];
static std::atomic<int> g_xg_gen{0}, g_xg_done{0};
static int g_x_nt = 0;
static int g_xchk = 0;
static int32_t g_xc[16][64];
static const int64_t XTILE = 256;   // 4-row groups per scheduling tile

// 4 rows x 16 columns, K = nstep*8. Destinations must be even registers.
static void x100_gemm_4x16(int32_t * out, const int8_t * A, const int8_t * B, int64_t nstep) {
    asm volatile(
        "vsetvli t0, x0, e32, m1   \n\t"
        "vxor.vv v16, v16, v16     \n\t"
        "vxor.vv v18, v18, v18     \n\t"
        "vxor.vv v20, v20, v20     \n\t"
        "vxor.vv v22, v22, v22     \n\t"
        "vxor.vv v24, v24, v24     \n\t"
        "vxor.vv v26, v26, v26     \n\t"
        "vxor.vv v28, v28, v28     \n\t"
        "vxor.vv v30, v30, v30     \n\t"
        "mv     t3, %[NS]          \n\t"
        "mv     t4, %[A]           \n\t"
        "mv     t5, %[B]           \n\t"
        "1:                        \n\t"
        "vle32.v v0, (t4)          \n\t"   // rows 0-1 (rows 2-3 land in the dead half)
        "addi    t6, t4, 16        \n\t"
        "vle32.v v1, (t6)          \n\t"   // rows 2-3
        "addi    t4, t4, 32        \n\t"
        "vle32.v v2, (t5)          \n\t"   // columns 0-3
        "addi    t5, t5, 32        \n\t"
        "vle32.v v3, (t5)          \n\t"   // columns 4-7
        "addi    t5, t5, 32        \n\t"
        "vle32.v v4, (t5)          \n\t"   // columns 8-11
        "addi    t5, t5, 32        \n\t"
        "vle32.v v5, (t5)          \n\t"   // columns 12-15
        "addi    t5, t5, 32        \n\t"
        "vmadot  v16, v0, v2       \n\t"
        "vmadot  v18, v0, v3       \n\t"
        "vmadot  v20, v0, v4       \n\t"
        "vmadot  v22, v0, v5       \n\t"
        "vmadot  v24, v1, v2       \n\t"
        "vmadot  v26, v1, v3       \n\t"
        "vmadot  v28, v1, v4       \n\t"
        "vmadot  v30, v1, v5       \n\t"
        "addi    t3, t3, -1        \n\t"
        "bgtz    t3, 1b            \n\t"
        "mv     t1, %[O]           \n\t"
        "vse32.v v16, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v18, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v20, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v22, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v24, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v26, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v28, (t1)         \n\t"
        "addi    t1, t1, 32        \n\t"
        "vse32.v v30, (t1)         \n\t"
        : /* no outputs */
        : [NS] "r"(nstep), [A] "r"(A), [B] "r"(B), [O] "r"(out)
        : "t0", "t1", "t3", "t4", "t5", "t6",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// A record -> X100 order. The A100 record is, per 16-k step, four rows of 16
// bytes; X100 wants the four rows' 8-byte halves adjacent. Both are exactly 64
// bytes per 16 k, so this is a permutation of eight 8-byte blocks and the buffer
// does not grow.
static std::atomic<int> g_xdbg_threads{0};
static double g_xdbg_a, g_xdbg_b;
// One matrix of weights, X100 column share only. Same reasoning as above: the
// payload is k bytes per column either way, only the order changes.
static int8_t * g_xp_src;
static int64_t  g_xp_k, g_xp_n, g_xp_c0, g_xp_nb;
static float *  g_xp_bsc;
static int8_t * g_xp_dst;

static void task_x8_pack_b(int tid) {
    const int64_t kst = g_xp_k / 16;
    const int64_t ds  = g_xp_k / 8;
    const float * bsvbase = (const float *) (g_xp_src + (size_t) g_xp_n * g_xp_k);
    const int64_t nb0 = g_xp_nb * tid / g_participants;
    const int64_t nb1 = g_xp_nb * (tid + 1) / g_participants;
    for (int64_t nb = nb0; nb < nb1; ++nb) {
        const int64_t cbase = g_xp_c0 + nb * 16;
        const int64_t cb = cbase / 64, lc0 = cbase % 64;
        const int8_t * srcblk = g_xp_src + cb * kst * 1024;
        const float * bsv = bsvbase + cb * 256;
        int8_t * d = g_xp_dst + nb * ds * 128;
        float * bsc = g_xp_bsc + nb * 32;
        for (int64_t t2 = 0; t2 < ds; ++t2) {
            int8_t * out = d + t2 * 128;
            for (int i = 0; i < 16; ++i) {
                const int8_t * s = srcblk + (t2 / 2) * 1024 + (lc0 + i) * 16 + (t2 % 2) * 8;
                memcpy(out + (i / 4) * 32 + (i % 4) * 8, s, 8);
            }
        }
        // Duplicate each column scale across the four lanes it covers, so the
        // epilogue is one vle32 with no shuffle.
        for (int cg = 0; cg < 4; ++cg)
            for (int lane = 0; lane < 8; ++lane) {
                const int64_t lc = lc0 + cg * 4 + (lane % 4);
                bsc[cg * 8 + lane] = bsv[(lc / 8) * 32 + (lc % 8)];
            }
    }
}

static double g_xt_pack, g_xt_run;

// The X100 A copy is written by the activation quantiser itself, so all this has
// to do is size the buffers once and keep g_x_rowstride in step with the matrix
// about to run.
static void x100_alloc(void) {
    const int64_t ng = g_pf_rows / 4;
    if (g_a8x) return;
    const int64_t bytes = ng * (int64_t) 4 * N_FF;      // widest A: down, k = N_FF
    g_a8x  = (int8_t *) hugemem((size_t) bytes);
    g_a8xs = (float *) hugemem((size_t) ng * 4 * sizeof(float));
    if (!g_a8x || !g_a8xs) { fprintf(stderr, "oom x100 A\n"); g_x100 = 0; }
    else use_huge(g_a8x, (size_t) bytes);
}

static void x100_prepare(const PfJob * js, int nj) {
    const double _tp = now_s();
    g_x_rowstride = 4 * js[0].k;
    g_xjn = 0;
    for (int j = 0; j < nj; ++j) {
        const int64_t nx = (js[j].n * g_xpct_j[j] / 100) & ~(int64_t) 63;
        if (nx <= 0) continue;
        const int64_t nb = nx / 16;
        const size_t need = (size_t) nx * js[j].k + 4096;
        if (g_xb_cap[j] < need) {
            if (g_xb[j]) free(g_xb[j]);
            g_xb[j] = (int8_t *) hugemem(need);
            g_xb_cap[j] = g_xb[j] ? need : 0;
            if (!g_xb[j]) { fprintf(stderr, "oom x100 B\n"); g_x100 = 0; return; }
        }
        // task_x8_pack_b decodes the plain int8 copy (payload plus per-8-column
        // float scales), not the vendor i8i4 record the A100 kernel reads: the two
        // blobs describe the same Q4_0 weights with different byte layouts. Feeding
        // it js[j].b was why -x100 emitted a wrong token from step 0.
        g_xp_src = (int8_t *) (uintptr_t) g_xsrc[j]; g_xp_k = js[j].k; g_xp_n = js[j].n;
        g_xp_c0 = js[j].n - nx;      // X100 takes the tail columns
        g_xp_nb = nb; g_xp_dst = g_xb[j];
        g_xp_bsc = (float *) (g_xb[j] + (size_t) nx * js[j].k);
        run(task_x8_pack_b);
        g_xj[g_xjn].b = g_xb[j];
        g_xj[g_xjn].nsteps = js[j].k / 8;
        g_xj[g_xjn].nb = nb;
        g_xj[g_xjn].ldc = js[j].ldc;
        g_xj[g_xjn].c = js[j].c;
        // The X100 cluster owns the tail columns, so its outputs start at nlim --
        // not at 0. Omitting this offset wrote its results over the block the A100
        // kernel had just finished, which corrupted every column of the matrix.
        g_xj[g_xjn].c0 = js[j].n - nx;
        g_xj[g_xjn].bsc = g_xp_bsc;
        g_xj[g_xjn].ja = 0;
        ++g_xjn;
    }
    g_xt_pack = now_s() - _tp;
    for (int j = 0; j < 4; ++j) g_xchunk[j].store(0, std::memory_order_relaxed);
    g_xg_done.store(0, std::memory_order_relaxed);
    g_xg_gen.fetch_add(1, std::memory_order_release);
}

static void x100_wait(void) {
    const double _t = now_s();
    while (g_xg_done.load(std::memory_order_acquire) < g_x_nt) sched_yield();
    g_xt_run = now_s() - _t;
}

// Sets the column split for one projection's jobs and prepares the X100 side.
// phase: 0 qkv, 1 o, 2 gate+up, 3 down. srcs are the int8-copy pointers for the
// same matrices; a null entry disables the offload for that job rather than
// silently reading the wrong blob.
static void x100_begin(int phase, PfJob * js, int nj, const uint8_t * const * srcs) {
    const int pct = g_xpct_explicit ? g_x100_pct : g_x100_pct_tab[phase & 3];
    for (int j = 0; j < nj; ++j) {
        if (!srcs[j]) { g_x100 = 0; fprintf(stderr, "x100: no int8 copy for job %d\n", j); return; }
        g_xsrc[j] = srcs[j];
        g_xpct_j[j] = pct;
        js[j].nlim = js[j].n - ((js[j].n * pct / 100) & ~(int64_t) 63);
    }
    x100_prepare(js, nj);
}

static void task_pf_gemm_x100(int id) {
    const int64_t ng = g_pf_rows / 4;
    for (int j = 0; j < g_xjn; ++j) {
        const XJ & J = g_xj[j];
        const int64_t ntiles = (ng + XTILE - 1) / XTILE;
        const int64_t units = ntiles * J.nb;
        // One counter per job. A single shared counter was already past job 1's
        // unit count when job 0 finished, so with nj > 1 every job after the first
        // was skipped and its columns kept whatever the last layer left there.
        for (;;) {
            const int64_t u = g_xchunk[j].fetch_add(1, std::memory_order_relaxed);
            if (u >= units) break;
            const int64_t tile = u / J.nb, nb = u % J.nb;
            int64_t g0 = tile * XTILE, g1 = g0 + XTILE;
            if (g1 > ng) g1 = ng;
            const int64_t amg = g_pf_rows / 4;
            const int8_t * abase = g_a8x;
            (void) amg;
            const int8_t * bp = J.b + nb * J.nsteps * 128;
            const float * bsc = J.bsc + nb * 32;
            float * crow = J.c + J.c0 + nb * 16;
            for (int64_t g = g0; g < g1; ++g) {
                x100_gemm_4x16(g_xc[id], g_a8x + g * g_x_rowstride, bp, J.nsteps);
                // MEGA_XCHK: recompute one four-row by sixteen-column tile in
                // scalar code from the same records the kernel reads. A mismatch
                // here is a layout bug in the A copy, the B pack or the lane map;
                // a match means the error is downstream in the epilogue.
                if (g_xchk && g == g0 && nb == 0) {
                    // g_xc is [row pair][column group][8 lanes], and within a lane
                    // vector 0..3 is the pair's first row and 4..7 the second, so
                    // row r column c sits at (r/2)*32 + (c/4)*8 + (r%2)*4 + (c%4).
                    for (int r = 0; r < 4; ++r) {
                        for (int c = 0; c < 4; ++c) {
                            int32_t acc = 0;
                            for (int64_t s = 0; s < J.nsteps; ++s) {
                                const int8_t * ap = g_a8x + g * g_x_rowstride + s * 32 + r * 8;
                                const int8_t * bq = bp + s * 128 + c * 8;
                                for (int i = 0; i < 8; ++i) acc += (int32_t) ap[i] * (int32_t) bq[i];
                            }
                            const int idx = (r / 2) * 32 + (c / 4) * 8 + (r % 2) * 4 + (c % 4);
                            printf("xchk g%lld r%d c%d kernel %d scalar %d %s\n",
                                   (long long) g, r, c, (int) g_xc[id][idx], (int) acc,
                                   g_xc[id][idx] == acc ? "OK" : "MISMATCH");
                        }
                    }
                    g_xchk = 0;
                }
                const float * as = g_a8xs + g * 4;
                float av[2][8];
                for (int t = 0; t < 4; ++t) {
                    av[0][t] = as[0]; av[0][4 + t] = as[1];
                    av[1][t] = as[2]; av[1][4 + t] = as[3];
                }
                float * base = crow + (g * 4) * J.ldc;
                for (int rp = 0; rp < 2; ++rp) {
                    const size_t vl8 = __riscv_vsetvl_e32m1(8);
                    const vfloat32m1_t avv = __riscv_vle32_v_f32m1(av[rp], vl8);
                    for (int cg = 0; cg < 4; ++cg) {
                        vfloat32m1_t fv = __riscv_vfcvt_f_x_v_f32m1(
                            __riscv_vle32_v_i32m1(g_xc[id] + rp * 32 + cg * 8, vl8), vl8);
                        fv = __riscv_vfmul_vv_f32m1(
                            fv, __riscv_vle32_v_f32m1(bsc + cg * 8, vl8), vl8);
                        fv = __riscv_vfmul_vv_f32m1(fv, avv, vl8);
                        const size_t vl4 = __riscv_vsetvl_e32m1(4);
                        __riscv_vse32_v_f32m1(base + (rp * 2) * J.ldc + cg * 4, fv, vl4);
                        __riscv_vse32_v_f32m1(base + (rp * 2 + 1) * J.ldc + cg * 4,
                                              __riscv_vslidedown_vx_f32m1(fv, 4, vl8), vl4);
                    }
                }
            }
        }
    }
}

static const float * qa_src;
static int64_t qa_k;
static float * g_qa4 = nullptr;   // 4-row activation scratch for the hp decode path
static void task_quant(int tid) {
    const int64_t nb = qa_k / (g_bit3 || g_bit2 ? Q3_K : g_hpdec ? 256 : 32);
    const int64_t per = (nb + g_participants - 1) / g_participants;
    const int64_t b0 = per * tid;
    int64_t b1 = (tid == g_participants - 1) ? nb : per * (tid + 1);
    if (b1 > nb) b1 = nb;
    if (b0 >= b1) return;
    if (g_bit3 || g_bit2)
        spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, qa_src + b0 * Q3_K, (size_t) ((b1 - b0) * Q3_K), g_aq + b0 * 292);
    else if (g_hpdec)
        spacemit_kernels::rvv::quantize_a_row_i8_hp(256, qa_src + b0 * 256, (size_t) ((b1 - b0) * 256), g_aq + b0 * 290);
    else
        spacemit_kernels::rvv::quantize_a_row_i8(32, qa_src + b0 * 32, (size_t) ((b1 - b0) * 32), g_aq + b0 * 38);
}

static void quant_a(const float * a, int64_t k) {
    if (g_bit3 || g_bit2) {
        if (k < Q3_K) { spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a, (size_t) k, g_aq); return; }
        qa_src = a; qa_k = k;
        run(task_quant);
        return;
    }
    if (g_hpdec && k % 256) {
        memcpy(g_qa4, a, k * sizeof(float));
        memset(g_qa4 + k, 0, 3 * k * sizeof(float));
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_qa4, (size_t) k, g_aq);
        return;
    }
    if (k < 32 * g_participants) { spacemit_kernels::rvv::quantize_a_row_i8(32, a, (size_t) k, g_aq); return; }
    qa_src = a; qa_k = k;
    run(task_quant);
}

// ---- attention, position-split. Stages are separated by pool barriers so the
// per-head max/sum reductions are visible to every thread.
//
// RVV vector types cannot live in arrays on this toolchain, so the accumulators
// are spelled out one by one; the head-group sizes (8 for QK, 8 for PV) are
// chosen so the whole set stays in the 32 vector registers.
static int g_p0[N_GPU], g_p1[N_GPU];

// -W n caps attention to the last n positions. Nothing else changes: the causal
// mask already ends at g_pos, so dropping the old end is just a shorter range.
// At 10K the attention pass is compute bound at ~1 B/cycle/thread, so the cost
// is linear in the window and the prompt beyond it is simply not read.
static int g_win = 0;

// -MEGA_ATTN4=n: after prefill, time n four-row attention rounds over the whole
// prompt using the prefill int8 vmadot machinery. This is the measurement that
// decides whether four-row verification can pay at 10K, because decode's
// single-row attention is time-per-position rather than per-byte: four rows
// through it cost four times as much and cancel the weight amortisation.
static int g_attn4 = 0;

static void attn_ranges(void) {
    const int nb = g_pos + 1;
    const int lo = (g_win > 0 && nb > g_win) ? ((nb - g_win) / KT) * KT : 0;
    const int nt = (nb - lo + KT - 1) / KT;
    const int per = (nt + g_participants - 1) / g_participants;
    for (int t = 0; t < g_participants; ++t) {
        const int t0 = per * t, t1 = (t == g_participants - 1) ? nt : per * (t + 1);
        g_p0[t] = lo + ((t0 * KT > nb - lo) ? nb - lo : t0 * KT);
        g_p1[t] = lo + ((t1 * KT > nb - lo) ? nb - lo : t1 * KT);
    }
}

static double g_at[6];

// NEVER let GCC auto-vectorise these tiny loops: it emits masked vlse32 gathers
// at VLMAX (256 lanes) to reduce eight scalars, and strided vector access is
// trap-emulated on this core (~600 ns per instruction). That one mistake cost
// 29 ms/token, an 88x slowdown of the softmax pass.
__attribute__((optimize("no-tree-vectorize")))
static void compute_gmax(float * out) {
    if (g_dbg == 1) {
        for (int h = 0; h < N_HEAD; ++h) out[h] = g_lmax[g_pf_tid][h];
        return;
    }
    for (int h = 0; h < N_HEAD; ++h) {
        float m = -INFINITY;
        for (int t = 0; t < g_participants; ++t)
            if (g_p0[t] < g_p1[t]) m = fmaxf(m, g_lmax[t][h]);
        out[h] = m;
    }
}
__attribute__((optimize("no-tree-vectorize")))
static void compute_gsum(float * out) {
    for (int h = 0; h < N_HEAD; ++h) {
        double s = 0;
        for (int t = 0; t < g_participants; ++t)
            if (g_p0[t] < g_p1[t]) s += (double) g_lsum[t][h];
        out[h] = (float) s;
    }
}

static float g_am_val[N_GPU];
static int   g_am_idx[N_GPU];
static void task_argmax(int tid) {
    int64_t lo = (int64_t) VOCAB * tid / g_participants;
    int64_t hi = (int64_t) VOCAB * (tid + 1) / g_participants;
    if (tid == g_participants - 1) hi = VOCAB;
    if (lo >= hi) { g_am_val[tid] = -INFINITY; g_am_idx[tid] = 0; return; }
    int best = (int) lo;
    float bv = g_logits[lo];
    for (int64_t i = lo + 1; i < hi; ++i)
        if (g_logits[i] > bv) { bv = g_logits[i]; best = (int) i; }
    g_am_val[tid] = bv;
    g_am_idx[tid] = best;
}

// QK. Two things matter here on this core: vector loads have ~21 cycles of
// latency, and only 8 heads' worth of accumulator can be live. So the d loop is
// unrolled by 4 and all four K vectors are issued before any multiply, which
// hides the load latency behind 32 independent FMAs.
#define QK_ACC(H) \
    a##H = __riscv_vfmacc_vf_f32m1(a##H, qb[(H) * HEAD_DIM + d + 0], k0, vl); \
    a##H = __riscv_vfmacc_vf_f32m1(a##H, qb[(H) * HEAD_DIM + d + 1], k1, vl); \
    a##H = __riscv_vfmacc_vf_f32m1(a##H, qb[(H) * HEAD_DIM + d + 2], k2, vl); \
    a##H = __riscv_vfmacc_vf_f32m1(a##H, qb[(H) * HEAD_DIM + d + 3], k3, vl);

// 32 int8 lanes -> 32 f32 lanes. e8mf4 holds exactly the bytes a 32-lane f32m1
// result consumes, so the load needs no masking and the widening is in-register.
static inline vfloat32m1_t ld_i8_f32(const int8_t * p, size_t vl) {
    const vint8mf4_t b = __riscv_vle8_v_i8mf4(p, vl);
    return __riscv_vfcvt_f_x_v_f32m1(
        __riscv_vwcvt_x_x_v_i32m1(__riscv_vwcvt_x_x_v_i16mf2(b, vl), vl), vl);
}

static void attn_qk(int tid) {
    const int p0 = g_p0[tid], p1 = g_p1[tid];
    if (p0 >= p1) return;
    const double _w = now_s();
    float * sa = &g_sa[tid][0][0];
    const int len = p1 - p0;
    for (int kv = 0; kv < N_KV; ++kv) {
        const float * qb = g_q + (int64_t) kv * NH_PER_KV * HEAD_DIM;
        for (int pb = 0; pb < len; pb += KT) {
            const _Float16 * kb = ktile(g_layer, kv, (p0 + pb) / KT);
            const int8_t * kb8 = g_k8d + (int64_t) g_layer * K_LAYER_STRIDE +
                                 (((int64_t) kv * NKT + (p0 + pb) / KT) * HEAD_DIM) * KT;
            const size_t vl = __riscv_vsetvl_e32m1((size_t) (len - pb < KT ? len - pb : KT));
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int d = 0; d < HEAD_DIM; d += 4) {
                vfloat32m1_t k0, k1, k2, k3;
                if (g_k8dec) {
                    // int8 -> int16 -> int32 -> f32 is three conversions against the
                    // one widening a f16 load needs, but QK is memory bound at 10K
                    // (the whole dot loop is ~2 ms of the 40) so the trade is free.
                    k0 = ld_i8_f32(kb8 + (int64_t) (d + 0) * KT, vl);
                    k1 = ld_i8_f32(kb8 + (int64_t) (d + 1) * KT, vl);
                    k2 = ld_i8_f32(kb8 + (int64_t) (d + 2) * KT, vl);
                    k3 = ld_i8_f32(kb8 + (int64_t) (d + 3) * KT, vl);
                } else {
                    k0 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 0) * KT, vl), vl);
                    k1 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 1) * KT, vl), vl);
                    k2 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 2) * KT, vl), vl);
                    k3 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 3) * KT, vl), vl);
                }
                QK_ACC(0) QK_ACC(1) QK_ACC(2) QK_ACC(3)
                QK_ACC(4) QK_ACC(5) QK_ACC(6) QK_ACC(7)
            }
            const int64_t base = (int64_t) kv * NH_PER_KV * SMAX + pb;
            __riscv_vse32_v_f32m1(sa + base + 0 * SMAX, a0, vl);
            __riscv_vse32_v_f32m1(sa + base + 1 * SMAX, a1, vl);
            __riscv_vse32_v_f32m1(sa + base + 2 * SMAX, a2, vl);
            __riscv_vse32_v_f32m1(sa + base + 3 * SMAX, a3, vl);
            __riscv_vse32_v_f32m1(sa + base + 4 * SMAX, a4, vl);
            __riscv_vse32_v_f32m1(sa + base + 5 * SMAX, a5, vl);
            __riscv_vse32_v_f32m1(sa + base + 6 * SMAX, a6, vl);
            __riscv_vse32_v_f32m1(sa + base + 7 * SMAX, a7, vl);
        }
    }
    for (int h = 0; h < N_HEAD; ++h) {
        const float * row = sa + (int64_t) h * SMAX;
        vfloat32m1_t m = __riscv_vfmv_v_f_f32m1(-INFINITY, 32);
        int i = 0;
        for (; i + 32 <= len; i += 32)
            m = __riscv_vfmax_vv_f32m1(m, __riscv_vle32_v_f32m1(row + i, 32), 32);
        float mx = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m1_f32m1(m, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), 32));
        for (; i < len; ++i) mx = fmaxf(mx, row[i]);
        g_lmax[tid][h] = mx;
    }
    g_aw[0] += now_s() - _w;
}

// exp of (x - max) plus the running denominator. The polynomial is a serial
// dependency chain (~11 cycles per link), so two independent 32-lane chains are
// interleaved; without that this pass was latency bound at ~90 cycles per vector.
static inline vfloat32m1_t exp2f_poly(vfloat32m1_t t, size_t vl) {
    vint32m1_t ie = __riscv_vfcvt_x_f_v_i32m1(t, vl);
    const vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, __riscv_vfcvt_f_x_v_f32m1(ie, vl), vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t ex = __riscv_vadd_vx_i32m1(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m1(ex, 23, vl);
    return __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_i32m1_f32m1(ex), vl);
}

static inline vfloat32m2_t exp2f_poly_m2(vfloat32m2_t t, size_t vl) {
    vint32m2_t ie = __riscv_vfcvt_x_f_v_i32m2(t, vl);
    const vfloat32m2_t f = __riscv_vfsub_vv_f32m2(t, __riscv_vfcvt_f_x_v_f32m2(ie, vl), vl);
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    vint32m2_t ex = __riscv_vadd_vx_i32m2(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m2(ex, 23, vl);
    return __riscv_vfmul_vv_f32m2(p, __riscv_vreinterpret_v_i32m2_f32m2(ex), vl);
}

static inline vfloat32m4_t exp2f_poly_m4(vfloat32m4_t t, size_t vl) {
    vint32m4_t ie = __riscv_vfcvt_x_f_v_i32m4(t, vl);
    const vfloat32m4_t f = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(ie, vl), vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(0.00015403530f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(0.0013333558f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(0.0096181291f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(0.0555041087f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(0.2402265070f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(0.6931471806f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    vint32m4_t ex = __riscv_vadd_vx_i32m4(ie, 127, vl);
    ex = __riscv_vsll_vx_i32m4(ex, 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(ex), vl);
}

// ------------------------------------------------ exp2 without the polynomial
// The six-term polynomial above is twelve vector instructions on a chain six
// FMA deep: probe_exp3 puts it at 43 cycles per eight lanes on X100 and the
// chain, not the op count, is what costs. Two cheaper forms:
//
//   mode 1, three instructions. vfcvt.x.f of t*2^23 leaves floor(t) in the
//   exponent field and the fraction in the mantissa, so reinterpreting
//   (that + 127<<23) is 2^floor(t)*(1+f). Measured 6.15% maximum relative error,
//   always an overestimate, at 14 cycles per eight lanes.
//   mode 2, seven. Same, then multiply by a table of 2^f/(1+f) bucketed on the
//   top three mantissa bits: the residual is the variation of that ratio inside
//   a 1/8 bucket, a few tenths of a percent, at 21 cycles.
//
// The table has to fit one f32m1 register on the narrow cluster (VLMAX 8), so
// eight buckets, not thirty-two -- a 32-entry gather reads zeros there.
// Mode 1 is the default now. At 10K it cuts the prefill softmax from 153.9 to
// 119.0 core-s and the whole prefill from 75.5 to 74.0 s, and twelve greedy steps
// off a 100-token prompt are identical token for token, so the 6.15% exp error
// does not move the argmax here. Mode 2 buys nothing back for its extra gather.
static int g_exp3 = 1;
static float g_exp3_tab[32];
static void exp3_init(void) {
    for (int k = 0; k < 8; ++k) {
        const double f = (k + 0.5) / 8.0;
        g_exp3_tab[k] = (float) (pow(2.0, f) / (1.0 + f));
    }
    for (int k = 8; k < 32; ++k) g_exp3_tab[k] = 1.0f;
}

static inline vfloat32m1_t exp2f_e(vfloat32m1_t t, size_t vl) {
    if (!g_exp3) return exp2f_poly(t, vl);
    const vint32m1_t i = __riscv_vfcvt_x_f_v_i32m1(
        __riscv_vfmul_vf_f32m1(t, 8388608.0f, vl), vl);
    const vint32m1_t yb = __riscv_vadd_vx_i32m1(i, 127 << 23, vl);
    vfloat32m1_t y = __riscv_vreinterpret_v_i32m1_f32m1(yb);
    if (g_exp3 == 2) {
        const vuint32m1_t low = __riscv_vand_vx_u32m1(
            __riscv_vreinterpret_v_i32m1_u32m1(yb), 0x7FFFFF, vl);
        const vuint32m1_t idx = __riscv_vsrl_vx_u32m1(low, 20, vl);
        y = __riscv_vfmul_vv_f32m1(
            y, __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(g_exp3_tab, vl), idx, vl), vl);
    }
    return y;
}

static inline vfloat32m4_t exp2f_e_m4(vfloat32m4_t t, size_t vl) {
    if (!g_exp3) return exp2f_poly_m4(t, vl);
    const vint32m4_t i = __riscv_vfcvt_x_f_v_i32m4(
        __riscv_vfmul_vf_f32m4(t, 8388608.0f, vl), vl);
    const vint32m4_t yb = __riscv_vadd_vx_i32m4(i, 127 << 23, vl);
    vfloat32m4_t y = __riscv_vreinterpret_v_i32m4_f32m4(yb);
    if (g_exp3 == 2) {
        const vuint32m4_t low = __riscv_vand_vx_u32m4(
            __riscv_vreinterpret_v_i32m4_u32m4(yb), 0x7FFFFF, vl);
        const vuint32m4_t idx = __riscv_vsrl_vx_u32m4(low, 20, vl);
        y = __riscv_vfmul_vv_f32m4(
            y, __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(g_exp3_tab, vl), idx, vl), vl);
    }
    return y;
}

static void attn_exp_sum(int tid) {
    const int len = g_p1[tid] - g_p0[tid];
    if (len <= 0) return;
    const double _w = now_s();
    float gmax[N_HEAD];
    compute_gmax(gmax);
    float * sa = &g_sa[tid][0][0];
    for (int h = 0; h < N_HEAD; ++h) {
        float * row = sa + (int64_t) h * SMAX;
        const float mx = gmax[h];
        // int8 K leaves the scores unscaled, and a positive factor commutes with the
        // max subtraction: (s*ks - max*ks) = ks*(s - max). So the whole adjustment
        // is the log2(e) constant gaining the per-kv K scale.
        const float l2e = 1.4426950408889634f *
                          (g_k8dec ? g_k8dsc[g_layer][h / NH_PER_KV] : 1.0f);
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 32);
        int i = 0;
        for (; i + 64 <= len; i += 64) {
            const vfloat32m1_t t0 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), l2e, 32);
            const vfloat32m1_t t1 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i + 32, 32), mx, 32), l2e, 32);
            const vfloat32m1_t e0 = exp2f_e(__riscv_vfmax_vf_f32m1(t0, -126.0f, 32), 32);
            const vfloat32m1_t e1 = exp2f_e(__riscv_vfmax_vf_f32m1(t1, -126.0f, 32), 32);
            __riscv_vse32_v_f32m1(row + i, e0, 32);
            __riscv_vse32_v_f32m1(row + i + 32, e1, 32);
            vs = __riscv_vfadd_vv_f32m1(vs, __riscv_vfadd_vv_f32m1(e0, e1, 32), 32);
        }
        for (; i + 32 <= len; i += 32) {
            const vfloat32m1_t t0 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), l2e, 32);
            const vfloat32m1_t e0 = exp2f_e(__riscv_vfmax_vf_f32m1(t0, -126.0f, 32), 32);
            __riscv_vse32_v_f32m1(row + i, e0, 32);
            vs = __riscv_vfadd_vv_f32m1(vs, e0, 32);
        }
        float sum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(vs, __riscv_vfmv_v_f_f32m1(0.0f, 1), 32));
        for (; i < len; ++i) { const float e = expf(row[i] - mx); row[i] = e; sum += e; }
        g_lsum[tid][h] = sum;
    }
    g_aw[1] += now_s() - _w;
}

// PV. m1 (32 lanes) rather than m2: measured m2 FMA throughput is only 76% of m1
// per MAC on this core. Two dim chunks x 8 heads = 16 accumulators keeps enough
// independent chains to cover FMA latency, and both V chunks are loaded before
// the multiplies so the load latency overlaps.
static void attn_pv(int tid) {
    const int p0 = g_p0[tid], p1 = g_p1[tid];
    if (p0 >= p1) return;
    const double _w = now_s();
    const float * sa = &g_sa[tid][0][0];
    const int len = p1 - p0;
    const _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE;
    for (int kv = 0; kv < N_KV; ++kv)
        for (int half = 0; half < 2; ++half) {
            vfloat32m1_t a00 = __riscv_vfmv_v_f_f32m1(0.0f, 32), a01 = a00;
            vfloat32m1_t a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
            vfloat32m1_t a40 = a00, a41 = a00, a50 = a00, a51 = a00, a60 = a00, a61 = a00, a70 = a00, a71 = a00;
            const _Float16 * base = vb + (int64_t) kv * HEAD_DIM + half * 64;
            const int8_t * base8 = g_v8d + (int64_t) g_layer * KV_LAYER_STRIDE +
                                   (int64_t) kv * 2 * MAXCTX * 64 + (int64_t) half * MAXCTX * 64;
            const float * wk = sa + (int64_t) kv * NH_PER_KV * SMAX;
            for (int p = 0; p < len; ++p) {
                const _Float16 * vp = base + (int64_t) (p0 + p) * N_KV * HEAD_DIM;
                // int8 V is PV-order, so one position is 64 contiguous bytes and the
                // next position is the next 64: the whole pass is one sequential run.
                const int8_t * vp8 = base8 + (int64_t) (p0 + p) * 64;
                vfloat32m1_t v0, v1;
                if (g_k8dec) {
                    v0 = ld_i8_f32(vp8, 32);
                    v1 = ld_i8_f32(vp8 + 32, 32);
                } else {
                    v0 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp, 32), 32);
                    v1 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp + 32, 32), 32);
                }
                // MEGA_PVPROBE drops the eight scattered weight loads (one per
                // 10.5 KB score row, so eight pages per position) to separate the
                // V stream from the score reads. Results are garbage.
                float w0, w1, w2, w3, w4, w5, w6, w7;
                if (g_pvprobe) { w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 1e-4f; }
                else {
                    w0 = wk[0 * SMAX + p]; w1 = wk[1 * SMAX + p];
                    w2 = wk[2 * SMAX + p]; w3 = wk[3 * SMAX + p];
                    w4 = wk[4 * SMAX + p]; w5 = wk[5 * SMAX + p];
                    w6 = wk[6 * SMAX + p]; w7 = wk[7 * SMAX + p];
                }
                a00 = __riscv_vfmacc_vf_f32m1(a00, w0, v0, 32);
                a01 = __riscv_vfmacc_vf_f32m1(a01, w0, v1, 32);
                a10 = __riscv_vfmacc_vf_f32m1(a10, w1, v0, 32);
                a11 = __riscv_vfmacc_vf_f32m1(a11, w1, v1, 32);
                a20 = __riscv_vfmacc_vf_f32m1(a20, w2, v0, 32);
                a21 = __riscv_vfmacc_vf_f32m1(a21, w2, v1, 32);
                a30 = __riscv_vfmacc_vf_f32m1(a30, w3, v0, 32);
                a31 = __riscv_vfmacc_vf_f32m1(a31, w3, v1, 32);
                a40 = __riscv_vfmacc_vf_f32m1(a40, w4, v0, 32);
                a41 = __riscv_vfmacc_vf_f32m1(a41, w4, v1, 32);
                a50 = __riscv_vfmacc_vf_f32m1(a50, w5, v0, 32);
                a51 = __riscv_vfmacc_vf_f32m1(a51, w5, v1, 32);
                a60 = __riscv_vfmacc_vf_f32m1(a60, w6, v0, 32);
                a61 = __riscv_vfmacc_vf_f32m1(a61, w6, v1, 32);
                a70 = __riscv_vfmacc_vf_f32m1(a70, w7, v0, 32);
                a71 = __riscv_vfmacc_vf_f32m1(a71, w7, v1, 32);
            }
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 0] + 0 + half * 64, a00, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 0] + 32 + half * 64, a01, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 1] + 0 + half * 64, a10, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 1] + 32 + half * 64, a11, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 2] + 0 + half * 64, a20, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 2] + 32 + half * 64, a21, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 3] + 0 + half * 64, a30, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 3] + 32 + half * 64, a31, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 4] + 0 + half * 64, a40, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 4] + 32 + half * 64, a41, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 5] + 0 + half * 64, a50, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 5] + 32 + half * 64, a51, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 6] + 0 + half * 64, a60, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 6] + 32 + half * 64, a61, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 7] + 0 + half * 64, a70, 32);
            __riscv_vse32_v_f32m1(g_acc[tid][(int64_t) kv * NH_PER_KV + 7] + 32 + half * 64, a71, 32);
        }
    g_aw[2] += now_s() - _w;
}

// PV, one pass over V per kv head, vectorised over the head dimension at m4.
//
// The m1 version above walks the position range once per (kv, half) pair, so it
// touches four passes over the same KV plane and each pass reads a 128-byte
// quarter of every 512-byte position stride. It measures 23.9 ms at 10K for
// 215 MB of V, i.e. 9.0 GB/s, against 17.7 GB/s for a plain sequential stream.
// Eight m4 accumulators hold all 128 dims of a kv head, so one 128-byte load and
// eight FMAs per position cover what the old loop needed four passes to do.
//
// Measured: it is 2.4x slower (4K pv 9.84 -> 23.29 ms). Wide-LMUL FMA on this
// core costs far more per MAC than m1 -- the same effect that made m2 the wrong
// choice above, an order of magnitude worse at m4 -- so the 4-pass m1 loop stays
// despite its stride. Kept behind -P4 as the record of that.
static int g_pvm4 = 0;

static inline vfloat32m4_t ld_i8_f32_m4(const int8_t * p, size_t vl) {
    const vint8m1_t b = __riscv_vle8_v_i8m1(p, vl);
    return __riscv_vfcvt_f_x_v_f32m4(
        __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(b, vl), vl), vl);
}
static inline vfloat32m4_t ld_f16_f32_m4(const _Float16 * p, size_t vl) {
    return __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p, vl), vl);
}
static inline vfloat32m2_t ld_i8_f32_m2(const int8_t * p, size_t vl) {
    const vint8mf2_t b = __riscv_vle8_v_i8mf2(p, vl);
    return __riscv_vfcvt_f_x_v_f32m2(
        __riscv_vwcvt_x_x_v_i32m2(__riscv_vwcvt_x_x_v_i16m1(b, vl), vl), vl);
}

// PV at m2: one 64-byte load and eight 64-lane FMAs per position covers the whole
// half, against two loads and sixteen 32-lane FMAs at m1. Same MAC count, half the
// instructions, which is what matters once the PV-order V plane has removed the
// stride: at m1 the loop measures 8.1 GB/s for 8 MAC per byte, so it is issuing
// instructions, not waiting on memory. m2 cost 24% more per MAC in the old
// dimension-chunk loop, and this is the case where that trade wins.
static int g_pvm2 = 0;

static void attn_pv_m2(int tid) {
    const int p0 = g_p0[tid], p1 = g_p1[tid];
    if (p0 >= p1) return;
    const double _w = now_s();
    const int len = p1 - p0;
    const float * sa = &g_sa[tid][0][0];
    const _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE;
    const size_t vl = __riscv_vsetvl_e32m2(64);
    for (int kv = 0; kv < N_KV; ++kv)
        for (int half = 0; half < 2; ++half) {
            vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            const _Float16 * base = vb + (int64_t) kv * HEAD_DIM + half * 64;
            const int8_t * base8 = g_v8d + (int64_t) g_layer * KV_LAYER_STRIDE +
                                   (int64_t) kv * 2 * MAXCTX * 64 + (int64_t) half * MAXCTX * 64;
            const float * wk = sa + (int64_t) kv * NH_PER_KV * SMAX;
            for (int p = 0; p < len; ++p) {
                const vfloat32m2_t v = g_k8dec
                    ? ld_i8_f32_m2(base8 + (int64_t) (p0 + p) * 64, vl)
                    : __riscv_vfwcvt_f_f_v_f32m2(
                          __riscv_vle16_v_f16m1(base + (int64_t) (p0 + p) * N_KV * HEAD_DIM, vl), vl);
                a0 = __riscv_vfmacc_vf_f32m2(a0, wk[0 * SMAX + p], v, vl);
                a1 = __riscv_vfmacc_vf_f32m2(a1, wk[1 * SMAX + p], v, vl);
                a2 = __riscv_vfmacc_vf_f32m2(a2, wk[2 * SMAX + p], v, vl);
                a3 = __riscv_vfmacc_vf_f32m2(a3, wk[3 * SMAX + p], v, vl);
                a4 = __riscv_vfmacc_vf_f32m2(a4, wk[4 * SMAX + p], v, vl);
                a5 = __riscv_vfmacc_vf_f32m2(a5, wk[5 * SMAX + p], v, vl);
                a6 = __riscv_vfmacc_vf_f32m2(a6, wk[6 * SMAX + p], v, vl);
                a7 = __riscv_vfmacc_vf_f32m2(a7, wk[7 * SMAX + p], v, vl);
            }
            const int64_t off = half * 64;
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 0] + off, a0, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 1] + off, a1, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 2] + off, a2, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 3] + off, a3, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 4] + off, a4, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 5] + off, a5, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 6] + off, a6, vl);
            __riscv_vse32_v_f32m2(g_acc[tid][kv * NH_PER_KV + 7] + off, a7, vl);
        }
    g_aw[2] += now_s() - _w;
}

static void attn_pv_m4(int tid) {
    const int p0 = g_p0[tid], p1 = g_p1[tid];
    if (p0 >= p1) return;
    const double _w = now_s();
    const int len = p1 - p0;
    const float * sa = &g_sa[tid][0][0];
    const _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE;
    const int8_t * vb8 = g_v8d + (int64_t) g_layer * KV_LAYER_STRIDE;
    const size_t vl = __riscv_vsetvl_e32m4(HEAD_DIM);
    for (int kv = 0; kv < N_KV; ++kv) {
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), a1 = a0, a2 = a0, a3 = a0;
        vfloat32m4_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const _Float16 * base = vb + (int64_t) kv * HEAD_DIM;
        const int8_t * base8 = vb8 + (int64_t) kv * HEAD_DIM;
        const float * wk = sa + (int64_t) kv * NH_PER_KV * SMAX;
        for (int p = 0; p < len; ++p) {
            const int64_t off = (int64_t) (p0 + p) * N_KV * HEAD_DIM;
            // f16 only: this variant needs one 128-element load per kv head, which
            // the f16 plane has and the PV-order int8 plane does not.
            const vfloat32m4_t v = ld_f16_f32_m4(base + off, vl);
            a0 = __riscv_vfmacc_vf_f32m4(a0, wk[0 * SMAX + p], v, vl);
            a1 = __riscv_vfmacc_vf_f32m4(a1, wk[1 * SMAX + p], v, vl);
            a2 = __riscv_vfmacc_vf_f32m4(a2, wk[2 * SMAX + p], v, vl);
            a3 = __riscv_vfmacc_vf_f32m4(a3, wk[3 * SMAX + p], v, vl);
            a4 = __riscv_vfmacc_vf_f32m4(a4, wk[4 * SMAX + p], v, vl);
            a5 = __riscv_vfmacc_vf_f32m4(a5, wk[5 * SMAX + p], v, vl);
            a6 = __riscv_vfmacc_vf_f32m4(a6, wk[6 * SMAX + p], v, vl);
            a7 = __riscv_vfmacc_vf_f32m4(a7, wk[7 * SMAX + p], v, vl);
        }
        float * gp = g_acc[tid][kv * NH_PER_KV];
        __riscv_vse32_v_f32m4(gp + 0 * HEAD_DIM, a0, vl);
        __riscv_vse32_v_f32m4(gp + 1 * HEAD_DIM, a1, vl);
        __riscv_vse32_v_f32m4(gp + 2 * HEAD_DIM, a2, vl);
        __riscv_vse32_v_f32m4(gp + 3 * HEAD_DIM, a3, vl);
        __riscv_vse32_v_f32m4(gp + 4 * HEAD_DIM, a4, vl);
        __riscv_vse32_v_f32m4(gp + 5 * HEAD_DIM, a5, vl);
        __riscv_vse32_v_f32m4(gp + 6 * HEAD_DIM, a6, vl);
        __riscv_vse32_v_f32m4(gp + 7 * HEAD_DIM, a7, vl);
    }
    g_aw[2] += now_s() - _w;
}

// ---- rope. Two changes make this cheap:
//  * cos/sin come from a table built once for all positions, so no libm in the loop.
//  * q_proj/k_proj rows are repacked in rotation-pair order (even dims, then odd),
//    so the rotation is plain contiguous vector math instead of a strided gather.
//    A dot product is permutation invariant, and q and k share the order, so
//    the attention scores are unchanged.
static double g_t_rope_math, g_t_rope_app;
// LLAMA_ROPE_TYPE_NORM: pairs (2i, 2i+1), not rotate_half's (i, i+64). This is
// what llama.cpp uses for general.architecture = llama, and getting it wrong was
// invisible from inside the kernel: the numpy reference was written with the same
// rotate_half convention, so the two agreed to 0.9979 cosine with a matching
// argmax while both sat 0.87 away from llama.cpp. Changing only the reference to
// NORM moved it to 0.9776 against llama.cpp, which is the fix applied here.
//
// The deinterleave has to happen in registers. This core trap-emulates strided
// vector access, so an even/odd gather is built from the lane id (vid ^ 1 swaps
// each pair, vid >> 1 duplicates each angle) rather than with vlse32.
//
//   out[2i]   = x[2i]  *c_i - x[2i+1]*s_i
//   out[2i+1] = x[2i+1]*c_i + x[2i]  *s_i
//   => out = x*C + swap(x)*S,  C = [c0,c0,c1,c1,...], S = [-s0,+s0,-s1,+s1,...]
//
// One head, one copy, three callers (prefill, single-row decode, four-row decode).
// They used to be three separate implementations and they drifted: prefill was
// still rotate_half after decode was changed, and the mixed cache produced garbage
// logits and a segfault rather than a wrong number.
__attribute__((optimize("no-tree-vectorize")))
static inline void rope_head_norm(float * x, const float * cs, float post) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    const vuint32m1_t vid = __riscv_vid_v_u32m1(vl);
    const vuint32m1_t isw = __riscv_vxor_vx_u32m1(vid, 1, vl);   // [1,0,3,2,...]
    const vuint32m1_t idp = __riscv_vsrl_vx_u32m1(vid, 1, vl);   // [0,0,1,1,...]
    static const float sgn_tab[64] = {
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1 };
    const vfloat32m1_t vsg = __riscv_vle32_v_f32m1(sgn_tab, vl);
    const vfloat32m1_t vpost = __riscv_vfmv_v_f_f32m1(post, vl);
    for (int off = 0; off < HEAD_DIM; off += (int) vl) {
        const int j = off / 2;
        const vfloat32m1_t a = __riscv_vle32_v_f32m1(x + off, vl);
        const vfloat32m1_t as = __riscv_vrgather_vv_f32m1(a, isw, vl);
        const vfloat32m1_t cv = __riscv_vrgather_vv_f32m1(
            __riscv_vle32_v_f32m1(cs + j, vl), idp, vl);
        const vfloat32m1_t sv = __riscv_vfmul_vv_f32m1(
            __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(cs + 64 + j, vl), idp, vl), vsg, vl);
        vfloat32m1_t r = __riscv_vfmacc_vv_f32m1(__riscv_vfmul_vv_f32m1(a, cv, vl), as, sv, vl);
        if (post != 1.0f) r = __riscv_vfmul_vv_f32m1(r, vpost, vl);
        __riscv_vse32_v_f32m1(x + off, r, vl);
    }
}

static void rope_all(void) {
    const double _ta = now_s();
    const float * cs = g_rope_tab + (int64_t) g_pos * 128;
    const float inv = 1.0f / sqrtf((float) HEAD_DIM);
    for (int h = 0; h < N_HEAD; ++h) rope_head_norm(g_q + h * HEAD_DIM, cs, inv);
    for (int h = 0; h < N_KV; ++h) rope_head_norm(g_k + h * HEAD_DIM, cs, 1.0f);
    // q carries 1/sqrt(head_dim) inside rope_head_norm's `post`, so the scores need
    // no second factor -- the same convention the prefill path uses.
    g_t_rope_math += now_s() - _ta;

    const double _tb = now_s();
    const int pblk = g_pos / KT, pi = g_pos % KT;
    _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE + (int64_t) g_pos * N_KV * HEAD_DIM;
    int8_t * k8base = g_k8d + (int64_t) g_layer * K_LAYER_STRIDE;
    for (int kv = 0; kv < N_KV; ++kv) {
        _Float16 * kb = ktile(g_layer, kv, pblk);
        int8_t * kb8 = k8base + (((int64_t) kv * NKT + pblk) * HEAD_DIM) * KT;
        if (g_k8dec) {
            // Same scale as every already-cached position, so the new token is just
            // another sample of one fixed grid. A value past the grid clips at 127
            // rather than triggering a rescale, which would have to rewrite the
            // whole history and would move every earlier token's score.
            const float ik = g_k8dsc[g_layer][kv] > 0 ? 1.0f / g_k8dsc[g_layer][kv] : 1.0f;
            const float iv = g_v8dsc[g_layer][kv] > 0 ? 1.0f / g_v8dsc[g_layer][kv] : 1.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                int q = (int) lrintf(g_k[kv * HEAD_DIM + d] * ik);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                kb8[(int64_t) d * KT + pi] = (int8_t) q;
            }
            for (int d = 0; d < HEAD_DIM; d += 32) {
                const size_t vl = __riscv_vsetvl_e32m1(32);
                const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                    __riscv_vle32_v_f32m1(g_v + kv * HEAD_DIM + d, 32), iv, vl);
                vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(f, vl);
                q = __riscv_vmax_vx_i32m1(q, -127, vl);
                q = __riscv_vmin_vx_i32m1(q, 127, vl);
                // PV order: [kv][half][pos][64], so this writes the d-th block into
                // the half it belongs to rather than at kv*HEAD_DIM + d of a pos-major row.
                int8_t * vq = g_v8d + (int64_t) g_layer * KV_LAYER_STRIDE +
                              ((int64_t) kv * 2 * MAXCTX + (int64_t) g_pos) * 64 + d % 64;
                __riscv_vse8_v_i8mf4(vq,
                                     __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, vl), vl), vl);
            }
        }
        for (int d = 0; d < HEAD_DIM; ++d) kb[(int64_t) d * KT + pi] = (_Float16) g_k[kv * HEAD_DIM + d];
        for (int d = 0; d < HEAD_DIM; d += 32)
            __riscv_vse16_v_f16mf2(vb + kv * HEAD_DIM + d,
                                   __riscv_vfncvt_f_f_w_f16mf2(__riscv_vle32_v_f32m1(g_v + kv * HEAD_DIM + d, 32), 32), 32);
    }
    g_t_rope_app += now_s() - _tb;
}

// Combine the per-thread position partials. Heads are split across the pool; the
// old version had every thread redo all 16 heads, i.e. 8x redundant work.
static void attn_combine(int tid) {
    const double _w = now_s();
    float gsum[N_HEAD];
    compute_gsum(gsum);
    for (int hh = tid * 2; hh < tid * 2 + 2 && hh < N_HEAD; ++hh) {
        // The int8 V path never scales its operands, so the per-head V scale rides
        // along with the softmax denominator here: one multiply per output vector
        // instead of sixteen per position block inside PV.
        const float inv = (1.0f / gsum[hh]) *
                          (g_k8dec ? g_v8dsc[g_layer][hh / NH_PER_KV] : 1.0f);
        for (int d = 0; d < HEAD_DIM; d += 32) {
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 32);
            for (int t = 0; t < g_participants; ++t)
                if (g_p0[t] < g_p1[t]) acc = __riscv_vfadd_vv_f32m1(acc, __riscv_vle32_v_f32m1(g_acc[t][hh] + d, 32), 32);
            __riscv_vse32_v_f32m1(g_attn_out + hh * HEAD_DIM + d, __riscv_vfmul_vf_f32m1(acc, inv, 32), 32);
        }
    }
    g_aw[3] += now_s() - _w;
}

static void attention(void) {
    double _a = now_s();
    attn_ranges();
    g_at[0] += now_s() - _a; _a = now_s();
    run(attn_qk);
    g_at[1] += now_s() - _a; _a = now_s();
    run(attn_exp_sum);
    g_at[3] += now_s() - _a; _a = now_s();
    run(g_pvm4 ? attn_pv_m4 : g_pvm2 ? attn_pv_m2 : attn_pv);
    g_at[4] += now_s() - _a; _a = now_s();
    run(attn_combine);
    g_at[5] += now_s() - _a;
}

static void silu_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i));
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(g_gate + i, vl);
        const vfloat32m1_t u = __riscv_vle32_v_f32m1(g_up + i, vl);
        // x/(1+exp(-x)); clamp the exponent input, never x itself
        vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, -1.4426950408889634f, vl);
        t = __riscv_vfmax_vf_f32m1(t, -126.0f, vl);
        t = __riscv_vfmin_vf_f32m1(t,  126.0f, vl);
        const vfloat32m1_t den = __riscv_vfadd_vf_f32m1(exp2f_e(t, vl), 1.0f, vl);
        __riscv_vse32_v_f32m1(g_gate + i, __riscv_vfmul_vv_f32m1(__riscv_vfdiv_vv_f32m1(x, den, vl), u, vl), vl);
    }
}

static void residual_add_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i));
        const vfloat32m1_t a = __riscv_vle32_v_f32m1(g_x + i, vl);
        const vfloat32m1_t b = __riscv_vle32_v_f32m1(g_xa + i, vl);
        __riscv_vse32_v_f32m1(g_x + i, __riscv_vfadd_vv_f32m1(a, b, vl), vl);
    }
}

// ---------------------------------------------------------- prefill attention
// Decode attention walks the KV cache once per token, which is fine for one
// query row and hopeless for ten thousand: at pp10240 the same 220 MB of K and V
// would be re-read per row. Here a group of PFA_ROWS queries shares every K and
// V load: K is loaded once per 32-position tile and used for all rows, and V is
// loaded once per position and feeds all rows' accumulators. PFA_ROWS=4 keeps
// 4 rows x 4 dim chunks = 16 output accumulators live, which fits the register
// file; going wider would force a second V pass and give back the saving.
//
// Masking is exact: row r of the group queries absolute position ctx+qbase+r and
// zeroes every score above that.
static const int PFA_ROWS = 4;
// The int8 path tiles PFA_ROWS*I8_G = 16 queries per head, so four row groups
// consume the same K and V blocks. One K block is 8 KB and stays in L1 across
// the tile; at 4-row tiles every group streamed its own copy from DRAM, which
// left QK bandwidth starved at ~0.4 GB/s of useful traffic.
//
// I8_G is overrideable at compile time (-DI8G=32) because the tile height sets how
// many times a 512-row prefill chunk walks the K plane: 64 rows is 8 passes, 128
// rows is 4. Which of the two is faster is a measurement, and the rows in a tile
// only decide co-residency -- each row's own dot products and their order are
// unchanged, so the scores are bit-identical either way.
#ifndef I8G
#define I8G 16
#endif
static const int I8_G = I8G;
static const int I8_ROWS = I8_G * PFA_ROWS;
// Number of four-row groups a tile actually computes. Prefill batches sixteen of
// them so a 64-row chunk walks each K block once; a decode step has four rows and
// wants one, because every stage -- QK, softmax, PV -- scales with the group count
// and a single-row tile is read-limited anyway (220 MB of int8 K+V is the floor).
// Buffers and strides stay sized for I8_G groups and only the loop bound moves.
static int g_i8g = I8_G;
static const int WBLK = (MAXCTX / 16) * 64;    // int8 attention weights per row group
static int64_t g_pfa_rows, g_pfa_ctx;
static float * g_pq;                        // [rows][N_HEAD][HEAD_DIM]
static float * g_pfa_out;                   // [rows][N_HEAD][HEAD_DIM], refilled per layer
// The int8 PV wrote its result only when layer == 0, so every layer past the
// first ran the out-projection on layer 0's attention output; layer 1 came out
// 0.49 relative away from the f32 reference while layer 0 was 0.03.
static float g_ppv_tmp[N_GPU][4 * N_HEAD * HEAD_DIM];
static float g_psa[N_GPU][PFA_ROWS][MAXCTX];    // per-thread scores (f16 path)
// The int8 path keeps its raw scores in int16 fixed point at 1/128. A row of
// scores is 40 KB in f32 and is written once and read twice, which made the
// score buffer - not the dots - the prefill attention bottleneck: 13.4 GB per
// layer of writes plus 26.8 GB of reads, i.e. DRAM bound at ~11 GB/s. int16
// halves that, and the max pass can run in the integer domain.
static const float PSA_SC = 128.0f;
// Scores are stored in 64-byte blocks: four consecutive queries x eight
// positions, in the lane order vmadot produces (query r, position c -> lane
// r*8+c). One block is exactly one store, so the QK epilogue needs no slidedown,
// and softmax can still walk positions in order because block i holds positions
// 8i..8i+7 of all four rows of a group. probe_qk_epi.cpp on one A core: this
// layout runs QK at 93.8 GMAC/s against a 101.4 vmadot ceiling, where the old
// per-row shape manages 48.6 -- the four 8-lane slidedown+store pairs ate 53% of
// the inner loop.
static const int SC_NBLK = MAXCTX / 8;             // blocks per row group
static const int SC_GSTRIDE = SC_NBLK * 32;        // int16 per row group
static const int SC_TSTRIDE = I8_G * SC_GSTRIDE;   // int16 per tile
static const int SC_SHIFT = 6;                     // raw >> 6 always fits int16
static int16_t * g_psa16;
// Each thread's timers get a cache line of their own. These are incremented once
// per attention row -- 6.7 M rows at 10K -- and with eight cores sharing one line
// the increment alone measured 160 core-s of the softmax slice, i.e. 24 us per row
// for a 5000-element row. Stride 8 doubles == one 64-byte line.
static const int PFA_TSTRIDE = 8;
static double g_pfa_qk[N_GPU * PFA_TSTRIDE], g_pfa_sm[N_GPU * PFA_TSTRIDE],
              g_pfa_pv[N_GPU * PFA_TSTRIDE];
static double g_pfa_mx[N_GPU * PFA_TSTRIDE], g_pfa_ex[N_GPU * PFA_TSTRIDE];
static double g_pfa_smb[8 * PFA_TSTRIDE];
static double g_pfa_mxb[8 * PFA_TSTRIDE], g_pfa_exb[8 * PFA_TSTRIDE];
static thread_local int g_pfa_smid;
static thread_local int g_pfa_tid;
static float * g_kf;                        // [kv][pblk][dim][32] f32, one layer
static int64_t g_kf_npb;
static inline float * kftile(int kv, int pblk) {
    return g_kf + ((int64_t) kv * g_kf_npb + pblk) * HEAD_DIM * KT;
}
static float * g_vf;                        // [kv][pos][dim] f32, one layer
static int g_pfa_f32 = 0;
// Per-thread, and written only when MEGA_PFADBG is set: these were global and
// stored on every attention row, which is the same false-sharing trap as the
// timers -- eight cores writing one 64-byte line 6.7 M times.
static const bool g_pfa_dbg_on = getenv("MEGA_PFADBG") != nullptr;
static float g_pfa_dbg_sum[N_GPU * 8], g_pfa_dbg_mx[N_GPU * 8];

__attribute__((optimize("no-tree-vectorize")))
static void pfa_softmax_row(float * row, int len, double * t_sm) {
    // Measured: a block-buffered single-pass version (copy the row into a 4 KB
    // scratch, max and exp and normalise out of it) is SLOWER at 10K, because the
    // row is already hot when the three passes run back to back and the copy adds a
    // transfer of its own. Three passes over a hot 40 KB row beat one pass plus a
    // copy. What the passes cost is the exp itself, and that is what -e changes:
    // 152 -> 118 core-s for the three-instruction form, worth 1.5 s of prefill.
    const double _w = now_s();
    vfloat32m1_t m = __riscv_vfmv_v_f_f32m1(-INFINITY, 32);
    int i = 0;
    for (; i + 32 <= len; i += 32)
        m = __riscv_vfmax_vv_f32m1(m, __riscv_vle32_v_f32m1(row + i, 32), 32);
    float mx = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m1_f32m1(m, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), 32));
    for (; i < len; ++i) mx = fmaxf(mx, row[i]);

    vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 32);
    i = 0;
    for (; i + 32 <= len; i += 32) {
        const vfloat32m1_t t = __riscv_vfmul_vf_f32m1(
            __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), 1.4426950408889634f, 32);
        const vfloat32m1_t e = exp2f_e(__riscv_vfmax_vf_f32m1(t, -126.0f, 32), 32);
        __riscv_vse32_v_f32m1(row + i, e, 32);
        vs = __riscv_vfadd_vv_f32m1(vs, e, 32);
    }
    float sum = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(vs, __riscv_vfmv_v_f_f32m1(0.0f, 1), 32));
    for (; i < len; ++i) { const float e = expf(row[i] - mx); row[i] = e; sum += e; }
    const float inv = sum > 0 ? 1.0f / sum : 0.0f;
    for (i = 0; i + 32 <= len; i += 32)
        __riscv_vse32_v_f32m1(row + i, __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), inv, 32), 32);
    for (; i < len; ++i) row[i] *= inv;
    if (g_pfa_dbg_on) {
        g_pfa_dbg_sum[g_pfa_tid * 8] = sum;
        g_pfa_dbg_mx[g_pfa_tid * 8] = mx;
    }
    *t_sm += now_s() - _w;
}

// The f16 -> f32 widen (vfwcvt) is an emulated instruction on this core, ~86
// cycles. Paying it inside QK means every K element is widened once per row
// group, i.e. 2560 times at 10K. Converting a layer's K and V into a 21 MB f32
// scratch once costs ~0.02 s and removes the widen from the hot loop entirely.
static int g_w8off = 0;
static int g_vsmall = 0;
static int g_hoist = 0;
static int g_keepv = 0, g_keepw = 0;
static int g_vsrc = 0;
static int g_gstyle = 0;
static int g_tcmstage = 0;
static int g_pvchunk = 0;
// Both of these are on by default: they are the only two measured wins in the
// whole PV investigation, they cost nothing, and neither changes a reduction
// order (tile order and load form only), so worst |diff| stays 4.554e-03.
//   head-major tiles: a thread stays inside one head's KV instead of cycling
//   all sixteen, so eight threads no longer hold every head's V live (21 MB).
//   attention 31.70 -> 29.20 s, PV 114.8 -> 93.6 core-s.
//   plain vle32.v: same data as vl1re32.v, 4-byte instead of 128-byte alignment
//   requirement; PV 114.8 -> 110.9 core-s on its own.
static int g_headmajor = 1;
static int g_le32 = 1;
static int g_tilefast = 0;   // force the old head-fast order back
static int g_cv_layer;
static int64_t g_cv_npos;
static void cv_fn(int64_t lo, int64_t hi) {
    const int layer = g_cv_layer;
    const int64_t npos = g_cv_npos;
    for (int64_t i = lo; i < hi; ++i) {
        const int64_t p   = i % npos;
        const int64_t d   = (i / npos) % HEAD_DIM;
        const int     kv  = (int) (i / (npos * HEAD_DIM));
        const _Float16 * kb = ktile(layer, kv, (int) (p / KT));
        // Same tile shape as the f16 cache: a dim-major [dim][pos] layout would
        // put each of the 128 dims on its own page and pay a TLB miss per load.
        g_kf[(int64_t) (kv * g_kf_npb + p / KT) * HEAD_DIM * KT + d * KT + (p % KT)] =
            (float) kb[d * KT + (p % KT)];
        // V is consumed 32 dims at a time, so it stays position-major
        g_vf[(int64_t) (kv * npos + p) * HEAD_DIM + d] =
            (float) g_vpos[(int64_t) layer * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM + kv * HEAD_DIM + d];
    }
}
static void pfa_convert_kv(int layer, int64_t npos) {
    // [kv][dim][pos] so one dim's positions are contiguous: unit-stride vector load
    g_cv_layer = layer; g_cv_npos = npos;
    g_kf_npb = (npos + KT - 1) / KT;
    parallel_for((int64_t) N_KV * HEAD_DIM * npos, cv_fn);
}

// ------------------------------------------------------- int8 dot attention
// The float path above is bound by vfmacc issue rate: 4.4 GMAC/s per core, which
// put 10K prefill attention at 256 s. The same dot units that carry the GEMM can
// carry attention instead, because `vmadot` is symmetric -- the operand that
// supplies 8 "columns" can just as well supply 8 positions (for QK) or 8 dims
// (for PV). Layouts below are the ones verified bit-exact by
// probes/resolve_int8_dot_map.cpp and prototyped in probes/proto_int8_dot_attention.cpp.
//
//   QK: A = 4 queries x 16 dims, B = 64 positions x 16 dims,
//       out[j] lane r*8+c = query r, position j*8+c.
//   PV: A = 4 queries x 16 positions, B = 128 dims x 16 positions,
//       out[j] lane r*8+c = query r, dim j*8+c; contracted 16 positions per step.
//
// K is repacked per (kv, 64-position block) as 8 groups of 64 positions x 16 dims
// so one `vl8re32` feeds all eight accumulators. V is repacked per (kv, 16-position
// block) as 16 registers of 8 dims x 16 positions.
static int g_pfa_layer;
static int8_t * g_k8i, * g_v8i;
// Prefill converts layer L, attends over it, then converts layer L+1 into the same
// planes: one layer's worth of int8 KV is scratch that gets overwritten. A decode
// round needs every layer's plane to survive, so when -MEGA_ATTN4 keeps the run
// going into four-row rounds these per-layer copies are what attention reads.
static int8_t * g_k8i_l = nullptr, * g_v8i_l = nullptr;
static int64_t g_k8i_lsz = 0, g_v8i_lsz = 0;
static inline int8_t * k8i_base(int layer) {
    return g_k8i_l ? g_k8i_l + (int64_t) layer * g_k8i_lsz : g_k8i;
}
static inline int8_t * v8i_base(int layer) {
    return g_v8i_l ? g_v8i_l + (int64_t) layer * g_v8i_lsz : g_v8i;
}
static int8_t * g_q8i;                 // [(head*nqg + group)][8 ksteps][4 rows x 16]
static float *  g_q8isc;               // [rows][N_HEAD]
static float g_k8isc[4], g_v8isc[4];
static int64_t g_k8i_npb, g_v8i_npb16, g_q8i_nqg;
// The int8 KV layout interleaves (head, 64-position block), so the block stride
// depends on how many blocks exist. Deriving it from the current position count
// moved every already-converted block whenever the prefix grew, which is why the
// conversion used to rewrite the whole prefix each chunk. Fixing the stride at the
// buffer capacity keeps old blocks in place, so a chunk only converts its own rows.
static int64_t g_k8i_npb_cap, g_v8i_npb16_cap;
static int g_pfa_i8 = 0;
static int64_t g_pq_rows = 0;
static int g_pfa_dbg1 = 0;
static int g_nopv = 0, g_qk_trace = 0;
static __thread int16_t g_nspark[8 * 32];
static int g_nostore = 0;   // diagnostic: park score stores in one L1 block
static int g_gatemode = 0;  // correctness gate: emit only the generated token ids
static int g_top5 = 0;      // with -g, also emit the top-5 logits per step
static int g_dbg_h0 = 0, g_dbg_h0b = 0;

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

// A advances by 64 and B by 1024 per 16-position step: only the low 64 bytes of
// Diagnostic twin of pv4 with both loads hoisted out of the loop. Same arithmetic,
// deliberately wrong output; it exists to tell load issue cost apart from vmadot
// latency, which decides whether making V reuse across row groups is worth it.
static void pv4_hoist(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
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
        "vl1re32.v v0, (%[W])           \n\t"
        "vl8re32.v v8, (%[V])           \n\t"
        "mv     t3, %[NP]               \n\t"
        "1:                             \n\t"
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
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// Diagnostic twins that keep exactly one of the two loads in the loop, to find
// out which operand's fetch is the expensive one. Same arithmetic, wrong output.
static void pv4_keepv(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
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
        "vl1re32.v v0, (%[W])           \n\t"
        "mv     t3, %[NP]               \n\t"
        "mv     t5, %[V]                \n\t"
        "1:                             \n\t"
        "vl8re32.v v8, (t5)             \n\t"
        "addi    t5, t5, 1024           \n\t"
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
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

static void pv4_keepw(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
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
        "vl8re32.v v8, (%[V])           \n\t"
        "mv     t3, %[NP]               \n\t"
        "mv     t4, %[W]                \n\t"
        "1:                             \n\t"
        "vl1re32.v v0, (t4)             \n\t"
        "addi    t4, t4, 64             \n\t"
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
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t4",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// pv4 rewritten as a structural copy of gemm_i8d_4x64 -- chained "+r" pointers,
// per-instantiation label, constant addi strides -- because that kernel runs the
// same two-loads-plus-eight-dots body at 9 cycles per dot while pv4 sits at 168
// and the data source has been ruled out (weight blob: 106 vs 113 core-s).
static void pv4_gemmstyle(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
    int8_t * const * out = acc;
    asm volatile(
        "vsetvli t0, x0, e32, m1   \n\t"
        "vxor.vv v16, v16, v16     \n\t"
        "vxor.vv v18, v18, v18     \n\t"
        "vxor.vv v20, v20, v20     \n\t"
        "vxor.vv v22, v22, v22     \n\t"
        "vxor.vv v24, v24, v24     \n\t"
        "vxor.vv v26, v26, v26     \n\t"
        "vxor.vv v28, v28, v28     \n\t"
        "vxor.vv v30, v30, v30     \n\t"
        "mv     t3, %[NP]          \n\t"
        "I8P%=:                    \n\t"
        "vl1re32.v v0, (%[W])      \n\t"
        "vl8re32.v v8, (%[V])      \n\t"
        "addi   %[W], %[W], 64     \n\t"
        "addi   %[V], %[V], 1024   \n\t"
        "vmadot v16, v0, v8        \n\t"
        "vmadot v18, v0, v9        \n\t"
        "vmadot v20, v0, v10       \n\t"
        "vmadot v22, v0, v11       \n\t"
        "vmadot v24, v0, v12       \n\t"
        "vmadot v26, v0, v13       \n\t"
        "vmadot v28, v0, v14       \n\t"
        "vmadot v30, v0, v15       \n\t"
        "addi   t3, t3, -1         \n\t"
        "bgtz   t3, I8P%=          \n\t"
        "vse32.v v16, (%[O0])      \n\t"
        "vse32.v v18, (%[O1])      \n\t"
        "vse32.v v20, (%[O2])      \n\t"
        "vse32.v v22, (%[O3])      \n\t"
        "vse32.v v24, (%[O4])      \n\t"
        "vse32.v v26, (%[O5])      \n\t"
        "vse32.v v28, (%[O6])      \n\t"
        "vse32.v v30, (%[O7])      \n\t"
        : [W] "+r"(w8g), [V] "+r"(vblk)
        : [NP] "r"(npblk),
          [O0] "r"(out[0]), [O1] "r"(out[1]), [O2] "r"(out[2]), [O3] "r"(out[3]),
          [O4] "r"(out[4]), [O5] "r"(out[5]), [O6] "r"(out[6]), [O7] "r"(out[7])
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// Same dots with plain unit-stride loads instead of whole-register ones. vl1re32
// wants 128-byte natural alignment and the A pointer steps 64 bytes per
// iteration, so every other load straddles a register boundary. vle32.v only
// needs the element alignment, so this separates that penalty from the dots.
static void pv4_le32(int32_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
    asm volatile(
        "vsetvli t0, x0, e32, m1\n\t"
        "vxor.vv v16, v16, v16\n\t"
        "vxor.vv v18, v18, v18\n\t"
        "vxor.vv v20, v20, v20\n\t"
        "vxor.vv v22, v22, v22\n\t"
        "vxor.vv v24, v24, v24\n\t"
        "vxor.vv v26, v26, v26\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vxor.vv v30, v30, v30\n\t"
        "mv     t3, %[NP]\n\t"
        "mv     t4, %[W]\n\t"
        "mv     t5, %[V]\n\t"
        "1:\n\t"
        "vle32.v v0, (t4)\n\t"
        // This assembler rejects any non-zero displacement on a vector load
        // (vle32.v v9,128(t5) is an error; vle32.v v9,(t5) is fine), so the
        // 1 KB V block is walked by advancing the base instead.
        "vle32.v v8,  (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v9,  (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v10, (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v11, (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v12, (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v13, (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v14, (t5)\n\t"
        "addi    t5, t5, 128\n\t"
        "vle32.v v15, (t5)\n\t"
        "addi    t4, t4, 64\n\t"
        "addi    t5, t5, 128\n\t"
        "vmadot  v16, v0, v8\n\t"
        "vmadot  v18, v0, v9\n\t"
        "vmadot  v20, v0, v10\n\t"
        "vmadot  v22, v0, v11\n\t"
        "vmadot  v24, v0, v12\n\t"
        "vmadot  v26, v0, v13\n\t"
        "vmadot  v28, v0, v14\n\t"
        "vmadot  v30, v0, v15\n\t"
        "addi    t3, t3, -1\n\t"
        "bgtz    t3, 1b\n\t"
        "vse32.v v16, (%[O0])\n\t"
        "vse32.v v18, (%[O1])\n\t"
        "vse32.v v20, (%[O2])\n\t"
        "vse32.v v22, (%[O3])\n\t"
        "vse32.v v24, (%[O4])\n\t"
        "vse32.v v26, (%[O5])\n\t"
        "vse32.v v28, (%[O6])\n\t"
        "vse32.v v30, (%[O7])\n\t"
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t4", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// Accumulating twin of pv4: keeps whatever is already in the accumulators instead
// of zeroing them, so a dot loop can be split into chunks (TCM staging) without
// losing partial sums.
static void pv4_acc(int32_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vl1re32.v v16, (%[O0])         \n\t"
        "vl1re32.v v18, (%[O1])         \n\t"
        "vl1re32.v v20, (%[O2])         \n\t"
        "vl1re32.v v22, (%[O3])         \n\t"
        "vl1re32.v v24, (%[O4])         \n\t"
        "vl1re32.v v26, (%[O5])         \n\t"
        "vl1re32.v v28, (%[O6])         \n\t"
        "vl1re32.v v30, (%[O7])         \n\t"
        "mv     t3, %[NP]               \n\t"
        "mv     t4, %[W]                \n\t"
        "mv     t5, %[V]                \n\t"
        "1:                             \n\t"
        "vl1re32.v v0, (t4)             \n\t"
        "vl8re32.v v8, (t5)             \n\t"
        "addi    t4, t4, 64             \n\t"
        "addi    t5, t5, 1024           \n\t"
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
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t4", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// the A operand carry the 4 queries, so a 64-byte stride is exact (the prototype
// used 128 with a padded layout).
static void pv4(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk, int vstride) {
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
        "mv     t3, %[NP]               \n\t"
        "mv     t4, %[W]                \n\t"
        "mv     t5, %[V]                \n\t"
        "1:                             \n\t"
        "vl1re32.v v0, (t4)             \n\t"
        "vl8re32.v v8, (t5)             \n\t"
        "addi    t4, t4, 64             \n\t"
        "add     t5, t5, %[VS]          \n\t"
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
        : /* no outputs */
        : [NP] "r"(npblk), [W] "r"(w8g), [V] "r"(vblk), [VS] "r"(vstride),
          [O0] "r"(acc[0]), [O1] "r"(acc[1]), [O2] "r"(acc[2]), [O3] "r"(acc[3]),
          [O4] "r"(acc[4]), [O5] "r"(acc[5]), [O6] "r"(acc[6]), [O7] "r"(acc[7])
        : "t0", "t3", "t4", "t5",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// Repack one layer of K and V from the f16 cache into the dot layouts, and pack Q
// once. Scales are one per (kv head) per layer for K and V: a per-block scale is
// not usable because the dot contracts over the block and the score has to be a
// single scaled integer sum.
// Two passes over each layer's KV: a max reduction for the int8 scale, then the
// quantise. Scalar and single-threaded this cost 6.3 s of the 63.8 s end-to-end;
// it is embarrassingly parallel over (kv head, position band), so both passes run
// on all eight A cores now.
static float g_cv_mk[N_KV][16], g_cv_mv[N_KV][16];
static float g_cv_ik[N_KV], g_cv_iv[N_KV];

static int64_t g_cv_lo;                     // first position this call converts

static inline void cv_band(int tid, int64_t lo0, int64_t npos, int64_t * lo, int64_t * hi) {
    int64_t a, b; band_w(CV_W, tid, npos - lo0, &a, &b);
    *lo = lo0 + a; *hi = lo0 + b;
}

// Both conv passes were scalar element loops: 2.36 s of a 58 s prefill, and all of
// it in address arithmetic rather than arithmetic. The two layouts share one shape
// -- 16 lanes of a vector with a fixed stride, producing 16 *contiguous* output
// bytes -- so one vector step replaces sixteen scalar ones:
//
//   V  16 consecutive positions at fixed dim: source stride N_KV*HEAD_DIM*2,
//      destination (d%64)*16 + (p%16) is contiguous.
//   K  16 consecutive dims at fixed position: source stride KT*2 (the tile is
//      dim-major), destination (d%16) is contiguous.
//
// Rounding stays lrintf-compatible (vfcvt with the default round-to-nearest-even)
// and the max pass stays exact because |f16| values are a subset of |f32|.
static void task_pfa_conv_max(int tid) {
    int64_t lo, hi; cv_band(tid, g_cv_lo, g_cv_npos, &lo, &hi);
    // The dim walk is chunked by vsetvl rather than assuming one register holds a
    // whole head: the A cluster is VLEN=1024 and the X100 cluster VLEN=256, so
    // vsetvl_e16m4(128) returns 64 there and the tail would be skipped.
    for (int kv = 0; kv < N_KV; ++kv) {
        float mk = 0, mv = 0;
        for (int64_t p = lo; p < hi; ++p) {
            const _Float16 * kb = ktile(g_cv_layer, kv, (int) (p / KT));
            const _Float16 * vb = g_vpos + (int64_t) g_cv_layer * KV_LAYER_STRIDE
                                + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
            float a = 0, b = 0;
            for (int d = 0; d < HEAD_DIM; ) {
                const size_t vl = __riscv_vsetvl_e16m4((size_t) (HEAD_DIM - d));
                // reduce in f32: the f16 reduction form ICEs this GCC, and widening
                // is exact, so the maximum is the same real number either way
                const vfloat32m8_t ka = __riscv_vfabs_v_f32m8(__riscv_vfwcvt_f_f_v_f32m8(
                    __riscv_vlse16_v_f16m4(kb + d * KT + (p % KT),
                                           KT * (int64_t) sizeof(_Float16), vl), vl), vl);
                const vfloat32m8_t va = __riscv_vfabs_v_f32m8(
                    __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(vb + d, vl), vl), vl);
                const vfloat32m1_t kz = __riscv_vfmv_s_f_f32m1(0.0f, 1);
                const float da = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(ka, kz, vl));
                const float db = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(va, kz, vl));
                if (da > a) a = da;
                if (db > b) b = db;
                d += (int) vl;
            }
            if (a > mk) mk = a;
            if (b > mv) mv = b;
        }
        g_cv_mk[kv][tid] = mk; g_cv_mv[kv][tid] = mv;
    }
}

// 16 strided halves in, 16 contiguous int8 out.
static inline void conv16(const _Float16 * src, int64_t stride_elems, float inv, int8_t * dst) {
    // LMUL m2/m4 with a fixed sixteen lanes: e16mf2 holds only eight lanes on a
    // VLEN=256 core, and chunking it there left the second half of every block
    // unwritten. e16m2 holds sixteen on both clusters (VLEN 256 and 1024).
    const size_t vl = __riscv_vsetvl_e16m2(16);
    const vfloat16m2_t h = __riscv_vlse16_v_f16m2(src, stride_elems * (int64_t) sizeof(_Float16), vl);
    vfloat32m4_t f = __riscv_vfmul_vf_f32m4(__riscv_vfwcvt_f_f_v_f32m4(h, vl), inv, vl);
    vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4(f, vl);
    q = __riscv_vmax_vx_i32m4(q, -127, vl);
    q = __riscv_vmin_vx_i32m4(q, 127, vl);
    __riscv_vse8_v_i8m1(dst, __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(q, vl), vl), vl);
}

static void task_pfa_conv_q(int tid) {
    int64_t lo, hi; cv_band(tid, g_cv_lo, g_cv_npos, &lo, &hi);
    // whole 16-position V blocks only; lo/hi are position bounds, so round them
    // down to the block grid. A trailing partial block keeps the scalar loop that
    // this replaced -- writing a full 16-lane block past npos would walk off g_v8i.
    const int64_t vlo = lo & ~(int64_t) 15;
    int64_t vhi = (hi + 15) & ~(int64_t) 15;
    if (vhi > g_cv_npos) vhi = g_cv_npos & ~(int64_t) 15;
    for (int kv = 0; kv < N_KV; ++kv) {
        const float ik = g_cv_ik[kv], iv = g_cv_iv[kv];
        int8_t * vbase = v8i_base(g_cv_layer) + kv * g_v8i_npb16 * 2048;
        // K: per position, 16 dims at a time, all contiguous this position's lane
        for (int64_t p = lo; p < hi; ++p) {
            const _Float16 * kb = ktile(g_cv_layer, kv, (int) (p / KT));
            int8_t * kdst = k8i_base(g_cv_layer) + ((kv * g_k8i_npb + p / 64) * 8) * 1024
                          + (p % 64) * 16;
            for (int dg = 0; dg < HEAD_DIM / 16; ++dg)
                conv16(kb + dg * 16 * KT + (p % KT), KT, ik, kdst + dg * 1024);
        }
        // V: per 16-position block, 16 dims at a time
        for (int64_t pb = vlo; pb < vhi; pb += 16) {
            const _Float16 * vb = g_vpos + (int64_t) g_cv_layer * KV_LAYER_STRIDE
                                + pb * N_KV * HEAD_DIM + kv * HEAD_DIM;
            int8_t * vdst = vbase + (pb / 16) * 1024;
            for (int d = 0; d < HEAD_DIM; ++d)
                conv16(vb + d, N_KV * HEAD_DIM, iv,
                       vdst + ((d / 64) * g_v8i_npb16) * 1024 + (d % 64) * 16);
        }
        for (int64_t p = (vhi > lo ? vhi : lo); p < hi; ++p) {
            const _Float16 * vb = g_vpos + (int64_t) g_cv_layer * KV_LAYER_STRIDE
                                + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
            int8_t * vdst = vbase;
            for (int d = 0; d < HEAD_DIM; ++d) {
                int b = (int) lrintf((float) vb[d] * iv);
                if (b > 127) b = 127; if (b < -127) b = -127;
                vdst[((d / 64) * g_v8i_npb16 + p / 16) * 1024 + (d % 64) * 16 + (p % 16)] = (int8_t) b;
            }
        }
    }
}

// Every chunk reconverts the whole prefix: 20 chunks over 10240 rows is ~107k
// position conversions per layer instead of 10k, 6.4 s of the 76 s prefill.
// Converting only the new positions (one pass per chunk, rescaling the prefix when
// a new maximum appears) leaves the same bytes in the cache at the end, but the
// greedy chain follows the scale each position carried while prefill ran: at 10K
// that version picks 11127 where the whole-prefix rewrite picks 5, so the rewrite
// stays. The strides are still pinned to the buffer capacity, which keeps every
// already-converted block at one address instead of moving it as the prefix grows.
static void pfa_convert_kv_i8(int layer, int64_t npos) {
    g_k8i_npb = g_k8i_npb_cap;
    g_v8i_npb16 = g_v8i_npb16_cap;
    g_cv_layer = layer; g_cv_npos = npos;
    if (npos <= 0) return;
    const double _cm0 = now_s();
    g_cv_lo = 0;
    if (getenv("MEGA_NOEL_CONV")) run(task_pfa_conv_max); else run_el(task_pfa_conv_max);
    const double _cm1 = now_s();
    for (int kv = 0; kv < N_KV; ++kv) {
        // the reduction runs after run_el restored g_el_parts, so it has to use
        // the band count the scan actually wrote
        float mk = 0, mv = 0;
        for (int t = 0; t < g_el_last; ++t) {
            if (g_cv_mk[kv][t] > mk) mk = g_cv_mk[kv][t];
            if (g_cv_mv[kv][t] > mv) mv = g_cv_mv[kv][t];
        }
        const float sk = mk > 0 ? mk / 127.0f : 1.0f;
        const float sv = mv > 0 ? mv / 127.0f : 1.0f;
        g_k8isc[kv] = sk; g_v8isc[kv] = sv;
        // A four-row decode appends positions to these planes one row at a time,
        // and each appended position has to land on the grid the prefix already
        // uses. This is the last conversion of this layer, so it is also the one
        // whose bytes survive.
        g_k8i_sc[layer * N_KV + kv] = sk; g_v8i_sc[layer * N_KV + kv] = sv;
        if (getenv("MEGA_SCL") && layer == 0)
            fprintf(stderr, "  scl kv%d npos %lld sk %.9g sv %.9g\n", kv, (long long) npos, (double) sk, (double) sv);
        g_cv_ik[kv] = 1.0f / sk; g_cv_iv[kv] = 1.0f / sv;
    }
    if (getenv("MEGA_NOEL_CONV")) run(task_pfa_conv_q); else run_el(task_pfa_conv_q);
    if (getenv("MEGA_KVDDUMP") && layer == 0) {
        FILE * fp = fopen(getenv("MEGA_KVDDUMP"), "wb");
        fwrite(g_k8i, 1, (size_t) N_KV * g_k8i_npb * 8192, fp);
        fwrite(g_v8i, 1, (size_t) N_KV * g_v8i_npb16 * 2048, fp);
        fclose(fp);
    }
    if (getenv("MEGA_TIME"))
        fprintf(stderr, "  conv L%d: max %.3f s, quant %.3f s\n", layer, _cm1 - _cm0, now_s() - _cm1);
}

// One (row group, head) pack: max-scan the row, then scale to int8 into the
// 4-row x 16-dim block the QK dots read. Split across threads by group: the
// whole loop used to run on one core and cost 2.4 s of a 60 s prefill.
static void pfa_pack_q_band(int64_t lo, int64_t hi) {
    for (int64_t g = lo; g < hi; ++g)
        for (int h = 0; h < N_HEAD; ++h) {
            int8_t * dst = g_q8i + ((h * g_q8i_nqg + g) * 8) * 64;
            for (int r = 0; r < 4; ++r) {
                const int64_t qr = g * 4 + r;
                if (qr >= g_pq_rows) break;
                const float * q = g_pq + (qr * N_HEAD + h) * HEAD_DIM;
                const size_t v16 = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t vmx = __riscv_vfmv_v_f_f32m1(0.0f, v16);
                for (int d = 0; d < HEAD_DIM; d += 16)
                    vmx = __riscv_vfmax_vv_f32m1(vmx,
                        __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(q + d, v16), v16), v16);
                const float mx = __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredmax_vs_f32m1_f32m1(vmx, __riscv_vfmv_s_f_f32m1(0.0f, 1), v16));
                const float s = mx > 0 ? mx / 127.0f : 1.0f;
                g_q8isc[qr * N_HEAD + h] = s;
                const float inv = 1.0f / s;
                for (int d = 0; d < HEAD_DIM; d += 16) {
                    vfloat32m1_t v = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(q + d, v16), inv, v16);
                    vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(v, v16);
                    iv = __riscv_vmax_vx_i32m1(iv, -127, v16);
                    iv = __riscv_vmin_vx_i32m1(iv, 127, v16);
                    __riscv_vse8_v_i8mf4(dst + (d / 16) * 64 + r * 16,
                        __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(iv, v16), v16), v16);
                }
            }
        }
}

static void pfa_pack_q(int64_t rows) {
    g_q8i_nqg = (rows + 3) / 4;
    g_pq_rows = rows;
    parallel_for(g_q8i_nqg, pfa_pack_q_band);
    g_pq_rows = 0;
}

// One tile of I8_ROWS queries of one head: QK, softmax, requantise the weights,
// then PV. QK and softmax are shared with the pipelined path so both use the same
// score layout and there is only one copy of that arithmetic to keep correct.
static void pfa_qk_i8(int h, int64_t qbase, int16_t * sc);
static void pfa_sm_i8(int h, int64_t qbase, const int16_t * sc, int8_t * w8, float * wsc);

static void pfa_head_i8(int layer, int h, int64_t qbase) {
    const int kv = h / NH_PER_KV;
    const int nn = (int) (g_pfa_ctx + qbase + PFA_ROWS * g_i8g);
    const int npb = (nn + 63) / 64;
    int16_t * sc = g_psa16 + (size_t) g_pfa_tid * SC_TSTRIDE;
    alignas(64) static __thread int8_t w8[I8_G * WBLK + 256];
    static __thread float wsc[I8_ROWS];
    g_pfa_smid = g_pfa_tid & 7;

    const double _q = now_s();
    pfa_qk_i8(h, qbase, sc);
    g_pfa_qk[(g_pfa_tid) * PFA_TSTRIDE] += now_s() - _q;

    const double _s = now_s();
    pfa_sm_i8(h, qbase, sc, w8, wsc);
    g_pfa_sm[(g_pfa_tid) * PFA_TSTRIDE] += now_s() - _s;

    // PV: two passes of 64 dims, contracting 16 positions per step. The four row
    // groups read the same V blocks, so they stay hot across the tile.
    const double _v = now_s();
    alignas(64) static __thread int32_t praw[8][32];
    alignas(64) static __thread int32_t pacc[I8_G][8][32];
    const int nq = (nn + 15) / 16;
    for (int half = 0; half < 2; ++half) {
        // Each kv head owns two 64-dim halves, i.e. 2 * npb16 blocks of 1024 B.
        // Adding only half*npb16 put kv 1's first half on kv 0's second half, so
        // heads 8..15 attended over the wrong V.
        const int8_t * vb = v8i_base(layer) + ((int64_t) kv * 2 + half) * g_v8i_npb16 * 1024;
        if (g_vsrc == 1) vb = (const int8_t *) g_wblob + (size_t) half * 65536;
        if (g_vsrc == 2) vb = g_k8i + (kv * g_k8i_npb + half * 8) * 1024;
        for (int gg = 0; gg < g_i8g; ++gg) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) praw[j];
            pv4(op, w8 + gg * WBLK, vb, (int) (g_pfa_ctx + qbase + gg * 4 + 4 + 15) / 16, 1024);
            if (half == 0 && gg == 0) {
                g_dbg_h0 = praw[0][24]; g_dbg_h0b = praw[0][25];
                if (g_pfa_dbg1 && layer == 0 && h == 0 && qbase == 0 && g_pfa_tid == 0) {
                    long e = 0;
                    int8_t * vb0 = g_v8i + 0;
                    for (int p2 = 0; p2 < 16; ++p2) e += (long) w8[48 + p2] * vb0[p2];
                    printf("[p] gg0 praw r3 = %d %d %d %d | dot=%ld | w8 r3 =",
                           praw[0][24], praw[0][25], praw[0][26], praw[0][27], e);
                    for (int p2 = 0; p2 < 6; ++p2) printf(" %d", w8[48 + p2]);
                    printf(" | vb0 ="); for (int p2 = 0; p2 < 4; ++p2) printf(" %d", vb0[p2]);
                    printf(" | nq=%d\n", nq);
                }
            }
            if (!g_pfa_out) continue;
            const size_t vl8 = __riscv_vsetvl_e32m1(8);
            for (int r = 0; r < 4; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = wsc[rr] * g_v8isc[kv];
                float * dst = g_pfa_out + (((qbase + rr) * N_HEAD + h) * HEAD_DIM) + half * 64;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&praw[j][r * 8], vl8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    __riscv_vse32_v_f32m1(dst + j * 8, f, vl8);
                }
            }
        }
    }
    g_pfa_pv[(g_pfa_tid) * PFA_TSTRIDE] += now_s() - _v;
    if (g_pfa_dbg1 && layer == 0 && h == 0 && qbase == 0 && g_pfa_tid == 0) {
        g_pfa_dbg1 = 0;
        printf("[i8 dbg] nn=%d npb=%d nq=%d wsc=%.6g %.6g %.6g %.6g ksc=%.6g vsc=%.6g\n",
               nn, npb, nq, wsc[0], wsc[1], wsc[2], wsc[3], g_k8isc[kv], g_v8isc[kv]);
        printf("[i8 dbg] sum3=%.6g -> implied sum %.6g\n",
               1.0f / (127.0f * wsc[3]), 1.0f / (127.0f * wsc[3]));
        printf("[i8 dbg] w8 r3[0..5] = %d %d %d %d %d %d\n",
               w8[48], w8[49], w8[50], w8[51], w8[52], w8[53]);
        printf("[i8 dbg] praw0 row3 lanes 24..27 = %d %d %d %d\n",
               praw[0][24], praw[0][25], praw[0][26], praw[0][27]);
        int8_t * vb0 = g_v8i + 0;
        printf("[i8 dbg] v8 dim0 pos0..3 = %d %d %d %d\n",
               vb0[0], vb0[16], vb0[32], vb0[48]);
        printf("[i8 dbg] out r3 dim0 = %.6f  praw_h0=%d h0b=%d\n",
               g_pfa_out[((3 * N_HEAD) + 0) * HEAD_DIM], g_dbg_h0, g_dbg_h0b);
        long e0 = 0, e1 = 0;
        for (int p = 0; p < 16; ++p) { e0 += (long) w8[48 + p] * vb0[p]; e1 += (long) w8[49 + p] * vb0[p]; }
        printf("[i8 dbg] scalar dot dim0=%ld dim1=%ld  vb0[0..3]=%d %d %d %d\n",
               e0, e1, vb0[0], vb0[1], vb0[2], vb0[3]);
    }
}

static void task_pfa_i8(int tid) {
    g_pfa_tid = tid;
    const int h0 = tid * 2;
    for (int64_t qb = 0; qb + I8_ROWS <= g_pfa_rows; qb += I8_ROWS)
        for (int h = h0; h < h0 + 2; ++h)
            pfa_head_i8(g_pfa_layer, h, qb);
}

// Decode: one four-row group per head, every position in the cache. Head-major
// over the eight A cores, two heads each. The tile is four rows rather than
// I8_ROWS because at this size the cost is the KV stream, not the dots: QK,
// softmax and PV all scale with the group count while the bytes do not, so
// sixteen groups would do sixteen times the arithmetic for the same reads.
static void task_pfa_i8_4(int tid) {
    g_pfa_tid = tid;
    for (int h = tid; h < N_HEAD; h += g_participants)
        pfa_head_i8(g_pfa_layer, h, 0);
}

// ------------------------------------------ cross-cluster softmax pipeline
// QK and PV need `vmadot`, which only the A100 cluster implements (VLEN=1024,
// 512 MAC per instruction). The softmax between them is plain RVV: at 10K it
// costs 123 core-s, as much as PV, and it was serialising the A100 cores that
// could have been doing dots. The idle X100 cluster runs the same maths at
// 4.5 ns/element against A100's 2.24 (both measured bit-exact), so moving all of
// it there costs those eight cores ~15 s while QK+PV needs ~28 s of A100 time:
// the softmax disappears behind the dots instead of adding to them.
//
// Coordination is one stage flag per tile. A claims tiles for QK from a shared
// counter and publishes stage 1; B claims them from a second counter walked in
// the same order and publishes stage 2; A waits for stage 2 before PV. Every A
// thread owns three slots (two tiles in flight plus the one draining), so slot
// ownership is per-thread and no buffer needs locking.
// Tile order. Head-fast means a thread cycles through all sixteen heads, so its
// live V+K working set is every head's (2.6 MB) and eight threads hold 21 MB.
// Head-major keeps a thread inside one head's KV for a stretch instead: each head
// is 1.31 MB per kv plane and the other threads on the same kv share it.
static inline int tile_h(int64_t t) {
    if (!g_headmajor) return (int) (t % N_HEAD);
    const int64_t nqb = g_pfa_rows / I8_ROWS;
    return (int) (t / nqb);
}
static inline int64_t tile_qb(int64_t t) {
    if (!g_headmajor) return (t / N_HEAD) * I8_ROWS;
    const int64_t nqb = g_pfa_rows / I8_ROWS;
    return (t % nqb) * I8_ROWS;
}

// Slot count trades memory against pipelining depth. A slot holds one tile's
// scores (I8_G groups) plus its requantised weights, so the size follows I8_G;
// the product stays near the 56.7 MB measured at I8_G=16 and I8_NSLOT=24.
static const int I8_NSLOT = 24 * 16 / I8_G;
static const int I8_INFLIGHT = 2;

struct TileSlot {
    int16_t scores[SC_TSTRIDE];      // interleaved 4-row x 8-position blocks
    int8_t  w8[I8_G * WBLK + 4096];
    float   wsc[I8_ROWS];
};
struct TileRec {
    alignas(64) std::atomic<int> st;    // 0 free, 1 qk done, 2 softmax done
    int slot;
};
static TileSlot * g_tslot;
static TileRec  * g_tile;
static int64_t    g_tile_n;
static std::atomic<int64_t> g_qk_next{0}, g_sm_next{0};
static std::atomic<int> g_pipe_gen{0}, g_sm_done{0};
static int g_n_sm = 0;
static int g_pipe = 0;
static int g_pipe_layer = 0;

// QK for one tile. Scores land in the interleaved block layout described at
// SC_NBLK above: the epilogue is one vnsra plus one 64-byte store per position
// block. sc is slot memory in the pipelined path and thread-local in the
// single-cluster one, so the body is shared.
// Same dots as qk4, but the score stores happen inside the asm.
//
// qk4 hands its eight int32 accumulators back through memory: eight vse32, then
// the caller does eight vle32, eight vnsra and eight vse16. That round trip is
// two vector accesses per store for nothing -- the values are in v16..v30 and the
// narrowing can read them there. This variant does vnsra+vse16 straight from the
// accumulators, so the eight vse32 and eight vle32 disappear. Scores are
// bit-identical: same shift, same shift-out, same destination.
static void qk4n(int16_t * const * dst, const int8_t * q8g, const int8_t * kblk,
                 int nks, int shift) {
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
        "vl1re32.v v1, (%[Q])           \n\t"
        "vl8re32.v v8, (%[K])           \n\t"
        "addi    %[Q], %[Q], 64         \n\t"
        "addi    %[K], %[K], 1024       \n\t"
        "vmadot  v16, v1, v8            \n\t"
        "vmadot  v18, v1, v9            \n\t"
        "vmadot  v20, v1, v10           \n\t"
        "vmadot  v22, v1, v11           \n\t"
        "vmadot  v24, v1, v12           \n\t"
        "vmadot  v26, v1, v13           \n\t"
        "vmadot  v28, v1, v14           \n\t"
        "vmadot  v30, v1, v15           \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        // vsetvli changes the element width, so the mask-out and the narrowing
        // both use the 16-bit view of the same 32 accumulators.
        "vsetvli t0, x0, e16, mf2       \n\t"
        "vnsra.wx v16, v16, %[SH]       \n\t"
        "vnsra.wx v18, v18, %[SH]       \n\t"
        "vnsra.wx v20, v20, %[SH]       \n\t"
        "vnsra.wx v22, v22, %[SH]       \n\t"
        "vnsra.wx v24, v24, %[SH]       \n\t"
        "vnsra.wx v26, v26, %[SH]       \n\t"
        "vnsra.wx v28, v28, %[SH]       \n\t"
        "vnsra.wx v30, v30, %[SH]       \n\t"
        "vse16.v v16, (%[O0])           \n\t"
        "vse16.v v18, (%[O1])           \n\t"
        "vse16.v v20, (%[O2])           \n\t"
        "vse16.v v22, (%[O3])           \n\t"
        "vse16.v v24, (%[O4])           \n\t"
        "vse16.v v26, (%[O5])           \n\t"
        "vse16.v v28, (%[O6])           \n\t"
        "vse16.v v30, (%[O7])           \n\t"
        : [Q] "+r"(q8g), [K] "+r"(kblk)
        : [NK] "r"(nks), [SH] "r"(shift),
          [O0] "r"(dst[0]), [O1] "r"(dst[1]), [O2] "r"(dst[2]), [O3] "r"(dst[3]),
          [O4] "r"(dst[4]), [O5] "r"(dst[5]), [O6] "r"(dst[6]), [O7] "r"(dst[7])
        : "t0", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "memory");
}

// On by default: it measured 57.53 -> 54.13 s at 10K with QK 83.1 -> 58.6 core-s,
// i.e. the whole score epilogue. MEGA_QKFUSE=0 restores the memory round trip.
static int g_qkfuse = 1;
static int g_qk2 = 0;   // MEGA_QK2: two row groups per K fetch
static int g_nosm = 0;  // MEGA_NOSM: X100 skips the score reads (timing only)
static int g_qknost = 0; // MEGA_QKNOST: one score store in eight (timing only)

static void pfa_qk_i8(int h, int64_t qbase, int16_t * sc) {
    const int kv = h / NH_PER_KV;
    const int nn = (int) (g_pfa_ctx + qbase + PFA_ROWS * g_i8g);
    const int npb = (nn + 63) / 64;
    const int8_t * q8g = g_q8i + ((h * g_q8i_nqg) + qbase / 4) * 8 * 64;
    alignas(64) static __thread int32_t raw[8][32];
    const size_t vl32 = __riscv_vsetvl_e32m1(32);
    const size_t vl16 = __riscv_vsetvl_e16mf2(32);
    for (int pb = 0; pb < npb; ++pb) {
        const int8_t * kb = k8i_base(g_pfa_layer) + ((kv * g_k8i_npb + pb) * 8) * 1024;
        int16_t * blk = sc + (size_t) (pb * 8) * 32;
        for (int gg = 0; gg < g_i8g; ++gg) {
            if (g_qkfuse && !g_nostore) {
                int16_t * dp[8];
                for (int j = 0; j < 8; ++j)
                    dp[j] = blk + (size_t) gg * SC_GSTRIDE + j * 32;
                qk4n(dp, q8g + gg * 512, kb, HEAD_DIM / 16, SC_SHIFT);
                continue;
            }
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
            qk4(op, q8g + gg * 512, kb, HEAD_DIM / 16);
            // Diagnostic: parking every store in one L1 block removes the score
            // buffer's write traffic without changing the dot loop, which bounds
            // what compressing the scores could buy.
            int16_t * dst = g_nostore ? g_nspark : blk + (size_t) gg * SC_GSTRIDE;
            for (int j = 0; j < 8; ++j) {
                const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                // MEGA_QKNOST keeps one store in eight so the accumulators are
                // still live; the loop's cost without the seven missing stores is
                // what the score epilogue actually charges. Timing only.
                if (g_qknost && j) continue;
                __riscv_vse16_v_i16mf2(dst + j * 32,
                                       __riscv_vnsra_wx_i16mf2(a, SC_SHIFT, vl32), vl16);
            }
        }
    }
}

// Softmax + requantise, one row group at a time. The max pass is lane-wise over
// blocks and needs a single cross-lane reduce per group, because lane r*8+c of
// every block belongs to row r: only blocks 0..nb-1 are read, so the reduce
// never sees memory QK has not written. The exp pass runs 32 lanes (four rows x
// eight positions) per step; each row's eight weights go to its own slot of the
// pv4 A layout, which is what the four slidedowns at the end are for.
static void pfa_sm_i8(int h, int64_t qbase, const int16_t * sc, int8_t * w8, float * wsc) {
    const int kv = h / NH_PER_KV;
    const int L = (int) (g_pfa_ctx + qbase);
    alignas(64) static __thread float kkl[I8_G][32], mml[I8_G][32];
    alignas(64) static __thread int16_t mxs[I8_G][32];
    alignas(64) static __thread float accs[I8_G][32];
    const size_t vlz = __riscv_vsetvl_e16m2(32);
    const size_t vl4 = __riscv_vsetvl_e32m4(32);

    for (int gg = 0; gg < g_i8g; ++gg) {
        const int16_t * base = sc + (size_t) gg * SC_GSTRIDE;
        int8_t * wb = w8 + (size_t) gg * WBLK;
        const int nbv = (L + gg * 4 + 1) / 8;    // blocks all four rows of the group own
        // PV reads this group's blocks up to and including the one holding row 3's
        // last position, so the positions past each row's causal limit have to be
        // zero. Only the group's own tail is involved: clearing all of w8 instead
        // was 640 KB per tile against a few KB of real work.
        const int wnb = (L + gg * 4 + 4 + 15) / 16;
        const int zlo = (L + gg * 4 + 1) / 16;
        if (zlo < wnb) memset(wb + (size_t) zlo * 64, 0, (size_t) (wnb - zlo) * 64);

        // exponent scale per lane: score = S * 2^SC_SHIFT * q_scale * k_scale
        for (int r = 0; r < 4; ++r) {
            const float k = (float) (1 << SC_SHIFT) *
                            g_q8isc[((qbase + gg * 4 + r) * N_HEAD + h)] * g_k8isc[kv] *
                            1.4426950408889634f;
            for (int c = 0; c < 8; ++c) kkl[gg][r * 8 + c] = k;
        }

        // Max over the blocks every row owns, then the last few positions of each
        // row by hand: QK fills whole eight-position blocks, so the tail of this
        // group still holds scores for positions past a row's own causal limit and
        // letting them into the max would cost int8 resolution.
        const double _mx = now_s();
        vint16m2_t vm = __riscv_vmv_v_x_i16m2((int16_t) -32768, vlz);
        for (int i = 0; i < nbv; ++i) {
            const size_t vl = __riscv_vsetvl_e16m2(32);
            vm = __riscv_vmax_vv_i16m2(
                vm, __riscv_vle16_v_i16m2(base + (size_t) i * 32, vl), vl);
        }
        vm = __riscv_vmax_vv_i16m2(vm, __riscv_vslidedown_vx_i16m2(vm, 4, vlz), vlz);
        vm = __riscv_vmax_vv_i16m2(vm, __riscv_vslidedown_vx_i16m2(vm, 2, vlz), vlz);
        vm = __riscv_vmax_vv_i16m2(vm, __riscv_vslidedown_vx_i16m2(vm, 1, vlz), vlz);
        __riscv_vse16_v_i16m2(mxs[gg], vm, vlz);
        for (int r = 0; r < 4; ++r) {
            const int len = L + gg * 4 + r + 1;
            int m = mxs[gg][r * 8];
            for (int p = nbv * 8; p < len; ++p) {
                const int S = base[(size_t) (p / 8) * 32 + r * 8 + (p % 8)];
                if (S > m) m = S;
            }
            mxs[gg][r * 8] = (int16_t) m;
        }
        g_pfa_mxb[(g_pfa_smid) * PFA_TSTRIDE] += now_s() - _mx;

        const double _ex = now_s();
        for (int r = 0; r < 4; ++r) {
            const float mm = (float) mxs[gg][r * 8] * kkl[gg][r * 8];
            for (int c = 0; c < 8; ++c) mml[gg][r * 8 + c] = mm;
        }
        const vfloat32m4_t klane = __riscv_vle32_v_f32m4(kkl[gg], vl4);
        const vfloat32m4_t mlane = __riscv_vle32_v_f32m4(mml[gg], vl4);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl4);
        for (int i = 0; i < nbv; ++i) {
            const size_t v16 = __riscv_vsetvl_e16m2(32);
            const size_t vl = __riscv_vsetvl_e32m4(32);
            const vint32m4_t s32 = __riscv_vsext_vf2_i32m4(
                __riscv_vle16_v_i16m2(base + (size_t) i * 32, v16), vl);
            vfloat32m4_t t = __riscv_vfmul_vv_f32m4(__riscv_vfcvt_f_x_v_f32m4(s32, vl), klane, vl);
            t = exp2f_e_m4(__riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(t, mlane, vl),
                                                     -127.0f, vl), vl);
            acc = __riscv_vfadd_vv_f32m4(acc, t, vl);
            const vint32m4_t iv = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(t, 127.0f, vl), vl);
            const vint8m1_t b8 = __riscv_vncvt_x_x_w_i8m1(
                __riscv_vncvt_x_x_w_i16m2(iv, vl), vl);
            const size_t v8 = __riscv_vsetvl_e8m1(8);
            for (int r = 0; r < 4; ++r) {
                const vint8m1_t part = r
                    ? __riscv_vslidedown_vx_i8m1(b8, (size_t) (r * 8), vl) : b8;
                __riscv_vse8_v_i8m1(wb + (size_t) (i / 2) * 64 + r * 16 + (i & 1) * 8, part, v8);
            }
        }
        // per-row sums: lanes r*8..r*8+7 of acc hold row r's partial sums
        vfloat32m4_t sa = acc;
        sa = __riscv_vfadd_vv_f32m4(sa, __riscv_vslidedown_vx_f32m4(sa, 4, vl4), vl4);
        sa = __riscv_vfadd_vv_f32m4(sa, __riscv_vslidedown_vx_f32m4(sa, 2, vl4), vl4);
        sa = __riscv_vfadd_vv_f32m4(sa, __riscv_vslidedown_vx_f32m4(sa, 1, vl4), vl4);
        __riscv_vse32_v_f32m4(accs[gg], sa, vl4);

        for (int r = 0; r < 4; ++r) {
            const int len = L + gg * 4 + r + 1;
            float sum = accs[gg][r * 8];
            // positions the vector pass could not cover, and the causal mask past
            // them: at most a couple of eight-position blocks
            for (int p = nbv * 8; p < len; ++p) {
                const int16_t S = base[(size_t) (p / 8) * 32 + r * 8 + (p % 8)];
                const float v = exp2f(((float) S - (float) mxs[gg][r * 8]) * kkl[gg][r * 8]);
                sum += v;
                wb[(size_t) (p / 16) * 64 + r * 16 + (p % 16)] = (int8_t) lrintf(v * 127.0f);
            }
            wsc[gg * 4 + r] = sum > 0 ? 1.0f / (127.0f * sum) : 0.0f;
        }
        g_pfa_exb[(g_pfa_smid) * PFA_TSTRIDE] += now_s() - _ex;
        if (g_pfa_dbg1 && qbase == 0 && gg == 0 && h == 0) {
            long sw = 0;
            for (int i = 0; i < 256; ++i) sw += wb[i];
            long ss = 0;
            for (int i = 0; i < 64; ++i) ss += base[i];
            printf("[dbg0] key=%d sw=%ld ss=%ld sc=%ld wsc0=%.6g wsc3=%.6g\n",
                   (int) (g_pfa_ctx * 1000000), (int) sw, (int) ss,
                   (long) sc[0], wsc[0], wsc[3]);
        }
        if (g_pfa_dbg1 && h == 0 && qbase == 1024 && gg == I8_G - 1) {
            double wsum = 0, wref = 0;
            for (int g2 = 0; g2 < I8_G; ++g2) {
                const int16_t * bb = sc + (size_t) g2 * SC_GSTRIDE;
                for (int r = 0; r < 4; ++r) {
                    const int len = L + g2 * 4 + r + 1;
                    int mxr = -32768;
                    for (int p = 0; p < len; ++p) {
                        const int16_t S = bb[(size_t) (p / 8) * 32 + r * 8 + (p % 8)];
                        if (S > mxr) mxr = S;
                    }
                    double sr = 0;
                    for (int p = 0; p < len; ++p) {
                        const int16_t S = bb[(size_t) (p / 8) * 32 + r * 8 + (p % 8)];
                        sr += exp2(((double) S - mxr) * (double) kkl[g2][r * 8]);
                    }
                    const double sv = 1.0 / ((double) wsc[g2 * 4 + r] * 127.0);
                    wsum += sv; wref += sr;
                    if (gg == I8_G - 1 && (fabs(sv - sr) > 1e-3 * sr || g2 < 2))
                        printf("[smc] gg%d r%d vec %.6g ref %.6g ratio %.5f mx %d %d len %d\n",
                               g2, r, sv, sr, sv / sr, (int) mxs[g2][r * 8], mxr, len);
                }
            }
            printf("[smc] tile qbase=%lld sum vec %.6g ref %.6g\n", (long long) qbase, wsum, wref);
            g_pfa_dbg1 = 0;
        }
    }
}

// PV for one tile: two passes of 64 dims, contracting 16 positions per step.
static void pfa_pv_i8(int layer, int h, int64_t qbase,
                      const int8_t * w8, const float * wsc) {
    const int kv = h / NH_PER_KV;
    const int nn = (int) (g_pfa_ctx + qbase + PFA_ROWS * g_i8g);
    const int L = (int) (g_pfa_ctx + qbase);
    // Row group gg never looks past position L+4gg+3, so its dots stop at the
    // 16-position block holding that position. The weights past that point are
    // zero by construction, so the truncation is exact and it removes just under
    // half of PV's MACs over a full triangle.
    alignas(64) static __thread int32_t praw[8][32];
    alignas(64) static __thread int32_t pacc[I8_G][8][32];
    int nqg[I8_G];
    for (int gg = 0; gg < g_i8g; ++gg) nqg[gg] = (L + gg * 4 + 4 + 15) / 16;

    for (int half = 0; half < 2; ++half) {
        // Each kv head owns two 64-dim halves, i.e. 2 * npb16 blocks of 1024 B.
        // Adding only half*npb16 put kv 1's first half on kv 0's second half, so
        // heads 8..15 attended over the wrong V.
        const int8_t * vb = v8i_base(layer) + ((int64_t) kv * 2 + half) * g_v8i_npb16 * 1024;
        if (g_vsrc == 1) vb = (const int8_t *) g_wblob + (size_t) half * 65536;
        if (g_vsrc == 2) vb = g_k8i + (kv * g_k8i_npb + half * 8) * 1024;

        if (g_hoist) {
            // Per-group walk of V, kept for A/B: reads V's range sixteen times per
            // tile, which is what made PV DRAM bound (131 core-s against 54 here).
            for (int gg = 0; gg < g_i8g; ++gg) {
                int8_t * op[8];
                for (int j = 0; j < 8; ++j) op[j] = (int8_t *) praw[j];
                pv4_hoist(op, w8 + gg * WBLK + g_w8off, vb, nqg[gg]);
            }
        } else {
            // V block outer, row group inner. All sixteen groups' dots run against
            // one fetch of a V block, so V is read once per tile instead of once per
            // group, and the accumulators are the only thing that grows: sixteen
            // sets of eight 32-lane registers, spilled to L1 between blocks.
            for (int gg = 0; gg < g_i8g; ++gg)
                for (int j = 0; j < 8; ++j) memset(pacc[gg][j], 0, 32 * 4);
            // TCM staging exists so sixteen row groups can share one fetch of a V
            // block. With a single group there is one pass already, so the staging
            // is a pure copy: at four rows and 10K it cost ~0.4 ms per layer.
            uint8_t * tcm = (g_tcmstage && g_i8g > 1) ? g_tcm[g_pfa_tid] : nullptr;
            if (tcm && g_tcm_size >= 64 * 1024) {
                const int64_t stage = g_tcm_size / 1024;   // position blocks that fit
                const int nqmax = (nn + 15) / 16;
                for (int64_t p0 = 0; p0 < nqmax; p0 += stage) {
                    const int64_t left = nqmax - p0;
                    const int n = (int) (left < stage ? left : stage);
                    memcpy(tcm, vb + p0 * 1024, (size_t) n * 1024);
                    for (int gg = 0; gg < g_i8g; ++gg) {
                        const int ns = nqg[gg] - (int) p0;
                        if (ns <= 0) continue;
                        int32_t * op[8];
                        for (int j = 0; j < 8; ++j) op[j] = pacc[gg][j];
                        pv4_acc(op, w8 + gg * WBLK + p0 * 64 + g_w8off,
                                (const int8_t *) tcm, ns < n ? ns : n);
                    }
                }
            } else {
                const int nqmax = (nn + 15) / 16;
                for (int gg = 0; gg < g_i8g; ++gg) {
                    if (nqg[gg] <= 0) continue;
                    int32_t * op[8];
                    for (int j = 0; j < 8; ++j) op[j] = pacc[gg][j];
                    pv4_acc(op, w8 + gg * WBLK + g_w8off, vb, nqg[gg]);
                }
                (void) nqmax;
            }
        }

        if (!g_pfa_out) continue;
        const size_t vl8 = __riscv_vsetvl_e32m1(8);
        for (int gg = 0; gg < g_i8g; ++gg)
            for (int r = 0; r < 4; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = wsc[rr] * g_v8isc[kv];
                float * dst = g_pfa_out + (((qbase + rr) * N_HEAD + h) * HEAD_DIM) + half * 64;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(
                        g_hoist ? &praw[j][r * 8] : &pacc[gg][j][r * 8], vl8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    __riscv_vse32_v_f32m1(dst + j * 8, f, vl8);
                }
            }
    }
}

// A-cluster worker: claim a tile, QK it, publish, then drain the oldest tile
// whose softmax has landed. The two-tile lead lets B work on tile i while the
// A cores are already on tile i+1.
static void task_pfa_pipe_A(int tid) {
    g_pfa_tid = tid;
    int64_t pend[I8_INFLIGHT];
    int npend = 0, ring = 0;
    for (;;) {
        const int64_t t = g_qk_next.fetch_add(1, std::memory_order_relaxed);
        if (t >= g_tile_n) break;
        if (npend == I8_INFLIGHT) {
            const int64_t old = pend[0];
            TileRec & r = g_tile[old];
            while (r.st.load(std::memory_order_acquire) < 2) sched_yield();
            TileSlot & sl = g_tslot[r.slot];
            const int64_t qb = tile_qb(old);
            const double _v = now_s();
            if (!g_nopv) pfa_pv_i8(g_pipe_layer, tile_h(old), qb, sl.w8 + g_w8off, sl.wsc);
            g_pfa_pv[(tid) * PFA_TSTRIDE] += now_s() - _v;
            pend[0] = pend[1];
            --npend;
        }
        const int h = tile_h(t);
        const int64_t qb = tile_qb(t);
        const int slot = tid * 3 + ring; ring = (ring + 1) % 3;
        TileRec & r = g_tile[t];
        r.slot = slot;
        TileSlot & sl = g_tslot[slot];
        const double _q = now_s();
        pfa_qk_i8(h, qb, sl.scores);
        const double _qq = now_s() - _q;
        g_pfa_qk[(tid) * PFA_TSTRIDE] += _qq;
        if (g_qk_trace && qb + I8_ROWS + (int64_t) g_pfa_ctx > 10100)
            fprintf(stderr, "[qk] ctx=%lld qb=%lld npb=%lld %.3f ms -> %.1f GMAC/s\n",
                    (long long) g_pfa_ctx, (long long) qb,
                    (long long) ((g_pfa_ctx + qb + I8_ROWS + 63) / 64), _qq * 1e3,
                    (double) I8_ROWS * (g_pfa_ctx + qb + I8_ROWS) * HEAD_DIM / _qq / 1e9);
        r.st.store(1, std::memory_order_release);
        pend[npend++] = t;
    }
    while (npend > 0) {
        const int64_t old = pend[0];
        TileRec & r = g_tile[old];
        while (r.st.load(std::memory_order_acquire) < 2) sched_yield();
        TileSlot & sl = g_tslot[r.slot];
        const int64_t qb = tile_qb(old);
        const double _v = now_s();
        pfa_pv_i8(g_pipe_layer, tile_h(old), qb, sl.w8 + g_w8off, sl.wsc);
        g_pfa_pv[(tid) * PFA_TSTRIDE] += now_s() - _v;
        pend[0] = pend[1];
        --npend;
    }
}

// B-cluster worker: walk the same tile order and take every softmax. Each thread
// owns the tiles it claimed, so stage 1 -> 2 needs no compare-exchange.
static void task_pfa_pipe_B(int tid) {
    g_pfa_smid = tid;
    const double _s = now_s();
    for (;;) {
        const int64_t t = g_sm_next.fetch_add(1, std::memory_order_relaxed);
        if (t >= g_tile_n) break;
        TileRec & r = g_tile[t];
        while (r.st.load(std::memory_order_acquire) < 1) sched_yield();
        TileSlot & sl = g_tslot[r.slot];
        const int64_t qb = tile_qb(t);
        const double _sm = now_s();
        // MEGA_NOSM: keep the handshake but never touch the score buffer, so the
        // A cluster's QK runs with no consumer reading what it wrote. Every other
        // isolation so far (-D, -0) left the consumer's reads in place, which is
        // the only structural difference left against probe_qk_real.
        if (!g_nosm) pfa_sm_i8(tile_h(t), qb, sl.scores, sl.w8 + g_w8off, sl.wsc);
        g_pfa_sm[(tid) * PFA_TSTRIDE] += now_s() - _sm;
        r.st.store(2, std::memory_order_release);
    }
    g_pfa_smb[(tid) * PFA_TSTRIDE] += now_s() - _s;
}

static void * hugemem(size_t n);
static void task_pfa(int tid);

static int64_t g_tile_cap = 0;
static void pfa_pipe_alloc(void) {
    // The tile count follows the row count, and prefill changes it per chunk. The
    // counter used to be frozen at the first call, which silently left every later
    // chunk with the wrong number of tiles.
    g_tile_n = (g_pfa_rows / I8_ROWS) * N_HEAD;
    if (g_tslot && g_tile_n <= g_tile_cap) return;
    const size_t sn = (size_t) I8_NSLOT * sizeof(TileSlot);
    const int64_t want = g_tile_n > 0 ? g_tile_n : (MAXCTX / I8_ROWS) * N_HEAD;
    const size_t tn = (size_t) want * sizeof(TileRec) + (2 << 20);
    if (!g_tslot) g_tslot = (TileSlot *) hugemem(sn);
    g_tile = (TileRec *) hugemem(tn);
    if (!g_tslot || !g_tile) { fprintf(stderr, "oom pipeline slots\n"); g_pipe = 0; return; }
    g_tile_cap = want;
    // PV measured 115 vs 195 core-s on identical binaries, so the slot pages are
    // what decides it. Huge pages make the physical mapping contiguous and the
    // set distribution the same on every run instead of a lottery.
    use_huge(g_tslot, sn);
    use_huge(g_tile, tn);
    printf("  pipeline: %lld tiles, %d slots, %.1f MB scratch (w8off=%d)\n",
           (long long) g_tile_n, I8_NSLOT, sn / 1e6, g_w8off);
}

// ------------------------------------- four-row decode attention over both clusters
// A decode layer has exactly sixteen head-tiles. The eight A cores take QK and PV,
// the eight X100 cores take the softmax between them, and the handoff is one flag
// per head rather than the tile machinery the 64-row path needs: there is nothing
// to claim, because each head has exactly one owner on each side.
//
// The split is worth it because softmax is the largest of the three stages at four
// rows (0.1 of 0.3 core-s at 10K) and the X100 cluster is otherwise idle for the
// whole decode pass. Critical path becomes max(QK+PV, softmax) instead of their sum.
// On by default: the X100 cluster is idle for the whole decode pass, and the split
// only moves the softmax there, bit-identically (the same int16 scores, the same
// exp, the same fp32 weights; only which cluster issues them changes). Measured at
// 10K over 300 rounds it is 21.33 tok/s against 20.79 without, so it is worth a
// default rather than an environment variable.
static int g_a4p = 1;                       // MEGA_ATTN4P=0 turns it off
static int16_t * g_sc4;                     // N_HEAD * SC_GSTRIDE int16
static int8_t  * g_w4;                      // N_HEAD * WBLK bytes
static float   * g_wsc4;                    // N_HEAD * I8_ROWS floats
static std::atomic<int> g_a4_gen{0};
static std::atomic<int> g_a4_qk[N_HEAD], g_a4_sm[N_HEAD];
static std::atomic<int> g_a4_done{0};

static void attn4p_alloc(void) {
    if (g_sc4) return;
    const size_t sn = (size_t) N_HEAD * SC_GSTRIDE * sizeof(int16_t) + (2 << 20);
    const size_t wn = (size_t) N_HEAD * WBLK + (2 << 20);
    g_sc4  = (int16_t *) hugemem(sn);
    g_w4   = (int8_t *) hugemem(wn);
    g_wsc4 = (float *) aligned_alloc(64, (size_t) N_HEAD * I8_ROWS * 4 + 64);
    if (!g_sc4 || !g_w4 || !g_wsc4) { fprintf(stderr, "oom attn4p\n"); g_a4p = 0; return; }
    use_huge(g_sc4, sn);
    use_huge(g_w4, wn);
    for (int h = 0; h < N_HEAD; ++h) { g_a4_qk[h].store(0); g_a4_sm[h].store(0); }
    printf("  attn4p: %.1f MB scratch, %d heads, softmax on the X100 cluster\n",
           (sn + wn) / 1e6, N_HEAD);
}

// A side. Both of this thread's QKs go out before either PV, because the wait for
// a head's softmax is otherwise dead time: with the heads interleaved the X100
// cluster works on head 0 while this core is still doing head 1's dots, and the
// layer costs QK+PV instead of QK+softmax+PV.
static void task_attn4_a(int tid) {
    g_pfa_tid = tid;
    const int gen = g_a4_gen.load(std::memory_order_acquire);
    int hs[N_HEAD]; int n = 0;
    for (int h = tid; h < N_HEAD; h += g_participants) {
        const double _q = now_s();
        pfa_qk_i8(h, 0, g_sc4 + (size_t) h * SC_GSTRIDE);
        g_pfa_qk[(tid) * PFA_TSTRIDE] += now_s() - _q;
        g_a4_qk[h].store(gen, std::memory_order_release);
        hs[n++] = h;
    }
    for (int i = 0; i < n; ++i) {
        const int h = hs[i];
        while (g_a4_sm[h].load(std::memory_order_acquire) != gen) sched_yield();
        const double _v = now_s();
        pfa_pv_i8(g_pfa_layer, h, 0, g_w4 + (size_t) h * WBLK,
                  g_wsc4 + (size_t) h * I8_ROWS);
        g_pfa_pv[(tid) * PFA_TSTRIDE] += now_s() - _v;
    }
}

// X100 side. Head t and t+8, matching the A side's stride so every head has one
// owner on each cluster.
static void task_attn4_sm(int id) {
    const int gen = g_a4_gen.load(std::memory_order_acquire);
    g_pfa_smid = id;
    for (int h = id; h < N_HEAD; h += 8) {
        while (g_a4_qk[h].load(std::memory_order_acquire) != gen) sched_yield();
        int16_t * sc  = g_sc4 + (size_t) h * SC_GSTRIDE;
        int8_t  * w8  = g_w4 + (size_t) h * WBLK;
        float   * wsc = g_wsc4 + (size_t) h * I8_ROWS;
        const double _s = now_s();
        pfa_sm_i8(h, 0, sc, w8, wsc);
        g_pfa_sm[(id) * PFA_TSTRIDE] += now_s() - _s;
        g_a4_sm[h].store(gen, std::memory_order_release);
    }
}

// ---------------------------------- four-row attention, one pass per KV block
// The two functions above give each head its own walk over the KV plane, and the
// eight heads of a kv head read the same bytes. Measured at 10K, QK costs 0.144
// core-s and PV 0.095 for 215 MB of unique K and V: that is 47 GB/s of traffic for
// a cache that can only serve it because most of the repeats hit L2.
//
// Here a thread owns one kv head and a quarter of the positions, walking positions
// outside and heads inside, so each 1 KB block of K or V is fetched once and feeds
// all eight heads. Softmax stays full range on the X100 cluster -- it is the one
// stage that cannot be split, because the max and the sum span every position --
// and PV's partial sums over disjoint position ranges simply add, so no rescaling
// or second max pass is needed anywhere.
static float * g_o4part = nullptr;          // 8 threads x 8 heads x 4 rows x 128
static int g_a4s = 0;                       // -MEGA_A4S

static void a4s_alloc(void) {
    if (g_o4part) return;
    const size_t n = (size_t) 8 * NH_PER_KV * PFA_ROWS * HEAD_DIM * 4;
    g_o4part = (float *) hugemem(n + (2 << 20));
    if (!g_o4part) { fprintf(stderr, "oom a4s\n"); g_a4s = 0; return; }
    use_huge(g_o4part, n + (2 << 20));
}

// One 64-position block of one head's scores, with the K block already fetched.
static void a4s_qk_blk(int h, int64_t pb, const int8_t * kb, int16_t * sc) {
    const int8_t * q8g = g_q8i + (size_t) h * g_q8i_nqg * 8 * 64;
    alignas(64) int32_t raw[8][32];
    const size_t vl32 = __riscv_vsetvl_e32m1(32);
    const size_t vl16 = __riscv_vsetvl_e16mf2(32);
    int8_t * op[8];
    for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
    qk4(op, q8g, kb, HEAD_DIM / 16);
    int16_t * dst = sc + (size_t) (pb * 8) * 32;
    for (int j = 0; j < 8; ++j) {
        const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
        __riscv_vse16_v_i16mf2(dst + j * 32, __riscv_vnsra_wx_i16mf2(a, SC_SHIFT, vl32), vl16);
    }
}

// PV over one thread's position range. The weights are already normalised, so the
// four range outputs add directly; the only thing the range changes is where the
// V block walk starts, and both w8 and the V plane advance one block per sixteen
// positions, so a single offset moves both.
static void a4s_pv_range(int layer, int h, const int8_t * w8, const float * wsc,
                         int64_t q_lo, int64_t q_hi, float * dst) {
    const int kv = h / NH_PER_KV;
    alignas(64) int32_t pacc[8][32];
    const int64_t nqmax = (g_pfa_ctx + PFA_ROWS + 15) / 16;
    int64_t n = (q_hi < nqmax ? q_hi : nqmax) - q_lo;
    if (n < 0) n = 0;
    for (int half = 0; half < 2; ++half) {
        const int8_t * vb = v8i_base(layer) + ((int64_t) kv * 2 + half) * g_v8i_npb16 * 1024;
        for (int j = 0; j < 8; ++j) memset(pacc[j], 0, 32 * 4);
        if (n > 0) {
            int32_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = pacc[j];
            pv4_acc(op, w8 + q_lo * 64, vb + q_lo * 1024, (int) n);
        }
        const size_t vl8 = __riscv_vsetvl_e32m1(8);
        for (int r = 0; r < PFA_ROWS; ++r) {
            const float s_r = wsc[r] * g_v8isc[kv];
            float * d = dst + (size_t) r * HEAD_DIM + half * 64;
            for (int j = 0; j < 8; ++j) {
                const vint32m1_t a = __riscv_vle32_v_i32m1(&pacc[j][r * 8], vl8);
                const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                    __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                __riscv_vse32_v_f32m1(d + j * 8, f, vl8);
            }
        }
    }
}

// Two heads per thread, both from the same kv head. Sixteen heads over eight A
// cores is two each whatever the split, but taking the two from one kv head means
// the K block a thread has just fetched feeds both of its heads, so the plane is
// read four times per layer instead of eight. Packing both heads into one
// position-outer walk is what keeps that reuse; publishing after the walk keeps the
// per-head handoff to softmax, which is what lets the X100 cluster overlap.
static void task_attn4s(int tid) {
    g_pfa_tid = tid;
    const int gen = g_a4_gen.load(std::memory_order_acquire);
    const int kv = tid / 4;
    const int h0 = kv * NH_PER_KV + (tid % 4) * 2;
    const int64_t npb = (g_pfa_ctx + PFA_ROWS + 63) / 64;

    const double _q = now_s();
    const int8_t * kplane = k8i_base(g_pfa_layer);
    int16_t * sc0 = g_sc4 + (size_t) h0 * SC_GSTRIDE;
    for (int64_t pb = 0; pb < npb; ++pb) {
        const int8_t * kb = kplane + ((kv * g_k8i_npb + pb) * 8) * 1024;
        a4s_qk_blk(h0,     pb, kb, sc0);
        a4s_qk_blk(h0 + 1, pb, kb, sc0 + SC_GSTRIDE);
    }
    g_pfa_qk[(tid) * PFA_TSTRIDE] += now_s() - _q;

    // MEGA_A4S=2 keeps the softmax on this cluster. The handoff to the X100 cores
    // costs more than it saves at four rows: the A cores are idle for the whole
    // softmax and it cannot start until every position of a head is scored, so the
    // layer costs qk + sm + pv rather than overlapping them. Softmax is 2.24 ns per
    // element here against 4.5 on an X100, so running it in place is cheaper than
    // waiting for it elsewhere.
    const bool sm_here = (g_a4s == 2);
    if (sm_here) {
        g_pfa_smid = tid & 7;
        for (int i = 0; i < 2; ++i) {
            const int h = h0 + i;
            const double _s = now_s();
            pfa_sm_i8(h, 0, g_sc4 + (size_t) h * SC_GSTRIDE,
                      g_w4 + (size_t) h * WBLK, g_wsc4 + (size_t) h * I8_ROWS);
            g_pfa_sm[(tid) * PFA_TSTRIDE] += now_s() - _s;
        }
    } else {
        g_a4_qk[h0].store(gen, std::memory_order_release);
        g_a4_qk[h0 + 1].store(gen, std::memory_order_release);
    }

    for (int i = 0; i < 2; ++i) {
        const int h = h0 + i;
        if (!sm_here) while (g_a4_sm[h].load(std::memory_order_acquire) != gen) sched_yield();
        const double _v = now_s();
        pfa_pv_i8(g_pfa_layer, h, 0, g_w4 + (size_t) h * WBLK,
                  g_wsc4 + (size_t) h * I8_ROWS);
        g_pfa_pv[(tid) * PFA_TSTRIDE] += now_s() - _v;
    }
    (void) gen;
}

// The four quarters' partial outputs add: the softmax weights are already global.
static void task_attn4s_fin(int tid) {
    const size_t vl = __riscv_vsetvl_e32m1(8);
    const int64_t tstride = (int64_t) NH_PER_KV * PFA_ROWS * HEAD_DIM;
    for (int it = tid; it < N_HEAD * PFA_ROWS; it += g_participants) {
        const int h = it / PFA_ROWS, r = it % PFA_ROWS;
        const int kv = h / NH_PER_KV, i = h % NH_PER_KV;
        float * dst = g_pfa_out + ((size_t) r * N_HEAD + h) * HEAD_DIM;
        const float * base = g_o4part + (size_t) (kv * 4) * tstride
                           + (size_t) i * PFA_ROWS * HEAD_DIM + (size_t) r * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d += 8) {
            vfloat32m1_t a = __riscv_vle32_v_f32m1(base + d, vl);
            for (int q = 1; q < 4; ++q)
                a = __riscv_vfadd_vv_f32m1(a, __riscv_vle32_v_f32m1(base + q * tstride + d, vl), vl);
            __riscv_vse32_v_f32m1(dst + d, a, vl);
        }
    }
}

static void a4s_run(int L) {
    g_pfa_layer = L;
    g_i8g = 1;
    g_a4_gen.fetch_add(1, std::memory_order_acq_rel);
    run(task_attn4s);
    g_i8g = I8_G;
}

static void attn4p_run(int L) {
    g_pfa_layer = L;
    g_i8g = 1;
    g_a4_gen.fetch_add(1, std::memory_order_acq_rel);
    run(task_attn4_a);
    g_i8g = I8_G;
}

// X100 worker: no /proc/set_ai_thread (that would pin it to the A cluster), so
// it stays at VLEN=256 and runs the 8-position softmax shape.
static int bind_cpu_x100(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static void * sm_worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    bind_cpu_x100(id);
    g_ready[8 + id].store(1, std::memory_order_release);
    int gen = 0, xgen = 0, egen = 0, agen = 0;
    for (;;) {
        // Four kinds of work share these eight threads: elementwise bands
        // (norm, quantiser, silu, KV conversion), softmax for the pipelined
        // attention, softmax for the four-row decode attention, and the column
        // share of a prefill GEMM. The caller finishes each kind before starting
        // the next, so the order here only has to be consistent, not clever.
        const int e = g_el_gen.load(std::memory_order_acquire);
        if (e != egen) {
            egen = e;
            g_el_fn(8 + id);
            g_el_done.fetch_add(1, std::memory_order_release);
            continue;
        }
        // Four-row softmax is addressed by generation counters, not the tile
        // machinery: a decode layer has exactly sixteen head-tiles and the caller
        // waits on them directly, so there is nothing to claim.
        const int a = g_a4_gen.load(std::memory_order_acquire);
        if (a != agen) {
            agen = a;
            task_attn4_sm(id);
            g_a4_done.fetch_add(1, std::memory_order_release);
            continue;
        }
        while (g_pipe_gen.load(std::memory_order_acquire) == gen &&
               g_xg_gen.load(std::memory_order_acquire) == xgen &&
               g_a4_gen.load(std::memory_order_acquire) == agen) {
            if (g_quit.load(std::memory_order_acquire)) return nullptr;
            if (g_el_gen.load(std::memory_order_acquire) != egen) break;
            sched_yield();
        }
        if (g_el_gen.load(std::memory_order_acquire) != egen) continue;
        if (g_a4_gen.load(std::memory_order_acquire) != agen) continue;
        const int x = g_xg_gen.load(std::memory_order_acquire);
        if (x != xgen) {
            xgen = x;
            task_pf_gemm_x100(id);
            g_xg_done.fetch_add(1, std::memory_order_release);
            continue;
        }
        gen = g_pipe_gen.load(std::memory_order_acquire);
        task_pfa_pipe_B(id);
        g_sm_done.fetch_add(1, std::memory_order_release);
    }
}

static void pfa_run(int L) {
    g_pfa_layer = L;
    if (!g_pfa_i8) { run(task_pfa); return; }
    // Four rows is the decode shape and it has no tiles: one group per head, so it
    // bypasses the tile machinery entirely (g_tile_n would be zero).
    if (g_pfa_rows <= PFA_ROWS) {
        g_i8g = 1;
        if (g_a4s == 2 && g_w4) a4s_run(L);
        else if (g_a4s == 1 && g_o4part && g_n_sm == 8) a4s_run(L);
        else if (g_a4p && g_sc4 && g_n_sm == 8) attn4p_run(L);
        else run(task_pfa_i8_4);
        g_i8g = I8_G;
        return;
    }
    if (!g_pipe)   { run(task_pfa_i8); return; }
    pfa_pipe_alloc();   // keeps g_tile_n in step with g_pfa_rows
    for (int64_t t = 0; t < g_tile_n; ++t) g_tile[t].st.store(0, std::memory_order_relaxed);
    g_qk_next.store(0, std::memory_order_relaxed);
    g_sm_next.store(0, std::memory_order_relaxed);
    g_sm_done.store(0, std::memory_order_relaxed);
    g_pipe_layer = L;
    g_pipe_gen.fetch_add(1, std::memory_order_release);
    run(task_pfa_pipe_A);
    while (g_sm_done.load(std::memory_order_acquire) < g_n_sm) sched_yield();
}

// ------------------------------------------------------ four-row decode round
// A speculation round is four rows through one weight stream. Every piece is
// already proven somewhere else: the GEMM is the hp_m4 kernel with prefill's A
// record, the attention is what attn4p_run times at 10K, and the int8 K/V planes
// are the ones prefill filled for the whole prompt. What is new is that the four
// rows advance position by position instead of chunk by chunk.
static float * g_x4 = nullptr;      // 4 x N_EMBD residual stream
static float * g_xa4 = nullptr;     // 4 x N_EMBD: normed in, GEMM out
static float * g_q4 = nullptr;      // 4 x N_EMBD; also the packed Q attention reads
static float * g_k4 = nullptr;      // 4 x N_KV*HEAD_DIM
static float * g_v4 = nullptr;
static float * g_gate4 = nullptr;   // 4 x N_FF
static float * g_up4 = nullptr;
static float * g_logits4 = nullptr; // 4 x VOCAB
static int g_m4dec = 0;             // -MEGA_M4DEC=n: rounds of four rows
static int g_m4spec = 0;            // -MEGA_SPEC=n: rounds with the draft model
static int g_m4spec_k = 3;          // drafts per round (4-row verification)
static int64_t g_m4p0 = -1;         // MEGA_M4P0: aim the round at a prompt position
static int32_t g_m4tok[4] = { 0, 0, 0, 0 };
// The four answers, file-scope rather than a caller-stack array: m4_round inlines
// a lot and a caller-stack array came back with stale words after the round.
static int g_m4out[4];

static void m4dec_alloc(void) {
    if (g_x4) return;
    const size_t xe = (size_t) 4 * N_EMBD * 4 + (2 << 20);
    const size_t fe = (size_t) 4 * N_FF * 4 + (2 << 20);
    const size_t ke = (size_t) 4 * N_KV * HEAD_DIM * 4 + 4096;
    g_x4      = (float *) hugemem(xe);
    g_xa4     = (float *) hugemem(xe);
    g_q4      = (float *) hugemem(xe);
    g_gate4   = (float *) hugemem(fe);
    g_up4     = (float *) hugemem(fe);
    g_k4      = (float *) hugemem(ke);
    g_v4      = (float *) hugemem(ke);
    g_logits4 = (float *) hugemem((size_t) 4 * VOCAB * 4 + (2 << 20));
    if (!g_x4 || !g_xa4 || !g_q4 || !g_gate4 || !g_up4 || !g_k4 || !g_v4 ||
        !g_logits4) {
        fprintf(stderr, "oom four-row decode\n");
        g_x4 = nullptr;
        return;
    }
    m4_alloc();
    if (g_a4p) attn4p_alloc();
    if (g_a4s) a4s_alloc();
    printf("m4dec: %.1f MB scratch, four rows per round\n",
           (double) (4 * N_EMBD * 2 + 4 * N_FF * 2 + 4 * VOCAB) * 4 / 1e6);
}

// Append one layer's new positions to the int8 K/V planes. The prefix was
// quantised over the whole prompt with one scale per layer per kv, so a new
// position is another sample on that grid and nothing already written moves.
// K: one position's 16 dims fill one lane of a 64-position block, so a 16-lane
// conversion is exactly one position. V: a 16-position block interleaves
// positions, so a single new position is a byte store per dim.
static void conv_kv8_inc(int layer, int64_t p0, int nrows) {
    for (int kv = 0; kv < N_KV; ++kv) {
        const float sk = g_k8i_sc[layer * N_KV + kv];
        const float sv = g_v8i_sc[layer * N_KV + kv];
        const float ik = sk > 0 ? 1.0f / sk : 1.0f;
        const float iv = sv > 0 ? 1.0f / sv : 1.0f;
        int8_t * vbase = v8i_base(layer) + (int64_t) kv * g_v8i_npb16 * 2048;
        for (int64_t p = p0; p < p0 + nrows; ++p) {
            const _Float16 * kb = ktile(layer, kv, (int) (p / KT));
            int8_t * kdst = k8i_base(layer) + ((int64_t) (kv * g_k8i_npb + p / 64) * 8) * 1024
                          + (p % 64) * 16;
            for (int dg = 0; dg < HEAD_DIM / 16; ++dg)
                conv16(kb + dg * 16 * KT + (int) (p % KT), KT, ik, kdst + dg * 1024);
            const _Float16 * vb = g_vpos + (int64_t) layer * KV_LAYER_STRIDE
                                + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
            int8_t * vdst = vbase + (p / 16) * 1024 + (p % 16);
            for (int d = 0; d < HEAD_DIM; ++d) {
                int b = (int) lrintf((float) vb[d] * iv);
                if (b > 127) b = 127;
                if (b < -127) b = -127;
                vdst[((d / 64) * g_v8i_npb16) * 1024 + (d % 64) * 16] = (int8_t) b;
            }
        }
    }
}

static void silu4_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i));
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(g_gate4 + i, vl);
        const vfloat32m1_t u = __riscv_vle32_v_f32m1(g_up4 + i, vl);
        vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, -1.4426950408889634f, vl);
        t = __riscv_vfmax_vf_f32m1(t, -126.0f, vl);
        t = __riscv_vfmin_vf_f32m1(t,  126.0f, vl);
        const vfloat32m1_t den = __riscv_vfadd_vf_f32m1(exp2f_e(t, vl), 1.0f, vl);
        __riscv_vse32_v_f32m1(g_gate4 + i,
            __riscv_vfmul_vv_f32m1(__riscv_vfdiv_vv_f32m1(x, den, vl), u, vl), vl);
    }
}

static void add4_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i < 32 ? hi - i : 32));
        const vfloat32m1_t a = __riscv_vle32_v_f32m1(g_x4 + i, vl);
        const vfloat32m1_t b = __riscv_vle32_v_f32m1(g_xa4 + i, vl);
        __riscv_vse32_v_f32m1(g_x4 + i, __riscv_vfadd_vv_f32m1(a, b, vl), vl);
    }
}

// One round: four candidate tokens at positions p0..p0+3, one argmax per row in
// out[]. Row 0's answer is the real next token; rows 1..3 are what the model would
// predict if the drafts before them had been right, which is what the acceptance
// test consumes.
static const bool M4DUMP = getenv("MEGA_M4DUMP") != nullptr;
static const bool M4TIME = getenv("MEGA_M4TIME") != nullptr;
static double g_m4t[8];   // gemm, attn, norm, silu, rope+kv, quant, argmax, other
static void m4_report(void) {
    static const char * nm[8] = { "gemm", "attn", "norm", "silu", "rope+kv",
                                 "quant", "argmax", "other" };
    printf("m4dec: ");
    for (int i = 0; i < 8; ++i) printf("%s %.2f  ", nm[i], g_m4t[i] * 1e3);
    printf("(ms)\n");
    for (int i = 0; i < 8; ++i) g_m4t[i] = 0;
}
static const bool M4ARGMAX_ONE = getenv("MEGA_M4AX2") != nullptr;
// Four-row argmax: one band per thread over 4 x VOCAB, four running maxima each.
static int   g_m4b_i[N_GPU][4];
static float g_m4b_v[N_GPU][4];
static void task_argmax4_init(int tid) {
    for (int r = 0; r < 4; ++r) { g_m4b_i[tid][r] = -1; g_m4b_v[tid][r] = -INFINITY; }
}
static void argmax4_fn(int64_t lo, int64_t hi) {
    const int t = g_pf_tid;
    for (int64_t i = lo; i < hi; ++i) {
        const int r = (int) (i / VOCAB);
        const float v = g_logits4[i];
        if (v > g_m4b_v[t][r] || (v == g_m4b_v[t][r] && g_m4b_i[t][r] < 0)) {
            g_m4b_v[t][r] = v;
            g_m4b_i[t][r] = (int) (i - (int64_t) r * VOCAB);
        }
    }
}
static void dump_row(const char * tag, int layer, char when, const float * row);
static void dump_n(const char * tag, int layer, char when, const float * row, int64_t n);

// One row per thread from the pool, no internal barrier. Four rows through
// rmsnorm_par costs four barriers per call and 168 calls per round.
static void rmsnorm_rows(float * dst, const float * src, const float * w, int64_t rows);
// ============================================================== dspark drafter
// MiniCPM5 2.6B DSpark: a 5-layer head over the target, 324M parameters at Q4_0.
// DSPARK.md has the architecture and the cost model; the forward pass below is the
// one mega_decode.cpp validated against llama.cpp's dflash/dspark code.
//
// The drafter cannot make the output wrong. The verifier only emits a draft token
// after the target has reproduced it, so a bad drafter costs speed and nothing
// else -- which is what makes it safe to measure this path before optimising it.
//
// Only the first four slots of the block run. The reference block has seven and
// they do not see each other (the block_uniform mask makes every slot reuse slot
// 0's causal row), so slots 0..2 give the three drafts a four-row verification
// needs, and four rows is the shape the target's IME2 kernel is built for.
static const int DS_LAYER = 5;
static const int DS_SLOT = 4;
static int g_ds_layer;
static void ds_add_band(int64_t lo, int64_t hi);
static void ds_silu_band(int64_t lo, int64_t hi);
static bool m4_round(int64_t p0, const int * toks, int * out);
static const int DS_TARGET_LAYER[DS_LAYER] = { 2, 11, 21, 31, 40 };
static const int DS_RANK = 256;

struct DRef { uint8_t * b; int64_t k, n; };

static uint8_t * g_dsmap; static size_t g_dsmap_sz;
static uint8_t * g_dsblob;
static DRef g_ds_wq[DS_LAYER], g_ds_wk[DS_LAYER], g_ds_wv[DS_LAYER], g_ds_wo[DS_LAYER];
static DRef g_ds_wg[DS_LAYER], g_ds_wu[DS_LAYER], g_ds_wd[DS_LAYER];
static DRef g_ds_fc, g_ds_w2;
static const uint8_t * g_ds_w1;
static const float * g_ds_attn_norm[DS_LAYER], * g_ds_ffn_norm[DS_LAYER];
static const float * g_ds_qnorm[DS_LAYER], * g_ds_knorm[DS_LAYER];
static const float * g_ds_enc_norm, * g_ds_out_norm;
// The draft's own cache, position-major per layer: [layer][pos][kv][dim], fp16.
static _Float16 * g_dskc, * g_dsvc;
// The capture has to hold one whole prefill chunk: a chunk injects every four-row
// group it produced, so all of the chunk's rows are still needed at the end of it.
static int64_t g_ds_cap = 4;
static float * g_dshid;              // [layer][row][N_EMBD]: the target's residual
static float * g_dsx, * g_dsxa, * g_dsq, * g_dsk, * g_dsv, * g_dsgate, * g_dsup;
static float * g_dslog, * g_dsbias, * g_dsfeat, * g_dsatt, * g_dsenc;
static float * g_dssc;               // [thread][MAXCTX] scores
static int g_ds = 0, g_ds_ready = 0, g_ds_hidrow = 0;
static const bool g_dscos = getenv("MEGA_DSCOS") != nullptr;
static const bool g_dsdbg2 = getenv("MEGA_DSDBG2") != nullptr;
static int g_ds_pending = 0;         // slots of hidden state waiting to be injected
static int64_t g_ds_pend_p0 = 0;
static double g_ds_t_inj, g_ds_t_dec;
static long long g_ds_drafts = 0, g_ds_acc = 0;

static const uint8_t * ds_skip_val(const uint8_t * p, uint32_t t) {
    static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (t == 8) { uint64_t n; memcpy(&n, p, 8); return p + 8 + n; }
    if (t == 9) {
        uint32_t at; memcpy(&at, p, 4); p += 4;
        uint64_t n; memcpy(&n, p, 8); p += 8;
        for (uint64_t i = 0; i < n; ++i) p = ds_skip_val(p, at);
        return p;
    }
    return p + sz[t < 13 ? t : 0];
}

struct DsInfo { char name[64]; const uint8_t * data; int64_t ne0, ne1; uint32_t type; };
static DsInfo g_dst[96];
static int g_dsn;

static void ds_parse(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "dspark: cannot open %s\n", path); return; }
    fseek(f, 0, SEEK_END); g_dsmap_sz = (size_t) ftell(f); fseek(f, 0, SEEK_SET);
    g_dsmap = (uint8_t *) malloc(g_dsmap_sz);
    if (!g_dsmap || fread(g_dsmap, 1, g_dsmap_sz, f) != g_dsmap_sz) {
        fprintf(stderr, "dspark: read failed\n"); fclose(f); return;
    }
    fclose(f);
    const uint8_t * p = g_dsmap;
    uint32_t magic; memcpy(&magic, p, 4);
    if (magic != 0x46554747u) { fprintf(stderr, "dspark: bad magic\n"); return; }
    uint64_t nt, nkv;
    memcpy(&nt, p + 8, 8); memcpy(&nkv, p + 16, 8);
    p += 24;
    for (uint64_t i = 0; i < nkv; ++i) {
        uint64_t len; memcpy(&len, p, 8); p += 8 + len;
        uint32_t t; memcpy(&t, p, 4); p += 4;
        p = ds_skip_val(p, t);
    }
    struct Raw { char name[64]; uint64_t off; int64_t ne0, ne1; uint32_t type; };
    static Raw raw[96];
    for (uint64_t i = 0; i < nt && g_dsn < 96; ++i) {
        uint64_t len; memcpy(&len, p, 8); p += 8;
        const size_t c = len < 63 ? (size_t) len : 63;
        memcpy(raw[g_dsn].name, p, c); raw[g_dsn].name[c] = 0; p += len;
        uint32_t nd; memcpy(&nd, p, 4); p += 4;
        int64_t dims[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        memcpy(dims, p, 8 * nd); p += 8 * nd;
        uint32_t ty; memcpy(&ty, p, 4); p += 4;
        uint64_t off; memcpy(&off, p, 8); p += 8;
        raw[g_dsn].off = off; raw[g_dsn].type = ty;
        raw[g_dsn].ne0 = dims[0]; raw[g_dsn].ne1 = dims[1];
        ++g_dsn;
    }
    const size_t hdr = ((size_t) (p - g_dsmap) + 31) & ~(size_t) 31;
    for (int i = 0; i < g_dsn; ++i) {
        snprintf(g_dst[i].name, sizeof g_dst[i].name, "%s", raw[i].name);
        g_dst[i].data = g_dsmap + hdr + raw[i].off;
        g_dst[i].ne0 = raw[i].ne0; g_dst[i].ne1 = raw[i].ne1;
        g_dst[i].type = raw[i].type;
    }
    printf("dspark: %d tensors, %.1f MB\n", g_dsn, g_dsmap_sz / 1e6);
}

static const DsInfo * ds_find(const char * name) {
    for (int i = 0; i < g_dsn; ++i)
        if (!strcmp(g_dst[i].name, name)) return &g_dst[i];
    fprintf(stderr, "dspark: missing tensor %s\n", name);
    return nullptr;
}

// Q4_0 rows into the 576-byte group layout the hp kernel reads.
static void ds_repack_b(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k) {
    const int64_t nb = k / 32, rs = nb * 18;
    uint8_t * d = dst;
    for (int64_t b = 0; b < rows; b += 32)
        for (int64_t x = 0; x < nb; ++x) {
            _Float16 * dd = (_Float16 *) d;
            uint8_t * qs = d + 64;
            for (int i = 0; i < 32; ++i) {
                const uint8_t * sp = src + (b + i) * rs + x * 18;
                _Float16 sc; memcpy(&sc, sp, 2);
                dd[i] = sc;
                const uint8_t * sq = sp + 2;
                uint8_t * dq = qs + i * 16;
                for (int j = 0; j < 8; ++j) {
                    dq[j]     = (uint8_t) ((sq[2 * j] & 0x0F) | ((sq[2 * j + 1] & 0x0F) << 4));
                    dq[8 + j] = (uint8_t) ((sq[2 * j] >> 4)  | ((sq[2 * j + 1] & 0xF0)));
                }
            }
            d += 576;
        }
}

static bool ds_load(void) {
    const char * path = getenv("MEGA_DSPARK");
    if (!path) path = "/home/bianbu/workspace/mega/dspark-q4.gguf";
    ds_parse(path);
    if (!g_dsmap) return false;

    size_t need = 0;
    for (int i = 0; i < g_dsn; ++i) {
        if (g_dst[i].type != 2) continue;
        need += (size_t) (g_dst[i].ne1 / 32) * (g_dst[i].ne0 / 32) * 576;
    }
    const size_t blob = (need + (1 << 21)) & ~((size_t) (1 << 21) - 1);
    g_dsblob = (uint8_t *) hugemem(blob);
    if (!g_dsblob) { fprintf(stderr, "dspark: oom\n"); return false; }
    use_huge(g_dsblob, blob);

    char nm[64];
    size_t off = 0;
    auto take = [&](DRef & r, const char * n) {
        const DsInfo * t = ds_find(n);
        if (!t || t->type != 2) return;
        ds_repack_b(g_dsblob + off, t->data, t->ne1, t->ne0);
        r.b = g_dsblob + off; r.k = t->ne0; r.n = t->ne1;
        off += (size_t) (t->ne1 / 32) * (t->ne0 / 32) * 576;
    };
    auto f32 = [&](const float ** d, const char * n) {
        const DsInfo * t = ds_find(n);
        if (t) *d = (const float *) t->data;
    };
    take(g_ds_fc, "fc.weight");
    take(g_ds_w2, "markov_w2.weight");
    const DsInfo * w1 = ds_find("markov_w1.weight");
    if (!w1) return false;
    g_ds_w1 = w1->data;
    // The two 2048-wide norms are easy to swap; MEGA_DSSWAPNORM tests the other
    // assignment when acceptance is the thing being measured.
    if (getenv("MEGA_DSSWAPNORM")) {
        f32(&g_ds_enc_norm, "output_norm.weight");
        f32(&g_ds_out_norm, "enc.output_norm.weight");
    } else {
        f32(&g_ds_enc_norm, "enc.output_norm.weight");
        if (!g_ds_enc_norm) f32(&g_ds_enc_norm, "output_norm_enc.weight");
    }
    if (!g_ds_out_norm) f32(&g_ds_out_norm, "output_norm.weight");
    for (int L = 0; L < DS_LAYER; ++L) {
        snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", L);      take(g_ds_wq[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", L);      take(g_ds_wk[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", L);      take(g_ds_wv[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", L); take(g_ds_wo[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", L);    take(g_ds_wg[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", L);      take(g_ds_wu[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", L);    take(g_ds_wd[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", L);   f32(&g_ds_attn_norm[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", L);    f32(&g_ds_ffn_norm[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", L); f32(&g_ds_qnorm[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", L); f32(&g_ds_knorm[L], nm);
    }
    if (!g_ds_fc.b || !g_ds_w2.b || !g_ds_enc_norm || !g_ds_out_norm) {
        fprintf(stderr, "dspark: incomplete weights\n"); return false;
    }
    printf("dspark: repacked %.1f MB, fc %lldx%lld, w2 %lldx%lld, w1 %lldx%lld\n", off / 1e6,
           (long long) g_ds_fc.n, (long long) g_ds_fc.k,
           (long long) g_ds_w2.n, (long long) g_ds_w2.k,
           (long long) w1->ne0, (long long) w1->ne1);
    return true;
}

static bool ds_alloc(void) {
    if (g_dskc) return true;
    // The drafter's GEMMs go through the four-row kernel, so its A-record arena has
    // to exist before prefill runs, not when the first decode round allocates it.
    m4_alloc();
    if (!g_aq4d) { fprintf(stderr, "dspark: oom A records\n"); return false; }
    const size_t kvn = (size_t) DS_LAYER * MAXCTX * N_KV * HEAD_DIM * sizeof(_Float16);
    g_ds_cap = g_pf_chunk > 4 ? g_pf_chunk : 4;
    const size_t hn  = (size_t) DS_LAYER * g_ds_cap * N_EMBD * 4;
    const size_t xn  = (size_t) DS_SLOT * N_EMBD * 4;
    const size_t fn  = (size_t) DS_SLOT * N_FF * 4;
    const size_t kn  = (size_t) DS_SLOT * N_KV * HEAD_DIM * 4;
    g_dskc   = (_Float16 *) hugemem(kvn + (2 << 20));
    g_dsvc   = (_Float16 *) hugemem(kvn + (2 << 20));
    g_dshid  = (float *) hugemem(hn + (2 << 20));
    g_dsx    = (float *) hugemem(xn + (2 << 20));
    g_dsxa   = (float *) hugemem(xn + (2 << 20));
    g_dsq    = (float *) hugemem(xn + (2 << 20));
    g_dsatt  = (float *) hugemem(xn + (2 << 20));
    g_dsk    = (float *) hugemem(kn + (2 << 20));
    g_dsv    = (float *) hugemem(kn + (2 << 20));
    g_dsgate = (float *) hugemem(fn + (2 << 20));
    g_dsup   = (float *) hugemem(fn + (2 << 20));
    g_dsfeat = (float *) hugemem((size_t) DS_SLOT * DS_IN_K * 4 + (2 << 20));
    g_dsenc  = (float *) hugemem((size_t) DS_SLOT * N_EMBD * 4 + (2 << 20));
    g_dslog  = (float *) hugemem((size_t) DS_SLOT * VOCAB * 4 + (2 << 20));
    // four rows of bias, because the four-row kernel writes ldc = n per row and the
    // markov head runs through that same kernel to stay on one code path
    g_dsbias = (float *) hugemem((size_t) DS_SLOT * VOCAB * 4 + (2 << 20));
    // four slots' score rows per thread: the four-row draft pass shares one K stream
    g_dssc   = (float *) hugemem((size_t) N_GPU * DS_SLOT * MAXCTX * 4 + (2 << 20));
    if (!g_dskc || !g_dsvc || !g_dshid || !g_dsx || !g_dsxa || !g_dsq || !g_dsatt ||
        !g_dsk || !g_dsv || !g_dsgate || !g_dsup || !g_dsfeat || !g_dsenc ||
        !g_dslog || !g_dsbias || !g_dssc) {
        fprintf(stderr, "dspark: oom scratch\n"); return false;
    }
    printf("dspark: %.1f MB draft KV x2, %.1f MB scratch\n",
           (double) kvn / 1e6, (double) (hn + 4 * xn + 2 * fn + 2 * kn) / 1e6);
    return true;
}

// Per-head RMSNorm, 128 wide. Scalar on purpose: 18 of these per slot per layer.
static void ds_rmsvec(float * dst, const float * src, const float * w, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) src[i] * src[i];
    const float r = 1.0f / sqrtf((float) (ss / n) + RMS_EPS);
    for (int i = 0; i < n; ++i) dst[i] = src[i] * r * w[i];
}

// Pairwise (2i, 2i+1) rotation. The target uses the same convention
// (LLAMA_ROPE_TYPE_NORM, see rope_head_norm), which is why the halves-rotate form
// `MEGA_DSROPE=half` used to test 0.111 accepted per round against 0.278: it was
// not a different valid layout, it was the wrong one on both sides.
static int g_ds_rope_half = -1;
static void ds_rope(float * x, int pos) {
    if (g_ds_rope_half < 0) {
        const char * e = getenv("MEGA_DSROPE");
        g_ds_rope_half = (e && !strcmp(e, "half")) ? 1 : 0;
    }
    if (g_ds_rope_half) {
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            const float th = (float) pos * (1.0f / powf(ROPE_BASE, (float) (2 * i) / (float) HEAD_DIM));
            const float c = cosf(th), sn = sinf(th);
            const float x0 = x[i], x1 = x[i + HEAD_DIM / 2];
            x[i]                = x0 * c - x1 * sn;
            x[i + HEAD_DIM / 2] = x0 * sn + x1 * c;
        }
        return;
    }
    for (int i = 0; i < HEAD_DIM / 2; ++i) {
        const float th = (float) pos * (1.0f / powf(ROPE_BASE, (float) (2 * i) / (float) HEAD_DIM));
        const float c = cosf(th), sn = sinf(th);
        const float x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i]     = x0 * c - x1 * sn;
        x[2 * i + 1] = x0 * sn + x1 * c;
    }
}

// The encoder: the target's residual stream at five layers fused into one feature,
// then that feature projected to this layer's K and V and appended to the draft's
// cache. This is how the draft sees the committed prefix -- not as tokens.
//
// Four positions at a time so it runs through the same four-row IME2 kernel as
// verification. Prefill calls it once per 64-row chunk, which is where the ~1.7 s
// it adds to a 10K prefill goes.
static void ds_inject4(int64_t p0, int64_t row0, int n) {
    if (n <= 0) return;
    // The fused vector's layout is layer-major unless MEGA_DSINTER says otherwise:
    // if the export interleaves the five layers instead, the fc would be reading a
    // permuted vector and every draft would be a plausible-looking wrong token.
    static const bool interleave = getenv("MEGA_DSINTER") != nullptr;
    for (int r = 0; r < n; ++r) {
        float * dst = g_dsfeat + (int64_t) r * DS_IN_K;
        if (!interleave) {
            for (int L = 0; L < DS_LAYER; ++L)
                memcpy(dst + (int64_t) L * N_EMBD,
                       g_dshid + ((int64_t) L * g_ds_cap + row0 + r) * N_EMBD,
                       (size_t) N_EMBD * 4);
        } else {
            for (int j = 0; j < N_EMBD; ++j)
                for (int L = 0; L < DS_LAYER; ++L)
                    dst[(int64_t) j * DS_LAYER + L] =
                        g_dshid[((int64_t) L * g_ds_cap + row0 + r) * N_EMBD + j];
        }
    }
    for (int r = n; r < DS_SLOT; ++r)
        memset(g_dsfeat + (int64_t) r * DS_IN_K, 0, (size_t) DS_IN_K * 4);
    if (getenv("MEGA_DSZERO")) memset(g_dsfeat, 0, (size_t) DS_SLOT * DS_IN_K * 4);

    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsfeat, (size_t) DS_IN_K, m4_aq2(DS_IN_K));
    g_mjn = 1; g_mj[0] = { g_ds_fc.b, g_ds_fc.k, g_ds_fc.n, g_dsenc };
    run(task_mgemm_m4);
    rmsnorm_rows(g_dsenc, g_dsenc, g_ds_enc_norm, DS_SLOT);

    for (int L = 0; L < DS_LAYER; ++L) {
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsenc, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 2;
        // ldc is the matrix width, so each projection needs its own output buffer.
        g_mj[0] = { g_ds_wk[L].b, g_ds_wk[L].k, g_ds_wk[L].n, g_dsk };
        g_mj[1] = { g_ds_wv[L].b, g_ds_wv[L].k, g_ds_wv[L].n, g_dsv };
        run(task_mgemm_m4);
        for (int r = 0; r < n; ++r) {
            const int pos = (int) (p0 + r);
            for (int h = 0; h < N_KV; ++h) {
                float * k = g_dsk + ((int64_t) r * N_KV + h) * HEAD_DIM;
                ds_rmsvec(k, k, g_ds_knorm[L], HEAD_DIM);
                ds_rope(k, pos);
            }
            if (g_dsdbg2 && L == 0) {
                double a = 0, b = 0, c = 0;
                for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
                    a += (double) g_dsk[i] * g_dsk[i];
                    b += (double) g_dsv[i] * g_dsv[i];
                    c += (double) g_dsenc[(int64_t) r * N_EMBD + i] * g_dsenc[(int64_t) r * N_EMBD + i];
                }
                fprintf(stderr, "  [inj] pos=%d L0 |K|=%.3f |V|=%.3f |enc|_part=%.3f\n",
                        pos, sqrt(a), sqrt(b), sqrt(c));
            }
            _Float16 * kc = g_dskc + (int64_t) L * KV_LAYER_STRIDE + (int64_t) pos * (N_KV * HEAD_DIM);
            _Float16 * vc = g_dsvc + (int64_t) L * KV_LAYER_STRIDE + (int64_t) pos * (N_KV * HEAD_DIM);
            for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
                kc[i] = (_Float16) g_dsk[(int64_t) r * (N_KV * HEAD_DIM) + i];
                vc[i] = (_Float16) g_dsv[(int64_t) r * (N_KV * HEAD_DIM) + i];
            }
        }
    }
}

// The target's residual stream at the five layers the encoder reads. Captured
// inside the layer loops of both the prefill chunk and the four-row round.
// The capture site sits after the layer body, so what it holds is layer L's output
// -- the input to layer L+1. llama.cpp's t_layer_inp[il] is layer il's *input*, so
// the metadata index has to be shifted by one to mean the same tensor. MEGA_DSSL
// overrides the shift outright so the alternatives can be scored rather than argued.
static int g_ds_lsh = -999;
static inline int ds_target_slot(int L) {
    if (g_ds_lsh == -999) {
        const char * e = getenv("MEGA_DSSL");
        if (e) g_ds_lsh = atoi(e);
        else g_ds_lsh = getenv("MEGA_DSL1") ? 1 : 0;
    }
    for (int k = 0; k < DS_LAYER; ++k)
        if (DS_TARGET_LAYER[k] - g_ds_lsh == L) return k;
    return -1;
}

// Block attention: every slot sees the prefix plus slot 0, and nothing else. The
// scores are plain fp32 with vectorised dots -- 640K of them per layer at 10K, which
// the pool spreads over 64 (slot, head) units. An int8 path like the target's would
// make this ~4x cheaper, but acceptance does not depend on speed, so it waits until
// there is a measured acceptance rate that justifies it.
static int g_ds_np;
static const bool g_ds_attn4 = getenv("MEGA_DSATTN4") != nullptr;
// MEGA_DSWIN=N: the drafter attends the last N prefix positions. Zero is the full
// prefix. Default 256, measured over 300 rounds so the number is a stream average
// and not a stretch of one: 10K round 133.60 ms with 0.617 accepted per round
// (21.33 tok/s) against 135.81 / 0.538 (19.24) at 512, 144.25 / 0.179 (10.65) at
// 1024 and 305.4 / 0.050 (3.77) at the full prefix. At 2K the same two are 110.88
// ms / 0.929 (34.15 tok/s) and 113.48 / 0.916 (33.02).
// Acceptance rising as the window shrinks is the interesting half: a 324M drafter
// reads a recent window better than it reads ten thousand positions. Do not push
// it below 256 on the strength of the 10K numbers alone: at 128 and 32 it reads
// 0.804 and 0.922 per round only because this prompt's greedy stream falls into a
// `944 242 39` cycle, and a drafter that sees 32 tokens can replay a cycle it
// cannot generalise from.
static int g_ds_win = -1;
static inline int ds_win(void) {
    if (g_ds_win < 0) { const char * e = getenv("MEGA_DSWIN"); g_ds_win = e ? atoi(e) : 256; }
    return g_ds_win;
}

// One unit per (kv head, query head), all four slots inside it. The four slots ask
// their questions of the same prefix, so walking positions once and fanning the
// loaded K and V out to four accumulators cuts the stream fourfold -- and the
// stream is what this loop was spending its time on (4.3 GB/s measured against
// 17.5 available, one full pass over 164 MB of K per layer).
static void ds_attn4_band(int64_t lo, int64_t hi) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    const size_t vl16 = __riscv_vsetvl_e16mf2(32);
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    for (int64_t t = lo; t < hi; ++t) {
        // one unit per query head; the kv head follows the same 8:1 grouping the
        // target uses, which is what makes the four slots share a stream at all
        const int h = (int) t;
        const int kv = h / NH_PER_KV;
        const _Float16 * kb = g_dskc + (int64_t) g_ds_layer * KV_LAYER_STRIDE
                            + (int64_t) kv * HEAD_DIM;
        const _Float16 * vb = g_dsvc + (int64_t) g_ds_layer * KV_LAYER_STRIDE
                            + (int64_t) kv * HEAD_DIM;
        const int64_t kstride = N_KV * HEAD_DIM;
        // sc[r] is this slot's score row; the four are adjacent so the hot one stays
        // in cache while the other three are written.
        float * sc0 = g_dssc + (int64_t) g_pf_tid * DS_SLOT * MAXCTX;
        float * q0 = g_dsq + h * HEAD_DIM;
        float * o0 = g_dsatt + h * HEAD_DIM;

        float mx[DS_SLOT];
        for (int r = 0; r < DS_SLOT; ++r) mx[r] = -INFINITY;

        for (int64_t i = 0; i < g_ds_np; ++i) {
            const _Float16 * kp = kb + i * kstride;
            for (int d = 0; d < HEAD_DIM; d += 32) {
                const vfloat32m1_t kd =
                    __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kp + d, vl16), vl);
                for (int r = 0; r < DS_SLOT; ++r) {
                    const vfloat32m1_t qd = __riscv_vle32_v_f32m1(
                        q0 + (int64_t) r * N_EMBD + d, vl);
                    const vfloat32m1_t acc = __riscv_vfmul_vv_f32m1(kd, qd, vl);
                    const vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
                    const float v = __riscv_vfmv_f_s_f32m1_f32(
                        __riscv_vfredusum_vs_f32m1_f32m1(acc, z, vl)) * scale;
                    if (d == 0) sc0[(int64_t) r * MAXCTX + i] = v;
                    else        sc0[(int64_t) r * MAXCTX + i] += v;
                }
            }
            for (int r = 0; r < DS_SLOT; ++r) {
                const float v = sc0[(int64_t) r * MAXCTX + i];
                if (v > mx[r]) mx[r] = v;
            }
        }

        float sum[DS_SLOT];
        for (int r = 0; r < DS_SLOT; ++r) {
            float * sc = sc0 + (int64_t) r * MAXCTX;
            float sm = 0.0f;
            for (int64_t i = 0; i < g_ds_np; ++i) {
                const float e = expf(sc[i] - mx[r]);
                sc[i] = e; sm += e;
            }
            sum[r] = sm > 0.0f ? 1.0f / sm : 0.0f;
        }

        // PV over the same stream: one V load feeds four weighted accumulations.
        for (int r = 0; r < DS_SLOT; ++r)
            for (int d = 0; d < HEAD_DIM; d += 32)
                __riscv_vse32_v_f32m1(o0 + (int64_t) r * N_EMBD + d,
                                      __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        for (int64_t i = 0; i < g_ds_np; ++i) {
            const _Float16 * vp = vb + i * kstride;
            for (int d = 0; d < HEAD_DIM; d += 32) {
                const vfloat32m1_t vd =
                    __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp + d, vl16), vl);
                for (int r = 0; r < DS_SLOT; ++r) {
                    const float w = sc0[(int64_t) r * MAXCTX + i] * sum[r];
                    float * op = o0 + (int64_t) r * N_EMBD + d;
                    __riscv_vse32_v_f32m1(op, __riscv_vfmacc_vf_f32m1(
                        __riscv_vle32_v_f32m1(op, vl), w, vd, vl), vl);
                }
            }
        }
    }
}

static void ds_attn_band(int64_t lo, int64_t hi) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    const size_t vl16 = __riscv_vsetvl_e16mf2(32);
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    // The drafter only has to guess, so its attention does not need the whole
    // prefix: at 10K this loop is 195 ms of a 305 ms round, and it is a full walk
    // of K and V per (slot, head) unit with a vector reduction per position. A
    // window of the last W positions makes the cost O(W) and leaves the acceptance
    // to be measured rather than assumed.
    const int64_t n = g_ds_np;
    const int64_t wlo = (ds_win() > 0 && n > ds_win()) ? n - ds_win() : 0;
    for (int64_t t = lo; t < hi; ++t) {
        const int r = (int) (t / N_HEAD), h = (int) (t % N_HEAD);
        const int kv = h / NH_PER_KV;
        const float * q = g_dsq + (int64_t) r * N_EMBD + h * HEAD_DIM;
        float * out = g_dsatt + (int64_t) r * N_EMBD + h * HEAD_DIM;
        const _Float16 * kb = g_dskc + (int64_t) g_ds_layer * KV_LAYER_STRIDE
                            + (int64_t) kv * HEAD_DIM;
        const _Float16 * vb = g_dsvc + (int64_t) g_ds_layer * KV_LAYER_STRIDE
                            + (int64_t) kv * HEAD_DIM;
        const int64_t kstride = N_KV * HEAD_DIM;
        float * sc = g_dssc + (int64_t) g_pf_tid * MAXCTX;

        float mx = -INFINITY;
        for (int64_t i = wlo; i < n; ++i) {
            const _Float16 * kp = kb + i * kstride;
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (int d = 0; d < HEAD_DIM; d += 32)
                acc = __riscv_vfmacc_vv_f32m1(acc,
                    __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kp + d, vl16), vl),
                    __riscv_vle32_v_f32m1(q + d, vl), vl);
            const vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            const float v = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m1_f32m1(acc, z, vl)) * scale;
            sc[i - wlo] = v;
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        const int64_t cnt = n - wlo;
        for (int64_t i = 0; i < cnt; ++i) {
            const float e = expf(sc[i] - mx);
            sc[i] = e; sum += e;
        }
        const float rs = sum > 0.0f ? 1.0f / sum : 0.0f;
        if (g_dsdbg2 && r == 0 && h == 0) {
            // Which position owns the mass, and how big the injected keys are. A
            // single huge score means one key vector is out of scale; that shows up
            // as an argmax pinned to the same position across layers.
            float wmx = -1.0f; int64_t pmx = -1;
            for (int64_t i = 0; i < cnt; ++i) if (sc[i] > wmx) { wmx = sc[i]; pmx = wlo + i; }
            double kn = 0, kn0 = 0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float a = (float) kb[(int64_t) pmx * kstride + d];
                const float b = (float) kb[d];
                kn += (double) a * a; kn0 += (double) b * b;
            }
            fprintf(stderr, "  [attn] L%d np=%d win=%lld w0=%.4f wlast=%.4f max=%.3f "
                    "argmax_p=%lld w=%.4f |K@argmax|=%.3f |K@wlo|=%.3f\n",
                    g_ds_layer, g_ds_np, (long long) (n - wlo), sc[0] * rs, sc[cnt - 1] * rs, mx,
                    (long long) pmx, wmx * rs, sqrt(kn), sqrt(kn0));
        }
        for (int d = 0; d < HEAD_DIM; d += 32)
            __riscv_vse32_v_f32m1(out + d, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        for (int64_t i = 0; i < cnt; ++i) {
            const float w = sc[i] * rs;
            const _Float16 * vp = vb + (wlo + i) * kstride;
            for (int d = 0; d < HEAD_DIM; d += 32)
                __riscv_vse32_v_f32m1(out + d,
                    __riscv_vfmacc_vf_f32m1(__riscv_vle32_v_f32m1(out + d, vl), w,
                        __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp + d, vl16), vl), vl), vl);
        }
    }
}

// The 2.6 MMAC rounding table lives in g_dsbias as the bias for every vocabulary
// entry, so the markov head is one more row of the same GEMM.
static void ds_markov_bias(int prev, float * bias) {
    // The four-row quantizer reads four rows of A, so the rank-256 vector is
    // replicated. Handing it one row made it read 3 KB past the array, which is
    // where the occasional NaN came from.
    alignas(64) static float mk[DS_SLOT * DS_RANK];
    const uint8_t * row = g_ds_w1 + (int64_t) prev * 8 * 18;
    for (int b = 0; b < 8; ++b) {
        _Float16 sc; memcpy(&sc, row + b * 18, 2);
        const uint8_t * qs = row + b * 18 + 2;
        for (int j = 0; j < 16; ++j) {
            mk[b * 32 + j]      = (float) ((int) (qs[j] & 0x0F) - 8) * (float) sc;
            mk[b * 32 + j + 16] = (float) ((int) (qs[j] >> 4) - 8) * (float) sc;
        }
    }
    if (getenv("MEGA_DSTANH")) for (int i = 0; i < DS_RANK; ++i) mk[i] = tanhf(mk[i]);
    for (int r = 1; r < DS_SLOT; ++r) memcpy(mk + r * DS_RANK, mk, (size_t) DS_RANK * 4);
    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, mk, (size_t) DS_RANK, m4_aq2(DS_RANK_K));
    g_mjn = 1; g_mj[0] = { g_ds_w2.b, g_ds_w2.k, g_ds_w2.n, bias };
    run(task_mgemm_m4);      // all four rows are the same bias; callers read row 0
}

// The block: four slots at pos0..pos0+3, drafts out[0..K-1] after the markov chain.
// One-shot dumps for the float32 reference (ref_dspark.py). They capture the exact
// inputs and outputs of the draft forward -- the five captured target residuals,
// then the four logit rows and the markov bias -- so the reference can be fed
// bit-identical features and any mismatch is the draft model's own arithmetic
// rather than a disagreement about what the features were.
static void ds_dump_once(int anchor, int64_t pos0, int npref) {
    if (const char * hp = getenv("MEGA_DSHID")) {
        if (FILE * f = fopen(hp, "wb")) {
            // header: layers, rows, width, then the anchor/pos0/npref triple
            int32_t hdr[6] = { DS_LAYER, npref, N_EMBD, anchor, (int32_t) pos0, npref };
            fwrite(hdr, 4, 6, f);
            for (int L = 0; L < DS_LAYER; ++L)
                fwrite(g_dshid + (int64_t) L * g_ds_cap * N_EMBD, 4, (size_t) npref * N_EMBD, f);
            fclose(f);
            printf("dspark: features -> %s (%d rows x %d)\n", hp, npref, N_EMBD);
        }
    }
}

static void ds_draft(int anchor, int64_t pos0, int K, int32_t * drafts) {
    for (int i = 0; i < K; ++i) drafts[i] = anchor;
    if (!g_ds_ready) return;
    const bool dbg = getenv("MEGA_DSDBG") != nullptr;
    const double _t0 = now_s();
    static int dumped = 0;
    if (!dumped && getenv("MEGA_DSHID")) { dumped = 1; ds_dump_once(anchor, pos0, (int) pos0); }
    static int logd = 0;

    static float mask_emb[N_EMBD];
    static int mask_done = 0;
    const int32_t mask_tok = getenv("MEGA_DSMASK") ? atoi(getenv("MEGA_DSMASK")) : 75982;
    if (!mask_done) {
        dequant_row_t(mask_emb,
            g_emb->data + (int64_t) mask_tok * src_row_bytes_t(N_EMBD, (int) g_emb->type),
            N_EMBD, (int) g_emb->type);
        mask_done = 1;
    }
    dequant_row_t(g_dsx,
        g_emb->data + (int64_t) anchor * src_row_bytes_t(N_EMBD, (int) g_emb->type),
        N_EMBD, (int) g_emb->type);
    for (int r = 1; r < DS_SLOT; ++r) memcpy(g_dsx + (int64_t) r * N_EMBD, mask_emb, (size_t) N_EMBD * 4);

    g_ds_np = (int) pos0 + 1;         // prefix plus slot 0's own key
    for (int L = 0; L < DS_LAYER; ++L) {
        g_ds_layer = L;
        rmsnorm_rows(g_dsxa, g_dsx, g_ds_attn_norm[L], DS_SLOT);
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsxa, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 3;
        g_mj[0] = { g_ds_wq[L].b, g_ds_wq[L].k, g_ds_wq[L].n, g_dsq };
        g_mj[1] = { g_ds_wk[L].b, g_ds_wk[L].k, g_ds_wk[L].n, g_dsk };
        g_mj[2] = { g_ds_wv[L].b, g_ds_wv[L].k, g_ds_wv[L].n, g_dsv };
        run(task_mgemm_m4);

        for (int r = 0; r < DS_SLOT; ++r) {
            const int pos = (int) (pos0 + r);
            for (int h = 0; h < N_HEAD; ++h) {
                float * q = g_dsq + (int64_t) r * N_EMBD + h * HEAD_DIM;
                ds_rmsvec(q, q, g_ds_qnorm[L], HEAD_DIM);
                ds_rope(q, pos);
            }
            for (int h = 0; h < N_KV; ++h) {
                float * k = g_dsk + (int64_t) r * (N_KV * HEAD_DIM) + h * HEAD_DIM;
                ds_rmsvec(k, k, g_ds_knorm[L], HEAD_DIM);
                ds_rope(k, pos);
            }
            _Float16 * kc = g_dskc + (int64_t) L * KV_LAYER_STRIDE + (int64_t) pos * (N_KV * HEAD_DIM);
            _Float16 * vc = g_dsvc + (int64_t) L * KV_LAYER_STRIDE + (int64_t) pos * (N_KV * HEAD_DIM);
            for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
                kc[i] = (_Float16) g_dsk[(int64_t) r * (N_KV * HEAD_DIM) + i];
                vc[i] = (_Float16) g_dsv[(int64_t) r * (N_KV * HEAD_DIM) + i];
            }
        }
        // Measured at 10K: 338 ms/round against 305 for the per-slot band, so the
        // default stays per-slot. Sharing the stream cannot pay while the inner loop
        // issues one vector reduction per (position, slot) and reads K in 32-byte
        // granules across a 512-byte position stride; the fix is a dim-major tile,
        // which is the same conclusion the target's own attention reached.
        if (g_ds_attn4) parallel_for((int64_t) N_HEAD, ds_attn4_band);
        else parallel_for((int64_t) DS_SLOT * N_HEAD, ds_attn_band);

        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsatt, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 1; g_mj[0] = { g_ds_wo[L].b, g_ds_wo[L].k, g_ds_wo[L].n, g_dsxa };
        run(task_mgemm_m4);
        parallel_for((int64_t) DS_SLOT * N_EMBD, ds_add_band);

        rmsnorm_rows(g_dsxa, g_dsx, g_ds_ffn_norm[L], DS_SLOT);
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsxa, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 2;
        g_mj[0] = { g_ds_wg[L].b, g_ds_wg[L].k, g_ds_wg[L].n, g_dsgate };
        g_mj[1] = { g_ds_wu[L].b, g_ds_wu[L].k, g_ds_wu[L].n, g_dsup };
        run(task_mgemm_m4);
        parallel_for((int64_t) DS_SLOT * N_FF, ds_silu_band);
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsgate, (size_t) N_FF, m4_aq(N_FF));
        g_mjn = 1; g_mj[0] = { g_ds_wd[L].b, g_ds_wd[L].k, g_ds_wd[L].n, g_dsxa };
        run(task_mgemm_m4);
        parallel_for((int64_t) DS_SLOT * N_EMBD, ds_add_band);
        if (g_dsdbg2) {
            double a = 0, b = 0;
            for (int i = 0; i < N_EMBD; ++i) { a += (double) g_dsx[i] * g_dsx[i]; b += (double) g_dsatt[i] * g_dsatt[i]; }
            fprintf(stderr, "  [blk] L%d |x|=%.4f |attn|=%.4f\n", L, sqrt(a), sqrt(b));
        }
    }

    rmsnorm_rows(g_dsxa, g_dsx, g_ds_out_norm, DS_SLOT);
    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_dsxa, (size_t) N_EMBD, m4_aq(N_EMBD));
    g_mjn = 1; g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, g_dslog };
    run(task_mgemm_m4);

    if (dbg) fprintf(stderr, "  [ds] layers done %.2f ms\n", (now_s() - _t0) * 1e3);
    if (const char * lp = getenv("MEGA_DSLOGD")) {
        if (!logd) {
            logd = 1;
            if (FILE * f = fopen(lp, "wb")) {
                fwrite(g_dslog, 4, (size_t) DS_SLOT * VOCAB, f);
                fclose(f);
                printf("dspark: logits -> %s\n", lp);
            }
        }
    }
    const bool nobias = getenv("MEGA_DSNOBIAS") != nullptr;
    int prev = anchor;
    for (int i = 0; i < K; ++i) {
        if (nobias) memset(g_dsbias, 0, (size_t) VOCAB * 4);
        else ds_markov_bias(prev, g_dsbias);
        if (i == 0) {
            if (const char * bp = getenv("MEGA_DSBIASD")) {
                // The markov bias for the first position, so the reference can add
                // the same 130560-wide row to its own base logits.
                if (FILE * f = fopen(bp, "wb")) {
                    int32_t hdr[2] = { prev, anchor };
                    fwrite(hdr, 4, 2, f);
                    fwrite(g_dsbias, 4, VOCAB, f);
                    fclose(f);
                }
            }
        }
        const float * lg = g_dslog + (int64_t) i * VOCAB;
        int best = 0;
        float bv = lg[0] + g_dsbias[0];
        for (int v = 1; v < VOCAB; ++v) {
            const float x = lg[v] + g_dsbias[v];
            if (x > bv) { bv = x; best = v; }
        }
        drafts[i] = best;
        prev = best;
    }
    if (dbg) {
        // Separate the two halves of the decision: the target's lm_head on the draft's
        // hidden state, and the markov bigram bias. If the base argmax is already the
        // token the target emitted, the markov head is what is costing acceptance; if it
        // is not, the draft's own forward is wrong.
        const float * lg = g_dslog;
        int bt = 0; float bv = lg[0];
        for (int v = 1; v < VOCAB; ++v) if (lg[v] > bv) { bv = lg[v]; bt = v; }
        int b2 = 0, b3 = 0; float bv2 = -1e30f, bv3 = -1e30f;
        for (int v = 0; v < VOCAB; ++v) {
            if (v == bt) continue;
            if (lg[v] > bv2) { bv3 = bv2; b3 = b2; bv2 = lg[v]; b2 = v; }
            else if (lg[v] > bv3) { bv3 = lg[v]; b3 = v; }
        }
        fprintf(stderr, "  [ds] base top3 %d(%.2f) %d(%.2f) %d(%.2f)\n", bt, bv, b2, bv2, b3, bv3);
        int mt = 0; float mv = g_dsbias[0];
        for (int v = 1; v < VOCAB; ++v) if (g_dsbias[v] > mv) { mv = g_dsbias[v]; mt = v; }
        fprintf(stderr, "  [ds] draft %d %d %d in %.2f ms | base top %d (%.2f) | bias top %d (%.2f)\n",
                drafts[0], K > 1 ? drafts[1] : -1, K > 2 ? drafts[2] : -1,
                (now_s() - _t0) * 1e3, bt, bv, mt, mv);
    }
}

// Draft-local residual add and silu: add4_fn/silu4_fn are hard-wired to the target's
// four-row buffers, and the drafter runs before the round those belong to.
static void ds_add_band(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i < 32 ? hi - i : 32));
        __riscv_vse32_v_f32m1(g_dsx + i, __riscv_vfadd_vv_f32m1(
            __riscv_vle32_v_f32m1(g_dsx + i, vl), __riscv_vle32_v_f32m1(g_dsxa + i, vl), vl), vl);
    }
}
static void ds_silu_band(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i < 32 ? hi - i : 32));
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(g_dsgate + i, vl);
        const vfloat32m1_t u = __riscv_vle32_v_f32m1(g_dsup + i, vl);
        vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, -1.4426950408889634f, vl);
        t = __riscv_vfmax_vf_f32m1(t, -126.0f, vl);
        t = __riscv_vfmin_vf_f32m1(t,  126.0f, vl);
        const vfloat32m1_t den = __riscv_vfadd_vf_f32m1(exp2f_e(t, vl), 1.0f, vl);
        __riscv_vse32_v_f32m1(g_dsgate + i,
            __riscv_vfmul_vv_f32m1(__riscv_vfdiv_vv_f32m1(x, den, vl), u, vl), vl);
    }
}

// ---------------------------------------------- draft, verify, accept, advance
// One round: the drafter proposes K tokens, the target checks four rows in one
// weight pass, and the longest matching prefix is committed. The token that breaks
// the match is the target's own answer for that position, so the output is exactly
// what greedy decoding would have produced -- the drafts only change the speed.
// Both of these live at file scope for the same reason g_m4out does: m4_round
// inlines its whole four-row body, and a caller-stack array came back with stale
// words -- here the four candidate tokens, which then indexed the embedding table
// out of bounds.
static int32_t g_spec_dr[DS_SLOT];
static int g_spec_tok[DS_SLOT];

static bool m4spec_run(int64_t p0, int anchor, int rounds, int K) {
    if (K < 1) K = 1;
    if (K > DS_SLOT - 1) K = DS_SLOT - 1;
    int32_t * const dr = g_spec_dr;
    double cos_sum = 0.0; int cos_n = 0, top1_hit = 0, top8_hit = 0;
    long long acc = 0, total = 0;
    double best = 1e9, tot = 0;
    int shown = 0;
    for (int it = 0; it < rounds; ++it) {
        const double t0 = now_s();
        ds_draft(anchor, p0, K, dr);

        int * const tok = g_spec_tok;
        tok[0] = anchor;
        for (int i = 0; i < K; ++i) tok[1 + i] = dr[i];
        for (int i = K + 1; i < DS_SLOT; ++i) tok[i] = dr[K - 1];

        if (!m4_round(p0, tok, g_m4out)) return false;

        // Acceptance is a 0/1 summary of a comparison that is really a whole
        // distribution. Row 0 of the round is the anchor, so its logit row is the
        // target's answer to exactly the question the draft's slot 0 answered:
        // what follows the anchor. Cosine between the two, and where the target's
        // own argmax lands in the draft's ranking, says whether the draft is
        // slightly wrong (a near miss the markov head could fix) or looking at a
        // different context (which it cannot).
        if (g_dscos) {
            const float * a = g_dslog;               // draft slot 0
            const float * b = g_logits4;             // target, anchor row
            double ab = 0, aa = 0, bb = 0;
            for (int v = 0; v < VOCAB; ++v) {
                ab += (double) a[v] * b[v];
                aa += (double) a[v] * a[v];
                bb += (double) b[v] * b[v];
            }
            const double cs = (aa > 0 && bb > 0) ? ab / sqrt(aa * bb) : 0.0;
            // rank of the target's argmax inside the draft's own ranking
            const int want = g_m4out[0];
            int rank = 0;
            for (int v = 0; v < VOCAB; ++v) if (a[v] > a[want]) ++rank;
            cos_sum += cs; cos_n += 1;
            if (rank == 0) top1_hit += 1;
            if (rank < 8) top8_hit += 1;
            if (cos_n <= 12)
                fprintf(stderr, "  [cos] p=%lld cos=%.4f  target argmax %d at draft rank %d\n",
                        (long long) p0, cs, want, rank);
        }

        int n = 0;
        while (n < K && g_m4out[n] == dr[n]) ++n;
        acc += n; total += K;
        const int next = g_m4out[n];

        // The round's capture has the target's residual stream for every row it ran,
        // so the committed positions go into the draft's cache before the next round.
        ds_inject4(p0, 0, n + 1);
        if (shown < 12) {
            printf("spec %2d p=%lld  acc %d/%d  drafts %d %d %d  -> %d\n",
                   it, (long long) p0, n, K, dr[0], dr[1], dr[2], next);
            ++shown;
        }
        p0 += n + 1;
        anchor = next;
        const double dt = now_s() - t0;
        tot += dt;
        if (dt < best) best = dt;
    }
    const double emit = (double) rounds + (double) acc;
    printf("spec: %d rounds, %lld/%lld drafts accepted (%.3f fraction, %.3f accepted/round), %lld tokens\n",
           rounds, acc, total, total ? (double) acc / total : 0.0,
           rounds ? (double) acc / rounds : 0.0, (long long) emit);
    // Round 0 is cold (score and weight buffers after prefill), which is why `best`
    // is reported next to the mean, and the second rate is what the same rounds
    // would give if no draft ever matched.
    printf("spec: best %.2f ms/round, mean %.2f ms/round -> %.2f tok/s "
           "(%.2f with no draft accepted)\n",
           best * 1e3, tot / rounds * 1e3, emit / tot, (double) rounds / tot);
    if (cos_n)
        printf("spec: draft vs target logits: mean cos %.4f, target argmax at draft rank 0 "
               "in %d/%d rounds (%d/%d within 8)\n",
               cos_sum / cos_n, top1_hit, cos_n, top8_hit, cos_n);
    printf("spec: inject %.2f s total\n", g_ds_t_inj);
    return true;
}

static void ds_setup(void) {
    if (g_ds) return;
    if (!ds_load()) { fprintf(stderr, "dspark: disabled\n"); return; }
    if (!ds_alloc()) return;
    g_ds = 1; g_ds_ready = 1;
}


// ------------------------------------ four rows of rope and KV, spread over the pool
// rope_all is written against one position on one core, so four rows is four
// serial calls per layer, 168 per round. Inside it the f16 K tile write is 128
// strided stores per kv head -- the tile is [dim][position], so each store lands in
// a different cache line. One (row, kv) unit per pool thread makes eight
// independent jobs, and Q's sixteen heads split exactly eight per kv head.
static int64_t g_m4rope_p0;

static inline void m4_rope_apply(float * x, const float * cs, float post) {
    rope_head_norm(x, cs, post);
}

static void m4_rope_item(int r, int kv) {
    const int pos = (int) (g_m4rope_p0 + r);
    float * q = g_q4 + (int64_t) r * N_EMBD;
    float * k = g_k4 + (int64_t) r * (N_KV * HEAD_DIM);
    const float * v = g_v4 + (int64_t) r * (N_KV * HEAD_DIM);
    const float * cs = g_rope_tab + (int64_t) pos * 128;
    const float inv = 1.0f / sqrtf((float) HEAD_DIM);

    for (int i = 0; i < NH_PER_KV; ++i)
        m4_rope_apply(q + (kv * NH_PER_KV + i) * HEAD_DIM, cs, inv);
    m4_rope_apply(k + kv * HEAD_DIM, cs, 1.0f);

    const int pblk = pos / KT, pi = pos % KT;
    _Float16 * kb16 = ktile(g_layer, kv, pblk);
    _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE
                 + (int64_t) pos * N_KV * HEAD_DIM;
    if (g_k8dec) {
        int8_t * k8base = g_k8d + (int64_t) g_layer * K_LAYER_STRIDE;
        int8_t * kb8 = k8base + (((int64_t) kv * NKT + pblk) * HEAD_DIM) * KT;
        const float ik = g_k8dsc[g_layer][kv] > 0 ? 1.0f / g_k8dsc[g_layer][kv] : 1.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            int b = (int) lrintf(k[kv * HEAD_DIM + d] * ik);
            if (b > 127) b = 127;
            if (b < -127) b = -127;
            kb8[(int64_t) d * KT + pi] = (int8_t) b;
        }
    }
    for (int d = 0; d < HEAD_DIM; ++d) kb16[(int64_t) d * KT + pi] = (_Float16) k[kv * HEAD_DIM + d];
    for (int d = 0; d < HEAD_DIM; d += 32)
        __riscv_vse16_v_f16mf2(vb + kv * HEAD_DIM + d,
                               __riscv_vfncvt_f_f_w_f16mf2(__riscv_vle32_v_f32m1(v + kv * HEAD_DIM + d, 32), 32), 32);
}

static void m4_rope_band(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) m4_rope_item((int) (i / N_KV), (int) (i % N_KV));
}

static bool m4_round(int64_t p0, const int * toks, int * out) {
    if (!g_x4) { m4dec_alloc(); if (!g_x4) return false; }
    if (p0 + 4 > MAXCTX) return false;
    const bool m4dbg = getenv("MEGA_DSDBG") != nullptr;
    if (m4dbg) fprintf(stderr, "  [m4] round at %lld start\n", (long long) p0);
    for (int r = 0; r < 4; ++r)
        dequant_row_t(g_x4 + (int64_t) r * N_EMBD,
                      g_emb->data + (int64_t) toks[r] * src_row_bytes_t(N_EMBD, (int) g_emb->type),
                      N_EMBD, (int) g_emb->type);

    for (int L = 0; L < N_LAYER; ++L) {
        g_layer = L;
        if (M4DUMP && L < 2) dump_row("m4", L, 'e', g_x4);
        double _n = now_s();
        rmsnorm_rows(g_xa4, g_x4, g_attn_norm[L], 4);
        if (M4TIME) g_m4t[2] += now_s() - _n;
        if (M4DUMP && L < 2) dump_row("m4", L, 'n', g_xa4);
        double _t = now_s();
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_xa4, (size_t) N_EMBD, m4_aq(N_EMBD));
        if (M4TIME) { g_m4t[5] += now_s() - _t; _t = now_s(); }
        g_mjn = 3;
        g_mj[0] = { g_wq[L].b, g_wq[L].k, g_wq[L].n, g_q4 };
        g_mj[1] = { g_wk[L].b, g_wk[L].k, g_wk[L].n, g_k4 };
        g_mj[2] = { g_wv[L].b, g_wv[L].k, g_wv[L].n, g_v4 };
        run(task_mgemm_m4);
        if (M4TIME) { g_m4t[0] += now_s() - _t; _t = now_s(); }
        // q/k/v each own a buffer: ldc is the matrix width, so pointing all three
        // at one buffer makes k and v overwrite q's first rows.

        // rope and the f16 KV store, one (row, kv) unit per pool thread.
        g_m4rope_p0 = p0;
        parallel_for(4 * N_KV, m4_rope_band);
        // After rope, so the K/V here are comparable with the single-row dump.
        if (M4DUMP && L < 2) {
            dump_row("m4", L, 'q', g_q4);
            dump_n("m4", L, 'k', g_k4, N_KV * HEAD_DIM);
            dump_n("m4", L, 'v', g_v4, N_KV * HEAD_DIM);
        }
        conv_kv8_inc(L, p0, 4);
        if (M4TIME) { g_m4t[4] += now_s() - _t; _t = now_s(); }

        g_pq = g_q4;
        g_pfa_ctx = p0;
        g_pfa_rows = 4;
        double _pq = now_s();
        pfa_pack_q(4);
        if (M4TIME) { g_m4t[7] += now_s() - _pq; }
        // pfa_sm_i8 and pfa_pv_i8 dequantise with the globals g_k8isc/g_v8isc,
        // which prefill only keeps valid for the layer it just converted. Decode
        // appends to every layer's plane up front, so this round has to restore
        // this layer's own grid or every layer dequantises with layer 41's scale.
        for (int kv = 0; kv < N_KV; ++kv) {
            g_k8isc[kv] = g_k8i_sc[L * N_KV + kv];
            g_v8isc[kv] = g_v8i_sc[L * N_KV + kv];
        }
        pfa_run(L);
        if (M4TIME) { g_m4t[1] += now_s() - _t; _t = now_s(); }
        if (M4DUMP && L < 2) dump_row("m4", L, 'a', g_pfa_out);

        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_pfa_out, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 1; g_mj[0] = { g_wov[L].b, g_wov[L].k, g_wov[L].n, g_xa4 };
        run(task_mgemm_m4);
        if (M4TIME) { g_m4t[0] += now_s() - _t; _t = now_s(); }
        if (M4DUMP && L < 2) dump_row("m4", L, 'o', g_xa4);
        parallel_for(4 * N_EMBD, add4_fn);
        if (M4DUMP && L < 2) dump_row("m4", L, 'r', g_x4);

        double _n2 = now_s();
        rmsnorm_rows(g_xa4, g_x4, g_ffn_norm[L], 4);
        if (M4TIME) g_m4t[2] += now_s() - _n2;
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_xa4, (size_t) N_EMBD, m4_aq(N_EMBD));
        g_mjn = 2;
        g_mj[0] = { g_wg[L].b, g_wg[L].k, g_wg[L].n, g_gate4 };
        g_mj[1] = { g_wu[L].b, g_wu[L].k, g_wu[L].n, g_up4 };
        run(task_mgemm_m4);
        if (M4TIME) { g_m4t[0] += now_s() - _t; _t = now_s(); }
        if (M4DUMP && L < 2) {
            dump_n("m4", L, 'g', g_gate4, N_FF);
            dump_n("m4", L, 'u', g_up4, N_FF);
        }
        parallel_for(4 * N_FF, silu4_fn);
        if (M4TIME) { g_m4t[3] += now_s() - _t; _t = now_s(); }
        spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_gate4, (size_t) N_FF, m4_aq(N_FF));
        g_mjn = 1; g_mj[0] = { g_wd[L].b, g_wd[L].k, g_wd[L].n, g_xa4 };
        run(task_mgemm_m4);
        if (M4TIME) { g_m4t[0] += now_s() - _t; _t = now_s(); }
        parallel_for(4 * N_EMBD, add4_fn);
        // The drafter's encoder reads the residual stream of these layers. Captured
        // after the FFN residual add, which is what mega_decode feeds the same head.
        if (g_ds && ds_target_slot(L) >= 0) {
            const int k = ds_target_slot(L);
            for (int r = 0; r < 4; ++r)
                memcpy(g_dshid + ((int64_t) k * g_ds_cap + r) * N_EMBD,
                       g_x4 + (int64_t) r * N_EMBD, (size_t) N_EMBD * 4);
        }
    }

    rmsnorm_rows(g_xa4, g_x4, g_out_norm, 4);
    spacemit_kernels::rvv::quantize_a_4row_i8_hp(256, g_xa4, (size_t) N_EMBD, m4_aq(N_EMBD));
    g_mjn = 1; g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, g_logits4 };
    run(task_mgemm_m4);
    // task_argmax reads g_logits, so the four heads are taken one row at a time.
    // Ties go to the lower index exactly as the single-row path does.
    if (const char * ld = getenv("MEGA_M4LOGD")) {
        // Same dump the single-row path writes with MEGA_LOGD, so the two argmax
        // decisions can be compared over the whole vocabulary instead of by id.
        if (FILE * f = fopen(ld, "wb")) { fwrite(g_logits4, 4, VOCAB, f); fclose(f); }
        int idx[5] = { -1, -1, -1, -1, -1 };
        for (int j = 0; j < 5; ++j)
            for (int v = 0; v < VOCAB; ++v) {
                bool used = false;
                for (int q2 = 0; q2 < j; ++q2) if (idx[q2] == v) used = true;
                if (used) continue;
                if (idx[j] < 0 || g_logits4[v] > g_logits4[idx[j]]) idx[j] = v;
            }
        printf("m4dec: row0 top5");
        for (int j = 0; j < 5; ++j) printf(" %d:%.3f", idx[j], (double) g_logits4[idx[j]]);
        printf("\n");
        unsetenv("MEGA_M4LOGD");
    }
    // One pass over all four rows instead of four passes over g_logits: the rows
    // are contiguous, so a band per thread covers them naturally and each thread
    // keeps four running maxima. Scanning in increasing index with a strict ">"
    // keeps ties on the lower id, which is what the single-row path does.
    {
        if (M4ARGMAX_ONE) {
            run(task_argmax4_init);
            parallel_for(4 * VOCAB, argmax4_fn);
            for (int r = 0; r < 4; ++r) {
                int best = g_m4b_i[0][r];
                float bv = g_m4b_v[0][r];
                for (int t = 1; t < g_participants; ++t)
                    if (g_m4b_v[t][r] > bv || (g_m4b_v[t][r] == bv && g_m4b_i[t][r] < best)) {
                        bv = g_m4b_v[t][r]; best = g_m4b_i[t][r];
                    }
                out[r] = best;
                if (M4DUMP && r == 0)
                    printf("m4dec: ax2 row0 idx %d val %.4f\n", best, (double) bv);
            }
        } else {
            float * keep = g_logits;
            for (int r = 0; r < 4; ++r) {
                g_logits = g_logits4 + (int64_t) r * VOCAB;
                run(task_argmax);
                int best = g_am_idx[0];
                float bv = g_am_val[0];
                for (int t = 1; t < g_participants; ++t)
                    if (g_am_val[t] > bv || (g_am_val[t] == bv && g_am_idx[t] < best)) {
                        bv = g_am_val[t]; best = g_am_idx[t];
                    }
                out[r] = best;
                if (M4DUMP && r == 0)
                    printf("m4dec: per-row argmax row0 idx %d val %.4f (am0 %.4f idx0 %d)\n",
                           best, (double) bv, (double) g_am_val[0], g_am_idx[0]);
            }
            g_logits = keep;
        }
    }
    if (M4TIME) {
        // qk/pv are A-cluster core-s, sm is X100 core-s: the split says whether the
        // round's attention is dot-bound or bound by the KV stream.
        // qk/pv are per A thread, sm per X100 thread; both index 0..7.
        double qk = 0, sm = 0, pv = 0, mx = 0, ex = 0;
        for (int t = 0; t < N_GPU; ++t) {
            qk += g_pfa_qk[(t) * PFA_TSTRIDE]; sm += g_pfa_sm[(t) * PFA_TSTRIDE]; pv += g_pfa_pv[(t) * PFA_TSTRIDE];
            mx += g_pfa_mxb[(t) * PFA_TSTRIDE]; ex += g_pfa_exb[(t) * PFA_TSTRIDE];
        }
        printf("m4dec: attn core-s qk %.3f + pv %.3f on 8, sm %.3f (max %.3f exp %.3f) on %d\n",
               qk, pv, sm, mx, ex, g_n_sm);
        for (int t = 0; t < N_GPU; ++t) {
            g_pfa_qk[(t) * PFA_TSTRIDE] = 0; g_pfa_pv[(t) * PFA_TSTRIDE] = 0; g_pfa_sm[(t) * PFA_TSTRIDE] = 0;
            g_pfa_smb[(t) * PFA_TSTRIDE] = 0; g_pfa_mxb[(t) * PFA_TSTRIDE] = 0; g_pfa_exb[(t) * PFA_TSTRIDE] = 0;
        }
        m4_report();
    }
    return true;
}

// Repeat-the-anchor driver. Four identical rows at four positions is the
// cheapest round whose answer is known: row 0 must emit the token the single-row
// chain emits at the same position, and rows 1..3 must be a stable fixpoint.
static bool m4dec_run(int64_t p0, int anchor, int rounds) {
    m4dec_alloc();
    if (!g_x4) return false;
    // These counters are shared with prefill's attention, and a 10K prefill leaves
    // several hundred core-s in them. Clear them or the round reports prefill work.
    for (int t = 0; t < N_GPU; ++t) {
        g_pfa_qk[(t) * PFA_TSTRIDE] = 0; g_pfa_pv[(t) * PFA_TSTRIDE] = 0; g_pfa_sm[(t) * PFA_TSTRIDE] = 0;
        g_pfa_smb[(t) * PFA_TSTRIDE] = 0; g_pfa_mxb[(t) * PFA_TSTRIDE] = 0; g_pfa_exb[(t) * PFA_TSTRIDE] = 0;
    }
    if (!g_a4p) { fprintf(stderr, "m4dec needs the four-row attention (MEGA_ATTN4P=1)\n"); }
    int tok[4] = { anchor, anchor, anchor, anchor };
    int * out = g_m4out;
    if (g_m4p0 >= 0) { p0 = g_m4p0; for (int i = 0; i < 4; ++i) tok[i] = g_m4tok[i]; }
    double best = 1e9;
    for (int it = 0; it < rounds; ++it) {        const double t0 = now_s();
        if (!m4_round(p0, tok, out)) return false;
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        printf("m4dec: round %d at pos %lld  %.2f ms  rows -> %d %d %d %d\n",
               it, (long long) p0, dt * 1e3, out[0], out[1], out[2], out[3]);
        p0 += 4;
    }
    printf("m4dec: best %.2f ms/round -> %.2f tok/s at 4/round, %.2f at 3/4 accepted\n",
           best * 1e3, 4.0 / best, 3.0 / best);
    return true;
}


static void attn4_run(int rounds, int64_t ctx, int64_t rows);
static void attn4_bench(int rounds, int64_t ctxmax) {
    // I8_ROWS = I8_G * PFA_ROWS is the prefill row tile; four rows is the decode
    // tile, which the head-major loop drives one group at a time. Anything between
    // the two is not a shape the kernel has.
    if (g_a4p) attn4p_alloc();
    if (g_a4s) a4s_alloc();
    int64_t rows = getenv("MEGA_ATTN4_ROWS") ? atoll(getenv("MEGA_ATTN4_ROWS")) : I8_ROWS;
    if (rows != PFA_ROWS) {
        rows = rows < I8_ROWS ? I8_ROWS : rows;
        if (rows % I8_ROWS) rows = I8_ROWS;
    }
    for (int64_t i = 0; i < rows * N_HEAD * HEAD_DIM; ++i)
        g_pq[i] = 0.01f * (float) ((i * 37) % 101) - 0.5f;
    // Same caches, several context lengths: if the cost is not linear in the
    // position count the loop is not doing the whole context.
    int64_t cts[8]; int nct = 0;
    if (getenv("MEGA_ATTN4_CTX")) {
        char * s = getenv("MEGA_ATTN4_CTX");
        while (*s && nct < 8) {
            char * e; long long v = strtoll(s, &e, 10);
            if (e == s) break;
            cts[nct++] = v; s = *e == ',' ? e + 1 : e;
        }
    } else cts[nct++] = ctxmax;
    for (int ci = 0; ci < nct; ++ci) attn4_run(rounds, cts[ci], rows);
}

static void attn4_run(int rounds, int64_t ctx, int64_t rows) {
    if (ctx < 128) { printf("attn4: context %lld too short\n", (long long) ctx); return; }
    const bool four = rows == PFA_ROWS;
    (void) four;
    double best = 1e9;
    for (int r = 0; r < rounds; ++r) {
        for (int t = 0; t < N_GPU; ++t) g_pfa_qk[(t) * PFA_TSTRIDE] = g_pfa_sm[(t) * PFA_TSTRIDE] = g_pfa_pv[(t) * PFA_TSTRIDE] = 0;
        const double t0 = now_s();
        for (int L = 0; L < N_LAYER; ++L) {
            g_pfa_ctx = ctx;
            g_pfa_rows = rows;
            pfa_pack_q(rows);
            pfa_run(L);
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        double tq = 0, ts = 0, tv = 0;
        for (int t = 0; t < N_GPU; ++t) { tq += g_pfa_qk[(t) * PFA_TSTRIDE]; ts += g_pfa_sm[(t) * PFA_TSTRIDE]; tv += g_pfa_pv[(t) * PFA_TSTRIDE]; }
        // The four-row round's total MAC count is rows*heads*dim*ctx*2 over all
        // layers; dividing by the wall time gives aggregate GMAC/s, so it must
        // carry the layer count or it reads 42x low.
        const double mac = 2.0 * (double) rows * N_HEAD * HEAD_DIM * (double) ctx * N_LAYER;
        printf("attn4: round %d %.2f ms (%.2f ms/layer, ctx %lld, %s)\n"
               "       work qk %.1f  sm %.1f  pv %.1f (core-s)  -> %.1f GMAC/s aggregate\n",
               r, dt * 1e3, dt * 1e3 / N_LAYER, (long long) ctx,
               g_pipe ? "qk/pv A100 + softmax X100" : "single cluster",
               tq, ts, tv, mac / dt / 1e9);
    }
    printf("attn4: best %.2f ms for %d layers, %lld rows x %lld pos (%.3f ms/layer)\n",
           best * 1e3, N_LAYER, (long long) rows, (long long) ctx, best * 1e3 / N_LAYER);
    // The split path moves softmax to the other cluster but runs the same
    // pfa_sm_i8, so the two runs must agree exactly on the same synthetic input.
    {   double sum = 0, sq = 0;
        const int64_t n = rows * N_HEAD * HEAD_DIM;
        for (int64_t i = 0; i < n; ++i) { const double v = g_pfa_out[i]; sum += v; sq += v * v; }
        printf("attn4: out sum %.9f  sumsq %.6f  (split %d)\n", sum, sq, g_a4p ? 1 : 0);
    }
    g_i8g = I8_G;
}

static void pfa_head(int layer, int h, int64_t qbase) {
    const int kv = h / NH_PER_KV;
    _Float16 * vbase = g_vpos + (int64_t) layer * KV_LAYER_STRIDE + kv * HEAD_DIM;

    // scores for PFA_ROWS rows, every position they can see. The row
    // accumulators must stay in registers: taking their address made GCC spill
    // all four to the stack and turned each FMA into load-modify-store, which
    // cost more than an order of magnitude.
    const float * q0 = g_pq + ((qbase + 0) * N_HEAD + h) * HEAD_DIM;
    const float * q1 = g_pq + ((qbase + 1) * N_HEAD + h) * HEAD_DIM;
    const float * q2 = g_pq + ((qbase + 2) * N_HEAD + h) * HEAD_DIM;
    const float * q3 = g_pq + ((qbase + 3) * N_HEAD + h) * HEAD_DIM;
    const int len_max = (int) (g_pfa_ctx + qbase + PFA_ROWS);
    const double _q = now_s();
#define PFA_ACC(Q, A)                                                     \
    A = __riscv_vfmacc_vf_f32m1(A, (Q)[d + 0], k0, vl);                   \
    A = __riscv_vfmacc_vf_f32m1(A, (Q)[d + 1], k1, vl);                   \
    A = __riscv_vfmacc_vf_f32m1(A, (Q)[d + 2], k2, vl);                   \
    A = __riscv_vfmacc_vf_f32m1(A, (Q)[d + 3], k3, vl);
    const int64_t ntotal = g_pfa_ctx + g_pfa_rows;
    for (int pb = 0; pb < len_max; pb += KT) {
        const int npos = (len_max - pb < KT) ? (len_max - pb) : KT;
        const size_t vl = __riscv_vsetvl_e32m1((size_t) npos);
        const float * kt = kftile(kv, pb / KT);
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), a1 = a0, a2 = a0, a3 = a0;
        for (int d = 0; d < HEAD_DIM; d += 4) {
            const float * base = kt + d * KT;
            const vfloat32m1_t k0 = __riscv_vle32_v_f32m1(base + 0 * KT, vl);
            const vfloat32m1_t k1 = __riscv_vle32_v_f32m1(base + 1 * KT, vl);
            const vfloat32m1_t k2 = __riscv_vle32_v_f32m1(base + 2 * KT, vl);
            const vfloat32m1_t k3 = __riscv_vle32_v_f32m1(base + 3 * KT, vl);
            PFA_ACC(q0, a0) PFA_ACC(q1, a1) PFA_ACC(q2, a2) PFA_ACC(q3, a3)
        }
        __riscv_vse32_v_f32m1(g_psa[g_pfa_tid][0] + pb, a0, vl);
        __riscv_vse32_v_f32m1(g_psa[g_pfa_tid][1] + pb, a1, vl);
        __riscv_vse32_v_f32m1(g_psa[g_pfa_tid][2] + pb, a2, vl);
        __riscv_vse32_v_f32m1(g_psa[g_pfa_tid][3] + pb, a3, vl);
    }
#undef PFA_ACC
    g_pfa_qk[(g_pfa_tid) * PFA_TSTRIDE] += now_s() - _q;

    // causal trim + softmax, one row at a time
    for (int r = 0; r < PFA_ROWS; ++r) {
        float * row = g_psa[g_pfa_tid][r];
        const int len = (int) (g_pfa_ctx + qbase + r + 1);
        for (int i = len; i < len_max; ++i) row[i] = -INFINITY;
        pfa_softmax_row(row, len, &g_pfa_sm[(g_pfa_tid) * PFA_TSTRIDE]);
        for (int i = len; i < len_max; ++i) row[i] = 0.0f;
    }

    // PV: one V load per position, four rows of accumulators
    const double _v = now_s();
    vfloat32m1_t o00 = __riscv_vfmv_v_f_f32m1(0.0f, 32), o01 = o00, o02 = o00, o03 = o00;
    vfloat32m1_t o10 = o00, o11 = o00, o12 = o00, o13 = o00;
    vfloat32m1_t o20 = o00, o21 = o00, o22 = o00, o23 = o00;
    vfloat32m1_t o30 = o00, o31 = o00, o32 = o00, o33 = o00;
    const float * vfp = g_vf + (int64_t) kv * ntotal * HEAD_DIM;
    for (int p = 0; p < len_max; ++p) {
        const float * vrow = vfp + (int64_t) p * HEAD_DIM;
        const vfloat32m1_t v0 = __riscv_vle32_v_f32m1(vrow + 0, 32);
        const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(vrow + 32, 32);
        const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(vrow + 64, 32);
        const vfloat32m1_t v3 = __riscv_vle32_v_f32m1(vrow + 96, 32);
        const float w0 = g_psa[g_pfa_tid][0][p], w1 = g_psa[g_pfa_tid][1][p];
        const float w2 = g_psa[g_pfa_tid][2][p], w3 = g_psa[g_pfa_tid][3][p];
        o00 = __riscv_vfmacc_vf_f32m1(o00, w0, v0, 32); o01 = __riscv_vfmacc_vf_f32m1(o01, w0, v1, 32);
        o02 = __riscv_vfmacc_vf_f32m1(o02, w0, v2, 32); o03 = __riscv_vfmacc_vf_f32m1(o03, w0, v3, 32);
        o10 = __riscv_vfmacc_vf_f32m1(o10, w1, v0, 32); o11 = __riscv_vfmacc_vf_f32m1(o11, w1, v1, 32);
        o12 = __riscv_vfmacc_vf_f32m1(o12, w1, v2, 32); o13 = __riscv_vfmacc_vf_f32m1(o13, w1, v3, 32);
        o20 = __riscv_vfmacc_vf_f32m1(o20, w2, v0, 32); o21 = __riscv_vfmacc_vf_f32m1(o21, w2, v1, 32);
        o22 = __riscv_vfmacc_vf_f32m1(o22, w2, v2, 32); o23 = __riscv_vfmacc_vf_f32m1(o23, w2, v3, 32);
        o30 = __riscv_vfmacc_vf_f32m1(o30, w3, v0, 32); o31 = __riscv_vfmacc_vf_f32m1(o31, w3, v1, 32);
        o32 = __riscv_vfmacc_vf_f32m1(o32, w3, v2, 32); o33 = __riscv_vfmacc_vf_f32m1(o33, w3, v3, 32);
    }
    float * dst = (layer == 0 && g_pfa_out) ? g_pfa_out + ((qbase * N_HEAD + h) * HEAD_DIM) : g_ppv_tmp[g_pfa_tid];
    __riscv_vse32_v_f32m1(dst + 0, o00, 32);  __riscv_vse32_v_f32m1(dst + 32, o01, 32);
    __riscv_vse32_v_f32m1(dst + 64, o02, 32); __riscv_vse32_v_f32m1(dst + 96, o03, 32);
    dst += (int64_t) N_HEAD * HEAD_DIM;
    __riscv_vse32_v_f32m1(dst + 0, o10, 32);  __riscv_vse32_v_f32m1(dst + 32, o11, 32);
    __riscv_vse32_v_f32m1(dst + 64, o12, 32); __riscv_vse32_v_f32m1(dst + 96, o13, 32);
    dst += (int64_t) N_HEAD * HEAD_DIM;
    __riscv_vse32_v_f32m1(dst + 0, o20, 32);  __riscv_vse32_v_f32m1(dst + 32, o21, 32);
    __riscv_vse32_v_f32m1(dst + 64, o22, 32); __riscv_vse32_v_f32m1(dst + 96, o23, 32);
    dst += (int64_t) N_HEAD * HEAD_DIM;
    __riscv_vse32_v_f32m1(dst + 0, o30, 32);  __riscv_vse32_v_f32m1(dst + 32, o31, 32);
    __riscv_vse32_v_f32m1(dst + 64, o32, 32); __riscv_vse32_v_f32m1(dst + 96, o33, 32);
    g_pfa_pv[(g_pfa_tid) * PFA_TSTRIDE] += now_s() - _v;
}

static void task_pfa(int tid) {
    if (g_participants != 8) return;
    g_pfa_tid = tid;
    const int h0 = tid * 2;
    for (int64_t qb = 0; qb + PFA_ROWS <= g_pfa_rows; qb += PFA_ROWS)
        for (int h = h0; h < h0 + 2; ++h)
            pfa_head(g_pfa_layer, h, qb);
}

// aligned_alloc requires the size to be a multiple of the alignment; the bench
// asks for odd byte counts, which is UB (ASan refuses it outright).
static void * hugemem(size_t n) {
    const size_t a = 2 << 20;
    return aligned_alloc(a, (n + a - 1) & ~(a - 1));
}


static const char * work_path(const char * name);
// ---------------------------------------------------------------------------
// Real prefill.
//
// The two benches above proved the kernels but fed them synthetic activations;
// this drives the same int8 GEMM and int8 attention from a real prompt and
// leaves the f16 KV cache populated for decode.
//
// Layer-major within a chunk: all of a layer's seven GEMMs see every chunk row,
// so each weight matrix is streamed once per chunk. Chunking by rows keeps the
// row buffers small; the price is that the weight blob is re-streamed once per
// chunk, which is 20 passes of 1.27 GB at 10K.
// ---------------------------------------------------------------------------
static float * g_pfx, * g_pfxa, * g_pfgate, * g_pfup, * g_pfk, * g_pfv;

// Prefill scratch is sized for a whole 10K chunk (hundreds of MB) and decode
// needs none of it. Releasing it before decode keeps the working set of the
// generation loop inside the cache the attention pass cares about.
// f16 cache -> int8 cache for one layer, using the scale prefill's own max-abs
// scan produced for that layer. Called from the prefill layer loop, where the
// bytes are still in cache from the attention that just read them.
static float g_cv8_ik, g_cv8_iv;
// Per-kv-head reciprocals of the prefill scale. One cache-wide scale is wrong:
// the two kv heads have different max-abs, and decode restores each head with
// its own scale, so a shared conversion factor is a silent per-head gain.
static float g_cv8_ik_arr[N_KV], g_cv8_iv_arr[N_KV];
static void conv_kv8_band(int64_t lo, int64_t hi) {
    const int64_t L = g_layer;
    int8_t * k8 = g_k8d + L * K_LAYER_STRIDE;
    const _Float16 * k16 = g_kdim + L * K_LAYER_STRIDE;
    // The K cache is [kv][pos block][d][32 lanes], so a 32-byte step never
    // crosses a kv head; the scale for the head that owns this block is picked
    // per block. Using kv head 0's scale for the whole cache left head 1
    // multiplied by s1/s0, which is invisible at 512 positions and turns the
    // 10K first token from 8181 into 220.
    const int64_t kvsz = (int64_t) NKT * HEAD_DIM * KT;
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i < 32 ? hi - i : 32));
        const float ik = g_cv8_ik_arr[i / kvsz];
        const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
            __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(k16 + i, vl), vl), ik, vl);
        vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(f, vl);
        q = __riscv_vmax_vx_i32m1(q, -127, vl);
        q = __riscv_vmin_vx_i32m1(q, 127, vl);
        __riscv_vse8_v_i8mf4(k8 + i,
                             __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, vl), vl), vl);
    }
}
static void conv_kv8_v_band(int64_t lo, int64_t hi) {
    const int64_t L = g_layer;
    int8_t * v8 = g_v8d + L * KV_LAYER_STRIDE;
    const _Float16 * v16 = g_vpos + L * KV_LAYER_STRIDE;
    // Source V is [pos][kv][d]; the destination is [kv][half][pos][64], so the
    // block index decides both kv and half and the write lands in a plane.
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i < 32 ? hi - i : 32));
        const int64_t kv = (i / HEAD_DIM) % N_KV;
        const int64_t d = i % HEAD_DIM;
        const float iv = g_cv8_iv_arr[kv];
        const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
            __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(v16 + i, vl), vl), iv, vl);
        vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(f, vl);
        q = __riscv_vmax_vx_i32m1(q, -127, vl);
        q = __riscv_vmin_vx_i32m1(q, 127, vl);
        int8_t * dst = v8 + v8_off((int) kv, (int) (d / 64), i / (N_KV * HEAD_DIM)) + d % 64;
        __riscv_vse8_v_i8mf4(dst,
                             __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, vl), vl), vl);
    }
}
// Quantises positions [0, npos) of every layer from the f16 caches. Only used
// when the int8 cache is wanted but prefill did not fill it layer by layer.
static void conv_kv8_all(int64_t npos) {
    const int64_t nk = (int64_t) N_KV * NKT * HEAD_DIM * KT;
    const int64_t nv = npos * N_KV * HEAD_DIM;
    for (int L = 0; L < N_LAYER; ++L) {
        g_layer = L;
        for (int kv = 0; kv < N_KV; ++kv) {
            g_cv8_ik_arr[kv] = g_k8dsc[L][kv] > 0 ? 1.0f / g_k8dsc[L][kv] : 1.0f;
            g_cv8_iv_arr[kv] = g_v8dsc[L][kv] > 0 ? 1.0f / g_v8dsc[L][kv] : 1.0f;
        }
        g_cv8_ik = g_cv8_ik_arr[0];
        g_cv8_iv = g_cv8_iv_arr[0];
        (void) nk;
        parallel_for(nv, conv_kv8_v_band);
        parallel_for((int64_t) N_KV * NKT * HEAD_DIM * KT, conv_kv8_band);
    }
}

static void prefill_release(void) {
    // -MEGA_ATTN4 keeps everything the four-row attention needs. After prefill the
    // int8 K/V caches cover every position in the prompt, so a four-row round at
    // 10K is exactly the prefill attention with the row chunk replaced by the four
    // candidates. That is what a dspark verification round costs, and measuring it
    // here reuses machinery that is already proven instead of writing a second
    // attention kernel for decode.
    if (g_attn4) return;
    // The int8 weight copy is prefill-only: decode reads the bit-packed blob.
    // Freeing it before generation hands 2.2 GB back to the page cache and
    // stops the two weight streams from evicting each other.
    if (g_blob8) {
        for (int L = 0; L < N_LAYER; ++L) {
            g_wq[L].b8 = g_wk[L].b8 = g_wv[L].b8 = nullptr;
            g_wov[L].b8 = g_wg[L].b8 = g_wu[L].b8 = g_wd[L].b8 = nullptr;
        }
        g_wo.b8 = nullptr;
        free(g_blob8);
        g_blob8 = nullptr;
        g_i8w = 0;
        printf("prefill: int8 weight copy released\n");
        // Now that the big temporary is gone, put the decode working set back
        // on 2 MB pages: the weight blob, the two KV caches, and the int8 KV.
        report_huge("before collapse");
        huge_collapse(g_wblob, g_wblob_sz);
        huge_collapse(g_kdim, (size_t) N_LAYER * K_LAYER_STRIDE * 2);
        huge_collapse(g_vpos, (size_t) N_LAYER * KV_LAYER_STRIDE * 2);
        huge_collapse(g_k8d, (size_t) N_LAYER * K_LAYER_STRIDE);
        huge_collapse(g_v8d, (size_t) N_LAYER * KV_LAYER_STRIDE);
        report_huge("after collapse");
    }
    free(g_pfa);     g_pfa = nullptr;    free(g_pfx);     g_pfx = nullptr;
    free(g_pfxa);    g_pfxa = nullptr;
    free(g_pfgate);  g_pfgate = nullptr;
    free(g_pfup);    g_pfup = nullptr;
    free(g_pfk);     g_pfk = nullptr;
    free(g_pfv);     g_pfv = nullptr;
    free(g_pq);      g_pq = nullptr;
    free(g_pfa_out); g_pfa_out = nullptr;
    free(g_k8i);     g_k8i = nullptr;
    free(g_v8i);     g_v8i = nullptr;
    free(g_q8i);     g_q8i = nullptr;
    free(g_psa16);   g_psa16 = nullptr;
    free(g_q8isc);   g_q8isc = nullptr;
    free(g_tslot);   g_tslot = nullptr;
    free(g_tile);    g_tile = nullptr;
    g_pfa_i8 = 0;
    g_tile_n = 0;
    g_pq_rows = 0;
}

// Same accumulation shape as rmsnorm_par (double partial sums) so prefill and
// decode agree on the residual stream bit for bit where it matters. Rows are
// handed to the thread pool: at 2048 rows a serial pass over 2048x2048 floats
// twice per layer is tens of seconds, which is not a detail.
static float * rn_rows_dst;
static const float * rn_rows_src;
static const float * rn_rows_src_c;
static const float * rn_rows_w;
static int64_t rn_rows_n;
static void rn_rows_band(int64_t lo, int64_t hi) {
    const float * src = rn_rows_src; float * dst = rn_rows_dst;
    const float * w = rn_rows_w; (void) rn_rows_n;
    for (int64_t r = lo; r < hi; ++r) {
        const float * x = src + r * N_EMBD;
        float * y = dst + r * N_EMBD;
        double ss = 0;
        for (int i = 0; i < N_EMBD; ++i) ss += (double) x[i] * x[i];
        const float rn = 1.0f / sqrtf((float) (ss / N_EMBD) + RMS_EPS);
        for (int i = 0; i < N_EMBD; ++i) y[i] = x[i] * rn * w[i];
    }
}
static void rmsnorm_rows(float * dst, const float * src, const float * w, int64_t rows) {
    rn_rows_dst = dst; rn_rows_src = src; rn_rows_w = w; rn_rows_n = rows;
    if (g_eln) { int wa, wx; el_split(g_eln, &wa, &wx); parallel_for_w(rows, rn_rows_band, wa, wx); }
    else parallel_for(rows, rn_rows_band);
}
static void rmsnorm_rows_serial(float * dst, const float * src, const float * w, int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = src + r * N_EMBD;
        float * y = dst + r * N_EMBD;
        double ss = 0;
        for (int i = 0; i < N_EMBD; ++i) ss += (double) x[i] * x[i];
        const float rn = 1.0f / sqrtf((float) (ss / N_EMBD) + RMS_EPS);
        for (int i = 0; i < N_EMBD; ++i) y[i] = x[i] * rn * w[i];
    }
}

static float * silu_rows_buf;
static void silu_rows_band(int64_t lo, int64_t hi) {
    for (int64_t r = lo; r < hi; ++r) {
        float * p = silu_rows_buf + r * N_FF;
        for (int i = 0; i < N_FF; ) {
            const size_t vl = __riscv_vsetvl_e32m2((size_t) (N_FF - i));
            const vfloat32m2_t x = __riscv_vle32_v_f32m2(p + i, vl);
            // silu(x) = x / (1 + exp(-x)); exp is the vectorised 2^(t*log2e)
            // decomposition rather than a libm call, which was 14.5 s at 10K.
            const vfloat32m2_t nx = __riscv_vfneg_v_f32m2(x, vl);
            const vfloat32m2_t t = __riscv_vfmul_vf_f32m2(nx, 1.4426950408889634f, vl);
            // round-to-nearest-even via the integer convert (there is no vfround)
            const vint32m2_t ni = __riscv_vfcvt_x_f_v_i32m2(t, vl);
            const vfloat32m2_t f = __riscv_vfsub_vv_f32m2(t, __riscv_vfcvt_f_x_v_f32m2(ni, vl), vl);
            vfloat32m2_t q = __riscv_vfmv_v_f_f32m2(0.0013333558f, vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.0096181291f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.0555041087f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.2402265070f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.6931471806f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
            // 2^n by writing the exponent field directly; the bias of 127 has to
            // be added before the shift, else n=0 gives 0.0 and silu(x) collapses
            // to x, which made the prefill FFN 3.4x too large.
            const vint32m2_t bits = __riscv_vsll_vx_i32m2(
                __riscv_vadd_vx_i32m2(ni, 127, vl), 23, vl);
            q = __riscv_vfmul_vv_f32m2(q, __riscv_vreinterpret_v_i32m2_f32m2(bits), vl);
            const vfloat32m2_t den = __riscv_vfadd_vf_f32m2(q, 1.0f, vl);
            __riscv_vse32_v_f32m2(p + i, __riscv_vfdiv_vv_f32m2(x, den, vl), vl);
            i += (int) vl;
        }
    }
}
static void silu_rows(float * g, int64_t rows) {
    silu_rows_buf = g;
    parallel_for(rows, silu_rows_band);
}

// The FFN is silu(gate) * up, and the up half of the pair has to be folded in
// before the down projection. This used to multiply gate by up without the
// sigmoid at all, so prefill ran a plain gated-linear unit; layer 0 came out
// 3.4x the reference. silu and the multiply share one pass over the row, so
// the gate value is only loaded and stored once.
static const float * silu_mul_up;
static void silu_mul_band(int64_t lo, int64_t hi) {
    for (int64_t r = lo; r < hi; ++r) {
        float * p = silu_rows_buf + r * N_FF;
        const float * u = silu_mul_up + r * N_FF;
        for (int i = 0; i < N_FF; ) {
            const size_t vl = __riscv_vsetvl_e32m2((size_t) (N_FF - i));
            const vfloat32m2_t x = __riscv_vle32_v_f32m2(p + i, vl);
            const vfloat32m2_t t = __riscv_vfmul_vf_f32m2(
                __riscv_vfneg_v_f32m2(x, vl), 1.4426950408889634f, vl);
            const vint32m2_t ni = __riscv_vfcvt_x_f_v_i32m2(t, vl);
            const vfloat32m2_t f = __riscv_vfsub_vv_f32m2(
                t, __riscv_vfcvt_f_x_v_f32m2(ni, vl), vl);
            vfloat32m2_t q = __riscv_vfmv_v_f_f32m2(0.0013333558f, vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.0096181291f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.0555041087f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.2402265070f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(0.6931471806f, vl), vl);
            q = __riscv_vfmadd_vv_f32m2(q, f, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
            q = __riscv_vfmul_vv_f32m2(q, __riscv_vreinterpret_v_i32m2_f32m2(
                __riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(ni, 127, vl), 23, vl)), vl);
            const vfloat32m2_t s = __riscv_vfdiv_vv_f32m2(
                x, __riscv_vfadd_vf_f32m2(q, 1.0f, vl), vl);
            __riscv_vse32_v_f32m2(p + i,
                __riscv_vfmul_vv_f32m2(s, __riscv_vle32_v_f32m2(u + i, vl), vl), vl);
            i += (int) vl;
        }
    }
}
static void silu_mul_rows(float * g, const float * up, int64_t rows) {
    silu_rows_buf = g;
    silu_mul_up = up;
    if (g_els) { int wa, wx; el_split(g_els, &wa, &wx); parallel_for_w(rows, silu_mul_band, wa, wx); }
    else parallel_for(rows, silu_mul_band);
}

// Layer-by-layer probe: writes one row of the residual stream so the two paths
// (chunked prefill and single-token decode) can be compared numerically instead
// of by generated ids, which only says "same" or "different".
// Scalar Q4_0 dot for one output row: the reference the two GEMM paths must
// agree with. Writes `n` outputs = W(n x k) . x(k).
static void probe_ref_gemm(float * out, const uint8_t * w, const float * x, int64_t n, int64_t k) {
    const int64_t nb = k / 32, rs = nb * 18;
    for (int64_t r = 0; r < n; ++r) {
        const uint8_t * row = w + r * rs;
        double acc = 0;
        for (int64_t b = 0; b < nb; ++b) {
            _Float16 sc; memcpy(&sc, row + b * 18, 2);
            const uint8_t * q = row + b * 18 + 2;
            for (int j = 0; j < 16; ++j) {
                acc += (double) sc * ((double) ((int) (q[j] & 0x0F) - 8) * (double) x[b * 32 + j]
                                    + (double) ((int) (q[j] >> 4) - 8) * (double) x[b * 32 + 16 + j]);
            }
        }
        out[r] = (float) acc;
    }
}

static void dump_n(const char * tag, int layer, char when, const float * row, int64_t n) {
    char path[128];
    snprintf(path, sizeof path, "/tmp/%s_%02d%c.bin", tag, layer, when);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    size_t put = 0;
    const size_t bytes = (size_t) n * 4;
    while (put < bytes) { ssize_t w = write(fd, (const uint8_t *) row + put, bytes - put); if (w <= 0) break; put += (size_t) w; }
    close(fd);
}
static void dump_row(const char * tag, int layer, char when, const float * row) {
    char path[128];
    snprintf(path, sizeof path, "/tmp/%s_%02d%c.bin", tag, layer, when);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    size_t put = 0;
    const size_t bytes = (size_t) N_EMBD * 4;
    while (put < bytes) { ssize_t w = write(fd, (const uint8_t *) row + put, bytes - put); if (w <= 0) break; put += (size_t) w; }
    close(fd);
}

static float * add_dst; static const float * add_src;
static void add_rows_band(int64_t lo, int64_t hi) {
    for (int64_t r = lo; r < hi; ++r) {
        float * d = add_dst + r * N_EMBD;
        const float * s = add_src + r * N_EMBD;
        for (int i = 0; i < N_EMBD; ) {
            const size_t vl = __riscv_vsetvl_e32m2((size_t) (N_EMBD - i));
            __riscv_vse32_v_f32m2(d + i,
                __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(d + i, vl),
                                       __riscv_vle32_v_f32m2(s + i, vl), vl), vl);
            i += (int) vl;
        }
    }
}
static void add_rows(float * dst, const float * src, int64_t rows) {
    add_dst = dst; add_src = src;
    if (g_eln) { int wa, wx; el_split(g_eln, &wa, &wx); parallel_for_w(rows, add_rows_band, wa, wx); }
    else parallel_for(rows, add_rows_band);
}

// Rotates one row's Q and K at its own position. Half-split pairing, the llama
// convention: (x[i], x[i + HEAD_DIM/2]) share an angle. Q also carries the
// 1/sqrt(head_dim) factor, because the caller packs it straight into the int8
// attention operand.
static void prefill_qk_row(float * q, float * k, int64_t pos) {
    const float * cs = g_rope_tab + pos * 128;
    const float inv = 1.0f / sqrtf((float) HEAD_DIM);
    // Same helper as single-row and four-row decode. q carries 1/sqrt(head_dim) so
    // the int8 attention operand is pre-scaled.
    for (int h = 0; h < N_HEAD; ++h) rope_head_norm(q + h * HEAD_DIM, cs, inv);
    for (int h = 0; h < N_KV; ++h) rope_head_norm(k + h * HEAD_DIM, cs, 1.0f);
}

// The chunk's rotated K goes into the tiled f16 cache that attention reads, and
// V into the position-major one. Both are the arrays decode uses as well.
static void store_kv_band(int L, int64_t base, int64_t lo, int64_t hi) {
    // V is position-major already, so it is a straight f32->f16 copy per position.
    for (int64_t r = lo; r < hi; ++r) {
        const int64_t p = base + r;
        const float * v = g_pfv + r * (N_KV * HEAD_DIM);
        _Float16 * vb = g_vpos + (int64_t) L * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM;
        for (int i = 0; i < N_KV * HEAD_DIM; ) {
            const size_t vl = __riscv_vsetvl_e32m1((size_t) (N_KV * HEAD_DIM - i));
            __riscv_vse16_v_f16mf2(vb + i,
                __riscv_vfncvt_f_f_w_f16mf2(__riscv_vle32_v_f32m1(v + i, vl), vl), vl);
            i += (int) vl;
        }
    }
    // K lives in a [dim][pos-in-tile] tile, so the old per-position store wrote one
    // f16 every 64 bytes: 128 scattered lines per position. Walking whole tiles
    // instead makes every store a contiguous 32-position run, and the gather from
    // the row buffer is a single strided vector load per dim.
    const ptrdiff_t kstep = (ptrdiff_t) (N_KV * HEAD_DIM * 4);   // bytes between rows
    for (int64_t r = lo; r < hi; ) {
        const int64_t p = base + r;
        const int pblk = (int) (p / KT), pi = (int) (p % KT);
        const int n = (int) ((KT - pi < hi - r) ? (KT - pi) : (hi - r));
        for (int kv = 0; kv < N_KV; ++kv) {
            _Float16 * kb = ktile(L, kv, pblk) + pi;
            const float * src = g_pfk + r * (N_KV * HEAD_DIM) + kv * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
                const vfloat32m1_t vv = __riscv_vlse32_v_f32m1(src + d, kstep, vl);
                __riscv_vse16_v_f16mf2(kb + (int64_t) d * KT,
                    __riscv_vfncvt_f_f_w_f16mf2(vv, vl), vl);
            }
        }
        r += n;
    }
}

// RoPE and the KV cache store were serial passes over the chunk: 0.95 s and
// 1.13 s at 10K, both independent per row. One band does its own rows' rotation
// and then its own rows' cache stores, so the two passes collapse into one.
static int g_pfq_L; static int64_t g_pfq_base;
static void qk_band(int64_t lo, int64_t hi) {
    for (int64_t r = lo; r < hi; ++r)
        prefill_qk_row(g_pq + r * (N_HEAD * HEAD_DIM),
                       g_pfk + r * (N_KV * HEAD_DIM), g_pfq_base + r);
    store_kv_band(g_pfq_L, g_pfq_base, lo, hi);
}
static void prefill_qk_store(int L, int64_t base, int64_t rows) {
    g_pfq_L = L; g_pfq_base = base;
    parallel_for(rows, qk_band);
}

// Dumps the cache in the position-major layout the decode engine reads:
// [layer][pos][kv][dim], f16, stride MAXCTX.
static bool export_kv(int64_t npos) {
    const size_t per = (size_t) MAXCTX * N_KV * HEAD_DIM * 2;
    const size_t tot = (size_t) N_LAYER * per;
    const char * kp = work_path("kv_k.bin");
    const char * vp = work_path("kv_v.bin");
    int fk = open(kp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int fv = open(vp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fk < 0 || fv < 0) { perror("kv open"); return false; }
    uint8_t * line = (uint8_t *) aligned_alloc(64, per);
    if (!line) return false;
    for (int L = 0; L < N_LAYER; ++L) {
        // K: tiled -> position-major
        memset(line, 0, per);
        _Float16 * out = (_Float16 *) line;
        for (int64_t p = 0; p < npos; ++p) {
            const int pblk = (int) (p / KT), pi = (int) (p % KT);
            for (int kv = 0; kv < N_KV; ++kv) {
                const _Float16 * kb = ktile(L, kv, pblk);
                for (int d = 0; d < HEAD_DIM; ++d)
                    out[p * N_KV * HEAD_DIM + kv * HEAD_DIM + d] = kb[(int64_t) d * KT + pi];
            }
        }
        if (write(fk, line, per) != (ssize_t) per) { perror("kv write k"); return false; }
        // V is already position-major
        const _Float16 * vsrc = g_vpos + (int64_t) L * KV_LAYER_STRIDE;
        memset(line, 0, per);
        memcpy(line, vsrc, (size_t) npos * N_KV * HEAD_DIM * 2);
        if (write(fv, line, per) != (ssize_t) per) { perror("kv write v"); return false; }
    }
    close(fk); close(fv);
    free(line);
    printf("kv: %.0f MB x2 -> %s, %s (%lld positions)\n",
           tot / 1e6, kp, vp, (long long) npos);
    return true;
}

static bool real_prefill(const int32_t * toks, int64_t npref, int64_t R) {
    if (!g_i8w) { fprintf(stderr, "real prefill needs the int8 weight copy (-I)\n"); return false; }
    if (g_src_type != 2 && !g_pf_vendor && !g_bit3 && !g_bit2) {
        // The vmadot prefill path reads the int8 copy, which is only built for a
        // Q4_0 source. A K-quant file has to go through the vendor K256 kernel,
        // which reads the compact blob directly.
        fprintf(stderr, "source type %d needs vendor prefill (MEGA_VENDOR=2)\n", g_src_type);
        return false;
    }
    // The X100 softmax threads are already spinning by the time prefill starts, so
    // the elementwise passes can take sixteen bands instead of eight.
    g_el_on = g_n_sm == 8 && !getenv("MEGA_NOEL");
    const int64_t NP = npref;
    g_a8_stride = (N_FF / 16) * 64 + (N_FF / A_BLK) * 16 + 64;
    // The vendor path lays a 4-row group out as [scale x4][sum x4][int8 4x32] per
    // 32 k-elements, which is a wider record than the vmadot one.
    if (g_pf_vendor) g_a8_stride = (N_FF / 32) * 152;
    if (g_pf_vendor == 2) g_a8_stride = (N_FF / 256) * 1160;   // K256 superblocks
    const int64_t mgmax = (R + 63) / 64 * 16;          // groups, rounded like the chunk
    const size_t pfa_bytes = (size_t) 3 * mgmax * (size_t) g_a8_stride + (4 << 20);
    g_pfa     = (uint8_t *) hugemem(pfa_bytes);
    g_pfx     = (float *) hugemem((size_t) R * N_EMBD * 4);
    g_pfxa    = (float *) hugemem((size_t) R * N_EMBD * 4);
    g_pfgate  = (float *) hugemem((size_t) R * N_FF * 4);
    g_pfup    = (float *) hugemem((size_t) R * N_FF * 4);
    g_pfk     = (float *) hugemem((size_t) R * N_KV * HEAD_DIM * 4);
    g_pfv     = (float *) hugemem((size_t) R * N_KV * HEAD_DIM * 4);
    g_pq      = (float *) hugemem((size_t) R * N_HEAD * HEAD_DIM * 4);
    g_pfa_out = (float *) hugemem((size_t) R * N_HEAD * HEAD_DIM * 4);
    // A run that turns into a four-row decode keeps these planes and appends one
    // position per row, so cover the whole context from the start: a position's
    // block index is part of the plane address, and growing it later would move
    // every byte already written.
    const int64_t npz = g_attn4 ? MAXCTX : NP;
    g_k8i_npb_cap   = (npz + 63) / 64;
    g_v8i_npb16_cap = (npz + 15) / 16;
    g_k8i     = (int8_t *) hugemem((size_t) N_KV * ((npz + 63) / 64) * 8192 + (2 << 20));
    g_v8i     = (int8_t *) hugemem((size_t) N_KV * ((npz + 15) / 16) * 2048 + (2 << 20));
    g_q8i     = (int8_t *) hugemem((size_t) N_HEAD * ((R + 3) / 4) * 512 + (2 << 20));
    g_psa16   = (int16_t *) hugemem((size_t) N_GPU * sizeof(int16_t) * SC_TSTRIDE);
    g_q8isc   = (float *) aligned_alloc(64, (size_t) R * N_HEAD * 4);
    if (g_attn4) {
        // Prefill converts one layer into the shared planes, attends, then converts
        // the next layer over the top. A decode round reads every layer's plane, so
        // each layer gets its own: 42 x 6.3 MB at 12K, against the 528 MB f16 cache
        // those planes are the int8 copy of.
        g_k8i_lsz = (int64_t) N_KV * ((npz + 63) / 64) * 8192;
        g_v8i_lsz = (int64_t) N_KV * ((npz + 15) / 16) * 2048;
        const size_t ktot = (size_t) N_LAYER * g_k8i_lsz + (2 << 20);
        const size_t vtot = (size_t) N_LAYER * g_v8i_lsz + (2 << 20);
        g_k8i_l = (int8_t *) hugemem(ktot);
        g_v8i_l = (int8_t *) hugemem(vtot);
        if (!g_k8i_l || !g_v8i_l) { fprintf(stderr, "oom per-layer int8 kv\n"); return false; }
        use_huge(g_k8i_l, ktot);
        use_huge(g_v8i_l, vtot);
        printf("prefill: per-layer int8 kv %.0f MB (four-row decode)\n",
               (double) (ktot + vtot) / 1e6);
    }
    if (!g_pfa || !g_pfx || !g_pfxa || !g_pfgate || !g_pfup || !g_pfk || !g_pfv ||
        !g_pq || !g_pfa_out || !g_k8i || !g_v8i || !g_q8i || !g_psa16 || !g_q8isc) {
        fprintf(stderr, "oom prefill buffers\n");
        return false;
    }
    use_huge(g_pfa, pfa_bytes);
    use_huge(g_k8i, (size_t) N_KV * ((npz + 63) / 64) * 8192 + (2 << 20));
    use_huge(g_v8i, (size_t) N_KV * ((npz + 15) / 16) * 2048 + (2 << 20));
    use_huge(g_psa16, (size_t) N_GPU * sizeof(int16_t) * SC_TSTRIDE);
    g_pfa_i8 = 1;
    g_pfa_rows = R;              // pfa_pipe_alloc derives the tile count from this
    if (g_pipe) pfa_pipe_alloc();

    double t_norm=0,t_quant=0,t_gemm=0,t_attn=0,t_conv=0,t_misc=0,t_silu=0,t_packq=0;
    double t_rope=0,t_kvst=0,t_add=0,t_zero=0;   // TEMP detail
    double t_gqkv=0,t_go=0,t_ggu=0,t_gd=0;
    const double t0 = now_s();
    for (int64_t base = 0; base < npref; base += R) {
        const int64_t cm = (npref - base < R) ? (npref - base) : R;
        // attention tiles are whole 64-row groups, and the m4 GEMM wants whole
        // four-row groups; 64 covers both
        const int64_t C = (cm + 63) / 64 * 64;
        g_pf_rows = C;
        if (g_x100) x100_alloc();

        for (int64_t r = 0; r < cm; ++r) {
            dequant_row_t(g_pfx + r * N_EMBD,
                        g_emb->data + (int64_t) toks[base + r] * src_row_bytes_t(N_EMBD, (int) g_emb->type),
                        N_EMBD, (int) g_emb->type);
        }
        if (getenv("MEGA_DUMP") && cm == 1) dump_row("pf", 0, 'e', g_pfx);
        for (int64_t r = cm; r < C; ++r) memset(g_pfx + r * N_EMBD, 0, (size_t) N_EMBD * 4);

        for (int L = 0; L < N_LAYER; ++L) {
            double _a=now_s();
            rmsnorm_rows(g_pfxa, g_pfx, g_attn_norm[L], C);
            t_norm+=now_s()-_a; _a=now_s();
            g_x8_want = g_x100;
            g_pfa_jslot[1] = g_pfa_jslot[2] = 0;
            pf_quant(g_pfxa, C, N_EMBD, 0);
            g_x8_want = 0;
            t_quant+=now_s()-_a; _a=now_s();
            for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 3;
            g_pfj[0] = { g_pf_vendor ? g_wq[L].b : g_wq[L].b8, g_wq[L].k, g_wq[L].n, g_pq,  N_HEAD * HEAD_DIM, 0 };
            g_pfj[1] = { g_pf_vendor ? g_wk[L].b : g_wk[L].b8, g_wk[L].k, g_wk[L].n, g_pfk, N_KV * HEAD_DIM, 0 };
            g_pfj[2] = { g_pf_vendor ? g_wv[L].b : g_wv[L].b8, g_wv[L].k, g_wv[L].n, g_pfv, N_KV * HEAD_DIM, 0 };
            if (g_x100) {
                const uint8_t * xs[3] = { g_wq[L].b8, g_wk[L].b8, g_wv[L].b8 };
                x100_begin(0, g_pfj, 3, xs);
            }
            if (g_pf_vendor) run(task_pf_gemm); else run(task_pf_gemm_i8);
            if (g_x100) x100_wait();
            { const double _g=now_s()-_a; t_gemm+=_g; t_gqkv+=_g; } _a=now_s();

            double _r0=now_s();
            if (L == 0 && getenv("MEGA_PROBE_REF")) {
                static float rv[N_KV * HEAD_DIM];
                const uint8_t * wv = (const uint8_t *) find("blk.0.attn_v.weight")->data;
                probe_ref_gemm(rv, wv, g_pfxa, N_KV * HEAD_DIM, N_EMBD);
                // same dot, but on the activation quantised two ways: one scale for the
                // whole row (the prefill layout), one per 32-block (what decode uses)
                static float ar[N_EMBD], a32[N_EMBD];
                float mx = 0;
                for (int i = 0; i < N_EMBD; ++i) mx = fmaxf(mx, fabsf(g_pfxa[i]));
                const float sr = mx > 0 ? mx / 127.0f : 1.0f;
                for (int i = 0; i < N_EMBD; ++i)
                    ar[i] = sr * (float) (int) lrintf(g_pfxa[i] / sr);
                for (int b = 0; b < N_EMBD / 32; ++b) {
                    float m2 = 0;
                    for (int i = 0; i < 32; ++i) m2 = fmaxf(m2, fabsf(g_pfxa[b * 32 + i]));
                    const float s2 = m2 > 0 ? m2 / 127.0f : 1.0f;
                    for (int i = 0; i < 32; ++i)
                        a32[b * 32 + i] = s2 * (float) (int) lrintf(g_pfxa[b * 32 + i] / s2);
                }
                static float rv_row[N_KV * HEAD_DIM], rv_blk[N_KV * HEAD_DIM];
                probe_ref_gemm(rv_row, wv, ar, N_KV * HEAD_DIM, N_EMBD);
                probe_ref_gemm(rv_blk, wv, a32, N_KV * HEAD_DIM, N_EMBD);
                printf("ref f32A V %.5f %.5f   rowA %.5f %.5f   blkA %.5f %.5f\n",
                       rv[0], rv[1], rv_row[0], rv_row[1], rv_blk[0], rv_blk[1]);
                printf("pf int8  V %.5f %.5f   (rowA vs blkA spread %.5f)\n",
                       g_pfv[0], g_pfv[1], rv_row[0] - rv_blk[0]);
                {   // would a clipped abs-max scale buy back the per-row error?
                    float rms = 0;
                    for (int i = 0; i < N_EMBD; ++i) rms += g_pfxa[i] * g_pfxa[i];
                    rms = sqrtf(rms / N_EMBD);
                    static float aq[N_EMBD], rv2[N_KV * HEAD_DIM];
                    printf("row rms %.4f max %.4f\n", rms, mx);
                    static const float kks[7] = { 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 16.0f };
                    for (int ki = 0; ki < 7; ++ki) {
                        const float kk = kks[ki];
                        const float lim = fminf(mx, kk * rms);
                        const float sc = lim / 127.0f;
                        for (int i = 0; i < N_EMBD; ++i) {
                            float q = g_pfxa[i] / sc;
                            q = fmaxf(-127.0f, fminf(127.0f, q));
                            aq[i] = sc * (float) (int) lrintf(q);
                        }
                        probe_ref_gemm(rv2, wv, aq, N_KV * HEAD_DIM, N_EMBD);
                        printf("  clip %.1f rms (lim %.3f): V0 %.5f err %.5f\n",
                               kk, lim, rv2[0], rv2[0] - rv[0]);
                    }
                    printf("  blkA err %.5f   rowA(max) err %.5f\n", rv_blk[0] - rv[0], rv_row[0] - rv[0]);
                    static const int bss[5] = { 32, 64, 128, 256, 512 };
                    for (int bi = 0; bi < 5; ++bi) {
                        const int bs = bss[bi];
                        static float aq2[N_EMBD];
                        for (int b0 = 0; b0 < N_EMBD; b0 += bs) {
                            float m3 = 0;
                            for (int i = 0; i < bs; ++i) m3 = fmaxf(m3, fabsf(g_pfxa[b0 + i]));
                            const float s3 = m3 > 0 ? m3 / 127.0f : 1.0f;
                            for (int i = 0; i < bs; ++i)
                                aq2[b0 + i] = s3 * (float) (int) lrintf(g_pfxa[b0 + i] / s3);
                        }
                        probe_ref_gemm(rv2, wv, aq2, N_KV * HEAD_DIM, N_EMBD);
                        printf("  block %3d: V0 %.5f err %.5f\n", bs, rv2[0], rv2[0] - rv[0]);
                    }
                }
            }
            prefill_qk_store(L, base, cm);
            t_rope+=now_s()-_r0;

            t_misc+=now_s()-_a; _a=now_s();
            if (getenv("MEGA_ZV")) memset(g_v8i, 0, (size_t) N_KV * ((base + cm + 15) / 16) * 2048);
            pfa_convert_kv_i8(L, base + cm);
            // Prefill's per-layer max-abs scales are the ones decode has to keep using,
            // and the f16 bytes are hot here, so the decode int8 cache is filled in the
            // same pass instead of a second sweep over 440 MB at the end. Only the last
            // chunk counts: its scale is the max over the whole prefix, so any earlier
            // chunk would write a cache the next chunk immediately overwrites.
            if (g_k8dec && base + cm >= npref) {
                for (int kv = 0; kv < N_KV; ++kv) {
                    g_k8dsc[L][kv] = g_k8isc[kv];
                    g_v8dsc[L][kv] = g_v8isc[kv];
                }
                g_layer = L;
                for (int kv = 0; kv < N_KV; ++kv) {
                    g_cv8_ik_arr[kv] = g_k8isc[kv] > 0 ? 1.0f / g_k8isc[kv] : 1.0f;
                    g_cv8_iv_arr[kv] = g_v8isc[kv] > 0 ? 1.0f / g_v8isc[kv] : 1.0f;
                }
                g_cv8_ik = g_cv8_ik_arr[0];
                g_cv8_iv = g_cv8_iv_arr[0];
                parallel_for((int64_t) (base + cm) * N_KV * HEAD_DIM, conv_kv8_v_band);
                parallel_for((int64_t) N_KV * NKT * HEAD_DIM * KT, conv_kv8_band);
            }
            t_conv+=now_s()-_a; _a=now_s();
            g_pfa_ctx = base;
            g_pfa_rows = C;
            if (L == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/pf_q.bin", "wb");
                fwrite(g_pq, 4, (size_t) (C * N_HEAD * HEAD_DIM), fp); fclose(fp);
            }
            double _p=now_s();
            pfa_pack_q(C);
            t_packq+=now_s()-_p; _p=now_s();
            pfa_run(L);   // re-derives the tile count from g_pfa_rows
            t_attn+=now_s()-_p; _a=now_s();
            if (getenv("MEGA_PROBE")) printf("probe L%d v8sc %.6g %.6g k8sc %.6g %.6g\n",
                                             L, g_v8isc[0], g_v8isc[1], g_k8isc[0], g_k8isc[1]);
            // The last chunk's attention output at a chosen layer: four real prompt
            // positions, so a four-row decode round over those same positions has an
            // exact reference produced by the same int8 kernel.
            if (const char * pf = getenv("MEGA_PFDUMP")) {
                if (base + cm >= npref && L == atoi(pf)) {
                    FILE * fp = fopen("/tmp/pf_last.bin", "wb");
                    if (fp) { fwrite(g_pfa_out, 4, (size_t) (C * N_EMBD), fp); fclose(fp); }
                    fp = fopen("/tmp/pf_q.bin", "wb");
                    if (fp) { fwrite(g_pq, 4, (size_t) (C * N_EMBD), fp); fclose(fp); }
                    fp = fopen("/tmp/pf_w8.bin", "wb");   // group 66 = rows 264..267
                    if (fp) {
                        fwrite(g_q8i + 66 * 512, 1, 512, fp);
                        fclose(fp);
                    }
                    printf("pf_last: L=%d base=%lld cm=%lld C=%lld\n", L,
                           (long long) base, (long long) cm, (long long) C);
                    unsetenv("MEGA_PFDUMP");
                }
            }
            if (L == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/pf_oall.bin", "wb");
                fwrite(g_pfa_out, 4, (size_t) (C * N_EMBD), fp); fclose(fp);
            }
            if (getenv("MEGA_DUMP")) {
                dump_row("pf", L, 'o', g_pfa_out + (int64_t) (cm - 1) * N_EMBD);
                dump_n("pf", L, 'v', g_pfv, N_KV * HEAD_DIM);
                // what the f16 cache actually got for the last row
                static float vc[N_KV * HEAD_DIM];
                const int64_t vlast = g_pos - 1;
                for (int i = 0; i < N_KV * HEAD_DIM; ++i)
                    vc[i] = (float) g_vpos[(int64_t) L * KV_LAYER_STRIDE + (vlast % MAXCTX) * N_KV * HEAD_DIM + i];
                dump_n("pf", L, 'w', vc, N_KV * HEAD_DIM);
            }

            { double _z=now_s();
            memset(g_pfa_out + cm * N_EMBD, 0, (size_t) (C - cm) * N_EMBD * 4);
            t_zero+=now_s()-_z; }
            g_x8_want = g_x100;
            pf_quant(g_pfa_out, C, N_EMBD, 0);
            g_x8_want = 0;
            for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 1;
            g_pfj[0] = { g_pf_vendor ? g_wov[L].b : g_wov[L].b8, g_wov[L].k, g_wov[L].n, g_pfxa, N_EMBD, 0 };
            if (g_x100) {
                const uint8_t * xs[1] = { g_wov[L].b8 };
                x100_begin(1, g_pfj, 1, xs);
            }
            if (g_pf_vendor) run(task_pf_gemm); else run(task_pf_gemm_i8);
            if (g_x100) x100_wait();
            { const double _g=now_s()-_a; t_gemm+=_g; t_go+=_g; } _a=now_s();
            { double _ad=now_s(); add_rows(g_pfx, g_pfxa, C); t_add+=now_s()-_ad; }
            t_misc+=now_s()-_a; _a=now_s();
            if (getenv("MEGA_DUMP")) dump_row("pf", L, 'a', g_pfx + (int64_t) (cm - 1) * N_EMBD);

            rmsnorm_rows(g_pfxa, g_pfx, g_ffn_norm[L], C);
            if (getenv("MEGA_DUMP")) dump_row("pf", L, 'n', g_pfxa + (int64_t) (cm - 1) * N_EMBD);
            g_x8_want = g_x100;
            g_pfa_jslot[1] = 0;
            pf_quant(g_pfxa, C, N_EMBD, 0);
            g_x8_want = 0;
            for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 2;
            g_pfj[0] = { g_pf_vendor ? g_wg[L].b : g_wg[L].b8, g_wg[L].k, g_wg[L].n, g_pfgate, N_FF, 0 };
            g_pfj[1] = { g_pf_vendor ? g_wu[L].b : g_wu[L].b8, g_wu[L].k, g_wu[L].n, g_pfup,   N_FF, 0 };
            if (g_x100) {
                const uint8_t * xs[2] = { g_wg[L].b8, g_wu[L].b8 };
                x100_begin(2, g_pfj, 2, xs);
            }
            if (g_pf_vendor) run(task_pf_gemm); else run(task_pf_gemm_i8);
            if (g_x100) x100_wait();
            { const double _g=now_s()-_a; t_gemm+=_g; t_ggu+=_g; } _a=now_s();

            if (L == 0 && getenv("MEGA_PROBE")) printf("pfL0 gate/up: %.5f %.5f %.5f | %.5f %.5f\n",
                                                       g_pfgate[0], g_pfgate[1], g_pfgate[2], g_pfup[0], g_pfup[1]);
            if (L == 0 && getenv("MEGA_PROBE")) printf("pfL0 h: %.5f %.5f %.5f | h[6143]=%.5f\n",
                                                       g_pfgate[0], g_pfgate[1], g_pfgate[3072], g_pfgate[6143]);
            double _s=now_s();
            silu_mul_rows(g_pfgate, g_pfup, C);
            if (L == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/pf_h.bin", "wb");
                fwrite(g_pfgate + (int64_t) (C - 1) * N_FF, 4, (size_t) N_FF, fp); fclose(fp);
                fp = fopen("/tmp/pf_up.bin", "wb");
                fwrite(g_pfup + (int64_t) (C - 1) * N_FF, 4, (size_t) N_FF, fp); fclose(fp);
            }
            t_silu+=now_s()-_s; _s=now_s();
            g_x8_want = g_x100;
            pf_quant(g_pfgate, C, N_FF, 0);
            g_x8_want = 0;
            t_quant+=now_s()-_s;
            _a=now_s();   // else the next t_gemm also bills silu and this quant
            for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 1;
            g_pfj[0] = { g_pf_vendor ? g_wd[L].b : g_wd[L].b8, g_wd[L].k, g_wd[L].n, g_pfxa, N_EMBD, 0 };
            if (g_x100) {
                const uint8_t * xs[1] = { g_wd[L].b8 };
                x100_begin(3, g_pfj, 1, xs);
            }
            if (g_pf_vendor) run(task_pf_gemm); else run(task_pf_gemm_i8);
            if (g_x100) x100_wait();
            { const double _g=now_s()-_a; t_gemm+=_g; t_gd+=_g; } _a=now_s();
            if (L == 0 && getenv("MEGA_PROBE")) printf("pfL0 down: %.5f %.5f %.5f\n",
                                                       g_pfxa[0], g_pfxa[1], g_pfxa[2]);
            { double _ad=now_s(); add_rows(g_pfx, g_pfxa, C); t_add+=now_s()-_ad; }   // C rows, not C * N_EMBD: it takes a row count
            t_misc+=now_s()-_a; _a=now_s();
            if (getenv("MEGA_DUMP")) dump_row("pf", L, 'f', g_pfx + (int64_t) (cm - 1) * N_EMBD);
            if (g_ds && ds_target_slot(L) >= 0) {
                const int k = ds_target_slot(L);
                for (int64_t r = 0; r < cm; ++r)
                    memcpy(g_dshid + ((int64_t) k * g_ds_cap + r) * N_EMBD,
                           g_pfx + r * N_EMBD, (size_t) N_EMBD * 4);
            }
        }
        // Feed this chunk's fused features to the drafter's encoder, four positions
        // at a time so it goes through the same four-row kernel as everything else.
        // ~1.7 s over a 10K prompt, once per request.
        if (g_ds) {
            const double _i = now_s();
            for (int64_t r = 0; r < cm; r += DS_SLOT)
                ds_inject4(base + r, r, (int) ((cm - r) < DS_SLOT ? (cm - r) : DS_SLOT));
            g_ds_t_inj += now_s() - _i;
        }
    }
    const double dt = now_s() - t0;
    if (getenv("MEGA_KVHASH") && g_pfa_i8) {
        uint64_t h1 = 1469598103934665603ull, h2 = 1469598103934665603ull;
        const size_t n1 = (size_t) N_KV * g_k8i_npb_cap * 8192;
        const size_t n2 = (size_t) N_KV * g_v8i_npb16_cap * 2048;
        for (size_t i = 0; i < n1; ++i) { h1 ^= (uint8_t) g_k8i[i]; h1 *= 1099511628211ull; }
        for (size_t i = 0; i < n2; ++i) { h2 ^= (uint8_t) g_v8i[i]; h2 *= 1099511628211ull; }
        printf("kvhash K %016llx V %016llx\n", (unsigned long long) h1, (unsigned long long) h2);
    }
    printf("prefill: %lld tokens in %.3f s (%.2f tok/s, %lld-row chunks)\n",
           (long long) npref, dt, (double) npref / dt, (long long) R);
    {   // per-cluster work, summed over threads: these are core-seconds
        double tq=0, ts=0, tv=0;
        for (int t = 0; t < N_GPU; ++t) { tq += g_pfa_qk[(t) * PFA_TSTRIDE]; ts += g_pfa_sm[(t) * PFA_TSTRIDE]; tv += g_pfa_pv[(t) * PFA_TSTRIDE]; }
        printf("  attn work: qk %.1f  softmax %.1f  pv %.1f (core-s)\n", tq, ts, tv);
    if (g_pfa_i8 && g_pipe) {
        double tb = 0;
        for (int t = 0; t < 8; ++t) tb += g_pfa_smb[(t) * PFA_TSTRIDE];
        double mxb = 0, exb = 0;
        for (int t = 0; t < 8; ++t) { mxb += g_pfa_mxb[(t) * PFA_TSTRIDE]; exb += g_pfa_exb[(t) * PFA_TSTRIDE]; }
        printf("  x100 softmax %.1f core-s over %d (%.1f s wall if saturated)"
               "  [max %.1f exp %.1f other %.1f]\n",
               tb, g_n_sm, tb / (g_n_sm > 0 ? g_n_sm : 1), mxb, exb, tb - mxb - exb);
    }
    }
    printf("  norm %.2f  quant %.2f  silu %.2f  gemm %.2f  conv %.2f  packq %.2f  attn %.2f  misc %.2f (s)\n",
           t_norm, t_quant, t_silu, t_gemm, t_conv, t_packq, t_attn, t_misc);
    printf("  detail: rope %.2f kvst %.2f add %.2f zero %.2f (s)\n", t_rope, t_kvst, t_add, t_zero);
    printf("    gemm split: qkv %.2f  o %.2f  gate+up %.2f  down %.2f (s)\n",
           t_gqkv, t_go, t_ggu, t_gd);
    if (g_x100) printf("    x100 last: pack %.3f ms, wait %.3f ms\n",
                       g_xt_pack * 1e3, g_xt_run * 1e3);
    return export_kv(npref);
}

// Fill a KV cache of g_pfa_ctx positions, then run causal attention over
// g_pfa_rows new rows and check one row against a scalar reference.
static void bench_prefill_attn(void) {
    const int64_t ctx = g_pfa_ctx, rows = g_pfa_rows;
    printf("prefill attention: ctx=%lld rows=%lld (%d rows/group)\n",
           (long long) ctx, (long long) rows, PFA_ROWS);

    g_pq = (float *) aligned_alloc(64, (size_t) rows * N_HEAD * HEAD_DIM * 4);
    g_kf = (float *) hugemem((size_t) N_KV * HEAD_DIM * ((ctx + rows + KT - 1) / KT) * KT * 4);
    g_vf = (float *) hugemem((size_t) N_KV * HEAD_DIM * (ctx + rows) * 4);
    g_pfa_out = (float *) aligned_alloc(64, (size_t) rows * N_HEAD * HEAD_DIM * 4);
    for (int64_t i = 0; i < rows * N_HEAD * HEAD_DIM; ++i)
        g_pq[i] = sinf(0.0011f * (float) (i % 977)) * 0.6f;

    // KV for every position each query can see, all layers
    for (int L = 0; L < N_LAYER; ++L) {
        for (int kv = 0; kv < N_KV; ++kv) {
            for (int64_t p = 0; p < ctx + rows; ++p) {
                _Float16 * kb = ktile(L, kv, (int) (p / KT));
                _Float16 * vb = g_vpos + (int64_t) L * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const float val = sinf(0.00037f * (float) (p * 131 + d * 17 + kv * 7 + L));
                    kb[(int64_t) d * KT + (p % KT)] = (_Float16) val;
                    vb[d] = (_Float16) cosf(0.00029f * (float) (p * 97 + d * 23 + kv * 5 + L));
                }
            }
        }
    }

    if (g_pfa_i8) {
        const int64_t npz = ctx + rows;
        g_k8i_npb_cap   = (npz + 63) / 64;
        g_v8i_npb16_cap = (npz + 15) / 16;
        g_k8i     = (int8_t *) hugemem((size_t) N_KV * ((npz + 63) / 64) * 8192 + (2 << 20));
        g_v8i     = (int8_t *) hugemem((size_t) N_KV * ((npz + 15) / 16) * 2048 + (2 << 20));
        g_q8i     = (int8_t *) hugemem((size_t) N_HEAD * ((rows + 3) / 4) * 512 + (2 << 20));
        // PV walks the whole V range for every row group (16 passes per tile), so
        // V's physical layout decides whether those passes hit cache. Give the KV
        // planes huge pages like the weight blob already has.
        use_huge(g_k8i, (size_t) N_KV * ((npz + 63) / 64) * 8192 + (2 << 20));
        use_huge(g_v8i, (size_t) N_KV * ((npz + 15) / 16) * 2048 + (2 << 20));
        use_huge(g_q8i, (size_t) N_HEAD * ((rows + 3) / 4) * 512 + (2 << 20));
        report_huge("int8 kv");
        g_q8isc = (float *) aligned_alloc(64, (size_t) rows * N_HEAD * 4);
        if (!g_k8i || !g_v8i || !g_q8i || !g_q8isc) { fprintf(stderr, "oom attn int8\n"); return; }
        g_psa16 = (int16_t *) hugemem((size_t) N_GPU * sizeof(int16_t) * SC_TSTRIDE);
        if (!g_psa16) { fprintf(stderr, "oom scores\n"); return; }
        use_huge(g_psa16, (size_t) N_GPU * sizeof(int16_t) * SC_TSTRIDE);
        const double tq0 = now_s();
        pfa_pack_q(rows);
        printf("  int8 pack Q: %.3f s (%.1f MB)\n", now_s() - tq0,
               (double) N_HEAD * ((rows + 3) / 4) * 512 / 1e6);
    }

    if (g_pfa_i8 && g_pipe) pfa_pipe_alloc();

    for (int t = 0; t < N_GPU; ++t) g_pfa_qk[(t) * PFA_TSTRIDE] = g_pfa_sm[(t) * PFA_TSTRIDE] = g_pfa_pv[(t) * PFA_TSTRIDE] = 0;
    for (int t = 0; t < N_GPU; ++t) g_pfa_mx[(t) * PFA_TSTRIDE] = g_pfa_ex[(t) * PFA_TSTRIDE] = 0;
    for (int t = 0; t < 8; ++t) g_pfa_smb[(t) * PFA_TSTRIDE] = 0;
    const double t0 = now_s();
    for (int L = 0; L < N_LAYER; ++L) {
        if (g_pfa_i8) pfa_convert_kv_i8(L, ctx + rows);
        else          pfa_convert_kv(L, ctx + rows);
        pfa_run(L);
    }
    const double dt = now_s() - t0;
    const float first = g_pfa_out[3 * N_HEAD * HEAD_DIM];
    printf("  run1: out[0]=%.6f ksum=%.6f kmx=%.6f\n", first, g_pfa_dbg_sum[0], g_pfa_dbg_mx[0]);
    if (g_pfa_i8) pfa_convert_kv_i8(0, ctx + rows); else pfa_convert_kv(0, ctx + rows);
    pfa_run(0);
    printf("  run2: out[0]=%.6f ksum=%.6f kmx=%.6f  (repeatable=%s)\n",
           g_pfa_out[3 * N_HEAD * HEAD_DIM], g_pfa_dbg_sum[0], g_pfa_dbg_mx[0],
           first == g_pfa_out[3 * N_HEAD * HEAD_DIM] ? "yes" : "NO");

    // total QK+PV MACs actually performed
    double macs = 0;
    const int tile = g_pfa_i8 ? I8_ROWS : PFA_ROWS;
    for (int64_t qb = 0; qb + tile <= rows; qb += tile)
        for (int r = 0; r < tile; ++r) macs += (double) (ctx + qb + r + 1) * HEAD_DIM * 2.0;
    macs *= (double) N_LAYER * N_HEAD;
    double tq = 0, ts = 0, tv = 0, tmx = 0, tex = 0;
    for (int t = 0; t < N_GPU; ++t) { tq += g_pfa_qk[(t) * PFA_TSTRIDE]; ts += g_pfa_sm[(t) * PFA_TSTRIDE]; tv += g_pfa_pv[(t) * PFA_TSTRIDE]; tmx += g_pfa_mx[(t) * PFA_TSTRIDE]; tex += g_pfa_ex[(t) * PFA_TSTRIDE]; }
    printf("  %.3f s -> %.1f GMAC/s (%.1f per core)  [qk %.2f sm %.2f pv %.2f s]\n",
           dt, macs / dt / 1e9, macs / dt / 1e9 / g_participants, tq, ts, tv);
    printf("  softmax split: max %.2f s, exp+quant %.2f s, other %.2f s\n",
           tmx, tex, ts - tmx - tex);
    if (g_pfa_i8 && g_pipe) {
        double tb = 0, tmxb = 0, texb = 0;
        for (int t = 0; t < 8; ++t) { tb += g_pfa_smb[(t) * PFA_TSTRIDE]; tmxb += g_pfa_mxb[(t) * PFA_TSTRIDE]; texb += g_pfa_exb[(t) * PFA_TSTRIDE]; }
        printf("  pipeline: A-cluster qk %.2f + pv %.2f = %.2f core-s over 8; "
               "X100 softmax %.2f core-s over %d (max %.2f / exp %.2f)\n",
               tq, tv, tq + tv, tb, g_n_sm, tmxb, texb);
    }

    {   // scratch layout audit
        pfa_convert_kv(0, ctx + rows);
        const int64_t nt = ctx + rows;
        const _Float16 * k0 = ktile(0, 0, 0);
        printf("  K chk d=0 p=0..3: f32", 0);
        for (int p = 0; p < 4; ++p) printf(" %.4f", g_kf[0 * nt + p]);
        printf(" | f16");
        for (int p = 0; p < 4; ++p) printf(" %.4f", (float) k0[0 * KT + p]);
        printf("\n  K chk d=3 p=5: f32 %.4f f16 %.4f\n",
               g_kf[3 * nt + 5], (float) k0[3 * KT + 5]);
        printf("  V chk p=0 d=0..3: f32");
        for (int d = 0; d < 4; ++d) printf(" %.4f", g_vf[0 * HEAD_DIM + d]);
        printf(" | f16");
        for (int d = 0; d < 4; ++d)
            printf(" %.4f", (float) g_vpos[0 * KV_LAYER_STRIDE + 0 * N_KV * HEAD_DIM + 0 * HEAD_DIM + d]);
        printf("\n");
    }

    // scalar reference for layer 0, head 0, last row of the first group
    if (rows >= PFA_ROWS) {
        const int rr = PFA_ROWS - 1;
        const int qpos = (int) (ctx + rr);
        static float sc[MAXCTX + 64], w[MAXCTX + 64], ref[HEAD_DIM];
        const float * q = g_pq + ((int64_t) rr * N_HEAD + 0) * HEAD_DIM;
        const _Float16 * vb0 = g_vpos + 0 * KV_LAYER_STRIDE;
        double mx = -INFINITY;
        for (int p = 0; p <= qpos; ++p) {
            double s = 0;
            const _Float16 * kb = ktile(0, 0, p / KT);
            for (int d = 0; d < HEAD_DIM; ++d) s += (double) q[d] * (double) kb[(int64_t) d * KT + (p % KT)];
            sc[p] = (float) s;
            if (s > mx) mx = s;
        }
        double sum = 0;
        for (int p = 0; p <= qpos; ++p) { w[p] = expf((float) ((double) sc[p] - mx)); sum += w[p]; }
        for (int d = 0; d < HEAD_DIM; ++d) {
            double a = 0;
            for (int p = 0; p <= qpos; ++p)
                a += (double) w[p] * (double) vb0[(int64_t) p * N_KV * HEAD_DIM + 0 * HEAD_DIM + d];
            ref[d] = (float) (a / sum);
        }
        double worst = 0, peak = 0;
        const float * mine = g_pfa_out + ((int64_t) rr * N_HEAD + 0) * HEAD_DIM;
        int wd = -1;
        for (int d = 0; d < HEAD_DIM; ++d) {
            const double dd = fabs((double) mine[d] - (double) ref[d]);
            if (dd > worst) { worst = dd; wd = d; }
            peak = fmax(peak, fabs((double) ref[d]));
        }
        printf("  worst at dim %d: mine %.6g ref %.6g\n", wd, wd >= 0 ? mine[wd] : 0.0, wd >= 0 ? ref[wd] : 0.0);
        printf("  check layer0/head0/row%d: worst |diff| %.3e, peak %.3f\n", rr, worst, peak);
        printf("  mine:"); for (int d = 0; d < 4; ++d) printf(" %8.4f", mine[d]);
        printf("\n  ref :"); for (int d = 0; d < 4; ++d) printf(" %8.4f", ref[d]);
        printf("\n  sum_w=%.6f qpos=%d\n", sum, qpos);
    }

    // project to pp10240: full causal triangle over 10240 rows
    const double tri = 10240.0 * 10241.0 * 0.5 * HEAD_DIM * 2.0 * (double) N_LAYER * N_HEAD;
    const double rate = macs / dt;
    printf("  pp10240 attention projection: %.2f s at this rate (%.1f TMAC total)\n",
           tri / rate, tri / 1e12);
    free(g_pq); free(g_pfa_out); free(g_kf); free(g_vf);
    g_pq = nullptr; g_pfa_out = nullptr; g_kf = g_vf = nullptr;
}

// GEMM-only prefill benchmark: the hard ceiling for prefill, since every token
// must still be multiplied through every weight. Reports GMAC/s so it can be
// compared with the vendor path's effective rate directly.
static int g_verify = 0;
static float * ref_v;
static void bench_prefill_gemm(int64_t rows) {
    const int64_t mgroups = rows / 4;
    float * actb = (float *) hugemem((size_t) rows * 6144 * 4);
    for (int64_t i = 0; i < rows * 6144; ++i) actb[i] = sinf(0.001f * (float) (i % 4096)) * 2.0f;
    const float * act = actb;
    // scratch: rows x 6144 floats is enough for any of these GEMMs
    float * out = (float *) hugemem((size_t) rows * 6144 * 4);
    // Record stride must match what quant_a_i8d/quant_a_4row actually use, and
    // the k=6144 down projection needs the scale area on top of the payload.
    const int64_t pfa_stride = g_pf_vendor == 2 ? ((6144 / 256) * 1160)
                             : g_pf_vendor        ? ((6144 / 32) * 152)
                             : ((6144 / 16) * 64 + (6144 / A_BLK) * 16 + 512);
    uint8_t * pfa = (uint8_t *) hugemem((size_t) (rows / 4) * 3 * pfa_stride + (2 << 20));
    ref_v = (float *) hugemem((size_t) rows * 6144 * 4);
    g_pfa = pfa;
    g_pf_rows = rows;
    fprintf(stderr, "[bench] out=%p pfa=%p rows=%lld\n", (void*)out, (void*)pfa, (long long)rows);

    double t_quant = 0, t_gemm = 0;
    double macs = 0;
    const double t_q0 = now_s();
    for (int L = 0; L < N_LAYER; ++L) {
        for (int phase = 0; phase < 4; ++phase) {
            int nj = 0;
            const WRef * wr[3];
            if (phase == 0) { nj = 3; wr[0] = &g_wq[L]; wr[1] = &g_wk[L]; wr[2] = &g_wv[L]; }
            else if (phase == 1) { nj = 1; wr[0] = &g_wov[L]; }
            else if (phase == 2) { nj = 2; wr[0] = &g_wg[L]; wr[1] = &g_wu[L]; }
            else { nj = 1; wr[0] = &g_wd[L]; }
            g_pfjn = nj;
            const bool use_i8 = g_i8w && wr[0]->b8 != nullptr;
            for (int j = 0; j < nj; ++j)
                g_pfj[j] = { use_i8 ? wr[j]->b8 : wr[j]->b, wr[j]->k, wr[j]->n, out, wr[j]->n };
                    const double t0 = now_s();
            if (use_i8) {
                for (int j = 0; j < nj; ++j) quant_a_i8d(act, rows, wr[j]->k, j);
            } else {
                for (int j = 0; j < nj; ++j) quant_a_4row(act, rows, wr[j]->k, j);
            }
            if (use_i8) { for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed); }
            const double t1 = now_s();
            run(use_i8 ? task_pf_gemm_i8 : task_pf_gemm);
            const double t2 = now_s();
            t_quant += t1 - t0;
            t_gemm += t2 - t1;
            for (int j = 0; j < nj; ++j) macs += (double) rows * wr[j]->k * wr[j]->n;
        }
    }
    for (int rep = 0; rep < g_pf_reps; ++rep)
        for (int L = 0; L < N_LAYER; ++L) {
            g_pfjn = 1;
            g_pfj[0] = { g_wq[L].b, g_wq[L].k, g_wq[L].n, out, g_wq[L].n };
            run(task_pf_gemm);
        }
    if (g_pf_reps) {
        const double qmac = (double) g_pf_reps * N_LAYER * rows * g_wq[0].k * g_wq[0].n;
        printf("  [Q-only, B=%lldKB stays in L2] %.1f GMAC/s  (per core %.1f)\n",
               (long long) (g_wq[0].k * g_wq[0].n / 2 / 1024), qmac / (now_s() - t_q0) / 1e9,
               qmac / (now_s() - t_q0) / 1e9 / g_participants);
    }
    // Correctness gate for the dynamic scheduler: prefill the output with NaN and
    // confirm every element was written. This proves each row-group chunk was
    // processed, which is the one thing work stealing can get wrong.
    if (g_verify) {
        const WRef * w = &g_wg[0];
        const int64_t nn = rows * w->n;
        for (int64_t i = 0; i < nn; ++i) out[i] = __builtin_nanf("");
        g_pfjn = 1;
        g_pfj[0] = { w->b8, w->k, w->n, out, w->n };
        quant_a_i8d(act, rows, w->k, 0);
        for (int c = 0; c < NCHUNK; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
        run(task_pf_gemm_i8);
        int64_t holes = 0;
        double amax = 0, rms = 0;
        for (int64_t i = 0; i < nn; ++i) {
            if (__builtin_isnan(out[i])) { ++holes; continue; }
            amax = fmax(amax, fabs(out[i])); rms += (double) out[i] * out[i];
        }
        // and a spot-check against the exact float dot of the Q4_0 weights, so a
        // silent scale mistake cannot hide behind full coverage
        const Entry * e = find("blk.0.ffn_gate.weight");
        const int64_t rs = (w->k / 32) * 18;
        double sig = 0, noi = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c) {
                double ex = 0;
                static float row[8192];
                const uint8_t * sr = e->data + (int64_t) c * rs;
                for (int64_t b = 0; b < w->k / 32; ++b) {
                    _Float16 sc; memcpy(&sc, sr + b * 18, 2);
                    const uint8_t * q = sr + b * 18 + 2;
                    for (int j2 = 0; j2 < 16; ++j2) {
                        row[b * 32 + j2]      = (float) sc * (float) ((int) (q[j2] & 0x0F) - 8);
                        row[b * 32 + 16 + j2] = (float) sc * (float) ((int) (q[j2] >> 4) - 8);
                    }
                }
                for (int64_t i2 = 0; i2 < w->k; ++i2) ex += (double) act[r * (int64_t) w->k + i2] * (double) row[i2];
                const double got = out[r * (int64_t) w->n + c];
                sig += ex * ex; noi += (ex - got) * (ex - got);
            }
        printf("verify: %lld holes of %lld, max|out| %.4g, rms %.4g, vs exact Q4_0 dot SNR %.1f dB\n",
               (long long) holes, (long long) nn, amax, sqrt(rms / nn),
               10.0 * log10(sig / (noi > 1e-30 ? noi : 1e-30)));
    }

    const double tot = t_quant + t_gemm;
    printf("prefill GEMM-only, %lld rows: gemm %.3f s + quant %.3f s = %.3f s\n",
           (long long) rows, t_gemm, t_quant, tot);
    printf("  -> %.1f GMAC/s   (per core %.1f)   equivalent %.1f tok/s\n",
           macs / t_gemm / 1e9, macs / t_gemm / 1e9 / g_participants, rows / tot);
    printf("  (decimal check: %.2f TMAC in %.2f s)\n", macs / 1e12, t_gemm);
    free(out);
    free(pfa);
    free(actb);
}

// ------------------------------------------------------------ where files live
// Things that must survive a reboot belong on UFS, not tmpfs. /dev/shm is RAM:
// the model, the repacked caches and the binaries used to sit there, costing about
// 3.6 GB, and every recovery wiped them -- forcing a 13.6 s re-quantize and a full
// re-repack before anything could be measured.
//
// Only *storage* moves. The weights are still streamed from a RAM blob at runtime,
// because decode reads 1.26 GB per token against a 24 GB/s ceiling and UFS is an
// order of magnitude short of that. So: stored on UFS, loaded once into RAM.
static const char * work_path(const char * name) {
    static char buf[8][512];
    static unsigned which = 0;
    const char * d = getenv("MEGA_WORK");
    if (!d || !*d) d = "/home/bianbu/workspace/mega";
    char * b = buf[which++ & 7u];
    snprintf(b, sizeof buf[0], "%s/%s", d, name);
    return b;
}

#include <execinfo.h>
#include <signal.h>
// The process aborts with "stack smashing detected" after the last token, which
// means one canary-protected frame was overwritten but the SEGV-free path hides
// which one. Print the frames from the abort handler instead of guessing.
static void abort_bt(int) {
    void * bt[64];
    const int n = backtrace(bt, 64);
    backtrace_symbols_fd(bt, n, 2);
    _exit(134);
}

// Reference-locked decoding for precision checks. `-force refs/golden_10k.txt`
// walks the reference's token chain so every step scores the same prefix the
// reference scored; MEGA_STEPLOG appends one full logit row per step so those rows
// can be compared position by position.
static int   g_force_tok[4096];
static int   g_force_n = 0;
static FILE * g_steplog = nullptr;

static void force_load(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open force file %s\n", path); return; }
    // Accepts both the "  step N -> ID (val)" log format and one id per line.
    char line[512];
    while (fgets(line, sizeof line, f) && g_force_n < (int) (sizeof g_force_tok / sizeof g_force_tok[0])) {
        const char * arrow = strstr(line, "->");
        long v = arrow ? strtol(arrow + 2, nullptr, 10) : strtol(line, nullptr, 10);
        if (v > 0) g_force_tok[g_force_n++] = (int) v;
    }
    fclose(f);
    printf("force: %d reference tokens from %s\n", g_force_n, path);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    exp3_init();
    if (getenv("MEGA_BT")) signal(SIGABRT, abort_bt);
    const char * path = work_path("mc-q4o.gguf");
    int ngen = 16, prompt = 1;
    int64_t pos0 = 0;                 // first decode position (prompt length with -F)
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-force") && i + 1 < argc) force_load(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) g_kstep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-X") && i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
            g_dbg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q3")) g_bit3 = 1;
        else if (!strcmp(argv[i], "-q2")) g_bit2 = 1;
        else if (!strcmp(argv[i], "-H")) g_hpdec = 1;
        else if (!strcmp(argv[i], "-K8")) g_k8dec = 1;
        else if (!strcmp(argv[i], "-P4")) g_pvm4 = 1;
        else if (!strcmp(argv[i], "-P2")) g_pvm2 = 1;
        else if (!strcmp(argv[i], "-P") && i + 1 < argc) g_pf_rows = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-N") && i + 1 < argc) g_pf_nb = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) g_participants = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-C") && i + 1 < argc) g_cpu0 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-R") && i + 1 < argc) g_pf_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-M") && i + 1 < argc) g_pf_mt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-A") && i + 1 < argc) g_pfa_rows = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-x") && i + 1 < argc) g_pfa_ctx = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-I")) g_i8w = 1;
        else if (!strcmp(argv[i], "-E")) g_noepi = 1;
        else if (!strcmp(argv[i], "-V")) g_verify = 1;
        else if (!strcmp(argv[i], "-J")) g_pfa_i8 = 1;
        else if (!strcmp(argv[i], "-Y")) g_pfa_dbg1 = 1;
        else if (!strcmp(argv[i], "-D")) g_nostore = 1;
        else if (!strcmp(argv[i], "-0")) g_nopv = 1;
        else if (!strcmp(argv[i], "-1")) g_qk_trace = 1;
        else if (!strcmp(argv[i], "-K")) g_pipe = 1;
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) g_w8off = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z")) g_vsmall = 1;
        else if (!strcmp(argv[i], "-y")) g_hoist = 1;
        else if (!strcmp(argv[i], "-u")) g_keepv = 1;
        else if (!strcmp(argv[i], "-w")) g_keepw = 1;
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) g_vsrc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S")) g_gstyle = 1;
        else if (!strcmp(argv[i], "-Q")) g_tcmstage = 1;
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) g_pvchunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H")) g_headmajor = 1;
        else if (!strcmp(argv[i], "-L")) g_le32 = 1;
        else if (!strcmp(argv[i], "-h0")) { g_headmajor = 0; g_tilefast = 1; }
        else if (!strcmp(argv[i], "-l0")) g_le32 = 0;
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) g_pf_tile = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-F") && i + 1 < argc) g_pf_file = argv[++i];
        else if (!strcmp(argv[i], "-G") && i + 1 < argc) g_pf_chunk = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) g_exp3 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")) g_gatemode = 1;
        else if (!strcmp(argv[i], "-x100")) {
            g_x100 = 1;
            g_xchk = getenv("MEGA_XCHK") != nullptr;
            if (i + 1 < argc && argv[i + 1][0] >= '1' && argv[i + 1][0] <= '9') {
                g_x100_pct = atoi(argv[++i]);
                g_xpct_explicit = 1;
            }
        }
        else if (!strcmp(argv[i], "-pp")) g_pvprobe = 1;
        else if (!strcmp(argv[i], "-gg")) { g_gatemode = 1; g_top5 = 1; }
        else if (!strcmp(argv[i], "-W")) g_pf_vendor = 1;
        else if (!strcmp(argv[i], "-Z") && i + 1 < argc) g_win = atoi(argv[++i]);        else if (!strcmp(argv[i], "-X")) g_pf_vendor = 2;   // K256 hp m4, the default
        else if (!strcmp(argv[i], "-V0")) g_pf_vendor = 0;
    }

    if (getenv("MEGA_VENDOR") && atoi(getenv("MEGA_VENDOR"))) g_pf_vendor = atoi(getenv("MEGA_VENDOR"));
    // One logit row per decode step, for scoring against the reference's own rows.
    if (const char * sp = getenv("MEGA_STEPLOG")) {
        g_steplog = fopen(sp, "wb");
        if (!g_steplog) fprintf(stderr, "cannot open steplog %s\n", sp);
        else printf("steplog -> %s (%d f32 per step)\n", sp, VOCAB);
    }
    if (getenv("MEGA_PVPROBE")) g_pvprobe = 1;
    if (getenv("MEGA_QK2")) g_qk2 = atoi(getenv("MEGA_QK2"));
    if (getenv("MEGA_QKFUSE")) g_qkfuse = atoi(getenv("MEGA_QKFUSE"));
    if (getenv("MEGA_NOSM")) g_nosm = atoi(getenv("MEGA_NOSM"));
    if (getenv("MEGA_QKNOST")) g_qknost = atoi(getenv("MEGA_QKNOST"));
    if (getenv("MEGA_ELQ")) g_elq = atoi(getenv("MEGA_ELQ"));
    if (getenv("MEGA_ELN")) g_eln = atoi(getenv("MEGA_ELN"));
    if (getenv("MEGA_ELS")) g_els = atoi(getenv("MEGA_ELS"));
    if (getenv("MEGA_ATTN4")) g_attn4 = atoi(getenv("MEGA_ATTN4"));
    if (getenv("MEGA_M4DEC")) { g_m4dec = atoi(getenv("MEGA_M4DEC")); g_attn4 = 1; }
    if (getenv("MEGA_SPEC")) { g_m4spec = atoi(getenv("MEGA_SPEC")); g_attn4 = 1; }
    if (getenv("MEGA_SPEC_K")) g_m4spec_k = atoi(getenv("MEGA_SPEC_K"));
    if (getenv("MEGA_ATTN4P")) g_a4p = atoi(getenv("MEGA_ATTN4P"));
    if (getenv("MEGA_A4S")) g_a4s = atoi(getenv("MEGA_A4S"));
    // The vendor prefill helper for 3-bit weights is a one-row kernel, and the
    // 4-row K256 kernel it would otherwise call is 4-bit. Prefill therefore runs
    // on the int8 copy, which repack_b_i8_q3 builds from this same 3-bit blob.
    if (g_bit3 && g_pf_vendor) {
        printf("prefill: 3-bit weights have no 4-row vendor kernel, using the int8 copy\n");
        g_pf_vendor = 0;
    }
    // The prefill path is int8 attention, and that decision has to be visible
    // before the X100 softmax workers are created below.
    if (g_pf_file) {
        g_i8w = 1;
        if (!getenv("MEGA_NOI8")) {   // TEMP: isolates the f16 attention path
            g_pfa_i8 = 1;
            // MEGA_NOPIPE keeps the int8 path but runs QK+softmax+PV all on the A
            // cluster, which is the measurement that says whether handing softmax
            // to X100 is a net win or just moves the critical path.
            g_pipe = getenv("MEGA_NOPIPE") ? 0 : 1;
            g_tcmstage = 1; // -Q measured 49.71 -> 44.63 s on attn at 10K
        }
    }
    g_blk_bytes = g_bit2 ? Q2_BLK : (g_bit3 ? Q3_BLK : 576);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st); g_map_sz = (size_t) st.st_size;
    g_map = (uint8_t *) mmap(nullptr, g_map_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g_map == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);
    gguf_index();

    static char names[1 + N_LAYER * 7][64];
    struct Item { const char * name; WRef * ref; };
    static Item items[1 + N_LAYER * 7];
    int ni = 0, ii = 0;
    strcpy(names[ni], "output.weight"); items[ii++] = { names[ni++], &g_wo };
    for (int L = 0; L < N_LAYER; ++L) {
        const char * pat[7] = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
        WRef * dst[7] = { &g_wq[L], &g_wk[L], &g_wv[L], &g_wov[L], &g_wg[L], &g_wu[L], &g_wd[L] };
        for (int m = 0; m < 7; ++m) {
            snprintf(names[ni], 64, "blk.%d.%s.weight", L, pat[m]);
            items[ii++] = { names[ni++], dst[m] };
        }
    }
    // rotation-pair order: even dims first, then odd, per 128-dim head
    for (int r = 0; r < N_EMBD; ++r) {
        const int h = r / HEAD_DIM, j = r % HEAD_DIM;
        g_perm[r] = h * HEAD_DIM + (j < HEAD_DIM / 2 ? 2 * j : 2 * (j - HEAD_DIM / 2) + 1);
    }
    g_rope_tab = (float *) malloc((size_t) MAXCTX * 2 * (HEAD_DIM / 2) * sizeof(float));
    for (int pos = 0; pos < MAXCTX; ++pos)
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            const float th = (float) pos * (1.0f / powf(ROPE_BASE, (float) (2 * i) / (float) HEAD_DIM));
            g_rope_tab[(int64_t) pos * 128 + i] = cosf(th);
            g_rope_tab[(int64_t) pos * 128 + 64 + i] = sinf(th);
        }

    size_t total = 0;
    for (int i = 0; i < ii; ++i) {
        const Entry * e = find(items[i].name);
        if (!e) { fprintf(stderr, "missing %s\n", items[i].name); return 1; }
        items[i].ref->k = e->k; items[i].ref->n = e->n; items[i].ref->type = (int) e->type;
        if (i == 0) g_src_type = (int) e->type;
        // Files mix types: the output tensor is often Q6_K even when every block is
        // Q3_K. Each repack sets g_src_type from its own tensor, so this only picks
        // the one the cache name is keyed on.
        if ((int) e->type != g_src_type && i > 0 && !g_src_mixed) {
            g_src_mixed = 1;
            printf("weights: mixed source types (%s is %u, first is %d), repacking per tensor\n",
                   items[i].name, e->type, g_src_type);
        }
        total += (e->n / 32) * (g_bit2 ? (e->k / Q2_K) * Q2_BLK
                             : g_bit3 ? (e->k / Q3_K) * Q3_BLK : (e->k / 32) * 576);
    }
    if (!g_bit2 && !g_bit3 && g_src_type != 2) {
        // The plain decode blob is a permutation of Q4_0 blocks, not a requantisation,
        // so it only exists for a Q4_0 source. Anything else has to go through a packer.
        fprintf(stderr, "source type %d needs -q3 or -q2 (the Q4_0 blob is a permutation, not a converter)\n",
                g_src_type);
        return 1;
    }
    printf("weights: source type %d (%s), %s blob\n", g_src_type,
           g_src_type == 2 ? "Q4_0" : g_src_type == 10 ? "Q2_K" : g_src_type == 11 ? "Q3_K"
         : g_src_type == 8 ? "Q8_0" : g_src_type == 1 ? "F16" : g_src_type == 0 ? "F32" : "?",
           g_bit2 ? "2-bit" : g_bit3 ? "3-bit" : "Q4_0");
    const size_t tot_pad = (total + 4096 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
    // v2: the q2 packer's scale plane was wrong until probe_q2pack.cpp pinned it
    // down, and a same-named cache kept serving the broken bytes through several
    // rebuilds. Any change to a packer bumps this suffix.
    {
        // The cached blob is keyed on the source format as well: the same run with a
        // Q3_K file and with a Q4_0 file produce different bytes and the same size.
    }
    char cache_buf[64];
    snprintf(cache_buf, sizeof cache_buf, "mega-w%s-s%d.bin",
             g_bit2 ? "2v3" : g_bit3 ? "3v2" : "", g_src_type);
    const char * cache = work_path(cache_buf);
    struct stat cs;
    bool loaded = false;
    g_wblob = (uint8_t *) hugemem(tot_pad);
    if (!g_wblob) { fprintf(stderr, "oom weights\n"); return 1; }
    g_wblob_sz = tot_pad;
    use_huge(g_wblob, tot_pad);
    if (stat(cache, &cs) == 0 && (size_t) cs.st_size == total) {
        int cf = open(cache, O_RDONLY);
        size_t got = 0;
        while (got < total) { ssize_t r = read(cf, g_wblob + got, total - got); if (r <= 0) break; got += (size_t) r; }
        close(cf);
        loaded = (got == total);
    }
    if (!loaded) {
        const double t0 = now_s();
        uint8_t * p = g_wblob;
        for (int i = 0; i < ii; ++i) {
            const Entry * e = find(items[i].name);
            items[i].ref->b = p;
            g_src_type = items[i].ref->type;      // this tensor's own format
            const bool rope_pair = strstr(items[i].name, "attn_q.weight") ||
                                   strstr(items[i].name, "attn_k.weight");
            if (g_bit2) {
                repack_b_q2(p, e->data, e->n, e->k, nullptr);
                p += (e->n / 32) * (e->k / Q2_K) * Q2_BLK;
            } else if (g_bit3) {
                repack_b_q3(p, e->data, e->n, e->k, nullptr);
                p += (size_t) (e->n / 32) * (e->k / Q3_K) * Q3_BLK;
            } else {
                repack_b(p, e->data, e->n, e->k, nullptr);
                p += (size_t) (e->n / 32) * (e->k / 32) * 576;
            }
        }
        printf("weights: repacked in %.2f s (%.1f MB)\n", now_s() - t0, total / 1e6);
        int cf = open(cache, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (cf >= 0) { size_t put = 0; while (put < total) { ssize_t w = write(cf, g_wblob + put, total - put); if (w <= 0) break; put += (size_t) w; } close(cf); }
    } else {
        uint8_t * p = g_wblob;
        for (int i = 0; i < ii; ++i) {
            items[i].ref->b = p;
            p += g_bit2 ? (items[i].ref->n / 32) * (items[i].ref->k / Q2_K) * Q2_BLK
               : g_bit3 ? (items[i].ref->n / 32) * (items[i].ref->k / Q3_K) * Q3_BLK
                        : (size_t) (items[i].ref->n / 32) * (items[i].ref->k / 32) * 576;
        }
        printf("weights: %.1f MB (cached)\n", total / 1e6);
    }

    // Optional int8 copy for prefill. Kept separate from the compact format
    // because decode streams every byte and wants the smallest representation,
    // while prefill is compute bound and wants the densest instruction.
    //
    // -x100 needs it too even when the A100 side reads the compact blob: the X100
    // repacker consumes this layout, and without it the offload had no weights to
    // read at all.
    if (g_i8w && g_src_type != 2 && !g_bit3 && !g_x100) {
        // The int8 prefill copy is a re-encoding of Q4_0 rows; a K-quant source
        // skips it because the K256 vendor kernel reads the compact blob itself.
        // g_i8w stays set, since that is what gates the vendor path.
        printf("weights: int8 copy skipped (source type %d, vendor prefill reads the compact blob)\n",
               g_src_type);
    } else if (g_i8w) {
        size_t t8 = 0;
        for (int i = 0; i < ii; ++i)
            t8 += ((size_t) items[i].ref->n * items[i].ref->k + (size_t) items[i].ref->n * 16 + 4095) & ~(size_t) 4095;
        const size_t pad8 = (t8 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
        uint8_t * blob8 = (uint8_t *) hugemem(pad8);
        if (!blob8) { fprintf(stderr, "oom int8 weights\n"); return 1; }
        g_blob8 = blob8;        use_huge(blob8, pad8);
        const char * c8 = work_path(g_bit2 ? "mega-w8b-q2v3.bin"
                                    : g_bit3 ? "mega-w8b-q3v2.bin" : "mega-w8b.bin");
        struct stat cs8;
        bool l8 = false;
        if (stat(c8, &cs8) == 0 && (size_t) cs8.st_size == t8) {
            int cf = open(c8, O_RDONLY);
            size_t got = 0;
            while (got < t8) { ssize_t r = read(cf, blob8 + got, t8 - got); if (r <= 0) break; got += (size_t) r; }
            close(cf);
            l8 = (got == t8);
        }
        uint8_t * p = blob8;
        if (l8) {
            for (int i = 0; i < ii; ++i) {
                items[i].ref->b8 = p;
                p += ((size_t) items[i].ref->n * items[i].ref->k + (size_t) items[i].ref->n * 16 + 4095) & ~(size_t) 4095;
            }
            printf("weights: int8 copy %.1f MB (cached)\n", t8 / 1e6);
        } else {
            const double t0 = now_s();
            for (int i = 0; i < ii; ++i) {
                const Entry * e = find(items[i].name);
                items[i].ref->b8 = p;
                const bool rope_pair = strstr(items[i].name, "attn_q.weight") ||
                                       strstr(items[i].name, "attn_k.weight");
                if (g_bit2)      repack_b_i8_q2(p, items[i].ref->b, e->n, e->k);
                else if (g_bit3) {
                    if (getenv("MEGA_DBG8"))
                        fprintf(stderr, "i8copy: %s n=%lld k=%lld b=%p\n", items[i].name,
                                (long long) e->n, (long long) e->k, (void *) items[i].ref->b);
                    repack_b_i8_q3(p, items[i].ref->b, e->n, e->k);
                }
                else             repack_b_i8(p, e->data, e->n, e->k, nullptr);
                p += ((size_t) e->n * e->k + (size_t) e->n * 16 + 4095) & ~(size_t) 4095;
            }
            printf("weights: int8 copy in %.2f s (%.1f MB)\n", now_s() - t0, t8 / 1e6);
            int cf = open(c8, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (cf >= 0) { size_t put = 0; while (put < t8) { ssize_t w = write(cf, blob8 + put, t8 - put); if (w <= 0) break; put += (size_t) w; } close(cf); }
        }
    }

    g_emb = find("token_embd.weight");
    for (int L = 0; L < N_LAYER; ++L) {
        char nm[64];
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", L);
        g_attn_norm[L] = (const float *) find(nm)->data;
        snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", L);
        g_ffn_norm[L] = (const float *) find(nm)->data;
    }
    g_out_norm = (const float *) find("output_norm.weight")->data;

    g_x        = (float *) aligned_alloc(64, N_EMBD * 4);
    g_xa       = (float *) aligned_alloc(64, N_EMBD * 4);
    g_gate     = (float *) aligned_alloc(64, N_FF * 4);
    g_up       = (float *) aligned_alloc(64, N_FF * 4);
    g_q        = (float *) aligned_alloc(64, N_EMBD * 4);
    g_k        = (float *) aligned_alloc(64, N_KV * HEAD_DIM * 4);
    g_v        = (float *) aligned_alloc(64, N_KV * HEAD_DIM * 4);
    g_attn_out = (float *) aligned_alloc(64, N_EMBD * 4);
    g_logits   = (float *) aligned_alloc(64, VOCAB * 4);
    g_aq       = (uint8_t *) aligned_alloc(64, (size_t) N_FF * 5 + 65536);
    g_qa4      = (float *) aligned_alloc(64, (size_t) N_FF * 4 * sizeof(float) + 4096);
    const size_t kbytes = (size_t) N_LAYER * K_LAYER_STRIDE * 2;
    const size_t vbytes = (size_t) N_LAYER * KV_LAYER_STRIDE * 2;
    g_kdim = (_Float16 *) hugemem(kbytes);
    g_vpos = (_Float16 *) hugemem(vbytes);
    if (!g_kdim || !g_vpos) { fprintf(stderr, "oom kv\n"); return 1; }
    use_huge(g_kdim, kbytes);
    use_huge(g_vpos, vbytes);
    if (g_k8dec) {
        const size_t k8b = (size_t) N_LAYER * K_LAYER_STRIDE;
        const size_t v8b = (size_t) N_LAYER * KV_LAYER_STRIDE;
        g_k8d = (int8_t *) hugemem(k8b);
        g_v8d = (int8_t *) hugemem(v8b);
        if (!g_k8d || !g_v8d) { fprintf(stderr, "oom int8 kv\n"); return 1; }
        use_huge(g_k8d, k8b);
        use_huge(g_v8d, v8b);
        for (int L = 0; L < N_LAYER; ++L)
            for (int kv = 0; kv < N_KV; ++kv) { g_k8dsc[L][kv] = 0; g_v8dsc[L][kv] = 0; }
    }
    printf("kv: K %.0f MB (tiled), V %.0f MB (pos-major), f16\n", kbytes / 1e6, vbytes / 1e6);

    // Attention scratch has a 10.5 KB row stride, so without huge pages every
    // head row lands on its own page and the softmax reduction pays page walks.
    fprintf(stderr, "[dbg] before use_huge\n"); fflush(stderr);
    use_huge(g_sa, sizeof g_sa);
    use_huge(g_acc, sizeof g_acc);
    madvise(g_lmax, sizeof g_lmax, MADV_HUGEPAGE);
    madvise(g_lsum, sizeof g_lsum, MADV_HUGEPAGE);
    // Bind the main thread only after the workers exist: on this kernel a thread
    // that has written /proc/set_ai_thread cannot create threads any more --
    // pthread_create faults at NULL+8 inside libc's lock path.
    // The X100 softmax threads must exist before the main thread binds: a thread
    // that has written /proc/set_ai_thread faults inside libc's lock path if it
    // then calls pthread_create. They also must not write that file themselves --
    // it pins the caller to the A cluster. Each worker signals readiness so the
    // next creation never overlaps another thread's vendor handshake.
    if (g_x100 || g_a4p || (g_pipe && g_pfa_i8)) {
        g_n_sm = 8;
        g_x_nt = 8;
        for (int i = 0; i < g_n_sm; ++i) {
            pthread_t t;
            pthread_create(&t, nullptr, sm_worker, (void *) (intptr_t) i);
            pthread_detach(t);
            while (!g_ready[8 + i].load(std::memory_order_acquire)) sched_yield();
        }
        fprintf(stderr, "[dbg] %d X100 softmax threads up\n", g_n_sm);
    }
    for (int i = 1; i < g_participants; ++i) {
        pthread_t t;
        pthread_create(&t, nullptr, worker, (void *) (intptr_t) i);
        pthread_detach(t);
        while (!g_ready[i].load(std::memory_order_acquire)) sched_yield();
    }

    bind_cpu(g_cpu0);
    fprintf(stderr, "[dbg] threads up, participants=%d pf_rows=%lld\n", g_participants, (long long)g_pf_rows); fflush(stderr);
    tcm_attach(0);
    fprintf(stderr, "[tcm] size=%lld buf=%p\n", (long long) g_tcm_size, (void*) g_tcm[0]);
    if (g_pf_file) {
        FILE * f = fopen(g_pf_file, "rb");
        if (!f) { fprintf(stderr, "prompt open %s\n", g_pf_file); return 1; }
        fseek(f, 0, SEEK_END);
        const long fsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char * txt = (char *) malloc((size_t) fsz + 1);
        int32_t * toks = (int32_t *) malloc((size_t) 1 << 24);
        if (!txt || !toks) { fprintf(stderr, "oom prompt\n"); return 1; }
        const size_t got = fread(txt, 1, (size_t) fsz, f);
        txt[got] = 0;
        fclose(f);
        // Hand-rolled parse. fscanf("%d") here returned 1 for every token while
        // leaving the destination untouched, so every id came back as leftover
        // stack garbage and the first embedding lookup ran off the weight blob.
        int64_t nt = 0;
        const char * q = txt;
        while (*q && nt < (1 << 22)) {
            while (*q && (*q < '0' || *q > '9') && *q != '-') ++q;
            if (!*q) break;
            const bool neg = (*q == '-');
            if (neg) ++q;
            int32_t v = 0;
            if (*q < '0' || *q > '9') continue;
            while (*q >= '0' && *q <= '9') v = v * 10 + (*q++ - '0');
            toks[nt++] = neg ? -v : v;
        }
        free(txt);
        printf("prompt: %lld tokens from %s\n", (long long) nt, g_pf_file);
        // the last prompt token stays out: decode uses it as its first anchor
        const int64_t np = nt - 1;
        const int32_t anchor = toks[np];
        // The drafter has to be live before prefill: its encoder consumes the target's
        // residual stream at five layers of every prefill chunk, and those are exactly
        // the features its cache is built from. Set up after prefill and the draft sees
        // an empty prefix, which is what a first attempt did.
        if (g_m4spec) {
            ds_setup();
            if (!g_ds_ready) { fprintf(stderr, "spec: drafter unavailable\n"); return 1; }
        }
        if (!real_prefill(toks, np, g_pf_chunk)) return 1;
        if (const char * m4p = getenv("MEGA_M4P0")) {
            // Aim the round at four positions prefill already attended over, with the
            // real prompt tokens. Row r then has to reproduce the prefill attention
            // output for position p0+r, which is the same kernel on the same bytes.
            g_m4p0 = atoll(m4p);
            for (int i = 0; i < 4; ++i) g_m4tok[i] = toks[g_m4p0 + i];
            printf("m4p0: positions %lld..%lld tokens %d %d %d %d\n", (long long) g_m4p0,
                   (long long) (g_m4p0 + 3), g_m4tok[0], g_m4tok[1], g_m4tok[2], g_m4tok[3]);
        }
        free(toks);
        // The prefill just wrote positions 0..np-1 into the same f16 KV cache
        // decode reads, so generation continues from the last prompt token at
        // position np instead of starting from an empty context.
        prefill_release();
        prompt = anchor;
        pos0 = np;
        if (g_m4spec) {
            m4spec_run(pos0, prompt, g_m4spec, g_m4spec_k);
            return 0;
        }
        if (g_attn4 && g_m4dec) { m4dec_run(pos0, prompt, g_m4dec); return 0; }
        if (g_attn4) { attn4_bench(g_attn4, np); return 0; }
        printf("chain: decode starts at position %lld (anchor %d)\n",
               (long long) pos0, prompt);
    }
    // A chained -F run has already used these buffers; the benches would rerun
    // the synthetic attention on top of the prompt's geometry.
    if (!pos0 && g_pfa_rows > 0) { bench_prefill_attn(); return 0; }
    if (getenv("MEGA_M4GEMM")) { m4_gemm_bench(atoi(getenv("MEGA_M4GEMM"))); return 0; }
    if (!pos0 && g_pf_rows > 0) { bench_prefill_gemm(g_pf_rows); return 0; }

    // Unbuffered: several paths end in the pre-existing -O2 exit fault and a
    // buffered tail would lose every line printed before it.
    setvbuf(stdout, nullptr, _IONBF, 0);
    int tok = prompt;
    const double t_start = now_s();
    report_huge("decode");
    // Say which optional paths are live. A flag that silently fails to parse
    // (one `-X` used to swallow the argument after it) turns a whole A/B into
    // two runs of the same code, and the identical timings look like a result.
    printf("decode path: int8kv %d  bit3 %d  bit2 %d  hpdec %d  vendor %d  win %d  pv %s\n",
           g_k8dec, g_bit3, g_bit2, g_hpdec, g_pf_vendor, g_win,
           g_pvm4 ? "m4" : g_pvm2 ? "m2" : "m1");
    if (g_win) printf("attention window: last %d positions\n", g_win);
    printf("decode: %d tokens from %d\n", ngen, prompt);
    if (pos0 + ngen > MAXCTX) {
        fprintf(stderr, "context overflow: %lld prompt + %d generated exceeds %d\n",
                (long long) pos0, ngen, MAXCTX);
        return 1;
    }
    // MEGA_STEPS=1 prints the per-step stage deltas: the running average hides
    // whether a step is slow once or every time, which is the difference
    // between a first-touch cost and a steady-state one.
    const bool stepdbg = getenv("MEGA_STEPS") != nullptr;
    double prev_stage[S_NSTAGE];
    volatile double * vstage = g_stage;   // plain memcpy was folded away by -O2
    for (int step = 0; step < ngen; ++step) {
        TICK_DECL;
        if (stepdbg) for (int i = 0; i < S_NSTAGE; ++i) prev_stage[i] = vstage[i];
        if (step == ngen - 1) {
            memcpy(g_snap, g_stage, sizeof g_snap);
            g_snap_rope[0] = g_t_rope_math;
            g_snap_rope[1] = g_t_rope_app;
        }
        g_pos = (int) (pos0 + step);
        TICK();
        // The decode embedding fetch has to read the row at this file's own stride
        // and decode it with the embedding tensor's type. Hard-coding the Q4_0
        // stride (18 B per 32) and the last-repacked type meant a K-quant file fed
        // garbage as token 0's activation, and every later token inherited it.
        dequant_row_t(g_x,
                      g_emb->data + (int64_t) tok * src_row_bytes_t(N_EMBD, (int) g_emb->type),
                      N_EMBD, (int) g_emb->type);
        TOCK(S_EMB);
        if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", 0, 'e', g_x);

        for (int L = 0; L < N_LAYER; ++L) {
            g_layer = L;
            TICK();
            rmsnorm_par(g_xa, g_x, g_attn_norm[L], N_EMBD);
            TOCK(S_NORM);
            if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", L, 'n', g_xa);

            quant_a(g_xa, N_EMBD);
            TICK();
            g_mjn = 3;
            g_mj[0] = { g_wq[L].b, g_wq[L].k, g_wq[L].n, g_q };
            g_mj[1] = { g_wk[L].b, g_wk[L].k, g_wk[L].n, g_k };
            g_mj[2] = { g_wv[L].b, g_wv[L].k, g_wv[L].n, g_v };
            run(task_mgemm);
            TOCK(S_GEMM);

            TICK();
            rope_all();
            TOCK(S_ROPE);
            if (L == 0 && step == 0 && getenv("MEGA_DUMP")) {
                dump_n("dc", 0, 'v', g_v, N_KV * HEAD_DIM);
                dump_n("dc", 0, 'k', g_k, N_KV * HEAD_DIM);
            }

            TICK();
            for (int h = 0; h < N_HEAD; ++h)
                for (int d = 0; d < HEAD_DIM; ++d) g_attn_out[h * HEAD_DIM + d] = 0;
            attention();
            TOCK(S_ATTN);
            if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", L, 'i', g_attn_out);

            quant_a(g_attn_out, N_EMBD);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wov[L].b, g_wov[L].k, g_wov[L].n, g_xa };
            run(task_mgemm);
            TOCK(S_GEMM);
            if (L == 0 && step == 0 && getenv("MEGA_PROBE")) printf("dcL0 o-out: %.5f %.5f %.5f\n",
                                                                   g_xa[0], g_xa[1], g_xa[2]);
            TICK();
            parallel_for(N_EMBD, residual_add_fn);
            TOCK(S_MISC);
            if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", L, 'a', g_x);

            TICK();
            rmsnorm_par(g_xa, g_x, g_ffn_norm[L], N_EMBD);
            TOCK(S_NORM);

            quant_a(g_xa, N_EMBD);
            TICK();
            g_mjn = 2;
            g_mj[0] = { g_wg[L].b, g_wg[L].k, g_wg[L].n, g_gate };
            g_mj[1] = { g_wu[L].b, g_wu[L].k, g_wu[L].n, g_up };
            run(task_mgemm);
            TOCK(S_GEMM);

            if (L == 0 && step == 0 && getenv("MEGA_PROBE")) printf("dcL0 gate/up: %.5f %.5f %.5f | %.5f %.5f\n",
                                                                   g_gate[0], g_gate[1], g_gate[2], g_up[0], g_up[1]);
            if (L == 0 && step == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/dc_g.bin", "wb");
                fwrite(g_gate, 4, N_FF, fp); fclose(fp);
                fp = fopen("/tmp/dc_u.bin", "wb");
                fwrite(g_up, 4, N_FF, fp); fclose(fp);
                printf("vsetvl m1 = %d, m2 = %d\n",
                       (int) __riscv_vsetvl_e32m1(4096), (int) __riscv_vsetvl_e32m2(4096));
            }
            TICK();
            parallel_for(N_FF, silu_fn);
            TOCK(S_SILU);
            if (L == 0 && step == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/dc_h.bin", "wb");
                fwrite(g_gate, 4, N_FF, fp); fclose(fp);
            }

            quant_a(g_gate, N_FF);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wd[L].b, g_wd[L].k, g_wd[L].n, g_xa };
            run(task_mgemm);
            TOCK(S_GEMM);
            if (L == 0 && step == 0 && getenv("MEGA_DUMP")) {
                FILE * fp = fopen("/tmp/dc_d.bin", "wb");
                fwrite(g_xa, 4, N_EMBD, fp); fclose(fp);
            }
            TICK();
            parallel_for(N_EMBD, residual_add_fn);
            TOCK(S_MISC);
            if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", L, 'f', g_x);
        }

        TICK();
        rmsnorm_par(g_xa, g_x, g_out_norm, N_EMBD);
        TOCK(S_NORM);
        if (step == 0 && getenv("MEGA_DUMP")) dump_row("dc", 0, 'z', g_xa);
        quant_a(g_xa, N_EMBD);
        TICK();
        g_mjn = 1; g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, g_logits };
        run(task_mgemm);
        TOCK(S_GEMM);
        {
            // Same row llama.cpp's REF_LOG dumps: the logits for the last prompt
            // token, before any sampling. Comparing the two says whether a token
            // mismatch is upstream of the head or in it.
            if (step == 0 && getenv("MEGA_LOGD")) {
                if (FILE * f = fopen(getenv("MEGA_LOGD"), "wb")) {
                    fwrite(g_logits, 4, VOCAB, f);
                    fclose(f);
                    printf("logits -> %s\n", getenv("MEGA_LOGD"));
                }
            }
            // One row per step, concatenated, so a forced run can be scored against
            // the reference at every position rather than at step 0 only.
            if (g_steplog) {
                fwrite(g_logits, 4, VOCAB, g_steplog);
                fflush(g_steplog);
            }
        }
        TICK();
        run(task_argmax);
        {   // combine the per-thread maxima in vocabulary order so ties keep the
            // lowest index, matching llama.cpp's argmax
            int best = g_am_idx[0];
            float bv = g_am_val[0];
            for (int t = 1; t < g_participants; ++t) {
                if (g_am_val[t] > bv || (g_am_val[t] == bv && g_am_idx[t] < best)) {
                    bv = g_am_val[t]; best = g_am_idx[t];
                }
            }
            // Print enough of the stream to diff a whole aligned run against the
            // reference rather than a prefix of one.
            if (step < 64) printf("  step %d -> %d (%.4f)\n", step, best, bv);
            // MEGA_TOPSTEP picks which step the top-8 dump covers (default 0). The
            // margin between rank 0 and rank 1 at a chosen step is what separates
            // "resolved a near tie differently" from "lost accuracy".
            static const int topstep = getenv("MEGA_TOPSTEP") ? atoi(getenv("MEGA_TOPSTEP")) : 0;
            if (step == topstep && getenv("MEGA_TOP")) {
                // top-8 straight off the head, so the gap to the f32 reference
                // (8181 at 10.83, 280 at 9.65) is visible instead of hidden
                // behind a single argmax
                static int idx[8]; static float val[8];
                for (int j = 0; j < 8; ++j) { idx[j] = -1; val[j] = -INFINITY; }
                for (int v2 = 0; v2 < g_wo.n; ++v2) {
                    const float x2 = g_logits[v2];
                    if (x2 <= val[7]) continue;
                    int j = 7;
                    while (j > 0 && val[j - 1] < x2) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; --j; }
                    val[j] = x2; idx[j] = v2;
                }
                printf("  top8:");
                for (int j = 0; j < 8; ++j) printf(" %d(%.2f)", idx[j], val[j]);
                printf("\n");
            }
            tok = best;
            // Teacher forcing: walk the reference's token chain instead of our own
            // argmax. Without it a single near-tie splits the two chains and every
            // later step compares different prefixes, so the logits cannot be
            // scored against the reference at all.
            if (g_force_n > 0) {
                const int ft = step < g_force_n ? g_force_tok[step] : best;
                if (step < g_force_n && ft != best) printf("  forced step %d: our %d, ref %d\n", step, best, ft);
                tok = ft;
            }
        }
        TOCK(S_HEAD);
        if (stepdbg) {
            printf("  [step %2d]", step);
            for (int i = 0; i < S_NSTAGE; ++i) {
                const double d = g_stage[i] - prev_stage[i];
                if (d > 0) printf(" %s %.1f", g_stage_name[i], d * 1e3);
            }
            printf("\n");
            fflush(stdout);
        }
    }
    if (getenv("MEGA_EXPKV")) export_kv(pos0 + ngen);   // TEMP: dump the cache this path built
    const double dt = now_s() - t_start;
    printf("attn work (ms/token, tid0 only is not separated; sum of all threads' work): "
           "qk %.2f exp %.2f pv %.2f\n", g_aw[0]*1e3/ngen, g_aw[1]*1e3/ngen, g_aw[2]*1e3/ngen);
    printf("attn split (ms/token): ranges %.3f qk %.3f gmax %.3f exp %.3f pv %.3f fin %.3f\n",
           g_at[0]*1e3/ngen, g_at[1]*1e3/ngen, g_at[2]*1e3/ngen, g_at[3]*1e3/ngen,
           g_at[4]*1e3/ngen, g_at[5]*1e3/ngen);
    printf("last token (pos=%d):\n", ngen - 1);
    for (int i = 0; i < S_NSTAGE; ++i) {
        const double d = g_stage[i] - g_snap[i];
        if (d > 0) printf("    %-16s %7.3f ms\n", g_stage_name[i], d * 1e3);
    }
    printf("    rope-math        %7.3f ms\n", (g_t_rope_math - g_snap_rope[0]) * 1e3);
    printf("    rope-append      %7.3f ms\n", (g_t_rope_app - g_snap_rope[1]) * 1e3);
    printf("--- stage breakdown (ms/token) ---\n");
    for (int i = 0; i < S_NSTAGE; ++i)
        if (g_stage[i] > 0)
            printf("  %-16s %7.2f  (%4.1f%%)\n", g_stage_name[i], g_stage[i] * 1000.0 / ngen,
                   100.0 * g_stage[i] / dt);
    printf("decode: %d tokens in %.3f s -> %.2f tok/s\n", ngen, dt, ngen / dt);

    g_quit.store(1, std::memory_order_release);
    barrier();
    return 0;
}
