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
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
}
namespace ime2 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
                        float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc);
size_t gemm_kernel_i8i3k(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bzp,
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
static const int N_LAYER = 42, N_EMBD = 2048, N_FF = 6144, N_HEAD = 16, N_KV = 2, HEAD_DIM = 128;
static const int N_GPU = 8, MAXCTX = 10240, VOCAB = 130560;
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

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#define TICK_DECL double _t0
#define TICK() _t0 = now_s()
#define TOCK(st) g_stage[st] += now_s() - _t0

static void use_huge(void * p, size_t n) { if (p) madvise(p, n, MADV_HUGEPAGE); }

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
static const int Q3_K = 256;                 // K elements per superblock
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;   // 3648 B
static size_t g_blk_bytes = 576;

static void repack_b_q3(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t nb = k / 32, rs = nb * 18, nsup = k / Q3_K;
    static float tile[32][Q3_K];
    for (int64_t g = 0; g < rows; g += 32) {
        const int64_t grp = g / 32;
        for (int64_t sup = 0; sup < nsup; ++sup) {
            uint8_t * blk = dst + (grp * nsup + sup) * Q3_BLK;
            int8_t *   scl = (int8_t *) blk;
            uint16_t * hms = (uint16_t *) (blk + 512);
            uint8_t *  qs  = blk + 512 + 1024;
            _Float16 * s16 = (_Float16 *) (blk + 512 + 1024 + 2048);
            // dequantize this 32 x 256 tile out of Q4_0
            for (int i = 0; i < 32; ++i) {
                const int64_t sr = perm ? (int64_t) perm[g + i] : (g + i);
                const uint8_t * row = src + sr * rs;
                for (int64_t x = 0; x < Q3_K / 32; ++x) {
                    const int64_t b = sup * (Q3_K / 32) + x;
                    _Float16 sc; memcpy(&sc, row + b * 18, 2);
                    const uint8_t * q = row + b * 18 + 2;
                    for (int j = 0; j < 16; ++j) {
                        tile[i][x * 32 + j]      = (float) sc * (float) ((int) (q[j] & 0x0F) - 8);
                        tile[i][x * 32 + 16 + j] = (float) sc * (float) ((int) (q[j] >> 4) - 8);
                    }
                }
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
static int bind_cpu(int cpu) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) { if (write(fd, "0", 1) < 0) {} close(fd); }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    bind_cpu(g_cpu0 + id);
    tcm_attach(id);
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

// ---------------------------------------------------------------- model buffers
static uint8_t * g_wblob;
struct WRef { const uint8_t * b; int64_t k, n; const uint8_t * b8; };
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

static void dequant_row(float * dst, const uint8_t * row, int64_t k) {
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
            const int64_t kb = g_mj[j].k / (g_bit3 ? Q3_K : 32);
            const uint8_t * bp = g_mj[j].b + a * kb * g_blk_bytes;
            if (g_bit3)
                spacemit_kernels::ime2::gemm_kernel_i8i3k(
                    Q3_K, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
            else
                spacemit_kernels::ime2::gemm_kernel_i8i4(
                    32, g_aq, bp, nullptr, g_mj[j].c + a * 32,
                    1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) ((b2 - a) * 32));
        }
}

// ------------------------------------------------------------------ prefill GEMM
// The m=4 vendor path needs A as, per k_blk, [f32 scale x 4][int8 data 4x32]
// = 144 bytes; the grouped quantizer fills that. k_blks must be a multiple of 16.
// Loop order is n-outer / m-inner so the 576-byte B column tile is loaded once
// from DRAM and reused for every row group in the chunk: at m=1 the tile was
// re-read per row, which is what kept prefill pinned to DRAM bandwidth.
static int64_t g_pf_rows;
static int64_t g_pf_nb = 1;          // n-blocks per gemm call
static int g_pf_mt = 4;              // row groups (4 rows each) per shared-B tile
static int g_pf_reps = 0;            // repeat q_proj to isolate compute peak
static uint8_t * g_pfa;              // quantized A for the whole chunk
static int64_t g_pfa_rowstride;      // bytes per 4-row group

struct PfJob { const uint8_t * b; int64_t k, n; float * c; int64_t ldc; };
static PfJob g_pfj[4];
static int g_pfjn;

static void task_pf_gemm(int tid) {
    const int64_t mgroups = g_pf_rows / 4;
    for (int j = 0; j < g_pfjn; ++j) {
        const int64_t nb = g_pfj[j].n / 32;
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
            const uint8_t * asrc = g_pfa + (j * mgroups + mt * G) * g_pfa_rowstride;
            const uint8_t * abase = asrc;
            if (use_tcm) {
                memcpy(g_tcm[tid], asrc, (size_t) atile);
                abase = g_tcm[tid];
            }
            for (int64_t a = 0; a < nb; ++a) {
                const uint8_t * bcol = b + a * kb * 576;
                for (int64_t g = 0; g < G; ++g) {
                    spacemit_kernels::ime2::gemm_kernel_i8i4(
                        32, abase + g * g_pfa_rowstride, bcol, nullptr,
                        c + (mt * G + g) * 4 * ldc + a * 32,
                        4, 32, (size_t) kb, (size_t) ldc);
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
static void task_pf_quant(int tid) {
    const int64_t mg = g_qa_rows / 4;
    for (int64_t g = tid; g < mg; g += g_participants)
        spacemit_kernels::rvv::quantize_a_4row_i8(
            32, g_qa_src + g * 4 * g_qa_k, (size_t) g_qa_k,
            g_pfa + (g_qa_j * mg + g) * g_pfa_rowstride);
}
static void quant_a_4row(const float * a, int64_t rows, int64_t k, int64_t j) {
    // One 4-row record per k block is 152 B: [4 x f32 a_scale][4 x i16 a_sum]
    // [4 x 32 int8]. The vendor asm advances A by 16+8+128, so 144 overlaps the
    // next record by 8 bytes.
    g_pfa_rowstride = (k / 32) * 152;
    g_qa_src = a;
    g_qa_rows = rows;
    g_qa_k = k;
    g_qa_j = j;
    run(task_pf_quant);
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
static int64_t g_i8_blk(int64_t k) { return k / 16; }

static void repack_b_i8(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k, const int * perm) {
    const int64_t nb = k / 32, rs = nb * 18, kst = k / 16;
    int8_t *   pay = (int8_t *) dst;
    // Scales are stored pre-expanded to 32 lanes: a lane holds (row, col) as
    // row*8+col, so the same 8 column scales repeat four times. Doing it here
    // removes a per-iteration gather from the GEMM epilogue.
    float *    scl = (float *) (dst + (size_t) rows * k);
    static float row[8192];
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t sr = perm ? (int64_t) perm[r] : r;
        const uint8_t * s = src + sr * rs;
        for (int64_t b = 0; b < nb; ++b) {
            _Float16 sc; memcpy(&sc, s + b * 18, 2);
            const uint8_t * q = s + b * 18 + 2;
            for (int j = 0; j < 16; ++j) {
                row[b * 32 + j]      = (float) sc * (float) ((int) (q[j] & 0x0F) - 8);
                row[b * 32 + 16 + j] = (float) sc * (float) ((int) (q[j] >> 4) - 8);
            }
        }
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

// Activation packing: one record per 4-row group = [4 x f32 scale][4 rows x K int8]
// laid out per 16-element k-step as 4x16 = 64 bytes.
static const float * g_a8_src;
static int64_t g_a8_rows, g_a8_k, g_a8_j;
static int64_t g_a8_stride;
static void task_pf_quant_i8(int tid) {
    const int64_t k = g_a8_k, kst = k / 16, mg = g_a8_rows / 4;
    for (int64_t g = tid; g < mg; g += g_participants) {
        uint8_t * rec = g_pfa + (g_a8_j * mg + g) * g_a8_stride;
        int8_t *  pay = (int8_t *) rec;
        float *   sc  = (float *) (rec + kst * 64);
        for (int r = 0; r < 4; ++r) {
            const float * a = g_a8_src + ((g * 4 + r) * k);
            // pass 1: max |a| over the row
            float mx = 0;
            for (int64_t i = 0; i < k; ) {
                const size_t vl = __riscv_vsetvl_e32m4((size_t) (k - i));
                vfloat32m4_t v = __riscv_vle32_v_f32m4(a + i, vl);
                v = __riscv_vfabs_v_f32m4(v, vl);
                const vfloat32m1_t red =
                    __riscv_vfredmax_vs_f32m4_f32m1(v, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
                const float m = __riscv_vfmv_f_s_f32m1_f32(red);
                if (m > mx) mx = m;
                i += (int64_t) vl;
            }
            const float s = mx > 0 ? mx / 127.0f : 1.0f;
            const float inv = 1.0f / s;
            for (int lane = 0; lane < 32; ++lane) sc[r * 32 + lane] = s;
            // pass 2: scale, round to nearest even, clamp, narrow, store 16 bytes
            for (int64_t t = 0; t < kst; ++t) {
                const size_t vl = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v = __riscv_vle32_v_f32m1(a + t * 16, vl);
                v = __riscv_vfmul_vf_f32m1(v, inv, vl);
                vint32m1_t q = __riscv_vfcvt_x_f_v_i32m1(v, vl);
                q = __riscv_vmax_vx_i32m1(q, -127, vl);
                q = __riscv_vmin_vx_i32m1(q, 127, vl);
                const vint16mf2_t b16 = __riscv_vncvt_x_x_w_i16mf2(q, vl);
                const vint8mf4_t b8 = __riscv_vncvt_x_x_w_i8mf4(b16, vl);
                __riscv_vse8_v_i8mf4(pay + t * 64 + r * 16, b8, vl);
            }
        }
    }
    (void) kst;
}
static void quant_a_i8d(const float * a, int64_t rows, int64_t k, int64_t j) {
    g_a8_stride = (k / 16) * 64 + 128;  // payload, then 32 pre-expanded scales
    g_a8_src = a; g_a8_rows = rows; g_a8_k = k; g_a8_j = j;
    run(task_pf_quant_i8);
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
static float   g_i8t[8][256];
static float   g_a8s[8][4];
static const int64_t CH = 8;              // row groups per scheduling chunk
static std::atomic<int64_t> g_chunk[128];
static int g_noepi = 0;
static int64_t g_pf_tile = 1 << 24;  // row groups per thread before re-reading B

static void task_pf_gemm_i8(int tid) {
    const int64_t mgroups = g_pf_rows / 4;
    for (int j = 0; j < g_pfjn; ++j) {
        const int64_t nblk = g_pfj[j].n / 64, kst = g_pfj[j].k / 16;
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
                const int64_t rt = g_chunk[nb].fetch_add(CH, std::memory_order_relaxed);
                if (rt >= mgroups) break;
                int64_t rend = rt + CH;
                if (rend > mgroups) rend = mgroups;
                for (int64_t g = rt; g < rend; ++g) {
                    const uint8_t * arec = g_pfa + (j * mgroups + g) * g_a8_stride;
                    const float * asc = (const float *) (arec + kst * 64);
                    gemm_i8d_4x64((const int8_t *) arec, bcol, g_i8c[tid], kst);
                    if (g_noepi) { c[(g * 4) + nb] = (float) g_i8c[tid][0]; continue; }
                    for (int k2 = 0; k2 < 4; ++k2) g_a8s[tid][k2] = asc[k2 * 32];
                    float * base = crow + (g * 4) * ldc;
                    for (int cg = 0; cg < 8; ++cg) {
                        size_t vl = __riscv_vsetvl_e32m1(32);
                        const vint32m1_t iv = __riscv_vle32_v_i32m1(g_i8c[tid] + cg * 32, vl);
                        vfloat32m1_t fv = __riscv_vfcvt_f_x_v_f32m1(iv, vl);
                        fv = __riscv_vfmul_vv_f32m1(fv, __riscv_vle32_v_f32m1(bsv + cg * 32, vl), vl);
                        for (int r = 0; r < 4; ++r) {
                            vfloat32m1_t sh = r ? __riscv_vslidedown_vx_f32m1(fv, (size_t) (r * 8), vl) : fv;
                            vl = __riscv_vsetvl_e32m1(8);
                            sh = __riscv_vfmul_vf_f32m1(sh, g_a8s[tid][r], vl);
                            __riscv_vse32_v_f32m1(base + r * ldc + cg * 8, sh, vl);
                            vl = __riscv_vsetvl_e32m1(32);
                        }
                    }
                }
            }
        }
    }
}


static const float * qa_src;
static int64_t qa_k;
static void task_quant(int tid) {
    const int64_t nb = qa_k / (g_bit3 ? Q3_K : 32);
    const int64_t per = (nb + g_participants - 1) / g_participants;
    const int64_t b0 = per * tid;
    int64_t b1 = (tid == g_participants - 1) ? nb : per * (tid + 1);
    if (b1 > nb) b1 = nb;
    if (b0 >= b1) return;
    if (g_bit3)
        spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, qa_src + b0 * Q3_K, (size_t) ((b1 - b0) * Q3_K), g_aq + b0 * 292);
    else
        spacemit_kernels::rvv::quantize_a_row_i8(32, qa_src + b0 * 32, (size_t) ((b1 - b0) * 32), g_aq + b0 * 38);
}

static void quant_a(const float * a, int64_t k) {
    if (g_bit3) {
        if (k < Q3_K) { spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, a, (size_t) k, g_aq); return; }
        qa_src = a; qa_k = k;
        run(task_quant);
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

static void attn_ranges(void) {
    const int nb = g_pos + 1;
    const int nt = (nb + KT - 1) / KT;
    const int per = (nt + g_participants - 1) / g_participants;
    for (int t = 0; t < g_participants; ++t) {
        const int t0 = per * t, t1 = (t == g_participants - 1) ? nt : per * (t + 1);
        g_p0[t] = (t0 * KT > nb) ? nb : t0 * KT;
        g_p1[t] = (t1 * KT > nb) ? nb : t1 * KT;
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
            const size_t vl = __riscv_vsetvl_e32m1((size_t) (len - pb < KT ? len - pb : KT));
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int d = 0; d < HEAD_DIM; d += 4) {
                const vfloat32m1_t k0 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 0) * KT, vl), vl);
                const vfloat32m1_t k1 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 1) * KT, vl), vl);
                const vfloat32m1_t k2 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 2) * KT, vl), vl);
                const vfloat32m1_t k3 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(kb + (int64_t) (d + 3) * KT, vl), vl);
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
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 32);
        int i = 0;
        for (; i + 64 <= len; i += 64) {
            const vfloat32m1_t t0 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), 1.4426950408889634f, 32);
            const vfloat32m1_t t1 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i + 32, 32), mx, 32), 1.4426950408889634f, 32);
            const vfloat32m1_t e0 = exp2f_poly(__riscv_vfmax_vf_f32m1(t0, -126.0f, 32), 32);
            const vfloat32m1_t e1 = exp2f_poly(__riscv_vfmax_vf_f32m1(t1, -126.0f, 32), 32);
            __riscv_vse32_v_f32m1(row + i, e0, 32);
            __riscv_vse32_v_f32m1(row + i + 32, e1, 32);
            vs = __riscv_vfadd_vv_f32m1(vs, __riscv_vfadd_vv_f32m1(e0, e1, 32), 32);
        }
        for (; i + 32 <= len; i += 32) {
            const vfloat32m1_t t0 = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(row + i, 32), mx, 32), 1.4426950408889634f, 32);
            const vfloat32m1_t e0 = exp2f_poly(__riscv_vfmax_vf_f32m1(t0, -126.0f, 32), 32);
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
            const float * wk = sa + (int64_t) kv * NH_PER_KV * SMAX;
            for (int p = 0; p < len; ++p) {
                const _Float16 * vp = base + (int64_t) (p0 + p) * N_KV * HEAD_DIM;
                const vfloat32m1_t v0 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp, 32), 32);
                const vfloat32m1_t v1 = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(vp + 32, 32), 32);
                const float w0 = wk[0 * SMAX + p], w1 = wk[1 * SMAX + p];
                const float w2 = wk[2 * SMAX + p], w3 = wk[3 * SMAX + p];
                const float w4 = wk[4 * SMAX + p], w5 = wk[5 * SMAX + p];
                const float w6 = wk[6 * SMAX + p], w7 = wk[7 * SMAX + p];
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

// ---- rope. Two changes make this cheap:
//  * cos/sin come from a table built once for all positions, so no libm in the loop.
//  * q_proj/k_proj rows are repacked in rotation-pair order (even dims, then odd),
//    so the rotation is plain contiguous vector math instead of a strided gather.
//    A dot product is permutation invariant, and q and k share the order, so
//    the attention scores are unchanged.
static double g_t_rope_math, g_t_rope_app;
static void rope_all(void) {
    const double _ta = now_s();
    const float * cs = g_rope_tab + (int64_t) g_pos * 128;
    const vfloat32m1_t c0 = __riscv_vle32_v_f32m1(cs, 32);
    const vfloat32m1_t c1 = __riscv_vle32_v_f32m1(cs + 32, 32);
    const vfloat32m1_t s0 = __riscv_vle32_v_f32m1(cs + 64, 32);
    const vfloat32m1_t s1 = __riscv_vle32_v_f32m1(cs + 96, 32);
    for (int h = 0; h < N_HEAD + N_KV; ++h) {
        float * x = (h < N_HEAD) ? (g_q + h * HEAD_DIM) : (g_k + (h - N_HEAD) * HEAD_DIM);
        const vfloat32m1_t e0 = __riscv_vle32_v_f32m1(x, 32);
        const vfloat32m1_t e1 = __riscv_vle32_v_f32m1(x + 32, 32);
        const vfloat32m1_t o0 = __riscv_vle32_v_f32m1(x + 64, 32);
        const vfloat32m1_t o1 = __riscv_vle32_v_f32m1(x + 96, 32);
        __riscv_vse32_v_f32m1(x,      __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(e0, c0, 32), __riscv_vfmul_vv_f32m1(o0, s0, 32), 32), 32);
        __riscv_vse32_v_f32m1(x + 32, __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(e1, c1, 32), __riscv_vfmul_vv_f32m1(o1, s1, 32), 32), 32);
        __riscv_vse32_v_f32m1(x + 64, __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(e0, s0, 32), __riscv_vfmul_vv_f32m1(o0, c0, 32), 32), 32);
        __riscv_vse32_v_f32m1(x + 96, __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(e1, s1, 32), __riscv_vfmul_vv_f32m1(o1, c1, 32), 32), 32);
    }
    {   // q must be scaled by 1/sqrt(head_dim); scores feed only the softmax
        const float inv = 1.0f / sqrtf((float) HEAD_DIM);
        for (int h = 0; h < N_HEAD; ++h)
            for (int i = 0; i < HEAD_DIM; i += 32)
                __riscv_vse32_v_f32m1(g_q + h * HEAD_DIM + i,
                                      __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(g_q + h * HEAD_DIM + i, 32), inv, 32), 32);
    }
    g_t_rope_math += now_s() - _ta;

    const double _tb = now_s();
    const int pblk = g_pos / KT, pi = g_pos % KT;
    _Float16 * vb = g_vpos + (int64_t) g_layer * KV_LAYER_STRIDE + (int64_t) g_pos * N_KV * HEAD_DIM;
    for (int kv = 0; kv < N_KV; ++kv) {
        _Float16 * kb = ktile(g_layer, kv, pblk);
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
        const float inv = 1.0f / gsum[hh];
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
    run(attn_pv);
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
        const vfloat32m1_t den = __riscv_vfadd_vf_f32m1(
            __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_i32m1_f32m1(ex), vl), 1.0f, vl);
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
static const int I8_G = 16;
static const int I8_ROWS = I8_G * PFA_ROWS;
static const int WBLK = (MAXCTX / 16) * 64;    // int8 attention weights per row group
static int64_t g_pfa_rows, g_pfa_ctx;
static float * g_pq;                        // [rows][N_HEAD][HEAD_DIM]
static float * g_pfa_out;                   // [rows][N_HEAD][HEAD_DIM], layer 0 only
static float g_ppv_tmp[N_GPU][4 * N_HEAD * HEAD_DIM];
static float g_psa[N_GPU][PFA_ROWS][MAXCTX];    // per-thread scores (f16 path)
// The int8 path keeps its raw scores in int16 fixed point at 1/128. A row of
// scores is 40 KB in f32 and is written once and read twice, which made the
// score buffer - not the dots - the prefill attention bottleneck: 13.4 GB per
// layer of writes plus 26.8 GB of reads, i.e. DRAM bound at ~11 GB/s. int16
// halves that, and the max pass can run in the integer domain.
static const float PSA_SC = 128.0f;
static int16_t (*g_psa16)[I8_ROWS][MAXCTX];
static double g_pfa_qk[N_GPU], g_pfa_sm[N_GPU], g_pfa_pv[N_GPU];
static thread_local int g_pfa_tid;
static float * g_kf;                        // [kv][pblk][dim][32] f32, one layer
static int64_t g_kf_npb;
static inline float * kftile(int kv, int pblk) {
    return g_kf + ((int64_t) kv * g_kf_npb + pblk) * HEAD_DIM * KT;
}
static float * g_vf;                        // [kv][pos][dim] f32, one layer
static int g_pfa_f32 = 0;
static float g_pfa_dbg_sum[4], g_pfa_dbg_mx[4];

__attribute__((optimize("no-tree-vectorize")))
static void pfa_softmax_row(float * row, int len, double * t_sm) {
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
        const vfloat32m1_t e = exp2f_poly(__riscv_vfmax_vf_f32m1(t, -126.0f, 32), 32);
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
    g_pfa_dbg_sum[0] = sum;
    g_pfa_dbg_mx[0] = mx;
    *t_sm += now_s() - _w;
}

// The f16 -> f32 widen (vfwcvt) is an emulated instruction on this core, ~86
// cycles. Paying it inside QK means every K element is widened once per row
// group, i.e. 2560 times at 10K. Converting a layer's K and V into a 21 MB f32
// scratch once costs ~0.02 s and removes the widen from the hot loop entirely.
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
static int8_t * g_q8i;                 // [(head*nqg + group)][8 ksteps][4 rows x 16]
static float *  g_q8isc;               // [rows][N_HEAD]
static float g_k8isc[4], g_v8isc[4];
static int64_t g_k8i_npb, g_v8i_npb16, g_q8i_nqg;
static int g_pfa_i8 = 0;
static int g_pfa_dbg1 = 0;
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
// the A operand carry the 4 queries, so a 64-byte stride is exact (the prototype
// used 128 with a padded layout).
static void pv4(int8_t * const * acc, const int8_t * w8g, const int8_t * vblk, int npblk) {
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

// Repack one layer of K and V from the f16 cache into the dot layouts, and pack Q
// once. Scales are one per (kv head) per layer for K and V: a per-block scale is
// not usable because the dot contracts over the block and the score has to be a
// single scaled integer sum.
static void pfa_convert_kv_i8(int layer, int64_t npos) {
    g_k8i_npb = (npos + 63) / 64;
    g_v8i_npb16 = (npos + 15) / 16;
    for (int kv = 0; kv < N_KV; ++kv) {
        float mk = 0, mv = 0;
        for (int64_t p = 0; p < npos; ++p) {
            const _Float16 * kb = ktile(layer, kv, (int) (p / KT));
            const _Float16 * vb = g_vpos + (int64_t) layer * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float a = fabsf((float) kb[d * KT + (p % KT)]);
                const float b = fabsf((float) vb[d]);
                if (a > mk) mk = a;
                if (b > mv) mv = b;
            }
        }
        const float sk = mk > 0 ? mk / 127.0f : 1.0f;
        const float sv = mv > 0 ? mv / 127.0f : 1.0f;
        g_k8isc[kv] = sk; g_v8isc[kv] = sv;
        const float ik = 1.0f / sk, iv = 1.0f / sv;
        for (int64_t p = 0; p < npos; ++p) {
            const _Float16 * kb = ktile(layer, kv, (int) (p / KT));
            const _Float16 * vb = g_vpos + (int64_t) layer * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM + kv * HEAD_DIM;
            int8_t * kdst = g_k8i + ((kv * g_k8i_npb + p / 64) * 8) * 1024;
            int8_t * vdst = g_v8i + kv * g_v8i_npb16 * 2048;
            for (int d = 0; d < HEAD_DIM; ++d) {
                int a = (int) lrintf((float) kb[d * KT + (p % KT)] * ik);
                if (a > 127) a = 127; if (a < -127) a = -127;
                kdst[(d / 16) * 1024 + (p % 64) * 16 + (d % 16)] = (int8_t) a;
                int b = (int) lrintf((float) vb[d] * iv);
                if (b > 127) b = 127; if (b < -127) b = -127;
                vdst[((d / 64) * g_v8i_npb16 + p / 16) * 1024 + (d % 64) * 16 + (p % 16)] = (int8_t) b;
            }
        }
    }
}

static void pfa_pack_q(int64_t rows) {
    g_q8i_nqg = (rows + 3) / 4;
    for (int64_t g = 0; g < g_q8i_nqg; ++g)
        for (int h = 0; h < N_HEAD; ++h) {
            int8_t * dst = g_q8i + ((h * g_q8i_nqg + g) * 8) * 64;
            for (int r = 0; r < 4; ++r) {
                const int64_t qr = g * 4 + r;
                if (qr >= rows) break;
                const float * q = g_pq + (qr * N_HEAD + h) * HEAD_DIM;
                float mx = 0;
                for (int d = 0; d < HEAD_DIM; ++d) mx = fmaxf(mx, fabsf(q[d]));
                const float s = mx > 0 ? mx / 127.0f : 1.0f;
                g_q8isc[qr * N_HEAD + h] = s;
                const float inv = 1.0f / s;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int v = (int) lrintf(q[d] * inv);
                    if (v > 127) v = 127; if (v < -127) v = -127;
                    dst[(d / 16) * 64 + r * 16 + (d % 16)] = (int8_t) v;
                }
            }
        }
}

// One tile of I8_ROWS queries of one head: QK over every position the tile can
// see, exact causal masking, softmax, requantise the weights, then PV.
static void pfa_head_i8(int layer, int h, int64_t qbase) {
    const int kv = h / NH_PER_KV;
    const int nn = (int) (g_pfa_ctx + qbase + I8_ROWS);
    const int npb = (nn + 63) / 64;
    const int8_t * q8g = g_q8i + ((h * g_q8i_nqg) + qbase / 4) * 8 * 64;
    int16_t * row16[I8_ROWS];
    for (int r = 0; r < I8_ROWS; ++r) row16[r] = &g_psa16[g_pfa_tid][r][0];

    alignas(64) static __thread int32_t raw[8][32];
    alignas(64) static __thread int8_t w8[I8_G * WBLK + 256];
    static __thread float wsc[I8_ROWS];

    // Per-lane row scales, built once per tile: with VLMAX lanes the epilogue
    // converts all four rows and all eight columns of one raw register in a
    // single pass instead of four 8-lane passes.
    alignas(64) static __thread float svt[I8_G][32];
    for (int gg = 0; gg < I8_G; ++gg)
        for (int r = 0; r < 4; ++r) {
            const float s_r = g_q8isc[((qbase + gg * 4 + r) * N_HEAD + h)] * g_k8isc[kv] * PSA_SC;
            for (int c = 0; c < 8; ++c) svt[gg][r * 8 + c] = s_r;
        }

    const double _q = now_s();
    for (int pb = 0; pb < npb; ++pb) {
        const int8_t * kb = g_k8i + ((kv * g_k8i_npb + pb) * 8) * 1024;
        for (int gg = 0; gg < I8_G; ++gg) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
            qk4(op, q8g + gg * 512, kb, HEAD_DIM / 16);
            const size_t vl32 = __riscv_vsetvl_e32m1(32);
            const vfloat32m1_t sv = __riscv_vle32_v_f32m1(svt[gg], vl32);
            const size_t vl8 = __riscv_vsetvl_e16mf2(8);
            for (int j = 0; j < 8; ++j) {
                __riscv_vsetvl_e32m1(32);
                const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                const vfloat32m1_t f = __riscv_vfmul_vv_f32m1(
                    __riscv_vfcvt_f_x_v_f32m1(a, vl32), sv, vl32);
                const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(
                    __riscv_vfcvt_x_f_v_i32m1(f, vl32), vl32);
                for (int r = 0; r < 4; ++r) {
                    const vint16mf2_t sl = r
                        ? __riscv_vslidedown_vx_i16mf2(h16, (size_t) (r * 8), vl8) : h16;
                    __riscv_vse16_v_i16mf2(row16[gg * 4 + r] + pb * 64 + j * 8, sl, vl8);
                }
            }
        }
    }
    g_pfa_qk[g_pfa_tid] += now_s() - _q;

    // Two-pass softmax. The largest probability is always exactly 1 (it is
    // exp(mx-mx)), so the int8 weights use the fixed scale 1/127 and the 1/sum
    // factor folds into the PV scale.
    const double _s = now_s();
    const int wnb = (nn + 15) / 16;
    for (int gg = 0; gg < I8_G; ++gg) memset(w8 + gg * WBLK, 0, (size_t) wnb * 64);
    for (int rr = 0; rr < I8_ROWS; ++rr) {
        int16_t * rw = row16[rr];
        int8_t * wb = w8 + (rr / 4) * WBLK;
        const int r = rr % 4;
        const int len = (int) (g_pfa_ctx + qbase + rr + 1);
        // max in the integer domain: the scale is positive, so it is order
        // preserving and no conversion is needed on the pass that touches every
        // score
        int mx16 = -32768;
        int i2 = 0;
        for (; i2 + 32 <= len; i2 += 32) {
            const size_t vl = __riscv_vsetvl_e16m1(32);
            const vint16m1_t v = __riscv_vle16_v_i16m1(rw + i2, vl);
            const vint16m1_t m = __riscv_vredmax_vs_i16m1_i16m1(
                v, __riscv_vmv_s_x_i16m1((int16_t) -32768, 1), vl);
            const int t = (int) __riscv_vmv_x_s_i16m1_i16(m);
            if (t > mx16) mx16 = t;
        }
        for (; i2 + 16 <= len; i2 += 16) {
            const size_t vl = __riscv_vsetvl_e16m1(16);
            const vint16m1_t v = __riscv_vle16_v_i16m1(rw + i2, vl);
            const vint16m1_t m = __riscv_vredmax_vs_i16m1_i16m1(
                v, __riscv_vmv_s_x_i16m1((int16_t) -32768, 1), vl);
            const int t = (int) __riscv_vmv_x_s_i16m1_i16(m);
            if (t > mx16) mx16 = t;
        }
        for (; i2 < len; ++i2) if (rw[i2] > mx16) mx16 = rw[i2];
        const float mx = (float) mx16 / PSA_SC;
        const size_t vlz = __riscv_vsetvl_e32m1(32);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlz);
        int p = 0;
        for (; p + 32 <= len; p += 32) {
            const size_t vl = __riscv_vsetvl_e32m1(32);
            const vint32m1_t ivs = __riscv_vsext_vf2_i32m1(
                __riscv_vle16_v_i16mf2(rw + p, vl), vl);
            vfloat32m1_t v = __riscv_vfmul_vf_f32m1(
                __riscv_vfcvt_f_x_v_f32m1(ivs, vl), 1.0f / PSA_SC, vl);
            v = __riscv_vfsub_vf_f32m1(v, mx, vl);
            v = __riscv_vfmul_vf_f32m1(v, 1.4426950408889634f, vl);
            v = exp2f_poly(__riscv_vfmax_vf_f32m1(v, -126.0f, vl), vl);
            acc = __riscv_vfadd_vv_f32m1(acc, v, vl);
            const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, 127.0f, vl);
            const vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(q, vl);
            const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(iv, vl);
            const vint8mf4_t b8 = __riscv_vncvt_x_x_w_i8mf4(h16, vl);
            const size_t vl16 = __riscv_vsetvl_e8mf4(16);
            __riscv_vse8_v_i8mf4(wb + (p / 16) * 64 + r * 16, b8, vl16);
            __riscv_vse8_v_i8mf4(wb + (p / 16 + 1) * 64 + r * 16,
                                 __riscv_vslidedown_vx_i8mf4(b8, 16, vl16), vl16);
        }
        for (; p < len; p += 16) {
            const int n = (len - p < 16) ? (len - p) : 16;
            const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
            const vint32m1_t ivs = __riscv_vsext_vf2_i32m1(
                __riscv_vle16_v_i16mf2(rw + p, vl), vl);
            vfloat32m1_t v = __riscv_vfmul_vf_f32m1(
                __riscv_vfcvt_f_x_v_f32m1(ivs, vl), 1.0f / PSA_SC, vl);
            v = __riscv_vfsub_vf_f32m1(v, mx, vl);
            v = __riscv_vfmul_vf_f32m1(v, 1.4426950408889634f, vl);
            v = exp2f_poly(__riscv_vfmax_vf_f32m1(v, -126.0f, vl), vl);
            // accumulate at vl, not the full length: lanes at or past the tail are
            // not part of this row and would corrupt the normalisation sum
            acc = __riscv_vfadd_vv_f32m1(acc, v, vl);
            const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, 127.0f, vl);
            const vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(q, vl);
            const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(iv, vl);
            const vint8mf4_t b8 = __riscv_vncvt_x_x_w_i8mf4(h16, vl);
            __riscv_vse8_v_i8mf4(wb + (p / 16) * 64 + r * 16, b8, vl);
        }
        const float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
            acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlz));
        wsc[rr] = sum > 0 ? 1.0f / (127.0f * sum) : 0.0f;
    }
    g_pfa_sm[g_pfa_tid] += now_s() - _s;

    // PV: two passes of 64 dims, contracting 16 positions per step. The four row
    // groups read the same V blocks, so they stay hot across the tile.
    const double _v = now_s();
    alignas(64) static __thread int32_t praw[8][32];
    const int nq = (nn + 15) / 16;
    for (int half = 0; half < 2; ++half) {
        const int8_t * vb = g_v8i + (kv * g_v8i_npb16 + half * g_v8i_npb16) * 1024;
        for (int gg = 0; gg < I8_G; ++gg) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) praw[j];
            pv4(op, w8 + gg * WBLK, vb, nq);
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
            if (layer != 0 || !g_pfa_out) continue;
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
    g_pfa_pv[g_pfa_tid] += now_s() - _v;
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
    g_pfa_qk[g_pfa_tid] += now_s() - _q;

    // causal trim + softmax, one row at a time
    for (int r = 0; r < PFA_ROWS; ++r) {
        float * row = g_psa[g_pfa_tid][r];
        const int len = (int) (g_pfa_ctx + qbase + r + 1);
        for (int i = len; i < len_max; ++i) row[i] = -INFINITY;
        pfa_softmax_row(row, len, &g_pfa_sm[g_pfa_tid]);
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
    g_pfa_pv[g_pfa_tid] += now_s() - _v;
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
        g_k8i = (int8_t *) hugemem((size_t) N_KV * ((ctx + rows + 63) / 64) * 8192 + (2 << 20));
        g_v8i = (int8_t *) hugemem((size_t) N_KV * ((ctx + rows + 15) / 16) * 2048 + (2 << 20));
        g_q8i = (int8_t *) hugemem((size_t) N_HEAD * ((rows + 3) / 4) * 512 + (2 << 20));
        g_q8isc = (float *) aligned_alloc(64, (size_t) rows * N_HEAD * 4);
        if (!g_k8i || !g_v8i || !g_q8i || !g_q8isc) { fprintf(stderr, "oom attn int8\n"); return; }
        g_psa16 = (int16_t (*)[I8_ROWS][MAXCTX]) hugemem(
            (size_t) N_GPU * sizeof(int16_t) * I8_ROWS * MAXCTX);
        if (!g_psa16) { fprintf(stderr, "oom scores\n"); return; }
        const double tq0 = now_s();
        pfa_pack_q(rows);
        printf("  int8 pack Q: %.3f s (%.1f MB)\n", now_s() - tq0,
               (double) N_HEAD * ((rows + 3) / 4) * 512 / 1e6);
    }

    for (int t = 0; t < N_GPU; ++t) g_pfa_qk[t] = g_pfa_sm[t] = g_pfa_pv[t] = 0;
    const double t0 = now_s();
    for (int L = 0; L < N_LAYER; ++L) {
        if (g_pfa_i8) pfa_convert_kv_i8(L, ctx + rows);
        else          pfa_convert_kv(L, ctx + rows);
        g_pfa_layer = L;
        run(g_pfa_i8 ? task_pfa_i8 : task_pfa);
    }
    const double dt = now_s() - t0;
    const float first = g_pfa_out[3 * N_HEAD * HEAD_DIM];
    printf("  run1: out[0]=%.6f ksum=%.6f kmx=%.6f\n", first, g_pfa_dbg_sum[0], g_pfa_dbg_mx[0]);
    if (g_pfa_i8) pfa_convert_kv_i8(0, ctx + rows); else pfa_convert_kv(0, ctx + rows);
    g_pfa_layer = 0;
    run(g_pfa_i8 ? task_pfa_i8 : task_pfa);
    printf("  run2: out[0]=%.6f ksum=%.6f kmx=%.6f  (repeatable=%s)\n",
           g_pfa_out[3 * N_HEAD * HEAD_DIM], g_pfa_dbg_sum[0], g_pfa_dbg_mx[0],
           first == g_pfa_out[3 * N_HEAD * HEAD_DIM] ? "yes" : "NO");

    // total QK+PV MACs actually performed
    double macs = 0;
    const int tile = g_pfa_i8 ? I8_ROWS : PFA_ROWS;
    for (int64_t qb = 0; qb + tile <= rows; qb += tile)
        for (int r = 0; r < tile; ++r) macs += (double) (ctx + qb + r + 1) * HEAD_DIM * 2.0;
    macs *= (double) N_LAYER * N_HEAD;
    double tq = 0, ts = 0, tv = 0;
    for (int t = 0; t < N_GPU; ++t) { tq += g_pfa_qk[t]; ts += g_pfa_sm[t]; tv += g_pfa_pv[t]; }
    printf("  %.3f s -> %.1f GMAC/s (%.1f per core)  [qk %.2f sm %.2f pv %.2f s]\n",
           dt, macs / dt / 1e9, macs / dt / 1e9 / g_participants, tq, ts, tv);

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
    const int64_t pfa_stride = g_i8w ? ((6144 / 16) * 64 + 64) : ((6144 / 32) * 152);
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
            if (use_i8) { for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed); }
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
        for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
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

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char * path = "/dev/shm/mc-q4o.gguf";
    int ngen = 16, prompt = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) g_kstep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-X") && i + 1 < argc) g_dbg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q3")) g_bit3 = 1;
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
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) g_pf_tile = atoll(argv[++i]);
    }

    g_blk_bytes = g_bit3 ? Q3_BLK : 576;
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
        items[i].ref->k = e->k; items[i].ref->n = e->n;
        total += (size_t) (e->n / 32) * (g_bit3 ? (e->k / Q3_K) * Q3_BLK : (e->k / 32) * 576);
    }
    const size_t tot_pad = (total + 4096 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
    const char * cache = g_bit3 ? "/dev/shm/mega-w3.bin" : "/dev/shm/mega-w.bin";
    struct stat cs;
    bool loaded = false;
    g_wblob = (uint8_t *) hugemem(tot_pad);
    if (!g_wblob) { fprintf(stderr, "oom weights\n"); return 1; }
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
            const bool rope_pair = strstr(items[i].name, "attn_q.weight") ||
                                   strstr(items[i].name, "attn_k.weight");
            if (g_bit3) {
                repack_b_q3(p, e->data, e->n, e->k, rope_pair ? g_perm : nullptr);
                p += (size_t) (e->n / 32) * (e->k / Q3_K) * Q3_BLK;
            } else {
                repack_b(p, e->data, e->n, e->k, rope_pair ? g_perm : nullptr);
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
            p += g_bit3 ? (size_t) (items[i].ref->n / 32) * (items[i].ref->k / Q3_K) * Q3_BLK
                        : (size_t) (items[i].ref->n / 32) * (items[i].ref->k / 32) * 576;
        }
        printf("weights: %.1f MB (cached)\n", total / 1e6);
    }

    // Optional int8 copy for prefill. Kept separate from the compact format
    // because decode streams every byte and wants the smallest representation,
    // while prefill is compute bound and wants the densest instruction.
    if (g_i8w) {
        size_t t8 = 0;
        for (int i = 0; i < ii; ++i)
            t8 += ((size_t) items[i].ref->n * items[i].ref->k + (size_t) items[i].ref->n * 16 + 4095) & ~(size_t) 4095;
        const size_t pad8 = (t8 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
        uint8_t * blob8 = (uint8_t *) hugemem(pad8);
        if (!blob8) { fprintf(stderr, "oom int8 weights\n"); return 1; }
        use_huge(blob8, pad8);
        const char * c8 = "/dev/shm/mega-w8b.bin";
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
                repack_b_i8(p, e->data, e->n, e->k, rope_pair ? g_perm : nullptr);
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
    g_aq       = (uint8_t *) aligned_alloc(64, (size_t) N_FF * 2 + 8192);
    const size_t kbytes = (size_t) N_LAYER * K_LAYER_STRIDE * 2;
    const size_t vbytes = (size_t) N_LAYER * KV_LAYER_STRIDE * 2;
    g_kdim = (_Float16 *) hugemem(kbytes);
    g_vpos = (_Float16 *) hugemem(vbytes);
    if (!g_kdim || !g_vpos) { fprintf(stderr, "oom kv\n"); return 1; }
    use_huge(g_kdim, kbytes);
    use_huge(g_vpos, vbytes);
    printf("kv: K %.0f MB (tiled), V %.0f MB (pos-major), f16\n", kbytes / 1e6, vbytes / 1e6);

    // Attention scratch has a 10.5 KB row stride, so without huge pages every
    // head row lands on its own page and the softmax reduction pays page walks.
    fprintf(stderr, "[dbg] before use_huge\n"); fflush(stderr);
    use_huge(g_sa, sizeof g_sa);
    use_huge(g_acc, sizeof g_acc);
    madvise(g_lmax, sizeof g_lmax, MADV_HUGEPAGE);
    madvise(g_lsum, sizeof g_lsum, MADV_HUGEPAGE);
    bind_cpu(g_cpu0);
    for (int i = 1; i < g_participants; ++i) {
        pthread_t t;
        pthread_create(&t, nullptr, worker, (void *) (intptr_t) i);
        pthread_detach(t);
    }

    fprintf(stderr, "[dbg] threads up, participants=%d pf_rows=%lld\n", g_participants, (long long)g_pf_rows); fflush(stderr);
    tcm_attach(0);
    fprintf(stderr, "[tcm] size=%lld buf=%p\n", (long long) g_tcm_size, (void*) g_tcm[0]);
    if (g_pfa_rows > 0) { bench_prefill_attn(); return 0; }
    if (g_pf_rows > 0) { bench_prefill_gemm(g_pf_rows); return 0; }

    int tok = prompt;
    const double t_start = now_s();
    printf("decode: %d tokens from %d\n", ngen, prompt);
    for (int step = 0; step < ngen; ++step) {
        TICK_DECL;
        if (step == ngen - 1) {
            memcpy(g_snap, g_stage, sizeof g_snap);
            g_snap_rope[0] = g_t_rope_math;
            g_snap_rope[1] = g_t_rope_app;
        }
        g_pos = step;
        TICK();
        dequant_row(g_x, g_emb->data + (int64_t) tok * (N_EMBD / 32) * 18, N_EMBD);
        TOCK(S_EMB);

        for (int L = 0; L < N_LAYER; ++L) {
            g_layer = L;
            TICK();
            rmsnorm_par(g_xa, g_x, g_attn_norm[L], N_EMBD);
            TOCK(S_NORM);

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

            TICK();
            for (int h = 0; h < N_HEAD; ++h)
                for (int d = 0; d < HEAD_DIM; ++d) g_attn_out[h * HEAD_DIM + d] = 0;
            attention();
            TOCK(S_ATTN);

            quant_a(g_attn_out, N_EMBD);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wov[L].b, g_wov[L].k, g_wov[L].n, g_xa };
            run(task_mgemm);
            TOCK(S_GEMM);
            TICK();
            parallel_for(N_EMBD, residual_add_fn);
            TOCK(S_MISC);

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

            TICK();
            parallel_for(N_FF, silu_fn);
            TOCK(S_SILU);

            quant_a(g_gate, N_FF);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wd[L].b, g_wd[L].k, g_wd[L].n, g_xa };
            run(task_mgemm);
            TOCK(S_GEMM);
            TICK();
            parallel_for(N_EMBD, residual_add_fn);
            TOCK(S_MISC);
        }

        TICK();
        rmsnorm_par(g_xa, g_x, g_out_norm, N_EMBD);
        TOCK(S_NORM);
        quant_a(g_xa, N_EMBD);
        TICK();
        g_mjn = 1; g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, g_logits };
        run(task_mgemm);
        TOCK(S_GEMM);
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
            if (step < 10) printf("  step %d -> %d (%.4f)\n", step, best, bv);
            tok = best;
        }
        TOCK(S_HEAD);
    }
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
