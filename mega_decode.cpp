// MiniCPM5-2B decode megakernel for SpacemiT K3 -- standalone, no ggml graph.
//
// One hand-written loop drives the whole model: 42 layers of fused
// RMSNorm + activation quantization + IME2 i8xi4 GEMM, in-place RoPE with KV
// append, per-head attention, then lm_head and argmax. Only the vendor GEMM and
// its activation quantizer are reused; orchestration, weight repack, attention
// and the elementwise stages are hand-written.
//
// Decode is DRAM-bound: 1.26 GB of repacked weights plus an f16 KV cache. The
// measured raw ceiling on the A100 cluster is ~56 GB/s for 8 streaming threads,
// but the vendor GEMM only reaches ~24 GB/s because each thread runs a serial
// load->use chain over 32-k blocks (few loads in flight). The main lever is
// memory-level parallelism: matrices sharing the same quantized activations
// (q/k/v, gate/up) are issued interleaved from one thread, and within each
// matrix every thread walks its slice in small group steps.
//
// Contracts, both verified numerically against exact dequantized dots:
//   A  per 32-k block: [fp32 scale][int16 sum][int8 x32]  <- rvv::quantize_a_row_i8
//   B  per 32-n group / 32-k block: [fp16 scale x32][row-interleaved int4 x32]
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
#include <sys/stat.h>
#include <riscv_vector.h>

// ---------------------------------------------------------------- vendor kernels
namespace spacemit_kernels { namespace rvv {
void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);
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
static const float RMS_EPS = 1e-6f;
static const float ROPE_BASE = 5000000.0f;
// llama-family checkpoints rotate the first half of each head against the second
// half (x[i], x[i+64]); the GPT-NeoX interleaved pairing is a different model
// family. This switches the pairing without touching the weight layout.
static int g_rope_half = 1;   // llama convention; MEGA_ROPE=inter restores NeoX
static const int64_t KV_LAYER_STRIDE = (int64_t) MAXCTX * N_KV * HEAD_DIM;

// --------------------------------------------------------------- stage profiling
enum Stage { S_EMB, S_NORM, S_QUANT, S_GEMM, S_ATTN, S_SILU, S_ROPE, S_HEAD, S_MISC, S_NSTAGE };
static double g_stage[S_NSTAGE];
static const char * g_stage_name[S_NSTAGE] = {
    "embed", "rmsnorm", "quantA", "gemm", "attn", "silu", "rope", "lm_head+argmax", "misc" };

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#define TICK_DECL double _t0
#define TICK() _t0 = now_s()
#define TOCK(st) g_stage[st] += now_s() - _t0

static void use_huge(void * p, size_t n) {
    if (!p) return;
    madvise(p, n, MADV_HUGEPAGE);
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
// B: per 32-row group, per 32-k block: [fp16 scale x32][16 B row-interleaved int4 x32]
static void repack_b(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k) {
    const int64_t nb = k / 32, rs = nb * 18;
    uint8_t * d = dst;
    for (int64_t b = 0; b < rows; b += 32) {
        for (int64_t x = 0; x < nb; ++x) {
            _Float16 * dd = (_Float16 *) d;
            uint8_t *  qs = d + 64;
            for (int i = 0; i < 32; ++i) {
                const uint8_t * s = src + (b + i) * rs + x * 18;
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

// ------------------------------------------------------- 3-bit weights (Q3_K)
// Weight stream is the whole decode cost, so the bit width is the whole game:
// 4.5 bit/param is 1.26 GB and 19.0 tok/s at the 24 GB/s the vendor GEMM
// actually sustains, while 3.5625 bit/param is 0.98 GB and 24.5 tok/s. The
// vendor i8i3k kernel consumes ggml's Q3_K superblock layout, verified against
// its own reference implementation:
//   int8 scales[16][32], uint16 hmask[16][32], uint8 qs[4][32][16], fp16 s16[32]
// per 32 output columns x 256 K elements. Each stored value is a signed 3-bit
// integer in [-4,3] whose sign is carried by the hmask bit, so the effective
// weight is q3 * int8_scale[row][k16] * fp16_scale[row] -- no extra 1/16 term,
// that factor belongs to the Q3_K reference path only.
static int g_bit3 = 0;
static const int Q3_K = 256;
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;   // 3648 B
static size_t g_blk_bytes = 576;

static void repack_b_q3(uint8_t * dst, const uint8_t * src, int64_t rows, int64_t k) {
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
            for (int i = 0; i < 32; ++i) {
                const uint8_t * row = src + (int64_t) (g + i) * rs;
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
            // int8 scale per (row, k16) step, plus a per-row fp16 normalisation so
            // the int8 scale keeps its full range
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
        while (g_bar_gen.load(std::memory_order_acquire) == g) {
            asm volatile("pause" ::: "memory");
        }
    }
}

typedef void (*task_fn)(int tid);
static task_fn g_task;

static int g_cpu_base = 8;
static int bind_cpu(int cpu) {
    if (g_cpu_base == 8) {
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { if (write(fd, "0", 1) < 0) { /* best effort */ } close(fd); }
    }
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    bind_cpu(g_cpu_base + id);
    for (;;) {
        barrier();
        if (g_quit.load(std::memory_order_acquire)) break;
        g_task(id);
        barrier();
    }
    return nullptr;
}

// caller is tid 0 and participates, so all 8 A cores are used
static void run(task_fn fn) {
    g_task = fn;
    barrier();
    fn(0);
    barrier();
}

// generic parallel-for across the pool
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
struct WRef { const uint8_t * b; int64_t k, n; };
static WRef g_wo, g_wq[N_LAYER], g_wk[N_LAYER], g_wv[N_LAYER], g_wov[N_LAYER];
static WRef g_wg[N_LAYER], g_wu[N_LAYER], g_wd[N_LAYER];
static const Entry * g_emb;
static const float * g_attn_norm[N_LAYER], * g_ffn_norm[N_LAYER], * g_out_norm;

static float * g_x, * g_xa, * g_gate, * g_up, * g_logits;
static float * g_q, * g_k, * g_v, * g_attn_out;
static _Float16 * g_kcache, * g_vcache;
static uint8_t * g_aq;
// One score strip per (thread, row of the round). A round verifies up to
// MROW_MAX rows at once, and they all read the same K, so the strip has to hold
// every row: 8 x 8 x 10 KB is 2.6 MB and removes m-1 redundant reads of K.
static const int MROW_MAX = 8;                    // anchor + up to 7 drafts
static float g_sc[N_GPU][MROW_MAX][MAXCTX + 8];
static float g_ace[N_GPU][MROW_MAX][N_HEAD][HEAD_DIM];
static float g_rope_inv[HEAD_DIM / 2];
static int g_pos, g_layer;
static int64_t g_attn_np;          // positions row g_arow can see

// ------------------------------------------------------------ multi-row decode
// Verifying m drafts at once only pays if the weight stream is paid once rather
// than m times. probe_mscale measured 3 rows at 1.33x the cost of 1, 5 rows at
// 2.17x (DECODE.md). So every round -- plain or speculative -- runs m rows, and
// the GEMM nests rows innermost inside the n-block walk, after the B tile has
// been fetched, so each tile is read from DRAM once and used m times.
//
// The single-row functions above are reused unchanged: select_row repoints the
// global buffers at one row, so norm / rope / attention / silu keep working on one
// row at a time. Only the GEMM needs all rows at once, and it uses the row bases
// directly.
// 8, not 6: the m4 path writes four rows per group and a 5..7 row round needs two
// groups, so the row buffers must hold 4*ceil(m/4) rows or the second group lands in
// whatever follows. Sized for the worst case rather than for the round length.
// Row capacity of the float row buffers. Decode only ever needs MROW_MAX, but
// prefill drives a whole prompt chunk through the same GEMM, so it is a knob.
static int g_rowcap = 8;
static const char * g_prompt_file = nullptr;   // whitespace-separated token ids
// Widest activation row, which is not N_FF: the drafter's encoder feeds fc a
// 5*2048 = 10240-wide fused vector. Sized from that, or the encoder overruns the
// tile by 4864 bytes per row and takes the process down.
#define AQ_ROW_STRIDE (((10240 / 32) * 38))
#define AQ_ROW_MAX_K 10240

static int g_m = 1;            // rows this round
static int g_pos0 = 0;         // absolute position of row 0
static int g_nspec = 0;        // drafts per round
static int32_t g_drafts[MROW_MAX];

static float * b_x, * b_xa, * b_gate, * b_up, * b_q, * b_k, * b_v, * b_attn_out, * b_logits;
static uint8_t * b_aq;

// The IME2 dispatcher routes count_m >= 4 to a native four-row kernel, which packs
// four rows per 32-k block as [f32 scale x4][i16 sum x4][i8 x4 x32] and processes
// them in one sweep. probe_m4 measured it at 1.90x the rate of four separate
// single-row calls, bit-identical output. So m == 4 gets its own path; other row
// counts keep the per-row loop.
static uint8_t * b_aq4;
static int g_use_m4 = 0;
static int g_m4_groups = 1;                           // ceil(m/4), rows padded to 4*groups
#define AQ4_BLK 152                                   // 4*4 + 4*2 + 4*32
#define AQ4_ROW_STRIDE ((AQ_ROW_MAX_K / 32) * AQ4_BLK)

// b_* hold row 0; the existing globals point at the row being processed.
static void select_row(int r) {
    g_x        = b_x        + (int64_t) r * N_EMBD;
    g_xa       = b_xa       + (int64_t) r * N_EMBD;
    g_gate     = b_gate     + (int64_t) r * N_FF;
    g_up       = b_up       + (int64_t) r * N_FF;
    g_q        = b_q        + (int64_t) r * N_EMBD;
    g_k        = b_k        + (int64_t) r * (N_KV * HEAD_DIM);
    g_v        = b_v        + (int64_t) r * (N_KV * HEAD_DIM);
    g_attn_out = b_attn_out + (int64_t) r * N_EMBD;
    g_logits   = b_logits   + (int64_t) r * VOCAB;
    g_aq       = b_aq       + (int64_t) r * AQ_ROW_STRIDE;
}

// ------------------------------------------------------------------ fused pieces
// parallel rmsnorm: partial sums per thread, then a coordinated scale pass
static const float * rn_src, * rn_w;
static float * rn_dst;
static double rn_ss[N_GPU];
static float rn_r;

static thread_local int g_pf_tid_dummy;
static void rn_sum_fn(int64_t lo, int64_t hi) {
    double s = 0;
    for (int64_t i = lo; i < hi; ++i) s += (double) rn_src[i] * rn_src[i];
    rn_ss[g_pf_tid] = s;
}

static void rn_scale_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) rn_dst[i] = rn_src[i] * rn_r * rn_w[i];
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

// dequant one q4_0 row of token_embd into floats
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

// ---- multi-GEMM: several matrices sharing one quantized A, issued interleaved
struct MJob { const uint8_t * b; int64_t k, n; float * c; int64_t ldc; };
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
    for (int64_t s = 0; s < step_max; ++s) {
        for (int j = 0; j < g_mjn; ++j) {
            const int64_t a = n0[j] + s * g_kstep;
            if (a >= n1[j]) continue;
            int64_t b2 = a + g_kstep; if (b2 > n1[j]) b2 = n1[j];
            const int64_t kb = g_mj[j].k / (g_bit3 ? Q3_K : 32);
            const uint8_t * bp = g_mj[j].b + a * kb * g_blk_bytes;
            if (g_use_m4) {
                for (int g = 0; g < g_m4_groups; ++g) {
                    const uint8_t * aq = b_aq4 + (int64_t) g * (kb * AQ4_BLK);
                    float * c = g_mj[j].c + (int64_t) g * 4 * g_mj[j].ldc + a * 32;
                    spacemit_kernels::ime2::gemm_kernel_i8i4(
                        32, aq, bp, nullptr, c,
                        4, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) g_mj[j].ldc);
                }
                continue;
            }
            // rows innermost: bp is now hot in L1, so rows 1..m-1 cost only their
            // MACs instead of another DRAM pass over the same 36 KB of weights.
            for (int r = 0; r < g_m; ++r) {
                const uint8_t * aq = b_aq + (int64_t) r * AQ_ROW_STRIDE;
                float * c = g_mj[j].c + (int64_t) r * g_mj[j].ldc + a * 32;
                if (g_bit3)
                    spacemit_kernels::ime2::gemm_kernel_i8i3k(
                        Q3_K, aq, bp, nullptr, c,
                        1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) g_mj[j].ldc);
                else
                    spacemit_kernels::ime2::gemm_kernel_i8i4(
                        32, aq, bp, nullptr, c,
                        1, (size_t) ((b2 - a) * 32), (size_t) kb, (size_t) g_mj[j].ldc);
            }
        }
    }
}

// split quantize_a into N_GPU contiguous 32-block ranges
static const float * qa_src;
static int64_t qa_k;
static void quant_a(const float * a, int64_t k);   // defined below; stage_a needs it

// Quantise four rows into one m4 tile. Same convention as quantize_a_row_i8:
// scale = amax/127 and sum = -sum(q), so the result is bit-identical to packing
// the rows separately and lets the two paths be compared directly.
static const float * * pack_src;
static int64_t pack_k;
static uint8_t * pack_dst;

// One 32-k block, four rows, into the m4 tile. Mechanically the same work as
// quantize_a_row_i8 done four times; only the store offsets differ, so the RVV
// version costs the same as one row and the interleaving is free. Written for
// vlenb=128 (e32m1 gives vl=32, exactly one block); the caller falls back to the
// scalar loop on other vector lengths.
static inline void pack_m4_blk_rvv(const float * const * src, uint8_t * d) {
    const size_t vl = __riscv_vsetvl_e32m1(32);
    float sc[4];
    int16_t sum[4];
    for (int r = 0; r < 4; ++r) {
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(src[r], vl);
        const vfloat32m1_t va = __riscv_vfabs_v_f32m1(v, vl);
        const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        const float amax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(va, zero, vl));
        const float s_ = amax / 127.0f;
        const float rs = s_ ? 1.0f / s_ : 0.0f;
        sc[r] = s_;
        const vfloat32m1_t vs = __riscv_vfmul_vf_f32m1(v, rs, vl);
        const vint16mf2_t q16 = __riscv_vfncvt_x_f_w_i16mf2(vs, vl);
        const vint8mf4_t q8 = __riscv_vncvt_x_x_w_i8mf4(q16, vl);
        const vint16m1_t t0 = __riscv_vmv_v_x_i16m1(0, vl);
        sum[r] = (int16_t) -__riscv_vmv_x_s_i16m1_i16(__riscv_vwredsum_vs_i8mf4_i16m1(q8, t0, vl));
        __riscv_vse8_v_i8mf4((int8_t *) (d + 24 + 32 * r), q8, vl);
    }
    memcpy(d, sc, 16);
    memcpy(d + 16, sum, 8);
}

static void task_pack_m4(int tid) {
    const int64_t nb = pack_k / 32;
    const int64_t per = (nb + g_participants - 1) / g_participants;
    const int64_t b0 = per * tid;
    int64_t b1 = (tid == g_participants - 1) ? nb : per * (tid + 1);
    if (b1 > nb) b1 = nb;
    if (__riscv_vlenb() == 128) {
        for (int64_t b = b0; b < b1; ++b) {
            const float * rows[4] = { pack_src[0] + b * 32, pack_src[1] + b * 32,
                                      pack_src[2] + b * 32, pack_src[3] + b * 32 };
            pack_m4_blk_rvv(rows, pack_dst + b * AQ4_BLK);
        }
        return;
    }
    for (int64_t b = b0; b < b1; ++b) {
        uint8_t * d = pack_dst + b * AQ4_BLK;
        float sc[4];
        int16_t sum[4];
        int8_t q[4][32];
        for (int r = 0; r < 4; ++r) {
            const float * src = pack_src[r] + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) { const float v = fabsf(src[i]); if (v > amax) amax = v; }
            const float s_ = amax / 127.0f;
            const float rs = s_ ? 1.0f / s_ : 0.0f;
            sc[r] = s_;
            int32_t t = 0;
            for (int i = 0; i < 32; ++i) { q[r][i] = (int8_t) lrintf(src[i] * rs); t += q[r][i]; }
            sum[r] = (int16_t) -t;
        }
        memcpy(d, sc, 16);
        memcpy(d + 16, sum, 8);
        for (int r = 0; r < 4; ++r) memcpy(d + 24 + 32 * r, q[r], 32);
    }
}

// Stage a layer's activations for the GEMM that follows. Kept as one call so the
// m4 tile and the per-row tiles are never mixed up by a stray call site.
static void stage_a(const float * base, int64_t k) {
    if (g_use_m4) {
        // Rows past m are packed from a zero row: their outputs are never read, and
        // zeroing them keeps the padding out of the arithmetic entirely.
        static float zeros[AQ_ROW_MAX_K];
        static const float * rows[4];
        const int nb = (int) (k / 32);
        for (int g = 0; g < g_m4_groups; ++g) {
            for (int r = 0; r < 4; ++r) {
                const int row = g * 4 + r;
                rows[r] = (row < g_m) ? base + (int64_t) row * k
                                      : (r == 0 ? zeros : rows[r - 1]);
                if (row >= g_m) rows[r] = zeros;
            }
            pack_src = rows; pack_k = k; pack_dst = b_aq4 + (int64_t) g * nb * AQ4_BLK;
            run(task_pack_m4);
        }
        return;
    }
    for (int r = 0; r < g_m; ++r) {
        g_aq = b_aq + (int64_t) r * AQ_ROW_STRIDE;
        quant_a(base + (int64_t) r * k, k);
    }
}

static void task_quant(int tid) {
    const int64_t nb = qa_k / (g_bit3 ? Q3_K : 32);
    const int64_t per = (nb + g_participants - 1) / g_participants;
    const int64_t b0 = per * tid;
    int64_t b1 = (tid == g_participants - 1) ? nb : per * (tid + 1);
    if (b1 > nb) b1 = nb;
    if (b0 >= b1) return;
    if (g_bit3)
        spacemit_kernels::rvv::quantize_a_row_i8k(Q3_K, qa_src + b0 * Q3_K,
                                                  (size_t) ((b1 - b0) * Q3_K), g_aq + b0 * 292);
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

// ---- attention
//
// The scalar version this replaces cost 0.074 ms per position, because a 128-term
// dot was 128 scalar float multiplies plus an f16->f32 widen each time. At 10K
// positions that extrapolates to ~755 ms/token, so it had to go.
//
// On A100 (vlenb 128) one e32,m4 register group holds exactly HEAD_DIM = 128
// lanes, so a whole head-wide dot is one vle16 + vfwcvt + vfmacc + vfredusum. The
// same code works on X100 (vlenb 32) where it takes four iterations of 32 lanes,
// because the loop asks for its vl instead of assuming one.
static inline float dot_head(const float * q, const _Float16 * k) {
    const size_t vl = __riscv_vsetvl_e32m4(HEAD_DIM);
    vfloat32m4_t a = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int d = 0; d < HEAD_DIM; ) {
        const size_t n = __riscv_vsetvl_e32m4(HEAD_DIM - d);
        const vfloat16m2_t kh = __riscv_vle16_v_f16m2(k + d, n);
        const vfloat32m4_t kf = __riscv_vfwcvt_f_f_v_f32m4(kh, n);
        const vfloat32m4_t qv = __riscv_vle32_v_f32m4(q + d, n);
        a = __riscv_vfmacc_vv_f32m4(a, qv, kf, n);
        d += (int) n;
    }
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a, zero, vl));
}

// out += w * v, whole head at a time.
static inline void axpy_head(float * out, const _Float16 * v, float w) {
    for (int d = 0; d < HEAD_DIM; ) {
        const size_t n = __riscv_vsetvl_e32m4(HEAD_DIM - d);
        const vfloat16m2_t vh = __riscv_vle16_v_f16m2(v + d, n);
        const vfloat32m4_t vf = __riscv_vfwcvt_f_f_v_f32m4(vh, n);
        const vfloat32m4_t o = __riscv_vle32_v_f32m4(out + d, n);
        __riscv_vse32_v_f32m4(out + d, __riscv_vfmacc_vf_f32m4(o, w, vf, n), n);
        d += (int) n;
    }
}

// Each thread takes N_HEAD/N_GPU heads. With N_HEAD 16 and N_KV 2 the heads
// sharing a KV head are consecutive, so threads 0-3 all walk kv 0 and 4-7 walk
// kv 1: four threads read the same lines, and L2 serves three of them. Making the
// grouping head-major instead was worth 18% in prefill for the same reason.
static void task_attn(int tid) {
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    float * sc = &g_sc[tid][0][0];
    const int nh_per = N_HEAD / N_GPU;
    const int64_t stride = N_KV * HEAD_DIM;
    const int64_t np = g_attn_np;
    for (int hh = 0; hh < nh_per; ++hh) {
        const int h = tid * nh_per + hh;
        const int hkv = h / (N_HEAD / N_KV);
        const float * qh = g_q + h * HEAD_DIM;
        const _Float16 * kbase = g_kcache + (int64_t) g_layer * KV_LAYER_STRIDE + hkv * HEAD_DIM;
        const _Float16 * vbase = g_vcache + (int64_t) g_layer * KV_LAYER_STRIDE + hkv * HEAD_DIM;

        float mx = -INFINITY;
        for (int64_t p = 0; p < np; ++p) {
            const float s = dot_head(qh, kbase + p * stride) * scale;
            sc[p] = s;
            if (s > mx) mx = s;
        }
        float sum = 0;
        for (int64_t p = 0; p < np; ++p) { const float e = expf(sc[p] - mx); sc[p] = e; sum += e; }
        const float inv = 1.0f / sum;

        float * out = g_attn_out + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) out[d] = 0;
        for (int64_t p = 0; p < np; ++p) axpy_head(out, vbase + p * stride, sc[p] * inv);
    }
}

// ---- attention over a whole round's rows at once
// The per-row version above runs m times per layer, and every run streams the
// same 5.2 MB of K and V. At 10K that is the entire reason dspark lost money:
// four rows cost four context scans. This walks positions once, keeps the K (and
// then V) vector live in a register, and dots it against all m queries. The
// vfwcvt that costs ~86 cycles on A100 is also paid once per position instead of
// once per (position, row).
//
// Heads are still split across threads, and heads sharing a kv head stay
// consecutive so the L2 serves the duplicates, as in the per-row version.
static void task_attn_m(int tid) {
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    const int nh_per = N_HEAD / N_GPU;
    const int64_t stride = N_KV * HEAD_DIM;
    const int64_t np = g_attn_np;
    const int m = g_m;
    const size_t vl = __riscv_vsetvl_e32m4(HEAD_DIM);
    if (vl != 32) { /* vlenb 128 only: this kernel assumes 32 e32 lanes */ }

    for (int hh = 0; hh < nh_per; ++hh) {
        const int h = tid * nh_per + hh;
        const int hkv = h / (N_HEAD / N_KV);
        const _Float16 * kbase = g_kcache + (int64_t) g_layer * KV_LAYER_STRIDE + hkv * HEAD_DIM;
        const _Float16 * vbase = g_vcache + (int64_t) g_layer * KV_LAYER_STRIDE + hkv * HEAD_DIM;

        float * sc = &g_sc[tid][0][0];
        float mx[MROW_MAX], sum[MROW_MAX], inv[MROW_MAX];
        for (int r = 0; r < m; ++r) { mx[r] = -INFINITY; sum[r] = 0.0f; }

        for (int64_t p = 0; p < np; ++p) {
            const _Float16 * kp = kbase + p * stride;
            vfloat32m4_t k0, k1, k2, k3;
            k0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(kp + 0, vl), vl);
            k1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(kp + 32, vl), vl);
            k2 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(kp + 64, vl), vl);
            k3 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(kp + 96, vl), vl);
            for (int r = 0; r < m; ++r) {
                const float * qh = b_q + ((int64_t) r * N_EMBD) + h * HEAD_DIM;
                const vfloat32m4_t q0 = __riscv_vle32_v_f32m4(qh + 0, vl);
                const vfloat32m4_t q1 = __riscv_vle32_v_f32m4(qh + 32, vl);
                const vfloat32m4_t q2 = __riscv_vle32_v_f32m4(qh + 64, vl);
                const vfloat32m4_t q3 = __riscv_vle32_v_f32m4(qh + 96, vl);
                const vfloat32m4_t a = __riscv_vfmul_vv_f32m4(q0, k0, vl);
                const vfloat32m4_t b = __riscv_vfmul_vv_f32m4(q1, k1, vl);
                const vfloat32m4_t c = __riscv_vfmul_vv_f32m4(q2, k2, vl);
                const vfloat32m4_t d = __riscv_vfmul_vv_f32m4(q3, k3, vl);
                const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                const vfloat32m4_t s01 = __riscv_vfadd_vv_f32m4(
                    __riscv_vfadd_vv_f32m4(a, b, vl), __riscv_vfadd_vv_f32m4(c, d, vl), vl);
                const float sv = __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(s01, z, vl)) * scale;
                sc[(int64_t) r * (MAXCTX + 8) + p] = sv;
                if (sv > mx[r]) mx[r] = sv;
            }
        }
        for (int r = 0; r < m; ++r) {
            float * row = sc + (int64_t) r * (MAXCTX + 8);
            for (int64_t p = 0; p < np; ++p) {
                const float e = expf(row[p] - mx[r]);
                row[p] = e;
                sum[r] += e;
            }
            inv[r] = 1.0f / sum[r];
        }

        for (int r = 0; r < m; ++r)
            for (int d = 0; d < HEAD_DIM; ++d)
                g_ace[tid][r][h][d] = 0.0f;

        for (int64_t p = 0; p < np; ++p) {
            const _Float16 * vp = vbase + p * stride;
            const vfloat32m4_t v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(vp + 0, vl), vl);
            const vfloat32m4_t v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(vp + 32, vl), vl);
            const vfloat32m4_t v2 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(vp + 64, vl), vl);
            const vfloat32m4_t v3 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(vp + 96, vl), vl);
            for (int r = 0; r < m; ++r) {
                const float w = sc[(int64_t) r * (MAXCTX + 8) + p] * inv[r];
                float * o = g_ace[tid][r][h];
                __riscv_vse32_v_f32m4(o + 0, __riscv_vfmacc_vf_f32m4(__riscv_vle32_v_f32m4(o + 0, vl), w, v0, vl), vl);
                __riscv_vse32_v_f32m4(o + 32, __riscv_vfmacc_vf_f32m4(__riscv_vle32_v_f32m4(o + 32, vl), w, v1, vl), vl);
                __riscv_vse32_v_f32m4(o + 64, __riscv_vfmacc_vf_f32m4(__riscv_vle32_v_f32m4(o + 64, vl), w, v2, vl), vl);
                __riscv_vse32_v_f32m4(o + 96, __riscv_vfmacc_vf_f32m4(__riscv_vle32_v_f32m4(o + 96, vl), w, v3, vl), vl);
            }
        }
    }
    // heads are thread-partitioned, so each output row is complete here
    for (int r = 0; r < m; ++r)
        for (int hh = 0; hh < nh_per; ++hh) {
            const int h = tid * nh_per + hh;
            memcpy(b_attn_out + (int64_t) r * N_EMBD + h * HEAD_DIM,
                   g_ace[tid][r][h], sizeof(float) * HEAD_DIM);
        }
}

// ---- rope over all q heads + kv heads, parallel by head
static void rope_fn(int64_t lo, int64_t hi) {
    for (int64_t h = lo; h < hi; ++h) {
        float * x = (h < N_HEAD) ? (g_q + h * HEAD_DIM) : (g_k + (h - N_HEAD) * HEAD_DIM);
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            const float th = (float) g_pos * g_rope_inv[i];
            const float c = cosf(th), s = sinf(th);
            const int j0 = g_rope_half ? i : 2 * i;
            const int j1 = g_rope_half ? i + HEAD_DIM / 2 : 2 * i + 1;
            const float x0 = x[j0], x1 = x[j1];
            x[j0] = x0 * c - x1 * s;
            x[j1] = x0 * s + x1 * c;
        }
    }
}

static void rope_and_cache(void) {
    parallel_for(N_HEAD + N_KV, rope_fn);
    _Float16 * kc = g_kcache + (int64_t) g_layer * KV_LAYER_STRIDE + (int64_t) g_pos * (N_KV * HEAD_DIM);
    _Float16 * vc = g_vcache + (int64_t) g_layer * KV_LAYER_STRIDE + (int64_t) g_pos * (N_KV * HEAD_DIM);
    for (int i = 0; i < N_KV * HEAD_DIM; ++i) { kc[i] = (_Float16) g_k[i]; vc[i] = (_Float16) g_v[i]; }
}

// ---- silu(gate) * up, vectorised
//
// x * sigmoid(x) = x / (1 + exp(-x)). No vector exp exists, so reduce to
// 2^f on [0,1) with a degree-6 polynomial plus an exponent built by integer
// shifts. x is clamped to +-30 first: outside that, silu(x) is exactly x or 0
// at f32, and it keeps the 2^exponent field in range.
static int g_silu_mode = 3;

// x * sigmoid(x) = x / (1 + exp(-x)).
//
// There is no vector exp, so reduce to 2^f on [0,1) with a degree-6 polynomial
// and build the exponent with integer shifts. The clamp goes on the exponent
// input, not on x: clamping x would change silu(40) from 40 to 30, since the
// numerator must stay the original value. Keeping ie within [-126,127] keeps the
// f32 exponent field in range.
static void silu_fn(int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += 32) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t) (hi - i));
        const vfloat32m1_t x = __riscv_vle32_v_f32m1(g_gate + i, vl);
        const vfloat32m1_t u = __riscv_vle32_v_f32m1(g_up + i, vl);

        vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, -1.4426950408889634f, vl);
        t = __riscv_vfmax_vf_f32m1(t, -126.0f, vl);
        t = __riscv_vfmin_vf_f32m1(t,  126.0f, vl);
        vint32m1_t ie = __riscv_vfcvt_x_f_v_i32m1(t, vl);
        vfloat32m1_t e_int = __riscv_vfcvt_f_x_v_f32m1(ie, vl);
        vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, e_int, vl);

        vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(0.00015403530f, vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0013333558f, vl), vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0096181291f, vl), vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.0555041087f, vl), vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.2402265070f, vl), vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(0.6931471806f, vl), vl);
        p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);

        vint32m1_t ex = __riscv_vadd_vx_i32m1(ie, 127, vl);
        ex = __riscv_vsll_vx_i32m1(ex, 23, vl);
        const vfloat32m1_t sc = __riscv_vreinterpret_v_i32m1_f32m1(ex);
        const vfloat32m1_t e = __riscv_vfmul_vv_f32m1(p, sc, vl);
        const vfloat32m1_t den = __riscv_vfadd_vf_f32m1(e, 1.0f, vl);
        vfloat32m1_t s = __riscv_vfdiv_vv_f32m1(x, den, vl);
        s = __riscv_vfmul_vv_f32m1(s, u, vl);
        __riscv_vse32_v_f32m1(g_gate + i, s, vl);
    }
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

// ============================================================== dspark drafter
// MiniCPM5 2.6B DSpark: a 5-layer head over the target, 324M params, Q4_0 here.
// The architecture and cost model are in DSPARK.md; the authoritative forward pass is
// llama-mtmd/src/models/dflash.cpp and src/models/dspark.cpp.
//
// The drafter only exists to raise the acceptance rate. It cannot make the output
// wrong: the verifier emits a draft only after reproducing it with the target, so a
// bad drafter costs speed and nothing else. That is what makes it safe to take the
// shortcuts below where the reference harness is ambiguous.
static const int DS_LAYER = 5, DS_BLOCK = 7;
static const int DS_TARGET_LAYER[DS_LAYER] = { 2, 11, 21, 31, 40 };
static const int64_t DS_IN = (int64_t) DS_LAYER * N_EMBD;      // 10240 fused input
static const int DS_MARKOV_RANK = 256;

struct DRef { uint8_t * b; int64_t k, n; };

static uint8_t * g_dsmap; static size_t g_dsmap_sz;
static uint8_t * g_dsblob;

static DRef g_ds_wq[DS_LAYER], g_ds_wk[DS_LAYER], g_ds_wv[DS_LAYER], g_ds_wo[DS_LAYER];
static DRef g_ds_wg[DS_LAYER], g_ds_wu[DS_LAYER], g_ds_wd[DS_LAYER];
static DRef g_ds_fc, g_ds_w2;
static const uint8_t * g_ds_w1_raw;
static const float * g_ds_attn_norm[DS_LAYER], * g_ds_ffn_norm[DS_LAYER];
static const float * g_ds_qnorm[DS_LAYER], * g_ds_knorm[DS_LAYER];
static const float * g_ds_enc_norm, * g_ds_out_norm;
static float * g_ds_conf_row;
static float g_ds_conf_b;

static _Float16 * g_dskc, * g_dsvc;                            // [layer][pos][kv head][dim]
static float * b_hid;                                          // [layer][row][embd]
static float * b_concat;
static float * b_dsbias;
static float g_dssc[N_GPU][MAXCTX + 8];
static int g_ds_ready = 0, g_ds_layer, g_ds_on = 0, g_ds_post = 1;
static long long g_ds_hit0 = 0, g_ds_rounds = 0, g_ds_ranksum = 0;
static float * g_ds_lg1 = nullptr;   // draft row-0 distribution, ranked against the target
static int64_t g_ds_np;

// 128-wide RMSNorm, used for the per-head q/k normalisation. Scalar is right here:
// HEAD_DIM is 128 and this runs 16+2 times per row per layer.
static void rmsnorm_vec(float * dst, const float * src, const float * w, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) src[i] * src[i];
    const float r = 1.0f / sqrtf((float) (ss / n) + RMS_EPS);
    for (int i = 0; i < n; ++i) dst[i] = src[i] * r * w[i];
}

// ---------------------------------------------------------------- gguf reading
struct DsInfo { char name[64]; const uint8_t * data; int64_t ne0, ne1; uint32_t type; };
static DsInfo g_dst[96];
static int g_dsn;

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
    const size_t hdr = ((size_t) (p - g_dsmap) + 31) & ~(size_t) 31;   // GGUF alignment 32
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

static void ds_load(void) {
    const char * path = getenv("MEGA_DSPARK");
    if (!path) path = "/home/bianbu/workspace/mega/dspark-q4.gguf";
    ds_parse(path);
    if (!g_dsmap) return;

    size_t need = 0;
    for (int i = 0; i < g_dsn; ++i) {
        if (g_dst[i].type != 2) continue;
        need += (size_t) (g_dst[i].ne1 / 32) * (g_dst[i].ne0 / 32) * 576;
    }
    const size_t blob = (need + (1 << 21)) & ~((size_t) (1 << 21) - 1);
    g_dsblob = (uint8_t *) aligned_alloc(1 << 21, blob);
    if (!g_dsblob) { fprintf(stderr, "dspark: oom\n"); return; }
    use_huge(g_dsblob, blob);

    char nm[64];
    size_t off = 0;
    auto take = [&](DRef & r, const char * n) {
        const DsInfo * t = ds_find(n);
        if (!t || t->type != 2) return;
        repack_b(g_dsblob + off, t->data, t->ne1, t->ne0);
        r.b = g_dsblob + off; r.k = t->ne0; r.n = t->ne1;
        off += (size_t) (t->ne1 / 32) * (t->ne0 / 32) * 576;
    };
    take(g_ds_fc, "fc.weight");
    take(g_ds_w2, "markov_w2.weight");
    const DsInfo * w1 = ds_find("markov_w1.weight");
    if (!w1) return;
    g_ds_w1_raw = w1->data;
    for (int L = 0; L < DS_LAYER; ++L) {
        snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", L);      take(g_ds_wq[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", L);      take(g_ds_wk[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", L);      take(g_ds_wv[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", L); take(g_ds_wo[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", L);    take(g_ds_wg[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", L);      take(g_ds_wu[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", L);    take(g_ds_wd[L], nm);
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", L);   g_ds_attn_norm[L] = (const float *) ds_find(nm)->data;
        snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", L);    g_ds_ffn_norm[L]  = (const float *) ds_find(nm)->data;
        snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", L); g_ds_qnorm[L]     = (const float *) ds_find(nm)->data;
        snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", L); g_ds_knorm[L]     = (const float *) ds_find(nm)->data;
    }
    g_ds_enc_norm = (const float *) ds_find("enc.output_norm.weight")->data;
    g_ds_out_norm = (const float *) ds_find("output_norm.weight")->data;
    if (const DsInfo * cw = ds_find("conf_proj.weight")) {
        g_ds_conf_row = (float *) calloc((size_t) cw->ne0, 4);
        for (int64_t i = 0; i < cw->ne0; ++i) {
            uint16_t h; memcpy(&h, cw->data + 2 * i, 2);
            const uint32_t e = (uint32_t) (h & 0x7FFFu) << 16;
            float f; memcpy(&f, &e, 4);
            g_ds_conf_row[i] = (h & 0x8000u) ? -f : f;
        }
    }
    if (const DsInfo * cb = ds_find("conf_proj.bias")) {
        double v = 0; memcpy(&v, cb->data, 4); g_ds_conf_b = (float) v;
    }

    const size_t kvsz = (size_t) DS_LAYER * KV_LAYER_STRIDE;
    g_dskc = (_Float16 *) aligned_alloc(64, kvsz * 2);
    g_dsvc = (_Float16 *) aligned_alloc(64, kvsz * 2);
    // one row per position, and prefill pushes a whole chunk through at once
    b_hid = (float *) aligned_alloc(64, (size_t) DS_LAYER * g_rowcap * N_EMBD * 4);
    b_dsbias = (float *) aligned_alloc(64, (size_t) (VOCAB + 64) * 4);
    if (b_dsbias) for (int i = 0; i < 64; ++i) b_dsbias[VOCAB + i] = 1.5e30f;
    b_concat = (float *) aligned_alloc(64, (size_t) DS_IN * 4);
    if (!g_dskc || !g_dsvc || !b_hid || !b_dsbias || !b_concat) {
        fprintf(stderr, "dspark: oom kv\n"); return;
    }
    use_huge(g_dskc, kvsz * 2);
    use_huge(g_dsvc, kvsz * 2);

    g_ds_ready = 1;
    printf("dspark: ready (%zu MB repacked, kv %.0f MB)\n", off / 1000000, kvsz * 4 / 1e6);
}

// ------------------------------------------------------------- encoder + inject
// Fuse the target's five tracked hidden states into one 2048-wide vector, then turn it
// into this layer's K/V and append to the draft cache. This is how the draft sees the
// committed prefix: through features rather than tokens.
static void ds_encode_inject(int64_t pos, int row) {
    static float g[2][N_EMBD];
    static float kv[2 * N_KV * HEAD_DIM];
    float * gg = g[0];
    for (int L = 0; L < DS_LAYER; ++L)
        memcpy(b_concat + (int64_t) L * N_EMBD,
               b_hid + ((int64_t) L * g_rowcap + row) * N_EMBD, (size_t) N_EMBD * 4);

    // control: zero features isolate whether the injected prefix reaches the draft at all
    if (getenv("MEGA_DSZERO")) memset(b_concat, 0, (size_t) DS_IN * 4);
    if (getenv("MEGA_DSDBG")) printf("  [ds] enc pos=%lld row=%d: concat ok\n", (long long) pos, row);
    g_m = 1; g_use_m4 = 0; g_m4_groups = 1;
    stage_a(b_concat, DS_IN);
    if (getenv("MEGA_DSDBG")) printf("  [ds] staged a\n");
    g_mjn = 1;
    g_mj[0] = { g_ds_fc.b, g_ds_fc.k, g_ds_fc.n, gg, N_EMBD };
    run(task_mgemm);
    if (getenv("MEGA_DSDBG")) printf("  [ds] fc done\n");
    if (getenv("MEGA_DSDBG3")) {
        double a = 0, b = 0;
        for (int i = 0; i < DS_IN; ++i) a += (double) b_concat[i] * b_concat[i];
        for (int i = 0; i < N_EMBD; ++i) b += (double) gg[i] * gg[i];
        printf("  [enc] pos=%lld |concat|=%.4g |fc_out|=%.4g\n", (long long) pos, sqrt(a), sqrt(b));
    }
    rmsnorm_par(g[1], gg, g_ds_enc_norm, N_EMBD);
    gg = g[1];

    for (int L = 0; L < DS_LAYER; ++L) {
        g_m = 1; g_use_m4 = 0;
        stage_a(gg, N_EMBD);
        g_mjn = 2;
        g_mj[0] = { g_ds_wk[L].b, g_ds_wk[L].k, g_ds_wk[L].n, kv,                  N_KV * HEAD_DIM };
        g_mj[1] = { g_ds_wv[L].b, g_ds_wv[L].k, g_ds_wv[L].n, kv + N_KV * HEAD_DIM, N_KV * HEAD_DIM };
        run(task_mgemm);
        for (int h = 0; h < N_KV; ++h) {
            float * k = kv + h * HEAD_DIM;
            rmsnorm_vec(k, k, g_ds_knorm[L], HEAD_DIM);
            for (int i = 0; i < HEAD_DIM / 2; ++i) {
                const float th = (float) pos * g_rope_inv[i];
                const float c = cosf(th), s = sinf(th);
                const float x0 = k[2 * i], x1 = k[2 * i + 1];
                k[2 * i]     = x0 * c - x1 * s;
                k[2 * i + 1] = x0 * s + x1 * c;
            }
        }
        if (getenv("MEGA_DSDBG")) printf("  [ds] L%d wk/wv done\n", L);
        if (getenv("MEGA_DSDBG3")) {
            double a = 0, b = 0;
            for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
                a += (double) kv[i] * kv[i];
                b += (double) kv[N_KV * HEAD_DIM + i] * kv[N_KV * HEAD_DIM + i];
            }
            printf("  [enc] L%d pos=%lld |K|=%.4g |V|=%.4g\n", L, (long long) pos, sqrt(a), sqrt(b));
        }
        _Float16 * kc = g_dskc + (int64_t) L * KV_LAYER_STRIDE + pos * (N_KV * HEAD_DIM);
        _Float16 * vc = g_dsvc + (int64_t) L * KV_LAYER_STRIDE + pos * (N_KV * HEAD_DIM);
        for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
            kc[i] = (_Float16) kv[i];
            vc[i] = (_Float16) kv[N_KV * HEAD_DIM + i];
        }
    }
}

// --------------------------------------------------------------- block attention
// Each slot attends to the prefix plus slot 0 only (see the block_uniform note where
// g_ds_np is set), so the whole block is one shared query-visible set of pos0 + 1 keys.
static void task_ds_attn(int tid) {
    const int total = DS_BLOCK * N_HEAD;
    const int per = (total + g_participants - 1) / g_participants;
    const int lo = per * tid;
    int hi = (tid == g_participants - 1) ? total : per * (tid + 1);
    if (hi > total) hi = total;
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    const int gqa = N_HEAD / N_KV;
    for (int t = lo; t < hi; ++t) {
        const int r = t / N_HEAD, h = t % N_HEAD;
        const float * q = b_q + (int64_t) r * N_EMBD + h * HEAD_DIM;
        float * out = b_attn_out + (int64_t) r * N_EMBD + h * HEAD_DIM;
        const int hkv = h / gqa;
        const _Float16 * kb = g_dskc + (int64_t) g_ds_layer * KV_LAYER_STRIDE + (int64_t) hkv * HEAD_DIM;
        const _Float16 * vb = g_dsvc + (int64_t) g_ds_layer * KV_LAYER_STRIDE + (int64_t) hkv * HEAD_DIM;
        const int64_t np = g_ds_np;
        float * sc = g_dssc[tid];
        float mx = -INFINITY;
        for (int64_t i = 0; i < np; ++i) {
            const float v = dot_head(q, kb + i * (N_KV * HEAD_DIM)) * scale;
            sc[i] = v;
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < np; ++i) {
            const float e = expf(sc[i] - mx);
            sc[i] = e; sum += e;
        }
        const float rs = sum > 0.0f ? 1.0f / sum : 0.0f;
        if (getenv("MEGA_DSDBG4") && tid == 0 && r == 0 && h == 0) {
            printf("  [attn] L%d np=%lld pos0=%lld w:", g_ds_layer, (long long) np, (long long) (g_ds_np - 1));
            for (int64_t i = 0; i < np && i < 10; ++i) printf(" %.3f", sc[i] * rs);
            printf("  (last %.3f)\n", np > 10 ? sc[np - 1] * rs : 0.0f);
        }
        for (int d = 0; d < HEAD_DIM; ++d) out[d] = 0.0f;
        for (int64_t i = 0; i < np; ++i)
            axpy_head(out, vb + i * (N_KV * HEAD_DIM), sc[i] * rs);
    }
}

// ------------------------------------------------------------------- block forward
// The 7 noise positions through the 5 layers. Returns K drafts, after the Markov chain
// has turned per-position logits into tokens.
static void ds_draft(int32_t anchor, int64_t pos0, int K, int32_t * drafts) {
    if (!g_ds_ready) { for (int i = 0; i < K; ++i) drafts[i] = anchor; return; }
    // control: run the loader and allocate everything but skip the forward, to tell a
    // buffer overlap apart from a wrong draft
    if (getenv("MEGA_DSNOP")) { for (int i = 0; i < K; ++i) drafts[i] = anchor; return; }
    const int m = DS_BLOCK;

    // The draft block is [last token, mask * (block_size - 1)], not the anchor repeated:
    // speculative.cpp fills slot 0 with the committed token and every later slot with
    // llama_vocab_mask(). The vocabulary records that id as tokenizer.ggml.mask_token_id.
    static float ds_mask_emb[N_EMBD];
    static int   ds_mask_done = 0;
    const char * mtenv = getenv("MEGA_DSMASK");
    const int32_t mask_tok = mtenv ? (int32_t) atoi(mtenv) : 75982;
    if (!ds_mask_done) {
        if (mask_tok < 0 || mask_tok >= VOCAB) { fprintf(stderr, "dspark: bad mask token %d\n", mask_tok); return; }
        select_row(0);
        dequant_row(ds_mask_emb, g_emb->data + (int64_t) mask_tok * (N_EMBD / 32) * 18, N_EMBD);
        ds_mask_done = 1;
    }
    for (int i = 0; i < m; ++i) {
        select_row(i);
        if (i == 0)
            dequant_row(g_x, g_emb->data + (int64_t) anchor * (N_EMBD / 32) * 18, N_EMBD);
        else
            memcpy(g_x, ds_mask_emb, (size_t) N_EMBD * 4);
    }

    if ((getenv("MEGA_DSSTAGE") ? atoi(getenv("MEGA_DSSTAGE")) : 4) <= 1) {
        for (int i = 0; i < K; ++i) drafts[i] = anchor;
        return;
    }
    for (int L = 0; L < DS_LAYER; ++L) {
        g_ds_layer = L;
        g_m = m; g_use_m4 = 1; g_m4_groups = (m + 3) / 4;      // 7 rows = 2 groups
        for (int r = 0; r < m; ++r)
            rmsnorm_par(b_xa + (int64_t) r * N_EMBD, b_x + (int64_t) r * N_EMBD,
                        g_ds_attn_norm[L], N_EMBD);
        stage_a(b_xa, N_EMBD);
        g_mjn = 3;
        g_mj[0] = { g_ds_wq[L].b, g_ds_wq[L].k, g_ds_wq[L].n, b_q, N_EMBD };
        g_mj[1] = { g_ds_wk[L].b, g_ds_wk[L].k, g_ds_wk[L].n, b_k, N_KV * HEAD_DIM };
        g_mj[2] = { g_ds_wv[L].b, g_ds_wv[L].k, g_ds_wv[L].n, b_v, N_KV * HEAD_DIM };
        run(task_mgemm);

        for (int r = 0; r < m; ++r) {
            for (int h = 0; h < N_HEAD; ++h) {
                float * q = b_q + (int64_t) r * N_EMBD + h * HEAD_DIM;
                rmsnorm_vec(q, q, g_ds_qnorm[L], HEAD_DIM);
            }
            for (int h = 0; h < N_KV; ++h) {
                float * k = b_k + (int64_t) r * (N_KV * HEAD_DIM) + h * HEAD_DIM;
                rmsnorm_vec(k, k, g_ds_knorm[L], HEAD_DIM);
            }
            const int pos = (int) (pos0 + r);
            for (int h = 0; h < N_HEAD; ++h) {
                float * q = b_q + (int64_t) r * N_EMBD + h * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM / 2; ++i) {
                    const float th = (float) pos * g_rope_inv[i];
                    const float c = cosf(th), s = sinf(th);
                    const float x0 = q[2 * i], x1 = q[2 * i + 1];
                    q[2 * i]     = x0 * c - x1 * s;
                    q[2 * i + 1] = x0 * s + x1 * c;
                }
            }
            for (int h = 0; h < N_KV; ++h) {
                float * k = b_k + (int64_t) r * (N_KV * HEAD_DIM) + h * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM / 2; ++i) {
                    const float th = (float) pos * g_rope_inv[i];
                    const float c = cosf(th), s = sinf(th);
                    const float x0 = k[2 * i], x1 = k[2 * i + 1];
                    k[2 * i]     = x0 * c - x1 * s;
                    k[2 * i + 1] = x0 * s + x1 * c;
                }
            }
            _Float16 * kc = g_dskc + (int64_t) L * KV_LAYER_STRIDE + (pos0 + r) * (N_KV * HEAD_DIM);
            _Float16 * vc = g_dsvc + (int64_t) L * KV_LAYER_STRIDE + (pos0 + r) * (N_KV * HEAD_DIM);
            for (int i = 0; i < N_KV * HEAD_DIM; ++i) {
                kc[i] = (_Float16) b_k[(int64_t) r * (N_KV * HEAD_DIM) + i];
                vc[i] = (_Float16) b_v[(int64_t) r * (N_KV * HEAD_DIM) + i];
            }
        }
        // block_uniform mask: every slot uses the anchor's causal row, so the visible
        // keys are the prefix plus the anchor itself. The seven slots do NOT see each
        // other -- that is what makes the block independently denoisable.
        g_ds_np = pos0 + 1;
        run(task_ds_attn);

        stage_a(b_attn_out, N_EMBD);
        g_mjn = 1;
        g_mj[0] = { g_ds_wo[L].b, g_ds_wo[L].k, g_ds_wo[L].n, b_xa, N_EMBD };
        run(task_mgemm);
        for (int r = 0; r < m; ++r) {
            float * x = b_x + (int64_t) r * N_EMBD;
            const float * xa = b_xa + (int64_t) r * N_EMBD;
            for (int i = 0; i < N_EMBD; ++i) x[i] += xa[i];
        }

        for (int r = 0; r < m; ++r)
            rmsnorm_par(b_xa + (int64_t) r * N_EMBD, b_x + (int64_t) r * N_EMBD,
                        g_ds_ffn_norm[L], N_EMBD);
        stage_a(b_xa, N_EMBD);
        g_mjn = 2;
        g_mj[0] = { g_ds_wg[L].b, g_ds_wg[L].k, g_ds_wg[L].n, b_gate, N_FF };
        g_mj[1] = { g_ds_wu[L].b, g_ds_wu[L].k, g_ds_wu[L].n, b_up, N_FF };
        run(task_mgemm);
        for (int r = 0; r < m; ++r) {
            select_row(r);
            parallel_for(N_FF, silu_fn);
        }
        stage_a(b_gate, N_FF);
        g_mjn = 1;
        g_mj[0] = { g_ds_wd[L].b, g_ds_wd[L].k, g_ds_wd[L].n, b_xa, N_EMBD };
        run(task_mgemm);
        for (int r = 0; r < m; ++r) {
            float * x = b_x + (int64_t) r * N_EMBD;
            const float * xa = b_xa + (int64_t) r * N_EMBD;
            for (int i = 0; i < N_EMBD; ++i) x[i] += xa[i];
        }
    }

    // final norm, then the TARGET's lm_head. Only K positions are needed but the m4
    // path wants four rows, so row 3 runs and is ignored when K < 4.
    const int ds_stage = getenv("MEGA_DSSTAGE") ? atoi(getenv("MEGA_DSSTAGE")) : 4;
    if (ds_stage <= 2) { for (int i = 0; i < K; ++i) drafts[i] = anchor; return; }
    const int rows = K <= 4 ? 4 : K;
    for (int r = 0; r < K; ++r)
        rmsnorm_par(b_xa + (int64_t) r * N_EMBD, b_x + (int64_t) r * N_EMBD,
                    g_ds_out_norm, N_EMBD);
    for (int r = K; r < rows; ++r) memset(b_xa + (int64_t) r * N_EMBD, 0, (size_t) N_EMBD * 4);
    g_m = rows;
    g_use_m4 = (rows >= 4);
    g_m4_groups = g_use_m4 ? (rows + 3) / 4 : 1;
    stage_a(b_xa, N_EMBD);
    g_mjn = 1;
    g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, b_logits, VOCAB };
    run(task_mgemm);

    if (ds_stage <= 3) {
        for (int i = 0; i < K; ++i) drafts[i] = anchor;   // lm_head only, no markov
        return;
    }
    // Markov chain: rank-256 bigram bias, argmaxed greedily so each position conditions
    // on the previous choice. Started from the anchor, as sample_from_anchor implies.
    int32_t prev = anchor;
    static float markov[DS_MARKOV_RANK];
    static uint8_t w1row[8 * 18];
    for (int i = 0; i < K; ++i) {
        memcpy(w1row, g_ds_w1_raw + (int64_t) prev * 8 * 18, 8 * 18);
        for (int b = 0; b < 8; ++b) {
            _Float16 sc; memcpy(&sc, w1row + b * 18, 2);
            const uint8_t * qs = w1row + b * 18 + 2;
            for (int j = 0; j < 16; ++j) {
                markov[b * 32 + j]      = (float) ((int) (qs[j] & 0x0F) - 8) * (float) sc;
                markov[b * 32 + j + 16] = (float) ((int) (qs[j] >> 4) - 8) * (float) sc;
            }
        }
        g_m = 1; g_use_m4 = 0; g_m4_groups = 1;
        if (!getenv("MEGA_DSNOSTAGE")) stage_a(markov, DS_MARKOV_RANK);
        g_mjn = 1;
        g_mj[0] = { g_ds_w2.b, g_ds_w2.k, g_ds_w2.n, b_dsbias, VOCAB };
        if (!getenv("MEGA_DSNOW2")) run(task_mgemm);
        if (b_dsbias && g_ds_w2.n != VOCAB)
            printf("  [markov] w2 n=%lld vs VOCAB=%d\n", (long long) g_ds_w2.n, VOCAB);
        for (int gi = 0; gi < 64; ++gi)
            if (b_dsbias && b_dsbias[VOCAB + gi] != 1.5e30f) {
                printf("  [markov] CANARY clobbered at +%d = %g\n", gi, b_dsbias[VOCAB + gi]);
                b_dsbias[VOCAB + gi] = 1.5e30f;
            }

        const float * lg = b_logits + (int64_t) i * VOCAB;
        if (getenv("MEGA_DSDBG2") && i < 3) {
            double amax = 0, bmax = 0, bamin = 1e30;
            for (int v = 0; v < VOCAB; ++v) {
                const double a = fabs((double) lg[v]);
                if (a > amax) amax = a;
                const double b = fabs((double) b_dsbias[v]);
                if (b > bmax) bmax = b;
                if (b < bamin) bamin = b;
            }
            int bt = 0; float tv = lg[0];
            for (int v = 1; v < VOCAB; ++v) if (lg[v] > tv) { tv = lg[v]; bt = v; }
            printf("  [dbg] i=%d base|max|=%.4g base_top=%d(%.4g) bias|max|=%.4g |min|=%.4g\n",
                   i, amax, bt, tv, bmax, bamin);
        }
        int best = 0;
        float bv = lg[0] + b_dsbias[0];
        for (int v = 1; v < VOCAB; ++v) {
            const float x = lg[v] + b_dsbias[v];
            if (x > bv) { bv = x; best = v; }
        }
        if (i == 0) {
            if (!g_ds_lg1) g_ds_lg1 = (float *) malloc((size_t) VOCAB * 4);
            for (int v = 0; v < VOCAB; ++v) lg[v];
            if (g_ds_lg1) memcpy(g_ds_lg1, lg, (size_t) VOCAB * 4);
        }
        drafts[i] = best;
        prev = best;
    }
}

// The target model's forward for m rows at absolute positions pos0 .. pos0+m-1.
// Decode calls it once per verification block, prefill once per prompt chunk.
// want_logits is decode-only: a prefill chunk would need m * VOCAB floats when only
// the very last prompt position's distribution is ever read.
static void target_forward(int m, int pos0, const int32_t * rowtok, int want_logits) {
    TICK_DECL;
    g_m = m;
    g_use_m4 = (m >= 4 && !g_bit3);
    g_m4_groups = g_use_m4 ? (m + 3) / 4 : 1;
    g_pos0 = pos0;

        TICK();
        for (int r = 0; r < m; ++r) {
            select_row(r);
            dequant_row(g_x, g_emb->data + (int64_t) rowtok[r] * (N_EMBD / 32) * 18, N_EMBD);
        }
        TOCK(S_EMB);

        for (int L = 0; L < N_LAYER; ++L) {
            g_layer = L;
            // All rows are normalized and quantized before the GEMM because the
            // GEMM walks rows innermost and needs every row's activation ready.
            TICK();
            for (int r = 0; r < m; ++r) {
                select_row(r);
                rmsnorm_par(g_xa, g_x, g_attn_norm[L], N_EMBD);
            }
            TOCK(S_NORM);

            TICK();
            stage_a(b_xa, N_EMBD);
            TOCK(S_QUANT);

            TICK();
            g_mjn = 3;
            g_mj[0] = { g_wq[L].b, g_wq[L].k, g_wq[L].n, b_q, N_EMBD };
            g_mj[1] = { g_wk[L].b, g_wk[L].k, g_wk[L].n, b_k, N_KV * HEAD_DIM };
            g_mj[2] = { g_wv[L].b, g_wv[L].k, g_wv[L].n, b_v, N_KV * HEAD_DIM };
            run(task_mgemm);
            TOCK(S_GEMM);

            // Each row is rotated and cached at its own position, then attends to
            // everything up to and including itself. Rows are written before any
            // attention runs, so row r sees the earlier drafts of its own block.
            TICK();
            for (int r = 0; r < m; ++r) {
                select_row(r);
                g_pos = g_pos0 + r;
                rope_and_cache();
            }
            TOCK(S_ROPE);

            TICK();
            // One pass for the whole round: K/V are streamed once and dot against
            // every row, which is what makes verifying m drafts affordable at 10K.
            // The per-row path is kept as MEGA_ONEROW for A/B against it.
            if (getenv("MEGA_ONEROW")) {
                for (int r = 0; r < m; ++r) {
                    select_row(r);
                    g_attn_np = g_pos0 + r + 1;
                    run(task_attn);
                }
            } else {
                // every row of the round sees up to the last row's position; the
                // causal limit of each row is applied when the logits are picked,
                // and row r < m-1 keeps only its own prefix in target_forward
                g_attn_np = g_pos0 + m;
                run(task_attn_m);
            }
            TOCK(S_ATTN);

            TICK();
            stage_a(b_attn_out, N_EMBD);
            TOCK(S_QUANT);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wov[L].b, g_wov[L].k, g_wov[L].n, b_xa, N_EMBD };
            run(task_mgemm);
            TOCK(S_GEMM);
            TICK();
            for (int r = 0; r < m; ++r) {
                float * x = b_x + (int64_t) r * N_EMBD;
                const float * xa = b_xa + (int64_t) r * N_EMBD;
                for (int i = 0; i < N_EMBD; ++i) x[i] += xa[i];
            }
            TOCK(S_MISC);

            TICK();
            for (int r = 0; r < m; ++r) {
                select_row(r);
                rmsnorm_par(g_xa, g_x, g_ffn_norm[L], N_EMBD);
            }
            TOCK(S_NORM);

            TICK();
            stage_a(b_xa, N_EMBD);
            TOCK(S_QUANT);
            TICK();
            g_mjn = 2;
            g_mj[0] = { g_wg[L].b, g_wg[L].k, g_wg[L].n, b_gate, N_FF };
            g_mj[1] = { g_wu[L].b, g_wu[L].k, g_wu[L].n, b_up, N_FF };
            run(task_mgemm);
            TOCK(S_GEMM);

            TICK();
            for (int r = 0; r < m; ++r) {
                select_row(r);
                parallel_for(N_FF, silu_fn);
            }
            TOCK(S_SILU);

            TICK();
            stage_a(b_gate, N_FF);
            TOCK(S_QUANT);
            TICK();
            g_mjn = 1; g_mj[0] = { g_wd[L].b, g_wd[L].k, g_wd[L].n, b_xa, N_EMBD };
            run(task_mgemm);
            TOCK(S_GEMM);
            TICK();
            for (int r = 0; r < m; ++r) {
                float * x = b_x + (int64_t) r * N_EMBD;
                const float * xa = b_xa + (int64_t) r * N_EMBD;
                for (int i = 0; i < N_EMBD; ++i) x[i] += xa[i];
            }
            // The drafter's encoder consumes the residual stream at these layers. The
            // reference does not say whether extract_layers means the input to layer L
            // or its output, so both are available and the accuracy metric decides.
            if (g_ds_on && !g_ds_post)
                for (int k = 0; k < DS_LAYER; ++k)
                    if (DS_TARGET_LAYER[k] == L + 1)
                        for (int r = 0; r < m; ++r)
                            memcpy(b_hid + ((int64_t) k * g_rowcap + r) * N_EMBD,
                                   b_x + (int64_t) r * N_EMBD, (size_t) N_EMBD * 4);
            if (g_ds_on && g_ds_post)
                for (int k = 0; k < DS_LAYER; ++k)
                    if (DS_TARGET_LAYER[k] == L)
                        for (int r = 0; r < m; ++r)
                            memcpy(b_hid + ((int64_t) k * g_rowcap + r) * N_EMBD,
                                   b_x + (int64_t) r * N_EMBD, (size_t) N_EMBD * 4);
            TOCK(S_MISC);
        }

    if (!want_logits) return;
    if (m > MROW_MAX) {
        fprintf(stderr, "target_forward: %d rows of logits exceeds %d\n", m, MROW_MAX);
        return;
    }

        TICK();
        for (int r = 0; r < m; ++r) {
            select_row(r);
            rmsnorm_par(g_xa, g_x, g_out_norm, N_EMBD);
        }
        stage_a(b_xa, N_EMBD);
        TOCK(S_NORM);
        TICK();
        g_mjn = 1; g_mj[0] = { g_wo.b, g_wo.k, g_wo.n, b_logits, VOCAB };
        run(task_mgemm);
        TOCK(S_GEMM);
}

// The board's glibc fscanf returns 1 without writing the int back, which fills
// the token buffer with stack garbage and segfaults on the embedding lookup. Scan
// the file by hand instead; same reason as mega3.cpp.
static int32_t * scan_ints(const char * path, int * nout) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * txt = (char *) malloc((size_t) sz + 1);
    if (!txt) { fclose(f); return nullptr; }
    const size_t got = fread(txt, 1, (size_t) sz, f);
    fclose(f);
    txt[got] = 0;
    int32_t * out = (int32_t *) malloc(((size_t) got / 2 + 8) * 4);
    if (!out) { free(txt); return nullptr; }
    int n = 0;
    const char * q = txt;
    while (*q) {
        while (*q && (*q < '0' || *q > '9') && *q != '-') ++q;
        if (!*q) break;
        const bool neg = (*q == '-');
        if (neg) ++q;
        if (*q < '0' || *q > '9') continue;
        int32_t v = 0;
        while (*q >= '0' && *q <= '9') v = v * 10 + (*q++ - '0');
        out[n++] = neg ? -v : v;
    }
    free(txt);
    *nout = n;
    return out;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char * path = work_path("mc-q4o.gguf");
    int ngen = 16, prompt = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) g_kstep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q3")) g_bit3 = 1;
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_nspec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-D")) g_ds_on = 1;
        else if (!strcmp(argv[i], "-Dpre")) { g_ds_on = 1; g_ds_post = 0; }
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) g_prompt_file = argv[++i];
    }
    // The packed-block size drives every weight offset, so it has to follow the
    // mode before the blob is sized or repacked.
    g_blk_bytes = g_bit3 ? Q3_BLK : 576;
    // Must be settled before any row buffer is sized; rounded to whole four-row
    // groups because the m4 path always works in units of four.
    if (const char * rs = getenv("MEGA_ROWS")) {
        const int v = atoi(rs);
        if (v >= 4) g_rowcap = (v + 3) / 4 * 4;
    }
    if (const char * rp = getenv("MEGA_ROPE")) g_rope_half = strcmp(rp, "inter") != 0;
    if (g_nspec < 0) g_nspec = 0;
    if (g_nspec > MROW_MAX - 1) g_nspec = MROW_MAX - 1;

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
    size_t total = 0;
    for (int i = 0; i < ii; ++i) {
        const Entry * e = find(items[i].name);
        if (!e) { fprintf(stderr, "missing %s\n", items[i].name); return 1; }
        items[i].ref->k = e->k; items[i].ref->n = e->n;
        total += (size_t) (e->n / 32) * (e->k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes;
    }
    printf("weights: %.1f MB repacked across %d matrices\n", total / 1e6, ii);

    const size_t tot_pad = (total + 4096 + (2 << 20) - 1) / (2 << 20) * (2 << 20);
    const char * cache = work_path(g_bit3 ? "mega-w3.bin" : "mega-w.bin");
    struct stat cs;
    bool loaded = false;
    g_wblob = (uint8_t *) aligned_alloc(2 << 20, tot_pad);
    if (!g_wblob) { fprintf(stderr, "oom weights\n"); return 1; }
    use_huge(g_wblob, tot_pad);
    if (stat(cache, &cs) == 0 && (size_t) cs.st_size == total) {
        int cf = open(cache, O_RDONLY);
        size_t got = 0;
        while (got < total) {
            const size_t want = (total - got > (64u << 20)) ? (64u << 20) : (total - got);
            ssize_t r = read(cf, g_wblob + got, want);
            if (r <= 0) break;
            got += (size_t) r;
        }
        close(cf);
        loaded = (got == total);
    }
    if (!loaded) {
        double t0 = now_s();
        uint8_t * p = g_wblob;
        for (int i = 0; i < ii; ++i) {
            const Entry * e = find(items[i].name);
            items[i].ref->b = p;
            if (g_bit3) repack_b_q3(p, e->data, e->n, e->k);
            else        repack_b(p, e->data, e->n, e->k);
            p += (size_t) (e->n / 32) * (e->k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes;
        }
        printf("weights: repacked in %.2f s\n", now_s() - t0);
        int cf = open(cache, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (cf >= 0) { size_t put = 0; while (put < total) { ssize_t w = write(cf, g_wblob + put, total - put); if (w <= 0) break; put += (size_t) w; } close(cf); }
    } else {
        uint8_t * p = g_wblob;
        for (int i = 0; i < ii; ++i) {
            items[i].ref->b = p;
            p += (size_t) (items[i].ref->n / 32) *
                 (items[i].ref->k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes;
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

    b_x        = (float *) aligned_alloc(64, (size_t) g_rowcap * N_EMBD * 4);
    b_xa       = (float *) aligned_alloc(64, (size_t) g_rowcap * N_EMBD * 4);
    b_gate     = (float *) aligned_alloc(64, (size_t) g_rowcap * N_FF * 4);
    b_up       = (float *) aligned_alloc(64, (size_t) g_rowcap * N_FF * 4);
    b_q        = (float *) aligned_alloc(64, (size_t) g_rowcap * N_EMBD * 4);
    b_k        = (float *) aligned_alloc(64, (size_t) g_rowcap * N_KV * HEAD_DIM * 4);
    b_v        = (float *) aligned_alloc(64, (size_t) g_rowcap * N_KV * HEAD_DIM * 4);
    b_attn_out = (float *) aligned_alloc(64, (size_t) g_rowcap * N_EMBD * 4);
    // logits are only ever read for a handful of rows (a decode block, or the
    // last prefill position), so they keep the decode width whatever the rowcap is
    b_logits   = (float *) aligned_alloc(64, (size_t) MROW_MAX * VOCAB * 4);
    b_aq       = (uint8_t *) aligned_alloc(64, (size_t) g_rowcap * AQ_ROW_STRIDE + 64);
    b_aq4      = (uint8_t *) aligned_alloc(64, (size_t) (g_rowcap / 4) * AQ4_ROW_STRIDE + 4096);
    if (!b_x || !b_xa || !b_gate || !b_up || !b_q || !b_k || !b_v ||
        !b_attn_out || !b_logits || !b_aq || !b_aq4) { fprintf(stderr, "oom rows\n"); return 1; }
    select_row(0);
    const size_t kvsz = (size_t) N_LAYER * KV_LAYER_STRIDE;
    g_kcache = (_Float16 *) aligned_alloc(64, kvsz * 2);
    g_vcache = (_Float16 *) aligned_alloc(64, kvsz * 2);
    if (!g_kcache || !g_vcache) { fprintf(stderr, "oom kv\n"); return 1; }
    use_huge(g_kcache, kvsz * 2);
    use_huge(g_vcache, kvsz * 2);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
        g_rope_inv[i] = 1.0f / powf(ROPE_BASE, (float) (2 * i) / (float) HEAD_DIM);

    if (g_ds_on) ds_load();
    bind_cpu(g_cpu_base);
    if (getenv("MEGA_CPU_BASE")) g_cpu_base = atoi(getenv("MEGA_CPU_BASE"));
    if (getenv("MEGA_NG")) {
        g_participants = atoi(getenv("MEGA_NG"));
        if (g_participants < 1) g_participants = 1;
        if (g_participants > N_GPU) g_participants = N_GPU;
    }
    for (int i = 1; i < g_participants; ++i) {
        pthread_t t;
        pthread_create(&t, nullptr, worker, (void *) (intptr_t) i);
        pthread_detach(t);
    }

    // ---- micro: raw streaming read of the same weight blob, same threads
    if (getenv("MEGA_RAW")) {
        static volatile uint64_t raw_sink;
        const size_t bytes = ((size_t) (g_wo.n / 32) * (g_wo.k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes) & ~(size_t) 511;
        for (int round = 0; round < 3; ++round) {
            const double a = now_s();
            run([](int tid) {
                const size_t bytes = ((size_t) (g_wo.n / 32) * (g_wo.k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes) & ~(size_t) 511;
                const size_t per = bytes / g_participants;
                const uint8_t * p = g_wo.b + per * tid;
                const uint8_t * e = p + per;
                uint64_t acc = 0;
                while (p + 512 <= e) {
                    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
                    memcpy(&a0, p + 0, 8);   memcpy(&a1, p + 64, 8);
                    memcpy(&a2, p + 128, 8); memcpy(&a3, p + 192, 8);
                    memcpy(&a4, p + 256, 8); memcpy(&a5, p + 320, 8);
                    memcpy(&a6, p + 384, 8); memcpy(&a7, p + 448, 8);
                    acc ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
                    p += 512;
                }
                if (acc == 0x5a5a5a5a5a5a5a5aULL) raw_sink = acc;
            });
            const double d = now_s() - a;
            printf("  raw stream %zu MB  %6.3f ms  %6.1f GB/s\n", bytes >> 20, d * 1e3, bytes / d / 1e9);
        }
        g_quit.store(1, std::memory_order_release);
        barrier();
        return 0;
    }

    // ---- micro: per-matrix streaming rate through the same call path
    if (getenv("MEGA_BW")) {
        struct { const char * nm; WRef w; float * c; } t[] = {
            { "output  150MB", g_wo,    g_logits },
            { "ffn_gate  7MB", g_wg[0], g_gate },
            { "ffn_down  7MB", g_wd[0], g_xa },
            { "attn_q  2.4MB", g_wq[0], g_q },
        };
        for (auto & e : t) {
            double best = 1e9;
            for (int r = 0; r < 8; ++r) {
                const double a = now_s();
                quant_a(g_xa, e.w.k);
                g_mjn = 1; g_mj[0] = { e.w.b, e.w.k, e.w.n, e.c };
                run(task_mgemm);
                const double d = now_s() - a;
                if (r >= 2 && d < best) best = d;
            }
            const double mb = (double) (e.w.n / 32) *
                              (e.w.k / (g_bit3 ? Q3_K : 32)) * g_blk_bytes / 1e6;
            printf("  %s  %6.3f ms  %6.1f GB/s\n", e.nm, best * 1e3, mb / 1e3 / best);
        }
        g_quit.store(1, std::memory_order_release);
        barrier();
        return 0;
    }

    // ---- prefill
    // The prompt enters as token ids (llama-tokenize emits exactly these), goes
    // through the same target_forward the decode round uses, in chunks of g_rowcap
    // rows, and leaves its KV in the cache at positions 0 .. n-2. The last prompt
    // token is deliberately not consumed here: it becomes the decode round's anchor,
    // so the very first round already produces the first generated token.
    int32_t * ptok = nullptr;
    int n_prompt = 0;
    if (g_prompt_file) {
        ptok = scan_ints(g_prompt_file, &n_prompt);
        if (!ptok) return 1;
        printf("prompt: %d tokens from %s\n", n_prompt, g_prompt_file);
    }

    int produced = 0;
    if (n_prompt > 0) {
        const double tp0 = now_s();
        int chunk = g_rowcap;
        if (const char * pc = getenv("MEGA_PFCHUNK")) {
            const int v = atoi(pc);
            if (v >= 1 && v < chunk) chunk = v;
        }
        int done = 0;
        // leave the last token for the decode round
        const int total = n_prompt - 1;
        while (done < total) {
            const int cm = (total - done < chunk) ? (total - done) : chunk;
            target_forward(cm, done, ptok + done, 0);
            // The drafter sees committed positions only through fused target
            // features, so every prefilled position has to be injected too.
            if (g_ds_on)
                for (int r = 0; r < cm; ++r) ds_encode_inject(done + r, r);
            done += cm;
        }
        const double tp = now_s() - tp0;
        memset(g_stage, 0, sizeof g_stage);   // prefill must not pollute decode stats
        printf("prefill: %d tokens in %.3f s (%.2f tok/s, %d-row chunks)\n",
               total, tp, total / tp, chunk);
        produced = total;
    }

    // ---- decode
    int tok = (n_prompt > 0) ? ptok[n_prompt - 1] : prompt;
    const double t_start = now_s();
    // Replay drafts: a file of token ids from a plain greedy run, so the round has
    // known-correct drafts and acceptance must be m-1 on every round. This is how
    // the multi-row path is checked for exactness without any draft model.
    int32_t * g_replay = nullptr;
    int g_nreplay = 0;
    if (const char * rp = getenv("MEGA_REPLAY")) {
        g_replay = scan_ints(rp, &g_nreplay);
        if (!g_replay) return 1;
        printf("replay: %d tokens from %s\n", g_nreplay, rp);
    }

    printf("decode: %d tokens from pos %d (kstep=%d, drafts=%d, rows=%d)\n",
           ngen, produced, g_kstep, g_nspec, g_nspec + 1);

    // emitted holds the whole sequence, prompt included, so the dump is directly
    // comparable with the reference model's full token stream.
    int32_t * emitted = (int32_t *) malloc((size_t) (n_prompt + ngen + 8) * 4);
    if (n_prompt > 0) memcpy(emitted, ptok, (size_t) n_prompt * 4);
    // With a prompt, produced starts at the anchor's position (the last prompt
    // token), so generated tokens land one slot after the copied prompt.
    const int em_off = (n_prompt > 0) ? 1 : 0;
    int rounds = 0;
    long long accepted_drafts = 0;
    int32_t pred[MROW_MAX];
    // produced is an absolute position, so the bound is "prefix + ngen", not ngen.
    const int n_target = produced + ngen;
    while (produced < n_target) {
        TICK_DECL;
        const int K = g_nspec;
        const int m = K + 1;
        g_m = m;
        g_use_m4 = (m >= 4 && !g_bit3);
        g_m4_groups = g_use_m4 ? (m + 3) / 4 : 1;
        g_pos0 = produced;   // row r sits at position produced + r

        int32_t rowtok[MROW_MAX];
        rowtok[0] = tok;
        if (g_ds_on && K > 0) {
            TICK();
            ds_draft(tok, produced, K, g_drafts);
            TOCK(S_GEMM);
            // With a replay file the round must follow the known-good continuation, so the
            // drafter is scored against the real target instead of steering the run.
            for (int r = 0; r < K; ++r)
                rowtok[r + 1] = g_replay ? g_replay[produced + r] : g_drafts[r];
            // The drafter drives the row geometry for its own block and its markov head is
            // single row, so the target's geometry is restored before anything stages A.
            g_m = m;
            g_use_m4 = (m >= 4 && !g_bit3);
            g_m4_groups = g_use_m4 ? (m + 3) / 4 : 1;
        } else {
            for (int r = 0; r < K; ++r)
                // replay[i] is the i-th token plain greedy emitted, and `produced` of
                // them are already committed, so the draft for row r+1 is replay[produced+r].
                rowtok[r + 1] = g_replay ? g_replay[produced + r] : tok;
        }

        // Drafter first: it drives the row geometry for its own block. Then one
        // forward for the whole block, however those rows were chosen.
        TICK();
        if (g_ds_on && K > 0) {
            ds_draft(tok, produced, K, g_drafts);
            TOCK(S_GEMM);
            // With a replay file the round must follow the known-good continuation, so the
            // drafter is scored against the real target instead of steering the run.
            for (int r = 0; r < K; ++r)
                rowtok[r + 1] = g_replay ? g_replay[produced + r] : g_drafts[r];
        } else {
            TOCK(S_GEMM);
            for (int r = 0; r < K; ++r)
                // replay[i] is the i-th token plain greedy emitted, and `produced` of
                // them are already committed, so the draft for row r+1 is replay[produced+r].
                rowtok[r + 1] = g_replay ? g_replay[produced + r] : tok;
        }

        target_forward(m, produced, rowtok, 1);

        // Row r's argmax is the target's greedy choice after position pos0+r, so it
        // is directly comparable with draft r+1's token.
        TICK();
        for (int r = 0; r < m; ++r) {
            const float * lg = b_logits + (int64_t) r * VOCAB;
            int best = 0;
            float bv = lg[0];
            for (int i = 1; i < VOCAB; ++i)
                if (lg[i] > bv) { bv = lg[i]; best = i; }
            pred[r] = best;
        }
        TOCK(S_HEAD);

        // Longest matching prefix: row r is kept only if it predicted the draft that
        // row r+1 consumed. The last kept row contributes its own prediction, so
        // acceptance is at least 1 and anything accepted is byte-identical to what
        // plain greedy would have emitted.
        if (g_ds_on && K > 0) {
            ++g_ds_rounds;
            if (g_drafts[0] == pred[0]) ++g_ds_hit0;
            if (g_ds_lg1) {
                const float ref = g_ds_lg1[pred[0]];
                int rank = 0;
                for (int v = 0; v < VOCAB; ++v) if (g_ds_lg1[v] > ref) ++rank;
                g_ds_ranksum += rank;
                if (getenv("MEGA_DSDBG") && rounds < 12)
                    printf("  [ds] target0=%d rank_in_draft=%d (top1=%d)\n", pred[0], rank, g_drafts[0]);
            }
            if (getenv("MEGA_DSDBG") && rounds < 8) {
                printf("  [ds] draft0=%d target0=%d  %s\n", g_drafts[0], pred[0],
                       g_drafts[0] == pred[0] ? "HIT" : "miss");
            }
        }
        int acc = 1;
        while (acc <= K && pred[acc - 1] == rowtok[acc]) ++acc;
        accepted_drafts += acc - 1;

        if (rounds < 6 || getenv("MEGA_VERBOSE")) {
            printf("  round %d: drafts", rounds);
            for (int r = 0; r < K; ++r) printf(" %d", rowtok[r + 1]);
            printf(" | accepted %d/%d | next %d\n", acc - 1, K, pred[acc - 1]);
        }

        // rows 0..acc-1 are now committed; row acc-1's prediction is the next token
        for (int i = 0; i < acc && produced + em_off + i < n_prompt + ngen + 8; ++i)
            emitted[produced + em_off + i] = (i + 1 < acc) ? rowtok[i + 1] : pred[acc - 1];
        // The committed positions enter the drafter through fused target features, so
        // they are injected after acceptance, using the rows that were just verified.
        if (g_ds_on)
            for (int i = 0; i < acc; ++i) ds_encode_inject(produced + i, i);
        produced += acc;
        tok = pred[acc - 1];
        ++rounds;
    }
    if (const char * tp = getenv("MEGA_TOKOUT")) {
        FILE * f = fopen(tp, "w");
        if (f) {
            const int nout = produced + em_off;
            for (int i = 0; i < nout; ++i) fprintf(f, "%d ", emitted[i]);
            fputc('\n', f);
            fclose(f);
            printf("tokens -> %s\n", tp);
        }
    }
    if (g_ds_on && g_ds_rounds > 0)
        printf("dspark: first-draft accuracy %lld/%lld = %.1f%%, mean rank of target token %.0f\n",
               g_ds_hit0, g_ds_rounds, 100.0 * (double) g_ds_hit0 / (double) g_ds_rounds,
               (double) g_ds_ranksum / (double) g_ds_rounds);
    if (g_nspec > 0 && rounds > 0)
        printf("spec: %d rounds, %.2f drafts accepted per round (max %d)\n",
               rounds, (double) accepted_drafts / rounds, g_nspec);

    const double dt = now_s() - t_start;
    const int n_gen = produced - (n_prompt > 0 ? n_prompt - 1 : 0);
    printf("--- stage breakdown (ms/token) ---\n");
    for (int i = 0; i < S_NSTAGE; ++i)
        if (g_stage[i] > 0) printf("  %-16s %7.2f  (%4.1f%%)\n", g_stage_name[i],
                                   g_stage[i] * 1000.0 / (rounds ? rounds : 1),
                                   100.0 * g_stage[i] / dt);
    printf("decode: %d tokens in %d rounds, %.3f s -> %.2f tok/s (%.1f ms/round)\n",
           n_gen, rounds, dt, n_gen / dt, rounds ? dt * 1e3 / rounds : 0.0);

    g_quit.store(1, std::memory_order_release);
    barrier();
    return 0;
}
