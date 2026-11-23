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

// Same accumulation shape as rmsnorm_par (double partial sums) so prefill and
// decode agree on the residual stream bit for bit where it matters.
static void rmsnorm_rows(float * dst, const float * src, const float * w, int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = src + r * N_EMBD;
        float * y = dst + r * N_EMBD;
        double ss = 0;
        for (int i = 0; i < N_EMBD; ++i) ss += (double) x[i] * x[i];
        const float rn = 1.0f / sqrtf((float) (ss / N_EMBD) + RMS_EPS);
        for (int i = 0; i < N_EMBD; ++i) y[i] = x[i] * rn * w[i];
    }
}

static void silu_rows(float * g, int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) {
        float * p = g + r * N_FF;
        for (int i = 0; i < N_FF; ++i) p[i] = p[i] / (1.0f + expf(-p[i]));
    }
}

static void add_rows(float * dst, const float * src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
}

// Rotates one row's Q and K at its own position. Half-split pairing, the llama
// convention: (x[i], x[i + HEAD_DIM/2]) share an angle. Q also carries the
// 1/sqrt(head_dim) factor, because the caller packs it straight into the int8
// attention operand.
static void prefill_qk_row(float * q, float * k, int64_t pos) {
    const float * cs = g_rope_tab + pos * 128;
    const float inv = 1.0f / sqrtf((float) HEAD_DIM);
    for (int h = 0; h < N_HEAD + N_KV; ++h) {
        float * x = (h < N_HEAD) ? (q + h * HEAD_DIM) : (k + (h - N_HEAD) * HEAD_DIM);
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            const float c = cs[i], s = cs[64 + i];
            const float x0 = x[i], x1 = x[i + HEAD_DIM / 2];
            x[i]                = x0 * c - x1 * s;
            x[i + HEAD_DIM / 2] = x0 * s + x1 * c;
        }
        if (h < N_HEAD)
            for (int i = 0; i < HEAD_DIM; ++i) x[i] *= inv;
    }
}

// The chunk's rotated K goes into the tiled f16 cache that attention reads, and
// V into the position-major one. Both are the arrays decode uses as well.
static void store_kv_rows(int L, int64_t base, int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = base + r;
        const int pblk = (int) (p / KT), pi = (int) (p % KT);
        const float * k = g_pfk + r * (N_KV * HEAD_DIM);
        const float * v = g_pfv + r * (N_KV * HEAD_DIM);
        _Float16 * vb = g_vpos + (int64_t) L * KV_LAYER_STRIDE + p * N_KV * HEAD_DIM;
        for (int kv = 0; kv < N_KV; ++kv) {
            _Float16 * kb = ktile(L, kv, pblk);
            for (int d = 0; d < HEAD_DIM; ++d) {
                kb[(int64_t) d * KT + pi] = (_Float16) k[kv * HEAD_DIM + d];
                vb[kv * HEAD_DIM + d]     = (_Float16) v[kv * HEAD_DIM + d];
            }
        }
    }
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
    const int64_t NP = npref;
    g_a8_stride = (N_FF / 16) * 64 + 128;
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
    g_k8i     = (int8_t *) hugemem((size_t) N_KV * ((NP + 63) / 64) * 8192 + (2 << 20));
    g_v8i     = (int8_t *) hugemem((size_t) N_KV * ((NP + 15) / 16) * 2048 + (2 << 20));
    g_q8i     = (int8_t *) hugemem((size_t) N_HEAD * ((R + 3) / 4) * 512 + (2 << 20));
    g_psa16   = (int16_t (*)[I8_ROWS][MAXCTX]) hugemem((size_t) N_GPU * sizeof(int16_t) * I8_ROWS * MAXCTX);
    g_q8isc   = (float *) aligned_alloc(64, (size_t) R * N_HEAD * 4);
    if (!g_pfa || !g_pfx || !g_pfxa || !g_pfgate || !g_pfup || !g_pfk || !g_pfv ||
        !g_pq || !g_pfa_out || !g_k8i || !g_v8i || !g_q8i || !g_psa16 || !g_q8isc) {
        fprintf(stderr, "oom prefill buffers\n");
        return false;
    }
    use_huge(g_pfa, pfa_bytes);
    use_huge(g_k8i, (size_t) N_KV * ((NP + 63) / 64) * 8192 + (2 << 20));
    use_huge(g_v8i, (size_t) N_KV * ((NP + 15) / 16) * 2048 + (2 << 20));
    use_huge(g_psa16, (size_t) N_GPU * sizeof(int16_t) * I8_ROWS * MAXCTX);
    g_pfa_i8 = 1;
    if (g_pipe) pfa_pipe_alloc();

    const double t0 = now_s();
    for (int64_t base = 0; base < npref; base += R) {
        const int64_t cm = (npref - base < R) ? (npref - base) : R;
        // attention tiles are whole 64-row groups, and the m4 GEMM wants whole
        // four-row groups; 64 covers both
        const int64_t C = (cm + 63) / 64 * 64;
        g_pf_rows = C;

        for (int64_t r = 0; r < cm; ++r)
            dequant_row(g_pfx + r * N_EMBD,
                        g_emb->data + (int64_t) toks[base + r] * (N_EMBD / 32) * 18, N_EMBD);
        for (int64_t r = cm; r < C; ++r) memset(g_pfx + r * N_EMBD, 0, (size_t) N_EMBD * 4);

        for (int L = 0; L < N_LAYER; ++L) {
            rmsnorm_rows(g_pfxa, g_pfx, g_attn_norm[L], C);
            for (int j = 0; j < 3; ++j) quant_a_i8d(g_pfxa, C, N_EMBD, j);
            for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 3;
            g_pfj[0] = { g_wq[L].b8, g_wq[L].k, g_wq[L].n, g_pq,  N_HEAD * HEAD_DIM };
            g_pfj[1] = { g_wk[L].b8, g_wk[L].k, g_wk[L].n, g_pfk, N_KV * HEAD_DIM };
            g_pfj[2] = { g_wv[L].b8, g_wv[L].k, g_wv[L].n, g_pfv, N_KV * HEAD_DIM };
            run(task_pf_gemm_i8);

            for (int64_t r = 0; r < cm; ++r)
                prefill_qk_row(g_pq + r * (N_HEAD * HEAD_DIM),
                               g_pfk + r * (N_KV * HEAD_DIM), base + r);
            store_kv_rows(L, base, cm);

            pfa_convert_kv_i8(L, base + cm);
            g_pfa_ctx = base;
            g_pfa_rows = C;
            pfa_pack_q(C);
            pfa_run(L);

            memset(g_pfa_out + cm * N_EMBD, 0, (size_t) (C - cm) * N_EMBD * 4);
            quant_a_i8d(g_pfa_out, C, N_EMBD, 0);
            for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 1;
            g_pfj[0] = { g_wov[L].b8, g_wov[L].k, g_wov[L].n, g_pfxa, N_EMBD };
            run(task_pf_gemm_i8);
            add_rows(g_pfx, g_pfxa, C * N_EMBD);

            rmsnorm_rows(g_pfxa, g_pfx, g_ffn_norm[L], C);
            for (int j = 0; j < 2; ++j) quant_a_i8d(g_pfxa, C, N_EMBD, j);
            for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 2;
            g_pfj[0] = { g_wg[L].b8, g_wg[L].k, g_wg[L].n, g_pfgate, N_FF };
            g_pfj[1] = { g_wu[L].b8, g_wu[L].k, g_wu[L].n, g_pfup,   N_FF };
            run(task_pf_gemm_i8);

            silu_rows(g_pfgate, C);
            quant_a_i8d(g_pfgate, C, N_FF, 0);
            for (int c = 0; c < 128; ++c) g_chunk[c].store(0, std::memory_order_relaxed);
            g_pfjn = 1;
            g_pfj[0] = { g_wd[L].b8, g_wd[L].k, g_wd[L].n, g_pfxa, N_EMBD };
            run(task_pf_gemm_i8);
            add_rows(g_pfx, g_pfxa, C * N_EMBD);
        }
    }
    const double dt = now_s() - t0;
    printf("prefill: %lld tokens in %.3f s (%.2f tok/s, %lld-row chunks)\n",
           (long long) npref, dt, (double) npref / dt, (long long) R);
    return export_kv(npref);
}

