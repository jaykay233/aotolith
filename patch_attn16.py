import io

p = "mega3.cpp"
s = io.open(p, encoding="utf-8").read()

old = "static const int PFA_ROWS = 4;"
new = """static const int PFA_ROWS = 4;
// The int8 path tiles PFA_ROWS*I8_G = 16 queries per head, so four row groups
// consume the same K and V blocks. One K block is 8 KB and stays in L1 across
// the tile; at 4-row tiles every group streamed its own copy from DRAM, which
// left QK bandwidth starved at ~0.4 GB/s of useful traffic.
static const int I8_G = 4;
static const int I8_ROWS = I8_G * PFA_ROWS;
static const int WBLK = (MAXCTX / 16) * 128;   // int8 attention weights per row group"""
assert old in s
s = s.replace(old, new, 1)

old = "static float g_psa[N_GPU][4][MAXCTX];       // per-thread scores"
new = "static float g_psa[N_GPU][I8_ROWS][MAXCTX];   // per-thread scores"
assert old in s
s = s.replace(old, new, 1)

body = r'''// One tile of I8_ROWS queries of one head: QK over every position the tile can
// see, exact causal masking, softmax, requantise the weights, then PV.
static void pfa_head_i8(int layer, int h, int64_t qbase) {
    const int kv = h / NH_PER_KV;
    const int nn = (int) (g_pfa_ctx + qbase + I8_ROWS);
    const int npb = (nn + 63) / 64;
    const int8_t * q8g = g_q8i + ((h * g_q8i_nqg) + qbase / 4) * 8 * 64;
    float * row[I8_ROWS];
    for (int r = 0; r < I8_ROWS; ++r) row[r] = g_psa[g_pfa_tid][r];

    alignas(64) static __thread int32_t raw[8][32];
    alignas(64) static __thread int8_t w8[I8_G * WBLK + 256];
    static __thread float wsc[I8_ROWS];

    const double _q = now_s();
    for (int pb = 0; pb < npb; ++pb) {
        const int8_t * kb = g_k8i + ((kv * g_k8i_npb + pb) * 8) * 1024;
        for (int gg = 0; gg < I8_G; ++gg) {
            int8_t * op[8];
            for (int j = 0; j < 8; ++j) op[j] = (int8_t *) raw[j];
            qk4(op, q8g + gg * 512, kb, HEAD_DIM / 16);
            const size_t vl8 = __riscv_vsetvl_e32m1(8);
            for (int r = 0; r < 4; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = g_q8isc[((qbase + rr) * N_HEAD + h)] * g_k8isc[kv];
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&raw[j][r * 8], vl8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    __riscv_vse32_v_f32m1(row[rr] + pb * 64 + j * 8, f, vl8);
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
    for (int gg = 0; gg < I8_G; ++gg) memset(w8 + gg * WBLK, 0, (size_t) wnb * 128);
    for (int rr = 0; rr < I8_ROWS; ++rr) {
        float * rw = g_psa[g_pfa_tid][rr];
        int8_t * wb = w8 + (rr / 4) * WBLK;
        const int r = rr % 4;
        const int len = (int) (g_pfa_ctx + qbase + rr + 1);
        float mx = -INFINITY;
        int i2 = 0;
        for (; i2 + 16 <= len; i2 += 16) {
            const size_t vl = __riscv_vsetvl_e32m1(16);
            const vfloat32m1_t m = __riscv_vfredmax_vs_f32m1_f32m1(
                __riscv_vle32_v_f32m1(rw + i2, vl), __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vl);
            mx = fmaxf(mx, __riscv_vfmv_f_s_f32m1_f32(m));
        }
        for (; i2 < len; ++i2) mx = fmaxf(mx, rw[i2]);
        const size_t vlz = __riscv_vsetvl_e32m1(16);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlz);
        for (int p = 0; p < len; p += 16) {
            const int n = (len - p < 16) ? (len - p) : 16;
            const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(rw + p, vl);
            v = __riscv_vfsub_vf_f32m1(v, mx, vl);
            v = __riscv_vfmul_vf_f32m1(v, 1.4426950408889634f, vl);
            v = exp2f_poly(__riscv_vfmax_vf_f32m1(v, -126.0f, vl), vl);
            // accumulate at vl, not 16: lanes at or past the tail are not part of
            // this row, and adding them silently corrupts the normalisation sum
            acc = __riscv_vfadd_vv_f32m1(acc, v, vl);
            const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, 127.0f, vl);
            const vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(q, vl);
            const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(iv, vl);
            const vint8mf4_t b8 = __riscv_vncvt_x_x_w_i8mf4(h16, vl);
            __riscv_vse8_v_i8mf4(wb + (p / 16) * 128 + r * 16, b8, vl);
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
            if (half == 0 && gg == 0) { g_dbg_h0 = praw[0][24]; g_dbg_h0b = praw[0][25]; }
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

'''

a = s.index("// One 4-query group of one head: QK over every position it can see, exact causal")
b = s.index("static void task_pfa_i8(int tid) {")
s = s[:a] + body + s[b:]

old = """    for (int64_t qb = 0; qb + PFA_ROWS <= g_pfa_rows; qb += PFA_ROWS)
        for (int h = h0; h < h0 + 2; ++h)
            pfa_head_i8(g_pfa_layer, h, qb);"""
new = """    for (int64_t qb = 0; qb + I8_ROWS <= g_pfa_rows; qb += I8_ROWS)
        for (int h = h0; h < h0 + 2; ++h)
            pfa_head_i8(g_pfa_layer, h, qb);"""
assert old in s
s = s.replace(old, new, 1)

old = """    for (int64_t qb = 0; qb + PFA_ROWS <= rows; qb += PFA_ROWS)
        for (int r = 0; r < PFA_ROWS; ++r) macs += (double) (ctx + qb + r + 1) * HEAD_DIM * 2.0;"""
new = """    const int tile = g_pfa_i8 ? I8_ROWS : PFA_ROWS;
    for (int64_t qb = 0; qb + tile <= rows; qb += tile)
        for (int r = 0; r < tile; ++r) macs += (double) (ctx + qb + r + 1) * HEAD_DIM * 2.0;"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("patched")
