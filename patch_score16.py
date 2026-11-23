import io

p = "mega3.cpp"
s = io.open(p, encoding="utf-8").read()

# score buffer: f32 -> int16 fixed point (scale 1/128, range +-256)
old = "static float g_psa[N_GPU][I8_ROWS][MAXCTX];   // per-thread scores"
new = """static float g_psa[N_GPU][PFA_ROWS][MAXCTX];    // per-thread scores (f16 path)
// The int8 path keeps its raw scores in int16 fixed point at 1/128. A row of
// scores is 40 KB in f32 and is written once and read twice, which made the
// score buffer - not the dots - the prefill attention bottleneck: 13.4 GB per
// layer of writes plus 26.8 GB of reads, i.e. DRAM bound at ~11 GB/s. int16
// halves that, and the max pass can run in the integer domain.
static const float PSA_SC = 128.0f;
static int16_t (*g_psa16)[I8_ROWS][MAXCTX];"""
assert old in s
s = s.replace(old, new, 1)

# QK epilogue: store int16
old = """            const size_t vl8 = __riscv_vsetvl_e32m1(8);
            for (int r = 0; r < 4; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = g_q8isc[((qbase + rr) * N_HEAD + h)] * g_k8isc[kv];
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&raw[j][r * 8], vl8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    __riscv_vse32_v_f32m1(row[rr] + pb * 64 + j * 8, f, vl8);
                }
            }"""
new = """            const size_t vl8 = __riscv_vsetvl_e32m1(8);
            for (int r = 0; r < 4; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = g_q8isc[((qbase + rr) * N_HEAD + h)] * g_k8isc[kv] * PSA_SC;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(&raw[j][r * 8], vl8);
                    vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    f = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(f, -32760.0f, vl8),
                                               32760.0f, vl8);
                    const vint16mf2_t h16 = __riscv_vncvt_x_x_w_i16mf2(
                        __riscv_vfcvt_x_f_v_i32m1(f, vl8), vl8);
                    __riscv_vse16_v_i16mf2(row16[rr] + pb * 64 + j * 8, h16, vl8);
                }
            }"""
assert old in s
s = s.replace(old, new, 1)

# row pointers: two views of the tile
old = """    float * row[I8_ROWS];
    for (int r = 0; r < I8_ROWS; ++r) row[r] = g_psa[g_pfa_tid][r];"""
new = """    int16_t * row16[I8_ROWS];
    for (int r = 0; r < I8_ROWS; ++r) row16[r] = &g_psa16[g_pfa_tid][r][0];"""
assert old in s
s = s.replace(old, new, 1)

# softmax: read int16, widen, undo the store scale
old = """        float * rw = g_psa[g_pfa_tid][rr];
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
        for (; i2 < len; ++i2) mx = fmaxf(mx, rw[i2]);"""
new = """        int16_t * rw = row16[rr];
        int8_t * wb = w8 + (rr / 4) * WBLK;
        const int r = rr % 4;
        const int len = (int) (g_pfa_ctx + qbase + rr + 1);
        // max in the integer domain: the scale is positive, so it is order
        // preserving and no conversion is needed on the pass that touches every
        // score
        int mx16 = -32768;
        int i2 = 0;
        for (; i2 + 16 <= len; i2 += 16) {
            const size_t vl = __riscv_vsetvl_e16m1(16);
            const vint16m1_t v = __riscv_vle16_v_i16m1(rw + i2, vl);
            const vint16m1_t m = __riscv_vredmax_vs_i16m1_i16m1(
                v, __riscv_vmv_s_x_i16m1((int16_t) -32768, 1), vl);
            const int t = (int) __riscv_vmv_x_s_i16m1_i16(m);
            if (t > mx16) mx16 = t;
        }
        for (; i2 < len; ++i2) if (rw[i2] > mx16) mx16 = rw[i2];
        const float mx = (float) mx16 / PSA_SC;"""
assert old in s
s = s.replace(old, new, 1)

old = """            const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(rw + p, vl);
            v = __riscv_vfsub_vf_f32m1(v, mx, vl);"""
new = """            const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
            const vint32m1_t iv = __riscv_vsext_vf2_i32m1(
                __riscv_vle16_v_i16mf2(rw + p, vl), vl);
            vfloat32m1_t v = __riscv_vfmul_vf_f32m1(
                __riscv_vfcvt_f_x_v_f32m1(iv, vl), 1.0f / PSA_SC, vl);
            v = __riscv_vfsub_vf_f32m1(v, mx, vl);"""
assert old in s
s = s.replace(old, new, 1)

# allocate the int16 score buffer with the rest of the attention scratch
old = """        const double tq0 = now_s();
        pfa_pack_q(rows);"""
new = """        g_psa16 = (int16_t (*)[I8_ROWS][MAXCTX]) hugemem(
            (size_t) N_GPU * sizeof(int16_t) * I8_ROWS * MAXCTX);
        if (!g_psa16) { fprintf(stderr, "oom scores\\n"); return; }
        const double tq0 = now_s();
        pfa_pack_q(rows);"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("patched")
