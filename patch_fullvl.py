import io

p = "mega3.cpp"
s = io.open(p, encoding="utf-8").read()

# ---- 1) QK epilogue at full vector length -------------------------------------
old = """    const double _q = now_s();
    for (int pb = 0; pb < npb; ++pb) {"""
new = """    // Per-lane row scales, built once per tile: with VLMAX lanes the epilogue
    // converts all four rows and all eight columns of one raw register in a
    // single pass instead of four 8-lane passes.
    alignas(64) static __thread float svt[I8_G][32];
    for (int gg = 0; gg < I8_G; ++gg)
        for (int r = 0; r < 4; ++r) {
            const float s_r = g_q8isc[((qbase + gg * 4 + r) * N_HEAD + h)] * g_k8isc[kv] * PSA_SC;
            for (int c = 0; c < 8; ++c) svt[gg][r * 8 + c] = s_r;
        }

    const double _q = now_s();
    for (int pb = 0; pb < npb; ++pb) {"""
assert old in s
s = s.replace(old, new, 1)

old = """            const size_t vl8 = __riscv_vsetvl_e32m1(8);
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
new = """            const size_t vl32 = __riscv_vsetvl_e32m1(32);
            const vfloat32m1_t sv = __riscv_vle32_v_f32m1(svt[gg], vl32);
            const size_t vl8 = __riscv_vsetvl_e16m1(8);
            for (int j = 0; j < 8; ++j) {
                __riscv_vsetvl_e32m1(32);
                const vint32m1_t a = __riscv_vle32_v_i32m1(raw[j], vl32);
                const vfloat32m1_t f = __riscv_vfmul_vv_f32m1(
                    __riscv_vfcvt_f_x_v_f32m1(a, vl32), sv, vl32);
                const vint16m1_t h16 = __riscv_vncvt_x_x_w_i16m1(
                    __riscv_vfcvt_x_f_v_i32m1(f, vl32), vl32);
                for (int r = 0; r < 4; ++r) {
                    const vint16m1_t sl = r
                        ? __riscv_vslidedown_vx_i16m1(h16, (size_t) (r * 8), vl8) : h16;
                    __riscv_vse16_v_i16m1(row16[gg * 4 + r] + pb * 64 + j * 8, sl, vl8);
                }
            }"""
assert old in s
s = s.replace(old, new, 1)

# ---- 2) softmax: 32 positions per iteration ------------------------------------
old = """        int mx16 = -32768;
        int i2 = 0;
        for (; i2 + 16 <= len; i2 += 16) {
            const size_t vl = __riscv_vsetvl_e16m1(16);
            const vint16m1_t v = __riscv_vle16_v_i16m1(rw + i2, vl);
            const vint16m1_t m = __riscv_vredmax_vs_i16m1_i16m1(
                v, __riscv_vmv_s_x_i16m1((int16_t) -32768, 1), vl);
            const int t = (int) __riscv_vmv_x_s_i16m1_i16(m);
            if (t > mx16) mx16 = t;
        }"""
new = """        int mx16 = -32768;
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
        }"""
assert old in s
s = s.replace(old, new, 1)

# exp pass: two 16-position blocks per iteration, arithmetic at 32 lanes
old = """        const size_t vlz = __riscv_vsetvl_e32m1(16);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlz);
        for (int p = 0; p < len; p += 16) {
            const int n = (len - p < 16) ? (len - p) : 16;
            const size_t vl = __riscv_vsetvl_e32m1((size_t) n);
            const vint32m1_t ivs = __riscv_vsext_vf2_i32m1(
                __riscv_vle16_v_i16mf2(rw + p, vl), vl);
            vfloat32m1_t v = __riscv_vfmul_vf_f32m1(
                __riscv_vfcvt_f_x_v_f32m1(ivs, vl), 1.0f / PSA_SC, vl);
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
        }"""
new = """        const size_t vlz = __riscv_vsetvl_e32m1(32);
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
            __riscv_vse8_v_i8mf4(wb + (p / 16) * 128 + r * 16, b8, vl16);
            __riscv_vse8_v_i8mf4(wb + (p / 16 + 1) * 128 + r * 16,
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
            __riscv_vse8_v_i8mf4(wb + (p / 16) * 128 + r * 16, b8, vl);
        }"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("patched")
