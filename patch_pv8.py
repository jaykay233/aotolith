import io

p = "mega3.cpp"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- pv8
anchor = "// Repack one layer of K and V from the f16 cache into the dot layouts, and pack Q"
assert anchor in s

pv8 = r'''// PV for 8 queries at once: two weight registers against one V block, so each
// 1 KB of V feeds 8192 MAC instead of 4096 and the loop carries 16 independent
// accumulator chains instead of 8. v10..v25 hold the accumulators, v0/v1 the two
// weight groups, v2..v9 the eight V registers.
static void pv8(int32_t * out, const int8_t * wa, const int8_t * wb,
                const int8_t * vblk, int npblk) {
    asm volatile(
        "vsetvli t0, x0, e32, m1        \n\t"
        "vxor.vv v10, v10, v10          \n\t"
        "vxor.vv v11, v11, v11          \n\t"
        "vxor.vv v12, v12, v12          \n\t"
        "vxor.vv v13, v13, v13          \n\t"
        "vxor.vv v14, v14, v14          \n\t"
        "vxor.vv v15, v15, v15          \n\t"
        "vxor.vv v16, v16, v16          \n\t"
        "vxor.vv v17, v17, v17          \n\t"
        "vxor.vv v18, v18, v18          \n\t"
        "vxor.vv v19, v19, v19          \n\t"
        "vxor.vv v20, v20, v20          \n\t"
        "vxor.vv v21, v21, v21          \n\t"
        "vxor.vv v22, v22, v22          \n\t"
        "vxor.vv v23, v23, v23          \n\t"
        "vxor.vv v24, v24, v24          \n\t"
        "vxor.vv v25, v25, v25          \n\t"
        "mv     t3, %[NP]               \n\t"
        "mv     t4, %[W]                \n\t"
        "mv     t6, %[W2]               \n\t"
        "mv     t5, %[V]                \n\t"
        "1:                             \n\t"
        "vl1re32.v v0, (t4)             \n\t"
        "vl1re32.v v1, (t6)             \n\t"
        "vl8re32.v v2, (t5)             \n\t"
        "addi    t4, t4, 64             \n\t"
        "addi    t6, t6, 64             \n\t"
        "addi    t5, t5, 1024           \n\t"
        "vmadot  v10, v0, v2            \n\t"
        "vmadot  v18, v1, v2            \n\t"
        "vmadot  v11, v0, v3            \n\t"
        "vmadot  v19, v1, v3            \n\t"
        "vmadot  v12, v0, v4            \n\t"
        "vmadot  v20, v1, v4            \n\t"
        "vmadot  v13, v0, v5            \n\t"
        "vmadot  v21, v1, v5            \n\t"
        "vmadot  v14, v0, v6            \n\t"
        "vmadot  v22, v1, v6            \n\t"
        "vmadot  v15, v0, v7            \n\t"
        "vmadot  v23, v1, v7            \n\t"
        "vmadot  v16, v0, v8            \n\t"
        "vmadot  v24, v1, v8            \n\t"
        "vmadot  v17, v0, v9            \n\t"
        "vmadot  v25, v1, v9            \n\t"
        "addi    t3, t3, -1             \n\t"
        "bgtz    t3, 1b                 \n\t"
        "mv      t5, %[O]               \n\t"
        "vse32.v v10, 0(t5)             \n\t"
        "vse32.v v11, 128(t5)           \n\t"
        "vse32.v v12, 256(t5)           \n\t"
        "vse32.v v13, 384(t5)           \n\t"
        "vse32.v v14, 512(t5)           \n\t"
        "vse32.v v15, 640(t5)           \n\t"
        "vse32.v v16, 768(t5)           \n\t"
        "vse32.v v17, 896(t5)           \n\t"
        "vse32.v v18, 1024(t5)          \n\t"
        "vse32.v v19, 1152(t5)          \n\t"
        "vse32.v v20, 1280(t5)          \n\t"
        "vse32.v v21, 1408(t5)          \n\t"
        "vse32.v v22, 1536(t5)          \n\t"
        "vse32.v v23, 1664(t5)          \n\t"
        "vse32.v v24, 1792(t5)          \n\t"
        "vse32.v v25, 1920(t5)          \n\t"
        :
        : [NP] "r"(npblk), [W] "r"(wa), [W2] "r"(wb), [V] "r"(vblk), [O] "r"(out)
        : "t0", "t3", "t4", "t5", "t6",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "vl", "memory");
}

'''
s = s.replace(anchor, pv8 + anchor, 1)

# ---------------------------------------------------------------- call site
old = """    alignas(64) static __thread int32_t praw[8][32];
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
                    printf(" | nq=%d\\n", nq);
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
    }"""
new = """    alignas(64) static __thread int32_t praw8[16][32];
    int32_t * const pwide = &praw8[0][0];
    const int nq = (nn + 15) / 16;
    for (int half = 0; half < 2; ++half) {
        const int8_t * vb = g_v8i + (kv * g_v8i_npb16 + half * g_v8i_npb16) * 1024;
        for (int gg = 0; gg < I8_G; gg += 2) {
            pv8(pwide, w8 + gg * WBLK, w8 + (gg + 1) * WBLK, vb, nq);
            if (half == 0 && gg == 0) { g_dbg_h0 = praw8[0][24]; g_dbg_h0b = praw8[0][25]; }
            if (layer != 0 || !g_pfa_out) continue;
            const size_t vl8 = __riscv_vsetvl_e32m1(8);
            for (int r = 0; r < 8; ++r) {
                const int rr = gg * 4 + r;
                const float s_r = wsc[rr] * g_v8isc[kv];
                float * dst = g_pfa_out + (((qbase + rr) * N_HEAD + h) * HEAD_DIM) + half * 64;
                const int32_t * src = pwide + (r < 4 ? 0 : 256) + (r & 3) * 8;
                for (int j = 0; j < 8; ++j) {
                    const vint32m1_t a = __riscv_vle32_v_i32m1(src + j * 32, vl8);
                    const vfloat32m1_t f = __riscv_vfmul_vf_f32m1(
                        __riscv_vfcvt_f_x_v_f32m1(a, vl8), s_r, vl8);
                    __riscv_vse32_v_f32m1(dst + j * 8, f, vl8);
                }
            }
        }
    }"""
assert old in s
s = s.replace(old, new, 1)

# the old debug block at the end of the function referenced praw[]
del_start = s.index('    if (g_pfa_dbg1 && layer == 0 && h == 0 && qbase == 0 && g_pfa_tid == 0) {\n        g_pfa_dbg1 = 0;\n        printf("[i8 dbg] nn=%d')
del_end = s.index('static void task_pfa_i8(int tid) {')
s = s[:del_start] + "}\n\n" + s[del_end:]

io.open(p, "w", encoding="utf-8").write(s)
print("patched")
