// Validate the K-quant row decoders against an independent implementation.
//
// Prints, for one tensor, the raw first block bytes and the dequantized first
// values from dequant_row_t. A Python reader fed the same bytes checks the
// arithmetic; that separates "my Q3_K bit unpacking is wrong" from "the vendor
// 3-bit block packing or kernel is wrong", which otherwise look identical from
// the logits.
//
//   probe_kq <gguf> <tensor> <row>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static uint8_t * g_map;
static uint32_t rd_u32(const uint8_t ** p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t rd_q(const uint8_t ** p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static const uint8_t * rd_str(const uint8_t ** p, char * out) {
    const uint64_t n = rd_q(p);
    if (out) { memcpy(out, *p, n < 63 ? n : 63); out[n < 63 ? n : 63] = 0; }
    const uint8_t * s = *p; *p += n; return s;
}
static void skip_kv(const uint8_t ** p, uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: *p += 1; break;
        case 2: case 3: *p += 2; break;
        case 4: case 5: case 6: *p += 4; break;
        case 8: rd_str(p, nullptr); break;
        case 9: {
            const uint32_t at = rd_u32(p);
            uint64_t n = rd_q(p);
            for (uint64_t i = 0; i < n; ++i) skip_kv(p, at);
            break;
        }
        case 10: case 11: *p += 8; break;
        default: *p += 4; break;
    }
}

static inline float f16(const uint8_t * p) { uint16_t h; memcpy(&h, p, 2); _Float16 f; memcpy(&f, &h, 2); return (float) f; }

// Independent of mega3.cpp on purpose: a second transcription of ggml's
// dequantize_row_* so a copy of the same mistake in both is the only way to agree.
static void ref_dequant(float * y, const uint8_t * blk, int type) {
    if (type == 11) {
        static const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        const float d_all = f16(blk + 108);
        const uint8_t * q = blk + 32, * hm = blk;
        uint32_t aux[4];
        memcpy(aux, blk + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t * sc = (const int8_t *) aux;
        int is = 0; uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * (float) ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2; m = (uint8_t) (m << 1);
            }
            q += 32;
        }
    } else if (type == 12) {
        const float d = f16(blk), mn = f16(blk + 2);
        const uint8_t * sc = blk + 4, * q = blk + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            for (int half = 0; half < 2; ++half) {
                const int jj = is + half;
                uint8_t s, m;
                if (jj < 4) { s = sc[jj] & 63; m = sc[jj + 4] & 63; }
                else { s = (sc[jj + 4] & 0xF) | ((sc[jj - 4] >> 6) << 4);
                       m = (sc[jj + 4] >> 4)  | ((sc[jj - 0] >> 6) << 4); }
                const float dd = d * (float) s, bb = mn * (float) m;
                for (int l = 0; l < 32; ++l)
                    *y++ = dd * (float) (half ? (q[l] >> 4) : (q[l] & 0xF)) - bb;
            }
            q += 32; is += 2;
        }
    } else if (type == 14) {
        const uint8_t * ql = blk, * qh = blk + 128;
        const int8_t * sc = (const int8_t *) (blk + 192);
        const float d = f16(blk + 208);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * (float) sc[is + 0] * (float) q1;
                y[l + 32] = d * (float) sc[is + 2] * (float) q2;
                y[l + 64] = d * (float) sc[is + 4] * (float) q3;
                y[l + 96] = d * (float) sc[is + 6] * (float) q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

int main(int argc, char ** argv) {
    const char * path = argv[1];
    const char * want = argc > 2 ? argv[2] : "blk.0.attn_q.weight";
    const int row = argc > 3 ? atoi(argv[3]) : 0;
    const int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    g_map = (uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    const uint8_t * p = g_map + 4; rd_u32(&p);
    const uint64_t nt = rd_q(&p), nkv = rd_q(&p);
    for (uint64_t i = 0; i < nkv; ++i) {
        rd_str(&p, nullptr);
        const uint32_t kt = rd_u32(&p);
        skip_kv(&p, kt);
    }
    char name[64];
    uint32_t type = 0;
    uint64_t ne0 = 0, ne1 = 0;
    const uint8_t * off = nullptr, * data = nullptr;
    for (uint64_t i = 0; i < nt; ++i) {
        rd_str(&p, name);
        const uint32_t nd = rd_u32(&p);
        uint64_t d0 = 0, d1 = 1;
        for (uint32_t d = 0; d < nd; ++d) {
            const uint64_t v = rd_q(&p);
            if (d == 0) d0 = v; else if (d == 1) d1 = v;
        }
        const uint32_t t = rd_u32(&p);
        const uint64_t o = rd_q(&p);
        if (!strcmp(name, want)) { type = t; ne0 = d0; ne1 = d1; off = (const uint8_t *) (uintptr_t) o; }
    }
    (void) off;
    // the data section starts at the first 32-byte-aligned offset after the index
    const uint8_t * base = p;
    while ((uintptr_t) base % 32) ++base;
    data = base + (uintptr_t) off;

    static const int blkbytes[15] = { 0,0,18,20,20,24,24,26,34,36,84,110,144,176,210 };
    const int bb = blkbytes[type];
    static float y[256], z[256];
    const int64_t k = (int64_t) ne0;
    const uint8_t * r0 = data + (int64_t) row * (k / 256) * bb;
    printf("tensor %s type %u ne0 %llu ne1 %llu block %d B\n", want, type,
           (unsigned long long) ne0, (unsigned long long) ne1, bb);
    printf("raw  ");
    for (int i = 0; i < bb; ++i) printf("%02x", r0[i]);
    printf("\n");
    ref_dequant(z, r0, (int) type);
    printf("ref  ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", z[i]);
    printf("\n");

    // mega3's decoder is compiled in from the shared translation unit header form:
    // keep this probe independent by re-declaring only what it needs.
    extern void probe_mega_dequant(float *, const uint8_t *, int64_t, int);
    probe_mega_dequant(y, r0, k, (int) type);
    printf("mega ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", y[i]);
    printf("\n");
    double worst = 0;
    for (int i = 0; i < 256; ++i) {
        const double e = fabs((double) y[i] - z[i]);
        const double s = fabs((double) z[i]) > 1e-6 ? fabs((double) z[i]) : 1e-6;
        if (e / s > worst) worst = e / s;
    }
    printf("first block: max relative difference %.3g\n", worst);
    return 0;
}
