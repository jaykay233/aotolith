// Does the 3-bit packer produce the block layout the vendor kernel reads?
//
// Run against a synthetic 32-row x 256-k tile. The packer is a copy of
// repack_b_q3's arithmetic; the unpacker is written from the layout comment in
// mega3.cpp (int8 scales[16][32], uint16 hmask[16][32], uint8 qs[4][32][16],
// fp16 scales16[32]). If those two agree to a fraction of a step, the blob is
// what the kernel wants and a bad result has to come from somewhere else -- the
// A record, the kernel call's geometry, or the source dequantisation.
//
// The tile is scaled so per-row values span about [-8, 8], which is what the
// real weights look like after the Q4_0 decode.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

static const int Q3_K = 256;
static const size_t Q3_BLK = 32 * 16 + 32 * 32 + 32 * 64 + 32 * 2;   // 3648

// ---------------- packer, copied from repack_b_q3 ----------------
static void pack(uint8_t * blk, const float tile[32][Q3_K]) {
    int8_t *   scl = (int8_t *) blk;
    uint16_t * hms = (uint16_t *) (blk + 512);
    uint8_t *  qs  = blk + 512 + 1024;
    _Float16 * s16 = (_Float16 *) (blk + 512 + 1024 + 2048);
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

// ---------------- unpacker, from the layout comment ----------------
static float unpack1(const uint8_t * blk, int row, int kidx) {
    const int8_t *   scl = (const int8_t *) blk;
    const uint16_t * hms = (const uint16_t *) (blk + 512);
    const uint8_t *  qs  = blk + 512 + 1024;
    const _Float16 * s16 = (const _Float16 *) (blk + 512 + 1024 + 2048);
    const int kii = kidx / 16, bi = kidx % 16;
    const uint8_t byte = qs[(int64_t) (kii / 4) * 512 + row * 16 + bi];
    const int u = (byte >> ((kii % 4) * 2)) & 3;
    const int q3 = (hms[kii * 32 + row] & (1u << bi)) ? u : u - 4;
    return (float) q3 * (float) scl[kii * 32 + row] * (float) s16[row];
}

int main(void) {
    static uint8_t blk[Q3_BLK];
    static float tile[32][Q3_K];
    memset(blk, 0, sizeof blk);
    unsigned seed = 12345;
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < Q3_K; ++j) {
            seed = seed * 1103515245u + 12345u;
            tile[i][j] = ((float) ((seed >> 16) & 0x7fff) / 32767.0f * 2.0f - 1.0f) * 8.0f;
        }
    pack(blk, tile);
    double worst = 0, rms = 0;
    int worst_r = 0, worst_k = 0;
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < Q3_K; ++j) {
            const float got = unpack1(blk, i, j);
            const double e = fabs((double) got - tile[i][j]);
            rms += e * e;
            const double rel = e / (fabs((double) tile[i][j]) > 1e-3 ? fabs((double) tile[i][j]) : 1e-3);
            if (rel > worst) { worst = rel; worst_r = i; worst_k = j; }
        }
    rms = sqrt(rms / (32.0 * Q3_K));
    printf("32x256 tile, |x| <= 8: rms error %.4f, worst relative %.3f at row %d k %d\n",
           rms, worst, worst_r, worst_k);
    printf("worst case: want %.4f got %.4f\n", tile[worst_r][worst_k],
           unpack1(blk, worst_r, worst_k));
    // A step of a 3-bit level over this range is about 8/3 = 2.7, so an rms of a
    // few tenths is expected and anything near 1.0 or more means the byte layout
    // or the sign convention disagrees.
    return 0;
}
