#!/usr/bin/env python3
"""Float32 numpy reference for MiniCPM5-2B (Q4_0 GGUF), streaming one layer at a time.

Prints the top-5 logits of the last position and dumps per-layer attention and FFN
residuals to /tmp/rf_LL{a,f}.bin so the megakernel's dumps can be compared elementwise.
"""
import struct, sys, os
import numpy as np

GGUF = sys.argv[1] if len(sys.argv) > 1 else "mc-q4o.gguf"
IDS  = [int(x) for x in open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/p5.txt").read().split()]

FMT = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",8:"str",9:"arr",10:"Q",11:"q",12:"d"}
SZ  = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
TN  = {0:"F32",1:"F16",2:"Q4_0",8:"Q8_0",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K"}

f = open(GGUF, "rb")
def rd(n): return f.read(n)
def u32(): return struct.unpack("<I", rd(4))[0]
def u64(): return struct.unpack("<Q", rd(8))[0]
def st():
    n = u64(); return rd(n).decode("utf8", "replace")
def val(t):
    if t == 8: return st()
    if t == 9:
        et = u32(); n = u64(); return [val(et) for _ in range(n)]
    return struct.unpack("<" + FMT[t], rd(SZ[t]))[0]

magic = rd(4); ver = u32(); nt = u64(); nkv = u64()
for _ in range(nkv):
    st(); val(u32())
T = {}
for _ in range(nt):
    name = st(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
    T[name] = (dims, ty, off)
base = f.tell()
base += (-base) % 32          # tensor data is 32-byte aligned

def deq_row(name, r0, r1):
    """Dequantise output channels r0..r1-1 into an (n, in) float32 array."""
    dims, ty, off = T[name]
    n_in = dims[0]
    n = r1 - r0
    if ty == 0:
        f.seek(base + off + r0 * n_in * 4)
        return np.frombuffer(rd(n * n_in * 4), "<f4").reshape(n, n_in).astype(np.float32)
    if ty == 2:
        nblk = n_in // 32
        f.seek(base + off + r0 * nblk * 18)
        b = np.frombuffer(rd(n * nblk * 18), np.uint8).reshape(n, nblk, 18)
        d = b[:, :, :2].copy().view("<f2").astype(np.float32)          # (n, nblk, 1)
        qs = b[:, :, 2:]
        lo = (qs & 0xF).astype(np.float32) - 8.0
        hi = (qs >> 4).astype(np.float32) - 8.0
        v = np.concatenate([lo, hi], axis=2) * d
        return v.reshape(n, n_in)
    # K-quants, vectorised over output rows. Every offset below is a direct port
    # of the corresponding dequantize_row_* in ggml-quants.c: a transposed field
    # or a wrong shift produces plausible small numbers rather than an error, so
    # the check is the greedy token sequence, not eyeballing values.
    BLK = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210}
    if ty in BLK:
        nsup = n_in // 256
        f.seek(base + off + r0 * nsup * BLK[ty])
        b = np.frombuffer(rd(n * nsup * BLK[ty]), np.uint8).reshape(n, nsup, BLK[ty])
        out = np.empty((n, nsup, 256), np.float32)
        f16 = lambda sl: b[:, :, sl].copy().view("<f2").astype(np.float32)

        if ty == 10:                                    # Q2_K  d dmin sc[16] qs[64]
            d, dmin = f16(slice(0, 2)), f16(slice(2, 4))
            sc, qs = b[:, :, 4:20], b[:, :, 20:84]
            for g in range(16):                         # 16 groups of 16 quants
                s = sc[:, :, g]
                dl = d * (s & 0xF).astype(np.float32)[:, :, None]
                ml = dmin * (s >> 4).astype(np.float32)[:, :, None]
                blk, j, sub = g // 8, (g % 8) // 2, (g % 8) % 2
                q = qs[:, :, blk * 32 + sub * 16:blk * 32 + sub * 16 + 16]
                v = ((q >> (2 * j)) & 3).astype(np.float32)
                out[:, :, g * 16:(g + 1) * 16] = dl * v - ml

        elif ty == 11:                                  # Q3_K  hm[32] qs[64] sc[12] d
            hm, qs12 = b[:, :, 0:32], b[:, :, 32:96]
            d = f16(slice(108, 110))
            aux = np.zeros((n, nsup, 4), np.uint32)
            aux[:, :, :3] = b[:, :, 96:108].copy().view(np.uint32).reshape(n, nsup, 3)
            tmp, km1, km2 = aux[:, :, 2].copy(), 0x03030303, 0x0f0f0f0f
            aux[:, :, 2] = ((aux[:, :, 0] >> 4) & km2) | (((tmp >> 4) & km1) << 4)
            aux[:, :, 3] = ((aux[:, :, 1] >> 4) & km2) | (((tmp >> 6) & km1) << 4)
            aux[:, :, 0] = (aux[:, :, 0] & km2) | (((tmp >> 0) & km1) << 4)
            aux[:, :, 1] = (aux[:, :, 1] & km2) | (((tmp >> 2) & km1) << 4)
            scales = aux.view(np.uint8).reshape(n, nsup, 16).astype(np.float32) - 32.0
            for g in range(16):
                h, j, sub = g // 8, (g % 8) // 2, (g % 8) % 2
                dl = d * scales[:, :, g][:, :, None]
                q = qs12[:, :, h * 32 + sub * 16:h * 32 + sub * 16 + 16]
                v = ((q >> (2 * j)) & 3).astype(np.float32)
                m = (hm[:, :, sub * 16:sub * 16 + 16] >> (h * 4 + j)) & 1
                v = v - np.where(m != 0, 0.0, 4.0).astype(np.float32)
                out[:, :, g * 16:(g + 1) * 16] = dl * v

        elif ty in (12, 13):                            # Q4_K / Q5_K, same sc packing
            d, dmin = f16(slice(0, 2)), f16(slice(2, 4))
            sc12 = b[:, :, 4:16]
            qh = b[:, :, 16:48] if ty == 13 else None
            qs = b[:, :, 48:176] if ty == 13 else b[:, :, 16:144]
            def gsm(j):
                if j < 4:
                    return (sc12[:, :, j] & 63).astype(np.float32), (sc12[:, :, j + 4] & 63).astype(np.float32)
                dd = (sc12[:, :, j + 4] & 0xF) | ((sc12[:, :, j - 4] >> 6) << 4)
                mm = (sc12[:, :, j + 4] >> 4) | ((sc12[:, :, j] >> 6) << 4)
                return dd.astype(np.float32), mm.astype(np.float32)
            for j in range(4):                          # 4 chunks of 64 quants
                s1, m1 = gsm(2 * j)
                s2, m2 = gsm(2 * j + 1)
                d1, mm1 = d * s1[:, :, None], dmin * m1[:, :, None]
                d2, mm2 = d * s2[:, :, None], dmin * m2[:, :, None]
                q = qs[:, :, j * 32:j * 32 + 32]
                lo = (q & 0xF).astype(np.float32)
                hi = (q >> 4).astype(np.float32)
                if ty == 13:
                    hb = qh[:, :, 0:32]
                    lo = lo + (np.where((hb & (1 << (2 * j))) != 0, 16.0, 0.0)).astype(np.float32)
                    hi = hi + (np.where((hb & (2 << (2 * j))) != 0, 16.0, 0.0)).astype(np.float32)
                out[:, :, j * 64:j * 64 + 32] = d1 * lo - mm1
                out[:, :, j * 64 + 32:(j + 1) * 64] = d2 * hi - mm2

        else:                                           # Q6_K  ql[128] qh[64] sc[16] d
            ql, qhb = b[:, :, 0:128], b[:, :, 128:192]
            sc = b[:, :, 192:208].copy().view(np.int8).astype(np.float32)
            d = f16(slice(208, 210))
            for h in range(2):                          # two 128-quant blocks
                qb = ql[:, :, h * 64:h * 64 + 64]
                hb = qhb[:, :, h * 32:h * 32 + 32]
                sb = sc[:, :, h * 8:h * 8 + 8]
                for qi in range(4):
                    for half in range(2):
                        a = half * 16
                        if qi == 0:
                            qv = (qb[:, :, a:a + 16] & 0xF) | (((hb[:, :, a:a + 16] >> 0) & 3) << 4)
                        elif qi == 1:
                            qv = (qb[:, :, 32 + a:32 + a + 16] & 0xF) | (((hb[:, :, a:a + 16] >> 2) & 3) << 4)
                        elif qi == 2:
                            qv = (qb[:, :, a:a + 16] >> 4) | (((hb[:, :, a:a + 16] >> 4) & 3) << 4)
                        else:
                            qv = (qb[:, :, 32 + a:32 + a + 16] >> 4) | (((hb[:, :, a:a + 16] >> 6) & 3) << 4)
                        s = sb[:, :, half + 2 * qi][:, :, None]
                        o = h * 128 + qi * 32 + a
                        out[:, :, o:o + 16] = d * s * (qv.astype(np.float32) - 32.0)
        return out.reshape(n, n_in)
    raise SystemExit("unsupported type %s for %s" % (TN.get(ty, ty), name))

def f32w(name):
    dims, ty, off = T[name]
    f.seek(base + off)
    return np.frombuffer(rd(int(np.prod(dims)) * 4), "<f4").astype(np.float32)

E  = 2048
NF = 6144
NH, NKV, HD = 16, 2, 128
EPS = 1e-6
BASE_ROPE = 5000000.0

# embedding rows only
emb = np.zeros((len(IDS), E), np.float32)
for i, t in enumerate(IDS):
    emb[i] = deq_row("token_embd.weight", t, t + 1)[0]

N = len(IDS)
x = emb.copy()
pos = np.arange(N, dtype=np.float32)
inv_freq = 1.0 / (BASE_ROPE ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
ang = np.outer(pos, inv_freq)                    # (N, HD/2)
cos = np.cos(ang)[:, None, :]                    # (N, 1, HD/2)
sin = np.sin(ang)[:, None, :]

def rmsnorm(v, w):
    return v / np.sqrt((v * v).mean(-1, keepdims=True) + EPS) * w

def rope(v):
    # v: (N, NH, HD); llama rotate_half: pairs (i, i+HD/2)
    a = v[..., :HD // 2]
    b = v[..., HD // 2:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(np.float32)

mask = np.tril(np.ones((N, N), np.float32))
def attn(q, k, v):
    # q: (N, NH, HD); k,v: (N, NKV, HD) -> repeat kv
    kr = np.repeat(k, NH // NKV, axis=1)
    vr = np.repeat(v, NH // NKV, axis=1)
    out = np.zeros_like(q)
    for h in range(NH):
        s = q[:, h] @ kr[:, h].T / np.sqrt(HD)              # (N, N)
        s = np.where(mask > 0, s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(-1, keepdims=True)
        out[:, h] = p @ vr[:, h]
    return out.astype(np.float32)

for L in range(42):
    p = "blk.%d." % L
    xa = rmsnorm(x, f32w(p + "attn_norm.weight"))
    q = (xa @ deq_row(p + "attn_q.weight", 0, E).T).reshape(N, NH, HD)
    k = (xa @ deq_row(p + "attn_k.weight", 0, 2 * HD).T).reshape(N, NKV, HD)
    v = (xa @ deq_row(p + "attn_v.weight", 0, 2 * HD).T).reshape(N, NKV, HD)
    o = attn(rope(q), rope(k), v).reshape(N, E) @ deq_row(p + "attn_output.weight", 0, E).T
    x = (x + o).astype(np.float32)
    open("/tmp/rf_%02da.bin" % L, "wb").write(x.astype("<f4").tobytes())
    xn = rmsnorm(x, f32w(p + "ffn_norm.weight"))
    open("/tmp/rf_%02dn.bin" % L, "wb").write(xn.astype("<f4").tobytes())
    g = xn @ deq_row(p + "ffn_gate.weight", 0, NF).T
    u = xn @ deq_row(p + "ffn_up.weight", 0, NF).T
    h = (g / (1.0 + np.exp(-g))) * u
    d = h @ deq_row(p + "ffn_down.weight", 0, E).T
    x = (x + d).astype(np.float32)
    open("/tmp/rf_%02df.bin" % L, "wb").write(x.astype("<f4").tobytes())

xn = rmsnorm(x, f32w("output_norm.weight"))
last = xn[-1]
scores = np.empty(130560, np.float32)
CH = 8192
for lo in range(0, 130560, CH):
    W = deq_row("output.weight", lo, min(lo + CH, 130560))
    scores[lo:lo + len(W)] = W @ last
top = np.argsort(-scores)[:8]
if os.environ.get("RF_LOG"):
    open(os.environ["RF_LOG"], "wb").write(scores.astype("<f4").tobytes())
print("f32 reference top-8 (last position):")
for t in top:
    print("   %6d  %+.5f" % (t, scores[t]))
