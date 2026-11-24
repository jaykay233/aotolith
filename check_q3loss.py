#!/usr/bin/env python3
"""How much does the in-tree 3-bit packer itself lose, tensor by tensor?

repack_b_q3 does not store Q3_K; it re-quantises the dequantised weights into its
own uniform 3-bit layout:

    per (row, 16 k-elements): step = max(max_pos/3, max_neg/4)
    per row:                  fs   = max(step)/127, isc = round(step/fs) in [1,127]
    per element:              q3   = clamp(round(v/(isc*fs)), -4, 3)

so the effective step is itself quantised to 127 levels, which is a second loss
on top of the 3 bits. Sign is a separate high-bit plane, giving -4..+3.

This measures that quantisation in the weight domain, comparing against the same
row dequantised exactly from the file. A row's cosine here is what the GEMM will
see; the end-to-end logit cosine is lower because the error compounds over 42
layers, but a row that is already 0.9 here cannot be rescued downstream.

    check_q3loss.py <gguf> [rows]

Printed for each tensor: cosine of the requantised row vs the exact row, and the
same for a hypothetical 4-bit version of the same scheme, to show whether the
cliff is the bit count or the scheme.
"""
import sys, struct
import numpy as np

GGUF = sys.argv[1] if len(sys.argv) > 1 else "/home/bianbu/models/MiniCPM5-2B-Q3KM.gguf"
NROW = int(sys.argv[2]) if len(sys.argv) > 2 else 16

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TN = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
BLK = {2: 18, 10: 84, 11: 110, 12: 144, 13: 176, 14: 210}


class G:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        rd = f.read
        u32 = lambda: struct.unpack("<I", rd(4))[0]
        u64 = lambda: struct.unpack("<Q", rd(8))[0]
        def st():
            n = u64(); return rd(n).decode("utf8", "replace")
        def val(t):
            if t == 8: return st()
            if t == 9:
                et = u32(); n = u64()
                for _ in range(n): val(et)
                return None
            return struct.unpack("<" + FMT[t], rd(SZ[t]))[0]
        rd(4); u32(); nt = u64(); nkv = u64()
        for _ in range(nkv): st(); val(u32())
        self.T = {}
        for _ in range(nt):
            nm = st(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
            self.T[nm] = (dims, ty, off)
        base = f.tell()
        self.base = base + (-base) % 32

    def rows(self, name, nrow):
        dims, ty, off = self.T[name]
        k = dims[0]
        rb = (k // 32) * 18 if ty == 2 else (k // 256) * BLK[ty]
        self.f.seek(self.base + off)
        return np.frombuffer(self.f.read(nrow * rb), np.uint8).reshape(nrow, rb), k, ty


def deq(b, ty, k):
    """Exact dequantisation, one 256-block at a time (independent of ref_f32)."""
    n = b.shape[0]
    if ty == 2:
        nb = k // 32
        r = b.reshape(n, nb, 18)
        d = r[:, :, 0:2].copy().view("<f2").astype(np.float32)
        qs = r[:, :, 2:]
        lo = (qs & 0xF).astype(np.float32) - 8.0
        hi = (qs >> 4).astype(np.float32) - 8.0
        return (np.concatenate([lo, hi], axis=2) * d).reshape(n, k)
    nsup = k // 256
    nb = b.shape[1] // nsup
    r = b.reshape(n, nsup, nb)
    out = np.zeros((n, k), np.float32)
    for s in range(nsup):
        blk = r[:, s, :]
        y = out[:, s * 256:(s + 1) * 256]
        f2 = lambda sl: blk[:, sl].copy().view("<f2").astype(np.float32).reshape(-1)
        if ty == 10:
            d, dm = f2(slice(0, 2)), f2(slice(2, 4))
            sc, qs = blk[:, 4:20], blk[:, 20:84]
            for g in range(16):
                j, sub, bb = (g % 8) // 2, (g % 8) % 2, g // 8
                q = qs[:, bb * 32 + sub * 16:bb * 32 + sub * 16 + 16]
                y[:, g * 16:(g + 1) * 16] = (d * (sc[:, g] & 0xF))[:, None] * ((q >> (2 * j)) & 3) \
                                           - (dm * (sc[:, g] >> 4))[:, None]
        elif ty == 11:
            hm, qs, sc12 = blk[:, 0:32], blk[:, 32:96], blk[:, 96:108]
            d = f2(slice(108, 110))
            aux = np.zeros((n, 4), np.uint32)
            aux[:, :3] = sc12.copy().view(np.uint32)
            t = aux[:, 2].copy()
            km1, km2 = 0x03030303, 0x0f0f0f0f
            aux[:, 2] = ((aux[:, 0] >> 4) & km2) | (((t >> 4) & km1) << 4)
            aux[:, 3] = ((aux[:, 1] >> 4) & km2) | (((t >> 6) & km1) << 4)
            aux[:, 0] = (aux[:, 0] & km2) | (((t >> 0) & km1) << 4)
            aux[:, 1] = (aux[:, 1] & km2) | (((t >> 2) & km1) << 4)
            sc = aux.view(np.uint8).astype(np.float32) - 32.0
            for g in range(16):
                h, j, sub = g // 8, (g % 8) // 2, (g % 8) % 2
                q = qs[:, h * 32 + sub * 16:h * 32 + sub * 16 + 16]
                v = ((q >> (2 * j)) & 3).astype(np.float32)
                m = (hm[:, sub * 16:sub * 16 + 16] >> (h * 4 + j)) & 1
                y[:, g * 16:(g + 1) * 16] = (d * sc[:, g])[:, None] * (v - np.where(m != 0, 0.0, 4.0))
        elif ty in (12, 13):
            d, dm = f2(slice(0, 2)), f2(slice(2, 4))
            sc12 = blk[:, 4:16]
            qh = blk[:, 16:48] if ty == 13 else None
            qs = blk[:, 48:176] if ty == 13 else blk[:, 16:144]
            def gsm(j):
                if j < 4:
                    return (sc12[:, j] & 63).astype(np.float32), (sc12[:, j + 4] & 63).astype(np.float32)
                return (((sc12[:, j + 4] & 0xF) | ((sc12[:, j - 4] >> 6) << 4)).astype(np.float32),
                        ((sc12[:, j + 4] >> 4) | ((sc12[:, j] >> 6) << 4)).astype(np.float32))
            for j in range(4):
                s1, m1 = gsm(2 * j)
                s2, m2 = gsm(2 * j + 1)
                q = qs[:, j * 32:j * 32 + 32]
                lo = (q & 0xF).astype(np.float32)
                hi = (q >> 4).astype(np.float32)
                if ty == 13:
                    lo = lo + np.where((qh[:, 0:32] & (1 << (2 * j))) != 0, 16.0, 0.0)
                    hi = hi + np.where((qh[:, 0:32] & (2 << (2 * j))) != 0, 16.0, 0.0)
                y[:, j * 64:j * 64 + 32] = (d * s1)[:, None] * lo - (dm * m1)[:, None]
                y[:, j * 64 + 32:(j + 1) * 64] = (d * s2)[:, None] * hi - (dm * m2)[:, None]
        else:
            ql, qhb = blk[:, 0:128], blk[:, 128:192]
            sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
            d = f2(slice(208, 210))
            for h in range(2):
                qb, hb = ql[:, h * 64:h * 64 + 64], qhb[:, h * 32:h * 32 + 32]
                sb = sc[:, h * 8:h * 8 + 8]
                for qi in range(4):
                    for half in range(2):
                        a = half * 16
                        if qi == 0:
                            qv = (qb[:, a:a + 16] & 0xF) | (((hb[:, a:a + 16] >> 0) & 3) << 4)
                        elif qi == 1:
                            qv = (qb[:, 32 + a:32 + a + 16] & 0xF) | (((hb[:, a:a + 16] >> 2) & 3) << 4)
                        elif qi == 2:
                            qv = (qb[:, a:a + 16] >> 4) | (((hb[:, a:a + 16] >> 4) & 3) << 4)
                        else:
                            qv = (qb[:, 32 + a:32 + a + 16] >> 4) | (((hb[:, a:a + 16] >> 6) & 3) << 4)
                        o = qi * 32 + a
                        y[:, h * 128 + o:h * 128 + o + 16] = (d * sb[:, half + 2 * qi])[:, None] * (qv.astype(np.float32) - 32.0)
    return out


def repack_q3(W):
    """The packer's own arithmetic, applied in the weight domain."""
    n, k = W.shape
    V = W.reshape(n, k // 16, 16)
    mx = np.maximum(V.max(-1), 0.0) / 3.0
    mn = np.maximum(-V.min(-1), 0.0) / 4.0
    step = np.maximum(mx, mn)                       # per (row, k16)
    smax = step.max(-1, keepdims=True)
    fs = np.where(smax > 0, smax / 127.0, 1.0)      # fp16 in the real packer
    isc = np.clip(np.rint(step / fs), 1, 127)
    eff = isc * fs
    q = np.clip(np.rint(V / np.where(eff > 0, eff, 1.0)[:, :, None]), -4, 3)
    return (q * eff[:, :, None]).reshape(n, k)


def repack_q4(W):
    """Same scheme at 4 bits (-8..7), for the cliff comparison."""
    n, k = W.shape
    V = W.reshape(n, k // 16, 16)
    step = np.maximum(np.maximum(V.max(-1), 0.0) / 7.0, np.maximum(-V.min(-1), 0.0) / 8.0)
    smax = step.max(-1, keepdims=True)
    fs = np.where(smax > 0, smax / 127.0, 1.0)
    eff = np.clip(np.rint(step / fs), 1, 127) * fs
    q = np.clip(np.rint(V / np.where(eff > 0, eff, 1.0)[:, :, None]), -8, 7)
    return (q * eff[:, :, None]).reshape(n, k)


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


g = G(GGUF)
names = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
         "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.attn_v.weight",
         "blk.0.ffn_down.weight", "blk.0.attn_output.weight", "output.weight"]
print("%-24s %-6s %8s %8s %9s" % ("tensor", "type", "q3 cos", "q4 cos", "q3 rms"))
for nm in names:
    if nm not in g.T:
        continue
    b, k, ty = g.rows(nm, NROW)
    W = deq(b, ty, k)
    W3 = repack_q3(W)
    W4 = repack_q4(W)
    print("%-24s %-6s %8.4f %8.4f %9.4f" % (nm, TN.get(ty, ty), cos(W, W3), cos(W, W4),
                                             float(np.sqrt(((W - W3) ** 2).mean()))))
