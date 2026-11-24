#!/usr/bin/env python3
"""Check each K-quant dequantizer in ref_f32.py against an independent row.

Two files in the same directory are quantisations of the same checkpoint:
MiniCPM5-2B-Q4_0 for the body, MiniCPM5-2B-Q3KM for the same tensors in
Q3_K/Q4_K/Q5_K/Q6_K. Q4_0 dequantisation is already trusted (it reproduces
exact-fp32 tokens), so for any tensor present in both files the pairwise
correlation of the dequantised rows is the test: a transposed field, a wrong
shift or a bad superblock stride drops it towards zero, and a correct port of
ggml-quants.c leaves it near 0.99 (the two quantisations differ).

It also prints the ratio of the row norms, which catches a global scale error
(the f16 `d` read from the wrong offset) that correlation alone would miss.
"""
import sys, struct
import numpy as np

Q4 = "/home/bianbu/workspace/mega/mc-q4o.gguf"
QK = "/home/bianbu/models/MiniCPM5-2B-Q3KM.gguf"

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TN = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
BLK = {2: 18, 10: 84, 11: 110, 12: 144, 13: 176, 14: 210}


class G:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        def rd(n): return f.read(n)
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

    def raw(self, name, r0, nrow):
        dims, ty, off = self.T[name]
        k = dims[0]
        if ty == 2:
            rb = (k // 32) * 18
        elif ty in BLK:
            rb = (k // 256) * BLK[ty]
        else:
            raise SystemExit("row bytes unknown for type %d" % ty)
        self.f.seek(self.base + off + r0 * rb)
        return np.frombuffer(self.f.read(nrow * rb), np.uint8).reshape(nrow, rb), k, ty


def deq_q4(b):
    n, rb = b.shape
    nblk = rb // 18
    b = b.reshape(n, nblk, 18)
    d = b[:, :, 0:2].copy().view("<f2").astype(np.float32)
    qs = b[:, :, 2:]
    lo = (qs & 0xF).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    v = np.concatenate([lo, hi], axis=2) * d
    return v.reshape(n, nblk * 32)


def deq_k(b, ty, k):
    """Scalar-ish port, one 256-block at a time, written straight from the C."""
    nsup = k // 256
    nb = b.shape[1] // nsup
    b = b.reshape(b.shape[0], nsup, nb)
    out = np.zeros((b.shape[0], k), np.float32)
    for s in range(nsup):
        blk = b[:, s, :]
        y = out[:, s * 256:(s + 1) * 256]
        if ty == 10:
            d = blk[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1)
            dm = blk[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1)
            sc, qs = blk[:, 4:20], blk[:, 20:84]
            for g in range(16):
                j, sub = (g % 8) // 2, (g % 8) % 2
                blkk = g // 8
                dl = d * (sc[:, g] & 0xF).astype(np.float32)
                ml = dm * (sc[:, g] >> 4).astype(np.float32)
                q = qs[:, blkk * 32 + sub * 16:blkk * 32 + sub * 16 + 16]
                y[:, g * 16:(g + 1) * 16] = dl[:, None] * ((q >> (2 * j)) & 3) - ml[:, None]
        elif ty == 11:
            hm, qs, sc12 = blk[:, 0:32], blk[:, 32:96], blk[:, 96:108]
            d = blk[:, 108:110].copy().view("<f2").astype(np.float32).reshape(-1)
            aux = np.zeros((blk.shape[0], 4), np.uint32)
            aux[:, :3] = sc12.copy().view(np.uint32)
            tmp = aux[:, 2].copy()
            km1, km2 = 0x03030303, 0x0f0f0f0f
            aux[:, 2] = ((aux[:, 0] >> 4) & km2) | (((tmp >> 4) & km1) << 4)
            aux[:, 3] = ((aux[:, 1] >> 4) & km2) | (((tmp >> 6) & km1) << 4)
            aux[:, 0] = (aux[:, 0] & km2) | (((tmp >> 0) & km1) << 4)
            aux[:, 1] = (aux[:, 1] & km2) | (((tmp >> 2) & km1) << 4)
            scales = aux.view(np.uint8).astype(np.float32) - 32.0
            for g in range(16):
                h, j, sub = g // 8, (g % 8) // 2, (g % 8) % 2
                dl = d * scales[:, g]
                q = qs[:, h * 32 + sub * 16:h * 32 + sub * 16 + 16]
                v = ((q >> (2 * j)) & 3).astype(np.float32)
                m = (hm[:, sub * 16:sub * 16 + 16] >> (h * 4 + j)) & 1
                v = v - np.where(m != 0, 0.0, 4.0)
                y[:, g * 16:(g + 1) * 16] = dl[:, None] * v
        elif ty in (12, 13):
            d = blk[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1)
            dm = blk[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1)
            sc12 = blk[:, 4:16]
            qh = blk[:, 16:48] if ty == 13 else None
            qs = blk[:, 48:176] if ty == 13 else blk[:, 16:144]
            def gsm(j):
                if j < 4:
                    return (sc12[:, j] & 63).astype(np.float32), (sc12[:, j + 4] & 63).astype(np.float32)
                dd = ((sc12[:, j + 4] & 0xF) | ((sc12[:, j - 4] >> 6) << 4)).astype(np.float32)
                mm = ((sc12[:, j + 4] >> 4) | ((sc12[:, j] >> 6) << 4)).astype(np.float32)
                return dd, mm
            for j in range(4):
                s1, m1 = gsm(2 * j)
                s2, m2 = gsm(2 * j + 1)
                q = qs[:, j * 32:j * 32 + 32]
                lo = (q & 0xF).astype(np.float32)
                hi = (q >> 4).astype(np.float32)
                if ty == 13:
                    hb = qh[:, 0:32]
                    lo = lo + np.where((hb & (1 << (2 * j))) != 0, 16.0, 0.0)
                    hi = hi + np.where((hb & (2 << (2 * j))) != 0, 16.0, 0.0)
                y[:, j * 64:j * 64 + 32] = (d * s1)[:, None] * lo - (dm * m1)[:, None]
                y[:, j * 64 + 32:(j + 1) * 64] = (d * s2)[:, None] * hi - (dm * m2)[:, None]
        else:
            ql, qhb = blk[:, 0:128], blk[:, 128:192]
            sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
            d = blk[:, 208:210].copy().view("<f2").astype(np.float32).reshape(-1)
            for h in range(2):
                qb = ql[:, h * 64:h * 64 + 64]
                hb = qhb[:, h * 32:h * 32 + 32]
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
                        s = sb[:, half + 2 * qi]
                        o = qi * 32 + a
                        y[:, h * 128 + o:h * 128 + o + 16] = (d * s)[:, None] * (qv.astype(np.float32) - 32.0)
    return out


# This copy is deliberately independent of ref_f32.py's vectorised version, so
# that a shared indexing mistake cannot make both agree. It is the same scalar
# control; see the note at the end of FINDINGS section 9.
sys.path.insert(0, "/home/bianbu/workspace/mega")
import importlib.util
spec = importlib.util.spec_from_file_location("rf", "/home/bianbu/workspace/mega/ref_f32.py")

a, b = G(Q4), G(QK)
names = ["blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
         "blk.0.ffn_down.weight", "blk.5.ffn_down.weight", "token_embd.weight",
         "output.weight"]
print("%-24s %-7s %-8s %-9s %-9s" % ("tensor", "type", "corr", "norm ratio", "verdict"))
for nm in names:
    ty = b.T[nm][1]
    rawq, k, _ = b.raw(nm, 0, 8)
    vq = deq_k(rawq, ty, k)
    if nm not in a.T:
        print("%-24s %-7s (no Q4_0 twin)" % (nm, TN.get(ty, ty)))
        continue
    raw4, k4, _ = a.raw(nm, 0, 8)
    v4 = deq_q4(raw4)
    if v4.shape != vq.shape:
        print("%-24s %-7s shape %s vs %s" % (nm, TN.get(ty, ty), v4.shape, vq.shape))
        continue
    c = float(np.corrcoef(v4.ravel(), vq.ravel())[0, 1])
    nr = float(np.linalg.norm(vq) / np.linalg.norm(v4))
    print("%-24s %-7s %8.5f %9.4f  %s" % (nm, TN.get(ty, ty), c, nr,
                                          "ok" if c > 0.97 and abs(nr - 1) < 0.1 else "<-- BROKEN"))
